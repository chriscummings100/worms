"""Check that checkpoints resume real GPU worlds without changing their trajectory."""

import json

import numpy as np
import pytest
import slangpy as spy

from tests.gpu_helpers import assert_sim_buffers_equal, legacy_neural_buffers
from worms.benchmark import Benchmark
from worms.sim import SHADER_DIR, Sim
from worms.snapshots import (
    SIM_BUFFERS,
    SIM_SETTINGS,
    PeriodicSnapshots,
    load_snapshot,
    save_snapshot,
)

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def device():
    adapter = next(
        a for a in spy.Device.enumerate_adapters(spy.DeviceType.d3d12) if "NVIDIA" in a.name
    )
    device = spy.Device(
        type=spy.DeviceType.d3d12,
        adapter_luid=adapter.luid,
        enable_debug_layers=True,
        compiler_options={"include_paths": [SHADER_DIR]},
    )
    yield device
    device.wait()
    device.close()


def make_sim(device, **settings):
    defaults = {
        "num_worlds": 2,
        "creatures_per_world": 2,
        "creature_capacity": 5,
        "food_capacity": 32,
        "initial_food": 6,
        "target_food": 16,
        "food_per_second": 20,
        "food_age_start": 35,
        "food_age_fade": 7,
        "seed": 73,
    }
    return Sim(device, **(defaults | settings))


def advance(sim, count):
    encoder = sim.device.create_command_encoder()
    for _ in range(count):
        sim.update(encoder)
    sim.device.submit_command_buffer(encoder.finish())


def assert_same_state(first, second, *, continued=False):
    assert first.tick == second.tick
    assert first._food_energy_dirty == second._food_energy_dirty
    for setting in SIM_SETTINGS:
        assert getattr(first, setting) == getattr(second, setting), setting
    actual = {field: getattr(first, field).to_numpy() for field in SIM_BUFFERS}
    expected = {field: getattr(second, field).to_numpy() for field in SIM_BUFFERS}
    if continued:
        assert_sim_buffers_equal(actual, expected)
    else:
        for field in SIM_BUFFERS:
            np.testing.assert_array_equal(actual[field], expected[field], err_msg=field)


def test_restore_preserves_history_pending_settings_and_future_births_deaths(device, tmp_path):
    original = make_sim(device)
    advance(original, 7)
    layout = original._initialize.reflection.g_creatures.type_layout.element_type_layout
    cursor = spy.BufferCursor(layout, original.creatures)
    for index in (0, original.creature_capacity):
        cursor[index]["energy"] = cursor[index]["split_energy"].read() * 2
    cursor.apply()
    original.food_energy = 7.25  # The next step must still apply this pending UI setting.
    original.food_per_second = 31.5
    original.input_mutation_probability = 0.23
    original.policy_mutation_probability = 0.37
    path = save_snapshot(tmp_path / "resume.npz", original)
    restored = load_snapshot(path, device).sim
    assert_same_state(original, restored)
    assert restored.input_mutation_probability == 0.23
    assert restored.policy_mutation_probability == 0.37
    assert restored.inspect_vision(0, 0, age=5) == original.inspect_vision(0, 0, age=5)

    advance(original, 25)
    advance(restored, 25)
    assert_same_state(original, restored, continued=True)
    assert original.inspect_world(0)["births"] > 0
    advance(original, 55)
    advance(restored, 55)
    assert_same_state(original, restored, continued=True)
    assert original.inspect_world(0)["deaths"] > 0


def test_checkpoint_includes_benchmark_worlds_frozen_arrays_and_graph(device, tmp_path):
    live = make_sim(device)
    trial = make_sim(device, mutation_enabled=False, seed=91)
    advance(live, 5)
    advance(trial, 3)
    reference = trial.brain_weights.to_numpy().copy()
    metadata = {"benchmark": {"trial": 4, "history": [[10000, 0.61], [20000, 0.64]]}}
    path = save_snapshot(
        tmp_path / "session.npz",
        live,
        extra_sims={"benchmark": trial},
        metadata=metadata,
        arrays={"reference_weights": reference},
    )
    restored = load_snapshot(path, device)
    assert_same_state(live, restored.sim)
    assert_same_state(trial, restored.extra_sims["benchmark"])
    assert restored.metadata == metadata
    np.testing.assert_array_equal(restored.arrays["reference_weights"], reference)
    advance(trial, 12)
    advance(restored.extra_sims["benchmark"], 12)
    assert_same_state(trial, restored.extra_sims["benchmark"], continued=True)


def test_version_four_migrates_live_trial_and_frozen_reference_without_recapture(device, tmp_path):
    live = make_sim(device)
    benchmark = Benchmark(live, warmup_ticks=0, sample_size=2, trial_ticks=20)
    metadata, arrays = benchmark.snapshot_state()
    path = save_snapshot(
        tmp_path / "legacy-session.npz",
        live,
        extra_sims={"benchmark": benchmark.sim},
        metadata={"benchmark": metadata},
        arrays=arrays,
    )
    with np.load(path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    manifest = json.loads(str(payload["manifest"]))
    manifest["version"] = 4
    legacy_neural_buffers(payload, manifest)
    keys = [entry["buffers"]["brain_weights"] for entry in manifest["sims"]]
    keys.append(manifest["arrays"]["reference_weights"])
    for index, key in enumerate(keys):
        # Distinct FP32-only values prove every cohort converts its own bytes.
        payload[key].view(np.float32)[0] = 0.1234567 + 0.1 * index
    expected = [payload[key].view(np.float32).astype(np.float16) for key in keys]
    payload["manifest"] = np.array(json.dumps(manifest))
    np.savez_compressed(path, **payload)
    source_bytes = path.read_bytes()

    restored = load_snapshot(path, device)
    resumed = Benchmark.from_snapshot(
        restored.sim,
        restored.extra_sims["benchmark"],
        restored.metadata["benchmark"],
        restored.arrays,
    )
    assert resumed.reference_tick == benchmark.reference_tick
    assert resumed.trial == benchmark.trial and resumed.ready
    for buffer, values in zip(
        (restored.sim.brain_weights, resumed.sim.brain_weights, resumed.reference_weights), expected
    ):
        np.testing.assert_array_equal(buffer.to_numpy().view(np.float16).reshape(-1), values)
    assert restored.metadata == {"benchmark": metadata}
    assert path.read_bytes() == source_bytes


def test_periodic_snapshots_keep_latest_files_and_leave_manual_checkpoint(device, tmp_path):
    sim = make_sim(device, num_worlds=1, creatures_per_world=1, creature_capacity=1)
    manual = save_snapshot(tmp_path / "manual.npz", sim)
    store = PeriodicSnapshots(tmp_path, interval=2, keep=2)
    paths = []
    assert not store.due(sim.tick)
    for _ in range(3):
        advance(sim, 2)
        assert store.due(sim.tick)
        paths.append(store.save(sim))
        assert not store.due(sim.tick)
    assert manual.exists()
    assert not paths[0].exists()
    assert paths[1].exists() and paths[2].exists()
    assert store.latest == paths[-1]
    assert load_snapshot(store.latest, device).sim.tick == 6
    store.reset(1)
    assert not store.due(1)
    assert store.due(2)


def test_restore_rejects_incompatible_gpu_layout(device, tmp_path):
    path = save_snapshot(tmp_path / "wrong-layout.npz", make_sim(device))
    with np.load(path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    manifest = json.loads(str(payload["manifest"]))
    key = manifest["sims"][0]["buffers"]["creatures"]
    payload[key] = payload[key][:-1]
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="layout no longer matches: creatures"):
        load_snapshot(path, device)


def test_legacy_snapshot_restores_existing_state_with_zero_intake_counters(device, tmp_path):
    original = make_sim(device)
    advance(original, 3)
    path = save_snapshot(tmp_path / "legacy.npz", original)
    with np.load(path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    manifest = json.loads(str(payload["manifest"]))
    manifest["version"] = 1
    legacy_neural_buffers(payload, manifest)
    for entry in manifest["sims"]:
        del payload[entry["buffers"].pop("food_energy_totals")]
    payload["manifest"] = np.array(json.dumps(manifest))
    np.savez_compressed(path, **payload)

    restored = load_snapshot(path, device).sim

    assert restored.tick == original.tick
    np.testing.assert_array_equal(restored.food_energy_totals.to_numpy(), 0)
    for field in SIM_BUFFERS:
        if field != "food_energy_totals":
            np.testing.assert_array_equal(
                getattr(original, field).to_numpy(), getattr(restored, field).to_numpy()
            )


@pytest.mark.parametrize("version", [2, 3, 4])
def test_previous_snapshot_versions_retain_live_state_and_intake_counters(
    device, tmp_path, version
):
    original = make_sim(device)
    advance(original, 3)
    original.food_energy_totals.copy_from_numpy(
        np.arange(original.num_worlds * 3, dtype=np.uint64) + 17
    )
    path = save_snapshot(tmp_path / f"version-{version}.npz", original)
    with np.load(path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    manifest = json.loads(str(payload["manifest"]))
    manifest["version"] = version
    legacy_neural_buffers(payload, manifest)
    payload["manifest"] = np.array(json.dumps(manifest))
    np.savez_compressed(path, **payload)
    restored = load_snapshot(path, device).sim
    assert_same_state(original, restored)
    advance(original, 2)
    advance(restored, 2)
    assert_same_state(original, restored, continued=True)


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_snapshot_without_mutation_settings_keeps_legacy_probabilities(device, tmp_path, version):
    original = make_sim(
        device,
        food_capacity=0,
        initial_food=0,
        target_food=0,
        food_per_second=0,
        input_mutation_probability=0.01,
        policy_mutation_probability=0.01,
    )
    advance(original, 3)
    layout = original._initialize.reflection.g_creatures.type_layout.element_type_layout
    creatures = spy.BufferCursor(layout, original.creatures)
    for index in (0, original.creature_capacity):
        creatures[index]["energy"] = creatures[index]["split_energy"].read() * 2
    creatures.apply()
    path = save_snapshot(tmp_path / "legacy-policy.npz", original)
    with np.load(path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    manifest = json.loads(str(payload["manifest"]))
    manifest["version"] = version
    legacy_neural_buffers(payload, manifest)
    for entry in manifest["sims"]:
        del entry["settings"]["input_mutation_probability"]
        del entry["settings"]["policy_mutation_probability"]
        if version == 1:
            del payload[entry["buffers"].pop("food_energy_totals")]
    payload["manifest"] = np.array(json.dumps(manifest))
    np.savez_compressed(path, **payload)

    restored = load_snapshot(path, device).sim

    assert restored.input_mutation_probability == restored.policy_mutation_probability == 0.01
    assert_same_state(original, restored)
    advance(original, 2)
    advance(restored, 2)
    assert original.inspect_world(0)["births"] > 0
    assert_same_state(original, restored, continued=True)


def test_version_three_legacy_ids_remain_unique_when_restored_parents_split_into_two(
    device, tmp_path
):
    original = make_sim(
        device,
        num_worlds=2,
        creatures_per_world=2,
        creature_capacity=4,
        food_capacity=0,
        initial_food=0,
        target_food=0,
        food_per_second=0,
        mutation_enabled=False,
    )
    original.tick = 10
    split_counts = (7, 19)
    parent_generations = (6, 11)
    total_slots = original.num_worlds * original.creature_capacity
    layout = original._initialize.reflection.g_creatures.type_layout.element_type_layout
    creatures = spy.BufferCursor(layout, original.creatures)
    world_layout = original._initialize.reflection.g_worlds.type_layout.element_type_layout
    worlds = spy.BufferCursor(world_layout, original.worlds)
    legacy_guids = {}
    for world, completed_splits in enumerate(split_counts):
        worlds[world]["births"] = completed_splits
        for local in (0, 1):
            index = world * original.creature_capacity + local
            # Version 3 issued one identity per split, with the zero-based split
            # serial and world index interleaved. These are genuine old-scheme
            # IDs, rather than version-4 offspring saved with an older label.
            serial = completed_splits - 1 - local
            guid = total_slots + serial * original.num_worlds + world + 1
            legacy_guids[index] = guid
            creatures[index]["guid"] = guid
            creatures[index]["generation"] = parent_generations[world] - local
            creatures[index]["birth_tick"] = 4
            creatures[index]["age_start_tick"] = 4
            creatures[index]["max_speed"] = 0.0
            if local == 0:
                creatures[index]["energy"] = creatures[index]["split_energy"].read() * 2
    creatures.apply()
    worlds.apply()
    path = save_snapshot(tmp_path / "legacy-identities.npz", original)
    with np.load(path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    manifest = json.loads(str(payload["manifest"]))
    manifest["version"] = 3
    legacy_neural_buffers(payload, manifest)
    payload["manifest"] = np.array(json.dumps(manifest))
    np.savez_compressed(path, **payload)

    restored = load_snapshot(path, device).sim
    assert_same_state(original, restored)
    for world in range(2):
        for local in (0, 1):
            creature = restored.inspect(world, local)
            assert creature["guid"] == legacy_guids[world * restored.creature_capacity + local]
            assert creature["generation"] == parent_generations[world] - local

    advance(restored, 1)
    living_ids = []
    newborn_ids = []
    for world, completed_splits in enumerate(split_counts):
        assert restored.inspect_world(world)["births"] == completed_splits + 1
        original_parent = legacy_guids[world * restored.creature_capacity]
        for local in (0, 2):
            newborn = restored.inspect(world, local)
            assert newborn["alive"]
            assert newborn["generation"] == parent_generations[world] + 1
            assert newborn["parent_guid"] == original_parent
            assert newborn["birth_tick"] == newborn["age_start_tick"] == 11
            newborn_ids.append(newborn["guid"])
        survivor = restored.inspect(world, 1)
        assert survivor["guid"] == legacy_guids[world * restored.creature_capacity + 1]
        assert survivor["generation"] == parent_generations[world] - 1
        living_ids.extend(restored.inspect(world, local)["guid"] for local in (0, 1, 2))
    assert len(set(newborn_ids)) == 4
    assert set(newborn_ids).isdisjoint(legacy_guids.values())
    assert len(set(living_ids)) == 6
