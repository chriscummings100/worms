"""Check that checkpoints resume real GPU worlds without changing their trajectory."""

import json

import numpy as np
import pytest
import slangpy as spy

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


def assert_same_state(first, second):
    assert first.tick == second.tick
    assert first._food_energy_dirty == second._food_energy_dirty
    for setting in SIM_SETTINGS:
        assert getattr(first, setting) == getattr(second, setting), setting
    for field in SIM_BUFFERS:
        np.testing.assert_array_equal(
            getattr(first, field).to_numpy(), getattr(second, field).to_numpy(), err_msg=field
        )


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
    path = save_snapshot(tmp_path / "resume.npz", original)
    restored = load_snapshot(path, device).sim
    assert_same_state(original, restored)
    assert restored.inspect_vision(0, 0, age=5) == original.inspect_vision(0, 0, age=5)

    advance(original, 25)
    advance(restored, 25)
    assert_same_state(original, restored)
    assert original.inspect_world(0)["births"] > 0
    advance(original, 55)
    advance(restored, 55)
    assert_same_state(original, restored)
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
    assert_same_state(trial, restored.extra_sims["benchmark"])


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
