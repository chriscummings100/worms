"""Real GPU checks for fair cohorts, inherited scores and trial isolation."""

import numpy as np
import pytest
import slangpy as spy

from tests.gpu_helpers import assert_sim_buffers_equal, fixed_brains
from tests.test_contact_gpu import write_food
from worms.benchmark import PROTOCOL_VERSION, REFERENCE_WARMUP_TICKS, Benchmark
from worms.sim import BRAIN_PARAMETERS, ENERGY_PER_UNIT, INITIAL_ENERGY, SHADER_DIR, Sim
from worms.snapshots import SIM_BUFFERS, load_snapshot, save_snapshot

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


def make_sim(device, *, creatures=3, capacity=8, food=8, worlds=1):
    return Sim(
        device,
        num_worlds=worlds,
        creatures_per_world=creatures,
        creature_capacity=capacity,
        food_capacity=food,
        initial_food=food,
        target_food=food,
        food_per_second=0,
        seed=37,
    )


def creatures(sim):
    layout = sim._initialize.reflection.g_creatures.type_layout.element_type_layout
    return spy.BufferCursor(layout, sim.creatures)


def weights(sim):
    return sim.brain_weights.to_numpy().view(np.float16).reshape(-1, BRAIN_PARAMETERS)


def advance(benchmark, ticks):
    encoder = benchmark.sim.device.create_command_encoder()
    for _ in range(ticks):
        benchmark.update(encoder)
    benchmark.sim.device.submit_command_buffer(encoder.finish())


def feed(sim, meals):
    """Place food on chosen (world, local creature, energy quanta) contacts."""
    positions = [(490, 490)] * (sim.num_worlds * sim.food_capacity)
    energies = [0] * len(positions)
    offsets = [0] * sim.num_worlds
    for world, creature, energy in meals:
        index = world * sim.food_capacity + offsets[world]
        offsets[world] += 1
        positions[index] = sim.inspect(world, creature)["position"]
        energies[index] = energy
    write_food(sim, positions, energies)


def test_default_trials_have_fixed_sparse_food_and_long_enough_horizon(device):
    live = make_sim(device, food=200)
    live.food_per_second = 20
    live.food_energy = 9
    benchmark = Benchmark(live, warmup_ticks=0)
    assert benchmark.trial_ticks == 100_000
    assert benchmark.sim.initial_food == 150
    assert benchmark.sim.food_per_second == 10
    assert benchmark.sim.food_energy == 5
    assert [benchmark.sim.inspect_world(world)["num_food"] for world in range(2)] == [150, 150]


@pytest.mark.parametrize("overshoot", [0, 37])
def test_default_reference_waits_for_threshold_then_captures_survivors_once(device, overshoot):
    live = make_sim(device, creatures=4, capacity=8, food=0, worlds=2)
    benchmark = Benchmark(live, sample_size=2, trial_ticks=1)
    assert benchmark.warmup_ticks == REFERENCE_WARMUP_TICKS == 200_000
    assert not benchmark.ready and benchmark.reference_tick is None
    initial_trial = {
        field: getattr(benchmark.sim, field).to_numpy().copy() for field in SIM_BUFFERS
    }
    advance(benchmark, 100)
    assert benchmark.sim.tick == 0 and not benchmark.pending
    assert benchmark.history == []
    for field, before in initial_trial.items():
        np.testing.assert_array_equal(getattr(benchmark.sim, field).to_numpy(), before)

    # The reference must use the population surviving at capture, including
    # occupants from multiple worlds, rather than the original random networks.
    parameters = np.zeros_like(weights(live))
    parameters[:, 0] = np.arange(len(parameters)) + 101
    live.brain_weights.copy_from_numpy(parameters)
    survivor_indices = (2, 11)
    cursor = creatures(live)
    for index in range(len(parameters)):
        cursor[index]["alive"] = index in survivor_indices
    cursor.apply()
    live.tick = REFERENCE_WARMUP_TICKS - 1
    assert not benchmark.prepare(live)
    advance(benchmark, 3)
    assert benchmark.sim.tick == 0 and benchmark.reference_tick is None

    live.tick = REFERENCE_WARMUP_TICKS + overshoot
    assert benchmark.prepare(live)
    assert benchmark.ready and benchmark.reference_tick == live.tick
    assert benchmark.sample_live_tick == live.tick and benchmark.sim.tick == 0
    assert not benchmark.prepare(live)
    np.testing.assert_array_equal(
        benchmark.reference_weights.to_numpy(), live.brain_weights.to_numpy()
    )
    np.testing.assert_array_equal(
        benchmark.reference_creatures.to_numpy(), live.creatures.to_numpy()
    )
    captured = benchmark.reference_weights.to_numpy().copy()
    first_trial = weights(benchmark.sim)
    for world in range(2):
        start = world * benchmark.sim.creature_capacity
        current = first_trial[start : start + 2]
        reference = first_trial[start + 2 : start + 4]
        np.testing.assert_array_equal(current, reference)
        assert set(current[:, 0]) == set(parameters[list(survivor_indices), 0])

    captured_tick = benchmark.reference_tick
    live.brain_weights.copy_from_numpy(np.zeros_like(parameters))
    live.tick += 17
    advance(benchmark, 1)
    assert benchmark.pending
    assert benchmark.prepare(live)
    assert benchmark.reference_tick == captured_tick
    np.testing.assert_array_equal(benchmark.reference_weights.to_numpy(), captured)
    assert benchmark.history[0]["source_counts"] == [2, 2]
    # The following trial changes only its current cohort, keeping that captured
    # survivor population as the frozen reference.
    np.testing.assert_array_equal(weights(benchmark.sim)[:2], 0)
    assert set(weights(benchmark.sim)[2:4, 0]) == set(parameters[list(survivor_indices), 0])


def test_waiting_reference_snapshot_resumes_wait_and_captures_later_population(device, tmp_path):
    live = make_sim(device, food=0)
    live.tick = REFERENCE_WARMUP_TICKS - 10
    benchmark = Benchmark(live, sample_size=3, trial_ticks=2)
    metadata, arrays = benchmark.snapshot_state()
    assert metadata["protocol_version"] == PROTOCOL_VERSION
    assert metadata["reference_tick"] is None
    assert metadata["warmup_ticks"] == REFERENCE_WARMUP_TICKS
    path = save_snapshot(
        tmp_path / "waiting-reference.npz",
        live,
        extra_sims={"benchmark": benchmark.sim},
        metadata={"benchmark": metadata},
        arrays=arrays,
    )
    saved = load_snapshot(path, device)
    restored = Benchmark.from_snapshot(
        saved.sim, saved.extra_sims["benchmark"], saved.metadata["benchmark"], saved.arrays
    )
    assert not restored.ready and restored.reference_tick is None
    assert restored.warmup_ticks == REFERENCE_WARMUP_TICKS
    assert not restored.prepare(saved.sim)
    advance(restored, 5)
    assert restored.sim.tick == 0 and not restored.pending
    changed = np.zeros_like(weights(saved.sim))
    changed[:, 0] = np.arange(len(changed)) + 41
    saved.sim.brain_weights.copy_from_numpy(changed)
    saved.sim.tick = REFERENCE_WARMUP_TICKS + 3
    assert restored.prepare(saved.sim)
    assert restored.ready and restored.reference_tick == saved.sim.tick
    np.testing.assert_array_equal(
        restored.reference_weights.to_numpy(), saved.sim.brain_weights.to_numpy()
    )
    assert set(weights(restored.sim)[:3, 0]) == {41, 42, 43}


def test_swapped_equal_cohorts_have_fresh_identical_bodies_and_food_layout(device):
    live = make_sim(device)
    before = live.creatures.to_numpy().copy()
    benchmark = Benchmark(live, warmup_ticks=0, sample_size=3)
    assert benchmark.sim.num_worlds == 2
    assert not benchmark.sim.mutation_enabled
    for world in range(2):
        assert benchmark.sim.inspect_world(world)["num_creatures"] == 6
        for local in range(6):
            creature = benchmark.sim.inspect(world, local)
            assert creature["energy"] == INITIAL_ENERGY
            assert creature["split_energy"] == 2 * INITIAL_ENERGY
            assert creature["radii"] == pytest.approx((0.5, 0.315))
            assert creature["max_speed"] == 12
            assert creature["birth_tick"] == creature["age_start_tick"] == 0
            assert creature["generation"] == creature["parent_guid"] == 0
            other = benchmark.sim.inspect(1 - world, local)
            assert creature["position"] == other["position"]
            assert creature["heading"] == other["heading"]
            assert creature["lineage"] == 3 - other["lineage"]
    food = benchmark.sim.food.to_numpy().reshape(2, -1)
    np.testing.assert_array_equal(food[0], food[1])
    np.testing.assert_array_equal(live.creatures.to_numpy(), before)
    params = weights(benchmark.sim)
    np.testing.assert_array_equal(params[:3], params[11:14])
    np.testing.assert_array_equal(params[3:6], params[8:11])


def test_offspring_inherit_cohort_and_brain_but_do_not_create_collected_energy(device):
    live = make_sim(device, creatures=1, food=0)
    benchmark = Benchmark(live, warmup_ticks=0, sample_size=1, trial_ticks=1)
    params = weights(benchmark.sim).copy()
    cursor = creatures(benchmark.sim)
    original_guids = {}
    for index in (0, benchmark.sim.creature_capacity + 1):
        original_guids[index] = cursor[index]["guid"].read()
        cursor[index]["energy"] = 2 * INITIAL_ENERGY + 100
    cursor.apply()
    advance(benchmark, 1)
    for world, parent in ((0, 0), (1, 1)):
        start = world * benchmark.sim.creature_capacity
        siblings = [benchmark.sim.inspect(world, local) for local in (parent, 2)]
        assert len({child["guid"] for child in siblings}) == 2
        for local, child in zip((parent, 2), siblings, strict=True):
            assert child["alive"] and child["lineage"] == 1
            assert child["generation"] == 1
            assert child["guid"] != original_guids[start + parent]
            assert child["parent_guid"] == original_guids[start + parent]
            assert child["birth_tick"] == child["age_start_tick"] == 1
            # Benchmark mutations remain disabled for both new descendants.
            np.testing.assert_array_equal(
                weights(benchmark.sim)[start + local], params[start + parent]
            )
        assert benchmark.sim.inspect_world(world)["births"] == 1
    assert benchmark.prepare(live)
    assert benchmark.history[0]["current"] == 4
    assert benchmark.history[0]["reference"] == 2
    assert benchmark.history[0]["score"] is None
    assert benchmark.history[0]["current_energy"] == 0
    assert benchmark.history[0]["reference_energy"] == 0
    assert benchmark.history[0]["reason"] == "no_energy_collected"


def test_reference_and_ecology_stay_frozen_when_live_population_changes(device):
    live = make_sim(device, creatures=3, food=4)
    benchmark = Benchmark(live, warmup_ticks=0, sample_size=3, trial_ticks=1)
    original = benchmark.reference_weights.to_numpy().copy()
    live.brain_weights.copy_from_numpy(np.zeros_like(weights(live)))
    live.food_per_second = 17
    live.food_energy = 9
    live.food_age_start = 17
    live.food_age_fade = 19
    live.tick = 91
    advance(benchmark, 1)
    benchmark.prepare(live)
    np.testing.assert_array_equal(benchmark.reference_weights.to_numpy(), original)
    params = weights(benchmark.sim)
    assert np.count_nonzero(params[:3]) == 0
    assert np.count_nonzero(params[3:6]) > 0
    assert benchmark.sim.initial_food == 4  # Small test allocations cap the fixed start.
    assert benchmark.sim.food_per_second == 10
    assert benchmark.sim.food_energy == 5
    assert benchmark.sim.food_age_start == 100_000
    assert benchmark.sim.food_age_fade == 0
    assert benchmark.history[0]["live_tick"] == 0
    assert benchmark.history[0]["completed_live_tick"] == 91
    assert benchmark.sample_live_tick == 91


def test_sampling_uses_alive_population_across_worlds_without_replacement(device):
    live = make_sim(device, creatures=2, capacity=4, food=0, worlds=2)
    params = np.zeros_like(weights(live))
    params[:, 0] = np.arange(len(params)) + 1
    live.brain_weights.copy_from_numpy(params)
    cursor = creatures(live)
    cursor[0]["alive"] = False
    cursor[1]["alive"] = False
    cursor.apply()
    benchmark = Benchmark(live, warmup_ticks=0, sample_size=2)
    current = weights(benchmark.sim)[:2, 0]
    assert set(current) == {5, 6}
    assert benchmark.sim.inspect_world(0)["num_creatures"] == 4


def test_trial_stops_at_exact_boundary_and_equal_intake_scores_half(device):
    live = make_sim(device, food=6)
    benchmark = Benchmark(live, warmup_ticks=0, trial_ticks=2, sample_size=3)
    fixed_brains(benchmark.sim, movement=0)
    feed(
        benchmark.sim, [(world, local, ENERGY_PER_UNIT) for world in range(2) for local in range(6)]
    )
    advance(benchmark, 7)
    assert benchmark.pending
    assert benchmark.sim.tick == 2
    benchmark.prepare(live)
    assert benchmark.history[0]["score"] == 0.5
    assert benchmark.history[0]["current_energy"] == 6
    assert benchmark.history[0]["reference_energy"] == 6
    assert benchmark.history[0]["world_energy"] == [[3, 3], [3, 3]]
    assert benchmark.history[0]["world_counts"] == [[3, 3], [3, 3]]
    assert benchmark.sim.tick == 0
    assert benchmark.trial == 1
    assert not benchmark.pending
    assert not benchmark.prepare(live)
    feed(benchmark.sim, [])
    advance(benchmark, 2)
    benchmark.prepare(live)
    assert benchmark.history[1]["score"] is None  # Previous trial's intake was reset.
    assert benchmark.history[1]["current_energy"] == 0


def test_expired_creatures_do_not_score_consumed_food(device):
    live = make_sim(device, food=2)
    live.food_age_start = 1
    benchmark = Benchmark(live, warmup_ticks=0, trial_ticks=1)
    fixed_brains(benchmark.sim, movement=0)
    feed(benchmark.sim, [(0, 0, ENERGY_PER_UNIT), (0, 4, ENERGY_PER_UNIT)])
    advance(benchmark, 1)
    benchmark.prepare(live)
    assert benchmark.history[0]["score"] is None
    assert benchmark.history[0]["current"] == benchmark.history[0]["reference"] == 0
    assert benchmark.history[0]["current_energy"] == benchmark.history[0]["reference_energy"] == 0
    assert benchmark.history[0]["reason"] == "no_energy_collected"


@pytest.mark.parametrize("all_extinct", [False, True])
def test_intake_score_survives_collectors_death_and_is_independent_of_survivor_share(
    device, all_extinct
):
    live = make_sim(device, creatures=1, food=2)
    benchmark = Benchmark(live, warmup_ticks=0, sample_size=1, trial_ticks=2)
    fixed_brains(benchmark.sim, movement=0)
    # Current gets 6 + 3 units; the reference gets 1 + 2 units across swapped worlds.
    feed(
        benchmark.sim,
        [
            (0, 0, 6 * ENERGY_PER_UNIT),
            (0, 1, ENERGY_PER_UNIT),
            (1, 1, 3 * ENERGY_PER_UNIT),
            (1, 0, 2 * ENERGY_PER_UNIT),
        ],
    )
    advance(benchmark, 1)
    cursor = creatures(benchmark.sim)
    for index in (0, benchmark.sim.creature_capacity + 1):
        cursor[index]["energy"] = 0
    if all_extinct:
        for index in (1, benchmark.sim.creature_capacity):
            cursor[index]["energy"] = 0
    cursor.apply()
    advance(benchmark, 1)
    benchmark.prepare(live)
    result = benchmark.history[0]
    assert result["current"] == 0
    assert result["reference"] == (0 if all_extinct else 2)
    assert result["current_energy"] == 9
    assert result["reference_energy"] == 3
    assert result["world_energy"] == [[6, 1], [3, 2]]
    assert result["score"] == 0.75
    assert result["reason"] is None


def test_descendants_collect_for_the_same_cohort_without_recounting_split_reserves(device):
    live = make_sim(device, creatures=1, food=2)
    benchmark = Benchmark(live, warmup_ticks=0, sample_size=1, trial_ticks=3)
    fixed_brains(benchmark.sim, movement=0)
    cursor = creatures(benchmark.sim)
    cursor[0]["energy"] = 2 * INITIAL_ENERGY
    cursor.apply()
    feed(benchmark.sim, [(0, 0, 5 * ENERGY_PER_UNIT), (0, 1, ENERGY_PER_UNIT)])
    advance(benchmark, 1)
    for local in (0, 2):
        child = benchmark.sim.inspect(0, local)
        assert child["alive"] and child["lineage"] == 1 and child["generation"] == 1
    feed(benchmark.sim, [(0, 0, ENERGY_PER_UNIT), (0, 2, 2 * ENERGY_PER_UNIT)])
    advance(benchmark, 2)
    benchmark.prepare(live)
    result = benchmark.history[0]
    assert result["current_energy"] == 8
    assert result["reference_energy"] == 1
    assert result["score"] == pytest.approx(8 / 9)


def test_intake_score_uses_age_adjusted_energy_and_combined_credit_rounding(device):
    live = make_sim(device, creatures=1, food=2)
    live.food_age_start = 10
    live.food_age_fade = 3
    benchmark = Benchmark(live, warmup_ticks=0, sample_size=1, trial_ticks=11)
    fixed_brains(benchmark.sim, movement=0)
    benchmark.sim.tick = 10
    # Current receives two thirds of eight quanta: floor(8 * 2 / 3) = 5,
    # rather than separately rounding each piece to 2 + 2 quanta.
    feed(benchmark.sim, [(0, 0, 4), (0, 0, 4), (1, 0, 6)])
    advance(benchmark, 1)
    benchmark.prepare(live)
    result = benchmark.history[0]
    assert result["current_energy"] == 5 / ENERGY_PER_UNIT
    assert result["reference_energy"] == 4 / ENERGY_PER_UNIT
    assert result["score"] == pytest.approx(5 / 9)


def test_cumulative_intake_exceeds_32_bits_without_overflow(device):
    live = make_sim(device, creatures=1, capacity=2, food=1)
    benchmark = Benchmark(live, warmup_ticks=0, sample_size=1, trial_ticks=2)
    fixed_brains(benchmark.sim, movement=0)
    for _ in range(2):
        feed(benchmark.sim, [(0, 0, 3_000_000_000), (1, 0, 1)])
        advance(benchmark, 1)
    benchmark.prepare(live)
    result = benchmark.history[0]
    assert result["current_energy"] == 6_000_000_000 / ENERGY_PER_UNIT
    assert result["reference_energy"] == 2 / ENERGY_PER_UNIT


@pytest.mark.parametrize("elapsed", [1, 2])
def test_snapshot_resumes_mid_trial_or_pending_result_and_reference(device, tmp_path, elapsed):
    live = make_sim(device, food=2)
    benchmark = Benchmark(live, warmup_ticks=0, trial_ticks=2)
    fixed_brains(benchmark.sim, movement=0)
    feed(benchmark.sim, [(0, 0, 2 * ENERGY_PER_UNIT), (0, 4, ENERGY_PER_UNIT)])
    advance(benchmark, elapsed)
    metadata, arrays = benchmark.snapshot_state()
    path = save_snapshot(
        tmp_path / "comparison.npz",
        live,
        extra_sims={"benchmark": benchmark.sim},
        metadata={"benchmark": metadata},
        arrays=arrays,
    )
    saved = load_snapshot(path, device)
    restored = Benchmark.from_snapshot(
        saved.sim, saved.extra_sims["benchmark"], saved.metadata["benchmark"], saved.arrays
    )
    assert restored.pending == (elapsed == 2)
    assert restored.ready and restored.warmup_ticks == 0 and restored.reference_tick == 0
    if elapsed < 2:
        advance(benchmark, 2 - elapsed)
        advance(restored, 2 - elapsed)
    assert restored.pending
    restored.prepare(saved.sim)
    benchmark.prepare(live)
    assert restored.history == benchmark.history
    assert restored.history[0]["score"] == pytest.approx(2 / 3)
    assert restored.history[0]["current_energy"] == 2
    assert restored.history[0]["reference_energy"] == 1
    np.testing.assert_array_equal(
        restored.reference_weights.to_numpy(), arrays["reference_weights"]
    )
    assert restored.trial == 1
    assert_sim_buffers_equal(
        {field: getattr(restored.sim, field).to_numpy() for field in SIM_BUFFERS},
        {field: getattr(benchmark.sim, field).to_numpy() for field in SIM_BUFFERS},
    )


@pytest.mark.parametrize("protocol", [None, 1, 2, 3, 4, 5])
def test_old_benchmark_protocol_is_rejected(device, protocol):
    live = make_sim(device, food=0)
    benchmark = Benchmark(live, warmup_ticks=0, trial_ticks=2, sample_size=1)
    metadata, arrays = benchmark.snapshot_state()
    if protocol is None:
        metadata.pop("protocol_version")
    else:
        metadata["protocol_version"] = protocol
    with pytest.raises(ValueError, match="Unsupported benchmark protocol"):
        Benchmark.from_snapshot(live, benchmark.sim, metadata, arrays)


def test_shrinking_population_keeps_trial_density_layout_and_reference_fixed(device):
    live = make_sim(device, food=0)
    benchmark = Benchmark(live, warmup_ticks=0, sample_size=3, trial_ticks=1)
    positions = [benchmark.sim.inspect(0, index)["position"] for index in range(6)]
    reference = benchmark.reference_weights.to_numpy().copy()
    cursor = creatures(live)
    cursor[0]["alive"] = False
    cursor[1]["alive"] = False
    cursor.apply()
    advance(benchmark, 1)
    benchmark.prepare(live)
    assert benchmark.sim.inspect_world(0)["num_creatures"] == 6
    assert [benchmark.sim.inspect(0, index)["position"] for index in range(6)] == positions
    params = weights(benchmark.sim)
    for index in range(3):
        np.testing.assert_array_equal(params[index], weights(live)[2])
    np.testing.assert_array_equal(benchmark.reference_weights.to_numpy(), reference)


def test_missing_live_cohort_is_unscored_with_explicit_reason(device):
    live = make_sim(device, food=0)
    benchmark = Benchmark(live, warmup_ticks=0, sample_size=3, trial_ticks=1)
    cursor = creatures(live)
    for index in range(3):
        cursor[index]["alive"] = False
    cursor.apply()
    advance(benchmark, 1)
    benchmark.prepare(live)
    advance(benchmark, 1)
    benchmark.prepare(live)
    result = benchmark.history[-1]
    assert result["score"] is None
    assert result["reason"] == "no_current_population"
    assert result["source_counts"] == [0, 3]
