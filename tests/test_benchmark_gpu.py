"""Real GPU checks for fair cohorts, inherited scores and trial isolation."""

import numpy as np
import pytest
import slangpy as spy

from worms.benchmark import Benchmark
from worms.sim import BRAIN_PARAMETERS, INITIAL_ENERGY, SHADER_DIR, Sim
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
    return sim.brain_weights.to_numpy().view(np.float32).reshape(-1, BRAIN_PARAMETERS)


def advance(benchmark, ticks):
    encoder = benchmark.sim.device.create_command_encoder()
    for _ in range(ticks):
        benchmark.update(encoder)
    benchmark.sim.device.submit_command_buffer(encoder.finish())


def test_swapped_equal_cohorts_have_fresh_identical_bodies_and_food_layout(device):
    live = make_sim(device)
    before = live.creatures.to_numpy().copy()
    benchmark = Benchmark(live, sample_size=3)
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


def test_trial_scores_offspring_and_disables_mutation(device):
    live = make_sim(device, creatures=1, food=0)
    benchmark = Benchmark(live, sample_size=1, trial_ticks=1)
    params = weights(benchmark.sim).copy()
    cursor = creatures(benchmark.sim)
    for index in (0, benchmark.sim.creature_capacity + 1):
        cursor[index]["energy"] = 2 * INITIAL_ENERGY + 100
    cursor.apply()
    advance(benchmark, 1)
    for world, parent in ((0, 0), (1, 1)):
        child = benchmark.sim.inspect(world, 2)
        assert child["alive"]
        assert child["lineage"] == 1
        assert child["generation"] == 1
        start = world * benchmark.sim.creature_capacity
        np.testing.assert_array_equal(weights(benchmark.sim)[start + 2], params[start + parent])
    assert benchmark.prepare(live)
    assert benchmark.history[0]["current"] == 4
    assert benchmark.history[0]["reference"] == 2
    assert benchmark.history[0]["score"] == pytest.approx(2 / 3)


def test_reference_and_ecology_stay_frozen_when_live_population_changes(device):
    live = make_sim(device, creatures=3, food=4)
    benchmark = Benchmark(live, sample_size=3, trial_ticks=1)
    original = benchmark.reference_weights.to_numpy().copy()
    live.brain_weights.copy_from_numpy(np.zeros_like(weights(live)))
    live.food_per_second = 10
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
    assert benchmark.sim.food_per_second == 0
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
    benchmark = Benchmark(live, sample_size=2)
    current = weights(benchmark.sim)[:2, 0]
    assert set(current) == {5, 6}
    assert benchmark.sim.inspect_world(0)["num_creatures"] == 4


def test_trial_stops_at_exact_boundary_and_identical_cohorts_score_half(device):
    live = make_sim(device, food=0)
    benchmark = Benchmark(live, trial_ticks=2, sample_size=3)
    advance(benchmark, 7)
    assert benchmark.pending
    assert benchmark.sim.tick == 2
    benchmark.prepare(live)
    assert benchmark.history[0]["score"] == 0.5
    assert benchmark.history[0]["world_counts"] == [[3, 3], [3, 3]]
    assert benchmark.sim.tick == 0
    assert benchmark.trial == 1
    assert not benchmark.pending
    assert not benchmark.prepare(live)


def test_total_extinction_is_unscored_instead_of_a_false_loss(device):
    live = make_sim(device, food=0)
    live.food_age_start = 1
    benchmark = Benchmark(live, trial_ticks=1)
    advance(benchmark, 1)
    benchmark.prepare(live)
    assert benchmark.history[0]["score"] is None
    assert benchmark.history[0]["current"] == benchmark.history[0]["reference"] == 0
    assert benchmark.history[0]["reason"] == "extinction"


@pytest.mark.parametrize("elapsed", [1, 2])
def test_snapshot_resumes_mid_trial_or_pending_result_and_reference(device, tmp_path, elapsed):
    live = make_sim(device, food=0)
    benchmark = Benchmark(live, trial_ticks=2)
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
    if elapsed < 2:
        advance(benchmark, 2 - elapsed)
        advance(restored, 2 - elapsed)
    assert restored.pending
    restored.prepare(saved.sim)
    benchmark.prepare(live)
    assert restored.history == benchmark.history
    assert restored.history[0]["score"] == 0.5
    np.testing.assert_array_equal(
        restored.reference_weights.to_numpy(), arrays["reference_weights"]
    )
    assert restored.trial == 1
    for field in SIM_BUFFERS:
        np.testing.assert_array_equal(
            getattr(restored.sim, field).to_numpy(),
            getattr(benchmark.sim, field).to_numpy(),
            err_msg=field,
        )


def test_shrinking_population_keeps_trial_density_layout_and_reference_fixed(device):
    live = make_sim(device, food=0)
    benchmark = Benchmark(live, sample_size=3, trial_ticks=1)
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
    benchmark = Benchmark(live, sample_size=3, trial_ticks=1)
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
