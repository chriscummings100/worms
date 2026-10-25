"""Exercise checkpoint boundaries and comparison continuity through the real app."""

from copy import deepcopy

import numpy as np
import pytest

from tests.gpu_helpers import assert_sim_buffers_equal
from worms.app import App, main
from worms.benchmark import PROTOCOL_VERSION, Benchmark
from worms.snapshots import SIM_BUFFERS, load_snapshot

pytestmark = pytest.mark.gpu


def make_app(tmp_path, *, benchmark=True, warmup_ticks=None):
    app = App(
        debug_gpu=True,
        headless=True,
        creatures_per_world=2,
        creature_capacity=8,
        width=257,
        height=149,
        benchmark=benchmark,
        snapshot_dir=tmp_path,
    )
    if benchmark and warmup_ticks is not None:
        app.benchmark = Benchmark(app.sim, warmup_ticks=warmup_ticks)
    return app


def state(app):
    values = {
        "live": {field: getattr(app.sim, field).to_numpy().copy() for field in SIM_BUFFERS},
        "tick": app.sim.tick,
        "dirty": app.sim._food_energy_dirty,
    }
    if app.benchmark is not None:
        metadata, arrays = app.benchmark.snapshot_state()
        values["benchmark"] = deepcopy(metadata)
        values["benchmark_tick"] = app.benchmark.sim.tick
        values["trial"] = {
            field: getattr(app.benchmark.sim, field).to_numpy().copy() for field in SIM_BUFFERS
        }
        values["reference"] = {name: value.copy() for name, value in arrays.items()}
    return values


def assert_state(actual, expected, *, continued=False):
    assert actual.keys() == expected.keys()
    for field, value in expected.items():
        if field in ("live", "trial", "reference"):
            assert actual[field].keys() == value.keys()
            if continued and field in ("live", "trial"):
                assert_sim_buffers_equal(actual[field], value)
                continue
            for name, array in value.items():
                np.testing.assert_array_equal(actual[field][name], array, err_msg=f"{field}.{name}")
        else:
            assert actual[field] == value, field


def test_app_restores_pending_trial_and_replays_comparison_without_losing_history(tmp_path):
    app = make_app(tmp_path, warmup_ticks=0)
    try:
        app.benchmark.trial_ticks = 4
        app.speed = 2
        app.frame()
        app.frame()
        assert app.sim.tick == app.benchmark.sim.tick == 4
        assert app.benchmark.pending
        assert app.benchmark.history == []
        app.sim.food_energy = 7.25
        app.sim.food_per_second = 3.5
        app.sim.input_mutation_probability = 0.19
        app.sim.policy_mutation_probability = 0.37
        app.selected_creature = 0
        app.refresh_stats()
        app.set_follow(True)
        app.camera.scale = 8
        path = app.snapshot(tmp_path / "manual.npz")
        checkpoint = state(app)
        app.run(frames=3)
        expected = state(app)
        assert len(expected["benchmark"]["history"]) == 2

        app.restore(path)
        assert app.paused and app.speed == 2
        assert app.selected_creature is None and not app.following
        assert app.camera.scale == 1 and app.camera.offset == (0, 0)
        assert app.sim.food_energy == 7.25 and app.sim.food_per_second == 3.5
        assert app.sim.input_mutation_probability == 0.19
        assert app.sim.policy_mutation_probability == 0.37
        assert not app.benchmark.sim.mutation_enabled
        assert_state(state(app), checkpoint)
        app.frame()  # A restored viewer stays paused, including its benchmark steps.
        assert app.sim.tick == 4
        assert app.benchmark.sim.tick == 0
        app.paused = False
        app.run(frames=3)
        assert_state(state(app), expected, continued=True)
    finally:
        app.close()


@pytest.mark.parametrize("speed", [3, 4])
def test_waiting_benchmark_roundtrips_and_captures_only_a_submitted_batch(tmp_path, speed):
    app = make_app(tmp_path, warmup_ticks=5)
    try:
        app.benchmark.trial_ticks = 20
        app.speed = speed
        initial_weights = app.sim.brain_weights.to_numpy().copy()
        app.frame()
        assert app.sim.tick == speed
        assert app.benchmark.sim.tick == 0 and not app.benchmark.ready
        assert app.benchmark.reference_tick is None
        assert app.benchmark.history == [] and not app.benchmark.pending
        np.testing.assert_array_equal(
            app.benchmark.reference_creatures.to_numpy().view(np.uint8), 0
        )
        np.testing.assert_array_equal(app.benchmark.reference_weights.to_numpy().view(np.uint8), 0)

        # Distinguish the later reference from the initial random population.
        weights = initial_weights.copy()
        weights.view(np.float16).reshape(-1)[0] += 1.0
        app.sim.brain_weights.copy_from_numpy(weights)
        app.sim.input_mutation_probability = 0.17
        app.sim.policy_mutation_probability = 0.29
        app.sim.food_per_second = 8.5
        waiting_path = app.snapshot(tmp_path / "waiting.npz")
        waiting = state(app)
        assert waiting["benchmark"]["protocol_version"] == PROTOCOL_VERSION
        assert waiting["benchmark"]["warmup_ticks"] == 5
        assert waiting["benchmark"]["reference_tick"] is None

        app.frame()
        app.restore(waiting_path)
        assert app.paused and app.sim.tick == speed
        assert not app.benchmark.ready and app.benchmark.reference_tick is None
        assert app.benchmark.warmup_ticks == 5
        assert app.sim.input_mutation_probability == 0.17
        assert app.sim.policy_mutation_probability == 0.29
        assert app.sim.food_per_second == 8.5
        assert_state(state(app), waiting)
        app.frame()
        assert_state(state(app), waiting)

        app.paused = False
        app.frame()
        assert app.sim.tick == 2 * speed
        # Crossing tick 5 while recording a batch cannot capture unfinished GPU work.
        assert not app.benchmark.ready and app.benchmark.reference_tick is None
        assert app.benchmark.sim.tick == 0 and not app.benchmark.pending
        reference = {
            "reference_creatures": app.sim.creatures.to_numpy().copy(),
            "reference_weights": app.sim.brain_weights.to_numpy().copy(),
        }
        app.frame()
        assert app.sim.tick == 3 * speed
        assert app.benchmark.ready
        assert app.benchmark.reference_tick == app.benchmark.sample_live_tick == 2 * speed
        assert app.benchmark.sim.tick == speed
        assert app.benchmark.history == [] and not app.benchmark.pending
        for name, expected in reference.items():
            np.testing.assert_array_equal(getattr(app.benchmark, name).to_numpy(), expected)
        assert not np.array_equal(app.benchmark.reference_weights.to_numpy(), initial_weights)

        captured_path = app.snapshot(tmp_path / "captured.npz")
        captured = state(app)
        app.frame()
        app.restore(captured_path)
        assert app.paused and app.benchmark.ready
        assert app.benchmark.warmup_ticks == 5
        assert app.benchmark.reference_tick == 2 * speed
        assert_state(state(app), captured)
        app.frame()
        assert_state(state(app), captured)
        app.paused = False
        app.frame()
        assert app.benchmark.reference_tick == 2 * speed
        assert app.benchmark.sim.tick == 2 * speed
        for name, expected in reference.items():
            np.testing.assert_array_equal(getattr(app.benchmark, name).to_numpy(), expected)
    finally:
        app.close()


def test_run_saves_last_submitted_tick_when_it_ends_on_snapshot_boundary(tmp_path):
    app = make_app(tmp_path, benchmark=False)
    try:
        app.snapshots.interval = 4
        app.speed = 2
        app.run(frames=2)
        assert app.sim.tick == 4
        latest = app.snapshots.latest
        assert latest is not None
        saved = load_snapshot(latest, app.device)
        assert saved.sim.tick == 4
        for field in SIM_BUFFERS:
            np.testing.assert_array_equal(
                getattr(app.sim, field).to_numpy(), getattr(saved.sim, field).to_numpy()
            )
        app.paused = True
        app.run(frames=1)
        assert list(tmp_path.glob("auto-*.npz")) == [latest]
        app.restore(latest)
        app.step()
        app.run(frames=1)
        assert app.paused and app.sim.tick == 5
        assert app.snapshots.latest == latest
    finally:
        app.close()


def test_reset_keeps_live_settings_and_restarts_reference_and_checkpoint_schedule(tmp_path):
    app = make_app(tmp_path, warmup_ticks=0)
    try:
        app.benchmark.trial_ticks = 2
        app.run(frames=3)
        assert len(app.benchmark.history) == 1
        previous_benchmark = app.benchmark
        app.snapshots.last_tick = 3
        app.sim.food_energy = 6.25
        app.sim.food_per_second = 8.5
        app.sim.food_age_start = 1234
        app.sim.food_age_fade = 56
        app.sim.input_mutation_probability = 0.17
        app.sim.policy_mutation_probability = 0.25
        app.speed = 5
        app.paused = True
        app.reset()
        app.run(frames=1)
        assert app.sim.tick == 0 and app.paused and app.speed == 5
        assert app.snapshots.last_tick == 0
        assert app.benchmark is not previous_benchmark
        assert app.benchmark.history == [] and not app.benchmark.pending
        assert app.benchmark.warmup_ticks == 200_000
        assert not app.benchmark.ready and app.benchmark.reference_tick is None
        assert app.benchmark.sim.tick == 0 and app.benchmark.sample_live_tick == 0
        for sim in (app.sim, app.benchmark.sim):
            assert sim.food_age_start == 1234
            assert sim.food_age_fade == 56
        assert app.sim.food_energy == 6.25 and app.sim.food_per_second == 8.5
        assert app.sim.input_mutation_probability == 0.17
        assert app.sim.policy_mutation_probability == 0.25
        assert not app.benchmark.sim.mutation_enabled
        assert app.benchmark.sim.food_energy == 5 and app.benchmark.sim.food_per_second == 10
        assert app.benchmark.sim.initial_food == 150
        np.testing.assert_array_equal(
            app.benchmark.reference_creatures.to_numpy().view(np.uint8), 0
        )
        np.testing.assert_array_equal(app.benchmark.reference_weights.to_numpy().view(np.uint8), 0)
    finally:
        app.close()


def test_headless_cli_resume_advances_and_explicit_one_speed_overrides_saved_speed(
    tmp_path, capsys
):
    app = make_app(tmp_path, benchmark=False)
    try:
        app.run(frames=4)
        app.speed = 5
        path = app.snapshot(tmp_path / "cli.npz")
    finally:
        app.close()
    output = tmp_path / "resumed.png"
    result = main(
        [
            "--headless",
            "--gpu-debug",
            "--resume",
            str(path),
            "--no-benchmark",
            "--no-snapshots",
            "--speed",
            "1",
            "--frames",
            "2",
            "--output",
            str(output),
        ]
    )
    assert result == 0
    assert output.is_file()
    assert "Saved tick 6 to" in capsys.readouterr().out
