"""Profile real GPU work without changing checkpoint continuation or its input file."""

import hashlib
import json

import pytest
import slangpy as spy

from tests.gpu_helpers import assert_sim_buffers_equal
from worms.app import App
from worms.benchmark import Benchmark
from worms.dispatch import BoundDispatch
from worms.profile import main, profile_app
from worms.snapshots import SIM_BUFFERS

pytestmark = pytest.mark.gpu


def test_gpu_profile_measures_dispatches_and_frame_without_changing_kernels():
    app = App(
        debug_gpu=True,
        headless=True,
        creatures_per_world=2,
        creature_capacity=4,
        width=128,
        height=96,
        benchmark=False,
        snapshot_dir=None,
    )
    try:
        original_step = app.sim._step.kernel
        original_root = app.sim._step.root
        report = profile_app(app, steps_per_frame=2, samples=2, warmup=2)
        assert report["input_tick"] == 0
        assert report["start_tick"] == 2
        assert report["end_tick"] == app.sim.tick == 6
        assert report["throughput_ticks"] == report["detailed_ticks"] == {"live": 4}
        assert report["ticks_per_second"] > 0
        assert report["gpu_validation"] == {"d3d12_debug_layers": True, "rhi": True}
        assert report["median_frame"]["gpu_ms"] > 0
        stages = {stage["name"]: stage for stage in report["passes"]}
        assert stages["live.step"]["dispatches"] == 4
        assert stages["live.step"]["calls_per_live_tick"] == 1
        assert stages["live.step"]["kind"] == "dispatch"
        assert stages["live.evaluate_brains"]["dispatches"] == 4
        assert stages["live.build_grid"]["dispatches"] == 4
        assert stages["live.build_grid"]["calls_per_live_tick"] == 1
        assert stages["live.build_food_grid"]["dispatches"] == 4
        assert stages["live.build_food_grid"]["calls_per_live_tick"] == 1
        assert stages["render"]["dispatches"] == 2
        assert stages["render"]["kind"] == "scope"
        assert stages["render"]["mean_cpu_record_ms"] is None
        for stage in stages.values():
            assert stage["mean_ms_per_dispatch"] >= 0
            assert 0 <= stage["median_ms_per_dispatch"] <= stage["max_ms_per_dispatch"]
            assert stage["median_ms_per_dispatch"] <= stage["p95_ms_per_dispatch"]
            assert stage["p95_ms_per_dispatch"] <= stage["max_ms_per_dispatch"]
            if stage["kind"] == "dispatch":
                assert stage["mean_cpu_record_ms"] >= 0
                assert 0 <= stage["median_cpu_record_ms"] <= stage["p95_cpu_record_ms"]
        assert 0 < sum(stage["percent_gpu_frame"] for stage in stages.values()) <= 100.01
        assert len(report["frames"]) == len(report["detailed_frames"]) == 2
        assert all(frame["ticks"] == {"live": 2} for frame in report["frames"])
        assert isinstance(app.sim._step, BoundDispatch)
        assert app.sim._step.kernel is original_step
        assert app.sim._step.root is not original_root
        assert app.sim._step.bindings["g_creatures"] is app.sim.creatures
    finally:
        app.close()


def make_checkpoint(tmp_path):
    app = App(
        debug_gpu=True,
        headless=True,
        creatures_per_world=2,
        creature_capacity=4,
        width=128,
        height=96,
        snapshot_dir=None,
    )
    try:
        app.benchmark = Benchmark(app.sim, trial_ticks=5, warmup_ticks=0)
        app.sim.food_energy = 7.25
        app.sim.food_per_second = 3.5
        app.speed = 3
        app.frame()
        return app.snapshot(tmp_path / "source.npz")
    finally:
        app.close()


def sim_buffers(sim):
    return {name: getattr(sim, name).to_numpy().copy() for name in SIM_BUFFERS}


def test_checkpoint_profile_replays_same_work_and_counts_partial_benchmark_frames(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    source_bytes = checkpoint.read_bytes()
    app = App(
        debug_gpu=True,
        headless=True,
        creatures_per_world=2,
        creature_capacity=4,
        width=128,
        height=96,
        benchmark=False,
        snapshot_dir=None,
        resume=checkpoint,
    )
    try:
        assert app.paused
        assert app.sim.tick == app.benchmark.sim.tick == 3
        report = profile_app(app, steps_per_frame=4, samples=2, warmup=0)
        assert report["input_tick"] == report["start_tick"] == 3
        assert report["end_tick"] == app.sim.tick == 11
        assert (
            report["throughput_ticks"]
            == report["detailed_ticks"]
            == {
                "live": 8,
                "benchmark": 6,
            }
        )
        expected_ticks = [{"live": 4, "benchmark": 2}, {"live": 4, "benchmark": 4}]
        assert [frame["ticks"] for frame in report["frames"]] == expected_ticks
        assert [frame["ticks"] for frame in report["detailed_frames"]] == expected_ticks
        stages = {stage["name"]: stage for stage in report["passes"]}
        assert stages["live.step"]["dispatches"] == 8
        assert stages["benchmark.step"]["dispatches"] == 6
        assert stages["benchmark.step"]["calls_per_live_tick"] == 0.75
        assert stages["benchmark.count"]["dispatches"] == 1
        assert "benchmark.initialize_cohorts" not in stages  # Trial setup is excluded.
        assert {scene["label"] for scene in report["worlds"]} == {"live", "benchmark"}
        assert app.sim.food_energy == 7.25 and app.sim.food_per_second == 3.5
        assert checkpoint.read_bytes() == source_bytes
        assert list(tmp_path.iterdir()) == [checkpoint]
        assert isinstance(app.sim._step, BoundDispatch)
        assert isinstance(app.benchmark.sim._step, BoundDispatch)
        assert isinstance(app.benchmark._count, spy.ComputeKernel)

        profiled_live = sim_buffers(app.sim)
        profiled_benchmark = sim_buffers(app.benchmark.sim)
        profiled_history = app.benchmark.history.copy()
        app.restore(checkpoint)
        app.paused = False
        app.speed = 4
        app.run(frames=2)
        assert app.sim.tick == 11 and app.benchmark.sim.tick == 4
        assert app.benchmark.history == profiled_history
        for sim, expected in (
            (app.sim, profiled_live),
            (app.benchmark.sim, profiled_benchmark),
        ):
            assert_sim_buffers_equal(sim_buffers(sim), expected)
    finally:
        app.close()


def test_profile_cli_can_exclude_saved_benchmark_without_modifying_checkpoint(tmp_path, capsys):
    checkpoint = make_checkpoint(tmp_path)
    source_bytes = checkpoint.read_bytes()
    output = tmp_path / "profile.json"
    main(
        [
            "--gpu-debug",
            "--resume",
            str(checkpoint),
            "--no-benchmark",
            "--capacity",
            "4",
            "--creatures",
            "2",
            "--steps-per-frame",
            "2",
            "--samples",
            "1",
            "--warmup",
            "0",
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["gpu_validation"] == {"d3d12_debug_layers": True, "rhi": True}
    assert report["input_tick"] == report["start_tick"] == 3
    assert report["end_tick"] == 5
    assert report["throughput_ticks"] == report["detailed_ticks"] == {"live": 2}
    assert [scene["label"] for scene in report["worlds"]] == ["live"]
    assert not any(stage["name"].startswith("benchmark.") for stage in report["passes"])
    assert report["checkpoint"] == {
        "path": str(checkpoint.resolve()),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
    }
    assert checkpoint.read_bytes() == source_bytes
    assert sorted(path.name for path in tmp_path.iterdir()) == ["profile.json", "source.npz"]
    assert "ticks/s" in capsys.readouterr().out


def test_pending_trial_profile_reports_fresh_population_and_suspends_snapshot_store(tmp_path):
    automatic = tmp_path / "automatic"
    app = App(
        debug_gpu=True,
        headless=True,
        creatures_per_world=2,
        creature_capacity=4,
        width=128,
        height=96,
        snapshot_dir=automatic,
    )
    try:
        app.benchmark = Benchmark(app.sim, trial_ticks=2, warmup_ticks=0)
        app.benchmark.sim.food_age_start = 2
        app.speed = 2
        app.frame()
        assert app.benchmark.pending
        assert [app.benchmark.sim.inspect_world(world)["num_creatures"] for world in range(2)] == [
            0,
            0,
        ]
        checkpoint = app.snapshot(tmp_path / "pending.npz")
        app.restore(checkpoint)
        store = app.snapshots
        store.interval = 1
        store.reset()
        assert store.due(app.sim.tick)
        original_live_step = app.sim._step.kernel
        original_benchmark_step = app.benchmark.sim._step.kernel
        original_count = app.benchmark._count

        report = profile_app(app, steps_per_frame=1, samples=1, warmup=0)
        scenes = {scene["label"]: scene for scene in report["worlds"]}
        assert report["input_tick"] == report["start_tick"] == 2
        assert report["end_tick"] == 3
        assert scenes["live"]["tick"] == 2
        assert scenes["benchmark"]["tick"] == 0
        assert scenes["benchmark"]["population_at_start"] == [4, 4]
        assert scenes["benchmark"]["food_at_start"] == [150, 150]
        assert (
            report["throughput_ticks"]
            == report["detailed_ticks"]
            == {
                "live": 1,
                "benchmark": 1,
            }
        )
        assert app.benchmark.sim.tick == 1 and app.benchmark.trial == 1
        assert len(app.benchmark.history) == 1
        assert app.sim._step.kernel is original_live_step
        assert app.benchmark.sim._step.kernel is original_benchmark_step
        assert app.sim._step.bindings["g_creatures"] is app.sim.creatures
        assert app.benchmark.sim._step.bindings["g_creatures"] is app.benchmark.sim.creatures
        assert app.benchmark._count is original_count
        assert app.snapshots is store and store.last_tick == 0
        assert store.due(app.sim.tick)
        assert not automatic.exists()
        assert list(tmp_path.iterdir()) == [checkpoint]
    finally:
        app.close()


@pytest.mark.parametrize("baseline_warmup", [3, 100])
def test_profile_replays_waiting_baseline_and_excludes_capture_setup(baseline_warmup):
    app = App(
        debug_gpu=True,
        headless=True,
        creatures_per_world=2,
        creature_capacity=4,
        width=128,
        height=96,
        benchmark=False,
        snapshot_dir=None,
    )
    try:
        app.benchmark = Benchmark(
            app.sim, trial_ticks=20, sample_size=1, warmup_ticks=baseline_warmup
        )
        report = profile_app(app, steps_per_frame=2, samples=3, warmup=0)
        benchmark_ticks = [0, 0, 2] if baseline_warmup == 3 else [0, 0, 0]
        expected_ticks = [{"live": 2, "benchmark": ticks} for ticks in benchmark_ticks]
        assert [frame["ticks"] for frame in report["frames"]] == expected_ticks
        assert [frame["ticks"] for frame in report["detailed_frames"]] == expected_ticks
        assert (
            report["throughput_ticks"]
            == report["detailed_ticks"]
            == {
                "live": 6,
                "benchmark": sum(benchmark_ticks),
            }
        )
        assert report["benchmark"] == {
            "warmup_ticks": baseline_warmup,
            "waiting_at_start": True,
            "waiting_at_end": baseline_warmup == 100,
            "reference_tick_at_start": None,
            "reference_tick_at_end": 4 if baseline_warmup == 3 else None,
        }
        stages = {stage["name"]: stage for stage in report["passes"]}
        assert "benchmark.initialize" not in stages
        assert "benchmark.initialize_brains" not in stages
        assert "benchmark.copy_food" not in stages
        if baseline_warmup == 3:
            assert stages["benchmark.step"]["dispatches"] == 2
            assert app.benchmark.reference_tick == 4
        else:
            assert not any(name.startswith("benchmark.") for name in stages)
            assert app.benchmark.sim.tick == 0 and not app.benchmark.ready
    finally:
        app.close()
