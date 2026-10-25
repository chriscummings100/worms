"""Exercise sustained timing against actual D3D12 trial transitions and checkpoints."""

import hashlib

import pytest

from worms.app import App
from worms.benchmark import Benchmark
from worms.throughput import _FrameTimers, _measure_frame, _warmup, profile_throughput

pytestmark = pytest.mark.gpu


def make_app(snapshot_dir=None):
    app = App(
        debug_gpu=True,
        headless=True,
        creatures_per_world=2,
        creature_capacity=4,
        width=128,
        height=96,
        benchmark=False,
        snapshot_dir=snapshot_dir,
    )
    app.benchmark = Benchmark(app.sim, trial_ticks=5, sample_size=1, warmup_ticks=0)
    return app


@pytest.mark.parametrize("synchronized", [False, True])
def test_actual_trial_cap_and_restart_count_only_executed_benchmark_ticks(synchronized):
    app = make_app()
    try:
        _warmup(
            app,
            steps_per_frame=3,
            ticks=3,
            render=False,
            synchronized=synchronized,
            chunk_size=2,
        )
        assert app.sim.tick == app.benchmark.sim.tick == 3
        app.speed = 4
        timers = _FrameTimers(app.device, 2 if synchronized else 4)
        frames = [
            _measure_frame(app, timers, render=False, synchronized=synchronized, chunk_size=2)
            for _ in range(3)
        ]

        assert [frame["ticks"] for frame in frames] == [
            {"live": 4, "benchmark": 2},
            {"live": 4, "benchmark": 4},
            {"live": 4, "benchmark": 1},
        ]
        assert [len(frame["completed_trials"]) for frame in frames] == [0, 1, 0]
        assert frames[0]["benchmark_after"]["pending"]
        assert frames[1]["benchmark_before"]["tick"] == 5
        assert frames[1]["benchmark_after"]["trial"] == 1
        assert frames[1]["benchmark_after"]["tick"] == 4
        assert frames[1]["completed_trials"] == app.benchmark.history
        assert app.sim.tick == 15
        assert app.benchmark.sim.tick == 5 and app.benchmark.pending
        assert all(frame["gpu_ms"] > 0 and frame["wall_ms"] > 0 for frame in frames)
        assert all(frame["query_read_ms"] >= 0 for frame in frames)
        assert all(frame["chunks"] == (1 if synchronized else 2) for frame in frames)
        assert all(frame["submissions"] == frame["chunks"] for frame in frames)
        for frame in frames:
            assert frame["gpu_ms"] == pytest.approx(
                sum(scope["gpu_ms"] for scope in frame["gpu_scopes"])
            )
            assert frame["gpu_span_ms"] >= frame["gpu_ms"] - 1e-6
    finally:
        app.close()


def test_renderless_profile_preserves_source_and_suspends_automatic_snapshots(
    tmp_path, monkeypatch
):
    automatic = tmp_path / "automatic"
    app = make_app(snapshot_dir=automatic)
    try:
        _warmup(app, steps_per_frame=3, ticks=3, render=False)
        checkpoint = app.snapshot(tmp_path / "source.npz")
        source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        app.restore(checkpoint)
        assert app.paused and app.sim.tick == app.benchmark.sim.tick == 3
        store = app.snapshots
        store.interval = 1
        store.reset()
        assert store.due(app.sim.tick)

        def unexpected_render(encoder):
            pytest.fail("Render-off profiling must skip rendering during warmup and measurement")

        monkeypatch.setattr(app, "render", unexpected_render)
        report = profile_throughput(
            app,
            seconds=0.001,
            interval=1,
            steps_per_frame=4,
            warmup=6,
            render=False,
            chunk_size=3,
        )

        assert report["gpu_validation"] == {"d3d12_debug_layers": True, "rhi": True}
        assert report["render"] is False
        assert report["execution_mode"] == "chunked" and report["chunk_size"] == 3
        assert report["input_state"]["live_tick"] == 3
        assert report["start_tick"] == 9  # Warmup advances in batches of four and two.
        assert report["start_state"]["benchmark"]["trial"] == 1
        assert report["start_state"]["benchmark"]["tick"] == 2
        assert report["frame_count"] >= 1
        assert report["throughput_ticks"]["live"] == report["frame_count"] * 4
        assert report["end_tick"] == app.sim.tick == 9 + report["throughput_ticks"]["live"]
        expected_benchmark_ticks = (
            app.benchmark.trial * app.benchmark.trial_ticks + app.benchmark.sim.tick - 7
        )
        assert report["throughput_ticks"]["benchmark"] == expected_benchmark_ticks
        assert report["ticks_per_second"] > 0
        assert report["median_frame"]["gpu_ms"] > 0
        assert report["elapsed_seconds"] >= 0.001
        assert sum(window["elapsed_seconds"] for window in report["windows"]) == pytest.approx(
            report["elapsed_seconds"]
        )
        assert app.snapshots is store and store.last_tick == 0
        assert store.due(app.sim.tick)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == source_hash
        assert not automatic.exists()
        assert list(tmp_path.iterdir()) == [checkpoint]
    finally:
        app.close()


def test_chunk_and_render_timestamps_cover_all_submissions():
    app = make_app()
    try:
        app.speed = 6
        timers = _FrameTimers(app.device, 8)

        frame = _measure_frame(app, timers, render=True, chunk_size=2)

        assert frame["ticks"] == {"live": 6, "benchmark": 5}
        assert frame["chunks"] == frame["submissions"] == 3
        assert [scope["name"] for scope in frame["gpu_scopes"]] == ["update_chunk"] * 3 + ["render"]
        assert all(scope["gpu_ms"] > 0 for scope in frame["gpu_scopes"])
        assert frame["gpu_ms"] == pytest.approx(
            sum(scope["gpu_ms"] for scope in frame["gpu_scopes"])
        )
        assert frame["gpu_span_ms"] >= frame["gpu_ms"] - 1e-6
        assert frame["gpu_gap_ms"] == pytest.approx(frame["gpu_span_ms"] - frame["gpu_ms"])
        assert sum(
            frame[name]
            for name in (
                "cpu_record_ms",
                "cpu_finish_ms",
                "cpu_submit_ms",
                "wait_ms",
                "query_read_ms",
            )
        ) == pytest.approx(frame["wall_ms"])
    finally:
        app.close()
