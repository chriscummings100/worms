"""CPU checks for sustained timing accounting and checkpoint protection."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from worms import throughput


def state(live_tick, trial=0, tick=0, ready=True):
    return {
        "live_tick": live_tick,
        "benchmark": {"trial": trial, "tick": tick, "trial_ticks": 10, "ready": ready},
    }


def test_trial_reset_counts_work_without_counting_unused_frame_steps():
    # A 4-step frame hits the 10-tick trial cap after only two benchmark updates.
    assert throughput._tick_advances(state(100, tick=8), state(104, tick=10)) == {
        "live": 4,
        "benchmark": 2,
    }
    # The next frame resets tick 10 to zero, then performs four actual updates.
    assert throughput._tick_advances(state(104, tick=10), state(108, trial=1, tick=4)) == {
        "live": 4,
        "benchmark": 4,
    }
    assert throughput._tick_advances(state(99, ready=False), state(103, tick=4)) == {
        "live": 4,
        "benchmark": 4,
    }
    assert throughput._tick_advances(state(95, ready=False), state(99, ready=False)) == {
        "live": 4,
        "benchmark": 0,
    }


class Clock:
    def __init__(self):
        self.now = 0.0

    def advance(self, seconds):
        self.now += seconds


class FakeEncoder:
    def __init__(self, clock):
        self.clock = clock

    def finish(self):
        self.clock.advance(0.001)
        return self


class FakeDevice:
    info = SimpleNamespace(adapter_name="test GPU", api_name="test API")

    def __init__(self, clock):
        self.clock = clock
        self.waits = 0

    def create_command_encoder(self):
        self.clock.advance(0.001)
        return FakeEncoder(self.clock)

    def submit_command_buffer(self, command_buffer):
        self.clock.advance(0.001)
        return 1

    def wait_for_submit(self, submission):
        self.clock.advance(0.01)
        self.waits += 1

    def wait(self):
        pass


class FakeTimers:
    def __init__(self, device, count):
        self.clock = device.clock
        self.scopes = []
        self.count = count

    def reset(self):
        self.scopes.clear()

    @contextmanager
    def measure(self, encoder, name):
        yield
        self.scopes.append({"name": name, "gpu_ms": 10.0})

    def read(self):
        self.clock.advance(0.002)
        gpu_ms = 10.0 * len(self.scopes)
        gap = 4.0 * (len(self.scopes) - 1)
        assert self.count >= 2 * len(self.scopes)
        return {
            "gpu_ms": gpu_ms,
            "gpu_span_ms": gpu_ms + gap,
            "gpu_gap_ms": gap,
            "gpu_scopes": self.scopes.copy(),
        }


class FakeApp:
    def __init__(self, clock):
        self.device = FakeDevice(clock)
        self.clock = clock
        self.window = None
        self.snapshots = object()
        self.debug_gpu = False
        self.texture = SimpleNamespace(width=1280, height=720)
        self.sim = SimpleNamespace(tick=0, num_worlds=1, inspect_world=self.inspect_world)
        self.benchmark = SimpleNamespace(
            sim=SimpleNamespace(tick=0, num_worlds=2, inspect_world=self.inspect_world),
            trial=0,
            trial_ticks=10,
            ready=True,
            pending=False,
            reference_tick=0,
            history=[],
        )
        self.render_calls = 0
        self.prepare_calls = 0

    def inspect_world(self, world):
        self.clock.advance(0.005)
        return {"num_creatures": 2, "num_food": 3}

    def update(self, encoder):
        self.prepare_trial()
        self.advance_steps(self.speed)

    def prepare_trial(self):
        benchmark = self.benchmark
        if benchmark.pending:
            # Model the expensive CPU/readback/setup path at a trial rollover.
            self.clock.advance(0.03)
            self.prepare_calls += 1
            benchmark.history.append({"trial": benchmark.trial})
            benchmark.trial += 1
            benchmark.sim.tick = 0
            benchmark.pending = False

    def advance_steps(self, steps):
        benchmark = self.benchmark
        self.clock.advance(0.005)
        self.sim.tick += steps
        benchmark.sim.tick = min(benchmark.trial_ticks, benchmark.sim.tick + steps)
        benchmark.pending = benchmark.sim.tick == benchmark.trial_ticks

    def render(self, encoder):
        self.clock.advance(0.003)
        self.render_calls += 1


class ChunkedFakeApp(FakeApp):
    """Give the profiler known intermediate costs, including blocking waits."""

    def __init__(self, clock):
        super().__init__(clock)
        self.chunk_sizes = []

    def update_for_frame(self, final_encoder, *, chunk_size, chunk_scope=None):
        self.chunk_sizes.append(chunk_size)
        self.prepare_trial()
        stats = throughput.chunk_timings()
        remaining = self.speed
        while remaining:
            final = remaining <= chunk_size
            encoder = final_encoder if final else self.device.create_command_encoder()
            with chunk_scope(encoder) if chunk_scope is not None else nullcontext():
                self.advance_steps(min(remaining, chunk_size))
            remaining -= min(remaining, chunk_size)
            stats["chunks"] += 1
            if not final:
                submission = self.device.submit_command_buffer(encoder.finish())
                self.device.wait_for_submit(submission)
                stats["submissions"] += 1
                stats["waits"] += 1
                stats["cpu_finish_ms"] += 1.0
                stats["cpu_submit_ms"] += 1.0
                stats["wait_ms"] += 10.0
        return stats


def describe_worlds(app):
    return [
        {"label": label, "tick": sim.tick, "world_counters": [sim.inspect_world(0)]}
        for label, sim in throughput._sims(app)
    ]


def test_sustained_rate_includes_counters_queries_and_trial_setup(monkeypatch):
    clock = Clock()
    app = FakeApp(clock)
    snapshot_store = app.snapshots
    monkeypatch.setattr(throughput, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(throughput, "_FrameTimers", FakeTimers)
    monkeypatch.setattr(throughput, "_describe_worlds", describe_worlds)
    reports = []

    def progress(window):
        reports.append(window)
        clock.advance(0.007)

    report = throughput.profile_throughput(
        app,
        seconds=0.3,
        interval=0.08,
        steps_per_frame=4,
        warmup=0,
        render=False,
        progress=progress,
        synchronized=True,
    )
    assert app.snapshots is snapshot_store
    assert app.render_calls == 0
    assert app.device.waits == report["frame_count"]
    assert report["elapsed_seconds"] >= 0.3
    assert report["ticks_per_second"] == pytest.approx(app.sim.tick / report["elapsed_seconds"])
    assert report["throughput_ticks"]["benchmark"] == (
        app.benchmark.trial * app.benchmark.trial_ticks + app.benchmark.sim.tick
    )
    assert len(reports) == len(report["windows"])
    expected_counters = 0.01 + len(reports) * 0.015
    expected_reporting = (len(reports) - 1) * 0.007
    assert report["elapsed_seconds"] == pytest.approx(
        sum(frame["wall_ms"] for frame in report["frames"]) / 1000
        + expected_counters
        + expected_reporting
    )
    assert sum(frame["query_read_ms"] for frame in report["frames"]) == pytest.approx(
        report["frame_count"] * 2
    )
    transitions = [frame for frame in report["frames"] if frame["completed_trials"]]
    assert len(transitions) == app.prepare_calls > 0
    assert all(frame["cpu_record_ms"] >= 35 for frame in transitions)
    assert sum(window["elapsed_seconds"] for window in report["windows"]) == pytest.approx(
        report["elapsed_seconds"]
    )


def test_no_render_also_skips_warmup_render(monkeypatch):
    clock = Clock()
    app = FakeApp(clock)
    monkeypatch.setattr(throughput, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(throughput, "_FrameTimers", FakeTimers)
    monkeypatch.setattr(throughput, "_describe_worlds", describe_worlds)
    report = throughput.profile_throughput(
        app,
        seconds=0.01,
        interval=1,
        steps_per_frame=4,
        warmup=6,
        render=False,
        synchronized=True,
    )
    assert app.render_calls == 0
    assert report["start_tick"] == 6
    assert report["throughput_ticks"]["live"] == 4
    assert app.device.waits == 3  # Two warmup frames and one measured frame.


def test_chunked_phase_accounting_counts_intermediate_costs_once(monkeypatch):
    clock = Clock()
    app = ChunkedFakeApp(clock)
    app.speed = 8
    monkeypatch.setattr(throughput, "perf_counter", lambda: clock.now)
    timers = FakeTimers(app.device, 8)

    frame = throughput._measure_frame(app, timers, render=True, chunk_size=3)

    assert frame["ticks"] == {"live": 8, "benchmark": 8}
    assert (frame["chunks"], frame["submissions"], frame["waits"]) == (3, 3, 3)
    assert frame["cpu_record_ms"] == pytest.approx(21)
    assert frame["cpu_finish_ms"] == pytest.approx(3)
    assert frame["cpu_submit_ms"] == pytest.approx(3)
    assert frame["wait_ms"] == pytest.approx(30)
    assert frame["query_read_ms"] == pytest.approx(2)
    assert frame["wall_ms"] == pytest.approx(59)
    assert sum(
        frame[name]
        for name in ("cpu_record_ms", "cpu_finish_ms", "cpu_submit_ms", "wait_ms", "query_read_ms")
    ) == pytest.approx(frame["wall_ms"])
    assert [scope["name"] for scope in frame["gpu_scopes"]] == ["update_chunk"] * 3 + ["render"]


def test_frame_queries_sum_work_and_measure_cross_submission_span_once():
    calls = []

    def results(first, count):
        calls.append((first, count))
        return [100, 110, 130, 160, 170, 175]

    timers = throughput._FrameTimers.__new__(throughput._FrameTimers)
    timers.pool = SimpleNamespace(get_results=results)
    timers.frequency = 1000
    timers.index = 6
    timers.records = [
        {"name": name, "index": index}
        for name, index in (("update_chunk", 0), ("update_chunk", 2), ("render", 4))
    ]

    result = timers.read()

    assert calls == [(0, 6)]
    assert result["gpu_ms"] == 45
    assert result["gpu_span_ms"] == 75
    assert result["gpu_gap_ms"] == 30
    assert [scope["gpu_ms"] for scope in result["gpu_scopes"]] == [10, 30, 5]


def test_default_chunked_profile_warms_up_the_measured_path(monkeypatch):
    clock = Clock()
    app = ChunkedFakeApp(clock)
    monkeypatch.setattr(throughput, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(throughput, "_FrameTimers", FakeTimers)
    monkeypatch.setattr(throughput, "_describe_worlds", describe_worlds)

    report = throughput.profile_throughput(
        app, seconds=0.01, interval=1, steps_per_frame=4, warmup=6, render=False, chunk_size=3
    )

    assert app.chunk_sizes == [3, 3, 3]  # Two warmup batches and one measured batch.
    assert app.render_calls == 0
    assert report["execution_mode"] == "chunked"
    assert report["chunk_size"] == 3 and report["max_in_flight"] == 2
    assert report["gpu_timestamps_per_frame"] == 4
    assert report["start_tick"] == 6 and report["end_tick"] == 10
    assert report["frames"][0]["chunks"] == 2


@pytest.mark.parametrize(
    "option,value",
    [
        ("--seconds", "nan"),
        ("--seconds", "inf"),
        ("--seconds", "0"),
        ("--interval", "-1"),
        ("--steps-per-frame", "0"),
        ("--warmup", "-1"),
        ("--chunk-size", "0"),
    ],
)
def test_invalid_arguments_fail_before_gpu_creation(monkeypatch, option, value):
    def unexpected_app(**kwargs):
        pytest.fail("Invalid arguments must not create a GPU application")

    monkeypatch.setattr(throughput, "App", unexpected_app)
    with pytest.raises(SystemExit) as stopped:
        throughput.main(["--resume", "absent.npz", option, value])
    assert stopped.value.code == 2


def test_refuses_to_overwrite_checkpoint_before_gpu_creation(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "source.npz"
    checkpoint.write_bytes(b"original checkpoint")

    def unexpected_app(**kwargs):
        pytest.fail("The overwrite guard must run before GPU initialization")

    monkeypatch.setattr(throughput, "App", unexpected_app)
    with pytest.raises(SystemExit) as stopped:
        throughput.main(["--resume", str(checkpoint), "--output", str(checkpoint)])
    assert stopped.value.code == 2
    assert "--output must differ" in capsys.readouterr().err
    assert checkpoint.read_bytes() == b"original checkpoint"
