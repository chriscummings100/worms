"""CPU-only checks for chunk ordering, resource bounds and frame boundaries."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from worms.app import App
from worms.submissions import BoundedSubmissions


class Encoder:
    def __init__(self, device, label):
        self.device = device
        self.label = label
        self.commands = []

    def finish(self):
        self.device.events.append(("finish", self.label))
        return self


class Device:
    """GPU progress happens only at a wait, exposing an unbounded producer."""

    def __init__(self):
        self.events = []
        self.encoders = []
        self.submitted = []
        self.completed = 0
        self.peak_pending = 0

    def create_command_encoder(self):
        encoder = Encoder(self, len(self.encoders))
        self.encoders.append(encoder)
        return encoder

    def submit_command_buffer(self, buffer):
        self.submitted.append(buffer)
        submission = len(self.submitted)
        self.events.append(("submit", submission, buffer.label))
        self.peak_pending = max(self.peak_pending, submission - self.completed)
        return submission

    def is_submit_finished(self, submission):
        return submission <= self.completed

    def wait_for_submit(self, submission):
        self.events.append(("wait", submission))
        for buffer in self.submitted[self.completed : submission]:
            for command in buffer.commands:
                command()
        self.completed = submission


class Sim:
    num_worlds = 1

    def __init__(self, device):
        self.device = device
        self.tick = self.gpu_tick = 0
        self.settings_dirty = True
        self.food_energy = 7
        self.gpu_energy = None
        self.observed = []

    def apply_settings(self, encoder):
        if self.settings_dirty:
            self.device.events.append(("settings", encoder.label))
            energy = self.food_energy
            encoder.commands.append(lambda: setattr(self, "gpu_energy", energy))
            self.settings_dirty = False

    def update(self, encoder):
        self.tick += 1
        tick = self.tick
        self.device.events.append(("record", tick, encoder.label))

        def execute():
            self.gpu_tick = tick
            self.observed.append((tick, self.gpu_energy))

        encoder.commands.append(execute)

    def reset(self):
        self.device.wait_for_submit(len(self.device.submitted))
        self.tick = self.gpu_tick = 0
        self.observed.clear()


def make_app(*, speed=100, window=False):
    app = App.__new__(App)
    app.device = Device()
    app._chunk_submissions = BoundedSubmissions(app.device)
    app.sim = Sim(app.device)
    app.benchmark = None
    app.snapshots = None
    app._restore_requested = app._snapshot_requested = False
    app._reset = app._step = app.paused = False
    app.selected_creature = None
    app._selected_world = 0
    app._next_population_update = float("inf")
    app.frame_count = 0
    app.speed = speed
    app.window = SimpleNamespace() if window else None
    app.render = lambda encoder: app.device.events.append(("draw", app.sim.tick, encoder.label))
    return app


def test_cpu_records_next_chunk_before_waiting_and_draws_after_all_ticks():
    app = make_app(speed=103)
    app.frame()
    events = app.device.events
    first_submit = next(i for i, event in enumerate(events) if event[0] == "submit")
    first_wait = next(i for i, event in enumerate(events) if event[0] == "wait")
    assert events[first_submit - 1][0] == "finish"
    assert [event[1] for event in events[:first_submit] if event[0] == "record"] == list(
        range(1, 26)
    )
    assert ("record", 75, 3) in events[first_submit:first_wait]
    assert [event for event in events if event[0] == "draw"] == [("draw", 103, 0)]
    assert app.sim.tick == app.sim.gpu_tick == 103
    assert app.sim.observed == [(tick, 7) for tick in range(1, 104)]
    assert app.device.peak_pending == 2
    assert not app._chunk_submissions.pending


def test_window_previous_final_is_covered_without_a_device_wide_wait():
    app = make_app(window=True)
    for frame in range(3):
        final = app.device.create_command_encoder()
        result = app.update_for_frame(final)
        assert result["chunks"] == 4 and result["submissions"] == 3
        app.render(final)
        app.device.submit_command_buffer(final.finish())
        assert app.sim.tick == (frame + 1) * 100
        # Simulate SlangPy owning the final submission, still executing when
        # the next callback begins; its ID is never passed back to the app.
    app.device.wait_for_submit(len(app.device.submitted))
    assert app.device.peak_pending == 2
    assert app.sim.gpu_tick == 300


def test_trial_preparation_checkpoints_and_inspector_poll_once_per_refresh():
    app = make_app(speed=1000)
    calls = []
    app._prepare_benchmark = lambda: calls.append(("benchmark", app.sim.tick))
    app.snapshots = SimpleNamespace(due=lambda tick: True)
    app.snapshot = lambda: calls.append(("checkpoint", app.sim.tick))
    app.selected_creature = 0
    app._next_stats_update = 0
    app.refresh_stats = lambda: calls.append(("inspect", app.sim.tick))
    app.frame()
    assert calls == [("benchmark", 0), ("checkpoint", 0), ("inspect", 0)]
    assert app.frame_count == 1 and app.sim.gpu_tick == 1000
    assert len(app.device.submitted) == 40
    assert app.device.peak_pending == 2


def test_pause_single_step_and_reset_do_not_expand_to_fast_speed():
    app = make_app(speed=1000)
    app.paused = True
    app.frame()
    assert app.sim.tick == 0 and app.sim.gpu_energy == 7
    app.step()
    app.frame()
    assert app.sim.tick == app.sim.gpu_tick == 1 and app.paused
    app._reset = True
    app._step = True
    app.frame()
    assert app.sim.tick == app.sim.gpu_tick == 0 and app.paused
    assert not app._step and not app._reset
    app.paused = False
    app.frame()
    assert app.sim.tick == app.sim.gpu_tick == 1000


def test_completed_trial_waits_until_next_refresh_even_when_later_chunks_remain():
    app = make_app(speed=103)

    class Benchmark:
        tick = 0
        pending = False

        def __init__(self):
            self.history = []

        def prepare(self, live_sim):
            if not self.pending:
                return False
            self.history.append(live_sim.tick)
            self.tick = 0
            self.pending = False
            return True

        def update(self, encoder):
            if not self.pending:
                self.tick += 1
                self.pending = self.tick == 37

    app.benchmark = Benchmark()
    app.frame()
    assert app.sim.tick == 103
    assert app.benchmark.tick == 37 and app.benchmark.pending
    assert app.benchmark.history == []
    app.frame()
    assert app.sim.tick == 206
    assert app.benchmark.tick == 37 and app.benchmark.pending
    assert app.benchmark.history == [103]


def test_single_encoder_api_does_not_submit_or_wait():
    app = make_app()
    encoder = app.device.create_command_encoder()
    app.update(encoder)
    assert app.sim.tick == 100 and app.sim.gpu_tick == 0
    assert not app.device.submitted
    assert not any(event[0] == "wait" for event in app.device.events)


def test_timing_hook_visits_each_actual_encoder_in_execution_order():
    app = make_app(speed=61)
    scopes = []

    @contextmanager
    def scope(encoder):
        first_tick = app.sim.tick
        yield
        scopes.append((encoder.label, first_tick, app.sim.tick))

    final = app.device.create_command_encoder()
    result = app.update_for_frame(final, chunk_scope=scope)
    assert scopes == [(1, 0, 25), (2, 25, 50), (0, 50, 61)]
    assert result["chunks"] == 3 and result["submissions"] == 2
    assert result["waits"] == 1
    assert result["cpu_finish_ms"] >= 0
    assert result["cpu_submit_ms"] >= 0
    assert result["wait_ms"] >= 0


@pytest.mark.parametrize("options", [{"chunk_size": 0}, {"max_in_flight": 1}])
def test_invalid_limits_fail_before_advancing_state(options):
    app = make_app()
    with pytest.raises(ValueError):
        app.update_for_frame(app.device.create_command_encoder(), **options)
    assert app.sim.tick == app.frame_count == 0
