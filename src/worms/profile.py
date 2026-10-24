"""Measure the real D3D12 update/render path with GPU timestamps.

Run ``python -m worms.profile --output artifacts/profile.json``. The first
measurement uses only frame boundary timestamps. A second measurement inserts
timestamps around individual dispatches to attribute GPU time, separately from
the throughput result so instrumentation overhead cannot inflate it.
"""

import argparse
import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import slangpy as spy

from worms.app import App
from worms.sim import TIME_STEP


class _GpuTimers:
    def __init__(self, device, count):
        self.pool = device.create_query_pool(type=spy.QueryType.timestamp, count=count)
        self.frequency = device.info.timestamp_frequency
        self.records = []
        self.index = 0
        self.encoder = None

    def reset(self):
        self.pool.reset()
        self.records.clear()
        self.index = 0

    @contextmanager
    def measure(self, encoder, name):
        start = self.index
        self.index += 2
        encoder.write_timestamp(self.pool, start)
        yield
        encoder.write_timestamp(self.pool, start + 1)
        self.records.append((name, start))

    def read(self):
        values = self.pool.get_results(0, self.index)
        return [
            (name, (values[index + 1] - values[index]) * 1000 / self.frequency)
            for name, index in self.records
        ]


class _TimedKernel:
    def __init__(self, kernel, timers, name):
        self.kernel, self.timers, self.name = kernel, timers, name

    def __getattr__(self, name):
        return getattr(self.kernel, name)

    def dispatch(self, *args, **kwargs):
        if kwargs["command_encoder"] is not self.timers.encoder:
            return self.kernel.dispatch(*args, **kwargs)
        with self.timers.measure(kwargs["command_encoder"], self.name):
            return self.kernel.dispatch(*args, **kwargs)


@contextmanager
def _instrument(sims, timers):
    originals = []
    for label, sim in sims:
        for name, kernel in vars(sim).copy().items():
            if isinstance(kernel, spy.ComputeKernel):
                originals.append((sim, name, kernel))
                setattr(sim, name, _TimedKernel(kernel, timers, f"{label}.{name.lstrip('_')}"))
    try:
        yield
    finally:
        for sim, name, kernel in originals:
            setattr(sim, name, kernel)


def _measure_frames(app, timers, samples, *, detailed=False):
    frames = []
    dispatches = defaultdict(list)
    for _ in range(samples):
        # A completed trial launches fresh worlds in a separate command buffer.
        # Keep that occasional setup outside the steady-state frame measurement.
        if app.benchmark is not None and app.benchmark.pending:
            app.benchmark.prepare(app.sim)
            app.device.wait()
        timers.reset()
        started = perf_counter()
        encoder = app.device.create_command_encoder()
        timers.encoder = encoder
        with timers.measure(encoder, "frame"):
            app.update(encoder)
            if detailed:
                with timers.measure(encoder, "render"):
                    app.render(encoder)
            else:
                app.render(encoder)
        timers.encoder = None
        recorded = perf_counter()
        command_buffer = encoder.finish()
        finished = perf_counter()
        app.device.submit_command_buffer(command_buffer)
        submitted = perf_counter()
        app.device.wait()
        waited = perf_counter()
        frame = {
            "cpu_record_ms": (recorded - started) * 1000,
            "cpu_finish_ms": (finished - recorded) * 1000,
            "cpu_submit_ms": (submitted - finished) * 1000,
            "wait_ms": (waited - submitted) * 1000,
            "wall_ms": (waited - started) * 1000,
        }
        for name, elapsed in timers.read():
            if name == "frame":
                frame["gpu_ms"] = elapsed
            else:
                dispatches[name].append(elapsed)
        frames.append(frame)
    return frames, dispatches


def profile_app(app, *, steps_per_frame=100, samples=10, warmup=200, side_sims=()):
    """Advance an existing headless App; return JSON-compatible diagnostic data.

    This deliberately waits after each frame to separate CPU preparation and GPU
    execution. The result is synchronized throughput, not a claim about window
    presentation rate. Supply ``(label, sim)`` pairs for any comparison worlds.
    """
    if steps_per_frame < 1 or samples < 1 or warmup < 0:
        raise ValueError("Need positive steps/samples and nonnegative warmup")
    sims = [("live", app.sim), *side_sims]
    app.paused = False
    app.speed = min(steps_per_frame, max(1, warmup))
    remaining = warmup
    while remaining:
        app.speed = min(steps_per_frame, remaining)
        app.frame()
        remaining -= app.speed
    app.device.wait()
    app.speed = steps_per_frame
    start_tick = app.sim.tick
    scenes = [
        {
            "label": label,
            "count": sim.num_worlds,
            "initial_creatures": sim.creatures_per_world,
            "population_at_start": [
                sim.inspect_world(world)["num_creatures"] for world in range(sim.num_worlds)
            ],
            "creature_capacity": sim.creature_capacity,
            "initial_food": sim.initial_food,
            "food_capacity": sim.food_capacity,
        }
        for label, sim in sims
    ]

    # Boundary-only timings are the throughput measurement; detailed queries
    # can serialize short passes and are used only for relative attribution.
    frame_timers = _GpuTimers(app.device, 2)
    frames, _ = _measure_frames(app, frame_timers, samples)
    kernel_count = sum(
        isinstance(value, spy.ComputeKernel) for _, sim in sims for value in vars(sim).values()
    )
    pass_timers = _GpuTimers(app.device, 2 * (steps_per_frame * (kernel_count + 8) + 2))
    with _instrument(sims, pass_timers):
        detailed_frames, dispatches = _measure_frames(app, pass_timers, samples, detailed=True)

    medians = {name: median(frame[name] for frame in frames) for name in frames[0]}
    cpu_ms = sum(medians[name] for name in ("cpu_record_ms", "cpu_finish_ms", "cpu_submit_ms"))
    gpu_ms = medians["gpu_ms"]
    limiting_side = (
        "gpu" if gpu_ms > 1.25 * cpu_ms else "cpu" if cpu_ms > 1.25 * gpu_ms else "mixed"
    )
    wall_ms = sum(frame["wall_ms"] for frame in frames)
    ticks_per_second = steps_per_frame * samples * 1000 / wall_ms
    detailed_gpu_ms = sum(frame["gpu_ms"] for frame in detailed_frames)
    passes = [
        {
            "name": name,
            "dispatches": len(values),
            "mean_ms_per_dispatch": mean(values),
            "ms_per_live_tick": sum(values) / (samples * steps_per_frame),
            "percent_gpu_frame": sum(values) / detailed_gpu_ms * 100,
        }
        for name, values in dispatches.items()
    ]
    passes.sort(key=lambda item: item["ms_per_live_tick"], reverse=True)
    return {
        "adapter": app.device.info.adapter_name,
        "api": app.device.info.api_name,
        "resolution": [app.texture.width, app.texture.height],
        "steps_per_frame": steps_per_frame,
        "samples": samples,
        "warmup_ticks": warmup,
        "start_tick": start_tick,
        "end_tick": app.sim.tick,
        "worlds": scenes,
        "median_frame": medians,
        "cpu_preparation_ms": cpu_ms,
        "limiting_side": limiting_side,
        "ticks_per_second": ticks_per_second,
        "realtime_multiple": ticks_per_second * TIME_STEP,
        "detailed_gpu_frame_mean_ms": detailed_gpu_ms / samples,
        "passes": passes,
        "frames": frames,
        "method": (
            "D3D12 timestamp queries; synchronized headless update plus render; no native UI, "
            "vsync, presentation or benchmark trial setup. Throughput uses two GPU timestamps "
            "per frame. A subsequent "
            "run adds per-dispatch timestamps for attribution. CPU preparation and GPU timeline "
            "are compared with a 25% margin; near ties are labelled mixed."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--creatures", type=int, default=64)
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--benchmark", action="store_true", help="Include the comparison worlds")
    parser.add_argument("--steps-per-frame", type=int, default=100)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("artifacts/profile.json"))
    args = parser.parse_args()
    app = App(
        headless=True,
        num_worlds=args.worlds,
        creatures_per_world=args.creatures,
        creature_capacity=args.capacity,
        benchmark=args.benchmark,
        snapshot_dir=None,
    )
    try:
        report = profile_app(
            app,
            steps_per_frame=args.steps_per_frame,
            samples=args.samples,
            warmup=args.warmup,
            side_sims=[("benchmark", app.benchmark.sim)] if app.benchmark is not None else (),
        )
    finally:
        app.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"{report['adapter']} | {report['ticks_per_second']:.1f} ticks/s "
        f"({report['realtime_multiple']:.2f}x real time) | {report['limiting_side']} limited"
    )
    print(
        f"CPU preparation {report['cpu_preparation_ms']:.2f} ms/frame; "
        f"GPU {report['median_frame']['gpu_ms']:.2f} ms/frame"
    )
    for stage in report["passes"]:
        print(
            f"{stage['name']:32} {stage['ms_per_live_tick']:8.4f} ms/tick "
            f"{stage['percent_gpu_frame']:5.1f}%"
        )
    print(args.output)


if __name__ == "__main__":
    main()
