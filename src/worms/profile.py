"""Measure the real D3D12 update/render path with GPU timestamps.

Run ``python -m worms.profile --output artifacts/profile.json``. The first
measurement uses only frame boundary timestamps. The same state is then replayed
with timestamps around individual dispatches. Use --resume to profile a saved
world, preserving its benchmark unless --no-benchmark is supplied.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import slangpy as spy

from worms.app import App
from worms.dispatch import BoundDispatch
from worms.sim import TIME_STEP
from worms.snapshots import SIM_SETTINGS


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
    def measure(self, encoder, name, *, kind="scope"):
        start = self.index
        self.index += 2
        record = {"name": name, "index": start, "kind": kind, "cpu_record_ms": None}
        encoder.write_timestamp(self.pool, start)
        yield record
        encoder.write_timestamp(self.pool, start + 1)
        self.records.append(record)

    def read(self):
        values = self.pool.get_results(0, self.index)
        return [
            {
                **record,
                "gpu_ms": (values[record["index"] + 1] - values[record["index"]])
                * 1000
                / self.frequency,
            }
            for record in self.records
        ]


class _TimedKernel:
    def __init__(self, kernel, timers, name):
        self.kernel, self.timers, self.name = kernel, timers, name

    def __getattr__(self, name):
        return getattr(self.kernel, name)

    def dispatch(self, *args, **kwargs):
        if kwargs["command_encoder"] is not self.timers.encoder:
            return self.kernel.dispatch(*args, **kwargs)
        with self.timers.measure(kwargs["command_encoder"], self.name, kind="dispatch") as record:
            started = perf_counter()
            result = self.kernel.dispatch(*args, **kwargs)
            record["cpu_record_ms"] = (perf_counter() - started) * 1000
            return result


@contextmanager
def _instrument(owners, timers):
    originals = []
    for label, owner in owners:
        for name, kernel in vars(owner).copy().items():
            if isinstance(kernel, (spy.ComputeKernel, BoundDispatch)):
                originals.append((owner, name, kernel))
                setattr(owner, name, _TimedKernel(kernel, timers, f"{label}.{name.lstrip('_')}"))
    try:
        yield
    finally:
        for owner, name, kernel in originals:
            setattr(owner, name, kernel)


def _sims(app):
    return [("live", app.sim)] + (
        [("benchmark", app.benchmark.sim)] if app.benchmark is not None else []
    )


def _kernel_owners(app):
    return _sims(app) + ([("benchmark", app.benchmark)] if app.benchmark is not None else [])


def _measure_frames(app, timers, samples, *, detailed=False):
    frames = []
    dispatches = defaultdict(list)
    for _ in range(samples):
        # Capturing the baseline and restarting completed trials submit separate
        # work. Keep both outside the steady-state frame measurement.
        if app._prepare_benchmark():
            app.device.wait()
        ticks_before = {label: sim.tick for label, sim in _sims(app)}
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
            "ticks": {label: sim.tick - ticks_before[label] for label, sim in _sims(app)},
            "cpu_record_ms": (recorded - started) * 1000,
            "cpu_finish_ms": (finished - recorded) * 1000,
            "cpu_submit_ms": (submitted - finished) * 1000,
            "wait_ms": (waited - submitted) * 1000,
            "wall_ms": (waited - started) * 1000,
        }
        for record in timers.read():
            if record["name"] == "frame":
                frame["gpu_ms"] = record["gpu_ms"]
            else:
                dispatches[record["name"]].append(record)
        frames.append(frame)
    return frames, dispatches


def profile_app(app, *, steps_per_frame=100, samples=10, warmup=200):
    """Advance an existing headless App; return JSON-compatible diagnostic data.

    This deliberately waits after each frame to separate CPU preparation and GPU
    execution. The result is synchronized throughput, not a claim about window
    presentation rate. Replays the input state and warmup before detailed measurement,
    so the app advances by warmup + samples * steps_per_frame ticks in total.
    Uses the default live-world view; periodic snapshots are suspended.
    """
    if steps_per_frame < 1 or samples < 1 or warmup < 0:
        raise ValueError("Need positive steps/samples and nonnegative warmup")
    if app.window is not None:
        raise ValueError("Profile a headless App")
    snapshot_store = app.snapshots
    app.snapshots = None
    try:
        return _profile_app(app, steps_per_frame, samples, warmup)
    finally:
        app.snapshots = snapshot_store


def _warmup(app, steps_per_frame, ticks):
    app.paused = False
    remaining = ticks
    while remaining:
        app.speed = min(steps_per_frame, remaining)
        app.frame()
        remaining -= app.speed
    app._prepare_benchmark()
    app.device.wait()
    app.speed = steps_per_frame


def _describe_worlds(app):
    scenes = []
    for label, sim in _sims(app):
        worlds = [sim.inspect_world(world) for world in range(sim.num_worlds)]
        scenes.append(
            {
                "label": label,
                "count": sim.num_worlds,
                "tick": sim.tick,
                "initial_creatures": sim.creatures_per_world,
                "population_at_start": [world["num_creatures"] for world in worlds],
                "food_at_start": [world["num_food"] for world in worlds],
                "creature_capacity": sim.creature_capacity,
                "initial_food": sim.initial_food,
                "food_capacity": sim.food_capacity,
                "settings": {field: getattr(sim, field) for field in SIM_SETTINGS},
                "world_counters": worlds,
            }
        )
    return scenes


def _profile_app(app, steps_per_frame, samples, warmup):
    input_tick = app.sim.tick
    app.selected_world = 0 if app.sim.num_worlds == 1 else None
    app.show_comparison = False
    app._prepare_benchmark()

    # Boundary-only timings are the throughput measurement; detailed queries
    # can serialize short passes and are used only for relative attribution.
    with TemporaryDirectory(prefix="worms-profile-") as directory:
        # Capture before warmup: checkpoint compression can leave the GPU idle
        # long enough to downclock, distorting the first measured frame.
        replay = app.snapshot(Path(directory) / "replay.npz")
        _warmup(app, steps_per_frame, warmup)
        start_tick = app.sim.tick
        scenes = _describe_worlds(app)
        reference_tick_at_start = (
            app.benchmark.reference_tick if app.benchmark is not None else None
        )
        # Keep warmed pipeline objects when restore replaces the simulation buffers.
        # Otherwise the detailed CPU timings include first-use pipeline creation.
        warmed_kernels = [
            {
                name: value.kernel if isinstance(value, BoundDispatch) else value
                for name, value in vars(owner).items()
                if isinstance(value, (spy.ComputeKernel, BoundDispatch))
            }
            for _, owner in _kernel_owners(app)
        ]
        frame_timers = _GpuTimers(app.device, 2)
        frames, _ = _measure_frames(app, frame_timers, samples)
        app.restore(replay)
        owners = _kernel_owners(app)
        for (_, owner), kernels in zip(owners, warmed_kernels, strict=True):
            for name, kernel in kernels.items():
                current = getattr(owner, name)
                if isinstance(current, BoundDispatch):
                    # Keep the restored buffers, only transplant the warmed pipeline.
                    current.replace_kernel(kernel)
                else:
                    setattr(owner, name, kernel)
        _warmup(app, steps_per_frame, warmup)
        kernel_count = sum(len(kernels) for kernels in warmed_kernels)
        pass_timers = _GpuTimers(app.device, 2 * (steps_per_frame * (kernel_count + 8) + 2))
        with _instrument(owners, pass_timers):
            detailed_frames, dispatches = _measure_frames(app, pass_timers, samples, detailed=True)

    medians = {
        name: median(frame[name] for frame in frames) for name in frames[0] if name != "ticks"
    }
    throughput_ticks, detailed_ticks = (
        {label: sum(frame["ticks"][label] for frame in run) for label in run[0]["ticks"]}
        for run in (frames, detailed_frames)
    )
    cpu_ms = sum(medians[name] for name in ("cpu_record_ms", "cpu_finish_ms", "cpu_submit_ms"))
    gpu_ms = medians["gpu_ms"]
    limiting_side = (
        "gpu" if gpu_ms > 1.25 * cpu_ms else "cpu" if cpu_ms > 1.25 * gpu_ms else "mixed"
    )
    wall_ms = sum(frame["wall_ms"] for frame in frames)
    ticks_per_second = throughput_ticks["live"] * 1000 / wall_ms
    detailed_gpu_ms = sum(frame["gpu_ms"] for frame in detailed_frames)
    passes = []
    for name, records in dispatches.items():
        values = [record["gpu_ms"] for record in records]
        cpu_values = [record["cpu_record_ms"] for record in records]
        passes.append(
            {
                "name": name,
                "kind": records[0]["kind"],
                "dispatches": len(values),
                "calls_per_live_tick": len(values) / detailed_ticks["live"],
                "mean_ms_per_dispatch": mean(values),
                "median_ms_per_dispatch": median(values),
                "p95_ms_per_dispatch": float(np.percentile(values, 95)),
                "max_ms_per_dispatch": max(values),
                "mean_cpu_record_ms": (
                    mean(cpu_values) if records[0]["kind"] == "dispatch" else None
                ),
                "median_cpu_record_ms": median(cpu_values)
                if records[0]["kind"] == "dispatch"
                else None,
                "p95_cpu_record_ms": float(np.percentile(cpu_values, 95))
                if records[0]["kind"] == "dispatch"
                else None,
                "ms_per_live_tick": sum(values) / detailed_ticks["live"],
                "percent_gpu_frame": sum(values) / detailed_gpu_ms * 100,
            }
        )
    passes.sort(key=lambda item: item["ms_per_live_tick"], reverse=True)
    return {
        "adapter": app.device.info.adapter_name,
        "api": app.device.info.api_name,
        "gpu_validation": {"d3d12_debug_layers": app.debug_gpu, "rhi": app.debug_gpu},
        "resolution": [app.texture.width, app.texture.height],
        "steps_per_frame": steps_per_frame,
        "samples": samples,
        "warmup_ticks": warmup,
        "input_tick": input_tick,
        "start_tick": start_tick,
        "end_tick": app.sim.tick,
        "throughput_ticks": throughput_ticks,
        "detailed_ticks": detailed_ticks,
        "benchmark": (
            {
                "warmup_ticks": app.benchmark.warmup_ticks,
                "waiting_at_start": reference_tick_at_start is None,
                "waiting_at_end": not app.benchmark.ready,
                "reference_tick_at_start": reference_tick_at_start,
                "reference_tick_at_end": app.benchmark.reference_tick,
            }
            if app.benchmark is not None
            else None
        ),
        "worlds": scenes,
        "median_frame": medians,
        "cpu_preparation_ms": cpu_ms,
        "limiting_side": limiting_side,
        "ticks_per_second": ticks_per_second,
        "realtime_multiple": ticks_per_second * TIME_STEP,
        "detailed_gpu_frame_mean_ms": detailed_gpu_ms / samples,
        "detailed_to_boundary_gpu_ratio": detailed_gpu_ms / sum(f["gpu_ms"] for f in frames),
        "unattributed_gpu_ms_per_live_tick": (
            detailed_gpu_ms
            - sum(record["gpu_ms"] for records in dispatches.values() for record in records)
        )
        / detailed_ticks["live"],
        "passes": passes,
        "frames": frames,
        "detailed_frames": detailed_frames,
        "method": (
            "D3D12 timestamp queries; synchronized headless update plus render; no native UI, "
            "vsync, presentation, checkpoint I/O, baseline capture or benchmark trial setup. "
            "Throughput uses two "
            "GPU timestamps per frame. Detailed timings replay the same input checkpoint and "
            "warmup with per-dispatch timestamps, including barriers/transitions within each "
            "interval. Checkpoint I/O precedes warmup in both runs. "
            "Per-dispatch CPU recording excludes timestamp writes. CPU preparation and GPU "
            "timeline are compared with a 25% margin; near ties are labelled mixed. "
            "Benchmark ticks count advances of the paired simulation, not individual worlds."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu-debug", action="store_true", help="Include D3D12 and SlangPy GPU validation cost"
    )
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--creatures", type=int, default=64)
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument(
        "--resume", type=Path, help="Measure a checkpoint without changing its file"
    )
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument(
        "--benchmark", action="store_true", help="Enable comparison after its baseline warmup"
    )
    comparison.add_argument(
        "--no-benchmark", action="store_true", help="Exclude saved comparison worlds"
    )
    parser.add_argument("--steps-per-frame", type=int, default=100)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("artifacts/profile.json"))
    args = parser.parse_args(argv)
    checkpoint = None
    if args.resume is not None:
        if args.output.resolve() == args.resume.resolve():
            parser.error("--output must differ from the source checkpoint")
        with args.resume.open("rb") as file:
            checkpoint = {
                "path": str(args.resume.resolve()),
                "sha256": hashlib.file_digest(file, "sha256").hexdigest(),
            }
    app = App(
        headless=True,
        num_worlds=args.worlds,
        creatures_per_world=args.creatures,
        creature_capacity=args.capacity,
        benchmark=args.benchmark,
        snapshot_dir=None,
        resume=args.resume,
        debug_gpu=args.gpu_debug,
    )
    try:
        if args.no_benchmark:
            app.benchmark = None
        if args.benchmark and app.benchmark is None:
            parser.error("The checkpoint contains no benchmark reference; omit --benchmark")
        report = profile_app(
            app,
            steps_per_frame=args.steps_per_frame,
            samples=args.samples,
            warmup=args.warmup,
        )
    finally:
        app.close()
    if checkpoint is not None:
        report["checkpoint"] = checkpoint
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"{report['adapter']} | {report['ticks_per_second']:.1f} ticks/s "
        f"({report['realtime_multiple']:.2f}x real time) | {report['limiting_side']} limited | "
        f"GPU validation {'on' if args.gpu_debug else 'off'}"
    )
    print(
        f"CPU preparation {report['cpu_preparation_ms']:.2f} ms/frame; "
        f"GPU {report['median_frame']['gpu_ms']:.2f} ms/frame; "
        f"ticks {report['start_tick']:,} to {report['end_tick']:,}"
    )
    print(
        "Stage                             GPU mean ms    p95 ms    ms/tick  GPU share  CPU us/call"
    )
    for stage in report["passes"]:
        cpu = stage["mean_cpu_record_ms"]
        print(
            f"{stage['name']:32} {stage['mean_ms_per_dispatch']:11.4f} "
            f"{stage['p95_ms_per_dispatch']:9.4f} {stage['ms_per_live_tick']:10.4f} "
            f"{stage['percent_gpu_frame']:8.1f}% "
            + (f"{cpu * 1000:12.1f}" if cpu is not None else "           -")
        )
    print(f"Detailed / boundary GPU time: {report['detailed_to_boundary_gpu_ratio']:.3f}x")
    print(args.output)


if __name__ == "__main__":
    main()
