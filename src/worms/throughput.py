"""Measure sustained headless throughput from a saved session.

Run ``python -m worms.throughput --resume checkpoint.npz --seconds 180``.
The default is 100 simulation steps per frame in bounded chunks of 25, with the
saved benchmark intact. Use --synchronized for the original single-buffer loop.
These are evolution-throughput measurements, not window presentation rates.
"""

import argparse
import hashlib
import json
import math
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from worms.app import App
from worms.profile import _describe_worlds, _GpuTimers, _sims
from worms.sim import TIME_STEP
from worms.submissions import chunk_timings


class _FrameTimers(_GpuTimers):
    """Read every chunk once, distinguishing GPU work from gaps between chunks."""

    def read(self):
        values = self.pool.get_results(0, self.index)
        milliseconds = 1000 / self.frequency
        scopes = [
            {
                "name": record["name"],
                "gpu_ms": (values[record["index"] + 1] - values[record["index"]]) * milliseconds,
            }
            for record in self.records
        ]
        gpu_ms = sum(scope["gpu_ms"] for scope in scopes)
        gpu_span_ms = (values[self.index - 1] - values[0]) * milliseconds
        return {
            "gpu_ms": gpu_ms,
            "gpu_span_ms": gpu_span_ms,
            "gpu_gap_ms": max(0.0, gpu_span_ms - gpu_ms),
            "gpu_scopes": scopes,
        }


def _state(app):
    benchmark = app.benchmark
    return {
        "live_tick": app.sim.tick,
        "benchmark": (
            {
                "trial": benchmark.trial,
                "tick": benchmark.sim.tick,
                "trial_ticks": benchmark.trial_ticks,
                "ready": benchmark.ready,
                "pending": benchmark.pending,
                "reference_tick": benchmark.reference_tick,
                "history_count": len(benchmark.history),
            }
            if benchmark is not None
            else None
        ),
    }


def _tick_advances(before, after):
    """Count actual updates, including capped trials and their tick resets."""
    ticks = {"live": after["live_tick"] - before["live_tick"]}
    previous, current = before["benchmark"], after["benchmark"]
    if current is not None:
        # A waiting benchmark has no completed work, even if its buffers exist.
        previous_total = (
            previous["trial"] * previous["trial_ticks"] + previous["tick"]
            if previous is not None and previous["ready"]
            else 0
        )
        current_total = (
            current["trial"] * current["trial_ticks"] + current["tick"] if current["ready"] else 0
        )
        ticks["benchmark"] = current_total - previous_total
    return ticks


def _world_counters(app):
    return [
        {
            "label": label,
            "tick": sim.tick,
            "world_counters": [sim.inspect_world(world) for world in range(sim.num_worlds)],
        }
        for label, sim in _sims(app)
    ]


def _warmup(app, steps_per_frame, ticks, render, *, synchronized=False, chunk_size=25):
    remaining = ticks
    while remaining:
        app.speed = min(steps_per_frame, remaining)
        encoder = app.device.create_command_encoder()
        if synchronized:
            app.update(encoder)
        else:
            app.update_for_frame(encoder, chunk_size=chunk_size)
        if render:
            app.render(encoder)
        submission = app.device.submit_command_buffer(encoder.finish())
        app.device.wait_for_submit(submission)
        remaining -= app.speed
    app.speed = steps_per_frame
    app.device.wait()


def _measure_frame(app, timers, render, *, synchronized=False, chunk_size=25):
    started = perf_counter()
    before = _state(app)
    timers.reset()
    encoder = app.device.create_command_encoder()
    # Both update paths include benchmark preparation. Completed-trial readbacks
    # and new-trial submissions remain inside elapsed timing.
    intermediate = chunk_timings()
    if synchronized:
        with timers.measure(encoder, "frame"):
            app.update(encoder)
            if render:
                app.render(encoder)
        intermediate["chunks"] = 1
    else:
        intermediate = app.update_for_frame(
            encoder,
            chunk_size=chunk_size,
            chunk_scope=lambda chunk_encoder: timers.measure(chunk_encoder, "update_chunk"),
        )
        if render:
            with timers.measure(encoder, "render"):
                app.render(encoder)
    recorded = perf_counter()
    command_buffer = encoder.finish()
    finished = perf_counter()
    submission = app.device.submit_command_buffer(command_buffer)
    submitted = perf_counter()
    app.device.wait_for_submit(submission)
    waited = perf_counter()
    gpu = timers.read()
    after = _state(app)
    ended = perf_counter()
    completed = (
        app.benchmark.history[before["benchmark"]["history_count"] :]
        if app.benchmark is not None
        else []
    )
    return {
        "ticks": _tick_advances(before, after),
        "start_tick": before["live_tick"],
        "end_tick": after["live_tick"],
        "benchmark_before": before["benchmark"],
        "benchmark_after": after["benchmark"],
        "completed_trials": completed,
        "chunks": intermediate["chunks"],
        "submissions": intermediate["submissions"] + 1,
        "waits": intermediate["waits"] + 1,
        "cpu_record_ms": (recorded - started) * 1000
        - sum(intermediate[name] for name in ("cpu_finish_ms", "cpu_submit_ms", "wait_ms")),
        "cpu_finish_ms": (finished - recorded) * 1000 + intermediate["cpu_finish_ms"],
        "cpu_submit_ms": (submitted - finished) * 1000 + intermediate["cpu_submit_ms"],
        "wait_ms": (waited - submitted) * 1000 + intermediate["wait_ms"],
        "query_read_ms": (ended - waited) * 1000,
        "wall_ms": (ended - started) * 1000,
        **gpu,
    }


def _summarize(frames, elapsed_seconds):
    """Use observed elapsed time, including work between measured frames."""
    ticks = {
        label: sum(frame["ticks"].get(label, 0) for frame in frames) for label in frames[0]["ticks"]
    }
    medians = {
        name: median(frame[name] for frame in frames)
        for name in (
            "cpu_record_ms",
            "cpu_finish_ms",
            "cpu_submit_ms",
            "wait_ms",
            "query_read_ms",
            "wall_ms",
            "gpu_ms",
            "gpu_span_ms",
            "gpu_gap_ms",
        )
    }
    ticks_per_second = ticks["live"] / elapsed_seconds
    return {
        "elapsed_seconds": elapsed_seconds,
        "frames": len(frames),
        "start_tick": frames[0]["start_tick"],
        "end_tick": frames[-1]["end_tick"],
        "throughput_ticks": ticks,
        "ticks_per_second": ticks_per_second,
        "benchmark_ticks_per_second": ticks.get("benchmark", 0) / elapsed_seconds,
        "realtime_multiple": ticks_per_second * TIME_STEP,
        "frames_per_second": len(frames) / elapsed_seconds,
        "frame_p50_ms": medians["wall_ms"],
        "frame_p95_ms": float(np.percentile([frame["wall_ms"] for frame in frames], 95)),
        "frame_max_ms": max(frame["wall_ms"] for frame in frames),
        "median_frame": medians,
        "cpu_preparation_ms": median(
            sum(frame[name] for name in ("cpu_record_ms", "cpu_finish_ms", "cpu_submit_ms"))
            for frame in frames
        ),
    }


def _validate(seconds, interval, steps_per_frame, warmup, chunk_size=25):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("--seconds must be positive and finite")
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("--interval must be positive and finite")
    if not isinstance(steps_per_frame, int) or steps_per_frame < 1:
        raise ValueError("--steps-per-frame must be a positive integer")
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError("--warmup must be a nonnegative integer")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("--chunk-size must be a positive integer")


def profile_throughput(
    app,
    *,
    seconds=180,
    interval=10,
    steps_per_frame=100,
    warmup=2000,
    render=True,
    progress=None,
    synchronized=False,
    chunk_size=25,
):
    """Advance a headless App for at least seconds, retaining real trial transitions.

    Total/window rates include timestamp reads, periodic world-counter readbacks,
    benchmark completion/setup, and reporting between windows. Loading, warmup,
    automatic checkpoint I/O, final report serialization, UI and presentation are
    excluded. Chunked mode overlaps recording and GPU execution within a frame.
    GPU time sums chunk/render scopes; GPU span also includes intervening gaps.
    Separately submitted benchmark setup is included in elapsed wall time.
    """
    _validate(seconds, interval, steps_per_frame, warmup, chunk_size)
    if app.window is not None:
        raise ValueError("Profile a headless App")
    snapshot_store = app.snapshots
    app.snapshots = None
    try:
        return _profile_throughput(
            app,
            seconds,
            interval,
            steps_per_frame,
            warmup,
            render,
            progress,
            synchronized,
            chunk_size,
        )
    finally:
        app.snapshots = snapshot_store


def _profile_throughput(
    app, seconds, interval, steps_per_frame, warmup, render, progress, synchronized, chunk_size
):
    input_state = _state(app)
    app.paused = False
    app.selected_world = 0 if app.sim.num_worlds == 1 else None
    app.show_comparison = False
    _warmup(app, steps_per_frame, warmup, render, synchronized=synchronized, chunk_size=chunk_size)
    scope_count = 1 if synchronized else math.ceil(steps_per_frame / chunk_size) + int(render)
    timers = _FrameTimers(app.device, 2 * scope_count)
    start_state = _state(app)
    measured_started_at = datetime.now(UTC).isoformat()
    started = perf_counter()
    worlds = _describe_worlds(app)
    previous_worlds = _worlds_from_description(worlds)
    frames, windows = [], []
    window_start, window_first_frame = 0.0, 0
    while True:
        frame_start = perf_counter() - started
        frame = _measure_frame(
            app, timers, render, synchronized=synchronized, chunk_size=chunk_size
        )
        frame["start_seconds"] = frame_start
        frame["end_seconds"] = perf_counter() - started
        frames.append(frame)
        elapsed = frame["end_seconds"]
        if elapsed - window_start >= interval or elapsed >= seconds:
            current_worlds = _world_counters(app)
            elapsed = perf_counter() - started
            window = {
                **_summarize(frames[window_first_frame:], elapsed - window_start),
                "start_seconds": window_start,
                "end_seconds": elapsed,
                "worlds_at_start": previous_worlds,
                "worlds_at_end": current_worlds,
            }
            windows.append(window)
            previous_worlds = current_worlds
            window_start, window_first_frame = elapsed, len(frames)
            if progress is not None:
                progress(window)
        if elapsed >= seconds:
            break
    summary = _summarize(frames, elapsed)
    cpu_ms, gpu_ms = summary["cpu_preparation_ms"], summary["median_frame"]["gpu_ms"]
    return {
        **summary,
        "frame_count": len(frames),
        "adapter": app.device.info.adapter_name,
        "api": app.device.info.api_name,
        "gpu_validation": {"d3d12_debug_layers": app.debug_gpu, "rhi": app.debug_gpu},
        "resolution": [app.texture.width, app.texture.height],
        "steps_per_frame": steps_per_frame,
        "execution_mode": "synchronized" if synchronized else "chunked",
        "chunk_size": None if synchronized else chunk_size,
        "max_in_flight": 1 if synchronized else 2,
        "gpu_timestamps_per_frame": 2 * scope_count,
        "requested_seconds": seconds,
        "interval_seconds": interval,
        "warmup_ticks": warmup,
        "render": render,
        "time_step_seconds": TIME_STEP,
        "measured_started_at": measured_started_at,
        "input_state": input_state,
        "start_state": start_state,
        "end_state": _state(app),
        "worlds": worlds,
        "worlds_at_end": previous_worlds,
        "limiting_side": (
            "gpu" if gpu_ms > 1.25 * cpu_ms else "cpu" if cpu_ms > 1.25 * gpu_ms else "mixed"
        ),
        "windows": windows,
        "frames": frames,
        "method": (
            (
                "Synchronized headless frames, one submission and two GPU timestamps per frame. "
                if synchronized
                else "Chunked headless frames with at most two main submissions in flight. "
                "CPU recording overlaps GPU execution within each frame; every frame retires "
                "its final submission. Two GPU timestamps per update chunk and render scope. "
            )
            + "Whole/window elapsed rates include benchmark completion and setup, timestamp "
            "reads, world-counter readbacks and reporting between windows. GPU time is the "
            "sum of measured scopes; GPU span runs from the first scope's start to the last "
            "scope's end and includes gaps between them. Their difference is gpu_gap_ms; "
            "it includes unscoped boundary work as well as any queue starvation. "
            "In chunked mode, CPU phase totals overlap GPU time and must not be added to it. "
            "Separate benchmark "
            "setup submissions are covered by wall time. Excludes loading, initial pipeline "
            "warmup, checkpoint I/O, final report serialization, native UI, vsync and "
            "presentation. Any lazy compilation after warmup remains measured. Frame wall "
            "time includes timestamp reads but not interval counter reads. These are "
            "headless throughput rates, not window FPS. Benchmark ticks count paired "
            "simulation advances, not individual worlds; trial resets are accounted for. "
            "The final frame may overshoot the requested duration."
        ),
    }


def _worlds_from_description(worlds):
    return [{key: world[key] for key in ("label", "tick", "world_counters")} for world in worlds]


def _metadata():
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "git_revision": revision,
        "git_dirty": dirty,
        "python_version": platform.python_version(),
        "slangpy_version": version("slangpy"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--steps-per-frame", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument(
        "--synchronized", action="store_true", help="Record each frame in one command buffer"
    )
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/throughput.json"))
    args = parser.parse_args(argv)
    try:
        _validate(args.seconds, args.interval, args.steps_per_frame, args.warmup, args.chunk_size)
    except ValueError as error:
        parser.error(str(error))
    if args.output.resolve() == args.resume.resolve():
        parser.error("--output must differ from the source checkpoint")
    with args.resume.open("rb") as file:
        checkpoint = {
            "path": str(args.resume.resolve()),
            "sha256": hashlib.file_digest(file, "sha256").hexdigest(),
        }
    metadata = _metadata()

    def progress(window):
        print(
            f"{window['end_seconds']:6.1f}s | {window['ticks_per_second']:7.1f} ticks/s | "
            f"{window['realtime_multiple']:5.1f}x real time | "
            f"frame p50/p95 {window['frame_p50_ms']:.2f}/{window['frame_p95_ms']:.2f} ms | "
            f"CPU {window['cpu_preparation_ms']:.2f} / "
            f"GPU {window['median_frame']['gpu_ms']:.2f} ms "
            f"(span {window['median_frame']['gpu_span_ms']:.2f}) | "
            f"tick {window['end_tick']:,}",
            flush=True,
        )

    app = App(headless=True, resume=args.resume, snapshot_dir=None, debug_gpu=False)
    try:
        if args.no_benchmark:
            app.benchmark = None
        report = profile_throughput(
            app,
            seconds=args.seconds,
            interval=args.interval,
            steps_per_frame=args.steps_per_frame,
            warmup=args.warmup,
            render=not args.no_render,
            progress=progress,
            synchronized=args.synchronized,
            chunk_size=args.chunk_size,
        )
    finally:
        app.close()
    report.update(metadata)
    report["checkpoint"] = checkpoint
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Overall: {report['ticks_per_second']:.1f} ticks/s "
        f"({report['realtime_multiple']:.2f}x real time), "
        f"{report['throughput_ticks']['live']:,} live ticks in "
        f"{report['elapsed_seconds']:.2f}s. Saved {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
