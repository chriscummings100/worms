# GPU performance measurements

Use `worms.profile` to measure an isolated copy of a saved simulation with D3D12
hardware timestamps. The source checkpoint is unchanged. The report records its
path and SHA-256, the adapter, resolution, actual populations and food counts,
simulation settings, and measured tick range. Reports and generated checkpoints
belong in ignored `artifacts/`.

Normal runs now disable D3D12/RHI validation. Pass `--gpu-debug` to reproduce the
validation-enabled historical measurements below or diagnose GPU correctness.
New profile JSON includes `gpu_validation`; compare like modes. Shader-object
reuse and validation overhead are tracked in the optimization plan.

The [optimization plan and execution log](OPTIMIZATION_PLAN.md) tracks the agreed
changes and their validation. The 100,000-tick measurements below are the retained
pre-optimization baseline; milestone results are recorded separately.

## Profile a checkpoint

These commands compare the complete session with the live world alone, starting
from the same checkpoint:

```powershell
.\.conda\python.exe -m worms.profile --gpu-debug --resume artifacts/performance/checkpoint-100000.npz --steps-per-frame 100 --warmup 200 --samples 20 --output artifacts/performance/profile-session.json
.\.conda\python.exe -m worms.profile --gpu-debug --resume artifacts/performance/checkpoint-100000.npz --no-benchmark --steps-per-frame 100 --warmup 200 --samples 20 --output artifacts/performance/profile-live.json
```

`--resume PATH` preserves saved settings, brains, reference population, and the
unfinished benchmark. `--no-benchmark` excludes the saved comparison worlds from
the profiling copy. The profiler uses the standard live-world view, with creature
selection and the comparison graph cleared. Periodic snapshots are suspended.

For a fresh scene, omit `--resume`; `--worlds`, `--creatures` and `--capacity`
set its dimensions, and `--benchmark` adds the comparison pair. `--samples`
counts rendered frames, `--steps-per-frame` sets simulation ticks per frame,
and `--warmup` sets ticks before measurement. The default sample count is ten.
Use a stable checkpoint copy when the running app is rotating automatic saves.

## Method

The profiler saves its complete input state to a temporary checkpoint, then
advances the requested warmup ticks immediately before measuring. It measures
the update/render command stream with two GPU timestamps per frame, and CPU clocks
around command recording, finishing, submission and waiting.

It then restores that input state, repeats the same warmup, and measures the same
tick range with timestamps around every compute dispatch and the aggregate render
scope. Warmed `ComputeKernel` objects are retained across restoration, preventing
fresh kernel objects from adding first-use pipeline setup to the detailed CPU
measurements. Checkpoint I/O happens before warmup in both runs; no checkpoint
compression or restoration sits between warmup and timing. This avoids an idle
period that can let GPU clocks fall before the first measured frame.

A completed benchmark is prepared before collecting starting scene metadata,
so its reported counts describe the trial actually measured. The live simulation,
paired benchmark simulation, and benchmark lineage-count dispatch are all
instrumented automatically. Trial initialization occurs outside measured frames.

Throughput uses the frame-boundary measurement. Detailed queries can serialize
short passes, so their timings are used for attribution. The JSON report includes
the ratio of detailed to boundary-only GPU time to expose this effect. Dispatch
intervals include their resource transitions and barriers. The report also shows
GPU time outside the measured dispatch and render intervals.

Each stage reports GPU mean, median, 95th percentile and maximum milliseconds per
call, call count, calls per live tick, total milliseconds per live tick, and share
of the detailed GPU frame. Compute dispatches additionally report mean, median
and 95th percentile CPU recording time, excluding the timestamp writes themselves.
`render` is an aggregate scope rather than a compute dispatch. The JSON retains
individual frame timings and actual live and benchmark tick counts for both runs.
Benchmark ticks count advances of the paired simulation; they are not multiplied
by its two worlds. These counts also expose partial frames at trial boundaries.

Each measured frame waits for GPU completion. Reported ticks/s are therefore
**synchronized headless throughput**, including CPU preparation and GPU work.
Window presentation, vsync, native UI, checkpoint I/O and the occasional setup of
a new benchmark trial are excluded. Rendering the live view is included.
Headless App frames also wait for their submission to finish, bounding transient
D3D12 descriptors when running many ticks per render. A window can overlap CPU
recording with a previous GPU frame, so these numbers are not an exact window FPS
prediction. Simulation speed remains a plain number of ticks per rendered frame;
selecting 100× does not guarantee 6,000 ticks/s.

The CPU/GPU label compares median CPU preparation time with median GPU timeline
time, with a 25% margin for a clear winner. GPU wait alone is not used as evidence
of a GPU bottleneck. A GPU timeline bottleneck can be caused by serial kernels
using very little of the GPU's available parallelism.

## Checkpoint at 100,000 ticks — 25 October 2026

The reproducible baseline is `artifacts/performance/checkpoint-100000.npz`,
generated with seed 1 and default settings and advanced to 100,000 live ticks.
It contains 91 living creatures and 67 food pieces in the live world, plus a
fresh pair of benchmark worlds at trial tick zero. Each world reserves 1,024
creature slots and 8,192 food slots.

The measurement uses the NVIDIA GeForce RTX 3050 Ti Laptop GPU, driver 596.08,
SlangPy 0.43.1, D3D12 with debug validation enabled, and a 1280 × 720 headless
render. Both workloads use 100 ticks per frame, 200 warmup ticks, and twenty
measured frames, covering live ticks 100,200 through 102,200. Three sequential
repeats are used for each workload.

At the start of measurement, the live world has **91 creatures and 65 food**.
In the complete session, each benchmark world has **128 creatures and 148 food**,
at trial tick 200. Both measurement windows advance the live simulation by
2,000 ticks; the benchmark pair also advances by 2,000 ticks when enabled.
All six reports use the same source checkpoint, SHA-256
`a87b98d2df0a8a59af9b96b544454f082fb15c120ca53556baffc0679d42aefc`.

### Throughput and CPU/GPU limit

Entries are the median of three repeats, with the minimum–maximum in parentheses.
CPU and GPU frame columns summarize each repeat’s median frame costs; throughput
uses total elapsed time across all measured frames.

| Workload | Live ticks/s | Simulated time / wall time | CPU preparation ms/frame | GPU ms/frame | Limit |
|---|---:|---:|---:|---:|---|
| Live world | 760.7 (747.7–775.3) | 12.68 (12.46–12.92)× | 40.75 (38.58–42.42) | 89.30 (88.81–89.30) | GPU |
| Live + two benchmark worlds | 332.5 (332.4–332.7) | 5.54 (5.54–5.54)× | 74.25 (73.82–74.25) | 223.83 (223.51–224.03) | GPU |

GPU timeline time is about 2.2 times CPU preparation for the live world and three
times CPU preparation for the complete session. The benchmark pair lowers live
throughput from roughly 761 to 332 ticks/s. These are measured rates with the
existing serial simulation passes; this profiling change implements no simulation
optimization.

Detailed timestamps increase total GPU timeline time by **9.9–10.4%** for the
live-only workload and **8.15–8.18%** for the complete session. These ratios compare
total detailed and boundary-only GPU time across the twenty frames. The throughput
table uses the boundary-only run, so this detailed instrumentation cost is excluded.

### Every dispatch: live world

The following tables use repeat 2 of each workload. GPU times are milliseconds.
Calls and accumulated cost are normalized per live tick. The two `build_grid`
calls per tick are combined in their row; food growth and food-grid rebuilding
occur once every forty ticks. Rendering occurs once every hundred ticks.

| Stage | Calls/tick | Mean ms/call | P95 ms/call | ms/live tick | GPU share |
|---|---:|---:|---:|---:|---:|
| `live.build_grid` | 2 | 0.196100 | 0.197632 | 0.392199 | 39.87% |
| `live.resolve_lifecycle` | 1 | 0.340666 | 0.344064 | 0.340666 | 34.63% |
| `live.capture_vision` | 1 | 0.080068 | 0.084992 | 0.080068 | 8.14% |
| `live.evaluate_brains` | 1 | 0.050401 | 0.059392 | 0.050401 | 5.12% |
| `live.build_food_grid` | 0.025 | 0.929772 | 0.935014 | 0.023244 | 2.36% |
| `live.reset_newborn_history` | 1 | 0.014726 | 0.015360 | 0.014726 | 1.50% |
| `live.clear_grid` | 2 | 0.007330 | 0.008192 | 0.014660 | 1.49% |
| `live.collect_food` | 1 | 0.010382 | 0.011264 | 0.010382 | 1.06% |
| `live.step` | 1 | 0.007885 | 0.009216 | 0.007885 | 0.80% |
| `live.apply_food` | 1 | 0.007006 | 0.008192 | 0.007006 | 0.71% |
| `render` | 0.01 | 0.111667 | 0.112640 | 0.001117 | 0.11% |
| `live.grow_food` | 0.025 | 0.013926 | 0.022630 | 0.000348 | 0.04% |
| `live.clear_food_grid` | 0.025 | 0.007373 | 0.008192 | 0.000184 | 0.02% |

Time outside these dispatch/render scopes is 0.040805 ms/live tick (about 4.1%
of the detailed GPU frame). Mean CPU recording cost per compute dispatch ranges
from 21.3 to 30.6 microseconds. The JSON additionally retains median and P95 CPU
recording costs, plus median and maximum GPU dispatch times.

### Every dispatch: live world plus benchmark pair

Each `benchmark.*` dispatch processes **both benchmark worlds together**.
Its cost and call count must not be doubled. Shares below refer to the whole
session’s detailed GPU frame. The benchmark does not reach its trial boundary
in this window, so no lineage-count dispatch occurs.

| Stage | Calls/tick | Mean ms/call | P95 ms/call | ms/live tick | GPU share |
|---|---:|---:|---:|---:|---:|
| `live.build_grid` | 2 | 0.259256 | 0.324608 | 0.518512 | 21.41% |
| `benchmark.build_grid` | 2 | 0.258272 | 0.321536 | 0.516543 | 21.33% |
| `benchmark.resolve_lifecycle` | 1 | 0.361115 | 0.363520 | 0.361115 | 14.91% |
| `live.resolve_lifecycle` | 1 | 0.340938 | 0.344064 | 0.340938 | 14.08% |
| `benchmark.capture_vision` | 1 | 0.129644 | 0.138240 | 0.129644 | 5.35% |
| `live.capture_vision` | 1 | 0.103449 | 0.108544 | 0.103449 | 4.27% |
| `benchmark.evaluate_brains` | 1 | 0.099858 | 0.101376 | 0.099858 | 4.12% |
| `live.evaluate_brains` | 1 | 0.091764 | 0.093184 | 0.091764 | 3.79% |
| `benchmark.reset_newborn_history` | 1 | 0.028462 | 0.029696 | 0.028462 | 1.18% |
| `benchmark.build_food_grid` | 0.025 | 0.937595 | 0.941056 | 0.023440 | 0.97% |
| `live.build_food_grid` | 0.025 | 0.932332 | 0.940595 | 0.023308 | 0.96% |
| `benchmark.clear_grid` | 2 | 0.008421 | 0.009216 | 0.016842 | 0.70% |
| `live.reset_newborn_history` | 1 | 0.015639 | 0.016384 | 0.015639 | 0.65% |
| `live.clear_grid` | 2 | 0.006747 | 0.007168 | 0.013494 | 0.56% |
| `benchmark.collect_food` | 1 | 0.012996 | 0.013312 | 0.012996 | 0.54% |
| `live.collect_food` | 1 | 0.010447 | 0.011264 | 0.010447 | 0.43% |
| `live.step` | 1 | 0.009243 | 0.010240 | 0.009243 | 0.38% |
| `benchmark.step` | 1 | 0.008073 | 0.009216 | 0.008073 | 0.33% |
| `live.apply_food` | 1 | 0.007719 | 0.008192 | 0.007719 | 0.32% |
| `benchmark.apply_food` | 1 | 0.006574 | 0.007168 | 0.006574 | 0.27% |
| `render` | 0.01 | 0.116890 | 0.119808 | 0.001169 | 0.05% |
| `benchmark.grow_food` | 0.025 | 0.015852 | 0.029696 | 0.000396 | 0.02% |
| `live.grow_food` | 0.025 | 0.013230 | 0.021606 | 0.000331 | 0.01% |
| `benchmark.clear_food_grid` | 0.025 | 0.008479 | 0.009216 | 0.000212 | 0.01% |
| `live.clear_food_grid` | 0.025 | 0.008131 | 0.008755 | 0.000203 | 0.01% |

Time outside these dispatch/render scopes is 0.081649 ms/live tick (about 3.4%
of the detailed GPU frame). Mean CPU recording cost per compute dispatch ranges
from 11.0 to 28.0 microseconds. The render scope costs approximately 0.112 ms per
rendered frame for the live-only workload and 0.117 ms for the complete session.

### GPU utilization check

A separate device-wide telemetry run used the same 100,000-tick checkpoint,
200 warmup ticks, 100 ticks per rendered frame and 150 measured frames per
workload. Each workload covered live ticks 100,200 through 115,200. The live world
ended with 92 creatures and 66 food in both runs. No other Python simulation was
running, and a five-second idle baseline reported 0% GPU busy time.

`nvidia-smi` was sampled every 200 ms in a separate process. The first two seconds
and last second of each phase were excluded from telemetry summaries, retaining
83 readings over the live run and 207 over the complete session. The driver's
underlying sampling windows can overlap; these are correlated readings, not
independent trials. Values below are means over each retained interval.

| Workload | GPU busy time | Reported memory-access busy time | Power | Graphics clock | Device memory used |
|---|---:|---:|---:|---:|---:|
| Idle | 0% | 0% | 4.3 W | 210 MHz | 0 MiB |
| Live world | **68.3%** | 0% | 16.7 W | 1,723 MHz | 213 MiB |
| Live + two benchmark worlds | **74.3%** | 1.3% | 17.8 W | 1,717 MHz | 377 MiB |

The GPU has 4,096 MiB of device memory. During both workloads it remained in P0,
with memory clocks at 5,501 MHz, temperatures of 59–60°C, and no active clock-event
reasons reported. No throttling was reported during these samples. The power
limit query is unavailable on this laptop, so power is not expressed as a fraction
of its configured limit.

The synchronized runs achieved 758.5 and 333.2 live ticks/s. GPU timestamp spans
occupied approximately 67.6% and 74.6% of total phase elapsed time, consistent with
the driver's busy-time readings; CPU preparation and submission leave gaps
between GPU batches. These timestamp spans include barriers and stalls, and do
not measure instruction throughput. A windowed run can overlap CPU preparation
with earlier GPU work, so its busy-time percentage may differ.

**Busy time is not the fraction of GPU compute capacity in use.** NVIDIA defines
`utilization.gpu` as the percentage of a sampling interval during which one or
more kernels execute; `utilization.memory` similarly describes memory-access
activity, not the fraction of peak bandwidth achieved. A reported 0% memory
activity is a coarse utilization reading, not zero memory traffic.
See the [NVIDIA counter definitions](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

The grid and lifecycle passes use one thread per world: one thread in the live
dispatch, or two threads in the paired benchmark dispatch. Since these passes
account for roughly three quarters of GPU time, the code is substantially
underusing GPU parallelism even while the GPU is busy. The busy-time readings
support the earlier GPU timeline bottleneck finding; they do not establish high
SM occupancy or compute saturation.

Nsight Systems 2023.1.2 is installed and recognizes this GPU's SM activity,
occupancy and instruction-issue metrics. Its environment check reports no
administrator privileges. A one-second hardware-counter probe stalled during
event processing for over 130 seconds without producing a report and was stopped;
it emitted no definitive permission error. **Hardware SM occupancy remains
unmeasured.** A working Nsight hardware-counter capture would quantify how much
SM capacity is active and how effectively it issues instructions.

The raw readings, phase boundaries and frame timings are saved in
`artifacts/performance/utilization-100k.json`; the local reproduction script is
`artifacts/performance/sample_utilization.py`. To watch the same device counters
while running a checkpoint workload in another terminal:

```powershell
nvidia-smi --id=0 --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,pstate,power.draw,temperature.gpu,clocks.gr,clocks.mem,clocks_event_reasons.active --format=csv --loop-ms=200
```

### Optimization priorities

Creature-grid construction and lifecycle resolution account for **74.5%** of
the live-only detailed GPU timeline and **71.7%** of the complete session.
[`build_grid`](../src/worms/shaders/contact.slang) and
[`resolve_lifecycle`](../src/worms/shaders/lifecycle.slang) each launch one
thread per world and scan the reserved creature slots serially. Grid construction
runs twice per tick; lifecycle scans once for costs/deaths and again for births,
with a free-slot search for each eligible parent.

The first optimization targets are the serial creature-grid builds and the
per-creature cost/death work inside lifecycle. Parallelizing independent work
while preserving contact handling and birth allocation is likely to yield more
than optimizing rendering or neural evaluation. The measured shares identify
where to focus; the speedup still needs to be established by implementing and
remeasuring a change.

Raw reports are `artifacts/performance/profile-100k-live-{1,2,3}.json` and
`artifacts/performance/profile-100k-comparison-{1,2,3}.json`. The tables above use
the `-2.json` reports for dispatch detail. Preserve this checkpoint when comparing
future changes so population, food availability and benchmark phase remain fixed.

To generate another default checkpoint at the same tick with the normal app:

```powershell
.\.conda\python.exe -m worms --headless --frames 1000 --speed 100 --snapshot-dir artifacts/profile-checkpoints --output artifacts/profile-checkpoints/world.png
```

The final automatic file ends in `tick-000000100000.npz`. Pass its full filename
to `worms.profile --resume` using the commands above. The isolated snapshot
directory keeps this generated run separate from an interactive session’s saves.

## P1: parallel atomic grids — 25 October 2026

The first completed milestone in the [optimization plan](OPTIMIZATION_PLAN.md)
uses one thread per creature/food slot to insert into cell lists with atomic head
exchange. A 64-lane reduction computes each world's maximum live radius during
grid clearing. Food grids now rebuild every simulation tick. The second creature
grid build remains until the planned detection/resolution ordering change.
Saved buffer layouts and current simulation behavior are unchanged.

The following three-repeat measurements use exactly the baseline checkpoint,
starting world metadata, 200-tick warmup, 100 ticks/render and 20 measured frames.
Both timestamp modes cover ticks 100,200 through 102,200. Entries give medians,
with repeat ranges in parentheses. Throughput excludes detailed timestamp overhead.

| Workload | Live ticks/s | Change | CPU preparation ms/frame | GPU ms/frame | Limit |
|---|---:|---:|---:|---:|---|
| Live world | **999.4** (998.9–1,010.0) | **+31.4%** | 46.39 (46.36–46.67) | 50.10 (50.10–50.15) | Mixed |
| Live + benchmark pair | **484.4** (483.7–485.7) | **+45.7%** | 86.74 (85.81–87.02) | 116.66 (116.61–116.69) | GPU |

Repeat 2 provides the following detailed live-only comparison. Grid totals include
clearing, so the radius reduction and food's new rebuild frequency are accounted for.

| Work per live tick | Baseline ms | P1 ms |
|---|---:|---:|
| Two creature-grid clear/build pairs | 0.4069 | 0.0347 |
| Food-grid clear/build, amortized over every tick | 0.0234 | 0.0153 |
| Lifecycle resolution | 0.3407 | 0.3414 |
| Vision capture | 0.0801 | 0.0781 |

Creature insertion itself falls from 0.1961 to 0.0081 ms per dispatch. Food insertion
falls from 0.9298 to 0.0080 ms per dispatch; rebuilding it every tick is still cheaper
in aggregate than rebuilding the serial version every forty ticks. Atomic lists
have unspecified order, but food arbitration and vision ties remain order-independent.

CPU recording now matters more: rebuilding food every tick records extra dispatches,
and live CPU preparation is close to GPU timeline time. Lifecycle is now **56.0%**
of detailed live-only GPU time, or **50.9%** across both dispatch chains in the
complete session. The next work is sparse action resolution and parallel routine
lifecycle, followed by unified world dispatches. The roadmap includes removing the
second creature build by completing detection/vision before structural resolution.

Detailed queries add 17.8–19.1% GPU time live-only and 18.4–18.5% combined. These
larger ratios reinforce using boundary-only timings for throughput, and detailed
timings for attribution. Device utilization/SM occupancy was not remeasured for P1.

Validation: 74 relevant GPU tests passed, including crowded multi-workgroup grids,
world isolation, radius reset, death/reuse, contacts, vision, lifecycle, checkpoint
continuations, benchmarks and profiling. All 56 CPU tests and Ruff passed. Immediate
checkpoint restores remain byte-exact; advanced replays compare grid membership
without depending on atomic insertion order, and compare other buffers exactly.

Raw reports are `artifacts/performance/profile-p1-live-{1,2,3}.json` and
`artifacts/performance/profile-p1-comparison-{1,2,3}.json`. All six retain the original
checkpoint SHA-256. The source file was not changed.

## P2/P3/P3b/P6: sparse resolution, lifecycle, bindings and vision — 25 October 2026

The new tick contract captures vision and detects contacts from one post-movement
scene, then settles food, costs, deaths and splits. Each grid is built once per
tick. Food detection uses creature × local-cell workers, a deterministic per-food
winner reduction and one bounded consumption action per food. Parallel lifecycle
settles routine costs/deaths; serial resolution handles eligible parents and free
slots, and only actual newborns have their networks/history initialized.

Seventeen hot kernels retain SlangPy root shader objects, fixed buffer bindings,
pipelines and scalar cursors. Only changed scalar values are rewritten. GPU tests
verify reuse across unsubmitted encoders and rebinding after checkpoint restore.
A shared `ParameterBlock` worked in the API probe but did not further reduce CPU
cost relative to retained roots on this installed runtime; consolidation remains
an option with unified world state. See the [binding investigation](OPTIMIZATION_PLAN.md#p3b--reusable-shader-objects-and-parameter-blocks).

### Complete simulation throughput

Same checkpoint SHA, capacities, resolution, 100 ticks/render, 200 warmup ticks
and 20 measured frames as P0/P1; three sequential repeats per row. Values are
medians of the three reports. Ranges show their throughput spread.

| Implementation / validation | Workload | Live ticks/s (range) | CPU preparation ms/frame | GPU ms/frame | Limiting side |
|---|---|---:|---:|---:|---|
| P0 / enabled | Live | 760.7 (747.7–775.3) | 40.75 | 89.30 | GPU |
| P0 / enabled | Live + pair | 332.5 (332.4–332.7) | 74.25 | 223.83 | GPU |
| P1 / enabled | Live | 999.4 | 46.39 | 50.10 | Mixed |
| P1 / enabled | Live + pair | 484.4 | 86.74 | 116.66 | GPU |
| P2/P3/P3b/P6 / enabled | Live | **1,873.5 (1,702.2–2,129.3)** | 31.21 | 18.28 | CPU |
| P2/P3/P3b/P6 / enabled | Live + pair | **894.2 (869.2–911.1)** | 64.60 | 40.69 | CPU |
| P2/P3/P3b/P6 / disabled | Live | **3,610.6 (3,603.3–3,766.0)** | 10.83 | 15.28 | GPU |
| P2/P3/P3b/P6 / disabled | Live + pair | **1,661.3 (1,634.8–1,716.6)** | 21.35 | 37.13 | GPU |

Keeping validation enabled, throughput is **2.46× / 2.69×** the P0 baseline and
**1.87× / 1.85×** P1 for live / combined workloads. Normal production runs disable
both D3D12 debug layers and RHI validation; `--gpu-debug` enables both. Do not
attribute the additional validation-off gain to the algorithm changes alone.
The current production mode is GPU-timeline limited in all six reports, although
CPU preparation remains material and these synchronized runs do not overlap it
with GPU execution. Production throughput is approximately 60.2× / 27.7× real
time at the fixed 60 Hz simulation rate. This does not measure hardware occupancy.

Validation particularly increases native encoder finishing/submission: median
finish/submit is 13.96/3.14 ms for live and 27.43/6.67 ms combined with validation,
versus 2.40/0.089 ms and 4.60/0.095 ms in production. Shader objects remove repeated
Python resource binding; the installed SlangPy/RHI still constructs native binding
data per dispatch. The 1,500-dispatch shared-block probe showed no extra win from
a common parameter block or a single compute pass on this runtime.

### Attribution and compatibility

In production repeat 2, brain evaluation is now the largest named GPU stage:
0.0641 ms/live tick (23.8% of detailed time), followed by vision at 0.0463 ms
(17.2%). With benchmarks, both brain calls total 0.1926 ms/live tick (30.8%) and
vision totals 0.0981 ms (15.7%). The live world still has a separate dispatch chain
from the paired benchmark worlds. P4 combines these; compaction and indirect work
remain later milestones. Rendering is about 0.12 ms per frame.

Per-dispatch timestamps inflate GPU time substantially now that many passes are
short: **1.78–2.87×** live / **1.75–1.97×** combined with validation, and
**1.69–1.73×** / **1.62–1.63×** in production. Use these detailed runs to locate
costs, not to reconstruct throughput or predict additive savings. Throughput
above uses the separate run with only two boundary timestamps per frame. The
validation-enabled live spread is retained rather than hidden; GPU clocks were
not locked, and this run does not establish the cause of that variation.

All reports advance live tick 100,200 through 102,200. Starting live counts are
91 creatures / 65 food; paired worlds start at local tick 200 with 128 creatures
each and 151 / 148 food. Each report confirms one creature-grid and one food-grid
build per active simulation tick. The source checkpoint hash remains unchanged.
The post-movement observation order deliberately changes behavior: consumed food
can remain in that tick's captured vision, while births and new food first appear
on the next tick. At measurement time, snapshot format and benchmark protocol were version 3; versions
1 and 2 still load, retaining live evolution and the frozen reference while
restarting old-protocol benchmark trials/graphs. These performance comparisons
are not claims of identical long-run biological trajectories. These timings predate
protocol 4's change to two new offspring per split, which initializes both networks
and histories. That later reproduction change has not been reprofiled here.

Full correctness regression: **212 tests passed** (156 GPU, 56 CPU), with GPU
validation explicitly enabled; Ruff check and format checks passed. Coverage
includes stale identities, consumed food, crowded cells, seams, deterministic
food/birth priority, simultaneous deaths and slot reuse, mutation, history,
selection/rendering, snapshots and benchmark protocol migration.

Raw reports: `artifacts/performance/profile-p236-{debug,production}-{live,comparison}-{1,2,3}.json`.
Aggregate report: `artifacts/performance/profile-p236-summary.json`. Reproduce with
the commands above and matching output names, adding `--gpu-debug` for the debug
rows and omitting it for production. All GPU workloads were run sequentially.

### Isolated vision comparison

P6 uses one worker per creature × segment. A conservative cell-circle/wedge test
skips irrelevant linked-list traversal; nearest-object selection and deterministic
ties remain unchanged. An independent brute-force oracle covers crowded worlds,
toroidal seams and exact FOV/segment/range boundaries. Direct old/new capture
comparisons also matched every slot and frame header byte-for-byte.

| Scene | Previous ms/capture | Segment workers ms/capture | Speedup |
|---|---:|---:|---:|
| 100k checkpoint: 91 creatures, 67 food | 0.1642 | 0.0933 | 1.76× |
| 256 creatures, 1,024 food in a 48 m square | 1.2786 | 0.4154 | 3.08× |
| 1,024 creatures, 2,048 food in a 48 m square | 2.7695 | 1.2376 | 2.24× |

Five alternating batches of 50 captures per variant, validation enabled, identical
immutable inputs and retained bindings. Two boundary queries time each batch,
with no source/kernel swapping in the live simulation. This isolates capture;
its speedup is not an end-to-end simulation multiplier. Script/report:
`artifacts/performance/benchmark_vision_p6.py` and `vision-p6-comparison.json`.

### Production GPU busy-time check

A separate production-mode telemetry run used 500 synchronized frames per
workload, 100 ticks/frame and 200 warmup ticks. Both copies advanced live ticks
100,200–150,200 and ended with the same 32 creatures / 135 food. This longer run
has changing populations, so its throughput is separate from the short profiles
above. Device-wide `nvidia-smi` was sampled every 200 ms; the first two and last
one seconds were excluded, leaving 49 live and 134 combined samples. Idle was 0%.

| Workload | Mean GPU busy time | Memory-access busy time | Mean power | Graphics clock | Device memory |
|---|---:|---:|---:|---:|---:|
| Live | 52.8% | 1.9% | 17.7 W | 1,728 MHz | 202 MiB |
| Live + pair | 60.3% | 5.9% | 20.7 W | 1,725 MHz | 368 MiB |

GPU timestamp spans covered 52.2% / 61.6% of phase wall time, consistent with the
busy readings. Rates were 3,858.7 / 1,653.3 ticks/s across this longer trajectory.
Both workloads stayed in P0 with no active clock-event reasons, at mean 55.4°C /
59.3°C. Busy time measures activity, not SM occupancy or peak compute/bandwidth;
hardware occupancy remains unmeasured. The GPU still waits between synchronized
batches while the CPU prepares commands, so P8 overlap remains useful even when
GPU work takes longer than CPU preparation in the fixed-checkpoint profiles.
The source checkpoint is unchanged. Script/report:
`artifacts/performance/sample_utilization_p236.py` and
`utilization-p236-production.json`.

### Lifecycle stress check

Validation-enabled lifecycle-only timings, three warmups plus twenty independently
restored samples per scene:

| Scene at capacity 1,024 | Births/sample | Median total GPU ms | Resolve births ms | Initialize newborns ms |
|---|---:|---:|---:|---:|
| 91 living, routine | 0 | 0.0696 | 0.0092 | 0.0113 |
| 512 living, matched control | 0 | 0.0788 | 0.0102 | 0.0133 |
| 512 living, all eligible | 512 | 4.8092 | 0.8090 | 3.9404 |

Every heavy sample produced exactly 512 births, 1,024 living creatures and zero
deaths. Nonzero markers verified complete clearing of every newborn's 256-frame
history and activations; no-birth controls retained inactive markers. The heavy
initializer writes 42 MiB of history plus networks, while routine ticks skip it.
State restoration and readback are excluded; timestamps and validation contribute
to the measured cost. This is a stress diagnostic, not a whole-simulation rate or
a comparison against the previous implementation. Coalescing birth-history writes
is a possible later improvement if frequent mass births become significant.
Artifacts: `artifacts/performance/measure_lifecycle_p3.py` and
`lifecycle-p3-validation.json`.

## Historical measurements — 24 October 2026

The following results use the NVIDIA GeForce RTX 3050 Ti Laptop GPU, D3D12,
SlangPy 0.43.1, debug validation enabled, and a 1280 × 720 headless render.
These measurements start with fresh populations and the earlier benchmark
protocol. They predate replaying identical state for detailed profiling and
describe their specific scenes rather than the current checkpoint workload.
The initial population is 64 creatures in a 512 m square world, with 2,048 food
pieces and 8,192 reserved food slots, unless a row states otherwise.

### Baseline before the food-grid cache

The baseline is commit `b969d57`, before increasing the default creature capacity
and adding the comparison system. The 1,024-capacity row uses the same baseline
code with an explicit capacity override.

| Creature capacity | Ticks/render | CPU preparation/frame | GPU/frame | Ticks/s | Simulated time / wall time |
|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 1.21 ms | 1.58 ms | 294.0 | 4.90× |
| 256 | 10 | 4.49 ms | 14.74 ms | 504.8 | 8.41× |
| 256 | 100 | 44.99 ms | 147.13 ms | 511.8 | 8.53× |
| 1,024 | 100 | 41.90 ms | 195.59 ms | 418.9 | 6.98× |

At 256 capacity and 100 ticks/render, the serial food-grid rebuild took
**1.030 ms/tick, 65.1%** of the detailed GPU timeline. Vision capture took
0.189 ms/tick (12.0%), the two creature-grid builds together 0.109 ms/tick (6.9%),
and lifecycle resolution 0.100 ms/tick (6.3%). Brain evaluation was 3.0% and
newborn history cleanup only 0.6%. Rendering was about 0.10 ms per frame.

At 1,024 capacity the food-grid rebuild still cost 1.030 ms/tick (49.8%), while
creature-grid builds rose to 0.380 ms/tick (18.4%) and lifecycle to 0.306 ms/tick
(14.8%). This is a **GPU timeline bottleneck**, primarily serial scans, rather
than Python dispatch preparation or drawing.

Food positions remain fixed until spawning reuses a slot. The new cache rebuilds
the food grid after initialization and on food-spawn ticks. Eaten entries remain
in the lists and are ignored using their zero energy until the next rebuild.
At the default 1.5 food/s and 60 ticks/s, rebuilds occur once every 40 ticks.

### With the cache and 1,024 creature slots

| Workload | Initial creatures | CPU preparation/frame | GPU/frame | Live ticks/s | Simulated time / wall time |
|---|---:|---:|---:|---:|---:|
| Live world | 64 | 36.02 ms | 97.89 ms | 741.6 | 12.36× |
| Live plus two comparison worlds | 64 live; 128 in each trial | 68.55 ms | 245.08 ms | 315.0 | 5.25× |
| Full live world | 1,024 | 38.40 ms | 165.43 ms | 487.6 | 8.13× |

All three rows use 100 ticks/render and ten samples. The comparison pair advances
alongside every live tick; reported ticks/s count live progression, not the sum
of world updates. Both comparison worlds reserve 1,024 slots and start with 64
current and 64 reference creatures. These are short profiling runs, not evidence
of evolutionary improvement.

For the 64-creature live world, caching raised throughput from **418.9 to 741.6
ticks/s**, about **77%**, at the same 1,024 capacity. The food-grid contribution
fell from 49.8% to 2.4% of the detailed GPU timeline. Creature-grid construction
is now 36.1%, lifecycle resolution 30.3%, and vision capture 17.7%. The original
serial-per-world grid/lifecycle design is now the main performance limit.

With all 1,024 creature slots initially occupied, lifecycle resolution rises to
48.1%, creature grids use 23.6%, vision 14.0%, and neural evaluation 7.1%. Reserving
slots and filling them have different costs. With the comparison worlds enabled,
the two simulations' creature grids together use 39.1%, lifecycle 25.9%, and
vision 17.8% of the GPU timeline.

The next optimization candidates are parallelizing per-creature depletion and
death checks while retaining a serial birth allocator, and replacing the serial
creature-grid builds if profiling continues to justify it. These require more
careful synchronization and are not part of this change. Rendering and newborn
history cleanup are not the significant bottlenecks in these measurements.

The raw reports for this run are `artifacts/profile-baseline-{1,10,100}.json`,
`artifacts/profile-capacity-1024.json`, and
`artifacts/profile-optimized-{live,comparison,full}.json`. A fresh-scene invocation
of the current profiler measures current code rather than reproducing these
historical implementations.

### Complete session check

A subsequent 20,000-tick run with the comparison graph enabled completed two
10,000-tick trials and two automatic session snapshots in 68.05 seconds
(293.9 live ticks/s including checkpoint I/O). The live population grew from 64
to 81, with 17 births. The full checkpoint files were about 17.9 MB each and
preserved the initial reference, comparison history and next trial. Native window
checks exercised the graph alongside neuron activations, Snapshot now, Step and
Restore latest snapshot. The restored session was paused at its saved tick.

The two trial scores were 49.6% and 50.2%; this short validation establishes that
the system runs, not that evolution has already produced a meaningful gain.
