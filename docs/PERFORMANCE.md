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
changes and their validation. The 13-million-tick snapshot is the current
repeatable reference. Historical 100,000-tick measurements below use an older
benchmark protocol that the current application rejects; they are retained as
historical context, not directly comparable regression measurements.

## All-half inference follow-up — 25 October 2026

Inference now stays FP16 through input normalization, both dot products, `tanh`
and output saturation. Distance and size convert from FP32 at the observation
boundary. Initialization and mutation are outside inference and retain FP32.
Offline compilation of `evaluate_brains` reports **zero half-to-float conversions,
zero FP32 arithmetic operations and zero FP32 intrinsic calls**. The only two
static narrowing sites read observation distance/size. This checks DXIL types,
not the driver's internal implementation of transcendental operations.

A controlled one-device test interleaves three precision variants on identical,
unchanging 13M observations. It uses 20 warmup and 20 measured rounds, rotating
and reversing case order; each scope contains 128 inference calls and two GPU
timestamps. The FP32 variant widens the same quantized weights into FP32 buffers.
There are 345 valid live observations and 788 across the comparison worlds.

| Inference variant | Live GPU µs/call | Pair GPU µs/call |
|---|---:|---:|
| Previous half dots, FP32 normalization/activations | 47.14 | 97.43 |
| **All-half inference, adopted** | **48.98** | **100.87** |
| All-FP32 | 86.60 | 112.80 |

Removing the remaining FP32 operations is 3.9% / 3.5% slower in this controlled
test, while all-half remains 43.4% / 10.6% faster than all-FP32. Consistent types
do not by themselves guarantee a further speedup. These frozen-kernel timings
are not whole-simulation throughput or comparable to per-dispatch instrumented
times below. Separate-process trials had substantial live-kernel variance and
are retained as diagnostics, not used for the comparison above.

GPU results against the original saved FP32 activations have maximum normalized
control error 0.001025 live / 0.001182 pair, with means 0.000256 / 0.000241.
The source checkpoint is unchanged. Validation: 97 CPU tests and 21 distinct
targeted GPU cases passed, covering all neural tests, chunk equivalence, trial
handoffs and checkpoint continuation; Ruff lint/format checks pass. The NumPy
input reference permits one FP16 step near 1 because the compiler substitutes
an FP16 reciprocal for division by 24 and may contract operations.

Artifacts: `artifacts/performance/2026-10-25-all-half/` contains the before source,
`brain-all-half.dxil.txt`, `ir-verification.json`, `interleaved.json`,
`compare_interleaved.py`, accuracy/timing reports and their reproduction scripts.
The earlier window, sustained and Nsight results below describe the preceding
half-dot/FP32-activation implementation; they have not been relabeled as new runs.

## Bounded chunks and FP16 inference — 25 October 2026

Windowed and headless refreshes now submit simulation chunks of at most 25 ticks
on the same graphics queue, recording subsequent chunks while earlier work runs.
At high speed, two submission slots bound transient command resources; the last
chunk shares the refresh's draw. Settings, trial handoffs and snapshots stay at
refresh boundaries. The window still uses SlangPy's presentation and normal
population polling. Speed choices now include 200, 500 and 1,000 ticks/refresh.

Neural weights, stored activations, local activations and dot products use native
FP16. Vision normalization, nonlinearities and mutation arithmetic use FP32;
geometry, retained vision, energy and tick/identity counters keep their existing
precision. Checkpoint format 5 converts legacy network buffers, including frozen
references, without modifying the source archive. Quantization changes future
trajectories; it is not bit-identical continuation of the old FP32 policy.

### Actual window throughput

Two independent restores per case, 2,000 warmup ticks and 10,000 measured ticks
per restore, reversing workload order on the second repeat. These runs call the
real `window_app.run_frame()` with vsync, UI and normal population polling, then
drain the device before ending the elapsed interval. Autosaves and GPU validation
are off. The table reports the median of two rates, not a sustained-hour forecast.

| Workload | Previous FP32, 100 steps/refresh | New FP16 + chunks, 100 steps/refresh | New FP16 + chunks, 1,000 steps/refresh |
|---|---:|---:|---:|
| Live + comparison pair | 1,710 steps/s | **2,128 steps/s (+24.4%)** | **2,529 steps/s** |
| Live only | 1,880 steps/s | 1,766 steps/s (−6.0%) | **6,204 steps/s** |

The 1,000-step live-only rate is about 103× real time; comparisons reach 42×.
The old monolithic 1,000-step case failed a D3D12 descriptor allocation assertion,
so there is no valid old rate at that setting. Chunking bounds the transient
allocations and lets the larger refresh batches run.

Live-only window performance at 100 steps did not improve. Final headless runs
at that setting reached 4,944 steps/s live-only and 2,101 with comparisons, so
the small-window-batch result has substantial cost outside simulation work.
Presentation and polling remain in the window measurements; their individual
contributions have not been isolated. Do not describe the result as a universal
100-step window speedup. Increasing steps per refresh amortizes that overhead.

With identical chunk scheduling, FP32 headless runs reached 4,133 / 1,998
steps/s live / combined, versus the final 4,944 / 2,101. Desktop CPU variation
and diverging quantized trajectories limit attribution; the fixed-input accuracy
check and boundary/dispatch shader profiles below provide separate evidence.
Raw runs: `fp32-sync-window.json`, `final-chunk-window.json`,
`final-chunk-headless.json`, `fp32-chunk-headless.json`; medians and ranges are
in `comparison-summary.json` beside `measure_run.py`.

### Matching Nsight window captures

Both captures use the final FP16 implementation, 100 steps/refresh, 100 measured
frames, the same checkpoint/warmup and normal UI/polling. Only chunk scheduling
changes. NVTX `worms.capture` includes the final device drain. DX12 workload
intervals are clipped to that range and unioned, avoiding overlap double counts.

| Metric | Single command buffer | 25-step chunks |
|---|---:|---:|
| Instrumented completed steps/s | 1,784 | 2,135 (+19.7%) |
| Capture elapsed | 5.605 s | 4.684 s |
| Unioned GPU workload duration | 3.869 s | 3.863 s |
| GPU timeline coverage | 69.0% | 82.5% |
| Uncovered timeline | 1.737 s | 0.820 s (−52.8%) |
| Recording CPU time / frame | 14.71 ms | 14.87 ms |
| Frame p95 | 103.8 ms | 63.6 ms |

This directly supports overlap as the improvement: GPU work and recording time
are almost unchanged, while exposed gaps shrink. Timeline coverage is not SM
occupancy. The chunked `update` NVTX scope is longer because it now contains
intermediate submission and bounded GPU waits; it is no longer pure preparation.
No separate `finish` scope was recorded in these window captures.

Open `worms-window-100steps-chunked.nsys-rep` in Nsight Systems for the new path,
or `worms-window-100steps-synchronized.nsys-rep` for the matched control. Both
export successfully to SQLite and pass integrity checks. Analysis and script:
`nsight-window-analysis.json`, `analyze-window-captures.py` in the artifact folder
above. These instrumented rates are separate from uninstrumented window rates.

### Sustained 1,000-step validation

A final 60.41-second headless run advanced **151,000 live and paired ticks** at
**2,499.8 steps/s**, including one completed trial and restart. Ten-second windows
ranged from 2,449.8 to 2,546.5 steps/s. The live population ended at 363, and the
two trial populations at 294/293. GPU validation and autosaves were disabled;
small counter reads, timestamp queries and trial setup remain in elapsed time.

The median frame took 397.82 ms. Its summed GPU scopes took 387.04 ms, with a
first-to-last scope span of 387.77 ms: **only 0.74 ms median between scopes**.
Across the run, scoped GPU time occupied **99.75% of that span** and 97.46% of
measured frame wall time. Scope gaps include unmeasured boundary operations, not
only idle time; this establishes a nearly continuous command timeline inside
each 1,000-step refresh, not hardware occupancy. CPU record/finish/submit medians
were 149.81/63.18/5.37 ms, overlapping GPU execution rather than adding to it.

Each frame used 40 simulation chunks plus one render scope, with the last chunk
and render sharing a submission. No descriptor-allocation failures occurred.
Warmup also used 1,000-step frames, so the first measured trial starts at tick
1,000 instead of tick 1,500 in the short tests that warm up in 100-step frames.
The source checkpoint hash remains unchanged. Report:
`final-sustained-1000.json` in the artifact folder above.

```powershell
.\.conda\python.exe -m worms.throughput --resume checkpoints/13-million-ticks/snapshot.npz --seconds 60 --interval 10 --steps-per-frame 1000 --warmup 2000 --output artifacts/throughput-1000.json
```

Final validation: **289 tests passed (97 CPU, 192 GPU)**, including native half
accuracy, mutation/inheritance, legacy migrations, checkpoint continuation,
chunk ordering, trial boundaries, rendering and timing accounting. The 15 brain
tests ran separately; the remaining 274 passed in 319.75 seconds. Ruff lint and
formatting checks passed. `after-source.zip` and `manifest.json` preserve sources
and hashes; `final-regression.log` records the 274-test run.

### Precision experiments

Same saved 13M reference, 2,000 warmup ticks, ten measured 100-step frames with
comparisons, production validation disabled. Per-dispatch GPU timings come from
a separate replay and include timestamp overhead; whole-frame GPU timings use
two boundary queries. This retains single-buffer submission to isolate shader
changes from chunk scheduling.

| Inference | Brain live / pair, µs per dispatch | Boundary GPU ms / 100 steps |
|---|---:|---:|
| Original FP32 | 105.0 / 99.7 | 39.71 |
| FP16 storage, FP32 dot products (rejected) | 145.3 / 142.0 | 47.70 |
| Paired FP16 loads, FP32 dot products (rejected) | 202.6 / 199.8 | 59.33 |
| **Native FP16 dot products (adopted)** | **97.8 / 92.2** | **38.21** |

Scalar half storage alone adds a conversion on every weight read. Offline DXIL
inspection confirms the adopted implementation uses native half multiply/add
and avoids those per-weight conversions. It reduces brain dispatch time by
about 7–8%, and complete-session GPU time by 3.8% in this short replay. This is
not tensor-core execution.

An independent GPU evaluation of the snapshot's 345 valid live observations and
788 trial observations found maximum normalized control differences of 0.000993
and 0.001213 from FP32, with means near 0.0002. These are fixed-input checks,
not a claim about long-term evolutionary equivalence. Network allocations for
the live world, pair and frozen reference fall from 18,604,032 to 9,302,016 bytes
(17.74 to 8.87 MiB). Vision history remains much larger.

Artifacts are under ignored `artifacts/performance/2026-10-25-pipeline-half/`:
`profile-{fp32,half,packed-half-select,compute-half}.json`,
`accuracy-compute-half.json`, corresponding scripts, compiler dumps and isolated
source copies. Rejected variants are retained as experiments only.

## Before chunking: 13M reference, sustained 100-step capture — 25 October 2026

Reference: `checkpoints/13-million-ticks/snapshot.npz`, SHA-256
`378bca806376c52c1905872668148a47d3f357381089a2df512978a0cb49d8b1`.
The snapshot is unchanged. Simulation source is revision `d1352e6`; this capture
adds a sustained measurement utility without changing simulation or rendering.
Python 3.12.15, SlangPy 0.43.1, NVIDIA driver 596.08, RTX 3050 Ti Laptop GPU,
D3D12, production validation disabled, 1280 × 720 when drawing.

Each workload independently restores the same saved state, warms up 2,000 ticks,
then runs for 180 seconds with 100 live ticks per frame and ten-second summaries.
Runs are sequential: complete session, live-only, then live-only without drawing.
All start at live tick 13,002,000 with 347 living creatures; the comparison pair
has just restarted and contains 128 creatures per world. Capacities are 1,024
creatures and 8,192 food slots per world. All saved settings/seeds are retained.

| Workload | Live steps/s | Simulated / wall time | CPU preparation median ms/frame | GPU median ms/frame | Frame p50 / p95 ms | Live ticks measured |
|---|---:|---:|---:|---:|---:|---:|
| Live + comparison pair | **1,527.8** | **25.46×** | 16.85 | 41.32 | 58.99 / 93.40 | 275,100 |
| Live only | **2,704.5** | **45.08×** | 12.07 | 19.66 | 32.09 / 53.13 | 486,900 |
| Live only, drawing off | **2,455.8** | **40.93×** | 20.67 | 19.50 | 40.50 / 54.19 | 442,100 |

These are end-to-end synchronized headless rates, including query reads,
periodic small counter readbacks, benchmark completion and trial setup. Loading,
warmup, autosaves, final JSON serialization, native UI, vsync and presentation are
excluded. The GPU timestamps cover the main frame command buffer; separately
submitted trial setup is included in wall time. CPU preparation includes any
blocking readbacks in `App.update`, and is not pure CPU execution time. One
submission is retired before recording the next frame, as in `App.frame`.

### What changed during the three minutes

| Workload | First ~30s steps/s | Last ~30s steps/s | Ten-second range | End live population |
|---|---:|---:|---:|---:|
| Live + comparison pair | 1,530 | 1,490 | 1,099–1,777 | 352 |
| Live only | 3,660 | 2,184 | 1,236–3,686 | 366 |
| Live only, drawing off | 2,303 | 3,465 | 1,369–3,740 | 357 |

The dominant variation is in CPU preparation, not a consistent increase in GPU
cost. Live-only GPU window medians stay approximately 19.5–20.0 ms, while CPU
preparation grows from about 7 ms to 24 ms; drawing-off later recovers to 6.7 ms
CPU preparation and 3,740 steps/s. These sequential observations do **not** show
that disabling rendering makes the simulation slower. They also do not isolate
thermal effects or establish a clean regression: the desktop was active and a
process sample recorded other CPU consumers, including Windows compatibility
telemetry and the editor. The reports retain all stalls, including maximum
frames of 1.86 s combined and 0.91 s live-only. The specific cause of those stalls
is not established by wall-clock timing alone.

The full session completes two measured trial restarts. Their whole-frame costs
are 65.3 and 58.8 ms, not the long-tail stalls above. Comparison worlds end with
355 and 350 creatures. GPU temperatures span 65–74°C combined, 60–65°C live-only,
and 60–65°C with drawing off. Device-wide busy-time means are 63.6%, 52.6% and
47.4%, respectively. Graphics clocks occasionally fall during idle periods;
recorded active clock-event masks are zero or the GPU-idle bit. No thermal
clock-event flag is observed. Busy time is not hardware occupancy.

At these measured averages, one million live steps takes about **10.9 minutes
with comparisons** or **6.2 minutes live-only**. The speed setting of 100 means
steps per rendered frame; 100× real-time simulation at 60 ticks/s would require
6,000 live steps/s. The current headless loop leaves CPU preparation and GPU
execution mostly sequential, so both remain useful optimization targets.

### Every dispatch in the current full session

A separate short profile restores the same reference, warms up 2,000 ticks and
measures 30 frames / 3,000 live steps, then replays for detailed attribution.
It achieves 1,631.6 live steps/s with comparisons and 3,246.2 live-only in the
boundary-only captures. Full-session CPU preparation is 21.07 ms/frame and GPU
time is 39.86 ms/frame. Short and sustained rates use different measurement
windows and must not be substituted for one another.

Below, GPU timings are microseconds per dispatch. Each comparison dispatch
processes both comparison worlds together. CPU entries are mean recording times
for live / paired calls, excluding the profiler's timestamp writes.

| Dispatch | Live GPU mean / p95 µs | Pair GPU mean / p95 µs | CPU live / pair µs | Combined detailed GPU share |
|---|---:|---:|---:|---:|
| evaluate_brains | 105.1 / 106.5 | 99.9 / 101.4 | 3.9 / 3.9 | 31.5% |
| capture_vision | 53.6 / 55.3 | 52.1 / 55.3 | 4.4 / 4.2 | 16.3% |
| detect_food_contacts | 11.8 / 12.3 | 12.7 / 14.3 | 5.4 / 5.3 | 3.8% |
| clear_grid | 9.6 / 10.2 | 10.9 / 11.3 | 4.8 / 4.7 | 3.1% |
| build_food_grid | 9.4 / 10.2 | 9.0 / 10.2 | 3.9 / 3.8 | 2.8% |
| step | 9.6 / 10.2 | 8.5 / 10.2 | 3.2 / 2.9 | 2.8% |
| settle_lifecycle | 9.4 / 10.2 | 8.5 / 10.2 | 4.9 / 4.9 | 2.7% |
| initialize_newborns | 8.6 / 9.2 | 8.4 / 9.2 | 4.6 / 4.5 | 2.6% |
| build_grid | 8.8 / 9.2 | 7.8 / 9.2 | 6.5 / 4.5 | 2.6% |
| apply_food | 8.1 / 9.2 | 7.3 / 8.2 | 5.9 / 5.8 | 2.4% |
| resolve_food_contacts | 7.1 / 9.2 | 8.3 / 10.2 | 4.2 / 4.2 | 2.4% |
| clear_food_grid | 6.8 / 7.2 | 8.4 / 9.2 | 4.0 / 3.9 | 2.3% |
| clear_food_contacts | 6.8 / 7.2 | 7.9 / 8.2 | 4.8 / 4.8 | 2.3% |
| clear_lifecycle | 6.8 / 7.2 | 7.9 / 8.2 | 4.2 / 4.2 | 2.3% |
| resolve_lifecycle | 6.4 / 7.2 | 7.5 / 8.2 | 4.7 / 4.8 | 2.1% |
| grow_food (every six ticks) | 13.1 / 24.6 | 16.6 / 29.7 | 4.1 / 3.9 | 0.8% |

Rendering is once per 100 ticks: **111.5 µs mean / 112.6 µs p95 per frame**,
0.17% of detailed GPU time. Time outside named dispatch/render scopes is 17.2%.
Each full-session frame records about 3,033 compute dispatches. Brain evaluation
and vision together are 47.8% of detailed GPU time; reducing small-dispatch and
CPU preparation overhead remains relevant alongside optimizing those kernels.

Detailed queries inflate the GPU frame from about 39.9 to 65.1 ms (**1.63×**).
These instrumented shares identify hotspots; they cannot be added up to predict
uninstrumented savings. The detailed capture covers the beginning of the run,
not every ten-second window. Late-run per-dispatch behavior remains unmeasured.

### Reproduction and files

These commands select the original submission pattern. Reproducing the exact
FP32 implementation also requires the archived source; current shaders use
FP16 neural inference. Without `--synchronized`, the current utility measures
bounded chunks instead.

```powershell
.\.conda\python.exe -m worms.throughput --resume checkpoints/13-million-ticks/snapshot.npz --seconds 180 --interval 10 --steps-per-frame 100 --warmup 2000 --synchronized --output artifacts/throughput-comparison.json
.\.conda\python.exe -m worms.throughput --resume checkpoints/13-million-ticks/snapshot.npz --seconds 180 --synchronized --no-benchmark --output artifacts/throughput-live.json
.\.conda\python.exe -m worms.throughput --resume checkpoints/13-million-ticks/snapshot.npz --seconds 180 --synchronized --no-benchmark --no-render --output artifacts/throughput-live-no-render.json
```

Raw data is in ignored `artifacts/performance/2026-10-25-13m/`: `sustained-*.json`,
`profile-{comparison,live}.json`, `summary.json`, `telemetry.json`, and
`host-process-sample.json`. The complete chart is `throughput.png` / `throughput.svg`.
`capture.py` reproduces sequential runs and device telemetry; `analyze.py` rebuilds
the summary and figure. Plotting dependencies are isolated under ignored
`artifacts/performance/plot-deps/` and are not application dependencies.

### Nsight Systems viewer capture

Open `artifacts/performance/2026-10-25-13m/worms-13m-100steps-batch.nsys-rep`
in Nsight Systems using File → Open. Installed collector/viewer: 2023.1.2.43.
The capture includes restore and 2,000 warmup ticks, then 30 labeled frames of
100 steps each with the comparison pair enabled. Find the `worms.capture` NVTX
range (1.804 seconds) for the warmed section. Nested ranges label update, draw,
command-buffer finish, submission and GPU wait.

The native report was successfully exported to SQLite and verified: 850 D3D12
CPU API records, 89 GPU `ExecuteCommandLists` batch records, and 211 NVTX ranges,
including all 30 labeled frames. `nsight-session-metadata.json` preserves the
snapshot hash and exact tick counts; `nsight-validation.json` records the exported
table counts. `nsight_capture.py` is the reproduction target. This trace is
instrumented and is separate from the sustained throughput measurements.

The successful command uses `--trace=dx12,dx12-annotations,nvtx`,
`--dx12-gpu-workload=batch`, `--dx12-wait-calls=true`, `--sample=none`,
`--cpuctxsw=none`, `--gpu-metrics-device=none`, and `--python-sampling=false`,
with natural application exit. The first attempt with WDDM and NVTX capture
gating crashed the collector (`0xc0000005`) and produced an invalid partial
stream. The successful retry omits both. A second valid report requested
`--dx12-gpu-workload=individual`, but its export still contains 89 batch records,
not individual GPU dispatch events. Do not describe that report as per-kernel
GPU timing. Use the named-dispatch timestamp JSON above for those measurements.
CPU sampling, scheduling/context switches and hardware occupancy are absent.

Validation: 66 CPU tests passed; the two new sustained-profiler GPU tests and
six existing profile GPU tests passed (8 total, 45.55 seconds). These verify
trial caps/restarts, actual benchmark tick counts, render-off warmup, suspended
automatic saves and unchanged source checkpoints. Ruff check/format and Git
whitespace checks passed. No simulation behavior was changed.

## Profile a checkpoint

These commands compare the complete session with the live world alone, starting
from the same checkpoint:

```powershell
.\.conda\python.exe -m worms.profile --resume checkpoints/13-million-ticks/snapshot.npz --steps-per-frame 100 --warmup 2000 --samples 30 --output artifacts/performance/profile-session.json
.\.conda\python.exe -m worms.profile --resume checkpoints/13-million-ticks/snapshot.npz --no-benchmark --steps-per-frame 100 --warmup 2000 --samples 30 --output artifacts/performance/profile-live.json
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
