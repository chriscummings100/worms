# Simulation optimization plan and execution log

Living document. Started 25 October 2026. Update this file as each milestone is
implemented, tested and measured; checkboxes describe completed work, not intent.
The baseline and profiling method are in [PERFORMANCE.md](PERFORMANCE.md).
Existing uncommitted application work and saved sessions must be preserved.

## Objective and constraints

Increase simulation ticks per second, especially with the live world and the two
benchmark worlds running together. Keep the application simple and D3D12-specific.
Normal simulation work stays on the GPU: the CPU records dispatches and submits
batches. Diagnostic reads, explicit outputs and restorable checkpoints are exceptions.
Speed remains a plain loop of simulation ticks per render, including 100 ticks.
Do not add configurability or an elaborate scheduler without measured need.

This work changes execution and storage, not the ecological rules or benchmark
score. Creature combat is still a future feature, not implicitly introduced by
building a general contact/action queue. Any intentional tick-order change is
documented and tested explicitly.

## Baseline and success criteria

**Current reference (25 October, sustained capture):** use the preserved
`checkpoints/13-million-ticks/snapshot.npz`, SHA-256
`378bca806376c52c1905872668148a47d3f357381089a2df512978a0cb49d8b1`,
with `worms.throughput` at 100 steps/frame and 180 seconds per workload.
See [the current measurements](PERFORMANCE.md) for the method and results.
The original baseline below predates changed reproduction and benchmark rules;
its checkpoint's benchmark protocol is no longer accepted by the current app.
Use matching checkpoints, rules and validation settings for new comparisons.

The new three-minute averages are 2,704.5 live steps/s and 1,527.8 with the
comparison pair. CPU preparation varies substantially while GPU cost stays near
19.7 / 41.3 ms per 100-step frame. A sequential drawing-off run averages 2,455.8
steps/s; differing CPU conditions prevent attributing that difference to drawing.
Detailed brain/vision shares are 31.5% / 16.3% for the complete session; drawing
is 0.11 ms per frame. These captures add evidence and repeatable measurement,
not another completed optimization milestone. Nsight batch timelines and the
full named-dispatch table are linked from PERFORMANCE.md.

**Implemented next:** 25-tick submissions overlap CPU recording and GPU work
inside each refresh. Native FP16 neural inference halves network allocations;
all inference math is now half, while initialization and mutations stay FP32.
In two short actual-window
repeats, comparisons improve from 1,710 to 2,128 steps/s at 100 steps/refresh.
At 1,000 steps/refresh, comparisons reach 2,529 and live-only 6,204 steps/s.
Live-only at 100 steps/refresh is 6% slower in the window measurements, so the
gain is not universal. Those window measurements precede the final all-half
normalization and activation change; see PERFORMANCE.md for its separate kernel
measurements. See PERFORMANCE.md for matched shader profiles, precision
bounds and method limits. Async inspector/counter readback remains future work.

### Historical optimization baseline

Use the unchanged `artifacts/performance/checkpoint-100000.npz`, SHA-256
`a87b98d2df0a8a59af9b96b544454f082fb15c120ca53556baffc0679d42aefc`.
It has 91 live creatures and 67 food pieces, plus a fresh benchmark pair. Each
world reserves 1,024 creature and 8,192 food slots. The device is the RTX 3050 Ti
Laptop GPU, driver 596.08, SlangPy 0.43.1, D3D12 with debug validation enabled.

Measure live-only and complete-session workloads at 1280 × 720, 100 ticks/render,
200 warmup ticks and 20 measured frames. Use `--gpu-debug` for comparison to the
validation-enabled baseline, and report normal production mode separately.
Use three sequential repeats for a
completed milestone. Never run GPU tests and profiling concurrently. Preserve
the checkpoint, report its hash, and retain raw reports in ignored `artifacts/`.

| Baseline workload | Live ticks/s, median of 3 | CPU preparation ms/frame | GPU ms/frame |
|---|---:|---:|---:|
| Live world | 760.7 | 40.75 | 89.30 |
| Live + benchmark pair | 332.5 | 74.25 | 223.83 |

The two creature-grid builds and lifecycle account for 74.5% of live-only GPU
time and 71.7% with benchmarks. They currently scan slots serially per world.
Device busy time was 68.3% and 74.3%; that is not SM occupancy or compute
saturation. Hardware SM occupancy remains unmeasured. GPU timestamps and CPU
recording time must both be tracked as the bottleneck moves.

Acceptance for each milestone:

- Relevant real GPU correctness tests and the CPU suite pass; Ruff is clean.
- Profiles use identical inputs and measurement modes. Record both dispatch
  attribution and total throughput; reducing one kernel is not sufficient.
- Replays preserve intended semantics. Atomic list order is unspecified, so
  compare memberships and application state rather than post-update link bytes.
- Checkpoints remain restorable; format changes require migration tests.
- UI selection/follow, vision history, benchmark isolation and collected-energy
  accounting remain valid.

## Decisions agreed during design

1. **Atomic grids.** Insert each object in its centre cell using atomic exchange
   of the head; store the returned head in its next link. No CAS retry loop is
   needed. Traversal begins only in a subsequent ordered dispatch, after all
   links have been written. Clear/reset and consumer dependencies are explicit.
2. **Food grid every tick first.** Once parallel, rebuilding is the simple
   default. Measure before adding dirty flags. A future GPU-generated indirect
   dispatch uses zero groups when unchanged and `ceil(active_count/group_size)`
   otherwise. Do not clear a cached grid when its build is skipped. Compaction,
   slot reuse or relocation invalidate old links.
3. **Parallel contact detection.** Workers span creature × neighboring cell,
   traverse that cell, perform exact overlap tests and emit actual contacts.
   Most creatures should emit nothing. Dense single cells may still be costly;
   subdivide their traversal only if profiles justify it.
4. **Atomic append queues.** Use contiguous per-world event regions with atomic
   counts, rather than linked event nodes. Worlds resolve independently. Explicit
   overflow handling is required; sparse typical scenes do not bound worst cases.
5. **Unique contacts.** Centre-cell insertion and unique wrapped stencil visits
   avoid repeated target sightings. Creature pairs additionally emit only for
   `A.guid < B.guid`. If objects later occupy several cells, emit only from the
   lowest shared cell inside the local search rectangle. Compute that ownership
   in local unwrapped coordinates before wrapping addresses at world seams.
6. **Resolution is a queue of actions.** Actions can invalidate later actions.
   Before applying each action, check participant identities, alive/available
   state and its prerequisites. Consumed food cannot be eaten again; stale
   references must not affect a reused slot. Serial work per world is acceptable
   for a small number of real actions. Routine arithmetic remains parallel.
7. **Splitting belongs in resolution.** Finish all contact detection and vision
   reads from the post-movement scene before structural resolution. Split
   positions and newborns enter detection/sensing on the next tick. No consumer
   may traverse the old grid after structural changes. This replaces the earlier
   proposal to repair links after splits, and allows one full creature build/tick.
8. **Scheduling is not priority.** Atomic append order must not decide contested
   food, capacity allocation or future combat. Use explicit seeded priorities
   and deterministic ties. Preserve current food and birth arbitration initially;
   migrate slot-dependent keys to stable identities together with compaction.
9. **One world allocation and dispatch chain.** Live and benchmark worlds should
   run as independent lanes of the same simulation, with per-world settings and
   clocks. The benchmark pair already shares dispatches; the live world does not.
   Combining them removes duplicate dispatch chains, not necessarily 3× GPU time.
10. **Dense hot data, stable cold state.** Compact active creature/food records.
    Keep stable handles to networks and full vision history rather than copying
    84 KiB of history per creature each tick. Selection follows GUID, not a slot.
11. **Wider vision.** Parallelize across creature × segment while avoiding sixteen
    copies of the complete neighborhood scan. Preserve nearest-object occlusion
    and full-resolution history; history reduction remains separate future work.
12. **Batching and overlap.** Submit chunks of simulation ticks while recording
    subsequent chunks, with a bounded number in flight. Keep one draw per refresh
    and a synchronized profiling mode for comparisons. Async diagnostics remain
    a separate improvement.
13. **Reusable SlangPy shader objects.** SlangPy supports building a `ShaderObject`
    once and reusing it for repeated dispatches. Put the simulation's persistent
    buffers and mostly constant settings in a shared `ParameterBlock` in the
    compute shaders, backed by a retained shader object, instead of rebuilding
    Python binding dictionaries and GPU parameter data for every dispatch. Keep
    changing tick/world state separate or GPU-resident. Recreate bindings when
    buffers are replaced during restore/reset; update settings only when changed.
    Verify object lifetimes and recorded-dispatch snapshot semantics before reuse.
    This should substantially reduce CPU preparation; measure rather than assume
    whether CPU recording remains the bottleneck after GPU improvements.

## Execution milestones

| ID | Work | Status | Depends on |
|---|---|---|---|
| P0 | Fixed-checkpoint baseline and GPU utilization | Complete | — |
| P1 | Parallel atomic creature/food grids | Complete | P0 |
| P2 | Snapshot detection and sparse action resolution; one creature build | Complete; tested and measured | P1 |
| P3 | Parallel lifecycle and sparse newborn initialization | Complete; tested and measured | P2 contract |
| P3b | Reusable shader objects / shared parameter block | Retained roots complete; shared block measured, deferred | P2/P3 bindings |
| P4 | Unified live/benchmark world dispatch chain | Planned | P2/P3 stable |
| P5 | Stable identities, compact hot arrays and GUID lookup | Planned | P2/P4 |
| P6 | Creature × segment vision | Complete; correctness and isolated GPU timings verified | P2; coordinate with P5 |
| P7 | GPU indirect dispatch where useful | Measure first | P5 |
| P8 | Bounded CPU/GPU overlap and async UI readback | In-refresh overlap implemented; async UI reads planned | No P4 dependency for in-refresh chunks |
| P9 | End-to-end measurement and documentation | At each milestone; final pass pending | All accepted changes |

### P1 — parallel atomic grids

Files: `shaders/contact.slang`, `shaders/vision.slang`, `sim.py`, grid/contact,
vision, profiler and checkpoint tests.

- [x] One insertion thread per creature slot; clear each link and food credit.
- [x] One insertion thread per food slot; omit empty entries.
- [x] Preserve maximum live radius for contact search, including oversized bodies
  and empty worlds. Reduce it in parallel without changing the saved World layout.
- [x] Rebuild food grid every tick. Keep its current placement before vision for
  this isolated change; contact currently searches the creature grid.
- [x] Retain the second creature build temporarily, until P2 changes the snapshot
  contract. P1 alone must preserve existing splitting and vision semantics.
- [x] Add crowded multi-world/sparse-slot grid membership coverage. Adapt replay
  tests to unspecified atomic insertion order while retaining exact immediate
  checkpoint restore checks.
- [x] Compile and run real GPU regression tests: 74 relevant GPU tests passed;
  all 56 CPU tests and Ruff passed.
- [x] Run three live and three combined fixed-checkpoint measurements.
- [x] Record performance and compatibility results below.

Important implementation detail: publishing the head before writing `next` is
safe only because no traversal runs concurrently with insertion. Atomics alone
do not provide a whole-dispatch fence. Keep the command encoder's resource
dependencies/barriers between clear, build and consumer passes.

### P2 — immutable detection scene and action resolution

Files: `sim.py`, `shaders/contact.slang`, lifecycle/vision integration,
`world_render.slang`, inspection, snapshots and affected GPU tests.

- [x] Establish a tick contract: move using the previous brain outputs; build
  grids once; capture vision and detect contacts from those positions; evaluate
  next brain outputs; resolve contact actions, costs, deaths and splits; advance
  the tick. Structural changes are not read through those grids again.
- [x] Capture observer pose/tick with the observation. Rendering may show the
  settled creature alongside its pre-resolution observation; keep the overlay
  anchored to the stored pose. Newborn history is empty until first sensing.
- [x] Define newborn first-movement output explicitly (zero until first policy
  evaluation is the simple default), and clear dead-creature display activations.
- [x] Keep food additions after current contact resolution, so a newly spawned
  piece cannot be collected in that tick. Under the new snapshot contract it
  becomes visible on the next tick too. Build the food grid at the next detection
  boundary, after any previous additions or compaction.
- [x] Generate contacts across creature × local cell, with radius-dependent
  reach and unique wrapped cells even for large test bodies. Test exact overlap
  before enqueueing. Start with food collection; combat remains unimplemented.
- [x] Use per-world contiguous action storage/counters, identities and deterministic
  priority. Dispatches define separate food and birth action types; independent
  worlds have separate queue regions.
- [x] Preserve seeded arbitration for food. Either resolve all contenders using
  that priority before consuming, or reduce the winning candidate per food on
  GPU and emit one consumption action. Do not let first atomic append win.
- [x] Bound food actions by food capacity using one winner reduction per food.
  A GPU overflow flag and full-capacity tests guard the queue; the mathematical
  bound prevents overflow without a per-tick CPU read. General creature-pair
  queues remain future work and will require their own explicit overflow policy.
- [x] Resolve actions serially per world, checking identities and prerequisites
  immediately before each application. Earlier actions may invalidate later ones.
  Separate explicit ordering classes where required: collection before energy
  costs, deaths before free-slot allocation, splitting after contact settlement.
- [x] Remove the second creature-grid build. Never add link repair merely to
  preserve same-tick sensing of structural changes: the agreed next-tick contract
  is simpler. Rebuild next tick before any contact/vision traversal.
- [x] Test contested food, consumed-food invalidation, stale reused-slot actions,
  deterministic priorities, world isolation, seams, capacity/overflow, and a split
  crossing a cell edge with next-tick visibility. Test one build/tick directly.

This is an intentional observation-order change: a creature may retain its
pre-resolution observation of food eaten that tick. Deaths/splits do not rewrite
already captured history. Document this change in README and snapshot metadata
and version the benchmark execution protocol. Restart unfinished old-protocol
trials and begin a separate/annotated graph series, retaining the frozen reference
and live population. Do not silently combine old and new scores. Do not compare
long-run biological trajectories as if the old and new protocols were identical.

### P3 — routine lifecycle in parallel, structural actions sparse

Files: `shaders/lifecycle.slang`, `sim.py`, snapshot buffer definitions and tests.

- [x] Compute movement/idle depletion and fractional remainders per creature.
  Preserve food-before-cost and the hard lifespan limit before reproduction.
- [x] Generate death/split eligibility after food settlement; cost arithmetic may
  be prepared earlier, but decisions must use credited energy so food can fund a
  split or prevent depletion death in the same tick. Queue only structural actions.
  Revalidate split energy
  and available capacity during resolution; reset parent age only on success.
- [x] Allocate free slots once per world; avoid scanning all slots for every
  birth. Preserve deterministic parent priority under capacity pressure.
- [x] Initialize newborn networks across child × parameter and reset history only
  for actual births, instead of scanning capacity × 256 frames every tick.
- [x] Preserve energy division, lineage/cohort, GUID/mutation streams, separation,
  opposing headings, and prohibition on newborn reproduction in its birth tick.
- [x] Test full/extinct worlds, simultaneous death/reuse, multiple competing
  births, mutation repeatability and the existing energy/age accounting suite.
- [x] Profile routine ticks separately from a deliberately birth-heavy scene.

Lifecycle stress evidence (validation enabled, 3 warmups then 20 restored samples
per scene): 91 live/no birth costs 0.070 ms, matched 512 live/no birth 0.079 ms,
and 512 simultaneous births 4.809 ms. The heavy case spends 0.809 ms resolving
births and 3.940 ms initializing 512 networks plus 42 MiB of history. Every sample
produced exactly 512 births and 1,024 living creatures; markers verified all
newborn history/activations were cleared and inactive slots were untouched in
no-birth controls. Restore/readback is outside timing; per-stage timestamps and
validation add overhead. This measures lifecycle-only stress, not ordinary tick
throughput. Artifacts: `artifacts/performance/measure_lifecycle_p3.py` and
`lifecycle-p3-validation.json`. If birth initialization later matters, consider
coalescing history writes across children/frames alongside P5/P7.

### P3b — reusable shader objects and parameter blocks

- [x] Verify the installed SlangPy shader-object, cursor and compute-pass APIs,
  including reuse across encoders and how changes affect already recorded work.
- [x] Bind stable resources once through retained root shader objects/pipelines,
  avoiding repeated Python dictionaries and reflection. Probe a shared parameter
  block; retain current per-kernel layouts because the probe found no added gain.
- [x] Separate dynamic tick/dispatch constants from persistent resources. Prefer
  GPU per-world state as P4 lands; never make a whole recorded batch use its final
  tick or final settings accidentally.
- [x] Preserve explicit UAV dependencies and transitions between producer and
  consumer dispatches. Cached bindings do not remove resource synchronization.
- [x] Rebuild/rebind cached objects on buffer replacement and preserve warmed
  pipeline profiling. Test reset, snapshot restore, settings changes and batched
  ticks against separate submissions.
- [x] Measure CPU preparation and total throughput on both fixed-checkpoint
  workloads. Record whether shader objects remove CPU binding as a limiting cost.

Implementation decision: retain one root shader object per hot kernel now, with
fixed resources/capacities and cached scalar cursors. Initializers/picking retain
their existing interfaces. This removes per-tick Python resource dictionaries and
repeated reflection/binding. Profiler restoration borrows warmed pipelines but
rebuilds roots against restored buffers. A probe verifies that subsequent scalar
and resource edits do not retroactively alter recorded dispatches on this D3D12
version, including commands recorded in different encoders before submission.

A shared `ParameterBlock` remains an option when P4 consolidates world state,
but is not needed to obtain the binding reuse already implemented. The installed
SlangPy/RHI version still constructs native binding data per dispatch. A 1,500-call,
three-kernel/12-buffer probe found no further CPU reduction from a shared block
versus retained roots. Keep current SRV/UAV layouts until consolidation earns its
cost; a single all-UAV block could introduce unnecessary resource dependencies.

The same probe identified validation overhead: median record/finish/submit was
11.01/12.18/2.59 ms with validation, versus 3.16/3.15/0.045 ms without. Normal app
runs disable validation, with `--gpu-debug` for validation and CPU-comparable
historical measurements. GPU correctness tests explicitly keep validation enabled.
Record this mode in every profile; never present validation-off versus the old
validation-on baseline as a pure algorithmic speedup.

API evidence: [SlangPy API](https://slangpy.shader-slang.org/en/latest/src/api_reference.html),
[Slang parameter blocks](https://docs.shader-slang.org/en/stable/parameter-blocks.html),
and local probes in `artifacts/performance/shader_object_probe.py` and
`artifacts/performance/parameter_block_probe.py`, with their JSON outputs.

### P4 — all worlds in one dispatch chain

Files: `sim.py`, `benchmark.py`, `app.py`, shaders, `snapshots.py`, `profile.py`.

- [ ] Allocate live and benchmark worlds together; Benchmark controls a world
  range instead of owning a separate Sim.
- [ ] Move local tick, seed/random-world identity, rates, food/age/mutation
  settings, active state and trial-stop tick into per-world GPU state.
- [ ] Preserve random identity namespaces. Current child GUIDs depend on
  `num_worlds` and world index; allocation changes must not create collisions or
  silently change the live world's seeded random streams.
- [ ] Reset only benchmark ranges; preserve frozen reference brains, swapped
  layouts, fixed conditions, cumulative energy-collected totals and live sliders.
- [ ] Stop trial worlds exactly at the trial boundary while live worlds continue
  through the remaining ticks in the encoder. Prepare replacement trials safely.
- [ ] Update rendering/selection and profiler labels/counts for shared dispatches.
  Do not report shared work twice or multiply paired-world tick counts by two.
- [ ] Migrate old separate-allocation checkpoints, including time-major history,
  reference buffers, local clocks and unfinished trials. Verify restored sessions.
- [ ] Test multiple live worlds, trial boundary inside a 100-tick batch, restore,
  independent settings, GUID uniqueness and benchmark view selection.

P4 implementation audit (next step): introduce separate per-world settings/runtime
buffers before changing allocation ownership, then range-scoped initialization,
shared dispatches, selection/rendering and checkpoint migration. Preserve original
random-world IDs and GUID stride independently of physical world indices. Keep
the current `t` versus `t+1` uses explicit. Benchmark initialization must use one
RW binding with source/destination ranges when current and trial data share a
buffer; the old separate SRV/UAV bindings would alias. Sampling and frozen-reference
copies must cover only the live prefix. Legacy migration must interleave every
plane of time-major history and rebase nonnegative benchmark grid links, rather
than concatenate entire history arrays. Test a trial boundary within a 100-tick
batch, independent slider settings and unchanged live state during trial resets.
No API blocker was found. Combined production remains GPU-limited, so measure
shared GPU work and dispatch amortization without assuming a threefold gain.

### P5 — compact active records with stable state ownership

- [ ] Give food a stable spawn identity and use creature GUID/food identity for
  priorities and ties before indices can change. Record the seeded-protocol change.
- [ ] Compact hot creature and food records after their last old-index consumer.
  Maintain per-world dense active counts and stable network/history handles.
- [ ] Allocate/release stable state handles on birth/death; initialize reused
  history so it cannot leak observations from a previous creature.
- [ ] Keep grids valid by rebuilding at the next snapshot boundary. No consumer
  traverses old indices after compaction; include buffers in snapshot migration.
- [ ] Find selected creature by GUID on GPU once per rendered frame, sharing the
  result among following, outline, vision and neuron views. Clear selection on
  death, and never attach it to a slot's replacement occupant.
- [ ] Test reorder invariance, selected creature movement through compaction,
  extinction/full capacity, repeated reuse and complete checkpoint restoration.

### P6 — vision across creatures and segments

- [x] One thread per creature × segment, with conservative wedge/cell rejection
  before linked-list traversal. Each segment checks the 169 candidate cell bounds,
  but walks only lists whose enclosing cell circle intersects its wedge and range.
  A 0.0001 m tolerance protects boundary cases against floating-point rounding.
- [x] Preserve distance/bearing, nearest-object occlusion, kind/size/energy,
  deterministic ties, inclusive FOV/range boundaries and toroidal geometry.
- [x] Segment zero writes one pose/tick header per observer; each segment writes
  its full-precision slot. Retain the complete 256-frame history and rollover.
- [x] Compare observations against controlled reference scenes and the existing
  capture tests, then measure sparse and crowded scenes. Keep only measured wins.

Implemented in `src/worms/shaders/vision.slang`, the retained dispatch dimensions
in `src/worms/sim.py`, and `tests/test_vision_gpu.py`. Each worker keeps one winning
slot instead of arrays of sixteen slots and candidate indices. The grid builders,
exact point-to-segment calculation and deterministic tie helper are unchanged.

Three new tests compare against a brute-force GPU oracle that scans every object
without grids or cell rejection: two seeded crowded scenes across three worlds,
and twenty boundary layouts covering every angular/FOV edge on both sides,
wrapped cell corners, coincident points and the exact/just-outside 24 m range.
The complete `pytest --run-gpu` regression suite passed: 212 tests.

An independent before/after capture benchmark retains identical shader-object
bindings and immutable scene buffers for both versions. It warms each variant
with 400 captures, then alternates five measured batches of 50 captures, using
only two GPU timestamps per batch. D3D12 debug layers and RHI validation are both
enabled. Full current-frame headers and every segment match byte-for-byte between
versions in all three scenes; scene inputs and the source checkpoint are unchanged.

| Capture-only scene | Previous ms/capture | Segment workers ms/capture | GPU speedup |
|---|---:|---:|---:|
| 100k checkpoint: 91 creatures, 67 food | 0.16421 | 0.09327 | 1.76× |
| 256 creatures, 1,024 food in a 48 m square | 1.27861 | 0.41542 | 3.08× |
| 1,024 creatures, 2,048 food in a 48 m square | 2.76951 | 1.23761 | 2.24× |

These are median isolated capture costs, not application tick-rate measurements.
All scenes reserve 1,024 creature slots; sparse food capacity is 8,192 and crowded
food capacity is 2,048. CPU recording is 0.92–1.32 ms per 50 captures; encoder
finish is 1.21–1.48 ms. The complete comparison took 9.71 seconds.

Evidence: `artifacts/performance/vision-p6-comparison.json`; reproducible script:
`artifacts/performance/benchmark_vision_p6.py`; preserved original shader:
`artifacts/performance/vision-before-p6.slang`. No global shader file or simulation
kernel was swapped during measurement. End-to-end profiling remains under P9.

### P7 — indirect dispatch where it pays

- [ ] Use GPU active counts to write dispatch arguments after compaction.
  Dispatch groups are `ceil(N/group_size)`, with zero for empty work.
- [ ] Account for world dimensions: per-world indirect calls must not undo the
  shared-dispatch benefit. A flattened active work list may be preferable.
- [ ] If measured useful, generate food-grid dirty/build arguments on GPU. Clear
  only when rebuilding, with correct ordering even when work count is zero.
- [ ] Validate resource states, zero-world/zero-entry cases and checkpoint restore
  of derived state. Avoid CPU reads solely to decide dispatch dimensions.

### P8 — CPU/GPU overlap and UI diagnostics

The implemented first stage submits bounded chunks within one refresh. It does
not require combined live/benchmark allocations or duplicate simulation state.
The same queue orders every chunk, followed by one draw/present. At 100 steps,
the default records four 25-step chunks; CPU recording overlaps earlier GPU work.
SlangPy owns the final window encoder and presentation, while our FIFO retains
intermediate command resources until completion. Low-speed window presentation
remains bounded by SlangPy's swapchain.

- [x] Keep the existing simulation tick loop and submit it in bounded chunks.
  There were no per-tick waits to remove.
- [x] Keep at most two main submissions in flight at high speed, including a
  reserved slot for the SDK's final submission; retain command-buffer references.
- [ ] Stage inspector/counter reads asynchronously and tag them with world, tick
  and GUID. An old response must not follow a replacement creature.
- [x] Synchronize reset/restore, snapshot capture and benchmark trial handoffs at
  deliberate boundaries. Checkpoints must represent coherent completed ticks.
- [x] Measure pipelined throughput separately from synchronized pass attribution.
  Actual window comparisons, paired Nsight timelines and a 151,000-tick run pass.
  Window frame p95 improves from 103.8 to 63.6 ms in the matched Nsight capture.
  Resource bounds are tested; the 1,000-step run completes 6,040 main submissions
  without descriptor exhaustion. Network allocations halve. Hardware occupancy
  and a full device-memory residency profile remain unmeasured.

The adopted FP16 neural path uses half storage and all-half inference arithmetic.
FP32 observations convert only at the input boundary. Initialization and mutation
are outside inference and remain FP32. Scalar half storage with float accumulation
was slower and is not used. The all-half follow-up has a maximum measured control
difference of 0.001182 on fixed saved observations; future trajectories can diverge.
Offline DXIL has no widening, FP32 arithmetic or FP32 intrinsic calls in inference.
Interleaved fixed-input timings show this follow-up is 3–4% slower than the prior
half-dot/float-activation variant on this device, while still beating all-FP32.
Snapshot format 5 migrates old network buffers and preserves source archives.
Final validation totals 289 tests (97 CPU, 192 GPU), plus Ruff lint/format checks.
Detailed results and native Nsight capture names are in PERFORMANCE.md.

## Validation and measurement commands

```powershell
.\.conda\python.exe -m ruff check src tests
.\.conda\python.exe -m pytest
.\.conda\python.exe -m pytest --run-gpu
.\.conda\python.exe -m worms.profile --gpu-debug --resume artifacts/performance/checkpoint-100000.npz --no-benchmark --steps-per-frame 100 --warmup 200 --samples 20 --output artifacts/performance/profile-next-live-1.json
.\.conda\python.exe -m worms.profile --gpu-debug --resume artifacts/performance/checkpoint-100000.npz --steps-per-frame 100 --warmup 200 --samples 20 --output artifacts/performance/profile-next-comparison-1.json
```

Repeat profiles with suffixes 2 and 3, sequentially. Tests requiring changed
semantics should state the intended contract rather than simply relaxing checks.
For later steps add crowded/full-capacity and birth-heavy workloads alongside
the fixed sparse checkpoint. Remeasure GPU busy time after substantial changes;
claim hardware occupancy only if actual counters become available.

## Progress and evidence log

### 25 October — settled benchmark reference

- Protocol 6 waits for 200,000 submitted live ticks before freezing the surviving
  population as the fixed reference. Benchmark updates do no work during the wait.
- Capture occurs at a frame boundary; the actual tick is recorded. Trials last
  100,000 ticks, so the first result appears around live tick 300,000. Benchmark
  views appear after capture, and labels call the cohort the baseline/reference.
- New checkpoints preserve both the wait and the captured reference/trial state.
  Earlier benchmark protocols are unsupported, as requested; the historical
  migration notes below describe previous versions only.
- Profiling keeps the dormant allocation in its accounting but captures the
  reference outside timed frames, and records the waiting/capture state.
- Verification: 45 GPU benchmark/session/view/profiler checks and 56 CPU checks
  passed. These cover survivor-only sampling, exact and overshot thresholds,
  waiting/captured/pending checkpoint replay, reference stability, reset/view
  routing, and profiler batches crossing the capture threshold. Ruff is clean.


### 25 October — two offspring per split

- Splitting now retires the parent and creates two fresh identities, both at the
  original generation plus one. Both networks mutate independently from the
  original, and both histories, activations, commands and ages start fresh.
- Sparse resolution still uses one action per split; the 64-thread initializer
  now handles both offspring. The internal `births` field still counts split
  events and is labelled **Splits** in the UI. Snapshot buffer layouts are unchanged.
- Snapshot format is version 4 and comparison protocol is version 5. Formats 1–3 load their
  live state and original opponent; old benchmark trials/graphs restart. Earlier
  measurements below used the parent-survives rules and remain historical.
- Input-weight and bias/output mutation probabilities are separately adjustable
  and saved, now both defaulting to 10% with percentage text-entry controls. This
  adds no dispatches or normal-tick readback.
- Live and benchmark food respawn now use the default 10 pieces/s; live controls
  use numeric entry. Protocol 5 restarts earlier benchmark trials/graphs because
  their food baseline was 1.5 pieces/s, preserving live state and the frozen reference.
- Automatic checkpoint cadence is now 100,000 live ticks, retaining the latest
  five files; the UI reads the cadence from the snapshot store.
- Two-offspring verification: 169 GPU checks and 56 CPU checks passed, with Ruff checks and
  formatting clean. Coverage includes both offspring's ancestry/mutations/history,
  same-slot explicit reselection, legacy GUID uniqueness, protocol migration and
  restored continuation. The new checkpoint cadence was also checked at its
  100,000- and 200,000-tick boundaries.


### 25 October — resolution, lifecycle and binding reuse

- Added the requested reusable shader-object/parameter-block design as P3b.
  Installed SlangPy exposes shader-object creation and binding to compute passes.
  Lifetime, changing constants and barrier behavior passed dedicated GPU probes
  and retained-binding regression tests.
- P2/P3 implementation: one creature grid build; pre-resolution vision/brains;
  16 contact lanes per creature with per-food atomic priority reduction and one
  queued action per food; serial action validation; parallel costs/deaths;
  deterministic bitsets for eligible parents/free slots; sparse birth records
  and 64-thread initialization per actual child. This avoids sorting birth queues
  while preserving lowest-parent/lowest-free-slot allocation. Scratch is derived
  each tick and does not enlarge checkpoints.
- At this stage snapshot format and benchmark execution protocol became version 3. Versions 1
  and 2 remain readable; live state and frozen reference are kept while old-protocol
  benchmark trials and graph restart. Newborns first sense on their following tick.
- Full regression: **212 tests passed** in 211.85 seconds, including 156 GPU
  checks with D3D12/RHI validation enabled and 56 CPU checks. Ruff check and
  formatting passed. Final measurements follow below.
- P6 was moved ahead of P4 because vision was then the largest measured GPU pass;
  its P2 prerequisite is implemented. No dependency on changing world ownership
  is needed for segment-parallel capture.
- P6 complete: segment workers and conservative cell rejection preserve captured
  results. Three new brute-force parity tests pass within the 212-test full suite.
  The separate validation-enabled before/after benchmark found 1.76× faster
  sparse capture, 3.08× faster crowded capture and 2.24× at full creature capacity,
  with exact header/slot parity and unchanged inputs. Its method, timings and
  artifact paths are recorded in the P6 section above.

- Final profiles: twelve sequential runs, three for each workload/validation
  combination. With validation retained, median throughput is 1,873.5 live ticks/s
  and 894.2 with the pair (2.46× / 2.69× P0). Production mode reaches 3,610.6 and
  1,661.3 ticks/s. The checkpoint hash remains unchanged; tick ranges are
  100,200–102,200 and all grids build exactly once per active tick.
- Production CPU preparation is 10.83 / 21.35 ms per 100-tick frame, against GPU
  15.28 / 37.13 ms: all six production runs are GPU-timeline limited. Validation
  raises CPU preparation to 31.21 / 64.60 ms and changes the limit to CPU. Native
  encoder finishing/submission are a large part of that difference.
- Brain evaluation and vision now dominate named GPU stages. Per-dispatch timers
  inflate short-pass time by 1.62–1.73× in production and up to 2.87× with
  validation; total-throughput figures use separate boundary-only queries.
- The birth-heavy diagnostic passed all 20 repeated 512-birth samples; timing
  and initialization-marker evidence are recorded in P3 above. A final targeted
  profiler regression also passed after adding the explicit one-creature-build
  assertion.
- Raw complete profiles: `artifacts/performance/profile-p236-{debug,production}-{live,comparison}-{1,2,3}.json`;
  aggregate `profile-p236-summary.json`. Full tables, ranges and method limits
  are in PERFORMANCE.md. The shared parameter block remains deferred after its
  probe; per-kernel retained shader objects provide the binding reuse now.
- Production telemetry rechecked GPU busy time over 50,000 live ticks per
  workload: 52.8% live and 60.3% combined, with zero idle baseline and no active
  clock-event reasons. Timestamp wall fractions agree. Synchronized CPU gaps
  remain; this is not a hardware occupancy measurement. Evidence is in
  `artifacts/performance/utilization-p236-production.json` and PERFORMANCE.md.
- Next implementation milestone is P4: combine live and benchmark allocations
  while preserving per-world clocks, random identities and saved history. P5
  compaction, P7 indirect work and P8 overlap remain unchecked future work.

### 25 October — initial execution

- P0 complete: fixed checkpoint, three-repeat profiles, dispatch attribution and
  device-wide utilization recorded in PERFORMANCE.md.
- P1 code implemented: atomic insertion for both grids, parallel radius reduction
  in the first clearing group per world, food rebuild every tick, unchanged buffer
  layouts. GPU compilation/correctness and all six performance runs passed.
- P1 tests updated: crowded cells across workgroups and worlds, sparse high slots,
  death/reuse, radius reset, credits and unordered grid replay comparisons.
- P1 validation: 74 GPU tests passed in 92.83 seconds (contacts, vision, lifecycle,
  snapshots, sessions, benchmarks and profiling). CPU suite: 56 passed. Ruff clean.
- User clarified resolution semantics: queued actions can invalidate later ones;
  splitting is structural resolution after detection. P2 now uses an immutable
  detection/vision snapshot and next-tick participation, replacing link repair.
- P1 profiles: 999.4 live ticks/s (31.4% improvement) and 484.4 live ticks/s with
  benchmarks (45.7% improvement), medians of three runs. GPU frame medians fell
  from 89.30 to 50.10 ms and from 223.83 to 116.66 ms respectively. The unchanged
  source hash, starting world metadata and measured tick ranges match the baseline.
- Both creature clear/build pairs together now cost 0.0347 ms/live tick, down from
  0.4069 ms. Food clear/build every tick costs 0.0153 ms/tick, below the old cached
  serial grid's amortized 0.0234 ms. These are repeat-2 detailed timings.
- CPU preparation rose from 40.75 to 46.39 ms/live frame and 74.25 to 86.74 ms with
  benchmarks; per-tick food rebuilding records more dispatches. Live-only is now
  mixed CPU/GPU-limited by the profiler's 25% rule. Combined remains GPU-limited.
  Serial lifecycle now accounts for 56.0% of detailed live-only GPU time.
- Raw reports: `artifacts/performance/profile-p1-live-{1,2,3}.json` and
  `artifacts/performance/profile-p1-comparison-{1,2,3}.json`. Detailed timestamp
  overhead is 17.8–19.1% live-only and 18.4–18.5% combined; throughput excludes it.
- P1 preserves current tick behavior and snapshot layouts. No benchmark protocol
  restart is needed for P1. No post-change hardware occupancy claim is made.
- Next action at the end of P1: P2's snapshot/queue contract and tests, then remove the second
  creature build. Coordinate the sparse lifecycle actions with P3, now the largest
  measured GPU opportunity. The remainder of the roadmap is not implemented yet.

| Milestone | Live ticks/s | Combined live ticks/s | Evidence | Remaining risk |
|---|---:|---:|---|---|
| P0 baseline | 760.7 | 332.5 | PERFORMANCE.md, six baseline JSON reports | Serial grids/lifecycle dominate |
| P1 atomic grids | 999.4 | 484.4 | 74 GPU / 56 CPU tests; six profile-p1 JSON reports | Second build and serial lifecycle remain |
| P2/P3/P3b/P6, validation on | 1,873.5 | 894.2 | 212 tests; six debug profiles; vision and birth stress probes | Validation CPU overhead; separate live/benchmark chains |
| P2/P3/P3b/P6, production | 3,610.6 | 1,661.3 | Six production profiles; same checkpoint hash | Brain/vision GPU cost; compaction and P4 still pending |
