# GPU performance measurements

Measured on 24 October 2026 using the NVIDIA GeForce RTX 3050 Ti Laptop GPU,
D3D12, SlangPy 0.43.1, debug validation enabled, and a 1280 × 720 headless render.
The initial population is 64 creatures in a 512 m square world, with 2,048 food
pieces and 8,192 reserved food slots. Results describe this scene and hardware;
a densely populated evolved world can cost more.

Run the same measurement with:

```powershell
.\.conda\python.exe -m worms.profile --capacity 1024 --steps-per-frame 100 --output artifacts/profile.json
.\.conda\python.exe -m worms.profile --capacity 1024 --steps-per-frame 100 --benchmark --output artifacts/profile-comparison.json
```

`--worlds`, `--creatures`, `--samples` and `--warmup` control the test scene and
sample count. This is a diagnostic command: it advances an isolated simulation,
disables periodic checkpoints, and writes a JSON report under ignored
`artifacts/`. It does not load or alter a saved run.

## Method

The profiler warms the shaders and scene for 200 ticks. It then measures ten
frames with hardware timestamp queries around the entire update/render command
stream, and CPU clocks around command recording, finishing, submission and wait.
A second set of frames adds GPU timestamps before and after each compute
dispatch and around rendering. Per-pass attribution comes from this second set;
throughput uses only the two frame-boundary queries, so detailed instrumentation
does not distort that result. Dispatch costs include the transitions and barriers
inside their timestamp interval.

Each measured frame waits for GPU completion. Reported ticks/s are therefore
**synchronized headless throughput**, including CPU preparation and GPU work.
Window presentation, vsync, native UI, periodic checkpoint I/O and the occasional
setup of a new benchmark trial are excluded.
Headless App frames also wait for their submission to finish, bounding transient
D3D12 descriptors when running many ticks per render. A window can overlap CPU
recording with a previous GPU frame, so these numbers
are not an exact window FPS prediction. Simulation speed remains a plain number
of ticks per rendered frame; selecting 100× does not guarantee 6,000 ticks/s.

The CPU/GPU label compares median CPU preparation time with median GPU timeline
time, with a 25% margin for a clear winner. GPU wait alone is not used as evidence
of a GPU bottleneck. A GPU timeline bottleneck can be caused by serial kernels
using very little of the GPU's available parallelism.

## Baseline before the food-grid cache

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

## With the cache and 1,024 creature slots

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
`artifacts/profile-optimized-{live,comparison,full}.json`. The command above
regenerates reports for the current code and hardware.

## Complete session check

A subsequent 20,000-tick run with the comparison graph enabled completed two
10,000-tick trials and two automatic session snapshots in 68.05 seconds
(293.9 live ticks/s including checkpoint I/O). The live population grew from 64
to 81, with 17 births. The full checkpoint files were about 17.9 MB each and
preserved the initial reference, comparison history and next trial. Native window
checks exercised the graph alongside neuron activations, Snapshot now, Step and
Restore latest snapshot. The restored session was paused at its saved tick.

The two trial scores were 49.6% and 50.2%; this short validation establishes that
the system runs, not that evolution has already produced a meaningful gain.
