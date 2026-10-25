# Worms

A small Python + SlangPy artificial-life project. The application runs independent GPU worlds with creatures driven by evolving neural networks, clustered food, competitive benchmarks, restorable checkpoints, a windowed viewer, and a headless path for tests and image dumps.

## Run

From PowerShell in the repository root:

```powershell
.\.conda\python.exe -m pip install -e ".[dev]"

# Open the application (1 live world, 64 creatures, 1,024 slots).
.\.conda\python.exe -m worms

# Advance exactly 120 fixed ticks and save the final scene.
.\.conda\python.exe -m worms --headless --frames 120 --output artifacts/worlds.png

# Change the initial live world and creature counts.
.\.conda\python.exe -m worms --worlds 6 --creatures 128

# Run 300,000 ticks, including baseline capture, a comparison trial and snapshots.
.\.conda\python.exe -m worms --headless --frames 3000 --speed 100

# Restore the most recent automatic checkpoint; window starts paused.
.\.conda\python.exe -m worms --resume latest
```

The window has a Speed dropdown (Paused, 1x, 2x, 5x, 10x, 20x, 50x, 100x, 200x, 500x and 1000x), Step, Reset, and an all-worlds/single-world selector. Speed sets the number of simulation updates per rendered frame. Escape or the close button exits. `--frames N` also works in windowed mode for a finite run. PNG exports contain the scene without UI. A headless run defaults to 120 frames and `artifacts/worlds.png`.

Select a single world to navigate and inspect it (a one-world run starts in this view):

- **Mouse wheel:** zoom toward the cursor, from 1x to 256x. Each notch changes magnification by the same proportion, so the perceived zoom speed stays consistent at every scale.
- **Left drag:** pan with the pointer. Panning stops follow without jumping the camera.
- **Left click:** select a creature and show its stats, vision arc and live neural activations in a bottom strip. A small pixel tolerance makes tiny creatures easier to select. Click empty space or close the inspector to clear selection.
- **Follow:** track the selected creature, including across wrapping world edges. Wheel zoom still anchors under the cursor while following.
- **Fit world:** return to the full-world view. Switching worlds or resetting also resets the camera and selection.

Pause can help with selecting a fast-moving creature. The single-world view continues across the world's wrapping edges; it may show neighboring copies of the same world when zoomed out. Population counters and the inspector refresh five times per second with small explicit readbacks; following itself uses the current GPU position every frame.

The application currently targets D3D12 and selects the NVIDIA adapter. It uses the existing Python 3.12 environment and SlangPy 0.43.1; no extra UI dependency is needed. For a fresh environment, use `conda create --prefix .conda python=3.12 pip`, then install `requirements-dev.lock` and the editable project. A Python 3.12 venv also works; substitute its interpreter in the commands.

## Structure

- `src/worms/app.py`: device, window/UI, frame loop, tiled/focused views, PNG output.
- `src/worms/camera.py`: cursor-anchored zoom and pixel-to-world pan math.
- `src/worms/brain_view.py`: selected-creature activation display and its static labels.
- `src/worms/benchmark.py` and `comparison_view.py`: frozen-reference competitions and history graph.
- `src/worms/snapshots.py`: versioned, atomic save/restore and rolling checkpoints.
- `src/worms/profile.py`: hardware GPU timestamps and CPU/throughput measurements.
- `src/worms/throughput.py`: sustained throughput windows, trial restarts and render-off measurements.
- `src/worms/sim.py`: simulation buffers, compute dispatches, and drawing a world into a rectangle, with optional camera offset and scale.
- `src/worms/shaders/world.slang`: creature/world records, seeded initialization, and motion.
- `src/worms/shaders/food.slang`: periodic Perlin density, seeded placement, and gradual food growth.
- `src/worms/shaders/contact.slang`: spatial grid construction, food ownership and energy transfer.
- `src/worms/shaders/lifecycle.slang`: energy costs, death, splitting, inherited weight mutations and newborn history reset.
- `src/worms/shaders/vision.slang`: food grid, per-creature sensing and full observation history.
- `src/worms/shaders/vision_types.slang`: shared observation records and vision constants.
- `src/worms/shaders/brain.slang`: seeded network parameters, input normalization and inference.
- `src/worms/shaders/brain_types.slang`: fixed network dimensions and parameter layout.
- `src/worms/shaders/brain_render.slang`: GPU neuron activation and control-gauge rendering.
- `src/worms/shaders/world_render.slang`: pond backgrounds, food/creatures and selected vision overlay.
- `src/worms/shaders/inspect.slang`: picking within the focused world, reading back one index.
- `tests/`: layout checks and opt-in tests using the actual GPU.
- `docs/` and `assets/`: planning documents and original visual references.
- `checkpoints/`: selected milestone runs intentionally preserved in Git.
- `artifacts/`: generated outputs, ignored by Git.

Each world has a fixed block of creature slots in one shared GPU buffer, with **1,024 slots per world by default**, starting with 64 creatures. `Sim.creatures_per_world` is the initial count; `Sim.creature_capacity` is the allocation stride. Dead slots are reused without moving survivors. Each newborn receives a fresh ID, parent ID, generation and birth tick. The world records its current live count and cumulative splits/deaths, shown in the controls. The internal `births` counter counts completed split events: each replaces one creature with two, increasing the population by one. Retiring a parent during splitting does not count as a death. Seeded Reset reproduces the initial state. The action/target fields are placeholders for later simulation work.

One world unit is one metre. Worlds are **512 x 512 m**. A default creature's full body length averages **1 m**, with seeded variation from **0.8 to 1.2 m**; body width is 63% of its length. Body size is independent of population and world size. The 64-creature default is sparse: one creature per 4,096 square metres. At whole-world scale creatures are tiny specks; zoom in to inspect them at a useful scale.

One update applies each creature's steering and throttle for 1/60 second and wraps both coordinates at the world boundaries. Maximum speed is initialized to 8–16 m/s and stored separately from actual velocity. Rendering reads the same GPU state, including copies at wrapping boundaries. Each frame runs `speed` simulation updates in a loop, then renders once; 100x means 100 updates per render. Windowed and headless runs use the same loop, defaulting to one update per frame. Pause freezes the simulation, Step advances exactly once regardless of speed and remains paused, and Reset restores the seed without an extra tick that frame. Reset keeps the selected speed.

At higher speeds, updates are submitted in chunks of at most 25 ticks on the same GPU queue. Recording the next chunk overlaps execution of earlier work, with at most two submissions in flight. The final chunk shares the frame's draw submission. Settings, snapshots and trial transitions remain at refresh boundaries; a 100x refresh still draws only once.

## Food

Each world reserves **8,192 food slots**, starts with **2,048 pieces**, and adds **10 pieces per simulated second** toward **4,096 live pieces** (about 3.4 simulated minutes from reset if nothing is eaten). Each piece defaults to **five energy units**. The **Food energy** slider adjusts this from 0.25 to 10 units while running or paused, across all worlds. Existing uneaten pieces and future growth use the new value; positions and live counts stay unchanged. Reset keeps the chosen value. Previously recorded vision remains unchanged. These defaults live in `sim.py`; small experiments can override `Sim`'s food capacity, initial amount, target and growth rate without adding application configuration.

The **Food respawn (/s/world)** text field sets additions in **pieces per simulated second per world**; type a nonnegative rate, with zero stopping additions. Changes apply on the next simulation tick, and Reset keeps the chosen rate. The food counter shows the current live pieces against the target for the selected world, or totals across all worlds in the overview. It refreshes with the population counters five times per second.

Food shares one GPU allocation across all worlds, addressed as `world * food_capacity + slot`. Each 12-byte record stores position and integer energy; zero energy marks an empty slot. The world stores its live count, allocation cursor and spawn sequence. Growth reuses empty slots without moving surviving food, reallocating or reading back simulation state. A reused slot gets a fresh location from the spawn sequence. Pause, Step and seeded Reset apply to food as well as creatures.

Placement accepts random candidates according to a two-octave periodic Perlin density field, giving rich clusters and sparse gaps. The field wraps continuously at world edges. Candidate work is bounded to 128 attempts per piece, with the densest candidate as a fallback. Each world's sequence is independent of other world counts, creature population and growth batching. New food samples the same density field at new positions.

Food is drawn as lime specks with a **0.18 m radius** and a subtle halo. A minimum visual radius of 0.7 pixels keeps clusters readable when zoomed out. Zoom, pan and follow use the same camera and wrapping as creatures. Rendering visits the reserved slots and skips empty ones, so eating an early slot cannot hide surviving food later in the array.

## Contact and collection

Each tick moves creatures using stored brain commands, builds the creature and food grids once, captures vision and evaluates the next commands, detects contacts, then resolves food, energy costs, deaths and splits. Food grows after resolution. New food, newborns and split positions participate in sensing and contacts on the next tick. Every grid consumer finishes before structural changes, so no second creature-grid rebuild is needed. Each world's grids have **128 x 128 cells of 4 m**, with parallel clears and atomic head exchanges for insertion. Worlds have separate heads and no per-cell occupancy cap.

Contact detection runs across creature × nearby cell. Normally each creature checks 3 x 3 food-grid cells; larger bodies expand their search using their own radius. Cells wrap uniquely across world edges. Each worker performs exact toroidal circle overlap tests before proposing collection. Dense local crowds can still require many comparisons.

Touching food removes the whole piece. An atomic minimum picks one eater using a seeded priority and a deterministic tie, independently of scheduling. The first contender queues the food item exactly once, so the action queue is bounded by food capacity. One resolver per world visits only queued items, rechecks the eater's identity/alive state and food availability, and accumulates credits. An earlier action can invalidate a later one without allowing duplicate transfer.

A parallel pass scales each creature's total intake by its age and applies it. Scaling uses 64-bit integer arithmetic and rounds down once per tick to whole energy quanta. Unabsorbed energy is lost, including the whole piece at zero intake. Creatures start with **100 energy units per metre of body length**, with **1,024 integer quanta per unit**. Per-tick credits are 32-bit with a bounded food allocation; persistent creature energy is 64-bit and has no gameplay cap. The inspector displays energy in units.

Combat, invincibility and lifetime learning remain future work in the [project plan](docs/PROJECT_PLAN.md). `baseline.proposed.toml` remains planning data, not a supported application configuration. The earlier standalone smoke renderer has been replaced by this shared application path.

## Energy and splitting

For full body length `s` metres and actual speed `v` metres/second:

- Initial energy is `100 × s` units; the split threshold is twice that initial amount.
- Depletion is `s × (0.05 + 0.001 × v²)` units/second, charged after food collection.
- Energy is stored in integer quanta, with fractional depletion carried between ticks. Reaching zero kills the creature.

A stationary creature still spends energy. At these starting rates, initial reserves last about 33 simulated minutes at rest, 15 minutes at 8 m/s, or 5.4 minutes at 16 m/s without food. Both reserves and costs scale linearly with size, so these durations are independent of body length. The two cost constants live in `lifecycle.slang`.

Creatures default to a **100,000-tick lifespan**, with full food intake until they die at that age. Age counts simulation ticks since birth or the creature's most recent successful split. The **Decline age** slider defaults to **100,000 ticks** and **Fade length** to **0 ticks**, giving an immediate end of life. A positive fade length instead adds that many ticks of linearly declining food intake; death occurs at decline age plus fade length. Both sliders cover 0–100,000 ticks, apply to existing creatures on the next tick, and survive Reset. The inspector shows age and current intake percentage.

The age limit kills a creature even when it has stored energy, clearing its reserves and recording one death before reproduction is considered. Starvation can still kill it earlier. A successful split retires the parent and creates two offspring, each with a fresh identity, age zero, full food intake and a fresh lifespan. Both record the original parent's ID and generation plus one. A split blocked by a full world leaves the original creature and its age unchanged. Reset returns the initial population to age zero. Vision continues to report each food piece's stored energy before any age adjustment.

At the threshold, a creature splits if its world has a free slot. One offspring occupies the original slot and the other occupies the free slot. They share the remaining energy equally (the first keeps an odd quantum), preserving total energy. Both inherit the same body and maximum speed, independently mutate copies of the original network, and start with empty vision history and zero neural activations. They move to opposite sides of the split point, leaving a gap of one quarter of their body length. Each turns 90 degrees in the opposite direction, facing outward along their separation; positions wrap at world edges. Both have zero velocity and movement output until their first observation and policy evaluation on the next tick, then can move on the following tick. Each parent can split at most once per tick; newborns cannot split on their birth tick. A full world delays splitting without charging energy or discarding surplus reserves.

On each successful split, each offspring's 1,024 input-to-hidden weights independently have a **10% default mutation chance**. The remaining 50 parameters—hidden biases, output weights and output biases—also have a **10% default**. All mutations add Gaussian noise with **standard deviation 0.02**; about 107 parameters change per offspring on average. Both copies inherit the original network before either offspring's mutations are applied.

The **Input-weight mutation (%)** and **Bias/output mutation (%)** text fields independently control each block's mutation chance from 0% to 100%, applying to future offspring. Type a percentage directly; values are limited to 0–100%. Set both to zero for exact copies. Reset retains both settings, and snapshots save them. Older checkpoints without these settings restore the legacy 1% rates; the fields can change those rates in an existing run. Benchmark worlds keep mutation disabled and show the controls read-only.

Mutation runs on the GPU and is keyed by the simulation seed, world, split counter and offspring, so siblings get independent changes, reused slots get fresh mutations, and Reset replays splits on the same GPU. A descendant inherits its parent's already evolved network before applying its own mutations. Trials of 3% and 10% for the smaller block scored worse than 1% under the earlier parent-survives reproduction rules. Those results do not establish the best rates for the new two-offspring rules. Both defaults are now 10% for further experiments; judge changes with the fixed comparison graph. See the [mutation-rate results](docs/COMPARISON.md#mutation-rate-experiment--25-october-2026).

Successful feeding and reproduction can now spread new network variants, while depleted creatures die. Food availability, luck and the population cap also affect survival; mutation does not guarantee improvement. Body traits remain unchanged and there is no lifetime learning. The inspector shows energy, split threshold and lineage. Selection/follow is cleared when its creature dies, splits or its slot is reused.

## Vision

Every living creature senses **16 equal segments across a 180-degree forward arc**, out to **24 m**. Heading determines the direction of motion and can turn independently while stationary. Each segment sees only the **closest object**, comparing food and other creatures together by centre distance using the shortest wrapped displacement. That object hides every farther object in the same segment: food can hide a creature and a creature can hide food. This is occlusion at segment resolution, not a geometric raycast. Self, dead creatures and empty food slots are excluded. The range and outer arc boundaries are inclusive. Exact-distance ties prefer food, then the lower buffer index within the same kind, independent of grid traversal order.

Each segment retains one 20-byte observation:

| Fields | Meaning |
|---|---|
| `distance`, `bearing` | Visible object's centre distance in metres and bearing in radians relative to heading |
| `size` | Food diameter (0.36 m) or creature's full body length, in metres |
| `food_energy` | Visible food's own energy in units; zero for a creature |
| `kind` | Integer: 0 for empty, 1 for food, 2 for creature |

A distance of `-1` and kind of zero mean nothing was seen; the other fields then hold zero. Hidden objects contribute no properties or energy to the observation. A second 4 m spatial grid indexes food. It is rebuilt on reset and every tick before vision, with one insertion thread per food slot and atomic head exchanges. Empty slots are omitted. Capture runs across observer × segment, rejecting cells outside that segment before traversing their lists in either grid.

Vision records the post-movement scene before food collection, deaths and splitting. Food eaten during resolution therefore remains in that tick's captured observation; both offspring start without observations and first see the world on the next tick.

The GPU retains **256 complete observations per creature** in a ring buffer, about **4.3 simulated seconds** at 60 ticks/s. Each frame includes all 16 segments plus the observer's position, heading and tick. The default 1,024 reserved creature slots use 84 MiB per world for observations and frame headers. Reset clears history and records one initial observation; Pause adds none, and Step adds one. A creature retains the observation captured before it died; subsequent ticks produce invalid frames while older observations remain until overwritten; reusing a slot clears its entire history before recording the newborn's first observation. Rendering and sensing keep these buffers on the GPU; `sim.inspect_vision(world, local_index, age=0)` reads one retained frame only when explicitly requested for diagnostics.

Selecting a creature displays its latest captured arc with subtle segment boundaries. A segment's tint indicates the visible object's kind; lime rings mark visible food and orange rings mark visible creatures. Hidden objects receive no vision marker. The overlay uses the recorded observations and pose, including across world seams. Zoom in to see the arc clearly.

History is stored at full angular and temporal resolution. The brain currently uses only the newest observation. Reducing older samples into fewer brain inputs will be a separate pass, recomputed from this retained history.

## Neural control

Each creature has its own **64 → 16 → 2 feedforward MLP**, initialized with independent seeded Xavier-uniform weights and zero biases. Each creature's parameters remain fixed throughout its life; offspring inherit them with small mutations. Reset reproduces the initial population. Initial reactions to food and other creatures are arbitrary, with subsequent evolution driven by survival and reproduction.

Network weights, activations and all inference arithmetic use native FP16, including input normalization, dot products, nonlinearities and output saturation. Observations convert from FP32 once at the input boundary. Initialization and mutation are separate from inference and retain FP32 arithmetic. Geometry, vision and integer energy/tick accounting retain their existing precision. At 1,024 creature slots, weights plus activations use 2.26 MiB per world instead of 4.52 MiB; retained vision history is still the largest allocation. On the saved 13M observations, the largest normalized control-output difference from FP32 was 0.00119; long-term trajectories can diverge.

Inputs contain four values for each of the 16 vision segments, in segment order:

| Channel | Network value |
|---|---|
| Proximity | `1 - distance / 24`, zero when empty |
| Size | `size / (size + 1)`, zero when empty |
| Is food | 1 for visible food, otherwise 0 |
| Is creature | 1 for a visible creature, otherwise 0 |

The two type flags are mutually exclusive; both are zero for an empty segment. Bearings remain in vision history for the overlay but are omitted from network inputs because segment position already supplies direction. The visible food's energy is retained in history but is not a brain input. Normalization happens during inference and never changes recorded observations.

The 16 hidden neurons use `tanh`. The steering output also uses `tanh`, mapping negative values to left turns and positive values to right turns, up to **90 degrees/s**. Movement uses `clamp(0.5 + raw_output, 0, 1)` and multiplies the creature's maximum speed. Zero means a complete stop, one means full speed, and turning still works while stopped. Velocity is rebuilt from heading, maximum speed and movement each tick, so slowing down never permanently reduces the creature's speed capability. Empty vision with zero biases gives straight travel at half speed.

Reset captures initial vision and evaluates the brain before the first movement. Each later evaluation follows vision capture and supplies the next tick's command. The vision overlay and displayed activations therefore describe the same observation. Pause freezes both; Step advances them once.

The selected creature's bottom display shows all 64 inputs in a 16-column × 4-row grid, 16 hidden neurons and both outputs with steering/movement gauges. Positive activations glow warm, negative activations blue, and zeros stay dim. Activity comes directly from GPU values, without simulated flicker or per-frame CPU readback. Labels, circles and gauges keep their proportions on resize: the content stays at its normal size in wide windows and scales down uniformly when space is limited. The world viewport leaves room for the strip, which does not receive pan, zoom or selection gestures.

Parameters and activations use fixed GPU blocks per creature: **1,074 float32 parameters** and **82 float32 activations**, about 4.52 MiB combined per world for the default 1,024 reserved slots. `sim.inspect_brain(world, local_index)` explicitly reads one creature's current inputs, hidden activations and outputs for diagnostics.

## Competitive comparison

The benchmark waits until **200,000 live ticks** before freezing a reference from the surviving population. Capture happens at the first submitted frame boundary at or after that threshold, and its actual tick is shown in the UI. No benchmark simulation runs during the wait. After capture, two benchmark worlds run alongside the live world: every **100,000 ticks**, equal cohorts of current brains and that fixed reference finish a competition. The two worlds swap starting positions; all competitors start with identical bodies, energy and age. Benchmark offspring inherit their side and brain without mutation. Each benchmark world starts with **150 food pieces**, adds the default **10 pieces/second**, and gives **5 energy units per piece**. This avoids the live world's large starting food surplus. Food conditions and lifespan settings stay fixed across trials even when live settings change.

Toggle **Show comparison graph** to plot the current cohort's share of **food energy collected throughout the trial**, against the live tick when its brains were sampled. **50% means equal energy collected; above 50% means the current cohort collected more than the reference.** All descendants contribute, including eaters that later die. Collection counts actual energy gained after any age reduction, excluding starting reserves and energy shared during splitting. Survivor counts remain secondary diagnostics. A trial with no food energy gained has no score. The graph and neuron strip can be visible together. The first result arrives around live tick **300,000**. The first trial compares the captured population with itself; later trials measure subsequent change against that same reference. Sampling and competition add noise. See [the comparison protocol](docs/COMPARISON.md) for sampling, score interpretation and limits.

Reset clears the reference and graph and starts a new 200,000-tick wait. `--no-benchmark` runs only the live worlds. After reference capture, select **Benchmark A** or **Benchmark B** in the world dropdown to watch either trial. Both support zoom, pan, creature selection, follow, vision and neural activations. Teal creatures are from the current population and orange creatures are from the settled reference; the inspector also names the cohort. Counts, age and tick display belong to the viewed world. Benchmark food/lifespan settings are shown read-only to preserve the fixed comparison. **All live worlds** keeps the live overview separate. Starting the next trial clears benchmark selection/follow and fits the world again, because it contains a new cohort.

## Snapshots

The [13-million-tick milestone](checkpoints/13-million-ticks/README.md) is preserved
in this repository with its matching source archive and restore instructions. Its
latest completed comparison collected 90.99% of the combined food energy against
the frozen 200,000-tick baseline.

The app saves a restorable checkpoint every **100,000 live ticks** under ignored `artifacts/snapshots/`, keeping the **latest five** automatic files. **Snapshot now** creates a checkpoint in that same rolling set. **Restore latest snapshot** restores it and pauses the app so you can inspect it before continuing. Saves happen at submitted frame boundaries; crossing the interval within a frame can save a few ticks later. A finite run also saves if its final frame reaches a checkpoint boundary.

Checkpoints include creature identities, ages and energy, food, world counters, all network weights and activations, complete vision history, spatial grids, settings and the exact tick. They also preserve whether the benchmark is still waiting, its capture threshold and actual reference tick, the frozen reference, in-progress trials, food-intake counters and graph results. Restoring a waiting run continues toward its original threshold; restoring after capture retains the exact reference and trial without recapturing. Older benchmark protocols are unsupported; start a new run for the new baseline policy. Camera and creature selection reset; the saved speed is retained. Files are compressed NPZ archives written through an atomic replacement, without pickle. Same-GPU continuation under the current rules is tested exactly; cross-driver bit-identical replay is not promised.

Snapshot format 5 stores networks in FP16. Formats 1–4 migrate network buffers, including the frozen reference and active trial, when loaded; the source archive stays unchanged. The existing benchmark protocol check still applies. Quantizing a legacy network changes its future trajectory, so continuation across this precision change is not bit-identical. Nonfinite or out-of-range network values are rejected.

```powershell
# Load a particular checkpoint instead of the most recent automatic file.
.\.conda\python.exe -m worms --resume artifacts/snapshots/auto-<timestamp>-tick-<tick>.npz

# Headless restore continues immediately; an explicit speed overrides the saved one.
.\.conda\python.exe -m worms --headless --resume latest --frames 100 --speed 100

# Use a separate checkpoint folder, or disable automatic snapshotting.
.\.conda\python.exe -m worms --snapshot-dir artifacts/experiment-a
.\.conda\python.exe -m worms --no-snapshots
```

Saving synchronizes and reads GPU state, so it briefly pauses a run. A live-only checkpoint with 1,024 reserved slots, 64 living creatures and 256 history frames measured about 5.2 MB and 0.77 s to save on this machine; a full session includes the two comparison worlds and costs more. Copy a file outside the rolling `auto-*.npz` set to retain it permanently. `App.snapshot(path)` also writes a named file that automatic retention does not remove.

## Performance

Profile a checkpoint with hardware timestamps for every compute dispatch, CPU
recording costs, and simulation throughput:

```powershell
.\.conda\python.exe -m worms.profile --resume checkpoints/13-million-ticks/snapshot.npz --warmup 2000 --samples 30 --output artifacts/profile-comparison.json
.\.conda\python.exe -m worms.profile --resume checkpoints/13-million-ticks/snapshot.npz --no-benchmark --warmup 2000 --samples 30 --output artifacts/profile-live.json
```

The saved benchmark runs by default; `--no-benchmark` measures the live world alone.
Normal runs disable GPU validation; add `--gpu-debug` to the app or profiler to
enable D3D12 and RHI checks. GPU tests enable these checks explicitly. Profile
reports record the mode, since validation adds substantial CPU overhead.
The checkpoint file stays unchanged and automatic snapshots are disabled during
profiling. Both measurements replay the same state and warmup: throughput uses
frame-boundary timestamps, while a separate replay gathers mean, median, p95 and
maximum dispatch times. Omit `--resume` for a fresh scene, adding `--benchmark` to
include the comparison pair.

For sustained evolution throughput, record three minutes at 100 steps per frame:

```powershell
.\.conda\python.exe -m worms.throughput --resume checkpoints/13-million-ticks/snapshot.npz --seconds 180 --interval 10 --steps-per-frame 100 --output artifacts/throughput-comparison.json
.\.conda\python.exe -m worms.throughput --resume checkpoints/13-million-ticks/snapshot.npz --seconds 180 --no-benchmark --output artifacts/throughput-live.json
.\.conda\python.exe -m worms.throughput --resume checkpoints/13-million-ticks/snapshot.npz --seconds 180 --no-benchmark --no-render --output artifacts/throughput-live-no-render.json
```

This capture includes trial restarts, timestamp reads and periodic population
counters in elapsed throughput. It preserves the source checkpoint and disables
automatic saves. JSON retains every frame, ten-second windows, CPU/GPU timings,
world counters, trial transitions and source/runtime metadata. Drawing can be
disabled during both warmup and measurement. By default this uses the application's
bounded 25-step chunks, allowing CPU recording to overlap GPU execution, followed
by one draw and a final wait per headless frame. Add `--synchronized` for the old
single-command-buffer comparison, or `--chunk-size N` to vary the chunk size.
GPU reports distinguish summed batch durations from the first-to-last timestamp
span, which includes queue gaps. These headless rates exclude window presentation
and checkpoint I/O; they are not window FPS.

The 13-million-tick snapshot is the current repeatable reference. Historical
100,000-tick results used a smaller population and older simulation rules; its
benchmark protocol is no longer loadable by the current app. Brain evaluation
and vision remain the largest named GPU stages. The speed dropdown means steps
per rendered frame, not a guaranteed real-time multiplier.
See [the performance report](docs/PERFORMANCE.md) for measurements and reproduction
commands, and the [living optimization plan](docs/OPTIMIZATION_PLAN.md) for progress.

## Checks

```powershell
.\.conda\python.exe -m pytest
.\.conda\python.exe -m pytest --run-gpu
.\.conda\python.exe -m ruff check .
.\.conda\python.exe -m ruff format --check src tests
```

CPU checks cover layout, cursor anchoring, pan speed, click/drag handling and follow handoff. GPU checks also cover seeded worlds, motion/rendering, spatial-grid rebuilding, exact food-energy conservation, contested ownership, expanded/wrapped contact queries, food-slot reuse, growth and reset. Vision checks cover all segments, range and angular boundaries, nearest-object occlusion in both directions, tie-breaking, revealed objects, heading, wrapping, world isolation, next-tick growth visibility, retained consumed-food observations, history rollover, pause/reset and deterministic replay. Explicit GPU runs fail normally if graphics is unavailable. Both headless and finite windowed runs have been verified on the RTX 3050 Ti Laptop GPU.

Lifecycle checks cover size-scaled reserves and costs, fractional depletion, starvation, conserved splitting into two new identities, independent offspring mutations, inheritance across generations, capacity limits, slot reuse, newborn history, world isolation and seeded replay.

Aging checks cover full intake before the lifespan limit, death at the limit despite stored energy, optional fading, fresh ages and intake for both offspring, live slider settings and Reset.

Neural checks compare all activations against a NumPy reference and cover independent seeded weights, unchanged parameters during runs, input normalization, exact stop/full movement, bounded turning while stationary, speed preservation, observation timing and pause/reset. Render tests verify selected-world indexing, signed neuron colours, both motor gauges and paused stability; CPU interaction tests keep the bottom strip outside the navigable world.

The installed runtime emits `Cannot enable D3D12 Agility SDK: Calling SetSDKVersion failed.` at startup. Device creation, shaders, rendering, and GPU tests still pass with native debug layers and RHI validation enabled; the warning is not suppressed.

See [AGENTS.md](AGENTS.md) for development conventions and [visual references](docs/VISUALS.md) for the intended appearance. Window and UI management use SlangPy's [built-in application helper](https://github.com/shader-slang/slangpy/blob/v0.43.1/src/sgl/app/app.cpp).
