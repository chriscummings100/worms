# Worms

A small Python + SlangPy artificial-life project. The application runs independent GPU worlds with creatures driven by random neural networks, clustered food, a windowed viewer, and a headless path for tests and image dumps.

## Run

From PowerShell in the repository root:

```powershell
.\.conda\python.exe -m pip install -e ".[dev]"

# Open the application (1 world, 64 creatures).
.\.conda\python.exe -m worms

# Advance exactly 120 fixed ticks and save the final scene.
.\.conda\python.exe -m worms --headless --frames 120 --output artifacts/worlds.png

# Change the initial world and creature counts.
.\.conda\python.exe -m worms --worlds 6 --creatures 128
```

The window has a Speed dropdown (Paused, 1x, 2x, 5x, 10x, 20x, 50x and 100x), Step, Reset, and an all-worlds/single-world selector. Speed sets the number of simulation updates per rendered frame. Escape or the close button exits. `--frames N` also works in windowed mode for a finite run. PNG exports contain the scene without UI. A headless run defaults to 120 frames and `artifacts/worlds.png`.

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
- `artifacts/`: generated outputs, ignored by Git.

Each world has a fixed block of creature slots in one shared GPU buffer, with **four times its initial population reserved** (64 initial creatures and 256 slots per world by default). `Sim.creatures_per_world` is the initial count; `Sim.creature_capacity` is the allocation stride. Dead slots are reused without moving survivors. Each newborn receives a fresh ID, parent ID, generation and birth tick. The world records its current live count and cumulative births/deaths, shown in the controls. Seeded Reset reproduces the initial state. The action/target fields are placeholders for later simulation work.

One world unit is one metre. Worlds are **512 x 512 m**. A default creature's full body length averages **1 m**, with seeded variation from **0.8 to 1.2 m**; body width is 63% of its length. Body size is independent of population and world size. The 64-creature default is sparse: one creature per 4,096 square metres. At whole-world scale creatures are tiny specks; zoom in to inspect them at a useful scale.

One update applies each creature's steering and throttle for 1/60 second and wraps both coordinates at the world boundaries. Maximum speed is initialized to 8–16 m/s and stored separately from actual velocity. Rendering reads the same GPU state, including copies at wrapping boundaries. Each frame runs `speed` simulation updates in a loop, then renders once; 100x means 100 updates per render. Windowed and headless runs use the same loop, defaulting to one update per frame. Pause freezes the simulation, Step advances exactly once regardless of speed and remains paused, and Reset restores the seed without an extra tick that frame. Reset keeps the selected speed.

## Food

Each world reserves **8,192 food slots**, starts with **2,048 pieces**, and adds **1.5 pieces per simulated second** toward **4,096 live pieces** (about 22.8 simulated minutes from reset if nothing is eaten). Each piece defaults to **five energy units**. The **Food energy** slider adjusts this from 0.25 to 10 units while running or paused, across all worlds. Existing uneaten pieces and future growth use the new value; positions and live counts stay unchanged. Reset keeps the chosen value. Previously recorded vision remains unchanged. These defaults live in `sim.py`; small experiments can override `Sim`'s food capacity, initial amount, target and growth rate without adding application configuration.

The **Food respawn** slider controls additions from **0 to 256 pieces per simulated second per world**; zero stops additions. Changes apply on the next simulation tick, and Reset keeps the chosen rate. The food counter shows the current live pieces against the target for the selected world, or totals across all worlds in the overview. It refreshes with the population counters five times per second.

Food shares one GPU allocation across all worlds, addressed as `world * food_capacity + slot`. Each 12-byte record stores position and integer energy; zero energy marks an empty slot. The world stores its live count, allocation cursor and spawn sequence. Growth reuses empty slots without moving surviving food, reallocating or reading back simulation state. A reused slot gets a fresh location from the spawn sequence. Pause, Step and seeded Reset apply to food as well as creatures.

Placement accepts random candidates according to a two-octave periodic Perlin density field, giving rich clusters and sparse gaps. The field wraps continuously at world edges. Candidate work is bounded to 128 attempts per piece, with the densest candidate as a fallback. Each world's sequence is independent of other world counts, creature population and growth batching. New food samples the same density field at new positions.

Food is drawn as lime specks with a **0.18 m radius** and a subtle halo. A minimum visual radius of 0.7 pixels keeps clusters readable when zoomed out. Zoom, pan and follow use the same camera and wrapping as creatures. Rendering visits the reserved slots and skips empty ones, so eating an early slot cannot hide surviving food later in the array.

## Contact and collection

Each tick applies stored brain commands, moves creatures, rebuilds their spatial grid, collects food, applies energy credits, settles energy costs/deaths/splits, rebuilds the creature grid, grows new food, records vision, and evaluates brains for the next movement. New food becomes edible on the next tick and is visible immediately. Each world's grid has **128 x 128 cells of 4 m**. Heads are cleared in parallel, then one GPU thread per world inserts its living creatures serially into linked lists using ordinary writes. Worlds build independently; grid insertion has no atomics or per-cell occupancy cap.

Each food item queries nearby grid cells and tests toroidal circle contact. A creature's collision radius is its larger body radius, a conservative approximation of its visual ellipse. Normally the query visits 3 x 3 cells; larger bodies expand the search using that world's maximum radius. Queries never mix worlds or visit a wrapped cell twice. This avoids a global food-by-creature scan; dense local crowds can still require many comparisons.

Touching food removes the whole piece. Young creatures receive all its energy; older creatures receive a declining fraction as described below. One thread owns each food item and picks exactly one touching eater using a seeded priority, independent of grid-list traversal order. Integer atomic additions accumulate food credits, then a separate pass scales each creature's total by its age and applies the result. Scaling uses 64-bit integer arithmetic and rounds down once per tick to whole energy quanta. Unabsorbed energy is lost, including the whole piece at zero intake. Creatures start with **100 energy units per metre of body length**, with **1,024 integer quanta per unit**. Per-tick credits are 32-bit with a bounded food allocation; persistent creature energy is 64-bit and has no gameplay cap. The inspector displays energy in units.

Combat, invincibility and lifetime learning remain future work in the [project plan](docs/PROJECT_PLAN.md). `baseline.proposed.toml` remains planning data, not a supported application configuration. The earlier standalone smoke renderer has been replaced by this shared application path.

## Energy and splitting

For full body length `s` metres and actual speed `v` metres/second:

- Initial energy is `100 × s` units; the split threshold is twice that initial amount.
- Depletion is `s × (0.05 + 0.001 × v²)` units/second, charged after food collection.
- Energy is stored in integer quanta, with fractional depletion carried between ticks. Reaching zero kills the creature.

A stationary creature still spends energy. At these starting rates, initial reserves last about 33 simulated minutes at rest, 15 minutes at 8 m/s, or 5.4 minutes at 16 m/s without food. Both reserves and costs scale linearly with size, so these durations are independent of body length. The two cost constants live in `lifecycle.slang`.

Creatures default to a **100,000-tick lifespan**, with full food intake until they die at that age. Age counts simulation ticks since birth or the creature's most recent successful split. The **Decline age** slider defaults to **100,000 ticks** and **Fade length** to **0 ticks**, giving an immediate end of life. A positive fade length instead adds that many ticks of linearly declining food intake; death occurs at decline age plus fade length. Both sliders cover 0–100,000 ticks, apply to existing creatures on the next tick, and survive Reset. The inspector shows age and current intake percentage.

The age limit kills a creature even when it has stored energy, clearing its reserves and recording one death before reproduction is considered. Starvation can still kill it earlier. A successful split resets both parent and child to age zero with full food intake and a fresh lifespan. A split blocked by a full world does not reset age. The parent keeps its identity, network and vision history; its original birth tick is stored separately from the age reset tick. Reset returns the initial population to age zero. Vision continues to report each food piece's stored energy before any age adjustment.

At the threshold, a creature splits if its world has a free slot. Parent and child share the remaining energy equally (the parent keeps an odd quantum), preserving total energy. The child inherits the same body and maximum speed, copies the parent's network with small mutations, and starts with fresh vision history. Parent and child move to opposite sides of the split point, leaving a gap of one quarter of their body length. Each turns 90 degrees in the opposite direction, facing outward along their separation; positions wrap at world edges. The child begins moving next tick. Each parent can split at most once per tick; newborns cannot split on their birth tick. A full world delays splitting without charging energy or discarding surplus reserves.

On each successful split, every child weight and bias independently has a **1% chance of an additive Gaussian mutation with standard deviation 0.02**—about 11 of the 1,074 parameters per child on average. The parent's network stays unchanged. These two constants live in `lifecycle.slang`. Mutation runs on the GPU and is keyed by the simulation seed, world and birth counter, so siblings get independent changes, reused slots get fresh mutations, and Reset replays births on the same GPU. A descendant inherits its parent's already evolved network before applying its own mutations.

Successful feeding and reproduction can now spread new network variants, while depleted creatures die. Food availability, luck and the population cap also affect survival; mutation does not guarantee improvement. Body traits remain unchanged and there is no lifetime learning. The inspector shows energy, split threshold and lineage. Selection/follow is cleared when its creature dies or its slot is reused.

## Vision

Every living creature senses **16 equal segments across a 180-degree forward arc**, out to **24 m**. Heading determines the direction of motion and can turn independently while stationary. Each segment sees only the **closest object**, comparing food and other creatures together by centre distance using the shortest wrapped displacement. That object hides every farther object in the same segment: food can hide a creature and a creature can hide food. This is occlusion at segment resolution, not a geometric raycast. Self, dead creatures and empty food slots are excluded. The range and outer arc boundaries are inclusive. Exact-distance ties prefer food, then the lower buffer index within the same kind, independent of grid traversal order.

Each segment retains one 20-byte observation:

| Fields | Meaning |
|---|---|
| `distance`, `bearing` | Visible object's centre distance in metres and bearing in radians relative to heading |
| `size` | Food diameter (0.36 m) or creature's full body length, in metres |
| `food_energy` | Visible food's own energy in units; zero for a creature |
| `kind` | Integer: 0 for empty, 1 for food, 2 for creature |

A distance of `-1` and kind of zero mean nothing was seen; the other fields then hold zero. Hidden objects contribute no properties or energy to the observation. A second 4 m spatial grid indexes food, built serially per world after collection and growth. Each observer queries nearby cells in both grids rather than scanning whole populations.

The GPU retains **256 complete observations per creature** in a ring buffer, about **4.3 simulated seconds** at 60 ticks/s. Each frame includes all 16 segments plus the observer's position, heading and tick. The default 256 reserved creature slots use 21 MiB for observations and frame headers. Reset clears history and records one initial observation; Pause adds none, and Step adds one. Dead creatures produce invalid frames while their older observations remain until overwritten; reusing a slot clears its entire history before recording the newborn's first observation. Rendering and sensing keep these buffers on the GPU; `sim.inspect_vision(world, local_index, age=0)` reads one retained frame only when explicitly requested for diagnostics.

Selecting a creature displays its latest captured arc with subtle segment boundaries. A segment's tint indicates the visible object's kind; lime rings mark visible food and orange rings mark visible creatures. Hidden objects receive no vision marker. The overlay uses the recorded observations and pose, including across world seams. Zoom in to see the arc clearly.

History is stored at full angular and temporal resolution. The brain currently uses only the newest observation. Reducing older samples into fewer brain inputs will be a separate pass, recomputed from this retained history.

## Neural control

Each creature has its own **64 → 16 → 2 feedforward MLP**, initialized with independent seeded Xavier-uniform weights and zero biases. Each creature's parameters remain fixed throughout its life; offspring inherit them with small mutations. Reset reproduces the initial population. Initial reactions to food and other creatures are arbitrary, with subsequent evolution driven by survival and reproduction.

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

Parameters and activations use fixed GPU blocks per creature: **1,074 float32 parameters** and **82 float32 activations**, about 1.13 MiB combined for the default 256 reserved slots. `sim.inspect_brain(world, local_index)` explicitly reads one creature's current inputs, hidden activations and outputs for diagnostics.

## Checks

```powershell
.\.conda\python.exe -m pytest
.\.conda\python.exe -m pytest --run-gpu
.\.conda\python.exe -m ruff check .
.\.conda\python.exe -m ruff format --check src tests
```

CPU checks cover layout, cursor anchoring, pan speed, click/drag handling and follow handoff. GPU checks also cover seeded worlds, motion/rendering, spatial-grid rebuilding, exact food-energy conservation, contested ownership, expanded/wrapped contact queries, food-slot reuse, growth and reset. Vision checks cover all segments, range and angular boundaries, nearest-object occlusion in both directions, tie-breaking, revealed objects, heading, wrapping, world isolation, same-tick growth, retained consumed-food observations, history rollover, pause/reset and deterministic replay. Explicit GPU runs fail normally if graphics is unavailable. Both headless and finite windowed runs have been verified on the RTX 3050 Ti Laptop GPU.

Lifecycle checks cover size-scaled reserves and costs, fractional depletion, starvation, conserved splitting, sparse child mutations, unchanged parents, inheritance across generations, capacity limits, slot reuse, newborn history, world isolation and seeded replay.

Aging checks cover full intake before the lifespan limit, death at the limit despite stored energy, optional fading, age since birth or splitting, parent rejuvenation, fresh newborn intake, live slider settings and Reset.

Neural checks compare all activations against a NumPy reference and cover independent seeded weights, unchanged parameters during runs, input normalization, exact stop/full movement, bounded turning while stationary, speed preservation, observation timing and pause/reset. Render tests verify selected-world indexing, signed neuron colours, both motor gauges and paused stability; CPU interaction tests keep the bottom strip outside the navigable world.

The installed runtime emits `Cannot enable D3D12 Agility SDK: Calling SetSDKVersion failed.` at startup. Device creation, shaders, rendering, and GPU tests still pass with native debug layers and RHI validation enabled; the warning is not suppressed.

See [AGENTS.md](AGENTS.md) for development conventions and [visual references](docs/VISUALS.md) for the intended appearance. Window and UI management use SlangPy's [built-in application helper](https://github.com/shader-slang/slangpy/blob/v0.43.1/src/sgl/app/app.cpp).
