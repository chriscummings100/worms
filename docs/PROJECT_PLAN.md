# Worms: practical project plan

24 October 2026 · Python + SlangPy/Slang · Windows · planning specification

**Current food implementation (24 October):** food pieces occupy a fixed GPU block per world, with periodic Perlin-density placement and gradual additions at new locations. Defaults are 2,048 initial pieces, a 4,096-piece live target, 1.5 pieces/s and 8,192 reserved slots. A 4 m spatial grid is built serially within each world, with worlds running independently. Food contact removes the whole piece and transfers an age-dependent fraction of its energy to one touching creature. Each piece defaults to five energy units; a live UI slider adjusts existing food and future growth from 0.25 to 10 units without resetting the simulation; empty food slots are reused at fresh locations. Energy uses integer quanta. See the [README](../README.md#contact-and-collection) for implemented behavior.

**Current vision implementation (24 October):** each living creature records 16 segments over a 180-degree forward arc and 24 m range. Each segment sees only its nearest object across food and creatures; that object hides farther objects in the same segment. Its distance, bearing, size, kind and food energy are stored, with no contribution from hidden objects. A 256-frame GPU ring retains all segments plus observer pose and tick at full resolution. The selected creature's overlay shows this captured observation. Future brain inputs will reduce angular and temporal detail from retained history afresh each tick, allowing reduction experiments without clearing history. History reduction is not implemented. This supersedes the older compressed-memory/EMA proposals below; see [Vision](../README.md#vision) for the current schema and behavior.

**Current neural implementation (24 October):** each creature has an independently seeded, fixed-weight 64 → 16 → 2 MLP. Four channels per current vision segment supply normalized proximity and size, plus separate binary food and creature flags; bearings, food energy and older samples are omitted. Hidden activations and steering use tanh; movement clamps `0.5 + raw_output` to [0, 1]. Steering is bounded to ±90 degrees/s, movement scales an independent maximum-speed trait, and creatures can turn while stopped. Brains are evaluated after vision capture for the next movement tick. A bottom display shows the selected creature's actual GPU activations and both output gauges. There is no lifetime learning. Reproduction copies the parent's network, then each child weight and bias has a 1% chance of additive Gaussian mutation with standard deviation 0.02. The parent is unchanged; mutation streams are seeded per world and birth. This replaces the older network/action proposals below for the current application; see [Neural control](../README.md#neural-control).

**Current lifecycle implementation (24 October):** size is full body length in metres, with linear scaling. Starting energy is `100 × size`, splitting occurs at twice that amount, and depletion is `size × (0.05 + 0.001 × actual_speed²)` energy/s. Fractional quanta carry across ticks. Default lifespan is 100,000 ticks with full food intake and a zero-length fade. Live sliders adjust the decline age and optional fade duration; the creature dies at their sum regardless of stored energy, before reproduction is considered. Successful splits reset both parent and child to age zero with full intake and a fresh lifespan; blocked splits leave age unchanged. Costs settle after food, zero energy kills, and eligible parents divide their remaining energy with a child if a slot is free. Children inherit the same body and speed, with sparse mutations to inherited neural weights and biases. Parent and child separate symmetrically with a quarter-body-length gap, turn 90 degrees in opposite directions and face away from one another. Each world reserves four times its initial population, reuses dead slots with fresh identities/history, and tracks births/deaths. This supersedes the older size-squared costs, age death, maturity, birth charges and fixed newborn-energy proposals below; see [Energy and splitting](../README.md#energy-and-splitting).

**Agreed combat direction, not yet implemented:** creature contact chooses a winner randomly based on relative size, transfers energy according to the winner's size (limited by the loser's reserves), and kills a loser depleted to zero. Both participants then become invincible for N simulation ticks. The older finite-patch economy and explicit-attack proposals below are historical design hypotheses, superseded where they conflict with these agreed mechanics.

## 1. The goal and the first useful version

Build a visually beautiful, inspectable artificial ecosystem: simple organisms forage, spend energy, compete, reproduce and gradually change across generations. Several independent worlds occasionally exchange genomes. Evolution supplies each creature's starting brain and physical traits; optional neural learning adapts that brain during its life.

![User-supplied visual starting point](../assets/user-reference-original.png)

Keep the anatomy at this level: ellipsoids with eyes and a few simple limbs. Make the final view richer through translucent materials, soft internal shading, restrained glow, gentle particles and motion, rather than complicated creatures. The reference establishes shapes and composition; a later material pass establishes polish. See [visual direction and image provenance](VISUALS.md).

**The first playable milestone is one pond with 256 organisms, 32 food patches, a tiny brain, visible energy and births, and a pause/inspect view.** Start with evolution only. Add learning after accounting and behavior are trustworthy; add extra worlds after one world is understandable.

Scope boundaries: flat rectangular worlds that wrap left ↔ right and top ↔ bottom, circular contact proxies, asexual reproduction, fixed network architecture, fixed maximum sensing radius, no obstacles or fluid solver. There are no walls or blocking corners, so brains need no edge-avoidance behavior. Rendering may look 2.5D. Limbs and tails are cosmetic. Defer recurrence, topology evolution, sexual crossover, differentiable physics, multi-GPU execution and elaborate predators/carcasses.

## 2. Starting settings and stop conditions

| Item | Starting proposal | Grow only after measurement |
|---|---|---|
| World | One flat wrapping rectangle, initially 512 × 512 m | Independent width/height; four independently seeded worlds |
| Population | 256 alive, 512 allocated slots | 1,024 alive/world, 2,048 slots/world |
| Food | 32 stationary finite patches/world | 64–128, then slow relocations |
| Simulation | Fixed 1/60 s tick | Headless batches of fixed ticks |
| Policy | New action every 4 ticks; hold between decisions | Compare stride 1, 4, 8 |
| Brain | 128 inputs → 16 tanh units → 8 outputs | Hidden width 32 if useful |
| Inputs | 16 current bins + 8 fast + 4 slow; 4 channels each | Keep shapes fixed during evolution |
| Learning | Off initially; then SGD, 32 recorded decisions, horizon 32 | Ordered sweeps below |
| Lifecycle | Every 16 ticks, maximum 8 births/world/pass | Profile capacity and extinction |
| Migration | Off initially; later 1 genome/world/3,600 ticks | Compare none and rare exchange |
| Display | 1280 × 720, one focused world | World thumbnails, higher resolution |

The machine reports an **NVIDIA GeForce RTX 3050 Ti Laptop GPU** and **Intel Iris Xe Graphics**. Select and log the intended adapter explicitly; validate its usable memory and capabilities at runtime. No VRAM size or performance is assumed from the name. Target a comfortable 60 Hz presentation at the small baseline, but treat that as an acceptance goal, not a forecast. Reduce effects or live population before expanding architecture.

[baseline.proposed.toml](../baseline.proposed.toml) records these hypotheses. It is not yet a supported application configuration. Pin the first tested SlangPy wheel and record Python, Slang/compiler, backend, driver and device versions in every run manifest. Start with D3D12 for shared simulation/render resources on Windows, with a Vulkan fallback if the minimal smoke test requires it. PyTorch and CUDA interop are optional, not prerequisites. Current SlangPy provides wheels and graphics/compute backends; verify the selected version against the [official installation guide](https://slangpy.shader-slang.org/en/latest/).

## 3. Build in small stages

Effort estimates are focused developer days, not delivery promises. Stop at a completed stage if it is already enjoyable. Do not begin a later stage to avoid fixing a failed earlier gate.

| Stage | Deliverable | Done when | Rough effort |
|---|---|---|---|
| 0 — Prove the loop | Create device, dispatch a small Slang kernel, draw instanced ellipses from its buffer; GPU timing | Correct adapter selected, debug validation clean, data stays on GPU between update and draw | 0.5–1 day |
| 1 — Living postcard | Pond, food glow, 256 moving creatures; pause, speed, seed, selected-creature panel | Attractive reference-like view; fixed seed reproduces a short diagnostic run; frame cost measured | 1–2 days |
| 2 — Honest ecology | Spatial grid, finite food, costs, death, automatic births, mutable genomes | Energy ledger balances; no double feeding/slot reuse; scripted forager outlives random policy; extinction recorded | 2–3 days |
| 3 — Evolution and conflict | Evolve initial weights/FOV/size/speed; add energy-damage combat; basic lineage metrics | Several generations run; trait bounds hold; crowding/attack tests pass; saved genomes replay | 1–2 days |
| 4 — Learning proof | Pure policy loss, CPU gradient reference, future-energy tape in simple foraging arenas | Correct gradients and reward indexing; learning beats matched no-learning controls in repeated seeds | 2–4 days |
| 5 — Combined ecosystem | Enable lifetime learning in ecology; compare inheritance blends | No stale-policy/life updates; energy and offspring metrics reported separately; learning benefit survives held-out layouts | 2–3 days |
| 6 — Islands and polish | Four worlds, rare genome migration, translucent material pass, focused profiling | Migration ledger correct, no world leakage, effects remain readable and fit measured budget | 1–3 days |

**First work session:** complete stage 0, draw five creatures and three food clusters matching the supplied image, then scale the same draw to 256. Add an energy bar and a headless fixed-step loop before designing more UI. **First weekend cut:** stages 0–2; disable combat, migration and learning. This is already a useful toy.

## 4. Simulation rules and an auditable energy economy

### Time, geometry and food

Use a fixed timestep independent of rendering. Each world is a flat rectangle with independent width W and height H. Crossing the right side continues at the left; crossing the bottom continues at the top. Crossing a displayed corner wraps both coordinates. Heading, velocity, energy, policy and memory continue normally. The initial 512 × 512 m preset is just a square instance of this rectangular domain; non-square worlds use the same code. Initial contact uses body radius; visual ellipsoids do not require oriented collision tests. Creatures can overlap in the first version. Food patches are circular stores with maximum capacity and explicit regeneration per second. Food specks shown inside a patch are decorative samples of the store, not thousands of simulated resources.

**Wrapping contract:** store positions in [0, W) × [0, H) using positive modulo. For any relative displacement d_x, use d_x − W × floor(d_x/W + 0.5), and similarly for y with H. Use this shortest wrapped displacement consistently for distance, bearing, FOV/bin assignment, feeding, attacks and offspring placement. Keep sensing and contact ranges below half the smaller world dimension initially, and define deterministic ties at exactly half an extent. World bounds are a storage/display convention, not observations or collision surfaces.

Calculate movement costs from executed velocity/displacement before coordinate wrapping, so crossing a seam cannot appear to be a world-sized movement. For rendering, interpolate using shortest wrapped motion, and keep cosmetic trails in locally unwrapped coordinates or split them at seams. Draw translated copies of organisms, food, limbs, trails and glow where their visible footprint overlaps an opposite boundary (both axes at a corner). These are draw instances of the same entity: never duplicate its food, energy, sensor signal or interactions. Fading sensory history should see a neighbor crossing a join as continuous motion, with no artificial turn or memory reset.

A creature chooses at most one reachable patch per tick. Feeding is automatic when in contact and requested intake is positive. The patch's owner distributes available stock fairly among claimants, capped by per-tick intake and snapshot energy headroom. Use equal-share allocation with deterministic rotating remainders initially. Patch growth occurs after allocation and becomes available next tick. Empty food can regrow; a full patch's excess growth is not silently counted as stored energy.

### Suggested units and costs

One spatial unit is one metre. Creature size s = 1 means a full body length of 1 m (major radius 0.5 m), excluding cosmetic limbs. Physical body size is independent of population and world dimensions. The current motion demo initializes lengths in [0.8, 1.2] m; the trait bounds below are proposals for later evolution. The 512 m world extent replaces the original 256-unit display prototype.

At size s = 1: energy capacity 100, newborn energy 30, base maintenance 0.6 energy/s, patch capacity 80, regeneration 2 energy/s/patch, maximum intake 8 energy/s. These are tunable starting values. With 32 patches, the external food inflow ceiling is 64 energy/s; 256 baseline adults alone consume 153.6 energy/s before motion. This deliberately exposes oversubscription: first validate with 64–96 adults or raise regeneration to 6 energy/s for the 256-creature display baseline. Record the chosen regime; do not hide extinction by respawning.

Start the main 256-creature preset at **6 energy/s/patch**, not the scarce-food example. Capacity, initial stock, crowding and reproduction mean this arithmetic is only a feasibility check, not proof of a stable population.

Proposed trait scaling: E_cap = 100 s²; maintenance grows with s²; movement cost grows with s² × speed²; wider FOV and higher maximum-speed capability have small maintenance costs even when unused. Clamp size to [0.6, 1.6], speed capability to [3, 9] units/s, FOV to [60°, 300°]. Keep sensing range fixed at 24 units initially. Coefficients must be calibrated with scripted controls so no trait is a free improvement. Larger size can improve intake/contact/damage but also costs more. Do not claim these laws are biologically accurate.

Every tick tracks: previous creature energy + food intake − maintenance − executed movement cost − attack effort − received damage − birth allocation/cost − capacity spill − residual energy dissipated on age death. A creature dies at nonpositive energy or a configured maximum age (start at 300 simulated seconds). On age death, explicitly debit its remaining energy before zeroing the slot; carcass transfer is deferred. Clamp losses at available energy when updating the ledger: requested damage can exceed remaining energy, actual dissipated energy cannot. Initial stock and new food growth are external inputs. Initial seeding is recorded once.

For early conservation tests use fixed-point integer energy (for example 1,024 quanta/unit). Carry fractional remainders for all continuous rates: regeneration, maintenance, movement, sensing/capability costs and intake. Otherwise small costs disappear or introduce unintended trait thresholds. Positions and neural math remain floating point. Check int32 headroom for every bounded per-world sum; aggregate large telemetry totals using wider CPU arithmetic. Render energy continuously from these units.

### Combat and reproduction

Combat starts as costly, short-range energy damage. An attacker names one target within reach; every victim gathers valid incoming attacks. Attacks chosen from the tick snapshot resolve simultaneously, even if an attacker dies during that settlement. Damage dissipates energy; there is no immediate transfer to the attacker. This supports competition without inventing energy or racing to claim a victim. A carcass resource and predation can be added later through the same finite-resource allocation rules.

Automatic reproduction is the default: a mature, living creature above 0.8 E_cap may fund a child, provided it retains 0.25 of its own capacity afterward. Start maturity at 10 simulated seconds and cooldown at 10 seconds. Child energy is 0.3 of its own capacity, paid by the parent, plus an explicit 5-unit birth cost scaled by parent size². Recheck affordability after child morphology is known. The cooldown and fixed birth-rate cap prevent a population explosion. No free slot means no birth and no charge. Newborns become visible only after initialization completes.

Each accepted birth transfers energy once; mutation changes genes, not resource balances. Spawn near the parent and wrap the child's coordinates into the rectangle; start with empty observation history. Report live births, children surviving a fixed age and lineage persistence separately. Natural selection follows survival and reproduction; there is no global optimizer awarding an arbitrary fitness score each tick. A population cap can itself alter selection, so log rejected births and capacity occupancy.

## 5. GPU execution and synchronization

![Five-pass routine tick with occasional work](../assets/tick-pipeline.png)

Interpret **mostly one creature dispatch** as one main sense/history/policy/propose pass. The initial safe implementation uses five routine compute dispatches, plus lifecycle, learning bookkeeping/updates and rendering when enabled. Sacrificing correctness to force one total dispatch would undermine the experiments.

| Pass | Reads | Writes / ownership |
|---|---|---|
| 1. Clear | Frame constants | Grid counts, food credits, temporary flags |
| 2. Build grid | Immutable current state S_t | One world-keyed cell entry/creature; atomic slot reservation |
| 3. Creature behavior | S_t, food_t, grid, own weights/history | Own observation/history, action, movement proposal; one food and attack intent |
| 4. Food settlement | S_t, grid, intents, food_t | One patch owner updates that patch and its claimant credits |
| 5. Finalize | S_t, proposals, credits, attack intents, grid | One owner writes its entire S_(t+1), resolved energy and alive flag |

Sense and decide only on policy-decision ticks; move and settle costs every physics tick using held actions. Contact/attack intents are refreshed from the current snapshot every tick. All interaction ranges use pre-movement positions in S_t; new positions take effect next tick. This is a deliberate one-tick geometric convention. The last pass gathers attacking neighbors whose target is this creature, avoiding concurrent writes to victim energy. Food-credit writes are unique because a creature requests only one patch.

Record dependent passes in an ordered command encoder and use the selected backend's resource/UAV barriers or validated automatic transitions. Dispatch ordering alone is not a substitute for required memory visibility. There is no general cross-workgroup barrier inside a compute dispatch; do not spin-wait for peers or treat a thread-group barrier as global. Keep S_t immutable until all consumers finish. Publish S_(t+1) only after lifecycle completion. Use fences if compute and rendering later run on different queues; initially use one queue.

SlangPy exposes explicit compute kernels, buffer binding and dispatch recording. Use the [official compute tutorial](https://slangpy.shader-slang.org/en/latest/src/tutorials/compute_shader.html) and [API reference](https://slangpy.shader-slang.org/en/latest/src/api_reference.html) for the pinned release. Treat the table as architecture, not pasted API code. Record all helper dispatches in profiling so the cost is visible.

### Spatial grid, overflow and lifecycle

Flatten all worlds into shared buffers; include world ID in every cell key and interaction lookup. Configure grid dimensions independently as cells_x and cells_y. The 512 × 512 m preset uses 10 × 10 equal cells: cell widths of 51.2 m permit a 3 × 3 search for the initial maximum sensing radius of 24 m. For other rectangles, compute cell widths W/cells_x and H/cells_y separately and derive each axis's search radius from the largest relevant sensing/contact/patch-overlap range. Wrap cell indices modulo their respective grid dimensions. Never visit a wrapped cell or entity twice, including corner searches or grids smaller than the search stencil. Equal cells avoid a ragged edge-cell convention.

Use fixed-capacity cells first (64 entries/cell) and fixed patch lists. Count every insertion overflow, bounds check all reads, and mark an overflowed run invalid for science. Rerun with a larger capacity; never silently use truncated neighbors. Atomic insertion order is nondeterministic, so sort the small candidate list or use deterministic distance/ID selection and integer reductions where order matters. Worst-case crowding remains expensive even with a grid; measure neighbor comparisons and cap supported density through explicit settings.

Every 16 ticks, one coordinator per world scans eligible parents and free slots (small serial work is acceptable here). It pre-samples and stores proposed child morphology, computes its energy allocation and birth cost, checks affordability, then reserves disjoint births and charges parents. A following initialization dispatch consumes those approved morphology values without mutating them again, blends/mutates brain weights and initializes child state. No child reads a partly written parent. Use a deterministic seeded permutation to avoid always favoring low slot numbers. Rebuild the grid next tick. Define births to use the parent's learned weights **before** any learning update on the same tick. Rare migration admissions happen in this same ordered lifecycle phase.

### Data layout

| Buffer group | Suggested layout | Notes |
|---|---|---|
| Current/next state | SoA: position x/y, heading, energy, age, alive, world, generation | Double-buffer cross-creature readable state |
| Genome | Initial weights plus bounded morphology and blend gene | Immutable during life; create at birth |
| Runtime brain | Separate learned weights, baseline, policy version | One owner; no unnecessary double buffer |
| Observations | Fixed 128 values/slot; fast/slow trace storage | Reset at birth; masks distinguish unseen/empty |
| Actions/intents | Proposed motion, patch/target IDs with generation, request amounts | No arbitrary neighbor writes |
| World/resources | Patch position/radius/stock, grid counters/indices, run seed | Independent world ranges |
| Learning tape | [recorded decision, feature, slot] or measured tiled equivalent | Own life ID/policy version and validity per batch |
| Presentation | Position/heading references, cosmetic trail history, compact metrics | No per-tick full readback |

For one-thread-per-creature kernels, feature/weight-major arrays with slot as the contiguous dimension are the first candidate for coalescing. Benchmark against creature-major/tiled storage before freezing the format. Coalesced loads can compete with per-creature locality and register pressure. One thread per creature is a baseline, not a requirement to retain after profiling. Do not assume distinct per-creature networks become one efficient shared-weight matrix multiplication.

Use counter-based random streams keyed by run seed, world, slot index, slot generation, tick, purpose and draw index (including the weight index for mutation). Rendering randomness must never advance simulation RNG. Checkpoints include RNG counters, all states, genes, live weights, traces, tape/epoch metadata and config. Same-backend deterministic microtests are required; cross-driver bitwise identity is not promised.

## 6. Senses, history and the brain

![Fixed angular bins and fading history](../assets/sensory-history.png)

Use 16 fixed directions covering the full 360° body-relative circle. FOV is a gene that masks directions; changing FOV does not resize the network or improve angular sampling density for free. Out-of-view values are zero with an explicit visibility channel. Use a fixed maximum range, deterministic nearest-neighbor tie breaks, and bounded values.

Four channels per angular bin: food availability/proximity; nearest creature proximity; relative size of that nearest creature; observation visibility. Aggregate food with a specified saturating maximum rather than an order-dependent sum. No creature means zero proximity and zero relative-size entry. Current sensing supplies 64 values. Fast history has 8 × 4 = 32, slow history 4 × 4 = 16, totaling 112. Add eight self values: energy fraction, normalized age, held thrust, held turn, size, speed capability, FOV fraction, reproduction-cooldown fraction. Pad eight zeros to 128.

History is a compressed observation feature, not a trainable recurrent state. At every decision, rotate/rebin each previous trace into the new body frame using heading change, downsample current bins, then update m = lambda × reproject(m_old) + (1 − lambda) × observation, with lambda = exp(−decision_dt/tau). Start tau_fast = 0.25 s and tau_slow = 1 s. Carry the visibility/confidence channel through the same decay. This creates overlapping temporal summaries, not exact snapshots of particular past frames. Compare against literal recent/older downsampled snapshots if that distinction matters.

Angular reprojection handles turning approximately, not translation or occlusion. Do not claim the trace yields exact velocity. Test stationary food under turning; then compare no reprojection, reprojection and world-angle storage. The brain differentiates the current policy evaluation only; no backpropagation through sensory memory or world dynamics.

Brain outputs: three turn logits (left/straight/right), three thrust logits (rest/cruise/full), one attack logit and one reproduction logit. Attack/reproduction use Bernoulli sampling. The reproduction head is inactive under automatic births; exclude disabled heads from loss and entropy. Feeding remains automatic. Sample stochastic actions during learning, record the sampled action and evaluate its exact log probability. A probability-floor mixture can preserve exploration if its mixture probability is also used in the loss. Deterministic viewing is an evaluation option, not the training behavior.

## 7. Delayed learning without differentiating the ecosystem

Evolution and gradients operate at different levels. Selection changes which genomes leave descendants. Mutation changes inherited traits and initial weights. A policy gradient changes an individual's runtime weights based on its sampled actions. Neither selection nor reproduction is backpropagation.

Differentiate the neural policy's log probability while treating observations, actions and rewards as fixed data. Feeding eligibility, combat outcomes, death, resource arbitration and births can remain discrete. The score-function estimator does not require differentiating those events. This follows [REINFORCE (Williams, 1992)](https://doi.org/10.1007/BF00992696) and the [policy-gradient formulation (Sutton et al., 1999)](https://papers.nips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf). Slang's [automatic differentiation](https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/07-autodiff.html) can differentiate the pure MLP/loss; it does not make the whole simulation meaningfully differentiable.

### Reward definition

An action sampled at physics tick q is assessed over the **following** K = H × policy_stride ticks. Define u_j = alive_j × E_j / E_ref, using post-settlement energy and E_ref = 100 fixed energy units. Then:

**G_q(H) = (1/K) × sum of u_(q+k), for k = 1 … K.**

This is the future integrated energy divided by its duration; uniform dt cancels. It measures sustained reserves, not energy gained. A past energy window can be a baseline feature but is not the consequence of the just-selected action. No extra reward delay is needed initially: waiting for the future already creates delay. Record both this mean and the unnormalized integral.

After death, u is zero through the rest of the original horizon; never divide only by surviving samples. Default H = 32 decisions is about 2.13 s at stride 4 and 60 Hz. Sweep H = 8, 32, 128 (about 0.53, 2.13, 8.53 s). Keep stride fixed during this sweep. Compare fixed E_ref to own-capacity normalization explicitly: the latter rewards fullness and changes morphology incentives.

### A policy-consistent, bounded-memory schedule

![Delayed learning and inherited starting weights](../assets/learning-inheritance.png)

Freeze each participating creature's runtime weights and baseline for a global block. Record B = 32 consecutive decisions, then continue acting with the same weights for H = 32 additional decisions without collecting new training records. Every recorded action now has its full future window; the final tail interval is harmless extra coverage. Compute returns, accumulate one gradient over the recorded actions, update once and discard the batch. At the baseline this is an update every 256 physics ticks, about 4.27 s.

Use a running energy prefix sum rebased at each block and a bounded horizon queue to mature each recorded return. Sampling the prefix at action and horizon endpoints recovers the sum without storing every tick's energy. In learning mode, add a small energy-prefix bookkeeping pass after lifecycle settlement on every tick, so parent birth costs are included; count this extra dispatch in benchmarks. Store exact FP32 observations/actions; recompute the tiny MLP at the boundary while its policy version is still unchanged. Do not store a full parameter-gradient vector for every action.

For advantage A = G − b, maximize the mean of stop_gradient(A) × log pi(action | observation), optionally plus small entropy regularization. Use SGD, a conservative initial rate such as 0.0001, global per-creature gradient-norm clipping at 1, finite-value checks and a baseline fitted only from earlier completed batches. Hold that baseline constant during the block; tiny-batch self-normalization can distort the signal. Define ascent/descent signs once and verify them on a two-action toy. Check autodiff gradient initialization/accumulation explicitly.

This is a **truncated policy-gradient surrogate**. Frozen weights avoid stale behavior-policy mixing, but finite horizons, omitted tail-action gradients, changing opponents and reproduction mean it is not an unbiased gradient of whole-life or lineage fitness. Validate an exact finite-episode REINFORCE toy with full returns first. Do not update weights every tick while old rewards arrive; retaining an old log probability alone does not correct the intervening trajectory.

For the first learning proof, use fixed genomes in simple arenas with reproduction and combat disabled. In the combined ecosystem, include a life/generation ID and policy version in each batch. Newborns join training next block. When a learner dies, quarantine its slot/tape until block completion and fill future energy with zeros; no update can resurrect it or reach a replacement. Compute death-inclusive diagnostics even though a dead individual cannot benefit from learning. Allow births into genuinely free slots; reserve capacity above the live target. Apply the same quarantine/admission schedule in no-learning controls, report blocked births, and do not compare horizon settings whose different reserved occupancy silently changes ecology.

### Reward risks and explicit experiments

| Failure mode | Why it can happen | First diagnostic / response |
|---|---|---|
| Camping or hoarding | High stored energy is rewarded even without useful behavior | Track intake, movement, patch occupancy and depletion; compare fixed/relocating food |
| Avoiding reproduction | Giving energy to a child reduces parental return | Keep births automatic first; compare policy-controlled reproduction only later |
| Size dominates | Fixed-unit energy rewards larger stores | Compare fixed reference vs own-capacity fullness alongside morphology costs |
| Short horizons reject investment | Exploration or birth costs precede benefit | Sweep H; report learning updates per lifetime and adaptation delay |
| Long horizons mix unrelated events | Other creatures/environment dominate outcomes | Begin in controlled arenas; inspect variance and learning benefit across seeds |
| Survivor-only conclusions | Dead creatures have little chance to update | Include dead lives in outcome metrics and retain zero-padded returns |

For learned reproduction, compare pure parental energy, a one-time birth credit and a credit for children surviving a specified age. These are different objectives. Count each reward credit once, keep it separate from physical energy, and watch for reproduction spam. Do not add a permanent reproductive-investment bank to the energy integral. Fitness remains lineage survival/reproduction, not whichever learning reward appears most convenient.

## 8. Evolution and optional inheritance of learning

Keep immutable birth weights theta_G separate from mutable lifetime weights theta_L. At an accepted asexual birth, using the parent's pre-learning-update snapshot:

**child theta_G = (1 − alpha_parent) × parent theta_G + alpha_parent × parent theta_L + mutation**

**child theta_L = child theta_G**

The parent's blend gene controls this transfer. The child inherits and mutates that gene for its own future births. Encode evolving alpha by a bounded logit mapped through sigmoid; force exact 0 and 1 in endpoint control experiments. Mutate morphology and weights with small bounded Gaussian perturbations; start with fixed mutation scales, avoid topology changes and reject nonfinite genes. Suggested initial scales: mutate each weight with probability 0.01 and standard deviation 0.02; perturb size/speed in log space with standard deviation 0.02, FOV by 2 degrees, and evolving alpha logit by 0.05. Seed initial brains with small Xavier-style weights and zero biases. These are calibration settings, not evolved hyperparameters. Weight mutation is applied after blending. Clear all history, baseline and optimizer state at birth.

Alpha = 0 means no learned-weight inheritance; alpha = 1 copies the learned policy before mutation. Compare 0, 0.25, 1 and evolvable alpha, rather than assuming a blend is helpful. This is a computational analogy for inherited starting knowledge/teaching, not a literal model of parental care. Learned traits useful in one food layout may harm offspring elsewhere. Evaluate archived genomes in new layouts with learning both enabled and disabled to distinguish initial competence from capacity to adapt. No gradients cross births, descendants, mutation or migration.

## 9. Multiple worlds and rare genome migration

Begin with four equal worlds and independent seeds on one GPU. A single batched creature pass handles all slots; worlds need not have separate Python loops. World IDs fence sensing, food, combat and reproduction. Display one large world with small optional thumbnails; run all worlds at equal simulation time.

Every 3,600 ticks (60 simulated seconds), export at most one genome per world into a tiny shared ring/archive and propose it to another world. Sample a mature living donor, with an optional later comparison to lineage-based sampling. Default export contains theta_G, morphology and blend gene only; exporting learned weights is a separate treatment. No learner tape, sensory memories or optimizer state migrates.

An admitted genome initializes a new creature in a free destination slot. Fund its starting energy by deducting from a named destination patch/immigration reserve backed by food; reject if the budget or slot is unavailable. Alternatively log a configured external injection, but never silently create a funded adult. Stage migration in a separate ordered operation; source selection and destination admission are distinct. Keep a small source snapshot until the copy finishes. Record donor/recipient IDs and ancestry. During strict experiments disable rescue after extinction; demo reseeding must be visibly labeled and excluded from survival results.

## 10. Rendering the supplied concept beautifully

Use instanced quads with analytic ellipse coverage or signed-distance shapes. Shade a fake ellipsoid normal in the fragment shader for soft highlights and depth. Add two clear dark eyes, a thin luminous rim and two short curved limbs where desired; use simple procedural curves/quads. Body size reflects the size gene; facing reflects actual heading. Lineage color can vary subtly, while brightness can indicate energy with a minimum readable silhouette. Avoid relying solely on color for diagnostics.

Build polish in measurable layers:

1. **Readable base:** dark teal pond, soft body gradients, outlines, eyes and lime food. Match the supplied composition and shape vocabulary.
2. **Translucent material:** shaded gel-like interior, a subtle rim/Fresnel term, mild specular response. Keep silhouettes and eyes stronger than internal effects.
3. **Light:** HDR food/body emissive channel, thresholded half-resolution bloom, controlled tone mapping. Excess bloom must not merge neighboring organisms.
4. **Motion and atmosphere:** short fading trails, low-amplitude limb animation, sparse particles and subtle background variation. All are cosmetic and pause predictably.
5. **Inspection:** selected-creature FOV cone and bin display, current/fast/slow memory strips, energy accounting and action probabilities. Hide these overlays in the beauty view.

Transparency is a rendering problem with an explicit scope. Start with mostly opaque shaded bodies and a premultiplied-alpha rim, then evaluate body overlap. For real translucency, prototype weighted blended order-independent transparency or a documented stable draw ordering; neither should become a simulator dependency. Avoid full scene refraction and fluid simulation initially. Profile fill-rate/overdraw at crowded food patches, not just empty scenes. Dynamic bloom resolution and reduced particle density are the first quality controls.

The material target is [the refined concept](../assets/ellipsoid-glow-concept.png); it remains concept art rather than evidence of rendered code. The original [SVG](../assets/user-reference.svg) and [pixel image](../assets/user-reference-original.png) are the shape authority. The earlier [segmented-worm exploration](../assets/ecosystem-exploration.png) is optional mood reference only; it does not define required anatomy.

## 11. Performance, memory and verification

For a 128 → 16 → 8 MLP with biases, parameter count is 2,200. FP32 birth + live weights cost 17,600 bytes/allocated slot. At 4,096 allocated slots this is **68.75 MiB**; width 32 increases it to **137.25 MiB**. Four worlds × 2,048 slots is 8,192 allocated slots, so double those numbers even if only half the slots are alive.

A conservative tape record budget of 576 bytes (128 FP32 observations plus 64 bytes of metadata/returns/alignment) × 32 decisions × 4,096 slots is **72 MiB**, or 144 MiB for 8,192 slots. Per-creature gradient scratch adds 34.375 MiB at 4,096 slots if allocated for the full small network. EMA traces, state pairs, grids, render targets, resource buffers and profiler buffers are additional. Budget the complete allocation before scaling; learning records may cost more than simulation state. SGD avoids Adam's two extra parameter arrays. FP16 is a later measured optimization; action selection and replay must use the same quantized observations if adopted.

Warm up compilation before benchmarking. Measure headless simulation separately from rendered application performance. Capture per-pass GPU durations, p50/p95 tick/frame times, CPU command submission, active/allocated slots, neighbors tested, grid occupancy/overflow, rejected births, learning spikes, allocated VRAM, and renderer overdraw. Compare 256, 1,024 and 4,096 live creatures only as memory and correctness permit. Test several minutes for laptop power/thermal behavior. Do not infer GPU time from Python call duration.

Current SlangPy provides command-encoder GPU timing zones, frame statistics and trace export in its [profiling guide](https://slangpy.shader-slang.org/en/latest/src/profiling.html). Use the profiler available in the pinned wheel, or timestamp queries if necessary. Keep readbacks small and infrequent (for example metrics once/second), asynchronously staged; full state snapshots are explicit debug/checkpoint operations. For definitive traces, wait for outstanding measurements at capture end, not every tick.

Required checks before scientific runs:

- Energy conservation: one patch/many claimants, empty/full patches, simultaneous lethal attacks, age death with positive reserves, birth with/without free capacity, rejected migration, all fractional-rate remainders and overflow losses.
- Spatial and identity correctness: wrap across each side and both axes together, including a non-square 320 × 192 test world; separate worlds; cell overflow; duplicate-cell visits; dead/reused slots; stale target generations; no self attacks. Translate the same small scene across a join and verify equivalent sensed distances/bins, food credits, combat, movement costs and births within numeric tolerance.
- Learning: finite differences of the policy loss on CPU, known two-action learning direction, exact future-window indexing, zero padding after death, no gradients on disabled heads and no changed weights before replay.
- Integration: a slow CPU reference for tiny seeded worlds using the same semantic order; compare a handful of steps rather than demanding cross-platform bit identity for long chaotic runs.
- Presentation: pause/single-step, heading/FOV alignment, capped glow and distinguishable overlapping creatures; seam crossings must not create long trails, disappearing body halves or memory resets. Toggling effects and duplicate boundary draw instances must not change the simulation.

## 12. A small experiment ladder

Run experiments in this order; avoid a giant Cartesian sweep. Screen with three independent run seeds, then confirm the best comparison with ten and held-out food layouts. Start with fixed 60,000-physics-tick runs (1,000 simulated seconds); log wall time and extinction as outcomes. Worlds connected by migration are not independent replicates: the run seed is the experimental unit. Show medians/spread and individual run outcomes, not one attractive movie.

| Question | First comparison | Hold fixed / primary outcomes |
|---|---|---|
| Does the ecology work? | Random vs scripted forager vs evolved brains | Food inflow; survival, accounting, births, extinction |
| Does learning work at all? | Same fixed genomes, learning off/on in simple arenas | No combat/reproduction; future energy, intake, held-out layouts |
| How far ahead should rewards look? | H = 8, 32, 128 | B = 32, stride = 4; update frequency, variance, success and cost |
| Is the result an update-frequency artifact? | Match updates or sampled decisions in a follow-up | Horizon changes freeze duration; report both data and compute budgets |
| How much history helps? | No history, fast only, fast + slow | Keep 128 input shape and zero ablated channels; same weight count |
| Which energy objective? | Fixed E_ref vs own-capacity fullness | Trait bounds/costs; size distribution and reproductive success |
| Should learning be inherited? | Alpha 0 / 0.25 / 1 / evolving | Fixed chosen H; evaluate learned-offspring generalization |
| Does migration preserve useful variety? | None vs 1 genome/60 s | Same resource admission budget; lineage diversity and recovery |
| Does learned reproduction conflict? | Automatic vs policy births; then reward variants | Matched costs/cooldowns; surviving descendants, not reward alone |

Record population, age, energy, intake, expenditure, births and child survival, trait distributions, lineage diversity, extinction, learning updates/lifetime, policy entropy and GPU cost. Save config, seed, versions, checkpoint and a small metrics file per run. Archive a few representative genomes and short clips, not every frame. A reward improvement without survival/reproduction improvement is a finding, not success by definition.

## 13. Keep the eventual codebase small

Proposed modules: a Python entry point and fixed-step runner; config/checkpoint/metrics utilities; GPU resource setup; one Slang simulation file; one pure policy/loss file; a small renderer; and tiny CPU-reference tests. Split files only when real complexity warrants it. Keep new experiment settings in versioned configs rather than branching shader logic for every sweep. Logs should identify the semantic version of energy and reproduction rules.

The next concrete deliverable is the stage-0 compute-to-draw prototype plus the reference-style five-creature scene. The most valuable early success is an attractive, correctly accounted pond that can run and be inspected; learning, migration and material polish can then be added one controlled step at a time.
