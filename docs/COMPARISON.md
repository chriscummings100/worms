# Measuring competitive improvement

The comparison graph asks whether today's brains can outcompete the population
that survived the first 200,000 live ticks. It does not use food intake or population in the live world:
those can settle at an ecological equilibrium even as competitors improve.

The benchmark stays idle until the first submitted frame boundary at or after
**200,000 live ticks**. It then freezes the live creature records and brains once.
Only creatures alive in that captured state are eligible reference samples, so
initial dead ends that died during the wait are excluded. Later births and deaths
do not change this fixed opponent. The actual capture tick is saved and displayed;
a batch crossing the threshold may capture a few ticks later.

Each trial samples the current living population and that frozen reference.
By default each side contributes 64 brains. Sampling is seeded and
without replacement while enough creatures are available. Smaller populations are
repeated in shuffled order to keep trial population density constant. All live
worlds contribute to the current population. An empty source population produces
an unscored result with an explicit reason.

The brains compete in two separate 512 m worlds for 100,000 simulation ticks. Each
brain receives a fresh one-metre body, a maximum speed of 12 m/s, 100 energy units,
and age zero. This measures neural behaviour without rewarding a snapshot's stored
energy, age or inherited physical traits. The second world swaps the two cohorts'
starting positions. Initial food positions and initial headings match between the
worlds. Each world starts with **150 food pieces**, adds the default **10 pieces per simulated
second**, and supplies **5 energy units per piece**. Small test allocations cap the
initial amount at their food capacity and target. These scarce initial conditions
avoid the live world's starting surplus of 2,048 pieces. Over a full trial up to
16,666 new pieces can arrive, making the initial stock a small part of the food budget.
The longer trial lets feeding sustain a lineage after its starting reserves run
out. The layout seed and food conditions stay fixed across trials. Food capacity,
target and lifespan settings are copied from the live world when the benchmark
is constructed and remain fixed, even when the live UI settings change.

Food collection, energy costs, reproduction and death follow the ordinary
simulation rules. Mutation is disabled in comparison worlds: every descendant
keeps its founder's brain and cohort tag. The live world's evolution continues
normally. Both offspring receive fresh identities, generation plus one and empty
vision history. The live Input-weight mutation and Bias/output mutation settings
change variation in future offspring, but do not alter benchmark mutation or its
fixed conditions; changing them does not restart trials or the graph. Once the
reference is captured, comparison worlds advance once per live tick and stop at
the trial duration; a completed trial is collected before recording the next frame.
Only small population and food-intake counters are read back at each trial
boundary. A frame that crosses the boundary can advance the live world a few ticks
beyond it before the next pair starts.

The score is the current cohort's share of all food energy actually gained over
the whole trial, pooled across both comparison worlds:

`score = current food energy / (current food energy + reference food energy)`

- **50%** means both cohorts collected the same amount of energy.
- **Above 50%** means the current cohort collected more than the frozen reference.
- **Below 50%** means the reference collected more.
- If neither cohort gains food energy, the result is unscored rather than a false
  0% or 50% result. If only the reference feeds, the current cohort scores 0%.

Intake includes all descendants and remains counted after an eater dies or its
slot is reused. It measures energy transferred into creatures after any age-based
reduction, not the energy in food that was removed. Starting reserves and energy
transferred during splitting do not count as collection. Surviving population
counts remain secondary diagnostics; extinction does not erase energy already
collected.

The horizontal axis is the live tick at which the current brains were sampled,
not the later tick at which their trial completed. Results also retain each
world's separate counts, food-energy totals and the completion tick. The first
trial samples current and reference from the same captured population, establishing
a starting comparison. Its result arrives around live tick 300,000; subsequent
trials compare later populations against that original settled reference.

Reset clears the reference and graph and starts another 200,000-tick wait.
Protocol 6 snapshots preserve the waiting state and threshold, or the frozen
reference and actual capture tick together with in-progress trials, completed
results and pending endpoint counters. A restore never recaptures an existing
reference. Earlier benchmark protocols are unsupported; start a new run.
The 10 pieces/s food baseline and two-offspring reproduction rules are retained.

This is a repeatable competitive test, not a universal measure of intelligence.
Collection measures access to food under competition; it does not directly reward
energy efficiency or survival. Sampling causes noise, and two swapped starts
reduce position bias without eliminating it: future food spawning and contact
tie-breaking use independently seeded world streams.
Changing the live ecology can also favour strategies that do less well in the
fixed test environment. Improvements against one frozen reference need not imply
improvements against every later population; a future archive of opponents can
test that broader claim.

After capture, choose **Benchmark A** or **Benchmark B** in the world selector to inspect a trial with the usual camera, vision, follow and neuron display. Teal is the current cohort and orange is the settled reference. Benchmark settings are displayed read-only; change live settings by switching back to a live world. A new trial clears benchmark creature selection and resets its camera.


## Mutation-rate experiment — 25 October 2026

These historical results use protocol 3, where a split retained the parent and
mutated only the new child. They do not establish the best mutation rates under
the current two-offspring rules, which also change generation accounting.

Three live continuations started from the same unchanged 100,000-tick checkpoint
and evolved for another 200,000 ticks. Input-to-hidden weight mutation remained
1%, and Gaussian mutation standard deviation remained 0.02. Only the chance for
the other 50 parameters (hidden biases, output weights and output biases) varied.
Food, lifespan, body traits, energy reserves and split thresholds were unchanged.
Each final population was evaluated for three 100,000-tick trials against the
same frozen original reference, using the ordinary two swapped-start worlds.
Live evolution was paused during evaluation, and benchmark mutation was disabled.
These historical trials used 1.5 food pieces/s, before the new 10/s default.

| Bias/output mutation chance | Pooled food-energy score | Splits during training | Mean generation at end |
|---|---:|---:|---:|
| 1% (legacy) | **70.74%** | 78 | 3.10 |
| 3% | 68.14% | 67 | 3.51 |
| 10% | 68.65% | 74 | 3.30 |

Both higher rates underperformed this control, so the default initially stayed
at **1%**. Following the change to two offspring per split, both input-weight and
bias/output defaults were set to **10%** for further experimentation, with direct
percentage entry in the UI. This does not claim to have accelerated behavioural improvement. Higher mean
generation did not imply better competitive performance here.

These are three cohort samples from each final population on one fixed layout,
not independent training replicates. Scores pool collected energy across trials;
the starting population was not separately evaluated, so the table compares
terminal outcomes rather than measuring gain from the starting checkpoint. One
training seed cannot establish a universally best mutation rate.

Source checkpoint SHA-256:
`a87b98d2df0a8a59af9b96b544454f082fb15c120ca53556baffc0679d42aefc`.
Reproduction script: `artifacts/evolution/compare_policy_mutation.py`.
Reports: `artifacts/evolution/policy-mutation-comparison.json` and
`artifacts/evolution/policy-mutation-gentle.json`. Each trained checkpoint is saved
alongside those reports, retaining the frozen opponent and initializing the next
benchmark trial. The original checkpoint was not modified.

Verification at the time: 32 GPU lifecycle/snapshot/session checks and 56 CPU checks passed.
Those tests covered policy-block isolation, unchanged parents and input-weight mutation
streams, disabled mutation, settings recorded per birth, reset persistence,
nondefault checkpoint continuation and legacy settings migration.
