# Measuring competitive improvement

The comparison graph asks whether today's brains can outcompete the population
that started the run. It does not use food intake or population in the live world:
those can settle at an ecological equilibrium even as competitors improve.

Each trial samples the current living population and a frozen reference saved when
the run starts. By default each side contributes 64 brains. Sampling is seeded and
without replacement while enough creatures are available. Smaller populations are
repeated in shuffled order to keep trial population density constant. All live
worlds contribute to the current population. An empty source population produces
an unscored result with an explicit reason.

The brains compete in two separate 512 m worlds for 10,000 simulation ticks. Each
brain receives a fresh one-metre body, a maximum speed of 12 m/s, 100 energy units,
and age zero. This measures neural behaviour without rewarding a snapshot's stored
energy, age or inherited physical traits. The second world swaps the two cohorts'
starting positions. Initial food positions and initial headings match between the
worlds. The layout seed and initial food, respawn, food energy and lifespan settings
remain fixed across trials, even when the live UI sliders change.

Food collection, energy costs, reproduction and death follow the ordinary
simulation rules. Mutation is disabled in comparison worlds: every descendant
keeps its founder's brain and cohort tag. The live world's evolution continues
normally. Comparison worlds advance once per live tick and stop exactly at the
trial duration; a completed trial is collected before recording the next frame.
Only six integer counters are read back at each trial boundary. A frame that
crosses the boundary can advance the live world a few ticks beyond it before the
next pair starts.

The score is the current cohort's share of all living creatures, including
descendants, pooled across both comparison worlds:

`score = current survivors / (current survivors + reference survivors)`

- **50%** means equal representation at the trial endpoint.
- **Above 50%** means the sampled current brains gained representation against the
  frozen reference in this competition.
- **Below 50%** means the reference gained representation.
- If both cohorts die, the result is unscored rather than a false 0% or 50% result.

The horizontal axis is the live tick at which the current brains were sampled,
not the later tick at which their trial completed. Results also retain each
world's separate counts and the completion tick. Resetting the run freezes a new
reference and clears the comparison history. Restorable snapshots preserve the
reference, in-progress trials, completed results and pending endpoint counters.

This is a repeatable competitive test, not a universal measure of intelligence.
Short trials may stay at 50% until enough births or deaths happen. Sampling causes
noise, and two swapped starts reduce position bias without eliminating it: future
food spawning and contact tie-breaking use independently seeded world streams.
Changing the live ecology can also favour strategies that do less well in the
fixed test environment. Improvements against one frozen reference need not imply
improvements against every later population; a future archive of opponents can
test that broader claim.
