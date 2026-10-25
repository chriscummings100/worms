"""Repeated competitions against a frozen population, separate from live ecology."""

import numpy as np
import slangpy as spy

from .sim import ENERGY_PER_UNIT, FOOD_ENERGY, FOOD_PER_SECOND, SHADER_DIR, Sim

TRIAL_TICKS = 100_000
REFERENCE_WARMUP_TICKS = 200_000
SAMPLE_SIZE = 64
INITIAL_FOOD = 150
PROTOCOL_VERSION = 6  # Freeze surviving reference brains after the initial 200,000 ticks.


class Benchmark:
    """Two trials with swapped starts, fresh bodies, and inherited frozen brains.

    Call prepare before recording the frame, then update once per live tick.
    Only small population and intake counters are read back at a trial boundary.
    Live setting changes do not alter the reference or the trial environment.
    """

    def __init__(
        self,
        live_sim,
        *,
        trial_ticks=TRIAL_TICKS,
        sample_size=SAMPLE_SIZE,
        warmup_ticks=REFERENCE_WARMUP_TICKS,
    ):
        if trial_ticks < 1 or sample_size < 1 or live_sim.creature_capacity < 2:
            raise ValueError("A benchmark needs a positive duration and room for both cohorts")
        self.trial_ticks = trial_ticks
        self.warmup_ticks = warmup_ticks
        self.reference_tick = None
        self.sample_size = min(sample_size, live_sim.creature_capacity // 2)
        self.source_count = live_sim.num_worlds * live_sim.creature_capacity
        self.layout_seed = live_sim.seed ^ 0x6B8B4567
        self.trial = 0
        self.sample_live_tick = live_sim.tick
        self.pending = False
        self.history = []
        self.sim = Sim(
            live_sim.device,
            num_worlds=2,
            creatures_per_world=0,
            creature_capacity=live_sim.creature_capacity,
            seed=self.layout_seed,
            food_capacity=live_sim.food_capacity,
            initial_food=min(INITIAL_FOOD, live_sim.target_food),
            target_food=live_sim.target_food,
            food_per_second=FOOD_PER_SECOND,
            food_energy=FOOD_ENERGY / ENERGY_PER_UNIT,
            food_age_start=live_sim.food_age_start,
            food_age_fade=live_sim.food_age_fade,
            mutation_enabled=False,
        )
        self._create_resources(live_sim)
        # Waiting checkpoints must not include uninitialized GPU memory.
        encoder = self.sim.device.create_command_encoder()
        encoder.clear_buffer(self.reference_creatures)
        encoder.clear_buffer(self.reference_weights)
        self.sim.device.submit_command_buffer(encoder.finish())
        self.prepare(live_sim)

    @property
    def ready(self):
        return self.reference_tick is not None

    def _create_resources(self, live_sim):
        device = self.sim.device
        path = str(SHADER_DIR / "benchmark.slang")
        self._initialize = device.create_compute_kernel(
            program=device.load_program(path, ["initialize_cohorts"])
        )
        self._copy_food = device.create_compute_kernel(
            program=device.load_program(path, ["copy_food_layout"])
        )
        self._count = device.create_compute_kernel(
            program=device.load_program(path, ["count_lineages"])
        )
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
        self.reference_creatures = device.create_buffer(
            element_count=max(1, self.source_count),
            resource_type_layout=self._initialize.reflection.reference_creatures,
            usage=usage,
            label="benchmark-reference-creatures",
        )
        self.reference_weights = device.create_buffer(
            size=live_sim.brain_weights.size,
            struct_size=4,
            usage=usage,
            label="benchmark-reference-weights",
        )
        self._order = device.create_buffer(
            element_count=max(1, self.source_count),
            struct_size=4,
            usage=usage,
            label="benchmark-sample-order",
        )
        self._counts = device.create_buffer(
            element_count=6,
            struct_size=4,
            usage=usage,
            label="benchmark-lineage-counts",
            data=np.zeros(6, dtype=np.uint32),
        )

    def _start(self, live_sim):
        self.sample_live_tick = live_sim.tick
        order = np.random.default_rng([self.layout_seed, self.trial]).permutation(self.source_count)
        self._order.copy_from_numpy(order.astype(np.uint32))
        encoder = self.sim.device.create_command_encoder()
        self.sim.reset(encoder)
        self._initialize.dispatch(
            thread_count=[2, 1, 1],
            command_encoder=encoder,
            vars={
                "current_creatures": live_sim.creatures,
                "reference_creatures": self.reference_creatures,
                "current_weights": live_sim.brain_weights,
                "reference_weights": self.reference_weights,
                "sample_order": self._order,
                "trial_creatures": self.sim.creatures,
                "trial_worlds": self.sim.worlds,
                "trial_weights": self.sim.brain_weights,
                "source_count": self.source_count,
                "trial_capacity": self.sim.creature_capacity,
                "sample_size": self.sample_size,
                "layout_seed": self.layout_seed,
                "counts": self._counts,
            },
        )
        self._copy_food.dispatch(
            thread_count=[max(1, self.sim.food_capacity), 1, 1],
            command_encoder=encoder,
            vars={"trial_food": self.sim.food, "food_capacity": self.sim.food_capacity},
        )
        self.sim._update_grid(encoder)
        self.sim._update_vision(encoder)
        self.sim._update_brains(encoder)
        self.sim.device.submit_command_buffer(encoder.finish())
        self.pending = False

    def prepare(self, live_sim):
        """Capture the settled reference or collect a submitted trial and start the next."""
        if not self.ready:
            if live_sim.tick < self.warmup_ticks:
                return False
            # App calls this before recording a frame, so these copies see only
            # submitted live updates. A batch may cross the threshold slightly.
            encoder = self.sim.device.create_command_encoder()
            encoder.copy_buffer(
                self.reference_creatures, 0, live_sim.creatures, 0, live_sim.creatures.size
            )
            encoder.copy_buffer(
                self.reference_weights, 0, live_sim.brain_weights, 0, live_sim.brain_weights.size
            )
            self.sim.device.submit_command_buffer(encoder.finish())
            self.reference_tick = live_sim.tick
            self._start(live_sim)
            return True
        if not self.pending:
            return False
        values = self._counts.to_numpy().view(np.uint32).reshape(-1)
        counts = values[:4].reshape(2, 2)
        current, reference = map(int, counts.sum(axis=0))
        energy = self.sim.food_energy_totals.to_numpy().view(np.uint64).reshape(2, 3)[:, 1:]
        current_energy, reference_energy = map(int, energy.sum(axis=0))
        total_energy = current_energy + reference_energy
        reason = None
        if not values[4]:
            reason = "no_current_population"
        elif not values[5]:
            reason = "no_reference_population"
        elif not total_energy:
            reason = "no_energy_collected"
        self.history.append(
            {
                "live_tick": self.sample_live_tick,
                "completed_live_tick": live_sim.tick,
                "trial": self.trial,
                "current": current,
                "reference": reference,
                "current_energy": current_energy / ENERGY_PER_UNIT,
                "reference_energy": reference_energy / ENERGY_PER_UNIT,
                "world_energy": (energy / ENERGY_PER_UNIT).tolist(),
                "score": current_energy / total_energy if reason is None else None,
                "world_counts": counts.tolist(),
                "source_counts": values[4:].tolist(),
                "reason": reason,
            }
        )
        self.trial += 1
        self._start(live_sim)
        return True

    def update(self, encoder):
        """Record one tick, stopping exactly at the trial boundary until prepare."""
        if not self.ready or self.pending:
            return
        self.sim.update(encoder)
        if self.sim.tick == self.trial_ticks:
            self._count.dispatch(
                thread_count=[2, 1, 1],
                command_encoder=encoder,
                vars={
                    "trial_creatures": self.sim.creatures,
                    "trial_capacity": self.sim.creature_capacity,
                    "counts": self._counts,
                },
            )
            self.pending = True

    def snapshot_state(self):
        """Read explicit checkpoint extras; the caller also saves self.sim."""
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "trial_ticks": self.trial_ticks,
            "warmup_ticks": self.warmup_ticks,
            "reference_tick": self.reference_tick,
            "sample_size": self.sample_size,
            "source_count": self.source_count,
            "layout_seed": self.layout_seed,
            "trial": self.trial,
            "sample_live_tick": self.sample_live_tick,
            "pending": self.pending,
            "history": self.history,
        }
        arrays = {
            "reference_creatures": self.reference_creatures.to_numpy(),
            "reference_weights": self.reference_weights.to_numpy(),
            "benchmark_counts": self._counts.to_numpy(),
        }
        return metadata, arrays

    @classmethod
    def from_snapshot(cls, live_sim, trial_sim, metadata, arrays):
        """Resume waiting or an active trial without recapturing the frozen reference."""
        if metadata.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Unsupported benchmark protocol; start a new run")
        self = cls.__new__(cls)
        self.sim = trial_sim
        for name in (
            "trial_ticks",
            "warmup_ticks",
            "reference_tick",
            "sample_size",
            "source_count",
            "layout_seed",
            "trial",
            "sample_live_tick",
            "pending",
            "history",
        ):
            setattr(self, name, metadata[name])
        self._create_resources(live_sim)
        self.reference_creatures.copy_from_numpy(arrays["reference_creatures"])
        self.reference_weights.copy_from_numpy(arrays["reference_weights"])
        self._counts.copy_from_numpy(arrays["benchmark_counts"])
        return self
