"""GPU state, fixed-step motion, and drawing for a small batch of worlds."""

from math import pi
from pathlib import Path

import slangpy as spy

SHADER_DIR = Path(__file__).parent / "shaders"
WORLD_SIZE = 512.0  # Metres; matches the shared Slang world constant.
TIME_STEP = 1.0 / 60.0
CREATURE_CAPACITY = 1024
FOOD_CAPACITY = 8192
INITIAL_FOOD = 2048
TARGET_FOOD = 4096
FOOD_PER_SECOND = 1.5
ENERGY_PER_UNIT = 1024
FOOD_ENERGY = 5 * ENERGY_PER_UNIT
FOOD_AGE_START = 100_000
FOOD_AGE_FADE = 0
INITIAL_ENERGY = 100 * ENERGY_PER_UNIT  # Per metre of body length.
IDLE_ENERGY_COST = 0.05
SPEED_ENERGY_COST = 0.001
GRID_CELL_SIZE = 4.0
GRID_SIZE = 128
GRID_CELL_COUNT = GRID_SIZE * GRID_SIZE
VISION_SLOTS = 16
VISION_HISTORY = 256
VISION_RANGE = 24.0
VISION_FOV = pi
VISION_NONE = 0
VISION_FOOD = 1
VISION_CREATURE = 2
BRAIN_INPUTS = 64
BRAIN_HIDDEN = 16
BRAIN_OUTPUTS = 2
BRAIN_PARAMETERS = 1074
BRAIN_ACTIVATIONS = BRAIN_INPUTS + BRAIN_HIDDEN + BRAIN_OUTPUTS
MAX_TURN_RATE = pi / 2


class Sim:
    """N independent worlds with fixed blocks of creature and food records on the GPU.

    Food and dead creature slots are reused; newborns receive fresh identities.
    Simulation and drawing share these buffers without CPU readback.
    """

    def __init__(
        self,
        device,
        num_worlds=1,
        creatures_per_world=64,
        seed=1,
        *,
        creature_capacity=None,
        food_capacity=FOOD_CAPACITY,
        initial_food=INITIAL_FOOD,
        target_food=TARGET_FOOD,
        food_per_second=FOOD_PER_SECOND,
        food_energy=FOOD_ENERGY / ENERGY_PER_UNIT,
        food_age_start=FOOD_AGE_START,
        food_age_fade=FOOD_AGE_FADE,
        mutation_enabled=True,
    ):
        if num_worlds < 1 or creatures_per_world < 0:
            raise ValueError("Need at least one world and a nonnegative creature count")
        if creature_capacity is None:
            creature_capacity = CREATURE_CAPACITY
        if creature_capacity < creatures_per_world:
            raise ValueError("Creature capacity must hold the initial population")
        if not 0 <= initial_food <= target_food <= food_capacity or food_per_second < 0:
            raise ValueError("Need 0 <= initial food <= target <= capacity and a nonnegative rate")
        if food_age_start < 0 or food_age_fade < 0:
            raise ValueError("Food decline age and duration must be nonnegative")
        self.device = device
        self.num_worlds = num_worlds
        self.creatures_per_world = creatures_per_world
        self.creature_capacity = creature_capacity
        self.seed = seed
        self.food_capacity = food_capacity
        self.food_energy = food_energy
        self.initial_food = initial_food
        self.target_food = target_food
        self.food_per_second = food_per_second
        self.food_age_start = food_age_start
        self.food_age_fade = food_age_fade
        self.mutation_enabled = mutation_enabled
        self.tick = 0

        self._initialize = device.create_compute_kernel(
            program=device.load_program(str(SHADER_DIR / "world.slang"), ["initialize"])
        )
        self._step = device.create_compute_kernel(
            program=device.load_program(str(SHADER_DIR / "world.slang"), ["step"])
        )
        self._pick = device.create_compute_kernel(
            program=device.load_program(str(SHADER_DIR / "inspect.slang"), ["pick"])
        )
        self._initialize_food = device.create_compute_kernel(
            program=device.load_program(str(SHADER_DIR / "food.slang"), ["initialize_food"])
        )
        self._grow_food = device.create_compute_kernel(
            program=device.load_program(str(SHADER_DIR / "food.slang"), ["grow_food"])
        )
        self._set_food_energy = device.create_compute_kernel(
            program=device.load_program(str(SHADER_DIR / "food.slang"), ["set_food_energy"])
        )
        self._clear_grid = device.create_compute_kernel(
            program=device.load_program(str(SHADER_DIR / "contact.slang"), ["clear_grid"])
        )
        self._build_grid = device.create_compute_kernel(
            program=device.load_program(str(SHADER_DIR / "contact.slang"), ["build_grid"])
        )
        self._collect_food = device.create_compute_kernel(
            program=device.load_program(str(SHADER_DIR / "contact.slang"), ["collect_food"])
        )
        self._apply_food = device.create_compute_kernel(
            program=device.load_program(str(SHADER_DIR / "contact.slang"), ["apply_food"])
        )
        self._clear_vision = device.create_compute_kernel(
            program=device.load_program(str(SHADER_DIR / "vision.slang"), ["clear_vision"])
        )
        self._clear_food_grid = device.create_compute_kernel(
            program=device.load_program(str(SHADER_DIR / "vision.slang"), ["clear_food_grid"])
        )
        self._build_food_grid = device.create_compute_kernel(
            program=device.load_program(str(SHADER_DIR / "vision.slang"), ["build_food_grid"])
        )
        self._capture_vision = device.create_compute_kernel(
            program=device.load_program(str(SHADER_DIR / "vision.slang"), ["capture_vision"])
        )
        self._initialize_brains = device.create_compute_kernel(
            program=device.load_program(str(SHADER_DIR / "brain.slang"), ["initialize_brains"])
        )
        self._evaluate_brains = device.create_compute_kernel(
            program=device.load_program(str(SHADER_DIR / "brain.slang"), ["evaluate_brains"])
        )
        self._resolve_lifecycle = device.create_compute_kernel(
            program=device.load_program(str(SHADER_DIR / "lifecycle.slang"), ["resolve_lifecycle"])
        )
        self._reset_newborn_history = device.create_compute_kernel(
            program=device.load_program(
                str(SHADER_DIR / "lifecycle.slang"), ["reset_newborn_history"]
            )
        )
        usage = spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource
        self.worlds = device.create_buffer(
            element_count=num_worlds,
            resource_type_layout=self._initialize.reflection.g_worlds,
            usage=usage,
            label="worlds",
        )
        self.creatures = device.create_buffer(
            element_count=max(1, num_worlds * creature_capacity),
            resource_type_layout=self._initialize.reflection.g_creatures,
            usage=usage,
            label="creatures",
        )
        self.food = device.create_buffer(
            element_count=max(1, num_worlds * food_capacity),
            resource_type_layout=self._initialize_food.reflection.g_food,
            usage=usage,
            label="food",
        )
        self.grid_heads = device.create_buffer(
            element_count=num_worlds * GRID_CELL_COUNT,
            resource_type_layout=self._clear_grid.reflection.grid_heads,
            usage=usage,
            label="grid-heads",
        )
        self.grid_next = device.create_buffer(
            element_count=max(1, num_worlds * creature_capacity),
            resource_type_layout=self._build_grid.reflection.grid_next,
            usage=usage,
            label="grid-next",
        )
        self.food_credits = device.create_buffer(
            element_count=max(1, num_worlds * creature_capacity),
            resource_type_layout=self._build_grid.reflection.food_credits,
            usage=usage,
            label="food-credits",
        )
        self.food_grid_heads = device.create_buffer(
            element_count=num_worlds * GRID_CELL_COUNT,
            resource_type_layout=self._clear_food_grid.reflection.food_grid_heads,
            usage=usage,
            label="food-grid-heads",
        )
        self.food_grid_next = device.create_buffer(
            element_count=max(1, num_worlds * food_capacity),
            resource_type_layout=self._build_food_grid.reflection.food_grid_next,
            usage=usage,
            label="food-grid-next",
        )
        self.vision_frames = device.create_buffer(
            element_count=max(1, num_worlds * creature_capacity * VISION_HISTORY),
            resource_type_layout=self._capture_vision.reflection.vision_frames,
            usage=usage,
            label="vision-frames",
        )
        self.vision_history = device.create_buffer(
            element_count=max(1, num_worlds * creature_capacity * VISION_HISTORY * VISION_SLOTS),
            resource_type_layout=self._capture_vision.reflection.vision_history,
            usage=usage,
            label="vision-history",
        )
        self._pick_result = device.create_buffer(
            element_count=1,
            resource_type_layout=self._pick.reflection.result,
            usage=usage,
            label="picked-creature",
        )
        self.brain_weights = device.create_buffer(
            element_count=max(1, num_worlds * creature_capacity * BRAIN_PARAMETERS),
            resource_type_layout=self._initialize_brains.reflection.brain_weights,
            usage=usage,
            label="brain-weights",
        )
        self.brain_activations = device.create_buffer(
            element_count=max(1, num_worlds * creature_capacity * BRAIN_ACTIVATIONS),
            resource_type_layout=self._evaluate_brains.reflection.brain_activations,
            usage=usage,
            label="brain-activations",
        )
        self._background = device.create_render_pipeline(
            program=device.load_program(
                str(SHADER_DIR / "world_render.slang"),
                ["background_vertex", "background_fragment"],
            ),
            input_layout=None,
            targets=[{"format": spy.Format.rgba8_unorm_srgb}],
            rasterizer={"cull_mode": spy.CullMode.none},
        )
        blend = {
            "src_factor": spy.BlendFactor.one,
            "dst_factor": spy.BlendFactor.inv_src_alpha,
        }
        self._draw = device.create_render_pipeline(
            program=device.load_program(
                str(SHADER_DIR / "world_render.slang"), ["creature_vertex", "creature_fragment"]
            ),
            input_layout=None,
            targets=[
                {
                    "format": spy.Format.rgba8_unorm_srgb,
                    "enable_blend": True,
                    "color": blend,
                    "alpha": blend,
                }
            ],
            rasterizer={"cull_mode": spy.CullMode.none},
        )
        self._draw_food = device.create_render_pipeline(
            program=device.load_program(
                str(SHADER_DIR / "world_render.slang"), ["food_vertex", "food_fragment"]
            ),
            input_layout=None,
            targets=[
                {
                    "format": spy.Format.rgba8_unorm_srgb,
                    "enable_blend": True,
                    "color": blend,
                    "alpha": blend,
                }
            ],
            rasterizer={"cull_mode": spy.CullMode.none},
        )
        self._draw_vision = device.create_render_pipeline(
            program=device.load_program(
                str(SHADER_DIR / "world_render.slang"), ["vision_vertex", "vision_fragment"]
            ),
            input_layout=None,
            targets=[
                {
                    "format": spy.Format.rgba8_unorm_srgb,
                    "enable_blend": True,
                    "color": blend,
                    "alpha": blend,
                }
            ],
            rasterizer={"cull_mode": spy.CullMode.none},
        )
        self.reset()

    @property
    def food_energy(self):
        """Energy units per food piece, applied to live pieces at the next frame."""
        return self._food_energy_quanta / ENERGY_PER_UNIT

    @food_energy.setter
    def food_energy(self, value):
        quanta = round(value * ENERGY_PER_UNIT)
        # Zero denotes an empty food slot; per-tick credits must fit uint32.
        if not 1 <= quanta <= 0xFFFFFFFF // max(1, self.food_capacity):
            raise ValueError("Food energy must be positive and fit the per-tick credit range")
        self._food_energy_quanta = quanta
        self._food_energy_dirty = True

    def apply_settings(self, encoder):
        """Update live food in place, including while paused, without touching history."""
        if not self._food_energy_dirty:
            return
        self._set_food_energy.dispatch(
            thread_count=[max(1, self.num_worlds * self.food_capacity), 1, 1],
            command_encoder=encoder,
            vars={
                "num_worlds": self.num_worlds,
                "food_capacity": self.food_capacity,
                "food_energy": self._food_energy_quanta,
                "g_food": self.food,
            },
        )
        self._food_energy_dirty = False

    def reset(self, command_encoder=None):
        """Reinitialize GPU state from the seed, submitting when no encoder is supplied."""
        encoder = command_encoder or self.device.create_command_encoder()
        self._initialize.dispatch(
            thread_count=[max(self.num_worlds, self.num_worlds * self.creature_capacity), 1, 1],
            command_encoder=encoder,
            vars={
                "num_worlds": self.num_worlds,
                "creatures_per_world": self.creature_capacity,
                "seed": self.seed,
                "initial_food": self.initial_food,
                "initial_creatures": self.creatures_per_world,
                "g_worlds": self.worlds,
                "g_creatures": self.creatures,
            },
        )
        self._initialize_food.dispatch(
            thread_count=[max(1, self.num_worlds * self.food_capacity), 1, 1],
            command_encoder=encoder,
            vars={
                "num_worlds": self.num_worlds,
                "food_capacity": self.food_capacity,
                "initial_food": self.initial_food,
                "seed": self.seed,
                "food_energy": self._food_energy_quanta,
                "g_food": self.food,
            },
        )
        self._food_energy_dirty = False
        self._initialize_brains.dispatch(
            thread_count=[max(1, self.num_worlds * self.creature_capacity), 1, 1],
            command_encoder=encoder,
            vars={
                "num_worlds": self.num_worlds,
                "creatures_per_world": self.creature_capacity,
                "seed": self.seed,
                "brain_weights": self.brain_weights,
            },
        )
        self._update_grid(encoder)
        self.tick = 0
        self._clear_vision.dispatch(
            thread_count=[
                max(1, self.num_worlds * self.creature_capacity * VISION_HISTORY),
                1,
                1,
            ],
            command_encoder=encoder,
            vars={
                "num_worlds": self.num_worlds,
                "creatures_per_world": self.creature_capacity,
                "vision_frames": self.vision_frames,
                "vision_history": self.vision_history,
            },
        )
        self._update_vision(encoder)
        self._update_brains(encoder)
        if command_encoder is None:
            self.device.submit_command_buffer(encoder.finish())

    def _update_grid(self, encoder):
        """Clear cells in parallel, then insert creatures serially within each world."""
        self._clear_grid.dispatch(
            thread_count=[self.num_worlds * GRID_CELL_COUNT, 1, 1],
            command_encoder=encoder,
            vars={"num_worlds": self.num_worlds, "grid_heads": self.grid_heads},
        )
        self._build_grid.dispatch(
            thread_count=[self.num_worlds, 1, 1],
            command_encoder=encoder,
            vars={
                "num_worlds": self.num_worlds,
                "creatures_per_world": self.creature_capacity,
                "g_worlds": self.worlds,
                "g_creatures": self.creatures,
                "grid_heads": self.grid_heads,
                "grid_next": self.grid_next,
                "food_credits": self.food_credits,
            },
        )

    def _update_lifecycle(self, encoder):
        """Settle costs, deaths and energy-funded births, then clear reused histories."""
        self._resolve_lifecycle.dispatch(
            thread_count=[self.num_worlds, 1, 1],
            command_encoder=encoder,
            vars={
                "num_worlds": self.num_worlds,
                "creatures_per_world": self.creature_capacity,
                "seed": self.seed,
                "tick": self.tick + 1,
                "food_age_start": self.food_age_start,
                "food_age_fade": self.food_age_fade,
                "mutation_enabled": self.mutation_enabled,
                "g_worlds": self.worlds,
                "g_creatures": self.creatures,
                "brain_weights": self.brain_weights,
                "brain_activations": self.brain_activations,
            },
        )
        self._reset_newborn_history.dispatch(
            thread_count=[max(1, self.num_worlds * self.creature_capacity * VISION_HISTORY), 1, 1],
            command_encoder=encoder,
            vars={
                "num_worlds": self.num_worlds,
                "creatures_per_world": self.creature_capacity,
                "tick": self.tick + 1,
                "g_creatures": self.creatures,
                "vision_frames": self.vision_frames,
                "vision_history": self.vision_history,
            },
        )

    def _update_food_grid(self, encoder):
        self._clear_food_grid.dispatch(
            thread_count=[self.num_worlds * GRID_CELL_COUNT, 1, 1],
            command_encoder=encoder,
            vars={"num_worlds": self.num_worlds, "food_grid_heads": self.food_grid_heads},
        )
        self._build_food_grid.dispatch(
            thread_count=[self.num_worlds, 1, 1],
            command_encoder=encoder,
            vars={
                "num_worlds": self.num_worlds,
                "food_capacity": self.food_capacity,
                "g_food": self.food,
                "food_grid_heads": self.food_grid_heads,
                "food_grid_next": self.food_grid_next,
            },
        )

    def _update_vision(self, encoder, *, rebuild_food_grid=True):
        """Capture the settled scene without reducing history."""
        # Food stays still; capture skips eaten slots still linked in the grid.
        # Rebuild when spawning may have reused a slot elsewhere.
        if rebuild_food_grid:
            self._update_food_grid(encoder)
        self._capture_vision.dispatch(
            thread_count=[max(1, self.num_worlds * self.creature_capacity), 1, 1],
            command_encoder=encoder,
            vars={
                "num_worlds": self.num_worlds,
                "creatures_per_world": self.creature_capacity,
                "food_capacity": self.food_capacity,
                "tick": self.tick,
                "g_creatures": self.creatures,
                "g_food": self.food,
                "grid_heads": self.grid_heads,
                "grid_next": self.grid_next,
                "food_grid_heads": self.food_grid_heads,
                "food_grid_next": self.food_grid_next,
                "vision_frames": self.vision_frames,
                "vision_history": self.vision_history,
            },
        )

    def _update_brains(self, encoder):
        """Evaluate current vision and retain activations and the next movement command."""
        self._evaluate_brains.dispatch(
            thread_count=[max(1, self.num_worlds * self.creature_capacity), 1, 1],
            command_encoder=encoder,
            vars={
                "num_worlds": self.num_worlds,
                "creatures_per_world": self.creature_capacity,
                "tick": self.tick,
                "g_creatures": self.creatures,
                "vision_frames": self.vision_frames,
                "vision_history": self.vision_history,
                "brain_weights": self.brain_weights,
                "brain_activations": self.brain_activations,
            },
        )

    def update(self, command_encoder=None):
        """Record one 1/60 second step; supplied encoders keep update and draw ordered."""
        encoder = command_encoder or self.device.create_command_encoder()
        self.apply_settings(encoder)
        self._step.dispatch(
            thread_count=[max(1, self.num_worlds * self.creature_capacity), 1, 1],
            command_encoder=encoder,
            vars={
                "num_worlds": self.num_worlds,
                "creatures_per_world": self.creature_capacity,
                "g_creatures": self.creatures,
                "brain_activations": self.brain_activations,
            },
        )
        self._update_grid(encoder)
        self._collect_food.dispatch(
            thread_count=[max(1, self.num_worlds * self.food_capacity), 1, 1],
            command_encoder=encoder,
            vars={
                "num_worlds": self.num_worlds,
                "creatures_per_world": self.creature_capacity,
                "food_capacity": self.food_capacity,
                "seed": self.seed,
                "tick": self.tick,
                "g_worlds": self.worlds,
                "g_creatures": self.creatures,
                "g_food": self.food,
                "grid_heads": self.grid_heads,
                "grid_next": self.grid_next,
                "food_credits": self.food_credits,
            },
        )
        self._apply_food.dispatch(
            thread_count=[max(1, self.num_worlds * self.creature_capacity), 1, 1],
            command_encoder=encoder,
            vars={
                "num_worlds": self.num_worlds,
                "creatures_per_world": self.creature_capacity,
                "tick": self.tick + 1,
                "food_age_start": self.food_age_start,
                "food_age_fade": self.food_age_fade,
                "g_creatures": self.creatures,
                "food_credits": self.food_credits,
            },
        )
        self._update_lifecycle(encoder)
        self._update_grid(encoder)  # Vision must see settled deaths and newborn positions.
        # Cumulative whole pieces carry fractional rates across fixed ticks.
        # Counts and food records remain on the GPU; no per-frame readback.
        budget = int((self.tick + 1) * self.food_per_second * TIME_STEP) - int(
            self.tick * self.food_per_second * TIME_STEP
        )
        if budget:
            self._grow_food.dispatch(
                thread_count=[self.num_worlds, 1, 1],
                command_encoder=encoder,
                vars={
                    "num_worlds": self.num_worlds,
                    "food_capacity": self.food_capacity,
                    "food_energy": self._food_energy_quanta,
                    "target_food": self.target_food,
                    "spawn_budget": budget,
                    "seed": self.seed,
                    "g_worlds": self.worlds,
                    "g_food": self.food,
                },
            )
        self.tick += 1
        self._update_vision(encoder, rebuild_food_grid=bool(budget))
        self._update_brains(encoder)
        if command_encoder is None:
            self.device.submit_command_buffer(encoder.finish())

    def prepare_render(self, command_encoder):
        """Transition compute output before opening the render pass."""
        command_encoder.set_buffer_state(self.worlds, spy.ResourceState.shader_resource)
        command_encoder.set_buffer_state(self.creatures, spy.ResourceState.shader_resource)
        command_encoder.set_buffer_state(self.food, spy.ResourceState.shader_resource)
        command_encoder.set_buffer_state(self.vision_frames, spy.ResourceState.shader_resource)
        command_encoder.set_buffer_state(self.vision_history, spy.ResourceState.shader_resource)
        command_encoder.set_buffer_state(self.brain_activations, spy.ResourceState.shader_resource)

    def inspect_world(self, world):
        """Read one world's population and food counters for diagnostics or the UI."""
        layout = self._initialize.reflection.g_worlds.type_layout.element_type_layout
        return spy.BufferCursor(
            layout, self.worlds, size=layout.stride, offset=world * layout.stride
        )[0].read()

    def inspect(self, world, local_index):
        """Synchronously read one creature for the inspector, never the whole world."""
        layout = self._initialize.reflection.g_creatures.type_layout.element_type_layout
        cursor = spy.BufferCursor(
            layout,
            self.creatures,
            size=layout.stride,
            offset=(world * self.creature_capacity + local_index) * layout.stride,
        )
        creature = cursor[0].read()
        for field in ("position", "velocity", "radii", "color"):
            creature[field] = tuple(creature[field])
        return creature

    def inspect_brain(self, world, local_index):
        """Read one creature's current activations for diagnostics, never for drawing."""
        if not 0 <= world < self.num_worlds or not 0 <= local_index < self.creature_capacity:
            raise IndexError("Creature is outside the simulation")
        layout = self._evaluate_brains.reflection.brain_activations.type_layout.element_type_layout
        cursor = spy.BufferCursor(
            layout,
            self.brain_activations,
            size=BRAIN_ACTIVATIONS * layout.stride,
            offset=(world * self.creature_capacity + local_index)
            * BRAIN_ACTIVATIONS
            * layout.stride,
        )
        values = [cursor[i].read() for i in range(BRAIN_ACTIVATIONS)]
        return {
            "tick": self.tick,
            "inputs": values[:BRAIN_INPUTS],
            "hidden": values[BRAIN_INPUTS : BRAIN_INPUTS + BRAIN_HIDDEN],
            "steering": values[-2],
            "movement": values[-1],
        }

    def inspect_vision(self, world, local_index, age=0):
        """Read one retained observation for diagnostics; age is measured in simulation ticks."""
        if not 0 <= age < min(self.tick + 1, VISION_HISTORY):
            raise ValueError("Vision age is outside the retained history")
        if not 0 <= world < self.num_worlds or not 0 <= local_index < self.creature_capacity:
            raise IndexError("Creature is outside the simulation")
        tick = self.tick - age
        index = (tick % VISION_HISTORY) * self.num_worlds * self.creature_capacity
        index += world * self.creature_capacity + local_index
        layout = self._capture_vision.reflection.vision_frames.type_layout.element_type_layout
        frame = spy.BufferCursor(
            layout, self.vision_frames, size=layout.stride, offset=index * layout.stride
        )[0].read()
        if frame["tick"] != tick:
            return None  # Dead creatures do not record observations.
        frame["position"] = tuple(frame["position"])
        layout = self._capture_vision.reflection.vision_history.type_layout.element_type_layout
        slots = spy.BufferCursor(
            layout,
            self.vision_history,
            size=VISION_SLOTS * layout.stride,
            offset=index * VISION_SLOTS * layout.stride,
        )
        frame["slots"] = [slots[i].read() for i in range(VISION_SLOTS)]
        return frame

    def pick(
        self,
        world,
        position,
        tolerance,
        selected=-1,
        follow=False,
        selected_guid=0,
        follow_position=None,
    ):
        """Pick on the GPU, synchronizing only the resulting local creature index.

        Position uses ordinary world coordinates; when following, its origin is
        shifted from the world midpoint to the selected creature by the shader.
        """
        self._pick.dispatch(
            thread_count=[1, 1, 1],
            vars={
                "creatures": self.creatures,
                "worlds": self.worlds,
                "creatures_per_world": self.creature_capacity,
                "world_index": world,
                "pick_position": spy.float2(*position),
                "tolerance": tolerance,
                "selected": selected,
                "selected_guid": selected_guid,
                "follow": follow,
                "follow_position": spy.float2(*(follow_position or (WORLD_SIZE / 2,) * 2)),
                "result": self._pick_result,
            },
        )
        layout = self._pick.reflection.result.type_layout.element_type_layout
        index = spy.BufferCursor(layout, self._pick_result)[0].read()
        return index if index >= 0 else None

    def render(
        self,
        pass_encoder,
        rect,
        world,
        offset=(0.0, 0.0),
        scale=1.0,
        selected=-1,
        follow=False,
        wrap_view=False,
        selected_guid=0,
        follow_position=None,
    ):
        """Draw one world in a pixel rect; offset uses world units and scale is zoom."""
        x, y, width, height = rect
        pass_encoder.set_render_state(
            {
                "viewports": [spy.Viewport({"x": x, "y": y, "width": width, "height": height})],
                "scissor_rects": [
                    spy.ScissorRect(
                        {
                            "min_x": x,
                            "min_y": y,
                            "max_x": x + width,
                            "max_y": y + height,
                        }
                    )
                ],
            }
        )
        for pipeline, count in (
            (self._background, 1),
            (self._draw_vision, 9 * (1 + VISION_SLOTS) if selected >= 0 else 0),
            (self._draw_food, self.food_capacity * 9),
            (self._draw, self.creature_capacity * 9),
        ):
            if count == 0:
                continue
            cursor = spy.ShaderCursor(pass_encoder.bind_pipeline(pipeline))
            cursor["resolution"] = spy.float2(width, height)
            cursor["offset"] = spy.float2(*offset)
            cursor["scale"] = scale
            cursor["world_index"] = world
            cursor["selected"] = selected
            cursor["selected_guid"] = selected_guid
            cursor["follow"] = follow
            cursor["follow_position"] = spy.float2(*(follow_position or (WORLD_SIZE / 2,) * 2))
            cursor["wrap_view"] = wrap_view
            cursor["creatures_per_world"] = self.creature_capacity
            cursor["creatures"] = self.creatures
            if pipeline is self._draw:
                cursor["worlds"] = self.worlds
            if pipeline is self._draw_food:
                cursor["food"] = self.food
                cursor["food_capacity"] = self.food_capacity
            if pipeline is self._draw_vision:
                cursor["vision_frames"] = self.vision_frames
                cursor["vision_history"] = self.vision_history
                cursor["vision_frame_index"] = self.tick % VISION_HISTORY
                cursor["total_creatures"] = self.num_worlds * self.creature_capacity
            pass_encoder.draw({"vertex_count": 6, "instance_count": count})
