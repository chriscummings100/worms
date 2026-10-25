"""Windowed or headless entry point for the Worms application."""

import argparse
import math
from pathlib import Path
from time import monotonic

import slangpy as spy
from PIL import Image

from worms.benchmark import Benchmark
from worms.brain_view import BrainView, brain_panel_rect
from worms.camera import Camera
from worms.comparison_view import ComparisonView, comparison_result
from worms.sim import ENERGY_PER_UNIT, WORLD_SIZE, Sim
from worms.snapshots import PeriodicSnapshots, load_snapshot, save_snapshot

SIM_SPEEDS = (1, 2, 5, 10, 20, 50, 100)


def world_views(width, height, num_worlds, selected_world=None):
    """Return world indices and pixel rectangles, with gutters between views."""
    worlds = range(num_worlds) if selected_world is None else [selected_world]
    columns = math.ceil(math.sqrt(len(worlds)))
    rows = math.ceil(len(worlds) / columns)
    views = []
    for index, world in enumerate(worlds):
        column, row = index % columns, index // columns
        x, y = column * width // columns, row * height // rows
        right, bottom = (column + 1) * width // columns, (row + 1) * height // rows
        if right <= x or bottom <= y:
            continue  # A tiny resized window may not have one pixel per world.
        pad = min(8, (right - x - 1) // 2, (bottom - y - 1) // 2)
        views.append((world, (x + pad, y + pad, right - x - 2 * pad, bottom - y - 2 * pad)))
    return views


def contains(rect, point):
    x, y, width, height = rect
    return x <= point[0] < x + width and y <= point[1] < y + height


class App:
    """Own the device, simulation, frame target, and optional window."""

    def __init__(
        self,
        *,
        headless=False,
        num_worlds=1,
        creatures_per_world=64,
        creature_capacity=None,
        width=1280,
        height=720,
        benchmark=True,
        snapshot_dir=Path("artifacts/snapshots"),
        resume=None,
        debug_gpu=False,
    ):
        adapter = next(
            a for a in spy.Device.enumerate_adapters(spy.DeviceType.d3d12) if "NVIDIA" in a.name
        )
        self.debug_gpu = debug_gpu
        self.device = spy.Device(
            type=spy.DeviceType.d3d12,
            adapter_luid=adapter.luid,
            enable_debug_layers=debug_gpu,
            enable_rhi_validation=debug_gpu,
            compiler_options={"include_paths": [Path(__file__).parent / "shaders"]},
        )
        self.sim = Sim(
            self.device, num_worlds, creatures_per_world, creature_capacity=creature_capacity
        )
        self.benchmark = Benchmark(self.sim) if benchmark else None
        self.snapshots = PeriodicSnapshots(snapshot_dir) if snapshot_dir is not None else None
        self._snapshot_requested = False
        self._restore_requested = False
        self.snapshot_status = (
            f"Automatic snapshots every {self.snapshots.interval:,} ticks "
            f"(keep {self.snapshots.keep})"
            if self.snapshots is not None
            else ""
        )
        self.show_comparison = False
        self.brain_view = BrainView(self.device)
        self.comparison_view = ComparisonView(self.device, self.brain_view.background)
        self.texture = None
        self.resize(width, height)
        self.paused = False
        self.speed = 1
        self.camera = Camera(world_size=WORLD_SIZE)
        self.selected_world = 0 if num_worlds == 1 else None
        self.frame_count = 0
        self._step = False
        self._reset = False
        self._frame_limit = None
        self.window_app = None
        self.window = None
        if resume is not None:
            self.restore(resume)
        if not headless:
            self.window_app = spy.App(self.device)
            self.window = _Window(self)

    @property
    def selected_world(self):
        return self._selected_world

    @selected_world.setter
    def selected_world(self, world):
        self._selected_world = world
        self.selected_creature = None
        self.selected_guid = 0
        self.creature_stats = None
        self.population_stats = None
        self._next_stats_update = 0.0
        self._next_population_update = 0.0
        self.fit_world()

    @property
    def viewing_benchmark(self):
        return self.selected_world is not None and self.selected_world >= self.sim.num_worlds

    @property
    def view_sim(self):
        return self.benchmark.sim if self.viewing_benchmark else self.sim

    @property
    def view_world(self):
        return (
            self.selected_world - self.sim.num_worlds
            if self.viewing_benchmark
            else self.selected_world
        )

    def world_choices(self):
        choices = ["All live worlds"] + [f"World {i + 1}" for i in range(self.sim.num_worlds)]
        if self.benchmark is not None and self.benchmark.ready:
            choices += ["Benchmark A", "Benchmark B"]
        return choices

    def comparison_status(self):
        benchmark = self.benchmark
        if benchmark is None:
            return "Comparison disabled"
        if not benchmark.ready:
            return (
                "Waiting to capture baseline\n"
                f"Live tick {self.sim.tick:,}/{benchmark.warmup_ticks:,}"
            )
        latest = benchmark.history[-1] if benchmark.history else None
        return (
            f"Baseline captured at live tick {benchmark.reference_tick:,}\n"
            f"Benchmark {benchmark.sim.tick:,}/{benchmark.trial_ticks:,} ticks\n"
            f"{comparison_result(latest)}"
        )

    def fit_world(self):
        self.following = False
        self.camera.reset()
        self.cancel_drag()

    def views(self):
        left = self.view_left()
        return [
            (world, (x + left, y, width, height))
            for world, (x, y, width, height) in world_views(
                self.texture.width - left,
                self.world_bottom(),
                self.sim.num_worlds,
                self.selected_world,
            )
        ]

    def view_left(self):
        return (
            0
            if self.window is None
            else min(int(self.window.panel.size.x) + 32, self.texture.width // 2)
        )

    def brain_rect(self):
        if self.selected_world is None or self.selected_creature is None:
            return None
        return brain_panel_rect(self.texture.width, self.texture.height)

    def graph_rect(self):
        if not getattr(self, "show_comparison", False):
            return None
        bottom = self.brain_rect()[1] if self.brain_rect() is not None else self.texture.height
        height = min(224, self.texture.height // 3)
        left = self.view_left()
        return (left, bottom - height, self.texture.width - left, height)

    def world_bottom(self):
        graph = self.graph_rect()
        brain = self.brain_rect()
        if graph is not None:
            return graph[1]
        return brain[1] if brain is not None else self.texture.height

    def focused_rect(self):
        views = self.views()
        return views[0][1] if self.selected_world is not None and views else None

    def zoom(self, point, wheel_delta):
        rect = self.focused_rect()
        if rect is not None and contains(rect, point):
            self.camera.zoom_at(point, rect, wheel_delta)

    def cancel_drag(self):
        self._press = self._last = None
        self._dragging = False

    def mouse_down(self, point):
        self.cancel_drag()
        rect = self.focused_rect()
        if rect is not None and contains(rect, point):
            self._press = self._last = point

    def mouse_move(self, point):
        if self._press is None:
            return
        rect = self.focused_rect()
        if rect is None or not contains(rect, point):
            self.cancel_drag()
            return
        if not self._dragging and math.dist(point, self._press) >= 4:
            self.set_follow(False)
            self._dragging = True
        if self._dragging:
            self.camera.pan((point[0] - self._last[0], point[1] - self._last[1]), rect)
            self._last = point

    def mouse_up(self, point):
        self.mouse_move(point)
        click = self._press is not None and not self._dragging
        self.cancel_drag()
        if click:
            self.pick(point)

    def pick(self, point):
        rect = self.focused_rect()
        if rect is None or not contains(rect, point):
            return
        selected = self.view_sim.pick(
            self.view_world,
            self.camera.screen_to_world(point, rect),
            tolerance=5 / self.camera.pixels_per_unit(rect),
            selected=self.selected_creature if self.selected_creature is not None else -1,
            selected_guid=self.selected_guid,
            follow_position=self._follow_position(),
            follow=self.following,
        )
        if selected == self.selected_creature:
            if selected is None:
                return
            # A split can replace the selected creature between inspector polls.
            # A click on its old slot selects the new identity explicitly.
            self.refresh_stats()
            if selected == self.selected_creature:
                return
        self.set_follow(False)
        self.selected_creature = selected
        self.selected_guid = 0
        self.creature_stats = None
        if selected is not None:
            self.refresh_stats()

    def refresh_stats(self):
        stats = self.view_sim.inspect(self.view_world, self.selected_creature)
        if not stats["alive"] or (self.selected_guid and stats["guid"] != self.selected_guid):
            # A slot can be reused between inspector polls. Keep the last known
            # position, and never silently transfer selection to its newborn.
            if self.following and self.creature_stats is not None:
                self._leave_follow(self.creature_stats["position"])
            self.following = False
            self.selected_creature = None
            self.selected_guid = 0
            self.creature_stats = None
        else:
            self.creature_stats = stats
            self.selected_guid = stats["guid"]
        self._next_stats_update = monotonic() + 0.2

    def refresh_population(self):
        sim = self.view_sim
        worlds = range(sim.num_worlds) if self.selected_world is None else [self.view_world]
        records = [sim.inspect_world(world) for world in worlds]
        self.population_stats = {
            field: sum(record[field] for record in records)
            for field in ("num_creatures", "births", "deaths", "num_food")
        }
        self.population_stats["capacity"] = len(records) * sim.creature_capacity
        self.population_stats["food_target"] = len(records) * sim.target_food
        self._next_population_update = monotonic() + 0.2

    def clear_selection(self):
        self.set_follow(False)
        self.selected_creature = None
        self.selected_guid = 0
        self.creature_stats = None

    def _leave_follow(self, position):
        self.camera.offset = tuple(
            offset + coordinate - WORLD_SIZE / 2
            for offset, coordinate in zip(self.camera.offset, position, strict=True)
        )
        self.following = False

    def _follow_position(self):
        return (
            self.creature_stats["position"]
            if self.creature_stats is not None
            else (WORLD_SIZE / 2, WORLD_SIZE / 2)
        )

    def set_follow(self, enabled):
        if enabled and self.selected_creature is not None:
            self.following = True
            self.camera.offset = (0.0, 0.0)
        elif self.following:
            # Preserve the current view when handing control back to the mouse.
            self.refresh_stats()
            if self.following:
                self._leave_follow(self.creature_stats["position"])

    def resize(self, width, height):
        if self.texture is not None and (width, height) == (
            self.texture.width,
            self.texture.height,
        ):
            return
        self.texture = self.device.create_texture(
            width=width,
            height=height,
            format=spy.Format.rgba8_unorm_srgb,
            usage=(
                spy.TextureUsage.render_target
                | spy.TextureUsage.shader_resource
                | spy.TextureUsage.copy_source
            ),
            label="world-views",
        )

    def step(self):
        self.paused = True
        self._step = True

    def reset(self):
        self._reset = True
        self.selected_world = (
            (0 if self.sim.num_worlds == 1 else None)
            if self.viewing_benchmark
            else self.selected_world
        )

    def _snapshot_state(self):
        metadata = {"speed": self.speed, "paused": self.paused}
        state = {"metadata": metadata}
        if self.benchmark is not None:
            metadata["benchmark"], state["arrays"] = self.benchmark.snapshot_state()
            state["extra_sims"] = {"benchmark": self.benchmark.sim}
        return state

    def snapshot(self, path=None):
        """Save submitted state, including the reference and unfinished benchmark."""
        state = self._snapshot_state()
        if path is None:
            path = self.snapshots.save(self.sim, **state)
        else:
            path = save_snapshot(path, self.sim, **state)
        self.snapshot_status = f"Snapshot saved at tick {self.sim.tick:,}"
        return path

    def restore(self, path):
        saved = load_snapshot(path, self.device)
        self.sim = saved.sim
        self.benchmark = (
            Benchmark.from_snapshot(
                self.sim,
                saved.extra_sims["benchmark"],
                saved.metadata["benchmark"],
                saved.arrays,
            )
            if "benchmark" in saved.metadata
            else None
        )
        self.speed = saved.metadata.get("speed", 1)
        self.paused = True
        self._step = self._reset = False
        if self.benchmark is None:
            self.show_comparison = False
        self.selected_world = 0 if self.sim.num_worlds == 1 else None
        if self.snapshots is not None:
            self.snapshots.reset(self.sim.tick)
        self.snapshot_status = f"Restored tick {self.sim.tick:,} (paused)"
        if self.window is not None:
            self.window.world_selector.items = self.world_choices()

    def _prepare_benchmark(self):
        if self.benchmark is not None and self.benchmark.prepare(self.sim):
            if self.viewing_benchmark:
                # Trials reuse slots and initial IDs. Never transfer selection
                # or follow from a previous trial into a newly sampled cohort.
                self.selected_world = self.selected_world
            return True
        return False

    def update(self, encoder):
        # Checkpoint/benchmark readbacks only see work submitted by the previous
        # frame. Never read back a state whose updates are still in this encoder.
        if self._restore_requested:
            self.restore(self.snapshots.latest)
            self._restore_requested = False
        reset = self._reset
        if reset:
            self.sim.reset()
            if self.benchmark is not None:
                self.benchmark = Benchmark(self.sim)
            if self.snapshots is not None:
                self.snapshots.reset()
        else:
            self._prepare_benchmark()
        if self.snapshots is not None and (
            self._snapshot_requested or self.snapshots.due(self.sim.tick)
        ):
            self.snapshot()
        self._snapshot_requested = False
        self.sim.apply_settings(encoder)
        # Poll small inspector/counter records before recording the next update.
        if self.selected_creature is not None and monotonic() >= self._next_stats_update:
            self.refresh_stats()
        if (
            self.window is not None
            and not self._reset
            and monotonic() >= self._next_population_update
        ):
            self.refresh_population()
        if not reset and (self._step or not self.paused):
            for _ in range(1 if self._step else self.speed):
                self.sim.update(encoder)
                if self.benchmark is not None:
                    self.benchmark.update(encoder)
        self._reset = self._step = False
        self.frame_count += 1

    def render(self, encoder):
        sim = self.view_sim
        sim.prepare_render(encoder)
        brain_rect = self.brain_rect()
        if brain_rect is not None and brain_rect[2] > 0 and brain_rect[3] > 0:
            self.brain_view.prepare(encoder, brain_rect)
        graph_rect = self.graph_rect()
        if graph_rect is not None and graph_rect[2] > 0 and graph_rect[3] > 0:
            self.comparison_view.prepare(
                encoder, graph_rect, self.benchmark.history if self.benchmark is not None else []
            )
        with encoder.begin_render_pass(
            {
                "color_attachments": [
                    {
                        "view": self.texture.create_view({}),
                        "load_op": spy.LoadOp.clear,
                        "store_op": spy.StoreOp.store,
                        "clear_value": [0.008, 0.014, 0.02, 1.0],
                    }
                ]
            }
        ) as render_pass:
            for world, rect in self.views():
                if self.selected_world is None:
                    sim.render(render_pass, rect, world)
                else:
                    sim.render(
                        render_pass,
                        rect,
                        self.view_world,
                        offset=self.camera.offset,
                        scale=self.camera.scale,
                        selected=(
                            self.selected_creature if self.selected_creature is not None else -1
                        ),
                        selected_guid=self.selected_guid,
                        follow_position=self._follow_position(),
                        follow=self.following,
                        wrap_view=True,
                    )
            if brain_rect is not None and brain_rect[2] > 0 and brain_rect[3] > 0:
                self.brain_view.render(
                    render_pass,
                    brain_rect,
                    sim.brain_activations,
                    sim.creatures,
                    self.view_world * sim.creature_capacity + self.selected_creature,
                    self.selected_guid,
                )
            if graph_rect is not None and graph_rect[2] > 0 and graph_rect[3] > 0:
                self.comparison_view.render(render_pass, graph_rect)

    def frame(self):
        """Submit one headless frame using the same update and render as the window."""
        encoder = self.device.create_command_encoder()
        self.update(encoder)
        self.render(encoder)
        submission = self.device.submit_command_buffer(encoder.finish())
        # Headless frames have no swapchain to bound work in flight. Retire each
        # frame before recording more transient D3D12 descriptors at high speed.
        self.device.wait_for_submit(submission)

    def run(self, frames=None):
        if self.window_app is not None:
            self._frame_limit = None if frames is None else self.frame_count + frames
            self.window_app.run()
        else:
            for _ in range(120 if frames is None else frames):
                self.frame()
        self._prepare_benchmark()
        if self.snapshots is not None and self.snapshots.due(self.sim.tick):
            self.snapshot()

    def save(self, path):
        """Save the scene at the last completed frame, without the window's UI."""
        self.device.wait()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.texture.to_numpy()).save(path)

    def close(self):
        self.device.wait()
        if self.window_app is not None:
            self.window_app.terminate()
            self.window.owner = None
            self.window = None
            self.window_app = None
        self.device.close()


class _Window(spy.AppWindow):
    """SlangPy owns events, resize handling, UI composition and presentation."""

    def __init__(self, owner):
        super().__init__(
            owner.window_app,
            width=owner.texture.width,
            height=owner.texture.height,
            title=(f"Worms | {owner.sim.num_worlds} worlds"),
            enable_vsync=True,
        )
        self.owner = owner
        self.panel = panel = spy.ui.Window(
            self.screen, title="Worms###controls", position=spy.float2(16, 8), size=spy.float2(0, 0)
        )
        self.speed_selector = spy.ui.ComboBox(
            panel,
            label="Speed",
            items=["Paused"] + [f"{speed}x" for speed in SIM_SPEEDS],
            value=0 if owner.paused else SIM_SPEEDS.index(owner.speed) + 1,
            callback=self._set_speed,
        )
        spy.ui.Button(panel, label="Step", callback=owner.step)
        spy.ui.Button(panel, label="Reset", callback=owner.reset)
        self.world_selector = spy.ui.ComboBox(
            panel,
            label="",
            items=owner.world_choices(),
            callback=lambda value: setattr(owner, "selected_world", value - 1 if value else None),
        )
        self.world_selector.value = (
            owner.selected_world + 1 if owner.selected_world is not None else 0
        )
        self.view_info = spy.ui.Text(panel, text="")
        self.status = spy.ui.Text(panel, text="Alive 0/0  Splits 0  Died 0\nFood 0/0")
        self.food_respawn = spy.ui.InputFloat(
            panel,
            label="Food respawn (/s/world)",
            value=owner.sim.food_per_second,
            step=0.1,
            step_fast=1.0,
            format="%.2f",
            callback=lambda value: setattr(owner.sim, "food_per_second", max(0.0, value)),
        )
        self.fit = spy.ui.Button(panel, label="Fit world", callback=owner.fit_world)
        self.food_energy = spy.ui.SliderFloat(
            panel,
            label="Food energy",
            value=owner.sim.food_energy,
            min=0.25,
            max=10.0,
            format="%.2f",
            callback=lambda value: setattr(owner.sim, "food_energy", value),
        )
        self.food_age_start = spy.ui.SliderInt(
            panel,
            label="Decline age",
            value=owner.sim.food_age_start,
            min=0,
            max=100000,
            format="%d ticks",
            callback=lambda value: setattr(owner.sim, "food_age_start", value),
        )
        self.food_age_fade = spy.ui.SliderInt(
            panel,
            label="Fade length",
            value=owner.sim.food_age_fade,
            min=0,
            max=100000,
            format="%d ticks",
            callback=lambda value: setattr(owner.sim, "food_age_fade", value),
        )
        self.input_mutation = spy.ui.InputFloat(
            panel,
            label="Input-weight mutation (%)",
            value=owner.sim.input_mutation_probability * 100,
            step=0.1,
            step_fast=1.0,
            format="%.2f",
            callback=lambda value: setattr(
                owner.sim, "input_mutation_probability", min(1.0, max(0.0, value / 100))
            ),
        )
        self.policy_mutation = spy.ui.InputFloat(
            panel,
            label="Bias/output mutation (%)",
            value=owner.sim.policy_mutation_probability * 100,
            step=0.1,
            step_fast=1.0,
            format="%.2f",
            callback=lambda value: setattr(
                owner.sim, "policy_mutation_probability", min(1.0, max(0.0, value / 100))
            ),
        )
        self.comparison_toggle = spy.ui.CheckBox(
            panel,
            label="Show comparison graph",
            callback=lambda value: setattr(owner, "show_comparison", value),
        )
        self.comparison_status = spy.ui.Text(panel, text="")
        self.save_snapshot = spy.ui.Button(
            panel,
            label="Snapshot now",
            callback=lambda: setattr(owner, "_snapshot_requested", True),
        )
        self.restore_snapshot = spy.ui.Button(
            panel,
            label="Restore latest snapshot",
            callback=lambda: setattr(owner, "_restore_requested", True),
        )
        self.snapshot_status = spy.ui.Text(panel, text="")
        self.inspector = spy.ui.Window(
            self.screen,
            title="Creature",
            position=spy.float2(owner.texture.width - 356, 16),
            size=spy.float2(0, 0),
        )
        self.inspector.visible = False
        self._inspector_shown = False
        self.follow = spy.ui.CheckBox(self.inspector, label="Follow", callback=owner.set_follow)
        self.stats = spy.ui.Text(self.inspector, text="")

    def _set_speed(self, value):
        self.owner.paused = value == 0
        if value:
            self.owner.speed = SIM_SPEEDS[value - 1]

    def on_mouse_event(self, event):
        # AppWindow routes events through its UI first. These guards also cover
        # entering a panel before ImGui's next frame has updated mouse capture.
        point = (event.pos.x, event.pos.y)
        app = self.owner
        for panel in (self.panel, self.inspector):
            if panel.visible and contains(
                (panel.position.x, panel.position.y, panel.size.x, panel.size.y), point
            ):
                app.cancel_drag()
                return
        if event.type == spy.MouseEventType.scroll:
            app.zoom(point, event.scroll.y)
        elif event.type == spy.MouseEventType.move:
            app.mouse_move(point)
        elif event.button == spy.MouseButton.left:
            if event.type == spy.MouseEventType.button_down:
                app.mouse_down(point)
            elif event.type == spy.MouseEventType.button_up:
                app.mouse_up(point)

    def render(self, context):
        app = self.owner
        if self._inspector_shown and not self.inspector.visible:
            app.clear_selection()
        app.resize(context.surface_texture.width, context.surface_texture.height)
        bottom = app.brain_rect()[1] if app.brain_rect() is not None else app.texture.height
        for panel in (self.panel, self.inspector):
            limit = app.world_bottom() if panel is self.inspector else bottom
            available_height = max(1, limit - panel.position.y - 8)
            if panel.size.y > available_height:
                panel.size = spy.float2(panel.size.x, available_height)
        app.update(context.command_encoder)
        app.render(context.command_encoder)
        context.command_encoder.blit(context.surface_texture, app.texture)
        self.speed_selector.value = 0 if app.paused else SIM_SPEEDS.index(app.speed) + 1
        choices = app.world_choices()
        if self.world_selector.items != choices:
            self.world_selector.items = choices
        self.world_selector.value = app.selected_world + 1 if app.selected_world is not None else 0
        self.fit.enabled = app.selected_world is not None
        sim = app.view_sim
        label = choices[app.selected_world + 1] if app.viewing_benchmark else "Worms"
        self.panel.title = f"{label} | Tick {sim.tick}###controls"
        self.view_info.visible = app.viewing_benchmark
        self.view_info.text = (
            "Teal: current / Orange: baseline\nFixed benchmark settings; mutation disabled"
        )
        self.fit.label = f"Fit world ({app.camera.scale:.1f}x)"
        for control in (
            self.food_respawn,
            self.food_energy,
            self.food_age_start,
            self.food_age_fade,
            self.input_mutation,
            self.policy_mutation,
        ):
            control.enabled = not app.viewing_benchmark
        self.food_respawn.value = sim.food_per_second
        self.food_energy.value = sim.food_energy
        self.food_age_start.value = sim.food_age_start
        self.food_age_fade.value = sim.food_age_fade
        self.input_mutation.value = sim.input_mutation_probability * 100
        self.policy_mutation.value = sim.policy_mutation_probability * 100
        self.comparison_toggle.value = app.show_comparison
        self.comparison_toggle.enabled = app.benchmark is not None
        self.comparison_status.text = app.comparison_status()
        self.save_snapshot.enabled = app.snapshots is not None
        self.restore_snapshot.enabled = (
            app.snapshots is not None and app.snapshots.latest is not None
        )
        self.snapshot_status.text = app.snapshot_status if app.snapshots is not None else ""
        population = app.population_stats
        if population is not None:
            self.status.text = (
                f"Alive {population['num_creatures']}/{population['capacity']}"
                f"  Splits {population['births']}  Died {population['deaths']}"
                f"\nFood {population['num_food']}/{population['food_target']}"
            )
        stats = app.creature_stats
        if stats is not None:
            if not self.inspector.visible:
                self.inspector.position = spy.float2(max(16, app.texture.width - 356), 16)
                self.inspector.size = spy.float2(0, 0)
            self.follow.value = app.following
            age = sim.tick - stats["age_start_tick"]
            food_intake = (
                float(age < sim.food_age_start)
                if sim.food_age_fade == 0
                else max(0.0, min(1.0, 1.0 - (age - sim.food_age_start) / sim.food_age_fade))
            )
            cohort = (
                f"Cohort: {'Current' if stats['lineage'] == 1 else 'Baseline'}\n"
                if app.viewing_benchmark
                else ""
            )
            self.stats.text = (
                f"ID {stats['guid']}  Gen {stats['generation']}\n"
                f"{cohort}"
                f"Parent {stats['parent_guid'] or '-'}\n"
                f"Age {age} ticks\n"
                f"Food intake {food_intake:.0%}\n"
                f"Energy {stats['energy'] / ENERGY_PER_UNIT:.1f}\n"
                f"Split at {stats['split_energy'] / ENERGY_PER_UNIT:.1f}\n"
                f"X {stats['position'][0]:.1f}  Y {stats['position'][1]:.1f} m\n"
                f"Speed {math.hypot(*stats['velocity']):.2f} m/s\n"
                f"Length {stats['radii'][0] * 2:.2f} m\n"
                f"Width {stats['radii'][1] * 2:.2f} m"
            )
        self.inspector.visible = stats is not None
        self._inspector_shown = self.inspector.visible
        if app._frame_limit is not None and app.frame_count >= app._frame_limit:
            app.window_app.terminate()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--gpu-debug", action="store_true", help="Enable D3D12 and SlangPy GPU validation"
    )
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--creatures", type=int, default=64, help="Creatures per world")
    parser.add_argument("--frames", type=int, help="Frames before exit; headless defaults to 120")
    parser.add_argument("--output", type=Path, help="Save the final scene as a PNG")
    parser.add_argument("--speed", type=int, choices=SIM_SPEEDS, help="Ticks per render; default 1")
    parser.add_argument("--no-benchmark", action="store_true", help="Run only the live worlds")
    parser.add_argument("--no-snapshots", action="store_true")
    parser.add_argument("--snapshot-dir", type=Path, default=Path("artifacts/snapshots"))
    parser.add_argument("--resume", type=Path, help="Restore a snapshot path, or 'latest'")
    args = parser.parse_args(argv)
    if args.frames is not None and args.frames < 1:
        parser.error("--frames must be positive")
    resume = args.resume
    if resume == Path("latest"):
        resume = PeriodicSnapshots(args.snapshot_dir).latest
        if resume is None:
            parser.error("No automatic snapshots found in --snapshot-dir")
    app = App(
        headless=args.headless,
        num_worlds=args.worlds,
        creatures_per_world=args.creatures,
        benchmark=not args.no_benchmark,
        snapshot_dir=None if args.no_snapshots else args.snapshot_dir,
        resume=resume,
        debug_gpu=args.gpu_debug,
    )
    try:
        if args.speed is not None:
            app.speed = args.speed
        if args.no_benchmark:
            app.benchmark = None
        if args.headless:
            app.paused = False
        app.run(args.frames)
        output = args.output or (Path("artifacts/worlds.png") if args.headless else None)
        if output is not None:
            app.save(output)
            print(f"Saved tick {app.sim.tick} to {output.resolve()}")
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
