"""Windowed or headless entry point for the Worms application."""

import argparse
import math
from pathlib import Path

import slangpy as spy
from PIL import Image

from worms.sim import Sim


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


class App:
    """Own the device, simulation, frame target, and optional window."""

    def __init__(
        self, *, headless=False, num_worlds=4, creatures_per_world=64, width=1280, height=720
    ):
        adapter = next(
            a for a in spy.Device.enumerate_adapters(spy.DeviceType.d3d12) if "NVIDIA" in a.name
        )
        self.device = spy.Device(
            type=spy.DeviceType.d3d12,
            adapter_luid=adapter.luid,
            enable_debug_layers=True,
            compiler_options={"include_paths": [Path(__file__).parent / "shaders"]},
        )
        self.sim = Sim(self.device, num_worlds, creatures_per_world)
        self.texture = None
        self.resize(width, height)
        self.paused = False
        self.selected_world = None
        self.frame_count = 0
        self._step = False
        self._reset = False
        self._frame_limit = None
        self.window_app = None
        self.window = None
        if not headless:
            self.window_app = spy.App(self.device)
            self.window = _Window(self)

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

    def update(self, encoder):
        if self._reset:
            self.sim.reset(encoder)
        elif not self.paused or self._step:
            self.sim.update(encoder)
        self._reset = self._step = False
        self.frame_count += 1

    def render(self, encoder):
        self.sim.prepare_render(encoder)
        left = (
            0
            if self.window is None
            else min(int(self.window.panel.size.x) + 32, self.texture.width // 2)
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
            for world, rect in world_views(
                self.texture.width - left,
                self.texture.height,
                self.sim.num_worlds,
                self.selected_world,
            ):
                x, y, width, height = rect
                self.sim.render(render_pass, (x + left, y, width, height), world)

    def frame(self):
        """Submit one headless frame using the same update and render as the window."""
        encoder = self.device.create_command_encoder()
        self.update(encoder)
        self.render(encoder)
        self.device.submit_command_buffer(encoder.finish())

    def run(self, frames=None):
        if self.window_app is not None:
            self._frame_limit = None if frames is None else self.frame_count + frames
            self.window_app.run()
        else:
            for _ in range(120 if frames is None else frames):
                self.frame()

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
            title=(
                f"Worms | {owner.sim.num_worlds} worlds x {owner.sim.creatures_per_world} creatures"
            ),
            enable_vsync=True,
        )
        self.owner = owner
        self.panel = panel = spy.ui.Window(
            self.screen, title="Worms", position=spy.float2(16, 16), size=spy.float2(0, 0)
        )
        self.pause = spy.ui.CheckBox(
            panel, label="Paused", callback=lambda value: setattr(owner, "paused", value)
        )
        spy.ui.Button(panel, label="Step", callback=owner.step)
        spy.ui.Button(panel, label="Reset", callback=owner.reset)
        spy.ui.ComboBox(
            panel,
            label="",
            items=["All worlds"] + [f"World {i + 1}" for i in range(owner.sim.num_worlds)],
            callback=lambda value: setattr(owner, "selected_world", value - 1 if value else None),
        )
        self.status = spy.ui.Text(panel, text="Tick 0")

    def render(self, context):
        app = self.owner
        app.resize(context.surface_texture.width, context.surface_texture.height)
        app.update(context.command_encoder)
        app.render(context.command_encoder)
        context.command_encoder.blit(context.surface_texture, app.texture)
        self.pause.value = app.paused
        self.status.text = f"Tick {app.sim.tick}"
        if app._frame_limit is not None and app.frame_count >= app._frame_limit:
            app.window_app.terminate()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--creatures", type=int, default=64, help="Creatures per world")
    parser.add_argument("--frames", type=int, help="Frames before exit; headless defaults to 120")
    parser.add_argument("--output", type=Path, help="Save the final scene as a PNG")
    args = parser.parse_args(argv)
    if args.frames is not None and args.frames < 1:
        parser.error("--frames must be positive")
    app = App(headless=args.headless, num_worlds=args.worlds, creatures_per_world=args.creatures)
    try:
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
