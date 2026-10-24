"""GPU state, fixed-step motion, and drawing for a small batch of worlds."""

from pathlib import Path

import slangpy as spy

SHADER_DIR = Path(__file__).parent / "shaders"
WORLD_SIZE = 256.0
TIME_STEP = 1.0 / 60.0


class Sim:
    """N independent worlds, each with a fixed block of creature records on the GPU.

    All creatures stay alive in this first version, so every block is already
    compact. Simulation and drawing share these buffers without CPU readback.
    """

    def __init__(self, device, num_worlds=4, creatures_per_world=64, seed=1):
        if num_worlds < 1 or creatures_per_world < 0:
            raise ValueError("Need at least one world and a nonnegative creature count")
        self.device = device
        self.num_worlds = num_worlds
        self.creatures_per_world = creatures_per_world
        self.seed = seed
        self.tick = 0

        self._initialize = device.create_compute_kernel(
            program=device.load_program(str(SHADER_DIR / "world.slang"), ["initialize"])
        )
        self._step = device.create_compute_kernel(
            program=device.load_program(str(SHADER_DIR / "world.slang"), ["step"])
        )
        usage = spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource
        self.worlds = device.create_buffer(
            element_count=num_worlds,
            resource_type_layout=self._initialize.reflection.g_worlds,
            usage=usage,
            label="worlds",
        )
        self.creatures = device.create_buffer(
            element_count=max(1, num_worlds * creatures_per_world),
            resource_type_layout=self._initialize.reflection.g_creatures,
            usage=usage,
            label="creatures",
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
        self.reset()

    def reset(self, command_encoder=None):
        """Reinitialize GPU state from the seed, submitting when no encoder is supplied."""
        encoder = command_encoder or self.device.create_command_encoder()
        self._initialize.dispatch(
            thread_count=[max(self.num_worlds, self.num_worlds * self.creatures_per_world), 1, 1],
            command_encoder=encoder,
            vars={
                "num_worlds": self.num_worlds,
                "creatures_per_world": self.creatures_per_world,
                "seed": self.seed,
                "g_worlds": self.worlds,
                "g_creatures": self.creatures,
            },
        )
        self.tick = 0
        if command_encoder is None:
            self.device.submit_command_buffer(encoder.finish())

    def update(self, command_encoder=None):
        """Record one 1/60 second step; supplied encoders keep update and draw ordered."""
        encoder = command_encoder or self.device.create_command_encoder()
        self._step.dispatch(
            thread_count=[max(1, self.num_worlds * self.creatures_per_world), 1, 1],
            command_encoder=encoder,
            vars={
                "num_worlds": self.num_worlds,
                "creatures_per_world": self.creatures_per_world,
                "g_creatures": self.creatures,
            },
        )
        self.tick += 1
        if command_encoder is None:
            self.device.submit_command_buffer(encoder.finish())

    def prepare_render(self, command_encoder):
        """Transition compute output before opening the render pass."""
        command_encoder.set_buffer_state(self.worlds, spy.ResourceState.shader_resource)
        command_encoder.set_buffer_state(self.creatures, spy.ResourceState.shader_resource)

    def render(self, pass_encoder, rect, world, offset=(0.0, 0.0), scale=1.0):
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
        for pipeline, count in ((self._background, 1), (self._draw, self.creatures_per_world * 9)):
            if count == 0:
                continue
            cursor = spy.ShaderCursor(pass_encoder.bind_pipeline(pipeline))
            cursor["resolution"] = spy.float2(width, height)
            cursor["offset"] = spy.float2(*offset)
            cursor["scale"] = scale
            cursor["world_index"] = world
            if pipeline is self._draw:
                cursor["creatures_per_world"] = self.creatures_per_world
                cursor["creatures"] = self.creatures
                cursor["worlds"] = self.worlds
            pass_encoder.draw({"vertex_count": 6, "instance_count": count})
