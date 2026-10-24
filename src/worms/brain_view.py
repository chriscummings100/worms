"""A small GPU activation display with labels uploaded only when it is resized."""

from pathlib import Path

import numpy as np
import slangpy as spy
from PIL import Image, ImageDraw, ImageFont

BRAIN_PANEL_HEIGHT = 192
BRAIN_CONTENT_WIDTH = 1280


def brain_content_transform(width, height):
    scale = min(1.0, width / BRAIN_CONTENT_WIDTH, height / BRAIN_PANEL_HEIGHT)
    offset = (
        (width - BRAIN_CONTENT_WIDTH * scale) / 2,
        (height - BRAIN_PANEL_HEIGHT * scale) / 2,
    )
    return scale, offset


def brain_panel_rect(width, height):
    panel_height = min(BRAIN_PANEL_HEIGHT, height // 3)
    return (0, height - panel_height, width, panel_height)


class BrainView:
    """Read a selected creature's activation buffer directly during drawing."""

    def __init__(self, device):
        self.device = device
        shader = Path(__file__).parent / "shaders" / "brain_render.slang"
        self.background = device.create_render_pipeline(
            program=device.load_program(str(shader), ["label_vertex", "label_fragment"]),
            input_layout=None,
            targets=[{"format": spy.Format.rgba8_unorm_srgb}],
            rasterizer={"cull_mode": spy.CullMode.none},
        )
        blend = {
            "src_factor": spy.BlendFactor.one,
            "dst_factor": spy.BlendFactor.inv_src_alpha,
        }
        self.neurons = device.create_render_pipeline(
            program=device.load_program(str(shader), ["neuron_vertex", "neuron_fragment"]),
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
        self.labels = None
        self._size = None

    def prepare(self, encoder, rect):
        """Prepare static labels before the render pass, without reading brain state."""
        size = rect[2:]
        if self._size != size:
            background = (18, 26, 33, 255)
            image = Image.new("RGBA", (BRAIN_CONTENT_WIDTH, BRAIN_PANEL_HEIGHT), background)
            draw = ImageDraw.Draw(image)
            title_font = ImageFont.load_default(size=15)
            font = ImageFont.load_default(size=13)
            quiet = (146, 165, 177)
            bright = (216, 229, 235)
            draw.line((666, 20, 666, 172), fill=(41, 56, 67))
            draw.line((952, 20, 952, 172), fill=(41, 56, 67))
            draw.text((20, 15), "CURRENT VISION / 64 INPUTS", font=title_font, fill=bright)
            draw.text((710, 15), "16 HIDDEN / tanh", font=title_font, fill=bright)
            draw.text((987, 15), "2 MOTOR OUTPUTS", font=title_font, fill=bright)
            for row, label in enumerate(("Proximity", "Size", "Is food", "Is creature")):
                draw.text((20, 62 + row * 26), label, font=font, fill=quiet)
            draw.text((182, 169), "LEFT", font=font, fill=quiet)
            draw.text((355, 169), "FORWARD", font=font, fill=quiet)
            draw.text((550, 169), "RIGHT", font=font, fill=quiet)
            draw.text((619, 94), ">", font=title_font, fill=quiet)
            draw.text((710, 166), "Blue -    Amber +    Dim 0", font=font, fill=quiet)
            draw.text((1024, 56), "Steering", font=font, fill=quiet)
            draw.text((1024, 104), "Movement", font=font, fill=quiet)
            draw.text((1030, 143), "Stop", font=font, fill=quiet)
            draw.text((1204, 143), "Full", font=font, fill=quiet)
            draw.text((1030, 93), "L", font=font, fill=quiet)
            draw.text((1125, 93), "0", font=font, fill=quiet)
            draw.text((1235, 93), "R", font=font, fill=quiet)
            scale, (left, top) = brain_content_transform(*size)
            image = image.transform(
                size,
                Image.Transform.AFFINE,
                (1 / scale, 0, -left / scale, 0, 1 / scale, -top / scale),
                resample=Image.Resampling.BICUBIC,
                fillcolor=background,
            )
            ImageDraw.Draw(image).line((0, 0, size[0], 0), fill=(58, 78, 89), width=1)
            self.labels = self.device.create_texture(
                width=size[0],
                height=size[1],
                format=spy.Format.rgba8_unorm_srgb,
                usage=spy.TextureUsage.shader_resource,
                label="brain-labels",
            )
            self.labels.copy_from_numpy(np.array(image))
            self._size = size
        encoder.set_texture_state(self.labels, spy.ResourceState.shader_resource)

    def render(self, render_pass, rect, activations, creatures, creature_index, selected_guid=0):
        x, y, width, height = rect
        render_pass.set_render_state(
            {
                "viewports": [spy.Viewport({"x": x, "y": y, "width": width, "height": height})],
                "scissor_rects": [
                    spy.ScissorRect(
                        {"min_x": x, "min_y": y, "max_x": x + width, "max_y": y + height}
                    )
                ],
            }
        )
        cursor = spy.ShaderCursor(render_pass.bind_pipeline(self.background))
        cursor["labels"] = self.labels
        cursor["resolution"] = spy.float2(width, height)
        render_pass.draw({"vertex_count": 6, "instance_count": 1})
        cursor = spy.ShaderCursor(render_pass.bind_pipeline(self.neurons))
        cursor["activations"] = activations
        cursor["creatures"] = creatures
        cursor["creature_index"] = creature_index
        cursor["selected_guid"] = selected_guid
        scale, offset = brain_content_transform(width, height)
        cursor["resolution"] = spy.float2(width, height)
        cursor["content_scale"] = scale
        cursor["content_offset"] = spy.float2(*offset)
        render_pass.draw({"vertex_count": 6, "instance_count": 84})
