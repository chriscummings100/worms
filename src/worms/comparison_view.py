"""A small history plot, uploaded only when its samples or size change."""

import numpy as np
import slangpy as spy
from PIL import Image, ImageDraw, ImageFont


def comparison_result(sample):
    if sample is None:
        return "No result yet"
    if sample["score"] is None:
        return {
            "no_current_population": "No live population to sample",
            "no_reference_population": "No reference population to sample",
            "no_energy_collected": "No food energy collected: no score",
        }.get(sample.get("reason"), "No score")
    return f"Current energy share {sample['score']:.1%} (50% = equal)"


def comparison_image(size, history):
    width, height = size
    image = Image.new("RGBA", size, (18, 26, 33, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=13)
    quiet, bright, cyan = (146, 165, 177), (216, 229, 235), (72, 213, 192)
    draw.text((16, 10), "CURRENT vs BASELINE | food energy collected", font=font, fill=bright)
    if width < 200 or height < 100:
        return image
    left, top, right, bottom = 54, 43, width - 24, height - 32
    for score, label in ((0, "0%"), (0.5, "50%"), (1, "100%")):
        y = bottom - score * (bottom - top)
        draw.text((10, y - 7), label, font=font, fill=quiet)
        if score == 0.5:
            for x in range(left, right, 10):
                draw.line((x, y, min(x + 5, right), y), fill=(87, 106, 117))
        else:
            draw.line((left, y, right, y), fill=(41, 56, 67))
    last_tick = max((point["live_tick"] for point in history), default=0)
    # Include zero so consecutive trials keep a common visual origin.
    limit = max(last_tick, 1)
    previous = None
    for point in history:
        if point["score"] is None:
            previous = None
            continue
        x = left + point["live_tick"] / limit * (right - left)
        y = bottom - point["score"] * (bottom - top)
        if previous is not None:
            draw.line((*previous, x, y), fill=cyan, width=2)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=cyan)
        previous = x, y
    draw.text((left, height - 23), "0", font=font, fill=quiet)
    draw.text((right, height - 23), f"{last_tick:,} live ticks", anchor="ra", font=font, fill=quiet)
    if not history:
        draw.text((left + 10, top + 8), "Waiting for the first trial", font=font, fill=quiet)
    else:
        draw.text((left + 10, top + 8), comparison_result(history[-1]), font=font, fill=bright)
    return image


class ComparisonView:
    def __init__(self, device, pipeline):
        self.device = device
        self.pipeline = pipeline
        self.texture = None
        self._key = None

    def prepare(self, encoder, rect, history):
        size = rect[2:]
        key = (
            size,
            tuple(
                (sample["live_tick"], sample["score"], sample.get("reason")) for sample in history
            ),
        )
        if key != self._key:
            self.texture = self.device.create_texture(
                width=size[0],
                height=size[1],
                format=spy.Format.rgba8_unorm_srgb,
                usage=spy.TextureUsage.shader_resource,
                label="comparison-history",
            )
            self.texture.copy_from_numpy(np.array(comparison_image(size, history)))
            self._key = key
        encoder.set_texture_state(self.texture, spy.ResourceState.shader_resource)

    def render(self, render_pass, rect):
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
        cursor = spy.ShaderCursor(render_pass.bind_pipeline(self.pipeline))
        cursor["labels"] = self.texture
        cursor["resolution"] = spy.float2(width, height)
        render_pass.draw({"vertex_count": 6, "instance_count": 1})
