"""Small end-to-end checks that fail honestly if explicitly requested GPU work fails."""

import json
import math

import pytest
from PIL import Image

from worms.render import RenderConfig, render_smoke


@pytest.mark.gpu
@pytest.mark.parametrize("count", [5, 256])
def test_gpu_render_writes_image_and_measured_manifest(tmp_path, count):
    # Odd frame sizes and counts below/above a 64-thread group exercise bounds.
    config = RenderConfig(width=257, height=149, count=count, seed=17, time_seconds=0.5)
    output = tmp_path / f"smoke-{count}.png"

    manifest = render_smoke(output, config)

    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.size == (config.width, config.height)
        assert image.mode == "RGBA"
        channel_ranges = image.getextrema()
        assert channel_ranges[3] == (255, 255), "Every output pixel must be opaque"
        assert all(high - low >= 20 for low, high in channel_ranges[:3])
        assert image.getcolors(maxcolors=16) is None, "Expected a shaded, nonuniform scene"

    saved_manifest = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved_manifest == manifest
    for name in ("width", "height", "count", "seed", "backend"):
        assert manifest[name] == getattr(config, name)
    assert manifest["debug_layers"] is config.debug
    for name in ("adapter", "python", "slangpy"):
        assert isinstance(manifest[name], str) and manifest[name].strip()
    for name in ("compute_ms", "draw_ms"):
        assert isinstance(manifest[name], (int, float))
        assert math.isfinite(manifest[name]) and manifest[name] >= 0
