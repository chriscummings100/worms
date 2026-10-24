"""Validate the public render inputs without creating or importing a GPU runtime."""

import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from worms.render import RenderConfig


@pytest.mark.parametrize(
    "overrides",
    [
        {"width": 0},
        {"width": -1},
        {"width": 8193},
        {"width": 1280.5},
        {"width": True},
        {"height": 0},
        {"height": 8193},
        {"height": "720"},
        {"count": 0},
        {"count": 4097},
        {"count": 5.5},
        {"count": False},
        {"seed": -1},
        {"seed": 2**32},
        {"seed": 1.5},
        {"seed": True},
        {"time_seconds": float("nan")},
        {"time_seconds": float("inf")},
        {"time_seconds": float("-inf")},
        {"backend": "unsupported"},
        {"adapter": ""},
        {"adapter": "   "},
    ],
)
def test_rejects_invalid_render_configuration(overrides):
    with pytest.raises((TypeError, ValueError)):
        RenderConfig(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"width": 1, "height": 1, "count": 1, "seed": 0},
        {"width": 8192, "height": 8192, "count": 4096, "seed": 2**32 - 1},
        {"width": 257, "height": 149, "time_seconds": 1.25, "backend": "vulkan"},
    ],
)
def test_accepts_supported_configuration_boundaries(overrides):
    config = RenderConfig(**overrides)
    for name, value in overrides.items():
        assert getattr(config, name) == value


def test_configuration_cannot_change_after_validation():
    config = RenderConfig()
    with pytest.raises(FrozenInstanceError):
        config.seed = 42


def test_import_does_not_load_slangpy():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import worms.render; "
            "assert not any(name == 'slangpy' or name.startswith('slangpy.') "
            "for name in sys.modules), 'importing worms.render loaded SlangPy'",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
