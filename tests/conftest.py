"""Keep ordinary tests hardware independent; require real rendering when opted in."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run tests requiring a real GPU and SlangPy graphics backend.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-gpu"):
        return

    skip_gpu = pytest.mark.skip(reason="GPU tests require explicit --run-gpu")
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(skip_gpu)
