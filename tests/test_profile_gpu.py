"""The profiler must measure real GPU commands and restore wrapped kernels."""

import pytest
import slangpy as spy

from worms.app import App
from worms.profile import profile_app

pytestmark = pytest.mark.gpu


def test_gpu_profile_measures_dispatches_and_frame_without_changing_kernels():
    app = App(
        headless=True,
        creatures_per_world=2,
        creature_capacity=4,
        width=128,
        height=96,
        benchmark=False,
        snapshot_dir=None,
    )
    try:
        original_step = app.sim._step
        report = profile_app(app, steps_per_frame=2, samples=2, warmup=2)
        assert report["start_tick"] == 2
        assert report["end_tick"] == 10
        assert report["ticks_per_second"] > 0
        assert report["median_frame"]["gpu_ms"] > 0
        stages = {stage["name"]: stage for stage in report["passes"]}
        assert stages["live.step"]["dispatches"] == 4
        assert stages["live.evaluate_brains"]["dispatches"] == 4
        assert stages["render"]["dispatches"] == 2
        assert all(stage["mean_ms_per_dispatch"] >= 0 for stage in stages.values())
        assert 0 < sum(stage["percent_gpu_frame"] for stage in stages.values()) <= 100.01
        assert app.sim._step is original_step
        assert isinstance(app.sim._step, spy.ComputeKernel)
    finally:
        app.close()
