"""Benchmark views use their own GPU state for drawing and inspection."""

import numpy as np
import pytest
import slangpy as spy

from worms.app import App
from worms.benchmark import Benchmark
from worms.sim import BRAIN_ACTIVATIONS

pytestmark = pytest.mark.gpu


@pytest.fixture
def app():
    application = App(
        debug_gpu=True,
        headless=True,
        benchmark=False,
        snapshot_dir=None,
        num_worlds=2,
        creatures_per_world=2,
        creature_capacity=8,
        width=640,
        height=360,
    )
    application.benchmark = Benchmark(application.sim, trial_ticks=2, sample_size=1, warmup_ticks=0)
    application.paused = True
    sim = application.benchmark.sim
    layout = sim._initialize.reflection.g_creatures.type_layout.element_type_layout
    cursor = spy.BufferCursor(layout, sim.creatures)
    for world, color in enumerate(((1.0, 0.1, 0.1), (0.1, 0.1, 1.0))):
        creature = cursor[world * sim.creature_capacity]
        creature["position"] = spy.float2(256, 256)
        creature["radii"] = spy.float2(24, 16)
        creature["color"] = spy.float3(*color)
    cursor.apply()
    yield application
    application.close()


def direct_render(app, sim, world):
    """Draw a known Sim/world directly, independently of App's view routing."""
    encoder = app.device.create_command_encoder()
    sim.prepare_render(encoder)
    brain = app.brain_rect()
    if brain is not None:
        app.brain_view.prepare(encoder, brain)
    with encoder.begin_render_pass(
        {
            "color_attachments": [
                {
                    "view": app.texture.create_view({}),
                    "load_op": spy.LoadOp.clear,
                    "store_op": spy.StoreOp.store,
                    "clear_value": [0.008, 0.014, 0.02, 1.0],
                }
            ]
        }
    ) as render_pass:
        sim.render(
            render_pass,
            app.focused_rect(),
            world,
            offset=app.camera.offset,
            scale=app.camera.scale,
            selected=app.selected_creature if app.selected_creature is not None else -1,
            selected_guid=app.selected_guid,
            follow_position=app._follow_position(),
            follow=app.following,
            wrap_view=True,
        )
        if brain is not None:
            app.brain_view.render(
                render_pass,
                brain,
                sim.brain_activations,
                sim.creatures,
                world * sim.creature_capacity + app.selected_creature,
                app.selected_guid,
            )
    app.device.submit_command_buffer(encoder.finish())
    return app.texture.to_numpy().copy()


@pytest.mark.parametrize("world", [0, 1])
def test_benchmark_view_matches_direct_world_render(app, world):
    app.selected_world = app.sim.num_worlds + world
    assert app.viewing_benchmark
    assert app.view_sim is app.benchmark.sim
    assert app.view_world == world

    app.frame()
    actual = app.texture.to_numpy().copy()
    expected = direct_render(app, app.benchmark.sim, world)
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(actual, direct_render(app, app.benchmark.sim, 1 - world))
    assert not np.array_equal(actual, direct_render(app, app.sim, world))


def test_benchmark_b_picking_inspection_follow_and_brain_use_local_world(app):
    sim = app.benchmark.sim
    app.selected_world = app.sim.num_worlds + 1
    rect = app.focused_rect()
    app.pick((rect[0] + rect[2] / 2, rect[1] + rect[3] / 2))
    assert app.selected_creature == 0
    assert app.creature_stats == sim.inspect(1, 0)
    assert app.selected_guid == sim.inspect(1, 0)["guid"]

    layout = sim._initialize.reflection.g_worlds.type_layout.element_type_layout
    worlds = spy.BufferCursor(layout, sim.worlds)
    worlds[1]["births"] = 7
    worlds[1]["deaths"] = 3
    worlds.apply()
    app.refresh_population()
    assert app.population_stats["births"] == 7
    assert app.population_stats["deaths"] == 3
    assert app.population_stats["num_creatures"] == sim.inspect_world(1)["num_creatures"]
    assert app.population_stats["capacity"] == sim.creature_capacity

    activations = sim.brain_activations.to_numpy().view(np.float16).reshape(-1, BRAIN_ACTIVATIONS)
    activations[:] = 0
    activations[sim.creature_capacity] = 1
    sim.brain_activations.copy_from_numpy(activations)
    app.set_follow(True)
    assert app.following
    app.frame()
    actual = app.texture.to_numpy().copy()
    np.testing.assert_array_equal(actual, direct_render(app, sim, 1))

    # Removing only B's activation proves the rendered strip reads that slot.
    activations[sim.creature_capacity] = 0
    sim.brain_activations.copy_from_numpy(activations)
    quiet = direct_render(app, sim, 1)
    top = app.brain_rect()[1]
    assert not np.array_equal(actual[top:], quiet[top:])


@pytest.mark.parametrize("finish_with_run", [False, True])
def test_trial_restart_keeps_benchmark_view_but_clears_creature_identity(app, finish_with_run):
    selected = app.sim.num_worlds + 1
    app.selected_world = selected
    app.selected_creature = 0
    app.refresh_stats()
    app.set_follow(True)
    app.speed = 2
    app.paused = False
    if finish_with_run:
        app.run(frames=1)
    else:
        app.frame()
        assert app.benchmark.pending
        app.paused = True
        app.frame()
    assert app.benchmark.trial == 1
    assert app.selected_world == selected
    assert app.selected_creature is None
    assert app.selected_guid == 0
    assert app.creature_stats is None
    assert not app.following
    assert app.view_sim is app.benchmark.sim


def test_world_choices_and_switching_back_to_live_clear_selection(app):
    assert app.world_choices() == [
        "All live worlds",
        "World 1",
        "World 2",
        "Benchmark A",
        "Benchmark B",
    ]
    app.selected_world = app.sim.num_worlds
    app.selected_creature = 0
    app.refresh_stats()
    app.set_follow(True)
    app.selected_world = 1
    assert not app.viewing_benchmark
    assert app.view_sim is app.sim
    assert app.view_world == 1
    assert app.selected_creature is None
    assert not app.following

    app.benchmark = None
    assert app.world_choices() == ["All live worlds", "World 1", "World 2"]
    app.selected_world = None
    assert len(app.views()) == 2
    app.frame()


def test_waiting_benchmark_views_appear_only_after_submitted_baseline_capture(app):
    app.benchmark = Benchmark(app.sim, trial_ticks=10, sample_size=1, warmup_ticks=3)
    live_choices = ["All live worlds", "World 1", "World 2"]
    assert app.world_choices() == live_choices
    assert "Waiting to capture baseline" in app.comparison_status()
    app.paused = False
    app.speed = 2
    app.frame()
    app.frame()
    assert app.sim.tick == 4
    assert app.benchmark.sim.tick == 0 and not app.benchmark.ready
    assert app.world_choices() == live_choices
    assert "Live tick 4/3" in app.comparison_status()

    app.paused = True
    app.frame()  # Capture submitted tick 4, before recording any further updates.
    assert app.benchmark.reference_tick == 4
    assert app.world_choices() == live_choices + ["Benchmark A", "Benchmark B"]
    assert "Baseline captured at live tick 4" in app.comparison_status()
    app.selected_world = app.sim.num_worlds
    assert app.view_sim is app.benchmark.sim
    app.frame()


def test_reset_from_benchmark_view_returns_to_live_while_baseline_warms_up(app):
    app.selected_world = app.sim.num_worlds + 1
    app.selected_creature = 0
    app.refresh_stats()
    app.set_follow(True)
    app.reset()
    assert app.selected_world is None
    assert not app.viewing_benchmark and not app.following
    assert app.selected_creature is None
    app.frame()
    assert app.view_sim is app.sim
    assert not app.benchmark.ready
    assert app.world_choices() == ["All live worlds", "World 1", "World 2"]
    assert "Waiting to capture baseline" in app.comparison_status()
