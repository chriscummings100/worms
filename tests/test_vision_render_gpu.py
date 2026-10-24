"""Check that the selected overlay draws recorded sensors, including torus seams."""

import math

import numpy as np
import pytest
import slangpy as spy

from tests.gpu_helpers import fixed_brains
from worms.app import App
from worms.sim import FOOD_ENERGY, VISION_HISTORY, WORLD_SIZE, Sim

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def app():
    application = App(headless=True, num_worlds=1, creatures_per_world=0, width=800, height=640)
    yield application
    application.close()


def stage_scene(app, origin):
    sim = Sim(
        app.device,
        num_worlds=1,
        creatures_per_world=2,
        creature_capacity=2,
        food_capacity=1,
        initial_food=1,
        target_food=1,
        food_per_second=0,
    )
    app.sim = sim
    creatures = creature_cursor(sim)
    for index, delta in enumerate(((0, 0), (12, 6))):
        position = [(origin[axis] + delta[axis]) % WORLD_SIZE for axis in range(2)]
        creatures[index]["position"] = spy.float2(*position)
        creatures[index]["velocity"] = spy.float2(0, 0)
        creatures[index]["max_speed"] = 0.0
        creatures[index]["heading"] = 0.0
        creatures[index]["radii"] = spy.float2(0.5, 0.315)
    creatures.apply()
    food = food_cursor(sim)
    food[0]["position"] = spy.float2((origin[0] + 8) % WORLD_SIZE, (origin[1] - 4) % WORLD_SIZE)
    food[0]["energy"] = FOOD_ENERGY
    food.apply()
    fixed_brains(sim)
    sim.update()
    return sim


def creature_cursor(sim):
    layout = sim._initialize.reflection.g_creatures.type_layout.element_type_layout
    return spy.BufferCursor(layout, sim.creatures)


def food_cursor(sim):
    layout = sim._initialize_food.reflection.g_food.type_layout.element_type_layout
    return spy.BufferCursor(layout, sim.food)


def render(app, *, follow=False):
    encoder = app.device.create_command_encoder()
    app.sim.prepare_render(encoder)
    with encoder.begin_render_pass(
        {
            "color_attachments": [
                {
                    "view": app.texture.create_view({}),
                    "load_op": spy.LoadOp.clear,
                    "store_op": spy.StoreOp.store,
                }
            ]
        }
    ) as render_pass:
        app.sim.render(
            render_pass,
            (0, 0, app.texture.width, app.texture.height),
            0,
            scale=8,
            selected=0,
            follow=follow,
            wrap_view=True,
        )
    app.device.submit_command_buffer(encoder.finish())
    return app.texture.to_numpy()[:, :, :3].astype(np.int16)


def overlay_difference(app, *, follow=False):
    """Subtract the identical scene with its observation invalidated, isolating the overlay."""
    visible = render(app, follow=follow)
    sim = app.sim
    layout = sim._capture_vision.reflection.vision_frames.type_layout.element_type_layout
    index = (sim.tick % VISION_HISTORY) * sim.num_worlds * sim.creature_capacity
    frames = spy.BufferCursor(
        layout, sim.vision_frames, size=layout.stride, offset=index * layout.stride
    )
    frames[0]["tick"] = -1
    frames.apply()
    hidden = render(app, follow=follow)
    return visible - hidden


def test_overlay_uses_capture_pose_and_nearest_markers_after_scene_changes(app):
    sim = stage_scene(app, (256, 256))
    captured = sim.inspect_vision(0, 0)
    assert captured["heading"] == 0
    assert sum(slot["kind"] == 1 for slot in captured["slots"]) == 1
    assert sum(slot["kind"] == 2 for slot in captured["slots"]) == 1

    # Move the targets behind the observer and turn its body without recording
    # another observation. The overlay must continue showing the captured scene.
    creatures = creature_cursor(sim)
    creatures[0]["heading"] = math.pi
    creatures[1]["position"] = spy.float2(248, 252)
    creatures.apply()
    food = food_cursor(sim)
    food[0]["position"] = spy.float2(246, 256)
    food.apply()
    difference = overlay_difference(app)
    strength = np.max(np.abs(difference), axis=2)

    # The pose is at pixel (400,320), with 10 px/m. The forward arc ends at x640;
    # nothing from the old observation appears behind its recorded heading.
    assert np.count_nonzero(strength > 8) > 2000
    assert strength[:, :395].max() == 0
    assert strength[317:324, 638:643].max() > 30

    # Small bright rings surround the recorded centres, not the moved targets.
    # Checking all four sides distinguishes rings from the faint sector fill.
    for x, y, radius in ((480, 280, 8), (520, 380, 8)):
        for dx, dy in ((radius, 0), (-radius, 0), (0, radius), (0, -radius)):
            assert strength[y + dy - 1 : y + dy + 2, x + dx - 1 : x + dx + 2].max() > 60


def test_followed_overlay_keeps_geometry_when_scene_translates_across_both_seams(app):
    differences = []
    for origin in ((500, 500), (2, 2)):
        stage_scene(app, origin)
        differences.append(overlay_difference(app, follow=True))

    # The same relative scene crosses each world seam, but follow keeps its
    # captured fan and markers fixed on screen. Background tint can alter the
    # blended RGB result by a few quantization levels, never its geometry.
    assert np.count_nonzero(np.max(np.abs(differences[0]), axis=2) > 8) > 2000
    error = np.abs(differences[0] - differences[1])
    assert np.quantile(error, 0.999) <= 2
    assert error.max() <= 5


@pytest.mark.parametrize("front_kind", [1, 2])
def test_overlay_marks_only_the_nearest_object_in_a_segment(app, front_kind):
    sim = stage_scene(app, (256, 256))
    # Keep targets away from the segment boundary so the faint fan cannot be
    # mistaken for the bright marker surrounding the hidden target.
    angle = 0.1
    front = (256 + 8 * math.cos(angle), 256 + 8 * math.sin(angle))
    back = (256 + 16 * math.cos(angle), 256 + 16 * math.sin(angle))
    creatures = creature_cursor(sim)
    creatures[1]["position"] = spy.float2(*(back if front_kind == 1 else front))
    creatures.apply()
    food = food_cursor(sim)
    food[0]["position"] = spy.float2(*(front if front_kind == 1 else back))
    food.apply()
    sim.update()
    assert sim.inspect_vision(0, 0)["slots"][8]["kind"] == front_kind
    difference = overlay_difference(app)
    strength = np.max(np.abs(difference), axis=2)
    front_pixel = (round(400 + (front[0] - 256) * 10), round(320 + (front[1] - 256) * 10))
    back_pixel = (round(400 + (back[0] - 256) * 10), round(320 + (back[1] - 256) * 10))
    x, y = front_pixel
    assert strength[y - 10 : y + 11, x - 10 : x + 11].max() > 60
    x, y = back_pixel
    assert strength[y - 10 : y + 11, x - 10 : x + 11].max() < 30
