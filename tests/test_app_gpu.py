"""Exercise real D3D12 state and rendering; opt in with --run-gpu."""

import numpy as np
import pytest
from PIL import Image

from tests.gpu_helpers import fixed_brains
from worms.sim import TIME_STEP, WORLD_SIZE, Sim

pytestmark = pytest.mark.gpu


@pytest.fixture
def app():
    from worms.app import App

    # Odd dimensions and a partial compute group catch out-of-range writes.
    application = App(
        debug_gpu=True,
        headless=True,
        benchmark=False,
        snapshot_dir=None,
        num_worlds=3,
        creatures_per_world=17,
        creature_capacity=17,
        width=257,
        height=149,
    )
    yield application
    application.close()


def creature_cursor(sim):
    import slangpy as spy

    layout = sim._initialize.reflection.g_creatures.type_layout.element_type_layout
    return spy.BufferCursor(layout, sim.creatures)


def positions(sim):
    cursor = creature_cursor(sim)
    values = [cursor[i]["position"].read() for i in range(sim.num_worlds * sim.creatures_per_world)]
    return np.array([(value.x, value.y) for value in values], dtype=np.float32)


def test_worlds_have_separate_seeded_creatures_and_unique_ids(app):
    import slangpy as spy

    sim = app.sim
    layout = sim._initialize.reflection.g_worlds.type_layout.element_type_layout
    worlds = spy.BufferCursor(layout, sim.worlds)
    counts = [worlds[i]["num_creatures"].read() for i in range(sim.num_worlds)]
    np.testing.assert_array_equal(counts, [17, 17, 17])

    cursor = creature_cursor(sim)
    ids = [cursor[i]["guid"].read() for i in range(51)]
    assert len(set(ids)) == 51
    assert all(cursor[i]["alive"].read() for i in range(51))
    seeded = positions(sim).reshape(3, 17, 2)
    assert np.isfinite(seeded).all()
    assert ((seeded >= 0) & (seeded < WORLD_SIZE)).all()
    assert not np.array_equal(seeded[0], seeded[1])
    assert not np.array_equal(seeded[1], seeded[2])


def test_creature_body_size_is_one_metre_nominal_and_independent_of_population(app):
    sim = app.sim
    larger_population = Sim(app.device, sim.num_worlds, creatures_per_world=256, seed=sim.seed)
    cursors = creature_cursor(sim), creature_cursor(larger_population)
    sizes = []
    for population, cursor in zip((sim, larger_population), cursors, strict=True):
        values = [
            cursor[world * population.creature_capacity + local]["radii"].read()
            for world in range(sim.num_worlds)
            for local in range(sim.creatures_per_world)
        ]
        sizes.append(np.array([(value.x, value.y) for value in values]))

    np.testing.assert_array_equal(*sizes)
    lengths = sizes[0][:, 0] * 2
    assert ((lengths >= 0.8) & (lengths <= 1.2)).all()
    np.testing.assert_allclose(sizes[0][:, 1] / sizes[0][:, 0], 0.63)


def test_updates_move_persistent_gpu_state_and_reset_replays_seed(app):
    sim = app.sim
    buffer = sim.creatures
    initial = buffer.to_numpy().copy()
    before = positions(sim)

    sim.update()
    sim.update()

    assert sim.tick == 2
    assert sim.creatures is buffer
    after = positions(sim)
    assert np.any(after != before)
    assert np.isfinite(after).all()
    assert ((after >= 0) & (after < WORLD_SIZE)).all()
    sim.reset()
    assert sim.tick == 0
    assert sim.creatures is buffer
    np.testing.assert_array_equal(buffer.to_numpy(), initial)


def test_movement_wraps_both_axes_without_changing_identity(app):
    import slangpy as spy

    sim = app.sim
    cursor = creature_cursor(sim)
    edge = WORLD_SIZE - 0.01
    starts = np.array([(edge, 0.01), (0.01, edge), (edge, edge), (0.01, 0.01)], dtype=np.float32)
    velocities = np.array([(6, -6), (-6, 6), (6, 6), (-6, -6)], dtype=np.float32)
    ids = [cursor[i]["guid"].read() for i in range(4)]
    for i, (position, velocity) in enumerate(zip(starts, velocities, strict=True)):
        cursor[i]["position"] = spy.float2(*position)
        cursor[i]["velocity"] = spy.float2(*velocity)
        cursor[i]["heading"] = np.arctan2(velocity[1], velocity[0])
        cursor[i]["max_speed"] = float(np.linalg.norm(velocity))
    cursor.apply()
    fixed_brains(sim)

    sim.update()

    np.testing.assert_allclose(
        positions(sim)[:4],
        (starts + velocities * TIME_STEP) % WORLD_SIZE,
        rtol=0,
        atol=np.spacing(np.float32(WORLD_SIZE)),
    )
    updated = creature_cursor(sim)
    assert [updated[i]["guid"].read() for i in range(4)] == ids


def test_pause_single_step_and_reset_control_the_next_frame(app):
    initial = app.sim.creatures.to_numpy().copy()
    app.paused = True
    app.frame()
    assert app.sim.tick == 0
    np.testing.assert_array_equal(app.sim.creatures.to_numpy(), initial)

    app.step()
    app.frame()
    assert app.paused
    assert app.sim.tick == 1
    stepped = app.sim.creatures.to_numpy().copy()
    assert not np.array_equal(stepped, initial)
    app.frame()
    assert app.sim.tick == 1
    np.testing.assert_array_equal(app.sim.creatures.to_numpy(), stepped)

    app.paused = False
    app.reset()
    app.frame()
    assert app.sim.tick == 0
    np.testing.assert_array_equal(app.sim.creatures.to_numpy(), initial)
    app.frame()
    assert app.sim.tick == 1
    np.testing.assert_array_equal(app.sim.creatures.to_numpy(), stepped)


def test_fast_frame_matches_the_same_number_of_individual_fixed_ticks(app):
    sim = app.sim
    reference = Sim(
        app.device,
        num_worlds=sim.num_worlds,
        creatures_per_world=sim.creatures_per_world,
        creature_capacity=sim.creature_capacity,
        seed=sim.seed,
    )
    app.speed = 100

    app.frame()
    for _ in range(100):
        reference.update()

    assert sim.tick == reference.tick == 100
    for name in (
        "worlds",
        "creatures",
        "food",
        "vision_frames",
        "vision_history",
        "brain_weights",
        "brain_activations",
    ):
        np.testing.assert_array_equal(
            getattr(sim, name).to_numpy(), getattr(reference, name).to_numpy(), err_msg=name
        )


def test_high_speed_keeps_pause_step_reset_and_live_food_settings(app):
    from worms.sim import ENERGY_PER_UNIT

    initial = app.sim.creatures.to_numpy().copy()
    app.speed = 100
    app.paused = True
    app.sim.food_energy = 3.5
    app.frame()
    assert app.sim.tick == 0
    np.testing.assert_array_equal(app.sim.creatures.to_numpy(), initial)
    food = app.sim.food.to_numpy().view(np.uint32).reshape(-1, 3)
    assert set(food[:, 2]) == {0, int(3.5 * ENERGY_PER_UNIT)}

    app.step()
    app.frame()
    assert app.paused and app.sim.tick == 1 and app.speed == 100
    app.frame()
    assert app.sim.tick == 1

    app.step()
    app.reset()
    app.frame()
    assert app.sim.tick == 0 and app.speed == 100
    assert app.sim.food_energy == 3.5
    np.testing.assert_array_equal(app.sim.creatures.to_numpy(), initial)
    app.frame()
    assert app.sim.tick == 0

    app.paused = False
    app.frame()
    assert app.sim.tick == 100


def test_one_metre_creature_is_visible_at_inspection_zoom(app):
    import slangpy as spy

    sim = app.sim
    cursor = creature_cursor(sim)
    for index in range(sim.creatures_per_world):
        cursor[index]["alive"] = False
    # A lone survivor can occupy any reserved slot after deaths and births.
    index = sim.creature_capacity - 1
    worlds = spy.BufferCursor(
        sim._initialize.reflection.g_worlds.type_layout.element_type_layout, sim.worlds
    )
    worlds[0]["num_creatures"] = 1
    worlds.apply()
    cursor[index]["position"] = spy.float2(WORLD_SIZE / 2, WORLD_SIZE / 2)
    cursor[index]["radii"] = spy.float2(0.5, 0.315)
    cursor[index]["velocity"] = spy.float2(1, 0)
    cursor[index]["heading"] = 0.0
    images = []
    for alive in (False, True):
        cursor[index]["alive"] = alive
        cursor.apply()
        encoder = app.device.create_command_encoder()
        sim.prepare_render(encoder)
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
            sim.render(render_pass, (0, 0, app.texture.width, app.texture.height), 0, scale=32)
        app.device.submit_command_buffer(encoder.finish())
        app.device.wait()
        images.append(app.texture.to_numpy().copy())

    changed = np.any(images[0][:, :, :3] != images[1][:, :, :3], axis=2)
    assert np.count_nonzero(changed) > 20
    assert changed[app.texture.height // 2, app.texture.width // 2]


def test_grid_and_focused_worlds_render_png_without_advancing_paused_sim(app, tmp_path):
    app.paused = True
    initial = app.sim.creatures.to_numpy().copy()
    images = []
    for selected in (None, 0, 1):
        app.selected_world = selected
        app.frame()
        output = tmp_path / f"world-{selected}.png"
        app.save(output)
        with Image.open(output) as image:
            assert image.format == "PNG"
            assert image.size == (257, 149)
            assert image.mode == "RGBA"
            assert image.getextrema()[3] == (255, 255)
            assert image.getcolors(maxcolors=16) is None
            images.append(np.array(image))

    assert not np.array_equal(images[0], images[1])
    assert not np.array_equal(images[1], images[2])
    assert app.sim.tick == 0
    np.testing.assert_array_equal(app.sim.creatures.to_numpy(), initial)


def test_inspector_reads_the_requested_world_and_latest_creature_state(app):
    sim = app.sim
    fixed_brains(sim)  # A random policy can legitimately choose to stop.
    world, local_index = 2, 5
    index = world * sim.creatures_per_world + local_index
    before = sim.inspect(world, local_index)
    assert before["guid"] == index + 1

    sim.update()

    inspected = sim.inspect(world, local_index)
    expected = creature_cursor(sim)[index].read()
    for field in ("guid", "alive", "desired_action", "desired_target"):
        assert inspected[field] == expected[field]
    for field in ("position", "velocity", "radii", "color"):
        np.testing.assert_array_equal(inspected[field], tuple(expected[field]))
    assert inspected["position"] != before["position"]
    assert inspected["guid"] == before["guid"]


def test_picking_uses_the_visible_world_and_ignores_dead_creatures(app):
    import slangpy as spy

    sim = app.sim
    cursor = creature_cursor(sim)
    for index in range(sim.num_worlds * sim.creatures_per_world):
        cursor[index]["alive"] = False
        cursor[index]["position"] = spy.float2(100, 100)
    for world, local_index in ((0, 2), (1, 4)):
        index = world * sim.creatures_per_world + local_index
        cursor[index]["alive"] = True
    cursor.apply()
    worlds = spy.BufferCursor(
        sim._initialize.reflection.g_worlds.type_layout.element_type_layout, sim.worlds
    )
    for world in range(sim.num_worlds):
        worlds[world]["num_creatures"] = 1 if world < 2 else 0
    worlds.apply()

    assert sim.pick(0, (100, 100), tolerance=0) == 2
    assert sim.pick(1, (100, 100), tolerance=0) == 4
    assert sim.pick(2, (100, 100), tolerance=1) is None
    assert sim.pick(0, (200, 200), tolerance=1) is None

    empty = Sim(app.device, num_worlds=1, creatures_per_world=0)
    assert empty.pick(0, (100, 100), tolerance=1) is None


def test_picking_respects_body_orientation_tolerance_and_closest_hit(app):
    import slangpy as spy

    sim = app.sim
    cursor = creature_cursor(sim)
    for index in range(sim.creatures_per_world):
        cursor[index]["alive"] = False
    for index, position in ((0, (100, 100)), (1, (100, 100.2))):
        cursor[index]["alive"] = True
        cursor[index]["position"] = spy.float2(*position)
        cursor[index]["velocity"] = spy.float2(0, 1)
        cursor[index]["heading"] = np.pi / 2
        cursor[index]["radii"] = spy.float2(0.5, 0.2)
    cursor.apply()

    # The long body axis points down the screen, matching its velocity.
    assert sim.pick(0, (100, 99.6), tolerance=0) == 0
    assert sim.pick(0, (100.25, 100), tolerance=0) is None
    # A five-pixel affordance at 50 px/m lets a near miss select the small body.
    assert sim.pick(0, (100.25, 100), tolerance=5 / 50) == 0
    assert sim.pick(0, (100.4, 100), tolerance=5 / 50) is None
    assert sim.pick(0, (100, 100.2), tolerance=0) == 1

    cursor[1]["position"] = spy.float2(100, 100)
    cursor.apply()
    assert sim.pick(0, (100, 100), tolerance=0) == 0  # Stable tie-breaking.


def render_world(app, *, update=False, **options):
    """Read one test image; optionally update and draw in the same GPU submission."""
    import slangpy as spy

    encoder = app.device.create_command_encoder()
    if update:
        app.sim.update(encoder)
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
        app.sim.render(render_pass, (0, 0, app.texture.width, app.texture.height), 0, **options)
    app.device.submit_command_buffer(encoder.finish())
    return app.texture.to_numpy().copy()


@pytest.mark.parametrize("click_offspring", [False, True])
def test_split_retires_selected_parent_and_requires_a_click_to_select_an_offspring(
    app, click_offspring
):
    app.sim = Sim(
        app.device,
        num_worlds=1,
        creatures_per_world=1,
        creature_capacity=2,
        food_capacity=0,
        initial_food=0,
        target_food=0,
        food_per_second=0,
    )
    app.selected_world = 0
    app.selected_creature = 0
    app.refresh_stats()
    parent_guid = app.selected_guid
    parent_position = app.creature_stats["position"]
    app.camera.scale = 128
    app.set_follow(True)
    assert app.following
    fixed_brains(app.sim, movement=0)
    cursor = creature_cursor(app.sim)
    cursor[0]["energy"] = cursor[0]["split_energy"].read() + 100
    cursor.apply()

    app.sim.update()
    siblings = [app.sim.inspect(0, index) for index in range(2)]
    assert all(child["alive"] and child["guid"] != parent_guid for child in siblings)
    assert all(child["parent_guid"] == parent_guid for child in siblings)
    if click_offspring:
        # Click the same slot before the next inspector poll detects retirement.
        rect = app.focused_rect()
        scale = app.camera.pixels_per_unit(rect)
        point = tuple(
            rect[axis]
            + rect[axis + 2] / 2
            + (siblings[0]["position"][axis] - parent_position[axis]) * scale
            for axis in range(2)
        )
        app.pick(point)
        assert app.selected_creature == 0 and app.selected_guid == siblings[0]["guid"]
        assert app.creature_stats["parent_guid"] == parent_guid
    else:
        app.refresh_stats()
        assert app.selected_creature is None and app.selected_guid == 0
        assert app.creature_stats is None
    assert not app.following
    assert app.camera.offset == pytest.approx(
        tuple(coordinate - WORLD_SIZE / 2 for coordinate in parent_position)
    )


def test_reused_selection_does_not_follow_highlight_or_pick_the_new_creature(app):
    import slangpy as spy

    sim = app.sim
    cursor = creature_cursor(sim)
    selected_guid = cursor[0]["guid"].read()
    for index in range(sim.creature_capacity):
        cursor[index]["alive"] = index == 0
    cursor[0]["position"] = spy.float2(100, 100)
    cursor[0]["guid"] = selected_guid + 10000
    cursor.apply()

    reused = render_world(
        app, selected=0, selected_guid=selected_guid, follow=True, wrap_view=True, scale=32
    )
    unselected = render_world(app, wrap_view=True, scale=32)
    np.testing.assert_array_equal(reused, unselected)
    assert (
        sim.pick(
            0,
            (WORLD_SIZE / 2, WORLD_SIZE / 2),
            0,
            selected=0,
            selected_guid=selected_guid,
            follow=True,
        )
        is None
    )
    assert sim.pick(0, (100, 100), 0) == 0

    # Between inspector polls, the camera and picking share the last known
    # position instead of jumping back to the world midpoint.
    fallback_position = (99, 100)
    fallback = render_world(
        app,
        selected=0,
        selected_guid=selected_guid,
        follow=True,
        follow_position=fallback_position,
        wrap_view=True,
        scale=32,
    )
    manual = render_world(app, offset=(-157, -156), wrap_view=True, scale=32)
    np.testing.assert_array_equal(fallback, manual)
    assert (
        sim.pick(
            0,
            (WORLD_SIZE / 2 + 1, WORLD_SIZE / 2),
            0,
            selected=0,
            selected_guid=selected_guid,
            follow=True,
            follow_position=fallback_position,
        )
        == 0
    )


def test_follow_and_pick_stay_on_the_creature_when_both_world_axes_wrap(app):
    import slangpy as spy

    sim = app.sim
    cursor = creature_cursor(sim)
    for index in range(sim.creatures_per_world):
        cursor[index]["alive"] = False
    edge = WORLD_SIZE - 0.05
    for index, position in ((0, (edge, edge)), (1, (0.95, 0.95))):
        cursor[index]["alive"] = True
        cursor[index]["position"] = spy.float2(*position)
        cursor[index]["velocity"] = spy.float2(6, 6)
        cursor[index]["heading"] = np.pi / 4
        cursor[index]["max_speed"] = float(np.hypot(6, 6))
        cursor[index]["radii"] = spy.float2(0.5, 0.315)
    cursor.apply()
    fixed_brains(sim)
    # Refresh the staged scene's observation before comparing it with a real tick.
    encoder = app.device.create_command_encoder()
    sim._update_grid(encoder)
    sim._update_vision(encoder)
    app.device.submit_command_buffer(encoder.finish())

    midpoint = WORLD_SIZE / 2
    assert sim.pick(0, (-0.05, -0.05), tolerance=0) == 0
    assert sim.pick(0, (midpoint, midpoint), 0, selected=0, follow=True) == 0
    assert sim.pick(0, (midpoint + 1, midpoint + 1), 0, selected=0, follow=True) == 1
    before = render_world(app, selected=0, follow=True, wrap_view=True, scale=128)
    after = render_world(app, update=True, selected=0, follow=True, wrap_view=True, scale=128)

    assert all(coordinate < 0.1 for coordinate in sim.inspect(0, 0)["position"])
    assert sim.pick(0, (midpoint, midpoint), 0, selected=0, follow=True) == 0
    assert sim.pick(0, (midpoint + 1, midpoint + 1), 0, selected=0, follow=True) == 1
    for image in (before, after):
        assert image[app.texture.height // 2, app.texture.width // 2, :3].max() > 150

    # Bodies and the selection outline remain at the same screen coordinates.
    # The dark periodic background can vary slightly as the camera moves.
    bodies_before = before[:, :, :3].max(axis=2) > 150
    bodies_after = after[:, :, :3].max(axis=2) > 150
    assert bodies_before.sum() > 100
    assert np.count_nonzero(bodies_before ^ bodies_after) <= 4

    # Stopping follow at the same camera position preserves the wrapped view.
    position = sim.inspect(0, 0)["position"]
    manual = render_world(
        app,
        selected=0,
        wrap_view=True,
        offset=tuple(coordinate - midpoint for coordinate in position),
        scale=128,
    )
    np.testing.assert_allclose(manual, after, rtol=0, atol=1)
