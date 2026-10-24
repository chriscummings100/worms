"""Exercise real D3D12 state and rendering; opt in with --run-gpu."""

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.gpu


@pytest.fixture
def app():
    from worms.app import App

    # Odd dimensions and a partial compute group catch out-of-range writes.
    application = App(headless=True, num_worlds=3, creatures_per_world=17, width=257, height=149)
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
    sim = app.sim
    counts = sim.worlds.to_numpy().view(np.int32)
    np.testing.assert_array_equal(counts, [17, 17, 17])

    cursor = creature_cursor(sim)
    ids = [cursor[i]["guid"].read() for i in range(51)]
    assert len(set(ids)) == 51
    assert all(cursor[i]["alive"].read() for i in range(51))
    seeded = positions(sim).reshape(3, 17, 2)
    assert np.isfinite(seeded).all()
    assert ((seeded >= 0) & (seeded < 256)).all()
    assert not np.array_equal(seeded[0], seeded[1])
    assert not np.array_equal(seeded[1], seeded[2])


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
    assert ((after >= 0) & (after < 256)).all()
    sim.reset()
    assert sim.tick == 0
    assert sim.creatures is buffer
    np.testing.assert_array_equal(buffer.to_numpy(), initial)


def test_movement_wraps_both_axes_without_changing_identity(app):
    import slangpy as spy

    sim = app.sim
    cursor = creature_cursor(sim)
    starts = np.array([(255.99, 0.01), (0.01, 255.99), (255.99, 255.99), (0.01, 0.01)])
    velocities = np.array([(6, -6), (-6, 6), (6, 6), (-6, -6)])
    ids = [cursor[i]["guid"].read() for i in range(4)]
    for i, (position, velocity) in enumerate(zip(starts, velocities, strict=True)):
        cursor[i]["position"] = spy.float2(*position)
        cursor[i]["velocity"] = spy.float2(*velocity)
    cursor.apply()

    sim.update()

    np.testing.assert_allclose(positions(sim)[:4], (starts + velocities / 60) % 256, atol=3e-5)
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
