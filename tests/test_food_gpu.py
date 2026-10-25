"""Check seeded food placement, growth and drawing on the real D3D12 device."""

import numpy as np
import pytest
import slangpy as spy

from tests.gpu_helpers import fixed_brains
from worms.app import App
from worms.sim import (
    ENERGY_PER_UNIT,
    FOOD_ENERGY,
    IDLE_ENERGY_COST,
    INITIAL_ENERGY,
    TIME_STEP,
    WORLD_SIZE,
    Sim,
)

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def app():
    application = App(
        debug_gpu=True,
        headless=True,
        benchmark=False,
        snapshot_dir=None,
        num_worlds=1,
        creatures_per_world=0,
        width=257,
        height=149,
    )
    yield application
    application.close()


def food_cursor(sim):
    layout = sim._initialize_food.reflection.g_food.type_layout.element_type_layout
    return spy.BufferCursor(layout, sim.food)


def food_records(sim):
    """Decode mixed float positions and integer energy without reinterpreting energy."""
    dtype = np.dtype([("position", "<f4", (2,)), ("energy", "<u4")])
    records = sim.food.to_numpy().view(dtype).reshape(-1)
    return records[: sim.num_worlds * sim.food_capacity].reshape(sim.num_worlds, sim.food_capacity)


def food_counts(sim):
    layout = sim._initialize.reflection.g_worlds.type_layout.element_type_layout
    cursor = spy.BufferCursor(layout, sim.worlds)
    return np.array([cursor[world]["num_food"].read() for world in range(sim.num_worlds)])


def advance(sim, ticks):
    encoder = sim.device.create_command_encoder()
    for _ in range(ticks):
        sim.update(encoder)
    sim.device.submit_command_buffer(encoder.finish())


def test_food_has_reserved_world_blocks_and_independent_seeded_streams(app):
    settings = dict(food_capacity=137, initial_food=67, target_food=113, seed=41)
    sim = Sim(app.device, num_worlds=3, creatures_per_world=0, **settings)
    assert sim.food_energy == 5
    assert FOOD_ENERGY == 5 * ENERGY_PER_UNIT
    records = food_records(sim)
    np.testing.assert_array_equal(food_counts(sim), [67, 67, 67])
    assert np.isfinite(records["position"]).all()
    positions = records["position"][:, :67]
    assert ((positions >= 0) & (positions < WORLD_SIZE)).all()
    np.testing.assert_array_equal(records["energy"][:, :67], FOOD_ENERGY)
    np.testing.assert_array_equal(records["position"][:, 67:], 0)
    np.testing.assert_array_equal(records["energy"][:, 67:], 0)
    assert not np.array_equal(records[0], records[1])
    assert not np.array_equal(records[1], records[2])

    # Existing worlds retain their food when another world or creatures are added.
    larger = Sim(app.device, num_worlds=4, creatures_per_world=11, **settings)
    np.testing.assert_array_equal(food_records(larger)[:3], records)
    changed_seed = Sim(app.device, num_worlds=1, creatures_per_world=0, **(settings | {"seed": 42}))
    assert not np.array_equal(food_records(changed_seed)[0, :67], records[0, :67])


def test_food_reset_replays_all_bytes_and_growth_replays_the_same_positions(app):
    sim = Sim(
        app.device,
        num_worlds=2,
        creatures_per_world=0,
        food_capacity=73,
        initial_food=17,
        target_food=67,
        food_per_second=60,
    )
    buffer = sim.food
    initial = buffer.to_numpy().copy()
    advance(sim, 9)
    grown = buffer.to_numpy().copy()
    np.testing.assert_array_equal(food_counts(sim), [26, 26])
    assert not np.array_equal(grown, initial)

    sim.reset()
    assert sim.food is buffer
    assert sim.tick == 0
    np.testing.assert_array_equal(food_counts(sim), [17, 17])
    np.testing.assert_array_equal(buffer.to_numpy(), initial)
    advance(sim, 9)
    np.testing.assert_array_equal(buffer.to_numpy(), grown)


def test_food_energy_changes_revalue_only_live_slots_and_preserve_the_allocations(app):
    sim = Sim(
        app.device,
        num_worlds=2,
        creatures_per_world=0,
        food_capacity=7,
        initial_food=5,
        target_food=5,
        food_per_second=0,
    )
    cursor = food_cursor(sim)
    cursor[1]["energy"] = 0
    cursor[9]["energy"] = 0
    cursor.apply()
    layout = sim._initialize.reflection.g_worlds.type_layout.element_type_layout
    worlds = spy.BufferCursor(layout, sim.worlds)
    for world in range(2):
        worlds[world]["num_food"] = 4
    worlds.apply()
    original = food_records(sim).copy()
    world_bytes = sim.worlds.to_numpy().copy()
    buffer = sim.food

    sim.food_energy = 3.2503
    amount = round(3.2503 * ENERGY_PER_UNIT)
    assert sim.food_energy == amount / ENERGY_PER_UNIT
    np.testing.assert_array_equal(food_records(sim), original)
    encoder = app.device.create_command_encoder()
    sim.apply_settings(encoder)
    app.device.submit_command_buffer(encoder.finish())

    changed = food_records(sim)
    assert sim.food is buffer and sim.tick == 0
    np.testing.assert_array_equal(changed["position"], original["position"])
    np.testing.assert_array_equal(changed["energy"], np.where(original["energy"], amount, 0))
    np.testing.assert_array_equal(sim.worlds.to_numpy(), world_bytes)


def test_food_energy_setting_is_used_for_growth_and_retained_by_reset(app):
    sim = Sim(
        app.device,
        num_worlds=2,
        creatures_per_world=0,
        food_capacity=5,
        initial_food=1,
        target_food=5,
        food_per_second=60,
        food_energy=3.25,
    )
    original_positions = food_records(sim)["position"][:, :1].copy()
    np.testing.assert_array_equal(food_records(sim)["energy"][:, 0], 3.25 * ENERGY_PER_UNIT)
    sim.food_energy = 4.5
    advance(sim, 2)
    grown = food_records(sim).copy()
    np.testing.assert_array_equal(food_counts(sim), [3, 3])
    np.testing.assert_array_equal(grown["energy"][:, :3], 4.5 * ENERGY_PER_UNIT)
    np.testing.assert_array_equal(grown["energy"][:, 3:], 0)

    sim.reset()
    assert sim.food_energy == 4.5 and sim.tick == 0
    np.testing.assert_array_equal(food_counts(sim), [1, 1])
    np.testing.assert_array_equal(food_records(sim)["position"][:, :1], original_positions)
    np.testing.assert_array_equal(food_records(sim)["energy"][:, 0], 4.5 * ENERGY_PER_UNIT)
    advance(sim, 2)
    np.testing.assert_array_equal(food_records(sim), grown)


def test_changed_food_energy_is_transferred_on_contact_after_lifecycle_cost(app):
    sim = Sim(
        app.device,
        num_worlds=1,
        creatures_per_world=1,
        creature_capacity=1,
        food_capacity=1,
        initial_food=1,
        target_food=1,
        food_per_second=0,
    )
    layout = sim._initialize.reflection.g_creatures.type_layout.element_type_layout
    creatures = spy.BufferCursor(layout, sim.creatures)
    creatures[0]["position"] = spy.float2(100, 100)
    creatures[0]["max_speed"] = 0.0
    creatures[0]["radii"] = spy.float2(1.0, 0.63)
    creatures[0]["energy"] = INITIAL_ENERGY
    creatures[0]["energy_remainder"] = 0.0
    creatures.apply()
    food = food_cursor(sim)
    food[0]["position"] = spy.float2(100, 100)
    food.apply()
    fixed_brains(sim, movement=0)

    sim.food_energy = 3.125
    sim.update()
    cost = int(2 * IDLE_ENERGY_COST * TIME_STEP * ENERGY_PER_UNIT)
    assert cost > 0
    assert sim.inspect(0, 0)["energy"] == INITIAL_ENERGY + 3200 - cost
    assert food_records(sim)["energy"][0, 0] == 0
    np.testing.assert_array_equal(food_counts(sim), [0])


def test_food_energy_change_applies_while_paused_without_advancing_creatures(app):
    previous_sim, previous_pause = app.sim, app.paused
    try:
        app.sim = Sim(
            app.device,
            num_worlds=1,
            creatures_per_world=1,
            creature_capacity=1,
            food_capacity=3,
            initial_food=1,
            target_food=3,
            food_per_second=60,
        )
        app.paused = True
        creatures = app.sim.creatures.to_numpy().copy()
        history = app.sim.vision_history.to_numpy().copy()
        app.sim.food_energy = 5
        app.frame()
        assert app.sim.tick == 0
        np.testing.assert_array_equal(
            food_records(app.sim)["energy"], [[5 * ENERGY_PER_UNIT, 0, 0]]
        )
        np.testing.assert_array_equal(food_counts(app.sim), [1])
        np.testing.assert_array_equal(app.sim.creatures.to_numpy(), creatures)
        np.testing.assert_array_equal(app.sim.vision_history.to_numpy(), history)
    finally:
        app.sim, app.paused = previous_sim, previous_pause


def test_live_food_growth_rate_can_stop_resume_and_survive_reset(app):
    sim = Sim(
        app.device,
        num_worlds=2,
        creatures_per_world=0,
        food_capacity=32,
        initial_food=3,
        target_food=32,
        food_per_second=0,
    )
    initial = food_records(sim).copy()
    buffer = sim.food
    for rate, ticks, expected in ((0, 3, 3), (120, 4, 11), (0, 3, 11), (60, 2, 13)):
        before = food_records(sim).copy()
        sim.food_per_second = rate
        advance(sim, ticks)
        np.testing.assert_array_equal(food_counts(sim), [expected, expected])
        current = food_records(sim)
        live = before["energy"] > 0
        np.testing.assert_array_equal(current[live], before[live])
        np.testing.assert_array_equal(np.count_nonzero(current["energy"], axis=1), [expected] * 2)
        if rate == 0:
            np.testing.assert_array_equal(current, before)

    sim.reset()
    assert sim.food is buffer and sim.food_per_second == 60 and sim.tick == 0
    np.testing.assert_array_equal(food_records(sim), initial)
    advance(sim, 2)
    np.testing.assert_array_equal(food_counts(sim), [5, 5])


@pytest.mark.parametrize("capacity,target,rate", [(64, 23, 7.5), (32, 32, 7.5), (0, 0, 0)])
def test_growth_carries_fractional_rates_and_stops_at_target_and_capacity(
    app, capacity, target, rate
):
    initial = min(3, target)
    sim = Sim(
        app.device,
        num_worlds=2,
        creatures_per_world=0,
        food_capacity=capacity,
        initial_food=initial,
        target_food=target,
        food_per_second=rate,
    )
    initial_records = food_records(sim)[:, :initial].copy() if capacity else None
    for tick, expected_growth in ((1, 0), (7, 0), (8, 1), (60, 7), (120, 15), (240, 30), (360, 45)):
        advance(sim, tick - sim.tick)
        expected = min(target, initial + (expected_growth if rate else 0))
        np.testing.assert_array_equal(food_counts(sim), [expected, expected])
        if capacity:
            records = food_records(sim)
            np.testing.assert_array_equal(records[:, :initial], initial_records)
            np.testing.assert_array_equal(records["energy"][:, :expected], FOOD_ENERGY)
            np.testing.assert_array_equal(records["energy"][:, expected:], 0)
            np.testing.assert_array_equal(records["position"][:, expected:], 0)
            positions = records["position"][:, :expected]
            assert ((positions >= 0) & (positions < WORLD_SIZE)).all()

    assert sim.tick * TIME_STEP == 6


def test_food_obeys_app_pause_step_and_reset(app):
    app.sim = Sim(
        app.device,
        num_worlds=1,
        creatures_per_world=0,
        food_capacity=17,
        initial_food=3,
        target_food=13,
        food_per_second=60,
    )
    app.selected_world = 0
    app.paused = True
    initial = app.sim.food.to_numpy().copy()
    app.frame()
    assert app.sim.tick == 0
    np.testing.assert_array_equal(app.sim.food.to_numpy(), initial)

    app.step()
    app.frame()
    assert app.paused and app.sim.tick == 1
    np.testing.assert_array_equal(food_counts(app.sim), [4])
    stepped = app.sim.food.to_numpy().copy()
    app.frame()
    np.testing.assert_array_equal(app.sim.food.to_numpy(), stepped)

    app.paused = False
    app.reset()
    app.frame()
    assert app.sim.tick == 0
    np.testing.assert_array_equal(app.sim.food.to_numpy(), initial)
    app.frame()
    np.testing.assert_array_equal(app.sim.food.to_numpy(), stepped)


def test_food_forms_spatial_clusters_instead_of_uniform_random_scatter(app):
    sim = Sim(
        app.device,
        num_worlds=3,
        creatures_per_world=0,
        seed=19,
        food_capacity=4096,
        initial_food=4096,
        target_food=4096,
        food_per_second=0,
    )
    positions = food_records(sim)["position"]
    uniform = np.random.default_rng(1094).uniform(0, WORLD_SIZE, positions.shape)

    def occupancy_variance(samples):
        return np.mean(
            [
                np.histogram2d(world[:, 0], world[:, 1], bins=16, range=((0, WORLD_SIZE),) * 2)[
                    0
                ].var()
                for world in samples
            ]
        )

    # The same sample count, domain and bins provide a Poisson-noise baseline.
    # Require a substantial spatial density signal, without pinning exact pixels.
    assert occupancy_variance(positions) > 3 * occupancy_variance(uniform)


def render_food(app, sim, **options):
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
        sim.render(render_pass, (0, 0, app.texture.width, app.texture.height), 0, **options)
    app.device.submit_command_buffer(encoder.finish())
    return app.texture.to_numpy().copy()


def test_food_rendering_scales_and_wraps_with_the_camera(app):
    sim = Sim(
        app.device,
        num_worlds=1,
        creatures_per_world=0,
        food_capacity=2,
        initial_food=1,
        target_food=1,
        food_per_second=0,
    )
    cursor = food_cursor(sim)
    # A surviving high slot stays visible even though the live count is one.
    cursor[0]["energy"] = 0
    cursor[1]["position"] = spy.float2(0.125, 0.125)
    cursor[1]["energy"] = 0
    cursor.apply()
    offset = (0.125 - WORLD_SIZE / 2,) * 2
    empty = render_food(app, sim, scale=128, offset=offset, wrap_view=True)
    cursor[1]["energy"] = FOOD_ENERGY
    cursor.apply()
    close = render_food(app, sim, scale=128, offset=offset, wrap_view=True)
    changed = np.any(close[:, :, :3] != empty[:, :, :3], axis=2)
    assert changed.sum() > 20
    assert changed[app.texture.height // 2, app.texture.width // 2]
    assert close[app.texture.height // 2, app.texture.width // 2, 1] > 200

    wrapped = render_food(
        app,
        sim,
        scale=128,
        offset=tuple(value + WORLD_SIZE for value in offset),
        wrap_view=True,
    )
    np.testing.assert_allclose(wrapped, close, rtol=0, atol=1)
    far = render_food(app, sim, scale=32, offset=offset, wrap_view=True)
    assert (close[:, :, 1] > 160).sum() > 4 * (far[:, :, 1] > 160).sum()
