"""Exercise the GPU contact grid and exact, single-owner food settlement."""

import numpy as np
import pytest
import slangpy as spy

from tests.gpu_helpers import fixed_brains
from worms.app import App
from worms.sim import FOOD_ENERGY, GRID_CELL_SIZE, GRID_SIZE, INITIAL_ENERGY, Sim

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def app():
    application = App(
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


def cursor_for(sim, field):
    kernel = sim._initialize_food if field == "food" else sim._initialize
    layout = getattr(kernel.reflection, f"g_{field}").type_layout.element_type_layout
    return spy.BufferCursor(layout, getattr(sim, field))


def write_creatures(sim, positions, *, alive=None, radii=(0.5, 0.315), energy=INITIAL_ENERGY):
    alive = [True] * len(positions) if alive is None else alive
    cursor = cursor_for(sim, "creatures")
    for index, (position, living) in enumerate(zip(positions, alive, strict=True)):
        cursor[index]["position"] = spy.float2(*position)
        cursor[index]["velocity"] = spy.float2(0, 0)
        cursor[index]["max_speed"] = 0.0
        cursor[index]["radii"] = spy.float2(*radii)
        cursor[index]["alive"] = living
        cursor[index]["energy"] = energy
    cursor.apply()
    worlds = cursor_for(sim, "worlds")
    for world in range(sim.num_worlds):
        start = world * sim.creature_capacity
        worlds[world]["num_creatures"] = sum(alive[start : start + sim.creature_capacity])
    worlds.apply()
    fixed_brains(sim)


def write_food(sim, positions, energies=None):
    energies = [FOOD_ENERGY] * len(positions) if energies is None else energies
    cursor = cursor_for(sim, "food")
    for index, (position, energy) in enumerate(zip(positions, energies, strict=True)):
        cursor[index]["position"] = spy.float2(*position)
        cursor[index]["energy"] = energy
    cursor.apply()
    worlds = cursor_for(sim, "worlds")
    for world in range(sim.num_worlds):
        start = world * sim.food_capacity
        worlds[world]["num_food"] = sum(
            energy > 0 for energy in energies[start : start + sim.food_capacity]
        )
    worlds.apply()


def creature_energies(sim):
    cursor = cursor_for(sim, "creatures")
    count = sim.num_worlds * sim.creature_capacity
    return np.array(
        [cursor[index]["energy"].read() for index in range(count)],
        dtype=np.uint64,
    ).reshape(sim.num_worlds, sim.creature_capacity)


def food_records(sim):
    dtype = np.dtype([("position", "<f4", (2,)), ("energy", "<u4")])
    return sim.food.to_numpy().view(dtype).reshape(sim.num_worlds, sim.food_capacity)


def grid_contents(sim):
    heads = sim.grid_heads.to_numpy().view(np.int32).reshape(sim.num_worlds, GRID_SIZE**2)
    links = sim.grid_next.to_numpy().view(np.int32).reshape(-1)
    result = []
    for world in range(sim.num_worlds):
        start = world * sim.creature_capacity
        cells = {}
        seen = set()
        for cell in np.flatnonzero(heads[world] >= 0):
            members = set()
            index = int(heads[world, cell])
            while index != -1:
                # Fail on cross-world links, duplicate insertions and cycles.
                assert start <= index < start + sim.creature_capacity
                assert index not in seen
                seen.add(index)
                members.add(index)
                index = int(links[index])
            cells[(int(cell) % GRID_SIZE, int(cell) // GRID_SIZE)] = members
        result.append(cells)
    return result


def test_serial_grid_keeps_worlds_separate_and_rebuilds_without_dead_or_stale_entries(app):
    sim = Sim(
        app.device,
        num_worlds=3,
        creatures_per_world=4,
        creature_capacity=4,
        food_capacity=0,
        initial_food=0,
        target_food=0,
        food_per_second=0,
    )
    positions = [(1, 1), (1.5, 1), (8, 8), (100, 100)] * 3
    alive = [True, True, False, True, True, False, False, True, False, False, False, False]
    write_creatures(sim, positions, alive=alive)
    sim.update()

    assert grid_contents(sim) == [
        {(0, 0): {0, 1}, (25, 25): {3}},
        {(0, 0): {4}, (25, 25): {7}},
        {},
    ]
    positions[0] = (5 * GRID_CELL_SIZE, 1)
    alive[1] = False
    write_creatures(sim, positions, alive=alive)
    sim.update()
    assert grid_contents(sim)[0] == {(5, 0): {0}, (25, 25): {3}}
    links = sim.grid_next.to_numpy().view(np.int32).reshape(-1)
    np.testing.assert_array_equal(links[np.logical_not(alive)], -1)


def test_contested_food_has_one_eater_and_is_not_transferred_again(app):
    sim = Sim(
        app.device,
        num_worlds=2,
        creatures_per_world=4,
        creature_capacity=4,
        food_capacity=1,
        initial_food=1,
        target_food=1,
        food_per_second=0,
    )
    write_creatures(sim, [(20, 20)] * 8)
    write_food(sim, [(20, 20)] * 2, [4321, 7654])
    sim.update()

    energies = creature_energies(sim)
    for world, amount in enumerate((4321, 7654)):
        gain = energies[world] - INITIAL_ENERGY
        assert np.count_nonzero(gain) == 1
        assert int(gain.sum()) == amount
    np.testing.assert_array_equal(food_records(sim)["energy"], 0)
    worlds = cursor_for(sim, "worlds")
    assert [worlds[i]["num_food"].read() for i in range(2)] == [0, 0]
    sim.update()
    np.testing.assert_array_equal(creature_energies(sim), energies - 1)


def test_parallel_food_credits_conserve_energy_and_replay_with_large_creature_reserves(app):
    sim = Sim(
        app.device,
        num_worlds=2,
        creatures_per_world=13,
        creature_capacity=13,
        seed=37,
        food_capacity=193,
        initial_food=193,
        target_food=193,
        food_per_second=0,
    )
    starting_energy = 2**32 + 17
    food_energy = [FOOD_ENERGY + i for i in range(386)]

    def collect():
        write_creatures(sim, [(32, 32)] * 26, energy=starting_energy)
        write_food(sim, [(32, 32)] * 386, food_energy)
        sim.update()
        return creature_energies(sim), sim.food.to_numpy().copy()

    first, food = collect()
    for world in range(2):
        expected = 13 * starting_energy + sum(food_energy[world * 193 : (world + 1) * 193])
        assert sum(map(int, first[world])) == expected
        assert np.count_nonzero(first[world] > starting_energy) > 1
    np.testing.assert_array_equal(food_records(sim)["energy"], 0)
    assert sim.inspect(1, 12)["energy"] == int(first[1, 12])

    sim.reset()
    replay, replay_food = collect()
    np.testing.assert_array_equal(replay, first)
    np.testing.assert_array_equal(replay_food, food)


@pytest.mark.parametrize(
    "creature,food,consumed",
    [
        ((3.8, 16), (4.2, 16), True),  # Adjacent cells.
        ((0.1, 24), (511.8, 24), True),  # Wrapping X boundary.
        ((0.1, 0.1), (511.9, 511.9), True),  # Both wrapping boundaries.
        ((40, 40), (40.67, 40), True),  # Radius 0.5 + food radius 0.18.
        ((40, 40), (40.69, 40), False),
    ],
)
def test_contact_uses_wrapped_circle_distance_after_grid_lookup(app, creature, food, consumed):
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
    write_creatures(sim, [creature])
    write_food(sim, [food])
    sim.update()
    assert int(creature_energies(sim)[0, 0]) == INITIAL_ENERGY + consumed * FOOD_ENERGY
    assert int(food_records(sim)["energy"][0, 0]) == (0 if consumed else FOOD_ENERGY)


def test_food_queries_expand_for_bodies_larger_than_a_grid_cell(app):
    sim = Sim(
        app.device,
        num_worlds=1,
        creatures_per_world=1,
        creature_capacity=1,
        food_capacity=2,
        initial_food=2,
        target_food=2,
        food_per_second=0,
    )
    write_creatures(sim, [(20, 20)], radii=(0.5, 10.0))
    write_food(sim, [(30, 20), (30.3, 20)])
    sim.update()
    np.testing.assert_array_equal(food_records(sim)["energy"], [[0, FOOD_ENERGY]])
    assert int(creature_energies(sim)[0, 0]) == INITIAL_ENERGY + FOOD_ENERGY


def test_grid_and_collection_use_positions_after_movement(app):
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
    write_creatures(sim, [(3.5, 10)])
    creatures = cursor_for(sim, "creatures")
    creatures[0]["velocity"] = spy.float2(60, 0)
    creatures[0]["heading"] = 0.0
    creatures[0]["max_speed"] = 60.0
    creatures.apply()
    write_food(sim, [(5, 10)])
    sim.update()
    assert grid_contents(sim) == [{(1, 2): {0}}]
    # The 60 m/s motion costs 62 energy quanta during this tick.
    assert int(creature_energies(sim)[0, 0]) == INITIAL_ENERGY + FOOD_ENERGY - 62


def test_growth_reuses_consumed_holes_without_moving_survivors_or_replaying_positions(app):
    sim = Sim(
        app.device,
        num_worlds=1,
        creatures_per_world=1,
        creature_capacity=1,
        food_capacity=3,
        initial_food=3,
        target_food=3,
        food_per_second=60,
    )
    write_creatures(sim, [(100, 100)])
    write_food(sim, [(100, 100), (200, 200), (300, 300)])
    before = food_records(sim).copy()
    sim.update()
    first = food_records(sim).copy()
    np.testing.assert_array_equal(first[:, 1:], before[:, 1:])
    assert not np.array_equal(first["position"][0, 0], before["position"][0, 0])
    assert int(creature_energies(sim)[0, 0]) == INITIAL_ENERGY + FOOD_ENERGY

    creatures = cursor_for(sim, "creatures")
    creatures[0]["position"] = spy.float2(300, 300)
    creatures.apply()
    sim.update()
    second = food_records(sim)
    np.testing.assert_array_equal(second[:, :2], first[:, :2])
    assert not np.array_equal(second["position"][0, 2], first["position"][0, 2])
    assert not np.array_equal(second["position"][0, 2], first["position"][0, 0])
    np.testing.assert_array_equal(second["energy"], FOOD_ENERGY)
    worlds = cursor_for(sim, "worlds")
    assert worlds[0]["num_food"].read() == 3
    assert worlds[0]["food_spawn_serial"].read() == 5
    assert int(creature_energies(sim)[0, 0]) == INITIAL_ENERGY + 2 * FOOD_ENERGY - 1


def test_collection_respects_app_pause_single_step_and_reset(app):
    app.sim = Sim(
        app.device,
        num_worlds=1,
        creatures_per_world=1,
        creature_capacity=1,
        food_capacity=1,
        initial_food=1,
        target_food=1,
        food_per_second=0,
    )
    initial_creatures = app.sim.creatures.to_numpy().copy()
    initial_food = app.sim.food.to_numpy().copy()
    write_creatures(app.sim, [(100, 100)])
    write_food(app.sim, [(100, 100)])
    app.paused = True
    app.frame()
    assert app.sim.tick == 0
    assert int(creature_energies(app.sim)[0, 0]) == INITIAL_ENERGY

    app.step()
    app.frame()
    assert app.paused and app.sim.tick == 1
    assert int(creature_energies(app.sim)[0, 0]) == INITIAL_ENERGY + FOOD_ENERGY
    np.testing.assert_array_equal(food_records(app.sim)["energy"], 0)
    app.frame()
    assert app.sim.tick == 1
    assert int(creature_energies(app.sim)[0, 0]) == INITIAL_ENERGY + FOOD_ENERGY

    app.reset()
    app.frame()
    assert app.sim.tick == 0
    np.testing.assert_array_equal(app.sim.creatures.to_numpy(), initial_creatures)
    np.testing.assert_array_equal(app.sim.food.to_numpy(), initial_food)
