"""Check GPU energy accounting, inherited networks and reusable creature slots."""

import math

import numpy as np
import pytest
import slangpy as spy

from tests.gpu_helpers import fixed_brains
from worms.app import App
from worms.sim import (
    BRAIN_ACTIVATIONS,
    BRAIN_PARAMETERS,
    ENERGY_PER_UNIT,
    IDLE_ENERGY_COST,
    INITIAL_ENERGY,
    SPEED_ENERGY_COST,
    TIME_STEP,
    WORLD_SIZE,
    Sim,
)

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def app():
    application = App(headless=True, num_worlds=1, creatures_per_world=0, width=257, height=149)
    yield application
    application.close()


def make_sim(app, *, creatures=1, capacity=None, worlds=1, food=0, seed=37):
    return Sim(
        app.device,
        num_worlds=worlds,
        creatures_per_world=creatures,
        creature_capacity=capacity,
        food_capacity=food,
        initial_food=food,
        target_food=food,
        food_per_second=0,
        seed=seed,
    )


def cursor_for(sim, field="creatures"):
    kernel = sim._initialize_food if field == "food" else sim._initialize
    layout = getattr(kernel.reflection, f"g_{field}").type_layout.element_type_layout
    return spy.BufferCursor(layout, getattr(sim, field))


def stage(sim, index, *, size=1.0, energy=INITIAL_ENERGY, speed=0.0, position=(100, 100)):
    cursor = cursor_for(sim)
    cursor[index]["position"] = spy.float2(*position)
    cursor[index]["radii"] = spy.float2(size / 2, size * 0.315)
    cursor[index]["max_speed"] = speed
    cursor[index]["heading"] = 0.0
    cursor[index]["velocity"] = spy.float2(0, 0)
    cursor[index]["energy"] = energy
    cursor[index]["split_energy"] = round(2 * INITIAL_ENERGY * size)
    cursor[index]["energy_remainder"] = 0.0
    cursor.apply()


def advance(sim, ticks):
    encoder = sim.device.create_command_encoder()
    for _ in range(ticks):
        sim.update(encoder)
    sim.device.submit_command_buffer(encoder.finish())


def parameters(sim):
    return sim.brain_weights.to_numpy().view(np.float32).reshape(-1, BRAIN_PARAMETERS)


def assert_small_mutation(child, parent):
    delta = child - parent
    # Fixed seeds should change a few parameters, keeping almost all inherited exactly.
    assert 0 < np.count_nonzero(delta) < BRAIN_PARAMETERS * 0.05
    assert np.isfinite(child).all()
    assert np.max(np.abs(delta)) < 0.12


def population(sim, world=0):
    return [sim.inspect(world, index) for index in range(sim.creature_capacity)]


def test_initial_energy_and_split_threshold_scale_with_body_length_and_reserve_slots(app):
    sim = make_sim(app, creatures=7, worlds=2)
    assert sim.creatures_per_world == 7
    assert sim.creature_capacity == 28
    guids = []
    for world in range(2):
        records = population(sim, world)
        assert [creature["alive"] for creature in records] == [True] * 7 + [False] * 21
        for creature in records[:7]:
            expected = round(INITIAL_ENERGY * 2 * creature["radii"][0])
            assert creature["energy"] == expected
            assert creature["split_energy"] == 2 * expected
            assert creature["parent_guid"] == creature["generation"] == creature["birth_tick"] == 0
            assert creature["age_start_tick"] == 0
            guids.append(creature["guid"])
        assert all(creature["energy"] == 0 for creature in records[7:])
        stats = sim.inspect_world(world)
        assert (stats["num_creatures"], stats["births"], stats["deaths"]) == (7, 0, 0)
    assert len(set(guids)) == 14


def test_cost_scales_linearly_with_size_and_quadratically_with_actual_speed(app):
    sim = make_sim(app, creatures=6, capacity=6)
    # The last two have identical actual speed despite different maximum speed.
    cases = [(1, 0, 0), (2, 0, 0), (1, 10, 1), (2, 10, 1), (1, 20, 0.5), (1, 20, 0)]
    for index, (size, speed, _) in enumerate(cases):
        stage(sim, index, size=size, energy=INITIAL_ENERGY * 10, speed=speed)
    fixed_brains(sim)
    weights = parameters(sim).copy()
    for index, (_, _, movement) in enumerate(cases):
        weights[index, 1073] = movement - 0.5
    sim.brain_weights.copy_from_numpy(weights)
    encoder = sim.device.create_command_encoder()
    sim._update_brains(encoder)
    sim.device.submit_command_buffer(encoder.finish())

    advance(sim, 120)
    actual_costs = []
    for index, (size, speed, movement) in enumerate(cases):
        creature = sim.inspect(0, index)
        debit = INITIAL_ENERGY * 10 - creature["energy"]
        total_cost = (debit + creature["energy_remainder"]) / ENERGY_PER_UNIT
        expected = (
            120
            * TIME_STEP
            * size
            * (IDLE_ENERGY_COST + SPEED_ENERGY_COST * (speed * movement) ** 2)
        )
        assert total_cost == pytest.approx(expected, abs=2e-6)
        assert 0 <= creature["energy_remainder"] < 1
        assert math.hypot(*creature["velocity"]) == pytest.approx(speed * movement)
        actual_costs.append(total_cost)
    assert actual_costs[1] == pytest.approx(2 * actual_costs[0])
    assert actual_costs[3] == pytest.approx(2 * actual_costs[2])
    assert actual_costs[2] == pytest.approx(actual_costs[4])
    assert actual_costs[0] == pytest.approx(actual_costs[5])
    assert actual_costs[2] > actual_costs[0] > 0


def test_fractional_idle_cost_accumulates_and_death_is_recorded_once(app):
    sim = make_sim(app, creatures=1, capacity=1)
    stage(sim, 0, energy=1)
    fixed_brains(sim, movement=0)
    sim.update()
    assert sim.inspect(0, 0)["energy"] == 1  # Less than one quantum this tick.
    sim.update()
    dead = sim.inspect(0, 0)
    assert not dead["alive"]
    assert dead["energy"] == dead["energy_remainder"] == 0
    assert dead["velocity"] == (0, 0)
    assert sim.inspect_vision(0, 0) is None
    np.testing.assert_array_equal(sim.brain_activations.to_numpy().view(np.float32), 0)
    np.testing.assert_array_equal(sim.grid_heads.to_numpy().view(np.int32), -1)
    advance(sim, 4)
    stats = sim.inspect_world(0)
    assert (stats["num_creatures"], stats["births"], stats["deaths"]) == (0, 0, 1)


def test_food_is_credited_before_depletion_can_kill_a_creature(app):
    sim = make_sim(app, creatures=1, capacity=1, food=1)
    stage(sim, 0, size=2, energy=0)
    fixed_brains(sim, movement=0)
    food = cursor_for(sim, "food")
    food[0]["position"] = spy.float2(100, 100)
    food[0]["energy"] = 2
    food.apply()
    sim.update()
    creature = sim.inspect(0, 0)
    assert creature["alive"] and creature["energy"] == 1
    assert sim.inspect_world(0)["deaths"] == 0
    assert cursor_for(sim, "food")[0]["energy"].read() == 0
    sim.update()
    assert not sim.inspect(0, 0)["alive"]
    assert sim.inspect_world(0)["deaths"] == 1


def test_food_crossing_split_threshold_shares_energy_without_creating_any(app):
    sim = make_sim(app, food=1, capacity=4)
    stage(sim, 0, energy=2 * INITIAL_ENERGY - ENERGY_PER_UNIT)
    fixed_brains(sim, movement=0)
    food = cursor_for(sim, "food")
    food[0]["position"] = spy.float2(100, 100)
    food[0]["energy"] = ENERGY_PER_UNIT
    food.apply()
    sim.update()
    records = population(sim)
    assert [creature["alive"] for creature in records] == [True, True, False, False]
    assert records[0]["energy"] == records[1]["energy"] == INITIAL_ENERGY
    assert sum(creature["energy"] for creature in records) == 2 * INITIAL_ENERGY
    assert sim.inspect_world(0)["births"] == 1


def test_split_mutates_inherited_network_and_copies_body_with_fresh_identity_and_history(app):
    sim = make_sim(app, capacity=4, worlds=2)
    for world in range(2):
        stage(sim, world * sim.creature_capacity, energy=1_000_001, position=(511.8, 511.8))
    original = parameters(sim).copy()
    parents = [sim.inspect(world, 0) for world in range(2)]
    sim.update()
    guids = set()
    for world in range(2):
        parent, child, unused1, unused2 = population(sim, world)
        assert parent["guid"] == parents[world]["guid"]
        assert parent["energy"] == 500_001 and child["energy"] == 500_000
        assert child["guid"] > sim.num_worlds * sim.creature_capacity
        assert child["parent_guid"] == parent["guid"]
        assert child["generation"] == 1 and child["birth_tick"] == 1
        for field in ("radii", "color", "max_speed", "split_energy"):
            assert child[field] == parent[field]
        assert child["energy_remainder"] == 0
        assert child["velocity"] == (0, 0)
        delta = np.array(child["position"]) - parent["position"]
        delta -= WORLD_SIZE * np.floor(delta / WORLD_SIZE + 0.5)
        assert np.linalg.norm(delta) == pytest.approx(1.25, abs=1e-4)
        assert all(0 <= coordinate < WORLD_SIZE for coordinate in child["position"])
        assert not unused1["alive"] and not unused2["alive"]  # No cascade in its birth tick.
        offset = world * sim.creature_capacity
        np.testing.assert_array_equal(parameters(sim)[offset], original[offset])
        assert_small_mutation(parameters(sim)[offset + 1], original[offset])
        np.testing.assert_array_equal(
            parameters(sim)[offset + 2 : offset + 4], original[offset + 2 : offset + 4]
        )
        assert sim.inspect_vision(world, 1)["tick"] == 1
        assert sim.inspect_vision(world, 1, age=1) is None
        assert sim.inspect_vision(world, 0, age=1)["tick"] == 0
        stats = sim.inspect_world(world)
        assert (stats["num_creatures"], stats["births"], stats["deaths"]) == (2, 1, 0)
        guids.update((parent["guid"], child["guid"]))
    assert len(guids) == 4


def test_split_turns_both_creatures_outwards_and_separates_them_across_world_seams(app):
    cases = [
        (0.0, 0.5, (0.03, 0.04)),
        (math.pi / 2, 1.0, (0.03, 0.04)),
        (-math.pi / 2, 2.0, (511.97, 511.96)),
        (math.pi - 0.01, 1.0, (0.03, 0.04)),
        (-math.pi + 0.01, 1.0, (511.97, 511.96)),
        (0.3, 1.5, (100.0, 100.0)),
    ]
    speed = 6.0
    sim = make_sim(app, capacity=2, worlds=len(cases))
    for world, (heading, size, position) in enumerate(cases):
        index = world * sim.creature_capacity
        stage(
            sim,
            index,
            size=size,
            energy=round(2 * INITIAL_ENERGY * size) + 100,
            speed=speed,
            position=position,
        )
        cursor = cursor_for(sim)
        cursor[index]["heading"] = heading
        cursor.apply()
    fixed_brains(sim, steering=0, movement=1)
    sim.update()

    for world, (heading, size, position) in enumerate(cases):
        parent, child = population(sim, world)
        forward = np.array([math.cos(heading), math.sin(heading)])
        side = np.array([-math.sin(heading), math.cos(heading)])
        # Splitting happens after this tick's movement, around the parent's new position.
        center = np.asarray(position) + forward * speed * TIME_STEP
        for creature, sign in ((parent, -1), (child, 1)):
            expected_heading = (heading + sign * math.pi / 2 + math.pi) % math.tau - math.pi
            assert -np.float32(math.pi) <= creature["heading"] < np.float32(math.pi)
            assert creature["heading"] == pytest.approx(expected_heading, abs=1e-6)
            expected_position = (center + sign * side * 0.625 * size) % WORLD_SIZE
            np.testing.assert_allclose(creature["position"], expected_position, atol=1e-4, rtol=0)
            assert all(0 <= coordinate < WORLD_SIZE for coordinate in creature["position"])
        np.testing.assert_allclose(parent["velocity"], -side * speed, atol=1e-5, rtol=0)
        assert child["velocity"] == (0, 0)
        separation = np.asarray(child["position"]) - parent["position"]
        separation -= WORLD_SIZE * np.floor(separation / WORLD_SIZE + 0.5)
        np.testing.assert_allclose(separation, side * 1.25 * size, atol=1e-4, rtol=0)
        assert sim.inspect_world(world)["births"] == 1

    # Restore straight-ahead policies to isolate geometry from the newborn's mutation.
    fixed_brains(sim, steering=0, movement=1)
    sim.update()
    for world, (heading, size, _) in enumerate(cases):
        parent, child = population(sim, world)
        side = np.array([-math.sin(heading), math.cos(heading)])
        separation = np.asarray(child["position"]) - parent["position"]
        separation -= WORLD_SIZE * np.floor(separation / WORLD_SIZE + 0.5)
        expected = side * (1.25 * size + 2 * speed * TIME_STEP)
        np.testing.assert_allclose(separation, expected, atol=1e-4, rtol=0)
        np.testing.assert_allclose(parent["velocity"], -side * speed, atol=1e-5, rtol=0)
        np.testing.assert_allclose(child["velocity"], side * speed, atol=1e-5, rtol=0)


def test_full_capacity_preserves_energy_then_reuses_a_slot_that_dies_that_tick(app):
    sim = make_sim(app, creatures=2, capacity=2)
    for index in range(2):
        stage(sim, index, energy=2 * INITIAL_ENERGY + 1)
    fixed_brains(sim, movement=0)
    before = population(sim)
    original_weights = parameters(sim).copy()
    sim.tick = 100
    sim.update()
    np.testing.assert_array_equal(parameters(sim), original_weights)
    assert sim.inspect_world(0)["births"] == 0
    assert [creature["energy"] for creature in population(sim)] == [2 * INITIAL_ENERGY + 1] * 2
    previous = population(sim)
    for original, unchanged in zip(before, previous, strict=True):
        for field in ("position", "heading", "velocity", "birth_tick", "age_start_tick"):
            assert unchanged[field] == original[field]
        assert sim.tick - unchanged["age_start_tick"] == 101
    cursor = cursor_for(sim)
    cursor[0]["energy"] = 1  # Its carried idle cost makes it die during the next tick.
    cursor.apply()
    sim.update()
    child, parent = population(sim)
    assert child["alive"] and parent["alive"]
    assert child["guid"] != previous[0]["guid"]
    assert child["parent_guid"] == previous[1]["guid"]
    assert parent["guid"] == previous[1]["guid"]
    assert parent["birth_tick"] == previous[1]["birth_tick"]
    assert parent["age_start_tick"] == child["age_start_tick"] == sim.tick
    assert child["energy"] == parent["energy"] == INITIAL_ENERGY
    stats = sim.inspect_world(0)
    assert (stats["num_creatures"], stats["births"], stats["deaths"]) == (2, 1, 1)
    # Both the reused low slot and surviving high slot are present after rebuilding.
    heads = sim.grid_heads.to_numpy().view(np.int32).reshape(-1)
    links = sim.grid_next.to_numpy().view(np.int32).reshape(-1)
    members = set()
    for head in heads[heads >= 0]:
        while head >= 0:
            assert int(head) not in members
            members.add(int(head))
            head = links[head]
    assert members == {0, 1}


def test_repeated_slot_reuse_clears_previous_lifetime_history_and_never_reuses_guids(app):
    sim = make_sim(app, creatures=2, capacity=3)
    for index in range(2):
        stage(sim, index)
    fixed_brains(sim, movement=0)
    advance(sim, 5)
    assert all(sim.inspect_vision(0, 0, age=age) is not None for age in range(6))
    guids = {sim.inspect(0, 0)["guid"], sim.inspect(0, 1)["guid"]}
    original_parent_weights = parameters(sim)[1].copy()
    siblings = []
    for death_count in (1, 2):
        cursor = cursor_for(sim)
        cursor[0]["energy"] = 0
        cursor[1]["energy"] = 2 * INITIAL_ENERGY + 10
        cursor.apply()
        sim.update()
        child = sim.inspect(0, 0)
        assert child["guid"] not in guids
        guids.add(child["guid"])
        assert child["birth_tick"] == sim.tick
        assert sim.inspect_vision(0, 0)["tick"] == sim.tick
        assert all(sim.inspect_vision(0, 0, age=age) is None for age in range(1, sim.tick + 1))
        assert sim.inspect_vision(0, 1, age=sim.tick)["tick"] == 0
        weights = parameters(sim)
        np.testing.assert_array_equal(weights[1], original_parent_weights)
        assert_small_mutation(weights[0], original_parent_weights)
        siblings.append(weights[0].copy())
        stats = sim.inspect_world(0)
        assert (stats["num_creatures"], stats["births"], stats["deaths"]) == (
            2,
            death_count,
            death_count,
        )
    assert not np.array_equal(siblings[0], siblings[1])


def test_next_generation_inherits_its_parents_mutated_network(app):
    sim = make_sim(app, capacity=3)
    stage(sim, 0, energy=2 * INITIAL_ENERGY + 10)
    original = parameters(sim)[0].copy()
    sim.update()
    first_child = sim.inspect(0, 1)
    first_child_weights = parameters(sim)[1].copy()
    assert_small_mutation(first_child_weights, original)

    advance(sim, 5)
    history = [sim.inspect_vision(0, 1, age=age) for age in range(6)]
    assert all(frame is not None for frame in history)
    stage(sim, 1, energy=2 * INITIAL_ENERGY + 10)
    sim.update()
    renewed_parent = sim.inspect(0, 1)
    grandchild = sim.inspect(0, 2)
    weights = parameters(sim)
    assert grandchild["parent_guid"] == first_child["guid"]
    assert grandchild["generation"] == 2
    for field in ("guid", "parent_guid", "generation", "birth_tick"):
        assert renewed_parent[field] == first_child[field]
    assert renewed_parent["age_start_tick"] == grandchild["age_start_tick"] == sim.tick
    for age, frame in enumerate(history):
        assert sim.inspect_vision(0, 1, age=age + 1) == frame
    assert sim.inspect_vision(0, 2, age=1) is None
    np.testing.assert_array_equal(weights[0], original)
    np.testing.assert_array_equal(weights[1], first_child_weights)
    assert_small_mutation(weights[2], first_child_weights)


def test_mutation_depends_on_seed_and_world_but_not_unrelated_worlds_or_birth_tick(app):
    def first_children(*, worlds, seed=37, delay=0):
        sim = make_sim(app, capacity=2, worlds=worlds, seed=seed)
        fixed_brains(sim, movement=0)
        advance(sim, delay)
        for world in range(worlds):
            stage(sim, world * sim.creature_capacity, energy=2 * INITIAL_ENERGY + 10)
        sim.update()
        return parameters(sim)[1::2].copy()

    together = first_children(worlds=2)
    isolated_later = first_children(worlds=1, delay=3)
    different_seed = first_children(worlds=1, seed=38)
    np.testing.assert_array_equal(together[0], isolated_later[0])
    assert not np.array_equal(together[0], together[1])
    assert not np.array_equal(together[0], different_seed[0])


def test_lifecycle_reset_replays_births_deaths_weights_and_history(app):
    sim = make_sim(app, creatures=2, capacity=4, worlds=2)
    initial = sim.creatures.to_numpy().copy()

    def run():
        for world in range(2):
            stage(sim, world * sim.creature_capacity, energy=1)
            stage(sim, world * sim.creature_capacity + 1, energy=900_001)
        fixed_brains(sim, movement=0)
        advance(sim, 4)
        assert all(sim.inspect_world(world)["births"] >= 2 for world in range(2))
        assert all(sim.inspect_world(world)["deaths"] == 1 for world in range(2))
        return [
            getattr(sim, name).to_numpy().copy()
            for name in ("creatures", "worlds", "brain_weights", "vision_history", "vision_frames")
        ]

    first = run()
    sim.reset()
    np.testing.assert_array_equal(sim.creatures.to_numpy(), initial)
    assert all(sim.inspect_world(world)["births"] == 0 for world in range(2))
    assert all(sim.inspect_world(world)["deaths"] == 0 for world in range(2))
    for original, replay in zip(first, run(), strict=True):
        np.testing.assert_array_equal(replay, original)


def test_empty_population_can_keep_reserved_slots_without_spawning_spontaneously(app):
    sim = make_sim(app, creatures=0, capacity=5, worlds=2)
    advance(sim, 3)
    assert all(not creature["alive"] for world in range(2) for creature in population(sim, world))
    for world in range(2):
        stats = sim.inspect_world(world)
        assert (stats["num_creatures"], stats["births"], stats["deaths"]) == (0, 0, 0)
    assert sim.brain_activations.to_numpy().view(np.float32).size == 10 * BRAIN_ACTIVATIONS
    np.testing.assert_array_equal(sim.brain_activations.to_numpy().view(np.float32), 0)
