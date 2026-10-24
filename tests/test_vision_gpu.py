"""Check GPU observations against controlled scenes and retained history."""

import math

import numpy as np
import pytest
import slangpy as spy

from tests.gpu_helpers import fixed_brains, rebuild_food_grid
from worms.app import App
from worms.sim import (
    ENERGY_PER_UNIT,
    FOOD_ENERGY,
    GRID_SIZE,
    VISION_FOV,
    VISION_HISTORY,
    VISION_RANGE,
    VISION_SLOTS,
    WORLD_SIZE,
    Sim,
)

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


def make_sim(app, creatures=1, food=0, worlds=1, **kwargs):
    return Sim(
        app.device,
        num_worlds=worlds,
        creatures_per_world=creatures,
        creature_capacity=creatures,
        food_capacity=food,
        initial_food=kwargs.pop("initial_food", food),
        target_food=food,
        food_per_second=kwargs.pop("food_per_second", 0),
        **kwargs,
    )


def cursor_for(sim, field):
    kernel = sim._initialize_food if field == "food" else sim._initialize
    layout = getattr(kernel.reflection, f"g_{field}").type_layout.element_type_layout
    return spy.BufferCursor(layout, getattr(sim, field))


def write_creatures(sim, positions, *, heading=0, alive=None):
    alive = [True] * len(positions) if alive is None else alive
    cursor = cursor_for(sim, "creatures")
    for index, position in enumerate(positions):
        cursor[index]["position"] = spy.float2(*position)
        cursor[index]["velocity"] = spy.float2(0, 0)
        cursor[index]["max_speed"] = 0.0
        cursor[index]["heading"] = heading
        cursor[index]["radii"] = spy.float2(0.5, 0.315)
        cursor[index]["alive"] = alive[index]
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
    rebuild_food_grid(sim)


def assert_empty(slots):
    assert all(slot["distance"] == -1 for slot in slots)
    assert all(slot["food_energy"] == 0 for slot in slots)
    assert all(slot["kind"] == 0 for slot in slots)
    assert all(slot["bearing"] == 0 and slot["size"] == 0 for slot in slots)


def test_all_segments_keep_exact_bearings_distances_and_unquantized_food_energy(app):
    sim = make_sim(app, food=VISION_SLOTS)
    write_creatures(sim, [(100, 100)])
    bearings = -VISION_FOV / 2 + (np.arange(VISION_SLOTS) + 0.5) * VISION_FOV / VISION_SLOTS
    positions = np.column_stack((100 + 10 * np.cos(bearings), 100 + 10 * np.sin(bearings)))
    energies = [FOOD_ENERGY + 17 * index for index in range(VISION_SLOTS)]
    write_food(sim, positions, energies)
    sim.update()

    frame = sim.inspect_vision(0, 0)
    assert frame["tick"] == 1
    assert frame["position"] == pytest.approx((100, 100))
    assert frame["heading"] == 0
    assert len(frame["slots"]) == 16
    for index, slot in enumerate(frame["slots"]):
        assert slot["distance"] == pytest.approx(10, abs=1e-5)
        assert slot["bearing"] == pytest.approx(bearings[index], abs=1e-6)
        assert slot["food_energy"] == pytest.approx(energies[index] / ENERGY_PER_UNIT)
        assert slot["size"] == pytest.approx(0.36)
        assert slot["kind"] == 1


def test_nearest_food_hides_creatures_and_distant_food_energy(app):
    sim = make_sim(app, creatures=3, food=3)
    write_creatures(sim, [(100, 100), (106, 100), (112, 100)])
    creatures = cursor_for(sim, "creatures")
    creatures[1]["radii"] = spy.float2(0.75, 0.45)
    creatures.apply()
    write_food(sim, [(104, 100), (108, 100), (110, 100)], [1024, 2048, 3072])
    sim.update()

    slots = sim.inspect_vision(0, 0)["slots"]
    assert slots[8] == pytest.approx(
        {
            "distance": 4,
            "bearing": 0,
            "food_energy": 1,
            "size": 0.36,
            "kind": 1,
        }
    )
    assert_empty(slots[:8] + slots[9:])


@pytest.mark.parametrize("front_kind", [1, 2])
def test_removing_the_nearest_object_reveals_the_other_kind_without_rewriting_history(
    app, front_kind
):
    sim = make_sim(app, creatures=2, food=1)
    food_distance, creature_distance = (4, 8) if front_kind == 1 else (8, 4)
    write_creatures(sim, [(100, 100), (100 + creature_distance, 100)])
    write_food(sim, [(100 + food_distance, 100)], [3 * FOOD_ENERGY])
    sim.update()

    before = sim.inspect_vision(0, 0)
    nearest = before["slots"][8]
    assert nearest["kind"] == front_kind
    assert nearest["distance"] == 4
    assert nearest["food_energy"] == (3 * FOOD_ENERGY / ENERGY_PER_UNIT if front_kind == 1 else 0)
    assert nearest["size"] == pytest.approx(0.36 if front_kind == 1 else 1)

    if front_kind == 1:
        food = cursor_for(sim, "food")
        food[0]["energy"] = 0
        food.apply()
    else:
        creatures = cursor_for(sim, "creatures")
        creatures[1]["alive"] = False
        creatures.apply()
    sim.update()
    revealed = sim.inspect_vision(0, 0)["slots"][8]
    assert revealed["kind"] == 3 - front_kind
    assert revealed["distance"] == 8
    assert revealed["food_energy"] == (0 if front_kind == 1 else 3 * FOOD_ENERGY / ENERGY_PER_UNIT)
    assert sim.inspect_vision(0, 0, age=1) == before


def test_occlusion_is_per_segment_even_when_object_bearings_differ(app):
    sim = make_sim(app, creatures=3, food=1)
    # The food and first creature have different bearings but share segment 8.
    # The second creature is farther away in adjacent segment 9 and stays visible.
    angles = (0.05, 0.16, 0.25)
    positions = [
        (100 + distance * math.cos(angle), 100 + distance * math.sin(angle))
        for distance, angle in zip((4, 8, 12), angles, strict=True)
    ]
    write_creatures(sim, [(100, 100), positions[1], positions[2]])
    write_food(sim, [positions[0]])
    sim.update()
    slots = sim.inspect_vision(0, 0)["slots"]
    assert slots[8]["kind"] == 1
    assert slots[8]["bearing"] == pytest.approx(angles[0], abs=1e-6)
    assert slots[8]["distance"] == pytest.approx(4, abs=1e-5)
    assert slots[9]["kind"] == 2
    assert slots[9]["distance"] == pytest.approx(12, abs=1e-5)
    assert_empty(slots[:8] + slots[10:])


def test_exact_distance_ties_prefer_food_then_the_lowest_food_index(app):
    sim = make_sim(app, creatures=2, food=2)
    write_creatures(sim, [(100, 100), (108, 100)])
    write_food(sim, [(108, 100), (108, 100)], [FOOD_ENERGY, 2 * FOOD_ENERGY])
    # Capture directly: an update would collect the co-located food first.
    encoder = sim.device.create_command_encoder()
    sim._update_grid(encoder)
    sim._update_vision(encoder)
    sim.device.submit_command_buffer(encoder.finish())
    slot = sim.inspect_vision(0, 0)["slots"][8]
    assert slot["distance"] == 8
    assert slot["kind"] == 1
    assert slot["food_energy"] == FOOD_ENERGY / ENERGY_PER_UNIT


def test_range_and_fov_endpoints_are_inclusive_but_points_outside_are_excluded(app):
    sim = make_sim(app, food=7)
    write_creatures(sim, [(100, 100)])
    write_food(
        sim,
        [(100, 90), (100, 110), (124, 100), (124.01, 100), (99.99, 90), (99.99, 110), (90, 100)],
    )
    sim.update()
    slots = sim.inspect_vision(0, 0)["slots"]

    assert [index for index, slot in enumerate(slots) if slot["kind"] != 0] == [0, 8, 15]
    assert slots[0]["bearing"] == pytest.approx(-math.pi / 2)
    assert slots[15]["bearing"] == pytest.approx(math.pi / 2)
    assert slots[8]["distance"] == VISION_RANGE
    assert sum(slot["food_energy"] for slot in slots) == 3 * FOOD_ENERGY / ENERGY_PER_UNIT


def test_stationary_heading_is_retained_and_movement_changes_capture_pose_first(app):
    sim = make_sim(app, food=1)
    write_creatures(sim, [(100, 100)], heading=math.pi / 2)
    write_food(sim, [(100, 110)])
    sim.update()
    stationary = sim.inspect_vision(0, 0)
    assert stationary["heading"] == pytest.approx(math.pi / 2)
    assert stationary["slots"][8]["distance"] == pytest.approx(10)
    assert stationary["slots"][8]["bearing"] == pytest.approx(0, abs=1e-6)

    creatures = cursor_for(sim, "creatures")
    creatures[0]["velocity"] = spy.float2(-60, 0)
    creatures[0]["heading"] = math.pi
    creatures[0]["max_speed"] = 60.0
    creatures.apply()
    sim.update()
    turned = sim.inspect_vision(0, 0)
    assert turned["position"] == pytest.approx((99, 100))
    assert abs(turned["heading"]) == pytest.approx(math.pi)
    assert_empty(turned["slots"])
    assert sim.inspect_vision(0, 0, age=1) == stationary


def test_wrapped_visibility_keeps_worlds_separate_and_ignores_empty_slots(app):
    sim = make_sim(app, creatures=2, food=2, worlds=2)
    write_creatures(sim, [(511, 510), (5, 4), (511, 510), (250, 250)], heading=math.pi / 4)
    write_food(sim, [(3, 2), (2, 1), (200, 200), (3, 2)], [FOOD_ENERGY, 0, FOOD_ENERGY, 0])
    sim.update()

    visible = sim.inspect_vision(0, 0)["slots"][8]
    assert visible["distance"] == pytest.approx(math.sqrt(32))
    assert visible["bearing"] == pytest.approx(0, abs=1e-6)
    assert visible["food_energy"] == FOOD_ENERGY / ENERGY_PER_UNIT
    assert visible["kind"] == 1  # The nearer food hides the creature across both seams.
    assert_empty(sim.inspect_vision(1, 0)["slots"])


def test_equal_distance_creatures_have_a_stable_tie_and_dead_observers_keep_old_history(app):
    sim = make_sim(app, creatures=4)
    write_creatures(
        sim, [(100, 100), (108, 100), (108, 100), (104, 100)], alive=[True, True, True, False]
    )
    creatures = cursor_for(sim, "creatures")
    creatures[1]["radii"] = spy.float2(0.75, 0.45)
    creatures[2]["radii"] = spy.float2(1.25, 0.75)
    creatures.apply()
    sim.update()
    before = sim.inspect_vision(0, 0)
    assert before["slots"][8]["distance"] == 8
    assert before["slots"][8]["size"] == 1.5
    assert before["slots"][8]["kind"] == 2
    assert before["slots"][8]["food_energy"] == 0
    assert sim.inspect_vision(0, 3) is None

    creatures = cursor_for(sim, "creatures")
    creatures[0]["alive"] = False
    creatures.apply()
    sim.update()
    assert sim.inspect_vision(0, 0) is None
    assert sim.inspect_vision(0, 0, age=1) == before


def test_collection_removes_food_from_current_vision_without_rewriting_history(app):
    sim = make_sim(app, food=1)
    write_creatures(sim, [(100, 100)])
    write_food(sim, [(104, 100)])
    sim.update()
    before = sim.inspect_vision(0, 0)
    assert before["slots"][8]["distance"] == 4

    write_creatures(sim, [(104, 100)])
    sim.update()
    assert_empty(sim.inspect_vision(0, 0)["slots"])
    assert sim.inspect_vision(0, 0, age=1) == before
    assert sim.inspect_vision(0, 0)["position"] == pytest.approx((104, 100))


def test_food_spawned_this_tick_is_visible_in_the_same_capture(app):
    sim = make_sim(app, food=1, initial_food=0, food_per_second=60)
    sim.update()
    spawn_position = tuple(cursor_for(sim, "food")[0]["position"].read())
    sim.reset()
    observer = ((spawn_position[0] - 8) % WORLD_SIZE, spawn_position[1])
    write_creatures(sim, [observer])
    sim.update()

    frame = sim.inspect_vision(0, 0)
    assert frame["slots"][8]["distance"] == pytest.approx(8, abs=2e-5)
    assert frame["slots"][8]["food_energy"] == FOOD_ENERGY / ENERGY_PER_UNIT
    assert_empty(sim.inspect_vision(0, 0, age=1)["slots"])


def test_cached_food_grid_keeps_high_slots_and_ignores_eaten_links_without_cross_world_entries(app):
    sim = make_sim(app, food=5, worlds=2)
    write_creatures(sim, [(100, 100), (100, 100)])
    write_food(
        sim,
        [(108, 100), (108, 100), (109, 100), (109, 100), (109, 100)] * 2,
        [0, FOOD_ENERGY, 0, 0, 2 * FOOD_ENERGY] * 2,
    )
    sim.update()

    def members():
        heads = sim.food_grid_heads.to_numpy().view(np.int32).reshape(2, GRID_SIZE**2)
        links = sim.food_grid_next.to_numpy().view(np.int32).reshape(-1)
        found = []
        for world, cells in enumerate(heads):
            seen = set()
            for head in cells[cells >= 0]:
                index = int(head)
                while index >= 0:
                    assert world * 5 <= index < (world + 1) * 5
                    assert index not in seen
                    seen.add(index)
                    index = int(links[index])
            found.append(seen)
        return found, links

    assert members()[0] == [{1, 4}, {6, 9}]
    for world in range(2):
        assert (
            sim.inspect_vision(world, 0)["slots"][8]["food_energy"] == FOOD_ENERGY / ENERGY_PER_UNIT
        )
    food = cursor_for(sim, "food")
    food[4]["energy"] = 0
    food.apply()
    sim.update()
    found, links = members()
    assert found == [{1, 4}, {6, 9}]
    assert links[4] == 1
    # Rebuilding removes the dead link; retaining it must not change sensing.
    rebuild_food_grid(sim)
    assert members()[0] == [{1}, {6, 9}]
    assert members()[1][4] == -1
    assert sim.inspect_vision(0, 0)["slots"][8]["food_energy"] == FOOD_ENERGY / ENERGY_PER_UNIT


def test_changed_food_energy_is_captured_without_rewriting_older_observations(app):
    sim = make_sim(app, food=1)
    write_creatures(sim, [(100, 100)])
    write_food(sim, [(104, 100)])
    sim.update()
    before = sim.inspect_vision(0, 0)
    assert before["slots"][8]["food_energy"] == FOOD_ENERGY / ENERGY_PER_UNIT

    sim.food_energy = 6.25
    sim.update()
    current = sim.inspect_vision(0, 0)
    assert current["slots"][8]["food_energy"] == 6.25
    assert current["slots"][8]["distance"] == before["slots"][8]["distance"]
    assert sim.inspect_vision(0, 0, age=1) == before


def test_respawning_an_eaten_slot_rebuilds_its_cached_location(app):
    sim = make_sim(app, food=1)
    write_creatures(sim, [(100, 100)])
    write_food(sim, [(104, 100)])
    sim.update()
    write_creatures(sim, [(104, 100)])
    sim.update()
    assert cursor_for(sim, "food")[0]["energy"].read() == 0
    old_cell = 25 * GRID_SIZE + 26
    assert sim.food_grid_heads.to_numpy().view(np.int32).reshape(-1)[old_cell] == 0

    sim.food_per_second = 60
    sim.update()
    spawned = tuple(cursor_for(sim, "food")[0]["position"].read())
    new_cell = int(spawned[1] // 4) * GRID_SIZE + int(spawned[0] // 4)
    assert new_cell != old_cell
    heads = sim.food_grid_heads.to_numpy().view(np.int32).reshape(-1)
    assert heads[old_cell] == -1 and heads[new_cell] == 0
    sim.food_per_second = 0
    write_creatures(sim, [((spawned[0] - 8) % WORLD_SIZE, spawned[1])])
    sim.update()
    visible = sim.inspect_vision(0, 0)["slots"][8]
    assert visible["kind"] == 1
    assert visible["distance"] == pytest.approx(8, abs=2e-5)


def test_history_rollover_preserves_full_frames_and_reset_invalidates_old_samples(app):
    sim = make_sim(app)
    assert sim.inspect_vision(0, 0)["tick"] == 0
    with pytest.raises(ValueError):
        sim.inspect_vision(0, 0, age=1)
    with pytest.raises(ValueError):
        sim.inspect_vision(0, 0, age=-1)
    write_creatures(sim, [(250, 250)])
    creatures = cursor_for(sim, "creatures")
    creatures[0]["velocity"] = spy.float2(60, 0)
    creatures[0]["heading"] = 0.0
    creatures[0]["max_speed"] = 60.0
    creatures.apply()
    encoder = app.device.create_command_encoder()
    for _ in range(VISION_HISTORY + 4):
        sim.update(encoder)
    app.device.submit_command_buffer(encoder.finish())

    newest = sim.inspect_vision(0, 0)
    oldest = sim.inspect_vision(0, 0, age=VISION_HISTORY - 1)
    assert newest["tick"] == VISION_HISTORY + 4
    assert newest["position"] == pytest.approx((510, 250))
    assert oldest["tick"] == 5
    assert oldest["position"] == pytest.approx((255, 250))
    with pytest.raises(ValueError):
        sim.inspect_vision(0, 0, age=VISION_HISTORY)
    sim.reset()
    assert sim.inspect_vision(0, 0)["tick"] == 0
    with pytest.raises(ValueError):
        sim.inspect_vision(0, 0, age=1)
    headers = (
        sim.vision_frames.to_numpy()
        .view(np.dtype([("position", "<f4", (2,)), ("heading", "<f4"), ("tick", "<i4")]))
        .reshape(-1)
    )
    assert np.count_nonzero(headers["tick"] >= 0) == 1


def test_pause_does_not_append_history_and_single_step_appends_once(app):
    previous_sim, previous_pause = app.sim, app.paused
    try:
        app.sim = make_sim(app, food=1)
        app.paused = True
        before = app.sim.vision_history.to_numpy().copy()
        headers = app.sim.vision_frames.to_numpy().copy()
        app.frame()
        app.frame()
        assert app.sim.tick == 0
        np.testing.assert_array_equal(app.sim.vision_history.to_numpy(), before)
        np.testing.assert_array_equal(app.sim.vision_frames.to_numpy(), headers)
        app.step()
        app.frame()
        assert app.sim.tick == 1
        assert app.sim.inspect_vision(0, 0)["tick"] == 1
        assert app.sim.inspect_vision(0, 0, age=1)["tick"] == 0
        app.frame()
        assert app.sim.tick == 1
    finally:
        app.sim, app.paused = previous_sim, previous_pause


def test_seeded_capture_and_full_history_replay_on_the_same_gpu(app):
    sim = make_sim(app, creatures=4, food=64, worlds=2, seed=37)

    def run():
        encoder = app.device.create_command_encoder()
        for _ in range(12):
            sim.update(encoder)
        app.device.submit_command_buffer(encoder.finish())
        return sim.vision_history.to_numpy().copy(), sim.vision_frames.to_numpy().copy()

    first_slots, first_frames = run()
    sim.reset()
    replay_slots, replay_frames = run()
    np.testing.assert_array_equal(replay_slots, first_slots)
    np.testing.assert_array_equal(replay_frames, first_frames)
