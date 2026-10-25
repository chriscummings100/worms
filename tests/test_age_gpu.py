"""Check lifespan, optional food-intake fading and renewal on the real GPU."""

import numpy as np
import pytest

from tests.gpu_helpers import fixed_brains, grid_membership
from tests.test_contact_gpu import creature_energies, cursor_for, food_records, write_food
from tests.test_lifecycle_gpu import stage
from worms.app import App
from worms.sim import FOOD_ENERGY, INITIAL_ENERGY, Sim

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


def make_sim(app, *, creatures=1, capacity=None, worlds=1, food=1, **settings):
    return Sim(
        app.device,
        num_worlds=worlds,
        creatures_per_world=creatures,
        creature_capacity=creatures if capacity is None else capacity,
        food_capacity=food,
        initial_food=food,
        target_food=food,
        food_per_second=0,
        **settings,
    )


@pytest.mark.parametrize(
    "age,alive",
    [(99_998, True), (99_999, True), (100_000, False), (100_001, False)],
)
def test_default_lifespan_keeps_full_intake_until_death_at_100000_ticks(app, age, alive):
    sim = make_sim(app)
    assert sim.food_age_start == 100_000 and sim.food_age_fade == 0
    stage(sim, 0)
    fixed_brains(sim, movement=0)
    write_food(sim, [(100, 100)])
    sim.tick = age - 1

    sim.update()

    creature = sim.inspect(0, 0)
    assert creature["energy"] == (INITIAL_ENERGY + FOOD_ENERGY if alive else 0)
    assert creature["alive"] == alive
    assert sim.tick == age and creature["birth_tick"] == creature["age_start_tick"] == 0
    if alive:
        assert 0 < creature["energy_remainder"] < 1  # Maintenance still applies.
    else:
        assert creature["energy_remainder"] == 0
    assert sim.inspect_world(0)["deaths"] == int(not alive)
    assert sim.inspect_world(0)["num_food"] == 0
    np.testing.assert_array_equal(food_records(sim)["energy"], 0)


@pytest.mark.parametrize(
    "age,gain",
    [(20_000, 5120), (20_001, 5119), (25_000, 2560), (29_999, 0), (30_000, None)],
)
def test_optional_fade_still_reduces_intake_and_dies_at_its_end(app, age, gain):
    sim = make_sim(app, food_age_start=20_000, food_age_fade=10_000)
    stage(sim, 0)
    fixed_brains(sim, movement=0)
    write_food(sim, [(100, 100)])
    sim.tick = age - 1

    sim.update()

    creature = sim.inspect(0, 0)
    assert creature["alive"] == (gain is not None)
    assert creature["energy"] == (0 if gain is None else INITIAL_ENERGY + gain)
    assert sim.inspect_world(0)["num_food"] == 0


def test_expiry_kills_without_food_despite_reserves_before_it_can_split(app):
    sim = make_sim(app, capacity=2, food=0)
    stage(sim, 0, energy=3 * INITIAL_ENERGY)
    fixed_brains(sim, movement=0)
    sim.tick = 99_999

    sim.update()

    creature = sim.inspect(0, 0)
    assert not creature["alive"] and creature["energy"] == 0
    assert creature["velocity"] == (0, 0)
    assert creature["age_start_tick"] == 0
    assert not sim.inspect(0, 1)["alive"]
    assert sim.inspect_vision(0, 0)["tick"] == 100_000
    np.testing.assert_array_equal(sim.brain_activations.to_numpy().view(np.float16), 0)
    assert grid_membership(sim.grid_heads.to_numpy(), sim.grid_next.to_numpy()) == [{(25, 25): {0}}]
    for _ in range(3):
        sim.update()
    assert sim.inspect_vision(0, 0) is None
    assert sim.inspect_vision(0, 0, age=3)["tick"] == 100_000
    np.testing.assert_array_equal(sim.grid_heads.to_numpy().view(np.int32), -1)
    world = sim.inspect_world(0)
    assert (world["num_creatures"], world["births"], world["deaths"]) == (0, 0, 1)


def test_intake_uses_each_creatures_age_start_tick_in_each_world(app):
    sim = make_sim(app, creatures=2, worlds=2, food=2, food_age_start=10, food_age_fade=20)
    positions = [(100, 100), (200, 200)] * 2
    for index, position in enumerate(positions):
        stage(sim, index, position=position)
    fixed_brains(sim, movement=0)
    creatures = cursor_for(sim, "creatures")
    for index, age_start_tick in enumerate((31, 21, 11, 40)):
        creatures[index]["age_start_tick"] = age_start_tick
    creatures.apply()
    write_food(sim, positions)
    sim.tick = 40

    sim.update()  # Settled ages: 10, 20, 30, 1.

    expected = np.array(
        [[INITIAL_ENERGY + 5120, INITIAL_ENERGY + 2560], [0, INITIAL_ENERGY + 5120]],
        dtype=np.uint64,
    )
    np.testing.assert_array_equal(creature_energies(sim), expected)
    np.testing.assert_array_equal(food_records(sim)["energy"], 0)
    assert [sim.inspect_world(world)["num_food"] for world in range(2)] == [0, 0]
    assert [sim.inspect_world(world)["deaths"] for world in range(2)] == [0, 1]


def test_collected_energy_is_summed_before_age_rounding_without_integer_overflow(app):
    sim = make_sim(app, food=2, food_age_start=10, food_age_fade=3)
    starting_energy = 2**32 + 17
    stage(sim, 0, energy=starting_energy)
    fixed_brains(sim, movement=0)
    # Two-thirds of each piece has a fractional quantum. Round the combined
    # intake once, and use a wide product even though credits fit in uint32.
    write_food(sim, [(100, 100)] * 2, [2_147_483_002] * 2)
    sim.tick = 10

    sim.update()

    assert sim.inspect(0, 0)["energy"] == starting_energy + 2_863_310_669
    np.testing.assert_array_equal(food_records(sim)["energy"], 0)
    assert sim.inspect_world(0)["num_food"] == 0


def test_splitting_creates_two_new_lifetimes_with_full_intake(app):
    sim = make_sim(app, capacity=2, food=2)
    stage(sim, 0, energy=2 * INITIAL_ENERGY + 10)
    fixed_brains(sim, movement=0)
    creatures = cursor_for(sim, "creatures")
    creatures[0]["birth_tick"] = 123
    creatures[0]["age_start_tick"] = 123
    creatures.apply()
    original_guid = sim.inspect(0, 0)["guid"]
    write_food(sim, [(300, 300)] * 2, [0, 0])
    sim.tick = 99_998

    sim.update()

    parent, child = [sim.inspect(0, index) for index in range(2)]
    assert parent["guid"] != original_guid and child["guid"] != original_guid
    assert parent["guid"] != child["guid"]
    assert parent["birth_tick"] == child["birth_tick"] == 99_999
    assert parent["parent_guid"] == child["parent_guid"] == original_guid
    assert parent["age_start_tick"] == child["age_start_tick"] == sim.tick
    assert sim.inspect_world(0)["births"] == 1
    # Hold both in place with a fresh cost remainder to isolate their intake.
    for index, creature in enumerate((parent, child)):
        stage(sim, index, position=creature["position"])
    write_food(sim, [creature["position"] for creature in (parent, child)])

    sim.update()

    parent, child = [sim.inspect(0, index) for index in range(2)]
    assert parent["alive"] and child["alive"]
    assert parent["energy"] == child["energy"] == INITIAL_ENERGY + FOOD_ENERGY
    assert parent["birth_tick"] == child["birth_tick"] == 99_999
    assert sim.tick - parent["age_start_tick"] == sim.tick - child["age_start_tick"] == 1
    np.testing.assert_array_equal(food_records(sim)["energy"], 0)

    sim.tick = 100_122  # The retired parent's lifespan does not apply to either child.
    sim.update()
    assert all(sim.inspect(0, index)["alive"] for index in range(2))
    assert sim.inspect_world(0)["deaths"] == 0


def test_live_age_settings_apply_on_step_and_survive_paused_reset(app):
    previous_sim, previous_pause = app.sim, app.paused
    try:
        app.sim = sim = make_sim(app)
        stage(sim, 0)
        fixed_brains(sim, movement=0)
        write_food(sim, [(100, 100)])
        sim.tick = 24_999
        app.paused = True
        sim.food_age_start = 24_000
        sim.food_age_fade = 4_000
        before = sim.creatures.to_numpy().copy()
        app.frame()
        assert sim.tick == 24_999
        np.testing.assert_array_equal(sim.creatures.to_numpy(), before)
        assert food_records(sim)["energy"][0, 0] == FOOD_ENERGY

        app.step()
        app.frame()
        assert sim.tick == 25_000
        assert sim.inspect(0, 0)["energy"] == INITIAL_ENERGY + 3840

        sim.food_age_fade = 0
        stage(sim, 0)
        write_food(sim, [(100, 100)])
        app.step()
        app.frame()
        assert not sim.inspect(0, 0)["alive"]
        assert sim.inspect(0, 0)["energy"] == 0
        assert sim.inspect_world(0)["deaths"] == 1
        assert sim.inspect_world(0)["num_food"] == 0

        app.reset()
        app.frame()
        assert app.paused and sim.tick == 0
        assert sim.food_age_start == 24_000 and sim.food_age_fade == 0
        reset_creature = sim.inspect(0, 0)
        assert reset_creature["birth_tick"] == reset_creature["age_start_tick"] == 0
        stage(sim, 0)
        fixed_brains(sim, movement=0)
        write_food(sim, [(100, 100)])
        app.step()
        app.frame()
        assert sim.tick == 1
        assert sim.inspect(0, 0)["energy"] == INITIAL_ENERGY + FOOD_ENERGY
    finally:
        app.sim, app.paused = previous_sim, previous_pause
