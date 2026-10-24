"""Compare real GPU policies with NumPy and check their movement contract."""

import math

import numpy as np
import pytest
import slangpy as spy

from tests.gpu_helpers import fixed_brains, rebuild_food_grid
from worms.app import App
from worms.sim import (
    BRAIN_ACTIVATIONS,
    BRAIN_HIDDEN,
    BRAIN_INPUTS,
    BRAIN_PARAMETERS,
    FOOD_ENERGY,
    MAX_TURN_RATE,
    TIME_STEP,
    VISION_CREATURE,
    VISION_FOOD,
    VISION_NONE,
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


def make_sim(app, *, worlds=1, creatures=1, food=0, seed=17):
    return Sim(
        app.device,
        num_worlds=worlds,
        creatures_per_world=creatures,
        creature_capacity=creatures,
        food_capacity=food,
        initial_food=food,
        target_food=food,
        food_per_second=0,
        seed=seed,
    )


def creature_cursor(sim):
    layout = sim._initialize.reflection.g_creatures.type_layout.element_type_layout
    return spy.BufferCursor(layout, sim.creatures)


def weights(sim):
    return sim.brain_weights.to_numpy().view(np.float32).reshape(-1, BRAIN_PARAMETERS)


def activations(sim):
    return sim.brain_activations.to_numpy().view(np.float32).reshape(-1, BRAIN_ACTIVATIONS)


def evaluate(sim):
    encoder = sim.device.create_command_encoder()
    sim._update_brains(encoder)
    sim.device.submit_command_buffer(encoder.finish())


def capture(sim):
    encoder = sim.device.create_command_encoder()
    sim._update_grid(encoder)
    sim._update_vision(encoder)
    sim._update_brains(encoder)
    sim.device.submit_command_buffer(encoder.finish())


def reference(slots, parameters):
    inputs = []
    for slot in slots:
        food = slot["kind"] == VISION_FOOD
        creature = slot["kind"] == VISION_CREATURE
        present = food or creature
        inputs.extend(
            (
                np.clip(1 - slot["distance"] / VISION_RANGE, 0, 1) if present else 0,
                slot["size"] / (slot["size"] + 1) if present else 0,
                float(food),
                float(creature),
            )
        )
    inputs = np.array(inputs)
    hidden = np.tanh(parameters[:1024].reshape(16, 64) @ inputs + parameters[1024:1040])
    raw = parameters[1040:1072].reshape(2, 16) @ hidden + parameters[1072:1074]
    return np.concatenate((inputs, hidden, [np.tanh(raw[0]), np.clip(0.5 + raw[1], 0, 1)]))


def test_all_neurons_match_numpy_for_distinct_worlds_segments_and_missing_targets(app):
    sim = make_sim(app, worlds=2, creatures=3)
    layout = sim._capture_vision.reflection.vision_history.type_layout.element_type_layout
    slots = spy.BufferCursor(layout, sim.vision_history)
    for creature in range(6):
        for segment in range(VISION_SLOTS):
            slot = slots[creature * VISION_SLOTS + segment]
            kind = (creature + segment) % 3
            slot["distance"] = float(segment) if kind != VISION_NONE else -1.0
            slot["bearing"] = -1.3 + segment * 0.16
            slot["food_energy"] = float(creature + segment + 1)
            slot["size"] = 0.36 if kind == VISION_FOOD else 0.5 + creature * 0.25
            slot["kind"] = kind
    slots.apply()
    parameters = weights(sim).copy()
    # Exercise both bias arrays as well as the ordinary zero-bias initialization.
    parameters[:, 1024:1040] = np.linspace(-0.15, 0.15, BRAIN_HIDDEN)
    parameters[:, 1072:1074] = (0.1, -0.2)
    sim.brain_weights.copy_from_numpy(parameters)
    before_history = sim.vision_history.to_numpy().copy()
    evaluate(sim)

    actual = activations(sim)
    for index in range(6):
        world, local = divmod(index, 3)
        observation = sim.inspect_vision(world, local)
        expected = reference(observation["slots"], parameters[index])
        np.testing.assert_allclose(actual[index], expected, rtol=2e-5, atol=2e-6)
        segments = actual[index, :64].reshape(VISION_SLOTS, 4)
        for segment, slot in enumerate(observation["slots"]):
            kind = slot["kind"]
            np.testing.assert_array_equal(segments[segment, 2:], (kind == 1, kind == 2))
            if kind == VISION_NONE:
                np.testing.assert_array_equal(segments[segment], 0)
        inspected = sim.inspect_brain(world, local)
        np.testing.assert_array_equal(inspected["inputs"], actual[index, :64])
        np.testing.assert_array_equal(inspected["hidden"], actual[index, 64:80])
        assert inspected["steering"] == actual[index, 80]
        assert inspected["movement"] == actual[index, 81]
        assert inspected["tick"] == 0
    assert np.ptp(actual[:, 80]) > 0.1
    np.testing.assert_array_equal(sim.vision_history.to_numpy(), before_history)

    # Bearing and food energy are retained for inspection, never fed to this brain.
    for index in range(6 * VISION_SLOTS):
        slots[index]["bearing"] = 0.0
        slots[index]["food_energy"] = 10000.0
    slots.apply()
    evaluate(sim)
    np.testing.assert_array_equal(activations(sim), actual)


def test_seeded_weights_are_independent_stable_by_identity_and_fixed_between_ticks(app):
    sim = make_sim(app, worlds=2, creatures=3, food=32)
    original = weights(sim).copy()
    assert np.isfinite(original).all()
    np.testing.assert_array_equal(original[:, 1024:1040], 0)
    np.testing.assert_array_equal(original[:, 1072:1074], 0)
    assert np.abs(original[:, :1024]).max() <= math.sqrt(6 / 80)
    assert np.abs(original[:, 1040:1072]).max() <= math.sqrt(6 / 18)
    assert len({row.tobytes() for row in original}) == 6

    larger = make_sim(app, worlds=3, creatures=5)
    expanded = weights(larger).reshape(3, 5, BRAIN_PARAMETERS)
    np.testing.assert_array_equal(expanded[:2, :3], original.reshape(2, 3, BRAIN_PARAMETERS))
    changed_seed = make_sim(app, worlds=2, creatures=3, seed=18)
    assert not np.array_equal(weights(changed_seed), original)

    def run():
        encoder = app.device.create_command_encoder()
        for _ in range(12):
            sim.update(encoder)
        app.device.submit_command_buffer(encoder.finish())
        return sim.creatures.to_numpy().copy(), activations(sim).copy()

    first_creatures, first_activity = run()
    np.testing.assert_array_equal(weights(sim), original)
    sim.reset()
    np.testing.assert_array_equal(weights(sim), original)
    replay_creatures, replay_activity = run()
    np.testing.assert_array_equal(replay_creatures, first_creatures)
    np.testing.assert_array_equal(replay_activity, first_activity)


def test_empty_vision_produces_straight_half_speed_with_random_weights(app):
    sim = make_sim(app, worlds=2)
    activity = activations(sim)
    np.testing.assert_array_equal(activity[:, :81], 0)
    np.testing.assert_array_equal(activity[:, 81], 0.5)
    for world in range(2):
        before = sim.inspect(world, 0)
        assert 8 <= before["max_speed"] <= 16
    sim.update()
    for world in range(2):
        creature = sim.inspect(world, 0)
        assert np.linalg.norm(creature["velocity"]) == pytest.approx(
            creature["max_speed"] / 2, abs=1e-6
        )
        assert sim.inspect_brain(world, 0)["steering"] == 0


@pytest.mark.parametrize("front_kind", [VISION_FOOD, VISION_CREATURE])
def test_hidden_objects_cannot_change_inputs_or_network_activity(app, front_kind):
    sim = make_sim(app, creatures=2, food=1)
    creatures = creature_cursor(sim)
    creatures[0]["position"] = spy.float2(100, 100)
    creatures[0]["heading"] = 0.0
    creatures[1]["position"] = spy.float2(108 if front_kind == VISION_FOOD else 104, 100)
    creatures[1]["radii"] = spy.float2(0.5, 0.315)
    creatures.apply()
    layout = sim._initialize_food.reflection.g_food.type_layout.element_type_layout
    food = spy.BufferCursor(layout, sim.food)
    food[0]["position"] = spy.float2(104 if front_kind == VISION_FOOD else 108, 100)
    food[0]["energy"] = FOOD_ENERGY
    food.apply()
    capture(sim)
    observed = sim.inspect_vision(0, 0)
    before = activations(sim)[0].copy()
    assert observed["slots"][8]["kind"] == front_kind
    np.testing.assert_array_equal(
        before[8 * 4 + 2 : 8 * 4 + 4],
        (front_kind == VISION_FOOD, front_kind == VISION_CREATURE),
    )

    if front_kind == VISION_FOOD:
        creatures[1]["position"] = spy.float2(112, 100)
        creatures[1]["radii"] = spy.float2(2, 1.26)
        creatures.apply()
    else:
        food[0]["position"] = spy.float2(112, 100)
        food[0]["energy"] = 100 * FOOD_ENERGY
        food.apply()
    capture(sim)
    assert sim.inspect_vision(0, 0) == observed
    np.testing.assert_array_equal(activations(sim)[0], before)


@pytest.mark.parametrize("steering,movement", [(-1, 0), (1, 0), (-1, 0.25), (1, 1)])
def test_turning_and_throttle_use_max_speed_even_when_stopped_or_wrapping(app, steering, movement):
    sim = make_sim(app)
    cursor = creature_cursor(sim)
    cursor[0]["position"] = spy.float2(WORLD_SIZE - 0.01, WORLD_SIZE - 0.01)
    cursor[0]["heading"] = 0.0
    cursor[0]["max_speed"] = 12.0
    cursor[0]["velocity"] = spy.float2(999, -999)
    cursor.apply()
    fixed_brains(sim, steering=steering, movement=movement)
    expected_position = np.array((WORLD_SIZE - 0.01,) * 2)
    for tick in range(1, 4):
        sim.update()
        creature = sim.inspect(0, 0)
        heading = steering * MAX_TURN_RATE * TIME_STEP * tick
        velocity = np.array((math.cos(heading), math.sin(heading))) * 12 * movement
        expected_position = (expected_position + velocity * TIME_STEP) % WORLD_SIZE
        assert creature["heading"] == pytest.approx(heading, abs=1e-6)
        assert creature["velocity"] == pytest.approx(velocity, abs=2e-6)
        assert creature["position"] == pytest.approx(expected_position, abs=8e-5)
        assert creature["max_speed"] == 12


@pytest.mark.parametrize("raw_movement,expected", [(-4, 0), (4, 1)])
def test_movement_output_clamps_to_exact_stop_and_full_speed(app, raw_movement, expected):
    sim = make_sim(app)
    fixed_brains(sim, steering=1, movement=raw_movement)
    brain = sim.inspect_brain(0, 0)
    assert brain["movement"] == expected
    assert brain["steering"] == pytest.approx(1)
    sim.update()
    creature = sim.inspect(0, 0)
    assert np.linalg.norm(creature["velocity"]) == pytest.approx(creature["max_speed"] * expected)


def test_current_vision_drives_the_next_step_and_old_history_remains_raw(app):
    sim = make_sim(app, food=1)
    creatures = creature_cursor(sim)
    creatures[0]["position"] = spy.float2(100, 100)
    creatures[0]["heading"] = 0.0
    creatures[0]["max_speed"] = 0.0
    creatures.apply()
    layout = sim._initialize_food.reflection.g_food.type_layout.element_type_layout
    food = spy.BufferCursor(layout, sim.food)
    food[0]["position"] = spy.float2(108, 100)
    food[0]["energy"] = FOOD_ENERGY
    food.apply()
    parameters = np.zeros((1, BRAIN_PARAMETERS), dtype=np.float32)
    parameters[0, 8 * 4] = 1  # Forward food proximity -> first hidden neuron.
    parameters[0, 1040] = 1  # First hidden neuron -> steering.
    sim.brain_weights.copy_from_numpy(parameters)
    capture(sim)
    observed = sim.inspect_vision(0, 0)
    command = sim.inspect_brain(0, 0)["steering"]
    assert command == pytest.approx(math.tanh(math.tanh(1 - 8 / 24)), rel=2e-5, abs=2e-6)

    food[0]["position"] = spy.float2(92, 100)
    food.apply()
    rebuild_food_grid(sim)
    sim.update()
    creature = sim.inspect(0, 0)
    assert creature["heading"] == pytest.approx(command * MAX_TURN_RATE * TIME_STEP, abs=1e-6)
    assert sim.inspect_brain(0, 0)["steering"] == 0
    assert sim.inspect_vision(0, 0, age=1) == observed
    current = sim.inspect_vision(0, 0)
    assert sim.inspect_brain(0, 0)["tick"] == current["tick"] == 1
    np.testing.assert_allclose(activations(sim)[0], reference(current["slots"], parameters[0]))
    sim.update()
    assert sim.inspect(0, 0)["heading"] == creature["heading"]


def test_dead_creatures_clear_activity_and_do_not_execute_stale_commands(app):
    sim = make_sim(app)
    fixed_brains(sim, steering=1, movement=1)
    creatures = creature_cursor(sim)
    creatures[0]["alive"] = False
    creatures.apply()
    before = sim.inspect(0, 0)
    evaluate(sim)
    np.testing.assert_array_equal(activations(sim), 0)
    sim.update()
    after = sim.inspect(0, 0)
    assert after["position"] == before["position"]
    assert after["heading"] == before["heading"]
    np.testing.assert_array_equal(activations(sim), 0)


def test_pause_freezes_activity_and_single_step_keeps_brain_and_vision_ticks_together(app):
    previous_sim, previous_pause = app.sim, app.paused
    try:
        app.sim = make_sim(app, creatures=2, food=8)
        app.paused = True
        before = activations(app.sim).copy()
        app.frame()
        app.frame()
        assert app.sim.tick == 0
        np.testing.assert_array_equal(activations(app.sim), before)
        app.step()
        app.frame()
        assert app.sim.tick == 1
        assert app.paused
        stepped = activations(app.sim).copy()
        for local in range(2):
            frame = app.sim.inspect_vision(0, local)
            brain = app.sim.inspect_brain(0, local)
            assert brain["tick"] == frame["tick"] == 1
            np.testing.assert_allclose(
                stepped[local], reference(frame["slots"], weights(app.sim)[local]), atol=2e-6
            )
        app.frame()
        np.testing.assert_array_equal(activations(app.sim), stepped)
        assert len(brain["inputs"]) == BRAIN_INPUTS
    finally:
        app.sim, app.paused = previous_sim, previous_pause
