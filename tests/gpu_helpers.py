"""Controlled policies for GPU tests of geometry, contacts and observation."""

import math

import numpy as np

from worms.sim import BRAIN_PARAMETERS, GRID_SIZE

GRID_BUFFERS = {"grid_heads", "grid_next", "food_grid_heads", "food_grid_next"}


def grid_membership(heads, links):
    """Read unordered cells, rejecting cycles, duplicate entries and world leakage."""
    heads = heads.view(np.int32).reshape(-1, GRID_SIZE**2)
    links = links.view(np.int32).reshape(-1)
    capacity = len(links) // len(heads)
    result = []
    for world, world_heads in enumerate(heads):
        cells = {}
        seen = set()
        for cell in np.flatnonzero(world_heads >= 0):
            members = set()
            index = int(world_heads[cell])
            while index != -1:
                assert world * capacity <= index < (world + 1) * capacity
                assert index not in seen, "Duplicate grid entry or linked-list cycle"
                seen.add(index)
                members.add(index)
                index = int(links[index])
            cells[(int(cell) % GRID_SIZE, int(cell) // GRID_SIZE)] = members
        result.append(cells)
    return result


def assert_sim_buffers_equal(actual, expected):
    """Compare continued GPU state; atomic grid insertion order is unspecified.

    Use byte-exact comparisons instead for an immediate checkpoint round trip.
    All simulation, vision and neural buffers remain exact here too.
    """
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        if name not in GRID_BUFFERS:
            np.testing.assert_array_equal(actual[name], value, err_msg=name)
    for prefix in ("grid", "food_grid"):
        heads, links = f"{prefix}_heads", f"{prefix}_next"
        assert grid_membership(actual[heads], actual[links]) == grid_membership(
            expected[heads], expected[links]
        ), prefix


def fixed_brains(sim, *, steering=0.0, movement=1.0):
    """Replace weights with a constant policy, then evaluate it before movement."""
    count = sim.num_worlds * sim.creature_capacity
    weights = np.zeros((count, BRAIN_PARAMETERS), dtype=np.float16)
    weights[:, 1072] = math.atanh(steering) if abs(steering) < 1 else math.copysign(20, steering)
    weights[:, 1073] = movement - 0.5
    sim.brain_weights.copy_from_numpy(weights)
    encoder = sim.device.create_command_encoder()
    sim._update_brains(encoder)
    sim.device.submit_command_buffer(encoder.finish())


def rebuild_food_grid(sim):
    """Refresh the grid after direct edits when capturing without a full tick."""
    encoder = sim.device.create_command_encoder()
    sim._update_food_grid(encoder)
    sim.device.submit_command_buffer(encoder.finish())


def legacy_neural_buffers(payload, manifest):
    """Build an authentic pre-v5 FP32 neural layout from a modern test archive."""
    keys = [
        entry["buffers"][field]
        for entry in manifest["sims"]
        for field in ("brain_weights", "brain_activations")
    ]
    if "reference_weights" in manifest["arrays"]:
        keys.append(manifest["arrays"]["reference_weights"])
    for key in keys:
        payload[key] = payload[key].view(np.float16).astype(np.float32).view(np.uint8)
