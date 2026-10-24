"""Controlled policies for GPU tests of geometry, contacts and observation."""

import math

import numpy as np

from worms.sim import BRAIN_PARAMETERS


def fixed_brains(sim, *, steering=0.0, movement=1.0):
    """Replace weights with a constant policy, then evaluate it before movement."""
    count = sim.num_worlds * sim.creature_capacity
    weights = np.zeros((count, BRAIN_PARAMETERS), dtype=np.float32)
    weights[:, 1072] = math.atanh(steering) if abs(steering) < 1 else math.copysign(20, steering)
    weights[:, 1073] = movement - 0.5
    sim.brain_weights.copy_from_numpy(weights)
    encoder = sim.device.create_command_encoder()
    sim._update_brains(encoder)
    sim.device.submit_command_buffer(encoder.finish())


def rebuild_food_grid(sim):
    """Refresh the spatial cache after a test directly relocates food records."""
    encoder = sim.device.create_command_encoder()
    sim._update_food_grid(encoder)
    sim.device.submit_command_buffer(encoder.finish())
