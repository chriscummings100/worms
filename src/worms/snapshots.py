"""Versioned checkpoints with explicit GPU readback and atomic file replacement.

Call these after submitting the current frame, never while its encoder still holds
simulation work. Checkpoints include the full vision ring and spatial grids, so
restoring does not reset observations or change the next simulation step. Bump the
format version when changing the meaning or layout of saved GPU records.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from worms.sim import Sim

FORMAT_VERSION = 1
SNAPSHOT_INTERVAL = 10_000
SNAPSHOT_KEEP = 5
SIM_BUFFERS = (
    "worlds",
    "creatures",
    "food",
    "grid_heads",
    "grid_next",
    "food_credits",
    "food_grid_heads",
    "food_grid_next",
    "vision_frames",
    "vision_history",
    "brain_weights",
    "brain_activations",
)
SIM_SETTINGS = (
    "num_worlds",
    "creatures_per_world",
    "seed",
    "creature_capacity",
    "food_capacity",
    "initial_food",
    "target_food",
    "food_per_second",
    "food_energy",
    "food_age_start",
    "food_age_fade",
    "mutation_enabled",
)


@dataclass
class Snapshot:
    sim: Sim
    extra_sims: dict
    metadata: dict
    arrays: dict


def _write_archive(path, payload):
    """Leave an existing checkpoint intact if writing its replacement fails."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as file:
            temporary = Path(file.name)
            np.savez_compressed(file, **payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def save_snapshot(path, sim, *, extra_sims=None, metadata=None, arrays=None):
    """Save the live simulation and optional benchmark state to one portable NPZ.

    GPU buffers are read back synchronously. Additional state must be JSON or
    numeric NumPy arrays; object arrays and pickle are never used. Reproducible
    continuation is tested on the same GPU/driver, not across different devices.
    """
    extra_sims = extra_sims or {}
    metadata = metadata or {}
    arrays = arrays or {}
    manifest = {"version": FORMAT_VERSION, "metadata": metadata, "sims": [], "arrays": {}}
    json.dumps(metadata)  # Reject unsupported state before waiting for the GPU.
    payload = {}
    for index, (name, value) in enumerate(arrays.items()):
        value = np.asarray(value)
        if value.dtype.hasobject:
            raise ValueError("Snapshot arrays cannot contain Python objects")
        key = f"array_{index}"
        payload[key] = value
        manifest["arrays"][name] = key

    sim.device.wait()
    for index, (name, current) in enumerate(((None, sim), *extra_sims.items())):
        entry = {
            "name": name,
            "settings": {field: getattr(current, field) for field in SIM_SETTINGS},
            "tick": current.tick,
            "food_energy_dirty": current._food_energy_dirty,
            "buffers": {},
        }
        for field in SIM_BUFFERS:
            key = f"sim_{index}_{field}"
            payload[key] = getattr(current, field).to_numpy().view(np.uint8).reshape(-1).copy()
            entry["buffers"][field] = key
        manifest["sims"].append(entry)
    payload["manifest"] = np.array(json.dumps(manifest))
    return _write_archive(path, payload)


def load_snapshot(path, device):
    """Create restored simulations on the supplied device without advancing them."""
    simulations = []
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"]))
        if manifest["version"] != FORMAT_VERSION:
            raise ValueError(f"Unsupported snapshot version: {manifest['version']}")
        entries = manifest["sims"]
        if not entries or entries[0]["name"] is not None:
            raise ValueError("Snapshot has no live simulation")
        for entry in entries:
            current = Sim(device, **entry["settings"])
            # Validate the complete layout before uploading any checkpoint bytes.
            saved = {}
            for field in SIM_BUFFERS:
                value = archive[entry["buffers"][field]]
                if value.dtype != np.uint8 or value.ndim != 1:
                    raise ValueError(f"Invalid snapshot buffer: {field}")
                if value.nbytes != getattr(current, field).size:
                    raise ValueError(f"Snapshot buffer layout no longer matches: {field}")
                saved[field] = value
            device.wait()  # Finish constructor initialization before replacing buffers.
            for field, value in saved.items():
                getattr(current, field).copy_from_numpy(value)
            current.tick = entry["tick"]
            current._food_energy_dirty = entry["food_energy_dirty"]
            simulations.append(current)
        arrays = {name: archive[key].copy() for name, key in manifest["arrays"].items()}
    return Snapshot(
        sim=simulations[0],
        extra_sims={entry["name"]: current for entry, current in zip(entries[1:], simulations[1:])},
        metadata=manifest["metadata"],
        arrays=arrays,
    )


class PeriodicSnapshots:
    """Save at tick intervals and retain only this store's latest automatic files."""

    def __init__(self, directory, interval=SNAPSHOT_INTERVAL, keep=SNAPSHOT_KEEP):
        if interval < 1 or keep < 1:
            raise ValueError("Snapshot interval and retention must be positive")
        self.directory = Path(directory)
        self.interval = interval
        self.keep = keep
        self.last_tick = 0

    @property
    def latest(self):
        return max(self.directory.glob("auto-*.npz"), default=None)

    def reset(self, tick=0):
        """Restart the schedule after resetting a run or restoring a checkpoint."""
        self.last_tick = tick

    def due(self, tick):
        return tick // self.interval > self.last_tick // self.interval

    def save(self, sim, **state):
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.directory / f"auto-{timestamp}-tick-{sim.tick:012d}.npz"
        save_snapshot(path, sim, **state)
        self.last_tick = sim.tick
        for old in sorted(self.directory.glob("auto-*.npz"))[: -self.keep]:
            old.unlink()
        return path
