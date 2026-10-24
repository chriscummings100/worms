"""Check checkpoint file safety without a GPU."""

import json

import numpy as np
import pytest

from worms.snapshots import _write_archive, load_snapshot


def test_atomic_archive_keeps_previous_checkpoint_if_writing_fails(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.npz"
    _write_archive(path, {"value": np.arange(8)})
    before = path.read_bytes()

    def fail_after_partial_write(file, **payload):
        file.write(b"incomplete checkpoint")
        raise OSError("Disk write failed")

    monkeypatch.setattr(np, "savez_compressed", fail_after_partial_write)
    with pytest.raises(OSError, match="Disk write failed"):
        _write_archive(path, {"value": np.arange(16)})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_unknown_version_is_rejected_before_creating_gpu_resources(tmp_path):
    path = tmp_path / "future.npz"
    np.savez(path, manifest=np.array(json.dumps({"version": -1})))
    with pytest.raises(ValueError, match="Unsupported snapshot version"):
        load_snapshot(path, device=None)
