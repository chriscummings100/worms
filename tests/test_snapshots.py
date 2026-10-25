"""Check checkpoint file safety without a GPU."""

import json

import numpy as np
import pytest

from worms.snapshots import _neural_buffer, _write_archive, load_snapshot


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


@pytest.mark.parametrize("version", [1, 2, 3, 4])
@pytest.mark.parametrize("field", ["brain_weights", "brain_activations", "reference_weights"])
def test_legacy_neural_storage_quantizes_without_changing_source(version, field):
    original = np.array([-0.82397646, -0.000071, 0, 0.64057761, 0.9998], dtype=np.float32)
    source = original.view(np.uint8).copy()
    migrated = _neural_buffer(source, original.size * 2, field, version)
    np.testing.assert_array_equal(migrated.view(np.float16), original.astype(np.float16))
    np.testing.assert_array_equal(source, original.view(np.uint8))
    assert migrated.nbytes == source.nbytes // 2
    np.testing.assert_allclose(migrated.view(np.float16), original, rtol=0.0005, atol=3e-8)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 65520.0, -65520.0])
def test_legacy_neural_storage_rejects_values_that_cannot_be_stored_finitely(value):
    source = np.array([value], dtype=np.float32).view(np.uint8)
    with pytest.raises(ValueError, match="reference_weights.*finite FP16 range"):
        _neural_buffer(source, 2, "reference_weights", 4)


def test_neural_storage_checks_layout_and_preserves_new_half_bytes():
    source = np.array([0.1, -0.3, 1], dtype=np.float16).view(np.uint8)
    assert _neural_buffer(source, source.nbytes, "brain_weights", 5) is source
    with pytest.raises(ValueError, match="layout no longer matches: brain_weights"):
        _neural_buffer(source[:-1], source.nbytes, "brain_weights", 5)
    with pytest.raises(ValueError, match="Invalid snapshot buffer: brain_weights"):
        _neural_buffer(source.reshape(2, 3), source.nbytes, "brain_weights", 5)
