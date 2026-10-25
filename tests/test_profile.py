"""Profiler command-line guards must protect its checkpoint before creating a GPU app."""

import pytest

from worms.profile import main


def test_profile_refuses_to_overwrite_source_checkpoint(tmp_path, capsys):
    checkpoint = tmp_path / "source.npz"
    contents = b"checkpoint must remain unchanged"
    checkpoint.write_bytes(contents)
    with pytest.raises(SystemExit) as stopped:
        main(["--resume", str(checkpoint), "--output", str(checkpoint)])
    assert stopped.value.code == 2
    assert "--output must differ from the source checkpoint" in capsys.readouterr().err
    assert checkpoint.read_bytes() == contents
