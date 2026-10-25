# Saved evolved population — 13,000,000 ticks

This milestone checkpoint was captured on 25 October 2026 and is intentionally
versioned in Git. Automatic checkpoints remain under ignored `artifacts/snapshots/`;
their five-file retention policy cannot remove this copy.

- `snapshot.npz`: complete live world, frozen reference from tick 200,000,
  in-progress benchmark, 127 completed comparison results and saved settings.
- Latest completed benchmark: **90.9917% of collected food energy**, sampled at
  live tick 12,800,445 and completed at tick 12,900,445. The current population
  collected 152,725 energy units and the baseline collected 15,120.
- `source.zip`: source, shaders, tests and documentation as they existed when the
  checkpoint was backed up, including the then-uncommitted changes. Preserve this
  archive for future code compatibility.
- `runtime-versions.json`: Python/platform and installed package versions.
- `backup-info.json`: original local paths, checksums and saved-state metadata.
  Its Git commit is the base revision at capture; the source archive also includes
  the changes that had not yet been committed.

From the repository root, restore with:

```powershell
.\.conda\python.exe -m worms --resume checkpoints/13-million-ticks/snapshot.npz
```

The app restores paused. See the root README for environment setup. If future code
no longer supports this checkpoint, extract `source.zip` into a separate directory,
recreate a Python 3.12 environment using `requirements-dev.lock`, install with
`python -m pip install -e .`, and run
`python -m worms --resume <absolute path to snapshot.npz>` there. This version
requires Windows, D3D12 and an NVIDIA adapter.

The snapshot and source archive passed ZIP CRC checks and match the SHA-256 hashes
recorded in `backup-info.json`. The original local backup remains at
`artifacts/saved-runs/20261025T135417Z-000013000000/`.
