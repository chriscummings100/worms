# Worms

A Python + SlangPy project exploring a small artificial ecosystem: simple creatures, glowing food, and eventually evolution and learning. The starting deliverable is a GPU rendering smoke test. The ecosystem described in the [project plan](docs/PROJECT_PLAN.md) remains future work.

## Setup

From PowerShell in this directory, use the local Python environment at `.conda`:

```powershell
.\.conda\python.exe -m pip install -r requirements-dev.lock
.\.conda\python.exe -m pip install -e ".[dev]"
.\.conda\python.exe -m pytest
.\.conda\python.exe -m worms.render --output artifacts/smoke.png
.\.conda\python.exe -m pytest --run-gpu
.\.conda\python.exe -m ruff check .
```

The existing `.conda` environment uses Python 3.12. For a fresh checkout, create it with `conda create --prefix .conda python=3.12 pip`. Alternatively, run `py -3.12 -m venv .venv` and use `.\.venv\Scripts\python.exe` in place of `.\.conda\python.exe` in every command above. Activation is optional.

The editable install uses `pyproject.toml`; `requirements-dev.lock` records the exact tested Windows/Python 3.12 package versions. SlangPy is pinned to 0.43.1. NumPy and Pillow support image handling, and pytest and Ruff provide development checks. CUDA and PyTorch are not required for this milestone. Default tests run without a GPU; `--run-gpu` additionally runs hardware checks and reports failures if the graphics path is unavailable.

## Rendering proof

The renderer explicitly selects the NVIDIA adapter on D3D12 by default, with GPU debug layers and RHI validation enabled. A Slang compute dispatch creates a structured creature buffer, and a six-vertex instanced graphics draw reads that same buffer without copying it through Python. The final RGBA image and GPU timestamp queries are read back for the PNG and adjacent JSON run manifest. Shader files are packaged with the Python module.

```powershell
# Scale the same pipeline to 256 creatures.
.\.conda\python.exe -m worms.render --count 256 --output artifacts/smoke-256.png

# A reproducible alternate pose; this is not a simulation timestep.
.\.conda\python.exe -m worms.render --seed 42 --time 2 --output artifacts/pose.png

# Explicit alternative backend or adapter (no silent fallback).
.\.conda\python.exe -m worms.render --backend vulkan --adapter NVIDIA
```

Use `--width`, `--height`, and `--count` for scene size, and `--adapter` for a unique device name fragment. `--no-debug` is available for machines without native debug layers. The Vulkan option must be verified on its intended machine; the initial acceptance run used D3D12.

Verified on 24 October 2026 with Python 3.12.15, SlangPy 0.43.1, Slang compiler 2026.12, and NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GiB, driver 596.08): five- and 256-creature renders, 27 CPU tests plus 2 GPU tests, Ruff, and dependency consistency. GPU tests also check odd image dimensions, nonuniform opaque pixels, and saved manifests. Timings are single cold GPU frames, excluding shader compilation, image readback and PNG encoding; they are not a steady-state benchmark.

This machine emits `Cannot enable D3D12 Agility SDK: Calling SetSDKVersion failed.` at SlangPy startup. D3D12 device creation, shader compilation and all rendering tests still pass using the available runtime, with debug layers and RHI validation enabled. No graphics validation errors were reported. The warning is recorded here rather than suppressed.

This is the first rendering proof; it does not establish a running ecosystem, learning behavior, or the full stage-0 performance and correctness gates. Generated output is ignored by Git. The implementation follows the official [SlangPy installation guide](https://slangpy.shader-slang.org/en/latest/), [compute tutorial](https://slangpy.shader-slang.org/en/latest/src/tutorials/compute_shader.html), and [0.43.1 graphics example](https://github.com/shader-slang/slangpy/blob/v0.43.1/examples/render_pipeline/render_pipeline.py).

## Layout

```text
src/worms/              Python package and Slang shaders
tests/                  CPU tests and opt-in GPU checks
docs/                   Project plan and visual documentation
assets/                 Reference images and diagrams
artifacts/              Generated renders and run reports (ignored)
baseline.proposed.toml  Proposed settings; not application configuration
download.svg            Original supplied SVG, preserved unchanged
```

See [AGENTS.md](AGENTS.md) for development conventions, the [original planning overview](docs/OVERVIEW.md) for context, and [visual references and provenance](docs/VISUALS.md) for the intended direction.
