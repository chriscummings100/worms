# Worms

A small Python + SlangPy artificial-life project. The first application runs independent GPU worlds with moving creatures, a windowed viewer, and a headless path for tests and image dumps.

## Run

From PowerShell in the repository root:

```powershell
.\.conda\python.exe -m pip install -e ".[dev]"

# Open the application (4 worlds, 64 creatures per world).
.\.conda\python.exe -m worms

# Advance exactly 120 fixed ticks and save the final scene.
.\.conda\python.exe -m worms --headless --frames 120 --output artifacts/worlds.png

# Change the initial world and creature counts.
.\.conda\python.exe -m worms --worlds 6 --creatures 128
```

The window has Pause, Step, Reset, and an all-worlds/single-world selector. Escape or the close button exits. `--frames N` also works in windowed mode for a finite run. PNG exports contain the scene without UI. A headless run defaults to 120 frames and `artifacts/worlds.png`.

The application currently targets D3D12 and selects the NVIDIA adapter. It uses the existing Python 3.12 environment and SlangPy 0.43.1; no extra UI dependency is needed. For a fresh environment, use `conda create --prefix .conda python=3.12 pip`, then install `requirements-dev.lock` and the editable project. A Python 3.12 venv also works; substitute its interpreter in the commands.

## Structure

- `src/worms/app.py`: device, window/UI, frame loop, tiled/focused views, PNG output.
- `src/worms/sim.py`: simulation buffers, compute dispatches, and drawing a world into a rectangle, with optional camera offset and scale.
- `src/worms/shaders/world.slang`: creature/world records, seeded initialization, and motion.
- `src/worms/shaders/world_render.slang`: pond backgrounds and instanced creature rendering.
- `tests/`: layout checks and opt-in tests using the actual GPU.
- `docs/` and `assets/`: planning documents and original visual references.
- `artifacts/`: generated outputs, ignored by Git.

Each world has a fixed block of creatures in one shared GPU buffer. The world buffer records its live count. All creatures currently remain alive, so those blocks are already compact. IDs are unique across the active worlds; seeded reset reproduces the same initial state. The action/target fields are placeholders for later simulation work.

One update moves each creature at its initialized velocity for 1/60 second and wraps both coordinates in a 256 x 256 world. Rendering reads the same GPU state, including copies at wrapping boundaries. The window advances one tick per displayed frame; wall-clock pacing independent of display refresh is future work. Headless runs execute the requested number of ticks directly. Pause freezes the simulation, Step advances once and remains paused, and Reset restores the seed without an extra tick that frame.

This is a motion and application foundation. Food, energy, births, death, interaction, evolution, and learning are still future work in the [project plan](docs/PROJECT_PLAN.md). `baseline.proposed.toml` remains planning data, not a supported application configuration. The earlier standalone smoke renderer has been replaced by this shared application path.

## Checks

```powershell
.\.conda\python.exe -m pytest
.\.conda\python.exe -m pytest --run-gpu
.\.conda\python.exe -m ruff check .
.\.conda\python.exe -m ruff format --check src tests
```

GPU checks cover separate seeded worlds, unique IDs, persistent state, wrapping across both axes, reset, pause/step controls, and grid/focused PNGs. Explicit GPU runs fail normally if graphics is unavailable. Both headless and finite windowed runs have been verified on the RTX 3050 Ti Laptop GPU.

The installed runtime emits `Cannot enable D3D12 Agility SDK: Calling SetSDKVersion failed.` at startup. Device creation, shaders, rendering, and GPU tests still pass with native debug layers and RHI validation enabled; the warning is not suppressed.

See [AGENTS.md](AGENTS.md) for development conventions and [visual references](docs/VISUALS.md) for the intended appearance. Window and UI management use SlangPy's [built-in application helper](https://github.com/shader-slang/slangpy/blob/v0.43.1/src/sgl/app/app.cpp).
