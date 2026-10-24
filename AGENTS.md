# Working on Worms

Worms is a small artificial-life project built with Python and SlangPy/Slang. Read `docs/PROJECT_PLAN.md` for the staged design and `docs/VISUALS.md` for the visual references. The plan describes future work, not features already implemented.

- Keep Python code and packaged Slang shaders in `src/worms/`, tests in `tests/`, planning and development notes in `docs/`, and source art in `assets/`. Generated renders and run reports belong in ignored `artifacts/`.
- Use the project environment: `.\.conda\python.exe -m pip install -e ".[dev]"`. Run commands from the repository root; do not depend on a globally activated Python environment.
- Run CPU tests with `.\.conda\python.exe -m pytest`. Mark hardware tests `@pytest.mark.gpu`; run them explicitly with `.\.conda\python.exe -m pytest --run-gpu`. GPU tests must report actual failures when requested rather than silently substitute CPU rendering.
- Run the window with `.\.conda\python.exe -m worms`; check the same render path headlessly with `.\.conda\python.exe -m worms --headless --frames 120 --output artifacts/worlds.png`. The app currently targets D3D12 on the NVIDIA adapter.
- Keep the current application simple: App owns the device, window/UI and frame loop; Sim owns the GPU worlds and creatures. Multiple independent worlds are in scope. Defer ecology, evolution and learning until requested. Add configuration and error handling only when needed; fail-fast GPU errors are fine.
- Keep simulation and draw data on the GPU between update and rendering. Read back only explicit outputs, diagnostics, or checkpoints; make synchronization and ownership clear.
- Use explicit seeds for reproducible scenes and tests. Keep deterministic CPU checks small; do not promise bit-identical GPU results across drivers or hardware.
- Preserve supplied assets, including root `download.svg`. Keep relative documentation links valid when moving files. Treat `baseline.proposed.toml` as design hypotheses until the application supports it.
