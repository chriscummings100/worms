"""Render a small GPU scene without a window: compute -> instanced draw -> PNG."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


def _runtime_versions(package_dir: Path, adapter_name: str) -> dict:
    """Best-effort diagnostics; unsupported version queries are recorded as null."""
    compiler_version = None
    library_name = (
        "slang-compiler.dll" if platform.system() == "Windows" else "libslang-compiler.so"
    )
    try:
        library = ctypes.CDLL(str(package_dir / library_name))
        # Official Slang C API; no compiler session or C++ object access needed.
        library.spGetBuildTagString.argtypes = []
        library.spGetBuildTagString.restype = ctypes.c_char_p
        compiler_version = library.spGetBuildTagString().decode("utf-8")
    except (OSError, AttributeError):
        pass

    driver_version = None
    if "nvidia" in adapter_name.casefold():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                name, version = line.rsplit(",", 1)
                if name.strip() == adapter_name:
                    driver_version = version.strip()
                    break
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return {"slang_compiler": compiler_version, "nvidia_driver": driver_version}


@dataclass(frozen=True)
class RenderConfig:
    """Settings for the render proof, independent of the proposed simulator config."""

    width: int = 1280
    height: int = 720
    count: int = 5
    seed: int = 1
    time_seconds: float = 0.0
    backend: str = "d3d12"
    adapter: str = "NVIDIA"
    debug: bool = True

    def __post_init__(self) -> None:
        for name, maximum in (("width", 8192), ("height", 8192), ("count", 4096)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer from 1 to {maximum}")
        if type(self.seed) is not int or not 0 <= self.seed <= 0xFFFFFFFF:
            raise ValueError("seed must be an integer from 0 to 4294967295")
        if isinstance(self.time_seconds, bool) or not isinstance(self.time_seconds, (int, float)):
            raise ValueError("time_seconds must be a finite number")
        if not math.isfinite(self.time_seconds):
            raise ValueError("time_seconds must be finite")
        if self.backend not in ("d3d12", "vulkan"):
            raise ValueError("backend must be d3d12 or vulkan")
        if not isinstance(self.adapter, str) or not self.adapter.strip():
            raise ValueError("adapter must be a nonblank name or name fragment")


def render_smoke(output: Path, config: RenderConfig = RenderConfig()) -> dict:
    """Save an actual GPU render and JSON run manifest, or raise on any GPU failure.

    The creature buffer stays on the GPU from dispatch through draw. Only the final
    image and timestamp queries are read back. This one-shot path deliberately
    waits for completion; its cold-frame timings are not a throughput benchmark.
    """
    output = Path(output)
    if output.suffix.lower() != ".png":
        raise ValueError("output must have a .png extension")

    # Keep CLI help, configuration, and CPU tests usable without a graphics driver.
    import slangpy as spy
    from PIL import Image

    device_type = getattr(spy.DeviceType, config.backend)
    adapters = spy.Device.enumerate_adapters(device_type)
    matches = [a for a in adapters if config.adapter.strip().casefold() in a.name.casefold()]
    if len(matches) != 1:
        names = ", ".join(a.name for a in adapters) or "none"
        raise RuntimeError(
            f"Expected one {config.backend} adapter matching {config.adapter!r}; "
            f"found {len(matches)}. Available: {names}"
        )
    adapter = matches[0]
    shader_dir = Path(__file__).parent / "shaders"
    device = spy.Device(
        type=device_type,
        adapter_luid=adapter.luid,
        enable_debug_layers=config.debug,
        enable_rhi_validation=True,
        compiler_options={"include_paths": [shader_dir]},
    )
    try:
        compute_program = device.load_program("smoke.slang", ["initialize"])
        kernel = device.create_compute_kernel(program=compute_program)
        creatures = device.create_buffer(
            element_count=config.count,
            resource_type_layout=kernel.reflection.initialize.output_creatures,
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
            label="creatures",
        )
        texture = device.create_texture(
            format=spy.Format.rgba8_unorm,
            width=config.width,
            height=config.height,
            usage=spy.TextureUsage.render_target | spy.TextureUsage.copy_source,
            label="smoke-render",
        )
        graphics_program = device.load_program("smoke.slang", ["vertex_main", "fragment_main"])
        pipeline = device.create_render_pipeline(
            program=graphics_program,
            input_layout=None,
            targets=[{"format": spy.Format.rgba8_unorm}],
            rasterizer={"cull_mode": spy.CullMode.none},
        )
        queries = device.create_query_pool(type=spy.QueryType.timestamp, count=3)
        encoder = device.create_command_encoder()
        encoder.write_timestamp(queries, 0)
        kernel.dispatch(
            thread_count=[config.count, 1, 1],
            command_encoder=encoder,
            count=config.count,
            seed=config.seed,
            time_seconds=config.time_seconds,
            resolution=spy.float2(config.width, config.height),
            output_creatures=creatures,
        )
        encoder.write_timestamp(queries, 1)
        # Transition the compute output for vertex shader reads, with no CPU copy.
        encoder.set_buffer_state(creatures, spy.ResourceState.shader_resource)
        with encoder.begin_render_pass(
            {
                "color_attachments": [
                    {
                        "view": texture.create_view({}),
                        "load_op": spy.LoadOp.clear,
                        "store_op": spy.StoreOp.store,
                        "clear_value": [0.015, 0.045, 0.055, 1.0],
                    }
                ]
            }
        ) as render_pass:
            cursor = spy.ShaderCursor(render_pass.bind_pipeline(pipeline))
            cursor["creatures"] = creatures
            cursor["resolution"] = spy.float2(config.width, config.height)
            render_pass.set_render_state(
                {
                    "viewports": [spy.Viewport.from_size(config.width, config.height)],
                    "scissor_rects": [spy.ScissorRect.from_size(config.width, config.height)],
                }
            )
            render_pass.draw({"vertex_count": 6, "instance_count": config.count})
        encoder.write_timestamp(queries, 2)
        device.submit_command_buffer(encoder.finish())
        device.wait()
        timestamps = queries.get_timestamp_results(0, 3)
        pixels = texture.to_numpy()
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(output)

        manifest = {
            **asdict(config),
            "created_utc": datetime.now(UTC).isoformat(),
            "output": str(output.resolve()),
            "adapter": device.info.adapter_name,
            "adapter_vendor_id": adapter.vendor_id,
            "adapter_device_id": adapter.device_id,
            "graphics_api": device.info.api_name,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "slangpy": spy.__version__,
            "sgl": spy.SGL_VERSION,
            **_runtime_versions(Path(spy.__file__).parent, device.info.adapter_name),
            "debug_layers": config.debug,
            "rhi_validation": True,
            "compute_ms": (timestamps[1] - timestamps[0]) * 1000,
            "draw_ms": (timestamps[2] - timestamps[1]) * 1000,
            "timing_note": "Cold GPU frame; excludes compilation, readback and PNG encoding.",
        }
        output.with_suffix(".json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest
    finally:
        device.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke.png"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--time", dest="time_seconds", type=float, default=0.0)
    parser.add_argument("--backend", choices=("d3d12", "vulkan"), default="d3d12")
    parser.add_argument("--adapter", default="NVIDIA", help="Unique adapter name fragment")
    parser.add_argument("--no-debug", action="store_true", help="Disable native GPU debug layers")
    args = parser.parse_args(argv)
    try:
        config = RenderConfig(
            width=args.width,
            height=args.height,
            count=args.count,
            seed=args.seed,
            time_seconds=args.time_seconds,
            backend=args.backend,
            adapter=args.adapter,
            debug=not args.no_debug,
        )
        manifest = render_smoke(args.output, config)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Render failed: {error}\n")
    print(f"Rendered {config.count} creatures on {manifest['adapter']} ({config.backend}).")
    print(f"GPU compute: {manifest['compute_ms']:.3f} ms; draw: {manifest['draw_ms']:.3f} ms")
    print(f"PNG: {args.output.resolve()}")
    print(f"Run manifest: {args.output.with_suffix('.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
