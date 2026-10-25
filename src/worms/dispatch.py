"""Retained SlangPy bindings for repeatedly dispatching the same compute kernel."""

import slangpy as spy


class BoundDispatch:
    """Keep a root shader object and only write scalars that change between calls.

    D3D12 captures bindings at dispatch recording time, so later scalar writes do
    not change previously recorded commands. The normal compute pass retains
    SlangPy's resource tracking and barriers.
    """

    def __init__(self, device, kernel, thread_count, bindings, dynamic=()):
        self.device = device
        self.thread_count = spy.uint3(*thread_count)
        self.bindings = bindings
        self.dynamic = dynamic
        self.replace_kernel(kernel)

    def __getattr__(self, name):
        return getattr(self.kernel, name)

    def replace_kernel(self, kernel):
        """Reuse a warmed pipeline while binding this instance's current buffers."""
        self.kernel = kernel
        self._pipeline = kernel.pipeline
        self.root = self.device.create_root_shader_object(kernel.program)
        cursor = spy.ShaderCursor(self.root)
        cursor.write(self.bindings)
        self._cursors = {name: cursor[name] for name in self.dynamic}
        self._values = {}

    def dispatch(self, *, command_encoder, **uniforms):
        for name, value in uniforms.items():
            if self._values.get(name) != value:
                self._cursors[name].write(value)
                self._values[name] = value
        with command_encoder.begin_compute_pass() as compute:
            compute.bind_pipeline(self._pipeline, self.root)
            compute.dispatch(self.thread_count)
