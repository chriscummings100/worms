"""Retained shader bindings preserve recorded values and buffer ownership."""

import numpy as np
import pytest
import slangpy as spy

from worms.dispatch import BoundDispatch

pytestmark = pytest.mark.gpu


def test_retained_roots_capture_scalars_across_encoders_and_rebind_restored_buffers():
    adapter = next(
        a for a in spy.Device.enumerate_adapters(spy.DeviceType.d3d12) if "NVIDIA" in a.name
    )
    device = spy.Device(
        type=spy.DeviceType.d3d12, adapter_luid=adapter.luid, enable_debug_layers=True
    )
    module = device.load_module_from_source(
        "retained_dispatch_test",
        """
        RWStructuredBuffer<uint> output;
        uint tick;
        uint slot;
        [shader("compute")]
        [numthreads(1, 1, 1)]
        void stamp(uint3 id : SV_DispatchThreadID)
        {
            output[slot] = tick;
            output[0] += tick;
        }
        """,
    )
    kernel = device.create_compute_kernel(
        device.link_program([module], [module.entry_point("stamp")])
    )
    buffers = [
        device.create_buffer(
            element_count=4,
            struct_size=4,
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
            data=np.zeros(4, dtype=np.uint32),
        )
        for _ in range(2)
    ]
    dispatch = BoundDispatch(
        device, kernel, [1, 1, 1], {"output": buffers[0]}, dynamic=("tick", "slot")
    )
    original_root = dispatch.root
    first = device.create_command_encoder()
    dispatch.dispatch(command_encoder=first, tick=3, slot=1)
    dispatch.dispatch(command_encoder=first, tick=9, slot=2)
    second = device.create_command_encoder()
    dispatch.dispatch(command_encoder=second, tick=9, slot=3)  # Same tick skips scalar rewrite.
    assert dispatch.root is original_root

    # A restored simulation keeps its own buffers while borrowing warmed kernels.
    restored = BoundDispatch(
        device, kernel, [1, 1, 1], {"output": buffers[1]}, dynamic=("tick", "slot")
    )
    old_restored_root = restored.root
    restored.replace_kernel(dispatch.kernel)
    assert restored.root is not old_restored_root
    assert restored.kernel is kernel
    restored.dispatch(command_encoder=second, tick=11, slot=1)

    # Both encoders were recorded before submission, so no wait can hide a
    # retroactive constant update or incorrect retained-resource dependency.
    device.submit_command_buffer(first.finish())
    device.submit_command_buffer(second.finish())
    device.wait()
    assert buffers[0].to_numpy().view(np.uint32).tolist() == [21, 3, 9, 9]
    assert buffers[1].to_numpy().view(np.uint32).tolist() == [11, 11, 0, 0]
