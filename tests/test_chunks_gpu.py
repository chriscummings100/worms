"""Real queue-order and checkpoint checks for simulation chunk submission."""

import numpy as np
import pytest
import slangpy as spy

from tests.test_session_gpu import assert_state, state
from worms.app import App
from worms.benchmark import Benchmark
from worms.dispatch import BoundDispatch
from worms.sim import Sim
from worms.submissions import BoundedSubmissions, chunk_timings

pytestmark = pytest.mark.gpu


@pytest.fixture
def app():
    application = App(
        headless=True,
        debug_gpu=True,
        creatures_per_world=2,
        creature_capacity=8,
        benchmark=False,
        snapshot_dir=None,
        width=129,
        height=73,
    )
    application.sim = Sim(
        application.device,
        num_worlds=1,
        creatures_per_world=2,
        creature_capacity=8,
        seed=1203,
        food_capacity=64,
        initial_food=16,
        target_food=32,
    )
    yield application
    application.close()


def direct_frame(app):
    encoder = app.device.create_command_encoder()
    app.update(encoder)
    app.render(encoder)
    submission = app.device.submit_command_buffer(encoder.finish())
    app.device.wait_for_submit(submission)


@pytest.mark.parametrize("warmup_ticks", [0, 50])
def test_chunked_frames_match_direct_recording_at_trial_and_snapshot_boundaries(
    app, tmp_path, warmup_ticks
):
    app.benchmark = Benchmark(app.sim, warmup_ticks=warmup_ticks, trial_ticks=37)
    app.speed = 103  # Exercise a partial last chunk as well as the regular chunks.
    app.sim.food_energy = 3.25
    initial = app.snapshot(tmp_path / "initial.npz")

    direct_frame(app)
    first_expected = state(app)
    direct_frame(app)
    second_expected = state(app)
    assert app.sim.tick == 206
    assert app.benchmark.sim.tick == 37 and app.benchmark.pending
    assert len(app.benchmark.history) == int(warmup_ticks == 0)

    app.restore(initial)
    app.paused = False
    app.frame()
    assert_state(state(app), first_expected, continued=True)
    # A checkpoint at the submitted chunk boundary must restore every live,
    # trial and retained reference buffer, including the pending trial flag.
    checkpoint = app.snapshot(tmp_path / "chunked.npz")
    checkpoint_state = state(app)
    app.restore(checkpoint)
    assert_state(state(app), checkpoint_state)
    app.paused = False
    app.frame()
    assert_state(state(app), second_expected, continued=True)


def test_fast_chunked_pause_step_reset_and_settings_keep_existing_semantics(app):
    initial = app.sim.creatures.to_numpy().copy()
    app.speed = 1000
    app.paused = True
    app.sim.food_energy = 4.25
    app.frame()
    assert app.sim.tick == 0
    np.testing.assert_array_equal(app.sim.creatures.to_numpy(), initial)
    app.step()
    app.frame()
    assert app.sim.tick == 1 and app.paused
    app._reset = True
    app._step = True
    app.frame()
    assert app.sim.tick == 0 and app.paused
    np.testing.assert_array_equal(app.sim.creatures.to_numpy(), initial)
    app.paused = False
    app.frame()
    assert app.sim.tick == 1000
    assert app.sim.food_energy == 4.25


def test_retained_shader_scalars_survive_submitting_while_recording_next_chunk(app):
    device = app.device
    module = device.load_module_from_source(
        "chunk_scalar_lifetime",
        """
        RWStructuredBuffer<uint> output;
        uint tick;
        [shader("compute")]
        [numthreads(1, 1, 1)]
        void stamp(uint3 id : SV_DispatchThreadID)
        {
            output[tick] = tick;
            output[0] += tick;
        }
        """,
    )
    kernel = device.create_compute_kernel(
        device.link_program([module], [module.entry_point("stamp")])
    )
    output = device.create_buffer(
        element_count=104,
        struct_size=4,
        usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
        data=np.zeros(104, dtype=np.uint32),
    )
    dispatch = BoundDispatch(device, kernel, [1, 1, 1], {"output": output}, dynamic=("tick",))
    submissions = BoundedSubmissions(device)
    stats = chunk_timings()
    submissions.begin_frame(stats)
    for first in range(1, 104, 25):
        encoder = device.create_command_encoder()
        for tick in range(first, min(first + 25, 104)):
            dispatch.dispatch(command_encoder=encoder, tick=tick)
        last = submissions.submit(encoder, 2, stats)
    device.wait_for_submit(last)
    submissions.retire_finished()
    expected = np.arange(104, dtype=np.uint32)
    expected[0] = expected.sum()
    np.testing.assert_array_equal(output.to_numpy().view(np.uint32), expected)
    assert not submissions.pending
