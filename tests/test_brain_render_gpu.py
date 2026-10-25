"""Verify the displayed neurons and motor bars use the selected GPU record."""

import numpy as np
import pytest

from worms.app import App

pytestmark = pytest.mark.gpu


def test_panel_uses_selected_gpu_activations_and_pauses_without_changes():
    app = App(
        debug_gpu=True,
        headless=True,
        benchmark=False,
        snapshot_dir=None,
        num_worlds=2,
        creatures_per_world=2,
    )
    try:
        app.selected_world = 1
        app.selected_creature = 1
        app.paused = True
        capacity = app.sim.creature_capacity
        values = np.zeros(2 * capacity * 82, dtype=np.float32)
        selected = values.reshape(2 * capacity, 82)[capacity + 1]
        selected[3 * 4 + 2] = 1.0
        selected[64] = -1.0
        selected[80] = -1.0
        selected[81] = 0.25
        app.sim.brain_activations.copy_from_numpy(values)
        app.frame()
        app.device.wait()
        image = app.texture.to_numpy().copy()
        top = app.brain_rect()[1]

        # Segment-major inputs: channel 2 of segment 3 is the fourth column,
        # third row. Signed hidden/steering values show blue, positives amber.
        positive = image[top + 122, 268, :3].astype(int)
        negative = image[top + 84, 720, :3].astype(int)
        steering = image[top + 84, 998, :3].astype(int)
        assert positive[0] > positive[2] + 100
        assert negative[2] > negative[0] + 100
        assert steering[2] > steering[0] + 100
        assert image[top + 130, 1050, 0] > image[top + 130, 1200, 0] + 100
        assert image[top + 84, 1070, 2] > image[top + 84, 1200, 2] + 100

        app.frame()
        app.device.wait()
        np.testing.assert_array_equal(app.texture.to_numpy(), image)

        # The same local index in another world must use its own brain.
        app.selected_world = 0
        app.selected_creature = 1
        app.frame()
        app.device.wait()
        other = app.texture.to_numpy()
        assert int(other[top + 122, 268, 0]) < int(positive[0]) - 100
        assert int(other[top + 84, 720, 2]) < int(negative[2]) - 100
    finally:
        app.close()


def test_resizing_panel_keeps_neurons_round_and_labels_and_gauges_in_proportion():
    app = App(
        debug_gpu=True,
        headless=True,
        benchmark=False,
        snapshot_dir=None,
        num_worlds=1,
        creatures_per_world=1,
        creature_capacity=1,
    )
    try:
        app.selected_creature = 0
        app.paused = True
        values = np.zeros(82, dtype=np.float32)
        values[0] = 1.0
        values[64] = -1.0
        values[80] = -1.0
        values[81] = 0.25
        app.sim.brain_activations.copy_from_numpy(values)
        panels = []
        for width, height, scale, left, top in (
            (1280, 720, 1, 0, 0),
            (1920, 1080, 1, 320, 0),
            (900, 600, 900 / 1280, 0, 28.5),
        ):
            app.resize(width, height)
            app.frame()
            app.device.wait()
            panel = app.texture.to_numpy()[app.brain_rect()[1] :, :, :3].copy()
            panels.append(panel)

            def pixel(x, y):
                return panel[round(top + y * scale), round(left + x * scale)].astype(int)

            # A bright isolated input circle must retain its aspect ratio.
            x, y = round(left + 190 * scale), round(top + 70 * scale)
            patch = panel[y - 12 : y + 12, x - 12 : x + 12].astype(int)
            rows, columns = np.where(patch[:, :, 0] > patch[:, :, 2] + 100)
            diameter_x = columns.max() - columns.min() + 1
            diameter_y = rows.max() - rows.min() + 1
            assert abs(diameter_x - diameter_y) <= 1
            assert abs(diameter_x - 14 * scale) <= 2

            # Gauges and signed neurons follow the same uniform transform.
            assert pixel(720, 84)[2] > pixel(720, 84)[0] + 100
            assert pixel(1050, 130)[0] > pixel(1200, 130)[0] + 100
            assert pixel(1070, 84)[2] > pixel(1200, 84)[2] + 100

        # A wider window adds margins; text and content retain their native size.
        np.testing.assert_array_equal(panels[1][:50, 320:1600], panels[0][:50])
        np.testing.assert_allclose(panels[1][:, 320:1600], panels[0], atol=2)
    finally:
        app.close()


def test_reused_slot_hides_neurons_before_the_next_inspector_poll():
    import slangpy as spy

    app = App(
        debug_gpu=True,
        headless=True,
        benchmark=False,
        snapshot_dir=None,
        num_worlds=1,
        creatures_per_world=1,
        creature_capacity=1,
    )
    try:
        app.selected_creature = 0
        app.refresh_stats()
        app._next_stats_update = float("inf")
        app.paused = True
        app.sim.brain_activations.copy_from_numpy(np.ones(82, dtype=np.float32))
        app.frame()
        bright = app.texture.to_numpy().copy()
        top = app.brain_rect()[1]

        layout = app.sim._initialize.reflection.g_creatures.type_layout.element_type_layout
        cursor = spy.BufferCursor(layout, app.sim.creatures)
        cursor[0]["guid"] = app.selected_guid + 1000
        cursor.apply()
        app.frame()
        hidden = app.texture.to_numpy()

        assert app.selected_creature == 0  # The inspector has not polled yet.
        assert int(bright[top + 70, 190, 0]) > int(hidden[top + 70, 190, 0]) + 100
        assert int(bright[top + 84, 720, 0]) > int(hidden[top + 84, 720, 0]) + 100
    finally:
        app.close()
