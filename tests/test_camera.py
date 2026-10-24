"""Navigation invariants that do not need a window or GPU."""

import pytest

from worms.camera import Camera


@pytest.mark.parametrize("rect", [(0, 0, 512, 512), (280, 8, 992, 704), (31, 53, 240, 600)])
@pytest.mark.parametrize("scale", [1.0, 16.0, 200.0])
def test_zoom_keeps_cursor_world_position_fixed(rect, scale):
    camera = Camera(scale=scale, offset=(37.0, -82.0))
    point = (rect[0] + rect[2] * 0.8, rect[1] + rect[3] * 0.3)
    anchor = camera.screen_to_world(point, rect)

    camera.zoom_at(point, rect, 2.5)

    assert camera.screen_to_world(point, rect) == pytest.approx(anchor)


def test_equal_wheel_steps_have_equal_relative_magnification():
    camera = Camera()
    scales = [camera.scale]
    for _ in range(8):
        camera.zoom_at((256, 256), (0, 0, 512, 512), 1)
        scales.append(camera.scale)

    assert [after / before for before, after in zip(scales, scales[1:])] == pytest.approx([1.2] * 8)


def test_zoom_out_reverses_zoom_in():
    camera = Camera(scale=4.0, offset=(-27.0, 103.0))
    rect, point = (300, 24, 960, 640), (1060, 150)

    camera.zoom_at(point, rect, 3.5)
    camera.zoom_at(point, rect, -3.5)

    assert camera.scale == pytest.approx(4.0)
    assert camera.offset == pytest.approx((-27.0, 103.0))


@pytest.mark.parametrize("wheel_delta, expected_scale", [(-100, 1.0), (100, 256.0)])
def test_zoom_limit_keeps_cursor_anchor_without_further_drift(wheel_delta, expected_scale):
    camera = Camera(scale=16.0, offset=(41.0, -16.0))
    rect, point = (300, 24, 960, 640), (1060, 150)
    anchor = camera.screen_to_world(point, rect)

    camera.zoom_at(point, rect, wheel_delta)
    limited_offset = camera.offset
    camera.zoom_at(point, rect, wheel_delta)

    assert camera.scale == expected_scale
    assert camera.offset == limited_offset
    assert camera.screen_to_world(point, rect) == pytest.approx(anchor)


@pytest.mark.parametrize("scale", [1.0, 8.0, 256.0])
def test_pan_keeps_grabbed_world_point_under_moving_cursor(scale):
    camera = Camera(scale=scale, offset=(48.0, -10.0))
    rect, point, delta = (280, 8, 992, 704), (700, 300), (47, -83)
    grabbed = camera.screen_to_world(point, rect)

    camera.pan(delta, rect)

    moved_point = (point[0] + delta[0], point[1] + delta[1])
    assert camera.screen_to_world(moved_point, rect) == pytest.approx(grabbed)


def test_reset_restores_world_center_and_fit_scale():
    camera = Camera(world_size=320.0, scale=64.0, offset=(40.0, -90.0))

    camera.reset()

    assert camera.scale == 1.0
    assert camera.offset == (0.0, 0.0)
    assert camera.screen_to_world((550, 300), (300, 100, 500, 400)) == (160.0, 160.0)
    assert camera.pixels_per_unit((300, 100, 500, 400)) == 1.25
