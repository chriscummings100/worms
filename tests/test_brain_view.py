"""Selection reserves space for the brain without admitting world gestures there."""

from types import SimpleNamespace
from unittest.mock import Mock

from worms.app import App
from worms.brain_view import brain_panel_rect
from worms.camera import Camera


def make_app():
    app = object.__new__(App)
    app.camera = Camera()
    app.texture = SimpleNamespace(width=1280, height=720)
    app.window = None
    app.sim = SimpleNamespace(num_worlds=2, pick=Mock(), inspect=Mock())
    app.selected_world = 0
    return app


def test_selection_reserves_bottom_strip_and_clearing_restores_world_view():
    app = make_app()
    assert app.brain_rect() is None
    original = app.focused_rect()

    app.selected_creature = 1

    assert app.brain_rect() == (0, 528, 1280, 192)
    assert app.focused_rect() == (8, 8, 1264, 512)
    app.clear_selection()
    assert app.brain_rect() is None
    assert app.focused_rect() == original


def test_brain_strip_does_not_zoom_pan_or_pick_world():
    app = make_app()
    app.selected_creature = 1
    app.zoom((600, 600), 2)
    app.mouse_down((600, 600))
    app.mouse_move((650, 600))
    app.mouse_up((650, 600))
    app.pick((650, 600))

    assert app.camera == Camera()
    app.sim.pick.assert_not_called()


def test_drag_into_brain_strip_cancels_without_selecting():
    app = make_app()
    app.selected_creature = 1
    app.mouse_down((600, 500))
    app.mouse_move((600, 600))
    app.mouse_up((600, 500))

    assert app.camera == Camera()
    app.sim.pick.assert_not_called()


def test_small_window_keeps_two_thirds_available_for_world():
    assert brain_panel_rect(480, 300) == (0, 200, 480, 100)
