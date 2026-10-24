"""Exercise pointer gestures and inspection state without creating a GPU device."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from worms.app import App
from worms.camera import Camera


@pytest.fixture
def app():
    app = object.__new__(App)
    app.camera = Camera()
    app.texture = SimpleNamespace(width=1280, height=720)
    app.window = SimpleNamespace(panel=SimpleNamespace(size=SimpleNamespace(x=248)))
    app.sim = SimpleNamespace(
        num_worlds=4,
        pick=Mock(return_value=None),
        inspect=Mock(return_value={"position": (86.0, 392.0), "guid": 42, "alive": True}),
    )
    app.selected_world = 2
    return app


def test_focused_view_excludes_control_panel(app):
    assert app.focused_rect() == (288, 8, 984, 704)


def test_overview_ignores_zoom_drag_and_picking(app):
    app.selected_world = None

    app.zoom((500, 300), 2)
    app.mouse_down((500, 300))
    app.mouse_move((550, 300))
    app.mouse_up((550, 300))
    app.pick((500, 300))

    assert app.focused_rect() is None
    assert app.camera == Camera()
    app.sim.pick.assert_not_called()


def test_pointer_outside_focused_view_does_not_navigate(app):
    app.zoom((100, 300), 2)
    app.mouse_down((100, 300))
    app.mouse_move((500, 300))
    app.mouse_up((500, 300))

    assert app.camera == Camera()
    app.sim.pick.assert_not_called()


def test_zoom_in_focused_view_preserves_cursor_anchor(app):
    point = (1100, 250)
    anchor = app.camera.screen_to_world(point, app.focused_rect())

    app.zoom(point, 4)

    assert app.camera.scale > 1
    assert app.camera.screen_to_world(point, app.focused_rect()) == pytest.approx(anchor)


def test_click_with_small_pointer_jitter_selects_and_inspects_creature(app):
    app.sim.pick.return_value = 7

    app.mouse_down((500, 300))
    app.mouse_move((502, 301))
    app.mouse_up((502, 301))

    assert app.camera.offset == (0.0, 0.0)
    assert app.selected_creature == 7
    assert app.creature_stats == {"position": (86.0, 392.0), "guid": 42, "alive": True}
    app.sim.pick.assert_called_once()
    app.sim.inspect.assert_called_once_with(2, 7)


def test_four_pixel_drag_moves_grabbed_point_and_does_not_pick(app):
    rect = app.focused_rect()
    grabbed = app.camera.screen_to_world((500, 300), rect)

    app.mouse_down((500, 300))
    app.mouse_move((502, 301))
    app.mouse_move((504, 300))
    app.mouse_up((504, 300))

    assert app.camera.screen_to_world((504, 300), rect) == pytest.approx(grabbed)
    app.sim.pick.assert_not_called()


def test_release_also_detects_drag_when_no_move_event_arrived(app):
    app.mouse_down((500, 300))
    app.mouse_up((510, 300))

    assert app.camera.offset[0] < 0
    app.sim.pick.assert_not_called()


def test_release_without_press_has_no_effect(app):
    app.mouse_up((500, 300))

    assert app.camera == Camera()
    app.sim.pick.assert_not_called()


def test_release_outside_view_cancels_drag_and_reentry_is_inert(app):
    app.mouse_down((500, 300))
    app.mouse_move((510, 300))
    offset = app.camera.offset
    app.mouse_up((100, 300))
    app.mouse_move((600, 300))
    app.mouse_up((600, 300))

    assert app.camera.offset == offset
    app.sim.pick.assert_not_called()


def test_cancellation_clears_drag_even_without_release(app):
    app.mouse_down((500, 300))
    app.mouse_move((510, 300))
    offset = app.camera.offset
    app.cancel_drag()
    app.mouse_move((600, 300))
    app.mouse_up((600, 300))

    assert app.camera.offset == offset
    app.sim.pick.assert_not_called()


def test_pan_stops_following_without_a_camera_jump(app):
    app.selected_creature = 7
    app.set_follow(True)
    app.camera.offset = (3.0, -2.0)
    rect = app.focused_rect()
    center = (rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)
    moved = (center[0] + 12, center[1] - 9)
    grabbed = (89.0, 390.0)  # Followed position plus the camera's existing offset.

    app.mouse_down(center)
    app.mouse_move(moved)
    app.mouse_up(moved)

    assert not app.following
    assert app.selected_creature == 7
    assert app.camera.screen_to_world(moved, rect) == pytest.approx(grabbed)
    app.sim.inspect.assert_called_once_with(2, 7)
    app.sim.pick.assert_not_called()


def test_follow_requires_selection_and_centers_selected_creature(app):
    app.set_follow(True)
    assert not app.following

    app.selected_creature = 7
    app.camera.offset = (40.0, -90.0)
    app.camera.scale = 16.0
    app.set_follow(True)

    assert app.following
    assert app.camera.offset == (0.0, 0.0)
    assert app.camera.scale == 16.0
    app.sim.inspect.assert_not_called()


@pytest.mark.parametrize("world", [0, None])
def test_world_change_clears_selection_follow_and_navigation(app, world):
    app.selected_creature = 7
    app.creature_stats = {"position": (86.0, 392.0), "guid": 42, "alive": True}
    app.set_follow(True)
    app.camera.scale = 16.0
    app.camera.offset = (40.0, -90.0)
    app.mouse_down((500, 300))

    app.selected_world = world
    app.mouse_up((500, 300))

    assert app.selected_creature is None
    assert app.creature_stats is None
    assert not app.following
    assert app.camera == Camera()
    app.sim.pick.assert_not_called()
    app.sim.inspect.assert_not_called()


def test_fit_world_restores_overview_of_selected_world_and_preserves_inspection(app):
    app.selected_creature = 7
    app.creature_stats = {"position": (86.0, 392.0), "guid": 42, "alive": True}
    app.set_follow(True)
    app.camera.scale = 64.0
    app.camera.offset = (40.0, -90.0)
    app.mouse_down((500, 300))

    app.fit_world()
    app.mouse_up((500, 300))

    assert app.selected_world == 2
    assert app.selected_creature == 7
    assert app.creature_stats == {"position": (86.0, 392.0), "guid": 42, "alive": True}
    assert not app.following
    assert app.camera == Camera()
    app.sim.pick.assert_not_called()


def test_pick_passes_world_position_pixel_tolerance_and_follow_context(app):
    app.selected_creature = 12
    app.following = True
    app.camera.scale = 4.0
    app.camera.offset = (20.0, -30.0)
    app.sim.pick.return_value = 12

    app.pick((835, 305))

    app.sim.pick.assert_called_once_with(
        2,
        pytest.approx((289.75, 236.25)),
        tolerance=pytest.approx(5 / 4),
        selected=12,
        selected_guid=0,
        follow_position=(256.0, 256.0),
        follow=True,
    )
    assert app.following
    assert app.camera.offset == (20.0, -30.0)
    app.sim.inspect.assert_not_called()


def test_empty_space_clears_selected_creature_and_stops_follow(app):
    app.selected_creature = 12
    app.set_follow(True)

    app.pick((835, 305))

    assert app.selected_creature is None
    assert app.creature_stats is None
    assert not app.following


@pytest.mark.parametrize("alive,guid", [(False, 42), (True, 99)])
def test_inspection_clears_dead_or_reused_slot_without_following_new_occupant(app, alive, guid):
    app.selected_creature = 7
    app.refresh_stats()
    app.set_follow(True)
    app.sim.inspect.return_value = {"position": (10.0, 20.0), "guid": guid, "alive": alive}

    app.refresh_stats()

    assert app.selected_creature is None
    assert app.selected_guid == 0
    assert app.creature_stats is None
    assert not app.following
    assert app.camera.offset == (-170.0, 136.0)


def test_stopping_follow_is_safe_when_selected_creature_just_died(app):
    app.selected_creature = 7
    app.refresh_stats()
    app.set_follow(True)
    app.sim.inspect.return_value = {"position": (86.0, 392.0), "guid": 42, "alive": False}

    app.clear_selection()

    assert app.selected_creature is None
    assert not app.following
    assert app.camera.offset == (-170.0, 136.0)


def test_population_counts_include_only_the_visible_worlds(app):
    app.sim.creature_capacity = 256
    app.sim.target_food = 4096
    app.sim.inspect_world = Mock(
        return_value={"num_creatures": 64, "births": 7, "deaths": 3, "num_food": 2048}
    )
    app.refresh_population()
    assert app.population_stats == {
        "num_creatures": 64,
        "capacity": 256,
        "births": 7,
        "deaths": 3,
        "num_food": 2048,
        "food_target": 4096,
    }
    app.sim.inspect_world.assert_called_once_with(2)

    app.selected_world = None
    app.refresh_population()
    assert app.population_stats == {
        "num_creatures": 256,
        "capacity": 1024,
        "births": 28,
        "deaths": 12,
        "num_food": 8192,
        "food_target": 16384,
    }
