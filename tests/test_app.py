"""Screen layout remains testable without creating a graphics device."""

from worms.app import world_views


def test_grid_contains_every_world_in_separate_screen_rectangles():
    width, height = 257, 149
    views = world_views(width, height, 5)

    assert [world for world, _ in views] == list(range(5))
    for index, (_, (x, y, w, h)) in enumerate(views):
        assert w > 0 and h > 0
        assert 0 <= x < x + w <= width
        assert 0 <= y < y + h <= height
        for _, (other_x, other_y, other_w, other_h) in views[index + 1 :]:
            assert (
                x + w <= other_x
                or other_x + other_w <= x
                or y + h <= other_y
                or other_y + other_h <= y
            )


def test_selected_world_uses_the_whole_inset_view():
    assert world_views(257, 149, 5, selected_world=3) == [(3, (8, 8, 241, 133))]


def test_tiny_resized_window_has_no_out_of_bounds_viewports():
    views = world_views(2, 1, 4)
    assert views
    for _, (x, y, width, height) in views:
        assert 0 <= x < x + width <= 2
        assert 0 <= y < y + height <= 1
