"""Pixel-to-world navigation for a single world, independent of the GPU."""

from dataclasses import dataclass


@dataclass
class Camera:
    """Offset from the world midpoint (or the followed creature), in metres."""

    world_size: float = 512.0
    scale: float = 1.0
    offset: tuple[float, float] = (0.0, 0.0)

    def pixels_per_unit(self, rect):
        return min(rect[2], rect[3]) / self.world_size * self.scale

    def screen_to_world(self, point, rect):
        """Map a screen position to world metres for a midpoint-based camera."""
        x, y, width, height = rect
        pixels_per_unit = self.pixels_per_unit(rect)
        return (
            self.world_size / 2 + self.offset[0] + (point[0] - x - width / 2) / pixels_per_unit,
            self.world_size / 2 + self.offset[1] + (point[1] - y - height / 2) / pixels_per_unit,
        )

    def zoom_at(self, point, rect, wheel_delta):
        """Keep the world position under the cursor fixed while changing zoom."""
        before = self.pixels_per_unit(rect)
        # Equal wheel steps give equal proportional magnification: an even perceived
        # speed at every scale, rather than slowing down as additive zoom grows.
        self.scale = min(256.0, max(1.0, self.scale * 1.2**wheel_delta))
        correction = 1 / before - 1 / self.pixels_per_unit(rect)
        x, y, width, height = rect
        self.offset = (
            self.offset[0] + (point[0] - x - width / 2) * correction,
            self.offset[1] + (point[1] - y - height / 2) * correction,
        )

    def pan(self, delta_pixels, rect):
        """Drag the world with the pointer, with the same pixel speed at any zoom."""
        pixels_per_unit = self.pixels_per_unit(rect)
        self.offset = (
            self.offset[0] - delta_pixels[0] / pixels_per_unit,
            self.offset[1] - delta_pixels[1] / pixels_per_unit,
        )

    def reset(self):
        self.scale = 1.0
        self.offset = (0.0, 0.0)
