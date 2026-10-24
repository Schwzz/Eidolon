"""Pure fit-to-window camera: presentation scale never changes world state."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class Camera:
    scale: float
    offset: tuple[float, float]
    world_size: tuple[float, float]

    @classmethod
    def fit(cls, world_size: tuple[float, float], viewport: tuple[float, float, float, float]) -> "Camera":
        width, height = world_size
        x, y, view_width, view_height = viewport
        if not isfinite(x) or not isfinite(y):
            raise ValueError("viewport origin must be finite")
        if any(not isfinite(value) or value <= 0 for value in (width, height, view_width, view_height)):
            raise ValueError("world and viewport dimensions must be finite and positive")
        scale = min(view_width / width, view_height / height)
        return cls(scale, (x + (view_width - width * scale) / 2, y + (view_height - height * scale) / 2), world_size)

    def world_to_screen(self, position: tuple[float, float]) -> tuple[float, float]:
        return tuple(origin + coordinate * self.scale for origin, coordinate in zip(self.offset, position))

    def screen_to_world(self, position: tuple[float, float]) -> tuple[float, float]:
        return tuple((coordinate - origin) / self.scale for origin, coordinate in zip(self.offset, position))
