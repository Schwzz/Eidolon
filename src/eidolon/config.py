"""Validated application settings; no window or simulation side effects."""

from dataclasses import dataclass
from math import isfinite


MIN_WINDOW_WIDTH = 480
MIN_WINDOW_HEIGHT = 320


@dataclass(frozen=True, slots=True)
class AppConfig:
    width: int = 1280
    height: int = 720
    max_fps: int = 60  # Zero means uncapped rendering.
    simulation_hz: int = 60
    max_frame_seconds: float = 0.25
    max_updates_per_frame: int = 8
    seed: int = 17

    def __post_init__(self) -> None:
        for name in ("width", "height", "max_fps", "simulation_hz", "max_updates_per_frame", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.width < MIN_WINDOW_WIDTH or self.height < MIN_WINDOW_HEIGHT:
            raise ValueError(
                f"window size must be at least {MIN_WINDOW_WIDTH} x {MIN_WINDOW_HEIGHT}"
            )
        if self.max_fps < 0:
            raise ValueError("max_fps must be nonnegative (0 means uncapped)")
        if self.simulation_hz <= 0 or self.max_updates_per_frame <= 0:
            raise ValueError("simulation_hz and max_updates_per_frame must be positive")
        if not isfinite(self.max_frame_seconds) or self.max_frame_seconds <= 0:
            raise ValueError("max_frame_seconds must be finite and positive")
