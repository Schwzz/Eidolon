"""Fixed simulation steps driven by elapsed wall time, independent of drawing."""

from collections.abc import Callable
from math import isfinite

from eidolon.config import AppConfig


class FixedStepClock:
    def __init__(self, config: AppConfig) -> None:
        self.step_seconds = 1.0 / config.simulation_hz
        self._max_frame_seconds = config.max_frame_seconds
        self._max_updates = config.max_updates_per_frame
        self._accumulator = 0.0

    def advance(self, elapsed: float, update: Callable[[float], None]) -> int:
        """Run due steps, bounding catch-up work after a stall.

        Excess whole steps are dropped to keep the desktop responsive. Retain
        the fractional remainder, so ordinary frame jitter does not lose time.
        """
        if not isfinite(elapsed) or elapsed < 0:
            raise ValueError("elapsed time must be finite and nonnegative")
        self._accumulator += min(elapsed, self._max_frame_seconds)
        # Tolerate roundoff at exact step boundaries (e.g. 30 FPS vs 60 Hz).
        due = int((self._accumulator + self.step_seconds * 1e-9) / self.step_seconds)
        self._accumulator = max(0.0, self._accumulator - due * self.step_seconds)
        steps = min(due, self._max_updates)
        for _ in range(steps):
            update(self.step_seconds)
        return steps
