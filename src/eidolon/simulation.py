"""Rendering-independent simulation state. Phase 0 contains no entities."""

from math import isfinite


class Simulation:
    def __init__(self) -> None:
        self.steps = 0
        self.elapsed_seconds = 0.0

    def update(self, dt: float) -> None:
        """Advance by a simulation timestep, measured in seconds."""
        if not isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.steps += 1
        self.elapsed_seconds += dt
