"""Rendering-independent simulation state for a single animated symbiote."""

from math import isfinite

from eidolon.organism import Symbiote


class Simulation:
    def __init__(self, seed: int = 17) -> None:
        self.organism = Symbiote(seed)
        self.paused = False
        self.animation_speed = 1.0
        self.reset()

    def reset(self) -> None:
        """Restart the same specimen, preserving pause and speed controls."""
        self.steps = 0
        self.elapsed_seconds = 0.0
        self.organism.reset()

    def set_animation_speed(self, speed: float) -> None:
        if not isfinite(speed) or not 0.25 <= speed <= 2.0:
            raise ValueError("animation speed must be between 0.25 and 2.0")
        self.animation_speed = speed

    def update(self, dt: float) -> None:
        """Advance by a simulation timestep, measured in seconds."""
        if not isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if self.paused:
            self.organism.update(0.0)
            return
        self.steps += 1
        self.elapsed_seconds += dt
        self.organism.update(dt * self.animation_speed)
