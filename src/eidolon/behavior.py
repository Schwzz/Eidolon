"""Replaceable, stateless survival decisions over immutable sensory snapshots."""

from dataclasses import dataclass
from math import atan2, cos, sin
from typing import Protocol

from eidolon.world import Food


@dataclass(frozen=True, slots=True)
class Senses:
    position: tuple[float, float]
    direction: float
    age: float
    energy_fraction: float
    clearance: float
    world_size: tuple[float, float]
    food: Food | None


@dataclass(frozen=True, slots=True)
class Decision:
    direction: float
    speed_fraction: float = 1.0


class Controller(Protocol):
    def decide(self, senses: Senses) -> Decision: ...


class SurvivalRules:
    def __init__(self, seed: int) -> None:
        self.phase = (seed % 65536) * 0.61803398875

    def decide(self, senses: Senses) -> Decision:
        x, y = senses.position
        if senses.food is not None:
            target = senses.food.position
            return Decision(atan2(target[1] - y, target[0] - x))
        # A smooth wandering heading; wall repulsion steers before contact.
        angle = senses.direction + 0.65 * sin(senses.age * 0.7 + self.phase)
        angle += 0.20 * sin(senses.age * 1.7 - self.phase)
        vx, vy = cos(angle), sin(angle)
        width, height = senses.world_size
        reach = senses.clearance + 80.0
        vx += max(0.0, (reach - x) / 80) - max(0.0, (x - (width - reach)) / 80)
        vy += max(0.0, (reach - y) / 80) - max(0.0, (y - (height - reach)) / 80)
        return Decision(atan2(vy, vx))
