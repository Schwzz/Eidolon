"""Biological state in world units, independent of appearance and rendering."""

from dataclasses import dataclass
from math import atan2, cos, hypot, sin


# Conservative disk containing the placeholder body and all animated appendages.
SPRITE_EXTENT = 6.0


@dataclass(frozen=True, slots=True)
class LifePose:
    position: tuple[float, float]
    scale: float
    direction: float


@dataclass(slots=True)
class LifeState:
    position: tuple[float, float]
    direction: float
    base_radius: float
    energy: float
    max_energy: float
    velocity: tuple[float, float] = (0.0, 0.0)
    age_seconds: float = 0.0
    body_scale: float = 1.0
    nutrition: float = 0.0
    foods_eaten: int = 0
    alive: bool = True
    previous_position: tuple[float, float] = (0.0, 0.0)
    previous_scale: float = 1.0
    previous_direction: float = 0.0

    def __post_init__(self) -> None:
        self.alive = self.energy > 0
        self.begin_step()

    @property
    def health(self) -> float:
        """Energy-derived vitality (0–100); there is no injury model yet."""
        return min(100.0, max(0.0, self.energy / self.max_energy * 100)) if self.alive else 0.0

    @property
    def physical_radius(self) -> float:
        """Nominal body radius, independent of the camera."""
        return self.base_radius * self.body_scale

    @property
    def collision_radius(self) -> float:
        return self.physical_radius * SPRITE_EXTENT

    @property
    def speed(self) -> float:
        return hypot(*self.velocity)

    def begin_step(self) -> None:
        self.previous_position = self.position
        self.previous_scale = self.body_scale
        self.previous_direction = self.direction

    def sample(self, alpha: float = 1.0) -> LifePose:
        alpha = min(1.0, max(0.0, alpha))
        turn = atan2(sin(self.direction - self.previous_direction), cos(self.direction - self.previous_direction))
        return LifePose(
            tuple(a + (b - a) * alpha for a, b in zip(self.previous_position, self.position)),
            self.previous_scale + (self.body_scale - self.previous_scale) * alpha,
            self.previous_direction + turn * alpha,
        )

    def die(self) -> None:
        self.energy = 0.0
        self.alive = False
        self.velocity = (0.0, 0.0)
