"""World-space, force-limited pointer input with no display dependency."""

from dataclasses import dataclass
from math import hypot


@dataclass(slots=True)
class GrabConstraint:
    node: int
    target: tuple[float, float]
    offset: tuple[float, float]
    applied_force: float = 0.0

    def force(self, position, velocity, world_size, radius, scale):
        target = tuple(min(size - radius, max(radius, point + offset))
                       for size, point, offset in zip(world_size, self.target, self.offset))
        fx = 120 * (target[0] - position[0]) - 20 * velocity[0]
        fy = 120 * (target[1] - position[1]) - 20 * velocity[1]
        magnitude = hypot(fx, fy)
        limit = 14_000 * scale
        if magnitude > limit:
            fx, fy = fx * limit / magnitude, fy * limit / magnitude
        self.applied_force = min(magnitude, limit)
        return fx, fy
