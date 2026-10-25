"""Bounded local rest-length actuators; no locomotion forces or graphics.

Regions use the original material coordinates, never the current deformed pose.
Reconnected bonds therefore inherit stable muscle membership without moving nodes.
"""

from math import isfinite, pi, sin
from numbers import Real

from eidolon.config import MuscleConfig


class MuscleSystem:
    REGION_COUNT = 8
    REGION_NAMES = tuple(f"{'upper' if row == 0 else 'lower'}-{column + 1}"
                         for row in range(2) for column in range(4))

    def __init__(self, material_positions, config: MuscleConfig | None = None) -> None:
        self.config = config if config is not None else MuscleConfig()
        self.material_positions = tuple(tuple(point) for point in material_positions)
        if (not self.material_positions or any(len(point) != 2 or not all(isfinite(v) for v in point)
                                               for point in self.material_positions)):
            raise ValueError("muscle material positions must be finite 2D points")
        self._left = min(point[0] for point in self.material_positions)
        self._width = max(point[0] for point in self.material_positions) - self._left
        self._middle = (min(point[1] for point in self.material_positions)
                        + max(point[1] for point in self.material_positions)) / 2
        self.node_regions = tuple(self._region(point) for point in self.material_positions)
        counts = tuple(self.node_regions.count(region) for region in range(self.REGION_COUNT))
        self._weights = tuple(count / len(self.node_regions) for count in counts)
        self.targets = (0.0,) * self.REGION_COUNT
        self.activations = (0.0,) * self.REGION_COUNT

    def _region(self, point) -> int:
        column = min(3, max(0, int(4 * (point[0] - self._left) / max(self._width, 1e-9))))
        return column + (4 if point[1] >= self._middle else 0)

    def bond_region(self, first: int, second: int) -> int:
        a, b = self.material_positions[first], self.material_positions[second]
        return self._region(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))

    def set_targets(self, values) -> None:
        try:
            values = tuple(values)
        except TypeError as error:
            raise ValueError("muscle targets must contain eight numbers in [0, 1]") from error
        if (len(values) != self.REGION_COUNT or any(isinstance(value, bool) or not isinstance(value, Real)
                or not isfinite(value) or not 0 <= value <= 1 for value in values)):
            raise ValueError("muscle targets must contain eight finite numbers in [0, 1]")
        self.targets = tuple(float(value) for value in values)

    def relax(self) -> None:
        self.targets = (0.0,) * self.REGION_COUNT

    def advance(self, dt: float, energy_budget: float = float("inf"), *, active: bool = True) -> float:
        """Slew activations; return energy spent. Unpowered tissue relaxes passively.

        Holding cost is weighted by material share, not bond count, so growth and
        reconnection cannot multiply the actuator's energy allowance.
        """
        if not isfinite(dt) or dt <= 0 or energy_budget < 0 or energy_budget != energy_budget:
            raise ValueError("muscle timestep must be positive and budget nonnegative")
        config = self.config
        targets = self.targets if active else (0.0,) * self.REGION_COUNT
        next_values = tuple(current + min(config.contraction_rate * dt, target - current)
                            if target >= current else max(target, current - config.relaxation_rate * dt)
                            for current, target in zip(self.activations, targets))
        cost = sum(weight * (config.holding_cost * value * dt
                             + config.contraction_cost * max(0.0, value - current))
                   for weight, current, value in zip(self._weights, self.activations, next_values))
        if not active or cost > energy_budget:
            # Relaxation cannot generate new contraction when fuel is exhausted.
            next_values = tuple(max(0.0, value - config.relaxation_rate * dt) for value in self.activations)
            cost = 0.0
        self.activations = next_values
        return cost

    @property
    def length_factors(self) -> tuple[float, ...]:
        return tuple(1 - self.config.max_contraction * value for value in self.activations)


def wave_activations(seconds: float) -> tuple[float, ...]:
    """Untrained four-second traveling contraction, after three seconds to settle."""
    if not isfinite(seconds) or seconds < 0:
        raise ValueError("muscle pattern time must be finite and nonnegative")
    if seconds < 3:
        return (0.0,) * MuscleSystem.REGION_COUNT
    phase = (seconds - 3) * (2 * pi / 4)
    ramp = min(1.0, (seconds - 3) / 1.0)
    return tuple(ramp * (0.65 if row == 0 else 0.85) * (0.5 + 0.5 * sin(phase - column * pi / 2 + row * pi))
                 for row in range(2) for column in range(4))
