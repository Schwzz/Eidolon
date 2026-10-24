"""Seeded appearance and animation state, with no Pygame or display dependency."""

from colorsys import hsv_to_rgb
from dataclasses import dataclass
from math import sin, tau
from random import Random


Color = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class BodyParameters:
    width: float
    height: float
    asymmetry: float
    lobes: float
    phase: float


@dataclass(frozen=True, slots=True)
class TendrilParameters:
    angle: float
    length: float
    curl: float
    sway: float
    frequency: float
    phase: float
    thickness: float


@dataclass(frozen=True, slots=True)
class CoreParameters:
    radius: float
    elongation: float
    offset: tuple[float, float]
    pulse_rate: float
    phase: float


@dataclass(frozen=True, slots=True)
class Appearance:
    seed: int
    membrane_color: Color
    tissue_color: Color
    core_color: Color
    body: BodyParameters
    tendrils: tuple[TendrilParameters, ...]
    core: CoreParameters


def generate_appearance(seed: int) -> Appearance:
    """Use a private RNG once; drawing and updates never consume random numbers.

    Shape dimensions are relative to the body's nominal radius, angles are in
    radians, and frequencies are radians per simulation second.
    """
    rng = Random(seed)
    hue = rng.uniform(0.40, 0.73)

    def color(h: float, saturation: float, value: float) -> Color:
        return tuple(round(channel * 255) for channel in hsv_to_rgb(h % 1, saturation, value))

    membrane = color(hue, rng.uniform(0.35, 0.55), 0.85)
    tissue = color(hue + 0.04, 0.55, 0.25)
    core_color = color(hue + 0.24, 0.40, 1.0)
    body = BodyParameters(
        width=rng.uniform(0.85, 1.05), height=rng.uniform(1.05, 1.30),
        asymmetry=rng.uniform(0.14, 0.24), lobes=rng.uniform(0.05, 0.10),
        phase=rng.uniform(0, tau),
    )
    count = rng.randint(7, 11)
    tendrils = tuple(
        TendrilParameters(
            angle=tau * index / count + rng.uniform(-0.16, 0.16),
            length=rng.uniform(1.15, 2.05), curl=rng.uniform(-0.85, 0.85),
            sway=rng.uniform(0.16, 0.32), frequency=rng.uniform(0.65, 1.30),
            phase=rng.uniform(0, tau), thickness=rng.uniform(0.65, 1.20),
        )
        for index in range(count)
    )
    core = CoreParameters(
        radius=rng.uniform(0.18, 0.25), elongation=rng.uniform(1.05, 1.35),
        offset=(rng.uniform(-0.12, 0.12), rng.uniform(-0.13, 0.13)),
        pulse_rate=rng.uniform(1.6, 2.4), phase=rng.uniform(0, tau),
    )
    return Appearance(seed, membrane, tissue, core_color, body, tendrils, core)


@dataclass(frozen=True, slots=True)
class OrganismPose:
    seconds: float
    offset: tuple[float, float]
    rotation: float
    breath: float
    core_pulse: float


class Symbiote:
    """One organism's animation clock; no intelligence or environmental behavior."""

    def __init__(self, seed: int) -> None:
        self.appearance = generate_appearance(seed)
        self.reset()

    def reset(self) -> None:
        self.age_seconds = 0.0
        self.previous_age_seconds = 0.0

    def update(self, dt: float) -> None:
        self.previous_age_seconds = self.age_seconds
        self.age_seconds += dt

    def pose(self, alpha: float = 1.0) -> OrganismPose:
        """Interpolate the animation clock without mutating simulation state."""
        alpha = min(1.0, max(0.0, alpha))
        time = self.previous_age_seconds + (self.age_seconds - self.previous_age_seconds) * alpha
        phase = self.appearance.body.phase
        core = self.appearance.core
        return OrganismPose(
            seconds=time,
            offset=(0.035 * sin(time * 0.27 + phase), 0.045 * sin(time * 0.21)),
            rotation=0.13 * sin(time * 0.31 + phase),
            breath=1.0 + 0.035 * sin(time * 1.65 + phase),
            core_pulse=1.0 + 0.10 * sin(time * core.pulse_rate + core.phase),
        )
