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
    profile: str
    shear: float
    fold_depth: float
    ridge_count: int
    membrane_alpha: int


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
    accent_color: Color


def generate_appearance(seed: int) -> Appearance:
    """Use a private RNG once; drawing and updates never consume random numbers.

    Shape dimensions are relative to the body's nominal radius, angles are in
    radians, and frequencies are radians per simulation second.
    """
    rng = Random(seed)
    # Distinct tissue palettes, including warm mineral and bruised organic hues.
    hue = rng.choice((0.08, 0.55, 0.94, 0.44, 0.73)) + rng.uniform(-0.025, 0.025)

    def color(h: float, saturation: float, value: float) -> Color:
        return tuple(round(channel * 255) for channel in hsv_to_rgb(h % 1, saturation, value))

    membrane = color(hue, rng.uniform(0.35, 0.55), 0.67)
    tissue = color(hue + 0.025, 0.55, rng.uniform(0.26, 0.34))
    core_color = color(hue + rng.uniform(0.30, 0.46), 0.46, 0.95)
    accent = color(hue + 0.08, 0.35, 0.48)
    profile = rng.choice(("mantle", "cleft", "crown", "blade"))
    body = BodyParameters(
        width=rng.uniform(0.85, 1.15), height=rng.uniform(0.95, 1.15),
        asymmetry=rng.uniform(0.14, 0.24), lobes=rng.uniform(0.08, 0.16),
        phase=rng.uniform(0, tau),
        profile=profile, shear=rng.uniform(-0.22, 0.22),
        fold_depth=rng.uniform(0.12, 0.24), ridge_count=rng.randint(4, 7),
        membrane_alpha=rng.randint(38, 65),
    )
    count = rng.randint(7, 11)
    tendrils = tuple(
        TendrilParameters(
            angle=tau * index / count + rng.uniform(-0.16, 0.16),
            length=rng.uniform(1.15, 2.05), curl=rng.uniform(-0.85, 0.85),
            sway=rng.uniform(0.16, 0.32), frequency=rng.uniform(0.65, 1.30),
            phase=body.phase + index * 0.35 + rng.uniform(-0.15, 0.15),
            thickness=rng.uniform(0.80, 1.35),
        )
        for index in range(count)
    )
    core_anchor = {
        "mantle": (-0.12, 0.12), "cleft": (-0.40, 0.05),
        "crown": (-0.16, 0.08), "blade": (-0.42, -0.02),
    }[profile]
    core = CoreParameters(
        radius=rng.uniform(0.18, 0.25), elongation=rng.uniform(1.05, 1.35),
        offset=(core_anchor[0] + rng.uniform(-0.06, 0.06), core_anchor[1] + rng.uniform(-0.06, 0.06)),
        pulse_rate=rng.uniform(1.6, 2.4), phase=rng.uniform(0, tau),
    )
    return Appearance(seed, membrane, tissue, core_color, body, tendrils, core, accent)


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
