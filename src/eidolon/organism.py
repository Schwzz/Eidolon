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
    fold_count: int
    membrane_alpha: int
    trail_angle: float


@dataclass(frozen=True, slots=True)
class TendrilParameters:
    angle: float
    length: float
    curl: float
    sway: float
    frequency: float
    phase: float
    thickness: float
    heading: float


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
    # Ink-like tissue palettes; brightness belongs to internal light, not gloss.
    hue = rng.choice((0.10, 0.55, 0.92, 0.44, 0.73)) + rng.uniform(-0.025, 0.025)

    def color(h: float, saturation: float, value: float) -> Color:
        return tuple(round(channel * 255) for channel in hsv_to_rgb(h % 1, saturation, value))

    membrane = color(hue, rng.uniform(0.35, 0.55), 0.67)
    tissue = color(hue + 0.025, 0.43, rng.uniform(0.29, 0.36))
    core_color = color(hue + rng.uniform(0.30, 0.46), 0.46, 0.95)
    accent = color(hue + 0.08, 0.35, 0.48)
    profile = rng.choice(("blob", "parasite", "veil", "jelly", "creeper"))
    # Each plan has a trailing edge; attachments never surround the whole body.
    trail_angle, attachment_start, attachment_end = {
        "blob": (1.15, 0.13, 0.48),
        "parasite": (1.80, 0.23, 0.50),
        "veil": (1.05, 0.15, 0.55),
        "jelly": (1.55, 0.17, 0.52),
        "creeper": (2.20, 0.24, 0.63),
    }[profile]
    body = BodyParameters(
        width=rng.uniform(0.85, 1.15), height=rng.uniform(0.95, 1.15),
        asymmetry=rng.uniform(0.14, 0.24), lobes=rng.uniform(0.08, 0.16),
        phase=rng.uniform(0, tau),
        profile=profile, shear=rng.uniform(-0.22, 0.22),
        fold_depth=rng.uniform(0.12, 0.24), fold_count=rng.randint(2, 4),
        membrane_alpha=rng.randint(38, 65),
        trail_angle=trail_angle + rng.uniform(-0.20, 0.20),
    )
    count = rng.randint(7, 11)
    tendrils = tuple(
        TendrilParameters(
            angle=tau * (attachment_start + (attachment_end - attachment_start)
                         * (index + 0.5) / count + rng.uniform(-0.006, 0.006)),
            length=rng.uniform(1.35, 3.10), curl=rng.uniform(-0.85, 0.85),
            sway=rng.uniform(0.24, 0.42), frequency=rng.uniform(0.65, 1.30),
            phase=body.phase + index * 0.35 + rng.uniform(-0.15, 0.15),
            thickness=rng.uniform(0.75, 1.12),
            heading=body.trail_angle + rng.uniform(-0.40, 0.40),
        )
        for index in range(count)
    )
    core_anchor = {
        "blob": (-0.24, -0.08), "parasite": (-0.10, -0.70),
        "veil": (-0.38, -0.04), "jelly": (-0.12, -0.48),
        "creeper": (-0.30, -0.13),
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
