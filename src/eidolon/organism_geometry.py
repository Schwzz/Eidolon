"""Display-independent tissue contours and tapered appendage meshes.

All dimensions use body-radius units. Geometry is sampled from a pose, never
integrated here, so rendering frequency cannot change the specimen's motion.
"""

from dataclasses import dataclass
from math import atan2, cos, pi, sin, tau

from eidolon.organism import Appearance, OrganismPose, TendrilParameters


Point = tuple[float, float]

# Clockwise, deliberately unequal lobes and deep recesses, rather than ovals.
BODY_PROFILES: dict[str, tuple[Point, ...]] = {
    "mantle": (
        (0.95, -0.40), (0.72, 0.55), (0.28, 1.58), (-0.35, 1.05),
        (-1.25, 0.55), (-0.80, -0.12), (-0.85, -1.15), (-0.15, -1.85),
        (0.24, -0.72),
    ),
    "cleft": (
        (1.20, -1.18), (0.40, -0.65), (0.20, 0.0), (0.60, 0.65),
        (1.20, 1.18), (-0.05, 1.42), (-1.12, 0.75), (-1.28, -0.48),
        (-0.50, -1.40),
    ),
    "crown": (
        (1.42, -0.72), (0.65, 0.02), (1.20, 1.08), (0.12, 1.40),
        (-0.45, 0.62), (-1.45, 0.55), (-1.08, -0.42), (-0.35, -0.55),
        (-0.12, -1.65), (0.70, -1.20),
    ),
    "blade": (
        (1.90, -0.45), (1.05, 0.18), (0.35, 0.42), (-0.35, 1.12),
        (-1.42, 0.42), (-1.72, -0.32), (-0.65, -0.85), (0.15, -0.45),
        (1.05, -1.03),
    ),
}


def deform(point: Point, appearance: Appearance, pose: OrganismPose) -> Point:
    """One shared contraction field for tissue, roots, and embedded core."""
    x, y = point
    body = appearance.body
    time = pose.seconds
    contraction = sin(time * 1.05 + body.phase)
    fold = body.fold_depth * sin(time * 0.72 + y * 1.7 + body.phase)
    return (
        (x * body.width * (pose.breath + 0.035 * contraction) + body.shear * y
         + fold * (0.4 + abs(y) * 0.3)),
        y * body.height * pose.breath
        + body.lobes * sin(time * 0.85 + x * 2.0 + body.phase)
        + body.asymmetry * 0.25 * x * sin(time * 0.6 + body.phase),
    )


def body_anchors(appearance: Appearance, pose: OrganismPose) -> tuple[Point, ...]:
    return tuple(deform(point, appearance, pose) for point in BODY_PROFILES[appearance.body.profile])


def contour_point(anchors: tuple[Point, ...], position: float) -> Point:
    """Closed Catmull–Rom contour, with position wrapping over [0, 1)."""
    location = (position % 1.0) * len(anchors)
    index = int(location)
    u = location - index
    p0, p1, p2, p3 = (anchors[(index + offset) % len(anchors)] for offset in (-1, 0, 1, 2))
    u2, u3 = u * u, u * u * u
    return tuple(
        0.5 * (
            2 * p1[axis] + (-p0[axis] + p2[axis]) * u
            + (2 * p0[axis] - 5 * p1[axis] + 4 * p2[axis] - p3[axis]) * u2
            + (-p0[axis] + 3 * p1[axis] - 3 * p2[axis] + p3[axis]) * u3
        )
        for axis in (0, 1)
    )


@dataclass(frozen=True, slots=True)
class AppendageMesh:
    centerline: tuple[Point, ...]
    left: tuple[Point, ...]
    right: tuple[Point, ...]
    widths: tuple[float, ...]
    root: Point

    @property
    def silhouette(self) -> tuple[Point, ...]:
        return self.left + self.right[::-1]


def appendage_mesh(
    anchors: tuple[Point, ...], tendril: TendrilParameters,
    appearance: Appearance, pose: OrganismPose, segments: int = 24,
) -> AppendageMesh:
    """Arc-length stepping with broad overlapping roots and a continuous taper.

    Shared proximal contraction ties the appendage to the body; smaller delayed
    waves travel toward the tip. The curve bends as tissue instead of translating
    individual wire segments sideways.
    """
    position = tendril.angle / tau
    root = contour_point(anchors, position)
    before, after = contour_point(anchors, position - 0.002), contour_point(anchors, position + 0.002)
    dx, dy = after[0] - before[0], after[1] - before[1]
    normal_angle = atan2(-dx, dy)
    # Roots start beneath the body; the body membrane covers their seam.
    x = root[0] - cos(normal_angle) * 0.30
    y = root[1] - sin(normal_angle) * 0.30
    centers, left, right, widths = [], [], [], []
    shared = sin(pose.seconds * 1.05 + appearance.body.phase)
    for index in range(segments + 1):
        u = index / segments
        delayed = sin(pose.seconds * tendril.frequency - u * 3.2 + tendril.phase)
        bend = tendril.curl * u * 2.4
        bend += tendril.sway * sin(pi * u) * (shared + 0.30 * delayed)
        angle = normal_angle + bend
        if index:
            x += cos(angle) * tendril.length / segments
            y += sin(angle) * tendril.length / segments
        width = (0.24 + 0.08 * tendril.thickness) * tendril.thickness * (1 - u) ** 0.85
        width *= 1 + 0.10 * sin(pose.seconds * 1.05 - u * 4 + appearance.body.phase)
        nx, ny = -sin(angle) * width, cos(angle) * width
        centers.append((x, y))
        left.append((x + nx, y + ny))
        right.append((x - nx, y - ny))
        widths.append(width)
    return AppendageMesh(tuple(centers), tuple(left), tuple(right), tuple(widths), root)


def inset_outline(outline: tuple[Point, ...], center: Point, scale: float) -> tuple[Point, ...]:
    return tuple((center[0] + (x - center[0]) * scale, center[1] + (y - center[1]) * scale) for x, y in outline)
