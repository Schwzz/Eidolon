"""Display-independent tissue contours and tapered appendage meshes.

All dimensions use body-radius units. Geometry is sampled from a pose, never
integrated here, so rendering frequency cannot change the specimen's motion.
"""

from dataclasses import dataclass
from math import atan2, cos, sin, tau

from eidolon.organism import Appearance, OrganismPose, TendrilParameters


Point = tuple[float, float]

# Clockwise rounded masses. No alternating pointed lobes or blade-like corners.
BODY_PROFILES: dict[str, tuple[Point, ...]] = {
    "blob": (
        (1.20, -0.32), (1.24, 0.25), (0.98, 0.68), (0.46, 0.90),
        (-0.04, 0.76), (-0.72, 1.05), (-1.25, 0.60), (-1.12, 0.10),
        (-1.43, -0.44), (-0.95, -0.94), (-0.42, -1.06),
        (0.26, -0.78), (0.83, -0.91),
    ),
    "parasite": (
        (0.62, -1.35), (0.80, -0.70), (0.66, -0.15), (0.33, 0.52),
        (0.24, 1.24), (-0.18, 1.65), (-0.58, 1.10), (-0.42, 0.40),
        (-0.77, -0.15), (-0.88, -0.88), (-0.49, -1.55), (0.10, -1.70),
    ),
    "veil": (
        (1.58, -0.44), (1.78, 0.08), (1.32, 0.68), (0.56, 0.83),
        (0.10, 0.62), (-0.63, 0.98), (-1.35, 0.70), (-1.70, 0.16),
        (-1.42, -0.43), (-0.82, -0.87), (-0.12, -0.62), (0.80, -0.85),
    ),
    "jelly": (
        (1.08, -0.40), (1.20, 0.12), (0.90, 0.56), (0.44, 0.40),
        (0.08, 0.60), (-0.26, 0.38), (-0.80, 0.62), (-1.20, 0.22),
        (-1.35, -0.38), (-0.93, -1.00), (-0.36, -1.30),
        (0.40, -1.07), (0.82, -0.85),
    ),
    "creeper": (
        (1.35, -0.23), (1.08, 0.40), (0.52, 0.58), (-0.08, 0.52),
        (-0.62, 0.86), (-1.26, 0.42), (-1.50, -0.08),
        (-1.18, -0.72), (-0.45, -0.84), (0.20, -0.49), (0.82, -0.69),
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
    """Soft, trailing ribbons with broad overlapping roots and a smooth taper.

    Shared proximal contraction ties the appendage to the body; smaller delayed
    waves travel toward the tip. The local normal blends into a shared trailing
    direction; S-bends replace the old rigid, outward-curving horns.
    """
    position = tendril.angle / tau
    root = contour_point(anchors, position)
    before, after = contour_point(anchors, position - 0.002), contour_point(anchors, position + 0.002)
    dx, dy = after[0] - before[0], after[1] - before[1]
    normal_angle = atan2(-dx, dy)
    steering = atan2(sin(tendril.heading - normal_angle), cos(tendril.heading - normal_angle))
    # Roots start beneath the body; the body membrane covers their seam.
    x = root[0] - cos(normal_angle) * 0.30
    y = root[1] - sin(normal_angle) * 0.30
    centers, left, right, widths = [], [], [], []
    shared = sin(pose.seconds * 1.05 + appearance.body.phase)
    for index in range(segments + 1):
        u = index / segments
        delayed = sin(pose.seconds * tendril.frequency - u * 3.2 + tendril.phase)
        turn = min(1.0, u * 3.0)
        turn = turn * turn * (3 - 2 * turn)
        bend = tendril.curl * sin(u * tau - 0.4) * u * 0.85
        bend += tendril.curl * 2.2 * u**3
        bend += tendril.sway * u * (shared * 0.35 + delayed)
        angle = normal_angle + steering * turn + bend
        if index:
            x += cos(angle) * tendril.length / segments
            y += sin(angle) * tendril.length / segments
        width = (0.05 + 0.26 * tendril.thickness) * (1 - u) ** 1.35
        width *= 1 + 0.06 * sin(pose.seconds * 1.05 - u * 4 + appearance.body.phase)
        nx, ny = -sin(angle) * width, cos(angle) * width
        centers.append((x, y))
        left.append((x + nx, y + ny))
        right.append((x - nx, y - ny))
        widths.append(width)
    return AppendageMesh(tuple(centers), tuple(left), tuple(right), tuple(widths), root)


def inset_outline(outline: tuple[Point, ...], center: Point, scale: float) -> tuple[Point, ...]:
    return tuple((center[0] + (x - center[0]) * scale, center[1] + (y - center[1]) * scale) for x, y in outline)
