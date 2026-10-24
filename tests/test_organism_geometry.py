from math import hypot, isfinite, pi, tau

import pytest

from eidolon.organism import Symbiote
from eidolon.organism_geometry import (
    BODY_PROFILES, appendage_mesh, body_anchors, contour_point,
)


PROFILE_SEEDS = [0, 2, 3, 4, 8]  # jelly, veil, creeper, blob, parasite


def sample_body(organism):
    anchors = body_anchors(organism.appearance, organism.pose())
    return tuple(contour_point(anchors, index / 160) for index in range(160))


@pytest.mark.parametrize("seed", PROFILE_SEEDS)
def test_body_is_soft_and_lopsided_rather_than_a_regular_spiked_star(seed) -> None:
    points = sample_body(Symbiote(seed))
    edges = tuple(zip(points, points[1:] + points[:1]))
    area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in edges)) / 2
    perimeter = sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in edges)
    # Soft masses may be much rounder than the old deeply cut, pointed profiles;
    # elongated parasites have lower circularity than the broad membrane plans.
    assert 0.50 < 4 * pi * area / perimeter**2 < 0.97
    # Opposite sides must also differ, rather than forming a regular star.
    opposite_error = sum(hypot(a[0] + b[0], a[1] + b[1]) for a, b in zip(points[:80], points[80:])) / 80
    assert opposite_error > 0.20


def test_all_five_profiles_have_distinct_outlines() -> None:
    organisms = [Symbiote(seed) for seed in PROFILE_SEEDS]
    assert {organism.appearance.body.profile for organism in organisms} == set(BODY_PROFILES)
    outlines = [sample_body(organism) for organism in organisms]
    for index, first in enumerate(outlines):
        for second in outlines[index + 1:]:
            mean_distance = sum(hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(first, second)) / len(first)
            assert mean_distance > 0.3


@pytest.mark.parametrize("seed", PROFILE_SEEDS)
def test_contour_is_closed_and_deformation_is_continuous(seed) -> None:
    organism = Symbiote(seed)
    anchors = body_anchors(organism.appearance, organism.pose())
    assert contour_point(anchors, 0) == contour_point(anchors, 1)
    near_start, near_end = contour_point(anchors, 0.000001), contour_point(anchors, 0.999999)
    assert hypot(near_start[0] - near_end[0], near_start[1] - near_end[1]) < 0.0001
    first = sample_body(organism)
    organism.update(1 / 60)
    next_frame = sample_body(organism)
    distances = [hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(first, next_frame)]
    assert 0 < max(distances) < 0.02
    organism.update(1)
    later = sample_body(organism)
    assert max(hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(first, later)) > 0.04


@pytest.mark.parametrize("seed", PROFILE_SEEDS)
def test_appendages_have_integrated_roots_broad_muscles_and_tapered_tips(seed) -> None:
    organism = Symbiote(seed)
    appearance, pose = organism.appearance, organism.pose()
    anchors = body_anchors(appearance, pose)
    for tendril in appearance.tendrils:
        mesh = appendage_mesh(anchors, tendril, appearance, pose)
        assert mesh.root == contour_point(anchors, tendril.angle / tau)
        root_overlap = hypot(mesh.centerline[0][0] - mesh.root[0], mesh.centerline[0][1] - mesh.root[1])
        assert root_overlap == pytest.approx(0.30)
        assert mesh.widths[0] > 0.20  # Half-width, in nominal body-radius units.
        assert mesh.widths[-1] == 0
        assert all(first > second for first, second in zip(mesh.widths, mesh.widths[1:]))
        for point in mesh.silhouette:
            assert all(isfinite(component) for component in point)
        lengths = [hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(mesh.centerline, mesh.centerline[1:])]
        assert sum(lengths) == pytest.approx(tendril.length)


def test_tissue_motion_is_smooth_repeatable_and_tracks_the_body() -> None:
    organism = Symbiote(17)
    appearance = organism.appearance

    def mesh():
        pose = organism.pose()
        return appendage_mesh(body_anchors(appearance, pose), appearance.tendrils[0], appearance, pose)

    first = mesh()
    assert mesh() == first
    organism.update(1 / 60)
    second = mesh()
    assert first.root != second.root
    displacement = max(hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(first.centerline, second.centerline))
    assert 0 < displacement < 0.04
