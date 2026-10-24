from dataclasses import asdict

import pygame
import pytest

from eidolon.camera import Camera
from eidolon.colony import Colony
from eidolon.colony_rendering import ColonyRenderer
from eidolon.organism import generate_appearance


@pytest.mark.parametrize("seed", [0, 17, 42])
def test_rendered_nodes_form_one_cohesive_membrane_without_mutating_physics(seed) -> None:
    colony = Colony(seed, (400, 300), 24)
    renderer = ColonyRenderer()
    surface = pygame.Surface((800, 600), pygame.SRCALPHA)
    camera = Camera.fit((800, 600), (0, 0, 800, 600))
    initial = [asdict(node) for node in colony.nodes]
    renderer.draw(surface, surface.get_rect(), colony, camera, generate_appearance(seed))
    first = pygame.image.tobytes(surface, "RGBA")
    components = pygame.mask.from_surface(surface, threshold=40).get_bounding_rects()
    assert len([rect for rect in components if rect.width * rect.height > 16]) == 1
    surface.fill((0, 0, 0, 0))
    renderer.draw(surface, surface.get_rect(), colony, camera, generate_appearance(seed))
    assert pygame.image.tobytes(surface, "RGBA") == first
    assert [asdict(node) for node in colony.nodes] == initial


def test_rendering_tracks_node_deformation_and_reuses_local_surfaces() -> None:
    colony = Colony(17, (400, 300), 24)
    renderer = ColonyRenderer()
    surface = pygame.Surface((800, 600), pygame.SRCALPHA)
    camera = Camera.fit((800, 600), (0, 0, 800, 600))
    appearance = generate_appearance(17)
    renderer.draw(surface, surface.get_rect(), colony, camera, appearance)
    ink, sprite = renderer._ink, renderer._sprite
    first = pygame.image.tobytes(surface, "RGBA")
    for _ in range(60):
        colony.advance(1 / 120, (50, 0), (800, 600))
    surface.fill((0, 0, 0, 0))
    renderer.draw(surface, surface.get_rect(), colony, camera, appearance)
    assert renderer._ink is ink and renderer._sprite is sprite
    assert pygame.image.tobytes(surface, "RGBA") != first


def test_renderer_restores_clip_and_stays_inside_viewport() -> None:
    colony = Colony(17, (400, 300), 24)
    surface = pygame.Surface((800, 600), pygame.SRCALPHA)
    viewport = pygame.Rect(300, 200, 200, 200)
    clip = pygame.Rect(200, 100, 400, 400)
    surface.set_clip(clip)
    camera = Camera.fit((800, 600), (0, 0, 800, 600))
    ColonyRenderer().draw(surface, viewport, colony, camera, generate_appearance(17))
    assert surface.get_clip() == clip
    assert surface.get_at((50, 50)).a == 0


def test_detached_fragment_is_visible_without_a_ghost_membrane_bridge() -> None:
    colony = Colony(17, (400, 300), 24)
    node = max(colony.nodes, key=lambda candidate: candidate.position[0])
    colony.bonds = [bond for bond in colony.bonds if node.identifier not in (bond.first, bond.second)]
    colony.rebuild_topology()
    node.position = (700, 300)
    colony.begin_step()
    initial = [asdict(node) for node in colony.nodes]
    surface = pygame.Surface((800, 600), pygame.SRCALPHA)
    camera = Camera.fit((800, 600), (0, 0, 800, 600))
    renderer = ColonyRenderer()
    renderer.draw(surface, surface.get_rect(), colony, camera, generate_appearance(17))
    components = pygame.mask.from_surface(surface, threshold=40).get_bounding_rects()
    assert len([rect for rect in components if rect.width * rect.height > 16]) == 2
    assert surface.get_at((700, 300)).a > 40
    assert surface.get_at((560, 300)).a == 0
    assert [asdict(node) for node in colony.nodes] == initial
    assert len(renderer._buffers) <= 8


def test_microscopic_grains_are_render_only_and_glow_is_composited_into_tissue() -> None:
    colony = Colony(17, (400, 300), 24)
    appearance = generate_appearance(17)
    renderer = ColonyRenderer()
    surface = pygame.Surface((800, 600), pygame.SRCALPHA)
    camera = Camera.fit((800, 600), (0, 0, 800, 600))
    renderer.draw(surface, surface.get_rect(), colony, camera, appearance)
    assert len(renderer._grains) == 4 * len(colony.nodes) >= 340
    node = colony.nodes[colony.anchor_index]
    point = (round(node.position[0] + colony.spacing * 0.9), round(node.position[1]))
    color = surface.get_at(point)
    difference = sum(abs(color[axis] - appearance.tissue_color[axis]) for axis in (0, 1, 2))
    full_glow = sum(abs(appearance.core_color[axis] - appearance.tissue_color[axis]) for axis in (0, 1, 2))
    assert color.a > 200 and difference < full_glow * 0.5
