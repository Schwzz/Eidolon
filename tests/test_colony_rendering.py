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
