from math import cos, sin

import pygame
import pytest

from eidolon.organism import Symbiote
from eidolon.organism_rendering import SymbioteRenderer
from eidolon.organism_geometry import appendage_mesh, body_anchors
from eidolon.rendering import ChamberRenderer
from eidolon.simulation import Simulation


@pytest.fixture(autouse=True)
def cleanup_pygame():
    pygame.font.init()
    yield
    pygame.quit()


def pixels(surface):
    return pygame.image.tobytes(surface, "RGB")


def test_rendering_is_repeatable_and_does_not_advance_simulation() -> None:
    simulation = Simulation(seed=17)
    simulation.update(0.5)
    pose = simulation.organism.pose()
    renderer = ChamberRenderer()
    surface = pygame.Surface((1280, 720))
    renderer.draw(surface, simulation)
    first = pixels(surface)
    renderer.draw(surface, simulation)
    assert pixels(surface) == first
    assert simulation.organism.pose() == pose
    assert simulation.steps == 1


def test_motion_changes_body_and_tendril_pixels() -> None:
    organism = Symbiote(17)
    renderer = SymbioteRenderer()
    chamber = pygame.Rect(0, 0, 640, 640)
    surface = pygame.Surface(chamber.size)

    def frame():
        surface.fill((0, 0, 0))
        renderer.draw(surface, chamber, organism)
        body = pixels(surface.subsurface((240, 240, 160, 160)))
        tendrils = surface.copy()
        tendrils.fill((0, 0, 0), pygame.Rect(200, 200, 240, 240))
        return body, pixels(tendrils)

    before = frame()
    organism.update(1)
    after = frame()
    assert before[0] != after[0]
    assert before[1] != after[1]


@pytest.mark.parametrize("size", [(480, 320), (1280, 720), (1600, 900)])
def test_organism_is_visible_and_stays_inside_chamber_when_resized(size) -> None:
    simulation = Simulation()
    renderer = ChamberRenderer()
    occupied = pygame.Surface(size)
    empty = pygame.Surface(size)
    renderer.draw(occupied, simulation)
    renderer.draw(empty)
    chamber = pygame.Rect(24, 82, size[0] - 48, size[1] - 128)
    position = renderer.camera.world_to_screen(simulation.life.position)
    center = occupied.get_at(tuple(round(value) for value in position))[:3]
    assert center != ChamberRenderer.CHAMBER
    assert pixels(occupied.subsurface(chamber)) != pixels(empty.subsurface(chamber))
    assert occupied.get_at((0, 0))[:3] == ChamberRenderer.BACKGROUND
    assert occupied.get_at((24, chamber.centery))[:3] == ChamberRenderer.BORDER


def test_renderer_reuses_surfaces_and_rebuilds_them_on_resize() -> None:
    organism = Symbiote(17)
    renderer = SymbioteRenderer()
    surface = pygame.Surface((1280, 720))
    chamber = surface.get_rect()
    renderer.draw(surface, chamber, organism)
    canvas, glow = renderer._canvas, renderer._core_glow
    organism.update(1 / 60)
    renderer.draw(surface, chamber, organism)
    assert renderer._canvas is canvas
    assert renderer._core_glow is glow
    renderer.draw(surface, pygame.Rect(0, 0, 480, 320), organism)
    assert renderer._canvas is not canvas


def test_renderer_preserves_destination_clip() -> None:
    surface = pygame.Surface((640, 640))
    surface.fill((1, 2, 3))
    clip = pygame.Rect(100, 100, 400, 400)
    surface.set_clip(clip)
    SymbioteRenderer().draw(surface, surface.get_rect(), Symbiote(17))
    assert surface.get_clip() == clip
    assert surface.get_at((50, 50))[:3] == (1, 2, 3)


@pytest.mark.parametrize("seed", [0, 2, 3, 4, 8, 17])
def test_body_and_appendages_form_one_connected_silhouette(seed) -> None:
    renderer = SymbioteRenderer()
    surface = pygame.Surface((800, 800), pygame.SRCALPHA)
    organism = Symbiote(seed)
    organism.update(2.5)
    renderer.draw(surface, surface.get_rect(), organism)
    components = pygame.mask.from_surface(surface, threshold=40).get_bounding_rects()
    significant = [rect for rect in components if rect.width * rect.height > 16]
    assert len(significant) == 1


def test_renderer_rebuilds_tissue_effects_when_appearance_changes() -> None:
    renderer = SymbioteRenderer()
    surface = pygame.Surface((800, 800))
    renderer.draw(surface, surface.get_rect(), Symbiote(17))
    old_layer, old_mask, old_glow = renderer._layer, renderer._mask, renderer._core_glow
    renderer.draw(surface, surface.get_rect(), Symbiote(13))
    assert renderer._layer is not old_layer
    assert renderer._mask is not old_mask
    assert renderer._core_glow is not old_glow


def test_rendering_after_reset_has_no_cached_tissue_or_mask_ghosts() -> None:
    renderer = ChamberRenderer()
    simulation = Simulation(17)
    surface = pygame.Surface((1280, 720))
    renderer.draw(surface, simulation)
    initial = pixels(surface)
    simulation.update(9)
    renderer.draw(surface, simulation)
    simulation.reset()
    renderer.draw(surface, simulation)
    assert pixels(surface) == initial
    ChamberRenderer().draw(surface, simulation)
    assert pixels(surface) == initial


@pytest.mark.parametrize("seed", [0, 2, 3, 4, 8])
def test_trailing_tissue_has_flat_ink_color_without_a_tube_highlight(seed) -> None:
    organism = Symbiote(seed)
    renderer = SymbioteRenderer()
    surface = pygame.Surface((800, 800), pygame.SRCALPHA)
    renderer.draw(surface, surface.get_rect(), organism)
    appearance, pose = organism.appearance, organism.pose()
    radius = renderer._cache_key[1]
    cosine, sine = cos(pose.rotation), sin(pose.rotation)

    def project(point):
        x, y = point
        return (round(renderer._origin + radius * (x * cosine - y * sine)),
                round(renderer._origin + radius * (x * sine + y * cosine)))

    checked = 0
    for tendril in appearance.tendrils:
        mesh = appendage_mesh(body_anchors(appearance, pose), tendril, appearance, pose, renderer.TENDRIL_SEGMENTS)
        for index in range(16, 23):
            center, edge = mesh.centerline[index], mesh.left[index]
            if mesh.widths[index] * radius < 4:
                continue
            samples = [project((center[0] + (edge[0] - center[0]) * offset,
                                center[1] + (edge[1] - center[1]) * offset))
                       for offset in (-0.25, 0, 0.25)]
            if any(renderer._mask.get_at(point).a for point in samples):
                continue
            # Compare cross-sections outside the body/glow, where the old
            # renderer drew off-center directional highlight strips.
            assert all(renderer._canvas.get_at(point) == (*appearance.tissue_color, 205) for point in samples)
            checked += 1
    assert checked > 0
