import pygame
import pytest

from eidolon.organism import Symbiote
from eidolon.organism_rendering import SymbioteRenderer
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
    center = occupied.get_at(chamber.center)[:3]
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
