from dataclasses import asdict

import pygame
import pytest

from eidolon.camera import Camera
from eidolon.config import AppConfig, LifeConfig, WorldConfig
from eidolon.rendering import ChamberRenderer
from eidolon.colony_rendering import ColonyRenderer
from eidolon.simulation import Simulation


def test_camera_fits_without_stretching_and_roundtrips_world_coordinates() -> None:
    camera = Camera.fit((1200, 800), (24, 80, 1000, 500))
    assert camera.scale == 0.625
    assert camera.offset == (149, 80)
    for position in ((0, 0), (600, 400), (1200, 800), (31.25, 47.5)):
        assert camera.screen_to_world(camera.world_to_screen(position)) == pytest.approx(position)


def test_resizing_changes_only_camera_and_pixel_size_not_physical_size_or_world(monkeypatch) -> None:
    pygame.font.init()
    try:
        simulation = Simulation()
        simulation.update(2)
        initial_life = asdict(simulation.life)
        initial_food = tuple(simulation.world.foods)
        initial_rng = simulation.world._random.getstate()
        initial_time = simulation.world.elapsed_seconds
        initial_nodes = [asdict(node) for node in simulation.colony.nodes]
        renderer = ChamberRenderer()
        radii = []
        draw = ColonyRenderer.draw

        def record(instance, surface, chamber, colony, camera, appearance, alpha=1.0, seconds=0.0):
            positions = [camera.world_to_screen(node.position) for node in colony.nodes]
            span = max(point[0] for point in positions) - min(point[0] for point in positions)
            radii.append((span, camera.scale))
            draw(instance, surface, chamber, colony, camera, appearance, alpha, seconds)

        monkeypatch.setattr(ColonyRenderer, "draw", record)
        for size in ((1280, 720), (480, 320), (1600, 900)):
            renderer.draw(pygame.Surface(size), simulation)
        assert len({scale for _, scale in radii}) == 3
        world_span = max(node.position[0] for node in simulation.colony.nodes) - min(node.position[0] for node in simulation.colony.nodes)
        for pixel_span, camera_scale in radii:
            assert pixel_span / camera_scale == pytest.approx(world_span)
        assert [asdict(node) for node in simulation.colony.nodes] == initial_nodes
        assert asdict(simulation.life) == initial_life
        assert tuple(simulation.world.foods) == initial_food
        assert simulation.world._random.getstate() == initial_rng
        assert simulation.world.elapsed_seconds == initial_time
        assert (simulation.world.config.width, simulation.world.config.height) == (1200, 800)
    finally:
        pygame.quit()


def test_growth_changes_world_radius_without_changing_camera_scale() -> None:
    from eidolon.world import Food
    pygame.font.init()
    try:
        simulation = Simulation(world_config=WorldConfig(initial_food=0, regeneration_interval=0),
                                life_config=LifeConfig(movement_speed=0, metabolism=0, growth_rate=0.1))
        renderer = ChamberRenderer()
        surface = pygame.Surface((1280, 720))
        renderer.draw(surface, simulation)
        camera_scale = renderer.camera.scale
        radius = simulation.life.physical_radius
        initial_span = max(node.position[0] for node in simulation.colony.nodes) - min(node.position[0] for node in simulation.colony.nodes)
        simulation.world.foods = [Food(1000, simulation.life.position, 18, 5)]
        simulation.update(1)
        renderer.draw(surface, simulation)
        assert simulation.life.physical_radius == pytest.approx(radius * 1.1)
        assert renderer.camera.scale == camera_scale
        final_span = max(node.position[0] for node in simulation.colony.nodes) - min(node.position[0] for node in simulation.colony.nodes)
        assert final_span > initial_span
    finally:
        pygame.quit()


def test_world_configuration_is_independent_of_window_configuration() -> None:
    from eidolon.app import Application
    config = AppConfig(width=800, height=600, world=WorldConfig(width=1600, height=1000), life=LifeConfig())
    app = Application(config)
    assert (app.simulation.world.config.width, app.simulation.world.config.height) == (1600, 1000)
    assert app.simulation.life.physical_radius == 24


def test_dynamic_hud_text_has_a_bounded_cache() -> None:
    pygame.font.init()
    try:
        renderer = ChamberRenderer()
        for age in range(1000):
            renderer._label(f"AGE {age}s", renderer.MUTED)
        assert len(renderer._labels) <= 128
    finally:
        pygame.quit()


@pytest.mark.parametrize("world,viewport", [((0, 10), (0, 0, 100, 100)), ((10, 10), (0, 0, -1, 100)), ((10, float("inf")), (0, 0, 100, 100))])
def test_invalid_camera_dimensions_are_rejected(world, viewport) -> None:
    with pytest.raises(ValueError):
        Camera.fit(world, viewport)
