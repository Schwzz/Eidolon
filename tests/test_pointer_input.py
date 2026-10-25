from dataclasses import asdict

import pygame
import pytest

from eidolon.app import Application
from eidolon.config import AppConfig
from eidolon.rendering import ChamberRenderer


def click_node(app, size):
    camera = ChamberRenderer.camera_for_window(size, (1200, 800))
    node = app.simulation.colony.nodes[0]
    position = tuple(round(value) for value in camera.world_to_screen(node.position))
    event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=position)
    app._pointer_event(event, size)
    return camera, node, position


@pytest.mark.parametrize("size", [(480, 320), (1280, 720), (1600, 900)])
def test_pointer_uses_current_fit_camera_even_before_first_draw(size) -> None:
    app = Application(AppConfig(mode="survival"))
    original = [asdict(node) for node in app.simulation.colony.nodes]
    camera, node, position = click_node(app, size)
    assert app.simulation.colony.grab.node == node.identifier
    assert app.simulation.colony.grab.target == pytest.approx(camera.screen_to_world(position))
    target = (position[0] + 20, position[1] + 10)
    app._pointer_event(pygame.event.Event(pygame.MOUSEMOTION, pos=target, buttons=(1, 0, 0)), size)
    assert app.simulation.colony.grab.target == pytest.approx(camera.screen_to_world(target))
    assert [asdict(node) for node in app.simulation.colony.nodes] == original
    app._pointer_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(-10, -10)), size)
    assert app.simulation.colony.grab is None


def test_letterbox_hud_right_click_and_paused_click_do_not_grab() -> None:
    app = Application(AppConfig(mode="survival"))
    for position in ((10, 10), (100, 300), (640, 690)):
        app._pointer_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=position), (1280, 720))
        assert app.simulation.colony.grab is None
    camera = ChamberRenderer.camera_for_window((1280, 720), (1200, 800))
    position = camera.world_to_screen(app.simulation.colony.nodes[0].position)
    app._pointer_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=position), (1280, 720))
    assert app.simulation.colony.grab is None
    app.simulation.paused = True
    click_node(app, (1280, 720))
    assert app.simulation.colony.grab is None


@pytest.mark.parametrize("second_event", [
    pygame.event.Event(pygame.VIDEORESIZE, w=800, h=600),
    pygame.event.Event(pygame.WINDOWFOCUSLOST),
    pygame.event.Event(pygame.WINDOWLEAVE),
])
def test_resize_and_lost_focus_release_the_grab_without_regrabbing(monkeypatch, second_event) -> None:
    app = Application(AppConfig(mode="survival", max_fps=0))
    camera = ChamberRenderer.camera_for_window((1280, 720), (1200, 800))
    position = camera.world_to_screen(app.simulation.colony.nodes[0].position)
    events = iter([
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=position)],
        [second_event], [pygame.event.Event(pygame.QUIT)],
    ])
    monkeypatch.setattr(pygame.event, "get", lambda: next(events))
    flags = []
    draw = ChamberRenderer.draw

    def observe(renderer, surface, simulation, alpha):
        draw(renderer, surface, simulation, alpha)
        flags.append(simulation.colony.grab is not None)

    monkeypatch.setattr(ChamberRenderer, "draw", observe)
    app.run()
    assert flags == [True, False]
    assert app.simulation.colony.grab is None
    assert not pygame.display.get_init()
