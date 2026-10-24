import pygame
import pytest

from eidolon.app import Application
from eidolon.config import AppConfig
from eidolon.rendering import ChamberRenderer


@pytest.fixture(autouse=True)
def cleanup_pygame():
    yield
    pygame.quit()


@pytest.mark.parametrize("size", [(480, 320), (1100, 720), (1600, 900)])
def test_renderer_draws_empty_dark_chamber(size) -> None:
    pygame.font.init()
    surface = pygame.Surface(size)
    ChamberRenderer().draw(surface)
    assert surface.get_at((0, 0))[:3] == ChamberRenderer.BACKGROUND
    assert surface.get_at((size[0] // 2, size[1] // 2))[:3] == ChamberRenderer.CHAMBER
    assert surface.get_at((24, size[1] // 2))[:3] == ChamberRenderer.BORDER


@pytest.mark.parametrize(
    ("requested", "expected"),
    [((800, 600), (800, 600)), ((100, 100), (480, 320))],
)
def test_launch_resize_update_render_and_close(monkeypatch, requested, expected) -> None:
    app = Application(AppConfig(max_fps=0))
    events = iter([
        [],
        [pygame.event.Event(pygame.VIDEORESIZE, w=requested[0], h=requested[1])],
        [pygame.event.Event(pygame.QUIT)],
    ])
    times = iter([0.0, 1 / 30, 1 / 30 + 1 / 120])
    monkeypatch.setattr(pygame.event, "get", lambda: next(events))
    monkeypatch.setattr("eidolon.app.perf_counter", lambda: next(times))
    frames = []
    draw = ChamberRenderer.draw

    def record_draw(renderer, surface, simulation=None, alpha=1.0):
        draw(renderer, surface, simulation, alpha)
        frames.append((surface.get_size(), app.simulation.steps))

    monkeypatch.setattr(ChamberRenderer, "draw", record_draw)
    app.run()
    assert frames == [((1280, 720), 2), (expected, 2)]
    assert app.simulation.elapsed_seconds == pytest.approx(1 / 30)
    assert not pygame.display.get_init()
    assert not pygame.font.get_init()


@pytest.mark.parametrize("max_fps", [0, 30, 144])
def test_frame_limiter_uses_configured_limit(monkeypatch, max_fps) -> None:
    limits = []

    class FrameLimiter:
        def tick(self, limit):
            limits.append(limit)

    events = iter([[], [pygame.event.Event(pygame.QUIT)]])
    monkeypatch.setattr(pygame.event, "get", lambda: next(events))
    monkeypatch.setattr(pygame.time, "Clock", FrameLimiter)
    Application(AppConfig(max_fps=max_fps)).run()
    assert limits == [max_fps]


def test_escape_exits_before_drawing(monkeypatch) -> None:
    monkeypatch.setattr(
        pygame.event, "get",
        lambda: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)],
    )
    app = Application()
    app.run()
    assert app.simulation.steps == 0
    assert not pygame.display.get_init()


def test_cleanup_also_runs_when_rendering_fails(monkeypatch) -> None:
    monkeypatch.setattr(pygame.event, "get", lambda: [])

    def fail_draw(renderer, surface, simulation=None, alpha=1.0):
        raise RuntimeError("render failure")

    monkeypatch.setattr(ChamberRenderer, "draw", fail_draw)
    with pytest.raises(RuntimeError, match="render failure"):
        Application().run()
    assert not pygame.display.get_init()
    assert not pygame.font.get_init()


def test_animation_controls_pause_speed_and_restart(monkeypatch) -> None:
    app = Application(AppConfig(seed=123, max_fps=0))
    key = lambda value: pygame.event.Event(pygame.KEYDOWN, key=value)
    events = iter([
        [], [key(pygame.K_SPACE)], [key(pygame.K_SPACE), key(pygame.K_EQUALS)],
        [key(pygame.K_r)], [pygame.event.Event(pygame.QUIT)],
    ])
    times = iter([0.0, 1 / 60, 2 / 60, 3 / 60, 4 / 60])
    monkeypatch.setattr(pygame.event, "get", lambda: next(events))
    monkeypatch.setattr("eidolon.app.perf_counter", lambda: next(times))
    states = []
    draw = ChamberRenderer.draw

    def record_draw(renderer, surface, simulation, alpha):
        draw(renderer, surface, simulation, alpha)
        states.append((simulation.paused, simulation.animation_speed, simulation.organism.age_seconds))

    monkeypatch.setattr(ChamberRenderer, "draw", record_draw)
    app.run()
    assert states[0] == (False, 1.0, pytest.approx(1 / 60))
    assert states[1] == (True, 1.0, pytest.approx(1 / 60))
    assert states[2] == (False, 1.25, pytest.approx(2.25 / 60))
    assert states[3] == (False, 1.25, pytest.approx(1.25 / 60))
    assert app.simulation.organism.appearance.seed == 123


def test_animation_speed_keys_stay_within_bounds(monkeypatch) -> None:
    key = lambda value: pygame.event.Event(pygame.KEYDOWN, key=value)
    events = iter([
        [key(pygame.K_MINUS)] * 20,
        [key(pygame.K_KP_PLUS)] * 20,
        [pygame.event.Event(pygame.QUIT)],
    ])
    monkeypatch.setattr(pygame.event, "get", lambda: next(events))
    speeds = []

    def record_draw(renderer, surface, simulation, alpha):
        speeds.append(simulation.animation_speed)

    monkeypatch.setattr(ChamberRenderer, "draw", record_draw)
    Application(AppConfig(max_fps=0)).run()
    assert speeds == [0.25, 2.0]
