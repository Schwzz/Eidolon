"""Own the desktop lifecycle, events, and coordination of updates and drawing."""

from time import perf_counter

import pygame

from eidolon.config import AppConfig, MIN_WINDOW_HEIGHT, MIN_WINDOW_WIDTH
from eidolon.rendering import ChamberRenderer
from eidolon.simulation import Simulation
from eidolon.timing import FixedStepClock


class Application:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config if config is not None else AppConfig()
        self.simulation = Simulation()

    def run(self) -> None:
        """Run until the window closes or Escape is pressed; always release SDL."""
        try:
            # No audio is needed in Phase 0; avoid initializing unused subsystems.
            pygame.display.init()
            pygame.font.init()
            surface = pygame.display.set_mode(
                (self.config.width, self.config.height), pygame.RESIZABLE
            )
            pygame.display.set_caption("EIDOLON — Simulation Chamber")
            renderer = ChamberRenderer()
            simulation_clock = FixedStepClock(self.config)
            frame_limiter = pygame.time.Clock()
            previous_time = perf_counter()

            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.type == pygame.VIDEORESIZE:
                        surface = pygame.display.set_mode(
                            (
                                max(MIN_WINDOW_WIDTH, event.w),
                                max(MIN_WINDOW_HEIGHT, event.h),
                            ),
                            pygame.RESIZABLE,
                        )
                if not running:
                    break

                now = perf_counter()
                simulation_clock.advance(now - previous_time, self.simulation.update)
                previous_time = now
                renderer.draw(surface)
                pygame.display.flip()
                frame_limiter.tick(self.config.max_fps)
        finally:
            pygame.quit()
