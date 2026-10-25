"""Own the desktop lifecycle, events, and coordination of updates and drawing."""

from time import perf_counter

import pygame

from eidolon.config import AppConfig, MIN_WINDOW_HEIGHT, MIN_WINDOW_WIDTH
from eidolon.behavior import Controller
from eidolon.rendering import ChamberRenderer
from eidolon.simulation import Simulation
from eidolon.timing import FixedStepClock


class Application:
    def __init__(self, config: AppConfig | None = None, controller: Controller | None = None) -> None:
        self.config = config if config is not None else AppConfig()
        if controller is None and self.config.controller == "neural":
            from eidolon.neural_controller import NeuralController

            controller = NeuralController.from_seed(self.config.seed)
        self.simulation = Simulation(self.config.seed, self.config.world, self.config.life, controller,
                                     self.config.physics, self.config.muscles, self.config.muscle_test, self.config.body)

    def _muscle_key(self, event) -> None:
        if self.simulation.muscle_test != "manual":
            return
        if pygame.K_1 <= event.key <= pygame.K_8:
            values = list(self.simulation.colony.muscles.targets)
            values[event.key - pygame.K_1] = 1.0 if event.type == pygame.KEYDOWN else 0.0
            self.simulation.set_muscle_activations(values)
        elif event.key == pygame.K_0:
            self.simulation.relax_muscles()

    def _pointer_event(self, event, size: tuple[int, int]) -> None:
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.simulation.release_grab()
        elif ((event.type == pygame.MOUSEBUTTONDOWN and event.button == 1)
              or (event.type == pygame.MOUSEMOTION and self.simulation.colony.grab is not None)):
            world = self.simulation.world_config
            camera = ChamberRenderer.camera_for_window(size, (world.width, world.height))
            position = camera.screen_to_world(event.pos)
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.simulation.begin_grab(position, padding=8 / camera.scale)
            else:
                self.simulation.move_grab(position)

    def run(self) -> None:
        """Run until the window closes or Escape is pressed; always release SDL."""
        try:
            # Avoid initializing unused audio subsystems.
            pygame.display.init()
            pygame.font.init()
            surface = pygame.display.set_mode(
                (self.config.width, self.config.height), pygame.RESIZABLE
            )
            pygame.display.set_caption("EIDOLON — Simulation Chamber")
            renderer = ChamberRenderer(show_nodes=self.config.show_nodes)
            simulation_clock = FixedStepClock(self.config)
            frame_limiter = pygame.time.Clock()
            previous_time = perf_counter()

            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        self._muscle_key(event)
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_SPACE:
                            self.simulation.paused = not self.simulation.paused
                        elif event.key == pygame.K_r:
                            self.simulation.reset()
                        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                            self.simulation.set_speed(
                                max(0.25, self.simulation.speed - 0.25)
                            )
                        elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                            self.simulation.set_speed(
                                min(2.0, self.simulation.speed + 0.25)
                            )
                    elif event.type == pygame.KEYUP:
                        self._muscle_key(event)
                    elif event.type == pygame.VIDEORESIZE:
                        self.simulation.release_grab()
                        self.simulation.relax_muscles()
                        surface = pygame.display.set_mode(
                            (
                                max(MIN_WINDOW_WIDTH, event.w),
                                max(MIN_WINDOW_HEIGHT, event.h),
                            ),
                            pygame.RESIZABLE,
                        )
                    elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
                        self._pointer_event(event, surface.get_size())
                    elif event.type in (pygame.WINDOWFOCUSLOST, pygame.WINDOWLEAVE):
                        self.simulation.release_grab()
                        self.simulation.relax_muscles()
                if not running:
                    break

                now = perf_counter()
                simulation_clock.advance(now - previous_time, self.simulation.update)
                previous_time = now
                renderer.draw(surface, self.simulation, simulation_clock.interpolation_alpha)
                pygame.display.flip()
                frame_limiter.tick(self.config.max_fps)
        finally:
            self.simulation.release_grab()
            self.simulation.relax_muscles()
            pygame.quit()
