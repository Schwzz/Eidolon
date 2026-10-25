"""Own the desktop lifecycle, events, and coordination of updates and drawing."""

from time import perf_counter
from pathlib import Path
from dataclasses import replace

import pygame
import sys

from eidolon.config import AppConfig, MIN_WINDOW_HEIGHT, MIN_WINDOW_WIDTH
from eidolon.behavior import Controller
from eidolon.rendering import ChamberRenderer
from eidolon.simulation import Simulation
from eidolon.timing import FixedStepClock
from eidolon.laboratory import LaboratoryRenderer, VIEWS


class Application:
    def __init__(self, config: AppConfig | None = None, controller: Controller | None = None, *, brain=None) -> None:
        self.config = config if config is not None else AppConfig()
        self.view = self.config.view
        self.laboratory = None
        self.saved_metrics = None
        self._learning = self._survival = None
        self._brain = brain
        if self.config.training_path:
            from eidolon.evolution import load_metrics
            path = Path(self.config.training_path)
            self.metrics_path = path / "metrics.json" if path.is_dir() else path
            self.saved_metrics = load_metrics(self.metrics_path)
        normal_requested = (self.config.brain_path is not None or controller is not None
                            or self.config.controller != "rules" or self.config.muscle_test is not None
                            or self.config.training_path is not None)
        self.mode = "survival" if self.config.mode == "survival" or (self.config.mode == "auto" and normal_requested) else "live"
        self._steering_controller = controller
        self._switch_mode(self.mode)

    def _make_survival(self):
        if self.config.brain_path:
            from eidolon.muscle_controller import NeuralMuscleController
            brain = self._brain if self._brain is not None else NeuralMuscleController.load(self.config.brain_path)
            settings = brain.episode
            return Simulation(self.config.seed,
                replace(self.config.world, width=settings.world_size[0], height=settings.world_size[1]),
                replace(self.config.life, base_radius=settings.base_radius),
                physics_config=settings.physics, muscle_config=settings.muscles,
                body=self.config.body, muscle_controller=brain)
        controller = self._steering_controller
        if controller is None and self.config.controller == "neural":
            from eidolon.neural_controller import NeuralController

            controller = NeuralController.from_seed(self.config.seed)
        return Simulation(self.config.seed, self.config.world, self.config.life, controller,
                          self.config.physics, self.config.muscles, self.config.muscle_test, self.config.body)

    def _switch_mode(self, mode):
        if hasattr(self, "simulation"):
            self.simulation.release_grab()
        if mode == "live":
            if self._learning is None:
                from eidolon.live_learning import LiveLearning
                from eidolon.evolution import EvolutionConfig
                from eidolon.training import EpisodeConfig
                config = EvolutionConfig(seed=self.config.seed, population=6, elites=2,
                    seeds=(self.config.seed, self.config.seed + 1),
                    episode=EpisodeConfig(seed=self.config.seed, duration=6, target_distance=20))
                self._learning = LiveLearning(self.config.live_run, config, budget_ms=self.config.live_budget_ms)
            self.simulation = self._learning
        else:
            if self._survival is None:
                self._survival = self._make_survival()
            self.simulation = self._survival
        self.mode = mode

    def close(self):
        if self._learning is not None:
            try:
                self._learning.close()
            except (OSError, ValueError) as error:
                print(f"Live learning save failed: {error}", file=sys.stderr)

    def _mode_command(self, mode):
        try:
            self._switch_mode(mode)
            self.laboratory.mode_error = None
        except (OSError, ValueError) as error:
            self.laboratory.mode_error = str(error)

    def _learning_command(self, command):
        try:
            command()
        except (OSError, ValueError) as error:
            self.simulation.save_error = str(error)
            self.simulation.stopped = True

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
            if self.view not in ("simulation", "split"):
                return
            if self.view == "split":
                if event.pos[0] >= size[0] // 2 and self.simulation.colony.grab is None:
                    return
                size = (size[0] // 2, size[1])
            world = self.simulation.world_config
            camera = ChamberRenderer.camera_for_window(size, (world.width, world.height))
            position = camera.screen_to_world(event.pos)
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.simulation.begin_grab(position, padding=8 / camera.scale)
            else:
                self.simulation.move_grab(position)

    def _switch_view(self, view):
        self.view = view
        if self.laboratory is not None:
            self.laboratory.view = view
        self.simulation.release_grab()

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
            self.laboratory = LaboratoryRenderer(renderer, view=self.view, metrics=self.saved_metrics)
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
                        if event.key in (pygame.K_F1, pygame.K_F2, pygame.K_F3, pygame.K_F4):
                            self._switch_view(VIEWS[event.key - pygame.K_F1])
                        elif event.key == pygame.K_F5 and self.config.training_path:
                            from eidolon.evolution import load_metrics
                            try:
                                self.saved_metrics = load_metrics(self.metrics_path)
                                self.laboratory.metrics = self.saved_metrics
                                self.laboratory.metrics_error = None
                            except (OSError, ValueError) as error:
                                self.laboratory.metrics_error = str(error)
                        elif event.key == pygame.K_l:
                            self._mode_command("survival" if self.mode == "live" else "live")
                        elif event.key == pygame.K_s and self.mode == "live":
                            self._learning_command(self.simulation.toggle_stopped)
                        elif event.key == pygame.K_n and self.mode == "live":
                            self._learning_command(self.simulation.restart_learning)
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
                        mode = (self.laboratory.mode_at(event.pos, surface.get_size())
                                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 else None)
                        if mode:
                            self._mode_command(mode)
                        elif (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                                and self.laboratory.select_at(event.pos, surface.get_size())):
                            self._switch_view(self.laboratory.view)
                        else:
                            self._pointer_event(event, surface.get_size())
                    elif event.type in (pygame.WINDOWFOCUSLOST, pygame.WINDOWLEAVE):
                        self.simulation.release_grab()
                        self.simulation.relax_muscles()
                if not running:
                    break

                now = perf_counter()
                if self.mode == "live":
                    self.simulation.update(min(self.config.max_frame_seconds, now - previous_time))
                else:
                    if self._learning is not None:
                        self._learning.poll_save()
                    simulation_clock.advance(now - previous_time, self.simulation.update)
                previous_time = now
                self.laboratory.live_session = self._learning
                alpha = 1.0 if self.mode == "live" else simulation_clock.interpolation_alpha
                self.laboratory.draw(surface, self.simulation, alpha)
                pygame.display.flip()
                frame_limiter.tick(self.config.max_fps)
        finally:
            self.simulation.release_grab()
            self.simulation.relax_muscles()
            self.close()
            pygame.quit()
