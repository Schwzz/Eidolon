"""Chamber layout and controls; organism drawing has its own renderer."""

import pygame

from eidolon.colony_rendering import ColonyRenderer
from eidolon.camera import Camera
from eidolon.simulation import Simulation


class ChamberRenderer:
    BACKGROUND = (10, 14, 20)
    CHAMBER = (15, 22, 29)
    BORDER = (39, 55, 64)
    TEXT = (205, 220, 222)
    MUTED = (124, 148, 155)
    ACCENT = (100, 193, 178)

    def __init__(self, *, show_nodes: bool = False) -> None:
        self.show_nodes = show_nodes
        # Pygame's bundled font avoids external assets and platform font paths.
        self._title_font = pygame.font.Font(None, 32)
        self._title = self._title_font.render("EIDOLON", True, self.TEXT)
        self._label_font = pygame.font.Font(None, 20)
        self._muscle_font = pygame.font.Font(None, 15)
        self._colony_renderer = ColonyRenderer()
        self._labels: dict[tuple[str, tuple[int, int, int]], pygame.Surface] = {}
        self.camera: Camera | None = None

    def _label(self, text: str, color: tuple[int, int, int]) -> pygame.Surface:
        key = (text, color)
        if key not in self._labels:
            if len(self._labels) >= 128:
                self._labels.clear()  # Dynamic age/energy labels must not accumulate forever.
            self._labels[key] = self._label_font.render(text, True, color)
        return self._labels[key]

    @staticmethod
    def camera_for_window(size: tuple[int, int], world_size: tuple[float, float]) -> Camera:
        width, height = size
        return Camera.fit(world_size, (34, 92, width - 68, height - 196))

    def draw(
        self, surface: pygame.Surface, simulation: Simulation | None = None,
        alpha: float = 1.0,
    ) -> None:
        """Recompute layout from the current window size on every frame."""
        width, height = surface.get_size()
        surface.fill(self.BACKGROUND)
        surface.blit(self._title, (24, 19))
        surface.blit(
            self._label("SIMULATION CHAMBER", self.MUTED), (24, 50)
        )
        live = simulation is not None and getattr(simulation, "is_live_learning", False)
        mode = ("LIVE LEARNING" if live else f"MUSCLE {simulation.muscle_test.upper()}" if simulation is not None and simulation.muscle_test
                else simulation.controller_name.upper() if simulation is not None else "")
        caption = "PHASE 2B.2" + (f" | {mode}" if mode else "")
        phase = self._label(caption, self.ACCENT)
        surface.blit(phase, (width - phase.get_width() - 24, 29))

        chamber = pygame.Rect(24, 82, max(1, width - 48), max(1, height - 128))
        pygame.draw.rect(surface, self.CHAMBER, chamber, border_radius=10)
        pygame.draw.rect(surface, self.BORDER, chamber, width=1, border_radius=10)

        if simulation is not None:
            world = simulation.world.config
            self.camera = self.camera_for_window(surface.get_size(), (world.width, world.height))
            camera = self.camera
            world_rect = pygame.Rect(*camera.offset, world.width * camera.scale, world.height * camera.scale)
            pygame.draw.rect(surface, (30, 44, 48), world_rect, width=1)
            for food in simulation.world.foods:
                position = tuple(round(value) for value in camera.world_to_screen(food.sample(1.0 if simulation.paused else alpha)))
                radius = max(1, round(food.radius * camera.scale))
                pygame.draw.circle(surface, (36, 53, 39), position, radius + 2)
                pygame.draw.circle(surface, (161, 191, 119), position, radius)
            interpolation = 1.0 if simulation.paused else alpha
            self._colony_renderer.draw(
                surface, world_rect, simulation.colony, camera,
                simulation.organism.appearance, interpolation, simulation.organism.age_seconds,
            )
            if self.show_nodes:
                for x, y, _ in simulation.colony.sample(interpolation):
                    pygame.draw.circle(surface, self.ACCENT,
                                       tuple(round(v) for v in camera.world_to_screen((x, y))), 2, 1)
            if live:
                notice = self._label(simulation.notice, self.MUTED)
                surface.blit(notice, (40, 100), (0, 0, width - 80, notice.get_height()))
            life = simulation.life
            energy = f"ENERGY {life.energy:.0f}/{life.max_energy:.0f}   AGE {life.age_seconds:.0f}s"
            replay = getattr(simulation, "is_replay", False)
            if replay:
                metrics = simulation.environment.metrics()
                energy = f"REWARD {metrics['reward']:+.3f}   MUSCLE E {metrics['muscle_energy']:.2f}"
            if live:
                metrics = simulation.environment.metrics()
                energy = f"LIVE G{simulation.generation_number} A{simulation.active_attempt}  R {metrics['reward']:+.3f}"
            surface.blit(self._label(energy, self.MUTED), (40, height - 91))
            status = ("UNSTABLE" if not metrics["stable"] else "COMPLETE" if simulation.environment.done else "REPLAY") if replay else "ALIVE" if life.alive else "DEAD"
            if live:
                status = f"BEST {simulation.best_score:+.3f}" if simulation.best_score is not None else "BEST pending"
            status_label = self._label(status, self.ACCENT if life.alive and status != "UNSTABLE" else (184, 106, 118))
            surface.blit(status_label, (width - status_label.get_width() - 40, height - 91))
            growth = f"SIZE {life.body_scale:.2f}x   FOOD {len(simulation.world.foods)}"
            if replay:
                growth = ("INTERACTIVE: NOT BENCHMARK" if simulation.intervened else
                          f"DX {simulation.muscle_displacement:+.2f}   TIME {simulation.elapsed_seconds:.1f}s")
            if live:
                growth = f"C{simulation.active_candidate + 1}/{simulation.trainer.config.population} S{simulation.active_seed} {simulation.status} | FOOD/STARVATION OFF"
            surface.blit(self._label(growth, self.MUTED), (40, height - 68))
            speed = f"{'PAUSED' if simulation.paused else 'RUNNING'} {simulation.speed:.2f}x"
            if replay and simulation.environment.done:
                speed = "FINISHED | R Reset"
            speed_label = self._label(speed, self.ACCENT)
            surface.blit(speed_label, (width - speed_label.get_width() - 40, height - 68))
            controls = "LMB Grab  SPACE Pause  R Reset  +/-"
            self._draw_muscles(surface, simulation, width, height)
            if live:
                controls = "SPACE Pause  S Stop/Continue  R Retry  N New Run"
            elif simulation.muscle_test == "manual" and width >= 800:
                controls += "  1-8 Muscles  0 Relax"
        else:
            controls = "EMPTY CHAMBER"
        surface.blit(self._label(controls, self.MUTED), (24, height - 29))
        hint = self._label("ESC  Exit", self.MUTED)
        surface.blit(hint, (width - hint.get_width() - 24, height - 29))

    def _draw_muscles(self, surface, simulation, width, height):
        edge = max(20, min(36, (width - 80) // 8))
        for region, (activation, target) in enumerate(zip(simulation.colony.muscles.activations, simulation.colony.muscles.targets)):
            motion = "+" if target > activation + 1e-4 else "-" if target < activation - 1e-4 else "="
            color = self.ACCENT if motion != "-" else (176, 116, 180)
            rect = pygame.Rect(40 + region * edge, height - 46, edge - 2, 14)
            pygame.draw.rect(surface, self.BORDER, rect)
            pygame.draw.rect(surface, color, (rect.x, rect.y, round(rect.width * activation), rect.height))
            label = self._muscle_font.render(f"{region + 1}{motion}{activation * 100:02.0f}", True, self.TEXT)
            surface.blit(label, (rect.x + 1, rect.y + 1))
        if width >= 600:
            surface.blit(self._muscle_font.render("actual %  +contract -relax", True, self.MUTED), (48 + edge * 8, height - 44))
