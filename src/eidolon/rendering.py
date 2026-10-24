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

    def __init__(self) -> None:
        # Pygame's bundled font avoids external assets and platform font paths.
        self._title_font = pygame.font.Font(None, 32)
        self._title = self._title_font.render("EIDOLON", True, self.TEXT)
        self._label_font = pygame.font.Font(None, 20)
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
        phase = self._label("PHASE 1C-C", self.ACCENT)
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
                position = tuple(round(value) for value in camera.world_to_screen(food.position))
                radius = max(1, round(food.radius * camera.scale))
                pygame.draw.circle(surface, (36, 53, 39), position, radius + 2)
                pygame.draw.circle(surface, (161, 191, 119), position, radius)
            interpolation = 1.0 if simulation.paused or not simulation.life.alive else alpha
            self._colony_renderer.draw(
                surface, world_rect, simulation.colony, camera,
                simulation.organism.appearance, interpolation, simulation.organism.age_seconds,
            )
            life = simulation.life
            energy = f"ENERGY {life.energy:.0f}/{life.max_energy:.0f}   AGE {life.age_seconds:.0f}s"
            surface.blit(self._label(energy, self.MUTED), (40, height - 91))
            status_label = self._label("ALIVE" if life.alive else "DEAD", self.ACCENT if life.alive else (184, 106, 118))
            surface.blit(status_label, (width - status_label.get_width() - 40, height - 91))
            growth = f"SIZE {life.body_scale:.2f}x   FOOD {len(simulation.world.foods)}"
            surface.blit(self._label(growth, self.MUTED), (40, height - 68))
            speed = f"{'PAUSED' if simulation.paused else 'RUNNING'} {simulation.speed:.2f}x"
            speed_label = self._label(speed, self.ACCENT)
            surface.blit(speed_label, (width - speed_label.get_width() - 40, height - 68))
            controls = "LMB Grab  SPACE Pause  R Reset  +/-"
        else:
            controls = "EMPTY CHAMBER"
        surface.blit(self._label(controls, self.MUTED), (24, height - 29))
        hint = self._label("ESC  Exit", self.MUTED)
        surface.blit(hint, (width - hint.get_width() - 24, height - 29))
