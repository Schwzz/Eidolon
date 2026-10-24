"""Chamber layout and controls; organism drawing has its own renderer."""

import pygame

from eidolon.organism_rendering import SymbioteRenderer
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
        self._organism_renderer = SymbioteRenderer()
        self._labels: dict[tuple[str, tuple[int, int, int]], pygame.Surface] = {}

    def _label(self, text: str, color: tuple[int, int, int]) -> pygame.Surface:
        key = (text, color)
        if key not in self._labels:
            self._labels[key] = self._label_font.render(text, True, color)
        return self._labels[key]

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
        phase = self._label("PHASE 1A.2", self.ACCENT)
        surface.blit(phase, (width - phase.get_width() - 24, 29))

        chamber = pygame.Rect(24, 82, max(1, width - 48), max(1, height - 128))
        pygame.draw.rect(surface, self.CHAMBER, chamber, border_radius=10)
        pygame.draw.rect(surface, self.BORDER, chamber, width=1, border_radius=10)

        if simulation is not None:
            self._organism_renderer.draw(
                surface, chamber, simulation.organism, 1.0 if simulation.paused else alpha,
            )
            appearance = simulation.organism.appearance
            specimen = f"SEED {appearance.seed}   /   {len(appearance.tendrils)} TENDRILS"
            status = f"{'PAUSED' if simulation.paused else 'LIVE'}  {simulation.animation_speed:.2f}x"
            surface.blit(self._label(specimen, self.MUTED), (40, height - 68))
            status_label = self._label(status, self.ACCENT)
            surface.blit(status_label, (width - status_label.get_width() - 40, height - 68))
            controls = "SPACE Pause   R Reset   -/+ Speed"
        else:
            controls = "EMPTY CHAMBER"
        surface.blit(self._label(controls, self.MUTED), (24, height - 29))
        hint = self._label("ESC  Exit", self.MUTED)
        surface.blit(hint, (width - hint.get_width() - 24, height - 29))
