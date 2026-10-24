"""Pygame presentation for the empty simulation chamber."""

import pygame


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
        self._label_font = pygame.font.Font(None, 20)

    def draw(self, surface: pygame.Surface) -> None:
        """Recompute layout from the current window size on every frame."""
        width, height = surface.get_size()
        surface.fill(self.BACKGROUND)
        surface.blit(self._title_font.render("EIDOLON", True, self.TEXT), (24, 19))
        surface.blit(
            self._label_font.render("SIMULATION CHAMBER", True, self.MUTED), (24, 50)
        )
        phase = self._label_font.render("PHASE 0", True, self.ACCENT)
        surface.blit(phase, (width - phase.get_width() - 24, 29))

        chamber = pygame.Rect(24, 82, max(1, width - 48), max(1, height - 128))
        pygame.draw.rect(surface, self.CHAMBER, chamber, border_radius=10)
        pygame.draw.rect(surface, self.BORDER, chamber, width=1, border_radius=10)

        surface.blit(
            self._label_font.render("EMPTY CHAMBER", True, self.MUTED),
            (24, height - 29),
        )
        hint = self._label_font.render("ESC  Exit", True, self.MUTED)
        surface.blit(hint, (width - hint.get_width() - 24, height - 29))
