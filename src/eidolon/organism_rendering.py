"""Small, reusable CPU surfaces and procedural curves for one symbiote."""

from math import cos, sin, tau

import numpy as np
import pygame
import pygame.gfxdraw

from eidolon.organism import Appearance, OrganismPose, Symbiote


class SymbioteRenderer:
    MEMBRANE_SAMPLES = 80
    TENDRIL_SEGMENTS = 24

    def __init__(self) -> None:
        self._cache_key: tuple[Appearance, int] | None = None
        self._angles = tuple(tau * index / self.MEMBRANE_SAMPLES for index in range(self.MEMBRANE_SAMPLES))
        self._directions = tuple((cos(angle), sin(angle)) for angle in self._angles)
        self._samples = tuple(index / self.TENDRIL_SEGMENTS for index in range(self.TENDRIL_SEGMENTS + 1))
        self._tapers = tuple((1 - u) ** 1.2 for u in self._samples[:-1])

    @staticmethod
    def _glow(radius: int, color: tuple[int, int, int]) -> pygame.Surface:
        """Build a radial alpha stamp only when appearance or view scale changes."""
        axis = np.arange(-radius, radius + 1, dtype=np.float32)
        distance_squared = (axis[:, None] ** 2 + axis[None, :] ** 2) / radius**2
        alpha = np.exp(-distance_squared * 4.5) * np.clip(1 - distance_squared, 0, 1) * 210
        stamp = pygame.Surface((len(axis), len(axis)), pygame.SRCALPHA)
        pixels = pygame.surfarray.pixels3d(stamp)
        pixels[:] = color
        del pixels
        pixels_alpha = pygame.surfarray.pixels_alpha(stamp)
        pixels_alpha[:] = alpha.astype(np.uint8)
        del pixels_alpha
        return stamp

    def _prepare(self, appearance: Appearance, radius: int) -> None:
        key = (appearance, radius)
        if key == self._cache_key:
            return
        self._cache_key = key
        self._radius = radius
        self._canvas = pygame.Surface((radius * 8, radius * 8), pygame.SRCALPHA)
        self._origin = radius * 4
        self._core_glow = self._glow(max(1, round(radius * 0.85)), appearance.core_color)
        self._tip_glow = self._glow(max(2, round(radius * 0.12)), appearance.membrane_color)
        self._profile = tuple(self._body_profile(angle, appearance) for angle in self._angles)

    @staticmethod
    def _body_profile(angle: float, appearance: Appearance) -> float:
        body = appearance.body
        return (
            1.0 + body.asymmetry * cos(angle - body.phase)
            + body.lobes * sin(3 * angle + body.phase)
            + 0.045 * cos(2 * angle - body.phase * 0.7)
        )

    def _body_point(
        self, angle: float, appearance: Appearance, pose: OrganismPose,
        profile: float | None = None, direction: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        if profile is None:
            profile = self._body_profile(angle, appearance)
        wave = 0.025 * sin(3 * angle + pose.seconds * 1.2)
        wave += 0.016 * sin(7 * angle - pose.seconds * 0.8 + appearance.body.phase)
        radial = (profile + wave) * pose.breath
        if direction is None:
            direction = (cos(angle), sin(angle))
        return (
            direction[0] * appearance.body.width * radial,
            direction[1] * appearance.body.height * radial,
        )

    def draw(
        self, destination: pygame.Surface, chamber: pygame.Rect,
        organism: Symbiote, alpha: float = 1.0,
    ) -> None:
        appearance = organism.appearance
        # Leave room for long tendrils and drift even at the minimum window size.
        radius = max(1, min(105, int(min(chamber.size) / 8.5)))
        self._prepare(appearance, radius)
        pose = organism.pose(alpha)
        canvas = self._canvas
        canvas.fill((0, 0, 0, 0))
        cosine, sine = cos(pose.rotation), sin(pose.rotation)

        def project(point: tuple[float, float]) -> tuple[int, int]:
            x, y = point
            return (
                round(self._origin + radius * (x * cosine - y * sine)),
                round(self._origin + radius * (x * sine + y * cosine)),
            )

        membrane = appearance.membrane_color
        shadow = tuple(round(channel * 0.32) for channel in membrane)
        highlight = tuple(round(channel * 0.65 + 255 * 0.35) for channel in membrane)

        for tendril in appearance.tendrils:
            root = self._body_point(tendril.angle, appearance, pose)
            outward = (cos(tendril.angle), sin(tendril.angle))
            tangent = (-outward[1], outward[0])
            phase = pose.seconds * tendril.frequency + tendril.phase
            base_wave = sin(-phase)
            points = []
            for u in self._samples:
                along = tendril.length * u
                sway = tendril.sway * u * (sin(5.0 * u - phase) - base_wave)
                across = tendril.curl * u * u + sway
                points.append(project((
                    root[0] + outward[0] * along + tangent[0] * across,
                    root[1] + outward[1] * along + tangent[1] * across,
                )))
            for index in range(self.TENDRIL_SEGMENTS):
                taper = self._tapers[index]
                width = max(1, round(radius * 0.12 * tendril.thickness * taper))
                pygame.draw.line(canvas, shadow, points[index], points[index + 1], width + 2)
                pygame.draw.line(canvas, membrane, points[index], points[index + 1], width)
            pygame.draw.aalines(canvas, highlight, False, points)
            # Short proximal bands suggest a subtle biomechanical sheath.
            for index in (3, 6):
                x, y = points[index]
                dx, dy = points[index + 1][0] - x, points[index + 1][1] - y
                length = max(1.0, (dx * dx + dy * dy) ** 0.5)
                half_width = radius * 0.035 * tendril.thickness
                band = (-dy / length * half_width, dx / length * half_width)
                pygame.draw.aaline(canvas, highlight, (x - band[0], y - band[1]), (x + band[0], y + band[1]))
            tip = points[-1]
            canvas.blit(self._tip_glow, self._tip_glow.get_rect(center=tip))

        body = [
            self._body_point(angle, appearance, pose, profile, direction)
            for angle, profile, direction in zip(self._angles, self._profile, self._directions)
        ]
        outer = [project(point) for point in body]
        pygame.gfxdraw.filled_polygon(canvas, outer, appearance.tissue_color)
        # A few nested silhouettes give the tissue depth without a per-frame
        # blur, texture synthesis, or full-window transparency buffers.
        for scale, brightness in ((0.94, 0.86), (0.87, 0.74), (0.78, 0.65)):
            inner = [project((x * scale, y * scale)) for x, y in body]
            inner_color = tuple(round(channel * brightness) for channel in appearance.tissue_color)
            pygame.gfxdraw.filled_polygon(canvas, inner, inner_color)

        core = appearance.core
        core_position = project(core.offset)
        # Structural filaments are visual details, not neural models.
        for index in range(0, self.MEMBRANE_SAMPLES, 10):
            end = body[index]
            points = []
            for step in range(9):
                u = step / 8
                bend = sin(u * 3.14159) * 0.10 * sin(pose.seconds * 0.5 + index)
                points.append(project((
                    core.offset[0] * (1 - u) + end[0] * 0.84 * u + bend,
                    core.offset[1] * (1 - u) + end[1] * 0.84 * u,
                )))
            pygame.draw.aalines(canvas, shadow, False, points)

        self._core_glow.set_alpha(round(200 + 45 * (pose.core_pulse - 0.9) / 0.2))
        canvas.blit(self._core_glow, self._core_glow.get_rect(center=core_position))
        core_radius = max(2, round(radius * core.radius * pose.core_pulse))
        ring = pygame.Rect(0, 0, core_radius * 3.5, core_radius * 3.5)
        ring.center = core_position
        for index in range(3):
            angle = index * tau / 3 + pose.rotation
            pygame.draw.arc(canvas, membrane, ring, angle, angle + 1.4, max(1, radius // 50))
        nucleus = pygame.Rect(0, 0, core_radius * 2, round(core_radius * 2 * core.elongation))
        nucleus.center = core_position
        pygame.draw.ellipse(canvas, appearance.core_color, nucleus)
        nucleus.inflate_ip(-core_radius * 0.75, -core_radius * 0.75)
        bright_core = tuple(round(channel * 0.30 + 255 * 0.70) for channel in appearance.core_color)
        pygame.draw.ellipse(canvas, bright_core, nucleus)

        pygame.draw.lines(canvas, shadow, True, outer, max(2, radius // 24))
        pygame.draw.aalines(canvas, membrane, True, outer)
        pygame.draw.aalines(canvas, (*highlight, 130), True, [project((x * 0.96, y * 0.96)) for x, y in body])

        center = (
            round(chamber.centerx + pose.offset[0] * min(chamber.size)),
            round(chamber.centery + pose.offset[1] * min(chamber.size)),
        )
        old_clip = destination.get_clip()
        try:
            destination.set_clip(chamber.inflate(-4, -4).clip(old_clip))
            destination.blit(canvas, (center[0] - self._origin, center[1] - self._origin))
        finally:
            destination.set_clip(old_clip)
