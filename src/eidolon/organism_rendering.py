"""Layered living tissue, broad appendages, and embedded light on CPU surfaces."""

from math import cos, sin, tau

import numpy as np
import pygame

from eidolon.organism import Appearance, Color, Symbiote
from eidolon.organism_geometry import (
    AppendageMesh, Point, appendage_mesh, body_anchors, contour_point,
    deform, inset_outline,
)


def mix(first: Color, second: Color, amount: float) -> Color:
    return tuple(round(a + (b - a) * amount) for a, b in zip(first, second))


class SymbioteRenderer:
    MEMBRANE_SAMPLES = 80
    TENDRIL_SEGMENTS = 24

    def __init__(self) -> None:
        self._cache_key: tuple[Appearance, int] | None = None
        self._samples = tuple(index / self.MEMBRANE_SAMPLES for index in range(self.MEMBRANE_SAMPLES))
        self._core_angles = tuple(tau * index / 32 for index in range(32))

    @staticmethod
    def _glow(radius: int, color: Color) -> pygame.Surface:
        """Build a small radial stamp only when appearance or scale changes."""
        axis = np.arange(-radius, radius + 1, dtype=np.float32)
        distance_squared = (axis[:, None] ** 2 + axis[None, :] ** 2) / radius**2
        alpha = np.exp(-distance_squared * 4.5) * np.clip(1 - distance_squared, 0, 1) * 190
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
        self._origin = radius * 5
        size = (radius * 10, radius * 10)
        self._canvas = pygame.Surface(size, pygame.SRCALPHA)
        self._layer = pygame.Surface(size, pygame.SRCALPHA)
        self._mask = pygame.Surface(size, pygame.SRCALPHA)
        self._core_glow = self._glow(max(1, round(radius * 0.95)), appearance.core_color)
        self._tissue_light = self._glow(max(1, round(radius * 1.65)), appearance.membrane_color)
        self._tissue_light.set_alpha(85)

    @staticmethod
    def _polygon(surface, points, color) -> None:
        pygame.draw.polygon(surface, color, points)
        pygame.draw.aalines(surface, color, True, points)

    @staticmethod
    def _ribbon(mesh: AppendageMesh, scale: float, bias: float = 0.0) -> tuple[Point, ...]:
        left, right = [], []
        for center, edge in zip(mesh.centerline, mesh.left):
            dx, dy = edge[0] - center[0], edge[1] - center[1]
            left.append((center[0] + dx * (scale + bias), center[1] + dy * (scale + bias)))
            right.append((center[0] + dx * (-scale + bias), center[1] + dy * (-scale + bias)))
        return tuple(left + right[::-1])

    def draw(
        self, destination: pygame.Surface, chamber: pygame.Rect,
        organism: Symbiote, alpha: float = 1.0,
    ) -> None:
        appearance = organism.appearance
        radius = max(1, min(105, int(min(chamber.size) / 8.8)))
        self._prepare(appearance, radius)
        pose = organism.pose(alpha)
        canvas, layer, mask = self._canvas, self._layer, self._mask
        canvas.fill((0, 0, 0, 0))
        cosine, sine = cos(pose.rotation), sin(pose.rotation)

        def project(point: Point) -> tuple[int, int]:
            x, y = point
            return (
                round(self._origin + radius * (x * cosine - y * sine)),
                round(self._origin + radius * (x * sine + y * cosine)),
            )

        def polygon(surface, points, color):
            self._polygon(surface, [project(point) for point in points], color)

        tissue = appearance.tissue_color
        membrane = appearance.membrane_color
        dark = mix(tissue, (6, 10, 16), 0.48)
        anchors = body_anchors(appearance, pose)
        outline = tuple(contour_point(anchors, u) for u in self._samples)
        core_center = deform(appearance.core.offset, appearance, pose)
        meshes = tuple(
            appendage_mesh(anchors, tendril, appearance, pose, self.TENDRIL_SEGMENTS)
            for tendril in appearance.tendrils
        )

        # Filled ribbon meshes replace stroked wires. Broad roots underlap the
        # body, and off-center highlight strips give each muscle a rounded form.
        for mesh in meshes:
            polygon(canvas, mesh.silhouette, dark)
            for scale, bias, light in (
                (0.94, 0.01, 0.0), (0.80, 0.05, 0.05), (0.65, 0.10, 0.10),
                (0.49, 0.15, 0.14), (0.32, 0.20, 0.17), (0.16, 0.24, 0.20),
            ):
                polygon(canvas, self._ribbon(mesh, scale, bias), mix(tissue, membrane, light))
            for index in (4, 8, 12):
                center, edge = mesh.centerline[index], mesh.left[index]
                dx, dy = edge[0] - center[0], edge[1] - center[1]
                band = (
                    project((center[0] - dx * 0.70, center[1] - dy * 0.70)),
                    project((center[0] + dx * 0.40, center[1] + dy * 0.40)),
                    project((center[0] + dx * 0.75 - 0.025, center[1] + dy * 0.75 - 0.04)),
                )
                pygame.draw.aalines(canvas, mix(tissue, appearance.accent_color, 0.16), False, band)

        outer_pixels = [project(point) for point in outline]
        self._polygon(canvas, outer_pixels, dark)
        xs, ys = zip(*outer_pixels)
        body_rect = pygame.Rect(min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)
        body_rect = body_rect.inflate(4, 4).clip(canvas.get_rect())
        # Composite only the tissue's occupied region, not the large transparent
        # margins retained for appendages. Clip changes do not allocate surfaces.
        layer.set_clip(body_rect)
        mask.set_clip(body_rect)
        mask.fill((0, 0, 0, 0))
        self._polygon(mask, outer_pixels, (255, 255, 255, 255))
        layer.fill((0, 0, 0, 0))
        # Offset, unequally contracted folds replace concentric cell walls.
        for index in range(7):
            scale = 0.99 - index * 0.035
            shift = (-index * 0.012, -index * 0.008)
            shade = mix(tissue, membrane, index * 0.014)
            folded = tuple(
                (x + shift[0] + 0.05 * sin(index * 0.25 + pose.seconds * 0.7), y + shift[1])
                for index, (x, y) in enumerate(inset_outline(outline, core_center, scale))
            )
            polygon(layer, folded, shade)
        layer.blit(mask, body_rect, body_rect, special_flags=pygame.BLEND_RGBA_MULT)
        canvas.blit(layer, body_rect, body_rect)

        # Details and translucent folds are masked by the actual body silhouette,
        # including its clefts. Local surfaces are reused until view scale changes.
        layer.fill((0, 0, 0, 0))
        light_position = project((core_center[0] - 0.28, core_center[1] - 0.35))
        layer.blit(self._tissue_light, self._tissue_light.get_rect(center=light_position))
        for positions, offset in (((0.06, 0.22, 0.40), (-0.12, 0.12)), ((0.48, 0.66, 0.92), (0.14, -0.10))):
            edge = tuple(contour_point(anchors, u) for u in positions)
            underside = tuple(
                (core_center[0] + (x - core_center[0]) * 0.56 + offset[0],
                 core_center[1] + (y - core_center[1]) * 0.56 + offset[1])
                for x, y in edge[::-1]
            )
            flap_anchors = edge + underside
            flap = tuple(contour_point(flap_anchors, index / 40) for index in range(40))
            polygon(layer, flap, (*membrane, appearance.body.membrane_alpha))
        layer.blit(mask, body_rect, body_rect, special_flags=pygame.BLEND_RGBA_MULT)
        canvas.blit(layer, body_rect, body_rect)

        layer.fill((0, 0, 0, 0))
        # Curved embedded struts follow the tissue around the core, rather than
        # radiating from it as spokes or encircling it with mechanical rings.
        for index in range(appearance.body.ridge_count):
            first = outline[(8 + index * 4) % self.MEMBRANE_SAMPLES]
            last = outline[(48 + index * 3) % self.MEMBRANE_SAMPLES]
            control = (core_center[0] - 0.42 + index * 0.09, core_center[1] - 0.70 + index * 0.19)
            points = []
            for step in range(15):
                u = step / 14
                points.append(project((
                    (1 - u) ** 2 * first[0] * 0.82 + 2 * (1 - u) * u * control[0] + u * u * last[0] * 0.82,
                    (1 - u) ** 2 * first[1] * 0.82 + 2 * (1 - u) * u * control[1] + u * u * last[1] * 0.82,
                )))
            pygame.draw.lines(layer, (*dark, 105), False, points, max(1, radius // 27))
            pygame.draw.aalines(layer, (*appearance.accent_color, 65), False, points)

        core_position = project(core_center)
        self._core_glow.set_alpha(round(155 + (pose.core_pulse - 0.9) * 250))
        layer.blit(self._core_glow, self._core_glow.get_rect(center=core_position))
        core = appearance.core
        # The light sits inside a pulsing, irregular sheath. Its ends merge into
        # the surrounding tissue instead of looking like an isolated round lamp.
        core_outline = tuple(
            (core_center[0] + cos(angle) * core.radius * pose.core_pulse
             * (0.80 + 0.15 * sin(3 * angle + core.phase)),
             core_center[1] + sin(angle) * core.radius * core.elongation * pose.core_pulse * 1.5)
            for angle in self._core_angles
        )
        polygon(layer, inset_outline(core_outline, core_center, 1.65), (*mix(tissue, membrane, 0.28), 240))
        core_color = appearance.core_color
        polygon(layer, inset_outline(core_outline, core_center, 1.20), (*mix(tissue, core_color, 0.52), 255))
        polygon(layer, core_outline, (*core_color, 255))
        polygon(layer, inset_outline(core_outline, core_center, 0.52), (*mix(core_color, (244, 245, 226), 0.44), 255))
        # A few illuminated conduits visually connect the core to its sheath.
        for index in (12, 38, 62):
            end = outline[index]
            points = tuple(project((
                core_center[0] * (1 - u) + end[0] * u * 0.62 + 0.08 * sin(u * 5 + pose.seconds * 0.4),
                core_center[1] * (1 - u) + end[1] * u * 0.62,
            )) for u in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8))
            pygame.draw.aalines(layer, (*mix(core_color, tissue, 0.55), 135), False, points)
        layer.blit(mask, body_rect, body_rect, special_flags=pygame.BLEND_RGBA_MULT)
        canvas.blit(layer, body_rect, body_rect)

        # Restrained edge sheen on a fold, not a bright continuous circular rim.
        rim = [project(point) for point in outline[8:35]]
        pygame.draw.aalines(canvas, (*mix(tissue, membrane, 0.45), 165), False, rim)
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
