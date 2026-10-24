"""Flat translucent silhouettes, drifting membranes, and restrained internal glow."""

from math import cos, pi, sin, tau

import numpy as np
import pygame

from eidolon.organism import Appearance, Color, Symbiote
from eidolon.organism_geometry import (
    Point, appendage_mesh, body_anchors, contour_point, deform, inset_outline,
)


def mix(first: Color, second: Color, amount: float) -> Color:
    return tuple(round(a + (b - a) * amount) for a, b in zip(first, second))


class SymbioteRenderer:
    MEMBRANE_SAMPLES = 96
    TENDRIL_SEGMENTS = 32

    def __init__(self) -> None:
        self._cache_key: tuple[Appearance, int] | None = None
        self._samples = tuple(index / self.MEMBRANE_SAMPLES for index in range(self.MEMBRANE_SAMPLES))
        self._core_angles = tuple(tau * index / 32 for index in range(32))

    @staticmethod
    def _glow(radius: int, color: Color) -> pygame.Surface:
        """Cache light at the core only; there is no directional surface lighting."""
        axis = np.arange(-radius, radius + 1, dtype=np.float32)
        distance_squared = (axis[:, None] ** 2 + axis[None, :] ** 2) / radius**2
        alpha = np.exp(-distance_squared * 4.5) * np.clip(1 - distance_squared, 0, 1) * 150
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
        self._origin = radius * 6
        size = (radius * 12, radius * 12)
        self._canvas = pygame.Surface(size, pygame.SRCALPHA)
        self._layer = pygame.Surface(size, pygame.SRCALPHA)
        self._mask = pygame.Surface(size, pygame.SRCALPHA)
        self._core_glow = self._glow(max(1, round(radius * 0.95)), appearance.core_color)

    @staticmethod
    def _polygon(surface, points, color) -> None:
        pygame.draw.polygon(surface, color, points)
        pygame.draw.aalines(surface, color, True, points)

    def draw(
        self, destination: pygame.Surface, chamber: pygame.Rect,
        organism: Symbiote, alpha: float = 1.0,
        *, center: tuple[float, float] | None = None,
        body_radius: float | None = None, heading: float | None = None,
    ) -> None:
        appearance = organism.appearance
        radius = (max(1, round(body_radius)) if body_radius is not None
                  else max(1, min(95, int(min(chamber.size) / 9.8))))
        self._prepare(appearance, radius)
        pose = organism.pose(alpha)
        canvas, layer, mask = self._canvas, self._layer, self._mask
        canvas.fill((0, 0, 0, 0))
        rotation = pose.rotation if heading is None else heading - appearance.body.trail_angle + pi + pose.rotation
        cosine, sine = cos(rotation), sin(rotation)

        def project(point: Point) -> tuple[int, int]:
            x, y = point
            return (
                round(self._origin + radius * (x * cosine - y * sine)),
                round(self._origin + radius * (x * sine + y * cosine)),
            )

        def polygon(surface, points, color):
            self._polygon(surface, [project(point) for point in points], color)

        tissue, membrane = appearance.tissue_color, appearance.membrane_color
        anchors = body_anchors(appearance, pose)
        outline = tuple(contour_point(anchors, u) for u in self._samples)
        core_center = deform(appearance.core.offset, appearance, pose)
        meshes = tuple(
            appendage_mesh(anchors, tendril, appearance, pose, self.TENDRIL_SEGMENTS)
            for tendril in appearance.tendrils
        )

        # A few flat, translucent webs span neighboring trailing roots. They are
        # anatomical sheets, with no lit edges, bevels, rib bands, or tube shading.
        for index in range(0, min(len(meshes) - 1, appearance.body.fold_count * 2), 2):
            first, second = meshes[index], meshes[index + 1]
            flap_anchors = (
                first.centerline[0], first.left[4], first.left[8],
                second.right[7], second.right[3], second.centerline[0],
            )
            flap = tuple(contour_point(flap_anchors, u / 40) for u in range(40))
            polygon(canvas, flap, (*membrane, 48))

        # A single ink color across each appendage deliberately keeps it flat.
        for mesh in meshes:
            polygon(canvas, mesh.silhouette, (*tissue, 205))
            pygame.draw.circle(canvas, (*tissue, 205), project(mesh.centerline[-1]), 1)

        outer_pixels = [project(point) for point in outline]
        self._polygon(canvas, outer_pixels, (*tissue, 218))
        xs, ys = zip(*outer_pixels)
        body_rect = pygame.Rect(min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)
        body_rect = body_rect.inflate(4, 4).clip(canvas.get_rect())
        layer.set_clip(body_rect)
        mask.set_clip(body_rect)
        mask.fill((0, 0, 0, 0))
        self._polygon(mask, outer_pixels, (255, 255, 255, 255))

        # Broad irregular color washes act like overlapping 2D pigment/membrane.
        # No concentric shells, directional highlights, or simulated 3D volume.
        for index in range(appearance.body.fold_count):
            layer.fill((0, 0, 0, 0))
            start = 0.12 + appearance.body.phase / tau * 0.55 + index * 0.22
            positions = (start, start + 0.14, start + 0.29)
            edge = tuple(contour_point(anchors, u) for u in positions)
            inset = tuple(
                (core_center[0] + (x - core_center[0]) * 0.40
                 + 0.08 * sin(pose.seconds * 0.45 + index),
                 core_center[1] + (y - core_center[1]) * 0.48)
                for x, y in edge[::-1]
            )
            flap_anchors = edge + inset
            flap = tuple(contour_point(flap_anchors, u / 48) for u in range(48))
            color = mix(tissue, appearance.accent_color, 0.55)
            polygon(layer, flap, (*color, appearance.body.membrane_alpha))
            layer.blit(mask, body_rect, body_rect, special_flags=pygame.BLEND_RGBA_MULT)
            canvas.blit(layer, body_rect, body_rect)

        layer.fill((0, 0, 0, 0))
        core_position = project(core_center)
        self._core_glow.set_alpha(round(165 + (pose.core_pulse - 0.9) * 200))
        layer.blit(self._core_glow, self._core_glow.get_rect(center=core_position))
        core = appearance.core
        # An irregular luminous organ suspended in ink; no spherical highlight
        # or hard ring. Its soft emission is the only radial gradient in the art.
        core_outline = tuple(
            (core_center[0] + cos(angle) * core.radius * pose.core_pulse
             * (1.0 + 0.17 * sin(3 * angle + core.phase)),
             core_center[1] + sin(angle) * core.radius * core.elongation * pose.core_pulse)
            for angle in self._core_angles
        )
        # Two faint flowing filaments bind the light to the membrane, without
        # mechanical struts or radial spokes.
        for index in (0, 1):
            end = contour_point(anchors, 0.64 + index * 0.17)
            points = tuple(project((
                core_center[0] * (1 - u) + end[0] * u * 0.65
                + 0.06 * sin(u * 5 + pose.seconds * 0.35 + index),
                core_center[1] * (1 - u) + end[1] * u * 0.65,
            )) for u in (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65))
            pygame.draw.aalines(layer, (*mix(appearance.core_color, tissue, 0.45), 65), False, points)
        polygon(layer, core_outline, (*appearance.core_color, 168))
        polygon(layer, inset_outline(core_outline, core_center, 0.65), (*appearance.core_color, 210))
        layer.blit(mask, body_rect, body_rect, special_flags=pygame.BLEND_RGBA_MULT)
        canvas.blit(layer, body_rect, body_rect)

        # Balance the trailing tissue in the viewport rather than centering only
        # the body and letting its appendages disappear below the chamber.
        layout = (0.65 * cos(appearance.body.trail_angle), 0.65 * sin(appearance.body.trail_angle))
        if center is None:
            center = (
                round(chamber.centerx + pose.offset[0] * min(chamber.size)
                      - radius * (layout[0] * cosine - layout[1] * sine)),
                round(chamber.centery + pose.offset[1] * min(chamber.size)
                      - radius * (layout[0] * sine + layout[1] * cosine)),
            )
        old_clip = destination.get_clip()
        try:
            destination.set_clip(chamber.inflate(-4, -4).clip(old_clip))
            destination.blit(canvas, (center[0] - self._origin, center[1] - self._origin))
        finally:
            destination.set_clip(old_clip)
