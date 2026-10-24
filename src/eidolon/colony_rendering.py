"""A flat membrane drawn from physical nodes and local mesh triangles."""

from math import ceil, sin

import numpy as np
import pygame

from eidolon.camera import Camera
from eidolon.colony import Colony, MEMBRANE_RADIUS
from eidolon.organism import Appearance


class ColonyRenderer:
    SUPERSAMPLE = 2

    def __init__(self) -> None:
        self._cache_key = None

    def _prepare(self, appearance: Appearance, pixel_radius: int) -> None:
        key = (appearance, pixel_radius)
        if self._cache_key == key:
            return
        self._cache_key = key
        self._edge = pixel_radius * 12 + 16
        size = self._edge * self.SUPERSAMPLE
        self._ink = pygame.Surface((size, size), pygame.SRCALPHA)
        self._sprite = pygame.Surface((self._edge, self._edge), pygame.SRCALPHA)
        radius = max(2, round(pixel_radius * self.SUPERSAMPLE * 0.65))
        axis = np.arange(-radius, radius + 1, dtype=np.float32)
        distance = (axis[:, None] ** 2 + axis[None, :] ** 2) / radius**2
        self._glow = pygame.Surface((len(axis), len(axis)), pygame.SRCALPHA)
        rgb = pygame.surfarray.pixels3d(self._glow)
        rgb[:] = appearance.core_color
        del rgb
        alpha = pygame.surfarray.pixels_alpha(self._glow)
        alpha[:] = (np.exp(-distance * 4) * np.clip(1 - distance, 0, 1) * 145).astype(np.uint8)
        del alpha

    def draw(
        self, destination: pygame.Surface, viewport: pygame.Rect, colony: Colony,
        camera: Camera, appearance: Appearance, alpha: float = 1.0,
        seconds: float = 0.0,
    ) -> None:
        self._prepare(appearance, max(1, ceil(colony.base_radius * colony.scale * camera.scale)))
        samples = colony.sample(alpha)
        mass = sum(node.mass for node in colony.nodes)
        center = tuple(sum(point[axis] * node.mass for point, node in zip(samples, colony.nodes)) / mass for axis in (0, 1))
        scale = camera.scale * self.SUPERSAMPLE
        origin = self._ink.get_width() / 2
        points = [(round(origin + (x - center[0]) * scale), round(origin + (y - center[1]) * scale))
                  for x, y, _ in samples]
        radii = [max(1, round(radius * MEMBRANE_RADIUS * scale)) for _, _, radius in samples]
        self._ink.fill((0, 0, 0, 0))
        color = (*appearance.tissue_color, 232)
        # These primitives are a union of physical neighborhoods, not a rigid
        # sprite, convex hull, or analytic squid/cell silhouette.
        for triangle in colony.triangles:
            pygame.draw.polygon(self._ink, color, [points[index] for index in triangle])
        for bond in colony.bonds:
            if bond.rest_length < colony.spacing * 1.5:
                width = max(1, min(radii[bond.first], radii[bond.second]) * 2)
                pygame.draw.line(self._ink, color, points[bond.first], points[bond.second], width)
        for index, (position, radius) in enumerate(zip(points, radii)):
            pygame.draw.circle(self._ink, color, position, radius)
        pigment = tuple(round(a * 0.85 + b * 0.15) for a, b in zip(appearance.tissue_color, appearance.membrane_color))
        for index in range(0, len(points), 4):
            pygame.draw.circle(self._ink, (*pigment, 232), points[index], max(1, radii[index] // 2))
        anchor = colony.anchor_index
        lights = (anchor, *colony.nodes[anchor].connections[:2])
        for index, node_index in enumerate(lights):
            self._glow.set_alpha(round(105 + 25 * sin(seconds * 1.8 + index)))
            self._ink.blit(self._glow, self._glow.get_rect(center=points[node_index]))
            pygame.draw.circle(self._ink, (*appearance.core_color, 175), points[node_index], max(1, radii[node_index] // 5))
        pygame.transform.smoothscale(self._ink, (self._edge, self._edge), self._sprite)
        screen_center = camera.world_to_screen(center)
        previous_clip = destination.get_clip()
        try:
            destination.set_clip(viewport.inflate(-2, -2).clip(previous_clip))
            destination.blit(self._sprite, (round(screen_center[0] - self._edge / 2), round(screen_center[1] - self._edge / 2)))
        finally:
            destination.set_clip(previous_clip)
