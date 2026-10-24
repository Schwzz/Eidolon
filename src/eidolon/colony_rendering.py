"""Continuous flat membrane from a small, local density field per component."""

from math import ceil, sin
from random import Random

import numpy as np
import pygame

from eidolon.camera import Camera
from eidolon.colony import Colony, MEMBRANE_RADIUS
from eidolon.organism import Appearance


class ColonyRenderer:
    SUPERSAMPLE = 2

    def __init__(self) -> None:
        self._cache_key = None

    def _prepare(self, appearance: Appearance, pixel_spacing: int, node_count: int) -> None:
        key = (appearance, pixel_spacing, node_count)
        if self._cache_key == key:
            return
        self._cache_key = key
        self._buffers = {}
        radius = max(2, ceil(pixel_spacing * 1.35))
        axis = np.arange(-radius, radius + 1, dtype=np.float32)
        distance = (axis[:, None] ** 2 + axis[None, :] ** 2) / radius**2
        self._stamp = pygame.Surface((len(axis), len(axis)))
        rgb = pygame.surfarray.pixels3d(self._stamp)
        density = (110 * np.exp(-3 * distance) * np.clip(1 - distance, 0, 1)).astype(np.uint8)
        rgb[:] = density[:, :, None]
        del rgb
        coverage = np.clip((np.arange(256, dtype=np.float32) - 50) / 40, 0, 1)
        self._coverage = (coverage * coverage * (3 - 2 * coverage) * 232).astype(np.uint8)
        self._glow = pygame.Surface((len(axis), len(axis)), pygame.SRCALPHA)
        rgb = pygame.surfarray.pixels3d(self._glow)
        rgb[:] = appearance.core_color
        del rgb
        alpha = pygame.surfarray.pixels_alpha(self._glow)
        alpha[:] = (np.exp(-distance * 4) * np.clip(1 - distance, 0, 1) * 55).astype(np.uint8)
        del alpha
        rng = Random(appearance.seed ^ 0xB10A55)
        # Four material grains per node; these are drawn, never simulated.
        self._grains = tuple((index, rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(0.3, 0.6))
                             for index in range(node_count) for _ in range(4))

    def _select_buffers(self, required_edge: int) -> None:
        required_edge = ceil(required_edge * 1.25)  # Headroom avoids reallocating on small elastic changes.
        edge = max(32, 1 << (required_edge - 1).bit_length())
        if edge not in self._buffers:
            if len(self._buffers) >= 8:
                self._buffers.pop(next(iter(self._buffers)))
            self._buffers[edge] = (
                pygame.Surface((edge * self.SUPERSAMPLE, edge * self.SUPERSAMPLE), pygame.SRCALPHA),
                pygame.Surface((edge, edge), pygame.SRCALPHA),
                pygame.Surface((edge * self.SUPERSAMPLE, edge * self.SUPERSAMPLE)),
            )
        self._edge = edge
        self._ink, self._sprite, self._field = self._buffers[edge]

    def draw(
        self, destination: pygame.Surface, viewport: pygame.Rect, colony: Colony,
        camera: Camera, appearance: Appearance, alpha: float = 1.0,
        seconds: float = 0.0,
    ) -> None:
        self._prepare(appearance, max(1, ceil(colony.spacing * colony.scale * camera.scale * self.SUPERSAMPLE)),
                      len(colony.nodes))
        samples = colony.sample(alpha)
        previous_clip = destination.get_clip()
        try:
            destination.set_clip(viewport.inflate(-2, -2).clip(previous_clip))
            for component in colony.components:
                self._draw_component(destination, colony, camera, appearance, samples, component, seconds)
        finally:
            destination.set_clip(previous_clip)

    def _draw_component(self, destination, colony, camera, appearance, samples, component, seconds):
        members = set(component)
        mass = sum(colony.nodes[index].mass for index in component)
        center = tuple(sum(samples[index][axis] * colony.nodes[index].mass for index in component) / mass for axis in (0, 1))
        extent = max(max(abs(samples[index][0] - center[0]), abs(samples[index][1] - center[1]))
                     + colony.spacing * colony.scale * 1.5 for index in component)
        self._select_buffers(ceil(extent * camera.scale * 2 + 12))
        scale = camera.scale * self.SUPERSAMPLE
        origin = self._ink.get_width() / 2
        points = [(round(origin + (x - center[0]) * scale), round(origin + (y - center[1]) * scale))
                  for x, y, _ in samples]
        radii = [max(1, round(radius * MEMBRANE_RADIUS * scale)) for _, _, radius in samples]
        self._field.fill((0, 0, 0))
        # Composite optical accents onto opaque pigment before applying the
        # coverage mask; replacing alpha on a transparent RGB glow makes a box.
        self._ink.fill((*appearance.tissue_color, 255))
        # The field joins intact local tissue. Separate components never share
        # density, so rendering cannot invent a bridge before physical capture.
        for triangle in colony.triangles:
            if triangle[0] in members:
                pygame.draw.polygon(self._field, (70, 70, 70), [points[index] for index in triangle])
        for bond in colony.bonds:
            if bond.first in members and bond.rest_length < colony.spacing * 1.5:
                width = max(1, min(radii[bond.first], radii[bond.second]))
                pygame.draw.line(self._field, (70, 70, 70), points[bond.first], points[bond.second], width)
        for index, position in enumerate(points):
            if index in members:
                self._field.blit(self._stamp, self._stamp.get_rect(center=position), special_flags=pygame.BLEND_RGB_ADD)
        pigment = tuple(round(a * 0.82 + b * 0.18) for a, b in zip(appearance.tissue_color, appearance.membrane_color))
        for index, dx, dy, radius in self._grains:
            if index in members:
                point = (round(points[index][0] + dx * radii[index]), round(points[index][1] + dy * radii[index]))
                pygame.draw.circle(self._ink, (*pigment, 232), point, max(1, round(radius * colony.scale * scale)))
        if component == colony.components[0]:
            anchor = points[colony.anchor_index] if colony.anchor_index in members else (round(origin), round(origin))
            self._glow.set_alpha(round(150 + 20 * sin(seconds * 1.8)))
            self._ink.blit(self._glow, self._glow.get_rect(center=anchor))
            core = pygame.Rect(0, 0, max(3, round(colony.base_radius * colony.scale * scale * 0.48)),
                               max(2, round(colony.base_radius * colony.scale * scale * 0.3)))
            core.center = anchor
            pygame.draw.ellipse(self._ink, (*appearance.core_color, 232), core)
        density = pygame.surfarray.pixels3d(self._field)
        alpha = pygame.surfarray.pixels_alpha(self._ink)
        alpha[:] = self._coverage[density[:, :, 0]]
        del alpha, density
        pygame.transform.smoothscale(self._ink, (self._edge, self._edge), self._sprite)
        screen_center = camera.world_to_screen(center)
        destination.blit(self._sprite, (round(screen_center[0] - self._edge / 2), round(screen_center[1] - self._edge / 2)))
