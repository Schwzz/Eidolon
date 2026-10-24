"""Deterministic position-based biomass physics, independent of rendering.

Nodes and fixed elastic bonds are separate data so later interaction work can
address individual particles/edges. This milestone never breaks or adds bonds.
"""

from dataclasses import dataclass
from math import ceil, cos, exp, hypot, isfinite, sin, sqrt
from random import Random


Point = tuple[float, float]
MEMBRANE_RADIUS = 1.5


@dataclass(slots=True)
class Node:
    identifier: int
    position: Point
    velocity: Point
    mass: float
    rest_radius: float
    radius: float
    connections: tuple[int, ...] = ()
    previous_position: Point = (0.0, 0.0)
    previous_radius: float = 0.0


@dataclass(frozen=True, slots=True)
class Bond:
    first: int
    second: int
    rest_length: float
    compliance: float
    rest_direction: Point
    phase: float


class Colony:
    MAX_STEP = 1 / 120
    SOLVER_ITERATIONS = 5
    ROW_PLANS = ((3, 5, 6, 5, 4), (3, 4, 5, 6, 5, 3), (2, 4, 5, 6, 5, 3))

    def __init__(self, seed: int, center: Point, base_radius: float) -> None:
        if not isfinite(base_radius) or base_radius <= 0:
            raise ValueError("base_radius must be finite and positive")
        rng = Random(seed ^ 0xC010F1)
        rows = rng.choice(self.ROW_PLANS)
        self.base_radius = base_radius
        self.spacing = base_radius * 0.85
        self.scale = 1.0
        self.elapsed_seconds = 0.0
        self.wall_contacts = 0
        self.phase = rng.uniform(0, 6.283185307)
        self.nodes: list[Node] = []
        for row, count in enumerate(rows):
            row_shift = 0.42 * sin(row * 1.2 + self.phase) + rng.uniform(-0.12, 0.12)
            for column in range(count):
                x = (column - (count - 1) / 2 + row_shift + rng.uniform(-0.06, 0.06)) * self.spacing
                y = (row - (len(rows) - 1) / 2 + rng.uniform(-0.06, 0.06)) * self.spacing * 0.866
                radius = self.spacing * rng.uniform(0.37, 0.41)
                self.nodes.append(Node(len(self.nodes), (x, y), (0, 0), rng.uniform(0.85, 1.15), radius, radius))
        offset = self.center
        for node in self.nodes:
            node.position = (node.position[0] - offset[0] + center[0], node.position[1] - offset[1] + center[1])
        self.bonds: list[Bond] = []
        neighbors: list[list[int]] = [[] for _ in self.nodes]
        for first, a in enumerate(self.nodes):
            for second in range(first + 1, len(self.nodes)):
                b = self.nodes[second]
                dx, dy = b.position[0] - a.position[0], b.position[1] - a.position[1]
                distance = hypot(dx, dy)
                if distance <= self.spacing * 2.05:
                    compliance = 0.0005 if distance < self.spacing * 1.5 else 0.004
                    self.bonds.append(Bond(first, second, distance, compliance,
                                           (dx / distance, dy / distance), (a.position[0] - center[0]) / self.spacing))
                    neighbors[first].append(second)
                    neighbors[second].append(first)
        for node, connections in zip(self.nodes, neighbors):
            node.connections = tuple(connections)
        # Local triangles fill the interstitial membrane, not a convex hull.
        self.triangles = tuple(
            (a, b, c)
            for a in range(len(self.nodes)) for b in neighbors[a] if b > a
            for c in neighbors[b] if c > b and c in neighbors[a]
            if max(hypot(self.nodes[i].position[0] - self.nodes[j].position[0],
                         self.nodes[i].position[1] - self.nodes[j].position[1])
                   for i, j in ((a, b), (b, c), (a, c))) < self.spacing * 1.5
        )
        self.anchor_index = min(range(len(self.nodes)), key=lambda i: hypot(
            self.nodes[i].position[0] - center[0], self.nodes[i].position[1] - center[1]))
        self.begin_step()

    @property
    def center(self) -> Point:
        mass = sum(node.mass for node in self.nodes)
        return tuple(sum(node.position[axis] * node.mass for node in self.nodes) / mass for axis in (0, 1))

    @property
    def velocity(self) -> Point:
        mass = sum(node.mass for node in self.nodes)
        return tuple(sum(node.velocity[axis] * node.mass for node in self.nodes) / mass for axis in (0, 1))

    def begin_step(self) -> None:
        for node in self.nodes:
            node.previous_position = node.position
            node.previous_radius = node.radius

    def sample(self, alpha: float = 1.0) -> tuple[tuple[float, float, float], ...]:
        alpha = min(1.0, max(0.0, alpha))
        return tuple((
            node.previous_position[0] + (node.position[0] - node.previous_position[0]) * alpha,
            node.previous_position[1] + (node.position[1] - node.previous_position[1]) * alpha,
            node.previous_radius + (node.radius - node.previous_radius) * alpha,
        ) for node in self.nodes)

    def set_growth(self, scale: float) -> None:
        if not isfinite(scale) or scale <= 0:
            raise ValueError("colony scale must be finite and positive")
        self.scale = scale
        for node in self.nodes:
            node.radius = node.rest_radius * scale

    def translate_to(self, center: Point) -> None:
        """Setup/reposition utility; locomotion never calls this method."""
        old = self.center
        for node in self.nodes:
            node.position = (node.position[0] + center[0] - old[0], node.position[1] + center[1] - old[1])
        self.begin_step()

    def freeze(self) -> None:
        for node in self.nodes:
            node.velocity = (0.0, 0.0)
        self.begin_step()

    def advance(self, dt: float, desired_velocity: Point, world_size: Point) -> None:
        if not isfinite(dt) or dt <= 0 or any(not isfinite(v) for v in desired_velocity):
            raise ValueError("physics timestep must be positive and inputs finite")
        if any(not isfinite(value) or value <= 0 for value in world_size):
            raise ValueError("physics world dimensions must be finite and positive")
        if min(world_size) <= max(node.radius for node in self.nodes) * MEMBRANE_RADIUS * 2:
            raise ValueError("physics world must fit the biomass nodes")
        count = max(1, ceil(dt / self.MAX_STEP - 1e-9))
        self.wall_contacts = 0
        for _ in range(count):
            self._step(dt / count, desired_velocity, world_size)

    def _step(self, dt: float, desired: Point, world_size: Point) -> None:
        nodes = self.nodes
        center, velocity = self.center, self.velocity
        desired_speed = hypot(*desired)
        drive = (desired[0] - velocity[0], desired[1] - velocity[1])
        direction = (desired[0] / desired_speed, desired[1] / desired_speed) if desired_speed else (0, 0)
        activity = min(1.0, desired_speed / 55)
        maximum_speed = 220 * sqrt(self.scale)
        old_positions = [node.position for node in nodes]
        for node in nodes:
            if not all(isfinite(v) for v in (*node.position, *node.velocity)):
                raise FloatingPointError("nonfinite colony state")
            if not isfinite(node.mass) or node.mass <= 0:
                raise ValueError("node mass must be finite and positive")
            projection = ((node.position[0] - center[0]) * direction[0]
                          + (node.position[1] - center[1]) * direction[1]) / (self.spacing * self.scale * 2.5)
            traction = min(1.8, max(0.25, 1 + projection * 0.9))
            vx = (node.velocity[0] + drive[0] * 12 * traction / node.mass * dt) * exp(-0.8 * dt)
            vy = (node.velocity[1] + drive[1] * 12 * traction / node.mass * dt) * exp(-0.8 * dt)
            # Damp relative motion and angular drift without forcing a pose or
            # rotating the colony to match its steering direction.
            relative_damping = exp(-2.0 * dt)
            vx = velocity[0] + (vx - velocity[0]) * relative_damping
            vy = velocity[1] + (vy - velocity[1]) * relative_damping
            magnitude = hypot(vx, vy)
            if magnitude > maximum_speed:
                vx, vy = vx * maximum_speed / magnitude, vy * maximum_speed / magnitude
            node.position = (node.position[0] + vx * dt, node.position[1] + vy * dt)
        self.elapsed_seconds += dt
        targets = [bond.rest_length * self.scale * (
            1 + 0.015 * sin(self.elapsed_seconds * 1.6 + self.phase)
            + 0.065 * activity * sin(self.elapsed_seconds * 2.3 - bond.phase * 0.8 + self.phase)
        ) for bond in self.bonds]
        lambdas = [0.0] * len(self.bonds)
        for _ in range(self.SOLVER_ITERATIONS):
            for index, bond in enumerate(self.bonds):
                a, b = nodes[bond.first], nodes[bond.second]
                dx, dy = b.position[0] - a.position[0], b.position[1] - a.position[1]
                distance = hypot(dx, dy)
                nx, ny = (dx / distance, dy / distance) if distance > 1e-9 else bond.rest_direction
                inverse_a, inverse_b = 1 / a.mass, 1 / b.mass
                compliance = bond.compliance / (dt * dt)
                change = (-(distance - targets[index]) - compliance * lambdas[index]) / (inverse_a + inverse_b + compliance)
                lambdas[index] += change
                # Additional hard extension limit catches unusually strong pulls.
                if distance > targets[index] * 1.75:
                    change -= (distance - targets[index] * 1.75) / (inverse_a + inverse_b)
                a.position = (a.position[0] - nx * change * inverse_a, a.position[1] - ny * change * inverse_a)
                b.position = (b.position[0] + nx * change * inverse_b, b.position[1] + ny * change * inverse_b)
            for first, a in enumerate(nodes):
                for b in nodes[first + 1:]:
                    dx, dy = b.position[0] - a.position[0], b.position[1] - a.position[1]
                    minimum = (a.radius + b.radius) * 0.92
                    distance = hypot(dx, dy)
                    if distance < minimum:
                        nx, ny = (dx / distance, dy / distance) if distance > 1e-9 else (1.0, 0.0)
                        inverse_a, inverse_b = 1 / a.mass, 1 / b.mass
                        correction = (minimum - distance) / (inverse_a + inverse_b)
                        a.position = (a.position[0] - nx * correction * inverse_a, a.position[1] - ny * correction * inverse_a)
                        b.position = (b.position[0] + nx * correction * inverse_b, b.position[1] + ny * correction * inverse_b)
            for node in nodes:
                radius = node.radius * MEMBRANE_RADIUS
                if not (radius <= node.position[0] <= world_size[0] - radius
                        and radius <= node.position[1] <= world_size[1] - radius):
                    self.wall_contacts += 1
                node.position = (min(world_size[0] - radius, max(radius, node.position[0])),
                                 min(world_size[1] - radius, max(radius, node.position[1])))
        for node, old in zip(nodes, old_positions):
            vx, vy = (node.position[0] - old[0]) / dt, (node.position[1] - old[1]) / dt
            magnitude = hypot(vx, vy)
            if magnitude > maximum_speed:
                vx, vy = vx * maximum_speed / magnitude, vy * maximum_speed / magnitude
            node.velocity = (vx, vy)
