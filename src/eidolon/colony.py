"""Deterministic position-based biomass physics, independent of rendering.

Nodes and elastic bonds are separate data. Graph edits are local and deterministic;
detached components share the same biology and recover toward the largest part.
"""

from dataclasses import dataclass
from math import ceil, cos, exp, hypot, isfinite, sin, sqrt
from random import Random

from eidolon.interaction import GrabConstraint


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
    created_at: float = -1.0


class Colony:
    MAX_STEP = 1 / 120
    SOLVER_ITERATIONS = 5
    ROW_PLANS = ((3, 5, 6, 5, 4), (3, 4, 5, 6, 5, 3), (2, 4, 5, 6, 5, 3))
    BREAK_FORCE = 6000.0
    BREAK_STRAIN = 0.035
    BREAK_DWELL = 0.12
    RECONNECT_DELAY = 0.7

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
        self.grab: GrabConstraint | None = None
        self.broken_bonds = 0
        self.reconnected_bonds = 0
        self._tension: dict[tuple[int, int], float] = {}
        self._cooldown: dict[tuple[int, int], float] = {}
        self._next_break = 0.0
        self._next_reconnect = 0.0
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
        for first, a in enumerate(self.nodes):
            for second in range(first + 1, len(self.nodes)):
                b = self.nodes[second]
                dx, dy = b.position[0] - a.position[0], b.position[1] - a.position[1]
                distance = hypot(dx, dy)
                if distance <= self.spacing * 2.05:
                    compliance = 0.0005 if distance < self.spacing * 1.5 else 0.004
                    self.bonds.append(Bond(first, second, distance, compliance,
                                           (dx / distance, dy / distance), (a.position[0] - center[0]) / self.spacing))
        self.rebuild_topology()
        self.anchor_index = min(range(len(self.nodes)), key=lambda i: hypot(
            self.nodes[i].position[0] - center[0], self.nodes[i].position[1] - center[1]))
        self.begin_step()

    @property
    def center(self) -> Point:
        identifiers = self.components[0] if hasattr(self, "components") else range(len(self.nodes))
        return self._mean(identifiers, "position")

    @property
    def velocity(self) -> Point:
        identifiers = self.components[0] if hasattr(self, "components") else range(len(self.nodes))
        return self._mean(identifiers, "velocity")

    def _mean(self, identifiers, field: str) -> Point:
        mass = sum(self.nodes[index].mass for index in identifiers)
        return tuple(sum(getattr(self.nodes[index], field)[axis] * self.nodes[index].mass
                         for index in identifiers) / mass for axis in (0, 1))

    def rebuild_topology(self) -> None:
        """Refresh adjacency, connected components, and only intact membrane faces."""
        neighbors = [[] for _ in self.nodes]
        short = set()
        for bond in self.bonds:
            neighbors[bond.first].append(bond.second)
            neighbors[bond.second].append(bond.first)
            if bond.rest_length < self.spacing * 1.5:
                short.add((bond.first, bond.second))
        for node, connections in zip(self.nodes, neighbors):
            node.connections = tuple(sorted(connections))
        self.triangles = tuple(
            (a, b, c) for a in range(len(self.nodes)) for b in neighbors[a] if b > a
            for c in neighbors[b] if c > b
            and (a, b) in short and (a, c) in short and (b, c) in short
        )
        unseen, components = set(range(len(self.nodes))), []
        while unseen:
            pending, found = [min(unseen)], []
            while pending:
                identifier = pending.pop()
                if identifier in unseen:
                    unseen.remove(identifier)
                    found.append(identifier)
                    pending.extend(self.nodes[identifier].connections)
            components.append(tuple(sorted(found)))
        self.components = tuple(sorted(components, key=lambda group: (-len(group), group[0])))
        self._component_index = {node: index for index, group in enumerate(self.components) for node in group}

    def begin_grab(self, position: Point, padding: float = 0.0) -> int | None:
        if any(not isfinite(value) for value in position) or not isfinite(padding) or padding < 0:
            raise ValueError("grab coordinates/padding must be finite and padding nonnegative")
        self.release_grab()
        candidates = [(hypot(node.position[0] - position[0], node.position[1] - position[1]), node.identifier)
                      for node in self.nodes]
        distance, identifier = min(candidates)
        node = self.nodes[identifier]
        if distance > node.radius * MEMBRANE_RADIUS + padding:
            return None
        self.grab = GrabConstraint(identifier, position, (node.position[0] - position[0], node.position[1] - position[1]))
        self._tension.clear()
        return identifier

    def move_grab(self, position: Point) -> None:
        if any(not isfinite(value) for value in position):
            raise ValueError("grab coordinates must be finite")
        if self.grab is not None:
            self.grab.target = position

    def release_grab(self) -> None:
        self.grab = None
        self._tension.clear()

    def _tear(self, dt: float) -> None:
        if self.grab is None or self.grab.applied_force < self.BREAK_FORCE * self.scale:
            self._tension.clear()
            return
        eligible = []
        for bond in self.bonds:
            key = (bond.first, bond.second)
            if self.grab.node not in key or self.elapsed_seconds - bond.created_at < self.RECONNECT_DELAY:
                continue
            a, b = self.nodes[bond.first], self.nodes[bond.second]
            strain = hypot(a.position[0] - b.position[0], a.position[1] - b.position[1]) / (bond.rest_length * self.scale) - 1
            self._tension[key] = self._tension.get(key, 0.0) + dt if strain > self.BREAK_STRAIN else 0.0
            if self._tension[key] >= self.BREAK_DWELL:
                eligible.append((strain, key))
        if eligible and self.elapsed_seconds >= self._next_break:
            _, key = max(eligible, key=lambda value: (value[0], -value[1][0], -value[1][1]))
            self.bonds = [bond for bond in self.bonds if (bond.first, bond.second) != key]
            self._tension.pop(key, None)
            self._cooldown[key] = self.elapsed_seconds + self.RECONNECT_DELAY
            self._next_break = self.elapsed_seconds + 0.05
            self._next_reconnect = self.elapsed_seconds + self.RECONNECT_DELAY
            self.broken_bonds += 1
            self.rebuild_topology()

    def _heal(self) -> None:
        if len(self.components) == 1 or self.elapsed_seconds < self._next_reconnect:
            return
        held = self._component_index[self.grab.node] if self.grab is not None else None
        candidates = []
        reach = self.spacing * self.scale * 1.2
        for component_index, component in enumerate(self.components[1:], 1):
            if component_index == held or held == 0:
                continue
            for first in self.components[0]:
                for second in component:
                    key = tuple(sorted((first, second)))
                    if self.elapsed_seconds < self._cooldown.get(key, 0):
                        continue
                    a, b = self.nodes[key[0]], self.nodes[key[1]]
                    distance = hypot(a.position[0] - b.position[0], a.position[1] - b.position[1])
                    if 1e-6 < distance <= reach and hypot(a.velocity[0] - b.velocity[0], a.velocity[1] - b.velocity[1]) <= 90:
                        candidates.append((distance, key))
        if candidates:
            distance, key = min(candidates)
            a, b = self.nodes[key[0]], self.nodes[key[1]]
            self.bonds.append(Bond(*key, distance / self.scale, 0.001,
                                   ((b.position[0] - a.position[0]) / distance, (b.position[1] - a.position[1]) / distance),
                                   0.0, self.elapsed_seconds))
            self.bonds.sort(key=lambda bond: (bond.first, bond.second))
            self._next_reconnect = self.elapsed_seconds + 0.10
            self.reconnected_bonds += 1
            self.rebuild_topology()

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
        self.release_grab()
        for node in self.nodes:
            node.velocity = (0.0, 0.0)
        self.begin_step()

    def advance(self, dt: float, desired_velocity: Point, world_size: Point) -> Point:
        """Advance physics and return the primary component's material displacement."""
        if not isfinite(dt) or dt <= 0 or any(not isfinite(v) for v in desired_velocity):
            raise ValueError("physics timestep must be positive and inputs finite")
        if any(not isfinite(value) or value <= 0 for value in world_size):
            raise ValueError("physics world dimensions must be finite and positive")
        if min(world_size) <= max(node.radius for node in self.nodes) * MEMBRANE_RADIUS * 2:
            raise ValueError("physics world must fit the biomass nodes")
        count = max(1, ceil(dt / self.MAX_STEP - 1e-9))
        self.wall_contacts = 0
        old_positions = [node.position for node in self.nodes]
        for _ in range(count):
            self._step(dt / count, desired_velocity, world_size)
        primary = self.components[0]
        mass = sum(self.nodes[index].mass for index in primary)
        return tuple(sum((self.nodes[index].position[axis] - old_positions[index][axis]) * self.nodes[index].mass
                         for index in primary) / mass for axis in (0, 1))

    def _step(self, dt: float, desired: Point, world_size: Point) -> None:
        nodes = self.nodes
        center, velocity = self.center, self.velocity
        held = self._component_index[self.grab.node] if self.grab is not None else None
        main_goal = ((0.0, 0.0) if held == 0 else
                     tuple(value * (0.35 if len(self.components) > 1 else 1.0) for value in desired))
        desired_speed = hypot(*main_goal)
        direction = (main_goal[0] / desired_speed, main_goal[1] / desired_speed) if desired_speed else (0, 0)
        goals = [(main_goal, velocity)]
        for component_index, component in enumerate(self.components[1:], 1):
            fragment_velocity = self._mean(component, "velocity")
            if held == component_index:
                goal = (0.0, 0.0)
            else:
                distance, first, second = min(
                    (hypot(nodes[a].position[0] - nodes[b].position[0], nodes[a].position[1] - nodes[b].position[1]), a, b)
                    for a in component for b in self.components[0]
                )
                a, b = nodes[first], nodes[second]
                closing_speed = min(75.0, max(0.0, distance - self.spacing * self.scale * 0.84) * 3)
                goal = (velocity[0] + (b.position[0] - a.position[0]) / max(distance, 1e-6) * closing_speed,
                        velocity[1] + (b.position[1] - a.position[1]) / max(distance, 1e-6) * closing_speed)
            goals.append((goal, fragment_velocity))
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
            component_index = self._component_index[node.identifier]
            goal, component_velocity = goals[component_index]
            traction = min(1.8, max(0.25, 1 + projection * 0.9)) if component_index == 0 else 1.0
            fx = (goal[0] - component_velocity[0]) * 12 * traction
            fy = (goal[1] - component_velocity[1]) * 12 * traction
            if self.grab is not None and self.grab.node == node.identifier:
                grab_force = self.grab.force(node.position, node.velocity, world_size, node.radius * MEMBRANE_RADIUS, self.scale)
                fx += grab_force[0]
                fy += grab_force[1]
            vx = (node.velocity[0] + fx / node.mass * dt) * exp(-0.8 * dt)
            vy = (node.velocity[1] + fy / node.mass * dt) * exp(-0.8 * dt)
            # Damp relative motion and angular drift without forcing a pose or
            # rotating the colony to match its steering direction.
            relative_damping = exp(-2.0 * dt)
            vx = component_velocity[0] + (vx - component_velocity[0]) * relative_damping
            vy = component_velocity[1] + (vy - component_velocity[1]) * relative_damping
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
        self._tear(dt)
        self._heal()
