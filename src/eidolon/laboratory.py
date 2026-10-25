"""Read-only laboratory views. Training and inference never run while drawing."""

from math import hypot

import pygame

from eidolon.rendering import ChamberRenderer


VIEWS = ("simulation", "neural", "training", "split")


def neural_activity(simulation):
    controller = simulation.controller
    network = getattr(controller, "network", None)
    if network is None or getattr(simulation, "muscle_test", None) is not None:
        return None, None, "No neural controller active"
    if getattr(simulation, "is_replay", False) and simulation.environment.done:
        status = "Episode completed" if simulation.environment.metrics()["stable"] else "Episode stopped: unstable"
        return network, None, status + "; inference stopped"
    if not simulation.life.alive:
        return network, None, "Organism dead; inference stopped"
    trace = getattr(network, "last_activations", None)
    status = ("Awaiting first decision" if trace is None else "Paused: last decision" if simulation.paused else
              "Actual last decision (held between actions)")
    return network, trace, status


class LaboratoryRenderer:
    def __init__(self, chamber, *, view="simulation", metrics=None):
        self.chamber = chamber
        self.view = view
        self.metrics = metrics
        self.font = pygame.font.Font(None, 20)
        self.small = pygame.font.Font(None, 15)
        self._graph_key = None

    @staticmethod
    def tab_rects(size):
        width = size[0]
        edge = min(148, (width - 48) // 4)
        return [pygame.Rect(24 + index * edge, 49, edge - 4, 26) for index in range(4)]

    def select_at(self, position, size):
        for view, rect in zip(VIEWS, self.tab_rects(size)):
            if rect.collidepoint(position):
                self.view = view
                return True
        return False

    def _text(self, surface, text, position, color=ChamberRenderer.TEXT, small=False):
        surface.blit((self.small if small else self.font).render(str(text), True, color), position)

    def draw(self, surface, simulation, alpha):
        width, height = surface.get_size()
        if self.view == "simulation":
            self.chamber.draw(surface, simulation, alpha)
        elif self.view == "split":
            surface.fill(ChamberRenderer.BACKGROUND)
            self.chamber.draw(surface.subsurface((0, 0, width // 2, height)), simulation, alpha)
            self._neural(surface, pygame.Rect(width // 2 + 12, 92, width // 2 - 36, height - 150), simulation)
        else:
            self.chamber.draw(surface, None, alpha)
            rect = pygame.Rect(34, 92, width - 68, height - 150)
            if self.view == "neural":
                self._neural(surface, rect, simulation)
            else:
                self._training(surface, rect)
            pygame.draw.rect(surface, ChamberRenderer.BACKGROUND, (24, height - 36, width - 130, 29))
            status = ("EPISODE COMPLETE | R Reset" if getattr(simulation, "is_replay", False) and simulation.environment.done
                      else "SIMULATION PAUSED" if simulation.paused else "SIMULATION CONTINUES")
            self._text(surface, status + " | F1-F4 Tabs", (24, height - 29), ChamberRenderer.MUTED, small=True)
        pygame.draw.rect(surface, ChamberRenderer.BACKGROUND, (24, 48, width - 48, 29))
        labels = ("Simulation", "Neural Network", "Training", "Split View") if width >= 700 else ("Sim", "Neural", "Training", "Split")
        for view, label, rect in zip(VIEWS, labels, self.tab_rects(surface.get_size())):
            pygame.draw.rect(surface, (32, 58, 63) if view == self.view else (20, 30, 37), rect, border_radius=4)
            self._text(surface, label, (rect.x + 10, rect.y + 5), ChamberRenderer.ACCENT if view == self.view else ChamberRenderer.MUTED)

    def _neural(self, surface, rect, simulation):
        old_clip = surface.get_clip()
        surface.set_clip(rect.clip(old_clip))
        try:
            network, trace, status = neural_activity(simulation)
            self._text(surface, "NEURAL NETWORK", (rect.x + 8, rect.y + 5), ChamberRenderer.ACCENT)
            self._text(surface, status, (rect.x + 8, rect.y + 28), ChamberRenderer.MUTED, small=True)
            if network is None:
                self._text(surface, "Load --brain or select --controller neural.", (rect.x + 8, rect.y + 62), small=True)
                return
            if network.layers[0] == 30:
                from eidolon.training import OBSERVATION_NAMES
                input_names = OBSERVATION_NAMES
                output_names = tuple(f"M{i + 1}" for i in range(8))
                note = "Outputs: tanh -> (value+1)/2 muscle actions"
            else:
                from eidolon.neural_controller import INPUT_NAMES
                input_names, output_names = INPUT_NAMES, ("Steering", "Effort")
                note = "Compatibility steering network; two outputs"
            key = (id(network), tuple(rect), network.layers)
            if self._graph_key != key:
                self._graph_key = key
                graph = pygame.Surface(rect.size)
                graph.fill(ChamberRenderer.CHAMBER)
                left, right = min(170, rect.width * 0.32), rect.width - 68
                self._nodes = []
                for layer, count in enumerate(network.layers):
                    x = left + (right - left) * layer / (len(network.layers) - 1)
                    ys = [78 + index * max(1, rect.height - 146) / max(1, count - 1) for index in range(count)]
                    self._nodes.append([(round(x), round(y)) for y in ys])
                for layer, weight in enumerate(network.weights):
                    magnitude = max(1.0, float(abs(weight).max()))
                    for out_index, row in enumerate(weight):
                        for in_index, value in enumerate(row):
                            intensity = int(25 + min(1, abs(float(value)) / magnitude) * 90)
                            color = (30, intensity, max(30, intensity - 5)) if value >= 0 else (intensity, 35, intensity)
                            pygame.draw.line(graph, color, self._nodes[layer][in_index], self._nodes[layer + 1][out_index])
                self._graph = graph
            # Static edges are cached; colors/values below use the real inference snapshot.
            surface.blit(self._graph, rect.topleft)
            self._text(surface, "NEURAL NETWORK  " + " -> ".join(map(str, network.layers)), (rect.x + 8, rect.y + 5), ChamberRenderer.ACCENT)
            self._text(surface, status, (rect.x + 8, rect.y + 28), ChamberRenderer.MUTED, small=True)
            mouse = pygame.mouse.get_pos() if pygame.display.get_init() else (-100, -100)
            hovered = None
            for layer, nodes in enumerate(self._nodes):
                label = "Inputs" if layer == 0 else "Muscles (8)" if layer == len(self._nodes) - 1 and network.layers[-1] == 8 else "Outputs" if layer == len(self._nodes) - 1 else f"Hidden {layer}"
                self._text(surface, label, (rect.x + nodes[0][0] - 25, rect.y + 52), small=True)
                for index, (x, y) in enumerate(nodes):
                    value = float(trace[layer][index]) if trace is not None else None
                    strength = abs(value) if value is not None else 0
                    color = ((65, int(90 + strength * 150), 157) if value is not None and value >= 0
                             else (int(90 + strength * 150), 85, 170) if value is not None else (66, 78, 86))
                    point = (rect.x + x, rect.y + y)
                    pygame.draw.circle(surface, color, point, 4)
                    if layer == 0:
                        name = input_names[index]
                        limit = max(4, int((x - 12) / 6))
                        self._text(surface, name[:limit], (rect.x + 4, point[1] - 5), small=True)
                    elif layer == len(self._nodes) - 1:
                        self._text(surface, output_names[index], (point[0] + 9, point[1] - 5), small=True)
                    if hypot(mouse[0] - point[0], mouse[1] - point[1]) < 10:
                        bias = float(network.biases[layer - 1][index]) if layer else None
                        hovered = f"Layer {layer}, neuron {index + 1}: value {value if value is not None else 'inactive'}; bias {bias}"
            self._text(surface, "Green: positive weights/values   Purple: negative   Hover: actual value/bias",
                       (rect.x + 8, rect.bottom - 43), small=True)
            self._text(surface, hovered or note, (rect.x + 8, rect.bottom - 24), small=True)
        finally:
            surface.set_clip(old_clip)

    def _training(self, surface, rect):
        old_clip = surface.get_clip()
        surface.set_clip(rect.clip(old_clip))
        try:
            self._text(surface, "TRAINING — SAVED SNAPSHOT", (rect.x + 8, rect.y + 6), ChamberRenderer.ACCENT)
            self._text(surface, "No live training runs in this window. Reload with F5.", (rect.x + 8, rect.y + 32), small=True)
            if getattr(self, "metrics_error", None):
                self._text(surface, "Reload failed: " + self.metrics_error[:100], (rect.x + 8, rect.bottom - 12), (210, 110, 120), small=True)
            data = self.metrics
            if data is None or not data["history"]:
                self._text(surface, "Open --training-run runs/example/metrics.json", (rect.x + 8, rect.y + 70), small=True)
                return
            history, config = data["history"], data["config"]
            self._text(surface, f"{len(history)} completed generations | population {config['population']} | seeds {config['seeds']}",
                       (rect.x + 8, rect.y + 60), small=True)
            chart = pygame.Rect(rect.x + 40, rect.y + 105, max(30, rect.width - 70), max(40, rect.height * 0.45))
            pygame.draw.rect(surface, (20, 30, 37), chart)
            values = [row[key] for row in history for key in ("best", "mean")]
            low, high = min(values), max(values)
            span = max(0.01, high - low)
            self._text(surface, f"Training reward  {low:.3f} to {high:.3f}   best (green) / mean (purple)", (chart.x, chart.y - 22), small=True)
            indices = sorted(set([0, len(history) - 1, *range(0, len(history), max(1, len(history) // 300))]))
            for key, color in (("best", ChamberRenderer.ACCENT), ("mean", (160, 115, 183))):
                points = [(round(chart.x + index / max(1, len(history) - 1) * (chart.width - 1)),
                           round(chart.bottom - 8 - (history[index][key] - low) / span * (chart.height - 16))) for index in indices]
                if len(points) > 1:
                    pygame.draw.lines(surface, color, False, points, 2)
                else:
                    pygame.draw.circle(surface, color, points[0], 3)
            self._text(surface, f"Generation 1 to {len(history)}", (chart.x, chart.bottom + 6), small=True)
            y = chart.bottom + 35
            self._text(surface, "Best candidates in last completed generation", (rect.x + 8, y), small=True)
            for rank, candidate in enumerate(sorted(history[-1]["candidates"], key=lambda item: (-item["score"], item["index"]))[:5]):
                self._text(surface, f"#{candidate['index']}  reward {candidate['score']:+.4f}   dx {candidate['distance']:+.2f}   energy {candidate['energy']:.2f}",
                           (rect.x + 8, y + 22 + rank * 20), small=True)
        finally:
            surface.set_clip(old_clip)
