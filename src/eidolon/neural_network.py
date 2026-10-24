"""Small feedforward tanh network, implemented with NumPy and no simulation imports."""

import json
from math import sqrt
from pathlib import Path

import numpy as np


MAX_WIDTH = 256
MAX_LAYERS = 8
MAX_PARAMETERS = 65_536
MAX_FILE_BYTES = 8 * 1024 * 1024
FORMAT = "eidolon.feedforward"


def _layers(value) -> tuple[int, ...]:
    try:
        sizes = tuple(value)
    except TypeError as error:
        raise ValueError("layers must be a sequence of integer widths") from error
    if not 2 <= len(sizes) <= MAX_LAYERS:
        raise ValueError(f"network must have 2–{MAX_LAYERS} layers, including input and output")
    if any(isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer))
           or not 1 <= size <= MAX_WIDTH for size in sizes):
        raise ValueError(f"layer widths must be integers between 1 and {MAX_WIDTH}")
    if sum((a + 1) * b for a, b in zip(sizes, sizes[1:])) > MAX_PARAMETERS:
        raise ValueError(f"network exceeds the {MAX_PARAMETERS} parameter limit")
    return tuple(int(size) for size in sizes)


def _numeric_values(value) -> bool:
    if isinstance(value, (list, tuple)):
        return all(_numeric_values(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.dtype.kind in "iuf"
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    if not _numeric_values(value):
        raise ValueError(f"{name} must contain real numbers, without booleans or strings")
    try:
        with np.errstate(over="raise", invalid="raise"):
            array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError, FloatingPointError) as error:
        raise ValueError(f"{name} must be a rectangular array of finite float64 numbers") from error
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numbers")
    array.setflags(write=False)
    return array


def _reject_constant(value):
    raise ValueError(f"network JSON must contain finite numbers, got {value}")


class FeedForwardNetwork:
    """Immutable parameters; weights have shape (output width, input width).

    Every layer applies tanh(W @ x + b), including the output. No training,
    recurrent state, random draws during inference, or serialized code is used.
    """

    def __init__(self, layers, seed: int = 0) -> None:
        self._layers = _layers(layers)
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise ValueError("network seed must be an integer")
        rng = np.random.default_rng(int(seed) % (1 << 64))
        weights, biases = [], []
        for inputs, outputs in zip(self.layers, self.layers[1:]):
            # Xavier normal initialization, scaled for tanh.
            weight = rng.standard_normal((outputs, inputs)) * sqrt(2 / (inputs + outputs))
            weight.setflags(write=False)
            bias = np.zeros(outputs, dtype=np.float64)
            bias.setflags(write=False)
            weights.append(weight)
            biases.append(bias)
        self._weights, self._biases = tuple(weights), tuple(biases)

    @property
    def layers(self) -> tuple[int, ...]:
        return self._layers

    @property
    def weights(self) -> tuple[np.ndarray, ...]:
        return self._weights

    @property
    def biases(self) -> tuple[np.ndarray, ...]:
        return self._biases

    def forward(self, inputs) -> np.ndarray:
        values = _array(inputs, (self.layers[0],), "inputs")
        try:
            with np.errstate(over="raise", invalid="raise"):
                for weight, bias in zip(self.weights, self.biases):
                    values = weight @ values + bias
                    if not np.isfinite(values).all():
                        raise ValueError("network inference produced nonfinite values")
                    values = np.tanh(values)
        except FloatingPointError as error:
            raise ValueError("network inference overflowed; check input and parameter magnitudes") from error
        return values

    def to_parameters(self) -> dict:
        """Return independent, JSON-compatible parameter data."""
        return dict(format=FORMAT, version=1, activation="tanh", layers=list(self.layers),
                    weights=[weight.tolist() for weight in self.weights],
                    biases=[bias.tolist() for bias in self.biases])

    @classmethod
    def from_parameters(cls, parameters: dict) -> "FeedForwardNetwork":
        keys = {"format", "version", "activation", "layers", "weights", "biases"}
        if not isinstance(parameters, dict) or set(parameters) != keys:
            raise ValueError(f"network parameters must contain exactly: {', '.join(sorted(keys))}")
        if (parameters["format"] != FORMAT or type(parameters["version"]) is not int
                or parameters["version"] != 1 or parameters["activation"] != "tanh"):
            raise ValueError("unsupported network format, version, or activation (expected version 1, tanh)")
        layers = _layers(parameters["layers"])
        weights, biases = parameters["weights"], parameters["biases"]
        if (not isinstance(weights, (list, tuple)) or not isinstance(biases, (list, tuple))
                or len(weights) != len(layers) - 1 or len(biases) != len(layers) - 1):
            raise ValueError("network needs one weight matrix and bias vector per layer transition")
        network = cls.__new__(cls)
        network._layers = layers
        network._weights = tuple(_array(weight, (b, a), f"weights[{index}]")
                                 for index, (weight, a, b) in enumerate(zip(weights, layers, layers[1:])))
        network._biases = tuple(_array(bias, (size,), f"biases[{index}]")
                                for index, (bias, size) in enumerate(zip(biases, layers[1:])))
        return network

    def save(self, path: str | Path) -> None:
        """Write a UTF-8 JSON snapshot; existing files are replaced explicitly."""
        data = json.dumps(self.to_parameters(), allow_nan=False, indent=2)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeedForwardNetwork":
        with Path(path).open("rb") as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(f"network JSON exceeds the {MAX_FILE_BYTES}-byte file limit")
        try:
            parameters = json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
            return cls.from_parameters(parameters)
        except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
            raise ValueError(f"invalid UTF-8 network JSON: {error}") from error
