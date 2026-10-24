from copy import deepcopy
from math import tanh

import numpy as np
import pytest

from eidolon.neural_network import FeedForwardNetwork


def test_seeded_initialization_shapes_biases_dtype_and_independent_parameters() -> None:
    first = FeedForwardNetwork((3, 5, 2), seed=-17)
    second = FeedForwardNetwork((3, 5, 2), seed=-17)
    assert first.to_parameters() == second.to_parameters()
    assert first.to_parameters() != FeedForwardNetwork((3, 5, 2), seed=18).to_parameters()
    assert [array.shape for array in first.weights] == [(5, 3), (2, 5)]
    assert [array.shape for array in first.biases] == [(5,), (2,)]
    for array in (*first.weights, *first.biases):
        assert array.dtype == np.float64 and not array.flags.writeable
        assert np.isfinite(array).all()
    assert all(not bias.any() for bias in first.biases)
    exported = first.to_parameters()
    exported["weights"][0][0][0] += 100
    assert first.to_parameters() == second.to_parameters()


def test_forward_matches_hand_calculated_two_layer_tanh_network() -> None:
    parameters = FeedForwardNetwork((2, 2, 1)).to_parameters()
    parameters["weights"] = [[[1, 0], [0, 1]], [[0.5, -0.25]]]
    parameters["biases"] = [[0.1, -0.2], [0.3]]
    network = FeedForwardNetwork.from_parameters(parameters)
    output = network.forward([0.2, 0.4])
    assert output.shape == (1,)
    assert output[0] == pytest.approx(tanh(0.5 * tanh(0.3) - 0.25 * tanh(0.2) + 0.3), abs=1e-14)
    assert network.to_parameters() == parameters


@pytest.mark.parametrize("layers", [(4, 2), (4, 3, 2), (4, 8, 5, 2)])
def test_forward_is_stateless_finite_and_bounded_for_configurable_layers(layers) -> None:
    network = FeedForwardNetwork(layers, seed=42)
    parameters = network.to_parameters()
    first = network.forward([-1, 0, 0.5, 1])
    assert first.shape == (2,) and np.isfinite(first).all()
    assert np.all(np.abs(first) <= 1)
    np.testing.assert_array_equal(first, network.forward([-1, 0, 0.5, 1]))
    np.testing.assert_array_equal(network.forward([0] * 4), [0, 0])
    assert network.to_parameters() == parameters


@pytest.mark.parametrize("layers", [None, (), (3,), (0, 2), (-1, 2), (True, 2),
                                          (2.5, 2), (257, 2), (256, 256, 256), (1,) * 9])
def test_invalid_or_excessively_large_architectures_are_rejected(layers) -> None:
    with pytest.raises(ValueError):
        FeedForwardNetwork(layers)


@pytest.mark.parametrize("seed", [True, 1.5, "17", None])
def test_invalid_seeds_are_rejected(seed) -> None:
    with pytest.raises(ValueError, match="seed"):
        FeedForwardNetwork((2, 1), seed)


@pytest.mark.parametrize("inputs", [[1], [[1, 2]], [float("nan"), 0], [0, float("inf")],
                                   ["1", 2], [True, 1], [1j, 2], np.array([1, 2], dtype=object), [[1], 2]])
def test_invalid_inputs_fail_with_useful_errors(inputs) -> None:
    with pytest.raises(ValueError, match="inputs"):
        FeedForwardNetwork((2, 1)).forward(inputs)


@pytest.mark.parametrize("key,value", [
    ("version", 2), ("version", True), ("activation", "relu"), ("format", "pickle"),
    ("layers", [2, True]), ("weights", []), ("biases", []),
    ("weights", [[[1, 2], [3, 4]]]), ("weights", [[[1, 2], [3]]]),
    ("weights", [[[True, 1]]]), ("weights", [[["1", 1]]]),
    ("weights", [[[float("nan"), 1]]]), ("biases", [[float("inf")]]),
    ("biases", [[1, 2]]), ("weights", [np.array([[1j, 2]])]),
])
def test_malformed_parameters_are_rejected(key, value) -> None:
    parameters = FeedForwardNetwork((2, 1)).to_parameters()
    parameters[key] = value
    with pytest.raises(ValueError):
        FeedForwardNetwork.from_parameters(parameters)


def test_missing_extra_and_non_mapping_documents_are_rejected() -> None:
    parameters = FeedForwardNetwork((2, 1)).to_parameters()
    missing, extra = deepcopy(parameters), deepcopy(parameters)
    del missing["weights"]
    extra["code"] = "print('never executed')"
    for document in (missing, extra, [], None):
        with pytest.raises(ValueError, match="exactly"):
            FeedForwardNetwork.from_parameters(document)


def test_large_finite_preactivations_saturate_but_overflow_is_reported() -> None:
    parameters = FeedForwardNetwork((1, 1)).to_parameters()
    parameters["weights"] = [[[1000]]]
    network = FeedForwardNetwork.from_parameters(parameters)
    assert network.forward([1])[0] == 1
    parameters["weights"] = [[[1e308]]]
    with pytest.raises(ValueError, match="overflow|nonfinite"):
        FeedForwardNetwork.from_parameters(parameters).forward([2])


def test_save_load_roundtrip_preserves_parameters_and_exact_outputs(tmp_path) -> None:
    network = FeedForwardNetwork((3, 8, 4, 2), 711)
    path = tmp_path / "models" / "example.json"
    network.save(path)
    loaded = FeedForwardNetwork.load(path)
    assert loaded.to_parameters() == network.to_parameters()
    np.testing.assert_array_equal(loaded.forward([1, -1, 0.2]), network.forward([1, -1, 0.2]))
    assert path.read_text(encoding="utf-8").startswith('{\n  "format": "eidolon.feedforward"')
    with pytest.raises(OSError):
        FeedForwardNetwork.load(tmp_path / "absent.json")


@pytest.mark.parametrize("contents", [b"not json", b"[]", b'{"weights": NaN}',
                                      b'{"weights": Infinity}', b"\xff", b"[" * 1100])
def test_malformed_files_fail_without_executing_code(tmp_path, contents) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(contents)
    with pytest.raises(ValueError):
        FeedForwardNetwork.load(path)


def test_oversize_model_is_rejected_before_decoding(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("eidolon.neural_network.MAX_FILE_BYTES", 32)
    path = tmp_path / "large.json"
    path.write_text(" " * 33)
    with pytest.raises(ValueError, match="file limit"):
        FeedForwardNetwork.load(path)
