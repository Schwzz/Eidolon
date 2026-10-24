import pytest

from eidolon.__main__ import main


def test_cli_passes_window_and_frame_settings_to_application(monkeypatch) -> None:
    configs = []
    monkeypatch.setattr("eidolon.__main__.Application.run", lambda app: configs.append(app.config))
    assert main(["--width", "800", "--height", "600", "--max-fps", "120", "--seed", "42"]) == 0
    assert (configs[0].width, configs[0].height, configs[0].max_fps) == (800, 600, 120)
    assert configs[0].seed == 42


def test_cli_rejects_invalid_settings(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--max-fps", "-1"])
    assert error.value.code == 2
    assert "max_fps must be nonnegative" in capsys.readouterr().err


def test_cli_selects_architecture_network_seed_and_saves_then_loads_parameters(tmp_path, monkeypatch) -> None:
    from eidolon.neural_controller import NeuralController

    records = []
    monkeypatch.setattr("eidolon.__main__.Application.run", lambda app: records.append(app))
    path = tmp_path / "models" / "network.json"
    assert main(["--controller", "neural", "--seed", "42", "--network-seed", "123",
                 "--hidden-layers", "8", "4", "--save-network", str(path)]) == 0
    app = records[-1]
    assert app.config.seed == 42 and app.config.controller == "neural"
    assert app.simulation.controller.network.layers == (12, 8, 4, 2)
    expected = NeuralController.from_seed(123, (8, 4)).network.to_parameters()
    assert app.simulation.controller.network.to_parameters() == expected
    assert main(["--controller", "neural", "--load-network", str(path), "--seed", "17"]) == 0
    assert records[-1].simulation.controller.network.to_parameters() == expected
    assert records[-1].config.seed == 17


@pytest.mark.parametrize("options", [["--controller", "invalid"], ["--network-seed", "3"],
    ["--load-network", "missing.json"], ["--save-network", "unused.json"], ["--hidden-layers", "8"],
    ["--controller", "neural", "--hidden-layers", "0"],
    ["--controller", "neural", "--load-network", "missing.json"],
    ["--controller", "neural", "--load-network", ""],
    ["--controller", "neural", "--save-network", ""],
    ["--controller", "neural", "--load-network", "model.json", "--network-seed", "1"],
    ["--controller", "neural", "--load-network", "model.json", "--hidden-layers", "8"]])
def test_invalid_controller_options_fail_before_window_start(options, monkeypatch, capsys) -> None:
    def unexpected_start(app):
        pytest.fail("invalid model/options must not open a window")

    monkeypatch.setattr("eidolon.__main__.Application.run", unexpected_start)
    with pytest.raises(SystemExit) as error:
        main(options)
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_invalid_or_incompatible_model_file_reports_useful_error(tmp_path, capsys) -> None:
    from eidolon.neural_network import FeedForwardNetwork

    path = tmp_path / "network.json"
    path.write_text("not JSON")
    with pytest.raises(SystemExit):
        main(["--controller", "neural", "--load-network", str(path)])
    assert "invalid UTF-8 network JSON" in capsys.readouterr().err
    FeedForwardNetwork((2, 1)).save(path)
    with pytest.raises(SystemExit):
        main(["--controller", "neural", "--load-network", str(path)])
    assert "12 inputs and 2 outputs" in capsys.readouterr().err


def test_runtime_numeric_failure_is_reported_without_a_traceback(monkeypatch, capsys) -> None:
    def fail(app):
        raise ValueError("network inference overflowed")

    monkeypatch.setattr("eidolon.__main__.Application.run", fail)
    assert main(["--controller", "neural"]) == 1
    assert "EIDOLON could not run: network inference overflowed" in capsys.readouterr().err
