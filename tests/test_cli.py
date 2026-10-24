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
