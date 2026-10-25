from dataclasses import asdict, replace
from math import isfinite
import json
import subprocess
import sys

import pytest

from eidolon.config import PhysicsConfig
from eidolon.train import evaluate, main
from eidolon.training import EpisodeConfig, LocomotionEnvironment, OBSERVATION_NAMES, RewardConfig, reward_for_tick


def test_observation_and_action_dimensions_normalization_and_invalid_actions():
    environment = LocomotionEnvironment(EpisodeConfig(duration=1))
    assert len(environment.colony.nodes) == 10
    assert len(environment.observe()) == len(OBSERVATION_NAMES) == 30
    before = environment.metrics(), [asdict(n) for n in environment.colony.nodes]
    for actions in ([], [1] * 7, [0] * 9, [True] * 8, [float("nan")] * 8, [1.1] * 8):
        with pytest.raises(ValueError):
            environment.step(actions)
        assert (environment.metrics(), [asdict(n) for n in environment.colony.nodes]) == before
    result = environment.step([0.8] * 8)
    assert all(isfinite(v) and -1 <= v <= 1 for v in result.observation)
    assert len(result.observation) == 30 and result.info["muscle_energy"] > 0
    assert result.observation[12:20] == environment.colony.muscles.activations
    assert result.observation[20:28] == (0.8,) * 8


@pytest.mark.parametrize("body", ["simple", "detailed"])
@pytest.mark.parametrize("pattern", ["idle", "random", "scripted"])
def test_headless_episode_and_reset_reproduce_every_measurement(body, pattern):
    environment = LocomotionEnvironment(EpisodeConfig(body=body, duration=1))
    initial = environment.observe(), [asdict(n) for n in environment.colony.nodes]
    report = evaluate(environment, pattern)
    final = environment.observe(), [asdict(n) for n in environment.colony.nodes]
    assert report["stable"] and report["elapsed_seconds"] == 1
    assert not report["propulsion_enabled"]
    with pytest.raises(RuntimeError):
        environment.step([0] * 8)
    environment.reset()
    assert (environment.observe(), [asdict(n) for n in environment.colony.nodes]) == initial
    assert evaluate(environment, pattern) == report
    assert (environment.observe(), [asdict(n) for n in environment.colony.nodes]) == final


def test_fixed_action_interval_does_not_change_a_constant_action_trajectory():
    first = LocomotionEnvironment(EpisodeConfig(duration=2, action_interval=0.1))
    second = LocomotionEnvironment(EpisodeConfig(duration=2, action_interval=0.25))
    while not first.done:
        first.step([0.6] * 8)
    while not second.done:
        second.step([0.6] * 8)
    assert first.metrics() == second.metrics()
    assert [asdict(n) for n in first.colony.nodes] == [asdict(n) for n in second.colony.nodes]


def test_final_action_is_shortened_to_exact_duration():
    environment = LocomotionEnvironment(EpisodeConfig(duration=0.25, action_interval=0.1))
    results = [environment.step([0] * 8) for _ in range(3)]
    assert [r.truncated for r in results] == [False, False, True]
    assert results[-1].info["elapsed_seconds"] == 0.25
    assert results[-1].info["physics_ticks"] == 30


def test_rewards_use_material_motion_instead_of_requested_displacement(monkeypatch):
    environment = LocomotionEnvironment(EpisodeConfig(duration=0.5))
    reference = LocomotionEnvironment(EpisodeConfig(duration=0.5))
    advance = environment.colony.advance

    def misleading_return(*args, **kwargs):
        advance(*args, **kwargs)
        return (1e9, 0)

    monkeypatch.setattr(environment.colony, "advance", misleading_return)
    assert evaluate(environment, "idle") == evaluate(reference, "idle")


def test_left_target_and_explicit_seed_reset():
    environment = LocomotionEnvironment(EpisodeConfig(target_distance=-1, duration=20))
    assert environment.observe()[4] == -1
    report = evaluate(environment, "random")
    assert report["reached_target"] and report["distance"] <= -1 and report["target_progress"] >= 1
    observation = environment.reset(seed=42)
    other = LocomotionEnvironment(EpisodeConfig(seed=42, target_distance=-1, duration=20))
    assert observation == other.observe()
    assert evaluate(environment, "random") == evaluate(other, "random")


def test_training_has_no_starvation_or_energy_budget_while_normal_life_is_unchanged():
    environment = LocomotionEnvironment(EpisodeConfig(duration=100, action_interval=0.5))
    while not environment.done:
        result = environment.step([1] * 8)
    assert result.truncated and result.info["stable"]
    assert result.info["muscle_energy"] > 100
    assert environment.colony.muscles.activations == (1,) * 8


def test_training_forces_propulsion_off_and_internal_muscles_need_ground_traction(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("training used a drive or position translation")

    monkeypatch.setattr("eidolon.colony.ground_drive", forbidden)
    displacements = []
    for friction in (0, 0.55):
        environment = LocomotionEnvironment(EpisodeConfig(duration=20,
            physics=PhysicsConfig(propulsion_enabled=True, floor_friction=friction)))
        monkeypatch.setattr(environment.colony, "translate_to", forbidden)
        displacements.append(evaluate(environment, "scripted")["distance"])
        assert not environment.colony.physics.propulsion_enabled
    assert abs(displacements[0]) < 1e-7
    assert displacements[1] > 2


def test_reward_progress_is_signed_bounded_and_cycles_cancel():
    config = RewardConfig(energy_cost=0, unproductive_cost=0, fragment_cost=0)
    assert reward_for_tick(0, 2, 10, 0, 0.1, config) == pytest.approx(0.2)
    assert reward_for_tick(2, 0, 10, 0, 0.1, config) == pytest.approx(-0.2)
    assert sum(reward_for_tick(a, b, 10, 0, 0.1, config) for a, b in ((0, 2), (2, 0))) == 0
    assert reward_for_tick(0, 100, 10, 0, 0.1, config) == 1
    assert reward_for_tick(0, -100, 10, 0, 0.1, config) == -1


def test_energy_idle_fragment_and_instability_penalties():
    config = RewardConfig()
    assert reward_for_tick(0, 1, 10, 2, 0.1, config) == pytest.approx(0.1 - 0.04)
    assert reward_for_tick(0, 0, 10, 0, 0.1, config) == pytest.approx(-0.001)
    assert reward_for_tick(0, 0, 10, 0, 0.1, config, components=2) == pytest.approx(-0.011)
    assert reward_for_tick(0, 999, 10, 0, 0.1, config, reached=True, stable=False) == -5
    assert reward_for_tick(9, 10, 10, 0, 0.1, config, reached=True) == pytest.approx(5.1)


def test_target_bonus_is_awarded_once_and_ends_physical_episode():
    environment = LocomotionEnvironment(EpisodeConfig(duration=30, target_distance=2))
    report = evaluate(environment, "scripted")
    assert report["reached_target"] and report["elapsed_seconds"] < 30
    assert report["reward"] > 4
    with pytest.raises(RuntimeError):
        environment.step([0] * 8)


@pytest.mark.parametrize("fault", ["nan", "teleport", "node_loss", "mass", "velocity", "propulsion"])
def test_physics_faults_never_earn_progress_and_return_finite_observations(fault):
    environment = LocomotionEnvironment(EpisodeConfig(duration=1))
    if fault == "nan":
        environment.colony.nodes[0].position = (float("nan"), 400)
    elif fault == "teleport":
        environment.colony.translate_to((700, 770))
    elif fault == "node_loss":
        environment.colony.nodes.pop()
    elif fault == "mass":
        environment.colony.nodes[0].mass *= 10
    elif fault == "velocity":
        environment.colony.nodes[0].velocity = (1e9, 0)
    else:
        environment.colony.physics = PhysicsConfig()
    result = environment.step([0] * 8)
    assert result.terminated and not result.info["stable"]
    assert result.reward == -5 and result.info["target_progress"] == 0
    assert all(isfinite(v) for v in result.observation)


def test_fragment_membership_changes_do_not_create_centroid_progress():
    environment = LocomotionEnvironment(EpisodeConfig(duration=1))
    center = environment.colony.material_center
    environment.colony.bonds = []
    environment.colony.rebuild_topology()
    assert environment.colony.material_center == center
    result = environment.step([0] * 8)
    assert result.reward <= 0 and abs(result.info["distance"]) < 0.01


@pytest.mark.parametrize("changes", [dict(body="bad"), dict(seed=True), dict(timestep=0), dict(timestep=0.1),
    dict(duration=0), dict(duration=float("nan")), dict(action_interval=0.00000000001),
    dict(action_interval=0.012), dict(settle_seconds=-1), dict(target_distance=0),
    dict(target_distance=1000), dict(target_distance=float("inf")), dict(world_size=(100, 100))])
def test_invalid_episode_configuration(changes):
    with pytest.raises(ValueError):
        EpisodeConfig(**changes)


def test_cli_headless_process_never_imports_pygame_or_numpy_and_reports_json():
    result = subprocess.run([sys.executable, "-c", """
import sys
sys.modules['pygame'] = None
sys.modules['numpy'] = None
from eidolon.train import main
raise SystemExit(main(['--duration', '0.5', '--pattern', 'random', '--benchmark']))
"""], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["episodes"][0]["nodes"] == 10
    assert output["benchmark"]["physics_ticks_per_second"] > 0


def test_cli_patterns_body_seeds_and_validation(capsys):
    assert main(["--duration", "0.5", "--body", "detailed", "--pattern", "random", "--episodes", "2"]) == 0
    reports = json.loads(capsys.readouterr().out)["episodes"]
    assert [r["seed"] for r in reports] == [17, 18]
    assert all(85 <= r["nodes"] <= 98 and r["stable"] for r in reports)
    for options in (["--hz", "0"], ["--episodes", "0"], ["--duration", "-1"], ["--pattern", "bad"]):
        with pytest.raises(SystemExit) as error:
            main(options)
        assert error.value.code == 2
