"""Visible, sequential evolutionary evaluation with a small per-frame work budget."""

from dataclasses import replace
from math import isfinite
from pathlib import Path
from statistics import fmean
from time import perf_counter

from eidolon.biology import LifeState
from eidolon.config import WorldConfig
from eidolon.evolution import EvolutionConfig, EvolutionTrainer
from eidolon.experiment_io import read_json, write_json
from eidolon.muscle_controller import NeuralMuscleController
from eidolon.organism import Symbiote
from eidolon.replay import MuscleReplay
from eidolon.run_store import RunLock, SnapshotWriter
from eidolon.training import EpisodeConfig, LocomotionEnvironment
from eidolon.world import World


class LiveLearning(MuscleReplay):
    is_replay = False
    is_live_learning = True

    def __init__(self, directory="runs/live", config=None, *, budget_ms=4.0, max_ticks=24):
        self.directory = Path(directory)
        self.lock = RunLock(self.directory)
        try:
            self.budget_ms, self.max_ticks = budget_ms, max_ticks
            if not isfinite(budget_ms) or budget_ms <= 0 or type(max_ticks) is not int or max_ticks < 1:
                raise ValueError("live frame budget and tick limit must be positive")
            self.candidates, self.seed_reports = [], []
            self._paused = self.stopped = False
            self.animation_speed = 1.0
            self.notice = "New run: evaluating initial neural population"
            if (self.directory / "live.json").exists():
                self._restore(read_json(self.directory / "live.json"))
                self.notice = "Resumed saved progress; unfinished attempt restarts reproducibly"
            elif (self.directory / "checkpoint.json").exists():
                self.trainer = EvolutionTrainer.load(self.directory / "checkpoint.json")
                self.notice = "Resumed completed-generation checkpoint"
            else:
                config = config if config is not None else EvolutionConfig(population=6, elites=2, seeds=(17, 18),
                    episode=EpisodeConfig(duration=6, target_distance=20))
                self.trainer = EvolutionTrainer(config)
            if self.trainer.config.episode.body != "simple":
                raise ValueError("Live Learning uses the ten-node simple body; detailed-body runs remain available headlessly")
            self.writer = SnapshotWriter(self.directory)
            self.save_error = None
            self.steps = 0
            self._closed = False
            self.total_ticks = 0
            self._transition = False
            self._start_attempt()
            self.save_progress()
        except BaseException:
            if hasattr(self, "writer"):
                self.writer.executor.shutdown(wait=True)
            self.lock.close()
            raise

    def _restore(self, data):
        if (not isinstance(data, dict) or set(data) != {"format", "version", "trainer", "candidates", "seed_reports", "paused", "stopped"}
                or data["format"] != "eidolon.live-learning" or type(data["version"]) is not int or data["version"] != 1
                or type(data["paused"]) is not bool or type(data["stopped"]) is not bool):
            raise ValueError("invalid live learning checkpoint")
        self.trainer = EvolutionTrainer.from_checkpoint_data(data["trainer"])
        self.candidates, self.seed_reports = data["candidates"], data["seed_reports"]
        if not isinstance(self.candidates, list) or not 0 <= len(self.candidates) < self.trainer.config.population:
            raise ValueError("invalid live candidate cursor")
        fields = {"index", "score", "distance", "energy", "stable", "success"}
        for index, row in enumerate(self.candidates):
            if (not isinstance(row, dict) or set(row) != fields or type(row["index"]) is not int or row["index"] != index
                    or any(type(row[key]) not in (int, float) or not isfinite(row[key]) for key in fields - {"index"})
                    or row["energy"] < 0 or not 0 <= row["stable"] <= 1 or not 0 <= row["success"] <= 1):
                raise ValueError("invalid completed live candidate result")
        if not isinstance(self.seed_reports, list) or len(self.seed_reports) >= len(self.trainer.config.seeds):
            raise ValueError("invalid live seed cursor")
        for index, row in enumerate(self.seed_reports):
            if (not isinstance(row, dict) or set(row) != {"seed", "reward", "distance", "muscle_energy", "stable", "reached_target"}
                    or type(row["seed"]) is not int or row["seed"] != self.trainer.config.seeds[index]
                    or any(type(row[key]) not in (int, float) or not isfinite(row[key]) for key in ("reward", "distance", "muscle_energy"))
                    or row["muscle_energy"] < 0 or type(row["stable"]) is not bool or type(row["reached_target"]) is not bool):
                raise ValueError("invalid completed live attempt result")
        self._paused, self.stopped = data["paused"], data["stopped"]

    @property
    def generation_number(self):
        return len(self.trainer.history) + 1

    @property
    def attempt_number(self):
        return ((self.generation_number - 1) * self.trainer.config.population * len(self.trainer.config.seeds)
                + len(self.candidates) * len(self.trainer.config.seeds) + len(self.seed_reports) + 1)

    @property
    def status(self):
        if self.save_error:
            return "SAVE ERROR — STOPPED"
        if self.stopped:
            return "STOPPED"
        if self.paused:
            return "PAUSED"
        return "NEXT ATTEMPT" if self._transition else "SETTLING" if self.environment.settling else "LEARNING"

    @property
    def best_score(self):
        return self.trainer.history[-1]["best"] if self.trainer.history else None

    def _start_attempt(self):
        config = self.trainer.config
        self.active_candidate = len(self.candidates)
        self.active_seed = config.seeds[len(self.seed_reports)]
        self.active_attempt = self.attempt_number
        self.episode = replace(config.episode, seed=self.active_seed)
        self.controller = NeuralMuscleController(self.trainer.population[self.active_candidate], self.episode, config.seeds)
        self.seed = self.active_seed
        self.world_config = WorldConfig(width=self.episode.world_size[0], height=self.episode.world_size[1],
                                        initial_food=0, food_capacity=0, regeneration_interval=0)
        self.world = World(self.world_config, self.seed, 0, self.episode.physics)
        self.organism = Symbiote(self.seed)
        self.environment = LocomotionEnvironment(self.episode, defer_settle=True)
        self.life = LifeState(self.colony.center, 0, self.episode.base_radius, 100, 100)
        self.controller.clear_activity()
        self._actions = (0.0,) * 8
        self._accumulator = 0.0
        self._transition = False
        self.intervened = False

    def reset(self):
        """Retry this candidate/seed without losing completed evaluations or weights."""
        self._start_attempt()
        self.notice = "Current attempt restarted; population and completed scores retained"

    def _finish_attempt(self):
        report = self.environment.metrics()
        self.notice = f"Attempt {self.active_attempt} finished: reward {report['reward']:+.4f}; starting next attempt"
        self.seed_reports.append({key: report[key] for key in ("seed", "reward", "distance", "muscle_energy", "stable", "reached_target")})
        if len(self.seed_reports) == len(self.trainer.config.seeds):
            rows = self.seed_reports
            self.candidates.append(dict(index=len(self.candidates), score=fmean(r["reward"] for r in rows),
                distance=fmean(r["distance"] for r in rows), energy=fmean(r["muscle_energy"] for r in rows),
                stable=fmean(r["stable"] for r in rows), success=fmean(r["reached_target"] for r in rows)))
            self.seed_reports = []
        if len(self.candidates) == self.trainer.config.population:
            row = self.trainer.finish_generation(self.candidates)
            self.candidates = []
            self.notice = f"Generation {row['generation']} completed: best {row['best']:+.4f}; real selection and mutation applied"
        self._transition = True
        self._accumulator = 0.0
        self.save_progress()

    def update(self, dt):
        if not isfinite(dt) or dt < 0:
            raise ValueError("frame time must be finite and nonnegative")
        self.poll_save()
        if self.paused or self.stopped:
            return
        if len(self.trainer.history) >= 10000:
            self.stopped = True
            self.notice = "Generation-history limit reached; N archives this run and starts a new one"
            self.save_progress()
            return
        self.steps += 1
        self._accumulator = min(0.25, self._accumulator + dt * self.speed)
        deadline = perf_counter() + self.budget_ms / 1000
        self.colony.begin_step()
        self.life.begin_step()
        for _ in range(self.max_ticks):
            if self._accumulator + 1e-10 < self.episode.timestep or perf_counter() >= deadline:
                break
            if self._transition:
                self._start_attempt()
                break  # A new attempt gets its own visible frame; no settling burst.
            if self.environment.settling:
                self.environment.settle_tick()
            else:
                new_action = self.environment._ticks % self.environment._action_ticks == 0
                if new_action:
                    self._actions = self.controller.act(self.environment.observe())
                self.environment.step(self._actions, ticks=1, new_action=new_action)
            self.total_ticks += 1
            self._accumulator = max(0.0, self._accumulator - self.episode.timestep)
            if self.environment.done:
                self._finish_attempt()
                break
        self.life.position, self.life.velocity = self.colony.center, self.colony.material_velocity
        self.life.age_seconds = self.elapsed_seconds
        self.life.muscle_energy_spent = self.environment.metrics()["muscle_energy"]
        self.organism.update(max(0.0, self.elapsed_seconds - self.organism.age_seconds))

    def begin_grab(self, position, padding=0.0):
        return None  # External forces would invalidate fitness; grabbing remains in Survival mode.

    def move_grab(self, position):
        pass

    def snapshot(self):
        return dict(format="eidolon.live-learning", version=1, trainer=self.trainer.checkpoint_data(),
                    candidates=list(self.candidates), seed_reports=list(self.seed_reports), paused=self.paused, stopped=self.stopped)

    def save_progress(self, *, wait=False):
        try:
            self.writer.submit(self.snapshot())
            if wait:
                self.writer.flush()
            self.save_error = None
        except (OSError, ValueError) as error:
            self.save_error = str(error)
            self.stopped = True
            if wait:
                raise

    def poll_save(self):
        try:
            self.writer.poll()
        except (OSError, ValueError) as error:
            self.save_error, self.stopped = str(error), True

    def toggle_stopped(self):
        if self.save_error:
            self.save_progress(wait=True)
        self.stopped = not self.stopped
        self.save_progress()

    def restart_learning(self):
        """Explicit new run; archive the complete previous state before replacing it."""
        self.save_progress(wait=True)
        index = 1
        while (self.directory / f"archive-{index:03}").exists():
            index += 1
        archive = self.directory / f"archive-{index:03}"
        write_json(archive / "live.json", self.snapshot())
        self.trainer.save(archive)
        for name in ("best.json", "initial-best.json"):
            (self.directory / name).unlink(missing_ok=True)
        self.trainer = EvolutionTrainer(self.trainer.config)
        self.candidates, self.seed_reports = [], []
        self._paused = self.stopped = False
        self._start_attempt()
        self.notice = f"New run started; previous progress archived in {archive.name}"
        self.save_progress()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.save_progress(wait=True)
        finally:
            try:
                self.writer.close()
            finally:
                self.lock.close()
