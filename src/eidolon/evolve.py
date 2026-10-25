"""CPU-only evolutionary muscle training, resume, and held-out comparisons."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from statistics import fmean

from eidolon.evolution import EvolutionConfig, EvolutionTrainer, evaluate_policy
from eidolon.muscle_controller import NeuralMuscleController
from eidolon.train import evaluate
from eidolon.training import EpisodeConfig, LocomotionEnvironment
from eidolon.run_store import RunLock, save_snapshot


def compare(controller, seeds, baseline=None):
    reports = {"trained": evaluate_policy(controller, controller.episode, seeds)}
    if baseline is not None:
        if baseline.episode != controller.episode:
            raise ValueError("baseline brain episode settings must match the trained brain")
        reports["initial_population_champion"] = evaluate_policy(baseline, controller.episode, seeds)
    for pattern in ("idle", "random", "scripted"):
        episodes = [evaluate(LocomotionEnvironment(replace(controller.episode, seed=seed)), pattern) for seed in seeds]
        reports[pattern] = dict(score=fmean(r["reward"] for r in episodes), distance=fmean(r["distance"] for r in episodes),
                                energy=fmean(r["muscle_energy"] for r in episodes), stable=fmean(r["stable"] for r in episodes),
                                success=fmean(r["reached_target"] for r in episodes), episodes=episodes)
    return dict(seeds=list(seeds), training_seeds=list(controller.training_seeds),
                held_out=bool(controller.training_seeds) and not bool(set(seeds).intersection(controller.training_seeds)), results=reports)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--output", required=True)
    train.add_argument("--generations", type=int, default=10)
    train.add_argument("--population", type=int, default=12)
    train.add_argument("--elites", type=int, default=3)
    train.add_argument("--seed", type=int, default=123)
    train.add_argument("--seeds", type=int, nargs="+", default=[0, 17, 42])
    train.add_argument("--hidden-layers", type=int, nargs="+", default=[16])
    train.add_argument("--mutation-std", type=float, default=0.15)
    train.add_argument("--mutation-rate", type=float, default=0.2)
    train.add_argument("--body", choices=("simple", "detailed"), default="simple")
    train.add_argument("--duration", type=float, default=20)
    train.add_argument("--target-distance", type=float, default=100)
    resume = commands.add_parser("resume")
    resume.add_argument("--checkpoint", required=True)
    resume.add_argument("--output", help="default: checkpoint directory")
    resume.add_argument("--generations", type=int, default=5, help="additional generations")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--brain", required=True)
    evaluation.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    evaluation.add_argument("--compare", action="store_true")
    evaluation.add_argument("--baseline-brain")
    evaluation.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            if len(args.seeds) != len(set(args.seeds)) or not 1 <= len(args.seeds) <= 32:
                raise ValueError("evaluation requires 1–32 unique seeds")
            controller = NeuralMuscleController.load(args.brain)
            if args.compare:
                baseline = NeuralMuscleController.load(args.baseline_brain) if args.baseline_brain else None
                report = compare(controller, args.seeds, baseline)
            else:
                report = dict(held_out=bool(controller.training_seeds) and not bool(set(args.seeds).intersection(controller.training_seeds)),
                              **evaluate_policy(controller, controller.episode, args.seeds))
            if args.output:
                from eidolon.experiment_io import write_json
                write_json(args.output, report)
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if not 1 <= args.generations <= 10000:
            raise ValueError("generations must be 1–10000")
        directory = Path(args.output) if args.output else Path(args.checkpoint).parent
        with RunLock(directory):
            if args.command == "train":
                if any((directory / name).exists() for name in ("live.json", "checkpoint.json", "best.json", "metrics.json", "initial-best.json")):
                    raise ValueError("output already contains training artifacts; use resume or a new directory")
                trainer = EvolutionTrainer(EvolutionConfig(seed=args.seed, population=args.population, elites=args.elites,
                    seeds=tuple(args.seeds), hidden_layers=tuple(args.hidden_layers), mutation_std=args.mutation_std,
                    mutation_rate=args.mutation_rate, episode=EpisodeConfig(body=args.body, duration=args.duration,
                                                                            target_distance=args.target_distance)))
            else:
                trainer = EvolutionTrainer.load(args.checkpoint)
                if (directory.resolve() != Path(args.checkpoint).parent.resolve()
                        and any((directory / name).exists() for name in ("live.json", "checkpoint.json", "best.json", "metrics.json"))):
                    raise ValueError("resume output contains another run; choose a new directory")
            for _ in range(args.generations):
                row = trainer.generation()
                if (directory / "live.json").exists():
                    # Keep the authoritative GUI snapshot ahead of derived artifacts too.
                    save_snapshot(directory, dict(format="eidolon.live-learning", version=1,
                        trainer=trainer.checkpoint_data(), candidates=[], seed_reports=[], paused=False, stopped=False))
                else:
                    trainer.save(directory)
                print(json.dumps({key: row[key] for key in row if key != "candidates"}, allow_nan=False), flush=True)
        return 0
    except (ValueError, OSError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print("Stopped. Resume from the last completed generation checkpoint.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
