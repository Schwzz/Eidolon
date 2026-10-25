"""Bounded JSON and atomic experiment snapshots; no executable serialization."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

from eidolon.config import MuscleConfig, PhysicsConfig
from eidolon.training import EpisodeConfig, RewardConfig


def read_json(path, max_bytes=32 * 1024 * 1024):
    def invalid(value):
        raise ValueError(f"JSON must contain finite numbers, got {value}")

    with Path(path).open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("experiment JSON exceeds size limit")
    try:
        return json.loads(data.decode("utf-8"), parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"invalid experiment JSON: {error}") from error


def write_json(path, data):
    data = json.dumps(data, allow_nan=False, indent=2) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def restore_episode(data):
    if not isinstance(data, dict) or set(data) != set(EpisodeConfig.__dataclass_fields__):
        raise ValueError("invalid episode configuration fields")
    try:
        values = dict(data)
        values["world_size"] = tuple(values["world_size"])
        for name, kind in (("physics", PhysicsConfig), ("muscles", MuscleConfig), ("rewards", RewardConfig)):
            if not isinstance(values[name], dict) or set(values[name]) != set(kind.__dataclass_fields__):
                raise ValueError(f"invalid {name} configuration fields")
            values[name] = kind(**values[name])
        return EpisodeConfig(**values)
    except (TypeError, OverflowError) as error:
        raise ValueError(f"invalid episode configuration: {error}") from error


def episode_parameters(config):
    return asdict(config)
