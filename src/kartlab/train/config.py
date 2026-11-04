"""Training configuration, loaded from YAML (see ``configs/``)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from kartlab.common.io import read_yaml
from kartlab.env.kart_env import EnvConfig

DEFAULT_EVAL_TRACKS = tuple(f"eval:{i}" for i in range(1, 11))
DEFAULT_MILESTONES = (0, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000)


@dataclass(frozen=True)
class PPOParams:
    learning_rate: float = 3e-4
    n_steps: int = 512  # per env and rollout
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: tuple[int, ...] = (64, 64)
    log_std_init: float = -0.5


@dataclass(frozen=True)
class CurriculumStage:
    """From ``from_steps`` on, new random tracks use these ranges (None keeps the current)."""

    from_steps: int
    track_length: tuple[float, float] | None = None
    track_width: tuple[float, float] | None = None
    track_curviness: tuple[float, float] | None = None


@dataclass(frozen=True)
class TrainConfig:
    name: str
    seed: int = 0
    total_steps: int = 1_000_000
    n_envs: int = 8
    vec_env: str = "subproc"  # "subproc" or "dummy"
    torch_threads: int = 1
    low_priority: bool = True
    normalize_obs: bool = False  # observations are already scaled to [-1, 1]
    normalize_reward: bool = True
    env: dict[str, Any] = field(default_factory=dict)  # EnvConfig overrides
    ppo: PPOParams = field(default_factory=PPOParams)
    checkpoint_every: int = 20_000
    checkpoint_at: tuple[int, ...] = DEFAULT_MILESTONES  # log-scale milestones for videos
    eval_every: int = 100_000
    eval_tracks: tuple[str, ...] = DEFAULT_EVAL_TRACKS
    eval_laps: int = 3
    curriculum: tuple[CurriculumStage, ...] = ()

    def __post_init__(self) -> None:
        if self.vec_env not in ("subproc", "dummy"):
            raise ValueError("vec_env must be 'subproc' or 'dummy'")
        if self.n_envs < 1 or self.torch_threads < 1:
            raise ValueError("n_envs and torch_threads must be at least 1")
        self.env_config()  # fail early on unknown or invalid env options

    def env_config(self) -> EnvConfig:
        known = {f.name for f in fields(EnvConfig)}
        unknown = set(self.env) - known
        if unknown:
            raise ValueError(f"unknown env options: {sorted(unknown)}")
        return EnvConfig(**{k: _tuple(v) for k, v in self.env.items()})

    def to_dict(self) -> dict[str, Any]:
        return _plain(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        data = dict(data)
        _reject_unknown(cls, data)
        if "ppo" in data:
            _reject_unknown(PPOParams, data["ppo"])
            data["ppo"] = PPOParams(**{k: _tuple(v) for k, v in data["ppo"].items()})
        if "curriculum" in data:
            stages = []
            for stage in data["curriculum"]:
                _reject_unknown(CurriculumStage, stage)
                stages.append(CurriculumStage(**{k: _tuple(v) for k, v in stage.items()}))
            data["curriculum"] = tuple(stages)
        for key in ("checkpoint_at", "eval_tracks"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Path) -> TrainConfig:
        return cls.from_dict(read_yaml(path))


def _reject_unknown(cls, data: dict[str, Any]) -> None:
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} options: {sorted(unknown)}")


def _tuple(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _plain(value: Any) -> Any:
    """Tuples to lists, recursively, so the dict dumps to clean YAML."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value
