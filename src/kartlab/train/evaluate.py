"""Evaluate a policy on reserved tracks: N laps from the start line, deterministic actions.

All tracks run in lockstep so the policy is queried once per step for all of them.
Run from the command line with ``uv run python scripts/evaluate.py --help``.
"""

from __future__ import annotations

import argparse
import math
import pickle
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from kartlab.common.io import write_yaml_atomic
from kartlab.env.kart_env import EnvConfig, KartEnv
from kartlab.sim.tracksets import resolve_track

MIN_AVERAGE_SPEED = 6.0  # time limit: the laps at this average speed (m/s)


@dataclass(frozen=True)
class TrackResult:
    track: str
    success: bool  # completed every lap without leaving the track
    laps: int
    progress: float  # fraction of the target distance driven
    seconds: float
    lap_times: tuple[float, ...]
    end: str  # "laps", "off_track", "stuck" or "time"


@dataclass(frozen=True)
class EvalResult:
    tracks: tuple[TrackResult, ...]

    @property
    def success_rate(self) -> float:
        return float(np.mean([t.success for t in self.tracks]))

    @property
    def progress_mean(self) -> float:
        return float(np.mean([t.progress for t in self.tracks]))

    @property
    def lap_time_mean(self) -> float | None:
        times = [lap for t in self.tracks for lap in t.lap_times]
        return float(np.mean(times)) if times else None

    def score(self) -> tuple[float, float, float]:
        """Higher is better: success rate, then distance driven, then faster laps."""
        lap = self.lap_time_mean
        return self.success_rate, self.progress_mean, -lap if lap is not None else -math.inf


def evaluate(
    predict: Callable[[np.ndarray], np.ndarray],
    tracks: Sequence[str],
    env_config: EnvConfig,
    laps: int = 3,
) -> EvalResult:
    """``predict`` maps a batch of observations to a batch of actions."""
    envs, observations = [], []
    for choice in tracks:
        track = resolve_track(choice)
        config = replace(
            env_config,
            track=None,
            random_start=False,
            terminate_off_track=True,
            max_laps=laps,
            max_seconds=laps * track.length / MIN_AVERAGE_SPEED,
        )
        env = KartEnv(config)
        obs, _ = env.reset(seed=0, options={"track": track})
        envs.append(env)
        observations.append(obs)

    results: dict[int, TrackResult] = {}
    while len(results) < len(envs):
        active = [i for i in range(len(envs)) if i not in results]
        actions = predict(np.stack([observations[i] for i in active]))
        for i, action in zip(active, actions, strict=True):
            obs, _, terminated, truncated, info = envs[i].step(action)
            observations[i] = obs
            if terminated or truncated:
                target = laps * envs[i].track.length
                end = (
                    "laps"
                    if info["laps"] >= laps
                    else "off_track"
                    if info["off_track"]
                    else "stuck"
                    if info["stuck"]
                    else "time"
                )
                results[i] = TrackResult(
                    track=tracks[i],
                    success=end == "laps",
                    laps=info["laps"],
                    progress=max(0.0, min(info["progress"] / target, 1.0)),
                    seconds=info["time"],
                    lap_times=info["lap_times"],
                    end=end,
                )
    return EvalResult(tuple(results[i] for i in range(len(envs))))


def model_predictor(model, vecnormalize_path: Path | None = None):
    """Deterministic batch predictor for an SB3 model, normalising observations if the
    run used VecNormalize with ``norm_obs``."""
    normalizer = None
    if vecnormalize_path is not None and Path(vecnormalize_path).exists():
        with open(vecnormalize_path, "rb") as f:
            stats = pickle.load(f)
        if stats.norm_obs:
            normalizer = stats

    def predict(obs: np.ndarray) -> np.ndarray:
        if normalizer is not None:
            obs = normalizer.normalize_obs(obs)
        actions, _ = model.predict(obs, deterministic=True)
        return actions

    return predict


def main(argv: list[str] | None = None) -> None:
    from stable_baselines3 import PPO

    from kartlab.train.config import TrainConfig
    from kartlab.train.run import RunPaths, latest_run

    parser = argparse.ArgumentParser(description="Evalúa un modelo en los circuitos reservados.")
    parser.add_argument("--run", type=Path, help="carpeta del run (por defecto, el último)")
    parser.add_argument(
        "--checkpoint", default="best", help="best, latest o el número de pasos de un checkpoint"
    )
    parser.add_argument("--tracks", type=int, default=20, help="circuitos de evaluación (1-20)")
    parser.add_argument("--track", action="append", help="circuito concreto (se puede repetir)")
    parser.add_argument("--laps", type=int, default=3)
    args = parser.parse_args(argv)

    run = RunPaths(args.run) if args.run else latest_run()
    if run is None:
        raise SystemExit("No hay ningún run en runs/")
    config = TrainConfig.from_yaml(run.config)
    if args.checkpoint == "best":
        model_path, stats_path = run.best_model, run.best_vecnormalize
    else:
        checkpoint = (
            run.read_latest()
            if args.checkpoint == "latest"
            else run.checkpoint(int(args.checkpoint))
        )
        model_path, stats_path = checkpoint.model, checkpoint.vecnormalize
    tracks = args.track or [f"eval:{i}" for i in range(1, args.tracks + 1)]
    model = PPO.load(model_path, device="cpu")
    result = evaluate(model_predictor(model, stats_path), tracks, config.env_config(), args.laps)

    print(f"{'circuito':>12} {'éxito':>6} {'vueltas':>8} {'avance':>7} {'fin':>10}  vueltas (s)")
    for t in result.tracks:
        laps = " ".join(f"{x:.1f}" for x in t.lap_times)
        print(
            f"{t.track:>12} {'sí' if t.success else 'no':>6} {t.laps:>8} "
            f"{t.progress:>6.0%} {t.end:>10}  {laps}"
        )
    print(f"\nÉxito: {result.success_rate:.0%} ({args.laps} vueltas sin salirse)")
    out = run.root / f"eval_{model_path.stem}.yaml"
    write_yaml_atomic(
        out,
        {
            "model": model_path.name,
            "laps": args.laps,
            "success_rate": result.success_rate,
            "progress_mean": result.progress_mean,
            "lap_time_mean": result.lap_time_mean,
            "tracks": [{**asdict(t), "lap_times": list(t.lap_times)} for t in result.tracks],
        },
    )
    print(f"Resultados en {out}")
