"""Training callbacks: checkpoints, CSV metrics, periodic evaluation and curriculum."""

from __future__ import annotations

import csv
import json
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from kartlab.common.io import atomic_write, write_text_atomic, write_yaml_atomic
from kartlab.env.kart_env import EnvConfig
from kartlab.train.config import CurriculumStage
from kartlab.train.evaluate import EvalResult, evaluate, model_predictor
from kartlab.train.run import RunPaths

METRIC_FIELDS = (
    "steps",
    "time_s",
    "steps_per_s",
    "episodes",
    "reward_mean",
    "length_mean",
    "laps_mean",
    "lap_episodes",
)
EVAL_FIELDS = ("steps", "time_s", "success_rate", "progress_mean", "lap_time_mean", "best")


def save_model(model, model_path, vecnormalize_path) -> None:
    """Write the policy and its VecNormalize statistics atomically."""

    def write_model(tmp):
        with open(tmp, "wb") as f:
            model.save(f)

    atomic_write(model_path, write_model)
    vecnormalize = model.get_vec_normalize_env()
    if vecnormalize is not None:
        atomic_write(vecnormalize_path, vecnormalize.save)


class CheckpointCallback(BaseCallback):
    """Save every ``every`` steps and at each milestone, then announce it in latest.json."""

    def __init__(self, run: RunPaths, every: int, milestones: Sequence[int]) -> None:
        super().__init__()
        self.run = run
        self.every = every
        self.milestones = sorted(set(milestones))
        self.saved: list[int] = []
        self._start = time.perf_counter()

    def _next_due(self) -> int:
        last = self.saved[-1] if self.saved else -1
        periodic = (last // self.every + 1) * self.every
        upcoming = [m for m in self.milestones if m > last]
        return min([periodic, *upcoming])

    def _on_training_start(self) -> None:
        self._start = time.perf_counter()
        if 0 in self.milestones and self.num_timesteps == 0:
            self.save()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_due():
            self.save()
        return True

    def save(self) -> None:
        steps = self.num_timesteps
        if self.saved and self.saved[-1] == steps:
            return
        checkpoint = self.run.checkpoint(steps)
        save_model(self.model, checkpoint.model, checkpoint.vecnormalize)
        latest = {
            "steps": steps,
            "model": checkpoint.model.name,
            "vecnormalize": checkpoint.vecnormalize.name,
            "time_s": round(time.perf_counter() - self._start, 1),
        }
        write_text_atomic(self.run.latest, json.dumps(latest))
        self.saved.append(steps)


class StopFileCallback(BaseCallback):
    """Stop training cleanly when ``path`` exists (the main menu's "Parar" button creates it).

    ``stopped`` tells ``train`` the run was cut short on purpose: it still saves the last
    checkpoint and the summary, but skips the final evaluation.
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.stopped = False

    def _on_step(self) -> bool:
        if self.path.exists():
            self.stopped = True
            return False
        return True


class MetricsCallback(BaseCallback):
    """One CSV row (and TensorBoard scalars) per rollout; records the first completed lap."""

    def __init__(self, run: RunPaths, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.run = run
        self.episodes: deque[tuple[float, float, int]] = deque(maxlen=100)
        self.total_episodes = 0
        self.first_lap: dict | None = None
        self._file = None
        self._writer = None

    def _on_training_start(self) -> None:
        self._start = time.perf_counter()
        self._last = (self.num_timesteps, self._start)
        new = not self.run.metrics.exists()
        # Kept open for the whole run (closed in _on_training_end): one row per rollout.
        self._file = open(self.run.metrics, "a", encoding="utf-8", newline="")  # noqa: SIM115
        self._writer = csv.writer(self._file)
        if new:
            self._writer.writerow(METRIC_FIELDS)
            self._file.flush()

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if self.first_lap is None and info.get("laps", 0) >= 1:
                self.first_lap = {
                    "steps": int(self.num_timesteps),
                    "seconds": round(time.perf_counter() - self._start, 1),
                }
                write_yaml_atomic(self.run.root / "first_lap.yaml", self.first_lap)
                if self.verbose:
                    print(f"Primera vuelta completada: {self.first_lap}", flush=True)
            episode = info.get("episode")
            if episode is not None:
                self.episodes.append((episode["r"], episode["l"], episode.get("laps", 0)))
                self.total_episodes += 1
        return True

    def _on_rollout_end(self) -> None:
        now = time.perf_counter()
        steps = self.num_timesteps
        rate = (steps - self._last[0]) / max(now - self._last[1], 1e-9)
        self._last = (steps, now)
        if self.episodes:
            rewards, lengths, laps = (
                np.array(x, dtype=float) for x in zip(*self.episodes, strict=True)
            )
            row_stats = (rewards.mean(), lengths.mean(), laps.mean(), int((laps >= 1).sum()))
        else:
            row_stats = (float("nan"), float("nan"), float("nan"), 0)
        reward, length, laps_mean, lap_episodes = row_stats
        row = (steps, round(now - self._start, 2), round(rate, 1), self.total_episodes,
               round(reward, 4), round(length, 1), round(laps_mean, 3), lap_episodes)  # fmt: skip
        self._writer.writerow(row)
        self._file.flush()
        self.logger.record("kart/laps_mean", laps_mean)
        self.logger.record("kart/steps_per_s", rate)
        if self.verbose:
            print(
                f"{steps:>9} pasos  {now - self._start:7.0f} s  {rate:6.0f} pasos/s  "
                f"recompensa {reward:8.2f}  vueltas {laps_mean:5.2f}",
                flush=True,
            )

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.close()


class EvalCallback(BaseCallback):
    """Evaluate on the reserved tracks every ``every`` steps and keep the best model."""

    def __init__(
        self,
        run: RunPaths,
        every: int,
        tracks: Sequence[str],
        env_config: EnvConfig,
        laps: int,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.run = run
        self.every = every
        self.tracks = tuple(tracks)
        self.env_config = env_config
        self.laps = laps
        self.best: tuple[float, float] | None = None
        self.best_result: EvalResult | None = None
        self.last_steps = -1

    def _on_training_start(self) -> None:
        self._start = time.perf_counter()
        self._next = self.every

    def _on_step(self) -> bool:
        if self.tracks and self.num_timesteps >= self._next:
            self.run_eval()
            self._next = (self.num_timesteps // self.every + 1) * self.every
        return True

    def run_eval(self) -> EvalResult | None:
        if not self.tracks or self.num_timesteps == self.last_steps:
            return None
        self.last_steps = self.num_timesteps
        vecnormalize = self.model.get_vec_normalize_env()
        stats = None
        if vecnormalize is not None and vecnormalize.norm_obs:
            stats = self.run.root / ".eval_vecnormalize.pkl"
            vecnormalize.save(stats)
        result = evaluate(
            model_predictor(self.model, stats), self.tracks, self.env_config, self.laps
        )
        best = self.best is None or result.score() > self.best
        if best:
            self.best = result.score()
            self.best_result = result
            save_model(self.model, self.run.best_model, self.run.best_vecnormalize)
        new = not self.run.evals.exists()
        with open(self.run.evals, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(EVAL_FIELDS)
            lap_time = result.lap_time_mean
            writer.writerow(
                (
                    self.num_timesteps,
                    round(time.perf_counter() - self._start, 1),
                    round(result.success_rate, 3),
                    round(result.progress_mean, 3),
                    "" if lap_time is None else round(lap_time, 2),
                    int(best),
                )  # fmt: skip
            )
        self.logger.record("eval/success_rate", result.success_rate)
        self.logger.record("eval/progress_mean", result.progress_mean)
        if self.verbose:
            print(
                f"Evaluación a {self.num_timesteps} pasos: éxito {result.success_rate:.0%}, "
                f"avance {result.progress_mean:.0%}{' (mejor)' if best else ''}",
                flush=True,
            )
        return result


class CurriculumCallback(BaseCallback):
    """Switch the random track ranges of every env when a stage's step count is reached."""

    def __init__(self, stages: Sequence[CurriculumStage], verbose: int = 1) -> None:
        super().__init__(verbose)
        self.stages = sorted(stages, key=lambda s: s.from_steps)
        self.current = -1

    def _on_step(self) -> bool:
        while (
            self.current + 1 < len(self.stages)
            and self.num_timesteps >= self.stages[self.current + 1].from_steps
        ):
            self.current += 1
            stage = self.stages[self.current]
            self.training_env.env_method(
                "set_track_ranges",
                length=stage.track_length,
                width=stage.track_width,
                curviness=stage.track_curviness,
            )
            self.logger.record("kart/curriculum_stage", self.current)
            if self.verbose:
                print(
                    f"Currículo: etapa {self.current} desde {self.num_timesteps} pasos", flush=True
                )
        return True
