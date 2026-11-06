"""The versions (checkpoints) of a run, loaded on demand and cached.

Loading a checkpoint takes 60-130 ms, and the very first one about 8 s because PyTorch and
Stable-Baselines3 have to be imported. None of that may freeze a window, so the library
loads in a background thread: ``request`` starts a load, ``poll`` (called once per frame)
collects what finished. In synchronous mode (headless captures, tests) ``request`` loads
right away and the result is identical.
"""

from __future__ import annotations

import math
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import yaml

from kartlab.common.io import read_yaml
from kartlab.show.drivers import Version, load_version
from kartlab.show.overlays import version_name
from kartlab.train.config import TrainConfig
from kartlab.train.run import RUNS_DIR, Checkpoint, RunPaths

BEST = -1  # key of best_model.zip, the best evaluated policy
PROGRESSION_TARGETS = (0, 10_000, 30_000, 100_000, 300_000, 1_000_000)
REFRESH_SECONDS = 5.0  # how often the checkpoint folder is listed again (a run may be training)


def pick_checkpoints(
    run: RunPaths, targets: tuple[int, ...] = PROGRESSION_TARGETS
) -> list[Checkpoint]:
    """The checkpoints closest (on a log scale) to each target, plus the final one."""
    available = run.list_checkpoints()
    if not available:
        raise SystemExit(f"{run.root} no tiene checkpoints")
    picks: dict[int, Checkpoint] = {}
    final = available[-1]
    for target in targets:
        if target >= final.steps:
            continue
        best = min(available, key=lambda c: abs(math.log1p(c.steps) - math.log1p(target)))
        picks[best.steps] = best
    picks[final.steps] = final
    return [picks[s] for s in sorted(picks)]


def latest_run_with_checkpoints(runs_dir: Path = RUNS_DIR) -> RunPaths | None:
    """The most recent run that has a config and at least one checkpoint to show."""
    for folder in sorted(runs_dir.glob("*"), reverse=True):
        candidate = RunPaths(folder)
        if candidate.config.exists() and candidate.list_checkpoints():
            return candidate
    return None


def eval_success_rate(run: RunPaths) -> float | None:
    """``success_rate`` of ``eval_best_model.yaml`` (written by ``evaluate.py`` on the reserved
    tracks), or ``None`` if the run was not evaluated or the file is unreadable."""
    try:
        value = read_yaml(run.root / "eval_best_model.yaml")["success_rate"]
        rate = None if isinstance(value, bool) else float(value)
        return rate if rate is not None and math.isfinite(rate) else None
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        return None


def best_generalist_run(runs_dir: Path = RUNS_DIR) -> RunPaths | None:
    """The run to race against by default: the one with the highest success rate on the
    reserved evaluation tracks (a ``best_model.zip`` that drives them all), the most recent
    on a tie. Without any evaluated run, the latest one with checkpoints."""
    scored = []
    for folder in runs_dir.glob("*"):
        run = RunPaths(folder)
        rate = eval_success_rate(run)
        usable = run.config.exists() and run.best_model.exists() and run.list_checkpoints()
        if rate is not None and usable:
            scored.append((rate, folder.name, run))
    if scored:
        return max(scored, key=lambda item: item[:2])[2]
    return latest_run_with_checkpoints(runs_dir)


def load_best_version(run: RunPaths) -> Version:
    """``best_model.zip`` (the best evaluated policy) as a driveable version."""
    from stable_baselines3 import PPO

    from kartlab.show.drivers import model_value
    from kartlab.train.evaluate import model_predictor

    if not run.best_model.exists():
        raise SystemExit(f"{run.root} no tiene best_model.zip")
    model = PPO.load(run.best_model, device="cpu")
    value, in_reward_units = model_value(model, run.best_vecnormalize)
    return Version(
        int(model.num_timesteps),
        model_predictor(model, run.best_vecnormalize),
        value,
        in_reward_units,
    )


def warm_up_libraries() -> None:
    """Import PyTorch and Stable-Baselines3 (the slow part of the first load) and keep
    PyTorch to one thread: the policies are tiny and the render needs the other cores."""
    import torch
    from stable_baselines3 import PPO  # noqa: F401

    torch.set_num_threads(1)


class VersionLibrary:
    """Checkpoints of a run by step count, with a cache and background loading."""

    def __init__(self, run: RunPaths, synchronous: bool = False) -> None:
        self.run = run
        self.synchronous = synchronous
        self.checkpoints: list[Checkpoint] = run.list_checkpoints()
        try:
            self.normalize_obs = TrainConfig.from_yaml(run.config).normalize_obs
        except (OSError, ValueError, KeyError, TypeError):
            self.normalize_obs = False
        self._cache: dict[int, Version] = {}
        self._futures: dict[int, Future] = {}
        self._failed: dict[int, str] = {}
        self.notices: list[str] = []  # for the viewer to show once (read from any thread)
        self._pool: ThreadPoolExecutor | None = None
        self._since_refresh = 0.0
        if not synchronous:
            self._pool = ThreadPoolExecutor(1, thread_name_prefix="version-loader")
            self._warming: Future | None = self._pool.submit(warm_up_libraries)
        else:
            self._warming = None
            warm_up_libraries()

    # -- listing ----------------------------------------------------------------------------

    @property
    def steps(self) -> list[int]:
        return [c.steps for c in self.checkpoints]

    def __len__(self) -> int:
        return len(self.checkpoints)

    def refresh(self, dt: float) -> bool:
        """List the checkpoint folder again every few seconds; True if versions appeared."""
        self._since_refresh += dt
        if self._since_refresh < REFRESH_SECONDS:
            return False
        self._since_refresh = 0.0
        found = self.run.list_checkpoints()
        if len(found) == len(self.checkpoints):
            return False
        self.checkpoints = found
        return True

    def checkpoint(self, steps: int) -> Checkpoint | None:
        return next((c for c in self.checkpoints if c.steps == steps), None)

    # -- loading ----------------------------------------------------------------------------

    def _load(self, steps: int) -> Version:
        if steps == BEST:
            if self.run.best_model.exists():
                self._check_stats(self.run.best_vecnormalize, "best_model")
                return load_best_version(self.run)
            steps = self.checkpoints[-1].steps  # no best model: the latest version
        checkpoint = self.checkpoint(steps)
        if checkpoint is None:
            raise FileNotFoundError(f"no existe el checkpoint de {steps} pasos")
        self._check_stats(checkpoint.vecnormalize, version_name(checkpoint.steps))
        return load_version(checkpoint)

    def _check_stats(self, path, name: str) -> None:
        """A missing VecNormalize file is harmless when observations are not normalised (the
        default: they are already in [-1, 1] and only rewards are normalised); if the run did
        normalise them, the policy would see wrong inputs, so say so."""
        if self.normalize_obs and not path.exists():
            self.notices.append(
                f"Falta {path.name}: {name} se verá con observaciones sin normalizar"
            )

    def cached(self, steps: int) -> Version | None:
        return self._cache.get(steps)

    def failed(self, steps: int) -> str | None:
        return self._failed.get(steps)

    def clear_failure(self, steps: int) -> None:
        self._failed.pop(steps, None)

    def is_loading(self, steps: int) -> bool:
        return steps in self._futures

    @property
    def busy(self) -> bool:
        """True while the background thread is importing PyTorch or loading a version (the
        frame rate dips a little meanwhile, which is not a reason for the light mode)."""
        warming = self._warming is not None and not self._warming.done()
        return warming or bool(self._futures)

    def request(self, steps: int) -> Version | None:
        """Start loading ``steps`` (a no-op if cached or on its way). In synchronous mode
        the version is returned already loaded."""
        if steps in self._cache:
            return self._cache[steps]
        if steps in self._futures:
            return None
        self._failed.pop(steps, None)
        if self._pool is None:
            self._finish(steps, self._attempt(steps))
            return self._cache.get(steps)
        future = self._pool.submit(self._attempt, steps)
        self._futures[steps] = future
        return None

    def _attempt(self, steps: int) -> Version | str:
        try:
            return self._load(steps)
        except SystemExit as error:  # load_best_version reports with SystemExit
            return str(error)
        except Exception as error:  # a corrupt zip or pickle must never reach the main loop
            return str(error) or type(error).__name__

    def _finish(self, steps: int, result: Version | str) -> None:
        if isinstance(result, str):
            self._failed[steps] = result
        else:
            self._cache[steps] = result

    def poll(self) -> list[int]:
        """Collect the loads that finished; returns their keys (failed ones included)."""
        done = [steps for steps, future in self._futures.items() if future.done()]
        for steps in done:
            future = self._futures.pop(steps)
            if future.cancelled():
                continue
            self._finish(steps, future.result())
        return done

    def cancel_except(self, keep: set[int]) -> None:
        """Drop the loads that have not started and nobody wants any more."""
        for steps in [s for s in self._futures if s not in keep]:
            if self._futures[steps].cancel():
                del self._futures[steps]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
