"""The versions (checkpoints) of a run, loaded on demand and cached.

Loading a checkpoint takes 60-130 ms, and the very first one about 8 s because PyTorch and
Stable-Baselines3 have to be imported. None of that may freeze a window, so the library
loads in a background thread: ``request`` starts a load, ``poll`` (called once per frame)
collects what finished. In synchronous mode (headless captures, tests) ``request`` loads
right away and the result is identical.
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from kartlab.show.drivers import Version, load_version
from kartlab.train.run import Checkpoint, RunPaths

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


def load_best_version(run: RunPaths) -> Version:
    """``best_model.zip`` (the best evaluated policy) as a driveable version."""
    from stable_baselines3 import PPO

    from kartlab.train.evaluate import model_predictor

    if not run.best_model.exists():
        raise SystemExit(f"{run.root} no tiene best_model.zip")
    model = PPO.load(run.best_model, device="cpu")
    return Version(int(model.num_timesteps), model_predictor(model, run.best_vecnormalize))


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
        self._cache: dict[int, Version] = {}
        self._futures: dict[int, Future] = {}
        self._failed: dict[int, str] = {}
        self._lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None
        self._since_refresh = 0.0
        if not synchronous:
            self._pool = ThreadPoolExecutor(1, thread_name_prefix="version-loader")
            self._pool.submit(warm_up_libraries)
        else:
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
                return load_best_version(self.run)
            steps = self.checkpoints[-1].steps  # no best model: the latest version
        checkpoint = self.checkpoint(steps)
        if checkpoint is None:
            raise FileNotFoundError(f"no existe el checkpoint de {steps} pasos")
        return load_version(checkpoint)

    def cached(self, steps: int) -> Version | None:
        return self._cache.get(steps)

    def failed(self, steps: int) -> str | None:
        return self._failed.get(steps)

    def is_loading(self, steps: int) -> bool:
        return steps in self._futures

    @property
    def busy(self) -> bool:
        return bool(self._futures)

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
        except (OSError, ValueError, EOFError, KeyError, RuntimeError) as error:
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
