"""Policies driving karts: shared by the live viewer and the video tools."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from kartlab.env.kart_env import EnvConfig, KartEnv
from kartlab.sim.track import Track
from kartlab.train.evaluate import model_predictor
from kartlab.train.run import Checkpoint

RESTART_SECONDS = 3.0  # stuck or off the track this long: back to the start line


def format_steps(steps: int) -> str:
    if steps >= 1_000_000:
        return f"{steps / 1_000_000:.1f}M".replace(".", ",")
    if steps >= 1000:
        return f"{steps // 1000}k"
    return str(steps)


@dataclass
class Version:
    steps: int
    predict: object  # batch obs -> batch actions

    @property
    def label(self) -> str:
        return f"{format_steps(self.steps)} pasos"


def load_version(checkpoint: Checkpoint) -> Version:
    from stable_baselines3 import PPO

    model = PPO.load(checkpoint.model, device="cpu")
    return Version(checkpoint.steps, model_predictor(model, checkpoint.vecnormalize))


class PolicyDriver:
    """One kart driven by a policy, advanced one physics step per call."""

    def __init__(self, version: Version, track: Track, env_config: EnvConfig, color) -> None:
        self.version = version
        self.color = color
        config = replace(
            env_config,
            track=None,
            random_start=False,
            terminate_off_track=False,
            stuck_seconds=math.inf,
            max_seconds=math.inf,
            max_laps=None,
        )
        self.env = KartEnv(config)
        self.env.reset(seed=0, options={"track": track})
        self.repeat = config.action_repeat
        self.best_lap: float | None = None
        self.restarts = 0
        self.ramp = 0  # colour in the 3D view (0 is the newest version)
        self.trail: deque[tuple[float, float]] = deque(maxlen=14)
        self._steps = 0
        self._phase = 0
        self._action = (0.0, 0.0)
        self._best_distance = 0.0
        self._last_progress_time = 0.0
        self._off_since: float | None = None

    @property
    def racer(self):
        return self.env.racer

    @property
    def throttle(self) -> float:
        return self._action[1]

    def place(self, start_s: float, lateral: float = 0.0) -> None:
        """Put the kart at an arc position and lateral offset (e.g. a grid slot)."""
        self.restart(start_s, lateral)
        self.restarts -= 1  # not a restart caused by getting stuck

    def restart(self, start_s: float = 0.0, lateral: float = 0.0) -> None:
        self.racer.reset(start_s, lateral)
        self.env.hold_action(np.zeros(2))
        self._phase = 0
        self._best_distance = 0.0
        self._last_progress_time = 0.0
        self._off_since = None
        self.restarts += 1
        self.trail.clear()

    def step(self) -> str | None:
        """Advance one physics step; returns "lap" or "restart" when one happens."""
        racer = self.racer
        if self._phase == 0:
            obs = self.env.observe()
            action = self.version.predict(obs[None])[0]
            self._action = self.env.hold_action(action)
        self._phase = (self._phase + 1) % self.repeat
        laps = racer.laps
        racer.step(*self._action)
        self._steps += 1
        if self._steps % 6 == 0:
            self.trail.appendleft((racer.kart.x, racer.kart.y))

        if racer.progress.distance > self._best_distance + 1.0:
            self._best_distance = racer.progress.distance
            self._last_progress_time = racer.time
        off = abs(racer.position.lateral) > self.env.track.half_width + 0.5
        if off and self._off_since is None:
            self._off_since = racer.time
        elif not off:
            self._off_since = None
        if racer.laps > laps:
            lap = racer.lap_times[-1]
            self.best_lap = lap if self.best_lap is None else min(self.best_lap, lap)
            return "lap"
        stuck = racer.time - self._last_progress_time > RESTART_SECONDS
        lost = self._off_since is not None and racer.time - self._off_since > RESTART_SECONDS
        if stuck or lost:
            self.restart()
            return "restart"
        return None
