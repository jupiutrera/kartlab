"""Policies driving karts: shared by the live viewer and the video tools."""

from __future__ import annotations

import math
import pickle
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from kartlab.env.kart_env import EnvConfig, KartEnv
from kartlab.sim.contact import RESTART_GRACE
from kartlab.sim.kart import DT
from kartlab.sim.track import Track
from kartlab.train.evaluate import model_predictor
from kartlab.train.run import Checkpoint

RESTART_SECONDS = 3.0  # stuck or off the track this long: back to the start line
VALUE_SECONDS = 10.0  # how much history of V(s) the brain view keeps
LIMITER_BAND = 2.0  # speed limiter: throttle fades out over this many m/s below the cap
CONTACT_HOLD = 1.0  # after a bump the stuck and off-track timers stay frozen this long (s)


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
    value: object | None = None  # batch obs -> batch V(s); None if the version cannot tell
    value_in_reward_units: bool = False  # False: V is still in VecNormalize's normalised units

    @property
    def label(self) -> str:
        return f"{format_steps(self.steps)} pasos"


def model_value(model, vecnormalize_path: Path | None = None):
    """Batch value function of an SB3 model, through the public API only
    (``policy.predict_values``). Returns ``(value, in_reward_units)``.

    The policy is trained on rewards divided by the running standard deviation of the
    discounted return (VecNormalize), so its raw output is in those normalised units.
    When the saved statistics are there, the output is multiplied back by that deviation:
    V(s) is then the expected discounted reward in the units of the reward function.
    """
    import torch

    stats = None
    if vecnormalize_path is not None and Path(vecnormalize_path).exists():
        with open(vecnormalize_path, "rb") as f:
            stats = pickle.load(f)
    normalize_obs = stats is not None and stats.norm_obs
    scale = 1.0
    if stats is not None and stats.norm_reward:
        scale = float(math.sqrt(stats.ret_rms.var + stats.epsilon))

    def value(obs: np.ndarray) -> np.ndarray:
        if normalize_obs:
            obs = stats.normalize_obs(obs)
        with torch.no_grad():
            values = model.policy.predict_values(torch.as_tensor(obs, dtype=torch.float32))
        return values.numpy().reshape(-1) * scale

    return value, stats is not None


def load_version(checkpoint: Checkpoint) -> Version:
    from stable_baselines3 import PPO

    model = PPO.load(checkpoint.model, device="cpu")
    value, in_reward_units = model_value(model, checkpoint.vecnormalize)
    return Version(
        checkpoint.steps,
        model_predictor(model, checkpoint.vecnormalize),
        value,
        in_reward_units,
    )


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
        # Racing against a human (see ``scenes.HumanRaceScene``): a speed cap (m/s, ``None``
        # for none) sets how hard the AI pushes, getting stuck puts the kart back where it is
        # instead of at the line, and ``origin`` is how far behind the line it started, so
        # ``race_distance`` is zero on the line.
        self.speed_cap: float | None = None
        self._throttle = 0.0  # throttle actually applied (after the limiter)
        self.restart_in_place = False
        self.origin = 0.0
        self.no_contact_until = 0.0  # race time before which other karts do not touch it (s)
        self._contact_left = 0.0
        self.trail: deque[tuple[float, float]] = deque(maxlen=14)
        # What the brain view shows: the observation the policy acted on, and V(s) at each
        # decision while tracking is on (a bounded history, about VALUE_SECONDS long).
        self.last_obs: np.ndarray | None = None
        self.values: deque[float] = deque(maxlen=max(round(VALUE_SECONDS * 60 / self.repeat), 2))
        self.track_value = False
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
        return self._throttle

    @property
    def action(self) -> tuple[float, float]:
        """The (steer, throttle) held right now: the deterministic output of the policy."""
        return self._action

    @property
    def race_distance(self) -> float:
        """Net metres driven along the centerline, counted from the start line (negative
        on a grid slot behind it). ``restart_here`` keeps it; ``restart`` starts again."""
        return self.racer.progress.distance - self.origin

    def touch(self) -> None:
        """A kart bumped into this one: the stuck and off-track timers wait for a while."""
        self._contact_left = CONTACT_HOLD

    def set_tracking(self, on: bool) -> None:
        """Start or stop computing V(s) at every decision (it costs nothing while off)."""
        if on != self.track_value:
            self.track_value = on
            self.values.clear()

    def place(self, start_s: float, lateral: float = 0.0) -> None:
        """Put the kart at an arc position and lateral offset (e.g. a grid slot)."""
        self.restart(start_s, lateral)
        self.restarts -= 1  # not a restart caused by getting stuck

    def restart(self, start_s: float = 0.0, lateral: float = 0.0) -> None:
        self.racer.reset(start_s, lateral)
        self.origin = -start_s
        self._contact_left = 0.0
        self.no_contact_until = 0.0
        self.env.hold_action(np.zeros(2))
        self._phase = 0
        self._best_distance = 0.0
        self._last_progress_time = 0.0
        self._off_since = None
        self.restarts += 1
        self.trail.clear()
        self.values.clear()

    def restart_here(self) -> None:
        """Put the kart back on the road at its current arc position: centred, along the
        track, standing still. The race distance, the laps and the time are kept (unlike
        ``restart``, which sends the kart back to the line), and for a short while other
        karts do not touch it."""
        racer = self.racer
        x, y, heading = racer.track.pose_at(racer.position.s)
        racer.displace(x - racer.kart.x, y - racer.kart.y)
        kart = racer.kart
        kart.heading, kart.vx, kart.vy, kart.steer = heading, 0.0, 0.0, 0.0
        self.env.hold_action(np.zeros(2))
        self._action = (0.0, 0.0)
        self._phase = 0
        self._best_distance = racer.progress.distance
        self._last_progress_time = racer.time
        self._off_since = None
        self._contact_left = 0.0
        self.no_contact_until = racer.time + RESTART_GRACE
        self.restarts += 1
        self.trail.clear()
        self.values.clear()

    def step(self) -> str | None:
        """Advance one physics step; returns "lap" or "restart" when one happens."""
        racer = self.racer
        if self._phase == 0:
            obs = self.env.observe()
            self.last_obs = obs
            action = self.version.predict(obs[None])[0]
            if self.track_value and self.version.value is not None:
                self.values.append(float(self.version.value(obs[None])[0]))
            self._action = self.env.hold_action(action)
        self._phase = (self._phase + 1) % self.repeat
        laps = racer.laps
        steer, throttle = self._action
        if self.speed_cap is not None and throttle > 0.0:
            # A speed limiter, not a cap on the pedal: asphalt drag makes the top speed almost
            # flat from 65 % to 100 % throttle, so a pedal cap would barely change the lap.
            room = (self.speed_cap - racer.speed) / LIMITER_BAND
            throttle = min(throttle, max(0.0, min(1.0, room)))
        self._throttle = throttle
        racer.step(steer, throttle)
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
        if self._contact_left > 0.0:  # being pushed around is not being stuck
            self._contact_left -= DT
            self._last_progress_time = racer.time
            if self._off_since is not None:
                self._off_since = racer.time
        stuck = racer.time - self._last_progress_time > RESTART_SECONDS
        lost = self._off_since is not None and racer.time - self._off_since > RESTART_SECONDS
        if stuck or lost:
            if self.restart_in_place:
                self.restart_here()
            else:
                self.restart()
            return "restart"
        return None
