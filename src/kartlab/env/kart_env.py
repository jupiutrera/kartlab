"""Gymnasium environment: one kart learns to drive from ray sensors.

Observation (float32, all in [-1, 1]):

- ``ray_count`` ray distances to the track edges, divided by the ray range ([0, 1]);
- forward speed divided by the kart's top speed;
- sine and cosine of the heading relative to the track direction;
- previous action (steer, throttle);
- with ``extra_observations``: steering wheel position and lateral (slip) speed.

Action (float32 in [-1, 1]): ``[steer, throttle]``; steer > 0 turns left, throttle < 0
brakes and then reverses. The agent acts every ``action_repeat`` physics steps
(4 x 1/60 s, about 15 Hz) and its action is held in between.

Reward per agent step: net progress along the centerline (negative when driving
backwards), a bonus per completed lap, a small penalty for jerky steering and a
penalty for leaving the track. Raw speed is never rewarded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from kartlab.sim.kart import DT, KartParams
from kartlab.sim.race import Racer
from kartlab.sim.sensors import RaySensor
from kartlab.sim.track import Track, TrackParams
from kartlab.sim.tracksets import resolve_track, sample_train_seed, train_track

LATERAL_SPEED_SCALE = 5.0  # m/s mapped to 1 in the optional slip observation


@dataclass(frozen=True)
class EnvConfig:
    # Track: a fixed one (showcase key, "eval:N" or a training seed) or, with None, a
    # new random training track every episode with params drawn from these ranges.
    track: str | None = None
    track_length: tuple[float, float] = (900.0, 900.0)
    track_width: tuple[float, float] = (12.0, 12.0)
    track_curviness: tuple[float, float] = (0.5, 0.5)

    # Start anywhere on the track (position, lateral offset, heading, speed) or on the line.
    random_start: bool = True
    start_lateral: float = 0.4  # fraction of the half width
    start_heading: float = 0.25  # max heading offset (rad)
    start_speed: float = 5.0  # max initial speed (m/s)

    action_repeat: int = 4
    ray_count: int = 11
    ray_fov_deg: float = 180.0
    ray_range: float = 60.0
    extra_observations: bool = False

    progress_reward: float = 0.1  # per metre of net progress
    lap_bonus: float = 10.0
    steer_change_penalty: float = 0.025  # times the squared change of the steer command
    off_track_penalty: float = 10.0
    terminate_off_track: bool = True  # False in visualisation: the kart rolls on the grass
    off_track_tolerance: float = 0.5  # how far past the edge the kart centre may go (m)

    stuck_seconds: float = 4.0  # end the episode without this much new progress...
    stuck_progress: float = 1.0  # ...measured in metres beyond the best so far
    max_seconds: float = 150.0  # time limit (truncation)
    max_laps: int | None = None  # end (success) after this many laps

    def __post_init__(self) -> None:
        if self.action_repeat < 1:
            raise ValueError("action_repeat must be at least 1")
        for name in ("track_length", "track_width", "track_curviness"):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"{name} must be (low, high)")


class KartEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": round(1 / (DT * 4))}

    def __init__(self, config: EnvConfig | None = None, **overrides: Any) -> None:
        self.config = replace(config or EnvConfig(), **overrides)
        cfg = self.config
        self.kart_params = KartParams()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        extra = 2 if cfg.extra_observations else 0
        low = np.concatenate([np.zeros(cfg.ray_count), -np.ones(5 + extra)]).astype(np.float32)
        high = np.ones(cfg.ray_count + 5 + extra, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self._fixed_track = resolve_track(cfg.track) if cfg.track is not None else None
        self.track: Track | None = None
        self.racer: Racer | None = None
        self.sensor: RaySensor | None = None
        self._prev_action = np.zeros(2)
        self._best_distance = 0.0
        self._best_time = 0.0

    # -- episode control -------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        track = options.get("track") or self._fixed_track or self._random_track()
        self._use_track(track)
        cfg, rng = self.config, self.np_random
        if cfg.random_start:
            self.racer.reset(
                start_s=float(rng.uniform(0.0, track.length)),
                lateral=float(rng.uniform(-1.0, 1.0)) * cfg.start_lateral * track.half_width,
                heading_offset=float(rng.uniform(-1.0, 1.0)) * cfg.start_heading,
                speed=float(rng.uniform(0.0, cfg.start_speed)),
            )
        else:
            self.racer.reset()
        self._prev_action = np.zeros(2)
        self._best_distance = 0.0
        self._best_time = 0.0
        return self._observe(), self._info(off_track=False, stuck=False)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        cfg, racer = self.config, self.racer
        if racer is None:
            raise RuntimeError("call reset() before step()")
        steer, throttle = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0).tolist()
        limit = self.track.half_width + cfg.off_track_tolerance
        laps_before = racer.laps
        progress = 0.0
        off_track = False
        for _ in range(cfg.action_repeat):
            progress += racer.step(steer, throttle)
            off_track = abs(racer.position.lateral) > limit
            if off_track and cfg.terminate_off_track:
                break

        reward = cfg.progress_reward * progress
        reward += cfg.lap_bonus * (racer.laps - laps_before)
        reward -= cfg.steer_change_penalty * (steer - self._prev_action[0]) ** 2
        self._prev_action = np.array([steer, throttle])

        terminated = False
        if off_track and cfg.terminate_off_track:
            reward -= cfg.off_track_penalty
            terminated = True
        if racer.progress.distance > self._best_distance + cfg.stuck_progress:
            self._best_distance = racer.progress.distance
            self._best_time = racer.time
        stuck = racer.time - self._best_time > cfg.stuck_seconds
        terminated = terminated or stuck
        if cfg.max_laps is not None and racer.laps >= cfg.max_laps:
            terminated = True
        # The tolerance absorbs the rounding of summing 1/60 s steps.
        truncated = not terminated and racer.time >= cfg.max_seconds - 1e-9
        return self._observe(), float(reward), terminated, truncated, self._info(off_track, stuck)

    # -- curriculum ------------------------------------------------------------------

    def set_track_ranges(
        self,
        length: tuple[float, float] | None = None,
        width: tuple[float, float] | None = None,
        curviness: tuple[float, float] | None = None,
    ) -> None:
        """Change the random track ranges from the next episode (for a curriculum)."""
        updates = {
            key: tuple(value)
            for key, value in (
                ("track_length", length),
                ("track_width", width),
                ("track_curviness", curviness),
            )
            if value is not None
        }
        self.config = replace(self.config, **updates)

    # -- internals -------------------------------------------------------------------

    def _random_track(self) -> Track:
        cfg, rng = self.config, self.np_random
        params = TrackParams(
            length=float(rng.uniform(*cfg.track_length)),
            width=float(rng.uniform(*cfg.track_width)),
            curviness=float(rng.uniform(*cfg.track_curviness)),
        )
        return train_track(sample_train_seed(rng), params)

    def _use_track(self, track: Track) -> None:
        if track is self.track:
            return
        cfg = self.config
        self.track = track
        self.racer = Racer(track, self.kart_params)
        self.sensor = RaySensor(track, cfg.ray_count, math.radians(cfg.ray_fov_deg), cfg.ray_range)

    def _observe(self) -> np.ndarray:
        cfg, racer, kart = self.config, self.racer, self.racer.kart
        rays = self.sensor.cast(kart.x, kart.y, kart.heading, racer.position.index)
        error = racer.heading_error()
        parts = [
            rays / cfg.ray_range,
            [
                kart.forward_speed / self.kart_params.max_speed,
                math.sin(error),
                math.cos(error),
                self._prev_action[0],
                self._prev_action[1],
            ],
        ]
        if cfg.extra_observations:
            parts.append([kart.steer, kart.lateral_speed / LATERAL_SPEED_SCALE])
        obs = np.concatenate(parts).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _info(self, off_track: bool, stuck: bool) -> dict[str, Any]:
        racer = self.racer
        return {
            "laps": racer.laps,
            "progress": racer.progress.distance,
            "lap_times": tuple(racer.lap_times),
            "time": racer.time,
            "speed": racer.speed,
            "off_track": off_track,
            "stuck": stuck,
            "track": self.track.name,
            "track_seed": self.track.seed,
        }
