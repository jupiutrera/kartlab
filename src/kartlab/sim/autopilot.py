"""Reference driver: pure pursuit on the centerline plus a curvature-based speed plan.

It proves that tracks and physics are drivable, gives reference lap times and lets
the debug view show a clean line. It is not used to train the agent.
"""

from __future__ import annotations

import math

import numpy as np

from kartlab.sim.kart import KartParams, cruise_speed, max_path_curvature
from kartlab.sim.race import Racer
from kartlab.sim.track import Track


def speed_profile(
    track: Track, params: KartParams, grip_margin: float = 0.8, brake_margin: float = 0.5
) -> np.ndarray:
    """Target speed at each centerline sample: corner limits plus braking distances."""
    bend = np.abs(track.curvature)
    bend = np.maximum.reduce([np.roll(bend, k) for k in range(-4, 5)])  # look a few metres around
    usable = params.max_curvature * grip_margin
    with np.errstate(divide="ignore"):
        limit = params.curvature_speed * np.sqrt(np.maximum(usable / bend - 1.0, 0.0))
    plan = np.minimum(limit, cruise_speed(params)).tolist()
    decel = (params.brake + params.rolling_drag) * brake_margin
    seg = track.segment_lengths.tolist()
    n = len(plan)
    for _ in range(2):  # two passes so braking wraps around the start line
        for i in range(n - 1, -1, -1):
            nxt = plan[(i + 1) % n]
            plan[i] = min(plan[i], math.sqrt(nxt * nxt + 2.0 * decel * seg[i]))
    return np.array(plan)


class Autopilot:
    def __init__(self, track: Track, params: KartParams | None = None) -> None:
        self.track = track
        self.params = params or KartParams()
        self.plan = speed_profile(track, self.params).tolist()

    def act(self, racer: Racer) -> tuple[float, float]:
        kart = racer.kart
        v = kart.forward_speed
        lookahead = 4.0 + 0.45 * abs(v)
        tx, ty, _ = self.track.pose_at(racer.position.s + lookahead)
        dx, dy = tx - kart.x, ty - kart.y
        alpha = math.remainder(math.atan2(dy, dx) - kart.heading, math.tau)
        wanted = 2.0 * math.sin(alpha) / max(math.hypot(dx, dy), 1e-6)
        steer = wanted / max_path_curvature(max(abs(v), 1.0), self.params)

        lead = self.track.index_at(racer.position.s + 0.3 * abs(v))
        target = self.plan[lead]
        throttle = (target - v) * 0.6
        return max(-1.0, min(1.0, steer)), max(-1.0, min(1.0, throttle))
