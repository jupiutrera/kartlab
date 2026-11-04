"""Arcade kart physics on a fixed 1/60 s step.

Controls are ``steer`` in [-1, 1] (positive turns left) and ``throttle`` in [-1, 1]
(positive accelerates, negative brakes and then reverses). The steering wheel moves
towards the command at a limited rate, the available turning curvature shrinks with
speed, and a lateral grip term lets the kart slide a little in fast corners.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DT = 1.0 / 60.0  # physics step (s)
STOP_SPEED = 0.3  # below this forward speed, "brake" engages reverse (m/s)


@dataclass(frozen=True)
class KartParams:
    max_speed: float = 26.0  # engine limit on asphalt; with drag it settles near 24 m/s
    accel: float = 14.0  # full-throttle acceleration from standstill (m/s²)
    brake: float = 22.0  # full-brake deceleration (m/s²)
    reverse_accel: float = 6.0  # (m/s²)
    max_reverse_speed: float = 5.0  # (m/s)
    # Asphalt drag: a small constant part plus one proportional to speed. Even a light
    # throttle moves the kart from rest, while coasting at speed still slows ~2 m/s².
    rolling_drag: float = 0.5  # constant deceleration on asphalt (m/s²)
    linear_drag: float = 0.065  # extra deceleration per m/s of speed on asphalt (1/s)
    grass_max_speed: float = 9.0  # engine limit on grass (m/s)
    grass_drag: float = 5.0  # constant deceleration on grass (m/s²)
    overspeed_drag: float = 15.0  # extra deceleration above the surface's limit (m/s²)
    steer_rate: float = 3.0  # steering wheel travel per second towards a turn (1/s)
    steer_return_rate: float = 4.5  # travel per second back towards centre (1/s)
    max_curvature: float = 0.25  # path curvature at full lock when slow (1/m)
    curvature_speed: float = 8.3  # speed at which the available curvature halves (m/s)
    grip: float = 12.0  # lateral velocity damping on asphalt (1/s)
    grass_grip: float = 5.0  # lateral velocity damping on grass (1/s)
    length: float = 1.8  # (m)
    width: float = 1.2  # (m)


@dataclass(slots=True)
class KartState:
    x: float
    y: float
    heading: float
    vx: float = 0.0
    vy: float = 0.0
    steer: float = 0.0  # current steering wheel position in [-1, 1]

    @property
    def forward_speed(self) -> float:
        return self.vx * math.cos(self.heading) + self.vy * math.sin(self.heading)

    @property
    def lateral_speed(self) -> float:
        return -self.vx * math.sin(self.heading) + self.vy * math.cos(self.heading)


def cruise_speed(params: KartParams) -> float:
    """Speed where full throttle balances the drag on asphalt (m/s).

    Solves accel * (1 - (v / max_speed)^2) = rolling_drag + linear_drag * v.
    """
    a = params.accel / params.max_speed**2
    b = params.linear_drag
    c = params.rolling_drag - params.accel
    return (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)


def max_path_curvature(speed: float, params: KartParams) -> float:
    """Tightest path curvature (1/m) the kart can follow at ``speed`` with full lock."""
    return params.max_curvature / (1.0 + (speed / params.curvature_speed) ** 2)


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def step_kart(
    state: KartState,
    steer: float,
    throttle: float,
    on_grass: bool,
    params: KartParams,
    dt: float = DT,
) -> None:
    """Advance ``state`` by one step in place."""
    steer = _clamp(steer, -1.0, 1.0)
    throttle = _clamp(throttle, -1.0, 1.0)

    # Steering wheel with a limited rate: faster back to centre than into a turn.
    towards_centre = abs(steer) < abs(state.steer) or steer * state.steer < 0.0
    rate = params.steer_return_rate if towards_centre else params.steer_rate
    state.steer += _clamp(steer - state.steer, -rate * dt, rate * dt)

    cos_h, sin_h = math.cos(state.heading), math.sin(state.heading)
    v = state.vx * cos_h + state.vy * sin_h
    v_lat = -state.vx * sin_h + state.vy * cos_h

    top = params.grass_max_speed if on_grass else params.max_speed
    if throttle > 0.0:
        if v < -STOP_SPEED:
            accel = throttle * params.brake
        else:
            accel = throttle * params.accel * max(0.0, 1.0 - (v / top) ** 2)
    elif throttle < 0.0:
        if v > STOP_SPEED:
            accel = throttle * params.brake
        else:
            reverse = min(-v, params.max_reverse_speed) / params.max_reverse_speed
            accel = throttle * params.reverse_accel * (1.0 - reverse**2)
    else:
        accel = 0.0
    v += accel * dt

    # Drag always opposes motion and never reverses it.
    drag = params.grass_drag if on_grass else params.rolling_drag + params.linear_drag * abs(v)
    if abs(v) > top:
        drag += params.overspeed_drag
    v = math.copysign(max(abs(v) - drag * dt, 0.0), v)
    v = _clamp(v, -params.max_reverse_speed, params.max_speed)

    grip = params.grass_grip if on_grass else params.grip
    v_lat *= math.exp(-grip * dt)

    state.vx = v * cos_h - v_lat * sin_h
    state.vy = v * sin_h + v_lat * cos_h
    state.heading += v * state.steer * max_path_curvature(v, params) * dt
    state.heading = math.remainder(state.heading, math.tau)
    state.x += state.vx * dt
    state.y += state.vy * dt
