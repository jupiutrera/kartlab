"""Manual sweep ("barrido"): a controlled slide with a push on the way out.

Hold the drift button while steering to start a sweep. The tyres give much less side
grip, so the kart slides and its nose turns faster than its path. The longer the
sweep, the bigger the push when the button is released: a continuous amount
proportional to the time swept, up to a cap. There are no levels, no countdown.

This is a manual-drive feature only. The RL environment never passes a ``DriftState``,
so training, the autopilot and every recorded lap keep the plain physics in
``kart.step_kart``. The sweep reuses that function with adjusted parameters, which
keeps one implementation of the longitudinal and lateral dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

from kartlab.sim.kart import KartParams, KartState, step_kart


@dataclass(frozen=True)
class DriftParams:
    # Entry and exit.
    entry_speed: float = 10.0  # minimum forward speed to start a sweep (m/s)
    entry_steer: float = 0.35  # minimum |steer| to start a sweep
    exit_speed: float = 7.0  # the sweep ends below this forward speed (m/s)
    # Dynamics while sweeping.
    grip: float = 2.5  # lateral velocity damping (1/s); 12 on asphalt normally
    turn_gain: float = 1.3  # how much tighter the nose turns than in normal driving
    scrub: float = 0.5  # extra deceleration from the sliding tyres (m/s²)
    # The wheel is remapped to [neutral - span, neutral + span] of full lock, always
    # in the sweep direction: steering into it tightens, counter-steering widens.
    steer_neutral: float = 0.65
    steer_span: float = 0.35
    # Push on release: proportional to the time swept, saturating at ``full_time``.
    full_time: float = 1.5  # time swept that gives the biggest push (s)
    boost_speed: float = 6.0  # extra top speed at the biggest push (m/s)
    boost_time: float = 1.3  # length of the biggest push (s)
    boost_push: float = 24.0  # extra forward acceleration at the start of the biggest push (m/s²)


DEFAULT_DRIFT = DriftParams()


@dataclass(slots=True)
class DriftState:
    """Mutable sweep state for one kart. ``hold`` is the input: set it before each step."""

    hold: bool = False  # the drift button is down
    params: DriftParams = DEFAULT_DRIFT
    active: bool = False
    direction: float = 0.0  # +1 sweeps to the left, -1 to the right; fixed at entry
    time: float = 0.0  # time swept so far (s)
    armed: bool = True  # may start a sweep (released since the last forced end)
    push_left: float = 0.0  # time left of the push after a release (s)
    push_total: float = 0.0  # length the current push started with (s)
    push_speed: float = 0.0  # extra top speed the current push started with (m/s)

    @property
    def charge(self) -> float:
        """Push the sweep would give if released now, in [0, 1] (0 when not sweeping)."""
        return charge_for(self.time, self.params) if self.active else 0.0

    @property
    def push_level(self) -> float:
        """Remaining push in [0, 1] (1 right after a release, fading to 0)."""
        return self.push_left / self.push_total if self.push_left > 0.0 else 0.0

    @property
    def pushing(self) -> bool:
        return self.push_left > 0.0

    def reset(self) -> None:
        """Back to the initial state (``hold`` is an input and is kept)."""
        self.active = False
        self.direction = 0.0
        self.time = 0.0
        self.armed = True
        self.push_left = 0.0
        self.push_total = 0.0
        self.push_speed = 0.0


def charge_for(time_swept: float, params: DriftParams = DEFAULT_DRIFT) -> float:
    """Fraction of the maximum push earned by sweeping for ``time_swept`` seconds."""
    return min(max(time_swept, 0.0) / params.full_time, 1.0)


def push_for(time_swept: float, params: DriftParams = DEFAULT_DRIFT) -> tuple[float, float]:
    """(extra top speed in m/s, duration in s) of the push after a sweep of that length."""
    charge = charge_for(time_swept, params)
    return charge * params.boost_speed, charge * params.boost_time


def remap_steer(steer: float, direction: float, params: DriftParams = DEFAULT_DRIFT) -> float:
    """Wheel command while sweeping: always in ``direction``, between 0.3 and 1.0 of lock."""
    into = direction * steer  # > 0 steering into the sweep, < 0 against it
    return direction * (params.steer_neutral + params.steer_span * min(max(into, -1.0), 1.0))


@lru_cache(maxsize=8)
def _sweep_params(params: KartParams, drift: DriftParams) -> KartParams:
    return replace(
        params,
        grip=drift.grip,
        max_curvature=params.max_curvature * drift.turn_gain,
        rolling_drag=params.rolling_drag + drift.scrub,
    )


def kart_params(params: KartParams, drift: DriftState) -> KartParams:
    """The physics parameters in force this step (``params`` itself when not sweeping)."""
    effective = _sweep_params(params, drift.params) if drift.active else params
    if drift.push_left > 0.0:
        extra = drift.push_speed * drift.push_level
        effective = replace(effective, max_speed=effective.max_speed + extra)
    return effective


def _update_mode(
    state: KartState, steer: float, on_grass: bool, params: KartParams, drift: DriftState
) -> None:
    """Start or end the sweep and award the push, before the physics step."""
    cfg = drift.params
    speed = state.forward_speed
    if not drift.hold:
        drift.armed = True
        if drift.active:  # released: the push is proportional to the time swept
            extra, length = push_for(drift.time, cfg)
            drift.active = False
            if length > 0.0:
                drift.push_speed, drift.push_total, drift.push_left = extra, length, length
            drift.time = 0.0
            drift.direction = 0.0
    elif drift.active:
        if on_grass or speed < cfg.exit_speed:  # lost the sweep: no push, press again
            drift.active = False
            drift.armed = False
            drift.time = 0.0
            drift.direction = 0.0
    elif drift.armed and speed > cfg.entry_speed and abs(steer) > cfg.entry_steer and not on_grass:
        drift.active = True
        drift.direction = math.copysign(1.0, steer)
        drift.time = 0.0
    if on_grass:
        drift.push_left = 0.0  # the push only works on asphalt


def step_drift(
    state: KartState,
    steer: float,
    throttle: float,
    on_grass: bool,
    params: KartParams,
    dt: float,
    drift: DriftState,
) -> None:
    """One physics step with the sweep logic: ``step_kart`` with ``drift`` attached."""
    steer = min(max(steer, -1.0), 1.0)
    _update_mode(state, steer, on_grass, params, drift)
    if not drift.active and drift.push_left <= 0.0:
        step_kart(state, steer, throttle, on_grass, params, dt)  # plain physics
        return
    effective = kart_params(params, drift)
    wheel = remap_steer(steer, drift.direction, drift.params) if drift.active else steer
    step_kart(state, wheel, throttle, on_grass, effective, dt)
    if drift.active:
        drift.time += dt
    if drift.push_left > 0.0:
        level = drift.push_level
        cap = effective.max_speed
        if state.forward_speed < cap:  # extra shove that does not depend on the throttle
            shove = drift.params.boost_push * level * dt
            state.vx += math.cos(state.heading) * shove
            state.vy += math.sin(state.heading) * shove
        drift.push_left = max(drift.push_left - dt, 0.0)


__all__ = [
    "DEFAULT_DRIFT",
    "DriftParams",
    "DriftState",
    "charge_for",
    "kart_params",
    "push_for",
    "remap_steer",
    "step_drift",
]
