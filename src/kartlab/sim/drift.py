"""Manual sweep ("barrido") and hop ("brinco"): a controlled slide with a push on the way out.

Pressing the drift button makes the kart hop. Landing with the button still down and the
wheel turned starts a sweep: the tyres give much less side grip, so the kart slides and its
nose turns faster than its path. The longer the sweep, the bigger the push when the button
is released: a continuous amount of the time swept (after a short dead zone), up to a cap.
There are no levels, no countdown.

In the air the kart has no throttle and no grip: its velocity is kept (only the nose
turns a little), grass still brakes it, and it lands where it was going. The hop is not a
way to skip anything. With ``hop_time = 0`` there is no hop and the sweep starts the moment
the button is held with the wheel turned.

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
    # Push on release: grows linearly from ``charge_delay`` to ``full_time`` of sweeping.
    charge_delay: float = 0.2  # a short touch gives nothing (s)
    full_time: float = 2.4  # time swept that gives the biggest push (s)
    boost_speed: float = 6.0  # extra top speed at the biggest push (m/s)
    boost_time: float = 1.2  # length of the biggest push (s)
    boost_push: float = 24.0  # extra forward acceleration at the start of the biggest push (m/s²)
    # The hop (0 = no hop: the sweep starts as soon as the button and the wheel say so).
    hop_time: float = 0.35  # time in the air (s)
    hop_height: float = 0.5  # height at the top of the arc (m)
    hop_yaw: float = 1.8  # nose turn in the air per unit of wheel (rad/s)
    hop_yaw_max: float = 0.35  # most the nose turns in one hop (rad)
    land_grace: float = 0.25  # after landing, the time to turn the wheel and start a sweep (s)


DEFAULT_DRIFT = DriftParams()


NEVER = 1.0e9  # "a long time ago", for ``DriftState.landed_at``


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
    push_id: int = 0  # counts the pushes awarded (lets the renderer flash once per push)
    # The hop.
    height: float = 0.0  # height of the kart over the ground (m); 0 on the ground
    vz: float = 0.0  # vertical speed (m/s)
    hop_left: float = 0.0  # time left in the air (s)
    hop_turned: float = 0.0  # nose turn accumulated in this hop (rad)
    landed_at: float = NEVER  # time since the last landing (s)
    land_id: int = 0  # counts the landings (lets the renderer puff dust once per landing)
    was_hold: bool = False  # the button in the previous step (the hop starts on the press)

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

    @property
    def airborne(self) -> bool:
        return self.hop_left > 0.0

    @property
    def lift(self) -> float:
        """Height of the kart over the ground (m), for the renderers."""
        return self.height

    def reset(self) -> None:
        """Back to the initial state (``hold`` is an input and is kept)."""
        self.height = 0.0
        self.vz = 0.0
        self.hop_left = 0.0
        self.hop_turned = 0.0
        self.landed_at = NEVER
        self.was_hold = self.hold  # a button already down is not a new press
        self.active = False
        self.direction = 0.0
        self.time = 0.0
        self.armed = not self.hold  # a sweep is born from a hop: a held button waits for release
        self.push_left = 0.0
        self.push_total = 0.0
        self.push_speed = 0.0


def charge_for(time_swept: float, params: DriftParams = DEFAULT_DRIFT) -> float:
    """Fraction of the maximum push earned by sweeping for ``time_swept`` seconds."""
    span = max(params.full_time - params.charge_delay, 1e-9)
    return min(max((time_swept - params.charge_delay) / span, 0.0), 1.0)


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
                drift.push_id += 1
            drift.time = 0.0
            drift.direction = 0.0
    elif drift.active:
        if on_grass or speed < cfg.exit_speed:  # lost the sweep: no push, press again
            drift.active = False
            drift.armed = False
            drift.time = 0.0
            drift.direction = 0.0
    elif (
        (drift.armed or drift.landed_at < cfg.land_grace)  # or just landed from a hop
        and speed > cfg.entry_speed
        and abs(steer) > cfg.entry_steer
        and not on_grass
    ):
        drift.active = True
        drift.direction = math.copysign(1.0, steer)
        drift.time = 0.0
        drift.landed_at = NEVER
    if on_grass:
        drift.push_left = 0.0  # the push only works on asphalt


def _steer_wheel(state: KartState, steer: float, params: KartParams, dt: float) -> None:
    """The steering wheel follows the command at its limited rate (as in ``step_kart``)."""
    towards_centre = abs(steer) < abs(state.steer) or steer * state.steer < 0.0
    rate = params.steer_return_rate if towards_centre else params.steer_rate
    state.steer += min(max(steer - state.steer, -rate * dt), rate * dt)


def _start_hop(drift: DriftState) -> None:
    drift.hop_left = drift.params.hop_time
    drift.hop_turned = 0.0
    drift.armed = False  # a new press is needed after this hop, unless it lands in a sweep


def _air_step(
    state: KartState,
    steer: float,
    on_grass: bool,
    params: KartParams,
    dt: float,
    drift: DriftState,
) -> None:
    """One step in the air: no throttle, no grip, the velocity is kept and only the nose
    turns. Only the constant part of the rolling resistance (``rolling_drag``, not the term
    proportional to speed) acts, which is what keeps ~99 % of the speed; grass brakes exactly
    as on the ground."""
    cfg = drift.params
    _steer_wheel(state, steer, params, dt)
    turned = min(
        max(drift.hop_turned + cfg.hop_yaw * steer * dt, -cfg.hop_yaw_max), cfg.hop_yaw_max
    )
    state.heading = math.remainder(state.heading + turned - drift.hop_turned, math.tau)
    drift.hop_turned = turned

    if on_grass:
        drift.push_left = 0.0  # the push only works on asphalt
    top = params.grass_max_speed if on_grass else kart_params(params, drift).max_speed
    speed = math.hypot(state.vx, state.vy)
    if speed > 0.0:
        drag = params.grass_drag if on_grass else params.rolling_drag
        if speed > top:
            drag += params.overspeed_drag
        keep = max(speed - drag * dt, 0.0) / speed
        state.vx *= keep
        state.vy *= keep
    state.x += state.vx * dt
    state.y += state.vy * dt

    # Parabola with its top at half the time: g = 8h/T², launch speed g·T/2.
    gravity = 8.0 * cfg.hop_height / cfg.hop_time**2
    drift.hop_left = max(drift.hop_left - dt, 0.0)
    t = cfg.hop_time - drift.hop_left
    drift.height = max(gravity * t * (cfg.hop_time - t) / 2.0, 0.0)
    drift.vz = gravity * (cfg.hop_time / 2.0 - t)
    if drift.push_left > 0.0:
        drift.push_left = max(drift.push_left - dt, 0.0)
    if drift.hop_left < 1e-9:  # landing
        drift.hop_left = 0.0
        drift.height = 0.0
        drift.vz = 0.0
        drift.landed_at = 0.0
        drift.land_id += 1
        if not drift.hold:
            drift.armed = True


def step_drift(
    state: KartState,
    steer: float,
    throttle: float,
    on_grass: bool,
    params: KartParams,
    dt: float,
    drift: DriftState,
) -> None:
    """One physics step with the sweep and hop logic: ``step_kart`` with ``drift`` attached."""
    steer = min(max(steer, -1.0), 1.0)
    pressed = drift.hold and not drift.was_hold
    drift.was_hold = drift.hold
    if (
        pressed
        and drift.armed
        and not drift.active
        and drift.hop_left <= 0.0
        and drift.params.hop_time > 0.0
    ):
        _start_hop(drift)
    if drift.hop_left > 0.0:
        _air_step(state, steer, on_grass, params, dt, drift)
        return
    _update_mode(state, steer, on_grass, params, drift)
    drift.landed_at += dt
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
    "NEVER",
    "charge_for",
    "kart_params",
    "push_for",
    "remap_steer",
    "step_drift",
]
