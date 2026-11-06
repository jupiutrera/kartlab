"""The manual sweep ("barrido") and hop ("brinco"): the plain physics must stay bit-for-bit
the same."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace

import numpy as np
import pytest

from kartlab.sim.autopilot import Autopilot
from kartlab.sim.drift import (
    DEFAULT_DRIFT,
    DriftParams,
    DriftState,
    _sweep_params,
    charge_for,
    kart_params,
    push_for,
    remap_steer,
)
from kartlab.sim.kart import (
    DT,
    STOP_SPEED,
    KartParams,
    KartState,
    _clamp,
    max_path_curvature,
    step_kart,
)
from kartlab.sim.race import Racer
from kartlab.sim.showcase import showcase_track

P = KartParams()


def reference_step_kart(state, steer, throttle, on_grass, params, dt=DT) -> None:
    """Verbatim copy of ``step_kart`` as it was before the sweep existed."""
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


def varied_inputs(steps: int = 600):
    """Steer, throttle and grass flag per step: a fixed script of turns, braking, reverse
    and grass on and off, then random changes."""
    script = [  # (steps, steer, throttle, on_grass)
        (80, 0.0, 1.0, False),  # accelerate
        (60, 1.0, 1.0, False),  # full lock left
        (50, -0.4, 0.6, False),
        (40, 0.0, -1.0, False),  # brake hard to a stop
        (90, -0.5, -1.0, False),  # then reverse
        (30, 1.5, 0.0, False),  # coast, steer beyond range
        (70, 0.3, 1.0, True),  # accelerate on grass
        (50, -1.0, 1.0, False),  # back on the asphalt
    ]
    out = [(steer, throttle, grass) for n, steer, throttle, grass in script for _ in range(n)]
    rng = np.random.default_rng(7)
    steer = throttle = 0.0
    grass = False
    for i in range(steps - len(out)):
        if i % 25 == 0:
            steer = float(rng.choice([-1.5, -1.0, -0.4, 0.0, 0.4, 1.0, 1.5]))
        if i % 40 == 0:
            throttle = float(rng.choice([-1.0, -0.3, 0.0, 0.5, 1.0, 1.0, 1.0]))
        if i % 90 == 0:
            grass = not grass
        out.append((steer, throttle, grass))
    return out[:steps]


def run(step, inputs, **kwargs) -> list[KartState]:
    state = KartState(0.0, 0.0, 0.3)
    history = []
    for steer, throttle, grass in inputs:
        step(state, steer, throttle, grass, P, **kwargs)
        history.append(replace(state))
    return history


def test_inputs_cover_the_cases_that_matter():
    inputs = varied_inputs()
    assert any(g for _, _, g in inputs) and any(not g for _, _, g in inputs)
    assert any(t < 0 for _, t, _ in inputs) and any(abs(s) > 1 for s, _, _ in inputs)
    history = run(reference_step_kart, inputs)
    assert min(s.forward_speed for s in history) < -1.0  # it does drive in reverse


def test_drift_none_is_exactly_the_old_physics():
    inputs = varied_inputs()
    expected = run(reference_step_kart, inputs)
    assert run(step_kart, inputs) == expected  # == on floats, no approx
    assert run(step_kart, inputs, drift=None) == expected


def test_racer_step_without_drift_is_unchanged():
    track = showcase_track("harbor")
    a, b = Racer(track), Racer(track)
    for steer, throttle, _ in varied_inputs():
        gain_a = a.step(steer, throttle)
        gain_b = b.step(steer, throttle, drift=None)
        assert gain_a == gain_b
    assert a.kart == b.kart and a.position == b.position


def moving(speed: float) -> KartState:
    return KartState(0.0, 0.0, 0.0, vx=speed)


def hold_state() -> DriftState:
    return DriftState(hold=True)


NO_HOP = replace(DEFAULT_DRIFT, hop_time=0.0)  # the sweep as it was before the hop existed
BEFORE_THE_HOP = DriftParams(  # the turbo as it was: charges at once, fills in 1.5 s
    charge_delay=0.0, full_time=1.5, boost_time=1.3, hop_time=0.0
)


def hop_and_land(
    drift: DriftState,
    state: KartState,
    steer: float = 1.0,
    throttle: float = 1.0,
    grass: bool = False,
) -> int:
    """Step until the hop that the first step starts is over (the landing step included)."""
    steps = 0
    while True:
        step_kart(state, steer, throttle, grass, P, drift=drift)
        steps += 1
        if not drift.airborne:
            return steps
        assert steps < 100


def sweeping(speed: float = 20.0, steer: float = 1.0) -> tuple[DriftState, KartState]:
    """A kart that pressed the button, hopped, landed with the wheel turned and is sweeping."""
    drift, state = hold_state(), moving(speed)
    hop_and_land(drift, state, steer)
    step_kart(state, steer, 1.0, False, P, drift=drift)
    return drift, state


@pytest.mark.parametrize(
    "speed, steer, grass",
    [
        (9.9, 1.0, False),  # too slow
        (0.0, 1.0, False),
        (20.0, 0.35, False),  # not enough steering (the limit itself does not count)
        (20.0, -0.2, False),
        (20.0, 0.0, False),
        (20.0, 1.0, True),  # on grass
    ],
)
def test_no_entry_without_speed_steering_or_asphalt(speed, steer, grass):
    drift, state = hold_state(), moving(speed)
    hop_and_land(drift, state, steer, throttle=0.0, grass=grass)  # no throttle: speed stays low
    for _ in range(30):  # the whole landing window and beyond
        step_kart(state, steer, 0.0, grass, P, drift=drift)
        assert not drift.active and drift.time == 0.0 and drift.direction == 0.0


def test_entry_fixes_the_direction_and_starts_charging():
    for steer, direction in ((1.0, 1.0), (-0.6, -1.0)):
        drift, state = hold_state(), moving(15.0)
        hop_and_land(drift, state, steer)
        assert not drift.active  # the landing step itself is still the hop
        step_kart(state, steer, 1.0, False, P, drift=drift)
        assert drift.active and drift.direction == direction
        assert drift.time == pytest.approx(DT)
        step_kart(state, -steer, 1.0, False, P, drift=drift)  # counter-steering
        assert drift.direction == direction  # does not flip


def test_without_the_button_nothing_happens():
    drift, state = DriftState(), moving(20.0)
    for _ in range(30):
        step_kart(state, 1.0, 1.0, False, P, drift=drift)
    assert not drift.active and drift.time == 0.0


def test_a_press_without_the_wheel_gives_no_sweep_until_the_button_is_released():
    drift, state = hold_state(), moving(20.0)
    hop_and_land(drift, state, 0.0)  # straight hop
    for _ in range(round(DEFAULT_DRIFT.land_grace / DT) + 5):  # the landing window passes
        step_kart(state, 0.0, 1.0, False, P, drift=drift)
    assert not drift.active and not drift.armed
    step_kart(state, 1.0, 1.0, False, P, drift=drift)  # steering late, button still down
    assert not drift.active
    drift.hold = False
    step_kart(state, 1.0, 1.0, False, P, drift=drift)  # released: armed again
    assert drift.armed
    drift.hold = True
    hop_and_land(drift, state, 1.0)  # a new press hops again and, with the wheel, sweeps
    step_kart(state, 1.0, 1.0, False, P, drift=drift)
    assert drift.active


def test_the_sweep_ended_on_grass_waits_for_a_new_press():
    drift, state = sweeping()
    step_kart(state, 1.0, 1.0, True, P, drift=drift)  # grass ends it
    assert not drift.active and not drift.armed
    for _ in range(40):  # still held: no instant re-entry, no new hop
        step_kart(state, 1.0, 1.0, False, P, drift=drift)
        assert not drift.active and not drift.airborne


def test_the_sweep_ends_on_grass_or_when_too_slow_and_gives_no_push():
    for kind in ("grass", "slow"):
        drift, state = hold_state(), moving(20.0)
        for _ in range(60):
            step_kart(state, 1.0, 1.0, False, P, drift=drift)
        assert drift.active
        if kind == "grass":
            step_kart(state, 1.0, 1.0, True, P, drift=drift)
        else:
            state.vx = state.vy = 0.5 * math.cos(state.heading)
            step_kart(state, 1.0, 0.0, False, P, drift=drift)
        assert not drift.active and not drift.pushing


def test_sweeping_lowers_grip_and_slides_more():
    plain, swept = moving(20.0), moving(20.0)
    drift = hold_state()
    for _ in range(40):
        step_kart(plain, 1.0, 1.0, False, P)
        step_kart(swept, 1.0, 1.0, False, P, drift=drift)
    assert drift.active
    assert kart_params(P, drift).grip == DEFAULT_DRIFT.grip < P.grip
    assert abs(swept.lateral_speed) > 3 * abs(plain.lateral_speed)
    assert swept.heading > plain.heading  # the nose turns more than in normal driving


def test_remapped_steering_tightens_into_the_sweep_and_widens_against_it():
    into = remap_steer(1.0, 1.0)
    neutral = remap_steer(0.0, 1.0)
    against = remap_steer(-1.0, 1.0)
    assert 0.0 < against < neutral < into <= 1.0
    assert remap_steer(0.0, -1.0) == -neutral and remap_steer(1.0, -1.0) == -against


def release_after(seconds: float, speed: float = 22.0) -> tuple[DriftState, KartState]:
    """Hop, sweep for ``seconds`` (the time swept, the hop not included) and release."""
    drift, state = sweeping(speed)
    for _ in range(round(seconds / DT) - 1):
        step_kart(state, 1.0, 1.0, False, P, drift=drift)
    assert drift.time == pytest.approx(seconds)
    drift.hold = False
    step_kart(state, 0.0, 1.0, False, P, drift=drift)
    return drift, state


def test_push_is_monotonic_in_the_time_swept_and_capped():
    times = np.linspace(0.0, 4.0, 41)
    pushes = [push_for(float(t)) for t in times]
    speeds, lengths = zip(*pushes, strict=True)
    assert all(a <= b for a, b in zip(speeds, speeds[1:], strict=False))
    assert all(a <= b for a, b in zip(lengths, lengths[1:], strict=False))
    assert speeds[0] == lengths[0] == 0.0
    assert max(speeds) == DEFAULT_DRIFT.boost_speed and max(lengths) == DEFAULT_DRIFT.boost_time
    assert speeds[-1] == speeds[-5]  # saturated
    delay, full = DEFAULT_DRIFT.charge_delay, DEFAULT_DRIFT.full_time
    assert charge_for(delay + (full - delay) / 2) == pytest.approx(0.5)  # linear after the delay
    assert charge_for(delay + 0.5) * 2 == pytest.approx(charge_for(delay + 1.0))


def test_the_charge_has_a_short_dead_zone_and_a_continuous_ramp():
    # (time swept, charge): the table of the design, with the new slower turbo.
    for seconds, charge in ((0.0, 0.0), (0.2, 0.0), (1.0, 0.36), (1.5, 0.59), (2.0, 0.82)):
        assert charge_for(seconds) == pytest.approx(charge, abs=0.01)
    assert charge_for(2.4) == 1.0 and charge_for(5.0) == 1.0
    steps = [charge_for(t / 60) for t in range(0, 200)]
    assert all(b - a < 0.01 for a, b in zip(steps, steps[1:], strict=False))  # no jumps
    assert push_for(0.15) == (0.0, 0.0)  # a touch gives no push at all


def test_released_sweeps_gain_more_speed_the_longer_they_lasted():
    peaks = []
    for seconds in (0.6, 1.0, 1.5, 2.0, 2.4, 3.5):
        drift, state = release_after(seconds)
        assert drift.pushing
        peak = state.forward_speed
        for _ in range(120):
            step_kart(state, 0.0, 1.0, False, P, drift=drift)
            peak = max(peak, state.forward_speed)
        peaks.append(peak)
    assert all(a < b for a, b in zip(peaks[:5], peaks[1:5], strict=False))
    assert peaks[5] == pytest.approx(peaks[4], abs=0.5)  # capped from 2.4 s on
    assert peaks[5] <= P.max_speed + DEFAULT_DRIFT.boost_speed + 1.0


def test_the_push_fades_out_and_the_speed_limit_comes_back():
    drift, state = release_after(2.0)
    total = drift.push_total
    for _ in range(round(total / DT) + 5):
        step_kart(state, 0.0, 1.0, False, P, drift=drift)
    assert not drift.pushing and drift.push_level == 0.0
    for _ in range(120):
        step_kart(state, 0.0, 1.0, False, P, drift=drift)
    assert state.forward_speed <= P.max_speed


def test_after_the_sweep_the_next_steps_equal_the_plain_physics():
    # Ended by leaving the asphalt: no push, so the very next step is the plain one.
    drift, state = hold_state(), moving(20.0)
    for _ in range(50):
        step_kart(state, 1.0, 1.0, False, P, drift=drift)
    step_kart(state, 1.0, 1.0, True, P, drift=drift)
    assert not drift.active and not drift.pushing
    assert kart_params(P, drift) is P
    plain = replace(state)
    for steer, throttle, grass in varied_inputs(120):
        step_kart(state, steer, throttle, grass, P, drift=drift)
        reference_step_kart(plain, steer, throttle, grass, P)
        assert state == plain


def test_after_the_push_expires_the_physics_is_plain_again():
    drift, state = release_after(1.0)
    while drift.pushing:
        step_kart(state, 0.0, 1.0, False, P, drift=drift)
    plain = replace(state)
    drift.hold = False
    for steer, throttle, grass in varied_inputs(120):
        step_kart(state, steer, throttle, grass, P, drift=drift)
        reference_step_kart(plain, steer, throttle, grass, P)
        assert state == plain


def test_grip_during_the_push_is_the_normal_grip():
    drift, _ = release_after(1.0)
    assert drift.pushing and not drift.active
    assert kart_params(P, drift).grip == P.grip
    assert kart_params(P, drift).max_speed > P.max_speed


def test_leaving_the_asphalt_cancels_the_push():
    drift, state = release_after(1.0)
    step_kart(state, 0.0, 1.0, True, P, drift=drift)
    assert not drift.pushing


def test_reset_clears_the_state_but_keeps_the_input():
    drift = hold_state()
    drift.active, drift.time, drift.push_left = True, 1.0, 0.5
    drift.reset()
    assert drift.hold and not drift.active and drift.time == 0.0 and not drift.pushing
    assert drift.charge == 0.0


def test_a_racer_drives_a_corner_with_the_sweep():
    track = showcase_track("harbor")
    racer = Racer(track)
    drift = DriftState(hold=True)
    for i in range(900):
        steer = racer.heading_error()
        racer.step(float(np.clip(-3.0 * steer, -1, 1)), 1.0, drift=drift)
        if i == 400:
            drift.hold = False
    assert math.isfinite(racer.kart.x) and racer.progress.distance > 100.0


# -- the hop ("brinco") ---------------------------------------------------------------------


def hop_trace(speed: float = 22.0, steer: float = 0.0, throttle: float = 1.0, grass: bool = False):
    """Press the button on the ground and record (height, horizontal speed) until landing."""
    drift, state = hold_state(), moving(speed)
    heights, speeds = [], []
    while True:
        step_kart(state, steer, throttle, grass, P, drift=drift)
        heights.append(drift.height)
        speeds.append(math.hypot(state.vx, state.vy))
        if not drift.airborne:
            return drift, state, heights, speeds
        assert len(heights) < 100


def test_the_hop_lasts_hop_time_and_peaks_at_hop_height():
    _, _, heights, _ = hop_trace()
    cfg = DEFAULT_DRIFT
    in_the_air = sum(1 for h in heights if h > 0.0)
    assert abs(in_the_air * DT - cfg.hop_time) <= DT + 1e-9  # hop_time ± one step
    assert max(heights) == pytest.approx(cfg.hop_height, rel=0.02)
    peak = int(np.argmax(heights))
    assert abs(peak - len(heights) / 2) <= 1  # the top is halfway
    assert heights[-1] == 0.0 and all(h >= 0.0 for h in heights)


def test_in_the_air_the_speed_is_kept_and_the_throttle_does_nothing():
    _, full, _, speeds = hop_trace(22.0, 0.0, 1.0)
    _, none, _, coasting = hop_trace(22.0, 0.0, 0.0)
    _, braked, _, braking = hop_trace(22.0, 0.0, -1.0)
    assert min(speeds) >= 22.0 * 0.99  # kept to within 1 %
    assert speeds == coasting == braking  # neither throttle nor brake reaches the road
    assert full.vx == none.vx == braked.vx


def test_the_nose_turns_a_little_in_the_air_but_the_path_does_not():
    drift, state, _, _ = hop_trace(22.0, 1.0)
    assert 0.0 < state.heading <= DEFAULT_DRIFT.hop_yaw_max + 1e-9  # capped at 0.35 rad
    assert drift.hop_turned == pytest.approx(DEFAULT_DRIFT.hop_yaw_max)
    assert state.vy == 0.0 and state.vx > 0.0  # the velocity points where it did
    _, half, _, _ = hop_trace(22.0, 0.25)
    assert 0.0 < half.heading < state.heading  # less wheel, less turn
    _, right, _, _ = hop_trace(22.0, -1.0)
    assert right.heading == pytest.approx(-state.heading)


def test_the_hop_starts_only_on_a_press_on_the_ground_without_a_sweep():
    drift, state = DriftState(hold=True, was_hold=True), moving(22.0)  # held since before
    step_kart(state, 0.0, 1.0, False, P, drift=drift)
    assert not drift.airborne
    drift.hold = False
    step_kart(state, 0.0, 1.0, False, P, drift=drift)
    drift.hold = True
    step_kart(state, 0.0, 1.0, False, P, drift=drift)  # a press: up it goes
    assert drift.airborne and drift.height > 0.0
    drift.hold = False
    step_kart(state, 0.0, 1.0, False, P, drift=drift)
    drift.hold = True
    land_id = drift.land_id
    step_kart(state, 0.0, 1.0, False, P, drift=drift)  # pressing in the air does not re-hop
    left = drift.hop_left
    assert drift.airborne and left < DEFAULT_DRIFT.hop_time
    while drift.airborne:
        step_kart(state, 0.0, 1.0, False, P, drift=drift)
    assert drift.land_id == land_id + 1 and drift.height == 0.0
    sweep, moving_kart = sweeping()  # no hop while sweeping, even with a new "press"
    sweep.was_hold = False
    step_kart(moving_kart, 1.0, 1.0, False, P, drift=sweep)
    assert sweep.active and not sweep.airborne


def test_there_is_no_speed_threshold_for_the_hop():
    drift, state = hold_state(), moving(0.0)
    step_kart(state, 0.0, 0.0, False, P, drift=drift)
    assert drift.airborne
    while drift.airborne:
        step_kart(state, 0.0, 0.0, False, P, drift=drift)
    for _ in range(30):  # but at a standstill there is nothing to sweep with
        step_kart(state, 1.0, 0.0, False, P, drift=drift)
        assert not drift.active


def test_landing_with_the_wheel_turned_starts_a_sweep_that_slides():
    drift, state = hold_state(), moving(22.0)
    hop_and_land(drift, state, 1.0)
    assert abs(state.lateral_speed) > 1.0  # the nose turned, the path did not: it slides
    step_kart(state, 1.0, 1.0, False, P, drift=drift)
    assert drift.active and drift.direction == 1.0
    right, kart_right = hold_state(), moving(22.0)
    hop_and_land(right, kart_right, -1.0)
    step_kart(kart_right, -1.0, 1.0, False, P, drift=right)
    assert right.active and right.direction == -1.0


def test_landing_without_the_wheel_gives_no_sweep_until_released():
    drift, state = hold_state(), moving(22.0)
    hop_and_land(drift, state, 0.0)
    assert not drift.active
    for _ in range(round(DEFAULT_DRIFT.land_grace / DT) + 3):
        step_kart(state, 0.0, 1.0, False, P, drift=drift)
    assert not drift.armed
    for _ in range(30):
        step_kart(state, 1.0, 1.0, False, P, drift=drift)
        assert not drift.active
    drift.hold = False
    step_kart(state, 1.0, 1.0, False, P, drift=drift)
    assert drift.armed


def test_the_wheel_counts_only_inside_the_landing_grace():
    grace_steps = round(DEFAULT_DRIFT.land_grace / DT)
    for wait, expected in ((2, True), (grace_steps - 3, True), (grace_steps + 3, False)):
        drift, state = hold_state(), moving(22.0)
        hop_and_land(drift, state, 0.0)
        for _ in range(wait):
            step_kart(state, 0.0, 1.0, False, P, drift=drift)
        step_kart(state, 1.0, 1.0, False, P, drift=drift)
        assert drift.active is expected, wait


def test_the_wheel_during_the_hop_helps_when_it_lands_and_a_late_release_does_not_sweep():
    drift, state = hold_state(), moving(22.0)
    hop_and_land(drift, state, 1.0)
    drift.hold = False  # released in the very landing step: no sweep, nothing pending
    step_kart(state, 1.0, 1.0, False, P, drift=drift)
    assert not drift.active and drift.armed and not drift.pushing


def test_a_hop_over_grass_brakes_as_much_as_staying_on_it():
    # Over the grass limit the kart slows by 20 m/s²; the hop must not save any of it.
    hopping, plain = hold_state(), moving(20.0)
    still_on_the_ground = DriftState()
    held, staying = moving(20.0), moving(20.0)
    for _ in range(40):
        step_kart(held, 0.0, 1.0, True, P, drift=hopping)
        step_kart(staying, 0.0, 1.0, True, P, drift=still_on_the_ground)
        step_kart(plain, 0.0, 1.0, True, P)
    assert hopping.land_id == 1  # it did hop and land inside those steps
    assert held.forward_speed == pytest.approx(staying.forward_speed, rel=0.05)
    assert staying.forward_speed == plain.forward_speed
    assert held.forward_speed < P.grass_max_speed * 1.2  # and it did brake hard


def test_a_hop_does_not_keep_the_push_over_grass():
    drift, state = release_after(1.5)
    assert drift.pushing
    drift.hold = True
    drift.was_hold = False
    step_kart(state, 0.0, 1.0, False, P, drift=drift)  # hop while the push still runs
    assert drift.airborne
    step_kart(state, 0.0, 1.0, True, P, drift=drift)  # over the grass in the air
    assert not drift.pushing


def test_reset_puts_the_kart_back_on_the_ground():
    drift, state = hold_state(), moving(22.0)
    step_kart(state, 1.0, 1.0, False, P, drift=drift)
    assert drift.airborne
    drift.reset()
    assert drift.height == 0.0 and drift.vz == 0.0 and not drift.airborne and drift.lift == 0.0
    assert drift.hold and drift.was_hold  # a button already down is not a new press


def test_reset_with_the_button_held_gives_no_sweep_without_a_hop():
    # R, Tab or a new track with Shift down: the sweep still needs a hop, so a new press.
    drift, state = hold_state(), moving(22.0)
    drift.reset()
    assert drift.hold and not drift.armed
    for _ in range(30):
        step_kart(state, 1.0, 1.0, False, P, drift=drift)
        assert not drift.active and not drift.airborne
    drift.hold = False
    step_kart(state, 1.0, 1.0, False, P, drift=drift)
    assert drift.armed
    drift.hold = True
    hop_and_land(drift, state, 1.0)  # released and pressed again: hop, then sweep
    step_kart(state, 1.0, 1.0, False, P, drift=drift)
    assert drift.active


def test_hop_time_zero_is_the_sweep_as_it_was():
    """Without the hop the sweep starts on the very step the button and the wheel say so, and
    the kart moves exactly as ``step_kart`` does with the sweep parameters."""
    cfg = replace(NO_HOP, charge_delay=0.0, full_time=1.5)
    drift, state = DriftState(hold=True, params=cfg), moving(20.0)
    reference = moving(20.0)
    sweep = _sweep_params(P, cfg)
    for i in range(90):
        steer = 1.0 if i < 60 else -0.5
        step_kart(state, steer, 1.0, False, P, drift=drift)
        step_kart(reference, remap_steer(steer, 1.0, cfg), 1.0, False, sweep, DT)
        assert state == reference
        assert drift.height == 0.0 and not drift.airborne and drift.land_id == 0
    assert drift.active and drift.time == pytest.approx(90 * DT)


def fraction_under_push(cfg: DriftParams, sweep: float, pause: float, seconds: float = 120.0):
    """A simulated player that chains sweeps of ``sweep`` seconds (after the hop) with a pause."""
    state, drift = moving(22.0), DriftState(params=cfg)
    hold = sweep + cfg.hop_time
    steps = round(seconds / DT)
    pushing = 0
    for i in range(steps):
        drift.hold = (i * DT) % (hold + pause) < hold
        step_kart(state, 1.0, 1.0, False, P, drift=drift)
        pushing += drift.pushing
    return pushing / steps


def test_chained_sweeps_spend_far_less_time_under_the_push():
    before = fraction_under_push(BEFORE_THE_HOP, 1.5, 0.3)
    now = fraction_under_push(DEFAULT_DRIFT, 1.5, 0.3)
    assert before > 0.6  # the complaint: the turbo was on most of the time
    assert now <= 0.35 and now <= before / 2


@pytest.mark.parametrize("name", ["harbor", "canyon", "plateau"])
def test_the_hop_is_no_shortcut_for_the_autopilot(name):
    track = showcase_track(name)

    def lap_time(hop_every: int | None) -> float:
        racer, pilot, drift = Racer(track), Autopilot(track), DriftState()
        for i in range(60 * 300):
            steer, throttle = pilot.act(racer)
            # One press per second and nothing else: no sweep, only the hop.
            drift.hold = hop_every is not None and i % hop_every == 0
            racer.step(steer, throttle, drift=drift if hop_every else None)
            if racer.laps >= 1:
                return racer.time
        raise AssertionError("no lap in 300 s")

    assert lap_time(60) >= lap_time(None) * 0.99


# -- the manual drive mode, headless ------------------------------------------------------


def _hold(monkeypatch, *pressed: int):
    """Make ``pygame.key.get_pressed`` report exactly these keys as down."""
    import pygame

    keys = defaultdict(bool, dict.fromkeys(pressed, True))
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: keys)


@pytest.mark.parametrize("view", ["3d", "topdown"])
def test_drive_mode_draws_a_sweep_through_a_corner(monkeypatch, view):
    import pygame

    from kartlab.render.palette import Ink
    from kartlab.show.drive import DriveApp, drift_pressed

    app = DriveApp(showcase_track("harbor"), scale=1, view=view, realtime=False)
    try:
        _hold(monkeypatch, pygame.K_UP)
        assert not drift_pressed(pygame.key.get_pressed())
        app.run(max_frames=150)  # build up speed on the straight
        assert app.racer.speed > 12.0 and not app.drift.active
        _hold(monkeypatch, pygame.K_UP, pygame.K_LEFT, pygame.K_LSHIFT)
        assert drift_pressed(pygame.key.get_pressed())
        sparks = 0
        peak = 0.0
        for _ in range(60):  # the hop takes 21 frames, then the sweep builds up its charge
            app.run(max_frames=1)
            peak = max(peak, app.drift.height)
            if app.game is not None:
                sparks += sum(1 for p in app.game.world.particles.items if p[7] == Ink.YELLOW)
        assert peak == pytest.approx(DEFAULT_DRIFT.hop_height, rel=0.02)
        assert app.drift.land_id == 1 and app.drift.active and app.drift.charge > 0.15
        if app.game is not None:
            assert sparks > 0
        before = pygame.surfarray.array3d(app.screen).copy()
        _hold(monkeypatch, pygame.K_UP)  # release
        flashes = 0
        for _ in range(30):
            app.run(max_frames=1)
            if app.game is not None:
                flashes += sum(1 for p in app.game.world.particles.items if p[7] == Ink.ORANGE)
        assert not app.drift.active and app.drift.push_id == 1
        if app.game is not None:
            assert flashes > 0
        assert (pygame.surfarray.array3d(app.screen) != before).any()
    finally:
        pygame.quit()


def test_the_autopilot_never_sweeps(monkeypatch):
    import pygame

    from kartlab.show.drive import DriveApp

    app = DriveApp(showcase_track("harbor"), scale=1, autopilot=True, realtime=False)
    try:
        _hold(monkeypatch, pygame.K_LSHIFT, pygame.K_LEFT)
        app.run(max_frames=200)
        assert not app.drift.active and app.drift.push_id == 0
    finally:
        pygame.quit()
