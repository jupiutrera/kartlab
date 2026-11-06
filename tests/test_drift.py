"""The manual sweep ("barrido"): the plain physics must stay bit-for-bit the same."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import replace

import numpy as np
import pytest

from kartlab.sim.drift import (
    DEFAULT_DRIFT,
    DriftParams,
    DriftState,
    _sweep_params,
    charge_for,
    kart_params,
    push_for,
    remap_steer,
    step_drift,
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


BEFORE_THE_SLOW_TURBO = DriftParams(  # the turbo as it was: charges at once, fills in 1.5 s
    charge_delay=0.0, full_time=1.5, boost_time=1.3
)


def sweeping(speed: float = 20.0, steer: float = 1.0) -> tuple[DriftState, KartState]:
    """A kart with the button down and the wheel turned that is already sweeping."""
    drift, state = hold_state(), moving(speed)
    step_kart(state, steer, 1.0, False, P, drift=drift)
    assert drift.active
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
    step_kart(state, steer, 1.0, grass, P, drift=drift)
    assert not drift.active and drift.time == 0.0 and drift.direction == 0.0


def test_entry_fixes_the_direction_and_starts_charging():
    for steer, direction in ((1.0, 1.0), (-0.6, -1.0)):
        drift, state = hold_state(), moving(15.0)
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


def test_holding_the_button_from_the_start_works_but_ends_arm_until_release():
    drift, state = hold_state(), moving(20.0)
    step_kart(state, 0.0, 1.0, False, P, drift=drift)  # straight: no entry yet
    assert not drift.active
    step_kart(state, 1.0, 1.0, False, P, drift=drift)  # steering with the button down
    assert drift.active
    step_kart(state, 1.0, 1.0, True, P, drift=drift)  # grass ends it
    assert not drift.active
    step_kart(state, 1.0, 1.0, False, P, drift=drift)  # still held: no instant re-entry
    assert not drift.active
    drift.hold = False
    step_kart(state, 1.0, 1.0, False, P, drift=drift)
    drift.hold = True
    step_kart(state, 1.0, 1.0, False, P, drift=drift)
    assert drift.active


def test_the_sweep_ended_on_grass_waits_for_a_new_press():
    drift, state = sweeping()
    step_kart(state, 1.0, 1.0, True, P, drift=drift)  # grass ends it
    assert not drift.active and not drift.armed
    for _ in range(40):  # still held: no instant re-entry
        step_kart(state, 1.0, 1.0, False, P, drift=drift)
        assert not drift.active


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
    """Sweep for ``seconds`` (the time swept) and release."""
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


def test_reset_with_the_button_held_waits_for_its_release():
    # R, Tab or a new track with Shift down: no sweep until the button is released and pressed.
    drift, state = hold_state(), moving(22.0)
    drift.reset()
    assert drift.hold and not drift.armed
    for _ in range(30):
        step_kart(state, 1.0, 1.0, False, P, drift=drift)
        assert not drift.active
    drift.hold = False
    step_kart(state, 1.0, 1.0, False, P, drift=drift)
    assert drift.armed
    drift.hold = True
    step_kart(state, 1.0, 1.0, False, P, drift=drift)
    assert drift.active
    fresh = DriftState()
    fresh.reset()
    assert fresh.armed  # with the button up, a reset arms the sweep


def test_the_sweep_is_step_kart_with_the_sweep_parameters():
    """The sweep starts on the very step the button and the wheel say so, and the kart moves
    exactly as ``step_kart`` does with the sweep parameters."""
    cfg = replace(DEFAULT_DRIFT, charge_delay=0.0, full_time=1.5)
    drift, state = DriftState(hold=True, params=cfg), moving(20.0)
    reference = moving(20.0)
    sweep = _sweep_params(P, cfg)
    for i in range(90):
        steer = 1.0 if i < 60 else -0.5
        step_kart(state, steer, 1.0, False, P, drift=drift)
        step_kart(reference, remap_steer(steer, 1.0, cfg), 1.0, False, sweep, DT)
        assert state == reference
    assert drift.active and drift.time == pytest.approx(90 * DT)


# The trace of random sweeps below was recorded with the code that still had the jump before
# the button (PR #15) with its time set to 0; the code without the jump must give the same.
# The end of 8 random traces of 2500 steps each (the button toggling, the wheel, the throttle
# and the grass changing, resets with and without the button down), recorded with the jump of
# PR #15 switched off (``hop_time = 0``) before it was removed: x, y, heading, vx, vy, steps
# sweeping, steps under the push and pushes awarded. The positions are compared with a
# tolerance (like the physics fingerprint): the libm of Linux and Windows round sin and cos
# differently in the last bit, so a bit-exact hash of every step would only hold on one OS.
SWEEP_TRACE_ENDS = (
    (301.461907, -174.566712, 0.649168, 4.796473, 3.417173, 523, 77, 19),
    (-239.238089, 139.648665, -2.397331, -17.227852, -15.825903, 714, 114, 26),
    (199.943193, 185.140184, 0.427515, 17.769537, 6.373735, 312, 15, 6),
    (254.196932, 180.313856, -0.07306, 2.226932, -0.17533, 445, 66, 12),
    (174.286324, 191.109042, 1.434483, 1.37079, 15.878162, 330, 42, 9),
    (-51.213551, 348.765685, 1.60652, -3.480576, 22.464935, 621, 111, 18),
    (244.554072, 112.153381, -0.554846, 14.501364, -8.342247, 351, 39, 10),
    (168.642405, 323.33628, 1.795123, -4.048226, 14.825571, 592, 89, 21),
)


def sweep_trace_ends() -> list[tuple]:
    """For each of the 8 traces: the kart at the end, the steps sweeping and under the push,
    and the pushes awarded."""
    ends = []
    for seed in range(8):
        rng = random.Random(seed)
        drift = DriftState()
        state = KartState(0.0, 0.0, 0.3, vx=rng.uniform(0, 25))
        steer = throttle = 0.0
        grass = False
        swept = pushing = 0
        for i in range(2500):
            if i % 7 == 0:
                drift.hold = rng.random() < 0.6 if i % 70 else not drift.hold
            if i % 11 == 0:
                steer = rng.choice([-1.5, -1.0, -0.6, -0.2, 0.0, 0.2, 0.6, 1.0, 1.5])
            if i % 23 == 0:
                throttle = rng.choice([-1.0, 0.0, 0.5, 1.0, 1.0, 1.0])
            if i % 150 == 0:
                grass = not grass if rng.random() < 0.4 else False
            if i % 600 == 599:
                drift.hold = rng.random() < 0.5
                drift.reset()
            step_drift(state, steer, throttle, grass, P, DT, drift)
            swept += drift.active
            pushing += drift.push_left > 0.0
        ends.append(
            (state.x, state.y, state.heading, state.vx, state.vy, swept, pushing, drift.push_id)
        )
    return ends


def test_the_sweep_is_exactly_what_it_was_with_the_jump_switched_off():
    ends = sweep_trace_ends()
    assert sum(e[5] for e in ends) > 3000 and sum(e[7] for e in ends) > 20  # it does sweep
    for got, pinned in zip(ends, SWEEP_TRACE_ENDS, strict=True):
        assert got[5:] == pinned[5:]  # the same sweeps and pushes, step for step in count
        assert got[:5] == pytest.approx(pinned[:5], rel=1e-6, abs=1e-6)


def fraction_under_push(cfg: DriftParams, sweep: float, pause: float, seconds: float = 120.0):
    """A simulated player that chains sweeps of ``sweep`` seconds with a pause between."""
    state, drift = moving(22.0), DriftState(params=cfg)
    steps = round(seconds / DT)
    pushing = 0
    for i in range(steps):
        drift.hold = (i * DT) % (sweep + pause) < sweep
        step_kart(state, 1.0, 1.0, False, P, drift=drift)
        pushing += drift.pushing
    return pushing / steps


def test_chained_sweeps_spend_far_less_time_under_the_push():
    before = fraction_under_push(BEFORE_THE_SLOW_TURBO, 1.5, 0.3)
    now = fraction_under_push(DEFAULT_DRIFT, 1.5, 0.3)
    assert before > 0.6  # the complaint: the turbo was on most of the time
    assert now <= 0.42 and now <= before * 0.6


# -- the manual drive mode, headless ------------------------------------------------------


def _hold(monkeypatch, *pressed: int):
    """Make ``pygame.key.get_pressed`` report exactly these keys as down."""
    import pygame

    keys = defaultdict(bool, dict.fromkeys(pressed, True))
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: keys)


@pytest.mark.parametrize("view", ["3d", "topdown"])
def test_drive_mode_draws_a_sweep_through_a_corner(monkeypatch, view):
    import pygame

    from kartlab.render.sparks import EMISSIVE
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
        for _ in range(60):  # the sweep starts at once and builds up its charge
            app.run(max_frames=1)
            if app.game is not None:
                sparks += sum(1 for p in app.game.world.particles.items if p[7] >= EMISSIVE)
        assert app.drift.active and app.drift.charge > 0.15
        if app.game is not None:
            assert sparks > 0
        before = pygame.surfarray.array3d(app.screen).copy()
        _hold(monkeypatch, pygame.K_UP)  # release
        flashes = 0
        for _ in range(30):
            app.run(max_frames=1)
            if app.game is not None:
                flashes += sum(1 for p in app.game.world.particles.items if p[7] >= EMISSIVE)
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
