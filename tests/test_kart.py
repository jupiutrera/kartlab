import math

import numpy as np
import pytest

from kartlab.sim.kart import (
    DT,
    KartParams,
    KartState,
    cruise_speed,
    max_path_curvature,
    step_kart,
)

P = KartParams()


def drive(state: KartState, steer: float, throttle: float, seconds: float, on_grass=False):
    speeds = []
    for _ in range(round(seconds / DT)):
        step_kart(state, steer, throttle, on_grass, P)
        speeds.append(state.forward_speed)
    return speeds


def moving(speed: float) -> KartState:
    return KartState(0.0, 0.0, 0.0, vx=speed)


def test_full_throttle_reaches_cruise_speed_and_never_exceeds_the_limit():
    speeds = drive(KartState(0.0, 0.0, 0.0), 0.0, 1.0, 10.0)
    assert max(speeds) <= P.max_speed
    assert speeds[-1] == pytest.approx(cruise_speed(P), rel=0.01)
    assert speeds[round(4.0 / DT)] > 0.9 * cruise_speed(P)


def test_partial_throttle_settles_at_a_lower_speed():
    full = drive(KartState(0.0, 0.0, 0.0), 0.0, 1.0, 15.0)[-1]
    half = drive(KartState(0.0, 0.0, 0.0), 0.0, 0.2, 15.0)[-1]
    assert 0.0 < half < full


def test_coasting_slows_to_a_full_stop_without_reversing():
    speeds = drive(moving(20.0), 0.0, 0.0, 15.0)
    assert speeds[-1] == 0.0
    assert min(speeds) >= 0.0


def test_braking_stops_quickly_then_reverses_slowly():
    speeds = drive(moving(24.0), 0.0, -1.0, 1.5)
    assert speeds[-1] <= 0.0
    reverse = drive(KartState(0.0, 0.0, 0.0), 0.0, -1.0, 10.0)
    assert -P.max_reverse_speed <= min(reverse) < -0.8 * P.max_reverse_speed


def test_throttle_while_rolling_backwards_brakes_first():
    speeds = drive(KartState(0.0, 0.0, 0.0, vx=-4.0), 0.0, 1.0, 0.25)
    assert speeds[0] > -4.0
    assert speeds[-1] > 0.0


def test_grass_slows_the_kart_down_hard():
    speeds = drive(moving(24.0), 0.0, 1.0, 1.2, on_grass=True)
    assert speeds[-1] <= P.grass_max_speed
    steady = drive(KartState(0.0, 0.0, 0.0), 0.0, 1.0, 10.0, on_grass=True)
    assert steady[-1] < P.grass_max_speed


def test_steering_wheel_moves_at_a_limited_rate():
    state = moving(10.0)
    step_kart(state, 1.0, 0.0, False, P)
    assert state.steer == pytest.approx(P.steer_rate * DT)
    drive(state, 1.0, 0.0, 1.0)
    assert state.steer == 1.0
    step_kart(state, 0.0, 0.0, False, P)
    assert state.steer == pytest.approx(1.0 - P.steer_return_rate * DT)


def test_kart_cannot_turn_while_stopped():
    state = KartState(0.0, 0.0, 0.3, steer=1.0)
    drive(state, 1.0, 0.0, 1.0)
    assert state.heading == pytest.approx(0.3)


def test_turning_gets_wider_with_speed():
    curvatures = []
    for speed in (5.0, 10.0, 20.0):
        state = KartState(0.0, 0.0, 0.0, vx=speed, steer=1.0)
        step_kart(state, 1.0, 0.0, False, P)
        curvatures.append(state.heading / (state.forward_speed * DT))
        assert curvatures[-1] == pytest.approx(max_path_curvature(speed, P), rel=0.05)
    assert curvatures[0] > curvatures[1] > curvatures[2]


def test_fast_corners_slide_a_little_and_recover():
    state = moving(22.0)
    drive(state, 1.0, 1.0, 1.0)
    assert 0.2 < abs(state.lateral_speed) < 3.0
    drive(state, 0.0, 1.0, 1.0)
    assert abs(state.lateral_speed) < 0.05


def test_random_inputs_keep_the_state_finite_and_bounded():
    rng = np.random.default_rng(0)
    state = KartState(0.0, 0.0, 0.0)
    for _ in range(20_000):
        steer, throttle = rng.uniform(-1.0, 1.0, 2)
        step_kart(state, steer, throttle, bool(rng.integers(2)), P)
        assert -math.pi <= state.heading <= math.pi
        assert -P.max_reverse_speed <= state.forward_speed <= P.max_speed
    assert all(math.isfinite(v) for v in (state.x, state.y, state.vx, state.vy, state.steer))


def test_inputs_are_clamped():
    a, b = moving(10.0), moving(10.0)
    drive(a, 5.0, 3.0, 1.0)
    drive(b, 1.0, 1.0, 1.0)
    assert (a.x, a.y, a.heading) == pytest.approx((b.x, b.y, b.heading))
