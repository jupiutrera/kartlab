"""Soft contact between the human kart and an AI kart (sim/contact.py), and Racer.displace."""

import math

import pytest

from kartlab.sim.contact import (
    CIRCLE_RADIUS,
    CORRECTION_RATE,
    MAX_CORRECTION,
    MAX_DV_CONTACT,
    MAX_DV_STEP,
    QUIET_RESET,
    RESTART_GRACE,
    SHARE_AI,
    SHARE_HUMAN,
    START_GRACE,
    Body,
    PairState,
    circles,
    resolve,
)
from kartlab.sim.kart import DT, KartState
from kartlab.sim.race import Racer
from kartlab.sim.showcase import showcase_track

LATE = START_GRACE + 10.0  # a race time with no grace window left
TOUCH = 2.0 * CIRCLE_RADIUS  # circles closer than this overlap


def body(x, y, heading=0.0, vx=0.0, vy=0.0, **kw):
    return Body(KartState(x, y, heading, vx, vy), **kw)


def apply(human, ai, contact):
    """What the scene does with a contact, on the bare karts."""
    for body_, push, dv in (
        (human, contact.push_human, contact.dv_human),
        (ai, contact.push_ai, contact.dv_ai),
    ):
        body_.kart.x += push[0]
        body_.kart.y += push[1]
        body_.kart.vx += dv[0]
        body_.kart.vy += dv[1]


def min_circle_distance(a: KartState, b: KartState) -> float:
    return min(math.hypot(p[0] - q[0], p[1] - q[1]) for p in circles(a) for q in circles(b))


def test_no_contact_without_overlap_or_approach():
    assert resolve(body(0, 0), body(3.0, 0), PairState(), LATE) is None
    assert (
        resolve(body(0, 0, vx=5.0), body(30.0, 0), PairState(), LATE) is None
    )  # far, even closing in
    assert resolve(body(0, 0, vx=-5.0), body(3.0, 0), PairState(), LATE) is None  # moving away


def test_overlap_pushes_apart_with_a_60_40_share_that_grows_with_the_depth():
    shallow = resolve(body(0, 0), body(1.8, 0), PairState(), LATE)  # circles overlap by 0.3 m
    deep = resolve(body(0, 0), body(1.5, 0), PairState(), LATE)  # by 0.6 m
    for contact, depth in ((shallow, 0.3), (deep, 0.6)):
        assert contact is not None and contact.depth == pytest.approx(depth)
        total = contact.push_ai[0] - contact.push_human[0]
        assert total == pytest.approx(CORRECTION_RATE * depth)
        assert contact.push_ai[0] / total == pytest.approx(SHARE_AI)
        assert -contact.push_human[0] / total == pytest.approx(SHARE_HUMAN)
    assert deep.push_ai[0] == pytest.approx(2 * shallow.push_ai[0])
    buried = resolve(body(0, 0), body(0.2, 0), PairState(), LATE)
    assert buried.push_ai[0] - buried.push_human[0] == pytest.approx(MAX_CORRECTION)  # capped


def test_the_approach_is_removed_entirely_with_the_same_share_and_no_bounce():
    human, ai = body(0, 0, vx=0.4), body(1.0, 0, vx=0.0)  # closing at 0.4 m/s, overlapping
    contact = resolve(human, ai, PairState(), LATE)
    assert contact.dv_ai[0] == pytest.approx(SHARE_AI * 0.4)
    assert contact.dv_human[0] == pytest.approx(-SHARE_HUMAN * 0.4)
    apply(human, ai, contact)
    assert human.kart.vx == pytest.approx(ai.kart.vx)  # they move together: restitution 0
    # Separating karts keep their velocities (only the positions are corrected).
    parting = resolve(body(0, 0, vx=0.0), body(1.0, 0, vx=5.0), PairState(), LATE)
    assert parting.dv_ai == (0.0, 0.0) and parting.dv_human == (0.0, 0.0)


def test_the_ai_never_takes_more_than_its_caps_and_the_human_absorbs_the_rest():
    human, ai = body(0, 0, vx=20.0), body(1.0, 0, vx=0.0)
    contact = resolve(human, ai, PairState(), LATE)
    assert contact.dv_ai[0] == pytest.approx(MAX_DV_STEP)
    assert contact.dv_human[0] == pytest.approx(-(20.0 - MAX_DV_STEP))
    apply(human, ai, contact)
    assert human.kart.vx == pytest.approx(ai.kart.vx)  # still no approach left, still no bounce


def test_a_glued_human_cannot_keep_pushing_the_ai_more_than_the_contact_total():
    """A human that keeps closing in (the AI brakes in front of it) moves the AI at most
    ``MAX_DV_CONTACT`` in all, and the total starts again after a pause."""
    state = PairState()
    pushed = 0.0
    for _ in range(120):
        human, ai = body(0, 0, vx=8.0), body(1.0, 0, vx=0.0)  # the same approach every step
        contact = resolve(human, ai, state, LATE)
        pushed += contact.dv_ai[0]
        assert contact.dv_ai[0] - contact.dv_human[0] == pytest.approx(8.0)  # approach removed
    assert pushed == pytest.approx(MAX_DV_CONTACT)
    for _ in range(round(QUIET_RESET / DT) + 1):
        assert resolve(body(0, 0), body(9.0, 0), state, LATE) is None
    contact = resolve(body(0, 0, vx=8.0), body(1.0, 0), state, LATE)
    assert contact.dv_ai[0] == pytest.approx(MAX_DV_STEP)


def test_symmetry_mirrored_positions_give_mirrored_results():
    left = resolve(body(0, 0), body(0.0, 1.0, vy=-3.0), PairState(), LATE)
    right = resolve(body(0, 0), body(0.0, -1.0, vy=3.0), PairState(), LATE)
    assert left is not None and right is not None
    assert left.push_ai[1] == pytest.approx(-right.push_ai[1])
    assert left.push_human[1] == pytest.approx(-right.push_human[1])
    assert left.dv_ai[1] == pytest.approx(-right.dv_ai[1])
    assert left.dv_human[1] == pytest.approx(-right.dv_human[1])
    assert left.depth == pytest.approx(right.depth)


# -- never through each other, at any speed ---------------------------------------------------


def run_pass(human, ai, steps=240, throttle_push=0.0):
    """Both karts drive on in a straight line (``step_kart`` is not needed: the point is the
    geometry) with contact resolved every step. Returns the smallest distance between any
    two circles seen at the start of a step."""
    closest = math.inf
    state = PairState()
    for i in range(steps):
        for b in (human, ai):
            b.kart.x += b.kart.vx * DT
            b.kart.y += b.kart.vy * DT
        closest = min(closest, min_circle_distance(human.kart, ai.kart))
        contact = resolve(human, ai, state, LATE + i * DT)
        if contact is not None:
            apply(human, ai, contact)
        if throttle_push:  # the driver keeps the pedal down: the speed comes back every step
            human.kart.vx += throttle_push * DT
    return closest


@pytest.mark.parametrize("closing", [2.0, 6.0, 10.0, 24.0, 40.0, 52.0])
def test_a_kart_never_goes_through_another_from_behind(closing):
    human = body(-12.0, 0.0, vx=closing)
    ai = body(0.0, 0.0, vx=0.0)
    assert run_pass(human, ai) >= 0.8 * TOUCH
    assert human.kart.x < ai.kart.x  # still behind it


@pytest.mark.parametrize("ai_speed", [0.0, 12.0, 18.0, 23.0])
def test_overtaking_fast_does_not_cross_the_slower_kart(ai_speed):
    human = body(-9.0, 0.0, vx=24.0)  # 24 m/s on a kart doing less, dead behind it
    ai = body(0.0, 0.0, vx=ai_speed)
    assert run_pass(human, ai, throttle_push=14.0) >= 0.8 * TOUCH
    assert human.kart.x < ai.kart.x


@pytest.mark.parametrize("speed", [5.0, 13.0, 26.0])
def test_head_on_karts_do_not_cross(speed):
    human = body(-15.0, 0.0, heading=0.0, vx=speed)
    ai = body(15.0, 0.0, heading=math.pi, vx=-speed)  # closing at up to 52 m/s
    assert run_pass(human, ai) >= 0.8 * TOUCH
    assert human.kart.x < ai.kart.x


@pytest.mark.parametrize("speed", [8.0, 26.0])
def test_a_side_on_hit_does_not_cross(speed):
    human = body(0.0, -10.0, heading=math.pi / 2, vy=speed)  # hits it broadside
    ai = body(0.0, 0.0, heading=0.0, vx=0.0)
    assert run_pass(human, ai) >= 0.8 * TOUCH
    assert human.kart.y < ai.kart.y


@pytest.mark.parametrize("offset", [0.9, 1.3, 1.8])
def test_overtaking_alongside_at_speed_does_not_overlap(offset):
    human = body(-6.0, offset, vx=24.0)
    ai = body(0.0, 0.0, vx=18.0)
    closest = run_pass(human, ai, steps=120)
    assert closest >= 0.8 * TOUCH


def test_a_clean_pass_wide_of_the_other_kart_is_not_touched():
    human = body(-10.0, 3.0, vx=24.0)
    ai = body(0.0, 0.0, vx=10.0)
    state = PairState()
    for i in range(90):
        for b in (human, ai):
            b.kart.x += b.kart.vx * DT
        assert resolve(human, ai, state, LATE + i * DT) is None
    assert human.kart.vx == 24.0


# -- when it does not apply ---------------------------------------------------------------------


def test_no_contact_between_different_levels():
    flat = body(0, 0)
    assert resolve(flat, body(1.0, 0, level_z=1.0), PairState(), LATE) is None
    assert resolve(flat, body(1.0, 0, level_z=-1.5), PairState(), LATE) is None
    assert resolve(flat, body(1.0, 0, level_z=0.9), PairState(), LATE) is not None
    # The level and the height add up: a hop over a bridge counts from the ground.
    assert resolve(body(0, 0, level_z=1.0), body(1.0, 0), PairState(), LATE) is None


def test_no_contact_while_a_kart_is_in_the_air():
    assert resolve(body(0, 0, height=0.3), body(1.0, 0), PairState(), LATE) is None
    assert resolve(body(0, 0), body(1.0, 0, height=0.5), PairState(), LATE) is None
    assert resolve(body(0, 0, height=0.2), body(1.0, 0), PairState(), LATE) is not None


def test_grace_windows():
    start = START_GRACE  # the scene gives every kart this grace until the start is over
    human, ai = body(0, 0, grace_until=start), body(1.0, 0, grace_until=start)
    assert resolve(human, ai, PairState(), start - 0.1) is None
    assert resolve(human, ai, PairState(), start + 0.1) is not None
    restarted = body(1.0, 0, grace_until=20.0 + RESTART_GRACE)  # an AI put back on the track
    assert resolve(body(0, 0), restarted, PairState(), 20.5) is None
    assert resolve(body(0, 0), restarted, PairState(), 20.0 + RESTART_GRACE + 0.1) is not None


# -- Racer ------------------------------------------------------------------------------------


def test_displace_moves_the_kart_and_relocates_it_on_the_track():
    track = showcase_track("harbor")
    racer = Racer(track, start_s=100.0)
    before = racer.position.s
    nudge = 0.5
    racer.displace(0.0, nudge)
    assert racer.kart.y == pytest.approx(track.pose_at(100.0)[1] + nudge)
    assert racer.position.s == pytest.approx(before, abs=2.0)
    x0, y0, _ = track.pose_at(100.0)
    assert math.hypot(racer.kart.x - x0, racer.kart.y - y0) == pytest.approx(nudge)
    # Pushed well onto the grass, the surface follows.
    racer.displace(0.0, 0.0)
    lateral_edge = track.half_width + 3.0
    x, y, _ = track.pose_at(100.0, lateral_edge)
    racer.displace(x - racer.kart.x, y - racer.kart.y)
    assert racer.on_grass


def test_level_z_is_flat_for_now():
    assert Racer(showcase_track("harbor")).level_z == 0.0
