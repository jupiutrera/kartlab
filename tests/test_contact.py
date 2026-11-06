"""Soft contact between the human kart and an AI kart (sim/contact.py), and Racer.displace."""

import math

import pytest

from kartlab.sim.contact import (
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
    resolve,
)
from kartlab.sim.kart import KartState
from kartlab.sim.race import Racer
from kartlab.sim.showcase import showcase_track

LATE = START_GRACE + 10.0  # a race time with no grace window left


def body(x, y, heading=0.0, vx=0.0, vy=0.0, **kw):
    return Body(KartState(x, y, heading, vx, vy), **kw)


def test_no_contact_without_overlap():
    contact = resolve(body(0, 0), body(3.0, 0), PairState(), LATE)
    assert contact is None


def test_overlap_pushes_apart_with_a_60_40_share():
    human, ai = body(0, 0), body(1.0, 0)  # 1 m apart: the circles overlap by ~0.2 m or more
    contact = resolve(human, ai, PairState(), LATE)
    assert contact is not None and contact.depth > 0
    assert contact.push_ai[0] > 0 > contact.push_human[0]  # the AI goes forward, the human back
    total = contact.push_ai[0] - contact.push_human[0]
    assert total == pytest.approx(MAX_CORRECTION)  # capped
    assert contact.push_ai[0] / total == pytest.approx(SHARE_AI)
    assert -contact.push_human[0] / total == pytest.approx(SHARE_HUMAN)


def test_small_overlap_is_corrected_fully_and_never_beyond_the_cap():
    human, ai = body(0, 0), body(1.8, 0)  # front circle of the human meets the rear one
    contact = resolve(human, ai, PairState(), LATE)
    assert contact is not None
    pushed = contact.push_ai[0] - contact.push_human[0]
    assert pushed == pytest.approx(min(contact.depth, MAX_CORRECTION))


def test_symmetry_mirrored_positions_give_mirrored_results():
    left = resolve(body(0, 0), body(0.0, 1.0, vy=-3.0), PairState(), LATE)
    right = resolve(body(0, 0), body(0.0, -1.0, vy=3.0), PairState(), LATE)
    assert left is not None and right is not None
    assert left.push_ai[1] == pytest.approx(-right.push_ai[1])
    assert left.push_human[1] == pytest.approx(-right.push_human[1])
    assert left.dv_ai[1] == pytest.approx(-right.dv_ai[1])
    assert left.dv_human[1] == pytest.approx(-right.dv_human[1])
    assert left.depth == pytest.approx(right.depth)


def test_velocity_change_removes_approach_but_never_adds_speed():
    human = body(0, 0, vx=10.0)
    ai = body(1.0, 0, vx=0.0)  # the human closes in at 10 m/s on a standing kart
    state = PairState()
    contact = resolve(human, ai, state, LATE)
    dv = contact.dv_ai[0] - contact.dv_human[0]  # change of the relative velocity
    assert dv == pytest.approx(MAX_DV_STEP)  # capped per step
    assert contact.dv_ai[0] == pytest.approx(SHARE_AI * MAX_DV_STEP)
    assert contact.dv_human[0] == pytest.approx(-SHARE_HUMAN * MAX_DV_STEP)
    # Separating karts: positions are corrected but no velocity is touched.
    separating = resolve(body(0, 0, vx=0.0), body(1.0, 0, vx=5.0), PairState(), LATE)
    assert separating.dv_ai == (0.0, 0.0) and separating.dv_human == (0.0, 0.0)


def test_accumulated_velocity_change_is_capped_per_contact_and_starts_again_after_a_pause():
    state = PairState()
    removed = 0.0
    for _ in range(100):
        contact = resolve(body(0, 0, vx=10.0), body(1.0, 0, vx=0.0), state, LATE)
        removed += contact.dv_ai[0] - contact.dv_human[0]
    assert removed == pytest.approx(MAX_DV_CONTACT)
    for _ in range(round(QUIET_RESET * 60) + 1):  # half a second apart
        assert resolve(body(0, 0), body(9.0, 0), state, LATE) is None
    contact = resolve(body(0, 0, vx=10.0), body(1.0, 0, vx=0.0), state, LATE)
    assert contact.dv_ai[0] - contact.dv_human[0] == pytest.approx(MAX_DV_STEP)


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
