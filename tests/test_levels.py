"""Bridge and tunnel at a level crossing: heights, branches, walls and the embankment."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kartlab.sim.autopilot import Autopilot
from kartlab.sim.kart import DT
from kartlab.sim.levels import (
    EMBANKMENT_MARGIN,
    WALL_INSET,
    WALL_MIN_HEIGHT,
    BridgeSpec,
    deck_half_length,
)
from kartlab.sim.race import MAX_STEP_DISTANCE, Racer
from kartlab.sim.timetrial import RecordStore, TimeTrial, record_filename
from kartlab.sim.track import generate_track
from kartlab.sim.trackfile import build_track, load_spec, load_track_file

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "tracks" / "ocho_puente.json"
PLAIN = ROOT / "tracks" / "ocho.json"


@pytest.fixture(scope="module")
def bridge():
    return load_track_file(BRIDGE)


@pytest.fixture(scope="module")
def plain():
    return load_track_file(PLAIN)


@pytest.fixture(scope="module")
def levels(bridge):
    assert bridge.levels is not None
    return bridge.levels


def drive(racer: Racer, steps: int, steer: float = 0.0, throttle: float = 1.0) -> None:
    for _ in range(steps):
        racer.step(steer, throttle)


def upper_frame(bridge):
    """Origin, axis and left normal of the upper road at the crossing."""
    levels = bridge.levels
    x, y, heading = bridge.pose_at(levels.s_over)
    return x, y, math.cos(heading), math.sin(heading)


# -- the track without a bridge is untouched ---------------------------------------------------


def test_only_the_bridge_track_has_levels(bridge, plain):
    assert bridge.levels is not None
    assert plain.levels is None and generate_track(5).levels is None
    assert plain.height_at(plain.crossings[0].s_b) == 0.0
    assert generate_track(5).height_at(100.0) == 0.0


def test_the_bridge_changes_nothing_while_no_wall_is_touched(bridge, plain):
    """Same file but the bridge: the autopilot drives bit-identical laps (the extra rules
    only act at walls and the embankment, which a centered kart never touches)."""
    a, b = Racer(plain), Racer(bridge)
    pilot_a, pilot_b = Autopilot(plain), Autopilot(bridge)
    for _ in range(int(plain.length / 18 / DT)):
        a.step(*pilot_a.act(a))
        b.step(*pilot_b.act(b))
        assert (a.kart.x, a.kart.y, a.kart.heading) == (b.kart.x, b.kart.y, b.kart.heading)
        assert not b.wall_hit
    assert a.progress.distance == b.progress.distance


def test_the_autopilot_drives_laps_over_the_bridge_and_through_the_tunnel(bridge):
    racer, pilot = Racer(bridge), Autopilot(bridge)
    top, grass = 0.0, 0
    while racer.laps < 2 and racer.time < 200:
        racer.step(*pilot.act(racer))
        top = max(top, racer.level_z)
        grass += racer.on_grass
    assert racer.laps == 2 and grass == 0
    assert top == pytest.approx(bridge.levels.deck_height)


def test_the_other_branch_can_go_over_too(bridge):
    """``over: "first"``: the first branch is the bridge and the second one the tunnel."""
    spec = replace(load_spec(BRIDGE), bridge=BridgeSpec(0, "first", 5.0, 40.0))
    track = build_track(spec)
    crossing = track.crossings[0]
    assert track.levels.s_over == crossing.s_a and track.levels.s_under == crossing.s_b
    assert track.height_at(crossing.s_a) == 5.0 and track.height_at(crossing.s_b) == 0.0
    racer, pilot = Racer(track), Autopilot(track)
    top = 0.0
    while racer.laps < 1 and racer.time < 100:
        racer.step(*pilot.act(racer))
        top = max(top, racer.level_z)
        assert not racer.on_grass and not racer.wall_hit
    assert racer.laps == 1 and top == 5.0


# -- heights --------------------------------------------------------------------------------


def test_the_lower_branch_is_at_ground_level(bridge, levels):
    away = np.abs(levels.wrap(bridge.s - levels.s_under))
    assert np.all(levels.height[away < levels.reach + 20] == 0.0)
    assert bridge.height_at(levels.s_under) == 0.0


def test_the_deck_is_flat_and_the_ramps_rise_monotonically(bridge, levels):
    away = np.abs(levels.wrap(bridge.s - levels.s_over))
    deck = levels.height[away <= levels.deck_half]
    assert np.all(deck == levels.deck_height)
    for side in (+1, -1):
        offsets = np.linspace(levels.deck_half, levels.reach, 80)
        heights = np.array([bridge.height_at(levels.s_over + side * d) for d in offsets])
        assert np.all(np.diff(heights) <= 1e-12)  # only goes down away from the deck
        assert heights[0] == pytest.approx(levels.deck_height, abs=0.01)
        assert heights[-1] == pytest.approx(0.0, abs=0.01)
    assert np.all(levels.height[away >= levels.reach] == 0.0)
    assert bridge.height_at(levels.s_over + levels.reach + 5) == 0.0


def test_the_deck_covers_the_whole_footprint_of_the_road_underneath(bridge, levels):
    angle = bridge.crossings[0].angle
    assert levels.deck_half == pytest.approx(deck_half_length(bridge.width, angle))
    assert levels.deck_half >= bridge.width / math.sin(angle) / 2 + 3.0


def test_height_at_interpolates_between_samples(bridge, levels):
    s = levels.s_over + levels.deck_half + 3.3
    low, high = sorted((levels.height_at(s - 0.5), levels.height_at(s + 0.5)))
    assert low <= levels.height_at(s) <= high
    assert bridge.height_at(s) == levels.height_at(s)


def test_which_branch_is_over_follows_the_file(bridge):
    crossing = bridge.crossings[0]
    assert bridge.levels.s_over == crossing.s_b and bridge.levels.s_under == crossing.s_a
    assert bridge.height_at(crossing.s_a) == 0.0 and bridge.height_at(crossing.s_b) > 3.0


# -- the kart keeps its branch ----------------------------------------------------------------


def test_locate_stays_on_the_branch_of_the_hint_even_on_the_other_asphalt(bridge, levels):
    c = bridge.crossings[0]
    # The same point of the plane: on the lower road's index it is the lower road, and on the
    # upper road's it is the upper one, whatever asphalt is under the kart.
    low = bridge.locate(c.x, c.y, hint=bridge.index_at(levels.s_under))
    high = bridge.locate(c.x, c.y, hint=bridge.index_at(levels.s_over))
    assert abs(low.s - levels.s_under) < 2.0 and abs(high.s - levels.s_over) < 2.0
    # Off the lower road's centerline but on the upper road's asphalt: still the lower road.
    x, y, ux, uy = upper_frame(bridge)
    pos = bridge.locate(x + 5.0 * ux, y + 5.0 * uy, hint=bridge.index_at(levels.s_under))
    assert abs(pos.s - levels.s_under) < 8.0 and abs(pos.lateral) > 4.0


def test_edge_covered_is_empty_with_levels(bridge, plain):
    for covered in bridge.edge_covered():
        assert not covered.any()
    assert any(covered.any() for covered in plain.edge_covered())  # the flat crossing has some


@pytest.mark.parametrize("which", ["under", "over"])
@pytest.mark.parametrize("turn", [math.pi / 2, -math.pi / 2])
def test_a_kart_turning_90_degrees_in_the_crossing_keeps_its_branch(bridge, levels, which, turn):
    s = levels.s_under if which == "under" else levels.s_over
    racer = Racer(bridge, start_s=s)
    racer.reset(s, heading_offset=turn, speed=20.0)
    z = racer.level_z
    drive(racer, 180)
    assert racer.level_z == z
    assert abs(racer.position.s - s) < levels.deck_half + 10  # still at its own crossing
    assert abs(racer.position.lateral) <= bridge.half_width - WALL_INSET + 1e-3
    assert not racer.on_grass


@pytest.mark.parametrize("which", ["under", "over"])
def test_on_grass_is_decided_by_the_own_branch(bridge, plain, levels, which):
    s = levels.s_under if which == "under" else levels.s_over
    # Seven metres off the centerline is beyond the asphalt of the own road, but at the
    # crossing it is on the asphalt of the other one.
    on_bridge = Racer(bridge, start_s=s, lateral=bridge.half_width + 0.5)
    assert on_bridge.on_grass
    flat = Racer(plain, start_s=s, lateral=bridge.half_width + 0.5)
    assert not flat.on_grass  # the flat crossing counts the other road as asphalt
    assert not Racer(bridge, start_s=s, lateral=bridge.half_width - 0.5).on_grass


# -- walls ------------------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["under", "over"])
@pytest.mark.parametrize("speed", [8.0, 26.0])
@pytest.mark.parametrize("angle", [math.radians(60), math.pi / 2])
def test_the_walls_stop_a_kart_at_any_speed_and_angle(bridge, levels, which, speed, angle):
    s = levels.s_under if which == "under" else levels.s_over
    inner = bridge.half_width - WALL_INSET
    racer = Racer(bridge, start_s=s - 3.0)
    racer.reset(s - 3.0, heading_offset=angle, speed=speed)
    hits = 0
    for _ in range(90):
        racer.step(0.0, 1.0)
        hits += racer.wall_hit
        if levels.walled[racer.position.index]:
            assert abs(racer.position.lateral) <= inner + 1e-3
        if racer.wall_hit:
            kart, pos = racer.kart, racer.position
            nx = -bridge._dir_y[pos.index] * math.copysign(1.0, pos.lateral)
            ny = bridge._dir_x[pos.index] * math.copysign(1.0, pos.lateral)
            assert kart.vx * nx + kart.vy * ny <= 1e-9  # nothing is left pushing outwards
    assert hits > 0
    assert (racer.level_z == 0.0) == (which == "under")


def test_the_wall_keeps_the_speed_along_it(bridge, levels):
    racer = Racer(bridge, start_s=levels.s_over - 5.0)
    racer.reset(levels.s_over - 5.0, heading_offset=math.radians(10), speed=15.0)
    along = []
    for _ in range(60):
        racer.step(0.0, 1.0)
        along.append(racer.kart.forward_speed)
    assert min(along) > 10.0  # a glancing blow does not stop the kart: no friction


def test_walls_only_stand_on_the_raised_road_and_the_tunnel(bridge, levels):
    away = np.abs(levels.wrap(bridge.s - levels.s_over))
    assert levels.walled[away <= levels.deck_half].all()
    assert not levels.walled[away > levels.reach + 1][
        np.abs(levels.wrap(bridge.s - levels.s_under))[away > levels.reach + 1] > levels.tunnel_half
    ].any()
    far = np.abs(levels.wrap(bridge.s - levels.s_under)) > levels.tunnel_half
    assert not levels.walled[far & (levels.height <= WALL_MIN_HEIGHT)].any()
    assert levels.walled[bridge.index_at(levels.s_under)]  # the tunnel


def test_a_kart_on_the_grass_is_not_pulled_onto_the_ramp(bridge, levels):
    """The walls are one-way: a kart beside the road stays outside, whatever it does."""
    s = levels.s_over - levels.deck_half - 8.0  # on a ramp, where it is walled
    assert levels.walled[bridge.index_at(s)]
    racer = Racer(bridge, start_s=s, lateral=bridge.half_width + 3.0)
    racer.reset(s, lateral=bridge.half_width + 3.0, heading_offset=-math.pi / 2, speed=10.0)
    for _ in range(120):
        racer.step(0.0, 0.5)
        assert abs(racer.position.lateral) > bridge.half_width


def test_a_kart_inside_the_tunnel_cannot_leave_it_sideways(bridge, levels):
    racer = Racer(bridge, start_s=levels.s_under - 2.0)
    racer.reset(levels.s_under - 2.0, heading_offset=math.pi / 2, speed=14.0)
    drive(racer, 120)
    assert racer.wall_hit or abs(racer.position.lateral) <= bridge.half_width - WALL_INSET + 1e-3
    assert racer.level_z == 0.0 and not racer.on_grass


# -- the embankment ---------------------------------------------------------------------------


def lower_kart(bridge, d: float, across: float, toward: float) -> Racer:
    """A lower-road kart on the grass at ``d`` along and ``across`` from the upper axis,
    heading to the axis at ``toward`` m/s."""
    x, y, ux, uy = upper_frame(bridge)
    nx, ny = -uy, ux
    racer = Racer(bridge, start_s=bridge.levels.s_under)
    kart = racer.kart
    kart.x, kart.y = x + d * ux + across * nx, y + d * uy + across * ny
    kart.heading = math.atan2(-ny * math.copysign(1, across), -nx * math.copysign(1, across))
    kart.vx, kart.vy = toward * math.cos(kart.heading), toward * math.sin(kart.heading)
    racer.position = bridge.locate(kart.x, kart.y, hint=bridge.index_at(bridge.levels.s_under))
    return racer


@pytest.mark.parametrize("side", [+1.0, -1.0])
@pytest.mark.parametrize("d", [14.0, 20.0, -25.0])
def test_the_embankment_cannot_be_crossed_from_the_grass_below(bridge, levels, side, d):
    racer = lower_kart(bridge, d, side * 14.0, 12.0)
    bank = bridge.half_width + EMBANKMENT_MARGIN
    x0, y0, ux, uy = upper_frame(bridge)
    pushed = 0
    for _ in range(240):
        racer.step(0.0, 1.0)
        pushed += racer.wall_hit
        across = abs(-(racer.kart.x - x0) * uy + (racer.kart.y - y0) * ux)
        assert across >= bank - 0.05, f"entered the embankment ({across:.2f} m from the axis)"
    assert pushed > 0


def test_the_tunnel_can_be_driven_through(bridge, levels):
    racer = Racer(bridge, start_s=levels.s_under - 40.0)
    racer.reset(levels.s_under - 40.0, speed=15.0)
    start = racer.position.s
    for _ in range(600):
        racer.step(0.0, 0.6)
        assert not racer.wall_hit and not racer.on_grass and racer.level_z == 0.0
        if racer.position.s > levels.s_under + 40.0:
            break
    assert racer.position.s > levels.s_under + 40.0 > start


# -- other karts, records and ghosts ------------------------------------------------------------


def test_two_karts_at_the_crossing_are_at_different_heights(bridge, levels):
    under = Racer(bridge, start_s=levels.s_under)
    over = Racer(bridge, start_s=levels.s_over)
    assert math.hypot(under.kart.x - over.kart.x, under.kart.y - over.kart.y) < 1.5
    assert over.level_z - under.level_z >= 1.0  # the contact rule ignores pairs this far apart
    assert under.level_z == 0.0 and over.level_z == pytest.approx(levels.deck_height)


def test_level_z_is_zero_everywhere_without_a_bridge(plain):
    racer = Racer(plain, start_s=plain.crossings[0].s_b)
    assert racer.level_z == 0.0
    assert Racer(generate_track(3)).level_z == 0.0


def test_the_two_eights_keep_separate_records(bridge, plain):
    assert plain.name != bridge.name
    assert record_filename(plain) != record_filename(bridge)


def test_the_ghost_crosses_the_bridge_without_changing_branch(bridge, levels, tmp_path):
    trial = TimeTrial(bridge, RecordStore(tmp_path))
    racer, pilot = Racer(bridge), Autopilot(bridge)
    trial.begin(racer)
    while racer.laps < 1:
        racer.step(*pilot.act(racer))
        event = trial.update(racer)
    assert event is not None and event.record
    hint = bridge.index_at(0.0)
    previous, highest = 0.0, 0.0
    for k in range(int(racer.lap_times[0] / DT)):
        x, y, _ = trial.ghost_pose(k * DT)
        pos = bridge.locate_following(x, y, hint)
        hint = pos.index
        jump = (pos.s - previous + bridge.length / 2) % bridge.length - bridge.length / 2
        assert abs(jump) < MAX_STEP_DISTANCE, f"the ghost jumped branch at t={k * DT:.2f}"
        previous = pos.s
        highest = max(highest, bridge.height_at(pos.s))
    assert highest == pytest.approx(levels.deck_height)
