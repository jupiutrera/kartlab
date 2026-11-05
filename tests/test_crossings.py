"""Level crossings: format 2, validation, branch-following location, surface and progress."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kartlab.env.kart_env import KartEnv
from kartlab.sim.autopilot import Autopilot
from kartlab.sim.race import MAX_STEP_DISTANCE, Racer
from kartlab.sim.track import (
    Crossing,
    crossing_zone,
    find_crossings,
    generate_track,
    has_clearance,
    has_clearance_with_crossings,
)
from kartlab.sim.trackfile import (
    CLEARANCE,
    FORMAT_VERSION,
    TrackFileError,
    TrackSpec,
    build_track,
    check_centerline,
    load_spec,
    load_track_file,
    save_spec,
    spec_from_dict,
    spec_to_dict,
    validate_spec,
)
from kartlab.sim.tracksets import eval_track

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "tracks" / "ocho.json"

# The showcase design: a figure eight whose two roads cross at right angles at the origin.
EIGHT = TrackSpec(
    "Lazo del Molino",
    13.0,
    (
        (-130.0, -130.0, 45.0),
        (-215.0, 200.0, 50.0),
        (150.0, -150.0, 45.0),
        (250.0, 20.0, 70.0),
        (150.0, 150.0, 45.0),
    ),
    3,
    ((0.0, 0.0),),
    (206.6, -54.3),
)


@pytest.fixture(scope="module")
def eight():
    return build_track(EIGHT)


@pytest.fixture(scope="module")
def crossing(eight):
    (c,) = eight.crossings
    return c


def problems(spec: TrackSpec) -> str:
    return "\n".join(validate_spec(spec).problems)


# -- the file format ---------------------------------------------------------------------


def test_the_showcase_file_is_the_same_design():
    assert load_spec(FILE) == EIGHT
    track = load_track_file(FILE)
    assert track.name == "Lazo del Molino" and len(track.crossings) == 1
    assert 900.0 <= track.length <= 1400.0 and track.width >= 12.0


def test_format_2_round_trips_and_format_1_is_untouched(tmp_path):
    path = tmp_path / "ocho.json"
    save_spec(path, EIGHT)
    raw = path.read_bytes()
    assert b"\r" not in raw and json.loads(raw) == spec_to_dict(EIGHT)
    assert json.loads(raw)["format"] == 2 and json.loads(raw)["crossings"] == [[0, 0]]
    assert load_spec(path) == EIGHT
    plain = replace(EIGHT, crossings=())
    assert spec_to_dict(plain)["format"] == FORMAT_VERSION and "crossings" not in spec_to_dict(
        plain
    )


def test_crossings_need_format_2_and_format_2_needs_crossings():
    data = spec_to_dict(EIGHT)
    with pytest.raises(TrackFileError, match="format': 2"):
        spec_from_dict({**data, "format": 1})
    with pytest.raises(TrackFileError, match="no soportado"):
        spec_from_dict({**data, "crossings": []})
    del data["crossings"]
    with pytest.raises(TrackFileError, match="no soportado"):
        spec_from_dict(data)


@pytest.mark.parametrize(
    "bad", [5, "x", [[1]], [[1, "a"]], [[0, 0]] * 5, [[float("nan"), 0]]], ids=str
)
def test_malformed_crossings_are_reported(bad):
    with pytest.raises(TrackFileError, match="cruce|crossings"):
        spec_from_dict({**spec_to_dict(EIGHT), "crossings": bad})


# -- validation --------------------------------------------------------------------------


def test_the_figure_eight_is_valid_and_has_one_clean_crossing(eight, crossing):
    report = validate_spec(EIGHT)
    assert report.ok and report.crossings == 1
    assert report.gap is not None and report.gap >= CLEARANCE
    assert math.degrees(crossing.angle) == pytest.approx(90.0, abs=5.0)
    assert math.hypot(crossing.x, crossing.y) < 10.0
    assert abs(crossing.s_a - crossing.s_b) > 100.0
    # The start line is on a stretch far from the crossing.
    assert math.hypot(eight.points[0, 0] - crossing.x, eight.points[0, 1] - crossing.y) > 100.0


def test_a_crossing_must_be_declared():
    undeclared = replace(EIGHT, crossings=())
    assert "se cruza a sí mismo" in problems(undeclared)  # the format 1 rule, unchanged
    # Declared far from the real one: both the declaration and the real crossing are errors.
    elsewhere = replace(EIGHT, crossings=((100.0, 100.0),))
    text = problems(elsewhere)
    assert "no existe" in text and "no está declarado" in text


def test_an_enormous_track_with_crossings_is_rejected_before_the_quadratic_checks():
    huge = replace(EIGHT, corners=tuple((x * 8.0, y * 8.0, 45.0) for x, y, _ in EIGHT.corners))
    report = validate_spec(huge)
    assert not report.ok and "longitud" in "\n".join(report.problems)


def test_a_declared_crossing_must_exist():
    corners = EIGHT.corners[:1] + ((300.0, 0.0, 40.0), (150.0, 200.0, 40.0))
    plain = replace(EIGHT, corners=corners, start=None)
    assert "no existe" in problems(plain)


def test_an_oblique_crossing_is_rejected():
    spec = TrackSpec(
        "Oblicuo",
        12.0,
        (
            (-300.0, -150.0, 40.0),
            (-300.0, 150.0, 40.0),
            (300.0, -150.0, 40.0),
            (300.0, 150.0, 40.0),
        ),
        0,
        ((0.0, 0.0),),
    )
    assert "demasiado oblicuo" in problems(spec)


def test_a_crossing_next_to_a_corner_is_rejected():
    # Same design, one corner pulled in so its fillet starts almost at the crossing.
    corners = ((-60.0, -60.0, 45.0), *EIGHT.corners[1:])
    assert "demasiado cerca de una curva" in problems(replace(EIGHT, corners=corners))


def test_a_crossing_on_the_start_line_is_rejected(eight, crossing):
    rolled = np.roll(eight.points, -crossing.index_a, axis=0)
    text = "\n".join(check_centerline(rolled, eight.width, EIGHT.crossings).problems)
    assert "línea de meta" in text


def test_the_build_keeps_the_start_away_from_the_crossing(eight):
    for c in find_crossings(eight.points):
        assert math.hypot(eight.points[0, 0] - c.x, eight.points[0, 1] - c.y) > 70.0


# -- the declared start line (the same on every platform) ----------------------------------


def test_the_declared_start_is_where_sample_0_goes(eight):
    assert math.hypot(eight.points[0, 0] - 206.6, eight.points[0, 1] + 54.3) < 0.6
    # Another straight stretch of the same circuit: far from the first start and from the
    # crossing, with no bend within 30 m either way.
    bend = np.abs(eight.curvature)
    straight = np.array(
        [bend[np.arange(i - 30, i + 31) % len(bend)].max() for i in range(len(bend))]
    )
    away = np.hypot(*(eight.points - eight.points[0]).T) > 150.0
    cross = eight.crossings[0]
    away &= np.hypot(eight.points[:, 0] - cross.x, eight.points[:, 1] - cross.y) > 100.0
    pick = int(np.flatnonzero(away & (straight < 1 / 500))[0])
    target = (round(float(eight.points[pick, 0]), 1), round(float(eight.points[pick, 1]), 1))
    moved = build_track(replace(EIGHT, start=target))
    assert math.hypot(moved.points[0, 0] - target[0], moved.points[0, 1] - target[1]) < 0.6
    assert moved.length == pytest.approx(eight.length, abs=0.01)  # same circuit, other line


def test_without_start_the_line_is_still_placed_automatically(crossing):
    auto = build_track(replace(EIGHT, start=None))
    assert len(auto.crossings) == 1
    c = auto.crossings[0]
    assert math.hypot(auto.points[0, 0] - c.x, auto.points[0, 1] - c.y) > 70.0


def test_a_bad_start_is_rejected(eight):
    assert "no está sobre el circuito" in problems(replace(EIGHT, start=(0.0, 300.0)))
    bend = np.argmax(np.abs(eight.curvature))
    corner = tuple(float(v) for v in eight.points[bend])
    assert "está en una curva" in problems(replace(EIGHT, start=corner))
    near = (eight.crossings[0].x, eight.crossings[0].y)
    assert "línea de meta" in problems(replace(EIGHT, start=near))


def test_start_is_part_of_format_2_only(tmp_path):
    data = spec_to_dict(EIGHT)
    assert data["start"] == [206.6, -54.3]
    assert spec_from_dict(data).start == (206.6, -54.3)
    with pytest.raises(TrackFileError, match="'start'"):
        spec_from_dict({k: v for k, v in data.items() if k != "crossings"} | {"format": 1})
    for bad in ([1], "x", [1, "a"], [float("inf"), 0]):
        with pytest.raises(TrackFileError, match="'start'"):
            spec_from_dict({**data, "start": bad})


def test_other_near_touching_sections_are_still_rejected(eight, crossing):
    # Pull the far end of the right lobe back onto the left road: two stretches that are
    # nowhere near the crossing now touch.
    points = eight.points.copy()
    far = np.hypot(points[:, 0] - crossing.x, points[:, 1] - crossing.y) > 150.0
    points[far & (points[:, 0] > 0)] += np.array([-300.0, 0.0])
    assert not has_clearance_with_crossings(points, eight.width, CLEARANCE, [crossing])


def test_clearance_with_crossings_is_a_superset_of_has_clearance(eight, crossing):
    assert has_clearance_with_crossings(eight.points, eight.width, CLEARANCE, [crossing])
    assert not has_clearance(eight.points, eight.width, CLEARANCE)  # the plain rule refuses it
    for seed in (1, 2, 3):
        pts = generate_track(seed).points
        assert has_clearance_with_crossings(pts, 12.0, CLEARANCE, []) == has_clearance(
            pts, 12.0, CLEARANCE
        )
    zone = crossing_zone(eight.width, CLEARANCE)
    assert zone > eight.width + CLEARANCE


def test_procedural_tracks_have_no_crossings():
    assert generate_track(4).crossings == ()
    assert eval_track(0).crossings == ()
    assert find_crossings(generate_track(4).points) == []


# -- locating a point ------------------------------------------------------------------


def test_a_hinted_locate_stays_on_its_own_branch(eight, crossing):
    # A point inside the crossing box, off both centerlines: it is on both roads.
    x, y, _ = eight.pose_at(crossing.s_a, 3.0)
    on_a = eight.locate(x, y, crossing.index_a)
    on_b = eight.locate(x, y, crossing.index_b)
    assert abs(on_a.index - crossing.index_a) < 8 and abs(on_b.index - crossing.index_b) < 8
    assert abs(on_a.lateral) <= eight.half_width and abs(on_b.lateral) <= eight.half_width
    # A point on road B just beyond the edge of road A: hinted on A it is on grass relative
    # to A, so the locate moves to B, where it is on asphalt.
    x, y, _ = eight.pose_at(crossing.s_b + eight.half_width + 3.0, 0.0)
    moved = eight.locate(x, y, crossing.index_a)
    assert abs(moved.index - (crossing.index_b + eight.half_width + 3.0)) < 3
    assert abs(moved.lateral) < 0.5
    # Without a hint, an exact tie is decided by the nearest sample: always the same answer.
    assert eight.locate(crossing.x, crossing.y) == eight.locate(crossing.x, crossing.y)


def test_the_other_road_is_asphalt_not_grass(eight, crossing):
    assert eight.contains(crossing.x, crossing.y)
    for index in (crossing.index_a, crossing.index_b):
        for lateral in (-eight.half_width + 0.5, 0.0, eight.half_width - 0.5):
            x, y, _ = eight.pose_at(eight.s[index] + 4.0, lateral)
            assert eight.contains(x, y)
    # Just outside the corner of the crossing box it is grass.
    d = eight.half_width + 2.0
    assert not eight.contains(crossing.x + d * math.sqrt(2), crossing.y)


def test_reset_at_the_crossing_picks_the_branch_of_start_s(eight, crossing):
    for s_c, index in ((crossing.s_a, crossing.index_a), (crossing.s_b, crossing.index_b)):
        racer = Racer(eight, start_s=s_c)
        assert abs(racer.position.index - index) <= 1
        assert racer.position.s == pytest.approx(s_c, abs=1.0)
        assert not racer.on_grass
        racer.reset(start_s=s_c - 20.0)
        assert racer.position.s == pytest.approx(s_c - 20.0, abs=0.1)


# -- driving through it ----------------------------------------------------------------


def racer_at(track, s: float, speed: float = 0.0) -> Racer:
    racer = Racer(track)
    racer.reset(start_s=s, speed=speed)
    return racer


def drive(racer: Racer, steps: int, steer: float = 0.0, throttle: float = 0.6):
    """Step and record (on_grass, progress delta, branch index) for each step."""
    trace = []
    for _ in range(steps):
        delta = racer.step(steer, throttle)
        trace.append((racer.on_grass, delta, racer.position.index))
    return trace


@pytest.mark.parametrize("branch", ["a", "b"])
def test_driving_straight_through_the_crossing_is_never_grass(eight, crossing, branch):
    s_c = crossing.s_a if branch == "a" else crossing.s_b
    racer = racer_at(eight, s_c - 70.0, 12.0)
    trace = []
    while racer.position.s < s_c + 30.0 and len(trace) < 60 * 20:  # straight, never steering
        trace += drive(racer, 1)
    assert racer.position.s >= s_c + 30.0
    assert not any(grass for grass, _, _ in trace)
    assert max(abs(delta) for _, delta, _ in trace) < 0.5  # no jump in progress
    indices = [i for _, _, i in trace]
    assert all(0 <= b - a <= 1 for a, b in zip(indices, indices[1:], strict=False))
    assert racer.progress.distance == pytest.approx(sum(d for _, d, _ in trace))
    assert racer.progress.distance == pytest.approx(100.0, abs=3.0) and racer.laps == 0
    assert abs(racer.position.lateral) < 3.0


def test_a_lateral_cut_across_the_crossing_road_is_never_grass(eight, crossing):
    # Drive along one branch while crossing the other road at right angles: the kart is on
    # the second road's asphalt the whole time it is inside the zone.
    racer = racer_at(eight, crossing.s_b - 40.0, 12.0)
    seen_inside = 0
    for _ in range(60 * 4):
        racer.step(0.0, 0.6)
        inside = math.hypot(racer.kart.x - crossing.x, racer.kart.y - crossing.y) < 15.0
        seen_inside += inside
        assert not (inside and racer.on_grass)
    assert seen_inside > 20


def turn_onto(racer: Racer, heading: float) -> None:
    """Point the kart along ``heading`` keeping its speed (a sharp turn, in one step)."""
    speed = racer.kart.forward_speed
    racer.kart.heading = heading
    racer.kart.vx, racer.kart.vy = speed * math.cos(heading), speed * math.sin(heading)


def test_a_90_degree_turn_inside_the_crossing_follows_the_new_branch(eight, crossing):
    racer = racer_at(eight, crossing.s_a - 15.0, 10.0)
    drive(racer, 60)  # arrive at the crossing on branch A
    assert abs(racer.position.index - crossing.index_a) < 40
    assert abs(racer.position.s - crossing.s_a) < 15.0
    turn_onto(racer, eight.heading_at_index(crossing.index_b))
    before = racer.progress.distance
    trace = drive(racer, 60 * 3, throttle=0.3)
    assert not any(grass for grass, _, _ in trace)
    assert racer.position.s > crossing.s_b  # on branch B, going its way (not back to A)
    assert racer.position.s < crossing.s_b + 100.0
    assert abs(racer.position.lateral) < eight.half_width
    # The switch did not credit the jump from one branch to the other.
    assert max(abs(delta) for _, delta, _ in trace) < MAX_STEP_DISTANCE
    assert 0.0 < racer.progress.distance - before < 3 * 20.0
    assert racer.laps == 0


def test_circling_at_the_crossing_gains_nothing(eight, crossing):
    for steer in (0.3, -0.3, 1.0, -1.0):
        racer = racer_at(eight, crossing.s_a, 8.0)
        drive(racer, 60 * 60, steer=steer, throttle=0.25)
        assert abs(racer.progress.distance) < 60.0 and racer.laps == 0


def test_skipping_a_loop_at_the_crossing_does_not_count_laps(eight, crossing):
    """Turning from branch A onto B at the crossing repeats the second loop. The distance
    keeps adding up honestly, but a lap needs every part of the track, so none counts."""
    racer = racer_at(eight, crossing.s_b + 20.0, 10.0)
    pilot = Autopilot(eight)
    turns, last_turn = 0, -99.0
    for _ in range(60 * 600):
        racer.step(*pilot.act(racer))
        if abs(racer.position.s - crossing.s_a) < 2.0 and racer.time - last_turn > 10.0:
            turn_onto(racer, eight.heading_at_index(crossing.index_b))
            turns, last_turn = turns + 1, racer.time
            if turns == 3:
                break
    assert turns == 3
    assert racer.progress.distance > 1.5 * eight.length
    assert racer.laps == 0 and racer.lap_times == []


def test_the_autopilot_drives_laps_through_the_crossing(eight):
    racer = Racer(eight)
    pilot = Autopilot(eight)
    grass = 0
    for _ in range(60 * 200):
        racer.step(*pilot.act(racer))
        grass += racer.on_grass
        if racer.laps >= 2:
            break
    assert racer.laps == 2 and grass == 0
    first, second = racer.lap_times
    assert first == pytest.approx(second, rel=0.05)
    assert racer.progress.distance == pytest.approx(2 * eight.length, abs=15.0)


# -- procedural tracks keep the old rules -----------------------------------------------


def test_on_grass_on_other_tracks_is_the_lateral_rule():
    track = generate_track(5)
    racer = Racer(track)
    rng = np.random.default_rng(0)
    for _ in range(300):
        racer.reset(
            start_s=float(rng.uniform(0, track.length)), lateral=float(rng.uniform(-12, 12))
        )
        assert racer.on_grass == (abs(racer.position.lateral) > track.half_width)
        racer.step(float(rng.uniform(-1, 1)), 1.0)
        assert racer.on_grass == (abs(racer.position.lateral) > track.half_width)
    assert racer.laps == racer.progress.laps


# -- the reward ------------------------------------------------------------------------


def test_the_reward_through_the_crossing_is_plain_progress(eight, crossing):
    env = KartEnv(
        random_start=False, terminate_off_track=True, steer_change_penalty=0.0, max_seconds=60.0
    )
    env.reset(options={"track": eight})
    env.racer.reset(start_s=crossing.s_a - 60.0, speed=12.0)
    rewards, progress = [], []
    for _ in range(15 * 6):
        before = env.racer.progress.distance
        _, reward, terminated, _, info = env.step(np.array([0.0, 0.6]))
        assert not terminated
        rewards.append(reward)
        progress.append(env.racer.progress.distance - before)
    assert all(
        r == pytest.approx(0.1 * p, abs=1e-9) for r, p in zip(rewards, progress, strict=True)
    )
    assert max(rewards) < 0.1 * 6.0  # 15 Hz decisions at 12-20 m/s: a few metres each
    assert info["laps"] == 0


def test_crossing_dataclass_is_plain_data(crossing):
    assert isinstance(crossing, Crossing) and crossing.index_a < crossing.index_b
