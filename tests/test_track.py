import math

import numpy as np
import pytest

from kartlab.sim.track import (
    SAMPLE_SPACING,
    Track,
    TrackParams,
    generate_track,
    has_clearance,
)

SEEDS = range(12)


@pytest.fixture(scope="module")
def tracks() -> dict[int, Track]:
    return {seed: generate_track(seed) for seed in SEEDS}


@pytest.fixture(scope="module")
def track(tracks) -> Track:
    return tracks[0]


def _segments_cross(points: np.ndarray, stride: int = 3) -> bool:
    """Independent check: does any pair of non-adjacent segments of a closed polyline cross?"""
    pts = points[::stride]
    a, b = pts, np.roll(pts, -1, axis=0)

    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (
            r[..., 0] - p[..., 0]
        )

    # Proper crossings only: endpoints strictly on opposite sides. A tolerance keeps
    # nearly collinear segments of a straight from flipping signs on rounding noise.
    eps = 1e-9
    a0, a1 = a[:, None, :], b[:, None, :]
    b0, b1 = a[None, :, :], b[None, :, :]
    cross = (orient(a0, a1, b0) * orient(a0, a1, b1) < -eps) & (
        orient(b0, b1, a0) * orient(b0, b1, a1) < -eps
    )
    n = len(pts)
    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    adjacent = np.minimum(sep, n - sep) <= 1
    return bool(np.any(cross & ~adjacent))


def test_same_seed_gives_the_same_track():
    np.testing.assert_array_equal(generate_track(7).points, generate_track(7).points)


def test_different_seeds_give_different_tracks(tracks):
    assert not np.allclose(tracks[1].points, tracks[2].points)


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_track_respects_the_limits(tracks, seed):
    track, params = tracks[seed], TrackParams()
    assert track.length == pytest.approx(params.length, rel=1e-9)
    assert np.max(np.abs(track.curvature)) <= 1 / params.min_radius
    assert has_clearance(track.points, params.width, params.clearance)
    assert track.segment_lengths.min() > 0.5 * SAMPLE_SPACING
    assert track.segment_lengths.max() < 1.5 * SAMPLE_SPACING


@pytest.mark.parametrize("seed", SEEDS)
def test_track_edges_never_cross(tracks, seed):
    track = tracks[seed]
    assert not _segments_cross(track.points)
    assert not _segments_cross(track.left_edge)
    assert not _segments_cross(track.right_edge)


def test_tracks_go_both_ways(tracks):
    turning = [np.sum(t.curvature * t.segment_lengths) for t in tracks.values()]
    assert all(abs(abs(total) - 2 * math.pi) < 0.05 for total in turning)
    assert any(total > 0 for total in turning) and any(total < 0 for total in turning)


def test_start_line_sits_on_the_straightest_stretch(tracks):
    for track in tracks.values():
        around_start = np.abs(np.concatenate([track.curvature[-30:], track.curvature[:30]]))
        assert around_start.mean() <= np.abs(track.curvature).mean()


def test_curviness_adds_corners():
    def total_turning(curviness):
        params = TrackParams(curviness=curviness)
        return np.mean([np.sum(np.abs(generate_track(s, params).curvature)) for s in range(6)])

    assert total_turning(1.0) > total_turning(0.0) * 1.3


def test_width_and_length_follow_the_params():
    track = generate_track(3, TrackParams(width=16.0, length=1200.0, min_radius=20.0))
    assert track.width == 16.0
    assert track.length == pytest.approx(1200.0)


@pytest.mark.parametrize(
    "kwargs", [{"curviness": 1.5}, {"curviness": -0.1}, {"width": 12.0, "min_radius": 5.0}]
)
def test_invalid_params_are_rejected(kwargs):
    with pytest.raises(ValueError):
        TrackParams(**kwargs)


def test_locate_points_on_the_centerline(track):
    for i in range(0, len(track), 37):
        x, y = track.points[i]
        pos = track.locate(float(x), float(y))
        assert abs(pos.lateral) < 1e-9
        assert pos.s == pytest.approx(track.s[i], abs=1e-6)


@pytest.mark.parametrize("lateral", [-4.0, -1.0, 2.5, 5.5])
def test_locate_recovers_arc_position_and_lateral_offset(track, lateral):
    for s in (12.3, 250.0, 611.7):
        x, y, _ = track.pose_at(s, lateral)
        pos = track.locate(x, y)
        assert pos.s == pytest.approx(s, abs=0.05)
        assert pos.lateral == pytest.approx(lateral, abs=0.05)


def test_locate_with_hint_follows_the_kart_continuously(track):
    pos = track.locate(*track.pose_at(0.0, 1.0)[:2])
    previous_s = pos.s
    travelled = 0.0
    for s in np.arange(0.4, 2 * track.length, 0.4):
        x, y, _ = track.pose_at(float(s), 1.0)
        pos = track.locate(x, y, hint=pos.index)
        step = (pos.s - previous_s) % track.length
        assert step == pytest.approx(0.4, abs=0.05)
        travelled += step
        previous_s = pos.s
    assert travelled == pytest.approx(2 * track.length - 0.4, rel=1e-3)


def test_pose_at_wraps_around(track):
    assert track.pose_at(track.length + 5.0) == pytest.approx(track.pose_at(5.0))


def test_heading_matches_direction_of_travel(track):
    x0, y0, heading = track.pose_at(100.2)
    x1, y1, _ = track.pose_at(100.2001)
    assert math.atan2(y1 - y0, x1 - x0) == pytest.approx(heading, abs=1e-6)
