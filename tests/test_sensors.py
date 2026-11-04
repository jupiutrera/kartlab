import math

import numpy as np
import pytest

from kartlab.sim.sensors import RaySensor
from kartlab.sim.track import Track, generate_track


def straight_loop(width: float = 10.0, length: float = 400.0) -> Track:
    """A long thin rectangle with rounded ends: the start line sits on a straight."""
    half = length / 4
    radius = 40.0
    pts = []
    for x in np.arange(-half, half, 1.0):
        pts.append((x, -radius))
    for a in np.arange(-math.pi / 2, math.pi / 2, 1.0 / radius):
        pts.append((half + radius * math.cos(a), radius * math.sin(a)))
    for x in np.arange(half, -half, -1.0):
        pts.append((x, radius))
    for a in np.arange(math.pi / 2, 3 * math.pi / 2, 1.0 / radius):
        pts.append((-half + radius * math.cos(a), radius * math.sin(a)))
    return Track(np.array(pts), width, name="test")


def test_side_rays_measure_the_distance_to_each_edge():
    track = straight_loop(width=10.0)
    sensor = RaySensor(track, count=3, fov=math.pi, max_range=60.0)
    # On the bottom straight, driving towards +x, 2 m left of the centerline.
    x, y, heading = 0.0, -40.0 + 2.0, 0.0
    index = track.locate(x, y).index
    right, ahead, left = sensor.cast(x, y, heading, index)
    assert left == pytest.approx(3.0, abs=1e-6)
    assert right == pytest.approx(7.0, abs=1e-6)
    assert ahead == 60.0


def test_diagonal_ray_follows_trigonometry():
    track = straight_loop(width=10.0)
    sensor = RaySensor(track, count=1, fov=0.0, max_range=60.0)
    x, y = 0.0, -40.0
    index = track.locate(x, y).index
    (dist,) = sensor.cast(x, y, math.radians(30), index)
    assert dist == pytest.approx(5.0 / math.sin(math.radians(30)), abs=1e-6)


def test_rays_are_capped_and_symmetric_on_the_centerline():
    track = straight_loop(width=10.0)
    sensor = RaySensor(track, count=11, fov=math.pi, max_range=25.0)
    x, y = -20.0, -40.0
    dist = sensor.cast(x, y, 0.0, track.locate(x, y).index)
    assert np.all(dist <= 25.0) and np.all(dist > 0.0)
    np.testing.assert_allclose(dist, dist[::-1], atol=1e-9)


def brute_force(track: Track, x, y, angles, max_range) -> np.ndarray:
    """Independent reference: every full-resolution edge segment, one ray at a time."""
    starts = np.concatenate([track.left_edge, track.right_edge])
    ends = np.concatenate(
        [np.roll(track.left_edge, -1, axis=0), np.roll(track.right_edge, -1, axis=0)]
    )
    out = []
    for a in angles:
        d = np.array([math.cos(a), math.sin(a)])
        best = max_range
        for p, q in zip(starts, ends, strict=True):
            e = q - p
            denom = d[0] * e[1] - d[1] * e[0]
            if abs(denom) < 1e-12:
                continue
            w = p - (x, y)
            t = (w[0] * e[1] - w[1] * e[0]) / denom
            u = (w[0] * d[1] - w[1] * d[0]) / denom
            if t > 0 and 0 <= u <= 1:
                best = min(best, t)
        out.append(best)
    return np.array(out)


def test_rays_match_a_brute_force_reference_anywhere_on_the_asphalt():
    track = generate_track(4)
    sensor = RaySensor(track, count=9, fov=math.radians(200), max_range=60.0)
    rng = np.random.default_rng(0)
    for _ in range(25):
        s = rng.uniform(0, track.length)
        lateral = rng.uniform(-track.half_width, track.half_width) * 0.9
        x, y, heading = track.pose_at(s, lateral)
        heading += rng.uniform(-0.6, 0.6)
        index = track.locate(x, y).index
        expected = brute_force(track, x, y, heading + sensor.angles, 60.0)
        # 2 m chords sit < 4 cm off the edge; grazing rays stretch that along the ray.
        np.testing.assert_allclose(
            sensor.cast(x, y, heading, index), expected, atol=0.05, rtol=0.005
        )
