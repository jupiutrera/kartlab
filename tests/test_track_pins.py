"""Pins the geometry of the reference tracks.

The evaluation tracks, a batch of training tracks and the showcase tracks must never
move: a change to the generator, the smoothing or the track format code would silently
invalidate every comparison with earlier runs. If this test fails after an intended
change, the change is not allowed to touch these tracks: fix the change, not the data.

Platform note: Linux and Windows draw the same shape, but ``finish_centerline`` picks the
start line with an argmin over sums of curvature whose near-ties resolve differently with
the last bits of libm and numpy SIMD, so the start sample moves by a few samples (the
whole centerline is shifted cyclically; found on CI, where Linux was 8 samples earlier on
evaluation track 1). Comparing points by index or hashing bytes therefore fails across
platforms. The pin is invariant to the start index instead. For each track,
``tests/data/track_pins.json`` holds the point count, the width, the closed length, the
driving direction and 128 evenly spaced points (rounded to 1 micrometre), and the test checks:

- count and width (exact), length (1 mm) and direction;
- every pinned point lies on the generated closed polyline (forward);
- every 8th generated point lies close to the pinned 128-point polyline (backward,
  coarse because chords of that polyline cut the corners).

Any real change to the generator moves the shape by far more than these tolerances.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kartlab.sim.showcase import SHOWCASE, showcase_track
from kartlab.sim.track import Track, closed_length, generate_track
from kartlab.sim.tracksets import EVAL_SEEDS, eval_track

TRAIN_SEEDS = tuple(range(10))
LENGTH_TOLERANCE = 1e-3  # m
ON_LINE_TOLERANCE = 1e-5  # m; platform noise is around 1e-12, pins are rounded to 1e-6
CHORD_TOLERANCE = 1.0  # m; sagitta of a 128-point chord at the tightest corner is ~0.6 m
PINS = json.loads((Path(__file__).parent / "data" / "track_pins.json").read_text("utf-8"))


def distance_to_closed_polyline(queries: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Distance from each query point to the closed polyline through ``polyline``."""
    a = polyline
    ab = np.roll(polyline, -1, axis=0) - a
    sq = np.sum(ab**2, axis=1)
    out = np.empty(len(queries))
    for k, q in enumerate(queries):
        t = np.clip(np.sum((q - a) * ab, axis=1) / sq, 0.0, 1.0)
        out[k] = np.hypot(*(a + ab * t[:, None] - q).T).min()
    return out


def is_counterclockwise(points: np.ndarray) -> bool:
    x, y = points[:, 0], points[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) > 0


def assert_pinned(track: Track, pin: dict) -> None:
    points = track.points
    assert len(points) == pin["count"]
    assert track.width == pin["width"]
    assert closed_length(points) == pytest.approx(pin["length"], abs=LENGTH_TOLERANCE)
    assert is_counterclockwise(points) == pin["ccw"]
    pinned = np.array(pin["points"])
    forward = distance_to_closed_polyline(pinned, points)
    assert forward.max() < ON_LINE_TOLERANCE, f"shape moved by {forward.max():.6f} m"
    backward = distance_to_closed_polyline(points[::8], pinned)
    assert backward.max() < CHORD_TOLERANCE, f"shape moved by {backward.max():.3f} m"


def test_pin_data_covers_every_reference_track():
    assert set(PINS["eval"]) == {str(s) for s in EVAL_SEEDS}
    assert set(PINS["train"]) == {str(s) for s in TRAIN_SEEDS}
    assert set(PINS["showcase"]) == set(SHOWCASE)


@pytest.mark.parametrize("index", range(len(EVAL_SEEDS)))
def test_evaluation_tracks_do_not_move(index):
    assert_pinned(eval_track(index), PINS["eval"][str(EVAL_SEEDS[index])])


@pytest.mark.parametrize("seed", TRAIN_SEEDS)
def test_training_tracks_do_not_move(seed):
    assert_pinned(generate_track(seed), PINS["train"][str(seed)])


@pytest.mark.parametrize("key", sorted(SHOWCASE))
def test_showcase_tracks_do_not_move(key):
    assert_pinned(showcase_track(key), PINS["showcase"][key])


@pytest.mark.parametrize("shift", [1, 8, -8, 123, 450])
def test_pin_ignores_a_cyclic_shift_of_the_start_line(shift):
    track = generate_track(0)
    moved = Track(np.roll(track.points, shift, axis=0), track.width, "x")
    assert_pinned(moved, PINS["train"]["0"])


def test_pin_is_sensitive_to_geometry_width_direction_and_track():
    track = generate_track(0)
    pin = PINS["train"]["0"]
    assert_pinned(track, pin)
    for moved in (
        Track(track.points + [1e-3, 0.0], track.width, "x"),  # 1 mm
        Track(track.points + [0.0, 1e-3], track.width, "x"),
        Track(track.points, track.width + 1.0, "x"),
        Track(track.points[::-1].copy(), track.width, "x"),  # driven the other way
        Track(generate_track(1).points, track.width, "x"),
    ):
        with pytest.raises(AssertionError):
            assert_pinned(moved, pin)
