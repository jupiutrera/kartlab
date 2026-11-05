"""Pins the geometry of the reference tracks.

The evaluation tracks, a batch of training tracks and the showcase tracks must never
move: a change to the generator, the smoothing or the track format code would silently
invalidate every comparison with earlier runs. If this test fails after an intended
change, the change is not allowed to touch these tracks: fix the change, not the data.

For each track, ``tests/data/track_pins.json`` holds the number of points, the width and
16 evenly spaced points (rounded to 1 micrometre). Comparing values with a tolerance
instead of hashing exact bytes keeps the pin valid across platforms (Linux and Windows
differ in the last bits of libm and numpy SIMD results), while any real change to the
generator moves the points by far more than the tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kartlab.sim.showcase import SHOWCASE, showcase_track
from kartlab.sim.track import Track, generate_track
from kartlab.sim.tracksets import EVAL_SEEDS, eval_track

TRAIN_SEEDS = tuple(range(10))
SAMPLES = 16
TOLERANCE = 1e-5  # m; platform noise is around 1e-12, real changes are metres
PINS = json.loads((Path(__file__).parent / "data" / "track_pins.json").read_text("utf-8"))


def assert_pinned(track: Track, pin: dict) -> None:
    n = len(track.points)
    assert n == pin["count"]
    assert track.width == pin["width"]
    idx = [i * n // SAMPLES for i in range(SAMPLES)]
    np.testing.assert_allclose(track.points[idx], pin["points"], rtol=0, atol=TOLERANCE)


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


def test_pin_is_sensitive_to_geometry_and_width():
    track = generate_track(0)
    pin = PINS["train"]["0"]
    assert_pinned(track, pin)
    for moved in (
        Track(track.points + [1e-3, 0.0], track.width, "x"),
        Track(track.points, track.width + 1.0, "x"),
        Track(generate_track(1).points, track.width, "x"),
    ):
        with pytest.raises(AssertionError):
            assert_pinned(moved, pin)
