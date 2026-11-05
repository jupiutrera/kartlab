"""Pins the exact geometry of the reference tracks.

The evaluation tracks, a batch of training tracks and the showcase tracks must never
move: a change to the generator, the smoothing or the track format code would silently
invalidate every comparison with earlier runs. If this test fails after an intended
change, the change is not allowed to touch these tracks: fix the change, not the hashes.
"""

from __future__ import annotations

import hashlib

import numpy as np

from kartlab.sim.showcase import SHOWCASE, showcase_track
from kartlab.sim.track import Track, generate_track
from kartlab.sim.tracksets import EVAL_SEEDS, eval_track

TRAIN_SEEDS = tuple(range(10))


def track_digest(track: Track) -> str:
    """sha256 of the centerline (rounded to 1 micrometre, -0.0 normalised) plus the width."""
    points = np.round(track.points, 6) + 0.0
    data = points.tobytes() + np.float64(track.width).tobytes()
    return hashlib.sha256(data).hexdigest()[:20]


EXPECTED_EVAL: dict[int, str] = {
    1000000: "c994f1a6ae0b1998043a",
    1000001: "f3a1d86be934fc5eba63",
    1000002: "57504d06f056f75c2383",
    1000003: "5b741fc36d47bd4c6d34",
    1000004: "7470ae40241b3b785132",
    1000005: "2110e7619181bc31df13",
    1000006: "f2d65541d3c4b9bd7546",
    1000007: "b0bf1919010746f02260",
    1000008: "adda7d53060b0458dd38",
    1000009: "0992935a0aa57edbca15",
    1000010: "4e3a5b71b86d15e2bd80",
    1000011: "d20be264e48f2a2647bc",
    1000012: "db36ef6516b8f82bdad9",
    1000013: "418bb2ef276b426dc4fc",
    1000014: "1dbe1cb8da6ce4278f9a",
    1000015: "b4a51add3a3b43979125",
    1000016: "bf78279b850c9f651036",
    1000017: "b4ef2a6e6c24c3a89bc2",
    1000018: "611f03b30653910b0ee9",
    1000019: "01ac19d57b45712cc4d7",
}
EXPECTED_TRAIN: dict[int, str] = {
    0: "043a42d83a17e943568c",
    1: "e56f174334f3c9b3f015",
    2: "e4a3463709b5e94cfcb9",
    3: "0586252d0af251dcab55",
    4: "1a8071658178803d753e",
    5: "ad256701ac9f050f9ad4",
    6: "1502b7203f2958306830",
    7: "22ce2b5315c341c30615",
    8: "281495d0ca4614e841bb",
    9: "c730d4174d0866fcc4a0",
}
EXPECTED_SHOWCASE: dict[str, str] = {
    "harbor": "707b9f3a9e82074f4d97",
    "canyon": "9c101535632ad45bb0c7",
    "plateau": "bbdd472cc123c5fc89a1",
}


def test_evaluation_tracks_do_not_move():
    got = {seed: track_digest(eval_track(i)) for i, seed in enumerate(EVAL_SEEDS)}
    assert got == EXPECTED_EVAL


def test_training_tracks_do_not_move():
    got = {seed: track_digest(generate_track(seed)) for seed in TRAIN_SEEDS}
    assert got == EXPECTED_TRAIN


def test_showcase_tracks_do_not_move():
    got = {key: track_digest(showcase_track(key)) for key in SHOWCASE}
    assert got == EXPECTED_SHOWCASE


def test_digest_is_sensitive_to_geometry_and_width():
    track = generate_track(0)
    moved = Track(track.points + [1e-3, 0.0], track.width, "x")
    wider = Track(track.points, track.width + 1.0, "x")
    assert len({track_digest(track), track_digest(moved), track_digest(wider)}) == 3
