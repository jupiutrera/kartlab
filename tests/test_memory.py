"""Peak memory of the heavy one-off computations: loading a track in the viewer and
validating the longest track a file may declare.

A float64 matrix between every texel (or every sample) and every centerline point is over
a gigabyte on a big track, enough for a MemoryError on a busy laptop. These tests measure
the peak with ``tracemalloc`` (numpy reports its buffers to it) and fail if one of those
computations goes back to allocating that way.
"""

from __future__ import annotations

import dataclasses
import tracemalloc
from pathlib import Path

import pygame
import pytest

from kartlab.render.camera import CHASE
from kartlab.render.scene import WorldRenderer
from kartlab.render.texture import build_floor_texture
from kartlab.render.topdown import TrackMap
from kartlab.sim.trackfile import TrackSpec, build_track, load_spec, validate_spec

ROOT = Path(__file__).resolve().parents[1]
MIB = 2**20
LOAD_LIMIT = 300 * MIB  # building the texture and scene of a track
CHECK_LIMIT = 150 * MIB  # one validation or one top-down map


def peak(function) -> tuple[object, int]:
    tracemalloc.start()
    try:
        result = function()
        return result, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def enlarged(spec: TrackSpec, factor: float) -> TrackSpec:
    """The same layout scaled up around the origin (the longest a file can be is 2500 m)."""
    return dataclasses.replace(
        spec,
        corners=tuple((x * factor, y * factor, r * factor) for x, y, r in spec.corners),
        crossings=tuple((x * factor, y * factor) for x, y in spec.crossings),
        start=None if spec.start is None else (spec.start[0] * factor, spec.start[1] * factor),
    )


@pytest.fixture(scope="module")
def eight_spec():
    return load_spec(ROOT / "tracks" / "ocho.json")


def test_building_the_texture_of_the_figure_eight_stays_small(eight_spec):
    track = build_track(eight_spec)
    _, used = peak(lambda: build_floor_texture(track, seed=track.decoration_seed))
    assert used < LOAD_LIMIT


def test_loading_the_figure_eight_in_the_viewer_stays_small(eight_spec):
    pygame.display.init()
    pygame.display.set_mode((64, 64))
    track = build_track(eight_spec)
    _, used = peak(lambda: WorldRenderer(track, CHASE))  # texture, scenery, shadows, sprites
    assert used < LOAD_LIMIT


def test_the_top_down_map_of_the_longest_figure_eight_stays_small(eight_spec):
    spec = enlarged(eight_spec, 1.95)
    assert validate_spec(spec).ok and validate_spec(spec).length > 2400.0
    track = build_track(spec)
    _, used = peak(lambda: TrackMap(track, 4.0, 20.0))
    assert used < CHECK_LIMIT


@pytest.mark.parametrize("name", ["ocho", "ejemplo"])
def test_validating_the_longest_allowed_track_stays_small(name):
    spec = enlarged(load_spec(ROOT / "tracks" / f"{name}.json"), 1.95)
    report, used = peak(lambda: validate_spec(spec))
    assert report.ok and report.length > 2400.0  # near MAX_LENGTH: the worst case
    assert used < CHECK_LIMIT
