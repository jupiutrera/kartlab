"""Polygons that reach far off screen are cut to it: faster, and not a pixel different."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pygame
import pytest

from kartlab.render import bridge as bridge_module
from kartlab.render.bridge import _fit
from kartlab.render.camera import BROADCAST, CHASE, CameraPose
from kartlab.render.scene import KartVisual, WorldRenderer
from kartlab.render.sprites import SpriteBank
from kartlab.sim.kart import KartState
from kartlab.sim.trackfile import load_track_file

BRIDGE = Path(__file__).resolve().parents[1] / "tracks" / "ocho_puente.json"


def test_a_polygon_inside_the_margin_comes_back_untouched():
    points = [(10.0, 10.0), (300.0, 20.0), (310.0, 170.0), (-50.0, 160.0)]
    assert _fit(points, 320, 180) is points


def test_a_polygon_far_off_screen_is_cut_to_it():
    points = [(-5000.0, 90.0), (5000.0, 90.0), (5000.0, 100.0), (-5000.0, 100.0)]
    cut = _fit(points, 320, 180)
    assert len(cut) >= 4
    assert all(-1.0 - 1e-9 <= x <= 321.0 + 1e-9 for x, _ in cut)
    assert all(90.0 - 1e-9 <= y <= 100.0 + 1e-9 for _, y in cut)


def test_a_polygon_wholly_off_screen_vanishes():
    far = [(1000.0, 1000.0), (2000.0, 1000.0), (2000.0, 2000.0), (1000.0, 2000.0)]
    assert _fit(far, 320, 180) == []


@pytest.fixture(scope="module", autouse=True)
def unlimited_sprite_budget():
    """The sprite bank draws new sprites within a time budget per frame: on a busy machine a
    sprite could be missing from a frame. Tests compare frames, so the budget is lifted."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(SpriteBank, "BUDGET_MS", math.inf)
        yield


@pytest.fixture(scope="module")
def world():
    pygame.init()
    track = load_track_file(BRIDGE)
    return WorldRenderer(track, CHASE, "day", seed=track.decoration_seed)


def frame(world, pose, lens, visuals=()):
    world.set_lens(lens)
    world.render(pose, list(visuals), 1.0)
    return world.world_array()


def test_cutting_polygons_does_not_change_a_pixel(world, monkeypatch):
    track = world.track
    levels = track.levels
    spots = [
        (levels.s_over - 10.0, 3.5, BROADCAST),
        (levels.s_over - 40.0, 0.0, BROADCAST),
        (levels.s_over - 2.0, 3.5, CHASE),
        (levels.s_over - 30.0, 0.0, CHASE),
        (levels.s_under - 6.0, 0.0, CHASE),
        (levels.s_under - 14.0, 0.0, CHASE),
    ]
    for s, z, lens in spots:
        x, y, heading = track.pose_at(s)
        pose = CameraPose(x - 8 * math.cos(heading), y - 8 * math.sin(heading), heading, z)
        shown = [KartVisual(KartState(x, y, heading), 0)]
        monkeypatch.setattr(bridge_module, "FIT_MARGIN", 1.0e12)
        plain = frame(world, pose, lens, shown)
        monkeypatch.setattr(bridge_module, "FIT_MARGIN", 200.0)
        fitted = frame(world, pose, lens, shown)
        assert np.array_equal(plain, fitted), f"differs at s={s:.0f}, lens={lens.name}"
