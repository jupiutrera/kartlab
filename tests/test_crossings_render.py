"""The pseudo-3D and top-down views at a level crossing, and the guarantee that every other
track renders exactly as before."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from kartlab.render.camera import BroadcastCamera
from kartlab.render.palette import Ink
from kartlab.render.scene import bake_shadows, place_props
from kartlab.render.texture import build_floor_texture
from kartlab.render.topdown import EDGE, TrackMap
from kartlab.sim.autopilot import Autopilot
from kartlab.sim.kart import DT, KartState
from kartlab.sim.race import Racer
from kartlab.sim.showcase import showcase_track
from kartlab.sim.trackfile import build_track, load_spec

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "tracks" / "ocho.json"


@pytest.fixture(scope="module")
def eight():
    return build_track(load_spec(FILE))


@pytest.fixture(scope="module")
def crossing(eight):
    return eight.crossings[0]


@pytest.fixture(scope="module")
def texture(eight):
    return build_floor_texture(eight, seed=eight.decoration_seed)


def distance_to_track(track, x, y):
    """Distance from points to the nearest part of the whole centerline (both roads)."""
    pts = track.points
    nxt = np.roll(pts, -1, axis=0)
    seg = nxt - pts
    length2 = (seg**2).sum(axis=1)
    px = x[..., None] - pts[:, 0]
    py = y[..., None] - pts[:, 1]
    t = np.clip((px * seg[:, 0] + py * seg[:, 1]) / length2, 0.0, 1.0)
    d = np.hypot(px - t * seg[:, 0], py - t * seg[:, 1])
    return d.min(axis=-1)


# -- the camera -----------------------------------------------------------------------


def target_headings(track, positions, hinted: bool):
    camera = BroadcastCamera()
    headings = []
    for x, y in positions:
        if hinted:
            camera.update(KartState(x, y, 0.0), DT, track)
            headings.append(camera.target_heading)
        else:  # what the camera did before: nearest part of the track, no memory
            headings.append(track.heading_at_index(track.locate(x, y).index))
    return np.array(headings)


def test_the_camera_heading_does_not_snap_at_the_crossing(eight, crossing):
    # The autopilot drives the whole lap, which passes the crossing twice.
    racer, pilot, camera = Racer(eight), Autopilot(eight), BroadcastCamera()
    targets, near = [], []
    for _ in range(60 * 70):
        racer.step(*pilot.act(racer))
        camera.update(racer.kart, DT, eight)
        targets.append(camera.target_heading)
        near.append(math.hypot(racer.kart.x - crossing.x, racer.kart.y - crossing.y) < 20.0)
    assert sum(near) > 20  # it did pass through
    step = np.abs(np.angle(np.exp(1j * np.diff(targets))))
    assert step.max() < math.radians(10.0)  # per frame at 60 fps
    assert camera.pose is not None


def test_without_the_hint_the_heading_would_snap(eight, crossing):
    # A path 3 m off the centerline of one road: right by the crossing the nearest sample
    # belongs to the other road for a few metres, so the unhinted heading flips by 90 degrees.
    s = np.arange(crossing.s_a - 30.0, crossing.s_a + 30.0, 0.3)
    positions = [eight.pose_at(float(v), 3.0)[:2] for v in s]
    flips = np.abs(np.angle(np.exp(1j * np.diff(target_headings(eight, positions, False)))))
    assert flips.max() > math.radians(45.0)
    hinted = np.abs(np.angle(np.exp(1j * np.diff(target_headings(eight, positions, True)))))
    assert hinted.max() < math.radians(2.0)


def test_other_tracks_keep_using_the_plain_locate():
    track = showcase_track("harbor")
    x, y, _ = track.pose_at(200.0, 2.0)
    assert track.locate_following(x, y, 5) == track.locate(x, y)
    assert track.locate_following(x, y, None) == track.locate(x, y)


# -- the floor texture ------------------------------------------------------------------


def window_points(crossing, half=45.0, step=0.25):
    axis = np.arange(-half, half, step)
    gx, gy = np.meshgrid(crossing.x + axis, crossing.y + axis)
    return gx, gy


def sample(texture, gx, gy):
    u, v = texture.to_texel(gx, gy)
    return texture.base[v.astype(int), u.astype(int)]


def test_both_roads_are_plain_asphalt_across_the_crossing(eight, crossing, texture):
    gx, gy = window_points(crossing)
    inside = distance_to_track(eight, gx, gy) < eight.half_width - 1.0
    asphalt = (Ink.ASPHALT, Ink.ASPHALT_LIGHT, Ink.ASPHALT_DARK)
    assert inside.sum() > 3000
    found = sample(texture, gx, gy)[inside]
    assert np.isin(found, asphalt).all()  # no grass, kerb or line on either road


def test_edge_lines_and_kerbs_follow_the_outline_of_both_roads(eight, crossing, texture):
    gx, gy = window_points(crossing)
    d = distance_to_track(eight, gx, gy)
    colours = sample(texture, gx, gy)
    hw = eight.half_width
    lines = colours == Ink.WHITE
    assert lines.sum() > 500
    assert ((d[lines] > hw - 0.7) & (d[lines] < hw)).all()  # only along the union's edge
    kerbs = np.isin(colours, (Ink.KERB_RED, Ink.RED, Ink.MIST))
    assert (d[kerbs] > hw).all()  # a kerb is never painted on the other road's asphalt
    assert not np.isin(colours[d > hw + 4.0], (Ink.ASPHALT, Ink.WHITE)).any()


def test_props_are_never_placed_on_either_road(eight):
    props = place_props(eight, eight.decoration_seed)
    assert len(props) > 50
    xs = np.array([p.x for p in props])
    ys = np.array([p.y for p in props])
    assert distance_to_track(eight, xs, ys).min() > eight.half_width + 1.0


# -- the top-down view -----------------------------------------------------------------


def test_the_top_down_map_has_no_edge_line_across_the_other_road(eight, crossing):
    tmap = TrackMap(eight, 4.0, 20.0)
    heading = eight.heading_at_index(crossing.index_a)

    def pixel(distance, lateral):
        # A point on road A's left edge (lateral = half width), ``distance`` along A.
        x, y, _ = eight.pose_at(crossing.s_a + distance, lateral)
        px, py = tmap.to_px(x, y)
        return tuple(tmap.surface.get_at((round(px), round(py)))[:3])

    assert heading == pytest.approx(eight.heading_at_index(crossing.index_a))
    hw = eight.half_width
    assert pixel(0.0, hw - 0.15) != EDGE[:3]  # at the crossing the other road covers it
    edge_seen = [pixel(d, hw - 0.15) == EDGE[:3] for d in np.arange(20.0, 40.0, 0.5)]
    assert any(edge_seen)  # but the edge line is there once the roads part


# -- nothing changes for tracks without crossings -----------------------------------------

# Texels of each palette index (and props) of the baked floor, measured with the code from
# before level crossings existed (the very same textures were also compared pixel by pixel).
# Compared with a tolerance, as float rounding differs a little between platforms: a change
# in how tracks without crossings are painted would move these by far more.
PINNED_TEXTURES = {
    "harbor": (
        135,
        {
            0: 16335,
            1: 270398,
            2: 10234,
            8: 2101,
            9: 20935,
            11: 2078,
            15: 4814,
            29: 116023,
            30: 1372681,
            31: 1375753,
            32: 72528,
        },
    ),
    "canyon": (
        155,
        {
            0: 17881,
            1: 243923,
            2: 9121,
            8: 2658,
            9: 22131,
            11: 2654,
            15: 6198,
            29: 122309,
            30: 1461183,
            31: 1458757,
            32: 76935,
        },
    ),
}


@pytest.mark.parametrize("key", PINNED_TEXTURES)
def test_textures_of_tracks_without_crossings_are_unchanged(key):
    track = showcase_track(key)
    texture = build_floor_texture(track, 3)
    props = place_props(track, 3)
    bake_shadows(texture, props)
    count, expected = PINNED_TEXTURES[key]
    assert len(props) == count
    counts = np.bincount(texture.base.ravel(), minlength=64)
    assert {int(i) for i in np.flatnonzero(counts)} == set(expected)
    for ink, texels in expected.items():
        assert counts[ink] == pytest.approx(texels, rel=0.005)


def test_the_showcase_file_declares_its_crossing():
    assert load_spec(FILE).crossings == ((0.0, 0.0),)
