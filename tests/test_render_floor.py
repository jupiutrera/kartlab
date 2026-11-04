import math

import numpy as np
import pytest

from kartlab.render.camera import BROADCAST, CHASE, TOP, BroadcastCamera, CameraPose, ChaseCamera
from kartlab.render.floor import FloorRenderer, Projection
from kartlab.render.palette import Ink, game_palette
from kartlab.render.sky import Sky
from kartlab.render.texture import build_floor_texture
from kartlab.sim.kart import DT, KartState
from kartlab.sim.showcase import showcase_track


@pytest.fixture(scope="module")
def track():
    return showcase_track("harbor")


@pytest.fixture(scope="module")
def texture(track):
    return build_floor_texture(track)


# -- floor texture ----------------------------------------------------------------------


def test_texture_paints_the_track(track, texture):
    hw = track.half_width
    assert texture.sample(*track.pose_at(300.0, 1.0)[:2]) in (
        Ink.ASPHALT,
        Ink.ASPHALT_LIGHT,
        Ink.ASPHALT_DARK,
    )
    far = texture.sample(*track.pose_at(300.0, hw + 30)[:2])
    assert far in (Ink.GRASS, Ink.GRASS_LIGHT, Ink.GRASS_DARK, Ink.LEAF_LIGHT)
    line = [
        texture.sample(*track.pose_at(0.2, lat)[:2]) for lat in np.linspace(-hw + 0.5, hw - 0.5, 9)
    ]
    assert {Ink.BLACK, Ink.WHITE} <= set(line)  # chequered start line
    corner = int(np.argmax(np.abs(track.curvature)))
    side = -math.copysign(1.0, track.curvature[corner])
    kerb = texture.sample(*track.pose_at(float(track.s[corner]), side * (hw + 0.4))[:2])
    assert kerb in (Ink.KERB_RED, Ink.WHITE, Ink.RED, Ink.MIST)


def test_mipmaps_halve_and_stay_in_the_palette(texture):
    shapes = [lvl.shape for lvl in texture.levels]
    assert len(shapes) > 4
    for (h0, w0), (h1, w1) in zip(shapes, shapes[1:], strict=False):
        assert h1 == h0 // 2 and w1 == w0 // 2
    assert all(lvl.max() < 64 for lvl in texture.levels)


def test_stamp_paints_a_disc(track):
    tex = build_floor_texture(track)
    x, y, _ = track.pose_at(100.0)
    tex.stamp(x, y, 0.5, Ink.PINK)
    assert tex.sample(x, y) == Ink.PINK


# -- perspective floor ------------------------------------------------------------------


@pytest.mark.parametrize("lens", [CHASE, BROADCAST, TOP], ids=lambda lens: lens.name)
def test_floor_pixels_show_the_ground_points_that_project_onto_them(track, texture, lens):
    proj = Projection(320, 180, lens)
    floor = FloorRenderer(proj, texture)
    x, y, heading = track.pose_at(200.0)
    pose = CameraPose(x - 8 * math.cos(heading), y - 8 * math.sin(heading), heading)
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(200):
        row = int(rng.integers(floor.first_row, 180))
        col = int(rng.integers(0, 320))
        i = row - floor.first_row
        ahead, lat = floor.ahead_t[i] / texture.tpm, floor.lat_t[i, col] / texture.tpm
        if ahead > 80:
            continue
        wx = pose.x + math.cos(heading) * ahead + math.sin(heading) * lat
        wy = pose.y + math.sin(heading) * ahead - math.cos(heading) * lat
        sx, sy, depth = proj.project(pose, wx, wy)
        assert depth > 0
        assert abs(sx - (col + 0.5)) < 0.05 and abs(sy - (row + 0.5)) < 0.05
        checked += 1
    assert checked > 50


def test_floor_render_writes_palette_indices_below_the_horizon(texture):
    proj = Projection(320, 180, CHASE)
    floor = FloorRenderer(proj, texture)
    buf = np.full((180, 320), 200, dtype=np.uint8)
    floor.render(buf, CameraPose(0.0, 0.0, 0.3), game_palette().fog_table("day"))
    assert np.all(buf[: floor.first_row] == 200)
    assert buf[floor.first_row :].max() < 64


def test_light_mode_repeats_rows(texture):
    proj = Projection(320, 180, CHASE)
    floor = FloorRenderer(proj, texture, row_step=2)
    buf = np.zeros((180, 320), dtype=np.uint8)
    floor.render(buf, CameraPose(10.0, 5.0, 1.0), game_palette().fog_table("day"))
    first = floor.first_row
    assert np.array_equal(
        buf[first + 1 : 180 : 2], buf[first:179:2][: len(buf[first + 1 : 180 : 2])]
    )


def test_sky_meets_the_floor_without_a_gap(texture):
    proj = Projection(320, 180, CHASE)
    floor = FloorRenderer(proj, texture)
    sky = Sky(proj, game_palette(), "day", floor.first_row)
    buf = np.full((180, 320), 200, dtype=np.uint8)
    sky.render(buf, 0.0)
    floor.render(buf, CameraPose(0.0, 0.0, 0.0), game_palette().fog_table("day"))
    assert buf.max() < 64  # every row painted by one or the other


# -- cameras ---------------------------------------------------------------------------


def test_chase_camera_never_turns_faster_than_its_limit():
    camera = ChaseCamera()
    kart = KartState(0.0, 0.0, 0.0)
    camera.update(kart, DT)
    kart.heading = math.pi * 0.9  # sudden spin
    previous = camera.pose.heading
    for _ in range(30):
        pose = camera.update(kart, DT)
        assert abs(math.remainder(pose.heading - previous, math.tau)) <= camera.max_turn * DT + 1e-9
        previous = pose.heading


def test_broadcast_camera_follows_the_track_not_the_kart(track):
    camera = BroadcastCamera()
    x, y, heading = track.pose_at(300.0)
    for _ in range(600):
        pose = camera.update(KartState(x, y, heading + 2.5), DT, track)  # kart facing backwards
    assert abs(math.remainder(pose.heading - heading, math.tau)) < 0.05
