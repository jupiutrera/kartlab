"""Rendering of the bridge and the tunnel: camera height, floor scale, geometry and colours."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from kartlab.render.camera import BROADCAST, CHASE, CameraPose, ChaseCamera
from kartlab.render.floor import FloorRenderer, Projection
from kartlab.render.palette import game_palette
from kartlab.render.texture import build_floor_texture
from kartlab.sim.kart import KartState
from kartlab.sim.trackfile import load_track_file

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "tracks" / "ocho_puente.json"
PLAIN = ROOT / "tracks" / "ocho.json"

# sha1 of the floor of the flat eight, seen from s = 100 m with z = 0, as drawn before cameras
# had a height: the floor must not change by a single index.
FLOOR_SHA1 = {
    "chase": "d0dcff589ac2eea2b5020caee95aebb2e89cba45",
    "broadcast": "47c4147f797a24f6e0711a019f340407c92b7a09",
}


@pytest.fixture(scope="module")
def plain():
    return load_track_file(PLAIN)


@pytest.fixture(scope="module")
def plain_texture(plain):
    return build_floor_texture(plain, 0)


def floor_of(texture, lens, pose) -> np.ndarray:
    out = np.zeros((180, 320), np.uint8)
    renderer = FloorRenderer(Projection(320, 180, lens), texture)
    renderer.render(out, pose, game_palette().fog_table("day"))
    return out


@pytest.mark.parametrize("lens", [CHASE, BROADCAST], ids=lambda lens: lens.name)
def test_the_floor_at_z_zero_is_the_floor_it_always_was(plain, plain_texture, lens):
    x, y, heading = plain.pose_at(100.0)
    out = floor_of(plain_texture, lens, CameraPose(x, y, heading))
    assert hashlib.sha1(out.tobytes()).hexdigest() == FLOOR_SHA1[lens.name]
    assert np.array_equal(out, floor_of(plain_texture, lens, CameraPose(x, y, heading, 0.0)))


@pytest.mark.parametrize("z", [-1.0, 0.8, 3.5, 7.0])
def test_a_camera_off_its_lens_height_sees_a_scaled_floor(plain, plain_texture, z):
    x, y, heading = plain.pose_at(100.0)
    flat = floor_of(plain_texture, CHASE, CameraPose(x, y, heading))
    out = floor_of(plain_texture, CHASE, CameraPose(x, y, heading, z))
    assert out.shape == flat.shape and not np.array_equal(out, flat)
    assert out.max() < 64  # palette indices, nothing else


def test_projection_uses_the_camera_height(plain):
    projection = Projection(320, 180, CHASE)
    x, y, heading = plain.pose_at(100.0)
    ahead = (x + 10 * np.cos(heading), y + 10 * np.sin(heading))
    low = projection.project(CameraPose(x, y, heading), *ahead, 0.0)
    high = projection.project(CameraPose(x, y, heading, 3.5), *ahead, 0.0)
    assert high[1] > low[1]  # seen from higher up, the floor point drops towards the bottom
    same = projection.project(CameraPose(x, y, heading, 3.5), *ahead, 3.5)
    assert same[1] == pytest.approx(low[1])  # a point at the camera's own extra height


def test_a_plain_track_never_moves_the_camera_up(plain):
    camera = ChaseCamera()
    kart = KartState(*plain.pose_at(50.0))
    for _ in range(30):
        pose = camera.update(kart, 1 / 60, plain)
    assert pose.z == 0.0
