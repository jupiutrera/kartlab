import math

import numpy as np
import pygame
import pytest

from kartlab.render.camera import CHASE, CameraPose, ChaseCamera
from kartlab.render.palette import FOG_COLOR, TRANSPARENT, Ink, game_palette
from kartlab.render.scene import KartVisual, WorldRenderer, place_props
from kartlab.render.sprites import SpriteBank, kart_voxels
from kartlab.render.texture import build_floor_texture
from kartlab.sim.kart import DT, KartState
from kartlab.sim.showcase import showcase_track


@pytest.fixture(scope="module")
def track():
    return showcase_track("harbor")


@pytest.fixture(scope="module")
def texture(track):
    return build_floor_texture(track)


@pytest.fixture(scope="module")
def world(track):
    pygame.display.init()
    pygame.display.set_mode((64, 64))
    return WorldRenderer(track, CHASE)


# -- sprites ---------------------------------------------------------------------------


def test_kart_model_has_body_wheels_and_driver():
    voxels = kart_voxels()
    colours = set(np.unique(voxels)) - {TRANSPARENT}
    assert {Ink.BLACK, Ink.WHITE, Ink.NAVY} <= colours


def test_kart_sprites_grow_with_scale_and_change_with_angle(world):
    bank = SpriteBank(game_palette())
    small, _ = bank.kart(0, 0.0, math.radians(22), 0.4)
    big, _ = bank.kart(0, 0.0, math.radians(22), 2.0)
    assert big.get_width() > 3 * small.get_width()
    side, _ = bank.kart(0, math.pi / 2, math.radians(22), 1.0)
    back, _ = bank.kart(0, 0.0, math.radians(22), 1.0)
    assert pygame.image.tobytes(side, "RGB") != pygame.image.tobytes(back, "RGB")


def test_scenery_stays_off_the_track(track):
    props = place_props(track)
    assert len(props) > 50
    for prop in props:
        assert abs(track.locate(prop.x, prop.y).lateral) > track.half_width


# -- the whole frame -------------------------------------------------------------------


def test_world_frame_draws_sky_floor_and_kart(track, world):
    kart = KartState(*track.pose_at(100.0))
    camera = ChaseCamera()
    for _ in range(60):
        pose = camera.update(kart, DT)
    before = world.world.copy()
    timings = world.render(pose, [KartVisual(kart, 0, label="Manual")], 0.0)
    assert set(timings) == {"cielo", "suelo", "copia", "sprites"}
    pixels = world.world_array()
    palette = game_palette()
    assert tuple(pixels[0, 160]) in {tuple(palette.rgb[i]) for i in range(64)}
    kart_rgb = tuple(palette.rgb[Ink.ORANGE])
    assert (pixels[120:170, 120:200] == kart_rgb).all(axis=-1).any()
    assert pygame.image.tobytes(before, "RGB") != pygame.image.tobytes(world.world, "RGB")


@pytest.mark.parametrize("variant", ["sunset", "night"])
def test_variants_change_the_colours_of_the_same_frame(track, world, variant):
    pose = CameraPose(*track.pose_at(50.0))
    world.set_variant("day")
    world.render(pose, [], 0.0)
    day = world.world_array()
    world.set_variant(variant)
    world.render(pose, [], 0.0)
    other = world.world_array()
    world.set_variant("day")
    assert (day != other).any(axis=-1).mean() > 0.5
    assert tuple(other[0, 0]) != tuple(day[0, 0])


def test_skidding_leaves_marks_on_the_asphalt(track):
    world = WorldRenderer(track, CHASE)
    x, y, heading = track.pose_at(400.0)
    kart = KartState(x, y, heading, vx=-math.sin(heading) * 5, vy=math.cos(heading) * 5)
    world.update_effects([KartVisual(kart)], DT)
    rear_x, rear_y = x - math.cos(heading) * 0.7, y - math.sin(heading) * 0.7
    marks = [
        world.texture.sample(rear_x + dx, rear_y + dy)
        for dx in (-0.6, 0, 0.6)
        for dy in (-0.6, 0, 0.6)
    ]
    assert Ink.ASPHALT_DARK in marks


def test_fog_colour_is_the_sky_at_the_horizon(world):
    assert FOG_COLOR["day"] == Ink.SKY_LOW
