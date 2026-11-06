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


# -- the hop ("brinco"): lift, shadow and dust ------------------------------------------------


def _kart_rows(world, pose, kart, lift):
    """Rows (top, bottom) where the orange of the kart body shows in a frame."""
    world.render(pose, [KartVisual(kart, 0, lift=lift)], 0.0)
    pixels = world.world_array()
    orange = (pixels == tuple(game_palette().rgb[Ink.ORANGE])).all(axis=-1)
    rows = np.flatnonzero(orange.any(axis=1))
    assert rows.size, "the kart is not in the frame"
    return rows.min(), rows.max()


def test_lift_raises_the_sprite_and_the_shadow_stays_on_the_floor(track):
    world = WorldRenderer(track, CHASE)
    kart = KartState(*track.pose_at(100.0))
    camera = ChaseCamera()
    for _ in range(60):
        pose = camera.update(kart, DT)
    ground_top, ground_bottom = _kart_rows(world, pose, kart, 0.0)
    air_top, air_bottom = _kart_rows(world, pose, kart, 0.5)
    assert air_top < ground_top - 2 and air_bottom < ground_bottom - 2  # it rises
    # The shadow is blitted at the same place in both frames, but smaller and fainter up there.
    calls = []
    real = world.sprites.shadow

    def spy(width, faint=False):
        calls.append((width, faint))
        return real(width, faint)

    world.sprites.shadow = spy
    world.render(pose, [KartVisual(kart, 0)], 0.0)
    world.render(pose, [KartVisual(kart, 0, lift=0.5)], 0.0)
    (ground_width, ground_faint), (air_width, air_faint) = calls
    assert not ground_faint and air_faint and air_width < ground_width
    _, sy, _ = world.projection.project(pose, kart.x, kart.y)
    black = tuple(game_palette().rgb[Ink.BLACK])
    band = world.world_array()[int(sy) - 1 : int(sy) + 2]
    empty = WorldRenderer(track, CHASE)
    empty.render(pose, [], 0.0)
    bare = empty.world_array()[int(sy) - 1 : int(sy) + 2]
    assert (band == black).all(axis=-1).sum() > (bare == black).all(axis=-1).sum()  # shadow there


def test_lift_zero_is_the_frame_of_always(track):
    kart = KartState(*track.pose_at(100.0))
    pose = CameraPose(kart.x - 6.0, kart.y, 0.0)
    world = WorldRenderer(track, CHASE)
    plain = b""
    for _ in range(20):  # the sprite cache fills within a time budget: wait for a stable frame
        world.render(pose, [KartVisual(kart, 0)], 0.0)
        frame = pygame.image.tobytes(world.world, "RGB")
        if frame == plain:
            break
        plain = frame
    else:
        raise AssertionError("the frame never settled")
    world.render(pose, [KartVisual(kart, 0, lift=0.0)], 0.0)
    assert pygame.image.tobytes(world.world, "RGB") == plain


@pytest.mark.parametrize("grass", [False, True])
def test_landing_puffs_dust_once(track, grass):
    from kartlab.sim.drift import DriftState

    world = WorldRenderer(track, CHASE)
    kart = KartState(*track.pose_at(100.0))
    drift = DriftState()
    visual = KartVisual(kart, 0, on_grass=grass, drift=drift)
    ink = Ink.BROWN if grass else Ink.MIST
    world.update_effects([visual], DT)
    assert not [p for p in world.particles.items if p[7] == ink]  # nothing without a landing
    drift.land_id += 1
    world.update_effects([visual], DT)
    assert len([p for p in world.particles.items if p[7] == ink]) == 6
    world.update_effects([visual], DT)  # the same landing does not puff again
    assert len([p for p in world.particles.items if p[7] == ink]) == 6


def test_a_kart_in_the_air_leaves_no_marks(track):
    world = WorldRenderer(track, CHASE)
    x, y, heading = track.pose_at(400.0)
    kart = KartState(x, y, heading, vx=-math.sin(heading) * 5, vy=math.cos(heading) * 5)
    world.update_effects([KartVisual(kart, lift=0.3)], DT)
    rear_x, rear_y = x - math.cos(heading) * 0.7, y - math.sin(heading) * 0.7
    marks = [
        world.texture.sample(rear_x + dx, rear_y + dy)
        for dx in (-0.6, 0, 0.6)
        for dy in (-0.6, 0, 0.6)
    ]
    assert Ink.ASPHALT_DARK not in marks
