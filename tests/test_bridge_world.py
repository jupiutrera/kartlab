"""The bridge world: what the camera sees from the four points of the circuit, the order of
the karts, the cameras that ride up the ramp and duck into the tunnel, and the interface."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pygame
import pytest

from kartlab.render.camera import (
    BROADCAST,
    CHASE,
    BroadcastCamera,
    CameraPose,
    ChaseCamera,
    TopCamera,
)
from kartlab.render.palette import Ink, game_palette
from kartlab.render.scene import KartVisual, WorldRenderer, place_props
from kartlab.render.topdown import TopDownView
from kartlab.show.drive import DriveApp, main
from kartlab.sim.kart import KartState
from kartlab.sim.trackfile import load_track_file

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "tracks" / "ocho_puente.json"
PLAIN = ROOT / "tracks" / "ocho.json"


@pytest.fixture(scope="module")
def bridge():
    return load_track_file(BRIDGE)


@pytest.fixture(scope="module")
def levels(bridge):
    return bridge.levels


@pytest.fixture(scope="module")
def world(bridge):
    pygame.init()
    return WorldRenderer(bridge, CHASE, "day", seed=bridge.decoration_seed)


def render(world, pose, visuals=(), lens=CHASE, variant="day") -> np.ndarray:
    world.set_variant(variant)
    world.set_lens(lens)
    for _ in range(3):  # the sprite bank builds new sprites within a time budget per frame
        world.render(pose, list(visuals), 1.0)
    return world.world_array()


def pose_at(track, s, lateral=0.0, back=0.0, z=0.0) -> CameraPose:
    x, y, heading = track.pose_at(s, lateral)
    return CameraPose(x - back * math.cos(heading), y - back * math.sin(heading), heading, z)


def kart_at(track, s, lateral=0.0) -> KartState:
    return KartState(*track.pose_at(s, lateral))


def test_only_a_track_with_levels_gets_a_bridge_renderer(world):
    assert world.bridge is not None
    assert WorldRenderer(load_track_file(PLAIN), CHASE).bridge is None


# The four points of the circuit that matter, as (arc position, camera height, lens).
POINTS = {
    "climbing": lambda lv: lv.s_over - lv.deck_half - 12.0,
    "on the bridge": lambda lv: lv.s_over,
    "at the tunnel": lambda lv: lv.s_under - 20.0,
    "in the tunnel": lambda lv: lv.s_under + 3.0,
}


@pytest.mark.parametrize("variant", ["day", "sunset", "night"])
@pytest.mark.parametrize("where", list(POINTS))
def test_every_pixel_is_in_the_palette(bridge, levels, world, where, variant):
    s = POINTS[where](levels)
    z = bridge.height_at(s)
    if where == "in the tunnel":
        z = levels.ceiling - 0.4 - CHASE.height
    pose = pose_at(bridge, s, back=8.0 if where != "in the tunnel" else 0.0, z=z)
    shown = KartVisual(kart_at(bridge, s), 0, ground=bridge.height_at(s))
    image = render(world, pose, [shown], variant=variant)
    colours = {tuple(c) for c in image.reshape(-1, 3).tolist()}
    allowed = {tuple(c) for c in game_palette().rgb.tolist()}
    assert colours <= allowed, f"colours outside the 64: {sorted(colours - allowed)[:4]}"


def test_the_viewpoints_look_different_and_nothing_is_nan(bridge, levels, world):
    z_in = levels.ceiling - 0.4 - CHASE.height
    tunnel = render(world, pose_at(bridge, levels.s_under - 20.0))
    height = bridge.height_at(levels.s_over - 20.0)
    on_bridge = render(world, pose_at(bridge, levels.s_over - 20.0, z=height))
    inside = render(world, pose_at(bridge, levels.s_under + 3.0, z=z_in))
    assert not np.array_equal(tunnel, on_bridge)
    assert not np.array_equal(tunnel, inside)
    assert not np.array_equal(on_bridge, inside)
    assert np.isfinite(world.bridge.ring_x).all() and np.isfinite(world.bridge.ring_z).all()
    for image in (tunnel, on_bridge, inside):
        assert image.shape == (180, 320, 3) and image.dtype == np.uint8


def test_the_mouth_is_a_dark_opening_seen_from_the_lower_road(bridge, levels, world):
    image = render(world, pose_at(bridge, levels.s_under - 20.0))
    black = game_palette().rgb[Ink.ASPHALT_DARK]
    dark = (image == black).all(axis=2)
    assert dark[70:95, 100:220].mean() > 0.5  # the opening fills the middle of the view


def test_inside_the_tunnel_the_roof_covers_the_sky(bridge, levels, world):
    z = levels.ceiling - 0.4 - CHASE.height
    inside = render(world, pose_at(bridge, levels.s_under + 3.0, z=z))
    sky = game_palette().rgb[Ink.SKY_TOP]
    assert world.bridge.camera_in_tunnel
    assert not (inside[:20] == sky).all(axis=2).any()  # the top rows are roof, not sky
    render(world, pose_at(bridge, levels.s_under - 30.0))
    assert not world.bridge.camera_in_tunnel


def test_the_shadow_of_the_deck_shows_on_the_floor(bridge, levels, world):
    """The texture under the deck (the tunnel's floor) is not the road's outside it."""
    texture = world.texture
    cx, cy, heading = bridge.pose_at(levels.s_under)
    dx, dy = math.cos(heading), math.sin(heading)

    def patch(distance: float) -> np.ndarray:
        values = []
        for t in np.linspace(-1.0, 1.0, 9):
            for lateral in np.linspace(-3.0, 3.0, 7):
                x = cx + dx * (distance + t) - dy * lateral
                y = cy + dy * (distance + t) + dx * lateral
                values.append(texture.sample(x, y))
        return np.array(values)

    inside, outside = patch(0.0), patch(levels.tunnel_half + 12.0)
    assert (inside == Ink.ASPHALT_DARK).mean() > 0.25 + (outside == Ink.ASPHALT_DARK).mean()
    assert Ink.ASPHALT_DARK == Ink.BLACK  # hence the dither is ASPHALT / ASPHALT_DARK, never black


def test_the_raised_road_is_not_painted_on_the_floor(bridge, levels, world):
    for s in (levels.s_over - 20.0, levels.s_over, levels.s_over + 20.0):
        x, y, _ = bridge.pose_at(s)
        assert world.texture.sample(x, y) not in (Ink.ASPHALT, Ink.ASPHALT_LIGHT)
    x, y, _ = bridge.pose_at(levels.s_over + levels.reach + 6.0)  # past the ramp: plain road
    assert world.texture.sample(x, y) in (Ink.ASPHALT, Ink.ASPHALT_LIGHT, Ink.ASPHALT_DARK)


def test_scenery_keeps_off_both_roads_at_the_crossing(bridge):
    for prop in place_props(bridge, bridge.decoration_seed):
        distance = np.hypot(bridge.points[:, 0] - prop.x, bridge.points[:, 1] - prop.y).min()
        assert distance > bridge.half_width + 1.0


def test_a_kart_on_the_deck_is_drawn_and_one_in_the_tunnel_is_hidden_from_above(
    bridge, levels, world
):
    s = levels.s_over + 6.0
    height = bridge.height_at(s)
    pose = pose_at(bridge, s, back=10.0, z=height)
    empty = render(world, pose)
    with_kart = render(world, pose, [KartVisual(kart_at(bridge, s), 1, ground=height)])
    assert not np.array_equal(empty, with_kart)
    # Seen from above the deck, a kart in the tunnel under it is hidden by the deck.
    under = KartVisual(kart_at(bridge, levels.s_under + 1.0), 2)
    top_pose = pose_at(bridge, levels.s_over - 15.0, z=height)
    one = render(world, top_pose, [], lens=BROADCAST)
    two = render(world, top_pose, [under], lens=BROADCAST)
    assert np.array_equal(one, two)


def test_a_kart_before_the_mouth_is_seen_and_one_inside_it_is_not(bridge, levels, world):
    pose = pose_at(bridge, levels.s_under - 24.0)
    nothing = render(world, pose)
    before = KartVisual(kart_at(bridge, levels.s_under - 12.0), 1)
    inside = KartVisual(kart_at(bridge, levels.s_under + 1.0), 2)
    assert not np.array_equal(nothing, render(world, pose, [before]))
    assert np.array_equal(nothing, render(world, pose, [inside]))


def test_karts_on_the_deck_vanish_when_the_camera_is_in_the_tunnel(bridge, levels, world):
    pose = pose_at(bridge, levels.s_under + 3.0, z=levels.ceiling - 0.4 - CHASE.height)
    on_deck = KartVisual(kart_at(bridge, levels.s_over), 1, ground=levels.deck_height)
    assert np.array_equal(render(world, pose), render(world, pose, [on_deck]))
    below = KartVisual(kart_at(bridge, levels.s_under + 8.0), 2, ground=0.0)
    assert not np.array_equal(render(world, pose), render(world, pose, [below]))


def test_the_hop_still_lifts_the_sprite_on_the_deck(bridge, levels, world):
    s = levels.s_over
    pose = pose_at(bridge, s - 6.0, back=4.0, z=levels.deck_height)
    down = render(world, pose, [KartVisual(kart_at(bridge, s), 1, ground=levels.deck_height)])
    up = render(
        world, pose, [KartVisual(kart_at(bridge, s), 1, lift=0.5, ground=levels.deck_height)]
    )
    assert not np.array_equal(down, up)


def test_tyres_leave_no_marks_on_the_floor_from_the_deck(bridge, levels):
    pygame.init()
    fresh = WorldRenderer(bridge, CHASE, "day", seed=bridge.decoration_seed)
    kart = kart_at(bridge, levels.s_over)
    kart.vx, kart.vy = -4.0 * math.sin(kart.heading), 4.0 * math.cos(kart.heading)  # sliding
    before = fresh.texture.base.copy()
    fresh.update_effects([KartVisual(kart, 0, ground=levels.deck_height)], 1 / 60)
    assert np.array_equal(before, fresh.texture.base)
    for _ in range(10):  # on the ground the same slide leaves marks
        kart.x += 0.11 * math.cos(kart.heading)
        kart.y += 0.11 * math.sin(kart.heading)
        fresh.update_effects([KartVisual(kart, 0, ground=0.0)], 1 / 60)
    assert not np.array_equal(before, fresh.texture.base)


# -- cameras ----------------------------------------------------------------------------------


def run_camera(camera, bridge, s, frames=240):
    kart = kart_at(bridge, s)
    pose = None
    for _ in range(frames):
        pose = camera.update(kart, 1 / 60, bridge)
    return pose


def test_the_chase_camera_rides_up_to_the_deck(bridge, levels):
    pose = run_camera(ChaseCamera(), bridge, levels.s_over)
    assert pose.z == pytest.approx(levels.deck_height, abs=0.05)
    assert run_camera(ChaseCamera(), bridge, levels.s_over - levels.reach - 20.0).z == 0.0


def test_the_chase_camera_ducks_in_the_tunnel_and_stays_under_the_roof(bridge, levels):
    pose = run_camera(ChaseCamera(), bridge, levels.s_under + 2.0)
    assert CHASE.height + pose.z < levels.ceiling
    assert pose.z < 0.0
    assert run_camera(ChaseCamera(), bridge, levels.s_under - 40.0).z == 0.0


def test_the_broadcast_camera_rides_but_does_not_duck_and_the_top_one_stays(bridge, levels):
    on_deck = run_camera(BroadcastCamera(), bridge, levels.s_over)
    assert on_deck.z == pytest.approx(levels.deck_height, abs=0.05)
    assert run_camera(BroadcastCamera(), bridge, levels.s_under + 2.0).z == 0.0
    assert run_camera(TopCamera(), bridge, levels.s_over).z == 0.0


def test_the_camera_height_is_continuous_driving_over_the_bridge(bridge, levels):
    camera = ChaseCamera()
    previous, worst = None, 0.0
    for k in range(int(2 * (levels.reach + 20) / 0.4)):  # 24 m/s, one frame each 0.4 m
        s = levels.s_over - levels.reach - 20.0 + 0.4 * k
        pose = camera.update(kart_at(bridge, s), 1 / 60, bridge)
        if previous is not None:
            worst = max(worst, abs(pose.z - previous))
        previous = pose.z
    assert worst < 0.1


def test_the_camera_height_is_continuous_driving_through_the_tunnel(bridge, levels):
    camera = ChaseCamera()
    previous, worst, lowest = None, 0.0, 0.0
    for k in range(int(2 * (levels.tunnel_half + 40) / 0.4)):
        s = levels.s_under - levels.tunnel_half - 40.0 + 0.4 * k
        pose = camera.update(kart_at(bridge, s), 1 / 60, bridge)
        if previous is not None:
            worst = max(worst, abs(pose.z - previous))
        previous, lowest = pose.z, min(lowest, pose.z)
    assert worst < 0.15 and lowest < -0.5


# -- the interface ------------------------------------------------------------------------------


def test_the_topdown_map_draws_the_bridge_and_hides_karts_in_the_tunnel(bridge, levels):
    pygame.init()
    view = TopDownView(bridge, (640, 360))
    target = pygame.Surface((640, 360))
    here = bridge.pose_at(levels.s_over)
    on_bridge = (KartState(*here), (255, 156, 44), 0.0, levels.deck_height)
    view.draw(target, [on_bridge], here[:2], True)
    alone = pygame.surfarray.array3d(target).copy()
    under = (KartState(*bridge.pose_at(levels.s_under)), (1, 2, 3), 0.0, 0.0)
    view.draw(target, [on_bridge, under], here[:2], True)
    assert np.array_equal(alone, pygame.surfarray.array3d(target))  # the one under is hidden
    far = (KartState(*bridge.pose_at(levels.s_under - 60.0)), (1, 2, 3), 0.0, 0.0)
    view.draw(target, [on_bridge, far], bridge.pose_at(levels.s_under - 60.0)[:2], True)
    assert not np.array_equal(alone, pygame.surfarray.array3d(target))


def test_the_ghost_reads_the_height_of_its_own_branch(bridge, levels, tmp_path):
    pygame.init()
    app = DriveApp(bridge, scale=1, realtime=False, records_dir=tmp_path)
    seen = [
        app.ghost_ground(KartState(*bridge.pose_at(levels.s_over - 60.0 + k))) for k in range(120)
    ]
    assert max(seen) == pytest.approx(levels.deck_height, abs=0.01)
    assert all(abs(b - a) < 0.5 for a, b in zip(seen, seen[1:], strict=False))  # no jumps
    app._ghost_hint = bridge.index_at(levels.s_under - 10.0)  # coming through the tunnel
    for k in range(0, 21, 5):  # right at the crossing it keeps to its own road
        assert app.ghost_ground(KartState(*bridge.pose_at(levels.s_under - 10.0 + k))) == 0.0
    pygame.quit()


def test_drive_starts_where_at_and_lateral_say(bridge, levels, tmp_path):
    shot = tmp_path / "puente.png"
    main(
        [
            "--track-file", str(BRIDGE), "--at", f"{levels.s_over:.1f}", "--lateral", "1.5",
            "--frames", "3", "--scale", "1", "--screenshot", str(shot),
        ]
    )  # fmt: skip
    assert shot.exists() and shot.stat().st_size > 1000
    pygame.init()
    app = DriveApp(bridge, scale=1, realtime=False, start=(levels.s_over, 1.5))
    assert app.racer.level_z == pytest.approx(levels.deck_height)
    assert app.racer.position.lateral == pytest.approx(1.5, abs=0.05)
    app.reset()
    assert app.racer.level_z == pytest.approx(levels.deck_height)
    pygame.quit()
