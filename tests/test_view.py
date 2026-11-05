import math

import pygame
import pytest

from kartlab.render.palette import game_palette
from kartlab.render.scene import KartVisual
from kartlab.render.sprites import SpriteBank
from kartlab.render.view import GameView, HudInfo, Smoothed, scale_for
from kartlab.sim.kart import DT, KartState
from kartlab.sim.showcase import showcase_track


@pytest.fixture(scope="module")
def track():
    return showcase_track("plateau")


@pytest.fixture(scope="module")
def view(track):
    pygame.display.init()
    pygame.display.set_mode((64, 64))
    return GameView(track, (1366, 768))


@pytest.mark.parametrize(
    "window, scale", [((1280, 720), 4), ((1366, 768), 4), ((1920, 1080), 6), ((800, 600), 2)]
)
def test_world_uses_the_largest_integer_scale(window, scale):
    assert scale_for(window) == scale


def test_world_is_centred_with_black_bars(view):
    assert view.world_rect.size == (1280, 720)
    assert view.world_rect.topleft == (43, 24)


def test_frame_draws_world_and_hud(track, view):
    screen = pygame.Surface((1366, 768))
    kart = KartState(*track.pose_at(30.0))
    visual = KartVisual(kart, 0)
    pose = view.update([visual], DT)
    info = HudInfo(laps=1, lap_time=12.3, best_lap=55.0, speed=20.0, label="Modelo 1,2M pasos")
    view.draw(screen, pose, [visual], info, DT)
    assert screen.get_at((5, 5))[:3] == (0, 0, 0)  # letterbox
    assert set(view.last_timings) >= {"suelo", "sprites", "escalado", "HUD"}
    assert screen.get_at((683, 300))[:3] != (0, 0, 0)


def test_low_fps_switches_to_light_mode(track):
    view = GameView(track, (640, 360))
    screen = pygame.Surface((640, 360))
    visual = KartVisual(KartState(*track.pose_at(10.0)))
    pose = view.update([visual], DT)
    for _ in range(130):
        view.draw(screen, pose, [visual], HudInfo(), DT, fps=22.0)
    assert view.world.light_mode and view.target_fps == 30


def test_cameras_and_variants_cycle(view):
    assert view.cycle_camera() == "broadcast"
    assert view.cycle_camera() == "top"
    assert view.cycle_camera() == "chase"
    assert view.cycle_variant() == "sunset"
    assert view.cycle_variant() == "night"
    assert view.cycle_variant() == "day"


def test_displayed_numbers_ease_towards_their_value():
    speed = Smoothed(rate=10.0)
    assert speed.update(50.0, DT) == 50.0
    eased = speed.update(100.0, DT)
    assert 50.0 < eased < 100.0


def test_sprite_budget_reuses_a_neighbour_instead_of_drawing(view):
    bank = SpriteBank(game_palette())
    bank.kart(2, 0.0, math.radians(22), 1.0)
    bank._spent = bank.BUDGET_MS  # this frame's budget is gone
    cached = len(bank._karts)
    surface, _ = bank.kart(2, 0.0, math.radians(22), 1.12)  # next size up, not cached
    assert len(bank._karts) == cached and surface is not None
    bank.begin_frame()
    bank.kart(2, 0.0, math.radians(22), 1.12)
    assert len(bank._karts) == cached + 1


def test_fixed_time_makes_the_hud_clock_frame_exact(track):
    view = GameView(track, (640, 360))
    view.fixed_time = 5.0
    assert view.now() == 5.0
    view.notify("Aviso")
    assert view._notice == ("Aviso", 5.0)
    view.fixed_time = None
    assert view.now() < 5.0  # back to the wall clock


def test_eight_kart_colours_are_distinct():
    from kartlab.render.palette import KART_RAMPS

    assert len(KART_RAMPS) == 8
    assert len({ramp[1] for ramp in KART_RAMPS}) == 8


def _render_with_effects(track, seed: int) -> bytes:
    """A kart skidding on the grass with the throttle down: particles and marks every frame."""
    view = GameView(track, (640, 360), "chase", seed=seed)
    screen = pygame.Surface((640, 360))
    x, y, heading = track.pose_at(30.0)
    for frame in range(45):
        view.fixed_time = 1.0 + frame * DT
        kart = KartState(x + frame * 0.1, y, heading, vx=6.0 * math.cos(heading), vy=6.0)
        visual = KartVisual(kart, 0, on_grass=True, throttle=1.0)
        pose = view.update([visual], DT)
        view.draw(screen, pose, [visual], HudInfo(), DT)
    return pygame.image.tobytes(screen, "RGB")


def test_same_seed_gives_identical_frames_with_effects(track):
    first = _render_with_effects(track, seed=5)
    assert _render_with_effects(track, seed=5) == first


def test_effects_depend_on_the_seed(track):
    assert _render_with_effects(track, seed=5) != _render_with_effects(track, seed=6)


def test_hud_can_be_left_out(track, view):
    kart = KartState(*track.pose_at(30.0))
    visual = KartVisual(kart, 0)
    pose = view.update([visual], DT)
    with_hud, without = pygame.Surface((1366, 768)), pygame.Surface((1366, 768))
    view.draw(with_hud, pose, [visual], HudInfo(laps=1, lap_time=12.3), DT)
    view.draw(without, pose, [visual], HudInfo(laps=1, lap_time=12.3), DT, hud=False)
    corner = (60, 40, 150, 60)  # where the lap panel sits
    assert pygame.image.tobytes(with_hud.subsurface(corner), "RGB") != pygame.image.tobytes(
        without.subsurface(corner), "RGB"
    )
    assert without.get_at((683, 300))[:3] != (0, 0, 0)  # the world is still drawn


def test_forgetting_fps_samples_keeps_the_light_mode_off(track):
    view = GameView(track, (640, 360))
    screen = pygame.Surface((640, 360))
    visual = KartVisual(KartState(*track.pose_at(10.0)))
    pose = view.update([visual], DT)
    for _ in range(130):
        view.draw(screen, pose, [visual], HudInfo(), DT, fps=22.0)
        view.forget_fps()  # e.g. while a checkpoint loads in the background
    assert not view.world.light_mode
