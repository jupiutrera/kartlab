from collections import defaultdict

import pygame
import pytest

from kartlab.show.drive import DriveApp, keyboard_controls
from kartlab.sim.showcase import showcase_track


def keys(*pressed: int) -> defaultdict:
    return defaultdict(bool, dict.fromkeys(pressed, True))


@pytest.mark.parametrize(
    "pressed, controls",
    [
        ((), (0.0, 0.0)),
        ((pygame.K_LEFT, pygame.K_UP), (1.0, 1.0)),
        ((pygame.K_d, pygame.K_s), (-1.0, -1.0)),
        ((pygame.K_LEFT, pygame.K_RIGHT), (0.0, 0.0)),
        ((pygame.K_w, pygame.K_DOWN), (0.0, 0.0)),
    ],
)
def test_keyboard_controls(pressed, controls):
    assert keyboard_controls(keys(*pressed)) == controls


@pytest.fixture
def app():
    app = DriveApp(showcase_track("harbor"), scale=2, autopilot=True, realtime=False)
    yield app
    pygame.quit()


@pytest.fixture
def topdown_app():
    app = DriveApp(
        showcase_track("harbor"), scale=2, autopilot=True, view="topdown", realtime=False
    )
    yield app
    pygame.quit()


def test_keys_switch_tracks_cameras_and_modes(topdown_app):
    app = topdown_app
    app.handle_key(pygame.K_2)
    assert app.track.name == "Cañón"
    app.handle_key(pygame.K_e)
    assert app.track.name == "Evaluación 1"
    app.handle_key(pygame.K_n)
    assert app.track.name.startswith("Aleatorio")
    app.handle_key(pygame.K_c)
    assert not app.follow
    app.handle_key(pygame.K_TAB)
    assert not app.autopilot_on
    app.handle_key(pygame.K_p)
    assert app.paused


def test_3d_keys_cycle_cameras_variants_and_overlays(app):
    assert app.game is not None and app.game.camera_name == "chase"
    app.handle_key(pygame.K_c)
    assert app.game.camera_name == "broadcast"
    app.handle_key(pygame.K_v)
    assert app.game.world.variant == "sunset"
    app.handle_key(pygame.K_g)
    app.handle_key(pygame.K_F3)
    app.run(max_frames=10)
    assert app.game.world.show_rays and app.game.diagnostics
    app.handle_key(pygame.K_l)
    assert app.game.world.light_mode and not app.game.auto_light
    app.handle_key(pygame.K_t)
    assert app.view_mode == "topdown" and app.view is not None
    app.run(max_frames=5)


def test_reset_returns_to_the_start_line(app):
    app.run(max_frames=60)
    assert app.racer.progress.distance > 0.0
    app.handle_key(pygame.K_r)
    assert app.racer.progress.distance == 0.0
    assert app.racer.position.s == pytest.approx(0.0, abs=1e-6)
