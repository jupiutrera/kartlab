from collections import defaultdict

import pygame
import pytest

from kartlab.show.drive import DriveApp, keyboard_controls, resolve_track
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


def test_resolve_track_choices():
    assert resolve_track("harbor").name == "Puerto"
    assert resolve_track("eval:1").name == "Evaluación 1"
    assert resolve_track("42").seed == 42


@pytest.fixture
def app():
    app = DriveApp(showcase_track("harbor"), scale=2, autopilot=True, realtime=False)
    yield app
    pygame.quit()


def test_keys_switch_tracks_cameras_and_modes(app):
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


def test_reset_returns_to_the_start_line(app):
    app.run(max_frames=60)
    assert app.racer.progress.distance > 0.0
    app.handle_key(pygame.K_r)
    assert app.racer.progress.distance == 0.0
    assert app.racer.position.s == pytest.approx(0.0, abs=1e-6)
