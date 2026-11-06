"""The screens moved to the pixel canvas: only palette inks, on the 320x180 grid, magnified
by an integer factor together with the world."""

from pathlib import Path

import numpy as np
import pygame
import pytest

from kartlab.render import pixelfont
from kartlab.render import pixelui as ui
from kartlab.render.display import to_logical
from kartlab.render.palette import game_palette
from kartlab.render.scene import KartVisual
from kartlab.render.topdown import TopDownView
from kartlab.render.view import GameView, HudInfo
from kartlab.show import editor as ed
from kartlab.show.charts import RewardChart
from kartlab.sim.kart import DT, KartState
from kartlab.sim.showcase import showcase_track
from kartlab.sim.trackfile import load_spec

PALETTE = {tuple(int(c) for c in row) for row in game_palette().rgb}
EXAMPLE = Path("tracks/ejemplo.json")


def palette_only(surface: pygame.Surface) -> bool:
    pixels = pygame.surfarray.array3d(surface).reshape(-1, 3)
    return {tuple(int(c) for c in p) for p in np.unique(pixels, axis=0)} <= PALETTE


@pytest.fixture
def pygame_session():
    pygame.init()
    yield
    pygame.quit()


# -- the reward chart --------------------------------------------------------------------------


@pytest.mark.parametrize("log_x", [False, True])
def test_the_reward_chart_is_palette_only_with_and_without_data(pygame_session, log_x):
    chart = RewardChart((90, 54), log_x=log_x)
    assert palette_only(chart.draw([], 0, None))  # waiting for data
    rows = [{"steps": 10_000.0 * k, "reward_mean": float(k * k - 3)} for k in range(1, 30)]
    surface = chart.draw(rows, 1, marker=120_000)
    assert surface.get_size() == (90, 54) and palette_only(surface)
    lit = pygame.surfarray.array3d(surface)
    assert (lit == ui.rgb(ui.YELLOW)).all(axis=2).any()  # the curve
    assert (lit == ui.rgb(ui.ORANGE)).all(axis=2).any()  # the marker


def test_a_chart_with_a_single_flat_value_does_not_divide_by_zero(pygame_session):
    chart = RewardChart((60, 40))
    rows = [{"steps": 1000.0 * k, "reward_mean": 5.0} for k in range(1, 5)]
    assert palette_only(chart.draw(rows, 1, marker=None))


# -- the pixel top-down view -------------------------------------------------------------------


def test_the_pixel_top_down_map_uses_only_palette_colours(pygame_session):
    track = showcase_track("harbor")
    view = TopDownView(track, (224, 180), pixel=True)
    surface = pygame.Surface((224, 180))
    kart = KartState(*track.pose_at(30.0))
    trail = [(kart.x - k, kart.y) for k in range(8)]
    view.draw(surface, [(kart, ui.rgb(ui.ORANGE))], (0.0, 0.0), follow=False, trail=trail)
    assert palette_only(surface)
    view.draw(surface, [(kart, ui.rgb(ui.ORANGE), 0.6)], (kart.x, kart.y), follow=True)
    assert palette_only(surface)


def test_the_window_top_down_view_is_unchanged_by_the_pixel_option(pygame_session):
    track = showcase_track("harbor")
    surface = pygame.Surface((320, 200))
    TopDownView(track, (320, 200)).draw(surface, [], (0.0, 0.0), follow=False)
    assert not palette_only(surface)  # the debug view keeps its own colours


# -- the 3D view with pixel interface ----------------------------------------------------------


def test_pixel_interface_is_drawn_on_the_world_and_magnified_with_it(pygame_session):
    track = showcase_track("plateau")
    pygame.display.set_mode((64, 64))
    view = GameView(track, (1366, 768))
    screen = pygame.Surface((1366, 768))
    visual = KartVisual(KartState(*track.pose_at(30.0)), 0)
    pose = view.update([visual], DT)
    seen = []

    def interface(surface):
        seen.append(surface.get_size())
        ui.plate(surface, (10, 10, 40, 20), ui.NAVY)
        ui.text(surface, "ESPERA", 14, 16, ui.YELLOW)

    view.draw(screen, pose, [visual], HudInfo(), DT, hud=False, pixel_ui=interface)
    assert seen == [(320, 180)]
    assert "interfaz" in view.last_timings
    block = pygame.surfarray.array3d(screen)[43 + 4 * 10 : 43 + 4 * 50, 24 + 4 * 10 : 24 + 4 * 30]
    small = pygame.surfarray.array3d(view.world.world)[10:50, 10:30]
    assert np.array_equal(block, np.kron(small, np.ones((4, 4, 1), dtype=np.uint8)))


@pytest.mark.parametrize("variant", ["day", "sunset", "night"])
def test_night_and_sunset_do_not_tint_the_interface(pygame_session, variant):
    track = showcase_track("plateau")
    pygame.display.set_mode((64, 64))
    view = GameView(track, (1280, 720), variant=variant)
    screen = pygame.Surface((1280, 720))
    visual = KartVisual(KartState(*track.pose_at(30.0)), 0)
    pose = view.update([visual], DT)

    def interface(surface):
        ui.plate(surface, (10, 10, 40, 20), ui.SLATE)
        ui.text(surface, "I", 14, 16, ui.WHITE)

    view.draw(screen, pose, [visual], HudInfo(), DT, hud=False, pixel_ui=interface)
    world = view.world.world
    assert world.get_at((10, 10))[:3] == ui.rgb(ui.NIGHT)
    assert world.get_at((12, 12))[:3] == ui.rgb(ui.SLATE)
    assert world.get_at((14, 16))[:3] == ui.rgb(ui.WHITE)


# -- the editor --------------------------------------------------------------------------------


@pytest.fixture
def editor(tmp_path):
    apps = []

    def make(size=(1280, 720)):
        app = ed.EditorApp(
            load_spec(EXAMPLE),
            path=EXAMPLE,
            size=size,
            tracks_dir=tmp_path / "tracks",
            root=tmp_path,
            background=False,
        )
        apps.append(app)
        return app

    yield make
    for app in apps:
        app.close()
    pygame.quit()


def test_every_state_of_the_editor_is_palette_only(editor):
    app = editor()
    app.settle()
    app.draw()
    assert palette_only(app.surface) and palette_only(app.screen)
    app.show_help = True
    app.draw()
    assert palette_only(app.surface)
    app.show_help = False
    app.open_rename()
    app.prompt.insert("Circuito de prueba")
    app.draw()
    assert palette_only(app.surface)
    app.prompt.text = "x" * app.prompt.max_length  # the "limit" hint
    app.draw()
    assert palette_only(app.surface)
    app._close_prompt()
    app.model.move_corner(0, 330.0, 250.0)  # an invalid design: red rings, diamonds, issue list
    app.changed()
    app.settle()
    app.mouse = (app.panel_rect.x + 10, 110)  # over the issue list
    app.say("Guardado " + "x" * 300, "error")
    app.draw()
    assert app.verdict == "invalid" and palette_only(app.surface)
    for kind in ("ok", "warn", "info"):
        app.say("Un aviso con tildes: ¡ÁÉÍÓÚ Ñ! ¿Sí?", kind)
        app.draw()
        assert palette_only(app.surface)


def test_the_help_screen_fits_the_canvas_and_every_text_has_glyphs(editor):
    app = editor()
    key_w = max(pixelfont.measure(k) for k, _ in ed.HELP_ROWS) + 10
    for key, desc in ed.HELP_ROWS:
        assert all(pixelfont.has_glyph(c) for c in key + desc), (key, desc)
        assert 6 + key_w + pixelfont.measure(desc) <= ui.CANVAS_SIZE[0] - 8 - 6
    assert 19 + len(ed.HELP_ROWS) * ui.LINE <= ui.CANVAS_SIZE[1] - 6
    del app


def test_the_panel_texts_fit_in_the_panel(editor):
    app = editor()
    app.settle()
    inner = ed.PANEL_W - 2 * ed.PAD - 2
    for hint in ed.HINTS:
        assert pixelfont.measure(hint) <= inner
    for label in ("LONGITUD", "RADIO MÍN", "ESQUINAS", "ANCHURA", "SEMILLA", "HIERBA MÍN"):
        assert pixelfont.measure(label) + 36 <= inner  # room for a value like "16,5 M"


def test_mouse_positions_are_translated_from_the_window_to_the_canvas(editor):
    app = editor(size=(1366, 768))  # scale 4, with bars: the canvas starts at (43, 24)
    app.settle()
    lx, ly = (round(v) for v in app.corner_screen(2))
    window = (43 + 4 * lx + 2, 24 + 4 * ly + 2)  # the middle of that canvas pixel
    assert to_logical(window, (1366, 768)) == (lx, ly)
    down = pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=window, button=1)
    app.handle_event(down)
    assert app.selected == 2 and app._drag is not None
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=window, button=1))
    assert app._drag is None


def test_the_editor_is_always_drawn_at_320_by_180(editor):
    for size in ((1280, 720), (1000, 700), (1920, 1080)):
        app = editor(size=size)
        app.draw()
        assert app.surface.get_size() == (320, 180)
        assert app.screen.get_size() == size
        assert app.canvas.width + app.panel_rect.width == 320
