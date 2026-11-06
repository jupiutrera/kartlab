"""The menu on the 320x180 canvas: palette inks only, everything fits at 1x, texts have glyphs."""

import shutil

import numpy as np
import pygame
import pytest

from kartlab.common.paths import REPO_ROOT
from kartlab.render import pixelfont
from kartlab.render import pixelui as ui
from kartlab.render.palette import game_palette
from kartlab.show.hub import ENTRIES, KartLabApp
from kartlab.show.menu import HINT_Y, Dialog, H, W
from kartlab.show.title import MenuEntry, TitleScreen, build_wordmark, veil

PALETTE = {tuple(int(c) for c in row) for row in game_palette().rgb}
CANVAS = pygame.Rect(0, 0, W, H)


def palette_only(surface: pygame.Surface) -> bool:
    pixels = pygame.surfarray.array3d(surface).reshape(-1, 3)
    return {tuple(int(c) for c in p) for p in np.unique(pixels, axis=0)} <= PALETTE


@pytest.fixture
def make_hub(tmp_path):
    created = []

    def make(scale=1, **kwargs) -> KartLabApp:
        tracks = tmp_path / "tracks"
        tracks.mkdir(exist_ok=True)
        shutil.copy(REPO_ROOT / "tracks" / "ejemplo.json", tracks / "ejemplo.json")
        configs = tmp_path / "configs"
        configs.mkdir(exist_ok=True)
        (configs / "demo.yaml").write_text(
            "# una configuración de prueba\nname: demo\ntotal_steps: 4096\n"
            "n_envs: 1\nvec_env: dummy\n",
            encoding="utf-8",
        )
        options = {
            "scale": scale,
            "realtime": False,
            "background": False,
            "activity_frames": 3,
            "settings_path": tmp_path / "settings.json",
            "root": tmp_path,
            "runs_dir": tmp_path / "runs",
            "logs_dir": tmp_path / "logs",
            "tracks_dir": tracks,
            "configs_dir": configs,
            "spawn": lambda *a, **k: None,
            "opener": lambda path: None,
        }
        app = KartLabApp(**{**options, **kwargs})
        created.append(app)
        return app

    yield make
    pygame.quit()


def entry(app, key):
    return next(e for e in app.entries if e.key == key)


def inside(rect: pygame.Rect) -> bool:
    return CANVAS.contains(rect)


# -- everything fits at 1x ---------------------------------------------------------------------


def test_every_screen_fits_the_320_by_180_window_without_cuts(make_hub):
    app = make_hub(scale=1)
    assert app.screen.get_size() == (320, 180)
    app.run(30)  # past the intro animation
    main = app.main_screen
    assert palette_only(app.screen)
    assert main.rects and all(inside(r) for r in main.rects.values())
    assert inside(pygame.Rect(0, main.desc_y, W, 2 * ui.LINE + 3))
    assert main.desc_y + 2 * ui.LINE + 3 < HINT_Y  # the description never touches the hints
    for e in app.entries:  # every description and hint fits its line or two
        assert len(pixelfont.wrap(e.description, 290)) <= 2, e.key
    assert pixelfont.measure(main.hint) <= W - 6

    screens = []
    for key, steps in (
        ("contrarreloj", 1),  # the kart screen
        ("opciones", 1),
        ("entrenar", 1),
    ):
        app.go_menu()
        app.open_entry(entry(app, key))
        for _ in range(steps - 1):
            app.advance(0)
        screens.append(app.current)
        app.run(20)
        assert palette_only(app.screen), key
    kart, options, train = screens[0], screens[1], screens[2]
    assert kart.__class__.__name__ == "KartScreen"
    assert all(inside(r) for r in kart.swatches)
    assert inside(kart.accept) and inside(kart.arrow_left) and inside(kart.arrow_right)
    assert all(inside(r) for r in options.rects) and options.rects[-1].bottom < HINT_Y
    assert all(inside(r) for r in train.rects.values())
    assert (
        all(inside(r) for r in train.buttons.values()) and train.buttons["launch"].bottom < HINT_Y
    )
    assert inside(train.panel_rect)
    for screen in (kart, options, train):
        assert pixelfont.measure(screen.hint) <= W - 6, screen.__class__.__name__


def test_the_circuit_cards_and_the_race_options_fit_at_1x(make_hub):
    app = make_hub(scale=1)
    app.open_entry(entry(app, "contrarreloj"))
    app.advance(0)
    track = app.current
    assert track.__class__.__name__ == "TrackScreen"
    app.run(20)
    assert track.rects and all(inside(r) for r in track.rects.values())
    assert max(r.bottom for r in track.rects.values()) < 152  # above the details strip
    assert palette_only(app.screen)
    track.selected = len(track.options) - 1  # the evaluation card, with its arrows
    track._reveal()
    app.run(3)
    assert track.eval_arrows is not None and all(inside(r) for r in track.eval_arrows)
    assert pixelfont.measure(track.hint) <= W - 6


def test_the_dialog_box_and_its_buttons_fit_whatever_the_text(make_hub):
    app = make_hub(scale=1)
    long_text = "Una frase larga que se parte en varias líneas. " * 6
    for choices in (None, [("Parar y salir", None), ("Dejar corriendo", None), ("Cancelar", None)]):
        dialog = Dialog(
            "Hay un entrenamiento en marcha con un título muy largo para probar",
            long_text,
            choices=choices,
        )
        assert inside(dialog.box)
        assert all(dialog.box.contains(b) for b in dialog.buttons)
        app.dialog = dialog
        app.run(3)
        assert palette_only(app.screen)
    app.dialog = None


# -- the mouse -------------------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1, 3, 4])
def test_the_mouse_hits_the_rectangles_in_canvas_pixels(make_hub, scale):
    app = make_hub(scale=scale)
    app.run(10)
    rect = app.main_screen.rects[2]
    k = scale
    window = (rect.centerx * k + k // 2, rect.centery * k + k // 2)
    app.handle_event(
        pygame.event.Event(pygame.MOUSEMOTION, pos=window, rel=(0, 0), buttons=(0,) * 3)
    )
    assert app.main_screen.selected == 2
    assert app.mouse == (rect.centerx, rect.centery)  # the hub keeps the position in canvas pixels


# -- the title and the wordmark -------------------------------------------------------------------


def test_the_wordmark_is_the_font_scaled_with_palette_bands_and_an_outline():
    for scale in (3, 4, 5):
        mark, silhouette = build_wordmark(scale=scale)
        assert mark.get_size() == silhouette.get_size()
        pixels = pygame.surfarray.array3d(mark).reshape(-1, 3)
        colours = {tuple(int(c) for c in p) for p in np.unique(pixels, axis=0)}
        assert colours - {(255, 0, 255)} <= PALETTE  # only palette inks besides the colour key
        assert len(colours) >= 6  # outline, extrusion and the fill bands
        assert mark.get_height() >= 7 * scale


def test_the_title_screen_draws_with_palette_inks_in_every_moment():
    pygame.init()
    entries = [
        MenuEntry(k, n, d)
        for k, n, d in (
            ("a", "Ver conducir", "Una versión al volante"),
            ("b", "Carrera de versiones", "Todas a la vez"),
            ("c", "Progresión", "De sin entrenar a la última"),
            ("d", "Carrera contra la IA", "Tú contra karts de la IA"),
        )
    ]
    title = TitleScreen(entries)
    canvas = pygame.Surface((320, 180))
    title.open(0.0)
    for now in (0.1, 0.8, 2.0, 5.2, 5.5, 9.9):  # the drop, the menu, a shine sweep, a blink
        canvas.fill(ui.rgb(ui.SLATE))
        title.draw(
            canvas,
            now,
            "Entrenamiento 2026-11-04_2222_demo-live",
            "7 versiones · de 10k a 401k pasos",
            "↑ ↓ elegir · Intro empezar · Esc salir",
        )
        assert palette_only(canvas), now
    title.move(1)
    assert title.entry.key == "b"
    title.move(-2)
    assert title.entry.key == "d"  # wraps around
    pygame.quit()


def test_the_veil_only_adds_the_darkest_ink_in_a_dither():
    pygame.init()
    canvas = pygame.Surface((320, 180))
    canvas.fill(ui.rgb(ui.WHITE))
    veil(canvas)
    pixels = pygame.surfarray.array3d(canvas).reshape(-1, 3)
    assert {tuple(int(c) for c in p) for p in np.unique(pixels, axis=0)} == {
        ui.rgb(ui.WHITE),
        ui.rgb(ui.NIGHT),
    }
    covered = (pygame.surfarray.array3d(canvas) == ui.rgb(ui.NIGHT)).all(axis=2)  # (x, y)
    middle, edge = covered[:, 70:110].mean(), covered[:, :30].mean()
    assert edge > middle  # denser where the text sits
    pygame.quit()


@pytest.mark.parametrize(
    "name",
    ["Bahía del Faro", "Evaluación 12", "Lazo del molino", "Circuito nuevo", "Cañón", "Meseta"],
)
def test_a_circuit_name_fits_its_card_in_two_lines_without_an_ellipsis(name):
    lines = ui.wrap(name, 70 - 8)
    assert len(lines) <= 2 and all(ui.measure(line) <= 62 for line in lines)
    assert "…" not in "".join(lines) and " ".join(lines) == name


def test_the_training_box_leaves_out_the_log_lines_that_carry_a_path():
    from kartlab.show.menu import readable_log

    log = [
        "Run: runs/2026-11-04_2222_demo-live",
        "Entrenando 10k/400k",
        "Config: configs/random-tracks.yaml",
        r"C:\Users\x\runs\a\metrics.csv",
        "paso 3000",
    ]
    assert readable_log(log) == ["Entrenando 10k/400k", "paso 3000"]


def test_a_caption_is_a_plate_with_centred_text_on_any_background():
    pygame.init()
    canvas = pygame.Surface((W, H))
    canvas.fill(ui.rgb(ui.WHITE))
    ui.dither(canvas, CANVAS, 2)  # a busy background
    box = ui.caption(canvas, "Intro aceptar, Esc salir", 160, 171, ui.STONE_LIGHT)
    assert box.centerx == 160 and box.h == 12 and CANVAS.contains(box)
    inside = pygame.surfarray.array3d(canvas)[box.x + 1 : box.right - 1, box.y + 1 : box.bottom - 1]
    colours = {tuple(int(c) for c in p) for p in inside.reshape(-1, 3)}
    assert colours <= {ui.rgb(ui.SLATE), ui.rgb(ui.STONE_LIGHT)}  # no background shows through
    pygame.quit()


def test_every_menu_text_has_glyphs():
    for e in ENTRIES:
        for text in (e.label, e.description, e.group or ""):
            assert all(pixelfont.has_glyph(c) for c in text), text
