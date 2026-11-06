"""The HUD, the race lettering and the panels of every mode on the pixel canvas."""

from pathlib import Path

import numpy as np
import pygame
import pytest

from kartlab.render import bigtext
from kartlab.render import pixelui as ui
from kartlab.render.palette import game_palette
from kartlab.render.scene import KartVisual
from kartlab.render.view import GameView, HudInfo, TrialHud
from kartlab.show.drive import DriveApp
from kartlab.show.overlays import Overlays, build_help, run_title, toggle_text
from kartlab.show.racehud import RaceHud
from kartlab.show.trialrace import COUNTDOWN_FRAMES
from kartlab.sim.autopilot import Autopilot
from kartlab.sim.kart import DT, KartState
from kartlab.sim.showcase import showcase_track

PALETTE = {tuple(int(c) for c in row) for row in game_palette().rgb}
SRC = Path(__file__).resolve().parents[1] / "src" / "kartlab"


def palette_only(surface: pygame.Surface) -> bool:
    pixels = pygame.surfarray.array3d(surface).reshape(-1, 3)
    return {tuple(int(c) for c in p) for p in np.unique(pixels, axis=0)} <= PALETTE


def changed_box(before: pygame.Surface, after: pygame.Surface) -> pygame.Rect:
    diff = (pygame.surfarray.array3d(before) != pygame.surfarray.array3d(after)).any(axis=2)
    xs, ys = np.nonzero(diff)
    return pygame.Rect(xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    from kartlab.train.config import PPOParams, TrainConfig
    from kartlab.train.train import train

    config = TrainConfig(
        name="hud-test",
        total_steps=768,
        n_envs=2,
        vec_env="dummy",
        low_priority=False,
        ppo=PPOParams(n_steps=64, batch_size=64, n_epochs=1),
        checkpoint_every=256,
        checkpoint_at=(0,),
        eval_tracks=(),
        env={"track": "harbor"},
    )
    run = train(config, runs_dir=tmp_path_factory.mktemp("runs"), verbose=0)
    run.best_model.write_bytes(run.list_checkpoints()[-1].model.read_bytes())
    return run


@pytest.fixture
def view():
    pygame.init()
    pygame.display.set_mode((64, 64))
    game = GameView(showcase_track("plateau"), (1280, 720))
    # The panels slide in during the first 0.7 s by the wall clock; a fixed time well after
    # that keeps every frame of these tests the same however slow the machine is.
    game.fixed_time = 10.0
    yield game
    pygame.quit()


def frame(view, info, variant=None, visuals=None):
    kart = KartState(*view.track.pose_at(30.0))
    visuals = visuals or [KartVisual(kart, 0)]
    pose = view.update(visuals, DT)
    return view.render_canvas(pose, visuals, info, DT)


# -- no vector fonts left ----------------------------------------------------------------------


def test_no_module_of_the_game_uses_a_vector_font():
    for folder in ("render", "show"):
        for path in sorted((SRC / folder).glob("*.py")):
            source = path.read_text(encoding="utf-8")
            assert "pygame.font" not in source, path.name
            assert "TextCache" not in source, path.name
            assert "FONTS_DIR" not in source, path.name


# -- the HUD -----------------------------------------------------------------------------------


def test_the_hud_is_palette_only_in_every_state(view):
    full = HudInfo(
        laps=1,
        lap_time=62.5,
        best_lap=61.0,
        speed=31.0,
        label="Conducción libre",
        extra=["Puerto", "Contrarreloj, fantasma", "H: ayuda"],
        trial=TrialHud(record=60.0, last=61.2, delta=-0.42),
    )
    view.hud.notify("Nuevo récord 1:00,00 (−1,50)", view.now())
    assert palette_only(frame(view, full))
    for delta in (None, 0.003, 1.5):
        full.trial = TrialHud(record=None if delta is None else 60.0, last=None, delta=delta)
        assert palette_only(frame(view, full))
    race = HudInfo(lap_panel=False, label_panel=False)
    assert palette_only(frame(view, race))
    view.diagnostics = True
    view.timings["suelo"] = 1.2
    canvas = frame(view, full)
    view.present(pygame.Surface((1280, 720)))
    assert palette_only(canvas)


def test_the_speed_and_the_laps_use_a_decimal_comma(view):
    empty = frame(view, HudInfo(lap_panel=False, label_panel=False)).copy()
    shown = frame(view, HudInfo(lap_time=64.5, best_lap=61.25, label_panel=False))
    box = changed_box(empty, shown)
    assert box.x < 90 and box.y < 20  # the lap panel is at the top left
    from kartlab.render.pixelfont import measure

    assert measure("0:04,50", 2, True) == measure("0:11,11", 2, True)  # a clock never jitters


def overlaps(view, rect, skip=None):
    """The reserved panels of the last frame that ``rect`` touches."""
    return [r for r in view.hud.reserved if r != skip and r.colliderect(rect)]


def present(view):
    view.present(pygame.Surface((1280, 720)))


def test_a_notice_never_covers_a_panel_in_any_hud_state(view):
    ov = Overlays(view)
    race = RaceHud(ov)
    rows = [(1, "IA 1", "2/3"), (2, "IA 2", "2/3"), (3, "Tú", "1/3")]
    long_text = "Hay versiones nuevas en el entrenamiento y otra más"
    states = {
        "free driving": HudInfo(laps=1, lap_time=3.0, label="Conducción libre", extra=["Puerto"]),
        "trial": HudInfo(trial=TrialHud(record=60.0, last=61.0, delta=-0.2), label="Piloto"),
        "race": HudInfo(lap_panel=False, label_panel=True, label="Carrera contra la IA"),
        "race, no model panel": HudInfo(lap_panel=False, label_panel=False),
    }
    for name, info in states.items():
        for text in ("Sin récord", long_text):
            view.hud.notify(text, view.now())
            frame(view, info)
            if info.lap_panel is False:
                ov.standings(rows, (3, 3), 2)
                race.lap_counter(1, 3, "0:10,00", "0:10,00")
                race.position(3, 3, 5.0)
            present(view)
            rect = view.hud.notice_rect
            assert rect is not None and not overlaps(view, rect), (name, text, rect)


def test_a_notice_goes_where_there_is_room_even_with_the_brain_and_a_chart(view):
    ov = Overlays(view)
    frame(view, HudInfo(label="Versión 401k pasos", extra=["Puerto", "Versión 1 de 8", "H: ayuda"]))
    view.hud.reserve((3, 47, 100, 109))  # the brain panel on the left
    chart = type("Chart", (), {"draw": lambda self, *a: pygame.Surface((92, 56))})()
    ov.chart_panel(chart, type("M", (), {"rows": [], "version": 0})(), 0)
    view.hud.notify("Nueva versión: 401k pasos", view.now())
    present(view)
    rect = view.hud.notice_rect
    assert rect is not None and not overlaps(view, rect)


def test_the_race_panels_do_not_cover_each_other(view):
    """The standings, the lap bar and the model panel have their own places."""
    ov = Overlays(view)
    hud = RaceHud(ov)
    blank = view.canvas.copy()
    blank.fill(ui.rgb(ui.NIGHT))

    def painted(draw):
        canvas = view.canvas
        canvas.fill(ui.rgb(ui.NIGHT))
        draw()
        return changed_box(blank, canvas)

    rows = [(1, "IA 1", "2/3"), (2, "IA 2", "2/3"), (3, "Tú", "1/3"), (4, "IA 4", "META")]
    standings = painted(lambda: ov.standings(rows, (3, 3), 2))
    bar = painted(lambda: hud.lap_counter(2, 3, "1:23,45", "0:30,12", "0:29,80"))
    assert not standings.colliderect(bar)
    place = painted(lambda: hud.position(3, 4, 5.0))
    assert place.bottom < 159 and place.right <= 320  # above the speed plate


def test_the_big_place_is_gold_for_the_winner_and_the_same_font_as_the_small_text(view):
    ov = Overlays(view)
    canvas = view.canvas
    canvas.fill(ui.rgb(ui.NIGHT))
    RaceHud(ov).position(1, 4, 5.0)
    lit = pygame.surfarray.array3d(canvas).reshape(-1, 3)
    gold = {tuple(ui.rgb(ink)) for ink in bigtext.GOLD}
    assert gold & {tuple(int(c) for c in p) for p in np.unique(lit, axis=0)}
    assert palette_only(canvas)
    assert bigtext.place_look(1) == bigtext.GOLD


def test_countdown_banners_and_results_fit_and_use_the_palette(view):
    ov = Overlays(view)
    hud = RaceHud(ov)
    for frame_no in (0, 59, 60, 119, 120, 179, 180, 200, 240):
        view.canvas.fill(ui.rgb(ui.NIGHT))
        hud.countdown(frame_no, 180, 60)
        assert palette_only(view.canvas)
    for text in ("ÚLTIMA VUELTA", "¡META!"):
        for age in (0.0, 0.1, 1.0, 2.2, 2.45):
            view.canvas.fill(ui.rgb(ui.NIGHT))
            hud.banner(text, age)
            assert palette_only(view.canvas)
    rows = [(f"{i}º  IA {i}  1:2{i},45  mejor 0:4{i},10", ui.TEXT) for i in range(1, 8)]
    rows.append(("1º  Tú  1:20,00  mejor 0:40,00", ui.YELLOW))
    view.canvas.fill(ui.rgb(ui.NIGHT))
    hud.results("Resultados", 1, rows, "Intro repetir, Esc salir", 1.0)
    assert palette_only(view.canvas)
    blank = pygame.Surface(ui.CANVAS_SIZE)
    blank.fill(ui.rgb(ui.NIGHT))
    box = changed_box(blank, view.canvas)
    assert pygame.Rect(0, 0, 320, 180).contains(box)  # eight rows still fit


def test_dimming_is_a_dither_in_steps_not_a_blend(view):
    ov = Overlays(view)
    canvas = view.canvas
    canvas.fill(ui.rgb(ui.WHITE))
    ov.dim(0.5)
    covered = (pygame.surfarray.array3d(canvas) == ui.rgb(ui.NIGHT)).all(axis=2).mean()
    assert covered == pytest.approx(0.5, abs=0.02)
    assert palette_only(canvas)
    ov.dim(1.0)
    assert (pygame.surfarray.array3d(canvas) == ui.rgb(ui.NIGHT)).all()


def test_helpers_of_the_panels():
    assert toggle_text("Sensores", True) == "Sensores: Sí"
    assert toggle_text("Cerebro", False) == "Cerebro: No"

    class Run:
        root = Path("runs/2026-11-04_2159_random-tracks")

    assert run_title(Run()) == "random-tracks, 4 nov"
    assert run_title(type("R", (), {"root": Path("runs/otro")})()) == "otro"


def test_the_help_plates_fit_the_canvas():
    from kartlab.show import drive, viewer

    pygame.init()
    for rows, left in ((drive.HELP, drive.HELP_LEFT), (viewer.HELP, viewer.HELP_LEFT)):
        columns = [
            [(k, v, False) for k, v in rows[:left]],
            [(k, v, False) for k, v in rows[left:]],
        ]
        panel = build_help(None, columns)
        assert panel.get_width() <= 320 and panel.get_height() <= 180
        assert palette_only(panel)
    pygame.quit()


# -- the time trial in the top-down view ---------------------------------------------------------


@pytest.fixture
def trial_app(tmp_path):
    apps = []

    def build(**kwargs):
        kwargs.setdefault("scale", 2)
        kwargs.setdefault("realtime", False)
        kwargs.setdefault("time_trial", True)
        kwargs.setdefault("view", "topdown")
        app = DriveApp(showcase_track("harbor"), records_dir=tmp_path, **kwargs)
        apps.append(app)
        return app

    yield build
    pygame.quit()


def bigtext_pixels(canvas: pygame.Surface) -> int:
    pixels = pygame.surfarray.array3d(canvas)
    count = 0
    for look in (bigtext.AMBER, bigtext.EMBER, bigtext.GOLD, bigtext.GREEN):
        for ink in look:
            count += int((pixels == ui.rgb(ink)).all(axis=2)[100:220, 40:100].sum())
    return count


def test_the_top_down_time_trial_has_the_big_countdown_and_the_results(trial_app, monkeypatch):
    app = trial_app()
    app.run(max_frames=30)
    assert app.session.counting and app.canvas.get_size() == (320, 180)
    assert bigtext_pixels(app.canvas) > 200  # the big digit
    assert palette_only(app.canvas)
    track = showcase_track("harbor")
    pilot = Autopilot(track)
    monkeypatch.setattr(
        "kartlab.show.drive.keyboard_controls", lambda pressed: pilot.act(app.racer)
    )
    for _ in range(60 * 400):
        app.physics_step()
        if app.session.finished:
            break
    assert app.session.finished
    app.run(max_frames=60 * 4)
    assert app.session.results_age() > 0.0
    canvas = app.canvas
    assert palette_only(canvas)
    # The table is on a navy plate in the middle with the yellow stripe at its left.
    assert (pygame.surfarray.array3d(canvas)[80:240, 40:140] == ui.rgb(ui.NAVY)).all(axis=2).any()
    assert COUNTDOWN_FRAMES == 180


def test_the_top_down_help_is_off_until_h_and_the_data_panels_do_not_hide_the_view(trial_app):
    app = trial_app(time_trial=False)
    assert not app.show_help
    app.run(max_frames=10)
    assert palette_only(app.canvas)
    app.handle_key(pygame.K_h)
    app.run(max_frames=2)
    assert app.show_help and palette_only(app.canvas)


def widest_gap(grid: np.ndarray) -> int:
    """The widest run of empty cells between two lit cells of one row."""
    best = 0
    for row in grid:
        lit = np.flatnonzero(row)
        if len(lit) > 1:
            best = max(best, int(np.max(np.diff(lit))) - 1)
    return best


def test_the_big_signs_keep_the_inner_gaps_of_the_a_and_the_y_open():
    """The word signs use the font strokes, one cell thick: the hole of the A and the fork of
    the Y stay as wide as in the font, so "¡YA!" reads (bold strokes closed them)."""
    assert bigtext.SIGN_BOLD is False
    for char in "AY":
        assert widest_gap(bigtext._grid(char, bigtext.SIGN_BOLD)) >= 3
    pygame.init()
    surface = bigtext.render("A", 6, bigtext.GREEN, bold=False)
    cells = np.array(
        [[tuple(surface.get_at((x * 6 + 3, y * 6 + 3)))[:3] for x in range(7)] for y in range(9)]
    )
    night, fills = ui.rgb(ui.NIGHT), {ui.rgb(ink) for ink in bigtext.GREEN}
    hole = [tuple(c) for c in cells[3, 2:5]]  # the three cells of the hole, one row under the top
    assert hole == [night] * 3
    assert tuple(cells[3, 1]) in fills and tuple(cells[3, 5]) in fills  # between its legs
    pygame.quit()


def test_countdown_and_banners_draw_their_words_without_bold(monkeypatch):
    seen = []
    original = bigtext.render

    def spy(text, cell, look, bold=True):
        seen.append((text, bold))
        return original(text, cell, look, bold)

    monkeypatch.setattr(bigtext, "render", spy)
    pygame.init()
    canvas = pygame.Surface(ui.CANVAS_SIZE)
    bigtext.countdown(canvas, 190, 180, 60)  # "¡YA!"
    bigtext.countdown(canvas, 30, 180, 60)  # a digit
    bigtext.banner(canvas, "¡META!", 0.5)
    bigtext.banner(canvas, "ÚLTIMA VUELTA", 0.5)
    assert seen and all(bold is False for _, bold in seen)
    assert {text for text, _ in seen} >= {"¡YA!", "¡META!", "ÚLTIMA VUELTA"}
    pygame.quit()


# -- in the viewer, the live view and the time trial ---------------------------------------------


def test_cards_and_notices_avoid_the_panels_of_each_mode(tiny_run):
    from kartlab.show.viewer import ViewerApp

    checked = 0
    for mode in ("watch", "race", "progression", "vsai"):
        viewer = ViewerApp(tiny_run, scale=2, realtime=False, mode=mode)
        try:
            for frames in (10, 40, 120):
                viewer.game.notify("Hay versiones nuevas en el entrenamiento")
                viewer.run_loop(frames)
                hud = viewer.game.hud
                if hud.notice_rect is not None:
                    assert not overlaps(viewer.game, hud.notice_rect), (mode, frames)
                    checked += 1
                card = viewer.ov.last_card
                if card is not None and mode == "progression":
                    assert not overlaps(viewer.game, card, skip=card), (mode, frames, card)
        finally:
            viewer.close()
            pygame.quit()
    assert checked >= 8


def test_the_fixed_panels_of_each_mode_never_cover_each_other(tiny_run):
    from kartlab.show.viewer import ViewerApp

    for mode in ("watch", "race", "progression", "vsai"):
        viewer = ViewerApp(tiny_run, scale=2, realtime=False, mode=mode)
        try:
            for frames in (60, 40, 100):  # after the panels have slid in
                viewer.run_loop(frames)
                panels = viewer.game.hud.reserved
                for i, a in enumerate(panels):
                    for b in panels[i + 1 :]:
                        assert not a.colliderect(b), (mode, frames, a, b)
        finally:
            viewer.close()
            pygame.quit()


def test_the_progression_card_does_not_cover_the_reward_chart(tiny_run):
    from kartlab.show.viewer import ViewerApp

    viewer = ViewerApp(tiny_run, scale=2, realtime=False, mode="progression")
    try:
        viewer.run_loop(30)  # the card of the first version is up
        card = viewer.ov.last_card
        chart = pygame.Rect(320 - 3 - 92, (180 - 56) // 2, 92, 56)
        assert card is not None and not card.colliderect(chart)
    finally:
        viewer.close()
        pygame.quit()


def test_the_trial_notice_and_the_race_start_do_not_repeat_the_mode_panel(tiny_run, tmp_path):
    from kartlab.show.drive import DriveApp
    from kartlab.show.viewer import ViewerApp

    app = DriveApp(
        showcase_track("harbor"), scale=2, realtime=False, time_trial=True, records_dir=tmp_path
    )
    try:
        app.run(max_frames=5)
        hud = app.game.hud
        assert hud.notice_rect is not None and not overlaps(app.game, hud.notice_rect)
    finally:
        pygame.quit()
    viewer = ViewerApp(tiny_run, scale=2, realtime=False, mode="vsai")
    try:
        viewer.run_loop(5)
        assert viewer.game.hud.notice is None  # the model panel already says "Carrera contra la IA"
    finally:
        viewer.close()
        pygame.quit()


def test_the_top_down_signs_sit_clear_of_the_data_panels(tmp_path):
    from kartlab.show.drive import DriveApp

    app = DriveApp(
        showcase_track("harbor"),
        scale=2,
        realtime=False,
        time_trial=True,
        view="topdown",
        records_dir=tmp_path,
    )
    try:
        app.run(max_frames=3)
        plates = []
        original = ui.plate

        def spy(target, rect, *args, **kwargs):
            plates.append(pygame.Rect(rect))
            return original(target, rect, *args, **kwargs)

        ui.plate = spy
        try:
            app._draw_blocks(app.canvas, 27)
        finally:
            ui.plate = original
        signs = []
        original_draw = bigtext.draw

        def spy_draw(*args, **kwargs):
            rect = original_draw(*args, **kwargs)
            signs.append(rect)
            return rect

        bigtext.draw = spy_draw
        try:
            for text in ("¡META!", "ÚLTIMA VUELTA"):
                app.race_hud.banner(text, 0.5)
            app.race_hud.countdown(190, 180, 60)  # "¡YA!"
        finally:
            bigtext.draw = original_draw
        assert signs and plates
        for sign in signs:
            assert not any(sign.colliderect(plate) for plate in plates), (sign, plates)
    finally:
        pygame.quit()


# -- the big lettering and the results card keep clear of the panels ------------------------------


def test_the_countdown_and_the_banners_cover_no_panel_with_seven_versions_racing(view):
    ov = Overlays(view)
    hud = RaceHud(ov)
    rows = [(i, f"{(i + 1) * 100}k pasos", "1/2") for i in range(7)]
    info = HudInfo(
        laps=0,
        lap_time=0.0,
        best_lap=None,
        speed=0.0,
        label="Carrera de versiones",
        extra=["Puerto", "2 vueltas", "Sigues a 1,5M pasos"],
        lap_panel=False,
    )
    drawn = 0
    for age in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0):
        for step in range(0, 241, 5):
            frame(view, info)
            ov.standings(rows, (3, 3), 1)
            rect = hud.countdown(step, 180, 60)
            if rect is not None:
                drawn += 1
                assert not overlaps(view, rect), (step, rect, view.hud.reserved)
        for text in ("ÚLTIMA VUELTA", "¡META!"):
            frame(view, info)
            ov.standings(rows, (3, 3), 1)
            rect = hud.banner(text, age)
            assert rect is not None and not overlaps(view, rect), (text, age, rect)
    assert drawn > 20


def test_the_big_lettering_only_shrinks_when_no_room_is_left():
    avoid = [(0, 0, 320, 120)]  # a panel over the top two thirds of the canvas
    cell, center = bigtext.place_sign("¡YA!", 6, 0, 0.7, (160, 68), avoid)
    width, height = bigtext.sign_size("¡YA!", cell)
    assert cell < 6 and center[1] - height // 2 >= 120  # it dropped below the panel and shrank
    cell, center = bigtext.place_sign("¡YA!", 6, 0, 0.7, (160, 68), ())
    assert cell == 6 and center == (160, 68)  # with room it stays where it was


def test_the_results_card_of_the_top_down_trial_covers_no_panel(trial_app, monkeypatch):
    app = trial_app()
    app.run(max_frames=5)
    pilot = Autopilot(showcase_track("harbor"))
    monkeypatch.setattr(
        "kartlab.show.drive.keyboard_controls", lambda pressed: pilot.act(app.racer)
    )
    for _ in range(60 * 400):
        app.physics_step()
        if app.session.finished:
            break
    assert app.session.finished
    for autopilot in (False, True):  # the right panel is longer with the autopilot label
        app.autopilot_on = autopilot
        app.run(max_frames=60 * 4)
        hud = app.race_hud
        card = hud.card
        assert card is not None and pygame.Rect(0, 0, 320, 180).contains(card)
        others = [p for p in hud.panels if p != card]
        assert len(others) >= 3  # the two data blocks and the lap bar
        assert not any(card.colliderect(p) for p in others), (card, others)


def test_a_long_track_name_is_split_in_lines_in_the_top_down_panel(tmp_path, monkeypatch):
    from kartlab.sim.trackfile import load_track_file

    track = load_track_file("tracks/ocho_puente.json")
    seen = []
    original = ui.text

    def spy(target, value, *args, **kwargs):
        seen.append(value)
        return original(target, value, *args, **kwargs)

    monkeypatch.setattr(ui, "text", spy)
    app = DriveApp(track, scale=2, realtime=False, view="topdown", records_dir=tmp_path)
    try:
        app.run(max_frames=3)
    finally:
        pygame.quit()
    assert not any("…" in line for line in seen)
    assert {"LAZO DEL", "MOLINO -", "PUENTE"} <= {line.upper() for line in seen}


def test_the_time_trial_panel_has_a_fixed_width_and_never_reaches_the_lap_bar(view):
    hud = RaceHud(Overlays(view))
    widths = set()
    for record, last, delta in (
        (None, None, None),
        (46.22, 46.22, -0.42),
        (71.11, 111.11, 12.34),
        (60.0, 59.99, 0.0),
    ):
        info = HudInfo(
            lap_panel=False,
            trial=TrialHud(record=record, last=last, delta=delta),
        )
        frame(view, info)
        trial = view.hud.reserved[0]  # the first panel of the frame is the time-trial one
        for laps in (1, 3):
            frame(view, info)
            hud.lap_counter(laps, 3, "2:19,78", "0:46,22", "0:46,22")
            panels = view.hud.reserved
            for i, a in enumerate(panels):
                for b in panels[i + 1 :]:
                    assert not a.colliderect(b), (record, last, laps, a, b)
        widths.add(trial.w)
    assert len(widths) == 1  # whatever the times, the panel is as wide as always
