"""The track editor, driven headless with synthetic mouse and keyboard input."""

import json
import time
from pathlib import Path

import pygame
import pytest

from kartlab.show import editor as ed
from kartlab.show.editor import EditorApp, EditorModel, Viewport, resolve_save_path, slugify
from kartlab.sim.trackfile import (
    MIN_CORNERS,
    TrackFileError,
    load_spec,
    load_track_file,
    spec_to_json,
    validate_spec,
)

EXAMPLE = Path("tracks/ejemplo.json")
EIGHT = Path("tracks/ocho.json")
NO_MODS = 0
CTRL = pygame.KMOD_CTRL
SHIFT = pygame.KMOD_SHIFT


class FakeProcess:
    def __init__(self) -> None:
        self.running = True

    def poll(self):
        return None if self.running else 0


@pytest.fixture
def launched():
    return []


@pytest.fixture
def make_app(tmp_path, launched):
    apps = []

    def spawn(command):
        process = FakeProcess()
        launched.append((list(command), process))
        return process

    def make(source=EXAMPLE, **kwargs):
        if isinstance(source, Path):
            spec, path = load_spec(source), source
        else:
            spec, path = source, None
        kwargs.setdefault("path", path)
        app = EditorApp(
            spec, size=(1280, 800), spawn=spawn, tracks_dir=tmp_path / "tracks", **kwargs
        )
        apps.append(app)
        return app

    yield make
    pygame.quit()


# -- helpers ---------------------------------------------------------------------------------


def corner_px(app, index):
    x, y = app.corner_screen(index)
    return round(x), round(y)


def drag(app, start, end, button=1, steps=6):
    app.on_mouse_down(start, button, NO_MODS)
    for k in range(1, steps + 1):
        pos = (
            round(start[0] + (end[0] - start[0]) * k / steps),
            round(start[1] + (end[1] - start[1]) * k / steps),
        )
        app.on_mouse_motion(pos, NO_MODS)
    app.on_mouse_up(end, button)


def edge_midpoint_px(app, edge):
    a = app.corner_screen(edge)
    b = app.corner_screen((edge + 1) % len(app.spec.corners))
    return round((a[0] + b[0]) / 2), round((a[1] + b[1]) / 2)


def banner_color(app):
    app.draw()
    return tuple(app.screen.get_at(app.banner_rect.center))[:3]


def key(app, k, mods=NO_MODS):
    app.on_key(k, mods)


# -- the model ---------------------------------------------------------------------------------


def test_the_new_track_template_is_valid():
    assert validate_spec(ed.NEW_SPEC).ok


def test_undo_and_redo_restore_the_exact_spec():
    spec = load_spec(EXAMPLE)
    model = EditorModel(spec)
    states = [spec]
    model.move_corner(2, 410.0, 95.0)
    states.append(model.spec)
    model.insert_corner(0, 150.0, -5.0)
    model.end_drag()
    states.append(model.spec)
    model.set_width(15.0)
    states.append(model.spec)
    model.delete_corner(4)
    states.append(model.spec)
    assert len({s for s in states}) == len(states)
    for expected in reversed(states[:-1]):
        assert model.undo()
        assert model.spec == expected
    assert not model.undo() and not model.dirty
    for expected in states[1:]:
        assert model.redo()
        assert model.spec == expected
    assert not model.redo()


def test_a_new_edit_clears_the_redo_history():
    model = EditorModel(load_spec(EXAMPLE))
    model.set_width(14.0)
    model.undo()
    assert model.can_redo
    model.move_corner(0, 1.0, 1.0)
    assert not model.can_redo


def test_the_history_is_bounded():
    model = EditorModel(load_spec(EXAMPLE))
    for i in range(ed.HISTORY_LIMIT + 50):
        model.set_name(f"Circuito {i}")
    assert model.undo_depth == ed.HISTORY_LIMIT
    for _ in range(ed.HISTORY_LIMIT):
        assert model.undo()
    assert not model.undo()
    assert model.spec.name == "Circuito 49"  # the 50 oldest steps were dropped


def test_repeated_tweaks_of_one_thing_are_one_undo_step():
    spec = load_spec(EXAMPLE)
    model = EditorModel(spec)
    for k in range(5):
        model.set_radius(1, spec.corners[1][2] + 2 * (k + 1), now=10.0 + 0.1 * k)
    assert model.spec.corners[1][2] == spec.corners[1][2] + 10
    assert model.undo_depth == 1
    model.set_radius(2, 20.0, now=10.5)  # another corner: a new step
    assert model.undo_depth == 2
    model.undo()
    model.undo()
    assert model.spec == spec


def test_a_drag_is_one_undo_step_and_undo_is_ignored_during_it():
    spec = load_spec(EXAMPLE)
    model = EditorModel(spec)
    model.begin_drag()
    for k in range(20):
        model.move_corner(3, 330.0 + k, 190.0)
    assert not model.undo()  # mid-gesture
    assert model.end_drag()
    assert model.undo_depth == 1
    model.undo()
    assert model.spec == spec


def test_the_radius_stays_within_what_the_validator_accepts():
    model = EditorModel(load_spec(EXAMPLE))
    model.set_radius(0, 1.0)
    assert model.spec.corners[0][2] == ed.MIN_RADIUS
    model.set_radius(0, 10_000.0)
    assert model.spec.corners[0][2] == ed.MAX_RADIUS
    model.set_width(100.0)
    assert model.spec.width == ed.MAX_WIDTH
    model.set_width(0.0)
    assert model.spec.width == ed.MIN_WIDTH


def test_corner_positions_are_kept_to_a_tenth_of_a_metre():
    model = EditorModel(load_spec(EXAMPLE))
    model.move_corner(0, 1.23456789, -9.87654321)
    assert model.spec.corners[0][:2] == (1.2, -9.9)
    model.move_corner(0, 1e9, -1e9)  # never beyond the world the validator accepts
    assert model.spec.corners[0][:2] == (ed.MAX_COORDINATE, -ed.MAX_COORDINATE)


def test_the_viewport_zoom_keeps_the_point_under_the_cursor():
    view = Viewport(10.0, 20.0, 2.0, (800, 600))
    before = view.to_world(123, 456)
    view.zoom_at((123, 456), 1.7)
    after = view.to_world(123, 456)
    assert after == pytest.approx(before)
    assert view.zoom == pytest.approx(3.4)
    x, y = view.to_screen(*view.to_world(300, 200))
    assert (x, y) == pytest.approx((300, 200))


def test_file_names_from_track_names():
    assert slugify("Bahía del Faro") == "bahia-del-faro"
    assert slugify("  ¡¡¡ ") == "circuito"
    assert resolve_save_path("Mi pista", Path("T")) == Path("T/mi-pista.json")
    assert resolve_save_path("otro/sitio.json", Path("T")) == Path("otro/sitio.json")
    assert resolve_save_path("x\\y", Path("T")) == Path("x\\y.json")


# -- mouse editing ---------------------------------------------------------------------------


def test_dragging_a_corner_moves_it_and_nothing_else(make_app):
    app = make_app()
    before = app.spec
    start = corner_px(app, 3)
    drag(app, start, (start[0] + 30, start[1] - 20))
    x0, y0, r0 = before.corners[3]
    x1, y1, r1 = app.spec.corners[3]
    assert x1 == pytest.approx(x0 + 30 / app.view.zoom, abs=0.1)
    assert y1 == pytest.approx(y0 + 20 / app.view.zoom, abs=0.1)
    assert r1 == r0
    assert app.spec.corners[:3] == before.corners[:3]
    assert app.spec.corners[4:] == before.corners[4:]
    assert app.model.dirty and app.model.undo_depth == 1
    assert app.selected == 3
    app.undo()
    assert app.spec == before and not app.model.dirty


def test_clicking_an_edge_inserts_a_corner_there(make_app):
    app = make_app()
    before = app.spec
    mid = edge_midpoint_px(app, 0)
    assert app.pick_corner(mid) is None
    app.on_mouse_down(mid, 1, NO_MODS)
    app.on_mouse_up(mid, 1)
    assert len(app.spec.corners) == len(before.corners) + 1
    x, y, radius = app.spec.corners[1]
    (ax, ay, ar), (bx, by, br) = before.corners[0], before.corners[1]
    # it lies on the old edge and the old corners are all still there, in order
    assert (x - ax) * (by - ay) - (y - ay) * (bx - ax) == pytest.approx(0.0, abs=0.1 * 330)
    assert app.spec.corners[0] == before.corners[0]
    assert app.spec.corners[2:] == before.corners[1:]
    assert radius == round((ar + br) / 2)
    assert app.selected == 1
    app.settle()
    assert app.verdict == "valid"  # a corner on a straight edge changes nothing
    app.undo()
    assert app.spec == before


def test_dragging_from_an_edge_inserts_and_places_the_corner_in_one_undo_step(make_app):
    app = make_app()
    before = app.spec
    mid = edge_midpoint_px(app, 1)
    drag(app, mid, (mid[0] + 25, mid[1] + 25))
    assert len(app.spec.corners) == len(before.corners) + 1
    assert app.model.undo_depth == 1
    app.undo()
    assert app.spec == before


def test_a_right_click_or_delete_removes_a_corner(make_app):
    app = make_app()
    before = app.spec
    app.on_mouse_down(corner_px(app, 2), 3, NO_MODS)
    app.on_mouse_up(corner_px(app, 2), 3)
    assert app.spec.corners == before.corners[:2] + before.corners[3:]
    app.undo()
    assert app.spec == before
    app.on_mouse_down(corner_px(app, 5), 1, NO_MODS)
    app.on_mouse_up(corner_px(app, 5), 1)
    key(app, pygame.K_DELETE)
    assert app.spec.corners == before.corners[:5] + before.corners[6:]
    assert app.selected is None


def test_a_track_never_goes_below_three_corners(make_app):
    app = make_app()
    app.model.apply(
        type(app.spec)(app.spec.name, app.spec.width, app.spec.corners[:MIN_CORNERS]), None
    )
    app.changed()
    app.on_mouse_down(corner_px(app, 0), 3, NO_MODS)
    assert len(app.spec.corners) == MIN_CORNERS
    assert "al menos" in app.message


def test_the_wheel_over_a_corner_changes_its_radius_and_elsewhere_zooms(make_app):
    app = make_app()
    before = app.spec
    pos = corner_px(app, 4)
    zoom = app.view.zoom
    app.on_wheel(pos, 1, NO_MODS)
    app.on_wheel(pos, 1, NO_MODS)
    assert app.spec.corners[4][2] == before.corners[4][2] + 2 * ed.RADIUS_STEP
    assert app.view.zoom == zoom
    app.on_wheel(pos, -1, SHIFT)
    assert app.spec.corners[4][2] == before.corners[4][2] + 2 * ed.RADIUS_STEP - 5 * ed.RADIUS_STEP
    assert app.model.undo_depth == 1  # the three notches are one step
    app.undo()
    assert app.spec == before
    empty = (60, 700)
    assert app.pick_corner(empty) is None
    world = app.view.to_world(*empty)
    app.on_wheel(empty, 2, NO_MODS)
    assert app.view.zoom > zoom
    assert app.view.to_world(*empty) == pytest.approx(world)
    assert app.spec == before


def test_pan_with_the_middle_button_and_with_space(make_app):
    app = make_app()
    spec = app.spec
    cx, cy, zoom = app.view.cx, app.view.cy, app.view.zoom
    drag(app, (400, 400), (460, 380), button=2)
    assert app.view.cx == pytest.approx(cx - 60 / zoom)
    assert app.view.cy == pytest.approx(cy - 20 / zoom)
    cx, cy = app.view.cx, app.view.cy
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, mod=0))
    start = corner_px(app, 0)  # even over a corner: space means pan
    drag(app, start, (start[0] - 40, start[1]))
    app.handle_event(pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE, mod=0))
    assert app.view.cx == pytest.approx(cx + 40 / zoom)
    assert app.spec == spec  # panning never edits


def test_zoom_keys_and_fit(make_app):
    app = make_app()
    zoom = app.view.zoom
    key(app, pygame.K_PLUS)
    assert app.view.zoom == pytest.approx(zoom * ed.ZOOM_KEY_FACTOR)
    key(app, pygame.K_MINUS)
    key(app, pygame.K_MINUS)
    assert app.view.zoom < zoom
    app.view.pan(500, 500)
    key(app, pygame.K_f)
    assert app.view.zoom == pytest.approx(zoom)
    assert app.view.cx == pytest.approx(app.view.cx)
    assert all(app.canvas.collidepoint(corner_px(app, i)) for i in range(len(app.spec.corners)))


def test_arrow_keys_nudge_the_selected_corner(make_app):
    app = make_app()
    app.on_mouse_down(corner_px(app, 1), 1, NO_MODS)
    app.on_mouse_up(corner_px(app, 1), 1)
    x, y, _ = app.spec.corners[1]
    key(app, pygame.K_RIGHT)
    key(app, pygame.K_UP, SHIFT)
    assert app.spec.corners[1][:2] == (x + 1, y + 10)
    assert app.model.undo_depth == 1


def test_width_and_seed_keys_and_buttons(make_app):
    app = make_app()
    width, seed = app.spec.width, app.spec.decoration_seed
    key(app, pygame.K_PAGEUP)
    assert app.spec.width == width + 1
    key(app, pygame.K_d)
    key(app, pygame.K_d, SHIFT)
    key(app, pygame.K_d, SHIFT)
    assert app.spec.decoration_seed == seed - 1
    app.draw()
    app.on_mouse_down(app.buttons["width-"].center, 1, NO_MODS)
    app.on_mouse_down(app.buttons["seed+"].center, 1, NO_MODS)
    assert app.spec.width == width
    assert app.spec.decoration_seed == seed
    key(app, pygame.K_PAGEDOWN)
    key(app, pygame.K_PAGEDOWN)
    assert app.spec.width == width - 2


# -- validation: live and complete --------------------------------------------------------------


def test_the_verdict_turns_red_on_a_self_crossing_edit_and_green_after_undo(make_app):
    app = make_app()
    app.settle()
    assert app.verdict == "valid"
    assert banner_color(app) == ed.VALID
    start = corner_px(app, 0)
    target = tuple(round(v) for v in app.view.to_screen(330, 250))
    app.on_mouse_down(start, 1, NO_MODS)
    app.on_mouse_motion(target, NO_MODS)
    app.update()  # mid-drag: the quick analysis already sees the crossing
    assert app.verdict == "invalid"
    assert banner_color(app) == ed.INVALID
    app.on_mouse_up(target, 1)
    app.settle()
    (issue,) = app.analysis.issues
    assert "se cruza a sí mismo" in issue.message and len(issue.corners) == 4
    assert banner_color(app) == ed.INVALID
    app.undo()
    app.settle()
    assert app.verdict == "valid"
    assert banner_color(app) == ed.VALID
    assert app.analysis.issues == ()


def test_the_panel_lists_the_validators_messages(make_app):
    app = make_app()
    app.model.move_corner(0, 330.0, 250.0)
    app.changed()
    app.settle()
    app.draw()
    shown = {index for _, index in app._issue_rows}
    assert shown == set(range(len(app.analysis.issues)))
    assert app.analysis.issues[0].message in validate_spec(app.spec).problems
    app.on_mouse_down(app._issue_rows[0][0].center, 1, NO_MODS)  # click to focus the issue
    assert app.selected in app.analysis.issues[0].corners


def test_the_slow_validation_waits_for_the_edits_to_stop(make_app):
    app = make_app()
    app.settle()
    app.model.move_corner(3, 335.0, 192.0)
    app.changed()
    t = app._last_change
    app.update(t)
    assert app.analysis.complete is False and app.verdict == "pending"
    app.update(t + ed.FULL_DEBOUNCE / 2)
    assert app.analysis.complete is False
    app.update(t + ed.FULL_DEBOUNCE + 0.01)
    assert app.analysis.complete and app.verdict == "valid"
    # never in the middle of a drag, however long it lasts
    app.on_mouse_down(corner_px(app, 3), 1, NO_MODS)
    app.on_mouse_motion((corner_px(app, 3)[0] + 5, corner_px(app, 3)[1]), NO_MODS)
    app.update(t + 100.0)
    assert app.analysis.complete is False
    app.on_mouse_up(corner_px(app, 3), 1)


def test_a_slow_quick_analysis_is_throttled_while_dragging(make_app):
    app = make_app()
    app.model.begin_drag()
    app.model.move_corner(3, 335.0, 192.0)
    app.changed()
    app.quick_cost = 10 * ed.QUICK_BUDGET
    app._last_quick_end = 100.0
    app.update(100.0 + 2 * app.quick_cost)
    assert app._stale  # too soon: waits for 3x its own cost
    app.update(100.0 + 3.1 * app.quick_cost)
    assert not app._stale
    app.model.end_drag()


def test_the_preview_is_the_loaders_centerline(make_app, tmp_path):
    app = make_app()
    app.settle()
    path = tmp_path / "preview.json"
    assert app.save(path)
    assert (app.analysis.points == load_track_file(path).points).all()


# -- saving ------------------------------------------------------------------------------------


def test_saving_a_valid_track_writes_a_file_the_loader_accepts(make_app, tmp_path):
    app = make_app()
    start = corner_px(app, 6)
    drag(app, start, (start[0] + 10, start[1] + 12))
    target = tmp_path / "tracks" / "editado.json"
    assert app.save(target)
    assert target.exists() and not app.model.dirty
    assert app.path == target
    raw = target.read_bytes()
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf") and raw.endswith(b"\n")
    assert raw.decode("utf-8") == spec_to_json(app.spec)
    assert load_spec(target) == app.spec
    track = load_track_file(target)
    assert track.name == app.spec.name
    assert list(target.parent.iterdir()) == [target]  # no temporary file left behind
    # saving again to the same file is fine (it is ours); the title loses its asterisk
    assert app.save()
    assert "Guardado" in app.message


def test_ctrl_s_saves_to_the_opened_file_and_a_new_track_to_tracks(make_app, tmp_path):
    copy = tmp_path / "copia.json"
    copy.write_bytes(EXAMPLE.read_bytes())
    app = make_app(copy)
    key(app, pygame.K_PAGEUP)
    assert app.model.dirty
    key(app, pygame.K_s, CTRL)
    assert load_spec(copy).width == app.spec.width and not app.model.dirty
    new = make_app(ed.NEW_SPEC, path=None, new=True)
    assert new.model.dirty  # never saved
    key(new, pygame.K_s, CTRL)
    target = tmp_path / "tracks" / "nuevo-circuito.json"
    assert load_track_file(target).name == "Nuevo circuito"
    assert new.path == target


def test_saving_an_invalid_track_is_refused_and_the_file_is_untouched(make_app, tmp_path):
    copy = tmp_path / "copia.json"
    copy.write_bytes(EXAMPLE.read_bytes())
    before = copy.read_bytes()
    app = make_app(copy)
    app.model.move_corner(0, 330.0, 250.0)
    app.changed()  # no update(): save must not trust a stale analysis
    assert not app.save()
    assert "No se guarda" in app.message and "no es válido" in app.message
    assert copy.read_bytes() == before
    assert app.model.dirty
    key(app, pygame.K_s, CTRL)
    assert copy.read_bytes() == before
    assert not app.save(tmp_path / "otro.json")
    assert not (tmp_path / "otro.json").exists()


def test_saving_never_overwrites_a_file_that_is_not_the_one_being_edited(make_app, tmp_path):
    app = make_app(ed.NEW_SPEC, path=None, new=True)
    target = tmp_path / "tracks" / "nuevo-circuito.json"
    target.parent.mkdir()
    target.write_text("{}", encoding="utf-8")
    assert not app.save()
    assert "ya existe" in app.message
    assert target.read_text(encoding="utf-8") == "{}"


def test_an_unwritable_target_is_reported_not_raised(make_app, tmp_path):
    app = make_app()
    blocker = tmp_path / "ocupado"
    blocker.write_text("x", encoding="utf-8")
    assert not app.save(blocker / "dentro.json")  # a "folder" that is a file
    assert "No se puede escribir" in app.message


def test_save_as_asks_for_a_name_in_the_window(make_app, tmp_path):
    app = make_app()
    key(app, pygame.K_s, CTRL | SHIFT)
    assert app.prompt is not None
    for _ in range(200):
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_BACKSPACE, mod=0))
    app.handle_event(pygame.event.Event(pygame.TEXTINPUT, text="Mi pista añeja"))
    app.draw()  # the prompt is drawn without errors
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0))
    target = tmp_path / "tracks" / "mi-pista-aneja.json"
    assert app.prompt is None and target.exists()
    assert load_spec(target) == app.spec
    # Esc cancels
    key(app, pygame.K_s, CTRL | SHIFT)
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0))
    assert app.prompt is None and app.running is False


def test_f2_renames_the_track(make_app):
    app = make_app()
    before = app.spec
    key(app, pygame.K_F2)
    for _ in range(60):
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_BACKSPACE, mod=0))
    app.handle_event(pygame.event.Event(pygame.TEXTINPUT, text="Valle Ñandú"))
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0))
    assert app.spec.name == "Valle Ñandú"
    app.undo()
    assert app.spec == before
    key(app, pygame.K_F2)
    for _ in range(60):
        app.prompt.backspace()
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0))
    assert "vacío" in app.message and app.spec == before


def test_leaving_with_unsaved_changes_asks_twice(make_app):
    app = make_app()
    app.running = True
    key(app, pygame.K_ESCAPE)
    assert app.running is False  # nothing to lose
    app.running = True
    key(app, pygame.K_PAGEUP)
    key(app, pygame.K_ESCAPE)
    assert app.running and "sin guardar" in app.message
    key(app, pygame.K_ESCAPE)
    assert app.running is False


# -- test drive --------------------------------------------------------------------------------


def test_test_drive_launches_the_game_on_a_saved_file(make_app, launched, tmp_path):
    copy = tmp_path / "copia.json"
    copy.write_bytes(EXAMPLE.read_bytes())
    app = make_app(copy)
    key(app, pygame.K_F5)
    ((command, process),) = launched
    assert command[-2:] == ["--track-file", str(copy)]
    assert "kartlab.show.drive" in command[2]
    key(app, pygame.K_F5)  # one at a time
    assert len(launched) == 1 and "en marcha" in app.message
    process.running = False
    app.update()
    key(app, pygame.K_t)
    assert len(launched) == 2


def test_test_drive_of_unsaved_changes_uses_a_temporary_copy(make_app, launched, tmp_path):
    copy = tmp_path / "copia.json"
    copy.write_bytes(EXAMPLE.read_bytes())
    before = copy.read_bytes()
    app = make_app(copy)
    key(app, pygame.K_PAGEUP)
    key(app, pygame.K_F5)
    ((command, process),) = launched
    temp = Path(command[-1])
    assert temp != copy and temp.exists()
    assert load_spec(temp) == app.spec
    assert copy.read_bytes() == before and app.model.dirty
    process.running = False
    app.update()
    assert not temp.exists()


def test_test_drive_needs_a_valid_track(make_app, launched):
    app = make_app()
    app.model.move_corner(0, 330.0, 250.0)
    app.changed()
    key(app, pygame.K_F5)
    assert launched == [] and "válido" in app.message


# -- format 2 -------------------------------------------------------------------------------


def test_a_figure_eight_opens_valid_and_an_unchanged_save_is_byte_identical(make_app, tmp_path):
    original = load_spec(EIGHT)
    app = make_app(EIGHT)
    app.settle()
    assert app.verdict == "valid" and app.analysis.report.crossings == 1
    assert app.spec.crossings == original.crossings == ((0.0, 0.0),)
    assert app.spec.start == original.start
    copy = tmp_path / "ocho-copia.json"
    assert app.save(copy)  # the opened file itself is never touched by this test
    assert copy.read_bytes() == EIGHT.read_bytes()


def test_the_crossings_and_start_survive_editing_and_saving(make_app, tmp_path):
    original = load_spec(EIGHT)
    app = make_app(EIGHT)
    start = corner_px(app, 3)
    drag(app, start, (start[0] + 6, start[1] + 6))
    assert app.spec.corners[3] != original.corners[3]
    target = tmp_path / "tracks" / "ocho-editado.json"
    assert app.save(target)
    saved = load_spec(target)
    assert saved.crossings == original.crossings and saved.start == original.start
    assert json.loads(target.read_text(encoding="utf-8"))["format"] == 2
    assert load_track_file(target).crossings  # and it loads as a track with a crossing


def test_moving_a_figure_eight_so_that_the_declared_crossing_vanishes_is_red(make_app, tmp_path):
    app = make_app(EIGHT)
    app.model.move_corner(0, -60.0, -130.0)
    app.changed()
    app.settle()
    assert app.verdict == "invalid"
    assert any("cruce" in i.message for i in app.analysis.issues)
    assert not app.save(tmp_path / "x.json")
    app.draw()


def test_a_format_2_file_shows_its_crossings_in_the_panel(make_app):
    app = make_app(EIGHT)
    app.draw()
    assert app.spec.crossings


# -- opening files and the command line --------------------------------------------------------


def test_an_invalid_but_well_formed_file_opens_so_it_can_be_fixed(make_app, tmp_path):
    broken = load_spec(EXAMPLE)
    corners = list(broken.corners)
    corners[0] = (330.0, 250.0, corners[0][2])
    path = tmp_path / "roto.json"
    path.write_text(
        spec_to_json(type(broken)(broken.name, broken.width, tuple(corners))), encoding="utf-8"
    )
    app = make_app(path)
    app.settle()
    assert app.verdict == "invalid"
    app.undo()  # nothing to undo yet
    start = corner_px(app, 0)
    target = tuple(round(v) for v in app.view.to_screen(0, 0))
    drag(app, start, target)
    app.settle()
    assert app.verdict == "valid"
    assert app.save()


def test_headless_run_draws_frames_and_writes_a_screenshot(make_app, tmp_path):
    app = make_app()
    out = tmp_path / "sub" / "editor.png"
    app.run(max_frames=60)
    app.save_screenshot(out)
    image = pygame.image.load(str(out))
    assert image.get_size() == (1280, 800)
    assert out.stat().st_size > 5_000


def test_main_runs_60_frames_and_saves_a_screenshot(tmp_path):
    out = tmp_path / "shot.png"
    ed.main([str(EXAMPLE), "--frames", "60", "--size", "1000x700", "--screenshot", str(out)])
    assert pygame.image.load(str(out)).get_size() == (1000, 700)


def test_main_new_starts_from_the_template(tmp_path):
    out = tmp_path / "new.png"
    ed.main(["--new", "--frames", "5", "--size", "900x600", "--screenshot", str(out)])
    assert out.exists()
    ed.main(["--frames", "2", "--size", "900x600"])  # no arguments means a new track too


def test_main_refuses_to_start_a_new_track_over_an_existing_file():
    with pytest.raises(SystemExit) as error:
        ed.main([str(EXAMPLE), "--new", "--frames", "1"])
    assert "ya existe" in str(error.value)


def test_main_reports_an_unreadable_file(tmp_path):
    bad = tmp_path / "mal.json"
    bad.write_text("{ no es json", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        ed.main([str(bad), "--frames", "1"])
    assert "No se puede abrir" in str(error.value)
    with pytest.raises(SystemExit):
        ed.main([str(tmp_path / "no-existe.json"), "--frames", "1"])
    with pytest.raises(TrackFileError):
        load_spec(bad)


def test_help_and_window_events_do_not_break_drawing(make_app):
    app = make_app()
    key(app, pygame.K_h)
    assert app.show_help
    app.draw()
    key(app, pygame.K_ESCAPE)  # closes the help first
    assert not app.show_help
    app.handle_event(pygame.event.Event(pygame.WINDOWRESIZED, x=1000, y=700))
    app.handle_event(pygame.event.Event(pygame.QUIT))


def test_a_smoke_run_of_a_drag_session_never_breaks(make_app):
    """60 frames in which the mouse works the corners hard: no exception, a verdict at the end."""
    app = make_app()
    app.running = True
    rng_moves = [(40, 30), (-80, 10), (15, -90), (200, 120), (-150, -60), (5, 5)]
    for frame in range(60):
        if frame % 10 == 0:
            i = (frame // 10) % len(app.spec.corners)
            start = corner_px(app, i)
            dx, dy = rng_moves[(frame // 10) % len(rng_moves)]
            app.on_mouse_down(start, 1, NO_MODS)
            app.on_mouse_motion((start[0] + dx, start[1] + dy), NO_MODS)
        app.update(time.perf_counter())
        app.draw()
        if frame % 10 == 9:
            app.on_mouse_up(app.mouse, 1)
    app.settle()
    assert app.verdict in ("valid", "invalid")
    assert app.analysis.complete
    for _ in range(20):
        app.undo()
    app.settle()
    assert app.spec == load_spec(EXAMPLE) and app.verdict == "valid"


def test_a_second_button_or_a_key_during_a_drag_cannot_break_it(make_app):
    app = make_app()
    before = app.spec
    start = corner_px(app, 2)
    app.on_mouse_down(start, 1, NO_MODS)
    app.on_mouse_down(corner_px(app, 5), 3, NO_MODS)  # right click mid-drag: ignored
    key(app, pygame.K_DELETE)
    key(app, pygame.K_PAGEUP)
    app.on_mouse_motion((start[0] + 10, start[1]), NO_MODS)
    app.on_mouse_up((start[0] + 10, start[1]), 1)
    assert len(app.spec.corners) == len(before.corners)
    assert app.spec.width == before.width
    assert app.spec.corners[2] != before.corners[2]


def test_the_window_layout_is_not_rebuilt_every_frame(make_app, monkeypatch):
    app = make_app()
    calls = []
    original = app._build_fonts
    monkeypatch.setattr(app, "_build_fonts", lambda: (calls.append(1), original()))
    app.run(max_frames=20)
    assert calls == []


def test_toasts_never_overflow_the_canvas_even_with_a_long_path(make_app, tmp_path):
    app = make_app()
    deep = tmp_path / ("carpeta-larga-" * 3) / ("otra-" * 4)
    assert app.save(deep / "circuito.json")
    assert app.message.startswith(
        "Guardado circuito.json"
    )  # only the file name outside the project
    app.say("Guardado " + "x" * 400, "ok")  # a word wider than the canvas is cut, not overflowed
    for line in ed._wrap(app.font, app.message, app.canvas.width - 80):
        assert app.font.size(line)[0] <= app.canvas.width - 80
    app.draw()
    assert tuple(app.screen.get_at((app.panel_rect.x + 30, 20)))[:3] != ed.GRASS
    app.say("Abierto tracks/ejemplo.json")
    assert app._toast_path(EXAMPLE.resolve()) == "tracks/ejemplo.json"
