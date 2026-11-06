"""The main menu, headless: synthetic keyboard and mouse events, one display frame per step.

Nothing here launches a real training: the menu gets a fake ``spawn``.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pygame
import pytest

from kartlab.common.paths import REPO_ROOT
from kartlab.show import hub as hub_module
from kartlab.show.cli import KART_COLOR_NAMES, kart_color_rgb, parse_kart_color
from kartlab.show.hub import ENTRIES, STEPS, KartLabApp, Step, main
from kartlab.show.menu import Entry
from kartlab.show.settings import Settings
from kartlab.train.config import PPOParams, TrainConfig

TRACKS = REPO_ROOT / "tracks"


class FakeProcess:
    pid = 4242

    def __init__(self) -> None:
        self.code: int | None = None
        self.terminated = False

    def poll(self):
        return self.code

    def terminate(self) -> None:
        self.terminated = True
        self.code = 1


class FakeSpawn:
    """Records the launches instead of starting a trainer."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self.process = FakeProcess()

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), kwargs))
        return self.process


@pytest.fixture
def tracks_dir(tmp_path):
    folder = tmp_path / "tracks"
    folder.mkdir()
    shutil.copy(TRACKS / "ejemplo.json", folder / "ejemplo.json")
    return folder


@pytest.fixture
def make_hub(tmp_path, tracks_dir):
    created = []

    def make(**kwargs) -> KartLabApp:
        options = {
            "scale": 2,
            "realtime": False,
            "background": False,
            "activity_frames": 3,
            "settings_path": tmp_path / "settings.json",
            "root": tmp_path,
            "runs_dir": tmp_path / "runs",
            "logs_dir": tmp_path / "logs",
            "tracks_dir": tracks_dir,
            "configs_dir": tmp_path / "configs",
            "spawn": FakeSpawn(),
            "opener": lambda path: None,
        }
        app = KartLabApp(**{**options, **kwargs})
        created.append(app)
        return app

    yield make
    pygame.quit()


def key(app, k):
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode=""))


def motion(app, pos):
    app.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=pos, rel=(0, 0), buttons=(0, 0, 0)))


def click(app, pos):
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1))


def select(app, entry_key):
    """Move the main list's selection to an entry."""
    app.main_screen.selected = next(i for i, e in enumerate(app.entries) if e.key == entry_key)


def write_config(folder: Path, name: str, steps: int) -> Path:
    folder.mkdir(exist_ok=True)
    path = folder / f"{name}.yaml"
    path.write_text(
        f"# {name}: una configuración de prueba\nname: {name}\ntotal_steps: {steps}\n"
        "n_envs: 1\nvec_env: dummy\n",
        encoding="utf-8",
    )
    return path


# -- start-up ---------------------------------------------------------------------------------


def test_importing_the_menu_does_not_load_torch():
    code = (
        "import sys, kartlab.show.hub;"
        "print(sorted(m for m in ('torch', 'stable_baselines3') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", check=True
    )
    assert result.stdout.strip().endswith("[]")


def test_jugar_bat_is_a_crlf_script_that_runs_the_menu():
    data = (REPO_ROOT / "Jugar.bat").read_bytes()
    assert data.count(b"\r\n") == data.count(b"\n")  # every line ends in CRLF: double click works
    lines = data.decode("utf-8").splitlines()
    assert lines == ["@echo off", 'cd /d "%~dp0"', "uv run kartlab", "if errorlevel 1 pause"]


def test_gitattributes_keeps_bat_files_in_crlf():
    attributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.bat text eol=crlf" in attributes


def test_the_command_and_python_dash_m_start_the_menu():
    project = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'kartlab = "kartlab.show.hub:main"' in project
    main_module = (REPO_ROOT / "src" / "kartlab" / "__main__.py").read_text(encoding="utf-8")
    assert "from kartlab.show.hub import main" in main_module


def test_settings_and_logs_are_ignored_by_git():
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert "settings.json" in ignored and "logs/" in ignored


def test_the_menu_opens_on_the_first_frame_and_draws_something(make_hub):
    app = make_hub()
    app.run(2)
    assert app.time_to_menu is not None and app.time_to_menu < 2.0
    assert app.current is app.main_screen
    pixels = pygame.surfarray.array3d(app.screen).reshape(-1, 3)
    assert len(np.unique(pixels, axis=0)) > 20


def test_screenshot_option_of_the_command(tmp_path, monkeypatch):
    monkeypatch.setattr(hub_module, "SETTINGS_PATH", tmp_path / "settings.json")
    out = tmp_path / "menu.png"
    main(["--scale", "2", "--frames", "30", "--screenshot", str(out)])
    assert out.stat().st_size > 0


# -- settings -----------------------------------------------------------------------------------


def test_settings_round_trip_and_defaults(tmp_path):
    path = tmp_path / "settings.json"
    assert Settings.load(path) == Settings()  # no file: the defaults
    saved = Settings(scale=3, fullscreen=True, light="night", camera="top", kart_ramp=5)
    saved.last_track, saved.last_run = "eval:7", "2026-01-01_0000_x"
    assert saved.save(path)
    assert Settings.load(path) == saved
    assert json.loads(path.read_text(encoding="utf-8"))["scale"] == 3
    Settings(scale=None).save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["scale"] == "auto"
    assert Settings.load(path).scale is None


@pytest.mark.parametrize("text", ["", "{", "[1, 2]", '{"scale": 99', "\xff\xfe", "null"])
def test_a_corrupt_settings_file_is_ignored(tmp_path, text):
    path = tmp_path / "settings.json"
    path.write_bytes(text.encode("latin-1"))
    assert Settings.load(path) == Settings()


def test_invalid_values_fall_back_one_by_one(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"scale": 9, "fullscreen": 1, "light": "night", "camera": 3, "kart_ramp": 8}),
        encoding="utf-8",
    )
    settings = Settings.load(path)
    assert settings.scale is None and settings.fullscreen is False
    assert settings.light == "night"  # the valid one survives
    assert settings.camera == "auto" and settings.kart_ramp == 0


# -- navigation ---------------------------------------------------------------------------------


def test_keyboard_moves_the_selection_and_wraps(make_hub):
    app = make_hub()
    menu = app.main_screen
    assert app.entries[menu.selected].key == "contrarreloj"
    key(app, pygame.K_DOWN)
    assert app.entries[menu.selected].key == "conduccion"
    key(app, pygame.K_UP)
    key(app, pygame.K_UP)
    assert app.entries[menu.selected].key == "salir"  # wrapped


def test_the_mouse_selects_by_hovering_and_activates_with_a_click(make_hub):
    app = make_hub()
    menu = app.main_screen
    target = next(i for i, e in enumerate(app.entries) if e.key == "opciones")
    motion(app, menu.rects[target].center)
    assert menu.selected == target
    click(app, menu.rects[target].center)
    assert app.current.__class__.__name__ == "OptionsScreen"
    key(app, pygame.K_ESCAPE)
    assert app.current is app.main_screen


def test_the_wheel_moves_the_selection(make_hub):
    app = make_hub()
    app.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1))
    assert app.entries[app.main_screen.selected].key == "conduccion"


def test_escape_in_the_main_menu_and_the_exit_entry_close_it(make_hub):
    app = make_hub()
    app.run(1)
    key(app, pygame.K_ESCAPE)
    assert app.running is False
    other = make_hub()
    other.running = True
    select(other, "salir")
    key(other, pygame.K_RETURN)
    assert other.running is False


def test_entries_without_a_run_explain_why_instead_of_opening(make_hub):
    app = make_hub()
    select(app, "ver")
    key(app, pygame.K_RETURN)
    assert app.dialog is not None and "Entrena antes a la IA" in app.dialog.text
    assert app.current is app.main_screen
    key(app, pygame.K_RETURN)  # any key closes the message
    assert app.dialog is None


def test_the_play_flow_goes_kart_then_circuit_and_escape_steps_back(make_hub):
    app = make_hub()
    key(app, pygame.K_RETURN)  # Contrarreloj
    assert type(app.current).__name__ == "KartScreen"
    key(app, pygame.K_RIGHT)
    key(app, pygame.K_RETURN)  # chooses the second colour
    assert type(app.current).__name__ == "TrackScreen"
    assert app.settings.kart_ramp == 1
    key(app, pygame.K_ESCAPE)
    assert type(app.current).__name__ == "KartScreen"
    key(app, pygame.K_ESCAPE)
    assert app.current is app.main_screen


def test_the_kart_is_chosen_with_arrows_or_the_mouse(make_hub):
    app = make_hub()
    key(app, pygame.K_RETURN)
    screen = app.current
    for _ in range(9):  # more than eight: wraps around
        key(app, pygame.K_RIGHT)
    assert screen.selected == 1
    key(app, pygame.K_LEFT)
    key(app, pygame.K_LEFT)
    assert screen.selected == 7
    motion(app, screen.swatches[3].center)
    assert screen.selected == 3
    click(app, screen.swatches[3].center)
    assert app.settings.kart_ramp == 3 and type(app.current).__name__ == "TrackScreen"
    saved = json.loads(app.settings_path.read_text(encoding="utf-8"))
    assert saved["kart_ramp"] == 3


@pytest.mark.parametrize("ramp", range(8))
def test_the_kart_preview_draws_every_colour(make_hub, ramp):
    app = make_hub(scale=3)
    app.settings.kart_ramp = ramp
    key(app, pygame.K_RETURN)
    app.run(40)  # about a third of a turn
    wanted = kart_color_rgb(ramp)
    pixels = pygame.surfarray.array3d(app.screen).reshape(-1, 3)
    assert (pixels == wanted).all(axis=1).sum() > 200  # the swatch alone is smaller than the kart


def test_the_circuit_cards_list_showcases_files_random_and_evaluation(make_hub):
    app = make_hub()
    key(app, pygame.K_RETURN)
    key(app, pygame.K_RETURN)
    screen = app.current
    keys = [o.key for o in screen.options]
    assert keys[:3] == ["harbor", "canyon", "plateau"]
    assert "file:tracks/ejemplo.json" in keys
    assert keys[-2:] == ["random", "eval:1"]
    option = screen.options[keys.index("file:tracks/ejemplo.json")]
    assert option.title == "Bahía del Faro" and option.track.length > 500  # name and length


def test_pageup_changes_the_evaluation_circuit_and_r_the_random_one(make_hub):
    app = make_hub()
    key(app, pygame.K_RETURN)
    key(app, pygame.K_RETURN)
    screen = app.current
    screen.selected = len(screen.options) - 1
    key(app, pygame.K_PAGEUP)
    assert screen.options[-1].key == "eval:2"
    key(app, pygame.K_PAGEDOWN)
    key(app, pygame.K_PAGEDOWN)
    assert screen.options[-1].key == "eval:20"  # wraps
    screen.selected = len(screen.options) - 2
    before = screen.options[-2].seed
    key(app, pygame.K_r)
    assert screen.options[-2].seed != before


def test_an_invalid_circuit_file_shows_the_reason_in_a_box(make_hub, tracks_dir):
    bad = '{"format": 1, "name": "x", "width": 12}'
    (tracks_dir / "mal.json").write_text(bad, encoding="utf-8")
    app = make_hub()
    key(app, pygame.K_RETURN)
    key(app, pygame.K_RETURN)
    screen = app.current
    index = next(i for i, o in enumerate(screen.options) if o.key.endswith("mal.json"))
    assert screen.options[index].error
    screen.selected = index
    key(app, pygame.K_RETURN)
    assert app.dialog is not None and app.dialog.title == "Circuito no válido"
    assert app.current is screen  # still choosing
    app.run(2)  # and the box draws without a traceback


# -- launching and coming back ------------------------------------------------------------------


def test_playing_runs_the_game_and_comes_back_with_a_valid_window(make_hub):
    app = make_hub()
    select(app, "contrarreloj")
    key(app, pygame.K_RETURN)
    key(app, pygame.K_RIGHT)
    key(app, pygame.K_RETURN)  # azul
    track_screen = app.current
    key(app, pygame.K_RETURN)  # the first circuit
    assert app.busy_text == "Cargando…"
    app.run(2)  # frame 1 shows "Cargando…", then the game runs for three frames and returns
    assert app.dialog is None and app.current is app.main_screen
    assert pygame.display.get_caption()[0] == "KART LAB"
    assert app.screen.get_size() == (640, 360)
    assert app.settings.last_track == "harbor" and track_screen.options[0].key == "harbor"
    app.ui.name.get("¿Sigue vivo?", (255, 255, 255))  # the fonts are still valid
    app.run(2)
    assert pygame.surfarray.array3d(app.screen).any()


def test_closing_the_window_inside_a_game_closes_the_menu_too(make_hub):
    app = make_hub()
    select(app, "conduccion")
    key(app, pygame.K_RETURN)
    key(app, pygame.K_RETURN)
    key(app, pygame.K_RETURN)
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    app.run(5)
    assert app.running is False


def test_open_option_goes_straight_to_the_game_with_the_saved_choices(make_hub):
    app = make_hub()
    app.settings.kart_ramp = 4
    app.settings.last_track = "canyon"
    app.open_direct("conduccion")
    seen = {}
    original = hub_module.launch_drive

    def spy(hub, choices, time_trial):
        seen.update(choices, time_trial=time_trial)
        original(hub, choices, time_trial)

    hub_module.launch_drive = spy
    try:
        app.run(2)
    finally:
        hub_module.launch_drive = original
    assert seen == {"kart": 4, "track": "canyon", "time_trial": False}
    assert app.current is app.main_screen


def test_open_the_track_editor_and_come_back(make_hub):
    app = make_hub()
    app.open_direct("editor")
    app.run(2)
    assert app.dialog is None and app.current is app.main_screen
    assert pygame.display.get_caption()[0] == "KART LAB"
    assert app.screen.get_size() == (640, 360)  # not the editor's size


def test_the_command_opens_an_entry_with_frames_and_exits_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(hub_module, "SETTINGS_PATH", tmp_path / "settings.json")
    shot = tmp_path / "game.png"
    main(
        ["--open", "contrarreloj", "--track", "plateau", "--scale", "2", "--frames", "3"]
        + ["--screenshot", str(shot)]
    )
    assert shot.stat().st_size > 0  # the screenshot is of the game, not of the menu
    main(["--open", "editor", "--scale", "2", "--frames", "3"])


def test_a_bad_track_name_with_open_shows_a_message_not_a_traceback(make_hub):
    app = make_hub()
    app.open_direct("conduccion", track="no-existe")
    app.run(2)
    assert app.dialog is not None and app.dialog.title == "Circuito no válido"


# -- the flow is a registry ---------------------------------------------------------------------


def test_a_new_mode_is_one_entry_and_a_new_step_is_one_registry_line(make_hub, monkeypatch):
    launched = []

    class Difficulty(hub_module.Screen):
        def __init__(self, hub, flow):
            super().__init__(hub)
            self.flow = flow

        def on_key(self, k):
            if k == pygame.K_RETURN:
                self.hub.advance("normal")
            elif k == pygame.K_ESCAPE:
                self.hub.back()

    monkeypatch.setitem(STEPS, "difficulty", Step(Difficulty, lambda hub: "normal"))
    entry = Entry(
        "carrera-prueba",
        "Carrera de prueba",
        "Un modo nuevo",
        "JUGAR",
        ("kart", "track", "difficulty"),
        lambda hub, choices: launched.append(dict(choices)),
    )
    app = make_hub(entries=[*ENTRIES[:2], entry, *ENTRIES[2:]])
    select(app, "carrera-prueba")
    key(app, pygame.K_RETURN)  # opens the entry: the kart screen
    key(app, pygame.K_RETURN)  # kart chosen
    key(app, pygame.K_RETURN)  # circuit chosen
    assert type(app.current) is Difficulty
    key(app, pygame.K_ESCAPE)  # steps back to the circuit
    assert type(app.current).__name__ == "TrackScreen"
    key(app, pygame.K_RETURN)
    key(app, pygame.K_RETURN)  # the extra step confirms
    app.run(2)
    assert launched == [{"kart": 0, "track": "harbor", "difficulty": "normal"}]


def test_the_entry_keys_are_unique_and_open_accepts_every_one(make_hub):
    keys = [entry.key for entry in ENTRIES]
    assert len(keys) == len(set(keys))
    assert {"contrarreloj", "conduccion", "editor", "entrenar", "ver"} <= set(keys)
    assert all(step in STEPS for entry in ENTRIES for step in entry.steps)


# -- options --------------------------------------------------------------------------------------


def test_options_change_apply_and_are_saved(make_hub):
    app = make_hub()
    select(app, "opciones")
    key(app, pygame.K_RETURN)
    screen = app.current
    key(app, pygame.K_RIGHT)  # window size: auto -> 1x
    assert app.settings.scale == 1 and app.screen.get_size() == (320, 180)
    key(app, pygame.K_RIGHT)
    assert app.settings.scale == 2 and app.screen.get_size() == (640, 360)
    key(app, pygame.K_DOWN)
    key(app, pygame.K_DOWN)  # light
    key(app, pygame.K_RIGHT)
    assert app.settings.light == "sunset"
    key(app, pygame.K_DOWN)
    key(app, pygame.K_LEFT)  # camera: auto -> top (backwards)
    assert app.settings.camera == "top"
    saved = json.loads(app.settings_path.read_text(encoding="utf-8"))
    assert saved["scale"] == 2 and saved["light"] == "sunset" and saved["camera"] == "top"
    assert screen.rects and app.ui.size == (640, 360)
    assert Settings.load(app.settings_path).camera == "top"


# -- the AI modes ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    from kartlab.train.train import train

    config = TrainConfig(
        name="hub-test",
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


def test_watching_the_ai_runs_the_viewer_embedded_and_comes_back(make_hub, tiny_run):
    app = make_hub(runs_dir=tiny_run.root.parent)
    assert app.disabled_reason(next(e for e in app.entries if e.key == "ver")) is None
    app.open_direct("ver")
    app.run(2)
    assert app.dialog is None and app.current is app.main_screen
    assert app.screen.get_size() == (640, 360)


def test_the_race_and_progression_modes_start_in_their_own_mode(make_hub, tiny_run):
    app = make_hub(runs_dir=tiny_run.root.parent)
    for entry in ("carrera-versiones", "progresion"):
        app.open_direct(entry)
        app.run(2)
        assert app.dialog is None and app.current is app.main_screen


def test_the_viewer_stops_with_escape_when_embedded(tiny_run):
    from kartlab.show.viewer import ViewerApp

    viewer = ViewerApp(tiny_run, scale=2, realtime=False, mode="watch", embedded=True)
    try:
        viewer.run_loop(3)
        viewer.handle_key(pygame.K_ESCAPE)
        assert viewer.running is False and viewer.state == "scene"  # no title screen
    finally:
        viewer.close()
        pygame.quit()


def test_a_run_without_checkpoints_raises_the_dedicated_exception(tiny_run, tmp_path):
    from kartlab.show.cli import NoCheckpoints
    from kartlab.show.viewer import ViewerApp
    from kartlab.train.run import RunPaths

    empty = RunPaths(tmp_path / "vacio")
    empty.root.mkdir()
    shutil.copy(tiny_run.config, empty.config)
    with pytest.raises(NoCheckpoints, match="no tiene checkpoints"):
        ViewerApp(empty, scale=2, realtime=False)


def test_the_chosen_run_is_used_and_the_latest_is_the_default(make_hub, tiny_run, tmp_path):
    runs = tmp_path / "runs"
    for name in ("2026-01-01_0000_a", "2026-02-01_0000_b"):
        shutil.copytree(tiny_run.root, runs / name)
    app = make_hub(runs_dir=runs)
    assert app.current_run().root.name == "2026-02-01_0000_b"
    app.settings.last_run = "2026-01-01_0000_a"
    app.invalidate()
    assert app.current_run().root.name == "2026-01-01_0000_a"
    app.settings.last_run = "borrado"
    app.invalidate()
    assert app.current_run().root.name == "2026-02-01_0000_b"  # a run that is gone: the latest


# -- training -------------------------------------------------------------------------------------


def open_training(app):
    select(app, "entrenar")
    key(app, pygame.K_RETURN)
    return app.current


def test_the_training_screen_lists_the_real_configs_with_an_estimate(make_hub):
    app = make_hub(configs_dir=REPO_ROOT / "configs")
    screen = open_training(app)
    names = [c.name for c in screen.configs]
    assert {"demo-live", "fixed-track", "random-tracks"} <= set(names)
    demo = screen.configs[names.index("demo-live")]
    assert demo.total_steps == 400_000 and demo.estimate == "unos 5 min"
    assert demo.comment.startswith("Demo en directo")
    app.run(2)  # draws


def test_navigating_the_whole_menu_never_starts_a_training(make_hub):
    spawn = FakeSpawn()
    app = make_hub(configs_dir=REPO_ROOT / "configs", spawn=spawn)
    screen = open_training(app)
    for k in (pygame.K_DOWN, pygame.K_UP, pygame.K_v, pygame.K_p, pygame.K_o, pygame.K_DOWN):
        key(app, k)
    app.dialog = None
    app.run(3)
    key(app, pygame.K_ESCAPE)
    assert screen is not app.current
    assert spawn.calls == []
    assert app.job is None and app.status_line() == ""


def test_launching_a_training_spawns_the_trainer_at_low_priority_with_a_log_file(
    make_hub, tmp_path
):
    configs = tmp_path / "configs"
    path = write_config(configs, "mini", 50_000)
    spawn = FakeSpawn()
    app = make_hub(spawn=spawn)
    screen = open_training(app)
    key(app, pygame.K_RETURN)
    assert len(spawn.calls) == 1
    command, options = spawn.calls[0]
    assert command[1:4] == ["-m", "kartlab.train.train", str(path)]
    assert command[command.index("--stop-file") + 1].endswith(".stop")
    assert Path(command[command.index("--stop-file") + 1]).parent == tmp_path / "logs"
    assert options["stdout"].name.endswith(".log")  # a file, never a pipe
    assert options["stderr"] == subprocess.STDOUT
    if sys.platform == "win32":
        flags = options["creationflags"]
        assert flags & 0x4000 and flags & 0x08000000 and flags & 0x200
    assert "Entrenando mini" in app.status_line()
    assert screen.button_states()["launch"][1]  # a second launch is refused while it runs
    # Parar creates the stop file; the process then ends by itself
    key(app, pygame.K_p)
    assert app.job.stop_path.exists() and app.job.stopping
    assert "Parando" in app.status_line()
    spawn.process.code = 0
    app.update(0.1)
    assert "terminado" in app.status_line()


def test_a_long_training_asks_twice_and_escape_cancels(make_hub, tmp_path):
    write_config(tmp_path / "configs", "larga", 3_000_000)  # about 31 minutes
    spawn = FakeSpawn()
    app = make_hub(spawn=spawn)
    screen = open_training(app)
    assert screen.configs[0].is_long and screen.configs[0].estimate == "unos 32 min"
    key(app, pygame.K_RETURN)
    assert app.dialog is not None and "más de 30 min" in app.dialog.text
    key(app, pygame.K_ESCAPE)
    assert app.dialog is None and spawn.calls == []
    key(app, pygame.K_RETURN)
    key(app, pygame.K_RETURN)  # Intro = yes
    assert len(spawn.calls) == 1


def test_an_invalid_config_is_listed_with_its_error_and_cannot_be_launched(make_hub, tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "rota.yaml").write_text("name: x\nplayers: 3\n", encoding="utf-8")
    spawn = FakeSpawn()
    app = make_hub(spawn=spawn)
    screen = open_training(app)
    assert screen.configs[0].error and "players" in screen.configs[0].error
    key(app, pygame.K_RETURN)
    assert spawn.calls == [] and app.dialog is not None
    app.run(2)


def test_stopping_with_nothing_running_is_explained(make_hub, tmp_path):
    write_config(tmp_path / "configs", "mini", 1000)
    app = make_hub()
    open_training(app)
    key(app, pygame.K_p)
    assert app.dialog is not None and app.dialog.title == "Nada que parar"


def test_live_view_and_folder_buttons_need_a_run(make_hub, tmp_path):
    write_config(tmp_path / "configs", "mini", 1000)
    opened = []
    app = make_hub(opener=opened.append)
    open_training(app)
    key(app, pygame.K_v)
    assert app.dialog is not None
    app.dialog = None
    key(app, pygame.K_o)
    assert app.dialog is not None and opened == []


def test_the_live_view_opens_from_the_training_screen_and_training_continues(
    make_hub, tiny_run, tmp_path
):
    write_config(tmp_path / "configs", "mini", 1000)
    old = time.time() - 3600
    os.utime(tiny_run.metrics, (old, old))  # the shared run is not being written to
    opened = []
    spawn = FakeSpawn()
    app = make_hub(runs_dir=tiny_run.root.parent, spawn=spawn, opener=opened.append)
    screen = open_training(app)
    key(app, pygame.K_RETURN)  # launches (fake) the trainer
    app.job._run = tiny_run  # as if the log had said "Run: ..."
    key(app, pygame.K_o)
    assert opened == [tiny_run.root]
    key(app, pygame.K_v)
    app.run(2)  # the live view runs three frames and returns
    assert app.dialog is None and app.current is screen
    assert app.job.running  # leaving the live view does not stop the training


def test_a_training_started_elsewhere_is_noticed_by_its_metrics_file(make_hub, tiny_run):
    metrics = tiny_run.root / "metrics.csv"
    assert metrics.exists()
    os.utime(metrics, (time.time() - 3600, time.time() - 3600))
    app = make_hub(runs_dir=tiny_run.root.parent)
    assert not app.training_alive() and app.status_line() == ""  # an hour old: not running
    os.utime(metrics, None)  # touched just now
    app.invalidate()
    assert app.training_alive()
    assert "Entrenando" in app.status_line()


# -- the player's colour --------------------------------------------------------------


def test_kart_colour_names_and_indices_agree_with_the_palette_ramps():
    assert KART_COLOR_NAMES == (
        "Naranja", "Azul", "Morado", "Turquesa", "Rosa", "Amarillo", "Blanco", "Rojo",
    )  # fmt: skip
    assert parse_kart_color("azul") == 1 and parse_kart_color("Rojo") == 7
    assert parse_kart_color("5") == 5
    for bad in ("verde", "8", "-1", ""):
        with pytest.raises(ValueError, match="Color de kart"):
            parse_kart_color(bad)
    assert len({kart_color_rgb(i) for i in range(8)}) == 8


def test_drive_uses_the_chosen_colour_in_both_views(tmp_path):
    from kartlab.show.drive import DriveApp
    from kartlab.show.drive import main as drive_main
    from kartlab.sim.showcase import showcase_track

    app = DriveApp(showcase_track("harbor"), scale=2, realtime=False, kart_ramp=3, view="topdown")
    try:
        app.run(max_frames=3)
        assert app.kart_ramp == 3
        assert app.ghost_ramp == 1  # blue ghost, unless the player is blue
        app.kart_ramp = 1
        assert app.ghost_ramp == 0
    finally:
        pygame.quit()
    out = tmp_path / "color.png"
    drive_main(["--color", "turquesa", "--scale", "2", "--frames", "5", "--screenshot", str(out)])
    assert out.stat().st_size > 0
    with pytest.raises(SystemExit, match="Color de kart"):
        drive_main(["--color", "verde", "--frames", "1"])


def test_an_activity_that_fails_shows_a_message_and_the_menu_keeps_working(make_hub, capsys):
    def broken(hub, choices):
        raise RuntimeError("se ha roto")

    entry = Entry("rota", "Rota", "Falla al abrirse", "JUGAR", ("track",), broken)
    app = make_hub(entries=[entry, *ENTRIES])
    select(app, "rota")
    key(app, pygame.K_RETURN)
    key(app, pygame.K_RETURN)
    app.run(2)
    assert app.dialog is not None and app.dialog.title == "Algo ha fallado"
    assert "se ha roto" in app.dialog.text
    assert app.current is app.main_screen and app.screen.get_size() == (640, 360)
    capsys.readouterr()  # the traceback went to stderr, not to the player
