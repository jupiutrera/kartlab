"""The menu entry "Carrera contra la IA": kart, circuit, race options, race and back.

Synthetic keyboard and mouse events on a headless menu, with a tiny run to race against.
"""

import json
import shutil

import pygame
import pytest

from kartlab.common.paths import REPO_ROOT
from kartlab.show.hub import ENTRIES, KartLabApp, main
from kartlab.show.settings import Settings
from kartlab.train.config import PPOParams, TrainConfig

TRACKS = REPO_ROOT / "tracks"


@pytest.fixture(scope="module")
def tiny_run_dir(tmp_path_factory):
    from kartlab.train.train import train

    config = TrainConfig(
        name="menu-race-test",
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
    final = run.list_checkpoints()[-1]
    run.best_model.write_bytes(final.model.read_bytes())
    return run.root


@pytest.fixture
def make_hub(tmp_path):
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    shutil.copy(TRACKS / "ejemplo.json", tracks / "ejemplo.json")
    shutil.copy(TRACKS / "ocho.json", tracks / "ocho.json")  # has a level crossing
    created = []

    def make(with_run: bool = True, **kwargs) -> KartLabApp:
        runs = tmp_path / "runs"
        runs.mkdir(exist_ok=True)
        if with_run and not any(runs.iterdir()):
            shutil.copytree(kwargs.pop("run_dir"), runs / "2026-01-01_0000_ai")
        options = {
            "scale": 2,
            "realtime": False,
            "background": False,
            "activity_frames": 4,
            "settings_path": tmp_path / "settings.json",
            "root": tmp_path,
            "runs_dir": runs,
            "logs_dir": tmp_path / "logs",
            "tracks_dir": tracks,
            "configs_dir": tmp_path / "configs",
            "opener": lambda path: None,
        }
        app = KartLabApp(**{**options, **kwargs})
        created.append(app)
        return app

    yield make
    pygame.quit()


def key(app, k):
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode=""))


def select(app, entry_key):
    app.main_screen.selected = next(i for i, e in enumerate(app.entries) if e.key == entry_key)


def open_race(app):
    select(app, "carrera")
    key(app, pygame.K_RETURN)
    assert type(app.current).__name__ == "KartScreen"
    key(app, pygame.K_RETURN)  # the colour as it is
    assert type(app.current).__name__ == "TrackScreen"


# -- the entry ----------------------------------------------------------------------------------


def test_the_entry_is_the_first_of_play_and_needs_a_run():
    entry = ENTRIES[0]
    assert entry.key == "carrera" and entry.group == "JUGAR"
    assert entry.steps == ("kart", "track", "race") and entry.needs_run
    assert [e.key for e in ENTRIES if e.group == "JUGAR"][:3] == [
        "carrera",
        "contrarreloj",
        "conduccion",
    ]


def test_the_entry_is_off_without_a_run_and_says_to_train_first(make_hub):
    app = make_hub(with_run=False)
    entry = next(e for e in app.entries if e.key == "carrera")
    assert app.disabled_reason(entry) == "Entrena antes a la IA"
    select(app, "carrera")
    key(app, pygame.K_RETURN)
    assert app.dialog is not None and "Entrena antes a la IA" in app.dialog.text
    assert app.flow is None
    app.run(2)  # and the menu draws it dimmed without a traceback


def test_the_entry_is_on_with_a_run(make_hub, tiny_run_dir):
    app = make_hub(run_dir=tiny_run_dir)
    entry = next(e for e in app.entries if e.key == "carrera")
    assert app.disabled_reason(entry) is None
    assert app.main_screen.selected == 0  # the menu opens on it


# -- the flow -----------------------------------------------------------------------------------


def test_circuits_with_level_crossings_are_not_offered(make_hub, tiny_run_dir):
    app = make_hub(run_dir=tiny_run_dir)
    open_race(app)
    keys = [option.key for option in app.current.options]
    assert "harbor" in keys and "file:tracks/ejemplo.json" in keys
    assert "file:tracks/ocho.json" not in keys  # it has a crossing
    assert all(
        option.track is not None and not option.track.crossings for option in app.current.options
    )


def test_the_whole_flow_races_and_comes_back_to_the_menu(make_hub, tiny_run_dir, tmp_path):
    app = make_hub(run_dir=tiny_run_dir)
    open_race(app)
    key(app, pygame.K_RETURN)  # the first circuit
    assert type(app.current).__name__ == "RaceOptionsScreen"
    screen = app.current
    assert screen.selected == 4  # the start button: Intro races with the saved options
    screen.selected = 0
    key(app, pygame.K_RIGHT)  # Normal -> Difícil
    screen.selected = 1
    key(app, pygame.K_RIGHT)  # 2 -> 3 rivals
    screen.selected = 2
    key(app, pygame.K_RIGHT)  # 3 -> 5 laps
    screen.selected = 3
    key(app, pygame.K_LEFT)  # bumps off
    seen = {}
    import kartlab.show.viewer as viewer

    original = viewer.ViewerApp.__init__

    def spy(self, *args, **kwargs):
        seen.update(kwargs, mode=args[3])
        original(self, *args, **kwargs)

    viewer.ViewerApp.__init__ = spy
    try:
        key(app, pygame.K_RETURN)
        assert app.busy_text == "Cargando…"
        app.run(2)  # "Cargando…", then the race for four frames, then the menu again
    finally:
        viewer.ViewerApp.__init__ = original
    assert seen["mode"] == "vsai" and seen["embedded"] is True
    assert (seen["difficulty"], seen["rivals"], seen["vs_laps"], seen["collisions"]) == (
        "dificil",
        3,
        5,
        False,
    )
    assert app.dialog is None and app.current is app.main_screen
    assert pygame.display.get_caption()[0] == "KART LAB" and app.screen.get_size() == (640, 360)
    saved = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert saved["race_difficulty"] == "dificil" and saved["race_rivals"] == 3
    assert saved["race_laps"] == 5 and saved["race_collisions"] is False
    app.run(2)
    assert pygame.surfarray.array3d(app.screen).any()


def test_escape_goes_back_one_step_at_a_time(make_hub, tiny_run_dir):
    app = make_hub(run_dir=tiny_run_dir)
    open_race(app)
    key(app, pygame.K_RETURN)
    assert type(app.current).__name__ == "RaceOptionsScreen"
    key(app, pygame.K_ESCAPE)
    assert type(app.current).__name__ == "TrackScreen"
    key(app, pygame.K_ESCAPE)
    assert type(app.current).__name__ == "KartScreen"
    key(app, pygame.K_ESCAPE)
    assert app.current is app.main_screen


def test_the_options_screen_works_with_the_mouse_and_the_wheel(make_hub, tiny_run_dir):
    app = make_hub(run_dir=tiny_run_dir)
    open_race(app)
    key(app, pygame.K_RETURN)
    screen = app.current
    row = screen.rects[2]  # Vueltas
    app.handle_event(
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(row.right - 5, row.centery), button=1)
    )
    assert app.settings.race_laps == 5  # a click on the right goes forward
    app.handle_event(
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(row.x + 5, row.centery), button=1)
    )
    assert app.settings.race_laps == 3  # and on the left, back
    app.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1))
    assert app.settings.race_laps == 5
    screen.selected = 0
    app.run(2)  # draws


def test_escape_inside_the_race_returns_to_the_menu(make_hub, tiny_run_dir):
    app = make_hub(run_dir=tiny_run_dir, activity_frames=600)
    open_race(app)
    key(app, pygame.K_RETURN)
    key(app, pygame.K_RETURN)  # start
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0, unicode=""))
    app.run(2)  # the race reads the Esc on its first frame and leaves
    assert app.current is app.main_screen and app.dialog is None
    app.run(2)


def test_closing_the_window_inside_the_race_closes_the_menu(make_hub, tiny_run_dir):
    app = make_hub(run_dir=tiny_run_dir, activity_frames=600)
    open_race(app)
    key(app, pygame.K_RETURN)
    key(app, pygame.K_RETURN)
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    app.run(5)
    assert app.running is False


def test_enter_repeats_the_race_and_escape_leaves_after_the_results(make_hub, tiny_run_dir):
    import kartlab.show.viewer as viewer

    app = make_hub(run_dir=tiny_run_dir, activity_frames=60)
    open_race(app)
    key(app, pygame.K_RETURN)
    created = []
    original = viewer.ViewerApp.__init__

    def spy(self, *args, **kwargs):
        original(self, *args, **kwargs)
        created.append(self)

    viewer.ViewerApp.__init__ = spy
    try:
        key(app, pygame.K_RETURN)
        app.run(2)
    finally:
        viewer.ViewerApp.__init__ = original
    race = created[0].scene
    assert race.name == "vsai" and created[0].embedded
    race.ended_at = 1.0  # as if the results were up: Intro repeats, Esc leaves
    assert race.on_key(pygame.K_RETURN) is True and race.ended_at is None
    race.on_escape()
    assert created[0].running is False


def test_open_race_goes_straight_to_the_game_with_the_saved_choices(make_hub, tiny_run_dir):
    app = make_hub(run_dir=tiny_run_dir)
    app.settings.last_track = "canyon"
    app.settings.race_laps = 1
    app.open_direct("carrera")
    assert app.flow.step is None or app.busy_text  # nothing left to ask
    app.run(2)
    assert app.dialog is None and app.current is app.main_screen


def test_main_open_carrera_with_frames(tiny_run_dir, tmp_path, monkeypatch):
    """``kartlab --open carrera --frames N`` with a run starts, races and exits cleanly."""
    runs = tmp_path / "runs"
    runs.mkdir()
    shutil.copytree(tiny_run_dir, runs / "2026-01-01_0000_ai")
    monkeypatch.setattr("kartlab.show.hub.RUNS_DIR", runs)
    monkeypatch.setattr("kartlab.show.hub.SETTINGS_PATH", tmp_path / "settings.json")
    code = None
    try:
        main(["--open", "carrera", "--scale", "2", "--frames", "3"])
    except SystemExit as exit_info:
        code = exit_info.code
    assert code in (None, 0)


def test_the_race_settings_are_validated():
    assert Settings.valid("race_difficulty", "campeon") and not Settings.valid(
        "race_difficulty", "x"
    )
    assert Settings.valid("race_rivals", 3) and not Settings.valid("race_rivals", 4)
    assert not Settings.valid("race_rivals", True)
    assert Settings.valid("race_laps", 5) and not Settings.valid("race_laps", 2)
    assert Settings.valid("race_collisions", False) and not Settings.valid("race_collisions", 0)


def test_the_settings_keys_match_the_difficulty_levels():
    from kartlab.show.humanrace import DIFFICULTIES
    from kartlab.show.settings import RACE_DIFFICULTIES

    assert tuple(level.key for level in DIFFICULTIES) == RACE_DIFFICULTIES
