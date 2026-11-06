"""KART LAB's main menu: one window, one command (``uv run kartlab`` or ``Jugar.bat``).

Every activity (driving, the exhibition viewer, the live view, the track editor) runs in
this same process, one after another: the menu starts its ``run`` loop, and when it
returns (Esc) the menu restores its own window. Nothing here imports torch or
Stable-Baselines3 at import time, so the menu opens in about a second; the AI modes pull
them in when they are used.

Adding a menu entry is one ``Entry`` in ``ENTRIES`` (and, for a new screen between the
menu and the game, one ``Step`` in ``STEPS``): modes declare the steps they go through.
"""

from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

from kartlab.common.paths import REPO_ROOT
from kartlab.render.display import open_window, set_window_mode
from kartlab.render.palette import game_palette
from kartlab.render.scene import KartVisual
from kartlab.render.sprites import SpriteBank
from kartlab.render.topdown import ACCENT
from kartlab.render.view import GameView, HudInfo
from kartlab.show.cli import NoCheckpoints
from kartlab.show.menu import (
    Dialog,
    Entry,
    Flow,
    MenuScreen,
    OptionsScreen,
    Screen,
    TrainScreen,
    Ui,
    open_folder,
)
from kartlab.show.pickers import KartScreen, TrackCatalog, TrackScreen
from kartlab.show.raceoptions import RaceOptionsScreen, race_choices
from kartlab.show.settings import SETTINGS_PATH, Settings
from kartlab.show.title import build_wordmark
from kartlab.show.training_job import CONFIGS_DIR, LOGS_DIR, TrainingJob, live_run
from kartlab.sim.autopilot import Autopilot
from kartlab.sim.kart import DT
from kartlab.sim.race import Racer
from kartlab.sim.showcase import showcase_track
from kartlab.sim.trackfile import TrackFileError, load_spec

TITLE = "KART LAB"
BACKGROUND = (14, 14, 22)  # until the 3D backdrop is ready
TRACKS_DIR = REPO_ROOT / "tracks"
RUNS_DIR = REPO_ROOT / "runs"
MAX_PENDING = 0.1  # never advance more than this after a stall (s)
CACHE_SECONDS = 2.0  # how long the list of runs is trusted before looking at the disk again
ENDED_SECONDS = 30.0  # how long the menu keeps showing that a training has finished
NO_RUN_REASON = "Entrena antes a la IA"
STILL_TRAINING = "El entrenamiento sigue en marcha; ábrelo de nuevo y pulsa Parar."


# -- the activities -----------------------------------------------------------------------------


def _camera(hub: KartLabApp, default: str | None) -> str | None:
    """The camera setting for an activity; ``auto`` leaves the choice to the activity."""
    return default if hub.settings.camera == "auto" else hub.settings.camera


def launch_drive(hub: KartLabApp, choices: dict, time_trial: bool) -> None:
    from kartlab.show.drive import DriveApp

    track = hub.resolve_track(choices["track"])
    app = DriveApp(
        track,
        scale=hub.scale,
        fullscreen=hub.fullscreen,
        camera=_camera(hub, "chase"),
        variant=hub.settings.light,
        realtime=hub.realtime,
        time_trial=time_trial,
        kart_ramp=choices.get("kart", hub.settings.kart_ramp),
    )
    hub.run_activity(app, lambda: app.run(max_frames=hub.activity_frames), app.save_screenshot)


def launch_time_trial(hub: KartLabApp, choices: dict) -> None:
    launch_drive(hub, choices, time_trial=True)


def launch_free_drive(hub: KartLabApp, choices: dict) -> None:
    launch_drive(hub, choices, time_trial=False)


def launch_viewer(hub: KartLabApp, choices: dict, mode: str) -> None:
    from kartlab.show.viewer import ViewerApp

    run = hub.current_run()
    if run is None:
        hub.message("Sin IA", NO_RUN_REASON + ".")
        return
    choice = choices["track"]
    track = hub.resolve_track(choice)
    try:
        app = ViewerApp(
            run,
            choice,
            track,
            mode,
            _camera(hub, None),
            hub.settings.light,
            hub.scale,
            hub.fullscreen,
            realtime=hub.realtime,
            embedded=True,
        )
    except NoCheckpoints as error:
        hub.message("Sin checkpoints", str(error))
        return
    hub.run_activity(
        app, lambda: app.run_loop(hub.activity_frames), app.save_screenshot, close=app.close
    )


def launch_watch(hub: KartLabApp, choices: dict) -> None:
    launch_viewer(hub, choices, "watch")


def launch_race(hub: KartLabApp, choices: dict) -> None:
    launch_viewer(hub, choices, "race")


def launch_progression(hub: KartLabApp, choices: dict) -> None:
    launch_viewer(hub, choices, "progression")


def launch_vs_ai(hub: KartLabApp, choices: dict) -> None:
    """Carrera contra la IA: ``kart``, ``track`` and ``race`` come from the flow's steps."""
    from kartlab.show.viewer import ViewerApp

    run = hub.opponent_run()
    if run is None:
        hub.message("Sin IA", NO_RUN_REASON + ".")
        return
    choice = choices["track"]
    track = hub.resolve_track(choice)
    if track.flat_crossings:
        raise ActivityError(
            "Circuito no válido", "La carrera contra la IA no admite circuitos con cruces a nivel."
        )
    race = choices["race"]
    try:
        app = ViewerApp(
            run,
            choice,
            track,
            "vsai",
            _camera(hub, None),
            hub.settings.light,
            hub.scale,
            hub.fullscreen,
            realtime=hub.realtime,
            embedded=True,
            vs_laps=race["laps"],
            rivals=race["rivals"],
            difficulty=race["difficulty"],
            collisions=race["collisions"],
            kart_ramp=choices.get("kart", hub.settings.kart_ramp),
        )
    except NoCheckpoints as error:
        hub.message("Sin checkpoints", str(error))
        return
    hub.run_activity(
        app, lambda: app.run_loop(hub.activity_frames), app.save_screenshot, close=app.close
    )


def no_crossings(option) -> bool:
    """The circuits of the race against the AI: playable ones without flat level crossings
    (a crossing with a bridge over a tunnel is fine: the two roads are apart)."""
    return option.track is not None and not option.track.flat_crossings


def launch_live(hub: KartLabApp, run) -> None:
    from kartlab.show.live import LiveApp

    app = LiveApp(
        run,
        None,
        2,
        hub.scale,
        hub.fullscreen,
        hub.realtime,
        "3d",
        _camera(hub, "broadcast"),
        hub.settings.light,
    )
    app.preload_ghosts()
    if not hub.realtime:
        app.loader.poll_once()  # no background thread without a window: load the latest now
    hub.run_activity(
        app,
        lambda: app.run(max_frames=hub.activity_frames),
        lambda path: pygame.image.save(app.screen, str(path)),
    )


def launch_editor(hub: KartLabApp, choices: dict) -> None:
    from kartlab.show.editor import NEW_SPEC, EditorApp

    choice = choices["edit_track"]
    if choice == "new":
        spec, path, new = NEW_SPEC, None, True
    else:
        path = hub.root / choice.removeprefix("file:")
        try:
            spec = load_spec(path)
        except TrackFileError as error:
            hub.message("No se puede abrir el circuito", str(error))
            return
        new = False
    app = EditorApp(spec, path, new=new, tracks_dir=hub.tracks_dir, root=hub.root)

    def shot(target: Path) -> None:
        app.settle()
        app.draw()
        app.save_screenshot(target)

    hub.run_activity(app, lambda: app.run(max_frames=hub.activity_frames), shot)


def launch_quit(hub: KartLabApp, choices: dict) -> None:
    hub.request_exit()


ENTRIES: list[Entry] = [
    Entry(
        "carrera",
        "Carrera contra la IA",
        "Tú contra karts de la IA: cuenta atrás, vueltas y choques suaves",
        "JUGAR",
        ("kart", "track", "race"),
        launch_vs_ai,
        needs_run=True,
        track_filter=no_crossings,
    ),
    Entry(
        "contrarreloj",
        "Contrarreloj con fantasma",
        "Bate tu récord: tu mejor vuelta corre a tu lado como fantasma",
        "JUGAR",
        ("kart", "track"),
        launch_time_trial,
    ),
    Entry(
        "conduccion",
        "Conducción libre",
        "Da vueltas a tu aire, sin reloj. Mayús para el barrido",
        "JUGAR",
        ("kart", "track"),
        launch_free_drive,
    ),
    Entry(
        "ver",
        "Ver a la IA",
        "Una versión de la IA al volante; cámbiala con ← →",
        "VER A LA IA",
        ("track",),
        launch_watch,
        needs_run=True,
    ),
    Entry(
        "carrera-versiones",
        "Carrera de versiones",
        "Todas las versiones a la vez desde la misma parrilla",
        "VER A LA IA",
        ("track",),
        launch_race,
        needs_run=True,
    ),
    Entry(
        "progresion",
        "Progresión",
        "De sin entrenar a la última, paso a paso",
        "VER A LA IA",
        ("track",),
        launch_progression,
        needs_run=True,
    ),
    Entry(
        "entrenar",
        "Entrenar a la IA",
        "Lanza un entrenamiento, míralo en directo y páralo cuando quieras",
        "CREAR",
        ("train",),
    ),
    Entry(
        "editor",
        "Editor de circuitos",
        "Dibuja un circuito con el ratón; se valida mientras lo editas",
        "CREAR",
        ("edit_track",),
        launch_editor,
    ),
    Entry(
        "opciones",
        "Opciones",
        "Tamaño de la ventana, luz, cámara y qué IA se muestra",
        None,
        ("options",),
    ),
    Entry("salir", "Salir", "Cierra KART LAB", None, (), launch_quit),
]


@dataclass(frozen=True)
class Step:
    """A screen between the menu and the game. ``default`` is what ``--open`` uses instead."""

    screen: Callable[[KartLabApp, Flow], Screen]
    default: Callable[[KartLabApp], object] | None = None


STEPS: dict[str, Step] = {
    "kart": Step(KartScreen, lambda hub: hub.settings.kart_ramp),
    "track": Step(TrackScreen, lambda hub: hub.settings.last_track),
    "edit_track": Step(
        lambda hub, flow: TrackScreen(hub, flow, for_editor=True), lambda hub: "new"
    ),
    "race": Step(RaceOptionsScreen, lambda hub: race_choices(hub.settings)),
    "options": Step(OptionsScreen),
    "train": Step(TrainScreen),
}


# -- the 3D backdrop ----------------------------------------------------------------------------


class Backdrop:
    """A kart driven by the reference autopilot behind the menu.

    The 3D view takes about two seconds to build (floor texture, clouds, mipmaps), so in real
    time it is built in a thread while the menu already answers; until it is ready the screen
    is a plain dark colour, and then it fades in. Headless runs build it at once."""

    FADE_SECONDS = 0.8

    def __init__(self, hub: KartLabApp) -> None:
        self.hub = hub
        self.game: GameView | None = None
        self.pending = 0.0
        self.fade = 0.0  # 0: not shown yet, 1: fully shown
        self.error: Exception | None = None
        self._thread: threading.Thread | None = None
        self._result: tuple | None = None
        self._black: pygame.Surface | None = None

    # -- building ------------------------------------------------------------------------

    def start(self, threaded: bool) -> None:
        hub = self.hub
        args = (hub.screen.get_size(), hub.settings.light, hub.settings.kart_ramp)
        if threaded:
            self._thread = threading.Thread(
                target=self._work, args=args, name="menu-backdrop", daemon=True
            )
            self._thread.start()
        else:
            self._work(*args)
            self._adopt()
            self.fade = 1.0  # deterministic captures: no fade

    def _work(self, size: tuple[int, int], light: str, ramp: int) -> None:
        try:
            track = showcase_track("harbor")
            game = GameView(track, size, "chase", light, seed=track.decoration_seed, ramps=(ramp,))
            self._result = (Racer(track), Autopilot(track), game)
        except Exception as error:  # no backdrop is better than no menu
            self.error = error

    def _adopt(self) -> None:
        """Take the finished view (in the main thread, which owns the window)."""
        if self._result is None:
            return
        self.racer, self.pilot, self.game = self._result
        self._result = None
        self.throttle = 0.0
        size = self.hub.screen.get_size()
        if self.game.window != size:  # the window changed while it was being built
            self.game.resize(size)
        self.set_light(self.hub.settings.light)

    def wait(self, timeout: float = 5.0) -> None:
        """Before the window goes away: let a running build finish."""
        if self._thread is not None:
            self._thread.join(timeout)

    # -- per frame -----------------------------------------------------------------------

    def resize(self, size: tuple[int, int]) -> None:
        self._black = None
        if self.game is not None:
            self.game.resize(size)

    def set_light(self, light: str) -> None:
        if self.game is not None and self.game.world.variant != light:
            self.game.world.set_variant(light)

    def update(self, dt: float) -> None:
        if self._thread is not None and not self._thread.is_alive():
            self._thread = None
            self._adopt()
        if self.game is None:
            return
        self.fade = min(self.fade + dt / self.FADE_SECONDS, 1.0)
        self.pending = min(self.pending + dt, MAX_PENDING)
        while self.pending >= DT:
            steer, self.throttle = self.pilot.act(self.racer)
            self.racer.step(steer, self.throttle)
            self.pending -= DT

    def draw(self, screen: pygame.Surface, dt: float, hide_kart: bool = False) -> bool:
        if self.game is None:
            return False
        racer = self.racer
        visual = KartVisual(
            racer.kart,
            self.hub.settings.kart_ramp,
            on_grass=racer.on_grass,
            throttle=self.throttle,
        )
        pose = self.game.update([visual], dt)
        self.game.fixed_time = None if self.hub.realtime else self.hub.t
        shown = [] if hide_kart else [visual]  # the camera still follows it
        self.game.draw(screen, pose, shown, HudInfo(), dt, None, hud=False)
        if self.fade < 1.0:  # fading in from the plain background
            if self._black is None or self._black.get_size() != screen.get_size():
                self._black = pygame.Surface(screen.get_size())
                self._black.fill(BACKGROUND)
            self._black.set_alpha(round(255 * (1.0 - self.fade)))
            screen.blit(self._black, (0, 0))
        return True


# -- the app ------------------------------------------------------------------------------------


class KartLabApp:
    def __init__(
        self,
        scale: int | None = None,
        fullscreen: bool = False,
        realtime: bool = True,
        activity_frames: int | None = None,
        activity_shot: Path | None = None,
        background: bool = True,
        settings_path: Path = SETTINGS_PATH,
        root: Path = REPO_ROOT,
        runs_dir: Path = RUNS_DIR,
        configs_dir: Path = CONFIGS_DIR,
        tracks_dir: Path = TRACKS_DIR,
        logs_dir: Path = LOGS_DIR,
        spawn: Callable | None = None,
        opener: Callable[[Path], None] = open_folder,
        entries: list[Entry] | None = None,
        started: float | None = None,
    ) -> None:
        self.started = started if started is not None else time.perf_counter()
        self.realtime = realtime
        self.activity_frames = activity_frames  # headless: every activity stops after N frames
        self.activity_shot = activity_shot  # headless: screenshot of the activity, not the menu
        self.activity_shot_taken = False
        self.use_background = background
        self.settings_path = Path(settings_path)
        self.settings = Settings.load(self.settings_path)
        self.root = Path(root)
        self.runs_dir = Path(runs_dir)
        self.configs_dir = Path(configs_dir)
        self.tracks_dir = Path(tracks_dir)
        self.logs_dir = Path(logs_dir)
        self.spawn = spawn if spawn is not None else subprocess.Popen
        self.opener = opener
        # Command-line values win over the saved ones until the player changes them.
        self.scale = scale if scale is not None else self.settings.scale
        self.fullscreen = fullscreen or self.settings.fullscreen
        self.running = False
        self.quit_requested = False
        self.t = 0.0
        self.mouse = (-1, -1)
        self.dialog: Dialog | None = None
        self.flow: Flow | None = None
        # A trainer launched by an earlier session of the menu and still running is ours again.
        self.job: TrainingJob | None = TrainingJob.adopt(self.logs_dir, self.runs_dir)
        self._exit_when_stopped = False  # "Parar y salir": leave once the trainer has saved
        self._job_ended_at: float | None = None
        self._jobs: list[tuple[str | None, Callable[[], None]]] = []
        self.busy_text: str | None = None
        self.time_to_menu: float | None = None
        self._run_cache: tuple[float, list, object | None] | None = None
        self._live_cache: tuple[float, object | None] | None = None
        self._opponent_cache: tuple[float, object | None] | None = None

        self.screen = open_window(TITLE, self.scale, self.fullscreen)
        self.clock = pygame.time.Clock()
        self.wordmark, _ = build_wordmark()
        self.sprites = SpriteBank(game_palette(), "day")
        self.sprites.BUDGET_MS = 12.0  # the kart preview may build a few views per frame
        self.catalog = TrackCatalog(self.tracks_dir, self.root, np.random.default_rng())
        self.backdrop = Backdrop(self)
        self.entries = entries if entries is not None else ENTRIES
        self.ui = Ui(self.screen.get_size())
        self.main_screen = MenuScreen(self, self.entries)
        self.current: Screen = self.main_screen
        self.current.enter()
        if self.use_background:
            if self.realtime:  # a thread: the menu answers while the 3D view is built
                self.backdrop.start(threaded=True)
            else:  # after the first frame, so the menu is already on screen
                self.defer(None, lambda: self.backdrop.start(threaded=False))

    # -- settings and window -----------------------------------------------------------------

    def save_settings(self) -> None:
        self.settings.save(self.settings_path)

    def layout(self) -> None:
        """Everything that depends on the window size."""
        self.screen = pygame.display.get_surface() or self.screen
        self.ui = Ui(self.screen.get_size())
        self.backdrop.resize(self.screen.get_size())
        self.main_screen.layout()
        self.current.layout()

    def apply_window(self) -> None:
        """Apply ``self.scale`` and ``self.fullscreen`` (the session's values) to the window."""
        self.screen = set_window_mode(self.scale, self.fullscreen)
        self.layout()

    def apply_light(self) -> None:
        self.backdrop.set_light(self.settings.light)

    def apply_kart(self) -> None:
        pass  # the backdrop reads the colour from the settings every frame

    def toggle_fullscreen(self) -> None:
        self.fullscreen = self.settings.fullscreen = not self.fullscreen
        self.save_settings()
        self.apply_window()  # keeps the scale the session started with (--scale)

    def restore_window(self) -> None:
        """After an activity: its window mode, caption, cursor and queued events are gone."""
        pygame.display.set_caption(TITLE)
        self.screen = set_window_mode(self.scale, self.fullscreen)
        with contextlib.suppress(pygame.error):
            pygame.key.stop_text_input()
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
        pygame.event.clear()
        self.invalidate()
        self.layout()
        self.clock.tick()  # do not count the activity as one long frame

    # -- runs and training ---------------------------------------------------------------------

    def invalidate(self) -> None:
        self._run_cache = None
        self._live_cache = None
        self._opponent_cache = None

    def _scan_runs(self) -> list:
        from kartlab.train.run import RunPaths

        found = []
        for folder in sorted(self.runs_dir.glob("*"), reverse=True):
            candidate = RunPaths(folder)
            if candidate.config.exists() and candidate.list_checkpoints():
                found.append(candidate)
        return found

    def _runs(self) -> list:
        now = time.perf_counter()
        if self._run_cache is None or now - self._run_cache[0] > CACHE_SECONDS:
            self._run_cache = (now, self._scan_runs(), None)
        return self._run_cache[1]

    def run_names(self) -> list[str]:
        return [run.root.name for run in self._runs()]

    def current_run(self):
        """The run the AI modes show: the chosen one if it still has checkpoints, else the
        latest with checkpoints; ``None`` if there is none."""
        runs = self._runs()
        wanted = self.settings.last_run
        for run in runs:
            if wanted and run.root.name == wanted:
                return run
        return runs[0] if runs else None

    def opponent_run(self):
        """The AI of the race against the AI: the run chosen in Opciones if there is one, else
        the best generalist (see ``versions.best_generalist_run``); ``None`` if no run has
        checkpoints."""
        runs = self._runs()
        wanted = self.settings.last_run
        for run in runs:
            if wanted and run.root.name == wanted:
                return run
        now = time.perf_counter()
        if self._opponent_cache is None or now - self._opponent_cache[0] > CACHE_SECONDS:
            from kartlab.show.versions import best_generalist_run

            self._opponent_cache = (now, best_generalist_run(self.runs_dir))
        return self._opponent_cache[1]

    def latest_run_with_config(self):
        from kartlab.train.run import latest_run

        return latest_run(self.runs_dir)

    def live_training(self):
        """A run being written right now by a training this menu did not start (or did)."""
        now = time.perf_counter()
        if self._live_cache is None or now - self._live_cache[0] > CACHE_SECONDS:
            self._live_cache = (now, live_run(self.runs_dir))
        return self._live_cache[1]

    def training_alive(self) -> bool:
        return (self.job is not None and self.job.running) or self.live_training() is not None

    def request_exit(self) -> None:
        """Leave the menu. With a training of ours running, ask what to do with it first."""
        job = self.job
        if job is None or not job.running or self._exit_when_stopped:
            self.running = False
            return
        self.dialog = Dialog(
            "Hay un entrenamiento en marcha",
            f"{job.status_line()}\n¿Qué hacemos con él al salir?",
            choices=[
                ("Parar y salir", self._stop_and_exit),
                ("Dejar corriendo", self._leave_running),
                ("Cancelar", None),
            ],
        )

    def _stop_and_exit(self) -> None:
        self.job.request_stop()
        self._exit_when_stopped = True  # ``update`` closes the menu when the trainer has saved

    def _leave_running(self) -> None:
        print(STILL_TRAINING, file=sys.stderr)
        self.running = False

    def status_line(self) -> str:
        job = self.job
        if job is not None:
            if job.running:
                return job.status_line()
            if self._job_ended_at is not None and self.t - self._job_ended_at < ENDED_SECONDS:
                return job.status_line()
        live = self.live_training()
        if live is not None:
            return f"Entrenando {live.root.name} (lanzado desde fuera del menú)"
        return ""

    def disabled_reason(self, entry: Entry) -> str | None:
        if entry.needs_run and self.current_run() is None:
            return NO_RUN_REASON
        return None

    # -- tracks --------------------------------------------------------------------------------

    def resolve_track(self, choice: str):
        try:
            return self.catalog.resolve(choice)
        except TrackFileError as error:
            raise ActivityError("Circuito no válido", str(error)) from None
        except (ValueError, IndexError, KeyError):
            raise ActivityError(
                "Circuito no válido", f"No conozco el circuito «{choice}»."
            ) from None

    # -- flow ----------------------------------------------------------------------------------

    def message(self, title: str, text: str) -> None:
        self.dialog = Dialog(title, text)

    def open_entry(self, entry: Entry) -> None:
        reason = self.disabled_reason(entry)
        if reason:
            self.message(entry.label, reason + ".")
            return
        self.flow = Flow(entry)
        self._show_step()

    def open_direct(self, key: str, track: str | None = None) -> None:
        """``--open``: skip the pick-a-thing screens that have a default; stop at the first
        step that is a place to be in (options, training)."""
        entry = next(e for e in self.entries if e.key == key)
        reason = self.disabled_reason(entry)
        if reason:
            self.message(entry.label, reason + ".")
            return
        self.flow = Flow(entry)
        while self.flow.step is not None:
            step = STEPS[self.flow.step]
            if step.default is None:
                break
            value = (
                track
                if (track and self.flow.step in ("track", "edit_track"))
                else step.default(self)
            )
            self.flow.choices[self.flow.step] = value
            self.flow.index += 1
        self._show_step()

    def _show_step(self) -> None:
        flow = self.flow
        if flow is None:
            return
        if flow.step is None:
            self._finish()
            return
        self.current = STEPS[flow.step].screen(self, flow)
        self.current.enter()

    def advance(self, value: object = None) -> None:
        flow = self.flow
        if flow is None:
            return
        flow.choices[flow.step] = value
        flow.index += 1
        self._show_step()

    def back(self) -> None:
        flow = self.flow
        if flow is not None and flow.index > 0:
            flow.index -= 1
            self._show_step()
        else:
            self.go_menu()

    def go_menu(self) -> None:
        self.flow = None
        self.current = self.main_screen
        self.invalidate()
        self.current.enter()

    def _finish(self) -> None:
        flow = self.flow
        entry = flow.entry
        if entry.launch is None:
            self.go_menu()
            return
        if not entry.steps:  # Salir: nothing to load
            entry.launch(self, flow.choices)
            return
        self.defer("Cargando…", lambda: self._run_launch(entry, dict(flow.choices)))

    def _run_launch(self, entry: Entry, choices: dict) -> None:
        self.go_menu()
        self.guarded(lambda: entry.launch(self, choices))

    def guarded(self, action: Callable[[], None]) -> None:
        """Run an activity; whatever goes wrong comes back as a message, never a traceback."""
        try:
            action()
        except ActivityError as error:
            self.message(error.title, error.text)
        except Exception as error:
            traceback.print_exc()
            self.restore_window()  # the activity may have failed with its own window open
            self.message("Algo ha fallado", f"{type(error).__name__}: {error}")

    def open_live(self, run) -> None:
        self.defer("Cargando…", lambda: self.guarded(lambda: launch_live(self, run)))

    def open_folder(self, path: Path) -> None:
        try:
            self.opener(path)
        except OSError as error:
            self.message("No se pudo abrir la carpeta", str(error))

    def run_activity(self, app, runner: Callable[[], None], shot=None, close=None) -> None:
        """Run an activity's loop, then take the menu's window back."""
        try:
            runner()
            if self.activity_shot is not None and shot is not None:
                shot(self.activity_shot)
                self.activity_shot_taken = True
        finally:
            if close is not None:
                close()
            if getattr(app, "quit_requested", False):
                self.request_exit()  # the window was closed inside the activity
            self.restore_window()

    # -- deferred work ---------------------------------------------------------------------------

    def defer(self, text: str | None, job: Callable[[], None]) -> None:
        """Run ``job`` after the next frame, which shows ``text`` as a loading notice."""
        self._jobs.append((text, job))
        if text is not None:
            self.busy_text = text

    def _run_deferred(self) -> None:
        if not self._jobs:
            return
        _, job = self._jobs.pop(0)
        job()
        self.busy_text = next((t for t, _ in self._jobs if t), None)
        self.clock.tick()

    # -- input ---------------------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.request_exit()
        elif self.busy_text:
            return  # something is loading: input waits
        elif self.dialog is not None:
            if event.type == pygame.KEYDOWN:
                self.dialog.on_key(self, event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.dialog.on_click(self, event.pos)
            elif event.type == pygame.MOUSEMOTION:
                self.dialog.on_motion(event.pos)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F11:
                self.toggle_fullscreen()
            else:
                self.current.on_key(event.key)
        elif event.type == pygame.MOUSEMOTION:
            self.mouse = event.pos
            self.current.on_motion(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.current.on_click(event.pos)
        elif event.type == pygame.MOUSEWHEEL:
            self.current.on_wheel(event.y)

    # -- main loop -----------------------------------------------------------------------------

    def update(self, dt: float) -> None:
        job = self.job
        if job is not None:
            job.update()
            if not job.running and self._job_ended_at is None:
                self._job_ended_at = self.t
                self.invalidate()  # a finished run has checkpoints to show
            elif job.running:
                self._job_ended_at = None
        if self._exit_when_stopped and (job is None or not job.running):
            self.running = False  # it stopped and saved: now the menu can go
        self.backdrop.update(dt)
        self.current.update(dt)

    def draw(self, dt: float) -> None:
        screen = self.screen
        if not self.backdrop.draw(screen, dt, self.current.hide_backdrop_kart):
            screen.fill(BACKGROUND)
        self.ui.draw_veil(screen, self.current.dim)
        self.current.draw(screen)
        if self.current.show_status:
            self._draw_status(screen)
        if self.busy_text:
            self._draw_loading(screen)
        if self.dialog is not None:
            self.dialog.draw(screen, self.ui)

    def _draw_status(self, screen: pygame.Surface) -> None:
        """The training line in the top-right corner, whatever screen the player is on."""
        status = self.status_line()
        if self._exit_when_stopped:
            status = "Saliendo en cuanto termine de guardar…"
        if not status:
            return
        ui = self.ui
        text = ui.mono.get(status, ACCENT)
        pad = ui.unit // 2
        w, h = text.get_width() + 2 * pad, text.get_height() + pad
        x = screen.get_width() - w - ui.unit
        screen.blit(ui.panel(w, h, 220), (x, ui.unit))
        screen.blit(text, (x + pad, ui.unit + pad // 2))

    def _draw_loading(self, screen: pygame.Surface) -> None:
        ui = self.ui
        dots = "." * (1 + int(self.t * 3) % 3)
        label = ui.name.get(f"{self.busy_text.removesuffix('…')}{dots}", ACCENT)
        pad = ui.unit
        w, h = label.get_width() + 2 * pad, label.get_height() + pad
        x = (screen.get_width() - w) // 2
        y = screen.get_height() - ui.unit * 8
        screen.blit(ui.panel(w, h, 235), (x, y))
        screen.blit(label, (x + pad, y + pad // 2))

    def run(self, max_frames: int | None = None) -> None:
        self.running = True
        frames = 0
        while self.running:
            if self.realtime:
                dt = self.clock.tick(60) / 1000.0
            else:
                self.clock.tick()
                dt = DT
            dt = min(dt, MAX_PENDING)
            self.t += dt
            for event in pygame.event.get():
                self.handle_event(event)
            self.update(dt)
            self.draw(dt)
            pygame.display.flip()
            if self.time_to_menu is None:
                self.time_to_menu = time.perf_counter() - self.started
            self._run_deferred()
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break

    def save_screenshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.screen, str(path))

    def close(self) -> None:
        self.backdrop.wait()  # a build in the thread must not outlive the window


class ActivityError(Exception):
    """Something the player should read in a box (an invalid circuit...), not a traceback."""

    def __init__(self, title: str, text: str) -> None:
        super().__init__(f"{title}: {text}")
        self.title = title
        self.text = text


def main(argv: list[str] | None = None) -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="KART LAB: menú principal con todas las opciones en una sola ventana."
    )
    parser.add_argument(
        "--open",
        choices=[entry.key for entry in ENTRIES],
        help="abre directamente esta entrada, con el último kart y circuito elegidos",
    )
    parser.add_argument("--track", help="con --open: harbor, canyon, plateau, eval:N o un .json")
    parser.add_argument("--scale", type=int, help="escala entera sobre 320x180")
    parser.add_argument("--fullscreen", action="store_true", help="pantalla completa")
    parser.add_argument(
        "--frames", type=int, help="sale tras N fotogramas, con un paso de física por fotograma"
    )
    parser.add_argument("--screenshot", type=Path, help="guarda el último fotograma en este PNG")
    args = parser.parse_args(argv)

    app = KartLabApp(
        scale=args.scale,
        fullscreen=args.fullscreen,
        realtime=args.frames is None,
        activity_frames=args.frames,
        activity_shot=args.screenshot if args.open else None,
        started=started,
    )
    try:
        if args.open:
            app.open_direct(args.open, args.track)
        app.run(args.frames)
        if args.screenshot and not app.activity_shot_taken:
            app.save_screenshot(args.screenshot)
    finally:
        if app.training_alive():
            print(STILL_TRAINING, file=sys.stderr)
        app.close()
        pygame.quit()
