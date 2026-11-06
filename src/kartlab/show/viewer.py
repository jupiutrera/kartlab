"""Exhibition viewer: a title screen and three ways to show a run off.

Run with ``uv run python scripts/viewer.py`` (latest run) or ``--run runs/<folder>``.
From the title screen pick a mode:

- watch: one version driving; change it with left/right (or the wheel, or by clicking the bar);
- race: every version at once on the same grid;
- progression: versions on a log scale of steps, one after another.

Everything runs on the CPU. The version at the wheel is a deterministic policy, checkpoints
load in a background thread (and are cached), and a "Cargando" notice shows while the
window keeps drawing. Headless: ``--frames N --screenshot f.png`` advances one physics step
per frame, like ``drive.py``.
"""

from __future__ import annotations

import argparse
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pygame

from kartlab.render.display import open_window, set_window_mode
from kartlab.render.scene import KartVisual
from kartlab.render.topdown import ACCENT
from kartlab.render.view import (
    CAMERA_NAMES,
    DIM,
    SANS,
    SANS_REGULAR,
    VARIANT_NAMES,
    WHITE,
    GameView,
    HudInfo,
    scale_for,
    unit_for,
)
from kartlab.show.brain import BrainPanel
from kartlab.show.charts import MetricsReader, RewardChart
from kartlab.show.cli import NoCheckpoints, existing_run_or_exit, resolve_track_or_exit
from kartlab.show.drivers import PolicyDriver, format_steps
from kartlab.show.overlays import Overlays, ease_in_out
from kartlab.show.scenes import SCENES, AttractScene, Scene, WatchScene
from kartlab.show.title import MenuEntry, TitleScreen
from kartlab.show.versions import VersionLibrary
from kartlab.sim.kart import DT, KartState
from kartlab.sim.track import Track
from kartlab.sim.tracksets import (
    EVAL_SEEDS,
    eval_track,
    resolve_track,
    sample_train_seed,
    train_track,
)
from kartlab.train.config import TrainConfig
from kartlab.train.run import RUNS_DIR, RunPaths

TITLE = "KART LAB"
MAX_PENDING = 0.1  # never simulate (or animate) more than this after a stall (s)
SPEEDS = (1.0, 0.5, 0.25)  # S cycles through these
SHOWCASE_KEYS = {pygame.K_1: "harbor", pygame.K_2: "canyon", pygame.K_3: "plateau"}
MAX_GAMES = 3  # 3D views kept in memory (one per track visited recently)
FADE_OUT, FADE_IN = 0.22, 0.32  # black transitions between scenes (s)
TITLE_VARIANTS = ("day", "sunset", "night")
TITLE_VARIANT_SECONDS = 22.0

MODES: dict[str, MenuEntry] = {
    "watch": MenuEntry("watch", "Ver conducir", "Una versión al volante; cámbiala con ← →"),
    "race": MenuEntry("race", "Carrera de versiones", "Todas a la vez desde la misma parrilla"),
    "progression": MenuEntry(
        "progression", "Progresión", "De sin entrenar a la última, paso a paso"
    ),
}
MODE_ORDER = tuple(MODES)

IGNORED_KEYS = {
    pygame.K_LSHIFT,
    pygame.K_RSHIFT,
    pygame.K_LCTRL,
    pygame.K_RCTRL,
    pygame.K_LALT,
    pygame.K_RALT,
    pygame.K_CAPSLOCK,
    pygame.K_NUMLOCK,
    pygame.K_SCROLLLOCK,
    pygame.K_LMETA,
    pygame.K_RMETA,
    pygame.K_MODE,
    pygame.K_PRINTSCREEN,
    pygame.K_PAUSE,
    pygame.K_MENU,
    *range(pygame.K_F1, pygame.K_F12 + 1),
}

HELP_SECTIONS = (
    (
        "Modos",
        (
            ("M", "cambiar de modo (ver · carrera · progresión)"),
            ("Esc", "volver al título (al menú, si vienes de él); en el título, salir"),
        ),
    ),
    (
        "Versiones",
        (
            ("← →", "versión anterior / siguiente"),
            ("Re Pág / Av Pág", "10 versiones atrás / adelante"),
            ("Inicio / Fin", "la primera / la última"),
            ("Rueda · clic", "en la barra de versiones"),
            ("", "En la carrera, ← → eligen a quién sigue la cámara."),
        ),
    ),
    (
        "Circuito",
        (
            ("1 2 3", "circuitos escaparate"),
            ("E", "siguiente circuito de evaluación"),
            ("N", "circuito aleatorio"),
            ("R", "reiniciar"),
        ),
    ),
    (
        "Vista",
        (
            ("C", "cámara: persecución, retransmisión, cenital"),
            ("V", "día, atardecer, noche"),
            ("G", "mostrar los sensores (rayos) en el mundo"),
            ("B", "panel cerebro: lo que ve y decide la IA"),
            ("L", "modo ligero (30 fps)"),
        ),
    ),
    (
        "Tiempo y pantalla",
        (
            ("P · Espacio", "pausa"),
            ("S", "cámara lenta: 0,5× y 0,25×"),
            ("F3", "diagnóstico: fps y ms de cada parte"),
            ("F11", "pantalla completa"),
            ("H", "esta ayuda"),
        ),
    ),
)


def latest_run_with_checkpoints(runs_dir: Path = RUNS_DIR) -> RunPaths | None:
    """The most recent run that has a config and at least one checkpoint to show."""
    for folder in sorted(runs_dir.glob("*"), reverse=True):
        candidate = RunPaths(folder)
        if candidate.config.exists() and candidate.list_checkpoints():
            return candidate
    return None


def speed_label(speed: float) -> str:
    return f"{speed:g}".replace(".", ",") + "×"


class ViewerApp:
    def __init__(
        self,
        run: RunPaths,
        track_choice: str = "harbor",
        track: Track | None = None,
        mode: str | None = None,
        camera: str | None = None,
        variant: str = "day",
        scale: int | None = None,
        fullscreen: bool = False,
        realtime: bool = True,
        laps: int = 2,
        version: int | None = None,
        rays: bool = False,
        diagnostics: bool = False,
        brain: bool = False,
        embedded: bool = False,
    ) -> None:
        if not run.list_checkpoints():
            raise NoCheckpoints(f"{run.root} no tiene checkpoints")
        self.run = run
        self.embedded = embedded  # run by the main menu: Esc leaves, no title screen
        self.realtime = realtime
        self.laps = laps
        self.start_version = version
        self.library = VersionLibrary(run, synchronous=not realtime)
        self.env_config = TrainConfig.from_yaml(run.config).env_config()
        self.metrics = MetricsReader(run.metrics)
        self.metrics.refresh()
        self.rng = np.random.default_rng(0)
        self.eval_index = -1
        self.camera_choice = camera  # None: each scene picks its own
        self.variant = variant
        self.rays = rays
        self.diagnostics = diagnostics
        self.brain = brain  # the brain panel (B)
        self.brain_panel: BrainPanel | None = None
        self._brain_target: PolicyDriver | None = None  # the driver whose V(s) is being tracked
        self._brain_ms = 0.0
        self.paused = False
        self.speed_index = 0
        self.help_open = False
        self.fullscreen = fullscreen
        self.scale = scale
        self.allow_auto_light = True
        self.running = False
        self.quit_requested = False  # the window was closed (the main menu closes too)
        self.t = 0.0  # UI clock (s): advances with the display, also while paused
        self._fade: list | None = None  # [start, callback, fired]
        self._job: tuple[str, Callable[[], None]] | None = None
        self.busy_text: str | None = None
        self._sim_ms = 0.0
        self._pending = 0.0  # simulated time not yet stepped (s)
        self._games: OrderedDict[str, GameView] = OrderedDict()
        self._help_surface: pygame.Surface | None = None
        self._title_variant_start = 0.0

        self.screen = open_window(TITLE, scale, fullscreen)
        self.clock = pygame.time.Clock()
        self.title = TitleScreen([MODES[name] for name in MODE_ORDER])
        self._splash("Cargando…")

        self.track = track if track is not None else resolve_track(track_choice)
        self.track_key = track_choice if track is None else f"file:{track.name}"
        self.game = self._game_for(self.track_key, self.track)
        self.ov = Overlays(self.screen, self.game)
        self._layout_widgets()

        self.mode = mode or MODE_ORDER[0]
        self.title.selected = MODE_ORDER.index(self.mode)
        self.attract = AttractScene(self)
        self.scene: Scene = self.attract
        self.state = "title"
        if mode is None and not embedded:
            self._enter_title()
        else:
            self._enter_scene(mode)
            self._fade = None

    # -- construction helpers -----------------------------------------------------------------

    def _splash(self, text: str) -> None:
        """First frame while the 3D view is built (about a second and a half)."""
        screen = self.screen
        screen.fill((14, 14, 22))
        size = screen.get_size()
        self.title.layout(size, scale_for(size), unit_for(size))
        mark = self.title._scaled
        screen.blit(mark, ((size[0] - mark.get_width()) // 2, size[1] // 3))
        label = self.title.sub.get(text, DIM)
        screen.blit(
            label, ((size[0] - label.get_width()) // 2, size[1] // 3 + mark.get_height() + 40)
        )
        pygame.display.flip()

    def _game_for(self, key: str, track: Track) -> GameView:
        """The 3D view of a track: built once (about 1.5 s) and kept for a while."""
        game = self._games.pop(key, None)
        previous = getattr(self, "game", None)
        if game is None:
            game = GameView(
                track,
                self.screen.get_size(),
                self.camera_choice or "broadcast",
                self.variant,
                seed=track.decoration_seed,
                ramps=tuple(range(8)),
            )
        self._games[key] = game
        while len(self._games) > MAX_GAMES:
            self._games.popitem(last=False)
        if previous is not None and previous is not game:
            game.world.set_light_mode(previous.world.light_mode)
        if game.world.variant != self.variant:
            game.world.set_variant(self.variant)
        game.diagnostics = self.diagnostics
        game.auto_light = self.allow_auto_light
        return game

    def _layout_widgets(self) -> None:
        """Everything that depends on the window size: chart, title, help."""
        size = self.screen.get_size()
        u = unit_for(size)
        self.chart = RewardChart(
            (round(u * 17.5), round(u * 9.5)),
            pygame.font.Font(SANS_REGULAR, round(u * 0.75)),
            log_x=True,
        )
        self.title.layout(size, scale_for(size), u)
        self._help_surface = None
        self.brain_panel = None  # rebuilt for the new size on the next frame

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self.screen = set_window_mode(self.scale, self.fullscreen)
        size = self.screen.get_size()
        for key in [k for k, g in self._games.items() if g is not self.game]:
            del self._games[key]  # built for the old size
        self.game.resize(size)
        self.ov.set_screen(self.screen, self.game)
        self._layout_widgets()

    # -- fades and deferred work --------------------------------------------------------------

    def fade(self, callback: Callable[[], None]) -> None:
        """Fade to black, call ``callback`` at the darkest point, fade back in."""
        if self._fade is not None and not self._fade[2]:
            self._fade[1]()  # the previous change still has to happen
        self._fade = [self.t, callback, False]

    def _fade_amount(self) -> float:
        if self._fade is None:
            return 0.0
        start, callback, fired = self._fade
        age = self.t - start
        if age < FADE_OUT:
            return ease_in_out(age / FADE_OUT)
        if not fired:
            self._fade[2] = True
            callback()
        if age < FADE_OUT + FADE_IN:
            return 1.0 - ease_in_out((age - FADE_OUT) / FADE_IN)
        self._fade = None
        return 0.0

    def defer(self, text: str, job: Callable[[], None]) -> None:
        """Run ``job`` after the next frame, which shows ``text`` as a loading notice."""
        self.busy_text = text
        self._job = (text, job)

    def notify(self, text: str) -> None:
        self.game.notify(text)

    # -- state changes -------------------------------------------------------------------------

    def _enter_title(self) -> None:
        self.state = "title"
        self.paused = False
        self.attract.rebuild()
        self.scene = self.attract
        self.attract.enter()
        self.title.open(self.t)
        self._title_variant_start = self.t
        self._apply_camera()
        self.reset_cameras()

    def _enter_scene(self, mode: str) -> None:
        self.mode = mode
        self.state = "scene"
        if self.game.world.variant != self.variant:
            self.game.world.set_variant(self.variant)
        scene_class = SCENES[mode]
        if scene_class is WatchScene:
            self.scene = WatchScene(self, self.start_version)
            self.start_version = None
        else:
            self.scene = scene_class(self)
        self.scene.enter()
        self._apply_camera()
        self.reset_cameras()
        self.game.notify(MODES[mode].name)

    def reset_cameras(self) -> None:
        for camera in self.game.cameras.values():
            camera.pose = None

    def _apply_camera(self) -> None:
        name = self.camera_choice or self.scene.default_camera()
        if self.game.camera_name != name:
            self.game.set_camera(name)

    def go_title(self) -> None:
        self.fade(self._enter_title)

    def start_mode(self, mode: str) -> None:
        self.fade(lambda: self._enter_scene(mode))

    def next_mode(self) -> None:
        self.start_mode(MODE_ORDER[(MODE_ORDER.index(self.mode) + 1) % len(MODE_ORDER)])

    def change_track(self, key: str, track: Track) -> None:
        self.defer(f"Cargando {track.name}…", lambda: self._switch_track(key, track))

    def _switch_track(self, key: str, track: Track) -> None:
        self.track, self.track_key = track, key
        self.game = self._game_for(key, track)
        self.ov.game = self.game
        self.attract.rebuild()
        self.scene.rebuild()
        self._apply_camera()
        self.reset_cameras()
        self.game.notify(track.name)

    # -- input ---------------------------------------------------------------------------------

    def handle_key(self, key: int) -> None:
        if key == pygame.K_F11:
            self.toggle_fullscreen()
        elif key == pygame.K_F3:
            self.diagnostics = not self.diagnostics
            self.game.diagnostics = self.diagnostics
        elif self.help_open:
            if key in (pygame.K_ESCAPE, pygame.K_h, pygame.K_F1):
                self.help_open = False
        elif key == pygame.K_h:
            self.help_open = True
        elif self.state == "title":
            self._title_key(key)
        else:
            self._scene_key(key)

    def _title_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key in (pygame.K_UP, pygame.K_LEFT, pygame.K_w, pygame.K_a):
            self.title.move(-1)
        elif key in (pygame.K_DOWN, pygame.K_RIGHT, pygame.K_s, pygame.K_d, pygame.K_m):
            self.title.move(1)
        elif key not in IGNORED_KEYS and self._fade is None:
            self.start_mode(self.title.entry.key)

    def _scene_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            if self.embedded:
                self.running = False  # back to the main menu
            else:
                self.go_title()
        elif key == pygame.K_m:
            self.next_mode()
        elif self.scene.on_key(key):
            return
        elif key in (pygame.K_p, pygame.K_SPACE):
            self.paused = not self.paused
        elif key == pygame.K_s:
            self.speed_index = (self.speed_index + 1) % len(SPEEDS)
            self.notify(
                "Velocidad normal"
                if self.speed == 1.0
                else f"Cámara lenta {speed_label(self.speed)}"
            )
        elif key == pygame.K_r:
            self.scene.restart()
            self.reset_cameras()
        elif key == pygame.K_c:
            names = list(CAMERA_NAMES)
            current = self.camera_choice or self.game.camera_name
            self.camera_choice = names[(names.index(current) + 1) % len(names)]
            self._apply_camera()
            self.notify(f"Cámara {CAMERA_NAMES[self.camera_choice]}")
        elif key == pygame.K_v:
            self.variant = self.game.cycle_variant()
            self.notify(f"Luz de {VARIANT_NAMES[self.variant]}")
        elif key == pygame.K_g:
            self.rays = not self.rays
            self.notify("Sensores " + ("activados" if self.rays else "ocultos"))
        elif key == pygame.K_b:
            self.toggle_brain()
        elif key == pygame.K_l:
            self.allow_auto_light = False
            self.game.auto_light = False
            self.game.world.set_light_mode(not self.game.world.light_mode)
            self.notify("Modo ligero " + ("activado" if self.game.world.light_mode else "apagado"))
        elif key in SHOWCASE_KEYS:
            name = SHOWCASE_KEYS[key]
            self.change_track(name, resolve_track(name))
        elif key == pygame.K_e:
            self.eval_index = (self.eval_index + 1) % len(EVAL_SEEDS)
            self.change_track(f"eval:{self.eval_index + 1}", eval_track(self.eval_index))
        elif key == pygame.K_n:
            seed = sample_train_seed(self.rng)
            self.change_track(str(seed), train_track(seed))

    @property
    def speed(self) -> float:
        return SPEEDS[self.speed_index]

    def toggle_brain(self) -> None:
        self.brain = not self.brain
        if self.brain and self.brain_panel is not None:
            self.brain_panel.open(self.game.now())
        self.notify("Cerebro " + ("activado" if self.brain else "oculto"))

    def _sync_brain(self) -> None:
        """V(s) is only computed for the kart the panel shows, and only while it is open."""
        wanted = self.scene.brain_driver() if self.brain and self.state == "scene" else None
        if wanted is not self._brain_target:
            if self._brain_target is not None:
                self._brain_target.set_tracking(False)
            if wanted is not None:
                wanted.set_tracking(True)
            self._brain_target = wanted

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.quit_requested = True
            self.running = False
        elif event.type == pygame.KEYDOWN:
            self.handle_key(event.key)
        elif event.type == pygame.MOUSEWHEEL and self.state == "scene" and not self.help_open:
            self.scene.on_wheel(event.y)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.help_open:
                self.help_open = False
            elif self.state == "title":
                if self._fade is None:
                    self.start_mode(self.title.entry.key)
            else:
                self.scene.on_click(event.pos)

    # -- main loop -----------------------------------------------------------------------------

    @property
    def target_fps(self) -> int:
        return self.game.target_fps

    def run_loop(self, max_frames: int | None = None) -> None:
        """Realtime mode keeps a fixed 1/60 s physics step with an accumulator; headless
        mode advances exactly one display frame (of 1/60 s) per iteration."""
        self.running = True
        frames = 0
        while self.running:
            if self.realtime:
                dt = self.clock.tick(self.target_fps) / 1000.0
            else:
                self.clock.tick()
                dt = DT
            ui_dt = min(dt, MAX_PENDING)
            self.t += ui_dt
            for event in pygame.event.get():
                self.handle_event(event)
            fade = self._fade_amount()
            self._housekeeping(ui_dt)
            self.scene.update()
            self._sync_brain()
            stopped = self.paused and self.state == "scene"
            factor = 1.0 if self.state == "title" else self.speed
            if stopped:
                self._pending = 0.0
            else:
                self._pending += ui_dt * factor
            t0 = time.perf_counter()
            while self._pending >= DT - 1e-9:
                self.scene.physics_step()
                self._pending -= DT
            self._sim_ms = self._sim_ms * 0.9 + (time.perf_counter() - t0) * 1000 * 0.1
            self.draw(ui_dt, 0.0 if stopped else ui_dt * factor, fade)
            pygame.display.flip()
            self._run_deferred()
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break

    def close(self) -> None:
        self.library.close()

    def _housekeeping(self, dt: float) -> None:
        self.library.poll()
        while self.library.notices:
            self.notify(self.library.notices.pop(0))
        if self.library.refresh(dt):
            self.notify("Hay versiones nuevas en el entrenamiento")
        if self.realtime:
            self.metrics.refresh()
        self.game.auto_light = self.allow_auto_light and not self.library.busy
        if self.library.busy:
            self.game.forget_fps()  # loading in the background dips the frame rate
        if self.state == "title":  # light changes slowly behind the title
            elapsed = self.t - self._title_variant_start
            wanted = TITLE_VARIANTS[
                (TITLE_VARIANTS.index(self.variant) + int(elapsed // TITLE_VARIANT_SECONDS))
                % len(TITLE_VARIANTS)
            ]
            if self.game.world.variant != wanted:
                self.game.world.set_variant(wanted)

    def _run_deferred(self) -> None:
        if self._job is None:
            return
        _, job = self._job
        self._job = None
        job()
        self.busy_text = None
        self.game.forget_fps()  # the stall was not the viewer being slow
        self.clock.tick()  # do not count the stall as the next frame's time

    # -- drawing -------------------------------------------------------------------------------

    def _parked(self) -> list[KartVisual]:
        return [KartVisual(KartState(*self.track.pose_at(0.0)), 0)]

    def draw(self, ui_dt: float, sim_dt: float, fade: float) -> None:
        game, scene, screen = self.game, self.scene, self.screen
        in_scene = self.state == "scene"
        got = scene.frame()
        if got is None:
            visuals, focus, info = self._parked(), 0, HudInfo()
        else:
            visuals, focus, info = got
        pose = game.update(visuals, sim_dt, focus)
        game.set_rays(visuals[focus] if (self.rays and in_scene and got) else None)
        if in_scene and got is not None:
            if self.paused:
                info.extra = [*info.extra, "PAUSA"]
            if self.speed < 1.0:
                info.extra = [*info.extra, f"Cámara lenta {speed_label(self.speed)}"]
            info.extra = [*info.extra, "H: ayuda"]
        game.fixed_time = None if self.realtime else self.t
        game.timings["simulación"] = self._sim_ms
        fps = self.clock.get_fps() if self.realtime else None
        game.draw(screen, pose, visuals, info, ui_dt, fps, hud=in_scene and got is not None)
        if in_scene:
            scene.draw_overlays()
            self._draw_brain()
            loading = scene.loading_text()
            if loading:
                self._draw_loading(loading)
        else:
            self._draw_title()
        if self.busy_text:
            self._draw_loading(self.busy_text)
        if self.help_open:
            self._draw_help()
        self.ov.dim(fade)

    def _draw_brain(self) -> None:
        driver = self.scene.brain_driver()
        if not self.brain or driver is None:
            self.game.timings.pop("cerebro", None)
            return
        t0 = time.perf_counter()
        game, panel = self.game, self.brain_panel
        if panel is None:
            config = self.env_config
            panel = self.brain_panel = BrainPanel(
                game.unit, config.ray_count, config.ray_fov_deg, config.ray_range
            )
            panel.open(game.now())
        top = game.world_rect.y + round(game.unit * 5.8)
        if self.scene.brain_side == "left":
            bottom = game.world_rect.bottom - game._minimap[0].get_height() - game.unit
        else:
            top += round(game.unit * 3.2)  # under the model label and its extra lines
            bottom = game.world_rect.bottom - game.unit * 4
        pos = panel.position(game.world_rect, self.scene.brain_side, top, bottom, game.now())
        panel.draw(self.screen, pos, driver)
        self._brain_ms = self._brain_ms * 0.9 + (time.perf_counter() - t0) * 1000 * 0.1
        game.timings["cerebro"] = self._brain_ms

    def _draw_title(self) -> None:
        lib = self.library
        steps = lib.steps
        last = format_steps(steps[-1])
        first = "sin entrenar" if steps[0] == 0 else format_steps(steps[0])
        versions = f"{len(steps)} versiones · de {first} a {last} pasos"
        self.title.draw(
            self.screen,
            self.t,
            f"Entrenamiento {self.run.root.name}",
            versions,
            "↑ ↓ elegir · Intro empezar · H ayuda · F11 pantalla completa · Esc salir",
        )

    def _draw_loading(self, text: str) -> None:
        """A small pill above the bottom edge with animated dots."""
        ov, game = self.ov, self.game
        u = ov.unit
        dots = "." * (1 + int(self.t * 3) % 3)
        label = game.text.get(f"{text.removesuffix('…')}{dots}", ACCENT)
        pad = u // 2
        w, h = label.get_width() + 2 * pad, label.get_height() + pad
        x = game.world_rect.centerx - w // 2
        y = game.world_rect.bottom - u * 7
        self.screen.blit(ov.panel(w, h), (x, y))
        self.screen.blit(label, (x + pad, y + pad // 2))

    def _build_help(self) -> pygame.Surface:
        size = self.screen.get_size()
        u = unit_for(size)
        head = pygame.font.Font(SANS, round(u * 1.2))
        key_font = pygame.font.Font(SANS, u)
        text_font = pygame.font.Font(SANS_REGULAR, u)
        pad = u
        columns: list[list[tuple[str, str, bool]]] = [[], []]
        for i, (title, rows) in enumerate(HELP_SECTIONS):
            column = columns[0 if i < 3 else 1]
            if column:
                column.append(("", "", False))
            column.append((title, "", True))
            column.extend((k, v, False) for k, v in rows)
        key_w = max(key_font.size(k)[0] for col in columns for k, _, h in col if not h) + u
        col_w = key_w + max(text_font.size(v)[0] for col in columns for _, v, _ in col) + u * 2
        line = max(text_font.get_height(), head.get_height())
        height = max(len(c) for c in columns) * line + pad * 3 + head.get_height()
        width = col_w * 2 + pad * 2
        surface = pygame.Surface(size, pygame.SRCALPHA)
        surface.fill((6, 8, 16, 170))
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((14, 14, 22, 240))
        pygame.draw.rect(panel, ACCENT, (0, 0, width, max(u // 4, 3)))
        title = head.render("Controles", True, WHITE)
        panel.blit(title, (pad, pad))
        for c, column in enumerate(columns):
            y = pad * 2 + head.get_height()
            for key, value, is_head in column:
                x = pad + c * col_w
                if is_head:
                    panel.blit(key_font.render(key.upper(), True, ACCENT), (x, y))
                elif key or value:
                    panel.blit(key_font.render(key, True, WHITE), (x, y))
                    panel.blit(text_font.render(value, True, DIM), (x + key_w, y))
                y += line
        surface.blit(panel, ((size[0] - width) // 2, (size[1] - height) // 2))
        return surface

    def _draw_help(self) -> None:
        if self._help_surface is None:
            self._help_surface = self._build_help()
        self.screen.blit(self._help_surface, (0, 0))

    def save_screenshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.screen, str(path))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Visor de exhibición: pantalla de título y tres modos para enseñar un run."
    )
    parser.add_argument("--run", type=Path, help="carpeta del run (por defecto, el último)")
    parser.add_argument(
        "--track",
        default="harbor",
        help="harbor, canyon, plateau, eval:N (1-20) o una semilla de entrenamiento",
    )
    parser.add_argument(
        "--track-file", type=Path, help="circuito en fichero JSON (sustituye a --track)"
    )
    parser.add_argument(
        "--mode",
        choices=tuple(MODES),
        help="empieza directamente en este modo, sin pantalla de título",
    )
    parser.add_argument(
        "--camera",
        choices=("chase", "broadcast", "top"),
        help="cámara 3D (por defecto, cada modo elige la suya)",
    )
    parser.add_argument("--variant", choices=("day", "sunset", "night"), default="day")
    parser.add_argument("--version", type=int, help="pasos de la versión inicial (modo watch)")
    parser.add_argument("--laps", type=int, default=2, help="vueltas de la carrera")
    parser.add_argument("--scale", type=int, help="escala entera sobre 320x180")
    parser.add_argument("--fullscreen", action="store_true", help="pantalla completa")
    parser.add_argument("--rays", action="store_true", help="muestra los sensores")
    parser.add_argument("--brain", action="store_true", help="abre el panel cerebro")
    parser.add_argument("--diagnostics", action="store_true", help="tiempos de cada parte")
    parser.add_argument(
        "--frames", type=int, help="sale tras N fotogramas, con un paso de física por fotograma"
    )
    parser.add_argument("--screenshot", type=Path, help="guarda el último fotograma en este PNG")
    args = parser.parse_args(argv)

    run = existing_run_or_exit(args.run) if args.run else latest_run_with_checkpoints()
    if run is None:
        raise SystemExit(
            "No hay ningún run con checkpoints en runs/. Lanza antes scripts/train.py."
        )
    if not run.list_checkpoints():
        raise SystemExit(f"{run.root} no tiene checkpoints")
    if args.track_file:
        track = resolve_track_or_exit(str(args.track_file))
    else:
        track = None
        resolve_track_or_exit(args.track)  # fail now, with a clear message
    try:
        app = ViewerApp(
            run,
            args.track,
            track,
            args.mode,
            args.camera,
            args.variant,
            args.scale,
            args.fullscreen,
            realtime=args.frames is None,
            laps=max(1, args.laps),
            version=args.version,
            rays=args.rays,
            diagnostics=args.diagnostics,
            brain=args.brain,
        )
    except NoCheckpoints as error:
        raise SystemExit(str(error)) from None
    try:
        app.run_loop(args.frames)
        if args.screenshot:
            app.save_screenshot(args.screenshot)
    finally:
        app.close()
        pygame.quit()
