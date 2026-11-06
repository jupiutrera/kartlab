"""Manual driving, in the pseudo-3D view (default) or the top-down debug view.

Run with ``uv run python scripts/drive.py`` (``--help`` lists the options).
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pygame

from kartlab.render import pixelui as ui
from kartlab.render.display import open_window, present, set_window_mode
from kartlab.render.palette import KART_RAMPS
from kartlab.render.scene import KartVisual
from kartlab.render.topdown import TopDownView, format_time
from kartlab.render.view import GameView, HudInfo, TrialHud, format_delta
from kartlab.show.cli import kart_color_rgb, parse_kart_color
from kartlab.show.overlays import Overlays, build_help, toggle_text
from kartlab.show.racehud import AMBER, GOLD, RaceHud
from kartlab.show.trialrace import (
    BANNER_SECONDS,
    COUNTDOWN_FRAMES,
    FINISH,
    TrialSession,
)
from kartlab.sim.autopilot import Autopilot
from kartlab.sim.drift import DriftState
from kartlab.sim.kart import DT, KartState
from kartlab.sim.race import Racer
from kartlab.sim.showcase import showcase_track
from kartlab.sim.timetrial import LapEvent, RecordStore, TimeTrial
from kartlab.sim.track import Track
from kartlab.sim.trackfile import TrackFileError, load_track_file
from kartlab.sim.tracksets import (
    EVAL_SEEDS,
    eval_track,
    resolve_track,
    sample_train_seed,
    train_track,
)

TITLE = "KART LAB: conducción libre"
TRAIL_SECONDS = 4.0
TRAIL_EVERY = 3  # physics steps between trail points
CAMERA_RESPONSE = 4.0  # top-down follow camera smoothing (1/s)
CAMERA_LEAD = 0.6  # the top-down follow camera aims this many seconds ahead (s)
GHOST_RAMP = 1  # the record's ghost is blue (orange if the player picked blue)
LABEL_MIN_GAP = 5.0  # no "Récord" tag on a ghost closer than this to the player (m)
GHOST_INK = 47  # the ghost in the top-down view
MESSAGE_SECONDS = 4.0  # how long a time-trial message stays in the top-down HUD
MAX_PENDING = 0.1  # never simulate more than this after a stall (s)
SHOWCASE_KEYS = {pygame.K_1: "harbor", pygame.K_2: "canyon", pygame.K_3: "plateau"}
HELP = (  # the first HELP_LEFT rows go in the left column
    ("Flechas/WASD", "conducir"),
    ("R", "reiniciar"),
    ("K", "contrarreloj"),
    ("F", "fantasma"),
    ("C", "cámara"),
    ("V", "luz"),
    ("T", "vista cenital"),
    ("G", "rayos"),
    ("P", "pausa"),
    ("Mayús", "derrape y turbo"),
    ("Tab", "piloto automático"),
    ("1 2 3", "Puerto, Cañón, Meseta"),
    ("E", "otra evaluación"),
    ("N", "circuito aleatorio"),
    ("L", "modo ligero"),
    ("F3", "diagnóstico"),
    ("F11", "pantalla completa"),
    ("H", "ayuda"),
    ("Esc", "volver"),
)
HELP_LEFT = 9


def keyboard_controls(pressed) -> tuple[float, float]:
    """Steer and throttle from the arrow keys or WASD (steer > 0 turns left)."""
    left = pressed[pygame.K_LEFT] or pressed[pygame.K_a]
    right = pressed[pygame.K_RIGHT] or pressed[pygame.K_d]
    up = pressed[pygame.K_UP] or pressed[pygame.K_w]
    down = pressed[pygame.K_DOWN] or pressed[pygame.K_s]
    return float(left) - float(right), float(up) - float(down)


def drift_pressed(pressed) -> bool:
    """Whether the sweep button (either Shift key) is down."""
    return bool(pressed[pygame.K_LSHIFT] or pressed[pygame.K_RSHIFT])


class DriveApp:
    def __init__(
        self,
        track: Track,
        scale: int | None = None,
        fullscreen: bool = False,
        autopilot: bool = False,
        view: str = "3d",
        camera: str = "chase",
        variant: str = "day",
        realtime: bool = True,
        seed: int = 0,
        time_trial: bool = False,
        records_dir: Path | None = None,
        kart_ramp: int = 0,
        race_format: bool = True,
        start: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.start = start  # arc position and lateral offset where every reset puts the kart
        self._start_track = track  # other tracks (keys 1-3, E, N) start on their line
        self._ghost_hint: int | None = None  # centerline index of the ghost (bridges)
        self.screen = open_window(TITLE, scale, fullscreen)
        self.scale = scale
        self.fullscreen = fullscreen
        self.kart_ramp = kart_ramp % len(KART_RAMPS)  # the player's colour
        self.autopilot_on = autopilot
        self.realtime = realtime
        self.rng = np.random.default_rng(seed)
        self.eval_index = -1
        self.view_mode = view
        self.camera3d = camera if camera in ("chase", "broadcast", "top") else "chase"
        self.follow = camera != "track"
        self.variant = variant
        self.paused = False
        self.show_help = False  # it covers the view: H opens it
        self.show_rays = False
        self.diagnostics = False
        self.running = False
        self.quit_requested = False  # the window was closed (the main menu closes too)
        self.controls = (0.0, 0.0)
        self.drift = DriftState()
        self.store = RecordStore(records_dir)
        self.time_trial = time_trial
        self.race_format = race_format  # the time trial is a countdown and a fixed number of laps
        self.session: TrialSession | None = None
        self.overlays: Overlays | None = None
        self.race_hud: RaceHud | None = None
        self.show_ghost = True
        self.trial: TimeTrial | None = None
        self.ghost_visual: KartVisual | None = None  # the ghost drawn in the last frame
        self.hud_trial: TrialHud | None = None
        self._message: tuple[str, float] | None = None
        self.clock = pygame.time.Clock()
        self.canvas = pygame.Surface(ui.CANVAS_SIZE)  # the top-down view and its panels
        self._help_surface: pygame.Surface | None = None
        self.view: TopDownView | None = None
        self.game: GameView | None = None
        self.load_track(track)

    # -- tracks and views -------------------------------------------------------------

    def load_track(self, track: Track) -> None:
        if track is not self._start_track:
            self.start = (0.0, 0.0)
        self.track = track
        self.racer = Racer(track)
        self.autopilot = Autopilot(track)
        self.trial = TimeTrial(track, self.store) if self.time_trial else None
        self.view = None
        self.game = None
        self.overlays = self.race_hud = None
        self._build_view()
        self.reset()
        if self.trial is not None:
            self._announce_trial()

    def _build_view(self) -> None:
        size = self.screen.get_size()
        if self.view_mode == "3d" and self.game is None:
            self._loading()
            self.game = GameView(
                self.track, size, self.camera3d, self.variant, seed=self.track.decoration_seed
            )
            self.game.diagnostics = self.diagnostics
        elif self.view_mode == "topdown" and self.view is None:
            self.view = TopDownView(self.track, ui.CANVAS_SIZE, pixel=True)

    def _loading(self) -> None:
        self.canvas.fill(ui.rgb(ui.NIGHT))
        ui.text(self.canvas, f"Cargando {self.track.name}…", 160, 86, ui.TEXT, align="center")
        present(self.canvas, self.screen)
        pygame.display.flip()

    def reset(self) -> None:
        self.racer.reset(*self.start)
        self._ghost_hint = None
        if self.trial is not None:
            self.trial.begin(self.racer, manual=not self.autopilot_on)
        self.drift.reset()
        self.session = TrialSession() if self.trial is not None and self.race_format else None
        self.camera = (self.racer.kart.x, self.racer.kart.y)
        self.trail: deque[tuple[float, float]] = deque(maxlen=int(TRAIL_SECONDS / DT / TRAIL_EVERY))
        self._steps = 0
        if self.game is not None:
            for camera in self.game.cameras.values():
                camera.pose = None

    # -- time trial ---------------------------------------------------------------------

    def notify(self, text: str) -> None:
        if self.game is not None and self.view_mode == "3d":
            self.game.notify(text)
        self._message = (text, time.perf_counter())

    def set_time_trial(self, on: bool) -> None:
        self.time_trial = on
        self.trial = TimeTrial(self.track, self.store) if on else None
        self.ghost_visual = self.hud_trial = None
        self.session = None
        if on:
            self.reset()
            self._announce_trial()
        else:
            self.notify(toggle_text("Contrarreloj", False))

    def _announce_trial(self) -> None:
        trial = self.trial
        if trial.problem:
            self.notify(trial.problem)
        elif trial.record is None:
            self.notify("Sin récord en este circuito")
        else:
            self.notify(f"Récord {format_time(trial.record.lap_time)}")

    def _on_lap(self, event: LapEvent) -> None:
        lap = format_time(event.lap_time)
        gap = (
            "" if event.previous is None else f" ({format_delta(event.lap_time - event.previous)})"
        )
        if event.record:
            text = f"Primer récord {lap}" if event.previous is None else f"Nuevo récord {lap}{gap}"
            if not event.saved:
                text += ", sin guardar"
        elif not event.valid:
            text = f"Vuelta {lap} (no cuenta)"
        else:
            text = f"Vuelta {lap}{gap}"
        self.notify(text)
        if self.session is not None:
            self.session.on_lap(event.lap_time, event.record, event.valid)

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self.screen = set_window_mode(self.scale, self.fullscreen)
        size = self.screen.get_size()
        if self.game is not None:
            self.game.resize(size)

    def handle_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_r or (
            key in (pygame.K_RETURN, pygame.K_KP_ENTER)
            and self.session is not None
            and self.session.finished
        ):
            self.reset()
        elif key in SHOWCASE_KEYS:
            self.load_track(showcase_track(SHOWCASE_KEYS[key]))
        elif key == pygame.K_e:
            self.eval_index = (self.eval_index + 1) % len(EVAL_SEEDS)
            self.load_track(eval_track(self.eval_index))
        elif key == pygame.K_n:
            self.load_track(train_track(sample_train_seed(self.rng)))
        elif key == pygame.K_c:
            if self.game is not None and self.view_mode == "3d":
                self.camera3d = self.game.cycle_camera()
            else:
                self.follow = not self.follow
        elif key == pygame.K_t:
            self.view_mode = "topdown" if self.view_mode == "3d" else "3d"
            self._build_view()
        elif key == pygame.K_v and self.game is not None:
            self.variant = self.game.cycle_variant()
        elif key == pygame.K_k:
            self.set_time_trial(not self.time_trial)
        elif key == pygame.K_f:
            self.show_ghost = not self.show_ghost
            if self.trial is not None:
                self.notify(toggle_text("Fantasma", self.show_ghost))
        elif key == pygame.K_g:
            self.show_rays = not self.show_rays
        elif key == pygame.K_F3:
            self.diagnostics = not self.diagnostics
            if self.game is not None:
                self.game.diagnostics = self.diagnostics
        elif key == pygame.K_l and self.game is not None:
            self.game.auto_light = False
            self.game.world.set_light_mode(not self.game.world.light_mode)
        elif key == pygame.K_TAB:
            self.autopilot_on = not self.autopilot_on
            self.drift.reset()
        elif key == pygame.K_p:
            self.paused = not self.paused
        elif key == pygame.K_F11:
            self.toggle_fullscreen()
        elif key == pygame.K_h:
            self.show_help = not self.show_help

    # -- main loop -------------------------------------------------------------------

    def physics_step(self) -> None:
        frozen = False  # the time trial session: the countdown, then the kart coasts at the end
        if self.session is not None:
            state = self.session.advance()
            if state == "countdown":
                self.controls = (0.0, 0.0)
                # A Shift held through the countdown starts no sweep until it is released.
                self.drift.hold = drift_pressed(pygame.key.get_pressed())
                self.drift.reset()
                return
            frozen = state == "finished"
        if frozen:
            steer, throttle, drift = 0.0, 0.0, None
        elif self.autopilot_on:
            steer, throttle = self.autopilot.act(self.racer)
            drift = None  # the sweep is for manual driving only
        else:
            pressed = pygame.key.get_pressed()
            steer, throttle = keyboard_controls(pressed)
            self.drift.hold = drift_pressed(pressed)
            drift = self.drift
        self.controls = (steer, throttle)
        self.racer.step(steer, throttle, drift=drift)
        if self.trial is not None and not frozen:
            event = self.trial.update(self.racer, manual=not self.autopilot_on)
            if event is not None:
                self._on_lap(event)
        self._steps += 1
        if self._steps % TRAIL_EVERY == 0:
            self.trail.append((self.racer.kart.x, self.racer.kart.y))

    @property
    def ghost_ramp(self) -> int:
        return GHOST_RAMP if self.kart_ramp != GHOST_RAMP else 0

    @property
    def target_fps(self) -> int:
        if self.view_mode == "3d" and self.game is not None:
            return self.game.target_fps
        return 60

    def run(self, max_frames: int | None = None) -> None:
        """Main loop. Realtime mode keeps a fixed 1/60 s physics step with an accumulator;
        otherwise each frame advances exactly one step (deterministic, for captures)."""
        self.running = True
        frames = 0
        pending = 0.0
        while self.running:
            if self.realtime:
                dt = self.clock.tick(self.target_fps) / 1000.0
            else:
                self.clock.tick()
                dt = DT
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit_requested = True
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)
            if not self.paused:
                if self.realtime:
                    pending = min(pending + dt, MAX_PENDING)
                    while pending >= DT:
                        self.physics_step()
                        pending -= DT
                else:
                    self.physics_step()
            if self.view_mode == "3d":
                self.draw_3d(dt)
            else:
                self.update_camera(dt)
                self.draw_topdown()
            pygame.display.flip()
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break

    # -- drawing -----------------------------------------------------------------------

    def draw_3d(self, dt: float) -> None:
        racer = self.racer
        visual = KartVisual(
            racer.kart,
            self.kart_ramp,
            on_grass=racer.on_grass,
            throttle=self.controls[1],
            drift=None if self.autopilot_on else self.drift,
            ground=racer.level_z,
        )
        visuals = [visual]
        self.ghost_visual = self.hud_trial = None
        if self.trial is not None:
            ghost = self.ghost_state()
            if ghost is not None:
                near = math.hypot(ghost.x - racer.kart.x, ghost.y - racer.kart.y) < LABEL_MIN_GAP
                label = None if near else "Fantasma"  # on top of you, the tag only gets in the way
                self.ghost_visual = KartVisual(
                    ghost, self.ghost_ramp, label, ghost=True, ground=self.ghost_ground(ghost)
                )
                visuals.append(self.ghost_visual)
            self.hud_trial = self.trial_hud()
        pose = self.game.update(visuals, 0.0 if self.paused else dt)
        self.game.set_rays(visual if self.show_rays else None)
        extra = [self.track.name]
        if self.trial is not None:
            extra.append("Fantasma: Sí" if self.show_ghost else "Fantasma: No")
        if self.paused:
            extra.append("PAUSA")
        if not self.show_help:
            extra.append("H: ayuda")
        info = HudInfo(
            laps=racer.laps,
            lap_time=racer.lap_time,
            best_lap=racer.best_lap,
            speed=racer.speed,
            label=self.mode_label(),
            extra=extra,
            trial=self.hud_trial,
        )
        if self.session is not None:  # the race HUD shows the laps; the mode panel only at first
            info.lap_panel = False
            info.label_panel = self.session.counting or self.show_help or self.paused
        fps = self.clock.get_fps() if self.realtime else None
        canvas = self.game.render_canvas(pose, visuals, info, dt, fps)
        if self.overlays is None:
            self.overlays = Overlays(self.game)
            self.race_hud = RaceHud(self.overlays)
        self._draw_session(canvas, self.race_hud)
        if self.show_help:
            self._draw_help(canvas)
        self.game.present(self.screen)

    def _draw_session(self, canvas: pygame.Surface, hud: RaceHud) -> None:
        """The race format of the time trial: lap counter, countdown, banners and results."""
        session = self.session
        if session is None:
            return
        racer = self.racer
        total = session.total if session.finished else racer.time
        hud.lap_counter(
            session.lap_number,
            session.laps,
            format_time(total),
            format_time(0.0 if session.finished else racer.lap_time),
            None if racer.best_lap is None else format_time(racer.best_lap),
        )
        hud.countdown(session.frame, COUNTDOWN_FRAMES, 60)
        age = session.banner_age()
        if age is not None:
            text = session.banner[0]
            hud.banner(text, age, BANNER_SECONDS, GOLD if text == FINISH else AMBER)
        if session.finished:
            record = self.trial.record.lap_time if self.trial and self.trial.record else None
            inks = {"best": ui.YELLOW, "record": ui.YELLOW, "dim": ui.STONE_LIGHT}
            lines = [(text, inks.get(kind, ui.TEXT)) for text, kind in session.result_lines(record)]
            hud.results(
                "Resultados", None, lines, "Intro repetir, Esc salir", session.results_age()
            )

    def _draw_help(self, canvas: pygame.Surface) -> None:
        if self._help_surface is None:
            columns = [
                [(k, v, False) for k, v in HELP[:HELP_LEFT]],
                [(k, v, False) for k, v in HELP[HELP_LEFT:]],
            ]
            self._help_surface = build_help(None, columns)
        ui.dither(canvas, canvas.get_rect(), 2)
        panel = self._help_surface
        canvas.blit(panel, ((320 - panel.get_width()) // 2, (180 - panel.get_height()) // 2))

    def ghost_state(self) -> KartState | None:
        """The record's kart at the current lap time (``None``: off, no record or finished)."""
        if self.trial is None or not self.show_ghost:
            return None
        pose = self.trial.ghost_pose(self.racer.lap_time)
        return None if pose is None else KartState(*pose)

    def ghost_ground(self, ghost: KartState) -> float:
        """Height of the road under the ghost (a bridge): the branch it is on follows its hint."""
        if self.track.levels is None:
            return 0.0
        position = self.track.locate_following(ghost.x, ghost.y, self._ghost_hint)
        self._ghost_hint = position.index
        if abs(position.lateral) > self.track.half_width:
            return 0.0  # on the grass beside a ramp: on the ground
        return self.track.height_at(position.s)

    def trial_hud(self) -> TrialHud:
        trial = self.trial
        record = trial.record.lap_time if trial.record else None
        return TrialHud(record=record, last=trial.last_lap, delta=trial.delta)

    def update_camera(self, dt: float) -> None:
        """North-up follow camera that eases towards a point ahead of the kart (never rotates)."""
        kart = self.racer.kart
        tx = kart.x + kart.vx * CAMERA_LEAD
        ty = kart.y + kart.vy * CAMERA_LEAD
        k = 1.0 - math.exp(-CAMERA_RESPONSE * dt)
        cx, cy = self.camera
        self.camera = (cx + (tx - cx) * k, cy + (ty - cy) * k)

    def mode_label(self) -> str:
        """What the mode panel calls the current mode."""
        if self.autopilot_on:
            return "Piloto automático"
        return "Contrarreloj" if self.trial is not None else "Conducción libre"

    def draw_topdown(self) -> None:
        racer = self.racer
        canvas = self.canvas
        karts = [
            (
                racer.kart,
                kart_color_rgb(self.kart_ramp),
                0.0,
                None if self.autopilot_on else self.drift,
                racer.level_z,
            )
        ]
        self.ghost_visual = self.hud_trial = None
        if self.trial is not None:
            ghost = self.ghost_state()
            if ghost is not None:
                self.ghost_visual = KartVisual(ghost, self.ghost_ramp, ghost=True)
                karts.append((ghost, ui.rgb(GHOST_INK), 0.0, None, self.ghost_ground(ghost)))
            self.hud_trial = self.trial_hud()
        self.view.draw(canvas, karts, self.camera, self.follow, self.trail)
        top = 27 if self.session is not None else 3  # below the lap bar of the race format
        if self.session is not None and self.race_hud is None:
            self.overlays = SimpleNamespace(canvas=canvas, game=None)
            self.race_hud = RaceHud(self.overlays, sign_y=105)
        hud = self.race_hud if self.session is not None else None
        if hud is not None:
            hud.begin_frame()  # this view has no HUD: the race HUD keeps the panels
        self._draw_blocks(canvas, top, hud)
        results_up = False
        if hud is not None:
            self._draw_session(canvas, hud)
            results_up = hud.card is not None
        if not results_up:  # the results table already says it
            self._message_toast(canvas)
        if self.show_help:
            self._draw_help(canvas)
        present(canvas, self.screen)

    def _draw_blocks(self, canvas: pygame.Surface, top: int, hud: RaceHud | None = None) -> None:
        """The data of the top-down view: the lap on the left, the state on the right. With a
        race HUD, the panels are reserved on it so that the card and the signs keep clear."""
        reserve = hud.reserve if hud is not None else pygame.Rect
        racer = self.racer
        rows: list[tuple[str, str, int]] = []
        if self.session is None:
            rows += [
                ("VUELTA", str(racer.laps + 1), ui.TEXT),
                ("TIEMPO", ui.fmt_clock(racer.lap_time), ui.TEXT),
                ("ÚLTIMA", ui.fmt_clock(racer.last_lap), ui.STONE_LIGHT),
                ("MEJOR", ui.fmt_clock(racer.best_lap), ui.YELLOW if racer.best_lap else ui.STONE),
            ]
        rows.append(("KM/H", f"{abs(racer.speed) * 3.6:.0f}", ui.TEXT))
        rows.append(
            (
                "SUELO",
                "hierba" if racer.on_grass else "asfalto",
                ui.ORANGE if racer.on_grass else ui.STONE_LIGHT,
            )
        )
        if self.hud_trial is not None:
            trial = self.hud_trial
            gap = "-:--,--" if trial.delta is None else format_delta(trial.delta)
            rows += [
                ("RÉCORD", ui.fmt_clock(trial.record), ui.YELLOW if trial.record else ui.STONE),
                ("DIF.", gap, ui.TEXT),
            ]
        w = 84
        h = 5 + 9 * (len(rows) + 1)
        ui.plate(canvas, reserve((3, top, w, h)), ui.SLATE)
        ui.text(canvas, ui.ellipsize(self.track.name, w - 8), 7, top + 4, ui.WHITE)
        for i, (label, value, ink) in enumerate(rows, start=1):
            ui.text(canvas, label, 7, top + 4 + 9 * i, ui.STONE_LIGHT)
            ui.text(canvas, value, 3 + w - 4, top + 4 + 9 * i, ink, align="right", tabular=True)
        state = [
            (self.mode_label(), ui.YELLOW),
            (f"Cámara: {'seguimiento' if self.follow else 'circuito'}", ui.STONE_LIGHT),
        ]
        if self.clock.get_fps():
            state.append((f"{self.clock.get_fps():.0f} fps", ui.STONE_LIGHT))
        if self.paused:
            state.append(("PAUSA", ui.ORANGE))
        if not self.show_help:
            state.append(("H: ayuda", ui.STONE_LIGHT))
        width = max(ui.measure(text) for text, _ in state) + 10
        height = 5 + 9 * len(state)
        x = 320 - 3 - width
        ui.plate(canvas, reserve((x, top, width, height)), ui.SLATE)
        for i, (text, ink) in enumerate(state):
            ui.text(canvas, text, x + 5, top + 4 + 9 * i, ink)

    def _message_toast(self, canvas: pygame.Surface) -> None:
        """The last time-trial message, as a plate near the bottom."""
        if not self._message or time.perf_counter() - self._message[1] >= MESSAGE_SECONDS:
            return
        text = ui.ellipsize(self._message[0], 230)
        w = ui.measure(text) + 10
        x = (320 - w) // 2
        ui.plate(canvas, (x, 158, w, 13), ui.NAVY)
        ui.text(canvas, text, 160, 161, ui.YELLOW, align="center")

    def save_screenshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.screen, str(path))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Conduce un kart a mano.")
    parser.add_argument(
        "--track",
        default="harbor",
        help="harbor, canyon, plateau, eval:N (1-20) o una semilla de entrenamiento",
    )
    parser.add_argument(
        "--track-file", type=Path, help="circuito en fichero JSON (sustituye a --track)"
    )
    parser.add_argument(
        "--scale", type=int, help="escala entera sobre 320x180 (por defecto, la mayor que quepa)"
    )
    parser.add_argument("--fullscreen", action="store_true", help="pantalla completa")
    parser.add_argument("--autopilot", action="store_true", help="empieza con piloto automático")
    parser.add_argument(
        "--view", choices=("3d", "topdown"), default="3d", help="3d o cenital de depuración"
    )
    parser.add_argument(
        "--camera",
        choices=("chase", "broadcast", "top", "follow", "track"),
        default="chase",
        help="3D: chase, broadcast o top; cenital: follow o track",
    )
    parser.add_argument("--variant", choices=("day", "sunset", "night"), default="day")
    parser.add_argument(
        "--color",
        default="0",
        help="color de tu kart: Naranja, Azul, Morado, Turquesa, Rosa, Amarillo, Blanco, Rojo "
        "(o un número de 0 a 7)",
    )
    parser.add_argument(
        "--time-trial",
        action="store_true",
        help="contrarreloj: guarda tu mejor vuelta y la muestra como fantasma (tecla K)",
    )
    parser.add_argument(
        "--records-dir", type=Path, help="carpeta de los récords (por defecto, records/)"
    )
    parser.add_argument(
        "--at",
        type=float,
        default=0.0,
        metavar="S",
        help="metro del circuito donde empieza el kart",
    )
    parser.add_argument(
        "--lateral",
        type=float,
        default=0.0,
        metavar="L",
        help="desplazamiento lateral inicial en metros (positivo a la izquierda)",
    )
    parser.add_argument("--rays", action="store_true", help="muestra los sensores")
    parser.add_argument("--diagnostics", action="store_true", help="tiempos de cada parte")
    parser.add_argument(
        "--frames", type=int, help="sale tras N fotogramas, con un paso de física por fotograma"
    )
    parser.add_argument("--screenshot", type=Path, help="guarda el último fotograma en este PNG")
    args = parser.parse_args(argv)

    try:
        ramp = parse_kart_color(args.color)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    try:
        track = load_track_file(args.track_file) if args.track_file else resolve_track(args.track)
    except TrackFileError as error:
        raise SystemExit(f"Circuito no válido:\n{error}") from None
    except (ValueError, IndexError):
        raise SystemExit(f"Circuito no válido: {args.track}") from None
    app = DriveApp(
        track,
        scale=args.scale,
        fullscreen=args.fullscreen,
        autopilot=args.autopilot,
        view=args.view,
        camera=args.camera,
        variant=args.variant,
        realtime=args.frames is None,
        time_trial=args.time_trial,
        records_dir=args.records_dir,
        kart_ramp=ramp,
        start=(args.at, args.lateral),
    )
    app.show_rays = args.rays
    if args.diagnostics:
        app.handle_key(pygame.K_F3)
    try:
        app.run(max_frames=args.frames)
        if args.screenshot:
            app.save_screenshot(args.screenshot)
    finally:
        pygame.quit()
