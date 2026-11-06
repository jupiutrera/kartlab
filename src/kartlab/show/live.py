"""Live training viewer: watches a run folder and shows each new version driving.

Run with ``uv run python scripts/live.py`` (latest run) or ``--run runs/<folder>``.
It is a separate process: closing it never affects training, and when training ends
it keeps showing the last version. It uses the pseudo-3D view (broadcast camera by
default, steady even with erratic early versions); T switches to the top-down view;
B opens the brain panel (what the current version sees and decides) and G draws its rays.
New checkpoints load in a background thread and enter at the end of a lap (or when
the kart is restarted after getting stuck).
"""

from __future__ import annotations

import argparse
import math
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pygame

from kartlab.render import pixelui as ui
from kartlab.render.display import open_window, present
from kartlab.render.palette import KART_RAMPS
from kartlab.render.scene import KartVisual
from kartlab.render.topdown import TopDownView, format_time
from kartlab.render.view import GameView, HudInfo
from kartlab.show.brain import BrainPanel
from kartlab.show.charts import MetricsReader, RewardChart
from kartlab.show.cli import existing_run_or_exit, resolve_track_or_exit
from kartlab.show.drivers import PolicyDriver, Version, format_steps, load_version
from kartlab.show.overlays import run_title
from kartlab.sim.kart import DT
from kartlab.sim.track import Track
from kartlab.sim.tracksets import resolve_track
from kartlab.train.config import TrainConfig
from kartlab.train.run import Checkpoint, RunPaths, latest_run

TITLE = "KART LAB: entrenamiento en directo"
POLL_SECONDS = 1.0
METRICS_POLL_SECONDS = 2.0
NOTICE_SECONDS = 3.0
CANVAS_W, CANVAS_H = ui.CANVAS_SIZE
PANEL_W = 96  # the side panel of the top-down view, in canvas pixels
MINIMAP_GAP = 3  # canvas pixels kept free above the minimap of the 3D view
LIGHT_MODE_FPS = 50.0  # below this (averaged over 2 s) the viewer drops to 30 fps


def ui_ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1.0 - (1.0 - t) ** 3


class ModelLoader:
    """Polls ``latest.json`` and loads new checkpoints into a queue, off the render thread."""

    def __init__(self, run: RunPaths) -> None:
        self.run_paths = run
        self.versions: queue.Queue[Version] = queue.Queue()
        self.loaded_steps = -1
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def poll_once(self) -> None:
        latest = self.run_paths.read_latest()
        if latest is None or latest.steps <= self.loaded_steps:
            return
        try:
            version = load_version(latest)
        except Exception:  # replaced or half written while reading: try again next time
            return
        self.loaded_steps = latest.steps
        self.versions.put(version)

    def start(self) -> None:
        def loop():
            while not self._stop.is_set():
                self.poll_once()
                self._stop.wait(POLL_SECONDS)

        self._thread = threading.Thread(target=loop, name="checkpoint-loader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


class LiveApp:
    def __init__(
        self,
        run: RunPaths,
        track_choice: str | None = None,
        ghosts: int = 2,
        scale: int | None = None,
        fullscreen: bool = False,
        realtime: bool = True,
        view: str = "3d",
        camera: str = "broadcast",
        variant: str = "day",
        track: Track | None = None,
        rays: bool = False,
        brain: bool = False,
    ) -> None:
        self.run_paths = run
        self.rays = rays
        self.brain = brain  # the brain panel (B)
        self.brain_panel: BrainPanel | None = None
        self._brain_target: PolicyDriver | None = None  # the driver whose V(s) is tracked
        self._brain_ms = 0.0
        self.view_mode = view
        self.camera = camera
        self.variant = variant
        self.game: GameView | None = None
        self._dt = DT
        self.config = TrainConfig.from_yaml(run.config)
        self.env_config = self.config.env_config()
        choice = track_choice or self.config.env.get("track") or "eval:1"
        self.track = track if track is not None else resolve_track(str(choice))
        self.max_ghosts = ghosts
        self.realtime = realtime
        self.screen = open_window(TITLE, scale, fullscreen)
        self.clock = pygame.time.Clock()
        self.loader = ModelLoader(run)
        self.metrics = MetricsReader(run.metrics)
        self.current: PolicyDriver | None = None
        self.ghosts: deque[PolicyDriver] = deque(maxlen=max(ghosts, 1))
        self.pending: Version | None = None
        self.notice: tuple[str, float] | None = None
        self.target_fps = 60
        self._fps_samples: deque[float] = deque(maxlen=120)
        self._metrics_checked = -math.inf
        self.running = False
        self.quit_requested = False  # the window was closed (the main menu closes too)
        self.canvas = pygame.Surface(ui.CANVAS_SIZE)  # the top-down view and its panels
        self._layout()

    # -- layout and versions ----------------------------------------------------------

    def _layout(self) -> None:
        width, height = self.screen.get_size()
        self.view_rect = pygame.Rect(0, 0, CANVAS_W - PANEL_W, CANVAS_H)
        self.panel_rect = pygame.Rect(CANVAS_W - PANEL_W, 0, PANEL_W, CANVAS_H)
        self.view = TopDownView(self.track, self.view_rect.size, pixel=True)
        self.chart = RewardChart((PANEL_W - 8, 54))
        self.brain_panel = None  # rebuilt on the next frame
        if self.view_mode == "3d" and self.game is None:
            self.game = GameView(
                self.track,
                (width, height),
                self.camera,
                self.variant,
                seed=self.track.decoration_seed,
                ramps=(0, 1, 2, 3),
            )
            self.chart3d = RewardChart((88, 40))

    def _install(self, version: Version) -> None:
        driver = PolicyDriver(version, self.track, self.env_config, ui.rgb(self._ink_of_ramp(0)))
        if self.current is not None and self.max_ghosts > 0:
            self._demote(self.current)
            self.ghosts.append(self.current)
        self.current = driver
        self.notice = (f"Nueva versión: {version.label}", time.perf_counter())
        if self.game is not None:
            self.game.notify(self.notice[0])

    @staticmethod
    def _ink_of_ramp(ramp: int) -> int:
        """The palette ink of a kart: the main colour of its ramp."""
        return KART_RAMPS[ramp % len(KART_RAMPS)][1]

    def _demote(self, driver: PolicyDriver, count: int | None = None) -> None:
        """Make ``driver`` a ghost: blue, purple or teal, one per earlier version."""
        driver.ramp = 1 + (len(self.ghosts) if count is None else count) % 3
        driver.color = ui.rgb(self._ink_of_ramp(driver.ramp))

    def preload_ghosts(self) -> None:
        """Start with earlier versions as ghosts, spaced on a log scale before the latest
        (about 1/30, 1/10 and 1/3 of its steps), so the contrast shows from the start."""
        latest = self.run_paths.read_latest()
        older = [c for c in self.run_paths.list_checkpoints() if latest and c.steps < latest.steps]
        if not older or self.max_ghosts == 0:
            return
        picks: list[Checkpoint] = []
        for fraction in (1 / 30, 1 / 10, 1 / 3)[-self.max_ghosts :]:
            target = latest.steps * fraction
            best = min(older, key=lambda c: abs(math.log1p(c.steps) - math.log1p(target)))
            if best not in picks:
                picks.append(best)
        for checkpoint in sorted(picks, key=lambda c: c.steps):
            driver = PolicyDriver(load_version(checkpoint), self.track, self.env_config, (0, 0, 0))
            self._demote(driver)
            self.ghosts.append(driver)

    def _take_new_versions(self) -> None:
        while True:
            try:
                version = self.loader.versions.get_nowait()
            except queue.Empty:
                break
            if self.current is None:
                self._install(version)
            else:
                self.pending = version  # enters at the end of the lap

    # -- main loop ----------------------------------------------------------------------

    def physics_step(self) -> None:
        if self.current is None:
            return
        event = self.current.step()
        if event is not None and self.pending is not None:
            version, self.pending = self.pending, None
            self._install(version)
        for ghost in self.ghosts:
            ghost.step()

    def run(self, max_frames: int | None = None) -> None:
        if self.realtime:
            self.loader.start()
        self.running = True
        frames = 0
        pending_time = 0.0
        try:
            while self.running:
                if self.realtime:
                    fps = self.game.target_fps if self.game is not None else self.target_fps
                    dt = self.clock.tick(fps) / 1000.0
                else:
                    self.clock.tick()
                    dt = DT
                self._dt = dt
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.quit_requested = True
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        self.handle_key(event.key)
                self._take_new_versions()
                if self.realtime:
                    pending_time = min(pending_time + dt, 0.1)
                    while pending_time >= DT:
                        self.physics_step()
                        pending_time -= DT
                else:
                    self.physics_step()
                self._sync_brain()
                now = time.perf_counter()
                if now - self._metrics_checked > METRICS_POLL_SECONDS:
                    self.metrics.refresh()
                    self._metrics_checked = now
                self._watch_fps()
                self.draw()
                pygame.display.flip()
                frames += 1
                if max_frames is not None and frames >= max_frames:
                    break
        finally:
            self.loader.stop()

    def handle_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_c and self.game is not None and self.view_mode == "3d":
            self.camera = self.game.cycle_camera()
        elif key == pygame.K_v and self.game is not None:
            self.variant = self.game.cycle_variant()
        elif key == pygame.K_b:
            self.brain = not self.brain
            if self.brain and self.brain_panel is not None:
                self.brain_panel.open(time.perf_counter())
        elif key == pygame.K_g:
            self.rays = not self.rays
        elif key == pygame.K_t:
            self.view_mode = "topdown" if self.view_mode == "3d" else "3d"
            self._layout()

    def _sync_brain(self) -> None:
        """V(s) is only computed for the current version, and only while the panel is open."""
        wanted = self.current if self.brain else None
        if wanted is not self._brain_target:
            if self._brain_target is not None:
                self._brain_target.set_tracking(False)
            if wanted is not None:
                wanted.set_tracking(True)
            self._brain_target = wanted

    def _draw_brain(
        self, target: pygame.Surface, area: pygame.Rect, top: int, bottom: int, now: float
    ) -> None:
        if not self.brain or self.current is None:
            return
        t0 = time.perf_counter()
        panel = self.brain_panel
        if panel is None:
            config = self.env_config
            panel = self.brain_panel = BrainPanel(
                config.ray_count, config.ray_fov_deg, config.ray_range
            )
            panel.open(now)
        panel.reserve = self.game.hud.reserve if self.game is not None else None
        pos = panel.position(area, "left", top, bottom, now)
        panel.draw(target, pos, self.current)
        self._brain_ms = self._brain_ms * 0.9 + (time.perf_counter() - t0) * 1000 * 0.1

    def _watch_fps(self) -> None:
        if not self.realtime or self.view_mode == "3d":
            return
        self._fps_samples.append(self.clock.get_fps())
        full = len(self._fps_samples) == self._fps_samples.maxlen
        if self.target_fps == 60 and full and np.mean(self._fps_samples) < LIGHT_MODE_FPS:
            self.target_fps = 30  # light mode: steady 30 beats an irregular 60

    # -- drawing ------------------------------------------------------------------------

    def draw(self) -> None:
        if self.view_mode == "3d":
            self._draw_3d()
        else:
            self._draw_topdown()

    def _draw_topdown(self) -> None:
        canvas = self.canvas
        view_surface = canvas.subsurface(self.view_rect)
        # (kart, colour, height, sweep state, height of the road): the last one hides the karts
        # that are in a bridge's tunnel, as in the driving view.
        karts = [(g.racer.kart, g.color, 0.0, None, g.racer.level_z) for g in self.ghosts]
        if self.current is not None:
            current = self.current
            karts.append((current.racer.kart, current.color, 0.0, None, current.racer.level_z))
        self.view.draw(view_surface, karts, (0.0, 0.0), follow=False)
        drivers = [*self.ghosts, self.current] if self.current else []
        placed: list[pygame.Rect] = []
        for driver in reversed(drivers):  # the current version keeps its spot
            kart = driver.racer.kart
            levels = self.track.levels
            if (
                levels is not None
                and driver.racer.level_z < 0.5
                and levels.in_tunnel_zone(kart.x, kart.y)
            ):
                continue  # in the tunnel: the view hides the kart, so its label goes too
            label = driver.version.label
            x, y = self.view.to_screen(kart.x, kart.y)
            rect = pygame.Rect(0, 0, ui.measure(label), 9)
            rect.midbottom = (round(x), round(y) - 5)
            while rect.collidelist(placed) != -1:  # karts side by side: stack the labels
                rect.move_ip(0, -rect.height)
            placed.append(rect)
            ink = self._ink_of_ramp(driver.ramp)
            ui.text(view_surface, label, rect.x, rect.y + 2, ink, outline=ui.NIGHT)
        self._draw_brain(canvas, self.view_rect, 3, CANVAS_H - 3, time.perf_counter())
        self._draw_panel(canvas)
        self._draw_notice(canvas)
        present(self.canvas, self.screen)

    def _draw_3d(self) -> None:
        game = self.game
        drivers = [*self.ghosts, self.current] if self.current else []
        if not drivers:
            self.canvas.fill(ui.rgb(ui.NIGHT))
            ui.text(
                self.canvas,
                "Esperando la primera versión…",
                CANVAS_W // 2,
                CANVAS_H // 2 - 3,
                ui.ORANGE,
                align="center",
            )
            present(self.canvas, self.screen)
            return
        visuals = [
            KartVisual(
                d.racer.kart,
                d.ramp,
                d.version.label,
                d is not self.current,
                d.racer.on_grass,
                d.throttle,
                list(d.trail),
                ground=d.racer.level_z,
            )
            for d in drivers
        ]
        pose = game.update(visuals, self._dt, focus=len(visuals) - 1)
        game.set_rays(visuals[-1] if self.rays else None)
        rows = self.metrics.rows
        last = rows[-1] if rows else None
        extra = [run_title(self.run_paths), self._status()[0]]
        if last is not None:
            extra.append(f"{format_steps(int(last['steps']))} pasos, {format_time(last['time_s'])}")
            extra.append(f"{ui.fmt_count(round(last['steps_per_s']))} pasos/s")
        if self.pending is not None:
            extra.append(f"Siguiente: {self.pending.label}")
        racer = self.current.racer
        best = min((d.best_lap for d in drivers if d.best_lap), default=None)
        info = HudInfo(
            laps=racer.laps,
            lap_time=racer.lap_time,
            best_lap=best,
            speed=racer.speed,
            label=f"Versión {self.current.version.label}",
            extra=extra,
        )
        fps = self.clock.get_fps() if self.realtime else None
        game.draw(
            self.screen,
            pose,
            visuals,
            info,
            self._dt,
            fps,
            pixel_ui=lambda surface: self._pixel_ui_3d(surface, game),
        )
        if self.brain:
            game.timings["cerebro"] = self._brain_ms
        else:
            game.timings.pop("cerebro", None)

    def _pixel_ui_3d(self, surface: pygame.Surface, game: GameView) -> None:
        """The interface of the 3D view, on the 320x180 world before it is magnified."""
        steps = self.current.version.steps
        chart = self.chart3d.draw(self.metrics.rows, self.metrics.version, steps)
        x, y = CANVAS_W - chart.get_width() - 4, (CANVAS_H - chart.get_height()) // 2
        game.hud.reserve((x, y, *chart.get_size()))
        surface.blit(chart, (x, y))
        if self.brain:
            top = 47  # under the lap panel
            bottom = CANVAS_H - game.hud.minimap_size[1] - MINIMAP_GAP
            self._draw_brain(surface, surface.get_rect(), top, bottom, game.now())

    def _status(self) -> tuple[str, str]:
        """What the run is doing, as a word and the icon that goes with it."""
        if self.current is None:
            return "Esperando la primera versión…", "ring"
        if self.run_paths.summary.exists():
            return "Entrenamiento terminado", "check"
        return "Entrenando", "play"

    def _draw_panel(self, canvas: pygame.Surface) -> None:
        rect = self.panel_rect
        ui.plate(canvas, rect, ui.SLATE)
        x, right = rect.x + 4, rect.right - 4
        inner = right - x
        y = 4
        rows = self.metrics.rows
        last = rows[-1] if rows else None
        for line in ui.wrap(run_title(self.run_paths), inner)[:2]:  # the name, on two lines if long
            ui.text(canvas, line, x, y, ui.TEXT)
            y += 9
        y += 1
        status, symbol = self._status()
        short = {"Entrenamiento terminado": "TERMINADO", "Entrenando": "ENTRENANDO"}.get(
            status, "ESPERANDO"
        )
        ink = ui.GREEN if symbol == "check" else ui.YELLOW if symbol == "play" else ui.ORANGE
        ui.icon(canvas, symbol, x, y + 1, ink)
        ui.text(canvas, short, x + 8, y + 1, ink)
        y += 11
        ui.hline(canvas, x, y - 2, inner, ui.NIGHT)
        ui.text(canvas, "CIRCUITO", x, y, ui.STONE_LIGHT)
        y += 9
        for line in ui.wrap_clip(self.track.name, inner, 3):  # split, not cut off
            ui.text(canvas, line, x, y, ui.TEXT)
            y += 9
        y += 2

        def row(label: str, value: str, ink_value: int = ui.TEXT) -> None:
            nonlocal y
            ui.text(canvas, label, x, y, ui.STONE_LIGHT)
            ui.text(canvas, value, right, y, ink_value, align="right")
            y += 9

        if last is not None:
            row("PASOS", format_steps(int(last["steps"])))
            row("TIEMPO", ui.fmt_clock(last["time_s"], 0))
            row("PASOS/S", ui.fmt_count(round(last["steps_per_s"])))
        y += 2
        if self.current is not None:
            drivers = [self.current, *self.ghosts]
            best = min((d.best_lap for d in drivers if d.best_lap), default=None)
            row("VERSIÓN", format_steps(self.current.version.steps), ui.ORANGE)
            row("VUELTA", ui.fmt_clock(self.current.racer.lap_time))
            row("MEJOR", ui.fmt_clock(best), ui.YELLOW if best is not None else ui.STONE)
            if self.pending is not None:
                row("SIGUIENTE", format_steps(self.pending.steps))
        if self.target_fps == 30:
            ui.icon(canvas, "warn", x, y + 1, ui.ORANGE)
            ui.text(canvas, "MODO LIGERO: SÍ", x + 8, y + 1, ui.ORANGE)
        marker = self.current.version.steps if self.current else None
        chart = self.chart.draw(rows, self.metrics.version, marker)
        canvas.blit(chart, (x - 1, rect.bottom - 4 - chart.get_height()))

    def _draw_notice(self, canvas: pygame.Surface) -> None:
        """The toast: a plate that slides down from the top edge, stays, and slides back up."""
        if self.notice is None:
            return
        value, start = self.notice
        age = time.perf_counter() - start
        if age > NOTICE_SECONDS:
            self.notice = None
            return
        height = 13
        enter = ui_ease(age / 0.4)
        leave = ui_ease((NOTICE_SECONDS - age) / 0.4)
        slide = round((1.0 - min(enter, leave)) * -(height + 1))
        width = ui.measure(value) + 10
        rect = pygame.Rect(self.view_rect.centerx - width // 2, 3 + slide, width, height)
        ui.plate(canvas, rect, ui.NAVY)
        ui.text(canvas, value, rect.centerx, rect.y + 3, ui.YELLOW, align="center")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Muestra un entrenamiento en directo.")
    parser.add_argument("--run", type=Path, help="carpeta del run (por defecto, el último)")
    parser.add_argument("--track", help="circuito (por defecto, el del entrenamiento o eval:1)")
    parser.add_argument(
        "--track-file", type=Path, help="circuito en fichero JSON (sustituye a --track)"
    )
    parser.add_argument("--ghosts", type=int, default=2, help="versiones anteriores (0-3)")
    parser.add_argument("--scale", type=int, help="escala entera sobre 320x180")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--view", choices=("3d", "topdown"), default="3d")
    parser.add_argument(
        "--camera",
        choices=("broadcast", "chase", "top"),
        default="broadcast",
        help="cámara 3D (por defecto, retransmisión: estable con modelos poco entrenados)",
    )
    parser.add_argument("--variant", choices=("day", "sunset", "night"), default="day")
    parser.add_argument("--rays", action="store_true", help="dibuja los sensores en el mundo (G)")
    parser.add_argument("--brain", action="store_true", help="abre el panel cerebro (B)")
    parser.add_argument(
        "--frames", type=int, help="sale tras N fotogramas, con un paso de física por fotograma"
    )
    parser.add_argument("--screenshot", type=Path, help="guarda el último fotograma en este PNG")
    args = parser.parse_args(argv)

    run = existing_run_or_exit(args.run) if args.run else latest_run()
    if run is None:
        raise SystemExit("No hay ninguna IA entrenada en runs/. Lanza antes scripts/train.py.")
    if not run.config.exists():
        raise SystemExit(f"{run.root} no tiene config.yaml")
    track = None
    if args.track_file:
        track = resolve_track_or_exit(str(args.track_file))
    elif args.track:
        track = resolve_track_or_exit(args.track)
    realtime = args.frames is None
    app = LiveApp(
        run,
        args.track,
        max(0, min(args.ghosts, 3)),
        args.scale,
        args.fullscreen,
        realtime,
        args.view,
        args.camera,
        args.variant,
        track,
        args.rays,
        args.brain,
    )
    try:
        app.preload_ghosts()
        if not realtime:
            app.loader.poll_once()  # no background thread: load the latest version now
        app.run(max_frames=args.frames)
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(app.screen, str(args.screenshot))
    finally:
        pygame.quit()
