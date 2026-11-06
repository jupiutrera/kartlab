"""Live training viewer: watches a run folder and shows each new version driving.

Run with ``uv run python scripts/live.py`` (latest run) or ``--run runs/<folder>``.
It is a separate process: closing it never affects training, and when training ends
it keeps showing the last version. It uses the pseudo-3D view (broadcast camera by
default, steady even with erratic early versions); T switches to the top-down view.
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

from kartlab.render.display import open_window
from kartlab.render.scene import KartVisual
from kartlab.render.topdown import (
    ACCENT,
    KART_COLOR,
    PANEL,
    TEXT,
    TEXT_DIM,
    WARNING,
    TextCache,
    TopDownView,
    format_time,
    load_fonts,
)
from kartlab.render.view import GameView, HudInfo
from kartlab.show.charts import MetricsReader, RewardChart
from kartlab.show.cli import existing_run_or_exit, resolve_track_or_exit
from kartlab.show.drivers import PolicyDriver, Version, format_steps, load_version
from kartlab.sim.kart import DT
from kartlab.sim.track import Track
from kartlab.sim.tracksets import resolve_track
from kartlab.train.config import TrainConfig
from kartlab.train.run import Checkpoint, RunPaths, latest_run

TITLE = "KART LAB · entrenamiento en directo"
GHOST_COLORS = ((110, 170, 255), (190, 140, 255), (120, 220, 190))
POLL_SECONDS = 1.0
METRICS_POLL_SECONDS = 2.0
NOTICE_SECONDS = 3.0
PANEL_FRACTION = 0.32  # share of the window width used by the side panel
LIGHT_MODE_FPS = 50.0  # below this (averaged over 2 s) the viewer drops to 30 fps


def ease_out(t: float) -> float:
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
    ) -> None:
        self.run_paths = run
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
        self._layout()

    # -- layout and versions ----------------------------------------------------------

    def _layout(self) -> None:
        width, height = self.screen.get_size()
        panel_w = int(width * PANEL_FRACTION)
        self.view_rect = pygame.Rect(0, 0, width - panel_w, height)
        self.panel_rect = pygame.Rect(width - panel_w, 0, panel_w, height)
        self.view = TopDownView(self.track, self.view_rect.size)
        regular, bold = load_fonts(height)
        self.text = TextCache(regular)
        self.title = TextCache(bold)
        pad = max(8, height // 60)
        self.pad = pad
        chart_h = int(height * 0.34)
        self.chart = RewardChart((panel_w - 2 * pad, chart_h), regular)
        self.panel_bg = pygame.Surface(self.panel_rect.size, pygame.SRCALPHA)
        self.panel_bg.fill(PANEL)
        if self.view_mode == "3d" and self.game is None:
            self.game = GameView(
                self.track,
                (width, height),
                self.camera,
                self.variant,
                seed=self.track.decoration_seed,
                ramps=(0, 1, 2, 3),
            )
            self.chart3d = RewardChart((int(width * 0.26), int(height * 0.2)), regular)

    def _install(self, version: Version) -> None:
        driver = PolicyDriver(version, self.track, self.env_config, KART_COLOR)
        if self.current is not None and self.max_ghosts > 0:
            self.current.color = GHOST_COLORS[len(self.ghosts) % len(GHOST_COLORS)]
            self.current.ramp = 1 + len(self.ghosts) % 3
            self.ghosts.append(self.current)
        self.current = driver
        self.notice = (f"Nueva versión: {version.label}", time.perf_counter())
        if self.game is not None:
            self.game.notify(self.notice[0])

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
            color = GHOST_COLORS[len(self.ghosts) % len(GHOST_COLORS)]
            driver = PolicyDriver(load_version(checkpoint), self.track, self.env_config, color)
            driver.ramp = 1 + len(self.ghosts) % 3
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
        elif key == pygame.K_t:
            self.view_mode = "topdown" if self.view_mode == "3d" else "3d"
            self._layout()

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
            return
        screen = self.screen
        view_surface = screen.subsurface(self.view_rect)
        karts = [(g.racer.kart, g.color) for g in self.ghosts]
        if self.current is not None:
            karts.append((self.current.racer.kart, self.current.color))
        self.view.draw(view_surface, karts, (0.0, 0.0), follow=False)
        drivers = [*self.ghosts, self.current] if self.current else []
        placed: list[pygame.Rect] = []
        for driver in reversed(drivers):  # the current version keeps its spot
            kart = driver.racer.kart
            label = self.text.get(driver.version.label, driver.color)
            x, y = self.view.to_screen(kart.x, kart.y)
            rect = label.get_rect(midbottom=(round(x), round(y) - 18))
            while rect.collidelist(placed) != -1:  # karts side by side: stack the labels
                rect.move_ip(0, -rect.height)
            placed.append(rect)
            view_surface.blit(label, rect)
        self._draw_panel()
        self._draw_notice()

    def _draw_3d(self) -> None:
        game = self.game
        drivers = [*self.ghosts, self.current] if self.current else []
        if not drivers:
            self.screen.fill((14, 14, 22))
            label = self.title.get("Esperando el primer checkpoint…", WARNING)
            self.screen.blit(label, label.get_rect(center=self.screen.get_rect().center))
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
            )
            for d in drivers
        ]
        pose = game.update(visuals, self._dt, focus=len(visuals) - 1)
        rows = self.metrics.rows
        last = rows[-1] if rows else None
        extra = [self.run_paths.root.name, self._status()[0]]
        if last is not None:
            extra.append(
                f"{format_steps(int(last['steps']))} pasos · {format_time(last['time_s'])}"
                f" · {last['steps_per_s']:.0f} pasos/s"
            )
        if self.pending is not None:
            extra.append(f"Siguiente: {self.pending.label} (al acabar la vuelta)")
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
        game.draw(self.screen, pose, visuals, info, self._dt, fps)
        chart = self.chart3d.draw(rows, self.metrics.version, self.current.version.steps)
        pad = self.pad
        x = game.world_rect.right - chart.get_width() - pad
        y = game.world_rect.centery - chart.get_height() // 2
        size = (chart.get_width() + 2 * pad, chart.get_height() + 2 * pad)
        background = pygame.Surface(size, pygame.SRCALPHA)
        background.fill(PANEL)
        self.screen.blit(background, (x - pad, y - pad))
        self.screen.blit(chart, (x, y))

    def _status(self) -> tuple[str, tuple[int, int, int]]:
        if self.current is None:
            return "Esperando el primer checkpoint…", WARNING
        if self.run_paths.summary.exists():
            return "Entrenamiento terminado", ACCENT
        return "Entrenando", ACCENT

    def _draw_panel(self) -> None:
        screen, pad = self.screen, self.pad
        screen.blit(self.panel_bg, self.panel_rect.topleft)
        x = self.panel_rect.x + pad
        y = pad
        rows = self.metrics.rows
        last = rows[-1] if rows else None
        finished = self.run_paths.summary.exists()
        if self.current is None:
            status, color = "Esperando el primer checkpoint…", WARNING
        elif finished:
            status, color = "Entrenamiento terminado", ACCENT
        else:
            status, color = "Entrenando", ACCENT
        title = self.title.get(self.run_paths.root.name, TEXT)
        screen.blit(title, (x, y))
        y += title.get_height() + 2
        lines = [(status, color), (f"Circuito: {self.track.name}", TEXT_DIM)]
        if last is not None:
            lines += [
                (f"Pasos: {format_steps(int(last['steps']))}", TEXT),
                (f"Tiempo: {format_time(last['time_s'])}", TEXT),
                (f"{last['steps_per_s']:.0f} pasos/s", TEXT_DIM),
            ]
        if self.current is not None:
            drivers = [self.current, *self.ghosts]
            best = min((d.best_lap for d in drivers if d.best_lap), default=None)
            lines += [
                (f"Versión: {self.current.version.label}", KART_COLOR),
                (f"Vuelta: {format_time(self.current.racer.lap_time)}", TEXT),
                (f"Mejor vuelta: {format_time(best)}", TEXT),
            ]
            if self.pending is not None:
                lines.append((f"Siguiente: {self.pending.label} (al acabar la vuelta)", TEXT_DIM))
        if self.target_fps == 30:
            lines.append(("Modo ligero (30 fps)", WARNING))
        for text, color in lines:
            surface = self.text.get(text, color)
            screen.blit(surface, (x, y))
            y += surface.get_height() + 2
        marker = self.current.version.steps if self.current else None
        chart = self.chart.draw(rows, self.metrics.version, marker)
        screen.blit(chart, (x, self.panel_rect.bottom - pad - chart.get_height()))

    def _draw_notice(self) -> None:
        if self.notice is None:
            return
        text, start = self.notice
        age = time.perf_counter() - start
        if age > NOTICE_SECONDS:
            self.notice = None
            return
        alpha = ease_out(age / 0.4) if age < NOTICE_SECONDS - 0.6 else (NOTICE_SECONDS - age) / 0.6
        label = self.title.get(text, ACCENT)
        box = pygame.Surface((label.get_width() + 2 * self.pad, label.get_height() + self.pad))
        box.fill((14, 16, 22))
        box.blit(label, (self.pad, self.pad // 2))
        box.set_alpha(round(255 * max(0.0, min(alpha, 1.0))))
        slide = round((1.0 - ease_out(age / 0.4)) * -box.get_height())
        cx = self.view_rect.centerx - box.get_width() // 2
        self.screen.blit(box, (cx, self.pad + slide))


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
    parser.add_argument(
        "--frames", type=int, help="sale tras N fotogramas, con un paso de física por fotograma"
    )
    parser.add_argument("--screenshot", type=Path, help="guarda el último fotograma en este PNG")
    args = parser.parse_args(argv)

    run = existing_run_or_exit(args.run) if args.run else latest_run()
    if run is None:
        raise SystemExit("No hay ningún run en runs/. Lanza antes scripts/train.py.")
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
