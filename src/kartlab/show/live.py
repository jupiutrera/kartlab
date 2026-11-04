"""Live training viewer: watches a run folder and shows each new version driving.

Run with ``uv run python scripts/live.py`` (latest run) or ``--run runs/<folder>``.
It is a separate process: closing it never affects training, and when training ends
it keeps showing the last version. New checkpoints load in a background thread and
enter at the end of a lap (or when the kart is restarted after getting stuck).
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pygame

from kartlab.env.kart_env import EnvConfig, KartEnv
from kartlab.render.display import open_window
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
from kartlab.sim.kart import DT
from kartlab.sim.track import Track
from kartlab.sim.tracksets import resolve_track
from kartlab.train.config import TrainConfig
from kartlab.train.evaluate import model_predictor
from kartlab.train.run import Checkpoint, RunPaths, latest_run

TITLE = "KART LAB · entrenamiento en directo"
GHOST_COLORS = ((110, 170, 255), (190, 140, 255), (120, 220, 190))
RESTART_SECONDS = 3.0  # stuck or off the track this long: back to the start line
POLL_SECONDS = 1.0
METRICS_POLL_SECONDS = 2.0
NOTICE_SECONDS = 3.0
PANEL_FRACTION = 0.32  # share of the window width used by the side panel
LIGHT_MODE_FPS = 50.0  # below this (averaged over 2 s) the viewer drops to 30 fps


def format_steps(steps: int) -> str:
    if steps >= 1_000_000:
        return f"{steps / 1_000_000:.1f}M".replace(".", ",")
    if steps >= 1000:
        return f"{steps // 1000}k"
    return str(steps)


def ease_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1.0 - (1.0 - t) ** 3


@dataclass
class Version:
    steps: int
    predict: object  # batch obs -> batch actions

    @property
    def label(self) -> str:
        return f"{format_steps(self.steps)} pasos"


def load_version(checkpoint: Checkpoint) -> Version:
    from stable_baselines3 import PPO

    model = PPO.load(checkpoint.model, device="cpu")
    return Version(checkpoint.steps, model_predictor(model, checkpoint.vecnormalize))


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
        except (OSError, ValueError, EOFError, KeyError):
            return  # replaced while reading: try again next time
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


class PolicyDriver:
    """One kart driven by a policy, advanced one physics step per call."""

    def __init__(self, version: Version, track: Track, env_config: EnvConfig, color) -> None:
        self.version = version
        self.color = color
        config = replace(
            env_config,
            track=None,
            random_start=False,
            terminate_off_track=False,
            stuck_seconds=math.inf,
            max_seconds=math.inf,
            max_laps=None,
        )
        self.env = KartEnv(config)
        self.env.reset(seed=0, options={"track": track})
        self.repeat = config.action_repeat
        self.best_lap: float | None = None
        self.restarts = 0
        self._phase = 0
        self._action = (0.0, 0.0)
        self._best_distance = 0.0
        self._last_progress_time = 0.0
        self._off_since: float | None = None

    @property
    def racer(self):
        return self.env.racer

    def restart(self) -> None:
        self.racer.reset()
        self.env.hold_action(np.zeros(2))
        self._phase = 0
        self._best_distance = 0.0
        self._last_progress_time = 0.0
        self._off_since = None
        self.restarts += 1

    def step(self) -> str | None:
        """Advance one physics step; returns "lap" or "restart" when one happens."""
        racer = self.racer
        if self._phase == 0:
            obs = self.env.observe()
            action = self.version.predict(obs[None])[0]
            self._action = self.env.hold_action(action)
        self._phase = (self._phase + 1) % self.repeat
        laps = racer.laps
        racer.step(*self._action)

        if racer.progress.distance > self._best_distance + 1.0:
            self._best_distance = racer.progress.distance
            self._last_progress_time = racer.time
        off = abs(racer.position.lateral) > self.env.track.half_width + 0.5
        if off and self._off_since is None:
            self._off_since = racer.time
        elif not off:
            self._off_since = None
        if racer.laps > laps:
            lap = racer.lap_times[-1]
            self.best_lap = lap if self.best_lap is None else min(self.best_lap, lap)
            return "lap"
        stuck = racer.time - self._last_progress_time > RESTART_SECONDS
        lost = self._off_since is not None and racer.time - self._off_since > RESTART_SECONDS
        if stuck or lost:
            self.restart()
            return "restart"
        return None


class MetricsReader:
    """Reads metrics.csv when it changes; tolerates a half-written last line."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: list[dict[str, float]] = []
        self.version = 0  # bumps whenever rows change
        self._mtime = None

    def refresh(self) -> None:
        try:
            mtime = self.path.stat().st_mtime_ns
            if mtime == self._mtime:
                return
            text = self.path.read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError):
            return
        self._mtime = mtime
        lines = text.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines = lines[:-1]  # still being written
        rows = []
        for row in csv.DictReader(io.StringIO("".join(lines))):
            try:
                rows.append({k: float(v) for k, v in row.items()})
            except (TypeError, ValueError):
                continue
        if len(rows) != len(self.rows):
            self.rows = rows
            self.version += 1


class RewardChart:
    """Mean episode reward against training steps, redrawn only when data changes."""

    def __init__(self, size: tuple[int, int], font: pygame.font.Font) -> None:
        self.size = size
        self.font = font
        self.surface = pygame.Surface(size, pygame.SRCALPHA)
        self._key = None

    def draw(self, rows: list[dict[str, float]], data_version: int, marker: int | None):
        key = (data_version, marker, self.size)
        if key == self._key:
            return self.surface
        self._key = key
        surf = self.surface
        surf.fill((0, 0, 0, 0))
        width, height = self.size
        pad = 6
        title = self.font.render("Recompensa media por episodio", True, TEXT_DIM)
        surf.blit(title, (0, 0))
        top = title.get_height() + pad
        box = pygame.Rect(0, top, width, height - top)
        pygame.draw.rect(surf, (255, 255, 255, 30), box, 1)
        points = [(r["steps"], r["reward_mean"]) for r in rows if not math.isnan(r["reward_mean"])]
        if len(points) < 2:
            msg = self.font.render("Esperando datos…", True, TEXT_DIM)
            surf.blit(msg, (box.x + pad, box.y + pad))
            return surf
        xs, ys = np.array(points).T
        x0, x1 = 0.0, max(xs.max(), 1.0)
        y0, y1 = min(ys.min(), 0.0), max(ys.max(), 1.0)

        def to_px(x, y):
            px = box.x + pad + (x - x0) / (x1 - x0) * (box.width - 2 * pad)
            py = box.bottom - pad - (y - y0) / (y1 - y0) * (box.height - 2 * pad)
            return px, py

        if y0 < 0:
            pygame.draw.line(surf, (255, 255, 255, 40), to_px(x0, 0), to_px(x1, 0))
        pygame.draw.aalines(surf, ACCENT, False, [to_px(x, y) for x, y in points])
        if marker is not None:
            mx = to_px(min(marker, x1), y0)[0]
            pygame.draw.line(surf, KART_COLOR, (mx, box.y + pad), (mx, box.bottom - pad), 2)
        for value, y in ((y1, box.y + pad), (y0, box.bottom - pad - self.font.get_height())):
            label = self.font.render(f"{value:.0f}", True, TEXT_DIM)
            surf.blit(label, (box.right - label.get_width() - pad, y))
        return surf


class LiveApp:
    def __init__(
        self,
        run: RunPaths,
        track_choice: str | None = None,
        ghosts: int = 2,
        scale: int | None = None,
        fullscreen: bool = False,
        realtime: bool = True,
    ) -> None:
        self.run_paths = run
        self.config = TrainConfig.from_yaml(run.config)
        self.env_config = self.config.env_config()
        choice = track_choice or self.config.env.get("track") or "eval:1"
        self.track = resolve_track(str(choice))
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

    def _install(self, version: Version) -> None:
        driver = PolicyDriver(version, self.track, self.env_config, KART_COLOR)
        if self.current is not None and self.max_ghosts > 0:
            self.current.color = GHOST_COLORS[len(self.ghosts) % len(GHOST_COLORS)]
            self.ghosts.append(self.current)
        self.current = driver
        self.notice = (f"Nueva versión: {version.label}", time.perf_counter())

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
                    dt = self.clock.tick(self.target_fps) / 1000.0
                else:
                    self.clock.tick()
                    dt = DT
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        self.running = False
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

    def _watch_fps(self) -> None:
        if not self.realtime:
            return
        self._fps_samples.append(self.clock.get_fps())
        full = len(self._fps_samples) == self._fps_samples.maxlen
        if self.target_fps == 60 and full and np.mean(self._fps_samples) < LIGHT_MODE_FPS:
            self.target_fps = 30  # light mode: steady 30 beats an irregular 60

    # -- drawing ------------------------------------------------------------------------

    def draw(self) -> None:
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
    parser.add_argument("--ghosts", type=int, default=2, help="versiones anteriores (0-3)")
    parser.add_argument("--scale", type=int, help="escala entera sobre 320x180")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument(
        "--frames", type=int, help="sale tras N fotogramas, con un paso de física por fotograma"
    )
    parser.add_argument("--screenshot", type=Path, help="guarda el último fotograma en este PNG")
    args = parser.parse_args(argv)

    run = RunPaths(args.run) if args.run else latest_run()
    if run is None:
        raise SystemExit("No hay ningún run en runs/. Lanza antes scripts/train.py.")
    if not run.config.exists():
        raise SystemExit(f"{run.root} no tiene config.yaml")
    realtime = args.frames is None
    app = LiveApp(
        run, args.track, max(0, min(args.ghosts, 3)), args.scale, args.fullscreen, realtime
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
