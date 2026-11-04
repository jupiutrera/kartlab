"""Manual driving on the top-down debug view.

Run with ``uv run python scripts/drive.py`` (``--help`` lists the options).
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path

import numpy as np
import pygame

from kartlab.render.display import open_window, set_window_mode
from kartlab.render.topdown import (
    ACCENT,
    KART_COLOR,
    TEXT,
    TEXT_DIM,
    WARNING,
    DebugHud,
    TopDownView,
    format_time,
)
from kartlab.sim.autopilot import Autopilot
from kartlab.sim.kart import DT
from kartlab.sim.race import Racer
from kartlab.sim.showcase import SHOWCASE, showcase_track
from kartlab.sim.track import Track
from kartlab.sim.tracksets import EVAL_SEEDS, eval_track, sample_train_seed, train_track

TITLE = "KART LAB · conducción manual"
TRAIL_SECONDS = 4.0
TRAIL_EVERY = 3  # physics steps between trail points
CAMERA_RESPONSE = 4.0  # follow camera smoothing (1/s)
CAMERA_LEAD = 0.6  # the follow camera aims this many seconds ahead of the kart (s)
MAX_PENDING = 0.1  # never simulate more than this after a stall (s)
SHOWCASE_KEYS = {pygame.K_1: "harbor", pygame.K_2: "canyon", pygame.K_3: "plateau"}
HELP = (
    "Flechas o WASD: conducir · R: reiniciar · 1-3: escaparate · E: evaluación · N: aleatorio",
    "C: cámara · Tab: piloto automático · P: pausa · F11: pantalla completa · H: ayuda"
    " · Esc: salir",
)


def keyboard_controls(pressed) -> tuple[float, float]:
    """Steer and throttle from the arrow keys or WASD (steer > 0 turns left)."""
    left = pressed[pygame.K_LEFT] or pressed[pygame.K_a]
    right = pressed[pygame.K_RIGHT] or pressed[pygame.K_d]
    up = pressed[pygame.K_UP] or pressed[pygame.K_w]
    down = pressed[pygame.K_DOWN] or pressed[pygame.K_s]
    return float(left) - float(right), float(up) - float(down)


def resolve_track(choice: str) -> Track:
    """A showcase key, ``eval:N`` (1-based) or a training seed."""
    if choice in SHOWCASE:
        return showcase_track(choice)
    if choice.startswith("eval:"):
        return eval_track(int(choice.removeprefix("eval:")) - 1)
    return train_track(int(choice))


class DriveApp:
    def __init__(
        self,
        track: Track,
        scale: int | None = None,
        fullscreen: bool = False,
        autopilot: bool = False,
        follow: bool = True,
        realtime: bool = True,
        seed: int = 0,
    ) -> None:
        self.screen = open_window(TITLE, scale, fullscreen)
        self.scale = scale
        self.fullscreen = fullscreen
        self.autopilot_on = autopilot
        self.realtime = realtime
        self.rng = np.random.default_rng(seed)
        self.eval_index = -1
        self.follow = follow
        self.paused = False
        self.show_help = True
        self.running = False
        self.clock = pygame.time.Clock()
        self.hud = DebugHud(self.screen.get_size())
        self.load_track(track)

    def load_track(self, track: Track) -> None:
        self.track = track
        self.racer = Racer(track)
        self.autopilot = Autopilot(track)
        self.view = TopDownView(track, self.screen.get_size())
        self.reset()

    def reset(self) -> None:
        self.racer.reset()
        self.camera = (self.racer.kart.x, self.racer.kart.y)
        self.trail: deque[tuple[float, float]] = deque(maxlen=int(TRAIL_SECONDS / DT / TRAIL_EVERY))
        self._steps = 0

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self.screen = set_window_mode(self.scale, self.fullscreen)
        size = self.screen.get_size()
        self.view.resize(size)
        self.hud.resize(size)

    def handle_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_r:
            self.reset()
        elif key in SHOWCASE_KEYS:
            self.load_track(showcase_track(SHOWCASE_KEYS[key]))
        elif key == pygame.K_e:
            self.eval_index = (self.eval_index + 1) % len(EVAL_SEEDS)
            self.load_track(eval_track(self.eval_index))
        elif key == pygame.K_n:
            self.load_track(train_track(sample_train_seed(self.rng)))
        elif key == pygame.K_c:
            self.follow = not self.follow
        elif key == pygame.K_TAB:
            self.autopilot_on = not self.autopilot_on
        elif key == pygame.K_p:
            self.paused = not self.paused
        elif key == pygame.K_F11:
            self.toggle_fullscreen()
        elif key == pygame.K_h:
            self.show_help = not self.show_help

    def physics_step(self) -> None:
        if self.autopilot_on:
            steer, throttle = self.autopilot.act(self.racer)
        else:
            steer, throttle = keyboard_controls(pygame.key.get_pressed())
        self.racer.step(steer, throttle)
        self._steps += 1
        if self._steps % TRAIL_EVERY == 0:
            self.trail.append((self.racer.kart.x, self.racer.kart.y))

    def run(self, max_frames: int | None = None) -> None:
        """Main loop. Realtime mode keeps a fixed 1/60 s physics step with an accumulator;
        otherwise each frame advances exactly one step (deterministic, for captures)."""
        self.running = True
        frames = 0
        pending = 0.0
        while self.running:
            if self.realtime:
                dt = self.clock.tick(60) / 1000.0
            else:
                self.clock.tick()
                dt = DT
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
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
            self.update_camera(dt)
            self.draw()
            pygame.display.flip()
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break

    def update_camera(self, dt: float) -> None:
        """North-up follow camera that eases towards a point ahead of the kart (never rotates)."""
        kart = self.racer.kart
        tx = kart.x + kart.vx * CAMERA_LEAD
        ty = kart.y + kart.vy * CAMERA_LEAD
        k = 1.0 - math.exp(-CAMERA_RESPONSE * dt)
        cx, cy = self.camera
        self.camera = (cx + (tx - cx) * k, cy + (ty - cy) * k)

    def draw(self) -> None:
        racer = self.racer
        self.view.draw(
            self.screen, [(racer.kart, KART_COLOR)], self.camera, self.follow, self.trail
        )
        lines = [
            (f"Vuelta     {racer.laps + 1}", TEXT),
            (f"Tiempo     {format_time(racer.lap_time)}", TEXT),
            (f"Última     {format_time(racer.last_lap)}", TEXT_DIM),
            (f"Mejor      {format_time(racer.best_lap)}", ACCENT if racer.best_lap else TEXT_DIM),
            (f"Velocidad  {abs(racer.speed) * 3.6:3.0f} km/h", TEXT),
            (
                f"Superficie {'hierba' if racer.on_grass else 'asfalto'}",
                WARNING if racer.on_grass else TEXT_DIM,
            ),
        ]
        self.hud.draw_block(self.screen, lines, "top-left", title=self.track.name)
        status = [
            ("Piloto automático" if self.autopilot_on else "Manual", ACCENT),
            (f"Cámara: {'seguimiento' if self.follow else 'circuito'}", TEXT_DIM),
            (f"{self.clock.get_fps():3.0f} fps", TEXT_DIM),
        ]
        if self.paused:
            status.append(("PAUSA", WARNING))
        self.hud.draw_block(self.screen, status, "top-right")
        if self.show_help:
            self.hud.draw_block(self.screen, [(text, TEXT_DIM) for text in HELP], "bottom-left")

    def save_screenshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.screen, str(path))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Conduce un kart a mano en la vista cenital.")
    parser.add_argument(
        "--track",
        default="harbor",
        help="harbor, canyon, plateau, eval:N (1-20) o una semilla de entrenamiento",
    )
    parser.add_argument(
        "--scale", type=int, help="escala entera sobre 320x180 (por defecto, la mayor que quepa)"
    )
    parser.add_argument("--fullscreen", action="store_true", help="pantalla completa")
    parser.add_argument("--autopilot", action="store_true", help="empieza con piloto automático")
    parser.add_argument(
        "--camera",
        choices=("follow", "track"),
        default="follow",
        help="follow: sigue al kart; track: circuito completo",
    )
    parser.add_argument(
        "--frames", type=int, help="sale tras N fotogramas, con un paso de física por fotograma"
    )
    parser.add_argument("--screenshot", type=Path, help="guarda el último fotograma en este PNG")
    args = parser.parse_args(argv)

    app = DriveApp(
        resolve_track(args.track),
        scale=args.scale,
        fullscreen=args.fullscreen,
        autopilot=args.autopilot,
        follow=args.camera == "follow",
        realtime=args.frames is None,
    )
    try:
        app.run(max_frames=args.frames)
        if args.screenshot:
            app.save_screenshot(args.screenshot)
    finally:
        pygame.quit()
