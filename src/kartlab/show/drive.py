"""Manual driving, in the pseudo-3D view (default) or the top-down debug view.

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
from kartlab.render.scene import KartVisual
from kartlab.render.topdown import (
    ACCENT,
    KART_COLOR,
    TEXT,
    TEXT_DIM,
    WARNING,
    DebugHud,
    TopDownView,
    format_time,
    load_fonts,
)
from kartlab.render.view import GameView, HudInfo
from kartlab.sim.autopilot import Autopilot
from kartlab.sim.drift import DriftState
from kartlab.sim.kart import DT
from kartlab.sim.race import Racer
from kartlab.sim.showcase import showcase_track
from kartlab.sim.track import Track
from kartlab.sim.tracksets import (
    EVAL_SEEDS,
    eval_track,
    resolve_track,
    sample_train_seed,
    train_track,
)

TITLE = "KART LAB · conducción manual"
TRAIL_SECONDS = 4.0
TRAIL_EVERY = 3  # physics steps between trail points
CAMERA_RESPONSE = 4.0  # top-down follow camera smoothing (1/s)
CAMERA_LEAD = 0.6  # the top-down follow camera aims this many seconds ahead (s)
MAX_PENDING = 0.1  # never simulate more than this after a stall (s)
SHOWCASE_KEYS = {pygame.K_1: "harbor", pygame.K_2: "canyon", pygame.K_3: "plateau"}
HELP = (
    "Flechas o WASD: conducir · R: reiniciar · 1-3: escaparate · E: evaluación · N: aleatorio",
    "C: cámara · V: día/atardecer/noche · G: sensores · T: vista cenital · Tab: piloto",
    "Mayús: barrido (deslizamiento): mantén con el volante girado y suelta para el impulso",
    "P: pausa · F3: diagnóstico · L: modo ligero · F11: pantalla completa · H: ayuda · Esc: salir",
)


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
    ) -> None:
        self.screen = open_window(TITLE, scale, fullscreen)
        self.scale = scale
        self.fullscreen = fullscreen
        self.autopilot_on = autopilot
        self.realtime = realtime
        self.rng = np.random.default_rng(seed)
        self.eval_index = -1
        self.view_mode = view
        self.camera3d = camera if camera in ("chase", "broadcast", "top") else "chase"
        self.follow = camera != "track"
        self.variant = variant
        self.paused = False
        self.show_help = view == "topdown"  # in 3D it would cover the HUD
        self.show_rays = False
        self.diagnostics = False
        self.running = False
        self.controls = (0.0, 0.0)
        self.drift = DriftState()
        self.clock = pygame.time.Clock()
        self.hud = DebugHud(self.screen.get_size())
        self.view: TopDownView | None = None
        self.game: GameView | None = None
        self.load_track(track)

    # -- tracks and views -------------------------------------------------------------

    def load_track(self, track: Track) -> None:
        self.track = track
        self.racer = Racer(track)
        self.autopilot = Autopilot(track)
        self.view = None
        self.game = None
        self._build_view()
        self.reset()

    def _build_view(self) -> None:
        size = self.screen.get_size()
        if self.view_mode == "3d" and self.game is None:
            self._loading()
            self.game = GameView(self.track, size, self.camera3d, self.variant)
            self.game.diagnostics = self.diagnostics
        elif self.view_mode == "topdown" and self.view is None:
            self.view = TopDownView(self.track, size)

    def _loading(self) -> None:
        font, _ = load_fonts(self.screen.get_height())
        self.screen.fill((14, 14, 22))
        text = font.render(f"Cargando {self.track.name}…", True, TEXT)
        self.screen.blit(text, text.get_rect(center=self.screen.get_rect().center))
        pygame.display.flip()

    def reset(self) -> None:
        self.racer.reset()
        self.drift.reset()
        self.camera = (self.racer.kart.x, self.racer.kart.y)
        self.trail: deque[tuple[float, float]] = deque(maxlen=int(TRAIL_SECONDS / DT / TRAIL_EVERY))
        self._steps = 0
        if self.game is not None:
            for camera in self.game.cameras.values():
                camera.pose = None

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self.screen = set_window_mode(self.scale, self.fullscreen)
        size = self.screen.get_size()
        self.hud.resize(size)
        if self.view is not None:
            self.view.resize(size)
        if self.game is not None:
            self.game.resize(size)

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
            if self.game is not None and self.view_mode == "3d":
                self.camera3d = self.game.cycle_camera()
            else:
                self.follow = not self.follow
        elif key == pygame.K_t:
            self.view_mode = "topdown" if self.view_mode == "3d" else "3d"
            self._build_view()
        elif key == pygame.K_v and self.game is not None:
            self.variant = self.game.cycle_variant()
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
        if self.autopilot_on:
            steer, throttle = self.autopilot.act(self.racer)
            drift = None  # the sweep is for manual driving only
        else:
            pressed = pygame.key.get_pressed()
            steer, throttle = keyboard_controls(pressed)
            self.drift.hold = drift_pressed(pressed)
            drift = self.drift
        self.controls = (steer, throttle)
        self.racer.step(steer, throttle, drift=drift)
        self._steps += 1
        if self._steps % TRAIL_EVERY == 0:
            self.trail.append((self.racer.kart.x, self.racer.kart.y))

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
            0,
            on_grass=racer.on_grass,
            throttle=self.controls[1],
            drift=None if self.autopilot_on else self.drift,
        )
        pose = self.game.update([visual], 0.0 if self.paused else dt)
        self.game.set_rays(visual if self.show_rays else None)
        extra = [self.track.name]
        if self.paused:
            extra.append("PAUSA")
        if not self.show_help:
            extra.append("H: ayuda")
        info = HudInfo(
            laps=racer.laps,
            lap_time=racer.lap_time,
            best_lap=racer.best_lap,
            speed=racer.speed,
            label="Piloto automático" if self.autopilot_on else "Conducción manual",
            extra=extra,
            drift=None if self.autopilot_on else self.drift,
        )
        fps = self.clock.get_fps() if self.realtime else None
        self.game.draw(self.screen, pose, [visual], info, dt, fps)
        if self.show_help:
            self.hud.draw_block(self.screen, [(t, TEXT_DIM) for t in HELP], "bottom-left")

    def update_camera(self, dt: float) -> None:
        """North-up follow camera that eases towards a point ahead of the kart (never rotates)."""
        kart = self.racer.kart
        tx = kart.x + kart.vx * CAMERA_LEAD
        ty = kart.y + kart.vy * CAMERA_LEAD
        k = 1.0 - math.exp(-CAMERA_RESPONSE * dt)
        cx, cy = self.camera
        self.camera = (cx + (tx - cx) * k, cy + (ty - cy) * k)

    def draw_topdown(self) -> None:
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
        if not self.autopilot_on and self.drift.active:
            status.append((f"Barrido {self.drift.charge * 100:3.0f} %", ACCENT))
        elif not self.autopilot_on and self.drift.pushing:
            status.append((f"Impulso {self.drift.push_level * 100:3.0f} %", ACCENT))
        if self.paused:
            status.append(("PAUSA", WARNING))
        self.hud.draw_block(self.screen, status, "top-right")
        if self.show_help:
            self.hud.draw_block(self.screen, [(text, TEXT_DIM) for text in HELP], "bottom-left")

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
    parser.add_argument("--rays", action="store_true", help="muestra los sensores")
    parser.add_argument("--diagnostics", action="store_true", help="tiempos de cada parte")
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
        view=args.view,
        camera=args.camera,
        variant=args.variant,
        realtime=args.frames is None,
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
