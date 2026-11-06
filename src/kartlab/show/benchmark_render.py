"""Render benchmark: time of each part of a pseudo-3D frame, without a window.

One kart and three ghosts drive with the autopilot on a showcase track while each
camera renders at 1280x720 and 1920x1080. Run with
``uv run python scripts/benchmark_render.py``. The window flip is not included
(there is no real window); the diagnostics overlay (F3 in ``drive.py``) shows the
live numbers on screen.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from kartlab.render.scene import KartVisual  # noqa: E402
from kartlab.render.view import GameView, HudInfo  # noqa: E402
from kartlab.sim.autopilot import Autopilot  # noqa: E402
from kartlab.sim.drift import DriftState  # noqa: E402
from kartlab.sim.kart import DT  # noqa: E402
from kartlab.sim.race import Racer  # noqa: E402
from kartlab.sim.tracksets import resolve_track  # noqa: E402


def stage_drift(drift: DriftState, frame: int) -> None:
    """Fake a 2 s sweep followed by a 1 s push, over and over, to exercise the visuals."""
    phase = frame % 180
    if phase < 120:
        drift.active, drift.direction, drift.time = True, 1.0, phase / 60
        drift.push_left = 0.0
    else:
        if drift.active:
            drift.push_id += 1
            drift.push_speed, drift.push_total = 6.0, 1.3
        drift.active = False
        drift.push_left = max(1.3 - (phase - 120) / 60, 0.01)


def bench(
    track_choice: str,
    size: tuple[int, int],
    camera: str,
    frames: int,
    light: bool,
    with_drift: bool = False,
):
    screen = pygame.display.set_mode(size)
    track = resolve_track(track_choice)
    view = GameView(track, size, camera, light_mode=light, ramps=(0, 1, 2, 3))
    view.auto_light = False
    racers = [Racer(track) for _ in range(4)]
    for k, racer in enumerate(racers):
        racer.reset(start_s=-10.0 * k)
    pilots = [Autopilot(track) for _ in racers]
    trails: list[list[tuple[float, float]]] = [[] for _ in racers]
    drift = DriftState() if with_drift else None
    parts: dict[str, list[float]] = defaultdict(list)
    totals = []
    for frame in range(frames + 120):  # the first 2 s warm the sprite caches
        for k, (racer, pilot) in enumerate(zip(racers, pilots, strict=True)):
            racer.step(*pilot.act(racer))
            if frame % 6 == 0:
                trails[k] = ([(racer.kart.x, racer.kart.y)] + trails[k])[:14]
        if drift is not None:
            stage_drift(drift, frame)
        visuals = [
            KartVisual(r.kart, k, None if k == 0 else f"{k * 100}k pasos", k > 0, r.on_grass, 1.0,
                       trails[k])
            for k, r in enumerate(racers)
        ]  # fmt: skip
        if drift is not None:
            visuals[0].drift = drift
        start = time.perf_counter()
        pose = view.update(visuals, DT)
        info = HudInfo(racers[0].laps, racers[0].lap_time, racers[0].best_lap, racers[0].speed)
        view.draw(screen, pose, visuals, info, DT)
        elapsed = (time.perf_counter() - start) * 1000
        if frame >= 120:
            totals.append(elapsed)
            for key, value in view.last_timings.items():
                parts[key].append(value)
    return {k: float(np.mean(v)) for k, v in parts.items()}, np.array(totals)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mide el tiempo de cada parte del render.")
    parser.add_argument("--track", default="harbor")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--light", action="store_true", help="modo ligero")
    parser.add_argument(
        "--drift", action="store_true", help="el kart principal barre y da impulso en bucle"
    )
    args = parser.parse_args(argv)
    pygame.display.init()
    print(f"{'ventana':>10} {'cámara':>10}  media   p99  partes (ms)")
    for size in ((1280, 720), (1920, 1080)):
        for camera in ("chase", "broadcast", "top"):
            parts, totals = bench(args.track, size, camera, args.frames, args.light, args.drift)
            detail = "  ".join(f"{k} {v:.1f}" for k, v in parts.items())
            p99 = np.percentile(totals, 99)
            window = f"{size[0]}x{size[1]}"
            print(
                f"{window:>10} {camera:>10} {totals.mean():5.1f} {p99:5.1f}  {detail}", flush=True
            )
    pygame.quit()
