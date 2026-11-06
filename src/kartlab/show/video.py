"""Offline videos: progression, ghost race and demo on an unseen track.

Frames are rendered faster or slower than real time (it does not matter), one physics
step (1/60 s) per video frame, and piped straight to ffmpeg: no frame ever touches the
disk. Output is always 1920x1080 at exactly 60 fps (H.264, yuv420p). The 320x180 canvas,
with the world and the whole interface on it, is scaled x6 (exact). Actions are
deterministic, so the same run and track give the same video.

Run with ``uv run python scripts/make_video.py <progression|race|demo> --run runs/<folder>``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import imageio_ffmpeg
import pygame

from kartlab.render import pixelui as ui
from kartlab.render.scene import KartVisual
from kartlab.render.topdown import KART_COLOR, format_time
from kartlab.render.view import GameView, HudInfo
from kartlab.show.cli import existing_run_or_exit, resolve_track_or_exit
from kartlab.show.drivers import PolicyDriver, format_steps, load_version
from kartlab.show.overlays import (
    CARD_SECONDS,
    Overlays,
    ease_in_out,
    version_name,
    visual_of,
)
from kartlab.show.versions import (
    PROGRESSION_TARGETS,
    load_best_version,
    pick_checkpoints,
)
from kartlab.sim.kart import DT
from kartlab.sim.track import Track
from kartlab.sim.tracksets import resolve_track
from kartlab.train.config import TrainConfig
from kartlab.train.run import RunPaths

VIDEO_SIZE = (1920, 1080)
FPS = 60
CRF = 18
PRESET = "medium"
PIXEL_FORMAT = "bgr0"  # memory layout of a 32-bit pygame surface (little endian)
BGRX_MASKS = (0xFF0000, 0x00FF00, 0x0000FF)

FADE_IN = 0.8  # seconds from black at the start of a segment
FADE_OUT = 0.5  # seconds to black at its end
COUNTDOWN_SECONDS = 3.0
FINISH_SECONDS = 4.0  # race: how long the cars keep going after the winner's last lap
GRID_FIRST, GRID_ROW, GRID_SIDE = (
    4.0,
    7.0,
    2.4,
)  # start grid: m behind the line, per row, off-centre
MAX_RESTARTS = 2  # progression: a version that gets stuck this many times has shown enough
FINAL_FACTOR = 2.0  # the last (best) version gets this many times the time limit, to finish a lap
SUMMARY_SECONDS = 3.5  # demo: how long the results card stays up at the end

Progress = Callable[[int], None]  # called with the number of frames written so far


# -- encoding ------------------------------------------------------------------------------


class VideoWriter:
    """Pipes RGB frames to ffmpeg (libx264, yuv420p, 1080p60 by default).

    The file is written as ``<name>.part.mp4`` and renamed when closed, so a failed
    render never leaves a half-written video under the final name.
    """

    def __init__(
        self,
        path: Path,
        size: tuple[int, int] = VIDEO_SIZE,
        fps: int = FPS,
        crf: int = CRF,
        preset: str = PRESET,
    ) -> None:
        self.path = Path(path)
        self.size = size
        self.fps = fps
        self.frames = 0
        self._frame = bytearray(size[0] * size[1] * 4)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_name(f"{self.path.stem}.part{self.path.suffix}")
        self._gen = imageio_ffmpeg.write_frames(
            str(self._tmp),
            size,
            pix_fmt_in=PIXEL_FORMAT,
            fps=fps,
            quality=None,
            codec="libx264",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
            output_params=[
                "-crf", str(crf),
                "-preset", preset,
                "-vf", "scale=out_color_matrix=bt709:out_range=tv",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-r", str(fps),
                "-movflags", "+faststart",
            ],
        )  # fmt: skip
        self._gen.send(None)

    def write(self, surface: pygame.Surface) -> None:
        if surface.get_size() != self.size:
            raise ValueError(f"frame is {surface.get_size()}, the video is {self.size}")
        if surface.get_masks()[:3] != BGRX_MASKS or surface.get_pitch() != surface.get_width() * 4:
            raise ValueError("frames must be 32-bit surfaces with 0xFF0000 as the red mask")
        # One copy into a reusable buffer: no new 8 MB array per frame, and the surface is
        # not left locked (the generator keeps a reference to what it was sent).
        buffer = surface.get_buffer()
        try:
            self._frame[:] = memoryview(buffer).cast("B")
        finally:
            del buffer
        self._gen.send(self._frame)
        self.frames += 1

    def close(self) -> Path:
        self._gen.close()
        os.replace(self._tmp, self.path)
        return self.path

    def abort(self) -> None:
        try:
            self._gen.close()
        finally:
            self._tmp.unlink(missing_ok=True)

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


class VideoInfo(NamedTuple):
    size: tuple[int, int]
    fps: float
    frames: int
    seconds: float
    codec: str
    pixel_format: str


def probe_video(path: Path) -> VideoInfo:
    """Size, frame rate, frame count and codec of a video, read with ffmpeg."""
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    reader.close()
    return VideoInfo(
        tuple(meta["size"]), float(meta["fps"]), frames, seconds, meta["codec"], meta["pix_fmt"]
    )


# -- composition -----------------------------------------------------------------------------


class Compositor(Overlays):
    """Draws 1080p frames (3D world plus overlays) and hands them to the writer."""

    def __init__(
        self,
        track: Track,
        writer: VideoWriter,
        camera: str = "broadcast",
        variant: str = "day",
        ramps: tuple[int, ...] = (0,),
        progress: Progress | None = None,
    ) -> None:
        self.track = track
        self.writer = writer
        self.camera = camera
        self.variant = variant
        self.ramps = ramps
        self.progress = progress
        self.scene_start = 0
        self.screen = pygame.Surface(writer.size)
        super().__init__(self._new_game())

    def _new_game(self) -> GameView:
        game = GameView(self.track, self.writer.size, self.camera, self.variant, ramps=self.ramps)
        game.world.sprites.BUDGET_MS = math.inf  # offline: draw every sprite exactly
        self.scene_start = self.writer.frames
        return game

    def new_scene(self) -> None:
        """A fresh world (clean texture, camera and HUD entrance) for the next segment."""
        self.game = self._new_game()

    @property
    def scene_frame(self) -> int:
        return self.writer.frames - self.scene_start

    # -- per frame --------------------------------------------------------------------------

    def draw_world(self, visuals: list[KartVisual], focus: int, info: HudInfo) -> None:
        self.game.fixed_time = self.scene_frame / FPS
        pose = self.game.update(visuals, DT, focus)
        self.game.render_canvas(pose, visuals, info, DT)

    def commit(self) -> None:
        self.game.present(self.screen)
        self.writer.write(self.screen)
        if self.progress is not None and self.writer.frames % FPS == 0:
            self.progress(self.writer.frames)

    def segment_fade(self, frame: int, end_frame: int | None) -> float:
        """Black amount for ``frame`` of a segment that fades in and (if ended) out."""
        amount = 1.0 - ease_in_out(frame / (FADE_IN * FPS))
        if end_frame is not None:
            amount = max(amount, ease_in_out(1.0 - (end_frame - frame) / (FADE_OUT * FPS)))
        return amount


# -- choosing versions -----------------------------------------------------------------------


def _env_config(run: RunPaths):
    return TrainConfig.from_yaml(run.config).env_config()


def _warn_if_trained_on(run: RunPaths, choice: str, warn: Callable[[str], None]) -> None:
    trained = TrainConfig.from_yaml(run.config).env.get("track")
    if trained is not None and str(trained) == choice:
        warn(f"Aviso: el circuito '{choice}' es el del entrenamiento, no es un circuito inédito.")


# -- progression -----------------------------------------------------------------------------


def render_progression(
    run: RunPaths,
    out: Path,
    track_choice: str = "harbor",
    max_seconds: float = 30.0,
    camera: str = "broadcast",
    variant: str = "day",
    targets: tuple[int, ...] = PROGRESSION_TARGETS,
    progress: Progress | None = None,
) -> Path:
    """The same track driven by checkpoints on a log scale of steps, one after another.

    Each segment lasts one lap or ``max_seconds`` (twice that for the final version).
    Bad versions keep rolling on the grass; the driver restarts them at the line after
    getting stuck, and the segment ends after the second restart. A step counter and
    the reward curve with a marker show where in the training each version is.
    """
    from kartlab.show.charts import MetricsReader, RewardChart

    track = resolve_track(track_choice)
    env_config = _env_config(run)
    checkpoints = pick_checkpoints(run, targets)
    metrics = MetricsReader(run.metrics)
    metrics.refresh()

    with VideoWriter(out) as writer:
        comp = Compositor(track, writer, camera, variant, progress=progress)
        chart = RewardChart((92, 56), log_x=True)
        names = [format_steps(c.steps) for c in checkpoints]
        previous = 0
        for index, checkpoint in enumerate(checkpoints):
            if index > 0:
                comp.new_scene()
            version = load_version(checkpoint)
            driver = PolicyDriver(version, track, env_config, KART_COLOR)
            last = index == len(checkpoints) - 1
            max_frames = round(max_seconds * (FINAL_FACTOR if last else 1.0) * FPS)
            end_frame: int | None = None
            frame = 0
            while end_frame is None or frame < end_frame:
                event = driver.step()
                if end_frame is None and (
                    event == "lap" or frame + 1 >= max_frames or driver.restarts >= MAX_RESTARTS
                ):
                    end_frame = frame + round(FADE_OUT * FPS)
                racer = driver.racer
                info = HudInfo(
                    laps=racer.laps,
                    lap_time=racer.lap_time,
                    best_lap=driver.best_lap,
                    speed=racer.speed,
                    label="Versión sin entrenar"
                    if version.steps == 0
                    else f"Versión {version.label}",
                    extra=[track.name, f"{index + 1} de {len(checkpoints)}"],
                )
                comp.draw_world([visual_of(driver, 0)], 0, info)
                shown = round(
                    previous + (version.steps - previous) * ease_in_out(frame / (1.4 * FPS))
                )
                comp.counter(shown, "pasos de entrenamiento")
                comp.chart_panel(chart, metrics, version.steps)
                comp.timeline(names, index)
                comp.card(
                    f"Versión {version_name(version.steps)}" if version.steps else "Sin entrenar",
                    [track.name],
                    age=frame / FPS,
                )
                comp.dim(comp.segment_fade(frame, end_frame))
                comp.commit()
                frame += 1
            previous = version.steps
    return out


# -- ghost race --------------------------------------------------------------------------------


@dataclass
class Racer:
    driver: PolicyDriver
    ramp: int
    steps: int
    laps_done: int = 0


def render_race(
    run: RunPaths,
    out: Path,
    track_choice: str = "harbor",
    laps: int = 2,
    max_seconds: float = 240.0,
    camera: str = "broadcast",
    variant: str = "day",
    targets: tuple[int, ...] = PROGRESSION_TARGETS,
    progress: Progress | None = None,
) -> Path:
    """Every version at once on the same track, without collisions, from the same line.

    The best (last) version is the solid kart the camera follows; the rest are ghosts
    with a halo and a trail. They start on a two-column grid with the best version at
    the back, so it has to overtake the rest. Ends after ``laps`` laps of the best one.
    """
    track = resolve_track(track_choice)
    env_config = _env_config(run)
    checkpoints = pick_checkpoints(run, targets)
    racers: list[Racer] = []
    for rank, checkpoint in enumerate(reversed(checkpoints)):  # best first, ramp 0
        driver = PolicyDriver(load_version(checkpoint), track, env_config, KART_COLOR)
        driver.ramp = rank
        racers.append(Racer(driver, rank, checkpoint.steps))
    best = racers[0]
    for slot, racer in enumerate(reversed(racers)):  # the worst on pole, the best last
        side = -1.0 if slot % 2 == 0 else 1.0
        racer.driver.place(-GRID_FIRST - GRID_ROW * (slot // 2), side * GRID_SIDE)

    with VideoWriter(out) as writer:
        comp = Compositor(
            track, writer, camera, variant, ramps=tuple(r.ramp for r in racers), progress=progress
        )
        start_frames = round(COUNTDOWN_SECONDS * FPS)
        max_frames = start_frames + round(max_seconds * FPS)
        end_frame: int | None = None
        frame = 0
        while end_frame is None or frame < end_frame:
            racing = frame >= start_frames
            if racing:
                for r in racers:
                    if r.driver.step() == "lap":
                        r.laps_done += 1
                if end_frame is None and (best.laps_done >= laps or frame + 1 >= max_frames):
                    end_frame = frame + round(FINISH_SECONDS * FPS)
            visuals = [
                visual_of(
                    r.driver,
                    r.ramp,
                    None if not racing else _short_label(r.steps),
                    ghost=r is not best,
                )  # no labels under the countdown digit
                for r in racers
            ]
            racer = best.driver.racer
            info = HudInfo(
                laps=min(best.laps_done, laps - 1),
                lap_time=racer.lap_time,
                best_lap=best.driver.best_lap,
                speed=racer.speed,
                label="Carrera de versiones",
                extra=[
                    track.name,
                    f"{ui.plural(laps, 'vuelta', 'vueltas')}, cámara de retransmisión",
                ],
            )
            comp.draw_world(visuals, 0, info)
            order = sorted(racers, key=lambda r: -r.driver.racer.progress.distance)
            rows = [
                (r.ramp, version_name(r.steps), f"{min(r.laps_done + 1, laps)}/{laps}")
                for r in order
            ]
            comp.standings(rows, (3, 3))
            comp.countdown(frame, start_frames, FPS)
            if end_frame is not None:
                age = frame - (end_frame - round(FINISH_SECONDS * FPS))
                comp.card(
                    f"Ganador: {version_name(order[0].steps)}",
                    [f"{track.name}, {ui.plural(laps, 'vuelta', 'vueltas')}"],
                    age=age / FPS,
                    life=FINISH_SECONDS,
                )
            comp.dim(comp.segment_fade(frame, end_frame))
            comp.commit()
            frame += 1
    return out


def _short_label(steps: int) -> str:
    return "0" if steps == 0 else format_steps(steps)


# -- demo on an unseen track ---------------------------------------------------------------------


def render_demo(
    run: RunPaths,
    out: Path,
    track_choice: str = "eval:3",
    laps: int = 3,
    max_seconds: float = 300.0,
    camera: str = "chase",
    variant: str = "day",
    progress: Progress | None = None,
    warn: Callable[[str], None] = print,
) -> Path:
    """``best_model.zip`` driving ``laps`` laps on a track it never saw while training."""
    track = resolve_track(track_choice)
    _warn_if_trained_on(run, track_choice, warn)
    version = load_best_version(run)
    driver = PolicyDriver(version, track, _env_config(run), KART_COLOR)
    lap_times: list[float] = []
    intro = CARD_SECONDS + 1.0

    with VideoWriter(out) as writer:
        comp = Compositor(track, writer, camera, variant, progress=progress)
        max_frames = round(max_seconds * FPS)
        end_frame: int | None = None
        summary_frames = round(SUMMARY_SECONDS * FPS)
        frame = 0
        while end_frame is None or frame < end_frame:
            if driver.step() == "lap":
                lap_times.append(driver.racer.lap_times[-1])
            if end_frame is None and (len(lap_times) >= laps or frame + 1 >= max_frames):
                end_frame = frame + summary_frames
            racer = driver.racer
            finished = len(lap_times) >= laps
            info = HudInfo(
                laps=min(len(lap_times), laps - 1),
                lap_time=lap_times[-1] if finished else racer.lap_time,  # frozen at the flag
                best_lap=driver.best_lap,
                speed=racer.speed,
                label=f"Mejor versión: {format_steps(version.steps)} pasos",
                extra=[track.name, "Nunca visto en el entrenamiento"],
            )
            comp.draw_world([visual_of(driver, 0)], 0, info)
            comp.card(
                "Circuito nuevo",
                [track.name, "Nunca visto en el entrenamiento"],
                age=frame / FPS,
                life=intro,
            )
            if end_frame is not None:
                age = frame - (end_frame - summary_frames)
                best = min(lap_times) if lap_times else None
                lines = [f"Vuelta {i + 1}  {format_time(t)}" for i, t in enumerate(lap_times)]
                if best is not None:
                    lines.append(f"Mejor vuelta  {format_time(best)}")
                comp.card(
                    f"{len(lap_times)} de {ui.plural(laps, 'vuelta', 'vueltas')}",
                    lines or ["Sin vueltas"],
                    age=age / FPS,
                    life=summary_frames / FPS,
                )
            comp.dim(comp.segment_fade(frame, end_frame))
            comp.commit()
            frame += 1
    return out


# -- command line --------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    import argparse
    import sys
    import time

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Spanish text in any console
    parser = argparse.ArgumentParser(
        description="Vídeos 1080p a 60 fps de un entrenamiento, renderizados fuera de tiempo real."
    )
    parser.add_argument("kind", choices=("progression", "race", "demo"))
    parser.add_argument("--run", type=Path, help="carpeta del run (por defecto, el último)")
    parser.add_argument("--out", type=Path, help="vídeo de salida (por defecto, videos/<tipo>.mp4)")
    parser.add_argument(
        "--track",
        help="circuito: harbor, canyon, plateau, eval:N o semilla "
        "(por defecto harbor; en demo, eval:3)",
    )
    parser.add_argument(
        "--seconds", type=float, help="progression: máximo por versión (30); race/demo: tope total"
    )
    parser.add_argument("--laps", type=int, help="race: vueltas de la mejor (2); demo: vueltas (3)")
    parser.add_argument(
        "--camera",
        choices=("broadcast", "chase", "top"),
        help="cámara (por defecto retransmisión, y persecución en demo)",
    )
    parser.add_argument("--variant", choices=("day", "sunset", "night"), default="day")
    parser.add_argument(
        "--steps",
        help="pasos de las versiones separados por comas (por defecto escala logarítmica: "
        "0,10000,30000,100000,300000,1000000 y el final)",
    )
    parser.add_argument(
        "--normal-priority", action="store_true", help="no bajar la prioridad del proceso"
    )
    args = parser.parse_args(argv)

    from kartlab.train.run import latest_run

    if args.run:
        run = existing_run_or_exit(args.run)
    else:
        run = latest_run()
        if run is None or not run.config.exists():
            raise SystemExit("No hay ningún run válido. Usa --run runs/<carpeta>.")
    if args.track:
        resolve_track_or_exit(args.track)  # fail now, not after loading the checkpoints
    if not args.normal_priority:
        from kartlab.common.system import lower_priority

        lower_priority()  # ffmpeg inherits it: the machine stays usable while rendering
    targets = tuple(int(s) for s in args.steps.split(",")) if args.steps else PROGRESSION_TARGETS
    out = args.out or Path("videos") / f"{args.kind}.mp4"
    started = time.perf_counter()

    def progress(frames: int) -> None:
        if frames % (FPS * 10) == 0:
            elapsed = time.perf_counter() - started
            print(
                f"  {frames / FPS:6.0f} s de vídeo, {elapsed:5.0f} s de render"
                f" ({frames / elapsed:4.1f} fotogramas/s)",
                flush=True,
            )

    common = {"progress": progress, "variant": args.variant}
    if args.kind == "progression":
        render_progression(
            run, out, args.track or "harbor", args.seconds or 30.0,
            args.camera or "broadcast", targets=targets, **common,
        )  # fmt: skip
    elif args.kind == "race":
        render_race(
            run, out, args.track or "harbor", args.laps or 2, args.seconds or 240.0,
            args.camera or "broadcast", targets=targets, **common,
        )  # fmt: skip
    else:
        render_demo(
            run, out, args.track or "eval:3", args.laps or 3, args.seconds or 300.0,
            args.camera or "chase", **common,
        )  # fmt: skip
    info = probe_video(out)
    elapsed = time.perf_counter() - started
    print(
        f"{out}: {info.size[0]}x{info.size[1]}, {info.fps:g} fps, {info.frames} fotogramas,"
        f" {info.seconds:.1f} s"
    )
    print(f"Render: {elapsed:.0f} s ({info.frames / elapsed:.1f} fotogramas/s)")
