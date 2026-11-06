"""Window composition for the pseudo-3D view.

The 320x180 world is the canvas of the whole game: the HUD, the panels of every mode and the
menus are drawn on it with the pixel interface kit (``pixelui``) and the canvas is then
scaled by the largest integer factor that fits the window (no smoothing, black bars if
needed). A diagnostics line shows the time of every part.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pygame

from kartlab.render.camera import CAMERAS, CameraPose
from kartlab.render.hud import (
    CAMERA_NAMES,
    VARIANT_NAMES,
    Hud,
    Smoothed,
    ease_out,
    format_delta,
)
from kartlab.render.scene import WORLD_SIZE, KartVisual, WorldRenderer
from kartlab.render.sprites import VOXEL
from kartlab.render.topdown import format_time
from kartlab.sim.sensors import RaySensor
from kartlab.sim.track import Track

LIGHT_MODE_FPS = 50.0

__all__ = [
    "CAMERA_NAMES",
    "VARIANT_NAMES",
    "GameView",
    "HudInfo",
    "Smoothed",
    "TrialHud",
    "ease_out",
    "format_delta",
    "format_time",
    "scale_for",
]


@dataclass
class TrialHud:
    """Time-trial figures for the HUD: the record, the last lap and the live gap to the record."""

    record: float | None = None
    last: float | None = None
    delta: float | None = None  # seconds; negative = ahead of the record


@dataclass
class HudInfo:
    laps: int = 0
    lap_time: float = 0.0
    best_lap: float | None = None
    speed: float = 0.0  # m/s
    label: str = "Conducción manual"
    ramp: int = 0
    notice: str | None = None
    extra: list[str] = field(default_factory=list)
    trial: TrialHud | None = None  # time trial: gap, record and last lap
    lap_panel: bool = True  # False: a race HUD (show/racehud.py) shows the laps and times instead
    label_panel: bool = True  # False: hide the model label and view settings (top right)


def scale_for(window: tuple[int, int]) -> int:
    return max(1, min(window[0] // WORLD_SIZE[0], window[1] // WORLD_SIZE[1]))


class GameView:
    def __init__(
        self,
        track: Track,
        window: tuple[int, int],
        camera: str = "chase",
        variant: str = "day",
        light_mode: bool = False,
        seed: int = 0,
        ramps: tuple[int, ...] = (0,),
    ) -> None:
        self.track = track
        self.cameras = {name: cls() for name, cls in CAMERAS.items()}
        self.camera_name = camera
        self.world = WorldRenderer(track, self.cameras[camera].lens, variant, light_mode, seed)
        self.diagnostics = False
        self.auto_light = True
        self.timings: dict[str, float] = {}  # smoothed, for the overlay
        self.last_timings: dict[str, float] = {}
        self._fps: deque[float] = deque(maxlen=120)
        self._start = time.perf_counter()
        self.fixed_time: float | None = None  # offline rendering: time of the frame (s)
        self._fps_now: float | None = None
        self.hud = Hud(self)
        self.sensor = RaySensor(track, 11, math.pi, 60.0)
        self._ray_hint: int | None = None  # previous centerline index (tracks with crossings)
        self.resize(window)
        # The karts in front of the chase camera, in every view, before the first frame.
        chase = self.cameras["chase"]
        focal = (WORLD_SIZE[0] / 2) / math.tan(chase.lens.fov / 2)
        for ramp in ramps:
            self.world.sprites.warm_up(
                ramp, math.atan2(chase.lens.height, chase.distance), focal * VOXEL / chase.distance
            )

    # -- configuration ------------------------------------------------------------------

    def resize(self, window: tuple[int, int]) -> None:
        self.window = window
        self.scale = scale_for(window)
        w, h = WORLD_SIZE[0] * self.scale, WORLD_SIZE[1] * self.scale
        self.world_rect = pygame.Rect((window[0] - w) // 2, (window[1] - h) // 2, w, h)

    def set_camera(self, name: str) -> None:
        self.camera_name = name
        self.cameras[name].pose = None
        self.world.set_lens(self.cameras[name].lens)

    def cycle_camera(self) -> str:
        names = list(CAMERAS)
        self.set_camera(names[(names.index(self.camera_name) + 1) % len(names)])
        return self.camera_name

    def cycle_variant(self) -> str:
        names = list(VARIANT_NAMES)
        variant = names[(names.index(self.world.variant) + 1) % len(names)]
        self.world.set_variant(variant)
        return variant

    def now(self) -> float:
        """Seconds since the view opened; frame-exact if ``fixed_time`` is set (videos)."""
        if self.fixed_time is not None:
            return self.fixed_time
        return time.perf_counter() - self._start

    def notify(self, text: str) -> None:
        self.hud.notify(text, self.now())

    # -- per frame ------------------------------------------------------------------------

    def update(self, visuals: list[KartVisual], dt: float, focus: int = 0) -> CameraPose:
        self.world.update_effects(visuals, dt)
        return self.cameras[self.camera_name].update(visuals[focus].kart, dt, self.track)

    def set_rays(self, visual: KartVisual | None) -> None:
        if visual is None:
            self.world.show_rays = []
            self._ray_hint = None
            return
        kart = visual.kart
        index = self.track.locate_following(kart.x, kart.y, self._ray_hint).index
        self._ray_hint = index
        self.world.ray_lengths = self.sensor.cast(kart.x, kart.y, kart.heading, index)
        self.world.ray_angles = self.sensor.angles
        self.world.ray_range = self.sensor.max_range
        self.world.show_rays = [(kart.x, kart.y, kart.heading)]
        self.world.ray_ground = visual.ground

    @property
    def canvas(self) -> pygame.Surface:
        """The 320x180 surface the world is drawn on and the interface is painted over."""
        return self.world.world

    def render_canvas(
        self,
        pose: CameraPose,
        visuals: list[KartVisual],
        info: HudInfo,
        dt: float,
        fps: float | None = None,
        hud: bool = True,
        pixel_ui: Callable[[pygame.Surface], None] | None = None,
    ) -> pygame.Surface:
        """Draw the world and the HUD on the canvas, without magnifying it yet. ``pixel_ui``
        (or the caller, afterwards) paints more interface on it; ``present`` then shows it."""
        now = self.now()
        timings = self.world.render(pose, visuals, now)
        canvas = self.world.world
        t0 = time.perf_counter()
        if hud:
            self.hud.draw(canvas, visuals, info, dt, now)
        if pixel_ui is not None:
            pixel_ui(canvas)
        timings["interfaz"] = (time.perf_counter() - t0) * 1000
        self.last_timings = timings
        self._fps_now = fps
        return canvas

    def present(self, screen: pygame.Surface) -> None:
        """Magnify the canvas into the window by the integer factor, centred."""
        canvas = self.world.world
        self.hud.draw_notice(canvas, self.now())  # last: it goes where no panel is
        if self.diagnostics:
            self.hud.draw_diagnostics(canvas, self._fps_now)
        t1 = time.perf_counter()
        if self.world_rect.size != screen.get_size():
            screen.fill((0, 0, 0))
        target = screen.subsurface(self.world_rect)
        if self.scale == 1:
            target.blit(canvas, (0, 0))
        else:
            pygame.transform.scale(canvas, self.world_rect.size, target)
        self.last_timings["escalado"] = (time.perf_counter() - t1) * 1000
        for key, value in self.last_timings.items():  # smoothed for a readable overlay
            self.timings[key] = self.timings.get(key, value) * 0.9 + value * 0.1
        if self._fps_now is not None:
            self._watch_fps(self._fps_now)

    def draw(
        self,
        screen: pygame.Surface,
        pose: CameraPose,
        visuals: list[KartVisual],
        info: HudInfo,
        dt: float,
        fps: float | None = None,
        hud: bool = True,
        pixel_ui: Callable[[pygame.Surface], None] | None = None,
    ) -> None:
        """Draw a whole frame into ``screen``: ``render_canvas`` and ``present``."""
        self.render_canvas(pose, visuals, info, dt, fps, hud, pixel_ui)
        self.present(screen)

    def forget_fps(self) -> None:
        """Drop the frame-rate samples (after a stall that was not the view being slow)."""
        self._fps.clear()

    def _watch_fps(self, fps: float) -> None:
        self._fps.append(fps)
        if (
            self.auto_light
            and not self.world.light_mode
            and len(self._fps) == self._fps.maxlen
            and np.mean(self._fps) < LIGHT_MODE_FPS
        ):
            self.world.set_light_mode(True)
            self.notify("Modo ligero: Sí (30 fps)")

    @property
    def target_fps(self) -> int:
        return 30 if self.world.light_mode else 60
