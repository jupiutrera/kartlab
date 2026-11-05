"""Window composition for the pseudo-3D view.

The 320x180 world is scaled by the largest integer factor that fits (no smoothing,
black bars if needed) and the HUD is drawn on top at window resolution: crisp text,
rendered once per distinct string. Panels slide in with easing, numbers ease towards
their value, and an optional diagnostics line shows the time of every part.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pygame

from kartlab.common.paths import FONTS_DIR
from kartlab.render.camera import CAMERAS, CameraPose
from kartlab.render.palette import KART_RAMPS, Ink, game_palette
from kartlab.render.scene import WORLD_SIZE, KartVisual, WorldRenderer
from kartlab.render.sprites import VOXEL
from kartlab.render.topdown import TextCache, format_time
from kartlab.sim.drift import DriftState
from kartlab.sim.sensors import RaySensor
from kartlab.sim.track import Track

SANS = FONTS_DIR / "ibm-plex-sans" / "IBMPlexSans-SemiBold.ttf"
SANS_REGULAR = FONTS_DIR / "ibm-plex-sans" / "IBMPlexSans-Regular.ttf"
MONO = FONTS_DIR / "ibm-plex-mono" / "IBMPlexMono-SemiBold.ttf"
PANEL = (14, 14, 22, 190)
WHITE = (240, 240, 236)
DIM = (160, 164, 178)
ACCENT = (251, 185, 84)
LIGHT_MODE_FPS = 50.0
CAMERA_NAMES = {"chase": "persecución", "broadcast": "retransmisión", "top": "cenital"}
VARIANT_NAMES = {"day": "día", "sunset": "atardecer", "night": "noche"}


def ease_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1.0 - (1.0 - t) ** 3


class Smoothed:
    """A displayed number that eases towards its target (no jumps on screen)."""

    def __init__(self, rate: float = 10.0) -> None:
        self.value: float | None = None
        self.rate = rate

    def update(self, target: float, dt: float) -> float:
        if self.value is None:
            self.value = target
        self.value += (target - self.value) * (1.0 - math.exp(-self.rate * dt))
        return self.value


def format_delta(delta: float) -> str:
    """Gap to the record as ``−0,42`` (ahead) or ``+1,03`` (behind), two decimals."""
    rounded = round(delta, 2)
    sign = "−" if rounded < 0 else "+"
    return f"{sign}{abs(rounded):.2f}".replace(".", ",")


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
    drift: DriftState | None = None  # manual sweep: shows the charge bar
    trial: TrialHud | None = None  # time trial: gap, record and last lap


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
        pygame.font.init()
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
        self._speed = Smoothed()
        rgb = game_palette().rgb  # gap colours come from the palette like everything else
        self._ahead = tuple(int(c) for c in rgb[Ink.LEAF_LIGHT])
        self._behind = tuple(int(c) for c in rgb[Ink.KERB_RED])
        self._notice: tuple[str, float] | None = None
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
        unit = max(window[1] // 40, 12)
        self.unit = unit
        self.text = TextCache(pygame.font.Font(SANS, unit))
        self.small = TextCache(pygame.font.Font(SANS_REGULAR, max(int(unit * 0.75), 10)))
        self.big = TextCache(pygame.font.Font(MONO, int(unit * 2.0)))
        self.mono = TextCache(pygame.font.Font(MONO, unit))
        self.tiny = TextCache(pygame.font.Font(MONO, max(int(unit * 0.7), 10)))
        self._panels: dict[tuple[int, int], pygame.Surface] = {}
        self._minimap = self._build_minimap(int(window[1] * 0.24))

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

    def notify(self, text: str) -> None:
        self._notice = (text, time.perf_counter())

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
        self.world.show_rays = [(kart.x, kart.y, kart.heading)]

    def draw(
        self,
        screen: pygame.Surface,
        pose: CameraPose,
        visuals: list[KartVisual],
        info: HudInfo,
        dt: float,
        fps: float | None = None,
    ) -> None:
        now = time.perf_counter() - self._start
        timings = self.world.render(pose, visuals, now)
        t0 = time.perf_counter()
        if self.world_rect.size != screen.get_size():
            screen.fill((0, 0, 0))
        target = screen.subsurface(self.world_rect)
        if self.scale == 1:
            target.blit(self.world.world, (0, 0))
        else:
            pygame.transform.scale(self.world.world, self.world_rect.size, target)
        t1 = time.perf_counter()
        self._draw_hud(screen, visuals, info, dt, now)
        t2 = time.perf_counter()
        timings["escalado"] = (t1 - t0) * 1000
        timings["HUD"] = (t2 - t1) * 1000
        self.last_timings = timings
        for key, value in timings.items():  # smoothed for a readable overlay
            self.timings[key] = self.timings.get(key, value) * 0.9 + value * 0.1
        if fps is not None:
            self._watch_fps(fps)
        if self.diagnostics:
            self._draw_diagnostics(screen, fps)

    def _watch_fps(self, fps: float) -> None:
        self._fps.append(fps)
        if (
            self.auto_light
            and not self.world.light_mode
            and len(self._fps) == self._fps.maxlen
            and np.mean(self._fps) < LIGHT_MODE_FPS
        ):
            self.world.set_light_mode(True)
            self.notify("Modo ligero activado (30 fps)")

    @property
    def target_fps(self) -> int:
        return 30 if self.world.light_mode else 60

    # -- HUD ------------------------------------------------------------------------------

    def _panel(self, w: int, h: int) -> pygame.Surface:
        key = (w, h)
        panel = self._panels.get(key)
        if panel is None:
            panel = pygame.Surface(key, pygame.SRCALPHA)
            panel.fill(PANEL)
            self._panels[key] = panel
        return panel

    def _draw_hud(self, screen, visuals, info: HudInfo, dt: float, now: float) -> None:
        u = self.unit
        pad = u // 2
        slide = ease_out(now / 0.7)  # panels slide in when the view opens
        left = self.world_rect.x + pad - round((1 - slide) * 12 * u)
        right = self.world_rect.right - pad + round((1 - slide) * 12 * u)
        top = self.world_rect.y + pad

        # Lap and times (top left).
        lap = self.text.get(f"VUELTA {info.laps + 1}", ACCENT)
        clock = self.big.get(format_time(info.lap_time), WHITE)
        best = self.small.get(f"Mejor  {format_time(info.best_lap)}", DIM)
        w = max(lap.get_width(), clock.get_width(), best.get_width()) + 2 * pad
        h = lap.get_height() + clock.get_height() + best.get_height() + pad
        screen.blit(self._panel(w, h), (left, top))
        y = top + pad // 2
        for surface in (lap, clock, best):
            screen.blit(surface, (left + pad, y))
            y += surface.get_height()
        if info.trial is not None:
            self._draw_trial(screen, info.trial, left, top + h + pad // 2, w)

        # Model label and view settings (top right).
        colour = tuple(int(c) for c in game_palette().rgb[KART_RAMPS[info.ramp % 6][1]])
        label = self.text.get(info.label, WHITE)
        sub = self.small.get(
            f"Cámara {CAMERA_NAMES[self.camera_name]} · {VARIANT_NAMES[self.world.variant]}", DIM
        )
        extra = [self.small.get(text, DIM) for text in info.extra]
        w = max(label.get_width() + u, sub.get_width(), *(e.get_width() for e in extra)) + 2 * pad
        h = label.get_height() + sub.get_height() + sum(e.get_height() for e in extra) + pad
        x = right - w
        screen.blit(self._panel(w, h), (x, top))
        pygame.draw.circle(
            screen, colour, (x + pad + u // 3, top + pad // 2 + label.get_height() // 2), u // 3
        )
        screen.blit(label, (x + pad + u, top + pad // 2))
        y = top + pad // 2 + label.get_height()
        for surface in (sub, *extra):
            screen.blit(surface, (x + pad, y))
            y += surface.get_height()

        # Speed (bottom right), eased so it never flickers.
        kmh = abs(self._speed.update(info.speed * 3.6, dt))
        speed = self.big.get(f"{kmh:3.0f}", WHITE)
        unit_text = self.small.get("km/h", DIM)
        w = speed.get_width() + unit_text.get_width() + 3 * pad
        h = speed.get_height() + pad // 2
        bottom = self.world_rect.bottom - pad
        screen.blit(self._panel(w, h), (right - w, bottom - h))
        screen.blit(speed, (right - w + pad, bottom - h))
        screen.blit(
            unit_text, (right - unit_text.get_width() - pad, bottom - unit_text.get_height() - pad)
        )

        self._draw_minimap(screen, visuals, (left, bottom - self._minimap[0].get_height()))
        if info.drift is not None:
            self._draw_drift(screen, info.drift)
        self._draw_notice(screen)

    def _draw_trial(self, screen, trial: TrialHud, left: int, top: int, width: int) -> None:
        """Gap to the record (green ahead, red behind), then the record and the last lap."""
        pad = self.unit // 2
        if trial.delta is None:
            gap = self.text.get("Sin récord: ¡marca uno!" if trial.record is None else "…", DIM)
        else:
            colour = self._ahead if trial.delta < -0.005 else self._behind
            if abs(trial.delta) < 0.005:
                colour = WHITE
            gap = self.big.get(format_delta(trial.delta), colour)
        lines = [
            self.small.get(f"Récord  {format_time(trial.record)}", ACCENT),
            self.small.get(f"Última  {format_time(trial.last)}", DIM),
        ]
        w = max(width, gap.get_width() + 2 * pad, *(line.get_width() + 2 * pad for line in lines))
        h = gap.get_height() + sum(line.get_height() for line in lines) + pad
        screen.blit(self._panel(w, h), (left, top))
        y = top + pad // 2
        screen.blit(gap, (left + pad, y))
        y += gap.get_height()
        for line in lines:
            screen.blit(line, (left + pad, y))
            y += line.get_height()

    def _draw_drift(self, screen, drift: DriftState) -> None:
        """Linear bar above the bottom edge: the charge while sweeping, the push draining after."""
        if drift.active:
            fraction, ink, name = drift.charge, Ink.YELLOW, "BARRIDO"
        elif drift.pushing:
            fraction, ink, name = drift.push_level, Ink.ORANGE, "IMPULSO"
        else:
            return
        u = self.unit
        rgb = game_palette().rgb
        width, height = u * 14, max(u * 7 // 10, 8)
        label = self.small.get(name, WHITE)
        x = self.world_rect.centerx - width // 2
        y = self.world_rect.bottom - u * 4
        pad = max(u // 4, 3)
        screen.blit(
            self._panel(width + 2 * pad, height + label.get_height() + 2 * pad),
            (x - pad, y - label.get_height() - pad),
        )
        screen.blit(label, (x, y - label.get_height()))
        screen.fill(tuple(int(c) for c in rgb[Ink.BLACK]), (x, y, width, height))
        filled = round((width - 2) * fraction)
        if filled > 0:
            screen.fill(tuple(int(c) for c in rgb[ink]), (x + 1, y + 1, filled, height - 2))
        if drift.active and fraction >= 1.0:  # full: the frame lights up
            pygame.draw.rect(
                screen, tuple(int(c) for c in rgb[Ink.WHITE]), (x, y, width, height), 1
            )

    def _build_minimap(self, size: int):
        pts = self.track.points
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        span = max(hi - lo)
        pad = max(size // 10, 4)
        ppm = (size - 2 * pad) / span
        w = int((hi[0] - lo[0]) * ppm) + 2 * pad
        h = int((hi[1] - lo[1]) * ppm) + 2 * pad
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill(PANEL)

        def to_px(x, y):
            return pad + (x - lo[0]) * ppm, h - pad - (y - lo[1]) * ppm

        line = [to_px(x, y) for x, y in pts[::3]]
        pygame.draw.lines(panel, (90, 90, 110), True, line, max(size // 28, 3))
        pygame.draw.lines(panel, (200, 200, 210), True, line, max(size // 70, 1))
        sx, sy = to_px(*pts[0])
        pygame.draw.circle(panel, WHITE, (sx, sy), max(size // 40, 2))
        return panel, to_px

    def _draw_minimap(self, screen, visuals: list[KartVisual], at: tuple[int, int]) -> None:
        panel, to_px = self._minimap
        screen.blit(panel, at)
        rgb = game_palette().rgb
        radius = max(panel.get_height() // 30, 3)
        for k, v in enumerate(visuals):
            x, y = to_px(v.kart.x, v.kart.y)
            colour = tuple(int(c) for c in rgb[KART_RAMPS[v.ramp % 6][1]])
            r = radius + (1 if k == 0 else 0)
            pygame.draw.circle(screen, (10, 10, 16), (at[0] + x, at[1] + y), r + 1)
            pygame.draw.circle(screen, colour, (at[0] + x, at[1] + y), r)

    def _draw_notice(self, screen) -> None:
        if self._notice is None:
            return
        text, start = self._notice
        age = time.perf_counter() - start
        if age > 3.0:
            self._notice = None
            return
        label = self.text.get(text, ACCENT)
        pad = self.unit // 2
        box = pygame.Surface((label.get_width() + 2 * pad, label.get_height() + pad))
        box.fill((14, 14, 22))
        box.blit(label, (pad, pad // 2))
        alpha = ease_out(age / 0.35) if age < 2.4 else (3.0 - age) / 0.6
        box.set_alpha(round(255 * max(0.0, min(alpha, 1.0))))
        offset = round((1 - ease_out(age / 0.35)) * -box.get_height())
        screen.blit(
            box,
            (self.world_rect.centerx - box.get_width() // 2, self.world_rect.y + pad * 6 + offset),
        )

    def _draw_diagnostics(self, screen, fps: float | None) -> None:
        parts = " · ".join(f"{k} {v:.1f}" for k, v in self.timings.items())
        total = sum(self.timings.values())
        head = f"{fps:.0f} fps · " if fps else ""
        mode = " · ligero" if self.world.light_mode else ""
        text = self.tiny.get(f"{head}{parts} · total {total:.1f} ms{mode}", WHITE)
        pad = self.unit // 3
        x = self.world_rect.centerx - text.get_width() // 2
        y = self.world_rect.bottom - text.get_height() - pad * 2
        screen.blit(
            self._panel(text.get_width() + 2 * pad, text.get_height() + pad),
            (x - pad, y - pad // 2),
        )
        screen.blit(text, (x, y))
