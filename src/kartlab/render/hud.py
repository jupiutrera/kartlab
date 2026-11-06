"""The driving HUD, drawn on the 320x180 world grid with the pixel interface kit.

Lap and times, speed, minimap, the view's notices, the model panel and the time-trial panel
are plates of palette inks with the bitmap font, painted on the world surface before it is
magnified (so a pixel of text is a pixel of the world, and night or sunset never tint it).

Layout (canvas pixels): lap panel top left, model panel top right, speed bottom right,
minimap bottom left. Every panel of a frame (these and the ones of the modes) registers its
rectangle with ``reserve``; the notices and the cards are placed afterwards where no panel is
(``free_rect``), so a transient message never covers a fixed one.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pygame

from kartlab.render import pixelui as ui
from kartlab.render.palette import KART_RAMPS
from kartlab.render.scene import KartVisual

if TYPE_CHECKING:
    from kartlab.render.view import GameView, HudInfo, TrialHud

CANVAS_W, CANVAS_H = ui.CANVAS_SIZE
MARGIN = 3
MINIMAP = 40  # longest side of the minimap, in canvas pixels
NOTICE_SECONDS = 3.0
CAMERA_NAMES = {"chase": "Persecución", "broadcast": "Retransm.", "top": "Cenital"}
VARIANT_NAMES = {"day": "día", "sunset": "atardecer", "night": "noche"}
SIDE_PANEL_W = 150  # widest the model panel gets


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


def free_rect(w: int, h: int, candidates, reserved) -> pygame.Rect:
    """The first of the candidate top-left corners where a ``w`` x ``h`` plate fits on the canvas
    without touching a ``reserved`` rectangle, or the one that covers the least."""
    screen = pygame.Rect(0, 0, CANVAS_W, CANVAS_H)
    best, best_area = None, None
    for x, y in candidates:
        rect = pygame.Rect(x, y, w, h).clamp(screen)
        area = sum(r.w * r.h for r in (rect.clip(o) for o in reserved))
        if area == 0:
            return rect
        if best_area is None or area < best_area:
            best, best_area = rect, area
    return best if best is not None else pygame.Rect(0, 0, w, h)


def format_delta(delta: float) -> str:
    """Gap to the record as ``−0,42`` (ahead) or ``+1,03`` (behind), two decimals."""
    rounded = round(delta, 2)
    sign = "−" if rounded < 0 else "+"
    return f"{sign}{abs(rounded):.2f}".replace(".", ",")


def kart_ink(ramp: int) -> int:
    return KART_RAMPS[ramp % len(KART_RAMPS)][1]


class Hud:
    def __init__(self, game: GameView) -> None:
        self.game = game
        self.speed = Smoothed()
        self.notice: tuple[str, float] | None = None
        self.reserved: list[pygame.Rect] = []  # the panels drawn in this frame
        self.notice_rect: pygame.Rect | None = None  # where the notice was last drawn
        self.minimap_size = (0, 0)
        self._minimap: pygame.Surface | None = None
        self._to_px = None
        self.build_minimap()

    # -- state ---------------------------------------------------------------------------

    def begin_frame(self) -> None:
        self.reserved.clear()

    def reserve(self, rect) -> pygame.Rect:
        """Mark an area as taken by a panel in this frame."""
        rect = pygame.Rect(rect)
        self.reserved.append(rect)
        return rect

    def free_rect(self, w: int, h: int, candidates) -> pygame.Rect:
        """The first of the candidate top-left corners where a ``w`` x ``h`` plate fits
        without touching a reserved panel, or the one that covers the least."""
        return free_rect(w, h, candidates, self.reserved)

    def notify(self, text: str, now: float) -> None:
        self.notice = (text, now)

    def build_minimap(self) -> None:
        """The track as a 1 px line in a plate, once per track."""
        pts = self.game.track.points
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        span = float(max(hi - lo))
        pad = 4
        ppm = (MINIMAP - 2 * pad) / span
        w = int((hi[0] - lo[0]) * ppm) + 2 * pad
        h = int((hi[1] - lo[1]) * ppm) + 2 * pad
        panel = pygame.Surface((w, h))
        ui.plate(panel, (0, 0, w, h), ui.SLATE)

        def to_px(x, y):
            return round(pad + (x - lo[0]) * ppm), round(h - pad - (y - lo[1]) * ppm)

        line = [to_px(x, y) for x, y in pts[::3]]
        pygame.draw.lines(panel, ui.rgb(ui.STONE_LIGHT), True, line)
        sx, sy = to_px(*pts[0])
        panel.fill(ui.rgb(ui.WHITE), (sx - 1, sy - 1, 3, 3))
        self._minimap, self._to_px = panel, to_px
        self.minimap_size = panel.get_size()

    # -- frame ---------------------------------------------------------------------------

    def draw(
        self,
        canvas: pygame.Surface,
        visuals: list[KartVisual],
        info: HudInfo,
        dt: float,
        now: float,
    ) -> None:
        self.begin_frame()
        slide = ease_out(now / 0.7)  # panels slide in when the view opens
        away = round((1 - slide) * 90)
        left, right, top = MARGIN - away, CANVAS_W - MARGIN + away, MARGIN
        trial_top, trial_width = top, 0
        if info.lap_panel:
            trial_top, trial_width = self._lap_panel(canvas, info, left, top)
        if info.trial is not None:
            self._trial_panel(canvas, info.trial, left, trial_top, trial_width)
        if info.label_panel:
            self._label_panel(canvas, info, right, top if info.lap_panel else 27)
        # Speed (bottom right), eased so it never flickers.
        kmh = abs(self.speed.update(info.speed * 3.6, dt))
        number = f"{kmh:.0f}"
        digits = ui.measure("888", 2, True)
        w = digits + 6 + ui.measure("KM/H") + 8
        h = 18
        x, y = right - w, CANVAS_H - MARGIN - h
        ui.plate(canvas, self.reserve((x, y, w, h)), ui.SLATE)
        ui.text(canvas, number, x + 4 + digits, y + 2 + 2, ui.WHITE, 2, align="right", tabular=True)
        ui.text(canvas, "KM/H", right - 4, y + 10, ui.STONE_LIGHT, align="right")
        self._draw_minimap(canvas, visuals, (left, CANVAS_H - MARGIN - self.minimap_size[1]))

    def _lap_panel(self, canvas, info: HudInfo, x: int, y: int) -> tuple[int, int]:
        time_w = ui.measure("0:00,00", 2, True)
        w = max(time_w, ui.measure("MEJOR 0:00,00")) + 8
        h = 42
        ui.plate(canvas, self.reserve((x, y, w, h)), ui.SLATE)
        ui.text(canvas, f"VUELTA {info.laps + 1}", x + 4, y + 4, ui.YELLOW)
        ui.text(canvas, ui.fmt_clock(info.lap_time), x + 4, y + 14, ui.WHITE, 2, tabular=True)
        ui.text(canvas, f"MEJOR {ui.fmt_clock(info.best_lap)}", x + 4, y + 31, ui.STONE_LIGHT)
        return y + h + 2, w

    def _trial_panel(self, canvas, trial: TrialHud, x: int, y: int, width: int) -> None:
        """Gap to the record (green ahead, orange behind, with its sign), the record and the
        last lap."""
        lines = [
            (f"RÉCORD {ui.fmt_clock(trial.record)}", ui.YELLOW),
            (f"ÚLTIMA {ui.fmt_clock(trial.last)}", ui.STONE_LIGHT),
        ]
        if trial.delta is None:
            gap, gap_ink, scale = ("SIN RÉCORD" if trial.record is None else "…"), ui.STONE_LIGHT, 1
        else:
            gap = format_delta(trial.delta)
            gap_ink = ui.GREEN if trial.delta < -0.005 else ui.ORANGE
            if abs(trial.delta) < 0.005:
                gap_ink = ui.WHITE
            scale = 2
        # A fixed width: the digits are tabular and the widest time is the template, so the
        # panel never grows with "0:46,22" (and never reaches the lap bar of the race HUD).
        template = max(ui.measure(f"{name} 0:00,00", tabular=True) for name in ("RÉCORD", "ÚLTIMA"))
        w = max(width, ui.measure(gap, scale, True) + 8, template + 8)
        gap_h = 7 * scale
        h = 4 + gap_h + 3 + 2 * 9 + 1
        ui.plate(canvas, self.reserve((x, y, w, h)), ui.SLATE)
        ui.text(canvas, gap, x + 4, y + 4, gap_ink, scale, tabular=scale > 1)
        ty = y + 4 + gap_h + 3
        for text, ink in lines:
            ui.text(canvas, text, x + 4, ty, ink, tabular=True)
            ty += 9

    def _label_panel(self, canvas, info: HudInfo, right: int, top: int) -> None:
        game = self.game
        sub = f"{CAMERA_NAMES[game.camera_name]}, {VARIANT_NAMES[game.world.variant]}"
        inner = SIDE_PANEL_W - 8
        head = ui.ellipsize(info.label, inner - 11)
        extra = [ui.ellipsize(text, inner) for text in info.extra]
        w = max(ui.measure(head) + 11, ui.measure(sub), *(ui.measure(t) for t in extra)) + 8
        h = 4 + 9 * (2 + len(extra)) + 1
        x = right - w
        ui.plate(canvas, self.reserve((x, top, w, h)), ui.SLATE)
        ui.icon(canvas, "dot", x + 4, top + 3, kart_ink(info.ramp), outline=ui.NIGHT)
        ui.text(canvas, head, x + 14, top + 4, ui.TEXT)
        y = top + 13
        for text in (sub, *extra):
            ui.text(canvas, text, x + 4, y, ui.STONE_LIGHT)
            y += 9

    def _draw_minimap(self, canvas, visuals: list[KartVisual], at: tuple[int, int]) -> None:
        canvas.blit(self._minimap, at)
        self.reserve((*at, *self.minimap_size))
        for k, v in enumerate(visuals):
            x, y = self._to_px(v.kart.x, v.kart.y)
            ink = ui.rgb(kart_ink(v.ramp))
            r = 2 if k == 0 else 1
            canvas.fill(
                ui.rgb(ui.NIGHT), (at[0] + x - r - 1, at[1] + y - r - 1, 2 * r + 3, 2 * r + 3)
            )
            canvas.fill(ink, (at[0] + x - r, at[1] + y - r, 2 * r + 1, 2 * r + 1))

    def draw_notice(self, canvas: pygame.Surface, now: float) -> None:
        """The notice: a plate that steps in from the nearest edge, stays, and steps out. It is
        placed where no panel of the frame is (top centre if it fits, else lower)."""
        self.notice_rect = None
        if self.notice is None:
            return
        text, start = self.notice
        age = now - start
        if age > NOTICE_SECONDS:
            self.notice = None
            return
        text = ui.ellipsize(text, CANVAS_W - 40)
        w, h = ui.measure(text) + 10, 13
        spots = [
            ((CANVAS_W - w) // 2 + dx, y)
            for y in (3, 27, 46, 75, 100, 120, 146, 164)
            for dx in (0, -56, 56, -112, 112)
        ]
        rect = self.free_rect(w, h, spots)
        self.notice_rect = rect
        enter = ease_out(age / 0.35)
        leave = ease_out((NOTICE_SECONDS - age) / 0.35)
        away = round((1 - min(enter, leave)) * (h + 4))
        offset = -away if rect.y < CANVAS_H // 2 else away
        ui.plate(canvas, rect.move(0, offset), ui.NAVY)
        ui.text(canvas, text, rect.centerx, rect.y + offset + 3, ui.YELLOW, align="center")

    def draw_diagnostics(self, canvas, fps: float | None) -> None:
        timings = self.game.timings
        parts = "  ".join(f"{k.upper()} {ui.fmt_decimal(v)}" for k, v in timings.items())
        head = f"{fps:.0f} FPS  " if fps else ""
        mode = "  LIGERO" if self.game.world.light_mode else ""
        line = f"{head}{parts}  TOTAL {ui.fmt_decimal(sum(timings.values()))} MS{mode}"
        width = CANVAS_W - 2 * 72  # between the minimap and the speed
        lines = ui.wrap(line, width - 8)
        h = 9 * len(lines) + 3
        y = CANVAS_H - h - MARGIN
        ui.plate(canvas, (72, y, width, h), ui.NIGHT, ui.STONE)
        for k, text in enumerate(lines):
            ui.text(canvas, text, 76, y + 3 + 9 * k, ui.TEXT)
