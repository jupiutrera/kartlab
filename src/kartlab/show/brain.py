"""The "brain" panel: what the policy sees and what it decides, for one kart.

It is pixel interface: drawn on the 320x180 canvas (before the integer magnification, like
the world), with palette inks and the interface font. Everything that does not change is
painted once into a base surface; each frame blits it and draws the moving parts with a few
1 px lines and cached strings, so the cost is a fraction of a millisecond. Nothing is
computed here: the data comes from ``PolicyDriver``.

- Sensors: the rays as the network receives them (the observation, not a fresh cast), in a
  fan (distance on a square-root scale, so the near rays stay readable) with the kart at the
  bottom. Each ray takes the colour of its distance band, warm when near and cool when far,
  and the free space between them is filled.
- Action (VOL, ACE, FRE): steering (centred bar), throttle and brake, as held by the driver.
  The viewers use the deterministic policy, so this is the mean of the action distribution,
  not a sample.
- Value: V(s) at every decision over the last ~10 s.
- Observation: the other numbers the network receives (speed, heading error, slip, wheel).
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from kartlab.render import pixelui as ui
from kartlab.render.palette import RAY_RAMP
from kartlab.show.drivers import VALUE_SECONDS, PolicyDriver

SLIDE_SECONDS = 0.35
RING_METERS = (10.0, 30.0)  # guide arcs of the fan besides the range itself

WIDTH = 100
PAD = 3
FAN_RX = 40  # the fan is a half ellipse: wider than tall, to fit the panel
FAN_RY = 26
ROW = 8  # pitch of the bar rows
LABEL_W = 19  # label column of a bar row
VALUE_W = 29  # value column of a bar row
CHART_H = 11
# The bars are ``BAR_X .. BAR_X + BAR_W`` inside the panel; values are right aligned at the edge.
BAR_X = PAD + LABEL_W
BAR_W = WIDTH - 2 * PAD - LABEL_W - VALUE_W


def decimal(value: float, digits: int = 2) -> str:
    return ui.fmt_decimal(value, digits)


def ray_colors() -> list[tuple[int, int, int]]:
    """Ink of each distance band, near to far: warm to cool, never red (red is for errors).
    The same ramp as the rays in the 3D world, without its first (red) stop."""
    return [ui.rgb(ink) for ink in RAY_RAMP[1:]]


def ease_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1.0 - (1.0 - t) ** 3


def _signed(value: float) -> str:
    arrow = "←" if value > 0.005 else "→" if value < -0.005 else " "
    return f"{arrow}{decimal(abs(value))}"


def _angle(degrees: float) -> str:
    arrow = "←" if degrees > 0.5 else "→" if degrees < -0.5 else " "
    return f"{arrow}{abs(degrees):.0f}°"


class BrainPanel:
    def __init__(
        self, ray_count: int = 11, fov_deg: float = 180.0, ray_range: float = 60.0
    ) -> None:
        self.ray_count = ray_count
        self.ray_range = ray_range
        self.angles = (
            np.linspace(-math.radians(fov_deg) / 2, math.radians(fov_deg) / 2, ray_count)
            if ray_count > 1
            else np.zeros(1)
        )
        self.colors = ray_colors()
        self.opened_at = -math.inf
        self.reserve = None  # set by the viewer: ``hud.reserve``, so notices avoid the panel
        self._lo = 0.0  # chart range, eased so the curve does not jump
        self._hi = 1.0
        self._build()

    # -- static layer -----------------------------------------------------------------------

    def _build(self) -> None:
        # Vertical layout, top to bottom (everything relative to the panel).
        self.y_head = 3
        self.center = (WIDTH // 2, 14 + FAN_RY)
        y = self.center[1] + 3
        self.y_fan_labels = y
        y += 10
        self.rows = {}
        for name in ("steer", "throttle", "brake"):
            self.rows[name] = y
            y += ROW
        y += 2
        self.y_value = y
        y += 9
        self.chart = pygame.Rect(PAD, y, WIDTH - 2 * PAD, CHART_H)
        y += CHART_H + 3
        self.cells = []
        cell_w = (WIDTH - 2 * PAD - 4) // 2
        for r in range(2):
            for c in range(2):
                self.cells.append(pygame.Rect(PAD + c * (cell_w + 4), y + r * ROW, cell_w, ROW))
        y += 2 * ROW + 2
        self.size = (WIDTH, y)

        base = pygame.Surface(self.size)
        ui.plate(base, (0, 0, *self.size), ui.SLATE)
        ui.text(base, "CEREBRO", PAD + 1, self.y_head, ui.YELLOW)
        label = f"{self.ray_count} RAYOS"
        ui.text(base, label, WIDTH - PAD - 1, self.y_head, ui.STONE_LIGHT, align="right")
        ui.hline(base, 1, 12, WIDTH - 2, ui.NIGHT)
        self._paint_fan(base)
        names = {"steer": "VOL", "throttle": "ACE", "brake": "FRE"}
        for name, ry in self.rows.items():
            ui.text(base, names[name], PAD + 1, ry + 1, ui.STONE_LIGHT)
            ui.bar(base, (BAR_X, ry, BAR_W, ROW - 1), 0.0, ui.GREEN, centered=name == "steer")
        ui.text(base, "V(S)", PAD + 1, self.y_value + 1, ui.STONE_LIGHT)
        ui.text(base, f"{VALUE_SECONDS:.0f} S", PAD + 28, self.y_value + 1, ui.STONE)
        ui.plate(base, self.chart, ui.NIGHT, ui.STONE)
        self.base = base

    def _fan_point(self, angle: float, reach: float) -> tuple[int, int]:
        cx, cy = self.center
        return round(cx - math.sin(angle) * FAN_RX * reach), round(
            cy - math.cos(angle) * FAN_RY * reach
        )

    def _paint_fan(self, base: pygame.Surface) -> None:
        cx, cy = self.center
        half = float(self.angles[-1]) if self.ray_count > 1 else math.pi / 2
        for meters in (*RING_METERS, self.ray_range):
            ring = math.sqrt(min(meters / self.ray_range, 1.0))
            for a in np.linspace(-half, half, 40):
                x, y = self._fan_point(float(a), ring)
                base.set_at((x, y), ui.rgb(ui.STONE))
        ui.hline(base, cx - FAN_RX, cy, 2 * FAN_RX + 1, ui.STONE)
        ui.text(base, "←", cx - FAN_RX, cy + 3, ui.STONE_LIGHT)
        ui.text(base, "→", cx + FAN_RX, cy + 3, ui.STONE_LIGHT, align="right")
        ui.text(base, f"{self.ray_range:.0f} M", cx, cy + 3, ui.STONE_LIGHT, align="center")

    # -- placement --------------------------------------------------------------------------

    def open(self, now: float) -> None:
        """The panel was just switched on: it slides in."""
        self.opened_at = now

    def position(
        self, area: pygame.Rect, side: str, top: int, bottom: int, now: float
    ) -> tuple[int, int]:
        """Top-left corner: against the ``side`` edge of ``area``, centred between ``top`` and
        ``bottom`` (or pinned to ``top`` if it does not fit), sliding in from the edge."""
        width, height = self.size
        margin = 3
        slide = ease_out((now - self.opened_at) / SLIDE_SECONDS)
        away = round((1.0 - slide) * (width + margin))
        x = area.x + margin - away if side == "left" else area.right - margin - width + away
        y = max(top, top + (bottom - top - height) // 2)
        return x, y

    # -- per frame --------------------------------------------------------------------------

    def draw(self, target: pygame.Surface, pos: tuple[int, int], driver: PolicyDriver) -> None:
        ox, oy = pos
        if self.reserve is not None:
            self.reserve((ox, oy, *self.size))  # the notices go elsewhere
        target.blit(self.base, pos)
        obs = driver.last_obs
        n = self.ray_count
        if obs is None or len(obs) < n + 5:
            return
        self._draw_fan(target, ox, oy, obs[:n])
        steer, throttle = driver.action
        self._draw_actions(target, ox, oy, steer, throttle)
        self._draw_value(target, ox, oy, driver)
        self._draw_observation(target, ox, oy, obs)

    def _draw_fan(self, target, ox, oy, rays) -> None:
        cx, cy = self.center
        cx, cy = cx + ox, cy + oy
        top = len(self.colors) - 1
        tips = []
        for a, d in zip(self.angles, rays, strict=True):
            d = min(max(float(d), 0.0), 1.0)
            x, y = self._fan_point(float(a), math.sqrt(d))  # near rays stay readable
            tips.append((x + ox, y + oy, d))
        free = ui.rgb(ui.NAVY_LIGHT)
        for (x0, y0, _), (x1, y1, _) in zip(tips, tips[1:], strict=False):
            pygame.draw.polygon(target, free, ((cx, cy), (x0, y0), (x1, y1)))
        outline = [(x, y) for x, y, _ in tips]
        if len(outline) > 1:
            pygame.draw.lines(target, ui.rgb(ui.STONE_LIGHT), False, outline)
        for x, y, d in tips:
            colour = self.colors[round(d * top)]
            pygame.draw.line(target, colour, (cx, cy), (x, y))
            target.fill(colour, (x - 1, y - 1, 3, 3))
        kart = ((cx, cy - 5), (cx - 3, cy + 1), (cx + 3, cy + 1))  # the kart, pointing up
        pygame.draw.polygon(target, ui.rgb(ui.WHITE), kart)
        pygame.draw.lines(target, ui.rgb(ui.NIGHT), True, kart)

    def _draw_actions(self, target, ox, oy, steer: float, throttle: float) -> None:
        right = ox + WIDTH - PAD - 1
        gas, brake = max(throttle, 0.0), max(-throttle, 0.0)
        entries = (
            ("steer", steer, ui.BLUE, True, _signed(steer)),
            ("throttle", gas, ui.GREEN, False, decimal(gas)),
            ("brake", brake, ui.ORANGE, False, decimal(brake)),
        )
        for name, value, ink, centred, shown in entries:
            ry = oy + self.rows[name]
            ui.bar(target, (ox + BAR_X, ry, BAR_W, ROW - 1), value, ink, centered=centred)
            ui.text(target, shown, right, ry + 1, ui.TEXT, align="right")

    def _draw_value(self, target, ox, oy, driver: PolicyDriver) -> None:
        box = self.chart.move(ox, oy)
        values = driver.values
        if driver.version.value is None:
            self._chart_message(target, box, "SIN ESTIMACIÓN")
            return
        if len(values) < 2:
            self._chart_message(target, box, "CALCULANDO…")
            return
        data = np.fromiter(values, dtype=np.float64, count=len(values))
        lo, hi = float(data.min()), float(data.max())
        span = max(hi - lo, 0.25 * max(abs(lo), abs(hi), 1.0))
        mid = (hi + lo) / 2
        lo, hi = mid - span * 0.6, mid + span * 0.6
        self._lo += (lo - self._lo) * 0.2
        self._hi += (hi - self._hi) * 0.2
        inner = box.inflate(-4, -4)
        samples = values.maxlen
        xs = (
            inner.right
            - 1
            - (len(data) - 1 - np.arange(len(data))) * ((inner.w - 1) / (samples - 1))
        )
        ys = inner.bottom - 1 - (data - self._lo) / (self._hi - self._lo) * (inner.h - 1)
        ys = np.clip(ys, inner.top, inner.bottom - 1)
        points = np.column_stack((np.round(xs), np.round(ys))).astype(int).tolist()
        if self._lo < 0.0 < self._hi:
            zero = round(
                inner.bottom - 1 - (0.0 - self._lo) / (self._hi - self._lo) * (inner.h - 1)
            )
            for x in range(inner.left, inner.right, 2):
                target.set_at((x, zero), ui.rgb(ui.STONE))
        floor = [(points[-1][0], inner.bottom - 1), (points[0][0], inner.bottom - 1)]
        pygame.draw.polygon(target, ui.rgb(ui.NAVY), [*points, *floor])
        pygame.draw.lines(target, ui.rgb(ui.YELLOW), False, points)
        x, y = points[-1]
        target.fill(ui.rgb(ui.WHITE), (x - 1, y - 1, 3, 3))
        ui.text(
            target,
            decimal(float(data[-1]), 1),
            ox + WIDTH - PAD - 1,
            oy + self.y_value + 1,
            ui.WHITE,
            align="right",
        )

    @staticmethod
    def _chart_message(target, box: pygame.Rect, message: str) -> None:
        ui.text(target, message, box.centerx, box.y + 3, ui.STONE_LIGHT, align="center")

    def _draw_observation(self, target, ox, oy, obs) -> None:
        n = self.ray_count
        speed = float(obs[n])
        error = math.degrees(math.atan2(float(obs[n + 1]), float(obs[n + 2])))
        extra = len(obs) >= n + 7
        wheel = float(obs[n + 5]) if extra else 0.0
        slip = float(obs[n + 6]) if extra else 0.0
        entries = [
            ("VEL", decimal(speed)),
            ("RUM", _angle(error)),
            ("LAT", _signed(slip)),
            ("RUE", _signed(wheel)),
        ]
        if not extra:
            entries = entries[:2]
        for cell, (name, shown) in zip(self.cells, entries, strict=False):
            rect = cell.move(ox, oy)
            ui.text(target, name, rect.x + 1, rect.y + 1, ui.STONE_LIGHT)
            ui.text(target, shown, rect.right - 2, rect.y + 1, ui.TEXT, align="right")
