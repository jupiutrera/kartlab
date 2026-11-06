"""The "brain" panel: what the policy sees and what it decides, for one kart.

It is UI at window resolution, drawn over the 3D view (or the top-down view). Everything
that does not change is painted once into a base surface; each frame blits it and draws the
moving parts with a few ``pygame.draw`` calls and cached texts, so the cost is a fraction
of a millisecond. Nothing is computed here: the data comes from ``PolicyDriver``.

- Sensors: the rays as the network receives them (the observation, not a fresh cast), in a
  radar (radius proportional to the square root of the distance, so the near rays stay
  readable) with the kart at the bottom and the colour of each ray by distance, red when it is
  near and blue when it is far (the same ramp as the rays drawn in the 3D world).
- Action: steering (centred bar), throttle and brake, as held by the driver. The viewers use
  the deterministic policy, so this is the mean of the action distribution, not a sample.
- Value: V(s) at every decision over the last ~10 s.
- Observation: the other numbers the network receives, with the heading as an angle.
"""

from __future__ import annotations

import contextlib
import math

import numpy as np
import pygame

from kartlab.render.palette import RAY_RAMP, game_palette
from kartlab.render.topdown import ACCENT, TextCache
from kartlab.render.view import DIM, MONO, PANEL, SANS, SANS_REGULAR, WHITE
from kartlab.show.drivers import VALUE_SECONDS, PolicyDriver

SLIDE_SECONDS = 0.35
STEER_COLOR = (110, 190, 255)
THROTTLE_COLOR = (96, 214, 128)
BRAKE_COLOR = (255, 96, 80)
TRACK_COLOR = (30, 33, 46)
GUIDE_COLOR = (58, 62, 82)
PANEL_COLOR = (*PANEL[:3], 226)  # denser than the HUD panels: it holds fine detail
RING_METERS = (10.0, 30.0)  # guide circles of the radar besides the range itself
LEVELS = 64  # steps in the distance colour ramps


def decimal(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def lerp_color(a, b, t: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b, strict=True))


def ray_colors() -> list[tuple[int, int, int]]:
    """The ray ramp (palette inks, near to far) sampled at ``LEVELS`` distances."""
    rgb = game_palette().rgb
    stops = [tuple(int(c) for c in rgb[ink]) for ink in RAY_RAMP]
    colors = []
    for i in range(LEVELS):
        position = i / (LEVELS - 1) * (len(stops) - 1)
        k = min(int(position), len(stops) - 2)
        colors.append(lerp_color(stops[k], stops[k + 1], position - k))
    return colors


def ease_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1.0 - (1.0 - t) ** 3


class BrainPanel:
    def __init__(
        self, unit: int, ray_count: int = 11, fov_deg: float = 180.0, ray_range: float = 60.0
    ) -> None:
        pygame.font.init()
        self.unit = u = unit
        self.ray_count = ray_count
        self.ray_range = ray_range
        self.angles = (
            np.linspace(-math.radians(fov_deg) / 2, math.radians(fov_deg) / 2, ray_count)
            if ray_count > 1
            else np.zeros(1)
        )
        self.colors = ray_colors()
        self.wedge_colors = [lerp_color(PANEL_COLOR[:3], c, 0.36) for c in self.colors]
        self.head = TextCache(pygame.font.Font(SANS, max(round(u * 0.95), 11)))
        self.label = TextCache(pygame.font.Font(SANS_REGULAR, max(round(u * 0.68), 9)))
        self.num = TextCache(pygame.font.Font(MONO, max(round(u * 0.68), 9)))
        self.big = TextCache(pygame.font.Font(MONO, max(round(u * 1.0), 11)))
        self.opened_at = -math.inf
        self._lo = 0.0  # chart range, eased so the curve does not jump
        self._hi = 1.0
        self._build()

    # -- static layer -----------------------------------------------------------------------

    def _build(self) -> None:
        u, label = self.unit, self.label
        pad = round(0.7 * u)
        self.pad = pad
        width = round(14.2 * u)
        row = round(1.05 * u)
        label_h = label.get("Ag", DIM).get_height()
        head_h = self.head.get("Ag", WHITE).get_height()

        # Vertical layout, top to bottom (all relative to the panel).
        y = round(0.5 * u)
        self.y_head = y
        y += head_h + round(0.3 * u)
        self.y_sensors = y
        y += label_h + round(0.1 * u)
        radius = (width - 2 * pad) // 2 - round(0.5 * u)
        self.radius = radius
        self.center = (width // 2, y + round(0.2 * u) + radius)
        y = self.center[1] + round(0.1 * u) + label_h
        y += round(0.4 * u)
        self.rows = {}
        for name in ("steer", "throttle", "brake"):
            self.rows[name] = y
            y += row
        y += round(0.35 * u)
        self.y_value = y
        y += max(label_h, self.big.get("0", WHITE).get_height())
        chart_h = round(2.8 * u)
        self.chart = pygame.Rect(pad, y, width - 2 * pad, chart_h)
        y += chart_h + round(0.1 * u)
        self.y_unit = y
        y += label_h * 9 // 10 + round(0.3 * u)
        cell_h = round(1.45 * u)
        gap = round(0.8 * u)
        cell_w = (width - 2 * pad - gap) // 2
        self.cells = []
        for r in range(2):
            for c in range(2):
                x = pad + c * (cell_w + gap)
                self.cells.append(pygame.Rect(x, y + r * cell_h, cell_w, cell_h))
        y += 2 * cell_h + round(0.4 * u)
        self.size = (width, y)

        # Columns of the action bars.
        self.bar_x0 = pad + round(4.4 * u)
        self.bar_x1 = width - pad - round(2.9 * u)
        self.bar_h = max(round(0.5 * u), 4)

        base = pygame.Surface(self.size, pygame.SRCALPHA)
        base.fill(PANEL_COLOR)
        pygame.draw.rect(base, ACCENT, (0, 0, width, max(u // 7, 2)))
        base.blit(self.head.get("CEREBRO", ACCENT), (pad, self.y_head))
        tag = label.get("lo que ve y decide", DIM)
        base.blit(tag, (width - pad - tag.get_width(), self.y_head + head_h - tag.get_height() - 1))
        caption = f"SENSORES · {self.ray_count} rayos · {self.ray_range:.0f} m"
        base.blit(label.get(caption, DIM), (pad, self.y_sensors))
        self._paint_radar(base)
        names = {"steer": "Volante", "throttle": "Acelerador", "brake": "Freno"}
        for name, ry in self.rows.items():
            text = label.get(names[name], DIM)
            base.blit(text, (pad, ry + (row - text.get_height()) // 2))
            track = pygame.Rect(self.bar_x0, ry + (row - self.bar_h) // 2, self._bar_w, self.bar_h)
            pygame.draw.rect(base, TRACK_COLOR, track)
            if name == "steer":  # the centre mark
                pygame.draw.line(
                    base, DIM, (track.centerx, track.y - 2), (track.centerx, track.bottom + 1)
                )
        base.blit(
            label.get(f"VALOR ESTIMADO (V) · {VALUE_SECONDS:.0f} s", DIM), (pad, self.y_value)
        )
        pygame.draw.rect(base, TRACK_COLOR, self.chart)
        names = ("Velocidad", "Rumbo", "Derrape", "Ruedas")
        for cell, name in zip(self.cells, names, strict=False):
            base.blit(label.get(name, DIM), (cell.x, cell.y))
        with contextlib.suppress(pygame.error):  # no window yet: keep the plain surface
            base = base.convert_alpha()  # same format as the window: a faster blit
        self.base = base

    @property
    def _bar_w(self) -> int:
        return self.bar_x1 - self.bar_x0

    def _paint_radar(self, base: pygame.Surface) -> None:
        cx, cy = self.center
        r = self.radius
        half = float(self.angles[-1]) if self.ray_count > 1 else math.pi / 2
        for meters in (*RING_METERS, self.ray_range):
            ring = r * math.sqrt(min(meters / self.ray_range, 1.0))
            arc = [
                (cx - math.sin(a) * ring, cy - math.cos(a) * ring)
                for a in np.linspace(-half, half, 48)
            ]
            pygame.draw.lines(base, GUIDE_COLOR, False, arc)
        for a in self.angles:
            pygame.draw.line(
                base, GUIDE_COLOR, (cx, cy), (cx - math.sin(a) * r, cy - math.cos(a) * r)
            )
        pygame.draw.line(base, GUIDE_COLOR, (cx - r, cy), (cx + r, cy))
        left, right = self.label.get("izq.", DIM), self.label.get("der.", DIM)
        base.blit(left, (cx - r, cy + round(0.1 * self.unit)))
        base.blit(right, (cx + r - right.get_width(), cy + round(0.1 * self.unit)))

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
        margin = self.unit // 2
        slide = ease_out((now - self.opened_at) / SLIDE_SECONDS)
        away = round((1.0 - slide) * (width + margin))
        x = area.x + margin - away if side == "left" else area.right - margin - width + away
        y = max(top, top + (bottom - top - height) // 2)
        return x, y

    # -- per frame --------------------------------------------------------------------------

    def draw(self, screen: pygame.Surface, pos: tuple[int, int], driver: PolicyDriver) -> None:
        ox, oy = pos
        screen.blit(self.base, pos)
        obs = driver.last_obs
        n = self.ray_count
        if obs is None or len(obs) < n + 5:
            return
        self._draw_radar(screen, ox, oy, obs[:n])
        steer, throttle = driver.action
        self._draw_actions(screen, ox, oy, steer, throttle)
        self._draw_value(screen, ox, oy, driver)
        self._draw_observation(screen, ox, oy, obs, driver)

    def _draw_radar(self, screen, ox, oy, rays) -> None:
        cx, cy = self.center
        cx, cy, r = cx + ox, cy + oy, self.radius
        top = LEVELS - 1
        tips = []
        for a, d in zip(self.angles, rays, strict=True):
            d = min(max(float(d), 0.0), 1.0)
            reach = r * math.sqrt(d)  # square-root scale: the near rays stay readable
            tips.append((cx - math.sin(a) * reach, cy - math.cos(a) * reach, d))
        for (x0, y0, d0), (x1, y1, d1) in zip(tips, tips[1:], strict=False):
            color = self.wedge_colors[round((d0 + d1) / 2 * top)]
            pygame.draw.polygon(screen, color, ((cx, cy), (x0, y0), (x1, y1)))
        width = max(round(self.unit * 0.17), 2)
        dot = max(round(self.unit * 0.24), 2)
        for x, y, d in tips:
            color = self.colors[round(d * top)]
            pygame.draw.line(screen, color, (cx, cy), (x, y), width)
            pygame.draw.circle(screen, color, (x, y), dot)
        s = self.unit * 0.42  # the kart, pointing up
        kart = ((cx, cy - s), (cx - s * 0.7, cy + s * 0.55), (cx + s * 0.7, cy + s * 0.55))
        pygame.draw.polygon(screen, WHITE, kart)

    def _draw_actions(self, screen, ox, oy, steer: float, throttle: float) -> None:
        x0, w = ox + self.bar_x0, self._bar_w
        h = self.bar_h
        value_x = ox + self.bar_x1 + round(0.4 * self.unit)
        num = self.num
        pad_y = (round(1.05 * self.unit) - h) // 2
        # Steering: centred, filling towards the side the kart turns to (left is positive).
        y = oy + self.rows["steer"] + pad_y
        half = w / 2
        reach = round(abs(steer) * half)
        centre = x0 + w // 2
        if reach:
            start = centre - reach if steer > 0 else centre
            screen.fill(STEER_COLOR, (start, y, reach, h))
        arrow = "←" if steer > 0.005 else "→" if steer < -0.005 else " "
        screen.blit(num.get(f"{arrow}{decimal(abs(steer))}", WHITE), (value_x, y - 2))
        # Throttle and brake come from the sign of the same command.
        gas, brake = max(throttle, 0.0), max(-throttle, 0.0)
        pedals = (("throttle", gas, THROTTLE_COLOR), ("brake", brake, BRAKE_COLOR))
        for name, value, color in pedals:
            y = oy + self.rows[name] + pad_y
            fill = round(value * w)
            if fill:
                screen.fill(color, (x0, y, fill, h))
            screen.blit(num.get(decimal(value), WHITE), (value_x, y - 2))

    def _draw_value(self, screen, ox, oy, driver: PolicyDriver) -> None:
        box = self.chart.move(ox, oy)
        values = driver.values
        small = self.label
        unit_text = (
            "recompensa futura esperada"
            if driver.version.value_in_reward_units
            else "unidades normalizadas"
        )
        screen.blit(small.get(unit_text, DIM), (ox + self.pad, oy + self.y_unit))
        if driver.version.value is None:
            self._chart_message(screen, box, "sin estimación de valor")
            return
        if len(values) < 2:
            self._chart_message(screen, box, "calculando…")
            return
        data = np.fromiter(values, dtype=np.float64, count=len(values))
        lo, hi = float(data.min()), float(data.max())
        span = max(hi - lo, 0.25 * max(abs(lo), abs(hi), 1.0))
        mid = (hi + lo) / 2
        lo, hi = mid - span * 0.6, mid + span * 0.6
        self._lo += (lo - self._lo) * 0.2
        self._hi += (hi - self._hi) * 0.2
        inner = box.inflate(-4, -6)
        samples = values.maxlen
        xs = inner.right - (len(data) - 1 - np.arange(len(data))) * (inner.width / (samples - 1))
        ys = inner.bottom - (data - self._lo) / (self._hi - self._lo) * inner.height
        ys = np.clip(ys, inner.top, inner.bottom)
        points = np.column_stack((xs, ys)).tolist()
        if self._lo < 0.0 < self._hi:
            zero = inner.bottom - (0.0 - self._lo) / (self._hi - self._lo) * inner.height
            pygame.draw.line(screen, GUIDE_COLOR, (inner.left, zero), (inner.right, zero))
        floor = (points[-1][0], inner.bottom), (points[0][0], inner.bottom)
        pygame.draw.polygon(screen, lerp_color(TRACK_COLOR, ACCENT, 0.22), [*points, *floor])
        pygame.draw.lines(screen, ACCENT, False, points, 2)
        pygame.draw.circle(screen, WHITE, points[-1], max(self.unit // 6, 2))
        text = self.big.get(decimal(float(data[-1]), 1), WHITE)
        screen.blit(text, (box.right - text.get_width(), oy + self.y_value - 2))

    def _chart_message(self, screen, box: pygame.Rect, text: str) -> None:
        surface = self.label.get(text, DIM)
        screen.blit(surface, surface.get_rect(center=box.center))

    def _draw_observation(self, screen, ox, oy, obs, driver: PolicyDriver) -> None:
        n = self.ray_count
        speed = float(obs[n])
        error = math.degrees(math.atan2(float(obs[n + 1]), float(obs[n + 2])))
        extra = len(obs) >= n + 7
        wheel = float(obs[n + 5]) if extra else 0.0
        slip = float(obs[n + 6]) if extra else 0.0
        entries = [
            # (value text, bar fraction, centred)
            (decimal(speed), speed, False),
            (self._angle(error), error / 90.0, True),
            (self._signed(slip), slip, True),
            (self._signed(wheel), wheel, True),
        ]
        if not extra:
            entries = entries[:2]
        bar_h = max(round(self.unit * 0.3), 3)
        line_h = self.label.get("A", DIM).get_height()
        for cell, (text, fraction, centred) in zip(self.cells, entries, strict=False):
            rect = cell.move(ox, oy)
            value = self.num.get(text, WHITE)
            screen.blit(value, (rect.right - value.get_width(), rect.y))
            bar = pygame.Rect(rect.x, rect.y + line_h + 2, rect.width, bar_h)
            screen.fill(TRACK_COLOR, bar)
            fraction = min(max(fraction, -1.0), 1.0)
            if centred:
                reach = round(abs(fraction) * bar.width / 2)
                start = bar.centerx - reach if fraction > 0 else bar.centerx
                screen.fill(STEER_COLOR, (start, bar.y, reach, bar_h))
                screen.fill(DIM, (bar.centerx, bar.y - 1, 1, bar_h + 2))
            else:
                screen.fill(THROTTLE_COLOR, (bar.x, bar.y, round(fraction * bar.width), bar_h))

    @staticmethod
    def _angle(degrees: float) -> str:
        side = " izq." if degrees > 0.5 else " der." if degrees < -0.5 else ""
        return f"{abs(degrees):.0f}°{side}"

    @staticmethod
    def _signed(value: float) -> str:
        arrow = "←" if value > 0.005 else "→" if value < -0.005 else " "
        return f"{arrow}{decimal(abs(value))}"
