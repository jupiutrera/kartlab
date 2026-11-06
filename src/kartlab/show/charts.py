"""Training curves: reading metrics.csv and drawing the reward chart in pixel art.

The chart lives on the 320x180 grid like the rest of the interface: a flat plate, a 1 px
line, the numbers in the interface font. It returns a small surface that the caller blits
onto the canvas.
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

import numpy as np
import pygame

from kartlab.render import pixelui as ui
from kartlab.show.drivers import format_steps


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


TITLE = "RECOMPENSA MEDIA"
TITLE_SHORT = "RECOMPENSA"


class RewardChart:
    """Mean episode reward against training steps, redrawn only when data changes.

    ``size`` is in canvas pixels. With ``log_x`` the step axis is logarithmic (ticks at 10k,
    100k, 1M), which keeps the early versions of a progression readable; the marker then
    also gets a dot on the curve. The marker (the version on screen) is a vertical line,
    so it never depends on its colour alone.
    """

    def __init__(self, size: tuple[int, int], log_x: bool = False) -> None:
        self.size = size
        self.log_x = log_x
        self.surface = pygame.Surface(size)
        self._key = None

    def draw(self, rows: list[dict[str, float]], data_version: int, marker: int | None):
        key = (data_version, marker, self.size)
        if key == self._key:
            return self.surface
        self._key = key
        surf = self.surface
        width, height = self.size
        ui.plate(surf, (0, 0, width, height), ui.SLATE)
        title = TITLE if ui.measure(TITLE) <= width - 6 else TITLE_SHORT
        ui.text(surf, title, 3, 3, ui.STONE_LIGHT)
        axis = 8 if self.log_x else 0  # room for the step ticks
        box = pygame.Rect(3, 13, width - 6, height - 13 - 3 - axis)
        ui.plate(surf, box, ui.NIGHT, ui.STONE)
        inner = box.inflate(-2, -2)
        points = [(r["steps"], r["reward_mean"]) for r in rows if not math.isnan(r["reward_mean"])]
        if len(points) < 2:
            ui.text(surf, "ESPERANDO DATOS…", inner.x + 3, inner.y + 3, ui.STONE_LIGHT)
            return surf
        xs, ys = np.array(points).T
        if self.log_x:
            x0, x1 = math.log10(max(xs.min() / 2, 1.0)), math.log10(max(xs.max(), 2.0))
        else:
            x0, x1 = 0.0, max(xs.max(), 1.0)
        y0, y1 = min(ys.min(), 0.0), max(ys.max(), 1.0)
        plot = inner.inflate(-4, -4)

        def to_x(steps):
            return math.log10(max(steps, 1.0)) if self.log_x else steps

        def to_px(x, y):
            x = min(max(x, x0), x1)
            px = plot.x + (x - x0) / (x1 - x0) * (plot.w - 1)
            py = plot.bottom - 1 - (y - y0) / (y1 - y0) * (plot.h - 1)
            return round(px), round(py)

        if y0 < 0:  # the zero line
            zero = to_px(x0, 0)[1]
            for x in range(plot.x, plot.right, 2):
                surf.set_at((x, zero), ui.rgb(ui.STONE))
        if self.log_x:
            for tick in (1e4, 1e5, 1e6, 1e7):
                if not x0 <= to_x(tick) <= x1:
                    continue
                px = to_px(to_x(tick), y0)[0]
                ui.vline(surf, px, box.bottom - 3, 2, ui.STONE_LIGHT)
                ui.text(
                    surf,
                    format_steps(int(tick)),
                    px,
                    box.bottom + 2,
                    ui.STONE_LIGHT,
                    align="center",
                )
        curve = [to_px(to_x(x), y) for x, y in points]
        pygame.draw.lines(surf, ui.rgb(ui.YELLOW), False, curve)
        if marker is not None:
            mx = to_px(to_x(marker), y0)[0]
            ui.vline(surf, mx, plot.y, plot.h, ui.ORANGE)
            value = float(np.interp(to_x(marker), [to_x(x) for x in xs], ys))
            my = to_px(to_x(marker), value)[1]
            surf.fill(ui.rgb(ui.NIGHT), (mx - 2, my - 2, 5, 5))
            surf.fill(ui.rgb(ui.ORANGE), (mx - 1, my - 1, 3, 3))
        # The scale, top left: the curve rises to the right.
        ui.text(surf, f"{y1:.0f}", inner.x + 2, inner.y + 2, ui.STONE_LIGHT)
        ui.text(surf, f"{y0:.0f}", inner.right - 2, inner.bottom - 8, ui.STONE_LIGHT, align="right")
        return surf
