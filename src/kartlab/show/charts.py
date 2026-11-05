"""Training curves: reading metrics.csv and drawing the reward chart."""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

import numpy as np
import pygame

from kartlab.render.topdown import ACCENT, KART_COLOR, TEXT_DIM
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


class RewardChart:
    """Mean episode reward against training steps, redrawn only when data changes.

    With ``log_x`` the step axis is logarithmic (ticks at 10k, 100k, 1M), which keeps the
    early versions of a progression video readable; the marker then also gets a dot on
    the curve.
    """

    def __init__(self, size: tuple[int, int], font: pygame.font.Font, log_x: bool = False) -> None:
        self.size = size
        self.font = font
        self.log_x = log_x
        self.surface = pygame.Surface(size, pygame.SRCALPHA)
        self._key = None

    def draw(self, rows: list[dict[str, float]], data_version: int, marker: int | None):
        key = (data_version, marker, self.size)
        if key == self._key:
            return self.surface
        self._key = key
        surf = self.surface
        surf.fill((0, 0, 0, 0))
        width, height = self.size
        pad = 6
        title = self.font.render("Recompensa media por episodio", True, TEXT_DIM)
        surf.blit(title, (0, 0))
        top = title.get_height() + pad
        axis = self.font.get_height() if self.log_x else 0  # room for the step ticks
        box = pygame.Rect(0, top, width, height - top - axis)
        pygame.draw.rect(surf, (255, 255, 255, 30), box, 1)
        points = [(r["steps"], r["reward_mean"]) for r in rows if not math.isnan(r["reward_mean"])]
        if len(points) < 2:
            msg = self.font.render("Esperando datos…", True, TEXT_DIM)
            surf.blit(msg, (box.x + pad, box.y + pad))
            return surf
        xs, ys = np.array(points).T
        if self.log_x:
            x0, x1 = math.log10(max(xs.min() / 2, 1.0)), math.log10(max(xs.max(), 2.0))
        else:
            x0, x1 = 0.0, max(xs.max(), 1.0)
        y0, y1 = min(ys.min(), 0.0), max(ys.max(), 1.0)

        def to_x(steps):
            return math.log10(max(steps, 1.0)) if self.log_x else steps

        def to_px(x, y):
            x = min(max(x, x0), x1)
            px = box.x + pad + (x - x0) / (x1 - x0) * (box.width - 2 * pad)
            py = box.bottom - pad - (y - y0) / (y1 - y0) * (box.height - 2 * pad)
            return px, py

        if y0 < 0:
            pygame.draw.line(surf, (255, 255, 255, 40), to_px(x0, 0), to_px(x1, 0))
        curve = [to_px(to_x(x), y) for x, y in points]
        pygame.draw.aalines(surf, ACCENT, False, curve)
        if marker is not None:
            mx = to_px(to_x(marker), y0)[0]
            pygame.draw.line(surf, KART_COLOR, (mx, box.y + pad), (mx, box.bottom - pad), 2)
            if self.log_x:
                value = float(np.interp(to_x(marker), [to_x(x) for x in xs], ys))
                pygame.draw.circle(surf, KART_COLOR, to_px(to_x(marker), value), 5)
        for value, y in ((y1, box.y + pad), (y0, box.bottom - pad - self.font.get_height())):
            label = self.font.render(f"{value:.0f}", True, TEXT_DIM)
            surf.blit(label, (box.x + pad + 2, y))  # top left: the curve rises to the right
        if self.log_x:
            for tick in (1e4, 1e5, 1e6, 1e7):
                if to_x(tick) < x0 or to_x(tick) > x1:
                    continue
                label = self.font.render(format_steps(int(tick)), True, TEXT_DIM)
                px = to_px(to_x(tick), y0)[0]
                pygame.draw.line(surf, (255, 255, 255, 40), (px, box.y), (px, box.bottom))
                surf.blit(label, (px - label.get_width() / 2, box.bottom + 2))
        return surf
