"""Training curves: reading metrics.csv and drawing the reward chart."""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

import numpy as np
import pygame

from kartlab.render.topdown import ACCENT, KART_COLOR, TEXT_DIM


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
    """Mean episode reward against training steps, redrawn only when data changes."""

    def __init__(self, size: tuple[int, int], font: pygame.font.Font) -> None:
        self.size = size
        self.font = font
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
        box = pygame.Rect(0, top, width, height - top)
        pygame.draw.rect(surf, (255, 255, 255, 30), box, 1)
        points = [(r["steps"], r["reward_mean"]) for r in rows if not math.isnan(r["reward_mean"])]
        if len(points) < 2:
            msg = self.font.render("Esperando datos…", True, TEXT_DIM)
            surf.blit(msg, (box.x + pad, box.y + pad))
            return surf
        xs, ys = np.array(points).T
        x0, x1 = 0.0, max(xs.max(), 1.0)
        y0, y1 = min(ys.min(), 0.0), max(ys.max(), 1.0)

        def to_px(x, y):
            px = box.x + pad + (x - x0) / (x1 - x0) * (box.width - 2 * pad)
            py = box.bottom - pad - (y - y0) / (y1 - y0) * (box.height - 2 * pad)
            return px, py

        if y0 < 0:
            pygame.draw.line(surf, (255, 255, 255, 40), to_px(x0, 0), to_px(x1, 0))
        pygame.draw.aalines(surf, ACCENT, False, [to_px(x, y) for x, y in points])
        if marker is not None:
            mx = to_px(min(marker, x1), y0)[0]
            pygame.draw.line(surf, KART_COLOR, (mx, box.y + pad), (mx, box.bottom - pad), 2)
        for value, y in ((y1, box.y + pad), (y0, box.bottom - pad - self.font.get_height())):
            label = self.font.render(f"{value:.0f}", True, TEXT_DIM)
            surf.blit(label, (box.right - label.get_width() - pad, y))
        return surf
