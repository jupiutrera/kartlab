"""Top-down debug view of the 2D simulation, drawn with pygame-ce software surfaces.

The track is drawn once per scale onto a surface (``TrackMap``); each frame only blits
it and draws karts, trails and the HUD on top.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import pygame

from kartlab.common.paths import FONTS_DIR
from kartlab.sim.kart import KartState
from kartlab.sim.race import WORLD_MARGIN
from kartlab.sim.track import Track

GRASS = (52, 101, 58)
GRASS_GRID = (60, 112, 66)
ASPHALT = (76, 80, 90)
EDGE = (226, 228, 220)
KERB_RED = (204, 62, 62)
KERB_WHITE = (238, 238, 232)
CENTER_MARK = (104, 109, 120)
CHECKER_DARK = (24, 24, 28)
KART_COLOR = (255, 156, 44)
KART_OUTLINE = (24, 24, 30)
TRAIL_COLOR = (255, 214, 150)
PANEL = (14, 16, 22, 200)
TEXT = (236, 238, 242)
TEXT_DIM = (150, 157, 170)
ACCENT = (255, 186, 72)
WARNING = (255, 118, 96)

FOLLOW_VIEW_HEIGHT = 90.0  # metres visible vertically with the follow camera
GRID_STEP = 10.0  # grass grid spacing, so speed is visible even on the grass (m)
KERB_CURVATURE = 1 / 45  # corners tighter than this radius get red and white kerbs
MAX_MAP_PIXELS = 12_000_000  # cap for the pre-drawn map surface
MIN_KART_PIXELS = 18.0  # karts are never drawn shorter than this

FONT_REGULAR = FONTS_DIR / "ibm-plex-mono" / "IBMPlexMono-Regular.ttf"
FONT_BOLD = FONTS_DIR / "ibm-plex-mono" / "IBMPlexMono-SemiBold.ttf"


class TrackMap:
    """The track pre-drawn onto a surface at ``ppm`` pixels per metre."""

    def __init__(self, track: Track, ppm: float, margin: float) -> None:
        xmin, ymin, xmax, ymax = track.bounds
        span_x, span_y = xmax - xmin + 2 * margin, ymax - ymin + 2 * margin
        ppm = min(ppm, math.sqrt(MAX_MAP_PIXELS / (span_x * span_y)))
        self.ppm = ppm
        self.origin = (xmin - margin, ymax + margin)  # world position of the top-left pixel
        self.surface = pygame.Surface((math.ceil(span_x * ppm), math.ceil(span_y * ppm)))
        self._draw(track)
        if pygame.display.get_surface() is not None:
            self.surface = self.surface.convert()

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.origin[0]) * self.ppm, (self.origin[1] - y) * self.ppm

    def to_px_array(self, points: np.ndarray) -> np.ndarray:
        out = np.empty_like(points)
        out[:, 0] = (points[:, 0] - self.origin[0]) * self.ppm
        out[:, 1] = (self.origin[1] - points[:, 1]) * self.ppm
        return out

    def _draw(self, track: Track) -> None:
        surf, ppm = self.surface, self.ppm
        surf.fill(GRASS)
        width, height = surf.get_size()
        px = (math.ceil(self.origin[0] / GRID_STEP) * GRID_STEP - self.origin[0]) * ppm
        while px < width:
            pygame.draw.line(surf, GRASS_GRID, (px, 0), (px, height))
            px += GRID_STEP * ppm
        py = (self.origin[1] - math.floor(self.origin[1] / GRID_STEP) * GRID_STEP) * ppm
        while py < height:
            pygame.draw.line(surf, GRASS_GRID, (0, py), (width, py))
            py += GRID_STEP * ppm

        # Asphalt as a union of discs along the centerline: no seams between segments.
        radius = max(1, round(track.half_width * ppm))
        for px, py in self.to_px_array(track.points).tolist():
            pygame.draw.circle(surf, ASPHALT, (px, py), radius)

        left = self.to_px_array(track.left_edge).tolist()
        right = self.to_px_array(track.right_edge).tolist()
        edge_px = max(1, round(0.3 * ppm))
        pygame.draw.lines(surf, EDGE, True, left, edge_px)
        pygame.draw.lines(surf, EDGE, True, right, edge_px)
        kerb_px = max(2, round(0.9 * ppm))
        kerb = np.abs(track.curvature) > KERB_CURVATURE
        n = len(track)
        for i in np.flatnonzero(kerb).tolist():
            color = KERB_RED if (i // 2) % 2 else KERB_WHITE
            j = (i + 1) % n
            pygame.draw.line(surf, color, left[i], left[j], kerb_px)
            pygame.draw.line(surf, color, right[i], right[j], kerb_px)

        centre = self.to_px_array(track.points).tolist()
        dash = max(1, round(0.15 * ppm))
        for i in range(0, n, 6):
            pygame.draw.line(surf, CENTER_MARK, centre[i], centre[(i + 3) % n], dash)
        self._draw_arrows(track)
        self._draw_start_line(track)

    def _draw_arrows(self, track: Track, every: float = 80.0) -> None:
        size = 2.2
        for s in np.arange(every / 2, track.length, every):
            x, y, h = track.pose_at(float(s))
            c, sn = math.cos(h), math.sin(h)
            tip = (x + c * size, y + sn * size)
            back_l = (x - c * size - sn * size, y - sn * size + c * size)
            back_r = (x - c * size + sn * size, y - sn * size - c * size)
            pts = [self.to_px(*p) for p in (back_l, tip, back_r)]
            pygame.draw.lines(self.surface, CENTER_MARK, False, pts, max(1, round(0.4 * self.ppm)))

    def _draw_start_line(self, track: Track) -> None:
        x0, y0, h = track.pose_at(0.0)
        tx, ty = math.cos(h), math.sin(h)
        nx, ny = -ty, tx
        cols = max(2, round(track.width / 1.0))
        square = track.width / cols
        for row in range(2):
            for col in range(cols):
                color = CHECKER_DARK if (row + col) % 2 else KERB_WHITE
                s0, s1 = (row - 1) * square, row * square
                l0 = -track.half_width + col * square
                l1 = l0 + square
                corners = [
                    (x0 + tx * s + nx * lat, y0 + ty * s + ny * lat)
                    for s, lat in ((s0, l0), (s1, l0), (s1, l1), (s0, l1))
                ]
                pygame.draw.polygon(self.surface, color, [self.to_px(*p) for p in corners])


class TopDownView:
    """Draws the track, trails and karts with a follow or whole-track camera."""

    def __init__(self, track: Track, size: tuple[int, int]) -> None:
        self.track = track
        self.resize(size)

    def resize(self, size: tuple[int, int]) -> None:
        self.size = size
        width, height = size
        self.follow_map = TrackMap(self.track, height / FOLLOW_VIEW_HEIGHT, WORLD_MARGIN)
        xmin, ymin, xmax, ymax = self.track.bounds
        margin = 10.0
        fit_ppm = min(
            width * 0.92 / (xmax - xmin + 2 * margin), height * 0.84 / (ymax - ymin + 2 * margin)
        )
        self.fit_map = TrackMap(self.track, fit_ppm, margin)

    def draw(
        self,
        target: pygame.Surface,
        karts: Iterable[tuple[KartState, tuple[int, int, int]]],
        camera: tuple[float, float],
        follow: bool,
        trail: Sequence[tuple[float, float]] = (),
    ) -> None:
        width, height = self.size
        tmap = self.follow_map if follow else self.fit_map
        if follow:
            cx, cy = tmap.to_px(*camera)
            ox, oy = width / 2 - cx, height / 2 - cy
        else:
            map_w, map_h = tmap.surface.get_size()
            ox, oy = (width - map_w) / 2, (height - map_h) / 2
        target.fill(GRASS)
        target.blit(tmap.surface, (round(ox), round(oy)))

        if len(trail) > 1:
            pts = [(px + ox, py + oy) for px, py in (tmap.to_px(x, y) for x, y in trail)]
            pygame.draw.aalines(target, TRAIL_COLOR, False, pts)
        for kart, color in karts:
            self._draw_kart(target, tmap, kart, color, (ox, oy))

    @staticmethod
    def _draw_kart(target, tmap: TrackMap, kart: KartState, color, offset) -> None:
        length = max(1.8 * tmap.ppm, MIN_KART_PIXELS)
        width = length * 0.62
        px, py = tmap.to_px(kart.x, kart.y)
        px, py = px + offset[0], py + offset[1]
        # Screen y grows downwards, so the heading is mirrored.
        c, s = math.cos(kart.heading), -math.sin(kart.heading)
        shape = ((0.5, 0.0), (-0.5, 0.5), (-0.28, 0.0), (-0.5, -0.5))
        pts = [
            (px + c * u * length - s * v * width, py + s * u * length + c * v * width)
            for u, v in shape
        ]
        pygame.draw.polygon(target, color, pts)
        pygame.draw.polygon(target, KART_OUTLINE, pts, 2)


class TextCache:
    """Renders each distinct text once; the HUD only re-renders strings that change."""

    def __init__(self, font: pygame.font.Font, max_items: int = 256) -> None:
        self.font = font
        self.max_items = max_items
        self._cache: dict[tuple[str, tuple[int, ...]], pygame.Surface] = {}

    def get(self, text: str, color: tuple[int, ...]) -> pygame.Surface:
        key = (text, color)
        surface = self._cache.get(key)
        if surface is None:
            if len(self._cache) >= self.max_items:
                self._cache.clear()
            surface = self.font.render(text, True, color)
            self._cache[key] = surface
        return surface


def load_fonts(height: int) -> tuple[pygame.font.Font, pygame.font.Font]:
    """Regular and bold HUD fonts sized for a window ``height`` pixels tall."""
    pygame.font.init()
    size = max(12, height // 38)
    return pygame.font.Font(FONT_REGULAR, size), pygame.font.Font(FONT_BOLD, round(size * 1.15))


def format_time(seconds: float | None) -> str:
    if seconds is None:
        return "-:--.--"
    minutes, rest = divmod(seconds, 60.0)
    return f"{int(minutes)}:{rest:05.2f}"


class DebugHud:
    """Text panels for the debug view: lap data, speed, mode, fps and key help."""

    def __init__(self, size: tuple[int, int]) -> None:
        self.resize(size)

    def resize(self, size: tuple[int, int]) -> None:
        self.size = size
        regular, bold = load_fonts(size[1])
        self.text = TextCache(regular)
        self.title = TextCache(bold)
        self.line_height = regular.get_linesize()
        self.pad = max(6, size[1] // 90)
        self._panels: dict[tuple[int, int], pygame.Surface] = {}

    def _panel(self, width: int, height: int) -> pygame.Surface:
        key = (width, height)
        panel = self._panels.get(key)
        if panel is None:
            panel = pygame.Surface(key, pygame.SRCALPHA)
            panel.fill(PANEL)
            self._panels[key] = panel
        return panel

    def draw_block(
        self,
        target: pygame.Surface,
        lines: Sequence[tuple[str, tuple[int, int, int]]],
        anchor: str,
        title: str | None = None,
    ) -> None:
        """Draw text lines on a translucent panel anchored to a screen corner."""
        surfaces = [self.title.get(title, ACCENT)] if title else []
        surfaces += [self.text.get(text, color) for text, color in lines]
        width = max(s.get_width() for s in surfaces) + 2 * self.pad
        width = (width + 31) // 32 * 32  # stable panel width while digits change
        height = sum(s.get_height() for s in surfaces) + 2 * self.pad
        screen_w, screen_h = self.size
        x = self.pad if "left" in anchor else screen_w - width - self.pad
        y = self.pad if "top" in anchor else screen_h - height - self.pad
        target.blit(self._panel(width, height), (x, y))
        cy = y + self.pad
        for surface in surfaces:
            target.blit(surface, (x + self.pad, cy))
            cy += surface.get_height()
