"""The title screen: a pixel wordmark over a dimmed, animated 3D scene.

Everything is drawn on the 320x180 grid of the world with palette inks and the interface
font. The wordmark is original: the letters of ``pixelfont`` at scale 4, with a 1 px outline,
an extrusion down and right, and a fill in four bands of ink. A chequered strip scrolls under
it and a highlight sweeps across it from time to time. The dimming over the 3D scene is a
dither pattern, not a blend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygame

from kartlab.render import pixelui as ui
from kartlab.render.palette import Ink
from kartlab.render.pixelfont import CAP_TOP, text_mask

TITLE_TEXT = "KART LAB"
SUBTITLE = "Un kart que aprende a conducir"
WORDMARK_SCALE = 4
SHINE_PERIOD = 5.0  # seconds between highlight sweeps
SHINE_SECONDS = 0.9
BLINK_PERIOD = 1.2
CHIP_H = 11  # a menu chip: plate with a 9-row inside (two for the accents, seven for the capitals)
FILL_BANDS = (  # from the top of the letters down: highlight, yellow, amber, orange
    (0.00, Ink.WHITE),
    (0.12, Ink.YELLOW),
    (0.45, Ink.AMBER),
    (0.75, Ink.ORANGE),
)
_KEY = (255, 0, 255)  # colour key of the wordmark surfaces: not in the palette, never shown


@dataclass(frozen=True)
class MenuEntry:
    key: str
    name: str
    description: str


def ease_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1.0 - (1.0 - t) ** 3


def veil(canvas: pygame.Surface, extra: int = 0) -> None:
    """Darken the 3D scene so the text reads: a dither of the darkest ink, denser at the top
    and the bottom (where the text sits) than in the middle. ``extra`` (0-255, as the screens
    declare it) thickens the middle for screens with a lot to read."""
    w, h = canvas.get_size()
    edge, mid = (3 if extra >= 100 else 2), (2 if extra >= 60 else 1)
    for y0, y1, level in ((0, 40, edge), (40, h - 40, mid), (h - 40, h, edge)):
        ui.dither(canvas, (0, y0, w, y1 - y0), level)


def _ink_array(rows: int, cols: int) -> np.ndarray:
    colours = np.empty((rows, cols, 3), dtype=np.uint8)
    colours[:] = _KEY
    return colours


def _surface(colours: np.ndarray) -> pygame.Surface:
    surface = pygame.surfarray.make_surface(np.ascontiguousarray(colours.transpose(1, 0, 2)))
    surface.set_colorkey(_KEY)
    return surface


def build_wordmark(
    text: str = TITLE_TEXT, scale: int = WORDMARK_SCALE
) -> tuple[pygame.Surface, pygame.Surface]:
    """The wordmark at 1 canvas pixel per pixel, and its white silhouette (for the shine).

    Both are colour-keyed surfaces of the same size."""
    caps = text_mask(text, scale)[CAP_TOP * scale :]
    h, w = caps.shape
    depth = max(2, scale - 1)
    margin = 1
    rows, cols = h + 2 * margin + depth, w + 2 * margin + depth
    grid = np.zeros((rows, cols), dtype=bool)
    grid[margin : margin + h, margin : margin + w] = caps

    def shifted(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
        out = np.zeros_like(mask)
        out[dy:, dx:] = mask[: rows - dy, : cols - dx]
        return out

    def grown(mask: np.ndarray) -> np.ndarray:
        out = mask.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                moved = np.zeros_like(mask)
                ys, xs = slice(max(dy, 0), rows + min(dy, 0)), slice(max(dx, 0), cols + min(dx, 0))
                yo, xo = (
                    slice(max(-dy, 0), rows + min(-dy, 0)),
                    slice(max(-dx, 0), cols + min(-dx, 0)),
                )
                moved[ys, xs] = mask[yo, xo]
                out |= moved
        return out

    colours = _ink_array(rows, cols)
    for step in range(depth, 0, -1):  # the extrusion, darkest at the back
        layer = shifted(grid, step, step)
        colours[grown(layer)] = ui.rgb(ui.NIGHT)
    for step in range(depth, 0, -1):
        layer = shifted(grid, step, step)
        colours[layer] = ui.rgb(Ink.RED if step == 1 else Ink.RED_DARK)
    colours[grown(grid)] = ui.rgb(ui.NIGHT)  # the outline
    fill = np.zeros_like(colours)
    for y in range(rows):
        fraction = max(0.0, (y - margin) / max(h, 1))
        ink = [band for start, band in FILL_BANDS if fraction >= start][-1]
        fill[y, :] = ui.rgb(ink)
    colours[grid] = fill[grid]
    silhouette = _ink_array(rows, cols)
    silhouette[grid] = ui.rgb(ui.WHITE)
    return _surface(colours), _surface(silhouette)


class TitleScreen:
    """State and drawing of the title screen (the 3D scene behind it is the app's)."""

    def __init__(self, entries: list[MenuEntry]) -> None:
        self.entries = entries
        self.selected = 0
        self.opened_at = 0.0
        self._wordmark, self._silhouette = build_wordmark()
        self._glow = pygame.Surface(self._wordmark.get_size())
        self._glow.set_colorkey(_KEY)
        ys, xs = np.mgrid[0 : self._wordmark.get_height(), 0 : self._wordmark.get_width()]
        self._diagonal = xs + ys  # position of every pixel along the sweep
        self._lit = pygame.surfarray.array3d(self._silhouette).transpose(1, 0, 2)
        self._lit = (self._lit == ui.rgb(ui.WHITE)).all(axis=2)

    def open(self, now: float) -> None:
        self.opened_at = now

    def move(self, delta: int) -> None:
        self.selected = (self.selected + delta) % len(self.entries)

    @property
    def entry(self) -> MenuEntry:
        return self.entries[self.selected]

    # -- drawing -----------------------------------------------------------------------------

    def draw(
        self, canvas: pygame.Surface, now: float, run_line: str, versions_line: str, hint: str
    ) -> None:
        age = now - self.opened_at
        width = canvas.get_width()
        veil(canvas)

        # Wordmark: drops in with easing, then bobs one pixel up and down.
        mark = self._wordmark
        drop = round((1.0 - ease_out(age / 0.9)) * -40)
        bob = round(math.sin(now * 1.7))
        x, y = (width - mark.get_width()) // 2, 10 + drop + bob
        canvas.blit(mark, (x, y))
        self._draw_shine(canvas, x, y, now)

        # Chequered strip under the wordmark, scrolling slowly.
        strip_y = y + mark.get_height() + 3
        self._draw_strip(canvas, x, strip_y, mark.get_width(), now)
        cx = width // 2
        if age > 0.5:
            ui.caption(canvas, SUBTITLE, cx, strip_y + 12)

        self._draw_menu(canvas, strip_y + 26, age)

        if age > 0.5:
            base = 140
            if run_line:
                ui.caption(canvas, ui.ellipsize(run_line, 300), cx, base, ui.STONE_LIGHT)
            ui.caption(canvas, ui.ellipsize(versions_line, 300), cx, base + 12, ui.YELLOW)
            # "Pulsa una tecla" blinks in whole steps.
            if int(now * 2 / BLINK_PERIOD * 1.0) % 2 == 0:
                ui.caption(canvas, "PULSA UNA TECLA", cx, 157, ui.YELLOW, ui.NAVY)
            ui.caption(canvas, ui.ellipsize(hint, 300), cx, 171, ui.STONE_LIGHT)

    def _draw_shine(self, canvas: pygame.Surface, x: int, y: int, now: float) -> None:
        phase = (now % SHINE_PERIOD) / SHINE_SECONDS
        if phase > 1.0:
            return
        w, h = self._wordmark.get_size()
        lo = round(-h + phase * (w + 2 * h))
        band = self._lit & (self._diagonal >= lo) & (self._diagonal < lo + max(h // 3, 8))
        if not band.any():
            return
        colours = _ink_array(h, w)
        colours[band] = ui.rgb(ui.WHITE)
        canvas.blit(_surface(colours), (x, y))

    @staticmethod
    def _draw_strip(canvas: pygame.Surface, x: int, y: int, width: int, now: float) -> None:
        cell, rows = 4, 2
        shift = int(now * cell * 1.5) % (cell * 2)
        clip = canvas.get_clip()
        canvas.set_clip(pygame.Rect(x, y, width, cell * rows).clip(clip))
        for row in range(rows):
            for col in range(-2, width // cell + 3):
                ink = ui.TEXT if (col + row) % 2 == 0 else ui.NAVY
                canvas.fill(ui.rgb(ink), (x + col * cell + shift, y + row * cell, cell, cell))
        canvas.set_clip(clip)

    def _draw_menu(self, canvas: pygame.Surface, top: int, age: float) -> None:
        """The modes as a column of chips, the description of the chosen one underneath."""
        if age < 0.7:
            return
        width = canvas.get_width()
        slide = round((1.0 - ease_out((age - 0.7) / 0.5)) * 8)
        chip_w = max(ui.measure(entry.name) for entry in self.entries) + 16
        for i, entry in enumerate(self.entries):
            rect = pygame.Rect(
                (width - chip_w) // 2, top + i * (CHIP_H + 1) + slide, chip_w, CHIP_H
            )
            chosen = i == self.selected
            ui.plate(canvas, rect, ui.YELLOW if chosen else ui.SLATE)
            ui.text(
                canvas,
                entry.name,
                rect.centerx,
                rect.y + 3,
                ui.NIGHT if chosen else ui.TEXT,
                align="center",
            )
        below = top + len(self.entries) * (CHIP_H + 1) + 2 + slide
        for k, line in enumerate(ui.wrap(self.entry.description, 300)[:2]):
            ui.text(
                canvas,
                line,
                width // 2,
                below + k * ui.LINE + 1,
                ui.TEXT,
                outline=ui.NIGHT,
                align="center",
            )
