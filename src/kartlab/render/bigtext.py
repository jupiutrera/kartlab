"""Big pixel-art lettering: the place, the 3-2-1 countdown, the banners.

It is the interface font (``pixelfont``) drawn with a bigger cell: every lit pixel of the 5x7
glyph becomes a ``cell`` x ``cell`` block, so the big digits are the same letters as the small
ones, never scaled with smoothing. Strokes are two cells thick (bold), the letters have a dark
outline and a drop shadow one cell wide, and the fill is shaded in three bands of palette inks
(light on top, saturated in the middle, deep at the bottom), like the lettering of 16-bit games.
Each text is rendered once per (text, cell, look) and cached.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from kartlab.render import pixelui as ui
from kartlab.render.palette import Ink
from kartlab.render.pixelfont import CAP_TOP, text_mask

# Three bands of ink, top to bottom.
GOLD = (28, Ink.YELLOW, Ink.AMBER)
SILVER = (Ink.WHITE, Ink.MIST, Ink.STONE_LIGHT)
BRONZE = (23, 22, 21)
COOL = (48, 47, 46)
EMBER = (62, 22, Ink.ORANGE)
AMBER = (23, Ink.AMBER, Ink.ORANGE)
GREEN = (Ink.LEAF_LIGHT, 31, 30)
PAPER = (Ink.WHITE, Ink.MIST, Ink.STONE_LIGHT)
Look = tuple[int, int, int]

# Words (the countdown, the banners) keep the strokes of the font, one cell thick: bold strokes
# with an outline close the inner gaps of the A, the Y and the exclamation marks. A single
# digit, the place, is bold.
SIGN_BOLD = False
POP_SECONDS = 0.3
POP_STEPS = (3, 2, 1, 0)  # extra cells added to the cell size, one step every POP_SECONDS / 4

_cache: dict[tuple, pygame.Surface] = {}


def place_look(place: int) -> Look:
    """Look of a finishing place: gold, silver, bronze, then cool blues."""
    return (GOLD, SILVER, BRONZE)[place - 1] if 1 <= place <= 3 else COOL


def pop_extra(age: float) -> int:
    """Extra cells of a "pop" that is ``age`` seconds old: a jump up, then back in steps."""
    if age < 0.0 or age >= POP_SECONDS:
        return 0
    return POP_STEPS[min(int(age / (POP_SECONDS / len(POP_STEPS))), len(POP_STEPS) - 1)]


def _grid(text: str, bold: bool) -> np.ndarray:
    """The text as a cell grid with the empty accent rows cut away (and the accents kept if any)."""
    mask = text_mask(text, 1)
    rows = np.flatnonzero(mask.any(axis=1))
    mask = mask[rows[0] : CAP_TOP + 7] if len(rows) else mask[CAP_TOP : CAP_TOP + 7]
    if bold:
        thick = mask.copy()
        thick[:, 1:] |= mask[:, :-1]
        mask = thick
    return mask


def _grow(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1)
    h, w = mask.shape
    out = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy : dy + h, dx : dx + w]
    return out


def render(text: str, cell: int, look: Look, bold: bool = True) -> pygame.Surface:
    """``text`` as a surface (colour key transparent): fill, outline and shadow in whole cells."""
    key = (text, cell, look, bold)
    surface = _cache.get(key)
    if surface is not None:
        return surface
    grid = np.pad(_grid(text, bold), 1)
    body = _grow(grid)
    shadow = np.zeros_like(grid)
    shadow[1:, 1:] = body[:-1, :-1]
    rows, cols = grid.shape
    surface = pygame.Surface(((cols + 1) * cell, (rows + 1) * cell))
    surface.fill(ui._KEY)
    surface.set_colorkey(ui._KEY)

    def paint(mask: np.ndarray, ink: int) -> None:
        for y, x in zip(*np.nonzero(mask), strict=True):
            surface.fill(ui.rgb(ink), (int(x) * cell, int(y) * cell, cell, cell))

    paint(shadow & ~body, ui.NIGHT)
    paint(body & ~grid, ui.NIGHT)
    lit = np.flatnonzero(grid.any(axis=1))
    top, height = (lit[0], len(lit)) if len(lit) else (0, 1)
    for band in range(3):
        start = top + band * height // 3
        stop = top + (band + 1) * height // 3 if band < 2 else top + height
        band_mask = np.zeros_like(grid)
        band_mask[start:stop] = grid[start:stop]
        paint(band_mask, look[band])
    if len(_cache) >= 160:
        _cache.clear()
    _cache[key] = surface
    return surface


def draw(
    canvas: pygame.Surface,
    text: str,
    cell: int,
    look: Look,
    *,
    center: tuple[int, int] | None = None,
    topright: tuple[int, int] | None = None,
    bold: bool = True,
) -> pygame.Rect:
    surface = render(text, max(cell, 1), look, bold)
    rect = surface.get_rect()
    if center is not None:
        rect.center = center
    if topright is not None:
        rect.topright = topright
    canvas.blit(surface, rect)
    return rect


def fit_cell(text: str, cell: int, share: float = 0.8, bold: bool = True) -> int:
    """Largest cell up to ``cell`` that keeps ``text`` within ``share`` of the canvas width."""
    columns = _grid(text, bold).shape[1] + 3
    return max(1, min(cell, int(ui.CANVAS_SIZE[0] * share / columns)))


def countdown(
    canvas: pygame.Surface, frame: int, start_frames: int, fps: int, y: int | None = None
) -> None:
    """3, 2, 1 in big digits at the centre, then "¡YA!" for a moment."""
    if frame >= start_frames + round(0.9 * fps):
        return
    if frame < start_frames:
        left = (start_frames - frame) / fps
        number = math.ceil(left)
        text, look = str(number), (AMBER, EMBER, GOLD)[min(number, 3) - 1]
        age = 1.0 - (left - (number - 1))  # seconds since the digit appeared
    else:
        text, look = "¡YA!", GREEN
        age = (frame - start_frames) / fps
    cell = fit_cell(text, 6 + pop_extra(age), 0.7, SIGN_BOLD)
    draw(
        canvas,
        text,
        cell,
        look,
        center=(ui.CANVAS_SIZE[0] // 2, 68 if y is None else y),
        bold=SIGN_BOLD,
    )


def banner(
    canvas: pygame.Surface,
    text: str,
    age: float,
    life: float = 2.5,
    look: Look = AMBER,
    y: int | None = None,
) -> None:
    """A word across the upper half ("ÚLTIMA VUELTA", "¡META!"): pops in, holds, leaves."""
    if age < 0.0 or age >= life:
        return
    cell = fit_cell(text, 4 + pop_extra(age), 0.85, SIGN_BOLD)
    slide = 0 if age < life - 0.4 else round((age - (life - 0.4)) / 0.4 * ui.CANVAS_SIZE[0] / 2)
    draw(
        canvas,
        text,
        cell,
        look,
        center=(ui.CANVAS_SIZE[0] // 2 + slide, 56 if y is None else y),
        bold=SIGN_BOLD,
    )
