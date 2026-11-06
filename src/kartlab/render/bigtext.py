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


def sign_size(text: str, cell: int, bold: bool = SIGN_BOLD) -> tuple[int, int]:
    """Pixel size of ``render(text, cell, ...)`` without drawing it."""
    rows, cols = _grid(text, bold).shape
    return (cols + 3) * cell, (rows + 3) * cell


_slots: dict[tuple, tuple[int, int] | None] = {}


def _free_center(
    size: tuple[int, int], center: tuple[int, int], avoid: tuple[tuple[int, int, int, int], ...]
) -> tuple[int, int] | None:
    """The point nearest to ``center`` where a ``size`` rectangle touches no ``avoid`` one and
    stays on the canvas (None if there is none). Searched on a 2 px lattice and cached."""
    key = (size, center, avoid)
    if key in _slots:
        return _slots[key]
    w, h = size
    cw, ch = ui.CANVAS_SIZE
    margin = 2
    rects = [pygame.Rect(r) for r in avoid]
    found = None
    if w <= cw - 2 * margin and h <= ch - 2 * margin:
        xs = range(margin + w // 2, cw - margin - w // 2 + 1, 2)
        ys = range(margin + h // 2, ch - margin - h // 2 + 1, 2)
        points = sorted(
            ((x, y) for x in xs for y in ys),
            key=lambda p: (p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2,
        )
        for x, y in points:
            box = pygame.Rect(x - w // 2, y - h // 2, w, h)
            if box.collidelist(rects) < 0:
                found = (x, y)
                break
    if len(_slots) >= 256:
        _slots.clear()
    _slots[key] = found
    return found


def place_sign(
    text: str,
    cell: int,
    pop: int,
    share: float,
    center: tuple[int, int],
    avoid=(),
    bold: bool = SIGN_BOLD,
) -> tuple[int, tuple[int, int]]:
    """Where and how big a sign is drawn so that it covers no panel: ``(cell, centre)``.

    The settled size (without the pop) looks for the free spot nearest to ``center``, shrinking
    the cell until one exists; the centre then stays put while the pop grows the cell, as far
    as it still fits there. With nothing free at the smallest cell, the default spot is used."""
    rects = tuple(tuple(r) for r in avoid)
    settled = fit_cell(text, cell, share, bold)
    for c in range(settled, 1, -1):
        spot = _free_center(sign_size(text, c, bold), center, rects)
        if spot is not None:
            break
    else:
        return 1, center
    for grown in range(fit_cell(text, cell + pop, share, bold), c, -1):
        if _free_center(sign_size(text, grown, bold), spot, rects) == spot:
            return grown, spot
    return c, spot


def countdown(
    canvas: pygame.Surface,
    frame: int,
    start_frames: int,
    fps: int,
    y: int | None = None,
    avoid=(),
) -> pygame.Rect | None:
    """3, 2, 1 in big digits at the centre, then "¡YA!" for a moment. It keeps clear of the
    rectangles in ``avoid`` (the panels) and returns where it was drawn."""
    if frame >= start_frames + round(0.9 * fps):
        return None
    if frame < start_frames:
        left = (start_frames - frame) / fps
        number = math.ceil(left)
        text, look = str(number), (AMBER, EMBER, GOLD)[min(number, 3) - 1]
        age = 1.0 - (left - (number - 1))  # seconds since the digit appeared
    else:
        text, look = "¡YA!", GREEN
        age = (frame - start_frames) / fps
    center = (ui.CANVAS_SIZE[0] // 2, 68 if y is None else y)
    cell, center = place_sign(text, 6, pop_extra(age), 0.7, center, avoid)
    return draw(canvas, text, cell, look, center=center, bold=SIGN_BOLD)


def banner(
    canvas: pygame.Surface,
    text: str,
    age: float,
    life: float = 2.5,
    look: Look = AMBER,
    y: int | None = None,
    avoid=(),
) -> pygame.Rect | None:
    """A word across the upper half ("ÚLTIMA VUELTA", "¡META!"): pops in, holds, leaves. It
    keeps clear of the rectangles in ``avoid`` (the panels) and returns where it was drawn."""
    if age < 0.0 or age >= life:
        return None
    center = (ui.CANVAS_SIZE[0] // 2, 56 if y is None else y)
    cell, center = place_sign(text, 4, pop_extra(age), 0.85, center, avoid)
    slide = 0 if age < life - 0.4 else round((age - (life - 0.4)) / 0.4 * ui.CANVAS_SIZE[0] / 2)
    return draw(canvas, text, cell, look, center=(center[0] + slide, center[1]), bold=SIGN_BOLD)
