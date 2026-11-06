"""Pixel interface kit: the 16-bit look for every screen that is not the world.

The interface is drawn on the same 320x180 grid as the world, with the 64 colours of the
palette and nothing else, and the whole canvas is then magnified by the same integer
factor. A pixel of text is therefore a pixel of the world: no smoothing, no vector fonts,
no translucent panels, no rounded corners.

Style rules (the ``pixelart-canvas`` guide, adapted to this palette and canvas):

- Integer coordinates and integer scales (1 to 4 for text).
- Flat plates with a 1 px border in the darkest ink (``NIGHT``); text over art gets a 1 px
  outline in the same ink.
- One font (``pixelfont``), upper case, Spanish decimal comma.
- Red only means error, and colour is never the only signal: states carry a shape or a word.
- Veils darken with a dither pattern, never with a blend.

Everything draws on a ``pygame.Surface`` the caller owns (the 320x180 canvas, or the world
surface just before it is scaled). Strings are rendered once into small colour-keyed
surfaces and cached, so a frame of interface is a few dozen blits.
"""

from __future__ import annotations

import numpy as np
import pygame

from kartlab.render.palette import Ink, game_palette
from kartlab.render.pixelfont import (
    CAP_TOP,
    GLYPH_H,
    LINE,
    ellipsize,
    measure,
    text_mask,
    wrap,
    wrap_clip,
)

__all__ = [
    "CANVAS_SIZE",
    "LINE",
    "NIGHT",
    "Rolling",
    "bar",
    "caption",
    "box",
    "comma",
    "dither",
    "ellipsize",
    "fmt_clock",
    "fmt_count",
    "fmt_decimal",
    "hline",
    "icon",
    "keycap",
    "measure",
    "plate",
    "rgb",
    "tag",
    "text",
    "vline",
    "wrap",
    "wrap_clip",
]

CANVAS_SIZE = (320, 180)  # the logical resolution of the world and of the interface

# -- roles: palette inks by use --------------------------------------------------------------

NIGHT = Ink.BLACK  # borders, outlines, holes: the darkest ink
SLATE = Ink.ASPHALT  # plates
SLATE_LIGHT = Ink.ASPHALT_LIGHT  # a raised plate, a pressed key
STONE = Ink.STONE  # muted text, guides, zero values
STONE_LIGHT = Ink.STONE_LIGHT  # secondary text
TEXT = Ink.MIST  # main text
WHITE = Ink.WHITE  # the brightest highlight
NAVY = Ink.NAVY  # title bars, screens
NAVY_LIGHT = Ink.DUSK
YELLOW = Ink.YELLOW  # what to look at now, the best value
ORANGE = Ink.ORANGE  # the main kart, warnings
GREEN = 31  # correct, ahead
RED = Ink.KERB_RED  # errors only
BLUE = 47
CYAN = 48
LIME = Ink.LEAF_LIGHT
PURPLE = Ink.PURPLE

_KEY = (255, 0, 255)  # colour key of the cached surfaces: not in the palette, never on screen
_rgb_table: list[tuple[int, int, int]] = []


def rgb(ink: int) -> tuple[int, int, int]:
    """The colour of a palette index."""
    if not _rgb_table:
        _rgb_table.extend(tuple(int(c) for c in row) for row in game_palette().rgb)
    return _rgb_table[ink]


# -- numbers in Spanish ----------------------------------------------------------------------


def comma(text: str) -> str:
    """Decimal comma: ``1.5`` -> ``1,5``."""
    return text.replace(".", ",")


def fmt_decimal(value: float, digits: int = 1) -> str:
    """``fmt_decimal(0.5, 2)`` -> ``0,50``. A negative zero is shown without sign."""
    shown = f"{value:.{digits}f}"
    if float(shown) == 0:
        shown = shown.lstrip("-")
    return comma(shown)


def fmt_clock(seconds: float | None, digits: int = 2) -> str:
    """A lap time as ``1:23,45``; ``-:--,--`` when there is none."""
    if seconds is None:
        return "-:--," + "-" * digits
    minutes, rest = divmod(seconds, 60.0)
    width = digits + 3 if digits else 2
    return comma(f"{int(minutes)}:{rest:0{width}.{digits}f}")


def fmt_count(value: int) -> str:
    """Thousands separated by dots: ``1503232`` -> ``1.503.232``."""
    return f"{value:,}".replace(",", ".")


# -- cached strings --------------------------------------------------------------------------

_MAX_CACHED = 1024
_strings: dict[tuple, pygame.Surface] = {}


def _shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(mask)
    h, w = mask.shape
    out[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] = mask[
        max(-dy, 0) : h + min(-dy, 0), max(-dx, 0) : w + min(-dx, 0)
    ]
    return out


def _string_surface(
    value: str, ink: int, scale: int, outline: int | None, shadow: int | None, tabular: bool
) -> pygame.Surface:
    key = (value, ink, scale, outline, shadow, tabular)
    surface = _strings.get(key)
    if surface is not None:
        return surface
    mask = text_mask(value, scale, tabular)
    pad = 1 if (outline is not None or shadow is not None) else 0
    h, w = mask.shape
    mask = np.pad(mask, pad)
    colours = np.empty((h + 2 * pad, w + 2 * pad, 3), dtype=np.uint8)
    colours[:] = _KEY
    if shadow is not None:
        colours[_shift(mask, 1, 1)] = rgb(shadow)
    if outline is not None:
        grown = np.zeros_like(mask)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                grown |= _shift(mask, dy, dx)
        colours[grown] = rgb(outline)
    colours[mask] = rgb(ink)
    surface = pygame.surfarray.make_surface(np.ascontiguousarray(colours.transpose(1, 0, 2)))
    surface.set_colorkey(_KEY)
    if len(_strings) >= _MAX_CACHED:
        _strings.clear()
    _strings[key] = surface
    return surface


def text(
    target: pygame.Surface,
    value: str,
    x: int,
    y: int,
    ink: int = TEXT,
    scale: int = 1,
    outline: int | None = None,
    shadow: int | None = None,
    align: str = "left",
    tabular: bool = False,
) -> int:
    """Draw ``value`` with the top of its capitals at ``y``; ``x`` is the left, centre or
    right edge according to ``align``. The outline and the shadow are always 1 px thick,
    whatever the scale. Returns the width of the text (without the outline)."""
    if not value:
        return 0
    width = measure(value, scale, tabular)
    surface = _string_surface(value, ink, scale, outline, shadow, tabular)
    pad = 1 if (outline is not None or shadow is not None) else 0
    if align == "center":
        x -= width // 2
    elif align == "right":
        x -= width
    target.blit(surface, (x - pad, y - CAP_TOP * scale - pad))
    return width


# -- shapes ----------------------------------------------------------------------------------


def _rect(r) -> pygame.Rect:
    return r if isinstance(r, pygame.Rect) else pygame.Rect(r)


def hline(target: pygame.Surface, x: int, y: int, length: int, ink: int) -> None:
    target.fill(rgb(ink), (x, y, length, 1))


def vline(target: pygame.Surface, x: int, y: int, length: int, ink: int) -> None:
    target.fill(rgb(ink), (x, y, 1, length))


def box(target: pygame.Surface, rect, ink: int) -> None:
    """A 1 px frame."""
    r = _rect(rect)
    hline(target, r.x, r.y, r.w, ink)
    hline(target, r.x, r.bottom - 1, r.w, ink)
    vline(target, r.x, r.y, r.h, ink)
    vline(target, r.right - 1, r.y, r.h, ink)


def plate(target: pygame.Surface, rect, fill: int = SLATE, edge: int | None = NIGHT) -> pygame.Rect:
    """A flat plate: fill and a 1 px border. Returns the inside, for placing content."""
    r = _rect(rect)
    if edge is None:
        target.fill(rgb(fill), r)
        return r
    target.fill(rgb(edge), r)
    target.fill(rgb(fill), r.inflate(-2, -2))
    return r.inflate(-2, -2)


def tag(
    target: pygame.Surface, label: str, x: int, y: int, ink: int = TEXT, fill: int = SLATE
) -> pygame.Rect:
    """A neutral label: text on a small plate (never red). ``x, y`` is its top left."""
    rect = pygame.Rect(x, y, measure(label) + 6, CAP_TOP + 9)
    plate(target, rect, fill)
    text(target, label, x + 3, y + 2, ink)
    return rect


def caption(
    target: pygame.Surface, label: str, cx: int, y: int, ink: int = TEXT, fill: int = SLATE
) -> pygame.Rect:
    """A line of text on its own plate, centred on ``cx``, capitals at ``y``: it reads over
    any background, a dither veil included."""
    rect = pygame.Rect(0, 0, measure(label) + 8, 12)
    rect.midtop = (cx, y - 3)
    plate(target, rect, fill)
    text(target, label, cx, y, ink, align="center")
    return rect


def keycap(target: pygame.Surface, label: str, x: int, y: int, ink: int = TEXT) -> pygame.Rect:
    """A key as printed in a hint: a raised plate with its name. Top left at ``x, y``."""
    rect = pygame.Rect(x, y, measure(label) + 5, 11)
    plate(target, rect, SLATE_LIGHT)
    hline(target, rect.x + 1, rect.bottom - 2, rect.w - 2, NIGHT)
    text(target, label, x + 3, y + 2, ink)
    return rect


def bar(
    target: pygame.Surface,
    rect,
    value: float,
    ink: int,
    centered: bool = False,
    track: int = SLATE,
    edge: int = NIGHT,
) -> None:
    """A horizontal meter in a plate. ``value`` is 0..1, or -1..1 when ``centered`` (it
    grows from the middle: right for positive, left for negative; a tick marks the middle)."""
    inside = plate(target, rect, track, edge)
    if inside.w <= 0 or inside.h <= 0:
        return
    if centered:
        value = max(-1.0, min(1.0, value))
        half = inside.w // 2
        reach = round(abs(value) * half)
        mid = inside.x + half
        if reach:
            x = mid if value > 0 else mid - reach
            target.fill(rgb(ink), (x, inside.y, reach, inside.h))
        vline(target, mid, inside.y, inside.h, STONE_LIGHT if not reach else NIGHT)
    else:
        value = max(0.0, min(1.0, value))
        reach = round(value * inside.w)
        if reach:
            target.fill(rgb(ink), (inside.x, inside.y, reach, inside.h))


_ICONS: dict[str, tuple[str, ...]] = {
    "check": ("....#", "...##", "#.##.", ".##..", "..#.."),
    "cross": ("#...#", ".#.#.", "..#..", ".#.#.", "#...#"),
    "right": ("#....", "##...", "###..", "##...", "#...."),
    "left": ("....#", "...##", "..###", "...##", "....#"),
    "up": ("..#..", ".###.", "#####", ".....", "....."),
    "down": (".....", ".....", "#####", ".###.", "..#.."),
    "play": ("#....", "##...", "###..", "##...", "#...."),
    "pause": ("##.##",) * 5,
    "dot": (".###.", "#####", "#####", "#####", ".###."),
    "ring": (".###.", "#...#", "#...#", "#...#", ".###."),
    "warn": ("..#..", "..#..", "..#..", ".....", "..#.."),
}
_icon_surfaces: dict[tuple[str, int, int | None], pygame.Surface] = {}


def icon(
    target: pygame.Surface, name: str, x: int, y: int, ink: int = TEXT, outline: int | None = None
) -> None:
    """A 5x5 symbol with its top left at ``x, y``."""
    key = (name, ink, outline)
    surface = _icon_surfaces.get(key)
    if surface is None:
        rows = _ICONS[name]
        mask = np.array([[c == "#" for c in row] for row in rows], dtype=bool)
        pad = 0 if outline is None else 1
        mask = np.pad(mask, pad)
        colours = np.empty((*mask.shape, 3), dtype=np.uint8)
        colours[:] = _KEY
        if outline is not None:
            grown = np.zeros_like(mask)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    grown |= _shift(mask, dy, dx)
            colours[grown] = rgb(outline)
        colours[mask] = rgb(ink)
        surface = pygame.surfarray.make_surface(np.ascontiguousarray(colours.transpose(1, 0, 2)))
        surface.set_colorkey(_KEY)
        _icon_surfaces[key] = surface
    pad = 0 if outline is None else 1
    target.blit(surface, (x - pad, y - pad))


# Ordered-dither thresholds (4x4 Bayer): a veil of level k covers k/4 of the pixels.
_BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])
_dithers: dict[tuple[int, ...], pygame.Surface] = {}


def dither(target: pygame.Surface, rect, level: int, ink: int = NIGHT) -> None:
    """Darken an area with a dither pattern: ``level`` 1 to 3 covers a quarter, a half or
    three quarters of the pixels. The 16-bit way of making a veil with a closed palette."""
    r = _rect(rect).clip(target.get_rect())
    if r.w <= 0 or r.h <= 0 or level <= 0:
        return
    level = min(level, 4)
    key = (r.x % 4, r.y % 4, r.w, r.h, level, ink)
    surface = _dithers.get(key)
    if surface is None:
        ys, xs = np.mgrid[r.y : r.y + r.h, r.x : r.x + r.w]
        covered = _BAYER[ys % 4, xs % 4] < level * 4
        colours = np.empty((r.h, r.w, 3), dtype=np.uint8)
        colours[:] = _KEY
        colours[covered] = rgb(ink)
        surface = pygame.surfarray.make_surface(np.ascontiguousarray(colours.transpose(1, 0, 2)))
        surface.set_colorkey(_KEY)
        if len(_dithers) >= 64:
            _dithers.clear()
        _dithers[key] = surface
    target.blit(surface, r.topleft)


class Rolling:
    """A counter whose digits turn like an odometer: fixed 6 px cells, right aligned, and
    only the digit that changes moves (80 ms, ease out). The number never reads broken."""

    DURATION = 0.08

    def __init__(self) -> None:
        self.shown = ""
        self.previous = ""
        self.since = -1.0

    def draw(
        self,
        target: pygame.Surface,
        value: int,
        x: int,
        y: int,
        ink: int,
        scale: int = 1,
        now: float = 0.0,
        outline: int | None = NIGHT,
    ) -> int:
        """Draw ``value`` with its left edge at ``x``; returns the x after the last cell."""
        current = str(value)
        if not self.shown:
            self.shown = self.previous = current
        elif current != self.shown:
            self.previous, self.shown = self.shown, current
            self.since = now
        length = max(len(current), len(self.previous))
        new = current.rjust(length)
        old = self.previous.rjust(length)
        cell = 6 * scale
        height = 7 * scale
        progress = min(1.0, max(0.0, (now - self.since) / self.DURATION))
        eased = 1.0 - (1.0 - progress) ** 3
        offset = round(eased * (height + 2))
        skip = length - len(current) if progress >= 1.0 else 0
        cx = x
        for i in range(skip, length):

            def digit(ch: str, at: int, left: int = cx) -> None:
                if ch != " ":
                    text(target, ch, left, at, ink, scale, outline, tabular=True)

            if progress < 1.0 and old[i] != new[i]:
                clip = target.get_clip()
                target.set_clip(pygame.Rect(cx - 1, y - 1, cell, height + 2).clip(clip))
                digit(old[i], y - offset)
                digit(new[i], y + height + 2 - offset)
                target.set_clip(clip)
            else:
                digit(new[i], y)
            cx += cell
        return cx


def line_height(scale: int = 1) -> int:
    """Pitch of consecutive lines of text at ``scale``."""
    return LINE * scale if scale > 1 else LINE


def paragraph(
    target: pygame.Surface,
    value: str,
    x: int,
    y: int,
    width: int,
    ink: int = TEXT,
    scale: int = 1,
    max_lines: int | None = None,
    pitch: int | None = None,
) -> int:
    """Wrapped text; returns the y after the last line. Lines beyond ``max_lines`` are
    dropped and the last one ends in an ellipsis."""
    lines = wrap(value, width, scale)
    pitch = pitch if pitch is not None else line_height(scale)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = ellipsize(lines[-1] + " …", width, scale)
    for line in lines:
        text(target, line, x, y, ink, scale)
        y += pitch
    return y


__all__ += ["GLYPH_H", "line_height", "paragraph"]
