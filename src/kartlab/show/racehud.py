"""Race HUD in pixel art: position, lap counter, countdown, banners and results.

Shared by the human-versus-AI race (``scenes.HumanRaceScene``) and the time trial of manual
driving (``drive.py``). Everything is drawn at window resolution. Digits and letters are
grids of cells (5 columns by 7 rows) drawn in whole blocks of N pixels with a dark outline,
a drop shadow and a vertical gradient fill; they are never scaled with smoothing, only
redrawn with a bigger cell, so a "pop" is a jump of whole cells. Every text is rendered once
per (text, cell size, colours) and cached.

The glyph set is the one the HUD needs and nothing else. The look is original.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from kartlab.render.view import DIM, MONO, WHITE

Color = tuple[int, int, int]

# 5x7 glyphs, one string per row, "#" is a lit cell. Shorter glyphs sit on the baseline.
GLYPHS: dict[str, tuple[str, ...]] = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "01010", "01010", "00100"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Ú": ("00100", "00000", "00000", "10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "/": ("00001", "00001", "00010", "00100", "01000", "10000", "10000"),
    "!": ("00100", "00100", "00100", "00100", "00100", "00000", "00100"),
    "¡": ("00100", "00000", "00100", "00100", "00100", "00100", "00100"),
    "º": ("010", "101", "010"),
    " ": ("000", "000", "000", "000", "000", "000", "000"),
}
GLYPH_ROWS = 7

# Fill gradients (top, bottom) and outline of each look.
GOLD = ((255, 232, 112), (232, 150, 28))
SILVER = ((236, 242, 252), (128, 146, 186))
BRONZE = ((246, 176, 112), (170, 88, 52))
COOL = ((170, 196, 240), (84, 108, 176))
RED = ((255, 150, 130), (214, 64, 64))
AMBER = ((255, 214, 120), (236, 128, 40))
GREEN = ((196, 250, 150), (70, 176, 92))
PAPER = ((250, 250, 244), (178, 184, 204))
OUTLINE: Color = (16, 14, 28)
SHADOW: Color = (8, 8, 16)

POP_SECONDS = 0.3
POP_STEPS = (3, 2, 1, 0)  # extra cells added to the cell size, one step every POP_SECONDS / 4


def place_look(place: int) -> tuple[Color, Color]:
    """Fill gradient of a finishing place: gold, silver, bronze, then cool blues."""
    return (GOLD, SILVER, BRONZE)[place - 1] if 1 <= place <= 3 else COOL


def ordinal(place: int) -> str:
    return f"{place}º"


def text_mask(text: str) -> np.ndarray:
    """Boolean grid of ``text`` (one empty column between glyphs, baseline aligned).

    Every glyph is made bold (each lit cell also lights the one to its right and the one
    below), so strokes are two cells thick and survive the outline; "º" stays thin and sits
    at the top."""
    glyphs = []
    for char in text:
        rows = GLYPHS.get(char)
        if rows is None:
            raise KeyError(f"sin glifo para {char!r}")
        glyphs.append((char, rows))
    height = max(len(rows) for _, rows in glyphs) + 1
    columns: list[np.ndarray] = []
    for char, rows in glyphs:
        thin = char == "º"
        grid = np.zeros((height, len(rows[0]) + (0 if thin else 1)), dtype=bool)
        top = 0 if thin else height - len(rows) - 1
        for r, row in enumerate(rows):
            grid[top + r, : len(row)] = [c == "1" for c in row]
        if not thin:
            base = grid.copy()
            grid[:, 1:] |= base[:, :-1]
            grid[1:, :] |= base[:-1, :]
        columns.append(grid)
        columns.append(np.zeros((height, 1), dtype=bool))
    return np.concatenate(columns[:-1], axis=1)


def _dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1)
    out = np.zeros_like(mask)
    h, w = mask.shape
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy : dy + h, dx : dx + w]
    return out


def render_cells(
    mask: np.ndarray, cell: int, look: tuple[Color, Color], outline: Color = OUTLINE
) -> pygame.Surface:
    """Draw a cell grid with an outline, a drop shadow one cell down-right and a gradient."""
    h, w = mask.shape
    pad = 1
    grid = np.pad(mask, pad)
    body = _dilate(grid)  # fill plus one cell of outline
    shadow = np.zeros_like(grid)
    shadow[1:, 1:] = body[:-1, :-1]
    rows, cols = grid.shape
    surface = pygame.Surface(((cols + 1) * cell, (rows + 1) * cell), pygame.SRCALPHA)
    for y, x in zip(*np.nonzero(shadow & ~body), strict=True):
        surface.fill(SHADOW, (int(x) * cell, int(y) * cell, cell, cell))
    for y, x in zip(*np.nonzero(body & ~grid), strict=True):
        surface.fill(outline, (int(x) * cell, int(y) * cell, cell, cell))
    top, bottom = look
    for y, x in zip(*np.nonzero(grid), strict=True):
        t = (int(y) - pad) / max(h - 1, 1)
        colour = tuple(round(a + (b - a) * t) for a, b in zip(top, bottom, strict=True))
        surface.fill(colour, (int(x) * cell, int(y) * cell, cell, cell))
    return surface


class PixelText:
    """Cache of pixel-art texts keyed by (text, cell size, look)."""

    def __init__(self, max_items: int = 160) -> None:
        self.max_items = max_items
        self._cache: dict[tuple, pygame.Surface] = {}

    def get(self, text: str, cell: int, look: tuple[Color, Color] = PAPER) -> pygame.Surface:
        key = (text, cell, look)
        surface = self._cache.get(key)
        if surface is None:
            if len(self._cache) >= self.max_items:
                self._cache.clear()
            surface = render_cells(text_mask(text), max(cell, 1), look)
            self._cache[key] = surface
        return surface

    def width(self, text: str, cell: int) -> int:
        return (text_mask(text).shape[1] + 3) * max(cell, 1)


def pop_extra(age: float) -> int:
    """Extra cells of a "pop" that is ``age`` seconds old: a jump up, then back in steps."""
    if age < 0.0 or age >= POP_SECONDS:
        return 0
    return POP_STEPS[min(int(age / (POP_SECONDS / len(POP_STEPS))), len(POP_STEPS) - 1)]


class RaceHud:
    """The race interface over a ``GameView`` frame. ``ov`` is the ``Overlays`` that owns the
    screen, the unit size and the text caches."""

    def __init__(self, ov) -> None:
        self.ov = ov
        self.pixels = PixelText()
        self._mono: pygame.font.Font | None = None
        self._mono_unit = 0

    # -- sizes -------------------------------------------------------------------------------

    @property
    def big_cell(self) -> int:
        return max(round(self.ov.unit * 0.5), 3)

    @property
    def small_cell(self) -> int:
        return max(round(self.ov.unit * 0.25), 2)

    def _fit(self, text: str, cell: int, share: float = 0.8) -> int:
        """Largest cell size up to ``cell`` that keeps ``text`` within ``share`` of the width."""
        columns = text_mask(text).shape[1] + 3
        return max(1, min(cell, int(self.ov.size[0] * share / columns)))

    def mono(self) -> pygame.font.Font:
        if self._mono is None or self._mono_unit != self.ov.unit:
            self._mono = pygame.font.Font(MONO, round(self.ov.unit * 1.05))
            self._mono_unit = self.ov.unit
        return self._mono

    # -- always on ---------------------------------------------------------------------------

    def position(self, place: int, total: int, age: float) -> None:
        """The place in a corner (bottom right, above the speed): a big number with a small
        "º" and "/total", gold for the first, cooler for the rest, popping when it changes."""
        ov, u = self.ov, self.ov.unit
        cell = self.big_cell * 3 // 2 + pop_extra(age)
        number = self.pixels.get(str(place), cell, place_look(place))
        ordinal_mark = self.pixels.get("º", max(cell // 2, 2), place_look(place))
        of = self.pixels.get(f"/{total}", max(self.small_cell, cell // 3), COOL)
        pad = u // 2
        right = ov.size[0] - pad
        bottom = ov.game.world_rect.bottom - round(u * 3.6)
        # Anchored at the corner: the pop grows the number up and to the left.
        column = max(ordinal_mark.get_width(), of.get_width())
        x = right - column - number.get_width() + cell
        top = bottom - number.get_height()
        ov.screen.blit(number, (x, top))
        ov.screen.blit(ordinal_mark, (x + number.get_width() - cell, top))
        ov.screen.blit(of, (x + number.get_width() - cell, bottom - of.get_height()))

    def lap_counter(
        self, lap: int, laps: int, total_time: str, lap_time: str, best: str | None = None
    ) -> None:
        """ "VUELTA 1/3" in small pixel art at the top centre, with the total time, the time of
        this lap and the best one beside it. One row, so the notices of the 3D view (drawn
        just below) stay readable. It replaces the lap panel of the view (``lap_panel``)."""
        ov, u = self.ov, self.ov.unit
        game = ov.game
        sign = self.pixels.get(f"VUELTA {lap}/{laps}", self.small_cell, PAPER)
        text = f"Total {total_time}  ·  Vuelta {lap_time}"
        if best is not None:
            text += f"  ·  Mejor {best}"
        clock = game.small.get(text, WHITE)
        gap = u
        w = sign.get_width() + gap + clock.get_width() + u
        h = sign.get_height() + u // 4
        x = (ov.size[0] - w) // 2
        y = game.world_rect.y + u // 4
        ov.screen.blit(ov.panel(w, h), (x, y))
        ov.screen.blit(sign, (x + u // 2, y + u // 8))
        ov.screen.blit(
            clock, (x + u // 2 + sign.get_width() + gap, y + (h - clock.get_height()) // 2)
        )

    # -- events ------------------------------------------------------------------------------

    def countdown(self, frame: int, start_frames: int, fps: int) -> None:
        """3, 2, 1 in big pixel digits at the centre, then "¡YA!" for a moment."""
        if frame >= start_frames + round(0.9 * fps):
            return
        if frame < start_frames:
            left = (start_frames - frame) / fps
            number = math.ceil(left)
            text, look = str(number), (RED, AMBER, GOLD)[min(number, 3) - 1]
            age = 1.0 - (left - (number - 1))  # seconds since the digit appeared
        else:
            text, look = "¡YA!", GREEN
            age = (frame - start_frames) / fps
        cell = self.big_cell * 2 + pop_extra(age)
        surface = self.pixels.get(text, self._fit(text, cell, 0.6), look)
        rect = surface.get_rect(center=(self.ov.size[0] // 2, round(self.ov.size[1] * 0.38)))
        self.ov.screen.blit(surface, rect)

    def banner(self, text: str, age: float, life: float = 2.5, look=AMBER) -> None:
        """A word across the upper half ("ÚLTIMA VUELTA", "¡META!"): pops in, holds, leaves."""
        if age < 0.0 or age >= life:
            return
        cell = self._fit(text, self.big_cell + self.big_cell // 2 + pop_extra(age), 0.7)
        surface = self.pixels.get(text, cell, look)
        slide = 0 if age < life - 0.4 else round((age - (life - 0.4)) / 0.4 * self.ov.size[0] / 2)
        rect = surface.get_rect(center=(self.ov.size[0] // 2 + slide, round(self.ov.size[1] * 0.3)))
        self.ov.screen.blit(surface, rect)

    # -- results -----------------------------------------------------------------------------

    def results(
        self,
        kicker: str,
        place: int | None,
        lines: list[tuple[str, Color]],
        footer: str,
        age: float,
    ) -> None:
        """Results panel: a big place (if any), a kicker and fixed-width lines, plus the keys.
        Lines are drawn in a monospaced font, so the caller aligns columns with padding."""
        ov, u = self.ov, self.ov.unit
        alpha = min(max(age / 0.4, 0.0), 1.0)
        if alpha <= 0.0:
            return
        font = self.mono()
        head = ov.kicker.get(kicker.upper(), (251, 185, 84))
        texts = [font.render(text, True, colour) for text, colour in lines]
        foot = ov.body.get(footer, DIM)
        badge = None
        if place is not None:
            badge = self.pixels.get(ordinal(place), self.big_cell, place_look(place))
        pad = round(u * 1.2)
        line_h = max(t.get_height() for t in texts)
        body_w = max(t.get_width() for t in texts)
        top_h = max(head.get_height(), badge.get_height() if badge else 0)
        w = max(body_w, foot.get_width(), head.get_width() + (badge.get_width() if badge else 0))
        w += 2 * pad
        h = pad * 3 + top_h + line_h * len(texts) + foot.get_height()
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        card.fill((14, 14, 22, 238))
        pygame.draw.rect(card, (251, 185, 84), (0, 0, max(u // 4, 4), h))
        y = pad
        if badge is not None:
            card.blit(badge, (pad, y))
            head_y = y + (badge.get_height() - head.get_height()) // 2
            card.blit(head, (pad + badge.get_width() + u, head_y))
        else:
            card.blit(head, (pad, y))
        y += top_h + pad
        for surface in texts:
            card.blit(surface, (pad, y))
            y += line_h
        card.blit(foot, (pad, y + pad))
        card.set_alpha(round(255 * alpha))
        slide = round((1.0 - alpha) * u * 2)
        cx, cy = ov.size[0] // 2, round(ov.size[1] * 0.46)
        ov.screen.blit(card, (cx - w // 2, cy - h // 2 + slide))
