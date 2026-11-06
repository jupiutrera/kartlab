"""The title screen: a pixel wordmark over a dimmed, animated 3D scene.

The wordmark is original: the Silkscreen pixel font (OFL) rendered without smoothing,
filled with a gradient of palette colours, outlined and extruded, then scaled by the same
integer factor as the world so its pixels match the ones behind it. A chequered strip
scrolls under it and a highlight sweeps across it from time to time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from kartlab.render.palette import Ink, game_palette
from kartlab.render.scene import LABEL_FONT
from kartlab.render.topdown import ACCENT, TextCache
from kartlab.render.view import DIM, MONO, SANS, SANS_REGULAR, WHITE
from kartlab.show.overlays import ease_out

TITLE_TEXT = "KART LAB"
SUBTITLE = "Un kart que aprende a conducir"
PIXEL_FONT_SIZE = 32  # Silkscreen is drawn on an 8 px grid: multiples of 8 stay sharp
SHINE_PERIOD = 5.0  # seconds between highlight sweeps
SHINE_SECONDS = 0.9
BLINK_PERIOD = 1.2


@dataclass(frozen=True)
class MenuEntry:
    key: str
    name: str
    description: str


def _rgb(ink: int) -> tuple[int, int, int]:
    return tuple(int(c) for c in game_palette().rgb[ink])


def build_wordmark(text: str = TITLE_TEXT) -> tuple[pygame.Surface, pygame.Surface]:
    """The wordmark at 1 pixel per logical pixel, and its white silhouette (for the shine)."""
    pygame.font.init()
    font = pygame.font.Font(LABEL_FONT, PIXEL_FONT_SIZE)
    glyphs = font.render(text, False, (255, 255, 255))
    mask = pygame.mask.from_surface(glyphs)
    box = mask.get_bounding_rects()
    left = min(r.left for r in box)
    top = min(r.top for r in box)
    right = max(r.right for r in box)
    bottom = max(r.bottom for r in box)
    glyphs = glyphs.subsurface((left, top, right - left, bottom - top)).copy()
    mask = pygame.mask.from_surface(glyphs)
    w, h = glyphs.get_size()

    depth, margin = 4, 3
    canvas = pygame.Surface((w + 2 * margin + depth, h + 2 * margin + depth), pygame.SRCALPHA)
    ink_edge, ink_depth = _rgb(Ink.BLACK), _rgb(Ink.RED_DARK)

    def stamp(colour, dx, dy):
        canvas.blit(mask.to_surface(setcolor=colour, unsetcolor=(0, 0, 0, 0)), (dx, dy))

    for step in range(depth, 0, -1):  # extrusion down and right, darkest at the back
        shade = ink_depth if step > 1 else _rgb(Ink.RED)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                stamp(ink_edge, margin + step + dx, margin + step + dy)
        stamp(shade, margin + step, margin + step)
    for dx in (-1, 0, 1):  # one-pixel outline
        for dy in (-1, 0, 1):
            stamp(ink_edge, margin + dx, margin + dy)

    bands = (  # gradient from the top of the letters down: highlight, yellow, amber, orange
        (0.00, Ink.WHITE),
        (0.10, Ink.YELLOW),
        (0.45, Ink.AMBER),
        (0.75, Ink.ORANGE),
    )
    gradient = pygame.Surface((w, h), pygame.SRCALPHA)
    for y in range(h):
        fraction = y / h
        ink = [ink for start, ink in bands if fraction >= start][-1]
        gradient.fill(_rgb(ink), (0, y, w, 1))
    fill = mask.to_surface(setcolor=None, unsetcolor=(0, 0, 0, 0), surface=gradient)
    canvas.blit(fill, (margin, margin))
    silhouette = pygame.Surface(canvas.get_size(), pygame.SRCALPHA)
    silhouette.blit(
        mask.to_surface(setcolor=(255, 255, 255, 255), unsetcolor=(0, 0, 0, 0)), (margin, margin)
    )
    return canvas, silhouette


class PixelText:
    """Silkscreen text without smoothing (crisp pixels), rendered once per string."""

    def __init__(self, size: int) -> None:
        self.font = pygame.font.Font(LABEL_FONT, size)
        self._cache: dict[tuple[str, tuple[int, ...]], pygame.Surface] = {}

    def get(self, text: str, color: tuple[int, ...]) -> pygame.Surface:
        key = (text, color)
        if key not in self._cache:
            self._cache[key] = self.font.render(text, False, color)
        return self._cache[key]


class TitleScreen:
    """State and drawing of the title screen (the 3D scene behind it is the app's)."""

    def __init__(self, entries: list[MenuEntry]) -> None:
        self.entries = entries
        self.selected = 0
        self.opened_at = 0.0
        self._size: tuple[int, int] | None = None
        self._wordmark, self._silhouette = build_wordmark()
        self._scaled: pygame.Surface | None = None
        self._scaled_silhouette: pygame.Surface | None = None
        self._veil: pygame.Surface | None = None

    def open(self, now: float) -> None:
        self.opened_at = now

    def move(self, delta: int) -> None:
        self.selected = (self.selected + delta) % len(self.entries)

    @property
    def entry(self) -> MenuEntry:
        return self.entries[self.selected]

    # -- layout ------------------------------------------------------------------------------

    def layout(self, size: tuple[int, int], scale: int, unit: int) -> None:
        """(Re)build everything that depends on the window size."""
        self._size = size
        self.scale = scale
        self.unit = unit
        w, h = self._wordmark.get_size()
        # The wordmark never takes more than 70 % of the width, even in tiny windows.
        k = max(1, min(scale, int(size[0] * 0.7) // w))
        self.k = k
        self._scaled = pygame.transform.scale(self._wordmark, (w * k, h * k))
        self._scaled_silhouette = pygame.transform.scale(self._silhouette, (w * k, h * k))
        self._band = pygame.Surface(self._scaled.get_size(), pygame.SRCALPHA)
        self._shine = pygame.Surface(self._scaled.get_size(), pygame.SRCALPHA)
        self.pixel = PixelText(8 * max(1, k * 3 // 4))
        self.sub = TextCache(pygame.font.Font(SANS_REGULAR, round(unit * 1.5)))
        self.name = TextCache(pygame.font.Font(SANS, round(unit * 1.5)))
        self.desc = TextCache(pygame.font.Font(SANS_REGULAR, round(unit * 1.05)))
        self.info = TextCache(pygame.font.Font(MONO, round(unit * 0.9)))
        self.hint = TextCache(pygame.font.Font(SANS_REGULAR, round(unit * 0.95)))
        self._veil = self._build_veil(size)
        self._menu_panel: pygame.Surface | None = None

    @staticmethod
    def _build_veil(size: tuple[int, int]) -> pygame.Surface:
        """Dark at the top and bottom (where the text sits), lighter in the middle."""
        w, h = size
        veil = pygame.Surface((1, h), pygame.SRCALPHA)
        for y in range(h):
            t = y / max(h - 1, 1)
            edge = abs(t - 0.5) * 2  # 0 in the middle, 1 at the edges
            veil.set_at((0, y), (6, 8, 16, round(95 + 105 * edge**2)))
        return pygame.transform.scale(veil, (w, h))

    # -- drawing -----------------------------------------------------------------------------

    def draw(
        self, screen: pygame.Surface, now: float, run_line: str, versions_line: str, hint: str
    ) -> None:
        if self._size != screen.get_size() or self._veil is None:
            raise RuntimeError("call layout() before draw()")
        age = now - self.opened_at
        width, height = self._size
        u, k = self.unit, self.k
        screen.blit(self._veil, (0, 0))

        # Wordmark: drops in with easing, then bobs one logical pixel up and down.
        mark = self._scaled
        drop = (1.0 - ease_out(age / 0.9)) * -height * 0.25
        bob = round(math.sin(now * 1.7)) * k
        x = (width - mark.get_width()) // 2
        y = round(height * 0.11 + drop) + bob
        screen.blit(mark, (x, y))
        self._draw_shine(screen, x, y, now)

        # Chequered strip under the wordmark, scrolling slowly.
        strip_y = y + mark.get_height() + u // 2
        self._draw_strip(screen, x, strip_y, mark.get_width(), now)

        alpha = min(max((age - 0.5) / 0.6, 0.0), 1.0)
        sub = self.sub.get(SUBTITLE, WHITE)
        sub_y = strip_y + k * 2 * 3 + u
        self._blit_alpha(screen, sub, ((width - sub.get_width()) // 2, sub_y), alpha)

        self._draw_menu(screen, height * 0.5, age)

        info = self.info.get(run_line, DIM)
        more = self.info.get(versions_line, ACCENT)
        base = round(height * 0.835)
        self._blit_alpha(screen, info, ((width - info.get_width()) // 2, base), alpha)
        self._blit_alpha(
            screen, more, ((width - more.get_width()) // 2, base + info.get_height()), alpha
        )

        # "Pulsa una tecla" blinks softly (a sine, not an on/off flash).
        blink = 0.55 + 0.45 * math.sin(now * math.tau / BLINK_PERIOD)
        label = self.pixel.get("PULSA UNA TECLA", ACCENT)
        self._blit_alpha(
            screen,
            label,
            ((width - label.get_width()) // 2, round(height * 0.915)),
            alpha * blink,
        )
        hints = self.hint.get(hint, DIM)
        screen.blit(hints, ((width - hints.get_width()) // 2, height - hints.get_height() - u // 2))

    @staticmethod
    def _blit_alpha(screen, surface, at, alpha: float) -> None:
        if alpha <= 0.01:
            return
        if alpha >= 0.99:
            screen.blit(surface, at)
            return
        surface.set_alpha(round(255 * alpha))
        screen.blit(surface, at)
        surface.set_alpha(255)

    def _draw_shine(self, screen, x: int, y: int, now: float) -> None:
        phase = (now % SHINE_PERIOD) / SHINE_SECONDS
        if phase > 1.0:
            return
        w, h = self._band.get_size()
        reach = w + h
        offset = round(-h + phase * reach)
        thick = max(h // 3, 8)
        self._band.fill((0, 0, 0, 0))
        points = [
            (offset, h),
            (offset + thick, h),
            (offset + thick + h // 2, 0),
            (offset + h // 2, 0),
        ]
        pygame.draw.polygon(self._band, (255, 255, 255, 150), points)
        self._shine.fill((0, 0, 0, 0))
        self._shine.blit(self._scaled_silhouette, (0, 0))
        self._shine.blit(self._band, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        screen.blit(self._shine, (x, y))

    def _draw_strip(self, screen, x: int, y: int, width: int, now: float) -> None:
        cell = self.k * 3
        rows = 2
        dark, light = _rgb(Ink.NAVY), _rgb(Ink.WHITE)
        shift = int(now * cell * 1.5) % (cell * 2)
        screen.set_clip(pygame.Rect(x, y, width, cell * rows))
        for row in range(rows):
            for col in range(-2, width // cell + 3):
                colour = light if (col + row) % 2 == 0 else dark
                screen.fill(colour, (x + col * cell + shift, y + row * cell, cell, cell))
        screen.set_clip(None)

    def _draw_menu(self, screen, top: float, age: float) -> None:
        """Three chips side by side (the road and the kart stay visible), the description
        of the chosen one underneath."""
        u = self.unit
        width = self._size[0]
        appear = ease_out((age - 0.7) / 0.5)
        if appear <= 0.01:
            return
        dark = (14, 14, 22)
        chips = []
        for i, entry in enumerate(self.entries):
            chosen = i == self.selected
            chips.append((chosen, self.name.get(entry.name, dark if chosen else WHITE)))
        gap, pad_x, pad_y = u, round(u * 1.4), u // 2
        height = chips[0][1].get_height() + 2 * pad_y
        total = sum(c.get_width() + 2 * pad_x for _, c in chips) + gap * (len(chips) - 1)
        x = (width - total) // 2
        y = round(top + (1 - appear) * u)
        layer = pygame.Surface((width, height + u * 3), pygame.SRCALPHA)
        cx = x
        for chosen, text in chips:
            w = text.get_width() + 2 * pad_x
            if chosen:
                pygame.draw.rect(layer, (*ACCENT, 255), (cx, 0, w, height))
                pygame.draw.rect(layer, (255, 255, 255, 255), (cx, 0, w, max(u // 8, 2)))
            else:
                pygame.draw.rect(layer, (14, 14, 22, 190), (cx, 0, w, height))
                pygame.draw.rect(layer, (255, 255, 255, 60), (cx, 0, w, height), 1)
            layer.blit(text, (cx + pad_x, pad_y))
            cx += w + gap
        desc = self.desc.get(self.entry.description, WHITE)
        box = pygame.Rect(0, 0, desc.get_width() + 2 * u, desc.get_height() + u // 2)
        box.midtop = (width // 2, height + u // 2)
        pygame.draw.rect(layer, (14, 14, 22, 170), box)
        layer.blit(desc, (box.x + u, box.y + u // 4))
        layer.set_alpha(round(255 * appear))
        screen.blit(layer, (0, y))
