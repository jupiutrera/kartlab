"""Race HUD in pixel art: position, lap bar, countdown, banners and results.

Shared by the human-versus-AI race (``scenes.HumanRaceScene``) and the time trial of manual
driving (``drive.py``). It is drawn on the 320x180 canvas with the interface font: the big
place, the 3-2-1 countdown and the banners are the same 5x7 letters drawn with a bigger cell
(``render/bigtext.py``), outlined, shadowed and shaded in three bands of palette inks.
"""

from __future__ import annotations

import re

import pygame

from kartlab.render import bigtext
from kartlab.render import pixelui as ui
from kartlab.render.bigtext import AMBER, COOL, GOLD, place_look, pop_extra
from kartlab.render.hud import free_rect
from kartlab.render.view import GameView

__all__ = ["AMBER", "GOLD", "RaceHud", "ordinal"]

CANVAS_W, CANVAS_H = ui.CANVAS_SIZE
SPEED_TOP = 159  # the speed plate of the HUD starts here: the place sits above it


def ordinal(place: int) -> str:
    return f"{place}º"


class RaceHud:
    """The race interface over a ``GameView`` frame. ``ov`` is the ``Overlays`` that owns the
    canvas."""

    def __init__(self, ov, sign_y: int | None = None) -> None:
        self.ov = ov
        self.sign_y = sign_y  # where the countdown and the banners are centred (None: default)
        self.panels: list[pygame.Rect] = []  # the panels of this frame, if the view has no HUD
        self._slot: tuple | None = None  # the last placement of the results card
        self.card: pygame.Rect | None = None  # where the results card was last drawn

    def begin_frame(self) -> None:
        """Forget the panels of the last frame (only needed without a view HUD, which does it)."""
        self.panels.clear()
        self.card = None

    def _reserve(self, rect):
        """Tell the view's HUD that a panel is here; without one (the top-down view), the race
        HUD keeps the list itself, as do the host's own panels through ``reserve``."""
        hud = getattr(self.ov.game, "hud", None)
        if hud is not None:
            return hud.reserve(rect)
        rect = pygame.Rect(rect)
        self.panels.append(rect)
        return rect

    reserve = _reserve

    @property
    def avoid(self) -> list[pygame.Rect]:
        """The panels that the big lettering and the results card must keep clear of."""
        hud = getattr(self.ov.game, "hud", None)
        return hud.reserved if hud is not None else self.panels

    @property
    def canvas(self):
        return self.ov.canvas

    @property
    def game(self) -> GameView:
        return self.ov.game

    # -- always on ---------------------------------------------------------------------------

    def position(self, place: int, total: int, age: float) -> None:
        """The place in the bottom right corner, above the speed: a big number with a small
        "º" and "/total", gold for the first, silver, bronze, then cooler, popping when it
        changes."""
        look = place_look(place)
        cell = 4 + pop_extra(age)
        canvas = self.canvas
        number = bigtext.render(str(place), cell, look)
        mark = bigtext.render("º", 2, look, bold=False)
        of = bigtext.render(f"/{total}", 2, COOL, bold=False)
        right = CANVAS_W - 2
        bottom = SPEED_TOP - 2
        column = max(mark.get_width(), of.get_width())
        x = right - column - number.get_width() + cell
        top = bottom - number.get_height()
        self._reserve((x, top, right - x, bottom - top))
        canvas.blit(number, (x, top))
        canvas.blit(mark, (x + number.get_width() - cell, top))
        canvas.blit(of, (x + number.get_width() - cell, bottom - of.get_height()))

    def lap_counter(
        self, lap: int, laps: int, total_time: str, lap_time: str, best: str | None = None
    ) -> None:
        """A bar at the top centre: the lap on the left and the total on the right, then the
        time of this lap and the best one. It replaces the lap panel of the view
        (``lap_panel``); the notices of the view are drawn just below."""
        canvas = self.canvas
        left = [f"VUELTA {lap}/{laps}", f"Vuelta {lap_time}"]
        right = [f"Total {total_time}", f"Mejor {best}" if best is not None else ""]
        width = max(
            ui.measure(left[0]) + ui.measure(right[0]), ui.measure(left[1]) + ui.measure(right[1])
        )
        w = width + 20
        x, y = (CANVAS_W - w) // 2, 3
        ui.plate(canvas, self._reserve((x, y, w, 21)), ui.SLATE)
        ui.text(canvas, left[0], x + 5, y + 4, ui.YELLOW, tabular=True)
        ui.text(canvas, right[0], x + w - 5, y + 4, ui.WHITE, align="right", tabular=True)
        ui.text(canvas, left[1], x + 5, y + 12, ui.STONE_LIGHT, tabular=True)
        ui.text(canvas, right[1], x + w - 5, y + 12, ui.STONE_LIGHT, align="right", tabular=True)

    # -- events ------------------------------------------------------------------------------

    def countdown(self, frame: int, start_frames: int, fps: int) -> pygame.Rect | None:
        """3, 2, 1 in big digits at the centre, then "¡YA!" for a moment, clear of the panels."""
        return bigtext.countdown(self.canvas, frame, start_frames, fps, self.sign_y, self.avoid)

    def banner(self, text: str, age: float, life: float = 2.5, look=AMBER) -> pygame.Rect | None:
        """A word across the upper half ("ÚLTIMA VUELTA", "¡META!"): pops in, holds, leaves,
        clear of the panels."""
        return bigtext.banner(self.canvas, text, age, life, look, self.sign_y, self.avoid)

    # -- results -----------------------------------------------------------------------------

    def _results_spot(self, w: int, h: int) -> tuple[int, int]:
        """Top-left corner of the results card: centred if no panel is there, else the nearest
        spot (sideways or up and down) that touches none. It stays put while the card is up."""
        avoid = [tuple(r) for r in self.avoid]
        key = (w, h, tuple(avoid))
        if self._slot is not None and self._slot[0] == key:
            return self._slot[1]
        x0, y0 = (CANVAS_W - w) // 2, (CANVAS_H - h) // 2
        offsets = sorted(
            ((dx, dy) for dx in range(-96, 97, 8) for dy in range(-72, 97, 4)),
            key=lambda d: d[0] ** 2 + d[1] ** 2,
        )
        rect = free_rect(w, h, [(x0 + dx, y0 + dy) for dx, dy in offsets], self.avoid)
        self._slot = (key, (rect.x, rect.y))
        return rect.x, rect.y

    def results(
        self,
        title: str,
        place: int | None,
        lines: list[tuple[str, int]],
        footer: str,
        age: float,
    ) -> None:
        """Results panel: a big place (if any), the title, the rows in columns (cells apart
        by two spaces) and the keys. ``lines`` are (text, ink)."""
        alpha = min(max(age / 0.4, 0.0), 1.0)
        if alpha <= 0.0:
            return
        canvas = self.canvas
        rows = [(re.split(r" {2,}", text), ink) for text, ink in lines]
        columns = max(len(cells) for cells, _ in rows)
        widths = [0] * columns
        for cells, _ in rows:
            if len(cells) > 1:  # a one-cell row is a sentence: it spans the columns
                for c, cell in enumerate(cells):
                    widths[c] = max(widths[c], ui.measure(cell, tabular=True))
        gap = 8
        table_w = sum(widths) + gap * (columns - 1)
        sentence_w = max((ui.measure(cells[0]) for cells, _ in rows if len(cells) == 1), default=0)
        badge = bigtext.render(ordinal(place), 3, place_look(place)) if place is not None else None
        head_w = ui.measure(title, 2) + (badge.get_width() + 8 if badge else 0)
        w = max(table_w, sentence_w, head_w, ui.measure(footer)) + 24
        top_h = max(14, badge.get_height() if badge else 0)
        sentences = sum(
            1
            for k, (cells, _) in enumerate(rows)
            if len(cells) == 1 and k and len(rows[k - 1][0]) > 1
        )  # a sentence after the table gets a little air
        h = 8 + top_h + 8 + 9 * len(rows) + 3 * sentences + 6 + 9 + 6
        x, y = self._results_spot(w, h)
        self.card = pygame.Rect(x, y, w, h)
        y += round((1.0 - alpha) * 12)
        ui.plate(canvas, self._reserve((x, y, w, h)), ui.NAVY)
        canvas.fill(ui.rgb(ui.YELLOW), (x + 1, y + 1, 3, h - 2))
        cx = x + 12
        if badge is not None:
            canvas.blit(badge, (cx, y + 6))
            cx += badge.get_width() + 8
        ui.text(canvas, title, cx, y + 8 + (top_h - 14) // 2, ui.WHITE, 2)
        ry = y + 8 + top_h + 8
        previous = None
        for cells, ink in rows:
            if len(cells) == 1 and previous is not None and len(previous) > 1:
                ry += 3
            previous = cells
            if len(cells) == 1:
                ui.text(canvas, cells[0], x + 12, ry, ink)
            else:
                cursor = x + 12
                for c, cell in enumerate(cells):
                    ui.text(canvas, cell, cursor, ry, ink, tabular=True)
                    cursor += widths[c] + gap
            ry += 9
        ui.text(canvas, footer, x + 12, ry + 6, ui.STONE_LIGHT)
