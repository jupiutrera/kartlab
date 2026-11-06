"""Panels, cards and counters drawn over a ``GameView`` frame, on the 320x180 canvas.

Shared by the offline videos (1920x1080 = the canvas x 6, written to ffmpeg) and the
exhibition viewer. Everything is drawn with the pixel interface kit on the canvas of the
``GameView`` (``game.canvas``), before the canvas is magnified, so a pixel of a panel is a
pixel of the world.
"""

from __future__ import annotations

import re

import pygame

from kartlab.render import bigtext
from kartlab.render import pixelui as ui
from kartlab.render.hud import kart_ink
from kartlab.render.palette import KART_RAMPS
from kartlab.render.scene import KartVisual
from kartlab.render.view import GameView
from kartlab.show.drivers import PolicyDriver, format_steps

CARD_SECONDS = 2.0  # how long a card stays up by default
CANVAS_W, CANVAS_H = ui.CANVAS_SIZE
MONTHS = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")


def ease_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1.0 - (1.0 - t) ** 3


def ease_in_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 3 * t * t - 2 * t * t * t


def version_name(steps: int) -> str:
    return "Sin entrenar" if steps == 0 else f"{format_steps(steps)} pasos"


def count_steps(steps: int) -> str:
    """Exact step counter with Spanish thousands separators: 1.503.232."""
    return ui.fmt_count(steps)


def toggle_text(name: str, value: bool) -> str:
    """A setting notice, always in the same form: ``Sensores: Sí``."""
    return f"{name}: {'Sí' if value else 'No'}"


def run_title(run) -> str:
    """A training as the player knows it: ``random-tracks, 4 nov`` for the folder
    ``2026-11-04_2159_random-tracks``. The folder name itself never goes on screen."""
    name = run.root.name if hasattr(run, "root") else str(run)
    found = re.match(r"^(\d{4})-(\d{2})-(\d{2})_\d{4}_(.+)$", name)
    if not found:
        return name
    _, month, day, label = found.groups()
    return f"{label}, {int(day)} {MONTHS[int(month) - 1]}"


def visual_of(driver: PolicyDriver, ramp: int, label=None, ghost=False) -> KartVisual:
    racer = driver.racer
    return KartVisual(
        racer.kart,
        ramp,
        label,
        ghost,
        racer.on_grass,
        driver.throttle,
        list(driver.trail),
        ground=racer.level_z,
    )


def ramp_ink(ramp: int) -> int:
    return KART_RAMPS[ramp % len(KART_RAMPS)][1]


class Overlays:
    """Drawing helpers over the canvas of ``self.game``."""

    def __init__(self, game: GameView) -> None:
        self.game = game
        self.last_card: pygame.Rect | None = None

    @property
    def canvas(self) -> pygame.Surface:
        return self.game.canvas

    @property
    def size(self) -> tuple[int, int]:
        return ui.CANVAS_SIZE

    # -- building blocks --------------------------------------------------------------------

    def dim(self, amount: float) -> None:
        """Cover the frame with dark ink in dither steps: 0 leaves it as it is, 1 is black."""
        if amount > 0.003:
            ui.dither(self.canvas, self.canvas.get_rect(), round(min(amount, 1.0) * 4))

    def card(
        self, title: str, lines: list[str] = (), age: float = 0.0, life: float = CARD_SECONDS
    ) -> None:
        """A centred title card that steps up while it comes in and goes out."""
        fade_in, fade_out = 0.45, 0.4
        alpha = ease_out(age / fade_in) * min(1.0, max(0.0, (life - age) / fade_out))
        if alpha <= 0.003:
            return
        title = ui.ellipsize(title, CANVAS_W - 40, 2)
        body = [ui.ellipsize(line, CANVAS_W - 40) for line in lines]
        w = max(ui.measure(title, 2), *(ui.measure(line) for line in body), 0) + 22
        h = 8 + 14 + 6 + 9 * len(body) + 4
        hud = self.game.hud
        spots = [
            ((CANVAS_W - w) // 2 + dx, 64 - h // 2 + dy)
            for dy in (0, 28, 56)
            for dx in (0, -30, -60, -90, -120, 30, 60)
        ]
        rect = hud.reserve(hud.free_rect(w, h, spots))  # where no panel is
        self.last_card = rect
        x, y = rect.x, rect.y + round((1.0 - alpha) * 12)
        ui.plate(self.canvas, (x, y, w, h), ui.NAVY)
        self.canvas.fill(ui.rgb(ui.YELLOW), (x + 1, y + 1, 3, h - 2))
        ui.text(self.canvas, title, x + 12, y + 8, ui.WHITE, 2)
        for k, line in enumerate(body):
            ui.text(self.canvas, line, x + 12, y + 28 + 9 * k, ui.STONE_LIGHT)

    def counter(self, steps: int, subtitle: str) -> None:
        """Top-centre panel with the training step counter."""
        number = count_steps(steps)
        w = max(ui.measure(number, 2, True), ui.measure(subtitle)) + 16
        x = (CANVAS_W - w) // 2
        ui.plate(self.canvas, self.game.hud.reserve((x, 3, w, 33)), ui.SLATE)
        ui.text(self.canvas, number, CANVAS_W // 2, 7, ui.WHITE, 2, align="center", tabular=True)
        ui.text(self.canvas, subtitle, CANVAS_W // 2, 24, ui.STONE_LIGHT, align="center")

    def standings(
        self,
        rows: list[tuple[int, str, str]],
        at: tuple[int, int] = (3, 3),
        focus: int | None = None,
    ) -> None:
        """Ranking panel: rows of (ramp, label, note), first row first. ``focus`` is the
        index of the row to highlight (the kart the camera follows or the player)."""
        label_w = max(ui.measure(label) for _, label, _ in rows)
        note_w = max(ui.measure(note) for _, _, note in rows)
        w = 4 + 7 + 9 + label_w + 10 + note_w + 4
        h = 9 * len(rows) + 5
        ui.plate(self.canvas, self.game.hud.reserve((*at, w, h)), ui.SLATE)
        for i, (ramp, label, note) in enumerate(rows):
            y = at[1] + 3 + 9 * i
            ink = ui.YELLOW if i == focus else ui.TEXT
            if i == focus:  # a marker besides the colour
                ui.icon(self.canvas, "right", at[0] + 1, y + 1, ui.YELLOW)
            ui.text(self.canvas, str(i + 1), at[0] + 8, y, ui.STONE_LIGHT)
            ui.icon(self.canvas, "dot", at[0] + 15, y + 1, kart_ink(ramp), outline=ui.NIGHT)
            ui.text(self.canvas, label, at[0] + 24, y, ink)
            ui.text(self.canvas, note, at[0] + w - 4, y, ui.STONE_LIGHT, align="right")

    def chart_panel(self, chart, metrics, marker: int) -> None:
        """The reward chart, centred on the right edge."""
        surface = chart.draw(metrics.rows, metrics.version, marker)
        w, h = surface.get_size()
        at = (CANVAS_W - 3 - w, (CANVAS_H - h) // 2)
        self.game.hud.reserve((*at, w, h))
        self.canvas.blit(surface, at)

    def timeline(self, names: list[str], current: int) -> None:
        """Bottom strip with every version on the way; the current one is highlighted."""
        gap = 8
        widths = [ui.measure(name) for name in names]
        while sum(widths) + gap * (len(names) - 1) > CANVAS_W - 20 and gap > 3:
            gap -= 1
        total = sum(widths) + gap * (len(names) - 1)
        w, h = min(total, CANVAS_W - 8) + 10, 15
        x, y = (CANVAS_W - w) // 2, CANVAS_H - 3 - h
        ui.plate(self.canvas, self.game.hud.reserve((x, y, w, h)), ui.SLATE)
        cursor = x + 5
        for i, name in enumerate(names):
            ink = ui.YELLOW if i == current else ui.TEXT if i < current else ui.STONE
            ui.text(self.canvas, name, cursor, y + 3, ink)
            if i == current:
                ui.hline(self.canvas, cursor, y + 11, widths[i], ui.YELLOW)
            cursor += widths[i] + gap

    def countdown(self, frame: int, start_frames: int, fps: int) -> pygame.Rect | None:
        """3, 2, 1 before ``start_frames`` and "¡YA!" right after, in big letters, clear of
        the panels of the frame."""
        return bigtext.countdown(
            self.canvas, frame, start_frames, fps, avoid=self.game.hud.reserved
        )


def build_help(title: str | None, columns: list[list[tuple[str, str, bool]]]) -> pygame.Surface:
    """A help plate: ``columns`` of (key, action, is_heading) rows, side by side, each with
    the width its own rows need."""
    key_ws = [max((ui.measure(k) for k, _, h in col if not h), default=0) + 6 for col in columns]
    col_ws = [
        key_w + max((ui.measure(v) for _, v, _ in col), default=0)
        for key_w, col in zip(key_ws, columns, strict=True)
    ]
    gap = 8
    lines = max(len(c) for c in columns)
    width = min(sum(col_ws) + gap * (len(columns) - 1) + 16, CANVAS_W - 4)
    top = 26 if title else 9
    height = top + lines * 9 + 6
    panel = pygame.Surface((width, height))
    ui.plate(panel, (0, 0, width, height), ui.NAVY)
    panel.fill(ui.rgb(ui.YELLOW), (1, 1, width - 2, 2))
    if title:
        ui.text(panel, title, 8, 8, ui.WHITE, 2)
    x = 8
    for column, key_w, col_w in zip(columns, key_ws, col_ws, strict=True):
        y = top
        for key, value, is_head in column:
            if is_head:
                ui.text(panel, key, x, y, ui.YELLOW)
            elif key or value:
                ui.text(panel, key, x, y, ui.WHITE)
                ui.text(panel, value, x + key_w, y, ui.STONE_LIGHT)
            y += 9
        x += col_w + gap
    return panel
