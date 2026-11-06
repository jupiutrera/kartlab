"""The main menu's building blocks: entries and flows, the list, dialogs, options and training.

An ``Entry`` is one line of the menu. It names the *steps* the player goes through before it
runs (``("kart", "track")``: pick a colour, pick a circuit, then play) and a ``launch``
function. The steps are screens registered in ``hub.STEPS``; adding a mode, or a step to a
mode, is one line in ``hub.ENTRIES`` / ``hub.STEPS`` and does not touch the flow logic.

Every screen is drawn on the 320x180 canvas of the world, with the pixel font and palette
inks (``render/pixelui.py``); the window magnifies the canvas by a whole number. Positions
are canvas pixels, also the rectangles that the mouse hits (the hub translates the mouse).
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pygame

from kartlab.render import pixelui as ui
from kartlab.show.overlays import run_title
from kartlab.show.settings import CAMERAS, LIGHTS, SCALES
from kartlab.show.title import ease_out
from kartlab.show.training_job import ConfigInfo, TrainingError, TrainingJob, list_configs

if TYPE_CHECKING:
    from kartlab.show.hub import KartLabApp

W, H = ui.CANVAS_SIZE
STATUS_H = 10  # the strip at the top where the training line goes (headings start below it)
HEADING_Y = 14  # top of the capitals of a screen's heading (scale 2)
HINT_Y = 171  # top of the capitals of the key hints line
ROW_H = 11  # a list row: a plate with room for the accents (2 rows) and the capitals (7)
IGNORED_KEYS = {
    pygame.K_LSHIFT,
    pygame.K_RSHIFT,
    pygame.K_LCTRL,
    pygame.K_RCTRL,
    pygame.K_LALT,
    pygame.K_RALT,
    pygame.K_CAPSLOCK,
}
ESC_SECONDS = 2.0  # Esc in the main menu: press it again within this time to leave
CONFIRM_KEYS = (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE)
LIGHT_NAMES = {"day": "Día", "sunset": "Atardecer", "night": "Noche"}
CAMERA_NAMES = {
    "auto": "Automática",
    "chase": "Persecución",
    "broadcast": "Retransmisión",
    "top": "Cenital",
}


def readable_log(lines: list[str]) -> list[str]:
    """The lines of a trainer's log that say something to a player: those that carry a path
    or a ``Run:`` line (folders, files) are left out."""
    path = re.compile(
        r"(^run\b|[A-Za-z]:[\\/]|[\\/][\w.-]+[\\/]|\.(yaml|csv|zip|pkl|json)\b)", re.I
    )
    return [line for line in lines if not path.search(line)]


# -- entries and flows ---------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    """One line of the main menu."""

    key: str  # what ``--open`` takes
    label: str
    description: str
    group: str | None  # block of the list (a gap separates blocks); None: the closing lines
    steps: tuple[str, ...] = ()  # screens before ``launch`` (see ``hub.STEPS``)
    launch: Callable[[KartLabApp, dict], None] | None = None
    needs_run: bool = False  # needs a run with checkpoints (otherwise: "Entrena antes a la IA")
    track_filter: Callable[[object], bool] | None = None  # tracks the circuit step offers


@dataclass
class Flow:
    """Where the player is inside an entry's steps, and what they have chosen so far."""

    entry: Entry
    index: int = 0
    choices: dict = field(default_factory=dict)

    @property
    def step(self) -> str | None:
        return self.entry.steps[self.index] if self.index < len(self.entry.steps) else None


# -- shared drawing -------------------------------------------------------------------------


def heading(canvas: pygame.Surface, text: str, age: float = 9.0, y: int = HEADING_Y) -> int:
    """The screen's title: scale 2, outlined, with a hard shadow, sliding in. Returns the y
    just below it."""
    slide = round((1.0 - ease_out(age / 0.35)) * -20)
    ui.text(
        canvas,
        text,
        W // 2,
        y + slide,
        ui.YELLOW,
        2,
        outline=ui.NIGHT,
        shadow=ui.NIGHT,
        align="center",
    )
    return y + 2 * 7 + 3


def row(
    canvas: pygame.Surface,
    rect: pygame.Rect,
    chosen: bool,
    label: str,
    value: str | None = None,
    arrows: bool = False,
    dim: bool = False,
) -> None:
    """A list row: a flat plate, the label on the left and an optional value on the right.
    The chosen row is yellow with dark text and, with ``arrows``, wraps its value in arrows."""
    ui.plate(canvas, rect, ui.YELLOW if chosen else ui.SLATE)
    ink = ui.NIGHT if chosen else (ui.STONE if dim else ui.TEXT)
    ui.text(canvas, label, rect.x + 5, rect.y + 3, ink)
    if value is not None:
        shown = f"← {value} →" if arrows and chosen else value
        ui.text(
            canvas,
            shown,
            rect.right - 5,
            rect.y + 3,
            ui.NIGHT if chosen else ui.YELLOW,
            align="right",
        )


def button(
    canvas: pygame.Surface, rect: pygame.Rect, label: str, off: bool = False, chosen: bool = False
) -> None:
    """A button: a plate with the label centred; a lit base line when it can be pressed."""
    ui.plate(canvas, rect, ui.YELLOW if chosen and not off else ui.SLATE)
    ink = ui.STONE if off else (ui.NIGHT if chosen else ui.TEXT)
    ui.text(canvas, label, rect.centerx, rect.y + (rect.h - 7) // 2 + 1, ink, align="center")
    if not off and not chosen:
        ui.hline(canvas, rect.x + 1, rect.bottom - 2, rect.w - 2, ui.YELLOW)


class Screen:
    """One screen of the menu. The hub calls these hooks; the screen draws over the veil."""

    hint = ""
    dim = 0  # extra darkness (0-255) over the backdrop, for screens with a lot to read
    hide_backdrop_kart = False  # the autopilot kart of the 3D backdrop is not drawn
    show_status = True  # the training line («Entrenando X · N/M · 3:12») in the top strip

    def __init__(self, hub: KartLabApp) -> None:
        self.hub = hub
        self.opened_at = hub.t

    def enter(self) -> None:
        self.opened_at = self.hub.t
        self.layout()

    def layout(self) -> None:
        """(Re)compute the rectangles of the screen."""

    def update(self, dt: float) -> None:
        pass

    def on_key(self, key: int) -> None:
        pass

    def on_motion(self, pos: tuple[int, int]) -> None:
        pass

    def on_click(self, pos: tuple[int, int]) -> None:
        pass

    def on_wheel(self, dy: int) -> None:
        pass

    def draw(self, canvas: pygame.Surface) -> None:
        pass

    @property
    def age(self) -> float:
        return self.hub.t - self.opened_at

    def draw_hint(
        self, canvas: pygame.Surface, text: str | None = None, ink: int = ui.STONE_LIGHT
    ) -> None:
        shown = ui.ellipsize(self.hint if text is None else text, W - 6)
        ui.caption(canvas, shown, W // 2, HINT_Y, ink)

    def draw_reason(self, canvas: pygame.Surface, reason: str) -> None:
        """Why a button is off, in the place of the hint: a warning icon and the words."""
        shown = ui.ellipsize(reason, W - 20)
        box = ui.caption(canvas, shown, W // 2 + 4, HINT_Y, ui.ORANGE)
        ui.icon(canvas, "warn", box.x - 7, HINT_Y + 1, ui.ORANGE, outline=ui.NIGHT)


# -- dialogs --------------------------------------------------------------------------------


class Dialog:
    """A message (any key closes it), a question (Intro = yes, Esc = no) or a choice between
    several buttons (``choices``: label and what to do; ← → select, Intro accepts, Esc closes
    without doing anything) over everything."""

    MAX_W = 280
    BUTTON_H = 13

    def __init__(
        self,
        title: str,
        text: str,
        on_yes: Callable[[], None] | None = None,
        footer: str | None = None,
        choices: list[tuple[str, Callable[[], None] | None]] | None = None,
    ) -> None:
        self.title = title
        self.text = text
        self.on_yes = on_yes
        self.choices = choices or []
        self.selected = 0
        self.footer = footer or (
            "←→ elegir, Intro aceptar, Esc cancelar"
            if self.choices
            else "Intro sí, Esc no"
            if on_yes is not None
            else "Pulsa una tecla"
        )
        self.buttons: list[pygame.Rect] = []
        self._layout()

    def _layout(self) -> None:
        labels = [label for label, _ in self.choices]
        buttons_w = sum(ui.measure(label) + 16 for label in labels) + 4 * max(len(labels) - 1, 0)
        longest = max(
            ui.measure(line) for line in [*self.text.split("\n"), self.title, self.footer]
        )
        self.width = min(max(longest + 16, 150, buttons_w + 16), self.MAX_W)
        self.lines = ui.wrap(self.text, self.width - 16)[:9]
        body = len(self.lines) * ui.LINE
        buttons = self.BUTTON_H + 6 if self.choices else 0
        self.height = 15 + 4 + body + buttons + 4 + 9 + 4
        self.box = pygame.Rect(0, 0, self.width, self.height)
        self.box.center = (W // 2, H // 2)
        y = self.box.y + 15 + 4 + body + 3
        x = self.box.x + 8
        self.buttons = []
        for label in labels:
            rect = pygame.Rect(x, y, ui.measure(label) + 16, self.BUTTON_H)
            self.buttons.append(rect)
            x = rect.right + 4

    def close_with(self, hub: KartLabApp, yes: bool) -> None:
        hub.dialog = None
        if yes and self.on_yes is not None:
            self.on_yes()

    def choose(self, hub: KartLabApp, index: int) -> None:
        hub.dialog = None
        action = self.choices[index][1]
        if action is not None:
            action()

    def select(self, delta: int) -> None:
        self.selected = (self.selected + delta) % len(self.choices)

    def on_key(self, hub: KartLabApp, key: int) -> None:
        if key in IGNORED_KEYS:
            return
        if self.choices:
            if key in (pygame.K_LEFT, pygame.K_UP, pygame.K_a, pygame.K_w):
                self.select(-1)
            elif key in (pygame.K_RIGHT, pygame.K_DOWN, pygame.K_d, pygame.K_s, pygame.K_TAB):
                self.select(1)
            elif key in CONFIRM_KEYS:
                self.choose(hub, self.selected)
            elif key == pygame.K_ESCAPE:
                self.close_with(hub, False)
        elif self.on_yes is None:
            self.close_with(hub, False)
        elif key in CONFIRM_KEYS:
            self.close_with(hub, True)
        elif key == pygame.K_ESCAPE:
            self.close_with(hub, False)

    def on_motion(self, pos) -> None:
        for i, rect in enumerate(self.buttons):
            if rect.collidepoint(pos) and i != self.selected:
                self.selected = i

    def on_click(self, hub: KartLabApp, pos) -> None:
        if self.choices:
            for i, rect in enumerate(self.buttons):
                if rect.collidepoint(pos):
                    self.choose(hub, i)
        elif self.on_yes is None:
            self.close_with(hub, False)

    def draw(self, canvas: pygame.Surface) -> None:
        ui.dither(canvas, canvas.get_rect(), 3)
        box = self.box
        ui.plate(canvas, box, ui.SLATE)
        ui.plate(canvas, (box.x, box.y, box.w, 15), ui.NAVY)
        ui.text(canvas, ui.ellipsize(self.title, box.w - 16), box.x + 8, box.y + 4, ui.YELLOW)
        y = box.y + 15 + 4
        for line in self.lines:
            ui.text(canvas, line, box.x + 8, y + 2, ui.TEXT)
            y += ui.LINE
        for i, rect in enumerate(self.buttons):
            button(canvas, rect, self.choices[i][0], chosen=i == self.selected)
        ui.text(canvas, self.footer, box.x + 8, box.bottom - 11, ui.STONE_LIGHT)


# -- the main list ---------------------------------------------------------------------------


class MenuScreen(Screen):
    """The title and the list of entries, in blocks, in one or two columns."""

    hint = "↑↓ elegir, Intro aceptar, Esc salir"
    COLUMN_W = 152
    HEAD_H = 10
    GAP_H = 3

    def __init__(self, hub: KartLabApp, entries: list[Entry]) -> None:
        super().__init__(hub)
        self.entries = entries
        # It opens on the first entry that can be opened (the AI modes are off without a run).
        self.selected = next(
            (i for i, e in enumerate(entries) if hub.disabled_reason(e) is None), 0
        )
        self.esc_until = -1.0  # a second Esc before this time leaves
        self._rows: list[tuple[str, Entry | str, pygame.Rect]] = []
        self.rects: dict[int, pygame.Rect] = {}  # entry index -> its row
        self.mark_y = 13

    # -- layout --------------------------------------------------------------------------

    def _blocks(self) -> list[tuple[str | None, list[int]]]:
        """The entries by header: consecutive entries of one group are a block."""
        blocks: list[tuple[str | None, list[int]]] = []
        for index, entry in enumerate(self.entries):
            if blocks and blocks[-1][0] == entry.group:
                blocks[-1][1].append(index)
            else:
                blocks.append((entry.group, [index]))
        return blocks

    def _block_h(self, block: tuple[str | None, list[int]]) -> int:
        return self.GAP_H + len(block[1]) * (ROW_H + 1)

    def layout(self) -> None:
        blocks = self._blocks()
        sizes = [self._block_h(b) for b in blocks]
        # Two columns when there is enough to fill them: split where the heights are closest.
        split = len(blocks)
        if len(blocks) > 2:
            split = min(range(1, len(blocks)), key=lambda k: abs(sum(sizes[:k]) - sum(sizes[k:])))
        columns = [blocks[:split], blocks[split:]] if split < len(blocks) else [blocks]
        mark_h = self.hub.wordmark.get_height()
        top = self.mark_y + mark_h + 3
        gap = 8
        total = len(columns) * self.COLUMN_W + (len(columns) - 1) * gap
        left = (W - total) // 2
        self._rows = []
        self.rects = {}
        self.list_bottom = top
        for c, column in enumerate(columns):
            x = left + c * (self.COLUMN_W + gap)
            y = top
            for _group, indices in column:
                y += self.GAP_H
                for index in indices:
                    rect = pygame.Rect(x, y, self.COLUMN_W, ROW_H)
                    self._rows.append(("entry", self.entries[index], rect))
                    self.rects[index] = rect
                    y += ROW_H + 1
            self.list_bottom = max(self.list_bottom, y)
        self.desc_y = self.list_bottom + 3

    # -- input ---------------------------------------------------------------------------

    def move(self, delta: int) -> None:
        self.selected = (self.selected + delta) % len(self.entries)

    def on_key(self, key: int) -> None:
        hub = self.hub
        if key == pygame.K_ESCAPE:
            if hub.t < self.esc_until:  # the second Esc: leave
                hub.request_exit()
            else:
                self.esc_until = hub.t + ESC_SECONDS
        elif key in (pygame.K_UP, pygame.K_w):
            self.move(-1)
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.move(1)
        elif key == pygame.K_HOME:
            self.selected = 0
        elif key == pygame.K_END:
            self.selected = len(self.entries) - 1
        elif key in (*CONFIRM_KEYS, pygame.K_RIGHT, pygame.K_d):
            hub.open_entry(self.entries[self.selected])

    def entry_at(self, pos) -> int | None:
        for index, rect in self.rects.items():
            if rect.collidepoint(pos):
                return index
        return None

    def on_motion(self, pos) -> None:
        index = self.entry_at(pos)
        if index is not None:
            self.selected = index

    def on_click(self, pos) -> None:
        index = self.entry_at(pos)
        if index is not None:
            self.selected = index
            self.hub.open_entry(self.entries[index])

    def on_wheel(self, dy: int) -> None:
        self.move(-1 if dy > 0 else 1)

    # -- drawing -------------------------------------------------------------------------

    def draw(self, canvas: pygame.Surface) -> None:
        hub = self.hub
        age = self.age
        mark = hub.wordmark
        drop = round((1.0 - ease_out(age / 0.7)) * -36)
        bob = round(math.sin(hub.t * 1.7))
        canvas.blit(mark, ((W - mark.get_width()) // 2, self.mark_y + drop + bob))
        appear = ease_out((age - 0.25) / 0.45)
        slide = round((1.0 - appear) * 6)
        for _kind, value, rect in self._rows:
            moved = rect.move(0, slide)
            index = self.entries.index(value)
            disabled = hub.disabled_reason(value) is not None
            row(canvas, moved, index == self.selected, value.label, dim=disabled)
            if disabled:  # a cross and the dimmed label: the reason is in the description
                ui.icon(canvas, "cross", moved.right - 10, moved.y + 3, ui.STONE)
        self._draw_description(canvas)
        self._draw_esc_notice(canvas)
        self.draw_hint(canvas)

    def _draw_description(self, canvas: pygame.Surface) -> None:
        hub = self.hub
        entry = self.entries[self.selected]
        reason = hub.disabled_reason(entry)
        lines = ui.wrap(reason or entry.description, 290)[:2]
        width = max(ui.measure(line) for line in lines) + 12
        rect = pygame.Rect((W - width) // 2, self.desc_y, width, len(lines) * ui.LINE + 3)
        ui.plate(canvas, rect, ui.NAVY)
        for k, line in enumerate(lines):
            ui.text(
                canvas,
                line,
                rect.centerx,
                rect.y + 3 + k * ui.LINE,
                ui.ORANGE if reason else ui.TEXT,
                align="center",
            )

    def _draw_esc_notice(self, canvas: pygame.Surface) -> None:
        """«Pulsa Esc otra vez para salir» after the first Esc, for a couple of seconds."""
        if self.hub.t >= self.esc_until:
            return
        text = "Esc otra vez para salir"
        rect = pygame.Rect((W - ui.measure(text) - 12) // 2, HINT_Y - 14, ui.measure(text) + 12, 11)
        ui.plate(canvas, rect, ui.YELLOW)
        ui.text(canvas, text, rect.centerx, rect.y + 3, ui.NIGHT, align="center")


# -- options ---------------------------------------------------------------------------------


class OptionsScreen(Screen):
    dim = 90
    hint = "↑↓ elegir, ←→ cambiar, Esc volver"
    ROW = 15

    def __init__(self, hub: KartLabApp, flow: Flow) -> None:
        super().__init__(hub)
        self.flow = flow
        self.selected = 0
        self.rects: list[pygame.Rect] = []

    # Each row: (label, value text, change(direction)).
    def rows(self) -> list[tuple[str, str, Callable[[int], None]]]:
        s = self.hub.settings
        runs = self.hub.run_names()

        def cycle_scale(d: int) -> None:
            i = SCALES.index(s.scale)
            s.scale = self.hub.scale = SCALES[(i + d) % len(SCALES)]
            self.hub.apply_window()

        def cycle_fullscreen(d: int) -> None:
            s.fullscreen = self.hub.fullscreen = not s.fullscreen
            self.hub.apply_window()

        def cycle_light(d: int) -> None:
            s.light = LIGHTS[(LIGHTS.index(s.light) + d) % len(LIGHTS)]
            self.hub.apply_light()

        def cycle_camera(d: int) -> None:
            s.camera = CAMERAS[(CAMERAS.index(s.camera) + d) % len(CAMERAS)]

        def cycle_run(d: int) -> None:
            options = ["", *runs]
            current = options.index(s.last_run) if s.last_run in options else 0
            s.last_run = options[(current + d) % len(options)]

        scale = (
            "Automática" if s.scale is None else f"{s.scale}×  ({320 * s.scale}×{180 * s.scale})"
        )
        run = run_title(s.last_run) if s.last_run and s.last_run in runs else "La última"
        rows = [
            ("Ventana", scale, cycle_scale),
            ("Pantalla completa", "Sí" if s.fullscreen else "No", cycle_fullscreen),
            ("Luz", LIGHT_NAMES[s.light], cycle_light),
            ("Cámara", CAMERA_NAMES[s.camera], cycle_camera),
            ("IA", run if runs else "Ninguna", cycle_run),
        ]
        return rows

    def change(self, delta: int) -> None:
        self.rows()[self.selected][2](delta)
        self.hub.save_settings()

    def layout(self) -> None:
        width = 260
        x = (W - width) // 2
        self.rects = [pygame.Rect(x, 36 + i * self.ROW, width, self.ROW - 2) for i in range(5)]

    def on_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.hub.back()
        elif key in (pygame.K_UP, pygame.K_w):
            self.selected = (self.selected - 1) % len(self.rects)
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.selected = (self.selected + 1) % len(self.rects)
        elif key in (pygame.K_LEFT, pygame.K_a):
            self.change(-1)
        elif key in (pygame.K_RIGHT, pygame.K_d, *CONFIRM_KEYS):
            self.change(1)

    def _row_at(self, pos) -> int | None:
        for i, rect in enumerate(self.rects):
            if rect.collidepoint(pos):
                return i
        return None

    def on_motion(self, pos) -> None:
        row_index = self._row_at(pos)
        if row_index is not None:
            self.selected = row_index

    def on_click(self, pos) -> None:
        row_index = self._row_at(pos)
        if row_index is not None:
            self.selected = row_index
            # the left third goes back, the rest forward
            rect = self.rects[row_index]
            self.change(-1 if pos[0] < rect.x + rect.w // 3 else 1)

    def on_wheel(self, dy: int) -> None:
        self.change(1 if dy > 0 else -1)

    def draw(self, canvas: pygame.Surface) -> None:
        heading(canvas, "OPCIONES", self.age)
        for i, (label, value, _) in enumerate(self.rows()):
            row(canvas, self.rects[i], i == self.selected, label, value, arrows=True)
        self.draw_hint(canvas)


# -- training --------------------------------------------------------------------------------


class TrainScreen(Screen):
    """Pick a config, launch it, watch it live, stop it. Nothing starts without a keypress."""

    dim = 90
    show_status = False  # it has its own panel
    hint = "Intro lanzar, V ver, P parar, O carpeta"
    ROW = 23
    PANEL_H = 43
    BUTTON_H = 13

    def __init__(self, hub: KartLabApp, flow: Flow) -> None:
        super().__init__(hub)
        self.flow = flow
        self.configs: list[ConfigInfo] = []
        self.selected = 0
        self.scroll = 0  # first visible config
        self.visible = 1  # how many fit above the job panel
        self.rects: dict[int, pygame.Rect] = {}  # config index -> its row (visible ones)
        self._laid_out_for = -1
        self.buttons: dict[str, pygame.Rect] = {}

    def enter(self) -> None:
        self.configs = list_configs(self.hub.configs_dir)
        super().enter()

    # -- state ---------------------------------------------------------------------------

    def busy_reason(self) -> str | None:
        """Why a new training cannot start now."""
        job = self.hub.job
        if job is not None and job.running:
            return "Ya hay un entrenamiento en marcha"
        if self.hub.live_training() is not None:
            return "Ya hay un entrenamiento en marcha"
        return None

    def run_for_live(self):
        """The run the live view and the folder button use."""
        hub = self.hub
        if hub.job is not None and hub.job.run_paths() is not None:
            return hub.job.run_paths()
        return hub.live_training() or hub.latest_run_with_config()

    def button_states(self) -> dict[str, tuple[str, str | None]]:
        """Button -> (label, reason it is off or None)."""
        job = self.hub.job
        config = self.configs[self.selected] if self.configs else None
        launch_off = self.busy_reason()
        if launch_off is None and (config is None or config.error):
            launch_off = "Configuración no válida"  # the box says why
        run = self.run_for_live()
        return {
            "launch": ("Lanzar", launch_off),
            "live": ("Ver", None if run else "Aún no hay entrenamientos"),
            "stop": (
                "Parar",
                None if job is not None and job.running and not job.stopping else "Nada que parar",
            ),
            "folder": ("Carpeta", None if run else "Aún no hay entrenamientos"),
        }

    # -- actions -------------------------------------------------------------------------

    def launch(self) -> None:
        hub = self.hub
        reason = self.button_states()["launch"][1]
        if reason:
            config = self.configs[self.selected] if self.configs else None
            detail = config.error if config is not None and config.error else reason
            hub.message("No se puede lanzar", detail)
            return
        config = self.configs[self.selected]
        if config.is_long:
            hub.dialog = Dialog(
                "Entrenamiento largo",
                f"«{config.name}» tarda {config.estimate}. ¿Lanzar?",
                on_yes=lambda: self._start(config),
            )
        else:
            self._start(config)

    def _start(self, config: ConfigInfo) -> None:
        hub = self.hub
        try:
            job = TrainingJob(config.path, hub.logs_dir, hub.runs_dir, spawn=hub.spawn)
            job.start()
        except TrainingError as error:
            hub.message("No se pudo lanzar", str(error))
            return
        hub.job = job

    def stop(self) -> None:
        job = self.hub.job
        if job is None or not job.running or job.stopping:
            self.hub.message("Nada que parar", "No hay nada que parar.")
            return
        job.request_stop()

    def live(self) -> None:
        run = self.run_for_live()
        if run is None:
            self.hub.message("Aún no hay entrenamientos", "Lanza un entrenamiento primero.")
            return
        self.hub.open_live(run)

    def folder(self) -> None:
        run = self.run_for_live()
        if run is None:
            self.hub.message("Aún no hay entrenamientos", "Lanza un entrenamiento primero.")
            return
        self.hub.open_folder(run.root)

    # -- layout and input ----------------------------------------------------------------

    def layout(self) -> None:
        self.width = 280
        self.x = (W - self.width) // 2
        self.top = 30
        buttons_y = HINT_Y - 6 - self.BUTTON_H - 2
        panel_y = buttons_y - 3 - self.PANEL_H
        self.visible = max(1, min(len(self.configs) or 1, (panel_y - 4 - self.top) // self.ROW))
        self._laid_out_for = len(self.configs)
        self._place()
        self.panel_rect = pygame.Rect(self.x, panel_y, self.width, self.PANEL_H)
        names = ("launch", "live", "stop", "folder")
        gap = 4
        bw = (self.width - gap * (len(names) - 1)) // len(names)
        self.buttons = {
            name: pygame.Rect(self.x + i * (bw + gap), buttons_y, bw, self.BUTTON_H)
            for i, name in enumerate(names)
        }

    def _place(self) -> None:
        """Rows of the configs in view; the selected one always is."""
        last = max(len(self.configs) - self.visible, 0)
        if self.selected < self.scroll:
            self.scroll = self.selected
        elif self.selected >= self.scroll + self.visible:
            self.scroll = self.selected - self.visible + 1
        self.scroll = min(max(self.scroll, 0), last)
        shown = range(self.scroll, min(self.scroll + self.visible, len(self.configs)))
        self.rects = {
            i: pygame.Rect(self.x, self.top + n * self.ROW, self.width, self.ROW - 2)
            for n, i in enumerate(shown)
        }

    def update(self, dt: float) -> None:
        if self._laid_out_for != len(self.configs):
            self.layout()

    def move(self, delta: int) -> None:
        if self.configs:
            self.selected = (self.selected + delta) % len(self.configs)
            self._place()

    def on_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.hub.back()
        elif key in (pygame.K_UP, pygame.K_w):
            self.move(-1)
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.move(1)
        elif key in CONFIRM_KEYS[:2]:
            self.launch()
        elif key == pygame.K_v:
            self.live()
        elif key == pygame.K_p:
            self.stop()
        elif key == pygame.K_o:
            self.folder()

    def on_motion(self, pos) -> None:
        for i, rect in self.rects.items():
            if rect.collidepoint(pos):
                self.selected = i

    def on_wheel(self, dy: int) -> None:
        self.move(-1 if dy > 0 else 1)

    def on_click(self, pos) -> None:
        for i, rect in self.rects.items():
            if rect.collidepoint(pos):
                self.selected = i
                return
        actions = {
            "launch": self.launch,
            "live": self.live,
            "stop": self.stop,
            "folder": self.folder,
        }
        for name, rect in self.buttons.items():
            if rect.collidepoint(pos):
                actions[name]()

    # -- drawing -------------------------------------------------------------------------

    def draw(self, canvas: pygame.Surface) -> None:
        heading(canvas, "ENTRENAR A LA IA", self.age)
        if not self.configs:
            ui.icon(canvas, "warn", self.x, self.top + 1, ui.ORANGE, outline=ui.NIGHT)
            ui.text(
                canvas,
                "No hay configuraciones",
                self.x + 8,
                self.top,
                ui.ORANGE,
                outline=ui.NIGHT,
            )
        for i, rect in self.rects.items():
            config = self.configs[i]
            chosen = i == self.selected
            ui.plate(canvas, rect, ui.YELLOW if chosen else ui.SLATE)
            ink = ui.NIGHT if chosen else ui.TEXT
            sub = ui.NIGHT if chosen else ui.STONE_LIGHT
            inner = rect.w - 12
            if config.error:
                name = f"{config.path.name}, no válido"
                title_ink = ui.NIGHT if chosen else ui.ORANGE
                if not chosen:
                    ui.icon(canvas, "warn", rect.right - 9, rect.y + 3, ui.ORANGE)
                detail = ui.ellipsize(config.error.splitlines()[-1], inner)
            else:
                name, title_ink = config.name, ink
                steps = f"{config.total_steps:,}".replace(",", ".")
                detail = f"{steps} pasos, {config.estimate}"
                with_comment = f"{detail}. {config.comment}" if config.comment else detail
                if ui.measure(with_comment) <= inner:  # a comment is shown whole or not at all
                    detail = with_comment
            ui.text(canvas, ui.ellipsize(name, inner - 10), rect.x + 5, rect.y + 4, title_ink)
            ui.text(canvas, ui.ellipsize(detail, inner), rect.x + 5, rect.y + 13, sub)
        if len(self.configs) > self.visible:  # more configs than rows: where we are
            shown = f"{self.scroll + 1}-{self.scroll + len(self.rects)} de {len(self.configs)} ↑ ↓"
            ui.text(
                canvas,
                shown,
                self.x + self.width,
                self.top - 9,
                ui.STONE_LIGHT,
                outline=ui.NIGHT,
                align="right",
            )
        self._draw_job_panel(canvas)
        hover = self._draw_buttons(canvas)
        if hover:
            self.draw_reason(canvas, hover)
        else:
            self.draw_hint(canvas)

    def _draw_job_panel(self, canvas: pygame.Surface) -> None:
        hub = self.hub
        rect = self.panel_rect
        inside = ui.plate(canvas, rect, ui.SLATE)
        ui.hline(canvas, rect.x + 1, rect.y + 1, rect.w - 2, ui.YELLOW)
        status = hub.status_line()
        if status:
            ui.icon(canvas, "play", inside.x + 3, inside.y + 4, ui.YELLOW)
            ui.text(
                canvas, ui.ellipsize(status, inside.w - 16), inside.x + 11, inside.y + 4, ui.YELLOW
            )
        else:
            ui.text(
                canvas,
                "Sin entrenamiento en marcha",
                inside.x + 4,
                inside.y + 4,
                ui.STONE_LIGHT,
            )
        job = hub.job
        if job is not None:
            y = inside.y + 15
            for line in readable_log(job.log_tail(12))[-3:]:
                ui.text(canvas, ui.ellipsize(line, inside.w - 8), inside.x + 4, y, ui.STONE_LIGHT)
                y += 9

    def _draw_buttons(self, canvas: pygame.Surface) -> str | None:
        """The four buttons; returns the reason of the disabled one under the mouse, if any."""
        hover = None
        for name, (label, reason) in self.button_states().items():
            rect = self.buttons[name]
            off = reason is not None
            button(canvas, rect, label, off)
            if off and reason and self._hover(rect):
                hover = reason
        return hover

    def _hover(self, rect: pygame.Rect) -> bool:
        return rect.collidepoint(self.hub.mouse)


# -- external helpers (replaceable in tests) --------------------------------------------------


def open_folder(path: Path) -> None:
    """Show a folder in the system's file manager."""
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
