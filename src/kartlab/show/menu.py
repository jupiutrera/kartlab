"""The main menu's building blocks: entries and flows, the list, dialogs, options and training.

An ``Entry`` is one line of the menu. It names the *steps* the player goes through before it
runs (``("kart", "track")``: pick a colour, pick a circuit, then play) and a ``launch``
function. The steps are screens registered in ``hub.STEPS``; adding a mode, or a step to a
mode, is one line in ``hub.ENTRIES`` / ``hub.STEPS`` and does not touch the flow logic.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pygame

from kartlab.render.topdown import ACCENT, TextCache
from kartlab.render.view import DIM, MONO, SANS, SANS_REGULAR, WHITE, scale_for, unit_for
from kartlab.show.overlays import ease_out
from kartlab.show.settings import CAMERAS, LIGHTS, SCALES
from kartlab.show.title import PixelText
from kartlab.show.training_job import ConfigInfo, TrainingError, TrainingJob, list_configs

if TYPE_CHECKING:
    from kartlab.show.hub import KartLabApp

DARK = (14, 14, 22)
WARN = (255, 128, 96)
OK = (140, 220, 140)
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


# -- entries and flows ---------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    """One line of the main menu."""

    key: str  # what ``--open`` takes
    label: str
    description: str
    group: str | None  # header it sits under; None: the closing lines (Opciones, Salir)
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


def wrap_text(font: pygame.font.Font, text: str, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split(" "):
            trial = f"{line} {word}".strip()
            if line and font.size(trial)[0] > width:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return lines


class Ui:
    """Fonts, sizes and cached panels for one window size."""

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.scale = scale_for(size)
        self.unit = u = unit_for(size)
        self.heading = PixelText(8 * max(1, round(self.scale * 0.75)))
        self.label = PixelText(8 * max(1, self.scale // 2))
        self.name = TextCache(pygame.font.Font(SANS, round(u * 1.3)))
        self.big = TextCache(pygame.font.Font(SANS, round(u * 2.0)))
        self.desc = TextCache(pygame.font.Font(SANS_REGULAR, round(u * 1.0)))
        self.small = TextCache(pygame.font.Font(SANS_REGULAR, max(round(u * 0.85), 10)))
        self.mono = TextCache(pygame.font.Font(MONO, max(round(u * 0.85), 10)))
        self._panels: dict[tuple, pygame.Surface] = {}
        self._veil = self._build_veil(size)

    @staticmethod
    def _build_veil(size: tuple[int, int]) -> pygame.Surface:
        """Dark all over (the text must read), a little lighter in the middle."""
        w, h = size
        veil = pygame.Surface((1, h), pygame.SRCALPHA)
        for y in range(h):
            edge = abs(y / max(h - 1, 1) - 0.5) * 2
            veil.set_at((0, y), (6, 8, 16, round(120 + 90 * edge**2)))
        return pygame.transform.scale(veil, (w, h))

    def draw_veil(self, screen: pygame.Surface, extra: int = 0) -> None:
        screen.blit(self._veil, (0, 0))
        if extra:
            screen.blit(self.panel(*self.size, extra, edge=False), (0, 0))

    def panel(
        self, w: int, h: int, alpha: int = 200, accent: bool = False, edge: bool = True
    ) -> pygame.Surface:
        key = (w, h, alpha, accent, edge)
        panel = self._panels.get(key)
        if panel is None:
            panel = pygame.Surface((max(w, 1), max(h, 1)), pygame.SRCALPHA)
            panel.fill((*DARK, alpha))
            if accent:
                pygame.draw.rect(panel, ACCENT, (0, 0, w, max(self.unit // 6, 2)))
            elif edge:
                pygame.draw.rect(panel, (255, 255, 255, 60), (0, 0, w, h), 1)
            self._panels[key] = panel
        return panel

    def heading_text(self, screen: pygame.Surface, text: str, y: int, age: float = 9.0) -> int:
        """The screen's title in the pixel font, centred, with a hard shadow. Returns its bottom."""
        width = screen.get_width()
        shadow = self.heading.get(text, (6, 8, 16))
        face = self.heading.get(text, ACCENT)
        slide = round((1.0 - ease_out(age / 0.35)) * -self.unit * 2)
        x = (width - face.get_width()) // 2
        off = max(self.scale // 2, 1) * 2
        screen.blit(shadow, (x + off, y + off + slide))
        screen.blit(face, (x, y + slide))
        return y + face.get_height() + off

    def centered(self, screen, surface, y: int, x: int | None = None) -> None:
        cx = screen.get_width() // 2 if x is None else x
        screen.blit(surface, (cx - surface.get_width() // 2, y))


class Screen:
    """One screen of the menu. The hub calls these hooks; the screen draws over the veil."""

    hint = ""
    dim = 0  # extra darkness (0-255) over the backdrop, for screens with a lot to read
    show_status = True  # the training line («Entrenando X · N/M · 3:12») in the top corner

    def __init__(self, hub: KartLabApp) -> None:
        self.hub = hub
        self.opened_at = hub.t

    def enter(self) -> None:
        self.opened_at = self.hub.t
        self.layout()

    def layout(self) -> None:
        """(Re)compute everything that depends on the window size."""

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

    def draw(self, screen: pygame.Surface) -> None:
        pass

    @property
    def age(self) -> float:
        return self.hub.t - self.opened_at

    def draw_hint(self, screen: pygame.Surface) -> None:
        ui = self.hub.ui
        hint = ui.small.get(self.hint, DIM)
        screen.blit(
            hint,
            (
                (screen.get_width() - hint.get_width()) // 2,
                screen.get_height() - hint.get_height() - ui.unit // 2,
            ),
        )


# -- dialogs --------------------------------------------------------------------------------


class Dialog:
    """A message (any key closes it), a question (Intro = yes, Esc = no) or a choice between
    several buttons (``choices``: label and what to do; ← → select, Intro accepts, Esc closes
    without doing anything) over everything."""

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
            "← → elegir · Intro aceptar · Esc cancelar"
            if self.choices
            else "Intro = sí · Esc = no"
            if on_yes is not None
            else "Pulsa una tecla para seguir"
        )
        self.buttons: list[pygame.Rect] = []
        self._surface: pygame.Surface | None = None
        self._size: tuple[int, int] | None = None

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
        self._surface = None  # the highlight moved

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
                self._surface = None

    def on_click(self, hub: KartLabApp, pos) -> None:
        if self.choices:
            for i, rect in enumerate(self.buttons):
                if rect.collidepoint(pos):
                    self.choose(hub, i)
        elif self.on_yes is None:
            self.close_with(hub, False)

    def draw(self, screen: pygame.Surface, ui: Ui) -> None:
        if self._surface is None or self._size != screen.get_size():
            self._surface, self._size = self._build(screen.get_size(), ui), screen.get_size()
        screen.blit(self._surface, (0, 0))

    def _build(self, size: tuple[int, int], ui: Ui) -> pygame.Surface:
        u = ui.unit
        body = pygame.font.Font(SANS_REGULAR, u)
        longest = max(body.size(line)[0] for line in self.text.split("\n"))
        labels = [ui.name.get(label, DARK) for label, _ in self.choices]
        buttons_w = sum(t.get_width() + 3 * u for t in labels) + len(labels) * u // 2 + 3 * u
        width = min(max(longest + 3 * u, u * 22, buttons_w), size[0] - 2 * u, u * 60)
        lines = wrap_text(body, self.text, width - 3 * u)
        line_h = body.get_height()
        heading = ui.name.get(self.title, ACCENT)
        foot = ui.small.get(self.footer, DIM)
        button_h = round(u * 2.2) if self.choices else 0
        height = u * 2 + heading.get_height() + u + len(lines) * line_h + u + foot.get_height() + u
        height += button_h + (u if self.choices else 0)
        height = min(height, size[1] - 2 * u)
        surface = pygame.Surface(size, pygame.SRCALPHA)
        surface.fill((6, 8, 16, 170))
        box = pygame.Rect(0, 0, width, height)
        box.center = (size[0] // 2, size[1] // 2)
        surface.blit(ui.panel(box.w, box.h, 245, accent=True), box.topleft)
        y = box.y + u * 2
        surface.blit(heading, (box.x + 1.5 * u, y))
        y += heading.get_height() + u
        for line in lines:
            surface.blit(body.render(line, True, WHITE), (box.x + 1.5 * u, y))
            y += line_h
        if self.choices:
            x = box.x + 1.5 * u
            self.buttons = []
            for i, text in enumerate(labels):
                rect = pygame.Rect(x, y + u // 2, text.get_width() + 3 * u, button_h)
                self.buttons.append(rect)
                chosen = i == self.selected
                pygame.draw.rect(surface, ACCENT if chosen else (40, 42, 58), rect)
                label = ui.name.get(self.choices[i][0], DARK if chosen else WHITE)
                surface.blit(label, label.get_rect(center=rect.center))
                x += rect.w + u // 2
        surface.blit(foot, (box.x + 1.5 * u, box.bottom - foot.get_height() - u))
        return surface


# -- the main list ---------------------------------------------------------------------------


class MenuScreen(Screen):
    """The title and the list of entries, grouped under headers."""

    hint = "↑ ↓ elegir · Intro aceptar · o con el ratón · F11 pantalla completa · Esc salir"

    def __init__(self, hub: KartLabApp, entries: list[Entry]) -> None:
        super().__init__(hub)
        self.entries = entries
        self.selected = 0
        self.esc_until = -1.0  # a second Esc before this time leaves
        self._rows: list[tuple[str, Entry | str, pygame.Rect]] = []
        self.rects: dict[int, pygame.Rect] = {}  # entry index -> its row
        self._mark: pygame.Surface | None = None
        self._strip: pygame.Surface | None = None

    # -- layout --------------------------------------------------------------------------

    def layout(self) -> None:
        hub = self.hub
        ui = hub.ui
        w, h = ui.size
        u = ui.unit
        mark = hub.wordmark
        k = max(1, min(ui.scale, int(w * 0.40) // mark.get_width()))
        self.k = k
        self._mark = pygame.transform.scale(mark, (mark.get_width() * k, mark.get_height() * k))
        self._mark_y = round(h * 0.05)
        top = self._mark_y + self._mark.get_height() + round(u * 2.2)
        reserved = round(u * 6.2)  # description box, status line and key hints
        available = h - top - reserved
        headers = len({e.group for e in self.entries if e.group})
        gap = round(u * 0.8) if any(e.group is None for e in self.entries) else 0
        row = u * 1.9
        head = u * 1.6
        need = len(self.entries) * row + headers * head + gap
        f = min(1.0, available / need)
        self.row_h = max(round(row * f), 12)
        head_h = max(round(head * f), 10)
        self.f = f
        self.name = TextCache(pygame.font.Font(SANS, max(round(u * 1.3 * f), 10)))
        self.group = PixelText(8 * max(1, int(ui.scale * f) // 2))
        self.width = min(round(w * 0.5), u * 30)
        x = (w - self.width) // 2
        y = top
        self._rows = []
        self.rects = {}
        last_group: str | None = "__none__"
        for index, entry in enumerate(self.entries):
            if entry.group != last_group:
                if entry.group is None:
                    y += gap
                else:
                    self._rows.append(
                        ("header", entry.group, pygame.Rect(x, y, self.width, head_h))
                    )
                    y += head_h
                last_group = entry.group
            rect = pygame.Rect(x, y, self.width, self.row_h - max(u // 8, 2))
            self._rows.append(("entry", entry, rect))
            self.rects[index] = rect
            y += self.row_h
        self.list_bottom = y
        self.desc_y = h - round(u * 6.0)
        self._chip = ui.panel(self.width, self.row_h - max(u // 8, 2), 175)
        sel = pygame.Surface(self._chip.get_size(), pygame.SRCALPHA)
        sel.fill((*ACCENT, 255))
        pygame.draw.rect(sel, (255, 255, 255, 255), (0, 0, sel.get_width(), max(u // 8, 2)))
        self._selected_chip = sel

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

    def draw(self, screen: pygame.Surface) -> None:
        hub = self.hub
        ui = hub.ui
        w, h = ui.size
        u = ui.unit
        age = self.age
        mark = self._mark
        drop = (1.0 - ease_out(age / 0.7)) * -h * 0.2
        bob = round(math.sin(hub.t * 1.7)) * self.k
        screen.blit(mark, ((w - mark.get_width()) // 2, round(self._mark_y + drop) + bob))
        appear = ease_out((age - 0.25) / 0.45)
        slide = round((1.0 - appear) * u)
        for kind, value, rect in self._rows:
            if kind == "header":
                text = self.group.get(value, ACCENT)
                screen.blit(text, (rect.x + u // 2, rect.bottom - text.get_height() + slide))
                continue
            index = self.entries.index(value)
            chosen = index == self.selected
            disabled = hub.disabled_reason(value) is not None
            moved = rect.move(0, slide)
            screen.blit(self._selected_chip if chosen else self._chip, moved.topleft)
            colour = DARK if chosen else (DIM if disabled else WHITE)
            label = self.name.get(value.label, colour)
            screen.blit(label, (moved.x + u, moved.centery - label.get_height() // 2))
            if disabled and not chosen:
                reason = ui.small.get("sin runs", DIM)
                screen.blit(
                    reason,
                    (
                        moved.right - reason.get_width() - u,
                        moved.centery - reason.get_height() // 2,
                    ),
                )
        self._draw_description(screen)
        self._draw_esc_notice(screen)
        self.draw_hint(screen)

    def _draw_description(self, screen: pygame.Surface) -> None:
        hub = self.hub
        ui = hub.ui
        u = ui.unit
        entry = self.entries[self.selected]
        reason = hub.disabled_reason(entry)
        text = reason or entry.description
        surface = ui.desc.get(text, WARN if reason else WHITE)
        box_w = surface.get_width() + 2 * u
        box_h = surface.get_height() + u
        x = (ui.size[0] - box_w) // 2
        screen.blit(ui.panel(box_w, box_h, 190), (x, self.desc_y))
        screen.blit(surface, (x + u, self.desc_y + u // 2))

    def _draw_esc_notice(self, screen: pygame.Surface) -> None:
        """«Pulsa Esc otra vez para salir» after the first Esc, for a couple of seconds."""
        hub = self.hub
        if hub.t >= self.esc_until:
            return
        ui = hub.ui
        surface = ui.desc.get("Pulsa Esc otra vez para salir", DARK)
        w, h = surface.get_width() + 2 * ui.unit, surface.get_height() + ui.unit
        x = (ui.size[0] - w) // 2
        y = self.desc_y + round(ui.unit * 2.5)
        pygame.draw.rect(screen, ACCENT, (x, y, w, h))
        screen.blit(surface, (x + ui.unit, y + ui.unit // 2))


# -- options ---------------------------------------------------------------------------------


class OptionsScreen(Screen):
    dim = 90
    hint = "↑ ↓ elegir · ← → cambiar · Esc volver"

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
        run = s.last_run if s.last_run and s.last_run in runs else "la más reciente"
        rows = [
            ("Tamaño de la ventana", scale, cycle_scale),
            ("Pantalla completa", "Sí" if s.fullscreen else "No", cycle_fullscreen),
            ("Luz", LIGHT_NAMES[s.light], cycle_light),
            ("Cámara", CAMERA_NAMES[s.camera], cycle_camera),
            ("IA que se muestra", run if runs else "(no hay runs)", cycle_run),
        ]
        return rows

    def change(self, delta: int) -> None:
        self.rows()[self.selected][2](delta)
        self.hub.save_settings()

    def layout(self) -> None:
        ui = self.hub.ui
        w, h = ui.size
        u = ui.unit
        width = min(round(w * 0.62), u * 40)
        self.rect = pygame.Rect((w - width) // 2, round(h * 0.26), width, 0)
        self.row_h = round(u * 2.3)
        self.rects = [
            pygame.Rect(self.rect.x, self.rect.y + i * self.row_h, width, self.row_h - u // 4)
            for i in range(5)
        ]

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
        row = self._row_at(pos)
        if row is not None:
            self.selected = row

    def on_click(self, pos) -> None:
        row = self._row_at(pos)
        if row is not None:
            self.selected = row
            # the left third goes back, the rest forward
            self.change(-1 if pos[0] < self.rects[row].x + self.rects[row].w // 3 else 1)

    def on_wheel(self, dy: int) -> None:
        self.change(1 if dy > 0 else -1)

    def draw(self, screen: pygame.Surface) -> None:
        ui = self.hub.ui
        u = ui.unit
        ui.heading_text(screen, "OPCIONES", round(ui.size[1] * 0.08), self.age)
        for i, (label, value, _) in enumerate(self.rows()):
            rect = self.rects[i]
            chosen = i == self.selected
            if chosen:
                pygame.draw.rect(screen, ACCENT, rect)
                pygame.draw.rect(screen, (255, 255, 255), (rect.x, rect.y, rect.w, max(u // 8, 2)))
            else:
                screen.blit(ui.panel(rect.w, rect.h, 175), rect.topleft)
            name = ui.name.get(label, DARK if chosen else WHITE)
            screen.blit(name, (rect.x + u, rect.centery - name.get_height() // 2))
            text = ui.name.get(f"←  {value}  →" if chosen else value, DARK if chosen else ACCENT)
            screen.blit(
                text, (rect.right - text.get_width() - u, rect.centery - text.get_height() // 2)
            )
        note = ui.small.get("Los cambios se guardan solos en settings.json", DIM)
        screen.blit(note, ((ui.size[0] - note.get_width()) // 2, self.rects[-1].bottom + u))
        self.draw_hint(screen)


# -- training --------------------------------------------------------------------------------


class TrainScreen(Screen):
    """Pick a config, launch it, watch it live, stop it. Nothing starts without a keypress."""

    dim = 90
    show_status = False  # it has its own panel
    hint = "↑ ↓ elegir · Intro lanzar · V ver en directo · P parar · O abrir carpeta · Esc volver"

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
            return "Ya hay un entrenamiento en marcha (lanzado desde otra parte)"
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
            launch_off = "Esta configuración no es válida"  # the box says why
        run = self.run_for_live()
        return {
            "launch": ("Lanzar  Intro", launch_off),
            "live": ("Ver en directo  V", None if run else "Aún no hay ningún run"),
            "stop": (
                "Parar  P",
                None if job is not None and job.running and not job.stopping else "Nada que parar",
            ),
            "folder": ("Abrir carpeta  O", None if run else "Aún no hay ningún run"),
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
                f"«{config.name}» durará más de {30} min ({config.estimate}).\n¿Lanzar?",
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
            self.hub.message(
                "Nada que parar", "No hay ningún entrenamiento lanzado desde este menú."
            )
            return
        job.request_stop()

    def live(self) -> None:
        run = self.run_for_live()
        if run is None:
            self.hub.message("Aún no hay ningún run", "Lanza un entrenamiento primero.")
            return
        self.hub.open_live(run)

    def folder(self) -> None:
        run = self.run_for_live()
        if run is None:
            self.hub.message("Aún no hay ningún run", "Lanza un entrenamiento primero.")
            return
        self.hub.open_folder(run.root)

    # -- layout and input ----------------------------------------------------------------

    def layout(self) -> None:
        ui = self.hub.ui
        w, h = ui.size
        u = ui.unit
        self.width = min(round(w * 0.72), u * 52)
        self.x = (w - self.width) // 2
        top = round(h * 0.2)
        self.row_h = round(u * 3.4)
        panel_h, button_h = round(u * 6.5), round(u * 2.4)
        below = panel_h + u + u // 2 + button_h + round(u * 3.2)  # panel, buttons, hover note, hint
        self.visible = max(1, min(len(self.configs) or 1, (h - top - below) // self.row_h))
        self._laid_out_for = len(self.configs)
        self.top = top
        self._place()
        rows_bottom = top + self.visible * self.row_h
        self.panel_rect = pygame.Rect(self.x, rows_bottom + u, self.width, panel_h)
        names = ("launch", "live", "stop", "folder")
        gap = u // 2
        bw = (self.width - gap * (len(names) - 1)) // len(names)
        by = self.panel_rect.bottom + u // 2
        self.buttons = {
            name: pygame.Rect(self.x + i * (bw + gap), by, bw, button_h)
            for i, name in enumerate(names)
        }

    def _place(self) -> None:
        """Rows of the configs in view; the selected one always is."""
        u = self.hub.ui.unit
        last = max(len(self.configs) - self.visible, 0)
        if self.selected < self.scroll:
            self.scroll = self.selected
        elif self.selected >= self.scroll + self.visible:
            self.scroll = self.selected - self.visible + 1
        self.scroll = min(max(self.scroll, 0), last)
        shown = range(self.scroll, min(self.scroll + self.visible, len(self.configs)))
        self.rects = {
            i: pygame.Rect(self.x, self.top + n * self.row_h, self.width, self.row_h - u // 4)
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

    def draw(self, screen: pygame.Surface) -> None:
        hub = self.hub
        ui = hub.ui
        u = ui.unit
        ui.heading_text(screen, "ENTRENAR A LA IA", round(ui.size[1] * 0.07), self.age)
        if not self.configs:
            text = ui.desc.get("No hay ningún fichero en configs/", WARN)
            screen.blit(text, ((ui.size[0] - text.get_width()) // 2, self.top))
        for i, rect in self.rects.items():
            config = self.configs[i]
            chosen = i == self.selected
            if chosen:
                pygame.draw.rect(screen, ACCENT, rect)
                pygame.draw.rect(screen, (255, 255, 255), (rect.x, rect.y, rect.w, max(u // 8, 2)))
            else:
                screen.blit(ui.panel(rect.w, rect.h, 175), rect.topleft)
            ink = DARK if chosen else WHITE
            sub = DARK if chosen else DIM
            if config.error:
                title = ui.name.get(f"{config.path.name}  (no válido)", DARK if chosen else WARN)
                reason = config.error.splitlines()[-1]
                detail = ui.small.get(self._fit(ui.small, reason, rect.w - 2 * u), sub)
            else:
                title = ui.name.get(config.name, ink)
                steps = f"{config.total_steps:,}".replace(",", ".")
                detail_text = f"{steps} pasos · {config.estimate}"
                if config.comment:
                    detail_text += f" · {config.comment}"
                detail = ui.small.get(self._fit(ui.small, detail_text, rect.w - 2 * u), sub)
            screen.blit(title, (rect.x + u, rect.y + u // 3))
            screen.blit(detail, (rect.x + u, rect.bottom - detail.get_height() - u // 3))
        if len(self.configs) > self.visible:  # more configs than rows: where we are
            more = ui.small.get(
                f"{self.scroll + 1}-{self.scroll + len(self.rects)} de {len(self.configs)}  ↑ ↓",
                DIM,
            )
            screen.blit(
                more, (self.x + self.width - more.get_width(), self.top - more.get_height())
            )
        self._draw_job_panel(screen)
        self._draw_buttons(screen)
        self.draw_hint(screen)

    @staticmethod
    def _fit(cache: TextCache, text: str, width: int) -> str:
        while cache.font.size(text)[0] > width and len(text) > 4:
            text = text[:-2].rstrip() + "…"
        return text

    def _draw_job_panel(self, screen: pygame.Surface) -> None:
        hub = self.hub
        ui = hub.ui
        u = ui.unit
        rect = self.panel_rect
        screen.blit(ui.panel(rect.w, rect.h, 200, accent=True), rect.topleft)
        status = hub.status_line()
        lines = []
        if status:
            lines.append((status, ACCENT, ui.name))
        else:
            lines.append(("No hay ningún entrenamiento en marcha", DIM, ui.name))
        job = hub.job
        if job is not None:
            for line in job.log_tail(3):
                lines.append((self._fit(ui.mono, line, rect.w - 2 * u), DIM, ui.mono))
        y = rect.y + u
        for text, colour, cache in lines:
            surface = cache.get(text, colour)
            screen.blit(surface, (rect.x + u, y))
            y += surface.get_height() + u // 4

    def _draw_buttons(self, screen: pygame.Surface) -> None:
        ui = self.hub.ui
        u = ui.unit
        for name, (label, reason) in self.button_states().items():
            rect = self.buttons[name]
            off = reason is not None
            screen.blit(ui.panel(rect.w, rect.h, 150 if off else 215), rect.topleft)
            if not off:
                pygame.draw.rect(
                    screen, ACCENT, (rect.x, rect.bottom - max(u // 6, 2), rect.w, max(u // 6, 2))
                )
            text = ui.small.get(label, DIM if off else WHITE)
            screen.blit(
                text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2)
            )
            if off and reason and self._hover(rect):
                note = ui.small.get(reason, WARN)
                screen.blit(note, (rect.x, rect.bottom + u // 4))

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
