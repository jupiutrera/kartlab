"""The options screen of "Carrera contra la IA": difficulty, rivals, laps and bumps.

It sits between the circuit picker and the race. What is chosen here is remembered in
``settings.json`` (the ``race_*`` keys) and handed to the race as ``choices["race"]``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pygame

from kartlab.render.topdown import ACCENT
from kartlab.render.view import DIM, WHITE
from kartlab.show.humanrace import DIFFICULTIES, difficulty_of
from kartlab.show.menu import CONFIRM_KEYS, DARK, Flow, Screen
from kartlab.show.settings import RACE_LAPS, RACE_RIVALS

if TYPE_CHECKING:
    from kartlab.show.hub import KartLabApp

START_ROW = 4  # the "¡A correr!" button under the four values


def race_choices(settings) -> dict:
    """The race options as the launcher takes them."""
    return {
        "difficulty": settings.race_difficulty,
        "rivals": settings.race_rivals,
        "laps": settings.race_laps,
        "collisions": settings.race_collisions,
    }


class RaceOptionsScreen(Screen):
    dim = 90
    hint = "↑ ↓ elegir · ← → cambiar · Intro empezar · Esc volver"

    def __init__(self, hub: KartLabApp, flow: Flow) -> None:
        super().__init__(hub)
        self.flow = flow
        self.selected = START_ROW  # Intro right away races with what was chosen last time
        self.rects: list[pygame.Rect] = []

    # Each row: (label, value text, change(direction)); the last one is the start button.
    def rows(self) -> list[tuple[str, str, Callable[[int], None]]]:
        s = self.hub.settings
        keys = [level.key for level in DIFFICULTIES]

        def cycle_difficulty(d: int) -> None:
            s.race_difficulty = keys[(keys.index(s.race_difficulty) + d) % len(keys)]

        def cycle_rivals(d: int) -> None:
            s.race_rivals = RACE_RIVALS[(RACE_RIVALS.index(s.race_rivals) + d) % len(RACE_RIVALS)]

        def cycle_laps(d: int) -> None:
            s.race_laps = RACE_LAPS[(RACE_LAPS.index(s.race_laps) + d) % len(RACE_LAPS)]

        def toggle_collisions(d: int) -> None:
            s.race_collisions = not s.race_collisions

        return [
            ("Dificultad", difficulty_of(s.race_difficulty).label, cycle_difficulty),
            ("Rivales", str(s.race_rivals), cycle_rivals),
            ("Vueltas", str(s.race_laps), cycle_laps),
            ("Choques", "Sí, suaves" if s.race_collisions else "No", toggle_collisions),
            ("¡A correr!", "", lambda d: None),
        ]

    def change(self, delta: int) -> None:
        if self.selected < START_ROW:
            self.rows()[self.selected][2](delta)
            self.hub.save_settings()

    def start(self) -> None:
        self.hub.save_settings()
        self.hub.advance(race_choices(self.hub.settings))

    def layout(self) -> None:
        ui = self.hub.ui
        w, h = ui.size
        u = ui.unit
        width = min(round(w * 0.62), u * 40)
        self.row_h = round(u * 2.3)
        x, y = (w - width) // 2, round(h * 0.26)
        self.rects = [
            pygame.Rect(x, y + i * self.row_h + (u if i == START_ROW else 0), width, self.row_h)
            for i in range(START_ROW + 1)
        ]
        for rect in self.rects:
            rect.height -= u // 4

    def on_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.hub.back()
        elif key in (pygame.K_UP, pygame.K_w):
            self.selected = (self.selected - 1) % len(self.rects)
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.selected = (self.selected + 1) % len(self.rects)
        elif key in (pygame.K_LEFT, pygame.K_a):
            self.change(-1)
        elif key in (pygame.K_RIGHT, pygame.K_d):
            self.change(1)
        elif key in CONFIRM_KEYS:
            self.start()

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
        if row is None:
            return
        self.selected = row
        if row == START_ROW:
            self.start()
        else:  # the left third goes back, the rest forward
            rect = self.rects[row]
            self.change(-1 if pos[0] < rect.x + rect.w // 3 else 1)

    def on_wheel(self, dy: int) -> None:
        self.change(1 if dy > 0 else -1)

    def draw(self, screen: pygame.Surface) -> None:
        ui = self.hub.ui
        u = ui.unit
        ui.heading_text(screen, "CARRERA CONTRA LA IA", round(ui.size[1] * 0.08), self.age)
        for i, (label, value, _) in enumerate(self.rows()):
            rect = self.rects[i]
            chosen = i == self.selected
            if chosen:
                pygame.draw.rect(screen, ACCENT, rect)
                pygame.draw.rect(screen, (255, 255, 255), (rect.x, rect.y, rect.w, max(u // 8, 2)))
            else:
                screen.blit(ui.panel(rect.w, rect.h, 175), rect.topleft)
            name = ui.name.get(label, DARK if chosen else WHITE)
            if i == START_ROW:
                screen.blit(
                    name,
                    (rect.centerx - name.get_width() // 2, rect.centery - name.get_height() // 2),
                )
                continue
            screen.blit(name, (rect.x + u, rect.centery - name.get_height() // 2))
            text = ui.name.get(f"←  {value}  →" if chosen else value, DARK if chosen else ACCENT)
            screen.blit(
                text, (rect.right - text.get_width() - u, rect.centery - text.get_height() // 2)
            )
        run = self.hub.opponent_run()
        rival = run.root.name if run is not None else "(sin IA)"
        s = self.hub.settings
        notes = [
            f"Rival: {rival}",
            f"{s.race_laps} vueltas con cuenta atrás. La IA no ve karts: choques suaves.",
        ]
        y = self.rects[-1].bottom + u
        for note in notes:
            text = ui.small.get(note, DIM)
            screen.blit(text, ((ui.size[0] - text.get_width()) // 2, y))
            y += text.get_height()
        self.draw_hint(screen)
