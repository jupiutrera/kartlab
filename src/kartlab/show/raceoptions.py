"""The options screen of "Carrera contra la IA": difficulty, rivals, laps and bumps.

It sits between the circuit picker and the race. What is chosen here is remembered in
``settings.json`` (the ``race_*`` keys) and handed to the race as ``choices["race"]``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pygame

from kartlab.show.humanrace import DIFFICULTIES, difficulty_of
from kartlab.show.menu import CONFIRM_KEYS, Flow, Screen, W, button, heading, row
from kartlab.show.settings import RACE_LAPS, RACE_RIVALS

if TYPE_CHECKING:
    from kartlab.show.hub import KartLabApp

START_ROW = 4  # the "Empezar" button under the four values


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
    hint = "←→ cambiar, Intro empezar, Esc volver"

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
            ("Colisiones", "Sí" if s.race_collisions else "No", toggle_collisions),
            ("Empezar", "", lambda d: None),
        ]

    def change(self, delta: int) -> None:
        if self.selected < START_ROW:
            self.rows()[self.selected][2](delta)
            self.hub.save_settings()

    def start(self) -> None:
        self.hub.save_settings()
        self.hub.advance(race_choices(self.hub.settings))

    def layout(self) -> None:
        width = 260
        x = (W - width) // 2
        self.rects = [
            pygame.Rect(x, 34 + i * 16 + (6 if i == START_ROW else 0), width, 14)
            for i in range(START_ROW + 1)
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

    def draw(self, canvas: pygame.Surface) -> None:
        heading(canvas, "CARRERA CONTRA LA IA", self.age)
        for i, (label, value, _) in enumerate(self.rows()):
            rect = self.rects[i]
            chosen = i == self.selected
            if i == START_ROW:
                button(canvas, rect, label.upper(), chosen=chosen)
            else:
                row(canvas, rect, chosen, label, value, arrows=True)
        self.draw_hint(canvas)
