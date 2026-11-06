"""Panels, cards and counters drawn over a ``GameView`` frame.

Shared by the offline videos (1080p, written to ffmpeg) and the exhibition viewer (window
resolution). Everything is drawn at screen resolution, sharp, and cached: text goes
through ``TextCache`` and panels are built once per size.
"""

from __future__ import annotations

import math

import pygame

from kartlab.render.palette import KART_RAMPS, game_palette
from kartlab.render.scene import KartVisual
from kartlab.render.topdown import ACCENT, TextCache
from kartlab.render.view import DIM, MONO, PANEL, SANS, SANS_REGULAR, WHITE, GameView
from kartlab.show.drivers import PolicyDriver, format_steps

CARD_SECONDS = 2.0  # how long a card stays up by default


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
    return f"{steps:,}".replace(",", ".")


def visual_of(driver: PolicyDriver, ramp: int, label=None, ghost=False) -> KartVisual:
    racer = driver.racer
    return KartVisual(
        racer.kart, ramp, label, ghost, racer.on_grass, driver.throttle, list(driver.trail)
    )


def ramp_color(ramp: int) -> tuple[int, int, int]:
    rgb = game_palette().rgb[KART_RAMPS[ramp % len(KART_RAMPS)][1]]
    return tuple(int(c) for c in rgb)


def u_off(unit: int) -> int:
    return max(unit // 6, 3)


class Overlays:
    """Drawing helpers over ``self.screen``; ``self.game`` provides fonts and the unit size."""

    def __init__(self, screen: pygame.Surface, game: GameView) -> None:
        pygame.font.init()
        self.screen = screen
        self.game = game
        self._panels: dict[tuple[int, int], pygame.Surface] = {}
        self._black = pygame.Surface(screen.get_size())
        self._black.fill((0, 0, 0))
        self.setup_fonts()

    @property
    def size(self) -> tuple[int, int]:
        return self.screen.get_size()

    @property
    def unit(self) -> int:
        return self.game.unit

    def setup_fonts(self) -> None:
        """(Re)build the text caches; call after the window or the view changes size."""
        unit = self.game.unit
        self.kicker = TextCache(pygame.font.Font(SANS, round(unit * 1.1)))
        self.title = TextCache(pygame.font.Font(SANS, round(unit * 3.4)))
        self.body = TextCache(pygame.font.Font(SANS_REGULAR, round(unit * 1.2)))
        self.countdown_font = TextCache(pygame.font.Font(MONO, unit * 8))

    def set_screen(self, screen: pygame.Surface, game: GameView | None = None) -> None:
        self.screen = screen
        if game is not None:
            self.game = game
        self._panels.clear()
        self._black = pygame.Surface(screen.get_size())
        self._black.fill((0, 0, 0))
        self.setup_fonts()

    # -- building blocks --------------------------------------------------------------------

    def panel(self, w: int, h: int) -> pygame.Surface:
        panel = self._panels.get((w, h))
        if panel is None:
            panel = pygame.Surface((w, h), pygame.SRCALPHA)
            panel.fill(PANEL)
            self._panels[(w, h)] = panel
        return panel

    def dim(self, amount: float) -> None:
        """Cover the frame with black: 0 leaves it as it is, 1 is fully black."""
        if amount > 0.003:
            self._black.set_alpha(round(255 * min(amount, 1.0)))
            self.screen.blit(self._black, (0, 0))

    def card(
        self,
        kicker: str,
        title: str,
        lines: list[str] = (),
        age: float = 0.0,
        life: float = CARD_SECONDS,
    ) -> None:
        """A centred title card that slides up and fades in/out with easing."""
        fade_in, fade_out = 0.45, 0.4
        alpha = ease_out(age / fade_in) * min(1.0, max(0.0, (life - age) / fade_out))
        if alpha <= 0.003:
            return
        u = self.unit
        texts = [
            self.kicker.get(kicker.upper(), ACCENT),
            self.title.get(title, WHITE),
            *(self.body.get(line, DIM) for line in lines),
        ]
        pad = round(u * 1.4)
        w = max(t.get_width() for t in texts) + 2 * pad + u // 2
        h = sum(t.get_height() for t in texts) + 2 * pad
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        card.fill((14, 14, 22, 235))
        pygame.draw.rect(card, ACCENT, (0, 0, max(u // 4, 4), h))
        y = pad
        for text in texts:
            card.blit(text, (pad + u // 2, y))
            y += text.get_height()
        card.set_alpha(round(255 * alpha))
        slide = round((1.0 - ease_out(age / fade_in)) * u * 2)
        cx, cy = self.size[0] // 2, round(self.size[1] * 0.42)
        self.screen.blit(card, (cx - w // 2, cy - h // 2 + slide))

    def big_number(self, text: str, alpha: float) -> None:
        label = self.countdown_font.get(text, WHITE)
        shade = self.countdown_font.get(text, (14, 14, 22))
        for surface, offset in ((shade, u_off(self.unit)), (label, 0)):
            surface.set_alpha(round(255 * alpha))
            rect = surface.get_rect(
                center=(self.size[0] // 2 + offset, self.size[1] * 0.4 + offset)
            )
            self.screen.blit(surface, rect)

    def counter(self, steps: int, subtitle: str) -> None:
        """Top-centre panel with the (rolling) training step counter."""
        u = self.unit
        number = self.game.big.get(count_steps(steps), WHITE)
        sub = self.game.small.get(subtitle, DIM)
        w = max(number.get_width(), sub.get_width()) + 3 * u
        h = number.get_height() + sub.get_height() + u // 2
        x, y = (self.size[0] - w) // 2, u // 2
        self.screen.blit(self.panel(w, h), (x, y))
        self.screen.blit(number, (x + (w - number.get_width()) // 2, y + u // 5))
        self.screen.blit(sub, (x + (w - sub.get_width()) // 2, y + u // 5 + number.get_height()))

    def standings(
        self, rows: list[tuple[int, str, str]], at: tuple[int, int], focus: int | None = None
    ) -> None:
        """Ranking panel: rows of (ramp, label, note), first row first. ``focus`` is the
        index of the row to highlight (the kart the camera follows)."""
        u = self.unit
        pad = u // 2
        texts = [
            (
                ramp,
                self.game.text.get(label, ACCENT if i == focus else WHITE),
                self.game.small.get(note, DIM),
            )
            for i, (ramp, label, note) in enumerate(rows)
        ]
        line = max(t[1].get_height() for t in texts)
        w = max(t[1].get_width() + t[2].get_width() for t in texts) + 5 * u
        h = line * len(texts) + pad * 2
        self.screen.blit(self.panel(w, h), at)
        for i, (ramp, label, note) in enumerate(texts):
            y = at[1] + pad + i * line
            self.screen.blit(self.game.small.get(str(i + 1), DIM), (at[0] + pad, y + u // 5))
            cx, cy = at[0] + pad + round(u * 1.4), y + line // 2
            pygame.draw.circle(self.screen, (10, 10, 16), (cx, cy), u // 3 + 1)
            pygame.draw.circle(self.screen, ramp_color(ramp), (cx, cy), u // 3)
            self.screen.blit(label, (at[0] + pad + round(u * 2.2), y))
            self.screen.blit(note, (at[0] + w - pad - note.get_width(), y + u // 5))

    def chart_panel(self, chart, metrics, marker: int) -> None:
        """The reward chart in a panel, centred on the right edge."""
        surface = chart.draw(metrics.rows, metrics.version, marker)
        pad = self.unit // 2
        w, h = surface.get_size()
        x = self.size[0] - pad - w - 2 * pad
        y = (self.size[1] - h) // 2 - pad
        self.screen.blit(self.panel(w + 2 * pad, h + 2 * pad), (x, y))
        self.screen.blit(surface, (x + pad, y + pad))

    def timeline(self, names: list[str], current: int) -> None:
        """Bottom strip with every version on the way; the current one is highlighted."""
        u = self.unit
        pad = u // 2
        items = []
        for i, name in enumerate(names):
            colour = ACCENT if i == current else (WHITE if i < current else DIM)
            items.append(
                self.game.text.get(name, colour)
                if i == current
                else self.game.small.get(name, colour)
            )
        gap = u
        w = sum(t.get_width() for t in items) + gap * (len(items) - 1) + 2 * pad
        h = max(t.get_height() for t in items) + pad
        x = (self.size[0] - w) // 2
        y = self.size[1] - pad - h
        self.screen.blit(self.panel(w, h), (x, y))
        cursor = x + pad
        for i, text in enumerate(items):
            if i == current:
                pygame.draw.rect(
                    self.screen, ACCENT, (cursor, y + h - u // 6 - 2, text.get_width(), u // 6)
                )
            self.screen.blit(text, (cursor, y + (h - text.get_height()) // 2 - u // 10))
            cursor += text.get_width() + gap

    def countdown(self, frame: int, start_frames: int, fps: int) -> None:
        """3, 2, 1 before ``start_frames`` and a fading "¡YA!" right after."""
        if frame >= start_frames + round(0.8 * fps):
            return
        if frame < start_frames:
            remaining = (start_frames - frame) / fps
            text, phase = str(math.ceil(remaining)), remaining % 1.0
            alpha = ease_out(1.0 - phase * 0.6) if phase else 1.0
        else:
            age = (frame - start_frames) / fps
            text, alpha = "¡YA!", 1.0 - ease_in_out(age / 0.8)
        self.big_number(text, min(alpha, 1.0))
