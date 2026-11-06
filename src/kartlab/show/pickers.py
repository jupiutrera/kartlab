"""The two pick-a-thing screens that come before a game: "ELIGE TU KART" and "ELIGE CIRCUITO".

Both are steps of an ``Entry`` (see ``menu.py``): the first stores the kart colour (an index
of ``KART_RAMPS``), the second the circuit (``harbor``, ``file:tracks/x.json``, ``random``,
``eval:7``...). The kart is drawn with the world's own ``SpriteBank`` (sprite stacking), at
the size it is shown, and then enlarged by the same whole number as the rest of the pixel
art; the circuit cards draw the centerline in small.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pygame

from kartlab.render.sprites import ELEVATIONS, SCALE_STEPS, bucket_scale, scale_bucket
from kartlab.render.topdown import ACCENT
from kartlab.render.view import DIM, WHITE
from kartlab.show.cli import KART_COLOR_NAMES, kart_color_rgb
from kartlab.show.menu import CONFIRM_KEYS, DARK, WARN, Flow, Screen
from kartlab.sim.showcase import SHOWCASE
from kartlab.sim.track import Track
from kartlab.sim.trackfile import TrackFileError, load_spec, load_track_file
from kartlab.sim.tracksets import EVAL_SEEDS, eval_track, train_track

if TYPE_CHECKING:
    from kartlab.show.hub import KartLabApp

PREVIEW_ELEVATION = ELEVATIONS[2]  # 35 degrees: the kart seen from above and behind
RANDOM_CACHE = 16  # random circuits kept (each "R" makes a new one)
SPIN_SECONDS = 7.0  # one turn of the kart on its turntable


# -- ELIGE TU KART ----------------------------------------------------------------------------


class KartScreen(Screen):
    dim = 110
    hide_backdrop_kart = True  # the autopilot's kart would get in the way of the big one
    hint = "← → o clic: elegir color · Intro, segundo clic o Aceptar: confirmar · Esc volver"

    def __init__(self, hub: KartLabApp, flow: Flow) -> None:
        super().__init__(hub)
        self.flow = flow
        self.selected = hub.settings.kart_ramp
        self._scaled: dict[int, tuple[pygame.Surface, pygame.Surface]] = {}
        self.swatches: list[pygame.Rect] = []

    def layout(self) -> None:
        ui = self.hub.ui
        w, h = ui.size
        u = ui.unit
        self._scaled.clear()
        self.k = ui.scale
        # Pixels per voxel so the kart is about 40 % of the height, never past the finest size.
        wanted = (h * 0.40) / (self.k * 30.6)
        self.ppv = bucket_scale(min(scale_bucket(wanted), SCALE_STEPS - 1))
        side = round(u * 2.6)
        gap = round(u * 0.7)
        total = 8 * side + 7 * gap
        x = (w - total) // 2
        y = round(h * 0.77)
        self.swatches = [pygame.Rect(x + i * (side + gap), y, side, side) for i in range(8)]
        self.accept = pygame.Rect(0, 0, round(u * 9), round(u * 2.3))
        self.accept.midtop = (w // 2, y + side + round(u * 0.9))
        self.ground = (w // 2, round(h * 0.53))
        self.arrow_left = pygame.Rect(0, 0, round(u * 4), round(u * 6))
        self.arrow_left.center = (round(w * 0.24), self.ground[1] - round(h * 0.08))
        self.arrow_right = self.arrow_left.copy()
        self.arrow_right.centerx = w - self.arrow_left.centerx

    # -- input ---------------------------------------------------------------------------

    def move(self, delta: int) -> None:
        self.selected = (self.selected + delta) % 8

    def confirm(self) -> None:
        hub = self.hub
        hub.settings.kart_ramp = self.selected
        hub.save_settings()
        hub.apply_kart()
        hub.advance(self.selected)

    def on_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.hub.back()
        elif key in (pygame.K_LEFT, pygame.K_a, pygame.K_UP, pygame.K_w):
            self.move(-1)
        elif key in (pygame.K_RIGHT, pygame.K_d, pygame.K_DOWN, pygame.K_s):
            self.move(1)
        elif key in CONFIRM_KEYS:
            self.confirm()

    def on_click(self, pos) -> None:
        """A click on a colour only selects it (the big kart turns that colour); a second click
        on the selected one, Intro or the Aceptar button confirm."""
        if self.arrow_left.collidepoint(pos):
            self.move(-1)
        elif self.arrow_right.collidepoint(pos):
            self.move(1)
        elif self.accept.collidepoint(pos):
            self.confirm()
        else:
            for i, rect in enumerate(self.swatches):
                if rect.collidepoint(pos):
                    if i == self.selected:
                        self.confirm()
                    else:
                        self.selected = i
                    return

    def on_wheel(self, dy: int) -> None:
        self.move(-1 if dy > 0 else 1)

    # -- drawing -------------------------------------------------------------------------

    def _kart_surface(self, angle: float) -> tuple[pygame.Surface, tuple[float, float]]:
        bank = self.hub.sprites
        surface, anchor = bank.kart(self.selected, angle, PREVIEW_ELEVATION, self.ppv)
        cached = self._scaled.get(id(surface))
        if cached is None:
            big = pygame.transform.scale(
                surface, (surface.get_width() * self.k, surface.get_height() * self.k)
            )
            cached = self._scaled[id(surface)] = (surface, big)  # keeps ``surface`` alive
        return cached[1], (anchor[0] * self.k, anchor[1] * self.k)

    def draw(self, screen: pygame.Surface) -> None:
        hub = self.hub
        ui = hub.ui
        w, h = ui.size
        u = ui.unit
        ui.heading_text(screen, "ELIGE TU KART", round(h * 0.06), self.age)

        # Turntable: a dark ellipse with an accent rim, and the kart spinning on it.
        cx, ground_y = self.ground
        plate_w = round(w * 0.30)
        plate = pygame.Rect(0, 0, plate_w, round(plate_w * 0.30))
        plate.center = (cx, ground_y)
        colour = kart_color_rgb(self.selected)
        pygame.draw.ellipse(screen, (10, 10, 18), plate.move(0, u // 2))
        pygame.draw.ellipse(screen, (28, 30, 44), plate)
        pygame.draw.ellipse(screen, colour, plate, max(u // 4, 2))

        hub.sprites.begin_frame()
        angle = hub.t * math.tau / SPIN_SECONDS
        sprite, (ax, ay) = self._kart_surface(angle)
        bob = round(math.sin(hub.t * 2.4) * u * 0.15)
        screen.blit(sprite, (cx - ax, ground_y - ay + bob))

        name = KART_COLOR_NAMES[self.selected].upper()
        label = ui.heading.get(name, colour)
        shadow = ui.heading.get(name, (6, 8, 16))
        ly = ground_y + plate.h // 2 + round(u * 1.2)
        screen.blit(shadow, (cx - label.get_width() // 2 + 3, ly + 3))
        screen.blit(label, (cx - label.get_width() // 2, ly))

        for rect, direction in ((self.arrow_left, -1), (self.arrow_right, 1)):
            half = rect.h // 5
            tip = rect.centerx + direction * half // 2
            back = rect.centerx - direction * half // 2
            top, bottom = rect.centery - half, rect.centery + half
            outline = [(tip, rect.centery), (back, top), (back, bottom)]
            pygame.draw.polygon(screen, ACCENT, outline)
            pygame.draw.polygon(screen, (255, 255, 255), outline, max(u // 6, 2))

        for i, rect in enumerate(self.swatches):
            chosen = i == self.selected
            if chosen:
                pygame.draw.rect(screen, ACCENT, rect.inflate(u, u))
            pygame.draw.rect(screen, (6, 8, 16), rect.inflate(u // 3, u // 3))
            pygame.draw.rect(screen, kart_color_rgb(i), rect)
            pygame.draw.rect(screen, (255, 255, 255), (rect.x, rect.y, rect.w, max(u // 8, 2)))
        pygame.draw.rect(screen, ACCENT, self.accept)
        pygame.draw.rect(
            screen, (255, 255, 255), (*self.accept.topleft, self.accept.w, max(u // 8, 2))
        )
        text = ui.name.get("Aceptar", DARK)
        screen.blit(text, text.get_rect(center=self.accept.center))
        self.draw_hint(screen)


# -- ELIGE CIRCUITO ---------------------------------------------------------------------------


@dataclass
class TrackOption:
    key: str  # "harbor", "file:tracks/x.json", "random", "eval:7" or "new" (editor)
    title: str
    track: Track | None
    error: str | None = None  # why the file cannot be played (a TrackFileError text)
    seed: int | None = None  # random: the seed whose track is shown


def file_key(path: Path, root: Path) -> str:
    try:
        return "file:" + path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return "file:" + path.resolve().as_posix()


class TrackCatalog:
    """The circuits the picker offers, built once and cached (the thumbnails need the tracks)."""

    def __init__(self, tracks_dir: Path, root: Path, rng: np.random.Generator) -> None:
        self.tracks_dir = Path(tracks_dir)
        self.root = Path(root)
        self.rng = rng
        self.random_seed = self.new_seed()
        self.eval_index = 0
        self._cache: dict[tuple, Track] = {}
        self._random: OrderedDict[int, Track] = OrderedDict()  # LRU: a seed per "otro aleatorio"

    def new_seed(self) -> int:
        from kartlab.sim.tracksets import sample_train_seed

        return sample_train_seed(self.rng)

    def _cached(self, key: tuple, build) -> Track:
        track = self._cache.get(key)
        if track is None:
            track = self._cache[key] = build()
        return track

    def showcase(self) -> list[TrackOption]:
        from kartlab.sim.showcase import showcase_track

        return [
            TrackOption(
                name, design.name, self._cached(("s", name), lambda n=name: showcase_track(n))
            )
            for name, design in SHOWCASE.items()
        ]

    def files(self, for_editor: bool = False) -> list[TrackOption]:
        options = []
        for path in sorted(self.tracks_dir.glob("*.json")):
            key = file_key(path, self.root)
            stamp = path.stat().st_mtime_ns
            try:
                track = self._cached(("f", key, stamp), lambda p=path: load_track_file(p))
                title = track.name
                error = None
            except TrackFileError as problem:
                track, title, error = None, path.stem, str(problem)
            if for_editor:  # the editor opens a file whose circuit is not valid yet
                try:
                    title = load_spec(path).name
                    error = None
                except TrackFileError as problem:
                    error = str(problem)
            options.append(TrackOption(key, title, track, error))
        return options

    def random(self) -> TrackOption:
        seed = self.random_seed
        track = self._random.pop(seed, None) or train_track(seed)
        self._random[seed] = track  # most recently used last
        while len(self._random) > RANDOM_CACHE:
            self._random.popitem(last=False)
        return TrackOption("random", "Aleatorio", track, seed=self.random_seed)

    def evaluation(self) -> TrackOption:
        i = self.eval_index
        track = self._cached(("e", i), lambda: eval_track(i))
        return TrackOption(f"eval:{i + 1}", f"Evaluación {i + 1}", track)

    def play_options(self) -> list[TrackOption]:
        return [*self.showcase(), *self.files(), self.random(), self.evaluation()]

    def editor_options(self) -> list[TrackOption]:
        return [*self.files(for_editor=True), TrackOption("new", "Circuito nuevo", None)]

    def resolve(self, choice: str) -> Track:
        """The track of a choice, built fresh. ``TrackFileError`` if a file is not valid."""
        from kartlab.sim.showcase import showcase_track

        if choice in SHOWCASE:
            return showcase_track(choice)
        if choice.startswith("file:") or choice.lower().endswith(".json"):
            path = Path(choice.removeprefix("file:"))
            if not path.is_absolute() and (self.root / path).exists():
                path = self.root / path  # relative to the repository, else to the working folder
            return load_track_file(path)
        if choice.startswith("eval:"):
            number = int(choice.removeprefix("eval:"))
            if not 1 <= number <= len(EVAL_SEEDS):
                raise ValueError(f"evaluación fuera de rango: {number}")
            return eval_track(number - 1)
        if choice.startswith("random"):
            seed = int(choice.partition(":")[2] or self.new_seed())
            return train_track(seed)
        raise ValueError(f"circuito desconocido: {choice}")


class TrackScreen(Screen):
    """Cards with the name, the length and a small drawing of the circuit."""

    dim = 60

    def __init__(self, hub: KartLabApp, flow: Flow, for_editor: bool = False) -> None:
        super().__init__(hub)
        self.flow = flow
        self.for_editor = for_editor
        self.options: list[TrackOption] = []
        self.selected = 0
        self.scroll = 0  # first visible row
        self.rects: dict[int, pygame.Rect] = {}
        self._thumbs: dict[tuple, pygame.Surface] = {}
        self.eval_arrows: tuple[pygame.Rect, pygame.Rect] | None = None

    @property
    def hint(self) -> str:  # type: ignore[override]
        base = "← → ↑ ↓ elegir · Intro o clic confirmar · Esc volver"
        if self.for_editor:
            return base
        current = self.options[self.selected].key if self.options else ""
        if current == "random":
            return base + " · R otro aleatorio"
        if current.startswith("eval:"):
            return base + " · Re Pág / Av Pág otra evaluación"
        return base

    def enter(self) -> None:
        self.build_options()
        super().enter()

    def build_options(self) -> None:
        catalog = self.hub.catalog
        if self.for_editor:
            options = catalog.editor_options()
            wanted = None
        else:
            options = catalog.play_options()
            wanted = self.hub.settings.last_track
            number = wanted.partition(":")[2] if wanted.startswith("eval:") else ""
            if number.isdigit() and 1 <= int(number) <= len(EVAL_SEEDS):
                catalog.eval_index = int(number) - 1
                options = catalog.play_options()
        filt = self.flow.entry.track_filter
        if filt is not None:
            options = [o for o in options if filt(o)]
        self.options = options
        self.selected = next((i for i, o in enumerate(options) if o.key == wanted), 0)

    def refresh_dynamic(self) -> None:
        """After the random seed or the evaluation number changed: only those cards differ."""
        catalog = self.hub.catalog
        for i, option in enumerate(self.options):
            if option.key == "random":
                self.options[i] = catalog.random()
            elif option.key.startswith("eval:"):
                self.options[i] = catalog.evaluation()

    # -- layout --------------------------------------------------------------------------

    def layout(self) -> None:
        ui = self.hub.ui
        w, h = ui.size
        u = ui.unit
        usable = round(w * 0.86)
        gap = round(u * 0.8)
        cols = max(1, min(4, (usable + gap) // (u * 13 + gap)))
        card_w = min((usable - gap * (cols - 1)) // cols, u * 19)
        self.cols = cols
        self.card_w = card_w
        self.card_h = round(card_w * 0.92)
        top = round(h * 0.19)
        bottom = h - round(u * 3.2)
        self.rows_visible = max(1, (bottom - top + gap) // (self.card_h + gap))
        self.gap = gap
        self.top = top
        total_w = cols * card_w + (cols - 1) * gap
        self.left = (w - total_w) // 2
        self._thumbs.clear()
        self._place()

    def _place(self) -> None:
        self.rects = {}
        last_row = (len(self.options) - 1) // self.cols
        self.scroll = min(max(self.scroll, 0), max(last_row - self.rows_visible + 1, 0))
        for i in range(len(self.options)):
            row, col = divmod(i, self.cols)
            if self.scroll <= row < self.scroll + self.rows_visible:
                self.rects[i] = pygame.Rect(
                    self.left + col * (self.card_w + self.gap),
                    self.top + (row - self.scroll) * (self.card_h + self.gap),
                    self.card_w,
                    self.card_h,
                )

    def _reveal(self) -> None:
        row = self.selected // self.cols
        if row < self.scroll:
            self.scroll = row
        elif row >= self.scroll + self.rows_visible:
            self.scroll = row - self.rows_visible + 1
        self._place()

    # -- input ---------------------------------------------------------------------------

    def move(self, dx: int, dy: int) -> None:
        count = len(self.options)
        if not count:
            return
        if dy:
            target = self.selected + dy * self.cols
            if 0 <= target < count:
                self.selected = target
            elif dy > 0 and self.selected // self.cols < (count - 1) // self.cols:
                self.selected = count - 1
        if dx:
            self.selected = (self.selected + dx) % count
        self._reveal()

    def confirm(self) -> None:
        hub = self.hub
        option = self.options[self.selected]
        if option.error and not self.for_editor:
            hub.message("Circuito no válido", option.error)
            return
        value = option.key
        if option.key == "random" and option.seed is not None:
            value = f"random:{option.seed}"
        if not self.for_editor:
            hub.settings.last_track = option.key
            hub.save_settings()
        hub.advance(value)

    def change_dynamic(self, delta: int) -> None:
        option = self.options[self.selected]
        catalog = self.hub.catalog
        if option.key == "random" and delta != 0:
            catalog.random_seed = catalog.new_seed()
        elif option.key.startswith("eval:"):
            catalog.eval_index = (catalog.eval_index + delta) % len(EVAL_SEEDS)
        else:
            return
        self.refresh_dynamic()

    def on_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.hub.back()
        elif key in (pygame.K_LEFT, pygame.K_a):
            self.move(-1, 0)
        elif key in (pygame.K_RIGHT, pygame.K_d):
            self.move(1, 0)
        elif key in (pygame.K_UP, pygame.K_w):
            self.move(0, -1)
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.move(0, 1)
        elif key == pygame.K_PAGEUP:
            self.change_dynamic(1)
        elif key == pygame.K_PAGEDOWN:
            self.change_dynamic(-1)
        elif key == pygame.K_r:
            self.change_dynamic(1)
        elif key in CONFIRM_KEYS:
            self.confirm()

    def _card_at(self, pos) -> int | None:
        for i, rect in self.rects.items():
            if rect.collidepoint(pos):
                return i
        return None

    def on_motion(self, pos) -> None:
        i = self._card_at(pos)
        if i is not None:
            self.selected = i

    def on_click(self, pos) -> None:
        if self.eval_arrows is not None:
            for sign, rect in zip((-1, 1), self.eval_arrows, strict=True):
                if rect.collidepoint(pos):
                    self.change_dynamic(sign)
                    return
        i = self._card_at(pos)
        if i is not None:
            self.selected = i
            self.confirm()

    def on_wheel(self, dy: int) -> None:
        self.move(0, -1 if dy > 0 else 1)

    # -- drawing -------------------------------------------------------------------------

    def thumbnail(self, option: TrackOption, size: tuple[int, int]) -> pygame.Surface | None:
        track = option.track
        if track is None:
            return None
        key = (option.key, getattr(option, "seed", None), size)
        thumb = self._thumbs.get(key)
        if thumb is not None:
            return thumb
        thumb = pygame.Surface(size, pygame.SRCALPHA)
        pts = track.points
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        scale = min(size[0] / span[0], size[1] / span[1]) * 0.86
        centre = (lo + hi) / 2
        xs = size[0] / 2 + (pts[:, 0] - centre[0]) * scale
        ys = size[1] / 2 - (pts[:, 1] - centre[1]) * scale  # y points up in the world
        line = [(float(x), float(y)) for x, y in zip(xs, ys, strict=True)]
        width = max(round(track.width * scale), 2)
        edge = (22, 24, 34)
        road = (196, 200, 212)
        pygame.draw.lines(thumb, edge, True, line, width + 4)
        pygame.draw.lines(thumb, road, True, line, width)
        for x, y in line[::6]:  # round the joints between the segments
            pygame.draw.circle(thumb, road, (round(x), round(y)), max(width // 2, 1))
        sx, sy = line[0]
        cell = max(width // 2, 2)
        for dx in range(2):
            for dy in range(2):
                tone = (250, 250, 250) if (dx + dy) % 2 == 0 else (20, 20, 28)
                thumb.fill(
                    tone, (round(sx) - cell + dx * cell, round(sy) - cell + dy * cell, cell, cell)
                )
        self._thumbs[key] = thumb
        return thumb

    def draw(self, screen: pygame.Surface) -> None:
        hub = self.hub
        ui = hub.ui
        w, h = ui.size
        u = ui.unit
        title = "ABRE UN CIRCUITO" if self.for_editor else "ELIGE CIRCUITO"
        ui.heading_text(screen, title, round(h * 0.06), self.age)
        self.eval_arrows = None
        for i, rect in self.rects.items():
            option = self.options[i]
            chosen = i == self.selected
            screen.blit(ui.panel(rect.w, rect.h, 205), rect.topleft)
            if chosen:
                pygame.draw.rect(screen, ACCENT, rect, max(u // 5, 2))
                pygame.draw.rect(screen, (255, 255, 255), (rect.x, rect.y, rect.w, max(u // 8, 2)))
            pad = u // 2
            thumb_size = (rect.w - 2 * pad, rect.h - round(u * 3.4) - pad)
            thumb = self.thumbnail(option, thumb_size)
            if thumb is not None:
                screen.blit(thumb, (rect.x + pad, rect.y + pad))
            elif option.key == "new":
                plus = ui.big.get("+", ACCENT)
                screen.blit(
                    plus,
                    (
                        rect.centerx - plus.get_width() // 2,
                        rect.y + pad + thumb_size[1] // 2 - plus.get_height() // 2,
                    ),
                )
            else:
                bad = ui.small.get("No válido", WARN)
                screen.blit(
                    bad, (rect.centerx - bad.get_width() // 2, rect.y + pad + thumb_size[1] // 2)
                )
            name = ui.name.get(option.title, ACCENT if chosen else WHITE)
            screen.blit(name, (rect.x + pad, rect.bottom - round(u * 3.1)))
            if option.error and not self.for_editor:
                sub = ui.small.get("Fichero no válido", WARN)
            elif option.track is not None:
                sub = ui.small.get(f"{option.track.length:,.0f} m".replace(",", "."), DIM)
            else:
                sub = ui.small.get("Empieza de cero", DIM)
            screen.blit(sub, (rect.x + pad, rect.bottom - round(u * 1.6)))
            if option.key.startswith("eval:") and chosen:
                self.eval_arrows = self._draw_eval_arrows(screen, rect)
        total_rows = (len(self.options) - 1) // self.cols + 1
        if total_rows > self.rows_visible:  # a thin scroll bar at the right of the cards
            track_h = self.rows_visible * (self.card_h + self.gap) - self.gap
            bar_h = max(round(track_h * self.rows_visible / total_rows), u)
            bar_y = self.top + round(
                (track_h - bar_h) * self.scroll / (total_rows - self.rows_visible)
            )
            x = self.left + self.cols * (self.card_w + self.gap)
            pygame.draw.rect(screen, (60, 64, 80), (x, self.top, max(u // 4, 3), track_h))
            pygame.draw.rect(screen, ACCENT, (x, bar_y, max(u // 4, 3), bar_h))
        self.draw_hint(screen)

    def _draw_eval_arrows(self, screen, rect) -> tuple[pygame.Rect, pygame.Rect]:
        ui = self.hub.ui
        u = ui.unit
        size = round(u * 2)
        left = pygame.Rect(rect.right - 2 * size - u // 2, rect.bottom - round(u * 3.2), size, size)
        right = left.move(size, 0)
        for box, glyph in ((left, "‹"), (right, "›")):
            pygame.draw.rect(screen, ACCENT, box)
            text = ui.name.get(glyph, DARK)
            screen.blit(
                text, (box.centerx - text.get_width() // 2, box.centery - text.get_height() // 2)
            )
        return left, right
