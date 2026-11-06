"""The two pick-a-thing screens that come before a game: "ELIGE KART" and "ELIGE CIRCUITO".

Both are steps of an ``Entry`` (see ``menu.py``): the first stores the kart colour (an index
of ``KART_RAMPS``), the second the circuit (``harbor``, ``file:tracks/x.json``, ``random``,
``eval:7``...). The kart is drawn with the world's own ``SpriteBank`` (sprite stacking), at
the size it is shown on the 320x180 canvas; the circuit cards draw the centerline in small.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pygame

from kartlab.render import pixelui as ui
from kartlab.render.palette import KART_RAMPS
from kartlab.render.sprites import ELEVATIONS, SCALE_STEPS, bucket_scale, scale_bucket
from kartlab.show.cli import KART_COLOR_NAMES
from kartlab.show.menu import CONFIRM_KEYS, Flow, Screen, W, button, heading
from kartlab.sim.showcase import SHOWCASE
from kartlab.sim.track import Track
from kartlab.sim.trackfile import TrackFileError, load_spec, load_track_file
from kartlab.sim.tracksets import EVAL_SEEDS, eval_track, train_track

if TYPE_CHECKING:
    from kartlab.show.hub import KartLabApp

PREVIEW_ELEVATION = ELEVATIONS[2]  # 35 degrees: the kart seen from above and behind
RANDOM_CACHE = 16  # random circuits kept (each "R" makes a new one)
SPIN_SECONDS = 7.0  # one turn of the kart on its turntable
SWATCH = 14  # side of a colour swatch (px)
SWATCH_GAP = 6


# -- ELIGE TU KART ----------------------------------------------------------------------------


class KartScreen(Screen):
    dim = 110
    hide_backdrop_kart = True  # the autopilot's kart would get in the way of the big one
    hint = "←→ elegir, Intro aceptar, Esc volver"

    def __init__(self, hub: KartLabApp, flow: Flow) -> None:
        super().__init__(hub)
        self.flow = flow
        self.selected = hub.settings.kart_ramp
        self.swatches: list[pygame.Rect] = []

    def layout(self) -> None:
        # Pixels per voxel so the kart is about 40 % of the height, never past the finest size.
        wanted = (ui.CANVAS_SIZE[1] * 0.40) / 30.6
        self.ppv = bucket_scale(min(scale_bucket(wanted), SCALE_STEPS - 1))
        total = 8 * SWATCH + 7 * SWATCH_GAP
        x = (W - total) // 2
        y = 133
        self.swatches = [
            pygame.Rect(x + i * (SWATCH + SWATCH_GAP), y, SWATCH, SWATCH) for i in range(8)
        ]
        self.accept = pygame.Rect(0, 0, 56, 13)
        self.accept.midtop = (W // 2, y + SWATCH + 6)
        self.ground = (W // 2, 92)
        self.arrow_left = pygame.Rect(0, 0, 14, 22)
        self.arrow_left.center = (W // 4 - 4, self.ground[1] - 14)
        self.arrow_right = self.arrow_left.copy()
        self.arrow_right.centerx = W - self.arrow_left.centerx

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
                if rect.inflate(SWATCH_GAP, SWATCH_GAP).collidepoint(pos):
                    if i == self.selected:
                        self.confirm()
                    else:
                        self.selected = i
                    return

    def on_wheel(self, dy: int) -> None:
        self.move(-1 if dy > 0 else 1)

    # -- drawing -------------------------------------------------------------------------

    def draw(self, canvas: pygame.Surface) -> None:
        hub = self.hub
        heading(canvas, "ELIGE KART", self.age)

        # Turntable: a dark ellipse with a rim in the kart's colour, and the kart spinning on it.
        cx, ground_y = self.ground
        ink = KART_RAMPS[self.selected][1]
        plate = pygame.Rect(0, 0, 96, 28)
        plate.center = (cx, ground_y)
        pygame.draw.ellipse(canvas, ui.rgb(ui.NIGHT), plate.move(0, 3))
        pygame.draw.ellipse(canvas, ui.rgb(ui.SLATE), plate)
        pygame.draw.ellipse(canvas, ui.rgb(ink), plate, 2)

        hub.sprites.begin_frame()
        angle = hub.t * math.tau / SPIN_SECONDS
        sprite, (ax, ay) = hub.sprites.kart(self.selected, angle, PREVIEW_ELEVATION, self.ppv)
        bob = round(math.sin(hub.t * 2.4) * 0.6)
        canvas.blit(sprite, (round(cx - ax), round(ground_y - ay) + bob))

        name = KART_COLOR_NAMES[self.selected].upper()
        ui.text(
            canvas,
            name,
            cx,
            ground_y + 20,
            ink,
            2,
            outline=ui.NIGHT,
            shadow=ui.NIGHT,
            align="center",
        )

        for rect, direction in ((self.arrow_left, -1), (self.arrow_right, 1)):
            tip = rect.centerx + direction * rect.w // 2
            back = rect.centerx - direction * rect.w // 2
            outline = [(tip, rect.centery), (back, rect.y), (back, rect.bottom - 1)]
            pygame.draw.polygon(canvas, ui.rgb(ui.YELLOW), outline)
            pygame.draw.polygon(canvas, ui.rgb(ui.NIGHT), outline, 1)

        for i, rect in enumerate(self.swatches):
            if i == self.selected:  # the chosen one is framed in yellow and marked with a dot
                ui.plate(canvas, rect.inflate(6, 6), ui.YELLOW)
            ui.plate(canvas, rect, KART_RAMPS[i][1])
        ui.icon(
            canvas,
            "dot",
            self.swatches[self.selected].centerx - 2,
            self.swatches[self.selected].centery - 2,
            ui.NIGHT,
        )
        button(canvas, self.accept, "ACEPTAR", chosen=True)
        self.draw_hint(canvas)


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
    COLS = 4
    CARD_W = 70
    CARD_H = 50
    GAP = 6
    TOP = 34
    THUMB = (62, 18)

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
        parts = ["↑↓←→ elegir", "Intro aceptar"]
        if not self.for_editor:
            current = self.options[self.selected].key if self.options else ""
            if current == "random":
                parts.append("R otro circuito")
            elif current.startswith("eval:"):
                parts.append("Re/Av Pág cambian")
        parts.append("Esc volver")
        return ", ".join(parts)

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
        self.cols = self.COLS
        self.card_w, self.card_h, self.gap, self.top = self.CARD_W, self.CARD_H, self.GAP, self.TOP
        # The cards leave a strip at the bottom for the details of the chosen circuit.
        bottom = 148
        self.rows_visible = max(1, (bottom - self.top + self.gap) // (self.card_h + self.gap))
        total_w = self.cols * self.card_w + (self.cols - 1) * self.gap
        self.left = (W - total_w) // 2
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

    def thumbnail(self, option: TrackOption) -> pygame.Surface | None:
        """The centerline in small: dark edge, light road, a chequered start. Colour-keyed."""
        track = option.track
        if track is None:
            return None
        size = self.THUMB
        key = (option.key, option.seed)
        thumb = self._thumbs.get(key)
        if thumb is not None:
            return thumb
        thumb = pygame.Surface(size)
        thumb.fill(ui._KEY)
        thumb.set_colorkey(ui._KEY)
        pts = track.points
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        scale = min((size[0] - 6) / span[0], (size[1] - 6) / span[1])
        centre = (lo + hi) / 2
        xs = size[0] / 2 + (pts[:, 0] - centre[0]) * scale
        ys = size[1] / 2 - (pts[:, 1] - centre[1]) * scale  # y points up in the world
        line = [(round(float(x)), round(float(y))) for x, y in zip(xs, ys, strict=True)]
        night, road = ui.rgb(ui.NIGHT), ui.rgb(ui.TEXT)
        pygame.draw.lines(thumb, night, True, line, 4)
        for x, y in line[::4]:  # round the joints between the segments
            pygame.draw.circle(thumb, night, (x, y), 2)
        pygame.draw.lines(thumb, road, True, line, 2)
        for x, y in line[::4]:
            pygame.draw.circle(thumb, road, (x, y), 1)
        sx, sy = line[0]
        for dx in range(2):
            for dy in range(2):
                ink = ui.rgb(ui.WHITE) if (dx + dy) % 2 == 0 else night
                thumb.fill(ink, (sx - 2 + dx * 2, sy - 2 + dy * 2, 2, 2))
        self._thumbs[key] = thumb
        return thumb

    def draw(self, canvas: pygame.Surface) -> None:
        title = "EDITAR CIRCUITO" if self.for_editor else "ELIGE CIRCUITO"
        heading(canvas, title, self.age)
        self.eval_arrows = None
        for i, rect in self.rects.items():
            option = self.options[i]
            chosen = i == self.selected
            ui.plate(canvas, rect, ui.SLATE, ui.YELLOW if chosen else ui.NIGHT)
            if chosen:
                ui.box(canvas, rect.inflate(-2, -2), ui.YELLOW)
            area = pygame.Rect(rect.x + 4, rect.y + 3, *self.THUMB)
            thumb = self.thumbnail(option)
            if thumb is not None:
                canvas.blit(thumb, area.topleft)
            elif option.key == "new":
                ui.text(canvas, "+", area.centerx, area.centery - 6, ui.YELLOW, 2, align="center")
            else:
                ui.icon(canvas, "cross", area.centerx - 2, area.centery - 2, ui.RED)
            lines = ui.wrap(option.title, rect.w - 8)
            if len(lines) > 2:  # two lines of the name; the strip below has the whole of it
                lines = [lines[0], ui.ellipsize(" ".join(lines[1:]), rect.w - 8)]
            for k, line in enumerate(lines):
                ink = ui.YELLOW if chosen else ui.TEXT
                ui.text(canvas, line, rect.x + 4, rect.y + 23 + 9 * k, ink)
            if option.error and not self.for_editor:
                ui.text(canvas, "NO VÁLIDO", rect.x + 4, rect.bottom - 9, ui.RED)
            elif option.track is not None:
                length = ui.fmt_count(round(option.track.length))
                ui.text(canvas, f"{length} M", rect.x + 4, rect.bottom - 9, ui.STONE_LIGHT)
            else:
                ui.text(canvas, "EN BLANCO", rect.x + 4, rect.bottom - 9, ui.STONE_LIGHT)
            if option.key.startswith("eval:") and chosen:
                self.eval_arrows = self._draw_eval_arrows(canvas, rect)
        self._draw_details(canvas)
        total_rows = (len(self.options) - 1) // self.cols + 1
        if total_rows > self.rows_visible:  # a thin scroll bar at the right of the cards
            track_h = self.rows_visible * (self.card_h + self.gap) - self.gap
            bar_h = max(round(track_h * self.rows_visible / total_rows), 6)
            bar_y = self.top + round(
                (track_h - bar_h) * self.scroll / (total_rows - self.rows_visible)
            )
            x = self.left + self.cols * (self.card_w + self.gap) - 2
            canvas.fill(ui.rgb(ui.SLATE), (x, self.top, 3, track_h))
            canvas.fill(ui.rgb(ui.YELLOW), (x, bar_y, 3, bar_h))
        self.draw_hint(canvas)

    def _draw_details(self, canvas: pygame.Surface) -> None:
        """The chosen circuit's full name and length, in a strip under the cards."""
        if not self.options:
            return
        option = self.options[self.selected]
        rect = pygame.Rect(self.left, 152, self.cols * (self.card_w + self.gap) - self.gap, 13)
        ui.plate(canvas, rect, ui.NAVY)
        ui.text(canvas, ui.ellipsize(option.title, rect.w - 140), rect.x + 5, rect.y + 3, ui.YELLOW)
        if option.error and not self.for_editor:
            ui.text(
                canvas, "INTRO MUESTRA EL ERROR", rect.right - 5, rect.y + 3, ui.RED, align="right"
            )
        elif option.track is not None:
            length = ui.fmt_count(round(option.track.length))
            ui.text(canvas, f"{length} M", rect.right - 5, rect.y + 3, ui.TEXT, align="right")

    def _draw_eval_arrows(self, canvas, rect) -> tuple[pygame.Rect, pygame.Rect]:
        left = pygame.Rect(rect.right - 20, rect.bottom - 24, 9, 9)
        right = left.move(10, 0)
        for box, name in ((left, "left"), (right, "right")):
            ui.plate(canvas, box, ui.YELLOW)
            ui.icon(canvas, name, box.x + 2, box.y + 2, ui.NIGHT)
        return left, right
