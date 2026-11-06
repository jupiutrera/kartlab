"""Mouse-driven track editor: draw a circuit, see it validated live, save it as JSON.

Run with ``uv run python scripts/editor.py [tracks/x.json] [--new]``.

The track is the same corner polygon the JSON file holds (``TrackSpec``). What you see is
built through the loader's own path (``trackfile``): the road is the smoothed centerline
the AI and ``drive.py`` would get, and the verdict is the validator's, in Spanish, with the
offending corners and sections highlighted (``trackissues``).

Format 2 files (with level crossings) are opened and edited like any other: the declared
``crossings`` and ``start`` are kept untouched and written back on save. The editor has no
control to declare or move them; if a dragged corner makes a declared crossing disappear
the validator says so in red, as for any other mistake.

``EditorModel`` (the design and its undo history) knows nothing of the screen. ``Viewport``
maps metres to pixels. ``EditorApp`` owns the window, the input and the drawing.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pygame

from kartlab.common.paths import REPO_ROOT
from kartlab.render.display import desktop_size, enable_dpi_awareness
from kartlab.render.topdown import (
    ACCENT,
    ASPHALT,
    CENTER_MARK,
    CHECKER_DARK,
    EDGE,
    FONT_BOLD,
    FONT_REGULAR,
    GRASS,
    GRASS_GRID,
    KART_OUTLINE,
    KERB_WHITE,
    PANEL,
    TEXT,
    TEXT_DIM,
    TextCache,
)
from kartlab.sim.trackfile import (
    MAX_COORDINATE,
    MAX_CORNERS,
    MAX_DECORATION_SEED,
    MAX_NAME,
    MAX_RADIUS,
    MAX_WIDTH,
    MIN_CORNERS,
    MIN_RADIUS,
    MIN_WIDTH,
    TrackFileError,
    TrackSpec,
    load_spec,
    save_spec,
)
from kartlab.sim.trackissues import Analysis, Issue, analyze

TITLE = "KART LAB · editor de circuitos"
TRACKS_DIR = REPO_ROOT / "tracks"

# Editing
HISTORY_LIMIT = 200  # undo steps kept
COALESCE_WINDOW = 0.8  # repeated wheel / key nudges within this many seconds are one undo step
RADIUS_STEP = 2.0  # m per wheel notch over a corner (x5 with Shift)
WIDTH_STEP = 1.0  # m
NUDGE_STEP = 1.0  # m per arrow key press on the selected corner (x10 with Shift)
SNAP_STEP = 5.0  # m, dragging with Ctrl held
COORD_DECIMALS = 1  # corner positions are kept to 0.1 m, so the JSON stays readable
DEFAULT_RADIUS = 30.0  # radius of a corner inserted between two others (m)
NEW_SPEC = TrackSpec(
    "Nuevo circuito",
    12.0,
    (
        (0.0, 0.0, 40.0),
        (260.0, 0.0, 40.0),
        (340.0, 110.0, 40.0),
        (260.0, 220.0, 40.0),
        (0.0, 220.0, 40.0),
        (-80.0, 110.0, 40.0),
    ),
)

# View
ZOOM_MIN = 0.1  # px per m
ZOOM_MAX = 14.0
ZOOM_KEY_FACTOR = 1.25
ZOOM_WHEEL_FACTOR = 1.15
CORNER_PICK = 12.0  # px
EDGE_PICK = 8.0  # px
FIT_MARGIN = 70  # px
GRID_STEPS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
GRID_MIN_PX = 38.0  # smallest on-screen grid spacing

# Analysis scheduling (see ``EditorApp.update``)
QUICK_BUDGET = 0.012  # s; a drag rebuilds as often as the frames allow while this holds
FULL_DEBOUNCE = 0.12  # s without edits before the complete (slow) validation runs

# Colours
VALID = (96, 214, 126)
INVALID = (255, 96, 88)
PENDING = (255, 196, 84)
HANDLE = (240, 242, 248)
HANDLE_HOVER = (255, 226, 130)
PANEL_SOLID = (20, 23, 31)
PANEL_LINE = (52, 58, 72)
ROAD_EDGE_BAD = (236, 84, 84)

HELP_ROWS: tuple[tuple[str, str], ...] = (
    ("Arrastrar una esquina", "moverla (con Ctrl se ajusta a 5 m)"),
    ("Clic en un borde", "insertar una esquina y arrastrarla"),
    ("Clic derecho / Supr", "borrar la esquina (mínimo 3)"),
    ("Rueda sobre una esquina", "radio ±2 m (con Mayús ±10 m)"),
    ("Rueda en otro sitio, + / -", "acercar y alejar"),
    ("Botón central, Espacio + arrastrar", "mover la vista (arrastrar en vacío también)"),
    ("F", "encuadrar el circuito"),
    ("Flechas", "mover la esquina elegida 1 m (Mayús 10 m)"),
    ("Re Pág / Av Pág", "anchura del asfalto ±1 m"),
    ("D / Mayús + D", "semilla del decorado ±1"),
    ("F2", "cambiar el nombre"),
    ("Ctrl + Z / Ctrl + Y", "deshacer / rehacer"),
    ("Ctrl + S", "guardar (solo si es válido)"),
    ("Ctrl + Mayús + S", "guardar como… (nombre o ruta)"),
    ("F5 o T", "probar el circuito conduciendo"),
    ("H", "esta ayuda"),
    ("Esc", "salir (pide confirmar si hay cambios)"),
)


# -- pure helpers ----------------------------------------------------------------------------


def slugify(name: str) -> str:
    """File name stem for a track name: ``Bahía del Faro`` -> ``bahia-del-faro``."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    stem = "".join(c.lower() if c.isalnum() else "-" for c in ascii_name)
    return "-".join(part for part in stem.split("-") if part) or "circuito"


def resolve_save_path(text: str, tracks_dir: Path = TRACKS_DIR) -> Path:
    """The file a "save as" answer means: a bare name goes to ``tracks/<slug>.json``;
    anything with a folder or a ``.json`` suffix is taken as a path."""
    text = text.strip()
    if "/" in text or "\\" in text or text.lower().endswith(".json"):
        path = Path(text)
        return path if path.suffix else path.with_suffix(".json")
    return tracks_dir / f"{slugify(text)}.json"


def project_on_segment(
    p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> tuple[float, tuple[float, float]]:
    """Closest point to ``p`` on segment ``ab``: its parameter in [0, 1] and the point."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = (
        0.0
        if length2 == 0
        else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
    )
    return t, (a[0] + t * dx, a[1] + t * dy)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _round_xy(x: float, y: float) -> tuple[float, float]:
    x = clamp(x, -MAX_COORDINATE, MAX_COORDINATE)
    y = clamp(y, -MAX_COORDINATE, MAX_COORDINATE)
    return round(x, COORD_DECIMALS), round(y, COORD_DECIMALS)


# -- the design and its history ----------------------------------------------------------------


class EditorModel:
    """The track being edited, with a bounded undo/redo history.

    ``TrackSpec`` is immutable, so a history entry is just the previous spec and undoing
    restores it exactly. A drag changes the spec many times but is one history step
    (``begin_drag`` / ``end_drag``); repeated wheel or arrow-key tweaks of the same
    thing within ``COALESCE_WINDOW`` seconds also count as one.
    """

    def __init__(self, spec: TrackSpec, saved: bool = True) -> None:
        self.spec = spec
        self.saved_spec: TrackSpec | None = spec if saved else None
        self._undo: deque[TrackSpec] = deque(maxlen=HISTORY_LIMIT)
        self._redo: list[TrackSpec] = []
        self._drag_origin: TrackSpec | None = None
        self._key: object = None
        self._key_time = -math.inf

    @property
    def dirty(self) -> bool:
        return self.spec != self.saved_spec

    @property
    def dragging(self) -> bool:
        return self._drag_origin is not None

    @property
    def undo_depth(self) -> int:
        return len(self._undo)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def mark_saved(self) -> None:
        self.saved_spec = self.spec

    # history

    def _record(self, before: TrackSpec, key: object, now: float) -> None:
        if key is not None and key == self._key and now - self._key_time <= COALESCE_WINDOW:
            self._key_time = now
        else:
            self._undo.append(before)
            self._key, self._key_time = key, now
        self._redo.clear()

    def apply(self, spec: TrackSpec, key: object = None, now: float = 0.0) -> bool:
        """Replace the design as one undoable step. False if nothing changed."""
        if spec == self.spec:
            return False
        before, self.spec = self.spec, spec
        if self._drag_origin is None:
            self._record(before, key, now)
        return True

    def undo(self) -> bool:
        if self.dragging or not self._undo:
            return False
        self._redo.append(self.spec)
        self.spec = self._undo.pop()
        self._key = None
        return True

    def redo(self) -> bool:
        if self.dragging or not self._redo:
            return False
        self._undo.append(self.spec)
        self.spec = self._redo.pop()
        self._key = None
        return True

    def begin_drag(self) -> None:
        """Start a gesture: every change until ``end_drag`` is one undo step."""
        if self._drag_origin is None:
            self._drag_origin = self.spec

    def end_drag(self, now: float = 0.0) -> bool:
        origin, self._drag_origin = self._drag_origin, None
        if origin is None or origin == self.spec:
            return False
        self._record(origin, None, now)
        return True

    # edits

    def move_corner(self, index: int, x: float, y: float, now: float = 0.0) -> bool:
        corners = list(self.spec.corners)
        x, y = _round_xy(x, y)
        corners[index] = (x, y, corners[index][2])
        return self.apply(replace(self.spec, corners=tuple(corners)), None, now)

    def insert_corner(self, edge: int, x: float, y: float) -> int:
        """Insert a corner on the edge from corner ``edge`` to the next one, at (x, y).
        Starts a drag gesture, so inserting and placing it is one undo step. Returns the
        new corner's index."""
        corners = list(self.spec.corners)
        a, b = corners[edge], corners[(edge + 1) % len(corners)]
        radius = clamp(round((a[2] + b[2]) / 2), MIN_RADIUS, MAX_RADIUS)
        x, y = _round_xy(x, y)
        index = edge + 1
        corners.insert(index, (x, y, radius if radius else DEFAULT_RADIUS))
        self.begin_drag()
        self.apply(replace(self.spec, corners=tuple(corners)))
        return index

    def delete_corner(self, index: int, now: float = 0.0) -> bool:
        if len(self.spec.corners) <= MIN_CORNERS:
            return False
        corners = list(self.spec.corners)
        del corners[index]
        return self.apply(replace(self.spec, corners=tuple(corners)), None, now)

    def set_radius(self, index: int, radius: float, now: float = 0.0) -> bool:
        corners = list(self.spec.corners)
        x, y, _ = corners[index]
        corners[index] = (x, y, round(clamp(radius, MIN_RADIUS, MAX_RADIUS), 1))
        return self.apply(replace(self.spec, corners=tuple(corners)), ("radius", index), now)

    def nudge_corner(self, index: int, dx: float, dy: float, now: float = 0.0) -> bool:
        x, y, _ = self.spec.corners[index]
        corners = list(self.spec.corners)
        nx, ny = _round_xy(x + dx, y + dy)
        corners[index] = (nx, ny, corners[index][2])
        return self.apply(replace(self.spec, corners=tuple(corners)), ("nudge", index), now)

    def set_width(self, width: float, now: float = 0.0) -> bool:
        width = round(clamp(width, MIN_WIDTH, MAX_WIDTH), 1)
        return self.apply(replace(self.spec, width=width), "width", now)

    def set_seed(self, seed: int, now: float = 0.0) -> bool:
        seed = int(seed) % (MAX_DECORATION_SEED + 1)
        return self.apply(replace(self.spec, decoration_seed=seed), "seed", now)

    def set_name(self, name: str, now: float = 0.0) -> bool:
        return self.apply(replace(self.spec, name=name.strip()), None, now)


# -- camera ------------------------------------------------------------------------------------


@dataclass
class Viewport:
    """Maps world metres (y up) to canvas pixels (y down)."""

    cx: float = 0.0
    cy: float = 0.0
    zoom: float = 1.0  # px per m
    size: tuple[int, int] = (800, 600)

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        return (
            (x - self.cx) * self.zoom + self.size[0] / 2,
            self.size[1] / 2 - (y - self.cy) * self.zoom,
        )

    def to_world(self, sx: float, sy: float) -> tuple[float, float]:
        return (
            self.cx + (sx - self.size[0] / 2) / self.zoom,
            self.cy + (self.size[1] / 2 - sy) / self.zoom,
        )

    def to_screen_array(self, pts: np.ndarray) -> np.ndarray:
        out = np.empty_like(pts, dtype=np.float64)
        out[:, 0] = (pts[:, 0] - self.cx) * self.zoom + self.size[0] / 2
        out[:, 1] = self.size[1] / 2 - (pts[:, 1] - self.cy) * self.zoom
        return out

    def zoom_at(self, pos: tuple[float, float], factor: float) -> None:
        """Zoom keeping the world point under ``pos`` where it is."""
        wx, wy = self.to_world(*pos)
        self.zoom = clamp(self.zoom * factor, ZOOM_MIN, ZOOM_MAX)
        self.cx = wx - (pos[0] - self.size[0] / 2) / self.zoom
        self.cy = wy - (self.size[1] / 2 - pos[1]) / self.zoom

    def pan(self, dx: float, dy: float) -> None:
        """Move the view by a mouse displacement in pixels."""
        self.cx -= dx / self.zoom
        self.cy += dy / self.zoom

    def fit(self, bounds: tuple[float, float, float, float], margin: float = FIT_MARGIN) -> None:
        xmin, ymin, xmax, ymax = bounds
        span_x, span_y = max(xmax - xmin, 20.0), max(ymax - ymin, 20.0)
        w, h = self.size
        self.zoom = clamp(
            min((w - 2 * margin) / span_x, (h - 2 * margin) / span_y), ZOOM_MIN, ZOOM_MAX
        )
        self.cx, self.cy = (xmin + xmax) / 2, (ymin + ymax) / 2


# -- text prompt -------------------------------------------------------------------------------


class TextPrompt:
    """A one-line text field in the window (rename, save as)."""

    def __init__(
        self,
        label: str,
        text: str,
        on_accept: Callable[[str], None],
        max_length: int = 80,
    ) -> None:
        self.label = label
        self.text = text
        self.on_accept = on_accept
        self.max_length = max_length

    def insert(self, chars: str) -> None:
        chars = "".join(c for c in chars if c.isprintable())
        self.text = (self.text + chars)[: self.max_length]

    def backspace(self) -> None:
        self.text = self.text[:-1]


@dataclass
class Toast:
    text: str
    color: tuple[int, int, int]
    until: float


# -- the application -----------------------------------------------------------------------


def _wrap(font: pygame.font.Font, text: str, width: int) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in text.split(" "):
        trial = f"{line} {word}" if line else word
        if line and font.size(trial)[0] > width:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


class EditorApp:
    def __init__(
        self,
        spec: TrackSpec,
        path: Path | None = None,
        new: bool = False,
        size: tuple[int, int] | None = None,
        spawn: Callable[[Sequence[str]], subprocess.Popen] | None = None,
        tracks_dir: Path = TRACKS_DIR,
    ) -> None:
        enable_dpi_awareness()
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption(TITLE)
        self.screen = pygame.display.set_mode(size or self._default_size(), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.model = EditorModel(spec, saved=not new)
        self.path: Path | None = Path(path) if path else None
        self.tracks_dir = tracks_dir
        self.spawn = spawn or self._spawn
        self.view = Viewport()
        self.running = False
        self.selected: int | None = None
        self.hover: tuple | None = None  # ("corner", i) or ("edge", i, (x, y))
        self.mouse: tuple[int, int] | None = None
        self.space_down = False
        self.show_help = False
        self.prompt: TextPrompt | None = None
        self.toasts: list[Toast] = []
        self.quit_armed_until = -math.inf
        self.process: subprocess.Popen | None = None
        self.temp_dir: tempfile.TemporaryDirectory | None = None
        self._drag: dict | None = None
        self._redraw = True
        self._last_caption = ""
        self._cursor: int | None = None

        self.analysis: Analysis = analyze(spec, full=False)
        self.analysis_spec = spec
        self._stale = False  # the design changed since ``analysis``
        self._full_pending = not self.analysis.complete
        self._last_change = -math.inf
        self._last_quick_end = -math.inf
        self.quick_cost = 0.0  # seconds of the last quick analysis
        self.full_cost = 0.0  # seconds of the last complete validation
        self.draw_cost = 0.0
        self._geometry: tuple[Analysis, np.ndarray, np.ndarray] | None = None
        self._issue_rows: list[tuple[pygame.Rect, int]] = []
        self.banner_rect = pygame.Rect(0, 0, 0, 0)  # the verdict banner, as last drawn
        self._help_surface: pygame.Surface | None = None
        self._panel_cache: dict[tuple, pygame.Surface] = {}
        self._wrap_cache: dict[tuple, list[str]] = {}
        self._build_fonts()
        self._layout()
        self.fit_view()
        self.settle()
        if path and not new:
            self.say(f"Abierto {self._display_path(Path(path))}", "info")
        elif new:
            self.say("Circuito nuevo: clic en un borde para añadir esquinas", "info")

    # -- window ------------------------------------------------------------------------------

    @staticmethod
    def _default_size() -> tuple[int, int]:
        width, height = desktop_size()
        return min(1500, max(900, width - 48)), min(900, max(600, height - 96))

    def _build_fonts(self) -> None:
        self.ui = max(13, round(self.screen.get_height() / 54))
        self.font = pygame.font.Font(FONT_REGULAR, self.ui)
        self.bold = pygame.font.Font(FONT_BOLD, self.ui)
        self.big = pygame.font.Font(FONT_BOLD, round(self.ui * 1.25))
        self.small = pygame.font.Font(FONT_REGULAR, max(11, self.ui - 2))
        self.text = TextCache(self.font, 512)
        self.text_bold = TextCache(self.bold, 128)
        self.text_big = TextCache(self.big, 64)
        self.text_small = TextCache(self.small, 256)
        self._panel_cache.clear()
        self._wrap_cache.clear()
        self._help_surface = None

    def _layout(self) -> None:
        """Sizes that depend on the window: panel, canvas and the panel's buttons."""
        width, height = self.screen.get_size()
        self.scale = max(1.0, height / 800)
        self.panel_w = round(max(300, min(420, width * 0.28)) * max(1.0, self.ui / 15))
        self.panel_rect = pygame.Rect(width - self.panel_w, 0, self.panel_w, height)
        self.canvas = pygame.Rect(0, 0, width - self.panel_w, height)
        self.view.size = self.canvas.size
        self.line = self.font.get_linesize() + 2
        pad = round(14 * self.scale)
        self.pad = pad
        # Rows of the stats block; the buttons sit on the width and seed rows.
        self.stats_top = round(8 * self.scale) + self.line * 5 + pad
        self._row = {
            name: i
            for i, name in enumerate(("length", "radius", "corners", "width", "seed", "gap"))
        }
        btn = self.line
        right = self.panel_rect.right - pad
        self.buttons: dict[str, pygame.Rect] = {}
        for name in ("width", "seed"):
            y = self.stats_top + self._row[name] * self.line
            self.buttons[f"{name}-"] = pygame.Rect(right - 2 * btn - 4, y, btn, btn)
            self.buttons[f"{name}+"] = pygame.Rect(right - btn, y, btn, btn)

    def _ensure_size(self) -> None:
        surface = pygame.display.get_surface()
        if surface is None:
            return
        if surface is not self.screen or surface.get_size() != self.screen.get_size():
            self.screen = surface
            self._build_fonts()
            self._layout()
            self._redraw = True

    @staticmethod
    def _display_path(path: Path) -> str:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return str(path)

    # -- messages ----------------------------------------------------------------------------

    def say(self, text: str, kind: str = "info", seconds: float = 4.5) -> None:
        color = {"ok": VALID, "error": INVALID, "warn": PENDING}.get(kind, TEXT)
        self.toasts = [Toast(text, color, time.perf_counter() + seconds)]
        self._redraw = True

    @property
    def message(self) -> str | None:
        """The toast on screen, for tests and scripts."""
        return self.toasts[0].text if self.toasts else None

    # -- the design: analysis ----------------------------------------------------------------

    @property
    def spec(self) -> TrackSpec:
        return self.model.spec

    def changed(self, now: float | None = None) -> None:
        """Call after any change to the design: schedules a new analysis."""
        self._stale = True
        self._last_change = time.perf_counter() if now is None else now
        self._redraw = True
        if self.selected is not None and self.selected >= len(self.spec.corners):
            self.selected = None
        self._update_hover()

    def _run_quick(self) -> None:
        start = time.perf_counter()
        self.analysis = analyze(self.spec, full=False)
        self.analysis_spec = self.spec
        self._last_quick_end = time.perf_counter()
        self.quick_cost = self._last_quick_end - start
        self._stale = False
        self._full_pending = not self.analysis.complete
        self._redraw = True

    def _run_full(self) -> None:
        start = time.perf_counter()
        self.analysis = analyze(self.spec, full=True)
        self.analysis_spec = self.spec
        self.full_cost = time.perf_counter() - start
        self._stale = False
        self._full_pending = False
        self._redraw = True

    def update(self, now: float | None = None) -> None:
        """Run the analysis the design needs, as often as it can without stalling the UI.

        A quick analysis (a few ms) follows every change, throttled while dragging if it
        ever costs more than ``QUICK_BUDGET``. The complete validation (100-250 ms) waits
        until ``FULL_DEBOUNCE`` seconds pass without edits and never runs mid-drag.
        """
        now = time.perf_counter() if now is None else now
        dragging = self.model.dragging
        if self._stale:
            wait = self.quick_cost * 3 if dragging and self.quick_cost > QUICK_BUDGET else 0.0
            if now - self._last_quick_end >= wait:
                self._run_quick()
        idle = now - self._last_change >= FULL_DEBOUNCE
        if self._full_pending and not self._stale and not dragging and idle:
            self._run_full()
        self._poll_process()
        if self.toasts and time.perf_counter() >= self.toasts[0].until:
            self.toasts.clear()
            self._redraw = True

    def settle(self) -> Analysis:
        """Bring the analysis fully up to date now (save, test drive, tests, screenshots)."""
        if self._stale or not self.analysis.complete or self.analysis_spec != self.spec:
            self._run_full()
        return self.analysis

    @property
    def verdict(self) -> str:
        """``valid``, ``invalid`` or ``pending`` (nothing wrong so far, still checking)."""
        if self.analysis.issues:
            return "invalid"
        return "valid" if self.analysis.complete and not self._stale else "pending"

    # -- geometry cache -----------------------------------------------------------------------

    def _road_edges(self) -> tuple[np.ndarray, np.ndarray] | None:
        points = self.analysis.points
        if points is None:
            return None
        if self._geometry is None or self._geometry[0] is not self.analysis:
            tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
            norm = np.hypot(tangent[:, 0], tangent[:, 1])
            norm[norm == 0] = 1.0
            normal = np.column_stack([-tangent[:, 1], tangent[:, 0]]) / norm[:, None]
            half = self.analysis_spec.width / 2
            self._geometry = (self.analysis, points + normal * half, points - normal * half)
        return self._geometry[1], self._geometry[2]

    # -- picking ------------------------------------------------------------------------------

    def corner_screen(self, index: int) -> tuple[float, float]:
        x, y, _ = self.spec.corners[index]
        return self.view.to_screen(x, y)

    def pick_corner(self, pos: tuple[float, float]) -> int | None:
        best, best_d = None, (CORNER_PICK * self.scale) ** 2
        for i in range(len(self.spec.corners)):
            sx, sy = self.corner_screen(i)
            d = (sx - pos[0]) ** 2 + (sy - pos[1]) ** 2
            if d <= best_d:
                best, best_d = i, d
        return best

    def pick_edge(self, pos: tuple[float, float]) -> tuple[int, tuple[float, float]] | None:
        """The polygon edge under ``pos`` and the world point on it nearest to ``pos``."""
        corners = self.spec.corners
        n = len(corners)
        best, best_d = None, (EDGE_PICK * self.scale) ** 2
        for i in range(n):
            a = self.corner_screen(i)
            b = self.corner_screen((i + 1) % n)
            t, (px, py) = project_on_segment(pos, a, b)
            d = (px - pos[0]) ** 2 + (py - pos[1]) ** 2
            if d <= best_d:
                best, best_d = (i, self.view.to_world(px, py)), d
        return best

    def _update_hover(self) -> None:
        self.hover = None
        if self.mouse is None or not self.canvas.collidepoint(self.mouse) or self._drag:
            return
        corner = self.pick_corner(self.mouse)
        if corner is not None:
            self.hover = ("corner", corner)
            return
        edge = self.pick_edge(self.mouse)
        if edge is not None:
            self.hover = ("edge", edge[0], edge[1])

    # -- input: mouse -------------------------------------------------------------------------

    def on_mouse_down(self, pos: tuple[int, int], button: int, mods: int | None = None) -> None:
        self.mouse = pos
        self._redraw = True
        if self.prompt is not None or self._drag is not None:
            return  # one gesture at a time
        if self.panel_rect.collidepoint(pos):
            if button == 1:
                self._panel_click(pos)
            return
        if button == 2 or (button == 1 and self.space_down):
            self._drag = {"kind": "pan", "last": pos}
            return
        if button == 3:
            corner = self.pick_corner(pos)
            if corner is not None:
                self.delete_corner(corner)
            return
        if button != 1:
            return
        corner = self.pick_corner(pos)
        if corner is not None:
            self.selected = corner
            wx, wy = self.view.to_world(*pos)
            cx, cy, _ = self.spec.corners[corner]
            self.model.begin_drag()
            self._drag = {"kind": "corner", "index": corner, "offset": (cx - wx, cy - wy)}
            return
        edge = self.pick_edge(pos)
        if edge is not None:
            if len(self.spec.corners) >= MAX_CORNERS:
                self.say(f"Como máximo {MAX_CORNERS} esquinas", "warn")
                return
            index = self.model.insert_corner(edge[0], *edge[1])
            self.selected = index
            self._drag = {"kind": "corner", "index": index, "offset": (0.0, 0.0)}
            self.changed()
            return
        self.selected = None
        self._drag = {"kind": "pan", "last": pos}

    def on_mouse_motion(self, pos: tuple[int, int], mods: int | None = None) -> None:
        self.mouse = pos
        self._redraw = True
        drag = self._drag
        if drag is None:
            self._update_hover()
            return
        if drag["kind"] == "pan":
            self.view.pan(pos[0] - drag["last"][0], pos[1] - drag["last"][1])
            drag["last"] = pos
        elif drag["kind"] == "corner":
            wx, wy = self.view.to_world(*pos)
            x, y = wx + drag["offset"][0], wy + drag["offset"][1]
            mods = pygame.key.get_mods() if mods is None else mods
            if mods & pygame.KMOD_CTRL:
                x, y = round(x / SNAP_STEP) * SNAP_STEP, round(y / SNAP_STEP) * SNAP_STEP
            if self.model.move_corner(drag["index"], x, y):
                self.changed()

    def on_mouse_up(self, pos: tuple[int, int], button: int) -> None:
        self.mouse = pos
        drag, self._drag = self._drag, None
        if drag is not None and drag["kind"] == "corner":
            self.model.end_drag(time.perf_counter())
            self.changed()
        self._update_hover()
        self._redraw = True

    def on_wheel(self, pos: tuple[int, int], dy: float, mods: int | None = None) -> None:
        self.mouse = pos
        self._redraw = True
        if self.prompt is not None or not self.canvas.collidepoint(pos) or dy == 0:
            return
        corner = self.pick_corner(pos)
        if corner is not None:
            mods = pygame.key.get_mods() if mods is None else mods
            step = RADIUS_STEP * (5 if mods & pygame.KMOD_SHIFT else 1)
            self.selected = corner
            radius = self.spec.corners[corner][2] + step * (1 if dy > 0 else -1) * abs(
                round(dy) or 1
            )
            self.model.set_radius(corner, radius, time.perf_counter())
            self.changed()
        else:
            self.view.zoom_at(pos, ZOOM_WHEEL_FACTOR**dy)
            self._update_hover()

    def _panel_click(self, pos: tuple[int, int]) -> None:
        now = time.perf_counter()
        for name, rect in self.buttons.items():
            if rect.collidepoint(pos):
                sign = 1 if name.endswith("+") else -1
                if name.startswith("width"):
                    self.model.set_width(self.spec.width + sign * WIDTH_STEP, now)
                else:
                    self.model.set_seed(self.spec.decoration_seed + sign, now)
                self.changed()
                return
        for rect, index in self._issue_rows:
            if rect.collidepoint(pos):
                self.focus_issue(self.analysis.issues[index])
                return

    def focus_issue(self, issue: Issue) -> None:
        """Select the first corner an issue names and bring it into view."""
        if issue.corners:
            self.selected = issue.corners[0]
        if issue.points:
            sx, sy = self.view.to_screen(*issue.points[0])
            if not self.canvas.inflate(-120, -120).collidepoint(sx, sy):
                self.view.cx, self.view.cy = issue.points[0]
        self._redraw = True

    # -- editing actions ------------------------------------------------------------------------

    def delete_corner(self, index: int) -> None:
        if len(self.spec.corners) <= MIN_CORNERS:
            self.say(f"Hacen falta al menos {MIN_CORNERS} esquinas", "warn")
            return
        self.model.delete_corner(index, time.perf_counter())
        if self.selected == index:
            self.selected = None
        self.changed()

    def undo(self) -> None:
        if self.model.undo():
            self.changed()
            self.say("Deshecho", "info", 1.5)

    def redo(self) -> None:
        if self.model.redo():
            self.changed()
            self.say("Rehecho", "info", 1.5)

    def fit_view(self) -> None:
        xs = [c[0] for c in self.spec.corners]
        ys = [c[1] for c in self.spec.corners]
        self.view.fit((min(xs), min(ys), max(xs), max(ys)))
        self._update_hover()
        self._redraw = True

    def zoom_keys(self, factor: float) -> None:
        center = (self.canvas.width / 2, self.canvas.height / 2)
        self.view.zoom_at(center, factor)
        self._update_hover()

    # -- input: keyboard -----------------------------------------------------------------------

    def on_key(self, key: int, mods: int | None = None) -> None:
        self._redraw = True
        mods = pygame.key.get_mods() if mods is None else mods
        ctrl, shift = bool(mods & pygame.KMOD_CTRL), bool(mods & pygame.KMOD_SHIFT)
        now = time.perf_counter()
        if self.prompt is not None:
            self._prompt_key(key)
            return
        if self._drag is not None and key != pygame.K_ESCAPE:
            return  # the corner under the mouse must not change under it
        if key == pygame.K_ESCAPE:
            if self.show_help:
                self.show_help = False
            else:
                self.request_quit()
        elif key == pygame.K_h:
            self.show_help = not self.show_help
        elif ctrl and key == pygame.K_s:
            if shift:
                self.open_save_as()
            else:
                self.save()
        elif ctrl and key == pygame.K_z:
            self.redo() if shift else self.undo()
        elif ctrl and key == pygame.K_y:
            self.redo()
        elif key in (pygame.K_DELETE, pygame.K_BACKSPACE):
            index = self.hover[1] if self.hover and self.hover[0] == "corner" else self.selected
            if index is not None:
                self.delete_corner(index)
        elif key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
            self.zoom_keys(ZOOM_KEY_FACTOR)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.zoom_keys(1 / ZOOM_KEY_FACTOR)
        elif key == pygame.K_f:
            self.fit_view()
        elif key == pygame.K_PAGEUP:
            self.model.set_width(self.spec.width + WIDTH_STEP, now)
            self.changed()
        elif key == pygame.K_PAGEDOWN:
            self.model.set_width(self.spec.width - WIDTH_STEP, now)
            self.changed()
        elif key == pygame.K_d:
            self.model.set_seed(self.spec.decoration_seed + (-1 if shift else 1), now)
            self.changed()
        elif key == pygame.K_F2:
            self.open_rename()
        elif key in (pygame.K_F5, pygame.K_t):
            self.test_drive()
        elif key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN) and (
            self.selected is not None
        ):
            step = NUDGE_STEP * (10 if shift else 1)
            dx = step * ((key == pygame.K_RIGHT) - (key == pygame.K_LEFT))
            dy = step * ((key == pygame.K_UP) - (key == pygame.K_DOWN))
            self.model.nudge_corner(self.selected, dx, dy, now)
            self.changed()

    def on_key_up(self, key: int) -> None:
        if key == pygame.K_SPACE:
            self.space_down = False

    def handle_event(self, event: pygame.event.Event) -> None:
        kind = event.type
        if kind == pygame.QUIT:
            self.request_quit()
        elif kind == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and self.prompt is None:
                self.space_down = True
            self.on_key(event.key, event.mod)
        elif kind == pygame.KEYUP:
            self.on_key_up(event.key)
        elif kind == pygame.MOUSEBUTTONDOWN and event.button in (1, 2, 3):
            self.on_mouse_down(event.pos, event.button)
        elif kind == pygame.MOUSEBUTTONUP and event.button in (1, 2, 3):
            self.on_mouse_up(event.pos, event.button)
        elif kind == pygame.MOUSEMOTION:
            self.on_mouse_motion(event.pos)
        elif kind == pygame.MOUSEWHEEL:
            self.on_wheel(self.mouse or pygame.mouse.get_pos(), event.y)
        elif kind == pygame.TEXTINPUT and self.prompt is not None:
            self.prompt.insert(event.text)
            self._redraw = True
        elif kind in (pygame.VIDEORESIZE, pygame.WINDOWRESIZED, pygame.WINDOWEXPOSED):
            self._redraw = True

    def request_quit(self) -> None:
        now = time.perf_counter()
        if self.model.dirty and now > self.quit_armed_until:
            self.quit_armed_until = now + 4.0
            self.say(
                "Hay cambios sin guardar: pulsa Esc otra vez para salir sin guardar", "warn", 4.0
            )
            return
        self.running = False

    # -- prompts ------------------------------------------------------------------------------

    def open_rename(self) -> None:
        self.prompt = TextPrompt("Nombre del circuito", self.spec.name, self._accept_name, MAX_NAME)
        pygame.key.start_text_input()

    def open_save_as(self) -> None:
        suggestion = self._display_path(self.path) if self.path else slugify(self.spec.name)
        self.prompt = TextPrompt(
            "Guardar como (nombre o ruta .json)", suggestion, self._accept_save_as, 120
        )
        pygame.key.start_text_input()

    def _prompt_key(self, key: int) -> None:
        prompt = self.prompt
        if key == pygame.K_ESCAPE:
            self._close_prompt()
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._close_prompt()
            prompt.on_accept(prompt.text)
        elif key == pygame.K_BACKSPACE:
            prompt.backspace()

    def _close_prompt(self) -> None:
        self.prompt = None
        pygame.key.stop_text_input()
        self._redraw = True

    def _accept_name(self, text: str) -> None:
        name = text.strip()
        if not name:
            self.say("El nombre no puede estar vacío", "error")
        elif not name.isprintable():
            self.say("El nombre tiene caracteres no imprimibles", "error")
        else:
            self.model.set_name(name, time.perf_counter())
            self.changed()

    def _accept_save_as(self, text: str) -> None:
        if not text.strip():
            self.say("Escribe un nombre de fichero", "error")
            return
        self.save(resolve_save_path(text, self.tracks_dir))

    # -- saving -------------------------------------------------------------------------------

    def default_path(self) -> Path:
        return self.tracks_dir / f"{slugify(self.spec.name)}.json"

    def save(self, target: Path | None = None) -> bool:
        """Write the design if (and only if) it is valid. Atomic, UTF-8, LF."""
        analysis = self.settle()
        if analysis.issues:
            n = len(analysis.issues)
            self.say(
                f"No se guarda: el circuito no es válido ({n} problema{'s' if n != 1 else ''}). "
                "Corrige lo marcado en rojo",
                "error",
                6.0,
            )
            return False
        target = Path(target or self.path or self.default_path())
        own = self.path is not None and _same_file(target, self.path)
        if target.exists() and not own:
            self.say(
                f"{self._display_path(target)} ya existe: elige otro nombre con Ctrl+Mayús+S "
                "o ábrelo para editarlo",
                "error",
                6.0,
            )
            return False
        try:
            save_spec(target, self.spec)
        except OSError as error:
            self.say(f"No se puede escribir {target}: {error.strerror or error}", "error", 6.0)
            return False
        self.path = target
        self.model.mark_saved()
        self.say(f"Guardado {self._display_path(target)} ({analysis.report.length:.0f} m)", "ok")
        return True

    # -- test drive ---------------------------------------------------------------------------

    @staticmethod
    def _spawn(command: Sequence[str]) -> subprocess.Popen:
        return subprocess.Popen(command, cwd=REPO_ROOT)

    def drive_command(self, track_file: Path) -> list[str]:
        code = "from kartlab.show.drive import main; main()"
        return [sys.executable, "-c", code, "--track-file", str(track_file)]

    def test_drive(self) -> bool:
        """Open ``drive.py`` on the current design (a valid one) in another process.

        A design that is saved and unchanged is driven from its own file; otherwise from a
        temporary copy, so trying it never writes to ``tracks/``."""
        analysis = self.settle()
        if analysis.issues:
            self.say("Para probarlo el circuito tiene que ser válido", "error")
            return False
        if self.process is not None and self.process.poll() is None:
            self.say("Ya hay una prueba en marcha: ciérrala antes (Esc en el juego)", "warn")
            return False
        if self.path is not None and self.path.exists() and not self.model.dirty:
            target, note = self.path, ""
        else:
            self._drop_temp()
            self.temp_dir = tempfile.TemporaryDirectory(prefix="kartlab-editor-")
            target = Path(self.temp_dir.name) / f"{slugify(self.spec.name)}.json"
            save_spec(target, self.spec)
            note = " (copia temporal: aún sin guardar)"
        try:
            self.process = self.spawn(self.drive_command(target))
        except OSError as error:
            self.say(f"No se puede lanzar el juego: {error.strerror or error}", "error")
            return False
        self.say(f"Probando en el juego{note}", "ok")
        return True

    def _poll_process(self) -> None:
        if self.process is not None and self.process.poll() is not None:
            self.process = None
            self._drop_temp()

    def _drop_temp(self) -> None:
        if self.temp_dir is not None:
            with contextlib.suppress(OSError):
                self.temp_dir.cleanup()
            self.temp_dir = None

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return  # leave the game open; the temporary copy goes with the OS temp folder
        self._drop_temp()

    # -- main loop -----------------------------------------------------------------------------

    def run(self, max_frames: int | None = None) -> None:
        """Main loop. With ``max_frames`` (headless capture) it runs uncapped and exits."""
        self.running = True
        frames = 0
        while self.running:
            self.clock.tick(60 if max_frames is None else 0)
            for event in pygame.event.get():
                self.handle_event(event)
            self._ensure_size()
            self.update()
            if self._redraw or max_frames is not None:
                self.draw()
                pygame.display.flip()
                self._redraw = False
            self._set_cursor()
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
        self.close()

    def _set_cursor(self) -> None:
        if self._drag is not None:
            wanted = pygame.SYSTEM_CURSOR_SIZEALL
        elif self.hover is not None and self.hover[0] == "corner":
            wanted = pygame.SYSTEM_CURSOR_HAND
        elif self.hover is not None:
            wanted = pygame.SYSTEM_CURSOR_CROSSHAIR
        else:
            wanted = pygame.SYSTEM_CURSOR_ARROW
        if wanted != self._cursor:
            with contextlib.suppress(pygame.error):
                pygame.mouse.set_cursor(wanted)
            self._cursor = wanted

    # -- drawing: canvas -------------------------------------------------------------------------

    def draw(self) -> None:
        start = time.perf_counter()
        self._ensure_size()
        screen = self.screen
        screen.set_clip(self.canvas)
        self._draw_grid(screen)
        self._draw_road(screen)
        self._draw_polygon(screen)
        self._draw_issues(screen)
        self._draw_handles(screen)
        self._draw_canvas_hud(screen)
        screen.set_clip(None)
        self._draw_panel(screen)
        self._draw_toast(screen)
        if self.show_help:
            self._draw_help(screen)
        if self.prompt is not None:
            self._draw_prompt(screen)
        self._update_caption()
        self.draw_cost = time.perf_counter() - start

    def _update_caption(self) -> None:
        mark = "*" if self.model.dirty else ""
        caption = f"{self.spec.name}{mark} · {TITLE}"
        if caption != self._last_caption:
            pygame.display.set_caption(caption)
            self._last_caption = caption

    def _grid_step(self) -> float:
        for step in GRID_STEPS:
            if step * self.view.zoom >= GRID_MIN_PX * self.scale:
                return step
        return GRID_STEPS[-1]

    def _draw_grid(self, screen: pygame.Surface) -> None:
        screen.fill(GRASS, self.canvas)
        step = self._grid_step()
        w, h = self.canvas.size
        x0, y1 = self.view.to_world(0, 0)
        x1, y0 = self.view.to_world(w, h)
        major = GRASS_GRID
        for i in range(math.floor(x0 / step), math.ceil(x1 / step) + 1):
            sx = round(self.view.to_screen(i * step, 0)[0])
            pygame.draw.line(screen, major, (sx, 0), (sx, h))
        for j in range(math.floor(y0 / step), math.ceil(y1 / step) + 1):
            sy = round(self.view.to_screen(0, j * step)[1])
            pygame.draw.line(screen, major, (0, sy), (w, sy))
        # axes through the origin
        ox, oy = self.view.to_screen(0, 0)
        axis = (74, 128, 82)
        if 0 <= ox <= w:
            pygame.draw.line(screen, axis, (round(ox), 0), (round(ox), h))
        if 0 <= oy <= h:
            pygame.draw.line(screen, axis, (0, round(oy)), (w, round(oy)))

    def _draw_road(self, screen: pygame.Surface) -> None:
        edges = self._road_edges()
        if edges is None:
            # Not buildable (a corner does not fit, a crossing polygon...): hint the width
            # with the raw polygon so the design never vanishes under the mouse.
            pts = [self.corner_screen(i) for i in range(len(self.spec.corners))]
            width = max(2, round(self.spec.width * self.view.zoom))
            if len(pts) > 2:
                pygame.draw.lines(screen, (92, 62, 66), True, pts, width)
            return
        left = self.view.to_screen_array(edges[0])
        right = self.view.to_screen_array(edges[1])
        centre = self.view.to_screen_array(self.analysis.points)
        # A closed road is a ring: outer boundary, a bridge, the inner boundary backwards.
        # The scanline fill is even-odd, so the bridge cancels and the ring is filled.
        ring = np.concatenate([left, left[:1], right[:1], right[::-1]])
        pygame.draw.polygon(screen, ASPHALT, ring.tolist())
        bad = bool(self.analysis.issues)
        color = ROAD_EDGE_BAD if bad else EDGE
        edge_px = max(1, round(0.3 * self.view.zoom))
        pygame.draw.lines(screen, color, True, left.tolist(), edge_px)
        pygame.draw.lines(screen, color, True, right.tolist(), edge_px)
        pygame.draw.lines(screen, CENTER_MARK, True, centre[::6].tolist(), 1)
        self._draw_arrows(screen)
        self._draw_start_line(screen)

    def _draw_arrows(self, screen: pygame.Surface) -> None:
        points = self.analysis.points
        n = len(points)
        step = max(1, round(n / max(4, round(n / 90))))  # ~one arrow every 90 samples
        size = max(5.0, 2.4 * self.view.zoom)
        for i in range(step // 2, n, step):
            p, q = points[i], points[(i + 2) % n]
            heading = math.atan2(q[1] - p[1], q[0] - p[0])
            c, s = math.cos(heading), -math.sin(heading)  # screen y points down
            x, y = self.view.to_screen(float(p[0]), float(p[1]))
            tip = (x + c * size, y + s * size)
            back_l = (x - c * size - s * size, y - s * size + c * size)
            back_r = (x - c * size + s * size, y - s * size - c * size)
            pygame.draw.lines(screen, CENTER_MARK, False, [back_l, tip, back_r], 2)

    def _draw_start_line(self, screen: pygame.Surface) -> None:
        points = self.analysis.points
        p, q = points[0], points[1]
        heading = math.atan2(q[1] - p[1], q[0] - p[0])
        tx, ty = math.cos(heading), math.sin(heading)
        nx, ny = -ty, tx
        width = self.analysis_spec.width
        cols = max(2, round(width))
        square = width / cols
        for row in range(2):
            for col in range(cols):
                color = CHECKER_DARK if (row + col) % 2 else KERB_WHITE
                s0, s1 = (row - 1) * square, row * square
                l0 = -width / 2 + col * square
                l1 = l0 + square
                quad = [
                    self.view.to_screen(
                        float(p[0]) + tx * s + nx * lat, float(p[1]) + ty * s + ny * lat
                    )
                    for s, lat in ((s0, l0), (s1, l0), (s1, l1), (s0, l1))
                ]
                pygame.draw.polygon(screen, color, quad)

    def _draw_polygon(self, screen: pygame.Surface) -> None:
        pts = [self.corner_screen(i) for i in range(len(self.spec.corners))]
        if len(pts) < 2:
            return
        pygame.draw.lines(screen, KART_OUTLINE, True, pts, 3)
        pygame.draw.lines(screen, (206, 212, 226), True, pts, 1)
        for x, y in self.analysis_spec.crossings:  # declared level crossings (format 2)
            sx, sy = self.view.to_screen(x, y)
            r = round(9 * self.scale)
            pygame.draw.line(screen, ACCENT, (sx - r, sy - r), (sx + r, sy + r), 3)
            pygame.draw.line(screen, ACCENT, (sx - r, sy + r), (sx + r, sy - r), 3)

    def _draw_issues(self, screen: pygame.Surface) -> None:
        pulse = 0.5 + 0.5 * math.sin(time.perf_counter() * 5.0)
        for issue in self.analysis.issues:
            for a, b in issue.edges:
                n = len(self.spec.corners)
                if a < n and b < n:
                    pygame.draw.line(
                        screen, INVALID, self.corner_screen(a), self.corner_screen(b), 5
                    )
            if len(issue.points) == 2 and not issue.edges:
                pygame.draw.line(
                    screen, INVALID, *(self.view.to_screen(*p) for p in issue.points), 2
                )
            for p in issue.points:
                sx, sy = self.view.to_screen(*p)
                r = round((11 + 3 * pulse) * self.scale)
                pygame.draw.circle(screen, INVALID, (sx, sy), r, 3)
        if self.analysis.issues:
            self._redraw = True  # the rings pulse

    def _draw_handles(self, screen: pygame.Surface) -> None:
        flagged = {c for issue in self.analysis.issues for c in issue.corners}
        radius = round(7 * self.scale)
        hover_corner = self.hover[1] if self.hover and self.hover[0] == "corner" else None
        if self.hover and self.hover[0] == "edge":
            sx, sy = self.view.to_screen(*self.hover[2])
            pygame.draw.circle(screen, HANDLE_HOVER, (sx, sy), radius - 1, 2)
            pygame.draw.line(screen, HANDLE_HOVER, (sx - 3, sy), (sx + 3, sy), 2)
            pygame.draw.line(screen, HANDLE_HOVER, (sx, sy - 3), (sx, sy + 3), 2)
        for i, (x, y, r) in enumerate(self.spec.corners):
            sx, sy = self.view.to_screen(x, y)
            if i in flagged:
                fill = INVALID
            elif i == self.selected:
                fill = ACCENT
            elif i == hover_corner:
                fill = HANDLE_HOVER
            else:
                fill = HANDLE
            rad = radius + 2 if i == self.selected else radius
            pygame.draw.circle(screen, KART_OUTLINE, (sx, sy), rad + 2)
            pygame.draw.circle(screen, fill, (sx, sy), rad)
            label = self.text_small.get(str(i + 1), (255, 255, 255))
            shadow = self.text_small.get(str(i + 1), KART_OUTLINE)
            lx, ly = sx + rad + 4, sy - rad - label.get_height()
            screen.blit(shadow, (lx + 1, ly + 1))
            screen.blit(label, (lx, ly))
            if i in (self.selected, hover_corner):
                tag = self.text_small.get(f"radio {r:g} m", (255, 255, 255))
                box = tag.get_rect(topleft=(sx + rad + 6, sy + rad + 2)).inflate(8, 4)
                pygame.draw.rect(screen, PANEL_SOLID, box, border_radius=4)
                screen.blit(tag, (box.x + 4, box.y + 2))

    def _draw_canvas_hud(self, screen: pygame.Surface) -> None:
        # scale bar
        step = next((s for s in GRID_STEPS if s * self.view.zoom >= 70 * self.scale), 1000.0)
        length = step * self.view.zoom
        x, y = 18, self.canvas.height - 22
        for dx, dy in ((1, 1), (0, 0)):
            color = KART_OUTLINE if dx else TEXT
            pygame.draw.line(screen, color, (x + dx, y + dy), (x + length + dx, y + dy), 2)
            pygame.draw.line(screen, color, (x + dx, y - 5 + dy), (x + dx, y + 5 + dy), 2)
            pygame.draw.line(
                screen, color, (x + length + dx, y - 5 + dy), (x + length + dx, y + 5 + dy), 2
            )
        label = self.text_small.get(f"{step:g} m", TEXT)
        screen.blit(label, (x, y - label.get_height() - 4))
        if self.mouse is not None and self.canvas.collidepoint(self.mouse):
            wx, wy = self.view.to_world(*self.mouse)
            coords = self.text_small.get(f"x {wx:7.1f}   y {wy:7.1f} m", TEXT_DIM)
            screen.blit(
                coords, (self.canvas.right - coords.get_width() - 14, self.canvas.bottom - 28)
            )

    # -- drawing: panel --------------------------------------------------------------------------

    def _wrapped(self, font: pygame.font.Font, text: str, width: int) -> list[str]:
        key = (id(font), text, width)
        lines = self._wrap_cache.get(key)
        if lines is None:
            if len(self._wrap_cache) > 256:
                self._wrap_cache.clear()
            lines = self._wrap_cache[key] = _wrap(font, text, width)
        return lines

    def _draw_panel(self, screen: pygame.Surface) -> None:
        rect = self.panel_rect
        screen.fill(PANEL_SOLID, rect)
        pygame.draw.line(screen, PANEL_LINE, rect.topleft, rect.bottomleft)
        pad, x = self.pad, rect.x + self.pad
        inner = rect.width - 2 * pad
        y = round(8 * self.scale)
        screen.blit(self.text_big.get("Editor de circuitos", ACCENT), (x, y))
        y += self.big.get_linesize()
        dirty = self.model.dirty
        name = self.spec.name + (" *" if dirty else "")
        screen.blit(self.text_bold.get(self._fit(name, self.bold, inner), TEXT), (x, y))
        y += self.line
        where = self._display_path(self.path) if self.path else "sin guardar"
        screen.blit(self.text_small.get(self._fit(where, self.small, inner), TEXT_DIM), (x, y))
        y += self.line
        # verdict banner
        verdict = self.verdict
        n = len(self.analysis.issues)
        label, color = {
            "valid": ("CIRCUITO VÁLIDO", VALID),
            "invalid": (f"NO VÁLIDO · {n} problema{'s' if n != 1 else ''}", INVALID),
            "pending": ("COMPROBANDO…", PENDING),
        }[verdict]
        banner = self.banner_rect = pygame.Rect(x, y, inner, round(self.line * 1.5))
        pygame.draw.rect(screen, color, banner, border_radius=6)
        text = self.text_bold.get(label, (14, 16, 22))
        screen.blit(text, text.get_rect(center=banner.center))
        y = self.stats_top
        report = self.analysis.report
        spec = self.spec
        rows = [
            ("Longitud", f"{report.length:.0f} m" if report.length else "-", TEXT),
            (
                "Radio mínimo",
                f"{report.min_radius:.1f} m" if report.min_radius else "-",
                INVALID if report.min_radius and report.min_radius < MIN_RADIUS - 1e-6 else TEXT,
            ),
            ("Esquinas", str(len(spec.corners)), TEXT),
            ("Anchura", f"{spec.width:g} m", TEXT),
            ("Semilla decorado", str(spec.decoration_seed), TEXT),
            (
                "Hierba mínima",
                f"{report.gap:.1f} m" if report.gap not in (None, math.inf) else "-",
                TEXT,
            ),
        ]
        for i, (key, value, vcolor) in enumerate(rows):
            ry = y + i * self.line
            screen.blit(self.text.get(key, TEXT_DIM), (x, ry))
            has_buttons = key in ("Anchura", "Semilla decorado")
            right_edge = self.buttons["width-"].x - 8 if has_buttons else rect.right - pad
            val = self.text.get(value, vcolor)
            screen.blit(val, (right_edge - val.get_width(), ry))
        for name_, button in self.buttons.items():
            hot = self.mouse is not None and button.collidepoint(self.mouse)
            pygame.draw.rect(screen, (58, 66, 86) if hot else PANEL_LINE, button, border_radius=4)
            glyph = self.text_bold.get("+" if name_.endswith("+") else "−", TEXT)
            screen.blit(glyph, glyph.get_rect(center=button.center))
        y += len(rows) * self.line
        if spec.crossings:
            note = f"{len(spec.crossings)} cruce(s) a nivel declarados: se conservan al guardar"
            for line in self._wrapped(self.small, note, inner):
                screen.blit(self.text_small.get(line, ACCENT), (x, y))
                y += self.small.get_linesize()
        y += round(8 * self.scale)
        pygame.draw.line(screen, PANEL_LINE, (x, y), (rect.right - pad, y))
        y += round(8 * self.scale)
        self._draw_issue_list(screen, x, y, inner, rect.bottom - round(self.line * 3.4))
        hints = ("H ayuda · Ctrl+S guardar · F5 probar", "Ctrl+Z deshacer · Ctrl+Y rehacer")
        hy = rect.bottom - round(self.line * 2.2)
        for hint in hints:
            screen.blit(self.text_small.get(hint, TEXT_DIM), (x, hy))
            hy += self.small.get_linesize()

    def _draw_issue_list(
        self, screen: pygame.Surface, x: int, y: int, inner: int, limit: int
    ) -> None:
        self._issue_rows = []
        issues = self.analysis.issues
        if not issues:
            if self.verdict == "valid":
                text = "Nada que corregir. Guarda con Ctrl+S o pruébalo con F5."
            else:
                text = "Comprobando el circuito completo…"
            for line in self._wrapped(self.font, text, inner):
                screen.blit(self.text.get(line, TEXT_DIM), (x, y))
                y += self.line
            return
        screen.blit(self.text_bold.get("Qué falla", INVALID), (x, y))
        y += self.line
        for index, issue in enumerate(issues):
            lines = self._wrapped(self.small, issue.message, inner - 14)
            height = len(lines) * self.small.get_linesize() + 6
            if y + height > limit:
                rest = len(issues) - index
                more = self.text_small.get(f"… y {rest} más", TEXT_DIM)
                screen.blit(more, (x, y))
                break
            row = pygame.Rect(x - 4, y - 2, inner + 8, height)
            located = bool(issue.corners or issue.points)
            if self.mouse is not None and row.collidepoint(self.mouse) and located:
                pygame.draw.rect(screen, (40, 28, 32), row, border_radius=4)
            pygame.draw.circle(screen, INVALID, (x + 3, y + self.small.get_linesize() // 2 + 1), 3)
            ly = y
            for line in lines:
                screen.blit(self.text_small.get(line, (255, 178, 170)), (x + 14, ly))
                ly += self.small.get_linesize()
            self._issue_rows.append((row, index))
            y += height + 2

    @staticmethod
    def _fit(text: str, font: pygame.font.Font, width: int) -> str:
        if font.size(text)[0] <= width:
            return text
        while text and font.size("…" + text)[0] > width:
            text = text[1:]
        return "…" + text

    # -- drawing: overlays ---------------------------------------------------------------------

    def _draw_toast(self, screen: pygame.Surface) -> None:
        if not self.toasts:
            return
        toast = self.toasts[0]
        lines = self._wrapped(self.font, toast.text, self.canvas.width - 80)
        surfaces = [self.text.get(line, toast.color) for line in lines]
        width = max(s.get_width() for s in surfaces) + 28
        height = sum(s.get_height() for s in surfaces) + 16
        box = pygame.Rect(0, 0, width, height)
        box.midtop = (self.canvas.centerx, 14)
        shade = pygame.Surface(box.size, pygame.SRCALPHA)
        shade.fill(PANEL)
        screen.blit(shade, box)
        pygame.draw.rect(screen, toast.color, box, 2, border_radius=6)
        cy = box.y + 8
        for surface in surfaces:
            screen.blit(surface, (box.x + 14, cy))
            cy += surface.get_height()

    def _draw_help(self, screen: pygame.Surface) -> None:
        if self._help_surface is None or self._help_surface.get_size()[0] > screen.get_width():
            self._help_surface = self._render_help()
        surface = self._help_surface
        screen.blit(surface, surface.get_rect(center=self.canvas.center))

    def _render_help(self) -> pygame.Surface:
        pad = 18
        key_w = max(self.bold.size(k)[0] for k, _ in HELP_ROWS) + 24
        desc_w = max(self.font.size(d)[0] for _, d in HELP_ROWS)
        title = self.big.get_linesize() + 8
        width = pad * 2 + key_w + desc_w
        height = pad * 2 + title + len(HELP_ROWS) * self.line + self.line
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        surface.fill((14, 16, 22, 236))
        pygame.draw.rect(surface, ACCENT, surface.get_rect(), 2, border_radius=8)
        surface.blit(self.text_big.get("Ayuda del editor", ACCENT), (pad, pad))
        y = pad + title
        for key, desc in HELP_ROWS:
            surface.blit(self.text_bold.get(key, TEXT), (pad, y))
            surface.blit(self.text.get(desc, TEXT_DIM), (pad + key_w, y))
            y += self.line
        surface.blit(self.text_small.get("H o Esc para cerrar", TEXT_DIM), (pad, y + 4))
        return surface

    def _draw_prompt(self, screen: pygame.Surface) -> None:
        prompt = self.prompt
        width = min(self.canvas.width - 60, 640)
        height = self.line * 4
        box = pygame.Rect(0, 0, width, height)
        box.center = self.canvas.center
        pygame.draw.rect(screen, (14, 16, 22), box, border_radius=8)
        pygame.draw.rect(screen, ACCENT, box, 2, border_radius=8)
        screen.blit(self.text_bold.get(prompt.label, ACCENT), (box.x + 16, box.y + 10))
        field = pygame.Rect(box.x + 16, box.y + self.line + 12, width - 32, self.line + 4)
        pygame.draw.rect(screen, PANEL_SOLID, field, border_radius=4)
        shown = self._fit(prompt.text, self.font, field.width - 16)
        text = self.font.render(shown, True, TEXT)
        screen.blit(text, (field.x + 8, field.y + 2))
        if int(time.perf_counter() * 2) % 2 == 0:
            cx = field.x + 8 + text.get_width() + 2
            pygame.draw.line(screen, TEXT, (cx, field.y + 4), (cx, field.bottom - 4), 2)
        hint = self.text_small.get("Intro: aceptar · Esc: cancelar", TEXT_DIM)
        screen.blit(hint, (box.x + 16, box.bottom - hint.get_height() - 6))
        self._redraw = True  # the caret blinks

    def save_screenshot(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.screen, str(path))


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a == b


def parse_size(text: str) -> tuple[int, int]:
    try:
        width, height = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "usa el formato ANCHOxALTO, por ejemplo 1280x800"
        ) from None
    if not (640 <= width <= 4096 and 480 <= height <= 2304):
        raise argparse.ArgumentTypeError("el tamaño debe estar entre 640x480 y 4096x2304")
    return width, height


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Editor de circuitos con ratón.")
    parser.add_argument("track", nargs="?", type=Path, help="circuito .json para editar")
    parser.add_argument(
        "--new",
        action="store_true",
        help="empieza un circuito nuevo; con TRACK, ese será el fichero (no puede existir)",
    )
    parser.add_argument("--size", type=parse_size, help="tamaño de la ventana, ANCHOxALTO")
    parser.add_argument(
        "--frames", type=int, help="sale tras N fotogramas (para capturas y pruebas sin ventana)"
    )
    parser.add_argument("--screenshot", type=Path, help="guarda el último fotograma en este PNG")
    args = parser.parse_args(argv)

    spec, path, new = NEW_SPEC, args.track, True
    if args.track is not None and not args.new:
        try:
            spec = load_spec(args.track)
        except TrackFileError as error:
            raise SystemExit(f"No se puede abrir el circuito:\n{error}") from None
        new = False
    elif args.track is not None and args.track.exists():
        raise SystemExit(
            f"{args.track} ya existe: quita --new para editarlo (no se sobrescribe un circuito)"
        )
    app = EditorApp(spec, path, new=new, size=args.size)
    try:
        app.run(max_frames=args.frames)
        if args.screenshot:
            app.settle()
            app.draw()
            app.save_screenshot(args.screenshot)
    finally:
        pygame.quit()
