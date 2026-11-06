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
import threading
import time
import unicodedata
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pygame

from kartlab.common.paths import REPO_ROOT
from kartlab.render import pixelfont
from kartlab.render import pixelui as ui
from kartlab.render.display import (
    desktop_size,
    enable_dpi_awareness,
    pick_scale,
    present,
    to_logical,
)
from kartlab.render.palette import Ink
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

TITLE = "KART LAB: editor de circuitos"
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

# View (in canvas pixels: the editor is drawn on the 320x180 grid and magnified like the world)
ZOOM_MIN = 0.03  # px per m
ZOOM_MAX = 6.0
ZOOM_KEY_FACTOR = 1.25
ZOOM_WHEEL_FACTOR = 1.15
CORNER_PICK = 4.0  # px
EDGE_PICK = 3.0  # px
FIT_MARGIN = 14  # px
GRID_STEPS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
GRID_MIN_PX = 16.0  # smallest on-screen grid spacing
SCALE_BAR_PX = 24.0  # the scale bar is at least this long

# Analysis scheduling (see ``EditorApp.update``)
QUICK_BUDGET = 0.012  # s; a drag rebuilds as often as the frames allow while this holds
FULL_DEBOUNCE = 0.12  # s without edits before the complete (slow) validation runs

# Layout of the 320x180 canvas
PANEL_W = 100
ROW = 8  # pitch of the stats and issue rows
PAD = 3

# Colours: palette inks, as RGB where a test or the scene needs the value
GRASS = ui.rgb(Ink.GRASS_DARK)
VALID = ui.rgb(ui.GREEN)
INVALID = ui.rgb(ui.RED)
PENDING = ui.rgb(ui.YELLOW)
TOAST_INKS = {"ok": ui.GREEN, "error": ui.RED, "warn": ui.YELLOW, "info": ui.TEXT}
TOAST_ICONS = {"ok": "check", "error": "cross", "warn": "warn"}

HELP_ROWS: tuple[tuple[str, str], ...] = (
    ("Arrastrar esquina", "moverla (Ctrl: a 5 m)"),
    ("Clic en un borde", "insertar una esquina"),
    ("Clic der. / Supr", "borrar esquina (mín. 3)"),
    ("Rueda sobre esquina", "radio ±2 m (Mayús ±10 m)"),
    ("Rueda, + / -", "zoom"),
    ("Botón central / Espacio", "mover la vista"),
    ("F", "encuadrar el circuito"),
    ("Flechas", "mover esquina 1 m (Mayús 10)"),
    ("Re Pág / Av Pág", "anchura del asfalto ±1 m"),
    ("D / Mayús + D", "cambiar el decorado"),
    ("F2", "cambiar el nombre"),
    ("Ctrl + Z / Ctrl + Y", "deshacer / rehacer"),
    ("Ctrl + S", "guardar (si es válido)"),
    ("Ctrl + Mayús + S", "guardar como"),
    ("F5 / T", "probar conduciendo"),
    ("H", "ayuda"),
    ("Esc", "salir"),
)

HINTS = ("H  AYUDA", "F5  PROBAR", "CTRL+S  GUARDAR")


# -- pure helpers ----------------------------------------------------------------------------


def slugify(name: str) -> str:
    """File name stem for a track name: ``Bahía del Faro`` -> ``bahia-del-faro``."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    stem = "".join(c.lower() if c.isalnum() else "-" for c in ascii_name)
    return "-".join(part for part in stem.split("-") if part) or "circuito"


def resolve_save_path(text: str, tracks_dir: Path = TRACKS_DIR, root: Path = REPO_ROOT) -> Path:
    """The file a "save as" answer means: a bare name goes to ``tracks/<slug>.json``;
    anything with a folder or a ``.json`` suffix is a path (relative to the project root).

    Raises ``ValueError`` (Spanish message) for a path that leaves the project, so a
    typo such as ``../x`` can never write elsewhere on the disk."""
    text = text.strip()
    if not ("/" in text or "\\" in text or text.lower().endswith(".json")):
        return tracks_dir / f"{slugify(text)}.json"
    if text.endswith(("/", "\\")) or Path(text).name.lower() == ".json":
        raise ValueError("Falta el nombre del fichero")
    path = Path(text)
    path = path if path.suffix else path.with_suffix(".json")
    path = path if path.is_absolute() else root / path
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Solo se guarda dentro del proyecto")
    return path


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


# -- background validation -----------------------------------------------------------------


class FullValidator:
    """Runs the complete (slow) validation in one worker thread, off the UI thread.

    The newest request wins: asking while it is busy replaces whatever was waiting, and the
    caller drops a result whose spec is no longer the current one. ``TrackSpec`` is
    immutable, so sharing it between threads is safe, and results are only ever read on the
    UI thread (``take``). The thread is a daemon and starts on the first request.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._request: TrackSpec | None = None
        self._result: tuple[TrackSpec, Analysis | None, float] | None = None
        self._thread: threading.Thread | None = None
        self._stop = False

    def submit(self, spec: TrackSpec) -> None:
        with self._lock:
            self._request = spec
        if self._thread is None:
            self._thread = threading.Thread(target=self._work, name="editor-validator", daemon=True)
            self._thread.start()
        self._wake.set()

    def take(self) -> tuple[TrackSpec, Analysis | None, float] | None:
        """The finished validation (spec, analysis or None if it failed, seconds), once."""
        with self._lock:
            result, self._result = self._result, None
        return result

    def close(self) -> None:
        self._stop = True
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _work(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stop:
                return
            with self._lock:
                spec, self._request = self._request, None
            if spec is None:
                continue
            start = time.perf_counter()
            try:
                analysis = analyze(spec, full=True)
            except Exception:  # never kill the worker: the UI thread falls back to its own run
                analysis = None
            with self._lock:
                self._result = (spec, analysis, time.perf_counter() - start)


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
        self.selected = bool(text)  # the prefilled text is selected: typing replaces it

    @property
    def full(self) -> bool:
        return len(self.text) >= self.max_length

    def insert(self, chars: str) -> None:
        chars = "".join(c for c in chars if c.isprintable())
        if chars and self.selected:
            self.text = ""
        if chars:
            self.selected = False
        self.text = (self.text + chars)[: self.max_length]

    def backspace(self) -> None:
        self.selected = False  # Backspace edits the text as it is
        self.text = self.text[:-1]


@dataclass
class Toast:
    text: str
    kind: str
    until: float


# -- the application -----------------------------------------------------------------------


class EditorApp:
    def __init__(
        self,
        spec: TrackSpec,
        path: Path | None = None,
        new: bool = False,
        size: tuple[int, int] | None = None,
        spawn: Callable[[Sequence[str]], subprocess.Popen] | None = None,
        tracks_dir: Path = TRACKS_DIR,
        root: Path = REPO_ROOT,
        background: bool = True,
    ) -> None:
        enable_dpi_awareness()
        pygame.display.init()
        pygame.display.set_caption(TITLE)
        self.screen = pygame.display.set_mode(size or self._default_size(), pygame.RESIZABLE)
        self.surface = pygame.Surface(
            ui.CANVAS_SIZE
        )  # what is drawn; the window shows it magnified
        self.clock = pygame.time.Clock()
        self.model = EditorModel(spec, saved=not new)
        self.path: Path | None = Path(path) if path else None
        self._shown_path: tuple[Path, str] | None = None
        self.tracks_dir = tracks_dir
        self.root = root  # "save as" never writes outside it
        self.spawn = spawn or self._spawn
        self.view = Viewport()
        self.running = False
        self.quit_requested = False  # the window was closed (the main menu closes too)
        self.selected: int | None = None
        self.hover: tuple | None = None  # ("corner", i) or ("edge", i, (x, y))
        self.mouse: tuple[int, int] | None = None
        self.space_down = False
        self.show_help = False
        self.prompt: TextPrompt | None = None
        self.toasts: list[Toast] = []
        self.quit_armed_until = -math.inf
        self._armed_from_window = False  # the pending "unsaved changes" warning came from the X
        self.process: subprocess.Popen | None = None
        self.temp_dir: tempfile.TemporaryDirectory | None = None
        self._drag: dict | None = None
        self._redraw = True
        self._last_caption = ""
        self._cursor: int | None = None

        self.validator = FullValidator() if background else None
        self._submitted: TrackSpec | None = None  # the spec the worker was last asked about
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
        self._layout()
        self.fit_view()
        self.settle()
        if path and not new:
            self.say(f"Abierto {self._toast_path(Path(path))}", "info")
        elif new:
            self.say("Clic en un borde para añadir esquinas", "info")

    # -- window ------------------------------------------------------------------------------

    @staticmethod
    def _default_size() -> tuple[int, int]:
        k = pick_scale(desktop_size())
        return 320 * k, 180 * k

    def _layout(self) -> None:
        """Panel, canvas and the panel's buttons, in canvas pixels."""
        width, height = ui.CANVAS_SIZE
        self.panel_rect = pygame.Rect(width - PANEL_W, 0, PANEL_W, height)
        self.canvas = pygame.Rect(0, 0, width - PANEL_W, height)
        self.view.size = self.canvas.size
        self.stats_top = 47
        self._row = {
            name: i
            for i, name in enumerate(("length", "radius", "corners", "width", "seed", "gap"))
        }
        right = self.panel_rect.right - PAD
        self.buttons: dict[str, pygame.Rect] = {}
        for name in ("width", "seed"):
            y = self.stats_top + self._row[name] * ROW - 1
            self.buttons[f"{name}-"] = pygame.Rect(right - 19, y, 9, ROW + 1)
            self.buttons[f"{name}+"] = pygame.Rect(right - 9, y, 9, ROW + 1)

    def _ensure_size(self) -> None:
        surface = pygame.display.get_surface()
        if surface is None:
            return
        if surface is not self.screen or surface.get_size() != self.screen.get_size():
            self.screen = surface
            self._redraw = True

    def _logical(self, pos: tuple[int, int]) -> tuple[int, int]:
        """The canvas pixel under a window position."""
        return to_logical(pos, self.screen.get_size())

    def _path_text(self) -> str:
        """The path shown in the panel; resolving it is slow, so it is done once per path."""
        if self._shown_path is None or self._shown_path[0] != self.path:
            self._shown_path = (self.path, self._display_path(self.path))
        return self._shown_path[1]

    @staticmethod
    def _display_path(path: Path) -> str:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return str(path)

    @staticmethod
    def _toast_path(path: Path) -> str:
        """A path for a toast: inside the project as is, anywhere else just the file name."""
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.name

    # -- messages ----------------------------------------------------------------------------

    def say(self, text: str, kind: str = "info", seconds: float = 4.5) -> None:
        self.toasts = [Toast(text, kind, time.perf_counter() + seconds)]
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
        # A quick analysis replaces whatever the worker validated, even for the very same
        # design (undo, a no-op change): the next request must not be skipped as a repeat.
        self._submitted = None
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
        until ``FULL_DEBOUNCE`` seconds pass without edits, never starts mid-drag and runs
        in the ``FullValidator`` thread (``settle`` runs it synchronously instead).
        """
        now = time.perf_counter() if now is None else now
        dragging = self.model.dragging
        if self._stale:
            wait = self.quick_cost * 3 if dragging and self.quick_cost > QUICK_BUDGET else 0.0
            if now - self._last_quick_end >= wait:
                self._run_quick()
        self._apply_background_result()
        idle = now - self._last_change >= FULL_DEBOUNCE
        if self._full_pending and not self._stale and not dragging and idle:
            if self.validator is None:
                self._run_full()
            elif self._submitted != self.spec:
                self._submitted = self.spec
                self.validator.submit(self.spec)
        self._poll_process()
        if self.toasts and time.perf_counter() >= self.toasts[0].until:
            self.toasts.clear()
            self._redraw = True

    def _apply_background_result(self) -> None:
        """Take the worker's finished validation if it still answers the current design."""
        if self.validator is None or (result := self.validator.take()) is None:
            return
        spec, analysis, cost = result
        if spec != self.spec or self._stale or not self._full_pending:
            self._submitted = None  # an old answer: ask again if it is still needed
            return
        if analysis is None:
            self._run_full()  # the worker failed: run it here, where the error is visible
            return
        self.analysis, self.analysis_spec, self.full_cost = analysis, spec, cost
        self._full_pending = False
        self._submitted = None
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
        best, best_d = None, CORNER_PICK**2
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
        best, best_d = None, EDGE_PICK**2
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
                issue = self.analysis.issues[index]
                self.focus_issue(issue)
                self.say(issue.message, "error", 8.0)  # the list cuts long messages: show it all
                return

    def focus_issue(self, issue: Issue) -> None:
        """Select the first corner an issue names and bring it into view."""
        if issue.corners:
            self.selected = issue.corners[0]
        if issue.points:
            sx, sy = self.view.to_screen(*issue.points[0])
            if not self.canvas.inflate(-30, -30).collidepoint(sx, sy):
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
            self.request_quit(from_window=True)
        elif kind == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and self.prompt is None:
                self.space_down = True
            self.on_key(event.key, event.mod)
        elif kind == pygame.KEYUP:
            self.on_key_up(event.key)
        elif kind == pygame.MOUSEBUTTONDOWN and event.button in (1, 2, 3):
            self.on_mouse_down(self._logical(event.pos), event.button)
        elif kind == pygame.MOUSEBUTTONUP and event.button in (1, 2, 3):
            self.on_mouse_up(self._logical(event.pos), event.button)
        elif kind == pygame.MOUSEMOTION:
            self.on_mouse_motion(self._logical(event.pos))
        elif kind == pygame.MOUSEWHEEL:
            self.on_wheel(self.mouse or self._logical(pygame.mouse.get_pos()), event.y)
        elif kind == pygame.TEXTINPUT and self.prompt is not None:
            self.prompt.insert(event.text)
            self._redraw = True
        elif kind in (pygame.VIDEORESIZE, pygame.WINDOWRESIZED, pygame.WINDOWEXPOSED):
            self._redraw = True

    def request_quit(self, from_window: bool = False) -> None:
        """Esc or the window's close button. With unsaved changes the first request only warns;
        the second one (within a few seconds) leaves. ``quit_requested`` (the main menu closes
        too) is set when the player asked with the window's button, also if the second request
        that discarded the changes was an Esc."""
        now = time.perf_counter()
        if self.model.dirty and now > self.quit_armed_until:
            self.quit_armed_until = now + 4.0
            self._armed_from_window = from_window
            self.say("Sin guardar. Esc otra vez para salir", "warn", 4.0)
            return
        armed = self._armed_from_window and self.model.dirty and now <= self.quit_armed_until
        if from_window or armed:
            self.quit_requested = True
        self.running = False

    # -- prompts ------------------------------------------------------------------------------

    def open_rename(self) -> None:
        self.prompt = TextPrompt("Nombre del circuito", self.spec.name, self._accept_name, MAX_NAME)
        pygame.key.start_text_input()

    def open_save_as(self) -> None:
        suggestion = self._display_path(self.path) if self.path else slugify(self.spec.name)
        self.prompt = TextPrompt("Guardar como", suggestion, self._accept_save_as, 120)
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
            self.say("Nombre no válido", "error")
        else:
            self.model.set_name(name, time.perf_counter())
            self.changed()

    def _accept_save_as(self, text: str) -> None:
        if not text.strip():
            self.say("Escribe un nombre de fichero", "error")
            return
        if self.path is not None and _same_file(Path(text.strip()), self.path):
            target = self.path  # the file being edited is always allowed
        else:
            try:
                target = resolve_save_path(text, self.tracks_dir, self.root)
            except ValueError as error:
                self.say(str(error), "error", 6.0)
                return
        self.save(target)

    # -- saving -------------------------------------------------------------------------------

    def default_path(self) -> Path:
        return self.tracks_dir / f"{slugify(self.spec.name)}.json"

    def save(self, target: Path | None = None) -> bool:
        """Write the design if (and only if) it is valid. Atomic, UTF-8, LF."""
        analysis = self.settle()
        if analysis.issues:
            n = len(analysis.issues)
            self.say(
                f"No se guarda: hay {n} problema{'s' if n != 1 else ''}",
                "error",
                6.0,
            )
            return False
        target = Path(target or self.path or self.default_path())
        own = self.path is not None and _same_file(target, self.path)
        if target.exists() and not own:
            self.say(
                f"Ya existe {self._toast_path(target)}. Usa otro nombre",
                "error",
                6.0,
            )
            return False
        try:
            save_spec(target, self.spec)
        except OSError as error:
            self.say(f"No se pudo escribir {target}: {error.strerror or error}", "error", 6.0)
            return False
        self.path = target
        self.model.mark_saved()
        length = ui.fmt_count(round(analysis.report.length))
        self.say(f"Guardado {self._toast_path(target)} ({length} m)", "ok")
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
            self.say("Corrige los problemas para probarlo", "error")
            return False
        if self.process is not None and self.process.poll() is None:
            self.say("Ya hay una prueba abierta", "warn")
            return False
        if self.path is not None and self.path.exists() and not self.model.dirty:
            target, note = self.path, ""
        else:
            self._drop_temp()
            self.temp_dir = tempfile.TemporaryDirectory(prefix="kartlab-editor-")
            target = Path(self.temp_dir.name) / f"{slugify(self.spec.name)}.json"
            save_spec(target, self.spec)
            note = " (copia sin guardar)"
        try:
            self.process = self.spawn(self.drive_command(target))
        except OSError as error:
            self.say(f"No se pudo lanzar el juego: {error.strerror or error}", "error")
            return False
        self.say(f"Probando{note}", "ok")
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
        if self.validator is not None:
            self.validator.close()
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
        """Draw the editor on the 320x180 canvas and show it magnified in the window."""
        start = time.perf_counter()
        self._ensure_size()
        surface = self.surface
        surface.set_clip(self.canvas)
        self._draw_grid(surface)
        self._draw_road(surface)
        self._draw_polygon(surface)
        self._draw_issues(surface)
        self._draw_handles(surface)
        self._draw_canvas_hud(surface)
        surface.set_clip(None)
        self._draw_panel(surface)
        self._draw_toast(surface)
        if self.show_help:
            self._draw_help(surface)
        if self.prompt is not None:
            self._draw_prompt(surface)
        present(surface, self.screen)
        self._update_caption()
        self.draw_cost = time.perf_counter() - start

    def _update_caption(self) -> None:
        mark = "*" if self.model.dirty else ""
        caption = f"{self.spec.name}{mark}, {TITLE}"
        if caption != self._last_caption:
            pygame.display.set_caption(caption)
            self._last_caption = caption

    def _grid_step(self) -> float:
        for step in GRID_STEPS:
            if step * self.view.zoom >= GRID_MIN_PX:
                return step
        return GRID_STEPS[-1]

    def _draw_grid(self, surface: pygame.Surface) -> None:
        surface.fill(GRASS, self.canvas)
        step = self._grid_step()
        w, h = self.canvas.size
        x0, y1 = self.view.to_world(0, 0)
        x1, y0 = self.view.to_world(w, h)
        for i in range(math.floor(x0 / step), math.ceil(x1 / step) + 1):
            sx = round(self.view.to_screen(i * step, 0)[0])
            ui.vline(surface, sx, 0, h, Ink.MOSS_DARK)
        for j in range(math.floor(y0 / step), math.ceil(y1 / step) + 1):
            sy = round(self.view.to_screen(0, j * step)[1])
            ui.hline(surface, 0, sy, w, Ink.MOSS_DARK)
        # axes through the origin
        ox, oy = self.view.to_screen(0, 0)
        if 0 <= ox <= w:
            ui.vline(surface, round(ox), 0, h, Ink.MOSS)
        if 0 <= oy <= h:
            ui.hline(surface, 0, round(oy), w, Ink.MOSS)

    def _draw_road(self, surface: pygame.Surface) -> None:
        edges = self._road_edges()
        if edges is None:
            # Not buildable (a corner does not fit, a crossing polygon...): hint the width
            # with the raw polygon so the design never vanishes under the mouse.
            pts = [self.corner_screen(i) for i in range(len(self.spec.corners))]
            width = max(2, round(self.spec.width * self.view.zoom))
            if len(pts) > 2:
                pygame.draw.lines(surface, ui.rgb(Ink.ASPHALT), True, pts, width)
            return
        left = self.view.to_screen_array(edges[0])
        right = self.view.to_screen_array(edges[1])
        centre = self.view.to_screen_array(self.analysis.points)
        # A closed road is a ring: outer boundary, a bridge, the inner boundary backwards.
        # The scanline fill is even-odd, so the bridge cancels and the ring is filled.
        ring = np.concatenate([left, left[:1], right[:1], right[::-1]])
        pygame.draw.polygon(surface, ui.rgb(Ink.ASPHALT_LIGHT), ring.tolist())
        edge = ui.rgb(ui.RED if self.analysis.issues else ui.TEXT)
        pygame.draw.lines(surface, edge, True, left.tolist())
        pygame.draw.lines(surface, edge, True, right.tolist())
        pygame.draw.lines(surface, ui.rgb(ui.STONE), True, centre[::6].tolist())
        self._draw_arrows(surface)
        self._draw_start_line(surface)

    def _draw_arrows(self, surface: pygame.Surface) -> None:
        points = self.analysis.points
        n = len(points)
        step = max(1, round(n / max(4, round(n / 90))))  # ~one arrow every 90 samples
        size = max(2.5, 1.2 * self.view.zoom)
        ink = ui.rgb(ui.STONE_LIGHT)
        for i in range(step // 2, n, step):
            p, q = points[i], points[(i + 2) % n]
            heading = math.atan2(q[1] - p[1], q[0] - p[0])
            c, s = math.cos(heading), -math.sin(heading)  # screen y points down
            x, y = self.view.to_screen(float(p[0]), float(p[1]))
            tip = (x + c * size, y + s * size)
            back_l = (x - c * size - s * size, y - s * size + c * size)
            back_r = (x - c * size + s * size, y - s * size - c * size)
            pygame.draw.lines(surface, ink, False, [back_l, tip, back_r])

    def _draw_start_line(self, surface: pygame.Surface) -> None:
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
                ink = ui.NIGHT if (row + col) % 2 else ui.WHITE
                s0, s1 = (row - 1) * square, row * square
                l0 = -width / 2 + col * square
                l1 = l0 + square
                quad = [
                    self.view.to_screen(
                        float(p[0]) + tx * s + nx * lat, float(p[1]) + ty * s + ny * lat
                    )
                    for s, lat in ((s0, l0), (s1, l0), (s1, l1), (s0, l1))
                ]
                pygame.draw.polygon(surface, ui.rgb(ink), quad)

    def _draw_polygon(self, surface: pygame.Surface) -> None:
        pts = [self.corner_screen(i) for i in range(len(self.spec.corners))]
        if len(pts) < 2:
            return
        pygame.draw.lines(surface, ui.rgb(ui.CYAN), True, pts)
        for x, y in self.analysis_spec.crossings:  # declared level crossings (format 2)
            sx, sy = self.view.to_screen(x, y)
            ui.icon(surface, "cross", round(sx) - 2, round(sy) - 2, ui.YELLOW, outline=ui.NIGHT)

    def _draw_issues(self, surface: pygame.Surface) -> None:
        red = ui.rgb(ui.RED)
        pulse = round(0.5 + 0.5 * math.sin(time.perf_counter() * 5.0))  # a step, not a glide
        n = len(self.spec.corners)
        for issue in self.analysis.issues:
            for a, b in issue.edges:
                if a < n and b < n:
                    pygame.draw.line(surface, red, self.corner_screen(a), self.corner_screen(b), 2)
            if len(issue.points) == 2 and not issue.edges:
                pygame.draw.line(surface, red, *(self.view.to_screen(*p) for p in issue.points))
            for p in issue.points:
                sx, sy = self.view.to_screen(*p)
                pygame.draw.circle(surface, red, (round(sx), round(sy)), 6 + pulse, 1)
        if self.analysis.issues:
            self._redraw = True  # the rings pulse

    def _draw_handles(self, surface: pygame.Surface) -> None:
        flagged = {c for issue in self.analysis.issues for c in issue.corners}
        hover_corner = self.hover[1] if self.hover and self.hover[0] == "corner" else None
        night = ui.rgb(ui.NIGHT)
        if self.hover and self.hover[0] == "edge":  # a plus where a click inserts a corner
            sx, sy = (round(v) for v in self.view.to_screen(*self.hover[2]))
            pygame.draw.circle(surface, night, (sx, sy), 4)
            pygame.draw.circle(surface, ui.rgb(ui.YELLOW), (sx, sy), 3)
            ui.hline(surface, sx - 2, sy, 5, ui.NIGHT)
            ui.vline(surface, sx, sy - 2, 5, ui.NIGHT)
        for i, (x, y, r) in enumerate(self.spec.corners):
            sx, sy = (round(v) for v in self.view.to_screen(x, y))
            if i in flagged:
                fill = ui.RED
            elif i == self.selected:
                fill = ui.YELLOW
            elif i == hover_corner:
                fill = ui.ORANGE
            else:
                fill = ui.WHITE
            rad = 4 if i == self.selected else 3
            if i in flagged:  # a diamond: the shape says "problem", not only the colour
                for grow, ink in ((2, ui.NIGHT), (0, fill)):
                    k = rad + grow
                    pts = ((sx, sy - k), (sx + k, sy), (sx, sy + k), (sx - k, sy))
                    pygame.draw.polygon(surface, ui.rgb(ink), pts)
            else:
                pygame.draw.circle(surface, night, (sx, sy), rad + 1)
                pygame.draw.circle(surface, ui.rgb(fill), (sx, sy), rad)
            ui.text(surface, str(i + 1), sx + rad + 2, sy - rad - 9, ui.WHITE, outline=ui.NIGHT)
            if i in (self.selected, hover_corner):
                label = f"RADIO {ui.comma(f'{r:g}')} M"
                box = pygame.Rect(sx + rad + 3, sy + rad + 1, pixelfont.measure(label) + 6, 11)
                if box.right > self.canvas.right - 1:  # no room to the right: it goes to the left
                    box.right = sx - rad - 3
                ui.plate(surface, box, ui.NAVY)
                ui.text(surface, label, box.x + 3, box.y + 2, ui.TEXT)

    def _draw_canvas_hud(self, surface: pygame.Surface) -> None:
        # scale bar
        step = next((s for s in GRID_STEPS if s * self.view.zoom >= SCALE_BAR_PX), 1000.0)
        length = round(step * self.view.zoom)
        x, y = 6, self.canvas.height - 7
        ui.hline(surface, x, y + 1, length + 1, ui.NIGHT)
        ui.hline(surface, x, y, length + 1, ui.WHITE)
        for tick in (x, x + length):
            ui.vline(surface, tick, y - 2, 5, ui.NIGHT)
            ui.vline(surface, tick, y - 1, 3, ui.WHITE)
        ui.text(surface, f"{step:g} M", x, y - 12, ui.WHITE, outline=ui.NIGHT)
        if self.mouse is not None and self.canvas.collidepoint(self.mouse):
            wx, wy = self.view.to_world(*self.mouse)
            coords = f"X {ui.fmt_decimal(wx)}  Y {ui.fmt_decimal(wy)} M"
            ui.text(
                surface,
                coords,
                self.canvas.right - 5,
                self.canvas.bottom - 12,
                ui.TEXT,
                outline=ui.NIGHT,
                align="right",
            )

    # -- drawing: panel --------------------------------------------------------------------------

    def _draw_panel(self, surface: pygame.Surface) -> None:
        rect = self.panel_rect
        ui.plate(surface, rect, ui.SLATE)
        x = rect.x + PAD + 1
        right = rect.right - PAD - 1
        inner = right - x
        y = 4
        ui.text(surface, "EDITOR", x, y, ui.YELLOW)
        y += 10
        name = ui.ellipsize(self.spec.name, inner - (7 if self.model.dirty else 0))
        width = ui.text(surface, name, x, y, ui.TEXT)
        if self.model.dirty:  # unsaved changes: a star after the name
            ui.text(surface, "*", x + width + 3, y, ui.YELLOW)
        y += 9
        where = self._path_text() if self.path else "sin guardar"
        ui.text(surface, ui.ellipsize(where, inner, keep="end"), x, y, ui.STONE_LIGHT)
        y += 10
        # verdict banner: colour, symbol and word
        issues = len(self.analysis.issues)
        label, ink, symbol = {
            "valid": ("VÁLIDO", ui.GREEN, "check"),
            "invalid": (f"{issues} PROBLEMA{'S' if issues != 1 else ''}", ui.RED, "cross"),
            "pending": ("COMPROBANDO…", ui.YELLOW, "ring"),
        }[self.verdict]
        banner = self.banner_rect = pygame.Rect(x - 1, y, inner + 2, 13)
        ui.plate(surface, banner, ink)
        ui.icon(surface, symbol, banner.x + 4, banner.y + 4, ui.NIGHT)
        ui.text(surface, label, banner.x + 13, banner.y + 3, ui.NIGHT)
        y = self.stats_top
        report = self.analysis.report
        spec = self.spec
        radius_bad = bool(report.min_radius and report.min_radius < MIN_RADIUS - 1e-6)
        rows = [
            (
                "LONGITUD",
                f"{ui.fmt_count(round(report.length))} M" if report.length else "-",
                ui.TEXT,
            ),
            (
                "RADIO MÍN",
                ui.comma(f"{report.min_radius:.1f} M") if report.min_radius else "-",
                ui.RED if radius_bad else ui.TEXT,
            ),
            ("ESQUINAS", str(len(spec.corners)), ui.TEXT),
            ("ANCHURA", ui.comma(f"{spec.width:g} M"), ui.TEXT),
            ("DECORADO", str(spec.decoration_seed), ui.TEXT),
            (
                "HIERBA MÍN",
                ui.comma(f"{report.gap:.1f} M") if report.gap not in (None, math.inf) else "-",
                ui.TEXT,
            ),
        ]
        for i, (key, value, vink) in enumerate(rows):
            ry = y + i * ROW
            ui.text(surface, key, x, ry, ui.STONE_LIGHT)
            has_buttons = key in ("ANCHURA", "DECORADO")
            right_edge = self.buttons["width-"].x - 3 if has_buttons else right
            ui.text(surface, value, right_edge, ry, vink, align="right")
        for name_, button in self.buttons.items():
            hot = self.mouse is not None and button.collidepoint(self.mouse)
            ui.plate(surface, button, ui.SLATE_LIGHT if hot else ui.NAVY)
            ui.text(
                surface,
                "+" if name_.endswith("+") else "-",
                button.centerx,
                button.y + 2,
                ui.TEXT,
                align="center",
            )
        y += len(rows) * ROW + 1
        if spec.crossings:
            n_cross = len(spec.crossings)
            note = "1 CRUCE A NIVEL" if n_cross == 1 else f"{n_cross} CRUCES A NIVEL"
            y = ui.paragraph(surface, note, x, y, inner, ui.YELLOW, max_lines=2, pitch=ui.LINE)
        ui.hline(surface, x - 1, y, inner + 2, ui.NIGHT)
        y += 4
        hints_top = rect.bottom - len(HINTS) * ROW - 3
        self._draw_issue_list(surface, x, y, inner, hints_top - 2)
        for k, hint in enumerate(HINTS):
            ui.text(surface, hint, x, hints_top + k * ROW, ui.STONE_LIGHT)

    def _draw_issue_list(
        self, surface: pygame.Surface, x: int, y: int, inner: int, limit: int
    ) -> None:
        self._issue_rows = []
        issues = self.analysis.issues
        if not issues:
            message = "SIN PROBLEMAS" if self.verdict == "valid" else "COMPROBANDO…"
            ui.paragraph(surface, message, x, y, inner, ui.TEXT, pitch=ui.LINE)
            return
        ui.icon(surface, "cross", x, y + 1, ui.RED)
        ui.text(surface, "PROBLEMAS", x + 8, y + 1, ui.RED)
        y += ui.LINE + 1
        for index, issue in enumerate(issues):
            lines = pixelfont.wrap(issue.message, inner - 8)
            if len(lines) > 3:  # a long message is cut: clicking the row shows all of it
                lines = [*lines[:2], pixelfont.ellipsize(lines[2] + " …", inner - 8)]
            height = len(lines) * ui.LINE + 1
            if y + height > limit:
                rest = len(issues) - index
                ui.text(surface, f"… Y {rest} MÁS", x, y + 1, ui.STONE_LIGHT)
                break
            row = pygame.Rect(x - 2, y - 1, inner + 4, height)
            located = bool(issue.corners or issue.points)
            if self.mouse is not None and row.collidepoint(self.mouse) and located:
                surface.fill(ui.rgb(ui.NAVY), row)
            ui.icon(surface, "cross", x, y + 1, ui.RED)
            ly = y
            for line in lines:
                ui.text(surface, line, x + 8, ly + 1, ui.TEXT)
                ly += ui.LINE
            self._issue_rows.append((row, index))
            y += height + 1

    # -- drawing: overlays ---------------------------------------------------------------------

    def _draw_toast(self, surface: pygame.Surface) -> None:
        if not self.toasts:
            return
        toast = self.toasts[0]
        ink = TOAST_INKS.get(toast.kind, ui.TEXT)
        symbol = TOAST_ICONS.get(toast.kind)
        lead = 9 if symbol else 0
        lines = pixelfont.wrap(toast.text, self.canvas.width - 24 - lead)[:5]
        width = max(pixelfont.measure(line) for line in lines) + 10 + lead
        box = pygame.Rect(0, 0, width, len(lines) * ui.LINE + 5)
        box.midtop = (self.canvas.centerx, 4)
        ui.plate(surface, box, ui.NAVY)
        if symbol:
            ui.icon(surface, symbol, box.x + 4, box.y + 4, ink)
        for k, line in enumerate(lines):
            ui.text(surface, line, box.x + 5 + lead, box.y + 3 + k * ui.LINE, ink)

    def _draw_help(self, surface: pygame.Surface) -> None:
        box = pygame.Rect(4, 3, ui.CANVAS_SIZE[0] - 8, ui.CANVAS_SIZE[1] - 6)
        ui.plate(surface, box, ui.NAVY)
        ui.text(surface, "AYUDA", box.x + 6, box.y + 5, ui.YELLOW)
        ui.text(surface, "H O ESC CIERRAN", box.right - 6, box.y + 5, ui.STONE_LIGHT, align="right")
        ui.hline(surface, box.x + 1, box.y + 14, box.w - 2, ui.NIGHT)
        key_w = max(pixelfont.measure(k) for k, _ in HELP_ROWS) + 10
        y = box.y + 19
        for key, desc in HELP_ROWS:
            ui.text(surface, key, box.x + 6, y, ui.TEXT)
            ui.text(surface, desc, box.x + 6 + key_w, y, ui.STONE_LIGHT)
            y += ui.LINE

    def _draw_prompt(self, surface: pygame.Surface) -> None:
        prompt = self.prompt
        box = pygame.Rect(0, 0, min(self.canvas.width - 16, 200), 47)
        box.center = self.canvas.center
        ui.plate(surface, box, ui.NAVY)
        ui.text(surface, prompt.label, box.x + 5, box.y + 5, ui.YELLOW)
        field = pygame.Rect(box.x + 5, box.y + 16, box.w - 10, 13)
        ui.plate(surface, field, ui.NIGHT, ui.STONE)
        shown = pixelfont.ellipsize(prompt.text, field.w - 8, keep="end")
        width = pixelfont.measure(shown)
        if prompt.selected and shown:
            surface.fill(ui.rgb(ui.NAVY_LIGHT), (field.x + 2, field.y + 2, width + 3, 9))
        ui.text(surface, shown, field.x + 4, field.y + 3, ui.TEXT)
        if int(time.perf_counter() * 2) % 2 == 0 and not prompt.selected:
            ui.vline(surface, field.x + 5 + width, field.y + 2, 9, ui.WHITE)
        if prompt.full:
            hint = f"LÍMITE DE {prompt.max_length} CARACTERES"
            ui.text(surface, hint, box.x + 5, box.bottom - 13, ui.YELLOW)
        else:
            ui.text(
                surface,
                "INTRO ACEPTAR, ESC CANCELAR",
                box.x + 5,
                box.bottom - 13,
                ui.STONE_LIGHT,
            )
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
            raise SystemExit(f"No se pudo abrir el circuito:\n{error}") from None
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
