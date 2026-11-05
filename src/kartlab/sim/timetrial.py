"""Time trial: the best lap of a track, saved to disk, and the live gap to it.

While driving, ``TimeTrial`` samples the kart pose (x, y, heading) together with the lap
time and the net distance along the centerline. A lap that is completed in less time than
the stored record replaces it. The record is only poses, so it does not depend on how the
lap was driven (drifting included) and it never records inputs.

Storage is one small JSON file per track in ``records/`` (see ``RecordStore``). Its key:

- the file name holds the track identity: name, seed, width, sample count, length and
  bounding box (to the metre), hashed (all platform independent);
- inside, the record carries ``PHYSICS_VERSION`` and a signature of the centerline (64
  points, compared with a tolerance): the generator may pick a different start sample on
  Linux and Windows (see ``tests/test_track_pins.py``), which moves the finish line, and
  hashing rounded values would flip at the rounding boundaries.

A record whose key does not match, or that is corrupt, is ignored with a warning: the
player simply starts without a record. Nothing here imports pygame.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kartlab.common.io import write_text_atomic
from kartlab.common.paths import RECORDS_DIR
from kartlab.sim.kart import DT
from kartlab.sim.race import Racer
from kartlab.sim.track import Track
from kartlab.sim.version import PHYSICS_VERSION

FORMAT = "kartlab-record"
FORMAT_VERSION = 1
RECORD_EVERY = 2  # one sample every N physics steps (30 Hz); the lap end is always sampled
SIGNATURE_POINTS = 64
SIGNATURE_TOLERANCE = 0.05  # m, between a saved centerline point and the loaded one
LENGTH_TOLERANCE = 0.01  # m
MAX_SAMPLES = 200_000
MAX_FILE_BYTES = 8_000_000
GHOST_HOLD = 1.0  # the ghost stays on the finish line this long after its lap (s)
DELTA_RATE = 6.0  # smoothing of the displayed gap (1/s)
DECIMALS = {"t": 5, "x": 3, "y": 3, "heading": 4, "d": 2}


class RecordWarning(UserWarning):
    """A saved record was ignored (corrupt, other track or other physics)."""


class RecordError(ValueError):
    """The record content is not valid."""


def track_signature(track: Track) -> dict[str, Any]:
    """What identifies the geometry of a track: counts, size and evenly spaced points."""
    index = np.arange(SIGNATURE_POINTS) * len(track) // SIGNATURE_POINTS
    return {
        "name": track.name,
        "seed": track.seed,
        "width": round(track.width, 6),
        "count": len(track),
        "length": round(track.length, 3),
        "points": np.round(track.points[index], 2).tolist(),
    }


def signature_matches(saved: Any, track: Track) -> bool:
    """Same track within tolerance; equal by identity, not by bytes (see the module doc)."""
    if not isinstance(saved, dict):
        return False
    try:
        mine = track_signature(track)
        saved_points = np.asarray(saved["points"], dtype=np.float64)
        return bool(
            saved["name"] == mine["name"]
            and saved["seed"] == mine["seed"]
            and saved["count"] == mine["count"]
            and abs(float(saved["width"]) - mine["width"]) < 1e-6
            and abs(float(saved["length"]) - mine["length"]) <= LENGTH_TOLERANCE
            and saved_points.shape == (SIGNATURE_POINTS, 2)
            and np.abs(saved_points - np.asarray(mine["points"])).max() <= SIGNATURE_TOLERANCE
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def record_filename(track: Track) -> str:
    """``<name>-<hash>.json``; the hash only uses values that are stable across platforms."""
    ascii_name = unicodedata.normalize("NFKD", track.name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")[:40] or "circuito"
    # Length and bounding box to the nearest metre: they do not change when the generator
    # picks another start sample, and platform noise (1e-12 m) will not cross a rounding
    # boundary in practice. Tracks that only differ in shape get their own file.
    low, high = np.round(track.points.min(axis=0)), np.round(track.points.max(axis=0))
    shape = f"{round(track.length)}|{low[0]:.0f},{low[1]:.0f}|{high[0]:.0f},{high[1]:.0f}"
    identity = f"{track.name}|{track.seed}|{track.width:.3f}|{len(track)}|{shape}"
    return f"{slug}-{hashlib.sha1(identity.encode('utf-8')).hexdigest()[:8]}.json"


@dataclass
class Record:
    """A lap: poses sampled over time. ``d`` is the net distance along the centerline."""

    lap_time: float
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray  # unwrapped, so it interpolates without jumps
    d: np.ndarray
    signature: dict[str, Any]
    physics_version: int = PHYSICS_VERSION

    def __post_init__(self) -> None:
        # First sample at each new maximum distance: a strictly increasing curve, so the
        # time at a distance is well defined even after standing still or reversing.
        peak = np.maximum.accumulate(self.d)
        keep = np.concatenate([[True], self.d[1:] > peak[:-1]])
        self._dist = self.d[keep]
        self._dist_time = self.t[keep]

    @classmethod
    def from_samples(
        cls, lap_time: float, samples: list[tuple[float, ...]], signature: dict[str, Any]
    ) -> Record:
        """``samples`` are ``(t, x, y, heading, d)``; values are rounded as they are saved."""
        cols = np.asarray(samples, dtype=np.float64).T
        t, x, y, heading, d = (
            np.round(c, DECIMALS[k]) for c, k in zip(cols, DECIMALS, strict=True)
        )
        heading = np.round(np.unwrap(heading), DECIMALS["heading"])
        return cls(round(float(lap_time), 6), t, x, y, heading, d, signature)

    def __len__(self) -> int:
        return len(self.t)

    def pose_at(self, lap_time: float) -> tuple[float, float, float]:
        """Interpolated ``(x, y, heading)`` at a lap time (clamped to the lap)."""
        return (
            float(np.interp(lap_time, self.t, self.x)),
            float(np.interp(lap_time, self.t, self.y)),
            float(np.interp(lap_time, self.t, self.heading)),
        )

    def time_at(self, distance: float) -> float:
        """Lap time at which the record had driven ``distance`` metres (clamped)."""
        return float(np.interp(distance, self._dist, self._dist_time))

    def to_json(self) -> str:
        def column(values: np.ndarray, decimals: int) -> str:
            return (
                "["
                + ",".join(f"{v:.{decimals}f}".rstrip("0").rstrip(".") or "0" for v in values)
                + "]"
            )

        parts = [
            f'  "format": "{FORMAT}"',
            f'  "version": {FORMAT_VERSION}',
            f'  "physics_version": {int(self.physics_version)}',
            f'  "lap_time": {self.lap_time!r}',
            f'  "track": {json.dumps(self.signature, ensure_ascii=False, separators=(",", ":"))}',
        ]
        for name, values in (("t", self.t), ("x", self.x), ("y", self.y),
                             ("heading", self.heading), ("d", self.d)):  # fmt: skip
            parts.append(f'  "{name}": {column(values, DECIMALS[name])}')
        return "{\n" + ",\n".join(parts) + "\n}\n"

    @classmethod
    def from_json(cls, text: str) -> Record:
        """Parse and validate; raises ``RecordError`` for anything that is not a record."""

        def reject(name: str) -> None:
            raise RecordError(f"constante no válida {name}")

        try:
            data = json.loads(text, parse_constant=reject)
        except (ValueError, RecursionError) as error:
            raise RecordError(f"JSON no válido: {error}") from None
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise RecordError("no es un récord de KART LAB")
        if data.get("version") != FORMAT_VERSION:
            raise RecordError(f"versión de formato {data.get('version')!r} no soportada")
        physics = data.get("physics_version")
        if isinstance(physics, bool) or not isinstance(physics, int):
            raise RecordError("falta 'physics_version'")
        lap_time = data.get("lap_time")
        if isinstance(lap_time, bool) or not isinstance(lap_time, (int, float)):
            raise RecordError("'lap_time' debe ser un número")
        try:
            lap_time = float(lap_time)  # a huge JSON integer overflows here
        except OverflowError:
            raise RecordError("'lap_time' es demasiado grande") from None
        if not math.isfinite(lap_time) or lap_time <= 0:
            raise RecordError("'lap_time' debe ser positivo")
        columns = []
        for name in DECIMALS:
            values = data.get(name)
            if not isinstance(values, list) or not 2 <= len(values) <= MAX_SAMPLES:
                raise RecordError(f"'{name}' debe ser una lista de 2 a {MAX_SAMPLES} números")
            try:
                array = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                raise RecordError(f"'{name}' contiene valores que no son números") from None
            if array.ndim != 1 or not np.isfinite(array).all():
                raise RecordError(f"'{name}' contiene valores no finitos")
            columns.append(array)
        if len({len(c) for c in columns}) != 1:
            raise RecordError("las columnas tienen longitudes distintas")
        t, x, y, heading, d = columns
        if np.any(np.diff(t) < 0) or t[0] != 0 or abs(t[-1] - lap_time) > 1e-3:
            raise RecordError("los tiempos no son crecientes o no coinciden con 'lap_time'")
        if not isinstance(data.get("track"), dict):
            raise RecordError("falta la firma del circuito")
        return cls(lap_time, t, x, y, heading, d, data["track"], physics)


class RecordStore:
    """Reads and writes the record of each track in a folder."""

    def __init__(self, directory: Path | str | None = None) -> None:
        self.directory = Path(directory) if directory is not None else RECORDS_DIR
        self.problem: str | None = None  # why the last ``load`` ignored a file

    def path_for(self, track: Track) -> Path:
        return self.directory / record_filename(track)

    def load(self, track: Track) -> Record | None:
        """The saved record of ``track``, or ``None`` (with a warning) if there is none usable."""
        self.problem = None
        path = self.path_for(track)
        if not path.is_file():
            return None
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                raise RecordError("fichero demasiado grande")
            record = Record.from_json(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, RecordError, OverflowError) as error:
            return self._ignore(path, f"fichero dañado ({error})")
        if record.physics_version != PHYSICS_VERSION:
            return self._ignore(
                path, f"física {record.physics_version}, la actual es la {PHYSICS_VERSION}"
            )
        if not signature_matches(record.signature, track):
            return self._ignore(path, "el circuito ya no es el mismo")
        return record

    def _ignore(self, path: Path, reason: str) -> None:
        self.problem = f"Récord ignorado, {reason}"
        warnings.warn(f"{self.problem}: {path}", RecordWarning, stacklevel=3)
        return None

    def save(self, track: Track, record: Record) -> Path:
        path = self.path_for(track)
        write_text_atomic(path, record.to_json())
        return path


@dataclass(frozen=True)
class LapEvent:
    """A lap was completed. ``record`` tells whether it is the new best; ``valid`` whether
    the lap could count at all (driven by hand from the start line, a full lap of progress)."""

    lap_time: float
    record: bool
    previous: float | None  # the record it had to beat
    saved: bool = True
    valid: bool = True


class TimeTrial:
    """Records the current lap, keeps the best and measures the gap to it.

    Call ``begin`` after every ``Racer.reset`` and ``update`` after every ``Racer.step``.
    """

    def __init__(self, track: Track, store: RecordStore | None = None) -> None:
        self.track = track
        self.store = store or RecordStore()
        self.record: Record | None = self.store.load(track)
        self.problem = self.store.problem
        self.last_lap: float | None = None
        self.delta: float | None = None  # lap time minus the record's time at this point (s)
        self.raw_delta: float | None = None
        self._signature = track_signature(track)
        self._samples: list[tuple[float, ...]] = []
        self._valid = False
        self._base = 0.0
        self._laps = 0
        self._tick = 0

    # -- lap life cycle -------------------------------------------------------------------

    def begin(self, racer: Racer, manual: bool = True) -> None:
        """Start a fresh lap (after a reset): whatever was being recorded is dropped."""
        self._laps = racer.laps
        self._base = racer.laps * self.track.length
        self._samples = []
        self._tick = 0
        self._valid = manual
        self.delta = self.raw_delta = None
        self._sample(racer)

    def update(self, racer: Racer, manual: bool = True) -> LapEvent | None:
        """Account for one physics step. Returns an event when a lap is completed."""
        if not manual:
            self._valid = False  # the autopilot drives no records
        event = None
        if racer.laps > self._laps:
            event = self._finish(racer)
            self.begin(racer, manual)
            return event
        self._tick += 1
        if self._tick % RECORD_EVERY == 0:
            self._sample(racer)
        if self.record is not None:
            self.raw_delta = racer.lap_time - self.record.time_at(self._distance(racer))
            if self.delta is None:
                self.delta = self.raw_delta
            else:
                self.delta += (self.raw_delta - self.delta) * (1.0 - math.exp(-DELTA_RATE * DT))
        return None

    def _distance(self, racer: Racer) -> float:
        return racer.progress.distance - self._base

    def _sample(self, racer: Racer) -> None:
        kart = racer.kart
        self._samples.append((racer.lap_time, kart.x, kart.y, kart.heading, self._distance(racer)))

    def _finish(self, racer: Racer) -> LapEvent:
        lap_time = racer.lap_times[-1]
        self.last_lap = lap_time
        previous = self.record.lap_time if self.record else None
        # Only a clean lap counts: driven by hand from the start and, in net terms, a full
        # lap along the centerline (going back and forth over the line never adds up).
        complete = self._valid and self._distance(racer) >= self.track.length - 1e-6
        if not complete:
            return LapEvent(lap_time, False, previous, valid=False)
        if previous is not None and lap_time >= previous:
            return LapEvent(lap_time, False, previous)
        self._sample(racer)
        self._samples[-1] = (lap_time, *self._samples[-1][1:])
        record = Record.from_samples(lap_time, self._samples, self._signature)
        self.record = record
        try:
            self.store.save(self.track, record)
        except OSError as error:
            warnings.warn(f"No se pudo guardar el récord: {error}", RecordWarning, stacklevel=2)
            return LapEvent(lap_time, True, previous, saved=False)
        return LapEvent(lap_time, True, previous)

    # -- ghost ----------------------------------------------------------------------------

    def ghost_pose(self, lap_time: float) -> tuple[float, float, float] | None:
        """Where the record's kart is ``lap_time`` seconds into its lap (``None``: not shown).

        The ghost follows the clock of the current lap, so each new lap makes it start
        again from the line; it waits there briefly if it finishes before you do.
        """
        if self.record is None or lap_time > self.record.lap_time + GHOST_HOLD:
            return None
        return self.record.pose_at(lap_time)
