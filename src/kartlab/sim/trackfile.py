"""Hand-made tracks as JSON files: format, validator and loader.

A file describes a closed polygon of corners with a turning radius at each one, like
the showcase designs. The track is built through exactly the same path as the generator
(``filleted_polygon`` then ``finish_centerline``), so a valid file gives the same kind of
centerline the AI was trained on. Nothing reaches the environment without passing
``validate_spec``: ``load_track_file`` always validates.

Format (version 1)::

    {
      "format": 1,
      "name": "Mi circuito",
      "width": 12.0,
      "decoration_seed": 0,
      "corners": [[x, y, radius], ...]
    }

``corners`` are in metres, in driving order (the direction follows from the polygon:
counterclockwise or clockwise are both fine). The start line is placed automatically on
the straightest stretch, as the generator does. ``decoration_seed`` only affects the
scenery of the pseudo-3D view.

Format 2 is a version 1 file plus ``"crossings": [[x, y], ...]``: the approximate points
(metres) where the centerline crosses itself on one level, like a figure eight with a
level crossing (no bridge or tunnel). Format 2 is *for* tracks with crossings, so it
requires a non-empty list; a track without crossings stays a format 1 file, and every
format 1 file keeps loading and behaving exactly as before. The validator accepts a
self-intersection only where one is declared (within ``CROSSING_MATCH`` metres), and only
if it is a clean X: the branches meet at 60 degrees or more, both stay straight inside
the crossing zone, the zone is clear of the start line, and no other parts of the track
touch (``has_clearance_with_crossings``).

A format 2 file may also say where the start line goes: ``"start": [x, y]``, a point within
3 m of the centerline on a straight stretch. Sample 0 is then the centerline sample nearest
to it. Without ``start`` the line goes on the straightest stretch away from the crossing,
but near-ties between equally straight stretches can resolve differently on another
platform, so a file that is meant to be shared should declare it. ``start`` is only
valid in format 2.

Extending the format: every file declares ``format``. Unknown fields are an error, so a
file written for a newer version is never silently loaded half-understood. Format 1
rejects ``crossings`` with its own message.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kartlab.common.io import write_text_atomic
from kartlab.sim.kart import KartParams
from kartlab.sim.showcase import SHOWCASE
from kartlab.sim.track import (
    CROSSING_MAX_CURVATURE,
    CROSSING_MIN_ANGLE,
    CROSSING_START_MARGIN,
    MAX_ATTEMPTS,
    Crossing,
    Track,
    TrackParams,
    closed_length,
    crossing_zone,
    curvature,
    filleted_polygon,
    find_crossings,
    finish_centerline,
    finish_crossing_centerline,
    has_clearance,
    has_clearance_with_crossings,
    random_layout,
)

FORMAT_VERSION = 1  # tracks without crossings are written in this format
CROSSINGS_FORMAT_VERSION = 2  # tracks with level crossings
KNOWN_FIELDS = frozenset(
    {"format", "name", "width", "decoration_seed", "corners", "crossings", "start"}
)
START_SNAP = 3.0  # a declared start must be this close to the centerline (m)
START_HALF_WINDOW = 30.0  # the start line sits on a straight stretch this long around it (m)
START_MAX_CURVATURE = 1 / 100  # (1/m)
CROSSING_MATCH = 12.0  # a declared crossing must be this close to a real one (m)
MAX_CROSSINGS = 4

_GENERATOR = TrackParams()
# The tightest path curvature the generator allows, never above what the kart can steer.
MIN_RADIUS = max(_GENERATOR.min_radius, 1.0 / KartParams().max_curvature)  # centerline (m)
CLEARANCE = _GENERATOR.clearance  # grass gap between non-adjacent sections (m)
MIN_WIDTH = 8.0  # asphalt (m); the generator's training range starts at 10
MAX_WIDTH = 20.0
MIN_LENGTH = 400.0  # centerline (m); generated tracks are 700-1200
MAX_LENGTH = 2500.0
MIN_CORNERS = 3
MAX_CORNERS = 40
MAX_COORDINATE = 3000.0  # m
MAX_RADIUS = 500.0  # m
MIN_EDGE = 5.0  # shortest straight edge of the polygon, before filleting (m)
MAX_NAME = 40
MAX_FILE_BYTES = 1_000_000
MAX_DECORATION_SEED = 2**31 - 1

_EPS = 1e-9


class TrackFileError(ValueError):
    """A track file that cannot be read or fails validation; ``problems`` lists why."""

    def __init__(self, problems: list[str] | str) -> None:
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("\n".join(self.problems))


@dataclass(frozen=True)
class TrackSpec:
    """The contents of a track file: what the author draws."""

    name: str
    width: float
    corners: tuple[tuple[float, float, float], ...]  # (x, y, radius) in metres
    decoration_seed: int = 0
    crossings: tuple[tuple[float, float], ...] = ()  # approximate (x, y) of level crossings
    start: tuple[float, float] | None = None  # (x, y) of the start line (format 2 only)


@dataclass(frozen=True)
class TrackReport:
    """Result of validating a spec: ``problems`` is empty when the track is valid."""

    problems: tuple[str, ...]
    length: float | None = None  # centerline length (m)
    min_radius: float | None = None  # tightest radius of the centerline (m)
    gap: float | None = None  # narrowest grass gap between non-adjacent sections (m)
    crossings: int = 0  # level crossings found on the centerline

    @property
    def ok(self) -> bool:
        return not self.problems


# -- reading and writing ---------------------------------------------------------------


def _number(value: Any) -> bool:
    """A real JSON number that fits a float (a 400-digit integer does not)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def spec_from_dict(data: Any) -> TrackSpec:
    """Parse the JSON object of a file. Raises ``TrackFileError`` on any structural problem
    (this checks types and fields only; ``validate_spec`` checks the geometry)."""
    if not isinstance(data, dict):
        raise TrackFileError("el fichero debe contener un objeto JSON { ... }")
    problems: list[str] = []
    unknown = sorted(data.keys() - KNOWN_FIELDS)
    if unknown:
        problems.append(f"campos desconocidos: {', '.join(map(repr, unknown))}")

    version = data.get("format")
    if "format" not in data:
        problems.append("falta el campo 'format'")
    elif not isinstance(version, int) or isinstance(version, bool):
        problems.append("'format' debe ser un número entero")
    elif version not in (FORMAT_VERSION, CROSSINGS_FORMAT_VERSION):
        problems.append(
            f"formato {version} no soportado (esta versión lee el {FORMAT_VERSION} y "
            f"el {CROSSINGS_FORMAT_VERSION})"
        )
    elif version == FORMAT_VERSION and "crossings" in data:
        problems.append(
            "el campo 'crossings' está reservado para una versión futura del formato y el "
            f"formato {FORMAT_VERSION} no lo admite; los circuitos con cruces a nivel usan "
            f"'format': {CROSSINGS_FORMAT_VERSION}"
        )
    elif version == CROSSINGS_FORMAT_VERSION and not data.get("crossings"):
        problems.append(
            f"formato {CROSSINGS_FORMAT_VERSION} no soportado sin 'crossings': es el formato "
            f"de los circuitos con cruces a nivel; si no tiene cruces usa 'format': "
            f"{FORMAT_VERSION}"
        )

    if data.get("format") == FORMAT_VERSION and "start" in data:
        problems.append(
            f"el campo 'start' solo existe en el formato {CROSSINGS_FORMAT_VERSION}, el de los "
            "circuitos con cruces"
        )

    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        problems.append("'name' debe ser un texto no vacío")
    elif len(name) > MAX_NAME or not name.isprintable():
        problems.append(f"'name' debe tener como máximo {MAX_NAME} caracteres imprimibles")

    width = data.get("width")
    if not _number(width):
        problems.append("'width' debe ser un número (metros)")

    seed = data.get("decoration_seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= MAX_DECORATION_SEED:
        problems.append(f"'decoration_seed' debe ser un entero entre 0 y {MAX_DECORATION_SEED}")

    corners = data.get("corners")
    parsed: list[tuple[float, float, float]] = []
    if not isinstance(corners, list):
        problems.append("'corners' debe ser una lista de esquinas [x, y, radio]")
    else:
        for i, corner in enumerate(corners, start=1):
            if not (isinstance(corner, list) and len(corner) == 3 and all(map(_number, corner))):
                problems.append(f"esquina {i}: debe ser [x, y, radio] con tres números")
            else:
                parsed.append((float(corner[0]), float(corner[1]), float(corner[2])))
    crossings = _parse_crossings(data.get("crossings", []), problems)
    start = None
    if "start" in data:
        raw = data["start"]
        if isinstance(raw, list) and len(raw) == 2 and all(map(_number, raw)):
            start = (float(raw[0]), float(raw[1]))
        else:
            problems.append("'start' debe ser [x, y] con dos números (metros)")
    if problems:
        raise TrackFileError(problems)
    return TrackSpec(str(name).strip(), float(width), tuple(parsed), int(seed), crossings, start)


def _parse_crossings(raw: Any, problems: list[str]) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw, list):
        problems.append("'crossings' debe ser una lista de puntos [x, y]")
        return ()
    if len(raw) > MAX_CROSSINGS:
        problems.append(f"'crossings' admite como máximo {MAX_CROSSINGS} cruces")
    parsed = []
    for i, point in enumerate(raw[:MAX_CROSSINGS], start=1):
        if not (isinstance(point, list) and len(point) == 2 and all(map(_number, point))):
            problems.append(f"cruce {i}: debe ser [x, y] con dos números")
        else:
            parsed.append((float(point[0]), float(point[1])))
    return tuple(parsed)


def spec_to_dict(spec: TrackSpec) -> dict[str, Any]:
    data: dict[str, Any] = {
        "format": CROSSINGS_FORMAT_VERSION if spec.crossings else FORMAT_VERSION,
        "name": spec.name,
        "width": spec.width,
        "decoration_seed": spec.decoration_seed,
        "corners": [list(c) for c in spec.corners],
    }
    if spec.crossings:
        data["crossings"] = [list(c) for c in spec.crossings]
    if spec.start is not None:
        data["start"] = list(spec.start)
    return data


def _plain(value: float) -> float | int:
    return int(value) if float(value).is_integer() and abs(value) < 1e15 else float(value)


def spec_to_json(spec: TrackSpec) -> str:
    """Readable JSON text: one corner per line, ending in a newline."""
    corners = ",\n".join(
        "    [" + ", ".join(json.dumps(_plain(v)) for v in corner) + "]" for corner in spec.corners
    )
    crossings = ""
    if spec.crossings:
        lines = ",\n".join(
            "    [" + ", ".join(json.dumps(_plain(v)) for v in point) + "]"
            for point in spec.crossings
        )
        crossings = f',\n  "crossings": [\n{lines}\n  ]'
        if spec.start is not None:
            start = ", ".join(json.dumps(_plain(v)) for v in spec.start)
            crossings += f',\n  "start": [{start}]'
    version = CROSSINGS_FORMAT_VERSION if spec.crossings else FORMAT_VERSION
    return (
        "{\n"
        f'  "format": {version},\n'
        f'  "name": {json.dumps(spec.name, ensure_ascii=False)},\n'
        f'  "width": {json.dumps(_plain(spec.width))},\n'
        f'  "decoration_seed": {spec.decoration_seed},\n'
        '  "corners": [\n'
        f"{corners}\n"
        f"  ]{crossings}\n"
        "}\n"
    )


def save_spec(path: Path | str, spec: TrackSpec) -> None:
    """Atomically write a spec as UTF-8 with LF line endings. Does not validate."""
    write_text_atomic(Path(path), spec_to_json(spec))


def _reject_constant(name: str) -> Any:
    raise ValueError(f"valor no válido {name}")


def load_spec(path: Path | str) -> TrackSpec:
    """Read a track file. Raises ``TrackFileError`` with a clear message if it is missing,
    too large, not UTF-8, not JSON (corrupt or truncated) or structurally wrong."""
    path = Path(path)
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            raise TrackFileError(f"{path}: el fichero es demasiado grande para ser un circuito")
        # utf-8-sig: PowerShell 5.1 writes a BOM; files are always written without one
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise TrackFileError(f"{path}: no existe el fichero") from None
    except UnicodeDecodeError:
        raise TrackFileError(f"{path}: no está en UTF-8") from None
    except OSError as error:
        raise TrackFileError(f"{path}: no se puede leer ({error.strerror or error})") from None
    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except RecursionError:
        raise TrackFileError(f"{path}: JSON anidado demasiado hondo para ser un circuito") from None
    except ValueError as error:
        raise TrackFileError(
            f"{path}: JSON no válido o incompleto ({error}); ¿está truncado o corrupto?"
        ) from None
    try:
        return spec_from_dict(data)
    except TrackFileError as error:
        raise TrackFileError([f"{path}: {p}" for p in error.problems]) from None


# -- export ----------------------------------------------------------------------------


def spec_from_seed(seed: int, params: TrackParams | None = None) -> TrackSpec:
    """The corner design behind ``generate_track(seed, params)``, as an editable spec.

    It replays the generator's random draws up to the accepted attempt and applies the
    same scaling, rotation and driving direction to the corners. Rebuilding the spec
    gives a centerline that follows the generated one (corners are rounded to 1 mm and
    the start line may sit elsewhere), but it is not bit-identical: the reserved tracks
    themselves never go through this path.
    """
    if seed < 0:
        raise ValueError("la semilla no puede ser negativa")
    params = params or TrackParams()
    rng = np.random.default_rng(seed)
    for _ in range(MAX_ATTEMPTS):
        verts, radii = random_layout(rng, params)
        rotation = rng.uniform(0.0, 2 * np.pi)
        clockwise = bool(rng.integers(2))
        path, used = filleted_polygon(verts, radii)
        if np.min(used) < params.min_radius:
            continue
        if (
            finish_centerline(
                path, params.width, params.min_radius, params.clearance, params.length
            )
            is None
        ):
            continue
        scale = params.length / closed_length(path)
        c, s = math.cos(rotation), math.sin(rotation)
        corners = np.column_stack([verts * scale @ np.array([[c, s], [-s, c]]), radii * scale])
        if clockwise:
            corners = corners[::-1]
        rounded = tuple((float(x), float(y), float(r)) for x, y, r in np.round(corners, 3))
        return TrackSpec(f"Semilla {seed}", params.width, rounded, seed % (MAX_DECORATION_SEED + 1))
    raise RuntimeError(f"could not generate a valid track for seed {seed}")


def spec_from_showcase(key: str) -> TrackSpec:
    """The hand-made showcase design ``key`` (``harbor``, ``canyon``, ``plateau``)."""
    design = SHOWCASE[key]
    return TrackSpec(design.name, design.width, design.corners)


# -- validation ------------------------------------------------------------------------


def _crossing(points: np.ndarray, stride: int = 1) -> tuple[int, int] | None:
    """First pair of non-adjacent segments of a closed polyline that properly cross."""
    pts = points[::stride]
    a, b = pts, np.roll(pts, -1, axis=0)

    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (
            r[..., 0] - p[..., 0]
        )

    a0, a1 = a[:, None, :], b[:, None, :]
    b0, b1 = a[None, :, :], b[None, :, :]
    cross = (orient(a0, a1, b0) * orient(a0, a1, b1) < -_EPS) & (
        orient(b0, b1, a0) * orient(b0, b1, a1) < -_EPS
    )
    n = len(pts)
    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    cross &= np.minimum(sep, n - sep) > 1
    found = np.argwhere(np.triu(cross))
    return (int(found[0, 0]), int(found[0, 1])) if len(found) else None


def narrowest_gap(points: np.ndarray, width: float, crossings: Sequence[Crossing] = ()) -> float:
    """Smallest grass gap (m) between parts of the track that ``has_clearance`` compares
    (or, with ``crossings``, ``has_clearance_with_crossings``).

    Informative only: whether a track is valid is decided by the clearance check itself.
    """
    sub = points[::2]
    spacing = closed_length(points) / len(sub)
    idx = np.arange(len(sub))
    sep = np.abs(idx[:, None] - idx[None, :])
    arc = np.minimum(sep, len(sub) - sep) * spacing
    required = width + CLEARANCE
    far = arc > (math.pi / 2) * required
    for c in crossings:
        inside = np.hypot(sub[:, 0] - c.x, sub[:, 1] - c.y) <= crossing_zone(width, CLEARANCE)
        far &= ~(inside[:, None] & inside[None, :])
    if not far.any():
        return math.inf
    d2 = np.sum(sub**2, axis=1)[:, None] + np.sum(sub**2, axis=1)[None, :] - 2.0 * sub @ sub.T
    return float(math.sqrt(max(float(d2[far].min()), 0.0)) - width)


def _check_parameters(spec: TrackSpec) -> list[str]:
    problems = []
    if not MIN_WIDTH <= spec.width <= MAX_WIDTH:
        problems.append(
            f"anchura {spec.width:g} m fuera de rango: debe estar entre {MIN_WIDTH:g} y "
            f"{MAX_WIDTH:g} m"
        )
    count = len(spec.corners)
    if not MIN_CORNERS <= count <= MAX_CORNERS:
        problems.append(f"{count} esquinas: debe haber entre {MIN_CORNERS} y {MAX_CORNERS}")
    for i, (x, y, radius) in enumerate(spec.corners, start=1):
        if abs(x) > MAX_COORDINATE or abs(y) > MAX_COORDINATE:
            problems.append(
                f"esquina {i}: la posición ({x:g}, {y:g}) se sale de ±{MAX_COORDINATE:g} m"
            )
        if radius < MIN_RADIUS:
            problems.append(
                f"esquina {i}: radio {radius:g} m demasiado cerrado; el mínimo es {MIN_RADIUS:g} m"
            )
        elif radius > MAX_RADIUS:
            problems.append(f"esquina {i}: radio {radius:g} m excesivo (máximo {MAX_RADIUS:g} m)")
    return problems


def _check_polygon(corners: np.ndarray, allow_crossings: bool = False) -> list[str]:
    """Edge lengths and crossings of the corner polygon, before any rounding.

    With ``allow_crossings`` (a format 2 file) the polygon may cross itself: whether each
    crossing is declared and clean is decided later, on the finished centerline."""
    problems = []
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    n = len(corners)
    for i in np.flatnonzero(lengths < MIN_EDGE):
        problems.append(
            f"las esquinas {i + 1} y {(i + 1) % n + 1} están a {lengths[i]:.1f} m: "
            f"separa más las esquinas (mínimo {MIN_EDGE:g} m entre dos consecutivas)"
        )
    if problems or allow_crossings:
        return problems
    pair = _crossing(corners)
    if pair is not None:
        (a, b), (c, d) = (pair[0] + 1, (pair[0] + 1) % n + 1), (pair[1] + 1, (pair[1] + 1) % n + 1)
        problems.append(
            f"el circuito se cruza a sí mismo: el tramo de la esquina {a} a la {b} "
            f"corta el de la {c} a la {d}"
        )
    return problems


def _build_points(spec: TrackSpec) -> tuple[np.ndarray | None, list[str]]:
    """Centerline of a spec through the generator's own functions, plus the problems found
    while rounding the corners.

    The final curvature and clearance checks are left out here (``finish_centerline`` is
    called with neutral limits) so each can be reported on its own afterwards. A spec with
    crossings goes through ``finish_crossing_centerline`` instead, which keeps the start
    line away from them.
    """
    corners = np.array(spec.corners, dtype=np.float64)
    path, used = filleted_polygon(corners[:, :2], corners[:, 2])
    # As in the generator, a radius is an upper bound: when the neighbouring arcs would
    # use more than 90 % of an edge it shrinks, and only a result below the minimum
    # radius is rejected.
    problems = []
    for i, (want, got) in enumerate(zip(corners[:, 2], used, strict=True), start=1):
        if got < MIN_RADIUS - 1e-6:
            problems.append(
                f"esquina {i}: el radio {want:g} m no cabe entre sus vecinas y se queda en "
                f"{got:.1f} m (mínimo {MIN_RADIUS:g} m); separa las esquinas o acorta el radio"
            )
    if problems:
        return None, problems
    # finish_centerline keeps the length of the path, so a wrong length is known here;
    # the checks that follow cost O(samples^2) and must never see an enormous track.
    length = closed_length(path)
    if not MIN_LENGTH <= length <= MAX_LENGTH:
        return None, [_length_problem(length)]
    if spec.crossings:
        points = finish_crossing_centerline(path, spec.width, CLEARANCE, start=spec.start)
        if spec.start is not None:
            problems = _check_start(points, spec.start)
            if problems:
                return None, problems
        return points, []
    return finish_centerline(path, spec.width, 0.0, -spec.width), []


def _check_start(points: np.ndarray, start: tuple[float, float]) -> list[str]:
    """A declared start must be on the centerline, on a straight stretch (sample 0 is it)."""
    offset = math.hypot(points[0, 0] - start[0], points[0, 1] - start[1])
    if offset > START_SNAP:
        return [
            f"la salida ({start[0]:g}, {start[1]:g}) no está sobre el circuito: la línea central "
            f"más cercana queda a {offset:.1f} m (máximo {START_SNAP:g} m)"
        ]
    reach = round(START_HALF_WINDOW * len(points) / closed_length(points))
    window = np.arange(-reach, reach + 1) % len(points)
    if float(np.max(np.abs(curvature(points))[window])) > START_MAX_CURVATURE:
        return [
            f"la salida ({start[0]:g}, {start[1]:g}) está en una curva: la línea de meta necesita "
            f"{2 * START_HALF_WINDOW:g} m rectos a su alrededor"
        ]
    return []


def _length_problem(length: float) -> str:
    return (
        f"longitud {length:.0f} m fuera de rango: debe estar entre {MIN_LENGTH:g} y "
        f"{MAX_LENGTH:g} m"
    )


def _check_crossings(
    points: np.ndarray, width: float, declared: Sequence[tuple[float, float]]
) -> tuple[list[str], list[Crossing]]:
    """The centerline may cross itself only where declared, and only as a clean X."""
    found = find_crossings(points)
    problems = []
    zone = crossing_zone(width, CLEARANCE)
    claimed: dict[int, int] = {}
    for k, (dx, dy) in enumerate(declared, start=1):
        dist = [math.hypot(c.x - dx, c.y - dy) for c in found]
        best = int(np.argmin(dist)) if dist else -1
        if best < 0 or dist[best] > CROSSING_MATCH:
            problems.append(
                f"el cruce {k} declarado en ({dx:g}, {dy:g}) no existe: la línea central no se "
                f"cruza a menos de {CROSSING_MATCH:g} m de ahí"
            )
        elif best in claimed:
            problems.append(f"los cruces {claimed[best]} y {k} declaran el mismo cruce")
        else:
            claimed[best] = k
    bend = np.abs(curvature(points))
    for n, c in enumerate(found):
        where = f"({c.x:.0f}, {c.y:.0f})"
        if n not in claimed:
            problems.append(
                f"la línea central se cruza en {where} y ese cruce no está declarado en 'crossings'"
            )
            continue
        if c.angle < CROSSING_MIN_ANGLE:
            problems.append(
                f"el cruce en {where} es demasiado oblicuo: {math.degrees(c.angle):.0f}° "
                f"(mínimo {math.degrees(CROSSING_MIN_ANGLE):.0f}°)"
            )
        near = np.hypot(points[:, 0] - c.x, points[:, 1] - c.y) <= zone
        if float(np.max(bend[near])) > CROSSING_MAX_CURVATURE:
            problems.append(
                f"el cruce en {where} está demasiado cerca de una curva: los dos tramos deben "
                f"ser rectos en {zone:g} m a su alrededor"
            )
        if math.hypot(c.x - points[0, 0], c.y - points[0, 1]) < zone + CROSSING_START_MARGIN:
            problems.append(f"el cruce en {where} está demasiado cerca de la línea de meta")
    return problems, found


def check_centerline(
    points: np.ndarray, width: float, declared: Sequence[tuple[float, float]] = ()
) -> TrackReport:
    """The checks on a finished centerline: length, tightest curve, self-crossing and
    clearance between non-adjacent sections. The same limits as the generator's, so it
    also accepts every generated and showcase track.

    With ``declared`` crossings (a format 2 file) the centerline may cross itself exactly
    there, see ``_check_crossings``; without them any self-crossing is an error."""
    problems = []
    length = closed_length(points)
    kappa = float(np.max(np.abs(curvature(points))))
    min_radius = 1.0 / kappa if kappa > 0 else math.inf
    if not MIN_LENGTH <= length <= MAX_LENGTH:
        problems.append(_length_problem(length))
    if kappa * MIN_RADIUS > 1.0:
        problems.append(
            f"curva demasiado cerrada: radio mínimo {min_radius:.1f} m en el trazado "
            f"(mínimo {MIN_RADIUS:g} m); ensancha alguna esquina o aléjala de las vecinas"
        )
    if not MIN_LENGTH <= length <= MAX_LENGTH:
        # The O(samples^2) checks below are skipped: the track may be enormous.
        return TrackReport(tuple(problems), length, min_radius)
    gap = None
    found: list[Crossing] = []
    if declared:
        crossing_problems, found = _check_crossings(points, width, declared)
        problems += crossing_problems
        gap = narrowest_gap(points, width, found)
        if not has_clearance_with_crossings(points, width, CLEARANCE, found):
            problems.append(
                f"dos partes del circuito están demasiado cerca fuera de los cruces: quedan "
                f"{gap:.1f} m de hierba entre ellas y se necesitan al menos {CLEARANCE:g} m"
            )
    elif _crossing(points, stride=2) is not None:
        problems.append("la línea central se cruza a sí misma")
    else:
        gap = narrowest_gap(points, width)
        if not has_clearance(points, width, CLEARANCE):
            problems.append(
                f"dos partes del circuito están demasiado cerca: quedan {gap:.1f} m de hierba "
                f"entre ellas y se necesitan al menos {CLEARANCE:g} m"
            )
    return TrackReport(tuple(problems), length, min_radius, gap, len(found))


def _analyze(spec: TrackSpec) -> tuple[TrackReport, np.ndarray | None]:
    problems = _check_parameters(spec)
    if not MIN_CORNERS <= len(spec.corners) <= MAX_CORNERS or any(
        abs(v) > MAX_COORDINATE for c in spec.corners for v in c[:2]
    ):
        return TrackReport(tuple(problems)), None  # nothing sensible to draw
    problems += _check_polygon(
        np.array(spec.corners, dtype=np.float64)[:, :2], allow_crossings=bool(spec.crossings)
    )
    if problems:
        return TrackReport(tuple(problems)), None  # the track is not built from a bad design
    points, rounding = _build_points(spec)
    if points is None:
        return TrackReport(tuple(rounding)), None
    report = check_centerline(points, spec.width, spec.crossings)
    return report, points if report.ok else None


def validate_spec(spec: TrackSpec) -> TrackReport:
    """Check a spec against everything the generator guarantees. Never raises for a bad
    design: the reasons come back in ``TrackReport.problems`` (Spanish, one per line)."""
    return _analyze(spec)[0]


def build_track(spec: TrackSpec) -> Track:
    """Validate ``spec`` and build its ``Track``. Raises ``TrackFileError`` if it is invalid."""
    report, points = _analyze(spec)
    if points is None:
        raise TrackFileError(list(report.problems))
    return Track(
        points,
        spec.width,
        name=spec.name,
        clearance=CLEARANCE,
        decoration_seed=spec.decoration_seed,
        crossings=find_crossings(points) if spec.crossings else (),
    )


def load_track_file(path: Path | str) -> Track:
    """Read, validate and build a track file. The single entry point for loading custom
    tracks (evaluation, training configs, manual driving and the viewers).

    Raises ``TrackFileError`` (a ``ValueError``) with a readable message otherwise.
    """
    spec = load_spec(path)
    try:
        return build_track(spec)
    except TrackFileError as error:
        raise TrackFileError([f"{path}: {p}" for p in error.problems]) from None
