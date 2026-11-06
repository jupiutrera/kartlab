"""Where a track design fails: the validator's problems with locations, for the editor.

``validate_spec`` says *what* is wrong in Spanish; the editor also needs to know *where*
(which corners, which point of the track) to highlight it. ``analyze`` runs the very same
checks as ``validate_spec`` (it calls the same functions, so the messages are identical)
and attaches the corners and the world points each problem refers to. It does not change
the validator: ``trackfile.py`` stays the single source of truth for what is valid.

Two levels, because the full validation costs ~100-250 ms (it compares every part of the
track with every other):

- ``analyze(spec, full=False)``, the *quick* level, runs everything that is cheap (the
  parameters, the corner polygon, the rounding, the length and the tightest curve) and
  builds the centerline: ~2-15 ms. Everything it reports is also reported by the full
  level, but a design with no quick problems may still fail the full level.
- ``analyze(spec)`` is the complete validation, with the same verdict as ``validate_spec``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from kartlab.sim.track import closed_length, crossing_zone, curvature, find_crossings
from kartlab.sim.trackfile import (
    CLEARANCE,
    MAX_COORDINATE,
    MAX_CORNERS,
    MAX_LENGTH,
    MIN_CORNERS,
    MIN_LENGTH,
    MIN_RADIUS,
    TrackReport,
    TrackSpec,
    _build_points,
    _check_parameters,
    _check_polygon,
    _crossing,
    _length_problem,
    check_centerline,
)

Point = tuple[float, float]

_CORNER = re.compile(r"^esquina (\d+):")
_EDGE_TOO_SHORT = re.compile(r"^las esquinas (\d+) y (\d+) están")
_POLYGON_CROSS = re.compile(
    r"^el circuito se cruza a sí mismo: el tramo de la esquina (\d+) a la (\d+) "
    r"corta el de la (\d+) a la (\d+)"
)
_COORDINATES = re.compile(r"\((-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\)")


@dataclass(frozen=True)
class Issue:
    """One validator problem and where it is: ``corners`` are 0-based indices into
    ``TrackSpec.corners``, ``points`` are world positions (m) and ``edges`` are pairs of
    corner indices. Any may be empty (a bad width or length has no place)."""

    message: str
    corners: tuple[int, ...] = ()
    points: tuple[Point, ...] = ()
    edges: tuple[tuple[int, int], ...] = ()  # polygon edges (corner to corner) at fault


@dataclass(frozen=True)
class Analysis:
    """A design analysed: the verdict, the located problems and the centerline.

    ``points`` is the centerline whenever it could be built, valid or not (the editor
    draws it either way); it is the same array ``build_track`` would use when valid.
    ``complete`` is False for a quick analysis, whose verdict is only partial."""

    report: TrackReport
    issues: tuple[Issue, ...]
    points: np.ndarray | None
    complete: bool

    @property
    def ok(self) -> bool:
        """No problem found (and, if ``complete``, the track is valid)."""
        return not self.issues


def analyze(spec: TrackSpec, full: bool = True) -> Analysis:
    """Analyse a design; never raises for a bad one. See the module docstring."""
    problems = _check_parameters(spec)
    if not MIN_CORNERS <= len(spec.corners) <= MAX_CORNERS or any(
        abs(v) > MAX_COORDINATE for c in spec.corners for v in c[:2]
    ):
        return _result(spec, problems, None, TrackReport(tuple(problems)), full)
    problems += _check_polygon(
        np.array(spec.corners, dtype=np.float64)[:, :2], allow_crossings=bool(spec.crossings)
    )
    if problems:
        return _result(spec, problems, None, TrackReport(tuple(problems)), full)
    points, rounding = _build_points(spec)
    if points is None:
        return _result(spec, rounding, None, TrackReport(tuple(rounding)), full)
    report = (
        check_centerline(points, spec.width, spec.crossings, spec.bridge)
        if full
        else _quick_report(points)
    )
    return _result(spec, list(report.problems), points, report, full)


def _quick_report(points: np.ndarray) -> TrackReport:
    """The cheap part of ``check_centerline``: length and tightest curve."""
    length = closed_length(points)
    kappa = float(np.max(np.abs(curvature(points))))
    min_radius = 1.0 / kappa if kappa > 0 else math.inf
    problems = []
    if not MIN_LENGTH <= length <= MAX_LENGTH:
        problems.append(_length_problem(length))
    if kappa * MIN_RADIUS > 1.0:
        problems.append(
            f"curva demasiado cerrada: radio mínimo {min_radius:.1f} m en el trazado "
            f"(mínimo {MIN_RADIUS:g} m); ensancha alguna esquina o aléjala de las vecinas"
        )
    return TrackReport(tuple(problems), length, min_radius)


# -- locations ---------------------------------------------------------------------------


def _result(
    spec: TrackSpec,
    problems: Sequence[str],
    points: np.ndarray | None,
    report: TrackReport,
    full: bool,
) -> Analysis:
    issues = tuple(_locate(spec, message, points) for message in problems)
    # A design that fails before its centerline exists gets the same verdict at either level.
    return Analysis(report, issues, points, full or points is None)


def _nearest_corners(spec: TrackSpec, *where: Point) -> tuple[int, ...]:
    corners = np.array(spec.corners, dtype=np.float64)[:, :2]
    found: list[int] = []
    for x, y in where:
        index = int(np.argmin(np.hypot(corners[:, 0] - x, corners[:, 1] - y)))
        if index not in found:
            found.append(index)
    return tuple(found)


def _segment_intersection(a: Point, b: Point, c: Point, d: Point) -> Point:
    """Where segment ab meets cd (the midpoint of the two midpoints if they do not)."""
    r = (b[0] - a[0], b[1] - a[1])
    s = (d[0] - c[0], d[1] - c[1])
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) > 1e-12:
        t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / denom
        return a[0] + t * r[0], a[1] + t * r[1]
    return (a[0] + b[0] + c[0] + d[0]) / 4, (a[1] + b[1] + c[1] + d[1]) / 4


def _locate(spec: TrackSpec, message: str, points: np.ndarray | None) -> Issue:
    count = len(spec.corners)
    if found := _CORNER.match(message):
        index = int(found[1]) - 1
        if 0 <= index < count:
            x, y, _ = spec.corners[index]
            return Issue(message, (index,), ((x, y),))
    if found := _EDGE_TOO_SHORT.match(message):
        a, b = int(found[1]) - 1, int(found[2]) - 1
        return Issue(message, (a, b), tuple(spec.corners[i][:2] for i in (a, b)), ((a, b),))
    if found := _POLYGON_CROSS.match(message):
        a, b, c, d = (int(g) - 1 for g in found.groups())
        p = _segment_intersection(*(spec.corners[i][:2] for i in (a, b, c, d)))
        return Issue(message, (a, b, c, d), (p,), ((a, b), (c, d)))
    if points is not None:
        if message.startswith("curva demasiado cerrada"):
            i = int(np.argmax(np.abs(curvature(points))))
            where = (float(points[i, 0]), float(points[i, 1]))
            return Issue(message, _nearest_corners(spec, where), (where,))
        if message == "la línea central se cruza a sí misma":
            return _locate_self_crossing(spec, message, points)
        if message.startswith("dos partes del circuito están demasiado cerca"):
            return _locate_closest_parts(spec, message, points)
    if found := _COORDINATES.search(message):
        where = (float(found[1]), float(found[2]))
        return Issue(message, _nearest_corners(spec, where) if count else (), (where,))
    return Issue(message)


def _locate_self_crossing(spec: TrackSpec, message: str, points: np.ndarray) -> Issue:
    pair = _crossing(points, stride=2)
    if pair is None:
        return Issue(message)
    pts = points[::2]
    n = len(pts)
    a, b = pair
    p = _segment_intersection(
        tuple(pts[a]), tuple(pts[(a + 1) % n]), tuple(pts[b]), tuple(pts[(b + 1) % n])
    )
    return Issue(message, _nearest_corners(spec, p), (p,))


def _locate_closest_parts(spec: TrackSpec, message: str, points: np.ndarray) -> Issue:
    """The two samples of non-neighbouring sections that come closest to each other
    (the pairs ``has_clearance`` compares, minus the declared crossings' zones)."""
    sub = points[::2]
    spacing = closed_length(points) / len(sub)
    required = spec.width + CLEARANCE
    idx = np.arange(len(sub))
    sep = np.abs(idx[:, None] - idx[None, :])
    arc = np.minimum(sep, len(sub) - sep) * spacing
    far = arc > (math.pi / 2) * required
    if spec.crossings:
        zone = crossing_zone(spec.width, CLEARANCE)
        for c in find_crossings(points):
            inside = np.hypot(sub[:, 0] - c.x, sub[:, 1] - c.y) <= zone
            far &= ~(inside[:, None] & inside[None, :])
    if not far.any():
        return Issue(message)
    d2 = np.sum(sub**2, axis=1)[:, None] + np.sum(sub**2, axis=1)[None, :] - 2.0 * sub @ sub.T
    d2 = np.where(far, d2, np.inf)
    i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
    a = (float(sub[i, 0]), float(sub[i, 1]))
    b = (float(sub[j, 0]), float(sub[j, 1]))
    return Issue(message, _nearest_corners(spec, a, b), (a, b))
