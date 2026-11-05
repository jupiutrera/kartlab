"""Closed tracks: a smooth centerline with a constant width.

World units are metres. ``x`` grows to the right and ``y`` upwards, and headings are
counterclockwise radians from the +x axis. Lateral offsets are positive to the left
of the driving direction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

SAMPLE_SPACING = 1.0  # target distance between centerline samples (m)
SMOOTHING = 4.0  # gaussian smoothing of the outline: curvature ramps in and out (m)
START_STRAIGHT = 60.0  # length of the straight stretch centred on the start line (m)
MAX_ATTEMPTS = 200


@dataclass(frozen=True)
class TrackParams:
    """Difficulty knobs for procedurally generated tracks."""

    length: float = 900.0  # centerline length (m)
    width: float = 12.0  # asphalt width (m)
    curviness: float = 0.5  # 0 = smooth and round, 1 = many tight corners
    min_radius: float = 14.0  # tightest corner radius allowed on the centerline (m)
    clearance: float = 8.0  # minimum grass gap between non-adjacent parts of the track (m)

    def __post_init__(self) -> None:
        if not 0.0 <= self.curviness <= 1.0:
            raise ValueError("curviness must be in [0, 1]")
        if self.min_radius <= self.width / 2:
            raise ValueError("min_radius must exceed half the width or the inner edge folds")


# Level crossings (figure-eight tracks). Two stretches of the centerline cross on the same
# plane: there is no bridge, so the asphalt of both is one surface at the crossing.
CROSSING_MIN_ANGLE = math.radians(60.0)  # the branches must meet at least this steeply
CROSSING_MAX_CURVATURE = 1 / 150  # both branches are straight within the crossing zone (1/m)
CROSSING_START_MARGIN = START_STRAIGHT / 2 + 10.0  # start line to crossing zone, at least (m)
CROSSING_MERGE = 3.0  # intersections closer than this are one crossing (m)
BRANCH_GUARD = 4.0  # arc distance beyond which two stretches are different branches, in widths


def crossing_zone(width: float, clearance: float) -> float:
    """Radius (m) of the disc around a crossing where the two branches may touch.

    Outside it two straight branches meeting at 60 degrees or more are at least
    ``width + clearance`` apart, so only inside it is the usual grass gap waived.
    """
    return 2.0 * width + clearance


@dataclass(frozen=True, slots=True)
class Crossing:
    """Where two non-adjacent stretches of the centerline cross."""

    x: float
    y: float
    index_a: int  # segment of the first branch (the smaller index)
    index_b: int  # segment of the second branch
    s_a: float  # arc position on the first branch
    s_b: float
    angle: float  # acute angle between the branches, in (0, pi/2]


@dataclass(frozen=True, slots=True)
class TrackPosition:
    """A point of the plane expressed relative to the track."""

    index: int  # centerline segment (sample ``index`` to ``index + 1``)
    s: float  # arc position along the centerline, in [0, length)
    lateral: float  # signed distance from the centerline, positive to the left


class Track:
    """A closed centerline sampled roughly every metre, with a constant width.

    Sample 0 is the start/finish line. The last sample connects back to the first.
    """

    def __init__(
        self,
        points: np.ndarray,
        width: float,
        name: str,
        seed: int | None = None,
        clearance: float = 0.0,
        decoration_seed: int = 0,
        crossings: Sequence[Crossing] = (),
    ) -> None:
        pts = np.ascontiguousarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 8:
            raise ValueError("points must be an (N, 2) array with N >= 8")
        self.points = pts
        self.width = float(width)
        self.half_width = self.width / 2
        self.name = name
        self.seed = seed
        # Seed for the scenery of the pseudo-3D view (props, texture noise); it never
        # affects the simulation.
        self.decoration_seed = int(decoration_seed)
        # Guaranteed grass gap between sections: closer than half_width + clearance to
        # the centerline, a point cannot be on another section's asphalt.
        self.clearance = float(clearance)
        # Level crossings (empty for every procedural track). Tracks that have some use the
        # branch-following ``locate`` and ``contains``; the rest behave exactly as before.
        self.crossings: tuple[Crossing, ...] = tuple(crossings)

        seg = np.roll(pts, -1, axis=0) - pts
        self.segment_lengths = np.hypot(seg[:, 0], seg[:, 1])
        self.segment_dirs = seg / self.segment_lengths[:, None]
        self.s = np.concatenate([[0.0], np.cumsum(self.segment_lengths)[:-1]])
        self.length = float(self.segment_lengths.sum())

        chord = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
        self.tangents = chord / np.linalg.norm(chord, axis=1, keepdims=True)
        self.normals = np.stack([-self.tangents[:, 1], self.tangents[:, 0]], axis=1)
        self.headings = np.arctan2(self.tangents[:, 1], self.tangents[:, 0])
        self.curvature = curvature(pts)
        self.left_edge = pts + self.normals * self.half_width
        self.right_edge = pts - self.normals * self.half_width

        # Plain Python copies for the per-step scalar code (much faster than numpy scalars).
        self._x = pts[:, 0].tolist()
        self._y = pts[:, 1].tolist()
        self._s = self.s.tolist()
        self._seg_len = self.segment_lengths.tolist()
        self._dir_x = self.segment_dirs[:, 0].tolist()
        self._dir_y = self.segment_dirs[:, 1].tolist()
        self._n = len(pts)

    def __len__(self) -> int:
        return self._n

    def __repr__(self) -> str:
        return f"Track({self.name!r}, length={self.length:.0f} m, width={self.width:.1f} m)"

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)`` of the asphalt."""
        edges = np.concatenate([self.left_edge, self.right_edge])
        (xmin, ymin), (xmax, ymax) = edges.min(axis=0), edges.max(axis=0)
        return float(xmin), float(ymin), float(xmax), float(ymax)

    def wrap_s(self, s: float) -> float:
        return s % self.length

    def index_at(self, s: float) -> int:
        """Segment containing arc position ``s``."""
        i = int(np.searchsorted(self.s, self.wrap_s(s), side="right")) - 1
        return max(i, 0)

    def pose_at(self, s: float, lateral: float = 0.0) -> tuple[float, float, float]:
        """``(x, y, heading)`` at arc position ``s`` and a lateral offset."""
        s = self.wrap_s(s)
        i = self.index_at(s)
        t = (s - self._s[i]) / self._seg_len[i]
        dx, dy = self._dir_x[i], self._dir_y[i]
        x = self._x[i] + dx * self._seg_len[i] * t - dy * lateral
        y = self._y[i] + dy * self._seg_len[i] * t + dx * lateral
        return x, y, math.atan2(dy, dx)

    def heading_at_index(self, index: int) -> float:
        return math.atan2(self._dir_y[index], self._dir_x[index])

    def locate(self, x: float, y: float, hint: int | None = None) -> TrackPosition:
        """Project a point onto the centerline.

        With ``hint`` (the index from the previous step) the search follows the
        centerline locally, so the result stays continuous while the kart moves. It
        only jumps to another part of the track when the point is clearly on that
        other part's asphalt.
        """
        if hint is None:
            return self._project(x, y, self._nearest_index(x, y))
        if self.crossings:
            return self._locate_branch(x, y, hint % self._n)
        pos = self._project(x, y, self._climb(x, y, hint % self._n))
        if abs(pos.lateral) > self.half_width + self.clearance:
            nearest = self._project(x, y, self._nearest_index(x, y))
            if abs(nearest.lateral) <= self.half_width:
                return nearest
        return pos

    def locate_following(self, x: float, y: float, hint: int | None) -> TrackPosition:
        """``locate`` for a point that moves from frame to frame (cameras, overlays).

        Tracks with crossings follow the branch ``hint`` belongs to, so the result does not
        flip to the other road when the point passes the crossing. Every other track gives
        exactly ``locate(x, y)`` and ignores the hint, as the unhinted callers always did.
        """
        if self.crossings and hint is not None:
            return self.locate(x, y, hint)
        return self.locate(x, y)

    def contains(self, x: float, y: float) -> bool:
        """True if the point is on the asphalt of any part of the track.

        Unlike ``position.lateral`` of a located point (which is relative to one branch),
        this is true on the second road of a level crossing too.
        """
        return abs(self._project(x, y, self._nearest_index(x, y)).lateral) <= self.half_width

    def _branch_guard(self) -> int:
        """Samples between two points of the same branch at most; farther is another branch."""
        return math.ceil(BRANCH_GUARD * self.width * self._n / self.length)

    def _locate_branch(self, x: float, y: float, hint: int) -> TrackPosition:
        """``locate`` for tracks with crossings: stay on the hinted branch while the point is
        on its asphalt; when it leaves it, move to the other branch only if the point is on
        that branch's asphalt (a turn inside the crossing). Otherwise it stays on grass,
        projected on the hinted branch."""
        if self._dist2(x, y, hint) > (2.0 * self.width + 10.0) ** 2:
            return self._project(x, y, self._nearest_index(x, y))  # not a continuous move
        pos = self._project(x, y, self._climb(x, y, hint))
        if abs(pos.lateral) <= self.half_width:
            return pos
        d2 = (self.points[:, 0] - x) ** 2 + (self.points[:, 1] - y) ** 2
        gap = np.abs(np.arange(self._n) - pos.index)
        d2[np.minimum(gap, self._n - gap) <= self._branch_guard()] = np.inf
        other = self._project(x, y, int(np.argmin(d2)))
        return other if abs(other.lateral) <= self.half_width else pos

    def edge_covered(self) -> tuple[np.ndarray, np.ndarray]:
        """Which left and right edge samples lie on another branch's asphalt (boolean arrays).

        Only crossings have any: those edge stretches are not edges at all, since the road
        of the other branch passes over them. Always all False for tracks without crossings.
        """
        covered = [np.zeros(self._n, dtype=bool), np.zeros(self._n, dtype=bool)]
        if not self.crossings:
            return covered[0], covered[1]
        idx = np.arange(self._n)
        gap = np.abs(idx[:, None] - idx[None, :])
        far = np.minimum(gap, self._n - gap) > self._branch_guard()
        for k, edge in enumerate((self.left_edge, self.right_edge)):
            d2 = ((edge[:, None, :] - self.points[None, :, :]) ** 2).sum(axis=2)
            covered[k] = np.where(far, d2, np.inf).min(axis=1) < (self.half_width + 0.1) ** 2
        return covered[0], covered[1]

    def _dist2(self, x: float, y: float, i: int) -> float:
        dx = x - self._x[i]
        dy = y - self._y[i]
        return dx * dx + dy * dy

    def _nearest_index(self, x: float, y: float) -> int:
        d2 = (self.points[:, 0] - x) ** 2 + (self.points[:, 1] - y) ** 2
        return int(np.argmin(d2))

    def _climb(self, x: float, y: float, i: int) -> int:
        """Walk along the samples while the distance to (x, y) decreases."""
        n = self._n
        best = self._dist2(x, y, i)
        while True:
            nxt = (i + 1) % n
            d = self._dist2(x, y, nxt)
            if d < best:
                i, best = nxt, d
                continue
            prv = (i - 1) % n
            d = self._dist2(x, y, prv)
            if d < best:
                i, best = prv, d
                continue
            return i

    def _project(self, x: float, y: float, i: int) -> TrackPosition:
        """Project onto the closer of the two segments around sample ``i``.

        The segment starting at ``i`` goes first, so a point exactly on sample ``i``
        gets ``s[i]`` (0 on the start line) instead of the end of the previous segment.
        """
        n = self._n
        best: TrackPosition | None = None
        best_d2 = math.inf
        for seg in (i, (i - 1) % n):
            ax, ay = self._x[seg], self._y[seg]
            dx, dy = self._dir_x[seg], self._dir_y[seg]
            length = self._seg_len[seg]
            along = min(max((x - ax) * dx + (y - ay) * dy, 0.0), length)
            px, py = ax + dx * along, ay + dy * along
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 < best_d2:
                lateral = dx * (y - ay) - dy * (x - ax)
                s = (self._s[seg] + along) % self.length
                best, best_d2 = TrackPosition(seg, s, lateral), d2
        assert best is not None
        return best


def curvature(points: np.ndarray) -> np.ndarray:
    """Signed curvature (1/m) at each sample of a closed polyline; positive turns left."""
    seg = np.roll(points, -1, axis=0) - points
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    heading = np.arctan2(seg[:, 1], seg[:, 0])
    turn = np.angle(np.exp(1j * (heading - np.roll(heading, 1))))
    return turn / ((seg_len + np.roll(seg_len, 1)) / 2)


def closed_length(points: np.ndarray) -> float:
    seg = np.roll(points, -1, axis=0) - points
    return float(np.hypot(seg[:, 0], seg[:, 1]).sum())


def resample_closed(points: np.ndarray, count: int) -> np.ndarray:
    """``count`` points evenly spaced along a closed polyline."""
    loop = np.vstack([points, points[:1]])
    seg = np.diff(loop, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    targets = np.linspace(0.0, cum[-1], count, endpoint=False)
    return np.stack([np.interp(targets, cum, loop[:, 0]), np.interp(targets, cum, loop[:, 1])], 1)


def smooth_closed(points: np.ndarray, sigma: float, spacing: float) -> np.ndarray:
    """Periodic gaussian smoothing (``sigma`` in metres) of evenly spaced closed points."""
    freqs = np.fft.rfftfreq(len(points), d=spacing)
    gain = np.exp(-0.5 * (2 * np.pi * freqs * sigma) ** 2)
    spectrum = np.fft.rfft(points, axis=0) * gain[:, None]
    return np.fft.irfft(spectrum, n=len(points), axis=0)


def has_clearance(points: np.ndarray, width: float, clearance: float) -> bool:
    """True if parts of the track far apart along the centerline stay far apart in space.

    Samples whose arc separation exceeds a quarter turn around the required gap must
    be at least ``width + clearance`` apart, so the asphalt never overlaps and a strip
    of grass always separates neighbouring sections.
    """
    sub = points[::2]
    spacing = closed_length(points) / len(sub)
    required = width + clearance
    d2 = np.sum(sub**2, axis=1)[:, None] + np.sum(sub**2, axis=1)[None, :] - 2.0 * sub @ sub.T
    idx = np.arange(len(sub))
    sep = np.abs(idx[:, None] - idx[None, :])
    arc = np.minimum(sep, len(sub) - sep) * spacing
    far = arc > (math.pi / 2) * required
    return bool(np.all(d2[far] >= required**2))


def find_crossings(points: np.ndarray) -> list[Crossing]:
    """Every place where two non-adjacent stretches of a closed polyline cross.

    A crossing through a sample point counts once, and a grazing contact that yields
    several neighbouring segment pairs is merged into one crossing. Sorted by the arc
    position of the first branch.
    """
    n = len(points)
    d = np.roll(points, -1, axis=0) - points
    seg_len = np.hypot(d[:, 0], d[:, 1])
    arc = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]])
    idx = np.arange(n)
    found: list[Crossing] = []
    for start in range(0, n, 256):
        rows = slice(start, min(start + 256, n))
        # Segment i of the block is a_i + t d_i; segment j is a_j + u d_j.
        dx_i, dy_i = d[rows, 0, None], d[rows, 1, None]
        dx_j, dy_j = d[None, :, 0], d[None, :, 1]
        denom = dx_i * dy_j - dy_i * dx_j
        px = points[None, :, 0] - points[rows, 0, None]
        py = points[None, :, 1] - points[rows, 1, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (px * dy_j - py * dx_j) / denom
            u = (px * dy_i - py * dx_i) / denom
        sep = np.abs(idx[rows, None] - idx[None, :])
        hit = (t >= 0.0) & (t < 1.0) & (u >= 0.0) & (u < 1.0) & (np.minimum(sep, n - sep) > 1)
        for row, j in np.argwhere(hit).tolist():
            i = row + start
            if i >= j:
                continue  # each pair once, smaller index first
            ti, uj = float(t[row, j]), float(u[row, j])
            cos = abs(float(d[i] @ d[j])) / float(seg_len[i] * seg_len[j])
            found.append(
                Crossing(
                    float(points[i, 0] + ti * d[i, 0]),
                    float(points[i, 1] + ti * d[i, 1]),
                    i,
                    j,
                    float(arc[i] + ti * seg_len[i]),
                    float(arc[j] + uj * seg_len[j]),
                    math.acos(min(cos, 1.0)),
                )
            )
    merged: list[Crossing] = []
    for c in sorted(found, key=lambda c: (c.s_a, c.s_b)):
        if not any(math.hypot(c.x - m.x, c.y - m.y) < CROSSING_MERGE for m in merged):
            merged.append(c)
    return merged


def has_clearance_with_crossings(
    points: np.ndarray, width: float, clearance: float, crossings: Sequence[Crossing]
) -> bool:
    """``has_clearance`` for a track whose centerline crosses itself at ``crossings``.

    The same rule (far-apart parts keep ``width + clearance`` between them), except that
    two samples inside the same crossing zone (``crossing_zone``) are not compared with
    each other: that is where the branches meet. Everywhere else the grass gap applies,
    so no other near-touching sections are accepted. With no crossings it is exactly
    ``has_clearance``.
    """
    sub = points[::2]
    spacing = closed_length(points) / len(sub)
    required = width + clearance
    d2 = np.sum(sub**2, axis=1)[:, None] + np.sum(sub**2, axis=1)[None, :] - 2.0 * sub @ sub.T
    idx = np.arange(len(sub))
    sep = np.abs(idx[:, None] - idx[None, :])
    arc = np.minimum(sep, len(sub) - sep) * spacing
    far = arc > (math.pi / 2) * required
    zone = crossing_zone(width, clearance)
    for c in crossings:
        inside = np.hypot(sub[:, 0] - c.x, sub[:, 1] - c.y) <= zone
        far &= ~(inside[:, None] & inside[None, :])
    return bool(np.all(d2[far] >= required**2))


def finish_crossing_centerline(
    path: np.ndarray,
    width: float,
    clearance: float,
    length: float | None = None,
    start: tuple[float, float] | None = None,
) -> np.ndarray:
    """``finish_centerline`` for paths that cross themselves, without its validation.

    It resamples and smooths the same way, and rolls the samples so that sample 0 (the
    start line) is the sample nearest to ``start`` when given (the author's choice, the same
    on every platform), or else the straightest stretch that stays clear of every crossing
    zone. The caller validates the result (``find_crossings``,
    ``has_clearance_with_crossings``, whether ``start`` is on the track).
    """
    target = length if length is not None else closed_length(path)
    path = path * (target / closed_length(path))
    count = max(int(round(target / SAMPLE_SPACING)), 8)
    pts = smooth_closed(resample_closed(path, count), SMOOTHING, target / count)
    pts *= target / closed_length(pts)
    if start is not None:
        nearest = int(np.argmin(np.hypot(pts[:, 0] - start[0], pts[:, 1] - start[1])))
        return np.roll(pts, -nearest, axis=0)
    bend = np.abs(curvature(pts))
    keep_out = crossing_zone(width, clearance) + CROSSING_START_MARGIN
    for c in find_crossings(pts):
        dist = np.hypot(pts[:, 0] - c.x, pts[:, 1] - c.y)
        # Among equally straight stretches prefer the one farthest from the crossing.
        bend = bend + 0.002 * np.clip(1.0 - dist / (3.0 * keep_out), 0.0, 1.0)
        bend = np.where(dist < keep_out, 1.0, bend)
    return np.roll(pts, -_straightest_index(bend, target / count), axis=0)


def filleted_polygon(
    vertices: np.ndarray, radii: np.ndarray, step: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """Closed path of straight edges joined by a circular arc at each vertex.

    Each arc uses the vertex's radius, shrunk if needed so that consecutive arcs never
    use more than 90 % of an edge. Returns the dense path and the radii actually used.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    edges = np.roll(verts, -1, axis=0) - verts
    edge_len = np.hypot(edges[:, 0], edges[:, 1])
    out_dir = edges / edge_len[:, None]
    in_dir = np.roll(out_dir, 1, axis=0)
    cross = in_dir[:, 0] * out_dir[:, 1] - in_dir[:, 1] * out_dir[:, 0]
    dot = np.sum(in_dir * out_dir, axis=1)
    turn = np.arctan2(cross, dot)
    tan_half = np.tan(np.abs(turn) / 2)
    tangent = np.minimum(
        np.asarray(radii, dtype=np.float64) * tan_half,
        0.45 * np.minimum(edge_len, np.roll(edge_len, 1)),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        used = np.where(tan_half > 1e-9, tangent / tan_half, np.inf)

    pieces = []
    n = len(verts)
    for i in range(n):
        arc_start = verts[i] - in_dir[i] * tangent[i]
        arc_end = verts[i] + out_dir[i] * tangent[i]
        if np.isfinite(used[i]):
            left = np.array([-in_dir[i, 1], in_dir[i, 0]])
            centre = arc_start + left * used[i] * np.sign(turn[i])
            phi0 = math.atan2(arc_start[1] - centre[1], arc_start[0] - centre[0])
            count = max(2, math.ceil(abs(turn[i]) * used[i] / step))
            phi = phi0 + np.linspace(0.0, turn[i], count, endpoint=False)
            pieces.append(centre + used[i] * np.stack([np.cos(phi), np.sin(phi)], axis=1))
        nxt = verts[(i + 1) % n] - out_dir[i] * tangent[(i + 1) % n]
        count = max(1, math.ceil(float(np.linalg.norm(nxt - arc_end)) / step))
        t = np.linspace(0.0, 1.0, count, endpoint=False)[:, None]
        pieces.append(arc_end + (nxt - arc_end) * t)
    return np.concatenate(pieces), used


def finish_centerline(
    path: np.ndarray,
    width: float,
    min_radius: float,
    clearance: float,
    length: float | None = None,
) -> np.ndarray | None:
    """Resample, smooth and validate a dense closed path, or return None if it is invalid.

    With ``length`` the path is scaled to that centerline length. The returned samples
    start at the middle of the straightest stretch, which becomes the start/finish line.
    """
    target = length if length is not None else closed_length(path)
    path = path * (target / closed_length(path))
    count = max(int(round(target / SAMPLE_SPACING)), 8)
    pts = smooth_closed(resample_closed(path, count), SMOOTHING, target / count)
    pts *= target / closed_length(pts)
    kappa = curvature(pts)
    if np.max(np.abs(kappa)) * min_radius > 1.0:
        return None
    if not has_clearance(pts, width, clearance):
        return None
    return np.roll(pts, -_straightest_index(kappa, target / count), axis=0)


def _straightest_index(kappa: np.ndarray, spacing: float) -> int:
    window = max(int(round(START_STRAIGHT / spacing)), 1)
    bend = np.abs(kappa)
    tiled = np.concatenate([bend[-(window // 2) :], bend, bend[: window - window // 2]])
    sums = np.convolve(tiled, np.ones(window), mode="valid")[: len(bend)]
    return int(np.argmin(sums))


def random_layout(rng: np.random.Generator, params: TrackParams) -> tuple[np.ndarray, np.ndarray]:
    """Random polygon (vertices and corner radii) whose filleted outline is about ``length``.

    The polygon is star-shaped around the origin (so it never crosses itself) and then
    stretched along a random axis, which gives long straights. More curviness means more
    corners, deeper bays and tighter radii.
    """
    c = params.curviness
    count = 5 + int(round(6 * c)) + int(rng.integers(0, 3))
    angles = (np.arange(count) + rng.uniform(-0.35, 0.35, count)) * (2 * np.pi / count)
    reach = 1.0 + rng.uniform(-1.0, 1.0, count) * (0.1 + 0.35 * c)
    verts = np.stack([reach * np.cos(angles), reach * np.sin(angles)], axis=1)

    axis = rng.uniform(0.0, np.pi)
    stretch = rng.uniform(1.0, 1.8)
    ca, sa = math.cos(axis), math.sin(axis)
    rot = np.array([[ca, -sa], [sa, ca]])
    verts = verts @ rot @ np.diag([stretch, 1.0]) @ rot.T
    verts *= params.length / closed_length(verts)

    low = params.min_radius + 30.0 * (1.0 - c)
    high = 30.0 + 90.0 * (1.0 - c)
    radii = low + (max(high, low) - low) * rng.uniform(0.0, 1.0, count) ** 1.5
    return verts, radii


def generate_track(seed: int, params: TrackParams | None = None, name: str | None = None) -> Track:
    """Deterministic procedural track for ``seed``: same seed and params, same track."""
    params = params or TrackParams()
    rng = np.random.default_rng(seed)
    for _ in range(MAX_ATTEMPTS):
        verts, radii = random_layout(rng, params)
        rotation = rng.uniform(0.0, 2 * np.pi)
        clockwise = bool(rng.integers(2))
        path, used = filleted_polygon(verts, radii)
        if np.min(used) < params.min_radius:
            continue
        pts = finish_centerline(
            path, params.width, params.min_radius, params.clearance, length=params.length
        )
        if pts is None:
            continue
        c, s = math.cos(rotation), math.sin(rotation)
        pts = pts @ np.array([[c, s], [-s, c]])
        if clockwise:
            # Reverse the driving direction but keep sample 0 on the start line.
            pts = np.roll(pts[::-1], 1, axis=0)
        return Track(
            pts,
            params.width,
            name=name or f"Aleatorio #{seed}",
            seed=seed,
            clearance=params.clearance,
        )
    raise RuntimeError(f"could not generate a valid track for seed {seed} with {params}")
