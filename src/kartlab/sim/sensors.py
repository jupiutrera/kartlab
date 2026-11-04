"""Distance rays from the kart to the track edges, in a fan around its heading."""

from __future__ import annotations

import math

import numpy as np

from kartlab.sim.track import Track

EDGE_STRIDE = 2  # use every 2nd edge sample: ~2 m chords, < 4 cm off the true edge


class RaySensor:
    """Casts ``count`` rays spread over ``fov`` radians and returns the distance to the
    first track edge each one crosses, capped at ``max_range``.

    Only edge segments within an arc window around the kart's centerline index are
    tested. A straight ray that stays on the asphalt cannot cover much more arc length
    than its own length, so a window of 1.5 times the range (plus a margin) always
    contains the first crossing. The edges are stored with wrapped copies at both
    ends, so every window is a contiguous slice (a view, no copies).
    """

    def __init__(
        self, track: Track, count: int = 11, fov: float = math.pi, max_range: float = 60.0
    ) -> None:
        if count < 1:
            raise ValueError("count must be at least 1")
        self.count = count
        self.max_range = float(max_range)
        angles = np.linspace(-fov / 2, fov / 2, count) if count > 1 else np.zeros(1)
        self.angles = angles

        edges = np.stack([track.left_edge[::EDGE_STRIDE], track.right_edge[::EDGE_STRIDE]])
        m = edges.shape[1]
        spacing = track.length / m
        window = min(math.ceil((1.5 * self.max_range + 10.0) / spacing), m // 2)
        vectors = np.roll(edges, -1, axis=1) - edges
        wrapped = np.r_[np.arange(m - window, m), np.arange(m), np.arange(window + 1)]
        self._sx = np.ascontiguousarray(edges[:, wrapped, 0])
        self._sy = np.ascontiguousarray(edges[:, wrapped, 1])
        self._ex = np.ascontiguousarray(vectors[:, wrapped, 0])
        self._ey = np.ascontiguousarray(vectors[:, wrapped, 1])
        self._width = 2 * window + 1
        self._m = m

    def cast(self, x: float, y: float, heading: float, index: int) -> np.ndarray:
        """Distances (m) along each ray, for a kart at (x, y) near centerline ``index``."""
        j = (index // EDGE_STRIDE) % self._m
        window = slice(j, j + self._width)
        px = self._sx[:, window] - x
        py = self._sy[:, window] - y
        ex = self._ex[:, window]
        ey = self._ey[:, window]
        ang = heading + self.angles
        dx = np.cos(ang)[:, None, None]
        dy = np.sin(ang)[:, None, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = dx * ey - dy * ex
            t = (px * ey - py * ex) / denom
            u = (px * dy - py * dx) / denom
        hit = (t > 0.0) & (u >= 0.0) & (u <= 1.0)
        dist = np.where(hit, t, np.inf).min(axis=(1, 2))
        return np.minimum(dist, self.max_range)
