"""Floor texture generated from a track: palette indices plus mipmaps.

Everything is painted once at load time, lighting included (darker asphalt and grass
along the edges, bevelled kerbs, baked shadows of the scenery), so a frame only
samples it. Texel rows grow southwards: texel (u, v) covers world x from
``origin_x + u / tpm`` and world y down from ``origin_y - v / tpm``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from kartlab.render.palette import Ink, Palette, game_palette
from kartlab.sim.track import Track

TEXELS_PER_METRE = 5
MARGIN = 48.0  # grass painted around the track bounds (m); beyond it the border repeats
STRIPE_METRES = 5.0  # width of the mowing stripes on the grass
STRIPE_ANGLE = 0.6  # direction of the stripes (rad)
BAND = 4.0  # exact track coordinates are computed this far past the asphalt edge (m)
KERB_WIDTH = 1.0
KERB_STRIPE = 1.5  # kerb colour changes every this many metres along the track
KERB_CURVATURE = 1 / 45  # corners tighter than this radius get kerbs
EDGE_LINE = (0.15, 0.45)  # white line between these distances inside the edge (m)
START_LINE = 1.0  # the chequered line spans this far each side of s = 0 (m)
MIN_MIP_SIZE = 8
SUN_DIRECTION = (0.55, -0.35)  # where shadows fall (world x/y), the same as the scenery's
SHADOW_REACH = 0.35  # shadow length per metre of height
STRIP_ROWS = 128  # texel rows painted (and averaged for the mipmaps) at a time


def hash_noise(u: np.ndarray, v: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic per-texel noise in [0, 1)."""
    h = u.astype(np.uint32) * np.uint32(73856093)
    h ^= v.astype(np.uint32) * np.uint32(19349663)
    h ^= np.uint32(seed * 83492791 & 0xFFFFFFFF)
    h ^= h >> np.uint32(13)
    h *= np.uint32(0x5BD1E995)
    h ^= h >> np.uint32(15)
    return (h & np.uint32(0xFFFF)).astype(np.float32) / 65536.0


BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0


@dataclass
class FloorTexture:
    levels: list[np.ndarray]  # uint8 palette indices, level 0 at full resolution
    origin: tuple[float, float]  # world (x, y) of the top-left corner of texel (0, 0)
    tpm: float = TEXELS_PER_METRE
    palette: Palette = field(default_factory=game_palette, repr=False)

    @property
    def base(self) -> np.ndarray:
        return self.levels[0]

    def to_texel(self, x, y):
        """Fractional texel coordinates (u, v) at level 0."""
        return (np.asarray(x) - self.origin[0]) * self.tpm, (
            self.origin[1] - np.asarray(y)
        ) * self.tpm

    def sample(self, x: float, y: float) -> int:
        u, v = self.to_texel(x, y)
        rows, cols = self.base.shape
        return int(self.base[min(max(int(v), 0), rows - 1), min(max(int(u), 0), cols - 1)])

    def build_mips(self) -> None:
        """Rebuild levels 1+ from level 0: average 2x2 blocks in RGB, back to the palette."""
        self.levels = [self.levels[0]]
        rgb = self.palette.rgb
        while min(self.levels[-1].shape) >= 2 * MIN_MIP_SIZE:
            prev = self.levels[-1]
            h, w = prev.shape[0] // 2 * 2, prev.shape[1] // 2 * 2
            level = np.empty((h // 2, w // 2), dtype=np.uint8)
            for r0 in range(0, h, STRIP_ROWS):  # in strips: RGB of the whole level is large
                r1 = min(r0 + STRIP_ROWS, h)
                colors = rgb[prev[r0:r1, :w]].astype(np.uint16)
                avg = (
                    colors[0::2, 0::2]
                    + colors[1::2, 0::2]
                    + colors[0::2, 1::2]
                    + colors[1::2, 1::2]
                ) >> 2
                level[r0 // 2 : r1 // 2] = self.palette.quantize(avg.astype(np.uint8))
            self.levels.append(level)

    def stamp(self, x: float, y: float, radius: float, index: int, density: float = 1.0) -> None:
        """Paint a dithered disc on level 0 (skid marks, baked shadows)."""
        u, v = self.to_texel(x, y)
        r = radius * self.tpm
        rows, cols = self.base.shape
        u0, u1 = max(int(u - r), 0), min(int(u + r) + 1, cols)
        v0, v1 = max(int(v - r), 0), min(int(v + r) + 1, rows)
        if u0 >= u1 or v0 >= v1:
            return
        vv, uu = np.mgrid[v0:v1, u0:u1]
        inside = (uu + 0.5 - u) ** 2 + (vv + 0.5 - v) ** 2 <= r * r
        if density < 1.0:
            inside &= BAYER4[vv % 4, uu % 4] < density
        self.base[v0:v1, u0:u1][inside] = index


def _coarse_field(track: Track, origin, shape, reach: float, skip: np.ndarray | None = None):
    """Nearest centerline segment and its distance for 1 m cells within ``reach``.

    Segments flagged in ``skip`` (the raised road of a bridge, drawn as geometry) are left out."""
    rows, cols = shape
    dist = np.full(shape, np.inf, dtype=np.float32)
    seg = np.full(shape, -1, dtype=np.int32)
    pts = track.points
    nxt = np.roll(pts, -1, axis=0)
    for i in range(len(pts)):
        if skip is not None and skip[i]:
            continue
        (ax, ay), (bx, by) = pts[i], nxt[i]
        c0 = max(int(math.floor(min(ax, bx) - reach - origin[0])), 0)
        c1 = min(int(math.ceil(max(ax, bx) + reach - origin[0])), cols)
        r0 = max(int(math.floor(origin[1] - max(ay, by) - reach)), 0)
        r1 = min(int(math.ceil(origin[1] - min(ay, by) + reach)), rows)
        if c0 >= c1 or r0 >= r1:
            continue
        cx = origin[0] + np.arange(c0, c1) + 0.5
        cy = origin[1] - (np.arange(r0, r1) + 0.5)
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        px = cx[None, :] - ax
        py = cy[:, None] - ay
        t = np.clip((px * dx + py * dy) / length2, 0.0, 1.0)
        d = np.hypot(px - t * dx, py - t * dy).astype(np.float32)
        block_d = dist[r0:r1, c0:c1]
        better = d < block_d
        block_d[better] = d[better]
        seg[r0:r1, c0:c1][better] = i
    return dist, seg


def _project(track: Track, x: np.ndarray, y: np.ndarray, seg: np.ndarray):
    """Exact (s, signed lateral, segment) of points near segment ``seg`` (checks +-1)."""
    n = len(track)
    best_d = np.full(x.shape, np.inf)
    best_s = np.zeros(x.shape)
    best_lat = np.zeros(x.shape)
    best_seg = seg.copy()
    for offset in (-1, 0, 1):
        i = (seg + offset) % n
        ax, ay = track.points[i, 0], track.points[i, 1]
        dx, dy = track.segment_dirs[i, 0], track.segment_dirs[i, 1]
        length = track.segment_lengths[i]
        along = np.clip((x - ax) * dx + (y - ay) * dy, 0.0, length)
        px, py = ax + dx * along, ay + dy * along
        d = np.hypot(x - px, y - py)
        better = d < best_d
        best_d = np.where(better, d, best_d)
        best_s = np.where(better, track.s[i] + along, best_s)
        best_lat = np.where(better, dx * (y - ay) - dy * (x - ax), best_lat)
        best_seg = np.where(better, i, best_seg)
    return best_s % track.length, best_lat, best_seg


def _refine_crossings(track: Track, x, y, s, lat, nearest):
    """Exact nearest-branch coordinates around each level crossing.

    The coarse field picks one segment per 1 m cell, which can be the wrong road where the
    two are equally near. Here every segment of both branches within reach is tried, so the
    distance is that to the closest of the two roads: the asphalt is their union, the edge
    lines and kerbs follow its outline, and nothing of one road is painted across the other.
    """
    n = len(track)
    reach = 2.5 * track.width
    window = math.ceil((reach + 4.0) * n / track.length)
    s, lat, nearest = s.copy(), lat.copy(), nearest.copy()
    levels = track.levels
    for crossing in track.crossings:
        near = np.hypot(x - crossing.x, y - crossing.y) < reach
        if not near.any():
            continue
        px, py = x[near], y[near]
        best_d = np.full(px.shape, np.inf)
        best_s = np.zeros(px.shape)
        best_lat = np.zeros(px.shape)
        best_seg = np.zeros(px.shape, dtype=np.int64)
        branches = (crossing.index_a, crossing.index_b)
        if levels is not None:  # a bridge: only the road on the ground is part of the floor
            branches = (crossing.index_b if levels.over_first else crossing.index_a,)
        for centre in branches:
            for offset in range(-window, window + 1):
                i = (centre + offset) % n
                ax, ay = track.points[i]
                dx, dy = track.segment_dirs[i]
                length = track.segment_lengths[i]
                along = np.clip((px - ax) * dx + (py - ay) * dy, 0.0, length)
                d = np.hypot(px - (ax + dx * along), py - (ay + dy * along))
                better = d < best_d
                best_d = np.where(better, d, best_d)
                best_s = np.where(better, track.s[i] + along, best_s)
                best_lat = np.where(better, dx * (py - ay) - dy * (px - ax), best_lat)
                best_seg = np.where(better, i, best_seg)
        s[near], lat[near], nearest[near] = best_s % track.length, best_lat, best_seg
    return s, lat, nearest


def build_floor_texture(track: Track, seed: int = 0, tpm: int = TEXELS_PER_METRE) -> FloorTexture:
    xmin, ymin, xmax, ymax = track.bounds
    origin = (xmin - MARGIN, ymax + MARGIN)
    cols = math.ceil((xmax - xmin + 2 * MARGIN) * tpm)
    rows = math.ceil((ymax - ymin + 2 * MARGIN) * tpm)
    hw = track.half_width

    # Exact track coordinates only in a band around the asphalt.
    coarse_shape = (math.ceil(rows / tpm), math.ceil(cols / tpm))
    skip = None if track.levels is None else track.levels.raised
    dist, seg = _coarse_field(track, origin, coarse_shape, hw + BAND + 1.5, skip)
    near_cells = dist < hw + BAND + 1.0

    # Painted in strips of rows: each texel depends only on itself, so the result equals one
    # pass over the whole image, with a small fraction of the temporary memory.
    tex = np.empty((rows, cols), dtype=np.uint8)
    for v0 in range(0, rows, STRIP_ROWS):
        _paint_strip(track, tex[v0 : v0 + STRIP_ROWS], v0, origin, seed, tpm, near_cells, seg)
    texture = FloorTexture([tex], origin, float(tpm))
    texture.build_mips()
    return texture


def _paint_strip(
    track: Track,
    tex: np.ndarray,
    v0: int,
    origin: tuple[float, float],
    seed: int,
    tpm: int,
    near_cells: np.ndarray,
    seg: np.ndarray,
) -> None:
    """Paint texel rows ``v0`` onwards into ``tex``, a view of the rows of the texture."""
    hw = track.half_width
    v_idx, u_idx = np.mgrid[v0 : v0 + len(tex), 0 : tex.shape[1]]

    # Grass everywhere: mowing stripes in world space plus a sprinkle of lighter blades.
    wx = origin[0] + (u_idx + 0.5) / tpm
    wy = origin[1] - (v_idx + 0.5) / tpm
    stripe = np.floor((wx * math.cos(STRIPE_ANGLE) + wy * math.sin(STRIPE_ANGLE)) / STRIPE_METRES)
    light = (stripe.astype(np.int64) & 1).astype(bool)
    noise = hash_noise(u_idx, v_idx, seed)
    tex[...] = np.where(light, Ink.GRASS_LIGHT, Ink.GRASS).astype(np.uint8)
    tex[light & (noise < 0.05)] = Ink.LEAF_LIGHT
    tex[~light & (noise < 0.05)] = Ink.GRASS_DARK

    near = near_cells[v_idx // tpm, u_idx // tpm]
    vs, us = np.nonzero(near)
    if len(vs) == 0:
        _shade_bridge(track, tex, v0, origin, tpm)
        return
    x, y = wx[vs, us], wy[vs, us]
    s, lat, nearest = _project(track, x, y, seg[(vs + v0) // tpm, us // tpm])
    if track.crossings:
        s, lat, nearest = _refine_crossings(track, x, y, s, lat, nearest)
    alat = np.abs(lat)
    n_band = noise[vs, us]
    dither = BAYER4[(vs + v0) % 4, us % 4]
    out = tex[vs, us]

    # Grass shadow along the outside of the edge: darker, fading out with dithering.
    shade = (alat > hw) & (alat < hw + 1.5)
    fade = (alat - hw) / 1.5
    out[shade & (dither > fade)] = Ink.GRASS_DARK

    # Kerbs on the corners (curvature widened a few metres before and after).
    bend = np.abs(track.curvature)
    bend = np.maximum.reduce([np.roll(bend, k) for k in range(-6, 7)])
    kerb = (bend[nearest] > KERB_CURVATURE) & (alat > hw) & (alat <= hw + KERB_WIDTH)
    red = (np.floor(s / KERB_STRIPE).astype(np.int64) & 1).astype(bool)
    outer = alat > hw + KERB_WIDTH * 0.7  # bevel: the outer edge is painted darker
    out[kerb & red] = Ink.KERB_RED
    out[kerb & red & outer] = Ink.RED
    out[kerb & ~red] = Ink.WHITE
    out[kerb & ~red & outer] = Ink.MIST

    # Asphalt with grain, darker towards the edges.
    asphalt = alat <= hw
    out[asphalt] = Ink.ASPHALT
    edge_ao = np.clip((alat - (hw - 1.2)) / 1.2, 0.0, 1.0)
    out[asphalt & (n_band < 0.035)] = Ink.ASPHALT_LIGHT
    out[asphalt & (n_band > 0.985 - 0.25 * edge_ao)] = Ink.ASPHALT_DARK
    out[asphalt & (dither < edge_ao * 0.45)] = Ink.ASPHALT_DARK
    line = asphalt & (alat >= hw - EDGE_LINE[1]) & (alat <= hw - EDGE_LINE[0])
    out[line] = Ink.WHITE

    # Chequered start/finish line.
    signed_s = np.where(s > track.length / 2, s - track.length, s)
    start = asphalt & (np.abs(signed_s) <= START_LINE)
    rows_k = np.floor((signed_s + START_LINE) / (2 * START_LINE / 3)).astype(np.int64)
    cols_k = np.floor((lat + hw) / (track.width / round(track.width))).astype(np.int64)
    checker = ((rows_k + cols_k) & 1).astype(bool)
    out[start & checker] = Ink.BLACK
    out[start & ~checker] = Ink.WHITE

    tex[vs, us] = out
    _shade_bridge(track, tex, v0, origin, tpm)


def _shade_bridge(
    track: Track, tex: np.ndarray, v0: int, origin: tuple[float, float], tpm: int
) -> None:
    """Shadows of a bridge, baked like those of the scenery.

    The floor of the tunnel (the lower road under the deck) is dark, in a dither of
    ``ASPHALT_DARK`` and ``ASPHALT`` that grows denser towards the inside; beside the
    embankment, on the side the sun does not reach, a band of shadow falls on the grass and
    on the asphalt. ``ASPHALT_DARK`` is the same index as ``BLACK``, so it is never mixed
    with black.
    """
    levels = track.levels
    if levels is None:
        return
    cx, cy, heading = track.pose_at(levels.s_over)
    ux, uy = math.cos(heading), math.sin(heading)  # axis of the upper road; (-uy, ux) is left
    lx, ly, lower = track.pose_at(levels.s_under)
    lcos, lsin = math.cos(lower), math.sin(lower)
    reach = levels.reach + levels.embankment_half + 4.0
    # Only the texels around the bridge (a box of the strip), nothing else.
    u0 = max(int((cx - reach - origin[0]) * tpm), 0)
    u1 = min(int((cx + reach - origin[0]) * tpm) + 1, tex.shape[1])
    r0 = max(int((origin[1] - cy - reach) * tpm) - v0, 0)
    r1 = min(int((origin[1] - cy + reach) * tpm) + 1 - v0, len(tex))
    if u0 >= u1 or r0 >= r1:
        return
    block = tex[r0:r1, u0:u1]
    vv, uu = np.mgrid[v0 + r0 : v0 + r1, u0:u1]
    x = origin[0] + (uu + 0.5) / tpm
    y = origin[1] - (vv + 0.5) / tpm
    dx, dy = x - cx, y - cy
    along = dx * ux + dy * uy
    across = -dx * uy + dy * ux  # to the left of the upper road
    dither = BAYER4[vv % 4, uu % 4]
    wide = levels.embankment_half

    # Tunnel floor: inside the embankment, on the lower road.
    lat_low = -(x - lx) * lsin + (y - ly) * lcos
    tunnel = (np.abs(across) <= wide) & (np.abs(along) <= levels.reach)
    tunnel &= np.abs(lat_low) <= track.half_width
    depth = np.clip((wide - np.abs(across)) / 3.0, 0.0, 1.0)  # 0 at the mouth, 1 inside
    dark = dither < 0.45 + 0.4 * depth
    block[tunnel & dark] = Ink.ASPHALT_DARK
    block[tunnel & ~dark] = Ink.ASPHALT

    # Shadow of the embankment on the side away from the sun.
    sun = np.array(SUN_DIRECTION) / math.hypot(*SUN_DIRECTION)
    h = levels.deck_height * (
        1.0 - _smoothstep((np.abs(along) - levels.deck_half) / levels.ramp)
    )  # height of the bridge beside each texel
    for side in (-1.0, 1.0):
        outward = side * (-uy * sun[0] + ux * sun[1])  # sun component along the face normal
        if outward <= 0.0:
            continue  # the sun lights this face
        band = SHADOW_REACH * (h + 0.9) * outward
        gap = side * across - wide  # distance beyond the face
        shade = (gap >= 0.0) & (gap < band) & (np.abs(along) <= levels.reach)
        fade = np.where(band > 0, gap / np.maximum(band, 1e-6), 1.0)
        shade &= dither > fade * 0.9
        grass = (block == Ink.GRASS) | (block == Ink.GRASS_LIGHT) | (block == Ink.LEAF_LIGHT)
        asphalt = (block == Ink.ASPHALT) | (block == Ink.ASPHALT_LIGHT)
        block[shade & grass] = Ink.GRASS_DARK
        block[shade & asphalt] = Ink.ASPHALT_DARK


def _smoothstep(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)
