"""The bridge and the tunnel of a track with levels, drawn as flat polygons.

Everything is precomputed at load time as arrays of world points: a ring of eight points
(plus the points of the deck markings) at every ``SECTION`` metres of the raised stretch,
the portals of the tunnel and the inside of the tunnel. A frame projects all of them with
one call, culls what faces away or is off screen with array operations, and draws the rest
with ``pygame.draw.polygon``: no Python loop per pixel, nothing allocated per pixel.

The bridge is drawn section by section, from far to near (not polygon by polygon), as
drawables mixed with the sprites of ``WorldRenderer``, so a kart on the deck, one in the
tunnel and one on the grass all end up in front of or behind the right sections. Inside a
section the strips are drawn in a fixed order and only the ones that face the camera.

Cross section of the raised road, left to right (lateral offset, height)::

        cap       cap
       ┌──┐       ┌──┐      parapets, ``PARAPET`` high on the deck, 1.3 m thick
    ┌──┘  └───────┘  └──┐   deck: asphalt between the inner faces of the parapets
    │                    │  smooth sides of the ramps and of the embankment, down to the ground
    columns 0 1  2     3 4 5  6 7   (see ``COLUMNS``)

The walls of the simulation (``sim/levels.py``) stand where the parapets and the tunnel walls
are drawn: ``WALL_INSET`` inside the asphalt edge.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pygame

from kartlab.render.camera import CameraPose, Lens
from kartlab.render.floor import Projection
from kartlab.render.palette import Ink
from kartlab.sim.levels import WALL_INSET
from kartlab.sim.track import Track

NEAR = 0.8  # nothing is drawn closer than this to the camera (m)
SECTION = 4.0  # length of a section of the raised road (m)
PARAPET = 0.9  # height of the parapets over the deck (m)
PARAPET_FULL = 0.8  # the parapet is full height from this deck height up (m)
MARK_OUT = 0.45  # edge lines: this far inside the inner face of the parapet (m)
MARK_WIDTH = 0.25
DASH = (1.0, 3.0)  # centre dash of every other section, from-to along it (m)
DASH_HALF_WIDTH = 0.12
LINTEL = 0.5  # height of the lintel over a tunnel mouth (m)
JAMB = 0.35  # how far the lintel and the jambs stand out beside the opening (m)
FIT_MARGIN = 200.0  # polygons reaching farther than this off screen are cut to it (px)
LOD_DEPTH = 32.0  # farther than this, pairs of sections are drawn as one of 8 m (m)
LINES_UNTIL = 55.0  # the painted lines of the deck are drawn up to this depth (m)
LAMP_EVERY = 3  # a lamp strip on the roof of the tunnel every this many segments

SLAB = 0.5  # thickness of the deck, seen as a cornice that sticks out of the face (m)
OVERHANG = 0.7  # how far the cornice sticks out (m)
COURSES = (1.0, 2.0, 3.0)  # heights of the joints between courses of stone (m)

# Ink of each part. Faces take the lit or the shaded one according to the sun.
DECK = (Ink.ASPHALT, Ink.ASPHALT)  # the two alternate every section


class Inks(NamedTuple):
    """The ink of each part of the bridge, per light. The palette is remapped at sunset and at
    night (an ink goes to the nearest colour of its tinted version), and the sandstone of the
    day falls on the asphalt's colour at night: the night set uses the stones that stay lighter."""

    face: tuple[int, int]  # two blocks that alternate every section (lit side)
    face_shade: tuple[int, int]
    joint: int
    joint_shade: int
    under: int  # the underside of the cornice, in shade
    front: int
    front_shade: int
    ledge: int
    up: int  # the outer face of the parapet
    up_shade: int
    cap: int
    inner: int
    inner_shade: int
    line: int  # painted lines of the deck


# Blocks of sandstone, the road is dark and purple (and white where it is painted).
DAY_INKS = Inks((4, 3), (3, 5), 5, Ink.ASPHALT_DARK, Ink.ASPHALT, 4, 3, 4, 4, 3, 4, 3, 5, Ink.MIST)
NIGHT_INKS = Inks(
    (7, 4), (4, 2), Ink.ASPHALT_DARK, Ink.ASPHALT_DARK, Ink.ASPHALT_DARK, 7, 4, 9, 7, 4, 9, 7, 2,
    Ink.WHITE,
)  # fmt: skip
INKS = {"day": DAY_INKS, "sunset": DAY_INKS, "night": NIGHT_INKS}
LINTEL_INK = Ink.STONE_LIGHT
WALL_INKS = (Ink.ASPHALT_DARK, Ink.ASPHALT)
ROOF_INK, LAMP_INK = Ink.ASPHALT_DARK, Ink.ASPHALT_LIGHT

# Columns of a ring (lateral offset, height) and the strips between them.
# 0-1 outer face of the embankment, 1-2 outer face of the parapet, 2-3 cap, 3-4 inner face,
# 4-5 deck, 5-6 inner face, 6-7 cap, 7-8 outer face of the parapet, 8-9 outer face of the
# embankment, then the two edge lines (10-11, 12-13).
STRIPS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (10, 11),
    (12, 13),
    # The cornice of the deck, on each side: its underside, its front and its top.
    (14, 15),
    (15, 16),
    (16, 1),
    (17, 18),
    (18, 19),
    (19, 8),
)
(
    LOW_L,
    UP_L,
    CAP_L,
    INNER_L,
    DECK_STRIP,
    INNER_R,
    CAP_R,
    UP_R,
    LOW_R,
    LINE_L,
    LINE_R,
    UNDER_L,
    FRONT_L,
    LEDGE_L,
    UNDER_R,
    FRONT_R,
    LEDGE_R,
) = range(len(STRIPS))
# Columns 14-19 are the cornice (the point on the face under it, the underside edge, the front
# top) and 20-25 the points of the joints between courses of stone, left then right.
# Each section is drawn in two parts. The far part (the side of the road farthest from the camera,
# then the deck and its lines) goes first and the near part (the nearer parapet and its outer
# faces) last, with the karts that are on the deck in between: they are in front of the far
# parapet and behind the near one.
FAR_LEFT = (LOW_L, UP_L, LEDGE_L, FRONT_L, UNDER_L, CAP_L, INNER_L, DECK_STRIP, LINE_L, LINE_R)
FAR_RIGHT = (LOW_R, UP_R, LEDGE_R, FRONT_R, UNDER_R, CAP_R, INNER_R, DECK_STRIP, LINE_L, LINE_R)
NEAR_LEFT = (INNER_L, CAP_L, UP_L, LOW_L, LEDGE_L, UNDER_L, FRONT_L)
NEAR_RIGHT = (INNER_R, CAP_R, UP_R, LOW_R, LEDGE_R, UNDER_R, FRONT_R)
LEFT_STRIPS = (LOW_L, UP_L, CAP_L, INNER_L, UNDER_L, FRONT_L, LEDGE_L)
RIGHT_STRIPS = (LOW_R, UP_R, CAP_R, INNER_R, UNDER_R, FRONT_R, LEDGE_R)

Rgb = Callable[[int, int], tuple[int, int, int]]


def _lerp(p: tuple[float, float], q: tuple[float, float], t: float) -> tuple[float, float]:
    return p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t


def _fit(points: list[tuple[float, float]], width: int, height: int) -> list[tuple[float, float]]:
    """A polygon cut to the screen when it reaches far outside it.

    ``pygame.draw.polygon`` walks every row of the polygon, on screen or not, so a strip that
    is almost edge-on (thousands of pixels tall) costs a millisecond. Most polygons are already
    inside the margin and come back untouched.
    """
    margin = FIT_MARGIN
    if all(-margin <= x <= width + margin and -margin <= y <= height + margin for x, y in points):
        return points
    for axis, bound, keep_above in (
        (0, -1.0, True),
        (0, width + 1.0, False),
        (1, -1.0, True),
        (1, height + 1.0, False),
    ):
        out = []
        count = len(points)
        for i in range(count):
            a, b = points[i], points[(i + 1) % count]
            a_in = (a[axis] >= bound) if keep_above else (a[axis] <= bound)
            b_in = (b[axis] >= bound) if keep_above else (b[axis] <= bound)
            if a_in:
                out.append(a)
            if a_in != b_in:
                t = (bound - a[axis]) / (b[axis] - a[axis])
                out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        points = out
        if len(points) < 3:
            return []
    return points


def _clip(points: list[tuple[float, float, float]], depths: list[float]):
    """Part of a polygon in front of the near plane (Sutherland-Hodgman), in world points."""
    out = []
    count = len(points)
    for i in range(count):
        a, b = points[i], points[(i + 1) % count]
        da, db = depths[i], depths[(i + 1) % count]
        if da > NEAR:
            out.append(a)
        if (da > NEAR) != (db > NEAR):
            t = (NEAR - da) / (db - da)
            out.append(
                (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)
            )
    return out


class BridgeRenderer:
    """Geometry of one bridge and its tunnel, and the frame-by-frame drawing."""

    def __init__(self, track: Track, surface: pygame.Surface, rgb: Rgb, sun: tuple[float, float]):
        levels = track.levels
        assert levels is not None
        self.track, self.levels = track, levels
        self.surface = surface
        self.rgb = rgb
        hw = track.half_width
        self.wide = levels.embankment_half  # outer face, from the upper centerline
        self.inner = hw - WALL_INSET  # inner face of the parapets and walls
        self.sun = np.asarray(sun, dtype=np.float64) / math.hypot(*sun)
        self.inks = DAY_INKS
        self._build_sections()
        self._build_tunnel()
        self.visible_sections: list[int] = []
        self._frame: dict | None = None
        self.camera_in_tunnel = False
        self._size = (320, 180)
        self._piece_frame = ([], [], [])
        self.exit_depth = 0.0  # depth of the far mouth while the camera is inside

    # -- geometry ------------------------------------------------------------------------

    def _build_sections(self) -> None:
        track, levels = self.track, self.levels
        half = levels.reach + 1.0
        count = math.ceil(2.0 * half / SECTION)
        step = 2.0 * half / count
        s = levels.s_over - half + step * np.arange(count + 1)
        poses = np.array([track.pose_at(float(v)) for v in s])
        heights = np.array([levels.height_at(float(v)) for v in s])
        parapet = PARAPET * np.clip(heights / PARAPET_FULL, 0.0, 1.0)
        top = heights + parapet
        eps = 0.01
        w, wide = self.inner, self.wide
        mark_out, mark_in = w - MARK_OUT, w - MARK_OUT - MARK_WIDTH
        lat = np.array(
            [wide, wide, wide, w, w, -w, -w, -wide, -wide, -wide]
            + [mark_out, mark_in, -mark_in, -mark_out]
        )
        ground = np.zeros_like(heights)
        # The cornice: the deck is a slab SLAB thick whose edge sticks out of the face, but only
        # where the deck is high enough to clear the mouth of the tunnel.
        floor = levels.ceiling + LINTEL + 0.05
        under = np.minimum(heights, np.maximum(heights - SLAB, floor))
        self.slab = heights - under
        lat = np.concatenate(
            [
                lat,
                [wide, wide + OVERHANG, wide + OVERHANG, -wide, -wide - OVERHANG, -wide - OVERHANG],
                [wide] * len(COURSES),
                [-wide] * len(COURSES),
            ]
        )
        joints = [np.minimum(level, heights) for level in COURSES]
        z = np.stack(
            [ground, heights, top, top, heights, heights, top, top, heights, ground]
            + [heights + eps] * 4
            + [under, under, heights, under, under, heights]
            + joints
            + joints,
            axis=1,
        )
        c, sn = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        self.centre = poses[:, :2]
        self.heading = poses[:, 2]
        self.ring_x = poses[:, 0, None] - sn[:, None] * lat[None, :]
        self.ring_y = poses[:, 1, None] + c[:, None] * lat[None, :]
        self.ring_z = z
        self.count = count
        self.s = s
        self.height = heights
        self.top = top
        self.lat = lat
        # Dashes on the deck: the middle of every other section.
        dash = []
        for k in range(0, count, 2):
            a, b = s[k] + DASH[0], s[k] + DASH[1]
            quad = []
            for along, side in ((a, 1.0), (b, 1.0), (b, -1.0), (a, -1.0)):
                x, y, _ = track.pose_at(float(along), side * DASH_HALF_WIDTH)
                quad.append((x, y, levels.height_at(float(along)) + eps))
            dash.append(quad)
        self.dash = np.array(dash).reshape(-1, 4, 3)
        self.dash_section = np.arange(0, count, 2)
        # Which ring columns every strip of every section uses.
        ks = np.arange(count)[:, None, None]
        a = np.array([p[0] for p in STRIPS])[None, :, None]
        b = np.array([p[1] for p in STRIPS])[None, :, None]
        self.corner_k = np.concatenate(
            [ks + 0 * a, ks + 0 * a, ks + 1 + 0 * a, ks + 1 + 0 * a], axis=2
        )
        self.corner_j = np.broadcast_to(
            np.concatenate([a, b, b, a], axis=2), (count, len(STRIPS), 4)
        ).copy()
        # Strips with no height (the foot of a ramp) have nothing to draw.
        p0, p1 = parapet[:-1], parapet[1:]
        ok = np.ones((count, len(STRIPS)), dtype=bool)
        g0, g1 = heights[:-1], heights[1:]
        ok[:, [LOW_L, LOW_R]] = ((g0 > 0.02) | (g1 > 0.02))[:, None]
        ok[:, [UP_L, UP_R, INNER_L, INNER_R]] = ((p0 > 0.02) | (p1 > 0.02))[:, None]
        s0, s1 = self.slab[:-1], self.slab[1:]
        cornice = [UNDER_L, FRONT_L, LEDGE_L, UNDER_R, FRONT_R, LEDGE_R]
        ok[:, cornice] = ((s0 > 0.05) & (s1 > 0.05))[:, None]
        self.strip_ok = ok
        # Lit or shaded sides: the sun shines from the opposite way to the shadows.
        normal = np.stack([-np.sin(self.heading), np.cos(self.heading)], axis=1)  # left of the road
        lit_left = (normal @ -self.sun) > 0.0
        self.lit_left = lit_left[:-1] | lit_left[1:]
        self._pair_first = np.arange(0, count - 1, 2)  # first section of every pair
        self._col_a = np.array([p[0] for p in STRIPS])
        self._col_b = np.array([p[1] for p in STRIPS])
        self._in_far_left = np.ones((count, len(STRIPS)), dtype=bool)
        self._in_far_left[:, list(RIGHT_STRIPS)] = False  # the far side is the left one
        self._in_far_right = np.ones((count, len(STRIPS)), dtype=bool)
        self._in_far_right[:, list(LEFT_STRIPS)] = False
        # Plain Python copies for the polygons that need clipping, and every strip's ink.
        self._ring_x = self.ring_x.tolist()
        self._ring_y = self.ring_y.tolist()
        self._ring_z = self.ring_z.tolist()
        self._corners = [
            [
                [(int(self.corner_k[k, s, i]), int(self.corner_j[k, s, i])) for i in range(4)]
                for s in range(len(STRIPS))
            ]
            for k in range(count)
        ]
        self._corners_pair = [
            [
                [(k + 2 * (i // 2), j) for i, (_, j) in enumerate(self._corners[k][s])]
                for s in range(len(STRIPS))
            ]
            for k in range(count)
        ]
        self._inks = [
            [self._ink(k, s, bool(self.lit_left[k])) for s in range(len(STRIPS))]
            for k in range(count)
        ]

    def set_variant(self, variant: str) -> None:
        """Pick the inks of the light (day, sunset or night)."""
        self.inks = INKS[variant]
        self._inks = [
            [self._ink(k, s, bool(self.lit_left[k])) for s in range(len(STRIPS))]
            for k in range(self.count)
        ]

    def _build_tunnel(self) -> None:
        track, levels = self.track, self.levels
        crossing = levels.crossing
        ox, oy = crossing.x, crossing.y
        _, _, upper = track.pose_at(levels.s_over)
        _, _, lower = track.pose_at(levels.s_under)
        ux, uy = math.cos(upper), math.sin(upper)
        nx, ny = -uy, ux  # left of the upper road
        lx, ly = math.cos(lower), math.sin(lower)
        mx, my = -ly, lx  # left of the lower road
        self.origin = (ox, oy)
        self.upper_normal = (nx, ny)
        ln, mn = lx * nx + ly * ny, mx * nx + my * ny
        w, wide, roof = self.inner, self.wide, levels.ceiling
        self.roof = roof
        planes = {}
        for tau in (-1.0, 1.0):
            row = {}
            for sigma in (1.0, -1.0):  # left and right wall of the tunnel
                t = (tau * wide - sigma * w * mn) / ln
                row[sigma] = (ox + t * lx + sigma * w * mx, oy + t * ly + sigma * w * my, t)
            planes[tau] = row
        entry = -1.0 if ln > 0.0 else 1.0
        self.portals = []  # (tau, left point, right point), entry first
        for tau in (entry, -entry):
            self.portals.append((tau, planes[tau][1.0][:2], planes[tau][-1.0][:2]))
        # Inside: walls and roof, cut into segments so they have some rhythm.
        (ax, ay, at), (bx, by, bt) = planes[entry][1.0], planes[-entry][1.0]
        (cx, cy, _), (dx, dy, _) = planes[entry][-1.0], planes[-entry][-1.0]
        length = abs(bt - at)
        segments = max(int(round(length / 3.0)), 1)
        quads, inks = [], []
        for k in range(segments):
            t0, t1 = k / segments, (k + 1) / segments
            left = (_lerp((ax, ay), (bx, by), t0), _lerp((ax, ay), (bx, by), t1))
            right = (_lerp((cx, cy), (dx, dy), t0), _lerp((cx, cy), (dx, dy), t1))
            mouth = k in (0, segments - 1)  # lighter where the daylight comes in
            for side in (left, right):
                (p0x, p0y), (p1x, p1y) = side
                quads.append([(p0x, p0y, 0.0), (p1x, p1y, 0.0), (p1x, p1y, roof), (p0x, p0y, roof)])
                inks.append(Ink.ASPHALT if mouth else WALL_INKS[k % 2])
            (l0x, l0y), (l1x, l1y) = left
            (r0x, r0y), (r1x, r1y) = right
            quads.append([(l0x, l0y, roof), (l1x, l1y, roof), (r1x, r1y, roof), (r0x, r0y, roof)])
            inks.append(LAMP_INK if k % LAMP_EVERY == 1 else Ink.ASPHALT if mouth else ROOF_INK)
        self.tunnel_quads = np.array(quads)
        self.tunnel_inks = inks
        self.entry_planes = (entry, planes)
        self.tunnel_ends = (at, bt)
        # Sections each mouth of the tunnel lies against (for the order of drawing).
        self.portal_sections = []
        for _, left, right in self.portals:
            along = [((px - ox) * ux + (py - oy) * uy) for px, py in (left, right)]
            lo, hi = min(along) - JAMB, max(along) + JAMB
            touched = [
                k
                for k in range(self.count)
                if self._section_along(k, ux, uy, ox, oy)[1] >= lo
                and self._section_along(k, ux, uy, ox, oy)[0] <= hi
            ]
            self.portal_sections.append(touched)
        self._build_holes(ux, uy, ox, oy)
        self._finish_holes()

    def _finish_holes(self) -> None:
        self.hole_world = np.array(self._hole_pieces).reshape(-1, 4, 3)

    def _build_holes(self, ux: float, uy: float, ox: float, oy: float) -> None:
        """The face of the embankment with the mouth of the tunnel cut out of it.

        For every section that the opening touches, the strip of face is replaced by up to
        three polygons: what is left of the opening, what is right of it and what is above it.
        What is behind the hole (the inside of the tunnel, the world through its far mouth)
        then shows, instead of a flat dark block that hides it (and the kart) when the camera
        is close.
        """
        roof = self.roof
        self.holes: dict[tuple[int, int], list[list[tuple[float, float, float]]]] = {}
        self.hole_span: dict[tuple[int, int], tuple[float, float]] = {}  # opening, as fractions
        self._hole_pieces: list[list[tuple[float, float, float]]] = []
        for tau, a, b in self.portals:
            strip, col = (LOW_L, 0) if tau > 0 else (LOW_R, 9)
            lo = min((p[0] - ox) * ux + (p[1] - oy) * uy for p in (a, b))
            hi = max((p[0] - ox) * ux + (p[1] - oy) * uy for p in (a, b))
            for k in range(self.count):
                x0, y0, x1, y1 = (
                    self.ring_x[k, col], self.ring_y[k, col],
                    self.ring_x[k + 1, col], self.ring_y[k + 1, col],
                )  # fmt: skip
                h0, h1 = self.height[k], self.height[k + 1]
                a0 = (x0 - ox) * ux + (y0 - oy) * uy
                a1 = (x1 - ox) * ux + (y1 - oy) * uy
                if abs(a1 - a0) < 1e-9 or max(a0, a1) <= lo or min(a0, a1) >= hi:
                    continue

                def at(f, z=None, x0=x0, y0=y0, x1=x1, y1=y1, h0=h0, h1=h1):
                    height = h0 + (h1 - h0) * f
                    return (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, height if z is None else z)

                f_lo, f_hi = sorted(((lo - a0) / (a1 - a0), (hi - a0) / (a1 - a0)))
                pieces = []
                if f_lo > 0.0:  # beside the opening, full height
                    pieces.append([at(0.0, 0.0), at(f_lo, 0.0), at(f_lo), at(0.0)])
                if f_hi < 1.0:
                    pieces.append([at(f_hi, 0.0), at(1.0, 0.0), at(1.0), at(f_hi)])
                f0, f1 = max(f_lo, 0.0), min(f_hi, 1.0)
                top = [max(h0 + (h1 - h0) * f, roof) for f in (f0, f1)]
                if max(top) > roof + 1e-6:  # above the opening
                    pieces.append([at(f0, roof), at(f1, roof), at(f1, top[1]), at(f0, top[0])])
                self.holes[(k, strip)] = list(
                    range(len(self._hole_pieces), len(self._hole_pieces) + len(pieces))
                )
                self._hole_pieces.extend(pieces)
                self.hole_span[(k, strip)] = (f_lo, f_hi)

    def _section_along(self, k: int, ux: float, uy: float, ox: float, oy: float):
        """Range of the section's centre along the upper road's axis at the crossing."""
        pa = (self.centre[k, 0] - ox) * ux + (self.centre[k, 1] - oy) * uy
        pb = (self.centre[k + 1, 0] - ox) * ux + (self.centre[k + 1, 1] - oy) * uy
        return min(pa, pb), max(pa, pb)

    # -- per frame -----------------------------------------------------------------------

    def kart_bias(self, pose: CameraPose) -> float:
        """How much nearer (m) a kart on the bridge is sorted, so it goes after its own section.

        Looking along the road a section is 4 m long in depth; looking across it, the kart is
        already sorted between the far and the near part. So it follows the view direction."""
        return SECTION / 2.0 * abs(math.cos(pose.heading - float(self.heading[self.count // 2])))

    def lateral_of_camera(self, pose: CameraPose) -> np.ndarray:
        dx, dy = pose.x - self.centre[:, 0], pose.y - self.centre[:, 1]
        return -dx * np.sin(self.heading) + dy * np.cos(self.heading)

    def prepare(
        self, pose: CameraPose, projection: Projection, lens: Lens
    ) -> list[tuple[float, int, tuple]]:
        """Project the geometry for this frame; the drawables to mix with the sprites."""
        eye = lens.height + pose.z
        ox, oy = self.origin
        nx, ny = self.upper_normal
        lat_u = (pose.x - ox) * nx + (pose.y - oy) * ny
        inside = self._inside_tunnel(pose, eye)
        self.camera_in_tunnel = inside
        self._lens = lens
        self._pose = pose
        self._projection = projection
        self._setup_camera(pose, projection, lens)
        self._size = (projection.width, projection.height)
        drawables: list[tuple[float, int, tuple]] = []
        if inside:
            # The inside is drawn after what is beyond the far mouth (seen through it) and
            # before the karts in the tunnel; whatever stands beside the tunnel is hidden.
            middles = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for _, a, b in self.portals]
            self.exit_depth = max(
                float(projection.project(pose, mx, my, self.roof / 2)[2]) for mx, my in middles
            )
            drawables.append((self.exit_depth, 0, (self._draw_interior,)))
            return drawables

        points = self.hole_world
        psx, psy, pdepth = projection.project(pose, points[..., 0], points[..., 1], points[..., 2])
        self._piece_frame = (psx.tolist(), psy.tolist(), (pdepth > NEAR).all(axis=1).tolist())
        if any(tau * lat_u > self.wide for tau, _, _ in self.portals):
            # Facing a mouth: the inside shows through the hole in the face. It is drawn after
            # what is beyond the far mouth and before the face and the karts in the tunnel.
            middles = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for _, a, b in self.portals]
            self.exit_depth = max(
                float(projection.project(pose, mx, my, self.roof / 2)[2]) for mx, my in middles
            )
            drawables.append((self.exit_depth, 0, (self._draw_interior,)))
        sx, sy, depth = projection.project(pose, self.ring_x, self.ring_y, self.ring_z)
        x0, x1 = self._span(sx)
        y0, y1 = self._span(sy)
        d0, d1 = self._span(depth)
        ahead = d0 > NEAR  # every corner of the strip is in front of the camera
        partial = (d1 > NEAR) & ~ahead  # some are not: the strip is cut at the near plane
        width, height = projection.width, projection.height
        seen = (x1 >= 0) & (x0 <= width) & (y1 >= 0) & (y0 <= height)
        # Less than a pixel across: nothing to draw. Painted lines only where they can be seen.
        seen &= ((x1 - x0) >= 1.0) | ((y1 - y0) >= 1.0)
        seen[:, [LINE_L, LINE_R]] &= d0[:, [LINE_L, LINE_R]] < LINES_UNTIL
        show = self.strip_ok & ((ahead & seen) | partial) & self._facing(pose, eye)
        # How far each side of each section is: the farther goes first.
        left_depth = depth[:, 0:3].mean(axis=1)
        right_depth = depth[:, 7:10].mean(axis=1)
        left_depth = (left_depth[:-1] + left_depth[1:]) / 2.0
        right_depth = (right_depth[:-1] + right_depth[1:]) / 2.0
        far_is_left = left_depth > right_depth
        far_key = np.maximum(left_depth, right_depth)
        near_key = np.minimum(left_depth, right_depth)
        in_far = self._in_far_left.copy()
        in_far[~far_is_left] = self._in_far_right[~far_is_left]
        # Far away, two neighbouring sections are one polygon per strip: half the polygons for
        # a difference of less than a pixel (the ramp is a smooth curve; 8 m of it is flat enough).
        pair = np.zeros(self.count, dtype=bool)
        first, second = self._pair_first, self._pair_first + 1
        pair[first] = (near_key[first] > LOD_DEPTH) & (near_key[second] > LOD_DEPTH)
        pair[first] &= ~(partial[first].any(axis=1) | partial[second].any(axis=1))
        head = first[pair[first]]
        show[head] |= show[head + 1]
        show[head + 1] = False
        far_key[head] = (far_key[head] + far_key[head + 1]) / 2.0
        near_key[head] = (near_key[head] + near_key[head + 1]) / 2.0
        self._frame = {
            "sx": sx.tolist(),
            "sy": sy.tolist(),
            "partial": partial.tolist(),
            "show": show.tolist(),
            "far_is_left": far_is_left.tolist(),
            "pair": pair.tolist(),
        }
        for index in np.flatnonzero((show & in_far).any(axis=1)).tolist():
            drawables.append((float(far_key[index]), 0, (self._draw_section, index, True)))
        for index in np.flatnonzero((show & ~in_far).any(axis=1)).tolist():
            drawables.append((float(near_key[index]), 1, (self._draw_section, index, False)))
        # Dashes and portals go with their sections.
        section_depth = near_key
        self._section_depth = section_depth
        self._dash_frame = self._project_dashes(pose, projection)
        for number, (tau, _, _) in enumerate(self.portals):
            if tau * lat_u > self.wide:  # outside, facing the mouth
                touched = self.portal_sections[number]
                key = min(float(section_depth[i]) for i in touched) - 0.05 if touched else 0.0
                drawables.append((key, 0, (self._draw_portal, number)))
        return drawables

    def _inside_tunnel(self, pose: CameraPose, eye: float) -> bool:
        if eye >= self.roof:
            return False
        return self.levels.in_tunnel_zone(pose.x, pose.y, 0.0) and self._between_mouths(pose)

    def _between_mouths(self, pose: CameraPose) -> bool:
        ox, oy = self.origin
        nx, ny = self.upper_normal
        return abs((pose.x - ox) * nx + (pose.y - oy) * ny) <= self.wide

    def _span(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Smallest and largest of the four corners of every strip of every section."""
        first, second = values[:, self._col_a], values[:, self._col_b]
        low, high = np.minimum(first, second), np.maximum(first, second)
        return np.minimum(low[:-1], low[1:]), np.maximum(high[:-1], high[1:])

    def _facing(self, pose: CameraPose, eye: float) -> np.ndarray:
        """Which strips of which sections face the camera (a boolean (sections, strips))."""
        lat = self.lateral_of_camera(pose)  # per ring
        out = np.zeros((self.count, len(STRIPS)), dtype=bool)

        def both(values: np.ndarray) -> np.ndarray:
            return values[:-1] | values[1:]

        out[:, [LOW_L, UP_L]] = both(lat > self.wide)[:, None]
        out[:, [LOW_R, UP_R]] = both(lat < -self.wide)[:, None]
        out[:, FRONT_L] = both(lat > self.wide + OVERHANG)
        out[:, FRONT_R] = both(lat < -self.wide - OVERHANG)
        under = self.height - self.slab
        out[:, UNDER_L] = both(lat > self.wide) & both(eye < under)
        out[:, UNDER_R] = both(lat < -self.wide) & both(eye < under)
        out[:, INNER_L] = both(lat < self.inner)
        out[:, INNER_R] = both(lat > -self.inner)
        up = {
            CAP_L: self.top,
            CAP_R: self.top,
            DECK_STRIP: self.height,
            LINE_L: self.height,
            LINE_R: self.height,
            LEDGE_L: self.height,
            LEDGE_R: self.height,
        }
        for strip, z in up.items():
            out[:, strip] = both(eye > z)
        return out

    def _project_dashes(self, pose: CameraPose, projection: Projection):
        points = self.dash
        sx, sy, depth = projection.project(pose, points[..., 0], points[..., 1], points[..., 2])
        ok = (depth > NEAR).all(axis=1)
        return sx.tolist(), sy.tolist(), ok

    # -- drawing -------------------------------------------------------------------------

    def _fog(self, depth: float) -> int:
        """Fog level of a polygon. Like the scenery's sprites, the bridge is not fogged: a flat
        polygon pushed to a fog level lands on odd palette entries (an olive stone), where the
        floor hides that with a dither. Kept as a hook so it is one place to change."""
        return 0

    def _draw_section(self, k: int, far: bool) -> None:
        frame = self._frame
        show = frame["show"][k]
        fog = self._fog(float(self._section_depth[k]))
        rgb = self.rgb
        surface = self.surface
        inks = self._inks[k]
        sx, sy = frame["sx"], frame["sy"]
        far_is_left = frame["far_is_left"][k]
        corners_pair = self._corners_pair[k]
        if far:
            order = FAR_LEFT if far_is_left else FAR_RIGHT
        else:
            order = NEAR_RIGHT if far_is_left else NEAR_LEFT
        for strip in order:
            if not show[strip]:
                continue
            if (k, strip) in self.holes and not frame["pair"][k]:
                for number in self.holes[(k, strip)]:
                    self._draw_piece(number, inks[strip], fog)
                self._draw_joints_around_hole(k, strip)
                continue
            if frame["partial"][k][strip]:
                points = self._clipped(k, strip)
                if len(points) < 3:
                    continue
            else:
                corners = (corners_pair if frame["pair"][k] else self._corners[k])[strip]
                points = [(sx[a][b], sy[a][b]) for a, b in corners]
            points = _fit(points, self._size[0], self._size[1])
            if len(points) >= 3:
                pygame.draw.polygon(surface, rgb(inks[strip], fog), points)
            if strip in (LOW_L, LOW_R):
                self._draw_joints(k, strip, frame)
        if far and k % 2 == 0 and show[DECK_STRIP]:
            self._draw_dash(k // 2, fog)

    def _draw_joints(self, k: int, strip: int, frame: dict) -> None:
        """Joints between the blocks of the face: one up each section boundary and one along
        each course, so the face reads as masonry and not as one more asphalt-coloured slab."""
        if frame["pair"][k] or frame["partial"][k][strip] or self._section_depth[k] > LINES_UNTIL:
            return
        left = strip == LOW_L
        base = 20 if left else 23
        top_col = 1 if left else 8
        ground_col = 0 if left else 9
        lit = bool(self.lit_left[k]) if left else not bool(self.lit_left[k])
        ink = self.inks.joint if lit else self.inks.joint_shade
        colour = self.rgb(ink, 0)
        sx, sy = frame["sx"], frame["sy"]
        surface = self.surface
        heights = self._ring_z
        if heights[k][top_col] > 0.3:
            pygame.draw.line(
                surface,
                colour,
                (sx[k][ground_col], sy[k][ground_col]),
                (sx[k][top_col], sy[k][top_col]),
            )
        for level, col in zip(COURSES, range(base, base + len(COURSES)), strict=True):
            if heights[k][top_col] > level + 0.15 and heights[k + 1][top_col] > level + 0.15:
                pygame.draw.line(
                    surface, colour, (sx[k][col], sy[k][col]), (sx[k + 1][col], sy[k + 1][col])
                )

    def _draw_piece(self, number: int, ink: int, fog: int) -> None:
        """A piece of the face around the opening, from the projection done for the frame."""
        sxl, syl, whole = self._piece_frame
        if whole[number]:
            points = _fit(
                list(zip(sxl[number], syl[number], strict=True)), self._size[0], self._size[1]
            )
            if len(points) >= 3:
                pygame.draw.polygon(self.surface, self.rgb(ink, fog), points)
        else:
            self._world_polygon([tuple(p) for p in self.hole_world[number].tolist()], ink, fog)

    def _draw_joints_around_hole(self, k: int, strip: int) -> None:
        """The joints of a section the mouth of the tunnel is cut out of: only outside it."""
        if self._section_depth[k] > LINES_UNTIL:
            return
        left = strip == LOW_L
        col = 0 if left else 9
        lit = bool(self.lit_left[k]) if left else not bool(self.lit_left[k])
        colour = self.rgb(self.inks.joint if lit else self.inks.joint_shade, 0)
        f_lo, f_hi = self.hole_span[(k, strip)]
        x0, y0, x1, y1 = (
            self._ring_x[k][col], self._ring_y[k][col],
            self._ring_x[k + 1][col], self._ring_y[k + 1][col],
        )  # fmt: skip
        h0, h1 = self._ring_z[k][1 if left else 8], self._ring_z[k + 1][1 if left else 8]

        def line(f0: float, z0: float, f1: float, z1: float) -> None:
            a = (x0 + (x1 - x0) * f0, y0 + (y1 - y0) * f0, z0)
            b = (x0 + (x1 - x0) * f1, y0 + (y1 - y0) * f1, z1)
            if min(self._depth_of(*a), self._depth_of(*b)) > NEAR:
                pygame.draw.line(self.surface, colour, self._screen_of(*a), self._screen_of(*b))

        if h0 > 0.3:  # the joint on the boundary of the section: above the opening if it is there
            line(0.0, self.roof if f_lo <= 0.0 < f_hi else 0.0, 0.0, h0)
        for level in COURSES:
            if h0 > level + 0.15 and h1 > level + 0.15:
                if level >= self.roof:
                    line(0.0, level, 1.0, level)
                else:
                    if f_lo > 0.0:
                        line(0.0, level, min(f_lo, 1.0), level)
                    if f_hi < 1.0:
                        line(max(f_hi, 0.0), level, 1.0, level)

    def _ink(self, k: int, strip: int, lit_left: bool) -> int:
        inks = self.inks
        if strip == DECK_STRIP:
            return DECK[k % 2]
        if strip in (LINE_L, LINE_R):
            return inks.line
        if strip in (CAP_L, CAP_R):
            return inks.cap
        left = strip in (LOW_L, UP_L, INNER_R, FRONT_L)  # the face looks to the left of the road
        lit = lit_left if left else not lit_left
        if strip in (UP_L, UP_R):
            return inks.up if lit else inks.up_shade
        if strip in (LOW_L, LOW_R):
            return (inks.face if lit else inks.face_shade)[k % 2]
        if strip in (UNDER_L, UNDER_R):
            return inks.under
        if strip in (FRONT_L, FRONT_R):
            return inks.front if lit else inks.front_shade
        if strip in (LEDGE_L, LEDGE_R):
            return inks.ledge
        return inks.inner if lit else inks.inner_shade

    def _draw_dash(self, index: int, fog: int) -> None:
        xs, ys, ok = self._dash_frame
        if not ok[index]:
            return
        pygame.draw.polygon(
            self.surface,
            self.rgb(self.inks.line, fog),
            list(zip(xs[index], ys[index], strict=True)),
        )

    def _setup_camera(self, pose: CameraPose, projection: Projection, lens: Lens) -> None:
        """The constants of the projection as plain floats, for the few polygons that are
        cut at the near plane (scalar math is much faster than numpy for four points)."""
        self._cam = (
            pose.x,
            pose.y,
            lens.height + pose.z,
            math.cos(pose.heading),
            math.sin(pose.heading),
            math.cos(lens.pitch),
            math.sin(lens.pitch),
            projection.focal,
            projection.centre[0],
            projection.centre[1],
        )

    def _depth_of(self, x: float, y: float, z: float) -> float:
        px, py, eye, ch, sh, cp, sp, _, _, _ = self._cam
        dx, dy = x - px, y - py
        return (dx * ch + dy * sh) * cp - (z - eye) * sp

    def _screen_of(self, x: float, y: float, z: float) -> tuple[float, float]:
        px, py, eye, ch, sh, cp, sp, focal, cx, cy = self._cam
        dx, dy, up = x - px, y - py, z - eye
        ahead, right = dx * ch + dy * sh, dx * sh - dy * ch
        depth = ahead * cp - up * sp
        return cx + focal * right / depth, cy - focal * (ahead * sp + up * cp) / depth

    def _clipped(self, k: int, strip: int):
        """A strip that crosses the near plane: clip it in the world and project the rest."""
        world = [
            (self._ring_x[a][b], self._ring_y[a][b], self._ring_z[a][b])
            for a, b in self._corners[k][strip]
        ]
        depths = [self._depth_of(*p) for p in world]
        return [self._screen_of(*p) for p in _clip(world, depths)]

    def _world_polygon(
        self, world: list[tuple[float, float, float]], ink: int, fog: int = 0
    ) -> None:
        """Project a polygon given in world points (cut at the near plane) and draw it."""
        depths = [self._depth_of(*p) for p in world]
        if min(depths) <= NEAR:
            if max(depths) <= NEAR:
                return
            world = _clip(world, depths)
        points = _fit([self._screen_of(*p) for p in world], self._size[0], self._size[1])
        if len(points) >= 3:
            pygame.draw.polygon(self.surface, self.rgb(ink, fog), points)

    def _draw_portal(self, number: int) -> None:
        """The mouth of the tunnel seen from outside: lintel and jambs around the opening."""
        tau, (ax, ay), (bx, by) = self.portals[number]
        roof = self.roof
        span = math.hypot(bx - ax, by - ay)
        dx, dy = (bx - ax) / span, (by - ay) / span
        a2, b2 = (ax - dx * JAMB, ay - dy * JAMB), (bx + dx * JAMB, by + dy * JAMB)
        top = roof + LINTEL
        fog = self._fog(self._depth_of((ax + bx) / 2, (ay + by) / 2, roof / 2))
        self._world_polygon(
            [(a2[0], a2[1], roof), (b2[0], b2[1], roof), (b2[0], b2[1], top), (a2[0], a2[1], top)],
            LINTEL_INK,
            fog,
        )
        for (px, py), (qx, qy) in (((a2[0], a2[1]), (ax, ay)), ((bx, by), (b2[0], b2[1]))):
            self._world_polygon(
                [(px, py, 0), (qx, qy, 0), (qx, qy, top), (px, py, top)], LINTEL_INK, fog
            )

    def _draw_interior(self) -> None:
        """Inside the tunnel: walls and roof. They are projected together; only the quads that
        cross the near plane are cut (one by one, in scalar math)."""
        quads = self.tunnel_quads
        sx, sy, depth = self._projection.project(
            self._pose, quads[..., 0], quads[..., 1], quads[..., 2]
        )
        whole = (depth > NEAR).all(axis=1).tolist()
        sxl, syl = sx.tolist(), sy.tolist()
        width, height = self._size
        for i, ink in enumerate(self.tunnel_inks):
            if whole[i]:
                points = _fit(list(zip(sxl[i], syl[i], strict=True)), width, height)
                if len(points) >= 3:
                    pygame.draw.polygon(self.surface, self.rgb(ink, 0), points)
            else:
                self._world_polygon([tuple(p) for p in quads[i].tolist()], ink)


__all__ = ["BridgeRenderer"]
