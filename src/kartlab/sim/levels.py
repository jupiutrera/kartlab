"""A bridge and a tunnel at a level crossing: heights, walls and the solid embankment.

A track with a ``bridge`` (see ``trackfile.py``) turns one of its level crossings into two
levels. One branch climbs a ramp, runs over a flat deck and comes down again; the other one
stays on the ground and goes through a tunnel under the deck. The simulation stays two
dimensional (the height changes no speed and no grip): the height is for the renderer, for
karts that must not touch because they are at different heights, and for the walls.

- ``height`` is the deck height per centerline sample: ``deck_height`` on the deck, a smooth
  ramp of ``ramp`` metres on each side of it and 0 elsewhere (the whole lower branch).
- ``walled`` marks the samples with solid walls on both sides of the road: the parapets of
  the raised stretch (and its ramps) and the tunnel under the deck. The wall stands
  ``WALL_INSET`` inside the asphalt edge and is one-way: a kart that is inside the walls
  stays inside, one that is outside stays outside, so a kart on the grass is never pulled
  onto the road.
- The embankment is the solid ground under the ramps and the deck: a strip around the upper
  road, ``EMBANKMENT_MARGIN`` wider than the asphalt on each side. A kart of the lower
  branch cannot enter it except through the tunnel.

Nothing here knows about the environment, the renderer or pygame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kartlab.sim.kart import KartState
    from kartlab.sim.track import Crossing, Track, TrackPosition

OVER_CHOICES = ("first", "second")  # which branch, in driving order from the line, goes over
DEFAULT_OVER = "second"  # first the tunnel, then the bridge
DEFAULT_DECK_HEIGHT = 3.5  # (m)
DECK_HEIGHT_RANGE = (2.0, 6.0)
DEFAULT_RAMP = 25.0  # horizontal length of each ramp (m)
RAMP_RANGE = (15.0, 50.0)

DECK_MARGIN = 3.0  # flat deck beyond the footprint of the crossing, on each side (m)
WALL_INSET = 0.7  # the wall stands this far inside the asphalt edge (m)
WALL_MIN_HEIGHT = 0.2  # the road is walled where it is higher than this (m)
EMBANKMENT_MARGIN = 0.6  # solid ground beyond the asphalt edge of the upper road (m)
OUTSIDE_EPS = 0.01  # a kart pushed out of a wall rests this far beyond the asphalt edge (m)
DECK_THICKNESS = 0.7  # the underside of the deck is this far below its surface (m)
MIN_CEILING = 1.6  # the tunnel is never lower than this (m)
PORTAL_MARGIN = 1.0  # extra tunnel length at each portal (m)
START_CLEARANCE = 10.0  # the whole ramp keeps this far from the start line (m)
MIN_LENGTH_RATIO = 4.0  # the track is at least this many times longer than the bridge


@dataclass(frozen=True)
class BridgeSpec:
    """The ``bridge`` key of a track file: which crossing is a bridge, and its shape."""

    crossing: int  # index into the file's ``crossings``
    over: str = DEFAULT_OVER  # "first" or "second": the branch (in driving order) that goes over
    deck_height: float = DEFAULT_DECK_HEIGHT
    ramp: float = DEFAULT_RAMP


def deck_half_length(width: float, angle: float) -> float:
    """Half the flat deck (m): it covers the whole footprint of the road underneath.

    ``angle`` is the acute angle between the branches. The lower road, ``width`` wide,
    crosses the upper one obliquely: its footprint along the upper axis is
    ``width / sin`` plus the upper road's own half width over ``tan`` on each end.
    """
    half = width / 2
    return width / math.sin(angle) / 2 + half / math.tan(angle) + DECK_MARGIN


def tunnel_half_length(width: float, angle: float) -> float:
    """Half the tunnel (m), measured along the lower road (same geometry as the deck)."""
    half = width / 2
    reach = half + EMBANKMENT_MARGIN
    return reach / math.sin(angle) + half / math.tan(angle) + PORTAL_MARGIN


def bridge_reach(width: float, angle: float, bridge: BridgeSpec) -> float:
    """Half length of the whole raised stretch, ramps included (m)."""
    return deck_half_length(width, angle) + bridge.ramp


def smoothstep(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


class Levels:
    """Heights, walls and embankment of a track with a bridge. See the module docstring."""

    def __init__(self, track: Track, bridge: BridgeSpec, crossing: Crossing) -> None:
        self.bridge = bridge
        self.crossing = crossing
        self.length = track.length
        n = len(track)
        self._n = n
        self.half_width = track.half_width
        self.over_first = bridge.over == "first"
        self.s_over = crossing.s_a if self.over_first else crossing.s_b
        self.s_under = crossing.s_b if self.over_first else crossing.s_a
        self.deck_height = float(bridge.deck_height)
        self.ramp = float(bridge.ramp)
        self.deck_half = deck_half_length(track.width, crossing.angle)
        self.reach = self.deck_half + self.ramp  # half length of the raised stretch (m)
        self.tunnel_half = tunnel_half_length(track.width, crossing.angle)
        self.embankment_half = track.half_width + EMBANKMENT_MARGIN  # across the upper road (m)
        self.ceiling = max(self.deck_height - DECK_THICKNESS, MIN_CEILING)  # tunnel roof (m)

        s = track.s
        away = np.abs(self.wrap(s - self.s_over))
        ramp_progress = smoothstep((away - self.deck_half) / self.ramp)
        self.height = self.deck_height * (1.0 - ramp_progress)  # per sample
        under = np.abs(self.wrap(s - self.s_under))
        self.walled = (self.height > WALL_MIN_HEIGHT) | (under <= self.tunnel_half)
        # The raised stretch (ramps included): the renderer draws it as geometry, not as floor.
        self.raised = away <= self.reach
        # Samples whose road may meet the embankment (the lower branch near the crossing).
        self._near_under = under <= self.reach + 2.0 * self.embankment_half
        # Samples that make up the embankment: the upper road's raised stretch.
        self._bank = np.flatnonzero(away <= self.reach + 1.0)
        self._bank_x = np.ascontiguousarray(track.points[self._bank, 0])
        self._bank_y = np.ascontiguousarray(track.points[self._bank, 1])
        # Plain copies for the per-step scalar code.
        self._height_list = self.height.tolist()
        self._walled_list = self.walled.tolist()
        self._near_under_list = self._near_under.tolist()
        self._track = track

    # -- heights -------------------------------------------------------------------------

    def wrap(self, value):
        """Signed arc distance in [-length/2, length/2); works on numbers and arrays."""
        return (value + self.length / 2) % self.length - self.length / 2

    def height_at(self, s: float) -> float:
        """Deck height (m) at arc position ``s``, linear between samples."""
        track = self._track
        s = s % self.length
        i = track.index_at(s)
        t = (s - track._s[i]) / track._seg_len[i]
        h0 = self._height_list[i]
        h1 = self._height_list[(i + 1) % self._n]
        return h0 + (h1 - h0) * t

    def in_tunnel_zone(self, x: float, y: float, margin: float = 0.0) -> bool:
        """True if the point of the plane is over the tunnel's floor, or within ``margin`` metres
        of either mouth (the straight stretch of the lower road under the deck)."""
        track = self._track
        cx, cy, heading = track.pose_at(self.s_under)
        dx, dy = x - cx, y - cy
        along = dx * math.cos(heading) + dy * math.sin(heading)
        across = -dx * math.sin(heading) + dy * math.cos(heading)
        return abs(across) <= self.half_width and abs(along) <= self.tunnel_half + margin

    def is_upper(self, s: float) -> bool:
        """True where the road is raised (deck and ramps)."""
        return self.height_at(s) > WALL_MIN_HEIGHT

    # -- walls and embankment ------------------------------------------------------------

    def constrain(
        self, kart: KartState, position: TrackPosition, previous: TrackPosition | None
    ) -> bool:
        """Keep ``kart`` out of the walls and the embankment (in place); True if it moved.

        ``position`` is where the kart is now (on its own branch) and ``previous`` where it
        was one step ago. The correction relocates the kart to the surface of the obstacle
        and removes only the velocity that points into it: no tangential friction.
        """
        wall = self._wall(kart, position, previous)
        moved = wall != 0
        # ``position`` is from before the wall moved the kart: a kart the wall pushed back into
        # the tunnel is in the tunnel, whatever its old lateral offset said.
        if self._near_under_list[position.index] and self._embankment(
            kart, position, pushed_in=wall == 1
        ):
            moved = True
        return moved

    def _wall(self, kart: KartState, position: TrackPosition, previous: TrackPosition | None):
        """0 if the wall did nothing, 1 if it pushed the kart in, 2 if it pushed it out."""
        i = position.index
        if not self._walled_list[i]:
            return 0
        track = self._track
        lateral = position.lateral
        reach = abs(lateral)
        outer = self.half_width  # the outside face of the wall: the asphalt edge
        inner = outer - WALL_INSET  # the inside face
        # Inside the walls unless the kart came from the grass (it was beyond the asphalt).
        inside = previous is None or abs(previous.lateral) <= outer
        if inside and reach <= inner or not inside and reach > outer:
            return 0
        sign = 1.0 if lateral >= 0.0 else -1.0
        nx, ny = -track._dir_y[i], track._dir_x[i]  # left normal: lateral grows along it
        away = sign if inside else -sign  # which way the wall blocks the kart
        # Towards the free side; outside, a hair past the face so the next step still sees the
        # kart as outside (``outer`` itself counts as asphalt).
        shift = (inner - reach) if inside else (outer + OUTSIDE_EPS - reach)
        kart.x += sign * nx * shift
        kart.y += sign * ny * shift
        self._cancel(kart, away * nx, away * ny)
        return 1 if inside else 2

    def _embankment(self, kart: KartState, position: TrackPosition, pushed_in: bool) -> bool:
        """Push a kart of the lower branch out of the solid ground under the upper road."""
        inner = self.half_width - WALL_INSET
        if self._walled_list[position.index] and (
            pushed_in or abs(position.lateral) <= inner + 1e-6
        ):
            return False  # inside the tunnel
        hit = self._bank_hit(kart.x, kart.y)
        if hit is None:
            return False
        dx, dy, depth = hit
        kart.x += dx * depth
        kart.y += dy * depth
        self._cancel(kart, -dx, -dy)
        return True

    def _bank_hit(self, x: float, y: float) -> tuple[float, float, float] | None:
        """``(dx, dy, depth)`` to push a point out of the embankment, or None if outside.

        ``(dx, dy)`` is the unit direction of the shortest way out and ``depth`` how far.
        """
        track = self._track
        nearest = int(np.argmin((self._bank_x - x) ** 2 + (self._bank_y - y) ** 2))
        if nearest == 0 or nearest == len(self._bank) - 1:
            return None  # beyond the end of the raised stretch
        index = int(self._bank[nearest])
        pos = track._project(x, y, index)
        along = self.wrap(pos.s - self.s_over)
        across = abs(pos.lateral)
        if across >= self.embankment_half or abs(along) >= self.reach:
            return None
        ux, uy = track._dir_x[pos.index], track._dir_y[pos.index]
        if self.embankment_half - across <= self.reach - abs(along):
            side = 1.0 if pos.lateral >= 0.0 else -1.0
            return -uy * side, ux * side, self.embankment_half - across
        end = 1.0 if along >= 0.0 else -1.0
        return ux * end, uy * end, self.reach - abs(along)

    @staticmethod
    def _cancel(kart: KartState, ox: float, oy: float) -> None:
        """Remove the velocity component along ``(ox, oy)`` (the way the kart is blocked)."""
        into = kart.vx * ox + kart.vy * oy
        if into > 0.0:
            kart.vx -= into * ox
            kart.vy -= into * oy
