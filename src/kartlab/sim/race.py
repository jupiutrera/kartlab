"""One kart driving on a track: physics, surface, progress and lap times."""

from __future__ import annotations

import math

from kartlab.sim.drift import DriftState
from kartlab.sim.kart import DT, KartParams, KartState, step_kart
from kartlab.sim.progress import Progress
from kartlab.sim.track import Track, TrackPosition

WORLD_MARGIN = 60.0  # grass around the track's bounding box before an invisible wall (m)
MAX_STEP_DISTANCE = 5.0  # any larger jump along the centerline in one step is a shortcut (m)
LAP_SECTORS = 10  # on tracks with crossings a lap needs driving through every tenth of it


class Racer:
    def __init__(
        self,
        track: Track,
        params: KartParams | None = None,
        start_s: float = 0.0,
        lateral: float = 0.0,
    ) -> None:
        self.track = track
        self.params = params or KartParams()
        xmin, ymin, xmax, ymax = track.bounds
        self.world = (
            xmin - WORLD_MARGIN,
            ymin - WORLD_MARGIN,
            xmax + WORLD_MARGIN,
            ymax + WORLD_MARGIN,
        )
        self.reset(start_s, lateral)

    def reset(
        self,
        start_s: float = 0.0,
        lateral: float = 0.0,
        heading_offset: float = 0.0,
        speed: float = 0.0,
    ) -> None:
        x, y, heading = self.track.pose_at(start_s, lateral)
        heading += heading_offset
        self.kart = KartState(
            x, y, heading, vx=speed * math.cos(heading), vy=speed * math.sin(heading)
        )
        # With crossings the start belongs to the branch of ``start_s``, even if the kart is
        # placed right on the crossing; without them this is the plain nearest-point locate.
        hint = self.track.index_at(start_s) if self.track.crossings else None
        self.position: TrackPosition = self.track.locate(x, y, hint)
        self._update_surface()
        self.progress = Progress(self.track.length, self.position.s)
        self.time = 0.0
        self.lap_times: list[float] = []
        self._lap_start = 0.0
        self._laps = 0
        self._visited: set[int] = set()
        self.wall_hit = False  # the last step pushed the kart off a wall or the embankment

    @property
    def on_grass(self) -> bool:
        """True if the kart is on no part of the track.

        Tracks with level crossings ask the whole track (the second road of a crossing is
        asphalt too, though it is another branch); every other track compares the lateral
        offset with the half width, exactly as it always did. With a bridge each branch is
        its own surface: the lateral offset from its own centerline decides.
        """
        if self.track.crossings and self.track.levels is None:
            return self._grass
        return abs(self.position.lateral) > self.track.half_width

    @property
    def level_z(self) -> float:
        """Height (m) of the road under the kart: 0 except on a bridge and its ramps.

        Other code that asks "how high is this kart" (contact between karts, the renderer's
        ``lift``) adds the hop height of the sweep on top of it.
        """
        return self.track.height_at(self.position.s)

    def displace(self, dx: float, dy: float) -> None:
        """Move the kart by ``(dx, dy)`` metres outside of the physics step (a nudge from
        another kart, a put-back) and relocate it on the track and its surface. Progress
        is untouched: the next step credits whatever distance the nudge moved it."""
        self.kart.x += dx
        self.kart.y += dy
        self._keep_in_world()
        self.position = self.track.locate(self.kart.x, self.kart.y, hint=self.position.index)
        self._update_surface()

    def _update_surface(self) -> None:
        if self.track.crossings and self.track.levels is None:
            self._grass = not self.track.contains(self.kart.x, self.kart.y)

    @property
    def speed(self) -> float:
        return self.kart.forward_speed

    @property
    def laps(self) -> int:
        """Completed laps. Other tracks just use the net distance. On a track with crossings,
        turning from one branch onto the other inside the crossing skips part of the track:
        the jump is charged to the distance (see ``step``), so repeating a lobe adds nothing,
        and as a second guard a lap only counts if the kart also drove forward through every
        tenth of the track since the last one."""
        return self._laps if self.track.crossings else self.progress.laps

    @property
    def lap_time(self) -> float:
        """Time spent on the current lap (s)."""
        return self.time - self._lap_start

    @property
    def last_lap(self) -> float | None:
        return self.lap_times[-1] if self.lap_times else None

    @property
    def best_lap(self) -> float | None:
        return min(self.lap_times) if self.lap_times else None

    def heading_error(self) -> float:
        """Kart heading minus track heading, in (-pi, pi]."""
        track_heading = self.track.heading_at_index(self.position.index)
        return math.remainder(self.kart.heading - track_heading, math.tau)

    def step(self, steer: float, throttle: float, drift: DriftState | None = None) -> float:
        """Advance one physics step and return the net progress gained (m).

        ``drift`` is only passed by manual driving; with ``None`` nothing changes.
        """
        step_kart(self.kart, steer, throttle, self.on_grass, self.params, drift=drift)
        self._keep_in_world()
        previous = self.position
        self.position = self.track.locate(self.kart.x, self.kart.y, hint=previous.index)
        if self.track.levels is not None:
            self._collide(previous)
        self._update_surface()
        laps = self.progress.laps
        jump = (self.position.s - previous.s + self.track.length / 2) % self.track.length
        if abs(jump - self.track.length / 2) > MAX_STEP_DISTANCE:
            if self.track.crossings:
                # A branch switch at a crossing: s jumps although the kart barely moved. It
                # never credits distance; it is charged as the way back along the centerline,
                # so any closed loop (one lobe over and over) nets zero, however it is driven.
                back = (self.position.s - previous.s) % self.track.length
                delta = back - self.track.length if back > 0.0 else 0.0
                self.progress.charge(self.position.s, delta)
            else:
                self.progress.rebase(self.position.s)
                delta = 0.0
        else:
            delta = self.progress.update(self.position.s)
        self.time += DT
        if self.track.crossings:
            self._count_lap(delta, laps)
        elif self.progress.laps > laps:
            self.lap_times.append(self.time - self._lap_start)
            self._lap_start = self.time
        return delta

    def _collide(self, previous: TrackPosition) -> None:
        """Walls of the bridge and tunnel, and the solid ground under the ramps."""
        levels = self.track.levels
        assert levels is not None
        self.wall_hit = False
        for _ in range(2):  # a push out of the ground can end against a wall
            if not levels.constrain(self.kart, self.position, previous):
                break
            self.wall_hit = True
            self.position = self.track.locate(self.kart.x, self.kart.y, hint=self.position.index)

    def _count_lap(self, delta: float, laps_before: int) -> None:
        """Lap counting for tracks with crossings (see ``laps``)."""
        if delta > 0.0:
            self._visited.add(int(self.position.s / self.track.length * LAP_SECTORS) % LAP_SECTORS)
        if self.progress.laps > laps_before and len(self._visited) == LAP_SECTORS:
            self._laps += 1
            self._visited = set()
            self.lap_times.append(self.time - self._lap_start)
            self._lap_start = self.time

    def _keep_in_world(self) -> None:
        kart = self.kart
        xmin, ymin, xmax, ymax = self.world
        if not xmin <= kart.x <= xmax:
            kart.x = min(max(kart.x, xmin), xmax)
            kart.vx = 0.0
        if not ymin <= kart.y <= ymax:
            kart.y = min(max(kart.y, ymin), ymax)
            kart.vy = 0.0
