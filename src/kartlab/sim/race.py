"""One kart driving on a track: physics, surface, progress and lap times."""

from __future__ import annotations

import math

from kartlab.sim.kart import DT, KartParams, KartState, step_kart
from kartlab.sim.progress import Progress
from kartlab.sim.track import Track, TrackPosition

WORLD_MARGIN = 60.0  # grass around the track's bounding box before an invisible wall (m)
MAX_STEP_DISTANCE = 5.0  # any larger jump along the centerline in one step is a shortcut (m)


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
        self.position: TrackPosition = self.track.locate(x, y)
        self.progress = Progress(self.track.length, self.position.s)
        self.time = 0.0
        self.lap_times: list[float] = []
        self._lap_start = 0.0

    @property
    def on_grass(self) -> bool:
        return abs(self.position.lateral) > self.track.half_width

    @property
    def speed(self) -> float:
        return self.kart.forward_speed

    @property
    def laps(self) -> int:
        return self.progress.laps

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

    def step(self, steer: float, throttle: float) -> float:
        """Advance one physics step and return the net progress gained (m)."""
        step_kart(self.kart, steer, throttle, self.on_grass, self.params)
        self._keep_in_world()
        previous = self.position
        self.position = self.track.locate(self.kart.x, self.kart.y, hint=previous.index)
        laps = self.progress.laps
        jump = (self.position.s - previous.s + self.track.length / 2) % self.track.length
        if abs(jump - self.track.length / 2) > MAX_STEP_DISTANCE:
            self.progress.rebase(self.position.s)
            delta = 0.0
        else:
            delta = self.progress.update(self.position.s)
        self.time += DT
        if self.progress.laps > laps:
            self.lap_times.append(self.time - self._lap_start)
            self._lap_start = self.time
        return delta

    def _keep_in_world(self) -> None:
        kart = self.kart
        xmin, ymin, xmax, ymax = self.world
        if not xmin <= kart.x <= xmax:
            kart.x = min(max(kart.x, xmin), xmax)
            kart.vx = 0.0
        if not ymin <= kart.y <= ymax:
            kart.y = min(max(kart.y, ymin), ymax)
            kart.vy = 0.0
