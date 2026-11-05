"""Cameras for the pseudo-3D view.

A ``Lens`` is the fixed part (height, pitch, field of view, fog) from which the floor
tables are precomputed. A ``CameraPose`` is where it is this frame. The controllers
move the pose smoothly: positions ease towards their target and the heading turns
with damping and a maximum angular speed, so the view never snaps or shakes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from kartlab.sim.kart import KartState
from kartlab.sim.track import Track


@dataclass(frozen=True)
class Lens:
    name: str
    height: float  # camera height above the ground (m)
    pitch: float  # downward tilt (rad); pi/2 looks straight down
    fov: float  # horizontal field of view (rad)
    fog_start: float  # fog begins at this distance along the view (m)
    fog_end: float  # fully fogged from this distance on (m)
    far: float  # floor rows farther than this are left to the sky (m)


CHASE = Lens("chase", 3.2, math.radians(7.0), math.radians(70.0), 60.0, 175.0, 230.0)
BROADCAST = Lens("broadcast", 20.0, math.radians(30.0), math.radians(62.0), 110.0, 300.0, 380.0)
TOP = Lens("top", 60.0, math.radians(90.0), math.radians(60.0), 1e9, 2e9, 1e9)


@dataclass
class CameraPose:
    x: float
    y: float
    heading: float


def _approach_angle(current: float, target: float, rate: float, max_speed: float, dt: float):
    """Damped turn towards ``target`` that never exceeds ``max_speed`` rad/s."""
    error = math.remainder(target - current, math.tau)
    step = error * (1.0 - math.exp(-rate * dt))
    limit = max_speed * dt
    return current + max(-limit, min(limit, step))


class ChaseCamera:
    """Behind the kart, a little raised. Turns with damping and a capped angular speed."""

    lens = CHASE

    def __init__(self, distance: float = 8.0, turn_rate: float = 4.0, max_turn: float = 1.8):
        self.distance = distance
        self.turn_rate = turn_rate
        self.max_turn = max_turn
        self.pose: CameraPose | None = None

    def update(self, kart: KartState, dt: float, track: Track | None = None) -> CameraPose:
        if self.pose is None:
            self.pose = CameraPose(kart.x, kart.y, kart.heading)
        heading = _approach_angle(
            self.pose.heading, kart.heading, self.turn_rate, self.max_turn, dt
        )
        x = kart.x - math.cos(heading) * self.distance
        y = kart.y - math.sin(heading) * self.distance
        k = 1.0 - math.exp(-12.0 * dt)
        self.pose = CameraPose(
            self.pose.x + (x - self.pose.x) * k, self.pose.y + (y - self.pose.y) * k, heading
        )
        return self.pose


class BroadcastCamera:
    """High and steady, like a TV camera: it follows the track direction at the kart's
    position (not the kart's own heading), so an erratic agent does not shake it."""

    lens = BROADCAST

    def __init__(self, distance: float = 28.0, turn_rate: float = 1.2, max_turn: float = 0.6):
        self.distance = distance
        self.turn_rate = turn_rate
        self.max_turn = max_turn
        self.pose: CameraPose | None = None
        self._focus: tuple[float, float] | None = None

    def update(self, kart: KartState, dt: float, track: Track | None = None) -> CameraPose:
        if track is not None:
            target_heading = track.heading_at_index(track.locate(kart.x, kart.y).index)
        else:
            target_heading = kart.heading
        if self.pose is None:
            self.pose = CameraPose(kart.x, kart.y, target_heading)
            self._focus = (kart.x, kart.y)
        k = 1.0 - math.exp(-3.0 * dt)
        fx, fy = self._focus
        self._focus = (fx + (kart.x - fx) * k, fy + (kart.y - fy) * k)
        heading = _approach_angle(
            self.pose.heading, target_heading, self.turn_rate, self.max_turn, dt
        )
        fx, fy = self._focus
        self.pose = CameraPose(
            fx - math.cos(heading) * self.distance, fy - math.sin(heading) * self.distance, heading
        )
        return self.pose


class TopCamera:
    """Straight down over the kart, north always up (heading fixed)."""

    lens = TOP

    def __init__(self) -> None:
        self.pose: CameraPose | None = None

    def update(self, kart: KartState, dt: float, track: Track | None = None) -> CameraPose:
        if self.pose is None:
            self.pose = CameraPose(kart.x, kart.y, math.pi / 2)
        k = 1.0 - math.exp(-4.0 * dt)
        self.pose = CameraPose(
            self.pose.x + (kart.x - self.pose.x) * k,
            self.pose.y + (kart.y - self.pose.y) * k,
            math.pi / 2,
        )
        return self.pose


CAMERAS = {"chase": ChaseCamera, "broadcast": BroadcastCamera, "top": TopCamera}
