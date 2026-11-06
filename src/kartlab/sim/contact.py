"""Soft kart-to-kart contact for the human-versus-AI race.

Only the race against the AI uses this. The RL environment never imports it: the agent
trains alone on the track and does not see other karts, so a bump is only a small nudge
that the policy recovers from by itself.

Each kart is two circles of 0.6 m, 0.45 m ahead of and behind its centre. When a circle
of one kart overlaps a circle of the other, the deepest overlap defines a normal and two
things happen, both gentle on purpose:

- a position correction of at most 0.05 m per step pushes the karts apart;
- the approaching part of the relative velocity along the normal is removed (restitution
  0), at most 0.3 m/s per step and about 2 m/s in total per contact (the total starts
  again after 0.5 s without contact).

The push is shared 60 % for the AI and 40 % for the human. There is no spin, no damage
and no loss of progress. Nothing happens inside a grace window (the start, an AI restart),
between karts at different heights (a bridge over a tunnel) or while one of them is in
the air (a hop clears the other kart).

The module is pure: ``resolve`` returns what to apply and the caller applies it
(``Racer.displace`` for the positions, so the surface is relocated too).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from kartlab.sim.kart import DT, KartState

CIRCLE_RADIUS = 0.6  # (m)
CIRCLE_OFFSET = 0.45  # distance of each circle from the kart centre along its heading (m)
MAX_CORRECTION = 0.05  # position correction per step (m)
MAX_DV_STEP = 0.3  # velocity removed per step (m/s)
MAX_DV_CONTACT = 2.0  # velocity removed in one contact (m/s)
QUIET_RESET = 0.5  # seconds without contact after which the total above starts again
SHARE_AI = 0.6  # share of the push the AI kart takes (the human takes the rest)
SHARE_HUMAN = 1.0 - SHARE_AI
START_GRACE = 4.0  # no contact this long after the start (s)
RESTART_GRACE = 1.5  # no contact this long after an AI kart is put back on the track (s)
LEVEL_GAP = 1.0  # karts at least this far apart in height do not touch (m)
MAX_AIR = 0.25  # a kart higher than this clears the other one (m)

_REACH = 2.0 * (CIRCLE_RADIUS + CIRCLE_OFFSET)  # centres further apart than this cannot touch


@dataclass(slots=True)
class Body:
    """What contact needs of a kart. ``grace_until`` is a race time (s)."""

    kart: KartState
    level_z: float = 0.0  # height of the road under the kart (bridges)
    height: float = 0.0  # height of the kart above the road (hop)
    grace_until: float = 0.0

    @property
    def z(self) -> float:
        return self.level_z + self.height


@dataclass(slots=True)
class PairState:
    """Memory of one human-AI pair: the velocity already removed in this contact."""

    used: float = 0.0
    quiet: float = QUIET_RESET


@dataclass(frozen=True)
class Contact:
    """What one step of contact does. Positions are metres to add to ``x`` and ``y``;
    velocities are m/s to add to ``vx`` and ``vy``."""

    push_human: tuple[float, float]
    push_ai: tuple[float, float]
    dv_human: tuple[float, float]
    dv_ai: tuple[float, float]
    depth: float  # deepest overlap before the correction (m)


def circles(kart: KartState) -> tuple[tuple[float, float], tuple[float, float]]:
    """Centres of the front and rear circle of a kart."""
    cx, cy = math.cos(kart.heading) * CIRCLE_OFFSET, math.sin(kart.heading) * CIRCLE_OFFSET
    return (kart.x + cx, kart.y + cy), (kart.x - cx, kart.y - cy)


def touching(human: Body, ai: Body, now: float) -> bool:
    """Whether the two karts may touch at all right now (grace, levels and hops)."""
    if now < human.grace_until or now < ai.grace_until:
        return False
    if abs(human.z - ai.z) >= LEVEL_GAP:
        return False
    return human.height <= MAX_AIR and ai.height <= MAX_AIR


def deepest_overlap(human: KartState, ai: KartState) -> tuple[float, float, float] | None:
    """Overlap of the deepest pair of circles as ``(depth, nx, ny)`` with the unit normal
    pointing from the human to the AI kart; ``None`` if no circles overlap."""
    if math.hypot(ai.x - human.x, ai.y - human.y) > _REACH:
        return None
    best: tuple[float, float, float] | None = None
    for hx, hy in circles(human):
        for ax, ay in circles(ai):
            dx, dy = ax - hx, ay - hy
            dist = math.hypot(dx, dy)
            depth = 2.0 * CIRCLE_RADIUS - dist
            if depth > 0.0 and (best is None or depth > best[0]):
                if dist < 1e-9:  # circles on top of each other: push along the centres
                    dx, dy = ai.x - human.x, ai.y - human.y
                    dist = math.hypot(dx, dy)
                    if dist < 1e-9:
                        dx, dy, dist = 1.0, 0.0, 1.0
                best = (depth, dx / dist, dy / dist)
    return best


def resolve(human: Body, ai: Body, state: PairState, now: float, dt: float = DT) -> Contact | None:
    """One step of contact between the human kart and one AI kart.

    ``now`` is the race time (s). Returns ``None`` if the karts do not touch; the caller
    applies the result (``Racer.displace`` and the velocity changes).
    """
    overlap = deepest_overlap(human.kart, ai.kart) if touching(human, ai, now) else None
    if overlap is None:
        state.quiet += dt
        if state.quiet >= QUIET_RESET:
            state.used = 0.0
        return None
    state.quiet = 0.0
    depth, nx, ny = overlap

    correction = min(depth, MAX_CORRECTION)
    push_ai = (SHARE_AI * correction * nx, SHARE_AI * correction * ny)
    push_human = (-SHARE_HUMAN * correction * nx, -SHARE_HUMAN * correction * ny)

    closing = (ai.kart.vx - human.kart.vx) * nx + (ai.kart.vy - human.kart.vy) * ny
    dv = 0.0
    if closing < 0.0:  # approaching: remove that part of the relative velocity, gently
        dv = max(0.0, min(-closing, MAX_DV_STEP, MAX_DV_CONTACT - state.used))
        state.used += dv
    return Contact(
        push_human=push_human,
        push_ai=push_ai,
        dv_human=(-SHARE_HUMAN * dv * nx, -SHARE_HUMAN * dv * ny),
        dv_ai=(SHARE_AI * dv * nx, SHARE_AI * dv * ny),
        depth=depth,
    )
