"""Soft kart-to-kart contact for the human-versus-AI race.

Only the race against the AI uses this. The RL environment never imports it: the agent
trains alone on the track and does not see other karts, so a bump is only a small nudge
that the policy recovers from by itself.

Each kart is two circles of 0.6 m, 0.45 m ahead of and behind its centre. For every pair of
circles (one of each kart) the contact looks at the gap between them and at how fast they
approach along the line between their centres:

- the approach is never allowed to close more than the gap in one step (a "speculative"
  contact: the karts meet, they do not cross, whatever the speed, head on included), so the
  part of the relative velocity that would take them through each other is removed entirely
  (restitution 0: no bounce, no spin, no damage);
- if the circles already overlap, a position correction proportional to the overlap (a
  fraction of it per step, with a cap) pushes the karts apart.

Both the velocity that is removed and the position correction are shared 60 % for the AI and
40 % for the human, so the human is the one who gives way least. The AI never takes more than
``MAX_DV_STEP`` m/s of speed change in one step, nor more than ``MAX_DV_CONTACT`` in one
contact (it does not see karts and must be able to drive on, braking for a corner included,
with a human glued to it): above that the human absorbs the rest, so the approach is still
removed whole and the karts still never cross. The total of a contact starts again after
``QUIET_RESET`` s without contact. Nothing happens inside a grace window (the start, an AI
restart) or between karts at different heights (a bridge over a tunnel).

The module is pure: ``resolve`` returns what to apply and the caller applies it
(``Racer.displace`` for the positions, so the surface is relocated too).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from kartlab.sim.kart import DT, KartState

CIRCLE_RADIUS = 0.6  # (m)
CIRCLE_OFFSET = 0.45  # distance of each circle from the kart centre along its heading (m)
CORRECTION_RATE = 0.4  # share of the overlap corrected in one step
MAX_CORRECTION = 0.25  # most the positions are corrected in one step (m)
SHARE_AI = 0.6  # share of the push the AI kart takes (the human takes the rest)
SHARE_HUMAN = 1.0 - SHARE_AI
MAX_DV_STEP = 0.3  # most speed the AI kart changes in one step of contact (m/s)
MAX_DV_CONTACT = 1.2  # ...and in one whole contact (m/s)
QUIET_RESET = 0.5  # seconds without contact after which the total above starts again
START_GRACE = 4.0  # no contact this long after the start (s)
RESTART_GRACE = 1.5  # no contact this long after an AI kart is put back on the track (s)
LEVEL_GAP = 1.0  # karts at least this far apart in height do not touch (m)

_REACH = 2.0 * (CIRCLE_RADIUS + CIRCLE_OFFSET)  # centres further apart than this cannot touch


@dataclass(slots=True)
class Body:
    """What contact needs of a kart. ``grace_until`` is a race time (s)."""

    kart: KartState
    level_z: float = 0.0  # height of the road under the kart (bridges)
    grace_until: float = 0.0


@dataclass(slots=True)
class PairState:
    """Memory of one human-AI pair: the speed change the AI took in this contact."""

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
    depth: float  # deepest overlap before the correction (m); 0 if the karts only met


def circles(kart: KartState) -> tuple[tuple[float, float], tuple[float, float]]:
    """Centres of the front and rear circle of a kart."""
    cx, cy = math.cos(kart.heading) * CIRCLE_OFFSET, math.sin(kart.heading) * CIRCLE_OFFSET
    return (kart.x + cx, kart.y + cy), (kart.x - cx, kart.y - cy)


def touching(human: Body, ai: Body, now: float) -> bool:
    """Whether the two karts may touch at all right now (grace and levels)."""
    if now < human.grace_until or now < ai.grace_until:
        return False
    return abs(human.level_z - ai.level_z) < LEVEL_GAP


def resolve(human: Body, ai: Body, state: PairState, now: float, dt: float = DT) -> Contact | None:
    """One step of contact between the human kart and one AI kart.

    ``now`` is the race time (s). Returns ``None`` if the karts neither overlap nor would
    cross each other in this step; the caller applies the result (``Racer.displace`` and the
    velocity changes).
    """
    hk, ak = human.kart, ai.kart
    closing = math.hypot(ak.vx - hk.vx, ak.vy - hk.vy) * dt
    if not touching(human, ai, now) or math.hypot(ak.x - hk.x, ak.y - hk.y) > _REACH + closing:
        state.quiet += dt
        if state.quiet >= QUIET_RESET:
            state.used = 0.0
        return None

    hvx, hvy, avx, avy = hk.vx, hk.vy, ak.vx, ak.vy  # velocities after the pairs seen so far
    depth, normal = 0.0, (1.0, 0.0)
    for hx, hy in circles(hk):
        for ax, ay in circles(ak):
            dx, dy = ax - hx, ay - hy
            dist = math.hypot(dx, dy)
            if dist < 1e-9:  # circles on top of each other: use the line between the centres
                dx, dy = ak.x - hk.x, ak.y - hk.y
                dist = math.hypot(dx, dy)
                if dist < 1e-9:
                    dx, dy, dist = 1.0, 0.0, 1.0
            nx, ny = dx / dist, dy / dist  # from the human circle to the AI circle
            gap = dist - 2.0 * CIRCLE_RADIUS
            if gap < 0.0 and -gap > depth:
                depth, normal = -gap, (nx, ny)
            approach = -((avx - hvx) * nx + (avy - hvy) * ny)  # > 0: getting closer
            allowed = max(gap, 0.0) / dt  # the most it may approach and still not overlap
            if approach > allowed:
                remove = approach - allowed
                avx, avy = avx + SHARE_AI * remove * nx, avy + SHARE_AI * remove * ny
                hvx, hvy = hvx - SHARE_HUMAN * remove * nx, hvy - SHARE_HUMAN * remove * ny
    dv_human = (hvx - hk.vx, hvy - hk.vy)
    dv_ai = (avx - ak.vx, avy - ak.vy)
    kick = math.hypot(*dv_ai)
    room = min(MAX_DV_STEP, max(MAX_DV_CONTACT - state.used, 0.0))
    if kick > room:  # the AI keeps its share up to its cap; the human takes the rest
        keep = room / kick
        dv_human = (dv_human[0] - (1.0 - keep) * dv_ai[0], dv_human[1] - (1.0 - keep) * dv_ai[1])
        dv_ai = (keep * dv_ai[0], keep * dv_ai[1])
    state.used += math.hypot(*dv_ai)
    if depth == 0.0 and dv_human == (0.0, 0.0) and dv_ai == (0.0, 0.0):
        state.quiet += dt
        if state.quiet >= QUIET_RESET:
            state.used = 0.0
        return None
    state.quiet = 0.0

    correction = min(CORRECTION_RATE * depth, MAX_CORRECTION)
    nx, ny = normal
    return Contact(
        push_human=(-SHARE_HUMAN * correction * nx, -SHARE_HUMAN * correction * ny),
        push_ai=(SHARE_AI * correction * nx, SHARE_AI * correction * ny),
        dv_human=dv_human,
        dv_ai=dv_ai,
        depth=depth,
    )
