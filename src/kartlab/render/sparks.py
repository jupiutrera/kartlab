"""Sparks that show the sweep charge: hot metal that goes from dull red to white.

The colour follows ``DriftState.charge`` continuously through a ramp of palette inks
(``HEAT_RAMP``). There are no steps: every spark picks one of the two inks around the
charge at random, in proportion, so the mix of the cloud moves smoothly from one ink to the
next. The colour is never the only signal: the rate, the size, the life and the height of
the sparks grow with the charge too, and at full charge they blink and fly longer.

Pure functions only (no pygame, no state), shared by the pseudo-3D scene and the top-down
view. After the release the exhaust flame uses the same ramp, scaled by the push left.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from kartlab.render.palette import Ink

# Dull red, brick red, orange, amber, yellow, pale yellow, white: hotter and brighter each step.
HEAT_RAMP: tuple[int, ...] = (
    Ink.RED_DARK,
    Ink.RED,
    Ink.ORANGE,
    Ink.AMBER,
    Ink.YELLOW,
    28,  # pale yellow
    Ink.WHITE,
)

# Particles with this bit in their colour glow: the lighting variants (dusk, night) do not
# remap them. The ink is ``colour - EMISSIVE``.
EMISSIVE = 64

FULL_BLINK_HZ = 10.0  # at full charge the hottest sparks alternate between the two top inks


def emissive(ink: int) -> int:
    """The particle colour for a glowing spark of this palette ink."""
    return ink + EMISSIVE


def heat_position(heat: float) -> float:
    """Where ``heat`` in [0, 1] falls on the ramp, as an index in [0, len - 1]."""
    return min(max(heat, 0.0), 1.0) * (len(HEAT_RAMP) - 1)


def heat_ink(heat: float, u: float) -> int:
    """The ink of one spark at this heat. ``u`` is a uniform number in [0, 1): the spark takes
    the hotter neighbour of the heat with that probability, so the cloud blends smoothly."""
    pos = heat_position(heat)
    low = int(pos)
    if low >= len(HEAT_RAMP) - 1:
        return HEAT_RAMP[-1]
    return HEAT_RAMP[low + (1 if u < pos - low else 0)]


def mean_luminance(heat: float, luminance: list[float]) -> float:
    """Expected luminance of the cloud at this heat, given the luminance of each ramp ink."""
    pos = heat_position(heat)
    low = min(int(pos), len(HEAT_RAMP) - 2)
    return luminance[low] + (luminance[low + 1] - luminance[low]) * (pos - low)


def blink_ink(heat: float, time_s: float, u: float) -> int:
    """Like ``heat_ink`` but at full heat the top ink alternates with the one below it, a
    subtle flicker that says "ready" without text."""
    if heat >= 1.0 and int(time_s * FULL_BLINK_HZ * 2) % 2 == 0:
        return HEAT_RAMP[-2] if u < 0.85 else HEAT_RAMP[-1]
    return heat_ink(heat, u)


@dataclass(frozen=True)
class SparkSpec:
    rate: float  # sparks per second
    size: float  # edge of a spark (m)
    life: float  # mean life (s)
    lift: float  # mean vertical launch speed (m/s)
    reach: float  # mean speed backwards (m/s)


def spark_spec(charge: float) -> SparkSpec:
    """How the sparks of a sweep look at this charge (0: the dead zone, 1: full)."""
    c = min(max(charge, 0.0), 1.0)
    full = 1.0 if c >= 1.0 else 0.0
    return SparkSpec(
        rate=6.0 + 62.0 * c,  # barely a few sparks in the dead zone
        size=0.05 + 0.09 * c,
        life=0.16 + 0.22 * c + 0.08 * full,  # at full charge they fly longer
        lift=1.2 + 2.2 * c,
        reach=2.0 + 2.5 * c + 1.5 * full,
    )


@dataclass(frozen=True)
class FlameSpec:
    rate: float  # puffs per second
    size: float  # edge of a puff (m)
    life: float  # life of a puff (s)
    speed: float  # speed backwards (m/s)
    heat: float  # where on the ramp its colour falls, in [0, 1]


def flame_spec(level: float, charge: float) -> FlameSpec:
    """The exhaust flame while a push lasts: ``level`` is ``DriftState.push_level`` (1 at the
    release, 0 when it ends) and ``charge`` the one the sweep was released with. It grows
    with ``level`` and cools along the ramp as the push dies."""
    lv = min(max(level, 0.0), 1.0)
    ch = min(max(charge, 0.0), 1.0)
    return FlameSpec(
        rate=90.0 * lv * (0.5 + 0.5 * ch),
        size=0.06 + 0.10 * lv,
        life=0.08 + 0.12 * lv,
        speed=2.0 + 5.0 * lv,
        heat=0.85 * ch * lv,  # never white: the white is for the full charge
    )


def stochastic_count(expected: float, u: float) -> int:
    """Whole number of events for an expected count, unbiased: ``u`` is uniform in [0, 1)."""
    return int(expected + u)


def hash01(*keys: int) -> float:
    """Deterministic pseudo-random number in [0, 1) from integers (for the stateless views)."""
    h = 0.0
    for k in keys:
        h = math.sin((h + k + 1.0) * 12.9898 + 78.233) * 43758.5453
        h -= math.floor(h)
    return h


__all__ = [
    "EMISSIVE",
    "HEAT_RAMP",
    "FlameSpec",
    "SparkSpec",
    "blink_ink",
    "emissive",
    "flame_spec",
    "hash01",
    "heat_ink",
    "heat_position",
    "mean_luminance",
    "spark_spec",
    "stochastic_count",
]
