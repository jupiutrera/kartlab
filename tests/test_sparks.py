"""Sweep sparks: the colour, rate and size follow the charge; the flame follows the push."""

import math

import numpy as np
import pygame
import pytest

from kartlab.render.camera import CHASE, CameraPose
from kartlab.render.palette import game_palette
from kartlab.render.scene import KartVisual, WorldRenderer
from kartlab.render.sparks import (
    EMISSIVE,
    HEAT_RAMP,
    blink_ink,
    flame_spec,
    heat_ink,
    mean_luminance,
    spark_spec,
)
from kartlab.sim.drift import DEFAULT_DRIFT, DriftState
from kartlab.sim.kart import DT, KartState
from kartlab.sim.showcase import showcase_track

CHARGES = np.linspace(0.0, 1.0, 21)


@pytest.fixture(scope="module")
def track():
    return showcase_track("harbor")


def luminance(ink: int) -> float:
    r, g, b = game_palette().rgb[ink].astype(float)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def sweeping(charge: float) -> DriftState:
    """A drift state in the middle of a sweep with (about) this charge."""
    cfg = DEFAULT_DRIFT
    t = cfg.charge_delay + charge * (cfg.full_time - cfg.charge_delay)
    return DriftState(active=True, direction=1.0, time=t)


class Rig:
    """A renderer, a kart and a counter of the particles emitted."""

    def __init__(self, track, variant: str = "day") -> None:
        pygame.display.init()
        pygame.display.set_mode((64, 64))
        self.world = WorldRenderer(track, CHASE, variant=variant)
        self.kart = KartState(*track.pose_at(100.0))
        self.kart.vx, self.kart.vy = (
            math.cos(self.kart.heading) * 16,
            math.sin(self.kart.heading) * 16,
        )
        self.emitted: list[list[float]] = []
        real = self.world.particles.emit

        def spy(*args, **kwargs):
            self.emitted.append(list(args))
            real(*args, **kwargs)

        self.world.particles.emit = spy

    def run(self, drift: DriftState, frames: int, lift: float = 0.0) -> None:
        visual = KartVisual(self.kart, 0, drift=drift, lift=lift)
        for _ in range(frames):
            self.world.update_effects([visual], DT)

    def sparks(self) -> list[list[float]]:
        return [p for p in self.emitted if p[7] >= EMISSIVE]


# -- the ramp -------------------------------------------------------------------------


def test_the_ramp_is_made_of_palette_inks_and_gets_brighter_step_by_step():
    assert 5 <= len(HEAT_RAMP) <= 7
    assert all(0 <= ink < 64 for ink in HEAT_RAMP)
    lum = [luminance(ink) for ink in HEAT_RAMP]
    assert all(a < b for a, b in zip(lum, lum[1:], strict=False))


def test_the_colour_grows_hotter_with_the_charge_continuously():
    lum = [luminance(ink) for ink in HEAT_RAMP]
    mean = [mean_luminance(c, lum) for c in CHARGES]
    assert all(a < b for a, b in zip(mean, mean[1:], strict=False))
    # No steps: the biggest jump between neighbouring charges is a fraction of the whole range.
    assert max(b - a for a, b in zip(mean, mean[1:], strict=False)) < 0.2 * (mean[-1] - mean[0])
    # The same through the sampling the sparks really use (every spark picks an ink at random).
    us = (np.arange(400) + 0.5) / 400
    sampled = [np.mean([luminance(heat_ink(c, u)) for u in us]) for c in CHARGES]
    assert all(a < b + 1e-9 for a, b in zip(sampled, sampled[1:], strict=False))
    assert sampled[0] < 0.5 * sampled[-1]
    assert heat_ink(0.0, 0.5) == HEAT_RAMP[0]
    assert heat_ink(1.0, 0.5) == HEAT_RAMP[-1]


def test_intermediate_charges_mix_the_two_neighbouring_inks():
    pos = 2.5 / (len(HEAT_RAMP) - 1)
    inks = {heat_ink(pos, u) for u in np.linspace(0.0, 0.999, 50)}
    assert inks == {HEAT_RAMP[2], HEAT_RAMP[3]}


def test_full_charge_blinks_between_the_two_hottest_inks():
    assert {blink_ink(1.0, t / 100, 0.5) for t in range(40)} == set(HEAT_RAMP[-2:])
    # Below full charge it does not blink.
    assert len({blink_ink(0.6, t / 100, 0.5) for t in range(40)}) == 1


# -- rate, size and life --------------------------------------------------------------


def test_rate_size_life_and_height_grow_with_the_charge():
    specs = [spark_spec(c) for c in CHARGES]
    for field in ("rate", "size", "life", "lift", "reach"):
        values = [getattr(s, field) for s in specs]
        assert all(a < b for a, b in zip(values, values[1:], strict=False)), field
    assert specs[-1].rate > 5 * specs[0].rate  # not only a different colour


def test_more_sparks_and_bigger_ones_are_emitted_as_the_sweep_charges(track):
    counts, sizes = [], []
    for charge in (0.1, 0.4, 0.7, 1.0):
        rig = Rig(track)
        rig.run(sweeping(charge), 240)
        sparks = rig.sparks()
        counts.append(len(sparks))
        sizes.append(np.mean([p[8] for p in sparks]))
    assert counts == sorted(counts) and counts[-1] > 3 * counts[0]
    assert sizes == sorted(sizes) and sizes[-1] > 1.5 * sizes[0]


def test_the_dead_zone_has_hardly_any_sparks_and_dull_ones(track):
    rig = Rig(track)
    drift = DriftState(active=True, direction=1.0, time=0.05)
    assert drift.charge == 0.0
    rig.run(drift, 240)
    full = Rig(track)
    full.run(sweeping(1.0), 240)
    assert len(rig.sparks()) < 0.15 * len(full.sparks())
    assert {p[7] - EMISSIVE for p in rig.sparks()} == {HEAT_RAMP[0]}


def test_sparks_get_hotter_in_the_world(track):
    means = []
    for charge in (0.1, 0.4, 0.7, 1.0):
        rig = Rig(track)
        rig.run(sweeping(charge), 240)
        means.append(np.mean([luminance(p[7] - EMISSIVE) for p in rig.sparks()]))
    assert means == sorted(means) and means[-1] > means[0] + 50


def test_the_rate_does_not_depend_on_the_frame_rate(track):
    slow, fast = Rig(track), Rig(track)
    visual_slow = KartVisual(slow.kart, 0, drift=sweeping(0.8))
    visual_fast = KartVisual(fast.kart, 0, drift=sweeping(0.8))
    for _ in range(120):
        slow.world.update_effects([visual_slow], 1 / 30)
    for _ in range(240):
        fast.world.update_effects([visual_fast], 1 / 60)
    assert len(slow.sparks()) == pytest.approx(len(fast.sparks()), rel=0.2)


# -- the air --------------------------------------------------------------------------


def test_no_sparks_in_the_air(track):
    rig = Rig(track)
    drift = sweeping(1.0)
    rig.run(drift, 60, lift=0.3)  # the visual says it is in the air
    assert rig.sparks() == []
    rig.run(drift, 60)  # back on the ground in the sweep: they start
    assert rig.sparks()


# -- the flame ------------------------------------------------------------------------


def pushing(level: float, charge: float = 1.0) -> DriftState:
    cfg = DEFAULT_DRIFT
    total = charge * cfg.boost_time
    return DriftState(
        push_left=total * level,
        push_total=total,
        push_speed=charge * cfg.boost_speed,
        push_id=0,
    )


def test_the_flame_spec_grows_with_the_push_level():
    specs = [flame_spec(level, 1.0) for level in np.linspace(0.05, 1.0, 12)]
    for field in ("rate", "size", "life", "speed", "heat"):
        values = [getattr(s, field) for s in specs]
        assert all(a < b for a, b in zip(values, values[1:], strict=False)), field
    assert flame_spec(0.0, 1.0).rate == 0.0


def test_the_flame_is_proportional_to_the_push_level(track):
    counts, sizes = [], []
    for level in (0.25, 0.5, 1.0):
        rig = Rig(track)
        drift = pushing(level)
        drift.push_id = 1  # the release flash is not part of the flame
        rig.world._pushes_seen[id(drift)] = 1
        rig.run(drift, 240)
        flame = [p for p in rig.sparks() if p[5] == pytest.approx(0.2)]  # flame puffs
        counts.append(len(flame))
        sizes.append(np.mean([p[8] for p in flame]))
    assert counts == sorted(counts) and counts[-1] > 2 * counts[0]
    assert sizes == sorted(sizes) and counts[0] > 0
    assert counts[1] / counts[2] == pytest.approx(0.5, abs=0.25)
    assert counts[0] / counts[2] == pytest.approx(0.25, abs=0.2)


def test_the_flame_dies_with_the_push(track):
    rig = Rig(track)
    drift = pushing(0.0)
    rig.run(drift, 120)
    assert rig.sparks() == []


def test_the_flame_starts_behind_the_bumper(track):
    rig = Rig(track)
    drift = pushing(1.0)
    rig.world._pushes_seen[id(drift)] = drift.push_id
    rig.run(drift, 120)
    kart = rig.kart
    ahead = [
        (p[0] - kart.x) * math.cos(kart.heading) + (p[1] - kart.y) * math.sin(kart.heading)
        for p in rig.sparks()
    ]
    assert ahead and max(ahead) < -0.9  # never over the kart


# -- the palette, day and night -------------------------------------------------------


@pytest.mark.parametrize("variant", ["day", "sunset", "night"])
def test_the_world_with_sparks_and_flame_keeps_to_the_palette(track, variant):
    rig = Rig(track, variant)
    kart = rig.kart
    pose = CameraPose(kart.x - 6.0, kart.y, kart.heading)
    inks = {tuple(int(c) for c in rgb) for rgb in game_palette().rgb}
    for drift in (sweeping(1.0), pushing(1.0)):
        visual = KartVisual(kart, 0, drift=drift)
        for _ in range(40):
            kart.x += kart.vx * DT
            kart.y += kart.vy * DT
            rig.world.update_effects([visual], DT)
        rig.world.render(pose, [visual], 0.0)
        frame = rig.world.world_array().reshape(-1, 3)
        used = {tuple(int(c) for c in px) for px in np.unique(frame, axis=0)}
        assert used <= inks, used - inks
    assert rig.sparks()


@pytest.mark.parametrize("variant", ["sunset", "night"])
def test_sparks_glow_at_dusk_and_at_night(track, variant):
    rig = Rig(track, variant)
    ink = HEAT_RAMP[4]
    rig.world._draw_particle(10.0, 10.0, 3, ink + EMISSIVE)
    assert tuple(rig.world.world.get_at((10, 9)))[:3] == tuple(game_palette().rgb[ink])
    dust = rig.world._ink_rgb  # a plain ink is dimmed by the variant, a spark is not
    rig.world._draw_particle(10.0, 10.0, 3, ink)
    assert dust[ink] != dust[ink + EMISSIVE]


# -- the top-down view ----------------------------------------------------------------


def _topdown(track, drift, lift=0.0, with_drift=True):
    from kartlab.render.topdown import TopDownView

    view = TopDownView(track, (480, 270))
    screen = pygame.Surface((480, 270))
    x, y, h = track.pose_at(100.0)
    kart = KartState(x, y, h)
    item = (kart, (255, 156, 44), lift, drift) if with_drift else (kart, (255, 156, 44), lift)
    view.draw(screen, [item], (x, y), True)
    return pygame.surfarray.array3d(screen).reshape(-1, 3)


def _spark_pixels(frame) -> int:
    ramp = np.array([game_palette().rgb[ink] for ink in HEAT_RAMP])
    return int((frame[:, None, :] == ramp[None]).all(axis=2).any(axis=1).sum())


def test_topdown_sparks_grow_and_heat_up_with_the_charge(track):
    counts = [_spark_pixels(_topdown(track, sweeping(c))) for c in (0.1, 0.4, 0.7, 1.0)]
    assert counts == sorted(counts) and counts[-1] > 2 * counts[0]


def test_topdown_has_no_sparks_without_a_sweep_or_in_the_air(track):
    assert _spark_pixels(_topdown(track, DriftState())) == 0
    assert _spark_pixels(_topdown(track, sweeping(1.0), lift=0.3)) == 0
    assert _spark_pixels(_topdown(track, None, with_drift=False)) == 0


def test_topdown_flame_follows_the_push_level(track):
    counts = [_spark_pixels(_topdown(track, pushing(level))) for level in (0.2, 0.6, 1.0)]
    assert counts == sorted(counts) and counts[0] > 0 and counts[-1] > counts[0]
