import numpy as np
import pytest

from kartlab.sim.autopilot import Autopilot
from kartlab.sim.race import Racer
from kartlab.sim.showcase import SHOWCASE, showcase_track
from kartlab.sim.track import TrackParams, has_clearance
from kartlab.sim.tracksets import (
    EVAL_SEEDS,
    eval_track,
    is_eval_seed,
    sample_train_seed,
    train_track,
)


def test_training_never_draws_an_evaluation_seed():
    rng = np.random.default_rng(0)
    assert not any(is_eval_seed(sample_train_seed(rng)) for _ in range(20_000))
    assert all(is_eval_seed(seed) for seed in EVAL_SEEDS)


def test_training_refuses_evaluation_seeds():
    with pytest.raises(ValueError):
        train_track(EVAL_SEEDS[0])


def test_evaluation_tracks_are_fixed_and_named():
    assert len(EVAL_SEEDS) == 20
    a, b = eval_track(3), eval_track(3)
    np.testing.assert_array_equal(a.points, b.points)
    assert a.name == "Evaluación 4"


@pytest.mark.parametrize("key", SHOWCASE)
def test_showcase_tracks_respect_the_limits(key):
    track = showcase_track(key)
    limits = TrackParams(width=SHOWCASE[key].width)
    assert track.name == SHOWCASE[key].name
    assert np.max(np.abs(track.curvature)) <= 1 / limits.min_radius
    assert has_clearance(track.points, track.width, limits.clearance)


DRIVABLE = [pytest.param(lambda k=k: showcase_track(k), id=k) for k in SHOWCASE] + [
    pytest.param(lambda i=i: eval_track(i), id=f"eval-{i + 1}") for i in range(5)
]


@pytest.mark.parametrize("make_track", DRIVABLE)
def test_autopilot_laps_every_track_without_leaving_the_asphalt(make_track):
    track = make_track()
    racer = Racer(track)
    autopilot = Autopilot(track)
    widest = 0.0
    while racer.laps < 1 and racer.time < 200.0:
        racer.step(*autopilot.act(racer))
        widest = max(widest, abs(racer.position.lateral))
    assert racer.laps == 1
    assert widest < track.half_width - 1.0
    assert 12.0 < track.length / racer.lap_times[0] < 25.0
