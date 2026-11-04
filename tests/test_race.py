import pytest

from kartlab.sim.autopilot import Autopilot
from kartlab.sim.race import MAX_STEP_DISTANCE, Racer
from kartlab.sim.track import generate_track


@pytest.fixture(scope="module")
def track():
    return generate_track(5)


def test_racer_starts_on_the_line(track):
    racer = Racer(track)
    assert racer.position.s == pytest.approx(0.0, abs=1e-6)
    assert racer.position.lateral == pytest.approx(0.0, abs=1e-6)
    assert not racer.on_grass
    assert racer.laps == 0 and racer.time == 0.0
    assert racer.heading_error() == pytest.approx(0.0, abs=1e-6)


def test_reset_places_the_kart_anywhere(track):
    racer = Racer(track)
    racer.reset(start_s=300.0, lateral=-2.0, heading_offset=0.4, speed=7.0)
    assert racer.position.s == pytest.approx(300.0, abs=0.05)
    assert racer.position.lateral == pytest.approx(-2.0, abs=0.05)
    assert racer.heading_error() == pytest.approx(0.4, abs=0.02)
    assert racer.speed == pytest.approx(7.0)
    assert racer.kart.lateral_speed == pytest.approx(0.0, abs=1e-9)


def test_driving_forward_gains_progress(track):
    racer = Racer(track)
    gained = sum(racer.step(0.0, 1.0) for _ in range(120))
    assert gained > 10.0
    assert racer.progress.distance == pytest.approx(gained)


def test_spinning_in_circles_gains_nothing(track):
    racer = Racer(track)
    for _ in range(60 * 60):
        racer.step(1.0, 0.25)
    assert abs(racer.progress.distance) < 20.0
    assert racer.laps == 0


def test_a_shortcut_across_the_grass_does_not_count(track):
    racer = Racer(track)
    racer.step(0.0, 1.0)
    before = racer.progress.distance
    x, y, heading = track.pose_at(track.length / 2)
    racer.kart.x, racer.kart.y, racer.kart.heading = x, y, heading
    racer.step(0.0, 0.0)
    assert racer.position.s == pytest.approx(track.length / 2, abs=MAX_STEP_DISTANCE)
    assert racer.progress.distance == pytest.approx(before, abs=1.0)


def test_an_invisible_wall_keeps_the_kart_near_the_track(track):
    racer = Racer(track)
    xmin, ymin, xmax, ymax = racer.world
    racer.kart.x, racer.kart.y = xmax + 50.0, ymin - 50.0
    racer.step(0.0, 0.0)
    assert racer.kart.x <= xmax and racer.kart.y >= ymin


def test_laps_and_lap_times_are_recorded(track):
    racer = Racer(track)
    autopilot = Autopilot(track)
    while racer.laps < 2 and racer.time < 300.0:
        racer.step(*autopilot.act(racer))
    assert racer.laps == 2
    assert len(racer.lap_times) == 2
    assert racer.best_lap == min(racer.lap_times)
    assert racer.last_lap == racer.lap_times[-1]
    assert racer.lap_time < 1.0
