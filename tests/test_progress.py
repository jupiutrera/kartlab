import numpy as np
import pytest

from kartlab.sim.progress import Progress

LENGTH = 100.0


def drive_to(progress: Progress, start: float, end: float, step: float = 0.5) -> float:
    """Move along arc positions from ``start`` to ``end`` (unwrapped) and sum the deltas."""
    direction = 1.0 if end >= start else -1.0
    total = 0.0
    for s in np.arange(start + direction * step, end + direction * step / 2, direction * step):
        total += progress.update(float(s) % LENGTH)
    return total


def test_a_full_lap_counts_once():
    progress = Progress(LENGTH, 0.0)
    gained = drive_to(progress, 0.0, 100.5)
    assert gained == pytest.approx(100.5)
    assert progress.laps == 1


def test_laps_count_from_a_random_start():
    progress = Progress(LENGTH, 63.0)
    drive_to(progress, 63.0, 63.0 + 199.0)
    assert progress.laps == 1
    drive_to(progress, 63.0 + 199.0, 63.0 + 201.0)
    assert progress.laps == 2


def test_going_back_and_forth_over_the_line_adds_nothing():
    progress = Progress(LENGTH, 95.0)
    for _ in range(20):
        drive_to(progress, 95.0, 105.0)
        drive_to(progress, 105.0, 95.0)
    assert progress.distance == pytest.approx(0.0, abs=1e-9)
    assert progress.laps == 0


def test_driving_backwards_is_negative():
    progress = Progress(LENGTH, 50.0)
    gained = drive_to(progress, 50.0, 20.0)
    assert gained == pytest.approx(-30.0)
    assert progress.distance == pytest.approx(-30.0)


def test_backing_up_must_be_made_up_before_the_next_lap():
    progress = Progress(LENGTH, 0.0)
    drive_to(progress, 0.0, -20.0)
    drive_to(progress, -20.0, 99.0)
    assert progress.laps == 0
    drive_to(progress, 99.0, 100.5)
    assert progress.laps == 1


def test_a_lap_is_never_counted_twice():
    progress = Progress(LENGTH, 0.0)
    drive_to(progress, 0.0, 101.0)
    drive_to(progress, 101.0, 98.0)
    drive_to(progress, 98.0, 102.0)
    assert progress.laps == 1


def test_rebase_moves_without_counting_distance():
    progress = Progress(LENGTH, 10.0)
    progress.rebase(60.0)
    assert progress.distance == 0.0
    assert progress.update(61.0) == pytest.approx(1.0)
