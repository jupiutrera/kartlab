"""Time trial: record storage, the ghost, the live gap and the headless drive.py run."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pygame
import pytest

from kartlab.render.view import format_delta
from kartlab.show.drive import DriveApp, main
from kartlab.sim import timetrial
from kartlab.sim.autopilot import Autopilot
from kartlab.sim.drift import DriftState
from kartlab.sim.kart import DT, KartParams, KartState, step_kart
from kartlab.sim.race import Racer
from kartlab.sim.showcase import showcase_track
from kartlab.sim.timetrial import (
    LapEvent,
    Record,
    RecordStore,
    RecordWarning,
    TimeTrial,
    record_filename,
    track_signature,
)
from kartlab.sim.tracksets import eval_track
from kartlab.sim.version import PHYSICS_VERSION

# Pinned state of the physics fingerprint below (see sim/version.py).
PINNED_PHYSICS_VERSION = 1
PINNED_FINGERPRINT = {
    "plain": (92.92293, 134.78164, 1.28053, 5.64117, 19.83995),
    "drift": (-40.39967, 72.42166, 2.55791, -17.65391, 11.33072),
}


@pytest.fixture(scope="module")
def track():
    return showcase_track("harbor")


def make_trial(track, tmp_path: Path) -> tuple[Racer, TimeTrial, Autopilot]:
    racer = Racer(track)
    trial = TimeTrial(track, RecordStore(tmp_path))
    trial.begin(racer)
    return racer, trial, Autopilot(track)


def drive_lap(racer, trial, pilot, throttle_scale: float = 1.0, manual: bool = True):
    """Drive until a lap is completed; returns its event."""
    for _ in range(60 * 400):
        steer, throttle = pilot.act(racer)
        racer.step(steer, throttle * throttle_scale)
        event = trial.update(racer, manual)
        if event is not None:
            return event
    raise AssertionError("no lap completed in 400 s")


@pytest.fixture(scope="module")
def lap(track, tmp_path_factory):
    """One recorded autopilot lap (module-wide: driving a lap takes a moment)."""
    folder = tmp_path_factory.mktemp("lap")
    racer, trial, pilot = make_trial(track, folder)
    event = drive_lap(racer, trial, pilot)
    assert event.record
    return trial.record


def fresh_copy(record: Record, **changes) -> Record:
    fields = {
        "lap_time": record.lap_time,
        "t": record.t.copy(),
        "x": record.x.copy(),
        "y": record.y.copy(),
        "heading": record.heading.copy(),
        "d": record.d.copy(),
        "signature": record.signature,
        "physics_version": record.physics_version,
    }
    fields.update(changes)
    return Record(**fields)


def scaled(record: Record, factor: float) -> Record:
    """The same lap driven ``factor`` times as slowly (``< 1``: faster)."""
    return fresh_copy(record, lap_time=round(record.lap_time * factor, 6), t=record.t * factor)


# -- storage ------------------------------------------------------------------------------


def test_save_and_load_round_trip(track, lap, tmp_path):
    store = RecordStore(tmp_path)
    path = store.save(track, lap)
    assert path.parent == tmp_path and path.name == record_filename(track)
    loaded = store.load(track)
    assert loaded is not None
    assert loaded.lap_time == lap.lap_time
    assert loaded.physics_version == PHYSICS_VERSION
    for name in ("t", "x", "y", "heading", "d"):
        np.testing.assert_allclose(getattr(loaded, name), getattr(lap, name), atol=1e-9)
    assert loaded.signature == json.loads(json.dumps(lap.signature))
    assert not list(tmp_path.glob(".*.tmp")), "temporary files must not stay behind"


def test_record_file_is_utf8_with_lf_and_small(track, lap, tmp_path):
    path = RecordStore(tmp_path).save(track, lap)
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"}\n")
    assert not raw.startswith(b"\xef\xbb\xbf")
    data = json.loads(raw.decode("utf-8"))
    assert data["track"]["name"] == "Puerto"
    assert len(data["t"]) == len(data["x"]) == len(data["d"]) > 100
    assert len(raw) < 200_000


def test_non_ascii_track_name_survives_utf8(tmp_path):
    from kartlab.sim.track import Track

    base = showcase_track("canyon")
    track = Track(base.points, base.width, "Cañón del Águila", clearance=base.clearance)
    racer = Racer(track)
    trial = TimeTrial(track, RecordStore(tmp_path))
    trial.begin(racer)
    for _ in range(10):
        racer.step(0.0, 1.0)
        trial.update(racer)
    record = Record.from_samples(trial._samples[-1][0], trial._samples, trial._signature)
    path = RecordStore(tmp_path).save(track, record)
    assert "Cañón del Águila" in path.read_bytes().decode("utf-8")
    assert path.name.startswith("canon-del-aguila-")
    assert RecordStore(tmp_path).load(track) is not None


def test_record_with_other_physics_version_is_ignored(track, lap, tmp_path):
    store = RecordStore(tmp_path)
    store.save(track, fresh_copy(lap, physics_version=PHYSICS_VERSION + 1))
    with pytest.warns(RecordWarning, match="física"):
        assert store.load(track) is None
    assert "física" in store.problem


def test_record_of_a_changed_track_is_ignored(track, lap, tmp_path):
    from kartlab.sim.track import Track

    store = RecordStore(tmp_path)
    store.save(track, lap)
    moved = track.points.copy()
    moved[len(moved) // 3] += (1.0, 0.0)  # a metre-wide bump on a sampled point
    changed = Track(moved, track.width, track.name, clearance=track.clearance)
    changed_path = store.path_for(changed)
    assert changed_path == store.path_for(track), "same identity, different geometry"
    with pytest.warns(RecordWarning, match="circuito"):
        assert store.load(changed) is None
    assert store.load(track) is not None


def test_record_of_another_track_is_not_loaded(track, lap, tmp_path):
    store = RecordStore(tmp_path)
    store.save(track, lap)
    other = showcase_track("canyon")
    assert store.path_for(other) != store.path_for(track)
    assert store.load(other) is None and store.problem is None


def test_signature_tolerates_platform_noise_but_not_a_shifted_start(track, lap):
    noisy = track.points + np.random.default_rng(0).normal(0.0, 1e-9, track.points.shape)
    from kartlab.sim.track import Track

    same = Track(noisy, track.width, track.name, clearance=track.clearance)
    assert timetrial.signature_matches(lap.signature, same)
    shifted = Track(
        np.roll(track.points, 8, axis=0), track.width, track.name, clearance=track.clearance
    )
    assert not timetrial.signature_matches(lap.signature, shifted), "another finish line"


def test_filename_is_stable_for_generated_tracks():
    a, b = eval_track(0), eval_track(0)
    assert record_filename(a) == record_filename(b)
    assert record_filename(a) != record_filename(eval_track(1))
    assert track_signature(a) == track_signature(b)


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"{",
        b"[]",
        b'{"format": "kartlab-record"}',
        b"\xff\xfe\x00 not utf8",
        b'{"format":"kartlab-record","version":1,"physics_version":1,"lap_time":NaN}',
    ],
)
def test_corrupt_file_does_not_break_startup(track, tmp_path, content):
    store = RecordStore(tmp_path)
    store.path_for(track).write_bytes(content)
    with pytest.warns(RecordWarning, match="dañado"):
        trial = TimeTrial(track, store)
    assert trial.record is None and "dañado" in trial.problem


def test_truncated_file_does_not_break_startup(track, lap, tmp_path):
    store = RecordStore(tmp_path)
    path = store.save(track, lap)
    raw = path.read_bytes()
    for cut in (len(raw) // 2, len(raw) - 3, 10):
        path.write_bytes(raw[:cut])
        with pytest.warns(RecordWarning):
            assert TimeTrial(track, store).record is None


def test_inconsistent_record_is_rejected(track, lap, tmp_path):
    store = RecordStore(tmp_path)
    data = json.loads(lap.to_json())
    data["x"] = data["x"][:-1]  # columns of different lengths
    store.path_for(track).write_text(json.dumps(data), encoding="utf-8", newline="\n")
    with pytest.warns(RecordWarning):
        assert store.load(track) is None
    data = json.loads(lap.to_json())
    data["t"][3] = -1.0  # time going backwards
    store.path_for(track).write_text(json.dumps(data), encoding="utf-8", newline="\n")
    with pytest.warns(RecordWarning):
        assert store.load(track) is None


# -- laps ---------------------------------------------------------------------------------


def test_first_lap_becomes_the_record_and_a_slower_one_does_not_replace_it(track, tmp_path):
    racer, trial, pilot = make_trial(track, tmp_path)
    first = drive_lap(racer, trial, pilot)
    assert isinstance(first, LapEvent)
    assert first.record and first.previous is None and first.saved
    best = trial.record.lap_time
    assert best == pytest.approx(first.lap_time, abs=1e-6)
    assert RecordStore(tmp_path).load(track).lap_time == best
    # A slower lap (reduced throttle) is completed but leaves record and file untouched.
    slow = drive_lap(racer, trial, pilot, throttle_scale=0.4)
    assert not slow.record and slow.lap_time > best and slow.previous == best
    assert trial.last_lap == slow.lap_time
    assert trial.record.lap_time == best
    assert RecordStore(tmp_path).load(track).lap_time == best


def test_a_faster_lap_replaces_the_record(track, lap, tmp_path):
    store = RecordStore(tmp_path)
    store.save(track, scaled(lap, 1.2))
    racer, trial, pilot = make_trial(track, tmp_path)
    assert trial.record.lap_time == pytest.approx(lap.lap_time * 1.2)
    event = drive_lap(racer, trial, pilot)
    assert event.record and event.previous == pytest.approx(lap.lap_time * 1.2)
    assert RecordStore(tmp_path).load(track).lap_time == pytest.approx(event.lap_time)


def test_incomplete_laps_never_touch_the_record(track, lap, tmp_path):
    store = RecordStore(tmp_path)
    path = store.save(track, scaled(lap, 1.6))
    before = path.read_bytes()
    racer, trial, pilot = make_trial(track, tmp_path)
    # A fast partial lap, then a reset: the next lap starts from scratch.
    for _ in range(60 * 20):
        racer.step(*pilot.act(racer))
        assert trial.update(racer) is None
    racer.reset()
    trial.begin(racer)
    assert len(trial._samples) == 1
    assert path.read_bytes() == before
    # A lap driven on autopilot is valid for the lap count but never saved.
    event = drive_lap(racer, trial, pilot, manual=False)
    assert not event.record and event.lap_time < trial.record.lap_time
    assert path.read_bytes() == before
    # The lap that began under the autopilot stays void even if you take over mid-lap...
    event = drive_lap(racer, trial, pilot)
    assert not event.record and path.read_bytes() == before
    # ...and the next one, driven by hand from the line, counts.
    event = drive_lap(racer, trial, pilot)
    assert event.record
    assert path.read_bytes() != before


def test_a_lap_without_a_full_lap_of_net_progress_is_rejected(track, tmp_path):
    racer, trial, _ = make_trial(track, tmp_path)
    for _ in range(30):
        racer.step(0.0, 1.0)
        trial.update(racer)
    # A fake lap count (e.g. a bug in the lap logic) must not be enough to set a record.
    racer.progress.laps = 1
    racer.lap_times.append(10.0)
    event = trial.update(racer)
    assert event is not None and not event.record
    assert trial.record is None and not RecordStore(tmp_path).path_for(track).exists()


def test_going_back_and_forth_over_the_line_never_sets_a_record(track, tmp_path):
    racer, trial, _ = make_trial(track, tmp_path)
    for _ in range(60 * 5):
        racer.step(0.0, 1.0)
        trial.update(racer)
    for _ in range(60 * 5):
        racer.step(0.0, -1.0)
        trial.update(racer)
    assert racer.laps == 0 and trial.record is None


def test_drifting_laps_are_recorded_too(track, tmp_path):
    racer, trial, pilot = make_trial(track, tmp_path)
    drift = DriftState()
    steps = 0
    while True:
        steer, throttle = pilot.act(racer)
        drift.hold = steer > 0.4 and steps % 300 < 150
        racer.step(steer, throttle, drift=drift)
        steps += 1
        event = trial.update(racer)
        if event:
            break
        assert steps < 60 * 400
    assert event.record


# -- ghost and gap ---------------------------------------------------------------------------


def test_ghost_interpolation_is_continuous(lap):
    times = np.arange(0.0, lap.lap_time + 1.0, DT / 3)
    poses = [lap.pose_at(t) for t in times]
    xs, ys, hs = (np.array(c) for c in zip(*poses, strict=True))
    step = np.hypot(np.diff(xs), np.diff(ys))
    assert step.max() < 0.35 * KartParams().max_speed * DT / 3 * 3  # no teleports
    assert np.abs(np.diff(hs)).max() < 0.1
    # Exact at the samples and clamped outside the lap.
    k = len(lap) // 2
    assert lap.pose_at(lap.t[k]) == pytest.approx((lap.x[k], lap.y[k], lap.heading[k]))
    assert lap.pose_at(-5.0)[:2] == (lap.x[0], lap.y[0])
    assert lap.pose_at(lap.lap_time + 9.0)[:2] == (lap.x[-1], lap.y[-1])
    # A midpoint is between its neighbours.
    mid = lap.pose_at((lap.t[k] + lap.t[k + 1]) / 2)
    assert min(lap.x[k], lap.x[k + 1]) - 1e-9 <= mid[0] <= max(lap.x[k], lap.x[k + 1]) + 1e-9


def test_ghost_follows_the_lap_clock_and_waits_then_hides(track, lap, tmp_path):
    RecordStore(tmp_path).save(track, lap)
    trial = TimeTrial(track, RecordStore(tmp_path))
    start = trial.ghost_pose(0.0)
    end = trial.ghost_pose(lap.lap_time)
    assert math.hypot(start[0] - end[0], start[1] - end[1]) < 3.0, "a lap ends where it starts"
    assert trial.ghost_pose(lap.lap_time + 0.5) is not None
    assert trial.ghost_pose(lap.lap_time + 5.0) is None
    assert TimeTrial(track, RecordStore(tmp_path / "empty")).ghost_pose(1.0) is None


def test_gap_is_zero_replaying_the_same_lap(track, lap, tmp_path):
    RecordStore(tmp_path).save(track, lap)
    racer, trial, pilot = make_trial(track, tmp_path)
    worst = 0.0
    for step in range(int(lap.lap_time / DT) - 5):
        racer.step(*pilot.act(racer))
        trial.update(racer)
        if step > 30:
            worst = max(worst, abs(trial.delta), abs(trial.raw_delta))
    assert worst < 0.05, f"gap up to {worst:.3f} s on an identical lap"


def test_gap_is_negative_when_faster_and_positive_when_slower(track, lap, tmp_path):
    store = RecordStore(tmp_path)
    store.save(track, scaled(lap, 1.1))
    racer, trial, pilot = make_trial(track, tmp_path)
    for _ in range(60 * 30):
        racer.step(*pilot.act(racer))
        trial.update(racer)
    assert trial.delta < -1.0, "30 s into a lap against a record 10 % slower"
    store.save(track, scaled(lap, 0.9))
    racer, trial, pilot = make_trial(track, tmp_path)
    for _ in range(60 * 30):
        racer.step(*pilot.act(racer))
        trial.update(racer)
    assert trial.delta > 1.0


def test_gap_is_by_progress_not_by_clock(track, lap, tmp_path):
    """Standing still adds exactly the time lost, and the gap stays put when you catch up."""
    RecordStore(tmp_path).save(track, lap)
    racer, trial, _ = make_trial(track, tmp_path)
    for _ in range(120):  # two seconds of waiting at the line
        racer.step(0.0, 0.0)
        trial.update(racer)
    assert trial.raw_delta == pytest.approx(racer.lap_time - lap.time_at(0.0), abs=0.05)
    assert trial.raw_delta > 1.9


def test_gap_is_smoothed(track, lap, tmp_path):
    RecordStore(tmp_path).save(track, lap)
    racer, trial, pilot = make_trial(track, tmp_path)
    trial.raw_delta = None
    deltas = []
    for _ in range(600):
        racer.step(*pilot.act(racer))
        trial.update(racer)
        deltas.append(trial.delta)
    jumps = np.abs(np.diff(deltas))
    assert jumps.max() < 0.02, "the displayed gap moves by hundredths per step at most"


def test_time_at_distance_survives_standing_and_reversing(lap):
    d = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 1.5, 1.0, 2.5, 3.0])
    t = np.arange(len(d)) * 0.5
    record = fresh_copy(lap, t=t, x=np.zeros(9), y=np.zeros(9), heading=np.zeros(9), d=d,
                        lap_time=4.0)  # fmt: skip
    assert record.time_at(0.0) == 0.0
    assert record.time_at(2.0) == pytest.approx(2.0)  # first time it got there
    assert record.time_at(2.5) == pytest.approx(3.5)
    assert record.time_at(99.0) == pytest.approx(4.0)  # clamped
    assert record.time_at(-1.0) == 0.0


# -- physics fingerprint --------------------------------------------------------------------


def run_fingerprint(drifting: bool) -> tuple[float, ...]:
    kart = KartState(0.0, 0.0, 0.0)
    drift = DriftState() if drifting else None
    for step in range(600):
        steer = 0.6 if 100 <= step < 400 else -0.2
        if drift is not None:
            drift.hold = 150 <= step < 330
        step_kart(kart, steer, 1.0 if step < 500 else 0.0, False, KartParams(), drift=drift)
    return tuple(round(v, 5) for v in (kart.x, kart.y, kart.heading, kart.vx, kart.vy))


@pytest.mark.parametrize("name, drifting", [("plain", False), ("drift", True)])
def test_physics_fingerprint(name, drifting):
    """If this fails, the physics changed: bump PHYSICS_VERSION (sim/version.py), then
    update PINNED_PHYSICS_VERSION and PINNED_FINGERPRINT here, so old records are ignored."""
    assert PHYSICS_VERSION == PINNED_PHYSICS_VERSION
    assert run_fingerprint(drifting) == pytest.approx(PINNED_FINGERPRINT[name], abs=1e-4)


# -- the drive.py application -------------------------------------------------------------------


@pytest.fixture
def app_factory(tmp_path):
    apps = []

    def make(**kwargs):
        kwargs.setdefault("scale", 2)
        kwargs.setdefault("realtime", False)
        kwargs.setdefault("time_trial", True)
        kwargs.setdefault("records_dir", tmp_path)
        app = DriveApp(showcase_track("harbor"), **kwargs)
        apps.append(app)
        return app

    yield make
    pygame.quit()


def test_drive_time_trial_headless_smoke_with_ghost(track, lap, tmp_path, app_factory):
    RecordStore(tmp_path).save(track, lap)
    app = app_factory(autopilot=True)
    assert app.trial is not None and app.trial.record is not None
    app.run(max_frames=240)
    assert app.ghost_visual is not None and app.ghost_visual.ghost
    assert app.ghost_visual.kart.x == pytest.approx(app.trial.record.pose_at(app.racer.lap_time)[0])
    assert app.hud_trial.record == lap.lap_time and app.hud_trial.delta is not None
    app.handle_key(pygame.K_f)  # ghost off
    app.run(max_frames=2)
    assert app.ghost_visual is None
    app.handle_key(pygame.K_f)
    app.handle_key(pygame.K_t)  # top-down view also works
    app.run(max_frames=5)
    app.handle_key(pygame.K_k)  # leave the time trial
    assert app.trial is None
    app.run(max_frames=5)


def test_drive_reset_and_track_change_restart_the_recording(track, lap, tmp_path, app_factory):
    RecordStore(tmp_path).save(track, lap)
    app = app_factory(autopilot=True)
    app.run(max_frames=100)
    assert len(app.trial._samples) > 10
    app.handle_key(pygame.K_r)
    assert len(app.trial._samples) == 1
    app.handle_key(pygame.K_2)
    assert app.trial.record is None and app.track.name == "Cañón"
    app.handle_key(pygame.K_1)
    assert app.trial.record is not None


def test_drive_tab_autopilot_invalidates_the_lap(app_factory):
    app = app_factory(autopilot=False)
    app.run(max_frames=5)
    assert app.trial._valid
    app.handle_key(pygame.K_TAB)
    app.run(max_frames=2)
    assert not app.trial._valid


def test_drive_main_time_trial_screenshot(track, lap, tmp_path):
    RecordStore(tmp_path).save(track, lap)
    shot = tmp_path / "shot.png"
    main([
        "--time-trial", "--records-dir", str(tmp_path), "--autopilot", "--frames", "150",
        "--scale", "2", "--screenshot", str(shot),
    ])  # fmt: skip
    assert shot.stat().st_size > 1000
    assert os.environ["SDL_VIDEODRIVER"] == "dummy"


def test_drive_without_time_trial_touches_no_records(tmp_path, app_factory):
    app = app_factory(time_trial=False)
    app.run(max_frames=10)
    assert app.trial is None and app.ghost_visual is None and app.hud_trial is None
    assert not list(Path(tmp_path).iterdir())


@pytest.mark.parametrize(
    "delta, text",
    [(-0.42, "−0,42"), (1.034, "+1,03"), (0.0, "+0,00"), (-0.004, "+0,00"), (-12.5, "−12,50")],
)
def test_format_delta(delta, text):
    assert format_delta(delta) == text


def test_drive_manual_lap_saves_the_record_and_announces_it(
    track, tmp_path, app_factory, monkeypatch
):
    app = app_factory(autopilot=False)
    pilot = Autopilot(track)
    monkeypatch.setattr(
        "kartlab.show.drive.keyboard_controls", lambda pressed: pilot.act(app.racer)
    )
    for _ in range(60 * 120):
        app.physics_step()
        if app.trial.record is not None:
            break
    assert app.trial.record is not None
    assert app._message[0].startswith("¡Primer récord!")
    assert RecordStore(tmp_path).load(track).lap_time == app.trial.record.lap_time
    app.run(max_frames=3)  # and the HUD carries it
    assert app.hud_trial.record == app.trial.record.lap_time
    assert app.hud_trial.last == app.trial.last_lap


@pytest.mark.parametrize(
    "event, start",
    [
        (LapEvent(60.0, True, 61.5), "¡NUEVO RÉCORD! 1:00.00 (−1,50)"),
        (LapEvent(62.0, False, 61.5), "Vuelta 1:02.00 (+0,50)"),
        (LapEvent(62.0, False, 61.5, valid=False), "Vuelta 1:02.00 · no cuenta"),
        (LapEvent(60.0, True, 61.5, saved=False), "¡NUEVO RÉCORD! 1:00.00 (−1,50) · no se pudo"),
    ],
)
def test_drive_lap_messages(app_factory, event, start):
    app = app_factory()
    app._on_lap(event)
    assert app._message[0].startswith(start)


@pytest.mark.parametrize("field", ["lap_time", "x", "width", "points"])
def test_huge_integers_do_not_break_startup(track, lap, tmp_path, field):
    """JSON integers have no limit; converting 10**400 to a float must not crash the load."""
    store = RecordStore(tmp_path)
    data = json.loads(lap.to_json())
    huge = 10**400
    if field == "lap_time":
        data["lap_time"] = huge
    elif field == "x":
        data["x"][5] = huge
    elif field == "width":
        data["track"]["width"] = huge
    else:
        data["track"]["points"][3][0] = huge
    store.path_for(track).write_text(json.dumps(data), encoding="utf-8", newline="\n")
    with pytest.warns(RecordWarning):
        assert TimeTrial(track, store).record is None
