"""The race format of the drive.py time trial: countdown, three laps and the results."""

import pygame
import pytest

from kartlab.show.drive import DriveApp, main
from kartlab.show.trialrace import COUNTDOWN_FRAMES, TRIAL_LAPS, TrialSession
from kartlab.sim.autopilot import Autopilot
from kartlab.sim.showcase import showcase_track
from kartlab.sim.timetrial import RecordStore

# -- the session on its own --------------------------------------------------------------------


def test_countdown_then_racing_then_finished():
    session = TrialSession()
    states = [session.advance() for _ in range(COUNTDOWN_FRAMES)]
    assert set(states) == {"countdown"} and session.time == 0.0
    assert session.advance() == "racing" and session.time > 0.0
    for lap in (40.0, 41.0):
        session.on_lap(lap, False, True)
    assert session.banner[0] == "ÚLTIMA VUELTA" and session.lap_number == TRIAL_LAPS
    session.on_lap(39.5, False, True)
    assert session.finished and session.banner[0] == "¡META!"
    assert session.advance() == "finished"
    session.on_lap(30.0, True, True)  # nothing counts after the finish
    assert session.laps_done == 3 and session.total == pytest.approx(120.5)


def test_results_say_every_lap_the_total_and_the_record():
    session = TrialSession()
    for lap in (40.0, 38.5, 39.0):
        session.on_lap(lap, lap == 38.5, True)
    rows = session.result_lines(38.5)
    texts = [text for text, _ in rows]
    assert texts[0].startswith("Vuelta 1") and "0:38.50" in texts[1] and "mejor" in texts[1]
    assert any("Total" in text and "1:57.50" in text for text in texts)
    assert rows[-1][1] == "record" and "NUEVO RÉCORD" in rows[-1][0]
    lost = TrialSession()
    for lap in (40.0, 41.0, 42.0):
        lost.on_lap(lap, False, True)
    assert "Récord de vuelta 0:39.00" in lost.result_lines(39.0)[-1][0]
    assert "Sin récord" in lost.result_lines(None)[-1][0]
    piloted = TrialSession()
    for lap in (40.0, 41.0, 42.0):
        piloted.on_lap(lap, False, False)
    assert "piloto automático" in piloted.result_lines(None)[-1][0]


def test_a_single_lap_session_has_no_last_lap_banner():
    session = TrialSession(laps=1)
    session.on_lap(30.0, False, True)
    assert session.banner[0] == "¡META!"


# -- in drive.py -------------------------------------------------------------------------------


@pytest.fixture
def make(tmp_path):
    apps = []

    def build(**kwargs):
        kwargs.setdefault("scale", 2)
        kwargs.setdefault("realtime", False)
        kwargs.setdefault("time_trial", True)
        kwargs.setdefault("records_dir", tmp_path)
        app = DriveApp(showcase_track("harbor"), **kwargs)
        apps.append(app)
        return app

    yield build
    pygame.quit()


def test_the_controls_are_frozen_during_the_countdown(make, monkeypatch):
    app = make()
    monkeypatch.setattr("kartlab.show.drive.keyboard_controls", lambda pressed: (1.0, 1.0))
    app.run(max_frames=COUNTDOWN_FRAMES - 5)
    assert app.session.counting and app.racer.time == 0.0 and app.racer.kart.vx == 0.0
    app.run(max_frames=60)
    assert not app.session.counting and app.racer.time > 0.0


def test_free_driving_and_a_trial_without_the_format_have_no_countdown(make):
    free = make(time_trial=False)
    free.run(max_frames=5)
    assert free.session is None and free.racer.time > 0.0
    plain = make(race_format=False)
    plain.run(max_frames=5)
    assert plain.session is None and plain.racer.time > 0.0
    toggled = make(time_trial=False)
    toggled.handle_key(pygame.K_k)
    assert toggled.session is not None
    toggled.handle_key(pygame.K_k)
    assert toggled.session is None


def test_three_laps_with_the_pilot_end_with_results_and_enter_repeats(make, monkeypatch):
    track = showcase_track("harbor")
    app = make()
    pilot = Autopilot(track)
    monkeypatch.setattr(
        "kartlab.show.drive.keyboard_controls", lambda pressed: pilot.act(app.racer)
    )
    seen_last = False
    for _ in range(60 * 400):
        app.physics_step()
        seen_last = seen_last or (app.session.banner or ("",))[0] == "ÚLTIMA VUELTA"
        if app.session.finished:
            break
    assert seen_last and app.session.finished and app.session.laps_done == 3
    assert app.trial.record is not None  # the best lap is still saved as always
    assert app.session.new_record
    app.run(max_frames=60 * 4)  # banner, then the table
    assert app.session.results_age() > 0.0
    kart = app.racer.kart
    laps_before = app.racer.laps
    app.run(max_frames=30)
    assert app.racer.laps == laps_before and app.session.laps_done == 3  # frozen at the end
    assert abs(kart.forward_speed) < 40.0
    app.handle_key(pygame.K_RETURN)
    assert app.session.counting and app.session.laps_done == 0 and app.racer.time == 0.0
    assert app.trial.record is not None  # the record survives a new session


def test_the_autopilot_makes_the_session_say_no_record(make, monkeypatch):
    app = make(autopilot=True)
    for _ in range(60 * 400):
        app.physics_step()
        if app.session.finished:
            break
    assert app.session.finished and not app.session.valid and not app.session.new_record
    assert app.trial.record is None


def test_the_ghost_still_follows_the_record_during_the_session(make, tmp_path):
    from kartlab.sim.race import Racer
    from kartlab.sim.timetrial import TimeTrial

    track = showcase_track("harbor")
    racer = Racer(track)
    trial = TimeTrial(track, RecordStore(tmp_path))
    trial.begin(racer)
    pilot = Autopilot(track)
    for _ in range(60 * 200):
        racer.step(*pilot.act(racer))
        event = trial.update(racer, True)
        if event is not None:
            break
    app = make(autopilot=True)
    assert app.trial.record is not None
    app.run(max_frames=COUNTDOWN_FRAMES + 60)
    assert app.ghost_visual is not None


def test_screenshots_of_the_session_headless(tmp_path):
    shot = tmp_path / "trial.png"
    main(
        [
            "--time-trial", "--records-dir", str(tmp_path), "--frames", "100", "--scale", "2",
            "--screenshot", str(shot),
        ]
    )  # fmt: skip
    assert shot.stat().st_size > 1000
