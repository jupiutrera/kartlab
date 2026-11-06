"""The race against the AI: standings, the HUD, the driver changes and the scene, headless."""

import os
from pathlib import Path

import numpy as np
import pygame
import pytest

from kartlab.render import bigtext, pixelfont
from kartlab.render.bigtext import POP_SECONDS, place_look, pop_extra
from kartlab.render.palette import KART_RAMPS
from kartlab.show.drivers import PolicyDriver, Version
from kartlab.show.humanrace import (
    DIFFICULTIES,
    Entrant,
    classify,
    difficulty_of,
    place_of,
    result_lines,
)
from kartlab.show.racehud import RaceHud
from kartlab.show.scenes import HumanRaceScene, grid_slot
from kartlab.show.viewer import MODE_ORDER, MODES, ViewerApp, main
from kartlab.sim.kart import KartParams, cruise_speed
from kartlab.sim.race import Racer
from kartlab.sim.showcase import showcase_track
from kartlab.sim.trackfile import load_track_file
from kartlab.train.config import PPOParams, TrainConfig
from kartlab.train.run import REPO_ROOT, RunPaths

# -- standings and difficulty -----------------------------------------------------------------


def entrant(name="IA", human=False, laps=3, length=1000.0):
    return Entrant(name, 0, human, length, laps)


def test_entrant_counts_laps_and_finishes_once():
    e = entrant(laps=2)
    assert e.update(-4.0, 0.0) == 0 and e.laps_done == 0
    assert e.update(999.0, 40.0) == 0
    assert e.update(1001.0, 41.0) == 1 and e.laps_done == 1 and not e.finished
    assert e.update(900.0, 45.0) == 0  # going back and forth never adds laps
    assert e.update(1005.0, 46.0) == 0
    assert e.update(2001.0, 83.0) == 1 and e.finished and e.finish_time == 83.0
    assert e.update(3500.0, 120.0) == 0 and e.laps_done == 2  # never more than the race
    assert e.lap_times == [41.0, 42.0] and e.best_lap == 41.0
    assert e.lap_time_at(100.0) == pytest.approx(42.0)  # finished: the clock stays on the last lap
    assert e.lap_time_at(500.0) == pytest.approx(42.0)
    running = entrant(laps=2)
    running.update(1001.0, 41.0)
    assert running.lap_time_at(58.0) == pytest.approx(17.0)


def test_classification_puts_finishers_first_then_distance():
    slow, fast, far, near = entrant("a"), entrant("b"), entrant("c"), entrant("d")
    slow.update(3000.0, 100.0)
    fast.update(3000.0, 90.0)
    far.update(2500.0, 100.0)
    near.update(400.0, 100.0)
    order = classify([near, far, slow, fast])
    assert [e.name for e in order] == ["b", "a", "c", "d"]
    assert place_of([near, far, slow, fast], near) == 4
    rows = result_lines([near, far, slow, fast])
    assert rows[0][0].startswith("1º  b") and "faltan 500 m" in rows[2][0]


def test_difficulty_levels_are_ordered_and_the_champion_is_unlimited():
    speeds = [level.speed for level in DIFFICULTIES]
    assert speeds == sorted(speeds) and len(set(speeds)) == 4
    assert difficulty_of("campeon").speed == 1.0
    with pytest.raises(KeyError):
        difficulty_of("imposible")


# -- the HUD ----------------------------------------------------------------------------------


def test_every_hud_text_has_glyphs_and_a_solid_mask():
    for text in ("1º", "VUELTA 3/3", "ÚLTIMA VUELTA", "¡META!", "¡YA!", "3", "2", "1", "/4", "ñ"):
        assert all(pixelfont.has_glyph(c) for c in text)
        assert bigtext._grid(text, True).any()


def test_big_text_is_whole_cells_and_cached():
    pygame.init()
    small = bigtext.render("1", 3, place_look(1))
    big = bigtext.render("1", 6, place_look(1))
    assert big.get_width() == small.get_width() * 2  # an exact multiple: no smoothing
    assert bigtext.render("1", 3, place_look(1)) is small
    colours = {tuple(c) for c in pygame.surfarray.array3d(big).reshape(-1, 3)}
    assert len(colours) <= 6  # the key, the outline and three bands: flat colours, no blur
    assert place_look(1) != place_look(2) != place_look(4) and place_look(7) == place_look(4)


def test_big_text_is_the_interface_font_with_a_bigger_cell():
    """The big digits are the same letters as the small ones: the lit cells of the bold
    5x7 grid, each one a block."""
    grid = bigtext._grid("3", False)
    assert np.array_equal(grid, pixelfont.text_mask("3", 1)[pixelfont.CAP_TOP :])


def test_pop_jumps_in_whole_cells_and_ends():
    assert pop_extra(-1.0) == 0 and pop_extra(POP_SECONDS) == 0 and pop_extra(5.0) == 0
    sizes = [pop_extra(i * 0.01) for i in range(int(POP_SECONDS * 100))]
    assert sizes[0] == max(sizes) > 0 and sizes == sorted(sizes, reverse=True)


def test_race_hud_draws_every_piece_without_errors(tiny_app):
    app = tiny_app(mode="vsai")
    app.run_loop(10)
    hud = RaceHud(app.ov)
    for place in (1, 2, 4):
        hud.position(place, 4, 0.05)
        hud.position(place, 4, 5.0)
    hud.lap_counter(2, 3, "1:02,50", "0:31,20", "0:30,90")
    hud.lap_counter(1, 3, "0:05,00", "0:05,00")
    for frame in (0, 60, 120, 179, 180, 200, 400):
        hud.countdown(frame, 180, 60)
    hud.banner("ÚLTIMA VUELTA", 0.1)
    hud.banner("¡META!", 2.2)
    hud.banner("¡META!", 9.0)  # long gone: draws nothing
    lines = [("1º  Tú  1:00,00  mejor 0:20,00", 9), ("Sin récord", 8)]
    hud.results("Resultados", 1, lines, "Intro", 1.0)
    hud.results("Resultados", None, lines, "Intro", 0.0)


# -- the driver -------------------------------------------------------------------------------


def stub_version() -> Version:
    """A policy that floors the throttle straight ahead."""
    return Version(0, lambda obs: np.tile([0.0, 1.0], (len(obs), 1)))


@pytest.fixture
def driver():
    from kartlab.render.topdown import KART_COLOR

    track = showcase_track("harbor")
    config = TrainConfig(name="t", total_steps=1, env={"track": "harbor"}).env_config()
    return PolicyDriver(stub_version(), track, config, KART_COLOR)


def test_restart_here_keeps_the_race_distance_and_recentres_the_kart(driver):
    track = driver.env.track
    driver.place(*grid_slot(1))
    assert driver.race_distance == pytest.approx(grid_slot(1)[0])  # behind the line: negative
    # Lap two: pretend the kart got there and ended up crooked on the edge of the track.
    driver.racer.progress.distance = track.length * 1.5
    x, y, _ = track.pose_at(track.length * 0.5, track.half_width * 0.9)
    driver.racer.displace(x - driver.racer.kart.x, y - driver.racer.kart.y)
    driver.racer.kart.vx, driver.racer.kart.vy = 8.0, -3.0
    before = driver.race_distance
    laps, time = driver.racer.laps, driver.racer.time
    driver.restart_here()
    kart = driver.racer.kart
    assert driver.race_distance == pytest.approx(before, abs=1.0)  # it never goes back
    assert (kart.vx, kart.vy) == (0.0, 0.0)
    assert abs(driver.racer.position.lateral) < 0.2
    assert driver.racer.laps == laps and driver.racer.time == time
    assert driver.no_contact_until > driver.racer.time


def test_stuck_driver_restarts_in_place_not_at_the_line(driver):
    track = driver.env.track
    driver.restart_in_place = True
    driver.place(*grid_slot(0))
    driver.env.hold_action(np.zeros(2))
    # Standing still for longer than RESTART_SECONDS: stuck.
    steady = Version(0, lambda obs: np.zeros((len(obs), 2)))
    driver.version = steady
    s_before = driver.racer.position.s
    events = [driver.step() for _ in range(60 * 4)]
    assert "restart" in events
    assert abs(driver.racer.position.s - s_before) < 3.0  # still where it was, not on the line
    assert driver.race_distance < 0.0 < track.length


def test_contact_pauses_the_stuck_timer(driver):
    driver.restart_in_place = True
    driver.version = Version(0, lambda obs: np.zeros((len(obs), 2)))
    driver.place(*grid_slot(0))
    events = []
    for _ in range(60 * 5):
        driver.touch()  # somebody keeps pushing it
        events.append(driver.step())
    assert "restart" not in events


def test_contact_cannot_freeze_the_timers_for_ever(driver):
    driver.restart_in_place = True
    driver.version = Version(0, lambda obs: np.zeros((len(obs), 2)))
    driver.place(*grid_slot(0))
    events = []
    for _ in range(60 * 9):
        driver.touch()  # a human glued to it for nine seconds
        events.append(driver.step())
    assert "restart" in events  # frozen at most 3 s, then the usual 3 s


def test_an_ai_on_the_grass_with_a_human_pressed_against_it_still_comes_back(driver):
    track = driver.env.track
    driver.restart_in_place = True
    driver.version = Version(0, lambda obs: np.zeros((len(obs), 2)))
    driver.place(*grid_slot(0))
    x, y, _ = track.pose_at(driver.racer.position.s, track.half_width + 4.0)
    driver.racer.displace(x - driver.racer.kart.x, y - driver.racer.kart.y)
    assert driver.racer.on_grass
    first = None
    for i in range(60 * 6):
        driver.touch()
        if driver.step() == "restart":
            first = i / 60
            break
    assert first is not None and first < 4.5  # the contact does not hold the timers on grass
    assert not driver.racer.on_grass


def test_speed_limiter_caps_the_speed_but_not_a_full_speed_driver(driver):
    track = showcase_track("harbor")
    driver.place(0.0, 0.0)
    cap = 0.5 * cruise_speed(KartParams())
    driver.speed_cap = cap
    for _ in range(60 * 8):
        driver.step()
        if abs(driver.racer.position.lateral) > track.half_width:
            break
    assert driver.racer.speed <= cap + 0.5
    assert driver.racer.speed > cap - 3.0  # the pedal is still down: it drives at the cap


# -- the scene --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    from kartlab.train.train import train

    config = TrainConfig(
        name="vsai-test",
        total_steps=768,
        n_envs=2,
        vec_env="dummy",
        low_priority=False,
        ppo=PPOParams(n_steps=64, batch_size=64, n_epochs=1),
        checkpoint_every=256,
        checkpoint_at=(0,),
        eval_tracks=(),
        env={"track": "harbor"},
    )
    run = train(config, runs_dir=tmp_path_factory.mktemp("runs"), verbose=0)
    final = run.list_checkpoints()[-1]
    run.best_model.write_bytes(final.model.read_bytes())
    return run


@pytest.fixture
def tiny_app(tiny_run):
    created = []

    def make(**kwargs):
        kwargs.setdefault("scale", 2)
        viewer = ViewerApp(tiny_run, realtime=False, **kwargs)
        created.append(viewer)
        return viewer

    yield make
    for viewer in created:
        viewer.close()
    pygame.quit()


def test_the_mode_is_reachable_by_name_but_not_from_the_title(tiny_app):
    assert "vsai" in MODES and "vsai" not in MODE_ORDER
    app = tiny_app(mode="vsai")
    assert isinstance(app.scene, HumanRaceScene) and app.state == "scene"
    before = app.scene
    app.handle_key(pygame.K_m)  # M cycles the title modes: not from here
    app.run_loop(2)
    assert app.scene is before


@pytest.mark.smoke
def test_countdown_freezes_everybody_then_the_race_starts(tiny_app):
    app = tiny_app(mode="vsai", rivals=3)
    scene = app.scene
    app.run_loop(120)  # two of the three seconds
    assert scene.started and scene.race_steps == 0
    karts = [scene.human.kart] + [d.racer.kart for d in scene.drivers]
    assert all(k.vx == 0.0 and k.vy == 0.0 for k in karts)
    assert len(scene.drivers) == 3 and len(scene.entrants) == 4
    assert scene.human.time == 0.0
    app.run_loop(100)
    assert scene.race_steps > 0 and scene.human.time > 0.0
    # The human starts on the last grid slot, the AI ahead.
    assert scene.me.name == "Tú"
    assert all(e.distance > scene.me.distance - 20.0 for e in scene.entrants[1:])


def test_race_hud_replaces_the_lap_panel_and_the_mode_panel_hides_after_the_countdown(tiny_app):
    app = tiny_app(mode="vsai")
    scene = app.scene
    app.run_loop(100)  # counting down
    info = scene.frame()[2]
    assert not info.lap_panel and info.label_panel
    app.run_loop(100)  # racing
    assert not scene.frame()[2].label_panel
    app.handle_key(pygame.K_h)  # H brings it back (with the help)
    assert scene.frame()[2].label_panel
    app.handle_key(pygame.K_h)
    assert not scene.frame()[2].label_panel
    app.handle_key(pygame.K_p)  # and so does the pause
    assert scene.frame()[2].label_panel


def test_grid_is_two_by_two_with_the_human_last():
    first, second, third = grid_slot(0), grid_slot(1), grid_slot(2)
    assert first[0] == second[0] > third[0] and first[1] == -second[1]


def test_rivals_and_the_player_get_different_colours(tiny_app):
    app = tiny_app(mode="vsai", rivals=3, kart_ramp=3)
    app.run_loop(5)
    ramps = [e.ramp for e in app.scene.entrants]
    assert ramps[0] == 3 and len(set(ramps)) == 4 and max(ramps) < len(KART_RAMPS)


def test_a_still_human_finishes_last(tiny_app):
    app = tiny_app(mode="vsai", vs_laps=1, difficulty="campeon")
    scene = app.scene
    app.run_loop(60 * 3 + 60 * 12)
    assert scene.place == len(scene.entrants)


def test_the_human_on_autopilot_beats_the_easy_ai_and_gets_the_results(tiny_app):
    app = tiny_app(mode="vsai", vs_laps=1, difficulty="facil")
    scene = app.scene
    app.run_loop(5)
    scene.autopilot_on = True
    for _ in range(60 * 150):
        app.run_loop(60)
        if scene.ended_at is not None:
            break
    assert scene.me.finished and scene.final_place == 1
    app.run_loop(60 * 4)  # the banner, then the table
    assert scene.final[0][1] is True and scene.final[0][0].startswith("1º  Tú")
    app.handle_key(pygame.K_RETURN)  # Intro: race again
    app.run_loop(3)
    assert scene.started and scene.race_steps < 5 and scene.ended_at is None


def test_the_last_lap_is_announced_and_laps_are_counted_in_the_hud(tiny_app):
    app = tiny_app(mode="vsai", vs_laps=2, difficulty="facil")
    scene = app.scene
    app.run_loop(5)
    scene.autopilot_on = True
    seen_last = False
    for _ in range(60 * 90):
        app.run_loop(30)
        if scene.banner and scene.banner[0] == "ÚLTIMA VUELTA":
            seen_last = True
            break
    assert seen_last and scene.me.laps_done == 1


def test_collisions_can_be_turned_off(tiny_app, monkeypatch):
    for on in (True, False):
        app = tiny_app(mode="vsai", collisions=on)
        scene = app.scene
        calls = []
        monkeypatch.setattr(scene, "_bump", lambda t, calls=calls: calls.append(t))
        app.run_loop(60 * 3 + 30)
        assert bool(calls) is on


def test_karts_pass_through_each_other_without_collisions_and_bump_with_them(tiny_app):
    pushed = {}
    for on in (True, False):
        app = tiny_app(mode="vsai", collisions=on)
        scene = app.scene
        app.run_loop(60 * 3 + 5)
        scene.race_steps = 60 * 10  # past the grace of the start
        ai = scene.drivers[0].racer
        human = scene.human
        human.displace(ai.kart.x - human.kart.x, ai.kart.y - human.kart.y + 0.8)
        human.kart.vx = human.kart.vy = 0.0
        gap = np.hypot(human.kart.x - ai.kart.x, human.kart.y - ai.kart.y)
        scene._bump(scene.race_time) if on else None
        pushed[on] = np.hypot(human.kart.x - ai.kart.x, human.kart.y - ai.kart.y) - gap
    assert pushed[True] > 0.0  # pushed apart
    assert pushed[False] == 0.0


def hold_keys(monkeypatch, *keys):
    from collections import defaultdict

    pressed = defaultdict(bool, dict.fromkeys(keys, True))
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: pressed)
    return pressed


def test_shift_held_through_the_countdown_starts_no_sweep_until_released(tiny_app, monkeypatch):
    app = tiny_app(mode="vsai")
    scene = app.scene
    hold_keys(monkeypatch, pygame.K_LSHIFT, pygame.K_UP)
    app.run_loop(60 * 3 - 2)  # the countdown, with Shift down all the time
    assert scene.race_steps == 0 and scene.human.kart.vx == 0.0
    app.run_loop(90)  # the race is on, straight, Shift still down
    assert scene.human.kart.forward_speed > scene.drift.params.entry_speed
    hold_keys(monkeypatch, pygame.K_LSHIFT, pygame.K_UP, pygame.K_LEFT)  # now the wheel
    app.run_loop(10)
    assert not scene.drift.active  # Shift was never released: no sweep
    hold_keys(monkeypatch, pygame.K_UP, pygame.K_LEFT)  # let go and press again
    app.run_loop(3)
    hold_keys(monkeypatch, pygame.K_LSHIFT, pygame.K_UP, pygame.K_LEFT)
    app.run_loop(2)
    assert scene.drift.active  # the sweep starts at once


def test_a_forced_restart_in_lap_two_never_lowers_the_standings(tiny_app):
    app = tiny_app(mode="vsai", vs_laps=3, rivals=2)
    scene = app.scene
    app.run_loop(60 * 3 + 30)
    length = app.track.length
    ai = scene.drivers[0]
    ai.racer.progress.distance = length * 1.3  # lap two
    app.run_loop(1)
    order = [e.name for e in classify(scene.entrants)]
    distance = scene.entrants[1].distance
    assert order[0] == "IA 1" and distance > length
    ai.restart_here()
    app.run_loop(1)
    assert [e.name for e in classify(scene.entrants)] == order
    assert scene.entrants[1].distance >= distance - 2.0


def test_crossing_tracks_are_refused(tiny_run):
    ocho = Path(__file__).resolve().parents[1] / "tracks" / "ocho.json"
    assert load_track_file(ocho).crossings
    with pytest.raises(SystemExit, match="cruces"):
        main(
            [
                "--run",
                str(tiny_run.root),
                "--mode",
                "vsai",
                "--track-file",
                str(ocho),
                "--frames",
                "1",
            ]
        )


def test_cli_runs_the_mode_with_its_options(tiny_run, tmp_path):
    shot = tmp_path / "vsai.png"
    main(
        [
            "--run", str(tiny_run.root), "--mode", "vsai", "--scale", "2", "--frames", "40",
            "--rivals", "1", "--laps", "1", "--difficulty", "dificil", "--no-collisions",
            "--color", "2", "--screenshot", str(shot),
        ]
    )  # fmt: skip
    assert shot.exists() and shot.stat().st_size > 0


# -- which run the AI comes from -------------------------------------------------------------------


def make_run(root, tiny_run, name, success=None, best=True):
    import shutil

    folder = root / name
    shutil.copytree(tiny_run.root, folder)
    if not best:
        (folder / "best_model.zip").unlink()
    if success is not None:
        (folder / "eval_best_model.yaml").write_text(
            f"model: best_model.zip\nsuccess_rate: {success}\n", encoding="utf-8", newline="\n"
        )
    return folder


def test_default_opponent_is_the_best_evaluated_generalist(tmp_path, tiny_run):
    from kartlab.show.versions import best_generalist_run

    make_run(tmp_path, tiny_run, "2026-01-01_0000_fixed", success=0.55)
    best = make_run(tmp_path, tiny_run, "2026-01-02_0000_random", success=1.0)
    make_run(tmp_path, tiny_run, "2026-01-03_0000_weak", success=0.33)
    make_run(tmp_path, tiny_run, "2026-01-04_0000_unevaluated")  # newest, never evaluated
    assert best_generalist_run(tmp_path).root == best
    tie = make_run(tmp_path, tiny_run, "2026-01-05_0000_random2", success=1.0)
    assert best_generalist_run(tmp_path).root == tie  # a tie goes to the most recent
    # A run that scored well but has no best_model.zip cannot be raced against.
    make_run(tmp_path, tiny_run, "2026-01-06_0000_nobest", success=1.0, best=False)
    assert best_generalist_run(tmp_path).root == tie


def test_non_finite_success_rates_are_ignored(tmp_path, tiny_run):
    from kartlab.show.versions import best_generalist_run, eval_success_rate

    sane = make_run(tmp_path, tiny_run, "2026-01-01_0000_sane", success=0.5)
    for name, value in (("nan", ".nan"), ("inf", ".inf"), ("minf", "-.inf")):
        make_run(tmp_path, tiny_run, f"2026-01-02_0000_{name}", success=value)
    assert eval_success_rate(RunPaths(tmp_path / "2026-01-02_0000_inf")) is None
    assert best_generalist_run(tmp_path).root == sane  # not the run that says "infinity"


def test_default_opponent_falls_back_to_the_latest_run_when_nothing_was_evaluated(
    tmp_path, tiny_run
):
    from kartlab.show.versions import best_generalist_run

    make_run(tmp_path, tiny_run, "2026-01-01_0000_a")
    newest = make_run(tmp_path, tiny_run, "2026-01-02_0000_b")
    broken = make_run(tmp_path, tiny_run, "2026-01-03_0000_c")
    (broken / "eval_best_model.yaml").write_text("success_rate: [oops", encoding="utf-8")
    assert best_generalist_run(tmp_path).root == broken  # unreadable file: not evaluated
    assert best_generalist_run(tmp_path / "nothing") is None
    assert newest.exists()


def test_the_run_flag_beats_the_default_opponent(tiny_run, tmp_path, monkeypatch):
    import kartlab.show.viewer as viewer

    monkeypatch.setattr(viewer, "best_generalist_run", lambda: pytest.fail("not asked for"))
    shot = tmp_path / "x.png"
    main(
        [
            "--run", str(tiny_run.root), "--mode", "vsai", "--scale", "1", "--frames", "5",
            "--screenshot", str(shot),
        ]
    )  # fmt: skip
    assert shot.exists()


# -- the reference policy, if there is one ------------------------------------------------------


def reference_run() -> RunPaths | None:
    env = os.environ.get("KARTLAB_REFERENCE_RUN")
    if env:
        return RunPaths(Path(env))
    from kartlab.show.versions import best_generalist_run

    return best_generalist_run(REPO_ROOT / "runs")


def test_two_hundred_bumps_do_not_throw_the_reference_policy_off():
    """200 simulated hits by a human kart: the AI stays on the road and finishes the race."""
    run = reference_run()
    if run is None or not run.best_model.exists():
        pytest.skip("no hay run de referencia en runs/ (KARTLAB_REFERENCE_RUN)")
    from kartlab.show.versions import load_best_version

    stats = bump_race(run, load_best_version(run))
    assert stats["laps"] >= 3 and stats["bumps"] >= 200, stats
    assert stats["touching_steps"] >= 200, stats  # the hits really touched
    assert stats["worst_off_per_lap"] <= 6.0, stats


def bump_race(run, version, laps=3, track_name="plateau") -> dict:
    """Race the AI alone for ``laps`` laps (and at least 200 hits) while a second kart hits it
    every 50 steps (from
    the left, from the right and from behind in turn) through ``sim.contact``. Returns the
    laps completed, the bumps made, the AI's restarts and its worst time off the road in one
    lap."""
    from kartlab.render.topdown import KART_COLOR
    from kartlab.sim.contact import Body, PairState, resolve

    track = showcase_track(track_name)
    config = TrainConfig.from_yaml(run.config).env_config()
    ai = PolicyDriver(version, track, config, KART_COLOR)
    ai.restart_in_place = True
    human = Racer(track)
    state = PairState()
    off = [0.0]
    bumps = step = touches = 0
    cycle, contact_steps = 50, 20
    total = track.length * laps
    while (ai.race_distance < total or bumps < 200) and step < 60 * 600:
        phase = step % cycle
        kart = ai.racer.kart
        if phase == 0:  # line up beside or behind the AI and send the kart into it
            kind = (step // cycle) % 3
            c, s = np.cos(kart.heading), np.sin(kart.heading)
            if kind == 2:  # from behind, 7 m/s faster
                dx, dy, vx, vy = -2.2 * c, -2.2 * s, 7.0 * c, 7.0 * s
            else:  # from the side, 5 m/s towards it
                side = 1.0 if kind == 0 else -1.0
                dx, dy = -side * s * 1.15, side * c * 1.15
                vx, vy = side * s * 5.0, -side * c * 5.0
            human.displace(kart.x + dx - human.kart.x, kart.y + dy - human.kart.y)
            human.kart.heading = kart.heading
            human.kart.vx, human.kart.vy = kart.vx + vx, kart.vy + vy
            bumps += 1
        if phase < contact_steps:
            human.step(0.0, 0.0)
            contact = resolve(Body(human.kart), Body(kart), state, 100.0)
            if contact is not None:
                touches += 1
                human.displace(*contact.push_human)
                ai.racer.displace(*contact.push_ai)
                human.kart.vx += contact.dv_human[0]
                human.kart.vy += contact.dv_human[1]
                kart.vx += contact.dv_ai[0]
                kart.vy += contact.dv_ai[1]
                ai.touch()
        else:  # out of the way until the next hit
            human.displace(1000.0 - human.kart.x, 1000.0 - human.kart.y)
            human.kart.vx = human.kart.vy = 0.0
        before = ai.racer.laps
        ai.step()
        if abs(ai.racer.position.lateral) > track.half_width:
            off[-1] += 1 / 60
        if ai.racer.laps > before:
            off.append(0.0)
        step += 1
    return {
        "laps": int(ai.race_distance // track.length),
        "bumps": bumps,
        "touching_steps": touches,
        "restarts": ai.restarts,
        "worst_off_per_lap": max(off),
        "seconds": round(step / 60, 1),
    }


def test_one_lap_is_not_called_laps_in_any_panel(tiny_app):
    from kartlab.render import pixelui as ui
    from kartlab.show.overlays import version_name

    seen = []
    original = ui.text

    def spy(target, value, *args, **kwargs):
        seen.append(value)
        return original(target, value, *args, **kwargs)

    ui.text = spy
    try:
        app = tiny_app(mode="vsai", vs_laps=1)
        app.run_loop(30)
        assert "1 vuelta, Normal" in app.scene.frame()[2].extra
        app = tiny_app(mode="race", laps=1)
        app.run_loop(30)
        assert "1 vuelta" in app.scene.frame()[2].extra
    finally:
        ui.text = original
    assert any("1 vuelta" in line.lower() for line in seen)
    assert not any("1 vueltas" in line.lower() for line in seen)
    assert ui.plural(1, "vuelta", "vueltas") == "1 vuelta"
    assert ui.plural(0, "vuelta", "vueltas") == "0 vueltas"
    assert ui.plural(3, "vuelta", "vueltas") == "3 vueltas"
    assert version_name(1) == "1 paso" and version_name(300_000) == "300k pasos"
