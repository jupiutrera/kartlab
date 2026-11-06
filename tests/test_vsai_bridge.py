"""The race against the AI on the eight with a bridge: the crossing is two levels, so it is a
race circuit (the flat eight is not), karts at different heights never touch, and the
standings and the AI work all the way round."""

import os
from pathlib import Path

import pygame
import pytest

from kartlab.show.humanrace import classify
from kartlab.show.viewer import ViewerApp, main
from kartlab.sim.contact import Body, PairState, resolve
from kartlab.sim.race import Racer
from kartlab.sim.showcase import showcase_track
from kartlab.sim.trackfile import load_track_file
from kartlab.train.config import PPOParams, TrainConfig
from kartlab.train.run import RunPaths

TRACKS = Path(__file__).resolve().parents[1] / "tracks"


@pytest.fixture(scope="module")
def bridge():
    return load_track_file(TRACKS / "ocho_puente.json")


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    from kartlab.train.train import train

    config = TrainConfig(
        name="vsai-bridge-test",
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
def make_app(tiny_run, bridge):
    created = []

    def make(**kwargs):
        kwargs.setdefault("scale", 1)
        app = ViewerApp(tiny_run, "file", bridge, "vsai", realtime=False, **kwargs)
        created.append(app)
        return app

    yield make
    for app in created:
        app.close()
    pygame.quit()


def test_only_the_crossings_on_one_level_keep_a_circuit_out_of_the_race(bridge):
    flat = load_track_file(TRACKS / "ocho.json")
    assert len(bridge.crossings) == 1 and bridge.flat_crossings == 0
    assert len(flat.crossings) == 1 and flat.flat_crossings == 1
    assert showcase_track("harbor").flat_crossings == 0


def test_the_flat_eight_is_still_refused_and_the_bridge_one_is_not(tiny_run):
    with pytest.raises(SystemExit, match="cruces"):
        main(["--run", str(tiny_run.root), "--mode", "vsai", "--track-file",
              str(TRACKS / "ocho.json"), "--frames", "1"])  # fmt: skip
    main(["--run", str(tiny_run.root), "--mode", "vsai", "--track-file",
          str(TRACKS / "ocho_puente.json"), "--frames", "30", "--scale", "1",
          "--rivals", "2"])  # fmt: skip


def test_karts_at_the_crossing_on_different_levels_never_touch(bridge):
    levels = bridge.levels
    over, under = Racer(bridge, start_s=levels.s_over), Racer(bridge, start_s=levels.s_under)
    assert abs(over.kart.x - under.kart.x) < 1.5 and abs(over.kart.y - under.kart.y) < 1.5
    state = PairState()
    assert (
        resolve(Body(over.kart, over.level_z), Body(under.kart, under.level_z), state, 100.0)
        is None
    )
    # the same two karts on one level do touch: only the height keeps them apart
    other = Racer(bridge, start_s=levels.s_over)
    other.kart.x += 0.3
    assert resolve(Body(over.kart, over.level_z), Body(other.kart, other.level_z), state, 100.0)


def test_the_scene_does_not_bump_a_kart_on_the_deck_against_one_in_the_tunnel(make_app, bridge):
    app = make_app(rivals=1)
    scene = app.scene
    app.run_loop(200)  # the countdown, and a few steps of racing
    levels = bridge.levels
    scene.human.reset(levels.s_over)
    driver = scene.drivers[0]
    driver.racer.reset(levels.s_under)
    before = [(r.kart.x, r.kart.y, r.kart.vx, r.kart.vy) for r in (scene.human, driver.racer)]
    scene._bump(500.0)
    after = [(r.kart.x, r.kart.y, r.kart.vx, r.kart.vy) for r in (scene.human, driver.racer)]
    assert before == after
    assert scene.human.level_z > 3.0 and driver.racer.level_z == 0.0
    driver.racer.reset(levels.s_over)  # both on the deck: now they do
    driver.racer.kart.x += 0.4
    scene._bump(500.0)
    assert scene.human.kart.x != before[0][0] or scene.human.kart.vx != before[0][2]


def test_the_standings_and_the_distance_work_all_the_way_round(make_app, bridge):
    app = make_app(rivals=3, vs_laps=1, difficulty="facil")
    scene = app.scene
    app.run_loop(5)
    scene.autopilot_on = True
    crossed = False
    previous = [e.distance for e in scene.entrants]
    for _ in range(60 * 150):
        app.run_loop(10)
        crossed |= scene.human.level_z > 3.0  # the human has been over the bridge
        for k, _driver in enumerate(scene.drivers, start=1):
            assert scene.entrants[k].distance >= previous[k] - 1.0  # never goes back
        previous = [e.distance for e in scene.entrants]
        assert 1 <= scene.place <= len(scene.entrants)
        if scene.ended_at is not None:
            break
    assert crossed and scene.ended_at is not None
    order = [e.name for e in classify(scene.entrants)]
    assert len(order) == 4 and set(order) == {e.name for e in scene.entrants}


def reference_run() -> RunPaths | None:
    env = os.environ.get("KARTLAB_REFERENCE_RUN")
    return RunPaths(Path(env)) if env else None


def test_the_reference_ai_finishes_a_race_on_the_bridge_eight_without_restarts(bridge):
    run = reference_run()
    if run is None or not run.best_model.exists():
        pytest.skip("sin run de referencia (KARTLAB_REFERENCE_RUN)")
    app = ViewerApp(run, "file", bridge, "vsai", scale=1, realtime=False, vs_laps=2, rivals=3,
                    difficulty="campeon")  # fmt: skip
    try:
        scene = app.scene
        app.run_loop(5)
        scene.autopilot_on = True
        for _ in range(60 * 200):
            app.run_loop(60)
            if scene.ended_at is not None:
                break
        assert scene.ended_at is not None
        assert all(e.finished for e in scene.entrants[1:])  # the three AI finish
        assert all(d.restarts == 0 for d in scene.drivers)
        assert all(abs(e.best_lap - 52.5) < 3.0 for e in scene.entrants[1:])  # their usual lap
    finally:
        app.close()
        pygame.quit()
