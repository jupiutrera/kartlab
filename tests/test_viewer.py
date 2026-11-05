"""The exhibition viewer, headless: one display frame is one physics step (1/60 s)."""

import shutil
import time

import numpy as np
import pygame
import pytest

from kartlab.show.title import build_wordmark
from kartlab.show.versions import BEST, VersionLibrary, pick_checkpoints
from kartlab.show.viewer import SPEEDS, ViewerApp, latest_run_with_checkpoints, main
from kartlab.sim.kart import DT
from kartlab.train.config import PPOParams, TrainConfig


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    from kartlab.train.train import train

    config = TrainConfig(
        name="viewer-test",
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
    run.best_model.write_bytes(final.model.read_bytes())  # as a real run has it
    return run


@pytest.fixture
def app(tiny_run):
    created = []

    def make(**kwargs):
        viewer = ViewerApp(tiny_run, scale=2, realtime=False, **kwargs)
        created.append(viewer)
        return viewer

    yield make
    for viewer in created:
        viewer.close()
    pygame.quit()


def press(viewer, key):
    viewer.handle_key(key)


def colours(viewer) -> int:
    pixels = pygame.surfarray.array3d(viewer.screen).reshape(-1, 3)
    return len(np.unique(pixels, axis=0))


@pytest.mark.smoke
def test_title_screen_renders_and_the_best_model_drives_behind_it(app):
    viewer = app()
    assert viewer.state == "title"
    viewer.run_loop(150)
    assert viewer.attract.driver is not None and viewer.attract.driver.racer.time > 1.0
    assert viewer.screen.get_size() == (640, 360)
    assert colours(viewer) > 60  # a wordmark over a lit 3D scene, not a flat fill


def test_title_menu_picks_a_mode_and_enter_starts_it(app):
    viewer = app()
    viewer.run_loop(5)
    press(viewer, pygame.K_DOWN)
    assert viewer.title.entry.key == "race"
    press(viewer, pygame.K_DOWN)
    press(viewer, pygame.K_DOWN)
    assert viewer.title.entry.key == "watch"  # wraps around
    press(viewer, pygame.K_LSHIFT)  # a modifier alone does not start anything
    viewer.run_loop(5)
    assert viewer.state == "title"
    press(viewer, pygame.K_UP)
    press(viewer, pygame.K_RETURN)
    viewer.run_loop(60)
    assert viewer.state == "scene" and viewer.scene.name == "progression"


def test_escape_leaves_a_mode_for_the_title_and_the_title_for_the_exit(app):
    viewer = app(mode="watch")
    viewer.run_loop(5)
    assert viewer.state == "scene"
    press(viewer, pygame.K_ESCAPE)
    viewer.run_loop(60)
    assert viewer.state == "title"
    press(viewer, pygame.K_ESCAPE)
    assert viewer.running is False


@pytest.mark.smoke
@pytest.mark.parametrize("mode", ["watch", "race", "progression"])
def test_each_mode_renders_frames(app, mode):
    viewer = app(mode=mode)
    viewer.run_loop(240)  # past the race countdown
    assert viewer.state == "scene" and viewer.scene.name == mode
    assert colours(viewer) > 60
    if mode == "watch":
        driver = viewer.scene.driver  # an untrained net may get stuck and restart: still driving
        assert driver.restarts >= 1 or driver.racer.time > 3.0
    elif mode == "race":
        assert len(viewer.scene.entries) == len(pick_checkpoints(viewer.run))
        assert viewer.scene.best.driver.racer.time > 0.0
    else:
        assert viewer.scene.driver is not None and viewer.scene.frame_no > 0


def test_modes_cycle_with_m(app):
    viewer = app(mode="watch")
    viewer.run_loop(5)
    seen = []
    for _ in range(3):
        press(viewer, pygame.K_m)
        viewer.run_loop(40)
        seen.append(viewer.scene.name)
    assert seen == ["race", "progression", "watch"]


def test_version_switching_in_watch_mode(app):
    viewer = app(mode="watch")
    viewer.run_loop(5)
    steps = viewer.library.steps
    assert len(steps) >= 3
    scene = viewer.scene
    assert scene.shown == steps[-1]  # starts on the last version
    press(viewer, pygame.K_LEFT)
    viewer.run_loop(60)
    assert scene.shown == steps[-2] and scene.driver.version.steps == steps[-2]
    assert scene.driver.racer.time < 1.5  # the new version starts again at the line
    press(viewer, pygame.K_HOME)
    viewer.run_loop(60)
    assert scene.shown == steps[0]
    press(viewer, pygame.K_LEFT)  # clamped at the first one
    assert scene.index == 0
    press(viewer, pygame.K_END)
    viewer.run_loop(60)
    assert scene.shown == steps[-1]
    scene.on_wheel(-1)
    viewer.run_loop(60)
    assert scene.shown == steps[-2]


def test_start_version_picks_the_nearest_checkpoint(app):
    viewer = app(mode="watch", version=300)
    assert viewer.scene.steps[viewer.scene.index] == 256


def test_pause_and_slow_motion_scale_the_simulated_time(app):
    viewer = app(mode="watch")
    viewer.run_loop(60)
    racer = viewer.scene.driver.racer
    assert racer.time == pytest.approx(60 * DT, abs=2 * DT)
    base = racer.time

    press(viewer, pygame.K_p)  # pause: nothing advances, the view still draws
    viewer.run_loop(60)
    assert racer.time == base and viewer.paused
    press(viewer, pygame.K_SPACE)
    assert not viewer.paused

    press(viewer, pygame.K_s)  # half speed
    assert viewer.speed == SPEEDS[1] == 0.5
    viewer.run_loop(60)
    assert racer.time - base == pytest.approx(30 * DT, abs=2 * DT)
    base = racer.time

    press(viewer, pygame.K_s)  # quarter speed
    viewer.run_loop(60)
    assert racer.time - base == pytest.approx(15 * DT, abs=2 * DT)

    press(viewer, pygame.K_s)  # back to normal
    assert viewer.speed == 1.0


def test_race_pauses_with_the_countdown_and_changes_the_followed_kart(app):
    viewer = app(mode="race")
    viewer.run_loop(30)
    scene = viewer.scene
    assert all(e.driver.racer.time == 0.0 for e in scene.entries)  # still counting down
    assert scene.focus == 0
    press(viewer, pygame.K_LEFT)
    assert scene.focus == 1
    press(viewer, pygame.K_RIGHT)
    press(viewer, pygame.K_RIGHT)
    assert scene.focus == len(scene.entries) - 1  # wraps to the slowest
    viewer.run_loop(200)
    assert scene.best.driver.racer.time > 0.0


def test_view_keys(app):
    viewer = app(mode="watch")
    viewer.run_loop(5)
    camera = viewer.game.camera_name
    press(viewer, pygame.K_c)
    assert viewer.game.camera_name != camera and viewer.camera_choice == viewer.game.camera_name
    press(viewer, pygame.K_v)
    assert viewer.game.world.variant == viewer.variant == "sunset"
    press(viewer, pygame.K_g)
    viewer.run_loop(5)
    assert viewer.rays and viewer.game.world.show_rays
    press(viewer, pygame.K_F3)
    viewer.run_loop(5)
    assert viewer.diagnostics and viewer.game.diagnostics
    assert "simulación" in viewer.game.timings
    press(viewer, pygame.K_h)
    viewer.run_loop(2)
    assert viewer.help_open
    press(viewer, pygame.K_ESCAPE)  # closes the help first, it does not leave the mode
    assert not viewer.help_open and viewer.state == "scene"


def test_track_keys_switch_circuit_and_reuse_the_cached_view(app):
    viewer = app(mode="watch")
    viewer.run_loop(5)
    harbor_game = viewer.game
    press(viewer, pygame.K_2)
    viewer.run_loop(5)  # the first frame shows "Cargando", the next one has the new track
    assert viewer.track_key == "canyon" and viewer.game is not harbor_game
    assert viewer.scene.driver is not None and viewer.scene.driver.racer.time < 0.5
    press(viewer, pygame.K_1)
    viewer.run_loop(5)
    assert viewer.game is harbor_game  # cached: no rebuild
    press(viewer, pygame.K_e)
    viewer.run_loop(5)
    assert viewer.track_key == "eval:1"


def test_screenshot_option_writes_a_png(tiny_run, tmp_path):
    out = tmp_path / "shots" / "viewer.png"
    main(["--run", str(tiny_run.root), "--scale", "2", "--frames", "40", "--screenshot", str(out)])
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert pygame.image.load(str(out)).get_size() == (640, 360)


def test_viewer_runs_on_a_track_file(tiny_run, tmp_path):
    out = tmp_path / "file.png"
    argv = ["--run", str(tiny_run.root), "--scale", "2", "--mode", "watch", "--frames", "20"]
    main([*argv, "--track-file", "tracks/ejemplo.json", "--screenshot", str(out)])
    assert out.stat().st_size > 0


def test_invalid_track_file_exits_with_a_clear_message(tiny_run, tmp_path):
    bad = tmp_path / "mal.json"
    bad.write_text('{"format": 1, "name": "x", "width": 12, "corners": []}', encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        main(["--run", str(tiny_run.root), "--track-file", str(bad), "--frames", "1"])
    assert "Circuito no válido" in str(exit_info.value)


def test_a_run_without_checkpoints_exits_with_a_clear_message(tmp_path, tiny_run):
    empty = tmp_path / "2099-01-01_0000_empty"
    empty.mkdir()
    shutil.copy(tiny_run.config, empty / "config.yaml")
    with pytest.raises(SystemExit) as exit_info:
        main(["--run", str(empty), "--frames", "1"])
    assert "no tiene checkpoints" in str(exit_info.value)


def test_default_run_is_the_latest_one_that_has_checkpoints(tmp_path, tiny_run):
    older = tmp_path / "2026-01-01_0000_a"
    newer = tmp_path / "2026-02-01_0000_b"  # has a config but nothing to show yet
    shutil.copytree(tiny_run.root, older)
    newer.mkdir()
    shutil.copy(tiny_run.config, newer / "config.yaml")
    assert latest_run_with_checkpoints(tmp_path).root == older
    assert latest_run_with_checkpoints(tmp_path / "nothing") is None


def test_title_works_without_best_model_and_with_missing_checkpoint_steps(tmp_path, tiny_run):
    run_dir = tmp_path / "2026-03-01_0000_partial"
    shutil.copytree(tiny_run.root, run_dir)
    (run_dir / "best_model.zip").unlink()
    run = type(tiny_run)(run_dir)
    run.checkpoint(256).model.unlink()  # a hole in the list of versions
    viewer = ViewerApp(run, scale=2, realtime=False)
    try:
        viewer.run_loop(60)
        assert viewer.attract.driver is not None  # fell back to the latest version
        assert 256 not in viewer.library.steps
    finally:
        viewer.close()
        pygame.quit()


# -- the pieces ------------------------------------------------------------------------------


def test_wordmark_is_built_from_the_pixel_font():
    pygame.font.init()
    mark, silhouette = build_wordmark()
    assert mark.get_size() == silhouette.get_size()
    assert mark.get_width() > mark.get_height() > 20
    assert pygame.mask.from_surface(mark).count() > 500  # opaque pixels: it has been drawn


def test_library_loads_in_the_background_and_caches(tiny_run):
    library = VersionLibrary(tiny_run)
    try:
        steps = library.steps[-1]
        assert library.request(steps) is None  # asynchronous: nothing yet
        deadline = time.monotonic() + 60
        while library.cached(steps) is None and time.monotonic() < deadline:
            library.poll()
            time.sleep(0.02)
        version = library.cached(steps)
        assert version is not None and version.steps == steps
        assert library.request(steps) is version  # now served from the cache
        assert not library.busy
    finally:
        library.close()


def test_library_reports_a_broken_checkpoint_without_raising(tmp_path, tiny_run):
    run_dir = tmp_path / "2026-04-01_0000_broken"
    shutil.copytree(tiny_run.root, run_dir)
    run = type(tiny_run)(run_dir)
    run.checkpoint(256).model.write_bytes(b"not a zip")
    library = VersionLibrary(run, synchronous=True)
    assert library.request(256) is None
    assert library.failed(256)
    other = next(steps for steps in library.steps if steps != 256)
    assert library.request(other) is not None  # the rest still load
    assert library.cached(BEST) is None
    assert library.request(BEST) is not None  # best_model.zip


def test_library_notices_new_checkpoints_while_training(tmp_path, tiny_run):
    run_dir = tmp_path / "2026-05-01_0000_growing"
    shutil.copytree(tiny_run.root, run_dir)
    run = type(tiny_run)(run_dir)
    library = VersionLibrary(run, synchronous=True)
    count = len(library)
    assert library.refresh(1.0) is False  # too soon to look again
    last = run.list_checkpoints()[-1]
    run.checkpoint(1_000_000).model.write_bytes(last.model.read_bytes())
    assert library.refresh(10.0) is True and len(library) == count + 1


def test_fullscreen_toggle_resizes_everything_and_back(app):
    viewer = app(mode="watch")
    viewer.run_loop(5)
    windowed = viewer.screen.get_size()
    press(viewer, pygame.K_F11)
    viewer.run_loop(5)
    desktop = tuple(pygame.display.get_desktop_sizes()[0])
    assert viewer.fullscreen and viewer.screen.get_size() == desktop
    assert viewer.game.window == desktop and viewer.ov.size == desktop
    press(viewer, pygame.K_F11)
    viewer.run_loop(5)
    assert not viewer.fullscreen and viewer.screen.get_size() == windowed
    assert viewer.game.window == windowed


def test_function_keys_do_not_start_a_mode_from_the_title(app):
    viewer = app()
    viewer.run_loop(5)
    press(viewer, pygame.K_F5)
    viewer.run_loop(60)
    assert viewer.state == "title"
    press(viewer, pygame.K_x)  # any other key does
    viewer.run_loop(60)
    assert viewer.state == "scene"
