import json
import shutil

import numpy as np
import pygame
import pytest

from kartlab.env.kart_env import EnvConfig
from kartlab.show.charts import MetricsReader, RewardChart
from kartlab.show.drivers import PolicyDriver, Version, format_steps
from kartlab.show.live import LiveApp
from kartlab.sim.showcase import showcase_track
from kartlab.sim.trackfile import load_track_file
from kartlab.train.config import PPOParams, TrainConfig


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    from kartlab.train.train import train

    config = TrainConfig(
        name="live-test",
        total_steps=512,
        n_envs=2,
        vec_env="dummy",
        low_priority=False,
        ppo=PPOParams(n_steps=64, batch_size=64, n_epochs=1),
        checkpoint_every=256,
        checkpoint_at=(0,),
        eval_tracks=(),
        env={"track": "harbor"},
    )
    return train(config, runs_dir=tmp_path_factory.mktemp("runs"), verbose=0)


def still(obs):
    return np.zeros((len(obs), 2), dtype=np.float32)


@pytest.mark.parametrize(
    "steps, text", [(0, "0"), (512, "512"), (20_000, "20k"), (1_200_000, "1,2M")]
)
def test_format_steps(steps, text):
    assert format_steps(steps) == text


def test_a_stuck_kart_restarts_at_the_line():
    driver = PolicyDriver(Version(0, still), showcase_track("harbor"), EnvConfig(), (255, 0, 0))
    events = [driver.step() for _ in range(int(3.5 * 60))]
    assert "restart" in events
    assert driver.racer.position.s == pytest.approx(0.0, abs=1e-6)


def test_metrics_reader_skips_a_half_written_line(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("steps,reward_mean\n4096,1.5\n8192,2.5\n12288,3", encoding="utf-8")
    reader = MetricsReader(path)
    reader.refresh()
    assert [r["steps"] for r in reader.rows] == [4096, 8192]
    version = reader.version
    reader.refresh()
    assert reader.version == version


def test_reward_chart_is_cached_until_data_changes():
    pygame.init()
    chart = RewardChart((90, 54))
    rows = [{"steps": 4096.0 * k, "reward_mean": float(k)} for k in range(1, 5)]
    first = chart.draw(rows, 1, marker=8192)
    key = chart._key
    chart.draw(rows, 1, marker=8192)
    assert chart._key == key and first is chart.surface
    chart.draw(rows, 2, marker=8192)
    assert chart._key != key


def test_viewer_shows_the_latest_version_and_swaps_in_new_ones(tiny_run):
    app = LiveApp(tiny_run, track_choice="harbor", ghosts=2, scale=2, realtime=False)
    newer = tiny_run.checkpoint(1_000_000)
    try:
        app.loader.poll_once()
        app.run(max_frames=5)
        assert app.current is not None and app.current.version.steps == 512
        assert app.notice is not None and "512" in app.notice[0]

        # A newer checkpoint appears while the kart is mid-lap.
        latest = tiny_run.checkpoint(512)
        shutil.copy(latest.model, newer.model)
        shutil.copy(latest.vecnormalize, newer.vecnormalize)
        tiny_run.latest.write_text(json.dumps({"steps": 1_000_000}), encoding="utf-8")
        app.loader.poll_once()
        app.run(max_frames=1)
        assert app.pending is not None and app.current.version.steps == 512

        # It enters at the end of the lap or when the kart is restarted.
        for _ in range(60):
            app.run(max_frames=60)
            if app.current.version.steps == 1_000_000:
                break
        assert app.current.version.steps == 1_000_000
        assert [g.version.steps for g in app.ghosts] == [512]
    finally:
        pygame.quit()
        newer.model.unlink(missing_ok=True)
        newer.vecnormalize.unlink(missing_ok=True)
        tiny_run.latest.write_text(json.dumps({"steps": 512}), encoding="utf-8")


def test_headless_viewer_preloads_ghosts_and_saves_a_screenshot(tiny_run, tmp_path):
    from kartlab.show.live import main

    out = tmp_path / "live.png"
    main(["--run", str(tiny_run.root), "--scale", "2", "--frames", "30", "--screenshot", str(out)])
    assert out.stat().st_size > 0


def test_ghosts_are_earlier_versions_on_a_log_scale(tiny_run):
    app = LiveApp(tiny_run, track_choice="harbor", ghosts=2, scale=2, realtime=False)
    try:
        app.preload_ghosts()
        steps = [g.version.steps for g in app.ghosts]
        latest = tiny_run.read_latest().steps
        assert steps and all(s < latest for s in steps)
    finally:
        pygame.quit()


def test_a_driver_can_be_placed_on_a_grid_slot():
    track = showcase_track("harbor")
    driver = PolicyDriver(Version(0, still), track, EnvConfig(), (255, 0, 0))
    driver.place(-11.0, 2.4)
    position = driver.racer.position
    assert position.s == pytest.approx(track.length - 11.0, abs=0.5)
    assert position.lateral == pytest.approx(2.4, abs=0.1)
    assert driver.restarts == 0


def test_live_viewer_runs_on_a_track_file(tiny_run, tmp_path):
    from kartlab.show.live import main

    out = tmp_path / "file.png"
    argv = ["--run", str(tiny_run.root), "--scale", "2", "--frames", "10", "--screenshot", str(out)]
    main([*argv, "--track-file", "tracks/ejemplo.json"])
    assert out.stat().st_size > 0


def test_live_viewer_rejects_an_invalid_track_file(tiny_run, tmp_path):
    from kartlab.show.live import main

    bad = tmp_path / "mal.json"
    bad.write_text('{"format": 1, "name": "x", "width": 12, "corners": []}', encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        main(["--run", str(tiny_run.root), "--track-file", str(bad), "--frames", "1"])
    assert "Circuito no válido" in str(exit_info.value)


def _palette_only(surface):
    from kartlab.render.palette import game_palette

    palette = {tuple(int(c) for c in row) for row in game_palette().rgb}
    pixels = pygame.surfarray.array3d(surface).reshape(-1, 3)
    return {tuple(int(c) for c in p) for p in np.unique(pixels, axis=0)} <= palette


def test_the_live_top_down_view_is_a_320_by_180_canvas_of_palette_inks(tiny_run):
    app = LiveApp(
        tiny_run, track_choice="harbor", ghosts=2, scale=2, realtime=False, view="topdown"
    )
    try:
        app.preload_ghosts()
        app.loader.poll_once()
        app.run(max_frames=60)
        assert app.canvas.get_size() == (320, 180)
        assert app.view_rect.width + app.panel_rect.width == 320
        assert _palette_only(app.canvas)
        app.handle_key(pygame.K_b)  # with the brain panel too
        app.run(max_frames=30)
        assert _palette_only(app.canvas)
        assert app.screen.get_size() == (640, 360)
    finally:
        pygame.quit()


def test_the_live_3d_view_puts_the_chart_on_the_world_grid(tiny_run):
    from kartlab.render import pixelui as ui

    app = LiveApp(tiny_run, track_choice="harbor", ghosts=0, scale=2, realtime=False)
    try:
        app.loader.poll_once()
        app.run(max_frames=30)
        assert "interfaz" in app.game.timings
        world = app.game.world.world
        assert _palette_only(world)  # no ghosts: no additive halos
        w, h = app.chart3d.size
        x, y = 320 - w - 4, (180 - h) // 2
        assert world.get_at((x, y))[:3] == ui.rgb(ui.NIGHT)  # the plate's border
        assert world.get_at((x + 1, y + 1))[:3] == ui.rgb(ui.SLATE)
    finally:
        pygame.quit()


def test_the_live_viewer_waits_for_the_first_checkpoint_on_the_canvas(tiny_run, tmp_path):
    from kartlab.train.run import RunPaths

    paths = RunPaths(tmp_path / "run")  # a run that has no checkpoint yet
    paths.root.mkdir()
    shutil.copy(tiny_run.config, paths.config)
    app = LiveApp(paths, track_choice="harbor", ghosts=0, scale=2, realtime=False)
    try:
        app.run(max_frames=3)
        assert app.current is None and _palette_only(app.canvas)
        app.handle_key(pygame.K_t)
        app.run(max_frames=3)
        assert _palette_only(app.canvas)
    finally:
        pygame.quit()


def test_the_live_top_down_view_passes_the_road_height_so_a_tunnel_hides_its_karts(tiny_run):
    app = LiveApp(
        tiny_run,
        track_choice="harbor",
        ghosts=1,
        scale=2,
        realtime=False,
        view="topdown",
        track=load_track_file("tracks/ocho_puente.json"),
    )
    try:
        app.preload_ghosts()
        app.loader.poll_once()
        app.run(max_frames=5)
        seen = []
        original = app.view.draw
        app.view.draw = lambda target, karts, *a, **k: (
            seen.append(list(karts)),
            original(target, karts, *a, **k),
        )
        app.run(max_frames=2)
        assert seen and all(len(kart) == 5 for karts in seen for kart in karts)
        assert all(kart[4] == 0.0 for kart in seen[-1])  # on the ground here, not on the deck
    finally:
        pygame.quit()


def test_the_live_notice_does_not_cover_the_chart_the_brain_or_the_model_panel(tiny_run):
    app = LiveApp(tiny_run, track_choice="harbor", ghosts=0, scale=2, realtime=False, brain=True)
    try:
        app.loader.poll_once()
        app.run(max_frames=40)  # "Nueva versión" is up
        hud = app.game.hud
        assert hud.notice_rect is not None and len(hud.reserved) >= 5
        assert not [r for r in hud.reserved if r.colliderect(hud.notice_rect)]
    finally:
        pygame.quit()
