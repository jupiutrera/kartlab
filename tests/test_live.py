import json
import shutil

import numpy as np
import pygame
import pytest

from kartlab.env.kart_env import EnvConfig
from kartlab.show.live import (
    LiveApp,
    MetricsReader,
    PolicyDriver,
    RewardChart,
    Version,
    format_steps,
)
from kartlab.sim.showcase import showcase_track
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
    pygame.font.init()
    chart = RewardChart((300, 160), pygame.font.Font(None, 16))
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
