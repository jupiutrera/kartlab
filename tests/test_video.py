import pygame
import pytest

from kartlab.show import video
from kartlab.show.charts import RewardChart
from kartlab.show.overlays import count_steps, version_name
from kartlab.show.versions import pick_checkpoints
from kartlab.show.video import (
    FPS,
    VIDEO_SIZE,
    VideoWriter,
    probe_video,
    render_demo,
    render_progression,
    render_race,
)
from kartlab.train.config import PPOParams, TrainConfig
from kartlab.train.run import RunPaths


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    from kartlab.train.train import train

    config = TrainConfig(
        name="video-test",
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
    run = train(config, runs_dir=tmp_path_factory.mktemp("runs"), verbose=0)
    # The best model is a copy of the final one (no evaluation ran in this tiny run).
    final = run.checkpoint(512)
    run.best_model.write_bytes(final.model.read_bytes())
    return run


def check_video(path, frames):
    info = probe_video(path)
    assert info.size == VIDEO_SIZE == (1920, 1080)
    assert info.fps == FPS == 60
    assert info.frames == frames
    assert info.codec == "h264" and info.pixel_format.startswith("yuv420p")
    assert info.seconds == pytest.approx(frames / 60, abs=0.05)


@pytest.mark.parametrize(
    "steps, name", [(0, "Sin entrenar"), (10_000, "10k pasos"), (1_500_000, "1,5M pasos")]
)
def test_version_names(steps, name):
    assert version_name(steps) == name


def test_step_counter_uses_spanish_thousands_separators():
    assert count_steps(1_503_232) == "1.503.232"
    assert count_steps(512) == "512"


def test_checkpoints_follow_a_log_scale_and_end_with_the_final_one(tmp_path):
    run = RunPaths(tmp_path)
    run.checkpoints.mkdir()
    steps = [0, 4096, 10_240, 30_720, 98_304, 307_200, 1_003_520, 1_503_232]
    for s in steps:
        run.checkpoint(s).model.write_bytes(b"")
    picked = [c.steps for c in pick_checkpoints(run)]
    assert picked == [0, 10_240, 30_720, 98_304, 307_200, 1_003_520, 1_503_232]
    # With few checkpoints there are no repeats and the final one is always there.
    assert [c.steps for c in pick_checkpoints(run, (0, 1, 2))] == [0, 1_503_232]
    assert [c.steps for c in pick_checkpoints(run, (5_000_000,))] == [1_503_232]


def test_writer_rejects_frames_of_the_wrong_size(tmp_path):
    pygame.font.init()
    writer = VideoWriter(tmp_path / "a.mp4")
    try:
        with pytest.raises(ValueError):
            writer.write(pygame.Surface((320, 180)))
    finally:
        writer.abort()
    assert not list(tmp_path.glob("*.mp4"))  # no half-written file under any name


def test_writer_produces_exact_1080p60_and_renames_when_done(tmp_path):
    surface = pygame.Surface(VIDEO_SIZE)
    path = tmp_path / "plain.mp4"
    with VideoWriter(path) as writer:
        for i in range(12):
            surface.fill((i * 20, 80, 160))
            writer.write(surface)
        assert not path.exists()
    check_video(path, 12)
    assert not (tmp_path / "plain.part.mp4").exists()


def test_log_chart_marks_the_current_version():
    pygame.font.init()
    chart = RewardChart((300, 180), pygame.font.Font(None, 16), log_x=True)
    rows = [{"steps": 4096.0 * 2**k, "reward_mean": float(k)} for k in range(1, 9)]
    surface = chart.draw(rows, 1, marker=30_000)
    assert surface.get_bounding_rect().height > 100
    key = chart._key
    chart.draw(rows, 1, marker=0)  # the untrained version sits at the left edge
    assert chart._key != key


@pytest.fixture
def quick(monkeypatch):
    monkeypatch.setattr(video, "COUNTDOWN_SECONDS", 0.2)
    monkeypatch.setattr(video, "FINISH_SECONDS", 0.2)
    monkeypatch.setattr(video, "SUMMARY_SECONDS", 0.5)


@pytest.mark.smoke
def test_demo_video_is_1080p60_with_the_expected_frame_count(tiny_run, tmp_path, quick):
    out = tmp_path / "demo.mp4"
    warnings = []
    render_demo(tiny_run, out, "plateau", max_seconds=0.5, warn=warnings.append)
    # 30 frames of driving end at frame 29, then the 30 frames of the results card.
    check_video(out, 29 + 30)
    assert warnings == []  # plateau was not the training track
    assert not list(tmp_path.glob("*.part.*"))


def test_demo_warns_when_the_track_was_used_for_training(tiny_run, tmp_path, quick):
    messages = []
    render_demo(tiny_run, tmp_path / "d.mp4", "harbor", max_seconds=0.1, warn=messages.append)
    assert messages and "entrenamiento" in messages[0]


@pytest.mark.smoke
def test_progression_runs_every_version_on_the_same_track(tiny_run, tmp_path, quick):
    out = tmp_path / "progression.mp4"
    render_progression(tiny_run, out, "harbor", max_seconds=0.5, targets=(0,))
    # Two versions: untrained (30 driving frames) and final (twice the limit), each followed
    # by the 30 frames of fade out.
    check_video(out, (29 + 30) + (59 + 30))


@pytest.mark.smoke
def test_race_has_a_countdown_every_version_and_a_finish(tiny_run, tmp_path, quick):
    out = tmp_path / "race.mp4"
    render_race(tiny_run, out, "plateau", laps=1, max_seconds=0.5, targets=(0,))
    # countdown 12 + 30 racing frames (the last one ends it) + 12 of finish.
    check_video(out, 12 + 29 + 12)
