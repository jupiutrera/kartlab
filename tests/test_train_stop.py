"""The stop file: the main menu's way of ending a training run cleanly."""

from kartlab.common.io import read_yaml
from kartlab.train.callbacks import MetricsCallback, StopFileCallback
from kartlab.train.config import PPOParams, TrainConfig
from kartlab.train.train import main, train

TOTAL_STEPS = 2000


def minimal_config() -> TrainConfig:
    """One environment, 2 000 steps: a run that takes a few seconds."""
    return TrainConfig(
        name="stop-test",
        total_steps=TOTAL_STEPS,
        n_envs=1,
        vec_env="dummy",
        low_priority=False,
        ppo=PPOParams(n_steps=64, batch_size=64, n_epochs=1),
        checkpoint_every=500,
        checkpoint_at=(0,),
        eval_every=100_000,
        eval_tracks=("harbor",),
        eval_laps=1,
        env={"max_seconds": 5.0},
    )


def test_stop_file_callback_keeps_going_until_the_file_exists(tmp_path):
    callback = StopFileCallback(tmp_path / "stop")
    assert callback._on_step() is True
    assert not callback.stopped
    (tmp_path / "stop").write_text("", encoding="utf-8")
    assert callback._on_step() is False
    assert callback.stopped


def test_stop_file_saves_a_checkpoint_and_the_summary_but_skips_the_evaluation(
    tmp_path, monkeypatch
):
    stop = tmp_path / "train.stop"
    original = MetricsCallback._on_rollout_end

    def stop_after_a_few_rollouts(self):
        original(self)
        if self.num_timesteps >= 256:  # what the menu's "Parar" button does
            stop.write_text("", encoding="utf-8")

    monkeypatch.setattr(MetricsCallback, "_on_rollout_end", stop_after_a_few_rollouts)
    run = train(minimal_config(), runs_dir=tmp_path / "runs", verbose=0, stop_file=stop)

    summary = read_yaml(run.summary)
    assert summary["interrupted"] is True
    assert 256 <= summary["steps"] < TOTAL_STEPS
    assert summary["best_eval"] is None
    assert run.read_latest().steps == summary["steps"]  # the last checkpoint is the stop point
    assert run.checkpoint(summary["steps"]).model.exists()
    assert not run.evals.exists()  # no final evaluation after a stop
    assert not run.best_model.exists()


def test_without_a_stop_file_the_run_finishes_and_evaluates(tmp_path):
    run = train(
        minimal_config(), runs_dir=tmp_path / "runs", verbose=0, stop_file=tmp_path / "never"
    )
    summary = read_yaml(run.summary)
    assert summary["interrupted"] is False
    assert summary["steps"] >= TOTAL_STEPS
    assert run.evals.exists()


def test_train_main_accepts_the_stop_file_option(tmp_path):
    config = tmp_path / "mini.yaml"
    config.write_text(
        "name: cli-stop\ntotal_steps: 256\nn_envs: 1\nvec_env: dummy\nlow_priority: false\n"
        "ppo: {n_steps: 64, batch_size: 64, n_epochs: 1}\ncheckpoint_at: [0]\n"
        "eval_tracks: []\nenv: {max_seconds: 5.0}\n",
        encoding="utf-8",
    )
    stop = tmp_path / "stop"
    stop.write_text("", encoding="utf-8")  # already there: stops at the first step
    main([str(config), "--runs-dir", str(tmp_path / "runs"), "--stop-file", str(stop)])
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert read_yaml(run_dir / "summary.yaml")["interrupted"] is True
