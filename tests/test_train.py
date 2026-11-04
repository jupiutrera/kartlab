import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kartlab.common.io import read_yaml
from kartlab.common.paths import REPO_ROOT
from kartlab.env.kart_env import EnvConfig
from kartlab.train.callbacks import CheckpointCallback, CurriculumCallback
from kartlab.train.config import CurriculumStage, PPOParams, TrainConfig
from kartlab.train.evaluate import evaluate
from kartlab.train.run import RunPaths, create_run, latest_run

CONFIGS = sorted((REPO_ROOT / "configs").glob("*.yaml"))


def tiny_config(**overrides) -> TrainConfig:
    defaults = {
        "name": "smoke",
        "total_steps": 512,
        "n_envs": 2,
        "vec_env": "dummy",
        "low_priority": False,
        "ppo": PPOParams(n_steps=64, batch_size=64, n_epochs=1),
        "checkpoint_every": 256,
        "checkpoint_at": (0,),
        "eval_every": 256,
        "eval_tracks": ("harbor",),
        "eval_laps": 1,
        "env": {"max_seconds": 5.0},
    }
    return TrainConfig(**{**defaults, **overrides})


# -- configuration ----------------------------------------------------------------------


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_config_file_loads(path):
    config = TrainConfig.from_yaml(path)
    assert isinstance(config.env_config(), EnvConfig)


def test_config_round_trips_through_a_plain_dict():
    config = tiny_config(curriculum=(CurriculumStage(0, track_width=(16.0, 16.0)),))
    assert TrainConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    "data",
    [
        {"name": "x", "learning_rate": 1e-3},
        {"name": "x", "ppo": {"lr": 1e-3}},
        {"name": "x", "env": {"rays": 9}},
        {"name": "x", "curriculum": [{"from_steps": 0, "width": [1, 2]}]},
        {"name": "x", "vec_env": "threads"},
    ],
)
def test_unknown_or_invalid_options_are_rejected(data):
    with pytest.raises(ValueError):
        TrainConfig.from_dict(data)


# -- runs and checkpoints ------------------------------------------------------------------


def test_create_run_writes_config_and_commit(tmp_path):
    run = create_run({"name": "demo"}, "demo", tmp_path)
    assert run.root.name.endswith("_demo")
    assert read_yaml(run.config) == {"name": "demo"}
    meta = read_yaml(run.meta)
    assert len(meta["commit"]) == 40
    assert run.checkpoints.is_dir()
    second = create_run({"name": "demo"}, "demo", tmp_path)
    assert second.root != run.root
    assert latest_run(tmp_path) == second


def test_checkpoint_schedule_mixes_periodic_saves_and_milestones():
    callback = CheckpointCallback(RunPaths(Path("unused")), 20_000, (0, 10_000, 30_000, 100_000))
    due = []
    for _ in range(7):
        due.append(callback._next_due())
        callback.saved.append(due[-1])
    assert due == [0, 10_000, 20_000, 30_000, 40_000, 60_000, 80_000]


def test_curriculum_switches_track_ranges_at_each_stage():
    calls = []
    callback = CurriculumCallback(
        [CurriculumStage(0, track_width=(16.0, 16.0)), CurriculumStage(100, track_width=(12, 12))],
        verbose=0,
    )
    env = SimpleNamespace(env_method=lambda name, **kw: calls.append(kw))
    callback.model = SimpleNamespace(
        logger=SimpleNamespace(record=lambda *a: None), get_env=lambda: env
    )
    for steps in (0, 50, 120, 200):
        callback.num_timesteps = steps
        callback._on_step()
    assert [c["width"] for c in calls] == [(16.0, 16.0), (12, 12)]


def test_evaluation_of_a_policy_that_never_moves():
    result = evaluate(
        lambda obs: np.zeros((len(obs), 2), dtype=np.float32),
        ["harbor", "eval:1"],
        EnvConfig(),
        laps=1,
    )
    assert result.success_rate == 0.0 and result.progress_mean == 0.0
    assert {t.end for t in result.tracks} == {"stuck"}
    assert result.lap_time_mean is None


# -- a real (tiny) training run --------------------------------------------------------


def check_run_outputs(run: RunPaths, total: int) -> None:
    steps = [c.steps for c in run.list_checkpoints()]
    assert steps[0] == 0 and steps[-1] == total and 256 in steps
    assert run.read_latest().steps == total
    assert json.loads(run.latest.read_text(encoding="utf-8"))["model"] == f"model_{total:09d}.zip"
    for checkpoint in run.list_checkpoints():
        assert checkpoint.model.exists() and checkpoint.vecnormalize.exists()
    with open(run.metrics, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["steps"]) for r in rows] == [128 * k for k in range(1, len(rows) + 1)]
    with open(run.evals, encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) >= 1
    assert run.best_model.exists() and run.best_vecnormalize.exists()
    summary = read_yaml(run.summary)
    assert summary["steps"] == total and not summary["interrupted"]
    assert not list(run.root.rglob("*.tmp"))


@pytest.mark.smoke
def test_a_few_hundred_training_steps_produce_a_complete_run(tmp_path):
    from stable_baselines3 import PPO

    from kartlab.train.train import train

    run = train(tiny_config(), runs_dir=tmp_path, verbose=0)
    check_run_outputs(run, 512)
    model = PPO.load(run.best_model, device="cpu")
    action, _ = model.predict(np.zeros((1, 18), dtype=np.float32), deterministic=True)
    assert action.shape == (1, 2)


@pytest.mark.smoke
def test_training_with_spawned_subprocess_envs(tmp_path):
    from kartlab.train.train import train

    run = train(tiny_config(vec_env="subproc", eval_tracks=()), runs_dir=tmp_path, verbose=0)
    assert [c.steps for c in run.list_checkpoints()] == [0, 256, 512]


def test_best_model_ties_are_broken_by_faster_laps():
    from kartlab.train.evaluate import EvalResult, TrackResult

    def result(lap_time):
        track = TrackResult("eval:1", True, 3, 1.0, 3 * lap_time, (lap_time,) * 3, "laps")
        return EvalResult((track,))

    assert result(38.0).score() > result(40.0).score()
