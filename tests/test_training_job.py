"""The training launcher of the main menu, without any real process."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kartlab.common.paths import REPO_ROOT
from kartlab.show.training_job import (
    JOB_FILE_NAME,
    STOP_WAIT_SECONDS,
    TrainingError,
    TrainingJob,
    creation_flags,
    estimate_minutes,
    estimate_text,
    first_comment,
    last_metrics_steps,
    list_configs,
    live_run,
    process_started_at,
)


class FakeProcess:
    pid = 0

    def __init__(self) -> None:
        self.code: int | None = None
        self.terminated = False

    def poll(self):
        return self.code

    def terminate(self) -> None:
        self.terminated = True
        self.code = 1


class Clock:
    def __init__(self, now: float = 1_700_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "mini.yaml"
    path.write_text("# Una prueba\nname: mini\ntotal_steps: 400000\n", encoding="utf-8")
    return path


def make_job(config, tmp_path, process=None, clock=None, platform="win32"):
    calls = []
    process = process or FakeProcess()

    def spawn(command, **kwargs):
        calls.append((command, kwargs))
        return process

    job = TrainingJob(
        config,
        tmp_path / "logs",
        tmp_path / "runs",
        spawn=spawn,
        clock=clock or Clock(),
        platform=platform,
    )
    return job, calls, process


def test_estimates_are_static_and_round_up():
    assert estimate_minutes(160_000) == 2  # 100 s
    assert estimate_minutes(1) == 1
    assert estimate_text(400_000) == "unos 5 min"
    assert estimate_text(3_000_000) == "unos 32 min"


def test_configs_are_listed_with_name_steps_and_first_comment(tmp_path):
    (tmp_path / "a.yaml").write_text("# Primera línea\n# Segunda\nname: a\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("name: b\ntotal_steps: 1000\n", encoding="utf-8")
    (tmp_path / "c.yaml").write_text("name: c\nnope: 1\n", encoding="utf-8")
    a, b, c = list_configs(tmp_path)
    assert (a.name, a.comment, b.comment) == ("a", "Primera línea", "")
    assert b.total_steps == 1000 and not b.is_long
    assert c.error and "nope" in c.error and c.total_steps == 0
    assert first_comment(tmp_path / "missing.yaml") == ""


def test_the_command_logs_to_a_file_with_a_stop_file_and_low_priority(config, tmp_path):
    job, calls, _ = make_job(config, tmp_path)
    job.start()
    command, kwargs = calls[0]
    assert command[:3] == [sys.executable, "-m", "kartlab.train.train"]
    assert str(config) in command
    assert command[command.index("--runs-dir") + 1] == str(tmp_path / "runs")
    assert command[command.index("--stop-file") + 1] == str(job.stop_path)
    assert job.log_path.parent == job.stop_path.parent == tmp_path / "logs"
    assert job.log_path.exists()  # created before the trainer writes to it
    assert kwargs["stdout"].name == str(job.log_path) and kwargs["stderr"] == subprocess.STDOUT
    assert kwargs["env"]["PYTHONUTF8"] == "1" and kwargs["cwd"] == str(REPO_ROOT)
    assert creation_flags("win32") == 0x4000 | 0x08000000 | 0x200
    assert kwargs["creationflags"] == creation_flags("win32")
    assert creation_flags("linux") == 0


def test_on_linux_the_trainer_gets_its_own_session(config, tmp_path):
    job, calls, _ = make_job(config, tmp_path, platform="linux")
    job.start()
    assert calls[0][1]["start_new_session"] is True and "creationflags" not in calls[0][1]


def test_an_invalid_config_raises_the_reason_before_anything_starts(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\ntotal_steps: [", encoding="utf-8")
    with pytest.raises(TrainingError, match="bad.yaml"):
        TrainingJob(bad, tmp_path / "logs", tmp_path / "runs")
    with pytest.raises(TrainingError, match="no existe"):
        TrainingJob(tmp_path / "nada.yaml", tmp_path / "logs", tmp_path / "runs")
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("name: x\nplayers: 3\n", encoding="utf-8")
    with pytest.raises(TrainingError, match="players"):
        TrainingJob(unknown, tmp_path / "logs", tmp_path / "runs")


def test_a_failed_launch_becomes_a_training_error(config, tmp_path):
    def spawn(command, **kwargs):
        raise OSError("sin permiso")

    job = TrainingJob(config, tmp_path / "logs", tmp_path / "runs", spawn=spawn)
    with pytest.raises(TrainingError, match="sin permiso"):
        job.start()


def test_the_run_folder_comes_from_the_log_and_progress_from_its_metrics(config, tmp_path):
    job, _, _ = make_job(config, tmp_path)
    job.start()
    assert job.run_paths() is None and job.steps() == 0
    run_dir = tmp_path / "runs" / "2026-11-06_1200_mini"
    run_dir.mkdir(parents=True)
    job.log_path.write_text(f"Run: {run_dir}\n  1024 pasos\n", encoding="utf-8")
    assert job.run_paths().root == run_dir
    (run_dir / "metrics.csv").write_text("steps,time_s\n512,1\n120000,9\n", encoding="utf-8")
    assert job.steps() == 120000
    job.clock.now += 192
    assert job.status_line() == "Entrenando mini · 120k/400k · 3:12"
    assert job.log_tail(1) == ["1024 pasos"]


def test_stop_creates_the_file_and_terminates_only_after_the_grace_period(config, tmp_path):
    clock = Clock()
    job, _, process = make_job(config, tmp_path, clock=clock)
    job.start()
    job.request_stop()
    assert job.stop_path.exists() and job.stopping
    clock.now += STOP_WAIT_SECONDS - 1
    job.update()
    assert not process.terminated  # the trainer still has time to save its checkpoint
    clock.now += 2
    job.update()
    assert process.terminated
    assert "error" in job.status_line() or "terminado" in job.status_line()


def test_stop_does_nothing_when_the_trainer_already_ended(config, tmp_path):
    job, _, process = make_job(config, tmp_path)
    job.start()
    process.code = 0
    job.request_stop()
    assert not job.stop_path.exists()
    assert "terminado" in job.status_line()


def test_last_metrics_steps_reads_only_the_tail(tmp_path):
    path = tmp_path / "metrics.csv"
    assert last_metrics_steps(path) == 0
    rows = "\n".join(f"{i * 512},{i}" for i in range(1, 2000))
    path.write_text("steps,time_s\n" + rows + "\n", encoding="utf-8")
    assert last_metrics_steps(path) == 1999 * 512


def test_live_run_needs_a_metrics_file_touched_in_the_last_minute(tmp_path):
    run = tmp_path / "runs" / "2026-01-01_0000_x"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text("name: x\n", encoding="utf-8")
    assert live_run(tmp_path / "runs") is None  # no metrics yet
    metrics = run / "metrics.csv"
    metrics.write_text("steps\n1\n", encoding="utf-8")
    now = metrics.stat().st_mtime
    assert live_run(tmp_path / "runs", now + 30).root == run
    assert live_run(tmp_path / "runs", now + 120) is None


# -- adopting a trainer launched by an earlier session of the menu -----------------------------


def write_job_file(path, **overrides):
    data = {
        "format": 1,
        "pid": os.getpid(),
        "started_at": process_started_at(os.getpid()) or time.time(),
        "name": "previo",
        "total_steps": 400000,
        "config": "configs/previo.yaml",
        "log": str(path.parent / "train-previo.log"),
        "stop": str(path.parent / "train-previo.stop"),
        "runs_dir": str(path.parent / "runs"),
        "run": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**data, **overrides}), encoding="utf-8")
    return path


def test_launching_writes_the_job_file_and_the_run_is_added_when_known(config, tmp_path):
    job, _, process = make_job(config, tmp_path)
    process.pid = 4321
    job.start()
    data = json.loads(job.job_file.read_text(encoding="utf-8"))
    assert job.job_file == tmp_path / "logs" / JOB_FILE_NAME
    assert data["pid"] == 4321 and data["name"] == "mini" and data["total_steps"] == 400000
    assert data["stop"] == str(job.stop_path) and data["config"] == str(config)
    assert data["run"] is None
    run_dir = tmp_path / "runs" / "2026-11-06_1200_mini"
    run_dir.mkdir(parents=True)
    job.log_path.write_text(f"Run: {run_dir}\n", encoding="utf-8")
    job.run_paths()
    assert json.loads(job.job_file.read_text(encoding="utf-8"))["run"] == str(run_dir)
    assert not list((tmp_path / "logs").glob("*.tmp"))  # written atomically, nothing left over


def test_a_new_session_adopts_a_trainer_that_is_still_alive_and_can_stop_it(tmp_path):
    run_dir = tmp_path / "runs" / "2026-11-06_1200_previo"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.csv").write_text("steps,time_s\n128000,9\n", encoding="utf-8")
    path = write_job_file(tmp_path / "logs" / JOB_FILE_NAME, run=str(run_dir))
    job = TrainingJob.adopt(tmp_path / "logs", tmp_path / "runs")
    assert job is not None and job.running and not job.stopping
    assert job.config.name == "previo" and job.run_paths().root == run_dir
    assert job.steps() == 128000
    assert job.status_line().startswith("Entrenando previo · 128k/400k")
    job.request_stop()  # writes the stop file the trainer is watching
    assert job.stop_path.exists() and job.stopping and "Parando" in job.status_line()
    assert path.exists()


def test_an_adopted_job_that_was_already_stopping_says_so(tmp_path):
    path = write_job_file(tmp_path / "logs" / JOB_FILE_NAME)
    Path(json.loads(path.read_text(encoding="utf-8"))["stop"]).write_text("stop", encoding="utf-8")
    job = TrainingJob.adopt(tmp_path / "logs")
    assert job.stopping


def test_a_job_file_of_a_dead_process_is_ignored_and_removed(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    path = write_job_file(tmp_path / "logs" / JOB_FILE_NAME, pid=child.pid)
    assert TrainingJob.adopt(tmp_path / "logs") is None
    assert not path.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="the start time check is Windows only")
def test_a_reused_pid_is_not_adopted(tmp_path):
    path = write_job_file(tmp_path / "logs" / JOB_FILE_NAME, started_at=time.time() - 86_400)
    assert TrainingJob.adopt(tmp_path / "logs") is None  # alive, but not the trainer
    assert not path.exists()


@pytest.mark.parametrize("text", ["", "{", "[1]", '{"pid": "x"}', "null"])
def test_a_corrupt_job_file_is_ignored(tmp_path, text):
    path = tmp_path / "logs" / JOB_FILE_NAME
    path.parent.mkdir()
    path.write_text(text, encoding="utf-8")
    assert TrainingJob.adopt(tmp_path / "logs") is None
    assert not path.exists()
    assert TrainingJob.adopt(tmp_path / "logs") is None  # and without a file: nothing


def test_the_job_and_stop_files_go_away_when_the_trainer_ends(config, tmp_path):
    job, _, process = make_job(config, tmp_path)
    job.start()
    job.request_stop()
    assert job.job_file.exists() and job.stop_path.exists()
    process.code = 0
    job.update()
    assert not job.job_file.exists() and not job.stop_path.exists()
    assert TrainingJob.adopt(tmp_path / "logs") is None
