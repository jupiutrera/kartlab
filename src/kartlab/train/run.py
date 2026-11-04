"""Run folders: ``runs/<date>_<time>_<name>/`` with config, commit, checkpoints and metrics.

Layout::

    config.yaml          training configuration
    meta.yaml            commit, versions, machine and start time
    metrics.csv          one row per rollout (read by the live viewer and the videos)
    eval.csv             one row per evaluation on the reserved tracks
    summary.yaml         written at the end: steps, duration, first lap, best evaluation
    best_model.zip       best evaluated policy (and best_vecnormalize.pkl)
    checkpoints/         model_<steps>.zip, vecnormalize_<steps>.pkl and latest.json
    tensorboard/
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from kartlab.common.io import write_yaml_atomic
from kartlab.common.paths import REPO_ROOT

RUNS_DIR = REPO_ROOT / "runs"
CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.zip$")


@dataclass(frozen=True)
class Checkpoint:
    steps: int
    model: Path
    vecnormalize: Path


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def meta(self) -> Path:
        return self.root / "meta.yaml"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.csv"

    @property
    def evals(self) -> Path:
        return self.root / "eval.csv"

    @property
    def summary(self) -> Path:
        return self.root / "summary.yaml"

    @property
    def best_model(self) -> Path:
        return self.root / "best_model.zip"

    @property
    def best_vecnormalize(self) -> Path:
        return self.root / "best_vecnormalize.pkl"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def latest(self) -> Path:
        return self.checkpoints / "latest.json"

    @property
    def tensorboard(self) -> Path:
        return self.root / "tensorboard"

    def checkpoint(self, steps: int) -> Checkpoint:
        return Checkpoint(
            steps,
            self.checkpoints / f"model_{steps:09d}.zip",
            self.checkpoints / f"vecnormalize_{steps:09d}.pkl",
        )

    def list_checkpoints(self) -> list[Checkpoint]:
        found = []
        for path in self.checkpoints.glob("model_*.zip"):
            match = CHECKPOINT_PATTERN.search(path.name)
            if match:
                found.append(self.checkpoint(int(match.group(1))))
        return sorted(found, key=lambda c: c.steps)

    def read_latest(self) -> Checkpoint | None:
        """The newest complete checkpoint, as announced by ``latest.json``."""
        try:
            data = json.loads(self.latest.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, PermissionError):
            return None
        return self.checkpoint(int(data["steps"]))


def git_info() -> dict[str, object]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8"
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "commit": git("rev-parse", "HEAD") or None,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def create_run(config_dict: dict, name: str, runs_dir: Path = RUNS_DIR) -> RunPaths:
    """Create ``<runs_dir>/<date>_<time>_<name>/`` with its config and metadata."""
    import stable_baselines3
    import torch

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    root = runs_dir / f"{stamp}_{name}"
    suffix = 2
    while root.exists():
        root = runs_dir / f"{stamp}_{name}-{suffix}"
        suffix += 1
    run = RunPaths(root)
    run.checkpoints.mkdir(parents=True)
    write_yaml_atomic(run.config, config_dict)
    write_yaml_atomic(
        run.meta,
        {
            **git_info(),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "stable_baselines3": str(stable_baselines3.__version__),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    )
    return run


def latest_run(runs_dir: Path = RUNS_DIR) -> RunPaths | None:
    """The most recently created run (folder names start with the date and time)."""
    runs = sorted(p for p in runs_dir.glob("*") if (p / "config.yaml").exists())
    return RunPaths(runs[-1]) if runs else None
