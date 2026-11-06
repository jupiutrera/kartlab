"""Launching a training run from the main menu, as a separate low-priority process.

The menu never trains by itself: ``TrainingJob.start`` is only called when the player asks
for it. The trainer is an ordinary ``python -m kartlab.train.train`` subprocess (so closing
the menu does not stop it), its output goes to a file in ``logs/`` (never a pipe, which
would fill up and block it) and "Parar" creates a stop file that ``StopFileCallback``
watches. Nothing here imports torch or Stable-Baselines3.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from pathlib import Path

from kartlab.common.paths import REPO_ROOT

LOGS_DIR = REPO_ROOT / "logs"
CONFIGS_DIR = REPO_ROOT / "configs"
ESTIMATE_STEPS_PER_SECOND = 1600  # measured with scripts/benchmark_train.py on the target laptop
LONG_MINUTES = 30  # beyond this the menu asks twice
STOP_WAIT_SECONDS = 25.0  # after the stop file appears, then the process is terminated
ALIVE_SECONDS = 60.0  # metrics.csv touched this recently: a training is running

BELOW_NORMAL_PRIORITY_CLASS = 0x4000
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200


class TrainingError(Exception):
    """A training that cannot start; the message is shown to the player as it is."""


@dataclass(frozen=True)
class ConfigInfo:
    path: Path
    name: str
    total_steps: int
    comment: str  # first comment line of the file
    error: str | None = None  # why the file cannot be used

    @property
    def minutes(self) -> int:
        return estimate_minutes(self.total_steps)

    @property
    def estimate(self) -> str:
        return estimate_text(self.total_steps)

    @property
    def is_long(self) -> bool:
        return self.minutes > LONG_MINUTES


def estimate_minutes(total_steps: int) -> int:
    """Static estimate (rounded up, at least 1): steps / 1600 per second."""
    return max(1, ceil(total_steps / ESTIMATE_STEPS_PER_SECOND / 60))


def estimate_text(total_steps: int) -> str:
    return f"unos {estimate_minutes(total_steps)} min"


def first_comment(path: Path) -> str:
    """The first ``# ...`` line of a YAML file, without the marker."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
            if stripped:
                break
    except (OSError, UnicodeDecodeError):
        pass
    return ""


def describe_error(error: Exception) -> str:
    """One useful line for a config that does not load: where the YAML breaks (line and
    column) and why, or the first line of the message (unknown option, wrong type...)."""
    import yaml

    if isinstance(error, yaml.MarkedYAMLError):
        mark = error.problem_mark or error.context_mark
        reason = error.problem or error.context or "YAML no válido"
        where = f"línea {mark.line + 1}, columna {mark.column + 1}: " if mark is not None else ""
        return where + reason
    return (str(error).strip().splitlines() or [type(error).__name__])[0]


def load_config(path: Path):
    """``TrainConfig.from_yaml`` with every failure turned into a ``TrainingError``."""
    from kartlab.train.config import TrainConfig

    try:
        return TrainConfig.from_yaml(Path(path))
    except FileNotFoundError:
        raise TrainingError(f"{path}: no existe el fichero") from None
    except (OSError, UnicodeDecodeError) as error:
        raise TrainingError(f"{path}: no se puede leer ({error})") from None
    except Exception as error:  # YAML errors, unknown options, wrong types...
        raise TrainingError(
            f"{Path(path).name}: configuración no válida\n{describe_error(error)}"
        ) from None


def list_configs(configs_dir: Path = CONFIGS_DIR) -> list[ConfigInfo]:
    """Every ``configs/*.yaml``. An invalid one is still listed, with its error."""
    found = []
    for path in sorted(Path(configs_dir).glob("*.yaml")):
        comment = first_comment(path)
        try:
            config = load_config(path)
        except TrainingError as error:
            found.append(ConfigInfo(path, path.stem, 0, comment, str(error)))
        else:
            found.append(ConfigInfo(path, config.name, config.total_steps, comment))
    return found


def creation_flags(platform: str = sys.platform) -> int:
    if platform == "win32":
        return BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    return 0


def last_metrics_steps(path: Path) -> int:
    """Steps of the last row of ``metrics.csv`` (0 if there is none yet)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 2048))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return 0
    for line in reversed(lines):
        head = line.split(",", 1)[0]
        if head.isdigit():
            return int(head)
    return 0


def format_clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def live_run(runs_dir: Path, now: float | None = None):
    """The latest run whose ``metrics.csv`` changed in the last minute (a training is going
    on, started from here or from a terminal), or ``None``."""
    from kartlab.train.run import latest_run

    run = latest_run(runs_dir)
    if run is None:
        return None
    try:
        age = (now if now is not None else time.time()) - run.metrics.stat().st_mtime
    except OSError:
        return None
    return run if age <= ALIVE_SECONDS else None


Spawn = Callable[..., "subprocess.Popen"]


class TrainingJob:
    """One training subprocess started by the menu."""

    def __init__(
        self,
        config_path: Path,
        logs_dir: Path = LOGS_DIR,
        runs_dir: Path | None = None,
        spawn: Spawn = subprocess.Popen,
        clock: Callable[[], float] = time.time,
        platform: str = sys.platform,
    ) -> None:
        from kartlab.train.run import RUNS_DIR

        self.config_path = Path(config_path)
        self.logs_dir = Path(logs_dir)
        self.runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
        self.spawn = spawn
        self.clock = clock
        self.platform = platform
        self.config = load_config(self.config_path)  # raises TrainingError with the reason
        stamp = datetime.fromtimestamp(clock()).strftime("%Y-%m-%d_%H%M%S")
        self.log_path = self.logs_dir / f"train-{stamp}.log"
        self.stop_path = self.logs_dir / f"train-{stamp}.stop"
        self.process: subprocess.Popen | None = None
        self.started_at = 0.0
        self.stop_requested_at: float | None = None
        self.terminated = False
        self._run = None

    @property
    def command(self) -> Sequence[str]:
        return [
            sys.executable,
            "-m",
            "kartlab.train.train",
            str(self.config_path),
            "--runs-dir",
            str(self.runs_dir),
            "--stop-file",
            str(self.stop_path),
        ]

    def start(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.stop_path.unlink(missing_ok=True)
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
        options = {"cwd": str(REPO_ROOT), "env": env, "stdin": subprocess.DEVNULL}
        if self.platform == "win32":
            options["creationflags"] = creation_flags(self.platform)
        else:
            options["start_new_session"] = True
        try:
            with open(self.log_path, "wb") as log:  # a file, never a pipe: it cannot fill up
                self.process = self.spawn(
                    self.command, stdout=log, stderr=subprocess.STDOUT, **options
                )
        except OSError as error:
            raise TrainingError(f"No se pudo lanzar el entrenamiento: {error}") from None
        self.started_at = self.clock()

    # -- state ---------------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def stopping(self) -> bool:
        return self.running and self.stop_requested_at is not None

    @property
    def exit_code(self) -> int | None:
        return None if self.process is None else self.process.poll()

    def run_paths(self):
        """The run folder: the ``Run: ...`` line the trainer prints first, or, failing that,
        the newest run created after the launch."""
        if self._run is not None:
            return self._run
        from kartlab.train.run import RunPaths, latest_run

        try:
            head = self.log_path.read_bytes()[:4096].decode("utf-8", errors="replace")
        except OSError:
            head = ""
        for line in head.splitlines():
            if line.startswith("Run:"):
                folder = Path(line.removeprefix("Run:").strip())
                if folder.is_dir():
                    self._run = RunPaths(folder)
                    return self._run
        run = latest_run(self.runs_dir)
        if run is not None and run.root.stat().st_mtime >= self.started_at - 2:
            self._run = run
        return self._run

    def steps(self) -> int:
        run = self.run_paths()
        return 0 if run is None else last_metrics_steps(run.metrics)

    def elapsed(self) -> float:
        return self.clock() - self.started_at

    def status_line(self) -> str:
        """«Entrenando demo-live · 120k/400k · 3:12»."""
        from kartlab.show.drivers import format_steps

        total = self.config.total_steps
        if self.stopping:
            return f"Parando {self.config.name}…"
        if not self.running:
            code = self.exit_code
            ended = "terminado" if code == 0 else f"terminado con error ({code})"
            return f"Entrenamiento {self.config.name} {ended}"
        return (
            f"Entrenando {self.config.name} · {format_steps(self.steps())}/{format_steps(total)}"
            f" · {format_clock(self.elapsed())}"
        )

    def log_tail(self, lines: int = 3) -> list[str]:
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 1024))
                text = f.read().decode("utf-8", errors="replace")
        except OSError:
            return []
        return [line.strip() for line in text.splitlines() if line.strip()][-lines:]

    # -- stopping ------------------------------------------------------------------------

    def request_stop(self) -> None:
        """Ask the trainer to stop cleanly: it saves a checkpoint and the summary first."""
        if not self.running or self.stop_requested_at is not None:
            return
        self.stop_path.parent.mkdir(parents=True, exist_ok=True)
        self.stop_path.write_text("stop\n", encoding="utf-8")
        self.stop_requested_at = self.clock()

    def update(self) -> None:
        """Once per frame: if the trainer ignores the stop file for too long, end it."""
        if (
            self.stop_requested_at is not None
            and self.running
            and not self.terminated
            and self.clock() - self.stop_requested_at > STOP_WAIT_SECONDS
        ):
            self.process.terminate()
            self.terminated = True
