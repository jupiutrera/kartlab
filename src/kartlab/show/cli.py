"""Argument checks shared by the viewers and the video tool: clear messages, no tracebacks."""

from __future__ import annotations

from pathlib import Path

from kartlab.sim.track import Track
from kartlab.sim.trackfile import TrackFileError
from kartlab.sim.tracksets import resolve_track
from kartlab.train.run import RunPaths


def resolve_track_or_exit(choice: str) -> Track:
    """``resolve_track`` that exits with a Spanish message for a choice that does not exist."""
    try:
        return resolve_track(choice)
    except TrackFileError as error:
        raise SystemExit(f"Circuito no válido:\n{error}") from None
    except FileNotFoundError:
        raise SystemExit(f"El fichero de circuito «{choice}» no existe.") from None
    except (ValueError, IndexError, KeyError):
        raise SystemExit(
            f"Circuito no válido: «{choice}». Usa harbor, canyon, plateau, eval:1-20, "
            "una semilla de entrenamiento o un fichero .json."
        ) from None


def existing_run_or_exit(path: Path) -> RunPaths:
    """A run folder that exists and has a config, or a clear exit."""
    run = RunPaths(path)
    if not path.is_dir():
        raise SystemExit(f"El run {path} no existe.")
    if not run.config.exists():
        raise SystemExit(f"{path} no es un run: no tiene config.yaml")
    return run
