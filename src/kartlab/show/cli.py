"""Argument checks shared by the viewers and the video tool: clear messages, no tracebacks."""

from __future__ import annotations

from pathlib import Path

from kartlab.render.palette import KART_RAMPS, game_palette
from kartlab.sim.track import Track
from kartlab.sim.trackfile import TrackFileError
from kartlab.sim.tracksets import resolve_track
from kartlab.train.run import RunPaths

# Names of the eight kart colours, in the order of ``KART_RAMPS`` (``KartVisual.ramp``).
KART_COLOR_NAMES = ("Naranja", "Azul", "Morado", "Turquesa", "Rosa", "Amarillo", "Blanco", "Rojo")


class NoCheckpoints(Exception):
    """A run without checkpoints: the apps raise it; each ``main()`` turns it into an exit."""


def kart_color_rgb(ramp: int) -> tuple[int, int, int]:
    """Screen colour of a kart ramp (its main shade), for the top-down view and the menus."""
    rgb = game_palette().rgb[KART_RAMPS[ramp % len(KART_RAMPS)][1]]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def parse_kart_color(text: str) -> int:
    """A kart colour by name (``azul``) or index (``1``); ``ValueError`` if it is neither."""
    wanted = text.strip().lower()
    if wanted.isdigit() and int(wanted) < len(KART_COLOR_NAMES):
        return int(wanted)
    for index, name in enumerate(KART_COLOR_NAMES):
        if name.lower() == wanted:
            return index
    raise ValueError(
        f"Color de kart no válido: «{text}». Usa un nombre ({', '.join(KART_COLOR_NAMES)}) "
        f"o un número de 0 a {len(KART_COLOR_NAMES) - 1}."
    )


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
