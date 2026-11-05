"""Which tracks are used for what.

Training draws seeds from ``[0, TRAIN_SEED_LIMIT)``. Evaluation uses a fixed list of
seeds outside that range, so evaluation tracks are never seen during training.
"""

from __future__ import annotations

import numpy as np

from kartlab.sim.showcase import SHOWCASE, showcase_track
from kartlab.sim.track import Track, TrackParams, generate_track
from kartlab.sim.trackfile import load_track_file

TRAIN_SEED_LIMIT = 1_000_000
EVAL_SEEDS: tuple[int, ...] = tuple(range(TRAIN_SEED_LIMIT, TRAIN_SEED_LIMIT + 20))
EVAL_PARAMS = TrackParams()


def is_eval_seed(seed: int) -> bool:
    return seed >= TRAIN_SEED_LIMIT


def sample_train_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, TRAIN_SEED_LIMIT))


def train_track(seed: int, params: TrackParams | None = None) -> Track:
    if is_eval_seed(seed):
        raise ValueError(f"seed {seed} is reserved for evaluation")
    return generate_track(seed, params)


def eval_track(index: int) -> Track:
    """Evaluation track ``index`` (0-based), always generated with ``EVAL_PARAMS``."""
    return generate_track(EVAL_SEEDS[index], EVAL_PARAMS, name=f"Evaluación {index + 1}")


def resolve_track(choice: str) -> Track:
    """A showcase key (``harbor``, ``canyon``, ``plateau``), ``eval:N`` (1-based), a
    training seed or the path of a track file (``*.json``, validated on load: raises
    ``TrackFileError`` if it is invalid)."""
    if choice.lower().endswith(".json"):
        return load_track_file(choice)
    if choice in SHOWCASE:
        return showcase_track(choice)
    if choice.startswith("eval:"):
        return eval_track(int(choice.removeprefix("eval:")) - 1)
    return train_track(int(choice))
