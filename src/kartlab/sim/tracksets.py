"""Which tracks are used for what.

Training draws seeds from ``[0, TRAIN_SEED_LIMIT)``. Evaluation uses a fixed list of
seeds outside that range, so evaluation tracks are never seen during training.
"""

from __future__ import annotations

import numpy as np

from kartlab.sim.track import Track, TrackParams, generate_track

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
