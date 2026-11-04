"""Fast end-to-end check that the whole stack is installed and usable (run in CI).

Grows with each phase: check_env (phase 2), a few hundred PPO steps (phase 3)
and one frame rendered without a window (phase 4).
"""

import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

LAYERS = [
    "kartlab",
    "kartlab.common",
    "kartlab.sim",
    "kartlab.env",
    "kartlab.train",
    "kartlab.render",
    "kartlab.show",
]
STACK = [
    "numpy",
    "yaml",
    "gymnasium",
    "stable_baselines3",
    "torch",
    "tensorboard",
    "pygame",
    "imageio_ffmpeg",
]


@pytest.mark.parametrize("module", LAYERS + STACK)
def test_import(module):
    importlib.import_module(module)


def test_ffmpeg_binary_is_bundled():
    import imageio_ffmpeg

    assert Path(imageio_ffmpeg.get_ffmpeg_exe()).is_file()


def test_pygame_software_surface_shares_memory_with_numpy():
    import numpy as np
    import pygame

    surface = pygame.Surface((320, 180))
    surface.fill((10, 20, 30))
    pixels = pygame.surfarray.pixels3d(surface)
    assert pixels.shape == (320, 180, 3)
    assert np.all(pixels[0, 0] == (10, 20, 30))
