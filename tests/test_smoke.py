"""Fast end-to-end check that the whole stack is installed and usable (run in CI).

Grows with each phase: manual driving without a window (phase 1), check_env and
the benchmark (phase 2), a few hundred PPO steps (phase 3) and one pseudo-3D frame
(phase 4).
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


def test_drive_app_runs_without_a_window_and_saves_a_screenshot(tmp_path):
    from kartlab.show.drive import main

    out = tmp_path / "drive.png"
    main(
        [
            "--track",
            "canyon",
            "--autopilot",
            "--scale",
            "2",
            "--frames",
            "90",
            "--screenshot",
            str(out),
        ]
    )
    assert out.stat().st_size > 0


def test_track_gallery_writes_a_png(tmp_path):
    from kartlab.show.gallery import main

    out = tmp_path / "gallery.png"
    main(["--count", "2", "--showcase", "--out", str(out)])
    assert out.stat().st_size > 0


def test_env_passes_check_env():
    from stable_baselines3.common.env_checker import check_env

    from kartlab.env.kart_env import KartEnv

    check_env(KartEnv(), warn=True)


def test_benchmark_runs_one_env_and_spawned_subprocesses(tmp_path):
    from kartlab.env.benchmark import bench_vec, main
    from kartlab.env.kart_env import EnvConfig

    out = tmp_path / "bench.json"
    main(["--steps", "200", "--resets", "2", "--no-vec", "--json", str(out)])
    assert out.stat().st_size > 0
    assert bench_vec(EnvConfig(), "subproc", 2, 20)["steps_per_s"] > 0
