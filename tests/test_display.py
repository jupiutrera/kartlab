import subprocess
import sys

import pytest

from kartlab.render.display import pick_scale


@pytest.mark.parametrize(
    "screen, scale",
    [
        ((1366, 768), 4),  # target laptop panel: 1280x720
        ((1920, 1080), 5),  # windowed 1600x900; fullscreen uses the whole screen
        ((1280, 720), 3),
        ((2560, 1440), 7),
        ((3840, 2160), 11),
        ((800, 600), 2),
        ((320, 180), 1),
    ],
)
def test_window_uses_the_largest_integer_scale_that_fits(screen, scale):
    assert pick_scale(screen) == scale


@pytest.mark.skipif(sys.platform != "win32", reason="DPI awareness is a Windows setting")
def test_process_becomes_dpi_aware_on_windows():
    code = (
        "import ctypes\n"
        "from kartlab.render.display import enable_dpi_awareness\n"
        "assert enable_dpi_awareness()\n"
        "assert ctypes.windll.user32.IsProcessDPIAware()\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
