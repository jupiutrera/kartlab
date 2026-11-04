"""Window creation: DPI awareness, integer scaling of the 320x180 world, fullscreen."""

from __future__ import annotations

import sys

import pygame

WORLD_SIZE = (320, 180)  # resolution the world is drawn at
WINDOW_MARGIN = (32, 48)  # room left for window borders and the taskbar (px)
FALLBACK_DESKTOP = (1366, 768)


def enable_dpi_awareness() -> bool:
    """Declare the process DPI-aware so Windows does not blur a scaled window.

    Must run before ``pygame.display.init()``. Returns True if the process is aware.
    """
    if sys.platform != "win32":
        return False
    import ctypes

    user32 = ctypes.windll.user32
    if user32.IsProcessDPIAware():
        return True
    try:
        per_monitor_v2 = ctypes.c_void_p(-4)
        if user32.SetProcessDpiAwarenessContext(per_monitor_v2):
            return True
    except (AttributeError, OSError):
        pass
    try:
        if ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0:
            return True
    except (AttributeError, OSError):
        pass
    return bool(user32.SetProcessDPIAware())


def pick_scale(
    screen_size: tuple[int, int],
    base: tuple[int, int] = WORLD_SIZE,
    margin: tuple[int, int] = WINDOW_MARGIN,
) -> int:
    """Largest integer scale of ``base`` that fits on the screen with a margin."""
    width, height = screen_size
    return max(1, min((width - margin[0]) // base[0], (height - margin[1]) // base[1]))


def desktop_size() -> tuple[int, int]:
    sizes = pygame.display.get_desktop_sizes()
    return tuple(sizes[0]) if sizes else FALLBACK_DESKTOP


def open_window(title: str, scale: int | None = None, fullscreen: bool = False) -> pygame.Surface:
    """Create the (software) window: integer scale of the world size, or fullscreen."""
    enable_dpi_awareness()
    pygame.display.init()
    pygame.display.set_caption(title)
    return set_window_mode(scale, fullscreen)


def set_window_mode(scale: int | None = None, fullscreen: bool = False) -> pygame.Surface:
    if fullscreen:
        return pygame.display.set_mode(desktop_size(), pygame.FULLSCREEN)
    k = scale or pick_scale(desktop_size())
    return pygame.display.set_mode((WORLD_SIZE[0] * k, WORLD_SIZE[1] * k))
