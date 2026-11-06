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


def fit_rect(window: tuple[int, int], base: tuple[int, int] = WORLD_SIZE) -> pygame.Rect:
    """Where the 320x180 canvas goes in a window: the largest integer scale that fits,
    centred (black bars around it)."""
    k = max(1, min(window[0] // base[0], window[1] // base[1]))
    w, h = base[0] * k, base[1] * k
    return pygame.Rect((window[0] - w) // 2, (window[1] - h) // 2, w, h)


def present(canvas: pygame.Surface, screen: pygame.Surface) -> pygame.Rect:
    """Magnify a 320x180 canvas into the window by an integer factor, without smoothing.
    Returns the rectangle it covers."""
    rect = fit_rect(screen.get_size(), canvas.get_size())
    if rect.size != screen.get_size():
        screen.fill((0, 0, 0))
    target = screen.subsurface(rect)
    if rect.size == canvas.get_size():
        target.blit(canvas, (0, 0))
    else:
        pygame.transform.scale(canvas, rect.size, target)
    return rect


def to_logical(
    pos: tuple[int, int], window: tuple[int, int], base: tuple[int, int] = WORLD_SIZE
) -> tuple[int, int]:
    """The canvas pixel under a window position (it may lie outside the canvas)."""
    rect = fit_rect(window, base)
    k = max(1, rect.w // base[0])
    return (pos[0] - rect.x) // k, (pos[1] - rect.y) // k
