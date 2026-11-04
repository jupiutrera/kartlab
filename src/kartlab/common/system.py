"""Process settings for training: low priority and a bounded number of threads."""

from __future__ import annotations

import os
import sys

BELOW_NORMAL_PRIORITY_CLASS = 0x4000
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _kernel32():
    """kernel32 with explicit types: a HANDLE is 64-bit and must not pass as a C int."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetPriorityClass.argtypes = [wintypes.HANDLE]
    kernel32.GetPriorityClass.restype = wintypes.DWORD
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.SetPriorityClass.restype = wintypes.BOOL
    return kernel32


def lower_priority() -> None:
    """Run this process below normal priority so the viewer stays smooth.

    Child processes started afterwards inherit it: Windows passes BELOW_NORMAL on to
    children, and on Linux the nice value is inherited.
    """
    if sys.platform == "win32":
        kernel32 = _kernel32()
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
    elif hasattr(os, "nice"):
        os.nice(10)


def is_below_normal_priority() -> bool:
    if sys.platform == "win32":
        kernel32 = _kernel32()
        priority = kernel32.GetPriorityClass(kernel32.GetCurrentProcess())
        return priority == BELOW_NORMAL_PRIORITY_CLASS
    return os.nice(0) > 0


def limit_child_threads(count: int = 1) -> None:
    """Make processes started from now on use ``count`` math-library threads each.

    Env workers only do small numpy operations; extra threads just compete for cores.
    """
    for name in THREAD_VARIABLES:
        os.environ[name] = str(count)
