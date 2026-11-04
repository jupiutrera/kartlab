"""File helpers: UTF-8 text, YAML and atomic writes.

Atomic means another process (the live viewer) never sees a half-written file: data
goes to a temporary file in the same folder, which then replaces the target. On
Windows the replace fails while a reader has the target open, so it is retried.
"""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

REPLACE_RETRIES = 50
REPLACE_WAIT = 0.05  # s between retries


def replace_with_retry(source: Path, target: Path) -> None:
    for attempt in range(REPLACE_RETRIES):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == REPLACE_RETRIES - 1:
                raise
            time.sleep(REPLACE_WAIT)


def atomic_write(target: Path, write: Callable[[Path], None]) -> None:
    """Call ``write(tmp_path)`` and atomically move the result to ``target``."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        write(tmp)
        replace_with_retry(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def write_text_atomic(target: Path, text: str) -> None:
    atomic_write(target, lambda tmp: tmp.write_text(text, encoding="utf-8", newline="\n"))


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_yaml_atomic(target: Path, data: Any) -> None:
    write_text_atomic(target, yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
