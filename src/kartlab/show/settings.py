"""The main menu's settings: ``settings.json`` at the repository root (ignored by git).

Every value is validated on load; a missing, corrupt or hand-edited file never breaks the
menu, it just falls back to the defaults (key by key).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

from kartlab.common.io import write_text_atomic
from kartlab.common.paths import REPO_ROOT

SETTINGS_PATH = REPO_ROOT / "settings.json"
SCALES = (None, 1, 2, 3, 4, 5, 6)  # None: the largest that fits the screen
LIGHTS = ("day", "sunset", "night")
CAMERAS = ("auto", "chase", "broadcast", "top")
DEFAULT_TRACK = "harbor"
KART_COUNT = 8
RACE_DIFFICULTIES = ("facil", "normal", "dificil", "campeon")  # as in humanrace.DIFFICULTIES
RACE_LAPS = (1, 3, 5)
RACE_RIVALS = (1, 2, 3)


@dataclass
class Settings:
    scale: int | None = None  # integer scale over 320x180 (None: automatic)
    fullscreen: bool = False
    light: str = "day"
    camera: str = "auto"  # auto: each mode picks its own camera
    last_track: str = DEFAULT_TRACK  # "harbor", "file:tracks/x.json", "random", "eval:7"...
    last_run: str = ""  # folder name of the run the AI modes use ("": the latest)
    kart_ramp: int = 0  # the player's kart colour (an index of KART_RAMPS)
    # The race against the AI (the options screen of "Carrera contra la IA").
    race_difficulty: str = "normal"
    race_rivals: int = 2
    race_laps: int = 3
    race_collisions: bool = True

    # -- loading and saving --------------------------------------------------------------

    @classmethod
    def load(cls, path: Path = SETTINGS_PATH) -> Settings:
        """The saved settings, or the defaults for anything missing or invalid."""
        settings = cls()
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, RecursionError):  # no file, not UTF-8, not JSON, too deep
            return settings
        if not isinstance(data, dict):
            return settings
        if data.get("scale") == "auto":
            data["scale"] = None
        for field in fields(cls):
            if field.name in data and settings.valid(field.name, data[field.name]):
                setattr(settings, field.name, data[field.name])
        return settings

    @staticmethod
    def valid(name: str, value: object) -> bool:
        if name == "scale":
            return value is None or (type(value) is int and value in SCALES)
        if name == "fullscreen":
            return isinstance(value, bool)
        if name == "light":
            return value in LIGHTS
        if name == "camera":
            return value in CAMERAS
        if name in ("last_track", "last_run"):
            return isinstance(value, str) and len(value) < 300
        if name == "kart_ramp":
            return type(value) is int and 0 <= value < KART_COUNT
        if name == "race_difficulty":
            return value in RACE_DIFFICULTIES
        if name == "race_rivals":
            return type(value) is int and value in RACE_RIVALS
        if name == "race_laps":
            return type(value) is int and value in RACE_LAPS
        if name == "race_collisions":
            return isinstance(value, bool)
        return False

    def to_json(self) -> str:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["scale"] = "auto" if self.scale is None else self.scale
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"

    def save(self, path: Path = SETTINGS_PATH) -> bool:
        """Write atomically; ``False`` if the disk refuses (the menu keeps working)."""
        try:
            write_text_atomic(Path(path), self.to_json())
        except OSError:
            return False
        return True
