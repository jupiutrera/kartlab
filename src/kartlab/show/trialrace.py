"""The race format of the manual-driving time trial: a 3-2-1 countdown with the controls
frozen, a fixed number of laps and a results table with every lap, the total and whether a
lap record fell.

The lap record and the ghost of the best lap are untouched (``sim/timetrial.py``); this only
frames a session around them. Nothing here draws or imports pygame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kartlab.render.topdown import format_time
from kartlab.sim.kart import DT

TRIAL_LAPS = 3
COUNTDOWN_FRAMES = 180  # 3 s at 60 steps per second
BANNER_SECONDS = 2.5
RESULTS_DELAY = 2.2  # after the finish line: the "¡META!" banner first, then the table

LAST_LAP = "ÚLTIMA VUELTA"
FINISH = "¡META!"


@dataclass
class TrialSession:
    """One run of ``laps`` laps. ``advance`` is called once per physics step."""

    laps: int = TRIAL_LAPS
    frame: int = 0  # steps since the countdown began
    steps: int = 0  # steps since the start (controls live)
    lap_times: list[float] = field(default_factory=list)
    new_record: bool = False  # a lap of this session beat it
    valid: bool = True  # False once the autopilot drove (no records then)
    banner: tuple[str, int] | None = None  # text and the step it appeared on
    finished_step: int | None = None

    @property
    def counting(self) -> bool:
        return self.frame < COUNTDOWN_FRAMES

    @property
    def finished(self) -> bool:
        return self.finished_step is not None

    @property
    def state(self) -> str:
        return "countdown" if self.counting else "finished" if self.finished else "racing"

    @property
    def time(self) -> float:
        """Race time (s): zero during the countdown, then the time since the start."""
        return self.steps * DT

    @property
    def total(self) -> float:
        return sum(self.lap_times)

    @property
    def laps_done(self) -> int:
        return len(self.lap_times)

    @property
    def lap_number(self) -> int:
        """The lap being driven, 1-based, never beyond the last."""
        return min(self.laps_done + 1, self.laps)

    def advance(self) -> str:
        """Count one physics step; returns the state the step belongs to."""
        if self.counting:
            self.frame += 1
            return "countdown"
        self.frame += 1
        self.steps += 1
        return self.state

    def on_lap(self, lap_time: float, record: bool, valid: bool) -> None:
        """A lap was completed (the event of ``TimeTrial.update``)."""
        if self.finished:
            return
        self.lap_times.append(lap_time)
        self.new_record = self.new_record or record
        self.valid = self.valid and valid
        if self.laps_done >= self.laps:
            self.finished_step = self.steps
            self.banner = (FINISH, self.steps)
        elif self.laps_done == self.laps - 1 and self.laps > 1:
            self.banner = (LAST_LAP, self.steps)

    def banner_age(self) -> float | None:
        return None if self.banner is None else (self.steps - self.banner[1]) * DT

    def results_age(self) -> float:
        """Seconds the results table has been due (negative: not yet)."""
        if self.finished_step is None:
            return -1.0
        return (self.steps - self.finished_step) * DT - RESULTS_DELAY

    def result_lines(self, record: float | None) -> list[tuple[str, str]]:
        """Rows of the table as (text, kind), kind being "lap", "best", "total", "record",
        "info" or "dim". ``record`` is the lap record now on disk (after this session)."""
        best = min(self.lap_times) if self.lap_times else None
        rows = []
        for i, lap in enumerate(self.lap_times, start=1):
            mark = "  mejor" if lap == best and len(self.lap_times) > 1 else ""
            rows.append((f"Vuelta {i}  {format_time(lap)}{mark}", "best" if mark else "lap"))
        rows.append((f"Total  {format_time(self.total)}", "total"))
        if not self.valid:
            rows.append(("Sin récord con piloto automático", "dim"))
        elif self.new_record:
            rows.append((f"Nuevo récord {format_time(best)}", "record"))
        elif record is not None:
            rows.append((f"Récord {format_time(record)}", "dim"))
        else:
            rows.append(("Sin récord", "dim"))
        return rows
