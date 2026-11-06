"""Pure pieces of the race against the AI: difficulty levels, entrants and the standings.

Nothing here draws or touches pygame; ``scenes.HumanRaceScene`` puts it on screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kartlab.render.topdown import format_time

DEFAULT_LAPS = 3
DEFAULT_RIVALS = 2
MAX_RIVALS = 3
RACE_TIMEOUT = 25.0  # the race ends this long after the first AI finishes, if you have not (s)


@dataclass(frozen=True)
class Difficulty:
    key: str
    label: str
    speed: float  # speed limiter of the AI as a share of the kart's cruise speed (1 = none)


# A pedal cap (65 to 100 % throttle) barely changes the lap because the top speed is almost
# flat there, so the level is a speed limiter instead; measured lap times are in the PR.
DIFFICULTIES: tuple[Difficulty, ...] = (
    Difficulty("facil", "Fácil", 0.76),
    Difficulty("normal", "Normal", 0.84),
    Difficulty("dificil", "Difícil", 0.93),
    Difficulty("campeon", "Campeón", 1.0),
)
DEFAULT_DIFFICULTY = "normal"


def difficulty_of(key: str) -> Difficulty:
    for level in DIFFICULTIES:
        if level.key == key:
            return level
    raise KeyError(f"dificultad desconocida: {key}")


@dataclass
class Entrant:
    """One participant: the human or an AI kart. ``distance`` is the race distance in metres
    from the start line (``length * laps`` or more is the finish)."""

    name: str
    ramp: int
    human: bool
    length: float
    laps: int
    distance: float = 0.0
    finish_time: float | None = None
    marks: list[float] = field(default_factory=list)  # race time at each lap line crossed

    @property
    def finished(self) -> bool:
        return self.finish_time is not None

    @property
    def laps_done(self) -> int:
        return len(self.marks)

    def update(self, distance: float, time: float) -> int:
        """Take the new race distance at race ``time``; returns the laps completed now (0 or 1,
        2 if a step jumped over two lines, which only a teleport could do)."""
        self.distance = distance
        crossed = 0
        while len(self.marks) < self.laps and distance >= self.length * (len(self.marks) + 1):
            self.marks.append(time)
            crossed += 1
        if self.finish_time is None and len(self.marks) >= self.laps:
            self.finish_time = self.marks[-1]
        return crossed

    @property
    def lap_times(self) -> list[float]:
        starts = [0.0, *self.marks[:-1]]
        return [b - a for a, b in zip(starts, self.marks, strict=False)]

    @property
    def best_lap(self) -> float | None:
        times = self.lap_times
        return min(times) if times else None

    def lap_time_at(self, time: float) -> float:
        """Time spent on the current lap at race ``time``; once the race is finished, the clock
        stays on the last lap."""
        if self.finish_time is not None:
            return self.lap_times[-1]
        return max(time - (self.marks[-1] if self.marks else 0.0), 0.0)


def classify(entrants: list[Entrant]) -> list[Entrant]:
    """Standings: those who finished by finish time, then the rest by distance driven."""
    done = sorted((e for e in entrants if e.finished), key=lambda e: e.finish_time)
    rest = sorted((e for e in entrants if not e.finished), key=lambda e: -e.distance)
    return done + rest


def place_of(entrants: list[Entrant], entrant: Entrant) -> int:
    """1-based place of ``entrant`` in the standings."""
    return classify(entrants).index(entrant) + 1


def result_lines(entrants: list[Entrant]) -> list[tuple[str, bool]]:
    """The results table as (text, is_human) rows: place, name, total time (or the distance
    left for those who had not finished) and best lap."""
    rows = []
    for i, e in enumerate(classify(entrants), start=1):
        if e.finished:
            total = format_time(e.finish_time)
        else:
            total = f"faltan {max(e.length * e.laps - e.distance, 0.0):.0f} m"
        best = format_time(e.best_lap) if e.best_lap is not None else "-:--,--"
        rows.append((f"{i}º  {e.name}  {total}  mejor {best}", e.human))
    return rows
