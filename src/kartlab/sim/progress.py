"""Progress along the centerline and lap counting.

Progress is the *net* distance driven along the centerline since the start: driving
backwards subtracts, so going back and forth over the line never adds up, and a lap
only counts the first time the net distance reaches another multiple of the length.
"""

from __future__ import annotations


class Progress:
    def __init__(self, track_length: float, start_s: float) -> None:
        self.length = track_length
        self.s = start_s % track_length
        self.distance = 0.0  # net distance along the centerline since the start (m)
        self.laps = 0  # completed laps; never decreases

    def update(self, s: float) -> float:
        """Move to arc position ``s`` and return the net distance gained (may be negative)."""
        half = self.length / 2
        delta = (s - self.s + half) % self.length - half
        self.s = s % self.length
        self.distance += delta
        self.laps = max(self.laps, int(self.distance // self.length))
        return delta

    def rebase(self, s: float) -> None:
        """Jump to ``s`` without counting the distance (e.g. a shortcut across the grass)."""
        self.s = s % self.length

    @property
    def lap_fraction(self) -> float:
        """Fraction of the current lap already driven, in [0, 1)."""
        return (self.distance % self.length) / self.length
