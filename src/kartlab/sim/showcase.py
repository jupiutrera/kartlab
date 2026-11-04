"""Hand-designed showcase tracks for demos. Original layouts drawn for this project."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kartlab.sim.track import Track, TrackParams, filleted_polygon, finish_centerline


@dataclass(frozen=True)
class ShowcaseDesign:
    name: str  # display name
    corners: tuple[tuple[float, float, float], ...]  # (x, y, radius) in metres, driving order
    width: float  # asphalt width (m)


SHOWCASE: dict[str, ShowcaseDesign] = {
    # Long main straight into a hairpin, a back straight broken by a bay, wide final turn.
    "harbor": ShowcaseDesign(
        name="Puerto",
        corners=(
            (0, 0, 32),
            (330, 0, 26),
            (360, 150, 17),
            (225, 110, 30),
            (140, 175, 40),
            (15, 165, 34),
        ),
        width=13.0,
    ),
    # Technical: a pair of switchbacks up one side and a quick return down the other.
    "canyon": ShowcaseDesign(
        name="Cañón",
        corners=(
            (0, 0, 28),
            (215, -20, 18),
            (235, 65, 16),
            (150, 85, 18),
            (170, 155, 16),
            (265, 165, 22),
            (245, 250, 24),
            (20, 235, 32),
        ),
        width=11.0,
    ),
    # Fast and flowing: big sweepers, one tight corner and a gentle bay.
    "plateau": ShowcaseDesign(
        name="Meseta",
        corners=(
            (0, 0, 60),
            (380, -40, 45),
            (470, 120, 70),
            (330, 260, 24),
            (200, 220, 50),
            (60, 280, 55),
            (-60, 140, 80),
        ),
        width=15.0,
    ),
}


def showcase_track(key: str) -> Track:
    design = SHOWCASE[key]
    corners = np.array(design.corners, dtype=np.float64)
    path, used = filleted_polygon(corners[:, :2], corners[:, 2])
    limits = TrackParams(width=design.width)
    points = None
    if np.allclose(used, corners[:, 2]):
        points = finish_centerline(path, design.width, limits.min_radius, limits.clearance)
    if points is None:
        raise ValueError(f"showcase design {key!r} breaks the radius or clearance limits")
    return Track(points, design.width, name=design.name, clearance=limits.clearance)
