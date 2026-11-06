"""A grid of tracks in one PNG, for a quick look at the generator.

Run with ``uv run python scripts/track_gallery.py --out gallery.png``.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pygame

from kartlab.render import pixelui as ui
from kartlab.render.topdown import GRASS, TopDownView
from kartlab.sim.showcase import SHOWCASE, showcase_track
from kartlab.sim.track import Track, TrackParams
from kartlab.sim.tracksets import train_track


def render_gallery(
    tracks: Sequence[Track], columns: int = 4, cell: tuple[int, int] = (400, 300)
) -> pygame.Surface:
    scale = max(1, cell[1] // 150)  # the pixel font, at a whole scale
    rows = math.ceil(len(tracks) / columns)
    surface = pygame.Surface((columns * cell[0], rows * cell[1]))
    surface.fill(GRASS)
    for k, track in enumerate(tracks):
        rect = pygame.Rect((k % columns) * cell[0], (k // columns) * cell[1], *cell)
        TopDownView(track, cell).draw(surface.subsurface(rect), [], (0.0, 0.0), follow=False)
        tightest = 1.0 / float(np.max(np.abs(track.curvature)))
        label = f"{track.name}, {track.length:.0f} m, R mín {tightest:.0f} m"
        ui.text(surface, label, rect.x + 8, rect.y + 8, ui.TEXT, scale, outline=ui.NIGHT)
    return surface


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dibuja una galería de circuitos en un PNG.")
    parser.add_argument("--out", type=Path, required=True, help="PNG de salida")
    parser.add_argument("--count", type=int, default=8, help="circuitos aleatorios")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--curviness", type=float, default=TrackParams.curviness)
    parser.add_argument("--showcase", action="store_true", help="incluye los circuitos escaparate")
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args(argv)

    params = TrackParams(curviness=args.curviness)
    tracks = [showcase_track(key) for key in SHOWCASE] if args.showcase else []
    tracks += [train_track(args.start_seed + i, params) for i in range(args.count)]
    surface = render_gallery(tracks, columns=args.columns)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(args.out))
    pygame.quit()
