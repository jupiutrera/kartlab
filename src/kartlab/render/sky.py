"""Sky: dithered gradient plus panorama layers that scroll when the camera turns.

Layers are panoramas of palette indices one full turn wide (``TRANSPARENT`` where
empty). Each scrolls at its own rate with the camera heading, which gives a sense of
depth (parallax); the clouds also drift slowly. Everything is generated once per
variant; a frame only gathers columns.
"""

from __future__ import annotations

import math

import numpy as np

from kartlab.render.floor import Projection
from kartlab.render.palette import (
    FOG_COLOR,
    FOG_LEVELS,
    SKY_GRADIENT,
    TRANSPARENT,
    Ink,
    Palette,
)
from kartlab.render.texture import BAYER4, hash_noise

CLOUD_DRIFT = 2.0  # px per second


def _dithered_gradient(stops: tuple[int, ...], rows: int, width: int) -> np.ndarray:
    """Rows top to bottom blending through ``stops`` with an ordered dither."""
    t = np.linspace(0.0, 1.0, max(rows, 1))[:, None] * (len(stops) - 1)
    dither = BAYER4[np.arange(rows)[:, None] % 4, np.arange(width)[None, :] % 4]
    k = np.clip(np.floor(t + dither), 0, len(stops) - 1).astype(int)
    return np.asarray(stops, dtype=np.uint8)[k]


def _ridge(width: int, seed: int, octaves: tuple[tuple[int, float], ...]) -> np.ndarray:
    """Periodic 1D height profile in [0, 1] (sum of sines with random phases)."""
    rng = np.random.default_rng(seed)
    x = np.arange(width) / width * 2 * np.pi
    h = np.zeros(width)
    for freq, amp in octaves:
        h += amp * np.sin(freq * x + rng.uniform(0, 2 * np.pi))
    h -= h.min()
    return h / max(h.max(), 1e-9)


class Sky:
    def __init__(
        self,
        projection: Projection,
        palette: Palette,
        variant: str,
        floor_row: int,
        seed: int = 0,
    ):
        self.projection = projection
        width, height = projection.width, projection.height
        horizon = projection.horizon
        self.horizon_row = int(math.floor(min(max(horizon, 0), height)))
        # The sky reaches down to the first floor row: no gap at the horizon.
        self.rows = min(max(floor_row, self.horizon_row), height)
        self.pano_width = round(width * 2 * math.pi / projection.lens.fov)
        remap = palette.remap(variant)
        fog = palette.fog_table(variant).reshape(FOG_LEVELS, -1)
        rows = self.rows
        self.gradient = _dithered_gradient(
            (*SKY_GRADIENT[variant], FOG_COLOR[variant]), rows, width
        )

        p = self.pano_width
        self.layers: list[tuple[np.ndarray, float, float]] = []  # (pano, rate, drift)
        if rows > 0:
            far = self._mountains(rows, p, seed, 0.42, fog, (Ink.DUSK, Ink.STONE), 4)
            near = self._mountains(rows, p, seed + 1, 0.22, fog, (Ink.MOSS_DARK, Ink.MOSS), 2)
            clouds = self._clouds(rows, p, seed + 2, remap)
            extras = self._extras(rows, p, variant, seed + 3)
            self.layers = [(extras, 0.9, 0.0), (clouds, 0.7, CLOUD_DRIFT), (far, 0.85, 0.0),
                           (near, 1.0, 0.0)]  # fmt: skip

    @staticmethod
    def _mountains(rows, width, seed, peak, fog, colours, haze) -> np.ndarray:
        """A ridge rising from the floor's far edge. The slope facing the sun is lighter
        and the base fades into the fog (aerial perspective, dithered)."""
        dark, light = colours
        ridge = _ridge(width, seed, ((3, 1.0), (7, 0.45), (17, 0.2), (41, 0.08)))
        top = rows - 1 - (0.15 + ridge) / 1.15 * peak * rows
        y = np.arange(rows)[:, None]
        x = np.arange(width)[None, :]
        inside = y >= top[None, :]
        lit = np.gradient(ridge)[None, :] > 0
        dither = BAYER4[y % 4, x % 4]
        near_top = (y - top[None, :]) < 3 + 4 * dither
        colour = np.where(lit & near_top, light, dark)
        depth = np.clip((y - top[None, :]) / np.maximum(rows - top[None, :], 1), 0, 1)
        level = np.clip(np.floor(haze + depth * (FOG_LEVELS - 1 - haze) + dither - 0.5), 0,
                        FOG_LEVELS - 1).astype(int)  # fmt: skip
        layer = np.full((rows, width), TRANSPARENT, dtype=np.uint8)
        layer[inside] = fog[level, colour][inside]
        return layer

    @staticmethod
    def _clouds(rows: int, width: int, seed: int, remap: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(seed)
        layer = np.full((rows, width), TRANSPARENT, dtype=np.uint8)
        yy, xx = np.mgrid[0:rows, 0:width]
        for _ in range(max(width // 90, 3)):
            cx = rng.uniform(0, width)
            cy = rng.uniform(0.15, 0.55) * rows
            for _ in range(rng.integers(3, 6)):
                ex = cx + rng.uniform(-18, 18)
                ey = cy + rng.uniform(-3, 2)
                rx, ry = rng.uniform(8, 18), rng.uniform(3, 6)
                dx = (xx - ex + width / 2) % width - width / 2  # wraps around the panorama
                d = (dx / rx) ** 2 + ((yy - ey) / ry) ** 2
                layer[d <= 1.0] = remap[Ink.WHITE]
                shade = (d <= 1.0) & (yy > ey + ry * 0.35)
                layer[shade] = remap[Ink.MIST]
        return layer

    @staticmethod
    def _extras(rows: int, width: int, variant: str, seed: int) -> np.ndarray:
        layer = np.full((rows, width), TRANSPARENT, dtype=np.uint8)
        yy, xx = np.mgrid[0:rows, 0:width]
        if variant == "night":
            stars = hash_noise(xx, yy, seed) < 0.012
            layer[stars & (yy < rows * 0.7)] = Ink.WHITE
            moon = (xx - width * 0.3) ** 2 + (yy - rows * 0.3) ** 2 <= 16
            layer[moon] = Ink.MIST
        elif variant == "sunset":
            sun = (xx - width * 0.62) ** 2 + (yy - rows * 0.78) ** 2
            layer[sun <= 110] = Ink.AMBER
            layer[sun <= 60] = Ink.YELLOW
        return layer

    def render(self, out: np.ndarray, heading: float, time_s: float = 0.0) -> None:
        rows = self.gradient.shape[0]
        if rows == 0:
            return
        region = out[:rows]
        region[:] = self.gradient
        width = out.shape[1]
        cols = np.arange(width)
        turn = -heading / (2 * math.pi) * self.pano_width  # turning left moves the sky right
        for pano, rate, drift in self.layers:
            offset = int(round(turn * rate + drift * time_s)) % self.pano_width
            band = pano[:, (offset + cols) % self.pano_width]
            np.copyto(region, band, where=band != TRANSPARENT)
