"""The game palette (Resurrect 64 by Kerrie Lake) and everything derived from it.

The world is drawn with palette indices. Day, sunset and night are index remaps
computed once: each colour goes to the palette entry closest to its tinted version, so
every variant stays inside the same 64 colours. Fog works the same way: one
index-to-index table per fog level, combined with the variant, so applying both costs
a single lookup per pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from kartlab.common.paths import ASSETS_DIR

PALETTE_FILE = ASSETS_DIR / "palettes" / "resurrect-64.hex"
VARIANTS = ("day", "sunset", "night")
FOG_LEVELS = 8
TRANSPARENT = 255  # index meaning "no pixel" in sprite and sky layers


class Ink:
    """Palette indices by role."""

    BLACK = 0
    ASPHALT_DARK = 0
    ASPHALT = 1
    ASPHALT_LIGHT = 2
    STONE = 6
    STONE_LIGHT = 7
    MIST = 8
    WHITE = 9
    RED_DARK = 10
    RED = 11
    KERB_RED = 15
    ORANGE = 16
    AMBER = 17
    YELLOW = 18
    BROWN_DARK = 24
    BROWN = 21
    GRASS_DARK = 29
    GRASS = 30
    GRASS_LIGHT = 31
    LEAF_LIGHT = 32
    MOSS = 36
    MOSS_DARK = 35
    NAVY = 44
    DUSK = 45
    SKY_TOP = 46
    SKY = 47
    SKY_LOW = 48
    PURPLE = 52
    PINK = 60


# Kart body ramps (dark, main, light): one per agent version in a race (eight at most).
KART_RAMPS: tuple[tuple[int, int, int], ...] = (
    (21, 16, 23),  # orange
    (46, 47, 48),  # blue
    (51, 52, 53),  # purple
    (40, 41, 42),  # teal
    (59, 60, 61),  # pink
    (17, 18, 28),  # yellow
    (6, 8, 9),  # white
    (14, 15, 61),  # red
)

# Per variant: lighting of the world as (multiplier, light colour, light amount). The
# light colour shifts hues too (greens turn mossy at sunset), which a plain multiplier
# cannot do with 64 colours. Then the fog colour and the sky gradient (top to horizon).
LIGHTING = {
    "day": ((1.0, 1.0, 1.0), (0, 0, 0), 0.0),
    "sunset": ((0.80, 0.70, 0.62), (250, 130, 70), 0.22),
    "night": ((0.35, 0.40, 0.60), (50, 51, 83), 0.20),
}
FOG_COLOR = {"day": Ink.SKY_LOW, "sunset": 62, "night": Ink.NAVY}
SKY_GRADIENT = {
    "day": (Ink.SKY_TOP, Ink.SKY, Ink.SKY_LOW),
    "sunset": (50, 56, 61, 62, 63),
    "night": (0, 49, Ink.NAVY),
}


def load_palette_rgb() -> np.ndarray:
    lines = PALETTE_FILE.read_text(encoding="utf-8").split()
    return np.array([[int(h[k : k + 2], 16) for k in (0, 2, 4)] for h in lines], dtype=np.uint8)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """CIE L*a*b* (D65) of sRGB colours in 0-255, for perceptual nearest-colour search."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = linear @ m.T / np.array([0.95047, 1.0, 1.08883])
    d = 6 / 29
    f = np.where(xyz > d**3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


@dataclass(frozen=True)
class Palette:
    rgb: np.ndarray  # (64, 3) uint8
    lab: np.ndarray  # (64, 3) float

    def __len__(self) -> int:
        return len(self.rgb)

    def nearest(self, rgb: np.ndarray) -> np.ndarray:
        """Index of the closest palette colour (CIELAB distance) for each RGB colour."""
        lab = srgb_to_lab(rgb)
        flat = lab.reshape(-1, 3)
        d2 = ((flat[:, None, :] - self.lab[None, :, :]) ** 2).sum(axis=2)
        return d2.argmin(axis=1).astype(np.uint8).reshape(lab.shape[:-1])

    @property
    def quantizer(self) -> np.ndarray:
        """32x32x32 lookup table RGB (5 bits per channel) -> nearest index."""
        return _quantizer(self)

    def quantize(self, rgb: np.ndarray) -> np.ndarray:
        q = np.asarray(rgb, dtype=np.uint8) >> 3
        return self.quantizer[q[..., 0], q[..., 1], q[..., 2]]

    def remap(self, variant: str) -> np.ndarray:
        """Index -> index of the variant (day is the identity)."""
        return _remap(self, variant)

    def fog_table(self, variant: str) -> np.ndarray:
        """(FOG_LEVELS * 64,) table: entry ``level * 64 + index`` is the final index of a
        day-palette ``index`` in ``variant`` under fog ``level`` (0 clear, last = fog)."""
        return _fog_table(self, variant)


@lru_cache(maxsize=1)
def game_palette() -> Palette:
    rgb = load_palette_rgb()
    return Palette(rgb, srgb_to_lab(rgb))


# Palettes are frozen dataclasses holding arrays: cache by identity, not by value.
_cache: dict[tuple[int, str, str], np.ndarray] = {}


def _cached(palette: Palette, kind: str, variant: str, build) -> np.ndarray:
    key = (id(palette), kind, variant)
    if key not in _cache:
        _cache[key] = build()
    return _cache[key]


def _quantizer(palette: Palette) -> np.ndarray:
    def build():
        levels = (np.arange(32) << 3) + 4
        grid = np.stack(np.meshgrid(levels, levels, levels, indexing="ij"), axis=-1)
        return palette.nearest(grid.reshape(-1, 3)).reshape(32, 32, 32)

    return _cached(palette, "quantizer", "", build)


def _remap(palette: Palette, variant: str) -> np.ndarray:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")

    def build():
        if variant == "day":
            return np.arange(len(palette), dtype=np.uint8)
        multiplier, light, amount = LIGHTING[variant]
        tinted = np.clip(palette.rgb * np.array(multiplier) + np.array(light) * amount, 0, 255)
        return palette.nearest(tinted)

    return _cached(palette, "remap", variant, build)


def _fog_table(palette: Palette, variant: str) -> np.ndarray:
    def build():
        remap = palette.remap(variant)
        fog = palette.rgb[FOG_COLOR[variant]].astype(np.float64)
        base = palette.rgb[remap].astype(np.float64)
        table = np.empty((FOG_LEVELS, len(palette)), dtype=np.uint8)
        for level in range(FOG_LEVELS):
            alpha = level / (FOG_LEVELS - 1)
            table[level] = palette.nearest(base * (1 - alpha) + fog * alpha)
        table[0] = remap
        table[-1] = FOG_COLOR[variant]
        return table.reshape(-1)

    return _cached(palette, "fog", variant, build)
