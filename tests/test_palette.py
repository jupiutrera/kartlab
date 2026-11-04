import numpy as np
import pytest

from kartlab.render.palette import (
    FOG_COLOR,
    FOG_LEVELS,
    KART_RAMPS,
    SKY_GRADIENT,
    VARIANTS,
    Ink,
    game_palette,
)


@pytest.fixture(scope="module")
def palette():
    return game_palette()


def test_palette_has_64_distinct_colours(palette):
    assert palette.rgb.shape == (64, 3) and palette.rgb.dtype == np.uint8
    assert len({tuple(c) for c in palette.rgb}) == 64


def test_nearest_finds_exact_colours_and_close_ones(palette):
    assert np.array_equal(palette.nearest(palette.rgb), np.arange(64))
    near_white = np.array([[250, 252, 251]], dtype=np.uint8)
    assert palette.nearest(near_white)[0] == Ink.WHITE


def test_quantizer_agrees_with_exact_search_on_palette_colours(palette):
    assert np.mean(palette.quantize(palette.rgb) == np.arange(64)) > 0.9


@pytest.mark.parametrize("variant", VARIANTS)
def test_variants_stay_inside_the_palette(variant, palette):
    remap = palette.remap(variant)
    assert remap.shape == (64,) and remap.max() < 64
    if variant == "day":
        assert np.array_equal(remap, np.arange(64))


def test_night_is_darker_than_day(palette):
    lightness = palette.lab[:, 0]
    night = palette.remap("night")
    assert lightness[night].mean() < lightness.mean() - 15


@pytest.mark.parametrize("variant", VARIANTS)
def test_fog_goes_from_clear_to_the_fog_colour(variant, palette):
    table = palette.fog_table(variant).reshape(FOG_LEVELS, 64)
    assert np.array_equal(table[0], palette.remap(variant))
    assert set(table[-1]) == {FOG_COLOR[variant]}
    fog_lab = palette.lab[FOG_COLOR[variant]]
    distance = [np.linalg.norm(palette.lab[table[k]] - fog_lab, axis=1).mean() for k in range(8)]
    assert distance[0] > distance[FOG_LEVELS // 2] > distance[-1]


def test_role_indices_are_valid():
    roles = [v for k, v in vars(Ink).items() if not k.startswith("_")]
    used = roles + [i for ramp in KART_RAMPS for i in ramp]
    used += [i for grad in SKY_GRADIENT.values() for i in grad]
    assert all(0 <= i < 64 for i in used)
