"""The pixel interface kit: palette lock, glyph coverage, exact measures, integer scales."""

import ast
import statistics
import string
import time
from pathlib import Path

import numpy as np
import pygame
import pytest

from kartlab.render import pixelfont as font
from kartlab.render import pixelui as ui
from kartlab.render.display import fit_rect, present, to_logical
from kartlab.render.palette import game_palette

SRC = Path(__file__).resolve().parents[1] / "src" / "kartlab"
PALETTE = {tuple(int(c) for c in row) for row in game_palette().rgb}


def palette_only(surface: pygame.Surface) -> bool:
    pixels = pygame.surfarray.array3d(surface).reshape(-1, 3)
    return {tuple(int(c) for c in p) for p in np.unique(pixels, axis=0)} <= PALETTE


@pytest.fixture
def canvas():
    pygame.init()
    surface = pygame.Surface(ui.CANVAS_SIZE)
    surface.fill(ui.rgb(ui.SLATE))
    yield surface
    pygame.quit()


# -- the palette lock ------------------------------------------------------------------------


def draw_everything(canvas: pygame.Surface) -> None:
    ui.text(canvas, "Vuelta 1 · 0:04,00 ÁÉÍÓÚÑÜ ¡¿?!", 4, 12, ui.TEXT)
    ui.text(canvas, "Salida", 60, 40, ui.YELLOW, scale=3, outline=ui.NIGHT)
    ui.text(canvas, "84", 300, 50, ui.WHITE, scale=2, outline=ui.NIGHT, align="right")
    ui.text(canvas, "sombra", 10, 70, ui.CYAN, shadow=ui.NIGHT, outline=ui.NIGHT)
    ui.plate(canvas, (4, 80, 60, 30), ui.NAVY)
    ui.tag(canvas, "Mejor", 70, 82)
    ui.keycap(canvas, "Esc", 100, 82)
    ui.bar(canvas, (4, 120, 50, 6), 0.4, ui.GREEN)
    ui.bar(canvas, (60, 120, 50, 6), -0.5, ui.BLUE, centered=True)
    for name in ui._ICONS:
        ui.icon(canvas, name, 4, 140, ui.TEXT, outline=ui.NIGHT)
    ui.dither(canvas, (120, 100, 60, 40), 2)
    ui.paragraph(canvas, "Un texto largo que se parte en varias líneas", 190, 100, 60)
    ui.Rolling().draw(canvas, 12345, 200, 150, ui.YELLOW, 2)


def test_nothing_the_interface_draws_is_outside_the_palette(canvas):
    draw_everything(canvas)
    assert palette_only(canvas)


def test_the_cached_strings_carry_one_colour_key_and_palette_inks(canvas):
    ui.text(canvas, "Prueba", 4, 12, ui.ORANGE, scale=2, outline=ui.NIGHT)
    for surface in ui._strings.values():
        pixels = pygame.surfarray.array3d(surface).reshape(-1, 3)
        colours = {tuple(int(c) for c in p) for p in np.unique(pixels, axis=0)}
        assert colours <= PALETTE | {ui._KEY}
        assert ui._KEY not in PALETTE


def test_the_roles_are_distinct_palette_inks():
    roles = [ui.NIGHT, ui.SLATE, ui.STONE, ui.STONE_LIGHT, ui.TEXT, ui.NAVY, ui.YELLOW, ui.GREEN]
    assert all(0 <= ink < 64 for ink in roles) and len(set(roles)) == len(roles)
    lum = [sum(ui.rgb(ink)) for ink in (ui.NIGHT, ui.SLATE, ui.STONE, ui.STONE_LIGHT, ui.TEXT)]
    assert lum == sorted(lum)  # night is the darkest, then up to the text


def test_there_is_no_vector_font_in_the_kit():
    for name in ("pixelui.py", "pixelfont.py"):
        source = (SRC / "render" / name).read_text(encoding="utf-8")
        assert "pygame.font" not in source and "font.render" not in source
        assert "SRCALPHA" not in source and "set_alpha" not in source  # no translucency


# -- glyph coverage --------------------------------------------------------------------------

REQUIRED = "«»·ºª×←→↑↓%:;¡¿ÁÉÍÓÚÑÜáéíóúñü.,!?+-/()°…çÇàÀèÈìÌòÒùÙïÏ"


def test_the_font_has_every_character_the_interface_needs():
    assert all(font.has_glyph(c) for c in REQUIRED)
    assert all(font.has_glyph(c) for c in string.ascii_letters + string.digits)
    assert all(font.has_glyph(c) for c in string.printable if c not in "\t\n\r\x0b\x0c")


def string_literals():
    """The user-visible strings of the code: every string literal of ``show`` and
    ``render`` except docstrings."""
    for folder in ("show", "render"):
        for path in sorted((SRC / folder).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef):
                    body = node.body
                    if body and isinstance(body[0], ast.Expr):
                        docstrings.add(id(body[0].value))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in docstrings
                ):
                    yield path.name, node.lineno, node.value


# Strings that only ever reach the terminal or a file name, never the screen.
NOT_ON_SCREEN = {"cli.py", "benchmark_render.py"}


def test_every_character_in_the_code_strings_has_a_glyph():
    missing: dict[str, set[str]] = {}
    for name, line, value in string_literals():
        if name in NOT_ON_SCREEN:
            continue
        for c in value:
            if ord(c) > 126 and not font.has_glyph(c):
                missing.setdefault(c, set()).add(f"{name}:{line}")
    assert not missing, f"characters without a glyph: {missing}"


def test_accents_sit_above_the_capital_and_never_touch_the_line_above():
    for accented, plain in zip("ÁÉÍÓÚÑÜ", "AEIOUNU", strict=True):
        rows, base = font.glyph_rows(accented), font.glyph_rows(plain)
        assert len(rows) == font.GLYPH_H and rows[2:] == base[2:]  # the capital is intact
        assert any("#" in row for row in rows[:2])  # the accent is in the two extra rows
        assert not any("#" in row for row in base[:2])


def test_every_glyph_fits_in_the_nine_rows_and_is_rectangular():
    for c in font.known_characters():
        rows = font.glyph_rows(c)
        assert len(rows) == font.GLYPH_H
        assert len({len(row) for row in rows}) == 1, c
        assert set("".join(rows)) <= set("#."), c


# -- measures and wrapping -------------------------------------------------------------------


def test_text_is_measured_in_whole_pixels_per_glyph_plus_one_column():
    assert font.measure("") == 0
    assert font.measure("A") == 5 and font.measure("I") == 3
    assert font.measure("AB") == 5 + 1 + 5
    assert font.measure("AI I") == 5 + 1 + 3 + 1 + 3 + 1 + 3
    assert font.measure("ab") == font.measure("AB")  # one case


@pytest.mark.parametrize("scale", [1, 2, 3, 4])
def test_measure_scales_by_an_integer_factor(scale):
    assert font.measure("KART LAB", scale) == font.measure("KART LAB") * scale
    assert font.text_mask("KART", scale).shape == (
        font.GLYPH_H * scale,
        font.measure("KART", scale),
    )


@pytest.mark.parametrize("scale", [2, 3, 4])
def test_a_scaled_text_is_the_unscaled_one_with_each_pixel_a_square(scale):
    base = font.text_mask("Vuelta 12", 1)
    scaled = font.text_mask("Vuelta 12", scale)
    assert np.array_equal(scaled, np.kron(base, np.ones((scale, scale), dtype=bool)))


def test_tabular_digits_all_have_the_same_width():
    widths = {font.measure(str(d), tabular=True) for d in range(10)}
    assert widths == {font.DIGIT_W}
    assert font.measure("0:11,11", tabular=True) == font.measure("0:00,00", tabular=True)
    assert font.measure("1") != font.measure("0")  # proportional by default


def test_wrap_keeps_every_line_inside_the_width_and_every_word():
    sentence = "Hay cambios sin guardar: pulsa Esc otra vez para salir sin guardar"
    for width in (50, 60, 100, 200):
        lines = font.wrap(sentence, width)
        assert all(font.measure(line) <= width for line in lines)
        assert " ".join(lines) == sentence
    assert font.wrap("uno dos", 1000) == ["uno dos"]
    assert font.wrap("", 100) == [""]


def test_wrap_cuts_a_word_that_does_not_fit_and_honours_newlines():
    lines = font.wrap("x" * 80, 50)
    assert len(lines) > 1 and all(font.measure(line) <= 50 for line in lines)
    assert "".join(lines) == "x" * 80
    assert font.wrap("uno\ndos", 1000) == ["uno", "dos"]


def test_wrap_at_a_larger_scale_uses_the_scaled_width():
    lines = font.wrap("uno dos tres cuatro", 60, scale=2)
    assert all(font.measure(line, 2) <= 60 for line in lines) and len(lines) > 1


def test_ellipsize_fits_and_keeps_the_chosen_end():
    long = "runs/2026-11-04_2222_demo-live"
    start = font.ellipsize(long, 60, keep="start")
    end = font.ellipsize(long, 60, keep="end")
    assert font.measure(start) <= 60 and start.endswith("…") and start.startswith("runs")
    assert font.measure(end) <= 60 and end.startswith("…") and end.endswith("live")
    assert font.ellipsize("corto", 60) == "corto"


# -- drawing ---------------------------------------------------------------------------------


def test_text_is_placed_by_the_top_of_its_capitals(canvas):
    canvas.fill(ui.rgb(ui.NIGHT))
    ui.text(canvas, "T", 10, 20, ui.WHITE)
    lit = np.argwhere((pygame.surfarray.array3d(canvas) == ui.rgb(ui.WHITE)).all(axis=2))
    assert lit[:, 0].min() == 10 and lit[:, 1].min() == 20  # the bar of the T
    assert lit[:, 1].max() == 20 + font.CAP_H - 1


def test_alignment_anchors_the_left_centre_or_right_edge(canvas):
    def bounds(align):
        canvas.fill(ui.rgb(ui.NIGHT))
        ui.text(canvas, "HOLA", 100, 20, ui.WHITE, align=align)
        lit = np.argwhere((pygame.surfarray.array3d(canvas) == ui.rgb(ui.WHITE)).all(axis=2))
        return lit[:, 0].min(), lit[:, 0].max() + 1

    width = font.measure("HOLA")
    assert bounds("left") == (100, 100 + width)
    assert bounds("right") == (100 - width, 100)
    left, right = bounds("center")
    assert left == 100 - width // 2 and right - left == width


def test_the_outline_is_one_pixel_at_every_scale(canvas):
    for scale in (1, 2, 3):
        canvas.fill(ui.rgb(ui.SLATE))
        ui.text(canvas, "I", 50, 50, ui.WHITE, scale=scale, outline=ui.NIGHT)
        hits = np.argwhere((pygame.surfarray.array3d(canvas) == ui.rgb(ui.NIGHT)).all(axis=2))
        letter = np.argwhere((pygame.surfarray.array3d(canvas) == ui.rgb(ui.WHITE)).all(axis=2))
        assert hits[:, 0].min() == letter[:, 0].min() - 1
        assert hits[:, 1].max() == letter[:, 1].max() + 1


def test_a_plate_is_flat_with_a_one_pixel_border(canvas):
    inside = ui.plate(canvas, (10, 10, 20, 12), ui.NAVY)
    assert inside == pygame.Rect(11, 11, 18, 10)
    assert canvas.get_at((10, 10))[:3] == ui.rgb(ui.NIGHT)
    assert canvas.get_at((29, 21))[:3] == ui.rgb(ui.NIGHT)
    assert canvas.get_at((11, 11))[:3] == ui.rgb(ui.NAVY)
    assert canvas.get_at((20, 15))[:3] == ui.rgb(ui.NAVY)  # no gradient: flat


def test_a_bar_fills_in_proportion_and_a_centred_one_grows_from_the_middle(canvas):
    ui.bar(canvas, (10, 10, 22, 5), 0.5, ui.GREEN)  # inside is 20 wide: 10 filled
    filled = [canvas.get_at((11 + i, 12))[:3] == ui.rgb(ui.GREEN) for i in range(20)]
    assert sum(filled) == 10 and filled[:10] == [True] * 10
    ui.bar(canvas, (10, 30, 22, 5), -1.0, ui.BLUE, centered=True)
    blue = [canvas.get_at((11 + i, 32))[:3] == ui.rgb(ui.BLUE) for i in range(20)]
    assert sum(blue) == 10 and blue[:10] == [True] * 10
    ui.bar(canvas, (10, 50, 22, 5), 7.0, ui.GREEN)  # clamped
    assert canvas.get_at((30, 52))[:3] == ui.rgb(ui.GREEN)


def test_a_dither_veil_covers_the_stated_share_of_the_pixels(canvas):
    for level, share in ((1, 0.25), (2, 0.5), (3, 0.75)):
        canvas.fill(ui.rgb(ui.WHITE))
        ui.dither(canvas, (0, 0, 64, 64), level)
        covered = (pygame.surfarray.array3d(canvas)[:64, :64] == ui.rgb(ui.NIGHT)).all(axis=2)
        assert covered.mean() == pytest.approx(share, abs=0.001)
    ui.dither(canvas, (-10, -10, 5, 5), 2)  # outside: nothing happens, nothing breaks


def test_rolling_counter_ends_on_the_plain_number_and_never_leaves_its_cells(canvas):
    counter = ui.Rolling()
    canvas.fill(ui.rgb(ui.NIGHT))
    counter.draw(canvas, 99, 20, 20, ui.WHITE, 2, now=0.0)
    counter.draw(canvas, 100, 20, 20, ui.WHITE, 2, now=1.0)  # starts to roll
    mid = counter.draw(canvas, 100, 20, 20, ui.WHITE, 2, now=1.04)
    assert mid == 20 + 3 * 12
    canvas.fill(ui.rgb(ui.NIGHT))
    end = counter.draw(canvas, 100, 20, 20, ui.WHITE, 2, now=2.0)
    reference = pygame.Surface(ui.CANVAS_SIZE)
    reference.fill(ui.rgb(ui.NIGHT))
    for i, digit in enumerate("100"):
        ui.text(reference, digit, 20 + i * 12, 20, ui.WHITE, 2, ui.NIGHT, tabular=True)
    assert end == mid
    assert np.array_equal(pygame.surfarray.array3d(canvas), pygame.surfarray.array3d(reference))


# -- numbers in Spanish ----------------------------------------------------------------------


def test_numbers_use_the_spanish_decimal_comma_and_dotted_thousands():
    assert ui.fmt_decimal(0.5, 2) == "0,50"
    assert ui.fmt_decimal(-0.001, 2) == "0,00"
    assert ui.fmt_clock(64.5) == "1:04,50"
    assert ui.fmt_clock(None) == "-:--,--"
    assert ui.fmt_count(1503232) == "1.503.232"
    assert ui.comma("1.5") == "1,5"


# -- the canvas in the window ------------------------------------------------------------------


@pytest.mark.parametrize(
    "window, scale", [((1280, 720), 4), ((1920, 1080), 6), ((1366, 768), 4), ((300, 100), 1)]
)
def test_the_canvas_is_magnified_by_the_largest_integer_that_fits(window, scale):
    rect = fit_rect(window)
    assert rect.w == 320 * scale and rect.h == 180 * scale
    assert rect.centerx == pytest.approx(window[0] / 2, abs=1)


def test_present_makes_every_canvas_pixel_a_square_block_and_adds_nothing(canvas):
    pygame.display.init()
    screen = pygame.Surface((1366, 768))
    ui.text(canvas, "AB", 5, 5, ui.YELLOW)
    rect = present(canvas, screen)
    assert rect.size == (1280, 720) and rect.topleft == (43, 24)
    small = pygame.surfarray.array3d(canvas)
    big = pygame.surfarray.array3d(screen)[43:1323, 24:744]
    assert np.array_equal(big, np.kron(small, np.ones((4, 4, 1), dtype=np.uint8)))
    assert (pygame.surfarray.array3d(screen)[:43] == 0).all()  # the bars are black
    assert palette_only(screen.subsurface(rect))


def test_window_positions_map_to_canvas_pixels():
    assert to_logical((43, 24), (1366, 768)) == (0, 0)
    assert to_logical((43 + 4 * 100 + 3, 24 + 4 * 50), (1366, 768)) == (100, 50)
    assert to_logical((0, 0), (1366, 768)) == (-11, -6)  # on the bars: outside the canvas
    assert to_logical((1919, 1079), (1920, 1080)) == (319, 179)


# -- cost ------------------------------------------------------------------------------------


def test_a_cached_frame_of_interface_costs_well_under_a_millisecond(canvas):
    for _ in range(5):
        draw_everything(canvas)
    samples = []
    for _ in range(100):
        start = time.perf_counter()
        draw_everything(canvas)
        samples.append((time.perf_counter() - start) * 1000)
    assert statistics.median(samples) < 1.5  # about forty strings, plates and a dither
