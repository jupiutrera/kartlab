import pygame
import pytest

from kartlab.render.topdown import (
    ASPHALT,
    GRASS,
    GRASS_GRID,
    KART_COLOR,
    DebugHud,
    TextCache,
    TopDownView,
    TrackMap,
    format_time,
    load_fonts,
)
from kartlab.sim.race import Racer
from kartlab.sim.showcase import showcase_track


@pytest.fixture(scope="module")
def track():
    return showcase_track("harbor")


def _color(surface: pygame.Surface, px: tuple[float, float]) -> tuple[int, int, int]:
    return tuple(surface.get_at((round(px[0]), round(px[1]))))[:3]


def test_track_map_paints_asphalt_and_grass(track):
    tmap = TrackMap(track, ppm=3.0, margin=20.0)
    for s in (150.0, 420.0, 700.0):
        x, y, _ = track.pose_at(s, lateral=2.5)
        assert _color(tmap.surface, tmap.to_px(x, y)) == ASPHALT
        x, y, _ = track.pose_at(s, lateral=track.half_width + 6.0)
        assert _color(tmap.surface, tmap.to_px(x, y)) in (GRASS, GRASS_GRID)


def test_track_map_size_is_capped(track):
    tmap = TrackMap(track, ppm=100.0, margin=60.0)
    width, height = tmap.surface.get_size()
    assert width * height <= 12_100_000


@pytest.mark.parametrize("follow", [True, False])
def test_view_draws_the_kart_without_a_window(track, follow):
    size = (640, 360)
    view = TopDownView(track, size)
    racer = Racer(track)
    target = pygame.Surface(size)
    kart = racer.kart
    view.draw(target, [(kart, KART_COLOR)], (kart.x, kart.y), follow, trail=[(kart.x, kart.y)])
    tmap = view.follow_map if follow else view.fit_map
    if follow:
        centre = (size[0] / 2, size[1] / 2)
    else:
        mx, my = tmap.to_px(kart.x, kart.y)
        map_w, map_h = tmap.surface.get_size()
        centre = (mx + (size[0] - map_w) / 2, my + (size[1] - map_h) / 2)
    assert _color(target, centre) == KART_COLOR


def test_text_cache_renders_each_string_once():
    regular, _ = load_fonts(720)
    cache = TextCache(regular)
    first = cache.get("Vuelta 1", (255, 255, 255))
    assert cache.get("Vuelta 1", (255, 255, 255)) is first
    assert cache.get("Vuelta 2", (255, 255, 255)) is not first


def test_hud_draws_panels_in_every_corner():
    size = (640, 360)
    hud = DebugHud(size)
    target = pygame.Surface(size)
    target.fill((0, 0, 0))
    for anchor in ("top-left", "top-right", "bottom-left"):
        hud.draw_block(target, [("Velocidad 87 km/h", (255, 255, 255))], anchor, title="Puerto")
    assert target.get_at((hud.pad + 1, hud.pad + 1))[:3] != (0, 0, 0)
    assert target.get_at((size[0] - hud.pad - 2, hud.pad + 1))[:3] != (0, 0, 0)
    assert target.get_at((hud.pad + 1, size[1] - hud.pad - 2))[:3] != (0, 0, 0)


@pytest.mark.parametrize(
    "seconds, text", [(None, "-:--.--"), (0.0, "0:00.00"), (65.432, "1:05.43"), (600.0, "10:00.00")]
)
def test_format_time(seconds, text):
    assert format_time(seconds) == text
