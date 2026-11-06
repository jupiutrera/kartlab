import pygame
import pytest

from kartlab.render.topdown import (
    ASPHALT,
    GRASS,
    GRASS_GRID,
    KART_COLOR,
    TopDownView,
    TrackMap,
    format_time,
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


@pytest.mark.parametrize(
    "seconds, text", [(None, "-:--,--"), (0.0, "0:00,00"), (65.432, "1:05,43"), (600.0, "10:00,00")]
)
def test_format_time(seconds, text):
    assert format_time(seconds) == text


class _Recorder:
    """A drawing target that only notes the size of every rectangle filled on it."""

    def __init__(self):
        self.sizes = []

    def fill(self, color, rect):
        self.sizes.append((rect[2], rect[3]))


def test_the_sweep_sparks_are_one_or_two_pixels_on_the_320_by_180_grid(track, monkeypatch):
    from kartlab.sim.drift import DriftState
    from kartlab.sim.kart import KartState

    def no_circles(*args, **kwargs):
        raise AssertionError("the pixel view draws its sparks as squares, not circles")

    view = TopDownView(track, (224, 180), pixel=True)
    monkeypatch.setattr(pygame.draw, "circle", no_circles)  # the track maps are already drawn
    kart = KartState(*track.pose_at(200.0))
    for time_swept, biggest in ((0.1, 1), (2.0, 2)):
        drift = DriftState(active=True, direction=1.0, time=time_swept)
        recorder = _Recorder()
        view._draw_sparks(recorder, view.follow_map, kart, drift, (0.0, 0.0))
        assert recorder.sizes
        assert {w for w, _ in recorder.sizes} <= {1, 2} and {h for _, h in recorder.sizes} <= {1, 2}
        assert max(max(size) for size in recorder.sizes) == biggest
