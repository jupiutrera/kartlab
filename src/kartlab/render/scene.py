"""The pseudo-3D world at 320x180: sky, floor, scenery, karts and particles.

``WorldRenderer.render`` draws one frame into ``self.world`` and returns how long
each part took (for the diagnostics overlay). Floor and sky go into an 8-bit index
buffer; sprites are then blitted far to near onto the colour surface.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pygame

from kartlab.common.paths import FONTS_DIR
from kartlab.render.camera import CameraPose, Lens
from kartlab.render.floor import FloorRenderer, Projection
from kartlab.render.palette import FOG_COLOR, KART_RAMPS, Ink, game_palette
from kartlab.render.sky import Sky
from kartlab.render.sprites import PROP_HEIGHTS, VOXEL, SpriteBank, to_surface
from kartlab.render.texture import build_floor_texture
from kartlab.sim.kart import KartState
from kartlab.sim.track import Track

WORLD_SIZE = (320, 180)
LABEL_FONT = FONTS_DIR / "silkscreen" / "Silkscreen-Regular.ttf"
NEAR = 0.8  # nothing is drawn closer than this to the camera (m)
SUN = (0.55, -0.35)  # direction shadows are cast on the floor (unit-ish, world x/y)
SKID_SPEED = 1.6  # lateral speed above which tyres leave marks (m/s)
MAX_PARTICLES = 220


@dataclass
class KartVisual:
    kart: KartState
    ramp: int = 0
    label: str | None = None
    ghost: bool = False
    on_grass: bool = False
    throttle: float = 0.0
    trail: list[tuple[float, float]] = field(default_factory=list)


@dataclass(frozen=True)
class Prop:
    kind: str
    x: float
    y: float
    seed: int


def place_props(track: Track, seed: int = 0) -> list[Prop]:
    """Scenery along the track, deterministic for a track and seed."""
    rng = np.random.default_rng(seed)
    hw = track.half_width
    props: list[Prop] = []

    def add(kind: str, s: float, lateral: float, clearance: float) -> None:
        x, y, _ = track.pose_at(s, lateral)
        # The unhinted locate is the nearest part of the track, so with a level crossing
        # this also keeps props off the second road.
        if abs(track.locate(x, y).lateral) >= clearance:  # not on another part of the track
            props.append(Prop(kind, x, y, int(rng.integers(4))))

    for s in np.arange(0.0, track.length, 13.0):
        for side in (-1.0, 1.0):
            if rng.random() < 0.6:
                kind = "pine" if rng.random() < 0.55 else "tree"
                add(kind, s + rng.uniform(-5, 5), side * (hw + rng.uniform(9, 38)), hw + 7)

    bend = np.abs(track.curvature)
    for i in np.flatnonzero(bend > 1 / 40):
        window = bend[np.arange(i - 15, i + 16) % len(track)]
        if bend[i] < window.max():
            continue  # not the apex
        inside = math.copysign(1.0, track.curvature[i])
        s = float(track.s[i])
        for k in (-4.0, 0.0, 4.0):
            add("cone", s + k, inside * (hw + 1.8), hw + 1.2)
            add("tyres", s + 1.3 * k, -inside * (hw + 3.2), hw + 2.5)

    room = {side: abs(track.locate(*track.pose_at(25.0, side * (hw + 14))[:2]).lateral)
            for side in (-1.0, 1.0)}  # fmt: skip
    stand_side = max(room, key=room.get)
    add("grandstand", 25.0, stand_side * (hw + 14), hw + 9)
    for s in np.arange(-60.0, 61.0, 15.0):
        add("banner", s % track.length, -stand_side * (hw + 3.0), hw + 2)
    return props


def bake_shadows(texture, props: list[Prop]) -> None:
    """Paint the scenery's shadows on the floor texture (lighting is part of the art)."""
    for prop in props:
        height = PROP_HEIGHTS[prop.kind]
        reach = 0.35 * height
        texture.stamp(
            prop.x + SUN[0] * reach,
            prop.y + SUN[1] * reach,
            0.3 * height if prop.kind != "grandstand" else 6.0,
            Ink.GRASS_DARK,
            density=0.6,
        )


class Particles:
    """Dust and smoke puffs in world space (simple arrays, no per-pixel work)."""

    def __init__(self) -> None:
        self.items: list[list[float]] = []  # x, y, z, vx, vy, vz, life, colour

    def emit(self, x, y, z, vx, vy, vz, life, colour) -> None:
        if len(self.items) < MAX_PARTICLES:
            self.items.append([x, y, z, vx, vy, vz, life, colour])

    def update(self, dt: float) -> None:
        alive = []
        for p in self.items:
            p[0] += p[3] * dt
            p[1] += p[4] * dt
            p[2] = max(p[2] + p[5] * dt, 0.0)
            p[3] *= 0.92
            p[4] *= 0.92
            p[6] -= dt
            if p[6] > 0:
                alive.append(p)
        self.items = alive


class WorldRenderer:
    def __init__(
        self,
        track: Track,
        lens: Lens,
        variant: str = "day",
        light_mode: bool = False,
        seed: int = 0,
    ) -> None:
        pygame.font.init()
        self.track = track
        self.palette = game_palette()
        self.size = WORLD_SIZE
        self.seed = seed
        self.texture = build_floor_texture(track, seed)
        self.props = place_props(track, seed)
        bake_shadows(self.texture, self.props)
        self.texture.build_mips()
        self._prop_xy = np.array([(p.x, p.y) for p in self.props]).reshape(-1, 2)
        self._prop_h = np.array([PROP_HEIGHTS[p.kind] for p in self.props])
        self.sprites = SpriteBank(self.palette, variant)
        self.particles = Particles()
        self.buffer = np.empty((WORLD_SIZE[1], WORLD_SIZE[0]), dtype=np.uint8)
        self.indexed = pygame.Surface(WORLD_SIZE, depth=8)
        self.indexed.set_palette([tuple(int(c) for c in rgb) for rgb in self.palette.rgb])
        self.world = pygame.Surface(WORLD_SIZE)
        self.label_font = pygame.font.Font(LABEL_FONT, 8)
        self._labels: dict[tuple[str, int], pygame.Surface] = {}
        self._halos: dict[tuple[int, int], pygame.Surface] = {}
        self.show_rays: list[tuple[float, float, float]] = []  # (x, y, heading) for overlay
        self.ray_lengths: np.ndarray | None = None
        self.ray_angles: np.ndarray | None = None
        self.light_mode = light_mode
        self.variant = variant
        self.lens = lens
        self._configure()

    # -- configuration ------------------------------------------------------------------

    def _configure(self) -> None:
        width, height = self.size
        self.projection = Projection(width, height, self.lens)
        self.floor = FloorRenderer(self.projection, self.texture, 2 if self.light_mode else 1)
        self.sky = Sky(self.projection, self.palette, self.variant, self.floor.first_row)
        self.fog_table = self.palette.fog_table(self.variant)

    def set_lens(self, lens: Lens) -> None:
        if lens != self.lens:
            self.lens = lens
            self._configure()

    def set_variant(self, variant: str) -> None:
        if variant != self.variant:
            self.variant = variant
            self.sprites.set_variant(variant)
            self._halos.clear()
            self._configure()

    def set_light_mode(self, on: bool) -> None:
        if on != self.light_mode:
            self.light_mode = on
            self._configure()

    # -- simulation-side effects -------------------------------------------------------

    def update_effects(self, visuals: list[KartVisual], dt: float) -> None:
        """Skid marks and particles from what each kart is doing this frame."""
        rng = np.random.default_rng()
        rate = 0.5 if self.light_mode else 1.0
        for v in visuals:
            kart = v.kart
            speed = kart.forward_speed
            c, s = math.cos(kart.heading), math.sin(kart.heading)
            rear = [(kart.x - c * 0.7 + side * -s * 0.5, kart.y - s * 0.7 + side * c * 0.5)
                    for side in (-1.0, 1.0)]  # fmt: skip
            if abs(kart.lateral_speed) > SKID_SPEED and not v.on_grass and not v.ghost:
                for x, y in rear:
                    self.texture.stamp(x, y, 0.14, Ink.ASPHALT_DARK)
            if v.on_grass and abs(speed) > 3.0 and rng.random() < 0.6 * rate:
                x, y = rear[int(rng.integers(2))]
                self.particles.emit(x, y, 0.1, -c * speed * 0.2, -s * speed * 0.2, 1.2,
                                    0.6, Ink.BROWN)  # fmt: skip
            if v.throttle > 0.8 and 0.5 < speed < 9.0 and rng.random() < 0.35 * rate:
                x, y = rear[int(rng.integers(2))]
                self.particles.emit(x, y, 0.3, rng.normal(0, 0.4), rng.normal(0, 0.4), 0.8,
                                    0.7, Ink.MIST)  # fmt: skip
        self.particles.update(dt)

    # -- drawing -------------------------------------------------------------------------

    def render(self, pose: CameraPose, visuals: list[KartVisual], time_s: float = 0.0) -> dict:
        timings = {}
        t0 = time.perf_counter()
        self.buffer.fill(FOG_COLOR[self.variant])
        self.sky.render(self.buffer, pose.heading, time_s)
        t1 = time.perf_counter()
        self.floor.render(self.buffer, pose, self.fog_table)
        t2 = time.perf_counter()
        pygame.surfarray.blit_array(self.indexed, self.buffer.T)
        self.world.blit(self.indexed, (0, 0))
        t3 = time.perf_counter()
        self._draw_sprites(pose, visuals)
        t4 = time.perf_counter()
        timings["cielo"] = (t1 - t0) * 1000
        timings["suelo"] = (t2 - t1) * 1000
        timings["copia"] = (t3 - t2) * 1000
        timings["sprites"] = (t4 - t3) * 1000
        return timings

    def _draw_sprites(self, pose: CameraPose, visuals: list[KartVisual]) -> None:
        self.sprites.begin_frame()
        proj, lens = self.projection, self.lens
        focal = proj.focal
        width, height = self.size
        drawables: list[tuple[float, int, object]] = []  # (depth, order, callable)

        if len(self.props):
            sx, sy, depth = proj.project(pose, self._prop_xy[:, 0], self._prop_xy[:, 1])
            visible = (depth > NEAR) & (depth < lens.far * (0.6 if self.light_mode else 1.0))
            heights = self._prop_h * focal / np.maximum(depth, 1e-6)
            visible &= (heights >= 2) & (sx > -heights * 2) & (sx < width + heights * 2)
            visible &= sy > -heights
            for i in np.flatnonzero(visible):
                prop = self.props[i]
                drawables.append(
                    (float(depth[i]), 0, (self._draw_prop, prop, sx[i], sy[i], heights[i]))
                )

        for v in visuals:
            kart = v.kart
            sx, sy, depth = proj.project(pose, kart.x, kart.y)
            if depth <= NEAR:
                continue
            drawables.append(
                (float(depth), 2, (self._draw_kart, v, float(sx), float(sy), float(depth), pose))
            )
            for j, (tx, ty) in enumerate(v.trail if v.ghost else ()):
                tsx, tsy, tdepth = proj.project(pose, tx, ty, 0.25)
                if tdepth > NEAR:
                    drawables.append((float(tdepth), 1, (self._draw_dot, v.ramp, tsx, tsy, j)))

        for p in self.particles.items:
            psx, psy, pdepth = proj.project(pose, p[0], p[1], p[2])
            if pdepth > NEAR:
                size = max(1, int(0.35 * focal / pdepth))
                drawables.append(
                    (float(pdepth), 1, (self._draw_particle, psx, psy, size, int(p[7])))
                )

        drawables.sort(key=lambda d: (-d[0], d[1]))
        self._placed_labels: list[pygame.Rect] = []
        for _, _, (fn, *args) in drawables:
            fn(*args)
        if self.show_rays and self.ray_lengths is not None:
            self._draw_rays(pose)

    def _draw_prop(self, prop: Prop, sx: float, sy: float, height_px: float) -> None:
        surface, anchor = self.sprites.prop(prop.kind, height_px, prop.seed)
        self.world.blit(surface, (round(sx - anchor[0]), round(sy - anchor[1])))

    def _draw_kart(self, v: KartVisual, sx: float, sy: float, depth: float, pose: CameraPose):
        kart = v.kart
        focal = self.projection.focal
        bearing = math.atan2(kart.y - pose.y, kart.x - pose.x)
        horizontal = math.hypot(kart.x - pose.x, kart.y - pose.y)
        elevation = math.atan2(self.lens.height, max(horizontal, 1e-6))
        scale = focal * VOXEL / depth
        if v.ghost:
            halo = self._halo(v.ramp, max(int(2.6 * focal / depth), 4))
            where = (round(sx - halo.get_width() / 2), round(sy - halo.get_height() * 0.6))
            self.world.blit(halo, where, special_flags=pygame.BLEND_ADD)
        shadow = self.sprites.shadow(max(int(1.4 * focal / depth), 2))
        self.world.blit(
            shadow, (round(sx - shadow.get_width() / 2), round(sy - shadow.get_height() / 2))
        )
        surface, anchor = self.sprites.kart(v.ramp, kart.heading - bearing, elevation, scale)
        top = round(sy - anchor[1])
        self.world.blit(surface, (round(sx - anchor[0]), top))
        if v.label:
            label = self._label(v.label, v.ramp)
            rect = label.get_rect(midbottom=(round(sx), top - 1))
            while rect.collidelist(self._placed_labels) != -1:  # karts side by side: stack
                rect.move_ip(0, -rect.height)
            self._placed_labels.append(rect)
            self.world.blit(label, rect)

    def _draw_dot(self, ramp: int, sx: float, sy: float, age: int) -> None:
        if age % 2:
            return
        colour = self.palette.rgb[self.sprites.remap[KART_RAMPS[ramp % len(KART_RAMPS)][2]]]
        self.world.set_at((round(sx), round(sy)), tuple(int(c) for c in colour))

    def _draw_particle(self, sx: float, sy: float, size: int, colour: int) -> None:
        rgb = self.palette.rgb[self.sprites.remap[colour]]
        self.world.fill(
            tuple(int(c) for c in rgb), (round(sx - size / 2), round(sy - size), size, size)
        )

    def _draw_rays(self, pose: CameraPose) -> None:
        colour = tuple(int(c) for c in self.palette.rgb[Ink.YELLOW])
        for x, y, heading in self.show_rays:
            for angle, length in zip(self.ray_angles, self.ray_lengths, strict=True):
                ex = x + math.cos(heading + angle) * length
                ey = y + math.sin(heading + angle) * length
                points = []
                for t in np.linspace(0.0, 1.0, 12):
                    px, py, d = self.projection.project(
                        pose, x + (ex - x) * t, y + (ey - y) * t, 0.05
                    )
                    if d > NEAR:
                        points.append((float(px), float(py)))
                if len(points) > 1:
                    pygame.draw.lines(self.world, colour, False, points)

    def _label(self, text: str, ramp: int) -> pygame.Surface:
        key = (text, ramp)
        surface = self._labels.get(key)
        if surface is None:
            colour = self.palette.rgb[KART_RAMPS[ramp % len(KART_RAMPS)][2]]
            fg = self.label_font.render(text, False, tuple(int(c) for c in colour))
            surface = pygame.Surface((fg.get_width() + 2, fg.get_height() + 2))
            surface.fill((255, 0, 255))
            outline = self.label_font.render(
                text, False, tuple(int(c) for c in self.palette.rgb[Ink.BLACK])
            )
            for dx, dy in ((0, 1), (2, 1), (1, 0), (1, 2)):
                surface.blit(outline, (dx, dy))
            surface.blit(fg, (1, 1))
            surface.set_colorkey((255, 0, 255))
            self._labels[key] = surface
        return surface

    def _halo(self, ramp: int, size: int) -> pygame.Surface:
        """Soft glow, blurred once per size and colour, blended additively."""
        size = min(max(size, 4), 96)
        key = (ramp, size)
        halo = self._halos.get(key)
        if halo is None:
            yy, xx = np.mgrid[0:size, 0:size]
            r = np.hypot(xx + 0.5 - size / 2, yy + 0.5 - size / 2) / (size / 2)
            glow = np.exp(-3.0 * r * r) * (r < 1)
            colour = self.palette.rgb[self.sprites.remap[KART_RAMPS[ramp % len(KART_RAMPS)][1]]]
            rgb = (glow[..., None] * colour[None, None, :] * 0.55).astype(np.uint8)
            halo = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
            self._halos[key] = halo
        return halo

    def world_array(self) -> np.ndarray:
        """The last frame as an (height, width, 3) array (for videos and tests)."""
        return pygame.surfarray.array3d(self.world).transpose(1, 0, 2)


__all__ = ["KartVisual", "Prop", "WorldRenderer", "place_props", "to_surface"]
