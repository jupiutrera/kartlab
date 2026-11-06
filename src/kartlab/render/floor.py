"""Perspective floor ("Mode 7" style), one table lookup per screen pixel.

For a fixed lens everything that does not depend on the camera position is computed
once: the distance of each screen row, the lateral offset of each pixel, the mipmap
level of each row (texel footprint) and a dithered fog level per pixel. A frame then
does a few in-place vector operations, one ``take`` per mipmap row group and one last
``take`` that applies fog and the palette variant together. No Python loop per pixel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kartlab.render.camera import CameraPose, Lens
from kartlab.render.palette import FOG_LEVELS
from kartlab.render.texture import BAYER4, FloorTexture

MIP_BIAS = 0.35  # >0 picks coarser levels sooner: less shimmer far away


@dataclass(frozen=True)
class Projection:
    """Screen geometry of a lens: focal length (px) and centre."""

    width: int
    height: int
    lens: Lens

    @property
    def focal(self) -> float:
        return (self.width / 2) / math.tan(self.lens.fov / 2)

    @property
    def centre(self) -> tuple[float, float]:
        return self.width / 2, self.height / 2

    @property
    def horizon(self) -> float:
        """Screen row of the horizon (may be off screen when looking down)."""
        if self.lens.pitch >= math.pi / 2 - 1e-6:
            return -math.inf
        return self.height / 2 - self.focal * math.tan(self.lens.pitch)

    def project(self, pose: CameraPose, x, y, z=0.0):
        """Screen (sx, sy) and depth (m) of world points; depth <= 0 is behind the camera."""
        lens = self.lens
        dx, dy = np.asarray(x) - pose.x, np.asarray(y) - pose.y
        cos_h, sin_h = math.cos(pose.heading), math.sin(pose.heading)
        ahead = dx * cos_h + dy * sin_h
        right = dx * sin_h - dy * cos_h
        up = np.asarray(z) - (lens.height + pose.z)
        cos_p, sin_p = math.cos(lens.pitch), math.sin(lens.pitch)
        depth = ahead * cos_p - up * sin_p
        vertical = ahead * sin_p + up * cos_p
        safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
        cx, cy = self.centre
        return cx + self.focal * right / safe, cy - self.focal * vertical / safe, depth


class FloorRenderer:
    def __init__(self, projection: Projection, texture: FloorTexture, row_step: int = 1):
        self.projection = projection
        self.texture = texture
        self.row_step = row_step
        lens, width = projection.lens, projection.width
        focal = projection.focal
        cx, cy = projection.centre
        sin_p, cos_p = math.sin(lens.pitch), math.cos(lens.pitch)

        rows = np.arange(projection.height) + 0.5
        b = (rows - cy) / focal
        denom = sin_p + b * cos_p
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(denom > 1e-6, lens.height / denom, np.inf)
            ahead = t * (cos_p - b * sin_p)
        visible = np.isfinite(t) & (ahead > 0) & (ahead <= lens.far)
        first = int(np.argmax(visible)) if visible.any() else projection.height
        self.first_row = first
        rows_idx = np.arange(first, projection.height, row_step)
        self.rows = rows_idx
        a = (np.arange(width) + 0.5 - cx) / focal
        tpm = texture.tpm
        self.ahead_t = (ahead[rows_idx] * tpm).astype(np.float32)
        self.lat_t = (t[rows_idx, None] * a[None, :] * tpm).astype(np.float32)

        # Texel footprint per row: the larger of the lateral and the along-row spacing.
        spacing_along = np.abs(np.gradient(ahead[first:])) if projection.height - first > 1 else 1
        spacing_along = np.asarray(spacing_along)[rows_idx - first] * row_step
        footprint = np.maximum(t[rows_idx] / focal, spacing_along) * tpm
        levels = np.clip(
            np.floor(np.log2(np.maximum(footprint, 1e-9)) + MIP_BIAS), 0, len(texture.levels) - 1
        ).astype(int)
        self.groups: list[tuple[int, int, int]] = []
        start = 0
        for i in range(1, len(levels) + 1):
            if i == len(levels) or levels[i] != levels[start]:
                self.groups.append((start, i, int(levels[start])))
                start = i

        fog = np.clip((ahead[rows_idx] - lens.fog_start) / (lens.fog_end - lens.fog_start), 0, 1)
        dither = BAYER4[rows_idx[:, None] % 4, np.arange(width)[None, :] % 4]
        fog_level = np.clip(np.floor(fog[:, None] * (FOG_LEVELS - 1) + dither), 0, FOG_LEVELS - 1)
        self.fog_offset = (fog_level * 64).astype(np.int32)

        shape = self.lat_t.shape
        self._u = np.empty(shape, np.float32)
        self._v = np.empty(shape, np.float32)
        self._ui = np.empty(shape, np.int32)
        self._vi = np.empty(shape, np.int32)
        self._index = np.empty(shape, np.uint8)
        self._flat = np.empty(shape, np.int32)
        self._flat_levels = [lvl.ravel() for lvl in texture.levels]

    def render(self, out: np.ndarray, pose: CameraPose, fog_table: np.ndarray) -> None:
        """Write palette indices of the floor into ``out`` (screen rows x columns)."""
        if len(self.rows) == 0:
            return
        tex = self.texture
        cos_h, sin_h = math.cos(pose.heading), math.sin(pose.heading)
        u0 = (pose.x - tex.origin[0]) * tex.tpm
        v0 = (tex.origin[1] - pose.y) * tex.tpm
        # A camera higher (lower) than the lens by ``pose.z`` sees every floor point k times
        # farther (nearer) along its ray: k scales the tables, and the texels grow with it, so
        # the mipmap level moves by log2(k). With z = 0 this is exactly the plain frame.
        k = (self.projection.lens.height + pose.z) / self.projection.lens.height
        shift = round(math.log2(k)) if k != 1.0 else 0
        last = len(tex.levels) - 1
        # world = camera + forward * ahead + right * lateral, right = (sin h, -cos h)
        np.multiply(self.lat_t, sin_h * k, out=self._u)
        self._u += (u0 + cos_h * k * self.ahead_t)[:, None]
        np.multiply(self.lat_t, cos_h * k, out=self._v)
        self._v += (v0 - sin_h * k * self.ahead_t)[:, None]

        for r0, r1, level in self.groups:
            level = min(max(level + shift, 0), last)
            rows, cols = tex.levels[level].shape
            scale = 0.5**level
            u = self._ui[r0:r1]
            v = self._vi[r0:r1]
            np.copyto(u, self._u[r0:r1] * scale, casting="unsafe")
            np.copyto(v, self._v[r0:r1] * scale, casting="unsafe")
            np.clip(u, 0, cols - 1, out=u)
            np.clip(v, 0, rows - 1, out=v)
            flat = self._flat[r0:r1]
            np.multiply(v, cols, out=flat)
            flat += u
            np.take(self._flat_levels[level], flat, out=self._index[r0:r1], mode="clip")

        np.add(self.fog_offset, self._index, out=self._flat)
        target = out[self.first_row :: self.row_step][: len(self.rows)]
        np.take(fog_table, self._flat, out=target, mode="clip")
        if self.row_step > 1:  # light mode: repeat each computed row below itself
            for k in range(1, self.row_step):
                below = out[self.first_row + k :: self.row_step]
                below[:] = target[: len(below)]
