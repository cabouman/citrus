"""Grids and cone-beam system geometry, following MBIRTORCH conventions (docs/source/conventions.rst).

All lengths in cm, angles in radians. Volumes are [rows, cols, slices] with col->x, row->y, slice->z.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass(frozen=True)
class Grid:
    """Uniform cuboid voxel grid centered on the rotation axis.

    shape: (num_rows, num_cols, num_slices) -> (y, x, z) extents in voxels.
    """

    shape: tuple[int, int, int]
    delta_voxel: float
    voxel_row_aspect: float = 1.0
    voxel_slice_aspect: float = 1.0
    recon_slice_offset: float = 0.0

    # ---- sizes ---------------------------------------------------------------------------
    @property
    def num_rows(self) -> int:
        return self.shape[0]

    @property
    def num_cols(self) -> int:
        return self.shape[1]

    @property
    def num_slices(self) -> int:
        return self.shape[2]

    @property
    def num_voxels(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def deltas(self) -> tuple[float, float, float]:
        """(dx, dy, dz) voxel pitches in cm."""
        d = self.delta_voxel
        return (d, self.voxel_row_aspect * d, self.voxel_slice_aspect * d)

    @property
    def voxel_volume(self) -> float:
        dx, dy, dz = self.deltas
        return dx * dy * dz

    @property
    def extent(self) -> tuple[float, float, float]:
        """Physical (x, y, z) side lengths in cm."""
        dx, dy, dz = self.deltas
        return (self.num_cols * dx, self.num_rows * dy, self.num_slices * dz)

    # ---- coordinates -----------------------------------------------------------------------
    def axis_coordinates(self, device=None, dtype=torch.float32) -> tuple[Tensor, Tensor, Tensor]:
        """1-D center coordinates (x[num_cols], y[num_rows], z[num_slices]) in the recon frame."""
        dx, dy, dz = self.deltas
        nr, nc, ns = self.shape
        x = dx * (torch.arange(nc, device=device, dtype=dtype) - (nc - 1) / 2)
        y = dy * (torch.arange(nr, device=device, dtype=dtype) - (nr - 1) / 2)
        z = dz * (torch.arange(ns, device=device, dtype=dtype) - (ns - 1) / 2) + self.recon_slice_offset
        return x, y, z

    def center_coordinates(self, device=None, dtype=torch.float32) -> Tensor:
        """Voxel-center positions as a [rows, cols, slices, 3] tensor of (x, y, z)."""
        x, y, z = self.axis_coordinates(device, dtype)
        X = x[None, :, None].expand(self.shape)
        Y = y[:, None, None].expand(self.shape)
        Z = z[None, None, :].expand(self.shape)
        return torch.stack([X, Y, Z], dim=-1)

    def lower_corner(self) -> tuple[float, float, float]:
        """Physical (x, y, z) of the minimum corner of the grid box."""
        dx, dy, dz = self.deltas
        nr, nc, ns = self.shape
        return (-nc * dx / 2, -nr * dy / 2, -ns * dz / 2 + self.recon_slice_offset)

    # ---- coarsening -------------------------------------------------------------------------
    def coarsen(self, factor: int) -> "Grid":
        """Supervoxel grid with `factor` fine voxels per coarse voxel along each axis.

        Requires every shape entry to be divisible by factor so that the grid boundaries align.
        """
        if factor == 1:
            return self
        nr, nc, ns = self.shape
        if nr % factor or nc % factor or ns % factor:
            raise ValueError(f"grid shape {self.shape} not divisible by supervoxel factor {factor}")
        return replace(self, shape=(nr // factor, nc // factor, ns // factor), delta_voxel=self.delta_voxel * factor)

    def local_coordinates_of_fine_in_coarse(self, factor: int, device=None, dtype=torch.float32) -> Tensor:
        """Local LD coordinates u in [-1, 1] of each fine sub-voxel center inside its coarse cell.

        Returns a [factor] tensor; identical along each axis for a uniform integer coarsening.
        """
        f = factor
        return (2 * torch.arange(f, device=device, dtype=dtype) + 1) / f - 1


@dataclass(frozen=True)
class ConeGeometry:
    """Flat-panel cone-beam geometry, MBIRTORCH conventions.

    In the view frame the source sits at (0, +source_iso_dist, 0) and the panel lies in the plane
    y_view = source_iso_dist - source_detector_dist, with u along +x_view (channels) and v along +z (rows).
    """

    source_iso_dist: float
    source_detector_dist: float
    num_det_rows: int
    num_det_channels: int
    delta_det_row: float
    delta_det_channel: float
    det_row_offset: float = 0.0
    det_channel_offset: float = 0.0

    @property
    def magnification(self) -> float:
        return self.source_detector_dist / self.source_iso_dist

    @property
    def panel_y_view(self) -> float:
        return self.source_iso_dist - self.source_detector_dist

    @property
    def image_shape(self) -> tuple[int, int]:
        return (self.num_det_rows, self.num_det_channels)

    @property
    def pixel_area(self) -> float:
        return self.delta_det_row * self.delta_det_channel

    # ---- panel coordinates --------------------------------------------------------------------
    def pixel_uv(self, device=None, dtype=torch.float32) -> tuple[Tensor, Tensor]:
        """Pixel-center panel coordinates (u[num_det_channels], v[num_det_rows]) in cm."""
        nc, nr = self.num_det_channels, self.num_det_rows
        u = (torch.arange(nc, device=device, dtype=dtype) - (nc - 1) / 2) * self.delta_det_channel - self.det_channel_offset
        v = (torch.arange(nr, device=device, dtype=dtype) - (nr - 1) / 2) * self.delta_det_row - self.det_row_offset
        return u, v

    def panel_half_extent(self) -> tuple[float, float]:
        """Half widths (u, v) of the active panel area in cm (edge to edge, including pixel size)."""
        return (self.num_det_channels * self.delta_det_channel / 2, self.num_det_rows * self.delta_det_row / 2)

    # ---- recon-frame placement for a view -----------------------------------------------------
    @staticmethod
    def _rot(angle: float) -> tuple[float, float]:
        return math.cos(angle), math.sin(angle)

    def view_to_recon(self, p_view: Tensor, angle: float) -> Tensor:
        """Rotate points [..., 3] from the view frame into the recon frame."""
        c, s = self._rot(angle)
        xv, yv, zv = p_view[..., 0], p_view[..., 1], p_view[..., 2]
        return torch.stack([c * xv + s * yv, -s * xv + c * yv, zv], dim=-1)

    def recon_to_view(self, p: Tensor, angle: float) -> Tensor:
        """Rotate points [..., 3] from the recon frame into the view frame."""
        c, s = self._rot(angle)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        return torch.stack([c * x - s * y, s * x + c * y, z], dim=-1)

    def source_position(self, angle: float) -> tuple[float, float, float]:
        """Focal spot in the recon frame."""
        c, s = self._rot(angle)
        return (s * self.source_iso_dist, c * self.source_iso_dist, 0.0)

    def panel_center(self, angle: float) -> tuple[float, float, float]:
        """Point where the central ray meets the panel (u = v = 0), recon frame."""
        c, s = self._rot(angle)
        yv = self.panel_y_view
        return (s * yv, c * yv, 0.0)

    def panel_axes(self, angle: float) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        """Unit vectors (u_hat, v_hat, n_hat) of the panel in the recon frame; n_hat points toward the source."""
        c, s = self._rot(angle)
        u_hat = (c, -s, 0.0)
        v_hat = (0.0, 0.0, 1.0)
        n_hat = (s, c, 0.0)
        return u_hat, v_hat, n_hat

    def pixel_centers(self, angle: float, device=None, dtype=torch.float32) -> Tensor:
        """Pixel-center positions in the recon frame as [num_det_rows, num_det_channels, 3]."""
        u, v = self.pixel_uv(device, dtype)
        u_hat, v_hat, _ = self.panel_axes(angle)
        pc = torch.tensor(self.panel_center(angle), device=device, dtype=dtype)
        u_hat = torch.tensor(u_hat, device=device, dtype=dtype)
        v_hat = torch.tensor(v_hat, device=device, dtype=dtype)
        return pc + u[None, :, None] * u_hat + v[:, None, None] * v_hat

    def primary_directions(self, angle: float, device=None, dtype=torch.float32) -> Tensor:
        """Unit vectors from the focal spot to each pixel center, recon frame, [rows, channels, 3]."""
        src = torch.tensor(self.source_position(angle), device=device, dtype=dtype)
        d = self.pixel_centers(angle, device, dtype) - src
        return d / torch.linalg.norm(d, dim=-1, keepdim=True)

    def panel_distance_along(self, points: Tensor, omega: Tensor, angle: float) -> Tensor:
        """Signed distance t such that points + t*omega lies on the panel plane.

        points: [..., 3] recon frame; omega: [3] or [..., 3] unit direction. Returns [...].
        Negative or infinite values mean the direction does not reach the panel.
        """
        _, _, n_hat = self.panel_axes(angle)
        n = torch.tensor(n_hat, device=points.device, dtype=points.dtype)
        pc = torch.tensor(self.panel_center(angle), device=points.device, dtype=points.dtype)
        denom = (omega * n).sum(-1)
        return ((pc - points) * n).sum(-1) / denom

    def panel_uv_of_points(self, points: Tensor, angle: float) -> tuple[Tensor, Tensor]:
        """Panel coordinates (u, v) of points [..., 3] assumed to lie on the panel plane."""
        u_hat, v_hat, _ = self.panel_axes(angle)
        pc = torch.tensor(self.panel_center(angle), device=points.device, dtype=points.dtype)
        uh = torch.tensor(u_hat, device=points.device, dtype=points.dtype)
        vh = torch.tensor(v_hat, device=points.device, dtype=points.dtype)
        d = points - pc
        return (d * uh).sum(-1), (d * vh).sum(-1)

