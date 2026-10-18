"""Voxel grids, conservative resampling between them, and the flat-panel cone-beam geometry.

The conventions are MBIRTORCH's (docs/source/conventions.rst): lengths in cm, angles in radians, volumes
[rows, cols, slices] with columns along x, rows along y and slices along z. :class:`Grid` builds the solver's fine,
transport and quadrature grids, :func:`resample_conservative` carries densities between them by exact cell averages,
and :class:`ConeGeometry` places the source and the panel of each view.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass(frozen=True)
class Grid:
    """Uniform cuboid voxel grid centered on the rotation axis, in MBIRTORCH conventions.

    Attributes:
        shape (tuple[int, int, int]): (num_rows, num_cols, num_slices), the extents along (y, x, z) in voxels.
        delta_voxel (float): voxel pitch along x in cm.
        voxel_row_aspect (float): ratio of the pitch along y to ``delta_voxel``. Defaults to 1.0.
        voxel_slice_aspect (float): ratio of the pitch along z to ``delta_voxel``. Defaults to 1.0.
        recon_slice_offset (float): z of the grid center in cm. Defaults to 0.0.
    """

    shape: tuple[int, int, int]
    delta_voxel: float
    voxel_row_aspect: float = 1.0
    voxel_slice_aspect: float = 1.0
    recon_slice_offset: float = 0.0

    # ---- sizes ---------------------------------------------------------------------------
    @property
    def num_rows(self) -> int:
        """Number of rows (along y) of the grid."""
        return self.shape[0]

    @property
    def num_cols(self) -> int:
        """Number of columns (along x) of the grid."""
        return self.shape[1]

    @property
    def num_slices(self) -> int:
        """Number of slices (along z) of the grid."""
        return self.shape[2]

    @property
    def num_voxels(self) -> int:
        """Total number of voxels of the grid."""
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def deltas(self) -> tuple[float, float, float]:
        """Voxel pitches (dx, dy, dz) in cm."""
        d = self.delta_voxel
        return (d, self.voxel_row_aspect * d, self.voxel_slice_aspect * d)

    @property
    def voxel_volume(self) -> float:
        """Volume of one voxel in cm^3."""
        dx, dy, dz = self.deltas
        return dx * dy * dz

    @property
    def extent(self) -> tuple[float, float, float]:
        """Side lengths (x, y, z) of the grid box in cm."""
        dx, dy, dz = self.deltas
        return (self.num_cols * dx, self.num_rows * dy, self.num_slices * dz)

    # ---- coordinates -----------------------------------------------------------------------
    def axis_coordinates(self, device=None, dtype=torch.float32) -> tuple[Tensor, Tensor, Tensor]:
        """Return the voxel-center coordinates along each axis in the recon frame.

        Args:
            device (torch.device, optional): device of the tensors. Defaults to None.
            dtype (torch.dtype, optional): dtype of the tensors. Defaults to torch.float32.

        Returns:
            (Tensor, Tensor, Tensor): x [num_cols], y [num_rows] and z [num_slices] in cm.
        """
        dx, dy, dz = self.deltas
        nr, nc, ns = self.shape
        x = dx * (torch.arange(nc, device=device, dtype=dtype) - (nc - 1) / 2)
        y = dy * (torch.arange(nr, device=device, dtype=dtype) - (nr - 1) / 2)
        z = dz * (torch.arange(ns, device=device, dtype=dtype) - (ns - 1) / 2) + self.recon_slice_offset
        return x, y, z

    def center_coordinates(self, device=None, dtype=torch.float32) -> Tensor:
        """Return the voxel-center positions in the recon frame.

        Args:
            device (torch.device, optional): device of the tensors. Defaults to None.
            dtype (torch.dtype, optional): dtype of the tensor. Defaults to torch.float32.

        Returns:
            Tensor: [rows, cols, slices, 3] positions (x, y, z) in cm.
        """
        x, y, z = self.axis_coordinates(device, dtype)
        X = x[None, :, None].expand(self.shape)
        Y = y[:, None, None].expand(self.shape)
        Z = z[None, None, :].expand(self.shape)
        return torch.stack([X, Y, Z], dim=-1)

    def lower_corner(self) -> tuple[float, float, float]:
        """Return the minimum corner of the grid box in the recon frame.

        Returns:
            (float, float, float): (x, y, z) of the corner in cm.
        """
        dx, dy, dz = self.deltas
        nr, nc, ns = self.shape
        return (-nc * dx / 2, -nr * dy / 2, -ns * dz / 2 + self.recon_slice_offset)

    # ---- derived grids -----------------------------------------------------------------------
    @staticmethod
    def covering(grid: "Grid", cell_cm: float) -> "Grid":
        """Build the grid of cubic cells that covers the box of ``grid`` and is centred on it, the transport grid C.

        Each axis gets ``ceil(extent / cell_cm)`` cells (an extent within 1e-9 relative of a multiple counts as that
        multiple), which pad the box symmetrically with empty space (docs/source/conventions.rst, "Grids").

        Args:
            grid (Grid): the grid to cover, the fine grid F.
            cell_cm (float): cell edge in cm, > 0.

        Returns:
            Grid: the covering grid, with unit aspect ratios and the ``recon_slice_offset`` of ``grid``.
        """
        c = float(cell_cm)
        if not c > 0:
            raise ValueError(f"cell size must be > 0, got {cell_cm}")
        ex, ey, ez = grid.extent

        def count(e: float) -> int:
            r = e / c
            k = round(r)
            return max(1, int(k) if abs(r - k) <= 1e-9 * max(1.0, r) else int(math.ceil(r)))

        return Grid(shape=(count(ey), count(ex), count(ez)), delta_voxel=c,
                    recon_slice_offset=grid.recon_slice_offset)

    def refine(self, factor: int) -> "Grid":
        """Build the grid of the same box with ``factor`` voxels per voxel of this grid along each axis.

        This is the inverse of ``coarsen``; the fine voxel centers inside a voxel of this grid form the midpoint
        lattice at local coordinates ``(2j + 1)/factor - 1``, and the quadrature grid is ``Q = C.refine(q)``.

        Args:
            factor (int): refinement factor, >= 1.

        Returns:
            Grid: the refined grid (this grid itself for ``factor`` 1).
        """
        f = int(factor)
        if f < 1:
            raise ValueError(f"refinement factor must be >= 1, got {factor}")
        if f == 1:
            return self
        nr, nc, ns = self.shape
        return replace(self, shape=(nr * f, nc * f, ns * f), delta_voxel=self.delta_voxel / f)

    def axis_edges(self, device=None, dtype=torch.float64) -> tuple[Tensor, Tensor, Tensor]:
        """Return the voxel boundary coordinates along each axis in the recon frame.

        Args:
            device (torch.device, optional): device of the tensors. Defaults to None.
            dtype (torch.dtype, optional): dtype of the tensors. Defaults to torch.float64.

        Returns:
            (Tensor, Tensor, Tensor): x [num_cols + 1], y [num_rows + 1] and z [num_slices + 1] in cm.
        """
        dx, dy, dz = self.deltas
        nr, nc, ns = self.shape
        lx, ly, lz = self.lower_corner()
        return (lx + dx * torch.arange(nc + 1, device=device, dtype=dtype),
                ly + dy * torch.arange(nr + 1, device=device, dtype=dtype),
                lz + dz * torch.arange(ns + 1, device=device, dtype=dtype))

    def coarsen(self, factor: int) -> "Grid":
        """Build the supervoxel grid with ``factor`` voxels of this grid per coarse voxel along each axis.

        Args:
            factor (int): coarsening factor; every entry of ``shape`` must be divisible by it.

        Returns:
            Grid: the coarse grid (this grid itself for ``factor`` 1).
        """
        if factor == 1:
            return self
        nr, nc, ns = self.shape
        if nr % factor or nc % factor or ns % factor:
            raise ValueError(f"grid shape {self.shape} not divisible by the coarsening factor {factor}")
        return replace(self, shape=(nr // factor, nc // factor, ns // factor), delta_voxel=self.delta_voxel * factor)

    def local_coordinates_of_fine_in_coarse(self, factor: int, device=None, dtype=torch.float32) -> Tensor:
        """Return the local linear-discontinuous coordinates in [-1, 1] of the sub-voxel centers inside a coarse cell.

        The coordinates are the same along each axis for a uniform integer coarsening.

        Args:
            factor (int): number of sub-voxels per coarse cell along an axis.
            device (torch.device, optional): device of the tensors. Defaults to None.
            dtype (torch.dtype, optional): dtype of the tensor. Defaults to torch.float32.

        Returns:
            Tensor: [factor] local coordinates ``(2j + 1)/factor - 1``.
        """
        f = factor
        return (2 * torch.arange(f, device=device, dtype=dtype) + 1) / f - 1


# =====================================================================================================
# conservative resampling between grids (fine grid F -> transport grid C / quadrature grid Q)
# =====================================================================================================

_OVERLAP_CACHE: "OrderedDict[tuple, tuple[Tensor, Tensor, Tensor]]" = OrderedDict()
_OVERLAP_CACHE_SIZE = 32
_OVERLAP_SNAP = 1e-12  # overlap fractions within this of 0 or 1 are set to 0 or 1 (rounding of the edge coordinates)


def overlap_matrix(edges_from: Tensor, edges_to: Tensor) -> Tensor:
    """Compute the 1-D overlap fractions between two partitions of an axis.

    ``W[j, i]`` is the length of the overlap of source cell i and target cell j over the length of target cell j, so
    ``sum_i W[j, i] f_i`` is the exact average over target cell j of a piecewise-constant field (zero outside the source
    partition). Fractions within 1e-12 of 0 or 1 are snapped to 0 or 1.

    Args:
        edges_from (Tensor): [n_from + 1] ascending source cell edges.
        edges_to (Tensor): [n_to + 1] ascending target cell edges.

    Returns:
        Tensor: [n_to, n_from] float64 overlap fractions W.
    """
    ef = edges_from.to(torch.float64)
    et = edges_to.to(torch.float64)
    lo = torch.maximum(et[:-1, None], ef[None, :-1])
    hi = torch.minimum(et[1:, None], ef[None, 1:])
    w = (hi - lo).clamp_min(0.0) / (et[1:] - et[:-1])[:, None]
    w = torch.where(w < _OVERLAP_SNAP, torch.zeros_like(w), w)
    return torch.where((w - 1.0).abs() < _OVERLAP_SNAP, torch.ones_like(w), w)


def resampling_matrices(grid_from: Grid, grid_to: Grid, device=None,
                        dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor, Tensor]:
    """Return the per-axis overlap matrices from ``grid_from`` to ``grid_to``, computed in float64 and cached.

    The cache keeps the last 32 sets, keyed by the two grids, the device and the dtype.

    Args:
        grid_from (Grid): the source grid.
        grid_to (Grid): the target grid.
        device (torch.device, optional): device of the matrices; None is the CPU. Defaults to None.
        dtype (torch.dtype, optional): dtype of the matrices. Defaults to torch.float32.

    Returns:
        (Tensor, Tensor, Tensor): rows (y) [R_to, R_from], cols (x) [C_to, C_from] and slices (z) [S_to, S_from].
    """
    key = (grid_from, grid_to, str(torch.device(device) if device is not None else torch.device("cpu")), dtype)
    hit = _OVERLAP_CACHE.get(key)
    if hit is not None:
        _OVERLAP_CACHE.move_to_end(key)
        return hit
    xf, yf, zf = grid_from.axis_edges()
    xt, yt, zt = grid_to.axis_edges()
    mats = tuple(overlap_matrix(a, b).to(device=device, dtype=dtype).contiguous()
                 for a, b in ((yf, yt), (xf, xt), (zf, zt)))
    _OVERLAP_CACHE[key] = mats
    while len(_OVERLAP_CACHE) > _OVERLAP_CACHE_SIZE:
        _OVERLAP_CACHE.popitem(last=False)
    return mats


def resample_conservative(field: Tensor, grid_from: Grid, grid_to: Grid) -> Tensor:
    """Resample a voxel-wise constant field from one grid to another by exact cell averages.

    ``out[j] = sum_i field[i] |V_i n V_j| / |V_j|`` over the voxels V_i of ``grid_from`` and the cells V_j of
    ``grid_to``, applied as three per-axis matrix products; parts of a target cell outside the box of ``grid_from``
    count as zero, so the integral is conserved when ``grid_to`` covers ``grid_from``.

    Args:
        field (Tensor): [..., R, C, S] values on ``grid_from``.
        grid_from (Grid): the grid of ``field``.
        grid_to (Grid): the grid to resample to.

    Returns:
        Tensor: [..., R', C', S'] cell averages on ``grid_to`` (``field`` itself when the grids are equal).
    """
    if tuple(field.shape[-3:]) != tuple(grid_from.shape):
        raise ValueError(f"field spatial shape {tuple(field.shape[-3:])} != grid shape {grid_from.shape}")
    if grid_from == grid_to:
        return field
    Wr, Wc, Ws = resampling_matrices(grid_from, grid_to, field.device, field.dtype)
    x = torch.matmul(field, Ws.t())  # [..., R, C, S']
    x = torch.matmul(x.transpose(-1, -2), Wc.t()).transpose(-1, -2)  # [..., R, C', S']
    x = torch.matmul(x.movedim(-3, -1), Wr.t()).movedim(-1, -3)  # [..., R', C', S']
    return x.contiguous()


@dataclass(frozen=True)
class ConeGeometry:
    """Flat-panel cone-beam geometry in MBIRTORCH conventions (docs/source/conventions.rst).

    In the view frame the source sits at ``(0, +source_iso_dist, 0)`` and the panel lies in the plane
    ``y_view = source_iso_dist - source_detector_dist``, with u along +x_view (channels) and v along +z (rows).

    Attributes:
        source_iso_dist (float): distance from the source to the rotation axis in cm.
        source_detector_dist (float): distance from the source to the panel in cm.
        num_det_rows (int): number of detector rows.
        num_det_channels (int): number of detector channels.
        delta_det_row (float): detector row pitch in cm.
        delta_det_channel (float): detector channel pitch in cm.
        det_row_offset (float): offset in cm subtracted from the v of every pixel center. Defaults to 0.0.
        det_channel_offset (float): offset in cm subtracted from the u of every pixel center. Defaults to 0.0.
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
        """Magnification at the rotation axis, ``source_detector_dist / source_iso_dist``."""
        return self.source_detector_dist / self.source_iso_dist

    @property
    def panel_y_view(self) -> float:
        """y of the panel plane in the view frame in cm."""
        return self.source_iso_dist - self.source_detector_dist

    @property
    def image_shape(self) -> tuple[int, int]:
        """Shape (num_det_rows, num_det_channels) of a detector image."""
        return (self.num_det_rows, self.num_det_channels)

    @property
    def pixel_area(self) -> float:
        """Area of one detector pixel in cm^2."""
        return self.delta_det_row * self.delta_det_channel

    # ---- panel coordinates --------------------------------------------------------------------
    def pixel_uv(self, device=None, dtype=torch.float32) -> tuple[Tensor, Tensor]:
        """Return the panel coordinates of the pixel centers.

        Args:
            device (torch.device, optional): device of the tensors. Defaults to None.
            dtype (torch.dtype, optional): dtype of the tensors. Defaults to torch.float32.

        Returns:
            (Tensor, Tensor): u [num_det_channels] and v [num_det_rows] in cm.
        """
        nc, nr = self.num_det_channels, self.num_det_rows
        u = ((torch.arange(nc, device=device, dtype=dtype) - (nc - 1) / 2) * self.delta_det_channel
             - self.det_channel_offset)
        v = (torch.arange(nr, device=device, dtype=dtype) - (nr - 1) / 2) * self.delta_det_row - self.det_row_offset
        return u, v

    def panel_half_extent(self) -> tuple[float, float]:
        """Return the half widths of the active panel area, edge to edge (pixel size included).

        Returns:
            (float, float): half widths along u and v in cm.
        """
        return (self.num_det_channels * self.delta_det_channel / 2, self.num_det_rows * self.delta_det_row / 2)

    # ---- recon-frame placement for a view -----------------------------------------------------
    @staticmethod
    def _rot(angle: float) -> tuple[float, float]:
        """Return the cosine and sine of the view angle.

        Args:
            angle (float): view angle in rad.

        Returns:
            (float, float): ``cos(angle)`` and ``sin(angle)``.
        """
        return math.cos(angle), math.sin(angle)

    def view_to_recon(self, p_view: Tensor, angle: float) -> Tensor:
        """Rotate points from the view frame into the recon frame.

        Args:
            p_view (Tensor): [..., 3] points in the view frame.
            angle (float): view angle in rad.

        Returns:
            Tensor: [..., 3] points in the recon frame.
        """
        c, s = self._rot(angle)
        xv, yv, zv = p_view[..., 0], p_view[..., 1], p_view[..., 2]
        return torch.stack([c * xv + s * yv, -s * xv + c * yv, zv], dim=-1)

    def recon_to_view(self, p: Tensor, angle: float) -> Tensor:
        """Rotate points from the recon frame into the view frame.

        Args:
            p (Tensor): [..., 3] points in the recon frame.
            angle (float): view angle in rad.

        Returns:
            Tensor: [..., 3] points in the view frame.
        """
        c, s = self._rot(angle)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        return torch.stack([c * x - s * y, s * x + c * y, z], dim=-1)

    def source_position(self, angle: float) -> tuple[float, float, float]:
        """Return the focal spot position in the recon frame.

        Args:
            angle (float): view angle in rad.

        Returns:
            (float, float, float): (x, y, z) of the focal spot in cm.
        """
        c, s = self._rot(angle)
        return (s * self.source_iso_dist, c * self.source_iso_dist, 0.0)

    def panel_center(self, angle: float) -> tuple[float, float, float]:
        """Return the point where the central ray meets the panel (u = v = 0), in the recon frame.

        Args:
            angle (float): view angle in rad.

        Returns:
            (float, float, float): (x, y, z) of the panel center in cm.
        """
        c, s = self._rot(angle)
        yv = self.panel_y_view
        return (s * yv, c * yv, 0.0)

    def panel_axes(self, angle: float) -> tuple[tuple[float, float, float], tuple[float, float, float],
                                                tuple[float, float, float]]:
        """Return the unit vectors of the panel in the recon frame.

        Args:
            angle (float): view angle in rad.

        Returns:
            (tuple, tuple, tuple): ``u_hat``, ``v_hat`` and ``n_hat`` as (x, y, z); ``n_hat`` points toward the source.
        """
        c, s = self._rot(angle)
        u_hat = (c, -s, 0.0)
        v_hat = (0.0, 0.0, 1.0)
        n_hat = (s, c, 0.0)
        return u_hat, v_hat, n_hat

    def pixel_centers(self, angle: float, device=None, dtype=torch.float32) -> Tensor:
        """Return the pixel-center positions in the recon frame.

        Args:
            angle (float): view angle in rad.
            device (torch.device, optional): device of the tensors. Defaults to None.
            dtype (torch.dtype, optional): dtype of the tensor. Defaults to torch.float32.

        Returns:
            Tensor: [num_det_rows, num_det_channels, 3] positions in cm.
        """
        u, v = self.pixel_uv(device, dtype)
        u_hat, v_hat, _ = self.panel_axes(angle)
        pc = torch.tensor(self.panel_center(angle), device=device, dtype=dtype)
        u_hat = torch.tensor(u_hat, device=device, dtype=dtype)
        v_hat = torch.tensor(v_hat, device=device, dtype=dtype)
        return pc + u[None, :, None] * u_hat + v[:, None, None] * v_hat

    def primary_directions(self, angle: float, device=None, dtype=torch.float32) -> Tensor:
        """Return the unit vectors from the focal spot to each pixel center in the recon frame.

        Args:
            angle (float): view angle in rad.
            device (torch.device, optional): device of the tensors. Defaults to None.
            dtype (torch.dtype, optional): dtype of the tensor. Defaults to torch.float32.

        Returns:
            Tensor: [rows, channels, 3] unit directions.
        """
        source_position = torch.tensor(self.source_position(angle), device=device, dtype=dtype)
        d = self.pixel_centers(angle, device, dtype) - source_position
        return d / torch.linalg.norm(d, dim=-1, keepdim=True)

    def panel_distance_along(self, points: Tensor, omega: Tensor, angle: float) -> Tensor:
        """Compute the signed distance t such that ``points + t * omega`` lies on the panel plane.

        Negative or infinite values mean that the direction does not reach the panel.

        Args:
            points (Tensor): [..., 3] points in the recon frame in cm.
            omega (Tensor): [3] or [..., 3] unit directions.
            angle (float): view angle in rad.

        Returns:
            Tensor: [...] distances in cm.
        """
        _, _, n_hat = self.panel_axes(angle)
        n = torch.tensor(n_hat, device=points.device, dtype=points.dtype)
        pc = torch.tensor(self.panel_center(angle), device=points.device, dtype=points.dtype)
        denom = (omega * n).sum(-1)
        return ((pc - points) * n).sum(-1) / denom

    def panel_uv_of_points(self, points: Tensor, angle: float) -> tuple[Tensor, Tensor]:
        """Return the panel coordinates of points that lie on the panel plane.

        Args:
            points (Tensor): [..., 3] points in the recon frame in cm.
            angle (float): view angle in rad.

        Returns:
            (Tensor, Tensor): u and v [...] in cm.
        """
        u_hat, v_hat, _ = self.panel_axes(angle)
        pc = torch.tensor(self.panel_center(angle), device=points.device, dtype=points.dtype)
        uh = torch.tensor(u_hat, device=points.device, dtype=points.dtype)
        vh = torch.tensor(v_hat, device=points.device, dtype=points.dtype)
        d = points - pc
        return (d * uh).sum(-1), (d * vh).sum(-1)

