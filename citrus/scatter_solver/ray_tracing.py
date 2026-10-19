"""Exact line integrals of voxel fields (Siddon / Amanatides-Woo traversal) and the ``Projectors`` interface.

``Projectors`` is the protocol of the five ray-tracing operations of the scatter solver, each batched over a leading
group axis for one view angle in the recon frame. ``SiddonProjectors`` implements it with the PyTorch traversal of
``line_integrals``, or with the fused Triton kernels of
:mod:`citrus.scatter_solver.kernels.ray_tracing_kernels` on float32 CUDA tensors.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import Tensor

from .geometry import ConeGeometry, Grid, device_tensor
from .kernels import ray_tracing_kernels

__all__ = ["Projectors", "SiddonProjectors", "line_integrals"]


# ------------------------------------------------------------------------------------------------
# interface
# ------------------------------------------------------------------------------------------------

class Projectors(Protocol):
    """Protocol of the five ray-tracing operations of the scatter solver.

    Every operation works on one view angle in the recon frame, batched over a leading group axis G, with the volume
    zero outside the box of its grid. For a ``ParallelGeometry`` the rays "from the focal spot" are the beam's rays,
    which start upstream of the grid box (``ParallelGeometry.ray_starts``).
    """

    def upstream_cumulative(self, mu: Tensor, grid: Grid, geometry: ConeGeometry, angle: float) -> Tensor:
        """Line integral of ``mu`` from the focal spot to every voxel center.

        Args:
            mu (Tensor): [G, rows, cols, slices] attenuation coefficient mu per group in cm^-1.
            grid (Grid): the grid of ``mu``.
            geometry (ConeGeometry or ParallelGeometry): the scan geometry.
            angle (float): view angle in radians.

        Returns:
            Tensor: [G, rows, cols, slices] upstream optical depth ``tau_up`` (dimensionless).
        """
        ...

    def upstream_to_points(self, mu: Tensor, grid: Grid, geometry: ConeGeometry, angle: float,
                           points: Tensor) -> Tensor:
        """Line integral of ``mu`` from the focal spot to arbitrary points.

        Args:
            mu (Tensor): [G, rows, cols, slices] attenuation coefficient mu per group in cm^-1.
            grid (Grid): the grid of ``mu``.
            geometry (ConeGeometry or ParallelGeometry): the scan geometry.
            angle (float): view angle in radians.
            points (Tensor): [N, 3] end points (recon frame, cm), inside or outside the box of ``grid``.

        Returns:
            Tensor: [G, N] optical depth from the focal spot to each point.
        """
        ...

    def primary_project(self, mu: Tensor, grid: Grid, geometry: ConeGeometry, angle: float) -> Tensor:
        """Line integral of ``mu`` from the focal spot to every pixel center.

        With sub-pixel sampling the result is ``-log(mean(exp(-tau)))`` over the sub-rays of each pixel.

        Args:
            mu (Tensor): [G, rows, cols, slices] attenuation coefficient mu per group in cm^-1.
            grid (Grid): the grid of ``mu``.
            geometry (ConeGeometry or ParallelGeometry): the scan geometry.
            angle (float): view angle in radians.

        Returns:
            Tensor: [G, num_det_rows, num_det_channels] optical depth per pixel.
        """
        ...

    def downstream_cumulative(self, mu: Tensor, grid: Grid, geometry: ConeGeometry, angle: float,
                              omegas: Tensor) -> Tensor:
        """Line integral of ``mu`` from every voxel center to the panel plane along each direction.

        Args:
            mu (Tensor): [G, rows, cols, slices] attenuation coefficient mu per group in cm^-1.
            grid (Grid): the grid of ``mu``.
            geometry (ConeGeometry or ParallelGeometry): the scan geometry.
            angle (float): view angle in radians.
            omegas (Tensor): [K, 3] unit directions (recon frame).

        Returns:
            Tensor: [K, G, rows, cols, slices] downstream optical depth ``tau_down``; +inf for directions that do
            not head toward the panel.
        """
        ...

    def oblique_project(self, field: Tensor, grid: Grid, geometry: ConeGeometry, angle: float,
                        omegas: Tensor) -> Tensor:
        """Line integral of a field along each direction through every pixel center.

        Args:
            field (Tensor): [K, G, rows, cols, slices], or [G, rows, cols, slices] shared by all directions.
            grid (Grid): the grid of ``field``.
            geometry (ConeGeometry or ParallelGeometry): the scan geometry.
            angle (float): view angle in radians.
            omegas (Tensor): [K, 3] unit directions (recon frame).

        Returns:
            Tensor: [K, G, num_det_rows, num_det_channels] line integrals in units of field times cm.
        """
        ...


# ------------------------------------------------------------------------------------------------
# core
# ------------------------------------------------------------------------------------------------

def _box_tensors(grid: Grid, device, dtype) -> tuple[Tensor, Tensor, Tensor, tuple[int, int, int]]:
    """Return the corners, voxel sizes and voxel counts of the grid box in (x, y, z) order.

    Args:
        grid (Grid): the grid.
        device (torch.device): device of the returned tensors.
        dtype (torch.dtype): dtype of the returned tensors.

    Returns:
        (lo, hi, delta, n): lo and hi are the [3] lower and upper corners in cm, delta the [3] voxel sizes in cm and
        n the voxel counts (nx, ny, nz).
    """
    lo = device_tensor(grid.lower_corner(), device, dtype)
    delta = device_tensor(grid.deltas, device, dtype)
    n = (grid.num_cols, grid.num_rows, grid.num_slices)
    hi = lo + delta * device_tensor(n, device, dtype)
    return lo, hi, delta, n


def _trace_chunk(flat_volume: Tensor, lo: Tensor, hi: Tensor, delta: Tensor, n: tuple[int, int, int],
                 planes: list[Tensor], p0: Tensor, p1: Tensor, volume_offset: Tensor | None = None) -> Tensor:
    """Exact integrals of a flattened voxel field along the segments ``p0 -> p1`` of one chunk of rays.

    Args:
        flat_volume (Tensor): [V, G] volume flattened in (row, col, slice) order, or [K*V, G] for K stacked volumes.
        lo (Tensor): [3] lower corner of the grid box (x, y, z) in cm.
        hi (Tensor): [3] upper corner of the grid box (x, y, z) in cm.
        delta (Tensor): [3] voxel sizes (x, y, z) in cm.
        n (tuple[int, int, int]): voxel counts (nx, ny, nz).
        planes (list[Tensor]): interior plane coordinates per axis (x, y, z) in cm, each [n_a - 1].
        p0 (Tensor): [N, 3] segment starts (x, y, z) in cm.
        p1 (Tensor): [N, 3] segment ends (x, y, z) in cm.
        volume_offset (Tensor, optional): [N] row offset ``k * V`` of each ray's volume in ``flat_volume``.
            Defaults to None.

    Returns:
        Tensor: [N, G] line integrals in units of the field times cm.
    """
    d = p1 - p0  # [N, 3]
    nonzero = d != 0
    inv = 1.0 / torch.where(nonzero, d, torch.ones_like(d))
    a1 = (lo - p0) * inv
    a2 = (hi - p0) * inv
    inside_slab = (p0 >= lo) & (p0 <= hi)
    inf = torch.full_like(d, math.inf)
    amin = torch.where(nonzero, torch.minimum(a1, a2), torch.where(inside_slab, -inf, inf))
    amax = torch.where(nonzero, torch.maximum(a1, a2), torch.where(inside_slab, inf, -inf))
    a_lo = amin.amax(dim=-1).clamp(min=0.0)  # [N]
    a_hi = amax.amin(dim=-1).clamp(max=1.0)
    valid = a_hi > a_lo  # empty intersection -> zero-length segment at a = 0 (keeps everything finite)
    a_lo = torch.where(valid, a_lo, torch.zeros_like(a_lo))
    a_hi = torch.where(valid, a_hi, torch.zeros_like(a_hi))

    lists = [a_lo[:, None], a_hi[:, None]]
    for axis in range(3):
        axis_planes = planes[axis]
        if axis_planes.numel() == 0:
            continue
        a = (axis_planes[None, :] - p0[:, axis:axis + 1]) * inv[:, axis:axis + 1]  # [N, n_a - 1]
        a = torch.where(nonzero[:, axis:axis + 1], a, a_lo[:, None])
        a = torch.minimum(torch.maximum(a, a_lo[:, None]), a_hi[:, None])
        lists.append(a)
    alphas = torch.cat(lists, dim=1).sort(dim=1).values  # [N, P]
    segment = alphas[:, 1:] - alphas[:, :-1]  # [N, P-1]
    midpoint = 0.5 * (alphas[:, 1:] + alphas[:, :-1])
    del alphas

    nx, ny, nz = n
    flat = None
    strides = (nz, nx * nz, 1)  # x -> col stride, y -> row stride, z -> slice stride
    for axis, (cells_on_axis, stride) in enumerate(zip(n, strides)):
        pos = p0[:, axis:axis + 1] + midpoint * d[:, axis:axis + 1]
        cell_index = torch.floor((pos - lo[axis]) / delta[axis]).clamp_(0, cells_on_axis - 1).long()
        flat = cell_index * stride if flat is None else flat + cell_index * stride
    del midpoint
    if volume_offset is not None:
        flat = flat + volume_offset[:, None]

    length = segment * torch.linalg.norm(d, dim=-1, keepdim=True)  # [N, P-1]
    values = flat_volume.index_select(0, flat.reshape(-1)).reshape(*flat.shape, flat_volume.shape[1])  # [N, P-1, G]
    return torch.einsum("npg,np->ng", values, length)


def line_integrals(volume: Tensor, starts: Tensor, ends: Tensor, grid: Grid, *,
                   ray_chunk: int = 1 << 20, element_budget: int = 1 << 23,
                   volume_index: Tensor | None = None) -> Tensor:
    """Exact line integrals of a voxel field along the segments ``starts -> ends``, clipped to the grid box.

    Rays are processed in chunks of at most ``ray_chunk`` rays and ``element_budget`` (ray, plane) pairs.

    Args:
        volume (Tensor): [G, rows, cols, slices] field, zero outside the grid box, or [K, G, rows, cols, slices]
            with ``volume_index``.
        starts (Tensor): [N, 3] segment starts (x, y, z) in the recon frame in cm.
        ends (Tensor): [N, 3] segment ends (x, y, z) in the recon frame in cm.
        grid (Grid): the grid of ``volume``.
        ray_chunk (int, optional): maximum rays per chunk. Defaults to 2**20.
        element_budget (int, optional): maximum (ray, plane) pairs per chunk. Defaults to 2**23.
        volume_index (Tensor, optional): [N] integers in [0, K): ray n integrates ``volume[volume_index[n]]``.
            Required for a 5-D ``volume``. Defaults to None.

    Returns:
        Tensor: [G, N] line integrals in units of the field times cm, with the device and dtype of ``volume``.
    """
    if volume.dim() == 5:
        if volume_index is None:
            raise ValueError("a [K, G, rows, cols, slices] volume needs volume_index [N]")
        K = volume.shape[0]
        V = grid.num_voxels
        if tuple(volume.shape[2:]) != tuple(grid.shape):
            raise ValueError(f"volume spatial shape {tuple(volume.shape[2:])} != grid shape {grid.shape}")
        # [K, G, R, C, S] -> [K*V, G]: row k*V + v holds voxel v of volume k
        flat_volume = (volume.reshape(K, volume.shape[1], V).permute(0, 2, 1).reshape(K * V, volume.shape[1])
                       .contiguous())
        offset = volume_index.to(device=volume.device, dtype=torch.long).reshape(-1) * V
        return _line_integrals_flat(flat_volume, offset, starts, ends, grid, ray_chunk, element_budget)
    if volume.dim() != 4:
        raise ValueError(f"volume must be [G, rows, cols, slices], got {tuple(volume.shape)}")
    if tuple(volume.shape[1:]) != tuple(grid.shape):
        raise ValueError(f"volume spatial shape {tuple(volume.shape[1:])} != grid shape {grid.shape}")
    G = volume.shape[0]
    flat_volume = volume.reshape(G, -1).t().contiguous()  # [V, G]
    return _line_integrals_flat(flat_volume, None, starts, ends, grid, ray_chunk, element_budget)


def _line_integrals_flat(flat_volume: Tensor, offset: Tensor | None, starts: Tensor, ends: Tensor, grid: Grid,
                         ray_chunk: int, element_budget: int) -> Tensor:
    """Compute ``line_integrals`` on a flattened volume, chunk by chunk.

    Args:
        flat_volume (Tensor): [V, G] or [K*V, G] volume flattened in (row, col, slice) order.
        offset (Tensor | None): [N] row offset of each ray's volume in ``flat_volume``, or None.
        starts (Tensor): [N, 3] segment starts in cm.
        ends (Tensor): [N, 3] segment ends in cm.
        grid (Grid): the grid of the volume.
        ray_chunk (int): maximum rays per chunk.
        element_budget (int): maximum (ray, plane) pairs per chunk.

    Returns:
        Tensor: [G, N] line integrals in units of the field times cm.
    """
    device, dtype = flat_volume.device, flat_volume.dtype
    starts = starts.to(device=device, dtype=dtype).reshape(-1, 3)
    ends = ends.to(device=device, dtype=dtype).reshape(-1, 3)
    if starts.shape != ends.shape:
        raise ValueError("starts and ends must have the same number of rays")
    G = flat_volume.shape[1]
    N = starts.shape[0]
    if offset is not None and offset.shape[0] != N:
        raise ValueError("volume_index must have one entry per ray")
    out = torch.empty(G, N, device=device, dtype=dtype)
    if N == 0:
        return out

    lo, hi, delta, n = _box_tensors(grid, device, dtype)
    planes = [lo[a] + delta[a] * torch.arange(1, n[a], device=device, dtype=dtype) for a in range(3)]
    num_planes = sum(n) - 1
    per_chunk = max(1, min(ray_chunk, element_budget // max(1, num_planes)))
    for s in range(0, N, per_chunk):
        e = min(N, s + per_chunk)
        off = None if offset is None else offset[s:e]
        out[:, s:e] = _trace_chunk(flat_volume, lo, hi, delta, n, planes, starts[s:e], ends[s:e], off).t()
    return out


# ------------------------------------------------------------------------------------------------
# projector interface
# ------------------------------------------------------------------------------------------------

class SiddonProjectors:
    """Exact (Siddon) implementation of the ``Projectors`` protocol, the default projector of the solver.

    Float32 CUDA tensors use the fused Triton kernels when they pass their per-device self-check; float64 and CPU
    tensors always use the PyTorch traversal.

    Args:
        ray_chunk (int, optional): maximum rays per chunk of the PyTorch traversal. Defaults to 2**20.
        element_budget (int, optional): maximum (ray, plane) pairs per chunk of the PyTorch traversal.
            Defaults to 2**23.
        samples_per_pixel_axis (int, optional): default sub-rays per pixel axis of ``primary_project``.
            Defaults to 1.
        backend (str, optional): "auto" (Triton where usable, else PyTorch), "torch" (the PyTorch reference) or
            "triton" (raises if the kernels are unusable). Defaults to "auto".

    Attributes:
        ray_chunk (int): maximum rays per chunk of the PyTorch traversal.
        element_budget (int): maximum (ray, plane) pairs per chunk of the PyTorch traversal.
        samples_per_pixel_axis (int): default sub-rays per pixel axis of ``primary_project``.
        backend (str): "auto", "torch" or "triton".
    """

    def __init__(self, ray_chunk: int = 1 << 20, element_budget: int = 1 << 23, samples_per_pixel_axis: int = 1,
                 backend: str = "auto"):
        if samples_per_pixel_axis < 1:
            raise ValueError("samples_per_pixel_axis must be >= 1")
        if backend not in ("auto", "torch", "triton"):
            raise ValueError("backend must be 'auto', 'torch' or 'triton'")
        if backend == "triton":
            ok, why = ray_tracing_kernels.triton_available()
            if not ok:
                raise RuntimeError(f"backend='triton' requested but Triton is unusable: {why}")
        self.ray_chunk = int(ray_chunk)
        self.element_budget = int(element_budget)
        self.samples_per_pixel_axis = int(samples_per_pixel_axis)
        self.backend = backend
        self._triton_usable: dict[str, bool] = {}
        self._negative_normals: dict[tuple, Tensor] = {}  # -n_hat per (n_hat, device) for the Triton +inf flags

    # ---- backend --------------------------------------------------------------------------------
    def kernel_status(self, device: torch.device) -> tuple[bool, str]:
        """Report whether float32 tensors on ``device`` use the Triton kernels under this instance's backend.

        Args:
            device (torch.device): the device to check.

        Returns:
            (bool, str): whether the kernels are used, and the reason.
        """
        if self.backend == "torch":
            return False, "backend='torch'"
        if torch.device(device).type != "cuda":
            return False, "not a CUDA tensor"
        ok, why = ray_tracing_kernels.kernels_usable(torch.device(device))
        if not ok and self.backend == "triton":
            raise RuntimeError(f"backend='triton' requested but the Siddon kernels are unusable: {why}")
        return ok, why

    def _use_triton(self, t: Tensor) -> bool:
        """Decide whether an operation on ``t`` runs the Triton kernels, caching the result per device.

        Args:
            t (Tensor): the input tensor of the operation.

        Returns:
            bool: True for a float32 CUDA tensor on a device where the kernels are usable and allowed.
        """
        if self.backend == "torch" or t.device.type != "cuda" or t.dtype != torch.float32:
            return False
        key = str(t.device)
        ok = self._triton_usable.get(key)
        if ok is None:
            ok = self._triton_usable[key] = self.kernel_status(t.device)[0]
        return ok

    def _toward_panel(self, directions: Tensor, geometry: ConeGeometry, angle: float) -> Tensor:
        """Flag the directions that head toward the panel, ``omega . (-n_hat) > 0``, as the PyTorch path computes it.

        Args:
            directions (Tensor): [K, 3] unit directions (recon frame).
            geometry (ConeGeometry or ParallelGeometry): the scan geometry.
            angle (float): view angle in radians.

        Returns:
            Tensor: [K] uint8 flags, 1 for a direction toward the panel.
        """
        n_hat = tuple(geometry.panel_axes(angle)[2])
        key = (n_hat, str(directions.device), directions.dtype)
        neg = self._negative_normals.get(key)
        if neg is None:
            if len(self._negative_normals) >= 256:
                self._negative_normals.clear()
            neg = self._negative_normals[key] = -device_tensor(n_hat, directions.device, directions.dtype)
        return ((directions * neg).sum(-1) > 0).view(torch.uint8)

    # ---- core ---------------------------------------------------------------------------------
    def line_integrals(self, volume: Tensor, starts: Tensor, ends: Tensor, grid: Grid) -> Tensor:
        """Exact line integrals along the segments ``starts -> ends`` with this instance's backend and chunking.

        Args:
            volume (Tensor): [G, rows, cols, slices] field, zero outside the grid box.
            starts (Tensor): [N, 3] segment starts (x, y, z) in the recon frame in cm.
            ends (Tensor): [N, 3] segment ends (x, y, z) in the recon frame in cm.
            grid (Grid): the grid of ``volume``.

        Returns:
            Tensor: [G, N] line integrals in units of the field times cm.
        """
        if self._use_triton(volume):
            return ray_tracing_kernels.line_integrals_triton(volume, starts, ends, grid)
        return line_integrals(volume, starts, ends, grid, ray_chunk=self.ray_chunk,
                              element_budget=self.element_budget)

    # ---- (a) ----------------------------------------------------------------------------------
    def upstream_cumulative(self, mu: Tensor, grid: Grid, geometry: ConeGeometry, angle: float) -> Tensor:
        """Line integral of ``mu`` from the focal spot to every voxel center.

        Args:
            mu (Tensor): [G, rows, cols, slices] attenuation coefficient mu per group in cm^-1.
            grid (Grid): the grid of ``mu``.
            geometry (ConeGeometry or ParallelGeometry): the scan geometry.
            angle (float): view angle in radians.

        Returns:
            Tensor: [G, rows, cols, slices] upstream optical depth ``tau_up`` (dimensionless).
        """
        if self._use_triton(mu):
            return ray_tracing_kernels.upstream_cumulative(mu, grid, geometry, angle)
        centers = grid.center_coordinates(mu.device, mu.dtype).reshape(-1, 3)
        if geometry.is_parallel:  # each ray starts upstream of the box on the beam's line through its end
            return self.line_integrals(mu, geometry.ray_starts(centers, angle, grid), centers, grid).reshape(mu.shape)
        source_position = device_tensor(geometry.source_position(angle), mu.device, mu.dtype)
        tau = self.line_integrals(mu, source_position.expand_as(centers), centers, grid)
        return tau.reshape(mu.shape)

    def upstream_to_points(self, mu: Tensor, grid: Grid, geometry: ConeGeometry, angle: float,
                           points: Tensor) -> Tensor:
        """Line integral of ``mu`` from the focal spot to each point, clipped to the box of ``grid``.

        Args:
            mu (Tensor): [G, rows, cols, slices] attenuation coefficient mu per group in cm^-1.
            grid (Grid): the grid of ``mu``.
            geometry (ConeGeometry or ParallelGeometry): the scan geometry.
            angle (float): view angle in radians.
            points (Tensor): [N, 3] end points (recon frame, cm), inside or outside the box of ``grid``.

        Returns:
            Tensor: [G, N] optical depth from the focal spot to each point.
        """
        pts = points.to(device=mu.device, dtype=mu.dtype).reshape(-1, 3)
        if geometry.is_parallel:
            return self.line_integrals(mu, geometry.ray_starts(pts, angle, grid), pts, grid)
        source_position = device_tensor(geometry.source_position(angle), mu.device, mu.dtype)
        return self.line_integrals(mu, source_position.expand_as(pts), pts, grid)

    # ---- (b) ----------------------------------------------------------------------------------
    def primary_project(self, mu: Tensor, grid: Grid, geometry: ConeGeometry, angle: float,
                        samples_per_pixel_axis: int | None = None) -> Tensor:
        """Line integral of ``mu`` from the focal spot to every pixel center, or averaged over sub-rays.

        With ``n = samples_per_pixel_axis > 1``, ``n * n`` sub-rays on a midpoint lattice inside each pixel are traced
        and the result is ``-log(mean(exp(-tau)))``.

        Args:
            mu (Tensor): [G, rows, cols, slices] attenuation coefficient mu per group in cm^-1.
            grid (Grid): the grid of ``mu``.
            geometry (ConeGeometry or ParallelGeometry): the scan geometry.
            angle (float): view angle in radians.
            samples_per_pixel_axis (int, optional): sub-rays per pixel axis. Defaults to None (the instance's
                ``samples_per_pixel_axis``).

        Returns:
            Tensor: [G, num_det_rows, num_det_channels] optical depth per pixel.
        """
        samples_per_pixel_axis = (self.samples_per_pixel_axis if samples_per_pixel_axis is None
                                  else int(samples_per_pixel_axis))
        if self._use_triton(mu):
            return ray_tracing_kernels.primary_project(mu, grid, geometry, angle,
                                                       samples_per_pixel_axis=samples_per_pixel_axis)
        device, dtype = mu.device, mu.dtype
        G = mu.shape[0]
        pixels = geometry.pixel_centers(angle, device, dtype)  # [Rd, Cd, 3]
        Rd, Cd = pixels.shape[:2]
        if geometry.is_parallel:  # the beam's rays through the pixel (or sub-pixel) centers
            n = samples_per_pixel_axis
            ends = pixels.reshape(-1, 3) if n == 1 else geometry.subpixel_centers(angle, n, device, dtype).reshape(-1, 3)
            tau = self.line_integrals(mu, geometry.ray_starts(ends, angle, grid), ends, grid).reshape(G, Rd, Cd, n * n)
            return tau[..., 0] if n == 1 else -(torch.logsumexp(-tau, dim=-1) - math.log(n * n))
        source_position = device_tensor(geometry.source_position(angle), device, dtype)
        if samples_per_pixel_axis == 1:
            ends = pixels.reshape(-1, 3)
            tau = self.line_integrals(mu, source_position.expand_as(ends), ends, grid)
            return tau.reshape(G, Rd, Cd)
        n = samples_per_pixel_axis
        if geometry.use_curved_detector:  # sub-pixels on the cylinder
            ends = geometry.subpixel_centers(angle, n, device, dtype).reshape(-1, 3)
            tau = self.line_integrals(mu, source_position.expand_as(ends), ends, grid).reshape(G, Rd, Cd, n * n)
            return -(torch.logsumexp(-tau, dim=-1) - math.log(n * n))
        u_hat, v_hat, _ = geometry.panel_axes(angle)
        u_axis = device_tensor(u_hat, device, dtype)
        v_axis = device_tensor(v_hat, device, dtype)
        offsets = (2 * torch.arange(n, device=device, dtype=dtype) + 1) / (2 * n) - 0.5  # [n]
        offset_u = offsets * geometry.delta_det_channel
        offset_v = offsets * geometry.delta_det_row
        shift = offset_v[:, None, None] * v_axis + offset_u[None, :, None] * u_axis  # [n_v, n_u, 3]
        ends = (pixels[:, :, None, None, :] + shift).reshape(-1, 3)  # [Rd*Cd*n*n, 3]
        tau = self.line_integrals(mu, source_position.expand_as(ends), ends, grid).reshape(G, Rd, Cd, n * n)
        return -(torch.logsumexp(-tau, dim=-1) - math.log(n * n))

    # ---- (c) ----------------------------------------------------------------------------------
    def downstream_cumulative(self, mu: Tensor, grid: Grid, geometry: ConeGeometry, angle: float,
                              omegas: Tensor) -> Tensor:
        """Line integral of ``mu`` from every voxel center to the panel plane along each direction.

        Args:
            mu (Tensor): [G, rows, cols, slices] attenuation coefficient mu per group in cm^-1.
            grid (Grid): the grid of ``mu``.
            geometry (ConeGeometry or ParallelGeometry): the scan geometry.
            angle (float): view angle in radians.
            omegas (Tensor): [K, 3] unit directions (recon frame).

        Returns:
            Tensor: [K, G, rows, cols, slices] downstream optical depth ``tau_down``; +inf where
            ``omega . (-n_hat) <= 0`` and 0 for voxels beyond the panel plane.
        """
        if self._use_triton(mu):
            directions = omegas.to(device=mu.device, dtype=mu.dtype).reshape(-1, 3).contiguous()
            return ray_tracing_kernels.downstream_cumulative(mu, grid, geometry, angle, directions,
                                                             toward=self._toward_panel(directions, geometry, angle))
        device, dtype = mu.device, mu.dtype
        G = mu.shape[0]
        omegas = omegas.to(device=device, dtype=dtype).reshape(-1, 3)
        K = omegas.shape[0]
        out = torch.full((K, *mu.shape), math.inf, device=device, dtype=dtype)
        _, _, n_hat = geometry.panel_axes(angle)
        n = device_tensor(n_hat, device, dtype)
        toward = (omegas * (-n)).sum(-1) > 0
        toward_indices = torch.nonzero(toward).flatten()
        if toward_indices.numel() == 0:
            return out
        centers = grid.center_coordinates(device, dtype).reshape(-1, 3)  # [V, 3]
        V = centers.shape[0]
        diagonal = math.sqrt(sum(e * e for e in grid.extent)) * (1 + 1e-9) + 1e-9
        # batches of directions with at most ray_chunk rays keep the endpoint tensors small
        directions_per_batch = max(1, self.ray_chunk // max(1, V))
        for s in range(0, toward_indices.numel(), directions_per_batch):
            batch_indices = toward_indices[s:s + directions_per_batch]
            directions = omegas[batch_indices]  # [k, 3]
            t = geometry.panel_distance_along(centers[None, :, :], directions[:, None, :], angle)  # [k, V]
            t = t.clamp(min=0.0, max=diagonal)  # the box exit precedes `diagonal` for points inside the box
            ends = centers[None] + t[..., None] * directions[:, None, :]
            starts = centers[None].expand_as(ends)
            tau = self.line_integrals(mu, starts.reshape(-1, 3), ends.reshape(-1, 3), grid)
            out[batch_indices] = tau.reshape(G, batch_indices.numel(), *mu.shape[1:]).transpose(0, 1)
        return out

    # ---- (d) ----------------------------------------------------------------------------------
    def oblique_project(self, field: Tensor, grid: Grid, geometry: ConeGeometry, angle: float,
                        omegas: Tensor) -> Tensor:
        """Full line integral of a field along the line through each pixel center with direction ``omega_k``.

        The line is clipped to the grid box on both sides of the pixel center.

        Args:
            field (Tensor): [K, G, rows, cols, slices], or [G, rows, cols, slices] shared by all directions.
            grid (Grid): the grid of ``field``.
            geometry (ConeGeometry or ParallelGeometry): the scan geometry.
            angle (float): view angle in radians.
            omegas (Tensor): [K, 3] directions (recon frame).

        Returns:
            Tensor: [K, G, num_det_rows, num_det_channels] line integrals in units of field times cm.
        """
        if self._use_triton(field):
            return ray_tracing_kernels.oblique_project(field, grid, geometry, angle, omegas)
        device, dtype = field.device, field.dtype
        omegas = omegas.to(device=device, dtype=dtype).reshape(-1, 3)
        K = omegas.shape[0]
        pixels = geometry.pixel_centers(angle, device, dtype)
        Rd, Cd = pixels.shape[:2]
        p = pixels.reshape(-1, 3)  # [Np, 3]
        lo = device_tensor(grid.lower_corner(), device, dtype)
        center = lo + 0.5 * device_tensor(grid.extent, device, dtype)
        half = 0.5 * math.sqrt(sum(e * e for e in grid.extent)) * (1 + 1e-6) + 1e-6

        def endpoints(directions: Tensor) -> tuple[Tensor, Tensor]:
            # directions: [k, 3] -> segments [k*Np, 3] centred on the foot of the box center on each line
            omega_squared = (directions * directions).sum(-1, keepdim=True)  # [k, 1] (omegas need not be exactly unit)
            center_parameter = ((center - p)[None] * directions[:, None, :]).sum(-1) / omega_squared  # [k, Np]
            h = half / omega_squared.sqrt()  # [k, 1]
            a = p[None] + (center_parameter - h)[..., None] * directions[:, None, :]
            b = p[None] + (center_parameter + h)[..., None] * directions[:, None, :]
            return a.reshape(-1, 3), b.reshape(-1, 3)

        if field.dim() == 4:
            G = field.shape[0]
            a, b = endpoints(omegas)
            image = self.line_integrals(field, a, b, grid)  # [G, K*Np]
            return image.reshape(G, K, Rd, Cd).transpose(0, 1).contiguous()
        if field.dim() != 5 or field.shape[0] != K:
            raise ValueError(f"field must be [G, ...] or [K={K}, G, ...], got {tuple(field.shape)}")
        G = field.shape[1]
        if tuple(field.shape[2:]) != tuple(grid.shape):
            raise ValueError(f"field spatial shape {tuple(field.shape[2:])} != grid shape {grid.shape}")
        # all K directions in one traversal: ray block k (Np rays) integrates field[k]
        a, b = endpoints(omegas)
        volume_index = torch.arange(K, device=device).repeat_interleave(p.shape[0])
        image = line_integrals(field, a, b, grid, ray_chunk=self.ray_chunk, element_budget=self.element_budget,
                               volume_index=volume_index)  # [G, K*Np]
        return image.reshape(G, K, Rd, Cd).transpose(0, 1).contiguous()
