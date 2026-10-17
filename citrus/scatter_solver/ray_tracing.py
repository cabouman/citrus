"""Exact ray-box line integrals (Siddon / Amanatides-Woo parametric traversal) and the `Projectors` interface.

`Projectors` is the protocol of the five field-level ray-tracing operations the scatter solver needs (all batched
over a leading group axis G, for one view angle, in the recon frame, the volume being zero outside the grid box):

    upstream_cumulative    line integral from the focal spot to every voxel center          (uncollided fluence)
    upstream_to_points     line integral from the focal spot to arbitrary points           (uncollided fluence at the
                                                                                            quadrature-grid points)
    primary_project        line integral from the focal spot to every pixel center          (primary image)
    downstream_cumulative  line integral from every voxel center to the panel along Omega_k  (scatter attenuation)
    oblique_project        line integral of a field along Omega_k through every pixel center (scatter projection)

`SiddonProjectors` implements it exactly. Core idea (Siddon 1985): parametrize a segment as p(a) = p0 + a (p1 - p0),
a in [0, 1]. Clip it to the grid box (slab method) giving [a_lo, a_hi]; the parameters where it crosses the interior
voxel planes of each axis are clamped to [a_lo, a_hi], merged with a_lo and a_hi, and sorted. Consecutive sorted
parameters bound sub-segments that each lie inside exactly one voxel, found from the sub-segment midpoint. The
integral is sum(value(voxel) * length). Zero-length sub-segments (from clamped or coincident crossings, e.g. through
voxel corners) contribute nothing, so the result is exact to rounding for any segment, including axis-aligned ones
and ones starting/ending inside.

The PyTorch path below is batched over rays with a fixed number of planes per ray (the merged list always has
nx + ny + nz - 1 entries), so there is no per-ray Python loop. Memory is bounded by chunking over rays: a chunk
holds at most `ray_chunk` rays and at most `element_budget` (ray, plane) pairs. On float32 CUDA tensors the fused
Triton kernels of `kernels/ray_tracing_kernels.py` are used instead (same results to float32 rounding).
"""

from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import Tensor

from .geometry import ConeGeometry, Grid
from .kernels import ray_tracing_kernels

__all__ = ["Projectors", "SiddonProjectors", "line_integrals"]


# ------------------------------------------------------------------------------------------------
# interface
# ------------------------------------------------------------------------------------------------

class Projectors(Protocol):
    def upstream_cumulative(self, mu: Tensor, grid: Grid, geometry: ConeGeometry, angle: float) -> Tensor:
        """Line integral of mu from the focal spot to every voxel center.

        mu: [G, rows, cols, slices] (1/cm). Returns tau_up: [G, rows, cols, slices] (dimensionless).
        """
        ...

    def upstream_to_points(self, mu: Tensor, grid: Grid, geometry: ConeGeometry, angle: float,
                           points: Tensor) -> Tensor:
        """Line integral of mu (on `grid`, zero outside its box) from the focal spot to arbitrary points.

        mu: [G, rows, cols, slices]; points: [N, 3] (recon frame, cm; inside or outside the box).
        Returns [G, N]. Used for the uncollided fluence at the quadrature-grid points traced through another grid.
        """
        ...

    def primary_project(self, mu: Tensor, grid: Grid, geometry: ConeGeometry, angle: float) -> Tensor:
        """Line integral of mu from the focal spot to every pixel center.

        mu: [G, rows, cols, slices]. Returns [G, num_det_rows, num_det_channels].
        Implementations may sub-sample pixels internally; the returned value is the pixel-average
        of exp(-integral) re-expressed as -log(mean(exp(-integral))) when sub-sampling is used.
        """
        ...

    def downstream_cumulative(self, mu: Tensor, grid: Grid, geometry: ConeGeometry, angle: float,
                              omegas: Tensor) -> Tensor:
        """Line integral of mu from every voxel center to the panel plane along each direction.

        mu: [G, rows, cols, slices]; omegas: [K, 3] unit vectors (recon frame) pointing toward the panel.
        Returns tau_down: [K, G, rows, cols, slices]. Where a direction does not reach the panel
        plane (component along the panel normal toward the panel <= 0) the value is +inf.
        """
        ...

    def oblique_project(self, field: Tensor, grid: Grid, geometry: ConeGeometry, angle: float,
                        omegas: Tensor) -> Tensor:
        """Line integral of a volumetric field along each direction, landing on the panel.

        field: [K, G, rows, cols, slices] or [G, rows, cols, slices] (broadcast over K);
        omegas: [K, 3] unit vectors (recon frame). For pixel (r, c) the line passes through the
        pixel center with direction omega_k. Returns [K, G, num_det_rows, num_det_channels] in units of
        field x cm.
        """
        ...


# ------------------------------------------------------------------------------------------------
# core
# ------------------------------------------------------------------------------------------------

def _box_tensors(grid: Grid, device, dtype) -> tuple[Tensor, Tensor, Tensor, tuple[int, int, int]]:
    """(lo[3], hi[3], delta[3], (nx, ny, nz)) for the grid box in (x, y, z) order."""
    lo = torch.tensor(grid.lower_corner(), device=device, dtype=dtype)
    delta = torch.tensor(grid.deltas, device=device, dtype=dtype)
    n = (grid.num_cols, grid.num_rows, grid.num_slices)
    hi = lo + delta * torch.tensor(n, device=device, dtype=dtype)
    return lo, hi, delta, n


def _trace_chunk(flat_volume: Tensor, lo: Tensor, hi: Tensor, delta: Tensor, n: tuple[int, int, int],
                 planes: list[Tensor], p0: Tensor, p1: Tensor, volume_offset: Tensor | None = None) -> Tensor:
    """Exact integrals of the voxel field along segments p0 -> p1 for one chunk of rays.

    flat_volume: [V, G] volume flattened in (row, col, slice) order (or [K*V, G] for K stacked volumes, with
    volume_offset [N] = k(ray) * V selecting each ray's volume); p0, p1: [N, 3] (x, y, z).
    planes: interior plane coordinates per axis (x, y, z), each [n_a - 1]. Returns [N, G].
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
    """Exact line integrals of a voxel field along segments starts -> ends (recon frame, cm).

    volume: [G, rows, cols, slices] (zero outside the grid box); starts, ends: [N, 3] (x, y, z).
    Alternatively volume: [K, G, rows, cols, slices] with volume_index: [N] integers in [0, K), so ray n
    integrates volume[volume_index[n]] (all K volumes traced in one batched pass).
    Segments are clipped to the box. Returns [G, N] in units of volume x cm, with volume's
    device and dtype. Rays are processed in chunks of at most `ray_chunk` rays and at most
    `element_budget` (ray, plane) pairs.
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
    """line_integrals on a flattened volume flat_volume [V or K*V, G] with optional per-ray offsets [N]. -> [G, N]."""
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
    """Exact (Siddon) implementation of the `Projectors` protocol: the default projector of the solver (backend
    "auto", the fused Triton kernels on float32 CUDA) and, with backend="torch", the PyTorch reference.

    ray_chunk: maximum rays per traversal chunk. element_budget: maximum (ray, plane) pairs per
    chunk; peak working memory is roughly element_budget x (G + ~12) scalars.
    samples_per_pixel_axis: default sub-pixel sampling per axis for `primary_project` (samples_per_pixel_axis^2 sub-rays
    per pixel).

    backend: "auto" (default: the fused Triton ray tracer of `kernels/ray_tracing_kernels.py` for float32 CUDA
      tensors when `ray_tracing_kernels.kernels_usable(device)` passes its per-device self-check against the PyTorch
      path, else PyTorch), "torch" (the PyTorch traversal below: the reference and test oracle), or "triton" (raises
      if the kernels are unusable). float64 and CPU tensors always use PyTorch, so
      "auto" equals "torch" bit for bit there. The kernels need no chunking (ray_chunk / element_budget only
      apply to the PyTorch path) and equal it to float32 rounding (~1e-6 of the field maximum).
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
        """(uses Triton, reason) for float32 tensors on `device` under this instance's backend setting."""
        if self.backend == "torch":
            return False, "backend='torch'"
        if torch.device(device).type != "cuda":
            return False, "not a CUDA tensor"
        ok, why = ray_tracing_kernels.kernels_usable(torch.device(device))
        if not ok and self.backend == "triton":
            raise RuntimeError(f"backend='triton' requested but the Siddon kernels are unusable: {why}")
        return ok, why

    def _use_triton(self, t: Tensor) -> bool:
        if self.backend == "torch" or t.device.type != "cuda" or t.dtype != torch.float32:
            return False
        key = str(t.device)
        ok = self._triton_usable.get(key)
        if ok is None:
            ok = self._triton_usable[key] = self.kernel_status(t.device)[0]
        return ok

    def _toward_panel(self, directions: Tensor, geometry: ConeGeometry, angle: float) -> Tensor:
        """[K] uint8 flags omega . (-n_hat) > 0 with the PyTorch path's expression (cached -n_hat, no upload)."""
        n_hat = tuple(geometry.panel_axes(angle)[2])
        key = (n_hat, str(directions.device), directions.dtype)
        neg = self._negative_normals.get(key)
        if neg is None:
            if len(self._negative_normals) >= 256:
                self._negative_normals.clear()
            neg = self._negative_normals[key] = -torch.tensor(n_hat, device=directions.device, dtype=directions.dtype)
        return ((directions * neg).sum(-1) > 0).view(torch.uint8)

    # ---- core ---------------------------------------------------------------------------------
    def line_integrals(self, volume: Tensor, starts: Tensor, ends: Tensor, grid: Grid) -> Tensor:
        """See module-level `line_integrals`: volume [G, rows, cols, slices], starts/ends [N, 3] -> [G, N]."""
        if self._use_triton(volume):
            return ray_tracing_kernels.line_integrals_triton(volume, starts, ends, grid)
        return line_integrals(volume, starts, ends, grid, ray_chunk=self.ray_chunk,
                              element_budget=self.element_budget)

    # ---- (a) ----------------------------------------------------------------------------------
    def upstream_cumulative(self, mu: Tensor, grid: Grid, geometry: ConeGeometry, angle: float) -> Tensor:
        """Line integral of mu from the focal spot to every voxel center.

        mu: [G, rows, cols, slices] (1/cm). Returns tau_up: [G, rows, cols, slices].
        """
        if self._use_triton(mu):
            return ray_tracing_kernels.upstream_cumulative(mu, grid, geometry, angle)
        centers = grid.center_coordinates(mu.device, mu.dtype).reshape(-1, 3)
        source_position = torch.tensor(geometry.source_position(angle), device=mu.device, dtype=mu.dtype)
        tau = self.line_integrals(mu, source_position.expand_as(centers), centers, grid)
        return tau.reshape(mu.shape)

    def upstream_to_points(self, mu: Tensor, grid: Grid, geometry: ConeGeometry, angle: float,
                           points: Tensor) -> Tensor:
        """Line integral of mu from the focal spot to each of points [N, 3] (recon frame; the segment is clipped to the
        box of `grid`, so a point outside the box gets the integral of the part of its ray inside it).

        mu: [G, rows, cols, slices] on grid. Returns [G, N]. One ray per point (`line_integrals`; the fused kernel on
        float32 CUDA): the cost follows the number of points and the traversal of `grid`.
        """
        pts = points.to(device=mu.device, dtype=mu.dtype).reshape(-1, 3)
        source_position = torch.tensor(geometry.source_position(angle), device=mu.device, dtype=mu.dtype)
        return self.line_integrals(mu, source_position.expand_as(pts), pts, grid)

    # ---- (b) ----------------------------------------------------------------------------------
    def primary_project(self, mu: Tensor, grid: Grid, geometry: ConeGeometry, angle: float,
                        samples_per_pixel_axis: int | None = None) -> Tensor:
        """Line integral of mu from the focal spot to every pixel center.

        mu: [G, rows, cols, slices]. Returns [G, num_det_rows, num_det_channels].
        With samples_per_pixel_axis > 1 (per axis), samples_per_pixel_axis^2 sub-rays on a regular midpoint lattice
        inside each pixel are traced and the result is -log(mean(exp(-integral))).
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
        source_position = torch.tensor(geometry.source_position(angle), device=device, dtype=dtype)
        if samples_per_pixel_axis == 1:
            ends = pixels.reshape(-1, 3)
            tau = self.line_integrals(mu, source_position.expand_as(ends), ends, grid)
            return tau.reshape(G, Rd, Cd)
        u_hat, v_hat, _ = geometry.panel_axes(angle)
        u_axis = torch.tensor(u_hat, device=device, dtype=dtype)
        v_axis = torch.tensor(v_hat, device=device, dtype=dtype)
        n = samples_per_pixel_axis
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
        """Line integral of mu from every voxel center to the panel plane along each direction.

        mu: [G, rows, cols, slices]; omegas: [K, 3]. Returns tau_down [K, G, rows, cols, slices];
        +inf where omega . (-n_hat) <= 0 (direction does not head toward the panel). Voxels on the
        far side of the panel plane (never the case in CT) get 0.
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
        n = torch.tensor(n_hat, device=device, dtype=dtype)
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
        """Full line integral of a field along the line through each pixel center with direction omega_k.

        field: [K, G, rows, cols, slices] or [G, rows, cols, slices] (shared over K); omegas: [K, 3].
        Returns [K, G, num_det_rows, num_det_channels] (field x cm). The line is clipped to the box
        on both sides of the pixel center.
        """
        if self._use_triton(field):
            return ray_tracing_kernels.oblique_project(field, grid, geometry, angle, omegas)
        device, dtype = field.device, field.dtype
        omegas = omegas.to(device=device, dtype=dtype).reshape(-1, 3)
        K = omegas.shape[0]
        pixels = geometry.pixel_centers(angle, device, dtype)
        Rd, Cd = pixels.shape[:2]
        p = pixels.reshape(-1, 3)  # [Np, 3]
        lo = torch.tensor(grid.lower_corner(), device=device, dtype=dtype)
        center = lo + 0.5 * torch.tensor(grid.extent, device=device, dtype=dtype)
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
