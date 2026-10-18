"""Batched linear-discontinuous discrete-ordinates sweep on the transport grid.

:func:`~citrus.scatter_solver.transport_sweep.sweep` transports a linear-discontinuous source for every direction of
the octant-reordered quadrature and every energy group through a cuboid grid with vacuum boundaries, one diagonal
wavefront plane at a time, and returns the spherical-harmonic moments
``phi[g, n, k, cell] = sum_m w_m Y_n(Omega_m) psi_m[g, k, cell]`` of the angular fluence. Each plane step runs either
in PyTorch or as one fused Triton kernel (:mod:`citrus.scatter_solver.kernels.sweep_kernels`). The algorithm and the
storage layout are described in docs/source/method.rst, "Transport sweep", and docs/source/design_notes.rst,
"Transport sweep and source iteration".
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union

import torch
from torch import Tensor

from .angular_quadrature import Quadrature, SphericalHarmonicBasis, spherical_harmonic_basis
from .geometry import Grid
from .kernels import sweep_kernels
from .linear_discontinuous_cell import cell_right_hand_sides, cell_solve
from .wavefront import OCTANT_SIGNS, SweepPlan, build_sweep_plan

__all__ = [
    "SourceFunction",
    "Backend",
    "PlaneIndex",
    "SweepContext",
    "make_sweep_context",
    "sweep",
    "source_function_accepts_out",
]

SourceFunction = Callable[[Tensor], Tensor]
"""source_function(direction_index [Dsel] int64, reordered direction indices) -> linear-discontinuous source moments
[Dsel, G, 4, R, C, S]. Optionally source_function(direction_index, out=buffer) writes into the contiguous buffer
[Dsel, G, 4, R, C, S] and returns it (see `sweep`)."""

Backend = Literal["auto", "torch", "triton"]
_BACKENDS = ("auto", "torch", "triton")


@dataclass(frozen=True, eq=False)
class PlaneIndex:
    """Gather and scatter indices of one wavefront plane, int64 on the sweep device.

    Attributes:
        octant (Tensor): [E] octant code of each entry on the plane.
        cell (Tensor): [E] flat cell index, row-major over [rows, cols, slices].
        octant_column (Tensor): [E, 1] octant codes shaped to broadcast against ``upstream_cell``.
        upstream_cell (Tensor): [E, 3] flat index of the upstream neighbor along x, y, z; num_cells (the zero vacuum
            cell) on a vacuum face.
    """

    octant: Tensor
    cell: Tensor
    octant_column: Tensor
    upstream_cell: Tensor

    @property
    def num_entries(self) -> int:
        """Number of entries E on the plane."""
        return int(self.octant.shape[0])


@dataclass(frozen=True, eq=False)
class SweepContext:
    """Everything the sweep needs that depends only on the grid and the quadrature.

    The quadrature is reordered by octant: reordered direction ``m = octant * D8 + d``, with D8 directions per octant.

    Attributes:
        shape (tuple[int, int, int]): (rows, cols, slices) of the grid.
        deltas (tuple[float, float, float]): (dx, dy, dz) cell sizes in cm.
        L (int): spherical-harmonic degree of the returned moments.
        plan (SweepPlan): wavefront plan on the sweep device.
        quadrature (Quadrature): the quadrature reordered by octant.
        permutation (Tensor): [M] int64 original index of each reordered direction.
        basis (SphericalHarmonicBasis): tables of degree L on the reordered quadrature.
        directions_per_octant (int): number of directions per octant, D8.
        c (Tensor): [8, D8, 3] ``|Omega_a| / Delta_a`` of direction (octant, d) in cm^-1.
        s (Tensor): [8, D8, 3] ``sign(Omega_a)``, +1 for a zero component.
        planes (tuple[PlaneIndex, ...]): one index set per wavefront plane, in sweep order.
        backend (str): default plane-step backend of the sweeps, "auto", "torch" or "triton". Defaults to "auto".
        flat_index (FlatPlaneIndex, optional): plane-sorted int32 entry tables of the fused Triton kernel, CUDA contexts
            only. Defaults to None.
    """

    shape: tuple[int, int, int]
    deltas: tuple[float, float, float]
    L: int
    plan: SweepPlan
    quadrature: Quadrature
    permutation: Tensor
    basis: SphericalHarmonicBasis
    directions_per_octant: int
    c: Tensor
    s: Tensor
    planes: tuple[PlaneIndex, ...]
    backend: str = "auto"
    flat_index: Optional[sweep_kernels.FlatPlaneIndex] = None

    @property
    def num_cells(self) -> int:
        """Number of grid cells N."""
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def num_planes(self) -> int:
        """Number of wavefront planes."""
        return len(self.planes)

    @property
    def num_directions(self) -> int:
        """Number of directions M, 8 times D8."""
        return 8 * self.directions_per_octant

    @property
    def num_moments(self) -> int:
        """Number of spherical-harmonic moments, ``(L+1)^2``."""
        return (self.L + 1) ** 2

    @property
    def device(self) -> torch.device:
        """Device of the direction tables."""
        return self.c.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating dtype of the direction tables."""
        return self.c.dtype

    def direction_indices(self, start: int, stop: int, octants: Optional[Tensor] = None) -> Tensor:
        """Reordered direction indices of the per-octant chunk ``start <= d < stop``, octant-major.

        Args:
            start (int): first per-octant direction index of the chunk.
            stop (int): end (exclusive) of the chunk.
            octants (Tensor, optional): [K] int64 subset of octant codes. Defaults to None (all 8, in order).

        Returns:
            Tensor: ``[K * (stop - start)]`` int64 indices; entry ``i * (stop - start) + j`` is
            ``octants[i] * D8 + start + j``.
        """
        d = torch.arange(start, stop, device=self.device, dtype=torch.int64)
        o = torch.arange(8, device=self.device, dtype=torch.int64) if octants is None else octants.to(self.device)
        return (o[:, None] * self.directions_per_octant + d[None, :]).reshape(-1)


def make_sweep_context(
    grid: Grid,
    quadrature: Quadrature,
    L: int,
    device: Union[torch.device, str, None] = None,
    dtype: torch.dtype = torch.float32,
    backend: Backend = "auto",
) -> SweepContext:
    """Build the sweep context of a grid and a quadrature, reordering the quadrature by octant.

    The quadrature must hold the same number of directions in every octant, e.g. a product set with even ``num_polar``
    and ``num_azimuthal`` divisible by 4.

    Args:
        grid (Grid): transport grid.
        quadrature (Quadrature): direction set, in any order.
        L (int): spherical-harmonic degree of the returned moments, at least 0.
        device (torch.device | str, optional): device of the tables. Defaults to None (the device of the quadrature).
        dtype (torch.dtype, optional): floating dtype of the tables. Defaults to torch.float32.
        backend (Backend, optional): default plane-step backend, "auto", "torch" or "triton" (see
            :func:`~citrus.scatter_solver.transport_sweep.sweep`); "triton" requires CUDA and a usable Triton.
            Defaults to "auto".

    Returns:
        SweepContext: the context on ``device``.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
    D8 = quadrature.directions_per_octant
    if D8 is None or D8 < 1:
        raise ValueError("the quadrature must hold the same (non-zero) number of directions in every octant")
    if L < 0:
        raise ValueError("L must be >= 0")
    if device is None:
        device = quadrature.device
    device = torch.device(device)
    if backend == "triton":
        if device.type != "cuda":
            raise ValueError("backend='triton' requires a CUDA device")
        ok, why = sweep_kernels.triton_available()
        if not ok:
            raise RuntimeError(f"backend='triton' requested but Triton is unusable: {why}")
    permutation, _ = quadrature.by_octant()
    reordered = quadrature.reordered(permutation).to(device=device, dtype=dtype)
    permutation = permutation.to(device)
    # sanity: reordered direction o * D8 + d must belong to octant o
    expected = torch.arange(8, device=device).repeat_interleave(D8)
    if not torch.equal(reordered.octant.to(device), expected):
        raise RuntimeError("octant reordering failed")
    basis = spherical_harmonic_basis(L, reordered)
    deltas = tuple(float(v) for v in grid.deltas)
    deltas_tensor = torch.tensor(deltas, device=device, dtype=dtype)
    omega = reordered.omega.reshape(8, D8, 3)
    c = (omega.abs() / deltas_tensor).contiguous()
    s = torch.where(omega < 0, -torch.ones_like(omega), torch.ones_like(omega)).contiguous()
    if not torch.equal(s.to(torch.int64), OCTANT_SIGNS.to(device)[:, None, :].expand(8, D8, 3)):
        raise RuntimeError("direction signs disagree with the sweep plan's octant signs")

    plan = build_sweep_plan(tuple(grid.shape), device=device)
    N = plan.num_cells
    upstream = torch.where(plan.upstream < 0, torch.full_like(plan.upstream, N), plan.upstream)  # [E_tot, 3]
    offsets = plan.plane_offsets.tolist()
    planes = []
    for p in range(plan.num_planes):
        a, b = offsets[p], offsets[p + 1]
        octant = plan.octant[a:b].contiguous()
        planes.append(
            PlaneIndex(
                octant=octant,
                cell=plan.cell[a:b].contiguous(),
                octant_column=octant[:, None].contiguous(),
                upstream_cell=upstream[a:b].contiguous(),
            )
        )
    flat_index = (sweep_kernels.build_flat_index(plan.octant, plan.cell, upstream, offsets)
                  if device.type == "cuda" else None)
    return SweepContext(
        shape=tuple(grid.shape), deltas=deltas, L=L, plan=plan, quadrature=reordered, permutation=permutation,
        basis=basis, directions_per_octant=D8, c=c, s=s, planes=tuple(planes), backend=backend, flat_index=flat_index,
    )


def _use_triton(sweep_context: SweepContext, backend: Optional[str], device: torch.device,
                dtype: torch.dtype) -> bool:
    """Resolve the plane-step backend of a sweep and tell whether the fused Triton kernel runs.

    Args:
        sweep_context (SweepContext): sweep context.
        backend (str, optional): "auto", "torch" or "triton", or None for ``sweep_context.backend``.
        device (torch.device): device of the sweep tensors.
        dtype (torch.dtype): dtype of the sweep tensors.

    Returns:
        bool: True for the fused kernel, False for the PyTorch plane step.
    """
    chosen = sweep_context.backend if backend is None else backend
    if chosen not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {chosen!r}")
    if chosen == "torch":
        return False
    fits = device.type == "cuda" and dtype == torch.float32 and sweep_context.flat_index is not None
    if chosen == "auto":
        return fits and sweep_kernels.kernels_usable(device)[0]
    if not fits:
        raise ValueError("backend='triton' needs a float32 sweep on a CUDA context (float64 / CPU: use 'torch')")
    ok, why = sweep_kernels.kernels_usable(device)
    if not ok:
        raise RuntimeError(f"backend='triton' requested but the fused sweep kernel is unusable: {why}")
    return True


# =====================================================================================================
# plane step
# =====================================================================================================

def _plane_step(
    plane: PlaneIndex,
    store: Tensor,
    source: Tensor,
    mu_cells: Tensor,
    c: Tensor,
    s: Tensor,
) -> None:
    """Solve all entries of one wavefront plane for all chunk directions and groups, writing into ``store`` in place.

    The upstream moments and the source are gathered, the cell equations are solved in closed form
    (:func:`~citrus.scatter_solver.linear_discontinuous_cell.cell_solve`) and the result is scattered into ``store``.

    Args:
        plane (PlaneIndex): indices of the plane.
        store (Tensor): [8, Dc, N + 1, 4G] angular-fluence moments of the chunk; the last cell is the zero vacuum cell.
        source (Tensor): [8, Dc, N, 4G] source moments of the chunk.
        mu_cells (Tensor): [N, G] total attenuation coefficient mu_t per cell in cm^-1.
        c (Tensor): [8, Dc, 3] ``|Omega_a| / Delta_a`` of the chunk directions in cm^-1.
        s (Tensor): [8, Dc, 3] ``sign(Omega_a)`` of the chunk directions.
    """
    E = plane.octant.shape[0]
    Dc = store.shape[1]
    G = mu_cells.shape[1]
    neighbors = store[plane.octant_column, :, plane.upstream_cell, :].view(E, 3, Dc, G, 4)
    neighbor_psi0 = neighbors[..., 0].permute(2, 3, 1, 0)  # [Dc, G, 3(a), E]
    neighbor_psia = neighbors[..., 1:].permute(2, 3, 1, 4, 0)  # [Dc, G, 3(a), 3(b), E]: moment b of neighbor along a
    source_entries = source[plane.octant, :, plane.cell, :].view(E, Dc, G, 4).permute(1, 2, 3, 0)  # [Dc, G, 4, E]
    mu_entries = mu_cells.index_select(0, plane.cell).t()  # [G, E]
    c_entries = c.index_select(0, plane.octant).permute(1, 2, 0).unsqueeze(1)  # [Dc, 1, 3, E]
    s_entries = s.index_select(0, plane.octant).permute(1, 2, 0).unsqueeze(1)
    r0, ra = cell_right_hand_sides(c_entries, s_entries, source_entries, neighbor_psi0, neighbor_psia)
    psi0, psia = cell_solve(c_entries, s_entries, mu_entries, r0, ra)  # [Dc, G, E], [Dc, G, 3, E]
    out = torch.cat([psi0.unsqueeze(-2), psia], dim=-2)  # [Dc, G, 4, E]
    store[plane.octant, :, plane.cell, :] = out.permute(3, 0, 1, 2).reshape(E, Dc, 4 * G)


def _sweep_chunk(
    sweep_context: SweepContext,
    source: Tensor,
    mu_cells: Tensor,
    c: Tensor,
    s: Tensor,
    store: Optional[Tensor] = None,
    use_triton: bool = False,
) -> Tensor:
    """Sweep one direction chunk through all wavefront planes.

    Args:
        sweep_context (SweepContext): sweep context.
        source (Tensor): [8, Dc, N, 4G] source moments of the chunk.
        mu_cells (Tensor): [N, G] total attenuation coefficient mu_t per cell in cm^-1.
        c (Tensor): [8, Dc, 3] ``|Omega_a| / Delta_a`` of the chunk directions in cm^-1.
        s (Tensor): [8, Dc, 3] ``sign(Omega_a)`` of the chunk directions.
        store (Tensor, optional): [8, Dc, N + 1, 4G] buffer to reuse, zeroed here. Defaults to None (allocated).
        use_triton (bool, optional): one fused Triton kernel per plane instead of the PyTorch plane step (float32 CUDA,
            contiguous inputs). Defaults to False.

    Returns:
        Tensor: [8, Dc, N + 1, 4G] angular-fluence moments of the chunk.
    """
    N = sweep_context.num_cells
    Dc = c.shape[1]
    G = mu_cells.shape[1]
    if store is None:
        store = torch.zeros(8, Dc, N + 1, 4 * G, device=mu_cells.device, dtype=mu_cells.dtype)
    else:
        store.zero_()
    if use_triton:
        if sweep_context.flat_index is None:
            raise ValueError("the fused plane kernel needs a CUDA sweep context")
        sweep_kernels.sweep_planes(sweep_context.flat_index, store, source, mu_cells, c)
        return store
    for plane in sweep_context.planes:
        _plane_step(plane, store, source, mu_cells, c, s)
    return store


def _resolve_chunk(sweep_context: SweepContext, direction_chunk: Optional[int]) -> int:
    """Number of directions per octant swept together.

    Args:
        sweep_context (SweepContext): sweep context.
        direction_chunk (int, optional): requested chunk, at least 1, or None for all D8.

    Returns:
        int: the chunk, at most D8.
    """
    D8 = sweep_context.directions_per_octant
    if direction_chunk is None:
        return D8
    chunk = int(direction_chunk)
    if chunk < 1:
        raise ValueError("direction_chunk must be >= 1")
    return min(chunk, D8)


def source_function_accepts_out(source_function: SourceFunction) -> bool:
    """Tell whether a source function takes an ``out`` keyword, so the sweep can lend it memory for its result.

    Args:
        source_function (SourceFunction): the source function.

    Returns:
        bool: True if it accepts ``out`` as a positional-or-keyword or keyword-only parameter.
    """
    try:
        params = inspect.signature(source_function).parameters
    except (TypeError, ValueError):
        return False
    p = params.get("out")
    return p is not None and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def _fill_source(sweep_context: SweepContext, source_function: SourceFunction, d0: int, d1: int, G: int,
                 out: Tensor, scratch: Optional[Tensor] = None) -> None:
    """Evaluate the source of the chunk ``d0 <= d < d1`` of every octant and copy it cell-major into ``out``.

    The source function is called once with all ``8 dc`` directions, octant-major; if it accepts ``out``, it writes into
    a view of ``scratch``, so that no temporary is allocated.

    Args:
        sweep_context (SweepContext): sweep context.
        source_function (SourceFunction): source function of :func:`~citrus.scatter_solver.transport_sweep.sweep`.
        d0 (int): first per-octant direction index of the chunk.
        d1 (int): end (exclusive) of the chunk; ``dc = d1 - d0``.
        G (int): number of energy groups.
        out (Tensor): [8, dc, N, 4G] cell-major source moments, written in place.
        scratch (Tensor, optional): contiguous buffer of at least ``8 dc 4G N`` elements of the dtype and device of
            ``out``, contents destroyed. Defaults to None.
    """
    chunk = d1 - d0
    N = sweep_context.num_cells
    Q = 4 * G
    expected = (8 * chunk, G, 4) + tuple(sweep_context.shape)
    direction_index = sweep_context.direction_indices(d0, d1)
    if source_function_accepts_out(source_function):
        n = 8 * chunk * Q * N
        if scratch is not None and scratch.is_contiguous() and scratch.numel() >= n \
                and scratch.dtype == out.dtype and scratch.device == out.device:
            buffer = scratch.view(-1)[:n].view(expected)
        else:
            buffer = torch.empty(expected, device=out.device, dtype=out.dtype)
        chunk_source = source_function(direction_index, out=buffer)
    else:
        chunk_source = source_function(direction_index)
    if tuple(chunk_source.shape) != expected:
        raise ValueError(f"source_function returned shape {tuple(chunk_source.shape)}, expected {expected}")
    out.view(8 * chunk, N, Q).copy_(chunk_source.reshape(8 * chunk, Q, N).transpose(1, 2))
    del chunk_source


# =====================================================================================================
# public sweep
# =====================================================================================================

def sweep(
    sweep_context: SweepContext,
    mu_t: Tensor,
    source_function: SourceFunction,
    direction_chunk: Optional[int] = None,
    return_psi: bool = False,
    backend: Optional[Backend] = None,
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """Linear-discontinuous discrete-ordinates sweep of all directions and groups with vacuum boundaries.

    Args:
        sweep_context (SweepContext): context of the grid and the quadrature.
        mu_t (Tensor): [G, rows, cols, slices] total attenuation coefficient mu_t in cm^-1 on the context device; its
            dtype sets the computation dtype.
        source_function (SourceFunction): maps [Dsel] int64 reordered direction indices to [Dsel, G, 4, rows, cols,
            slices] source moments per sr, called once per chunk; it may write into an ``out`` keyword buffer.
        direction_chunk (int, optional): directions per octant swept together (Dc). Defaults to None (all D8).
        return_psi (bool, optional): also return the angular fluence (small grids and tests). Defaults to False.
        backend (Backend, optional): plane-step backend "auto", "torch" or "triton"; "auto" uses the fused Triton kernel
            for float32 CUDA sweeps when it is usable. Defaults to None (``sweep_context.backend``).

    Returns:
        Tensor: [G, (L+1)^2, 4, rows, cols, slices] spherical-harmonic moments phi of the angular fluence; with
        ``return_psi``, the tuple (phi, psi) with psi [M, G, 4, rows, cols, slices] in reordered direction order.
    """
    if mu_t.dim() != 4 or tuple(mu_t.shape[1:]) != tuple(sweep_context.shape):
        raise ValueError(f"mu_t must be [G, {sweep_context.shape}], got {tuple(mu_t.shape)}")
    G = int(mu_t.shape[0])
    Q = 4 * G
    N = sweep_context.num_cells
    D8 = sweep_context.directions_per_octant
    num_moments = sweep_context.num_moments
    dtype, device = mu_t.dtype, mu_t.device
    c_all = sweep_context.c.to(device=device, dtype=dtype)
    s_all = sweep_context.s.to(device=device, dtype=dtype)
    Y_weighted_all = sweep_context.basis.Y_weighted.to(device=device, dtype=dtype)
    Dc = _resolve_chunk(sweep_context, direction_chunk)
    use_triton = _use_triton(sweep_context, backend, device, dtype)
    lend_storage = source_function_accepts_out(source_function)

    mu_cells = mu_t.reshape(G, N).t().contiguous()  # [N, G]
    moments = torch.zeros(num_moments, N * Q, device=device, dtype=dtype)  # [n, (cell, q)]
    psi_out = torch.zeros(sweep_context.num_directions, G, 4, N, device=device, dtype=dtype) if return_psi else None
    for d0 in range(0, D8, Dc):
        d1 = min(d0 + Dc, D8)
        chunk = d1 - d0
        direction_index = sweep_context.direction_indices(d0, d1)
        c = c_all[:, d0:d1].contiguous()
        s = s_all[:, d0:d1].contiguous()
        # Memory: an `out`-capable source_function writes into the (not yet used) storage of the chunk; otherwise the
        # storage is allocated after the fill, so its transient result never coexists with it.
        source = torch.empty(8, chunk, N, Q, device=device, dtype=dtype)
        store = torch.empty(8, chunk, N + 1, Q, device=device, dtype=dtype) if lend_storage else None
        _fill_source(sweep_context, source_function, d0, d1, G, source, scratch=store)
        store = _sweep_chunk(sweep_context, source, mu_cells, c, s, store=store, use_triton=use_triton)
        del source
        psi_chunk = store.view(8 * chunk, N + 1, Q)[:, :N]  # [8 dc, N, Q] (view)
        moments.addmm_(Y_weighted_all.index_select(0, direction_index).t(), psi_chunk.reshape(8 * chunk, N * Q))
        if psi_out is not None:
            psi_out[direction_index] = psi_chunk.permute(0, 2, 1).reshape(8 * chunk, G, 4, N)
        del store, psi_chunk
    phi = moments.view(num_moments, N, G, 4).permute(2, 0, 3, 1).reshape(G, num_moments, 4, *sweep_context.shape)
    del moments
    if psi_out is not None:
        return phi, psi_out.view(sweep_context.num_directions, G, 4, *sweep_context.shape)
    return phi
