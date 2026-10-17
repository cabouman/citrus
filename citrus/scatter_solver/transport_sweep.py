"""Batched linear-discontinuous discrete-ordinates sweep on the supervoxel grid.

One call to `sweep` transports a given linear-discontinuous source for every direction of the (octant-reordered)
quadrature and every energy group through a cuboid grid with vacuum boundaries, and returns the
spherical-harmonic moments of the linear-discontinuous angular fluence:

    phi[g, n, k, cell] = sum_m w_m Y_n(Omega_m) psi_m[g, k, cell],     k in {0, x, y, z}

Algorithm
---------
Directions are grouped by octant (`Quadrature.by_octant`) with D8 directions per octant; reordered
direction m = octant * D8 + d. For a chunk of Dc directions per octant (8 * Dc directions in total)
the angular fluence lives in one storage tensor (Q = 4 G, index q = 4 g + k)

    store [8, Dc, num_cells + 1, Q]

whose last cell (index num_cells) is never written and stays zero: it is the "vacuum cell". Upstream
indices of -1 (vacuum, `wavefront.SweepPlan`) are replaced by num_cells, so a plain gather
returns zero incoming fluence on boundary faces. The source of the chunk is copied once into the same
cell-major layout, source [8, Dc, num_cells, Q], so that every gather reads contiguous runs of Q values.

The wavefront plan (`wavefront.build_sweep_plan`) lists, for each diagonal plane p, the
(octant, cell) entries of all 8 octants. Plane p only depends on planes < p, so the planes are
processed in order, each as ONE batched step over [Dc directions, G groups, E entries]:

    1. gather the upstream neighbors' moments       store[o_e, :, up_e[a], :]   -> [E, 3, Dc, G, 4]
    2. gather the entries' source moments and mu_t  source[o_e, :, n_e, :]      -> [E, Dc, G, 4], [E, G]
    3. gather c = |Omega_a|/Delta_a and s = sign(Omega_a) for (o_e, d)         -> [Dc, 1, 3, E]
    4. R_0, R_a = cell_right_hand_sides(...);  psi_0, psi_a = cell_solve(...)
                                                    (closed form, linear_discontinuous_cell.py)
    5. scatter (psi_0, psi_a) into store[o_e, :, n_e, :]

The number of sequential steps per chunk is num_planes = rows + cols + slices - 2. After a chunk the
moments are accumulated with one GEMM straight from the storage (no copy):

    acc[n, (cell, q)] += sum_(o, d) (w Y)[o D8 + d, n] store[o, d, cell, q]

so the angular fluence of all directions is never held at once (unless return_psi=True).

Signs: s_a = +1 for a zero direction component, consistent with `angular_quadrature.octant_of` and the sweep
plan's octant signs. Axis order a = 0 -> x (cols), 1 -> y (rows), 2 -> z (slices).

No negative-flux fix-up is applied: the linear-discontinuous scheme is particle conserving cell by cell, and the rare
negative cell means of the collided fluence are kept (they integrate correctly in the moments).

Backends
--------
The plane loop (steps 1-5) runs either as the PyTorch plane step above (~15 kernels per plane) or as ONE
fused Triton kernel per plane (`kernels/sweep_kernels.py`; same storage layout, so the source fill and the moment
GEMM are shared). Selection: `make_sweep_context(..., backend=...)` stores the default on the context;
`sweep(..., backend=...)` overrides it per call:

    "auto"   (default) the fused kernel for float32 CUDA sweeps when `sweep_kernels.kernels_usable(device)`
             passes (Triton importable, probe kernel runs, per-device self-check against the PyTorch plane
             step; `CITRUS_DISABLE_TRITON=1` disables it), else the PyTorch plane step. float64 and CPU
             sweeps always use the PyTorch plane step.
    "torch"  always the PyTorch plane step.
    "triton" the fused kernel; raises if the sweep is not float32 on CUDA or the kernel is unusable.
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
    """Gather/scatter indices of one wavefront plane (int64, on the sweep device). E = entries on the plane.

    octant        : [E]     octant code of each entry
    cell          : [E]     flat cell index (row-major over [rows, cols, slices])
    octant_column : [E, 1]  octant code, shaped to broadcast against upstream_cell
    upstream_cell : [E, 3]  flat index of the upstream neighbor along x, y, z; vacuum -> num_cells (the zero cell)
    """

    octant: Tensor
    cell: Tensor
    octant_column: Tensor
    upstream_cell: Tensor

    @property
    def num_entries(self) -> int:
        return int(self.octant.shape[0])


@dataclass(frozen=True, eq=False)
class SweepContext:
    """Everything the sweep needs that depends only on the grid and the quadrature.

    shape      : (rows, cols, slices)
    deltas     : (dx, dy, dz) cm
    L          : spherical-harmonic degree of the returned moments; num_moments = (L+1)^2
    plan       : SweepPlan on the sweep device
    quadrature : the quadrature REORDERED by octant: direction m = octant * D8 + d (D8 directions per octant)
    permutation: [M] int64, quadrature = original.reordered(permutation) (original index of each reordered
                 direction)
    basis      : SphericalHarmonicBasis of degree L on the reordered quadrature (Y, Y_weighted: [M, (L+1)^2])
    directions_per_octant : D8
    c          : [8, D8, 3] |Omega_a| / Delta_a of direction (octant, d)
    s          : [8, D8, 3] sign(Omega_a), +1 for a zero component
    planes     : tuple of PlaneIndex, one per wavefront plane, in sweep order
    backend    : default plane-step backend of `sweep`: "auto", "torch" or "triton"
    flat_index : plane-sorted int32 entry tables for the fused Triton kernel (CUDA contexts only, else None)
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
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def num_planes(self) -> int:
        return len(self.planes)

    @property
    def num_directions(self) -> int:
        return 8 * self.directions_per_octant

    @property
    def num_moments(self) -> int:
        return (self.L + 1) ** 2

    @property
    def device(self) -> torch.device:
        return self.c.device

    @property
    def dtype(self) -> torch.dtype:
        return self.c.dtype

    def direction_indices(self, start: int, stop: int, octants: Optional[Tensor] = None) -> Tensor:
        """Reordered direction indices of the per-octant chunk d in [start, stop), octant-major.

        octants: optional int64 [K] subset of octant codes (default all 8, in order).
        Returns int64 [K * (stop - start)] with entry i * (stop - start) + j = octants[i] * D8 + start + j.
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
    """Build the sweep context for `grid` and `quadrature` (any order; it is reordered by octant here).

    Requires quadrature.directions_per_octant is not None (equal direction counts per octant), e.g. a
    product set with even num_polar and num_azimuthal % 4 == 0.
    All tensors are placed on `device` (default: the quadrature's device) with `dtype`.
    backend: default plane-step backend of the sweeps of this context ("auto", "torch", "triton"; see the
    module docstring). "triton" raises here if the device is not CUDA or Triton is unusable.
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
    """Resolve the plane-step backend (None -> sweep_context.backend) for a sweep of `dtype` tensors on `device`."""
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
    """Solve all entries of one plane for all chunk directions and groups; writes into `store` in place.

    store    : [8, Dc, N + 1, 4G]   angular-fluence linear-discontinuous moments, last cell = zero vacuum cell
    source   : [8, Dc, N, 4G]       source linear-discontinuous moments of the chunk
    mu_cells : [N, G]               total attenuation per cell
    c, s     : [8, Dc, 3]           direction coefficients of the chunk
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
    """Sweep one direction chunk through all planes. source [8, Dc, N, 4G]. Returns store [8, Dc, N + 1, 4G].

    use_triton: one fused `sweep_kernels` kernel per plane instead of `_plane_step` (float32 CUDA tensors,
    sweep_context.flat_index set; source, mu_cells, c and a given store must be contiguous).
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
    D8 = sweep_context.directions_per_octant
    if direction_chunk is None:
        return D8
    chunk = int(direction_chunk)
    if chunk < 1:
        raise ValueError("direction_chunk must be >= 1")
    return min(chunk, D8)


def source_function_accepts_out(source_function: SourceFunction) -> bool:
    """True if source_function takes an `out` keyword (the sweep then lends it scratch memory for its result)."""
    try:
        params = inspect.signature(source_function).parameters
    except (TypeError, ValueError):
        return False
    p = params.get("out")
    return p is not None and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def _fill_source(sweep_context: SweepContext, source_function: SourceFunction, d0: int, d1: int, G: int,
                 out: Tensor, scratch: Optional[Tensor] = None) -> None:
    """Evaluate the source of the chunk d in [d0, d1) of every octant and copy it cell-major into out [8, dc, N, 4G].

    ONE call source_function(sweep_context.direction_indices(d0, d1)) with all 8 dc directions (octant-major, the
    layout of `out`; dc = d1 - d0), so a GEMM-type source (scatter_source_function) reads its moments once per chunk.
    If source_function accepts `out=`, it is given a [8 dc, G, 4, R, C, S] view of `scratch` (contiguous, >= 8 dc 4G N
    elements, contents destroyed; the sweep passes the not-yet-used storage tensor of the chunk), so no temporary is
    allocated; otherwise its result is one transient source-sized tensor.
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

    mu_t            : [G, rows, cols, slices] total attenuation (1/cm), on sweep_context.device; its dtype sets the
                      computation dtype (sweep_context tables are cast to it).
    source_function : direction_index [Dsel] int64 (reordered direction indices, see
                      SweepContext.direction_indices) -> linear-discontinuous source moments [Dsel, G, 4, rows, cols,
                      slices] (per steradian, moments [0, x, y, z]). Called ONCE per direction chunk with the Dsel = 8
                      Dc directions of all octants, octant-major: indices o * D8 + d for o = 0..7, d0 <= d < d0 + Dc
                      (sweep_context.direction_indices(d0, d0 + Dc)). It may take a keyword `out` (a contiguous
                      [Dsel, G, 4, rows, cols, slices] tensor of the sweep dtype, contents undefined) to write
                      its result into and return; the sweep then lends it the chunk's storage buffer as
                      scratch and no source-sized temporary is allocated.
    direction_chunk : directions per octant swept together (Dc); default all D8. Smaller chunks reduce
                      memory (store and source scale with 8 * Dc) at the cost of more plane steps.
    return_psi      : also return the angular fluence psi [M, G, 4, rows, cols, slices] in reordered
                      direction order (small grids / tests only).
    backend         : plane-step backend "auto" / "torch" / "triton" (default None: sweep_context.backend), see the
                      module docstring. The backends agree to float32 rounding.

    Returns phi [G, (L+1)^2, 4, rows, cols, slices] (and psi if return_psi).
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
