"""Batched linear-discontinuous (LD) discrete-ordinates sweep on the supervoxel grid.

One call to `sweep` transports a given LD source for every direction of the (octant-reordered)
quadrature and every energy group through a cuboid grid with vacuum boundaries, and returns the
spherical-harmonic moments of the LD angular fluence:

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
cell-major layout, src [8, Dc, num_cells, Q], so that every gather reads contiguous runs of Q values.

The wavefront plan (`wavefront.build_sweep_plan`) lists, for each diagonal plane p, the
(octant, cell) entries of all 8 octants. Plane p only depends on planes < p, so the planes are
processed in order, each as ONE batched step over [Dc directions, G groups, E entries]:

    1. gather the upstream neighbors' LD moments   store[o_e, :, up_e[a], :]   -> [E, 3, Dc, G, 4]
    2. gather the entries' source moments and mu_t  src[o_e, :, n_e, :]         -> [E, Dc, G, 4], [E, G]
    3. gather c = |Omega_a|/Delta_a and s = sign(Omega_a) for (o_e, d)         -> [Dc, 1, 3, E]
    4. R_0, R_a = ld_cell_rhs(...);  psi_0, psi_a = ld_cell_solve(...)       (closed form, ld_cell.py)
    5. scatter (psi_0, psi_a) into store[o_e, :, n_e, :]

The number of sequential steps per chunk is num_planes = rows + cols + slices - 2. After a chunk the
moments are accumulated with one GEMM straight from the storage (no copy):

    acc[n, (cell, q)] += sum_(o, d) (w Y)[o D8 + d, n] store[o, d, cell, q]

so the angular fluence of all directions is never held at once (unless return_psi=True).

Signs: s_a = +1 for a zero direction component, consistent with `angular_quadrature.octant_of` and the sweep
plan's octant signs. Axis order a = 0 -> x (cols), 1 -> y (rows), 2 -> z (slices).

No negative-flux fix-up is applied: the LD scheme is particle conserving cell by cell, and the rare negative cell
means of the collided fluence are kept (they integrate correctly in the moments).

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

from .angular_quadrature import Quadrature, SHBasis, sh_basis
from .geometry import Grid
from .kernels import sweep_kernels as _tri
from .ld_cell import ld_cell_rhs, ld_cell_solve
from .wavefront import OCTANT_SIGNS, SweepPlan, build_sweep_plan

__all__ = [
    "SourceFn",
    "Backend",
    "PlaneIndex",
    "SweepContext",
    "make_sweep_context",
    "sweep",
    "source_fn_accepts_out",
]

SourceFn = Callable[[Tensor], Tensor]
"""source_fn(dir_index [Dsel] int64, reordered direction indices) -> LD source moments [Dsel, G, 4, R, C, S].
Optionally source_fn(dir_index, out=buf) writes into the contiguous buffer buf [Dsel, G, 4, R, C, S] and returns it
(see `sweep`)."""

Backend = Literal["auto", "torch", "triton"]
_BACKENDS = ("auto", "torch", "triton")


@dataclass(frozen=True, eq=False)
class PlaneIndex:
    """Gather/scatter indices of one wavefront plane (int64, on the sweep device). E = entries on the plane.

    octant  : [E]     octant code of each entry
    cell    : [E]     flat cell index (row-major over [rows, cols, slices])
    octant3 : [E, 1]  octant code, shaped to broadcast against up_cell
    up_cell : [E, 3]  flat index of the upstream neighbor along x, y, z; vacuum -> num_cells (the zero cell)
    """

    octant: Tensor
    cell: Tensor
    octant3: Tensor
    up_cell: Tensor

    @property
    def num_entries(self) -> int:
        return int(self.octant.shape[0])


@dataclass(frozen=True, eq=False)
class SweepContext:
    """Everything the sweep needs that depends only on the grid and the quadrature.

    shape      : (rows, cols, slices)
    deltas     : (dx, dy, dz) cm
    L          : SH degree of the returned moments; num_moments = (L+1)^2
    plan       : SweepPlan on the sweep device
    quad       : the quadrature REORDERED by octant: direction m = octant * D8 + d
    perm       : [M] int64, quad = original.reordered(perm) (original index of each reordered direction)
    basis      : SHBasis of degree L on the reordered quadrature (Y, Yw: [M, (L+1)^2])
    directions_per_octant : D8
    c          : [8, D8, 3] |Omega_a| / Delta_a of direction (octant, d)
    s          : [8, D8, 3] sign(Omega_a), +1 for a zero component
    planes     : tuple of PlaneIndex, one per wavefront plane, in sweep order
    backend    : default plane-step backend of `sweep`: "auto", "torch" or "triton"
    flat       : plane-sorted int32 entry tables for the fused Triton kernel (CUDA contexts only, else None)
    """

    shape: tuple[int, int, int]
    deltas: tuple[float, float, float]
    L: int
    plan: SweepPlan
    quad: Quadrature
    perm: Tensor
    basis: SHBasis
    directions_per_octant: int
    c: Tensor
    s: Tensor
    planes: tuple[PlaneIndex, ...]
    backend: str = "auto"
    flat: Optional[_tri.FlatPlaneIndex] = None

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
    quad: Quadrature,
    L: int,
    device: Union[torch.device, str, None] = None,
    dtype: torch.dtype = torch.float32,
    backend: Backend = "auto",
) -> SweepContext:
    """Build the sweep context for `grid` and `quad` (any order; it is reordered by octant here).

    Requires quad.directions_per_octant is not None (equal direction counts per octant), e.g. a
    product set with even num_polar and num_azimuthal % 4 == 0.
    All tensors are placed on `device` (default: the quadrature's device) with `dtype`.
    backend: default plane-step backend of the sweeps of this context ("auto", "torch", "triton"; see the
    module docstring). "triton" raises here if the device is not CUDA or Triton is unusable.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
    D8 = quad.directions_per_octant
    if D8 is None or D8 < 1:
        raise ValueError("the quadrature must hold the same (non-zero) number of directions in every octant")
    if L < 0:
        raise ValueError("L must be >= 0")
    if device is None:
        device = quad.device
    device = torch.device(device)
    if backend == "triton":
        if device.type != "cuda":
            raise ValueError("backend='triton' requires a CUDA device")
        ok, why = _tri.triton_available()
        if not ok:
            raise RuntimeError(f"backend='triton' requested but Triton is unusable: {why}")
    perm, _ = quad.by_octant()
    q = quad.reordered(perm).to(device=device, dtype=dtype)
    perm = perm.to(device)
    # sanity: reordered direction o * D8 + d must belong to octant o
    expected = torch.arange(8, device=device).repeat_interleave(D8)
    if not torch.equal(q.octant.to(device), expected):
        raise RuntimeError("octant reordering failed")
    basis = sh_basis(L, q)
    deltas = tuple(float(v) for v in grid.deltas)
    delta_t = torch.tensor(deltas, device=device, dtype=dtype)
    omega = q.omega.reshape(8, D8, 3)
    c = (omega.abs() / delta_t).contiguous()
    s = torch.where(omega < 0, -torch.ones_like(omega), torch.ones_like(omega)).contiguous()
    if not torch.equal(s.to(torch.int64), OCTANT_SIGNS.to(device)[:, None, :].expand(8, D8, 3)):
        raise RuntimeError("direction signs disagree with the sweep plan's octant signs")

    plan = build_sweep_plan(tuple(grid.shape), device=device)
    N = plan.num_cells
    up = torch.where(plan.upstream < 0, torch.full_like(plan.upstream, N), plan.upstream)  # [E_tot, 3]
    offsets = plan.plane_offsets.tolist()
    planes = []
    for p in range(plan.num_planes):
        a, b = offsets[p], offsets[p + 1]
        octant = plan.octant[a:b].contiguous()
        planes.append(
            PlaneIndex(
                octant=octant,
                cell=plan.cell[a:b].contiguous(),
                octant3=octant[:, None].contiguous(),
                up_cell=up[a:b].contiguous(),
            )
        )
    flat = _tri.build_flat_index(plan.octant, plan.cell, up, offsets) if device.type == "cuda" else None
    return SweepContext(
        shape=tuple(grid.shape), deltas=deltas, L=L, plan=plan, quad=q, perm=perm, basis=basis,
        directions_per_octant=D8, c=c, s=s, planes=tuple(planes), backend=backend, flat=flat,
    )


def _use_triton(ctx: SweepContext, backend: Optional[str], device: torch.device, dtype: torch.dtype) -> bool:
    """Resolve the plane-step backend (None -> ctx.backend) for a sweep of `dtype` tensors on `device`."""
    b = ctx.backend if backend is None else backend
    if b not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {b!r}")
    if b == "torch":
        return False
    fits = device.type == "cuda" and dtype == torch.float32 and ctx.flat is not None
    if b == "auto":
        return fits and _tri.kernels_usable(device)[0]
    if not fits:
        raise ValueError("backend='triton' needs a float32 sweep on a CUDA context (float64 / CPU: use 'torch')")
    ok, why = _tri.kernels_usable(device)
    if not ok:
        raise RuntimeError(f"backend='triton' requested but the fused sweep kernel is unusable: {why}")
    return True


# =====================================================================================================
# plane step
# =====================================================================================================

def _plane_step(
    pl: PlaneIndex,
    store: Tensor,
    src: Tensor,
    mu_cells: Tensor,
    c: Tensor,
    s: Tensor,
) -> None:
    """Solve all entries of one plane for all chunk directions and groups; writes into `store` in place.

    store    : [8, Dc, N + 1, 4G]   angular-fluence LD moments, last cell = zero vacuum cell
    src      : [8, Dc, N, 4G]       source LD moments of the chunk
    mu_cells : [N, G]               total attenuation per cell
    c, s     : [8, Dc, 3]           direction coefficients of the chunk
    """
    E = pl.octant.shape[0]
    Dc = store.shape[1]
    G = mu_cells.shape[1]
    nbr = store[pl.octant3, :, pl.up_cell, :].view(E, 3, Dc, G, 4)
    nbr_psi0 = nbr[..., 0].permute(2, 3, 1, 0)  # [Dc, G, 3(a), E]
    nbr_psia = nbr[..., 1:].permute(2, 3, 1, 4, 0)  # [Dc, G, 3(a), 3(b), E]: moment b of neighbor along a
    src_e = src[pl.octant, :, pl.cell, :].view(E, Dc, G, 4).permute(1, 2, 3, 0)  # [Dc, G, 4, E]
    mu_e = mu_cells.index_select(0, pl.cell).t()  # [G, E]
    c_e = c.index_select(0, pl.octant).permute(1, 2, 0).unsqueeze(1)  # [Dc, 1, 3, E]
    s_e = s.index_select(0, pl.octant).permute(1, 2, 0).unsqueeze(1)
    r0, ra = ld_cell_rhs(c_e, s_e, src_e, nbr_psi0, nbr_psia)
    psi0, psia = ld_cell_solve(c_e, s_e, mu_e, r0, ra)  # [Dc, G, E], [Dc, G, 3, E]
    out = torch.cat([psi0.unsqueeze(-2), psia], dim=-2)  # [Dc, G, 4, E]
    store[pl.octant, :, pl.cell, :] = out.permute(3, 0, 1, 2).reshape(E, Dc, 4 * G)


def _sweep_chunk(
    ctx: SweepContext,
    src: Tensor,
    mu_cells: Tensor,
    c: Tensor,
    s: Tensor,
    store: Optional[Tensor] = None,
    use_triton: bool = False,
) -> Tensor:
    """Sweep one direction chunk through all planes. src [8, Dc, N, 4G]. Returns store [8, Dc, N + 1, 4G].

    use_triton: one fused `sweep_kernels` kernel per plane instead of `_plane_step` (float32 CUDA tensors,
    ctx.flat set; src, mu_cells, c and a given store must be contiguous).
    """
    N = ctx.num_cells
    Dc = c.shape[1]
    G = mu_cells.shape[1]
    if store is None:
        store = torch.zeros(8, Dc, N + 1, 4 * G, device=mu_cells.device, dtype=mu_cells.dtype)
    else:
        store.zero_()
    if use_triton:
        if ctx.flat is None:
            raise ValueError("the fused plane kernel needs a CUDA sweep context")
        _tri.sweep_planes(ctx.flat, store, src, mu_cells, c)
        return store
    for pl in ctx.planes:
        _plane_step(pl, store, src, mu_cells, c, s)
    return store


def _resolve_chunk(ctx: SweepContext, direction_chunk: Optional[int]) -> int:
    D8 = ctx.directions_per_octant
    if direction_chunk is None:
        return D8
    dc = int(direction_chunk)
    if dc < 1:
        raise ValueError("direction_chunk must be >= 1")
    return min(dc, D8)


def source_fn_accepts_out(source_fn: SourceFn) -> bool:
    """True if source_fn takes an `out` keyword (the sweep then lends it scratch memory for its result)."""
    try:
        params = inspect.signature(source_fn).parameters
    except (TypeError, ValueError):
        return False
    p = params.get("out")
    return p is not None and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def _fill_source(ctx: SweepContext, source_fn: SourceFn, d0: int, d1: int, G: int, out: Tensor,
                 scratch: Optional[Tensor] = None) -> None:
    """Evaluate the source of the chunk d in [d0, d1) of every octant and copy it cell-major into out [8, dc, N, 4G].

    ONE call source_fn(ctx.direction_indices(d0, d1)) with all 8 dc directions (octant-major, the layout of `out`),
    so a GEMM-type source (scatter_source_fn) reads its moments once per chunk. If source_fn accepts `out=`, it is
    given a [8 dc, G, 4, R, C, S] view of `scratch` (contiguous, >= 8 dc 4G N elements, contents destroyed; the sweep
    passes the not-yet-used storage tensor of the chunk), so no temporary is allocated; otherwise its result is one
    transient source-sized tensor.
    """
    dc = d1 - d0
    N = ctx.num_cells
    Q = 4 * G
    expected = (8 * dc, G, 4) + tuple(ctx.shape)
    idx = ctx.direction_indices(d0, d1)
    if source_fn_accepts_out(source_fn):
        n = 8 * dc * Q * N
        if scratch is not None and scratch.is_contiguous() and scratch.numel() >= n \
                and scratch.dtype == out.dtype and scratch.device == out.device:
            buf = scratch.view(-1)[:n].view(expected)
        else:
            buf = torch.empty(expected, device=out.device, dtype=out.dtype)
        so = source_fn(idx, out=buf)
    else:
        so = source_fn(idx)
    if tuple(so.shape) != expected:
        raise ValueError(f"source_fn returned shape {tuple(so.shape)}, expected {expected}")
    out.view(8 * dc, N, Q).copy_(so.reshape(8 * dc, Q, N).transpose(1, 2))
    del so


# =====================================================================================================
# public sweep
# =====================================================================================================

def sweep(
    ctx: SweepContext,
    mu_t: Tensor,
    source_fn: SourceFn,
    direction_chunk: Optional[int] = None,
    return_psi: bool = False,
    backend: Optional[Backend] = None,
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """LD discrete-ordinates sweep of all directions and groups with vacuum boundaries.

    mu_t            : [G, rows, cols, slices] total attenuation (1/cm), on ctx.device; its dtype sets the
                      computation dtype (ctx tables are cast to it).
    source_fn       : dir_index [Dsel] int64 (reordered direction indices, see SweepContext.direction_indices)
                      -> LD source moments [Dsel, G, 4, rows, cols, slices] (per steradian, moments
                      [0, x, y, z]). Called ONCE per direction chunk with the Dsel = 8 Dc directions of
                      all octants, octant-major: indices o * D8 + d for o = 0..7, d0 <= d < d0 + Dc
                      (ctx.direction_indices(d0, d0 + Dc)). It may take a keyword `out` (a contiguous
                      [Dsel, G, 4, rows, cols, slices] tensor of the sweep dtype, contents undefined) to write
                      its result into and return; the sweep then lends it the chunk's storage buffer as
                      scratch and no source-sized temporary is allocated.
    direction_chunk : directions per octant swept together (Dc); default all D8. Smaller chunks reduce
                      memory (store and source scale with 8 * Dc) at the cost of more plane steps.
    return_psi      : also return the angular fluence psi [M, G, 4, rows, cols, slices] in reordered
                      direction order (small grids / tests only).
    backend         : plane-step backend "auto" / "torch" / "triton" (default None: ctx.backend), see the
                      module docstring. The backends agree to float32 rounding.

    Returns phi [G, (L+1)^2, 4, rows, cols, slices] (and psi if return_psi).
    """
    if mu_t.dim() != 4 or tuple(mu_t.shape[1:]) != tuple(ctx.shape):
        raise ValueError(f"mu_t must be [G, {ctx.shape}], got {tuple(mu_t.shape)}")
    G = int(mu_t.shape[0])
    Q = 4 * G
    N = ctx.num_cells
    D8 = ctx.directions_per_octant
    Nsh = ctx.num_moments
    dtype, device = mu_t.dtype, mu_t.device
    c_all = ctx.c.to(device=device, dtype=dtype)
    s_all = ctx.s.to(device=device, dtype=dtype)
    Yw_all = ctx.basis.Yw.to(device=device, dtype=dtype)
    Dc = _resolve_chunk(ctx, direction_chunk)
    use_triton = _use_triton(ctx, backend, device, dtype)
    lend_store = source_fn_accepts_out(source_fn)

    mu_cells = mu_t.reshape(G, N).t().contiguous()  # [N, G]
    acc = torch.zeros(Nsh, N * Q, device=device, dtype=dtype)  # [n, (cell, q)]
    psi_out = torch.zeros(ctx.num_directions, G, 4, N, device=device, dtype=dtype) if return_psi else None
    for d0 in range(0, D8, Dc):
        d1 = min(d0 + Dc, D8)
        dc = d1 - d0
        idx = ctx.direction_indices(d0, d1)
        c = c_all[:, d0:d1].contiguous()
        s = s_all[:, d0:d1].contiguous()
        # Memory: an `out`-capable source_fn writes into the (not yet used) storage of the chunk; otherwise the
        # storage is allocated after the fill, so its transient result never coexists with it.
        src = torch.empty(8, dc, N, Q, device=device, dtype=dtype)
        store = torch.empty(8, dc, N + 1, Q, device=device, dtype=dtype) if lend_store else None
        _fill_source(ctx, source_fn, d0, d1, G, src, scratch=store)
        store = _sweep_chunk(ctx, src, mu_cells, c, s, store=store, use_triton=use_triton)
        del src
        psi_c = store.view(8 * dc, N + 1, Q)[:, :N]  # [8 dc, N, Q] (view)
        acc.addmm_(Yw_all.index_select(0, idx).t(), psi_c.reshape(8 * dc, N * Q))
        if psi_out is not None:
            psi_out[idx] = psi_c.permute(0, 2, 1).reshape(8 * dc, G, 4, N)
        del store, psi_c
    phi = acc.view(Nsh, N, G, 4).permute(2, 0, 3, 1).reshape(G, Nsh, 4, *ctx.shape)
    del acc
    if psi_out is not None:
        return phi, psi_out.view(ctx.num_directions, G, 4, *ctx.shape)
    return phi
