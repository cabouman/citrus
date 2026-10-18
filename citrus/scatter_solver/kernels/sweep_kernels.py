"""Fused Triton kernel of the linear-discontinuous cell solve of one wavefront plane of the transport sweep.

``_linear_discontinuous_plane_kernel`` solves every (entry, direction, group) of a diagonal plane in one launch, on
the storage layout of the PyTorch path of :mod:`citrus.scatter_solver.transport_sweep`. ``FlatPlaneIndex`` holds the
plane-sorted entry tables it reads, ``sweep_planes`` runs the planes in order, and ``kernels_usable`` checks the
kernel against the PyTorch plane step once per device.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .triton_utilities import DISABLE_ENV_VAR, triton_available

try:  # import errors are reported by triton_available(), never raised at import time
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

__all__ = [
    "DISABLE_ENV_VAR",
    "SELF_CHECK_REL_TOL",
    "FlatPlaneIndex",
    "build_flat_index",
    "triton_available",
    "kernels_usable",
    "sweep_planes",
]

SELF_CHECK_REL_TOL = 1e-4
BLOCK = 256
NUM_WARPS = 4

_SELF_CHECK: dict[str, tuple[bool, str]] = {}


@dataclass(frozen=True, eq=False)
class FlatPlaneIndex:
    """Plane-sorted flat entry tables of a sweep plan for the fused kernel (int32, on the sweep device).

    Attributes:
        octant (Tensor): [E_tot] octant code of each entry (E_tot = 8 * num_cells), sorted by plane.
        cell (Tensor): [E_tot] flat cell index of each entry.
        upstream (Tensor): [E_tot * 3] upstream neighbor along x, y, z (row-major [E_tot, 3]), num_cells for vacuum.
        offsets (tuple[int, ...]): num_planes + 1 host ints; the entries of plane p are [offsets[p], offsets[p + 1]).
    """

    octant: Tensor
    cell: Tensor
    upstream: Tensor
    offsets: tuple[int, ...]


def build_flat_index(octant: Tensor, cell: Tensor, upstream_cell: Tensor, offsets: list[int]) -> FlatPlaneIndex:
    """Build a ``FlatPlaneIndex`` from the plane-sorted tables of a sweep plan.

    Args:
        octant (Tensor): [E_tot] octant code of each entry.
        cell (Tensor): [E_tot] flat cell index of each entry.
        upstream_cell (Tensor): [E_tot, 3] upstream neighbor along x, y, z, with vacuum already replaced by num_cells.
        offsets (list[int]): num_planes + 1 plane offsets.

    Returns:
        FlatPlaneIndex: the int32 contiguous tables on the device of the inputs.
    """
    return FlatPlaneIndex(
        octant=octant.to(torch.int32).contiguous(),
        cell=cell.to(torch.int32).contiguous(),
        upstream=upstream_cell.to(torch.int32).contiguous().view(-1),
        offsets=tuple(int(v) for v in offsets),
    )


# ================================================================================================
# availability / self-check
# ================================================================================================

def kernels_usable(device: torch.device) -> tuple[bool, str]:
    """Report whether Triton is available and the fused plane loop reproduces the PyTorch plane step on ``device``.

    The self-check runs once per device on a small problem; the verdict is cached and the function never raises.

    Args:
        device (torch.device): the device to check.

    Returns:
        (bool, str): whether the kernel is usable, and the reason.
    """
    ok, why = triton_available()
    if not ok:
        return ok, why
    device = torch.device(device)
    if device.type != "cuda":
        return False, "not a CUDA device"
    key = str(device)
    if key not in _SELF_CHECK:
        try:
            _SELF_CHECK[key] = _self_check(device)
        except Exception as e:  # noqa: BLE001
            _SELF_CHECK[key] = (False, f"self-check failed ({type(e).__name__}: {str(e)[:300]})")
    return _SELF_CHECK[key]


def _self_check(device: torch.device) -> tuple[bool, str]:
    """Compare the fused plane loop with the PyTorch ``transport_sweep._sweep_chunk`` on a small random problem.

    Args:
        device (torch.device): the CUDA device to check.

    Returns:
        (bool, str): whether the largest relative deviation is within ``SELF_CHECK_REL_TOL``, and a message.
    """
    from ..angular_quadrature import product_quadrature
    from ..geometry import Grid
    from ..transport_sweep import _sweep_chunk, make_sweep_context

    grid = Grid((3, 4, 5), 0.7, 1.3, 0.9)
    quadrature = product_quadrature(4, 8, dtype=torch.float64)
    sweep_context = make_sweep_context(grid, quadrature, 1, device=device, dtype=torch.float32, backend="torch")
    G, N = 3, sweep_context.num_cells
    generator = torch.Generator().manual_seed(11)
    mu_cells = (torch.rand(N, G, generator=generator) * 2.0 + 0.1).to(device)
    worst = 0.0
    for Dc in (sweep_context.directions_per_octant, 3):
        source = torch.randn(8, Dc, N, 4 * G, generator=generator).to(device)
        c = sweep_context.c[:, :Dc].contiguous()
        s = sweep_context.s[:, :Dc].contiguous()
        reference = _sweep_chunk(sweep_context, source, mu_cells, c, s, use_triton=False)
        out = _sweep_chunk(sweep_context, source, mu_cells, c, s, use_triton=True)
        scale = float(reference.abs().max())
        worst = max(worst, float((reference - out).abs().max()) / max(scale, 1e-30))
    if worst > SELF_CHECK_REL_TOL:
        return False, f"self-check: fused plane kernel deviates from the PyTorch plane step by {worst:.2e} (rel)"
    return True, f"fused plane kernel matches the PyTorch plane step (max rel dev {worst:.1e})"


# ================================================================================================
# kernel
# ================================================================================================

if triton is not None:

    @triton.jit
    def _load4(ptr, off, mask, BLOCK: tl.constexpr):
        """Load four consecutive values ``ptr[off + 0..3]`` per lane with one vector load.

        Args:
            ptr: float32 tensor to read.
            off: [BLOCK] int64 element offset of the first value of each lane.
            mask: [BLOCK] bool, the lane is active (inactive lanes read zeros).
            BLOCK (constexpr): lanes per program.

        Returns:
            (v0, v1, v2, v3): four [BLOCK] tensors.
        """
        k = tl.arange(0, 4)
        v = tl.load(ptr + off[:, None] + k[None, :], mask=mask[:, None], other=0.0)  # [BLOCK, 4]
        lo, hi = tl.split(tl.reshape(v, [BLOCK, 2, 2]))  # lo = (v0, v2), hi = (v1, v3)
        v0, v2 = tl.split(lo)
        v1, v3 = tl.split(hi)
        return v0, v1, v2, v3

    @triton.jit(do_not_specialize=["start", "E"])
    def _linear_discontinuous_plane_kernel(store_ptr, source_ptr, mu_ptr, c_ptr, octant_ptr, cell_ptr, upstream_ptr,
                                           start, E, N,
                                           G: tl.constexpr, DC: tl.constexpr, BLOCK: tl.constexpr):
        """Solve one wavefront plane in place, one lane per (entry, direction, group) with the group fastest.

        Follows ``linear_discontinuous_cell`` term by term (docs/source/design_notes.rst, "Transport sweep and source
        iteration").

        Args:
            store_ptr: [8, DC, N + 1, 4G] float32 angular fluence moments; cell N is the zero vacuum cell.
            source_ptr: [8, DC, N, 4G] float32 source moments.
            mu_ptr: [N, G] float32 total attenuation per cell in cm^-1.
            c_ptr: [8, DC, 3] float32 ``c_a = |Omega_a| / Delta_a`` in cm^-1.
            octant_ptr: [E_tot] int32 octant code of each entry.
            cell_ptr: [E_tot] int32 cell index of each entry.
            upstream_ptr: [E_tot * 3] int32 upstream cell along x, y, z of each entry.
            start: index of the first entry of the plane.
            E: number of entries of the plane.
            N: number of cells.
            G (constexpr): number of energy groups.
            DC (constexpr): directions per octant of the chunk.
            BLOCK (constexpr): lanes per program.
        """
        Q: tl.constexpr = 4 * G
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = i < E.to(tl.int64) * (DC * G)
        g = i % G
        ed = i // G
        d = ed % DC
        e = start + ed // DC
        e = tl.where(mask, e, start)  # keep masked lanes' table reads in bounds
        o = tl.load(octant_ptr + e).to(tl.int64)
        n = tl.load(cell_ptr + e).to(tl.int64)
        ux = tl.load(upstream_ptr + 3 * e).to(tl.int64)
        uy = tl.load(upstream_ptr + 3 * e + 1).to(tl.int64)
        uz = tl.load(upstream_ptr + 3 * e + 2).to(tl.int64)
        od = o * DC + d
        cx = tl.load(c_ptr + od * 3)
        cy = tl.load(c_ptr + od * 3 + 1)
        cz = tl.load(c_ptr + od * 3 + 2)
        # s_a = +1 / -1 from the octant code bits: code = (s_x<0)*4 + (s_y<0)*2 + (s_z<0)
        sx = 1.0 - 2.0 * ((o >> 2) & 1).to(tl.float32)
        sy = 1.0 - 2.0 * ((o >> 1) & 1).to(tl.float32)
        sz = 1.0 - 2.0 * (o & 1).to(tl.float32)
        mu = tl.load(mu_ptr + n * G + g, mask=mask, other=1.0)

        row = od * (N + 1)
        q0 = 4 * g
        # upstream neighbors' linear-discontinuous moments [0, x, y, z]; the vacuum cell N is all zeros
        x0, xx, xy, xz = _load4(store_ptr, (row + ux) * Q + q0, mask, BLOCK)
        y0, yx, yy, yz = _load4(store_ptr, (row + uy) * Q + q0, mask, BLOCK)
        z0, zx, zy, zz = _load4(store_ptr, (row + uz) * Q + q0, mask, BLOCK)
        s0, s1, s2, s3 = _load4(source_ptr, (od * N + n) * Q + q0, mask, BLOCK)

        # cell_right_hand_sides: face-averaged inflow psibar_a = psi_0[N_a] + s_a psi_a[N_a]
        pbx = x0 + sx * xx
        pby = y0 + sy * yy
        pbz = z0 + sz * zz
        r0 = s0 + (cx * pbx + cy * pby + cz * pbz)
        # R_a = S_a - 3 s_a c_a psibar_a + sum_{b != a} c_b psi_a[N_b]
        rx = s1 - 3.0 * sx * cx * pbx + (cy * yx + cz * zx)
        ry = s2 - 3.0 * sy * cy * pby + (cx * xy + cz * zy)
        rz = s3 - 3.0 * sz * cz * pbz + (cx * xz + cy * yz)
        # cell_solve
        csum = cx + cy + cz
        dx = 3.0 * cx + (csum - cx) + mu
        dy = 3.0 * cy + (csum - cy) + mu
        dz = 3.0 * cz + (csum - cz) + mu
        qx = cx / dx
        qy = cy / dy
        qz = cz / dz
        denom = csum + mu + 3.0 * (cx * qx + cy * qy + cz * qz)
        p0 = (r0 - (sx * qx * rx + sy * qy * ry + sz * qz * rz)) / denom
        px = (rx + 3.0 * sx * cx * p0) / dx
        py = (ry + 3.0 * sy * cy * p0) / dy
        pz = (rz + 3.0 * sz * cz * p0) / dz
        # (p0, px, py, pz) -> [BLOCK, 4] in moment order: join(join(p0, py), join(px, pz)) is [i, k, j] with
        # element 2k + j = (p0, px, py, pz)[2k + j]
        out = tl.reshape(tl.join(tl.join(p0, py), tl.join(px, pz)), [BLOCK, 4])
        k = tl.arange(0, 4)
        tl.store(store_ptr + ((row + n) * Q + q0)[:, None] + k[None, :], out, mask=mask[:, None])


# ================================================================================================
# host wrapper
# ================================================================================================

def sweep_planes(index: FlatPlaneIndex, store: Tensor, source: Tensor, mu_cells: Tensor, c: Tensor) -> None:
    """Run the fused plane kernel for every plane of ``index`` in order (one launch per plane), in place.

    Args:
        index (FlatPlaneIndex): the plane-sorted entry tables.
        store (Tensor): [8, Dc, N + 1, 4G] float32 CUDA contiguous angular fluence moments; cell N must be zero.
        source (Tensor): [8, Dc, N, 4G] float32 contiguous source linear-discontinuous moments of the chunk.
        mu_cells (Tensor): [N, G] float32 contiguous total attenuation per cell in cm^-1.
        c (Tensor): [8, Dc, 3] float32 contiguous ``|Omega_a| / Delta_a`` of the chunk directions in cm^-1.
    """
    _, Dc, num_cells_plus_vacuum, Q = store.shape
    N = num_cells_plus_vacuum - 1
    G = Q // 4
    if not (store.is_contiguous() and source.is_contiguous() and mu_cells.is_contiguous() and c.is_contiguous()):
        raise ValueError("sweep_planes needs contiguous store, source, mu_cells and c")
    lanes_per_entry = Dc * G
    offsets = index.offsets
    kernel = _linear_discontinuous_plane_kernel
    for p in range(len(offsets) - 1):
        a = offsets[p]
        E = offsets[p + 1] - a
        grid = (triton.cdiv(E * lanes_per_entry, BLOCK),)
        kernel[grid](store, source, mu_cells, c, index.octant, index.cell, index.upstream, a, E, N,
                     G=G, DC=Dc, BLOCK=BLOCK, num_warps=NUM_WARPS)
