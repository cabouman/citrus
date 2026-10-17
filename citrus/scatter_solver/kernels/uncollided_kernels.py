"""Fused Triton kernel for the first-collision source evaluation (`uncollided.first_collision_source_at_directions`).

The PyTorch body of `first_collision_source_at_directions` materializes the interpolated per-voxel transfer table
`T [G_in, G_out, k, V]` (and its two gathered end points `[M, G_in, G_out, k V]`) for every sub-chunk of
directions before contracting it with phi_uncollided. This module computes the result directly:

    S[k, g, v] = sum_{g' <= g} sum_m rho_m(v) lerp(table[m, g', g], eta_k(v)) phi_UC,g'(v)
    eta_k(v)   = Omega_UC(v) . omega_k,   Omega_UC(v) = (r_v - r0) / |r_v - r0|

Kernel `_first_collision_kernel`, per program (block of BLOCK flattened voxels, block of KB directions):
  - pair mode (PAIR = True): loads the compact pair representation `cross_sections.MaterialPairs` of an M-material
    volume once, densities [2, V] and material indices [2, V] int32 (at most two nonzero materials per point);
    dense mode (PAIR = False): reads the densities of all MT materials [MT, V], slot m = material m (any MT >= 1, a
    compile-time constant; used where points mix more than two materials, e.g. block means of a seven-material field
    on the quadrature grid, and for libraries with M <= 2). It evaluates the voxel centers from the 1-D axis
    coordinates of the grid (x[cols], y[rows], z[slices]) and Omega_UC with IEEE sqrt / division (no [V, 3] Omega_UC
    tensor is formed);
  - for each of its directions: eta, the fractional index on the uniform eta grid (exactly the arithmetic of
    `cross_sections.transfer_eta_at`, with lo, hi = eta_grid[0], eta_grid[-1] = -1, 1: pos = (clamp(eta, lo, hi) - lo) /
    (hi - lo) (N - 1), i0 = clamp(floor(pos), 0, N-2), w = clamp(pos - i0, 0, 1)); then for g' = 0..G-1 (unrolled) a
    [GP, BLOCK] gather of the two table rows (all output groups at once) per material slot (two in pair mode; MT in
    dense mode, an unrolled loop that re-reads the slot densities from cache), mixed with rho, times phi_UC,g'(v),
    accumulated into acc [GP, BLOCK];
  - writes acc to out[k, :G, v] (coalesced along v).
The table is re-laid out once per (library tensor, device) as tab [N, G_in, M, GP] float32 (GP = next power of
two >= G) with zeros where g_out < g_in (the no-upscatter mask) or g_out >= G (padding), so for a given voxel all
entries it needs lie in two contiguous G M GP-float rows (one per eta node). The table (N G M GP floats) is small
enough to stay L2 resident; neighboring voxels hit neighboring eta rows.
In pair mode the material slot offset (index x GP) is per voxel; the gathers per (voxel, direction, g_in) are four,
so the cost is close to that of a two-material basis for any M (only the larger table and the per-voxel offsets
add). Dense mode gathers 2 MT rows per (voxel, direction, g_in) at the constant offsets m GP: about MT / 2 times the
pair-mode cost per point.

Memory: the kernel reads rho, phi_uncollided (and the tiny axis/direction/table inputs) and writes the [K, G, V] output;
no temporaries. The `chunk` argument of the PyTorch path is irrelevant here (all K directions in one launch).

Tile: BLOCK_V = 64 voxels x GP output groups with 8 warps (about 2 elements per thread), the best of the block sizes
and warp counts tried; the kernel is bound by the table gathers, not by DRAM (docs/source/design_notes.rst).

Numerics: float32 only (float64 and CPU tensors use the PyTorch path, which stays the reference). The result
equals the PyTorch path up to float32 rounding of Omega_UC / eta and the summation order of the (identical) terms.
Where a steep part of the table is hit (the real library's Rayleigh forward peak, eta -> 1), one float32 ulp of eta
moves the interpolation point by ~1e-4 table cells, and BOTH float32 paths then deviate from a float64 evaluation (and
from each other) by more than that rounding (tests/test_uncollided.py asserts 1e-5 between the paths on the synthetic
tables and on the WLAB solver problems).

Availability: `triton_utilities.triton_available()` (shared probe, kill switch `CITRUS_DISABLE_TRITON=1`) and a
per-device self-check of the kernel against the PyTorch path on a tiny random problem (rho = 0 voxels, directions
parallel / antiparallel to Omega_UC; dense mode with 2 materials and with 7 materials mixed in most voxels, pair
mode), cached per device; `backend="auto"` falls back to the PyTorch path on any
failure.
"""

from __future__ import annotations

import weakref
from typing import Optional

import torch
from torch import Tensor

from . import triton_utilities

SELF_CHECK_REL_TOL = 1e-5
BLOCK_V = 64  # voxels per program (tile [GP, BLOCK_V], about 2 elements per thread: best measured, see
# docs/source/design_notes.rst)
NUM_WARPS = 8

_SELF_CHECK: dict[str, tuple[bool, str]] = {}
# prepared tables: id(library.mu_s_eta) -> (weak reference to it, {(device, versions): (tab, eta_lo, eta_hi)}); the
# entry
# is dropped when the library tensor is freed. (A WeakKeyDictionary would compare tensors with ==.)
_TABLES: dict[int, tuple[weakref.ref, dict[tuple, tuple[Tensor, float, float]]]] = {}

triton = triton_utilities.triton
tl = triton_utilities.tl


# ================================================================================================
# availability
# ================================================================================================

def triton_available() -> tuple[bool, str]:
    """(usable, reason): the shared Triton probe of `triton_utilities` (honours CITRUS_DISABLE_TRITON=1)."""
    return triton_utilities.triton_available()


def kernel_usable(device: torch.device) -> tuple[bool, str]:
    """(usable, reason): triton_available() and the fused kernel reproduces the PyTorch path of
    `uncollided.first_collision_source_at_directions` on `device` for a tiny random problem (cached per device, never
    raises)."""
    ok, why = triton_available()
    if not ok:
        return ok, why
    key = str(torch.device(device))
    if key not in _SELF_CHECK:
        try:
            _SELF_CHECK[key] = _self_check(torch.device(device))
        except Exception as e:  # noqa: BLE001
            _SELF_CHECK[key] = (False, f"self-check failed ({type(e).__name__}: {str(e)[:300]})")
    return _SELF_CHECK[key]


def use_kernel(t: Tensor, backend: str) -> bool:
    """Whether `first_collision_source_at_directions` should run the kernel for phi_uncollided-like tensor `t` under
    `backend` ("auto" | "torch" | "triton"). float64 and non-CUDA tensors always use the PyTorch path. Raises for
    backend="triton" when the kernel is unusable on a float32 CUDA tensor, and for an unknown backend."""
    if backend not in ("auto", "torch", "triton"):
        raise ValueError("backend must be 'auto', 'torch' or 'triton'")
    if backend == "torch" or t.device.type != "cuda" or t.dtype != torch.float32:
        return False
    ok, why = kernel_usable(t.device)
    if not ok and backend == "triton":
        raise RuntimeError(f"backend='triton' requested but the first-collision-source kernel is unusable: {why}")
    return ok


def _self_check(device: torch.device) -> tuple[bool, str]:
    """Kernel vs the PyTorch path on a 5 x 7 x 6 grid, G = 4 (synthetic library), 6 directions including the
    directions parallel and antiparallel to Omega_UC of one voxel (eta = +-1 exactly), rho = 0 in some voxels; dense
    mode (2 materials; 7 materials with up to 7 nonzero per voxel) and pair mode (7 materials, random pairs)."""
    from ..cross_sections import material_pairs, synthetic_library
    from ..geometry import ConeGeometry, Grid
    from ..uncollided import (_first_collision_source_torch, _first_collision_source_torch_pairs,
                              primary_directions_to_voxels)

    library = synthetic_library(4, legendre_order=1, num_eta=201)
    grid = Grid(shape=(5, 7, 6), delta_voxel=1.0, voxel_slice_aspect=0.8)
    geometry = ConeGeometry(source_iso_dist=20.0, source_detector_dist=30.0, num_det_rows=8, num_det_channels=8,
                            delta_det_row=1.0, delta_det_channel=1.0)
    angle = 0.4
    generator = torch.Generator().manual_seed(5)
    rho = torch.rand(2, *grid.shape, generator=generator)
    rho[0, 1, 2] = 0.0
    rho[:, 3] = 0.0
    phi = torch.rand(4, *grid.shape, generator=generator)
    omega_uncollided = primary_directions_to_voxels(grid, geometry, angle, "cpu", torch.float32)[2, 3, 1]
    directions = torch.cat([torch.nn.functional.normalize(torch.randn(4, 3, generator=generator), dim=-1),
                            omega_uncollided[None], -omega_uncollided[None]])
    rho, phi, directions = rho.to(device), phi.to(device), directions.to(device)
    reference = _first_collision_source_torch(library, rho, phi, grid, geometry, angle, directions, chunk=4)
    out = first_collision_source(library, rho, phi, grid, geometry, angle, directions)
    # pair mode: 7 materials, at most two nonzero per voxel, against the PyTorch pair path
    lib7 = synthetic_library(4, legendre_order=1, num_eta=201, num_materials=7)
    rho7 = torch.zeros(7, *grid.shape)
    sel = torch.randint(0, 7, (2,) + tuple(grid.shape), generator=generator)
    rho7.scatter_(0, sel, torch.rand((2,) + tuple(grid.shape), generator=generator) + 0.1)
    rho7 = rho7.to(device)
    pairs7 = material_pairs(rho7)
    reference7 = _first_collision_source_torch_pairs(lib7, pairs7, phi, grid, geometry, angle, directions, chunk=4)
    out7 = first_collision_source(lib7, rho7, phi, grid, geometry, angle, directions, pairs=pairs7)
    # dense mode, 7 materials, most of them nonzero in every voxel
    rho7d = torch.rand(7, *grid.shape, generator=generator) * (torch.rand(7, *grid.shape, generator=generator) > 0.2)
    rho7d = rho7d.to(device)
    reference7d = _first_collision_source_torch(lib7, rho7d, phi, grid, geometry, angle, directions, chunk=4)
    out7d = first_collision_source(lib7, rho7d, phi, grid, geometry, angle, directions)
    torch.cuda.synchronize(device)
    worst = 0.0
    for o, r in ((out, reference), (out7, reference7), (out7d, reference7d)):
        if o.shape != r.shape or not bool(torch.isfinite(o).all()):
            return False, "self-check: wrong shape or non-finite output"
        worst = max(worst, float((o - r).abs().max()) / max(float(r.abs().max()), 1e-30))
    if worst > SELF_CHECK_REL_TOL:
        return False, f"self-check: kernel deviates from the PyTorch path by {worst:.2e} (rel)"
    return True, f"kernel matches the PyTorch path, dense (2 and 7 materials) and pair modes (max rel dev {worst:.1e})"


# ================================================================================================
# kernel
# ================================================================================================

if triton is not None:

    @triton.jit
    def _first_collision_kernel(rho_ptr, index_ptr, phi_ptr, x_axis_ptr, y_axis_ptr, z_axis_ptr, direction_ptr,
                                table_ptr, out_ptr,
                                V, C, S, K, N, r0x, r0y, r0z, eta_lo, eta_hi,
                                G: tl.constexpr, GP: tl.constexpr, KB: tl.constexpr, BLOCK: tl.constexpr,
                                MT: tl.constexpr, PAIR: tl.constexpr):
        """Table row layout [G_in, MT, GP] (MT = materials of the table). Dense mode (PAIR False): rho_ptr holds the
        densities of all MT materials [MT, V] (slot m = material m at table offset m GP). Pair mode: rho_ptr the two
        slot densities, index_ptr the two slot material indices (int32 [2, V]); the slot offsets ia GP, ib GP are per
        voxel."""
        pid_v = tl.program_id(0)
        pid_k = tl.program_id(1)
        v = pid_v * BLOCK + tl.arange(0, BLOCK)
        vm = v < V
        # voxel center (flattened [rows, cols, slices]) -> Omega_UC
        CS = C * S
        i = v // CS
        rem = v - i * CS
        j = rem // S
        kk = rem - j * S
        dx = tl.load(x_axis_ptr + j, mask=vm, other=0.0) - r0x
        dy = tl.load(y_axis_ptr + i, mask=vm, other=0.0) - r0y
        dz = tl.load(z_axis_ptr + kk, mask=vm, other=1.0) - r0z
        nrm = tl.sqrt_rn(dx * dx + dy * dy + dz * dz)  # IEEE sqrt / division (Triton's defaults are approximate)
        ux = tl.div_rn(dx, nrm)
        uy = tl.div_rn(dy, nrm)
        uz = tl.div_rn(dz, nrm)
        go = tl.arange(0, GP)
        gm = go < G
        smask = gm[:, None] & vm[None, :]
        nm1 = (N - 1).to(tl.float32)
        span = eta_hi - eta_lo
        ROW: tl.constexpr = G * MT * GP  # floats per eta row of the table
        if PAIR:
            rh0 = tl.load(rho_ptr + v, mask=vm, other=0.0)[None, :]
            rh1 = tl.load(rho_ptr + V + v, mask=vm, other=0.0)[None, :]
            sa = (tl.load(index_ptr + v, mask=vm, other=0) * GP)[None, :]
            sb = (tl.load(index_ptr + V + v, mask=vm, other=0) * GP)[None, :]
        for kb in range(KB):
            k = pid_k * KB + kb
            if k < K:
                ox = tl.load(direction_ptr + 3 * k)
                oy = tl.load(direction_ptr + 3 * k + 1)
                oz = tl.load(direction_ptr + 3 * k + 2)
                eta = ux * ox + uy * oy + uz * oz
                eta = tl.minimum(tl.maximum(eta, eta_lo), eta_hi)
                pos = tl.div_rn(eta - eta_lo, span) * nm1
                i0 = tl.floor(pos)
                i0 = tl.minimum(tl.maximum(i0, 0.0), nm1 - 1.0)
                w = tl.minimum(tl.maximum(pos - i0, 0.0), 1.0)[None, :]
                rowoff = (i0.to(tl.int32) * ROW)[None, :] + go[:, None]
                acc = tl.zeros((GP, BLOCK), dtype=tl.float32)
                for gi in tl.static_range(G):
                    ph = tl.load(phi_ptr + gi * V + v, mask=vm, other=0.0)
                    off = rowoff + gi * (MT * GP)
                    if PAIR:
                        offa = off + sa
                        offb = off + sb
                        a0 = tl.load(table_ptr + offa, mask=smask, other=0.0)
                        a1 = tl.load(table_ptr + offa + ROW, mask=smask, other=0.0)
                        b0 = tl.load(table_ptr + offb, mask=smask, other=0.0)
                        b1 = tl.load(table_ptr + offb + ROW, mask=smask, other=0.0)
                        T = rh0 * (a0 + (a1 - a0) * w) + rh1 * (b0 + (b1 - b0) * w)
                    else:
                        T = tl.zeros((GP, BLOCK), dtype=tl.float32)
                        for m in tl.static_range(MT):
                            rm = tl.load(rho_ptr + m * V + v, mask=vm, other=0.0)[None, :]
                            t0 = tl.load(table_ptr + off + m * GP, mask=smask, other=0.0)
                            t1 = tl.load(table_ptr + off + m * GP + ROW, mask=smask, other=0.0)
                            T += rm * (t0 + (t1 - t0) * w)
                    acc += T * ph[None, :]
                obase = (k.to(tl.int64) * G + go[:, None].to(tl.int64)) * V + v[None, :]
                tl.store(out_ptr + obase, acc, mask=smask)


# ================================================================================================
# wrapper
# ================================================================================================

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def prepared_table(library, device: torch.device) -> tuple[Tensor, float, float]:
    """Kernel layout of the eta table of `library` and the ends of its (uniform) eta grid.

    library.mu_s_eta [M, G_in, G_out, N] (any device/dtype) -> kernel_table [N, G_in, M, GP] float32 contiguous on
    `device`, GP = next power of two >= G, zero where g_out < g_in (no upscatter) or g_out >= G (padding); plus
    (eta_lo, eta_hi) = (library.eta_grid[0], library.eta_grid[-1]) as Python floats. Cached per (source tensor,
    device, tensor versions), so the host reads the grid ends once per library and device.
    """
    source_table = library.mu_s_eta
    key = (str(torch.device(device)), int(source_table._version), int(library.eta_grid._version),
           id(library.eta_grid))
    hit = _TABLES.get(id(source_table))
    if hit is not None and hit[0]() is source_table and key in hit[1]:
        return hit[1][key]
    M, G, G2, N = source_table.shape
    if G != G2:
        raise ValueError(f"transfer table must be [M, G, G, N], got {tuple(source_table.shape)}")
    GP = _next_pow2(G)
    t = source_table.to(device=device, dtype=torch.float32)
    t = t * torch.triu(torch.ones(G, G, device=device, dtype=torch.float32))[None, :, :, None]
    kernel_table = torch.zeros(N, G, M, GP, device=device, dtype=torch.float32)
    kernel_table[..., :G] = t.permute(3, 1, 0, 2)
    eta_grid = library.eta_grid.detach().to("cpu", torch.float64)
    entry = (kernel_table.contiguous(), float(eta_grid[0]), float(eta_grid[-1]))
    table_id = id(source_table)
    if hit is None or hit[0]() is not source_table:
        _TABLES[table_id] = (weakref.ref(source_table, lambda _r, table_id=table_id: _TABLES.pop(table_id, None)), {})
    per_version = _TABLES[table_id][1]
    per_version.clear()  # keep only the latest version per source tensor
    per_version[key] = entry
    return entry


def _directions_per_program(K: int, num_vblocks: int, device: torch.device) -> int:
    """KB: directions handled by one program. Enough programs to fill the GPU (>= 8 per SM), then as many
    directions per program as possible (up to 8; rho and Omega_UC are loaded once per program)."""
    num_multiprocessors = torch.cuda.get_device_properties(device).multi_processor_count
    target = 8 * num_multiprocessors
    per_program = 1
    while per_program < 8 and per_program < K and num_vblocks * -(-K // (2 * per_program)) >= target:
        per_program *= 2
    return per_program


def kernel_supports(library, rho: Tensor, phi_uncollided: Tensor, pairs=None) -> bool:
    """Shape preconditions of the kernel: a [M, G, G, N] table matching phi_uncollided's G and either the densities of
    all M
    materials (dense mode, pairs None) or the pair representation of an M-material volume."""
    G = phi_uncollided.shape[0]
    t = library.mu_s_eta
    if tuple(t.shape[1:3]) != (G, G) or t.shape[3] < 2:
        return False
    if pairs is not None:
        return int(pairs.num_materials) == int(t.shape[0]) and int(rho.shape[0]) == int(t.shape[0])
    return int(rho.shape[0]) == int(t.shape[0]) >= 1


def first_collision_source(library, rho: Tensor, phi_uncollided: Tensor, grid, geometry, angle: float, omegas: Tensor,
                           pairs=None) -> Tensor:
    """Fused evaluation of `uncollided.first_collision_source_at_directions` (float32 CUDA only; no availability check
    here).

    library: MultigroupLibrary (uses eta_grid [N], ascending and uniform, and mu_s_eta [M, G, G, N]);
    rho: [M, rows, cols, slices]; pairs: the `cross_sections.MaterialPairs` of rho (pair mode: the kernel gathers the
    two table rows of each voxel by index) or None (dense mode: all M table rows per voxel, about M / 2 times the cost
    of the pair mode); phi_uncollided: [G, rows, cols, slices]
    float32 CUDA; omegas: [K, 3] unit vectors (recon frame). Returns [K, G, rows, cols, slices] float32 on
    phi_uncollided's device.
    """
    device, dtype = phi_uncollided.device, phi_uncollided.dtype
    if device.type != "cuda" or dtype != torch.float32:
        raise ValueError("the fused first-collision-source kernel needs float32 CUDA tensors")
    if not kernel_supports(library, rho, phi_uncollided, pairs):
        raise ValueError("the fused first-collision-source kernel needs the densities of all M materials of the [M, G, "
                         "G, N] table, "
                         "or the pair representation of the volume with a matching table")
    G = phi_uncollided.shape[0]
    shape = tuple(phi_uncollided.shape[1:])
    if tuple(grid.shape) != shape:
        raise ValueError(f"phi_uncollided spatial shape {shape} != grid shape {tuple(grid.shape)}")
    R, C, S = shape
    V = R * C * S
    if pairs is None:
        rho_flat = rho.to(device=device, dtype=dtype).reshape(-1, V).contiguous()
        index_flat = rho_flat  # unused by the dense variant
    else:
        rho_flat = pairs.density.to(device=device, dtype=dtype).reshape(2, V).contiguous()
        index_flat = pairs.index.to(device=device, dtype=torch.int32).reshape(2, V).contiguous()
    phi_flat = phi_uncollided.reshape(G, V).contiguous()
    directions = omegas.to(device=device, dtype=dtype).reshape(-1, 3).contiguous()
    K = directions.shape[0]
    out = torch.empty(K, G, *shape, device=device, dtype=dtype)
    if K == 0 or V == 0:
        return out
    table, eta_lo, eta_hi = prepared_table(library, device)
    x, y, z = grid.axis_coordinates(device, dtype)
    r0 = geometry.source_position(angle)
    num_voxel_blocks = -(-V // BLOCK_V)
    directions_per_program = _directions_per_program(K, num_voxel_blocks, device)
    _first_collision_kernel[(num_voxel_blocks, -(-K // directions_per_program))](
        rho_flat, index_flat, phi_flat, x, y, z, directions, table, out, V, C, S, K, int(table.shape[0]),
        float(r0[0]), float(r0[1]), float(r0[2]), eta_lo, eta_hi,
        G=G, GP=_next_pow2(G), KB=directions_per_program, BLOCK=BLOCK_V, MT=int(table.shape[2]), PAIR=pairs is not None,
        num_warps=NUM_WARPS)
    return out


def first_collision_source_or_none(library, rho: Tensor, phi_uncollided: Tensor, grid, geometry, angle: float,
                                   omegas: Tensor,
                                   backend: str, pairs=None) -> Optional[Tensor]:
    """`first_collision_source(...)` when the kernel applies (`use_kernel(phi_uncollided, backend)` and
    `kernel_supports`), else None
    (the caller then runs the PyTorch path)."""
    if not use_kernel(phi_uncollided, backend) or not kernel_supports(library, rho, phi_uncollided, pairs):
        return None
    return first_collision_source(library, rho, phi_uncollided, grid, geometry, angle, omegas, pairs=pairs)
