"""Fused Triton kernel for the first-collision source evaluation (`uncollided.fc_source_at_directions`).

The PyTorch body of `fc_source_at_directions` materializes the interpolated per-voxel transfer table
`T [G_in, G_out, k, V]` (and its two gathered end points `[M, G_in, G_out, k V]`) for every sub-chunk of
directions before contracting it with phi_uc. This module computes the result directly:

    S[k, g, v] = sum_{g' <= g} sum_m rho_m(v) lerp(table[m, g', g], eta_k(v)) phi_UC,g'(v)
    eta_k(v)   = Omega_UC(v) . omega_k,   Omega_UC(v) = (r_v - r0) / |r_v - r0|

Kernel `_fc_kernel`, per program (block of BLOCK flattened voxels, block of KB directions):
  - loads rho [2, BLOCK] once (two basis materials, docs/source/conventions.rst) and evaluates the voxel centers from the
    1-D axis coordinates of the grid (x[cols], y[rows], z[slices]) and Omega_UC with IEEE sqrt / division (no
    [V, 3] Omega_UC tensor is formed);
  - for each of its directions: eta, the fractional index on the uniform eta grid (exactly the arithmetic of
    `cross_sections.transfer_eta_at`, with lo, hi = eta_grid[0], eta_grid[-1] = -1, 1: pos = (clamp(eta, lo, hi) - lo) /
    (hi - lo) (N - 1), i0 = clamp(floor(pos), 0, N-2), w = clamp(pos - i0, 0, 1)); then for g' = 0..G-1 (unrolled) a [GP, BLOCK] gather of the two table rows
    (all output groups at once), mixed with rho, times phi_UC,g'(v), accumulated into acc [GP, BLOCK];
  - writes acc to out[k, :G, v] (coalesced along v).
The table is re-laid out once per (library tensor, device) as tab [N, G_in, M, GP] float32 (GP = next power of
two >= G) with zeros where g_out < g_in (the no-upscatter mask) or g_out >= G (padding), so for a given voxel all
entries it needs lie in two contiguous 2 G M GP-float rows. At G = 6, N = 2001 the table is 0.8 MB and stays
L2 resident; neighboring voxels hit neighboring eta rows.

Memory: the kernel reads rho, phi_uc (and the tiny axis/direction/table inputs) and writes the [K, G, V] output;
no temporaries. The `chunk` argument of the PyTorch path is irrelevant here (all K directions in one launch).

Tile: BLOCK_V = 64 voxels x GP output groups with 8 warps (about 2 elements per thread), measured best among
BLOCK 32-512 x 2-16 warps; the kernel is bound by the table gathers, not by DRAM (docs/source/design_notes.rst).

Numerics: float32 only (float64 and CPU tensors use the PyTorch path, which stays the reference). The result
equals the PyTorch path up to float32 rounding of Omega_UC / eta and the summation order of the (identical) terms:
typically ~2e-7 of the field maximum. Where a steep part of the table is hit (the real library's Rayleigh forward
peak, eta -> 1), one float32 ulp of eta moves the interpolation point by ~1e-4 table cells, and BOTH float32 paths
then deviate from a float64 evaluation (and from each other) by up to ~7e-5 of the field maximum
(tests/test_uncollided.py asserts 1e-5 between the paths on the synthetic tables and on the WLAB solver problems).

Availability: `triton_utils.triton_available()` (shared probe, kill switch `CITRUS_DISABLE_TRITON=1`) and a
per-device self-check of the kernel against the PyTorch path on a tiny random problem (rho = 0 voxels, directions
parallel / antiparallel to Omega_UC), cached per device; `backend="auto"` falls back to the PyTorch path on any
failure.
"""

from __future__ import annotations

import weakref
from typing import Optional

import torch
from torch import Tensor

from . import triton_utils as _tu

SELF_CHECK_REL_TOL = 1e-5
BLOCK_V = 64  # voxels per program (tile [GP, BLOCK_V], about 2 elements per thread: best measured, see
# docs/source/design_notes.rst)
NUM_WARPS = 8

_SELF_CHECK: dict[str, tuple[bool, str]] = {}
# prepared tables: id(lib.mu_s_eta) -> (weak reference to it, {(device, versions): (tab, eta_lo, eta_hi)}); the entry
# is dropped when the library tensor is freed. (A WeakKeyDictionary would compare tensors with ==.)
_TABLES: dict[int, tuple[weakref.ref, dict[tuple, tuple[Tensor, float, float]]]] = {}

triton = _tu.triton
tl = _tu.tl


# ================================================================================================
# availability
# ================================================================================================

def triton_available() -> tuple[bool, str]:
    """(usable, reason): the shared Triton probe of `triton_utils` (honours CITRUS_DISABLE_TRITON=1)."""
    return _tu.triton_available()


def kernel_usable(device: torch.device) -> tuple[bool, str]:
    """(usable, reason): triton_available() and the fused kernel reproduces the PyTorch path of
    `uncollided.fc_source_at_directions` on `device` for a tiny random problem (cached per device, never raises)."""
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
    """Whether `fc_source_at_directions` should run the kernel for phi_uc-like tensor `t` under `backend`
    ("auto" | "torch" | "triton"). float64 and non-CUDA tensors always use the PyTorch path. Raises for
    backend="triton" when the kernel is unusable on a float32 CUDA tensor, and for an unknown backend."""
    if backend not in ("auto", "torch", "triton"):
        raise ValueError("backend must be 'auto', 'torch' or 'triton'")
    if backend == "torch" or t.device.type != "cuda" or t.dtype != torch.float32:
        return False
    ok, why = kernel_usable(t.device)
    if not ok and backend == "triton":
        raise RuntimeError(f"backend='triton' requested but the FC-source kernel is unusable: {why}")
    return ok


def _self_check(device: torch.device) -> tuple[bool, str]:
    """Kernel vs the PyTorch path on a 5 x 7 x 6 grid, G = 4 (synthetic library), 6 directions including the
    directions parallel and antiparallel to Omega_UC of one voxel (eta = +-1 exactly), rho = 0 in some voxels."""
    from ..cross_sections import synthetic_library
    from ..geometry import ConeGeometry, Grid
    from ..uncollided import _fc_source_torch, primary_directions_to_voxels

    lib = synthetic_library(4, legendre_order=1, num_eta=201)
    grid = Grid(shape=(5, 7, 6), delta_voxel=1.0, voxel_slice_aspect=0.8)
    geom = ConeGeometry(source_iso_dist=20.0, source_detector_dist=30.0, num_det_rows=8, num_det_channels=8,
                        delta_det_row=1.0, delta_det_channel=1.0)
    angle = 0.4
    gen = torch.Generator().manual_seed(5)
    rho = torch.rand(2, *grid.shape, generator=gen)
    rho[0, 1, 2] = 0.0
    rho[:, 3] = 0.0
    phi = torch.rand(4, *grid.shape, generator=gen)
    om_uc = primary_directions_to_voxels(grid, geom, angle, "cpu", torch.float32)[2, 3, 1]
    om = torch.cat([torch.nn.functional.normalize(torch.randn(4, 3, generator=gen), dim=-1), om_uc[None], -om_uc[None]])
    rho, phi, om = rho.to(device), phi.to(device), om.to(device)
    ref = _fc_source_torch(lib, rho, phi, grid, geom, angle, om, chunk=4)
    out = fc_source(lib, rho, phi, grid, geom, angle, om)
    torch.cuda.synchronize(device)
    if out.shape != ref.shape or not bool(torch.isfinite(out).all()):
        return False, "self-check: wrong shape or non-finite output"
    worst = float((out - ref).abs().max()) / max(float(ref.abs().max()), 1e-30)
    if worst > SELF_CHECK_REL_TOL:
        return False, f"self-check: kernel deviates from the PyTorch path by {worst:.2e} (rel)"
    return True, f"kernel matches the PyTorch path (max rel dev {worst:.1e})"


# ================================================================================================
# kernel
# ================================================================================================

if triton is not None:

    @triton.jit
    def _fc_kernel(rho_ptr, phi_ptr, xax_ptr, yax_ptr, zax_ptr, om_ptr, tab_ptr, out_ptr,
                   V, C, S, K, N, r0x, r0y, r0z, eta_lo, eta_hi,
                   G: tl.constexpr, GP: tl.constexpr, KB: tl.constexpr, BLOCK: tl.constexpr):
        """Two basis materials (M = 2, docs/source/conventions.rst); table row layout [G_in, M, GP]."""
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
        dx = tl.load(xax_ptr + j, mask=vm, other=0.0) - r0x
        dy = tl.load(yax_ptr + i, mask=vm, other=0.0) - r0y
        dz = tl.load(zax_ptr + kk, mask=vm, other=1.0) - r0z
        nrm = tl.sqrt_rn(dx * dx + dy * dy + dz * dz)  # IEEE sqrt / division (Triton's defaults are approximate)
        ux = tl.div_rn(dx, nrm)
        uy = tl.div_rn(dy, nrm)
        uz = tl.div_rn(dz, nrm)
        rh0 = tl.load(rho_ptr + v, mask=vm, other=0.0)[None, :]
        rh1 = tl.load(rho_ptr + V + v, mask=vm, other=0.0)[None, :]
        go = tl.arange(0, GP)
        gm = go < G
        smask = gm[:, None] & vm[None, :]
        nm1 = (N - 1).to(tl.float32)
        span = eta_hi - eta_lo
        ROW: tl.constexpr = G * 2 * GP  # floats per eta row of the table
        for kb in range(KB):
            k = pid_k * KB + kb
            if k < K:
                ox = tl.load(om_ptr + 3 * k)
                oy = tl.load(om_ptr + 3 * k + 1)
                oz = tl.load(om_ptr + 3 * k + 2)
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
                    off = rowoff + gi * (2 * GP)
                    a0 = tl.load(tab_ptr + off, mask=smask, other=0.0)
                    a1 = tl.load(tab_ptr + off + ROW, mask=smask, other=0.0)
                    b0 = tl.load(tab_ptr + off + GP, mask=smask, other=0.0)
                    b1 = tl.load(tab_ptr + off + (ROW + GP), mask=smask, other=0.0)
                    T = rh0 * (a0 + (a1 - a0) * w) + rh1 * (b0 + (b1 - b0) * w)
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


def prepared_table(lib, device: torch.device) -> tuple[Tensor, float, float]:
    """Kernel layout of the eta table of `lib` and the ends of its (uniform) eta grid.

    lib.mu_s_eta [M = 2, G_in, G_out, N] (any device/dtype) -> tab [N, G_in, M, GP] float32 contiguous on `device`,
    GP = next power of two >= G, zero where g_out < g_in (no upscatter) or g_out >= G (padding); plus
    (eta_lo, eta_hi) = (lib.eta_grid[0], lib.eta_grid[-1]) as Python floats. Cached per (source tensor, device,
    tensor versions), so the host reads the grid ends once per library and device.
    """
    table = lib.mu_s_eta
    key = (str(torch.device(device)), int(table._version), int(lib.eta_grid._version), id(lib.eta_grid))
    hit = _TABLES.get(id(table))
    if hit is not None and hit[0]() is table and key in hit[1]:
        return hit[1][key]
    M, G, G2, N = table.shape
    if G != G2 or M != 2:
        raise ValueError(f"transfer table must be [2, G, G, N], got {tuple(table.shape)}")
    GP = _next_pow2(G)
    t = table.to(device=device, dtype=torch.float32)
    t = t * torch.triu(torch.ones(G, G, device=device, dtype=torch.float32))[None, :, :, None]
    tab = torch.zeros(N, G, M, GP, device=device, dtype=torch.float32)
    tab[..., :G] = t.permute(3, 1, 0, 2)
    eg = lib.eta_grid.detach().to("cpu", torch.float64)
    entry = (tab.contiguous(), float(eg[0]), float(eg[-1]))
    tid = id(table)
    if hit is None or hit[0]() is not table:
        _TABLES[tid] = (weakref.ref(table, lambda _r, tid=tid: _TABLES.pop(tid, None)), {})
    per = _TABLES[tid][1]
    per.clear()  # keep only the latest version per source tensor
    per[key] = entry
    return entry


def _directions_per_program(K: int, num_vblocks: int, device: torch.device) -> int:
    """KB: directions handled by one program. Enough programs to fill the GPU (>= 8 per SM), then as many
    directions per program as possible (up to 8; rho and Omega_UC are loaded once per program)."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    target = 8 * sms
    kb = 1
    while kb < 8 and kb < K and num_vblocks * -(-K // (2 * kb)) >= target:
        kb *= 2
    return kb


def kernel_supports(lib, rho: Tensor, phi_uc: Tensor) -> bool:
    """Shape preconditions of the kernel: two basis materials and a [2, G, G, N] table matching phi_uc's G."""
    G = phi_uc.shape[0]
    return rho.shape[0] == 2 and tuple(lib.mu_s_eta.shape[:3]) == (2, G, G) and lib.mu_s_eta.shape[3] >= 2


def fc_source(lib, rho: Tensor, phi_uc: Tensor, grid, geom, angle: float, omegas: Tensor) -> Tensor:
    """Fused evaluation of `uncollided.fc_source_at_directions` (float32 CUDA only; no availability check here).

    lib: MultigroupLibrary (uses eta_grid [N], ascending and uniform, and mu_s_eta [2, G, G, N]);
    rho: [2, rows, cols, slices]; phi_uc: [G, rows, cols, slices] float32 CUDA; omegas: [K, 3] unit vectors
    (recon frame). Returns [K, G, rows, cols, slices] float32 on phi_uc's device.
    """
    dev, dt = phi_uc.device, phi_uc.dtype
    if dev.type != "cuda" or dt != torch.float32:
        raise ValueError("the fused FC-source kernel needs float32 CUDA tensors")
    if not kernel_supports(lib, rho, phi_uc):
        raise ValueError("the fused FC-source kernel needs two basis materials and a [2, G, G, N] table")
    G = phi_uc.shape[0]
    shape = tuple(phi_uc.shape[1:])
    if tuple(grid.shape) != shape:
        raise ValueError(f"phi_uc spatial shape {shape} != grid shape {tuple(grid.shape)}")
    R, C, S = shape
    V = R * C * S
    rho_f = rho.to(device=dev, dtype=dt).reshape(2, V).contiguous()
    phi_f = phi_uc.reshape(G, V).contiguous()
    om = omegas.to(device=dev, dtype=dt).reshape(-1, 3).contiguous()
    K = om.shape[0]
    out = torch.empty(K, G, *shape, device=dev, dtype=dt)
    if K == 0 or V == 0:
        return out
    tab, eta_lo, eta_hi = prepared_table(lib, dev)
    x, y, z = grid.axis_coordinates(dev, dt)
    r0 = geom.source_position(angle)
    nvb = -(-V // BLOCK_V)
    kb = _directions_per_program(K, nvb, dev)
    _fc_kernel[(nvb, -(-K // kb))](
        rho_f, phi_f, x, y, z, om, tab, out, V, C, S, K, int(tab.shape[0]),
        float(r0[0]), float(r0[1]), float(r0[2]), eta_lo, eta_hi,
        G=G, GP=_next_pow2(G), KB=kb, BLOCK=BLOCK_V, num_warps=NUM_WARPS)
    return out


def fc_source_or_none(lib, rho: Tensor, phi_uc: Tensor, grid, geom, angle: float, omegas: Tensor,
                      backend: str) -> Optional[Tensor]:
    """`fc_source(...)` when the kernel applies (`use_kernel(phi_uc, backend)` and `kernel_supports`), else None
    (the caller then runs the PyTorch path)."""
    if not use_kernel(phi_uc, backend) or not kernel_supports(lib, rho, phi_uc):
        return None
    return fc_source(lib, rho, phi_uc, grid, geom, angle, omegas)
