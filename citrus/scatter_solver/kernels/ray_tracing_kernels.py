"""Fused Triton kernels for the exact (Siddon / Amanatides-Woo) ray tracer of `ray_tracing.py`.

The PyTorch reference (`ray_tracing._trace_chunk`) materializes, per chunk of rays, every plane-crossing parameter of
every ray ([N, nx + ny + nz - 1]), sorts them, gathers [N, P, G] voxel values and contracts them. This module
traces each ray in registers instead: one program per block of BLOCK rays, each ray walks its voxel crossings in
parametric order and accumulates all G groups (a [GP, BLOCK] tile, GP = next power of two >= G, G a compile-time
constant) while the traversal is shared across the groups. Nothing but the output is written to memory.

Kernels
  `_ray_kernel` (generic DDA, one ray per lane; MODE selects where the segment end points come from):
    MODE 0  line_integrals:       p0, p1 read from starts / ends [N, 3] (any strides), optional per-ray volume index
    MODE 1  oblique_project:      the line through pixel center (r, c) with direction omega_k, clipped to the
                                  box-centered segment of `ray_tracing.SiddonProjectors.oblique_project` (computed in-kernel)
    MODE 2  upstream_cumulative:  focal spot -> voxel center
    MODE 3  primary_project:      focal spot -> pixel center (n_sub = 1; sub-pixel rays use MODE 0)
    Per ray: slab clipping exactly as the reference ([a_lo, a_hi] in the segment parameter a in [0, 1]), the entry
    voxel (floor of the entry point; ceil - 1 along axes the ray descends, so a ray starting on a plane enters the
    voxel it heads into), then the loop: a_next = min(next x / y / z plane parameter, a_hi); acc += vol[voxel]
    (a_next - a); step the axis with the smallest crossing (ties: x, then y, then z: the tied axis follows with a
    zero-length segment, as the reference's coincident sorted parameters). The plane parameters are recomputed
    from the voxel index each step ((lo + k delta - p0) / d, the reference's arithmetic), so no error accumulates.
    The loop runs to the block maximum of a per-ray bound (1 + sum_a (floor(L_a / delta_a) + 1) + 1 crossings),
    lanes that are done are masked.
  `_down_shared_kernel` (downstream_cumulative): all rays of a direction start at
    voxel centers and are parallel, so the distance from the start to the first x / y / z plane is half a voxel
    for EVERY ray and the whole crossing sequence t_m = (n_a + 1/2) delta_a / |omega_a| is identical for all rays
    of the direction (Amanatides-Woo with a shared tMax). Each program (direction k, BLOCK consecutive voxels in
    (row, col, slice) order) runs that scalar sequence once: the axis choice is uniform across the block (no
    divergence, no per-ray plane arithmetic), each ray only offsets its voxel index by the shared step, and the
    loads of a step are a contiguous strip of the volume shifted by the shared offset (coalesced). The ray stops
    at its box exit (per-ray index limits) or at the panel plane (per-ray t_end = clamp(panel distance, 0, diag),
    the reference's end point), so the result is exact for any direction (axis-aligned, through edges / corners).
    Directions with omega . (-n_hat) <= 0 give +inf; the decision is taken with the reference's own torch
    expression (passed in as a [K] flag tensor) so it is bit-identical for grazing directions.

Numerics: float32. Each ray's integral equals the reference up to float32 rounding of the crossing parameters and
the summation order: on the WLAB grids 2e-7 - 6e-7 of the maximum for downstream_cumulative and 1e-6 - 3.4e-6 for
oblique / upstream / primary at 0.5 cm, the same order as the float32 reference's own deviation from its float64
evaluation (up to 6.4e-6); tests/test_ray_tracing.py asserts 1e-5. No atomics: results are bitwise reproducible.

Availability (the pattern of `uncollided_kernels.py` / `sweep_kernels.py`): `triton_available()` is the shared
probe of `triton_utils` (kill switch CITRUS_DISABLE_TRITON=1); `kernels_usable(device)` additionally runs
every operation against the PyTorch path on a tiny problem once per device and caches the verdict;
`SiddonProjectors(backend="auto")` falls back to PyTorch on any failure. The kernels are compiled without value or
alignment specialization so the fast launch path of `triton_utils.launch` (direct CompiledKernel.run,
triton 3.8) reuses one binary per (kernel, constexprs, num_warps, device); `warmup(device, G)` compiles the
variants of one group count ahead of timing.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from . import triton_utils as _tu

SELF_CHECK_REL_TOL = 1e-5
BLOCK_RAYS = 128  # rays per program of `_ray_kernel` (tile [GP, 128], 4 warps)
BLOCK_DOWN = 128  # voxels per program of `_down_shared_kernel`
NUM_WARPS = 4
BIG = 3.0e38  # "no crossing" parameter (finite, so that products stay finite)
_BIG_INT = 1 << 30

_SELF_CHECK: dict[str, tuple[bool, str]] = {}

triton = _tu.triton
tl = _tu.tl

MODE_GENERIC, MODE_OBLIQUE, MODE_UPSTREAM, MODE_PRIMARY = 0, 1, 2, 3


# ================================================================================================
# availability
# ================================================================================================

def triton_available() -> tuple[bool, str]:
    """(usable, reason): the shared Triton probe of `triton_utils` (honours CITRUS_DISABLE_TRITON=1)."""
    return _tu.triton_available()


def kernels_usable(device: torch.device) -> tuple[bool, str]:
    """(usable, reason): triton_available() and every kernel of this module reproduces the PyTorch Siddon path on
    `device` for a tiny random problem (all four projector operations, line_integrals with a stacked volume index;
    cached per device, never raises)."""
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


def use_kernels(t: Tensor, backend: str) -> bool:
    """Whether a SiddonProjectors operation on tensor `t` should run the kernels under `backend`
    ("auto" | "torch" | "triton"). float64 and non-CUDA tensors always use PyTorch. Raises for backend="triton"
    when the kernels are unusable on a float32 CUDA tensor, and for an unknown backend."""
    if backend not in ("auto", "torch", "triton"):
        raise ValueError("backend must be 'auto', 'torch' or 'triton'")
    if backend == "torch" or t.device.type != "cuda" or t.dtype != torch.float32:
        return False
    ok, why = kernels_usable(t.device)
    if not ok and backend == "triton":
        raise RuntimeError(f"backend='triton' requested but the Siddon kernels are unusable: {why}")
    return ok


def _self_check(device: torch.device) -> tuple[bool, str]:
    """Kernels vs the PyTorch Siddon path on a 5 x 7 x 6 grid (anisotropic voxels), G = 2, at view angle 0.3:
    upstream, primary (n_sub 1 and 2), downstream (directions incl. an axis-aligned one and one pointing away
    from the panel), oblique (shared and per-direction fields) and line_integrals (random segments, a stacked
    volume index)."""
    from ..geometry import ConeGeometry, Grid
    from ..ray_tracing import SiddonProjectors, line_integrals

    grid = Grid(shape=(5, 7, 6), delta_voxel=1.0, voxel_row_aspect=1.25, voxel_slice_aspect=0.8,
                recon_slice_offset=0.2)
    geom = ConeGeometry(source_iso_dist=30.0, source_detector_dist=45.0, num_det_rows=9, num_det_channels=11,
                        delta_det_row=1.1, delta_det_channel=1.3)
    angle = 0.3
    gen = torch.Generator().manual_seed(9)
    mu = (0.3 * torch.rand(2, *grid.shape, generator=gen)).to(device)
    om = torch.tensor([[0.2, -1.0, 0.1], [-0.3, -1.0, 0.25], [0.0, -1.0, 0.0], [0.1, 1.0, 0.0]])
    om = geom.view_to_recon(torch.nn.functional.normalize(om, dim=-1), angle).to(device)
    K = om.shape[0]
    field = torch.rand(K, 2, *grid.shape, generator=gen).to(device)
    lo = torch.tensor(grid.lower_corner())
    ext = torch.tensor(grid.extent)
    a = (lo + (torch.rand(64, 3, generator=gen) * 1.6 - 0.3) * ext).to(device)
    b = (lo + (torch.rand(64, 3, generator=gen) * 1.6 - 0.3) * ext).to(device)
    kk = torch.randint(0, K, (64,), generator=gen).to(device)
    ref = SiddonProjectors(backend="torch")
    pairs = [
        (ref.upstream_cumulative(mu, grid, geom, angle), upstream_cumulative(mu, grid, geom, angle)),
        (ref.primary_project(mu, grid, geom, angle), primary_project(mu, grid, geom, angle)),
        (ref.primary_project(mu, grid, geom, angle, n_sub=2), primary_project(mu, grid, geom, angle, n_sub=2)),
        (ref.downstream_cumulative(mu, grid, geom, angle, om), downstream_cumulative(mu, grid, geom, angle, om)),
        (ref.oblique_project(field[0], grid, geom, angle, om), oblique_project(field[0], grid, geom, angle, om)),
        (ref.oblique_project(field, grid, geom, angle, om), oblique_project(field, grid, geom, angle, om)),
        (line_integrals(field, a, b, grid, volume_index=kk), line_integrals_triton(field, a, b, grid, volume_index=kk)),
    ]
    torch.cuda.synchronize(device)
    worst = 0.0
    for r, t in pairs:
        if r.shape != t.shape:
            return False, f"self-check: shape {tuple(t.shape)} != {tuple(r.shape)}"
        m = torch.isfinite(r)
        if not bool((torch.isfinite(t) == m).all()):
            return False, "self-check: +inf pattern differs"
        scale = float(r[m].abs().max()) if bool(m.any()) else 1.0
        if bool(m.any()):
            worst = max(worst, float((r[m] - t[m]).abs().max()) / max(scale, 1e-30))
    if worst > SELF_CHECK_REL_TOL:
        return False, f"self-check: kernels deviate from the PyTorch Siddon path by {worst:.2e} (rel)"
    return True, f"kernels match the PyTorch Siddon path (max rel dev {worst:.1e})"


# ================================================================================================
# kernels
# ================================================================================================

if triton is not None:

    _BIG_C = tl.constexpr(BIG)
    _BIG_INT_C = tl.constexpr(_BIG_INT)

    _RAY_INTS = ["M", "sa_n", "sa_c", "sb_n", "sb_c", "sK", "sG", "sR", "sC", "sS", "NX", "NY", "NZ", "Rd", "Cd"]
    _RAY_PTRS = ["vol_ptr", "out_ptr", "a_ptr", "b_ptr", "vidx_ptr", "om_ptr"]

    @triton.jit
    def _plane_param(lo, delta, idx, dpos, p0, inv, nz):
        """Parameter of the next plane crossing along one axis for voxel index idx (_BIG_C if the axis is constant)."""
        k = idx + dpos.to(tl.int32)
        return tl.where(nz, (lo + k.to(tl.float32) * delta - p0) * inv, _BIG_C)

    @triton.jit
    def _entry_index(p0, d, a, lo, delta, n):
        """Voxel index along one axis at parameter a (floor; ceil - 1 when descending), clamped to [0, n - 1]."""
        f = (p0 + a * d - lo) / delta
        i = tl.where(d < 0, tl.ceil(f) - 1.0, tl.floor(f))
        return tl.minimum(tl.maximum(i, 0.0), (n - 1).to(tl.float32)).to(tl.int32)

    @triton.jit
    def _trace(vol_ptr, vbase, p0x, p0y, p0z, p1x, p1y, p1z, live, sG, sR, sC, sS,
               lox, loy, loz, hix, hiy, hiz, ddx, ddy, ddz, NX, NY, NZ,
               G: tl.constexpr, GP: tl.constexpr, BLOCK: tl.constexpr):
        """Exact integral of the voxel field along p0 -> p1 for a block of rays.

        p0*, p1*: [BLOCK] end points; live [BLOCK]: lanes holding a ray; vbase [BLOCK] int64 offsets of each ray's
        volume. Returns [GP, BLOCK] (rows g >= G are zero)."""
        dX = p1x - p0x
        dY = p1y - p0y
        dZ = p1z - p0z
        nzx = dX != 0.0
        nzy = dY != 0.0
        nzz = dZ != 0.0
        invx = 1.0 / tl.where(nzx, dX, 1.0)
        invy = 1.0 / tl.where(nzy, dY, 1.0)
        invz = 1.0 / tl.where(nzz, dZ, 1.0)
        # slab clipping (the reference's arithmetic)
        a1 = (lox - p0x) * invx
        a2 = (hix - p0x) * invx
        insx = (p0x >= lox) & (p0x <= hix)
        aminx = tl.where(nzx, tl.minimum(a1, a2), tl.where(insx, -_BIG_C, _BIG_C))
        amaxx = tl.where(nzx, tl.maximum(a1, a2), tl.where(insx, _BIG_C, -_BIG_C))
        a1 = (loy - p0y) * invy
        a2 = (hiy - p0y) * invy
        insy = (p0y >= loy) & (p0y <= hiy)
        aminy = tl.where(nzy, tl.minimum(a1, a2), tl.where(insy, -_BIG_C, _BIG_C))
        amaxy = tl.where(nzy, tl.maximum(a1, a2), tl.where(insy, _BIG_C, -_BIG_C))
        a1 = (loz - p0z) * invz
        a2 = (hiz - p0z) * invz
        insz = (p0z >= loz) & (p0z <= hiz)
        aminz = tl.where(nzz, tl.minimum(a1, a2), tl.where(insz, -_BIG_C, _BIG_C))
        amaxz = tl.where(nzz, tl.maximum(a1, a2), tl.where(insz, _BIG_C, -_BIG_C))
        a_lo = tl.maximum(tl.maximum(tl.maximum(aminx, aminy), aminz), 0.0)
        a_hi = tl.minimum(tl.minimum(tl.minimum(amaxx, amaxy), amaxz), 1.0)
        act = live & (a_hi > a_lo)
        a_lo = tl.where(act, a_lo, 0.0)
        a_hi = tl.where(act, a_hi, 0.0)
        # entry voxel and first crossings
        ix = _entry_index(p0x, dX, a_lo, lox, ddx, NX)
        iy = _entry_index(p0y, dY, a_lo, loy, ddy, NY)
        iz = _entry_index(p0z, dZ, a_lo, loz, ddz, NZ)
        posx = dX > 0.0
        posy = dY > 0.0
        posz = dZ > 0.0
        stx = tl.where(posx, 1, tl.where(dX < 0.0, -1, 0))
        sty = tl.where(posy, 1, tl.where(dY < 0.0, -1, 0))
        stz = tl.where(posz, 1, tl.where(dZ < 0.0, -1, 0))
        anx = _plane_param(lox, ddx, ix, posx, p0x, invx, nzx)
        any_ = _plane_param(loy, ddy, iy, posy, p0y, invy, nzy)
        anz = _plane_param(loz, ddz, iz, posz, p0z, invz, nzz)
        # bound on the number of pieces: 1 + crossings per axis (<= floor(L_a / delta_a) + 1) + 1 spare
        span = a_hi - a_lo
        cnt = (tl.floor(tl.abs(dX) * span / ddx) + tl.floor(tl.abs(dY) * span / ddy)
               + tl.floor(tl.abs(dZ) * span / ddz) + 5.0)
        nmax = tl.max(tl.where(act, cnt, 0.0).to(tl.int32), axis=0)
        g = tl.arange(0, GP)
        gm = g < G
        gofs = g.to(tl.int64) * sG
        acc = tl.zeros([GP, BLOCK], dtype=tl.float32)
        a = a_lo
        for _ in range(0, nmax):
            an = tl.minimum(tl.minimum(anx, any_), tl.minimum(anz, a_hi))
            seg = tl.where(act, tl.maximum(an - a, 0.0), 0.0)
            inb = act & (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY) & (iz >= 0) & (iz < NZ)
            sp = iy.to(tl.int64) * sR + ix.to(tl.int64) * sC + iz.to(tl.int64) * sS
            lm = gm[:, None] & inb[None, :]
            v = tl.load(vol_ptr + (vbase + sp)[None, :] + gofs[:, None], mask=lm, other=0.0)
            acc += v * seg[None, :]
            selx = (anx <= any_) & (anx <= anz)
            sely = (any_ < anx) & (any_ <= anz)
            selz = (anz < anx) & (anz < any_)
            ix += tl.where(selx, stx, 0)
            iy += tl.where(sely, sty, 0)
            iz += tl.where(selz, stz, 0)
            anx = tl.where(selx, _plane_param(lox, ddx, ix, posx, p0x, invx, nzx), anx)
            any_ = tl.where(sely, _plane_param(loy, ddy, iy, posy, p0y, invy, nzy), any_)
            anz = tl.where(selz, _plane_param(loz, ddz, iz, posz, p0z, invz, nzz), anz)
            act = act & (an < a_hi)
            a = an
        length = tl.sqrt(dX * dX + dY * dY + dZ * dZ)
        return acc * length[None, :]

    @triton.jit(do_not_specialize=_RAY_INTS, do_not_specialize_on_alignment=_RAY_PTRS)
    def _ray_kernel(vol_ptr, out_ptr, a_ptr, b_ptr, vidx_ptr, om_ptr,
                    M, sa_n, sa_c, sb_n, sb_c, sK, sG, sR, sC, sS, NX, NY, NZ, Rd, Cd,
                    lox, loy, loz, hix, hiy, hiz, ddx, ddy, ddz, zoff,
                    qx, qy, qz, ux_, uy_, uz_, vx_, vy_, vz_,
                    rmid, cmid, drow, dch, rowoff, choff, bcx, bcy, bcz, half,
                    MODE: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, HAS_VIDX: tl.constexpr,
                    BLOCK: tl.constexpr):
        """One block of rays (program_id 0) of direction / major index k (program_id 1); out[(k G + g) M + m]."""
        k = tl.program_id(1)
        m = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = m < M
        mm = tl.where(live, m, 0)
        o_idx = mm
        kk = k.to(tl.int64)
        vbase = tl.zeros([BLOCK], dtype=tl.int64) + kk * sK
        if MODE == 0:
            ra = a_ptr + mm.to(tl.int64) * sa_n
            rb = b_ptr + mm.to(tl.int64) * sb_n
            p0x = tl.load(ra, mask=live, other=0.0)
            p0y = tl.load(ra + sa_c, mask=live, other=0.0)
            p0z = tl.load(ra + 2 * sa_c, mask=live, other=0.0)
            p1x = tl.load(rb, mask=live, other=0.0)
            p1y = tl.load(rb + sb_c, mask=live, other=0.0)
            p1z = tl.load(rb + 2 * sb_c, mask=live, other=0.0)
            if HAS_VIDX:
                vi = tl.load(vidx_ptr + mm, mask=live, other=0).to(tl.int64)
                vbase = vi * sK
        elif MODE == 1 or MODE == 3:
            # pixel center (r, c) = panel center + u u_hat + v v_hat (ConeGeometry.pixel_centers). Lanes run along
            # detector rows (v ~ z, the contiguous volume axis), so neighbouring rays gather neighbouring slices
            r = mm % Rd
            c = mm // Rd
            o_idx = r * Cd + c
            u = (c.to(tl.float32) - cmid) * dch - choff
            v = (r.to(tl.float32) - rmid) * drow - rowoff
            px = qx + u * ux_ + v * vx_
            py = qy + u * uy_ + v * vy_
            pz = qz + u * uz_ + v * vz_
            if MODE == 1:
                ox = tl.load(om_ptr + 3 * k)
                oy = tl.load(om_ptr + 3 * k + 1)
                oz = tl.load(om_ptr + 3 * k + 2)
                om2 = ox * ox + oy * oy + oz * oz
                tc = ((bcx - px) * ox + (bcy - py) * oy + (bcz - pz) * oz) / om2
                h = half / tl.sqrt(om2)
                p0x = px + (tc - h) * ox
                p0y = py + (tc - h) * oy
                p0z = pz + (tc - h) * oz
                p1x = px + (tc + h) * ox
                p1y = py + (tc + h) * oy
                p1z = pz + (tc + h) * oz
            else:
                p0x = tl.zeros([BLOCK], dtype=tl.float32) + bcx  # the focal spot (box-center slots)
                p0y = tl.zeros([BLOCK], dtype=tl.float32) + bcy
                p0z = tl.zeros([BLOCK], dtype=tl.float32) + bcz
                p1x = px
                p1y = py
                p1z = pz
        else:
            # voxel centers (Grid.axis_coordinates) of the flat (row, col, slice) index
            s = mm % NZ
            c = (mm // NZ) % NX
            r = mm // (NZ * NX)
            cx = ddx * (c.to(tl.float32) - (NX - 1) * 0.5)
            cy = ddy * (r.to(tl.float32) - (NY - 1) * 0.5)
            cz = ddz * (s.to(tl.float32) - (NZ - 1) * 0.5) + zoff
            p0x = tl.zeros([BLOCK], dtype=tl.float32) + qx
            p0y = tl.zeros([BLOCK], dtype=tl.float32) + qy
            p0z = tl.zeros([BLOCK], dtype=tl.float32) + qz
            p1x = cx
            p1y = cy
            p1z = cz
        acc = _trace(vol_ptr, vbase, p0x, p0y, p0z, p1x, p1y, p1z, live, sG, sR, sC, sS,
                     lox, loy, loz, hix, hiy, hiz, ddx, ddy, ddz, NX, NY, NZ, G, GP, BLOCK)
        g = tl.arange(0, GP)
        if MODE == 1 or MODE == 3:
            m = o_idx
        optr = out_ptr + (kk * G + g.to(tl.int64))[:, None] * M + m[None, :].to(tl.int64)
        tl.store(optr, acc, mask=(g < G)[:, None] & live[None, :])

    _DOWN_INTS = ["V", "sG", "sR", "sC", "sS", "NX", "NY", "NZ"]

    @triton.jit(do_not_specialize=_DOWN_INTS,
                do_not_specialize_on_alignment=["mu_ptr", "om_ptr", "tw_ptr", "out_ptr"])
    def _down_shared_kernel(mu_ptr, om_ptr, tw_ptr, out_ptr, V, sG, sR, sC, sS, NX, NY, NZ,
                            ddx, ddy, ddz, zoff, qx, qy, qz, nx_, ny_, nz_, diag,
                            G: tl.constexpr, GP: tl.constexpr, BLOCK: tl.constexpr):
        """tau_down of BLOCK consecutive voxels (program_id 0) for direction k (program_id 1) with the shared
        crossing sequence of the direction (module docstring); out[(k G + g) V + v]."""
        k = tl.program_id(1)
        m = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = m < V
        mm = tl.where(live, m, 0)
        ox = tl.load(om_ptr + 3 * k)
        oy = tl.load(om_ptr + 3 * k + 1)
        oz = tl.load(om_ptr + 3 * k + 2)
        toward = tl.load(tw_ptr + k) != 0
        s = mm % NZ
        c = (mm // NZ) % NX
        r = mm // (NZ * NX)
        cx = ddx * (c.to(tl.float32) - (NX - 1) * 0.5)
        cy = ddy * (r.to(tl.float32) - (NY - 1) * 0.5)
        cz = ddz * (s.to(tl.float32) - (NZ - 1) * 0.5) + zoff
        den = ox * nx_ + oy * ny_ + oz * nz_
        t_end = ((qx - cx) * nx_ + (qy - cy) * ny_ + (qz - cz) * nz_) / den
        t_end = tl.minimum(tl.maximum(t_end, 0.0), diag)
        # shared crossing geometry: plane m of axis a at t = (m + 1/2) delta_a / |omega_a|
        sgx = tl.where(ox > 0.0, 1, tl.where(ox < 0.0, -1, 0))
        sgy = tl.where(oy > 0.0, 1, tl.where(oy < 0.0, -1, 0))
        sgz = tl.where(oz > 0.0, 1, tl.where(oz < 0.0, -1, 0))
        tdx = tl.where(sgx != 0, ddx / tl.where(sgx != 0, tl.abs(ox), 1.0), _BIG_C)
        tdy = tl.where(sgy != 0, ddy / tl.where(sgy != 0, tl.abs(oy), 1.0), _BIG_C)
        tdz = tl.where(sgz != 0, ddz / tl.where(sgz != 0, tl.abs(oz), 1.0), _BIG_C)
        # per-ray number of crossings along each axis before leaving the box
        limx = tl.where(sgx > 0, NX - c, tl.where(sgx < 0, c + 1, _BIG_INT_C))
        limy = tl.where(sgy > 0, NY - r, tl.where(sgy < 0, r + 1, _BIG_INT_C))
        limz = tl.where(sgz > 0, NZ - s, tl.where(sgz < 0, s + 1, _BIG_INT_C))
        t_ex = tl.minimum(tl.minimum((limx.to(tl.float32) - 0.5) * tdx, (limy.to(tl.float32) - 0.5) * tdy),
                          (limz.to(tl.float32) - 0.5) * tdz)
        t_stop = tl.where(live, tl.minimum(t_ex, t_end), 0.0)
        T = tl.max(t_stop, axis=0)
        nst = (tl.minimum(tl.maximum(tl.ceil(T / tdx - 0.5), 0.0), NX * 1.0)
               + tl.minimum(tl.maximum(tl.ceil(T / tdy - 0.5), 0.0), NY * 1.0)
               + tl.minimum(tl.maximum(tl.ceil(T / tdz - 0.5), 0.0), NZ * 1.0) + 2.0)
        nsteps = tl.where(toward, nst, 0.0).to(tl.int32)
        base = r.to(tl.int64) * sR + c.to(tl.int64) * sC + s.to(tl.int64) * sS
        g = tl.arange(0, GP)
        gm = g < G
        gofs = g.to(tl.int64) * sG
        acc = tl.zeros([GP, BLOCK], dtype=tl.float32)
        nxc = 0
        nyc = 0
        nzc = 0
        t_prev = 0.0
        for _ in range(0, nsteps):
            tx = (nxc + 0.5) * tdx
            ty = (nyc + 0.5) * tdy
            tz = (nzc + 0.5) * tdz
            tn = tl.minimum(tl.minimum(tx, ty), tz)
            seg = tl.maximum(tl.minimum(tn, t_end) - tl.minimum(t_prev, t_end), 0.0)
            inb = live & (nxc < limx) & (nyc < limy) & (nzc < limz)
            shift = (nxc * sgx).to(tl.int64) * sC + (nyc * sgy).to(tl.int64) * sR + (nzc * sgz).to(tl.int64) * sS
            v = tl.load(mu_ptr + (base + shift)[None, :] + gofs[:, None], mask=gm[:, None] & inb[None, :],
                        other=0.0)
            acc += v * seg[None, :]
            ax = (tx <= ty) & (tx <= tz)
            ay = (ty < tx) & (ty <= tz)
            nxc += ax.to(tl.int32)
            nyc += ay.to(tl.int32)
            nzc += ((tz < tx) & (tz < ty)).to(tl.int32)
            t_prev = tn
        norm = tl.sqrt(ox * ox + oy * oy + oz * oz)
        res = tl.where(toward, acc * norm, float("inf"))
        optr = out_ptr + (k.to(tl.int64) * G + g.to(tl.int64))[:, None] * V + m[None, :].to(tl.int64)
        tl.store(optr, res, mask=gm[:, None] & live[None, :])


# ================================================================================================
# wrappers
# ================================================================================================

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _grid_scalars(grid) -> tuple:
    """(lox, loy, loz, hix, hiy, hiz, dx, dy, dz) as Python floats (the reference's box tensors in float64)."""
    lo = grid.lower_corner()
    d = grid.deltas
    n = (grid.num_cols, grid.num_rows, grid.num_slices)
    hi = tuple(lo[a] + d[a] * n[a] for a in range(3))
    return tuple(float(v) for v in (*lo, *hi, *d))


def _check_cuda_f32(t: Tensor, name: str) -> None:
    if t.device.type != "cuda" or t.dtype != torch.float32:
        raise ValueError(f"the Siddon kernels need a float32 CUDA {name}, got {t.dtype} on {t.device}")


def _launch_ray(vol: Tensor, out: Tensor, num_rays: int, num_major: int, mode: int, G: int, grid, *,
                a: Optional[Tensor] = None, b: Optional[Tensor] = None, vidx: Optional[Tensor] = None,
                om: Optional[Tensor] = None, vol_k_stride: int = 0, geo: Optional[dict] = None) -> None:
    """Launch `_ray_kernel` over num_major x ceil(num_rays / BLOCK_RAYS) programs (see its docstring).

    vol: [..., G, R, C, S] view whose last four strides are used (vol_k_stride: per major / volume index);
    geo: mode-specific scalars (q, u, v, pixel parameters, box center)."""
    geo = geo or {}
    sG, sR, sC, sS = (int(x) for x in vol.stride()[-4:])
    dummy = vol
    a_t = a if a is not None else dummy
    b_t = b if b is not None else dummy
    sa = (int(a.stride(0)), int(a.stride(1))) if a is not None else (0, 0)
    sb = (int(b.stride(0)), int(b.stride(1))) if b is not None else (0, 0)
    lox, loy, loz, hix, hiy, hiz, dx, dy, dz = _grid_scalars(grid)
    q = geo.get("q", (0.0, 0.0, 0.0))
    uh = geo.get("u", (0.0, 0.0, 0.0))
    vh = geo.get("v", (0.0, 0.0, 0.0))
    pixp = geo.get("pix", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))  # rmid, cmid, drow, dch, rowoff, choff
    bc = geo.get("bc", (0.0, 0.0, 0.0))
    args = (vol, out, a_t, b_t, vidx if vidx is not None else dummy, om if om is not None else dummy,
            int(num_rays), *sa, *sb, int(vol_k_stride), sG, sR, sC, sS,
            int(grid.num_cols), int(grid.num_rows), int(grid.num_slices), int(geo.get("Rd", 1)),
            int(geo.get("Cd", 1)),
            lox, loy, loz, hix, hiy, hiz, dx, dy, dz, float(grid.recon_slice_offset),
            *(float(x) for x in q), *(float(x) for x in uh), *(float(x) for x in vh),
            *(float(x) for x in pixp), *(float(x) for x in bc), float(geo.get("half", 0.0)))
    cx = dict(MODE=int(mode), G=int(G), GP=_next_pow2(G), HAS_VIDX=vidx is not None, BLOCK=BLOCK_RAYS)
    _tu.launch(_ray_kernel, (triton.cdiv(max(num_rays, 1), BLOCK_RAYS), int(num_major), 1), args, cx, NUM_WARPS)


def line_integrals_triton(volume: Tensor, starts: Tensor, ends: Tensor, grid, *,
                          volume_index: Optional[Tensor] = None) -> Tensor:
    """Fused `ray_tracing.line_integrals`: exact line integrals of a voxel field along segments starts -> ends.

    volume: [G, rows, cols, slices], or [K, G, rows, cols, slices] with volume_index [N] (ray n integrates
    volume[volume_index[n]]); float32 CUDA, any strides (no copy). starts, ends: [N, 3] (x, y, z) recon frame cm,
    any strides (a stride-0 row broadcasts one point). Returns [G, N] float32 (volume x cm); the volume is zero
    outside the grid box.
    """
    _check_cuda_f32(volume, "volume")
    dev = volume.device
    if volume.dim() == 5:
        if volume_index is None:
            raise ValueError("a [K, G, rows, cols, slices] volume needs volume_index [N]")
        K, G = volume.shape[:2]
        if tuple(volume.shape[2:]) != tuple(grid.shape):
            raise ValueError(f"volume spatial shape {tuple(volume.shape[2:])} != grid shape {grid.shape}")
        ks = int(volume.stride(0))
        vidx = volume_index.to(device=dev, dtype=torch.int64).reshape(-1)
    elif volume.dim() == 4:
        G = volume.shape[0]
        if tuple(volume.shape[1:]) != tuple(grid.shape):
            raise ValueError(f"volume spatial shape {tuple(volume.shape[1:])} != grid shape {grid.shape}")
        ks = 0
        vidx = None
    else:
        raise ValueError(f"volume must be [G, rows, cols, slices], got {tuple(volume.shape)}")
    a = starts.to(device=dev, dtype=torch.float32)
    b = ends.to(device=dev, dtype=torch.float32)
    if a.dim() != 2:
        a = a.reshape(-1, 3)
    if b.dim() != 2:
        b = b.reshape(-1, 3)
    if a.shape != b.shape or a.shape[-1] != 3:
        raise ValueError("starts and ends must both be [N, 3]")
    N = a.shape[0]
    if vidx is not None and vidx.shape[0] != N:
        raise ValueError("volume_index must have one entry per ray")
    out = torch.empty(G, N, device=dev, dtype=torch.float32)
    if N == 0:
        return out
    _launch_ray(volume, out, N, 1, MODE_GENERIC, G, grid, a=a, b=b, vidx=vidx, vol_k_stride=ks)
    return out


def _pixel_geo(geom, angle: float) -> dict:
    """Scalars of ConeGeometry.pixel_centers: panel center, axes and the pixel-coordinate parameters."""
    u_hat, v_hat, _ = geom.panel_axes(angle)
    return dict(q=geom.panel_center(angle), u=u_hat, v=v_hat, Rd=geom.num_det_rows,
                Cd=geom.num_det_channels,
                pix=((geom.num_det_rows - 1) / 2, (geom.num_det_channels - 1) / 2, geom.delta_det_row,
                     geom.delta_det_channel, geom.det_row_offset, geom.det_channel_offset))


def upstream_cumulative(mu: Tensor, grid, geom, angle: float) -> Tensor:
    """Fused `SiddonProjectors.upstream_cumulative`: line integral of mu from the focal spot to every voxel center.

    mu: [G, rows, cols, slices] float32 CUDA (any strides). Returns tau_up [G, rows, cols, slices]."""
    _check_cuda_f32(mu, "mu")
    G = mu.shape[0]
    V = grid.num_voxels
    out = torch.empty(G, *grid.shape, device=mu.device, dtype=torch.float32)
    _launch_ray(mu, out, V, 1, MODE_UPSTREAM, G, grid, geo=dict(q=geom.source_position(angle)))
    return out


def primary_project(mu: Tensor, grid, geom, angle: float, n_sub: int = 1) -> Tensor:
    """Fused `SiddonProjectors.primary_project`: line integral of mu from the focal spot to every pixel center.

    mu: [G, rows, cols, slices] float32 CUDA. Returns [G, num_det_rows, num_det_channels]. n_sub > 1: n_sub^2
    sub-rays per pixel on the reference's midpoint lattice, combined as -log(mean(exp(-integral)))."""
    _check_cuda_f32(mu, "mu")
    dev, dt = mu.device, mu.dtype
    G = mu.shape[0]
    Rd, Cd = geom.num_det_rows, geom.num_det_channels
    src = geom.source_position(angle)
    if n_sub == 1:
        out = torch.empty(G, Rd, Cd, device=dev, dtype=dt)
        geo = _pixel_geo(geom, angle)
        geo["bc"] = src  # MODE_PRIMARY reads the focal spot from the box-center slots
        _launch_ray(mu, out, Rd * Cd, 1, MODE_PRIMARY, G, grid, geo=geo)
        return out
    pix = geom.pixel_centers(angle, dev, dt)
    u_hat, v_hat, _ = geom.panel_axes(angle)
    uh = torch.tensor(u_hat, device=dev, dtype=dt)
    vh = torch.tensor(v_hat, device=dev, dtype=dt)
    off = (2 * torch.arange(n_sub, device=dev, dtype=dt) + 1) / (2 * n_sub) - 0.5
    shift = (off * geom.delta_det_row)[:, None, None] * vh + (off * geom.delta_det_channel)[None, :, None] * uh
    ends = (pix[:, :, None, None, :] + shift).reshape(-1, 3)
    starts = torch.tensor(src, device=dev, dtype=dt).expand_as(ends)
    tau = line_integrals_triton(mu, starts, ends, grid).reshape(G, Rd, Cd, n_sub * n_sub)
    return -(torch.logsumexp(-tau, dim=-1) - math.log(n_sub * n_sub))


def toward_flags(omegas: Tensor, geom, angle: float) -> Tensor:
    """[K] uint8: omega_k . (-n_hat) > 0, evaluated with the reference's torch expression (bit-identical)."""
    _, _, n_hat = geom.panel_axes(angle)
    n = torch.tensor(n_hat, device=omegas.device, dtype=omegas.dtype)
    return ((omegas * (-n)).sum(-1) > 0).to(torch.uint8)


def downstream_cumulative(mu: Tensor, grid, geom, angle: float, omegas: Tensor,
                          toward: Optional[Tensor] = None) -> Tensor:
    """Fused `SiddonProjectors.downstream_cumulative`: line integral of mu from every voxel center to the panel
    plane along each direction (+inf where omega . (-n_hat) <= 0).

    mu: [G, rows, cols, slices] float32 CUDA (any strides); omegas: [K, 3]. Returns [K, G, rows, cols, slices].
    One `_down_shared_kernel` launch (one crossing sequence per direction). toward: optional precomputed
    `toward_flags(omegas, geom, angle)`.
    """
    _check_cuda_f32(mu, "mu")
    dev = mu.device
    G = mu.shape[0]
    if tuple(mu.shape[1:]) != tuple(grid.shape):
        raise ValueError(f"mu spatial shape {tuple(mu.shape[1:])} != grid shape {grid.shape}")
    om = omegas.to(device=dev, dtype=torch.float32).reshape(-1, 3).contiguous()
    K = om.shape[0]
    V = grid.num_voxels
    out = torch.empty(K, G, *grid.shape, device=dev, dtype=torch.float32)
    if K == 0 or V == 0:
        return out
    tw = toward_flags(om, geom, angle) if toward is None else toward
    _, _, n_hat = geom.panel_axes(angle)
    pc = geom.panel_center(angle)
    diag = math.sqrt(sum(e * e for e in grid.extent)) * (1 + 1e-9) + 1e-9
    sG, sR, sC, sS = (int(x) for x in mu.stride())
    dx, dy, dz = (float(d) for d in grid.deltas)
    args = (mu, om, tw, out, int(V), sG, sR, sC, sS, int(grid.num_cols), int(grid.num_rows), int(grid.num_slices),
            dx, dy, dz, float(grid.recon_slice_offset), *(float(x) for x in pc), *(float(x) for x in n_hat),
            float(diag))
    _tu.launch(_down_shared_kernel, (triton.cdiv(V, BLOCK_DOWN), K, 1), args,
                dict(G=int(G), GP=_next_pow2(G), BLOCK=BLOCK_DOWN), NUM_WARPS)
    return out


def oblique_project(field: Tensor, grid, geom, angle: float, omegas: Tensor) -> Tensor:
    """Fused `SiddonProjectors.oblique_project`: full line integral of a field along the line through each pixel
    center with direction omega_k (clipped to the box).

    field: [K, G, rows, cols, slices] or [G, rows, cols, slices] (shared over K), float32 CUDA, any strides;
    omegas: [K, 3]. Returns [K, G, num_det_rows, num_det_channels] (field x cm).
    """
    _check_cuda_f32(field, "field")
    dev = field.device
    om = omegas.to(device=dev, dtype=torch.float32).reshape(-1, 3).contiguous()
    K = om.shape[0]
    if field.dim() == 4:
        G = field.shape[0]
        if tuple(field.shape[1:]) != tuple(grid.shape):
            raise ValueError(f"field spatial shape {tuple(field.shape[1:])} != grid shape {grid.shape}")
        ks = 0
    elif field.dim() == 5 and field.shape[0] == K:
        G = field.shape[1]
        if tuple(field.shape[2:]) != tuple(grid.shape):
            raise ValueError(f"field spatial shape {tuple(field.shape[2:])} != grid shape {grid.shape}")
        ks = int(field.stride(0))
    else:
        raise ValueError(f"field must be [G, ...] or [K={K}, G, ...], got {tuple(field.shape)}")
    Rd, Cd = geom.num_det_rows, geom.num_det_channels
    out = torch.empty(K, G, Rd, Cd, device=dev, dtype=torch.float32)
    if K == 0:
        return out
    lo = grid.lower_corner()
    ext = grid.extent
    geo = _pixel_geo(geom, angle)
    geo["bc"] = tuple(lo[a] + 0.5 * ext[a] for a in range(3))
    geo["half"] = 0.5 * math.sqrt(sum(e * e for e in ext)) * (1 + 1e-6) + 1e-6
    _launch_ray(field, out, Rd * Cd, K, MODE_OBLIQUE, G, grid, om=om, vol_k_stride=ks, geo=geo)
    return out


def warmup(device: torch.device, num_groups: int) -> None:
    """Compile (or load from Triton's disk cache) every kernel variant used by SiddonProjectors for `num_groups`
    groups on `device`, on a tiny problem, so that the first timed call does not pay the compilation."""
    from ..geometry import ConeGeometry, Grid

    grid = Grid(shape=(3, 4, 5), delta_voxel=1.0)
    geom = ConeGeometry(source_iso_dist=20.0, source_detector_dist=30.0, num_det_rows=4, num_det_channels=5,
                        delta_det_row=1.0, delta_det_channel=1.0)
    mu = torch.rand(num_groups, *grid.shape, device=device)
    om = torch.tensor([[0.1, -1.0, 0.2], [0.0, -1.0, 0.0]], device=device)
    upstream_cumulative(mu, grid, geom, 0.0)
    primary_project(mu, grid, geom, 0.0)
    primary_project(mu, grid, geom, 0.0, n_sub=2)
    downstream_cumulative(mu, grid, geom, 0.0, om)
    oblique_project(mu, grid, geom, 0.0, om)
    oblique_project(mu[None].expand(2, -1, -1, -1, -1), grid, geom, 0.0, om)
    torch.cuda.synchronize(device)
