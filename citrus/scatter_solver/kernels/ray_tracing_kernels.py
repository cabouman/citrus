"""Fused Triton kernels of the exact Siddon ray tracer of :mod:`citrus.scatter_solver.ray_tracing`.

``_ray_kernel`` traces one ray per lane in registers and accumulates all groups at once; its ``MODE`` selects the
segment end points (given segments, oblique lines through the pixel centers, focal spot to voxel centers, focal spot
to pixel centers) and ``PARALLEL`` replaces the focal spot by the upstream start of a parallel beam's ray.
``_downstream_shared_kernel`` computes the downstream optical depth with one crossing sequence shared by all rays of a
direction. The public wrappers mirror the ``SiddonProjectors`` operations on float32 CUDA
tensors, and ``kernels_usable`` checks them against the PyTorch path once per device.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from . import triton_utilities

SELF_CHECK_REL_TOL = 1e-5
BLOCK_RAYS = 128  # rays per program of `_ray_kernel` (tile [GP, 128], 4 warps)
BLOCK_DOWN = 128  # voxels per program of `_downstream_shared_kernel`
NUM_WARPS = 4
BIG = 3.0e38  # "no crossing" parameter (finite, so that products stay finite)
_BIG_INT = 1 << 30

_SELF_CHECK: dict[str, tuple[bool, str]] = {}

triton = triton_utilities.triton
tl = triton_utilities.tl

MODE_GENERIC, MODE_OBLIQUE, MODE_UPSTREAM, MODE_PRIMARY = 0, 1, 2, 3


# ================================================================================================
# availability
# ================================================================================================

def triton_available() -> tuple[bool, str]:
    """Report whether Triton kernels can run on this machine (the shared probe of ``triton_utilities``).

    Returns:
        (bool, str): whether Triton is usable, and the reason.
    """
    return triton_utilities.triton_available()


def kernels_usable(device: torch.device) -> tuple[bool, str]:
    """Report whether Triton is available and every kernel of this module reproduces the PyTorch path on ``device``.

    The self-check runs once per device on a tiny random problem; the verdict is cached and the function never raises.

    Args:
        device (torch.device): the CUDA device to check.

    Returns:
        (bool, str): whether the kernels are usable, and the reason.
    """
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
    """Decide whether a ``SiddonProjectors`` operation on ``t`` runs the kernels; float64 and CPU tensors never do.

    Args:
        t (Tensor): the input tensor of the operation.
        backend (str): "auto", "torch" or "triton".

    Returns:
        bool: True for a float32 CUDA tensor on a device where the kernels are usable, unless ``backend`` is "torch".

    Raises:
        ValueError: for an unknown ``backend``.
        RuntimeError: for backend "triton" when the kernels are unusable on a float32 CUDA tensor.
    """
    if backend not in ("auto", "torch", "triton"):
        raise ValueError("backend must be 'auto', 'torch' or 'triton'")
    if backend == "torch" or t.device.type != "cuda" or t.dtype != torch.float32:
        return False
    ok, why = kernels_usable(t.device)
    if not ok and backend == "triton":
        raise RuntimeError(f"backend='triton' requested but the Siddon kernels are unusable: {why}")
    return ok


def _self_check(device: torch.device) -> tuple[bool, str]:
    """Compare every kernel operation with the PyTorch Siddon path on a tiny random problem.

    The pixel operations run on a flat panel, on a curved panel with a helical z shift and on a parallel beam.

    Args:
        device (torch.device): the CUDA device to check.

    Returns:
        (bool, str): whether the largest relative deviation is within ``SELF_CHECK_REL_TOL``, and a message.
    """
    from ..geometry import ConeGeometry, Grid, ParallelGeometry
    from ..ray_tracing import SiddonProjectors, line_integrals

    grid = Grid(shape=(5, 7, 6), delta_voxel=1.0, voxel_row_aspect=1.25, voxel_slice_aspect=0.8,
                recon_slice_offset=0.2)
    geometry = ConeGeometry(source_iso_dist=30.0, source_detector_dist=45.0, num_det_rows=9, num_det_channels=11,
                            delta_det_row=1.1, delta_det_channel=1.3)
    angle = 0.3
    generator = torch.Generator().manual_seed(9)
    mu = (0.3 * torch.rand(2, *grid.shape, generator=generator)).to(device)
    directions = torch.tensor([[0.2, -1.0, 0.1], [-0.3, -1.0, 0.25], [0.0, -1.0, 0.0], [0.1, 1.0, 0.0]])
    directions = geometry.view_to_recon(torch.nn.functional.normalize(directions, dim=-1), angle).to(device)
    K = directions.shape[0]
    field = torch.rand(K, 2, *grid.shape, generator=generator).to(device)
    lo = torch.tensor(grid.lower_corner())
    extent = torch.tensor(grid.extent)
    a = (lo + (torch.rand(64, 3, generator=generator) * 1.6 - 0.3) * extent).to(device)
    b = (lo + (torch.rand(64, 3, generator=generator) * 1.6 - 0.3) * extent).to(device)
    volume_index = torch.randint(0, K, (64,), generator=generator).to(device)
    reference = SiddonProjectors(backend="torch")
    # a curved panel close to the source (its cylinder cuts the grid box) with a helical z shift
    curved = ConeGeometry(source_iso_dist=6.0, source_detector_dist=9.0, num_det_rows=9, num_det_channels=11,
                          delta_det_row=1.1, delta_det_channel=1.3, det_channel_offset=0.4, use_curved_detector=True,
                          helical_view_angles=(0.0, 1.0), helical_z_shifts=(-0.5, 1.5))
    # a parallel beam whose panel cuts the grid box
    parallel = ParallelGeometry(axis_detector_dist=2.5, num_det_rows=9, num_det_channels=11, delta_det_row=0.9,
                                delta_det_channel=1.1, det_row_offset=-0.3, det_channel_offset=0.4)
    pairs = [(line_integrals(field, a, b, grid, volume_index=volume_index),
              line_integrals_triton(field, a, b, grid, volume_index=volume_index))]
    for g in (geometry, curved, parallel):
        pairs += [
            (reference.upstream_cumulative(mu, grid, g, angle), upstream_cumulative(mu, grid, g, angle)),
            (reference.primary_project(mu, grid, g, angle), primary_project(mu, grid, g, angle)),
            (reference.primary_project(mu, grid, g, angle, samples_per_pixel_axis=2),
             primary_project(mu, grid, g, angle, samples_per_pixel_axis=2)),
            (reference.downstream_cumulative(mu, grid, g, angle, directions),
             downstream_cumulative(mu, grid, g, angle, directions)),
            (reference.oblique_project(field[0], grid, g, angle, directions),
             oblique_project(field[0], grid, g, angle, directions)),
            (reference.oblique_project(field, grid, g, angle, directions),
             oblique_project(field, grid, g, angle, directions)),
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
    _RAY_PTRS = ["volume_ptr", "out_ptr", "a_ptr", "b_ptr", "volume_index_ptr", "direction_ptr"]

    @triton.jit
    def _plane_param(lo, delta, idx, dpos, p0, inv, nz):
        """Segment parameter of the next plane crossing along one axis from voxel index ``idx``.

        Args:
            lo: lower corner of the grid box along the axis in cm.
            delta: voxel size along the axis in cm.
            idx: [BLOCK] int32 current voxel index along the axis.
            dpos: [BLOCK] bool, the ray ascends along the axis.
            p0: [BLOCK] segment start coordinate along the axis in cm.
            inv: [BLOCK] reciprocal of the segment extent along the axis (1 where it is zero).
            nz: [BLOCK] bool, the segment extent along the axis is nonzero.

        Returns:
            [BLOCK] parameter of the next crossing, ``BIG`` where the axis is constant along the ray.
        """
        k = idx + dpos.to(tl.int32)
        return tl.where(nz, (lo + k.to(tl.float32) * delta - p0) * inv, _BIG_C)

    @triton.jit
    def _entry_index(p0, d, a, lo, delta, n):
        """Voxel index along one axis at segment parameter ``a`` (floor; ceil - 1 when descending), clamped to the grid.

        Args:
            p0: [BLOCK] segment start coordinate along the axis in cm.
            d: [BLOCK] segment extent ``p1 - p0`` along the axis in cm.
            a: [BLOCK] segment parameter of the entry point.
            lo: lower corner of the grid box along the axis in cm.
            delta: voxel size along the axis in cm.
            n: number of voxels along the axis.

        Returns:
            [BLOCK] int32 voxel index in [0, n - 1].
        """
        f = (p0 + a * d - lo) / delta
        i = tl.where(d < 0, tl.ceil(f) - 1.0, tl.floor(f))
        return tl.minimum(tl.maximum(i, 0.0), (n - 1).to(tl.float32)).to(tl.int32)

    @triton.jit
    def _trace(volume_ptr, vbase, p0x, p0y, p0z, p1x, p1y, p1z, live, sG, sR, sC, sS,
               lox, loy, loz, hix, hiy, hiz, ddx, ddy, ddz, NX, NY, NZ,
               G: tl.constexpr, GP: tl.constexpr, BLOCK: tl.constexpr):
        """Exact integral of the voxel field along ``p0 -> p1`` for a block of rays, all groups at once.

        Args:
            volume_ptr: float32 voxel field, addressed with the strides below.
            vbase: [BLOCK] int64 element offset of each ray's volume.
            p0x: [BLOCK] x of the segment starts in cm.
            p0y: [BLOCK] y of the segment starts in cm.
            p0z: [BLOCK] z of the segment starts in cm.
            p1x: [BLOCK] x of the segment ends in cm.
            p1y: [BLOCK] y of the segment ends in cm.
            p1z: [BLOCK] z of the segment ends in cm.
            live: [BLOCK] bool, the lane holds a ray.
            sG: element stride of the group axis.
            sR: element stride of the row (y) axis.
            sC: element stride of the column (x) axis.
            sS: element stride of the slice (z) axis.
            lox: x of the lower box corner in cm.
            loy: y of the lower box corner in cm.
            loz: z of the lower box corner in cm.
            hix: x of the upper box corner in cm.
            hiy: y of the upper box corner in cm.
            hiz: z of the upper box corner in cm.
            ddx: voxel size along x in cm.
            ddy: voxel size along y in cm.
            ddz: voxel size along z in cm.
            NX: number of columns (voxels along x).
            NY: number of rows (voxels along y).
            NZ: number of slices (voxels along z).
            G (constexpr): number of energy groups.
            GP (constexpr): next power of two >= G.
            BLOCK (constexpr): rays per program.

        Returns:
            [GP, BLOCK] float32 line integrals in units of the field times cm; rows g >= G are zero.
        """
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
            v = tl.load(volume_ptr + (vbase + sp)[None, :] + gofs[:, None], mask=lm, other=0.0)
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
    def _ray_kernel(volume_ptr, out_ptr, a_ptr, b_ptr, volume_index_ptr, direction_ptr,
                    M, sa_n, sa_c, sb_n, sb_c, sK, sG, sR, sC, sS, NX, NY, NZ, Rd, Cd,
                    lox, loy, loz, hix, hiy, hiz, ddx, ddy, ddz, zoff,
                    qx, qy, qz, ux_, uy_, uz_, vx_, vy_, vz_,
                    rmid, cmid, drow, dch, rowoff, choff, bcx, bcy, bcz, half, nx_, ny_, nz_, rad,
                    wx, wy, wz, wdist,
                    MODE: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, HAS_VOLUME_INDEX: tl.constexpr,
                    CURVED: tl.constexpr, PARALLEL: tl.constexpr, BLOCK: tl.constexpr):
        """Trace one block of rays (program_id 0) of direction or volume index k (program_id 1) into ``out``.

        Writes ``out[(k G + g) M + m]``, with m the segment (MODE 0), the pixel ``r * Cd + c`` (MODE 1, 3) or the
        flat voxel index (MODE 2).

        Args:
            volume_ptr: float32 voxel field [G, rows, cols, slices] per volume, any strides.
            out_ptr: [K, G, M] float32 line integrals.
            a_ptr: [M, 3] float32 segment starts in cm (MODE 0).
            b_ptr: [M, 3] float32 segment ends in cm (MODE 0).
            volume_index_ptr: [M] int64 volume index per ray (MODE 0 with HAS_VOLUME_INDEX).
            direction_ptr: [K, 3] float32 directions (MODE 1).
            M: rays per direction or volume index.
            sa_n: element stride between the rows of ``a_ptr``.
            sa_c: element stride between the coordinates of ``a_ptr``.
            sb_n: element stride between the rows of ``b_ptr``.
            sb_c: element stride between the coordinates of ``b_ptr``.
            sK: element stride between volumes (0 for one shared volume).
            sG: element stride of the group axis.
            sR: element stride of the row (y) axis.
            sC: element stride of the column (x) axis.
            sS: element stride of the slice (z) axis.
            NX: number of columns (voxels along x).
            NY: number of rows (voxels along y).
            NZ: number of slices (voxels along z).
            Rd: number of detector rows.
            Cd: number of detector channels.
            lox: x of the lower box corner in cm.
            loy: y of the lower box corner in cm.
            loz: z of the lower box corner in cm.
            hix: x of the upper box corner in cm.
            hiy: y of the upper box corner in cm.
            hiz: z of the upper box corner in cm.
            ddx: voxel size along x in cm.
            ddy: voxel size along y in cm.
            ddz: voxel size along z in cm.
            zoff: slice offset of the grid (``recon_slice_offset``) in cm.
            qx: x of the panel center (MODE 1, 3) or of the focal spot (MODE 2) in cm.
            qy: y of the panel center (MODE 1, 3) or of the focal spot (MODE 2) in cm.
            qz: z of the panel center (MODE 1, 3) or of the focal spot (MODE 2) in cm.
            ux_: x component of the panel axis ``u_hat``.
            uy_: y component of the panel axis ``u_hat``.
            uz_: z component of the panel axis ``u_hat``.
            vx_: x component of the panel axis ``v_hat``.
            vy_: y component of the panel axis ``v_hat``.
            vz_: z component of the panel axis ``v_hat``.
            rmid: center detector row index ``(Rd - 1) / 2``.
            cmid: center detector channel index ``(Cd - 1) / 2``.
            drow: detector row spacing in cm.
            dch: detector channel spacing in cm.
            rowoff: detector row offset in cm.
            choff: detector channel offset in cm.
            bcx: x of the box center (MODE 1) or of the focal spot (MODE 3) in cm.
            bcy: y of the box center (MODE 1) or of the focal spot (MODE 3) in cm.
            bcz: z of the box center (MODE 1) or of the focal spot (MODE 3) in cm.
            half: half length of the oblique segment about the box center in cm (MODE 1).
            nx_: x component of the panel normal ``n_hat`` (CURVED).
            ny_: y component of the panel normal ``n_hat`` (CURVED).
            nz_: z component of the panel normal ``n_hat`` (CURVED).
            rad: radius of the curved panel, the source-detector distance, in cm (CURVED).
            wx: x component of the beam direction (PARALLEL).
            wy: y component of the beam direction (PARALLEL).
            wz: z component of the beam direction (PARALLEL).
            wdist: distance from the rotation axis of the plane where the beam's rays start, in cm (PARALLEL).
            MODE (constexpr): end points: 0 given segments, 1 oblique pixel lines, 2 upstream, 3 primary.
            G (constexpr): number of energy groups.
            GP (constexpr): next power of two >= G.
            HAS_VOLUME_INDEX (constexpr): read ``volume_index_ptr`` (MODE 0).
            CURVED (constexpr): the pixels lie on the cylinder of a curved panel (MODE 1, 3).
            PARALLEL (constexpr): a parallel beam: the rays of MODE 2 and 3 start at ``p - (p . w + wdist) w`` for
                the end point p (``ParallelGeometry.ray_starts``) instead of the focal spot.
            BLOCK (constexpr): rays per program.
        """
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
            if HAS_VOLUME_INDEX:
                vi = tl.load(volume_index_ptr + mm, mask=live, other=0).to(tl.int64)
                vbase = vi * sK
        elif MODE == 1 or MODE == 3:
            # pixel center (r, c) = panel center + u u_hat + v v_hat (ConeGeometry.pixel_centers). Lanes run along
            # detector rows (v ~ z, the contiguous volume axis), so neighbouring rays gather neighbouring slices
            r = mm % Rd
            c = mm // Rd
            o_idx = r * Cd + c
            u = (c.to(tl.float32) - cmid) * dch - choff
            v = (r.to(tl.float32) - rmid) * drow - rowoff
            if CURVED:
                # ConeGeometry.panel_points: R sin(g) u_hat + 2 R sin^2(g / 2) n_hat + v cos(g) v_hat, g = u / R
                gam = u / rad
                sh = tl.sin(0.5 * gam)
                au = rad * tl.sin(gam)
                an = 2.0 * rad * sh * sh
                av = v * tl.cos(gam)
                px = qx + au * ux_ + an * nx_ + av * vx_
                py = qy + au * uy_ + an * ny_ + av * vy_
                pz = qz + au * uz_ + an * nz_ + av * vz_
            else:
                px = qx + u * ux_ + v * vx_
                py = qy + u * uy_ + v * vy_
                pz = qz + u * uz_ + v * vz_
            if MODE == 1:
                ox = tl.load(direction_ptr + 3 * k)
                oy = tl.load(direction_ptr + 3 * k + 1)
                oz = tl.load(direction_ptr + 3 * k + 2)
                om2 = ox * ox + oy * oy + oz * oz
                tc = ((bcx - px) * ox + (bcy - py) * oy + (bcz - pz) * oz) / om2
                h = half / tl.sqrt(om2)
                p0x = px + (tc - h) * ox
                p0y = py + (tc - h) * oy
                p0z = pz + (tc - h) * oz
                p1x = px + (tc + h) * ox
                p1y = py + (tc + h) * oy
                p1z = pz + (tc + h) * oz
            elif PARALLEL:
                along = px * wx + py * wy + pz * wz + wdist  # the beam's ray through the pixel center
                p0x = px - along * wx
                p0y = py - along * wy
                p0z = pz - along * wz
                p1x = px
                p1y = py
                p1z = pz
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
            if PARALLEL:
                along = cx * wx + cy * wy + cz * wz + wdist  # the beam's ray through the voxel center
                p0x = cx - along * wx
                p0y = cy - along * wy
                p0z = cz - along * wz
            else:
                p0x = tl.zeros([BLOCK], dtype=tl.float32) + qx
                p0y = tl.zeros([BLOCK], dtype=tl.float32) + qy
                p0z = tl.zeros([BLOCK], dtype=tl.float32) + qz
            p1x = cx
            p1y = cy
            p1z = cz
        acc = _trace(volume_ptr, vbase, p0x, p0y, p0z, p1x, p1y, p1z, live, sG, sR, sC, sS,
                     lox, loy, loz, hix, hiy, hiz, ddx, ddy, ddz, NX, NY, NZ, G, GP, BLOCK)
        g = tl.arange(0, GP)
        if MODE == 1 or MODE == 3:
            m = o_idx
        optr = out_ptr + (kk * G + g.to(tl.int64))[:, None] * M + m[None, :].to(tl.int64)
        tl.store(optr, acc, mask=(g < G)[:, None] & live[None, :])

    _DOWN_INTS = ["V", "sG", "sR", "sC", "sS", "NX", "NY", "NZ"]

    @triton.jit(do_not_specialize=_DOWN_INTS,
                do_not_specialize_on_alignment=["mu_ptr", "direction_ptr", "toward_ptr", "out_ptr"])
    def _downstream_shared_kernel(mu_ptr, direction_ptr, toward_ptr, out_ptr, V, sG, sR, sC, sS, NX, NY, NZ,
                                  ddx, ddy, ddz, zoff, qx, qy, qz, nx_, ny_, nz_, diag, sx, sy, rad,
                                  G: tl.constexpr, GP: tl.constexpr, CURVED: tl.constexpr, BLOCK: tl.constexpr):
        """Downstream optical depth of BLOCK consecutive voxels (program_id 0) along direction k (program_id 1).

        All rays of the direction share one crossing sequence (docs/source/design_notes.rst, "Ray tracing and
        kernels"); writes ``out[(k G + g) V + v]``, +inf for a direction that does not head toward the panel.

        Args:
            mu_ptr: [G, rows, cols, slices] float32 attenuation coefficient in cm^-1, any strides.
            direction_ptr: [K, 3] float32 directions.
            toward_ptr: [K] uint8 flags, nonzero for a direction toward the panel.
            out_ptr: [K, G, V] float32 downstream optical depth.
            V: number of voxels.
            sG: element stride of the group axis.
            sR: element stride of the row (y) axis.
            sC: element stride of the column (x) axis.
            sS: element stride of the slice (z) axis.
            NX: number of columns (voxels along x).
            NY: number of rows (voxels along y).
            NZ: number of slices (voxels along z).
            ddx: voxel size along x in cm.
            ddy: voxel size along y in cm.
            ddz: voxel size along z in cm.
            zoff: slice offset of the grid (``recon_slice_offset``) in cm.
            qx: x of the panel center in cm.
            qy: y of the panel center in cm.
            qz: z of the panel center in cm.
            nx_: x component of the panel normal ``n_hat``.
            ny_: y component of the panel normal ``n_hat``.
            nz_: z component of the panel normal ``n_hat``.
            diag: upper bound of the path length (the box diagonal) in cm.
            sx: x of the focal spot in cm (CURVED).
            sy: y of the focal spot in cm (CURVED).
            rad: radius of the curved panel, the source-detector distance, in cm (CURVED).
            G (constexpr): number of energy groups.
            GP (constexpr): next power of two >= G.
            CURVED (constexpr): the panel is the cylinder of radius ``rad`` about the vertical axis through the focal
                spot (``ConeGeometry.panel_distance_along``).
            BLOCK (constexpr): voxels per program.
        """
        k = tl.program_id(1)
        m = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = m < V
        mm = tl.where(live, m, 0)
        ox = tl.load(direction_ptr + 3 * k)
        oy = tl.load(direction_ptr + 3 * k + 1)
        oz = tl.load(direction_ptr + 3 * k + 2)
        toward = tl.load(toward_ptr + k) != 0
        s = mm % NZ
        c = (mm // NZ) % NX
        r = mm // (NZ * NX)
        cx = ddx * (c.to(tl.float32) - (NX - 1) * 0.5)
        cy = ddy * (r.to(tl.float32) - (NY - 1) * 0.5)
        cz = ddz * (s.to(tl.float32) - (NZ - 1) * 0.5) + zoff
        if CURVED:
            # larger root of |h(c + t omega - source)| = rad (h: the x, y part), as ConeGeometry computes it
            hx = cx - sx
            hy = cy - sy
            qa = ox * ox + oy * oy
            qb = hx * ox + hy * oy
            qc = hx * hx + hy * hy - rad * rad
            disc = qb * qb - qa * qc
            root = tl.sqrt(tl.maximum(disc, 0.0))
            pos = qb >= 0.0
            qden = tl.where(pos, qb + root, tl.where(qa > 0.0, qa, 1.0))
            qnum = tl.where(pos, -qc, root - qb)
            ok = (qa > 0.0) & (disc >= 0.0) & (qden != 0.0)
            t_end = tl.where(ok, qnum / tl.where(qden != 0.0, qden, 1.0), diag)
        else:
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
    """Return the smallest power of two >= ``n``.

    Args:
        n (int): a positive integer.

    Returns:
        int: the smallest power of two >= ``n`` (1 for ``n <= 1``).
    """
    p = 1
    while p < n:
        p *= 2
    return p


def _grid_scalars(grid) -> tuple:
    """Return the box corners and voxel sizes of a grid as Python floats, computed like the PyTorch reference.

    Args:
        grid (Grid): the grid.

    Returns:
        tuple: (lox, loy, loz, hix, hiy, hiz, dx, dy, dz), the lower and upper corners and voxel sizes in cm.
    """
    lo = grid.lower_corner()
    d = grid.deltas
    n = (grid.num_cols, grid.num_rows, grid.num_slices)
    hi = tuple(lo[a] + d[a] * n[a] for a in range(3))
    return tuple(float(v) for v in (*lo, *hi, *d))


def _check_cuda_f32(t: Tensor, name: str) -> None:
    """Raise a ValueError unless ``t`` is a float32 CUDA tensor.

    Args:
        t (Tensor): the tensor to check.
        name (str): its name in the error message.
    """
    if t.device.type != "cuda" or t.dtype != torch.float32:
        raise ValueError(f"the Siddon kernels need a float32 CUDA {name}, got {t.dtype} on {t.device}")


def _launch_ray(volume: Tensor, out: Tensor, num_rays: int, num_major: int, mode: int, G: int, grid, *,
                a: Optional[Tensor] = None, b: Optional[Tensor] = None, volume_index: Optional[Tensor] = None,
                directions: Optional[Tensor] = None, volume_stride: int = 0,
                geometry_scalars: Optional[dict] = None) -> None:
    """Launch ``_ray_kernel`` over ``num_major`` x ceil(``num_rays`` / ``BLOCK_RAYS``) programs.

    Args:
        volume (Tensor): [..., G, rows, cols, slices] float32 CUDA field; its last four strides are used.
        out (Tensor): [num_major, G, num_rays] float32 output, written in place.
        num_rays (int): rays per direction or volume index.
        num_major (int): number of directions or volume indices (program_id 1).
        mode (int): the kernel's ``MODE`` (``MODE_GENERIC``, ``MODE_OBLIQUE``, ``MODE_UPSTREAM``, ``MODE_PRIMARY``).
        G (int): number of energy groups.
        grid (Grid): the grid of ``volume``.
        a (Tensor, optional): [N, 3] segment starts in cm (MODE 0). Defaults to None.
        b (Tensor, optional): [N, 3] segment ends in cm (MODE 0). Defaults to None.
        volume_index (Tensor, optional): [N] int64 volume index per ray (MODE 0). Defaults to None.
        directions (Tensor, optional): [K, 3] float32 contiguous directions (MODE 1). Defaults to None.
        volume_stride (int, optional): element stride between volumes. Defaults to 0.
        geometry_scalars (dict, optional): mode-specific scalars with keys "q", "u", "v", "Rd", "Cd", "pixel",
            "box_center", "half", "n", "radius", "curved", "beam", "upstream_distance" and "parallel"; missing keys
            are zero (1 for "Rd", "Cd", False for "curved" and "parallel"). Defaults to None.
    """
    geometry_scalars = geometry_scalars or {}
    sG, sR, sC, sS = (int(x) for x in volume.stride()[-4:])
    dummy = volume
    a_t = a if a is not None else dummy
    b_t = b if b is not None else dummy
    sa = (int(a.stride(0)), int(a.stride(1))) if a is not None else (0, 0)
    sb = (int(b.stride(0)), int(b.stride(1))) if b is not None else (0, 0)
    lox, loy, loz, hix, hiy, hiz, dx, dy, dz = _grid_scalars(grid)
    q = geometry_scalars.get("q", (0.0, 0.0, 0.0))
    u_axis = geometry_scalars.get("u", (0.0, 0.0, 0.0))
    v_axis = geometry_scalars.get("v", (0.0, 0.0, 0.0))
    # rmid, cmid, drow, dch, rowoff, choff
    pixel_parameters = geometry_scalars.get("pixel", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    box_center = geometry_scalars.get("box_center", (0.0, 0.0, 0.0))
    args = (volume, out, a_t, b_t, volume_index if volume_index is not None else dummy,
            directions if directions is not None else dummy,
            int(num_rays), *sa, *sb, int(volume_stride), sG, sR, sC, sS,
            int(grid.num_cols), int(grid.num_rows), int(grid.num_slices), int(geometry_scalars.get("Rd", 1)),
            int(geometry_scalars.get("Cd", 1)),
            lox, loy, loz, hix, hiy, hiz, dx, dy, dz, float(grid.recon_slice_offset),
            *(float(x) for x in q), *(float(x) for x in u_axis), *(float(x) for x in v_axis),
            *(float(x) for x in pixel_parameters), *(float(x) for x in box_center),
            float(geometry_scalars.get("half", 0.0)), *(float(x) for x in geometry_scalars.get("n", (0.0, 0.0, 0.0))),
            float(geometry_scalars.get("radius", 0.0)),
            *(float(x) for x in geometry_scalars.get("beam", (0.0, 0.0, 0.0))),
            float(geometry_scalars.get("upstream_distance", 0.0)))
    constexprs = dict(MODE=int(mode), G=int(G), GP=_next_pow2(G), HAS_VOLUME_INDEX=volume_index is not None,
                      CURVED=bool(geometry_scalars.get("curved", False)),
                      PARALLEL=bool(geometry_scalars.get("parallel", False)), BLOCK=BLOCK_RAYS)
    triton_utilities.launch(_ray_kernel, (triton.cdiv(max(num_rays, 1), BLOCK_RAYS), int(num_major), 1), args,
                            constexprs, NUM_WARPS)


def line_integrals_triton(volume: Tensor, starts: Tensor, ends: Tensor, grid, *,
                          volume_index: Optional[Tensor] = None) -> Tensor:
    """Fused ``ray_tracing.line_integrals``: line integrals of a voxel field along the segments ``starts -> ends``.

    Args:
        volume (Tensor): [G, rows, cols, slices] float32 CUDA field, zero outside the grid box, any strides; or
            [K, G, rows, cols, slices] with ``volume_index``.
        starts (Tensor): [N, 3] segment starts (x, y, z) in the recon frame in cm, any strides.
        ends (Tensor): [N, 3] segment ends (x, y, z) in the recon frame in cm, any strides.
        grid (Grid): the grid of ``volume``.
        volume_index (Tensor, optional): [N] integers in [0, K): ray n integrates ``volume[volume_index[n]]``.
            Defaults to None.

    Returns:
        Tensor: [G, N] float32 line integrals in units of the field times cm.
    """
    _check_cuda_f32(volume, "volume")
    device = volume.device
    if volume.dim() == 5:
        if volume_index is None:
            raise ValueError("a [K, G, rows, cols, slices] volume needs volume_index [N]")
        K, G = volume.shape[:2]
        if tuple(volume.shape[2:]) != tuple(grid.shape):
            raise ValueError(f"volume spatial shape {tuple(volume.shape[2:])} != grid shape {grid.shape}")
        volume_stride = int(volume.stride(0))
        volume_index = volume_index.to(device=device, dtype=torch.int64).reshape(-1)
    elif volume.dim() == 4:
        G = volume.shape[0]
        if tuple(volume.shape[1:]) != tuple(grid.shape):
            raise ValueError(f"volume spatial shape {tuple(volume.shape[1:])} != grid shape {grid.shape}")
        volume_stride = 0
        volume_index = None
    else:
        raise ValueError(f"volume must be [G, rows, cols, slices], got {tuple(volume.shape)}")
    a = starts.to(device=device, dtype=torch.float32)
    b = ends.to(device=device, dtype=torch.float32)
    if a.dim() != 2:
        a = a.reshape(-1, 3)
    if b.dim() != 2:
        b = b.reshape(-1, 3)
    if a.shape != b.shape or a.shape[-1] != 3:
        raise ValueError("starts and ends must both be [N, 3]")
    N = a.shape[0]
    if volume_index is not None and volume_index.shape[0] != N:
        raise ValueError("volume_index must have one entry per ray")
    out = torch.empty(G, N, device=device, dtype=torch.float32)
    if N == 0:
        return out
    _launch_ray(volume, out, N, 1, MODE_GENERIC, G, grid, a=a, b=b, volume_index=volume_index,
                volume_stride=volume_stride)
    return out


def _pixel_geometry_scalars(geometry, angle: float) -> dict:
    """Collect the scalars of ``ConeGeometry.pixel_centers`` for ``_launch_ray``.

    Args:
        geometry (ConeGeometry): the cone-beam geometry.
        angle (float): view angle in radians.

    Returns:
        dict: "q" panel center, "u" and "v" panel axes, "Rd", "Cd" detector size, "pixel" (center row, center
        channel, row spacing, channel spacing, row offset, channel offset) in cm, and for a curved panel "curved"
        True, "n" the panel normal and "radius" the source-detector distance in cm.
    """
    u_hat, v_hat, n_hat = geometry.panel_axes(angle)
    scalars = dict(q=geometry.panel_center(angle), u=u_hat, v=v_hat, Rd=geometry.num_det_rows,
                   Cd=geometry.num_det_channels,
                   pixel=((geometry.num_det_rows - 1) / 2, (geometry.num_det_channels - 1) / 2,
                          geometry.delta_det_row, geometry.delta_det_channel, geometry.det_row_offset,
                          geometry.det_channel_offset))
    if geometry.use_curved_detector:
        scalars.update(curved=True, n=n_hat, radius=geometry.source_detector_dist)
    return scalars


def upstream_cumulative(mu: Tensor, grid, geometry, angle: float) -> Tensor:
    """Fused ``SiddonProjectors.upstream_cumulative``: integral of ``mu`` from the focal spot to every voxel center.

    Args:
        mu (Tensor): [G, rows, cols, slices] float32 CUDA attenuation coefficient in cm^-1, any strides.
        grid (Grid): the grid of ``mu``.
        geometry (ConeGeometry): the cone-beam geometry.
        angle (float): view angle in radians.

    Returns:
        Tensor: [G, rows, cols, slices] upstream optical depth ``tau_up``.
    """
    _check_cuda_f32(mu, "mu")
    G = mu.shape[0]
    V = grid.num_voxels
    out = torch.empty(G, *grid.shape, device=mu.device, dtype=torch.float32)
    if geometry.is_parallel:
        geometry_scalars = _beam_scalars(geometry, angle, grid)
    else:
        geometry_scalars = dict(q=geometry.source_position(angle))
    _launch_ray(mu, out, V, 1, MODE_UPSTREAM, G, grid, geometry_scalars=geometry_scalars)
    return out


def _beam_scalars(geometry, angle: float, grid) -> dict:
    """Collect the scalars of ``ParallelGeometry.ray_starts`` for ``_launch_ray``.

    Args:
        geometry (ParallelGeometry): the parallel-beam geometry.
        angle (float): view angle in radians.
        grid (Grid): the grid the rays are traced through.

    Returns:
        dict: "parallel" True, "beam" the beam direction and "upstream_distance" the distance of the start plane in cm.
    """
    return dict(parallel=True, beam=geometry.beam_direction(angle), upstream_distance=geometry.upstream_distance(grid))


def primary_project(mu: Tensor, grid, geometry, angle: float, samples_per_pixel_axis: int = 1) -> Tensor:
    """Fused ``SiddonProjectors.primary_project``: line integral of ``mu`` from the focal spot to every pixel center.

    With ``n = samples_per_pixel_axis > 1``, ``n * n`` sub-rays on the reference's midpoint lattice are combined as
    ``-log(mean(exp(-tau)))``.

    Args:
        mu (Tensor): [G, rows, cols, slices] float32 CUDA attenuation coefficient in cm^-1.
        grid (Grid): the grid of ``mu``.
        geometry (ConeGeometry): the cone-beam geometry.
        angle (float): view angle in radians.
        samples_per_pixel_axis (int, optional): sub-rays per pixel axis. Defaults to 1.

    Returns:
        Tensor: [G, num_det_rows, num_det_channels] optical depth per pixel.
    """
    _check_cuda_f32(mu, "mu")
    device, dtype = mu.device, mu.dtype
    G = mu.shape[0]
    Rd, Cd = geometry.num_det_rows, geometry.num_det_channels
    if geometry.is_parallel:
        if samples_per_pixel_axis == 1:
            out = torch.empty(G, Rd, Cd, device=device, dtype=dtype)
            geometry_scalars = dict(_pixel_geometry_scalars(geometry, angle), **_beam_scalars(geometry, angle, grid))
            _launch_ray(mu, out, Rd * Cd, 1, MODE_PRIMARY, G, grid, geometry_scalars=geometry_scalars)
            return out
        ends = geometry.subpixel_centers(angle, samples_per_pixel_axis, device, dtype).reshape(-1, 3)
        tau = line_integrals_triton(mu, geometry.ray_starts(ends, angle, grid), ends, grid)
        tau = tau.reshape(G, Rd, Cd, samples_per_pixel_axis ** 2)
        return -(torch.logsumexp(-tau, dim=-1) - math.log(samples_per_pixel_axis * samples_per_pixel_axis))
    source_position = geometry.source_position(angle)
    if samples_per_pixel_axis == 1:
        out = torch.empty(G, Rd, Cd, device=device, dtype=dtype)
        geometry_scalars = _pixel_geometry_scalars(geometry, angle)
        geometry_scalars["box_center"] = source_position  # MODE_PRIMARY reads the focal spot from the box-center slots
        _launch_ray(mu, out, Rd * Cd, 1, MODE_PRIMARY, G, grid, geometry_scalars=geometry_scalars)
        return out
    if geometry.use_curved_detector:  # sub-pixels on the cylinder (the reference's lattice)
        ends = geometry.subpixel_centers(angle, samples_per_pixel_axis, device, dtype).reshape(-1, 3)
        starts = torch.tensor(source_position, device=device, dtype=dtype).expand_as(ends)
        tau = line_integrals_triton(mu, starts, ends, grid).reshape(G, Rd, Cd, samples_per_pixel_axis ** 2)
        return -(torch.logsumexp(-tau, dim=-1) - math.log(samples_per_pixel_axis * samples_per_pixel_axis))
    pixels = geometry.pixel_centers(angle, device, dtype)
    u_hat, v_hat, _ = geometry.panel_axes(angle)
    u_axis = torch.tensor(u_hat, device=device, dtype=dtype)
    v_axis = torch.tensor(v_hat, device=device, dtype=dtype)
    offsets = ((2 * torch.arange(samples_per_pixel_axis, device=device, dtype=dtype) + 1)
               / (2 * samples_per_pixel_axis) - 0.5)
    shift = ((offsets * geometry.delta_det_row)[:, None, None] * v_axis
             + (offsets * geometry.delta_det_channel)[None, :, None] * u_axis)
    ends = (pixels[:, :, None, None, :] + shift).reshape(-1, 3)
    starts = torch.tensor(source_position, device=device, dtype=dtype).expand_as(ends)
    tau = line_integrals_triton(mu, starts, ends, grid).reshape(G, Rd, Cd, samples_per_pixel_axis ** 2)
    return -(torch.logsumexp(-tau, dim=-1) - math.log(samples_per_pixel_axis * samples_per_pixel_axis))


def toward_flags(omegas: Tensor, geometry, angle: float) -> Tensor:
    """Flag the directions that head toward the panel, ``omega_k . (-n_hat) > 0``, as the PyTorch path computes it.

    Args:
        omegas (Tensor): [K, 3] directions (recon frame).
        geometry (ConeGeometry): the cone-beam geometry.
        angle (float): view angle in radians.

    Returns:
        Tensor: [K] uint8 flags, 1 for a direction toward the panel.
    """
    _, _, n_hat = geometry.panel_axes(angle)
    n = torch.tensor(n_hat, device=omegas.device, dtype=omegas.dtype)
    return ((omegas * (-n)).sum(-1) > 0).to(torch.uint8)


def downstream_cumulative(mu: Tensor, grid, geometry, angle: float, omegas: Tensor,
                          toward: Optional[Tensor] = None) -> Tensor:
    """Fused ``SiddonProjectors.downstream_cumulative``: line integral of ``mu`` from every voxel center to the panel.

    Args:
        mu (Tensor): [G, rows, cols, slices] float32 CUDA attenuation coefficient in cm^-1, any strides.
        grid (Grid): the grid of ``mu``.
        geometry (ConeGeometry): the cone-beam geometry.
        angle (float): view angle in radians.
        omegas (Tensor): [K, 3] unit directions (recon frame).
        toward (Tensor, optional): [K] uint8 flags from ``toward_flags``, computed here when None. Defaults to None.

    Returns:
        Tensor: [K, G, rows, cols, slices] downstream optical depth ``tau_down``; +inf where
        ``omega . (-n_hat) <= 0``.
    """
    _check_cuda_f32(mu, "mu")
    device = mu.device
    G = mu.shape[0]
    if tuple(mu.shape[1:]) != tuple(grid.shape):
        raise ValueError(f"mu spatial shape {tuple(mu.shape[1:])} != grid shape {grid.shape}")
    directions = omegas.to(device=device, dtype=torch.float32).reshape(-1, 3).contiguous()
    K = directions.shape[0]
    V = grid.num_voxels
    out = torch.empty(K, G, *grid.shape, device=device, dtype=torch.float32)
    if K == 0 or V == 0:
        return out
    toward = toward_flags(directions, geometry, angle) if toward is None else toward
    _, _, n_hat = geometry.panel_axes(angle)
    panel_center = geometry.panel_center(angle)
    diag = math.sqrt(sum(e * e for e in grid.extent)) * (1 + 1e-9) + 1e-9
    sG, sR, sC, sS = (int(x) for x in mu.stride())
    dx, dy, dz = (float(d) for d in grid.deltas)
    curved = bool(geometry.use_curved_detector)
    # the focal spot and the radius of a curved panel (unused by a flat panel, which a parallel beam has)
    source_position = geometry.source_position(angle) if curved else (0.0, 0.0, 0.0)
    radius = float(geometry.source_detector_dist) if curved else 0.0
    args = (mu, directions, toward, out, int(V), sG, sR, sC, sS, int(grid.num_cols), int(grid.num_rows),
            int(grid.num_slices), dx, dy, dz, float(grid.recon_slice_offset), *(float(x) for x in panel_center),
            *(float(x) for x in n_hat), float(diag), float(source_position[0]), float(source_position[1]), radius)
    triton_utilities.launch(_downstream_shared_kernel, (triton.cdiv(V, BLOCK_DOWN), K, 1), args,
                            dict(G=int(G), GP=_next_pow2(G), CURVED=curved, BLOCK=BLOCK_DOWN), NUM_WARPS)
    return out


def oblique_project(field: Tensor, grid, geometry, angle: float, omegas: Tensor) -> Tensor:
    """Fused ``SiddonProjectors.oblique_project``: line integrals through each pixel center along each direction.

    Args:
        field (Tensor): [K, G, rows, cols, slices], or [G, rows, cols, slices] shared by all directions; float32 CUDA,
            any strides.
        grid (Grid): the grid of ``field``.
        geometry (ConeGeometry): the cone-beam geometry.
        angle (float): view angle in radians.
        omegas (Tensor): [K, 3] directions (recon frame).

    Returns:
        Tensor: [K, G, num_det_rows, num_det_channels] line integrals in units of field times cm.
    """
    _check_cuda_f32(field, "field")
    device = field.device
    directions = omegas.to(device=device, dtype=torch.float32).reshape(-1, 3).contiguous()
    K = directions.shape[0]
    if field.dim() == 4:
        G = field.shape[0]
        if tuple(field.shape[1:]) != tuple(grid.shape):
            raise ValueError(f"field spatial shape {tuple(field.shape[1:])} != grid shape {grid.shape}")
        volume_stride = 0
    elif field.dim() == 5 and field.shape[0] == K:
        G = field.shape[1]
        if tuple(field.shape[2:]) != tuple(grid.shape):
            raise ValueError(f"field spatial shape {tuple(field.shape[2:])} != grid shape {grid.shape}")
        volume_stride = int(field.stride(0))
    else:
        raise ValueError(f"field must be [G, ...] or [K={K}, G, ...], got {tuple(field.shape)}")
    Rd, Cd = geometry.num_det_rows, geometry.num_det_channels
    out = torch.empty(K, G, Rd, Cd, device=device, dtype=torch.float32)
    if K == 0:
        return out
    lo = grid.lower_corner()
    extent = grid.extent
    geometry_scalars = _pixel_geometry_scalars(geometry, angle)
    geometry_scalars["box_center"] = tuple(lo[a] + 0.5 * extent[a] for a in range(3))
    geometry_scalars["half"] = 0.5 * math.sqrt(sum(e * e for e in extent)) * (1 + 1e-6) + 1e-6
    _launch_ray(field, out, Rd * Cd, K, MODE_OBLIQUE, G, grid, directions=directions, volume_stride=volume_stride,
                geometry_scalars=geometry_scalars)
    return out


def warmup(device: torch.device, num_groups: int) -> None:
    """Compile (or load from Triton's disk cache) every kernel variant of ``SiddonProjectors`` for one group count.

    Args:
        device (torch.device): the CUDA device.
        num_groups (int): number of energy groups G.
    """
    from ..geometry import ConeGeometry, Grid

    grid = Grid(shape=(3, 4, 5), delta_voxel=1.0)
    geometry = ConeGeometry(source_iso_dist=20.0, source_detector_dist=30.0, num_det_rows=4, num_det_channels=5,
                            delta_det_row=1.0, delta_det_channel=1.0)
    mu = torch.rand(num_groups, *grid.shape, device=device)
    directions = torch.tensor([[0.1, -1.0, 0.2], [0.0, -1.0, 0.0]], device=device)
    upstream_cumulative(mu, grid, geometry, 0.0)
    primary_project(mu, grid, geometry, 0.0)
    primary_project(mu, grid, geometry, 0.0, samples_per_pixel_axis=2)
    downstream_cumulative(mu, grid, geometry, 0.0, directions)
    oblique_project(mu, grid, geometry, 0.0, directions)
    oblique_project(mu[None].expand(2, -1, -1, -1, -1), grid, geometry, 0.0, directions)
    torch.cuda.synchronize(device)
