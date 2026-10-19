"""Fused Triton kernels of the first-collision source and of its uses in the sweep and the scatter image.

``_first_collision_kernel`` evaluates the first-collision source of every voxel and direction directly from the
transfer table, without forming the interpolated table, in a pair mode (two materials per point, gathered by index) or
a dense mode (all materials). Three kernels evaluate it where it is used, so that the [K, G, rows, cols, slices] source
is never stored: ``first_collision_moments`` applies the sampling correction and restricts it to the
linear-discontinuous moments of the transport cells (the sweep's source), ``first_collision_sums`` accumulates the
quadrature sums of the correction over all sweep directions, and ``attenuated_emission`` adds the multiple-collision
source of the cell and attenuates the sum to the panel (the scatter image's emission, with the sums of its mean
height). ``prepared_table`` lays out the transfer table for the kernels, and ``kernel_usable`` checks every kernel
against its PyTorch path once per device.
"""

from __future__ import annotations

import weakref
from typing import Optional

import torch
from torch import Tensor

from . import triton_utilities

SELF_CHECK_REL_TOL = 1e-5
BLOCK_V = 64  # points per program (tile [GP, BLOCK_V], about 2 elements per thread: the fastest size tried, see
# docs/source/design_notes.rst)
NUM_WARPS = 8

_SELF_CHECK: dict[str, tuple[bool, str]] = {}
# prepared tables: id(library.mu_s_eta) -> (weak reference to it, {(device, versions): (tab, eta_lo, eta_hi, map)});
# the entry is dropped when the library tensor is freed. (A WeakKeyDictionary would compare tensors with ==.)
_TABLES: dict[int, tuple[weakref.ref, dict[tuple, tuple[Tensor, float, float, Optional[Tensor]]]]] = {}

triton = triton_utilities.triton
tl = triton_utilities.tl


# ================================================================================================
# availability
# ================================================================================================

def triton_available() -> tuple[bool, str]:
    """Report whether Triton kernels can run on this machine (the shared probe of ``triton_utilities``).

    Returns:
        (bool, str): whether Triton is usable, and the reason.
    """
    return triton_utilities.triton_available()


def kernel_usable(device: torch.device) -> tuple[bool, str]:
    """Report whether Triton is available and the fused kernel reproduces the PyTorch path on ``device``.

    The self-check runs once per device on a tiny random problem; the verdict is cached and the function never raises.

    Args:
        device (torch.device): the CUDA device to check.

    Returns:
        (bool, str): whether the kernel is usable, and the reason.
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


def use_kernel(t: Tensor, backend: str) -> bool:
    """Decide whether ``first_collision_source_at_directions`` runs the kernel; float64 and CPU tensors never do.

    Args:
        t (Tensor): the uncollided fluence, or a tensor on its device with its dtype.
        backend (str): "auto", "torch" or "triton".

    Returns:
        bool: True for a float32 CUDA tensor on a device where the kernel is usable, unless ``backend`` is "torch".

    Raises:
        ValueError: for an unknown ``backend``.
        RuntimeError: for backend "triton" when the kernel is unusable on a float32 CUDA tensor.
    """
    if backend not in ("auto", "torch", "triton"):
        raise ValueError("backend must be 'auto', 'torch' or 'triton'")
    if backend == "torch" or t.device.type != "cuda" or t.dtype != torch.float32:
        return False
    ok, why = kernel_usable(t.device)
    if not ok and backend == "triton":
        raise RuntimeError(f"backend='triton' requested but the first-collision-source kernel is unusable: {why}")
    return ok


def _self_check(device: torch.device) -> tuple[bool, str]:
    """Compare the kernel with the PyTorch path in dense and pair mode, with a uniform and with an adaptive table.

    Args:
        device (torch.device): the CUDA device to check.

    Returns:
        (bool, str): whether every comparison is within ``SELF_CHECK_REL_TOL``, and a message.
    """
    worst = 0.0
    for grid_kind in ("uniform", "adaptive"):
        ok, deviation = _self_check_table(device, grid_kind)
        if not ok:
            return False, f"self-check ({grid_kind} table): {deviation}"
        worst = max(worst, float(deviation))
    ok, deviation = _self_check_fused(device)
    if not ok:
        return False, f"self-check (fused kernels): {deviation}"
    worst = max(worst, float(deviation))
    return True, (f"kernels match the PyTorch paths, dense (2 and 7 materials) and pair modes, uniform and adaptive "
                  f"tables, cone and parallel beams, the fused moment, sum and emission kernels with 1 to 3 points per "
                  f"cell (max rel dev {worst:.1e})")


def _self_check_table(device: torch.device, angular_table_grid: str) -> tuple[bool, object]:
    """Compare the kernel with the PyTorch path on a tiny random problem for one kind of angular table.

    Args:
        device (torch.device): the CUDA device to check.
        angular_table_grid (str): "uniform" or "adaptive".

    Returns:
        (bool, object): (True, largest relative deviation) or (False, reason).
    """
    from ..cross_sections import material_pairs, synthetic_library
    from ..geometry import ConeGeometry, Grid, ParallelGeometry
    from ..uncollided import (_first_collision_source_torch, _first_collision_source_torch_pairs,
                              primary_directions_to_voxels)

    library = synthetic_library(4, legendre_order=1, num_eta=201, angular_table_grid=angular_table_grid)
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
    lib7 = synthetic_library(4, legendre_order=1, num_eta=201, num_materials=7, angular_table_grid=angular_table_grid)
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
    # a helical view: the focal spot off the plane z = 0
    helical = geometry.with_helical_z_shifts([0.0, 1.0], [-1.5, 2.5])
    reference_helical = _first_collision_source_torch(library, rho, phi, grid, helical, angle, directions, chunk=4)
    out_helical = first_collision_source(library, rho, phi, grid, helical, angle, directions)
    # a parallel beam: one primary direction at every voxel
    parallel = ParallelGeometry(axis_detector_dist=10.0, num_det_rows=8, num_det_channels=8, delta_det_row=1.0,
                                delta_det_channel=1.0)
    reference_parallel = _first_collision_source_torch(library, rho, phi, grid, parallel, angle, directions, chunk=4)
    out_parallel = first_collision_source(library, rho, phi, grid, parallel, angle, directions)
    torch.cuda.synchronize(device)
    worst = 0.0
    for o, r in ((out, reference), (out7, reference7), (out7d, reference7d), (out_helical, reference_helical),
                 (out_parallel, reference_parallel)):
        if o.shape != r.shape or not bool(torch.isfinite(o).all()):
            return False, "wrong shape or non-finite output"
        worst = max(worst, float((o - r).abs().max()) / max(float(r.abs().max()), 1e-30))
    if worst > SELF_CHECK_REL_TOL:
        return False, f"kernel deviates from the PyTorch path by {worst:.2e} (rel)"
    return True, worst


def _self_check_fused(device: torch.device) -> tuple[bool, object]:
    """Compare the fused moment, sum and emission kernels with their PyTorch compositions on tiny random problems.

    The references are the PyTorch first-collision source with ``FirstCollisionSamplingCorrection.apply`` and
    ``restrict_to_linear_discontinuous_moments``, the weighted direction sums, and ``supervoxel_subpoint_emission`` of
    the source plus the cell's multiple-collision source with its mean-height sums; in pair and dense mode, for
    1, 2 and 3 points per cell, for a cone and a parallel beam.

    Args:
        device (torch.device): the CUDA device to check.

    Returns:
        (bool, object): (True, largest relative deviation) or (False, reason).
    """
    from ..cross_sections import material_pairs, synthetic_library
    from ..geometry import ConeGeometry, Grid, ParallelGeometry
    from ..scatter_image import supervoxel_subpoint_emission
    from ..uncollided import (FirstCollisionSamplingCorrection, _first_collision_source_torch,
                              _first_collision_source_torch_pairs, first_collision_exact_moments,
                              primary_directions_to_voxels, restrict_to_linear_discontinuous_moments)

    generator = torch.Generator().manual_seed(11)
    library = synthetic_library(4, legendre_order=1, num_eta=201, num_materials=7, angular_table_grid="adaptive")
    grid = Grid(shape=(6, 6, 6), delta_voxel=1.0, voxel_slice_aspect=0.8)
    cone = ConeGeometry(source_iso_dist=20.0, source_detector_dist=30.0, num_det_rows=8, num_det_channels=8,
                        delta_det_row=1.0, delta_det_channel=1.0)
    parallel = ParallelGeometry(axis_detector_dist=10.0, num_det_rows=8, num_det_channels=8, delta_det_row=1.0,
                                delta_det_channel=1.0)
    angle = 0.7
    rho = torch.zeros(7, *grid.shape)
    slots = torch.randint(0, 7, (2,) + tuple(grid.shape), generator=generator)
    rho.scatter_(0, slots, torch.rand((2,) + tuple(grid.shape), generator=generator) + 0.1)
    rho = rho.to(device)
    pairs = material_pairs(rho)
    rho_dense = (torch.rand(7, *grid.shape, generator=generator)
                 * (torch.rand(7, *grid.shape, generator=generator) > 0.3)).to(device)
    G = 4
    phi = torch.rand(G, *grid.shape, generator=generator).to(device)
    directions = torch.nn.functional.normalize(torch.randn(5, 3, generator=generator), dim=-1).to(device)
    weights = torch.rand(5, generator=generator).to(device)
    scale = (torch.rand(G, *grid.shape, generator=generator) + 0.5).to(device)
    beta = (2 * torch.rand(G, *grid.shape, generator=generator) - 1).to(device)
    worst = 0.0

    def deviation(a: Tensor, b: Tensor) -> float:
        if a.shape != b.shape or not bool(torch.isfinite(a).all()):
            return float("inf")
        return float((a - b).abs().max()) / max(float(b.abs().max()), 1e-30)

    for geometry in (cone, parallel):
        for density, pair in ((rho, pairs), (rho_dense, None)):
            if pair is not None:
                source = _first_collision_source_torch_pairs(library, pair, phi, grid, geometry, angle, directions)
            else:
                source = _first_collision_source_torch(library, density, phi, grid, geometry, angle, directions)
            omega_uncollided = primary_directions_to_voxels(grid, geometry, angle, device, torch.float32)
            eta = torch.einsum("rcsx,kx->krcs", omega_uncollided, directions)[:, None]  # [K, 1, R, C, S]
            sums = first_collision_sums(library, density, phi, grid, geometry, angle, directions, weights, pairs=pair,
                                        exact_moments=True)
            w = weights[:, None, None, None, None]
            reference_sums = torch.stack([(w * source).sum(0), (w * eta * source).sum(0),
                                          (w * eta * eta * source).sum(0)])
            worst = max(worst, deviation(sums[:3], reference_sums))
            exact = first_collision_exact_moments(library, density, phi.double(), l_max=1).float()
            worst = max(worst, deviation(sums[3:], exact))
            corrected = FirstCollisionSamplingCorrection(scale, beta, omega_uncollided).apply(source, directions)
            for f in (1, 2, 3):
                moments = first_collision_moments(library, density, phi, grid, geometry, angle, directions, scale,
                                                  beta, f, pairs=pair)
                worst = max(worst, deviation(moments, restrict_to_linear_discontinuous_moments(corrected, grid, f)))
                coarse = grid.coarsen(f)
                tau = torch.rand(5, G, *coarse.shape, generator=generator).to(device) * 3.0
                tau[0, 0, 0, 0, 0] = float("inf")
                mu = torch.rand(G, *coarse.shape, generator=generator).to(device)
                multiple = torch.rand(5, G, *coarse.shape, generator=generator).to(device)
                heights = torch.randn(grid.num_voxels, generator=generator).to(device)
                emission, height_sums = attenuated_emission(library, density, phi, grid, geometry, angle, directions,
                                                            tau, mu, multiple, f, heights, pairs=pair)
                total = (source.reshape(5, G, coarse.shape[0], f, coarse.shape[1], f, coarse.shape[2], f)
                         + multiple[:, :, :, None, :, None, :, None]).reshape(source.shape)
                reference = supervoxel_subpoint_emission(total, tau, mu, directions, grid, f)
                worst = max(worst, deviation(emission, reference))
                per_point = reference.sum(dim=1).reshape(5, -1)
                worst = max(worst, deviation(height_sums, torch.stack([per_point.sum(-1), per_point @ heights], -1)))
    torch.cuda.synchronize(device)
    if worst > SELF_CHECK_REL_TOL:
        return False, f"fused kernels deviate from the PyTorch compositions by {worst:.2e} (rel)"
    return True, worst


# ================================================================================================
# kernel
# ================================================================================================

if triton is not None:

    try:  # the accurate exponential of libdevice (tl.exp lowers to the approximate ex2)
        from triton.language.extra import libdevice as _libdevice

        _exp = _libdevice.exp
    except Exception:  # noqa: BLE001
        _exp = tl.exp

    @triton.jit
    def _point_indices(v, C, S):
        """Row, column and slice indices of the flattened [rows, cols, slices] index v.

        Args:
            v: [BLOCK] int32 flat indices.
            C: number of columns.
            S: number of slices.

        Returns:
            (i, j, kk): row, column and slice indices.
        """
        CS = C * S
        i = v // CS
        rem = v - i * CS
        j = rem // S
        return i, j, rem - j * S

    @triton.jit
    def _uncollided_direction(i, j, kk, vm, x_axis_ptr, y_axis_ptr, z_axis_ptr, r0x, r0y, r0z,
                              BLOCK: tl.constexpr, PARALLEL: tl.constexpr):
        """Primary direction ``Omega_UC`` at the points (i, j, kk): from the focal spot, or the beam direction.

        Args:
            i, j, kk: [BLOCK] row, column and slice indices of the points.
            vm: [BLOCK] mask of the valid points.
            x_axis_ptr, y_axis_ptr, z_axis_ptr: float32 point coordinates along the columns, rows and slices.
            r0x, r0y, r0z: focal spot in cm (the beam direction when PARALLEL).
            BLOCK (constexpr): points per program.
            PARALLEL (constexpr): a parallel beam.

        Returns:
            (ux, uy, uz): [BLOCK] components of the unit primary direction.
        """
        if PARALLEL:
            ux = tl.zeros([BLOCK], dtype=tl.float32) + r0x
            uy = tl.zeros([BLOCK], dtype=tl.float32) + r0y
            uz = tl.zeros([BLOCK], dtype=tl.float32) + r0z
        else:
            dx = tl.load(x_axis_ptr + j, mask=vm, other=0.0) - r0x
            dy = tl.load(y_axis_ptr + i, mask=vm, other=0.0) - r0y
            dz = tl.load(z_axis_ptr + kk, mask=vm, other=1.0) - r0z
            nrm = tl.sqrt_rn(dx * dx + dy * dy + dz * dz)  # IEEE sqrt / division (Triton's defaults are approximate)
            ux = tl.div_rn(dx, nrm)
            uy = tl.div_rn(dy, nrm)
            uz = tl.div_rn(dz, nrm)
        return ux, uy, uz

    @triton.jit
    def _pair_slots(v, vm, rho_ptr, index_ptr, V, PAIR: tl.constexpr):
        """Densities and material indices of the two material slots of the points (pair mode; zeros in dense mode).

        Args:
            v: [BLOCK] flat point indices.
            vm: [BLOCK] mask of the valid points.
            rho_ptr: [2, V] float32 slot densities (pair mode).
            index_ptr: [2, V] int32 slot material indices (pair mode).
            V: number of points.
            PAIR (constexpr): pair mode.

        Returns:
            (rh0, rh1, ma, mb): [1, BLOCK] densities and material indices of slots 0 and 1.
        """
        if PAIR:
            rh0 = tl.load(rho_ptr + v, mask=vm, other=0.0)[None, :]
            rh1 = tl.load(rho_ptr + V + v, mask=vm, other=0.0)[None, :]
            ma = tl.load(index_ptr + v, mask=vm, other=0)[None, :]
            mb = tl.load(index_ptr + V + v, mask=vm, other=0)[None, :]
        else:
            rh0 = (v * 0).to(tl.float32)[None, :]
            rh1 = rh0
            ma = (v * 0)[None, :]
            mb = ma
        return rh0, rh1, ma, mb

    @triton.jit
    def _first_collision_at(v, vm, ux, uy, uz, ox, oy, oz, rh0, rh1, ma, mb, rho_ptr, phi_ptr, table_ptr, map_ptr,
                            V, N, KM, eta_lo, eta_hi, G: tl.constexpr, GP: tl.constexpr, BLOCK: tl.constexpr,
                            MT: tl.constexpr, PAIR: tl.constexpr, MAPPED: tl.constexpr):
        """First-collision source of the points v for one direction (ox, oy, oz), every outgoing group.

        Args:
            v: [BLOCK] flat point indices.
            vm: [BLOCK] mask of the valid points.
            ux, uy, uz: [BLOCK] primary direction at the points.
            ox, oy, oz: the direction.
            rh0, rh1, ma, mb: pair slots of the points (``_pair_slots``).
            rho_ptr: [MT, V] float32 densities (dense mode).
            phi_ptr: [G, V] float32 uncollided fluence.
            table_ptr: [MT, N, G, GP] float32 transfer table.
            map_ptr: [KM] float32 node map of an adaptive table.
            V, N, KM: points, eta nodes, map knots.
            eta_lo, eta_hi: ends of the eta grid.
            G, GP, BLOCK, MT, PAIR, MAPPED (constexpr): as in ``_first_collision_kernel``.

        Returns:
            (acc, cosine): [GP, BLOCK] source (rows g >= G are 0) and the [BLOCK] unclamped cosine
            ``Omega_UC . omega``.
        """
        go = tl.arange(0, GP)
        smask = (go < G)[:, None] & vm[None, :]
        nm1 = (N - 1).to(tl.float32)
        km1 = (KM - 1).to(tl.float32)
        ROW: tl.constexpr = G * GP  # floats per eta node of one material; the next node's row follows
        cosine = ux * ox + uy * oy + uz * oz
        eta = tl.minimum(tl.maximum(cosine, eta_lo), eta_hi)
        if MAPPED:
            q = tl.sqrt_rn(tl.maximum((1.0 - eta) * 0.5, 0.0))
            p = q * km1
            k0 = tl.minimum(tl.maximum(tl.floor(p), 0.0), km1 - 1.0)
            a = tl.minimum(tl.maximum(p - k0, 0.0), 1.0)
            k0i = k0.to(tl.int32)
            u0 = tl.load(map_ptr + k0i, mask=vm, other=0.0)
            u1 = tl.load(map_ptr + k0i + 1, mask=vm, other=0.0)
            pos = (u0 + (u1 - u0) * a) * nm1
        else:
            pos = tl.div_rn(eta - eta_lo, eta_hi - eta_lo) * nm1
        i0 = tl.floor(pos)
        i0 = tl.minimum(tl.maximum(i0, 0.0), nm1 - 1.0)
        w = tl.minimum(tl.maximum(pos - i0, 0.0), 1.0)[None, :]
        rowoff = (i0.to(tl.int32) * ROW)[None, :] + go[:, None]
        basea = rowoff + ma * (N * ROW)
        baseb = rowoff + mb * (N * ROW)
        acc = tl.zeros((GP, BLOCK), dtype=tl.float32)
        for gi in tl.static_range(G):
            ph = tl.load(phi_ptr + gi * V + v, mask=vm, other=0.0)
            if PAIR:
                offa = basea + gi * GP
                offb = baseb + gi * GP
                a0 = tl.load(table_ptr + offa, mask=smask, other=0.0)
                a1 = tl.load(table_ptr + offa + ROW, mask=smask, other=0.0)
                b0 = tl.load(table_ptr + offb, mask=smask, other=0.0)
                b1 = tl.load(table_ptr + offb + ROW, mask=smask, other=0.0)
                T = rh0 * (a0 + (a1 - a0) * w) + rh1 * (b0 + (b1 - b0) * w)
            else:
                T = tl.zeros((GP, BLOCK), dtype=tl.float32)
                for m in tl.static_range(MT):
                    rm = tl.load(rho_ptr + m * V + v, mask=vm, other=0.0)[None, :]
                    off = rowoff + m * (N * ROW) + gi * GP
                    t0 = tl.load(table_ptr + off, mask=smask, other=0.0)
                    t1 = tl.load(table_ptr + off + ROW, mask=smask, other=0.0)
                    T += rm * (t0 + (t1 - t0) * w)
            acc += T * ph[None, :]
        return acc, cosine

    @triton.jit
    def _first_collision_kernel(rho_ptr, index_ptr, phi_ptr, x_axis_ptr, y_axis_ptr, z_axis_ptr, direction_ptr,
                                table_ptr, map_ptr, out_ptr,
                                V, C, S, K, N, KM, r0x, r0y, r0z, eta_lo, eta_hi,
                                G: tl.constexpr, GP: tl.constexpr, KB: tl.constexpr, BLOCK: tl.constexpr,
                                MT: tl.constexpr, PAIR: tl.constexpr, MAPPED: tl.constexpr, PARALLEL: tl.constexpr):
        """First-collision source of BLOCK voxels (program_id 0) for KB directions (program_id 1).

        Args:
            rho_ptr: float32 densities in g/cm^3, [MT, V] in dense mode or [2, V] slot densities in pair mode.
            index_ptr: [2, V] int32 material index of each slot (pair mode).
            phi_ptr: [G, V] float32 uncollided fluence.
            x_axis_ptr: [C] float32 x of the voxel centers (columns) in cm.
            y_axis_ptr: [R] float32 y of the voxel centers (rows) in cm.
            z_axis_ptr: [S] float32 z of the voxel centers (slices) in cm.
            direction_ptr: [K, 3] float32 unit directions.
            table_ptr: [MT, N, G, GP] float32 transfer table from ``prepared_table``.
            map_ptr: [KM] float32 node map of an adaptive table (MAPPED).
            out_ptr: [K, G, V] float32 first-collision source.
            V: number of voxels.
            C: number of columns.
            S: number of slices.
            K: number of directions.
            N: number of eta nodes of the table.
            KM: number of knots of the node map.
            r0x: x of the focal spot in cm (the beam direction's x component when PARALLEL).
            r0y: y of the focal spot in cm (the beam direction's y component when PARALLEL).
            r0z: z of the focal spot in cm (the beam direction's z component when PARALLEL).
            eta_lo: first node of the eta grid.
            eta_hi: last node of the eta grid.
            G (constexpr): number of energy groups.
            GP (constexpr): next power of two >= G.
            KB (constexpr): directions per program.
            BLOCK (constexpr): voxels per program.
            MT (constexpr): number of materials of the table.
            PAIR (constexpr): pair mode (two materials per voxel by index) instead of dense mode.
            MAPPED (constexpr): adaptive table read through ``map_ptr``; else uniform in eta.
            PARALLEL (constexpr): a parallel beam: ``Omega_UC`` is the beam direction (r0x, r0y, r0z) at every voxel.
        """
        pid_v = tl.program_id(0)
        pid_k = tl.program_id(1)
        v = pid_v * BLOCK + tl.arange(0, BLOCK)
        vm = v < V
        i, j, kk = _point_indices(v, C, S)
        ux, uy, uz = _uncollided_direction(i, j, kk, vm, x_axis_ptr, y_axis_ptr, z_axis_ptr, r0x, r0y, r0z,
                                           BLOCK, PARALLEL)
        rh0, rh1, ma, mb = _pair_slots(v, vm, rho_ptr, index_ptr, V, PAIR)
        go = tl.arange(0, GP)
        smask = (go < G)[:, None] & vm[None, :]
        for kb in range(KB):
            k = pid_k * KB + kb
            if k < K:
                ox = tl.load(direction_ptr + 3 * k)
                oy = tl.load(direction_ptr + 3 * k + 1)
                oz = tl.load(direction_ptr + 3 * k + 2)
                acc, cosine = _first_collision_at(v, vm, ux, uy, uz, ox, oy, oz, rh0, rh1, ma, mb, rho_ptr, phi_ptr,
                                                  table_ptr, map_ptr, V, N, KM, eta_lo, eta_hi, G, GP, BLOCK, MT, PAIR,
                                                  MAPPED)
                obase = (k.to(tl.int64) * G + go[:, None].to(tl.int64)) * V + v[None, :]
                tl.store(out_ptr + obase, acc, mask=smask)

    @triton.jit
    def _first_collision_moment_kernel(rho_ptr, index_ptr, phi_ptr, x_axis_ptr, y_axis_ptr, z_axis_ptr,
                                       direction_ptr, table_ptr, map_ptr, scale_ptr, beta_ptr, out_ptr,
                                       V, C, S, NC, CC, SC, K, N, KM, r0x, r0y, r0z, eta_lo, eta_hi, slope_scale,
                                       G: tl.constexpr, GP: tl.constexpr, KB: tl.constexpr, BLOCK: tl.constexpr,
                                       MT: tl.constexpr, F: tl.constexpr, PAIR: tl.constexpr, MAPPED: tl.constexpr,
                                       PARALLEL: tl.constexpr):
        """Linear-discontinuous moments of the corrected first-collision source of BLOCK transport cells.

        For every cell (program_id 0) and KB directions (program_id 1) the source is evaluated at the F^3 quadrature
        points of the cell, multiplied by the sampling correction ``scale (1 + beta cosine)`` and restricted to the
        cell mean and the three slopes (``uncollided.restrict_to_linear_discontinuous_moments``).

        Args:
            rho_ptr, index_ptr, phi_ptr, x_axis_ptr, y_axis_ptr, z_axis_ptr, direction_ptr, table_ptr, map_ptr:
                as in ``_first_collision_kernel``, on the quadrature grid.
            scale_ptr: [G, V] float32 correction scale.
            beta_ptr: [G, V] float32 correction coefficient beta.
            out_ptr: [K, G, 4, NC] float32 moments [mean, x slope, y slope, z slope] per cell.
            V, C, S: points, columns and slices of the quadrature grid.
            NC, CC, SC: cells, columns and slices of the transport grid.
            K, N, KM, r0x, r0y, r0z, eta_lo, eta_hi: as in ``_first_collision_kernel``.
            slope_scale: ``1 / mean(u^2)`` of the local coordinates ``u = (2j + 1)/F - 1``.
            G, GP, KB, MT, PAIR, MAPPED, PARALLEL (constexpr): as in ``_first_collision_kernel``.
            BLOCK (constexpr): cells per program.
            F (constexpr): quadrature points per axis per cell.
        """
        pid_c = tl.program_id(0)
        pid_k = tl.program_id(1)
        c = pid_c * BLOCK + tl.arange(0, BLOCK)
        cm = c < NC
        ic, jc, kc = _point_indices(c, CC, SC)
        go = tl.arange(0, GP)
        smask = (go < G)[:, None] & cm[None, :]
        inv_points = 1.0 / (F * F * F)
        for kb in range(KB):
            k = pid_k * KB + kb
            if k < K:
                ox = tl.load(direction_ptr + 3 * k)
                oy = tl.load(direction_ptr + 3 * k + 1)
                oz = tl.load(direction_ptr + 3 * k + 2)
                m0 = tl.zeros((GP, BLOCK), dtype=tl.float32)
                mx = tl.zeros((GP, BLOCK), dtype=tl.float32)
                my = tl.zeros((GP, BLOCK), dtype=tl.float32)
                mz = tl.zeros((GP, BLOCK), dtype=tl.float32)
                for p in range(F * F * F):  # the points of the cell, a runtime loop (one copy of the source code)
                    a = p // (F * F)  # rows (y)
                    b = (p // F) % F  # cols (x)
                    d = p % F  # slices (z)
                    ua = (2.0 * a + 1.0) / F - 1.0
                    ub = (2.0 * b + 1.0) / F - 1.0
                    ud = (2.0 * d + 1.0) / F - 1.0
                    i = ic * F + a
                    j = jc * F + b
                    kk = kc * F + d
                    v = (i * C + j) * S + kk
                    ux, uy, uz = _uncollided_direction(i, j, kk, cm, x_axis_ptr, y_axis_ptr, z_axis_ptr, r0x, r0y,
                                                       r0z, BLOCK, PARALLEL)
                    rh0, rh1, ma, mb = _pair_slots(v, cm, rho_ptr, index_ptr, V, PAIR)
                    acc, cosine = _first_collision_at(v, cm, ux, uy, uz, ox, oy, oz, rh0, rh1, ma, mb, rho_ptr,
                                                      phi_ptr, table_ptr, map_ptr, V, N, KM, eta_lo, eta_hi, G, GP,
                                                      BLOCK, MT, PAIR, MAPPED)
                    gv = go[:, None] * V + v[None, :]
                    sc = tl.load(scale_ptr + gv, mask=smask, other=0.0)
                    be = tl.load(beta_ptr + gv, mask=smask, other=0.0)
                    s = acc * sc * (1.0 + be * cosine[None, :])
                    m0 += s
                    mx += s * ub
                    my += s * ua
                    mz += s * ud
                obase = ((k.to(tl.int64) * G + go[:, None].to(tl.int64)) * 4) * NC + c[None, :]
                tl.store(out_ptr + obase, m0 * inv_points, mask=smask)
                tl.store(out_ptr + obase + NC, mx * (inv_points * slope_scale), mask=smask)
                tl.store(out_ptr + obase + 2 * NC, my * (inv_points * slope_scale), mask=smask)
                tl.store(out_ptr + obase + 3 * NC, mz * (inv_points * slope_scale), mask=smask)

    @triton.jit
    def _first_collision_sums_kernel(rho_ptr, index_ptr, phi_ptr, x_axis_ptr, y_axis_ptr, z_axis_ptr, direction_ptr,
                                     weight_ptr, table_ptr, map_ptr, moment_ptr, out_ptr,
                                     V, C, S, K, N, KM, r0x, r0y, r0z, eta_lo, eta_hi,
                                     G: tl.constexpr, GP: tl.constexpr, BLOCK: tl.constexpr, MT: tl.constexpr,
                                     PAIR: tl.constexpr, MAPPED: tl.constexpr, PARALLEL: tl.constexpr):
        """Quadrature sums I0, I1, I2 of the first-collision source over all K directions, and its exact moments.

        ``I_n = sum_k w_k cosine_k^n S_k`` with ``cosine_k = Omega_UC . omega_k`` (``uncollided.
        first_collision_quadrature_sums``), accumulated in float32 over the directions, and the exact Legendre moments
        ``E_l = sum_{g' <= g} mu_s_l^{g' -> g} phi_g'`` for l = 0, 1 (``uncollided.first_collision_exact_moments``),
        for BLOCK points.

        Args:
            rho_ptr, index_ptr, phi_ptr, x_axis_ptr, y_axis_ptr, z_axis_ptr, direction_ptr, table_ptr, map_ptr:
                as in ``_first_collision_kernel``.
            weight_ptr: [K] float32 quadrature weights.
            moment_ptr: [MT, G, 2, GP] float32 Legendre transfer moments l = 0, 1 (``prepared_moment_table``).
            out_ptr: [5, G, V] float32 sums I0, I1, I2 and moments E0, E1.
            V, C, S, K, N, KM, r0x, r0y, r0z, eta_lo, eta_hi: as in ``_first_collision_kernel``.
            G, GP, BLOCK, MT, PAIR, MAPPED, PARALLEL (constexpr): as in ``_first_collision_kernel``.
        """
        pid_v = tl.program_id(0)
        v = pid_v * BLOCK + tl.arange(0, BLOCK)
        vm = v < V
        i, j, kk = _point_indices(v, C, S)
        ux, uy, uz = _uncollided_direction(i, j, kk, vm, x_axis_ptr, y_axis_ptr, z_axis_ptr, r0x, r0y, r0z,
                                           BLOCK, PARALLEL)
        rh0, rh1, ma, mb = _pair_slots(v, vm, rho_ptr, index_ptr, V, PAIR)
        go = tl.arange(0, GP)
        smask = (go < G)[:, None] & vm[None, :]
        # the exact moments, once per point
        E0 = tl.zeros((GP, BLOCK), dtype=tl.float32)
        E1 = tl.zeros((GP, BLOCK), dtype=tl.float32)
        for gi in tl.static_range(G):
            ph = tl.load(phi_ptr + gi * V + v, mask=vm, other=0.0)[None, :]
            row = (gi * 2) * GP + go[:, None]
            if PAIR:
                e0a = tl.load(moment_ptr + ma * (2 * G * GP) + row, mask=smask, other=0.0)
                e1a = tl.load(moment_ptr + ma * (2 * G * GP) + row + GP, mask=smask, other=0.0)
                e0b = tl.load(moment_ptr + mb * (2 * G * GP) + row, mask=smask, other=0.0)
                e1b = tl.load(moment_ptr + mb * (2 * G * GP) + row + GP, mask=smask, other=0.0)
                E0 += (rh0 * e0a + rh1 * e0b) * ph
                E1 += (rh0 * e1a + rh1 * e1b) * ph
            else:
                for m in tl.static_range(MT):
                    rm = tl.load(rho_ptr + m * V + v, mask=vm, other=0.0)[None, :]
                    e0 = tl.load(moment_ptr + (m * G * 2) * GP + row, mask=(go < G)[:, None], other=0.0)
                    e1 = tl.load(moment_ptr + (m * G * 2) * GP + row + GP, mask=(go < G)[:, None], other=0.0)
                    E0 += rm * e0 * ph
                    E1 += rm * e1 * ph
        I0 = tl.zeros((GP, BLOCK), dtype=tl.float32)
        I1 = tl.zeros((GP, BLOCK), dtype=tl.float32)
        I2 = tl.zeros((GP, BLOCK), dtype=tl.float32)
        for k in range(K):
            ox = tl.load(direction_ptr + 3 * k)
            oy = tl.load(direction_ptr + 3 * k + 1)
            oz = tl.load(direction_ptr + 3 * k + 2)
            wk = tl.load(weight_ptr + k)
            acc, cosine = _first_collision_at(v, vm, ux, uy, uz, ox, oy, oz, rh0, rh1, ma, mb, rho_ptr, phi_ptr,
                                              table_ptr, map_ptr, V, N, KM, eta_lo, eta_hi, G, GP, BLOCK, MT, PAIR,
                                              MAPPED)
            ws = acc * wk
            mu = cosine[None, :]
            I0 += ws
            I1 += ws * mu
            I2 += ws * (mu * mu)
        obase = go[:, None].to(tl.int64) * V + v[None, :]
        tl.store(out_ptr + obase, I0, mask=smask)
        tl.store(out_ptr + G * V + obase, I1, mask=smask)
        tl.store(out_ptr + 2 * G * V + obase, I2, mask=smask)
        tl.store(out_ptr + 3 * G * V + obase, E0, mask=smask)
        tl.store(out_ptr + 4 * G * V + obase, E1, mask=smask)

    @triton.jit
    def _attenuated_emission_kernel(rho_ptr, index_ptr, phi_ptr, x_axis_ptr, y_axis_ptr, z_axis_ptr, direction_ptr,
                                    table_ptr, map_ptr, tau_ptr, mu_ptr, multiple_ptr, height_ptr, out_ptr,
                                    partial_ptr,
                                    V, C, S, NC, CC, SC, K, N, KM, r0x, r0y, r0z, eta_lo, eta_hi, dx, dy, dz, NP,
                                    G: tl.constexpr, GP: tl.constexpr, KB: tl.constexpr, BLOCK: tl.constexpr,
                                    MT: tl.constexpr, F: tl.constexpr, PAIR: tl.constexpr, MAPPED: tl.constexpr,
                                    PARALLEL: tl.constexpr):
        """Attenuated total emission of BLOCK quadrature points for KB directions, with the sums of its mean height.

        ``e = (S_FC + S_MC,cell) exp(-max(0, tau_cell - mu_t,cell (omega . (r - r_cell))))`` per point and group
        (``scatter_image.supervoxel_subpoint_emission`` of the first-collision plus the multiple-collision source), and
        per program the sums of ``e`` and ``e h`` over its points and groups, h the panel height of each point.

        Args:
            rho_ptr, index_ptr, phi_ptr, x_axis_ptr, y_axis_ptr, z_axis_ptr, direction_ptr, table_ptr, map_ptr:
                as in ``_first_collision_kernel``, on the quadrature grid.
            tau_ptr: [K, G, NC] float32 optical depth from the cell centers to the panel (+inf allowed).
            mu_ptr: [G, NC] float32 total attenuation of the cells in cm^-1.
            multiple_ptr: [K, G, NC] float32 multiple-collision source of the cells.
            height_ptr: [V] float32 panel height of the points in cm.
            out_ptr: [K, G, V] float32 attenuated emission.
            partial_ptr: [K, NP, 2] float32 sums of e and of e h per program.
            V, C, S: points, columns and slices of the quadrature grid.
            NC, CC, SC: cells, columns and slices of the transport grid.
            K, N, KM, r0x, r0y, r0z, eta_lo, eta_hi: as in ``_first_collision_kernel``.
            dx, dy, dz: point pitches of the quadrature grid along x (cols), y (rows), z (slices) in cm.
            NP: number of point programs (program_id 0).
            G, GP, KB, BLOCK, MT, PAIR, MAPPED, PARALLEL (constexpr): as in ``_first_collision_kernel``.
            F (constexpr): quadrature points per axis per cell.
        """
        pid_v = tl.program_id(0)
        pid_k = tl.program_id(1)
        v = pid_v * BLOCK + tl.arange(0, BLOCK)
        vm = v < V
        i, j, kk = _point_indices(v, C, S)
        ux, uy, uz = _uncollided_direction(i, j, kk, vm, x_axis_ptr, y_axis_ptr, z_axis_ptr, r0x, r0y, r0z,
                                           BLOCK, PARALLEL)
        rh0, rh1, ma, mb = _pair_slots(v, vm, rho_ptr, index_ptr, V, PAIR)
        ic = i // F
        jc = j // F
        kc = kk // F
        cell = (ic * CC + jc) * SC + kc
        half = (F - 1) * 0.5
        # offsets of the point from its cell center along rows (y), cols (x) and slices (z), in cm
        oy_cell = dy * ((i - ic * F).to(tl.float32) - half)
        ox_cell = dx * ((j - jc * F).to(tl.float32) - half)
        oz_cell = dz * ((kk - kc * F).to(tl.float32) - half)
        h = tl.load(height_ptr + v, mask=vm, other=0.0)
        go = tl.arange(0, GP)
        smask = (go < G)[:, None] & vm[None, :]
        gcell = go[:, None] * NC + cell[None, :]
        mu = tl.load(mu_ptr + gcell, mask=smask, other=0.0)
        for kb in range(KB):
            k = pid_k * KB + kb
            if k < K:
                ox = tl.load(direction_ptr + 3 * k)
                oy = tl.load(direction_ptr + 3 * k + 1)
                oz = tl.load(direction_ptr + 3 * k + 2)
                acc, cosine = _first_collision_at(v, vm, ux, uy, uz, ox, oy, oz, rh0, rh1, ma, mb, rho_ptr, phi_ptr,
                                                  table_ptr, map_ptr, V, N, KM, eta_lo, eta_hi, G, GP, BLOCK, MT, PAIR,
                                                  MAPPED)
                offset = oy * oy_cell + ox * ox_cell + oz * oz_cell
                kcell = k.to(tl.int64) * G * NC + gcell
                tau = tl.load(tau_ptr + kcell, mask=smask, other=0.0)
                multiple = tl.load(multiple_ptr + kcell, mask=smask, other=0.0)
                depth = tl.maximum(tau - mu * offset[None, :], 0.0)
                e = (acc + multiple) * _exp(-depth)
                e = tl.where(smask, e, 0.0)
                obase = (k.to(tl.int64) * G + go[:, None].to(tl.int64)) * V + v[None, :]
                tl.store(out_ptr + obase, e, mask=smask)
                per_point = tl.sum(e, axis=0)
                pbase = (k.to(tl.int64) * NP + pid_v) * 2
                tl.store(partial_ptr + pbase, tl.sum(per_point, axis=0))
                tl.store(partial_ptr + pbase + 1, tl.sum(per_point * h, axis=0))


# ================================================================================================
# wrapper
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


def prepared_table(library, device: torch.device) -> tuple[Tensor, float, float, Optional[Tensor]]:
    """Lay out the transfer table of ``library`` for the kernel, cached per table, device and tensor versions.

    Args:
        library (MultigroupLibrary): the library; uses ``mu_s_eta`` [M, G, G, N], ``eta_grid`` [N] and ``eta_map``.
        device (torch.device): the CUDA device.

    Returns:
        (table, eta_lo, eta_hi, map): table is the [M, N, G_in, GP] float32 transfer table in cm^2/g/sr, zero where
        ``g_out < g_in`` or ``g_out >= G``; eta_lo and eta_hi the ends of the eta grid; map the [KM] float32 node map
        (None for a uniform table).
    """
    source_table = library.mu_s_eta
    eta_map = getattr(library, "eta_map", None)
    key = (str(torch.device(device)), int(source_table._version), int(library.eta_grid._version),
           id(library.eta_grid), None if eta_map is None else (id(eta_map), int(eta_map._version)))
    hit = _TABLES.get(id(source_table))
    if hit is not None and hit[0]() is source_table and key in hit[1]:
        return hit[1][key]
    M, G, G2, N = source_table.shape
    if G != G2:
        raise ValueError(f"transfer table must be [M, G, G, N], got {tuple(source_table.shape)}")
    GP = _next_pow2(G)
    t = source_table.to(device=device, dtype=torch.float32)
    t = t * torch.triu(torch.ones(G, G, device=device, dtype=torch.float32))[None, :, :, None]
    kernel_table = torch.zeros(M, N, G, GP, device=device, dtype=torch.float32)
    kernel_table[..., :G] = t.permute(0, 3, 1, 2)
    eta_grid = library.eta_grid.detach().to("cpu", torch.float64)
    kernel_map = None if eta_map is None else eta_map.detach().to(device=device, dtype=torch.float32).contiguous()
    entry = (kernel_table.contiguous(), float(eta_grid[0]), float(eta_grid[-1]), kernel_map)
    table_id = id(source_table)
    if hit is None or hit[0]() is not source_table:
        _TABLES[table_id] = (weakref.ref(source_table, lambda _r, table_id=table_id: _TABLES.pop(table_id, None)), {})
    per_version = _TABLES[table_id][1]
    per_version.clear()  # keep only the latest version per source tensor
    per_version[key] = entry
    return entry


def prepared_moment_table(library, device: torch.device) -> Tensor:
    """Lay out the Legendre transfer moments l = 0, 1 of ``library`` for the sums kernel, cached per tensor and device.

    Args:
        library (MultigroupLibrary): the library; uses ``mu_s_l`` [M, G, G, L+1] with L >= 1.
        device (torch.device): the CUDA device.

    Returns:
        Tensor: [M, G_in, 2, GP] float32 moments in cm^2/g, zero where ``g_out < g_in`` or ``g_out >= G``.
    """
    source = library.mu_s_l
    key = (str(torch.device(device)), int(source._version), "moments")
    hit = _TABLES.get(id(source))
    if hit is not None and hit[0]() is source and key in hit[1]:
        return hit[1][key]
    M, G, G2, L1 = source.shape
    if G != G2 or L1 < 2:
        raise ValueError(f"transfer moments must be [M, G, G, L+1] with L >= 1, got {tuple(source.shape)}")
    GP = _next_pow2(G)
    t = source[..., :2].to(device=device, dtype=torch.float32)
    t = t * torch.triu(torch.ones(G, G, device=device, dtype=torch.float32))[None, :, :, None]
    table = torch.zeros(M, G, 2, GP, device=device, dtype=torch.float32)
    table[..., :G] = t.permute(0, 1, 3, 2)
    table = table.contiguous()
    source_id = id(source)
    if hit is None or hit[0]() is not source:
        _TABLES[source_id] = (weakref.ref(source, lambda _r, source_id=source_id: _TABLES.pop(source_id, None)), {})
    per_version = _TABLES[source_id][1]
    per_version.clear()
    per_version[key] = table
    return table


def _directions_per_program(K: int, num_vblocks: int, device: torch.device) -> int:
    """Choose the directions per program: as many as possible (up to 8) while keeping at least 8 programs per SM.

    Args:
        K (int): number of directions.
        num_vblocks (int): number of voxel blocks.
        device (torch.device): the CUDA device.

    Returns:
        int: directions per program, a power of two in [1, 8].
    """
    num_multiprocessors = torch.cuda.get_device_properties(device).multi_processor_count
    target = 8 * num_multiprocessors
    per_program = 1
    while per_program < 8 and per_program < K and num_vblocks * -(-K // (2 * per_program)) >= target:
        per_program *= 2
    return per_program


def kernel_supports(library, rho: Tensor, phi_uncollided: Tensor, pairs=None) -> bool:
    """Check the shape preconditions of the kernel.

    Args:
        library (MultigroupLibrary): the library; ``mu_s_eta`` must be [M, G, G, N] with N >= 2.
        rho (Tensor): [M, rows, cols, slices] densities of all M materials in g/cm^3.
        phi_uncollided (Tensor): [G, rows, cols, slices] uncollided fluence.
        pairs (MaterialPairs, optional): pair representation of ``rho`` for an M-material library. Defaults to None.

    Returns:
        bool: True when the table matches the groups of ``phi_uncollided`` and the materials of ``rho`` or ``pairs``.
    """
    G = phi_uncollided.shape[0]
    t = library.mu_s_eta
    if tuple(t.shape[1:3]) != (G, G) or t.shape[3] < 2:
        return False
    if pairs is not None:
        return int(pairs.num_materials) == int(t.shape[0]) and int(rho.shape[0]) == int(t.shape[0])
    return int(rho.shape[0]) == int(t.shape[0]) >= 1


def _kernel_inputs(library, rho: Tensor, phi_uncollided: Tensor, grid, geometry, angle: float, pairs) -> dict:
    """Flattened inputs shared by the first-collision kernels.

    Args:
        library (MultigroupLibrary): the library.
        rho (Tensor): [M, rows, cols, slices] densities in g/cm^3.
        phi_uncollided (Tensor): [G, rows, cols, slices] float32 CUDA uncollided fluence.
        grid (Grid): the grid of ``rho`` and ``phi_uncollided``.
        geometry (ConeGeometry or ParallelGeometry): the scan geometry.
        angle (float): view angle in radians.
        pairs (MaterialPairs or None): pair representation of ``rho``, or None for dense mode.

    Returns:
        dict: the kernel arguments ``rho``, ``index``, ``phi``, ``x``, ``y``, ``z``, ``table``, ``map``, the sizes
            ``V``, ``C``, ``S``, ``N``, ``KM``, the focal spot (or beam direction) ``r0``, ``eta_lo``, ``eta_hi`` and
            the constexprs ``G``, ``GP``, ``MT``, ``PAIR``, ``MAPPED``, ``PARALLEL``.

    Raises:
        ValueError: if the tensors are not float32 CUDA tensors or do not match the table and the grid.
    """
    device, dtype = phi_uncollided.device, phi_uncollided.dtype
    if device.type != "cuda" or dtype != torch.float32:
        raise ValueError("the fused first-collision-source kernels need float32 CUDA tensors")
    if not kernel_supports(library, rho, phi_uncollided, pairs):
        raise ValueError("the fused first-collision-source kernels need the densities of all M materials of the [M, G, "
                         "G, N] table, or the pair representation of the volume with a matching table")
    G = int(phi_uncollided.shape[0])
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
    table, eta_lo, eta_hi, kernel_map = prepared_table(library, device)
    x, y, z = grid.axis_coordinates(device, dtype)
    parallel = bool(geometry.is_parallel)
    r0 = geometry.beam_direction(angle) if parallel else geometry.source_position(angle)
    return dict(rho=rho_flat, index=index_flat, phi=phi_uncollided.reshape(G, V).contiguous(), x=x, y=y, z=z,
                table=table, map=table if kernel_map is None else kernel_map, V=V, C=C, S=S, N=int(table.shape[1]),
                KM=2 if kernel_map is None else int(kernel_map.shape[0]), r0=(float(r0[0]), float(r0[1]),
                                                                             float(r0[2])),
                eta_lo=eta_lo, eta_hi=eta_hi, G=G, GP=_next_pow2(G), MT=int(table.shape[0]), PAIR=pairs is not None,
                MAPPED=kernel_map is not None, PARALLEL=parallel)


def first_collision_source(library, rho: Tensor, phi_uncollided: Tensor, grid, geometry, angle: float, omegas: Tensor,
                           pairs=None) -> Tensor:
    """Fused ``uncollided.first_collision_source_at_directions`` on float32 CUDA tensors, without availability check.

    Args:
        library (MultigroupLibrary): the library; uses ``mu_s_eta`` [M, G, G, N], ``eta_grid`` and ``eta_map``.
        rho (Tensor): [M, rows, cols, slices] densities in g/cm^3.
        phi_uncollided (Tensor): [G, rows, cols, slices] float32 CUDA uncollided fluence.
        grid (Grid): the grid of ``rho`` and ``phi_uncollided``.
        geometry (ConeGeometry or ParallelGeometry): the scan geometry.
        angle (float): view angle in radians.
        omegas (Tensor): [K, 3] unit directions (recon frame).
        pairs (MaterialPairs, optional): pair representation of ``rho`` (pair mode), or None for dense mode.
            Defaults to None.

    Returns:
        Tensor: [K, G, rows, cols, slices] float32 first-collision source in photons/(cm^3 sr) per source photon/sr.
    """
    a = _kernel_inputs(library, rho, phi_uncollided, grid, geometry, angle, pairs)
    directions = omegas.to(device=phi_uncollided.device, dtype=torch.float32).reshape(-1, 3).contiguous()
    K, V = int(directions.shape[0]), a["V"]
    out = torch.empty(K, a["G"], *grid.shape, device=phi_uncollided.device, dtype=torch.float32)
    if K == 0 or V == 0:
        return out
    num_voxel_blocks = -(-V // BLOCK_V)
    directions_per_program = _directions_per_program(K, num_voxel_blocks, phi_uncollided.device)
    _first_collision_kernel[(num_voxel_blocks, -(-K // directions_per_program))](
        a["rho"], a["index"], a["phi"], a["x"], a["y"], a["z"], directions, a["table"], a["map"], out,
        V, a["C"], a["S"], K, a["N"], a["KM"], *a["r0"], a["eta_lo"], a["eta_hi"],
        G=a["G"], GP=a["GP"], KB=directions_per_program, BLOCK=BLOCK_V, MT=a["MT"], PAIR=a["PAIR"],
        MAPPED=a["MAPPED"], PARALLEL=a["PARALLEL"], num_warps=NUM_WARPS)
    return out


def first_collision_moments(library, rho: Tensor, phi_uncollided: Tensor, grid, geometry, angle: float,
                            omegas: Tensor, scale: Tensor, beta: Tensor, factor: int, pairs=None,
                            out: Optional[Tensor] = None) -> Tensor:
    """Linear-discontinuous moments of the corrected first-collision source on the cells of a refined grid.

    The fused form of ``first_collision_source_at_directions``, ``FirstCollisionSamplingCorrection.apply`` and
    ``restrict_to_linear_discontinuous_moments``: no [K, G, rows, cols, slices] source is formed.

    Args:
        library (MultigroupLibrary): the library.
        rho (Tensor): [M, rows, cols, slices] densities on the quadrature grid in g/cm^3.
        phi_uncollided (Tensor): [G, rows, cols, slices] float32 CUDA uncollided fluence.
        grid (Grid): the quadrature grid, ``factor`` points per axis per cell.
        geometry (ConeGeometry or ParallelGeometry): the scan geometry.
        angle (float): view angle in radians.
        omegas (Tensor): [K, 3] unit directions.
        scale (Tensor): [G, rows, cols, slices] correction scale.
        beta (Tensor): [G, rows, cols, slices] correction coefficient beta.
        factor (int): quadrature points per axis per cell.
        pairs (MaterialPairs, optional): pair representation of ``rho``, or None for dense mode. Defaults to None.
        out (Tensor, optional): contiguous [K, G, 4, rows/f, cols/f, slices/f] float32 tensor to fill. Defaults to None
            (allocated).

    Returns:
        Tensor: [K, G, 4, rows/f, cols/f, slices/f] moments ordered [0, x, y, z].
    """
    a = _kernel_inputs(library, rho, phi_uncollided, grid, geometry, angle, pairs)
    device = phi_uncollided.device
    f = int(factor)
    coarse = grid.coarsen(f)
    Rc, Cc, Sc = coarse.shape
    directions = omegas.to(device=device, dtype=torch.float32).reshape(-1, 3).contiguous()
    K, G = int(directions.shape[0]), a["G"]
    expected = (K, G, 4, Rc, Cc, Sc)
    if out is None:
        out = torch.empty(expected, device=device, dtype=torch.float32)
    elif tuple(out.shape) != expected or not out.is_contiguous() or out.dtype != torch.float32:
        raise ValueError(f"out must be a contiguous float32 tensor of shape {expected}")
    if K == 0:
        return out
    scale_flat = scale.to(device=device, dtype=torch.float32).reshape(G, -1).contiguous()
    beta_flat = beta.to(device=device, dtype=torch.float32).reshape(G, -1).contiguous()
    u = (2.0 * torch.arange(f, dtype=torch.float32) + 1.0) / f - 1.0
    slope_scale = float(1.0 / (u * u).mean()) if f > 1 else 0.0
    NC = Rc * Cc * Sc
    num_cell_blocks = -(-NC // BLOCK_V)
    _first_collision_moment_kernel[(num_cell_blocks, K)](
        a["rho"], a["index"], a["phi"], a["x"], a["y"], a["z"], directions, a["table"], a["map"], scale_flat,
        beta_flat, out, a["V"], a["C"], a["S"], NC, Cc, Sc, K, a["N"], a["KM"], *a["r0"], a["eta_lo"],
        a["eta_hi"], slope_scale,
        G=G, GP=a["GP"], KB=1, BLOCK=BLOCK_V, MT=a["MT"], F=f, PAIR=a["PAIR"], MAPPED=a["MAPPED"],
        PARALLEL=a["PARALLEL"], num_warps=NUM_WARPS)
    return out


def first_collision_sums(library, rho: Tensor, phi_uncollided: Tensor, grid, geometry, angle: float, omegas: Tensor,
                         weights: Tensor, pairs=None, exact_moments: bool = False) -> Tensor:
    """Quadrature sums I0, I1, I2 of the first-collision source per point and group, without forming the source.

    The fused form of ``uncollided.first_collision_quadrature_sums``; with ``exact_moments`` also the exact moments E0,
    E1 of ``uncollided.first_collision_exact_moments``, in float32.

    Args:
        library (MultigroupLibrary): the library.
        rho (Tensor): [M, rows, cols, slices] densities in g/cm^3.
        phi_uncollided (Tensor): [G, rows, cols, slices] float32 CUDA uncollided fluence.
        grid (Grid): the grid of ``rho`` and ``phi_uncollided``.
        geometry (ConeGeometry or ParallelGeometry): the scan geometry.
        angle (float): view angle in radians.
        omegas (Tensor): [K, 3] unit directions of the quadrature.
        weights (Tensor): [K] quadrature weights in sr.
        pairs (MaterialPairs, optional): pair representation of ``rho``, or None for dense mode. Defaults to None.
        exact_moments (bool, optional): also return the exact moments. Defaults to False.

    Returns:
        Tensor: [3, G, rows, cols, slices] float32 sums, or [5, G, rows, cols, slices] sums and moments E0, E1.
    """
    a = _kernel_inputs(library, rho, phi_uncollided, grid, geometry, angle, pairs)
    device = phi_uncollided.device
    directions = omegas.to(device=device, dtype=torch.float32).reshape(-1, 3).contiguous()
    weight = weights.to(device=device, dtype=torch.float32).reshape(-1).contiguous()
    K, G, V = int(directions.shape[0]), a["G"], a["V"]
    out = torch.zeros(5, G, *grid.shape, device=device, dtype=torch.float32)
    if V == 0:
        return out if exact_moments else out[:3]
    moments = prepared_moment_table(library, device)
    if int(moments.shape[0]) != a["MT"]:
        raise ValueError("the transfer moments and the transfer table have different material counts")
    _first_collision_sums_kernel[(-(-V // BLOCK_V),)](
        a["rho"], a["index"], a["phi"], a["x"], a["y"], a["z"], directions, weight, a["table"], a["map"], moments,
        out,
        V, a["C"], a["S"], K, a["N"], a["KM"], *a["r0"], a["eta_lo"], a["eta_hi"],
        G=G, GP=a["GP"], BLOCK=BLOCK_V, MT=a["MT"], PAIR=a["PAIR"], MAPPED=a["MAPPED"], PARALLEL=a["PARALLEL"],
        num_warps=NUM_WARPS)
    return out if exact_moments else out[:3]


def attenuated_emission(library, rho: Tensor, phi_uncollided: Tensor, grid, geometry, angle: float, omegas: Tensor,
                        tau_coarse: Tensor, mu_t_coarse: Tensor, multiple_collision: Tensor, factor: int,
                        heights: Optional[Tensor] = None, pairs=None) -> tuple[Tensor, Tensor]:
    """Attenuated total emission at the quadrature points and the sums of its mean panel height, per direction.

    The fused form of ``first_collision_source_at_directions``, the addition of the cell's multiple-collision source
    and ``scatter_image.supervoxel_subpoint_emission``; the per-direction sums replace the reduction of
    ``scatter_image._Footprint.mean_height``.

    Args:
        library (MultigroupLibrary): the library.
        rho (Tensor): [M, rows, cols, slices] densities on the quadrature grid in g/cm^3.
        phi_uncollided (Tensor): [G, rows, cols, slices] float32 CUDA uncollided fluence.
        grid (Grid): the quadrature grid, ``factor`` points per axis per transport cell.
        geometry (ConeGeometry or ParallelGeometry): the scan geometry.
        angle (float): view angle in radians.
        omegas (Tensor): [K, 3] unit directions.
        tau_coarse (Tensor): [K, G, Rc, Cc, Sc] optical depth to the panel from the cell centers (+inf allowed).
        mu_t_coarse (Tensor): [G, Rc, Cc, Sc] total attenuation of the cells in cm^-1.
        multiple_collision (Tensor): [K, G, Rc, Cc, Sc] multiple-collision source of the cells.
        factor (int): quadrature points per axis per cell.
        heights (Tensor, optional): [rows * cols * slices] panel heights of the points in cm. Defaults to None (zeros).
        pairs (MaterialPairs, optional): pair representation of ``rho``, or None for dense mode. Defaults to None.

    Returns:
        (Tensor, Tensor): the [K, G, rows, cols, slices] attenuated emission and the [K, 2] sums of the emission and of
            the emission times the height.
    """
    a = _kernel_inputs(library, rho, phi_uncollided, grid, geometry, angle, pairs)
    device = phi_uncollided.device
    f = int(factor)
    Rc, Cc, Sc = grid.coarsen(f).shape
    NC = Rc * Cc * Sc
    directions = omegas.to(device=device, dtype=torch.float32).reshape(-1, 3).contiguous()
    K, G, V = int(directions.shape[0]), a["G"], a["V"]
    for name, t in (("tau_coarse", tau_coarse), ("multiple_collision", multiple_collision)):
        if tuple(t.shape) != (K, G, Rc, Cc, Sc):
            raise ValueError(f"{name} {tuple(t.shape)} != {(K, G, Rc, Cc, Sc)}")
    if tuple(mu_t_coarse.shape) != (G, Rc, Cc, Sc):
        raise ValueError(f"mu_t_coarse {tuple(mu_t_coarse.shape)} != {(G, Rc, Cc, Sc)}")
    tau = tau_coarse.to(device=device, dtype=torch.float32).contiguous()
    multiple = multiple_collision.to(device=device, dtype=torch.float32).contiguous()
    mu = mu_t_coarse.to(device=device, dtype=torch.float32).contiguous()
    h = (torch.zeros(V, device=device, dtype=torch.float32) if heights is None
         else heights.to(device=device, dtype=torch.float32).reshape(-1).contiguous())
    out = torch.empty(K, G, *grid.shape, device=device, dtype=torch.float32)
    num_voxel_blocks = -(-V // BLOCK_V)
    partial = torch.empty(K, num_voxel_blocks, 2, device=device, dtype=torch.float32)
    if K == 0 or V == 0:
        return out, torch.zeros(K, 2, device=device, dtype=torch.float32)
    directions_per_program = _directions_per_program(K, num_voxel_blocks, device)
    dx, dy, dz = (float(d) for d in grid.deltas)
    _attenuated_emission_kernel[(num_voxel_blocks, -(-K // directions_per_program))](
        a["rho"], a["index"], a["phi"], a["x"], a["y"], a["z"], directions, a["table"], a["map"], tau, mu, multiple,
        h, out, partial, V, a["C"], a["S"], NC, Cc, Sc, K, a["N"], a["KM"], *a["r0"], a["eta_lo"], a["eta_hi"],
        dx, dy, dz, num_voxel_blocks,
        G=G, GP=a["GP"], KB=directions_per_program, BLOCK=BLOCK_V, MT=a["MT"], F=f, PAIR=a["PAIR"],
        MAPPED=a["MAPPED"], PARALLEL=a["PARALLEL"], num_warps=NUM_WARPS)
    return out, partial.sum(dim=1)


def fused_kernels_usable(library, rho: Tensor, phi_uncollided: Tensor, pairs=None, backend: str = "auto") -> bool:
    """Decide whether the fused first-collision kernels run for these inputs (``use_kernel`` and ``kernel_supports``).

    Args:
        library (MultigroupLibrary): the library.
        rho (Tensor): [M, rows, cols, slices] densities in g/cm^3.
        phi_uncollided (Tensor): [G, rows, cols, slices] uncollided fluence.
        pairs (MaterialPairs, optional): pair representation of ``rho``, or None for dense mode. Defaults to None.
        backend (str, optional): "auto", "torch" or "triton". Defaults to "auto".

    Returns:
        bool: True when the kernels can run.
    """
    return use_kernel(phi_uncollided, backend) and kernel_supports(library, rho, phi_uncollided, pairs)


def first_collision_source_or_none(library, rho: Tensor, phi_uncollided: Tensor, grid, geometry, angle: float,
                                   omegas: Tensor,
                                   backend: str, pairs=None) -> Optional[Tensor]:
    """Run ``first_collision_source`` when ``use_kernel`` and ``kernel_supports`` allow it, else return None.

    Args:
        library (MultigroupLibrary): the library.
        rho (Tensor): [M, rows, cols, slices] densities in g/cm^3.
        phi_uncollided (Tensor): [G, rows, cols, slices] uncollided fluence.
        grid (Grid): the grid of ``rho`` and ``phi_uncollided``.
        geometry (ConeGeometry or ParallelGeometry): the scan geometry.
        angle (float): view angle in radians.
        omegas (Tensor): [K, 3] unit directions (recon frame).
        backend (str): "auto", "torch" or "triton".
        pairs (MaterialPairs, optional): pair representation of ``rho``, or None for dense mode. Defaults to None.

    Returns:
        Tensor | None: [K, G, rows, cols, slices] first-collision source, or None when the PyTorch path must run.
    """
    if not use_kernel(phi_uncollided, backend) or not kernel_supports(library, rho, phi_uncollided, pairs):
        return None
    return first_collision_source(library, rho, phi_uncollided, grid, geometry, angle, omegas, pairs=pairs)
