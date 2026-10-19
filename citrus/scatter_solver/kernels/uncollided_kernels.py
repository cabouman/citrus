"""Fused Triton kernel of the first-collision source (``uncollided.first_collision_source_at_directions``).

``_first_collision_kernel`` evaluates the first-collision source of every voxel and direction directly from the
transfer table, without forming the interpolated table, in a pair mode (two materials per point, gathered by index) or
a dense mode (all materials). ``prepared_table`` lays out the transfer table for the kernel, and ``kernel_usable``
checks the kernel against the PyTorch path once per device.
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
    return True, (f"kernel matches the PyTorch path, dense (2 and 7 materials) and pair modes, uniform and adaptive "
                  f"tables (max rel dev {worst:.1e})")


def _self_check_table(device: torch.device, angular_table_grid: str) -> tuple[bool, object]:
    """Compare the kernel with the PyTorch path on a tiny random problem for one kind of angular table.

    Args:
        device (torch.device): the CUDA device to check.
        angular_table_grid (str): "uniform" or "adaptive".

    Returns:
        (bool, object): (True, largest relative deviation) or (False, reason).
    """
    from ..cross_sections import material_pairs, synthetic_library
    from ..geometry import ConeGeometry, Grid
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
    torch.cuda.synchronize(device)
    worst = 0.0
    for o, r in ((out, reference), (out7, reference7), (out7d, reference7d), (out_helical, reference_helical)):
        if o.shape != r.shape or not bool(torch.isfinite(o).all()):
            return False, "wrong shape or non-finite output"
        worst = max(worst, float((o - r).abs().max()) / max(float(r.abs().max()), 1e-30))
    if worst > SELF_CHECK_REL_TOL:
        return False, f"kernel deviates from the PyTorch path by {worst:.2e} (rel)"
    return True, worst


# ================================================================================================
# kernel
# ================================================================================================

if triton is not None:

    @triton.jit
    def _first_collision_kernel(rho_ptr, index_ptr, phi_ptr, x_axis_ptr, y_axis_ptr, z_axis_ptr, direction_ptr,
                                table_ptr, map_ptr, out_ptr,
                                V, C, S, K, N, KM, r0x, r0y, r0z, eta_lo, eta_hi,
                                G: tl.constexpr, GP: tl.constexpr, KB: tl.constexpr, BLOCK: tl.constexpr,
                                MT: tl.constexpr, PAIR: tl.constexpr, MAPPED: tl.constexpr):
        """First-collision source of BLOCK voxels (program_id 0) for KB directions (program_id 1).

        Args:
            rho_ptr: float32 densities in g/cm^3, [MT, V] in dense mode or [2, V] slot densities in pair mode.
            index_ptr: [2, V] int32 material index of each slot (pair mode).
            phi_ptr: [G, V] float32 uncollided fluence.
            x_axis_ptr: [C] float32 x of the voxel centers (columns) in cm.
            y_axis_ptr: [R] float32 y of the voxel centers (rows) in cm.
            z_axis_ptr: [S] float32 z of the voxel centers (slices) in cm.
            direction_ptr: [K, 3] float32 unit directions.
            table_ptr: [N, G, MT, GP] float32 transfer table from ``prepared_table``.
            map_ptr: [KM] float32 node map of an adaptive table (MAPPED).
            out_ptr: [K, G, V] float32 first-collision source.
            V: number of voxels.
            C: number of columns.
            S: number of slices.
            K: number of directions.
            N: number of eta nodes of the table.
            KM: number of knots of the node map.
            r0x: x of the focal spot in cm.
            r0y: y of the focal spot in cm.
            r0z: z of the focal spot in cm.
            eta_lo: first node of the eta grid.
            eta_hi: last node of the eta grid.
            G (constexpr): number of energy groups.
            GP (constexpr): next power of two >= G.
            KB (constexpr): directions per program.
            BLOCK (constexpr): voxels per program.
            MT (constexpr): number of materials of the table.
            PAIR (constexpr): pair mode (two materials per voxel by index) instead of dense mode.
            MAPPED (constexpr): adaptive table read through ``map_ptr``; else uniform in eta.
        """
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
        km1 = (KM - 1).to(tl.float32)
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
        (table, eta_lo, eta_hi, map): table is the [N, G_in, M, GP] float32 transfer table in cm^2/g/sr, zero where
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
    kernel_table = torch.zeros(N, G, M, GP, device=device, dtype=torch.float32)
    kernel_table[..., :G] = t.permute(3, 1, 0, 2)
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


def first_collision_source(library, rho: Tensor, phi_uncollided: Tensor, grid, geometry, angle: float, omegas: Tensor,
                           pairs=None) -> Tensor:
    """Fused ``uncollided.first_collision_source_at_directions`` on float32 CUDA tensors, without availability check.

    Args:
        library (MultigroupLibrary): the library; uses ``mu_s_eta`` [M, G, G, N], ``eta_grid`` and ``eta_map``.
        rho (Tensor): [M, rows, cols, slices] densities in g/cm^3.
        phi_uncollided (Tensor): [G, rows, cols, slices] float32 CUDA uncollided fluence.
        grid (Grid): the grid of ``rho`` and ``phi_uncollided``.
        geometry (ConeGeometry): the cone-beam geometry.
        angle (float): view angle in radians.
        omegas (Tensor): [K, 3] unit directions (recon frame).
        pairs (MaterialPairs, optional): pair representation of ``rho`` (pair mode), or None for dense mode.
            Defaults to None.

    Returns:
        Tensor: [K, G, rows, cols, slices] float32 first-collision source in photons/(cm^3 sr) per source photon/sr.
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
    table, eta_lo, eta_hi, kernel_map = prepared_table(library, device)
    x, y, z = grid.axis_coordinates(device, dtype)
    r0 = geometry.source_position(angle)
    num_voxel_blocks = -(-V // BLOCK_V)
    directions_per_program = _directions_per_program(K, num_voxel_blocks, device)
    _first_collision_kernel[(num_voxel_blocks, -(-K // directions_per_program))](
        rho_flat, index_flat, phi_flat, x, y, z, directions, table, table if kernel_map is None else kernel_map, out,
        V, C, S, K, int(table.shape[0]), 2 if kernel_map is None else int(kernel_map.shape[0]),
        float(r0[0]), float(r0[1]), float(r0[2]), eta_lo, eta_hi,
        G=G, GP=_next_pow2(G), KB=directions_per_program, BLOCK=BLOCK_V, MT=int(table.shape[2]), PAIR=pairs is not None,
        MAPPED=kernel_map is not None, num_warps=NUM_WARPS)
    return out


def first_collision_source_or_none(library, rho: Tensor, phi_uncollided: Tensor, grid, geometry, angle: float,
                                   omegas: Tensor,
                                   backend: str, pairs=None) -> Optional[Tensor]:
    """Run ``first_collision_source`` when ``use_kernel`` and ``kernel_supports`` allow it, else return None.

    Args:
        library (MultigroupLibrary): the library.
        rho (Tensor): [M, rows, cols, slices] densities in g/cm^3.
        phi_uncollided (Tensor): [G, rows, cols, slices] uncollided fluence.
        grid (Grid): the grid of ``rho`` and ``phi_uncollided``.
        geometry (ConeGeometry): the cone-beam geometry.
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
