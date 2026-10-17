"""Uncollided fluence and first-collision source of the LBTE solver.

For a point source at r0 emitting q_g = f_g b(Omega) photons per steradian in group g, the uncollided fluence and
the first-collision source are

    phi_UC,g(r)       = f_g b(Omega_UC(r)) / |r - r0|^2  exp(-tau_g(r0 -> r)),     Omega_UC(r) = (r - r0) / |r - r0|
    S_FC,g(r, Omega)  = sum_{g' <= g} mu_s^{g' -> g}(r; Omega_UC(r) . Omega) phi_UC,g'(r)

(per-steradian transfer mu_s(eta) tabulated in eta = cos(scattering angle)). All functions work for one view angle
in the recon frame and are device- and dtype-preserving (the dtype of `rho` / `phi_uncollided` decides the computation
dtype). The evaluation points are the voxel centers of the `grid` argument: in the solver the quadrature grid Q (q^3
points per transport cell, docs/source/conventions.rst, "Grids"), with the densities resampled onto Q.

The uncollided fluence is exact in energy within each group: upstream ray traces of the density maps (one per basis
material present in the volume, `present_materials`) and the library's fine-energy attenuation table
(`cross_sections.group_transmission`), not the group-averaged mu_t. The trace runs through any grid to the points of
any other (`uncollided_fluence(..., points=)`): the solver traces through the quadrature grid Q by default
(SolverConfig.uncollided_trace_grid). It is the composition of `uncollided_path_lengths` (the per-material upstream
line integrals, which can be traced one material map at a time: the solver does so for fields built from the
attenuation, `ScatterSolver.material_fields_from_attenuation`) and `uncollided_from_path_lengths`.

Materials: rho is [M, R, C, S] for any number of basis materials M (7 for the package's table). Where every point has
at most two nonzero materials, the first-collision source uses the compact pair representation
(`cross_sections.MaterialPairs`: the two table rows gathered by material index; the cost of a two-material basis);
otherwise (resampled densities can mix more materials per point) and for M <= 2 it evaluates all M materials densely
(`pairs` argument: a MaterialPairs, `DENSE`, or None to decide from rho, `resolve_pairs`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from .cross_sections import (MaterialPairs, MultigroupLibrary, group_transmission, material_pairs_if_compact,
                             transfer_eta_at, transfer_eta_at_pairs)
from .geometry import ConeGeometry, Grid
from .kernels import uncollided_kernels
from .ray_tracing import Projectors
from .source import SourceModel


def present_materials(rho: Tensor) -> Optional[list[int]]:
    """Library indices of the materials with any nonzero density in rho [M, ...] when M > 2 and some material is
    absent (one host synchronization), else None (= all materials). An absent material has zero line integrals, so
    tracing only the present ones gives the same transmission (`cross_sections.group_transmission(materials=...)`)."""
    M = int(rho.shape[0])
    if M <= 2:
        return None
    present = [int(i) for i in torch.nonzero(rho.reshape(M, -1).ne(0).any(1)).flatten().tolist()]
    if len(present) == M:
        return None
    return present if present else [0]


# `pairs` value selecting the dense evaluation over all M materials
DENSE = "dense"
PairsArgument = Union[MaterialPairs, str, None]


def resolve_pairs(rho: Tensor, pairs: PairsArgument) -> Union[MaterialPairs, str]:
    """The material representation of the first-collision source: `pairs` itself when it is a MaterialPairs or `DENSE`;
    for None, `cross_sections.material_pairs_if_compact(rho)` (the pair representation when M > 2 and every point has at
    most two nonzero materials; one host synchronization) or DENSE. Resolve once and pass the result when calling
    repeatedly with the same densities."""
    if isinstance(pairs, MaterialPairs):
        return pairs
    if pairs is None:
        compact = material_pairs_if_compact(rho)
        return DENSE if compact is None else compact
    if pairs != DENSE:
        raise ValueError(f"pairs must be a MaterialPairs, {DENSE!r} or None, got {pairs!r}")
    return DENSE


def primary_directions_to_voxels(grid: Grid, geometry: ConeGeometry, angle: float, device=None,
                                 dtype=torch.float32) -> Tensor:
    """Unit vectors Omega_UC(r) = (r - r0)/|r - r0| from the focal spot to each voxel center.

    Returns [rows, cols, slices, 3] (recon frame).
    """
    r = grid.center_coordinates(device, dtype)
    r0 = torch.tensor(geometry.source_position(angle), device=device, dtype=dtype)
    d = r - r0
    return d / torch.linalg.norm(d, dim=-1, keepdim=True)


def uncollided_fluence(library: MultigroupLibrary, rho: Tensor, grid: Grid, geometry: ConeGeometry, angle: float,
                       source: SourceModel, projectors: Projectors, points: Optional[Grid] = None) -> Tensor:
    """Uncollided scalar fluence at the voxel centers of `points` (default: of `grid`), exact in energy within each
    group, traced through the densities rho on `grid`.

        phi_UC,g(r) = b(Omega_UC(r)) / |r - r0|^2 * sum_{E in g} w_E exp(-sum_m P_m(r) m_m(E))

    with P_m(r) = int_{r0}^{r} rho_m dl the upstream line integrals of the density maps on `grid` (g/cm^2, zero outside
    its box; one ray trace per material PRESENT in the volume, `present_materials`: an absent material adds nothing):
    `projectors.upstream_cumulative(rho[present], grid)` when the points are the centers of `grid` itself, else
    `projectors.upstream_to_points(rho[present], grid, ..., points centers)` (one ray per point: the number of rays is
    the number of points, the path cost that of `grid`; a point outside the box gets the integral of the part of its
    ray inside the box). The library's fine-energy table: w_E = library.fine_weight
    (source photons/sr per fine bin; their sum over group g is f_g = source.per_group_strength), m_m(E) =
    library.fine_mu_t (`cross_sections.group_transmission`). Since mu_t is linear in the densities this is the exact
    group-integrated polychromatic Beer's law (up to the fine quadrature); the group-averaged attenuation
    f_g exp(-int mu_t,g dl) ignores beam hardening within a group (-1.4% after 20 cm of water with 6 groups).

    rho: [M, rows, cols, slices] g/cm^3 on grid. Returns phi_uncollided [G, *points.shape] in photons/cm^2 per source
    photon/sr (point values at the voxel centers of `points`).
    """
    if int(rho.shape[0]) != library.num_materials:
        raise ValueError(f"rho has {int(rho.shape[0])} materials, the library {library.num_materials}")
    point_grid = grid if points is None else points
    present = present_materials(rho)
    traced_rho = rho if present is None else rho[present]
    # [M', *point_grid.shape] g/cm^2
    path_lengths = uncollided_path_lengths(projectors, traced_rho, grid, geometry, angle, point_grid)
    return uncollided_from_path_lengths(library, path_lengths, present, point_grid, geometry, angle, source)


def uncollided_path_lengths(projectors: Projectors, rho: Tensor, grid: Grid, geometry: ConeGeometry, angle: float,
                            points: Optional[Grid] = None) -> Tensor:
    """Upstream density line integrals P_m(r) = int_{r0}^{r} rho_m dl (g/cm^2) of the maps rho [M', R, C, S] on `grid`
    (any subset of the materials, e.g. one map at a time) to the voxel centers of `points` (default: of `grid`):
    `projectors.upstream_cumulative` when the points are the centers of `grid`, else `projectors.upstream_to_points`.
    Returns [M', *points.shape]. Linear and channel by channel: tracing the materials one at a time and concatenating
    gives the same path lengths."""
    point_grid = grid if points is None else points
    if point_grid == grid:
        return projectors.upstream_cumulative(rho, grid, geometry, angle)  # [M', R, C, S] g/cm^2
    r = point_grid.center_coordinates(rho.device, rho.dtype)
    return projectors.upstream_to_points(rho, grid, geometry, angle, r.reshape(-1, 3)).reshape(-1, *point_grid.shape)


def uncollided_from_path_lengths(library: MultigroupLibrary, path_lengths: Tensor, materials: Optional[Sequence[int]],
                                 points: Grid, geometry: ConeGeometry, angle: float, source: SourceModel) -> Tensor:
    """phi_UC,g(r) = b(Omega_UC(r)) / |r - r0|^2 sum_{E in g} w_E exp(-sum_m P_m(r) m_m(E)) at the voxel centers of
    `points` from the upstream line integrals path [M', *points.shape] (g/cm^2; `uncollided_path_lengths`) of the
    materials `materials` (library indices of the rows; None = all, in library order). Returns [G, *points.shape]."""
    device, dtype = path_lengths.device, path_lengths.dtype
    transmission = group_transmission(library, path_lengths, materials=materials)  # [G, R, C, S]
    r = points.center_coordinates(device, dtype)
    r0 = torch.tensor(geometry.source_position(angle), device=device, dtype=dtype)
    d = r - r0
    distance_squared = (d * d).sum(-1)  # [R, C, S]
    b = source.angular_weight(d, geometry, angle)  # [R, C, S]
    return transmission.mul_((b / distance_squared)[None])


def first_collision_source_at_directions(library: MultigroupLibrary, rho: Tensor, phi_uncollided: Tensor, grid: Grid,
                                         geometry: ConeGeometry, angle: float, omegas: Tensor, chunk: int = 4,
                                         backend: str = "auto", pairs: PairsArgument = None) -> Tensor:
    """First-collision source at every voxel center of `grid` (in the solver the quadrature grid Q) for each
    direction.

        S_FC,g(r, omega_k) = sum_{g' <= g} mu_s^{g'->g}(r; Omega_UC(r) . omega_k) phi_UC,g'(r)

    Groups are ordered from high to low energy, so g' <= g is downscatter (no upscatter).
    rho: [M, rows, cols, slices]; phi_uncollided: [G, rows, cols, slices]; omegas: [K, 3] unit vectors.
    Returns [K, G, rows, cols, slices] in photons/(cm^3 sr) per source photon/sr.

    backend: "auto" (default) runs float32 CUDA inputs through the fused Triton kernel of
    `kernels/uncollided_kernels.py` when it is usable on the device (shared Triton probe, CITRUS_DISABLE_TRITON=1 kill
    switch, per-device self-check against the PyTorch path), else the PyTorch path; "torch" forces the PyTorch path;
    "triton" requires the kernel for float32 CUDA inputs (raises if it is unusable). float64 and CPU inputs always use
    the PyTorch path (the reference). The two paths agree to float32 rounding (~1e-6 relative).
    chunk: directions per batch of the PyTorch path (peak memory ~ 3 M G^2 chunk |F| scalars); the kernel
    evaluates all K directions in one launch without temporaries, so it ignores `chunk`.
    pairs: the material representation (`resolve_pairs`): a `cross_sections.MaterialPairs` of rho (at most two
    nonzero materials per point; both paths gather the two table rows of each point by material index: cost and
    memory of a two-material basis), `DENSE` (all M materials per point: the PyTorch path forms the M-material mix, the
    kernel's dense mode costs about M / 2 times its pair mode), or None (decided from rho here, one host
    synchronization: pass the resolved value when calling repeatedly with the same densities).
    """
    pairs = resolve_pairs(rho, pairs)
    compact_pairs = pairs if isinstance(pairs, MaterialPairs) else None
    fused = uncollided_kernels.first_collision_source_or_none(library, rho, phi_uncollided, grid, geometry, angle,
                                                              omegas, backend, pairs=compact_pairs)
    if fused is not None:
        return fused
    if compact_pairs is not None:
        return _first_collision_source_torch_pairs(library, compact_pairs, phi_uncollided, grid, geometry, angle,
                                                   omegas, chunk)
    return _first_collision_source_torch(library, rho, phi_uncollided, grid, geometry, angle, omegas, chunk)


def _first_collision_source_torch(library: MultigroupLibrary, rho: Tensor, phi_uncollided: Tensor, grid: Grid,
                                  geometry: ConeGeometry, angle: float, omegas: Tensor, chunk: int = 4) -> Tensor:
    """PyTorch path of `first_collision_source_at_directions` (any device / dtype; same shapes): materializes the
    interpolated transfer table [G_in, G_out, chunk, rows, cols, slices] per `chunk` directions."""
    device, dtype = phi_uncollided.device, phi_uncollided.dtype
    rho = rho.to(device=device, dtype=dtype)
    omegas = omegas.to(device=device, dtype=dtype).reshape(-1, 3)
    K = omegas.shape[0]
    G = phi_uncollided.shape[0]
    omega_uncollided = primary_directions_to_voxels(grid, geometry, angle, device, dtype)  # [R, C, S, 3]
    downscatter = torch.triu(torch.ones(G, G, device=device, dtype=dtype))  # [g_in, g_out], 1 where g_out >= g_in
    out = torch.empty(K, G, *phi_uncollided.shape[1:], device=device, dtype=dtype)
    M = rho.shape[0]
    for start in range(0, K, max(1, int(chunk))):
        directions = omegas[start:start + chunk]  # [k, 3]
        k = directions.shape[0]
        eta = torch.einsum("rcsx,kx->krcs", omega_uncollided, directions)  # [k, R, C, S]
        rho_k = rho[:, None].expand(M, k, *rho.shape[1:])
        T = transfer_eta_at(library, rho_k, eta)  # [G_in, G_out, k, R, C, S]
        out[start:start + k] = torch.einsum("abk...,ab,a...->kb...", T, downscatter, phi_uncollided)
    return out


def _first_collision_source_torch_pairs(library: MultigroupLibrary, pairs: MaterialPairs, phi_uncollided: Tensor,
                                        grid: Grid, geometry: ConeGeometry, angle: float, omegas: Tensor,
                                        chunk: int = 4) -> Tensor:
    """PyTorch path of `first_collision_source_at_directions` for the pair representation (any device / dtype): per
    `chunk` directions the two table rows of every voxel are gathered by material index
    (`cross_sections.transfer_eta_at_pairs`)."""
    device, dtype = phi_uncollided.device, phi_uncollided.dtype
    pairs = pairs.to(device=device, dtype=dtype)
    omegas = omegas.to(device=device, dtype=dtype).reshape(-1, 3)
    K = omegas.shape[0]
    G = phi_uncollided.shape[0]
    omega_uncollided = primary_directions_to_voxels(grid, geometry, angle, device, dtype)  # [R, C, S, 3]
    downscatter = torch.triu(torch.ones(G, G, device=device, dtype=dtype))
    out = torch.empty(K, G, *phi_uncollided.shape[1:], device=device, dtype=dtype)
    for start in range(0, K, max(1, int(chunk))):
        directions = omegas[start:start + chunk]
        k = directions.shape[0]
        eta = torch.einsum("rcsx,kx->krcs", omega_uncollided, directions)  # [k, R, C, S]
        T = transfer_eta_at_pairs(library, pairs, eta)  # [G_in, G_out, k, R, C, S]
        out[start:start + k] = torch.einsum("abk...,ab,a...->kb...", T, downscatter, phi_uncollided)
    return out


def restrict_to_linear_discontinuous_moments(field: Tensor, grid: Grid, factor: int) -> Tensor:
    """Restrict a field on a refined grid (the quadrature grid Q = C.refine(factor)) to linear-discontinuous moments on
    the grid it refines (the transport grid C).

    field: [..., rows, cols, slices] on `grid` (typically [K, G, rows, cols, slices]).
    Returns [..., 4, rows/f, cols/f, slices/f] (f = factor) with moments ordered [0, x, y, z] (x <-> cols,
    y <-> rows, z <-> slices), using the f^3 sub-cell centers as equal-weight quadrature points
    with local coordinates u_j = (2j + 1)/f - 1 (grid.local_coordinates_of_fine_in_coarse):

      moment 0 = mean_j S_j
      moment a = mean_j(S_j u_a,j) / mean_j(u_a,j^2)

    The midpoint quadrature gives mean_j(u_j^2) = (1 - 1/f^2)/3, not 1/3, so the continuous Galerkin
    formula (3 mean(S u)) would return s (1 - 1/f^2) for a field c + s u sampled at the sub-cell centers.
    Dividing by the discrete second moment instead reproduces a field that is linear within each
    cell exactly (slope s, mean c). For f = 1 the slopes are 0.
    """
    f = int(factor)
    coarse = grid.coarsen(f)  # validates divisibility
    if tuple(field.shape[-3:]) != tuple(grid.shape):
        raise ValueError(f"field spatial shape {tuple(field.shape[-3:])} != grid shape {grid.shape}")
    lead = field.shape[:-3]
    R, C, S = coarse.shape
    x = field.reshape(*lead, R, f, C, f, S, f)
    num_leading = len(lead)
    # sub-voxel axes for rows (y), cols (x), slices (z)
    axis_rows, axis_cols, axis_slices = num_leading + 1, num_leading + 3, num_leading + 5
    mean = x.mean(dim=(axis_rows, axis_cols, axis_slices))
    if f == 1:
        zero = torch.zeros_like(mean)
        return torch.stack([mean, zero, zero, zero], dim=-4)
    u = grid.local_coordinates_of_fine_in_coarse(f, device=field.device, dtype=field.dtype)  # [f]
    scale = 1.0 / (u * u).mean()

    def slope(axis: int) -> Tensor:
        shape = [1] * x.dim()
        shape[axis] = f
        return (x * u.reshape(shape)).mean(dim=(axis_rows, axis_cols, axis_slices)) * scale

    return torch.stack([mean, slope(axis_cols), slope(axis_rows), slope(axis_slices)], dim=-4)


# =====================================================================================================
# conservative sampling of the first-collision source at the sweep directions (docs/source/design_notes.rst)
# =====================================================================================================


def first_collision_exact_moments(library: MultigroupLibrary, rho: Tensor, phi_uncollided: Tensor,
                                  l_max: int = 1) -> Tensor:
    """Exact angular moments of the first-collision source about Omega_UC from the library Legendre moments.

        M_l[g](r) = int P_l(Omega . Omega_UC(r)) S_FC,g(r, Omega) dOmega = sum_{g' <= g} mu_s_l^{g'->g}(r) phi_UC,g'(r)

    (convention mu_s(eta) = sum_l (2l+1)/(4 pi) mu_s_l P_l(eta)), l = 0..l_max. rho: [M, R, C, S];
    phi_uncollided: [G, R, C, S]. Returns [l_max+1, G, R, C, S] in the dtype/device of phi_uncollided.
    """
    if library.legendre_order < l_max:
        raise ValueError(f"library Legendre order {library.legendre_order} < {l_max}")
    device, dtype = phi_uncollided.device, phi_uncollided.dtype
    G = phi_uncollided.shape[0]
    transfer = library.mu_s_l[..., :l_max + 1].to(device=device, dtype=dtype)  # [M, Gi, Go, l]
    downscatter = torch.triu(torch.ones(G, G, device=device, dtype=dtype))
    return torch.einsum("mabl,ab,m...,a...->lb...", transfer, downscatter, rho.to(device=device, dtype=dtype),
                        phi_uncollided)


@dataclass(frozen=True, eq=False)
class FirstCollisionSamplingCorrection:
    """Per-point, per-group correction of the pointwise-sampled first-collision source (on the grid of its points, in
    the solver the quadrature grid Q):

        S_m <- scale * S_m * (1 + beta * Omega_m . Omega_UC)

    scale, beta: [G, R, C, S]; omega_uncollided: [R, C, S, 3]. The factors make the quadrature-weighted l = 0 and l = 1
    (unless beta had to be clamped to [-1, 1] to keep the source non-negative) moments of the corrected source equal to
    `first_collision_exact_moments` (see `first_collision_sampling_correction`).
    """

    scale: Tensor
    beta: Tensor
    omega_uncollided: Tensor

    def apply(self, source: Tensor, omegas: Tensor) -> Tensor:
        """source: [K, G, R, C, S] = first_collision_source_at_directions(..., omegas [K, 3]). Returns the
        corrected source."""
        directions = omegas.to(device=source.device, dtype=source.dtype).reshape(-1, 3)
        eta = torch.einsum("rcsx,kx->krcs", self.omega_uncollided, directions)  # [K, R, C, S]
        return source * self.scale * (1.0 + self.beta[None] * eta[:, None])


def first_collision_sampling_correction(library: MultigroupLibrary, rho: Tensor, phi_uncollided: Tensor, grid: Grid,
                                        geometry: ConeGeometry, angle: float, omegas: Tensor, weights: Tensor,
                                        chunk: int = 4, batch: int = 8,
                                        pairs: PairsArgument = None) -> FirstCollisionSamplingCorrection:
    """Correction factors that make the quadrature (omegas [K, 3], weights [K]) integrate the sampled first-collision
    source exactly in angle, l = 0 and l = 1 about Omega_UC (one extra pass of `first_collision_source_at_directions`
    over all K directions, `batch` at a time).

    With the quadrature sums (per voxel and group, mu_m = Omega_m . Omega_UC(r), S_m the point-sampled source)
        I0 = sum_m w_m S_m,  I1 = sum_m w_m mu_m S_m,  I2 = sum_m w_m mu_m^2 S_m
    and the exact moments E0, E1 (`first_collision_exact_moments`): with c = E1 / E0, beta = (c I0 - I1) / (I2 - c I1)
    clamped to [-1, 1] (beta = 0 where E0 == 0 or the denominator vanishes), scale = E0 / (I0 + beta I1) (1 where that
    denominator is not positive). The l = 0 moment is always matched; l = 1 is matched unless beta was clamped
    (the clamp keeps 1 + beta mu >= 0, i.e. a non-negative source).
    Precision (`first_collision_quadrature_sums`): the sums are formed by matmuls in the input dtype (float32 on GPU)
    over blocks of at most `_FIRST_COLLISION_SUM_MAX_DIRECTIONS` directions and accumulated across blocks in float64;
    the factors are computed in float64. Returns a FirstCollisionSamplingCorrection in the dtype of phi_uncollided.
    pairs: as in `first_collision_source_at_directions`.
    """
    device, dtype = phi_uncollided.device, phi_uncollided.dtype
    omegas = omegas.to(device=device, dtype=dtype).reshape(-1, 3)
    weights = weights.to(device=device, dtype=dtype).reshape(-1)
    if weights.shape[0] != omegas.shape[0]:
        raise ValueError("weights and omegas disagree in length")
    omega_uncollided = primary_directions_to_voxels(grid, geometry, angle, device, dtype)
    sums = first_collision_quadrature_sums(library, rho, phi_uncollided, grid, geometry, angle, omegas, weights,
                                           omega_uncollided=omega_uncollided, chunk=chunk, batch=batch,
                                           pairs=resolve_pairs(rho, pairs))
    exact_moments = first_collision_exact_moments(library, rho, phi_uncollided.double(), l_max=1)
    scale, beta = first_collision_correction_factors(sums, exact_moments)
    return FirstCollisionSamplingCorrection(scale=scale.to(dtype), beta=beta.to(dtype),
                                            omega_uncollided=omega_uncollided)


# at most this many directions are summed in the input dtype (float32 on GPU) before a float64 accumulation
_FIRST_COLLISION_SUM_MAX_DIRECTIONS = 256


def _moment_weight_rows(omegas: Tensor, weights: Tensor) -> Tensor:
    """[10, K] rows (w, w o_x, w o_y, w o_z, w o_x^2, w o_y^2, w o_z^2, 2 w o_x o_y, 2 w o_x o_z, 2 w o_y o_z) such
    that, for P = rows @ S ([10, K] @ [K, n]) and u = Omega_UC, I0 = P_0, I1 = sum_a u_a P_{1+a} and
    I2 = sum_m w_m (omega_m . u)^2 S_m = u_x^2 P_4 + u_y^2 P_5 + u_z^2 P_6 + u_x u_y P_7 + u_x u_z P_8 + u_y u_z P_9."""
    x, y, z = omegas.unbind(-1)
    w2 = 2.0 * weights
    return torch.stack([weights, weights * x, weights * y, weights * z, weights * x * x, weights * y * y,
                        weights * z * z, w2 * x * y, w2 * x * z, w2 * y * z])


def first_collision_quadrature_sums(library: MultigroupLibrary, rho: Tensor, phi_uncollided: Tensor, grid: Grid,
                                    geometry: ConeGeometry, angle: float, omegas: Tensor, weights: Tensor,
                                    omega_uncollided: Tensor | None = None, chunk: int = 4, batch: int = 8,
                                    pairs: PairsArgument = None) -> Tensor:
    """Quadrature sums of the point-sampled first-collision source per voxel and group (see
    `first_collision_sampling_correction`):

        I0 = sum_m w_m S_m,  I1 = sum_m w_m mu_m S_m,  I2 = sum_m w_m mu_m^2 S_m,   mu_m = omega_m . Omega_UC(r)

    Returns [3, G, R, C, S] float64. One pass of `first_collision_source_at_directions` over the K directions, `batch`
    at a time (this sets the size of the [batch, G, R, C, S] source temporary). Since mu_m is linear in Omega_UC, a
    batch's contribution to all three sums is ONE matmul (`addmm`) of the [10, k] weight rows of `_moment_weight_rows`
    (quadrature weights times the monomials of omega_m up to degree 2) with the [k, G V] source, accumulated into a
    [10, G V] partial in the input dtype. Every `_FIRST_COLLISION_SUM_MAX_DIRECTIONS` directions (and at the end) the
    partial is contracted with the per-voxel monomials of Omega_UC (1 + 3 + 6 rows -> I0, I1, I2) and added to the
    float64 accumulators, so a float32 sum runs over at most `_FIRST_COLLISION_SUM_MAX_DIRECTIONS` directions and the
    sum across those blocks keeps float64. float64 inputs are summed in float64 throughout. (The float32 matmul assumes
    TF32 is off, `torch.backends.cuda.matmul.allow_tf32 = False`, the PyTorch default.) pairs: as in
    `first_collision_source_at_directions`.
    """
    pairs = resolve_pairs(rho, pairs)
    device, dtype = phi_uncollided.device, phi_uncollided.dtype
    omegas = omegas.to(device=device, dtype=dtype).reshape(-1, 3)
    weights = weights.to(device=device, dtype=dtype).reshape(-1)
    if omega_uncollided is None:
        omega_uncollided = primary_directions_to_voxels(grid, geometry, angle, device, dtype)
    G = phi_uncollided.shape[0]
    u = omega_uncollided.to(device=device, dtype=dtype).reshape(-1, 3).t()  # [3, V]
    V = u.shape[1]
    weight_rows = _moment_weight_rows(omegas, weights)  # [10, K]
    sums = torch.zeros(3, G, V, device=device, dtype=torch.float64)
    partial = torch.zeros(10, G * V, device=device, dtype=dtype)  # input-dtype partial sums of the current block

    def flush() -> None:
        partial_view = partial.view(10, G, V)
        ux, uy, uz = u[0], u[1], u[2]
        u2 = torch.stack([ux * ux, uy * uy, uz * uz, ux * uy, ux * uz, uy * uz])
        sums[0] += partial_view[0]
        sums[1] += (partial_view[1:4] * u[:, None]).sum(0)
        sums[2] += (partial_view[4:10] * u2[:, None]).sum(0)
        partial.zero_()

    batch_size = max(1, min(int(batch), _FIRST_COLLISION_SUM_MAX_DIRECTIONS))
    K = omegas.shape[0]
    directions_in_block = 0
    for s in range(0, K, batch_size):
        directions = omegas[s:s + batch_size]
        k = directions.shape[0]
        if directions_in_block + k > _FIRST_COLLISION_SUM_MAX_DIRECTIONS:
            flush()
            directions_in_block = 0
        source = first_collision_source_at_directions(library, rho, phi_uncollided, grid, geometry, angle, directions,
                                                      chunk=chunk, pairs=pairs)
        partial.addmm_(weight_rows[:, s:s + k], source.reshape(k, G * V))
        directions_in_block += k
        del source
    flush()
    return sums.reshape(3, *phi_uncollided.shape)


def first_collision_correction_factors(I: Tensor, E: Tensor) -> tuple[Tensor, Tensor]:
    """(scale, beta) of `first_collision_sampling_correction` from the quadrature sums I = [I0, I1, I2]
    (`first_collision_quadrature_sums`) and the exact moments E = [E0, E1] (`first_collision_exact_moments`, l_max = 1),
    computed in float64; returned in float64."""
    I = I.double()
    E = E.to(device=I.device, dtype=torch.float64)
    I0, I1, I2 = I[0], I[1], I[2]
    E0, E1 = E[0], E[1]
    pos = E0 > 0
    c = torch.where(pos, E1 / torch.where(pos, E0, torch.ones_like(E0)), torch.zeros_like(E0))
    den = I2 - c * I1
    ok = pos & (den.abs() > 1e-12 * I0.abs())
    beta = torch.where(ok, (c * I0 - I1) / torch.where(ok, den, torch.ones_like(den)), torch.zeros_like(den))
    beta = beta.clamp(-1.0, 1.0)
    d0 = I0 + beta * I1
    scale = torch.where(d0 > 0, E0 / torch.where(d0 > 0, d0, torch.ones_like(d0)), torch.ones_like(d0))
    return scale, beta
