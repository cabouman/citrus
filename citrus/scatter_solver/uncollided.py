"""Uncollided fluence and first-collision source of the LBTE solver, for one view.

The uncollided scalar fluence is exact in energy within each group: it is computed from upstream ray traces of the
density maps and the fine-energy attenuation table of the library. The first-collision source is evaluated from the
tabulated per-steradian transfer at the angle between the primary direction and each sweep direction, and the module
also restricts fields on the quadrature grid to linear-discontinuous moments and corrects the point-sampled
first-collision source so that the sweep quadrature reproduces its exact l = 0 and l = 1 moments. The equations are in
docs/source/method.rst, "Uncollided fluence and first-collision source" and "Source iteration".
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
    """Library indices of the materials with any nonzero density, or None for all materials.

    Tracing only the present materials gives the same transmission, since an absent material has zero line integrals;
    the test needs one host synchronization.

    Args:
        rho (Tensor): [M, ...] material densities in g/cm^3.

    Returns:
        list[int] | None: indices of the present materials ([0] if none), or None when M <= 2 or all are present.
    """
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
    """Material representation of the first-collision source: a ``MaterialPairs`` or ``DENSE``.

    Args:
        rho (Tensor): [M, rows, cols, slices] material densities in g/cm^3.
        pairs (MaterialPairs | str | None): returned as is when a ``MaterialPairs`` or ``DENSE``; None chooses the pair
            representation when M > 2 and every point has at most two nonzero materials (one host synchronization).

    Returns:
        MaterialPairs | str: the pair representation of rho, or ``DENSE``.
    """
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
    """Unit vectors ``Omega_UC(r) = (r - r0) / |r - r0|`` from the focal spot to each voxel center.

    For a parallel beam every voxel has the beam direction.

    Args:
        grid (Grid): grid of the voxel centers.
        geometry (ConeGeometry or ParallelGeometry): scanner geometry.
        angle (float): view angle in radians.
        device (torch.device | str, optional): device of the result. Defaults to None (CPU).
        dtype (torch.dtype, optional): dtype of the result. Defaults to torch.float32.

    Returns:
        Tensor: [rows, cols, slices, 3] unit directions in the reconstruction frame.
    """
    if geometry.is_parallel:
        beam = torch.tensor(geometry.beam_direction(angle), device=device, dtype=dtype)
        return beam.expand(*grid.shape, 3).contiguous()
    r = grid.center_coordinates(device, dtype)
    r0 = torch.tensor(geometry.source_position(angle), device=device, dtype=dtype)
    d = r - r0
    return d / torch.linalg.norm(d, dim=-1, keepdim=True)


def uncollided_fluence(library: MultigroupLibrary, rho: Tensor, grid: Grid, geometry: ConeGeometry, angle: float,
                       source: SourceModel, projectors: Projectors, points: Optional[Grid] = None) -> Tensor:
    """Uncollided scalar fluence at the voxel centers of ``points``, exact in energy within each group.

    ``phi_UC,g(r) = b(Omega_UC(r)) / |r - r0|^2 sum_{E in g} w_E exp(-sum_m P_m(r) m_m(E))``, with the upstream density
    line integrals P_m traced through ``grid`` for the materials present in the volume.

    Args:
        library (MultigroupLibrary): cross-section library with the fine-energy weights and mass attenuation.
        rho (Tensor): [M, rows, cols, slices] material densities rho in g/cm^3 on ``grid``.
        grid (Grid): grid of the densities, traced through.
        geometry (ConeGeometry or ParallelGeometry): scanner geometry.
        angle (float): view angle in radians.
        source (SourceModel): source model giving the angular weight b.
        projectors (Projectors): ray tracers of the upstream line integrals.
        points (Grid, optional): grid whose voxel centers receive the fluence. Defaults to None (``grid``).

    Returns:
        Tensor: [G, rows, cols, slices] fluence on ``points`` in photons/cm^2 per source photon/sr.
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
    """Upstream density line integrals ``P_m(r) = int_{r0}^{r} rho_m dl`` to the voxel centers of ``points``.

    The integrals are linear and channel by channel, so any subset of the materials can be traced, e.g. one map at a
    time; the density is zero outside the box of ``grid``.

    Args:
        projectors (Projectors): ray tracers (``upstream_cumulative`` when the points are the centers of ``grid``, else
            ``upstream_to_points``).
        rho (Tensor): [M', rows, cols, slices] density maps in g/cm^3 on ``grid``.
        grid (Grid): grid of the densities.
        geometry (ConeGeometry or ParallelGeometry): scanner geometry.
        angle (float): view angle in radians.
        points (Grid, optional): grid of the end points. Defaults to None (``grid``).

    Returns:
        Tensor: [M', rows, cols, slices] line integrals on ``points`` in g/cm^2.
    """
    point_grid = grid if points is None else points
    if point_grid == grid:
        return projectors.upstream_cumulative(rho, grid, geometry, angle)  # [M', R, C, S] g/cm^2
    r = point_grid.center_coordinates(rho.device, rho.dtype)
    return projectors.upstream_to_points(rho, grid, geometry, angle, r.reshape(-1, 3)).reshape(-1, *point_grid.shape)


def uncollided_from_path_lengths(library: MultigroupLibrary, path_lengths: Tensor, materials: Optional[Sequence[int]],
                                 points: Grid, geometry: ConeGeometry, angle: float, source: SourceModel) -> Tensor:
    """Uncollided scalar fluence at the voxel centers of ``points`` from the upstream density line integrals.

    Computes ``phi_UC,g(r) = b(Omega_UC(r)) / |r - r0|^2 sum_{E in g} w_E exp(-sum_m P_m(r) m_m(E))``; for a parallel
    beam ``phi_UC,g(r) = b(r) sum_{E in g} w_E exp(-sum_m P_m(r) m_m(E))`` with the beam weight b of
    ``SourceModel.beam_weight`` (unit incident fluence).

    Args:
        library (MultigroupLibrary): cross-section library with the fine-energy weights and mass attenuation.
        path_lengths (Tensor): [M', rows, cols, slices] upstream line integrals on ``points`` in g/cm^2.
        materials (Sequence[int] | None): library indices of the rows of ``path_lengths``, None for all materials in
            library order.
        points (Grid): grid of the voxel centers.
        geometry (ConeGeometry or ParallelGeometry): scanner geometry.
        angle (float): view angle in radians.
        source (SourceModel): source model giving the angular weight b.

    Returns:
        Tensor: [G, rows, cols, slices] fluence on ``points`` in photons/cm^2 per source photon/sr.
    """
    device, dtype = path_lengths.device, path_lengths.dtype
    transmission = group_transmission(library, path_lengths, materials=materials)  # [G, R, C, S]
    r = points.center_coordinates(device, dtype)
    if geometry.is_parallel:
        return transmission.mul_(source.beam_weight(r, geometry, angle)[None])
    r0 = torch.tensor(geometry.source_position(angle), device=device, dtype=dtype)
    d = r - r0
    distance_squared = (d * d).sum(-1)  # [R, C, S]
    b = source.angular_weight(d, geometry, angle)  # [R, C, S]
    return transmission.mul_((b / distance_squared)[None])


def first_collision_source_at_directions(library: MultigroupLibrary, rho: Tensor, phi_uncollided: Tensor, grid: Grid,
                                         geometry: ConeGeometry, angle: float, omegas: Tensor, chunk: int = 4,
                                         backend: str = "auto", pairs: PairsArgument = None) -> Tensor:
    """First-collision source at every voxel center of ``grid`` for each direction.

    ``S_FC,g(r, omega_k) = sum_{g' <= g} mu_s^{g'->g}(r; Omega_UC(r) . omega_k) phi_UC,g'(r)``, downscatter only.

    Args:
        library (MultigroupLibrary): cross-section library with the angular transfer table.
        rho (Tensor): [M, rows, cols, slices] material densities rho in g/cm^3.
        phi_uncollided (Tensor): [G, rows, cols, slices] uncollided fluence in photons/cm^2 per source photon/sr.
        grid (Grid): grid of the points (in the solver the quadrature grid).
        geometry (ConeGeometry or ParallelGeometry): scanner geometry.
        angle (float): view angle in radians.
        omegas (Tensor): [K, 3] unit directions.
        chunk (int, optional): directions per batch of the PyTorch path. Defaults to 4.
        backend (str, optional): "auto" (the fused Triton kernel for float32 CUDA inputs when it is usable, else
            PyTorch), "torch" or "triton". Defaults to "auto".
        pairs (MaterialPairs | str | None, optional): material representation, see ``resolve_pairs``; pass the resolved
            value when calling repeatedly. Defaults to None.

    Returns:
        Tensor: [K, G, rows, cols, slices] source in photons/(cm^3 sr) per source photon/sr.
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
    """PyTorch path of ``first_collision_source_at_directions`` with all M materials per point.

    The interpolated transfer table ``[G_in, G_out, chunk, rows, cols, slices]`` is formed per chunk of directions.

    Args:
        library (MultigroupLibrary): cross-section library with the angular transfer table.
        rho (Tensor): [M, rows, cols, slices] material densities rho in g/cm^3.
        phi_uncollided (Tensor): [G, rows, cols, slices] uncollided fluence in photons/cm^2 per source photon/sr.
        grid (Grid): grid of the points (in the solver the quadrature grid).
        geometry (ConeGeometry or ParallelGeometry): scanner geometry.
        angle (float): view angle in radians.
        omegas (Tensor): [K, 3] unit directions.
        chunk (int, optional): directions per batch. Defaults to 4.

    Returns:
        Tensor: [K, G, rows, cols, slices] source in photons/(cm^3 sr) per source photon/sr.
    """
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
    """PyTorch path of ``first_collision_source_at_directions`` for the pair representation.

    Per chunk of directions, the two table rows of every point are gathered by material index.

    Args:
        library (MultigroupLibrary): cross-section library with the angular transfer table.
        pairs (MaterialPairs): pair representation of the material densities.
        phi_uncollided (Tensor): [G, rows, cols, slices] uncollided fluence in photons/cm^2 per source photon/sr.
        grid (Grid): grid of the points (in the solver the quadrature grid).
        geometry (ConeGeometry or ParallelGeometry): scanner geometry.
        angle (float): view angle in radians.
        omegas (Tensor): [K, 3] unit directions.
        chunk (int, optional): directions per batch. Defaults to 4.

    Returns:
        Tensor: [K, G, rows, cols, slices] source in photons/(cm^3 sr) per source photon/sr.
    """
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
    """Restrict a field on a refined grid to linear-discontinuous moments on the grid it refines.

    The ``f^3`` sub-cell centers are equal-weight quadrature points at local coordinates ``u_j = (2j + 1)/f - 1``;
    moment 0 is ``mean_j S_j`` and moment a is ``mean_j(S_j u_a,j) / mean_j(u_a,j^2)``, exact for a field linear in the
    cell.

    Args:
        field (Tensor): [..., rows, cols, slices] values on ``grid``, typically [K, G, rows, cols, slices].
        grid (Grid): the refined grid (in the solver the quadrature grid).
        factor (int): refinement factor f; the grid dimensions must be divisible by it.

    Returns:
        Tensor: [..., 4, rows/f, cols/f, slices/f] moments ordered [0, x, y, z] (x along cols, y along rows, z along
        slices); the slopes are 0 for f = 1.
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
    """Exact Legendre moments of the first-collision source about the primary direction ``Omega_UC``.

    ``M_l[g] = int P_l(Omega . Omega_UC) S_FC,g dOmega = sum_{g' <= g} mu_s_l^{g'->g} phi_UC,g'`` for l = 0..l_max.

    Args:
        library (MultigroupLibrary): library with Legendre transfer moments of order at least ``l_max``.
        rho (Tensor): [M, R, C, S] material densities in g/cm^3.
        phi_uncollided (Tensor): [G, R, C, S] uncollided fluence in photons/cm^2 per source photon/sr.
        l_max (int, optional): highest Legendre order. Defaults to 1.

    Returns:
        Tensor: [l_max+1, G, R, C, S] moments in photons/cm^3 per source photon/sr, dtype and device of phi_uncollided.
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
    """Per-point, per-group correction ``S_m <- scale * S_m * (1 + beta * Omega_m . Omega_UC)`` of the sampled source.

    The factors make the quadrature-weighted l = 0 and l = 1 moments of the corrected first-collision source equal to
    the exact moments (l = 1 unless beta was clamped to [-1, 1] to keep the source non-negative).

    Attributes:
        scale (Tensor): [G, R, C, S] scale factor of each point and group.
        beta (Tensor): [G, R, C, S] linear coefficient in [-1, 1].
        omega_uncollided (Tensor): [R, C, S, 3] primary direction ``Omega_UC`` at each point.
    """

    scale: Tensor
    beta: Tensor
    omega_uncollided: Tensor

    def apply(self, source: Tensor, omegas: Tensor) -> Tensor:
        """Apply the correction to the sampled first-collision source.

        Args:
            source (Tensor): [K, G, R, C, S] source from ``first_collision_source_at_directions`` at ``omegas``.
            omegas (Tensor): [K, 3] unit directions of the samples.

        Returns:
            Tensor: [K, G, R, C, S] corrected source.
        """
        directions = omegas.to(device=source.device, dtype=source.dtype).reshape(-1, 3)
        eta = torch.einsum("rcsx,kx->krcs", self.omega_uncollided, directions)  # [K, R, C, S]
        return source * self.scale * (1.0 + self.beta[None] * eta[:, None])


def first_collision_sampling_correction(library: MultigroupLibrary, rho: Tensor, phi_uncollided: Tensor, grid: Grid,
                                        geometry: ConeGeometry, angle: float, omegas: Tensor, weights: Tensor,
                                        chunk: int = 4, batch: int = 8,
                                        pairs: PairsArgument = None) -> FirstCollisionSamplingCorrection:
    """Correction that makes a direction quadrature integrate the sampled first-collision source exactly for l = 0, 1.

    The factors follow from the quadrature sums (``first_collision_quadrature_sums``) and the exact moments
    (``first_collision_exact_moments``) by ``first_collision_correction_factors``.

    Args:
        library (MultigroupLibrary): cross-section library with the angular transfer table.
        rho (Tensor): [M, rows, cols, slices] material densities rho in g/cm^3.
        phi_uncollided (Tensor): [G, rows, cols, slices] uncollided fluence in photons/cm^2 per source photon/sr.
        grid (Grid): grid of the points (in the solver the quadrature grid).
        geometry (ConeGeometry or ParallelGeometry): scanner geometry.
        angle (float): view angle in radians.
        omegas (Tensor): [K, 3] unit directions of the quadrature.
        weights (Tensor): [K] quadrature weights in sr.
        chunk (int, optional): directions per batch of the PyTorch path of the source. Defaults to 4.
        batch (int, optional): directions per source evaluation. Defaults to 8.
        pairs (MaterialPairs | str | None, optional): as in ``first_collision_source_at_directions``. Defaults to None.

    Returns:
        FirstCollisionSamplingCorrection: the factors, in the dtype of ``phi_uncollided``.
    """
    device, dtype = phi_uncollided.device, phi_uncollided.dtype
    omegas = omegas.to(device=device, dtype=dtype).reshape(-1, 3)
    weights = weights.to(device=device, dtype=dtype).reshape(-1)
    if weights.shape[0] != omegas.shape[0]:
        raise ValueError("weights and omegas disagree in length")
    omega_uncollided = primary_directions_to_voxels(grid, geometry, angle, device, dtype)
    pairs = resolve_pairs(rho, pairs)
    compact_pairs = pairs if isinstance(pairs, MaterialPairs) else None
    if uncollided_kernels.fused_kernels_usable(library, rho, phi_uncollided, compact_pairs) and \
            library.legendre_order >= 1:
        # one kernel: the quadrature sums and the exact moments (in float32), without forming the source
        sums_and_moments = uncollided_kernels.first_collision_sums(library, rho, phi_uncollided, grid, geometry, angle,
                                                                   omegas, weights, pairs=compact_pairs,
                                                                   exact_moments=True).double()
        sums, exact_moments = sums_and_moments[:3], sums_and_moments[3:]
    else:
        sums = first_collision_quadrature_sums(library, rho, phi_uncollided, grid, geometry, angle, omegas, weights,
                                               omega_uncollided=omega_uncollided, chunk=chunk, batch=batch,
                                               pairs=pairs)
        exact_moments = first_collision_exact_moments(library, rho, phi_uncollided.double(), l_max=1)
    scale, beta = first_collision_correction_factors(sums, exact_moments)
    return FirstCollisionSamplingCorrection(scale=scale.to(dtype), beta=beta.to(dtype),
                                            omega_uncollided=omega_uncollided)


def first_collision_moments_at_directions(library: MultigroupLibrary, rho: Tensor, phi_uncollided: Tensor, grid: Grid,
                                          geometry: ConeGeometry, angle: float, omegas: Tensor,
                                          correction: FirstCollisionSamplingCorrection, factor: int, chunk: int = 4,
                                          pairs: PairsArgument = None, out: Optional[Tensor] = None) -> Tensor:
    """Linear-discontinuous moments of the corrected first-collision source on the cells of a refined grid.

    The source at the points of ``grid`` (``first_collision_source_at_directions``) is corrected
    (``FirstCollisionSamplingCorrection.apply``) and restricted to the cells
    (``restrict_to_linear_discontinuous_moments``); on float32 CUDA tensors one fused kernel does all three without
    forming the [K, G, rows, cols, slices] source (``uncollided_kernels.first_collision_moments``).

    Args:
        library (MultigroupLibrary): cross-section library with the angular transfer table.
        rho (Tensor): [M, rows, cols, slices] material densities on ``grid`` in g/cm^3.
        phi_uncollided (Tensor): [G, rows, cols, slices] uncollided fluence.
        grid (Grid): the quadrature grid, ``factor`` points per axis per cell.
        geometry (ConeGeometry or ParallelGeometry): scanner geometry.
        angle (float): view angle in radians.
        omegas (Tensor): [K, 3] unit directions.
        correction (FirstCollisionSamplingCorrection): the sampling correction of the view.
        factor (int): points per axis per cell.
        chunk (int, optional): directions per batch of the PyTorch path of the source. Defaults to 4.
        pairs (MaterialPairs | str | None, optional): as in ``first_collision_source_at_directions``. Defaults to None.
        out (Tensor, optional): contiguous [K, G, 4, rows/f, cols/f, slices/f] tensor to fill. Defaults to None.

    Returns:
        Tensor: [K, G, 4, rows/f, cols/f, slices/f] moments ordered [0, x, y, z].
    """
    pairs = resolve_pairs(rho, pairs)
    compact_pairs = pairs if isinstance(pairs, MaterialPairs) else None
    if uncollided_kernels.fused_kernels_usable(library, rho, phi_uncollided, compact_pairs) and (
            out is None or (out.is_contiguous() and out.dtype == torch.float32)):
        return uncollided_kernels.first_collision_moments(library, rho, phi_uncollided, grid, geometry, angle, omegas,
                                                          correction.scale, correction.beta, factor,
                                                          pairs=compact_pairs, out=out)
    source = first_collision_source_at_directions(library, rho, phi_uncollided, grid, geometry, angle, omegas,
                                                  chunk=chunk, pairs=pairs)
    moments = restrict_to_linear_discontinuous_moments(correction.apply(source, omegas), grid, factor)
    del source
    if out is None:
        return moments
    return out.copy_(moments)


# at most this many directions are summed in the input dtype (float32 on GPU) before a float64 accumulation
_FIRST_COLLISION_SUM_MAX_DIRECTIONS = 256


def _moment_weight_rows(omegas: Tensor, weights: Tensor) -> Tensor:
    """Weight rows that turn one matrix product with the sampled source into the partial quadrature sums.

    The rows are ``(w, w o_x, w o_y, w o_z, w o_x^2, w o_y^2, w o_z^2, 2 w o_x o_y, 2 w o_x o_z, 2 w o_y o_z)``, so that
    for ``P = rows @ S`` and ``u = Omega_UC``: ``I0 = P_0``, ``I1 = sum_a u_a P_(1+a)`` and I2 is the quadratic form of
    u in ``P_4..P_9``.

    Args:
        omegas (Tensor): [K, 3] unit directions.
        weights (Tensor): [K] quadrature weights in sr.

    Returns:
        Tensor: [10, K] weight rows.
    """
    x, y, z = omegas.unbind(-1)
    w2 = 2.0 * weights
    return torch.stack([weights, weights * x, weights * y, weights * z, weights * x * x, weights * y * y,
                        weights * z * z, w2 * x * y, w2 * x * z, w2 * y * z])


def first_collision_quadrature_sums(library: MultigroupLibrary, rho: Tensor, phi_uncollided: Tensor, grid: Grid,
                                    geometry: ConeGeometry, angle: float, omegas: Tensor, weights: Tensor,
                                    omega_uncollided: Tensor | None = None, chunk: int = 4, batch: int = 8,
                                    pairs: PairsArgument = None) -> Tensor:
    """Quadrature sums of the point-sampled first-collision source per point and group.

    ``I0 = sum_m w_m S_m``, ``I1 = sum_m w_m mu_m S_m`` and ``I2 = sum_m w_m mu_m^2 S_m`` with
    ``mu_m = omega_m . Omega_UC(r)``; float32 partial sums run over at most ``_FIRST_COLLISION_SUM_MAX_DIRECTIONS``
    directions and are accumulated in float64. On float32 CUDA tensors one fused kernel accumulates the sums in float32
    over all directions without forming the source (``uncollided_kernels.first_collision_sums``).

    Args:
        library (MultigroupLibrary): cross-section library with the angular transfer table.
        rho (Tensor): [M, rows, cols, slices] material densities rho in g/cm^3.
        phi_uncollided (Tensor): [G, rows, cols, slices] uncollided fluence in photons/cm^2 per source photon/sr.
        grid (Grid): grid of the points (in the solver the quadrature grid).
        geometry (ConeGeometry or ParallelGeometry): scanner geometry.
        angle (float): view angle in radians.
        omegas (Tensor): [K, 3] unit directions of the quadrature.
        weights (Tensor): [K] quadrature weights in sr.
        omega_uncollided (Tensor, optional): [R, C, S, 3] primary directions. Defaults to None (computed).
        chunk (int, optional): directions per batch of the PyTorch path of the source. Defaults to 4.
        batch (int, optional): directions per source evaluation, which sets the [batch, G, R, C, S] temporary.
            Defaults to 8.
        pairs (MaterialPairs | str | None, optional): as in ``first_collision_source_at_directions``. Defaults to None.

    Returns:
        Tensor: [3, G, R, C, S] float64 sums I0, I1, I2.
    """
    pairs = resolve_pairs(rho, pairs)
    compact_pairs = pairs if isinstance(pairs, MaterialPairs) else None
    if uncollided_kernels.fused_kernels_usable(library, rho, phi_uncollided, compact_pairs):
        return uncollided_kernels.first_collision_sums(library, rho, phi_uncollided, grid, geometry, angle, omegas,
                                                       weights, pairs=compact_pairs).double()
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
    """Scale and beta of the sampling correction from the quadrature sums and the exact moments, in float64.

    With ``c = E1 / E0``, ``beta = (c I0 - I1) / (I2 - c I1)`` clamped to [-1, 1] (0 where E0 is not positive or the
    denominator vanishes) and ``scale = E0 / (I0 + beta I1)`` (1 where that denominator is not positive).

    Args:
        I (Tensor): [3, G, R, C, S] quadrature sums I0, I1, I2 from ``first_collision_quadrature_sums``.
        E (Tensor): [2, G, R, C, S] exact moments E0, E1 from ``first_collision_exact_moments`` with ``l_max = 1``.

    Returns:
        (scale, beta): two [G, R, C, S] float64 tensors.
    """
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
