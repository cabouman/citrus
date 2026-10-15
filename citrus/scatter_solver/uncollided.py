"""Uncollided fluence and first-collision source of the LBTE solver.

For a point source at r0 emitting q_g = f_g b(Omega) photons per steradian in group g, the uncollided fluence and
the first-collision (FC) source are

    phi_UC,g(r)       = f_g b(Omega_UC(r)) / |r - r0|^2  exp(-tau_g(r0 -> r)),     Omega_UC(r) = (r - r0) / |r - r0|
    S_FC,g(r, Omega)  = sum_{g' <= g} mu_s^{g' -> g}(r; Omega_UC(r) . Omega) phi_UC,g'(r)

(per-steradian transfer mu_s(eta) tabulated in eta = cos(scattering angle)). All functions work for one view angle
in the recon frame, on the fine grid, and are device- and dtype-preserving (the dtype of `rho` / `phi_uc` decides
the computation dtype).

The uncollided fluence is exact in energy within each group: two upstream ray traces of the density maps and the
library's fine-energy attenuation table (`cross_sections.group_transmission`), not the group-averaged mu_t.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .cross_sections import MultigroupLibrary, group_transmission, transfer_eta_at
from .geometry import ConeGeometry, Grid
from .kernels import uncollided_kernels as _fc_tri
from .ray_tracing import Projectors
from .source import SourceModel


def primary_directions_to_voxels(grid: Grid, geom: ConeGeometry, angle: float, device=None,
                                 dtype=torch.float32) -> Tensor:
    """Unit vectors Omega_UC(r) = (r - r0)/|r - r0| from the focal spot to each voxel center.

    Returns [rows, cols, slices, 3] (recon frame).
    """
    r = grid.center_coordinates(device, dtype)
    r0 = torch.tensor(geom.source_position(angle), device=device, dtype=dtype)
    d = r - r0
    return d / torch.linalg.norm(d, dim=-1, keepdim=True)


def uncollided_fluence(lib: MultigroupLibrary, rho: Tensor, grid: Grid, geom: ConeGeometry, angle: float,
                       source: SourceModel, projectors: Projectors) -> Tensor:
    """Uncollided scalar fluence at every fine voxel center, exact in energy within each group.

        phi_UC,g(r) = b(Omega_UC(r)) / |r - r0|^2 * sum_{E in g} w_E exp(-(P_w(r) m_w(E) + P_b(r) m_b(E)))

    with P_m(r) = int_{r0}^{r} rho_m dl the upstream line integrals of the two density maps (g/cm^2, TWO ray
    traces, `projectors.upstream_cumulative(rho)`), and the library's fine-energy table: w_E = lib.fine_weight
    (source photons/sr per fine bin; their sum over group g is f_g = source.per_group_strength), m_m(E) =
    lib.fine_mu_t (`cross_sections.group_transmission`). Since mu_t is linear in the densities this is the exact
    group-integrated polychromatic Beer's law (up to the fine quadrature); the group-averaged attenuation
    f_g exp(-int mu_t,g dl) ignores beam hardening within a group (-1.4% after 20 cm of water with 6 groups).

    rho: [M, rows, cols, slices] g/cm^3. Returns phi_uc [G, rows, cols, slices] in photons/cm^2
    per source photon/sr (point values at voxel centers).
    """
    dev, dt = rho.device, rho.dtype
    path = projectors.upstream_cumulative(rho, grid, geom, angle)  # [M, R, C, S] g/cm^2
    trans = group_transmission(lib, path)  # [G, R, C, S]
    r = grid.center_coordinates(dev, dt)
    r0 = torch.tensor(geom.source_position(angle), device=dev, dtype=dt)
    d = r - r0
    dist2 = (d * d).sum(-1)  # [R, C, S]
    b = source.angular_weight(d, geom, angle)  # [R, C, S]
    return trans.mul_((b / dist2)[None])


def fc_source_at_directions(lib: MultigroupLibrary, rho: Tensor, phi_uc: Tensor, grid: Grid,
                            geom: ConeGeometry, angle: float, omegas: Tensor, chunk: int = 4,
                            backend: str = "auto") -> Tensor:
    """First-collision source at every fine voxel center for each direction.

        S_FC,g(r, omega_k) = sum_{g' <= g} mu_s^{g'->g}(r; Omega_UC(r) . omega_k) phi_UC,g'(r)

    Groups are ordered from high to low energy, so g' <= g is downscatter (no upscatter).
    rho: [M, rows, cols, slices]; phi_uc: [G, rows, cols, slices]; omegas: [K, 3] unit vectors.
    Returns [K, G, rows, cols, slices] in photons/(cm^3 sr) per source photon/sr.

    backend: "auto" (default) runs float32 CUDA inputs through the fused Triton kernel of `kernels/uncollided_kernels.py`
    when it is usable on the device (shared Triton probe, CITRUS_DISABLE_TRITON=1 kill switch, per-device
    self-check against the PyTorch path), else the PyTorch path; "torch" forces the PyTorch path; "triton"
    requires the kernel for float32 CUDA inputs (raises if it is unusable). float64 and CPU inputs always use
    the PyTorch path (the reference). The two paths agree to float32 rounding (~1e-6 relative).
    chunk: directions per batch of the PyTorch path (peak memory ~ 3 M G^2 chunk |F| scalars); the kernel
    evaluates all K directions in one launch without temporaries, so it ignores `chunk`.
    """
    fused = _fc_tri.fc_source_or_none(lib, rho, phi_uc, grid, geom, angle, omegas, backend)
    if fused is not None:
        return fused
    return _fc_source_torch(lib, rho, phi_uc, grid, geom, angle, omegas, chunk)


def _fc_source_torch(lib: MultigroupLibrary, rho: Tensor, phi_uc: Tensor, grid: Grid, geom: ConeGeometry,
                     angle: float, omegas: Tensor, chunk: int = 4) -> Tensor:
    """PyTorch path of `fc_source_at_directions` (any device / dtype; same shapes): materializes the
    interpolated transfer table [G_in, G_out, chunk, rows, cols, slices] per `chunk` directions."""
    dev, dt = phi_uc.device, phi_uc.dtype
    rho = rho.to(device=dev, dtype=dt)
    omegas = omegas.to(device=dev, dtype=dt).reshape(-1, 3)
    K = omegas.shape[0]
    G = phi_uc.shape[0]
    om_uc = primary_directions_to_voxels(grid, geom, angle, dev, dt)  # [R, C, S, 3]
    down = torch.triu(torch.ones(G, G, device=dev, dtype=dt))  # [g_in, g_out], 1 where g_out >= g_in
    out = torch.empty(K, G, *phi_uc.shape[1:], device=dev, dtype=dt)
    M = rho.shape[0]
    for s in range(0, K, max(1, int(chunk))):
        om = omegas[s:s + chunk]  # [k, 3]
        k = om.shape[0]
        eta = torch.einsum("rcsx,kx->krcs", om_uc, om)  # [k, R, C, S]
        rho_k = rho[:, None].expand(M, k, *rho.shape[1:])
        T = transfer_eta_at(lib, rho_k, eta)  # [G_in, G_out, k, R, C, S]
        out[s:s + k] = torch.einsum("abk...,ab,a...->kb...", T, down, phi_uc)
    return out


def restrict_fc_to_ld_moments(fc: Tensor, grid: Grid, factor: int) -> Tensor:
    """Restrict a fine-grid field to LD moments on the supervoxel grid.

    fc: [..., rows, cols, slices] on `grid` (typically [K, G, rows, cols, slices]).
    Returns [..., 4, rows/f, cols/f, slices/f] with moments ordered [0, x, y, z] (x <-> cols,
    y <-> rows, z <-> slices), using the f^3 fine sub-voxel centers as equal-weight quadrature points
    with local coordinates u_j = (2j + 1)/f - 1 (grid.local_coordinates_of_fine_in_coarse):

      moment 0 = mean_j S_j
      moment a = mean_j(S_j u_a,j) / mean_j(u_a,j^2)

    The midpoint quadrature gives mean_j(u_j^2) = (1 - 1/f^2)/3, not 1/3, so the continuous Galerkin
    formula (3 mean(S u)) would return s (1 - 1/f^2) for a field c + s u sampled at the sub-voxel centers.
    Dividing by the discrete second moment instead reproduces a field that is linear within each
    supervoxel exactly (slope s, mean c). For f = 1 the slopes are 0.
    """
    f = int(factor)
    coarse = grid.coarsen(f)  # validates divisibility
    if tuple(fc.shape[-3:]) != tuple(grid.shape):
        raise ValueError(f"field spatial shape {tuple(fc.shape[-3:])} != grid shape {grid.shape}")
    lead = fc.shape[:-3]
    R, C, S = coarse.shape
    x = fc.reshape(*lead, R, f, C, f, S, f)
    nl = len(lead)
    ax_r, ax_c, ax_s = nl + 1, nl + 3, nl + 5  # sub-voxel axes for rows (y), cols (x), slices (z)
    m0 = x.mean(dim=(ax_r, ax_c, ax_s))
    if f == 1:
        z = torch.zeros_like(m0)
        return torch.stack([m0, z, z, z], dim=-4)
    u = grid.local_coordinates_of_fine_in_coarse(f, device=fc.device, dtype=fc.dtype)  # [f]
    scale = 1.0 / (u * u).mean()

    def slope(axis: int) -> Tensor:
        shape = [1] * x.dim()
        shape[axis] = f
        return (x * u.reshape(shape)).mean(dim=(ax_r, ax_c, ax_s)) * scale

    return torch.stack([m0, slope(ax_c), slope(ax_r), slope(ax_s)], dim=-4)


# =====================================================================================================
# conservative sampling of the first-collision source at the sweep directions (docs/source/design_notes.rst)
# =====================================================================================================


def fc_exact_moments(lib: MultigroupLibrary, rho: Tensor, phi_uc: Tensor, l_max: int = 1) -> Tensor:
    """Exact angular moments of the first-collision source about Omega_UC from the library Legendre moments.

        M_l[g](r) = int P_l(Omega . Omega_UC(r)) S_FC,g(r, Omega) dOmega = sum_{g' <= g} mu_s_l^{g'->g}(r) phi_UC,g'(r)

    (convention mu_s(eta) = sum_l (2l+1)/(4 pi) mu_s_l P_l(eta)), l = 0..l_max. rho: [M, R, C, S];
    phi_uc: [G, R, C, S]. Returns [l_max+1, G, R, C, S] in the dtype/device of phi_uc.
    """
    if lib.legendre_order < l_max:
        raise ValueError(f"library Legendre order {lib.legendre_order} < {l_max}")
    dev, dt = phi_uc.device, phi_uc.dtype
    G = phi_uc.shape[0]
    t = lib.mu_s_l[..., :l_max + 1].to(device=dev, dtype=dt)  # [M, Gi, Go, l]
    down = torch.triu(torch.ones(G, G, device=dev, dtype=dt))
    return torch.einsum("mabl,ab,m...,a...->lb...", t, down, rho.to(device=dev, dtype=dt), phi_uc)


@dataclass(frozen=True, eq=False)
class FCSamplingCorrection:
    """Per-voxel, per-group correction of the pointwise-sampled FC source (fine grid):

        S_m <- scale * S_m * (1 + beta * Omega_m . Omega_UC)

    scale, beta: [G, R, C, S]; om_uc: [R, C, S, 3]. The factors make the quadrature-weighted l = 0 and l = 1 (unless
    beta had to be clamped to [-1, 1] to keep the source non-negative) moments of the corrected source equal to
    `fc_exact_moments` (see `fc_sampling_correction`).
    """

    scale: Tensor
    beta: Tensor
    om_uc: Tensor

    def apply(self, fc: Tensor, omegas: Tensor) -> Tensor:
        """fc: [K, G, R, C, S] = fc_source_at_directions(..., omegas [K, 3]). Returns the corrected source."""
        om = omegas.to(device=fc.device, dtype=fc.dtype).reshape(-1, 3)
        eta = torch.einsum("rcsx,kx->krcs", self.om_uc, om)  # [K, R, C, S]
        return fc * self.scale * (1.0 + self.beta[None] * eta[:, None])


def fc_sampling_correction(lib: MultigroupLibrary, rho: Tensor, phi_uc: Tensor, grid: Grid, geom: ConeGeometry,
                           angle: float, omegas: Tensor, weights: Tensor, chunk: int = 4,
                           batch: int = 8) -> FCSamplingCorrection:
    """Correction factors that make the quadrature (omegas [K, 3], weights [K]) integrate the sampled FC source
    exactly in angle, l = 0 and l = 1 about Omega_UC (one extra pass of `fc_source_at_directions` over all K
    directions, `batch` at a time).

    With the quadrature sums (per voxel and group, mu_m = Omega_m . Omega_UC(r), S_m the point-sampled source)
        I0 = sum_m w_m S_m,  I1 = sum_m w_m mu_m S_m,  I2 = sum_m w_m mu_m^2 S_m
    and the exact moments E0, E1 (`fc_exact_moments`): with c = E1 / E0, beta = (c I0 - I1) / (I2 - c I1) clamped
    to [-1, 1] (beta = 0 where E0 == 0 or the denominator vanishes), scale = E0 / (I0 + beta I1) (1 where that
    denominator is not positive). The l = 0 moment is always matched; l = 1 is matched unless beta was clamped
    (the clamp keeps 1 + beta mu >= 0, i.e. a non-negative source).
    Precision (`fc_quadrature_sums`): the sums are formed by matmuls in the input dtype (float32 on GPU) over blocks of
    at most `_FC_SUM_MAX_DIRECTIONS` directions and accumulated across blocks in float64; the factors are computed in
    float64. Returns an FCSamplingCorrection in the dtype of phi_uc.
    """
    dev, dt = phi_uc.device, phi_uc.dtype
    omegas = omegas.to(device=dev, dtype=dt).reshape(-1, 3)
    w = weights.to(device=dev, dtype=dt).reshape(-1)
    if w.shape[0] != omegas.shape[0]:
        raise ValueError("weights and omegas disagree in length")
    om_uc = primary_directions_to_voxels(grid, geom, angle, dev, dt)
    I = fc_quadrature_sums(lib, rho, phi_uc, grid, geom, angle, omegas, w, om_uc=om_uc, chunk=chunk, batch=batch)
    E = fc_exact_moments(lib, rho, phi_uc.double(), l_max=1)
    scale, beta = fc_correction_factors(I, E)
    return FCSamplingCorrection(scale=scale.to(dt), beta=beta.to(dt), om_uc=om_uc)


# at most this many directions are summed in the input dtype (float32 on GPU) before a float64 accumulation
_FC_SUM_MAX_DIRECTIONS = 256


def _moment_weight_rows(omegas: Tensor, weights: Tensor) -> Tensor:
    """[10, K] rows (w, w o_x, w o_y, w o_z, w o_x^2, w o_y^2, w o_z^2, 2 w o_x o_y, 2 w o_x o_z, 2 w o_y o_z) such
    that, for P = rows @ S ([10, K] @ [K, n]) and u = Omega_UC, I0 = P_0, I1 = sum_a u_a P_{1+a} and
    I2 = sum_m w_m (omega_m . u)^2 S_m = u_x^2 P_4 + u_y^2 P_5 + u_z^2 P_6 + u_x u_y P_7 + u_x u_z P_8 + u_y u_z P_9."""
    x, y, z = omegas.unbind(-1)
    w2 = 2.0 * weights
    return torch.stack([weights, weights * x, weights * y, weights * z, weights * x * x, weights * y * y,
                        weights * z * z, w2 * x * y, w2 * x * z, w2 * y * z])


def fc_quadrature_sums(lib: MultigroupLibrary, rho: Tensor, phi_uc: Tensor, grid: Grid, geom: ConeGeometry,
                       angle: float, omegas: Tensor, weights: Tensor, om_uc: Tensor | None = None, chunk: int = 4,
                       batch: int = 8) -> Tensor:
    """Quadrature sums of the point-sampled FC source per voxel and group (see `fc_sampling_correction`):

        I0 = sum_m w_m S_m,  I1 = sum_m w_m mu_m S_m,  I2 = sum_m w_m mu_m^2 S_m,   mu_m = omega_m . Omega_UC(r)

    Returns [3, G, R, C, S] float64. One pass of `fc_source_at_directions` over the K directions, `batch` at a time
    (this sets the size of the [batch, G, R, C, S] source temporary). Since mu_m is linear in Omega_UC, a batch's
    contribution to all three sums is ONE matmul (`addmm`) of the [10, k] weight rows of `_moment_weight_rows`
    (quadrature weights times the monomials of omega_m up to degree 2) with the [k, G V] source, accumulated into a
    [10, G V] partial in the input dtype. Every `_FC_SUM_MAX_DIRECTIONS` directions (and at the end) the partial is
    contracted with the per-voxel monomials of Omega_UC (1 + 3 + 6 rows -> I0, I1, I2) and added to the float64
    accumulators, so a float32 sum runs over at most `_FC_SUM_MAX_DIRECTIONS` directions and the sum across those
    blocks keeps float64. float64 inputs are summed in float64 throughout. (The float32 matmul assumes TF32 is off,
    `torch.backends.cuda.matmul.allow_tf32 = False`, the PyTorch default.)
    """
    dev, dt = phi_uc.device, phi_uc.dtype
    omegas = omegas.to(device=dev, dtype=dt).reshape(-1, 3)
    weights = weights.to(device=dev, dtype=dt).reshape(-1)
    if om_uc is None:
        om_uc = primary_directions_to_voxels(grid, geom, angle, dev, dt)
    G = phi_uc.shape[0]
    u = om_uc.to(device=dev, dtype=dt).reshape(-1, 3).t()  # [3, V]
    V = u.shape[1]
    rows = _moment_weight_rows(omegas, weights)  # [10, K]
    acc = torch.zeros(3, G, V, device=dev, dtype=torch.float64)
    part = torch.zeros(10, G * V, device=dev, dtype=dt)  # input-dtype partial sums of the current block

    def flush() -> None:
        p = part.view(10, G, V)
        ux, uy, uz = u[0], u[1], u[2]
        u2 = torch.stack([ux * ux, uy * uy, uz * uz, ux * uy, ux * uz, uy * uz])
        acc[0] += p[0]
        acc[1] += (p[1:4] * u[:, None]).sum(0)
        acc[2] += (p[4:10] * u2[:, None]).sum(0)
        part.zero_()

    step = max(1, min(int(batch), _FC_SUM_MAX_DIRECTIONS))
    K = omegas.shape[0]
    in_block = 0
    for s in range(0, K, step):
        om = omegas[s:s + step]
        k = om.shape[0]
        if in_block + k > _FC_SUM_MAX_DIRECTIONS:
            flush()
            in_block = 0
        fc = fc_source_at_directions(lib, rho, phi_uc, grid, geom, angle, om, chunk=chunk)  # [k, G, R, C, S]
        part.addmm_(rows[:, s:s + k], fc.reshape(k, G * V))
        in_block += k
        del fc
    flush()
    return acc.reshape(3, *phi_uc.shape)


def fc_correction_factors(I: Tensor, E: Tensor) -> tuple[Tensor, Tensor]:
    """(scale, beta) of `fc_sampling_correction` from the quadrature sums I = [I0, I1, I2] (`fc_quadrature_sums`) and
    the exact moments E = [E0, E1] (`fc_exact_moments`, l_max = 1), computed in float64; returned in float64."""
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
