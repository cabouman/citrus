"""Tests for citrus.scatter_solver.angular_quadrature: quadratures, spherical harmonics, scattering source."""

import math

import numpy as np
import pytest
import torch

from citrus.scatter_solver.angular_quadrature import (
    Quadrature,
    hemisphere_quadrature,
    legendre_polynomials,
    moments_from_ordinates,
    num_spherical_harmonics,
    ordinates_from_moments,
    product_quadrature,
    real_spherical_harmonics,
    scatter_source_at_directions,
    scatter_source_moments,
    spherical_harmonic_basis,
    spherical_harmonic_degree_index,
)
from citrus.scatter_solver.cross_sections import synthetic_library, transfer_eta_at, transfer_moments_volume

F64 = torch.float64
FOUR_PI = 4.0 * math.pi


def _random_dirs(n, seed=0, dtype=F64):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, 3, generator=g, dtype=dtype)
    return v / torch.linalg.norm(v, dim=-1, keepdim=True)


def _gram(L, quadrature):
    b = spherical_harmonic_basis(L, quadrature)
    return b.Y_weighted.T @ b.Y


# ---- (a) product quadrature -----------------------------------------------------------------------

@pytest.mark.parametrize("L", [3, 5, 7])
def test_product_orthonormality(L):
    quadrature = product_quadrature(L + 1, 2 * L + 1)
    assert abs(float(quadrature.weight.sum()) - FOUR_PI) < 1e-12
    err = (_gram(L, quadrature) - torch.eye(num_spherical_harmonics(L), dtype=F64)).abs().max().item()
    assert err < 1e-10, err


def test_product_default_orthonormality():
    quadrature = product_quadrature(8, 16)
    err = (_gram(7, quadrature) - torch.eye(64, dtype=F64)).abs().max().item()
    assert err < 1e-10, err


def test_product_orthonormality_fails_with_too_few_azimuths():
    L = 3
    quadrature = product_quadrature(L + 1, 4)  # needs >= 2L+1 = 7
    err = (_gram(L, quadrature) - torch.eye(num_spherical_harmonics(L), dtype=F64)).abs().max().item()
    assert err > 0.1


def test_product_orthonormality_fails_with_too_few_polar():
    L = 5
    quadrature = product_quadrature(3, 2 * L + 1)  # needs >= L+1 = 6
    err = (_gram(L, quadrature) - torch.eye(num_spherical_harmonics(L), dtype=F64)).abs().max().item()
    assert err > 1e-3


def test_product_octants():
    quadrature = product_quadrature(8, 16)
    assert quadrature.directions_per_octant == 16
    perm, counts = quadrature.by_octant()
    assert counts.tolist() == [16] * 8
    oct_sorted = quadrature.octant[perm]
    assert torch.equal(oct_sorted, torch.arange(8).repeat_interleave(16))
    # octant code definition
    directions = quadrature.omega
    code = (directions[:, 0] < 0).long() * 4 + (directions[:, 1] < 0).long() * 2 + (directions[:, 2] < 0).long()
    assert torch.equal(code, quadrature.octant)
    re = quadrature.reordered(perm)
    assert torch.equal(re.octant, oct_sorted)
    # odd polar count puts mu = 0 on the equator -> unbalanced
    assert product_quadrature(3, 8).directions_per_octant is None


# ---- (c) real spherical-harmonic vs scipy -------------------------------------------------------------------------

def _scipy_real_spherical_harmonics(L, omega):
    from scipy import special

    directions = omega.numpy()
    theta = np.arccos(np.clip(directions[:, 2], -1, 1))
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    cols = []
    for l in range(L + 1):
        for k in range(-l, l + 1):
            m = abs(k)
            if hasattr(special, "sph_harm_y"):
                c = special.sph_harm_y(l, m, theta, phi)
            else:  # older scipy: sph_harm(m, l, azimuth, polar)
                c = special.sph_harm(m, l, phi, theta)
            if k == 0:
                cols.append(c.real)
            elif k > 0:
                cols.append(math.sqrt(2) * (-1) ** m * c.real)
            else:
                cols.append(math.sqrt(2) * (-1) ** m * c.imag)
    return np.stack(cols, -1)


def test_real_sh_vs_scipy():
    L = 7
    directions = _random_dirs(200, seed=1)
    directions = torch.cat([directions, torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=F64)])
    Y = real_spherical_harmonics(L, directions)
    assert Y.shape == (directions.shape[0], 64)
    reference = _scipy_real_spherical_harmonics(L, directions)
    assert np.abs(Y.numpy() - reference).max() < 1e-10


def test_real_sh_low_order_explicit():
    directions = _random_dirs(10, seed=2)
    Y = real_spherical_harmonics(1, directions)
    c0, c1 = 1 / math.sqrt(4 * math.pi), math.sqrt(3 / (4 * math.pi))
    assert torch.allclose(Y[:, 0], torch.full((10,), c0, dtype=F64))
    assert torch.allclose(Y[:, 1], c1 * directions[:, 1])  # k = -1 -> y
    assert torch.allclose(Y[:, 2], c1 * directions[:, 2])  # k = 0 -> z
    assert torch.allclose(Y[:, 3], c1 * directions[:, 0])  # k = +1 -> x


def test_sh_degree_index():
    assert spherical_harmonic_degree_index(2).tolist() == [0, 1, 1, 1, 2, 2, 2, 2, 2]
    assert num_spherical_harmonics(7) == 64


def test_legendre_p_vs_numpy():
    x = torch.linspace(-1, 1, 101, dtype=F64)
    P = legendre_polynomials(8, x)
    for l in range(9):
        reference = np.polynomial.legendre.legval(x.numpy(), [0] * l + [1])
        assert np.abs(P[:, l].numpy() - reference).max() < 1e-13


# ---- (d) addition theorem -------------------------------------------------------------------------

def test_addition_theorem():
    L = 7
    a, b = _random_dirs(50, seed=3), _random_dirs(50, seed=4)
    Ya, Yb = real_spherical_harmonics(L, a), real_spherical_harmonics(L, b)
    P = legendre_polynomials(L, (a * b).sum(-1))
    lofn = spherical_harmonic_degree_index(L)
    for l in range(L + 1):
        sel = lofn == l
        lhs = FOUR_PI / (2 * l + 1) * (Ya[:, sel] * Yb[:, sel]).sum(-1)
        assert (lhs - P[:, l]).abs().max().item() < 1e-10


# ---- (e) moments / ordinates ----------------------------------------------------------------------

def test_isotropic_moments_and_inversion(device):
    L = 5
    quadrature = product_quadrature(L + 1, 2 * L + 2, device=device)
    b = spherical_harmonic_basis(L, quadrature)
    psi = torch.full((3, quadrature.num_directions), 2.5, dtype=F64, device=device)
    phi = moments_from_ordinates(psi, b)
    assert abs(phi[:, 0] - math.sqrt(FOUR_PI) * 2.5).max().item() < 1e-12
    assert phi[:, 1:].abs().max().item() < 1e-12
    # band-limited round trip
    g = torch.Generator().manual_seed(5)
    phi_r = torch.randn(4, 2, num_spherical_harmonics(L), generator=g, dtype=F64).to(device)
    psi_r = ordinates_from_moments(phi_r, b)
    assert psi_r.shape == (4, 2, quadrature.num_directions)
    assert (moments_from_ordinates(psi_r, b) - phi_r).abs().max().item() < 1e-11
    assert psi_r.device.type == device.type


# ---- (f) scattering operator normalization ---------------------------------------------------------

def test_scatter_operator_matches_direct_kernel():
    G, L = 3, 3
    library = synthetic_library(num_groups=G, legendre_order=5)
    g = torch.Generator().manual_seed(7)
    rho = torch.rand(2, 2, 2, 1, generator=g, dtype=F64) * torch.tensor([1.0, 0.5], dtype=F64)[:, None, None, None]
    mu_vol = transfer_moments_volume(library, rho)  # [G, G, 6, 2, 2, 1]
    quadrature = product_quadrature(8, 16)
    b = spherical_harmonic_basis(L, quadrature)
    M = quadrature.num_directions
    psi = ordinates_from_moments(torch.randn(G, num_spherical_harmonics(L), generator=g, dtype=F64), b)  # [G, M]

    # moment path, psi shared across voxels
    phi = moments_from_ordinates(psi, b)  # [G, N]
    phi_vox = phi[:, :, None, None, None].expand(G, num_spherical_harmonics(L), 2, 2, 1)
    T = scatter_source_moments(mu_vol, phi_vox)  # [G, N, 2, 2, 1]
    S_mom = ordinates_from_moments(T.permute(0, 2, 3, 4, 1), b)  # [G, 2, 2, 1, M]

    # direct path: S(Omega_m) = sum_gin sum_m' w_m' mu_s(eta = Omega_m'.Omega_m) psi[gin, m']
    eta = (quadrature.omega @ quadrature.omega.T).reshape(-1)  # [M*M] (m, m')
    S_dir = torch.zeros(G, 2, 2, 1, M, dtype=F64)
    for i in range(2):
        for j in range(2):
            r = rho[:, i, j, 0][:, None].expand(2, eta.numel())
            k = transfer_eta_at(library, r, eta).reshape(G, G, M, M)  # [gin, gout, m, m']
            S_dir[:, i, j, 0] = torch.einsum("abmp,p,ap->bm", k, quadrature.weight, psi)
    err = (S_mom - S_dir).abs().max().item()
    scale = S_dir.abs().max().item()
    assert err < 1e-8 * max(1.0, scale), (err, scale)


def test_scatter_operator_ld_axis_and_directions(device):
    G, L = 3, 2
    g = torch.Generator().manual_seed(11)
    mu_vol = torch.rand(G, G, 4, 3, 2, 2, generator=g, dtype=F64).to(device)
    phi_linear_discontinuous = torch.randn(G, num_spherical_harmonics(L), 4, 3, 2, 2, generator=g, dtype=F64).to(device)
    T = scatter_source_moments(mu_vol, phi_linear_discontinuous)
    assert T.shape == (G, num_spherical_harmonics(L), 4, 3, 2, 2)
    for a in range(4):
        Ta = scatter_source_moments(mu_vol, phi_linear_discontinuous[:, :, a])
        assert torch.allclose(T[:, :, a], Ta, atol=1e-13)
    # explicit reference
    lofn = spherical_harmonic_degree_index(L).to(device)
    reference = torch.einsum("abnxyz,anqxyz->bnqxyz", mu_vol[:, :, lofn], phi_linear_discontinuous)
    assert torch.allclose(T, reference, atol=1e-12)
    assert T.is_contiguous()
    # moment-major layout (used by source_iteration): same values, T.movedim(1, 0) contiguous
    Tm = scatter_source_moments(mu_vol, phi_linear_discontinuous, moment_major=True)
    assert torch.equal(Tm, T) and Tm.movedim(1, 0).is_contiguous()
    # non-contiguous input (the cell-mean slice used for the multiple-collision source)
    assert torch.allclose(scatter_source_moments(mu_vol, phi_linear_discontinuous[:, :, 0]), reference[:, :, 0],
                          atol=1e-12)
    quadrature = product_quadrature(4, 8, device=device)
    b = spherical_harmonic_basis(L, quadrature)
    S = scatter_source_at_directions(T, b.Y[:5])
    assert S.shape == (5, G, 4, 3, 2, 2)
    S_reference = ordinates_from_moments(T.movedim(1, -1), b)[..., :5].movedim(-1, 0)
    assert torch.allclose(S, S_reference, atol=1e-12)
    with pytest.raises(ValueError):
        scatter_source_moments(mu_vol[:, :, :2], phi_linear_discontinuous)  # degree 2 > transfer order 1


# ---- (g) cone sets ---------------------------------------------------------------------------------

@pytest.mark.parametrize("axis", [(0.0, 0.0, 1.0), (0.3, -0.8, 0.2), (-1.0, 0.0, 0.0)])
def test_hemisphere_quadrature(axis):
    a = torch.tensor(axis, dtype=F64)
    a = a / torch.linalg.norm(a)
    q = hemisphere_quadrature(axis, 6, 12)
    assert abs(float(q.weight.sum()) - 2 * math.pi) < 1e-12
    cos = q.omega @ a
    assert cos.min().item() > 0
    assert abs(float((q.weight * cos).sum()) - math.pi) < 1e-12
    assert torch.allclose(torch.linalg.norm(q.omega, dim=-1), torch.ones(q.num_directions, dtype=F64))
    # narrower cone: solid angle and cosine integral pi (1 - c0^2)
    th = 0.4
    q2 = hemisphere_quadrature(axis, 5, 8, max_polar_angle=th)
    c0 = math.cos(th)
    assert abs(float(q2.weight.sum()) - 2 * math.pi * (1 - c0)) < 1e-12
    assert abs(float((q2.weight * (q2.omega @ a)).sum()) - math.pi * (1 - c0 ** 2)) < 1e-12
    assert (q2.omega @ a).min().item() >= c0 - 1e-12
    # azimuthal balance: first moment perpendicular to the axis vanishes
    perp = (q.weight[:, None] * (q.omega - cos[:, None] * a)).sum(0)
    assert perp.abs().max().item() < 1e-12


# ---- (h) device / dtype ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_builders_device_dtype(device, dtype):
    qs = [
        product_quadrature(4, 8, device=device, dtype=dtype),
        hemisphere_quadrature([0.0, 1.0, 0.0], 4, 8, device=device, dtype=dtype),
        hemisphere_quadrature(torch.tensor([0.0, 1.0, 0.0], device=device), 4, 8, dtype=dtype),
    ]
    for q in qs:
        assert isinstance(q, Quadrature)
        assert q.omega.device.type == device.type and q.weight.device.type == device.type
        assert q.octant.device.type == device.type and q.octant.dtype == torch.int64
        assert q.omega.dtype == dtype and q.weight.dtype == dtype
        b = spherical_harmonic_basis(3, q)
        assert b.Y.device.type == device.type and b.Y.dtype == dtype and b.degree_of_moment.device.type == device.type
    x = torch.linspace(-1, 1, 5, device=device, dtype=dtype)
    assert legendre_polynomials(3, x).dtype == dtype and legendre_polynomials(3, x).device.type == device.type


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("linear_discontinuous", [True, False])
def test_scatter_moments_kernel_matches_torch(linear_discontinuous):
    """float32 CUDA: the fused scattering-moments kernel equals the PyTorch path (every non-zero transfer block added in
    place) to float32 rounding, from a strided view of cell-major flux moments, into each of the three storages."""
    from citrus.scatter_solver.angular_quadrature import _scatter_source_moments_torch, scatter_source_moments
    from citrus.scatter_solver.kernels import scattering_kernels

    device = torch.device("cuda")
    if not scattering_kernels.kernels_usable(device)[0]:
        pytest.skip("scattering-moments kernel unusable")
    g = torch.Generator().manual_seed(4)
    G, L, spatial = 4, 3, (5, 6, 7)
    transfer = torch.rand(G, G, L + 2, *spatial, generator=g)
    transfer[2, 0] = 0.0  # an upscatter block, skipped by the PyTorch path
    transfer = transfer.to(device)
    ld = (4,) if linear_discontinuous else ()
    storage = torch.randn((L + 1) ** 2, *spatial, G, *ld, generator=g).to(device)
    phi = scattering_kernels.cell_major_view(storage, 3)  # [G, N, (4), *spatial], not contiguous
    assert not phi.is_contiguous()
    reference = _scatter_source_moments_torch(transfer, phi, torch.zeros(G, *phi.shape[1:], device=device))
    for keywords, storage_order in (({}, None), (dict(moment_major=True), (1, 0)),
                                    (dict(cell_major=True), (1, 3, 4, 5, 0, 2) if ld else (1, 2, 3, 4, 0))):
        out = scatter_source_moments(transfer, phi, **keywords)
        assert out.shape == reference.shape
        torch.testing.assert_close(out, reference, rtol=1e-6, atol=1e-6 * float(reference.abs().max()))
        layout = out if storage_order is None else out.permute(*storage_order) if len(storage_order) > 2 \
            else out.movedim(1, 0)
        assert layout.is_contiguous(), keywords
