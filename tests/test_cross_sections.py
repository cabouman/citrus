import pytest
import torch

from citrus.scatter_solver.cross_sections import (synthetic_library, mu_t_volume, transfer_moments_volume,
                                                  transfer_eta_at)


def test_synthetic_library_consistency():
    library = synthetic_library(num_groups=4, legendre_order=5)
    # l=0 moment equals 2 pi int d mu/dOmega d eta
    eta = library.eta_grid
    integ = 2 * torch.pi * torch.trapz(library.mu_s_eta, eta, dim=-1)
    assert torch.allclose(integ, library.mu_s_l[..., 0], atol=1e-6)
    # no upscatter
    G = library.num_groups
    for gi in range(G):
        for go in range(gi):
            assert library.mu_s_l[:, gi, go].abs().max() == 0


def test_synthetic_fine_table():
    for n, spread in ((4, 0.0), (5, 0.6), (1, 0.3)):
        library = synthetic_library(num_groups=3, num_fine_per_group=n, fine_spread=spread)
        assert library.num_fine_energies == 3 * n and library.fine_group.dtype == torch.int64
        assert torch.allclose(library.fine_weight.sum(), torch.tensor(1.0, dtype=torch.float64))
        edges = library.group_edges
        assert bool(((library.fine_energy < edges[library.fine_group])
                     & (library.fine_energy > edges[library.fine_group + 1])).all())
        for g in range(3):
            sel = library.fine_group == g
            w = library.fine_weight[sel]
            assert torch.allclose(w.sum(), library.spectrum_fraction[g])
            # the weighted group mean of the fine attenuation is the group constant
            assert torch.allclose(library.fine_mu_t[:, sel] @ w / w.sum(), library.mu_t[:, g], rtol=1e-14)
            if spread > 0 and n > 1:  # lower energy, larger attenuation
                e_sel, m_sel = library.fine_energy[sel], library.fine_mu_t[0, sel]
                assert bool((m_sel[torch.argsort(e_sel)].diff() < 0).all())
            if spread == 0:
                assert bool((library.fine_mu_t[:, sel] == library.mu_t[:, g:g + 1]).all())


def test_group_transmission_sums_fine_energies(device):
    from citrus.scatter_solver.cross_sections import group_transmission

    library = synthetic_library(num_groups=3, num_fine_per_group=5, fine_spread=0.5)
    path = torch.tensor([[0.0, 1.0, 3.0], [0.0, 0.5, 0.0]], dtype=torch.float64, device=device)  # [M, 3 rays]
    T = group_transmission(library, path)
    assert T.shape == (3, 3) and T.device.type == device.type
    assert torch.allclose(T[:, 0], library.spectrum_fraction.to(device))  # no material: the source spectrum
    tau = library.fine_mu_t.to(device).t() @ path  # [E, 3]
    reference = torch.zeros(3, 3, dtype=torch.float64, device=device)
    reference.index_add_(0, library.fine_group.to(device), library.fine_weight.to(device)[:, None] * torch.exp(-tau))
    torch.testing.assert_close(T, reference, rtol=1e-14, atol=0)
    T32 = group_transmission(library.to(device, torch.float32), path.float())
    assert T32.dtype == torch.float32
    torch.testing.assert_close(T32.double(), T, rtol=1e-6, atol=1e-7)


def test_volume_helpers(device):
    library = synthetic_library()
    rho = torch.zeros(2, 2, 3, 4, device=device)
    rho[0] = 1.0
    rho[1, 0, 0, 0] = 0.85
    mu = mu_t_volume(library, rho)
    assert mu.shape == (library.num_groups, 2, 3, 4)
    expect = library.mu_t[0].to(device, mu.dtype)[:, None, None].expand(-1, 3, 4)
    assert torch.allclose(mu[:, 1], expect)  # row 1 is pure water at unit density
    expect0 = (library.mu_t[0] + 0.85 * library.mu_t[1]).to(device, mu.dtype)
    assert torch.allclose(mu[:, 0, 0, 0], expect0)
    tm = transfer_moments_volume(library, rho)
    assert tm.shape == (library.num_groups, library.num_groups, library.legendre_order + 1, 2, 3, 4)
    eta = torch.full((2, 3, 4), 0.3, device=device)
    te = transfer_eta_at(library, rho, eta)
    assert te.shape == (library.num_groups, library.num_groups, 2, 3, 4)
    # value at eta=0.3 for water-only voxel equals analytic a(1+b eta)/4pi with a=0.15, b=0.5 (g_in=g_out=0)
    assert torch.allclose(te[0, 0, 1, 1, 1], torch.tensor(0.15 * (1 + 0.5 * 0.3) / (4 * torch.pi), device=device,
                                                          dtype=te.dtype), rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_group_transmission_kernel_matches_torch():
    """float32 CUDA: the fused transmission kernel (fine energies summed per ray, group by group) equals the PyTorch
    path to float32 rounding, for a subset of the materials, zero and long paths and a non-contiguous path tensor."""
    from citrus.scatter_solver.cross_sections import _group_transmission_torch, group_transmission
    from citrus.scatter_solver.kernels import transmission_kernels

    device = torch.device("cuda")
    if not transmission_kernels.kernels_usable(device)[0]:
        pytest.skip("transmission kernel unusable")
    library = synthetic_library(num_groups=4, num_materials=5, num_fine_per_group=7, fine_spread=0.5)
    g = torch.Generator().manual_seed(2)
    materials = [1, 3, 4]
    path = (torch.rand(5, 9, 11, generator=g) * 30.0).to(device)[materials[::-1]].flip(0)  # a gathered, flipped view
    path[:, 0, 0] = 0.0
    path[:, 1, 1] = 300.0
    assert transmission_kernels.use_kernel(library, path)
    out = group_transmission(library, path, materials=materials)
    reference = _group_transmission_torch(library, path, 2 ** 22, materials)
    assert out.shape == reference.shape == (4, 9, 11)
    torch.testing.assert_close(out, reference, rtol=1e-6, atol=1e-7 * float(reference.abs().max()))
    torch.testing.assert_close(out[:, 0, 0], library.spectrum_fraction.to(device, torch.float32))
