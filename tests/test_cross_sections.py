import torch

from citrus.scatter_solver.cross_sections import synthetic_library, mu_t_volume, transfer_moments_volume, transfer_eta_at


def test_synthetic_library_consistency():
    lib = synthetic_library(num_groups=4, legendre_order=5)
    # l=0 moment equals 2 pi int d mu/dOmega d eta
    eta = lib.eta_grid
    integ = 2 * torch.pi * torch.trapz(lib.mu_s_eta, eta, dim=-1)
    assert torch.allclose(integ, lib.mu_s_l[..., 0], atol=1e-6)
    # no upscatter
    G = lib.num_groups
    for gi in range(G):
        for go in range(gi):
            assert lib.mu_s_l[:, gi, go].abs().max() == 0


def test_synthetic_fine_table():
    for n, spread in ((4, 0.0), (5, 0.6), (1, 0.3)):
        lib = synthetic_library(num_groups=3, num_fine_per_group=n, fine_spread=spread)
        assert lib.num_fine_energies == 3 * n and lib.fine_group.dtype == torch.int64
        assert torch.allclose(lib.fine_weight.sum(), torch.tensor(1.0, dtype=torch.float64))
        edges = lib.group_edges
        assert bool(((lib.fine_energy < edges[lib.fine_group]) & (lib.fine_energy > edges[lib.fine_group + 1])).all())
        for g in range(3):
            sel = lib.fine_group == g
            w = lib.fine_weight[sel]
            assert torch.allclose(w.sum(), lib.spectrum_fraction[g])
            # the weighted group mean of the fine attenuation is the group constant
            assert torch.allclose(lib.fine_mu_t[:, sel] @ w / w.sum(), lib.mu_t[:, g], rtol=1e-14)
            if spread > 0 and n > 1:  # lower energy, larger attenuation
                e_sel, m_sel = lib.fine_energy[sel], lib.fine_mu_t[0, sel]
                assert bool((m_sel[torch.argsort(e_sel)].diff() < 0).all())
            if spread == 0:
                assert bool((lib.fine_mu_t[:, sel] == lib.mu_t[:, g:g + 1]).all())


def test_group_transmission_sums_fine_energies(device):
    from citrus.scatter_solver.cross_sections import group_transmission

    lib = synthetic_library(num_groups=3, num_fine_per_group=5, fine_spread=0.5)
    path = torch.tensor([[0.0, 1.0, 3.0], [0.0, 0.5, 0.0]], dtype=torch.float64, device=device)  # [M, 3 rays]
    T = group_transmission(lib, path)
    assert T.shape == (3, 3) and T.device.type == device.type
    assert torch.allclose(T[:, 0], lib.spectrum_fraction.to(device))  # no material: the source spectrum
    tau = lib.fine_mu_t.to(device).t() @ path  # [E, 3]
    ref = torch.zeros(3, 3, dtype=torch.float64, device=device)
    ref.index_add_(0, lib.fine_group.to(device), lib.fine_weight.to(device)[:, None] * torch.exp(-tau))
    torch.testing.assert_close(T, ref, rtol=1e-14, atol=0)
    T32 = group_transmission(lib.to(device, torch.float32), path.float())
    assert T32.dtype == torch.float32
    torch.testing.assert_close(T32.double(), T, rtol=1e-6, atol=1e-7)


def test_volume_helpers(device):
    lib = synthetic_library()
    rho = torch.zeros(2, 2, 3, 4, device=device)
    rho[0] = 1.0
    rho[1, 0, 0, 0] = 0.85
    mu = mu_t_volume(lib, rho)
    assert mu.shape == (lib.num_groups, 2, 3, 4)
    expect = lib.mu_t[0].to(device, mu.dtype)[:, None, None].expand(-1, 3, 4)
    assert torch.allclose(mu[:, 1], expect)  # row 1 is pure water at unit density
    expect0 = (lib.mu_t[0] + 0.85 * lib.mu_t[1]).to(device, mu.dtype)
    assert torch.allclose(mu[:, 0, 0, 0], expect0)
    tm = transfer_moments_volume(lib, rho)
    assert tm.shape == (lib.num_groups, lib.num_groups, lib.legendre_order + 1, 2, 3, 4)
    eta = torch.full((2, 3, 4), 0.3, device=device)
    te = transfer_eta_at(lib, rho, eta)
    assert te.shape == (lib.num_groups, lib.num_groups, 2, 3, 4)
    # value at eta=0.3 for water-only voxel equals analytic a(1+b eta)/4pi with a=0.15, b=0.5 (g_in=g_out=0)
    assert torch.allclose(te[0, 0, 1, 1, 1], torch.tensor(0.15 * (1 + 0.5 * 0.3) / (4 * torch.pi), device=device, dtype=te.dtype), rtol=1e-5)
