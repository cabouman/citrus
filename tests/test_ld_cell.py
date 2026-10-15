import torch

from citrus.scatter_solver.ld_cell import ld_cell_rhs, ld_cell_solve, ld_cell_matrix


def test_closed_form_matches_dense_solve():
    torch.manual_seed(0)
    D, G, N = 5, 2, 7
    omega = torch.randn(D, 3, dtype=torch.float64)
    omega = omega / omega.norm(dim=1, keepdim=True)
    delta = torch.tensor([0.8, 1.1, 1.3], dtype=torch.float64)
    c = omega.abs() / delta
    s = torch.sign(omega)
    mu = torch.rand(G, N, dtype=torch.float64) * 0.5 + 0.01
    source = torch.randn(D, G, 4, N, dtype=torch.float64)
    nbr_psi0 = torch.randn(D, G, 3, N, dtype=torch.float64)
    nbr_psia = torch.randn(D, G, 3, 3, N, dtype=torch.float64)
    r0, ra = ld_cell_rhs(c, s, source, nbr_psi0, nbr_psia)
    psi0, psia = ld_cell_solve(c, s, mu, r0, ra)
    for d in range(D):
        for g in range(G):
            for n in range(N):
                A = ld_cell_matrix(c[d], s[d], float(mu[g, n]))
                # rebuild the rhs by hand from the definitions
                pb = torch.stack([nbr_psi0[d, g, a, n] + s[d, a] * nbr_psia[d, g, a, a, n] for a in range(3)])
                R0 = source[d, g, 0, n] + (c[d] * pb).sum()
                Ra = torch.stack([
                    source[d, g, 1 + a, n] - 3 * s[d, a] * c[d, a] * pb[a]
                    + sum(c[d, b] * nbr_psia[d, g, b, a, n] for b in range(3) if b != a)
                    for a in range(3)
                ])
                assert torch.allclose(r0[d, g, n], R0) and torch.allclose(ra[d, g, :, n], Ra)
                x = torch.linalg.solve(A, torch.cat([R0[None], Ra]))
                assert torch.allclose(psi0[d, g, n], x[0], atol=1e-12)
                assert torch.allclose(psia[d, g, :, n], x[1:], atol=1e-12)


def test_pure_absorber_limits():
    # slab along x, mu -> 0: outgoing average equals incoming; mu large: fluence decays
    c = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    s = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    src = torch.zeros(1, 1, 4, 1, dtype=torch.float64)
    nbr0 = torch.zeros(1, 1, 3, 1, dtype=torch.float64)
    nbr0[0, 0, 0, 0] = 2.0  # incoming from the -x neighbor
    nbra = torch.zeros(1, 1, 3, 3, 1, dtype=torch.float64)
    r0, ra = ld_cell_rhs(c, s, src, nbr0, nbra)
    psi0, psia = ld_cell_solve(c, s, torch.full((1, 1), 1e-10, dtype=torch.float64), r0, ra)
    assert abs(psi0.item() - 2.0) < 1e-8 and abs(psia[0, 0, 0, 0].item()) < 1e-8
    psi0b, _ = ld_cell_solve(c, s, torch.full((1, 1), 5.0, dtype=torch.float64), r0, ra)
    assert 0 < psi0b.item() < 2.0


def test_device_and_dtype(device):
    c = torch.rand(3, 3, device=device) + 0.1
    s = torch.ones(3, 3, device=device)
    src = torch.rand(3, 2, 4, 5, device=device)
    nbr0 = torch.rand(3, 2, 3, 5, device=device)
    nbra = torch.rand(3, 2, 3, 3, 5, device=device)
    r0, ra = ld_cell_rhs(c, s, src, nbr0, nbra)
    psi0, psia = ld_cell_solve(c, s, torch.rand(2, 5, device=device), r0, ra)
    assert psi0.device.type == device.type and psi0.dtype == torch.float32 and psia.shape == (3, 2, 3, 5)
