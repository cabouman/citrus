"""Linear-discontinuous (LD) Galerkin cell equations on a cuboid and their closed-form solve.

Derivation: docs/source/method.rst, "Linear-discontinuous cells" (verified against a dense Galerkin assembly to
2e-15, tests/test_ld_cell.py).

For one direction with signs s_a = sign(Omega_a) and c_a = |Omega_a| / Delta_a, a in {x, y, z},
the four LD unknowns (psi_0, psi_x, psi_y, psi_z) of a cell satisfy

    (E0)  (sum_a c_a + mu) psi_0 + sum_a s_a c_a psi_a        = R_0
    (Ea)  -3 s_a c_a psi_0 + d_a psi_a                           = R_a,   d_a = 3 c_a + sum_{b != a} c_b + mu

with
    R_0 = S_0 + sum_a c_a psibar_in_a
    R_a = S_a - 3 s_a c_a psibar_in_a + sum_{b != a} c_b psi_a[N_b]

where psibar_in_a = psi_0[N_a] + s_a psi_a[N_a] is the face-averaged incoming fluence on the a-face
(N_a = upstream neighbor along a, zero for vacuum) and psi_a[N_b] is the a-moment of the upstream
neighbor along b (transverse coupling). Solution:

    psi_0 = (R_0 - sum_a s_a c_a R_a / d_a) / (sum_a c_a + mu + 3 sum_a c_a^2 / d_a)
    psi_a = (R_a + 3 s_a c_a psi_0) / d_a

Axis order everywhere: a = 0 -> x (cols), 1 -> y (rows), 2 -> z (slices).
"""

from __future__ import annotations

import torch
from torch import Tensor


def ld_cell_rhs(
    c: Tensor,
    s: Tensor,
    source: Tensor,
    nbr_psi0: Tensor,
    nbr_psia: Tensor,
) -> tuple[Tensor, Tensor]:
    """Right-hand sides R_0, R_a from the source moments and upstream neighbor moments.

    c, s:      [D, 3]            direction coefficients |Omega_a|/Delta_a and signs (+-1), or
               [D, ..., 3, N]    (any c.dim() != 2) already shaped to broadcast against nbr_psi0, e.g.
                                 [D, 1, 3, N] when c and s vary per cell (sweeps of several octants)
    source:    [D, ..., 4, N]    LD moments of the source for each direction (axis order 0,x,y,z)
    nbr_psi0:  [D, ..., 3, N]    psi_0 of the upstream neighbor along a (index a), zero for vacuum
    nbr_psia:  [D, ..., 3, 3, N] psi_b of the upstream neighbor along a: index [a, b]
    Returns (R_0 [D, ..., N], R_a [D, ..., 3, N]).
    The "..." dims (e.g. groups) are shared; c and s broadcast over them.
    """
    if c.dim() == 2:
        nd = source.dim() - 2  # number of leading dims before (4, N): D plus "..."
        shp = (c.shape[0],) + (1,) * (nd - 1) + (3, 1)
        cc = c.reshape(shp)  # [D, 1..., 3, 1]
        ss = s.reshape(shp)
    else:
        cc, ss = c, s  # [D, 1..., 3, N] (broadcastable), per-cell coefficients
    s0 = source[..., 0, :]
    sa = source[..., 1:, :]  # [D, ..., 3, N]
    # face-averaged incoming fluence on each inflow face a: psi_0[N_a] + s_a psi_a[N_a]
    psibar_in = nbr_psi0 + ss * torch.diagonal(nbr_psia, dim1=-3, dim2=-2).movedim(-1, -2)  # [D, ..., 3, N]
    r0 = s0 + (cc * psibar_in).sum(dim=-2)
    # transverse coupling: sum_{b != a} c_b psi_a[N_b]  (nbr_psia[b, a] = a-moment of neighbor along b)
    ccb = cc.unsqueeze(-2)  # [D, 1..., 3(b), 1, 1]
    trans = (ccb * nbr_psia).sum(dim=-3)  # sums over b including b == a -> subtract the diagonal term
    diag = cc * torch.diagonal(nbr_psia, dim1=-3, dim2=-2).movedim(-1, -2)  # c_a psi_a[N_a]
    trans = trans - diag
    ra = sa - 3.0 * ss * cc * psibar_in + trans
    return r0, ra


def ld_cell_solve(c: Tensor, s: Tensor, mu: Tensor, r0: Tensor, ra: Tensor) -> tuple[Tensor, Tensor]:
    """Closed-form solve of the LD cell system.

    c, s: [D, 3], or (c.dim() != 2) already shaped [D, ..., 3, N] broadcastable against ra (see ld_cell_rhs);
    mu: [..., N] broadcastable to r0; r0: [D, ..., N]; ra: [D, ..., 3, N].
    Returns (psi_0 [D, ..., N], psi_a [D, ..., 3, N]).
    """
    nd = r0.dim()
    if c.dim() == 2:
        shp = (c.shape[0],) + (1,) * (nd - 2) + (3, 1)
        cc = c.reshape(shp)
        ss = s.reshape(shp)
    else:
        cc, ss = c, s
    csum = cc.sum(dim=-2)  # [D, 1..., 1]
    mu_e = mu.unsqueeze(-2) if mu.dim() == nd - 1 else mu  # allow mu [..., N] or [..., 1, N]
    d = 3.0 * cc + (csum.unsqueeze(-2) - cc) + mu_e  # [D, ..., 3, N]
    q = cc / d  # c_a / d_a
    denom = csum + mu_e.squeeze(-2) + 3.0 * (cc * q).sum(dim=-2)
    psi0 = (r0 - (ss * q * ra).sum(dim=-2)) / denom
    psia = (ra + 3.0 * ss * cc * psi0.unsqueeze(-2)) / d
    return psi0, psia


def ld_cell_matrix(c: Tensor, s: Tensor, mu: float) -> Tensor:
    """Dense 4x4 LD cell matrix for one direction (reference / testing). c, s: [3]."""
    A = torch.zeros(4, 4, dtype=c.dtype, device=c.device)
    A[0, 0] = c.sum() + mu
    for a in range(3):
        A[0, 1 + a] = s[a] * c[a]
        A[1 + a, 0] = -3.0 * s[a] * c[a]
        A[1 + a, 1 + a] = 3.0 * c[a] + (c.sum() - c[a]) + mu
    return A
