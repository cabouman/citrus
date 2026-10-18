"""Linear-discontinuous Galerkin cell equations on a cuboid and their closed-form solve.

The module forms the right-hand sides of the four-moment cell system of one direction from the source and the upstream
neighbor moments, solves the system in closed form and builds its dense 4 x 4 matrix for tests. The equations are
derived in docs/source/method.rst, "Linear-discontinuous cells"; axis a = 0, 1, 2 is x (cols), y (rows), z (slices).
"""

from __future__ import annotations

import torch
from torch import Tensor


def cell_right_hand_sides(
    c: Tensor,
    s: Tensor,
    source: Tensor,
    neighbor_psi0: Tensor,
    neighbor_psia: Tensor,
) -> tuple[Tensor, Tensor]:
    """Right-hand sides R_0 and R_a of the cell system from the source moments and the upstream neighbor moments.

    The "..." dimensions (such as groups) are shared; c and s broadcast over them.

    Args:
        c (Tensor): [D, 3] direction coefficients ``|Omega_a| / Delta_a`` in 1/cm, or (when not 2-D) already shaped
            [D, ..., 3, N] to broadcast, e.g. [D, 1, 3, N] when they vary per cell.
        s (Tensor): signs (+1 or -1) of the direction components, shaped like ``c``.
        source (Tensor): [D, ..., 4, N] linear-discontinuous source moments (0, x, y, z) of D directions in N cells.
        neighbor_psi0 (Tensor): [D, ..., 3, N] psi_0 of the upstream neighbor along each axis a, zero for vacuum.
        neighbor_psia (Tensor): [D, ..., 3, 3, N] moment b of the upstream neighbor along axis a at index [a, b].

    Returns:
        (Tensor, Tensor): R_0 [D, ..., N] and R_a [D, ..., 3, N].
    """
    if c.dim() == 2:
        num_dims = source.dim() - 2  # number of leading dims before (4, N): D plus "..."
        shape = (c.shape[0],) + (1,) * (num_dims - 1) + (3, 1)
        c_broadcast = c.reshape(shape)  # [D, 1..., 3, 1]
        s_broadcast = s.reshape(shape)
    else:
        c_broadcast, s_broadcast = c, s  # [D, 1..., 3, N] (broadcastable), per-cell coefficients
    source_mean = source[..., 0, :]
    source_slopes = source[..., 1:, :]  # [D, ..., 3, N]
    # face-averaged incoming fluence on each inflow face a: psi_0[N_a] + s_a psi_a[N_a]
    # [D, ..., 3, N]
    psibar_in = neighbor_psi0 + s_broadcast * torch.diagonal(neighbor_psia, dim1=-3, dim2=-2).movedim(-1, -2)
    r0 = source_mean + (c_broadcast * psibar_in).sum(dim=-2)
    # transverse coupling: sum_{b != a} c_b psi_a[N_b]  (neighbor_psia[b, a] = a-moment of neighbor along b)
    c_neighbor = c_broadcast.unsqueeze(-2)  # [D, 1..., 3(b), 1, 1]
    # sums over b including b == a -> subtract the diagonal term
    transverse = (c_neighbor * neighbor_psia).sum(dim=-3)
    diagonal = c_broadcast * torch.diagonal(neighbor_psia, dim1=-3, dim2=-2).movedim(-1, -2)  # c_a psi_a[N_a]
    transverse = transverse - diagonal
    ra = source_slopes - 3.0 * s_broadcast * c_broadcast * psibar_in + transverse
    return r0, ra


def cell_solve(c: Tensor, s: Tensor, mu: Tensor, r0: Tensor, ra: Tensor) -> tuple[Tensor, Tensor]:
    """Closed-form solve of the linear-discontinuous cell system for the moments psi_0 and psi_a.

    Args:
        c (Tensor): [D, 3] direction coefficients in 1/cm, or already shaped [D, ..., 3, N] as in
            ``cell_right_hand_sides``.
        s (Tensor): signs (+1 or -1) of the direction components, shaped like ``c``.
        mu (Tensor): [..., N] or [..., 1, N] total attenuation mu in 1/cm, broadcastable to ``r0``.
        r0 (Tensor): [D, ..., N] right-hand side R_0.
        ra (Tensor): [D, ..., 3, N] right-hand sides R_a.

    Returns:
        (Tensor, Tensor): psi_0 [D, ..., N] and psi_a [D, ..., 3, N].
    """
    num_dims = r0.dim()
    if c.dim() == 2:
        shape = (c.shape[0],) + (1,) * (num_dims - 2) + (3, 1)
        c_broadcast = c.reshape(shape)
        s_broadcast = s.reshape(shape)
    else:
        c_broadcast, s_broadcast = c, s
    c_sum = c_broadcast.sum(dim=-2)  # [D, 1..., 1]
    mu_broadcast = mu.unsqueeze(-2) if mu.dim() == num_dims - 1 else mu  # allow mu [..., N] or [..., 1, N]
    d = 3.0 * c_broadcast + (c_sum.unsqueeze(-2) - c_broadcast) + mu_broadcast  # [D, ..., 3, N]
    c_over_d = c_broadcast / d  # c_a / d_a
    denominator = c_sum + mu_broadcast.squeeze(-2) + 3.0 * (c_broadcast * c_over_d).sum(dim=-2)
    psi0 = (r0 - (s_broadcast * c_over_d * ra).sum(dim=-2)) / denominator
    psia = (ra + 3.0 * s_broadcast * c_broadcast * psi0.unsqueeze(-2)) / d
    return psi0, psia


def cell_matrix(c: Tensor, s: Tensor, mu: float) -> Tensor:
    """Dense 4 x 4 linear-discontinuous cell matrix of one direction, for reference and tests.

    Args:
        c (Tensor): [3] direction coefficients in 1/cm.
        s (Tensor): [3] signs (+1 or -1) of the direction components.
        mu (float): total attenuation in 1/cm.

    Returns:
        Tensor: [4, 4] matrix in the moment order (0, x, y, z).
    """
    A = torch.zeros(4, 4, dtype=c.dtype, device=c.device)
    A[0, 0] = c.sum() + mu
    for a in range(3):
        A[0, 1 + a] = s[a] * c[a]
        A[1 + a, 0] = -3.0 * s[a] * c[a]
        A[1 + a, 1 + a] = 3.0 * c[a] + (c.sum() - c[a]) + mu
    return A
