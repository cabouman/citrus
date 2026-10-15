"""Standalone 1D linear-discontinuous (LD) slab sweep (numpy, float64): the oracle for the 3D sweep.

Slab of K cells of width dx along one axis, vacuum on both sides, one direction with cosine mu_x != 0.
Per cell: total attenuation mu[k], source moments S0[k] (mean) and S1[k] (slope moment, local coordinate
u = 2(x - x_k)/dx in [-1, 1], f ~ f0 + f1 u, f1 = (3/dx) int f u dx).

Textbook LD equations written along the direction of travel (t = s u, s = sign(mu_x); slopes in the
travel coordinate are p1 = s psi1, Q1 = s S1), with c = |mu_x| / dx, psi_in the outgoing edge value of
the upstream cell (0 at the vacuum boundary) and psi_out = psi0 + p1:

    c (psi_out - psi_in) + mu psi0                 = S0
    3 c (psi_out + psi_in - 2 psi0) + mu p1        = Q1

Each cell is a 2x2 linear solve; cells are visited in the direction of travel. This code is independent
of citrus (it shares only the textbook equations).
"""

from __future__ import annotations

import numpy as np


def ld_slab_sweep(mu: np.ndarray, S0: np.ndarray, S1: np.ndarray, dx: float, mu_x: float) -> tuple[np.ndarray, np.ndarray]:
    """Sweep a 1D LD slab.

    mu, S0, S1 : [K] per-cell total attenuation (1/cm) and source moments (per steradian)
    dx         : cell width (cm); mu_x: direction cosine along the slab axis (non-zero)
    Returns (psi0 [K], psi1 [K]) in the fixed (axis) coordinate u.
    """
    mu = np.asarray(mu, dtype=np.float64)
    S0 = np.asarray(S0, dtype=np.float64)
    S1 = np.asarray(S1, dtype=np.float64)
    K = mu.shape[0]
    if mu_x == 0:
        raise ValueError("mu_x must be non-zero")
    s = 1.0 if mu_x > 0 else -1.0
    c = abs(mu_x) / dx
    psi0 = np.zeros(K)
    psi1 = np.zeros(K)
    order = range(K) if s > 0 else range(K - 1, -1, -1)
    psi_in = 0.0
    for k in order:
        Q1 = s * S1[k]
        # unknowns (psi0, p1):  (c + mu) psi0 + c p1 = S0 + c psi_in
        #                       -3c psi0 + (3c + mu) p1 = Q1 - 3c psi_in
        A = np.array([[c + mu[k], c], [-3.0 * c, 3.0 * c + mu[k]]])
        b = np.array([S0[k] + c * psi_in, Q1 - 3.0 * c * psi_in])
        p0, p1 = np.linalg.solve(A, b)
        psi0[k] = p0
        psi1[k] = s * p1
        psi_in = p0 + p1
    return psi0, psi1
