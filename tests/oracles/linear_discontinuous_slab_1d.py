"""Standalone 1D linear-discontinuous slab sweep (numpy, float64), the oracle for the 3D sweep.

A slab of K cells of width dx, with vacuum on both sides, is swept along one direction with cosine mu_x; each cell
has a mean and a slope moment in the local coordinate ``u = 2(x - x_k)/dx``. Each cell is a 2 x 2 solve of the
textbook equations ``c (psi_out - psi_in) + mu psi0 = S0`` and ``3 c (psi_out + psi_in - 2 psi0) + mu p1 = Q1``,
written along the direction of travel with ``c = |mu_x| / dx`` and ``psi_out = psi0 + p1``. The code shares nothing
with citrus but these equations.
"""

from __future__ import annotations

import numpy as np


def linear_discontinuous_slab_sweep(mu: np.ndarray, S0: np.ndarray, S1: np.ndarray, dx: float,
                                    mu_x: float) -> tuple[np.ndarray, np.ndarray]:
    """Sweep a 1D linear-discontinuous slab cell by cell in the direction of travel.

    Args:
        mu (ndarray): [K] total attenuation per cell in cm^-1.
        S0 (ndarray): [K] mean source moment per cell, per sr.
        S1 (ndarray): [K] slope source moment per cell in the axis coordinate u, per sr.
        dx (float): cell width in cm.
        mu_x (float): direction cosine along the slab axis, non-zero.

    Returns:
        (psi0, psi1): [K] mean and [K] slope moments of the angular fluence in the axis coordinate u.
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
