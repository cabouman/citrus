"""Source iteration over collision order and the multiple-collision source.

Each sweep transports the scattering source of the previous collision order, starting from the first-collision source,
until the newest order is small against the sum of all orders; the scattering source of that sum is the
multiple-collision source of the scatter image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor

from .angular_quadrature import scatter_source_moments
from .transport_sweep import SourceFunction, SweepContext, sweep

__all__ = [
    "SourceIterationResult",
    "source_iteration",
    "scatter_moments",
    "scatter_source_function",
    "multiple_collision_source_moments",
]


@dataclass
class SourceIterationResult:
    """Result of ``source_iteration``.

    Attributes:
        phi_total (Tensor): [G, (L+1)^2, 4, R, C, S] spherical-harmonic / linear-discontinuous moments of the collided
            fluence, summed over collision orders.
        num_iterations (int): number of sweeps performed (collision orders included).
        history (list[float]): relative change ``||phi_n|| / ||phi_total||`` of the scalar-flux cell means after each
            sweep. Defaults to [].
        phi_norms (list[float]): norm ``||phi_n||`` of the scalar-flux cell means after each sweep. Defaults to [].
        converged (bool): True if the last relative change was below the tolerance. Defaults to False.
    """

    phi_total: Tensor
    num_iterations: int
    history: list[float] = field(default_factory=list)
    phi_norms: list[float] = field(default_factory=list)
    converged: bool = False


def scatter_moments(mu_s_l_volume: Tensor, phi: Tensor) -> Tensor:
    """Scattering-source spherical-harmonic moments ``T[g, n] = sum_g_in mu_s_l_volume[g_in, g, l(n)] phi[g_in, n]``.

    T is a view of the sweep's cell-major storage ``[N, *spatial, G_out, (4)]``, so ``scatter_source_function``
    contracts over it without a copy.

    Args:
        mu_s_l_volume (Tensor): ``[G_in, G_out, L+1, *spatial]`` Legendre transfer moments in 1/cm.
        phi (Tensor): ``[G_in, N, *rest]`` moments, N = (L_phi+1)^2 with L_phi <= L, rest = spatial or (4, spatial).

    Returns:
        Tensor: ``[G_out, N, *rest]`` moments T, a non-contiguous view of cell-major storage.
    """
    return scatter_source_moments(mu_s_l_volume, phi, cell_major=True)


def scatter_source_function(sweep_context: SweepContext, T: Tensor) -> SourceFunction:
    """Source function for the sweep from the spherical-harmonic moments T of a scattering source.

    The callable maps reordered direction indices to the linear-discontinuous source moments
    ``sum_n Y[direction_index, n] T[:, n]`` with one matrix product, cell-major (the sweep's layout, ``cell_major``),
    written into ``out`` when a buffer is given or added to it with ``accumulate=True``.

    Args:
        sweep_context (SweepContext): sweep context with the spherical-harmonic basis Y of the sweep directions.
        T (Tensor): [G, (L+1)^2, 4, R, C, S] moments; one cell-major copy is made unless they come from
            ``scatter_moments``.

    Returns:
        SourceFunction: callable ``source(direction_index [Dsel], out=None, accumulate=False)`` returning
        [Dsel, R, C, S, G, 4] moments.

    Raises:
        ValueError: if T does not have the context's number of spherical-harmonic moments or shape.
    """
    if T.shape[1] != sweep_context.num_moments:
        raise ValueError(f"T has {T.shape[1]} spherical-harmonic moments, context has {sweep_context.num_moments}")
    if T.dim() != 6 or T.shape[2] != 4 or tuple(T.shape[3:]) != tuple(sweep_context.shape):
        raise ValueError(f"T must be [G, {sweep_context.num_moments}, 4, {sweep_context.shape}], got {tuple(T.shape)}")
    Y = sweep_context.basis.Y.to(device=T.device, dtype=T.dtype)
    G, num_moments = T.shape[0], T.shape[1]
    spatial = tuple(T.shape[3:])
    # cell-major moments [(L+1)^2, R, C, S, G, 4]: no copy when T is a view of cell-major storage
    moments_matrix = T.permute(1, 3, 4, 5, 0, 2).contiguous().view(num_moments, -1)

    def source(direction_index: Tensor, out: Optional[Tensor] = None, accumulate: bool = False) -> Tensor:
        Y_selected = Y.index_select(0, direction_index.to(Y.device))  # [Dsel, N_sh]
        if out is None:
            return (Y_selected @ moments_matrix).view((Y_selected.shape[0],) + spatial + (G, 4))
        flat = out.view(Y_selected.shape[0], -1)
        if accumulate:
            flat.addmm_(Y_selected, moments_matrix)
        else:
            torch.mm(Y_selected, moments_matrix, out=flat)
        return out

    source.cell_major = True  # type: ignore[attr-defined]
    return source


def multiple_collision_source_moments(mu_s_l_volume_coarse: Tensor, phi_total: Tensor) -> Tensor:
    """Spherical-harmonic moments of the multiple-collision source, the scattering operator applied to ``phi_total``.

    The source ``S_MC(r, Omega) = sum_n T[:, n] Y_n(Omega)`` is evaluated at the collapsed-cone directions with
    ``angular_quadrature.scatter_source_at_directions``.

    Args:
        mu_s_l_volume_coarse (Tensor): [G_in, G_out, L+1, R, C, S] Legendre transfer moments on the coarse grid in 1/cm.
        phi_total (Tensor): [G, (L+1)^2, 4, R, C, S] total collided fluence moments
            (``SourceIterationResult.phi_total``) of degree at most the transfer's Legendre order.

    Returns:
        Tensor: [G, (L+1)^2, 4, R, C, S] moments T, a view of moment-major storage (``scatter_moments``).
    """
    return scatter_moments(mu_s_l_volume_coarse, phi_total)


def _scalar_norms(*phis: Tensor) -> list[float]:
    """Frobenius norms of the scalar-flux cell means over all groups and cells, with one host synchronization.

    Args:
        *phis (Tensor): [G, (L+1)^2, 4, R, C, S] moments.

    Returns:
        list[float]: norm of ``phi[:, 0, 0]`` of each.
    """
    return torch.stack([torch.linalg.vector_norm(phi[:, 0, 0]) for phi in phis]).tolist()


def source_iteration(
    sweep_context: SweepContext,
    mu_t_coarse: Tensor,
    mu_s_l_volume_coarse: Tensor,
    first_collision_source_function: SourceFunction,
    tolerance: float,
    max_iterations: int,
    direction_chunk: Optional[int] = None,
    callback: Optional[Callable[[int, Tensor, float], None]] = None,
) -> SourceIterationResult:
    """Source iteration over collision order, starting from the first-collision source.

    The sweeps stop when ``||phi_n|| / ||phi_total||`` of the scalar-flux cell means falls below ``tolerance`` or after
    ``max_iterations`` sweeps.

    Args:
        sweep_context (SweepContext): sweep context of the coarse grid and the sweep quadrature.
        mu_t_coarse (Tensor): [G, R, C, S] total attenuation mu_t on the coarse grid in 1/cm.
        mu_s_l_volume_coarse (Tensor): [G_in, G_out, L_x+1, R, C, S] Legendre transfer moments in 1/cm, with
            L_x >= ``sweep_context.L``.
        first_collision_source_function (SourceFunction): maps reordered direction indices [Dsel] to the
            first-collision linear-discontinuous source moments [Dsel, G, 4, R, C, S].
        tolerance (float): stopping tolerance of the relative change.
        max_iterations (int): maximum number of sweeps (collision orders).
        direction_chunk (int, optional): passed to ``sweep``. Defaults to None.
        callback (Callable[[int, Tensor, float], None], optional): called as ``callback(n, phi_n, relative_change)``
            after each sweep (n from 1); it must not modify phi_n. Defaults to None.

    Returns:
        SourceIterationResult: the summed moments, the number of sweeps, the history and the convergence flag.

    Raises:
        ValueError: if max_iterations < 1.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    phi_total: Optional[Tensor] = None
    history: list[float] = []
    norms: list[float] = []
    converged = False
    source_function: Optional[SourceFunction] = first_collision_source_function
    n = 0
    for n in range(1, max_iterations + 1):
        phi_n = sweep(sweep_context, mu_t_coarse, source_function, direction_chunk=direction_chunk)
        source_function = None  # release the previous scattering-source moments before forming the next ones
        if phi_total is None:
            phi_total = phi_n.clone()
        else:
            phi_total.add_(phi_n)
        norm_n, norm_total = _scalar_norms(phi_n, phi_total)
        relative_change = norm_n / norm_total if norm_total > 0 else 0.0
        history.append(relative_change)
        norms.append(norm_n)
        if callback is not None:
            callback(n, phi_n, relative_change)
        if relative_change < tolerance:
            converged = True
            break
        if n == max_iterations:
            break
        T = scatter_moments(mu_s_l_volume_coarse, phi_n)
        del phi_n
        source_function = scatter_source_function(sweep_context, T)
        del T
    assert phi_total is not None
    return SourceIterationResult(phi_total=phi_total, num_iterations=n, history=history, phi_norms=norms,
                                 converged=converged)
