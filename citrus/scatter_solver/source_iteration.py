"""Source iteration over collision order and the multiple-collision source.

    phi^(1)   = sweep(first-collision source)                                  (n = 1)
    phi^(n)   = sweep(S^(n)),  S^(n)(Omega_m) = sum_n' T^(n)[:, n'] Y_n'(Omega_m),
                T^(n) = scatter_source_moments(mu_s_l, phi^(n-1))              (n >= 2)
    phi_total = sum_n phi^(n)

phi^(n) is the n-times-collided fluence; the multiple-collision source of the scatter image is the scattering
source of phi_total, S_MC(r, Omega) = sum_n T[:, n](r) Y_n(Omega) with T = scatter_source_moments(mu_s_l, phi_total).

stopped when ||phi^(n)[:, 0, 0]|| / ||phi_total[:, 0, 0]|| < tolerance (Frobenius norms of the scalar-flux
cell averages over all groups and cells) or after max_iterations sweeps. Each phi^(n) is the spherical-harmonic /
linear-discontinuous moment field [G, (L+1)^2, 4, R, C, S] of the n-times-collided fluence (G energy groups, (L+1)^2
spherical-harmonic moments up to degree L, 4 linear-discontinuous moments, R x C x S transport cells).

Memory: the scattering-source moments are formed by `scatter_moments`, i.e.
`angular_quadrature.scatter_source_moments(..., moment_major=True)`, accumulated pair by pair (g_in, g_out) directly
into a moment-major buffer, so the only temporary is one [(L+1)^2, *spatial] transfer gather. Peak during an iteration:
phi_total + T + the sweep's accumulator (3 fields of G (L+1)^2 4 N values) + the sweep buffers
(2 x 8 Dc N 4G values).
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
    """Result of `source_iteration`.

    phi_total      : [G, (L+1)^2, 4, R, C, S] sum over collision orders of the spherical-harmonic /
                     linear-discontinuous moments of the collided fluence (linear-discontinuous layout [0, x, y, z]).
    num_iterations : number of sweeps performed (= collision orders included).
    history        : relative change ||phi_n[:, 0, 0]|| / ||phi_total[:, 0, 0]|| after each iteration
                     (history[0] == 1 unless the first-collision fluence vanishes).
    phi_norms      : ||phi_n[:, 0, 0]|| per iteration.
    converged      : True if the last relative change was < tolerance.
    """

    phi_total: Tensor
    num_iterations: int
    history: list[float] = field(default_factory=list)
    phi_norms: list[float] = field(default_factory=list)
    converged: bool = False


def scatter_moments(mu_s_l_volume: Tensor, phi: Tensor) -> Tensor:
    """Scattering-source spherical-harmonic moments T[g, n] = sum_{g_in} mu_s_l_volume[g_in, g, l(n)] phi[g_in, n].

    Thin wrapper: `angular_quadrature.scatter_source_moments(mu_s_l_volume, phi, moment_major=True)` (the single,
    memory-lean implementation of the operator).
    mu_s_l_volume : [G_in, G_out, L+1, *spatial] Legendre transfer moments (1/cm).
    phi           : [G_in, N, *rest], N = (L_phi+1)^2 with L_phi <= L; *rest = *spatial or (4, *spatial).
    Returns T [G_out, N, *rest] as a (non-contiguous) view of a moment-major buffer [N, G_out, *rest], so
    T.movedim(1, 0) is contiguous (what `scatter_source_function` and
    `angular_quadrature.scatter_source_at_directions` contract over without copying). Zero transfer blocks (e.g.
    upscatter) are skipped.
    """
    return scatter_source_moments(mu_s_l_volume, phi, moment_major=True)


def scatter_source_function(sweep_context: SweepContext, T: Tensor) -> SourceFunction:
    """Source function for the sweep from the spherical-harmonic moments T of a scattering source.

    T: [G, (L+1)^2, 4, R, C, S] (from scatter_moments / scatter_source_moments). The returned callable maps
    reordered direction indices direction_index [Dsel] (Dsel selected directions) to the linear-discontinuous source
    moments [Dsel, G, 4, R, C, S] = sum_n Y[direction_index, n] T[:, n], i.e.
    angular_quadrature.scatter_source_at_directions(T, sweep_context.basis.Y[direction_index]). A moment-major copy of
    T is made once (none if T came from `scatter_moments`), so each call is one GEMM. The callable accepts `out=` (a
    contiguous [Dsel, G, 4, R, C, S] buffer, e.g. lent by `sweep`) to write the GEMM result into instead of allocating
    it.
    """
    if T.shape[1] != sweep_context.num_moments:
        raise ValueError(f"T has {T.shape[1]} spherical-harmonic moments, context has {sweep_context.num_moments}")
    Y = sweep_context.basis.Y.to(device=T.device, dtype=T.dtype)
    rest = tuple(T.shape[2:])
    G, num_moments = T.shape[0], T.shape[1]
    # no copy when T is a view of moment-major storage
    moments_matrix = T.movedim(1, 0).contiguous().view(num_moments, -1)

    def source(direction_index: Tensor, out: Optional[Tensor] = None) -> Tensor:
        Y_selected = Y.index_select(0, direction_index.to(Y.device))  # [Dsel, N_sh]
        if out is None:
            return (Y_selected @ moments_matrix).view(Y_selected.shape[0], G, *rest)
        torch.mm(Y_selected, moments_matrix, out=out.view(Y_selected.shape[0], -1))
        return out

    return source


def multiple_collision_source_moments(mu_s_l_volume_coarse: Tensor, phi_total: Tensor) -> Tensor:
    """Spherical-harmonic moments of the multiple-collision source: the scatter operator applied to phi_total.

    mu_s_l_volume_coarse : [G_in, G_out, L+1, R, C, S] Legendre transfer moments on the coarse grid (1/cm).
    phi_total            : [G, (L+1)^2, 4, R, C, S] total collided fluence moments (SourceIterationResult.phi_total),
                           with degree <= the transfer's Legendre order.
    Returns T [G, (L+1)^2, 4, R, C, S] (view of moment-major storage, see `scatter_moments`);
    S_MC(r, Omega) = sum_n T[:, n] Y_n(Omega): evaluate with angular_quadrature.scatter_source_at_directions at the
    collapsed-cone directions.
    """
    return scatter_moments(mu_s_l_volume_coarse, phi_total)


def _scalar_norm(phi: Tensor) -> float:
    return float(torch.linalg.vector_norm(phi[:, 0, 0]))


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
    """Source iteration over collision order (module docstring).

    sweep_context        : the `transport_sweep.SweepContext` of the coarse grid and the sweep quadrature.
    mu_t_coarse          : [G, R, C, S] total attenuation on the coarse (sweep) grid, 1/cm.
    mu_s_l_volume_coarse : [G_in, G_out, L_x+1, R, C, S] Legendre transfer moments (L_x >= sweep_context.L).
    first_collision_source_function : reordered direction indices [Dsel] -> first-collision linear-discontinuous
                           source moments [Dsel, G, 4, R, C, S] (e.g.
                           uncollided.restrict_to_linear_discontinuous_moments
                           evaluated at sweep_context.quadrature.omega[direction_index]).
    tolerance            : stop when ||phi_n[:, 0, 0]|| / ||phi_total[:, 0, 0]|| < tolerance.
    max_iterations       : maximum number of sweeps (collision orders).
    direction_chunk      : passed to `sweep`.
    callback             : optional callback(n, phi_n, relative_change) after each iteration (n starts at 1);
                           phi_n must not be modified.

    Returns SourceIterationResult (phi_total [G, (L+1)^2, 4, R, C, S], num_iterations, history, phi_norms,
    converged).
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
        norm_n = _scalar_norm(phi_n)
        norm_total = _scalar_norm(phi_total)
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
