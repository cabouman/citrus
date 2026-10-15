"""Source iteration over collision order and the multiple-collision source.

    phi^(1)   = sweep(first-collision source)                                  (n = 1)
    phi^(n)   = sweep(S^(n)),  S^(n)(Omega_m) = sum_n' T^(n)[:, n'] Y_n'(Omega_m),
                T^(n) = scatter_source_moments(mu_s_l, phi^(n-1))              (n >= 2)
    phi_total = sum_n phi^(n)

phi^(n) is the n-times-collided fluence; the multiple-collision source of the scatter image is the scattering
source of phi_total, S_MC(r, Omega) = sum_n T[:, n](r) Y_n(Omega) with T = scatter_source_moments(mu_s_l, phi_total).

stopped when ||phi^(n)[:, 0, 0]|| / ||phi_total[:, 0, 0]|| < tol (Frobenius norms of the scalar-flux
cell averages over all groups and cells) or after max_iterations sweeps. Each phi^(n) is the SH-moment /
LD-moment field [G, (L+1)^2, 4, R, C, S] of the n-times-collided fluence.

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
from .transport_sweep import SourceFn, SweepContext, sweep

__all__ = [
    "SIResult",
    "source_iteration",
    "scatter_moments",
    "scatter_source_fn",
    "multiple_collision_source_moments",
]


@dataclass
class SIResult:
    """Result of `source_iteration`.

    phi_total      : [G, (L+1)^2, 4, R, C, S] sum over collision orders of the SH/LD moments of the
                     collided fluence (LD layout [0, x, y, z]).
    num_iterations : number of sweeps performed (= collision orders included).
    history        : relative change ||phi_n[:, 0, 0]|| / ||phi_total[:, 0, 0]|| after each iteration
                     (history[0] == 1 unless the first-collision fluence vanishes).
    phi_norms      : ||phi_n[:, 0, 0]|| per iteration.
    converged      : True if the last relative change was < tol.
    """

    phi_total: Tensor
    num_iterations: int
    history: list[float] = field(default_factory=list)
    phi_norms: list[float] = field(default_factory=list)
    converged: bool = False


def scatter_moments(mu_s_l_vol: Tensor, phi: Tensor) -> Tensor:
    """Scattering-source SH moments T[g, n] = sum_{g_in} mu_s_l_vol[g_in, g, l(n)] phi[g_in, n].

    Thin wrapper: `angular_quadrature.scatter_source_moments(mu_s_l_vol, phi, moment_major=True)` (the single,
    memory-lean implementation of the operator).
    mu_s_l_vol : [G_in, G_out, L+1, *spatial] Legendre transfer moments (1/cm).
    phi        : [G_in, N, *rest], N = (L_phi+1)^2 with L_phi <= L; *rest = *spatial or (4, *spatial).
    Returns T [G_out, N, *rest] as a (non-contiguous) view of a moment-major buffer [N, G_out, *rest], so
    T.movedim(1, 0) is contiguous (what `scatter_source_fn` and `angular_quadrature.scatter_source_at_directions`
    contract over without copying). Zero transfer blocks (e.g. upscatter) are skipped.
    """
    return scatter_source_moments(mu_s_l_vol, phi, moment_major=True)


def scatter_source_fn(ctx: SweepContext, T: Tensor) -> SourceFn:
    """Source function for the sweep from the SH moments T of a scattering source.

    T: [G, (L+1)^2, 4, R, C, S] (from scatter_moments / scatter_source_moments). The returned callable maps
    reordered direction indices dir_index [Dsel] to the LD source moments [Dsel, G, 4, R, C, S] =
    sum_n Y[dir_index, n] T[:, n], i.e. angular_quadrature.scatter_source_at_directions(T, ctx.basis.Y[dir_index]).
    A moment-major copy of T is made once (none if T came from `scatter_moments`), so each call is one GEMM.
    The callable accepts `out=` (a contiguous [Dsel, G, 4, R, C, S] buffer, e.g. lent by `sweep`) to write the
    GEMM result into instead of allocating it.
    """
    if T.shape[1] != ctx.num_moments:
        raise ValueError(f"T has {T.shape[1]} SH moments, context has {ctx.num_moments}")
    Y = ctx.basis.Y.to(device=T.device, dtype=T.dtype)
    rest = tuple(T.shape[2:])
    G, Nsh = T.shape[0], T.shape[1]
    T_mm = T.movedim(1, 0).contiguous().view(Nsh, -1)  # no copy when T is a view of moment-major storage

    def fn(dir_index: Tensor, out: Optional[Tensor] = None) -> Tensor:
        Ysub = Y.index_select(0, dir_index.to(Y.device))  # [Dsel, N_sh]
        if out is None:
            return (Ysub @ T_mm).view(Ysub.shape[0], G, *rest)
        torch.mm(Ysub, T_mm, out=out.view(Ysub.shape[0], -1))
        return out

    return fn


def multiple_collision_source_moments(mu_s_l_vol_coarse: Tensor, phi_total: Tensor) -> Tensor:
    """SH moments of the multiple-collision source: the scatter operator applied to phi_total.

    mu_s_l_vol_coarse : [G_in, G_out, L+1, R, C, S] Legendre transfer moments on the coarse grid (1/cm).
    phi_total         : [G, (L+1)^2, 4, R, C, S] total collided fluence moments (SIResult.phi_total), with
                        degree <= the transfer's Legendre order.
    Returns T [G, (L+1)^2, 4, R, C, S] (view of moment-major storage, see `scatter_moments`);
    S_MC(r, Omega) = sum_n T[:, n] Y_n(Omega): evaluate with angular_quadrature.scatter_source_at_directions at the
    collapsed-cone directions.
    """
    return scatter_moments(mu_s_l_vol_coarse, phi_total)


def _scalar_norm(phi: Tensor) -> float:
    return float(torch.linalg.vector_norm(phi[:, 0, 0]))


def source_iteration(
    ctx: SweepContext,
    mu_t_coarse: Tensor,
    mu_s_l_vol_coarse: Tensor,
    fc_source_fn: SourceFn,
    tol: float,
    max_iterations: int,
    direction_chunk: Optional[int] = None,
    callback: Optional[Callable[[int, Tensor, float], None]] = None,
) -> SIResult:
    """Source iteration over collision order (module docstring).

    mu_t_coarse       : [G, R, C, S] total attenuation on the coarse (sweep) grid, 1/cm.
    mu_s_l_vol_coarse : [G_in, G_out, L_x+1, R, C, S] Legendre transfer moments (L_x >= ctx.L).
    fc_source_fn      : reordered direction indices [Dsel] -> first-collision LD source moments
                        [Dsel, G, 4, R, C, S] (e.g. uncollided.restrict_fc_to_ld_moments evaluated at
                        ctx.quad.omega[dir_index]).
    tol               : stop when ||phi_n[:, 0, 0]|| / ||phi_total[:, 0, 0]|| < tol.
    max_iterations    : maximum number of sweeps (collision orders).
    direction_chunk   : passed to `sweep`.
    callback          : optional callback(n, phi_n, rel_change) after each iteration (n starts at 1);
                        phi_n must not be modified.

    Returns SIResult (phi_total [G, (L+1)^2, 4, R, C, S], num_iterations, history, phi_norms, converged).
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    phi_total: Optional[Tensor] = None
    history: list[float] = []
    norms: list[float] = []
    converged = False
    source_fn: Optional[SourceFn] = fc_source_fn
    n = 0
    for n in range(1, max_iterations + 1):
        phi_n = sweep(ctx, mu_t_coarse, source_fn, direction_chunk=direction_chunk)
        source_fn = None  # release the previous scattering-source moments before forming the next ones
        if phi_total is None:
            phi_total = phi_n.clone()
        else:
            phi_total.add_(phi_n)
        nrm_n = _scalar_norm(phi_n)
        nrm_t = _scalar_norm(phi_total)
        rel = nrm_n / nrm_t if nrm_t > 0 else 0.0
        history.append(rel)
        norms.append(nrm_n)
        if callback is not None:
            callback(n, phi_n, rel)
        if rel < tol:
            converged = True
            break
        if n == max_iterations:
            break
        T = scatter_moments(mu_s_l_vol_coarse, phi_n)
        del phi_n
        source_fn = scatter_source_fn(ctx, T)
        del T
    assert phi_total is not None
    return SIResult(phi_total=phi_total, num_iterations=n, history=history, phi_norms=norms, converged=converged)
