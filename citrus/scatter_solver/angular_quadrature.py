"""Angular quadratures, spherical harmonics and the scattering-source operator.

Contents
--------
* Direction sets (`Quadrature`): the Gauss-Legendre x equal-azimuth product set of the sweep and the cone product
  sets of the collapsed-cone scatter image (`hemisphere_quadrature`).
* Legendre polynomials and real orthonormal spherical harmonics (`legendre_p`,
  `real_spherical_harmonics`, `sh_degree_index`).
* Moment <-> ordinate transforms (`SHBasis`, `moments_from_ordinates`, `ordinates_from_moments`).
* The scattering-source operator (`scatter_source_moments`, `scatter_source_at_directions`):

      S_g(Omega) = sum_{g'} sum_n mu_s,l(n)^{g' -> g} phi_n,g' Y_n(Omega)

Conventions (docs/source/conventions.rst, "Scattering convention", and docs/source/design_notes.rst)
----------------------------------------------------------------------------------------------------
Real orthonormal spherical harmonics, polar axis +z, polar angle theta from +z, azimuth phi from +x
toward +y, NO Condon-Shortley phase:

    Y_l0      = N_l0 P_l(cos theta)
    Y_l,+k    = sqrt(2) N_lk P_l^k(cos theta) cos(k phi)      (k = 1..l)
    Y_l,-k    = sqrt(2) N_lk P_l^k(cos theta) sin(k phi)      (k = 1..l)
    N_lk      = sqrt((2l+1)/(4 pi) (l-k)!/(l+k)!)

with P_l^k the associated Legendre function WITHOUT the (-1)^k Condon-Shortley factor
(so P_1^1 = +sin theta and Y_1,+1 = sqrt(3/(4 pi)) x, Y_1,-1 = sqrt(3/(4 pi)) y, Y_1,0 = sqrt(3/(4 pi)) z).
Flat index n = l*l + (k + l), n = 0..(L+1)^2 - 1. Orthonormal: int_{S^2} Y_n Y_n' dOmega = delta_nn'.

Relation to the complex harmonics of scipy (which include the Condon-Shortley phase):
Y_l,+k = sqrt(2) (-1)^k Re Y_l^k,  Y_l,-k = sqrt(2) (-1)^k Im Y_l^k  (k > 0),  Y_l0 = Y_l^0.

Scattering-source normalization (derivation; confirmed numerically in tests/test_angular.py)
-----------------------------------------------------------------------------------------
Convention: mu_s(Omega' -> Omega) = sum_l (2l+1)/(4 pi) mu_s_l P_l(Omega . Omega').
Addition theorem for orthonormal real SH: P_l(Omega . Omega') = 4 pi/(2l+1) sum_k Y_lk(Omega) Y_lk(Omega').
Hence

    S(Omega) = int mu_s(Omega' -> Omega) psi(Omega') dOmega'
             = sum_l (2l+1)/(4 pi) mu_s_l * 4 pi/(2l+1) sum_k Y_lk(Omega) int Y_lk(Omega') psi(Omega') dOmega'
             = sum_n mu_s_l(n) phi_n Y_n(Omega),          phi_n = int Y_n psi dOmega ~ sum_m w_m Y_n(Omega_m) psi_m

i.e. the (2l+1)/(4 pi) factor of the Legendre expansion cancels exactly against the addition theorem,
and NO extra factor appears in the moment-space operator. Summing over source groups g' gives the operator above.

Octant code
-----------
octant = (omega_x < 0) * 4 + (omega_y < 0) * 2 + (omega_z < 0), an int64 in 0..7. A zero component
counts as non-negative. Octant 0 is (+,+,+), octant 7 is (-,-,-).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "Quadrature",
    "octant_of",
    "product_quadrature",
    "hemisphere_quadrature",
    "legendre_p",
    "num_sh",
    "sh_degree_index",
    "real_spherical_harmonics",
    "SHBasis",
    "sh_basis",
    "moments_from_ordinates",
    "ordinates_from_moments",
    "scatter_source_moments",
    "scatter_source_at_directions",
]

FOUR_PI = 4.0 * math.pi
_DeviceLike = Union[torch.device, str, None]


# =====================================================================================================
# Direction sets
# =====================================================================================================

def octant_of(omega: Tensor) -> Tensor:
    """Octant code of directions omega [..., 3] -> int64 [...].

    octant = (omega_x < 0) * 4 + (omega_y < 0) * 2 + (omega_z < 0); zero components count as >= 0.
    """
    neg = (omega < 0).to(torch.int64)
    return neg[..., 0] * 4 + neg[..., 1] * 2 + neg[..., 2]


@dataclass(frozen=True, eq=False)
class Quadrature:
    """A discrete direction set.

    Attributes
    ----------
    omega  : [M, 3] unit directions in the recon frame.
    weight : [M] quadrature weights in steradians (sum to 4 pi for a full-sphere set, to the cone
             solid angle for a cone set).
    octant : [M] int64 in 0..7, (omega_x<0)*4 + (omega_y<0)*2 + (omega_z<0) (see `octant_of`).
    cell_width : optional [M, 2] (collapsed-cone sets built with tangent axes, e.g. by
             scatter_image.panel_direction_set):
             effective widths of the solid-angle cell each direction stands for, in the gnomonic coordinates
             (alpha, beta) = (d.u / d.a, d.v / d.a) about the cone axis a along the two tangent axes u, v; width =
             sqrt(12) x the solid-angle-weighted standard deviation of alpha (beta) over the cell (= the exact side
             length for a gnomonic rectangle cell). Used by the finite-cone footprint of the scatter image.
    """

    omega: Tensor
    weight: Tensor
    octant: Tensor
    cell_width: Optional[Tensor] = None

    @staticmethod
    def from_directions(omega: Tensor, weight: Tensor) -> "Quadrature":
        """Build from omega [M, 3] and weight [M]; computes the octant codes."""
        if omega.dim() != 2 or omega.shape[1] != 3:
            raise ValueError(f"omega must be [M, 3], got {tuple(omega.shape)}")
        if weight.shape != omega.shape[:1]:
            raise ValueError(f"weight must be [M], got {tuple(weight.shape)}")
        return Quadrature(omega=omega, weight=weight, octant=octant_of(omega))

    @property
    def num_directions(self) -> int:
        return int(self.omega.shape[0])

    @property
    def device(self) -> torch.device:
        return self.omega.device

    @property
    def dtype(self) -> torch.dtype:
        return self.omega.dtype

    def octant_counts(self) -> Tensor:
        """Number of directions per octant, int64 [8] on the quadrature's device."""
        return torch.bincount(self.octant, minlength=8)

    def by_octant(self) -> tuple[Tensor, Tensor]:
        """Permutation grouping the directions by octant.

        Returns (perm [M] int64, counts [8] int64): `omega[perm]` lists octant 0 first, then 1, ...,
        and within an octant the original order is kept (stable). `counts[o]` is the number of
        directions in octant o, so octant o occupies perm[sum(counts[:o]) : sum(counts[:o+1])].
        """
        perm = torch.sort(self.octant, stable=True).indices
        return perm, self.octant_counts()

    @property
    def directions_per_octant(self) -> Optional[int]:
        """Common per-octant count if every octant holds the same number of directions, else None."""
        counts = self.octant_counts().tolist()
        return int(counts[0]) if all(c == counts[0] for c in counts) else None

    def reordered(self, perm: Tensor) -> "Quadrature":
        """The same set with directions permuted by perm [M]."""
        cw = None if self.cell_width is None else self.cell_width[perm.to(self.cell_width.device)]
        return Quadrature(omega=self.omega[perm], weight=self.weight[perm], octant=self.octant[perm], cell_width=cw)

    def to(self, device: _DeviceLike = None, dtype: Optional[torch.dtype] = None) -> "Quadrature":
        kw = {}
        if device is not None:
            kw["device"] = device
        omega = self.omega.to(dtype=dtype, **kw) if dtype is not None else self.omega.to(**kw)
        weight = self.weight.to(dtype=dtype, **kw) if dtype is not None else self.weight.to(**kw)
        cw = self.cell_width
        if cw is not None:
            cw = cw.to(dtype=dtype, **kw) if dtype is not None else cw.to(**kw)
        return Quadrature(omega=omega, weight=weight, octant=self.octant.to(**kw), cell_width=cw)


def _gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes (ascending) and weights on [-1, 1], float64 numpy."""
    if n < 1:
        raise ValueError("number of Gauss-Legendre points must be >= 1")
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def product_quadrature(
    num_polar: int,
    num_azimuthal: int,
    device: _DeviceLike = None,
    dtype: torch.dtype = torch.float64,
) -> Quadrature:
    """Gauss-Legendre (in mu = cos theta, polar from +z) x equal-azimuth product set on the sphere.

    mu_i: the num_polar Gauss-Legendre nodes on [-1, 1]; phi_j = (j + 1/2) 2 pi / num_azimuthal;
    weight_ij = w_GL_i * 2 pi / num_azimuthal (sum = 4 pi). Flat index m = i * num_azimuthal + j.
    Exact for spherical harmonics products Y_n Y_n' up to degree L when num_polar >= L+1 and
    num_azimuthal >= 2L+1. With even num_polar and num_azimuthal divisible by 4, no direction lies
    on a coordinate plane and every octant holds num_polar*num_azimuthal/8 directions (asserted).

    Returns a Quadrature with omega [M, 3], weight [M], M = num_polar * num_azimuthal.
    """
    if num_azimuthal < 1:
        raise ValueError("num_azimuthal must be >= 1")
    mu_np, wgl_np = _gauss_legendre(num_polar)
    mu = torch.as_tensor(mu_np, dtype=torch.float64)
    wgl = torch.as_tensor(wgl_np, dtype=torch.float64)
    dphi = 2.0 * math.pi / num_azimuthal
    phi = (torch.arange(num_azimuthal, dtype=torch.float64) + 0.5) * dphi
    sin_t = torch.sqrt((1.0 - mu * mu).clamp_min(0.0))
    ox = sin_t[:, None] * torch.cos(phi)[None, :]
    oy = sin_t[:, None] * torch.sin(phi)[None, :]
    oz = mu[:, None].expand(num_polar, num_azimuthal)
    omega = torch.stack([ox, oy, oz], dim=-1).reshape(-1, 3)
    weight = (wgl[:, None] * dphi).expand(num_polar, num_azimuthal).reshape(-1)
    quad = Quadrature.from_directions(omega.to(device=device, dtype=dtype), weight.to(device=device, dtype=dtype))
    if num_polar % 2 == 0 and num_azimuthal % 4 == 0:
        expected = num_polar * num_azimuthal // 8
        assert quad.directions_per_octant == expected, "product set octants are not balanced"
    return quad


# ---- cone sets of the collapsed-cone scatter image ------------------------------------------------

def _as_vec3(v: Union[Tensor, Sequence[float]]) -> Tensor:
    t = torch.as_tensor(v, dtype=torch.float64).detach().to("cpu").reshape(-1)
    if t.numel() != 3:
        raise ValueError("axis must have 3 components")
    nrm = torch.linalg.norm(t)
    if not bool(nrm > 0):
        raise ValueError("axis must be non-zero")
    return t / nrm


def _local_frame(axis: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Deterministic right-handed orthonormal frame (e1, e2, a) about the unit vector axis [3] (float64).

    The helper vector is the coordinate axis least aligned with `axis` (lowest index on ties);
    e1 = normalize(helper - (helper . a) a), e2 = a x e1.
    """
    a = axis
    idx = int(torch.argmin(a.abs()))
    helper = torch.zeros(3, dtype=torch.float64)
    helper[idx] = 1.0
    e1 = helper - (helper @ a) * a
    e1 = e1 / torch.linalg.norm(e1)
    e2 = torch.linalg.cross(a, e1)
    return e1, e2, a


def _cone_product(
    axis: Tensor, num_polar: int, num_azimuthal: int, max_polar_angle: float
) -> tuple[Tensor, Tensor]:
    """float64 CPU cone product set. Returns (omega [M,3], weight [M])."""
    if not (0.0 < max_polar_angle <= math.pi):
        raise ValueError("max_polar_angle must be in (0, pi]")
    if num_azimuthal < 1:
        raise ValueError("num_azimuthal must be >= 1")
    c0 = math.cos(max_polar_angle)
    t, wgl = _gauss_legendre(num_polar)
    half_len = 0.5 * (1.0 - c0)
    c = torch.as_tensor(c0 + half_len * (t + 1.0), dtype=torch.float64).flip(0)  # descending: near-axis first
    wc = torch.as_tensor(wgl * half_len, dtype=torch.float64).flip(0)
    dphi = 2.0 * math.pi / num_azimuthal
    phi = (torch.arange(num_azimuthal, dtype=torch.float64) + 0.5) * dphi
    e1, e2, a = _local_frame(axis)
    s = torch.sqrt((1.0 - c * c).clamp_min(0.0))
    local = (
        (s[:, None] * torch.cos(phi)[None, :])[..., None] * e1
        + (s[:, None] * torch.sin(phi)[None, :])[..., None] * e2
        + c[:, None, None] * a
    )  # [P, A, 3]
    omega = local.reshape(-1, 3)
    omega = omega / torch.linalg.norm(omega, dim=-1, keepdim=True)
    weight = (wc[:, None] * dphi).expand(num_polar, num_azimuthal).reshape(-1)
    return omega, weight


def hemisphere_quadrature(
    axis: Union[Tensor, Sequence[float]],
    num_polar: int,
    num_azimuthal: int,
    max_polar_angle: float = math.pi / 2,
    device: _DeviceLike = None,
    dtype: torch.dtype = torch.float64,
    *,
    tangent_axes: Optional[tuple] = None,
    subsamples: int = 16,
) -> Quadrature:
    """Product set on the cone of half-angle max_polar_angle about `axis` (collapsed-cone direction set).

    Polar angle theta is measured from `axis`: Gauss-Legendre in cos(theta) on [cos(max_polar_angle), 1]
    times num_azimuthal equally spaced azimuths phi_j = (j + 1/2) 2 pi / num_azimuthal about `axis` in the
    deterministic local frame of `_local_frame`. Weights sum to the cone solid angle
    2 pi (1 - cos(max_polar_angle)) (2 pi for the default hemisphere). Flat index m = i * num_azimuthal + j
    with i = 0 the polar node closest to `axis`.

    For the panel of `ConeGeometry`, the directions reaching the panel are about axis = -n_hat
    (n_hat from `panel_axes` points toward the source).

    axis: [3] (tensor or sequence, need not be normalized). Returns omega [M, 3], weight [M].
    tangent_axes: optional (u_axis, v_axis); when given, `cell_width` [M, 2] is filled (the cell of node (i, j) is
      [c_{i+1/2}, c_{i-1/2}] x [phi_j -+ dphi/2] in (cos theta, phi), cos edges at the cumulative Gauss weights, its
      widths measured on a subsamples^2 equal-area lattice; `_cone_cell_widths`).
    """
    if device is None and isinstance(axis, Tensor):
        device = axis.device
    a = _as_vec3(axis)
    omega, weight = _cone_product(a, num_polar, num_azimuthal, max_polar_angle)
    q = Quadrature.from_directions(omega.to(device=device, dtype=dtype), weight.to(device=device, dtype=dtype))
    if tangent_axes is None:
        return q
    u, v = _tangent_frame(a, *tangent_axes)
    width = _cone_cell_widths(a, u, v, num_polar, num_azimuthal, max_polar_angle, subsamples)
    return replace(q, cell_width=width.to(device=device, dtype=dtype))


def _tangent_frame(a: Tensor, u_axis, v_axis) -> tuple[Tensor, Tensor]:
    """u, v: u_axis and v_axis Gram-Schmidt orthonormalized against the unit axis a (float64 CPU)."""
    u = _as_vec3(u_axis)
    u = u - (u @ a) * a
    u = u / torch.linalg.norm(u)
    v = _as_vec3(v_axis)
    v = v - (v @ a) * a - (v @ u) * u
    return u, v / torch.linalg.norm(v)


def _cone_cell_widths(a: Tensor, u: Tensor, v: Tensor, num_polar: int, num_azimuthal: int, max_polar_angle: float,
                      subsamples: int) -> Tensor:
    """Cell widths of the cone product set (`_cone_product` order): node (i, j) owns [c_{i+1/2}, c_{i-1/2}] x
    [j dphi, (j+1) dphi] in (cos theta, phi) about a (frame `_local_frame`), cos edges at the cumulative Gauss weights,
    so the cell's solid angle is its weight. Sampled on a subsamples x subsamples midpoint lattice in (cos theta, phi)
    (equal area). Returns float64 width [M, 2] = sqrt(12) x std of the gnomonic coordinates (alpha, beta) =
    (d.u / d.a, d.v / d.a) over the cell."""
    f64 = torch.float64
    s = int(subsamples)
    q = (torch.arange(s, dtype=f64) + 0.5) / s
    c0 = math.cos(max_polar_angle)
    _, wgl = _gauss_legendre(num_polar)
    wc = torch.as_tensor(wgl * 0.5 * (1.0 - c0), dtype=f64).flip(0)  # per polar node, near-axis first
    c_edge = torch.cat([torch.ones(1, dtype=f64), 1.0 - torch.cumsum(wc, 0)])  # [P + 1], descending
    c_edge[-1] = c0
    dphi = 2.0 * math.pi / num_azimuthal
    e1, e2, _ = _local_frame(a)
    cs = c_edge[1:, None] + (c_edge[:-1] - c_edge[1:])[:, None] * q[None, :]  # [P, s]
    ps = (torch.arange(num_azimuthal, dtype=f64)[:, None] + q[None, :]) * dphi  # [A, s]
    sn = torch.sqrt((1.0 - cs * cs).clamp_min(0.0))
    lat = torch.cos(ps)[..., None] * e1 + torch.sin(ps)[..., None] * e2  # [A, s, 3]
    # d[i, j, p, r] = sn[i, p] lat[j, r] + cs[i, p] a     [P, A, s, s, 3]
    d = sn[:, None, :, None, None] * lat[None, :, None, :, :] + cs[:, None, :, None, None] * a
    dn = (d @ a).clamp_min(1e-6)
    al, bt = (d @ u) / dn, (d @ v) / dn
    n = float(s * s)

    def width(x: Tensor) -> Tensor:
        mean = x.sum(dim=(2, 3)) / n
        var = ((x * x).sum(dim=(2, 3)) / n - mean * mean).clamp_min(0.0)
        return torch.sqrt(12.0 * var).reshape(-1)

    return torch.stack([width(al), width(bt)], dim=-1)


# =====================================================================================================
# Legendre polynomials and real spherical harmonics
# =====================================================================================================

def legendre_p(L: int, x: Tensor) -> Tensor:
    """Legendre polynomials P_0..P_L at x [...] -> [..., L+1] (same device/dtype as x).

    Three-term recurrence (l+1) P_{l+1} = (2l+1) x P_l - l P_{l-1}.
    """
    if L < 0:
        raise ValueError("L must be >= 0")
    out = [torch.ones_like(x)]
    if L >= 1:
        out.append(x.clone())
    for l in range(1, L):
        out.append(((2 * l + 1) * x * out[l] - l * out[l - 1]) / (l + 1))
    return torch.stack(out, dim=-1)


def num_sh(L: int) -> int:
    """Number of real spherical harmonics up to degree L: (L+1)^2."""
    return (L + 1) * (L + 1)


def sh_degree_index(L: int, device: _DeviceLike = None) -> Tensor:
    """Degree l of each flat SH index n = l*l + (k+l): int64 [(L+1)^2]."""
    return torch.cat([torch.full((2 * l + 1,), l, dtype=torch.int64) for l in range(L + 1)]).to(device)


def real_spherical_harmonics(L: int, omega: Tensor) -> Tensor:
    """Real orthonormal spherical harmonics Y [M, (L+1)^2] at unit directions omega [M, 3] (any leading dims).

    Convention in the module docstring (polar axis +z, no Condon-Shortley phase, flat index
    n = l*l + (k + l)). Evaluated without trigonometric functions or division by sin(theta):

        Y_l,+k = sqrt(2) Q_lk(z) Re((x + i y)^k),   Y_l,-k = sqrt(2) Q_lk(z) Im((x + i y)^k),   Y_l0 = Q_l0(z)

    where Q_lk = N_lk P_l^k / sin^k(theta) is a polynomial in z computed with the stable normalized
    recurrences
        Q_00 = 1/sqrt(4 pi),  Q_kk = sqrt((2k+1)/(2k)) Q_{k-1,k-1},  Q_{k+1,k} = sqrt(2k+3) z Q_kk,
        Q_lk = a_lk (z Q_{l-1,k} - b_lk Q_{l-2,k}),
        a_lk = sqrt((4l^2 - 1)/(l^2 - k^2)),  b_lk = sqrt(((l-1)^2 - k^2)/(4(l-1)^2 - 1)).
    omega is assumed unit-norm (not renormalized). Same device/dtype as omega.
    """
    if L < 0:
        raise ValueError("L must be >= 0")
    x, y, z = omega[..., 0], omega[..., 1], omega[..., 2]
    one = torch.ones_like(z)
    # Q[k][l] for l >= k
    Q: list[dict[int, Tensor]] = [dict() for _ in range(L + 1)]
    qkk = one * (1.0 / math.sqrt(4.0 * math.pi))
    for k in range(L + 1):
        if k > 0:
            qkk = qkk * math.sqrt((2 * k + 1) / (2.0 * k))
        Q[k][k] = qkk
        if k + 1 <= L:
            Q[k][k + 1] = math.sqrt(2 * k + 3) * z * qkk
        for l in range(k + 2, L + 1):
            a = math.sqrt((4.0 * l * l - 1.0) / (l * l - k * k))
            b = math.sqrt(((l - 1) ** 2 - k * k) / (4.0 * (l - 1) ** 2 - 1.0))
            Q[k][l] = a * (z * Q[k][l - 1] - b * Q[k][l - 2])
    # C_k = Re((x+iy)^k), S_k = Im((x+iy)^k)
    C = [one]
    S = [torch.zeros_like(z)]
    for k in range(1, L + 1):
        C.append(x * C[k - 1] - y * S[k - 1])
        S.append(x * S[k - 1] + y * C[k - 1])
    sqrt2 = math.sqrt(2.0)
    cols: list[Tensor] = []
    for l in range(L + 1):
        for k in range(-l, l + 1):
            if k == 0:
                cols.append(Q[0][l])
            elif k > 0:
                cols.append(sqrt2 * Q[k][l] * C[k])
            else:
                cols.append(sqrt2 * Q[-k][l] * S[-k])
    return torch.stack(cols, dim=-1)


# =====================================================================================================
# Moment / ordinate transforms
# =====================================================================================================

@dataclass(frozen=True, eq=False)
class SHBasis:
    """Real SH tables on a quadrature.

    L      : max degree; N = (L+1)^2.
    Y      : [M, N]  Y_n(Omega_m).
    Yw     : [M, N]  w_m Y_n(Omega_m).
    l_of_n : [N] int64 degree of each flat index.
    """

    L: int
    Y: Tensor
    Yw: Tensor
    l_of_n: Tensor

    @property
    def num_moments(self) -> int:
        return int(self.Y.shape[1])

    @property
    def num_directions(self) -> int:
        return int(self.Y.shape[0])


def sh_basis(L: int, quad: Quadrature) -> SHBasis:
    """Build the SH tables for degree L on `quad` (device/dtype of quad)."""
    Y = real_spherical_harmonics(L, quad.omega)
    return SHBasis(L=L, Y=Y, Yw=quad.weight[:, None] * Y, l_of_n=sh_degree_index(L, quad.omega.device))


def moments_from_ordinates(psi: Tensor, basis: SHBasis) -> Tensor:
    """phi_n = sum_m w_m Y_n(Omega_m) psi_m.  psi [..., M] -> phi [..., N]."""
    return psi @ basis.Yw


def ordinates_from_moments(phi: Tensor, basis: SHBasis) -> Tensor:
    """psi(Omega_m) = sum_n phi_n Y_n(Omega_m).  phi [..., N] -> psi [..., M]."""
    return phi @ basis.Y.T


# =====================================================================================================
# Scattering source
# =====================================================================================================

def scatter_source_moments(mu_s_l_vol: Tensor, phi: Tensor, *, moment_major: bool = False) -> Tensor:
    """SH moments of the scattering source: T[g, n] = sum_{g_in} mu_s_l_vol[g_in, g, l(n)] * phi[g_in, n].

    mu_s_l_vol : [G_in, G_out, L+1, *spatial]   Legendre transfer moments per voxel (1/cm), e.g. from
                 `cross_sections.transfer_moments_volume`; convention mu_s(Omega'->Omega) = sum_l (2l+1)/(4pi) mu_s_l P_l.
    phi        : [G_in, N, *rest]   SH flux moments, N = (L_phi+1)^2 with L_phi <= L; *rest is either
                 *spatial or (4, *spatial) (LD moments; the transfer broadcasts over that axis). May be a
                 non-contiguous view (e.g. phi_ld[:, :, 0]).
    moment_major : False (default): return a contiguous [G_out, N, *rest] tensor.
                 True: allocate the result moment-major, [N, G_out, *rest], and return the [G_out, N, *rest]
                 view `buf.movedim(0, 1)` (non-contiguous; `T.movedim(1, 0)` is contiguous). This is the layout
                 `source_iteration.scatter_source_fn` contracts over without a copy.
    Returns T  : [G_out, N, *rest], so that S(Omega) = sum_n T[:, n] Y_n(Omega) (no extra factor; see the
                 module docstring for the derivation).

    Implementation (memory-lean): the output buffer is allocated once and every non-zero transfer block
    (g_in, g_out) is accumulated into it in place, `out[g_out] += mu_s_l_vol[g_in, g_out, l(n)] * phi[g_in]`,
    so the only temporary is one [N, *spatial] gather of the transfer. Zero blocks (upscatter) are skipped
    (one host sync to find them). Peak extra memory ~ N |spatial| values instead of the
    [G_in, G_out, N, *spatial] gather plus an output-sized term of a broadcast implementation (which peaks
    at ~6.5 GB on a 40x60x80 grid with G = 6, L = 7).
    """
    G_in, G_out, Lp1 = mu_s_l_vol.shape[:3]
    spatial = tuple(mu_s_l_vol.shape[3:])
    if phi.shape[0] != G_in:
        raise ValueError(f"phi has {phi.shape[0]} groups, transfer has {G_in}")
    N = phi.shape[1]
    L_phi = math.isqrt(N) - 1
    if (L_phi + 1) ** 2 != N:
        raise ValueError(f"phi.shape[1]={N} is not a square (L+1)^2")
    if L_phi + 1 > Lp1:
        raise ValueError(f"phi has degree {L_phi} > transfer Legendre order {Lp1 - 1}")
    rest = tuple(phi.shape[2:])
    if rest == spatial:
        ld = False
    elif len(rest) == len(spatial) + 1 and rest[1:] == spatial:
        ld = True
    else:
        raise ValueError(f"phi trailing shape {rest} must be spatial {spatial} or (4, *spatial)")
    l_of_n = sh_degree_index(L_phi, mu_s_l_vol.device)
    nonzero = (mu_s_l_vol.reshape(G_in, G_out, -1) != 0).any(dim=-1).cpu()  # one host sync
    if moment_major:
        buf = torch.zeros((N, G_out) + rest, device=phi.device, dtype=phi.dtype)
        out = buf.movedim(0, 1)  # [G_out, N, *rest] view
    else:
        out = torch.zeros((G_out, N) + rest, device=phi.device, dtype=phi.dtype)
    for gi in range(G_in):
        for go in range(G_out):
            if not bool(nonzero[gi, go]):
                continue
            mn = mu_s_l_vol[gi, go].index_select(0, l_of_n)  # [N, *spatial]
            if ld:
                mn = mn.unsqueeze(1)  # [N, 1, *spatial]
            out[go].addcmul_(mn, phi[gi])
    return out


def scatter_source_at_directions(T: Tensor, Y_sub: Tensor) -> Tensor:
    """Scattering source at K directions: S[k, g, ...] = sum_n Y_sub[k, n] T[g, n, ...].

    T: [G, N, *rest] (from `scatter_source_moments`); Y_sub: [K, N] (rows of SHBasis.Y). Returns [K, G, *rest].
    """
    if T.shape[1] != Y_sub.shape[1]:
        raise ValueError(f"T has {T.shape[1]} moments, Y_sub has {Y_sub.shape[1]}")
    return torch.tensordot(Y_sub, T, dims=([1], [1]))
