"""Angular quadratures, real spherical harmonics and the scattering-source operator.

The module holds the direction sets of the transport sweep (Gauss-Legendre times equal-azimuth product sets) and of the
collapsed-cone scatter image (cone product sets), the Legendre polynomials and real orthonormal spherical harmonics,
and the transforms between angular ordinates and spherical-harmonic moments. The scattering-source operator computes
``S_g(Omega) = sum_g' sum_n mu_s_l(n) phi_n,g' Y_n(Omega)``. The conventions are listed in docs/source/conventions.rst,
"Scattering convention".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from .kernels import scattering_kernels

__all__ = [
    "Quadrature",
    "octant_of",
    "product_quadrature",
    "hemisphere_quadrature",
    "legendre_polynomials",
    "num_spherical_harmonics",
    "spherical_harmonic_degree_index",
    "real_spherical_harmonics",
    "SphericalHarmonicBasis",
    "spherical_harmonic_basis",
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
    """Octant code of each direction, ``(omega_x < 0) * 4 + (omega_y < 0) * 2 + (omega_z < 0)``.

    A zero component counts as non-negative.

    Args:
        omega (Tensor): [..., 3] directions.

    Returns:
        Tensor: [...] int64 octant codes in 0..7.
    """
    neg = (omega < 0).to(torch.int64)
    return neg[..., 0] * 4 + neg[..., 1] * 2 + neg[..., 2]


@dataclass(frozen=True, eq=False)
class Quadrature:
    """A discrete set of directions with quadrature weights.

    Attributes:
        omega (Tensor): [M, 3] unit directions in the reconstruction frame.
        weight (Tensor): [M] quadrature weights in sr; they sum to 4 pi for a full-sphere set and to the cone solid
            angle for a cone set.
        octant (Tensor): [M] int64 octant codes in 0..7, see
            :func:`~citrus.scatter_solver.angular_quadrature.octant_of`.
        cell_width (Tensor, optional): [M, 2] gnomonic widths of the solid-angle cell of each direction along the two
            tangent axes of a cone set, used by the finite-cone footprint of the scatter image. Defaults to None.
    """

    omega: Tensor
    weight: Tensor
    octant: Tensor
    cell_width: Optional[Tensor] = None

    @staticmethod
    def from_directions(omega: Tensor, weight: Tensor) -> "Quadrature":
        """Build a quadrature from directions and weights and compute the octant codes.

        Args:
            omega (Tensor): [M, 3] unit directions.
            weight (Tensor): [M] quadrature weights in sr.

        Returns:
            Quadrature: the direction set, without ``cell_width``.
        """
        if omega.dim() != 2 or omega.shape[1] != 3:
            raise ValueError(f"omega must be [M, 3], got {tuple(omega.shape)}")
        if weight.shape != omega.shape[:1]:
            raise ValueError(f"weight must be [M], got {tuple(weight.shape)}")
        return Quadrature(omega=omega, weight=weight, octant=octant_of(omega))

    @property
    def num_directions(self) -> int:
        """Number of directions M."""
        return int(self.omega.shape[0])

    @property
    def device(self) -> torch.device:
        """Device of the direction tensors."""
        return self.omega.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating dtype of the direction tensors."""
        return self.omega.dtype

    def octant_counts(self) -> Tensor:
        """Count the directions in each octant.

        Returns:
            Tensor: [8] int64 number of directions per octant, on the device of the quadrature.
        """
        return torch.bincount(self.octant, minlength=8)

    def by_octant(self) -> tuple[Tensor, Tensor]:
        """Permutation that groups the directions by octant, keeping the original order within an octant.

        Returns:
            (perm, counts): perm is the [M] int64 permutation with ``omega[perm]`` listing octant 0 first, counts is
            the [8] int64 number of directions per octant; octant o is ``perm[sum(counts[:o]):sum(counts[:o+1])]``.
        """
        perm = torch.sort(self.octant, stable=True).indices
        return perm, self.octant_counts()

    @property
    def directions_per_octant(self) -> Optional[int]:
        """Number of directions per octant if every octant holds the same number, else None."""
        counts = self.octant_counts().tolist()
        return int(counts[0]) if all(c == counts[0] for c in counts) else None

    def reordered(self, perm: Tensor) -> "Quadrature":
        """Return the same direction set with the directions permuted.

        Args:
            perm (Tensor): [M] int64 permutation of the directions.

        Returns:
            Quadrature: the permuted set (``cell_width`` permuted too).
        """
        cell_width = None if self.cell_width is None else self.cell_width[perm.to(self.cell_width.device)]
        return Quadrature(omega=self.omega[perm], weight=self.weight[perm], octant=self.octant[perm],
                          cell_width=cell_width)

    def to(self, device: _DeviceLike = None, dtype: Optional[torch.dtype] = None) -> "Quadrature":
        """Return a copy on another device and/or with another floating dtype.

        Args:
            device (torch.device | str, optional): target device. Defaults to None (unchanged).
            dtype (torch.dtype, optional): target dtype of ``omega``, ``weight`` and ``cell_width``. Defaults to None
                (unchanged).

        Returns:
            Quadrature: the converted copy; the octant codes stay int64.
        """
        keywords = {}
        if device is not None:
            keywords["device"] = device
        omega = self.omega.to(dtype=dtype, **keywords) if dtype is not None else self.omega.to(**keywords)
        weight = self.weight.to(dtype=dtype, **keywords) if dtype is not None else self.weight.to(**keywords)
        cell_width = self.cell_width
        if cell_width is not None:
            cell_width = cell_width.to(dtype=dtype, **keywords) if dtype is not None else cell_width.to(**keywords)
        return Quadrature(omega=omega, weight=weight, octant=self.octant.to(**keywords), cell_width=cell_width)


def _gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [-1, 1].

    Args:
        n (int): number of points, at least 1.

    Returns:
        (nodes, weights): [n] float64 numpy nodes in ascending order and [n] float64 numpy weights.
    """
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
    """Gauss-Legendre in ``mu = cos(theta)`` times equal-azimuth product set on the full sphere.

    The azimuths are ``phi_j = (j + 1/2) 2 pi / num_azimuthal`` and the weights ``w_i 2 pi / num_azimuthal`` sum to
    4 pi; with even ``num_polar`` and ``num_azimuthal`` divisible by 4 every octant holds the same number of directions.

    Args:
        num_polar (int): number of Gauss-Legendre nodes in ``cos(theta)``, theta measured from +z.
        num_azimuthal (int): number of equally spaced azimuths, phi measured from +x toward +y.
        device (torch.device | str, optional): device of the tensors. Defaults to None (CPU).
        dtype (torch.dtype, optional): floating dtype. Defaults to torch.float64.

    Returns:
        Quadrature: ``M = num_polar * num_azimuthal`` directions with flat index ``m = i * num_azimuthal + j``.
    """
    if num_azimuthal < 1:
        raise ValueError("num_azimuthal must be >= 1")
    nodes, gauss_weights_np = _gauss_legendre(num_polar)
    mu = torch.as_tensor(nodes, dtype=torch.float64)
    gauss_weights = torch.as_tensor(gauss_weights_np, dtype=torch.float64)
    azimuth_step = 2.0 * math.pi / num_azimuthal
    phi = (torch.arange(num_azimuthal, dtype=torch.float64) + 0.5) * azimuth_step
    sin_theta = torch.sqrt((1.0 - mu * mu).clamp_min(0.0))
    ox = sin_theta[:, None] * torch.cos(phi)[None, :]
    oy = sin_theta[:, None] * torch.sin(phi)[None, :]
    oz = mu[:, None].expand(num_polar, num_azimuthal)
    omega = torch.stack([ox, oy, oz], dim=-1).reshape(-1, 3)
    weight = (gauss_weights[:, None] * azimuth_step).expand(num_polar, num_azimuthal).reshape(-1)
    quadrature = Quadrature.from_directions(omega.to(device=device, dtype=dtype),
                                            weight.to(device=device, dtype=dtype))
    if num_polar % 2 == 0 and num_azimuthal % 4 == 0:
        expected = num_polar * num_azimuthal // 8
        assert quadrature.directions_per_octant == expected, "product set octants are not balanced"
    return quadrature


# ---- cone sets of the collapsed-cone scatter image ------------------------------------------------

def _unit_vector(v: Union[Tensor, Sequence[float]]) -> Tensor:
    """Normalize a 3-vector to unit length.

    Args:
        v (Tensor | Sequence[float]): [3] non-zero vector.

    Returns:
        Tensor: [3] float64 unit vector on the CPU.
    """
    t = torch.as_tensor(v, dtype=torch.float64).detach().to("cpu").reshape(-1)
    if t.numel() != 3:
        raise ValueError("axis must have 3 components")
    norm = torch.linalg.norm(t)
    if not bool(norm > 0):
        raise ValueError("axis must be non-zero")
    return t / norm


def _local_frame(axis: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Deterministic right-handed orthonormal frame (e1, e2, a) about a unit axis.

    The helper vector is the coordinate axis least aligned with ``axis`` (lowest index on ties), with
    ``e1 = normalize(helper - (helper . a) a)`` and ``e2 = a x e1``.

    Args:
        axis (Tensor): [3] float64 unit vector a.

    Returns:
        (e1, e2, a): three [3] float64 unit vectors.
    """
    a = axis
    least_aligned = int(torch.argmin(a.abs()))
    helper = torch.zeros(3, dtype=torch.float64)
    helper[least_aligned] = 1.0
    e1 = helper - (helper @ a) * a
    e1 = e1 / torch.linalg.norm(e1)
    e2 = torch.linalg.cross(a, e1)
    return e1, e2, a


def _cone_product(
    axis: Tensor, num_polar: int, num_azimuthal: int, max_polar_angle: float
) -> tuple[Tensor, Tensor]:
    """Cone product set about an axis: Gauss-Legendre in ``cos(theta)`` times equal azimuths.

    Args:
        axis (Tensor): [3] float64 unit cone axis on the CPU.
        num_polar (int): number of polar nodes, ordered from the axis outward.
        num_azimuthal (int): number of equally spaced azimuths in the frame of ``_local_frame``.
        max_polar_angle (float): cone half-angle in radians, in (0, pi].

    Returns:
        (omega, weight): [M, 3] unit directions and [M] weights in sr, float64 on the CPU.
    """
    if not (0.0 < max_polar_angle <= math.pi):
        raise ValueError("max_polar_angle must be in (0, pi]")
    if num_azimuthal < 1:
        raise ValueError("num_azimuthal must be >= 1")
    cos_max = math.cos(max_polar_angle)
    nodes, gauss_weights = _gauss_legendre(num_polar)
    half_length = 0.5 * (1.0 - cos_max)
    # descending: near-axis first
    cos_theta = torch.as_tensor(cos_max + half_length * (nodes + 1.0), dtype=torch.float64).flip(0)
    polar_weights = torch.as_tensor(gauss_weights * half_length, dtype=torch.float64).flip(0)
    azimuth_step = 2.0 * math.pi / num_azimuthal
    phi = (torch.arange(num_azimuthal, dtype=torch.float64) + 0.5) * azimuth_step
    e1, e2, a = _local_frame(axis)
    sin_theta = torch.sqrt((1.0 - cos_theta * cos_theta).clamp_min(0.0))
    local = (
        (sin_theta[:, None] * torch.cos(phi)[None, :])[..., None] * e1
        + (sin_theta[:, None] * torch.sin(phi)[None, :])[..., None] * e2
        + cos_theta[:, None, None] * a
    )  # [P, A, 3]
    omega = local.reshape(-1, 3)
    omega = omega / torch.linalg.norm(omega, dim=-1, keepdim=True)
    weight = (polar_weights[:, None] * azimuth_step).expand(num_polar, num_azimuthal).reshape(-1)
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
    """Product set on the cone of half-angle ``max_polar_angle`` about ``axis``, the direction set of the scatter image.

    Gauss-Legendre nodes in ``cos(theta)`` on [cos(max_polar_angle), 1], theta measured from ``axis``, times equal
    azimuths ``phi_j = (j + 1/2) 2 pi / num_azimuthal`` in the frame of ``_local_frame``; the weights sum to the cone
    solid angle ``2 pi (1 - cos(max_polar_angle))``.

    Args:
        axis (Tensor | Sequence[float]): [3] cone axis, need not be normalized (for the panel, ``-n_hat`` of
            ``panel_axes``).
        num_polar (int): number of polar nodes.
        num_azimuthal (int): number of azimuths.
        max_polar_angle (float, optional): cone half-angle in radians, in (0, pi]. Defaults to pi / 2 (hemisphere).
        device (torch.device | str, optional): device of the tensors. Defaults to None (the device of ``axis`` if it
            is a tensor, else the CPU).
        dtype (torch.dtype, optional): floating dtype. Defaults to torch.float64.
        tangent_axes (tuple, optional): (u_axis, v_axis), two [3] tangent axes; when given, ``cell_width`` is filled.
            Defaults to None.
        subsamples (int, optional): samples per cell side for the cell widths. Defaults to 16.

    Returns:
        Quadrature: ``M = num_polar * num_azimuthal`` directions with flat index ``m = i * num_azimuthal + j``, i = 0
        the polar node closest to ``axis``.
    """
    if device is None and isinstance(axis, Tensor):
        device = axis.device
    unit_axis = _unit_vector(axis)
    omega, weight = _cone_product(unit_axis, num_polar, num_azimuthal, max_polar_angle)
    quadrature = Quadrature.from_directions(omega.to(device=device, dtype=dtype),
                                            weight.to(device=device, dtype=dtype))
    if tangent_axes is None:
        return quadrature
    u, v = _tangent_frame(unit_axis, *tangent_axes)
    width = _cone_cell_widths(unit_axis, u, v, num_polar, num_azimuthal, max_polar_angle, subsamples)
    return replace(quadrature, cell_width=width.to(device=device, dtype=dtype))


def _tangent_frame(a: Tensor, u_axis, v_axis) -> tuple[Tensor, Tensor]:
    """Gram-Schmidt orthonormalize two tangent axes against a unit axis.

    Args:
        a (Tensor): [3] float64 unit axis.
        u_axis (Tensor | Sequence[float]): [3] first tangent axis.
        v_axis (Tensor | Sequence[float]): [3] second tangent axis.

    Returns:
        (u, v): two [3] float64 unit vectors orthogonal to a and to each other.
    """
    u = _unit_vector(u_axis)
    u = u - (u @ a) * a
    u = u / torch.linalg.norm(u)
    v = _unit_vector(v_axis)
    v = v - (v @ a) * a - (v @ u) * u
    return u, v / torch.linalg.norm(v)


def _cone_cell_widths(a: Tensor, u: Tensor, v: Tensor, num_polar: int, num_azimuthal: int, max_polar_angle: float,
                      subsamples: int) -> Tensor:
    """Gnomonic widths of the solid-angle cell of every direction of a cone product set.

    Node (i, j) owns the cell between the cumulative-Gauss-weight edges in ``cos(theta)`` and the azimuths
    ``[j dphi, (j+1) dphi]``, so its solid angle is its weight; the width is ``sqrt(12)`` times the standard deviation
    of the gnomonic coordinates ``(d.u / d.a, d.v / d.a)`` over an equal-area lattice in the cell.

    Args:
        a (Tensor): [3] float64 unit cone axis.
        u (Tensor): [3] float64 first tangent axis.
        v (Tensor): [3] float64 second tangent axis.
        num_polar (int): number of polar nodes.
        num_azimuthal (int): number of azimuths.
        max_polar_angle (float): cone half-angle in radians.
        subsamples (int): samples per cell side in ``(cos(theta), phi)``.

    Returns:
        Tensor: [M, 2] float64 widths along u and v, in the order of ``_cone_product``.
    """
    float64 = torch.float64
    num_subsamples = int(subsamples)
    fractions = (torch.arange(num_subsamples, dtype=float64) + 0.5) / num_subsamples
    cos_max = math.cos(max_polar_angle)
    _, gauss_weights = _gauss_legendre(num_polar)
    # per polar node, near-axis first
    polar_weights = torch.as_tensor(gauss_weights * 0.5 * (1.0 - cos_max), dtype=float64).flip(0)
    # [P + 1], descending
    cos_edges = torch.cat([torch.ones(1, dtype=float64), 1.0 - torch.cumsum(polar_weights, 0)])
    cos_edges[-1] = cos_max
    azimuth_step = 2.0 * math.pi / num_azimuthal
    e1, e2, _ = _local_frame(a)
    cos_samples = cos_edges[1:, None] + (cos_edges[:-1] - cos_edges[1:])[:, None] * fractions[None, :]  # [P, s]
    # [A, s]
    azimuth_samples = (torch.arange(num_azimuthal, dtype=float64)[:, None] + fractions[None, :]) * azimuth_step
    sin_samples = torch.sqrt((1.0 - cos_samples * cos_samples).clamp_min(0.0))
    lateral = torch.cos(azimuth_samples)[..., None] * e1 + torch.sin(azimuth_samples)[..., None] * e2  # [A, s, 3]
    # d[i, j, p, r] = sin_samples[i, p] lateral[j, r] + cos_samples[i, p] a     [P, A, s, s, 3]
    d = sin_samples[:, None, :, None, None] * lateral[None, :, None, :, :] + cos_samples[:, None, :, None, None] * a
    normal_component = (d @ a).clamp_min(1e-6)
    alpha, beta = (d @ u) / normal_component, (d @ v) / normal_component
    num_samples = float(num_subsamples * num_subsamples)

    def width(x: Tensor) -> Tensor:
        mean = x.sum(dim=(2, 3)) / num_samples
        var = ((x * x).sum(dim=(2, 3)) / num_samples - mean * mean).clamp_min(0.0)
        return torch.sqrt(12.0 * var).reshape(-1)

    return torch.stack([width(alpha), width(beta)], dim=-1)


# =====================================================================================================
# Legendre polynomials and real spherical harmonics
# =====================================================================================================

def legendre_polynomials(L: int, x: Tensor) -> Tensor:
    """Legendre polynomials P_0 to P_L by the recurrence ``(l+1) P_{l+1} = (2l+1) x P_l - l P_{l-1}``.

    Args:
        L (int): maximum degree, at least 0.
        x (Tensor): [...] arguments in [-1, 1].

    Returns:
        Tensor: [..., L+1] values, on the device and dtype of x.
    """
    if L < 0:
        raise ValueError("L must be >= 0")
    values = [torch.ones_like(x)]
    if L >= 1:
        values.append(x.clone())
    for l in range(1, L):
        values.append(((2 * l + 1) * x * values[l] - l * values[l - 1]) / (l + 1))
    return torch.stack(values, dim=-1)


def num_spherical_harmonics(L: int) -> int:
    """Number of real spherical harmonics up to degree L.

    Args:
        L (int): maximum degree.

    Returns:
        int: ``(L+1)^2``.
    """
    return (L + 1) * (L + 1)


def spherical_harmonic_degree_index(L: int, device: _DeviceLike = None) -> Tensor:
    """Degree l of every flat spherical-harmonic index ``n = l*l + (k + l)``.

    Args:
        L (int): maximum degree.
        device (torch.device | str, optional): device of the result. Defaults to None (CPU).

    Returns:
        Tensor: [(L+1)^2] int64 degrees.
    """
    return torch.cat([torch.full((2 * l + 1,), l, dtype=torch.int64) for l in range(L + 1)]).to(device)


def real_spherical_harmonics(L: int, omega: Tensor) -> Tensor:
    """Real orthonormal spherical harmonics at unit directions, polar axis +z, no Condon-Shortley phase.

    They are evaluated without trigonometric functions by stable normalized recurrences in z and ``(x + i y)^k``; the
    flat index is ``n = l*l + (k + l)``.

    Args:
        L (int): maximum degree, at least 0.
        omega (Tensor): [..., 3] unit directions (not renormalized).

    Returns:
        Tensor: [..., (L+1)^2] values ``Y_n(omega)``, on the device and dtype of omega.
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
class SphericalHarmonicBasis:
    """Real spherical-harmonic tables on the directions of a quadrature.

    Attributes:
        L (int): maximum degree; there are ``N = (L+1)^2`` moments.
        Y (Tensor): [M, N] values ``Y_n(Omega_m)``.
        Y_weighted (Tensor): [M, N] values ``w_m Y_n(Omega_m)`` with the quadrature weights ``w_m`` in sr.
        degree_of_moment (Tensor): [N] int64 degree l of each flat index n.
    """

    L: int
    Y: Tensor
    Y_weighted: Tensor
    degree_of_moment: Tensor

    @property
    def num_moments(self) -> int:
        """Number of spherical-harmonic moments N."""
        return int(self.Y.shape[1])

    @property
    def num_directions(self) -> int:
        """Number of directions M."""
        return int(self.Y.shape[0])


def spherical_harmonic_basis(L: int, quadrature: Quadrature) -> SphericalHarmonicBasis:
    """Build the spherical-harmonic tables of degree L on the directions of a quadrature.

    Args:
        L (int): maximum degree.
        quadrature (Quadrature): direction set.

    Returns:
        SphericalHarmonicBasis: the tables, on the device and dtype of the quadrature.
    """
    Y = real_spherical_harmonics(L, quadrature.omega)
    return SphericalHarmonicBasis(L=L, Y=Y, Y_weighted=quadrature.weight[:, None] * Y,
                                  degree_of_moment=spherical_harmonic_degree_index(L, quadrature.omega.device))


def moments_from_ordinates(psi: Tensor, basis: SphericalHarmonicBasis) -> Tensor:
    """Spherical-harmonic moments of angular ordinates, ``phi_n = sum_m w_m Y_n(Omega_m) psi_m``.

    Args:
        psi (Tensor): [..., M] values at the quadrature directions.
        basis (SphericalHarmonicBasis): tables of the quadrature.

    Returns:
        Tensor: [..., N] moments phi.
    """
    return psi @ basis.Y_weighted


def ordinates_from_moments(phi: Tensor, basis: SphericalHarmonicBasis) -> Tensor:
    """Angular ordinates from spherical-harmonic moments, ``psi_m = sum_n phi_n Y_n(Omega_m)``.

    Args:
        phi (Tensor): [..., N] moments.
        basis (SphericalHarmonicBasis): tables of the quadrature.

    Returns:
        Tensor: [..., M] values psi at the quadrature directions.
    """
    return phi @ basis.Y.T


# =====================================================================================================
# Scattering source
# =====================================================================================================

def scatter_source_moments(mu_s_l_volume: Tensor, phi: Tensor, *, moment_major: bool = False,
                           cell_major: bool = False) -> Tensor:
    """Spherical-harmonic moments T of the scattering source from the flux moments and the Legendre transfer moments.

    ``T[g, n] = sum_g' mu_s_l_volume[g', g, l(n)] phi[g', n]``, so that ``S_g(Omega) = sum_n T[g, n] Y_n(Omega)``. On
    float32 CUDA tensors one fused kernel forms T (``kernels.scattering_kernels``); the PyTorch path skips the all-zero
    transfer blocks (upscatter).

    Args:
        mu_s_l_volume (Tensor): ``[G_in, G_out, L+1, *spatial]`` Legendre transfer moments ``mu_s_l`` per voxel in
            cm^-1.
        phi (Tensor): ``[G_in, N, *rest]`` flux moments, ``N = (L_phi+1)^2`` with ``L_phi <= L``; rest is spatial or
            ``(4, *spatial)`` (linear-discontinuous moments), and phi may be a non-contiguous view.
        moment_major (bool, optional): allocate the result moment-major, ``[N, G_out, *rest]``, and return its
            ``movedim(0, 1)`` view (the layout ``combined_scatter_image`` contracts without a copy). Defaults to False.
        cell_major (bool, optional): allocate the result in the sweep's layout, ``[N, *spatial, G_out, (4)]``, and
            return its ``[G_out, N, *rest]`` view (the layout ``scatter_source_function`` contracts without a copy).
            Defaults to False.

    Returns:
        Tensor: ``[G_out, N, *rest]`` source moments T in cm^-1 times the unit of phi, contiguous unless
        ``moment_major`` or ``cell_major``.
    """
    G_in, G_out, num_legendre = mu_s_l_volume.shape[:3]
    spatial = tuple(mu_s_l_volume.shape[3:])
    if phi.shape[0] != G_in:
        raise ValueError(f"phi has {phi.shape[0]} groups, transfer has {G_in}")
    N = phi.shape[1]
    L_phi = math.isqrt(N) - 1
    if (L_phi + 1) ** 2 != N:
        raise ValueError(f"phi.shape[1]={N} is not a square (L+1)^2")
    if L_phi + 1 > num_legendre:
        raise ValueError(f"phi has degree {L_phi} > transfer Legendre order {num_legendre - 1}")
    rest = tuple(phi.shape[2:])
    if not (rest == spatial or (len(rest) == len(spatial) + 1 and rest[1:] == spatial)):
        raise ValueError(f"phi trailing shape {rest} must be spatial {spatial} or (4, *spatial)")
    if moment_major and cell_major:
        raise ValueError("choose at most one of moment_major and cell_major")
    if scattering_kernels.use_kernel(mu_s_l_volume, phi):
        layout = "cell_major" if cell_major else "moment_major" if moment_major else "contiguous"
        return scattering_kernels.scattering_moments(mu_s_l_volume, phi, layout)
    if moment_major:
        out = torch.zeros((N, G_out) + rest, device=phi.device, dtype=phi.dtype).movedim(0, 1)
    elif cell_major:
        storage = torch.zeros((N,) + spatial + (G_out,) + rest[:len(rest) - len(spatial)], device=phi.device,
                              dtype=phi.dtype)
        out = scattering_kernels.cell_major_view(storage, len(spatial))
    else:
        out = torch.zeros((G_out, N) + rest, device=phi.device, dtype=phi.dtype)
    return _scatter_source_moments_torch(mu_s_l_volume, phi, out)


def _scatter_source_moments_torch(mu_s_l_volume: Tensor, phi: Tensor, out: Tensor) -> Tensor:
    """PyTorch path of ``scatter_source_moments``: accumulate every non-zero transfer block in place into ``out``.

    Args:
        mu_s_l_volume (Tensor): ``[G_in, G_out, L+1, *spatial]`` Legendre transfer moments.
        phi (Tensor): ``[G_in, N, *rest]`` flux moments.
        out (Tensor): ``[G_out, N, *rest]`` zeros (any strides), filled in place.

    Returns:
        Tensor: ``out``.
    """
    G_in, G_out = mu_s_l_volume.shape[:2]
    spatial = tuple(mu_s_l_volume.shape[3:])
    N = phi.shape[1]
    linear_discontinuous_axis = tuple(phi.shape[2:]) != spatial
    degree_of_moment = spherical_harmonic_degree_index(math.isqrt(N) - 1, mu_s_l_volume.device)
    nonzero = (mu_s_l_volume.reshape(G_in, G_out, -1) != 0).any(dim=-1).cpu()  # one host sync
    for g_in in range(G_in):
        for g_out in range(G_out):
            if not bool(nonzero[g_in, g_out]):
                continue
            transfer_rows = mu_s_l_volume[g_in, g_out].index_select(0, degree_of_moment)  # [N, *spatial]
            if linear_discontinuous_axis:
                transfer_rows = transfer_rows.unsqueeze(1)  # [N, 1, *spatial]
            out[g_out].addcmul_(transfer_rows, phi[g_in])
    return out


def scatter_source_at_directions(T: Tensor, Y_selected: Tensor) -> Tensor:
    """Scattering source at K directions, ``S[k, g, ...] = sum_n Y_selected[k, n] T[g, n, ...]``.

    Args:
        T (Tensor): ``[G, N, *rest]`` source moments from
            :func:`~citrus.scatter_solver.angular_quadrature.scatter_source_moments`.
        Y_selected (Tensor): [K, N] rows of ``SphericalHarmonicBasis.Y`` at the K directions.

    Returns:
        Tensor: ``[K, G, *rest]`` source per direction.
    """
    if T.shape[1] != Y_selected.shape[1]:
        raise ValueError(f"T has {T.shape[1]} moments, Y_selected has {Y_selected.shape[1]}")
    return torch.tensordot(Y_selected, T, dims=([1], [1]))
