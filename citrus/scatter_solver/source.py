"""Source model of the solver: the spectrum, the emission profile and the collimation of the focal spot or the beam.

The focal spot emits one photon per steradian in total, split over the energy groups by the library's spectrum
fractions, collimated to the panel and optionally weighted by an emission profile (a bowtie or beam filter, the heel
effect) given per detector pixel. A parallel beam carries a unit fluence over the panel's area, weighted by the same
profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from .cross_sections import MultigroupLibrary
from .geometry import ConeGeometry, device_tensor

ArrayLike = Union[Tensor, np.ndarray, Sequence[float]]


def _as_float64_cpu(x: ArrayLike) -> Tensor:
    """Return an array as a float64 tensor on the CPU.

    Args:
        x (ArrayLike): the array.

    Returns:
        Tensor: the values, float64 on the CPU.
    """
    if isinstance(x, Tensor):
        return x.detach().to("cpu", torch.float64)
    return torch.as_tensor(np.asarray(x, dtype=np.float64))


@dataclass(frozen=True, eq=False)
class SourceModel:
    """Isotropic point source at ``geometry.source_position(angle)``, collimated to the panel, with an emission profile.

    The angular weight ``b(Omega)`` is 0 for directions whose ray from the focal spot misses the active panel area (edge
    to edge, outer half pixels included). Inside it is 1 without an emission profile, else the profile at the point
    where the ray hits the panel, interpolated bilinearly between the pixel centres and held constant from the outermost
    pixel centres to the panel edge. The scale of the profile cancels in the scatter-to-primary ratio. For a parallel
    beam (``ParallelGeometry``) the source is a unit fluence of parallel rays over the panel's area, weighted by the
    profile at the panel point of each ray (:meth:`beam_weight`).

    Attributes:
        emission_profile (Tensor, optional): [rows, channels] relative emission toward each pixel centre, the same for
            every view, or [V, rows, channels] one per view; non-negative. Defaults to None (uniform emission).
        emission_profile_angles (Sequence[float], optional): [V] view angles in rad of a per-view profile; a view uses
            the profile of the nearest angle. Defaults to None (required for V > 1).
    """

    emission_profile: Optional[ArrayLike] = None
    emission_profile_angles: Optional[ArrayLike] = None
    _device_cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Check the emission profile and store it as a float64 CPU tensor."""
        if self.emission_profile is None:
            if self.emission_profile_angles is not None:
                raise ValueError("emission_profile_angles needs an emission_profile")
            return
        profile = _as_float64_cpu(self.emission_profile)
        if profile.dim() not in (2, 3):
            raise ValueError(f"emission_profile must be [rows, channels] or [views, rows, channels], got "
                             f"{tuple(profile.shape)}")
        if not bool(torch.isfinite(profile).all()) or bool((profile < 0).any()):
            raise ValueError("emission_profile must be finite and non-negative")
        angles = None
        if profile.dim() == 3:
            if self.emission_profile_angles is None and profile.shape[0] > 1:
                raise ValueError("a per-view emission_profile needs emission_profile_angles")
            if self.emission_profile_angles is not None:
                angles = tuple(float(a) for a in _as_float64_cpu(self.emission_profile_angles).reshape(-1))
                if len(angles) != profile.shape[0]:
                    raise ValueError(f"emission_profile has {profile.shape[0]} views, emission_profile_angles "
                                     f"{len(angles)}")
        elif self.emission_profile_angles is not None:
            raise ValueError("emission_profile_angles applies to a per-view [views, rows, channels] profile only")
        object.__setattr__(self, "emission_profile", profile)
        object.__setattr__(self, "emission_profile_angles", angles)

    def per_group_strength(self, library: MultigroupLibrary) -> Tensor:
        """Photons per steradian of each group, ``f_g = library.spectrum_fraction`` (summing to 1).

        Args:
            library (MultigroupLibrary): the library.

        Returns:
            Tensor: [G] photons per steradian per group, per source photon/sr.
        """
        return library.spectrum_fraction

    def profile_at(self, angle: float, device=None, dtype: torch.dtype = torch.float32) -> Optional[Tensor]:
        """Return the emission profile of one view, or None for uniform emission.

        Args:
            angle (float): view angle in rad; a per-view profile gives the view of the nearest angle.
            device (torch.device, optional): device of the result. Defaults to None (the CPU).
            dtype (torch.dtype, optional): dtype of the result. Defaults to torch.float32.

        Returns:
            Tensor or None: [rows, channels] profile.
        """
        if self.emission_profile is None:
            return None
        key = (str(torch.device(device) if device is not None else torch.device("cpu")), dtype)
        profile = self._device_cache.get(key)
        if profile is None:
            profile = self.emission_profile.to(device=device, dtype=dtype)
            self._device_cache[key] = profile
        if profile.dim() == 2:
            return profile
        if self.emission_profile_angles is None:
            return profile[0]
        view = int(np.argmin(np.abs(np.asarray(self.emission_profile_angles) - float(angle))))
        return profile[view]

    def angular_weight(self, directions: Tensor, geometry: ConeGeometry, angle: float) -> Tensor:
        """Angular weight ``b(Omega)`` of the source: the emission profile (1 by default) where the ray hits the panel.

        Args:
            directions (Tensor): [..., 3] directions in the recon frame, not necessarily normalized.
            geometry (ConeGeometry): scan geometry.
            angle (float): view angle in rad.

        Returns:
            Tensor: [...] weights, on the device and in the dtype of ``directions``, 0 off the panel.
        """
        source_position = device_tensor(geometry.source_position(angle), directions.device, directions.dtype)
        t = geometry.panel_distance_along(source_position.expand_as(directions), directions, angle)  # [...]
        forward = torch.isfinite(t) & (t > 0)
        t_safe = torch.where(forward, t, torch.zeros_like(t))
        hit = source_position + t_safe[..., None] * directions
        u, v = geometry.panel_uv_of_points(hit, angle)
        half_u, half_v = geometry.panel_half_extent()
        inside = (((u + geometry.det_channel_offset).abs() <= half_u)
                  & ((v + geometry.det_row_offset).abs() <= half_v))
        weight = (forward & inside).to(directions.dtype)
        profile = self.profile_at(angle, directions.device, directions.dtype)
        if profile is None:
            return weight
        if tuple(profile.shape) != geometry.image_shape:
            raise ValueError(f"emission_profile is {tuple(profile.shape)} per view, the panel {geometry.image_shape}")
        return weight * panel_interpolate(profile, u, v, geometry)

    def beam_weight(self, points: Tensor, geometry, angle: float) -> Tensor:
        """Weight ``b(r)`` of a parallel beam at points.

        It is the emission profile (1 by default) at the panel point of the beam's ray through each point, and 0
        outside the beam's cross-section (the panel's area).

        Args:
            points (Tensor): [..., 3] points in the recon frame in cm.
            geometry (ParallelGeometry): parallel-beam geometry.
            angle (float): view angle in rad.

        Returns:
            Tensor: [...] weights, on the device and in the dtype of ``points``.
        """
        u, v = geometry.panel_uv_of_points(points, angle)
        half_u, half_v = geometry.panel_half_extent()
        weight = (((u + geometry.det_channel_offset).abs() <= half_u)
                  & ((v + geometry.det_row_offset).abs() <= half_v)).to(points.dtype)
        profile = self.profile_at(angle, points.device, points.dtype)
        if profile is None:
            return weight
        if tuple(profile.shape) != geometry.image_shape:
            raise ValueError(f"emission_profile is {tuple(profile.shape)} per view, the panel {geometry.image_shape}")
        return weight * panel_interpolate(profile, u, v, geometry)


def panel_interpolate(image: Tensor, u: Tensor, v: Tensor, geometry: ConeGeometry) -> Tensor:
    """Interpolate a detector image bilinearly at panel coordinates, constant beyond the outermost pixel centres.

    Args:
        image (Tensor): [rows, channels] values at the pixel centres.
        u (Tensor): [...] panel coordinates along the channels in cm.
        v (Tensor): [...] panel coordinates along the rows in cm, the shape of ``u``.
        geometry (ConeGeometry or ParallelGeometry): the panel.

    Returns:
        Tensor: [...] interpolated values, in the dtype of ``image``.
    """
    num_rows, num_channels = geometry.num_det_rows, geometry.num_det_channels

    def bracket(x: Tensor, n: int) -> tuple[Tensor, Tensor, Tensor]:
        x = x.clamp(0.0, n - 1.0)
        i0 = x.floor().long().clamp(0, max(n - 2, 0))
        i1 = (i0 + 1).clamp(max=n - 1)
        return i0, i1, (x - i0.to(x.dtype)).clamp(0.0, 1.0)

    channel = (u + geometry.det_channel_offset) / geometry.delta_det_channel + (num_channels - 1) / 2.0
    row = (v + geometry.det_row_offset) / geometry.delta_det_row + (num_rows - 1) / 2.0
    c0, c1, wc = bracket(channel.to(image.dtype), num_channels)
    r0, r1, wr = bracket(row.to(image.dtype), num_rows)
    top = image[r0, c0] + (image[r0, c1] - image[r0, c0]) * wc
    bottom = image[r1, c0] + (image[r1, c1] - image[r1, c0]) * wc
    return top + (bottom - top) * wr


def bowtie_emission_profile(geometry: ConeGeometry, edge_weight: float = 0.2) -> Tensor:
    """Build a bowtie-like emission profile across the channels, the same for every row.

    The profile is ``edge_weight ** ((u / u_max)^2)``: 1 on the central ray and ``edge_weight`` at the outermost
    channel, the transmission at one energy of a filter whose thickness grows with the square of the channel position.

    Args:
        geometry (ConeGeometry or ParallelGeometry): the panel.
        edge_weight (float, optional): profile at the outermost channel, in (0, 1]. Defaults to 0.2.

    Returns:
        Tensor: [rows, channels] profile, float64 on the CPU.
    """
    if not 0.0 < float(edge_weight) <= 1.0:
        raise ValueError("edge_weight must be in (0, 1]")
    u, _ = geometry.pixel_uv(dtype=torch.float64)
    u_max = float(u.abs().max())
    x = (u / u_max) if u_max > 0 else torch.zeros_like(u)
    profile = torch.exp(math.log(float(edge_weight)) * x * x)  # [channels]
    return profile[None, :].expand(geometry.num_det_rows, -1).clone()
