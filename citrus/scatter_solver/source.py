"""Point-source model of the solver: the spectrum and the collimation of the focal spot.

The focal spot emits one photon per steradian in total, split over the energy groups by the library's spectrum
fractions and collimated to the panel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .cross_sections import MultigroupLibrary
from .geometry import ConeGeometry


@dataclass(frozen=True)
class SourceModel:
    """Isotropic point source at ``geometry.source_position(angle)``, collimated to the panel, without a bowtie filter.

    The angular weight ``b(Omega)`` is 1 for directions whose ray from the focal spot hits the active panel area (edge
    to edge, outer half pixels included) and 0 otherwise.
    """

    def per_group_strength(self, library: MultigroupLibrary) -> Tensor:
        """Photons per steradian of each group, ``f_g = library.spectrum_fraction`` (summing to 1).

        Args:
            library (MultigroupLibrary): the library.

        Returns:
            Tensor: [G] photons per steradian per group, per source photon/sr.
        """
        return library.spectrum_fraction

    def angular_weight(self, directions: Tensor, geometry: ConeGeometry, angle: float) -> Tensor:
        """Angular weight ``b(Omega)`` of the source: 1 where the ray from the focal spot hits the panel, else 0.

        Args:
            directions (Tensor): [..., 3] directions in the recon frame, not necessarily normalized.
            geometry (ConeGeometry): scan geometry.
            angle (float): view angle in rad.

        Returns:
            Tensor: [...] weights, on the device and in the dtype of ``directions``.
        """
        source_position = torch.tensor(geometry.source_position(angle), device=directions.device,
                                       dtype=directions.dtype)
        t = geometry.panel_distance_along(source_position.expand_as(directions), directions, angle)  # [...]
        forward = torch.isfinite(t) & (t > 0)
        t_safe = torch.where(forward, t, torch.zeros_like(t))
        hit = source_position + t_safe[..., None] * directions
        u, v = geometry.panel_uv_of_points(hit, angle)
        half_u, half_v = geometry.panel_half_extent()
        inside = (((u + geometry.det_channel_offset).abs() <= half_u)
                  & ((v + geometry.det_row_offset).abs() <= half_v))
        return (forward & inside).to(directions.dtype)
