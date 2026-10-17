"""Point-source model: spectrum and collimation.

The focal spot emits 1 photon per steradian in total; group g carries f_g = spectrum_fraction[g]
photons per steradian (sum_g f_g = 1), modulated by an angular weight b(Omega) for collimation:

    q_g(r0, Omega) = f_g b(Omega) delta(r - r0)          (photons per steradian per source photon/sr)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .cross_sections import MultigroupLibrary
from .geometry import ConeGeometry


@dataclass(frozen=True)
class SourceModel:
    """Isotropic point source at geometry.source_position(angle), collimated to the panel.

    Collimation: b(Omega) = 1 if the ray from the focal spot along Omega hits the active panel area (edge to edge,
    including the outer half pixels), else 0: the source is collimated to the detector. No bowtie filter.
    """

    def per_group_strength(self, library: MultigroupLibrary) -> Tensor:
        """Photons per steradian per group, f_g = library.spectrum_fraction: [G]."""
        return library.spectrum_fraction

    def angular_weight(self, directions: Tensor, geometry: ConeGeometry, angle: float) -> Tensor:
        """Angular weight b(Omega) of the source for directions [..., 3] (recon frame). Returns [...].

        Directions need not be normalized. Same device and dtype as `directions`.
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
