"""Source model of the solver: the spectrum, emission profile, collimation and beam filter of the focal spot.

The focal spot emits one photon per steradian in total, split over the energy groups by the library's spectrum
fractions, collimated to the panel (optionally a field wider than the panel, and off-focal emission past the field
edge) and optionally weighted by an emission profile (the heel effect, a gray bowtie) given per detector pixel. A beam
filter (a bowtie) given as a thickness per pixel hardens the spectrum per direction. A parallel beam carries a unit
fluence over the field, weighted by the same profile. The model is described in docs/source/method.rst, "Source
emission profile and open field".
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from .cross_sections import Material, MultigroupLibrary, material_definition
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


def _non_negative(name: str, value: float) -> float:
    """Return a setting as a float after checking that it is finite and non-negative.

    Args:
        name (str): name of the setting, for the error message.
        value (float): the value.

    Returns:
        float: the value.
    """
    if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
    return float(value)


@dataclass(frozen=True, eq=False)
class SourceModel:
    """Isotropic point source at ``geometry.source_position(angle)``, collimated, with an emission profile and a filter.

    The angular weight ``b(Omega)`` is 1 for directions whose ray from the focal spot hits the field: the active panel
    area (edge to edge, outer half pixels included) widened by ``field_margin_channels_cm`` past both channel edges and
    ``field_margin_rows_cm`` past both row edges. Past the field edge it is the off-focal fraction of the distance
    from the ray's panel point to the field rectangle (0 without an off-focal table). It is multiplied by the emission
    profile at the ray's panel point, interpolated bilinearly between the pixel centres and held at the value of the
    outermost pixel centres beyond them. A beam filter of ``filter_material`` adds the path ``rho_f t(Omega)`` through
    the filter to every transmission from the focal spot, with the thickness interpolated like the profile. For a
    parallel beam (``ParallelGeometry``) the same rules apply to the panel point of each ray (:meth:`beam_weight`).

    Attributes:
        emission_profile (Tensor, optional): [rows, channels] relative emission toward each pixel centre, the same for
            every view, or [V, rows, channels] one per view; non-negative. Defaults to None (uniform emission).
        emission_profile_angles (Sequence[float], optional): [V] view angles in rad of a per-view profile; a view uses
            the profile of the nearest angle. Defaults to None (required for V > 1).
        field_margin_channels_cm (float): how far the collimated field reaches past both panel edges along the
            channels, in cm at the panel. Defaults to 0.0 (the field ends at the panel edges).
        field_margin_rows_cm (float): how far the field reaches past both panel edges along the rows, in cm at the
            panel. Defaults to 0.0.
        off_focal_distance_cm (Sequence[float], optional): [K] distances past the field edge in cm at the panel,
            ascending, at which ``off_focal_fraction`` is given. Defaults to None (no off-focal emission).
        off_focal_fraction (Sequence[float], optional): [K] emission past the field edge relative to the emission at
            the edge, interpolated linearly in the distance, its first value below the first distance and 0 beyond the
            last. Defaults to None.
        filter_material (str or Material, optional): material of the beam filter (catalogue name or ``Material``),
            at its nominal density. Defaults to None (no filter).
        filter_thickness_cm (Tensor, optional): [channels] (the same for every row) or [rows, channels] path length in
            cm through the filter of the ray from the focal spot to each pixel centre, its minimum thickness included;
            non-negative. Defaults to None.
    """

    emission_profile: Optional[ArrayLike] = None
    emission_profile_angles: Optional[ArrayLike] = None
    field_margin_channels_cm: float = 0.0
    field_margin_rows_cm: float = 0.0
    off_focal_distance_cm: Optional[ArrayLike] = None
    off_focal_fraction: Optional[ArrayLike] = None
    filter_material: Optional[Union[str, Material]] = None
    filter_thickness_cm: Optional[ArrayLike] = None
    _device_cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Check the settings and store the tables as float64 CPU tensors or tuples of floats."""
        object.__setattr__(self, "field_margin_channels_cm",
                           _non_negative("field_margin_channels_cm", self.field_margin_channels_cm))
        object.__setattr__(self, "field_margin_rows_cm", _non_negative("field_margin_rows_cm",
                                                                       self.field_margin_rows_cm))
        self._check_off_focal()
        self._check_filter()
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

    def _check_off_focal(self) -> None:
        """Check the off-focal table and store it as tuples of floats."""
        if self.off_focal_distance_cm is None and self.off_focal_fraction is None:
            return
        if self.off_focal_distance_cm is None or self.off_focal_fraction is None:
            raise ValueError("off_focal_distance_cm and off_focal_fraction must be given together")
        distance = _as_float64_cpu(self.off_focal_distance_cm).reshape(-1)
        fraction = _as_float64_cpu(self.off_focal_fraction).reshape(-1)
        if distance.numel() < 1 or distance.numel() != fraction.numel():
            raise ValueError(f"the off-focal table needs as many distances ({distance.numel()}) as fractions "
                             f"({fraction.numel()}), at least one")
        if not bool(torch.isfinite(distance).all()) or bool((distance < 0).any()) or bool(
                (distance[1:] <= distance[:-1]).any()):
            raise ValueError("off_focal_distance_cm must be finite, >= 0 and strictly ascending")
        if not bool(torch.isfinite(fraction).all()) or bool((fraction < 0).any()):
            raise ValueError("off_focal_fraction must be finite and non-negative")
        object.__setattr__(self, "off_focal_distance_cm", tuple(float(d) for d in distance))
        object.__setattr__(self, "off_focal_fraction", tuple(float(f) for f in fraction))

    def _check_filter(self) -> None:
        """Check the filter and store its thickness as a [rows or 1, channels] float64 CPU tensor."""
        if self.filter_material is None and self.filter_thickness_cm is None:
            return
        if self.filter_material is None or self.filter_thickness_cm is None:
            raise ValueError("filter_material and filter_thickness_cm must be given together")
        material_definition(self.filter_material)  # an unknown name raises here
        thickness = _as_float64_cpu(self.filter_thickness_cm)
        if thickness.dim() == 1:
            thickness = thickness[None, :]
        if thickness.dim() != 2 or thickness.numel() == 0:
            raise ValueError(f"filter_thickness_cm must be [channels] or [rows, channels], got "
                             f"{tuple(_as_float64_cpu(self.filter_thickness_cm).shape)}")
        if not bool(torch.isfinite(thickness).all()) or bool((thickness < 0).any()):
            raise ValueError("filter_thickness_cm must be finite and non-negative")
        object.__setattr__(self, "filter_thickness_cm", thickness.contiguous())

    # ---- properties --------------------------------------------------------------------------
    @property
    def has_filter(self) -> bool:
        """True when the source has a beam filter."""
        return self.filter_material is not None

    @property
    def filter_density(self) -> float:
        """Density of the filter material in g/cm^3 (its nominal density), 0.0 without a filter."""
        if self.filter_material is None:
            return 0.0
        return float(material_definition(self.filter_material).density)

    def field_half_extent(self, geometry) -> tuple[float, float]:
        """Half widths of the region of the panel plane that receives any emission, about the panel centre.

        Args:
            geometry (ConeGeometry or ParallelGeometry): the panel.

        Returns:
            (float, float): half widths along u and v in cm at the panel: the panel's, plus the field margins, plus the
            last distance of the off-focal table.
        """
        half_u, half_v = geometry.panel_half_extent()
        reach = self.off_focal_distance_cm[-1] if self.off_focal_distance_cm is not None else 0.0
        return (half_u + self.field_margin_channels_cm + reach, half_v + self.field_margin_rows_cm + reach)

    def per_group_strength(self, library: MultigroupLibrary) -> Tensor:
        """Photons per steradian of each group, ``f_g = library.spectrum_fraction`` (summing to 1).

        Args:
            library (MultigroupLibrary): the library.

        Returns:
            Tensor: [G] photons per steradian per group, per source photon/sr.
        """
        return library.spectrum_fraction

    # ---- tables on a device ------------------------------------------------------------------
    def _on_device(self, name: str, table: Tensor, device, dtype: torch.dtype) -> Tensor:
        """A table on a device and in a dtype, cached.

        Args:
            name (str): name of the table.
            table (Tensor): the float64 CPU table.
            device (torch.device): device of the result.
            dtype (torch.dtype): dtype of the result.

        Returns:
            Tensor: the table.
        """
        key = (name, str(torch.device(device) if device is not None else torch.device("cpu")), dtype)
        value = self._device_cache.get(key)
        if value is None:
            value = table.to(device=device, dtype=dtype)
            self._device_cache[key] = value
        return value

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
        profile = self._on_device("emission_profile", self.emission_profile, device, dtype)
        if profile.dim() == 2:
            return profile
        if self.emission_profile_angles is None:
            return profile[0]
        view = int(np.argmin(np.abs(np.asarray(self.emission_profile_angles) - float(angle))))
        return profile[view]

    def thickness_at(self, geometry, device=None, dtype: torch.dtype = torch.float32) -> Optional[Tensor]:
        """Return the filter thickness at every pixel centre, or None without a filter.

        Args:
            geometry (ConeGeometry or ParallelGeometry): the panel.
            device (torch.device, optional): device of the result. Defaults to None (the CPU).
            dtype (torch.dtype, optional): dtype of the result. Defaults to torch.float32.

        Returns:
            Tensor or None: [rows, channels] thickness in cm.
        """
        if self.filter_thickness_cm is None:
            return None
        thickness = self._on_device("filter_thickness", self.filter_thickness_cm, device, dtype)
        rows, channels = geometry.image_shape
        if int(thickness.shape[1]) != channels or int(thickness.shape[0]) not in (1, rows):
            raise ValueError(f"filter_thickness_cm is {tuple(thickness.shape)}, the panel {geometry.image_shape} "
                             f"(or [{channels}] for every row)")
        return thickness.expand(rows, channels)

    def filter_attenuation(self, library: MultigroupLibrary, device=None,
                           dtype: torch.dtype = torch.float32) -> Optional[Tensor]:
        """Mass attenuation of the filter at the library's fine energies, from the photon data, cached per library.

        Args:
            library (MultigroupLibrary): the library (its ``fine_energy``).
            device (torch.device, optional): device of the result. Defaults to None (the CPU).
            dtype (torch.dtype, optional): dtype of the result. Defaults to torch.float32.

        Returns:
            Tensor or None: [E] total mass attenuation in cm^2/g, None without a filter.
        """
        if self.filter_material is None:
            return None
        definition = material_definition(self.filter_material)
        key = (definition.name, float(definition.density), definition.composition,
               str(torch.device(device) if device is not None else torch.device("cpu")), dtype)
        return _fine_attenuation(library.fine_energy, key, lambda: definition)

    # ---- weights and filter paths ------------------------------------------------------------
    @staticmethod
    def _panel_point(directions: Tensor, geometry: ConeGeometry, angle: float) -> tuple[Tensor, Tensor, Tensor]:
        """Panel coordinates where the rays from the focal spot along ``directions`` meet the panel.

        Args:
            directions (Tensor): [..., 3] directions in the recon frame, not necessarily normalized.
            geometry (ConeGeometry): scan geometry.
            angle (float): view angle in rad.

        Returns:
            (Tensor, Tensor, Tensor): whether the ray reaches the panel surface forward [...] (bool), and u and v [...]
            in cm (meaningless where it does not).
        """
        source_position = device_tensor(geometry.source_position(angle), directions.device, directions.dtype)
        t = geometry.panel_distance_along(source_position.expand_as(directions), directions, angle)  # [...]
        forward = torch.isfinite(t) & (t > 0)
        t_safe = torch.where(forward, t, torch.zeros_like(t))
        hit = source_position + t_safe[..., None] * directions
        u, v = geometry.panel_uv_of_points(hit, angle)
        return forward, u, v

    def _field_weight(self, u: Tensor, v: Tensor, geometry) -> Tensor:
        """Collimation weight at panel points: 1 in the field, the off-focal fraction past its edge (0 without one).

        Args:
            u (Tensor): [...] panel coordinates along the channels in cm.
            v (Tensor): [...] panel coordinates along the rows in cm.
            geometry (ConeGeometry or ParallelGeometry): the panel.

        Returns:
            Tensor: [...] weights, bool without an off-focal table, else in the dtype of ``u``.
        """
        half_u, half_v = geometry.panel_half_extent()
        offset_u = (u + geometry.det_channel_offset).abs()
        offset_v = (v + geometry.det_row_offset).abs()
        inside = ((offset_u <= half_u + self.field_margin_channels_cm)
                  & (offset_v <= half_v + self.field_margin_rows_cm))
        if self.off_focal_distance_cm is None:
            return inside
        # past the field edge: the off-focal fraction of the distance to the field rectangle
        past_u = (offset_u - (half_u + self.field_margin_channels_cm)).clamp_min(0.0)
        past_v = (offset_v - (half_v + self.field_margin_rows_cm)).clamp_min(0.0)
        distance = torch.sqrt(past_u * past_u + past_v * past_v).contiguous()
        nodes = self._on_device("off_focal_distance", torch.tensor(self.off_focal_distance_cm, dtype=torch.float64),
                                u.device, u.dtype)
        values = self._on_device("off_focal_fraction", torch.tensor(self.off_focal_fraction, dtype=torch.float64),
                                 u.device, u.dtype)
        n = int(nodes.numel())
        above = torch.searchsorted(nodes, distance, right=True)  # nodes <= distance
        i0 = (above - 1).clamp(0, n - 1)
        i1 = above.clamp(max=n - 1)
        d0, d1 = nodes[i0], nodes[i1]
        span = d1 - d0
        w = torch.where(span > 0, (distance - d0) / torch.where(span > 0, span, torch.ones_like(span)),
                        torch.zeros_like(span)).clamp(0.0, 1.0)
        fraction = values[i0] + (values[i1] - values[i0]) * w
        fraction = torch.where(distance > nodes[-1], torch.zeros_like(fraction), fraction)
        return torch.where(inside, torch.ones_like(fraction), fraction)

    def _weight(self, inside: Tensor, u: Tensor, v: Tensor, geometry, angle: float) -> Tensor:
        """The field weight times the emission profile at panel points.

        Args:
            inside (Tensor): [...] field weight (bool or float).
            u (Tensor): [...] panel coordinates along the channels in cm.
            v (Tensor): [...] panel coordinates along the rows in cm.
            geometry (ConeGeometry or ParallelGeometry): the panel.
            angle (float): view angle in rad.

        Returns:
            Tensor: [...] weights in the dtype of ``u``.
        """
        weight = inside.to(u.dtype)
        profile = self.profile_at(angle, u.device, u.dtype)
        if profile is None:
            return weight
        if tuple(profile.shape) != geometry.image_shape:
            raise ValueError(f"emission_profile is {tuple(profile.shape)} per view, the panel {geometry.image_shape}")
        return weight * panel_interpolate(profile, u, v, geometry)

    def angular_weight(self, directions: Tensor, geometry: ConeGeometry, angle: float) -> Tensor:
        """Angular weight ``b(Omega)`` of the source: the field weight times the emission profile (1 by default).

        Args:
            directions (Tensor): [..., 3] directions in the recon frame, not necessarily normalized.
            geometry (ConeGeometry): scan geometry.
            angle (float): view angle in rad.

        Returns:
            Tensor: [...] weights, on the device and in the dtype of ``directions``, 0 off the field.
        """
        forward, u, v = self._panel_point(directions, geometry, angle)
        field_weight = self._field_weight(u, v, geometry)
        inside = (forward & field_weight) if field_weight.dtype == torch.bool else forward * field_weight
        return self._weight(inside, u, v, geometry, angle)

    def beam_weight(self, points: Tensor, geometry, angle: float) -> Tensor:
        """Weight ``b(r)`` of a parallel beam at points.

        It is the field weight times the emission profile (1 by default) at the panel point of the beam's ray through
        each point: 0 outside the beam's cross-section (the panel's area and the field margins) unless an off-focal
        table extends it.

        Args:
            points (Tensor): [..., 3] points in the recon frame in cm.
            geometry (ParallelGeometry): parallel-beam geometry.
            angle (float): view angle in rad.

        Returns:
            Tensor: [...] weights, on the device and in the dtype of ``points``.
        """
        u, v = geometry.panel_uv_of_points(points, angle)
        return self._weight(self._field_weight(u, v, geometry), u, v, geometry, angle)

    def _filter_path_at(self, u: Tensor, v: Tensor, geometry) -> Tensor:
        """Path through the filter at panel points, ``rho_f t(u, v)`` in g/cm^2.

        Args:
            u (Tensor): [...] panel coordinates along the channels in cm.
            v (Tensor): [...] panel coordinates along the rows in cm.
            geometry (ConeGeometry or ParallelGeometry): the panel.

        Returns:
            Tensor: [...] path in g/cm^2, in the dtype of ``u``.
        """
        thickness = self.thickness_at(geometry, u.device, u.dtype)
        return panel_interpolate(thickness, u, v, geometry) * self.filter_density

    def filter_path(self, directions: Tensor, geometry: ConeGeometry, angle: float) -> Optional[Tensor]:
        """Path ``rho_f t(Omega)`` through the beam filter of the rays from the focal spot, or None without a filter.

        Args:
            directions (Tensor): [..., 3] directions in the recon frame, not necessarily normalized.
            geometry (ConeGeometry): scan geometry.
            angle (float): view angle in rad.

        Returns:
            Tensor or None: [...] path in g/cm^2, on the device and in the dtype of ``directions``.
        """
        if self.filter_material is None:
            return None
        _, u, v = self._panel_point(directions, geometry, angle)
        return self._filter_path_at(u, v, geometry)

    def beam_filter_path(self, points: Tensor, geometry, angle: float) -> Optional[Tensor]:
        """Path through the beam filter of the rays of a parallel beam through points, or None without a filter.

        Args:
            points (Tensor): [..., 3] points in the recon frame in cm.
            geometry (ParallelGeometry): parallel-beam geometry.
            angle (float): view angle in rad.

        Returns:
            Tensor or None: [...] path in g/cm^2, on the device and in the dtype of ``points``.
        """
        if self.filter_material is None:
            return None
        u, v = geometry.panel_uv_of_points(points, angle)
        return self._filter_path_at(u, v, geometry)


# fine mass attenuation of filter materials: id(fine_energy) -> (weak reference to it, {(material, device, dtype): [E]})
_FINE_ATTENUATION: dict[int, tuple[weakref.ref, dict[tuple, Tensor]]] = {}


def _fine_attenuation(energies: Tensor, key: tuple, definition) -> Tensor:
    """Total mass attenuation of a material at fine energies from the photon data, cached per energy tensor.

    Args:
        energies (Tensor): [E] fine energies in keV (``MultigroupLibrary.fine_energy``).
        key (tuple): (name, density, composition, device, dtype) of the material and the result.
        definition (callable): returns the ``Material`` on a cache miss.

    Returns:
        Tensor: [E] mass attenuation in cm^2/g on the device and in the dtype of ``key``.
    """
    hit = _FINE_ATTENUATION.get(id(energies))
    if hit is None or hit[0]() is not energies:
        energies_id = id(energies)
        hit = (weakref.ref(energies, lambda _r, energies_id=energies_id, cache=_FINE_ATTENUATION:
                           cache.pop(energies_id, None)), {})
        _FINE_ATTENUATION[energies_id] = hit
    value = hit[1].get(key)
    if value is None:
        float64 = hit[1].get(key[:3])
        if float64 is None:
            from .detector import material_cross_sections  # the photon data (XCOM)

            values = np.asarray(energies.detach().to("cpu", torch.float64).numpy(), dtype=np.float64)
            float64 = torch.as_tensor(material_cross_sections(definition(), values)["total"], dtype=torch.float64)
            hit[1][key[:3]] = float64
        value = float64.to(device=key[3], dtype=key[4])
        hit[1][key] = value
    return value


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
