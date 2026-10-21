"""Collapsed-cone scatter image of the last-collided source and primary image of one view.

The scatter image traces the attenuated first- and multiple-collision emission to the panel along a cone product set of
directions with a finite-cone footprint per direction (:func:`combined_scatter_image`); :class:`DirectionSetContext`
holds everything of an image that depends only on the view and the direction set. The primary image is exact in
energy and is formed from per-material density line integrals (:func:`primary_path_lengths`,
:func:`primary_from_path_lengths`). The equations are in docs/source/method.rst, "Collapsed-cone scatter image",
"Detector and grid responses" and "The primary"; every function works on one view angle in the recon frame, in the
dtype of its inputs.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence, Union

import torch
from torch import Tensor

from .angular_quadrature import Quadrature, hemisphere_quadrature, real_spherical_harmonics
from .config import SolverConfig
from .cross_sections import MultigroupLibrary, group_transmission, mu_t_volume
from .detector import DetectorResponse, GridResponse, _bracket, detector_angle, grid_angles
from .geometry import ConeGeometry, Grid, device_tensor, resample_conservative
from .kernels import triton_utilities, uncollided_kernels
from .ray_tracing import Projectors
from .source import SourceModel
from .uncollided import first_collision_source_at_directions, present_materials, resolve_pairs

__all__ = [
    "covering_polar_angle",
    "panel_direction_set",
    "DirectionSetContext",
    "make_direction_context",
    "scatter_image_from_source",
    "supervoxel_subpoint_emission",
    "combined_scatter_image",
    "box_blur_rows_cols",
    "panel_height_field",
    "coarsen_mu_t",
    "primary_subpixel_geometry",
    "primary_path_lengths",
    "primary_from_path_lengths",
    "primary_image",
    "scatter_fraction",
]

SourceFunction = Callable[[Tensor], Tensor]
triton = triton_utilities.triton
tl = triton_utilities.tl


# =====================================================================================================
# direction sets
# =====================================================================================================

def covering_polar_angle(grid: Grid, geometry: ConeGeometry, angle: float, margin: float = 1e-6) -> float:
    """Smallest polar angle about the inward panel normal that contains every direction from the grid box to the panel.

    The maximum is attained at a pair of a panel corner and a box corner (on a curved panel, a point of its top or
    bottom edge, sampled four times per channel); the result is pi/2 when a box corner is not strictly on the source
    side of the panel along the normal at its center.

    Args:
        grid (Grid): the grid whose box is covered.
        geometry (ConeGeometry or ParallelGeometry): scan geometry.
        angle (float): view angle in radians.
        margin (float, optional): added to the angle, in radians. Defaults to 1e-6.

    Returns:
        float: cone half-angle in radians, at most pi/2.
    """
    lower = grid.lower_corner()
    extent = grid.extent
    corners = torch.tensor([[lower[0] + i * extent[0], lower[1] + j * extent[1], lower[2] + k * extent[2]]
                            for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=torch.float64)
    u_hat, v_hat, n_hat = (torch.tensor(a, dtype=torch.float64) for a in geometry.panel_axes(angle))
    panel_center = torch.tensor(geometry.panel_center(angle), dtype=torch.float64)
    half_u, half_v = geometry.panel_half_extent()
    center_u, center_v = -geometry.det_channel_offset, -geometry.det_row_offset
    if geometry.use_curved_detector:  # the top and bottom edges of the cylinder, sampled four times per channel
        edge_u = center_u + torch.linspace(-half_u, half_u, 4 * geometry.num_det_channels + 1, dtype=torch.float64)
        edge_v = center_v + torch.tensor([-half_v, half_v], dtype=torch.float64)
        panel_corners = geometry.panel_points(edge_u[:, None], edge_v[None, :], angle).reshape(-1, 3)
    else:
        panel_corners = torch.stack([panel_center + (center_u + su * half_u) * u_hat
                                     + (center_v + sv * half_v) * v_hat for su in (-1, 1) for sv in (-1, 1)])
    d = panel_corners[:, None, :] - corners[None, :, :]  # [P, 8, 3]
    normal_distance = (d * (-n_hat)).sum(-1)
    if bool((normal_distance <= 0).any()):
        return math.pi / 2
    lateral = torch.linalg.norm(d - normal_distance[..., None] * (-n_hat), dim=-1)
    return float(min(math.pi / 2, float(torch.atan2(lateral, normal_distance).max()) + margin))


def panel_direction_set(geometry: ConeGeometry, angle: float,
                        config_or_counts: Union[SolverConfig, tuple[int, int]], *,
                        max_polar_angle: Union[float, Grid] = math.pi / 2,
                        device=None, dtype: torch.dtype = torch.float32) -> Quadrature:
    """Collapsed-cone direction set about the inward panel normal: the directions that reach the panel.

    A Gauss-Legendre (in cos theta) by equal-azimuth product set
    (:func:`~citrus.scatter_solver.angular_quadrature.hemisphere_quadrature`) with the cell widths of the finite-cone
    footprint filled.

    Args:
        geometry (ConeGeometry or ParallelGeometry): scan geometry.
        angle (float): view angle in radians.
        config_or_counts (SolverConfig or tuple): a configuration (``cone_num_polar``, ``cone_num_azimuthal``) or a
            (num_polar, num_azimuthal) tuple.
        max_polar_angle (float or Grid, optional): cone half-angle in radians, or a grid whose box the cone covers
            (:func:`covering_polar_angle`). Defaults to pi/2.
        device (torch.device, optional): device of the set. Defaults to None.
        dtype (torch.dtype, optional): dtype of the set. Defaults to torch.float32.

    Returns:
        Quadrature: omega [K, 3], weight [K] in sr and cell_width [K, 2] (gnomonic widths along u_hat, v_hat).
    """
    if isinstance(config_or_counts, SolverConfig):
        num_polar, num_azimuthal = config_or_counts.cone_num_polar, config_or_counts.cone_num_azimuthal
    else:
        num_polar, num_azimuthal = (int(c) for c in config_or_counts)
    if isinstance(max_polar_angle, Grid):
        max_polar_angle = covering_polar_angle(max_polar_angle, geometry, angle)
    u_hat, v_hat, n_hat = geometry.panel_axes(angle)
    axis = [-c for c in n_hat]
    return hemisphere_quadrature(axis, num_polar, num_azimuthal, max_polar_angle, device=device, dtype=dtype,
                                 tangent_axes=(u_hat, v_hat))


# =====================================================================================================
# per-(view, direction set) precomputation
# =====================================================================================================

@dataclass(eq=False)
class DirectionSetContext:
    """Everything of a collapsed-cone image that depends only on the view, the direction set and the responses.

    Built by :func:`make_direction_context` and passed as ``context`` to the image functions to reuse it across the
    images of one view; direction-only tables (harmonics, sub-cell offsets, footprint heights) are cached in it on first
    use. K is the number of panel-facing directions, G the number of groups and Rd x Cd the detector rows x channels.

    Attributes:
        quadrature (Quadrature): the direction set the context was built from (checked on use).
        geometry (ConeGeometry or ParallelGeometry): the geometry it was built for.
        angle (float): view angle in radians.
        detector (DetectorResponse): detector response.
        grid_response (GridResponse): anti-scatter-grid response.
        group_energy (Tensor): [G] group energies in keV.
        omega (Tensor): [K, 3] panel-facing directions, one contiguous tensor.
        weight (Tensor): [K] quadrature weights in sr.
        cos_theta0 (Tensor): [K] cosine of the angle theta0 between each direction and the inward panel normal (at the
            panel center on a curved panel).
        detector_weight (Tensor): [K, G] detector response ``D_g(theta0_k)`` in keV per incident photon.
        grid_weight (Tensor): [K, G, Rd, 1] grid transmission per detector row when separable, else None.
        signal_weight (Tensor): signal weights ``w_k cos(theta0) D_g G_g``, or the factor of them that depends on the
            direction and the channel: [K, G, Rd, 1] when separable, else [K, G, 1, Cd] (Cd = 1 when the weights do not
            vary along the channels). The full weights are :meth:`pixel_signal_weight`.
        accumulation_weight (Tensor): [Rd G, 3, K] fused weights of :meth:`accumulate` (rows outer) when separable,
            else None.
        separable (bool): the signal weights vary along the detector rows only (flat panel).
        chunk (int): directions per chunk.
        chunk_bounds (tuple): (start, stop) of every chunk.
        omega_chunks (tuple): [k, 3] views of ``omega`` per chunk.
        weight_chunks (tuple): [k] views of ``weight`` per chunk.
        image_shape (tuple): (Rd, Cd).
        cell_width (Tensor): [K, 2] gnomonic cell widths of the directions for the finite-cone footprint, or None.
            Defaults to None.
        planar_weight (Tensor): [K, Cd] weights ``w_k cos(theta0_kc)`` with the incidence angle at each channel of a
            curved panel (zero where the direction meets the panel from behind), else None. Defaults to None.
        signal_group_weight (Tensor): [K, G] signal weights of the directions when they are the same at every pixel (a
            flat panel and a grid response that does not depend on the angles, such as no grid), else None; the signal
            is then the projection of the emission summed over the groups with them.
        signal_row_weight (Tensor): [K, G, Rd, 1] factor of the signal weights that varies along the rows when
            ``signal_weight`` varies along the channels (a flat panel with septa in both directions), else None.
            Defaults to None.
        pixel_grids (tuple): :class:`_PixelGrid` septa whose transmission varies along the rows and the channels
            together (a curved panel with septa along the channels), evaluated per chunk of directions. Defaults to ().
    """

    quadrature: Quadrature
    geometry: ConeGeometry
    angle: float
    detector: DetectorResponse
    grid_response: GridResponse
    group_energy: Tensor
    omega: Tensor
    weight: Tensor
    cos_theta0: Tensor
    detector_weight: Tensor
    grid_weight: Optional[Tensor]
    signal_weight: Tensor
    accumulation_weight: Optional[Tensor]
    separable: bool
    chunk: int
    chunk_bounds: tuple[tuple[int, int], ...]
    omega_chunks: tuple[Tensor, ...]
    weight_chunks: tuple[Tensor, ...]
    image_shape: tuple[int, int]
    cell_width: Optional[Tensor] = None
    planar_weight: Optional[Tensor] = None
    signal_group_weight: Optional[Tensor] = None
    signal_row_weight: Optional[Tensor] = None
    pixel_grids: tuple = ()
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def grid_per_pixel(self) -> bool:
        """True when some septa have a transmission that varies along the rows and the channels together."""
        return bool(self.pixel_grids)

    # ---- sizes ----------------------------------------------------------------------------------
    @property
    def num_directions(self) -> int:
        """Number of panel-facing directions K."""
        return int(self.omega.shape[0])

    @property
    def num_groups(self) -> int:
        """Number of energy groups G."""
        return int(self.group_energy.numel())

    @property
    def device(self) -> torch.device:
        """Device of the context's tensors."""
        return self.omega.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the context's tensors."""
        return self.omega.dtype

    @property
    def nbytes(self) -> int:
        """Bytes held by the context's tensors (distinct storages, lazily cached entries included)."""
        seen: dict[int, int] = {}
        tensors = [t for t in (self.group_energy, self.omega, self.weight, self.cos_theta0, self.detector_weight,
                               self.grid_weight, self.signal_weight, self.accumulation_weight, self.cell_width,
                               self.planar_weight, self.signal_group_weight, self.signal_row_weight)
                   if t is not None]
        for pixel_grid in self.pixel_grids:
            tensors.extend(t for t in (pixel_grid.theta_channels, pixel_grid.theta_pixels, pixel_grid.omega64)
                           if t is not None)
        for v in self._cache.values():
            tensors.extend(t for t in (v if isinstance(v, tuple) else (v,)) if isinstance(t, Tensor))
        for t in tensors:
            storage = t.untyped_storage()
            seen[storage.data_ptr()] = storage.nbytes()
        return int(sum(seen.values()))

    # ---- checks ----------------------------------------------------------------------------------
    def check(self, geometry: ConeGeometry, angle: float, quadrature: Quadrature, detector: DetectorResponse,
              grid_response: GridResponse, num_groups: int, device: torch.device, dtype: torch.dtype) -> None:
        """Check that the context was built for these image arguments (host-side checks only).

        Args:
            geometry (ConeGeometry or ParallelGeometry): scan geometry of the image.
            angle (float): view angle in radians.
            quadrature (Quadrature): direction set; must be the context's own object.
            detector (DetectorResponse): detector response; must be the context's own object.
            grid_response (GridResponse): grid response; must be the context's own object.
            num_groups (int): number of groups G of the image.
            device (torch.device): device of the image.
            dtype (torch.dtype): dtype of the image.

        Raises:
            ValueError: if any argument does not match.
        """
        mismatches = []
        if quadrature is not self.quadrature:
            mismatches.append("direction set (quadrature)")
        if geometry != self.geometry:
            mismatches.append("geometry")
        if abs(float(angle) - self.angle) > 1e-12:
            mismatches.append(f"angle ({angle} vs {self.angle})")
        if detector is not self.detector:
            mismatches.append("detector response")
        if grid_response is not self.grid_response:
            mismatches.append("grid response")
        if int(num_groups) != self.num_groups:
            mismatches.append(f"group count ({num_groups} vs {self.num_groups})")
        if torch.device(device) != self.device:
            mismatches.append(f"device ({device} vs {self.device})")
        if dtype != self.dtype:
            mismatches.append(f"dtype ({dtype} vs {self.dtype})")
        if mismatches:
            raise ValueError("DirectionSetContext does not match the image arguments: " + ", ".join(mismatches))

    # ---- lazily cached direction-only tables --------------------------------------------------------
    def spherical_harmonics(self, L: int) -> tuple[Tensor, ...]:
        """Real spherical harmonics of the context's directions per chunk (cached per order).

        Args:
            L (int): Legendre order.

        Returns:
            tuple: one [k, (L+1)^2] tensor per chunk.
        """
        key = ("spherical_harmonics", int(L))
        cached = self._cache.get(key)
        if cached is None:
            Y = real_spherical_harmonics(int(L), self.omega)  # [K, (L+1)^2]
            cached = (Y,) + tuple(Y[s:e] for s, e in self.chunk_bounds)
            self._cache[key] = cached
        return cached[1:]

    def subcell_offsets(self, grid_quadrature: Grid, q: int) -> tuple[Tensor, ...]:
        """Offsets ``Omega_k . (r_j - r_cell)`` of the q^3 quadrature points of a transport cell, per chunk (cached).

        Args:
            grid_quadrature (Grid): the quadrature grid Q.
            q (int): quadrature points per axis per transport cell.

        Returns:
            tuple: one [k, q, q, q] tensor per chunk in cm (axes rows, cols, slices).
        """
        f = int(q)
        key = ("subcell", tuple(float(d) for d in grid_quadrature.deltas), f)
        cached = self._cache.get(key)
        if cached is None:
            offsets = _subcell_offsets(self.omega, grid_quadrature, f)  # [K, q, q, q]
            cached = (offsets,) + tuple(offsets[s:e] for s, e in self.chunk_bounds)
            self._cache[key] = cached
        return cached[1:]

    # ---- accumulation ------------------------------------------------------------------------------
    def new_accumulator(self) -> Tensor:
        """Zeroed image accumulator for :meth:`accumulate`.

        Returns:
            Tensor: [Rd G, 3, Cd] when separable, else [3, G, Rd, Cd].
        """
        G, (Rd, Cd) = self.num_groups, self.image_shape
        if self.separable:
            return torch.zeros(Rd * G, 3, Cd, device=self.device, dtype=self.dtype)
        return torch.zeros(3, G, Rd, Cd, device=self.device, dtype=self.dtype)

    def accumulate(self, accumulator: Tensor, image: Tensor, chunk_index: int) -> None:
        """Add the weighted image of one direction chunk to the accumulator in place (fluence, planar fluence, signal).

        When separable this is one batched matrix product,
        ``accumulator[(r, g), j, c] += sum_k W[(r, g), j, k] image[k, g, r, c]``, which reads an image stored rows-outer
        in place.

        Args:
            accumulator (Tensor): the accumulator of :meth:`new_accumulator`.
            image (Tensor): [k, G, Rd, Cd] projected attenuated emission of the chunk in photons/(cm^2 sr), at best
                stored rows-outer ([k, Rd, G, Cd] seen as [k, G, Rd, Cd], as :func:`combined_scatter_image` projects
                it).
            chunk_index (int): index of the chunk.
        """
        start, stop = self.chunk_bounds[chunk_index]
        k = stop - start
        G, (Rd, Cd) = self.num_groups, self.image_shape
        if self.separable:
            image_view = image.permute(2, 1, 0, 3).reshape(Rd * G, k, Cd)  # a view for a rows-outer image
            accumulator.baddbmm_(self.accumulation_weight[:, :, start:stop], image_view)
            return
        weights = self.weight_chunks[chunk_index]
        accumulator[0].add_(torch.einsum("kgrc,k->grc", image, weights))
        if self.planar_weight is not None:  # curved panel: the incidence angle varies with the channel
            accumulator[1].add_(torch.einsum("kgrc,kc->grc", image, self.planar_weight[start:stop]))
        else:
            accumulator[1].add_(torch.einsum("kgrc,k->grc", image, weights * self.cos_theta0[start:stop]))
        if self.signal_row_weight is None and not self.pixel_grids:  # [k, G, 1, Cd]: one product
            signal_weight = self.signal_weight[start:stop, :, 0].expand(k, G, Cd)
            accumulator[2].add_(torch.einsum("kgrc,kgc->grc", image, signal_weight))
        else:
            accumulator[2].add_((image * self.pixel_signal_weight(start, stop)).sum(0))

    def pixel_signal_weight(self, start: int, stop: int) -> Tensor:
        """Signal weights of directions [start, stop) at the pixels, the product of the context's factors.

        Args:
            start (int): first direction.
            stop (int): end direction (excluded).

        Returns:
            Tensor: [stop - start, G, Rd, Cd] weights, or a shape that broadcasts to it when they do not vary along
            the rows or the channels.
        """
        weight = self.signal_weight[start:stop]
        if self.signal_row_weight is not None:
            weight = weight * self.signal_row_weight[start:stop]
        for pixel_grid in self.pixel_grids:
            weight = weight * pixel_grid.transmission(start, stop, self.geometry, self.angle)
        return weight

    def finalize(self, accumulator: Tensor) -> dict[str, Tensor]:
        """Image dict from an accumulator.

        Args:
            accumulator (Tensor): the accumulator filled by :meth:`accumulate`.

        Returns:
            dict: fluence, planar_fluence and signal_per_group [G, Rd, Cd] and signal [Rd, Cd].
        """
        G, (Rd, Cd) = self.num_groups, self.image_shape
        if self.separable:
            accumulator = accumulator.reshape(Rd, G, 3, Cd).permute(2, 1, 0, 3)
        fluence, planar, signal_per_group = (accumulator[j].contiguous() for j in range(3))
        return {"fluence": fluence, "planar_fluence": planar, "signal_per_group": signal_per_group,
                "signal": signal_per_group.sum(0)}


def _subcell_offsets(omega: Tensor, grid_quadrature: Grid, f: int) -> Tensor:
    """Offsets ``Omega_k . (r_j - r_cell)`` of the f^3 voxel centers of ``grid_quadrature`` inside one coarsened cell.

    Args:
        omega (Tensor): [K, 3] directions.
        grid_quadrature (Grid): the quadrature grid Q.
        f (int): points per axis per cell of ``grid_quadrature.coarsen(f)``.

    Returns:
        Tensor: [K, f, f, f] offsets in cm (axes rows, cols, slices).
    """
    device, dtype = omega.device, omega.dtype
    dx, dy, dz = grid_quadrature.deltas
    centered_index = torch.arange(f, device=device, dtype=dtype) - (f - 1) / 2.0
    directions = omega.reshape(-1, 3)
    return (directions[:, 1, None, None, None] * (dy * centered_index)[None, :, None, None]
            + directions[:, 0, None, None, None] * (dx * centered_index)[None, None, :, None]
            + directions[:, 2, None, None, None] * (dz * centered_index)[None, None, None, :])


# the panel-facing directions and how the septa vary over the panel, kept with the direction set: quadrature ->
# {(angle, device, dtype, id(geometry), id(grid response), factorize): [kept, dependence]}, dropped with the set
_SELECTIONS: "weakref.WeakKeyDictionary[Quadrature, dict]" = weakref.WeakKeyDictionary()
# the two septa sets of a crossed grid, each without crossed septa: grid response -> (septa, septa)
_SEPTA_SETS: "weakref.WeakKeyDictionary[GridResponse, tuple]" = weakref.WeakKeyDictionary()
_ANGLE_TOLERANCE = 1e-10  # rad; float64 rounding of the grid angles is ~1e-16


@dataclass(eq=False)
class _PixelGrid:
    """Septa whose transmission varies along the detector rows and the channels together, evaluated per direction chunk.

    The grid angles of such septa (a curved panel with septa along the channels) are the sum of a part per direction
    and channel, from the incidence on the local panel axes, and a part per pixel, from the focusing direction of the
    septa; the context keeps both and a chunk adds them and interpolates the table in the dtype of the images. Septa
    whose angles are not of that form keep the directions and the pixels and are evaluated from them in float64.

    Attributes:
        septa (GridResponse): one set of septa, without crossed septa.
        group_energy (Tensor): [G] group energies in keV.
        theta_channels (Tensor): [2, K, 1, Cd] grid angles (theta1, theta2) along the first row, or None.
        theta_pixels (Tensor): [2, 1, Rd, Cd] their change from the first row, the same for every direction, or None.
        omega64 (Tensor): [K, 3] float64 directions when the angles are not such a sum, else None.
        pixels64 (Tensor): [Rd, Cd, 3] float64 pixel centers when the angles are not such a sum, else None.
        theta1_nodes (Tensor): [T1] table angles theta1 when the table does not depend on theta2, else None.
        table_theta1 (Tensor): [T1, G] the table at the group energies on those nodes, else None.
    """

    septa: GridResponse
    group_energy: Tensor
    theta_channels: Optional[Tensor] = None
    theta_pixels: Optional[Tensor] = None
    omega64: Optional[Tensor] = None
    pixels64: Optional[Tensor] = None
    theta1_nodes: Optional[Tensor] = None
    table_theta1: Optional[Tensor] = None

    def transmission(self, start: int, stop: int, geometry: ConeGeometry, angle: float) -> Tensor:
        """Transmission of directions [start, stop) at every pixel.

        Args:
            start (int): first direction.
            stop (int): end direction (excluded).
            geometry (ConeGeometry): the scan geometry.
            angle (float): view angle in radians.

        Returns:
            Tensor: [stop - start, G, Rd, Cd] transmission in the dtype of ``group_energy``.
        """
        if self.table_theta1 is not None:  # a 1-D table in theta1: two gathers of all groups per pixel
            theta1 = self.theta_channels[0, start:stop] + self.theta_pixels[0]  # [k, Rd, Cd]
            i0, i1, w1 = _bracket(self.theta1_nodes, theta1.abs())
            a0 = self.table_theta1[i0]  # [k, Rd, Cd, G]
            return (a0 + (self.table_theta1[i1] - a0) * w1[..., None]).permute(0, 3, 1, 2)
        if self.theta_channels is not None:
            theta = self.theta_channels[:, start:stop] + self.theta_pixels  # [2, k, Rd, Cd]
            return self.septa.at(self.group_energy, theta[0], theta[1]).transpose(0, 1)
        return self.septa.transmission_at(self.group_energy.to(torch.float64), self.omega64[start:stop],
                                          self.pixels64, geometry, angle).transpose(0, 1).to(self.group_energy.dtype)


def _septa_sets(grid_response: GridResponse) -> tuple[GridResponse, ...]:
    """The sets of septa of a grid response, each without crossed septa (cached per response).

    Args:
        grid_response (GridResponse): the grid response.

    Returns:
        tuple: one GridResponse, or two for crossed septa.
    """
    if grid_response.crossed is None:
        return (grid_response,)
    sets = _SEPTA_SETS.get(grid_response)
    if sets is None:
        sets = _SEPTA_SETS[grid_response] = (replace(grid_response, crossed=None), grid_response.crossed)
    return sets


def _septa_dependence(septa: GridResponse, probe: Tensor, pixels: Tensor, geometry: ConeGeometry,
                      angle: float) -> str:
    """How the transmission of one set of septa varies over the panel, from the grid angles of probe directions.

    The angles are tested for the form ``a(direction, channel) + b(row, channel)``, the incidence on the local panel
    axes plus the focusing direction of the septa at the pixel; this needs host synchronizations, so the caller keeps
    the result.

    Args:
        septa (GridResponse): one set of septa, without crossed septa.
        probe (Tensor): [P, 3] float64 directions.
        pixels (Tensor): [Rd, Cd, 3] float64 pixel centers.
        geometry (ConeGeometry): the scan geometry.
        angle (float): view angle in radians.

    Returns:
        str: "none" (a table that does not depend on the angles), "direction", "rows", "channels", "sum" (a part per
        direction and channel plus a part per pixel) or "general".
    """
    if not septa.depends_on_angles:
        return "none"
    theta = torch.stack(grid_angles(probe, pixels, geometry, angle, septa.focal_distance_cm,
                                    septa.septa_along))  # [2, P, Rd, Cd]
    first_row = theta[:, :, :1]  # [2, P, 1, Cd]
    pixel_part = theta - first_row  # the same for every direction when the form holds
    if float((pixel_part - pixel_part[:, :1]).abs().max()) > _ANGLE_TOLERANCE:
        return "general"
    pixel_part = pixel_part[:, :1]
    no_pixel_part = float(pixel_part.abs().max()) <= _ANGLE_TOLERANCE
    rows_only = float((pixel_part - pixel_part[..., :1]).abs().max()) <= _ANGLE_TOLERANCE
    same_channels = float((first_row - first_row[..., :1]).abs().max()) <= _ANGLE_TOLERANCE
    if no_pixel_part:
        return "direction" if same_channels else "channels"
    return "rows" if rows_only and same_channels else "sum"


def make_direction_context(quadrature: Quadrature, geometry: ConeGeometry, angle: float, detector: DetectorResponse,
                           grid_response: GridResponse, library: Optional[MultigroupLibrary] = None,
                           device=None, dtype: Optional[torch.dtype] = None, *,
                           group_energy: Optional[Tensor] = None, chunk: int = 8,
                           factorize: bool = True) -> DirectionSetContext:
    """Precompute the emission-independent part of the collapsed-cone images of one view and direction set.

    Only the panel-facing directions are kept. The signal weights are stored as factors: one per direction (the
    quadrature weight, the incidence and the detector response on a flat panel), one per direction and channel (the
    incidence and the detector response on a curved panel, septa along the rows), one per direction and row (septa
    along the channels on a flat panel) and the septa that vary along both (:class:`_PixelGrid`). How each set of septa
    varies is decided once per direction set and view from probe directions (the only host synchronizations of the
    context). The context does not depend on the voxel grid, so it serves every image of the view.

    Args:
        quadrature (Quadrature): direction set, omega [M, 3] and weight [M].
        geometry (ConeGeometry or ParallelGeometry): scan geometry.
        angle (float): view angle in radians.
        detector (DetectorResponse): detector response.
        grid_response (GridResponse): anti-scatter-grid response.
        library (MultigroupLibrary, optional): source of the group energies when ``group_energy`` is None. Defaults to
            None.
        device (torch.device, optional): device of the images. Defaults to None, the device of ``quadrature``.
        dtype (torch.dtype, optional): dtype of the images. Defaults to None, the dtype of ``quadrature`` (float32 if
            it is not a float type).
        group_energy (Tensor, optional): [G] group energies in keV; takes precedence over ``library``. Defaults to None.
        chunk (int, optional): directions per chunk of the image loop. Defaults to 8.
        factorize (bool, optional): when the signal weights vary along the rows only, store them per detector row
            and accumulate with one batched matrix product, else per-pixel products. Defaults to True.

    Returns:
        DirectionSetContext: the context of the view (:class:`DirectionSetContext` lists the shapes).
    """
    if group_energy is None:
        if library is None:
            raise ValueError("pass library or group_energy")
        group_energy = library.group_energy
    device = torch.device(device) if device is not None else quadrature.omega.device
    if dtype is None:
        dtype = quadrature.omega.dtype if quadrature.omega.dtype.is_floating_point else torch.float32
    float64 = torch.float64
    energy64 = group_energy.detach().to(device, float64).reshape(-1)
    G = int(energy64.numel())
    omega64 = quadrature.omega.detach().to(device, float64)
    weight64 = quadrature.weight.detach().to(device, float64)
    _, _, n_hat = geometry.panel_axes(angle)
    cos64 = (omega64 * (-device_tensor(n_hat, device, float64))).sum(-1)
    # the selection and the dependence of the septa need host synchronizations; they are kept with the direction set
    known = _SELECTIONS.setdefault(quadrature, {})
    selection_key = (float(angle), str(device), dtype, id(geometry), id(grid_response), bool(factorize))
    if selection_key not in known:
        # the same selection as before in the dtype of the images (a direction grazing the panel plane must be kept
        # or dropped exactly as the projectors see it)
        cos_image_dtype = (quadrature.omega.detach().to(device, dtype)
                           * (-device_tensor(n_hat, device, dtype))).sum(-1)
        known[selection_key] = [torch.nonzero(cos_image_dtype > 0).reshape(-1), None]
    selection = known[selection_key]
    kept = selection[0]
    omega64, weight64, cos64 = omega64[kept], weight64[kept], cos64[kept]
    K = int(omega64.shape[0])
    pixels = geometry.pixel_centers(angle, device, float64)  # [Rd, Cd, 3]
    Rd, Cd = int(pixels.shape[0]), int(pixels.shape[1])
    curved = bool(getattr(geometry, "use_curved_detector", False))
    septa_sets = _septa_sets(grid_response)
    if selection[1] is None:
        probe = device_tensor(n_hat, device, float64)[None].neg()
        if K:
            probe = torch.cat([probe, omega64[torch.tensor([0, K // 2, K - 1], device=device)]])
        selection[1] = tuple(_septa_dependence(septa, probe, pixels, geometry, angle) for septa in septa_sets)
    dependence = selection[1]
    # detector response at theta0 to the normal at the panel center -> [G, K]
    detector_response = (detector.at(energy64, detector_angle(omega64, geometry, angle)) if K
                         else torch.zeros(G, 0, device=device, dtype=float64))
    planar64 = None
    if curved:  # the incidence angle varies with the channel
        normals = geometry.local_panel_axes(pixels[0], angle)[2]  # [Cd, 3], the same on every row
        cos_channel = (omega64 @ normals.neg().t()).clamp(0.0, 1.0)  # [K, Cd]; 0 for a panel met from behind
        planar64 = weight64[:, None] * cos_channel
        direction_factor = torch.ones(G, K, device=device, dtype=float64)
        channel_factor = planar64[None] * detector.at(energy64, torch.arccos(cos_channel))  # [G, K, Cd]
    else:
        direction_factor = (weight64 * cos64)[None] * detector_response  # [G, K]
        channel_factor = None
    # the septa: factors per direction, per row, per channel, or per pixel
    grid_direction = torch.ones(G, K, device=device, dtype=float64)
    row_factor = None
    pixel_grids = []
    zero = torch.zeros(1, device=device, dtype=float64)
    for septa, kind in zip(septa_sets, dependence):
        if kind == "none":
            grid_direction = grid_direction * septa.at(energy64, zero, zero)  # [G, 1]
        elif kind == "direction":
            grid_direction = grid_direction * septa.transmission_at(energy64, omega64, pixels[:1, :1], geometry,
                                                                    angle)[:, :, 0, 0]
        elif kind == "rows":
            transmission = septa.transmission_at(energy64, omega64, pixels[:, :1], geometry, angle)[..., 0]
            row_factor = transmission if row_factor is None else row_factor * transmission  # [G, K, Rd]
        elif kind == "channels":
            transmission = septa.transmission_at(energy64, omega64, pixels[:1], geometry, angle)[:, :, 0]
            channel_factor = transmission if channel_factor is None else channel_factor * transmission  # [G, K, Cd]
        elif kind == "sum":
            first_row = torch.stack(grid_angles(omega64, pixels[:1], geometry, angle, septa.focal_distance_cm,
                                                septa.septa_along))  # [2, K, 1, Cd]
            one_direction = torch.stack(grid_angles(omega64[:1], pixels, geometry, angle, septa.focal_distance_cm,
                                                    septa.septa_along))  # [2, 1, Rd, Cd]
            theta1_nodes = table_theta1 = None
            table = septa.transmission
            if bool((table == table[:, :, :1]).all()):  # host tables: no device synchronization
                theta1_nodes = septa.theta1_rad.to(device, dtype)
                table_theta1 = septa.at(energy64, septa.theta1_rad.to(device, float64),
                                        torch.zeros(1, device=device, dtype=float64)).t().to(dtype).contiguous()
            pixel_grids.append(_PixelGrid(septa, energy64.to(dtype), theta_channels=first_row.to(dtype).contiguous(),
                                          theta_pixels=(one_direction - one_direction[:, :, :1]).to(dtype)
                                          .contiguous(), theta1_nodes=theta1_nodes, table_theta1=table_theta1))
        else:
            pixel_grids.append(_PixelGrid(septa, energy64.to(dtype), omega64=omega64, pixels64=pixels))
    separable = bool(factorize and not curved and channel_factor is None and not pixel_grids)
    grid_weight = signal_weight = accumulation_weight = signal_row_weight = signal_group_weight = None
    if separable:
        grid_rows = grid_direction[..., None] * (row_factor if row_factor is not None else 1.0)  # [G, K, Rd or 1]
        grid_rows = grid_rows.expand(G, K, Rd)
        grid_weight = grid_rows.permute(1, 0, 2)[..., None].to(dtype).contiguous()  # [K, G, Rd, 1]
        signal_weight = (direction_factor[..., None] * grid_rows).permute(1, 0, 2)[..., None].to(dtype).contiguous()
        accumulation_weight = torch.empty(Rd, G, 3, K, device=device, dtype=dtype)  # rows outer
        accumulation_weight[:, :, 0] = weight64.to(dtype)
        accumulation_weight[:, :, 1] = (weight64 * cos64).to(dtype)
        accumulation_weight[:, :, 2] = signal_weight[..., 0].permute(2, 1, 0)
        accumulation_weight = accumulation_weight.reshape(Rd * G, 3, K)
    else:
        direction_factor = direction_factor * grid_direction  # [G, K]
        channel_weight = (direction_factor[..., None] if channel_factor is None
                          else direction_factor[..., None] * channel_factor)  # [G, K, Cd or 1]
        signal_weight = channel_weight.permute(1, 0, 2)[:, :, None].to(dtype).contiguous()  # [K, G, 1, Cd or 1]
        if row_factor is not None:
            signal_row_weight = row_factor.permute(1, 0, 2)[..., None].to(dtype).contiguous()  # [K, G, Rd, 1]
    if not curved and channel_factor is None and row_factor is None and not pixel_grids:
        # the same weight at every pixel: the signal can be projected from the emission summed over the groups
        signal_group_weight = (direction_factor * grid_direction if separable
                               else direction_factor).t().to(dtype).contiguous()  # [K, G]
    chunk = max(1, int(chunk))
    chunk_bounds = tuple((start, min(start + chunk, K)) for start in range(0, K, chunk))
    omega = omega64.to(dtype).contiguous()
    weight = weight64.to(dtype).contiguous()
    cell_width = getattr(quadrature, "cell_width", None)
    cell_width = None if cell_width is None else cell_width.detach().to(device, dtype)[kept].contiguous()
    return DirectionSetContext(
        quadrature=quadrature, geometry=geometry, angle=float(angle), detector=detector, grid_response=grid_response,
        group_energy=energy64.to(dtype), omega=omega, weight=weight,
        cos_theta0=cos64.to(dtype), detector_weight=detector_response.t().to(dtype).contiguous(),
        grid_weight=grid_weight, signal_weight=signal_weight, accumulation_weight=accumulation_weight,
        separable=separable, chunk=chunk, chunk_bounds=chunk_bounds,
        omega_chunks=tuple(omega[start:stop] for start, stop in chunk_bounds),
        weight_chunks=tuple(weight[start:stop] for start, stop in chunk_bounds), image_shape=(Rd, Cd),
        cell_width=cell_width, planar_weight=None if planar64 is None else planar64.to(dtype).contiguous(),
        signal_group_weight=signal_group_weight, signal_row_weight=signal_row_weight, pixel_grids=tuple(pixel_grids))


def _resolve_context(context: Optional[DirectionSetContext], quadrature: Quadrature, geometry: ConeGeometry,
                     angle: float, detector: DetectorResponse, grid_response: GridResponse,
                     library: Optional[MultigroupLibrary], mu_t: Tensor, chunk: int,
                     group_energy: Optional[Tensor]) -> DirectionSetContext:
    """Return the given context after checking it against the arguments, or build one for them.

    Args:
        context (DirectionSetContext): the context to check, or None to build one.
        quadrature (Quadrature): direction set.
        geometry (ConeGeometry or ParallelGeometry): scan geometry.
        angle (float): view angle in radians.
        detector (DetectorResponse): detector response.
        grid_response (GridResponse): anti-scatter-grid response.
        library (MultigroupLibrary): source of the group energies, or None.
        mu_t (Tensor): [G, ...] attenuation whose group count, device and dtype the context must match.
        chunk (int): directions per chunk of a new context.
        group_energy (Tensor): [G] group energies in keV, or None.

    Returns:
        DirectionSetContext: the checked or new context.
    """
    G = int(mu_t.shape[0])
    if context is None:
        if group_energy is None and library is None:
            raise ValueError("pass library or group_energy")
        context = make_direction_context(quadrature, geometry, angle, detector, grid_response, library, mu_t.device,
                                         mu_t.dtype, group_energy=group_energy, chunk=chunk)
        if context.num_groups != G:
            raise ValueError(f"group_energy has {context.num_groups} entries, mu_t has {G} groups")
        return context
    context.check(geometry, angle, quadrature, detector, grid_response, G, mu_t.device, mu_t.dtype)
    return context


# =====================================================================================================
# collapsed-cone scatter image
# =====================================================================================================

def scatter_image_from_source(source_function: SourceFunction, grid: Grid, geometry: ConeGeometry, angle: float,
                              quadrature: Quadrature, mu_t: Tensor, projectors: Projectors,
                              detector: DetectorResponse, grid_response: GridResponse,
                              library: Optional[MultigroupLibrary] = None, chunk: int = 8,
                              group_energy: Optional[Tensor] = None,
                              context: Optional[DirectionSetContext] = None, *,
                              footprint: bool = True) -> dict[str, Tensor]:
    """Collapsed-cone image of an arbitrary last-collided emission on one grid, for tests and diagnostics.

    The emission is constant per voxel and attenuated from the voxel center to the panel (docs/source/method.rst,
    "Collapsed-cone scatter image").

    Args:
        source_function (callable): maps directions [k, 3] to the emission density [k, G, rows, cols, slices] at the
            voxel centers of ``grid``, in photons/(cm^3 sr) per source photon per sr.
        grid (Grid): grid of the emission and of ``mu_t``.
        geometry (ConeGeometry or ParallelGeometry): scan geometry.
        angle (float): view angle in radians.
        quadrature (Quadrature): direction set (e.g. :func:`panel_direction_set`); directions away from the panel are
            skipped.
        mu_t (Tensor): [G, rows, cols, slices] total attenuation coefficient mu_t in cm^-1 on ``grid``.
        projectors (Projectors): ray tracers.
        detector (DetectorResponse): detector response.
        grid_response (GridResponse): anti-scatter-grid response.
        library (MultigroupLibrary, optional): source of the group energies when ``group_energy`` is None. Defaults to
            None.
        chunk (int, optional): directions per batch; a given ``context`` sets its own. Defaults to 8.
        group_energy (Tensor, optional): [G] group energies in keV. Defaults to None, ``library.group_energy``.
        context (DirectionSetContext, optional): direction context of the view, checked against the arguments.
            Defaults to None, built here.
        footprint (bool, optional): apply the finite-cone footprint (needs ``quadrature.cell_width``); False projects
            one ray per pixel and direction. Defaults to True.

    Returns:
        dict: fluence and planar_fluence [G, Rd, Cd] in photons/cm^2, signal_per_group [G, Rd, Cd] in keV/cm^2
            and signal [Rd, Cd], per source photon per sr.

    Raises:
        ValueError: if ``context`` does not match the arguments.
    """
    context = _resolve_context(context, quadrature, geometry, angle, detector, grid_response, library, mu_t, chunk,
                               group_energy)
    G = mu_t.shape[0]
    device, dtype = mu_t.device, mu_t.dtype
    footprint_blur = _Footprint(grid, geometry, angle, context) if footprint else None
    accumulator = context.new_accumulator()
    for i, directions in enumerate(context.omega_chunks):
        k = directions.shape[0]
        source = source_function(directions)
        if tuple(source.shape) != (k, G, *grid.shape):
            raise ValueError(f"source_function returned {tuple(source.shape)}, expected {(k, G, *grid.shape)}")
        tau = projectors.downstream_cumulative(mu_t, grid, geometry, angle, directions)  # [k, G, R, C, S]
        emission = tau.neg_().exp_().mul_(source.to(device, dtype))  # attenuated to the panel
        del source, tau
        # [k, G, Rd, Cd] photons/(cm^2 sr)
        image = projectors.oblique_project(emission, grid, geometry, angle, directions)
        if footprint_blur is not None:
            image = footprint_blur.blur(image, i, footprint_blur.mean_height(emission))
        del emission
        context.accumulate(accumulator, image, i)
        del image
    return context.finalize(accumulator)


# =====================================================================================================
# quadrature points of the transport grid
# =====================================================================================================

def _check_subpoint_shapes(source_quadrature: Tensor, tau_coarse: Tensor, grid_quadrature: Grid,
                           f: int) -> tuple[int, int, int]:
    """Check the shapes of a quadrature-point source and a transport-grid optical depth.

    Args:
        source_quadrature (Tensor): [k, G, Rq, Cq, Sq] source on ``grid_quadrature``.
        tau_coarse (Tensor): [k, G, Rc, Cc, Sc] optical depth on the transport grid.
        grid_quadrature (Grid): the quadrature grid Q.
        f (int): quadrature points per axis per transport cell.

    Returns:
        tuple: (Rc, Cc, Sc), the shape of the transport grid.
    """
    coarse = grid_quadrature.coarsen(f)
    k, G = source_quadrature.shape[:2]
    Rc, Cc, Sc = coarse.shape
    if tuple(source_quadrature.shape[2:]) != tuple(grid_quadrature.shape):
        raise ValueError(f"source_quadrature spatial shape {tuple(source_quadrature.shape[2:])} != quadrature grid "
                         f"{grid_quadrature.shape}")
    if tuple(tau_coarse.shape) != (k, G, Rc, Cc, Sc):
        raise ValueError(f"tau_coarse {tuple(tau_coarse.shape)} != {(k, G, Rc, Cc, Sc)}")
    return Rc, Cc, Sc


def supervoxel_subpoint_emission(source_quadrature: Tensor, tau_coarse: Tensor, mu_t_coarse: Tensor, omegas: Tensor,
                                 grid_quadrature: Grid, q: int, *, offsets: Optional[Tensor] = None) -> Tensor:
    """Attenuated emission ``e_j = S_j exp(-tau_j)`` at every quadrature point, kept on the quadrature grid.

    ``tau_j = max(0, tau_cell - mu_t_cell (Omega_k . (r_j - r_cell)))`` is the optical depth of the transport-cell
    center corrected to first order inside the cell; for q = 1 the result is ``S exp(-tau)``.

    Args:
        source_quadrature (Tensor): [k, G, Rq, Cq, Sq] emission density on ``grid_quadrature`` in photons/(cm^3 sr).
        tau_coarse (Tensor): [k, G, Rc, Cc, Sc] optical depth to the panel from the transport-cell centers (+inf
            allowed).
        mu_t_coarse (Tensor): [G, Rc, Cc, Sc] total attenuation coefficient mu_t in cm^-1 on the transport grid.
        omegas (Tensor): [k, 3] directions.
        grid_quadrature (Grid): the quadrature grid Q.
        q (int): quadrature points per axis per transport cell.
        offsets (Tensor, optional): [k, q, q, q] precomputed ``Omega_k . (r_j - r_cell)``
            (:meth:`DirectionSetContext.subcell_offsets`). Defaults to None, computed here.

    Returns:
        Tensor: [k, G, Rq, Cq, Sq] attenuated emission in photons/(cm^3 sr), on the device and dtype of ``tau_coarse``.

    Raises:
        ValueError: if a shape does not match.
    """
    f = int(q)
    k, G = source_quadrature.shape[:2]
    Rc, Cc, Sc = _check_subpoint_shapes(source_quadrature, tau_coarse, grid_quadrature, f)
    device, dtype = tau_coarse.device, tau_coarse.dtype
    source_quadrature = source_quadrature.to(device, dtype)
    if f == 1:
        return torch.exp(-tau_coarse).mul_(source_quadrature)
    source_cells = source_quadrature.reshape(k, G, Rc, f, Cc, f, Sc, f)
    # offsets of the quadrature points from the cell center along x (cols), y (rows), z (slices)
    if offsets is None:
        # [k, q_r, q_c, q_s]
        point_offsets = _subcell_offsets(omegas.to(device, dtype).reshape(k, 3), grid_quadrature, f)
    else:
        if tuple(offsets.shape) != (k, f, f, f):
            raise ValueError(f"offsets {tuple(offsets.shape)} != {(k, f, f, f)}")
        point_offsets = offsets.to(device, dtype)
    point_offsets = point_offsets[:, None, None, :, None, :, None, :]  # [k, 1, 1, f, 1, f, 1, f]
    mu_cells = mu_t_coarse.to(device, dtype)[None, :, :, None, :, None, :, None]  # [1, G, Rc, 1, Cc, 1, Sc, 1]
    # tau_j = max(0, tau - mu (Omega . (r_j - r_c)))  [k, G, Rc, f, Cc, f, Sc, f]
    tau_points = torch.addcmul(tau_coarse[:, :, :, None, :, None, :, None], mu_cells, point_offsets,
                               value=-1.0).clamp_(min=0.0)
    return tau_points.neg_().exp_().mul_(source_cells).reshape(source_quadrature.shape)


def coarsen_mu_t(mu_t_fine: Tensor, grid_fine: Grid, grid_coarse: Grid) -> Tensor:
    """Average an attenuation map over the cells of a coarser grid by conservative overlap resampling.

    Zero outside the fine box; equals the attenuation of the resampled densities, since mu_t is linear in them.

    Args:
        mu_t_fine (Tensor): [G, R, C, S] total attenuation coefficient mu_t in cm^-1 on ``grid_fine``.
        grid_fine (Grid): the grid of ``mu_t_fine``.
        grid_coarse (Grid): the target grid.

    Returns:
        Tensor: [G, Rc, Cc, Sc] cell-averaged attenuation in cm^-1 on ``grid_coarse``.
    """
    return resample_conservative(mu_t_fine, grid_fine, grid_coarse)


# =====================================================================================================
# finite-cone footprint
# =====================================================================================================

def box_blur_rows_cols(image: Tensor, width_channels: Tensor, width_rows: Tensor) -> Tensor:
    """Separable box blur of each image with its own fractional width in pixels along the channels and the rows.

    The window is centred on each pixel, clipped to the panel and renormalized there; a width of at most one pixel
    leaves the image unchanged. Images stored rows-outer ([k, Rd, G, Cd] seen as [k, G, Rd, Cd]) are blurred along
    the rows with one product per direction and returned in the same layout.

    Args:
        image (Tensor): [k, G, Rd, Cd] images.
        width_channels (Tensor): [k] widths along the channels in pixels.
        width_rows (Tensor): [k] widths along the rows in pixels.

    Returns:
        Tensor: [k, G, Rd, Cd] blurred images.
    """
    k, G, Rd, Cd = image.shape
    rows_outer = image.permute(0, 2, 1, 3)
    if G > 1 and rows_outer.is_contiguous():
        channel_matrix = box_matrix(Cd, width_channels, image.dtype)  # [k, Cd, Cd]
        row_matrix = box_matrix(Rd, width_rows, image.dtype)  # [k, Rd, Rd]
        blurred = torch.bmm(rows_outer.view(k, Rd * G, Cd), channel_matrix.transpose(1, 2))
        return torch.bmm(row_matrix, blurred.view(k, Rd, G * Cd)).view(k, Rd, G, Cd).permute(0, 2, 1, 3)
    return box_blur_rows(box_blur_channels(image, width_channels), width_rows)


def box_blur_channels(image: Tensor, width_channels: Tensor) -> Tensor:
    """The box blur of :func:`box_blur_rows_cols` along the channels only.

    Args:
        image (Tensor): [k, G, Rd, Cd] images.
        width_channels (Tensor): [k] widths along the channels in pixels.

    Returns:
        Tensor: [k, G, Rd, Cd] images blurred along the channels.
    """
    k, G, Rd, Cd = image.shape
    channel_matrix = box_matrix(Cd, width_channels, image.dtype)  # [k, Cd, Cd]
    return torch.bmm(image.reshape(k, G * Rd, Cd), channel_matrix.transpose(1, 2)).view(k, G, Rd, Cd)


def box_blur_rows(image: Tensor, width_rows: Tensor) -> Tensor:
    """The box blur of :func:`box_blur_rows_cols` along the rows only.

    Args:
        image (Tensor): [k, G, Rd, Cd] images.
        width_rows (Tensor): [k] widths along the rows in pixels.

    Returns:
        Tensor: [k, G, Rd, Cd] images blurred along the rows.
    """
    Rd = int(image.shape[2])
    row_matrix = box_matrix(Rd, width_rows, image.dtype)  # [k, Rd, Rd]
    return torch.matmul(row_matrix[:, None], image)


def box_matrix(N: int, w: Tensor, dtype: Optional[torch.dtype] = None) -> Tensor:
    """Overlap matrices of the one-dimensional box filter of :func:`box_blur_rows_cols`.

    ``B[c, j] = |[j, j + 1] n W_c| / |W_c|`` with ``W_c = [c + 1/2 - w/2, c + 1/2 + w/2] n [0, N]``; one Triton launch
    on float32 CUDA.

    Args:
        N (int): number of pixels.
        w (Tensor): [k] widths in pixels, clamped below at 1e-3.
        dtype (torch.dtype, optional): dtype of the result. Defaults to None, the dtype of ``w``.

    Returns:
        Tensor: [k, N, N] filter matrices.
    """
    dtype = dtype if dtype is not None else w.dtype
    if dtype == torch.float32 and w.is_cuda and triton_utilities.triton_available()[0]:
        widths = w.to(torch.float32).contiguous()
        k = int(widths.numel())
        out = torch.empty(k, N, N, device=w.device, dtype=torch.float32)
        if k:
            triton_utilities.launch(_box_matrix_kernel, (k, int(N), 1), (widths, out, int(N)),
                                    {"BLOCK": max(16, triton.next_power_of_2(int(N)))}, 4)
        return out
    centers = torch.arange(N, device=w.device, dtype=dtype) + 0.5
    half = 0.5 * w.to(dtype).clamp_min(1e-3)[:, None]
    lo = (centers[None] - half).clamp(0.0, float(N))[..., None]  # [k, N, 1]
    hi = (centers[None] + half).clamp(0.0, float(N))[..., None]
    j = torch.arange(N, device=w.device, dtype=dtype)[None, None, :]
    overlap = (torch.minimum(hi, j + 1.0) - torch.maximum(lo, j)).clamp_min(0.0)
    return overlap / (hi - lo)


if triton is not None:

    @triton.jit(do_not_specialize=["N"], do_not_specialize_on_alignment=["w_ptr", "out_ptr"])
    def _box_matrix_kernel(w_ptr, out_ptr, N, BLOCK: tl.constexpr):
        """Row ``c = program_id(1)`` of box matrix ``k = program_id(0)`` (:func:`box_matrix`).

        Args:
            w_ptr: [k] float32 widths in pixels.
            out_ptr: [k, N, N] float32 output matrices.
            N: number of pixels.
            BLOCK (constexpr): lanes per program, a power of two at least N.
        """
        k = tl.program_id(0)
        c = tl.program_id(1)
        w = tl.load(w_ptr + k)
        half = 0.5 * tl.maximum(w, 1e-3)
        nf = N * 1.0
        center = c * 1.0 + 0.5
        lo = tl.minimum(tl.maximum(center - half, 0.0), nf)
        hi = tl.minimum(tl.maximum(center + half, 0.0), nf)
        j = tl.arange(0, BLOCK)
        jf = j.to(tl.float32)
        ov = tl.maximum(tl.minimum(hi, jf + 1.0) - tl.maximum(lo, jf), 0.0) / (hi - lo)
        tl.store(out_ptr + (k * N + c).to(tl.int64) * N + j, ov, mask=j < N)


def panel_height_field(grid: Grid, geometry: ConeGeometry, angle: float, device=None,
                       dtype: torch.dtype = torch.float32) -> Tensor:
    """Normal distance ``h(r) = (r - panel_center) . n_hat`` of every voxel center from the panel plane.

    A direction cell of gnomonic widths (W_alpha, W_beta) seen from height h covers an h W_alpha x h W_beta rectangle of
    the panel.

    Args:
        grid (Grid): the voxel grid.
        geometry (ConeGeometry or ParallelGeometry): scan geometry.
        angle (float): view angle in radians.
        device (torch.device, optional): device of the result. Defaults to None.
        dtype (torch.dtype, optional): dtype of the result. Defaults to torch.float32.

    Returns:
        Tensor: [R, C, S] heights in cm, positive on the source side.
    """
    _, _, n_hat = geometry.panel_axes(angle)
    r = grid.center_coordinates(device, torch.float64)
    panel_center = device_tensor(geometry.panel_center(angle), device, torch.float64)
    return ((r - panel_center) @ device_tensor(n_hat, device, torch.float64)).to(dtype)


class _Footprint:
    """Finite-cone footprint of the direction chunks of one image (docs/source/method.rst, "Collapsed-cone scatter
    image").

    The single-ray image of direction k is box-blurred with the panel footprint ``hbar_k (W_alpha_k, W_beta_k)`` of
    its solid-angle cell, ``hbar_k`` the emission-weighted mean height above the panel.

    Attributes:
        context (DirectionSetContext): direction context of the view.
        scale (Tensor): [2] inverse pixel pitches (channels, rows) in cm^-1.
        heights (Tensor): [voxels] panel heights of the voxel centers of the grid in cm.
        center_height (float): panel height of the center of the grid box in cm.
    """

    def __init__(self, grid: Grid, geometry: ConeGeometry, angle: float, context: DirectionSetContext) -> None:
        """Look up or compute the footprint tables of a grid in the context's cache.

        Args:
            grid (Grid): grid of the emission that is projected.
            geometry (ConeGeometry or ParallelGeometry): scan geometry of the context.
            angle (float): view angle of the context in radians.
            context (DirectionSetContext): direction context with cell widths.

        Raises:
            ValueError: if ``context.cell_width`` is None.
        """
        if context.cell_width is None:
            raise ValueError("the finite-cone footprint needs quadrature.cell_width (panel_direction_set fills it)")
        self.context = context
        key = ("footprint", grid)
        # the tables depend on the grid only (geometry and angle are those of the context)
        cached = context._cache.get(key)
        if cached is None:
            device, dtype = context.device, context.dtype
            scale = device_tensor([1.0 / geometry.delta_det_channel, 1.0 / geometry.delta_det_row], device, dtype)
            heights = panel_height_field(grid, geometry, angle, device, dtype).reshape(-1)  # [voxels]
            # height of the box centre from its corners (host arithmetic)
            _, _, n_hat = geometry.panel_axes(angle)
            panel_center = geometry.panel_center(angle)
            lower, extent = grid.lower_corner(), grid.extent
            corner_heights = [sum((lower[a] + c[a] * extent[a] - panel_center[a]) * n_hat[a] for a in range(3))
                              for c in [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]]
            cached = (scale, heights, 0.5 * (min(corner_heights) + max(corner_heights)))
            context._cache[key] = cached
        self.scale, self.heights, self.center_height = cached

    def mean_height_from_sums(self, sums: Tensor) -> Tensor:
        """Mean panel height per direction from the sums of the emission and of the emission times the height.

        Args:
            sums (Tensor): [k, 2] sums ``(sum e, sum e h)`` over the voxels and groups (the fused emission kernel).

        Returns:
            Tensor: [k] mean heights in cm (the box-center height where the emission vanishes).
        """
        numerator, denominator = sums[:, 1], sums[:, 0]
        tiny = torch.finfo(denominator.dtype).tiny
        return torch.where(denominator > 0, numerator / denominator.clamp_min(tiny),
                           torch.full_like(denominator, self.center_height))

    def mean_height(self, emission: Tensor) -> Tensor:
        """Emission-weighted mean panel height of an attenuated emission per direction.

        Args:
            emission (Tensor): [k, G, rows, cols, slices] attenuated emission on the grid.

        Returns:
            Tensor: [k] mean heights in cm (the box-center height where the emission vanishes).
        """
        k = int(emission.shape[0])
        total = emission.sum(dim=1).reshape(k, -1)  # [k, voxels]
        numerator, denominator = total @ self.heights, total.sum(-1)
        tiny = torch.finfo(denominator.dtype).tiny
        return torch.where(denominator > 0, numerator / denominator.clamp_min(tiny),
                           torch.full_like(denominator, self.center_height))

    def blur(self, image: Tensor, chunk_index: int, mean_heights: Tensor) -> Tensor:
        """Apply the footprint blur to the single-ray images of one direction chunk.

        Args:
            image (Tensor): [k, G, Rd, Cd] single-ray images.
            chunk_index (int): index of the chunk.
            mean_heights (Tensor): [k] mean panel heights in cm (:meth:`mean_height`).

        Returns:
            Tensor: [k, G, Rd, Cd] footprint images.
        """
        widths = self.widths(chunk_index, mean_heights)
        return box_blur_rows_cols(image, widths[:, 0], widths[:, 1])

    def widths(self, chunk_index: int, mean_heights: Tensor) -> Tensor:
        """Box widths of the footprint of the directions of one chunk.

        Args:
            chunk_index (int): index of the chunk.
            mean_heights (Tensor): [k] mean panel heights in cm.

        Returns:
            Tensor: [k, 2] widths in pixels along the channels and the rows.
        """
        start, stop = self.context.chunk_bounds[chunk_index]
        return mean_heights[:, None] * self.context.cell_width[start:stop] * self.scale


# =====================================================================================================
# the scatter image: first-order + multiple-collision emission, traced together
# =====================================================================================================

def combined_scatter_image(library: MultigroupLibrary, rho_quadrature: Tensor, phi_uncollided: Tensor,
                           grid_quadrature: Grid, q: int, scatter_source_moments: Tensor, L: int,
                           geometry: ConeGeometry, angle: float, quadrature: Quadrature,
                           mu_t_coarse: Optional[Tensor], projectors: Projectors, detector: DetectorResponse,
                           grid_response: GridResponse, context: Optional[DirectionSetContext] = None,
                           chunk: int = 8, *, first_collision_chunk: int = 4, footprint: bool = True,
                           pairs=None, grid_coarse: Optional[Grid] = None,
                           signal_only: bool = False) -> dict[str, Tensor]:
    """Total scatter image (first order plus multiple collision) of one view in one collapsed-cone pass.

    Per direction chunk the first-collision source at the quadrature points and the multiple-collision source of the
    transport cells are summed, attenuated (:func:`supervoxel_subpoint_emission`), projected from the quadrature points,
    footprint-blurred and accumulated (docs/source/method.rst, "Collapsed-cone scatter image"). On float32 CUDA tensors
    one fused kernel per chunk forms the attenuated emission and the sums of its mean height
    (``uncollided_kernels.attenuated_emission``). With ``signal_only`` the groups are summed with the signal weights
    in that kernel when the weights are the same at every pixel, between the blurs along the rows and the channels
    when they vary along the rows or the channels only, or after the blur with the weights of each pixel, which gives
    the signal of the full image to float rounding (docs/source/design_notes.rst, "The signal alone").

    Args:
        library (MultigroupLibrary): cross-section library.
        rho_quadrature (Tensor): [M, Rq, Cq, Sq] densities in g/cm^3 on the quadrature grid.
        phi_uncollided (Tensor): [G, Rq, Cq, Sq] uncollided fluence at the quadrature points.
        grid_quadrature (Grid): the quadrature grid Q, q^3 equal-weight points per transport cell.
        q (int): quadrature points per axis per transport cell.
        scatter_source_moments (Tensor): [G, (L+1)^2, Rc, Cc, Sc] cell-mean moments T of the multiple-collision source,
            or [G, (L+1)^2, 4, Rc, Cc, Sc] whose cell means are used.
        L (int): Legendre order of the moments.
        geometry (ConeGeometry or ParallelGeometry): scan geometry.
        angle (float): view angle in radians.
        quadrature (Quadrature): collapsed-cone direction set.
        mu_t_coarse (Tensor): [G, Rc, Cc, Sc] total attenuation coefficient mu_t in cm^-1 on the transport grid, or
            None for the cell average of the attenuation of ``rho_quadrature``.
        projectors (Projectors): ray tracers.
        detector (DetectorResponse): detector response.
        grid_response (GridResponse): anti-scatter-grid response.
        context (DirectionSetContext, optional): direction context of the view; its chunking replaces ``chunk``.
            Defaults to None, built here.
        chunk (int, optional): directions per chunk. Defaults to 8.
        first_collision_chunk (int, optional): directions per inner batch of the first-collision source. Defaults to 4.
        footprint (bool, optional): apply the finite-cone footprint. Defaults to True.
        pairs (MaterialPairs, optional): material representation of ``rho_quadrature`` (or ``uncollided.DENSE``).
            Defaults to None, decided here.
        grid_coarse (Grid, optional): the transport grid C, with ``grid_quadrature = C.refine(q)``. Defaults to None,
            ``grid_quadrature.coarsen(q)``.
        signal_only (bool, optional): return the detector signal only. Defaults to False.

    Returns:
        dict: fluence, planar_fluence and signal_per_group [G, Rd, Cd] and signal [Rd, Cd] of the total scatter, or
        the signal alone with ``signal_only``.

    Raises:
        ValueError: if a shape or the direction context does not match.
    """
    device, dtype = phi_uncollided.device, phi_uncollided.dtype
    rho = rho_quadrature.to(device=device, dtype=dtype)
    pairs = resolve_pairs(rho, pairs)
    f = int(q)
    grid_coarse = grid_quadrature.coarsen(f) if grid_coarse is None else grid_coarse
    if (tuple(grid_coarse.shape) != tuple(s // f for s in grid_quadrature.shape)
            or any(s % f for s in grid_quadrature.shape)):
        raise ValueError(f"grid_coarse {grid_coarse.shape} is not grid_quadrature {grid_quadrature.shape} coarsened by "
                         f"q = {f}")
    mu_t_coarse = (coarsen_mu_t(mu_t_volume(library, rho), grid_quadrature, grid_coarse) if mu_t_coarse is None
                   else mu_t_coarse.to(device=device, dtype=dtype))
    G = int(mu_t_coarse.shape[0])
    Rc, Cc, Sc = grid_coarse.shape
    if tuple(mu_t_coarse.shape[1:]) != tuple(grid_coarse.shape):
        raise ValueError(f"mu_t_coarse spatial shape {tuple(mu_t_coarse.shape[1:])} != coarse grid {grid_coarse.shape}")
    if scatter_source_moments.dim() == 6:
        scatter_source_moments = scatter_source_moments[:, :, 0]
    if (scatter_source_moments.dim() != 5 or tuple(scatter_source_moments.shape[2:]) != tuple(grid_coarse.shape)
            or int(scatter_source_moments.shape[0]) != G):
        raise ValueError(f"scatter_source_moments must be [G={G}, N_sh, {Rc}, {Cc}, {Sc}], got "
                         f"{tuple(scatter_source_moments.shape)}")
    if scatter_source_moments.shape[1] != (L + 1) ** 2:
        raise ValueError(f"scatter_source_moments has {scatter_source_moments.shape[1]} moments, expected (L+1)^2 = "
                         f"{(L + 1) ** 2}")
    context = _resolve_context(context, quadrature, geometry, angle, detector, grid_response, library, mu_t_coarse,
                               chunk, None)
    subcell_offsets = context.subcell_offsets(grid_quadrature, f) if f > 1 else None
    harmonics = context.spherical_harmonics(L)  # per chunk [k, N_sh]
    num_moments = int(scatter_source_moments.shape[1])
    # [N_sh, G * cells], once per image
    moments_by_cell = scatter_source_moments.to(device=device, dtype=dtype).transpose(0, 1).reshape(num_moments, -1)
    quadrature_shape = tuple(grid_quadrature.shape)
    footprint_blur = _Footprint(grid_quadrature, geometry, angle, context) if footprint else None
    # one fused kernel per chunk on float32 CUDA: source, attenuation and mean-height sums without temporaries
    compact_pairs = pairs if not isinstance(pairs, str) else None
    fused = uncollided_kernels.fused_kernels_usable(library, rho, phi_uncollided, compact_pairs)
    # the signal alone: from the emission summed over the groups with the signal weights of each direction, from the
    # groups summed with the weights of each detector row between the blur of the rows and of the channels (or of each
    # channel between the blur of the channels and of the rows), or from the blurred images weighted per pixel
    group_weight = context.signal_group_weight if signal_only and fused else None
    factored = signal_only and group_weight is None and not context.pixel_grids
    row_weight = context.signal_weight if factored and context.separable else None  # [K, G, Rd, 1]
    channel_weight = context.signal_weight if factored and not context.separable else None  # [K, G, 1, Cd or 1]
    accumulator = (torch.zeros(context.image_shape, device=device, dtype=dtype) if signal_only
                   else context.new_accumulator())
    for i, directions in enumerate(context.omega_chunks):
        k = int(directions.shape[0])
        start, stop = context.chunk_bounds[i]
        # [k, G, Rc, Cc, Sc]
        tau = projectors.downstream_cumulative(mu_t_coarse, grid_coarse, geometry, angle, directions)
        # S_MC(Omega_k) per supervoxel
        multiple_collision = torch.mm(harmonics[i], moments_by_cell).view(k, G, Rc, Cc, Sc)
        mean_heights = None
        if fused:
            emission, height_sums = uncollided_kernels.attenuated_emission(
                library, rho, phi_uncollided, grid_quadrature, geometry, angle, directions, tau, mu_t_coarse,
                multiple_collision, f, None if footprint_blur is None else footprint_blur.heights,
                pairs=compact_pairs, signal_weight=None if group_weight is None else group_weight[start:stop])
            if group_weight is not None:
                emission = emission[:, None]  # one group: the signal's emission
            if footprint_blur is not None:
                mean_heights = footprint_blur.mean_height_from_sums(height_sums)
            del multiple_collision, tau, height_sums
        else:
            first_collision = first_collision_source_at_directions(library, rho, phi_uncollided, grid_quadrature,
                                                                   geometry, angle, directions,
                                                                   chunk=first_collision_chunk, pairs=pairs)
            if tuple(first_collision.shape) != (k, G, *quadrature_shape):
                raise ValueError(f"first-collision source {tuple(first_collision.shape)} != "
                                 f"{(k, G, *quadrature_shape)}")
            # one source at the quadrature points, one attenuation, one projection
            total_source = (first_collision.reshape(k, G, Rc, f, Cc, f, Sc, f)
                            .add(multiple_collision[:, :, :, None, :, None, :, None]).view(k, G, *quadrature_shape))
            del first_collision, multiple_collision
            emission = supervoxel_subpoint_emission(total_source, tau, mu_t_coarse, directions, grid_quadrature, f,
                                                    offsets=None if subcell_offsets is None else subcell_offsets[i])
            del total_source, tau
            if footprint_blur is not None:
                mean_heights = footprint_blur.mean_height(emission)
        # [k, G, Rd, Cd] photons/(cm^2 sr), stored rows-outer: the blur along the rows is then one product per
        # direction, and the accumulation reads the images in place
        Rd, Cd = context.image_shape
        rows_outer = torch.empty(k, Rd, int(emission.shape[1]), Cd, device=device, dtype=dtype)
        image = projectors.oblique_project(emission, grid_quadrature, geometry, angle, directions,
                                           out=rows_outer.permute(0, 2, 1, 3))
        del emission
        if row_weight is not None:
            widths = None if footprint_blur is None else footprint_blur.widths(i, mean_heights)
            if widths is not None:
                rows_outer = torch.bmm(box_matrix(Rd, widths[:, 1], dtype), rows_outer.view(k, Rd, G * Cd))
            weights = row_weight[start:stop, :, :, 0].transpose(1, 2)  # [k, Rd, G]
            image = torch.bmm(weights.reshape(k * Rd, 1, G), rows_outer.view(k * Rd, G, Cd)).view(k, 1, Rd, Cd)
            if widths is not None:
                image = box_blur_channels(image, widths[:, 0])
            accumulator.add_(image.sum(dim=(0, 1)))
        elif channel_weight is not None:
            widths = None if footprint_blur is None else footprint_blur.widths(i, mean_heights)
            weights = channel_weight[start:stop, :, 0].expand(k, G, Cd)  # [k, G, Cd]
            row_factor = context.signal_row_weight
            if row_factor is not None:  # the rows blurred and weighted per group, then the channels
                if widths is not None:
                    rows_outer = torch.bmm(box_matrix(Rd, widths[:, 1], dtype),
                                           rows_outer.view(k, Rd, G * Cd)).view(k, Rd, G, Cd)
                rows_outer = rows_outer * row_factor[start:stop].permute(0, 2, 1, 3)  # [k, Rd, G, 1]
                if widths is not None:
                    rows_outer = torch.bmm(rows_outer.reshape(k, Rd * G, Cd),
                                           box_matrix(Cd, widths[:, 0], dtype).transpose(1, 2)).view(k, Rd, G, Cd)
                accumulator.add_(torch.einsum("krgc,kgc->rc", rows_outer, weights))
            else:  # the channels blurred per group, the groups summed per channel, then the rows of one image
                if widths is not None:
                    rows_outer = torch.bmm(rows_outer.view(k, Rd * G, Cd),
                                           box_matrix(Cd, widths[:, 0], dtype).transpose(1, 2)).view(k, Rd, G, Cd)
                single = torch.einsum("krgc,kgc->krc", rows_outer, weights)  # [k, Rd, Cd]
                if widths is not None:
                    single = torch.bmm(box_matrix(Rd, widths[:, 1], dtype), single)
                accumulator.add_(single.sum(0))
        else:
            if footprint_blur is not None:
                image = footprint_blur.blur(image, i, mean_heights)
            if group_weight is not None:
                accumulator.add_(image.sum(dim=(0, 1)))
            elif signal_only:  # septa that vary along the rows and the channels together: weights per pixel
                accumulator.add_((image * context.pixel_signal_weight(start, stop)).sum(dim=(0, 1)))
            else:
                context.accumulate(accumulator, image, i)
        del image
    if signal_only:
        return {"signal": accumulator}
    return context.finalize(accumulator)


# =====================================================================================================
# primary
# =====================================================================================================

def primary_subpixel_geometry(geometry: ConeGeometry, samples_per_pixel_axis: int) -> ConeGeometry:
    """Panel with n times finer pixels on which the n^2 sub-rays of every pixel are traced.

    Sub-pixel (i, j) of pixel (r, c) is pixel (r n + i, c n + j) of the returned geometry.

    Args:
        geometry (ConeGeometry or ParallelGeometry): scan geometry.
        samples_per_pixel_axis (int): sub-pixel samples n per axis, at least 1.

    Returns:
        ConeGeometry: the refined geometry (``geometry`` itself for n = 1).
    """
    n = int(samples_per_pixel_axis)
    if n < 1:
        raise ValueError("samples_per_pixel_axis must be >= 1")
    if n == 1:
        return geometry
    return replace(geometry, num_det_rows=geometry.num_det_rows * n, num_det_channels=geometry.num_det_channels * n,
                   delta_det_row=geometry.delta_det_row / n, delta_det_channel=geometry.delta_det_channel / n)


def _primary_samples_per_pixel_axis(projectors: Projectors, samples_per_pixel_axis: Optional[int]) -> int:
    """Sub-pixel samples per axis of the primary: the given value, else the projector's own, else 1.

    Args:
        projectors (Projectors): ray tracers.
        samples_per_pixel_axis (int): requested samples per axis, or None.

    Returns:
        int: samples per axis, at least 1.
    """
    n = (int(samples_per_pixel_axis) if samples_per_pixel_axis is not None
         else int(getattr(projectors, "samples_per_pixel_axis", 1) or 1))
    if n < 1:
        raise ValueError("samples_per_pixel_axis must be >= 1")
    return n


def primary_path_lengths(projectors: Projectors, rho: Tensor, grid: Grid, geometry: ConeGeometry, angle: float,
                         samples_per_pixel_axis: Optional[int] = None) -> Tensor:
    """Density line integrals ``P_m`` from the focal spot to the pixel (or sub-pixel) centers.

    The sub-rays are traced as the pixels of :func:`primary_subpixel_geometry` with the projector's own sub-sampling
    off; the materials can be traced one at a time and concatenated.

    Args:
        projectors (Projectors): ray tracers.
        rho (Tensor): [M', R, C, S] density maps in g/cm^3 of any subset of the materials.
        grid (Grid): the grid of ``rho``.
        geometry (ConeGeometry or ParallelGeometry): scan geometry.
        angle (float): view angle in radians.
        samples_per_pixel_axis (int, optional): sub-pixel samples n per axis. Defaults to None, the projector's own
            ``samples_per_pixel_axis`` or 1.

    Returns:
        Tensor: [M', Rd n, Cd n] line integrals in g/cm^2.
    """
    n = _primary_samples_per_pixel_axis(projectors, samples_per_pixel_axis)
    subpixel_geometry = primary_subpixel_geometry(geometry, n)
    if getattr(projectors, "samples_per_pixel_axis", None) is not None:  # disable the projector's own sub-sampling
        # [M', Rd n, Cd n] g/cm^2
        return projectors.primary_project(rho, grid, subpixel_geometry, angle, samples_per_pixel_axis=1)
    return projectors.primary_project(rho, grid, subpixel_geometry, angle)


def primary_from_path_lengths(library: MultigroupLibrary, path: Tensor, materials: Optional[Sequence[int]],
                              geometry: ConeGeometry, angle: float, source: SourceModel, detector: DetectorResponse,
                              grid_response: GridResponse) -> dict[str, Tensor]:
    """Primary image of one view from its per-material density line integrals, exact in energy.

    The fine-energy transmission is averaged over the n x n sub-pixels of every pixel; the inverse-square and
    collimation factors (the beam weight of a parallel beam) and the responses are evaluated at the pixel centers
    (docs/source/method.rst, "The primary").

    Args:
        library (MultigroupLibrary): cross-section library.
        path (Tensor): [M', Rd n, Cd n] line integrals P_m in g/cm^2 from any projector, n >= 1 inferred from the shape.
        materials (Sequence[int]): library indices of the rows of ``path``, or None for all materials in library order.
        geometry (ConeGeometry or ParallelGeometry): scan geometry.
        angle (float): view angle in radians.
        source (SourceModel): source model.
        detector (DetectorResponse): detector response.
        grid_response (GridResponse): anti-scatter-grid response.

    Returns:
        dict: fluence and planar_fluence [G, Rd, Cd] in photons/cm^2, signal_per_group [G, Rd, Cd] in keV/cm^2
            and signal [Rd, Cd], on the device and dtype of ``path``.
    """
    device, dtype = path.device, path.dtype
    Rd, Cd = geometry.num_det_rows, geometry.num_det_channels
    n = int(path.shape[1]) // Rd if path.dim() == 3 else 0
    if n < 1 or tuple(path.shape[1:]) != (Rd * n, Cd * n):
        raise ValueError(f"path must be [M', {Rd} n, {Cd} n] (n >= 1 sub-pixels per axis), got {tuple(path.shape)}")
    transmission = group_transmission(library, path, materials=materials)  # [G, Rd n, Cd n]
    if n > 1:
        transmission = transmission.reshape(-1, Rd, n, Cd, n).mean(dim=(2, 4))
    pixels = geometry.pixel_centers(angle, device, dtype)
    if geometry.is_parallel:  # unit incident fluence along the beam, no inverse-square falloff
        fluence = source.beam_weight(pixels, geometry, angle)[None] * transmission
        omega_p = geometry.primary_directions(angle, device, dtype)
    else:
        r0 = device_tensor(geometry.source_position(angle), device, dtype)
        d = pixels - r0
        distance_squared = (d * d).sum(-1)
        b = source.angular_weight(d, geometry, angle)
        fluence = (b / distance_squared)[None] * transmission
        omega_p = d / distance_squared.sqrt()[..., None]
    group_energy = library.group_energy.to(device, dtype)
    theta0 = detector_angle(omega_p, geometry, angle, pixels)  # [Rd, Cd]
    planar = fluence * torch.cos(theta0)
    detector_response = detector.at(group_energy, theta0)  # [G, Rd, Cd]
    grid_transmission = grid_response.transmission_at(group_energy, omega_p, pixels, geometry, angle)  # [G, Rd, Cd]
    signal_per_group = planar * grid_transmission * detector_response
    return {"fluence": fluence, "planar_fluence": planar, "signal_per_group": signal_per_group,
            "signal": signal_per_group.sum(0)}


def primary_image(library: MultigroupLibrary, rho: Tensor, grid: Grid, geometry: ConeGeometry, angle: float,
                  source: SourceModel, projectors: Projectors, detector: DetectorResponse,
                  grid_response: GridResponse, samples_per_pixel_axis: Optional[int] = None) -> dict[str, Tensor]:
    """Primary image of one view, exact in energy, from the density maps.

    Projects the density map of every material present (:func:`primary_path_lengths`) and forms the transmission and the
    responses with :func:`primary_from_path_lengths`.

    Args:
        library (MultigroupLibrary): cross-section library.
        rho (Tensor): [M, R, C, S] densities in g/cm^3 of the library's materials on ``grid``.
        grid (Grid): the grid of ``rho``.
        geometry (ConeGeometry or ParallelGeometry): scan geometry.
        angle (float): view angle in radians.
        source (SourceModel): source model.
        projectors (Projectors): ray tracers.
        detector (DetectorResponse): detector response.
        grid_response (GridResponse): anti-scatter-grid response.
        samples_per_pixel_axis (int, optional): sub-pixel samples per axis. Defaults to None, the projector's own
            ``samples_per_pixel_axis`` or 1.

    Returns:
        dict: fluence and planar_fluence [G, Rd, Cd] in photons/cm^2, signal_per_group [G, Rd, Cd] in keV/cm^2
            and signal [Rd, Cd].
    """
    _primary_samples_per_pixel_axis(projectors, samples_per_pixel_axis)
    if int(rho.shape[0]) != library.num_materials:
        raise ValueError(f"rho has {int(rho.shape[0])} materials, the library {library.num_materials}")
    present = present_materials(rho)
    present_rho = rho if present is None else rho.index_select(0, device_tensor(present, rho.device, torch.long))
    # [M', Rd n, Cd n] g/cm^2
    path_lengths = primary_path_lengths(projectors, present_rho, grid, geometry, angle, samples_per_pixel_axis)
    return primary_from_path_lengths(library, path_lengths, present, geometry, angle, source, detector, grid_response)


def scatter_fraction(scatter_signal: Tensor, primary_signal: Tensor) -> Tensor:
    """Scatter fraction ``S / (P + S)`` elementwise, 0 where ``P + S = 0``.

    Args:
        scatter_signal (Tensor): scatter signal S.
        primary_signal (Tensor): primary signal P, same shape.

    Returns:
        Tensor: the scatter fraction.
    """
    denominator = primary_signal + scatter_signal
    safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return torch.where(denominator > 0, scatter_signal / safe_denominator, torch.zeros_like(denominator))
