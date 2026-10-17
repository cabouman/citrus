"""Collapsed-cone scatter image of the last-collided source and the primary image, for one view.

One view angle, recon frame, any device; the dtype of the inputs decides the computation dtype.

Collapsed-cone scatter image
----------------------------
For a last-collided emission density E(r, Omega) [photons/(cm^3 sr) per source photon/sr] the angular
fluence arriving at panel point r_D along Omega is the attenuated line integral

    psi_D(r_D, Omega) = int_0^inf dl S(r_D - l Omega, Omega) exp(-tau(r_D - l Omega -> r_D))
                      = int_0^inf dl E_Omega(r_D - l Omega),     E_Omega(r) = S(r, Omega) exp(-tau_down(r, Omega))

with tau_down(r, Omega) = int mu_t along Omega from r to the panel (`downstream_cumulative`). Because
E_Omega already contains the attenuation to the panel, it is a plain volumetric field for each fixed
Omega, and the line integral through each pixel center is `oblique_project` [photons/(cm^2 sr)].
The scalar fluence is the angular integral, done with the direction quadrature {Omega_k, w_k}:

    Phi(r_D) = int dOmega psi_D(r_D, Omega) ~ sum_k w_k psi_D(r_D, Omega_k)          [photons/cm^2]

No 1/r^2 appears: in the point-to-point form one integrates over the volume,
Phi(r_D) = int dV E(r) exp(-tau) / |r_D - r|^2, and the spherical volume element about r_D is
dV = l^2 dl dOmega, which cancels the 1/l^2 exactly, leaving int dOmega int dl E. The inverse square
law is therefore carried by the solid angle geometry of the direction set, not by an explicit factor.

Detector signal
---------------
"fluence" is the scalar fluence (photons per cm^2 perpendicular to each direction). The number of photons crossing
unit PANEL area is the planar fluence

    planar_fluence(r_D) = sum_k w_k cos(theta0_k) psi_D(r_D, Omega_k)                        [photons/cm^2]

with theta0_k the angle between Omega_k and the inward panel normal -n_hat (the same for every pixel of a
flat panel). The detector and grid responses are PER INCIDENT PHOTON (`detector.DetectorResponse`: keV
deposited per photon hitting the panel; `GridResponse`: transmitted fraction of the photons incident on the
grid), so the detector signal is

    signal_g(r_D) = sum_k w_k cos(theta0_k) psi_D,g(r_D, Omega_k) G_g(theta1_k(r_D), theta2_k(r_D)) D_g(theta0_k)

[keV/cm^2]. The primary image uses the same rule with the per-pixel primary direction.

Grids
-----
`combined_scatter_image` evaluates the first-collision source S_FC at the quadrature points (the voxel centers of the
quadrature grid Q = C.refine(q): q^3 equal-weight points per transport cell, docs/source/conventions.rst, "Grids")
and the multiple-collision source S_MC per transport cell, attenuates their sum with tau_down traced on the transport
grid C plus its first-order variation inside the cell (`supervoxel_subpoint_emission`), and projects the sub-point
emission on Q (each point traced to the panel separately), once per direction chunk (docs/source/design_notes.rst,
"Scatter image"). Nothing of the image touches the fine reconstruction grid.
`scatter_image_from_source` is the generic single-grid image of an arbitrary emission (constant per voxel,
exp(-tau_down) at the voxel center), used by tests and diagnostics.

Directions and the finite-cone footprint
----------------------------------------
The collapsed-cone set is a cone product set about -n_hat (toward the panel), `panel_direction_set`,
independent of the sweep quadrature. `covering_polar_angle` gives the smallest cone that contains every
direction from the grid box to the panel (directions outside it contribute exactly zero). Each direction stands
for a solid-angle cell; the finite-cone footprint blurs the projected image of direction k with a box of
(cell width) x (emission-weighted mean height above the panel), the panel footprint of the cell (`_Footprint`).
The scatter image functions take `footprint=False` only to compare single rays with brute-force references in the
tests; the solver always applies it.

Primary
-------
`primary_image` is exact in energy: it projects the density maps of the materials present (g/cm^2) and sums the
transmission over the
library's fine-energy table within each group (`cross_sections.group_transmission`), per sub-ray before the pixel
average, instead of exp(-int mu_t,g dl) with group-averaged mu_t (within-group beam hardening, -1.4 % at WLAB). It is
the composition of two steps that can be fed separately: `primary_path_lengths` (the projector: per-material density
line integrals [M', Rd n, Cd n] of the maps given, one material at a time if the caller wishes) and
`primary_from_path_lengths` (the fine-energy transmission, the source and inverse-square factors and the detector and
anti-scatter-grid responses). The CITRUS loop feeds the second with MBIRTORCH projections of the material maps of all
views at once (`citrus.citrus`, CitrusConfig.primary_projector).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence, Union

import torch
from torch import Tensor

from .angular_quadrature import Quadrature, hemisphere_quadrature, real_spherical_harmonics
from .config import SolverConfig
from .cross_sections import MultigroupLibrary, group_transmission, mu_t_volume
from .detector import DetectorResponse, GridResponse, detector_angle, grid_angles
from .geometry import ConeGeometry, Grid, resample_conservative
from .kernels import triton_utilities
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
    """Smallest polar angle about -n_hat containing every direction from a point of the grid box to a
    point of the active panel area (radians, plus `margin`, capped at pi/2).

    The polar angle of d = p - r is atan(|d_lateral| / d_normal); its sublevel sets are convex cones, so
    over the polytope {p - r} (a Minkowski difference of the box and the panel rectangle) the maximum is
    attained at a vertex, i.e. at a (panel corner) - (box corner) pair. Returns pi/2 if any box corner
    is not strictly on the source side of the panel plane.
    """
    lower = grid.lower_corner()
    extent = grid.extent
    corners = torch.tensor([[lower[0] + i * extent[0], lower[1] + j * extent[1], lower[2] + k * extent[2]]
                            for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=torch.float64)
    u_hat, v_hat, n_hat = (torch.tensor(a, dtype=torch.float64) for a in geometry.panel_axes(angle))
    panel_center = torch.tensor(geometry.panel_center(angle), dtype=torch.float64)
    half_u, half_v = geometry.panel_half_extent()
    center_u, center_v = -geometry.det_channel_offset, -geometry.det_row_offset
    panel_corners = torch.stack([panel_center + (center_u + su * half_u) * u_hat + (center_v + sv * half_v) * v_hat
                                 for su in (-1, 1) for sv in (-1, 1)])
    d = panel_corners[:, None, :] - corners[None, :, :]  # [4, 8, 3]
    normal_distance = (d * (-n_hat)).sum(-1)
    if bool((normal_distance <= 0).any()):
        return math.pi / 2
    lateral = torch.linalg.norm(d - normal_distance[..., None] * (-n_hat), dim=-1)
    return float(min(math.pi / 2, float(torch.atan2(lateral, normal_distance).max()) + margin))


def panel_direction_set(geometry: ConeGeometry, angle: float,
                        config_or_counts: Union[SolverConfig, tuple[int, int]], *,
                        max_polar_angle: Union[float, Grid] = math.pi / 2,
                        device=None, dtype: torch.dtype = torch.float32) -> Quadrature:
    """Collapsed-cone direction set about -n_hat (the directions that reach the panel).

    The Gauss-Legendre (in cos theta) x equal-azimuth product set of `angular_quadrature.hemisphere_quadrature`
    with config_or_counts = a SolverConfig (uses cone_num_polar, cone_num_azimuthal) or a (num_polar, num_azimuthal)
    tuple. max_polar_angle: cone half-angle in rad (default pi/2, the full hemisphere), or a Grid, in which case
    `covering_polar_angle(grid, geometry, angle)` is used (exact: no lost contribution). The cell widths
    (`Quadrature.cell_width` [M, 2], gnomonic coordinates about -n_hat along (u_hat, v_hat)) used by the finite-cone
    footprint are always filled. Returns a Quadrature (omega [M, 3], weight [M] sr) on `device` with `dtype`.
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
    """Everything of a collapsed-cone image that depends only on the direction set, the panel geometry, the
    view angle, the detector / grid responses and the group energies (not on the emission).
    Built by `make_direction_context`; pass it as `context=` to the scatter-image functions to reuse it across
    images of the same view (the scatter image of every solve and of every CITRUS outer iteration).

    K = number of panel-facing directions (Omega . (-n_hat) > 0; the others never reach the panel), G groups,
    Rd x Cd pixels (num_det_rows x num_det_channels).

    quadrature, geometry, angle, detector, grid_response : what the context was built from (checked on use)
    omega          [K, 3]  panel-facing directions, one contiguous tensor (the projectors key their host copy of the
                           directions by storage, so the chunk slices of one context synchronize at most once)
    weight         [K]     quadrature weights (sr)
    cos_theta0     [K]     Omega_k . (-n_hat)
    detector_weight [K, G] D_g(theta0_k) (keV per incident photon)
    grid_weight    [K, G, Rd, 1]  G_g(theta1_k(row), theta2_k) when `separable` (always for the flat ConeGeometry:
                           theta1 = atan2(a.v, a.n) - atan2(f.v, f.n) with f = focus - pixel, and f.v = -v_row,
                           f.n = focal distance do not depend on the channel; theta2 = atan2(a.u, a.n) depends on the
                           direction only); None otherwise (factorize=False or a non-separable panel: only the
                           per-pixel signal_weight is kept, Cg = Cd)
    signal_weight  [K, G, Rd, Cg]  w_k cos(theta0_k) D_g(theta0_k) G_g(...)
    accumulation_weight  separable: [G * Rd, 3, K], the fused accumulation weights (w_k, w_k cos(theta0_k),
                   signal_weight) laid out for one batched matmul per chunk (`accumulate`); else None
    chunk, chunk_bounds  directions per chunk and the (start, stop) of every chunk; omega_chunks / weight_chunks are
                   the matching views of omega / weight
    cell_width     [K, 2] or None: quadrature.cell_width of the kept directions (gnomonic widths of each direction's
                   cell along u_hat, v_hat; needed by the finite-cone footprint)

    Representation choice (exact, smallest): the detector response depends on the direction only through theta0 ->
    [K, G]; the grid response depends on (theta1, theta2), which for the flat panel vary with the direction and the
    detector ROW only -> [K, G, Rd] (per group because the table is energy dependent). The fused weights are
    [G * Rd, 3, K]: 2048 directions x 6 groups x 75 rows x 3 = 11 MB in float32, instead of 370 MB per weight for a
    per-pixel [K, G, Rd, Cd] table. All responses are evaluated in float64 and stored in the context dtype.

    Lazily cached per context: real spherical harmonics of the directions per order L (`spherical_harmonics`),
    the sub-cell offsets Omega_k . (r_j - r_cell) per (quadrature grid, q) (`subcell_offsets`) and the panel heights of
    the finite-cone footprint per grid (`_Footprint`), so that repeated images of a view need no host synchronization.
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
    _cache: dict = field(default_factory=dict, repr=False)

    # ---- sizes ----------------------------------------------------------------------------------
    @property
    def num_directions(self) -> int:
        return int(self.omega.shape[0])

    @property
    def num_groups(self) -> int:
        return int(self.group_energy.numel())

    @property
    def device(self) -> torch.device:
        return self.omega.device

    @property
    def dtype(self) -> torch.dtype:
        return self.omega.dtype

    @property
    def nbytes(self) -> int:
        """Bytes held by the context's tensors (distinct storages, lazily cached entries included)."""
        seen: dict[int, int] = {}
        tensors = [t for t in (self.group_energy, self.omega, self.weight, self.cos_theta0, self.detector_weight,
                               self.grid_weight, self.signal_weight, self.accumulation_weight, self.cell_width)
                   if t is not None]
        for v in self._cache.values():
            tensors.extend(t for t in (v if isinstance(v, tuple) else (v,)) if isinstance(t, Tensor))
        for t in tensors:
            storage = t.untyped_storage()
            seen[storage.data_ptr()] = storage.nbytes()
        return int(sum(seen.values()))

    # ---- checks ----------------------------------------------------------------------------------
    def check(self, geometry: ConeGeometry, angle: float, quadrature: Quadrature, detector: DetectorResponse,
              grid_response: GridResponse, num_groups: int, device: torch.device, dtype: torch.dtype) -> None:
        """Raise ValueError unless the context was built for these arguments (host-side checks only: identity of
        the direction set and the responses, geometry and angle equality, group count, device and dtype)."""
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
        """Real spherical harmonics of the context directions, one [k, (L+1)^2] view per chunk (cached per L)."""
        key = ("spherical_harmonics", int(L))
        cached = self._cache.get(key)
        if cached is None:
            Y = real_spherical_harmonics(int(L), self.omega)  # [K, (L+1)^2]
            cached = (Y,) + tuple(Y[s:e] for s, e in self.chunk_bounds)
            self._cache[key] = cached
        return cached[1:]

    def subcell_offsets(self, grid_quadrature: Grid, q: int) -> tuple[Tensor, ...]:
        """Omega_k . (r_j - r_cell) for the q^3 quadrature points r_j of a transport cell (the voxel centers of
        grid_quadrature inside the cell, `supervoxel_subpoint_emission`), one [k, q, q, q] view per chunk (axes: rows,
        cols, slices), cached per (grid_quadrature voxel deltas, q)."""
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
        """Zeroed image accumulator: [G * Rd, 3, Cd] (separable) or [3, G, Rd, Cd]."""
        G, (Rd, Cd) = self.num_groups, self.image_shape
        if self.separable:
            return torch.zeros(G * Rd, 3, Cd, device=self.device, dtype=self.dtype)
        return torch.zeros(3, G, Rd, Cd, device=self.device, dtype=self.dtype)

    def accumulate(self, accumulator: Tensor, image: Tensor, chunk_index: int) -> None:
        """accumulator += the weighted chunk image: image [k, G, Rd, Cd] = the projected attenuated emission of chunk
        `chunk_index` (k directions) in photons/(cm^2 sr). Separable: ONE batched matmul (baddbmm, no copy when image
        is contiguous; cuBLAS picks a small-N kernel, ~7 us per chunk on the WLAB panel):
        accumulator[(g, r), j, c] += sum_k W[(g, r), j, k] image[k, g, r, c], j = (fluence, planar, signal)."""
        start, stop = self.chunk_bounds[chunk_index]
        k = stop - start
        G, (Rd, Cd) = self.num_groups, self.image_shape
        if self.separable:
            image_view = image.permute(1, 2, 0, 3).reshape(G * Rd, k, Cd)  # a view for a contiguous image
            accumulator.baddbmm_(self.accumulation_weight[:, :, start:stop], image_view)
            return
        weights = self.weight_chunks[chunk_index]
        accumulator[0].add_(torch.einsum("kgrc,k->grc", image, weights))
        accumulator[1].add_(torch.einsum("kgrc,k->grc", image, weights * self.cos_theta0[start:stop]))
        accumulator[2].add_((image * self.signal_weight[start:stop]).sum(0))

    def finalize(self, accumulator: Tensor) -> dict[str, Tensor]:
        """The image dict of `scatter_image_from_source` from an accumulator."""
        G, (Rd, Cd) = self.num_groups, self.image_shape
        if self.separable:
            accumulator = accumulator.reshape(G, Rd, 3, Cd).permute(2, 0, 1, 3)
        fluence, planar, signal_per_group = (accumulator[j].contiguous() for j in range(3))
        return {"fluence": fluence, "planar_fluence": planar, "signal_per_group": signal_per_group,
                "signal": signal_per_group.sum(0)}


def _subcell_offsets(omega: Tensor, grid_quadrature: Grid, f: int) -> Tensor:
    """Omega_k . (r_j - r_cell) [K, q_r, q_c, q_s] for the q^3 = f^3 voxel centers of grid_quadrature inside one cell of
    grid_quadrature.coarsen(f); omega [K, 3]."""
    device, dtype = omega.device, omega.dtype
    dx, dy, dz = grid_quadrature.deltas
    centered_index = torch.arange(f, device=device, dtype=dtype) - (f - 1) / 2.0
    directions = omega.reshape(-1, 3)
    return (directions[:, 1, None, None, None] * (dy * centered_index)[None, :, None, None]
            + directions[:, 0, None, None, None] * (dx * centered_index)[None, None, :, None]
            + directions[:, 2, None, None, None] * (dz * centered_index)[None, None, None, :])


def make_direction_context(quadrature: Quadrature, geometry: ConeGeometry, angle: float, detector: DetectorResponse,
                           grid_response: GridResponse, library: Optional[MultigroupLibrary] = None,
                           device=None, dtype: Optional[torch.dtype] = None, *,
                           group_energy: Optional[Tensor] = None, chunk: int = 8,
                           factorize: bool = True) -> DirectionSetContext:
    """Precompute the emission-independent part of a collapsed-cone image for one (view, direction set).

    quadrature: direction set (omega [M, 3], weight [M]); only the panel-facing directions are kept (K of them, selected
      here once, with the only host synchronization of the context's life).
    library / group_energy: group energies [G] keV for the response lookups (group_energy wins).
    device, dtype: of the images the context will serve (default: quadrature's device, float32 / quadrature's float
    dtype). chunk: directions per chunk of the image loop (the context's chunking is used by every image
      it serves).
    factorize: store the grid response per detector row ([K, G, Rd, 1], exact for the flat panel; checked here
      numerically, a non-separable panel falls back automatically) and accumulate with one batched matmul per
      chunk; False keeps a per-pixel [K, G, Rd, Cd] table (reference path, 4 x K x G x pixels bytes).
    No grid argument: nothing in the per-chunk response work depends on the voxel grid, so one context serves every
    image of a view, on any grid. Grid-dependent tables (sub-cell offsets) are cached lazily by grid.
    Returns a DirectionSetContext (its docstring has the shapes and the representation).
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
    cos64 = (omega64 * (-torch.tensor(n_hat, device=device, dtype=float64))).sum(-1)
    # the same selection as before in the dtype of the images (a direction grazing the panel plane must be kept or
    # dropped exactly as the projectors see it)
    cos_image_dtype = (quadrature.omega.detach().to(device, dtype)
                       * (-torch.tensor(n_hat, device=device, dtype=dtype))).sum(-1)
    kept = torch.nonzero(cos_image_dtype > 0).reshape(-1)
    omega64, weight64, cos64 = omega64[kept], weight64[kept], cos64[kept]
    K = int(omega64.shape[0])
    pixels = geometry.pixel_centers(angle, device, float64)  # [Rd, Cd, 3]
    Rd, Cd = int(pixels.shape[0]), int(pixels.shape[1])
    focal_distance = grid_response.focal_distance_cm
    # detector response: theta0 per direction -> [G, K]
    detector_response = (detector.at(energy64, detector_angle(omega64, geometry, angle)) if K
                         else torch.zeros(G, 0, device=device, dtype=float64))
    # grid response: the grid angles of one probe direction over the full panel decide separability
    separable = False
    if factorize:
        probe = torch.tensor(n_hat, device=device, dtype=float64)[None].neg()
        if K:
            probe = torch.cat([probe, omega64[:1]])
        probe_theta1, probe_theta2 = grid_angles(probe, pixels, geometry, angle, focal_distance)  # [P, Rd, Cd]
        max_deviation = torch.maximum((probe_theta1 - probe_theta1[..., :1]).abs().max(),
                                      (probe_theta2 - probe_theta2[..., :1]).abs().max())
        separable = bool(max_deviation <= 1e-10)  # rad; float64 rounding of the pixel positions is ~1e-17
    pixel_columns = pixels[:, Cd // 2:Cd // 2 + 1] if separable else pixels  # [Rd, Cg, 3]
    Cg = int(pixel_columns.shape[1])
    detector_weight = detector_response.transpose(0, 1)  # [K, G]
    c = cos64
    # grid response and signal weights, evaluated in float64 over blocks of directions (bounded temporaries),
    # stored in the image dtype
    grid_weight = torch.empty(K, G, Rd, Cg, device=device, dtype=dtype) if separable else None
    signal_weight = torch.empty(K, G, Rd, Cg, device=device, dtype=dtype)
    block = max(1, (1 << 22) // max(1, G * Rd * Cg))
    for start in range(0, K, block):
        stop = min(start + block, K)
        th1, th2 = grid_angles(omega64[start:stop], pixel_columns, geometry, angle, focal_distance)  # [b, Rd, Cg]
        grid_block = grid_response.at(energy64, th1, th2).permute(1, 0, 2, 3)  # [b, G, Rd, Cg]
        if grid_weight is not None:
            grid_weight[start:stop] = grid_block
        signal_weight[start:stop] = ((weight64[start:stop] * c[start:stop])[:, None, None, None]
                                     * detector_weight[start:stop, :, None, None] * grid_block)
        del th1, th2, grid_block
    accumulation_weight = None
    if separable:
        accumulation_weight = torch.empty(G, Rd, 3, K, device=device, dtype=dtype)
        accumulation_weight[:, :, 0] = weight64.to(dtype)
        accumulation_weight[:, :, 1] = (weight64 * cos64).to(dtype)
        accumulation_weight[:, :, 2] = signal_weight[..., 0].permute(1, 2, 0)
        accumulation_weight = accumulation_weight.reshape(G * Rd, 3, K)
    chunk = max(1, int(chunk))
    chunk_bounds = tuple((start, min(start + chunk, K)) for start in range(0, K, chunk))
    omega = omega64.to(dtype).contiguous()
    weight = weight64.to(dtype).contiguous()
    cell_width = getattr(quadrature, "cell_width", None)
    cell_width = None if cell_width is None else cell_width.detach().to(device, dtype)[kept].contiguous()
    return DirectionSetContext(
        quadrature=quadrature, geometry=geometry, angle=float(angle), detector=detector, grid_response=grid_response,
        group_energy=energy64.to(dtype), omega=omega, weight=weight,
        cos_theta0=cos64.to(dtype), detector_weight=detector_weight.to(dtype).contiguous(), grid_weight=grid_weight,
        signal_weight=signal_weight, accumulation_weight=accumulation_weight, separable=separable, chunk=chunk,
        chunk_bounds=chunk_bounds, omega_chunks=tuple(omega[start:stop] for start, stop in chunk_bounds),
        weight_chunks=tuple(weight[start:stop] for start, stop in chunk_bounds), image_shape=(Rd, Cd),
        cell_width=cell_width)


def _resolve_context(context: Optional[DirectionSetContext], quadrature: Quadrature, geometry: ConeGeometry,
                     angle: float, detector: DetectorResponse, grid_response: GridResponse,
                     library: Optional[MultigroupLibrary], mu_t: Tensor, chunk: int,
                     group_energy: Optional[Tensor]) -> DirectionSetContext:
    """The given context (checked against the arguments) or a new one built for them on mu_t's device / dtype."""
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
    """Collapsed-cone image of a last-collided emission source on ONE grid (module docstring).

    The generic single-grid form of the collapsed-cone image (the solver's pass is `combined_scatter_image`; this
    function serves tests and diagnostics with an arbitrary emission).
    source_function(omegas [k, 3]) -> [k, G, rows, cols, slices]: emission density S_g(r, Omega_k) at the
      voxel centers of `grid` in photons/(cm^3 sr) per source photon/sr (treated as constant per voxel).
    mu_t: [G, rows, cols, slices] (1/cm) on `grid`, used for the attenuation to the panel.
    quadrature: direction set (e.g. `panel_direction_set`); directions with Omega . (-n_hat) <= 0 are skipped
      (they never reach the panel).
    group_energy [G] keV for the response lookups: taken from `library.group_energy` unless given.
    chunk: directions per batch (peak memory ~ 3 chunk G |grid| scalars plus the projector's working set).
    context: optional `make_direction_context(quadrature, geometry, angle, detector, grid_response, library, ...)` of
      this view (built here when None). It holds the panel-facing directions, their chunking (its `chunk` replaces
      the argument) and the precomputed response weights; it must match the other arguments (checked).
    footprint: apply the finite-cone footprint (`_Footprint`, heights weighted by this emission; needs
      quadrature.cell_width); False projects every direction as a single ray per pixel.

    For each chunk: tau = downstream_cumulative(mu_t, Omega_k); E_k = S(Omega_k) exp(-tau) (exp(-inf) = 0);
    img_k = oblique_project(E_k) [photons/(cm^2 sr)], then one fused weighted accumulation
    (`DirectionSetContext.accumulate`): fluence += w_k img_k; planar_fluence += w_k cos(theta0_k) img_k;
    signal_per_group += w_k img_k cos(theta0_k) G_g(theta1_k, theta2_k) D_g(theta0_k) (module docstring).

    Returns dict:
      fluence          [G, num_det_rows, num_det_channels]  scalar fluence, photons/cm^2 per source photon/sr
      planar_fluence   [G, num_det_rows, num_det_channels]  photons crossing unit panel area (no grid/detector)
      signal_per_group [G, num_det_rows, num_det_channels]  keV/cm^2 (or photons/cm^2 for a counting D)
      signal           [num_det_rows, num_det_channels]     sum over groups
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
    """Validate source_quadrature [k, G, Rq, Cq, Sq] on grid_quadrature and tau_coarse [k, G, Rc, Cc, Sc]; returns
    (Rc, Cc, Sc)."""
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
    """Attenuated emission at every quadrature point, kept on the quadrature grid Q (before the projection and the sum
    over j).

    The quadrature points r_j of a transport cell of C = grid_quadrature.coarsen(q) are the q^3 voxel centers of
    grid_quadrature inside it (equal weights 1/q^3):

        e[k, g, r_j] = S[k, g, r_j] exp(-tau_j),
        tau_j = max(0, tau_coarse[k, g, cell] - mu_t_coarse[g, cell] (Omega_k . (r_j - r_cell)))

    i.e. the downstream attenuation at the supervoxel center plus its first-order variation inside the cell (moving
    along Omega_k toward the panel shortens the path); exact for a uniform cell whose rays leave through the same
    face as the center ray. Projecting this field on grid_quadrature traces each point to the panel separately.
    source_quadrature [k, G, Rq, Cq, Sq] on grid_quadrature; tau_coarse [k, G, Rc, Cc, Sc] on C
    (projectors.downstream_cumulative; +inf allowed); mu_t_coarse [G, Rc, Cc, Sc] (1/cm); omegas [k, 3]. Returns
    e [k, G, Rq, Cq, Sq] in photons/(cm^3 sr) on grid_quadrature, in the device / dtype of tau_coarse. For q = 1 this is
    S exp(-tau) (no clamp). Peak extra memory ~ one quadrature-grid field [k, G, Rq, Cq, Sq]. offsets: optional
    precomputed Omega_k . (r_j - r_cell) [k, q, q, q] (`DirectionSetContext.subcell_offsets`), computed here when None.
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
    """Cell average of an attenuation map [G, R, C, S] on grid_fine over the cells of grid_coarse -> [G, Rc, Cc, Sc]
    (1/cm; `geometry.resample_conservative`, zero outside the box of grid_fine).

    Because mu_t is linear in the densities this equals mu_t_volume(library, resampled rho).
    """
    return resample_conservative(mu_t_fine, grid_fine, grid_coarse)


# =====================================================================================================
# finite-cone footprint
# =====================================================================================================

def box_blur_rows_cols(image: Tensor, width_channels: Tensor, width_rows: Tensor) -> Tensor:
    """Separable box filter of image [k, G, Rd, Cd] with a per-image width (in PIXELS, fractional allowed) along the
    channels (width_channels [k]) and the rows (width_rows [k]): out[c] = (1/|W|) int_W x over the window
    W = [c + 1/2 - w/2, c + 1/2 + w/2] clipped to the panel [0, Cd] (pixel j covers [j, j + 1]; x is piecewise constant;
    the window is renormalized at the panel edges). A width <= 1 pixel leaves a pixel unchanged (the window stays inside
    the pixel). Done as two batched matmuls with the [k, N, N] window-overlap matrices (`box_matrix`; ~0.1 ms per chunk
    of 8 directions on the 184 x 150 panel, cheaper than gathers from a cumulative sum). Returns [k, G, Rd, Cd]."""
    k, G, Rd, Cd = image.shape
    channel_matrix = box_matrix(Cd, width_channels, image.dtype)  # [k, Cd, Cd]
    row_matrix = box_matrix(Rd, width_rows, image.dtype)  # [k, Rd, Rd]
    out = torch.bmm(image.reshape(k, G * Rd, Cd), channel_matrix.transpose(1, 2)).view(k, G, Rd, Cd)
    return torch.matmul(row_matrix[:, None], out)


def box_matrix(N: int, w: Tensor, dtype: Optional[torch.dtype] = None) -> Tensor:
    """[k, N, N] matrices of the 1-D box filter of box_blur_rows_cols: B[c, j] = |[j, j + 1] n W_c| / |W_c|, W_c =
    [c + 1/2 - w/2, c + 1/2 + w/2] n [0, N], for widths w [k] (pixels; clamped below at 1e-3). float32 CUDA: one
    Triton launch (`_box_matrix_kernel`; the PyTorch expression below takes ~17 launches, ~0.3 ms per call)."""
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
        """Row c = program_id(1) of box matrix k = program_id(0) (box_matrix)."""
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
    """Normal distance of every voxel center from the panel plane, h(r) = (r - panel_center) . n_hat [R, C, S] (cm,
    positive on the source side). A ray from r along a direction with gnomonic coordinates (alpha, beta) about -n_hat
    hits the panel at the foot of r plus h (alpha, beta) (along u_hat, v_hat): the footprint of a direction CELL of
    gnomonic widths (W_alpha, W_beta) is an h W_alpha x h W_beta rectangle."""
    _, _, n_hat = geometry.panel_axes(angle)
    r = grid.center_coordinates(device, torch.float64)
    panel_center = torch.tensor(geometry.panel_center(angle), device=device, dtype=torch.float64)
    return ((r - panel_center) @ torch.tensor(n_hat, device=device, dtype=torch.float64)).to(dtype)


class _Footprint:
    """Finite-cone footprint of the direction chunks of one image (combined_scatter_image / scatter_image_from_source;
    docs/source/design_notes.rst, "Scatter image").

    Direction k stands for a solid-angle cell of gnomonic widths (W_alpha,k, W_beta,k) about -n_hat
    (`context.cell_width`).
    A point at height h above the panel sends that cell onto an h W_alpha x h W_beta rectangle of the panel, so the
    projected single-ray image of direction k is box-blurred with widths hbar_k (W_alpha,k, W_beta,k) / pixel, hbar_k =
    the emission-weighted mean panel height of the attenuated emission of the image along Omega_k (`mean_height`). A
    direction whose emission vanishes uses the height of the box centre. Exact up to the height spread of the emission
    and the cell shape (a box with the same marginal widths as the polar x azimuth cell)."""

    def __init__(self, grid: Grid, geometry: ConeGeometry, angle: float, context: DirectionSetContext) -> None:
        """grid: grid of the emission that is projected."""
        if context.cell_width is None:
            raise ValueError("the finite-cone footprint needs quadrature.cell_width (panel_direction_set fills it)")
        self.context = context
        key = ("footprint", grid)
        # the tables depend on the grid only (geometry and angle are those of the context)
        cached = context._cache.get(key)
        if cached is None:
            device, dtype = context.device, context.dtype
            scale = torch.tensor([1.0 / geometry.delta_det_channel, 1.0 / geometry.delta_det_row], device=device,
                                 dtype=dtype)
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

    def mean_height(self, emission: Tensor) -> Tensor:
        """Emission-weighted mean panel height [k] of an attenuated emission [k, G, *grid] (the box centre where the
        emission vanishes)."""
        k = int(emission.shape[0])
        total = emission.sum(dim=1).reshape(k, -1)  # [k, voxels]
        numerator, denominator = total @ self.heights, total.sum(-1)
        tiny = torch.finfo(denominator.dtype).tiny
        return torch.where(denominator > 0, numerator / denominator.clamp_min(tiny),
                           torch.full_like(denominator, self.center_height))

    def blur(self, image: Tensor, chunk_index: int, mean_heights: Tensor) -> Tensor:
        """Footprint image [k, G, Rd, Cd] of the single-ray image of chunk `chunk_index` with mean heights
        mean_heights [k]."""
        start, stop = self.context.chunk_bounds[chunk_index]
        # [k, 2] pixels (channels, rows)
        widths = mean_heights[:, None] * self.context.cell_width[start:stop] * self.scale
        return box_blur_rows_cols(image, widths[:, 0], widths[:, 1])


# =====================================================================================================
# the scatter image: first-order + multiple-collision emission, traced together
# =====================================================================================================

def combined_scatter_image(library: MultigroupLibrary, rho_quadrature: Tensor, phi_uncollided: Tensor,
                           grid_quadrature: Grid, q: int, scatter_source_moments: Tensor, L: int,
                           geometry: ConeGeometry, angle: float, quadrature: Quadrature,
                           mu_t_coarse: Optional[Tensor], projectors: Projectors, detector: DetectorResponse,
                           grid_response: GridResponse, context: Optional[DirectionSetContext] = None,
                           chunk: int = 8, *, first_collision_chunk: int = 4, footprint: bool = True,
                           pairs=None, grid_coarse: Optional[Grid] = None) -> dict[str, Tensor]:
    """Total scatter image (first-order + multiple-collision) in ONE collapsed-cone pass: the first- and
    multiple-collision sources are summed and ray-traced to the detector together, on the transport grid with q^3
    quadrature points per cell.

    library; rho_quadrature [M, Rq, Cq, Sq] g/cm^3 and phi_uncollided [G, Rq, Cq, Sq] (uncollided.uncollided_fluence at
    the points) on the quadrature grid grid_quadrature, whose voxel centers are the q^3 equal-weight quadrature points
    of each cell of the transport grid C = grid_quadrature.coarsen(q) (grid_coarse: C itself when given, e.g. the
    solver's transport grid, of which grid_quadrature = C.refine(q); it then also fixes the attenuation grid exactly);
    scatter_source_moments [G, (L+1)^2, Rc, Cc, Sc]: cell-mean spherical-harmonic moments of the multiple-collision
    emission (already contracted with the transfer moments, `ScatterSolver.multiple_collision_source_moments`), or the
    6-D [G, (L+1)^2, 4, Rc, Cc, Sc] tensor whose linear-discontinuous mean is used; mu_t_coarse [G, Rc, Cc, Sc] on C
    (None -> the cell average of mu_t_volume(library, rho_quadrature)). context: the view's `make_direction_context`
    (built here when None; its chunking replaces `chunk`). first_collision_chunk: directions per inner batch of the
    first-collision source (PyTorch path). pairs: the material representation of rho_quadrature
    (`uncollided.resolve_pairs`: a `cross_sections.MaterialPairs`, `uncollided.DENSE`, or None to decide here once).

    Per direction chunk (Omega_k, k of them):
      tau    = downstream_cumulative(mu_t_coarse, C, Omega_k)   [k, G, Rc, Cc, Sc]
      S_FC,j = first_collision_source_at_directions at the quadrature points r_j (exact eta table)    [k, G, Rq, Cq, Sq]
      S_MC   = Y(Omega_k) . T per cell (one matmul with the cached harmonics), broadcast to its quadrature points
      e_j    = (S_FC,j + S_MC) exp(-tau_j)   (`supervoxel_subpoint_emission`: tau at the cell center plus its in-cell
               variation)
    ONE oblique_project of e on grid_quadrature (each point traced to the panel separately; the attenuation stays on C),
    the finite-cone footprint blur (`_Footprint` with the heights of the Q points, footprint=True) and ONE fused
    accumulation (`DirectionSetContext.accumulate`). q = 1 is the single-grid image `scatter_image_from_source` of
    S_FC + S_MC on C. The image is linear in (phi_uncollided, scatter_source_moments) apart from the footprint heights,
    which follow the summed emission.

    Returns the image dict of the total scatter (fluence, planar_fluence, signal_per_group [G, Rd, Cd],
    signal [Rd, Cd]). Peak extra memory per chunk ~ 2-3 quadrature-grid fields [k, G, Rq, Cq, Sq].
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

    accumulator = context.new_accumulator()
    for i, directions in enumerate(context.omega_chunks):
        k = int(directions.shape[0])
        first_collision = first_collision_source_at_directions(library, rho, phi_uncollided, grid_quadrature, geometry,
                                                               angle, directions, chunk=first_collision_chunk,
                                                               pairs=pairs)
        if tuple(first_collision.shape) != (k, G, *quadrature_shape):
            raise ValueError(f"first-collision source {tuple(first_collision.shape)} != {(k, G, *quadrature_shape)}")
        # [k, G, Rc, Cc, Sc]
        tau = projectors.downstream_cumulative(mu_t_coarse, grid_coarse, geometry, angle, directions)
        # S_MC(Omega_k) per supervoxel
        multiple_collision = torch.mm(harmonics[i], moments_by_cell).view(k, G, Rc, Cc, Sc)
        # one source at the quadrature points, one attenuation, one projection
        total_source = (first_collision.reshape(k, G, Rc, f, Cc, f, Sc, f)
                        .add(multiple_collision[:, :, :, None, :, None, :, None]).view(k, G, *quadrature_shape))
        del first_collision, multiple_collision
        emission = supervoxel_subpoint_emission(total_source, tau, mu_t_coarse, directions, grid_quadrature, f,
                                                offsets=None if subcell_offsets is None else subcell_offsets[i])
        del total_source, tau
        # [k, G, Rd, Cd] photons/(cm^2 sr)
        image = projectors.oblique_project(emission, grid_quadrature, geometry, angle, directions)
        if footprint_blur is not None:
            image = footprint_blur.blur(image, i, footprint_blur.mean_height(emission))
        del emission
        context.accumulate(accumulator, image, i)
        del image
    return context.finalize(accumulator)


# =====================================================================================================
# primary
# =====================================================================================================

def primary_subpixel_geometry(geometry: ConeGeometry, samples_per_pixel_axis: int) -> ConeGeometry:
    """The panel of n = samples_per_pixel_axis times finer pixels on which `primary_path_lengths` traces the n^2
    sub-rays of every pixel: sub-pixel (i, j) of pixel (r, c) is pixel (r n + i, c n + j) of the returned geometry
    (same panel, same offsets). n = 1 returns geometry itself."""
    n = int(samples_per_pixel_axis)
    if n < 1:
        raise ValueError("samples_per_pixel_axis must be >= 1")
    if n == 1:
        return geometry
    return replace(geometry, num_det_rows=geometry.num_det_rows * n, num_det_channels=geometry.num_det_channels * n,
                   delta_det_row=geometry.delta_det_row / n, delta_det_channel=geometry.delta_det_channel / n)


def _primary_samples_per_pixel_axis(projectors: Projectors, samples_per_pixel_axis: Optional[int]) -> int:
    n = (int(samples_per_pixel_axis) if samples_per_pixel_axis is not None
         else int(getattr(projectors, "samples_per_pixel_axis", 1) or 1))
    if n < 1:
        raise ValueError("samples_per_pixel_axis must be >= 1")
    return n


def primary_path_lengths(projectors: Projectors, rho: Tensor, grid: Grid, geometry: ConeGeometry, angle: float,
                         samples_per_pixel_axis: Optional[int] = None) -> Tensor:
    """Primary density line integrals P_m = int_{r_0}^{r_D} rho_m dl (g/cm^2) from the focal spot to the pixel (or
    sub-pixel) centers: rho [M', R, C, S] (g/cm^3; any subset of the materials, e.g. one map at a time, or the maps
    of the materials present) on `grid` -> [M', Rd n, Cd n] on the sub-pixel panel
    `primary_subpixel_geometry(geometry, n)`. n = samples_per_pixel_axis: sub-pixel sampling per axis (None: the
    projector's own `samples_per_pixel_axis` attribute, 1 if it has none); the projector's own sub-sampling is switched
    off (the sub-rays are traced as pixels of the finer panel and the pixel average is taken AFTER the fine-energy
    transmission, by `primary_from_path_lengths`). The projector is linear and acts channel by channel, so tracing the
    materials one at a time and concatenating gives the same path lengths."""
    n = _primary_samples_per_pixel_axis(projectors, samples_per_pixel_axis)
    subpixel_geometry = primary_subpixel_geometry(geometry, n)
    if getattr(projectors, "samples_per_pixel_axis", None) is not None:  # disable the projector's own sub-sampling
        # [M', Rd n, Cd n] g/cm^2
        return projectors.primary_project(rho, grid, subpixel_geometry, angle, samples_per_pixel_axis=1)
    return projectors.primary_project(rho, grid, subpixel_geometry, angle)


def primary_from_path_lengths(library: MultigroupLibrary, path: Tensor, materials: Optional[Sequence[int]],
                              geometry: ConeGeometry, angle: float, source: SourceModel, detector: DetectorResponse,
                              grid_response: GridResponse) -> dict[str, Tensor]:
    """Primary image of one view from its per-material density line integrals (the second half of `primary_image`):

        T_g(u)              = < sum_{E in g} w_E exp(-sum_m P_m m_m(E)) >_pixel        (`group_transmission`)
        fluence_g(u)        = b(Omega_p) / |r_D - r_0|^2 T_g(u)                          photons/cm^2
        planar_fluence_g(u) = cos(theta0(u)) fluence_g(u)                               photons/cm^2
        signal_g(u)         = planar_fluence_g(u) G_g(theta1, theta2) D_g(theta0)      keV/cm^2

    path: [M', Rd n, Cd n] line integrals P_m (g/cm^2) on the panel refined n times per axis (n >= 1 inferred from the
    shape, `primary_subpixel_geometry`; the transmission is averaged over the n x n sub-pixels of every pixel), from
    any projector: `primary_path_lengths` (Siddon, one ray to each (sub-)pixel center) or a pixel-averaged projector
    such as MBIRTORCH's (`citrus.citrus.primary_signals`, n = 1). materials: library indices of the rows of
    path (None = all materials in library order; an absent material contributes nothing, so the rows of the materials
    present suffice). The 1/r^2 and b factors and the responses are evaluated at the pixel centers, the grid angles
    from `grid_angles` for the primary direction (theta1 = 0 when the grid is focused at the focal spot;
    theta2 = -atan(u/SDD)), theta0(u) the angle between Omega_p and the inward panel normal. Returns dict with fluence
    [G, Rd, Cd], planar_fluence [G, Rd, Cd], signal_per_group [G, Rd, Cd], signal [Rd, Cd], in the dtype and on the
    device of path."""
    device, dtype = path.device, path.dtype
    Rd, Cd = geometry.num_det_rows, geometry.num_det_channels
    n = int(path.shape[1]) // Rd if path.dim() == 3 else 0
    if n < 1 or tuple(path.shape[1:]) != (Rd * n, Cd * n):
        raise ValueError(f"path must be [M', {Rd} n, {Cd} n] (n >= 1 sub-pixels per axis), got {tuple(path.shape)}")
    transmission = group_transmission(library, path, materials=materials)  # [G, Rd n, Cd n]
    if n > 1:
        transmission = transmission.reshape(-1, Rd, n, Cd, n).mean(dim=(2, 4))
    pixels = geometry.pixel_centers(angle, device, dtype)
    r0 = torch.tensor(geometry.source_position(angle), device=device, dtype=dtype)
    d = pixels - r0
    distance_squared = (d * d).sum(-1)
    b = source.angular_weight(d, geometry, angle)
    fluence = (b / distance_squared)[None] * transmission
    omega_p = d / distance_squared.sqrt()[..., None]
    group_energy = library.group_energy.to(device, dtype)
    theta0 = detector_angle(omega_p, geometry, angle)  # [Rd, Cd]
    planar = fluence * torch.cos(theta0)
    detector_response = detector.at(group_energy, theta0)  # [G, Rd, Cd]
    theta1, theta2 = grid_angles(omega_p, pixels, geometry, angle, grid_response.focal_distance_cm)  # [Rd, Cd]
    grid_transmission = grid_response.at(group_energy, theta1, theta2)
    signal_per_group = planar * grid_transmission * detector_response
    return {"fluence": fluence, "planar_fluence": planar, "signal_per_group": signal_per_group,
            "signal": signal_per_group.sum(0)}


def primary_image(library: MultigroupLibrary, rho: Tensor, grid: Grid, geometry: ConeGeometry, angle: float,
                  source: SourceModel, projectors: Projectors, detector: DetectorResponse,
                  grid_response: GridResponse, samples_per_pixel_axis: Optional[int] = None) -> dict[str, Tensor]:
    """Primary image, per group at the pixels, exact in energy:

        fluence_g(u)        = b(Omega_p) / |r_D - r_0|^2 T_g(u)                                photons/cm^2
        T_g(u)              = < sum_{E in g} w_E exp(-sum_m P_m m_m(E)) >_pixel
        planar_fluence_g(u) = cos(theta0(u)) fluence_g(u)                                     photons/cm^2
        signal_g(u)         = planar_fluence_g(u) G_g(theta1, theta2) D_g(theta0)            keV/cm^2

    P_m = int_{r_0}^{r_D} rho_m dl: line integrals of the density maps (g/cm^2; one projection per material PRESENT,
    `projectors.primary_project(rho[present])`, `uncollided.present_materials`), w_E / m_m(E) the library's fine-energy
    table (library.fine_weight, library.fine_mu_t; `cross_sections.group_transmission`). fluence_g is the fine-energy
    sum within group g, so the per-group detector and grid responses (at library.group_energy) apply, and
    fluence.sum(0) / planar_fluence.sum(0) is the exact polychromatic primary (up to the fine quadrature; the
    group-averaged exp(-int mu_t,g dl) ignores within-group beam hardening).
    theta0(u) = angle between Omega_p and the inward panel normal (per incident photon responses, module docstring).

    with Omega_p the direction focal spot -> pixel center. The grid angles come from `grid_angles` for the
    primary direction (theta1 = 0 when the grid is focused at the focal spot; theta2 = -atan(u/SDD)).
    samples_per_pixel_axis: sub-pixel sampling per axis (None: the projector's own `samples_per_pixel_axis` attribute, 1
    if it has none). With n > 1 the density line integrals are traced to the n x n midpoint lattice of sub-pixel centers
    (a panel of n-times finer pixels), the fine-energy transmission is formed per sub-ray and THEN averaged over the
    pixel (<.>_pixel above: the exact pixel-averaged transmission, no -log(mean(exp)) per group); the 1/r^2 and b
    factors are evaluated at the pixel center.
    Equals `primary_from_path_lengths(library, primary_path_lengths(projectors, rho[present], grid, geometry, angle,
    samples_per_pixel_axis), present, geometry, angle, source, detector, grid_response)`.
    rho [M, rows, cols, slices]. Returns dict with fluence [G, Rd, Cd], planar_fluence [G, Rd, Cd],
    signal_per_group [G, Rd, Cd], signal [Rd, Cd].
    """
    _primary_samples_per_pixel_axis(projectors, samples_per_pixel_axis)
    if int(rho.shape[0]) != library.num_materials:
        raise ValueError(f"rho has {int(rho.shape[0])} materials, the library {library.num_materials}")
    present = present_materials(rho)
    present_rho = rho if present is None else rho[present]
    # [M', Rd n, Cd n] g/cm^2
    path_lengths = primary_path_lengths(projectors, present_rho, grid, geometry, angle, samples_per_pixel_axis)
    return primary_from_path_lengths(library, path_lengths, present, geometry, angle, source, detector, grid_response)


def scatter_fraction(scatter_signal: Tensor, primary_signal: Tensor) -> Tensor:
    """SF = S / (P + S) elementwise (0 where P + S = 0)."""
    denominator = primary_signal + scatter_signal
    safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return torch.where(denominator > 0, scatter_signal / safe_denominator, torch.zeros_like(denominator))
