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
`combined_scatter_image` evaluates the first-collision source S_FC at the sub-voxel points (the f^3 fine voxel
centers of each supervoxel, equal weights) and the multiple-collision source S_MC per supervoxel, attenuates their
sum with tau_down traced on the supervoxel grid plus its first-order variation inside the cell
(`supervoxel_subpoint_emission`), and projects the sub-point emission on the fine grid (each sub-point traced to
the panel separately), once per direction chunk (docs/source/design_notes.rst, "Scatter image").
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
`primary_image` is exact in energy: it projects the two density maps (g/cm^2) and sums the transmission over the
library's fine-energy table within each group (`cross_sections.group_transmission`), per sub-ray before the pixel
average, instead of exp(-int mu_t,g dl) with group-averaged mu_t (within-group beam hardening, -1.4 % at WLAB).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from .angular_quadrature import Quadrature, hemisphere_quadrature, real_spherical_harmonics
from .config import SolverConfig
from .cross_sections import MultigroupLibrary, group_transmission, mu_t_volume
from .detector import DetectorResponse, GridResponse, detector_angle, grid_angles
from .geometry import ConeGeometry, Grid
from .kernels import triton_utils as _tu
from .ray_tracing import Projectors
from .source import SourceModel
from .uncollided import fc_source_at_directions

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
    "primary_image",
    "scatter_fraction",
]

SourceFn = Callable[[Tensor], Tensor]
triton = _tu.triton
tl = _tu.tl


# =====================================================================================================
# direction sets
# =====================================================================================================

def covering_polar_angle(grid: Grid, geom: ConeGeometry, angle: float, margin: float = 1e-6) -> float:
    """Smallest polar angle about -n_hat containing every direction from a point of the grid box to a
    point of the active panel area (radians, plus `margin`, capped at pi/2).

    The polar angle of d = p - r is atan(|d_lateral| / d_normal); its sublevel sets are convex cones, so
    over the polytope {p - r} (a Minkowski difference of the box and the panel rectangle) the maximum is
    attained at a vertex, i.e. at a (panel corner) - (box corner) pair. Returns pi/2 if any box corner
    is not strictly on the source side of the panel plane.
    """
    lo = grid.lower_corner()
    ext = grid.extent
    corners = torch.tensor([[lo[0] + i * ext[0], lo[1] + j * ext[1], lo[2] + k * ext[2]]
                            for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=torch.float64)
    u_hat, v_hat, n_hat = (torch.tensor(a, dtype=torch.float64) for a in geom.panel_axes(angle))
    pc = torch.tensor(geom.panel_center(angle), dtype=torch.float64)
    hu, hv = geom.panel_half_extent()
    cu, cv = -geom.det_channel_offset, -geom.det_row_offset
    pcorn = torch.stack([pc + (cu + su * hu) * u_hat + (cv + sv * hv) * v_hat for su in (-1, 1) for sv in (-1, 1)])
    d = pcorn[:, None, :] - corners[None, :, :]  # [4, 8, 3]
    dn = (d * (-n_hat)).sum(-1)
    if bool((dn <= 0).any()):
        return math.pi / 2
    lat = torch.linalg.norm(d - dn[..., None] * (-n_hat), dim=-1)
    return float(min(math.pi / 2, float(torch.atan2(lat, dn).max()) + margin))


def panel_direction_set(geom: ConeGeometry, angle: float,
                        config_or_counts: Union[SolverConfig, tuple[int, int]], *,
                        max_polar_angle: Union[float, Grid] = math.pi / 2,
                        device=None, dtype: torch.dtype = torch.float32) -> Quadrature:
    """Collapsed-cone direction set about -n_hat (the directions that reach the panel).

    The Gauss-Legendre (in cos theta) x equal-azimuth product set of `angular_quadrature.hemisphere_quadrature`
    with config_or_counts = a SolverConfig (uses cone_num_polar, cone_num_azimuthal) or a (num_polar, num_azimuthal)
    tuple. max_polar_angle: cone half-angle in rad (default pi/2, the full hemisphere), or a Grid, in which case
    `covering_polar_angle(grid, geom, angle)` is used (exact: no lost contribution). The cell widths
    (`Quadrature.cell_width` [M, 2], gnomonic coordinates about -n_hat along (u_hat, v_hat)) used by the finite-cone
    footprint are always filled. Returns a Quadrature (omega [M, 3], weight [M] sr) on `device` with `dtype`.
    """
    if isinstance(config_or_counts, SolverConfig):
        num_polar, num_az = config_or_counts.cone_num_polar, config_or_counts.cone_num_azimuthal
    else:
        num_polar, num_az = (int(c) for c in config_or_counts)
    if isinstance(max_polar_angle, Grid):
        max_polar_angle = covering_polar_angle(max_polar_angle, geom, angle)
    u_hat, v_hat, n_hat = geom.panel_axes(angle)
    axis = [-c for c in n_hat]
    return hemisphere_quadrature(axis, num_polar, num_az, max_polar_angle, device=device, dtype=dtype,
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

    quad, geom, angle, detector, grid_response : what the context was built from (checked on use)
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
    acc_weight     separable: [G * Rd, 3, K], the fused accumulation weights (w_k, w_k cos(theta0_k),
                   signal_weight) laid out for one batched matmul per chunk (`accumulate`); else None
    chunk, bounds  directions per chunk and the (start, stop) of every chunk; omega_chunks / weight_chunks are the
                   matching views of omega / weight
    cell_width     [K, 2] or None: quad.cell_width of the kept directions (gnomonic widths of each direction's cell along
                   u_hat, v_hat; needed by the finite-cone footprint)

    Representation choice (exact, smallest): the detector response depends on the direction only through theta0 ->
    [K, G]; the grid response depends on (theta1, theta2), which for the flat panel vary with the direction and the
    detector ROW only -> [K, G, Rd] (per group because the table is energy dependent). The fused weights are
    [G * Rd, 3, K]: 2048 directions x 6 groups x 75 rows x 3 = 11 MB in float32, instead of 370 MB per weight for a
    per-pixel [K, G, Rd, Cd] table. All responses are evaluated in float64 and stored in the context dtype.

    Lazily cached per context: real spherical harmonics of the directions per order L (`spherical_harmonics`),
    the sub-cell offsets Omega_k . (r_j - r_cell) per (fine grid, factor) (`subcell_offsets`) and the panel heights of
    the finite-cone footprint per grid (`_Footprint`), so that repeated images of a view need no host synchronization.
    """

    quad: Quadrature
    geom: ConeGeometry
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
    acc_weight: Optional[Tensor]
    separable: bool
    chunk: int
    bounds: tuple[tuple[int, int], ...]
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
        ts = [t for t in (self.group_energy, self.omega, self.weight, self.cos_theta0, self.detector_weight,
                          self.grid_weight, self.signal_weight, self.acc_weight, self.cell_width) if t is not None]
        for v in self._cache.values():
            ts.extend(t for t in (v if isinstance(v, tuple) else (v,)) if isinstance(t, Tensor))
        for t in ts:
            st = t.untyped_storage()
            seen[st.data_ptr()] = st.nbytes()
        return int(sum(seen.values()))

    # ---- checks ----------------------------------------------------------------------------------
    def check(self, geom: ConeGeometry, angle: float, quad: Quadrature, detector: DetectorResponse,
              grid_response: GridResponse, num_groups: int, device: torch.device, dtype: torch.dtype) -> None:
        """Raise ValueError unless the context was built for these arguments (host-side checks only: identity of
        the direction set and the responses, geometry and angle equality, group count, device and dtype)."""
        bad = []
        if quad is not self.quad:
            bad.append("direction set (quad)")
        if geom != self.geom:
            bad.append("geometry")
        if abs(float(angle) - self.angle) > 1e-12:
            bad.append(f"angle ({angle} vs {self.angle})")
        if detector is not self.detector:
            bad.append("detector response")
        if grid_response is not self.grid_response:
            bad.append("grid response")
        if int(num_groups) != self.num_groups:
            bad.append(f"group count ({num_groups} vs {self.num_groups})")
        if torch.device(device) != self.device:
            bad.append(f"device ({device} vs {self.device})")
        if dtype != self.dtype:
            bad.append(f"dtype ({dtype} vs {self.dtype})")
        if bad:
            raise ValueError("DirectionSetContext does not match the image arguments: " + ", ".join(bad))

    # ---- lazily cached direction-only tables --------------------------------------------------------
    def spherical_harmonics(self, L: int) -> tuple[Tensor, ...]:
        """Real spherical harmonics of the context directions, one [k, (L+1)^2] view per chunk (cached per L)."""
        key = ("sh", int(L))
        hit = self._cache.get(key)
        if hit is None:
            Y = real_spherical_harmonics(int(L), self.omega)  # [K, (L+1)^2]
            hit = (Y,) + tuple(Y[s:e] for s, e in self.bounds)
            self._cache[key] = hit
        return hit[1:]

    def subcell_offsets(self, grid_fine: Grid, factor: int) -> tuple[Tensor, ...]:
        """Omega_k . (r_j - r_cell) for the f^3 fine voxel centers r_j of a supervoxel (`supervoxel_subpoint_emission`),
        one [k, f, f, f] view per chunk (axes: rows, cols, slices), cached per (fine voxel deltas, factor)."""
        f = int(factor)
        key = ("subcell", tuple(float(d) for d in grid_fine.deltas), f)
        hit = self._cache.get(key)
        if hit is None:
            proj = _subcell_offsets(self.omega, grid_fine, f)  # [K, f, f, f]
            hit = (proj,) + tuple(proj[s:e] for s, e in self.bounds)
            self._cache[key] = hit
        return hit[1:]

    # ---- accumulation ------------------------------------------------------------------------------
    def new_accumulator(self) -> Tensor:
        """Zeroed image accumulator: [G * Rd, 3, Cd] (separable) or [3, G, Rd, Cd]."""
        G, (Rd, Cd) = self.num_groups, self.image_shape
        if self.separable:
            return torch.zeros(G * Rd, 3, Cd, device=self.device, dtype=self.dtype)
        return torch.zeros(3, G, Rd, Cd, device=self.device, dtype=self.dtype)

    def accumulate(self, acc: Tensor, img: Tensor, i: int) -> None:
        """acc += the weighted chunk image: img [k, G, Rd, Cd] = the projected attenuated emission of chunk i in
        photons/(cm^2 sr). Separable: ONE batched matmul (baddbmm, no copy when img is contiguous; cuBLAS picks a
        small-N kernel, ~7 us per chunk on the WLAB panel):
        acc[(g, r), j, c] += sum_k W[(g, r), j, k] img[k, g, r, c], j = (fluence, planar, signal)."""
        s, e = self.bounds[i]
        k = e - s
        G, (Rd, Cd) = self.num_groups, self.image_shape
        if self.separable:
            a = img.permute(1, 2, 0, 3).reshape(G * Rd, k, Cd)  # a view for a contiguous img
            acc.baddbmm_(self.acc_weight[:, :, s:e], a)
            return
        w = self.weight_chunks[i]
        acc[0].add_(torch.einsum("kgrc,k->grc", img, w))
        acc[1].add_(torch.einsum("kgrc,k->grc", img, w * self.cos_theta0[s:e]))
        acc[2].add_((img * self.signal_weight[s:e]).sum(0))

    def finalize(self, acc: Tensor) -> dict[str, Tensor]:
        """The image dict of `scatter_image_from_source` from an accumulator."""
        G, (Rd, Cd) = self.num_groups, self.image_shape
        if self.separable:
            acc = acc.reshape(G, Rd, 3, Cd).permute(2, 0, 1, 3)
        fluence, planar, spg = (acc[j].contiguous() for j in range(3))
        return {"fluence": fluence, "planar_fluence": planar, "signal_per_group": spg, "signal": spg.sum(0)}


def _subcell_offsets(omega: Tensor, grid_fine: Grid, f: int) -> Tensor:
    """Omega_k . (r_j - r_cell) [K, f_r, f_c, f_s] for the f^3 fine voxel centers of a supervoxel; omega [K, 3]."""
    dev, dt = omega.device, omega.dtype
    dx, dy, dz = grid_fine.deltas
    idx = torch.arange(f, device=dev, dtype=dt) - (f - 1) / 2.0
    om = omega.reshape(-1, 3)
    return (om[:, 1, None, None, None] * (dy * idx)[None, :, None, None]
            + om[:, 0, None, None, None] * (dx * idx)[None, None, :, None]
            + om[:, 2, None, None, None] * (dz * idx)[None, None, None, :])


def make_direction_context(quad: Quadrature, geom: ConeGeometry, angle: float, detector: DetectorResponse,
                           grid_response: GridResponse, lib: Optional[MultigroupLibrary] = None,
                           device=None, dtype: Optional[torch.dtype] = None, *,
                           group_energy: Optional[Tensor] = None, chunk: int = 8,
                           factorize: bool = True) -> DirectionSetContext:
    """Precompute the emission-independent part of a collapsed-cone image for one (view, direction set).

    quad: direction set (omega [M, 3], weight [M]); only the panel-facing directions are kept (K of them, selected
      here once, with the only host synchronization of the context's life).
    lib / group_energy: group energies [G] keV for the response lookups (group_energy wins).
    device, dtype: of the images the context will serve (default: quad's device, float32 / quad's float dtype).
    chunk: directions per chunk of the image loop (the context's chunking is used by every image
      it serves).
    factorize: store the grid response per detector row ([K, G, Rd, 1], exact for the flat panel; checked here
      numerically, a non-separable panel falls back automatically) and accumulate with one batched matmul per
      chunk; False keeps a per-pixel [K, G, Rd, Cd] table (reference path, 4 x K x G x pixels bytes).
    No grid argument: nothing in the per-chunk response work depends on the voxel grid, so one context serves every
    image of a view, on any grid. Grid-dependent tables (sub-cell offsets) are cached lazily by grid.
    Returns a DirectionSetContext (its docstring has the shapes and the representation).
    """
    if group_energy is None:
        if lib is None:
            raise ValueError("pass lib or group_energy")
        group_energy = lib.group_energy
    dev = torch.device(device) if device is not None else quad.omega.device
    if dtype is None:
        dtype = quad.omega.dtype if quad.omega.dtype.is_floating_point else torch.float32
    f64 = torch.float64
    e64 = group_energy.detach().to(dev, f64).reshape(-1)
    G = int(e64.numel())
    om64 = quad.omega.detach().to(dev, f64)
    w64 = quad.weight.detach().to(dev, f64)
    _, _, n_hat = geom.panel_axes(angle)
    cos64 = (om64 * (-torch.tensor(n_hat, device=dev, dtype=f64))).sum(-1)
    # the same selection as before in the dtype of the images (a direction grazing the panel plane must be kept or
    # dropped exactly as the projectors see it)
    cos_dt = (quad.omega.detach().to(dev, dtype) * (-torch.tensor(n_hat, device=dev, dtype=dtype))).sum(-1)
    keep = torch.nonzero(cos_dt > 0).reshape(-1)
    om64, w64, cos64 = om64[keep], w64[keep], cos64[keep]
    K = int(om64.shape[0])
    pix = geom.pixel_centers(angle, dev, f64)  # [Rd, Cd, 3]
    Rd, Cd = int(pix.shape[0]), int(pix.shape[1])
    fd = grid_response.focal_distance_cm
    # detector response: theta0 per direction -> [G, K]
    d_resp = detector.at(e64, detector_angle(om64, geom, angle)) if K else torch.zeros(G, 0, device=dev, dtype=f64)
    # grid response: the grid angles of one probe direction over the full panel decide separability
    separable = False
    if factorize:
        probe = torch.tensor(n_hat, device=dev, dtype=f64)[None].neg()
        if K:
            probe = torch.cat([probe, om64[:1]])
        th1p, th2p = grid_angles(probe, pix, geom, angle, fd)  # [P, Rd, Cd]
        dev_max = torch.maximum((th1p - th1p[..., :1]).abs().max(), (th2p - th2p[..., :1]).abs().max())
        separable = bool(dev_max <= 1e-10)  # rad; float64 rounding of the pixel positions is ~1e-17
    cols = pix[:, Cd // 2:Cd // 2 + 1] if separable else pix  # [Rd, Cg, 3]
    Cg = int(cols.shape[1])
    d_w = d_resp.transpose(0, 1)  # [K, G]
    c = cos64
    # grid response and signal weights, evaluated in float64 over blocks of directions (bounded temporaries),
    # stored in the image dtype
    g_resp = torch.empty(K, G, Rd, Cg, device=dev, dtype=dtype) if separable else None
    sig = torch.empty(K, G, Rd, Cg, device=dev, dtype=dtype)
    block = max(1, (1 << 22) // max(1, G * Rd * Cg))
    for s in range(0, K, block):
        e = min(s + block, K)
        th1, th2 = grid_angles(om64[s:e], cols, geom, angle, fd)  # [b, Rd, Cg]
        gb = grid_response.at(e64, th1, th2).permute(1, 0, 2, 3)  # [b, G, Rd, Cg]
        if g_resp is not None:
            g_resp[s:e] = gb
        sig[s:e] = (w64[s:e] * c[s:e])[:, None, None, None] * d_w[s:e, :, None, None] * gb
        del th1, th2, gb
    acc_w = None
    if separable:
        acc_w = torch.empty(G, Rd, 3, K, device=dev, dtype=dtype)
        acc_w[:, :, 0] = w64.to(dtype)
        acc_w[:, :, 1] = (w64 * cos64).to(dtype)
        acc_w[:, :, 2] = sig[..., 0].permute(1, 2, 0)
        acc_w = acc_w.reshape(G * Rd, 3, K)
    chunk = max(1, int(chunk))
    bounds = tuple((s, min(s + chunk, K)) for s in range(0, K, chunk))
    omega = om64.to(dtype).contiguous()
    weight = w64.to(dtype).contiguous()
    cw = getattr(quad, "cell_width", None)
    cw = None if cw is None else cw.detach().to(dev, dtype)[keep].contiguous()
    return DirectionSetContext(
        quad=quad, geom=geom, angle=float(angle), detector=detector, grid_response=grid_response,
        group_energy=e64.to(dtype), omega=omega, weight=weight,
        cos_theta0=cos64.to(dtype), detector_weight=d_w.to(dtype).contiguous(), grid_weight=g_resp,
        signal_weight=sig, acc_weight=acc_w, separable=separable, chunk=chunk,
        bounds=bounds, omega_chunks=tuple(omega[s:e] for s, e in bounds),
        weight_chunks=tuple(weight[s:e] for s, e in bounds), image_shape=(Rd, Cd), cell_width=cw)


def _resolve_context(context: Optional[DirectionSetContext], quad: Quadrature, geom: ConeGeometry, angle: float,
                     detector: DetectorResponse, grid_response: GridResponse, lib: Optional[MultigroupLibrary],
                     mu_t: Tensor, chunk: int, group_energy: Optional[Tensor]) -> DirectionSetContext:
    """The given context (checked against the arguments) or a new one built for them on mu_t's device / dtype."""
    G = int(mu_t.shape[0])
    if context is None:
        if group_energy is None and lib is None:
            raise ValueError("pass lib or group_energy")
        context = make_direction_context(quad, geom, angle, detector, grid_response, lib, mu_t.device,
                                         mu_t.dtype, group_energy=group_energy, chunk=chunk)
        if context.num_groups != G:
            raise ValueError(f"group_energy has {context.num_groups} entries, mu_t has {G} groups")
        return context
    context.check(geom, angle, quad, detector, grid_response, G, mu_t.device, mu_t.dtype)
    return context


# =====================================================================================================
# collapsed-cone scatter image
# =====================================================================================================

def scatter_image_from_source(source_fn: SourceFn, grid: Grid, geom: ConeGeometry, angle: float,
                              quad: Quadrature, mu_t: Tensor, projectors: Projectors,
                              detector: DetectorResponse, grid_response: GridResponse,
                              lib: Optional[MultigroupLibrary] = None, chunk: int = 8,
                              group_energy: Optional[Tensor] = None,
                              context: Optional[DirectionSetContext] = None, *, footprint: bool = True) -> dict[str, Tensor]:
    """Collapsed-cone image of a last-collided emission source on ONE grid (module docstring).

    The generic single-grid form of the collapsed-cone image (the solver's pass is `combined_scatter_image`; this
    function serves tests and diagnostics with an arbitrary emission).
    source_fn(omegas [k, 3]) -> [k, G, rows, cols, slices]: emission density S_g(r, Omega_k) at the
      voxel centers of `grid` in photons/(cm^3 sr) per source photon/sr (treated as constant per voxel).
    mu_t: [G, rows, cols, slices] (1/cm) on `grid`, used for the attenuation to the panel.
    quad: direction set (e.g. `panel_direction_set`); directions with Omega . (-n_hat) <= 0 are skipped
      (they never reach the panel).
    group_energy [G] keV for the response lookups: taken from `lib.group_energy` unless given.
    chunk: directions per batch (peak memory ~ 3 chunk G |grid| scalars plus the projector's working set).
    context: optional `make_direction_context(quad, geom, angle, detector, grid_response, lib, ...)` of
      this view (built here when None). It holds the panel-facing directions, their chunking (its `chunk` replaces
      the argument) and the precomputed response weights; it must match the other arguments (checked).
    footprint: apply the finite-cone footprint (`_Footprint`, heights weighted by this emission; needs
      quad.cell_width); False projects every direction as a single ray per pixel.

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
    ctx = _resolve_context(context, quad, geom, angle, detector, grid_response, lib, mu_t, chunk, group_energy)
    G = mu_t.shape[0]
    dev, dt = mu_t.device, mu_t.dtype
    fp = _Footprint(grid, geom, angle, ctx) if footprint else None
    acc = ctx.new_accumulator()
    for i, om in enumerate(ctx.omega_chunks):
        k = om.shape[0]
        src = source_fn(om)
        if tuple(src.shape) != (k, G, *grid.shape):
            raise ValueError(f"source_fn returned {tuple(src.shape)}, expected {(k, G, *grid.shape)}")
        tau = projectors.downstream_cumulative(mu_t, grid, geom, angle, om)  # [k, G, R, C, S]
        emis = tau.neg_().exp_().mul_(src.to(dev, dt))  # attenuated to the panel
        del src, tau
        img = projectors.oblique_project(emis, grid, geom, angle, om)  # [k, G, Rd, Cd] photons/(cm^2 sr)
        if fp is not None:
            img = fp.blur(img, i, fp.mean_height(emis))
        del emis
        ctx.accumulate(acc, img, i)
        del img
    return ctx.finalize(acc)


# =====================================================================================================
# sub-voxel quadrature points of the supervoxel (transport) grid
# =====================================================================================================

def _check_subpoint_shapes(src_fine: Tensor, tau_coarse: Tensor, grid_fine: Grid, f: int) -> tuple[int, int, int]:
    """Validate src_fine [k, G, R, C, S] on grid_fine and tau_coarse [k, G, Rc, Cc, Sc]; returns (Rc, Cc, Sc)."""
    coarse = grid_fine.coarsen(f)
    k, G = src_fine.shape[:2]
    Rc, Cc, Sc = coarse.shape
    if tuple(src_fine.shape[2:]) != tuple(grid_fine.shape):
        raise ValueError(f"src_fine spatial shape {tuple(src_fine.shape[2:])} != fine grid {grid_fine.shape}")
    if tuple(tau_coarse.shape) != (k, G, Rc, Cc, Sc):
        raise ValueError(f"tau_coarse {tuple(tau_coarse.shape)} != {(k, G, Rc, Cc, Sc)}")
    return Rc, Cc, Sc


def supervoxel_subpoint_emission(src_fine: Tensor, tau_coarse: Tensor, mu_t_coarse: Tensor, omegas: Tensor,
                                 grid_fine: Grid, factor: int, *, offsets: Optional[Tensor] = None) -> Tensor:
    """Attenuated emission at every sub-voxel point, kept on the FINE grid (before the projection and the sum over j).

    The sub-voxel points r_j of a supervoxel of C = grid_fine.coarsen(factor) are its f^3 fine voxel centers (equal
    weights 1/f^3):

        e[k, g, r_j] = S[k, g, r_j] exp(-tau_j),
        tau_j = max(0, tau_coarse[k, g, cell] - mu_t_coarse[g, cell] (Omega_k . (r_j - r_cell)))

    i.e. the downstream attenuation at the supervoxel center plus its first-order variation inside the cell (moving
    along Omega_k toward the panel shortens the path); exact for a uniform cell whose rays leave through the same
    face as the center ray. Projecting this field on grid_fine traces each sub-point to the panel separately.
    src_fine [k, G, R, C, S] on grid_fine; tau_coarse [k, G, Rc, Cc, Sc] on C (projectors.downstream_cumulative;
    +inf allowed); mu_t_coarse [G, Rc, Cc, Sc] (1/cm); omegas [k, 3]. Returns e [k, G, R, C, S] in
    photons/(cm^3 sr) on grid_fine, in the device / dtype of tau_coarse. For factor 1 this is S exp(-tau) (no
    clamp). Peak extra memory ~ one fine-grid field [k, G, R, C, S]. offsets: optional precomputed
    Omega_k . (r_j - r_cell) [k, f, f, f] (`DirectionSetContext.subcell_offsets`), computed here when None.
    """
    f = int(factor)
    k, G = src_fine.shape[:2]
    Rc, Cc, Sc = _check_subpoint_shapes(src_fine, tau_coarse, grid_fine, f)
    dev, dt = tau_coarse.device, tau_coarse.dtype
    src_fine = src_fine.to(dev, dt)
    if f == 1:
        return torch.exp(-tau_coarse).mul_(src_fine)
    s = src_fine.reshape(k, G, Rc, f, Cc, f, Sc, f)
    # offsets of the fine centers from the supervoxel center along x (cols), y (rows), z (slices)
    if offsets is None:
        proj = _subcell_offsets(omegas.to(dev, dt).reshape(k, 3), grid_fine, f)  # [k, f_r, f_c, f_s]
    else:
        if tuple(offsets.shape) != (k, f, f, f):
            raise ValueError(f"offsets {tuple(offsets.shape)} != {(k, f, f, f)}")
        proj = offsets.to(dev, dt)
    proj = proj[:, None, None, :, None, :, None, :]  # [k, 1, 1, f, 1, f, 1, f]
    mu = mu_t_coarse.to(dev, dt)[None, :, :, None, :, None, :, None]  # [1, G, Rc, 1, Cc, 1, Sc, 1]
    # tau_j = max(0, tau - mu (Omega . (r_j - r_c)))  [k, G, Rc, f, Cc, f, Sc, f]
    t = torch.addcmul(tau_coarse[:, :, :, None, :, None, :, None], mu, proj, value=-1.0).clamp_(min=0.0)
    return t.neg_().exp_().mul_(s).reshape(src_fine.shape)


def coarsen_mu_t(mu_t_fine: Tensor, factor: int) -> Tensor:
    """Supervoxel average of a fine attenuation map [G, R, C, S] -> [G, R/f, C/f, S/f] (1/cm).

    Because mu_t is linear in the densities this equals mu_t_volume(lib, average-pooled rho).
    """
    f = int(factor)
    if f == 1:
        return mu_t_fine
    return F.avg_pool3d(mu_t_fine.unsqueeze(0), kernel_size=f, stride=f).squeeze(0)


# =====================================================================================================
# finite-cone footprint
# =====================================================================================================

def box_blur_rows_cols(img: Tensor, width_c: Tensor, width_r: Tensor) -> Tensor:
    """Separable box filter of img [k, G, Rd, Cd] with a per-image width (in PIXELS, fractional allowed) along the
    channels (width_c [k]) and the rows (width_r [k]): out[c] = (1/|W|) int_W x over the window W = [c + 1/2 - w/2,
    c + 1/2 + w/2] clipped to the panel [0, Cd] (pixel j covers [j, j + 1]; x is piecewise constant; the window is
    renormalized at the panel edges). A width <= 1 pixel leaves a pixel unchanged (the window stays inside the pixel).
    Done as two batched matmuls with the [k, N, N] window-overlap matrices (`box_matrix`; ~0.1 ms per chunk of 8
    directions on the 184 x 150 panel, cheaper than gathers from a cumulative sum). Returns [k, G, Rd, Cd]."""
    k, G, Rd, Cd = img.shape
    Bc = box_matrix(Cd, width_c, img.dtype)  # [k, Cd, Cd]
    Br = box_matrix(Rd, width_r, img.dtype)  # [k, Rd, Rd]
    out = torch.bmm(img.reshape(k, G * Rd, Cd), Bc.transpose(1, 2)).view(k, G, Rd, Cd)
    return torch.matmul(Br[:, None], out)


def box_matrix(N: int, w: Tensor, dtype: Optional[torch.dtype] = None) -> Tensor:
    """[k, N, N] matrices of the 1-D box filter of box_blur_rows_cols: B[c, j] = |[j, j + 1] n W_c| / |W_c|, W_c =
    [c + 1/2 - w/2, c + 1/2 + w/2] n [0, N], for widths w [k] (pixels; clamped below at 1e-3). float32 CUDA: one
    Triton launch (`_box_matrix_kernel`; the PyTorch expression below takes ~17 launches, ~0.3 ms per call)."""
    dt = dtype if dtype is not None else w.dtype
    if dt == torch.float32 and w.is_cuda and _tu.triton_available()[0]:
        wk = w.to(torch.float32).contiguous()
        k = int(wk.numel())
        out = torch.empty(k, N, N, device=w.device, dtype=torch.float32)
        if k:
            _tu.launch(_box_matrix_kernel, (k, int(N), 1), (wk, out, int(N)),
                       {"BLOCK": max(16, triton.next_power_of_2(int(N)))}, 4)
        return out
    centers = torch.arange(N, device=w.device, dtype=dt) + 0.5
    half = 0.5 * w.to(dt).clamp_min(1e-3)[:, None]
    lo = (centers[None] - half).clamp(0.0, float(N))[..., None]  # [k, N, 1]
    hi = (centers[None] + half).clamp(0.0, float(N))[..., None]
    j = torch.arange(N, device=w.device, dtype=dt)[None, None, :]
    ov = (torch.minimum(hi, j + 1.0) - torch.maximum(lo, j)).clamp_min(0.0)
    return ov / (hi - lo)


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


def panel_height_field(grid: Grid, geom: ConeGeometry, angle: float, device=None,
                       dtype: torch.dtype = torch.float32) -> Tensor:
    """Normal distance of every voxel center from the panel plane, h(r) = (r - panel_center) . n_hat [R, C, S] (cm,
    positive on the source side). A ray from r along a direction with gnomonic coordinates (alpha, beta) about -n_hat
    hits the panel at the foot of r plus h (alpha, beta) (along u_hat, v_hat): the footprint of a direction CELL of
    gnomonic widths (W_alpha, W_beta) is an h W_alpha x h W_beta rectangle."""
    _, _, n_hat = geom.panel_axes(angle)
    r = grid.center_coordinates(device, torch.float64)
    pc = torch.tensor(geom.panel_center(angle), device=device, dtype=torch.float64)
    return ((r - pc) @ torch.tensor(n_hat, device=device, dtype=torch.float64)).to(dtype)


class _Footprint:
    """Finite-cone footprint of the direction chunks of one image (combined_scatter_image / scatter_image_from_source;
    docs/source/design_notes.rst, "Scatter image").

    Direction k stands for a solid-angle cell of gnomonic widths (W_alpha,k, W_beta,k) about -n_hat (`ctx.cell_width`).
    A point at height h above the panel sends that cell onto an h W_alpha x h W_beta rectangle of the panel, so the
    projected single-ray image of direction k is box-blurred with widths hbar_k (W_alpha,k, W_beta,k) / pixel, hbar_k =
    the emission-weighted mean panel height of the attenuated emission of the image along Omega_k (`mean_height`). A
    direction whose emission vanishes uses the height of the box centre. Exact up to the height spread of the emission
    and the cell shape (a box with the same marginal widths as the polar x azimuth cell)."""

    def __init__(self, grid: Grid, geom: ConeGeometry, angle: float, ctx: DirectionSetContext) -> None:
        """grid: grid of the emission that is projected."""
        if ctx.cell_width is None:
            raise ValueError("the finite-cone footprint needs quad.cell_width (panel_direction_set fills it)")
        self.ctx = ctx
        key = ("footprint", grid)
        hit = ctx._cache.get(key)  # the tables depend on the grid only (geom and angle are those of the context)
        if hit is None:
            dev, dt = ctx.device, ctx.dtype
            scale = torch.tensor([1.0 / geom.delta_det_channel, 1.0 / geom.delta_det_row], device=dev, dtype=dt)
            h = panel_height_field(grid, geom, angle, dev, dt).reshape(-1)  # [voxels]
            # height of the box centre from its corners (host arithmetic)
            _, _, n_hat = geom.panel_axes(angle)
            pc = geom.panel_center(angle)
            lo, ext = grid.lower_corner(), grid.extent
            hs = [sum((lo[a] + c[a] * ext[a] - pc[a]) * n_hat[a] for a in range(3))
                  for c in [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]]
            hit = (scale, h, 0.5 * (min(hs) + max(hs)))
            ctx._cache[key] = hit
        self.scale, self.h, self.h_center = hit

    def mean_height(self, e: Tensor) -> Tensor:
        """Emission-weighted mean panel height [k] of an attenuated emission e [k, G, *grid] (the box centre where the
        emission vanishes)."""
        k = int(e.shape[0])
        tot = e.sum(dim=1).reshape(k, -1)  # [k, voxels]
        num, den = tot @ self.h, tot.sum(-1)
        tiny = torch.finfo(den.dtype).tiny
        return torch.where(den > 0, num / den.clamp_min(tiny), torch.full_like(den, self.h_center))

    def blur(self, img: Tensor, i: int, hbar: Tensor) -> Tensor:
        """Footprint image [k, G, Rd, Cd] of the single-ray image img of chunk i with mean heights hbar [k]."""
        s, e = self.ctx.bounds[i]
        w = hbar[:, None] * self.ctx.cell_width[s:e] * self.scale  # [k, 2] pixels (channels, rows)
        return box_blur_rows_cols(img, w[:, 0], w[:, 1])


# =====================================================================================================
# the scatter image: first-order + multiple-collision emission, traced together
# =====================================================================================================

def combined_scatter_image(lib: MultigroupLibrary, rho_fine: Tensor, phi_uc_fine: Tensor, grid_fine: Grid,
                           factor: int, T_mean: Tensor, L: int, geom: ConeGeometry, angle: float, quad: Quadrature,
                           mu_t_coarse: Optional[Tensor], projectors: Projectors, detector: DetectorResponse,
                           grid_response: GridResponse, context: Optional[DirectionSetContext] = None,
                           chunk: int = 8, *, fc_chunk: int = 4, footprint: bool = True) -> dict[str, Tensor]:
    """Total scatter image (first-order + multiple-collision) in ONE collapsed-cone pass: the first- and
    multiple-collision sources are summed and ray-traced to the detector together, on the supervoxel grid with
    sub-voxel quadrature points.

    lib; rho_fine [M, R, C, S] g/cm^3 and phi_uc_fine [G, R, C, S] (uncollided.uncollided_fluence) on grid_fine, whose
    voxel centers are the f^3 equal-weight sub-voxel points of each supervoxel of C = grid_fine.coarsen(factor);
    T_mean [G, (L+1)^2, Rc, Cc, Sc]: cell-mean SH moments of the multiple-collision emission (already contracted with
    the transfer moments, `ScatterSolver.mc_source_moments`), or the 6-D [G, (L+1)^2, 4, Rc, Cc, Sc] tensor whose LD
    mean is used; mu_t_coarse [G, Rc, Cc, Sc] on C (None -> coarsen_mu_t(mu_t_volume(lib, rho_fine))). context: the
    view's `make_direction_context` (built here when None; its chunking replaces `chunk`). fc_chunk: directions per
    inner batch of the FC source (PyTorch path).

    Per direction chunk (Omega_k, k of them):
      tau    = downstream_cumulative(mu_t_coarse, C, Omega_k)   [k, G, Rc, Cc, Sc]
      S_FC,j = fc_source_at_directions at the fine voxel centers r_j (exact eta table)    [k, G, R, C, S]
      S_MC   = Y(Omega_k) . T per supervoxel (one matmul with the cached harmonics), broadcast to its sub-points
      e_j    = (S_FC,j + S_MC) exp(-tau_j)   (`supervoxel_subpoint_emission`: tau at the supervoxel center plus its
               in-cell variation)
    ONE oblique_project of e on the fine grid (each sub-point traced to the panel separately; the attenuation stays
    on C), the finite-cone footprint blur (`_Footprint`, footprint=True) and ONE fused accumulation
    (`DirectionSetContext.accumulate`). Factor 1 is the single-grid image `scatter_image_from_source` of S_FC + S_MC.
    The image is linear in (phi_uc, T_mean) apart from the footprint heights, which follow the summed emission.

    Returns the image dict of the total scatter (fluence, planar_fluence, signal_per_group [G, Rd, Cd],
    signal [Rd, Cd]). Peak extra memory per chunk ~ 2-3 fine-grid fields [k, G, R, C, S].
    """
    dev, dt = phi_uc_fine.device, phi_uc_fine.dtype
    rho = rho_fine.to(device=dev, dtype=dt)
    f = int(factor)
    coarse = grid_fine.coarsen(f)
    mu_c = coarsen_mu_t(mu_t_volume(lib, rho), f) if mu_t_coarse is None else mu_t_coarse.to(device=dev, dtype=dt)
    G = int(mu_c.shape[0])
    Rc, Cc, Sc = coarse.shape
    if tuple(mu_c.shape[1:]) != tuple(coarse.shape):
        raise ValueError(f"mu_t_coarse spatial shape {tuple(mu_c.shape[1:])} != coarse grid {coarse.shape}")
    if T_mean.dim() == 6:
        T_mean = T_mean[:, :, 0]
    if T_mean.dim() != 5 or tuple(T_mean.shape[2:]) != tuple(coarse.shape) or int(T_mean.shape[0]) != G:
        raise ValueError(f"T_mean must be [G={G}, N_sh, {Rc}, {Cc}, {Sc}], got {tuple(T_mean.shape)}")
    if T_mean.shape[1] != (L + 1) ** 2:
        raise ValueError(f"T_mean has {T_mean.shape[1]} moments, expected (L+1)^2 = {(L + 1) ** 2}")
    ctx = _resolve_context(context, quad, geom, angle, detector, grid_response, lib, mu_c, chunk, None)
    offs = ctx.subcell_offsets(grid_fine, f) if f > 1 else None
    Y = ctx.spherical_harmonics(L)  # per chunk [k, N_sh]
    N = int(T_mean.shape[1])
    T_nm = T_mean.to(device=dev, dtype=dt).transpose(0, 1).reshape(N, -1)  # [N_sh, G * cells], once per image
    fine_shape = tuple(grid_fine.shape)
    fp = _Footprint(grid_fine, geom, angle, ctx) if footprint else None

    acc = ctx.new_accumulator()
    for i, om in enumerate(ctx.omega_chunks):
        k = int(om.shape[0])
        src = fc_source_at_directions(lib, rho, phi_uc_fine, grid_fine, geom, angle, om, chunk=fc_chunk)
        if tuple(src.shape) != (k, G, *fine_shape):
            raise ValueError(f"FC source {tuple(src.shape)} != {(k, G, *fine_shape)}")
        tau = projectors.downstream_cumulative(mu_c, coarse, geom, angle, om)  # [k, G, Rc, Cc, Sc]
        s_mc = torch.mm(Y[i], T_nm).view(k, G, Rc, Cc, Sc)  # S_MC(Omega_k) per supervoxel
        # one source at the sub-points, one attenuation, one projection
        both = src.reshape(k, G, Rc, f, Cc, f, Sc, f).add(s_mc[:, :, :, None, :, None, :, None]).view(k, G, *fine_shape)
        del src, s_mc
        e = supervoxel_subpoint_emission(both, tau, mu_c, om, grid_fine, f, offsets=None if offs is None else offs[i])
        del both, tau
        img = projectors.oblique_project(e, grid_fine, geom, angle, om)  # [k, G, Rd, Cd] photons/(cm^2 sr)
        if fp is not None:
            img = fp.blur(img, i, fp.mean_height(e))
        del e
        ctx.accumulate(acc, img, i)
        del img
    return ctx.finalize(acc)


# =====================================================================================================
# primary
# =====================================================================================================

def primary_image(lib: MultigroupLibrary, rho: Tensor, grid: Grid, geom: ConeGeometry, angle: float,
                  source: SourceModel, projectors: Projectors, detector: DetectorResponse,
                  grid_response: GridResponse, n_sub: Optional[int] = None) -> dict[str, Tensor]:
    """Primary image, per group at the pixels, exact in energy:

        fluence_g(u)        = b(Omega_p) / |r_D - r_0|^2 T_g(u)                                photons/cm^2
        T_g(u)              = < sum_{E in g} w_E exp(-(P_w m_w(E) + P_b m_b(E))) >_pixel
        planar_fluence_g(u) = cos(theta0(u)) fluence_g(u)                                     photons/cm^2
        signal_g(u)         = planar_fluence_g(u) G_g(theta1, theta2) D_g(theta0)            keV/cm^2

    P_m = int_{r_0}^{r_D} rho_m dl: line integrals of the two density maps (g/cm^2; two projections,
    `projectors.primary_project(rho)`), w_E / m_m(E) the library's fine-energy table (lib.fine_weight,
    lib.fine_mu_t; `cross_sections.group_transmission`). fluence_g is the fine-energy sum within group g, so the
    per-group detector and grid responses (at lib.group_energy) apply, and fluence.sum(0) /
    planar_fluence.sum(0) is the exact polychromatic primary (up to the fine quadrature; the group-averaged
    exp(-int mu_t,g dl) ignores within-group beam hardening).
    theta0(u) = angle between Omega_p and the inward panel normal (per incident photon responses, module docstring).

    with Omega_p the direction focal spot -> pixel center. The grid angles come from `grid_angles` for the
    primary direction (theta1 = 0 when the grid is focused at the focal spot; theta2 = -atan(u/SDD)).
    n_sub: sub-pixel sampling per axis (None: the projector's own `n_sub` attribute, 1 if it has none). With
    n > 1 the density line integrals are traced to the n x n midpoint lattice of sub-pixel centers (a panel of
    n-times finer pixels), the fine-energy transmission is formed per sub-ray and THEN averaged over the pixel
    (<.>_pixel above: the exact pixel-averaged transmission, no -log(mean(exp)) per group); the 1/r^2 and b
    factors are evaluated at the pixel center.
    rho [M, rows, cols, slices]. Returns dict with fluence [G, Rd, Cd], planar_fluence [G, Rd, Cd],
    signal_per_group [G, Rd, Cd], signal [Rd, Cd].
    """
    dev, dt = rho.device, rho.dtype
    n = int(n_sub) if n_sub is not None else int(getattr(projectors, "n_sub", 1) or 1)
    if n < 1:
        raise ValueError("n_sub must be >= 1")
    Rd, Cd = geom.num_det_rows, geom.num_det_channels
    geom_sub = geom if n == 1 else replace(geom, num_det_rows=Rd * n, num_det_channels=Cd * n,
                                           delta_det_row=geom.delta_det_row / n,
                                           delta_det_channel=geom.delta_det_channel / n)
    # sub-pixel (i, j) of pixel (r, c) is pixel (r n + i, c n + j) of geom_sub: same panel, same offsets
    if getattr(projectors, "n_sub", None) is not None:  # disable the projector's own sub-sampling
        path = projectors.primary_project(rho, grid, geom_sub, angle, n_sub=1)  # [M, Rd n, Cd n] g/cm^2
    else:
        path = projectors.primary_project(rho, grid, geom_sub, angle)
    trans = group_transmission(lib, path)  # [G, Rd n, Cd n]
    if n > 1:
        trans = trans.reshape(-1, Rd, n, Cd, n).mean(dim=(2, 4))
    pix = geom.pixel_centers(angle, dev, dt)
    r0 = torch.tensor(geom.source_position(angle), device=dev, dtype=dt)
    d = pix - r0
    dist2 = (d * d).sum(-1)
    b = source.angular_weight(d, geom, angle)
    fluence = (b / dist2)[None] * trans
    omega_p = d / dist2.sqrt()[..., None]
    e_g = lib.group_energy.to(dev, dt)
    theta0 = detector_angle(omega_p, geom, angle)  # [Rd, Cd]
    planar = fluence * torch.cos(theta0)
    d_resp = detector.at(e_g, theta0)  # [G, Rd, Cd]
    th1, th2 = grid_angles(omega_p, pix, geom, angle, grid_response.focal_distance_cm)  # [Rd, Cd]
    g_resp = grid_response.at(e_g, th1, th2)
    spg = planar * g_resp * d_resp
    return {"fluence": fluence, "planar_fluence": planar, "signal_per_group": spg, "signal": spg.sum(0)}


def scatter_fraction(scatter_signal: Tensor, primary_signal: Tensor) -> Tensor:
    """SF = S / (P + S) elementwise (0 where P + S = 0)."""
    den = primary_signal + scatter_signal
    return torch.where(den > 0, scatter_signal / torch.where(den > 0, den, torch.ones_like(den)),
                       torch.zeros_like(den))
