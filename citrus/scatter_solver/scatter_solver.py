"""The assembled LBTE source iteration scatter solver: uncollided fluence, source iteration, collapsed-cone scatter
image and primary image for one view.

Pipeline of `ScatterSolver.solve_view(rho_fine, grid_fine, angle)` (docs/source/method.rst has the derivation; grids
in docs/source/conventions.rst, "Grids"):

    rho_fine [M, R, C, S]  fine (reconstruction) grid F; M = 7 basis materials
    grid_coarse = Grid.covering(grid_fine, transport_cell_cm)          transport grid C (cubic cells, covers F)
    grid_quad   = grid_coarse.refine(q)                                quadrature grid Q, q = round(transport_cell_cm /
                                                                       quadrature_point_spacing_cm) bounded by the data
                                                                       resolution
    rho_coarse, rho_quad = conservative overlap resampling of rho_fine onto C and Q (`geometry.resample_conservative`)
    pairs_quad  = material_pairs_if_compact(rho_quad)                  pair or dense first-collision evaluation
    mu_t_coarse, mu_s_l_coarse from rho_coarse                        linear in rho: = averaged mu
    phi_uc         = uncollided_fluence at the Q points, traced through rho_quad on Q (uncollided_trace_grid
                     "quadrature", the default), rho_coarse on C ("coarse") or rho_fine on F ("fine"); exact in
                     energy: line integrals of the density maps of the materials present + the fine-energy table
                     (sum_{E in g} w_E exp(-sum_m m_m(E) P_m))
    S_FC(Omega_m)  = restrict_fc_to_ld_moments(corrected fc_source_at_directions(lib, rho_quad, ..., Omega_m), q)
                     exact eta table on Q, l = 0 and l = 1 conservative sampling correction
                     (uncollided.fc_sampling_correction)
    SI             = source_iteration(ctx, mu_t_coarse, mu_s_l_coarse, S_FC)                   LD sweeps on C
    T_MC           = scatter_source_moments(mu_s_l_coarse, phi_total[:, :, 0])                  cell-mean MC source
    scatter        = scatter_image.combined_scatter_image(..., T_MC, mu_t_coarse)               ONE collapsed-cone
                     pass: per direction chunk tau_down once on C, S_FC at the Q points + S_MC per cell, attenuated
                     per point, one projection on Q, finite-cone footprint
    primary        = scatter_image.primary_image(lib, rho_fine)       exact in energy (as phi_uc), on F
    SF             = S / (P + S)

The collided transport uses the library truncated to Legendre order L (`truncated_library`); the uncollided fluence,
the primary and the first-collision source use the full library (true mu_t, exact eta table).

No state is kept between calls apart from caches of grid/quadrature-dependent objects (transport grids, sweep contexts,
collapsed-cone direction sets, and the scatter-image direction contexts `scatter_image.make_direction_context` per
(view angle, direction set, grids), an LRU of `direction_context_cache` entries shared by the scatter images of a view
and by repeated calls such as the CITRUS step's). `clear_caches()` drops them all. All tensors live on
`solver.device` with `solver.dtype`.

Grids: every scatter order is transported on C. The first-order quantities (uncollided fluence, first-collision source
at the sweep and cone directions, its sampling correction and restriction, the first-order emission and its
projection) live on Q: q^3 points per transport cell whatever the fine voxel size (Acuros CTS, Maslowski et al. 2018,
Med Phys 45:1899, Section 2.B). Only the primary image (one projection per material present) and the resampling of
the densities touch the fine grid F.

Densities from the attenuation (`ScatterSolver.material_fields_from_attenuation(X, grid_fine, density_model)`): the
CITRUS loop has the attenuation X [R, C, S] (1/cm) on F, not the densities. The fields are then built material by
material: the density model evaluates ONE material's map rho_m = density_model.material(X, m) [R, C, S] at a time,
which is resampled onto C and Q and dropped, so the largest fine-grid tensor is one material map (never [M, R, C, S],
7 GB at 256 M voxels). The fields keep X and the model instead of rho_fine; the primary (and the "fine" uncollided
trace) then trace the materials one map at a time (`MaterialFields.fine_material`). The result equals
`material_fields(density_model(X), grid_fine)` to float rounding (the resampling is the same matrix products on one
material instead of seven).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Iterator, Optional, Protocol, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from .angular_quadrature import Quadrature, product_quadrature, scatter_source_moments
from .config import SolverConfig
from .cross_sections import (MaterialPairs, MultigroupLibrary, material_pairs_if_compact, mu_t_volume,
                             transfer_moments_volume)
from .detector import DetectorResponse, GridResponse
from .geometry import ConeGeometry, Grid, resample_conservative
from .ray_tracing import Projectors, SiddonProjectors
from .scatter_image import (DirectionSetContext, combined_scatter_image, make_direction_context, panel_direction_set,
                            primary_from_path_lengths, primary_image, primary_path_lengths, scatter_fraction)
from .source import SourceModel
from .source_iteration import SIResult, scatter_moments, scatter_source_fn, source_iteration
from .transport_sweep import SourceFn, SweepContext, make_sweep_context, sweep
from .uncollided import (DENSE, FCSamplingCorrection, fc_sampling_correction, fc_source_at_directions,
                         restrict_fc_to_ld_moments, uncollided_fluence,
                         uncollided_from_path_lengths, uncollided_path_lengths)

__all__ = [
    "truncated_library",
    "build_solver_library",
    "coarsen_density",
    "MaterialDensityModel",
    "MaterialFields",
    "ViewResult",
    "ScatterSolver",
]

DeviceLike = Union[torch.device, str, None]


# =====================================================================================================
# libraries
# =====================================================================================================

def build_solver_library(config: SolverConfig, spectrum=None, cache_dir: Optional[str] = "default",
                         materials=None) -> MultigroupLibrary:
    """`cross_section_library.build_library(config, spectrum, cache_dir, materials=materials)` (spectrum None -> the
    default 125 kVp spectrum; cache_dir "default" -> `cross_section_library.default_cache_dir()`, None -> no cache;
    materials None -> the seven-material table, `cross_sections.MATERIAL_TABLE`). Needs XrayPhysics unless the
    library is cached (about 16 s to build). float64 CPU tensors."""
    from .cross_section_library import build_library  # lazy: XrayPhysics only needed here

    return build_library(config, spectrum, cache_dir, materials=materials)


def truncated_library(lib: MultigroupLibrary, L: int) -> MultigroupLibrary:
    """Library of the COLLIDED transport (sweep, multiple-collision source): mu_s_l truncated to [..., :L+1]; everything
    else (mu_t, the eta table, energies, the fine-energy table) unchanged. lib must have legendre_order >= L."""
    if lib.legendre_order < L:
        raise ValueError(f"library Legendre order {lib.legendre_order} < legendre_order {L} of the solver")
    if lib.legendre_order == L:
        return lib
    return replace(lib, mu_s_l=lib.mu_s_l[..., :L + 1].clone())


def coarsen_density(rho_fine: Tensor, grid_fine: Grid, grid_coarse: Grid) -> Tensor:
    """Cell averages of densities [M, R, C, S] on grid_fine over the cells of grid_coarse -> [M, Rc, Cc, Sc]
    (conservative overlap resampling, `geometry.resample_conservative`: exact for voxel-wise constant densities, zero
    outside the fine box; any ratio of the voxel sizes). Every coefficient is linear in rho, so this equals averaging
    mu. Used for the transport grid C and the quadrature grid Q."""
    return resample_conservative(rho_fine, grid_fine, grid_coarse)


# =====================================================================================================
# per-density fields
# =====================================================================================================

class MaterialDensityModel(Protocol):
    """What `ScatterSolver.material_fields_from_attenuation` needs of a density model: `material(X, m)` maps the
    attenuation X [R, C, S] (1/cm) to the density map of library material m, [R, C, S] (g/cm^3), on X's device
    (e.g. `citrus.citrus.DensityModel`, `utilities.attenuation_to_density`)."""

    def material(self, X: Tensor, m: int) -> Tensor:
        ...


@dataclass(frozen=True, eq=False)
class MaterialFields:
    """Everything that depends on the densities only (not on the view). Built by `ScatterSolver.material_fields`
    (from the fine densities) or `ScatterSolver.material_fields_from_attenuation` (from the attenuation X and a
    density model, material by material).

    grid_fine     fine (reconstruction) grid F
    grid_coarse   transport grid C = Grid.covering(grid_fine, config.transport_cell_cm) (cubic cells, centred on F)
    grid_quad     quadrature grid Q = grid_coarse.refine(q)
    q             effective quadrature points per axis per transport cell (`ScatterSolver.quadrature_points`:
                  round(transport_cell_cm / config.quadrature_point_spacing_cm) bounded by round(transport_cell_cm /
                  largest fine voxel edge))
    rho_fine      [M, R, C, S]        g/cm^3 (M = the library's basis materials); used by the primary (and the
                                      "fine" uncollided trace) only. None when the fields were built from the
                                      attenuation: then `attenuation` and `density_model` give the fine maps one
                                      material at a time (`fine_material`)
    rho_coarse    [M, Rc, Cc, Sc]     cell averages on C (conservative overlap resampling, `coarsen_density`)
    rho_quad      [M, Rq, Cq, Sq]     averages over the Q cells (the density each quadrature point represents)
    mu_t_coarse   [G, Rc, Cc, Sc]     group total attenuation on C: sweep and the downstream attenuation of the
                                      scatter image
    mu_s_l_coarse [G, G, L+1, Rc, Cc, Sc]  Legendre transfer moments on C (1/cm), truncated at the solver's order L
    pairs_quad    `cross_sections.MaterialPairs` of rho_quad (index / density [2, Rq, Cq, Sq]) when M > 2 and every
                  quadrature point has at most two nonzero materials (always when Q equals a fine grid obeying the
                  density model's convention); None otherwise: the first-collision source then evaluates all M
                  materials (`fc_pairs`)
    attenuation   [R, C, S]           X (1/cm) on F, or None (fields built from rho_fine)
    density_model the `MaterialDensityModel` of X, or None
    fine_materials library indices of the materials with a nonzero fine map (in library order; at least one), set by
                  `material_fields_from_attenuation`; None for fields built from rho_fine (the primary then finds
                  them with `uncollided.present_materials`)
    No [G, ...] field lives on the fine grid; fields built from the attenuation hold no [M, ...] fine field either.
    """

    grid_fine: Grid
    grid_coarse: Grid
    grid_quad: Grid
    q: int
    rho_fine: Tensor
    rho_coarse: Tensor
    rho_quad: Tensor
    mu_t_coarse: Tensor
    mu_s_l_coarse: Tensor
    pairs_quad: Optional[MaterialPairs] = None
    attenuation: Optional[Tensor] = None
    density_model: Optional[MaterialDensityModel] = None
    fine_materials: Optional[tuple[int, ...]] = None

    @property
    def fc_pairs(self):
        """The `pairs` argument of the first-collision functions: pairs_quad, or `uncollided.DENSE` when it is None."""
        return DENSE if self.pairs_quad is None else self.pairs_quad

    @property
    def num_materials(self) -> int:
        return int(self.rho_coarse.shape[0])

    def fine_material(self, m: int) -> Tensor:
        """The fine density map of library material m, [R, C, S] (g/cm^3): rho_fine[m], or (fields built from the
        attenuation) density_model.material(attenuation, m), evaluated on demand on the dtype of rho_coarse (one map
        of the fine grid; the caller drops it after use)."""
        if self.rho_fine is not None:
            return self.rho_fine[m]
        if self.attenuation is None or self.density_model is None:
            raise ValueError("MaterialFields without rho_fine need attenuation and density_model")
        return self.density_model.material(self.attenuation, int(m)).to(self.rho_coarse.dtype)


# =====================================================================================================
# result
# =====================================================================================================

_IMAGE_KEYS = ("fluence", "planar_fluence", "signal_per_group", "signal")


@dataclass
class ViewResult:
    """Result of `ScatterSolver.solve_view` for one view angle. Detector images are [num_det_rows, num_det_channels].

    primary            : dict fluence [G, Rd, Cd] (scalar), planar_fluence [G, Rd, Cd], signal_per_group
                         [G, Rd, Cd], signal [Rd, Cd]   (scatter_image.primary_image)
    scatter            : same keys, the total scatter image (first order + multiple collision,
                         scatter_image.combined_scatter_image)
    scatter_signal     : [Rd, Cd] scatter["signal"]   (keV/cm^2)
    scatter_fraction   : [Rd, Cd] S / (P + S) of the signals
    si                 : SIResult (num_iterations, history, phi_norms, converged); its phi_total is an empty
                         tensor unless the solve was asked to keep it
    phi_total          : [G, (L+1)^2, 4, Rc, Cc, Sc] total collided-fluence moments, or None
    timings            : seconds per stage ("materials", "uncollided", "source_iteration", "mc_source",
                         "cone_directions" (direction set and direction context, cached per view), "scatter_image",
                         "primary", "total")
    peak_memory        : CUDA peak bytes allocated per stage (empty on CPU)
    peak_memory_bytes  : max over stages (None on CPU)
    All values are per source photon per steradian (docs/source/conventions.rst).
    """

    angle: float
    primary: dict[str, Tensor]
    scatter: dict[str, Tensor]
    scatter_signal: Tensor
    scatter_fraction: Tensor
    si: SIResult
    phi_total: Optional[Tensor] = None
    timings: dict[str, float] = field(default_factory=dict)
    peak_memory: dict[str, int] = field(default_factory=dict)
    peak_memory_bytes: Optional[int] = None

    @property
    def scatter_fluence(self) -> Tensor:
        """Total scatter scalar fluence [G, Rd, Cd]."""
        return self.scatter["fluence"]

    @property
    def scatter_planar_fluence(self) -> Tensor:
        """Total scatter planar fluence [G, Rd, Cd]."""
        return self.scatter["planar_fluence"]

    def to_numpy_dict(self) -> dict[str, np.ndarray]:
        """Flat dict of numpy arrays for `np.savez`: primary_<key> and scatter_<key> for every image key,
        scatter_signal, scatter_fraction, si_history, si_phi_norms, si_num_iterations, si_converged, timing_<stage>,
        peak_memory_bytes (-1 on CPU), angle and, if kept, phi_total."""
        def a(t: Tensor) -> np.ndarray:
            return t.detach().cpu().numpy()

        out: dict[str, np.ndarray] = {"angle": np.asarray(self.angle)}
        for name, d in (("primary", self.primary), ("scatter", self.scatter)):
            for k in _IMAGE_KEYS:
                out[f"{name}_{k}"] = a(d[k])
        out["scatter_signal"] = a(self.scatter_signal)
        out["scatter_fraction"] = a(self.scatter_fraction)
        out["si_history"] = np.asarray(self.si.history, dtype=np.float64)
        out["si_phi_norms"] = np.asarray(self.si.phi_norms, dtype=np.float64)
        out["si_num_iterations"] = np.asarray(self.si.num_iterations)
        out["si_converged"] = np.asarray(self.si.converged)
        for k, v in self.timings.items():
            out[f"timing_{k}"] = np.asarray(v)
        out["peak_memory_bytes"] = np.asarray(-1 if self.peak_memory_bytes is None else self.peak_memory_bytes)
        if self.phi_total is not None:
            out["phi_total"] = a(self.phi_total)
        return out


# =====================================================================================================
# solver
# =====================================================================================================

class ScatterSolver:
    """Deterministic kV CBCT scatter solver (LBTE source iteration solver), one view at a time.

    lib            : MultigroupLibrary with legendre_order >= config.legendre_order (`build_solver_library(config)`).
    config         : SolverConfig (validated here). Used fields: legendre_order, num_polar, num_azimuthal,
                     transport_cell_cm, quadrature_point_spacing_cm, uncollided_trace_grid, si_tol, si_max_iterations,
                     cone_num_polar,
                     cone_num_azimuthal,
                     direction_chunk (directions per batch in the scatter image AND directions per octant per sweep
                     chunk), dtype (when `dtype` is None).
    geom, projectors, detector, grid_response, source : system model; projectors is injected (any
                     `Projectors` implementation). None (default) is `SiddonProjectors()`: backend "auto", the fused
                     Triton ray tracer on float32 CUDA (PyTorch on CPU / float64). detector and grid_response are
                     required (they default to None only so that projectors can be omitted; pass them by keyword).
    device, dtype  : computation device (default: the library's device) and dtype (default config.dtype).
    fc_chunk       : directions per batch inside uncollided.fc_source_at_directions (PyTorch path).
    primary_n_sub  : sub-pixel sampling per axis of the primary (n^2 sub-rays per pixel, the transmission averaged
                     after the fine-energy sum; None: the projector's own n_sub attribute, else 1).
    direction_context_cache : number of scatter-image direction contexts kept (LRU, keyed by view angle, direction
                     set and grid shapes; 0 disables the cache). A context holds the response weights of one view's
                     collapsed-cone set: about 4.4 MiB at 512 directions (WLAB panel, G = 6).
    """

    def __init__(self, lib: MultigroupLibrary, config: SolverConfig, geom: ConeGeometry,
                 projectors: Optional[Projectors] = None, detector: Optional[DetectorResponse] = None,
                 grid_response: Optional[GridResponse] = None, source: Optional[SourceModel] = None,
                 device: DeviceLike = None, dtype: Optional[torch.dtype] = None, *,
                 fc_chunk: int = 4, primary_n_sub: Optional[int] = None,
                 direction_context_cache: int = 8) -> None:
        if detector is None or grid_response is None:
            raise TypeError("ScatterSolver needs a detector and a grid_response")
        config.validate()
        self.config = config
        self.L = int(config.legendre_order)
        self.device = torch.device(device) if device is not None else lib.mu_t.device
        self.dtype = dtype if dtype is not None else getattr(torch, config.dtype)
        self.lib = lib.to(self.device, self.dtype)
        self.lib_transport = truncated_library(self.lib, self.L)
        self.geom = geom
        self.projectors = projectors if projectors is not None else SiddonProjectors()
        self.detector = detector
        self.grid_response = grid_response
        self.source = source if source is not None else SourceModel()
        self.fc_chunk = int(fc_chunk)
        self.primary_n_sub = primary_n_sub
        self._ctx_cache: dict[Grid, SweepContext] = {}
        self._grid_cache: dict[Grid, Grid] = {}
        self._cone_cache: dict[tuple[Grid, float], Quadrature] = {}
        self.direction_context_cache = max(0, int(direction_context_cache))
        self._dir_ctx_cache: OrderedDict[tuple, DirectionSetContext] = OrderedDict()

    # ---- small helpers -------------------------------------------------------------------------
    @property
    def num_groups(self) -> int:
        return self.lib.num_groups

    def quadrature_points(self, grid_fine: Grid) -> int:
        """Effective quadrature points per axis per transport cell for this fine grid,
        `config.quadrature_points(max(grid_fine.deltas))` (round(cell / quadrature_point_spacing_cm) bounded by the data
        resolution)."""
        return self.config.quadrature_points(max(grid_fine.deltas))

    def transport_grid(self, grid_fine: Grid) -> Grid:
        """Transport grid C of a fine grid: `Grid.covering(grid_fine, config.transport_cell_cm)` (cached)."""
        g = self._grid_cache.get(grid_fine)
        if g is None:
            g = self._grid_cache[grid_fine] = Grid.covering(grid_fine, self.config.transport_cell_cm)
        return g

    def quadrature_grid(self, grid_fine: Grid) -> Grid:
        """Quadrature grid Q = transport_grid(grid_fine).refine(quadrature_points(grid_fine))."""
        return self.transport_grid(grid_fine).refine(self.quadrature_points(grid_fine))

    def sweep_quadrature(self) -> Quadrature:
        """The product sweep quadrature of the config (float64 CPU; the sweep context reorders it by octant)."""
        return product_quadrature(self.config.num_polar, self.config.num_azimuthal, dtype=torch.float64)

    def sweep_context(self, grid_coarse: Grid) -> SweepContext:
        """Cached `make_sweep_context(grid_coarse, sweep_quadrature(), L)` on the solver device/dtype."""
        ctx = self._ctx_cache.get(grid_coarse)
        if ctx is None:
            ctx = make_sweep_context(grid_coarse, self.sweep_quadrature(), self.L, device=self.device, dtype=self.dtype)
            self._ctx_cache[grid_coarse] = ctx
        return ctx

    def cone_quadrature(self, grid: Grid, angle: float) -> Quadrature:
        """Cached collapsed-cone direction set (`scatter_image.panel_direction_set` with the config's cone counts, cell
        widths filled) limited to `covering_polar_angle` of the box of `transport_grid(grid)` (which contains the box of
        the fine grid and every emitting point); omega [K, 3], weight [K] on the solver device. grid: the fine grid."""
        key = (grid, float(angle))
        q = self._cone_cache.get(key)
        if q is None:
            q = panel_direction_set(self.geom, angle, self.config, max_polar_angle=self.transport_grid(grid),
                                    device=self.device, dtype=self.dtype)
            self._cone_cache[key] = q
        return q

    def direction_context(self, fields: MaterialFields, angle: float,
                          quad: Optional[Quadrature] = None) -> DirectionSetContext:
        """Direction context of a view (`scatter_image.make_direction_context`: panel-facing directions, chunking,
        detector / grid response weights) for the collapsed-cone set `quad` (default: `cone_quadrature`), cached by
        (angle, id(quad), fine and coarse grid shapes, chunk); the cached context keeps a reference to quad, so the id
        stays valid while the entry lives. Used by `scatter_image` and reused by repeated calls with the same view."""
        quad = self.cone_quadrature(fields.grid_fine, angle) if quad is None else quad
        key = (float(angle), id(quad), tuple(fields.grid_fine.shape), tuple(fields.grid_coarse.shape),
               tuple(fields.grid_quad.shape), int(self.config.direction_chunk))
        ctx = self._dir_ctx_cache.get(key)
        if ctx is not None and ctx.quad is quad:
            self._dir_ctx_cache.move_to_end(key)
            return ctx
        ctx = make_direction_context(quad, self.geom, angle, self.detector, self.grid_response, self.lib,
                                     self.device, self.dtype, chunk=self.config.direction_chunk)
        if self.direction_context_cache > 0:
            self._dir_ctx_cache[key] = ctx
            while len(self._dir_ctx_cache) > self.direction_context_cache:
                self._dir_ctx_cache.popitem(last=False)
        return ctx

    def clear_caches(self) -> None:
        """Drop every cached object: sweep contexts, collapsed-cone direction sets and direction contexts (they are
        rebuilt on demand)."""
        self._ctx_cache.clear()
        self._grid_cache.clear()
        self._cone_cache.clear()
        self._dir_ctx_cache.clear()

    # ---- stages ----------------------------------------------------------------------------------
    def material_fields(self, rho_fine: Tensor, grid_fine: Grid) -> MaterialFields:
        """Density-only fields of `rho_fine` [M, R, C, S] on grid_fine (M = the library's number of materials): the
        transport grid C and quadrature grid Q, the densities resampled onto both (`coarsen_density`), the attenuation
        and transfer fields on C, and the material representation of rho_quad (`cross_sections.
        material_pairs_if_compact`, one host synchronization). Nothing of size [G, R, C, S] is formed on the fine
        grid."""
        if tuple(rho_fine.shape[1:]) != tuple(grid_fine.shape):
            raise ValueError(f"rho spatial shape {tuple(rho_fine.shape[1:])} != grid shape {grid_fine.shape}")
        if int(rho_fine.shape[0]) != self.lib.num_materials:
            raise ValueError(f"rho has {int(rho_fine.shape[0])} materials, the library {self.lib.num_materials} "
                             f"{tuple(self.lib.materials)}")
        rho = rho_fine.to(device=self.device, dtype=self.dtype)
        coarse = self.transport_grid(grid_fine)
        q = self.quadrature_points(grid_fine)
        quad = coarse.refine(q)
        rho_c = coarsen_density(rho, grid_fine, coarse)
        rho_q = rho_c if quad == coarse else coarsen_density(rho, grid_fine, quad)
        return MaterialFields(
            grid_fine=grid_fine, grid_coarse=coarse, grid_quad=quad, q=q, rho_fine=rho, rho_coarse=rho_c,
            rho_quad=rho_q, mu_t_coarse=mu_t_volume(self.lib_transport, rho_c),
            mu_s_l_coarse=transfer_moments_volume(self.lib_transport, rho_c),
            pairs_quad=material_pairs_if_compact(rho_q),
        )

    def material_fields_from_attenuation(self, X: Tensor, grid_fine: Grid,
                                         density_model: MaterialDensityModel) -> MaterialFields:
        """The `MaterialFields` of the densities density_model(X), built material by material: X [R, C, S] (1/cm,
        the attenuation on grid_fine) and density_model.material(X, m) -> [R, C, S] g/cm^3 (`MaterialDensityModel`,
        e.g. `citrus.citrus.DensityModel`). For every library material m the fine map is evaluated, tested for a
        nonzero value (one host synchronization per material: an absent material is neither resampled nor traced),
        resampled onto the transport grid C and the quadrature grid Q (`coarsen_density`) into rho_coarse [M, Rc, Cc,
        Sc] and rho_quad [M, Rq, Cq, Sq], and dropped; then mu_t_coarse, mu_s_l_coarse and pairs_quad as in
        `material_fields`. The largest fine-grid tensors are X and one material map (plus the density model's own
        temporaries of that size), never [M, R, C, S]. The fields keep X and the model (rho_fine is None), so the
        primary and the "fine" uncollided trace evaluate the fine maps again one at a time
        (`MaterialFields.fine_material`). Equals `material_fields(rho, grid_fine)` with rho the stacked maps to float
        rounding (identical resampling matrices, applied per material)."""
        if tuple(X.shape) != tuple(grid_fine.shape):
            raise ValueError(f"X shape {tuple(X.shape)} != grid shape {grid_fine.shape}")
        X = X.to(device=self.device, dtype=self.dtype)
        M = self.lib.num_materials
        coarse = self.transport_grid(grid_fine)
        q = self.quadrature_points(grid_fine)
        quad = coarse.refine(q)
        rho_c = torch.zeros((M,) + tuple(coarse.shape), device=self.device, dtype=self.dtype)
        rho_q = rho_c if quad == coarse else torch.zeros((M,) + tuple(quad.shape), device=self.device,
                                                         dtype=self.dtype)
        present = []
        for m in range(M):
            r = density_model.material(X, m).to(device=self.device, dtype=self.dtype)
            if tuple(r.shape) != tuple(grid_fine.shape):
                raise ValueError(f"density_model.material returned {tuple(r.shape)}, expected {grid_fine.shape}")
            if bool(r.ne(0).any()):
                present.append(m)
                rho_c[m] = coarsen_density(r, grid_fine, coarse)
                if quad != coarse:
                    rho_q[m] = coarsen_density(r, grid_fine, quad)
            del r
        return MaterialFields(
            grid_fine=grid_fine, grid_coarse=coarse, grid_quad=quad, q=q, rho_fine=None, rho_coarse=rho_c,
            rho_quad=rho_q, mu_t_coarse=mu_t_volume(self.lib_transport, rho_c),
            mu_s_l_coarse=transfer_moments_volume(self.lib_transport, rho_c),
            pairs_quad=material_pairs_if_compact(rho_q), attenuation=X, density_model=density_model,
            fine_materials=tuple(present) if present else (0,),
        )

    def uncollided(self, fields: MaterialFields, angle: float) -> Tensor:
        """Uncollided fluence phi_uc [G, Rq, Cq, Sq] at the quadrature points (the voxel centers of fields.grid_quad),
        ray traced from the focal spot (one ray per point) through Q itself (rho_quad; config.uncollided_trace_grid
        "quadrature", the default: material boundaries resolved at the Q spacing), through the transport grid C
        (rho_coarse, "coarse": a boundary inside a cell is smeared over it) or through the fine grid F (rho_fine,
        "fine"). The cost of all three is dominated by the fine-energy transmission, about the same."""
        if self.config.uncollided_trace_grid == "fine":
            if fields.rho_fine is None:  # fields from the attenuation: trace the fine maps one at a time
                mats = list(fields.fine_materials)
                path = torch.cat([uncollided_path_lengths(self.projectors, fields.fine_material(m)[None],
                                                          fields.grid_fine, self.geom, angle, fields.grid_quad)
                                  for m in mats])
                return uncollided_from_path_lengths(self.lib, path, mats, fields.grid_quad, self.geom, angle,
                                                    self.source)
            rho, grid = fields.rho_fine, fields.grid_fine
        elif self.config.uncollided_trace_grid == "quadrature":
            rho, grid = fields.rho_quad, fields.grid_quad
        else:
            rho, grid = fields.rho_coarse, fields.grid_coarse
        return uncollided_fluence(self.lib, rho, grid, self.geom, angle, self.source, self.projectors,
                                  points=fields.grid_quad)

    def fc_moment_source(self, fields: MaterialFields, angle: float, phi_uc: Tensor,
                         ctx: Optional[SweepContext] = None) -> SourceFn:
        """Sweep source function of the first-collision source: reordered direction indices [Dsel] ->
        LD moments [Dsel, G, 4, Rc, Cc, Sc] = restrict_fc_to_ld_moments(fc_source_at_directions(lib, rho_quad,
        phi_uc, grid_quad, ..., ctx.quad.omega[idx]), q) (exact eta table at the quadrature points, restricted to the
        transport cells). phi_uc: [G, Rq, Cq, Sq] (`uncollided`). The point samples are first multiplied by the
        per-point factors of `fc_sampling_correction` (computed here, once, by a pass over all sweep directions), so
        that the sweep quadrature integrates the FC source exactly in its l = 0 and l = 1 moments about the primary
        direction."""
        ctx = self.sweep_context(fields.grid_coarse) if ctx is None else ctx
        omega = ctx.quad.omega.to(device=self.device, dtype=self.dtype)
        corr = self.fc_sampling_correction(fields, angle, phi_uc, ctx)
        # the sweep calls the source once per chunk with 8 x direction_chunk directions; evaluate the FC source in
        # batches of at most direction_chunk directions so that its quadrature-grid temporaries stay bounded
        batch = max(1, min(int(self.config.direction_chunk), ctx.directions_per_octant))

        def fc_ld(idx: Tensor) -> Tensor:
            om = omega.index_select(0, idx.to(omega.device))
            fc = fc_source_at_directions(self.lib, fields.rho_quad, phi_uc, fields.grid_quad, self.geom, angle, om,
                                         chunk=self.fc_chunk, pairs=fields.fc_pairs)
            fc = corr.apply(fc, om)
            return restrict_fc_to_ld_moments(fc, fields.grid_quad, fields.q)

        def fn(idx: Tensor, out: Optional[Tensor] = None) -> Tensor:
            n = int(idx.numel())
            if n <= batch and out is None:
                return fc_ld(idx)
            for i in range(0, n, batch):
                part = fc_ld(idx[i:i + batch])
                if out is None:
                    out = torch.empty((n,) + tuple(part.shape[1:]), device=part.device, dtype=part.dtype)
                out[i:i + batch].copy_(part)
                del part
            return out

        return fn

    def fc_sampling_correction(self, fields: MaterialFields, angle: float, phi_uc: Tensor,
                               ctx: Optional[SweepContext] = None) -> FCSamplingCorrection:
        """The per-point factors (quadrature grid) that make the sweep quadrature integrate the FC source exactly in
        angle
        (l = 0 and l = 1 about Omega_UC): one extra pass of fc_source_at_directions over all sweep directions
        (uncollided.fc_sampling_correction). Computed once per call of `fc_moment_source`, i.e. once per view."""
        ctx = self.sweep_context(fields.grid_coarse) if ctx is None else ctx
        return fc_sampling_correction(self.lib, fields.rho_quad, phi_uc, fields.grid_quad, self.geom, angle,
                                      ctx.quad.omega, ctx.quad.weight, chunk=self.fc_chunk,
                                      batch=self.config.direction_chunk, pairs=fields.fc_pairs)

    def scatter_moment_source(self, fields: MaterialFields, phi: Tensor,
                              ctx: Optional[SweepContext] = None) -> SourceFn:
        """Sweep source function of the scattering source of collided moments phi [G, (L+1)^2, 4, Rc, Cc, Sc]:
        S(Omega) = sum_n T[:, n] Y_n(Omega), T = scatter_moments(mu_s_l_coarse, phi)."""
        ctx = self.sweep_context(fields.grid_coarse) if ctx is None else ctx
        return scatter_source_fn(ctx, scatter_moments(fields.mu_s_l_coarse, phi))

    def transport_sweep(self, fields: MaterialFields, source_fn: SourceFn) -> Tensor:
        """ONE LD sweep on the coarse grid: returns phi [G, (L+1)^2, 4, Rc, Cc, Sc]."""
        ctx = self.sweep_context(fields.grid_coarse)
        return sweep(ctx, fields.mu_t_coarse, source_fn, direction_chunk=self.config.direction_chunk)

    def run_source_iteration(self, fields: MaterialFields, fc_source_fn: SourceFn) -> SIResult:
        """Full source iteration with the config's tolerance and iteration cap."""
        ctx = self.sweep_context(fields.grid_coarse)
        cfg = self.config
        return source_iteration(ctx, fields.mu_t_coarse, fields.mu_s_l_coarse, fc_source_fn, tol=cfg.si_tol,
                                max_iterations=cfg.si_max_iterations, direction_chunk=cfg.direction_chunk)

    def mc_source_moments(self, fields: MaterialFields, phi_collided: Tensor) -> Tensor:
        """Cell-mean SH moments of the multiple-collision source: T [G, (L+1)^2, Rc, Cc, Sc] from the collided moments
        [G, (L+1)^2, 4, Rc, Cc, Sc] (only the LD mean is used by the scatter image)."""
        return scatter_source_moments(fields.mu_s_l_coarse, phi_collided[:, :, 0])

    def scatter_image(self, fields: MaterialFields, angle: float, phi_uc: Tensor, T_mean: Tensor,
                      quad: Optional[Quadrature] = None,
                      context: Optional[DirectionSetContext] = None) -> dict[str, Tensor]:
        """Total scatter image of a view in ONE collapsed-cone pass (scatter_image.combined_scatter_image): the FC
        source at the quadrature points (from phi_uc [G, Rq, Cq, Sq]) and the MC source of the cell-mean moments
        T_mean [G, (L+1)^2, Rc, Cc, Sc] share tau_down (mu_t_coarse), the per-point attenuation, one projection on Q
        per direction chunk and the finite-cone footprint. context: the view's direction context (default:
        `direction_context(fields, angle, quad)`, cached). Returns the image dict (fluence, planar_fluence,
        signal_per_group [G, Rd, Cd], signal [Rd, Cd])."""
        quad = self.cone_quadrature(fields.grid_fine, angle) if quad is None else quad
        context = self.direction_context(fields, angle, quad) if context is None else context
        return combined_scatter_image(
            self.lib, fields.rho_quad, phi_uc, fields.grid_quad, fields.q, T_mean, self.L, self.geom, angle, quad,
            fields.mu_t_coarse, self.projectors, self.detector, self.grid_response, context,
            chunk=self.config.direction_chunk, fc_chunk=self.fc_chunk, pairs=fields.fc_pairs,
            grid_coarse=fields.grid_coarse)

    def primary(self, fields: MaterialFields, angle: float) -> dict[str, Tensor]:
        """Primary image of a view (exact in energy, Siddon on the fine grid; `scatter_image.primary_image`). For
        fields built from the attenuation (rho_fine None) the materials present are traced one fine map at a time
        (`primary_path_lengths` per material, then `primary_from_path_lengths`): the same image to float rounding."""
        if fields.rho_fine is not None:
            return primary_image(self.lib, fields.rho_fine, fields.grid_fine, self.geom, angle, self.source,
                                 self.projectors, self.detector, self.grid_response, n_sub=self.primary_n_sub)
        mats = list(fields.fine_materials)
        path = torch.cat([self.primary_path_lengths(fields.fine_material(m)[None], fields.grid_fine, angle)
                          for m in mats])
        return self.primary_from_path_lengths(path, mats, angle)

    def primary_path_lengths(self, rho: Tensor, grid_fine: Grid, angle: float) -> Tensor:
        """`scatter_image.primary_path_lengths` with the solver's projectors and sub-pixel sampling: density maps
        rho [M', R, C, S] -> line integrals [M', Rd n, Cd n] (g/cm^2)."""
        return primary_path_lengths(self.projectors, rho.to(self.device, self.dtype), grid_fine, self.geom, angle,
                                    self.primary_n_sub)

    def primary_from_path_lengths(self, path: Tensor, materials: Optional[Sequence[int]],
                                  angle: float) -> dict[str, Tensor]:
        """`scatter_image.primary_from_path_lengths` with the solver's library, geometry, source, detector and grid
        response: per-material line integrals path [M', Rd n, Cd n] (g/cm^2) of the materials `materials` (library
        indices; None = all) -> the primary image dict of the view at `angle`."""
        return primary_from_path_lengths(self.lib, path.to(self.device, self.dtype), materials, self.geom, angle,
                                         self.source, self.detector, self.grid_response)

    # ---- timing ----------------------------------------------------------------------------------
    @contextmanager
    def _stage(self, name: str, timings: dict[str, float], peaks: dict[str, int]) -> Iterator[None]:
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.perf_counter()
        yield
        if cuda:
            torch.cuda.synchronize(self.device)
            peaks[name] = int(torch.cuda.max_memory_allocated(self.device))
        timings[name] = time.perf_counter() - t0

    # ---- public ----------------------------------------------------------------------------------
    def solve_view(self, rho_fine: Tensor, grid_fine: Grid, angle: float, keep_phi_total: bool = False) -> ViewResult:
        """Full solve for one view (module docstring).

        rho_fine: [M, R, C, S] basis-material densities (g/cm^3; M = 7 for the package's library) on grid_fine (any
        voxel size: the transport grid covers it, `transport_grid`). angle: view angle (rad). keep_phi_total: also
        return the collided moments [G, (L+1)^2, 4, Rc, Cc, Sc]. Returns ViewResult. On CUDA the per-stage peak memory
        is measured by
        resetting the device's peak-memory statistics at the start of every stage.
        """
        timings: dict[str, float] = {}
        peaks: dict[str, int] = {}
        t_all = time.perf_counter()
        with self._stage("materials", timings, peaks):
            fields = self.material_fields(rho_fine, grid_fine)
            ctx = self.sweep_context(fields.grid_coarse)
        with self._stage("uncollided", timings, peaks):
            phi_uc = self.uncollided(fields, angle)
        with self._stage("source_iteration", timings, peaks):
            si = self.run_source_iteration(fields, self.fc_moment_source(fields, angle, phi_uc, ctx))
        with self._stage("mc_source", timings, peaks):
            T_mean = self.mc_source_moments(fields, si.phi_total)
            phi_total = si.phi_total if keep_phi_total else None
            si = replace(si, phi_total=si.phi_total if keep_phi_total else torch.empty(0, device=self.device))
        with self._stage("cone_directions", timings, peaks):
            quad = self.cone_quadrature(grid_fine, angle)
            dctx = self.direction_context(fields, angle, quad)  # one per view
        with self._stage("scatter_image", timings, peaks):
            scatter = self.scatter_image(fields, angle, phi_uc, T_mean, quad, dctx)
        del phi_uc, T_mean
        with self._stage("primary", timings, peaks):
            prim = self.primary(fields, angle)
        s_sig = scatter["signal"]
        sf = scatter_fraction(s_sig, prim["signal"])
        timings["total"] = time.perf_counter() - t_all
        return ViewResult(angle=float(angle), primary=prim, scatter=scatter, scatter_signal=s_sig,
                          scatter_fraction=sf, si=si, phi_total=phi_total, timings=timings, peak_memory=peaks,
                          peak_memory_bytes=max(peaks.values()) if peaks else None)

    def solve_views(self, rho_fine: Tensor, grid_fine: Grid, angles: Sequence[float],
                    keep_phi_total: bool = False) -> list[ViewResult]:
        """`solve_view` for every angle (independent solves; the density-only fields are recomputed per view
        so that peak memory stays that of a single view)."""
        return [self.solve_view(rho_fine, grid_fine, float(a), keep_phi_total=keep_phi_total) for a in angles]
