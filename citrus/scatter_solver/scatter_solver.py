"""The assembled LBTE source iteration scatter solver: uncollided fluence, source iteration, collapsed-cone scatter
image and primary image for one view.

Pipeline of `ScatterSolver.solve_view(rho_fine, grid_fine, angle)` (docs/source/method.rst has the derivation):

    rho_fine [M, R, C, S]  (fine grid F; M = 7     rho_coarse = supervoxel average of rho_fine  (grid C)
                           basis materials)         pairs_fine = material_pairs(rho_fine) (at most 2 per voxel)
    mu_t_fine      = mu_t_volume(lib, rho_fine)                       group attenuation (diagnostics)
    mu_t_coarse, mu_s_l_coarse from rho_coarse                        linear in rho: = averaged mu
    phi_uc         = uncollided_fluence(lib, rho_fine)                exact in energy: line integrals of the density
                     maps of the materials present + the fine-energy table (sum_{E in g} w_E exp(-sum_m m_m(E) P_m))
    S_FC(Omega_m)  = restrict_fc_to_ld_moments(corrected fc_source_at_directions(lib, ..., Omega_m))   exact eta
                     table, l = 0 and l = 1 conservative sampling correction (uncollided.fc_sampling_correction)
    SI             = source_iteration(ctx, mu_t_coarse, mu_s_l_coarse, S_FC)                   LD sweeps on C
    T_MC           = scatter_source_moments(mu_s_l_coarse, phi_total[:, :, 0])                  cell-mean MC source
    scatter        = scatter_image.combined_scatter_image(..., T_MC, mu_t_coarse)               ONE collapsed-cone
                     pass: per direction chunk tau_down once on C, S_FC at the fine sub-voxel points + S_MC per
                     supervoxel, attenuated per sub-point, one projection on F, finite-cone footprint
    primary        = scatter_image.primary_image(lib, rho_fine)       exact in energy (as phi_uc)
    SF             = S / (P + S)

The collided transport uses the library truncated to Legendre order L (`truncated_library`); the uncollided fluence,
the primary and the first-collision source use the full library (true mu_t, exact eta table).

No state is kept between calls apart from caches of grid/quadrature-dependent objects (sweep contexts,
collapsed-cone direction sets, and the scatter-image direction contexts `scatter_image.make_direction_context` per
(view angle, direction set, grids), an LRU of `direction_context_cache` entries shared by the scatter images of a view
and by repeated calls such as the CITRUS step's). `clear_caches()` drops them all. All tensors live on
`solver.device` with `solver.dtype`.

Grids: every scatter order, including the first-order term, is transported on the supervoxel grid C. The fine grid F
is used only for the primary image and for the uncollided fluence and first-collision source at the sub-voxel
quadrature points (the f^3 fine voxel centers of each supervoxel), which feed both the LD source moments of the sweep
and the scatter-image emission; the scatter image is projected on F (each sub-point traced to the panel).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Iterator, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .angular_quadrature import Quadrature, product_quadrature, scatter_source_moments
from .config import SolverConfig
from .cross_sections import MaterialPairs, MultigroupLibrary, material_pairs, mu_t_volume, transfer_moments_volume
from .detector import DetectorResponse, GridResponse
from .geometry import ConeGeometry, Grid
from .ray_tracing import Projectors, SiddonProjectors
from .scatter_image import (DirectionSetContext, combined_scatter_image, make_direction_context, panel_direction_set,
                            primary_image, scatter_fraction)
from .source import SourceModel
from .source_iteration import SIResult, scatter_moments, scatter_source_fn, source_iteration
from .transport_sweep import SourceFn, SweepContext, make_sweep_context, sweep
from .uncollided import (FCSamplingCorrection, fc_sampling_correction, fc_source_at_directions,
                         restrict_fc_to_ld_moments, uncollided_fluence)

__all__ = [
    "truncated_library",
    "build_solver_library",
    "coarsen_density",
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


def coarsen_density(rho_fine: Tensor, factor: int) -> Tensor:
    """Supervoxel average of densities [M, R, C, S] -> [M, R/f, C/f, S/f] (every coefficient is linear in rho, so
    this equals averaging mu)."""
    f = int(factor)
    if f == 1:
        return rho_fine
    return F.avg_pool3d(rho_fine.unsqueeze(0), kernel_size=f, stride=f).squeeze(0)


# =====================================================================================================
# per-density fields
# =====================================================================================================

@dataclass(frozen=True, eq=False)
class MaterialFields:
    """Everything that depends on the densities only (not on the view). Built by `ScatterSolver.material_fields`.

    grid_fine, grid_coarse : fine grid F and supervoxel grid C = grid_fine.coarsen(factor)
    rho_fine      [M, R, C, S]        g/cm^3 (M = the library's basis materials)
    rho_coarse    [M, Rc, Cc, Sc]     supervoxel averages
    mu_t_fine     [G, R, C, S]        group total attenuation (1/cm); diagnostics only: the uncollided fluence and
                                      the primary use the density line integrals and the fine-energy table
    mu_t_coarse   [G, Rc, Cc, Sc]     group total attenuation on C: sweep and the downstream attenuation of the
                                      scatter image
    mu_s_l_coarse [G, G, L+1, Rc, Cc, Sc]  Legendre transfer moments on C (1/cm), truncated at the solver's order L
    pairs_fine    `cross_sections.MaterialPairs` of rho_fine (index / density [2, R, C, S]) when M > 2 (the
                  first-collision source gathers the two table rows of each voxel by index), None for M <= 2
    """

    grid_fine: Grid
    grid_coarse: Grid
    factor: int
    rho_fine: Tensor
    rho_coarse: Tensor
    mu_t_fine: Tensor
    mu_t_coarse: Tensor
    mu_s_l_coarse: Tensor
    pairs_fine: Optional[MaterialPairs] = None


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
                     supervoxel_factor, si_tol, si_max_iterations, cone_num_polar, cone_num_azimuthal,
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
        self._cone_cache: dict[tuple[Grid, float], Quadrature] = {}
        self.direction_context_cache = max(0, int(direction_context_cache))
        self._dir_ctx_cache: OrderedDict[tuple, DirectionSetContext] = OrderedDict()

    # ---- small helpers -------------------------------------------------------------------------
    @property
    def num_groups(self) -> int:
        return self.lib.num_groups

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
        widths filled) limited to `covering_polar_angle(grid, geom, angle)`; omega [K, 3], weight [K] on the solver
        device."""
        key = (grid, float(angle))
        q = self._cone_cache.get(key)
        if q is None:
            q = panel_direction_set(self.geom, angle, self.config, max_polar_angle=grid, device=self.device,
                                    dtype=self.dtype)
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
               int(self.config.direction_chunk))
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
        self._cone_cache.clear()
        self._dir_ctx_cache.clear()

    # ---- stages ----------------------------------------------------------------------------------
    def material_fields(self, rho_fine: Tensor, grid_fine: Grid) -> MaterialFields:
        """Attenuation and transfer fields of `rho_fine` [M, R, C, S] on grid_fine and its supervoxel grid (M = the
        library's number of materials; for M > 2 at most two nonzero materials per voxel, whose pair representation
        `cross_sections.material_pairs` is derived here once)."""
        if tuple(rho_fine.shape[1:]) != tuple(grid_fine.shape):
            raise ValueError(f"rho spatial shape {tuple(rho_fine.shape[1:])} != grid shape {grid_fine.shape}")
        if int(rho_fine.shape[0]) != self.lib.num_materials:
            raise ValueError(f"rho has {int(rho_fine.shape[0])} materials, the library {self.lib.num_materials} "
                             f"{tuple(self.lib.materials)}")
        f = int(self.config.supervoxel_factor)
        rho = rho_fine.to(device=self.device, dtype=self.dtype)
        coarse = grid_fine.coarsen(f)
        rho_c = coarsen_density(rho, f)
        return MaterialFields(
            grid_fine=grid_fine, grid_coarse=coarse, factor=f, rho_fine=rho, rho_coarse=rho_c,
            mu_t_fine=mu_t_volume(self.lib, rho),
            mu_t_coarse=mu_t_volume(self.lib_transport, rho_c),
            mu_s_l_coarse=transfer_moments_volume(self.lib_transport, rho_c),
            pairs_fine=material_pairs(rho) if int(rho.shape[0]) > 2 else None,
        )

    def uncollided(self, fields: MaterialFields, angle: float) -> Tensor:
        """Uncollided fluence phi_uc [G, R, C, S] on the fine grid."""
        return uncollided_fluence(self.lib, fields.rho_fine, fields.grid_fine, self.geom, angle, self.source,
                                  self.projectors)

    def fc_moment_source(self, fields: MaterialFields, angle: float, phi_uc: Tensor,
                         ctx: Optional[SweepContext] = None) -> SourceFn:
        """Sweep source function of the first-collision source: reordered direction indices [Dsel] ->
        LD moments [Dsel, G, 4, Rc, Cc, Sc] = restrict_fc_to_ld_moments(fc_source_at_directions(lib, ...,
        ctx.quad.omega[idx])) (exact eta table, fine grid, restricted to the supervoxels). The fine-grid samples are
        first multiplied by the per-voxel factors of `fc_sampling_correction` (computed here, once, by a pass over
        all sweep directions), so that the sweep quadrature integrates the FC source exactly in its l = 0 and l = 1
        moments about the primary direction."""
        ctx = self.sweep_context(fields.grid_coarse) if ctx is None else ctx
        omega = ctx.quad.omega.to(device=self.device, dtype=self.dtype)
        corr = self.fc_sampling_correction(fields, angle, phi_uc, ctx)
        # the sweep calls the source once per chunk with 8 x direction_chunk directions; evaluate the fine-grid FC
        # source in batches of at most direction_chunk directions so that its fine-grid temporaries stay bounded
        batch = max(1, min(int(self.config.direction_chunk), ctx.directions_per_octant))

        def fc_ld(idx: Tensor) -> Tensor:
            om = omega.index_select(0, idx.to(omega.device))
            fc = fc_source_at_directions(self.lib, fields.rho_fine, phi_uc, fields.grid_fine, self.geom, angle, om,
                                         chunk=self.fc_chunk, pairs=fields.pairs_fine)
            fc = corr.apply(fc, om)
            return restrict_fc_to_ld_moments(fc, fields.grid_fine, fields.factor)

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
        """The per-voxel factors (fine grid) that make the sweep quadrature integrate the FC source exactly in angle
        (l = 0 and l = 1 about Omega_UC): one extra pass of fc_source_at_directions over all sweep directions
        (uncollided.fc_sampling_correction). Computed once per call of `fc_moment_source`, i.e. once per view."""
        ctx = self.sweep_context(fields.grid_coarse) if ctx is None else ctx
        return fc_sampling_correction(self.lib, fields.rho_fine, phi_uc, fields.grid_fine, self.geom, angle,
                                      ctx.quad.omega, ctx.quad.weight, chunk=self.fc_chunk,
                                      batch=self.config.direction_chunk, pairs=fields.pairs_fine)

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
        source at the fine sub-voxel points (from phi_uc [G, R, C, S]) and the MC source of the cell-mean moments
        T_mean [G, (L+1)^2, Rc, Cc, Sc] share tau_down (mu_t_coarse), the per-sub-point attenuation, one projection
        per direction chunk and the finite-cone footprint. context: the view's direction context (default:
        `direction_context(fields, angle, quad)`, cached). Returns the image dict (fluence, planar_fluence,
        signal_per_group [G, Rd, Cd], signal [Rd, Cd])."""
        quad = self.cone_quadrature(fields.grid_fine, angle) if quad is None else quad
        context = self.direction_context(fields, angle, quad) if context is None else context
        return combined_scatter_image(
            self.lib, fields.rho_fine, phi_uc, fields.grid_fine, fields.factor, T_mean, self.L, self.geom, angle, quad,
            fields.mu_t_coarse, self.projectors, self.detector, self.grid_response, context,
            chunk=self.config.direction_chunk, fc_chunk=self.fc_chunk, pairs=fields.pairs_fine)

    def primary(self, fields: MaterialFields, angle: float) -> dict[str, Tensor]:
        """Primary image (exact in energy)."""
        return primary_image(self.lib, fields.rho_fine, fields.grid_fine, self.geom, angle, self.source,
                             self.projectors, self.detector, self.grid_response, n_sub=self.primary_n_sub)

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

        rho_fine: [M, R, C, S] basis-material densities (g/cm^3; M = 7 for the package's library, at most two nonzero
        per voxel) on grid_fine (shape divisible by the supervoxel factor). angle: view angle (rad). keep_phi_total: also return the collided moments
        [G, (L+1)^2, 4, Rc, Cc, Sc]. Returns ViewResult. On CUDA the per-stage peak memory is measured by
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
