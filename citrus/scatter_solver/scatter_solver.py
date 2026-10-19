"""The assembled LBTE scatter solver for one view: uncollided fluence, source iteration, scatter image and primary.

:class:`ScatterSolver` turns the basis-material densities of a volume (or its attenuation and a density model) into
the primary and scatter images and the scatter fraction of one view. :class:`MaterialFields` holds the density-only
fields on the fine, transport and quadrature grids, and :class:`ViewResult` the images of one view. The steps are
described in docs/source/method.rst, "The LBTE scatter solver", and the grids in docs/source/conventions.rst, "Grids".
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
from .cross_sections import (MaterialPairs, MultigroupLibrary, check_material_names, material_pairs_if_compact,
                             material_table, mu_t_volume, transfer_moments_volume)
from .detector import DetectorResponse, GridResponse
from .geometry import ConeGeometry, Grid, resample_conservative
from .kernels import uncollided_kernels
from .ray_tracing import Projectors, SiddonProjectors
from .scatter_image import (DirectionSetContext, combined_scatter_image, make_direction_context, panel_direction_set,
                            primary_from_path_lengths, primary_image, primary_path_lengths, scatter_fraction)
from .source import SourceModel
from .source_iteration import SourceIterationResult, scatter_moments, scatter_source_function, source_iteration
from .transport_sweep import SourceFunction, SweepContext, make_sweep_context, sweep
from .uncollided import (DENSE, FirstCollisionSamplingCorrection, first_collision_moments_at_directions,
                         first_collision_sampling_correction, uncollided_fluence, uncollided_from_path_lengths,
                         uncollided_path_lengths)

# The scatter image with the fused kernels takes up to this many directions per chunk, as many as fit the budget.
IMAGE_CHUNK_MAX_DIRECTIONS = 32
IMAGE_CHUNK_BUDGET_BYTES = 256 * 2 ** 20

__all__ = [
    "truncated_library",
    "build_solver_library",
    "coarsen_density",
    "MaterialDensityModel",
    "MaterialFields",
    "ViewResult",
    "ScatterSolver",
    "emission_profile_from_open_field",
]

DeviceLike = Union[torch.device, str, None]


# =====================================================================================================
# libraries
# =====================================================================================================

def build_solver_library(config: SolverConfig, spectrum, cache_dir: Optional[str] = "default",
                         materials=None) -> MultigroupLibrary:
    """Build (or load from the cache) the multigroup cross-section library of a solver configuration.

    Needs the photon data (``photon_data``).

    Args:
        config (SolverConfig): the solver configuration.
        spectrum (tuple): (energies [K] in keV, photon density per keV [K]) of the source; one node is a monoenergetic
            line (``cross_section_library.monoenergetic_spectrum``).
        cache_dir (str, optional): cache directory; "default" is ``cross_section_library.default_cache_dir()`` and None
            disables the cache. Defaults to "default".
        materials (list, optional): (name, nominal density, formula) per material, overriding ``config.materials``.
            Defaults to None, the table of ``config.materials``.

    Returns:
        MultigroupLibrary: float64 CPU library, its materials sorted by nominal attenuation.
    """
    from .cross_section_library import build_library  # lazy: the photon data are only needed here

    if materials is None:
        materials = material_table(check_material_names(config.materials))
    return build_library(config, spectrum, cache_dir, materials=materials)


def truncated_library(library: MultigroupLibrary, L: int) -> MultigroupLibrary:
    """Library of the collided transport: ``mu_s_l`` truncated to Legendre order ``L``, everything else unchanged.

    Args:
        library (MultigroupLibrary): library with ``legendre_order >= L``.
        L (int): Legendre order of the collided transport.

    Returns:
        MultigroupLibrary: the truncated library (``library`` itself when its order equals ``L``).

    Raises:
        ValueError: if ``library.legendre_order < L``.
    """
    if library.legendre_order < L:
        raise ValueError(f"library Legendre order {library.legendre_order} < legendre_order {L} of the solver")
    if library.legendre_order == L:
        return library
    return replace(library, mu_s_l=library.mu_s_l[..., :L + 1].clone())


def coarsen_density(rho_fine: Tensor, grid_fine: Grid, grid_coarse: Grid) -> Tensor:
    """Average densities over the cells of a coarser grid by conservative overlap resampling.

    Exact for voxel-wise constant densities and any ratio of the voxel sizes; zero outside the fine box.

    Args:
        rho_fine (Tensor): [..., R, C, S] densities in g/cm^3 on ``grid_fine``.
        grid_fine (Grid): the grid of ``rho_fine``.
        grid_coarse (Grid): the target grid (transport grid C or quadrature grid Q).

    Returns:
        Tensor: [..., Rc, Cc, Sc] cell-averaged densities in g/cm^3 on ``grid_coarse``.
    """
    return resample_conservative(rho_fine, grid_fine, grid_coarse)


# =====================================================================================================
# per-density fields
# =====================================================================================================

class MaterialDensityModel(Protocol):
    """Interface of the density model of :meth:`ScatterSolver.material_fields_from_attenuation`.

    Implemented by ``citrus.citrus.DensityModel``.
    """

    def material(self, X: Tensor, m: int) -> Tensor:
        """Density map of one library material from the attenuation.

        Args:
            X (Tensor): [R, C, S] attenuation in cm^-1.
            m (int): library index of the material.

        Returns:
            Tensor: [R, C, S] density of material ``m`` in g/cm^3, on the device of ``X``.
        """
        ...


@dataclass(frozen=True, eq=False)
class MaterialFields:
    """Density-only fields of a volume on the fine, transport and quadrature grids (independent of the view).

    Built by :meth:`ScatterSolver.material_fields` from the fine densities or by
    :meth:`ScatterSolver.material_fields_from_attenuation` from the attenuation; no [G, ...] field lives on the fine
    grid.

    Attributes:
        grid_fine (Grid): fine (reconstruction) grid F, R x C x S voxels.
        grid_coarse (Grid): transport grid C of cubic cells covering F, Rc x Cc x Sc cells.
        grid_quadrature (Grid): quadrature grid Q = ``grid_coarse.refine(q)``, Rq x Cq x Sq points.
        q (int): quadrature points per axis per transport cell (:meth:`ScatterSolver.quadrature_points`).
        rho_fine (Tensor): [M, R, C, S] densities in g/cm^3 on F, or None for fields built from the attenuation.
        rho_coarse (Tensor): [M, Rc, Cc, Sc] cell-averaged densities in g/cm^3 on C.
        rho_quadrature (Tensor): [M, Rq, Cq, Sq] densities in g/cm^3 averaged over the cells of Q.
        mu_t_coarse (Tensor): [G, Rc, Cc, Sc] total attenuation coefficient mu_t per group in cm^-1 on C.
        mu_s_l_coarse (Tensor): [G, G, L+1, Rc, Cc, Sc] Legendre transfer moments in cm^-1 on C, truncated at order L.
        pairs_quadrature (MaterialPairs): the at most two nonzero materials per point of ``rho_quadrature`` when M > 2
            and every point has at most two, else None. Defaults to None.
        attenuation (Tensor): [R, C, S] attenuation X in cm^-1 on F for fields built from it, else None. Defaults to
            None.
        density_model (MaterialDensityModel): the density model of ``attenuation``, or None. Defaults to None.
        fine_materials (tuple): library indices of the materials with a nonzero fine map (fields built from the
            attenuation), else None. Defaults to None.
    """

    grid_fine: Grid
    grid_coarse: Grid
    grid_quadrature: Grid
    q: int
    rho_fine: Tensor
    rho_coarse: Tensor
    rho_quadrature: Tensor
    mu_t_coarse: Tensor
    mu_s_l_coarse: Tensor
    pairs_quadrature: Optional[MaterialPairs] = None
    attenuation: Optional[Tensor] = None
    density_model: Optional[MaterialDensityModel] = None
    fine_materials: Optional[tuple[int, ...]] = None

    @property
    def first_collision_pairs(self):
        """The ``pairs`` argument of the first-collision functions: ``pairs_quadrature``, or ``uncollided.DENSE``."""
        return DENSE if self.pairs_quadrature is None else self.pairs_quadrature

    @property
    def num_materials(self) -> int:
        """Number of library materials M."""
        return int(self.rho_coarse.shape[0])

    def fine_material(self, m: int) -> Tensor:
        """Fine density map of one library material, from ``rho_fine`` or evaluated on demand by the density model.

        Args:
            m (int): library index of the material.

        Returns:
            Tensor: [R, C, S] density of material ``m`` in g/cm^3 on F.
        """
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
    """Primary and scatter images of one view from :meth:`ScatterSolver.solve_view`.

    An image dict has the keys fluence, planar_fluence and signal_per_group [G, Rd, Cd] and signal [Rd, Cd] (Rd x Cd
    detector rows x channels); all values are per source photon per steradian.

    Attributes:
        angle (float): view angle in radians.
        primary (dict): primary image dict.
        scatter (dict): total scatter image dict, first order plus multiple collision.
        scatter_signal (Tensor): [Rd, Cd] scatter signal ``scatter["signal"]`` in keV/cm^2.
        scatter_fraction (Tensor): [Rd, Cd] scatter fraction ``S / (P + S)`` of the scatter and primary signals.
        source_iteration_result (SourceIterationResult): iterations, history, norms and convergence flag; its
            ``phi_total`` is empty unless kept.
        phi_total (Tensor): [G, (L+1)^2, 4, Rc, Cc, Sc] collided-fluence moments, or None. Defaults to None.
        timings (dict): seconds per stage and "total". Defaults to an empty dict.
        peak_memory (dict): CUDA peak bytes allocated per stage, empty on the CPU. Defaults to an empty dict.
        peak_memory_bytes (int): largest stage peak in bytes, or None on the CPU. Defaults to None.
    """

    angle: float
    primary: dict[str, Tensor]
    scatter: dict[str, Tensor]
    scatter_signal: Tensor
    scatter_fraction: Tensor
    source_iteration_result: SourceIterationResult
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
        """Flatten the result into a dict of numpy arrays for ``np.savez``.

        Returns:
            dict: ``primary_<key>``, ``scatter_<key>``, ``scatter_signal``, ``scatter_fraction``,
                ``source_iteration_<item>``, ``timing_<stage>``, ``peak_memory_bytes`` (-1 on the CPU), ``angle`` and
                ``phi_total`` if kept.
        """
        def to_numpy(t: Tensor) -> np.ndarray:
            return t.detach().cpu().numpy()

        out: dict[str, np.ndarray] = {"angle": np.asarray(self.angle)}
        for name, images in (("primary", self.primary), ("scatter", self.scatter)):
            for k in _IMAGE_KEYS:
                out[f"{name}_{k}"] = to_numpy(images[k])
        out["scatter_signal"] = to_numpy(self.scatter_signal)
        out["scatter_fraction"] = to_numpy(self.scatter_fraction)
        out["source_iteration_history"] = np.asarray(self.source_iteration_result.history, dtype=np.float64)
        out["source_iteration_phi_norms"] = np.asarray(self.source_iteration_result.phi_norms, dtype=np.float64)
        out["source_iteration_num_iterations"] = np.asarray(self.source_iteration_result.num_iterations)
        out["source_iteration_converged"] = np.asarray(self.source_iteration_result.converged)
        for k, v in self.timings.items():
            out[f"timing_{k}"] = np.asarray(v)
        out["peak_memory_bytes"] = np.asarray(-1 if self.peak_memory_bytes is None else self.peak_memory_bytes)
        if self.phi_total is not None:
            out["phi_total"] = to_numpy(self.phi_total)
        return out


# =====================================================================================================
# solver
# =====================================================================================================

class ScatterSolver:
    """Deterministic CT scatter solver (LBTE source iteration) of a cone or parallel beam, one view at a time.

    With ``library=None`` the solver is lazy: the library is built on first use (the ``library`` property), and the
    configuration may be replaced until then.

    Args:
        library (MultigroupLibrary): library with ``legendre_order >= config.legendre_order``
            (:func:`build_solver_library`), or None for a lazy solver.
        config (SolverConfig): solver configuration, validated here.
        geometry (ConeGeometry or ParallelGeometry): scan geometry.
        projectors (Projectors, optional): ray tracers. Defaults to None, ``SiddonProjectors()``.
        detector (DetectorResponse): detector response; required, pass it by keyword.
        grid_response (GridResponse): anti-scatter-grid response; required, pass it by keyword.
        source (SourceModel, optional): source model. Defaults to None, ``SourceModel()``.
        device (torch.device or str, optional): computation device. Defaults to None, the library's device (the CPU
            for a lazy solver).
        dtype (torch.dtype, optional): computation dtype. Defaults to None, ``config.dtype``.
        first_collision_chunk (int, optional): directions per batch inside ``first_collision_source_at_directions``.
            Defaults to 4.
        primary_samples_per_pixel_axis (int, optional): sub-pixel samples n per axis of the primary (n^2 rays per
            pixel). Defaults to None, the projector's own ``samples_per_pixel_axis`` or 1.
        direction_context_cache_size (int, optional): number of scatter-image direction contexts kept (least recently
            used dropped first); 0 disables the cache. Defaults to 8.
        spectrum (tuple, optional): (energies [K] in keV, photon density per keV [K]) of the source, required for a
            lazy library (``library=None``). Defaults to None.

    Attributes:
        L (int): Legendre order of the collided transport (``config.legendre_order``).
        device (torch.device): computation device.
        dtype (torch.dtype): computation dtype.
        spectrum (tuple): source spectrum of a lazy library, or None.
        library_cache_dir (str): cache directory of a lazy library (:func:`build_solver_library`). Defaults to
            "default".
        geometry (ConeGeometry or ParallelGeometry): scan geometry.
        projectors (Projectors): ray tracers.
        detector (DetectorResponse): detector response.
        grid_response (GridResponse): anti-scatter-grid response.
        source (SourceModel): source model.
        first_collision_chunk (int): directions per batch of the first-collision source.
        primary_samples_per_pixel_axis (int): sub-pixel samples per axis of the primary, or None.
        direction_context_cache_size (int): number of direction contexts kept.
    """

    def __init__(self, library: Optional[MultigroupLibrary], config: SolverConfig, geometry: ConeGeometry,
                 projectors: Optional[Projectors] = None, detector: Optional[DetectorResponse] = None,
                 grid_response: Optional[GridResponse] = None, source: Optional[SourceModel] = None,
                 device: DeviceLike = None, dtype: Optional[torch.dtype] = None, *,
                 first_collision_chunk: int = 4, primary_samples_per_pixel_axis: Optional[int] = None,
                 direction_context_cache_size: int = 8, spectrum=None) -> None:
        """Create a solver; the class docstring lists the arguments."""
        if detector is None or grid_response is None:
            raise TypeError("ScatterSolver needs a detector and a grid_response")
        config.validate()
        if library is None and spectrum is None:
            raise ValueError("a lazy ScatterSolver (library=None) needs the source spectrum: spectrum=(energies [K] in "
                             "keV, photon density per keV [K]), e.g. load_spectrum_txt(path) or "
                             "monoenergetic_spectrum(energy_kev)")
        self._config = config
        self.L = int(config.legendre_order)
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = library.mu_t.device if library is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else getattr(torch, config.dtype)
        self.spectrum = spectrum
        self.library_cache_dir: Optional[str] = "default"  # where a lazy library is cached (`build_solver_library`)
        self._library: Optional[MultigroupLibrary] = None
        self._transport_library: Optional[MultigroupLibrary] = None
        self._materials: Optional[tuple[str, ...]] = None
        if library is not None:
            self._set_library(library)
        self.geometry = geometry
        self.projectors = projectors if projectors is not None else SiddonProjectors()
        self.detector = detector
        self.grid_response = grid_response
        self.source = source if source is not None else SourceModel()
        self.first_collision_chunk = int(first_collision_chunk)
        self.primary_samples_per_pixel_axis = primary_samples_per_pixel_axis
        self._sweep_contexts: dict[Grid, SweepContext] = {}
        self._transport_grids: dict[Grid, Grid] = {}
        self._cone_quadratures: dict[tuple[Grid, float], Quadrature] = {}
        self.direction_context_cache_size = max(0, int(direction_context_cache_size))
        self._direction_contexts: OrderedDict[tuple, DirectionSetContext] = OrderedDict()

    # ---- library and configuration --------------------------------------------------------------
    def _set_library(self, library: MultigroupLibrary) -> None:
        """Move a library to the solver's device and dtype and derive its collided-transport library.

        Args:
            library (MultigroupLibrary): the cross-section library.
        """
        self._library = library.to(self.device, self.dtype)
        self._transport_library = truncated_library(self._library, self.L)

    @property
    def library_built(self) -> bool:
        """True once the solver has its library (given to the constructor or built)."""
        return self._library is not None

    @property
    def library(self) -> MultigroupLibrary:
        """Cross-section library on the solver's device and dtype; a lazy solver builds it on first use."""
        if self._library is None:
            self._set_library(build_solver_library(self._config, self.spectrum, self.library_cache_dir))
        return self._library

    @property
    def transport_library(self) -> MultigroupLibrary:
        """Library of the collided transport (:func:`truncated_library` at order ``L``)."""
        if self._transport_library is None:
            _ = self.library
        return self._transport_library

    @property
    def config(self) -> SolverConfig:
        """Solver configuration; it can be replaced only while the library is not built."""
        return self._config

    @config.setter
    def config(self, config: SolverConfig) -> None:
        """Replace the configuration (validated); with a built library only an equal one is accepted."""
        config.validate()
        if config == self._config:
            return
        if self._library is not None:
            raise RuntimeError("the solver's library is already built for its configuration; the configuration can "
                               "only change before the library is built (construct the solver with library=None)")
        self._config = config
        self.L = int(config.legendre_order)
        self._materials = None
        self.clear_caches()

    @property
    def materials(self) -> tuple[str, ...]:
        """Names of the library's materials in library order, known without building a lazy library."""
        if self._library is not None:
            return tuple(self._library.materials)
        if self._materials is None:
            from .cross_section_library import resolve_backend, sort_material_table, spectrum_arrays

            backend = resolve_backend()
            table = sort_material_table(self._config, material_table(check_material_names(self._config.materials)),
                                        spectrum_arrays(self.spectrum), backend)
            self._materials = tuple(name for name, _, _ in table)
        return self._materials

    # ---- small helpers -------------------------------------------------------------------------
    @property
    def num_groups(self) -> int:
        """Number of energy groups G (``config.num_groups`` before a lazy library is built)."""
        return self._library.num_groups if self._library is not None else int(self._config.num_groups)

    def quadrature_points(self, grid_fine: Grid) -> int:
        """Quadrature points per axis per transport cell for a fine grid (``SolverConfig.quadrature_points``).

        Args:
            grid_fine (Grid): the fine grid F.

        Returns:
            int: q, ``round(transport_cell_cm / quadrature_point_spacing_cm)`` bounded by the largest fine voxel edge.
        """
        return self.config.quadrature_points(max(grid_fine.deltas))

    def transport_grid(self, grid_fine: Grid) -> Grid:
        """Transport grid C covering a fine grid (cached).

        Args:
            grid_fine (Grid): the fine grid F.

        Returns:
            Grid: ``Grid.covering(grid_fine, config.transport_cell_cm)``.
        """
        transport_grid = self._transport_grids.get(grid_fine)
        if transport_grid is None:
            transport_grid = self._transport_grids[grid_fine] = Grid.covering(grid_fine, self.config.transport_cell_cm)
        return transport_grid

    def quadrature_grid(self, grid_fine: Grid) -> Grid:
        """Quadrature grid Q of a fine grid: the transport grid refined by :meth:`quadrature_points`.

        Args:
            grid_fine (Grid): the fine grid F.

        Returns:
            Grid: the quadrature grid Q.
        """
        return self.transport_grid(grid_fine).refine(self.quadrature_points(grid_fine))

    def sweep_quadrature(self) -> Quadrature:
        """Product sweep quadrature of the configuration (``num_polar`` x ``num_azimuthal`` directions).

        Returns:
            Quadrature: float64 CPU quadrature.
        """
        return product_quadrature(self.config.num_polar, self.config.num_azimuthal, dtype=torch.float64)

    def sweep_context(self, grid_coarse: Grid) -> SweepContext:
        """Sweep context of a transport grid on the solver's device and dtype (cached).

        Args:
            grid_coarse (Grid): the transport grid C.

        Returns:
            SweepContext: the context of :func:`~citrus.scatter_solver.transport_sweep.make_sweep_context`.
        """
        context = self._sweep_contexts.get(grid_coarse)
        if context is None:
            context = make_sweep_context(grid_coarse, self.sweep_quadrature(), self.L, device=self.device,
                                         dtype=self.dtype)
            self._sweep_contexts[grid_coarse] = context
        return context

    def cone_quadrature(self, grid: Grid, angle: float) -> Quadrature:
        """Collapsed-cone direction set of a view (cached).

        The cone product set of the configuration is limited to the polar angle that covers the transport-grid box.

        Args:
            grid (Grid): the fine grid F.
            angle (float): view angle in radians.

        Returns:
            Quadrature: omega [K, 3] and weight [K] on the solver's device.
        """
        key = (grid, float(angle))
        quadrature = self._cone_quadratures.get(key)
        if quadrature is None:
            quadrature = panel_direction_set(self.geometry, angle, self.config,
                                             max_polar_angle=self.transport_grid(grid), device=self.device,
                                             dtype=self.dtype)
            self._cone_quadratures[key] = quadrature
        return quadrature

    def direction_context(self, fields: MaterialFields, angle: float,
                          quadrature: Optional[Quadrature] = None) -> DirectionSetContext:
        """Direction context of a view for the scatter image, kept in a least-recently-used cache.

        The cache key is the angle, the id of ``quadrature`` (the cached context holds a reference to it), the grid
        shapes and the direction chunk.

        Args:
            fields (MaterialFields): density fields of the volume.
            angle (float): view angle in radians.
            quadrature (Quadrature, optional): collapsed-cone direction set. Defaults to None, :meth:`cone_quadrature`.

        Returns:
            DirectionSetContext: panel-facing directions, their chunking and the detector and grid response weights.
        """
        quadrature = self.cone_quadrature(fields.grid_fine, angle) if quadrature is None else quadrature
        chunk = self.image_direction_chunk(fields)
        key = (float(angle), id(quadrature), tuple(fields.grid_fine.shape), tuple(fields.grid_coarse.shape),
               tuple(fields.grid_quadrature.shape), chunk)
        context = self._direction_contexts.get(key)
        if context is not None and context.quadrature is quadrature:
            self._direction_contexts.move_to_end(key)
            return context
        context = make_direction_context(quadrature, self.geometry, angle, self.detector, self.grid_response,
                                         self.library, self.device, self.dtype, chunk=chunk)
        if self.direction_context_cache_size > 0:
            self._direction_contexts[key] = context
            while len(self._direction_contexts) > self.direction_context_cache_size:
                self._direction_contexts.popitem(last=False)
        return context

    def image_direction_chunk(self, fields: MaterialFields) -> int:
        """Directions per chunk of the scatter image.

        With the fused first-collision kernels (float32 CUDA) a chunk holds the attenuated emission at the quadrature
        points, the optical depths and multiple-collision source of the cells, the images and the footprint matrices of
        its directions; it takes as many directions as fit ``IMAGE_CHUNK_BUDGET_BYTES``, between
        ``config.direction_chunk`` and ``IMAGE_CHUNK_MAX_DIRECTIONS``, since fewer and larger chunks launch fewer
        kernels. The PyTorch path, whose temporaries are several times larger, keeps ``config.direction_chunk``.

        Args:
            fields (MaterialFields): density fields of the volume (for the sizes of the grids).

        Returns:
            int: directions per chunk.
        """
        base = int(self.config.direction_chunk)
        fused = (self.device.type == "cuda" and self.dtype == torch.float32
                 and uncollided_kernels.kernel_usable(self.device)[0])
        if not fused:
            return base
        G = int(self.num_groups)
        Rd, Cd = int(self.geometry.num_det_rows), int(self.geometry.num_det_channels)
        per_direction = 4 * (G * (fields.grid_quadrature.num_voxels + 2 * fields.grid_coarse.num_voxels
                                  + 3 * Rd * Cd) + Rd * Rd + Cd * Cd)
        return int(max(base, min(IMAGE_CHUNK_MAX_DIRECTIONS, IMAGE_CHUNK_BUDGET_BYTES // max(1, per_direction))))

    def clear_caches(self) -> None:
        """Drop every cache: transport grids, sweep contexts, collapsed-cone direction sets and direction contexts."""
        self._sweep_contexts.clear()
        self._transport_grids.clear()
        self._cone_quadratures.clear()
        self._direction_contexts.clear()

    # ---- stages ----------------------------------------------------------------------------------
    def material_fields(self, rho_fine: Tensor, grid_fine: Grid) -> MaterialFields:
        """Build the density-only fields of fine densities on the transport and quadrature grids.

        Nothing of size [G, R, C, S] is formed on the fine grid.

        Args:
            rho_fine (Tensor): [M, R, C, S] densities in g/cm^3 of the library's M materials on ``grid_fine``.
            grid_fine (Grid): the fine grid F, any voxel size.

        Returns:
            MaterialFields: the fields of ``rho_fine``.
        """
        if tuple(rho_fine.shape[1:]) != tuple(grid_fine.shape):
            raise ValueError(f"rho spatial shape {tuple(rho_fine.shape[1:])} != grid shape {grid_fine.shape}")
        if int(rho_fine.shape[0]) != self.library.num_materials:
            raise ValueError(f"rho has {int(rho_fine.shape[0])} materials, the library {self.library.num_materials} "
                             f"{tuple(self.library.materials)}")
        rho = rho_fine.to(device=self.device, dtype=self.dtype)
        grid_coarse = self.transport_grid(grid_fine)
        q = self.quadrature_points(grid_fine)
        grid_quadrature = grid_coarse.refine(q)
        rho_coarse = coarsen_density(rho, grid_fine, grid_coarse)
        rho_quadrature = (rho_coarse if grid_quadrature == grid_coarse
                          else coarsen_density(rho, grid_fine, grid_quadrature))
        return MaterialFields(
            grid_fine=grid_fine, grid_coarse=grid_coarse, grid_quadrature=grid_quadrature, q=q, rho_fine=rho,
            rho_coarse=rho_coarse, rho_quadrature=rho_quadrature,
            mu_t_coarse=mu_t_volume(self.transport_library, rho_coarse),
            mu_s_l_coarse=transfer_moments_volume(self.transport_library, rho_coarse),
            pairs_quadrature=material_pairs_if_compact(rho_quadrature),
        )

    def material_fields_from_attenuation(self, X: Tensor, grid_fine: Grid,
                                         density_model: MaterialDensityModel) -> MaterialFields:
        """Build the :class:`MaterialFields` of the densities ``density_model(X)`` one material map at a time.

        Each nonzero material map is resampled onto C and Q and dropped, so the largest fine-grid tensors are ``X`` and
        one map; the result equals :meth:`material_fields` of the stacked maps to float rounding.

        Args:
            X (Tensor): [R, C, S] attenuation in cm^-1 on ``grid_fine``.
            grid_fine (Grid): the fine grid F.
            density_model (MaterialDensityModel): maps ``X`` to the density of one material at a time.

        Returns:
            MaterialFields: fields with ``rho_fine`` None that keep ``X``, the model and the materials present.
        """
        if tuple(X.shape) != tuple(grid_fine.shape):
            raise ValueError(f"X shape {tuple(X.shape)} != grid shape {grid_fine.shape}")
        X = X.to(device=self.device, dtype=self.dtype)
        M = self.library.num_materials
        grid_coarse = self.transport_grid(grid_fine)
        q = self.quadrature_points(grid_fine)
        grid_quadrature = grid_coarse.refine(q)
        rho_coarse = torch.zeros((M,) + tuple(grid_coarse.shape), device=self.device, dtype=self.dtype)
        rho_quadrature = (rho_coarse if grid_quadrature == grid_coarse
                          else torch.zeros((M,) + tuple(grid_quadrature.shape), device=self.device, dtype=self.dtype))
        present = []
        for m in range(M):
            r = density_model.material(X, m).to(device=self.device, dtype=self.dtype)
            if tuple(r.shape) != tuple(grid_fine.shape):
                raise ValueError(f"density_model.material returned {tuple(r.shape)}, expected {grid_fine.shape}")
            if bool(r.ne(0).any()):
                present.append(m)
                rho_coarse[m] = coarsen_density(r, grid_fine, grid_coarse)
                if grid_quadrature != grid_coarse:
                    rho_quadrature[m] = coarsen_density(r, grid_fine, grid_quadrature)
            del r
        return MaterialFields(
            grid_fine=grid_fine, grid_coarse=grid_coarse, grid_quadrature=grid_quadrature, q=q, rho_fine=None,
            rho_coarse=rho_coarse, rho_quadrature=rho_quadrature,
            mu_t_coarse=mu_t_volume(self.transport_library, rho_coarse),
            mu_s_l_coarse=transfer_moments_volume(self.transport_library, rho_coarse),
            pairs_quadrature=material_pairs_if_compact(rho_quadrature), attenuation=X, density_model=density_model,
            fine_materials=tuple(present) if present else (0,),
        )

    def uncollided_fluence(self, fields: MaterialFields, angle: float) -> Tensor:
        """Uncollided fluence phi_uncollided at the quadrature points, ray traced from the focal spot.

        The rays run through Q, C or F as set by ``config.uncollided_trace_grid`` ("quadrature", "coarse" or "fine").

        Args:
            fields (MaterialFields): density fields of the volume.
            angle (float): view angle in radians.

        Returns:
            Tensor: [G, Rq, Cq, Sq] uncollided scalar fluence per group at the points of Q (per source photon per sr).
        """
        if self.config.uncollided_trace_grid == "fine":
            if fields.rho_fine is None:  # fields from the attenuation: trace the fine maps one at a time
                materials = list(fields.fine_materials)
                path_lengths = torch.cat([uncollided_path_lengths(self.projectors, fields.fine_material(m)[None],
                                                                  fields.grid_fine, self.geometry, angle,
                                                                  fields.grid_quadrature)
                                          for m in materials])
                return uncollided_from_path_lengths(self.library, path_lengths, materials, fields.grid_quadrature,
                                                    self.geometry, angle, self.source)
            rho, grid = fields.rho_fine, fields.grid_fine
        elif self.config.uncollided_trace_grid == "quadrature":
            rho, grid = fields.rho_quadrature, fields.grid_quadrature
        else:
            rho, grid = fields.rho_coarse, fields.grid_coarse
        return uncollided_fluence(self.library, rho, grid, self.geometry, angle, self.source, self.projectors,
                                  points=fields.grid_quadrature)

    def first_collision_moment_source(self, fields: MaterialFields, angle: float, phi_uncollided: Tensor,
                                      sweep_context: Optional[SweepContext] = None) -> SourceFunction:
        """Sweep source function of the first-collision source ``S_FC`` on the transport cells.

        The source is evaluated at the quadrature points for the requested sweep directions, scaled by the sampling
        correction (:meth:`first_collision_sampling_correction`, computed once here) and restricted to
        linear-discontinuous moments (``uncollided.first_collision_moments_at_directions``, one fused kernel on float32
        CUDA tensors).

        Args:
            fields (MaterialFields): density fields of the volume.
            angle (float): view angle in radians.
            phi_uncollided (Tensor): [G, Rq, Cq, Sq] uncollided fluence (:meth:`uncollided_fluence`).
            sweep_context (SweepContext, optional): sweep context. Defaults to None, that of ``fields.grid_coarse``.

        Returns:
            SourceFunction: maps reordered direction indices [D] to moments [D, G, 4, Rc, Cc, Sc].
        """
        sweep_context = self.sweep_context(fields.grid_coarse) if sweep_context is None else sweep_context
        omega = sweep_context.quadrature.omega.to(device=self.device, dtype=self.dtype)
        correction = self.first_collision_sampling_correction(fields, angle, phi_uncollided, sweep_context)
        # the sweep calls the source once per chunk with 8 x direction_chunk directions; evaluate the first-collision
        # source in batches of at most direction_chunk directions so that its quadrature-grid temporaries stay bounded
        batch = max(1, min(int(self.config.direction_chunk), sweep_context.directions_per_octant))

        pairs = fields.first_collision_pairs
        fused = uncollided_kernels.fused_kernels_usable(
            self.library, fields.rho_quadrature, phi_uncollided, pairs if isinstance(pairs, MaterialPairs) else None)

        def restricted_source(direction_index: Tensor, out: Optional[Tensor] = None) -> Tensor:
            directions = omega.index_select(0, direction_index.to(omega.device))
            return first_collision_moments_at_directions(self.library, fields.rho_quadrature, phi_uncollided,
                                                         fields.grid_quadrature, self.geometry, angle, directions,
                                                         correction, fields.q, chunk=self.first_collision_chunk,
                                                         pairs=pairs, out=out)

        def source_function(direction_index: Tensor, out: Optional[Tensor] = None) -> Tensor:
            n = int(direction_index.numel())
            if fused or (n <= batch and out is None):  # the fused kernel forms no quadrature-grid temporaries
                return restricted_source(direction_index, out)
            for i in range(0, n, batch):
                part = restricted_source(direction_index[i:i + batch])
                if out is None:
                    out = torch.empty((n,) + tuple(part.shape[1:]), device=part.device, dtype=part.dtype)
                out[i:i + batch].copy_(part)
                del part
            return out

        return source_function

    def first_collision_sampling_correction(self, fields: MaterialFields, angle: float, phi_uncollided: Tensor,
                                            sweep_context: Optional[SweepContext] = None
                                            ) -> FirstCollisionSamplingCorrection:
        """Per-point factors that make the sweep quadrature integrate the first-collision source exactly.

        Exact in the l = 0 and l = 1 moments about the direction from the focal spot.

        Args:
            fields (MaterialFields): density fields of the volume.
            angle (float): view angle in radians.
            phi_uncollided (Tensor): [G, Rq, Cq, Sq] uncollided fluence (:meth:`uncollided_fluence`).
            sweep_context (SweepContext, optional): sweep context. Defaults to None, that of ``fields.grid_coarse``.

        Returns:
            FirstCollisionSamplingCorrection: the factors per quadrature point and group.
        """
        sweep_context = self.sweep_context(fields.grid_coarse) if sweep_context is None else sweep_context
        return first_collision_sampling_correction(self.library, fields.rho_quadrature, phi_uncollided,
                                                   fields.grid_quadrature, self.geometry, angle,
                                                   sweep_context.quadrature.omega, sweep_context.quadrature.weight,
                                                   chunk=self.first_collision_chunk,
                                                   batch=self.config.direction_chunk,
                                                   pairs=fields.first_collision_pairs)

    def scatter_moment_source(self, fields: MaterialFields, phi: Tensor,
                              sweep_context: Optional[SweepContext] = None) -> SourceFunction:
        """Sweep source function of the scattering source of collided moments, ``S(Omega) = sum_n T[:, n] Y_n(Omega)``.

        Args:
            fields (MaterialFields): density fields of the volume.
            phi (Tensor): [G, (L+1)^2, 4, Rc, Cc, Sc] collided-fluence moments.
            sweep_context (SweepContext, optional): sweep context. Defaults to None, that of ``fields.grid_coarse``.

        Returns:
            SourceFunction: the scattering source with ``T = scatter_moments(mu_s_l_coarse, phi)``.
        """
        sweep_context = self.sweep_context(fields.grid_coarse) if sweep_context is None else sweep_context
        return scatter_source_function(sweep_context, scatter_moments(fields.mu_s_l_coarse, phi))

    def transport_sweep(self, fields: MaterialFields, source_function: SourceFunction) -> Tensor:
        """Run one linear-discontinuous transport sweep on the transport grid.

        Args:
            fields (MaterialFields): density fields of the volume.
            source_function (SourceFunction): the sweep source.

        Returns:
            Tensor: [G, (L+1)^2, 4, Rc, Cc, Sc] fluence moments.
        """
        sweep_context = self.sweep_context(fields.grid_coarse)
        return sweep(sweep_context, fields.mu_t_coarse, source_function, direction_chunk=self.config.direction_chunk)

    def run_source_iteration(self, fields: MaterialFields,
                             first_collision_source_function: SourceFunction) -> SourceIterationResult:
        """Run source iteration to the configured tolerance and iteration cap.

        Args:
            fields (MaterialFields): density fields of the volume.
            first_collision_source_function (SourceFunction): first-collision sweep source
                (:meth:`first_collision_moment_source`).

        Returns:
            SourceIterationResult: the collided-fluence moments ``phi_total`` and the convergence history.
        """
        sweep_context = self.sweep_context(fields.grid_coarse)
        config = self.config
        return source_iteration(sweep_context, fields.mu_t_coarse, fields.mu_s_l_coarse,
                                first_collision_source_function, tolerance=config.source_iteration_tolerance,
                                max_iterations=config.max_source_iterations, direction_chunk=config.direction_chunk)

    def multiple_collision_source_moments(self, fields: MaterialFields, phi_collided: Tensor) -> Tensor:
        """Cell-mean spherical-harmonic moments T of the multiple-collision source.

        Args:
            fields (MaterialFields): density fields of the volume.
            phi_collided (Tensor): [G, (L+1)^2, 4, Rc, Cc, Sc] collided-fluence moments; only the cell means are used.

        Returns:
            Tensor: [G, (L+1)^2, Rc, Cc, Sc] source moments T.
        """
        return scatter_source_moments(fields.mu_s_l_coarse, phi_collided[:, :, 0])

    def scatter_image(self, fields: MaterialFields, angle: float, phi_uncollided: Tensor,
                      scatter_source_moments: Tensor, quadrature: Optional[Quadrature] = None,
                      context: Optional[DirectionSetContext] = None) -> dict[str, Tensor]:
        """Total scatter image of a view in one collapsed-cone pass (first-collision plus multiple-collision source).

        Both sources share the downstream attenuation, one projection per direction chunk and the finite-cone footprint
        (:func:`~citrus.scatter_solver.scatter_image.combined_scatter_image`).

        Args:
            fields (MaterialFields): density fields of the volume.
            angle (float): view angle in radians.
            phi_uncollided (Tensor): [G, Rq, Cq, Sq] uncollided fluence (:meth:`uncollided_fluence`).
            scatter_source_moments (Tensor): [G, (L+1)^2, Rc, Cc, Sc] multiple-collision source moments.
            quadrature (Quadrature, optional): collapsed-cone direction set. Defaults to None, :meth:`cone_quadrature`.
            context (DirectionSetContext, optional): direction context of the view. Defaults to None,
                :meth:`direction_context`.

        Returns:
            dict: fluence, planar_fluence and signal_per_group [G, Rd, Cd] and signal [Rd, Cd].
        """
        quadrature = self.cone_quadrature(fields.grid_fine, angle) if quadrature is None else quadrature
        context = self.direction_context(fields, angle, quadrature) if context is None else context
        return combined_scatter_image(
            self.library, fields.rho_quadrature, phi_uncollided, fields.grid_quadrature, fields.q,
            scatter_source_moments, self.L, self.geometry, angle, quadrature, fields.mu_t_coarse, self.projectors,
            self.detector, self.grid_response, context, chunk=self.config.direction_chunk,
            first_collision_chunk=self.first_collision_chunk, pairs=fields.first_collision_pairs,
            grid_coarse=fields.grid_coarse)

    def primary(self, fields: MaterialFields, angle: float) -> dict[str, Tensor]:
        """Primary image of a view, exact in energy, ray traced on the fine grid.

        Fields built from the attenuation are traced one material map at a time.

        Args:
            fields (MaterialFields): density fields of the volume.
            angle (float): view angle in radians.

        Returns:
            dict: fluence, planar_fluence and signal_per_group [G, Rd, Cd] and signal [Rd, Cd].
        """
        if fields.rho_fine is not None:
            return primary_image(self.library, fields.rho_fine, fields.grid_fine, self.geometry, angle, self.source,
                                 self.projectors, self.detector, self.grid_response,
                                 samples_per_pixel_axis=self.primary_samples_per_pixel_axis)
        materials = list(fields.fine_materials)
        path_lengths = torch.cat([self.primary_path_lengths(fields.fine_material(m)[None], fields.grid_fine, angle)
                                  for m in materials])
        return self.primary_from_path_lengths(path_lengths, materials, angle)

    def primary_path_lengths(self, rho: Tensor, grid_fine: Grid, angle: float) -> Tensor:
        """Line integrals of density maps to the detector pixels with the solver's projectors and sub-pixel sampling.

        Args:
            rho (Tensor): [M', R, C, S] density maps in g/cm^3.
            grid_fine (Grid): the grid of ``rho``.
            angle (float): view angle in radians.

        Returns:
            Tensor: [M', Rd n, Cd n] line integrals in g/cm^2, n sub-pixel samples per axis.
        """
        return primary_path_lengths(self.projectors, rho.to(self.device, self.dtype), grid_fine, self.geometry, angle,
                                    self.primary_samples_per_pixel_axis)

    def primary_from_path_lengths(self, path: Tensor, materials: Optional[Sequence[int]],
                                  angle: float) -> dict[str, Tensor]:
        """Primary image of a view from per-material line integrals, with the solver's library, source and responses.

        Args:
            path (Tensor): [M', Rd n, Cd n] line integrals in g/cm^2.
            materials (Sequence[int]): library indices of the M' materials, or None for all.
            angle (float): view angle in radians.

        Returns:
            dict: fluence, planar_fluence and signal_per_group [G, Rd, Cd] and signal [Rd, Cd].
        """
        return primary_from_path_lengths(self.library, path.to(self.device, self.dtype), materials, self.geometry,
                                         angle, self.source, self.detector, self.grid_response)

    def open_field_signal(self, angle: float, source: Optional[SourceModel] = None,
                          dtype: Optional[torch.dtype] = None) -> Tensor:
        """Detector signal of an empty volume (the solver's open field) for one view.

        It carries the inverse-square and obliquity factors of the point source (none for a parallel beam), the
        emission profile of ``source`` and the grid and detector responses.

        Args:
            angle (float): view angle in radians.
            source (SourceModel, optional): source model. Defaults to None (the solver's own).
            dtype (torch.dtype, optional): dtype of the computation. Defaults to None (the solver's).

        Returns:
            Tensor: [Rd, Cd] signal per source photon per steradian, on the solver's device.
        """
        Rd, Cd = self.geometry.num_det_rows, self.geometry.num_det_channels
        zero_path = torch.zeros(1, Rd, Cd, device=self.device, dtype=self.dtype if dtype is None else dtype)
        return primary_from_path_lengths(self.library, zero_path, [0], self.geometry, angle,
                                         self.source if source is None else source, self.detector,
                                         self.grid_response)["signal"]

    # ---- timing ----------------------------------------------------------------------------------
    @contextmanager
    def _stage(self, name: str, timings: dict[str, float], peaks: dict[str, int]) -> Iterator[None]:
        """Time a stage and record its CUDA peak memory, synchronizing the device before and after.

        Args:
            name (str): stage name.
            timings (dict): seconds per stage, updated in place.
            peaks (dict): CUDA peak bytes per stage, updated in place.

        Yields:
            None: while the stage runs.
        """
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
        """Solve one view: fields, uncollided fluence, source iteration, scatter image, primary and scatter fraction.

        The stages are described in docs/source/method.rst, "The LBTE scatter solver". For a fine grid that cannot hold
        the M-material volume, use :meth:`material_fields_from_attenuation` and :meth:`solve_view_from_fields`.

        Args:
            rho_fine (Tensor): [M, R, C, S] densities in g/cm^3 of the library's materials on ``grid_fine``.
            grid_fine (Grid): the fine grid F, any voxel size.
            angle (float): view angle in radians.
            keep_phi_total (bool, optional): also return the collided moments. Defaults to False.

        Returns:
            ViewResult: images, scatter fraction, iteration history, timings and peak memory of the view.
        """
        timings: dict[str, float] = {}
        peaks: dict[str, int] = {}
        start_time = time.perf_counter()
        with self._stage("materials", timings, peaks):
            fields = self.material_fields(rho_fine, grid_fine)
        return self._solve_view(fields, angle, keep_phi_total, timings, peaks, start_time)

    def solve_view_from_fields(self, fields: MaterialFields, angle: float, keep_phi_total: bool = False) -> ViewResult:
        """Solve one view from prebuilt :class:`MaterialFields`, which every view of a volume can share.

        Args:
            fields (MaterialFields): density fields of the volume.
            angle (float): view angle in radians.
            keep_phi_total (bool, optional): also return the collided moments. Defaults to False.

        Returns:
            ViewResult: images, scatter fraction, iteration history and timings of the view.
        """
        return self._solve_view(fields, angle, keep_phi_total, {}, {}, time.perf_counter())

    def _solve_view(self, fields: MaterialFields, angle: float, keep_phi_total: bool, timings: dict[str, float],
                    peaks: dict[str, int], start_time: float) -> ViewResult:
        """Run the stages of one view after the fields are built.

        Args:
            fields (MaterialFields): density fields of the volume.
            angle (float): view angle in radians.
            keep_phi_total (bool): keep the collided moments.
            timings (dict): seconds per stage so far, updated in place.
            peaks (dict): CUDA peak bytes per stage so far, updated in place.
            start_time (float): ``time.perf_counter()`` at the start of the solve.

        Returns:
            ViewResult: the result of the view.
        """
        with self._stage("sweep_context", timings, peaks):
            grid_fine = fields.grid_fine
            sweep_context = self.sweep_context(fields.grid_coarse)
        with self._stage("uncollided", timings, peaks):
            phi_uncollided = self.uncollided_fluence(fields, angle)
        with self._stage("source_iteration", timings, peaks):
            iteration_result = self.run_source_iteration(
                fields, self.first_collision_moment_source(fields, angle, phi_uncollided, sweep_context))
        with self._stage("multiple_collision_source", timings, peaks):
            scatter_source_moments = self.multiple_collision_source_moments(fields, iteration_result.phi_total)
            phi_total = iteration_result.phi_total if keep_phi_total else None
            iteration_result = replace(iteration_result, phi_total=iteration_result.phi_total if keep_phi_total
                                       else torch.empty(0, device=self.device))
        with self._stage("cone_directions", timings, peaks):
            quadrature = self.cone_quadrature(grid_fine, angle)
            direction_context = self.direction_context(fields, angle, quadrature)  # one per view
        with self._stage("scatter_image", timings, peaks):
            scatter = self.scatter_image(fields, angle, phi_uncollided, scatter_source_moments, quadrature,
                                         direction_context)
        del phi_uncollided, scatter_source_moments
        with self._stage("primary", timings, peaks):
            primary = self.primary(fields, angle)
        scatter_signal = scatter["signal"]
        fraction = scatter_fraction(scatter_signal, primary["signal"])
        timings["total"] = time.perf_counter() - start_time
        return ViewResult(angle=float(angle), primary=primary, scatter=scatter, scatter_signal=scatter_signal,
                          scatter_fraction=fraction, source_iteration_result=iteration_result, phi_total=phi_total,
                          timings=timings, peak_memory=peaks,
                          peak_memory_bytes=max(peaks.values()) if peaks else None)

    def solve_views(self, rho_fine: Tensor, grid_fine: Grid, angles: Sequence[float],
                    keep_phi_total: bool = False) -> list[ViewResult]:
        """Solve every view independently with :meth:`solve_view`, rebuilding the fields per view.

        Args:
            rho_fine (Tensor): [M, R, C, S] densities in g/cm^3 of the library's materials on ``grid_fine``.
            grid_fine (Grid): the fine grid F.
            angles (Sequence[float]): view angles in radians.
            keep_phi_total (bool, optional): also return the collided moments. Defaults to False.

        Returns:
            list: one :class:`ViewResult` per angle.
        """
        return [self.solve_view(rho_fine, grid_fine, float(a), keep_phi_total=keep_phi_total) for a in angles]


# =====================================================================================================
# emission profile from a measured open field
# =====================================================================================================

def emission_profile_from_open_field(open_field: Union[Tensor, np.ndarray], solver: ScatterSolver,
                                     angles: Optional[Sequence[float]] = None,
                                     detector_gain: Optional[Union[Tensor, np.ndarray]] = None) -> Tensor:
    """Emission profile of the source from a measured open field (flat field), for ``SourceModel.emission_profile``.

    The profile is the measured open field divided by the detector gain and by the solver's own open field of a
    uniform source (``ScatterSolver.open_field_signal``), which carries the inverse-square and obliquity falloff and
    the responses, normalized to a mean of one per view; without a gain map the whole open field counts as emission.

    Args:
        open_field (Tensor or ndarray): [Rd, Cd] open-field signal, the same for every view, or [V, Rd, Cd] per view.
        solver (ScatterSolver): solver of the scanner (geometry, spectrum, detector and grid responses).
        angles (Sequence[float], optional): [V] view angles in rad of a per-view open field; for a [Rd, Cd] open field
            the first angle (or 0) sets the view of the solver's open field. Defaults to None.
        detector_gain (Tensor or ndarray, optional): [Rd, Cd] multiplicative per-pixel gain. Defaults to None (1).

    Returns:
        Tensor: [Rd, Cd] or [V, Rd, Cd] profile with mean 1 per view, float64 on the CPU.
    """
    shape = solver.geometry.image_shape
    device = solver.device
    field = torch.as_tensor(open_field, dtype=torch.float64, device=device)
    if field.dim() not in (2, 3) or tuple(field.shape[-2:]) != shape:
        raise ValueError(f"the open field must be [rows, channels] = {list(shape)} or [views, rows, channels], got "
                         f"{tuple(field.shape)}")
    if not bool(torch.isfinite(field).all()) or bool((field < 0).any()):
        raise ValueError("the open field must be finite and non-negative")
    angle_list = [] if angles is None else [float(a) for a in np.asarray(angles, dtype=np.float64).reshape(-1)]
    if field.dim() == 3 and len(angle_list) != field.shape[0]:
        raise ValueError(f"a per-view open field of {field.shape[0]} views needs {field.shape[0]} angles, got "
                         f"{len(angle_list)}")
    gain = None
    if detector_gain is not None:
        gain = torch.as_tensor(detector_gain, dtype=torch.float64, device=device)
        if tuple(gain.shape) != shape or not bool(torch.isfinite(gain).all()) or bool((gain <= 0).any()):
            raise ValueError(f"detector_gain must be a positive [rows, channels] = {list(shape)} map")
    uniform = SourceModel()
    tiny = torch.finfo(torch.float64).tiny

    def profile_of(measured: Tensor, angle: float) -> Tensor:
        solver_open = solver.open_field_signal(angle, uniform, dtype=torch.float64)
        denominator = solver_open if gain is None else solver_open * gain
        ratio = measured / denominator.clamp_min(tiny)
        mean = float(ratio.mean())
        return ratio / mean if mean > 0 else ratio

    if field.dim() == 2:
        profile = profile_of(field, angle_list[0] if angle_list else 0.0)
    else:
        profile = torch.stack([profile_of(field[v], angle_list[v]) for v in range(field.shape[0])])
    return profile.cpu()
