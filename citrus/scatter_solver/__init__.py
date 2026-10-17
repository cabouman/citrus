"""The LBTE source iteration scatter solver.

A deterministic solver of the linear Boltzmann transport equation (LBTE) for kV cone-beam CT scatter: multigroup in
energy, discrete ordinates in angle, linear-discontinuous finite elements in space, source iteration over collision
order, and a collapsed-cone projection of the last-collided source onto the detector (docs/source/method.rst).

Modules
    config                 SolverConfig: every discretization parameter
    geometry               Grid (transport and quadrature grids), ConeGeometry (MBIRTORCH conventions), conservative
                           resampling between grids
    cross_sections         MultigroupLibrary container, per-voxel coefficients, synthetic test library
    cross_section_library  library builder from XrayPhysics (EPDL97) data, cached on disk
    source, detector       point source with collimation; detector and anti-scatter-grid responses
    angular_quadrature     direction sets, real spherical harmonics, scattering-source operator
    linear_discontinuous_cell  linear-discontinuous cell equations and their closed-form solve
    wavefront              wavefront ordering of the sweep
    transport_sweep        batched linear-discontinuous discrete-ordinates sweep
    source_iteration       source iteration over collision order
    uncollided             uncollided fluence and first-collision source
    scatter_image          collapsed-cone scatter image, finite-cone footprint, primary image
    ray_tracing            exact Siddon ray tracer behind the `Projectors` interface
    scatter_solver         ScatterSolver: the assembled solver for one view
    kernels                fused Triton kernels (float32 CUDA)
"""

from .config import SolverConfig
from .cross_sections import MultigroupLibrary, synthetic_library
from .detector import (csi_response, energy_weighted_response, focused_1d_grid, ideal_counting_response,
                       no_grid)
from .geometry import ConeGeometry, Grid
from .ray_tracing import Projectors, SiddonProjectors
from .scatter_solver import MaterialFields, ScatterSolver, ViewResult, build_solver_library

__all__ = [
    "SolverConfig",
    "Grid",
    "ConeGeometry",
    "MultigroupLibrary",
    "synthetic_library",
    "build_solver_library",
    "csi_response",
    "energy_weighted_response",
    "ideal_counting_response",
    "focused_1d_grid",
    "no_grid",
    "Projectors",
    "SiddonProjectors",
    "MaterialFields",
    "ScatterSolver",
    "ViewResult",
]
