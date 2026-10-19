"""The LBTE source iteration scatter solver.

A deterministic solver of the linear Boltzmann transport equation (LBTE) for cone-beam CT scatter: multigroup in
energy, discrete ordinates in angle, linear-discontinuous finite elements in space, source iteration over collision
order and a collapsed-cone projection of the last-collided source onto the detector. The package exports the main
classes and functions of its modules; docs/source/api.rst lists the modules and docs/source/method.rst describes the
method.
"""

from .config import SolverConfig
from .cross_section_library import load_spectrum_txt, monoenergetic_spectrum
from .cross_sections import Material, MultigroupLibrary, synthetic_library
from .detector import (csi_response, energy_weighted_response, focused_1d_grid, ideal_counting_response,
                       no_grid)
from .geometry import ConeGeometry, Grid
from .ray_tracing import Projectors, SiddonProjectors
from .scatter_solver import MaterialFields, ScatterSolver, ViewResult, build_solver_library

__all__ = [
    "SolverConfig",
    "Grid",
    "ConeGeometry",
    "Material",
    "MultigroupLibrary",
    "synthetic_library",
    "build_solver_library",
    "load_spectrum_txt",
    "monoenergetic_spectrum",
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
