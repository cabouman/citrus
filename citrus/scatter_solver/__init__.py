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
from .detector import (anti_scatter_grid, csi_response, detector_response, energy_weighted_response,
                       focused_1d_grid, ideal_counting_response, no_grid, tabulated_detector_response)
from .geometry import ConeGeometry, Grid, ParallelGeometry
from .ray_tracing import Projectors, SiddonProjectors
from .scatter_solver import (MaterialFields, ScatterSolver, ViewResult, build_solver_library,
                             emission_profile_from_open_field)
from .source import SourceModel, bowtie_emission_profile

__all__ = [
    "SolverConfig",
    "Grid",
    "ConeGeometry",
    "ParallelGeometry",
    "Material",
    "MultigroupLibrary",
    "synthetic_library",
    "build_solver_library",
    "load_spectrum_txt",
    "monoenergetic_spectrum",
    "csi_response",
    "energy_weighted_response",
    "ideal_counting_response",
    "detector_response",
    "tabulated_detector_response",
    "focused_1d_grid",
    "anti_scatter_grid",
    "no_grid",
    "Projectors",
    "SiddonProjectors",
    "MaterialFields",
    "ScatterSolver",
    "ViewResult",
    "SourceModel",
    "bowtie_emission_profile",
    "emission_profile_from_open_field",
]
