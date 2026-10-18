"""Monoenergetic sources (version 0.8.0): a one-node spectrum is a line; the library evaluates the fine bin that holds
it at the line's energy, so the effective energy and the primary attenuation are exactly those of the line, the
scattered photons fill the groups below, "equal_fluence" edges are refused, and the spectrum is required everywhere."""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from conftest import SPECTRUM_125KVP
from citrus.scatter_solver import load_spectrum_txt, monoenergetic_spectrum
from citrus.scatter_solver.config import SolverConfig
from citrus.scatter_solver.cross_sections import material_table
from citrus.scatter_solver.detector import ideal_counting_response, no_grid
from citrus.scatter_solver.geometry import Grid
from citrus.scatter_solver.ray_tracing import SiddonProjectors
from citrus.scatter_solver.scatter_solver import ScatterSolver
from citrus.utilities import wlab_geometry

SMALL = SolverConfig(num_groups=3, legendre_order=2, num_eta_table=51)
SMALL_BUILD = dict(fine_bin_width_kev=2.0, gauss_legendre_nodes=8)


@pytest.fixture(scope="module")
def library_module():
    cross_section_library = pytest.importorskip("citrus.scatter_solver.cross_section_library")
    try:
        cross_section_library.resolve_backend()
    except Exception as exc:
        pytest.skip(f"photon data unavailable ({exc})")
    return cross_section_library


def test_spectrum_density_of_a_line(library_module):
    """A one-node spectrum integrates to its weight from the line on, has no density, and ends at the line."""
    e, w = monoenergetic_spectrum(100.0)
    assert e.tolist() == [100.0] and w.tolist() == [1.0]
    density = library_module._Spectrum(e, 2.0 * w)
    assert density.monoenergetic and density.max_energy() == 100.0
    assert density.cumulative_integral(np.array([10.0, 99.999, 100.0, 125.0])).tolist() == [0.0, 0.0, 2.0, 2.0]
    assert density(np.array([100.0])).tolist() == [0.0]
    with pytest.raises(ValueError, match="> 0 keV"):
        monoenergetic_spectrum(0.0)
    with pytest.raises(ValueError, match="positive weight"):
        library_module._Spectrum(e, np.array([0.0]))


def test_one_line_file_is_a_line(library_module, tmp_path):
    """A spectrum file with one row loads as a monoenergetic spectrum."""
    path = tmp_path / "mono.txt"
    path.write_text("# a 6 MeV line\n6000 1\n")
    e, w = load_spectrum_txt(str(path))
    assert e.tolist() == [6000.0] and w.tolist() == [1.0]
    assert library_module._Spectrum(e, w).monoenergetic


def test_group_rules_for_a_line(library_module):
    """"scattered_fluence" and "log" edges run from the line down to energy_min_kev; "equal_fluence" is refused."""
    spectrum = monoenergetic_spectrum(100.0)
    for method in ("scattered_fluence", "log"):
        edges = library_module.make_group_edges(replace(SMALL, group_edges_method=method), spectrum,
                                                fine_bin_width_kev=2.0)
        assert edges[0] == 100.0 and edges[-1] == SMALL.energy_min_kev and np.all(np.diff(edges) < 0)
    with pytest.raises(ValueError, match="needs a continuous spectrum"):
        library_module.make_group_edges(replace(SMALL, group_edges_method="equal_fluence"), spectrum,
                                        fine_bin_width_kev=2.0)


def test_library_of_a_line_is_exact_at_the_line(library_module):
    """The fine bin that holds the line is evaluated at the line and carries all source photons, the effective energy
    is the line, and the primary transmission through water is exp(-mu(E0) t) with mu from the photon data."""
    backend = library_module.resolve_backend()
    energy = 100.0
    spectrum = monoenergetic_spectrum(energy)
    table = material_table(("water", "bone_cortical_icrp"))
    library, info = library_module.compute_library(SMALL, spectrum, backend, materials=table, **SMALL_BUILD)
    fine_energy, fine_weight = library.fine_energy.numpy(), library.fine_weight.numpy()
    hit = np.nonzero(fine_weight > 0)[0]
    assert hit.size == 1 and fine_energy[hit[0]] == energy and fine_weight[hit[0]] == pytest.approx(1.0)
    assert library.spectrum_fraction[0] == pytest.approx(1.0) and library.fine_group[hit[0]] == 0
    assert library_module._mean_energy(SMALL, spectrum, SMALL_BUILD["fine_bin_width_kev"], backend) == energy
    # the primary of the density model is sum_e w_e exp(-mu_e t): one term, the exact monoenergetic attenuation
    water = library.materials.index("water")
    composition = library_module.material_composition(backend, table)["water"]
    mu_water = sum(w * float(backend.sigma("total", Z, np.array([energy]))[0]) for Z, w in composition.items())
    assert float(library.fine_mu_t[water, hit[0]]) == pytest.approx(mu_water, rel=1e-12)
    thickness = 10.0
    transmission = float((library.fine_weight * torch.exp(-library.fine_mu_t[water] * thickness)).sum())
    assert transmission == pytest.approx(np.exp(-mu_water * thickness), rel=1e-12)
    # the group below the line holds the scattered photons: positive transfer out of group 0 into group 1 (Compton
    # from 100 keV reaches down to 72 keV, so the lowest group gets its photons through group 1)
    assert float(library.mu_s_l[water, 0, 1, 0]) > 0 and float(library.mu_s_l[water, 1, 2, 0]) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_solver_with_a_line_runs(library_module):
    """A lazy solver with a monoenergetic spectrum builds its library and solves a view with finite, positive signals;
    at the line the primary is the single-energy attenuation of the water slab."""
    spectrum = monoenergetic_spectrum(80.0)
    config = replace(SMALL, materials=("air", "water", "bone_cortical_icrp"))
    solver = ScatterSolver(None, config, wlab_geometry(), SiddonProjectors(), ideal_counting_response(), no_grid(),
                           device="cuda", dtype=torch.float32, spectrum=spectrum)
    solver.library_cache_dir = None
    grid = Grid((8, 8, 8), 2.0)
    rho = torch.zeros(3, *grid.shape, device="cuda")
    rho[1] = 1.0  # 16 cm of water
    result = solver.solve_view(rho, grid, 0.0)
    primary, scatter = result.primary["signal"], result.scatter_signal
    assert torch.isfinite(primary).all() and torch.isfinite(scatter).all()
    assert float(primary.max()) > 0 and float(scatter.max()) > 0
    assert solver.library.fine_energy[solver.library.fine_weight > 0].tolist() == [80.0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_attenuation_model_and_synthetic_data_of_a_line(library_module):
    """The library of a line has one fine energy; the mass attenuation at the effective energy is its tabulated
    value (no interpolation), so the attenuation volume, the primary and the synthetic counts are finite (the
    first run of demo 2 with a monoenergetic source hit NaN here)."""
    import numpy as np
    from citrus.utilities import (WlabPhantom, effective_energy, mass_attenuation_at, synthesize_cone_beam_data,
                                  wlab_geometry, wlab_grid)

    spectrum = monoenergetic_spectrum(100.0)
    config = replace(SMALL, materials=("air", "water", "bone_cortical_icrp"))
    solver = ScatterSolver(None, config, wlab_geometry(), SiddonProjectors(), ideal_counting_response(), no_grid(),
                           device="cuda", dtype=torch.float32, spectrum=spectrum)
    solver.library_cache_dir = None
    library = solver.library
    assert library.fine_energy.numel() == 1 and effective_energy(library) == 100.0
    assert torch.equal(mass_attenuation_at(library, 100.0), library.fine_mu_t[:, 0].to("cpu", torch.float64))
    grid = wlab_grid(2.0)
    data = synthesize_cone_beam_data(WlabPhantom(grid, solver.materials, device="cuda"), grid, wlab_geometry(),
                                     np.array([0.0, 0.3]), solver, I0=1.0e5, noise=True)
    for key in ("X_true", "P", "SPR", "Y"):
        assert np.isfinite(np.asarray(data[key])).all(), key
    assert np.asarray(data["Y"]).min() >= 0 and np.asarray(data["SPR"]).min() > 0


def test_polychromatic_spectra_are_unchanged(library_module):
    """The 125 kVp spectrum is not a line: ``place_line`` returns the midpoints unchanged."""
    density = library_module._Spectrum(*SPECTRUM_125KVP)
    assert not density.monoenergetic
    E = np.array([50.0, 60.0])
    assert density.place_line(E, E - 1.0, E + 1.0) is E


def test_spectrum_is_required_everywhere(library_module):
    """build_library, build_solver_library, a lazy solver and the group edges refuse a missing spectrum."""
    from citrus.scatter_solver.scatter_solver import build_solver_library

    with pytest.raises(ValueError, match="a source spectrum is required"):
        library_module.build_library(SMALL, None, cache_dir=None, **SMALL_BUILD)
    with pytest.raises(ValueError, match="a source spectrum is required"):
        build_solver_library(SMALL, None, cache_dir=None)
    with pytest.raises(ValueError, match="needs the source spectrum"):
        ScatterSolver(None, SMALL, wlab_geometry(), SiddonProjectors(), ideal_counting_response(), no_grid())
    assert os.environ.get("CITRUS_NO_DEFAULT_SPECTRUM") is None  # there is no switch that restores a default
