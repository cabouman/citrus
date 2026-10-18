"""Energy group rules (version 0.6.0, `SolverConfig.group_edges_method`, `grouping_object_size_cm`) on the default
125 kVp and the 6 MV (Mohan et al. 1985) spectra, the scattered-fluence estimate behind the default rule, the object
size of `citrus.utilities.object_size_cm`, and the lazy solver whose library is built after citrus_recon has measured
the object."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import SPECTRUM_125KVP
from citrus.scatter_solver.config import SolverConfig
from citrus.scatter_solver.cross_sections import material_table
from citrus.scatter_solver.geometry import Grid

SPECTRUM_6MV = Path(__file__).parents[1] / "demo" / "spectra" / "spectrum_6mv.txt"
# the equal-fluence edges of version 0.5.1 (read from its cached libraries): 125 kVp with 6 groups, 6 MV with 12
EQUAL_FLUENCE_125KVP_V051 = [125.0, 85.02386229823114, 69.2307011426586, 59.5472247094453, 51.660740550397065,
                             42.36917410642299, 10.0]
EQUAL_FLUENCE_6MV_V051 = [6000.0, 3912.7554332386762, 3161.9681406401496, 2553.337475612501, 2123.9281148201558,
                          1781.6218594466054, 1563.856014470915, 1358.5809159483686, 1160.915642344567,
                          972.9172331033687, 790.9825456553914, 609.047858207414, 10.0]


@pytest.fixture(scope="module")
def library_module():
    cross_section_library = pytest.importorskip("citrus.scatter_solver.cross_section_library")
    try:
        cross_section_library.resolve_backend()
    except Exception as exc:
        pytest.skip(f"photon data unavailable ({exc})")
    return cross_section_library


@pytest.fixture(scope="module")
def spectra(library_module):
    return {"125kvp": (SPECTRUM_125KVP, 6),
            "6mv": (library_module.load_spectrum_txt(str(SPECTRUM_6MV)), 12)}


@pytest.mark.parametrize("case", ["125kvp", "6mv"])
@pytest.mark.parametrize("method", ["scattered_fluence", "equal_fluence", "log"])
def test_group_rules(library_module, spectra, case, method):
    """Every rule gives G + 1 strictly descending edges from the spectrum maximum to energy_min_kev;
    "equal_fluence" reproduces the edges of version 0.5.1 bit for bit; "log" is geometric; "scattered_fluence" cuts
    the scattered-fluence estimate into equal areas and lies below the equal-fluence edges (the scattered photons are
    softer than the source)."""
    spectrum, groups = spectra[case]
    config = SolverConfig(num_groups=groups, group_edges_method=method)
    edges = library_module.make_group_edges(config, spectrum, fine_bin_width_kev=5.0 if case == "6mv" else 0.5)
    top = 125.0 if case == "125kvp" else 6000.0
    assert edges.shape == (groups + 1,) and bool(np.all(np.diff(edges) < 0))
    assert edges[0] == top and edges[-1] == 10.0
    if method == "equal_fluence":
        expected = EQUAL_FLUENCE_125KVP_V051 if case == "125kvp" else EQUAL_FLUENCE_6MV_V051
        assert edges.tolist() == expected
    elif method == "log":
        np.testing.assert_allclose(edges[:-1] / edges[1:], (top / 10.0) ** (1.0 / groups), rtol=1e-12)
    else:
        equal = library_module.make_group_edges(replace(config, group_edges_method="equal_fluence"), spectrum)
        assert bool(np.all(edges[1:-1] < equal[1:-1]))
        bin_edges, photons = library_module.scattered_fluence_spectrum(
            spectrum, energy_min_kev=10.0, object_size_cm=config.grouping_object_size_cm,
            fine_bin_width_kev=5.0 if case == "6mv" else 0.5)
        cumulative = np.concatenate([[0.0], np.cumsum(photons)])
        areas = np.diff(np.interp(edges[::-1], bin_edges, cumulative))
        np.testing.assert_allclose(areas, cumulative[-1] / groups, rtol=1e-9)


def test_scattered_fluence_puts_edges_below_600_kev_at_6_mv(library_module, spectra):
    """At 6 MV the equal-fluence rule has one group across 10-609 keV; the scattered-fluence rule (12 groups, a 20 cm
    object) puts several edges there, where the multiply scattered photons are."""
    spectrum, _ = spectra["6mv"]
    edges = library_module.make_group_edges(SolverConfig(num_groups=12), spectrum, fine_bin_width_kev=5.0)
    assert int(np.sum((edges > 10.0) & (edges < 600.0))) >= 4, edges
    assert int(np.sum((np.array(EQUAL_FLUENCE_6MV_V051) > 10.0) & (np.array(EQUAL_FLUENCE_6MV_V051) < 600.0))) == 0


def test_object_size_limits(library_module, spectra):
    """A vanishing object size recovers the equal-fluence edges; the edges move down monotonically as the object grows
    (more downscattered photons stay in the object) and approach the infinite-medium spectrum."""
    spectrum, _ = spectra["6mv"]
    config = SolverConfig(num_groups=12)
    equal = library_module.make_group_edges(replace(config, group_edges_method="equal_fluence"), spectrum)
    edges = {L: library_module.make_group_edges(replace(config, grouping_object_size_cm=L), spectrum,
                                                fine_bin_width_kev=5.0) for L in (1e-4, 10.0, 20.0, 40.0, 1e6)}
    np.testing.assert_allclose(edges[1e-4], equal, rtol=5e-3)
    for small, large in ((1e-4, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 1e6)):
        assert bool(np.all(edges[large][1:-1] < edges[small][1:-1])), (small, large)


def test_scattered_fluence_balance(library_module, spectra):
    """The balance phi = (S + inscatter) / (mu_t - mu_R + 1/L): with L -> 0 phi -> S L; the Klein-Nishina transfer
    conserves photons (the sum over the outgoing bins of a top-energy photon plus the part below energy_min is 1)."""
    spectrum, _ = spectra["125kvp"]
    edges, phi = library_module.scattered_fluence_spectrum(spectrum, energy_min_kev=10.0, object_size_cm=1e-6)
    density = library_module._Spectrum(*spectrum)
    source = density.cumulative_integral(edges[1:]) - density.cumulative_integral(edges[:-1])
    np.testing.assert_allclose(phi / 1e-6, source, rtol=1e-4, atol=1e-12 * source.max())
    k = np.array([[6000.0 / library_module.ELECTRON_REST_KEV]])
    epsilon = np.linspace(1.0 / (1.0 + 2.0 * k[0, 0]), 1.0, 200001)[None, :]
    a = 1.0 + 1.0 / k
    cos_theta = a - 1.0 / (k * epsilon)
    integrand = epsilon + 1.0 / epsilon - (1.0 - cos_theta ** 2)
    numeric = np.trapezoid(integrand, epsilon) if hasattr(np, "trapezoid") else np.trapz(integrand, epsilon)
    closed = (library_module._klein_nishina_cumulative(epsilon[:, -1:], k)
              - library_module._klein_nishina_cumulative(epsilon[:, :1], k))
    assert abs(float(closed[0, 0]) / float(numeric[0]) - 1.0) < 1e-6


def test_grouping_medium(library_module, spectra):
    """The medium of the balance: "auto" without a measurement and an explicit "water" give the water-only balance;
    a mapping of two reference materials gives other edges (a denser, bone-bearing medium keeps more low-energy
    photons); the cache key and the balance see the densities to 3 significant digits; names outside the catalogue or
    outside the reference materials raise."""
    spectrum, _ = spectra["6mv"]
    config = SolverConfig(num_groups=12)
    water = library_module.make_group_edges(replace(config, grouping_medium="water"), spectrum,
                                            fine_bin_width_kev=5.0)
    assert np.array_equal(library_module.make_group_edges(config, spectrum, fine_bin_width_kev=5.0), water)
    bin_edges, photons = library_module.scattered_fluence_spectrum(spectrum, energy_min_kev=10.0, object_size_cm=20.0,
                                                                   medium=(("water", 1.0),), fine_bin_width_kev=5.0)
    cumulative = np.concatenate([[0.0], np.cumsum(photons)])
    np.testing.assert_allclose(np.diff(np.interp(water[::-1], bin_edges, cumulative)), cumulative[-1] / 12, rtol=1e-9)
    mixture = replace(config, grouping_medium={"water": 0.8, "bone_cortical_icrp": 0.4})
    mixture.validate()
    assert mixture.grouping_medium == (("bone_cortical_icrp", 0.4), ("water", 0.8))
    edges = library_module.make_group_edges(mixture, spectrum, fine_bin_width_kev=5.0)
    assert bool(np.all(np.diff(edges) < 0)) and not np.allclose(edges, water)
    trace = replace(config, grouping_medium={"water": 0.99312, "bone_cortical_icrp": 1e-7})
    assert library_module.grouping_medium_densities(trace) \
        == (("water", 0.993),)
    for bad, message in (("unobtainium", "unknown grouping_medium"), ("lead", "not among the reference materials"),
                         ({"water": 1.0, "titanium": 0.1}, "not among the reference materials"),
                         ({"water": 0.0}, "at least one > 0")):
        with pytest.raises(ValueError, match=message):
            replace(config, grouping_medium=bad).validate()
    replace(config, materials=("air", "water", "lead"), grouping_medium="lead").validate()


def test_scattered_fluence_weights_the_groups(library_module, spectra):
    """With the "scattered_fluence" rule the energies within a group are weighted by the estimated total fluence: at
    6 MV the lowest group (10 keV to about 150 keV, nearly no source photons) gets the attenuation of its scattered
    photons, which sit near its top, instead of the flat source weighting's (dominated by the photoelectric attenuation
    near 10 keV)."""
    spectrum, _ = spectra["6mv"]
    water = material_table(("water",))
    built = {method: library_module.compute_library(SolverConfig(num_groups=12, legendre_order=1, num_eta_table=11,
                                                                 group_edges_method=method), spectrum,
                                                    fine_bin_width_kev=20.0, gauss_legendre_nodes=8,
                                                    materials=water)[0]
             for method in ("scattered_fluence",)}
    library = built["scattered_fluence"]
    low, high = float(library.group_edges[-1]), float(library.group_edges[-2])
    assert float(library.group_energy[-1]) > 0.5 * (low + high)  # the flat source weight gives the midpoint
    source_weighted = library_module.compute_library(
        SolverConfig(num_groups=12, legendre_order=1, num_eta_table=11, group_edges_kev=library.group_edges.tolist()),
        spectrum, fine_bin_width_kev=20.0, gauss_legendre_nodes=8, materials=water)[0]
    assert torch.equal(source_weighted.group_edges, library.group_edges)
    assert float(library.mu_t[0, -1]) < 0.7 * float(source_weighted.mu_t[0, -1])
    assert torch.allclose(library.mu_t[0, :3], source_weighted.mu_t[0, :3], rtol=2e-3)  # source-dominated groups


def test_config_validation():
    with pytest.raises(ValueError, match="group_edges_method"):
        SolverConfig(group_edges_method="equal").validate()
    with pytest.raises(ValueError, match="grouping_object_size_cm"):
        SolverConfig(grouping_object_size_cm=0.0).validate()
    with pytest.raises(ValueError, match="angular_table_grid"):
        SolverConfig(angular_table_grid="log").validate()


# ---- object size --------------------------------------------------------------------------------------------------

def test_object_size_of_boxes_and_empty_mask():
    """4 V / S of voxel boxes, rounded to whole cm: a 30 x 20 x 40 cm box (the WLAB prism without inserts, 18.5 cm)
    at 1 cm and 0.5 cm voxels, a box touching the grid boundary (its faces there are exposed), the default for an
    empty mask and a lower bound of 1 cm."""
    from citrus.utilities import object_size_cm

    for delta in (1.0, 0.5):
        n = int(round(1 / delta))
        grid = Grid((24 * n, 34 * n, 44 * n), delta)  # rows (y), cols (x), slices (z)
        X = torch.zeros(grid.shape)
        X[2 * n:22 * n, 2 * n:32 * n, 2 * n:42 * n] = 0.2
        assert object_size_cm(X, grid, 0.1) == 18.0  # 4 * 24000 / 5200 = 18.46
        grid_tight = Grid((20 * n, 30 * n, 40 * n), delta)
        assert object_size_cm(torch.full(grid_tight.shape, 0.2), grid_tight, 0.1) == 18.0
    grid = Grid((8, 8, 8), 1.0)
    assert object_size_cm(torch.zeros(grid.shape), grid, 0.1) == 20.0
    assert object_size_cm(-torch.ones(grid.shape), grid, 0.1, default=7.0) == 7.0
    single = torch.zeros(grid.shape)
    single[3, 3, 3] = 1.0
    assert object_size_cm(single, grid, 0.1) == 1.0  # 4 / 6 cm, at least 1 cm


def test_grouping_estimates_on_the_wlab_phantom(library_module):
    """The object's mean composition by the density model ("auto" medium): on the WLAB phantom's true attenuation the
    mask is the prism without its lung and air cavities, so the mean water density is 22240 / 22400 = 0.993 g/cm^3 and
    the bone 160 / 22400 x 1.85 = 0.0132 g/cm^3 (traces of other reference materials from rounding); the data
    generator sets the true mean densities on a lazy solver before its library is built."""
    from citrus.utilities import (WlabPhantom, _set_grouping_object_size, grouping_medium_estimate, wlab_grid,
                                  wlab_solver)

    grid = wlab_grid(1.0)
    solver = wlab_solver(1.0, device="cpu", spectrum=SPECTRUM_125KVP)
    phantom = WlabPhantom(grid, solver.materials)
    _, mass = library_module.mass_attenuation_preview(solver.config, solver.materials, solver.spectrum)
    X = sum(phantom.material(None, m) * float(mass[m]) for m in range(len(solver.materials)))
    medium = dict(grouping_medium_estimate(X, solver))
    assert abs(medium["water"] - 22240 / 22400) < 1e-4 and abs(medium["bone_cortical_icrp"] - 160 / 22400 * 1.85) < 1e-4
    assert sum(d for n, d in medium.items() if n not in ("water", "bone_cortical_icrp")) < 1e-5
    assert not solver.library_built
    assert _set_grouping_object_size(solver, phantom, grid, None, "cpu") == 14.0
    true = dict(solver.config.grouping_medium)
    assert set(true) == {"water", "bone_cortical_icrp"} and abs(true["water"] - 22240 / 22400) < 1e-9
    assert not solver.library_built
    assert library_module.grouping_medium_densities(solver.config) == (("bone_cortical_icrp", 0.0132), ("water", 0.993))


@pytest.mark.parametrize("delta_voxel", [1.0, 0.5])
def test_object_size_of_the_wlab_phantom(library_module, delta_voxel):
    """The WLAB phantom (30 x 20 x 40 cm water prism; the lung (water at 0.26 g/cm^3) and air inserts fall below the
    air / water threshold and are cavities of the mask, the bone insert is inside): V = 24000 - 1440 - 160 cm^3,
    S = 5200 + 960 + 320 cm^2, 4 V / S = 13.8 -> 14 cm, at 1 cm and 0.5 cm voxels, from the lazy solver's threshold
    (no library built) and from a built library."""
    from citrus.utilities import WlabPhantom, air_water_threshold, object_size_cm, wlab_grid, wlab_solver

    grid = wlab_grid(delta_voxel)
    solver = wlab_solver(delta_voxel, device="cpu", spectrum=SPECTRUM_125KVP)
    threshold = air_water_threshold(solver)
    assert not solver.library_built and 0.09 < threshold < 0.11  # (0.0002 + 0.2014) / 2 at about 63 keV
    library = library_module.compute_library(SolverConfig(num_groups=3, legendre_order=2, num_eta_table=51),
                                             SPECTRUM_125KVP, fine_bin_width_kev=2.0, gauss_legendre_nodes=8)[0]
    X = WlabPhantom(grid, library.materials).attenuation(library)
    assert object_size_cm(X, grid, solver) == 14.0
    assert object_size_cm(X, grid, library) == 14.0
    assert abs(air_water_threshold(library) / threshold - 1.0) < 0.01


# ---- the lazy solver and citrus_recon -----------------------------------------------------------------------------

def test_lazy_solver_config_and_build(monkeypatch):
    """A solver without a library builds it on first use, for the config it has then; the config can change before
    that and not after (an equal config is accepted)."""
    from citrus.scatter_solver import scatter_solver as scatter_solver_module
    from citrus.scatter_solver.cross_sections import synthetic_library
    from citrus.utilities import wlab_solver

    built = []

    def fake_build(config, spectrum=None, cache_dir="default", materials=None):
        built.append(config)
        return synthetic_library(config.num_groups, legendre_order=config.legendre_order, num_materials=7)

    monkeypatch.setattr(scatter_solver_module, "build_solver_library", fake_build)
    solver = wlab_solver(2.0, config=SolverConfig(num_groups=4, legendre_order=3, num_polar=4, num_azimuthal=8),
                         device="cpu", spectrum=SPECTRUM_125KVP)
    assert not solver.library_built and solver.num_groups == 4 and built == []
    solver.config = replace(solver.config, grouping_object_size_cm=12.0)
    assert solver.library.num_groups == 4 and solver.library_built and len(built) == 1
    assert built[0].grouping_object_size_cm == 12.0 and solver.transport_library.legendre_order == 3
    solver.config = replace(solver.config)  # equal: accepted
    with pytest.raises(RuntimeError, match="already built"):
        solver.config = replace(solver.config, grouping_object_size_cm=30.0)
    assert len(built) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the CITRUS reconstruction tests run on CUDA")
def test_citrus_recon_measures_the_object_and_builds_once(monkeypatch):
    """citrus_recon with a lazy solver and grouping_object_size "auto": the size is measured on the FDK volume of the
    WLAB data (between 10 and 20 cm on the 2 cm grid), written into the solver's config before its library is built,
    reported in info together with the object's mean composition (the "auto" medium, a soft-tissue mixture), and the
    library is built exactly once (in the first scatter stage); a solver with a prebuilt library keeps its config."""
    pytest.importorskip("mbirtorch")
    from citrus.citrus import CitrusConfig, citrus_recon
    from citrus.scatter_solver import scatter_solver as scatter_solver_module
    from citrus.scatter_solver.scatter_solver import build_solver_library
    from citrus.utilities import synthesize_wlab_dataset, wlab_geometry, wlab_solver

    try:
        library = build_solver_library(SolverConfig(), SPECTRUM_125KVP)
    except Exception as exc:  # no cached library and no photon data
        pytest.skip(f"real library unavailable: {exc}")
    geometry = replace(wlab_geometry(), num_det_rows=46, num_det_channels=38, delta_det_row=1.6, delta_det_channel=1.6)
    config = SolverConfig(transport_cell_cm=2.0)
    data = synthesize_wlab_dataset(8, 2.0, noise=False, seed=0, library=library, geometry=geometry, config=config)
    built = []

    def counting_build(config_, spectrum=None, cache_dir="default", materials=None):
        built.append(config_)
        return library  # the 20 cm library stands in for every size (only the build count is tested)

    monkeypatch.setattr(scatter_solver_module, "build_solver_library", counting_build)
    from citrus.utilities import wlab_grid

    grid = wlab_grid(2.0)
    solver = wlab_solver(2.0, config=config, geometry=geometry, device="cuda", spectrum=SPECTRUM_125KVP)
    X, info = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, solver,
                           CitrusConfig(num_outer_iterations=2))
    size, medium = info["grouping_object_size_cm"], dict(info["grouping_medium"])
    assert info["grouping_applied"] and 10.0 <= size <= 20.0 and size == round(size)
    assert len(built) == 1 and built[0].grouping_object_size_cm == size == solver.config.grouping_object_size_cm
    # the FDK of scatter-contaminated data lies below the nominal water attenuation (cupping), so the density model
    # splits it into lung / adipose / water mixtures (8 views on a 2 cm grid: a coarse FDK); a soft-tissue medium
    assert built[0].grouping_medium == solver.config.grouping_medium and 0.5 < sum(medium.values()) < 1.2
    assert info["library_seconds"] >= 0.0 and bool(torch.isfinite(X).all())
    prebuilt = wlab_solver(2.0, config=config, geometry=geometry, library=library, device="cuda")
    _, info2 = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, prebuilt,
                            CitrusConfig(num_outer_iterations=1, grouping_object_size=11.0))
    assert info2["grouping_object_size_cm"] == 20.0 and not info2["grouping_applied"]
    assert info2["grouping_medium"] == "auto"
    assert len(built) == 1
