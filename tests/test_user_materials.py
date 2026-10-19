"""User-defined reference materials (`cross_sections.Material`): a definition is checked on construction and stands
wherever a catalogue name does; a copy of a catalogue material builds the catalogue material's constants, a 1 %
iodine solution builds the mass-fraction mix of water and iodine and sorts between soft tissue and bone; the cache key
holds each definition (the default set keeps its key); the grouping medium, the lazy solver, the scatter solve, the
synthetic data and the CITRUS loop work with a user material."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from conftest import SPECTRUM_125KVP
from citrus.scatter_solver import Material, SolverConfig
from citrus.scatter_solver.cross_sections import (MATERIAL_CATALOGUE, MATERIALS, check_material_names,
                                                  material_definition, material_index, material_table, parse_formula)
from citrus.scatter_solver.detector import csi_response, no_grid
from citrus.scatter_solver.geometry import Grid
from citrus.scatter_solver.ray_tracing import SiddonProjectors
from citrus.scatter_solver.scatter_solver import ScatterSolver
from citrus.utilities import (grouping_medium_estimate, material_order, set_memory_cap, wlab_geometry, wlab_phantom)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
IODINE = Material("iodine_solution", 1.01, "0.99*water + 0.01*I")  # 1 % iodine by mass in water
MY_WATER = Material("my_water", 1.0, "H2O")
MY_BONE = Material("my_bone", 1.85, MATERIAL_CATALOGUE["bone_cortical_icrp"][1])
WATER_BY_NAME = Material("water_by_name", 1.0, "water")  # a catalogue material by name
WITH_IODINE = ("air", "water", IODINE, "bone_cortical_icrp")
# the small, fast library build of tests/test_material_catalogue.py
SMALL = SolverConfig(num_groups=3, legendre_order=2, num_eta_table=51)
SMALL_BUILD = dict(fine_bin_width_kev=2.0, gauss_legendre_nodes=8)
SMALL_SOLVER = replace(SMALL, num_polar=4, num_azimuthal=8, cone_num_polar=8, cone_num_azimuthal=16,
                       transport_cell_cm=4.0, source_iteration_tolerance=1e-4, max_source_iterations=40)
GRID = Grid((10, 16, 20), 2.0)
KEY_DEFAULT_V080 = "ce84c9793ffe2b097f66"  # the cache keys of tests/test_material_catalogue.py
KEY_SMALL_V080 = "6aa293182fc89d7c9054"
LIBRARY_TENSORS = ("mu_t", "mu_pe", "mu_s_l", "mu_s_eta", "fine_mu_t")


@pytest.fixture(scope="module")
def library_module():
    cross_section_library = pytest.importorskip("citrus.scatter_solver.cross_section_library")
    try:
        cross_section_library.resolve_backend()
    except Exception as exc:
        pytest.skip(f"photon data unavailable ({exc})")
    return cross_section_library


def _build(library_module, materials, config=SMALL):
    return library_module.compute_library(config, SPECTRUM_125KVP, library_module.resolve_backend(),
                                          materials=materials, **SMALL_BUILD)[0]


@pytest.fixture(scope="module")
def iodine_library(library_module):
    return _build(library_module, WITH_IODINE)


# ---- definition and checks (no photon data) ----------------------------------------------------------------------

def test_material_definition():
    """A Material is frozen and hashable with a float density; it stands in SolverConfig.materials (printable,
    hashable, JSON through to_dict), in material_table, material_index and material_definition; a copy of a catalogue
    entry becomes the catalogue name."""
    assert IODINE.density == 1.01 and Material("x", 1, "H2O").density == 1.0
    assert isinstance(Material("x", 1, "H2O").density, float) and Material("x", 1, "H2O") == Material("x", 1.0, "H2O")
    assert len({IODINE, Material("iodine_solution", 1.01, "0.99*water + 0.01*I"), MY_WATER}) == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        IODINE.density = 2.0
    config = SolverConfig(materials=list(WITH_IODINE))
    config.validate()
    assert config.materials == WITH_IODINE and hash(config) == hash(SolverConfig(materials=WITH_IODINE))
    assert "Material(name='iodine_solution', density=1.01, composition='0.99*water + 0.01*I')" in repr(config)
    record = json.loads(json.dumps(config.to_dict()))
    assert record["materials"][2] == {"name": "iodine_solution", "density": 1.01, "composition": "0.99*water + 0.01*I"}
    # copies of catalogue entries are the catalogue names (spaces in the composition ignored)
    copy = SolverConfig(materials=(Material("air", 0.0012, MATERIAL_CATALOGUE["air"][1].replace(" ", "")),
                                   Material("water", 1, "H2O"), "bone_cortical_icrp"))
    assert copy.materials == ("air", "water", "bone_cortical_icrp")
    assert check_material_names((Material("water", 1.0, "H2O"), IODINE)) == ("water", IODINE)
    assert material_table(("air", IODINE)) == (("air", 0.0012, MATERIAL_CATALOGUE["air"][1]),
                                               ("iodine_solution", 1.01, "0.99*water + 0.01*I"))
    assert material_index(WITH_IODINE, "iodine_solution") == 2 and material_index(WITH_IODINE, IODINE) == 2
    assert material_definition("iodine_solution", WITH_IODINE) is IODINE and material_definition(IODINE) is IODINE
    assert material_definition("water") == Material("water", 1.0, "H2O")
    with pytest.raises(ValueError, match="unknown material 'iodine_solution'"):
        material_definition("iodine_solution", MATERIALS)
    # the grouping medium: a user material by name, in pairs, or the Material itself (stored by name)
    SolverConfig(materials=WITH_IODINE, grouping_medium="iodine_solution").validate()
    SolverConfig(materials=WITH_IODINE, grouping_medium={"water": 0.9, "iodine_solution": 0.1}).validate()
    assert SolverConfig(materials=WITH_IODINE, grouping_medium=IODINE).grouping_medium == "iodine_solution"
    assert SolverConfig(materials=WITH_IODINE, grouping_medium={IODINE: 0.5}).grouping_medium == (
        ("iodine_solution", 0.5),)
    with pytest.raises(ValueError, match="unknown grouping_medium.*iodine_solution"):
        SolverConfig(grouping_medium="iodine_solution").validate()


def test_parse_formula_with_catalogue_names():
    """A lowercase part of a composition is a catalogue material by name; the mixture weighs its composition by the
    mass fraction."""
    water = parse_formula("H2O")
    assert parse_formula("water") == water
    iodine = parse_formula("0.99*water + 0.01*I")
    assert set(iodine) == {1, 8, 53} and iodine[53] == pytest.approx(0.01, abs=1e-15)
    for Z in (1, 8):
        assert iodine[Z] == pytest.approx(0.99 * water[Z], rel=1e-14)
    bone = parse_formula(MATERIAL_CATALOGUE["bone_cortical_icrp"][1])
    half = parse_formula("0.5*water + 0.5*bone_cortical_icrp")
    for Z in set(water) | set(bone):
        assert half[Z] == pytest.approx(0.5 * water.get(Z, 0.0) + 0.5 * bone.get(Z, 0.0), rel=1e-14)
    assert parse_formula("0.9*water + 0.1*iodine") == pytest.approx(parse_formula("0.9*water + 0.1*I"), rel=1e-15)


@pytest.mark.parametrize("definition, message", [
    (("x", 1.0, "H2O)"), "cannot parse the chemical formula 'H2O\\)'"),
    (("x", 1.0, "0.5*H2O + "), "cannot parse material term ''"),
    (("x", 1.0, "-0.5*H2O"), "cannot parse material term"),
    (("x", 1.0, "0*H + H2O"), "non-positive fraction"),
    (("x", 1.0, ""), "non-empty string"),
    (("x", 1.0, None), "non-empty string"),
    (("x", 1.0, "Xx2O"), "unknown element 'Xx'"),
    (("x", 1.0, "0.9*water + 0.1*Es"), "'Es' \\(Z = 99\\).*no scattering data"),
    (("x", 1.0, "0.9*watr + 0.1*I"), "unknown catalogue material 'watr'"),
    (("x", 0.0, "H2O"), "density must be > 0"),
    (("x", -1.0, "H2O"), "density must be > 0"),
    (("x", float("nan"), "H2O"), "density must be > 0"),
    (("x", "heavy", "H2O"), "density must be a number"),
    (("", 1.0, "H2O"), "non-empty string"),
    ((" x", 1.0, "H2O"), "without surrounding spaces"),
    (("water", 1.1, "H2O"), "'water' is a catalogue material"),
    (("lead", 11.35, "Pb2"), "'lead' is a catalogue material"),
])
def test_material_definition_errors(definition, message):
    with pytest.raises(ValueError, match=message):
        Material(*definition)


def test_duplicate_and_unknown_names():
    """Two reference materials with one name are refused, also a user copy of a catalogue entry next to the entry;
    an unknown catalogue name still names the catalogue and suggests a Material."""
    other = Material("iodine_solution", 1.02, "0.98*water + 0.02*I")
    with pytest.raises(ValueError, match="duplicate"):
        SolverConfig(materials=("water", IODINE, other)).validate()
    with pytest.raises(ValueError, match="duplicate"):
        SolverConfig(materials=("air", Material("water", 1.0, "H2O"), "water")).validate()
    with pytest.raises(ValueError, match="unknown material name.*polycarbonate.*Material\\(name, density"):
        SolverConfig(materials=("water", "polycarbonate")).validate()
    with pytest.raises(ValueError, match="at least 2"):
        SolverConfig(materials=(IODINE,)).validate()
    with pytest.raises(ValueError, match="sequence of material names"):
        SolverConfig(materials=IODINE).validate()


# ---- library builds -------------------------------------------------------------------------------------------------

def test_cache_key_holds_each_definition(library_module, tmp_path):
    """The default set keeps its cache keys (also given as a Material copy of water); a user material's density and
    composition are in the key, so two definitions under one name have two cache entries."""
    backend = library_module.resolve_backend()
    spectrum = library_module.spectrum_arrays(SPECTRUM_125KVP)
    key = library_module._cache_key
    assert key(SolverConfig(), spectrum, backend.name, 0.5, 32) == KEY_DEFAULT_V080
    assert key(SMALL, spectrum, backend.name, 2.0, 8) == KEY_SMALL_V080
    copies = SolverConfig(materials=tuple(Material("water", 1.0, "H2O") if n == "water" else n for n in MATERIALS))
    assert key(copies, spectrum, backend.name, 0.5, 32, material_table(copies.materials)) == KEY_DEFAULT_V080
    denser = replace(IODINE, density=1.02)
    stronger = replace(IODINE, composition="0.98*water + 0.02*I")
    keys = {key(SMALL, spectrum, backend.name, 2.0, 8, material_table(("air", "water", m))) for m in
            (IODINE, denser, stronger, MY_WATER)}
    assert len(keys) == 4 and KEY_SMALL_V080 not in keys
    a = library_module.build_library(SMALL, SPECTRUM_125KVP, cache_dir=str(tmp_path), materials=("water", IODINE),
                                     **SMALL_BUILD)
    b = library_module.build_library(SMALL, SPECTRUM_125KVP, cache_dir=str(tmp_path), materials=("water", stronger),
                                     **SMALL_BUILD)
    assert len(list(tmp_path.glob("mglib_*.pt"))) == 2 and a.materials == b.materials == ("water", "iodine_solution")
    assert bool((b.mu_t[1] > a.mu_t[1]).all())
    again = library_module.build_library(SMALL, SPECTRUM_125KVP, cache_dir=str(tmp_path), materials=(IODINE, "water"),
                                         **SMALL_BUILD)  # the cached entry
    assert torch.equal(again.mu_s_eta, a.mu_s_eta) and len(list(tmp_path.glob("mglib_*.pt"))) == 2


def test_copy_of_catalogue_material_builds_the_same_constants(library_module):
    """Water as H2O, water by its catalogue name and cortical bone with the catalogue's composition, under new names,
    build the catalogue materials' constants (float64)."""
    reference = _build(library_module, ("air", "water", "bone_cortical_icrp"))
    for copies in (("air", MY_WATER, MY_BONE), ("air", WATER_BY_NAME, "bone_cortical_icrp")):
        library = _build(library_module, copies)
        assert library.materials == tuple(m.name if isinstance(m, Material) else m for m in copies)
        assert torch.equal(library.nominal_density, reference.nominal_density)
        for field in LIBRARY_TENSORS:
            assert torch.allclose(getattr(library, field), getattr(reference, field), rtol=1e-14, atol=0), field
        assert torch.equal(library.group_edges, reference.group_edges) and torch.equal(library.eta_grid,
                                                                                      reference.eta_grid)


def test_iodine_solution_builds_and_orders(library_module, iodine_library):
    """1 % iodine in water sorts between soft tissue and bone among the defaults; its constants are the mass-fraction
    mix of the water and iodine constants; the preview attenuation agrees; 10 % iodine crosses bone and is refused."""
    backend = library_module.resolve_backend()
    extended = _build(library_module, MATERIALS + (IODINE,))
    assert extended.materials == MATERIALS[:6] + ("iodine_solution", "bone_cortical_icrp")
    assert material_order(extended) == list(range(8))
    assert extended.nominal_density[6].item() == 1.01
    parts = _build(library_module, ("water", "iodine"))
    row = material_index(extended, "iodine_solution")
    for field in LIBRARY_TENSORS:
        mix = 0.99 * getattr(parts, field)[0] + 0.01 * getattr(parts, field)[1]
        assert torch.allclose(getattr(extended, field)[row], mix, rtol=1e-12, atol=1e-18), field
    assert bool((extended.mu_pe[row] > 1.5 * extended.mu_pe[material_index(extended, "water")]).all())
    assert iodine_library.materials == ("air", "water", "iodine_solution", "bone_cortical_icrp")
    config = replace(SMALL, materials=WITH_IODINE)
    energy, mass = library_module.mass_attenuation_preview(config, ("iodine_solution", "water", "iodine"),
                                                           SPECTRUM_125KVP, backend, fine_bin_width_kev=2.0)
    assert mass[0] == pytest.approx(0.99 * mass[1] + 0.01 * mass[2], rel=1e-12)
    assert library_module.mass_attenuation_preview(SMALL, (IODINE,), energy_kev=energy)[1][0] == mass[0]
    with pytest.raises(ValueError, match="cross"):
        _build(library_module, ("water", Material("iodine_10_percent", 1.05, "0.9*water + 0.1*I"),
                                "bone_cortical_icrp"))


def test_user_material_as_grouping_medium(library_module):
    """A user material is a grouping medium by name, at its nominal density; its composition sets the group edges,
    also under the same name."""
    backend = library_module.resolve_backend()
    config = replace(SMALL, materials=WITH_IODINE, grouping_medium="iodine_solution")
    assert library_module.grouping_medium_densities(config) == (("iodine_solution", 1.01),)
    stronger = Material("iodine_solution", 1.01, "0.9*water + 0.1*I")
    other = replace(config, materials=("air", "water", stronger, "bone_cortical_icrp"))
    edges = [library_module.make_group_edges(c, SPECTRUM_125KVP, backend) for c in
             (replace(SMALL, materials=WITH_IODINE), config, other)]
    assert not np.allclose(edges[0], edges[1]) and not np.allclose(edges[1], edges[2])
    mixed = replace(config, grouping_medium={"water": 0.9, "iodine_solution": 0.2})
    library = _build(library_module, mixed.materials, mixed)
    assert library.materials == ("air", "water", "iodine_solution", "bone_cortical_icrp")
    spectrum = library_module.spectrum_arrays(SPECTRUM_125KVP)
    keys = {library_module._cache_key(c, spectrum, backend.name, 2.0, 8, material_table(c.materials))
            for c in (config, other, mixed)}
    assert len(keys) == 3


def test_lazy_solver_with_user_material(library_module):
    """A lazy solver sorts a user material among its materials and estimates a grouping medium that contains it,
    without building its library."""
    config = replace(SMALL_SOLVER, materials=(IODINE, "bone_cortical_icrp", "water", "air"))
    solver = ScatterSolver(None, config, wlab_geometry(), SiddonProjectors(), csi_response(), no_grid(),
                           spectrum=SPECTRUM_125KVP, device="cpu")
    assert solver.materials == ("air", "water", "iodine_solution", "bone_cortical_icrp")
    _, mass = library_module.mass_attenuation_preview(config, solver.materials, SPECTRUM_125KVP)
    X = torch.zeros(GRID.shape, dtype=torch.float64)
    X[2:8, 2:14, 2:18] = 1.0 * mass[1]  # water
    X[4:6, 6:8, 2:18] = 1.01 * mass[2]  # the iodine solution
    medium = dict(grouping_medium_estimate(X, solver, mask=X > 0))
    assert set(medium) == {"water", "iodine_solution"}
    assert medium["iodine_solution"] == pytest.approx(1.01 * 4 / 72, rel=1e-5)
    assert medium["water"] == pytest.approx(68 / 72, rel=1e-5)
    assert not solver.library_built


# ---- solves -------------------------------------------------------------------------------------------------------

def _small_solver(library, dtype=torch.float32):
    return ScatterSolver(library, SMALL_SOLVER, wlab_geometry(), SiddonProjectors(), csi_response(), no_grid(),
                         device=DEVICE, dtype=dtype)


def test_solve_view_with_user_materials(library_module, iodine_library):
    """A copy of water under another name gives the catalogue water's solve; the bone insert of WLAB filled with the
    iodine solution solves, attenuates more than water and less than bone."""
    set_memory_cap(4400.0)
    reference = _build(library_module, ("air", "water", "bone_cortical_icrp"))
    copy = _build(library_module, ("air", MY_WATER, "bone_cortical_icrp"))
    rho = wlab_phantom(GRID, materials=reference.materials, device=DEVICE)
    a = _small_solver(reference).solve_view(rho, GRID, 0.3)
    b = _small_solver(copy).solve_view(rho, GRID, 0.3)
    for name in ("primary", "scatter"):
        for k in ("signal", "planar_fluence"):
            assert torch.equal(getattr(a, name)[k], getattr(b, name)[k]), (name, k)
    rho = wlab_phantom(GRID, materials=iodine_library.materials, device=DEVICE)
    bone, iodine, water = (material_index(iodine_library, n) for n in ("bone_cortical_icrp", "iodine_solution",
                                                                      "water"))
    insert = rho[bone] > 0
    signals = {}
    for name, row, density in (("bone", bone, 1.85), ("iodine", iodine, 1.01), ("water", water, 1.0)):
        filled = rho.clone()
        filled[bone][insert] = 0.0
        filled[row][insert] = density
        result = _small_solver(iodine_library).solve_view(filled, GRID, 0.3)
        assert bool(torch.isfinite(result.scatter_signal).all()) and float(result.scatter_signal.sum()) > 0
        signals[name] = float(result.primary["signal"].sum())
    assert signals["bone"] < signals["iodine"] < signals["water"]


@cuda
def test_synthetic_data_and_citrus_recon_with_a_user_material(tmp_path):
    """End to end: data of a water block with a column of 1 % iodine solution from a lazy solver whose reference
    materials include the solution (the grouping medium measured from the object contains it, the meta records the
    definition); CITRUS reconstructs the column, and the density model of its library puts the solution there."""
    pytest.importorskip("mbirtorch")
    from citrus.citrus import CitrusConfig, DensityModel, citrus_recon, estimate_regularization, mbir_recon
    from citrus.recon.mbir_bridge import make_cone_model
    from citrus.utilities import synthesize_cone_beam_data, wlab_grid, wlab_solver

    set_memory_cap(4400.0)
    geometry = replace(wlab_geometry(), num_det_rows=46, num_det_channels=38, delta_det_row=1.6, delta_det_channel=1.6)
    config = replace(SMALL_SOLVER, materials=WITH_IODINE, transport_cell_cm=2.0)
    grid = wlab_grid(2.0)
    truth = wlab_solver(2.0, config=config, geometry=geometry, device="cuda", spectrum=SPECTRUM_125KVP,
                        cache_dir=str(tmp_path))
    names = truth.materials
    rho = torch.zeros((len(names),) + tuple(grid.shape), device="cuda")
    rho[names.index("water"), 2:8, 3:12, 4:16] = 1.0
    column = (slice(4, 6), slice(6, 8), slice(4, 16))
    rho[names.index("water")][column] = 0.0
    rho[names.index("iodine_solution")][column] = 1.01
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    data = synthesize_cone_beam_data(rho, grid, geometry, angles, truth, noise=False)
    meta = json.loads(json.dumps(data["meta"]))
    assert meta["materials"] == list(names) and "iodine_solution" in dict(meta["grouping_medium"])
    assert {"name": "iodine_solution", "density": 1.01, "composition": "0.99*water + 0.01*I"} in \
        meta["solver_config"]["materials"]
    library = truth.library
    solver = wlab_solver(2.0, config=truth.config, library=library, geometry=geometry, device="cuda")
    model = make_cone_model(grid, geometry, data["angles"], device="cuda")
    P = torch.as_tensor(data["P"], device="cuda")
    regularization = estimate_regularization(model, -torch.log(P / 1.0e5), torch.as_tensor(data["Y"], device="cuda")
                                             / 1.0e5)
    citrus_config = CitrusConfig(mbir_iterations_per_outer_iteration=2, num_outer_iterations=3)
    X, _ = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, solver, citrus_config,
                        regularization=regularization)
    X_none, _ = mbir_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, citrus_config,
                           regularization=regularization, device="cuda")
    X_true = torch.as_tensor(data["X_true"], device="cuda")
    error = {name: float((value - X_true).norm() / X_true.norm()) for name, value in (("citrus", X), ("none", X_none))}
    assert error["citrus"] < error["none"], error
    densities = DensityModel.from_library(library)(X)
    iodine = densities[names.index("iodine_solution")]
    centre = (slice(4, 6), slice(6, 8), slice(6, 14))
    outside = torch.zeros_like(iodine, dtype=torch.bool)
    outside[2:8, 3:12, 4:16] = True
    outside[3:7, 5:9, :] = False
    assert float(iodine[centre].mean()) > 0.5, float(iodine[centre].mean())
    assert float(iodine[outside].mean()) < 0.1 * float(iodine[centre].mean())
