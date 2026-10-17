"""The material catalogue and the user-selected reference materials (version 0.5.0): every catalogue entry builds in the
library; `SolverConfig.materials` selects the reference materials, the default set has one cache key and one library
(the legacy options of version 0.6.0 reproduce the library of version 0.5.1 bit for bit); any set is sorted by
nominal attenuation and has its own cache entry; the density model, the WLAB phantom, the data generator and the
solver work on a reduced set (air, water, bone) and on an extended one (the seven plus titanium); unknown names, a
single reference material and a set without water or bone for the WLAB phantom raise."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from citrus.citrus import DensityModel
from citrus.scatter_solver.config import SolverConfig
from citrus.scatter_solver.cross_sections import (BONE, COMPOUND_NAMES, ELEMENT_DENSITIES, ELEMENT_NAMES,
                                                  ELEMENT_SYMBOLS, MATERIAL_CATALOGUE, MATERIAL_TABLE, MATERIALS,
                                                  NOMINAL_DENSITIES, WATER, material_index,
                                                  material_table)
from citrus.scatter_solver.detector import csi_response, no_grid
from citrus.scatter_solver.geometry import Grid
from citrus.scatter_solver.ray_tracing import SiddonProjectors
from citrus.scatter_solver.scatter_solver import ScatterSolver
from citrus.utilities import (attenuation_to_densities, attenuation_to_density, densities_to_attenuation,
                              mass_attenuation_at, material_order, nominal_attenuation, set_memory_cap, wlab_geometry,
                              wlab_grid, wlab_phantom, wlab_setup)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
REDUCED = ("air", "water", "bone")
EXTENDED = MATERIALS + ("titanium",)
# the small, fast library build of tests/test_materials.py
SMALL = SolverConfig(num_groups=3, legendre_order=2, num_eta_table=51)
SMALL_BUILD = dict(fine_bin_width_kev=2.0, gauss_legendre_nodes=8)
# a solver configuration that runs on the small library (WLAB at 2 cm voxels in 4 cm cells)
SMALL_SOLVER = replace(SMALL, num_polar=4, num_azimuthal=8, cone_num_polar=8, cone_num_azimuthal=16,
                       transport_cell_cm=4.0, source_iteration_tolerance=1e-4, max_source_iterations=40)
GRID = Grid((10, 16, 20), 2.0)  # the WLAB problem of tests/test_materials.py
# cache keys of version 0.6.0 (default seven materials): SolverConfig() at 0.5 keV / 32 nodes, SMALL at 2 keV / 8
# (version 0.6.0 changed the build version and added the angular table grid, the group rule, the grouping object size
# and the grouping medium to the key)
KEY_DEFAULT_V060 = "5fada174f9a5ad6f6d10"
KEY_SMALL_V060 = "4f0c551f679bcdf821c3"
# the library options of versions <= 0.5 (uniform eta table, equal-fluence groups)
LEGACY_LIBRARY = dict(angular_table_grid="uniform", group_edges_method="equal_fluence")
# SHA-256 of the tensors of the SMALL build of version 0.5.1 (SMALL_BUILD, default seven materials; field name then the
# float64 / int64 bytes, in field order): the legacy options reproduce it bit for bit
SMALL_V051_SHA256 = "82733c2e779aa7a5667e7086b83f494685799dc010ad6aef2460778aac1f56a5"
REFERENCE = Path(__file__).parent / "data" / "pre_quadrature_reference.json"


@pytest.fixture(scope="module")
def library_module():
    cross_section_library = pytest.importorskip("citrus.scatter_solver.cross_section_library")
    try:
        cross_section_library.resolve_backend()
    except Exception as exc:
        pytest.skip(f"XrayPhysics unavailable ({exc})")
    return cross_section_library


@pytest.fixture(scope="module")
def small_libraries(library_module):
    """{set name: small library} for the default seven, the reduced and the extended set (no cache)."""
    backend = library_module.resolve_backend()
    return {name: library_module.compute_library(SMALL, None, backend, materials=material_table(names),
                                                 **SMALL_BUILD)[0]
            for name, names in (("default", MATERIALS), ("reduced", REDUCED), ("extended", EXTENDED))}


def _blocks(image, n: int = 8) -> np.ndarray:
    """Means of an image over n x n blocks (the layout of the reference, tests/test_quadrature_grid.py)."""
    image = image.detach().cpu().double().numpy()
    r = np.array_split(np.arange(image.shape[0]), n)
    c = np.array_split(np.arange(image.shape[1]), n)
    return np.array([[image[np.ix_(ri, ci)].mean() for ci in c] for ri in r])


def _check_reference(result, case: dict, tolerance: float) -> None:
    """A ViewResult against one case of the regression anchor (as tests/test_quadrature_grid.py does)."""
    def rel(a, b) -> float:
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        return float(np.abs(a - b).max() / np.abs(b).max())

    assert list(result.scatter_signal.shape) == case["image_shape"]
    assert rel(_blocks(result.scatter_signal), case["scatter_signal_blocks"]) < tolerance
    assert rel(_blocks(result.primary["signal"]), case["primary_signal_blocks"]) < tolerance
    assert abs(float(result.scatter_signal.double().sum()) / case["scatter_signal_sum"] - 1) < tolerance
    assert abs(float(result.primary["signal"].double().sum()) / case["primary_signal_sum"] - 1) < tolerance
    assert result.source_iteration_result.num_iterations == case["si_iterations"]


def _small_solver(library, dtype=torch.float32):
    return ScatterSolver(library, SMALL_SOLVER, wlab_geometry(), SiddonProjectors(), csi_response(), no_grid(),
                         device=DEVICE, dtype=dtype)


# ---- catalogue and names ---------------------------------------------------------------------------------------

def test_catalogue_and_default_set():
    assert tuple(MATERIAL_CATALOGUE)[:7] == MATERIALS
    assert MATERIALS == ("air", "lung", "adipose", "water", "muscle", "soft_tissue", "bone")
    assert set(MATERIAL_CATALOGUE) >= {"trabecular_bone", "brain", "blood", "cartilage", "skin", "pmma",
                                       "polyethylene", "aluminium", "titanium", "carbon_fibre"}
    assert MATERIAL_TABLE == tuple((n, MATERIAL_CATALOGUE[n][0]) for n in MATERIALS)
    assert NOMINAL_DENSITIES == (0.0012, 0.26, 0.95, 1.0, 1.05, 1.06, 1.85) and (WATER, BONE) == (3, 6)
    for name, (density, formula) in MATERIAL_CATALOGUE.items():
        assert density > 0 and isinstance(formula, str) and formula, name
    with pytest.raises(TypeError):
        MATERIAL_CATALOGUE["unobtainium"] = (1.0, "U")  # read-only
    assert SolverConfig().materials == MATERIALS
    assert SolverConfig(materials=["bone", "water", "air"]).materials == ("bone", "water", "air")  # a tuple
    assert material_table(REDUCED) == (("air", 0.0012, MATERIAL_CATALOGUE["air"][1]), ("water", 1.0, "H2O"),
                                       ("bone", 1.85, MATERIAL_CATALOGUE["bone"][1]))
    assert material_index(MATERIALS, "bone") == BONE and material_index(REDUCED, "bone") == 2


def test_material_names_are_checked():
    SolverConfig(materials=REDUCED).validate()
    with pytest.raises(ValueError, match="unknown material name.*unobtainium.*catalogue.*English name"):
        SolverConfig(materials=("water", "unobtainium")).validate()
    with pytest.raises(ValueError, match="at least 2"):
        SolverConfig(materials=("water",)).validate()
    with pytest.raises(ValueError, match="duplicate"):
        SolverConfig(materials=("water", "bone", "water")).validate()
    with pytest.raises(ValueError, match="sequence of material names"):
        SolverConfig(materials="air,water,bone").validate()
    with pytest.raises(ValueError, match="unknown"):
        material_table(("water", "Water"))
    with pytest.raises(ValueError, match="not among the materials"):
        material_index(REDUCED, "lung")
    with pytest.raises(ValueError, match="at least 2"):  # the solver validates its configuration
        ScatterSolver(None, replace(SMALL_SOLVER, materials=("bone",)), wlab_geometry(), SiddonProjectors(),
                      csi_response(), no_grid())


def test_wlab_phantom_by_name():
    """The phantom's rows are looked up by name: water and bone of any set; a set without them raises."""
    grid = wlab_grid(2.0)
    rho7 = wlab_phantom(grid, dtype=torch.float64)
    rho3 = wlab_phantom(grid, materials=REDUCED, dtype=torch.float64)
    shuffled = ("bone", "titanium", "water", "lung")
    rho4 = wlab_phantom(grid, materials=shuffled, dtype=torch.float64)
    assert rho3.shape == (3, *grid.shape) and rho4.shape == (4, *grid.shape)
    assert torch.equal(rho3[1], rho7[WATER]) and torch.equal(rho3[2], rho7[BONE]) and float(rho3[0].abs().max()) == 0
    assert torch.equal(rho4[2], rho7[WATER]) and torch.equal(rho4[0], rho7[BONE])
    assert torch.equal(wlab_setup(2.0, dtype=torch.float64, materials=REDUCED)[2], rho3)
    for names in (("air", "lung", "water"), ("air", "bone"), ("titanium", "aluminium")):
        with pytest.raises(ValueError, match="WLAB phantom needs 'water' and 'bone'"):
            wlab_phantom(grid, materials=names)


# ---- library builds -------------------------------------------------------------------------------------------

def test_every_catalogue_entry_parses(library_module):
    """Every catalogue entry (the compounds and the 100 elements) has a positive density and a composition that
    parses to mass fractions summing to 1; the elements follow the compounds, one per atomic number 1..100 (aluminium
    and titanium among the compounds), with XrayPhysics' standard densities."""
    backend = library_module.resolve_backend()
    names = tuple(MATERIAL_CATALOGUE)
    assert names[:len(COMPOUND_NAMES)] == COMPOUND_NAMES and len(set(names)) == len(names)
    assert len(ELEMENT_NAMES) == len(ELEMENT_SYMBOLS) == len(ELEMENT_DENSITIES) == 100
    assert set(ELEMENT_NAMES) <= set(names) and {"lead", "tungsten", "carbon"} <= set(names)
    for name, (density, formula) in MATERIAL_CATALOGUE.items():
        composition = library_module.parse_formula(formula, backend.atomic_mass)
        assert density > 0 and sum(composition.values()) == pytest.approx(1.0, abs=1e-12), name
    for Z, (name, symbol, density) in enumerate(zip(ELEMENT_NAMES, ELEMENT_SYMBOLS, ELEMENT_DENSITIES), start=1):
        assert MATERIAL_CATALOGUE[name][1] == symbol
        assert library_module.parse_formula(symbol, backend.atomic_mass) == {Z: 1.0}
        assert density == pytest.approx(backend._xray_physics.massDensity(Z), rel=1e-5), name
    assert MATERIAL_CATALOGUE["lead"] == (11.35, "Pb") and MATERIAL_CATALOGUE["tungsten"] == (19.3, "W")


# the compounds and a sample of the elements (every 10th atomic number, lead and tungsten): one build each
SAMPLE_ELEMENTS = tuple(ELEMENT_NAMES[Z - 1] for Z in range(10, 101, 10)) + ("lead", "tungsten")


@pytest.mark.parametrize("name", list(COMPOUND_NAMES) + list(SAMPLE_ELEMENTS))
def test_every_catalogue_material_builds(name, library_module):
    """Each compound and each sampled element alone builds a library (small configuration): shapes, positive
    coefficients, the nominal density, a composition whose mass fractions sum to 1 (lead and tungsten have their K
    edges at 88 and 69.5 keV, inside the 125 kVp spectrum)."""
    backend = library_module.resolve_backend()
    library, info = library_module.compute_library(SMALL, None, backend, materials=material_table((name,)),
                                                   **SMALL_BUILD)
    assert library.materials == (name,) and library.nominal_density.tolist() == [MATERIAL_CATALOGUE[name][0]]
    assert tuple(library.mu_t.shape) == (1, 3) and tuple(library.mu_s_eta.shape[:3]) == (1, 3, 3)
    assert bool((library.mu_t > 0).all()) and bool((library.fine_mu_t > 0).all())
    assert bool((library.mu_s_l[..., 0] >= 0).all()) and bool(torch.isfinite(library.mu_s_eta).all())
    composition = library_module.material_composition(backend, material_table((name,)))[name]
    assert sum(composition.values()) == pytest.approx(1.0, abs=1e-12)


def test_default_set_keeps_the_cache_key_and_library(library_module, tmp_path, monkeypatch):
    """The default reference materials give the cache keys of version 0.6.0 and the same library bit for bit, whether
    the table is None, the default set by name or the same set in another order (sorted by nominal attenuation before
    the key and the build); the options of version 0.5 (uniform table, equal-fluence groups) reproduce the library of
    version 0.5.1 bit for bit."""
    backend = library_module.resolve_backend()
    spectrum = library_module._spectrum_or_default(None, backend)
    reverse = material_table(tuple(reversed(MATERIALS)))
    assert (library_module.sort_material_table(SMALL, reverse, spectrum, backend, fine_bin_width_kev=2.0)
            == material_table(MATERIALS) == library_module.default_material_table())
    key = library_module._cache_key
    assert key(SolverConfig(), spectrum, backend.name, 0.5, 32) == KEY_DEFAULT_V060
    assert key(SolverConfig(), spectrum, backend.name, 0.5, 32, material_table(MATERIALS)) == KEY_DEFAULT_V060
    assert key(SMALL, spectrum, backend.name, 2.0, 8) == KEY_SMALL_V060
    legacy, _ = library_module.compute_library(replace(SMALL, **LEGACY_LIBRARY), None, backend, **SMALL_BUILD)
    digest = hashlib.sha256()
    for field in legacy.__dataclass_fields__:
        value = getattr(legacy, field)
        if isinstance(value, torch.Tensor):
            digest.update(field.encode())
            digest.update(value.contiguous().numpy().tobytes())
    assert legacy.eta_map is None and digest.hexdigest() == SMALL_V051_SHA256
    reference, _ = library_module.compute_library(SMALL, None, backend, **SMALL_BUILD)
    by_name = library_module.build_library(SMALL, None, cache_dir=str(tmp_path), materials=reverse, **SMALL_BUILD)
    assert [p.name for p in tmp_path.glob("mglib_*.pt")] == [f"mglib_{KEY_SMALL_V060}.pt"]
    for field in reference.__dataclass_fields__:
        a, b = getattr(reference, field), getattr(by_name, field)
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b, field
    assert reference.eta_map is not None and not torch.equal(reference.mu_s_eta, legacy.mu_s_eta)
    # build_solver_library(config) builds config.materials: the default set is the same cache entry
    from citrus.scatter_solver.scatter_solver import build_solver_library

    def cache_miss(*args, **keywords):
        raise AssertionError("cache miss")

    with monkeypatch.context() as patch:
        patch.setattr(library_module, "compute_library", cache_miss)
        assert library_module.build_library(SMALL, None, cache_dir=str(tmp_path), **SMALL_BUILD).materials == MATERIALS
    if (Path(library_module.default_cache_dir()) / f"mglib_{KEY_DEFAULT_V060}.pt").is_file():
        assert build_solver_library(SolverConfig()).materials == MATERIALS  # the cached default library


def test_each_set_has_its_own_cache_entry(library_module, small_libraries, tmp_path):
    """A set and any permutation of it share one cache entry, distinct from the default set's; the library comes out
    sorted by nominal attenuation; its water and bone rows equal the default library's to summation order."""
    build = library_module.build_library
    a = build(SMALL, None, cache_dir=str(tmp_path), materials=material_table(("bone", "water", "air")), **SMALL_BUILD)
    b = build(SMALL, None, cache_dir=str(tmp_path), materials=material_table(REDUCED), **SMALL_BUILD)
    c = build(SMALL, None, cache_dir=str(tmp_path), **SMALL_BUILD)
    names = sorted(p.name for p in tmp_path.glob("mglib_*.pt"))
    assert len(names) == 2 and f"mglib_{KEY_SMALL_V060}.pt" in names
    assert a.materials == b.materials == REDUCED and c.materials == MATERIALS
    assert torch.equal(a.mu_s_eta, b.mu_s_eta)
    extended = small_libraries["extended"]
    assert extended.materials == EXTENDED  # titanium attenuates most: sorted last
    assert material_order(extended) == list(range(8))
    default = small_libraries["default"]
    for k in ("mu_t", "mu_pe", "mu_s_l", "mu_s_eta", "fine_mu_t"):
        for library in (a, extended):
            rows = [material_index(library, "water"), material_index(library, "bone")]
            assert torch.allclose(getattr(library, k)[rows], getattr(default, k)[[WATER, BONE]], rtol=1e-13,
                                  atol=1e-16), k
    # a set whose members cross in attenuation is refused (skin crosses soft tissue and blood between 20 and 63 keV)
    with pytest.raises(ValueError, match="cross"):
        library_module.compute_library(SMALL, None, library_module.resolve_backend(),
                                       materials=material_table(("water", "soft_tissue", "blood", "skin")),
                                       **SMALL_BUILD)


# ---- density model on a reduced set ----------------------------------------------------------------------------

def test_reduced_set_density_model(small_libraries, device):
    """Reference materials air, water, bone: a water voxel is water, a bone voxel bone, a value between water and bone
    a water /
    bone mixture of the same attenuation, between air and water an air / water mixture; below air air of reduced
    density, above bone bone of higher density; X < 0 gives zero; DensityModel (from the library) does the same, one
    material at a time bit for bit."""
    library = small_libraries["reduced"]
    mu = nominal_attenuation(library)  # [air, water, bone]
    m = mass_attenuation_at(library, DensityModel.from_library(library).energy_kev)
    assert bool((torch.diff(mu) > 0).all())
    X = torch.tensor([float(mu[1]), float(mu[2]), 0.25 * float(mu[1]) + 0.75 * float(mu[2]),
                      0.5 * (float(mu[0]) + float(mu[1])), 0.5 * float(mu[0]), 1.5 * float(mu[2]), -0.1],
                     dtype=torch.float64, device=device)
    rho = attenuation_to_densities(library, X)
    expect = torch.zeros(3, 7, dtype=torch.float64, device=device)
    expect[1, 0] = 1.0  # water
    expect[2, 1] = 1.85  # bone
    expect[1, 2], expect[2, 2] = 0.25 * 1.0, 0.75 * 1.85  # water / bone mixture
    expect[0, 3], expect[1, 3] = 0.5 * 0.0012, 0.5 * 1.0  # air / water mixture
    expect[0, 4] = 0.5 * float(mu[0]) / float(m[0])  # air of reduced density
    expect[2, 5] = 1.5 * float(mu[2]) / float(m[2])  # bone, denser than nominal
    assert torch.allclose(rho, expect, rtol=1e-12, atol=1e-15)
    assert torch.allclose(densities_to_attenuation(library, rho), X.clamp_min(0), rtol=1e-12, atol=1e-15)
    model = DensityModel.from_library(library)
    full = model(X.float())
    assert full.shape == (3, 7)
    for k in range(3):
        assert torch.equal(model.material(X.float(), k), full[k])
        assert torch.equal(attenuation_to_density(library, X, k), rho[k])
    # the WLAB phantom: water and bone come back as themselves; lung and air (water of reduced density) are air /
    # water mixtures of the same attenuation
    phantom = wlab_phantom(wlab_grid(2.0), materials=library.materials, device=device, dtype=torch.float64)
    back = attenuation_to_densities(library, densities_to_attenuation(library, phantom))
    pure = (phantom[1] == 1.0) | (phantom[2] > 0)
    assert torch.allclose(back[:, pure], phantom[:, pure], rtol=1e-12, atol=1e-12)
    inserts = (phantom[1] > 0) & (phantom[1] < 1.0)
    assert bool((back[0][inserts] > 0).all()) and bool((back[1][inserts] > 0).all())
    assert float(back[2][inserts].abs().max()) == 0.0


# ---- the solver on other sets ----------------------------------------------------------------------------------

@pytest.mark.parametrize("set_name", ["reduced", "extended"])
def test_solve_view_on_other_sets_equals_default(set_name, small_libraries):
    """WLAB (water and bone only) solved with the reduced (air, water, bone) and the extended (seven plus titanium)
    reference materials equals the default-set solve to float32 rounding (the water and bone rows agree to summation
    order); a
    titanium rod in the extended set changes the solution and stays finite."""
    set_memory_cap(4400.0)
    library = small_libraries[set_name]
    rho7 = wlab_phantom(GRID, device=DEVICE)
    rho = wlab_phantom(GRID, materials=library.materials, device=DEVICE)
    a = _small_solver(library).solve_view(rho, GRID, 0.3)
    b = _small_solver(small_libraries["default"]).solve_view(rho7, GRID, 0.3)
    assert a.source_iteration_result.num_iterations == b.source_iteration_result.num_iterations
    for name in ("primary", "scatter"):
        for k in ("signal", "planar_fluence"):
            x, y = getattr(a, name)[k], getattr(b, name)[k]
            assert float((x - y).abs().max()) <= 2e-5 * float(y.abs().max()), (set_name, name, k)
    if set_name == "extended":
        titanium = material_index(library, "titanium")
        rod = rho.clone()
        rod[:, 4:6, 6:8] = 0.0
        rod[titanium, 4:6, 6:8] = 4.54
        c = _small_solver(library).solve_view(rod, GRID, 0.3)
        assert bool(torch.isfinite(c.scatter_signal).all()) and bool(torch.isfinite(c.primary["signal"]).all())
        assert float(c.primary["signal"].sum()) < float(a.primary["signal"].sum())


@cuda
def test_synthesize_wlab_dataset_on_a_reduced_set(small_libraries):
    """synthesize_wlab_dataset places the phantom in the library's water and bone rows (any set): the data of the
    reduced set equal those of the default set (the phantom is water and bone only); a library without water or bone
    is refused before any solve."""
    pytest.importorskip("mbirtorch")
    from citrus.utilities import synthesize_wlab_dataset

    set_memory_cap(4400.0)
    out = {}
    for name in ("default", "reduced"):
        out[name] = synthesize_wlab_dataset(2, 2.0, noise=False, config=SMALL_SOLVER, library=small_libraries[name],
                                            geometry=wlab_geometry(), device="cuda")
    reduced, default = out["reduced"], out["default"]
    assert reduced["X_true"].shape == tuple(wlab_grid(2.0).shape) and reduced["meta"]["materials"] == list(REDUCED)
    np.testing.assert_allclose(reduced["X_true"], default["X_true"], rtol=1e-6)
    np.testing.assert_allclose(reduced["SPR"], default["SPR"], rtol=1e-4)
    with pytest.raises(ValueError, match="WLAB phantom needs 'water' and 'bone'"):
        synthesize_wlab_dataset(2, 2.0, noise=False, config=SMALL_SOLVER, geometry=wlab_geometry(), device="cuda",
                                library=replace(small_libraries["reduced"], materials=("air", "water", "titanium")))


@cuda
def test_regression_anchors_default_and_reduced_sets():
    """The real library built through SolverConfig.materials reproduces the regression anchor of version 0.1.0 on WLAB
    (tests/data/pre_quadrature_reference.json, 1 cm voxels in 1 cm cells) for the default seven (the library of
    earlier versions: uniform eta table, equal-fluence groups) and for the reduced set air, water, bone (the WLAB
    phantom is water and bone only)."""
    from citrus.utilities import wlab_solver

    set_memory_cap(4400.0)
    case = json.loads(REFERENCE.read_text())["cases"]["wlab_1p0cm_f1"]
    grid = wlab_grid(1.0)
    for names in (MATERIALS, REDUCED):
        s = wlab_solver(1.0, config=replace(SolverConfig(), materials=names, **LEGACY_LIBRARY), device="cuda")
        assert s.library.materials == names
        _check_reference(s.solve_view(wlab_phantom(grid, materials=names, device="cuda"), grid, 0.3), case, 1e-5)
