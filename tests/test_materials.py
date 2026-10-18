"""The seven-material basis: the material table and its library (order check, cache key), the compact pair
representation in the first-collision source (PyTorch and Triton pair paths), the solver on seven materials against
the two-material solve, and the interpolating density model (citrus.utilities.attenuation_to_densities).

Ported from the research repository's tests/test_materials.py (the parts that apply to the release: no legacy
mapping, no local effective energy, one built-in material table)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from citrus.citrus import DensityModel
from citrus.scatter_solver.config import SolverConfig
from citrus.scatter_solver.cross_sections import (BONE, MATERIAL_TABLE, MATERIALS, NOMINAL_DENSITIES, WATER,
                                                  MultigroupLibrary, material_pairs, synthetic_library)
from citrus.scatter_solver.detector import csi_response, no_grid
from citrus.scatter_solver.geometry import ConeGeometry, Grid
from citrus.scatter_solver.ray_tracing import SiddonProjectors
from citrus.scatter_solver.scatter_solver import ScatterSolver
from citrus.scatter_solver.uncollided import (_first_collision_source_torch, _first_collision_source_torch_pairs,
                                              first_collision_source_at_directions,
                                              present_materials)
from citrus.utilities import (RHO_BONE, attenuation_to_densities, densities_to_attenuation, effective_energy,
                              mass_attenuation_at, material_order, nominal_attenuation, set_memory_cap, wlab_geometry,
                              wlab_phantom)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _cap() -> None:
    set_memory_cap(4400.0)


def _two_material_table():
    from citrus.scatter_solver.cross_section_library import MATERIAL_FORMULAS

    return [("water", 1.0, MATERIAL_FORMULAS["water"]), ("bone", 1.85, MATERIAL_FORMULAS["bone"])]


@pytest.fixture(scope="module")
def libraries():
    """(two-material water/bone, seven-material) real libraries for SolverConfig() (library cache)."""
    from citrus.scatter_solver.scatter_solver import build_solver_library

    try:
        return (build_solver_library(SolverConfig(), materials=_two_material_table()),
                build_solver_library(SolverConfig()))
    except Exception as exc:  # no cached library and no photon data
        pytest.skip(f"real library unavailable: {exc}")


def random_pairs_rho(M: int, shape, seed: int = 0, dtype=torch.float64) -> torch.Tensor:
    """rho [M, *shape] with at most two nonzero materials per voxel (and some empty voxels)."""
    g = torch.Generator().manual_seed(seed)
    rho = torch.zeros((M,) + tuple(shape), dtype=dtype)
    sel = torch.randint(0, M, (2,) + tuple(shape), generator=g)
    rho.scatter_(0, sel, torch.rand((2,) + tuple(shape), generator=g, dtype=dtype) + 0.05)
    rho[:, 0, 0] = 0.0
    return rho


# ---- material table and libraries --------------------------------------------------------------------------

def test_material_table_and_synthetic_libraries():
    assert MATERIALS == ("air", "lung", "adipose", "water", "muscle", "soft_tissue", "bone")
    assert NOMINAL_DENSITIES == (0.0012, 0.26, 0.95, 1.0, 1.05, 1.06, 1.85) and (WATER, BONE) == (3, 6)
    assert tuple(n for n, _ in MATERIAL_TABLE) == MATERIALS
    library2 = synthetic_library()
    assert library2.num_materials == 2 and library2.materials == ("water", "bone")
    assert library2.nominal_density.tolist() == [1.0, 1.85]
    library7 = synthetic_library(num_materials=7, fine_spread=0.5)
    assert library7.num_materials == 7 and library7.materials == MATERIALS
    assert library7.nominal_density.tolist() == list(NOMINAL_DENSITIES)
    # water and bone rows are exactly the two-material library's, the table is in attenuation order
    for k in ("mu_t", "mu_pe", "mu_s_l", "mu_s_eta"):
        assert torch.equal(getattr(library7, k)[[WATER, BONE]], getattr(library2, k)), k
    from citrus.scatter_solver.cross_section_library import check_material_order

    check_material_order(library7.fine_mu_t.numpy(), library7.nominal_density.numpy(), library7.materials)
    assert material_order(library7) == list(range(7))
    library5 = synthetic_library(num_materials=5)
    assert library5.materials == ("m0", "m1", "m2", "m3", "m4") and library5.nominal_density.tolist() == [1, 2, 3, 4, 5]
    # .to() and dataclasses.replace keep names and nominal densities
    l32 = library7.to(dtype=torch.float32)
    assert l32.materials == MATERIALS and l32.nominal_density.dtype == torch.float32
    assert replace(library7, mu_s_l=library7.mu_s_l[..., :2]).materials == MATERIALS
    with pytest.raises(ValueError):
        replace(library7, materials=("a", "b"))
    with pytest.raises(ValueError):
        synthetic_library(num_materials=1)


def test_library_build_order_and_cache(tmp_path, monkeypatch):
    """The seven-material library from the photon data (small config): names, nominal densities, shapes; its water and
    bone rows equal those of a two-material (water, bone) table to summation order; distinct cache keys (the
    two-material caches of citrus < 0.1.0 are never picked up), cache hit on rebuild, the order check refuses a
    crossing table."""
    cross_section_library = pytest.importorskip("citrus.scatter_solver.cross_section_library")
    try:
        cross_section_library.resolve_backend()
    except Exception as exc:
        pytest.skip(f"photon data unavailable ({exc})")
    config = SolverConfig(num_groups=3, legendre_order=2, num_eta_table=51)
    library = cross_section_library.build_library(config, None, cache_dir=str(tmp_path), fine_bin_width_kev=2.0,
                                                  gauss_legendre_nodes=8)
    wb = cross_section_library.build_library(config, None, cache_dir=str(tmp_path), fine_bin_width_kev=2.0,
                                             gauss_legendre_nodes=8, materials=_two_material_table())
    assert len(list(tmp_path.glob("mglib_*.pt"))) == 2  # two keys
    assert library.materials == MATERIALS and library.num_materials == 7 and wb.materials == ("water", "bone")
    assert library.nominal_density.tolist() == list(NOMINAL_DENSITIES)
    assert tuple(library.mu_s_eta.shape[:3]) == (7, 3, 3)
    assert tuple(library.fine_mu_t.shape) == (7, library.num_fine_energies)
    for k in ("mu_t", "mu_pe", "mu_s_l", "mu_s_eta", "fine_mu_t"):
        a, b = getattr(wb, k), getattr(library, k)[[WATER, BONE]]
        assert torch.allclose(a, b, rtol=1e-13, atol=1e-16), k
    assert torch.equal(wb.fine_weight, library.fine_weight) and torch.equal(wb.group_energy, library.group_energy)
    # the cache key depends on the table and on the build version (2-material keys of citrus 0.0.x differ)
    spec = cross_section_library._spectrum_or_default(None)
    k7 = cross_section_library._cache_key(config, spec, "xcom-xraylib", 2.0, 8)
    assert k7 != cross_section_library._cache_key(config, spec, "xcom-xraylib", 2.0, 8, _two_material_table())
    assert k7 != cross_section_library._cache_key(config, spec, "xrayphysics", 2.0, 8)  # the backend is in the key
    monkeypatch.setattr(cross_section_library, "_BUILD_VERSION", "citrus-xsec-1")
    assert cross_section_library._cache_key(config, spec, "xcom-xraylib", 2.0, 8) != k7
    monkeypatch.undo()

    def boom(*a, **k):
        raise AssertionError("cache miss")

    monkeypatch.setattr(cross_section_library, "compute_library", boom)
    library2 = cross_section_library.build_library(config, None, cache_dir=str(tmp_path), fine_bin_width_kev=2.0,
                                                   gauss_legendre_nodes=8)
    assert library2.materials == MATERIALS and torch.equal(library2.mu_s_eta, library.mu_s_eta)
    assert torch.equal(library2.nominal_density, library.nominal_density)
    monkeypatch.undo()
    # a library file without nominal densities (an older format) is rebuilt, not returned
    path = str(next(p for p in tmp_path.glob("mglib_*.pt") if MultigroupLibrary.load(str(p)).num_materials == 7))
    d = torch.load(path)
    del d["nominal_density"]
    torch.save(d, path)
    with pytest.raises(KeyError):
        MultigroupLibrary.load(path)
    with pytest.warns(RuntimeWarning, match="rebuilding"):
        library3 = cross_section_library.build_library(config, None, cache_dir=str(tmp_path), fine_bin_width_kev=2.0,
                                                       gauss_legendre_nodes=8)
    assert torch.equal(library3.nominal_density, library.nominal_density)
    # a table whose nominal attenuations cross is refused: soft tissue at 1.0 g/cm^3 crosses water (50-60 keV)
    bad = list(cross_section_library.default_material_table())
    bad[5] = ("soft_tissue", 1.0, bad[5][2])
    with pytest.raises(ValueError, match="cross"):
        cross_section_library.compute_library(config, None, cross_section_library.resolve_backend(),
                                              fine_bin_width_kev=2.0, gauss_legendre_nodes=8,
                                              materials=bad)


def test_material_order_real_library(libraries):
    from citrus.scatter_solver.cross_section_library import check_material_order

    wb, library = libraries
    assert material_order(library) == list(range(7)) and material_order(wb) == [0, 1]
    a = library.nominal_density[:, None] * library.fine_mu_t  # nominal attenuation at every fine energy (10-125 keV)
    assert bool((torch.diff(a, dim=0) > 0).all())
    mu = nominal_attenuation(library)  # at E_eff = 63.03 keV
    # XCOM + xraylib data (citrus >= 0.7.0)
    np.testing.assert_allclose(mu.numpy(), [0.00022, 0.0520, 0.1847, 0.2014, 0.2102, 0.2123, 0.5365], atol=6e-5)
    assert float(mu[WATER]) == pytest.approx(float(mass_attenuation_at(wb, effective_energy(wb))[0]), rel=1e-12)
    # muscle (1.05) is below soft tissue (1.06) at every energy: ratio 0.986-0.991
    r = library.fine_mu_t[4] * 1.05 / (library.fine_mu_t[5] * 1.06)
    assert 0.98 < float(r.min()) and float(r.max()) < 0.995
    with pytest.raises(ValueError, match="cross"):
        check_material_order(library.fine_mu_t[[3, 5]].numpy(), np.array([1.0, 1.0]), ["water", "soft_tissue_1.0"])
    nd = library.nominal_density.clone()
    nd[5] = 1.0
    with pytest.raises(ValueError, match="cross"):
        material_order(replace(library, nominal_density=nd))


# ---- pair representation and the first-collision source
# -------------------------------------------------------------------

def test_material_pairs_round_trip():
    rho = random_pairs_rho(7, (5, 4, 3))
    p = material_pairs(rho)
    assert p.index.dtype == torch.int32 and tuple(p.index.shape) == (2, 5, 4, 3) and p.num_materials == 7
    assert torch.equal(p.dense(), rho)
    both = p.density[1] != 0
    assert bool((p.index[0][both] < p.index[1][both]).all())
    bad = rho.clone()
    bad[:, 1, 1, 1] = 0.3
    with pytest.raises(ValueError, match="at most 2"):
        material_pairs(bad)
    w = wlab_phantom(Grid((10, 15, 20), 2.0), dtype=torch.float64)
    assert torch.equal(material_pairs(w).dense(), w)
    assert present_materials(w) == [WATER, BONE] and present_materials(w[[WATER, BONE]]) is None


@pytest.mark.parametrize("G", [4, 6])
def test_fc_source_pairs_match_dense(G, device):
    """Pair path (gather by index) == the dense M-material PyTorch path, float64 (CPU and CUDA)."""
    library = synthetic_library(G, legendre_order=2, num_eta=301, num_materials=7)
    grid = Grid(shape=(6, 7, 5), delta_voxel=1.0, voxel_slice_aspect=0.8, recon_slice_offset=0.3)
    geometry = ConeGeometry(source_iso_dist=20.0, source_detector_dist=30.0, num_det_rows=8, num_det_channels=8,
                            delta_det_row=1.0, delta_det_channel=1.0)
    rho = random_pairs_rho(7, grid.shape, seed=G).to(device)
    g = torch.Generator().manual_seed(3)
    phi = torch.rand(G, *grid.shape, generator=g, dtype=torch.float64).to(device)
    directions = torch.nn.functional.normalize(torch.randn(7, 3, generator=g, dtype=torch.float64), dim=-1).to(device)
    dense = _first_collision_source_torch(library, rho, phi, grid, geometry, 0.7, directions, chunk=3)
    pairs = _first_collision_source_torch_pairs(library, material_pairs(rho), phi, grid, geometry, 0.7, directions,
                                                chunk=3)
    assert torch.allclose(pairs, dense, rtol=1e-13, atol=1e-16 * float(dense.abs().max()))
    assert torch.equal(first_collision_source_at_directions(library, rho, phi, grid, geometry, 0.7, directions,
                                                            chunk=3), pairs)


@cuda
@pytest.mark.parametrize("G,K", [(4, 1), (6, 8), (6, 13)])
def test_fc_source_pair_kernel_matches_torch(G, K):
    """The Triton kernel's pair mode == the PyTorch pair path, float32, 7 materials, random pairs (research: 1.4e-7 of
    the field maximum)."""
    from citrus.scatter_solver.kernels import uncollided_kernels

    _cap()
    if not uncollided_kernels.kernel_usable(torch.device("cuda"))[0]:
        pytest.skip("first-collision-source kernel unusable on this machine")
    library = synthetic_library(G, legendre_order=1, num_eta=401, num_materials=7)
    grid = Grid(shape=(9, 8, 7), delta_voxel=1.0, voxel_slice_aspect=0.8, recon_slice_offset=0.3)
    geometry = ConeGeometry(source_iso_dist=20.0, source_detector_dist=30.0, num_det_rows=8, num_det_channels=8,
                            delta_det_row=1.0, delta_det_channel=1.0)
    rho = random_pairs_rho(7, grid.shape, seed=K).float().cuda()
    g = torch.Generator().manual_seed(K)
    phi = torch.rand(G, *grid.shape, generator=g).cuda()
    directions = torch.nn.functional.normalize(torch.randn(K, 3, generator=g), dim=-1).cuda()
    pairs = material_pairs(rho)
    tri = first_collision_source_at_directions(library, rho, phi, grid, geometry, 0.4, directions, backend="triton",
                                               pairs=pairs)
    reference = first_collision_source_at_directions(library, rho, phi, grid, geometry, 0.4, directions,
                                                     backend="torch", pairs=pairs)
    assert float((tri - reference).abs().max()) <= 1e-5 * float(reference.abs().max())
    # "auto" = kernel
    assert torch.equal(first_collision_source_at_directions(library, rho, phi, grid, geometry, 0.4, directions), tri)
    empty = rho.abs().sum(0) == 0
    assert float(tri[:, :, empty].abs().max()) == 0.0


# ---- the solver on seven materials against the two-material solve ---------------------------------------------

GRID = Grid((10, 16, 20), 2.0)  # the WLAB problem of tests/test_scatter_solver.py
CONFIG = SolverConfig(num_groups=4, legendre_order=3, num_polar=4, num_azimuthal=8, cone_num_polar=8,
                      cone_num_azimuthal=16, transport_cell_cm=4.0, source_iteration_tolerance=1e-4,
                      max_source_iterations=40, direction_chunk=8)


def _solver(library, config=CONFIG, dtype=torch.float32):
    return ScatterSolver(library, config, wlab_geometry(), SiddonProjectors(), csi_response(), no_grid(), device=DEVICE,
                         dtype=dtype)


def _compare_views(a, b, rtol):
    assert a.source_iteration_result.num_iterations == b.source_iteration_result.num_iterations
    for name in ("primary", "scatter"):
        for k in ("signal", "planar_fluence"):
            x, y = getattr(a, name)[k], getattr(b, name)[k]
            assert float((x - y).abs().max()) <= rtol * float(y.abs().max()), (name, k)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_solver_seven_materials_equals_two_materials_synthetic(dtype):
    """WLAB in the seven-material synthetic library (water and bone rows = the two-material library's): the pair path,
    the per-material traces and the seven-material linear fields reproduce the two-material solve to float32 rounding
    (float64: 1e-12)."""
    _cap()
    library2 = synthetic_library(CONFIG.num_groups, 5)
    library7 = synthetic_library(CONFIG.num_groups, 5, num_materials=7)
    rho7 = wlab_phantom(GRID, device=DEVICE, dtype=dtype)
    s7 = _solver(library7, dtype=dtype)
    f = s7.material_fields(rho7, GRID)
    assert f.pairs_quadrature is not None and torch.equal(f.pairs_quadrature.dense(), f.rho_quadrature)
    a = s7.solve_view(rho7, GRID, 0.3)
    b = _solver(library2, dtype=dtype).solve_view(rho7[[WATER, BONE]], GRID, 0.3)
    _compare_views(a, b, 2e-5 if dtype == torch.float32 else 1e-12)
    with pytest.raises(ValueError, match="materials"):
        s7.material_fields(rho7[[WATER, BONE]], GRID)


@cuda
def test_solver_seven_materials_equals_two_materials_real_library(libraries):
    """The real WLAB problem with the seven-material library against the two-material (water, bone) library: identical
    to float32 rounding (the library rows agree to 1e-16; research comparison 2e-5)."""
    _cap()
    wb, library = libraries
    config = replace(CONFIG, num_groups=6, legendre_order=7, num_polar=8, num_azimuthal=16,
                     source_iteration_tolerance=1e-3)
    rho = wlab_phantom(GRID, device=DEVICE)
    a = _solver(library, config).solve_view(rho, GRID, 0.0)
    b = _solver(wb, config).solve_view(rho[[WATER, BONE]], GRID, 0.0)
    _compare_views(a, b, 2e-5)


@cuda
def test_density_model_reproduces_true_density_spr(libraries):
    """Fixed-SPR check (cheap version of the research oracle study): the solver's SPR from the densities of the density
    model at the TRUE attenuation equals the SPR of the true densities (water and bone map to themselves; the lung and
    air inserts, water of reduced density in the truth, become the lung / adipose and air / lung mixtures of the same
    attenuation: research 3e-4 on all 180 views)."""
    _cap()
    _, library = libraries
    config = replace(CONFIG, num_groups=6, legendre_order=7, num_polar=8, num_azimuthal=16,
                     source_iteration_tolerance=1e-3)
    rho = wlab_phantom(GRID, device=DEVICE)
    rho_m = attenuation_to_densities(library, densities_to_attenuation(library, rho))
    assert int((rho_m != 0).any(dim=(1, 2, 3)).sum()) == 5  # air, lung, adipose, water, bone
    s = _solver(library, config)
    for ang in (0.0, 1.2):
        a, b = s.solve_view(rho_m, GRID, ang), s.solve_view(rho, GRID, ang)
        spr_a = a.scatter_signal / a.primary["signal"]
        spr_b = b.scatter_signal / b.primary["signal"]
        assert float((spr_a / spr_b - 1).abs().max()) < 2e-3


# ---- density model --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("library_kind", ["synthetic", "real"])
def test_interpolating_round_trip_and_bracketing(library_kind, request, device):
    library = (synthetic_library(num_materials=7, fine_spread=0.3) if library_kind == "synthetic"
               else request.getfixturevalue("libraries")[1])
    M = library.num_materials
    energies = (effective_energy(library), 45.0, 95.0)
    for E in energies:
        mu = nominal_attenuation(library, E)  # [M] increasing
        assert bool((torch.diff(mu) > 0).all())
        g = torch.Generator().manual_seed(1)
        hi = float(mu[-1]) * 1.4
        X = torch.cat([torch.rand(3000, generator=g, dtype=torch.float64) * hi - 0.02,
                       mu, 0.5 * (mu[1:] + mu[:-1]), torch.tensor([0.0, -0.1, 1.3 * hi],
                                                                  dtype=torch.float64)]).to(device)
        rho = attenuation_to_densities(library, X, E)
        assert rho.shape == (M, X.numel()) and bool((rho >= 0).all())
        assert int((rho != 0).sum(0).max()) <= 2
        # exact in attenuation at the requested energy
        assert torch.allclose(densities_to_attenuation(library, rho, E), X.clamp_min(0), rtol=1e-12, atol=1e-15)
        nom = library.nominal_density.to(device)
        n0 = 3000
        # pure material m at its nominal attenuation
        assert torch.allclose(rho[:, n0:n0 + M], torch.diag(nom), rtol=1e-12, atol=1e-15)
        # half way between consecutive materials: half of each at nominal density
        mid = rho[:, n0 + M:n0 + 2 * M - 1]
        for i in range(M - 1):
            expect = torch.zeros(M, dtype=torch.float64, device=device)
            expect[i], expect[i + 1] = 0.5 * nom[i], 0.5 * nom[i + 1]
            assert torch.allclose(mid[:, i], expect, rtol=1e-10), i
        # X = 0 and X < 0 -> zero; above the last material: the last material only, density X / m_last
        m = mass_attenuation_at(library, E).to(device)
        assert float(rho[:, -3:-1].abs().max()) == 0.0
        assert float(rho[M - 1, -1]) == pytest.approx(1.3 * hi / float(m[M - 1]), rel=1e-12)
        assert float(rho[:M - 1, -1].abs().max()) == 0.0
        # below the first material: the first material only, density X / m_first
        low = (X > 0) & (X < mu[0].to(device))
        if bool(low.any()):
            assert torch.allclose(rho[0, low], X[low] / m[0], rtol=1e-12)
    # float32 input: same device / dtype, round trip to float32 rounding
    X32 = (torch.rand(50, 7, generator=torch.Generator().manual_seed(4)) * 0.6).to(device)
    r32 = attenuation_to_densities(library, X32)
    assert r32.dtype == torch.float32 and r32.device.type == device.type
    assert torch.allclose(densities_to_attenuation(library, r32), X32, rtol=1e-5, atol=1e-7)


def test_wlab_bone_voxel_is_pure_bone(libraries):
    """The WLAB cortical-bone voxel (1.85 g/cm^3) maps to pure bone at 1.85 g/cm^3 (the two-material split of citrus
    < 0.1.0: unit water plus 1.156 g/cm^3 bone); unit water to pure water; DensityModel does the same after the
    clamp."""
    _, library = libraries
    E = effective_energy(library)
    m = mass_attenuation_at(library, E)
    X = torch.tensor([RHO_BONE * float(m[BONE]), float(m[WATER]), -0.05], dtype=torch.float64)
    rho = attenuation_to_densities(library, X, E)
    expect = torch.zeros(7, 3, dtype=torch.float64)
    expect[BONE, 0], expect[WATER, 1] = RHO_BONE, 1.0
    assert torch.allclose(rho, expect, atol=1e-12)
    dm = DensityModel.from_library(library)
    assert dm.energy_kev == pytest.approx(63.03, abs=0.01)
    out = dm(X.float())
    assert out.shape == (7, 3) and dm.last_negative_fraction == pytest.approx(1 / 3)
    assert torch.allclose(out.double(), expect, atol=1e-6)
