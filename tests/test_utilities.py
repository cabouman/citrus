"""Utilities (citrus/utilities.py): the WLAB phantom and grid, the attenuation <-> density model (the interpolating
seven-material model), the view angles (short scan by default) and synthetic cone-beam data with scatter (generic and
WLAB)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from conftest import SPECTRUM_125KVP
from citrus.scatter_solver.config import SolverConfig
from citrus.scatter_solver.cross_sections import BONE, MATERIALS, WATER, synthetic_library
from citrus.scatter_solver.geometry import ConeGeometry, Grid
from citrus.utilities import (RHO_AIR, RHO_BONE, RHO_LUNG, attenuation_to_densities, densities_to_attenuation,
                              effective_energy, fan_angle, full_field_wlab_geometry, load_dataset, mass_attenuation_at,
                              nominal_attenuation, synthesize_cone_beam_data, synthesize_wlab_dataset, view_angles,
                              wlab_geometry, wlab_grid, wlab_phantom, wlab_setup)


# ---- WLAB phantom ----------------------------------------------------------------------------------------

def _mask(grid, cx, cy, wx, wy):
    x, y, z = grid.axis_coordinates(dtype=torch.float64)
    mx = (x >= cx - wx / 2) & (x < cx + wx / 2)
    my = (y >= cy - wy / 2) & (y < cy + wy / 2)
    return (my[:, None, None] & mx[None, :, None]).expand(grid.shape)


@pytest.mark.parametrize("delta", [1.0, 0.5])
def test_wlab_phantom_volumes_and_materials(delta):
    grid = wlab_grid(delta)
    assert grid.shape == (round(20 / delta), round(30 / delta), round(40 / delta))
    assert grid.extent == pytest.approx((30.0, 20.0, 40.0))
    rho = wlab_phantom(grid, dtype=torch.float64)
    assert rho.shape == (7, *grid.shape) and rho.dtype == torch.float64
    assert (WATER, BONE) == (3, 6) and MATERIALS[WATER] == "water" and MATERIALS[BONE] == "bone_cortical_icrp"
    v = grid.voxel_volume
    w, b = rho[WATER], rho[BONE]
    others = [m for m in range(7) if m not in (WATER, BONE)]
    assert float(rho[others].abs().max()) == 0.0  # only the water and bone rows are used
    lung, air, bone = _mask(grid, -5, 0, 6, 6), _mask(grid, 3, 0, 2, 2), _mask(grid, 7, 0, 2, 2)
    # insert volumes (cm^3) and densities
    assert (w == RHO_LUNG).sum().item() * v == pytest.approx(6 * 6 * 40)
    assert (w == RHO_AIR).sum().item() * v == pytest.approx(2 * 2 * 40)
    assert (b == RHO_BONE).sum().item() * v == pytest.approx(2 * 2 * 40)
    assert (w[lung] == RHO_LUNG).all() and (w[air] == RHO_AIR).all() and (b[bone] == RHO_BONE).all()
    # bone channel only in the bone insert; no water there
    assert (b[~bone] == 0).all() and (w[bone] == 0).all()
    # water everywhere else (the grid tiles the prism exactly)
    rest = ~(lung | air | bone)
    assert (w[rest] == 1.0).all()
    assert rest.sum().item() * v == pytest.approx((20 * 30 - 36 - 4 - 4) * 40)


def test_wlab_phantom_on_larger_grid_and_options():
    grid = Grid(shape=(24, 34, 44), delta_voxel=1.0, recon_slice_offset=1.0)  # 2 cm margin, slice offset 1 cm
    rho = wlab_phantom(grid, lung_center=(5.0, 2.0), air_center=(-4.0, -3.0), bone_center=(-10.0, 4.0))
    assert rho.dtype == torch.float32
    assert (rho[WATER] > 0).sum().item() == (30 * 20 - 4) * 40  # all prism voxels except bone are water-based
    assert (rho[BONE] > 0).sum().item() == 4 * 40
    x, y, z = grid.axis_coordinates()
    assert rho[:, :, :, (z.abs() > 20)].abs().sum() == 0
    assert rho[WATER, 12, 19, 22].item() == pytest.approx(RHO_LUNG)  # (x, y, z) = (2.5, 0.5, 1.5) is in the lung
    with pytest.raises(ValueError):
        wlab_phantom(Grid(shape=(20, 30, 38), delta_voxel=1.0))  # too short in z
    with pytest.raises(ValueError):
        wlab_phantom(wlab_grid(1.0), lung_center=(0.0, 0.0), air_center=(1.0, 0.0))  # overlap
    with pytest.raises(ValueError):
        wlab_grid(0.7)


def test_wlab_setup():
    grid, geometry, rho = wlab_setup()
    assert grid.shape == (20, 30, 40) and grid.delta_voxel == 1.0
    assert geometry.source_iso_dist == 100.0 and geometry.image_shape == (75, 100)
    assert rho.shape == (7, 20, 30, 40)


def test_wlab_primary_beer_law_central_channels(device):
    from citrus.scatter_solver.ray_tracing import SiddonProjectors
    from citrus.scatter_solver.cross_sections import mu_t_volume, synthetic_library

    library = synthetic_library(6, num_materials=7)
    grid, geometry, rho = wlab_setup(1.0, device=device, dtype=torch.float64)
    mu = mu_t_volume(library, rho)
    tau = SiddonProjectors().primary_project(mu, grid, geometry, 0.0)  # [G, 75, 100]
    pix = geometry.pixel_centers(0.0, device=device, dtype=torch.float64)[:, 48:52]  # u in {-0.6..0.6}: water only
    source = torch.tensor(geometry.source_position(0.0), device=device, dtype=torch.float64)
    d = pix - source
    path = 20.0 * torch.linalg.norm(d, dim=-1) / d[..., 1].abs()  # full 20 cm of water along y
    reference = library.mu_t[WATER].to(device)[:, None, None] * path[None]
    torch.testing.assert_close(tau[:, :, 48:52], reference, rtol=0, atol=1e-10)


# ---- attenuation <-> density model and synthetic data ------------------------------------------------------

@pytest.fixture(scope="module")
def real_library():
    from citrus.scatter_solver.scatter_solver import build_solver_library

    try:
        return build_solver_library(SolverConfig(), SPECTRUM_125KVP)
    except Exception as exc:  # no cached library and no photon data
        pytest.skip(f"real library unavailable: {exc}")


# ---- materials ---------------------------------------------------------------------------------------

def test_effective_energy_and_interpolation(real_library):
    E = effective_energy(real_library)
    assert 62.5 < E < 63.5  # spectrum mean of the sample 125 kVp spectrum
    m = mass_attenuation_at(real_library, E)
    assert m.shape == (7,)
    assert 0.19 < float(m[WATER]) < 0.21 and 0.28 < float(m[BONE]) < 0.30  # water, cortical bone at ~63 keV (cm^2/g)
    j = 57  # at a table node the interpolation returns the table value
    e = float(real_library.fine_energy[j])
    assert torch.allclose(mass_attenuation_at(real_library, e), real_library.fine_mu_t[:, j].double(), rtol=1e-12)
    with pytest.raises(ValueError):
        mass_attenuation_at(real_library, 1.0)


@pytest.mark.parametrize("library_kind", ["synthetic", "real"])
def test_density_round_trip(library_kind, request, device):
    """The interpolating seven-material model on the WLAB phantom: exact in attenuation; water and bone voxels come
    back as themselves (pure water 1.0, pure bone 1.85 g/cm^3: the two-material split of citrus < 0.1.0 returned unit
    water plus 1.156 g/cm^3 bone for the bone insert); the lung (water at 0.26) and air (water at 0.0012) inserts map to
    the lung / adipose and air / lung mixtures of the same attenuation."""
    library = (synthetic_library(num_materials=7) if library_kind == "synthetic"
               else request.getfixturevalue("real_library"))
    E = 63.0 if library_kind == "real" else float(library.fine_energy.mean())
    m = mass_attenuation_at(library, E)
    mw, mb = float(m[WATER]), float(m[BONE])
    grid = wlab_grid(1.0)
    rho = wlab_phantom(grid, device=device, dtype=torch.float64)
    X = densities_to_attenuation(library, rho, E)
    assert X.shape == grid.shape and X.device.type == device.type and X.dtype == torch.float64
    assert torch.allclose(X, rho[WATER] * mw + rho[BONE] * mb)
    rho2 = attenuation_to_densities(library, X, E)
    assert rho2.shape == rho.shape == (7, *grid.shape)
    # the attenuation round trip is exact; at most two materials per voxel
    assert torch.allclose(densities_to_attenuation(library, rho2, E), X, rtol=1e-12, atol=1e-14)
    assert int((rho2 != 0).sum(0).max()) <= 2 and bool((rho2 >= 0).all())
    bone = rho[BONE] > 0
    water = rho[WATER] == 1.0
    # cortical bone stays pure bone at 1.85 g/cm^3, water pure unit water, vacuum zero
    assert torch.allclose(rho2[:, bone], rho[:, bone], rtol=1e-12, atol=1e-12)
    assert torch.allclose(rho2[:, water], rho[:, water], rtol=1e-12, atol=1e-12)
    assert bool((rho2[:, X == 0] == 0).all())
    # the lung and air inserts: two bracketing materials
    lung = (rho[WATER] == RHO_LUNG)
    air = (rho[WATER] == RHO_AIR)
    names = lambda mask: {MATERIALS[k] for k in range(7) if bool((rho2[k][mask] != 0).any())}  # noqa: E731
    if library_kind == "real":
        assert names(lung) == {"lung_icrp", "adipose_icrp"} and names(air) == {"air", "lung_icrp"}
    # negatives clamp to zero; X = mu_w is pure unit water; mu_w + 0.5 (mu_b - mu_soft) mixes soft tissue and bone
    mu = nominal_attenuation(library, E)
    Xn = torch.tensor([-0.1, 0.0, mw, 0.5 * (mu[5] + mu[6]), 2.0 * mb * 1.85], dtype=torch.float64, device=device)
    r = attenuation_to_densities(library, Xn, E)
    expect = torch.zeros(7, 5, dtype=torch.float64, device=device)
    expect[WATER, 2] = 1.0
    expect[5, 3], expect[BONE, 3] = 0.5 * 1.06, 0.5 * 1.85
    expect[BONE, 4] = 2.0 * 1.85
    assert torch.allclose(r, expect, rtol=1e-12, atol=1e-14)


# ---- view angles ------------------------------------------------------------------------------------------

def test_fan_angle_and_view_angles():
    geometry = full_field_wlab_geometry()
    fan = fan_angle(geometry)
    assert fan == pytest.approx(2 * math.atan(0.5 * 150 * 0.4 / 150.0), rel=1e-15)
    assert math.degrees(fan) == pytest.approx(22.62, abs=0.01)
    assert math.degrees(fan_angle(wlab_geometry())) == pytest.approx(15.19, abs=0.01)
    # default: short scan [start, start + pi + fan), endpoint excluded, evenly spaced
    a = view_angles(36, geometry)
    assert a.shape == (36,) and a.dtype == np.float64 and a[0] == 0.0
    step = (math.pi + fan) / 36
    assert np.allclose(np.diff(a), step, rtol=0, atol=1e-14)
    assert a[-1] + step == pytest.approx(math.pi + fan, rel=1e-14)
    b = view_angles(10, geometry, start_view_angle=0.5)
    assert b[0] == 0.5 and b[-1] + (math.pi + fan) / 10 == pytest.approx(0.5 + math.pi + fan, rel=1e-14)
    # explicit start / end; a full rotation
    c = view_angles(8, geometry, start_view_angle=-0.2, end_view_angle=1.4)
    assert np.allclose(c, -0.2 + 1.6 * np.arange(8) / 8, atol=1e-15)
    full = view_angles(180, None, 0.0, 2 * math.pi)
    assert np.allclose(full, 2 * math.pi * np.arange(180) / 180, atol=1e-14)
    with pytest.raises(ValueError):
        view_angles(10, None)  # the short-scan default needs the geometry
    with pytest.raises(ValueError):
        view_angles(10, geometry, 1.0, 1.0)
    with pytest.raises(ValueError):
        view_angles(0, geometry)


# ---- synthetic dataset ---------------------------------------------------------------------------------

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="dataset synthesis runs on CUDA")


def _shadow_median_spr(d: dict) -> float:
    return float(np.median(d["SPR"][d["line_integrals"] > 1.0]))


@cuda
def test_smoke_dataset_2cm(real_library, tmp_path):
    pytest.importorskip("mbirtorch")
    V = 4
    out = synthesize_wlab_dataset(V, 2.0, I0=1.0e5, noise=False, seed=0, out_path=tmp_path / "smoke",
                                  library=real_library,
                                  geometry=wlab_geometry(), end_view_angle=2 * np.pi)
    R, C, S = wlab_grid(2.0).shape
    for k in ("P", "S", "Y", "Y_mean", "SPR", "line_integrals", "solver_primary", "solver_scatter"):
        assert out[k].shape == (V, 75, 100) and out[k].dtype == np.float32, k
    assert out["X_true"].shape == (R, C, S) and "rho_true" not in out
    assert out["angles"].shape == (V,) and np.allclose(out["angles"], 2 * np.pi * np.arange(V) / V)
    assert out["meta"]["materials"] == list(MATERIALS) and out["mass_attenuation"].shape == (7,)
    assert out["meta"]["short_scan"] is False and out["meta"]["end_view_angle"] == pytest.approx(2 * np.pi)
    assert out["phi_air"].shape == (75, 100)
    # noise-free: Y = P + S, P = I0 exp(-A X), S = SPR P
    assert np.array_equal(out["Y"], out["Y_mean"])
    assert np.allclose(out["Y"], out["P"] + out["S"], rtol=1e-6)
    assert np.allclose(out["P"], 1.0e5 * np.exp(-out["line_integrals"]), rtol=1e-5)
    assert np.allclose(out["S"], out["SPR"] * out["P"], rtol=1e-5)
    assert np.allclose(out["SPR"], out["solver_scatter"] / out["solver_primary"], rtol=1e-5)
    assert out["meta"]["primary_projector"] == "mbirtorch" and out["meta"]["data_format"] == 2
    # plausible WLAB scatter: SPR behind the phantom of order 1 (no grid)
    assert 0.3 < _shadow_median_spr(out) < 1.5
    assert (out["SPR"] > 0).all() and out["meta"]["si_iterations"]["all_converged"]
    # files
    d = load_dataset(tmp_path / "smoke")
    assert np.array_equal(d["Y"], out["Y"]) and d["meta"]["num_views"] == V
    assert (tmp_path / "smoke.json").exists()
    # Poisson noise: reproducible per seed, unbiased
    n1 = synthesize_wlab_dataset(2, 2.0, I0=1.0e5, noise=True, seed=3, library=real_library, geometry=wlab_geometry())
    n2 = synthesize_wlab_dataset(2, 2.0, I0=1.0e5, noise=True, seed=3, library=real_library, geometry=wlab_geometry())
    # the default angular range is the short scan [0, pi + fan angle)
    assert np.allclose(n1["angles"], view_angles(2, wlab_geometry())) and n1["meta"]["short_scan"] is True
    # the rates themselves differ between runs at the float32 ulp level (atomic accumulation in MBIRTORCH's and the
    # solver's CUDA kernels), which flips a handful of Poisson draws; the rest is bitwise reproducible
    assert np.mean(n1["Y"] != n2["Y"]) < 1e-3 and not np.array_equal(n1["Y"], n1["Y_mean"])
    assert np.all(n1["Y"] == np.round(n1["Y"]))
    z = (n1["Y"] - n1["Y_mean"]) / np.sqrt(n1["Y_mean"])
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1) < 0.05


@cuda
def test_synthesize_cone_beam_data_generic():
    """Any density volume (a water box with a bone rod, the synthetic library): shapes, Y = P + S, S = SPR P,
    P = I0 exp(-A X), the scatter of the given solver; the solver must be built for the scan geometry."""
    pytest.importorskip("mbirtorch")
    from citrus.scatter_solver import ScatterSolver, ideal_counting_response, no_grid

    device = torch.device("cuda")
    grid = Grid((8, 8, 8), 2.0)
    geometry = ConeGeometry(100.0, 150.0, 24, 24, 1.6, 1.6)
    rho = torch.zeros(7, *grid.shape, device=device)
    rho[WATER, 1:7, 1:7, 1:7] = 1.0
    rho[WATER, 3:5, 3:5] = 0.0
    rho[BONE, 3:5, 3:5] = 1.5
    config = SolverConfig(num_groups=3, legendre_order=1, num_polar=2, num_azimuthal=4, cone_num_polar=4,
                          cone_num_azimuthal=8, transport_cell_cm=4.0, source_iteration_tolerance=1e-3)
    solver = ScatterSolver(synthetic_library(3, 2, num_materials=7), config, geometry,
                           detector=ideal_counting_response(),
                           grid_response=no_grid(), device=device, dtype=torch.float32)
    angles = [0.0, 1.0, 2.5]
    out = synthesize_cone_beam_data(rho, grid, geometry, angles, solver, I0=1.0e4, noise=False)
    assert out["Y"].shape == (3, 24, 24) and out["X_true"].shape == grid.shape and "rho_true" not in out
    assert np.allclose(out["Y"], out["P"] + out["S"], rtol=1e-6)
    assert np.allclose(out["P"], 1.0e4 * np.exp(-out["line_integrals"]), rtol=1e-5)
    assert np.allclose(out["S"], out["SPR"] * out["P"], rtol=1e-5)
    assert float(out["SPR"].max()) > 0 and out["meta"]["si_iterations"]["all_converged"]
    other = ConeGeometry(100.0, 150.0, 20, 24, 1.6, 1.6)
    with pytest.raises(ValueError, match="geometry"):
        synthesize_cone_beam_data(rho, grid, other, angles, solver)
    # the SPR's primary is the loop's MBIRTORCH per-material primary (meta unchanged since citrus 0.3.0); it agrees
    # with the solver's Siddon primary of solve_view (one ray per pixel center) except at the edges of the object
    assert out["meta"]["primary_projector"] == "mbirtorch" and out["meta"]["data_format"] == 2
    siddon = np.stack([solver.solve_view(rho, grid, a).primary["planar_fluence"].sum(0).cpu().numpy()
                       for a in angles])
    d = np.abs(out["solver_primary"] / siddon - 1)
    assert float(np.median(d)) < 0.01 and float(d.max()) > 0


def test_wlab_phantom_model_matches_the_stacked_volume():
    """`WlabPhantom` gives the maps of `wlab_phantom` one at a time, its attenuation equals
    `densities_to_attenuation` of the stacked volume, and `synthesize_cone_beam_data` gives the same data from the
    model and from the volume (the model path never holds the [M, R, C, S] volume)."""
    from citrus.utilities import WlabPhantom, densities_to_attenuation, wlab_phantom, wlab_solver

    grid = wlab_grid(2.0)
    phantom = WlabPhantom(grid, MATERIALS, device="cpu", dtype=torch.float64)
    rho = wlab_phantom(grid, device="cpu", dtype=torch.float64)
    assert torch.equal(phantom.densities(), rho)
    for m in range(7):
        assert torch.equal(phantom.material(None, m), rho[m])
    library = synthetic_library(num_materials=7)
    assert torch.allclose(phantom.attenuation(library), densities_to_attenuation(library, rho), rtol=1e-12, atol=0)
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    geometry = full_field_wlab_geometry()
    config = SolverConfig(num_groups=4, legendre_order=1, num_polar=2, num_azimuthal=4, cone_num_polar=4,
                          cone_num_azimuthal=8, transport_cell_cm=4.0, source_iteration_tolerance=1e-3)
    solver = wlab_solver(2.0, config=config, library=library, geometry=geometry, device=device)
    angles = [0.0, 1.2]
    from_model = synthesize_cone_beam_data(WlabPhantom(grid, MATERIALS, device=device), grid, geometry, angles, solver,
                                           noise=False)
    from_volume = synthesize_cone_beam_data(wlab_phantom(grid, device=device), grid, geometry, angles, solver,
                                            noise=False)
    for key in ("X_true", "Y", "SPR", "solver_primary", "solver_scatter", "phi_air"):
        assert np.allclose(from_model[key], from_volume[key], rtol=1e-4, atol=0), key  # projector atomics


# ---- phantoms of the demos ------------------------------------------------------------------------------

def test_demo_phantoms_paint_their_materials():
    """Each phantom fills the rows of its materials (in the given library order) and nothing else; a region
    replaces the regions painted before it, so every voxel holds at most one material."""
    from citrus.scatter_solver import Material
    from citrus.utilities import breast_phantom, dental_head_phantom, pelvis_phantom

    materials = ("air", "adipose_icrp", "water", "muscle_icru44", "soft_tissue_icru44", "bone_cortical_icrp")
    contrast = Material("iodine_contrast", 1.01, "0.99*water + 0.01*I")
    amalgam = Material("dental_amalgam", 11.0, "0.5*Hg + 0.3*Ag + 0.13*Sn + 0.07*Cu")
    dental = ("water", contrast, "bone_cortical_icrp", amalgam)
    cases = [
        (pelvis_phantom(Grid((34, 44, 4), 1.0), materials), materials,
         {"adipose_icrp": 0.95, "water": 1.0, "muscle_icru44": 1.05, "bone_cortical_icrp": 1.85}),
        (breast_phantom(Grid((110, 110, 2), 0.1), materials), materials,
         {"adipose_icrp": 0.95, "soft_tissue_icru44": 1.10, "bone_cortical_icrp": 1.85}),
        (dental_head_phantom(Grid((80, 80, 4), 0.25), dental, contrast, amalgam),
         ("water", "iodine_contrast", "bone_cortical_icrp", "dental_amalgam"),
         {"water": 1.0, "iodine_contrast": 1.01, "bone_cortical_icrp": 1.85, "dental_amalgam": 11.0}),
    ]
    for rho, names, densities in cases:
        assert int((rho > 0).sum(dim=0).max()) == 1
        for m, name in enumerate(names):
            if name in densities:
                assert math.isclose(float(rho[m].max()), densities[name], rel_tol=1e-6), name
            else:
                assert float(rho[m].abs().max()) == 0.0, name
    pelvis = cases[0][0]
    assert torch.equal(pelvis, pelvis.flip(2))  # symmetric about x = 0 (the femoral heads)


def test_shepp_logan_head_phantom():
    pytest.importorskip("mbirtorch")
    from citrus.utilities import shepp_logan_head_phantom

    rho = shepp_logan_head_phantom(wlab_grid(2.0), MATERIALS)
    assert rho.shape == (len(MATERIALS), *wlab_grid(2.0).shape) and int((rho > 0).sum(dim=0).max()) == 1
    assert float(rho[MATERIALS.index("bone_cortical_icrp")].max()) == pytest.approx(1.85)
    assert float(rho[MATERIALS.index("water")].abs().max()) == 0.0
