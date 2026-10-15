"""Utilities (citrus/utilities.py): the WLAB phantom and grid, the attenuation <-> density model, and synthetic
cone-beam data with scatter (generic and WLAB)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from citrus.scatter_solver.config import SolverConfig
from citrus.scatter_solver.cross_sections import synthetic_library
from citrus.scatter_solver.geometry import ConeGeometry, Grid
from citrus.utilities import (RHO_AIR, RHO_BONE, RHO_LUNG, attenuation_to_densities, densities_to_attenuation,
                              effective_energy, load_dataset, mass_attenuation_at, synthesize_cone_beam_data,
                              synthesize_wlab_dataset, wlab_geometry, wlab_grid, wlab_phantom, wlab_setup)


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
    assert rho.shape == (2, *grid.shape) and rho.dtype == torch.float64
    v = grid.voxel_volume
    w, b = rho[0], rho[1]
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
    assert (rho[0] > 0).sum().item() == (30 * 20 - 4) * 40  # all prism voxels except bone are water-based
    assert (rho[1] > 0).sum().item() == 4 * 40
    x, y, z = grid.axis_coordinates()
    assert rho[:, :, :, (z.abs() > 20)].abs().sum() == 0
    assert rho[0, 12, 19, 22].item() == pytest.approx(RHO_LUNG)  # (x, y, z) = (2.5, 0.5, 1.5) is in the lung
    with pytest.raises(ValueError):
        wlab_phantom(Grid(shape=(20, 30, 38), delta_voxel=1.0))  # too short in z
    with pytest.raises(ValueError):
        wlab_phantom(wlab_grid(1.0), lung_center=(0.0, 0.0), air_center=(1.0, 0.0))  # overlap
    with pytest.raises(ValueError):
        wlab_grid(0.7)


def test_wlab_setup():
    grid, geom, rho = wlab_setup()
    assert grid.shape == (20, 30, 40) and grid.delta_voxel == 1.0
    assert geom.source_iso_dist == 100.0 and geom.image_shape == (75, 100)
    assert rho.shape == (2, 20, 30, 40)


def test_wlab_primary_beer_law_central_channels(device):
    from citrus.scatter_solver.ray_tracing import SiddonProjectors
    from citrus.scatter_solver.cross_sections import mu_t_volume, synthetic_library

    lib = synthetic_library(6)
    grid, geom, rho = wlab_setup(1.0, device=device, dtype=torch.float64)
    mu = mu_t_volume(lib, rho)
    tau = SiddonProjectors().primary_project(mu, grid, geom, 0.0)  # [G, 75, 100]
    pix = geom.pixel_centers(0.0, device=device, dtype=torch.float64)[:, 48:52]  # u in {-0.6..0.6}: water only
    src = torch.tensor(geom.source_position(0.0), device=device, dtype=torch.float64)
    d = pix - src
    path = 20.0 * torch.linalg.norm(d, dim=-1) / d[..., 1].abs()  # full 20 cm of water along y
    ref = lib.mu_t[0].to(device)[:, None, None] * path[None]
    torch.testing.assert_close(tau[:, :, 48:52], ref, rtol=0, atol=1e-10)


# ---- attenuation <-> density model and synthetic data ------------------------------------------------------

@pytest.fixture(scope="module")
def real_lib():
    from citrus.scatter_solver.scatter_solver import build_solver_library

    try:
        return build_solver_library(SolverConfig())
    except Exception as exc:  # no cached library and no XrayPhysics
        pytest.skip(f"real library unavailable: {exc}")


# ---- materials ---------------------------------------------------------------------------------------

def test_effective_energy_and_interpolation(real_lib):
    E = effective_energy(real_lib)
    assert 62.5 < E < 63.5  # spectrum mean of the default 125 kVp spectrum
    m = mass_attenuation_at(real_lib, E)
    assert 0.19 < float(m[0]) < 0.21 and 0.28 < float(m[1]) < 0.30  # water, cortical bone at ~63 keV (cm^2/g)
    j = 57  # at a table node the interpolation returns the table value
    e = float(real_lib.fine_energy[j])
    assert torch.allclose(mass_attenuation_at(real_lib, e), real_lib.fine_mu_t[:, j].double(), rtol=1e-12)
    with pytest.raises(ValueError):
        mass_attenuation_at(real_lib, 1.0)


@pytest.mark.parametrize("lib_kind", ["synthetic", "real"])
def test_density_round_trip(lib_kind, request, device):
    lib = synthetic_library() if lib_kind == "synthetic" else request.getfixturevalue("real_lib")
    E = 63.0 if lib_kind == "real" else float(lib.fine_energy.mean())
    m = mass_attenuation_at(lib, E)
    mw, mb = float(m[0]), float(m[1])
    grid = wlab_grid(1.0)
    rho = wlab_phantom(grid, device=device, dtype=torch.float64)
    X = densities_to_attenuation(lib, rho, E)
    assert X.shape == grid.shape and X.device.type == device.type and X.dtype == torch.float64
    assert torch.allclose(X, rho[0] * mw + rho[1] * mb)
    rho2 = attenuation_to_densities(lib, X, E)
    assert rho2.shape == rho.shape
    # the attenuation round trip is exact
    assert torch.allclose(densities_to_attenuation(lib, rho2, E), X, rtol=1e-12, atol=1e-14)
    bone = rho[1] > 0
    # water / lung / air / vacuum come back exactly
    assert torch.allclose(rho2[:, ~bone], rho[:, ~bone], rtol=1e-12, atol=1e-14)
    # cortical bone (1.85 g/cm^3 bone) -> unit water + (1.85 - m_w/m_b) bone
    assert torch.allclose(rho2[0][bone], torch.ones_like(rho2[0][bone]))
    assert torch.allclose(rho2[1][bone], torch.full_like(rho2[1][bone], RHO_BONE - mw / mb), rtol=1e-12)
    # negatives clamp to zero, the boundary X = mu_w is pure unit water
    Xn = torch.tensor([-0.1, 0.0, mw, 0.5 * mw, mw + mb], dtype=torch.float64, device=device)
    r = attenuation_to_densities(lib, Xn, E)
    expect = torch.tensor([[0.0, 0.0, 1.0, 0.5, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float64, device=device)
    assert torch.allclose(r, expect, rtol=1e-12, atol=1e-14)


# ---- synthetic dataset ---------------------------------------------------------------------------------

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="dataset synthesis runs on CUDA")


def _shadow_median_spr(d: dict) -> float:
    return float(np.median(d["SPR"][d["line_integrals"] > 1.0]))


@cuda
def test_smoke_dataset_2cm(real_lib, tmp_path):
    pytest.importorskip("mbirtorch")
    V = 4
    out = synthesize_wlab_dataset(V, 2.0, I0=1.0e5, noise=False, seed=0, out_path=tmp_path / "smoke", lib=real_lib,
                                  geom=wlab_geometry())
    R, C, S = wlab_grid(2.0).shape
    for k in ("P", "S", "Y", "Y_mean", "SPR", "line_integrals", "solver_primary", "solver_scatter"):
        assert out[k].shape == (V, 75, 100) and out[k].dtype == np.float32, k
    assert out["X_true"].shape == (R, C, S) and out["rho_true"].shape == (2, R, C, S)
    assert out["angles"].shape == (V,) and np.allclose(out["angles"], 2 * np.pi * np.arange(V) / V)
    assert out["phi_air"].shape == (75, 100)
    # noise-free: Y = P + S, P = I0 exp(-A X), S = SPR P
    assert np.array_equal(out["Y"], out["Y_mean"])
    assert np.allclose(out["Y"], out["P"] + out["S"], rtol=1e-6)
    assert np.allclose(out["P"], 1.0e5 * np.exp(-out["line_integrals"]), rtol=1e-5)
    assert np.allclose(out["S"], out["SPR"] * out["P"], rtol=1e-5)
    assert np.allclose(out["SPR"], out["solver_scatter"] / out["solver_primary"], rtol=1e-5)
    # plausible WLAB scatter: SPR behind the phantom of order 1 (no grid)
    assert 0.3 < _shadow_median_spr(out) < 1.5
    assert (out["SPR"] > 0).all() and out["meta"]["si_iterations"]["all_converged"]
    # files
    d = load_dataset(tmp_path / "smoke")
    assert np.array_equal(d["Y"], out["Y"]) and d["meta"]["num_views"] == V
    assert (tmp_path / "smoke.json").exists()
    # Poisson noise: reproducible per seed, unbiased
    n1 = synthesize_wlab_dataset(2, 2.0, I0=1.0e5, noise=True, seed=3, lib=real_lib, geom=wlab_geometry())
    n2 = synthesize_wlab_dataset(2, 2.0, I0=1.0e5, noise=True, seed=3, lib=real_lib, geom=wlab_geometry())
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

    dev = torch.device("cuda")
    grid = Grid((8, 8, 8), 2.0)
    geom = ConeGeometry(100.0, 150.0, 24, 24, 1.6, 1.6)
    rho = torch.zeros(2, *grid.shape, device=dev)
    rho[0, 1:7, 1:7, 1:7] = 1.0
    rho[0, 3:5, 3:5] = 0.0
    rho[1, 3:5, 3:5] = 1.5
    cfg = SolverConfig(num_groups=3, legendre_order=1, num_polar=2, num_azimuthal=4, cone_num_polar=4,
                       cone_num_azimuthal=8, supervoxel_factor=2, si_tol=1e-3)
    solver = ScatterSolver(synthetic_library(3, 2), cfg, geom, detector=ideal_counting_response(),
                           grid_response=no_grid(), device=dev, dtype=torch.float32)
    angles = [0.0, 1.0, 2.5]
    out = synthesize_cone_beam_data(rho, grid, geom, angles, solver, I0=1.0e4, noise=False)
    assert out["Y"].shape == (3, 24, 24) and out["X_true"].shape == grid.shape and out["rho_true"].shape == rho.shape
    assert np.allclose(out["Y"], out["P"] + out["S"], rtol=1e-6)
    assert np.allclose(out["P"], 1.0e4 * np.exp(-out["line_integrals"]), rtol=1e-5)
    assert np.allclose(out["S"], out["SPR"] * out["P"], rtol=1e-5)
    assert float(out["SPR"].max()) > 0 and out["meta"]["si_iterations"]["all_converged"]
    other = ConeGeometry(100.0, 150.0, 20, 24, 1.6, 1.6)
    with pytest.raises(ValueError, match="geometry"):
        synthesize_cone_beam_data(rho, grid, other, angles, solver)
