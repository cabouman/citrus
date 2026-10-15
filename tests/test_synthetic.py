"""Attenuation <-> density model (citrus/utilities.py) and the synthetic WLAB dataset (citrus/recon/synthetic.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from citrus.scatter_solver.config import SolverConfig
from citrus.utilities import RHO_BONE, wlab_grid, wlab_phantom
from citrus.utilities import (attenuation_to_densities, densities_to_attenuation, effective_energy,
                                         mass_attenuation_at)
from citrus.scatter_solver.cross_sections import synthetic_library

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
    from citrus.recon.synthetic import load_dataset, synthesize_wlab_dataset

    V = 4
    from citrus.utilities import wlab_geometry

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
