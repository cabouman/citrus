"""End to end on a parallel beam: synthetic data of the WLAB phantom from 24 views over 180 degrees with the panel 30 cm
from the axis, reconstructed by ``citrus_recon``, which beats the uncorrected MBIR loop, on every view and with the
scatter step on every other view."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from conftest import SPECTRUM_125KVP
from citrus.citrus import CitrusConfig, citrus_recon, estimate_regularization, mbir_recon
from citrus.scatter_solver.config import SolverConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the CITRUS reconstruction tests run on CUDA")


@pytest.fixture(scope="module")
def parallel():
    """24 views over 180 degrees, the 2 cm WLAB grid with its parallel-beam geometry (20 rows, the slices, and 19
    channels of 2 cm, the panel 30 cm from the axis), noise-free data and the regularization of the true primary."""
    pytest.importorskip("mbirtorch")
    from citrus.recon.mbir_bridge import make_model
    from citrus.scatter_solver.scatter_solver import build_solver_library
    from citrus.utilities import (WlabPhantom, parallel_wlab_geometry, synthesize_cone_beam_data, view_angles,
                                  wlab_grid, wlab_solver)

    config = SolverConfig(transport_cell_cm=2.0)
    try:
        library = build_solver_library(config, SPECTRUM_125KVP)
    except Exception as exc:  # no cached library and no photon data
        pytest.skip(f"real library unavailable: {exc}")
    geometry = parallel_wlab_geometry(2.0, axis_detector_dist=30.0)
    angles = view_angles(24, geometry)
    grid = wlab_grid(2.0)
    solver = wlab_solver(2.0, config=config, library=library, geometry=geometry, device=DEVICE)
    phantom = WlabPhantom(grid, solver.materials, device=DEVICE, dtype=torch.float32)
    data = synthesize_cone_beam_data(phantom, grid, geometry, angles, solver, 1.0e5, noise=False)
    model = make_model(grid, geometry, angles, device=DEVICE)
    Y = torch.as_tensor(data["Y"], device=DEVICE)
    P = torch.as_tensor(data["P"], device=DEVICE)
    regularization = estimate_regularization(model, -torch.log(P / 1.0e5), Y / 1.0e5)
    return data, grid, geometry, solver, regularization


def _rmse(X, data) -> float:
    truth = torch.as_tensor(data["X_true"], device=X.device)
    return float((X - truth).norm() / truth.norm())


def test_parallel_data(parallel):
    data, grid, geometry, _, _ = parallel
    assert geometry.is_parallel and geometry.num_det_rows == grid.num_slices
    assert data["angles"][-1] < math.pi and data["meta"]["geometry"]["axis_detector_dist"] == 30.0
    assert bool(np.isfinite(data["Y"]).all()) and float(data["SPR"].min()) > 0
    # a parallel beam has a flat open field and the same object depth on every row of a view: the prism is 40 cm
    # long, so the line integrals do not depend on the row
    line_integrals = data["line_integrals"]
    assert np.allclose(line_integrals, line_integrals[:, :1], rtol=1e-4, atol=1e-4)
    assert float(data["phi_air"].max() / data["phi_air"].min()) == pytest.approx(1.0)
    # the beam sees the 30 cm side of the prism end on at 90 degrees, the 20 cm side at 0
    central = line_integrals[:, 10]
    assert central[12].max() > central[0].max()


def test_parallel_citrus_recon_beats_no_correction(parallel):
    data, grid, geometry, solver, regularization = parallel
    config = CitrusConfig(mbir_iterations_per_outer_iteration=2, num_outer_iterations=3)
    initial = []
    X, info = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, solver, config,
                           regularization=regularization,
                           callback=lambda k, X, record: initial.append(_rmse(X, data)) if k == 0 else None)
    assert X.shape == grid.shape and bool(torch.isfinite(X).all())
    assert initial[0] < 0.5  # the filtered back projection is a usable initial volume
    X_none, _ = mbir_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, config, regularization=regularization,
                           device=DEVICE)
    X_oracle, _ = mbir_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, config, spr=data["SPR"],
                             regularization=regularization, device=DEVICE)
    errors = {"citrus": _rmse(X, data), "none": _rmse(X_none, data), "oracle": _rmse(X_oracle, data)}
    assert errors["oracle"] < errors["none"] and errors["citrus"] < errors["none"], errors
    # the correction recovers most of what the true scatter would
    assert errors["none"] - errors["citrus"] > 0.5 * (errors["none"] - errors["oracle"]), errors
    # every other view: the skipped views' scatter is interpolated along the 180-degree arc
    sparse = replace(config, scatter_view_stride=2)
    X_sparse, sparse_info = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, solver, sparse,
                                         regularization=regularization)
    assert list(sparse_info["computed_views"]) == list(range(0, 24, 2))
    assert _rmse(X_sparse, data) < errors["none"], (_rmse(X_sparse, data), errors)
