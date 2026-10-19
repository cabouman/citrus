"""End to end on a curved panel with helical views: synthetic data of the WLAB phantom from a two-turn helix whose
panel sees less than half of the phantom's height at a time, reconstructed by ``citrus_recon``, which beats the
uncorrected MBIR loop, on every view and with the scatter step on every other view."""

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
def helical():
    """Two turns of 16 views, 12 cm of table feed per turn, a curved panel of 38 x 16 pixels of 1.6 cm (17 cm high
    at the isocenter), the 2 cm WLAB grid, noise-free data and the regularization of the true primary."""
    pytest.importorskip("mbirtorch")
    from citrus.recon.mbir_bridge import make_cone_model
    from citrus.scatter_solver.scatter_solver import build_solver_library
    from citrus.utilities import (WlabPhantom, helical_scan, synthesize_cone_beam_data, view_angles, wlab_geometry,
                                  wlab_grid, wlab_solver)

    config = SolverConfig(transport_cell_cm=2.0)
    try:
        library = build_solver_library(config, SPECTRUM_125KVP)
    except Exception as exc:  # no cached library and no photon data
        pytest.skip(f"real library unavailable: {exc}")
    panel = replace(wlab_geometry(), num_det_rows=16, num_det_channels=38, delta_det_row=1.6, delta_det_channel=1.6,
                    use_curved_detector=True)
    angles = view_angles(32, end_view_angle=4 * math.pi)
    geometry = helical_scan(panel, angles, pitch_cm=12.0)
    grid = wlab_grid(2.0)
    solver = wlab_solver(2.0, config=config, library=library, geometry=geometry, device=DEVICE)
    phantom = WlabPhantom(grid, solver.materials, device=DEVICE, dtype=torch.float32)
    data = synthesize_cone_beam_data(phantom, grid, geometry, angles, solver, 1.0e5, noise=False)
    model = make_cone_model(grid, geometry, angles, device=DEVICE)
    Y = torch.as_tensor(data["Y"], device=DEVICE)
    P = torch.as_tensor(data["P"], device=DEVICE)
    regularization = estimate_regularization(model, -torch.log(P / 1.0e5), Y / 1.0e5)
    return data, grid, geometry, solver, regularization


def _rmse(X, data) -> float:
    truth = torch.as_tensor(data["X_true"], device=X.device)
    return float((X - truth).norm() / truth.norm())


def test_helical_data(helical):
    data, grid, geometry, _, _ = helical
    assert geometry.is_helical and geometry.use_curved_detector
    z = geometry.z_shifts(data["angles"])
    assert z[-1] - z[0] == pytest.approx(12.0 * 2 * (31 / 32))  # two turns less one view of travel
    assert bool(np.isfinite(data["Y"]).all()) and float(data["SPR"].min()) > 0
    # the views see different slabs of the phantom: the panel moves with the helix
    middle_row = data["line_integrals"][:, 8].max(axis=1)
    assert middle_row.min() > 0.5  # the 40 cm prism is in every view
    assert data["meta"]["geometry"]["use_curved_detector"] is True


def test_helical_citrus_recon_beats_no_correction(helical):
    data, grid, geometry, solver, regularization = helical
    config = CitrusConfig(mbir_iterations_per_outer_iteration=2, num_outer_iterations=3)
    initial = []
    X, info = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, solver, config,
                           regularization=regularization,
                           callback=lambda k, X, record: initial.append(_rmse(X, data)) if k == 0 else None)
    assert X.shape == grid.shape and bool(torch.isfinite(X).all())
    assert initial[0] < 0.6  # the helical FDK is a usable initial volume
    X_none, _ = mbir_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, config, regularization=regularization,
                           device=DEVICE)
    X_oracle, _ = mbir_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, config, spr=data["SPR"],
                             regularization=regularization, device=DEVICE)
    errors = {"citrus": _rmse(X, data), "none": _rmse(X_none, data), "oracle": _rmse(X_oracle, data)}
    assert errors["oracle"] < errors["none"] and errors["citrus"] < errors["none"], errors
    # the correction recovers most of what the true scatter would
    assert errors["none"] - errors["citrus"] > 0.5 * (errors["none"] - errors["oracle"]), errors
    # every other view: the skipped views' scatter is interpolated along the helix
    sparse = replace(config, scatter_view_stride=2)
    X_sparse, sparse_info = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, solver, sparse,
                                         regularization=regularization)
    assert list(sparse_info["computed_views"]) == list(range(0, 32, 2))
    assert _rmse(X_sparse, data) < errors["none"], (_rmse(X_sparse, data), errors)
