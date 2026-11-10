"""The MBIR weight settings (measurement_weights, max_measurement_weight) and what the scatter solver sees
(air_threshold_hu, known_objects) of CitrusConfig."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from conftest import SPECTRUM_125KVP
from citrus import citrus as citrus_module
from citrus.citrus import CitrusConfig, DensityModel, citrus_recon, estimate_regularization, mbir_recon
from citrus.scatter_solver.config import SolverConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the CITRUS reconstruction tests run on CUDA")


@pytest.mark.parametrize("settings", [
    dict(max_measurement_weight=0.0), dict(max_measurement_weight=float("nan")), dict(max_measurement_weight=True),
    dict(air_threshold_hu="-900"), dict(air_threshold_hu=float("inf")), dict(measurement_weights=np.full(4, -1.0)),
    dict(measurement_weights=np.zeros((1, 1, 1, 4))),
])
def test_loop_options_are_validated(settings):
    with pytest.raises(ValueError):
        CitrusConfig(**settings).validate()


def test_loop_options_default_off():
    config = CitrusConfig()
    config.validate()
    assert config.measurement_weights is None and config.max_measurement_weight is None
    assert config.air_threshold_hu is None
    CitrusConfig(measurement_weights=np.ones(4), max_measurement_weight=1e-6, air_threshold_hu=-900.0).validate()


def test_field_of_view_radius():
    """The widest ray's distance from the axis: the outer edge of the farthest channel, offset included."""
    from citrus.scatter_solver.geometry import ConeGeometry
    from citrus.utilities import field_of_view_radius

    flat = ConeGeometry(100.0, 150.0, 4, 30, 1.0, 1.0)
    assert field_of_view_radius(flat) == pytest.approx(100.0 * math.sin(math.atan(15.0 / 150.0)))
    half_fan = replace(flat, det_channel_offset=-10.0)  # the long side reaches 25 cm on the panel
    assert field_of_view_radius(half_fan) == pytest.approx(100.0 * math.sin(math.atan(25.0 / 150.0)))
    curved = replace(flat, use_curved_detector=True)
    assert field_of_view_radius(curved) == pytest.approx(100.0 * math.sin(15.0 / 150.0))


@pytest.fixture(scope="module")
def tiny():
    """The small noise-free WLAB problem of test_citrus_recon (16 views, 2 cm grid, 46 x 38 panel of 1.6 cm pixels)."""
    pytest.importorskip("mbirtorch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    from citrus.recon.mbir_bridge import make_cone_model
    from citrus.scatter_solver.scatter_solver import build_solver_library
    from citrus.utilities import synthesize_wlab_dataset, wlab_geometry, wlab_grid, wlab_solver

    try:
        library = build_solver_library(SolverConfig(), SPECTRUM_125KVP)
    except Exception as exc:  # no cached library and no photon data
        pytest.skip(f"real library unavailable: {exc}")
    geometry = replace(wlab_geometry(), num_det_rows=46, num_det_channels=38, delta_det_row=1.6, delta_det_channel=1.6)
    config = SolverConfig(transport_cell_cm=2.0)
    data = synthesize_wlab_dataset(16, 2.0, noise=False, seed=0, library=library, geometry=geometry, config=config)
    grid = wlab_grid(2.0)
    solver = wlab_solver(2.0, config=config, library=library, geometry=geometry, device=DEVICE)
    model = make_cone_model(grid, geometry, data["angles"], device=DEVICE)
    Y = torch.as_tensor(data["Y"], device=DEVICE)
    P = torch.as_tensor(data["P"], device=DEVICE)
    regularization = estimate_regularization(model, -torch.log(P / 1.0e5), Y / 1.0e5)
    return data, grid, geometry, solver, regularization


@cuda
def test_measurement_weights_leave_measurements_out(tiny):
    """Weights of 1 change nothing; corrupted channels with weight 0 leave the reconstruction close to that of clean
    data, while unweighted they spoil it; the cap gives every measurement the same weight."""
    data, grid, geometry, _, regularization = tiny
    P = np.asarray(data["P"], dtype=np.float32)
    options = dict(regularization=regularization, spr=None)
    config = CitrusConfig(num_outer_iterations=1, mbir_iterations_per_outer_iteration=6)
    clean, _ = mbir_recon(P, 1.0e5, data["angles"], grid, geometry, config, **options)
    ones, _ = mbir_recon(P, 1.0e5, data["angles"], grid, geometry,
                         replace(config, measurement_weights=np.ones(P.shape[2])), **options)
    assert float((ones - clean).norm() / clean.norm()) < 1e-4  # the GPU projectors are not bit-for-bit repeatable
    corrupted = P.copy()
    corrupted[:, :, 14:20] *= 0.6
    keep = np.ones(P.shape[2], dtype=np.float32)
    keep[14:20] = 0.0
    start = dict(options, init_recon=clean)  # the same start, so that the FDK of the corrupted data plays no part
    spoiled, _ = mbir_recon(corrupted, 1.0e5, data["angles"], grid, geometry, config, **start)
    weighted_clean, _ = mbir_recon(P, 1.0e5, data["angles"], grid, geometry,
                                   replace(config, measurement_weights=keep), **start)
    weighted, _ = mbir_recon(corrupted, 1.0e5, data["angles"], grid, geometry,
                             replace(config, measurement_weights=keep), **start)
    # with weight 0 the corrupted channels play no part: the same volume as from clean data with those weights
    assert float((weighted - weighted_clean).norm() / weighted_clean.norm()) < 1e-4
    assert float((spoiled - clean).norm() / clean.norm()) > 1e-2

    seen = []
    original = citrus_module.mbir_iterations

    def spy(model, sinogram, X, **kwargs):
        seen.append(kwargs["weights"].clone())
        return original(model, sinogram, X, **kwargs)

    citrus_module.mbir_iterations = spy
    try:
        mbir_recon(P, 1.0e5, data["angles"], grid, geometry,
                   replace(config, max_measurement_weight=1e-6, measurement_weights=keep), **options)
    finally:
        citrus_module.mbir_iterations = original
    weights = seen[0]
    assert bool((weights[:, :, 14:20] == 0).all())
    assert torch.allclose(weights[:, :, :14], torch.full_like(weights[:, :, :14], 1e-6))


@cuda
def test_scene_options_change_only_what_the_scatter_solver_sees(tiny):
    """With the air threshold the solver's volume has no voxel between 0 and the threshold; the defaults pass the
    volume unchanged."""
    data, grid, geometry, solver, regularization = tiny
    seen = []
    original = solver.material_fields_from_attenuation

    def spy(X, *args, **kwargs):
        seen.append(X.clone())
        return original(X, *args, **kwargs)

    solver.material_fields_from_attenuation = spy
    loop = CitrusConfig(num_outer_iterations=2, mbir_iterations_per_outer_iteration=1)
    try:
        X_default, info_default = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, solver, loop,
                                               regularization=regularization)
        assert torch.equal(seen[0], info_default["init_recon"])
        seen.clear()
        options = replace(loop, air_threshold_hu=-500.0)
        X, info = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, solver, options,
                               regularization=regularization)
    finally:
        solver.material_fields_from_attenuation = original
    water = float(DensityModel.from_library(solver.library).energy_kev)  # E_eff of the loop's density model
    from citrus.citrus import _water_attenuation
    mu_water = _water_attenuation(DensityModel.from_library(solver.library, water))
    assert len(seen) == 2
    for scene in seen:
        assert bool((scene > 0).any())
        assert not bool(((scene > 0) & (scene < 0.5 * mu_water)).any())
    assert bool(torch.isfinite(X).all()) and X.shape == X_default.shape


def test_known_objects_are_validated():
    with pytest.raises(ValueError):
        CitrusConfig(known_objects=-np.ones((2, 2, 2))).validate()
    with pytest.raises(ValueError):
        CitrusConfig(known_objects=np.ones((2, 2))).validate()
    CitrusConfig(known_objects=np.zeros((2, 2, 2))).validate()


@cuda
def test_known_objects_come_off_the_reconstruction(tiny):
    """A part of the phantom given as a known object: the scatter solver and the primary see it, its line integrals
    come off the data, so MBIR reconstructs the rest (about 0 there) and the rest as well as without it."""
    data, grid, geometry, solver, regularization = tiny
    X_true = torch.as_tensor(data["X_true"], device=DEVICE, dtype=torch.float32)
    region = torch.zeros_like(X_true, dtype=torch.bool)
    region[:, :, :5] = True  # the bottom slices of the phantom
    known = torch.where(region, X_true, torch.zeros_like(X_true))
    assert float(known.sum()) > 0
    loop = CitrusConfig(num_outer_iterations=3, mbir_iterations_per_outer_iteration=2)
    X_plain, _ = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, solver, loop,
                              regularization=regularization)
    X, _ = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, solver,
                        replace(loop, known_objects=known.cpu().numpy()), regularization=regularization)
    inside = known > 0
    assert float(X[inside].abs().mean()) < 0.2 * float(known[inside].mean())
    rest = ~region
    error_plain = float((X_plain[rest] - X_true[rest]).norm() / X_true[rest].norm())
    error_known = float((X[rest] - X_true[rest]).norm() / X_true[rest].norm())
    assert error_known < 1.2 * error_plain + 0.01
