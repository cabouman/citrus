"""CITRUS reconstruction loop (citrus/citrus.py): the compact per-view accumulator store, citrus_recon and mbir_recon
on a tiny noise-free problem, the frozen regularization, and CITRUS with the true scatter == the fixed-SPR MBIR."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from citrus import citrus as citrus_module
from citrus.citrus import (CitrusConfig, CitrusState, StateStore, citrus_recon, estimate_regularization,
                           mbir_recon)
from citrus.scatter_solver.config import SolverConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the CITRUS reconstruction tests run on CUDA")


# ---- accumulator storage (no solver needed) -----------------------------------------------------------------

SHAPE = (3, 16, 4, 4, 5, 6)  # [G, (L+1)^2 with L = 3, 4, Rc, Cc, Sc]


def _phi(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # realistic magnitudes (collided moments ~1e-6 per source photon/sr: plain float16 would underflow)
    return (1e-6 * torch.randn(SHAPE, generator=g) * torch.logspace(0, -3, SHAPE[1])[None, :, None, None, None,
                                                                                     None]).to(DEVICE)


@pytest.mark.parametrize("location", ["gpu", "cpu"])
def test_state_store_round_trip(location):
    if location == "gpu" and DEVICE.type != "cuda":
        pytest.skip("no CUDA")
    store = StateStore(3, SHAPE, DEVICE, location=location)
    assert store.get(0) is None and store.location == location
    phis = [_phi(i) for i in range(3)]
    for i, p in enumerate(phis):
        store.put(i, p)
    n32 = int(np.prod(SHAPE)) * 4
    for i, p in enumerate(phis):
        q = store.get(i)
        assert q.shape == SHAPE and q.dtype == torch.float32 and q.device.type == DEVICE.type
        # linear-discontinuous cell mean kept (scaled float16: no underflow), slopes zero
        assert float((q[:, :, 0] - p[:, :, 0]).norm() / p[:, :, 0].norm()) < 2e-4
        assert float(q[:, :, 1:].abs().max()) == 0.0
    assert store.bytes_per_view < n32 / 7.5  # 1/8 plus one float32 scale per (group, moment)
    if location == "cpu" and DEVICE.type == "cuda":
        assert store._data[0].is_pinned() and store._data[0].device.type == "cpu"
    with pytest.raises(ValueError):
        store.put(0, phis[0][:, :9])
    store.close()


def test_state_store_auto_location():
    shape = (6, 64, 4, 20, 30, 40)  # the 0.5 cm WLAB accumulator with 1 cm transport cells
    per = int(np.prod(shape)) * 4
    s36 = StateStore(36, shape, DEVICE)
    assert s36.bytes_per_view < per / 7.9 and abs(s36.bytes_per_view / 2 ** 20 - 17.6) < 0.1
    assert s36.location == ("gpu" if DEVICE.type == "cuda" else "cpu")
    s180 = StateStore(180, shape, DEVICE)  # 180 x 17.6 MiB = 3164 MiB > 2600 MiB: host memory (nothing allocated yet)
    assert s180.location == "cpu" and s180.pinned == (DEVICE.type == "cuda")
    assert StateStore(180, shape, DEVICE, gpu_budget_mib=4000).location == ("gpu" if DEVICE.type == "cuda" else "cpu")


# ---- a tiny problem ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_library():
    from citrus.scatter_solver.scatter_solver import build_solver_library

    try:
        return build_solver_library(SolverConfig())
    except Exception as exc:  # no cached library and no photon data
        pytest.skip(f"real library unavailable: {exc}")


@pytest.fixture(scope="module")
def tiny(real_library):
    """16 views over the default short scan (180 degrees + the fan angle of the panel), 2 cm grid (10 x 15 x 20), a
    full-field panel of 1.6 cm pixels (46 x 38), noise-free (Y = P + S), the real library, the regularization estimated
    on the true primary (frozen for every reconstruction)."""
    pytest.importorskip("mbirtorch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    from citrus.recon.mbir_bridge import make_cone_model
    from citrus.utilities import synthesize_wlab_dataset, wlab_solver
    from citrus.utilities import wlab_geometry, wlab_grid

    geometry = replace(wlab_geometry(), num_det_rows=46, num_det_channels=38, delta_det_row=1.6, delta_det_channel=1.6)
    config = SolverConfig(transport_cell_cm=2.0)  # 2 cm transport cells on the 2 cm grid
    data = synthesize_wlab_dataset(16, 2.0, noise=False, seed=0, library=real_library, geometry=geometry, config=config)
    grid = wlab_grid(2.0)
    solver = wlab_solver(2.0, config=config, library=real_library, geometry=geometry, device=DEVICE)
    model = make_cone_model(grid, geometry, data["angles"], device=DEVICE)
    Y = torch.as_tensor(data["Y"], device=DEVICE)
    P = torch.as_tensor(data["P"], device=DEVICE)
    regularization = estimate_regularization(model, -torch.log(P / 1.0e5), Y / 1.0e5)
    return data, grid, geometry, solver, regularization


def _rmse(X, data) -> float:
    Xt = torch.as_tensor(data["X_true"], device=X.device)
    return float((X - Xt).norm() / Xt.norm())


@cuda
def test_citrus_recon_runs_and_beats_no_correction(tiny):
    data, grid, geometry, solver, regularization = tiny
    config = CitrusConfig(num_outer_iterations=3)
    seen = []
    X, info = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, solver, config,
                           regularization=regularization,
                           callback=lambda k, X, record: seen.append((k, _rmse(X, data))))
    assert X.shape == grid.shape and X.device.type == "cuda" and bool(torch.isfinite(X).all())
    assert [k for k, _ in seen] == [0, 1, 2, 3] and len(info["records"]) == 4
    assert info["regularization"] == regularization
    model = info["model"]
    assert model.get_params("auto_regularize_flag") is False
    assert float(model.get_params("sigma_y")) == regularization["sigma_y"]
    r = info["records"]
    assert r[1]["accumulator_change_mean"] == 1.0 and r[3]["accumulator_change_mean"] < 1.0
    assert "scatter_change_relative" in r[2] and r[1]["scatter_seconds"] > 0 and r[1]["mbir_seconds"] > 0
    assert info["spr"].shape == data["Y"].shape and float(info["spr"].min()) > 0
    assert info["state"]["num_views"] == 16 and info["state"]["location"] == "gpu"
    # the same MBIR loop without correction, and with the true SPR (the floor of the correction)
    seen_n = []
    X_none, no_correction_info = mbir_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, config,
                                            regularization=regularization,
                                            device=DEVICE, callback=lambda k, X, record: seen_n.append(_rmse(X, data)))
    X_orc, _ = mbir_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, config, spr=data["SPR"],
                          regularization=regularization,
                          device=DEVICE)
    assert seen_n[0] == seen[0][1]  # the same FDK initialization
    e = {"citrus": _rmse(X, data), "none": _rmse(X_none, data), "oracle": _rmse(X_orc, data)}
    assert e["oracle"] < e["none"] and e["citrus"] < e["none"], e
    assert no_correction_info["spr"] is None and float(no_correction_info["scatter_counts"].abs().max()) == 0.0


@cuda
def test_citrus_with_true_scatter_is_fixed_spr_mbir(tiny, monkeypatch):
    """With the scatter step replaced by the TRUE scatter, CITRUS runs the identical MBIR sequence as mbir_recon with
    spr = SPR_true (same seeds, same frozen sigmas, same weights); the accumulators go through the compact store
    (linear-discontinuous mean kept, slopes dropped)."""
    data, grid, geometry, solver, regularization = tiny
    angles = data["angles"]
    view_of = {float(a): v for v, a in enumerate(angles)}
    spr_true = torch.as_tensor(data["SPR"], device=DEVICE)
    coarse = solver.transport_grid(grid)
    shape = (solver.num_groups, (solver.L + 1) ** 2, 4) + tuple(coarse.shape)

    def fake_step(solver_, state, fields, angle, *, primary=True):
        v = view_of[float(angle)]
        assert not primary and fields.rho_fine is None and fields.attenuation is not None  # the loop's X path
        if state.collided_moments is not None:  # the stored accumulator comes back (cell mean only)
            assert torch.allclose(state.collided_moments[:, :, 0], torch.full_like(state.collided_moments[:, :, 0],
                                                                                   float(state.num_steps)), rtol=1e-3)
            assert float(state.collided_moments[:, :, 1:].abs().max()) == 0.0
        new = CitrusState(angle=float(angle), collided_moments=torch.full(shape, float(state.num_steps + 1),
                                                                          device=DEVICE),
                          num_steps=state.num_steps + 1, history=state.history + [0.5 ** state.num_steps])
        return spr_true[v] * 3.0, None, new

    def fake_primaries(solver_, fields, angles_, projector, views_per_chunk=None, out=None):
        return out.fill_(3.0)

    monkeypatch.setattr(citrus_module, "citrus_view_step", fake_step)
    monkeypatch.setattr(citrus_module, "primary_signals", fake_primaries)
    config = CitrusConfig(num_outer_iterations=3)
    X_citrus, info = citrus_recon(data["Y"], 1.0e5, angles, grid, geometry, solver, config,
                                  regularization=regularization)
    X_o, _ = mbir_recon(data["Y"], 1.0e5, angles, grid, geometry, config, spr=spr_true, regularization=regularization)
    assert info["records"][3]["accumulator_change_mean"] == 0.25
    rel = float((X_citrus - X_o).norm() / X_o.norm())
    assert rel < 1e-4, rel  # MBIRTORCH's float32 atomics


# ---- scatter step every n outer iterations ------------------------------------------------------------------

@cuda
def test_scatter_every_reuses_the_correction(tiny):
    """scatter_every=2 over 4 outer iterations: the scatter stage runs at k = 0 and 2 only (records 1 and 3), the
    other records reuse the ratio (no accumulator entries, scatter_seconds ~ 0), the accumulators advance twice, and the
    result is finite and corrected on every view. The scatter_every=2 run with 4 iterations must lie between the
    2-iteration and the 4-iteration every-iteration runs in final error (it has the scatter steps of the former and
    the MBIR iterations of the latter)."""
    data, grid, geometry, solver, regularization = tiny
    config = CitrusConfig(num_outer_iterations=4, scatter_every=2)
    X, info = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, solver, config,
                           regularization=regularization)
    assert bool(torch.isfinite(X).all())
    records = info["records"][1:]
    assert [r["scatter_refreshed"] for r in records] == [True, False, True, False]
    assert all("accumulator_change_mean" in r for r in records[::2]) and not any("accumulator_change_mean" in r
                                                                                 for r in records[1::2])
    assert all(r["scatter_seconds"] < 0.5 * min(x["scatter_seconds"] for x in records[::2]) for r in records[1::2])
    assert float(info["spr"].min()) > 0
    assert info["config"]["scatter_every"] == 2
    e_every2 = _rmse(X, data)
    X2, _ = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, solver, CitrusConfig(num_outer_iterations=2),
                         regularization=regularization)
    X4, _ = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, solver, CitrusConfig(num_outer_iterations=4),
                         regularization=regularization)
    e2, e4 = _rmse(X2, data), _rmse(X4, data)
    assert e4 < e_every2 < e2 + 1e-3, (e4, e_every2, e2)


# ---- sparse-view scatter (scatter step on every n-th view, angular interpolation of the others) --------------

@cuda
def test_sparse_views_close_to_every_view(tiny):
    """Strides 2 and 4 (Catmull-Rom, order 3) on the tiny problem, 3 outer iterations, final relative RMSE against
    the stride-1 run. Measured (RTX 3070): stride 1 0.2094; stride 2 0.2068 (order 3) / 0.2071 (1) / 0.2082 (2) /
    0.2112 (0); stride 4 0.2117 (3) / 0.2115 (1) / 0.2122 (2) / 0.2141 (0); order None (skipped views uncorrected)
    0.2547 / 0.2982; no correction 0.3101. Bound: |stride n - stride 1| < 0.01 (vs differences < 0.003; the
    uncorrected skipped views of order None fail it by 0.045 / 0.089)."""
    data, grid, geometry, solver, regularization = tiny
    e = {}
    for stride in (1, 2, 4):
        config = CitrusConfig(num_outer_iterations=3, scatter_view_stride=stride)
        X, info = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, solver, config,
                               regularization=regularization)
        assert bool(torch.isfinite(X).all())
        e[stride] = _rmse(X, data)
        computed = np.arange(0, 16, stride)
        np.testing.assert_array_equal(info["computed_views"], computed)
        assert info["view_interpolated"].dtype == bool and info["view_interpolated"].shape == (16,)
        assert list(np.flatnonzero(~info["view_interpolated"])) == list(computed)
        assert info["state"]["num_views"] == len(computed)  # accumulators only for the computed views
        assert float(info["spr"].min()) > 0  # every view is corrected
        assert ("interpolation_seconds" in info["records"][1]) == (stride > 1)
        assert info["config"]["scatter_view_stride"] == stride
    for stride in (2, 4):
        assert abs(e[stride] - e[1]) < 0.01, e
    # order None (diagnostics): the skipped views are not corrected and the error grows well past the bound
    X, info = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, solver,
                           CitrusConfig(num_outer_iterations=3, scatter_view_stride=4,
                                        scatter_view_interpolation_order=None),
                           regularization=regularization)
    skipped = np.setdiff1d(np.arange(16), np.arange(0, 16, 4))
    assert not info["view_interpolated"].any()
    assert float(info["spr"][torch.as_tensor(skipped)].abs().max()) == 0.0
    assert _rmse(X, data) > e[1] + 0.03, (_rmse(X, data), e)


@cuda
def test_sparse_views_interpolate_the_scatter_signal(tiny, monkeypatch):
    """The driver interpolates the scatter SIGNAL of the computed views and divides by the primary of every view
    (not the ratio): with a fake step (s_v = 2 + cos(angle)) and fake primaries of all views (`primary_signals`: 3 on
    the computed views, 5 on the skipped views), SPR_skipped = (M s_c) / 5 with M = interpolation_weights. The
    primaries come from one `primary_signals` call per scatter stage with MBIRTORCH's `ViewChunkProjector`, never from
    the step."""
    data, grid, geometry, solver, regularization = tiny
    angles = data["angles"]
    view_of = {float(a): v for v, a in enumerate(angles)}
    Rd, Cd = data["Y"].shape[1:]
    signal = {v: 2.0 + math.cos(float(a)) for v, a in enumerate(angles)}
    computed = np.arange(0, 16, 4)
    skipped = np.setdiff1d(np.arange(16), computed)
    calls = []

    def fake_step(solver_, state, fields, angle, *, primary=True):
        assert not primary  # the loop never traces the primary inside the step
        v = view_of[float(angle)]
        new = CitrusState(angle=float(angle), collided_moments=None, num_steps=state.num_steps + 1,
                          history=state.history + [1.0])
        return (torch.full((Rd, Cd), signal[v], device=DEVICE), None, new)

    class FakeStore(StateStore):
        def put(self, slot, accumulator):
            pass

    def fake_primaries(solver_, fields, angles_, projector, views_per_chunk=None, out=None):
        calls.append(type(projector).__name__)
        assert projector.num_views == 16 and views_per_chunk is None
        out.fill_(5.0)
        out[torch.as_tensor(computed, device=out.device)] = 3.0
        return out

    monkeypatch.setattr(citrus_module, "citrus_view_step", fake_step)
    monkeypatch.setattr(citrus_module, "StateStore", FakeStore)
    monkeypatch.setattr(citrus_module, "primary_signals", fake_primaries)
    for order in (1, 3):
        calls.clear()
        config = CitrusConfig(num_outer_iterations=1, scatter_view_stride=4, scatter_view_interpolation_order=order)
        _, info = citrus_recon(data["Y"], 1.0e5, angles, grid, geometry, solver, config, regularization=regularization)
        assert calls == ["ViewChunkProjector"] and "primary_seconds" in info["records"][1]
        M = citrus_module.interpolation_weights(angles, computed, order)
        expect = M @ np.array([signal[int(v)] for v in computed])
        spr = info["spr"][:, Rd // 2, Cd // 2].cpu().numpy()
        np.testing.assert_allclose(spr[computed], expect[computed] / 3.0, rtol=1e-6)
        np.testing.assert_allclose(spr[skipped], expect[skipped] / 5.0, rtol=1e-6)


@cuda
def test_resume_error_sinogram_matches_restart(tiny):
    """CitrusConfig.resume_error_sinogram: the loop keeps MBIRTORCH's error sinogram between outer iterations
    (shifted by each new corrected sinogram) and gives the reconstruction of the restarted MBIR (a fresh y_c - A X
    every iteration) to float rounding, with a scatter stage every 2nd iteration (the reused correction); mbir_recon
    too."""
    data, grid, geometry, solver, regularization = tiny
    out = {}
    for resume in (True, False):
        config = CitrusConfig(num_outer_iterations=4, scatter_every=2, resume_error_sinogram=resume)
        out[resume], info = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, solver, config,
                                         regularization=regularization)
        assert info["config"]["resume_error_sinogram"] is resume
    assert float((out[True] - out[False]).norm() / out[False].norm()) < 1e-5
    for resume in (True, False):
        out[resume], _ = mbir_recon(data["Y"], 1.0e5, data["angles"], grid, geometry,
                                    CitrusConfig(num_outer_iterations=3,
                                                 resume_error_sinogram=resume), regularization=regularization)
    assert float((out[True] - out[False]).norm() / out[False].norm()) < 1e-5


@cuda
def test_citrus_recon_on_short_scan(tiny):
    """The tiny data set is a short scan (the release default): citrus_recon initializes with the Parker-weighted FDK
    (MBIRTORCH's own FDK has no short-scan weighting and leaves the half-rotation data with a shading / scale error),
    and the loop runs and improves on that initialization."""
    from citrus.recon.mbir_bridge import fdk, is_short_scan
    from citrus.utilities import fan_angle, view_angles

    data, grid, geometry, solver, regularization = tiny
    assert is_short_scan(data["angles"]) and data["meta"]["short_scan"] is True
    np.testing.assert_allclose(data["angles"], view_angles(16, geometry), atol=1e-15)
    assert data["meta"]["end_view_angle"] == pytest.approx(math.pi + fan_angle(geometry))
    seen = []
    X, info = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geometry, solver,
                           CitrusConfig(num_outer_iterations=2),
                           regularization=regularization, callback=lambda k, X, record: seen.append(_rmse(X, data)))
    y_free = -torch.log(torch.as_tensor(data["P"], device=DEVICE) / 1.0e5)  # scatter-free: isolates the FDK weighting
    model = info["model"]
    e_parker = _rmse(fdk(model, y_free, geometry=geometry, angles=data["angles"]), data)
    e_plain = _rmse(fdk(model, y_free), data)
    # 16 views of 2 cm voxels: the FDK error is mostly view aliasing here (0.157 vs 0.176 unweighted); the weighting
    # itself is checked on a finer problem in tests/test_mbir_bridge.py
    assert e_parker < 0.95 * e_plain, (e_parker, e_plain)
    assert seen[-1] < seen[0] and bool(torch.isfinite(X).all())
