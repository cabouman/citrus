"""CITRUS reconstruction loop (citrus/citrus.py): the compact per-view accumulator store, citrus_recon and mbir_recon
on a tiny noise-free problem, the frozen regularization, and CITRUS with the true scatter == the fixed-SPR MBIR."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from citrus import citrus as citrus_mod
from citrus.citrus import (CitrusConfig, CitrusState, StateStore, citrus_recon, estimate_regularization,
                           mbir_recon)
from citrus.scatter_solver.config import SolverConfig

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the CITRUS reconstruction tests run on CUDA")


# ---- accumulator storage (no solver needed) -----------------------------------------------------------------

SHAPE = (3, 16, 4, 4, 5, 6)  # [G, (L+1)^2 with L = 3, 4, Rc, Cc, Sc]


def _phi(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # realistic magnitudes (collided moments ~1e-6 per source photon/sr: plain float16 would underflow)
    return (1e-6 * torch.randn(SHAPE, generator=g) * torch.logspace(0, -3, SHAPE[1])[None, :, None, None, None,
                                                                                       None]).to(DEV)


@pytest.mark.parametrize("location", ["gpu", "cpu"])
def test_state_store_round_trip(location):
    if location == "gpu" and DEV.type != "cuda":
        pytest.skip("no CUDA")
    store = StateStore(3, SHAPE, DEV, location=location)
    assert store.get(0) is None and store.location == location
    phis = [_phi(i) for i in range(3)]
    for i, p in enumerate(phis):
        store.put(i, p)
    n32 = int(np.prod(SHAPE)) * 4
    for i, p in enumerate(phis):
        q = store.get(i)
        assert q.shape == SHAPE and q.dtype == torch.float32 and q.device.type == DEV.type
        # LD cell mean kept (scaled float16: no underflow), slopes zero
        assert float((q[:, :, 0] - p[:, :, 0]).norm() / p[:, :, 0].norm()) < 2e-4
        assert float(q[:, :, 1:].abs().max()) == 0.0
    assert store.bytes_per_view < n32 / 7.5  # 1/8 plus one float32 scale per (group, moment)
    if location == "cpu" and DEV.type == "cuda":
        assert store._data[0].is_pinned() and store._data[0].device.type == "cpu"
    with pytest.raises(ValueError):
        store.put(0, phis[0][:, :9])
    store.close()


def test_state_store_auto_location():
    shape = (6, 64, 4, 20, 30, 40)  # the 0.5 cm WLAB accumulator with 1 cm transport cells
    per = int(np.prod(shape)) * 4
    s36 = StateStore(36, shape, DEV)
    assert s36.bytes_per_view < per / 7.9 and abs(s36.bytes_per_view / 2 ** 20 - 17.6) < 0.1
    assert s36.location == ("gpu" if DEV.type == "cuda" else "cpu")
    s180 = StateStore(180, shape, DEV)  # 180 x 17.6 MiB = 3164 MiB > 2600 MiB: host memory (nothing allocated yet)
    assert s180.location == "cpu" and s180.pin == (DEV.type == "cuda")
    assert StateStore(180, shape, DEV, gpu_budget_mib=4000).location == ("gpu" if DEV.type == "cuda" else "cpu")


# ---- a tiny problem ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_lib():
    from citrus.scatter_solver.scatter_solver import build_solver_library

    try:
        return build_solver_library(SolverConfig())
    except Exception as exc:  # no cached library and no XrayPhysics
        pytest.skip(f"real library unavailable: {exc}")


@pytest.fixture(scope="module")
def tiny(real_lib):
    """16 views, 2 cm grid (10 x 15 x 20), a full-field panel of 1.6 cm pixels (46 x 38), noise-free (Y = P + S),
    the real library, the regularization estimated on the true primary (frozen for every reconstruction)."""
    pytest.importorskip("mbirtorch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    from citrus.recon.mbir_bridge import make_cone_model
    from citrus.utilities import synthesize_wlab_dataset, wlab_solver
    from citrus.utilities import wlab_geometry, wlab_grid

    geom = replace(wlab_geometry(), num_det_rows=46, num_det_channels=38, delta_det_row=1.6, delta_det_channel=1.6)
    data = synthesize_wlab_dataset(16, 2.0, noise=False, seed=0, lib=real_lib, geom=geom)
    grid = wlab_grid(2.0)
    solver = wlab_solver(2.0, lib=real_lib, geom=geom, device=DEV)
    model = make_cone_model(grid, geom, data["angles"], device=DEV)
    Y = torch.as_tensor(data["Y"], device=DEV)
    P = torch.as_tensor(data["P"], device=DEV)
    reg = estimate_regularization(model, -torch.log(P / 1.0e5), Y / 1.0e5)
    return data, grid, geom, solver, reg


def _rmse(X, data) -> float:
    Xt = torch.as_tensor(data["X_true"], device=X.device)
    return float((X - Xt).norm() / Xt.norm())


@cuda
def test_citrus_recon_runs_and_beats_no_correction(tiny):
    data, grid, geom, solver, reg = tiny
    cfg = CitrusConfig(num_outer=3)
    seen = []
    X, info = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geom, solver, cfg, regularization=reg,
                           callback=lambda k, X, rec: seen.append((k, _rmse(X, data))))
    assert X.shape == grid.shape and X.device.type == "cuda" and bool(torch.isfinite(X).all())
    assert [k for k, _ in seen] == [0, 1, 2, 3] and len(info["records"]) == 4
    assert info["regularization"] == reg
    model = info["model"]
    assert model.get_params("auto_regularize_flag") is False and float(model.get_params("sigma_y")) == reg["sigma_y"]
    r = info["records"]
    assert r[1]["acc_change_mean"] == 1.0 and r[3]["acc_change_mean"] < 1.0
    assert "scatter_change_rel" in r[2] and r[1]["t_scatter"] > 0 and r[1]["t_mbir"] > 0
    assert info["spr"].shape == data["Y"].shape and float(info["spr"].min()) > 0
    assert info["state"]["num_views"] == 16 and info["state"]["location"] == "gpu"
    # the same MBIR loop without correction, and with the true SPR (the floor of the correction)
    seen_n = []
    X_none, info_n = mbir_recon(data["Y"], 1.0e5, data["angles"], grid, geom, cfg, regularization=reg,
                                device=DEV, callback=lambda k, X, rec: seen_n.append(_rmse(X, data)))
    X_orc, _ = mbir_recon(data["Y"], 1.0e5, data["angles"], grid, geom, cfg, spr=data["SPR"], regularization=reg,
                          device=DEV)
    assert seen_n[0] == seen[0][1]  # the same FDK initialization
    e = {"citrus": _rmse(X, data), "none": _rmse(X_none, data), "oracle": _rmse(X_orc, data)}
    assert e["oracle"] < e["none"] and e["citrus"] < e["none"], e
    assert info_n["spr"] is None and float(info_n["scatter_counts"].abs().max()) == 0.0


@cuda
def test_citrus_with_true_scatter_is_fixed_spr_mbir(tiny, monkeypatch):
    """With the scatter step replaced by the TRUE scatter, CITRUS runs the identical MBIR sequence as mbir_recon with
    spr = SPR_true (same seeds, same frozen sigmas, same weights); the accumulators go through the compact store (LD
    mean kept, slopes dropped)."""
    data, grid, geom, solver, reg = tiny
    angles = data["angles"]
    view_of = {float(a): v for v, a in enumerate(angles)}
    spr_true = torch.as_tensor(data["SPR"], device=DEV)
    coarse = grid.coarsen(solver.config.supervoxel_factor)
    shape = (solver.num_groups, (solver.L + 1) ** 2, 4) + tuple(coarse.shape)

    def fake_step(solver_, state, rho, grid_, angle, fields=None):
        v = view_of[float(angle)]
        if state.Phi_gt1 is not None:  # the stored accumulator comes back (cell mean only)
            assert torch.allclose(state.Phi_gt1[:, :, 0], torch.full_like(state.Phi_gt1[:, :, 0],
                                                                          float(state.num_steps)), rtol=1e-3)
            assert float(state.Phi_gt1[:, :, 1:].abs().max()) == 0.0
        new = CitrusState(angle=float(angle), Phi_gt1=torch.full(shape, float(state.num_steps + 1), device=DEV),
                          num_steps=state.num_steps + 1, history=state.history + [0.5 ** state.num_steps])
        p = torch.full_like(spr_true[v], 3.0)
        return spr_true[v] * p, p, new

    monkeypatch.setattr(citrus_mod, "citrus_step", fake_step)
    cfg = CitrusConfig(num_outer=3)
    X_c, info = citrus_recon(data["Y"], 1.0e5, angles, grid, geom, solver, cfg, regularization=reg)
    X_o, _ = mbir_recon(data["Y"], 1.0e5, angles, grid, geom, cfg, spr=spr_true, regularization=reg)
    assert info["records"][3]["acc_change_mean"] == 0.25
    rel = float((X_c - X_o).norm() / X_o.norm())
    assert rel < 1e-4, rel  # MBIRTORCH's float32 atomics


# ---- sparse-view scatter (scatter step on every n-th view, angular interpolation of the others) --------------

@cuda
def test_sparse_views_close_to_every_view(tiny):
    """Strides 2 and 4 (Catmull-Rom, order 3) on the tiny problem, 3 outer iterations, final relative RMSE against
    the stride-1 run. Measured (RTX 3070): stride 1 0.2094; stride 2 0.2068 (order 3) / 0.2071 (1) / 0.2082 (2) /
    0.2112 (0); stride 4 0.2117 (3) / 0.2115 (1) / 0.2122 (2) / 0.2141 (0); order None (skipped views uncorrected)
    0.2547 / 0.2982; no correction 0.3101. Bound: |stride n - stride 1| < 0.01 (vs differences < 0.003; the
    uncorrected skipped views of order None fail it by 0.045 / 0.089)."""
    data, grid, geom, solver, reg = tiny
    e = {}
    for stride in (1, 2, 4):
        cfg = CitrusConfig(num_outer=3, scatter_view_stride=stride)
        X, info = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geom, solver, cfg, regularization=reg)
        assert bool(torch.isfinite(X).all())
        e[stride] = _rmse(X, data)
        computed = np.arange(0, 16, stride)
        np.testing.assert_array_equal(info["computed_views"], computed)
        assert info["view_interpolated"].dtype == bool and info["view_interpolated"].shape == (16,)
        assert list(np.flatnonzero(~info["view_interpolated"])) == list(computed)
        assert info["state"]["num_views"] == len(computed)  # accumulators only for the computed views
        assert float(info["spr"].min()) > 0  # every view is corrected
        assert ("t_skipped" in info["records"][1]) == (stride > 1)
        assert info["config"]["scatter_view_stride"] == stride
    for stride in (2, 4):
        assert abs(e[stride] - e[1]) < 0.01, e
    # order None (diagnostics): the skipped views are not corrected and the error grows well past the bound
    X, info = citrus_recon(data["Y"], 1.0e5, data["angles"], grid, geom, solver,
                           CitrusConfig(num_outer=3, scatter_view_stride=4, scatter_view_interp_order=None),
                           regularization=reg)
    skipped = np.setdiff1d(np.arange(16), np.arange(0, 16, 4))
    assert not info["view_interpolated"].any()
    assert float(info["spr"][torch.as_tensor(skipped)].abs().max()) == 0.0
    assert _rmse(X, data) > e[1] + 0.03, (_rmse(X, data), e)


@cuda
def test_sparse_views_interpolate_the_scatter_signal(tiny, monkeypatch):
    """The driver interpolates the scatter SIGNAL of the computed views and divides by the solver primary of every
    view (not the ratio): with a fake step (s_v = 2 + cos(angle), p = 3 on computed views) and a fake primary (5 on
    the skipped views), SPR_skipped = (M s_c) / 5 with M = interpolation_weights."""
    data, grid, geom, solver, reg = tiny
    angles = data["angles"]
    view_of = {float(a): v for v, a in enumerate(angles)}
    Rd, Cd = data["Y"].shape[1:]
    sig = {v: 2.0 + math.cos(float(a)) for v, a in enumerate(angles)}
    primary_calls = []

    def fake_step(solver_, state, rho, grid_, angle, fields=None):
        v = view_of[float(angle)]
        new = CitrusState(angle=float(angle), Phi_gt1=None, num_steps=state.num_steps + 1,
                          history=state.history + [1.0])
        return (torch.full((Rd, Cd), sig[v], device=DEV), torch.full((Rd, Cd), 3.0, device=DEV), new)

    class FakeStore(StateStore):
        def put(self, i, Phi):
            pass

    def fake_primary(fields, angle):
        primary_calls.append(view_of[float(angle)])
        return {"signal": torch.full((Rd, Cd), 5.0, device=DEV)}

    monkeypatch.setattr(citrus_mod, "citrus_step", fake_step)
    monkeypatch.setattr(citrus_mod, "StateStore", FakeStore)
    monkeypatch.setattr(solver, "primary", fake_primary)
    for order in (1, 3):
        primary_calls.clear()
        cfg = CitrusConfig(num_outer=1, scatter_view_stride=4, scatter_view_interp_order=order)
        _, info = citrus_recon(data["Y"], 1.0e5, angles, grid, geom, solver, cfg, regularization=reg)
        computed = np.arange(0, 16, 4)
        skipped = np.setdiff1d(np.arange(16), computed)
        assert sorted(primary_calls) == list(skipped)  # the solver primary only on the skipped views
        M = citrus_mod.interpolation_weights(angles, computed, order)
        expect = M @ np.array([sig[int(v)] for v in computed])
        spr = info["spr"][:, Rd // 2, Cd // 2].cpu().numpy()
        np.testing.assert_allclose(spr[computed], expect[computed] / 3.0, rtol=1e-6)
        np.testing.assert_allclose(spr[skipped], expect[skipped] / 5.0, rtol=1e-6)
