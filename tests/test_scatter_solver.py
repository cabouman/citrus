"""End-to-end tests of the assembled solver (citrus/scatter_solver/scatter_solver.py), WLAB at 2 cm, supervoxel
factor 2."""

from __future__ import annotations

import math
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from citrus.scatter_solver.config import SolverConfig
from citrus.scatter_solver.cross_sections import MultigroupLibrary, synthetic_library
from citrus.scatter_solver.detector import csi_response, focused_1d_grid, no_grid
from citrus.scatter_solver.geometry import Grid
from citrus.scatter_solver.ray_tracing import SiddonProjectors
from citrus.scatter_solver.scatter_solver import ScatterSolver, ViewResult, coarsen_density, truncated_library
from citrus.scatter_solver.uncollided import (fc_exact_moments, fc_source_at_directions, primary_directions_to_voxels,
                                              restrict_fc_to_ld_moments)
from citrus.utilities import wlab_geometry, wlab_phantom

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 2 cm WLAB grid padded to 32 cm along x so that every axis is divisible by the supervoxel factor 2
GRID = Grid((10, 16, 20), 2.0)
CFG = SolverConfig(num_groups=4, legendre_order=3, num_polar=4, num_azimuthal=8, cone_num_polar=8,
                   cone_num_azimuthal=16, supervoxel_factor=2, si_tol=1e-4, si_max_iterations=40, direction_chunk=8)
IMAGE_KEYS = ("fluence", "planar_fluence", "signal_per_group", "signal")


def peaked_library(num_groups: int = 4, legendre_order: int = 4, g_hg: float = 0.7,
                   num_materials: int = 7) -> MultigroupLibrary:
    """synthetic_library with Henyey-Greenstein (forward-peaked) WITHIN-group kernels, so that all Legendre
    moments are non-zero (mu_s_l[g, g, l] = a g_hg^l)."""
    lib = synthetic_library(num_groups=num_groups, legendre_order=legendre_order, num_materials=num_materials)
    eta = lib.eta_grid
    mu_s_l = lib.mu_s_l.clone()
    mu_s_eta = lib.mu_s_eta.clone()
    l = torch.arange(legendre_order + 1, dtype=torch.float64)
    hg = (1 - g_hg ** 2) / (4 * math.pi * (1 + g_hg ** 2 - 2 * g_hg * eta) ** 1.5)
    for m in range(lib.num_materials):
        for g in range(num_groups):
            a = float(mu_s_l[m, g, g, 0])
            mu_s_l[m, g, g] = a * g_hg ** l
            mu_s_eta[m, g, g] = a * hg
    return replace(lib, mu_s_l=mu_s_l, mu_s_eta=mu_s_eta)


def make_solver(cfg: SolverConfig = CFG, lib: MultigroupLibrary | None = None, grid_response=None,
                **kw) -> ScatterSolver:
    lib = synthetic_library(cfg.num_groups, 5, num_materials=7) if lib is None else lib
    return ScatterSolver(lib, cfg, wlab_geometry(), SiddonProjectors(), csi_response(),
                         no_grid() if grid_response is None else grid_response, device=DEV, dtype=torch.float32, **kw)


@pytest.fixture(scope="module")
def rho() -> torch.Tensor:
    return wlab_phantom(GRID, device=DEV, dtype=torch.float32)


@pytest.fixture(scope="module")
def base_result(rho) -> ViewResult:
    return make_solver().solve_view(rho, GRID, 0.0, keep_phi_total=True)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def _zero_mc(s: ScatterSolver, f) -> torch.Tensor:
    return s.mc_source_moments(f, torch.zeros(s.num_groups, (s.L + 1) ** 2, 4, *f.grid_coarse.shape, device=DEV))


# ---- basic run -------------------------------------------------------------------------------------------

def test_solve_view_runs(base_result):
    r = base_result
    geom = wlab_geometry()
    G = CFG.num_groups
    assert r.si.converged and r.si.num_iterations < CFG.si_max_iterations, r.si.history
    assert r.si.history[-1] < CFG.si_tol and len(r.si.history) == r.si.num_iterations
    for d in (r.primary, r.scatter):
        for k in IMAGE_KEYS:
            t = d[k]
            assert t.shape == ((G,) if k != "signal" else ()) + geom.image_shape
            assert t.device.type == DEV.type and bool(torch.isfinite(t).all())
            assert float(t.min()) >= 0.0, k
        assert bool((d["planar_fluence"] <= d["fluence"] * (1 + 1e-6)).all())
        assert float(d["signal"].sum()) > 0
    assert torch.equal(r.scatter_signal, r.scatter["signal"])
    assert torch.equal(r.scatter_fluence, r.scatter["fluence"])
    assert torch.equal(r.scatter_planar_fluence, r.scatter["planar_fluence"])
    sf = r.scatter_fraction
    assert float(sf.min()) > 0.0 and float(sf.max()) < 1.0
    assert r.phi_total is not None and r.phi_total.shape == (G, 16, 4, 5, 8, 10)
    assert float(r.phi_total[:, 0, 0].min()) > 0
    for k in ("materials", "uncollided", "source_iteration", "mc_source", "cone_directions", "scatter_image",
              "primary", "total"):
        assert r.timings[k] >= 0
    if DEV.type == "cuda":
        assert r.peak_memory_bytes is not None and r.peak_memory_bytes < 4 * 2 ** 30
    d = r.to_numpy_dict()
    assert d["scatter_signal"].shape == geom.image_shape and int(d["si_num_iterations"]) == r.si.num_iterations
    assert "phi_total" in d and "primary_planar_fluence" in d and "timing_total" in d
    assert "scatter_planar_fluence" in d and "timing_scatter_image" in d
    np.testing.assert_allclose(d["scatter_signal"], d["scatter_signal_per_group"].sum(0), rtol=1e-5)


def test_coarse_fields_are_averages(rho):
    s = make_solver()
    f = s.material_fields(rho, GRID)
    assert f.grid_coarse.shape == (5, 8, 10) and f.grid_coarse.delta_voxel == 4.0
    # linear in rho: the coarse mu_t is the supervoxel average of the fine one
    mu_avg = coarsen_density(f.mu_t_fine, 2)
    assert torch.allclose(f.mu_t_coarse, mu_avg, rtol=1e-5, atol=1e-7)
    assert torch.allclose(f.rho_coarse.sum(), rho.sum() / 8, rtol=1e-5)
    assert f.mu_s_l_coarse.shape == (4, 4, CFG.legendre_order + 1, 5, 8, 10)


def test_truncated_library():
    lib = peaked_library(4, 4)
    tr = truncated_library(lib, 3)
    assert tr.mu_s_l.shape[-1] == 4 and torch.equal(tr.mu_t, lib.mu_t)
    assert torch.equal(tr.mu_s_l, lib.mu_s_l[..., :4]) and torch.equal(tr.mu_s_eta, lib.mu_s_eta)
    assert truncated_library(lib, 4) is lib
    with pytest.raises(ValueError):
        truncated_library(lib, 5)
    with pytest.raises(ValueError):
        make_solver(replace(CFG, legendre_order=4, num_polar=6, num_azimuthal=12), lib=synthetic_library(4, 3))


# ---- linearity in the source strength --------------------------------------------------------------------

def test_linearity_in_source_strength(rho, base_result):
    lib = synthetic_library(CFG.num_groups, 5, num_materials=7)
    lib2 = replace(lib, spectrum_fraction=2.5 * lib.spectrum_fraction, fine_weight=2.5 * lib.fine_weight)
    r2 = make_solver(lib=lib2).solve_view(rho, GRID, 0.0)
    r = base_result
    assert r2.si.num_iterations == r.si.num_iterations
    for name in ("primary", "scatter"):
        a, b = getattr(r2, name)["signal"], getattr(r, name)["signal"]
        assert torch.allclose(a, 2.5 * b, rtol=1e-4, atol=1e-6 * float(b.max())), name
    assert torch.allclose(r2.scatter_fraction, r.scatter_fraction, rtol=1e-4, atol=1e-6)


# ---- chunking ------------------------------------------------------------------------------------------------

def test_independent_of_direction_chunk(rho, base_result):
    r1 = make_solver(replace(CFG, direction_chunk=1)).solve_view(rho, GRID, 0.0, keep_phi_total=True)
    r = base_result
    assert r1.si.num_iterations == r.si.num_iterations
    assert _rel(r1.phi_total, r.phi_total) < 1e-5
    for name in ("primary", "scatter"):
        assert _rel(getattr(r1, name)["signal"], getattr(r, name)["signal"]) < 1e-5, name
    assert _rel(r1.scatter_signal, r.scatter_signal) < 1e-5


# ---- anti-scatter grid ----------------------------------------------------------------------------------------

def test_grid_lowers_scatter_fraction(rho, base_result):
    r_g = make_solver(grid_response=focused_1d_grid()).solve_view(rho, GRID, 0.0)
    assert bool((r_g.scatter_fraction < base_result.scatter_fraction).all())
    # the fluences do not depend on the detector side
    assert torch.allclose(r_g.scatter["fluence"], base_result.scatter["fluence"], rtol=1e-5)


# ---- config ---------------------------------------------------------------------------------------------------

def test_config_validation():
    SolverConfig().validate()
    for bad in (dict(num_polar=7), dict(num_azimuthal=14), dict(supervoxel_factor=0), dict(si_tol=0.0),
                dict(direction_chunk=0), dict(dtype="float16"), dict(group_edges_kev=(100.0, 10.0))):
        with pytest.raises(ValueError):
            replace(SolverConfig(), **bad).validate()


def test_fc_sampling_conservative(rho):
    """The FC source of the sweep is corrected (uncollided.fc_sampling_correction, l0 + l1): the sweep-quadrature
    integral of the restricted FC source equals the exact l = 0 transfer integral (restricted to the supervoxels, all
    four LD moments) to 1e-6, which the uncorrected point samples miss for a forward-peaked (HG g = 0.7) kernel; the
    l = 1 moment about Omega_UC is matched on the fine grid wherever the non-negativity clamp of the factor is
    inactive, and the source stays non-negative."""
    lib = peaked_library(CFG.num_groups, 5, 0.7)
    geom = wlab_geometry()
    sol = ScatterSolver(lib, CFG, geom, SiddonProjectors(), csi_response(), no_grid(), device=DEV, dtype=torch.float64)
    f = sol.material_fields(rho, GRID)
    phi_uc = sol.uncollided(f, 0.0)
    ctx = sol.sweep_context(f.grid_coarse)
    idx = torch.arange(ctx.num_directions, device=DEV)
    w = ctx.quad.weight.to(DEV, torch.float64)
    om = ctx.quad.omega.to(DEV, torch.float64)
    fc_point = fc_source_at_directions(sol.lib, f.rho_fine, phi_uc, GRID, geom, 0.0, om)  # uncorrected samples

    E = fc_exact_moments(sol.lib, f.rho_fine, phi_uc, l_max=1)  # [2, G, R, C, S]
    exact = restrict_fc_to_ld_moments(E[0], GRID, CFG.supervoxel_factor)
    scale = float(exact[:, 0].abs().max())
    got = torch.einsum("d,dg...->g...", w, sol.fc_moment_source(f, 0.0, phi_uc, ctx)(idx))  # [G, 4, Rc, Cc, Sc]
    pt = torch.einsum("d,dg...->g...", w, restrict_fc_to_ld_moments(fc_point, GRID, CFG.supervoxel_factor))
    assert float((got - exact).abs().max()) < 1e-6 * scale
    assert float((got[:, 0].sum() - exact[:, 0].sum()).abs()) < 1e-6 * float(exact[:, 0].sum())
    assert float((pt[:, 0] - exact[:, 0]).abs().max()) > 1e-3 * scale

    corr = sol.fc_sampling_correction(f, 0.0, phi_uc, ctx)
    fc = corr.apply(fc_point, om)
    assert float(fc.min()) >= 0.0
    mu = torch.einsum("rcsx,kx->krcs", primary_directions_to_voxels(GRID, geom, 0.0, DEV, torch.float64), om)
    J = torch.einsum("k,krcs,kgrcs->grcs", w, mu, fc)
    free = corr.beta.abs() < 1
    assert float(free.double().mean()) > 0.5
    e1 = float(E[1].abs().max())
    assert float((J - E[1])[free].abs().max()) < 1e-6 * e1


@pytest.mark.skipif(DEV.type != "cuda", reason="float32 CUDA accumulation")
def test_fc_sampling_float32_accumulation(rho, monkeypatch):
    """The sampling-correction sums are formed by float32 matmuls (blocks of at most _FC_SUM_MAX_DIRECTIONS
    directions) and accumulated across blocks in float64 (uncollided.fc_quadrature_sums). On float32 CUDA the factors
    differ from an all-float64 computation by < 1e-5 relative (scale relative; beta absolute, i.e. relative to the
    unit factor 1 + beta mu): (a) the same float32 source with every sum in float64 and (b) the whole pass in float64
    (float64 inputs, PyTorch FC path). Splitting the pass into several blocks changes nothing beyond rounding."""
    from citrus.scatter_solver import uncollided as uncollided_mod
    from citrus.scatter_solver.uncollided import fc_correction_factors, fc_quadrature_sums, fc_sampling_correction

    lib = peaked_library(CFG.num_groups, 5, 0.7)
    geom = wlab_geometry()
    sol = ScatterSolver(lib, CFG, geom, SiddonProjectors(), csi_response(), no_grid(), device=DEV, dtype=torch.float32)
    f = sol.material_fields(rho, GRID)
    phi_uc = sol.uncollided(f, 0.0)
    ctx = sol.sweep_context(f.grid_coarse)
    om32 = ctx.quad.omega.to(DEV, torch.float32)
    w64 = ctx.quad.weight.to(DEV, torch.float64)
    corr = sol.fc_sampling_correction(f, 0.0, phi_uc, ctx)
    assert corr.scale.dtype == torch.float32 and corr.beta.dtype == torch.float32
    E = fc_exact_moments(sol.lib, f.rho_fine.double(), phi_uc.double(), l_max=1)

    # (a) the same float32 FC source, all sums in float64
    fc = fc_source_at_directions(sol.lib, f.rho_fine, phi_uc, GRID, geom, 0.0, om32).double()
    mu = torch.einsum("rcsx,kx->krcs", primary_directions_to_voxels(GRID, geom, 0.0, DEV, torch.float32), om32).double()
    I = torch.stack([torch.einsum("k,kg...->g...", w64, fc), torch.einsum("k,k...,kg...->g...", w64, mu, fc),
                     torch.einsum("k,k...,kg...->g...", w64, mu * mu, fc)])
    ref_a = fc_correction_factors(I, E)
    # (b) everything in float64
    c64 = fc_sampling_correction(sol.lib, f.rho_fine.double(), phi_uc.double(), GRID, geom, 0.0,
                                 ctx.quad.omega.to(DEV, torch.float64), w64, batch=CFG.direction_chunk)
    ref_b = (c64.scale, c64.beta)
    for name, (s_ref, b_ref) in (("float64 sums", ref_a), ("all float64", ref_b)):
        ds = float(((corr.scale.double() - s_ref).abs() / s_ref.abs()).max())
        db = float((corr.beta.double() - b_ref).abs().max())
        assert ds < 1e-5 and db < 1e-5, (name, ds, db)
        assert float(b_ref.abs().max()) > 1e-2  # the correction is not trivial

    # several float32 blocks (block of 16 directions, FC batches of 5 that do not divide it) == one block
    I1 = fc_quadrature_sums(sol.lib, f.rho_fine, phi_uc, GRID, geom, 0.0, om32, w64, batch=8)
    monkeypatch.setattr(uncollided_mod, "_FC_SUM_MAX_DIRECTIONS", 16)
    I2 = fc_quadrature_sums(sol.lib, f.rho_fine, phi_uc, GRID, geom, 0.0, om32, w64, batch=5)
    assert I1.dtype == I2.dtype == torch.float64
    assert float((I1 - I2).abs().max()) < 1e-6 * float(I1[0].abs().max())
    assert float((I1 - I).abs().max()) < 1e-6 * float(I[0].abs().max())


# ---- real library (slow) ---------------------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("CITRUS_SLOW_TESTS") != "1", reason="set CITRUS_SLOW_TESTS=1 (builds the "
                    "XrayPhysics library, cached in the library cache)")
def test_real_library_wlab(rho):
    from citrus.scatter_solver.scatter_solver import build_solver_library

    cfg = replace(CFG, num_groups=6, si_tol=1e-3)
    lib = build_solver_library(cfg)
    assert lib.num_groups == 6
    r = make_solver(cfg, lib=lib).solve_view(rho, GRID, 0.0)
    assert r.si.converged and 3 < r.si.num_iterations < 30, r.si.history
    for d in (r.primary, r.scatter):
        assert float(d["signal"].min()) >= 0 and float(d["signal"].sum()) > 0
    sf = r.scatter_fraction
    assert float(sf.min()) > 0 and float(sf.max()) < 1
    # behind the 20 cm water slab (central row and column) the no-grid SF is large but below 1
    c = sf[sf.shape[0] // 2, sf.shape[1] // 2]
    assert 0.2 < float(c) < 0.9, float(c)


# ---- supervoxel factors: the transport grid and the sub-voxel points --------------------------------------------

GRID4 = Grid((12, 16, 20), 2.0)  # divisible by 4


@pytest.fixture(scope="module")
def rho4() -> torch.Tensor:
    return wlab_phantom(GRID4, device=DEV, dtype=torch.float32)


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_supervoxel_factors(rho4, factor):
    """solve_view for supervoxel factors 1, 2, 4: all images are finite and non-negative; at factor 1 the first-order
    image (the solver's scatter image with a zero multiple-collision source) is the single-grid collapsed-cone image
    of the FC source (scatter_image.scatter_image_from_source, same footprint); coarser transport grids change it by a
    bounded amount."""
    from citrus.scatter_solver.scatter_image import scatter_image_from_source

    cfg = replace(CFG, supervoxel_factor=factor)
    s = make_solver(cfg)
    r = s.solve_view(rho4, GRID4, 0.0)
    assert r.si.converged
    for d in (r.primary, r.scatter):
        for k in IMAGE_KEYS:
            assert bool(torch.isfinite(d[k]).all()) and float(d[k].min()) >= 0.0, k
        assert float(d["signal"].sum()) > 0
    assert float(r.scatter_fraction.min()) > 0 and float(r.scatter_fraction.max()) < 1
    f = s.material_fields(rho4, GRID4)
    first = s.scatter_image(f, 0.0, s.uncollided(f, 0.0), _zero_mc(s, f))
    # the single-grid image of the FC source on the fine grid, attenuated with the fine mu_t
    s1 = make_solver(replace(cfg, supervoxel_factor=1))
    f1 = s1.material_fields(rho4, GRID4)
    phi_uc = s1.uncollided(f1, 0.0)
    fine = scatter_image_from_source(
        lambda om: fc_source_at_directions(s1.lib, f1.rho_fine, phi_uc, GRID4, s1.geom, 0.0, om), GRID4, s1.geom, 0.0,
        s1.cone_quadrature(GRID4, 0.0), f1.mu_t_coarse, s1.projectors, s1.detector, s1.grid_response, s1.lib,
        chunk=cfg.direction_chunk)
    d_fine = _rel(first["signal"], fine["signal"])
    if factor == 1:
        assert d_fine < 1e-6, d_fine
    else:
        # 4 / 8 cm cells against 2 cm fine voxels, 8x16 cone, synthetic library
        assert 1e-4 < d_fine < 0.35, d_fine
    assert float(first["signal"].sum()) < float(r.scatter_signal.sum())  # the multiple collisions add to it


def test_scatter_image_is_combined_pass(rho):
    """solver.scatter_image == scatter_image.combined_scatter_image with the solver's fields (mu_t_coarse, cone set,
    chunking, footprint); a larger attenuation map attenuates more."""
    from citrus.scatter_solver.scatter_image import combined_scatter_image

    s = make_solver()
    f = s.material_fields(rho, GRID)
    phi_uc = s.uncollided(f, 0.0)
    T = _zero_mc(s, f)
    got = s.scatter_image(f, 0.0, phi_uc, T)
    q = s.cone_quadrature(GRID, 0.0)
    exp = combined_scatter_image(s.lib, f.rho_fine, phi_uc, GRID, 2, T, s.L, s.geom, 0.0, q, f.mu_t_coarse,
                                 s.projectors, s.detector, s.grid_response, chunk=CFG.direction_chunk)
    assert torch.allclose(got["signal"], exp["signal"], rtol=1e-6)
    more = combined_scatter_image(s.lib, f.rho_fine, phi_uc, GRID, 2, T, s.L, s.geom, 0.0, q, 1.2 * f.mu_t_coarse,
                                  s.projectors, s.detector, s.grid_response, chunk=CFG.direction_chunk)
    assert float(got["signal"].sum()) > float(more["signal"].sum())


# ---- direction context cache ----------------------------------------------------------------------------------

def test_direction_context_cache(rho, base_result, monkeypatch):
    """One direction context per view, reused by later solves and by the CITRUS step (same object), LRU-bounded,
    dropped by clear_caches; the images do not change."""
    import citrus.scatter_solver.scatter_solver as solver_mod
    from citrus.citrus import CitrusState, citrus_step

    built = []
    real = solver_mod.make_direction_context

    def counting(*a, **k):
        ctx = real(*a, **k)
        built.append(ctx)
        return ctx

    monkeypatch.setattr(solver_mod, "make_direction_context", counting)
    s = make_solver()
    r1 = s.solve_view(rho, GRID, 0.0)
    assert len(built) == 1
    fields = s.material_fields(rho, GRID)
    c1 = s.direction_context(fields, 0.0)
    assert c1 is built[0] and s.direction_context(fields, 0.0, s.cone_quadrature(GRID, 0.0)) is c1
    r2 = s.solve_view(rho, GRID, 0.0)
    assert len(built) == 1 and s.direction_context(fields, 0.0) is c1
    state = CitrusState()
    for _ in range(2):
        _, _, state = citrus_step(s, state, rho, GRID, 0.0)
    assert len(built) == 1, "the CITRUS step rebuilt the direction context"
    for k in IMAGE_KEYS:
        a, b, ref = r2.scatter[k], r1.scatter[k], base_result.scatter[k]
        assert _rel(a, b) < 1e-6 and _rel(a, ref) < 1e-6, k
    # no cache: identical images, nothing kept
    s0 = make_solver(direction_context_cache=0)
    r0 = s0.solve_view(rho, GRID, 0.0)
    assert len(s0._dir_ctx_cache) == 0
    assert _rel(r0.scatter["signal"], r1.scatter["signal"]) < 1e-6
    # LRU bound: one entry, the newest view kept
    s1 = make_solver(direction_context_cache=1)
    f1 = s1.material_fields(rho, GRID)
    a0 = s1.direction_context(f1, 0.0)
    a1 = s1.direction_context(f1, 0.5)
    assert len(s1._dir_ctx_cache) == 1 and s1.direction_context(f1, 0.5) is a1
    assert s1.direction_context(f1, 0.0) is not a0
    # clear_caches drops everything
    s.clear_caches()
    assert not (s._dir_ctx_cache or s._cone_cache or s._ctx_cache)
    assert s.direction_context(fields, 0.0) is not c1
