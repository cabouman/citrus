"""End-to-end tests of the assembled solver (citrus/scatter_solver/scatter_solver.py), WLAB at 2 cm, 4 cm transport
cells, q = 2 quadrature points per axis (the quadrature grid is the 2 cm grid itself)."""

from __future__ import annotations

import math
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from conftest import SPECTRUM_125KVP
from citrus.scatter_solver.config import SolverConfig
from citrus.scatter_solver.cross_sections import MultigroupLibrary, mu_t_volume, synthetic_library
from citrus.scatter_solver.detector import csi_response, focused_1d_grid, no_grid
from citrus.scatter_solver.geometry import Grid
from citrus.scatter_solver.ray_tracing import SiddonProjectors
from citrus.scatter_solver.scatter_solver import ScatterSolver, ViewResult, coarsen_density, truncated_library
from citrus.scatter_solver.uncollided import (first_collision_exact_moments, first_collision_source_at_directions,
                                              primary_directions_to_voxels,
                                              restrict_to_linear_discontinuous_moments)
from citrus.utilities import wlab_geometry, wlab_phantom

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 2 cm WLAB grid padded to 32 cm along x so that the 4 cm transport cells tile it
GRID = Grid((10, 16, 20), 2.0)
CONFIG = SolverConfig(num_groups=4, legendre_order=3, num_polar=4, num_azimuthal=8, cone_num_polar=8,
                      cone_num_azimuthal=16, transport_cell_cm=4.0, source_iteration_tolerance=1e-4,
                      max_source_iterations=40, direction_chunk=8)
IMAGE_KEYS = ("fluence", "planar_fluence", "signal_per_group", "signal")


def peaked_library(num_groups: int = 4, legendre_order: int = 4, g_hg: float = 0.7,
                   num_materials: int = 7) -> MultigroupLibrary:
    """synthetic_library with Henyey-Greenstein (forward-peaked) WITHIN-group kernels, so that all Legendre
    moments are non-zero (mu_s_l[g, g, l] = a g_hg^l)."""
    library = synthetic_library(num_groups=num_groups, legendre_order=legendre_order, num_materials=num_materials)
    eta = library.eta_grid
    mu_s_l = library.mu_s_l.clone()
    mu_s_eta = library.mu_s_eta.clone()
    l = torch.arange(legendre_order + 1, dtype=torch.float64)
    hg = (1 - g_hg ** 2) / (4 * math.pi * (1 + g_hg ** 2 - 2 * g_hg * eta) ** 1.5)
    for m in range(library.num_materials):
        for g in range(num_groups):
            a = float(mu_s_l[m, g, g, 0])
            mu_s_l[m, g, g] = a * g_hg ** l
            mu_s_eta[m, g, g] = a * hg
    return replace(library, mu_s_l=mu_s_l, mu_s_eta=mu_s_eta)


def make_solver(config: SolverConfig = CONFIG, library: MultigroupLibrary | None = None, grid_response=None,
                **kw) -> ScatterSolver:
    library = synthetic_library(config.num_groups, 5, num_materials=7) if library is None else library
    return ScatterSolver(library, config, wlab_geometry(), SiddonProjectors(), csi_response(),
                         no_grid() if grid_response is None else grid_response, device=DEVICE, dtype=torch.float32,
                         **kw)


@pytest.fixture(scope="module")
def rho() -> torch.Tensor:
    return wlab_phantom(GRID, device=DEVICE, dtype=torch.float32)


@pytest.fixture(scope="module")
def base_result(rho) -> ViewResult:
    return make_solver().solve_view(rho, GRID, 0.0, keep_phi_total=True)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def _zero_multiple_collision(s: ScatterSolver, f) -> torch.Tensor:
    return s.multiple_collision_source_moments(f, torch.zeros(s.num_groups, (s.L + 1) ** 2, 4, *f.grid_coarse.shape,
                                                              device=DEVICE))


# ---- basic run -------------------------------------------------------------------------------------------

def test_solve_view_runs(base_result):
    r = base_result
    geometry = wlab_geometry()
    G = CONFIG.num_groups
    iteration_result = r.source_iteration_result
    assert iteration_result.converged and iteration_result.num_iterations < CONFIG.max_source_iterations, \
        iteration_result.history
    assert iteration_result.history[-1] < CONFIG.source_iteration_tolerance
    assert len(iteration_result.history) == iteration_result.num_iterations
    for d in (r.primary, r.scatter):
        for k in IMAGE_KEYS:
            t = d[k]
            assert t.shape == ((G,) if k != "signal" else ()) + geometry.image_shape
            assert t.device.type == DEVICE.type and bool(torch.isfinite(t).all())
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
    for k in ("materials", "uncollided", "source_iteration", "multiple_collision_source", "cone_directions",
              "scatter_image",
              "primary", "total"):
        assert r.timings[k] >= 0
    if DEVICE.type == "cuda":
        assert r.peak_memory_bytes is not None and r.peak_memory_bytes < 4 * 2 ** 30
    d = r.to_numpy_dict()
    assert d["scatter_signal"].shape == geometry.image_shape
    assert int(d["source_iteration_num_iterations"]) == r.source_iteration_result.num_iterations
    assert "phi_total" in d and "primary_planar_fluence" in d and "timing_total" in d
    assert "scatter_planar_fluence" in d and "timing_scatter_image" in d
    np.testing.assert_allclose(d["scatter_signal"], d["scatter_signal_per_group"].sum(0), rtol=1e-5)


def test_coarse_fields_are_averages(rho):
    s = make_solver()
    f = s.material_fields(rho, GRID)
    assert f.grid_coarse.shape == (5, 8, 10) and f.grid_coarse.delta_voxel == 4.0
    # 2 points per 4 cm cell = the 2 cm grid
    assert f.q == 2 and f.grid_quadrature == GRID and f.rho_quadrature is f.rho_fine
    # linear in rho: the coarse mu_t is the cell average of the fine one
    mu_avg = coarsen_density(mu_t_volume(s.transport_library, rho), GRID, f.grid_coarse)
    assert torch.allclose(f.mu_t_coarse, mu_avg, rtol=1e-5, atol=1e-7)
    assert torch.allclose(f.rho_coarse.sum(), rho.sum() / 8, rtol=1e-5)
    assert f.mu_s_l_coarse.shape == (4, 4, CONFIG.legendre_order + 1, 5, 8, 10)
    assert not any(t.shape[0] == CONFIG.num_groups and tuple(t.shape[1:]) == GRID.shape
                   for t in vars(f).values() if isinstance(t, torch.Tensor))  # no [G, ...] fine-grid field


def test_truncated_library():
    library = peaked_library(4, 4)
    tr = truncated_library(library, 3)
    assert tr.mu_s_l.shape[-1] == 4 and torch.equal(tr.mu_t, library.mu_t)
    assert torch.equal(tr.mu_s_l, library.mu_s_l[..., :4]) and torch.equal(tr.mu_s_eta, library.mu_s_eta)
    assert truncated_library(library, 4) is library
    with pytest.raises(ValueError):
        truncated_library(library, 5)
    with pytest.raises(ValueError):
        make_solver(replace(CONFIG, legendre_order=4, num_polar=6, num_azimuthal=12), library=synthetic_library(4, 3))


# ---- linearity in the source strength --------------------------------------------------------------------

def test_linearity_in_source_strength(rho, base_result):
    library = synthetic_library(CONFIG.num_groups, 5, num_materials=7)
    library2 = replace(library, spectrum_fraction=2.5 * library.spectrum_fraction,
                       fine_weight=2.5 * library.fine_weight)
    r2 = make_solver(library=library2).solve_view(rho, GRID, 0.0)
    r = base_result
    assert r2.source_iteration_result.num_iterations == r.source_iteration_result.num_iterations
    for name in ("primary", "scatter"):
        a, b = getattr(r2, name)["signal"], getattr(r, name)["signal"]
        assert torch.allclose(a, 2.5 * b, rtol=1e-4, atol=1e-6 * float(b.max())), name
    assert torch.allclose(r2.scatter_fraction, r.scatter_fraction, rtol=1e-4, atol=1e-6)


# ---- chunking ------------------------------------------------------------------------------------------------

def test_independent_of_direction_chunk(rho, base_result):
    r1 = make_solver(replace(CONFIG, direction_chunk=1)).solve_view(rho, GRID, 0.0, keep_phi_total=True)
    r = base_result
    assert r1.source_iteration_result.num_iterations == r.source_iteration_result.num_iterations
    assert _rel(r1.phi_total, r.phi_total) < 1e-5
    for name in ("primary", "scatter"):
        assert _rel(getattr(r1, name)["signal"], getattr(r, name)["signal"]) < 1e-5, name
    assert _rel(r1.scatter_signal, r.scatter_signal) < 1e-5


def test_image_direction_chunk(rho, monkeypatch):
    """With the fused first-collision kernels the scatter image takes as many directions per chunk as fit
    IMAGE_CHUNK_BUDGET_BYTES, between config.direction_chunk and IMAGE_CHUNK_MAX_DIRECTIONS; otherwise
    config.direction_chunk. The image does not depend on the chunk."""
    import citrus.scatter_solver.scatter_solver as scatter_solver_module
    from citrus.scatter_solver.kernels import uncollided_kernels

    s = make_solver()
    f = s.material_fields(rho, GRID)
    chunk = s.image_direction_chunk(f)
    if not (DEVICE.type == "cuda" and uncollided_kernels.kernel_usable(DEVICE)[0]):
        assert chunk == CONFIG.direction_chunk
        pytest.skip("the fused first-collision kernels are unusable here")
    assert chunk == scatter_solver_module.IMAGE_CHUNK_MAX_DIRECTIONS > CONFIG.direction_chunk  # a small problem
    assert max(int(c.shape[0]) for c in s.direction_context(f, 0.0).omega_chunks) == chunk
    phi_uncollided = s.uncollided_fluence(f, 0.0)
    T = s.multiple_collision_source_moments(f, 0.01 * torch.rand(s.num_groups, (s.L + 1) ** 2, 4, *f.grid_coarse.shape,
                                                                 device=DEVICE))
    image = s.scatter_image(f, 0.0, phi_uncollided, T)
    monkeypatch.setattr(scatter_solver_module, "IMAGE_CHUNK_BUDGET_BYTES", 1)  # nothing fits: the configured chunk
    small = make_solver()
    assert small.image_direction_chunk(f) == CONFIG.direction_chunk
    assert max(int(c.shape[0]) for c in small.direction_context(f, 0.0).omega_chunks) == CONFIG.direction_chunk
    small_image = small.scatter_image(f, 0.0, phi_uncollided, T)
    for k in IMAGE_KEYS:
        assert _rel(image[k], small_image[k]) < 1e-6, k


# ---- anti-scatter grid ----------------------------------------------------------------------------------------

def test_grid_lowers_scatter_fraction(rho, base_result):
    r_g = make_solver(grid_response=focused_1d_grid()).solve_view(rho, GRID, 0.0)
    assert bool((r_g.scatter_fraction < base_result.scatter_fraction).all())
    # the fluences do not depend on the detector side
    assert torch.allclose(r_g.scatter["fluence"], base_result.scatter["fluence"], rtol=1e-5)


# ---- config ---------------------------------------------------------------------------------------------------

def test_config_validation():
    SolverConfig().validate()
    for bad in (dict(num_polar=7), dict(num_azimuthal=14), dict(transport_cell_cm=0.0),
                dict(quadrature_point_spacing_cm=0.0),
                dict(uncollided_trace_grid="medium"), dict(source_iteration_tolerance=0.0),
                dict(direction_chunk=0), dict(dtype="float16"), dict(group_edges_kev=(100.0, 10.0))):
        with pytest.raises(ValueError):
            replace(SolverConfig(), **bad).validate()


def test_fc_sampling_conservative(rho):
    """The first-collision source of the sweep is corrected (uncollided.first_collision_sampling_correction, l0 + l1):
    the sweep-quadrature integral of the restricted first-collision source equals the exact l = 0 transfer integral
    (restricted to the supervoxels, all four linear-discontinuous moments) to 1e-6, which the uncorrected point samples
    miss for a forward-peaked (HG g = 0.7) kernel; the l = 1 moment about Omega_UC is matched on the fine grid wherever
    the non-negativity clamp of the factor is inactive, and the source stays non-negative."""
    library = peaked_library(CONFIG.num_groups, 5, 0.7)
    geometry = wlab_geometry()
    solver = ScatterSolver(library, CONFIG, geometry, SiddonProjectors(), csi_response(), no_grid(), device=DEVICE,
                           dtype=torch.float64)
    f = solver.material_fields(rho, GRID)
    phi_uncollided = solver.uncollided_fluence(f, 0.0)
    context = solver.sweep_context(f.grid_coarse)
    index = torch.arange(context.num_directions, device=DEVICE)
    w = context.quadrature.weight.to(DEVICE, torch.float64)
    directions = context.quadrature.omega.to(DEVICE, torch.float64)
    Q = f.grid_quadrature
    # uncorrected samples
    first_collision_point = first_collision_source_at_directions(solver.library, f.rho_quadrature, phi_uncollided, Q,
                                                                 geometry, 0.0, directions)

    E = first_collision_exact_moments(solver.library, f.rho_quadrature, phi_uncollided, l_max=1)  # [2, G, Rq, Cq, Sq]
    exact = restrict_to_linear_discontinuous_moments(E[0], Q, f.q)
    scale = float(exact[:, 0].abs().max())
    # [G, 4, Rc, Cc, Sc]
    got = torch.einsum("d,dg...->g...", w, solver.first_collision_moment_source(f, 0.0, phi_uncollided, context)(index))
    pt = torch.einsum("d,dg...->g...", w, restrict_to_linear_discontinuous_moments(first_collision_point, Q, f.q))
    assert float((got - exact).abs().max()) < 1e-6 * scale
    assert float((got[:, 0].sum() - exact[:, 0].sum()).abs()) < 1e-6 * float(exact[:, 0].sum())
    assert float((pt[:, 0] - exact[:, 0]).abs().max()) > 1e-3 * scale

    corr = solver.first_collision_sampling_correction(f, 0.0, phi_uncollided, context)
    first_collision = corr.apply(first_collision_point, directions)
    assert float(first_collision.min()) >= 0.0
    mu = torch.einsum("rcsx,kx->krcs", primary_directions_to_voxels(Q, geometry, 0.0, DEVICE, torch.float64),
                      directions)
    J = torch.einsum("k,krcs,kgrcs->grcs", w, mu, first_collision)
    free = corr.beta.abs() < 1
    assert float(free.double().mean()) > 0.5
    e1 = float(E[1].abs().max())
    assert float((J - E[1])[free].abs().max()) < 1e-6 * e1


@pytest.mark.skipif(DEVICE.type != "cuda", reason="float32 CUDA accumulation")
def test_fc_sampling_float32_accumulation(rho, monkeypatch):
    """The sampling-correction sums are formed by float32 matmuls (blocks of at most _FIRST_COLLISION_SUM_MAX_DIRECTIONS
    directions) and accumulated across blocks in float64 (uncollided.first_collision_quadrature_sums). On float32 CUDA
    the factors differ from an all-float64 computation by < 1e-5 relative (scale relative; beta absolute, i.e. relative
    to the unit factor 1 + beta mu): (a) the same float32 source with every sum in float64 and (b) the whole pass in
    float64 (float64 inputs, PyTorch first-collision path). Splitting the pass into several blocks changes nothing
    beyond rounding."""
    from citrus.scatter_solver import uncollided as uncollided_module
    from citrus.scatter_solver.uncollided import (first_collision_correction_factors, first_collision_quadrature_sums,
                                                  first_collision_sampling_correction)

    library = peaked_library(CONFIG.num_groups, 5, 0.7)
    geometry = wlab_geometry()
    solver = ScatterSolver(library, CONFIG, geometry, SiddonProjectors(), csi_response(), no_grid(), device=DEVICE,
                           dtype=torch.float32)
    f = solver.material_fields(rho, GRID)
    phi_uncollided = solver.uncollided_fluence(f, 0.0)
    context = solver.sweep_context(f.grid_coarse)
    directions32 = context.quadrature.omega.to(DEVICE, torch.float32)
    w64 = context.quadrature.weight.to(DEVICE, torch.float64)
    corr = solver.first_collision_sampling_correction(f, 0.0, phi_uncollided, context)
    assert corr.scale.dtype == torch.float32 and corr.beta.dtype == torch.float32
    E = first_collision_exact_moments(solver.library, f.rho_quadrature.double(), phi_uncollided.double(), l_max=1)
    Q = f.grid_quadrature

    # (a) the same float32 first-collision source, all sums in float64
    first_collision = first_collision_source_at_directions(solver.library, f.rho_quadrature, phi_uncollided, Q,
                                                           geometry, 0.0, directions32).double()
    mu = torch.einsum("rcsx,kx->krcs", primary_directions_to_voxels(Q, geometry, 0.0, DEVICE,
                                                                    torch.float32), directions32).double()
    I = torch.stack([torch.einsum("k,kg...->g...", w64, first_collision), torch.einsum("k,k...,kg...->g...", w64, mu,
                                                                                       first_collision),
                     torch.einsum("k,k...,kg...->g...", w64, mu * mu, first_collision)])
    reference_a = first_collision_correction_factors(I, E)
    # (b) everything in float64
    c64 = first_collision_sampling_correction(solver.library, f.rho_quadrature.double(), phi_uncollided.double(), Q,
                                              geometry, 0.0, context.quadrature.omega.to(DEVICE, torch.float64), w64,
                                              batch=CONFIG.direction_chunk)
    reference_b = (c64.scale, c64.beta)
    for name, (s_reference, b_reference) in (("float64 sums", reference_a), ("all float64", reference_b)):
        ds = float(((corr.scale.double() - s_reference).abs() / s_reference.abs()).max())
        db = float((corr.beta.double() - b_reference).abs().max())
        assert ds < 1e-5 and db < 1e-5, (name, ds, db)
        assert float(b_reference.abs().max()) > 1e-2  # the correction is not trivial

    # several float32 blocks (block of 16 directions, first-collision batches of 5 that do not divide it) == one block
    I1 = first_collision_quadrature_sums(solver.library, f.rho_quadrature, phi_uncollided, Q, geometry, 0.0,
                                         directions32, w64,
                                         batch=8)
    monkeypatch.setattr(uncollided_module, "_FIRST_COLLISION_SUM_MAX_DIRECTIONS", 16)
    I2 = first_collision_quadrature_sums(solver.library, f.rho_quadrature, phi_uncollided, Q, geometry, 0.0,
                                         directions32, w64,
                                         batch=5)
    assert I1.dtype == I2.dtype == torch.float64
    assert float((I1 - I2).abs().max()) < 1e-6 * float(I1[0].abs().max())
    assert float((I1 - I).abs().max()) < 1e-6 * float(I[0].abs().max())


# ---- real library (slow) ---------------------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("CITRUS_SLOW_TESTS") != "1", reason="set CITRUS_SLOW_TESTS=1 (builds the "
                    "library, cached in the library cache)")
def test_real_library_wlab(rho):
    from citrus.scatter_solver.scatter_solver import build_solver_library

    config = replace(CONFIG, num_groups=6, source_iteration_tolerance=1e-3)
    library = build_solver_library(config, SPECTRUM_125KVP)
    assert library.num_groups == 6
    r = make_solver(config, library=library).solve_view(rho, GRID, 0.0)
    iteration_result = r.source_iteration_result
    assert iteration_result.converged and 3 < iteration_result.num_iterations < 30, iteration_result.history
    for d in (r.primary, r.scatter):
        assert float(d["signal"].min()) >= 0 and float(d["signal"].sum()) > 0
    sf = r.scatter_fraction
    assert float(sf.min()) > 0 and float(sf.max()) < 1
    # behind the 20 cm water slab (central row and column) the no-grid SF is large but below 1
    c = sf[sf.shape[0] // 2, sf.shape[1] // 2]
    assert 0.2 < float(c) < 0.9, float(c)


# ---- transport cells: the transport grid and the quadrature points ----------------------------------------------

GRID4 = Grid((12, 16, 20), 2.0)  # divisible by 4


@pytest.fixture(scope="module")
def rho4() -> torch.Tensor:
    return wlab_phantom(GRID4, device=DEVICE, dtype=torch.float32)


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_transport_cells(rho4, factor):
    """solve_view with transport cells of 1, 2, 4 fine voxels (2, 4, 8 cm) and q = cells / voxel (the quadrature grid
    is the fine grid; uncollided fluence traced through the fine grid, so that only the transport grid changes; the
    coarse trace is tested in test_quadrature_grid.py): all images are finite and non-negative;
    with 2 cm cells the first-order image (the solver's scatter image with a zero multiple-collision source) is the
    single-grid collapsed-cone image of the first-collision source (scatter_image.scatter_image_from_source, same
    footprint); coarser transport grids change it by a bounded amount."""
    from citrus.scatter_solver.scatter_image import scatter_image_from_source

    config = replace(CONFIG, transport_cell_cm=2.0 * factor, quadrature_point_spacing_cm=2.0,
                     uncollided_trace_grid="fine")
    s = make_solver(config)
    r = s.solve_view(rho4, GRID4, 0.0)
    assert r.source_iteration_result.converged
    for d in (r.primary, r.scatter):
        for k in IMAGE_KEYS:
            assert bool(torch.isfinite(d[k]).all()) and float(d[k].min()) >= 0.0, k
        assert float(d["signal"].sum()) > 0
    assert float(r.scatter_fraction.min()) > 0 and float(r.scatter_fraction.max()) < 1
    f = s.material_fields(rho4, GRID4)
    first = s.scatter_image(f, 0.0, s.uncollided_fluence(f, 0.0), _zero_multiple_collision(s, f))
    # the single-grid image of the first-collision source on the fine grid, attenuated with the fine mu_t
    s1 = make_solver(replace(config, transport_cell_cm=2.0, quadrature_point_spacing_cm=2.0))
    f1 = s1.material_fields(rho4, GRID4)
    phi_uncollided = s1.uncollided_fluence(f1, 0.0)
    fine = scatter_image_from_source(
        lambda directions: first_collision_source_at_directions(s1.library, f1.rho_fine, phi_uncollided, GRID4,
                                                                s1.geometry, 0.0, directions), GRID4, s1.geometry, 0.0,
        s1.cone_quadrature(GRID4, 0.0), f1.mu_t_coarse, s1.projectors, s1.detector, s1.grid_response, s1.library,
        chunk=config.direction_chunk)
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
    phi_uncollided = s.uncollided_fluence(f, 0.0)
    T = _zero_multiple_collision(s, f)
    got = s.scatter_image(f, 0.0, phi_uncollided, T)
    q = s.cone_quadrature(GRID, 0.0)
    exp = combined_scatter_image(s.library, f.rho_quadrature, phi_uncollided, f.grid_quadrature, f.q, T, s.L,
                                 s.geometry, 0.0, q, f.mu_t_coarse, s.projectors, s.detector, s.grid_response,
                                 chunk=CONFIG.direction_chunk, grid_coarse=f.grid_coarse)
    assert torch.allclose(got["signal"], exp["signal"], rtol=1e-6)
    more = combined_scatter_image(s.library, f.rho_quadrature, phi_uncollided, f.grid_quadrature, f.q, T, s.L,
                                  s.geometry, 0.0, q,
                                  1.2 * f.mu_t_coarse, s.projectors, s.detector, s.grid_response,
                                  chunk=CONFIG.direction_chunk)
    assert float(got["signal"].sum()) > float(more["signal"].sum())


# ---- direction context cache ----------------------------------------------------------------------------------

def test_direction_context_cache(rho, base_result, monkeypatch):
    """One direction context per view, reused by later solves and by the CITRUS step (same object), LRU-bounded,
    dropped by clear_caches; the images do not change."""
    import citrus.scatter_solver.scatter_solver as scatter_solver_module
    from citrus.citrus import CitrusState, citrus_step

    built = []
    real = scatter_solver_module.make_direction_context

    def counting(*a, **k):
        context = real(*a, **k)
        built.append(context)
        return context

    monkeypatch.setattr(scatter_solver_module, "make_direction_context", counting)
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
        a, b, reference = r2.scatter[k], r1.scatter[k], base_result.scatter[k]
        assert _rel(a, b) < 1e-6 and _rel(a, reference) < 1e-6, k
    # no cache: identical images, nothing kept
    s0 = make_solver(direction_context_cache_size=0)
    r0 = s0.solve_view(rho, GRID, 0.0)
    assert len(s0._direction_contexts) == 0
    assert _rel(r0.scatter["signal"], r1.scatter["signal"]) < 1e-6
    # LRU bound: one entry, the newest view kept
    s1 = make_solver(direction_context_cache_size=1)
    f1 = s1.material_fields(rho, GRID)
    a0 = s1.direction_context(f1, 0.0)
    a1 = s1.direction_context(f1, 0.5)
    assert len(s1._direction_contexts) == 1 and s1.direction_context(f1, 0.5) is a1
    assert s1.direction_context(f1, 0.0) is not a0
    # clear_caches drops everything
    s.clear_caches()
    assert not (s._direction_contexts or s._cone_quadratures or s._sweep_contexts)
    assert s.direction_context(fields, 0.0) is not c1
