"""Tests of the CITRUS per-view scatter step and its operators (citrus/citrus.py): with fixed densities N steps are
N source iterations."""

from __future__ import annotations

import pytest
import torch

from citrus.citrus import CitrusState, G1, G1to2, G_gt2, G_step, citrus_step, project_scatter
from citrus.scatter_solver.config import SolverConfig
from citrus.scatter_solver.cross_sections import synthetic_library
from citrus.scatter_solver.detector import csi_response, no_grid
from citrus.scatter_solver.geometry import Grid
from citrus.scatter_solver.ray_tracing import SiddonProjectors
from citrus.scatter_solver.scatter_solver import ScatterSolver
from citrus.scatter_solver.source_iteration import source_iteration
from citrus.utilities import wlab_geometry, wlab_phantom

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID = Grid((10, 16, 20), 2.0)
CFG = SolverConfig(num_groups=4, legendre_order=3, num_polar=4, num_azimuthal=8, cone_num_polar=8,
                   cone_num_azimuthal=16, transport_cell_cm=4.0, si_tol=1e-4, si_max_iterations=40)
ANGLE = 0.4


def _solver(cfg: SolverConfig = CFG) -> ScatterSolver:
    return ScatterSolver(synthetic_library(cfg.num_groups, 5, num_materials=7), cfg, wlab_geometry(), SiddonProjectors(),
                         csi_response(), no_grid(), device=DEV, dtype=torch.float32)


@pytest.fixture(scope="module")
def rho() -> torch.Tensor:
    return wlab_phantom(GRID, device=DEV, dtype=torch.float32)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def test_citrus_steps_reproduce_source_iteration(rho):
    """Fixed point: with fixed densities N CITRUS steps are N source iterations (accumulator, scatter image, step
    history)."""
    solver = _solver()
    ref = solver.solve_view(rho, GRID, ANGLE, keep_phi_total=True)
    n = ref.si.num_iterations
    assert ref.si.converged and n > 3
    state = CitrusState()
    scatter = primary = None
    for k in range(n):
        scatter, primary, new_state = citrus_step(solver, state, rho, GRID, ANGLE)
        assert new_state.num_steps == k + 1 and state.num_steps == k  # input state untouched
        state = new_state
    # with fixed densities, N steps ARE N source iterations
    assert _rel(state.Phi_gt1, ref.phi_total) < 2e-5
    assert _rel(scatter, ref.scatter_signal) < 2e-5
    assert torch.allclose(primary, ref.primary["signal"], rtol=1e-6)
    # the step-to-step change decays like the SI history (same contraction)
    h = state.history
    assert h[0] == 1.0 and h[-1] < CFG.si_tol and all(h[i + 1] < h[i] for i in range(1, len(h) - 1)), h
    assert max(abs(a - b) for a, b in zip(h, ref.si.history)) < 1e-3
    # a further step changes the scatter image by less than the SI tolerance
    s_next, _, _ = citrus_step(solver, state, rho, GRID, ANGLE)
    assert _rel(s_next, scatter) < CFG.si_tol


def test_partial_steps_match_truncated_si(rho):
    """After k < N steps the accumulator equals the SI truncated at k iterations (exactly the same recursion)."""
    solver = _solver()
    f = solver.material_fields(rho, GRID)
    phi_uc = G1(solver, rho, GRID, ANGLE, f)
    ctx = solver.sweep_context(f.grid_coarse)
    k = 3
    si = source_iteration(ctx, f.mu_t_coarse, f.mu_s_l_coarse, solver.fc_moment_source(f, ANGLE, phi_uc),
                          tol=1e-12, max_iterations=k, direction_chunk=CFG.direction_chunk)
    state = CitrusState()
    for _ in range(k):
        _, _, state = citrus_step(solver, state, rho, GRID, ANGLE)
    assert _rel(state.Phi_gt1, si.phi_total) < 2e-5


def test_individual_operators(rho):
    solver = _solver()
    f = solver.material_fields(rho, GRID)
    phi_uc = G1(solver, rho, GRID, ANGLE)
    assert phi_uc.shape == (4, *GRID.shape) and torch.equal(phi_uc, solver.uncollided(f, ANGLE))
    phi2 = G1to2(solver, rho, GRID, ANGLE, phi_uc)
    assert phi2.shape == (4, 16, 4, 5, 8, 10)
    # G1to2 is the first SI iterate
    seen = {}
    ctx = solver.sweep_context(f.grid_coarse)
    source_iteration(ctx, f.mu_t_coarse, f.mu_s_l_coarse, solver.fc_moment_source(f, ANGLE, phi_uc), tol=1e-12,
                     max_iterations=1, callback=lambda n, p, r: seen.setdefault(n, p.clone()))
    assert _rel(phi2, seen[1]) < 1e-6
    zero = G_gt2(solver, rho, GRID, ANGLE, None)
    assert zero.shape == phi2.shape and float(zero.abs().max()) == 0.0
    phi3 = G_gt2(solver, rho, GRID, ANGLE, phi2)
    assert phi3.shape == phi2.shape
    # linear: G_gt2(2 phi) = 2 G_gt2(phi); collided fluence decreases with collision order here
    assert _rel(G_gt2(solver, rho, GRID, ANGLE, 2 * phi2), 2 * phi3) < 1e-5
    assert float(phi3[:, 0, 0].norm()) < float(phi2[:, 0, 0].norm())
    with pytest.raises(ValueError):
        G_gt2(solver, rho, GRID, ANGLE, phi2[:, :9])
    # the merged operator (one sweep of the summed sources) equals the two separate sweeps (linearity)
    assert _rel(G_step(solver, rho, GRID, ANGLE, phi_uc, None), phi2) < 1e-6
    assert _rel(G_step(solver, rho, GRID, ANGLE, phi_uc, phi2), phi2 + phi3) < 1e-5
    with pytest.raises(ValueError):
        G_step(solver, rho, GRID, ANGLE, phi_uc, phi2[:, :9])
    # the one-pass scatter image of the accumulator: non-negative, larger than the first-order image alone
    tot = project_scatter(solver, rho, GRID, ANGLE, phi_uc, phi2 + phi3)
    assert set(tot) == {"fluence", "planar_fluence", "signal_per_group", "signal"} and float(tot["signal"].min()) >= 0
    first = project_scatter(solver, rho, GRID, ANGLE, phi_uc, torch.zeros_like(phi2))
    assert float(tot["signal"].sum()) > float(first["signal"].sum()) > 0
    # ... and with the converged accumulator it is the solver's scatter image
    ref = solver.solve_view(rho, GRID, ANGLE, keep_phi_total=True)
    tot_ref = project_scatter(solver, rho, GRID, ANGLE, phi_uc, ref.phi_total)
    for k in tot_ref:
        assert _rel(tot_ref[k], ref.scatter[k]) < 1e-6, k
    with pytest.raises(ValueError):
        citrus_step(solver, CitrusState(angle=0.0), rho, GRID, ANGLE)
    # precomputed fields (shared by the views of a CITRUS outer iteration) give the same step
    s_a, p_a, st_a = citrus_step(solver, CitrusState(), rho, GRID, ANGLE)
    s_b, p_b, st_b = citrus_step(solver, CitrusState(), rho, GRID, ANGLE, fields=f)
    assert _rel(s_b, s_a) < 1e-6 and torch.equal(p_b, p_a) and _rel(st_b.Phi_gt1, st_a.Phi_gt1) < 1e-6
