"""Tests of the CITRUS per-view scatter step and its operators (citrus/citrus.py): with fixed densities N steps are
N source iterations."""

from __future__ import annotations

import pytest
import torch

from citrus.citrus import (CitrusState, accumulator_update_operator, citrus_step, first_collision_operator,
                           multiple_collision_operator, project_scatter, uncollided_operator)
from citrus.scatter_solver.config import SolverConfig
from citrus.scatter_solver.cross_sections import synthetic_library
from citrus.scatter_solver.detector import csi_response, no_grid
from citrus.scatter_solver.geometry import Grid
from citrus.scatter_solver.ray_tracing import SiddonProjectors
from citrus.scatter_solver.scatter_solver import ScatterSolver
from citrus.scatter_solver.source_iteration import source_iteration
from citrus.utilities import wlab_geometry, wlab_phantom

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID = Grid((10, 16, 20), 2.0)
CONFIG = SolverConfig(num_groups=4, legendre_order=3, num_polar=4, num_azimuthal=8, cone_num_polar=8,
                      cone_num_azimuthal=16, transport_cell_cm=4.0, source_iteration_tolerance=1e-4,
                      max_source_iterations=40)
ANGLE = 0.4


def _solver(config: SolverConfig = CONFIG) -> ScatterSolver:
    return ScatterSolver(synthetic_library(config.num_groups, 5, num_materials=7), config, wlab_geometry(),
                         SiddonProjectors(),
                         csi_response(), no_grid(), device=DEVICE, dtype=torch.float32)


@pytest.fixture(scope="module")
def rho() -> torch.Tensor:
    return wlab_phantom(GRID, device=DEVICE, dtype=torch.float32)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def test_citrus_steps_reproduce_source_iteration(rho):
    """Fixed point: with fixed densities N CITRUS steps are N source iterations (accumulator, scatter image, step
    history)."""
    solver = _solver()
    reference = solver.solve_view(rho, GRID, ANGLE, keep_phi_total=True)
    n = reference.source_iteration_result.num_iterations
    assert reference.source_iteration_result.converged and n > 3
    state = CitrusState()
    scatter = primary = None
    for k in range(n):
        scatter, primary, new_state = citrus_step(solver, state, rho, GRID, ANGLE)
        assert new_state.num_steps == k + 1 and state.num_steps == k  # input state untouched
        state = new_state
    # with fixed densities, N steps ARE N source iterations
    assert _rel(state.collided_moments, reference.phi_total) < 2e-5
    assert _rel(scatter, reference.scatter_signal) < 2e-5
    assert torch.allclose(primary, reference.primary["signal"], rtol=1e-6)
    # the step-to-step change decays like the source-iteration history (same contraction)
    h = state.history
    assert h[0] == 1.0 and h[-1] < CONFIG.source_iteration_tolerance and all(h[i + 1] < h[i] for i in range(1, len(h)
                                                                                                            - 1)), h
    assert max(abs(a - b) for a, b in zip(h, reference.source_iteration_result.history)) < 1e-3
    # a further step changes the scatter image by less than the source-iteration tolerance
    s_next, _, _ = citrus_step(solver, state, rho, GRID, ANGLE)
    assert _rel(s_next, scatter) < CONFIG.source_iteration_tolerance


def test_partial_steps_match_truncated_si(rho):
    """After k < N steps the accumulator equals the source iteration truncated at k iterations (exactly the same
    recursion)."""
    solver = _solver()
    f = solver.material_fields(rho, GRID)
    phi_uncollided = uncollided_operator(solver, rho, GRID, ANGLE, f)
    context = solver.sweep_context(f.grid_coarse)
    k = 3
    iteration_result = source_iteration(context, f.mu_t_coarse, f.mu_s_l_coarse,
                                        solver.first_collision_moment_source(f, ANGLE,
                                                                             phi_uncollided),
                                        tolerance=1e-12, max_iterations=k, direction_chunk=CONFIG.direction_chunk)
    state = CitrusState()
    for _ in range(k):
        _, _, state = citrus_step(solver, state, rho, GRID, ANGLE)
    assert _rel(state.collided_moments, iteration_result.phi_total) < 2e-5


def test_individual_operators(rho):
    solver = _solver()
    f = solver.material_fields(rho, GRID)
    phi_uncollided = uncollided_operator(solver, rho, GRID, ANGLE)
    assert phi_uncollided.shape == (4, *GRID.shape) and torch.equal(phi_uncollided, solver.uncollided_fluence(f, ANGLE))
    phi2 = first_collision_operator(solver, rho, GRID, ANGLE, phi_uncollided)
    assert phi2.shape == (4, 16, 4, 5, 8, 10)
    # first_collision_operator is the first source-iteration iterate
    seen = {}
    context = solver.sweep_context(f.grid_coarse)
    source_iteration(context, f.mu_t_coarse, f.mu_s_l_coarse,
                     solver.first_collision_moment_source(f, ANGLE, phi_uncollided), tolerance=1e-12,
                     max_iterations=1, callback=lambda n, p, r: seen.setdefault(n, p.clone()))
    assert _rel(phi2, seen[1]) < 1e-6
    zero = multiple_collision_operator(solver, rho, GRID, ANGLE, None)
    assert zero.shape == phi2.shape and float(zero.abs().max()) == 0.0
    phi3 = multiple_collision_operator(solver, rho, GRID, ANGLE, phi2)
    assert phi3.shape == phi2.shape
    # linear: multiple_collision_operator(2 phi) = 2 multiple_collision_operator(phi); collided fluence decreases with
    # collision order here
    assert _rel(multiple_collision_operator(solver, rho, GRID, ANGLE, 2 * phi2), 2 * phi3) < 1e-5
    assert float(phi3[:, 0, 0].norm()) < float(phi2[:, 0, 0].norm())
    with pytest.raises(ValueError):
        multiple_collision_operator(solver, rho, GRID, ANGLE, phi2[:, :9])
    # the merged operator (one sweep of the summed sources) equals the two separate sweeps (linearity)
    assert _rel(accumulator_update_operator(solver, rho, GRID, ANGLE, phi_uncollided, None), phi2) < 1e-6
    assert _rel(accumulator_update_operator(solver, rho, GRID, ANGLE, phi_uncollided, phi2), phi2 + phi3) < 1e-5
    with pytest.raises(ValueError):
        accumulator_update_operator(solver, rho, GRID, ANGLE, phi_uncollided, phi2[:, :9])
    # the one-pass scatter image of the accumulator: non-negative, larger than the first-order image alone
    tot = project_scatter(solver, rho, GRID, ANGLE, phi_uncollided, phi2 + phi3)
    assert set(tot) == {"fluence", "planar_fluence", "signal_per_group", "signal"} and float(tot["signal"].min()) >= 0
    first = project_scatter(solver, rho, GRID, ANGLE, phi_uncollided, torch.zeros_like(phi2))
    assert float(tot["signal"].sum()) > float(first["signal"].sum()) > 0
    # ... and with the converged accumulator it is the solver's scatter image
    reference = solver.solve_view(rho, GRID, ANGLE, keep_phi_total=True)
    total_reference = project_scatter(solver, rho, GRID, ANGLE, phi_uncollided, reference.phi_total)
    for k in total_reference:
        assert _rel(total_reference[k], reference.scatter[k]) < 1e-6, k
    with pytest.raises(ValueError):
        citrus_step(solver, CitrusState(angle=0.0), rho, GRID, ANGLE)
    # precomputed fields (shared by the views of a CITRUS outer iteration) give the same step
    s_a, p_a, st_a = citrus_step(solver, CitrusState(), rho, GRID, ANGLE)
    s_b, p_b, st_b = citrus_step(solver, CitrusState(), rho, GRID, ANGLE, fields=f)
    assert _rel(s_b, s_a) < 1e-6 and torch.equal(p_b, p_a) and _rel(st_b.collided_moments, st_a.collided_moments) < 1e-6
