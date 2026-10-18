"""Adaptive angular transfer table (version 0.6.0, `SolverConfig.angular_table_grid`): the node map
(`cross_sections.angular_table_position` / `angular_table_nodes`), its interpolation error against the exact transfer
of the real library at 125 kVp and 6 MV (uniform vs adaptive at the same node count), the l = 0 integral of the
interpolated table against the group cross section, and the fused first-collision kernel against the PyTorch path on
adaptive tables. The uniform option reproducing the library of version 0.5.1 is tested in
tests/test_material_catalogue.py (with the regression anchors there and in tests/test_quadrature_grid.py)."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from citrus.scatter_solver.config import SolverConfig
from citrus.scatter_solver.cross_sections import (ANGULAR_MAP_KNOTS, angular_table_nodes, angular_table_position,
                                                  material_table, synthetic_angular_map, synthetic_library,
                                                  transfer_eta_at)

SPECTRUM_6MV = Path(__file__).parent / "data" / "spectrum_6mv.txt"  # Mohan et al. (1985), the WLAB 6 MV validation


# ---- the map ------------------------------------------------------------------------------------------------------

def test_nodes_are_integer_positions_and_uniform_is_unchanged():
    """The nodes of an adaptive table sit at integer positions of the lookup (both ends included), ascend from -1 to
    1, and are dense near eta = 1; without a map the lookup is the uniform arithmetic of version 0.5."""
    eta_map = synthetic_angular_map()
    assert eta_map.shape == (ANGULAR_MAP_KNOTS,) and float(eta_map[0]) == 1.0 and float(eta_map[-1]) == 0.0
    nodes = angular_table_nodes(2001, eta_map)
    assert float(nodes[0]) == -1.0 and float(nodes[-1]) == 1.0 and bool((nodes.diff() > 0).all())
    assert float(nodes.diff()[-1]) < 0.05 * 2.0 / 2000  # far denser than a uniform table at the forward end
    i0, w = angular_table_position(nodes, 2001, eta_map)
    assert float((i0.double() + w - torch.arange(2001)).abs().max()) < 1e-6
    eta = torch.linspace(-1.2, 1.2, 1001, dtype=torch.float64)
    i0, w = angular_table_position(eta, 401)
    pos = (eta.clamp(-1, 1) + 1) / 2 * 400
    assert torch.equal(i0, pos.floor().long().clamp(0, 399)) and torch.equal(w, (pos - i0).clamp(0, 1))
    with pytest.raises(ValueError):
        angular_table_nodes(11, eta_map.flip(0))


def test_synthetic_adaptive_library_interpolates_its_kernel():
    """The synthetic kernel a (1 + b eta) / 4 pi tabulated on the adaptive nodes is reproduced to the error of linear
    interpolation in q between nodes (small: the nodes are dense)."""
    library = synthetic_library(4, num_eta=401, angular_table_grid="adaptive")
    eta = torch.linspace(-1.0, 1.0, 3001, dtype=torch.float64)
    rho = torch.ones(2, eta.numel(), dtype=torch.float64)
    values = transfer_eta_at(library, rho, eta)
    exact = transfer_eta_at(synthetic_library(4, num_eta=401), rho, eta)  # uniform: exact for a linear kernel
    assert float((values - exact).abs().max() / exact.abs().max()) < 1e-3


# ---- the real library ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def library_module():
    cross_section_library = pytest.importorskip("citrus.scatter_solver.cross_section_library")
    try:
        cross_section_library.resolve_backend()
    except Exception as exc:
        pytest.skip(f"photon data unavailable ({exc})")
    return cross_section_library


# (spectrum, groups, fine bin width keV): water only and coarse fine bins keep the builds to seconds
CASES = {"125kvp": (None, 6, 1.0), "6mv": (SPECTRUM_6MV, 12, 20.0)}


@pytest.fixture(scope="module")
def tables(library_module):
    """{case: (uniform library, adaptive library, test points eta [n], exact transfer [M, G, G, n])} for water with
    2001 nodes; the test points are the midpoints between the nodes of both tables and 4001 points uniform in
    q = sin(theta / 2)."""
    backend = library_module.resolve_backend()
    out = {}
    for case, (path, groups, fine) in CASES.items():
        spectrum = None if path is None else library_module.load_spectrum_txt(str(path))
        config = SolverConfig(num_groups=groups, legendre_order=3, group_edges_method="equal_fluence")
        libraries = [library_module.compute_library(replace(config, angular_table_grid=grid), spectrum, backend,
                                                    fine_bin_width_kev=fine, materials=material_table(("water",)))[0]
                     for grid in ("uniform", "adaptive")]
        q = np.linspace(0.0, 1.0, 4001)
        points = [1.0 - 2.0 * q * q] + [0.5 * (lib.eta_grid.numpy()[1:] + lib.eta_grid.numpy()[:-1])
                                        for lib in libraries]
        eta = np.unique(np.concatenate(points))
        exact = library_module.transfer_at_eta(config, eta, spectrum, backend, fine_bin_width_kev=fine,
                                               materials=material_table(("water",)))
        out[case] = (libraries[0], libraries[1], eta, exact)
    return out


def _max_relative_error(library, eta: np.ndarray, exact: np.ndarray) -> float:
    """max over the (g, g') transfers of max |interpolated - exact| / max |exact| (water)."""
    values = transfer_eta_at(library, torch.ones(1, eta.size, dtype=torch.float64),
                             torch.as_tensor(eta)).numpy()  # [G, G, n]
    worst = 0.0
    G = values.shape[0]
    for gi in range(G):
        for go in range(gi, G):
            scale = float(np.abs(exact[0, gi, go]).max())
            if scale > 0:
                worst = max(worst, float(np.abs(values[gi, go] - exact[0, gi, go]).max()) / scale)
    return worst


@pytest.mark.parametrize("case", list(CASES))
def test_adaptive_table_interpolates_the_kernel(tables, case):
    """At the same 2001 nodes the adaptive table interpolates the exact transfer (Compton with the kinematic group
    fractions plus Rayleigh) far better than the uniform one: at 6 MV the uniform table cannot represent the forward
    peaks (a few 1e-7 wide in 1 - eta; maximum error of order 1), the adaptive one resolves them (below 1 %); at
    125 kVp both are fine and the adaptive one is better still."""
    uniform, adaptive, eta, exact = tables[case]
    assert uniform.eta_map is None and adaptive.eta_map is not None
    error_uniform = _max_relative_error(uniform, eta, exact)
    error_adaptive = _max_relative_error(adaptive, eta, exact)
    if case == "6mv":
        assert error_uniform > 0.3 and error_adaptive < 0.01 and error_adaptive < error_uniform / 50, (
            error_uniform, error_adaptive)
    else:
        assert error_adaptive < 1e-3 and error_adaptive < error_uniform, (error_uniform, error_adaptive)


@pytest.mark.parametrize("case", list(CASES))
def test_first_collision_source_integrates_to_the_group_cross_section(tables, case):
    """The l = 0 moment of the interpolated table over a fine direction set (400001 points uniform in q, d eta = 4 q dq;
    azimuthal symmetry), 2 pi int mu_s(eta) d eta, equals the group-to-group cross section mu_s_l[..., 0] (the exact
    Legendre moment of the build) within 5e-3 for every transfer above 1e-3 of the largest, for the adaptive table at
    both energies. The tolerance covers the difference between the two definitions of the transfer on the coarse
    fine-energy bins of this test (the table takes the exact fraction of every bin that lands in the outgoing group,
    the moments assign each Gauss node by the bin's midpoint energy): at 125 kVp the uniform table shows the same
    deviation, 0.4 % on a small off-diagonal transfer, and the two tables agree to 1e-3. At 6 MV the uniform table
    overestimates the within-group transfers by more than 5 % (the linear interpolation spreads the unresolved forward
    peak over a whole table cell)."""
    q = np.linspace(0.0, 1.0, 400001)
    weights = 4.0 * q * (q[1] - q[0])
    weights[0] *= 0.5
    weights[-1] *= 0.5
    eta = torch.as_tensor(1.0 - 2.0 * q * q)
    ratios = {}
    for name, library in zip(("uniform", "adaptive"), tables[case][:2]):
        values = transfer_eta_at(library, torch.ones(1, eta.numel(), dtype=torch.float64), eta).numpy()
        integral = 2.0 * math.pi * (values * weights).sum(-1)
        reference = library.mu_s_l[0, :, :, 0].numpy()
        significant = reference > 1e-3 * reference.max()
        ratios[name] = integral[significant] / reference[significant]
    assert np.abs(ratios["adaptive"] - 1.0).max() < 5e-3, ratios["adaptive"]
    if case == "6mv":
        assert ratios["uniform"].max() > 1.05
    else:
        assert np.abs(ratios["adaptive"] - ratios["uniform"]).max() < 1e-3


# ---- kernel against the PyTorch path ------------------------------------------------------------------------------

def _kernel_device():
    from citrus.scatter_solver.kernels import uncollided_kernels

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    device = torch.device("cuda")
    ok, why = uncollided_kernels.kernel_usable(device)
    if not ok:
        pytest.skip(f"first-collision-source kernel unusable: {why}")
    return device


@pytest.mark.parametrize("pair_mode", [False, True])
def test_kernel_matches_torch_on_adaptive_tables(pair_mode):
    """The fused first-collision kernel equals the PyTorch path (1e-5 of the field maximum) on the synthetic adaptive
    table, dense (2 materials) and pair (7 materials) modes, with directions parallel and antiparallel to Omega_UC."""
    from citrus.scatter_solver.cross_sections import material_pairs
    from citrus.scatter_solver.geometry import ConeGeometry, Grid
    from citrus.scatter_solver.uncollided import first_collision_source_at_directions, primary_directions_to_voxels

    device = _kernel_device()
    M = 7 if pair_mode else 2
    library = synthetic_library(6, legendre_order=1, num_eta=2001, num_materials=M, angular_table_grid="adaptive")
    grid = Grid(shape=(6, 9, 7), delta_voxel=0.7, voxel_slice_aspect=0.9)
    geometry = ConeGeometry(source_iso_dist=20.0, source_detector_dist=30.0, num_det_rows=8, num_det_channels=8,
                            delta_det_row=1.0, delta_det_channel=1.0)
    generator = torch.Generator().manual_seed(11)
    if pair_mode:
        rho = torch.zeros(M, *grid.shape)
        rho.scatter_(0, torch.randint(0, M, (2,) + tuple(grid.shape), generator=generator),
                     torch.rand((2,) + tuple(grid.shape), generator=generator) + 0.1)
    else:
        rho = torch.rand(M, *grid.shape, generator=generator)
    phi = torch.rand(6, *grid.shape, generator=generator) + 0.01
    omega_uncollided = primary_directions_to_voxels(grid, geometry, 0.5, "cpu", torch.float32)
    directions = torch.nn.functional.normalize(torch.randn(9, 3, generator=generator), dim=-1)
    directions[0], directions[1] = omega_uncollided[1, 2, 3], -omega_uncollided[4, 6, 2]
    rho, phi, directions = rho.to(device), phi.to(device), directions.to(device)
    pairs = material_pairs(rho) if pair_mode else None
    arguments = (library, rho, phi, grid, geometry, 0.5, directions)
    reference = first_collision_source_at_directions(*arguments, chunk=3, backend="torch",
                                                     pairs=pairs if pair_mode else "dense")
    out = first_collision_source_at_directions(*arguments, backend="triton", pairs=pairs if pair_mode else "dense")
    error = float((out.double() - reference.double()).abs().max() / reference.abs().max())
    assert error < 1e-5, error
