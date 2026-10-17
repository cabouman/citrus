"""The transport grid C (cubic cells covering the fine box), the quadrature grid Q = C.refine(q) of the first-order
quantities, the conservative resampling of the densities onto both, the dense / pair first-collision evaluation and
the regression anchor against the pre-quadrature-grid solver (version 0.1.0)."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from citrus.scatter_solver.config import SolverConfig
from citrus.scatter_solver.cross_sections import material_pairs, material_pairs_if_compact, synthetic_library
from citrus.scatter_solver.detector import csi_response, no_grid
from citrus.scatter_solver.geometry import Grid, overlap_matrix, resample_conservative
from citrus.scatter_solver.ray_tracing import SiddonProjectors
from citrus.scatter_solver.scatter_solver import ScatterSolver
from citrus.scatter_solver.uncollided import (DENSE, first_collision_source_at_directions, resolve_pairs,
                                              uncollided_fluence)
from citrus.scatter_solver.source import SourceModel
from citrus.utilities import wlab_geometry, wlab_grid, wlab_phantom

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
F64 = torch.float64
REFERENCE = Path(__file__).parent / "data" / "pre_quadrature_reference.json"
SLOW = pytest.mark.skipif(os.environ.get("CITRUS_SLOW_TESTS") != "1",
                          reason="set CITRUS_SLOW_TESTS=1 (real XrayPhysics library, cached)")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# the problem of tests/test_scatter_solver.py: 2 cm WLAB grid, 4 cm transport cells, q = 2 (Q is the 2 cm grid)
GRID = Grid((10, 16, 20), 2.0)
CONFIG = SolverConfig(num_groups=4, legendre_order=3, num_polar=4, num_azimuthal=8, cone_num_polar=8,
                      cone_num_azimuthal=16, transport_cell_cm=4.0, quadrature_point_spacing_cm=2.0,
                      source_iteration_tolerance=1e-4, max_source_iterations=40, direction_chunk=8)


def _solver(config: SolverConfig = CONFIG, device=DEVICE, dtype=torch.float32, library=None) -> ScatterSolver:
    library = synthetic_library(config.num_groups, 5, num_materials=7) if library is None else library
    return ScatterSolver(library, config, wlab_geometry(), SiddonProjectors(), csi_response(), no_grid(), device=device,
                         dtype=dtype)


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / np.abs(b).max())


def _blocks(image, n: int = 8) -> np.ndarray:
    """Means of an image over n x n blocks (numpy.array_split of rows and channels: the layout of the reference)."""
    image = np.asarray(image.detach().cpu().double() if isinstance(image, torch.Tensor) else image, dtype=np.float64)
    r = np.array_split(np.arange(image.shape[0]), n)
    c = np.array_split(np.arange(image.shape[1]), n)
    return np.array([[image[np.ix_(ri, ci)].mean() for ci in c] for ri in r])


def _check_reference(r, case: dict, tolerance: float) -> None:
    """A ViewResult against one case of the pre-quadrature-grid reference (block means, sums, source-iteration count
    and change)."""
    assert list(r.scatter_signal.shape) == case["image_shape"]
    assert _rel(_blocks(r.scatter_signal), case["scatter_signal_blocks"]) < tolerance
    assert _rel(_blocks(r.primary["signal"]), case["primary_signal_blocks"]) < tolerance
    assert abs(float(r.scatter_signal.double().sum()) / case["scatter_signal_sum"] - 1) < tolerance
    assert abs(float(r.primary["signal"].double().sum()) / case["primary_signal_sum"] - 1) < tolerance
    assert r.source_iteration_result.num_iterations == case["si_iterations"]
    assert r.source_iteration_result.history[-1] == pytest.approx(case["si_final_change"], rel=max(1e-3, tolerance))


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm())


# ---- grids ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("grid, cell, shape", [
    (Grid((40, 60, 80), 0.5), 1.0, (20, 30, 40)),  # the WLAB 0.5 cm grid: tiled exactly, no padding
    (Grid((67, 100, 134), 0.3, recon_slice_offset=1.0), 1.0, (21, 30, 41)),  # 20.1 x 30 x 40.2 cm: padded
    (Grid((15, 20, 12), 0.4, voxel_row_aspect=1.5, voxel_slice_aspect=0.75), 1.0, (9, 8, 4)),  # anisotropic voxels
    (Grid((10, 16, 20), 2.0), 3.0, (7, 11, 14)),
])
def test_covering_grid_is_centred_and_covers(grid, cell, shape):
    """Grid.covering: cubic cells of the given edge, ceil(extent / cell) per axis, the same centre as the fine box,
    covering it with less than one cell of padding per axis; refine(q) keeps the box."""
    c = Grid.covering(grid, cell)
    assert c.shape == shape and c.deltas == (cell, cell, cell)
    lo_f, lo_c = np.array(grid.lower_corner()), np.array(c.lower_corner())
    ext_f, ext_c = np.array(grid.extent), np.array(c.extent)
    np.testing.assert_allclose(lo_c + ext_c / 2, lo_f + ext_f / 2, atol=1e-12)  # centred
    assert np.all(ext_c >= ext_f - 1e-9) and np.all(ext_c < ext_f + cell)  # covers, less than one cell of padding
    q = c.refine(3)
    assert q.shape == tuple(3 * n for n in shape) and q.coarsen(3).shape == shape
    np.testing.assert_allclose(q.lower_corner(), c.lower_corner(), atol=1e-12)
    np.testing.assert_allclose(q.extent, c.extent, atol=1e-12)
    assert c.refine(1) is c


def test_covering_exact_tiling_is_coarsening():
    """When the cells tile the box (integer ratio) the covering grid is the old supervoxel grid grid.coarsen(f)."""
    for g, cell, f in ((wlab_grid(0.5), 1.0, 2), (wlab_grid(1.0), 1.0, 1), (GRID, 4.0, 2), (wlab_grid(0.25), 1.0, 4)):
        assert Grid.covering(g, cell) == g.coarsen(f)


def test_overlap_matrix_rows():
    """Overlap fractions: rows of a target cell inside the source partition sum to 1; a target cell half outside sums
    to 1/2; columns times the target width give the source width (conservation)."""
    ef = torch.linspace(-1.5, 1.5, 11, dtype=F64)  # 10 cells of 0.3
    et = torch.linspace(-2.0, 2.0, 5, dtype=F64)  # 4 cells of 1.0
    W = overlap_matrix(ef, et)
    np.testing.assert_allclose(W.sum(1).numpy(), [0.5, 1.0, 1.0, 0.5], atol=1e-12)
    np.testing.assert_allclose((W * 1.0).sum(0).numpy(), np.full(10, 0.3), atol=1e-12)


# ---- conservative resampling -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype, tolerance", [(torch.float64, 1e-14), (torch.float32, 1e-6)])
def test_resampling_integer_ratio_equals_block_mean(dtype, tolerance):
    """For an integer ratio with aligned boxes (the WLAB cases) the overlap resampling is the block mean (avg_pool3d)
    to float rounding; equal grids return the field itself."""
    g = Grid((8, 12, 16), 0.5)
    rho = torch.rand(3, *g.shape, dtype=dtype, generator=torch.Generator().manual_seed(1))
    for f in (2, 4):
        c = Grid.covering(g, 0.5 * f)
        got = resample_conservative(rho, g, c)
        reference = F.avg_pool3d(rho[None], f, f)[0]
        assert float((got - reference).abs().max()) <= tolerance * float(reference.abs().max())
    assert resample_conservative(rho, g, g) is rho


@pytest.mark.parametrize("grid", [Grid((67, 100, 134), 0.3, recon_slice_offset=0.7),
                                  Grid((14, 9, 11), 0.37, voxel_row_aspect=1.3, voxel_slice_aspect=0.6)])
def test_resampling_conserves_mass_at_non_integer_ratio(grid):
    """Non-integer voxel / cell ratios (0.3 cm voxels in 1 cm cells, padded box; anisotropic 0.37 cm voxels): the
    integral of every material is conserved on the transport grid C and on the quadrature grid Q (q = 2, 3), and a
    uniform field stays uniform in the cells fully inside the fine box."""
    rho = torch.rand(7, *grid.shape, dtype=F64, generator=torch.Generator().manual_seed(2))
    total = rho.sum(dim=(1, 2, 3)) * grid.voxel_volume
    c = Grid.covering(grid, 1.0)
    for target in (c, c.refine(2), c.refine(3)):
        r = resample_conservative(rho, grid, target)
        assert r.shape == (7, *target.shape)
        np.testing.assert_allclose((r.sum(dim=(1, 2, 3)) * target.voxel_volume).numpy(), total.numpy(), rtol=1e-12)
        assert float(r.min()) >= 0.0
    one = resample_conservative(torch.ones(1, *grid.shape, dtype=F64), grid, c)[0]
    # cells fully inside the fine box are 1; cells reaching into the padding are between 0 and 1
    assert float(one.max()) == pytest.approx(1.0, abs=1e-12) and float(one.min()) > 0.0


def test_material_fields_on_quadrature_grid():
    """material_fields at a non-integer ratio (0.3 cm voxels, 1 cm cells, q = 2): grid_quadrature =
    grid_coarse.refine(q), rho_coarse and rho_quadrature conserve every material, no [G, ...] field on the fine grid,
    and the pair representation is used exactly when every quadrature point has at most two nonzero materials."""
    grid = Grid((34, 50, 67), 0.3)
    config = replace(CONFIG, transport_cell_cm=1.0, quadrature_point_spacing_cm=0.5)
    s = _solver(config, device="cpu", dtype=F64)
    rho = torch.zeros(7, *grid.shape, dtype=F64)
    generator = torch.Generator().manual_seed(3)
    sel = torch.randint(0, 7, (2, *grid.shape), generator=generator)
    rho.scatter_(0, sel, torch.rand(2, *grid.shape, dtype=F64, generator=generator))  # 2 materials per fine voxel
    f = s.material_fields(rho, grid)
    assert f.q == 2 and f.grid_coarse == Grid.covering(grid, 1.0) and f.grid_quadrature == f.grid_coarse.refine(2)
    total = rho.sum(dim=(1, 2, 3)) * grid.voxel_volume
    for r, g in ((f.rho_coarse, f.grid_coarse), (f.rho_quadrature, f.grid_quadrature)):
        np.testing.assert_allclose((r.sum(dim=(1, 2, 3)) * g.voxel_volume).numpy(), total.numpy(), rtol=1e-12)
    assert f.pairs_quadrature is None and f.first_collision_pairs == DENSE  # block means mix more than two materials
    assert not any(isinstance(t, torch.Tensor) and t.shape[0] == CONFIG.num_groups and tuple(t.shape[1:]) == grid.shape
                   for t in vars(f).values())
    # a field with one material per block of the quadrature grid keeps the pair path
    pure = wlab_phantom(GRID, dtype=F64)
    f2 = _solver(device="cpu", dtype=F64).material_fields(pure, GRID)
    assert f2.pairs_quadrature is not None and torch.equal(f2.pairs_quadrature.dense(), f2.rho_quadrature)


@pytest.mark.parametrize("cell, voxel, spacing, q", [
    (1.0, 1.0, 0.5, 1), (1.0, 0.5, 0.5, 2), (1.0, 0.4, 0.5, 2), (1.0, 0.05, 0.5, 2), (1.0, 0.25, 0.25, 4),
    (1.0, 0.3, 0.25, 3), (1.0, 0.4, 0.25, 2), (1.0, 0.6, 0.25, 2), (1.0, 2.0, 0.5, 1), (4.0, 2.0, 2.0, 2),
    (4.0, 2.0, 4.0, 1), (3.0, 2.0, 1.5, 2), (2.0, 0.5, 0.5, 4), (2.0, 0.25, 0.5, 4), (2.0, 1.0, 0.5, 2),
    (0.5, 0.25, 0.5, 1), (0.5, 0.125, 0.5, 1), (1.0, 0.1, 0.4, 2), (1.0, 0.1, 0.3, 3), (3.0, 0.5, 0.5, 6),
])
def test_quadrature_points_rule(cell, voxel, spacing, q):
    """The effective q = min(max(1, round(cell / quadrature_point_spacing_cm)), max(1, round(cell / largest voxel
    edge))) (Python's round: 2.5 -> 2, 1.5 -> 2): points about quadrature_point_spacing_cm apart (default 0.5 cm: 1 cm
    cells -> 2, 2 cm cells -> 4, 0.5 cm cells -> 1), never much closer than the fine voxels."""
    config = replace(SolverConfig(), transport_cell_cm=cell, quadrature_point_spacing_cm=spacing)
    assert config.quadrature_points(voxel) == q
    assert SolverConfig().quadrature_point_spacing_cm == 0.5


def test_quadrature_points_in_fields():
    """material_fields records the effective q (the largest voxel edge counts: anisotropic voxels), and the solves with
    q = 1 and with an upper bound above the resolution (clamped to round(cell / voxel)) converge to finite,
    non-negative images; invalid values are rejected."""
    rho = wlab_phantom(GRID, device=DEVICE)
    for spacing, q in ((4.0, 1), (1.0, 2)):  # 4 cm cells on the 2 cm grid: at most 2 points per axis
        s = _solver(replace(CONFIG, quadrature_point_spacing_cm=spacing))
        f = s.material_fields(rho, GRID)
        assert f.q == q == s.quadrature_points(GRID) and f.grid_quadrature == f.grid_coarse.refine(q)
        r = s.solve_view(rho, GRID, 0.4)
        assert r.source_iteration_result.converged and bool(torch.isfinite(r.scatter_signal).all())
        assert float(r.scatter_signal.min()) >= 0
    aniso = Grid((10, 16, 20), 1.0, voxel_slice_aspect=2.0)  # 1 x 1 x 2 cm voxels in 4 cm cells: q = 2, not 4
    assert _solver(replace(CONFIG, quadrature_point_spacing_cm=1.0)).quadrature_points(aniso) == 2
    for bad in (dict(quadrature_point_spacing_cm=0.0), dict(quadrature_point_spacing_cm=-0.5),
                dict(transport_cell_cm=-1.0), dict(uncollided_trace_grid="medium")):
        with pytest.raises(ValueError):
            replace(CONFIG, **bad).validate()


def test_quadrature_points_converge():
    """(e) q = 1, 2, 3 points per axis in 8 cm cells of the 2 cm WLAB grid against q = 4 (Q = the fine grid),
    uncollided fluence traced through the fine grid: the scatter images approach the q = 4 image (very coarse cells
    for this object, so the differences are large)."""
    grid = Grid((12, 16, 20), 2.0)
    rho = wlab_phantom(grid, device=DEVICE)
    image = {}
    for q in (1, 2, 3, 4):
        s = _solver(replace(CONFIG, transport_cell_cm=8.0, quadrature_point_spacing_cm=8.0 / q,
                            uncollided_trace_grid="fine"))
        assert s.quadrature_points(grid) == q
        image[q] = s.solve_view(rho, grid, 0.4).scatter_signal
    d1, d2, d3 = (_rel_l2(image[q], image[4]) for q in (1, 2, 3))
    assert d3 < d2 < d1 < 0.3, (d1, d2, d3)


def test_trace_grids_synthetic():
    """(d) The uncollided fluence traced through C ("coarse"), Q ("quadrature") or F ("fine"): with q = cells / voxel
    "quadrature" equals "fine" (Q is F); "coarse" differs by the smearing of the 4 cm cells (bounded)."""
    rho = wlab_phantom(GRID, device=DEVICE)
    image = {}
    for trace in ("coarse", "quadrature", "fine"):
        image[trace] = _solver(replace(CONFIG, uncollided_trace_grid=trace)).solve_view(rho, GRID, 0.4).scatter_signal
    assert _rel_l2(image["quadrature"], image["fine"]) < 1e-6
    assert 1e-4 < _rel_l2(image["coarse"], image["fine"]) < 0.05


def test_non_integer_cells_synthetic():
    """3 cm cells on the 2 cm grid (ratio 1.5, q = round(1.5) = 2, padded 21 x 33 x 42 cm transport grid): the solve
    converges and its scatter image is closer to the 2 cm-cell solve (q = 1, the reference) than the 4 cm-cell solve
    (q = 2) is; the primary stays on the fine grid (identical)."""
    rho = wlab_phantom(GRID, device=DEVICE)
    r = {cell: _solver(replace(CONFIG, transport_cell_cm=cell)).solve_view(rho, GRID, 0.4) for cell in (2.0, 3.0, 4.0)}
    assert r[3.0].source_iteration_result.converged and float(r[3.0].scatter_signal.min()) >= 0
    d3 = _rel_l2(r[3.0].scatter_signal, r[2.0].scatter_signal)
    d4 = _rel_l2(r[4.0].scatter_signal, r[2.0].scatter_signal)
    assert d3 < 0.05 and d3 < d4, (d3, d4)
    assert torch.equal(r[3.0].primary["signal"], r[2.0].primary["signal"])


# ---- ray tracing to arbitrary points -----------------------------------------------------------------------------

def test_upstream_to_points(device):
    """upstream_to_points at the voxel centers equals upstream_cumulative; points outside the box get the integral of
    the part of the ray inside it (Triton == PyTorch on float32 CUDA); uncollided_fluence(points=) at the centers of
    its own grid equals the default."""
    geometry = wlab_geometry()
    g = Grid((5, 8, 10), 2.0)
    generator = torch.Generator().manual_seed(4)
    mu = torch.rand(3, *g.shape, dtype=F64, generator=generator).to(device)
    P = SiddonProjectors(backend="torch")
    cen = g.center_coordinates(device, F64).reshape(-1, 3)
    a = P.upstream_to_points(mu, g, geometry, 0.3, cen).reshape(mu.shape)
    torch.testing.assert_close(a, P.upstream_cumulative(mu, g, geometry, 0.3), rtol=1e-12, atol=1e-12)
    pts = ((torch.rand(500, 3, dtype=F64, generator=generator) - 0.5)
           * torch.tensor([40.0, 30.0, 50.0], dtype=F64)).to(device)
    reference = P.upstream_to_points(mu, g, geometry, 0.3, pts)
    if device.type == "cuda":
        got = SiddonProjectors().upstream_to_points(mu.float(), g, geometry, 0.3, pts.float())
        assert float((got.double() - reference).abs().max()) < 1e-5 * float(reference.abs().max())
    library = synthetic_library(3, 2, num_materials=7)
    rho = torch.zeros(7, *g.shape, dtype=F64, device=device)
    rho[3] = torch.rand(*g.shape, dtype=F64, generator=generator).to(device)
    phi = uncollided_fluence(library, rho, g, geometry, 0.3, SourceModel(), P)
    torch.testing.assert_close(uncollided_fluence(library, rho, g, geometry, 0.3, SourceModel(), P,
                                                  points=Grid.covering(g, 2.0)),
                               phi, rtol=0, atol=0)


# ---- dense / pair first-collision source --------------------------------------------------------------------------

@cuda
@pytest.mark.parametrize("G, K", [(4, 6), (6, 9)])
def test_fc_dense_kernel_seven_materials(G, K):
    """(c) The Triton dense mode with M = 7 (up to 7 nonzero materials per point) equals the PyTorch path (float64
    reference to float32 rounding); where every point has at most two materials the dense mode equals the pair mode."""
    from citrus.scatter_solver.kernels import uncollided_kernels

    ok, why = uncollided_kernels.kernel_usable(torch.device("cuda"))
    if not ok:
        pytest.skip(why)
    library = synthetic_library(G, 2, num_eta=301, num_materials=7)
    g = Grid((6, 9, 7), 1.5)
    geometry = wlab_geometry()
    generator = torch.Generator().manual_seed(G + K)
    rho = (torch.rand(7, *g.shape, dtype=F64, generator=generator)
           * (torch.rand(7, *g.shape, generator=generator) > 0.25))
    rho[:, 0] = 0.0
    phi = torch.rand(G, *g.shape, dtype=F64, generator=generator)
    directions = torch.nn.functional.normalize(torch.randn(K, 3, dtype=F64, generator=generator), dim=-1)
    assert int((rho != 0).sum(0).max()) > 2 and material_pairs_if_compact(rho) is None
    assert resolve_pairs(rho, None) == DENSE
    # float64 CPU, PyTorch path
    reference = first_collision_source_at_directions(library, rho, phi, g, geometry, 0.3, directions)
    device = torch.device("cuda")
    got = first_collision_source_at_directions(library, rho.float().to(device), phi.float().to(device), g, geometry,
                                               0.3,
                                               directions.float().to(device),
                                               backend="triton", pairs=DENSE)
    assert float((got.double().cpu() - reference).abs().max()) < 1e-5 * float(reference.abs().max())
    tor = first_collision_source_at_directions(library, rho.float().to(device), phi.float().to(device), g, geometry,
                                               0.3,
                                               directions.float().to(device),
                                               backend="torch", pairs=DENSE)
    assert float((got - tor).abs().max()) < 1e-5 * float(tor.abs().max())
    # at most two materials per point: dense mode == pair mode
    sel = torch.randint(0, 7, (2, *g.shape), generator=generator)
    rho2 = torch.zeros(7, *g.shape).scatter_(0, sel, torch.rand(2, *g.shape, generator=generator)).to(device)
    pairs = material_pairs(rho2)
    d = first_collision_source_at_directions(library, rho2, phi.float().to(device), g, geometry, 0.3,
                                             directions.float().to(device),
                                             pairs=DENSE)
    p = first_collision_source_at_directions(library, rho2, phi.float().to(device), g, geometry, 0.3,
                                             directions.float().to(device),
                                             pairs=pairs)
    assert float((d - p).abs().max()) < 1e-6 * float(p.abs().max())


# ---- regression anchor: q = cells / voxel with the fine trace reproduces version 0.1.0 --------------------------

@pytest.mark.parametrize("device, dtype, tolerance", [("cpu", torch.float64, 1e-10)]
                         + ([("cuda", torch.float32, 1e-5)] if torch.cuda.is_available() else []))
def test_reproduces_pre_quadrature_solver(device, dtype, tolerance):
    """(a) With q = transport cell / voxel (Q = F) and the fine trace, solve_view reproduces the solver of version 0.1.0
    (tests/data/pre_quadrature_reference.json: 8 x 8 block means of the signals, sums, source-iteration count, generated
    with it) to float rounding: the only change on this path is the overlap resampling onto C (the block mean up to
    rounding)."""
    case = json.loads(REFERENCE.read_text())["cases"][f"synthetic_{device}"]
    s = _solver(replace(CONFIG, uncollided_trace_grid="fine"), device=device, dtype=dtype)
    _check_reference(s.solve_view(wlab_phantom(GRID, device=device, dtype=dtype), GRID, 0.4), case, tolerance)


@SLOW
@cuda
def test_reproduces_pre_quadrature_solver_wlab():
    """(a) The real library on WLAB, one view: 1 cm voxels in 1 cm cells (q = 1: everything on one grid) and 0.5 cm
    voxels in 1 cm cells (q = 2) against version 0.1.0, with every trace grid (the WLAB phantom's boundaries lie on the
    cell faces here, so the three traces agree to rounding). The library of version 0.1.0 has a uniform eta table and
    equal-fluence groups (the defaults up to version 0.5)."""
    from citrus.utilities import wlab_solver

    cases = json.loads(REFERENCE.read_text())["cases"]
    for dv, q in ((1.0, 1), (0.5, 2)):
        grid = wlab_grid(dv)
        rho = wlab_phantom(grid, device="cuda")
        case = cases[f"wlab_{str(dv).replace('.', 'p')}cm_f{q}"]
        for trace in ("fine", "quadrature", "coarse"):
            config = replace(SolverConfig(), uncollided_trace_grid=trace, angular_table_grid="uniform",
                             group_edges_method="equal_fluence")
            s = wlab_solver(dv, config=config, device="cuda")
            assert s.quadrature_points(grid) == q
            _check_reference(s.solve_view(rho, grid, 0.3), case, 1e-5)


@SLOW
@cuda
def test_quadrature_points_and_ratios_wlab():
    """(d, e) WLAB, real library, 1 cm cells: q = 1 / 2 against q = 3 (0.25 cm voxels, which allow q up to 4; measured
    0.9 % / 0.2 % in the AP view at 0.5 cm); 0.4 cm voxels (ratio 2.5, q = 2) against 0.5 cm voxels (0.3 %, partly the
    different rasterization of the phantom: the primary differs by 0.2 %)."""
    from citrus.utilities import wlab_solver

    library = wlab_solver(0.5, device="cuda").library.to("cpu", torch.float64)
    g025 = wlab_grid(0.25)
    rho025 = wlab_phantom(g025, device="cuda")
    image = {}
    for q in (1, 2, 3):
        s = wlab_solver(0.25, config=replace(SolverConfig(), quadrature_point_spacing_cm=1.0 / q), library=library,
                        device="cuda")
        assert s.quadrature_points(g025) == q
        image[q] = s.solve_view(rho025, g025, 0.3).scatter_signal
    d1, d2 = _rel_l2(image[1], image[3]), _rel_l2(image[2], image[3])
    assert d2 < d1 < 0.02 and d2 < 0.005, (d1, d2)
    del rho025
    g05 = wlab_grid(0.5)
    image2_05 = wlab_solver(0.5, library=library, device="cuda").solve_view(wlab_phantom(g05, device="cuda"), g05, 0.3)
    g04 = wlab_grid(0.4)
    s = wlab_solver(0.4, library=library, device="cuda")
    r04 = s.solve_view(wlab_phantom(g04, device="cuda"), g04, 0.3)
    assert s.material_fields(wlab_phantom(g04, device="cuda"), g04).grid_coarse.shape == (20, 30, 40)
    assert _rel_l2(r04.scatter_signal, image2_05.scatter_signal) < 0.01
