"""Tests of source iteration over collision order (citrus/scatter_solver/source_iteration.py): fixed point,
particle balance, iteration cap, devices and the scattering-source operator."""

from __future__ import annotations

import pytest
import torch

from citrus.scatter_solver.angular_quadrature import product_quadrature
from citrus.scatter_solver.geometry import Grid
from citrus.scatter_solver.source_iteration import (
    multiple_collision_source_moments,
    scatter_source_function,
    source_iteration,
)
from citrus.scatter_solver.transport_sweep import make_sweep_context, sweep
from citrus.scatter_solver.cross_sections import mu_t_volume, synthetic_library, transfer_moments_volume

F64 = torch.float64


def _triton_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from citrus.scatter_solver.kernels.sweep_kernels import kernels_usable

    ok, why = kernels_usable(torch.device("cuda"))
    if not ok:
        pytest.skip(f"Triton sweep kernel unusable: {why}")


def _problem(device="cpu", dtype=F64, n=5, h=2.0, L=3, backend="auto"):
    """Small water/bone cube with the synthetic library; first-collision source in one corner, mildly anisotropic."""
    library = synthetic_library(num_groups=4, legendre_order=5)
    grid = Grid((n, n, n), h)
    rho = torch.zeros(2, n, n, n, dtype=F64)
    rho[0] = 1.0
    rho[1, 1:3, 2:4, 1:4] = 1.2  # a bone insert
    rho[0, 1:3, 2:4, 1:4] = 0.0
    mu_t = mu_t_volume(library, rho).to(device=device, dtype=dtype)
    mu_s = transfer_moments_volume(library, rho).to(device=device, dtype=dtype)
    q = product_quadrature(4, 8, dtype=F64)
    context = make_sweep_context(grid, q, L, device=device, dtype=dtype, backend=backend)
    G = library.num_groups
    iso = torch.zeros(G, 4, n, n, n, dtype=F64)
    iso[:, 0, :2, :2, :2] = torch.linspace(1.0, 0.4, G, dtype=F64)[:, None, None, None]
    iso[:, 1, :2, :2, :2] = 0.1  # x slope
    iso = iso.to(device=device, dtype=dtype)
    omega = context.quadrature.omega.to(device=device, dtype=dtype)

    def first_collision_source_function(index):
        aniso = 1.0 + 0.3 * omega.index_select(0, index)[:, 0]  # positive, direction dependent
        return aniso[:, None, None, None, None, None] * iso[None]

    return context, mu_t, mu_s, first_collision_source_function


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_source_iteration_converges_and_is_fixed_point(backend):
    """float64 PyTorch path (CPU), or the fused Triton path in float32 on CUDA (same tolerances: the relative
    change and the fixed-point residual are well above float32 rounding at tol = 1e-6)."""
    if backend == "torch":
        context, mu_t, mu_s, first_collision = _problem(backend="torch")
    else:
        _triton_or_skip()
        context, mu_t, mu_s, first_collision = _problem(device=torch.device("cuda"), dtype=torch.float32,
                                                        backend="triton")
    tolerance = 1e-6
    seen = []
    result = source_iteration(context, mu_t, mu_s, first_collision, tolerance=tolerance, max_iterations=60,
                              callback=lambda n, phi_n, rel: seen.append((n, rel)))
    assert result.converged and result.num_iterations < 30, (result.num_iterations, result.history)
    assert len(result.history) == result.num_iterations == len(result.phi_norms) == len(seen)
    assert result.history[0] == 1.0 and result.history[-1] < tolerance
    h = result.history
    assert all(h[k + 1] < h[k] for k in range(2, len(h) - 1)), h
    assert result.phi_total.shape == (4, context.num_moments, 4) + context.shape
    # fixed point: sweep(scatter source of phi_total) + sweep(first-collision source) == phi_total
    T = multiple_collision_source_moments(mu_s, result.phi_total)
    assert T.shape == result.phi_total.shape
    phi_fp = sweep(context, mu_t, scatter_source_function(context, T)) + sweep(context, mu_t, first_collision)
    rel = float((phi_fp - result.phi_total).norm() / result.phi_total.norm())
    assert rel < 10 * tolerance, rel
    # the scalar flux is positive and the cell means of the total fluence are physical
    assert float(result.phi_total[:, 0, 0].min()) > 0


def test_source_iteration_max_iterations():
    context, mu_t, mu_s, first_collision = _problem(h=4.0)  # optically thicker cells
    result = source_iteration(context, mu_t, mu_s, first_collision, tolerance=1e-12, max_iterations=3)
    assert result.num_iterations == 3 and not result.converged and len(result.history) == 3
    _, psi = sweep(context, mu_t, first_collision, return_psi=True)
    # thick cells: the linear-discontinuous angular fluence has negative cell means
    assert float(psi[:, :, 0].min()) < 0
    seen = []
    result_c = source_iteration(context, mu_t, mu_s, first_collision, tolerance=1e-4, max_iterations=30,
                                callback=lambda n, phi_n, rel: seen.append(n))
    assert result_c.converged and seen == list(range(1, result_c.num_iterations + 1))
    assert bool(torch.isfinite(result_c.phi_total).all()) and float(result_c.phi_total[:, 0, 0].sum()) > 0.0


def test_source_iteration_device_float32(device):
    context64, mu_t64, mu_s64, first_collision64 = _problem(n=4)
    reference = source_iteration(context64, mu_t64, mu_s64, first_collision64, tolerance=1e-5, max_iterations=40)
    context, mu_t, mu_s, first_collision = _problem(device=device, dtype=torch.float32, n=4)
    result = source_iteration(context, mu_t, mu_s, first_collision, tolerance=1e-5, max_iterations=40,
                              direction_chunk=2)
    assert result.phi_total.device.type == device.type and result.phi_total.dtype == torch.float32
    assert abs(result.num_iterations - reference.num_iterations) <= 1
    err = float((result.phi_total.cpu().double() - reference.phi_total).norm() / reference.phi_total.norm())
    assert err < 1e-4, err


def test_scatter_source_fn_matches_angular():
    """The scattering source function writes cell-major moments [D, R, C, S, G, 4] (the sweep's layout), from a
    contiguous T or from the cell-major T of scatter_moments, into a given buffer or added to it."""
    from citrus.scatter_solver.angular_quadrature import scatter_source_at_directions, scatter_source_moments
    from citrus.scatter_solver.source_iteration import scatter_moments
    from citrus.scatter_solver.transport_sweep import source_is_cell_major

    context, mu_t, mu_s, _ = _problem(n=3)
    phi = torch.randn(4, context.num_moments, 4, *context.shape, dtype=F64)
    T = scatter_source_moments(mu_s, phi)
    index = torch.tensor([0, 5, 17, 31, 30])
    reference = scatter_source_at_directions(T, context.basis.Y[index]).permute(0, 3, 4, 5, 1, 2)  # cell-major
    for moments in (T, scatter_moments(mu_s, phi)):
        source = scatter_source_function(context, moments)
        assert source_is_cell_major(source)
        out = source(index)
        assert out.shape == (5,) + context.shape + (4, 4)
        assert torch.allclose(out, reference, rtol=1e-13, atol=1e-14)
        # `out=` (lent by the sweep) is written in place, or added to with accumulate=True
        buf = torch.full((5,) + context.shape + (4, 4), float("nan"), dtype=F64)
        result = source(index, out=buf)
        assert result.data_ptr() == buf.data_ptr()
        assert torch.allclose(buf, reference, rtol=1e-13, atol=1e-14)
        source(index, out=buf, accumulate=True)
        assert torch.allclose(buf, 2 * reference, rtol=1e-13, atol=1e-14)


def test_scatter_moments_matches_angular(device):
    from citrus.scatter_solver.angular_quadrature import scatter_source_moments
    from citrus.scatter_solver.source_iteration import scatter_moments

    torch.manual_seed(3)
    G, Lx, spectrum_density = 3, 4, (2, 3, 4)
    mu_s = torch.rand(G, G, Lx + 1, *spectrum_density, dtype=F64, device=device)
    mu_s[1, 0] = 0.0  # a zero (upscatter) block is skipped
    for rest in [(4,) + spectrum_density, spectrum_density]:
        phi = torch.randn(G, 9, *rest, dtype=F64, device=device)  # degree 2 <= Lx
        reference = scatter_source_moments(mu_s, phi)
        out = scatter_moments(mu_s, phi)
        assert out.shape == reference.shape and out.device == reference.device
        assert torch.allclose(out, reference, rtol=1e-13, atol=1e-14)
        # a view of the sweep's cell-major storage [N, *spatial, G, (4)]
        cell_major = out.permute(1, 3, 4, 5, 0, 2) if len(rest) == 4 else out.permute(1, 2, 3, 4, 0)
        assert cell_major.is_contiguous()
