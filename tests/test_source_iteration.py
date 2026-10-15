"""Tests of source iteration over collision order (citrus/scatter_solver/source_iteration.py): fixed point,
particle balance, iteration cap, devices and the scattering-source operator."""

from __future__ import annotations

import pytest
import torch

from citrus.scatter_solver.angular_quadrature import product_quadrature
from citrus.scatter_solver.geometry import Grid
from citrus.scatter_solver.source_iteration import (
    multiple_collision_source_moments,
    scatter_source_fn,
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
    """Small water/bone cube with the synthetic library; FC source in one corner, mildly anisotropic."""
    lib = synthetic_library(num_groups=4, legendre_order=5)
    grid = Grid((n, n, n), h)
    rho = torch.zeros(2, n, n, n, dtype=F64)
    rho[0] = 1.0
    rho[1, 1:3, 2:4, 1:4] = 1.2  # a bone insert
    rho[0, 1:3, 2:4, 1:4] = 0.0
    mu_t = mu_t_volume(lib, rho).to(device=device, dtype=dtype)
    mu_s = transfer_moments_volume(lib, rho).to(device=device, dtype=dtype)
    q = product_quadrature(4, 8, dtype=F64)
    ctx = make_sweep_context(grid, q, L, device=device, dtype=dtype, backend=backend)
    G = lib.num_groups
    iso = torch.zeros(G, 4, n, n, n, dtype=F64)
    iso[:, 0, :2, :2, :2] = torch.linspace(1.0, 0.4, G, dtype=F64)[:, None, None, None]
    iso[:, 1, :2, :2, :2] = 0.1  # x slope
    iso = iso.to(device=device, dtype=dtype)
    omega = ctx.quad.omega.to(device=device, dtype=dtype)

    def fc_source_fn(idx):
        aniso = 1.0 + 0.3 * omega.index_select(0, idx)[:, 0]  # positive, direction dependent
        return aniso[:, None, None, None, None, None] * iso[None]

    return ctx, mu_t, mu_s, fc_source_fn


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_source_iteration_converges_and_is_fixed_point(backend):
    """float64 PyTorch path (CPU), or the fused Triton path in float32 on CUDA (same tolerances: the relative
    change and the fixed-point residual are well above float32 rounding at tol = 1e-6)."""
    if backend == "torch":
        ctx, mu_t, mu_s, fc = _problem(backend="torch")
    else:
        _triton_or_skip()
        ctx, mu_t, mu_s, fc = _problem(device=torch.device("cuda"), dtype=torch.float32, backend="triton")
    tol = 1e-6
    seen = []
    res = source_iteration(ctx, mu_t, mu_s, fc, tol=tol, max_iterations=60,
                           callback=lambda n, phi_n, rel: seen.append((n, rel)))
    assert res.converged and res.num_iterations < 30, (res.num_iterations, res.history)
    assert len(res.history) == res.num_iterations == len(res.phi_norms) == len(seen)
    assert res.history[0] == 1.0 and res.history[-1] < tol
    h = res.history
    assert all(h[k + 1] < h[k] for k in range(2, len(h) - 1)), h
    assert res.phi_total.shape == (4, ctx.num_moments, 4) + ctx.shape
    # fixed point: sweep(scatter source of phi_total) + sweep(FC source) == phi_total
    T = multiple_collision_source_moments(mu_s, res.phi_total)
    assert T.shape == res.phi_total.shape
    phi_fp = sweep(ctx, mu_t, scatter_source_fn(ctx, T)) + sweep(ctx, mu_t, fc)
    rel = float((phi_fp - res.phi_total).norm() / res.phi_total.norm())
    assert rel < 10 * tol, rel
    # the scalar flux is positive and the cell means of the total fluence are physical
    assert float(res.phi_total[:, 0, 0].min()) > 0


def test_source_iteration_max_iterations():
    ctx, mu_t, mu_s, fc = _problem(h=4.0)  # optically thicker cells
    res = source_iteration(ctx, mu_t, mu_s, fc, tol=1e-12, max_iterations=3)
    assert res.num_iterations == 3 and not res.converged and len(res.history) == 3
    _, psi = sweep(ctx, mu_t, fc, return_psi=True)
    assert float(psi[:, :, 0].min()) < 0  # thick cells: the LD angular fluence has negative cell means
    seen = []
    res_c = source_iteration(ctx, mu_t, mu_s, fc, tol=1e-4, max_iterations=30,
                             callback=lambda n, phi_n, rel: seen.append(n))
    assert res_c.converged and seen == list(range(1, res_c.num_iterations + 1))
    assert bool(torch.isfinite(res_c.phi_total).all()) and float(res_c.phi_total[:, 0, 0].sum()) > 0.0


def test_source_iteration_device_float32(device):
    ctx64, mu_t64, mu_s64, fc64 = _problem(n=4)
    ref = source_iteration(ctx64, mu_t64, mu_s64, fc64, tol=1e-5, max_iterations=40)
    ctx, mu_t, mu_s, fc = _problem(device=device, dtype=torch.float32, n=4)
    res = source_iteration(ctx, mu_t, mu_s, fc, tol=1e-5, max_iterations=40, direction_chunk=2)
    assert res.phi_total.device.type == device.type and res.phi_total.dtype == torch.float32
    assert abs(res.num_iterations - ref.num_iterations) <= 1
    err = float((res.phi_total.cpu().double() - ref.phi_total).norm() / ref.phi_total.norm())
    assert err < 1e-4, err


def test_scatter_source_fn_matches_angular():
    from citrus.scatter_solver.angular_quadrature import scatter_source_at_directions, scatter_source_moments

    ctx, mu_t, mu_s, _ = _problem(n=3)
    phi = torch.randn(4, ctx.num_moments, 4, *ctx.shape, dtype=F64)
    T = scatter_source_moments(mu_s, phi)
    idx = torch.tensor([0, 5, 17, 31, 30])
    ref = scatter_source_at_directions(T, ctx.basis.Y[idx])
    out = scatter_source_fn(ctx, T)(idx)
    assert out.shape == (5, 4, 4) + ctx.shape
    assert torch.allclose(out, ref, rtol=1e-13, atol=1e-14)
    buf = torch.full((5, 4, 4) + ctx.shape, float("nan"), dtype=F64)  # `out=` (lent by the sweep) is written in place
    res = scatter_source_fn(ctx, T)(idx, out=buf)
    assert res.data_ptr() == buf.data_ptr()
    assert torch.allclose(buf, ref, rtol=1e-13, atol=1e-14)


def test_scatter_moments_matches_angular(device):
    from citrus.scatter_solver.angular_quadrature import scatter_source_moments
    from citrus.scatter_solver.source_iteration import scatter_moments

    torch.manual_seed(3)
    G, Lx, sp = 3, 4, (2, 3, 4)
    mu_s = torch.rand(G, G, Lx + 1, *sp, dtype=F64, device=device)
    mu_s[1, 0] = 0.0  # a zero (upscatter) block is skipped
    for rest in [(4,) + sp, sp]:
        phi = torch.randn(G, 9, *rest, dtype=F64, device=device)  # degree 2 <= Lx
        ref = scatter_source_moments(mu_s, phi)
        out = scatter_moments(mu_s, phi)
        assert out.shape == ref.shape and out.device == ref.device
        assert torch.allclose(out, ref, rtol=1e-13, atol=1e-14)
        assert out.movedim(1, 0).is_contiguous()
