"""Uncollided fluence and first-collision source (citrus/scatter_solver/uncollided.py): source collimation, Beer's
law, the first-collision source against a direct evaluation, LD restriction, the sampling correction and the fused
Triton kernel."""

import os

import numpy as np
import pytest
import torch

from citrus.scatter_solver.geometry import ConeGeometry, Grid
from citrus.scatter_solver.ray_tracing import SiddonProjectors
from citrus.scatter_solver.source import SourceModel
from citrus.scatter_solver.uncollided import (fc_source_at_directions, primary_directions_to_voxels,
                               restrict_fc_to_ld_moments, uncollided_fluence)
from citrus.scatter_solver.cross_sections import synthetic_library

F64 = torch.float64
GRID = Grid(shape=(4, 6, 5), delta_voxel=1.0, voxel_slice_aspect=0.8)


def wide_geom(**kw) -> ConeGeometry:
    base = dict(source_iso_dist=20.0, source_detector_dist=30.0, num_det_rows=10, num_det_channels=12,
                delta_det_row=2.0, delta_det_channel=2.0)
    base.update(kw)
    return ConeGeometry(**base)


# ---- source --------------------------------------------------------------------------------------

def test_source_angular_weight(device):
    geom = wide_geom(num_det_rows=4, num_det_channels=6, delta_det_row=1.0, delta_det_channel=1.0, det_channel_offset=0.5)
    src = SourceModel()
    for angle in (0.0, 0.8):
        s = np.array(geom.source_position(angle))
        u_hat, v_hat, n_hat = (np.array(a) for a in geom.panel_axes(angle))
        pc = np.array(geom.panel_center(angle))
        rng = np.random.default_rng(3)
        uv = rng.uniform(-5, 5, (200, 2))
        targets = pc[None] + uv[:, :1] * u_hat + uv[:, 1:] * v_hat
        dirs = targets - s
        dirs = np.concatenate([dirs, -dirs[:5], (u_hat + 0 * dirs[:1])])  # away from the panel / parallel
        w = src.angular_weight(torch.tensor(dirs, dtype=F64, device=device), geom, angle).cpu().numpy()
        hu, hv = geom.panel_half_extent()
        ref = np.concatenate([((np.abs(uv[:, 0] + 0.5) <= hu) & (np.abs(uv[:, 1]) <= hv)).astype(float), np.zeros(6)])
        np.testing.assert_array_equal(w, ref)
        assert 0 < ref.sum() < 200
    lib = synthetic_library(3)
    assert torch.equal(SourceModel().per_group_strength(lib), lib.spectrum_fraction)


# ---- uncollided fluence ---------------------------------------------------------------------------

def test_primary_directions_to_voxels(device):
    geom = wide_geom()
    om = primary_directions_to_voxels(GRID, geom, 0.3, device, F64)
    assert om.shape == (*GRID.shape, 3) and om.device.type == device.type
    r = GRID.center_coordinates(device, F64)
    r0 = torch.tensor(geom.source_position(0.3), dtype=F64, device=device)
    torch.testing.assert_close(torch.linalg.norm(om, dim=-1), torch.ones(GRID.shape, dtype=F64, device=device))
    torch.testing.assert_close(torch.cross(om, r - r0, dim=-1), torch.zeros_like(om), atol=1e-12, rtol=0)


def test_uncollided_fluence_vacuum_and_uniform(device):
    lib = synthetic_library(4)
    geom = wide_geom()
    P = SiddonProjectors()
    src = SourceModel()
    r = GRID.center_coordinates(dtype=F64).numpy()
    s = np.array(geom.source_position(0.0))
    dist = np.linalg.norm(r - s, axis=-1)
    f = lib.spectrum_fraction.numpy()

    rho0 = torch.zeros(2, *GRID.shape, dtype=F64, device=device)
    phi0 = uncollided_fluence(lib, rho0, GRID, geom, 0.0, src, P)
    assert phi0.shape == (4, *GRID.shape) and phi0.dtype == F64
    np.testing.assert_allclose(phi0.cpu().numpy(), f[:, None, None, None] / dist[None] ** 2, rtol=1e-13)

    rho = torch.zeros(2, *GRID.shape, dtype=F64, device=device)
    rho[0] = 1.0
    phi = uncollided_fluence(lib, rho, GRID, geom, 0.0, src, P).cpu().numpy()
    Y2 = GRID.extent[1] / 2
    path = dist * (Y2 - r[..., 1]) / (geom.source_iso_dist - r[..., 1])  # in-box path, entering via y = +Y/2
    mu = lib.mu_t[0].numpy()
    ref = f[:, None, None, None] / dist[None] ** 2 * np.exp(-mu[:, None, None, None] * path[None])
    np.testing.assert_allclose(phi, ref, rtol=1e-12)


def _uniform_water_path(geom: ConeGeometry, rho_w: float = 1.0):
    """(dist [R, C, S], in-box water path in g/cm^2 [R, C, S]) for a uniform water box at angle 0."""
    r = GRID.center_coordinates(dtype=F64).numpy()
    s = np.array(geom.source_position(0.0))
    dist = np.linalg.norm(r - s, axis=-1)
    Y2 = GRID.extent[1] / 2
    return dist, rho_w * dist * (Y2 - r[..., 1]) / (geom.source_iso_dist - r[..., 1])


def test_uncollided_fluence_polychromatic_beer_law(device):
    """(a) Within-group hardening: phi_uc[g] = sum_{E in g} w_E exp(-m(E) L) / r^2 exactly (float64), which exceeds the
    group-wise f_g exp(-mu_t,g L) / r^2 (Jensen: the group mean of exp(-m L) >= exp(-<m> L)); equal without spread."""
    geom = wide_geom()
    P = SiddonProjectors()
    rho = torch.zeros(2, *GRID.shape, dtype=F64, device=device)
    rho[0] = 1.3
    dist, path = _uniform_water_path(geom, 1.3)
    for G, n, spread in ((4, 6, 0.6), (1, 24, 0.8), (3, 5, 0.0)):
        lib = synthetic_library(G, num_fine_per_group=n, fine_spread=spread)
        phi = uncollided_fluence(lib, rho, GRID, geom, 0.0, SourceModel(), P)
        assert phi.shape == (G, *GRID.shape) and phi.dtype == F64
        phi = phi.cpu().numpy()
        w, m, fg = lib.fine_weight.numpy(), lib.fine_mu_t[0].numpy(), lib.fine_group.numpy()
        # analytic polychromatic Beer's law, one voxel at a time
        for idx in ((0, 0, 0), (2, 3, 1), (3, 5, 4)):
            for g in range(G):
                sel = fg == g
                ref = float(np.sum(w[sel] * np.exp(-m[sel] * path[idx]))) / dist[idx] ** 2
                assert phi[(g,) + idx] == pytest.approx(ref, rel=1e-12), (G, spread, idx, g)
        # vs the group-wise attenuation (the path this replaces)
        old = lib.spectrum_fraction.numpy()[:, None, None, None] / dist[None] ** 2 * np.exp(
            -lib.mu_t[0].numpy()[:, None, None, None] * path[None])
        if spread == 0.0:
            np.testing.assert_allclose(phi, old, rtol=1e-12)
        else:
            assert np.all(phi >= old * (1 - 1e-12)) and np.all(phi.sum(0) > old.sum(0) * (1 + 1e-4))
    # one group spanning the whole spectrum: phi_uc is the full fine-energy integral of the spectrum
    lib1 = synthetic_library(1, num_fine_per_group=24, fine_spread=0.8)
    phi1 = uncollided_fluence(lib1, rho, GRID, geom, 0.0, SourceModel(), P)[0].cpu().numpy()
    ref1 = np.einsum("e,e...->...", lib1.fine_weight.numpy(),
                     np.exp(-lib1.fine_mu_t[0].numpy()[:, None, None, None] * path[None])) / dist ** 2
    np.testing.assert_allclose(phi1, ref1, rtol=1e-12)


def test_uncollided_fluence_two_materials_and_blocks(device):
    """Water + bone line integrals combine linearly per fine energy; the result is independent of the evaluation
    block size of cross_sections.group_transmission; float32 agrees with float64."""
    from citrus.scatter_solver.cross_sections import group_transmission

    geom = wide_geom()
    P = SiddonProjectors()
    lib = synthetic_library(3, num_fine_per_group=5, fine_spread=0.5)
    g = torch.Generator().manual_seed(4)
    rho = torch.rand(2, *GRID.shape, generator=g, dtype=F64).to(device)
    phi = uncollided_fluence(lib, rho, GRID, geom, 0.2, SourceModel(), P)
    path = P.upstream_cumulative(rho, GRID, geom, 0.2)  # [2, R, C, S] g/cm^2
    tau = torch.einsum("me,m...->e...", lib.fine_mu_t.to(device), path)
    ref = torch.zeros_like(phi)
    for e in range(lib.num_fine_energies):
        ref[int(lib.fine_group[e])] += float(lib.fine_weight[e]) * torch.exp(-tau[e])
    rel = GRID.center_coordinates(device, F64) - torch.tensor(geom.source_position(0.2), dtype=F64, device=device)
    ref *= (SourceModel().angular_weight(rel, geom, 0.2) / (rel * rel).sum(-1))[None]
    torch.testing.assert_close(phi, ref, rtol=1e-12, atol=0)
    for block in (1, 7, 1000):
        torch.testing.assert_close(group_transmission(lib, path, block=block), group_transmission(lib, path),
                                   rtol=1e-13, atol=0)
    phi32 = uncollided_fluence(lib, rho.float(), GRID, geom, 0.2, SourceModel(), P)
    assert phi32.dtype == torch.float32
    torch.testing.assert_close(phi32.double(), phi, rtol=2e-5, atol=1e-7 * float(phi.max()))


def test_uncollided_fluence_collimation(device):
    lib = synthetic_library(2)
    geom = wide_geom(num_det_rows=2, num_det_channels=3, delta_det_row=1.5, delta_det_channel=1.5)
    P = SiddonProjectors()
    rho0 = torch.zeros(2, *GRID.shape, dtype=F64, device=device)
    phi = uncollided_fluence(lib, rho0, GRID, geom, 0.4, SourceModel(), P)
    rel = GRID.center_coordinates(device, F64) - torch.tensor(geom.source_position(0.4), dtype=F64, device=device)
    phi_open = lib.spectrum_fraction.to(device)[:, None, None, None] / (rel * rel).sum(-1)[None]  # vacuum, 1/r^2
    b = SourceModel().angular_weight(rel, geom, 0.4)
    torch.testing.assert_close(phi, phi_open * b)
    assert 0 < b.sum() < b.numel()


# ---- first-collision source -----------------------------------------------------------------------

def test_fc_source_hand_evaluation(device):
    lib = synthetic_library(4)
    geom = wide_geom()
    angle = 0.5
    g = torch.Generator().manual_seed(0)
    rho = torch.rand(2, *GRID.shape, dtype=F64, generator=g).to(device) + 0.1
    phi = torch.rand(4, *GRID.shape, dtype=F64, generator=g).to(device)
    om = torch.nn.functional.normalize(torch.randn(5, 3, dtype=F64, generator=g), dim=-1).to(device)
    fc = fc_source_at_directions(lib, rho, phi, GRID, geom, angle, om, chunk=2)
    assert fc.shape == (5, 4, *GRID.shape)
    fc1 = fc_source_at_directions(lib, rho, phi, GRID, geom, angle, om, chunk=5)
    torch.testing.assert_close(fc, fc1)

    i, j, k = 1, 4, 3
    r = GRID.center_coordinates(dtype=F64)[i, j, k].numpy()
    s = np.array(geom.source_position(angle))
    om_uc = (r - s) / np.linalg.norm(r - s)
    eg = lib.eta_grid.numpy()
    tab = lib.mu_s_eta.numpy()
    rv = rho[:, i, j, k].cpu().numpy()
    pv = phi[:, i, j, k].cpu().numpy()
    for kk in range(5):
        eta = float(om_uc @ om[kk].cpu().numpy())
        for go in range(4):
            tot = 0.0
            for gi in range(go + 1):
                for m in range(2):
                    tot += rv[m] * np.interp(eta, eg, tab[m, gi, go]) * pv[gi]
            assert abs(fc[kk, go, i, j, k].item() - tot) < 1e-13 * max(1.0, abs(tot))


# ---- F -> C restriction ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape,factor", [((4, 6, 8), 2), ((6, 6, 3), 3), ((4, 4, 4), 4)])
def test_restrict_reproduces_linear_fields(device, shape, factor):
    grid = Grid(shape=shape, delta_voxel=0.5, voxel_row_aspect=1.3)
    coarse = grid.coarsen(factor)
    g = torch.Generator().manual_seed(1)
    K, G = 2, 3
    coef = torch.randn(K, G, 4, *coarse.shape, dtype=F64, generator=g).to(device)  # [c, s_x, s_y, s_z]
    u = grid.local_coordinates_of_fine_in_coarse(factor, device, F64)
    R, C, S = grid.shape
    uy = u.repeat(R // factor)[:, None, None]
    ux = u.repeat(C // factor)[None, :, None]
    uz = u.repeat(S // factor)[None, None, :]

    def up(t):  # coarse [..., Rc, Cc, Sc] -> fine
        return t.repeat_interleave(factor, -3).repeat_interleave(factor, -2).repeat_interleave(factor, -1)

    field = up(coef[:, :, 0]) + up(coef[:, :, 1]) * ux + up(coef[:, :, 2]) * uy + up(coef[:, :, 3]) * uz
    mom = restrict_fc_to_ld_moments(field, grid, factor)
    assert mom.shape == (K, G, 4, *coarse.shape)
    torch.testing.assert_close(mom, coef, rtol=0, atol=1e-12)

    # mean moment conserves the fine-grid integral
    torch.testing.assert_close(mom[:, :, 0].sum((-3, -2, -1)) * factor ** 3, field.sum((-3, -2, -1)))


def test_restrict_factor_one(device):
    f = torch.rand(2, *GRID.shape, dtype=F64, device=device)
    m = restrict_fc_to_ld_moments(f, GRID, 1)
    assert m.shape == (2, 4, *GRID.shape)
    torch.testing.assert_close(m[:, 0], f)
    assert (m[:, 1:] == 0).all()
    with pytest.raises(ValueError):
        restrict_fc_to_ld_moments(f, GRID, 4)


def test_uncollided_chain_float32_matches_float64(device):
    lib = synthetic_library(3)
    geom = wide_geom()
    rho = torch.rand(2, *GRID.shape, dtype=F64, device=device)
    om = torch.nn.functional.normalize(torch.tensor([[0.0, -1.0, 0.1], [0.4, -0.8, 0.0]], dtype=F64, device=device), dim=-1)
    P = SiddonProjectors()
    out = {}
    for dt in (torch.float32, F64):
        phi = uncollided_fluence(lib, rho.to(dt), GRID, geom, 0.2, SourceModel(), P)
        fc = fc_source_at_directions(lib, rho.to(dt), phi, GRID, geom, 0.2, om.to(dt))
        out[dt] = fc
        assert phi.dtype == dt and fc.dtype == dt and fc.device.type == device.type
    torch.testing.assert_close(out[torch.float32].double(), out[F64], rtol=1e-4, atol=1e-9)


# ---- fused Triton FC source (kernels/uncollided_kernels.py) -------------------------------------------------------------

def _fc_kernel_or_skip() -> torch.device:
    from citrus.scatter_solver.kernels import uncollided_kernels

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    dev = torch.device("cuda")
    ok, why = uncollided_kernels.kernel_usable(dev)
    if not ok:
        pytest.skip(f"FC-source kernel unusable: {why}")
    return dev


def _assert_rel(a: torch.Tensor, b: torch.Tensor, rel: float) -> float:
    """max |a - b| <= rel * max |b| (the relative-to-field-maximum metric of the kernel docs); returns the ratio."""
    scale = float(b.abs().max())
    err = float((a.double() - b.double()).abs().max()) / max(scale, 1e-30)
    assert err <= rel, f"max rel deviation {err:.2e} > {rel:.0e}"
    return err


@pytest.mark.parametrize("G", [4, 6])
@pytest.mark.parametrize("K", [1, 8, 13])
def test_fc_source_triton_matches_torch(G, K):
    """Triton path == PyTorch path to 1e-5 (relative to the field maximum, and elementwise rtol 1e-5) on random
    inputs: rho = 0 voxels (one material and both), directions exactly parallel / antiparallel to Omega_UC
    (eta = +-1, the table ends), an anisotropic grid with a slice offset, and an oblique view angle."""
    dev = _fc_kernel_or_skip()
    lib = synthetic_library(G, legendre_order=1, num_eta=401)
    grid = Grid(shape=(6, 9, 7), delta_voxel=0.7, voxel_row_aspect=1.2, voxel_slice_aspect=0.9,
                recon_slice_offset=0.4)
    geom = wide_geom()
    angle = 0.7
    g = torch.Generator().manual_seed(G * 100 + K)
    rho = torch.rand(2, *grid.shape, generator=g) * 2.0
    rho[0, :2] = 0.0  # water absent
    rho[:, :, 3] = 0.0  # vacuum column
    rho[1, 4, 5, :] = 0.0
    phi = torch.rand(G, *grid.shape, generator=g) + 0.01
    om_uc = primary_directions_to_voxels(grid, geom, angle, "cpu", torch.float32)
    om = torch.nn.functional.normalize(torch.randn(K, 3, generator=g), dim=-1)
    if K >= 2:
        om[0], om[1] = om_uc[1, 2, 3], -om_uc[4, 6, 2]
    else:
        om[0] = om_uc[2, 2, 2]
    rho, phi, om = rho.to(dev), phi.to(dev), om.to(dev)
    ref = fc_source_at_directions(lib, rho, phi, grid, geom, angle, om, chunk=3, backend="torch")
    out = fc_source_at_directions(lib, rho, phi, grid, geom, angle, om, backend="triton")
    assert out.shape == ref.shape == (K, G, *grid.shape) and out.dtype == torch.float32
    _assert_rel(out, ref, 1e-5)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6 * float(ref.abs().max()))
    assert bool((out[:, :, :, 3] == 0).all())  # rho = 0 -> exactly 0
    out_auto = fc_source_at_directions(lib, rho, phi, grid, geom, angle, om)
    assert torch.equal(out_auto, out)  # "auto" picks the kernel on CUDA float32


def test_fc_source_triton_hand_evaluation():
    """The kernel against the float64 reference of test_fc_source_hand_evaluation (float32 tolerance)."""
    dev = _fc_kernel_or_skip()
    lib = synthetic_library(4)
    geom = wide_geom()
    angle = 0.5
    g = torch.Generator().manual_seed(0)
    rho = torch.rand(2, *GRID.shape, dtype=F64, generator=g) + 0.1
    phi = torch.rand(4, *GRID.shape, dtype=F64, generator=g)
    om = torch.nn.functional.normalize(torch.randn(5, 3, dtype=F64, generator=g), dim=-1)
    ref = fc_source_at_directions(lib, rho, phi, GRID, geom, angle, om)  # float64 CPU reference
    out = fc_source_at_directions(lib, rho.float().to(dev), phi.float().to(dev), GRID, geom, angle,
                                  om.float().to(dev), backend="triton")
    _assert_rel(out.cpu(), ref, 1e-6)


def test_fc_source_backend_dispatch(device):
    """float64 and CPU inputs use the PyTorch path under every backend; "torch" is the PyTorch path bit for bit;
    an unknown backend raises."""
    from citrus.scatter_solver.uncollided import _fc_source_torch

    lib = synthetic_library(3)
    geom = wide_geom()
    rho = torch.rand(2, *GRID.shape, dtype=F64, device=device)
    phi = torch.rand(3, *GRID.shape, dtype=F64, device=device)
    om = torch.nn.functional.normalize(torch.randn(4, 3, dtype=F64, device=device), dim=-1)
    ref = _fc_source_torch(lib, rho, phi, GRID, geom, 0.3, om, chunk=2)
    for be in ("auto", "torch", "triton"):
        assert torch.equal(fc_source_at_directions(lib, rho, phi, GRID, geom, 0.3, om, chunk=2, backend=be), ref)
    r32, p32, o32 = rho.float(), phi.float(), om.float()
    assert torch.equal(fc_source_at_directions(lib, r32, p32, GRID, geom, 0.3, o32, chunk=2, backend="torch"),
                       _fc_source_torch(lib, r32, p32, GRID, geom, 0.3, o32, chunk=2))
    with pytest.raises(ValueError):
        fc_source_at_directions(lib, rho, phi, GRID, geom, 0.3, om, backend="cuda")


def test_fc_source_triton_kill_switch(monkeypatch):
    """CITRUS_DISABLE_TRITON=1 (read by the shared probe) sends "auto" to the PyTorch path and makes "triton" raise."""
    from citrus.scatter_solver.kernels import uncollided_kernels
    from citrus.scatter_solver.kernels import triton_utils

    monkeypatch.setenv(triton_utils.DISABLE_ENV_VAR, "1")
    monkeypatch.setattr(triton_utils, "_PROBE", None)
    monkeypatch.setattr(uncollided_kernels, "_SELF_CHECK", {})
    ok, why = uncollided_kernels.kernel_usable(torch.device("cuda"))
    assert not ok and triton_utils.DISABLE_ENV_VAR in why
    if torch.cuda.is_available():
        x = torch.rand(3, *GRID.shape, device="cuda")
        assert not uncollided_kernels.use_kernel(x, "auto")
        with pytest.raises(RuntimeError):
            uncollided_kernels.use_kernel(x, "triton")
    # monkeypatch restores triton_utils._PROBE to its previous value (the real probe result) afterwards


def _fc_moment_sources(lib, monkeypatch) -> tuple[torch.Tensor, torch.Tensor]:
    """solver.fc_moment_source (all sweep directions, LD moments, with the conservative sampling correction) of the
    2 cm WLAB problem of test_scatter_solver.py, once with the Triton FC source and once with the PyTorch path (forced by
    patching the names that scatter_solver.py and uncollided.fc_sampling_correction resolve at call time)."""
    import functools

    from citrus.scatter_solver import scatter_solver as solver_mod
    from citrus.scatter_solver import uncollided as uncollided_mod
    from citrus.scatter_solver.config import SolverConfig
    from citrus.scatter_solver.detector import csi_response, no_grid
    from citrus.utilities import wlab_geometry
    from citrus.utilities import wlab_phantom
    from citrus.scatter_solver.scatter_solver import ScatterSolver

    dev = torch.device("cuda")
    grid = Grid((10, 16, 20), 2.0)
    cfg = SolverConfig(num_groups=lib.num_groups, legendre_order=3, num_polar=4, num_azimuthal=8, cone_num_polar=8,
                       cone_num_azimuthal=16, supervoxel_factor=2, direction_chunk=8)
    sol = ScatterSolver(lib, cfg, wlab_geometry(), SiddonProjectors(),
                       csi_response(), no_grid(), device=dev, dtype=torch.float32)
    rho = wlab_phantom(grid, device=dev, dtype=torch.float32)
    f = sol.material_fields(rho, grid)
    phi_uc = sol.uncollided(f, 0.0)
    ctx = sol.sweep_context(f.grid_coarse)
    idx = torch.arange(ctx.num_directions, device=dev)
    original = uncollided_mod.fc_source_at_directions
    out = {}
    for be in ("triton", "torch"):
        with monkeypatch.context() as mp:
            fn = functools.partial(original, backend=be)
            mp.setattr(uncollided_mod, "fc_source_at_directions", fn)
            mp.setattr(solver_mod, "fc_source_at_directions", fn)
            out[be] = sol.fc_moment_source(f, 0.0, phi_uc, ctx)(idx)
    return out["triton"], out["torch"]


def test_fc_moment_source_triton_synthetic(monkeypatch):
    """End to end through solver.fc_moment_source (restricted LD moments; the sampling correction is computed from
    the same backend): Triton == PyTorch to 1e-5 on the 2 cm WLAB problem, synthetic library."""
    _fc_kernel_or_skip()
    tri, ref = _fc_moment_sources(synthetic_library(4, 5), monkeypatch)
    assert tri.shape == ref.shape
    _assert_rel(tri, ref, 1e-5)


@pytest.mark.skipif(os.environ.get("CITRUS_SLOW_TESTS") != "1", reason="set CITRUS_SLOW_TESTS=1 (builds the "
                    "XrayPhysics library, cached in the library cache)")
def test_fc_moment_source_triton_real_library(monkeypatch):
    """As the synthetic test, with the real G = 6 library (2001-point eta table, Rayleigh forward peak)."""
    from citrus.scatter_solver.config import SolverConfig
    from citrus.scatter_solver.scatter_solver import build_solver_library

    _fc_kernel_or_skip()
    lib = build_solver_library(SolverConfig(num_groups=6, legendre_order=10))
    tri, ref = _fc_moment_sources(lib, monkeypatch)
    _assert_rel(tri, ref, 1e-5)
