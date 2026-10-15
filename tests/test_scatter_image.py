"""Collapsed-cone scatter image and primary image (citrus/scatter_solver/scatter_image.py): direction sets, the image
against a brute-force point-to-point integral, single-voxel bookkeeping, the sub-voxel emission, the direction
context, the combined first-order + multiple-collision pass, and the primary.

The oracle tests compare single rays per pixel, so the image functions are called with footprint=False here; the
finite-cone footprint is tested in tests/test_footprint.py."""

import functools
import math

import numpy as np
import pytest
import torch

from citrus.scatter_solver.angular_quadrature import Quadrature, hemisphere_quadrature, num_sh
from citrus.scatter_solver.config import SolverConfig
from citrus.scatter_solver.detector import (csi_response, detector_angle, energy_weighted_response, focused_1d_grid,
                                  ideal_counting_response, no_grid)
from citrus.scatter_solver.geometry import ConeGeometry, Grid
from citrus.utilities import wlab_setup
from citrus.scatter_solver.ray_tracing import SiddonProjectors
from citrus.scatter_solver.source import SourceModel
from citrus.scatter_solver.uncollided import uncollided_fluence
from citrus.scatter_solver.uncollided import fc_source_at_directions
from citrus.scatter_solver import scatter_image as _si
from citrus.scatter_solver.scatter_image import (coarsen_mu_t, covering_polar_angle, panel_direction_set,
                                                 primary_image, scatter_fraction, supervoxel_subpoint_emission)
from citrus.scatter_solver.cross_sections import mu_t_volume, synthetic_library

# single rays per pixel (no finite-cone footprint) in every oracle comparison of this module
combined_scatter_image = functools.partial(_si.combined_scatter_image, footprint=False)
scatter_image_from_source = functools.partial(_si.scatter_image_from_source, footprint=False)

F64 = torch.float64
PROJ = SiddonProjectors()


# ---- direction sets ---------------------------------------------------------------------------------

def test_panel_direction_set():
    geom = ConeGeometry(50.0, 60.0, 8, 8, 2.0, 2.0)
    grid = Grid((4, 4, 4), 1.0)
    for angle in (0.0, 1.1):
        n_hat = torch.tensor(geom.panel_axes(angle)[2], dtype=F64)
        q = panel_direction_set(geom, angle, (6, 12), dtype=F64)
        assert q.num_directions == 72
        assert float(q.weight.sum()) == pytest.approx(2 * math.pi, rel=1e-12)
        assert bool(((q.omega @ -n_hat) > 0).all())
        cover = covering_polar_angle(grid, geom, angle)
        q2 = panel_direction_set(geom, angle, (6, 12), max_polar_angle=grid, dtype=F64)
        assert float(q2.weight.sum()) == pytest.approx(2 * math.pi * (1 - math.cos(cover)), rel=1e-12)
        # brute-force check of the covering angle: random box point -> random panel point
        g = torch.Generator().manual_seed(0)
        lo, ext = torch.tensor(grid.lower_corner(), dtype=F64), torch.tensor(grid.extent, dtype=F64)
        # random points plus the vertices (box corners x panel corners); the max is at a vertex pair
        unit = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=F64)
        r = lo + ext * torch.cat([torch.rand(400, 3, generator=g, dtype=F64), unit])
        hu, hv = geom.panel_half_extent()
        uh, vh, _ = (torch.tensor(a, dtype=F64) for a in geom.panel_axes(angle))
        pc = torch.tensor(geom.panel_center(angle), dtype=F64)
        sq = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=F64)
        uv = torch.cat([2 * torch.rand(400, 2, generator=g, dtype=F64) - 1, sq]) * torch.tensor([hu, hv], dtype=F64)
        p = pc + uv[:, :1] * uh + uv[:, 1:] * vh
        d = p[:, None] - r[None]
        th = torch.arccos((d @ -n_hat) / torch.linalg.norm(d, dim=-1))
        assert float(th.max()) == pytest.approx(cover - 1e-6, abs=1e-9)
    # a SolverConfig gives its cone counts
    qc = panel_direction_set(geom, 0.0, SolverConfig(cone_num_polar=6, cone_num_azimuthal=12), dtype=F64)
    assert torch.equal(qc.omega, panel_direction_set(geom, 0.0, (6, 12), dtype=F64).omega)


# ---- (b) collapsed cone vs brute-force point-to-point sum ----------------------------

def _brute_force_fluence(E: torch.Tensor, mu: torch.Tensor, grid: Grid, geom: ConeGeometry, angle: float,
                         n_sub: int = 4) -> torch.Tensor:
    """sum_voxels sum_subpoints E(r) exp(-tau(r -> r_d)) / |r_d - r|^2 dV_sub at each pixel center. [G, Rd, Cd]."""
    dev, dt = E.device, E.dtype
    c = grid.center_coordinates(dev, dt).reshape(-1, 3)
    off = (torch.arange(n_sub, device=dev, dtype=dt) + 0.5) / n_sub - 0.5
    d = torch.tensor(grid.deltas, device=dev, dtype=dt)
    sub = torch.stack(torch.meshgrid(off, off, off, indexing="ij"), -1).reshape(-1, 3) * d
    pts = (c[:, None] + sub[None]).reshape(-1, 3)
    G = E.shape[0]
    Ev = E.reshape(G, -1).repeat_interleave(n_sub ** 3, dim=1)
    pix = geom.pixel_centers(angle, dev, dt).reshape(-1, 3)
    starts = pts[None].expand(pix.shape[0], -1, -1).reshape(-1, 3)
    ends = pix[:, None].expand(-1, pts.shape[0], -1).reshape(-1, 3)
    tau = PROJ.line_integrals(mu, starts, ends, grid).reshape(G, pix.shape[0], -1)
    r2 = ((ends - starts) ** 2).sum(-1).reshape(pix.shape[0], -1)
    bf = (Ev[:, None, :] * torch.exp(-tau) / r2).sum(-1) * grid.voxel_volume / n_sub ** 3
    return bf.reshape(G, *geom.image_shape)


def test_collapsed_cone_vs_brute_force():
    # float64 oracle on CPU only: SiddonProjectors.oblique_project loops over directions in Python, and
    # 2048 small float64 launches take > 1 min on a (shared) consumer GPU.
    device = torch.device("cpu")
    grid = Grid((4, 4, 4), 1.0)
    geom = ConeGeometry(50.0, 60.0, 8, 8, 2.0, 2.0)
    lib = synthetic_library(num_groups=2, legendre_order=1)
    x, y, z = grid.axis_coordinates(device, F64)
    # smooth density (half water: mu_t ~ 0.11-0.15 /cm) and smooth isotropic emission
    rho = torch.zeros(2, *grid.shape, device=device, dtype=F64)
    rho[0] = 0.5 * (1.0 + 0.1 * x[None, :, None] + 0.05 * y[:, None, None]) + 0 * z
    mu = mu_t_volume(lib, rho)
    E = torch.stack([1 + 0.2 * x[None, :, None] + 0.1 * z[None, None, :] + 0 * y[:, None, None],
                     0.5 + 0.1 * y[:, None, None] + 0 * x[None, :, None] + 0 * z[None, None, :]])
    angle = 0.0
    ref = _brute_force_fluence(E, mu, grid, geom, angle)
    errs = []
    for P, A in ((8, 16), (16, 32), (32, 64)):
        q = panel_direction_set(geom, angle, (P, A), max_polar_angle=grid, device=device, dtype=F64)
        out = scatter_image_from_source(lambda om: E.expand(om.shape[0], -1, -1, -1, -1), grid, geom, angle, q,
                                        mu, PROJ, ideal_counting_response(), no_grid(), lib, chunk=256)
        fl = out["fluence"]
        # counting detector, no grid: the signal is the PLANAR fluence (per-incident-photon responses)
        np.testing.assert_allclose(out["signal_per_group"].cpu().numpy(), out["planar_fluence"].cpu().numpy(),
                                   rtol=1e-12)
        assert bool((out["planar_fluence"] <= fl * (1 + 1e-12)).all())
        errs.append(float((fl - ref).norm() / ref.norm()))
    msg = "rel L2 error vs brute force for 8x16, 16x32, 32x64 directions: " + ", ".join(f"{e:.4f}" for e in errs)
    assert errs[0] > errs[1] > errs[2], msg
    assert errs[2] < 0.02, msg


# ---- (c) bookkeeping: single voxel in vacuum ------------------------------------------------------

@pytest.mark.parametrize("angle", [0.0, math.pi / 2])
def test_single_voxel_bookkeeping(device, angle):
    """Sum over pixels of A_pix * fluence = sum_k w_k E V / cos(theta_k) over the directions that hit.

    For a cube of side a = pixel pitch whose x- and z-edges (angle 0) or y- and z-edges (angle pi/2) are
    parallel to the panel, the chord-length shadow along Omega is a convolution of two pixel-wide boxes
    with an oblique segment, so the pixel-center lattice sum equals its integral V / cos(theta) exactly.
    """
    grid = Grid((3, 3, 3), 1.0)
    geom = ConeGeometry(20.0, 30.0, 40, 40, 1.0, 1.0)
    E0 = torch.tensor([2.0, 0.5], device=device, dtype=F64)
    field = torch.zeros(2, *grid.shape, device=device, dtype=F64)
    field[:, 1, 1, 1] = E0
    mu = torch.zeros_like(field)
    n_hat = torch.tensor(geom.panel_axes(angle)[2], device=device, dtype=F64)
    q = hemisphere_quadrature(-n_hat, 6, 12, math.radians(55.0), device=device, dtype=F64)
    # plus a direction leaving away from the panel and one whose shadow misses the panel
    u_hat = torch.tensor(geom.panel_axes(angle)[0], device=device, dtype=F64)
    extra = torch.stack([n_hat, -n_hat * math.cos(math.radians(87)) + u_hat * math.sin(math.radians(87))])
    quad = Quadrature.from_directions(torch.cat([q.omega, extra]), torch.cat([q.weight, torch.tensor([0.3, 0.2], device=device, dtype=F64)]))
    lib = synthetic_library(num_groups=2, legendre_order=1)
    out = scatter_image_from_source(lambda om: field.expand(om.shape[0], -1, -1, -1, -1), grid, geom, angle, quad,
                                    mu, PROJ, energy_weighted_response(), no_grid(), lib, chunk=16)
    total = out["fluence"].sum(dim=(1, 2)) * geom.pixel_area  # [G]
    cos = q.omega @ -n_hat
    expected = E0 * grid.voxel_volume * (q.weight / cos).sum()
    np.testing.assert_allclose(total.cpu().numpy(), expected.cpu().numpy(), rtol=1e-6)
    # planar fluence (photons crossing the panel): the 1/cos cancels, every emitted photon of a direction that
    # hits the panel is counted once: sum_k w_k E V
    planar_total = out["planar_fluence"].sum(dim=(1, 2)) * geom.pixel_area
    expected_planar = E0 * grid.voxel_volume * q.weight.sum()
    np.testing.assert_allclose(planar_total.cpu().numpy(), expected_planar.cpu().numpy(), rtol=1e-6)
    # energy-weighted detector: signal = E x planar fluence
    sig = out["signal"].sum() * geom.pixel_area
    assert float(sig) == pytest.approx(float((expected_planar * lib.group_energy.to(device)).sum()), rel=1e-6)


# ---- (d) WLAB first-order and multiple-collision images -----------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_wlab_first_order_and_mc_cuda():
    dev = torch.device("cuda")
    grid, geom, rho = wlab_setup(2.0, device=dev, dtype=torch.float32)
    lib = synthetic_library(num_groups=3, legendre_order=2).to(dev, torch.float32)
    angle = 0.3
    src = SourceModel()
    phi_uc = uncollided_fluence(lib, rho, grid, geom, angle, src, PROJ)
    q = panel_direction_set(geom, angle, (16, 32), max_polar_angle=grid, device=dev, dtype=torch.float32)
    det, grd = csi_response(), focused_1d_grid()
    out = _first_order(lib, rho, phi_uc, grid, 1, geom, angle, q, None, PROJ, det, grd, chunk=16)
    fl = out["fluence"]
    assert fl.device.type == "cuda" and fl.shape == (3, *geom.image_shape)
    assert torch.isfinite(fl).all() and (fl >= 0).all() and fl.sum() > 0
    sig = out["signal"]
    assert (sig >= 0).all() and (sig <= (fl * lib.group_energy[:, None, None]).sum(0) * (1 + 1e-5)).all()
    # smooth: neighbouring pixels differ by < 10 % of the image maximum (8 x 16 directions give ~13 %
    # ray effects on the 0.4 cm pixels; 16 x 32 give ~6 %)
    tot = fl.sum(0)
    jump = max(float((tot[1:] - tot[:-1]).abs().max()), float((tot[:, 1:] - tot[:, :-1]).abs().max()))
    assert jump < 0.1 * float(tot.max())
    out2 = _first_order(lib, rho, 2.5 * phi_uc, grid, 1, geom, angle, q, None, PROJ, det, grd, chunk=16)
    assert float(out2["signal"].sum()) == pytest.approx(2.5 * float(sig.sum()), rel=1e-5)

    # multiple-collision term (zero FC source) with an isotropic (n = 0 only) random source
    L = 2
    g = torch.Generator(device="cpu").manual_seed(1)
    T = torch.zeros(3, num_sh(L), *grid.shape)
    T[:, 0] = torch.rand(3, *grid.shape, generator=g) * 0.01
    T = T.to(dev)
    mu = mu_t_volume(lib, rho)
    mc = combined_scatter_image(lib, rho, torch.zeros_like(phi_uc), grid, 1, T, L, geom, angle, q, mu, PROJ, det, grd,
                                chunk=16)
    assert (mc["fluence"] >= 0).all() and (mc["signal"] >= 0).all() and mc["fluence"].sum() > 0
    # the SH plumbing: isotropic T is the same as the per-direction field T0 Y00
    iso = T[:, 0] / math.sqrt(4 * math.pi)
    ref = scatter_image_from_source(lambda om: iso.expand(om.shape[0], -1, -1, -1, -1), grid, geom, angle, q,
                                    mu, PROJ, det, grd, lib, chunk=16)
    np.testing.assert_allclose(mc["signal"].cpu().numpy(), ref["signal"].cpu().numpy(), rtol=1e-5,
                               atol=1e-6 * float(ref["signal"].max()))
    # the 6-D LD-moment tensor is sliced to its cell mean
    T6 = torch.stack([T, torch.randn_like(T), torch.randn_like(T), torch.randn_like(T)], dim=2)
    mc6 = combined_scatter_image(lib, rho, torch.zeros_like(phi_uc), grid, 1, T6, L, geom, angle, q, mu, PROJ, det,
                                 grd, chunk=16)
    np.testing.assert_allclose(mc6["signal"].cpu().numpy(), mc["signal"].cpu().numpy(), rtol=1e-6)


# ---- (e) primary and scatter fraction -----------------------------------------------------------------

def test_primary_uniform_box(device):
    grid = Grid((4, 4, 4), 1.0)
    geom = ConeGeometry(50.0, 60.0, 5, 5, 2.0, 2.0)
    lib = synthetic_library(num_groups=3, legendre_order=1)
    rho = torch.zeros(2, *grid.shape, device=device, dtype=F64)
    rho[0] = 1.0
    det, grd = csi_response(), focused_1d_grid(energies=lib.group_energy)
    out = primary_image(lib, rho, grid, geom, 0.0, SourceModel(), PROJ, det, grd)
    mu = lib.mu_t[0].to(device)  # water, rho = 1
    f = lib.spectrum_fraction.to(device)
    expected = f * torch.exp(-4.0 * mu) / 60.0 ** 2
    np.testing.assert_allclose(out["fluence"][:, 2, 2].cpu().numpy(), expected.cpu().numpy(), rtol=1e-12)
    eg = lib.group_energy.to(device)
    zero = torch.zeros((), device=device, dtype=F64)
    resp = det.at(eg, zero) * grd.at(eg, zero, zero)
    np.testing.assert_allclose(out["signal_per_group"][:, 2, 2].cpu().numpy(), (expected * resp).cpu().numpy(), rtol=1e-12)
    assert float(out["signal"][2, 2]) == pytest.approx(float((expected * resp).sum()), rel=1e-12)
    # corner pixel: Beer's law along the oblique ray, 1/r^2
    src = torch.tensor(geom.source_position(0.0), device=device, dtype=F64)
    pix = geom.pixel_centers(0.0, device, F64)[0, 0]
    tau = PROJ.line_integrals(mu_t_volume(lib, rho), src[None], pix[None], grid)[:, 0]
    exp_c = f * torch.exp(-tau) / ((pix - src) ** 2).sum()
    np.testing.assert_allclose(out["fluence"][:, 0, 0].cpu().numpy(), exp_c.cpu().numpy(), rtol=1e-12)
    # planar fluence = cos(theta0) x fluence; theta0 = angle of the primary ray to the panel normal
    cos_c = 60.0 / float(torch.linalg.norm(pix - src))
    np.testing.assert_allclose(out["planar_fluence"][:, 0, 0].cpu().numpy(), (cos_c * exp_c).cpu().numpy(),
                               rtol=1e-12)
    # signal = planar fluence x per-incident-photon responses
    om_c = (pix - src) / torch.linalg.norm(pix - src)
    theta0 = detector_angle(om_c, geom, 0.0)
    th1, th2 = _si.grid_angles(om_c, pix, geom, 0.0, grd.focal_distance_cm)
    resp = det.at(lib.group_energy.to(device), theta0) * grd.at(lib.group_energy.to(device), th1, th2)
    np.testing.assert_allclose(out["signal_per_group"][:, 0, 0].cpu().numpy(),
                               (cos_c * exp_c * resp).cpu().numpy(), rtol=1e-10)

    # scatter fraction with a single-scatter image of the same box
    phi_uc = uncollided_fluence(lib, rho, grid, geom, 0.0, SourceModel(), PROJ)
    q = panel_direction_set(geom, 0.0, (6, 12), max_polar_angle=grid, device=device, dtype=F64)
    sc = _first_order(lib, rho, phi_uc, grid, 1, geom, 0.0, q, None, PROJ, det, grd, chunk=36)
    sf = scatter_fraction(sc["signal"], out["signal"])
    assert sf.shape == geom.image_shape
    assert bool((sf >= 0).all()) and bool((sf < 1).all()) and float(sf.max()) > 0
    assert float(scatter_fraction(torch.zeros(1), torch.zeros(1))) == 0.0


def test_primary_fine_energy_subpixel(device):
    """Primary with within-group hardening (synthetic fine_spread) and sub-pixel sampling: the fine-energy transmission
    is formed per sub-ray and THEN averaged over the pixel (exact pixel average), against an independent evaluation
    from explicit sub-ray line integrals of the two density maps; n_sub=None takes the projector's own n_sub."""
    grid = Grid((4, 5, 6), 1.0, voxel_slice_aspect=0.8)
    geom = ConeGeometry(30.0, 45.0, 4, 5, 2.5, 2.0, det_row_offset=0.3, det_channel_offset=-0.4)
    lib = synthetic_library(num_groups=3, legendre_order=1, num_fine_per_group=5, fine_spread=0.6)
    g = torch.Generator().manual_seed(7)
    rho = torch.rand(2, *grid.shape, generator=g, dtype=F64).to(device)
    det, grd = ideal_counting_response(), no_grid()
    angle, n = 0.35, 3
    out = primary_image(lib, rho, grid, geom, angle, SourceModel(), PROJ, det, grd, n_sub=n)
    # reference: n x n midpoint lattice of sub-pixel centers, line integrals of rho, fine-energy sum, mean over sub-rays
    pix = geom.pixel_centers(angle, device, F64)  # [Rd, Cd, 3]
    uh, vh, _ = (torch.tensor(a, device=device, dtype=F64) for a in geom.panel_axes(angle))
    off = (2 * torch.arange(n, device=device, dtype=F64) + 1) / (2 * n) - 0.5
    shift = (off * geom.delta_det_row)[:, None, None] * vh + (off * geom.delta_det_channel)[None, :, None] * uh
    ends = (pix[:, :, None, None, :] + shift).reshape(-1, 3)
    src = torch.tensor(geom.source_position(angle), device=device, dtype=F64)
    path = PROJ.line_integrals(rho, src.expand_as(ends), ends, grid)  # [2, Rd Cd n n]
    tau = lib.fine_mu_t.to(device).t() @ path  # [E, Npts]
    T = torch.zeros(lib.num_groups, path.shape[1], device=device, dtype=F64)
    T.index_add_(0, lib.fine_group.to(device), lib.fine_weight.to(device)[:, None] * torch.exp(-tau))
    T = T.reshape(-1, *geom.image_shape, n * n).mean(-1)
    d = pix - src
    r2 = (d * d).sum(-1)
    ref = T * (SourceModel().angular_weight(d, geom, angle) / r2)[None]
    np.testing.assert_allclose(out["fluence"].cpu().numpy(), ref.cpu().numpy(), rtol=1e-12)
    cos0 = torch.cos(detector_angle(d / r2.sqrt()[..., None], geom, angle))
    np.testing.assert_allclose(out["planar_fluence"].cpu().numpy(), (ref * cos0).cpu().numpy(), rtol=1e-12)
    assert torch.equal(out["signal"], out["planar_fluence"].sum(0))  # ideal counting, no grid
    # the projector's own n_sub is the default (and is not applied twice)
    out_p = primary_image(lib, rho, grid, geom, angle, SourceModel(), SiddonProjectors(n_sub=n), det, grd)
    torch.testing.assert_close(out_p["fluence"], out["fluence"], rtol=1e-13, atol=0)
    out_1 = primary_image(lib, rho, grid, geom, angle, SourceModel(), SiddonProjectors(n_sub=n), det, grd, n_sub=1)
    assert not torch.allclose(out_1["fluence"], out["fluence"], rtol=1e-6)
    # hardening: exceeds the group-wise exp(-int mu_t,g dl) at the pixel centers (Jensen, per group)
    mu_path = PROJ.primary_project(mu_t_volume(lib, rho), grid, geom, angle)
    old = lib.spectrum_fraction.to(device)[:, None, None] * torch.exp(-mu_path) * (
        SourceModel().angular_weight(d, geom, angle) / r2)[None]
    assert bool((out_1["fluence"] >= old * (1 - 1e-12)).all()) and float(out_1["fluence"].sum()) > float(old.sum())


# ---- (f) scatter image on the supervoxel grid (sub-voxel quadrature points) --------------------------------------

IMAGE_KEYS = ("fluence", "planar_fluence", "signal_per_group", "signal")


def _small_fc_case(device, dtype=F64):
    grid = Grid((4, 4, 4), 1.0)
    geom = ConeGeometry(50.0, 60.0, 8, 8, 2.0, 2.0)
    lib = synthetic_library(num_groups=3, legendre_order=2).to(device, dtype)
    x, y, z = grid.axis_coordinates(device, dtype)
    rho = torch.zeros(2, *grid.shape, device=device, dtype=dtype)
    rho[0] = 0.8 + 0.05 * x[None, :, None] + 0.03 * y[:, None, None] + 0.02 * z[None, None, :]
    rho[1, 1:3, 1:3, 1:3] = 0.4
    phi_uc = uncollided_fluence(lib, rho, grid, geom, 0.3, SourceModel(), PROJ)
    q = panel_direction_set(geom, 0.3, (6, 12), max_polar_angle=grid, device=device, dtype=dtype)
    return grid, geom, lib, rho, phi_uc, q


def _zero_mc(lib, grid: Grid, f: int, device, dtype=F64) -> torch.Tensor:
    """A zero multiple-collision source T [G, 1, Rc, Cc, Sc] (L = 0): combined_scatter_image is then the first-order
    image alone."""
    return torch.zeros(lib.num_groups, 1, *grid.coarsen(f).shape, device=device, dtype=dtype)


def _first_order(lib, rho, phi_uc, grid, f, geom, angle, q, mu_c, P, det, grd, **kw):
    """First-order image of combined_scatter_image (zero MC source)."""
    return combined_scatter_image(lib, rho, phi_uc, grid, f, _zero_mc(lib, grid, f, phi_uc.device, phi_uc.dtype), 0,
                                  geom, angle, q, mu_c, P, det, grd, **kw)


def _fc_fn(lib, rho, phi_uc, grid, geom, angle):
    return lambda om: fc_source_at_directions(lib, rho, phi_uc, grid, geom, angle, om)


def _upsample(t: torch.Tensor, f: int) -> torch.Tensor:
    """Coarse [..., Rc, Cc, Sc] replicated on the f^3 fine voxels of each cell."""
    return t.repeat_interleave(f, -3).repeat_interleave(f, -2).repeat_interleave(f, -1)


def _linear_subpoint_tau(tau_c, mu_c, om, grid: Grid, f: int) -> torch.Tensor:
    """Independent (explicit-coordinate) form of the in-cell attenuation of `supervoxel_subpoint_emission`:
    tau_j = max(0, tau(cell) - mu(cell) Omega . (r_j - r_cell)) at every fine voxel center r_j, [k, G, R, C, S]."""
    coarse = grid.coarsen(f)
    rf = grid.center_coordinates(tau_c.device, tau_c.dtype)  # [R, C, S, 3]
    rc = _upsample(coarse.center_coordinates(tau_c.device, tau_c.dtype).movedim(-1, 0), f).movedim(0, -1)
    proj = torch.einsum("rcsx,kx->krcs", rf - rc, om.to(tau_c.dtype))  # [k, R, C, S]
    return (_upsample(tau_c, f) - _upsample(mu_c, f)[None] * proj[:, None]).clamp(min=0.0)


def _subpoint_image(fn, grid, f, geom, angle, q, mu_c, P, det, grd, **kw):
    """Reference of the sub-point projection: the fine-grid image of S_j exp(-tau_j) (tau_j from the coarse
    downstream trace by `_linear_subpoint_tau`) with no further attenuation."""
    coarse = grid.coarsen(f)

    def src(om):
        tau_c = P.downstream_cumulative(mu_c, coarse, geom, angle, om)
        return fn(om) * torch.exp(-_linear_subpoint_tau(tau_c, mu_c, om, grid, f))

    zero = torch.zeros(mu_c.shape[0], *grid.shape, device=mu_c.device, dtype=mu_c.dtype)
    return scatter_image_from_source(src, grid, geom, angle, q, zero, P, det, grd, **kw)


def test_first_order_factor1_is_single_grid(device):
    """factor 1: the sub-voxel points are the voxel centers, so the first-order term of the combined pass is the
    single-grid image of the FC source (any attenuation map; None -> the true attenuation)."""
    grid, geom, lib, rho, phi_uc, q = _small_fc_case(device)
    det, grd = csi_response(), focused_1d_grid(energies=lib.group_energy)
    mu = mu_t_volume(lib, rho)
    fc = _fc_fn(lib, rho, phi_uc, grid, geom, 0.3)
    ref = scatter_image_from_source(fc, grid, geom, 0.3, q, 0.9 * mu, PROJ, det, grd, lib, chunk=16)
    out = _first_order(lib, rho, phi_uc, grid, 1, geom, 0.3, q, 0.9 * mu, PROJ, det, grd, chunk=16)
    assert set(out) == set(IMAGE_KEYS)
    _assert_images_close(out, ref, msg="factor 1")
    ref_true = scatter_image_from_source(fc, grid, geom, 0.3, q, mu, PROJ, det, grd, lib, chunk=16)
    out_true = _first_order(lib, rho, phi_uc, grid, 1, geom, 0.3, q, None, PROJ, det, grd, chunk=16)
    np.testing.assert_allclose(out_true["signal"].cpu().numpy(), ref_true["signal"].cpu().numpy(), rtol=1e-6)


def test_subpoint_emission(device):
    """supervoxel_subpoint_emission against an explicit per-point loop and against the explicit-coordinate form;
    the f = 2 first-order image equals the fine-grid image of the sub-point emission; bad offsets raise."""
    grid, geom, lib, rho, phi_uc, q = _small_fc_case(device)
    det, grd = csi_response(), no_grid()
    f = 2
    coarse = grid.coarsen(f)
    mu_c = coarsen_mu_t(mu_t_volume(lib, rho), f)
    om = q.omega[:5].to(device, F64)
    S = fc_source_at_directions(lib, rho, phi_uc, grid, geom, 0.3, om)  # [5, G, 4, 4, 4]
    tau = PROJ.downstream_cumulative(mu_c, coarse, geom, 0.3, om)  # [5, G, 2, 2, 2]
    e_sub = supervoxel_subpoint_emission(S, tau, mu_c, om, grid, f)  # [5, G, 4, 4, 4]
    rf = grid.center_coordinates(device, F64)
    rc = coarse.center_coordinates(device, F64)
    expected = torch.zeros_like(S)
    for i in range(4):
        for j in range(4):
            for m in range(4):
                ci, cj, cm = i // f, j // f, m // f
                d = rf[i, j, m] - rc[ci, cj, cm]
                t = (tau[:, :, ci, cj, cm] - mu_c[None, :, ci, cj, cm] * (om @ d)[:, None]).clamp(min=0)
                expected[:, :, i, j, m] = S[:, :, i, j, m] * torch.exp(-t)
    np.testing.assert_allclose(e_sub.cpu().numpy(), expected.cpu().numpy(), rtol=1e-12)
    ref = S * torch.exp(-_linear_subpoint_tau(tau, mu_c, om, grid, f))
    np.testing.assert_allclose(e_sub.cpu().numpy(), ref.cpu().numpy(), rtol=1e-12)
    # factor 1: S exp(-tau)
    mu = mu_t_volume(lib, rho)
    tau1 = PROJ.downstream_cumulative(mu, grid, geom, 0.3, om)
    torch.testing.assert_close(supervoxel_subpoint_emission(S, tau1, mu, om, grid, 1), S * torch.exp(-tau1))
    with pytest.raises(ValueError, match="offsets"):
        supervoxel_subpoint_emission(S, tau, mu_c, om, grid, f, offsets=torch.zeros(5, 1, 1, 1, device=device))
    # the image: the fine-grid projection of the sub-point emission
    out = _first_order(lib, rho, phi_uc, grid, f, geom, 0.3, q, mu_c, PROJ, det, grd, chunk=16)
    ref_img = _subpoint_image(_fc_fn(lib, rho, phi_uc, grid, geom, 0.3), grid, f, geom, 0.3, q, mu_c, PROJ, det, grd,
                              lib=lib, chunk=16)
    for k in IMAGE_KEYS:
        np.testing.assert_allclose(out[k].cpu().numpy(), ref_img[k].cpu().numpy(), rtol=1e-10, err_msg=k)


def _slab_case(device, mu_value: float):
    """Slab of uniform attenuation mu_value (group 1: 1.5x) on a 0.5 cm grid, smooth Gaussian emission, 25 deg cone."""
    grid = Grid((8, 24, 24), 0.5)
    geom = ConeGeometry(50.0, 60.0, 24, 24, 1.0, 1.0)
    mu = torch.full((2, *grid.shape), mu_value, device=device, dtype=F64)
    mu[1] *= 1.5
    x, y, z = grid.axis_coordinates(device, F64)
    gauss = torch.exp(-(x[None, :, None] ** 2 + z[None, None, :] ** 2) / (2 * 1.5 ** 2)) + 0 * y[:, None, None]
    E = torch.stack([gauss * (1 + 0.1 * y[:, None, None]), 0.5 * gauss])
    q = panel_direction_set(geom, 0.0, (8, 16), max_polar_angle=math.radians(25.0), device=device, dtype=F64)
    return grid, geom, mu, E, q


def test_subpoint_projection_converges(device):
    """Smooth emission, 2 cm supervoxels on a 0.5 cm fine grid (64 sub-points): projecting the sub-points with the
    linear in-cell attenuation reproduces the fine-grid image exactly in vacuum and closely in a uniform slab (exact
    there up to rays leaving through a lateral face)."""
    f = 4
    kw = dict(group_energy=torch.tensor([60.0, 40.0]), chunk=64)
    det, grd = ideal_counting_response(), no_grid()
    err, bias = {}, {}
    for case, mu_value in (("vacuum", 0.0), ("slab", 0.4)):
        grid, geom, mu, E, q = _slab_case(device, mu_value)

        def fn(om):
            return E.expand(om.shape[0], -1, -1, -1, -1)

        ref = scatter_image_from_source(fn, grid, geom, 0.0, q, mu, PROJ, det, grd, **kw)["fluence"]
        fl = _subpoint_image(fn, grid, f, geom, 0.0, q, coarsen_mu_t(mu, f), PROJ, det, grd, **kw)["fluence"]
        err[case] = float((fl - ref).norm() / ref.norm())
        bias[case] = float(fl.sum() / ref.sum() - 1)
    msg = "; ".join(f"{c}: rel L2 {err[c]:.2e}, sum {bias[c]:+.2e}" for c in err)
    # measured (CPU and CUDA, rel L2 / panel-sum bias): vacuum 0 / 0; slab 3.7e-3 / -1.5e-3
    assert err["vacuum"] < 1e-12, msg
    assert err["slab"] < 1e-2 and abs(bias["slab"]) < 4e-3, msg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_first_order_wlab_cuda():
    """WLAB 2 cm (padded), synthetic library, float32 CUDA, the default (fused) Siddon projectors: factor 1 equals the
    single-grid image to 1e-6; factor 2 (4 cm cells) stays non-negative and close to it."""
    from citrus.utilities import wlab_phantom

    dev = torch.device("cuda")
    geom = wlab_setup(2.0, device=dev, dtype=torch.float32)[1]
    grid = Grid((10, 16, 20), 2.0)  # padded along x to be divisible by 2
    rho = wlab_phantom(grid, device=dev, dtype=torch.float32)
    lib = synthetic_library(num_groups=3, legendre_order=2).to(dev, torch.float32)
    angle = 0.3
    phi_uc = uncollided_fluence(lib, rho, grid, geom, angle, SourceModel(), PROJ)
    q = panel_direction_set(geom, angle, (8, 16), max_polar_angle=grid, device=dev, dtype=torch.float32)
    det, grd = csi_response(), focused_1d_grid()
    mu = mu_t_volume(lib, rho)
    P = SiddonProjectors()
    ref = scatter_image_from_source(_fc_fn(lib, rho, phi_uc, grid, geom, angle), grid, geom, angle, q, mu, P, det, grd,
                                    lib, chunk=8)
    one = _first_order(lib, rho, phi_uc, grid, 1, geom, angle, q, mu, P, det, grd, chunk=8)
    _assert_images_close(one, ref, msg="factor 1")
    two = _first_order(lib, rho, phi_uc, grid, 2, geom, angle, q, coarsen_mu_t(mu, 2), P, det, grd, chunk=8)
    assert float(two["signal"].min()) >= 0 and float(two["fluence"].min()) >= 0
    e = float((two["signal"] - ref["signal"]).norm() / ref["signal"].norm())
    b = float(two["signal"].sum() / ref["signal"].sum() - 1)
    # measured (8x16): 0.174 / +7.1 %. 4 cm cells do not resolve the inserts, and the coarse mu_t under-attenuates the
    # emission of the denser sub-voxels of a mixed cell
    assert e < 0.25 and abs(b) < 0.12, (e, b)


# ---- direction-set context (per-view precomputation of the response weights) -------------------------------

def _reference_cone_image(source_fn, grid, geom, angle, quad, mu, projectors, detector, grid_response, e_g, chunk):
    """The plain accumulation loop of the collapsed-cone image (responses evaluated per chunk on the full panel)."""
    from citrus.scatter_solver.detector import detector_angle, grid_angles

    dev, dt = mu.device, mu.dtype
    G = mu.shape[0]
    om_all, w_all = quad.omega.to(dev, dt), quad.weight.to(dev, dt)
    n = torch.tensor(geom.panel_axes(angle)[2], device=dev, dtype=dt)
    cos_all = (om_all * -n).sum(-1)
    keep = cos_all > 0
    om_all, w_all, cos_all = om_all[keep], w_all[keep], cos_all[keep]
    pix = geom.pixel_centers(angle, dev, dt)
    out = {k: torch.zeros(G, *pix.shape[:2], device=dev, dtype=dt)
           for k in ("fluence", "planar_fluence", "signal_per_group")}
    for s in range(0, om_all.shape[0], chunk):
        om, w, c = om_all[s:s + chunk], w_all[s:s + chunk], cos_all[s:s + chunk]
        tau = projectors.downstream_cumulative(mu, grid, geom, angle, om)
        img = projectors.oblique_project(torch.exp(-tau) * source_fn(om), grid, geom, angle, om)
        img = img * w[:, None, None, None]
        out["fluence"] += img.sum(0)
        out["planar_fluence"] += torch.einsum("kgrc,k->grc", img, c)
        d = detector.at(e_g, detector_angle(om, geom, angle)) * c
        th1, th2 = grid_angles(om, pix, geom, angle, grid_response.focal_distance_cm)
        out["signal_per_group"] += torch.einsum("kgrc,gkrc,gk->grc", img, grid_response.at(e_g, th1, th2), d)
    out["signal"] = out["signal_per_group"].sum(0)
    return out


def _assert_images_close(got, ref, rtol=1e-6, msg=""):
    for k in IMAGE_KEYS:
        np.testing.assert_allclose(got[k].cpu().numpy(), ref[k].cpu().numpy(), rtol=rtol,
                                   atol=rtol * float(ref[k].abs().max()), err_msg=f"{msg} {k}")


def test_direction_context_matches_reference(device):
    """Context path (built on the fly, prebuilt and factorized, prebuilt per-pixel) == the pre-context loop with
    the full per-pixel responses, focused grid, oblique view; the factorized grid response is separable."""
    from citrus.scatter_solver.scatter_image import make_direction_context

    grid, geom, lib, rho, phi_uc, q = _small_fc_case(device)
    det, grd = csi_response(), focused_1d_grid(energies=lib.group_energy)
    mu = mu_t_volume(lib, rho)
    x, y, z = grid.axis_coordinates(device, F64)
    S = torch.stack([1 + 0.2 * x[None, :, None] + 0.1 * z[None, None, :] + 0 * y[:, None, None]] * 3)

    def src(om):
        return S[None] * (1 + 0.3 * om[:, 2])[:, None, None, None, None]

    ref = _reference_cone_image(src, grid, geom, 0.3, q, mu, PROJ, det, grd, lib.group_energy, 5)
    fly = scatter_image_from_source(src, grid, geom, 0.3, q, mu, PROJ, det, grd, lib, chunk=5)
    _assert_images_close(fly, ref, msg="on the fly")
    for factorize in (True, False):
        ctx = make_direction_context(q, geom, 0.3, det, grd, lib, device, F64, chunk=5, factorize=factorize)
        assert ctx.separable == factorize and ctx.num_directions == q.num_directions
        assert (ctx.grid_weight is None) != factorize
        if factorize:
            assert ctx.acc_weight.shape == (lib.num_groups * geom.num_det_rows, 3, q.num_directions)
        # the context's chunking replaces the chunk argument
        got = scatter_image_from_source(src, grid, geom, 0.3, q, mu, PROJ, det, grd, lib, chunk=99, context=ctx)
        _assert_images_close(got, ref, msg=f"factorize={factorize}")


def test_direction_context_all_images(device):
    """The combined image (factors 1 and 2) with a shared context == without (1e-6); the sub-cell
    offsets and the harmonics are cached on the context; the context is checked against the call."""
    from citrus.scatter_solver.scatter_image import make_direction_context

    grid, geom, lib, rho, phi_uc, q = _small_fc_case(device)
    det, grd = csi_response(), focused_1d_grid(energies=lib.group_energy)
    mu = mu_t_volume(lib, rho)
    ctx = make_direction_context(q, geom, 0.3, det, grd, lib, device, F64, chunk=7)
    L = 2
    for f in (1, 2):
        g = torch.Generator(device="cpu").manual_seed(3 + f)
        T = (torch.rand(3, num_sh(L), *grid.coarsen(f).shape, generator=g, dtype=F64) * 0.01).to(device)
        a = combined_scatter_image(lib, rho, phi_uc, grid, f, T, L, geom, 0.3, q, None, PROJ, det, grd, chunk=7)
        b = combined_scatter_image(lib, rho, phi_uc, grid, f, T, L, geom, 0.3, q, None, PROJ, det, grd, ctx)
        _assert_images_close(b, a, msg=f"f={f}")
    assert ("sh", L) in ctx._cache and any(k[0] == "subcell" for k in ctx._cache)

    def zero_source(om):
        return torch.zeros(om.shape[0], 3, *grid.shape, device=device, dtype=F64)

    # mismatched calls are refused
    for bad in ({"angle": 0.4}, {"detector": csi_response()}, {"grid_response": no_grid()}):
        kw = {"angle": 0.3, "detector": det, "grid_response": grd}
        kw.update(bad)
        with pytest.raises(ValueError, match="does not match"):
            scatter_image_from_source(zero_source, grid, geom, kw["angle"], q, mu, PROJ, kw["detector"],
                                      kw["grid_response"], lib, context=ctx)
    with pytest.raises(ValueError, match="dtype"):
        scatter_image_from_source(zero_source, grid, geom, 0.3, q, mu.float(), PROJ, det, grd, lib, context=ctx)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_direction_context_chunk_loop_no_sync():
    """With a prebuilt context, the whole chunk loop of the scatter images (the combined image with the fused FC
    source, a plain source) runs without any host-device synchronization."""
    from citrus.utilities import wlab_phantom
    from citrus.scatter_solver.kernels import ray_tracing_kernels
    from citrus.scatter_solver.scatter_image import make_direction_context

    dev = torch.device("cuda")
    if not ray_tracing_kernels.kernels_usable(dev)[0]:
        pytest.skip("Triton kernels unusable on this machine")
    geom = wlab_setup(2.0, device=dev, dtype=torch.float32)[1]
    grid = Grid((10, 16, 20), 2.0)  # padded along x to be divisible by 2
    rho = wlab_phantom(grid, device=dev, dtype=torch.float32)
    lib = synthetic_library(num_groups=3, legendre_order=2).to(dev, torch.float32)
    angle = 0.3
    phi_uc = uncollided_fluence(lib, rho, grid, geom, angle, SourceModel(), PROJ)
    q = panel_direction_set(geom, angle, (4, 8), max_polar_angle=grid, device=dev, dtype=torch.float32)
    det, grd = csi_response(), focused_1d_grid()
    P = SiddonProjectors()
    mu = mu_t_volume(lib, rho)
    coarse = grid.coarsen(2)
    mu_c = coarsen_mu_t(mu, 2)
    L = 2
    T = torch.rand(3, num_sh(L), *coarse.shape, device=dev) * 0.01
    S = torch.rand(1, 3, *coarse.shape, device=dev)
    ctx = make_direction_context(q, geom, angle, det, grd, lib, dev, torch.float32, chunk=8)

    def images():
        return (combined_scatter_image(lib, rho, phi_uc, grid, 2, T, L, geom, angle, q, mu_c, P, det, grd, ctx),
                scatter_image_from_source(lambda om: S.expand(om.shape[0], -1, -1, -1, -1), coarse, geom, angle, q,
                                          mu_c, P, det, grd, lib, context=ctx))

    ref = images()  # compiles; the direction storage of the context is read to the host once here
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")  # any stream synchronization inside the calls raises
    try:
        got = images()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    for a, b in zip(got, ref):
        _assert_images_close(a, b, rtol=1e-5, msg="no-sync rerun")


def test_direction_context_memory_wlab(capsys):
    """Size of the context of the WLAB 2048-direction set (32 x 64 in the covering cone, G = 6, 75 x 100 panel,
    focused grid, float32): the factorized weights are [G * rows, 3, K] (+ [K, G, rows] tables), not per pixel."""
    from citrus.scatter_solver.scatter_image import make_direction_context

    grid, geom, _ = wlab_setup(1.0, dtype=torch.float32)
    q = panel_direction_set(geom, 0.0, (32, 64), max_polar_angle=grid, dtype=torch.float32)
    lib = synthetic_library(num_groups=6, legendre_order=1)
    ctx = make_direction_context(q, geom, 0.0, csi_response(), focused_1d_grid(), lib, "cpu", torch.float32)
    K, G, (Rd, Cd) = ctx.num_directions, ctx.num_groups, ctx.image_shape
    per_pixel = K * G * Rd * Cd * 4
    msg = (f"WLAB direction context: K = {K} directions, G = {G}, panel {Rd} x {Cd}: {ctx.nbytes / 2 ** 20:.1f} MiB "
           f"(fused weights {ctx.acc_weight.numel() * 4 / 2 ** 20:.1f} MiB; one per-pixel [K, G, rows, channels] "
           f"table would be {per_pixel / 2 ** 20:.0f} MiB)")
    with capsys.disabled():
        print("\n" + msg)
    assert K == 2048 and ctx.separable, msg
    assert ctx.nbytes < 32 * 2 ** 20, msg


# ---- combined first-order + multiple-collision pass ---------------------------------------------------------

def _broadcast_mc_source(T, L, f):
    """Source on the fine grid of the MC emission S_MC = Y(Omega) . T replicated on the f^3 sub-points of its
    supervoxel."""
    from citrus.scatter_solver.angular_quadrature import real_spherical_harmonics, scatter_source_at_directions

    def src(om):
        s = scatter_source_at_directions(T, real_spherical_harmonics(L, om))  # [k, G, Rc, Cc, Sc]
        return _upsample(s, f)

    return src


def _combined_references(lib, rho, phi_uc, grid, f, T, L, geom, angle, q, mu_c, P, det, grd, ctx):
    """(first-order, multiple-collision) references of the combined pass, each the fine-grid image of its sub-point
    emission (`_subpoint_image`: explicit-coordinate in-cell attenuation, generic single-grid image)."""
    kw = dict(lib=lib, context=ctx)
    fo = _subpoint_image(_fc_fn(lib, rho, phi_uc, grid, geom, angle), grid, f, geom, angle, q, mu_c, P, det, grd, **kw)
    mc = _subpoint_image(_broadcast_mc_source(T, L, f), grid, f, geom, angle, q, mu_c, P, det, grd, **kw)
    return fo, mc


def test_combined_scatter_image_small(device):
    """float64, Siddon, focused grid, oblique view, factors 1 and 2: combined == first-order + MC reference; the image
    of each source alone (the other one zero) is its reference term; at factor 1 the MC term is the single-grid image
    of the MC source; a 6-D T is sliced to its LD mean; bad arguments raise."""
    from citrus.scatter_solver.angular_quadrature import real_spherical_harmonics, scatter_source_at_directions
    from citrus.scatter_solver.scatter_image import make_direction_context

    grid, geom, lib, rho, phi_uc, q = _small_fc_case(device)
    det, grd = csi_response(), focused_1d_grid(energies=lib.group_energy)
    mu = mu_t_volume(lib, rho)
    L = 2
    ctx = make_direction_context(q, geom, 0.3, det, grd, lib, device, F64, chunk=7)
    for f in (1, 2):
        coarse = grid.coarsen(f)
        g = torch.Generator(device="cpu").manual_seed(5 + f)
        T = (torch.rand(3, num_sh(L), *coarse.shape, generator=g, dtype=F64) * 0.01).to(device)
        T[:, 1:] *= 0.3
        mu_c = coarsen_mu_t(0.9 * mu, f)
        msg = f"f={f}"
        fo, mc = _combined_references(lib, rho, phi_uc, grid, f, T, L, geom, 0.3, q, mu_c, PROJ, det, grd, ctx)
        tot = combined_scatter_image(lib, rho, phi_uc, grid, f, T, L, geom, 0.3, q, mu_c, PROJ, det, grd, ctx)
        assert set(tot) == set(IMAGE_KEYS)
        _assert_images_close(tot, {k: fo[k] + mc[k] for k in IMAGE_KEYS}, msg=msg)
        fo_only = combined_scatter_image(lib, rho, phi_uc, grid, f, torch.zeros_like(T), L, geom, 0.3, q, mu_c, PROJ,
                                         det, grd, ctx)
        mc_only = combined_scatter_image(lib, rho, torch.zeros_like(phi_uc), grid, f, T, L, geom, 0.3, q, mu_c, PROJ,
                                         det, grd, ctx)
        _assert_images_close(fo_only, fo, msg=msg + " FC alone")
        _assert_images_close(mc_only, mc, msg=msg + " MC alone")
        if f == 1:  # the MC source attenuated at the cell centers: the single-grid image of S_MC
            ref = scatter_image_from_source(lambda om: scatter_source_at_directions(T, real_spherical_harmonics(L, om)),
                                            coarse, geom, 0.3, q, mu_c, PROJ, det, grd, lib, context=ctx)
            _assert_images_close(mc_only, ref, msg=msg + " MC single grid")
    # built-in context, 6-D T, mu_t_coarse None (true coarse attenuation), argument checks
    coarse = grid.coarsen(2)
    T = (torch.rand(3, num_sh(L), *coarse.shape, dtype=F64) * 0.01).to(device)
    T6 = torch.stack([T, torch.randn_like(T), torch.randn_like(T), torch.randn_like(T)], dim=2)
    a = combined_scatter_image(lib, rho, phi_uc, grid, 2, T6, L, geom, 0.3, q, None, PROJ, det, grd, chunk=5)
    b = combined_scatter_image(lib, rho, phi_uc, grid, 2, T, L, geom, 0.3, q, coarsen_mu_t(mu, 2), PROJ, det, grd, ctx)
    _assert_images_close(a, b, msg="6-D T / no context")
    with pytest.raises(ValueError, match="moments"):
        combined_scatter_image(lib, rho, phi_uc, grid, 2, T, 3, geom, 0.3, q, None, PROJ, det, grd, ctx)
    with pytest.raises(ValueError, match="T_mean"):
        combined_scatter_image(lib, rho, phi_uc, grid, 2, T[:, :, :1], L, geom, 0.3, q, None, PROJ, det, grd, ctx)
    with pytest.raises(ValueError, match="does not match"):
        combined_scatter_image(lib, rho, phi_uc, grid, 2, T, L, geom, 0.4, q, None, PROJ, det, grd, ctx)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_combined_scatter_image_wlab_cuda():
    """WLAB 2 cm (padded), synthetic library, float32 CUDA, the default (fused) Siddon projectors, factors 1 and 2:
    the combined image equals its first-order + MC reference to 1e-6 (every key), and each source alone its term."""
    from citrus.utilities import wlab_phantom
    from citrus.scatter_solver.scatter_image import make_direction_context

    dev = torch.device("cuda")
    geom = wlab_setup(2.0, device=dev, dtype=torch.float32)[1]
    grid = Grid((10, 16, 20), 2.0)  # padded along x to be divisible by 2
    rho = wlab_phantom(grid, device=dev, dtype=torch.float32)
    lib = synthetic_library(num_groups=3, legendre_order=2).to(dev, torch.float32)
    angle = 0.3
    phi_uc = uncollided_fluence(lib, rho, grid, geom, angle, SourceModel(), PROJ)
    q = panel_direction_set(geom, angle, (8, 16), max_polar_angle=grid, device=dev, dtype=torch.float32)
    det, grd = csi_response(), focused_1d_grid()
    mu = mu_t_volume(lib, rho)
    L = 2
    P = SiddonProjectors()
    ctx = make_direction_context(q, geom, angle, det, grd, lib, dev, torch.float32, chunk=8)
    for f in (1, 2):
        coarse = grid.coarsen(f)
        mu_c = coarsen_mu_t(mu, f)
        # a smooth MC source (the multiple-collision source follows the density), weakly anisotropic
        rc = torch.nn.functional.avg_pool3d(rho[:1].unsqueeze(0), f, f).squeeze(0)[0]
        T = torch.zeros(3, num_sh(L), *coarse.shape, device=dev)
        T[:, 0] = 0.01 * rc
        T[:, 2] = 0.002 * rc
        msg = f"f={f}"
        fo, mc = _combined_references(lib, rho, phi_uc, grid, f, T, L, geom, angle, q, mu_c, P, det, grd, ctx)
        tot = combined_scatter_image(lib, rho, phi_uc, grid, f, T, L, geom, angle, q, mu_c, P, det, grd, ctx)
        _assert_images_close(tot, {k: fo[k] + mc[k] for k in IMAGE_KEYS}, msg=msg)
        fo_only = combined_scatter_image(lib, rho, phi_uc, grid, f, torch.zeros_like(T), L, geom, angle, q, mu_c, P,
                                         det, grd, ctx)
        _assert_images_close(fo_only, fo, msg=msg + " FC alone")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_combined_scatter_image_no_sync():
    """The combined pass (one projection per chunk, with and without the finite-cone footprint) runs its whole chunk
    loop without any host-device synchronization with a prebuilt context and the fused Siddon projectors (sync debug
    mode "error")."""
    from citrus.utilities import wlab_phantom
    from citrus.scatter_solver.kernels import ray_tracing_kernels
    from citrus.scatter_solver.scatter_image import make_direction_context

    dev = torch.device("cuda")
    if not ray_tracing_kernels.kernels_usable(dev)[0]:
        pytest.skip("Triton kernels unusable on this machine")
    geom = wlab_setup(2.0, device=dev, dtype=torch.float32)[1]
    grid = Grid((10, 16, 20), 2.0)  # padded along x to be divisible by 2
    rho = wlab_phantom(grid, device=dev, dtype=torch.float32)
    lib = synthetic_library(num_groups=3, legendre_order=2).to(dev, torch.float32)
    angle = 0.3
    phi_uc = uncollided_fluence(lib, rho, grid, geom, angle, SourceModel(), PROJ)
    q = panel_direction_set(geom, angle, (4, 8), max_polar_angle=grid, device=dev, dtype=torch.float32)
    det, grd = csi_response(), focused_1d_grid()
    P = SiddonProjectors()
    coarse = grid.coarsen(2)
    mu_c = coarsen_mu_t(mu_t_volume(lib, rho), 2)
    L = 2
    T = torch.rand(3, num_sh(L), *coarse.shape, device=dev) * 0.01
    ctx = make_direction_context(q, geom, angle, det, grd, lib, dev, torch.float32, chunk=8)

    def images():
        out = []
        for fp in (False, True):
            out.append(_si.combined_scatter_image(lib, rho, phi_uc, grid, 2, T, L, geom, angle, q, mu_c, P, det, grd,
                                                  ctx, footprint=fp))
        return out

    ref = images()  # compiles; the direction storage of the context is read to the host once here
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")  # any stream synchronization inside the calls raises
    try:
        got = images()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    for a, b in zip(got, ref):
        _assert_images_close(a, b, rtol=1e-5, msg="no-sync rerun")
