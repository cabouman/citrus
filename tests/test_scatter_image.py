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

from citrus.scatter_solver.angular_quadrature import Quadrature, hemisphere_quadrature, num_spherical_harmonics
from citrus.scatter_solver.config import SolverConfig
from citrus.scatter_solver.detector import (csi_response, detector_angle, energy_weighted_response, focused_1d_grid,
                                            ideal_counting_response, no_grid)
from citrus.scatter_solver.geometry import ConeGeometry, Grid
from citrus.utilities import wlab_setup
from citrus.scatter_solver.ray_tracing import SiddonProjectors
from citrus.scatter_solver.source import SourceModel
from citrus.scatter_solver.uncollided import uncollided_fluence
from citrus.scatter_solver.uncollided import first_collision_source_at_directions
from citrus.scatter_solver import scatter_image as scatter_image_module
from citrus.scatter_solver.scatter_image import (coarsen_mu_t, covering_polar_angle, panel_direction_set,
                                                 primary_image, scatter_fraction, supervoxel_subpoint_emission)
from citrus.scatter_solver.cross_sections import material_pairs, mu_t_volume, synthetic_library

# single rays per pixel (no finite-cone footprint) in every oracle comparison of this module
combined_scatter_image = functools.partial(scatter_image_module.combined_scatter_image, footprint=False)
scatter_image_from_source = functools.partial(scatter_image_module.scatter_image_from_source, footprint=False)

F64 = torch.float64
PROJECTION = SiddonProjectors()


# ---- direction sets ---------------------------------------------------------------------------------

def test_panel_direction_set():
    geometry = ConeGeometry(50.0, 60.0, 8, 8, 2.0, 2.0)
    grid = Grid((4, 4, 4), 1.0)
    for angle in (0.0, 1.1):
        n_hat = torch.tensor(geometry.panel_axes(angle)[2], dtype=F64)
        q = panel_direction_set(geometry, angle, (6, 12), dtype=F64)
        assert q.num_directions == 72
        assert float(q.weight.sum()) == pytest.approx(2 * math.pi, rel=1e-12)
        assert bool(((q.omega @ -n_hat) > 0).all())
        cover = covering_polar_angle(grid, geometry, angle)
        q2 = panel_direction_set(geometry, angle, (6, 12), max_polar_angle=grid, dtype=F64)
        assert float(q2.weight.sum()) == pytest.approx(2 * math.pi * (1 - math.cos(cover)), rel=1e-12)
        # brute-force check of the covering angle: random box point -> random panel point
        g = torch.Generator().manual_seed(0)
        lo, ext = torch.tensor(grid.lower_corner(), dtype=F64), torch.tensor(grid.extent, dtype=F64)
        # random points plus the vertices (box corners x panel corners); the max is at a vertex pair
        unit = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=F64)
        r = lo + ext * torch.cat([torch.rand(400, 3, generator=g, dtype=F64), unit])
        hu, hv = geometry.panel_half_extent()
        uh, vh, _ = (torch.tensor(a, dtype=F64) for a in geometry.panel_axes(angle))
        pc = torch.tensor(geometry.panel_center(angle), dtype=F64)
        sq = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=F64)
        uv = torch.cat([2 * torch.rand(400, 2, generator=g, dtype=F64) - 1, sq]) * torch.tensor([hu, hv], dtype=F64)
        p = pc + uv[:, :1] * uh + uv[:, 1:] * vh
        d = p[:, None] - r[None]
        th = torch.arccos((d @ -n_hat) / torch.linalg.norm(d, dim=-1))
        assert float(th.max()) == pytest.approx(cover - 1e-6, abs=1e-9)
    # a SolverConfig gives its cone counts
    qc = panel_direction_set(geometry, 0.0, SolverConfig(cone_num_polar=6, cone_num_azimuthal=12), dtype=F64)
    assert torch.equal(qc.omega, panel_direction_set(geometry, 0.0, (6, 12), dtype=F64).omega)


# ---- (b) collapsed cone vs brute-force point-to-point sum ----------------------------

def _brute_force_fluence(E: torch.Tensor, mu: torch.Tensor, grid: Grid, geometry: ConeGeometry, angle: float,
                         samples_per_pixel_axis: int = 4) -> torch.Tensor:
    """sum_voxels sum_subpoints E(r) exp(-tau(r -> r_d)) / |r_d - r|^2 dV_sub at each pixel center. [G, Rd, Cd]."""
    device, dtype = E.device, E.dtype
    c = grid.center_coordinates(device, dtype).reshape(-1, 3)
    off = (torch.arange(samples_per_pixel_axis, device=device, dtype=dtype) + 0.5) / samples_per_pixel_axis - 0.5
    d = torch.tensor(grid.deltas, device=device, dtype=dtype)
    sub = torch.stack(torch.meshgrid(off, off, off, indexing="ij"), -1).reshape(-1, 3) * d
    pts = (c[:, None] + sub[None]).reshape(-1, 3)
    G = E.shape[0]
    Ev = E.reshape(G, -1).repeat_interleave(samples_per_pixel_axis ** 3, dim=1)
    pix = geometry.pixel_centers(angle, device, dtype).reshape(-1, 3)
    starts = pts[None].expand(pix.shape[0], -1, -1).reshape(-1, 3)
    ends = pix[:, None].expand(-1, pts.shape[0], -1).reshape(-1, 3)
    tau = PROJECTION.line_integrals(mu, starts, ends, grid).reshape(G, pix.shape[0], -1)
    r2 = ((ends - starts) ** 2).sum(-1).reshape(pix.shape[0], -1)
    bf = (Ev[:, None, :] * torch.exp(-tau) / r2).sum(-1) * grid.voxel_volume / samples_per_pixel_axis ** 3
    return bf.reshape(G, *geometry.image_shape)


def test_collapsed_cone_vs_brute_force():
    # float64 oracle on CPU only: SiddonProjectors.oblique_project loops over directions in Python, and
    # 2048 small float64 launches take > 1 min on a (shared) consumer GPU.
    device = torch.device("cpu")
    grid = Grid((4, 4, 4), 1.0)
    geometry = ConeGeometry(50.0, 60.0, 8, 8, 2.0, 2.0)
    library = synthetic_library(num_groups=2, legendre_order=1)
    x, y, z = grid.axis_coordinates(device, F64)
    # smooth density (half water: mu_t ~ 0.11-0.15 /cm) and smooth isotropic emission
    rho = torch.zeros(2, *grid.shape, device=device, dtype=F64)
    rho[0] = 0.5 * (1.0 + 0.1 * x[None, :, None] + 0.05 * y[:, None, None]) + 0 * z
    mu = mu_t_volume(library, rho)
    E = torch.stack([1 + 0.2 * x[None, :, None] + 0.1 * z[None, None, :] + 0 * y[:, None, None],
                     0.5 + 0.1 * y[:, None, None] + 0 * x[None, :, None] + 0 * z[None, None, :]])
    angle = 0.0
    reference = _brute_force_fluence(E, mu, grid, geometry, angle)
    errs = []
    for P, A in ((8, 16), (16, 32), (32, 64)):
        q = panel_direction_set(geometry, angle, (P, A), max_polar_angle=grid, device=device, dtype=F64)
        out = scatter_image_from_source(lambda directions: E.expand(directions.shape[0], -1, -1, -1, -1), grid,
                                        geometry, angle, q, mu, PROJECTION, ideal_counting_response(), no_grid(),
                                        library, chunk=256)
        fl = out["fluence"]
        # counting detector, no grid: the signal is the PLANAR fluence (per-incident-photon responses)
        np.testing.assert_allclose(out["signal_per_group"].cpu().numpy(), out["planar_fluence"].cpu().numpy(),
                                   rtol=1e-12)
        assert bool((out["planar_fluence"] <= fl * (1 + 1e-12)).all())
        errs.append(float((fl - reference).norm() / reference.norm()))
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
    geometry = ConeGeometry(20.0, 30.0, 40, 40, 1.0, 1.0)
    E0 = torch.tensor([2.0, 0.5], device=device, dtype=F64)
    field = torch.zeros(2, *grid.shape, device=device, dtype=F64)
    field[:, 1, 1, 1] = E0
    mu = torch.zeros_like(field)
    n_hat = torch.tensor(geometry.panel_axes(angle)[2], device=device, dtype=F64)
    q = hemisphere_quadrature(-n_hat, 6, 12, math.radians(55.0), device=device, dtype=F64)
    # plus a direction leaving away from the panel and one whose shadow misses the panel
    u_hat = torch.tensor(geometry.panel_axes(angle)[0], device=device, dtype=F64)
    extra = torch.stack([n_hat, -n_hat * math.cos(math.radians(87)) + u_hat * math.sin(math.radians(87))])
    quadrature = Quadrature.from_directions(torch.cat([q.omega, extra]),
                                            torch.cat([q.weight, torch.tensor([0.3, 0.2], device=device, dtype=F64)]))
    library = synthetic_library(num_groups=2, legendre_order=1)
    out = scatter_image_from_source(lambda directions: field.expand(directions.shape[0], -1, -1, -1, -1), grid,
                                    geometry, angle, quadrature, mu, PROJECTION, energy_weighted_response(), no_grid(),
                                    library, chunk=16)
    total = out["fluence"].sum(dim=(1, 2)) * geometry.pixel_area  # [G]
    cos = q.omega @ -n_hat
    expected = E0 * grid.voxel_volume * (q.weight / cos).sum()
    np.testing.assert_allclose(total.cpu().numpy(), expected.cpu().numpy(), rtol=1e-6)
    # planar fluence (photons crossing the panel): the 1/cos cancels, every emitted photon of a direction that
    # hits the panel is counted once: sum_k w_k E V
    planar_total = out["planar_fluence"].sum(dim=(1, 2)) * geometry.pixel_area
    expected_planar = E0 * grid.voxel_volume * q.weight.sum()
    np.testing.assert_allclose(planar_total.cpu().numpy(), expected_planar.cpu().numpy(), rtol=1e-6)
    # energy-weighted detector: signal = E x planar fluence
    sig = out["signal"].sum() * geometry.pixel_area
    assert float(sig) == pytest.approx(float((expected_planar * library.group_energy.to(device)).sum()), rel=1e-6)


# ---- (d) WLAB first-order and multiple-collision images -----------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_wlab_first_order_and_mc_cuda():
    device = torch.device("cuda")
    grid, geometry, rho = wlab_setup(2.0, device=device, dtype=torch.float32)
    library = synthetic_library(num_groups=3, legendre_order=2, num_materials=7).to(device, torch.float32)
    angle = 0.3
    source = SourceModel()
    phi_uncollided = uncollided_fluence(library, rho, grid, geometry, angle, source, PROJECTION)
    q = panel_direction_set(geometry, angle, (16, 32), max_polar_angle=grid, device=device, dtype=torch.float32)
    detector, grid_response = csi_response(), focused_1d_grid()
    out = _first_order(library, rho, phi_uncollided, grid, 1, geometry, angle, q, None, PROJECTION, detector,
                       grid_response, chunk=16)
    fl = out["fluence"]
    assert fl.device.type == "cuda" and fl.shape == (3, *geometry.image_shape)
    assert torch.isfinite(fl).all() and (fl >= 0).all() and fl.sum() > 0
    sig = out["signal"]
    assert (sig >= 0).all() and (sig <= (fl * library.group_energy[:, None, None]).sum(0) * (1 + 1e-5)).all()
    # smooth: neighbouring pixels differ by < 10 % of the image maximum (8 x 16 directions give ~13 %
    # ray effects on the 0.4 cm pixels; 16 x 32 give ~6 %)
    tot = fl.sum(0)
    jump = max(float((tot[1:] - tot[:-1]).abs().max()), float((tot[:, 1:] - tot[:, :-1]).abs().max()))
    assert jump < 0.1 * float(tot.max())
    out2 = _first_order(library, rho, 2.5 * phi_uncollided, grid, 1, geometry, angle, q, None, PROJECTION, detector,
                        grid_response, chunk=16)
    assert float(out2["signal"].sum()) == pytest.approx(2.5 * float(sig.sum()), rel=1e-5)

    # multiple-collision term (zero first-collision source) with an isotropic (n = 0 only) random source
    L = 2
    g = torch.Generator(device="cpu").manual_seed(1)
    T = torch.zeros(3, num_spherical_harmonics(L), *grid.shape)
    T[:, 0] = torch.rand(3, *grid.shape, generator=g) * 0.01
    T = T.to(device)
    mu = mu_t_volume(library, rho)
    multiple_collision = combined_scatter_image(library, rho, torch.zeros_like(phi_uncollided), grid, 1, T, L, geometry,
                                                angle, q, mu, PROJECTION, detector, grid_response, chunk=16)
    assert (multiple_collision["fluence"] >= 0).all() and (multiple_collision["signal"] >= 0).all()
    assert multiple_collision["fluence"].sum() > 0
    # the spherical-harmonic plumbing: isotropic T is the same as the per-direction field T0 Y00
    iso = T[:, 0] / math.sqrt(4 * math.pi)
    reference = scatter_image_from_source(lambda directions: iso.expand(directions.shape[0], -1, -1, -1, -1), grid,
                                          geometry, angle, q,
                                          mu, PROJECTION, detector, grid_response, library, chunk=16)
    np.testing.assert_allclose(multiple_collision["signal"].cpu().numpy(), reference["signal"].cpu().numpy(), rtol=1e-5,
                               atol=1e-6 * float(reference["signal"].max()))
    # the 6-D linear-discontinuous-moment tensor is sliced to its cell mean
    T6 = torch.stack([T, torch.randn_like(T), torch.randn_like(T), torch.randn_like(T)], dim=2)
    multiple_collision6 = combined_scatter_image(library, rho, torch.zeros_like(phi_uncollided), grid, 1, T6, L,
                                                 geometry, angle, q, mu,
                                                 PROJECTION, detector,
                                                 grid_response, chunk=16)
    np.testing.assert_allclose(multiple_collision6["signal"].cpu().numpy(), multiple_collision["signal"].cpu().numpy(),
                               rtol=1e-6)


# ---- (e) primary and scatter fraction -----------------------------------------------------------------

def test_primary_uniform_box(device):
    grid = Grid((4, 4, 4), 1.0)
    geometry = ConeGeometry(50.0, 60.0, 5, 5, 2.0, 2.0)
    library = synthetic_library(num_groups=3, legendre_order=1)
    rho = torch.zeros(2, *grid.shape, device=device, dtype=F64)
    rho[0] = 1.0
    detector, grid_response = csi_response(), focused_1d_grid(energies=library.group_energy)
    out = primary_image(library, rho, grid, geometry, 0.0, SourceModel(), PROJECTION, detector, grid_response)
    mu = library.mu_t[0].to(device)  # water, rho = 1
    f = library.spectrum_fraction.to(device)
    expected = f * torch.exp(-4.0 * mu) / 60.0 ** 2
    np.testing.assert_allclose(out["fluence"][:, 2, 2].cpu().numpy(), expected.cpu().numpy(), rtol=1e-12)
    eg = library.group_energy.to(device)
    zero = torch.zeros((), device=device, dtype=F64)
    response = detector.at(eg, zero) * grid_response.at(eg, zero, zero)
    np.testing.assert_allclose(out["signal_per_group"][:, 2, 2].cpu().numpy(), (expected * response).cpu().numpy(),
                               rtol=1e-12)
    assert float(out["signal"][2, 2]) == pytest.approx(float((expected * response).sum()), rel=1e-12)
    # corner pixel: Beer's law along the oblique ray, 1/r^2
    source = torch.tensor(geometry.source_position(0.0), device=device, dtype=F64)
    pix = geometry.pixel_centers(0.0, device, F64)[0, 0]
    tau = PROJECTION.line_integrals(mu_t_volume(library, rho), source[None], pix[None], grid)[:, 0]
    exp_c = f * torch.exp(-tau) / ((pix - source) ** 2).sum()
    np.testing.assert_allclose(out["fluence"][:, 0, 0].cpu().numpy(), exp_c.cpu().numpy(), rtol=1e-12)
    # planar fluence = cos(theta0) x fluence; theta0 = angle of the primary ray to the panel normal
    cos_c = 60.0 / float(torch.linalg.norm(pix - source))
    np.testing.assert_allclose(out["planar_fluence"][:, 0, 0].cpu().numpy(), (cos_c * exp_c).cpu().numpy(),
                               rtol=1e-12)
    # signal = planar fluence x per-incident-photon responses
    directions_c = (pix - source) / torch.linalg.norm(pix - source)
    theta0 = detector_angle(directions_c, geometry, 0.0)
    th1, th2 = scatter_image_module.grid_angles(directions_c, pix, geometry, 0.0, grid_response.focal_distance_cm)
    response = detector.at(library.group_energy.to(device), theta0) * grid_response.at(library.group_energy.to(device),
                                                                                       th1, th2)
    np.testing.assert_allclose(out["signal_per_group"][:, 0, 0].cpu().numpy(),
                               (cos_c * exp_c * response).cpu().numpy(), rtol=1e-10)

    # scatter fraction with a single-scatter image of the same box
    phi_uncollided = uncollided_fluence(library, rho, grid, geometry, 0.0, SourceModel(), PROJECTION)
    q = panel_direction_set(geometry, 0.0, (6, 12), max_polar_angle=grid, device=device, dtype=F64)
    sc = _first_order(library, rho, phi_uncollided, grid, 1, geometry, 0.0, q, None, PROJECTION, detector,
                      grid_response, chunk=36)
    sf = scatter_fraction(sc["signal"], out["signal"])
    assert sf.shape == geometry.image_shape
    assert bool((sf >= 0).all()) and bool((sf < 1).all()) and float(sf.max()) > 0
    assert float(scatter_fraction(torch.zeros(1), torch.zeros(1))) == 0.0


def test_primary_fine_energy_subpixel(device):
    """Primary with within-group hardening (synthetic fine_spread) and sub-pixel sampling: the fine-energy transmission
    is formed per sub-ray and THEN averaged over the pixel (exact pixel average), against an independent evaluation
    from explicit sub-ray line integrals of the two density maps; samples_per_pixel_axis=None takes the projector's own
    samples_per_pixel_axis."""
    grid = Grid((4, 5, 6), 1.0, voxel_slice_aspect=0.8)
    geometry = ConeGeometry(30.0, 45.0, 4, 5, 2.5, 2.0, det_row_offset=0.3, det_channel_offset=-0.4)
    library = synthetic_library(num_groups=3, legendre_order=1, num_fine_per_group=5, fine_spread=0.6)
    g = torch.Generator().manual_seed(7)
    rho = torch.rand(2, *grid.shape, generator=g, dtype=F64).to(device)
    detector, grid_response = ideal_counting_response(), no_grid()
    angle, n = 0.35, 3
    out = primary_image(library, rho, grid, geometry, angle, SourceModel(), PROJECTION, detector, grid_response,
                        samples_per_pixel_axis=n)
    # reference: n x n midpoint lattice of sub-pixel centers, line integrals of rho, fine-energy sum, mean over sub-rays
    pix = geometry.pixel_centers(angle, device, F64)  # [Rd, Cd, 3]
    uh, vh, _ = (torch.tensor(a, device=device, dtype=F64) for a in geometry.panel_axes(angle))
    off = (2 * torch.arange(n, device=device, dtype=F64) + 1) / (2 * n) - 0.5
    shift = (off * geometry.delta_det_row)[:, None, None] * vh + (off * geometry.delta_det_channel)[None, :, None] * uh
    ends = (pix[:, :, None, None, :] + shift).reshape(-1, 3)
    source = torch.tensor(geometry.source_position(angle), device=device, dtype=F64)
    path = PROJECTION.line_integrals(rho, source.expand_as(ends), ends, grid)  # [2, Rd Cd n n]
    tau = library.fine_mu_t.to(device).t() @ path  # [E, Npts]
    T = torch.zeros(library.num_groups, path.shape[1], device=device, dtype=F64)
    T.index_add_(0, library.fine_group.to(device), library.fine_weight.to(device)[:, None] * torch.exp(-tau))
    T = T.reshape(-1, *geometry.image_shape, n * n).mean(-1)
    d = pix - source
    r2 = (d * d).sum(-1)
    reference = T * (SourceModel().angular_weight(d, geometry, angle) / r2)[None]
    np.testing.assert_allclose(out["fluence"].cpu().numpy(), reference.cpu().numpy(), rtol=1e-12)
    cos0 = torch.cos(detector_angle(d / r2.sqrt()[..., None], geometry, angle))
    np.testing.assert_allclose(out["planar_fluence"].cpu().numpy(), (reference * cos0).cpu().numpy(), rtol=1e-12)
    assert torch.equal(out["signal"], out["planar_fluence"].sum(0))  # ideal counting, no grid
    # the projector's own samples_per_pixel_axis is the default (and is not applied twice)
    projectors = SiddonProjectors(samples_per_pixel_axis=n)
    out_p = primary_image(library, rho, grid, geometry, angle, SourceModel(), projectors, detector, grid_response)
    torch.testing.assert_close(out_p["fluence"], out["fluence"], rtol=1e-13, atol=0)
    out_1 = primary_image(library, rho, grid, geometry, angle, SourceModel(), projectors, detector, grid_response,
                          samples_per_pixel_axis=1)
    assert not torch.allclose(out_1["fluence"], out["fluence"], rtol=1e-6)
    # hardening: exceeds the group-wise exp(-int mu_t,g dl) at the pixel centers (Jensen, per group)
    mu_path = PROJECTION.primary_project(mu_t_volume(library, rho), grid, geometry, angle)
    old = library.spectrum_fraction.to(device)[:, None, None] * torch.exp(-mu_path) * (
        SourceModel().angular_weight(d, geometry, angle) / r2)[None]
    assert bool((out_1["fluence"] >= old * (1 - 1e-12)).all()) and float(out_1["fluence"].sum()) > float(old.sum())


# ---- (f) scatter image on the supervoxel grid (sub-voxel quadrature points) --------------------------------------

IMAGE_KEYS = ("fluence", "planar_fluence", "signal_per_group", "signal")


def _small_first_collision_case(device, dtype=F64):
    grid = Grid((4, 4, 4), 1.0)
    geometry = ConeGeometry(50.0, 60.0, 8, 8, 2.0, 2.0)
    library = synthetic_library(num_groups=3, legendre_order=2).to(device, dtype)
    x, y, z = grid.axis_coordinates(device, dtype)
    rho = torch.zeros(2, *grid.shape, device=device, dtype=dtype)
    rho[0] = 0.8 + 0.05 * x[None, :, None] + 0.03 * y[:, None, None] + 0.02 * z[None, None, :]
    rho[1, 1:3, 1:3, 1:3] = 0.4
    phi_uncollided = uncollided_fluence(library, rho, grid, geometry, 0.3, SourceModel(), PROJECTION)
    q = panel_direction_set(geometry, 0.3, (6, 12), max_polar_angle=grid, device=device, dtype=dtype)
    return grid, geometry, library, rho, phi_uncollided, q


def _zero_multiple_collision(library, grid: Grid, f: int, device, dtype=F64) -> torch.Tensor:
    """A zero multiple-collision source T [G, 1, Rc, Cc, Sc] (L = 0): combined_scatter_image is then the first-order
    image alone."""
    return torch.zeros(library.num_groups, 1, *grid.coarsen(f).shape, device=device, dtype=dtype)


def _first_order(library, rho, phi_uncollided, grid, f, geometry, angle, q, mu_coarse_values, P, detector,
                 grid_response, **kw):
    """First-order image of combined_scatter_image (zero multiple-collision source)."""
    zero = _zero_multiple_collision(library, grid, f, phi_uncollided.device, phi_uncollided.dtype)
    return combined_scatter_image(library, rho, phi_uncollided, grid, f, zero, 0, geometry, angle, q,
                                  mu_coarse_values, P, detector, grid_response, **kw)


def _first_collision_function(library, rho, phi_uncollided, grid, geometry, angle):
    return lambda directions: first_collision_source_at_directions(library, rho, phi_uncollided, grid, geometry, angle,
                                                                   directions)


def _upsample(t: torch.Tensor, f: int) -> torch.Tensor:
    """Coarse [..., Rc, Cc, Sc] replicated on the f^3 fine voxels of each cell."""
    return t.repeat_interleave(f, -3).repeat_interleave(f, -2).repeat_interleave(f, -1)


def _linear_subpoint_tau(tau_c, mu_coarse_values, directions, grid: Grid, f: int) -> torch.Tensor:
    """Independent (explicit-coordinate) form of the in-cell attenuation of `supervoxel_subpoint_emission`:
    tau_j = max(0, tau(cell) - mu(cell) Omega . (r_j - r_cell)) at every fine voxel center r_j, [k, G, R, C, S]."""
    coarse = grid.coarsen(f)
    rf = grid.center_coordinates(tau_c.device, tau_c.dtype)  # [R, C, S, 3]
    rc = _upsample(coarse.center_coordinates(tau_c.device, tau_c.dtype).movedim(-1, 0), f).movedim(0, -1)
    projection = torch.einsum("rcsx,kx->krcs", rf - rc, directions.to(tau_c.dtype))  # [k, R, C, S]
    return (_upsample(tau_c, f) - _upsample(mu_coarse_values, f)[None] * projection[:, None]).clamp(min=0.0)


def _subpoint_image(fn, grid, f, geometry, angle, q, mu_coarse_values, P, detector, grid_response, **kw):
    """Reference of the sub-point projection: the fine-grid image of S_j exp(-tau_j) (tau_j from the coarse
    downstream trace by `_linear_subpoint_tau`) with no further attenuation."""
    coarse = grid.coarsen(f)

    def source(directions):
        tau_c = P.downstream_cumulative(mu_coarse_values, coarse, geometry, angle, directions)
        return fn(directions) * torch.exp(-_linear_subpoint_tau(tau_c, mu_coarse_values, directions, grid, f))

    zero = torch.zeros(mu_coarse_values.shape[0], *grid.shape, device=mu_coarse_values.device,
                       dtype=mu_coarse_values.dtype)
    return scatter_image_from_source(source, grid, geometry, angle, q, zero, P, detector, grid_response, **kw)


def test_first_order_factor1_is_single_grid(device):
    """factor 1: the sub-voxel points are the voxel centers, so the first-order term of the combined pass is the
    single-grid image of the first-collision source (any attenuation map; None -> the true attenuation)."""
    grid, geometry, library, rho, phi_uncollided, q = _small_first_collision_case(device)
    detector, grid_response = csi_response(), focused_1d_grid(energies=library.group_energy)
    mu = mu_t_volume(library, rho)
    first_collision = _first_collision_function(library, rho, phi_uncollided, grid, geometry, 0.3)
    reference = scatter_image_from_source(first_collision, grid, geometry, 0.3, q, 0.9 * mu, PROJECTION, detector,
                                          grid_response, library, chunk=16)
    out = _first_order(library, rho, phi_uncollided, grid, 1, geometry, 0.3, q, 0.9 * mu, PROJECTION, detector,
                       grid_response, chunk=16)
    assert set(out) == set(IMAGE_KEYS)
    _assert_images_close(out, reference, msg="factor 1")
    reference_true = scatter_image_from_source(first_collision, grid, geometry, 0.3, q, mu, PROJECTION, detector,
                                               grid_response, library, chunk=16)
    out_true = _first_order(library, rho, phi_uncollided, grid, 1, geometry, 0.3, q, None, PROJECTION, detector,
                            grid_response, chunk=16)
    np.testing.assert_allclose(out_true["signal"].cpu().numpy(), reference_true["signal"].cpu().numpy(), rtol=1e-6)


def test_subpoint_emission(device):
    """supervoxel_subpoint_emission against an explicit per-point loop and against the explicit-coordinate form;
    the f = 2 first-order image equals the fine-grid image of the sub-point emission; bad offsets raise."""
    grid, geometry, library, rho, phi_uncollided, q = _small_first_collision_case(device)
    detector, grid_response = csi_response(), no_grid()
    f = 2
    coarse = grid.coarsen(f)
    mu_coarse_values = coarsen_mu_t(mu_t_volume(library, rho), grid, grid.coarsen(f))
    directions = q.omega[:5].to(device, F64)
    # [5, G, 4, 4, 4]
    S = first_collision_source_at_directions(library, rho, phi_uncollided, grid, geometry, 0.3, directions)
    tau = PROJECTION.downstream_cumulative(mu_coarse_values, coarse, geometry, 0.3, directions)  # [5, G, 2, 2, 2]
    e_sub = supervoxel_subpoint_emission(S, tau, mu_coarse_values, directions, grid, f)  # [5, G, 4, 4, 4]
    rf = grid.center_coordinates(device, F64)
    rc = coarse.center_coordinates(device, F64)
    expected = torch.zeros_like(S)
    for i in range(4):
        for j in range(4):
            for m in range(4):
                ci, cj, cm = i // f, j // f, m // f
                d = rf[i, j, m] - rc[ci, cj, cm]
                t = (tau[:, :, ci, cj, cm]
                     - mu_coarse_values[None, :, ci, cj, cm] * (directions @ d)[:, None]).clamp(min=0)
                expected[:, :, i, j, m] = S[:, :, i, j, m] * torch.exp(-t)
    np.testing.assert_allclose(e_sub.cpu().numpy(), expected.cpu().numpy(), rtol=1e-12)
    reference = S * torch.exp(-_linear_subpoint_tau(tau, mu_coarse_values, directions, grid, f))
    np.testing.assert_allclose(e_sub.cpu().numpy(), reference.cpu().numpy(), rtol=1e-12)
    # factor 1: S exp(-tau)
    mu = mu_t_volume(library, rho)
    tau1 = PROJECTION.downstream_cumulative(mu, grid, geometry, 0.3, directions)
    torch.testing.assert_close(supervoxel_subpoint_emission(S, tau1, mu, directions, grid, 1), S * torch.exp(-tau1))
    with pytest.raises(ValueError, match="offsets"):
        supervoxel_subpoint_emission(S, tau, mu_coarse_values, directions, grid, f, offsets=torch.zeros(5, 1, 1, 1,
                                                                                                        device=device))
    # the image: the fine-grid projection of the sub-point emission
    out = _first_order(library, rho, phi_uncollided, grid, f, geometry, 0.3, q, mu_coarse_values, PROJECTION, detector,
                       grid_response, chunk=16)
    reference_image = _subpoint_image(_first_collision_function(library, rho, phi_uncollided, grid, geometry, 0.3),
                                      grid, f, geometry, 0.3, q, mu_coarse_values, PROJECTION, detector,
                                      grid_response, library=library, chunk=16)
    for k in IMAGE_KEYS:
        np.testing.assert_allclose(out[k].cpu().numpy(), reference_image[k].cpu().numpy(), rtol=1e-10, err_msg=k)


def _slab_case(device, mu_value: float):
    """Slab of uniform attenuation mu_value (group 1: 1.5x) on a 0.5 cm grid, smooth Gaussian emission, 25 deg cone."""
    grid = Grid((8, 24, 24), 0.5)
    geometry = ConeGeometry(50.0, 60.0, 24, 24, 1.0, 1.0)
    mu = torch.full((2, *grid.shape), mu_value, device=device, dtype=F64)
    mu[1] *= 1.5
    x, y, z = grid.axis_coordinates(device, F64)
    gauss = torch.exp(-(x[None, :, None] ** 2 + z[None, None, :] ** 2) / (2 * 1.5 ** 2)) + 0 * y[:, None, None]
    E = torch.stack([gauss * (1 + 0.1 * y[:, None, None]), 0.5 * gauss])
    q = panel_direction_set(geometry, 0.0, (8, 16), max_polar_angle=math.radians(25.0), device=device, dtype=F64)
    return grid, geometry, mu, E, q


def test_subpoint_projection_converges(device):
    """Smooth emission, 2 cm supervoxels on a 0.5 cm fine grid (64 sub-points): projecting the sub-points with the
    linear in-cell attenuation reproduces the fine-grid image exactly in vacuum and closely in a uniform slab (exact
    there up to rays leaving through a lateral face)."""
    f = 4
    kw = dict(group_energy=torch.tensor([60.0, 40.0]), chunk=64)
    detector, grid_response = ideal_counting_response(), no_grid()
    err, bias = {}, {}
    for case, mu_value in (("vacuum", 0.0), ("slab", 0.4)):
        grid, geometry, mu, E, q = _slab_case(device, mu_value)

        def fn(directions):
            return E.expand(directions.shape[0], -1, -1, -1, -1)

        reference = scatter_image_from_source(fn, grid, geometry, 0.0, q, mu, PROJECTION, detector, grid_response,
                                              **kw)["fluence"]
        mu_coarse_values = coarsen_mu_t(mu, grid, grid.coarsen(f))
        fl = _subpoint_image(fn, grid, f, geometry, 0.0, q, mu_coarse_values, PROJECTION, detector, grid_response,
                             **kw)["fluence"]
        err[case] = float((fl - reference).norm() / reference.norm())
        bias[case] = float(fl.sum() / reference.sum() - 1)
    msg = "; ".join(f"{c}: rel L2 {err[c]:.2e}, sum {bias[c]:+.2e}" for c in err)
    # measured (CPU and CUDA, rel L2 / panel-sum bias): vacuum 0 / 0; slab 3.7e-3 / -1.5e-3
    assert err["vacuum"] < 1e-12, msg
    assert err["slab"] < 1e-2 and abs(bias["slab"]) < 4e-3, msg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_first_order_wlab_cuda():
    """WLAB 2 cm (padded), synthetic library, float32 CUDA, the default (fused) Siddon projectors: factor 1 equals the
    single-grid image to 1e-6; factor 2 (4 cm cells) stays non-negative and close to it."""
    from citrus.utilities import wlab_phantom

    device = torch.device("cuda")
    geometry = wlab_setup(2.0, device=device, dtype=torch.float32)[1]
    grid = Grid((10, 16, 20), 2.0)  # padded along x to be divisible by 2
    rho = wlab_phantom(grid, device=device, dtype=torch.float32)
    library = synthetic_library(num_groups=3, legendre_order=2, num_materials=7).to(device, torch.float32)
    angle = 0.3
    phi_uncollided = uncollided_fluence(library, rho, grid, geometry, angle, SourceModel(), PROJECTION)
    q = panel_direction_set(geometry, angle, (8, 16), max_polar_angle=grid, device=device, dtype=torch.float32)
    detector, grid_response = csi_response(), focused_1d_grid()
    mu = mu_t_volume(library, rho)
    P = SiddonProjectors()
    reference = scatter_image_from_source(_first_collision_function(library, rho, phi_uncollided, grid, geometry,
                                                                    angle),
                                          grid, geometry, angle, q, mu, P, detector, grid_response, library, chunk=8)
    one = _first_order(library, rho, phi_uncollided, grid, 1, geometry, angle, q, mu, P, detector, grid_response,
                       chunk=8)
    _assert_images_close(one, reference, msg="factor 1")
    two = _first_order(library, rho, phi_uncollided, grid, 2, geometry, angle, q,
                       coarsen_mu_t(mu, grid, grid.coarsen(2)), P, detector, grid_response, chunk=8)
    assert float(two["signal"].min()) >= 0 and float(two["fluence"].min()) >= 0
    e = float((two["signal"] - reference["signal"]).norm() / reference["signal"].norm())
    b = float(two["signal"].sum() / reference["signal"].sum() - 1)
    # measured (8x16): 0.174 / +7.1 %. 4 cm cells do not resolve the inserts, and the coarse mu_t under-attenuates the
    # emission of the denser sub-voxels of a mixed cell
    assert e < 0.25 and abs(b) < 0.12, (e, b)


# ---- direction-set context (per-view precomputation of the response weights) -------------------------------

def _reference_cone_image(source_function, grid, geometry, angle, quadrature, mu, projectors, detector, grid_response,
                          e_g, chunk):
    """The plain accumulation loop of the collapsed-cone image (responses evaluated per chunk on the full panel)."""
    from citrus.scatter_solver.detector import detector_angle, grid_angles

    device, dtype = mu.device, mu.dtype
    G = mu.shape[0]
    directions_all, w_all = quadrature.omega.to(device, dtype), quadrature.weight.to(device, dtype)
    n = torch.tensor(geometry.panel_axes(angle)[2], device=device, dtype=dtype)
    cos_all = (directions_all * -n).sum(-1)
    keep = cos_all > 0
    directions_all, w_all, cos_all = directions_all[keep], w_all[keep], cos_all[keep]
    pix = geometry.pixel_centers(angle, device, dtype)
    out = {k: torch.zeros(G, *pix.shape[:2], device=device, dtype=dtype)
           for k in ("fluence", "planar_fluence", "signal_per_group")}
    for s in range(0, directions_all.shape[0], chunk):
        directions, w, c = directions_all[s:s + chunk], w_all[s:s + chunk], cos_all[s:s + chunk]
        tau = projectors.downstream_cumulative(mu, grid, geometry, angle, directions)
        image = projectors.oblique_project(torch.exp(-tau) * source_function(directions), grid, geometry, angle,
                                           directions)
        image = image * w[:, None, None, None]
        out["fluence"] += image.sum(0)
        out["planar_fluence"] += torch.einsum("kgrc,k->grc", image, c)
        d = detector.at(e_g, detector_angle(directions, geometry, angle)) * c
        th1, th2 = grid_angles(directions, pix, geometry, angle, grid_response.focal_distance_cm)
        out["signal_per_group"] += torch.einsum("kgrc,gkrc,gk->grc", image, grid_response.at(e_g, th1, th2), d)
    out["signal"] = out["signal_per_group"].sum(0)
    return out


def _assert_images_close(got, reference, rtol=1e-6, msg=""):
    for k in IMAGE_KEYS:
        np.testing.assert_allclose(got[k].cpu().numpy(), reference[k].cpu().numpy(), rtol=rtol,
                                   atol=rtol * float(reference[k].abs().max()), err_msg=f"{msg} {k}")


def test_direction_context_matches_reference(device):
    """Context path (built on the fly, prebuilt and factorized, prebuilt per-pixel) == the pre-context loop with
    the full per-pixel responses, focused grid, oblique view; the factorized grid response is separable."""
    from citrus.scatter_solver.scatter_image import make_direction_context

    grid, geometry, library, rho, phi_uncollided, q = _small_first_collision_case(device)
    detector, grid_response = csi_response(), focused_1d_grid(energies=library.group_energy)
    mu = mu_t_volume(library, rho)
    x, y, z = grid.axis_coordinates(device, F64)
    S = torch.stack([1 + 0.2 * x[None, :, None] + 0.1 * z[None, None, :] + 0 * y[:, None, None]] * 3)

    def source(directions):
        return S[None] * (1 + 0.3 * directions[:, 2])[:, None, None, None, None]

    reference = _reference_cone_image(source, grid, geometry, 0.3, q, mu, PROJECTION, detector, grid_response,
                                      library.group_energy, 5)
    fly = scatter_image_from_source(source, grid, geometry, 0.3, q, mu, PROJECTION, detector, grid_response, library,
                                    chunk=5)
    _assert_images_close(fly, reference, msg="on the fly")
    for factorize in (True, False):
        context = make_direction_context(q, geometry, 0.3, detector, grid_response, library, device, F64, chunk=5,
                                         factorize=factorize)
        assert context.separable == factorize and context.num_directions == q.num_directions
        assert (context.grid_weight is None) != factorize
        if factorize:
            assert context.accumulation_weight.shape == (library.num_groups * geometry.num_det_rows, 3,
                                                         q.num_directions)
        # the context's chunking replaces the chunk argument
        got = scatter_image_from_source(source, grid, geometry, 0.3, q, mu, PROJECTION, detector, grid_response,
                                        library, chunk=99, context=context)
        _assert_images_close(got, reference, msg=f"factorize={factorize}")


def test_direction_context_all_images(device):
    """The combined image (factors 1 and 2) with a shared context == without (1e-6); the sub-cell
    offsets and the harmonics are cached on the context; the context is checked against the call."""
    from citrus.scatter_solver.scatter_image import make_direction_context

    grid, geometry, library, rho, phi_uncollided, q = _small_first_collision_case(device)
    detector, grid_response = csi_response(), focused_1d_grid(energies=library.group_energy)
    mu = mu_t_volume(library, rho)
    context = make_direction_context(q, geometry, 0.3, detector, grid_response, library, device, F64, chunk=7)
    L = 2
    for f in (1, 2):
        g = torch.Generator(device="cpu").manual_seed(3 + f)
        T = (torch.rand(3, num_spherical_harmonics(L), *grid.coarsen(f).shape, generator=g,
                        dtype=F64) * 0.01).to(device)
        a = combined_scatter_image(library, rho, phi_uncollided, grid, f, T, L, geometry, 0.3, q, None, PROJECTION,
                                   detector, grid_response,
                                   chunk=7)
        b = combined_scatter_image(library, rho, phi_uncollided, grid, f, T, L, geometry, 0.3, q, None, PROJECTION,
                                   detector, grid_response,
                                   context)
        _assert_images_close(b, a, msg=f"f={f}")
    assert ("spherical_harmonics", L) in context._cache and any(k[0] == "subcell" for k in context._cache)

    def zero_source(directions):
        return torch.zeros(directions.shape[0], 3, *grid.shape, device=device, dtype=F64)

    # mismatched calls are refused
    for bad in ({"angle": 0.4}, {"detector": csi_response()}, {"grid_response": no_grid()}):
        kw = {"angle": 0.3, "detector": detector, "grid_response": grid_response}
        kw.update(bad)
        with pytest.raises(ValueError, match="does not match"):
            scatter_image_from_source(zero_source, grid, geometry, kw["angle"], q, mu, PROJECTION, kw["detector"],
                                      kw["grid_response"], library, context=context)
    with pytest.raises(ValueError, match="dtype"):
        scatter_image_from_source(zero_source, grid, geometry, 0.3, q, mu.float(), PROJECTION, detector, grid_response,
                                  library, context=context)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_direction_context_chunk_loop_no_sync():
    """With a prebuilt context, the whole chunk loop of the scatter images (the combined image with the fused
    first-collision source, a plain source) runs without any host-device synchronization."""
    from citrus.utilities import wlab_phantom
    from citrus.scatter_solver.kernels import ray_tracing_kernels
    from citrus.scatter_solver.scatter_image import make_direction_context

    device = torch.device("cuda")
    if not ray_tracing_kernels.kernels_usable(device)[0]:
        pytest.skip("Triton kernels unusable on this machine")
    geometry = wlab_setup(2.0, device=device, dtype=torch.float32)[1]
    grid = Grid((10, 16, 20), 2.0)  # padded along x to be divisible by 2
    rho = wlab_phantom(grid, device=device, dtype=torch.float32)
    library = synthetic_library(num_groups=3, legendre_order=2, num_materials=7).to(device, torch.float32)
    angle = 0.3
    phi_uncollided = uncollided_fluence(library, rho, grid, geometry, angle, SourceModel(), PROJECTION)
    q = panel_direction_set(geometry, angle, (4, 8), max_polar_angle=grid, device=device, dtype=torch.float32)
    detector, grid_response = csi_response(), focused_1d_grid()
    P = SiddonProjectors()
    mu = mu_t_volume(library, rho)
    coarse = grid.coarsen(2)
    mu_coarse_values = coarsen_mu_t(mu, grid, grid.coarsen(2))
    L = 2
    T = torch.rand(3, num_spherical_harmonics(L), *coarse.shape, device=device) * 0.01
    S = torch.rand(1, 3, *coarse.shape, device=device)
    context = make_direction_context(q, geometry, angle, detector, grid_response, library, device, torch.float32,
                                     chunk=8)
    pairs = material_pairs(rho)  # built once per density field (ScatterSolver.material_fields): one host sync

    def images():
        return (combined_scatter_image(library, rho, phi_uncollided, grid, 2, T, L, geometry, angle, q,
                                       mu_coarse_values, P, detector, grid_response, context, pairs=pairs),
                scatter_image_from_source(lambda directions: S.expand(directions.shape[0], -1, -1, -1, -1), coarse,
                                          geometry, angle, q,
                                          mu_coarse_values, P, detector, grid_response, library, context=context))

    reference = images()  # compiles; the direction storage of the context is read to the host once here
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")  # any stream synchronization inside the calls raises
    try:
        got = images()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    for a, b in zip(got, reference):
        _assert_images_close(a, b, rtol=1e-5, msg="no-sync rerun")


def test_direction_context_memory_wlab(capsys):
    """Size of the context of the WLAB 2048-direction set (32 x 64 in the covering cone, G = 6, 75 x 100 panel,
    focused grid, float32): the factorized weights are [G * rows, 3, K] (+ [K, G, rows] tables), not per pixel."""
    from citrus.scatter_solver.scatter_image import make_direction_context

    grid, geometry, _ = wlab_setup(1.0, dtype=torch.float32)
    q = panel_direction_set(geometry, 0.0, (32, 64), max_polar_angle=grid, dtype=torch.float32)
    library = synthetic_library(num_groups=6, legendre_order=1)
    context = make_direction_context(q, geometry, 0.0, csi_response(), focused_1d_grid(), library, "cpu", torch.float32)
    K, G, (Rd, Cd) = context.num_directions, context.num_groups, context.image_shape
    per_pixel = K * G * Rd * Cd * 4
    msg = (f"WLAB direction context: K = {K} directions, G = {G}, panel {Rd} x {Cd}: {context.nbytes / 2 ** 20:.1f} "
           f"MiB (fused weights {context.accumulation_weight.numel() * 4 / 2 ** 20:.1f} MiB; one per-pixel "
           f"[K, G, rows, channels] table would be {per_pixel / 2 ** 20:.0f} MiB)")
    with capsys.disabled():
        print("\n" + msg)
    assert K == 2048 and context.separable, msg
    assert context.nbytes < 32 * 2 ** 20, msg


# ---- combined first-order + multiple-collision pass ---------------------------------------------------------

def _broadcast_multiple_collision_source(T, L, f):
    """Source on the fine grid of the multiple-collision emission S_MC = Y(Omega) . T replicated on the f^3 sub-points
    of its supervoxel."""
    from citrus.scatter_solver.angular_quadrature import real_spherical_harmonics, scatter_source_at_directions

    def source(directions):
        s = scatter_source_at_directions(T, real_spherical_harmonics(L, directions))  # [k, G, Rc, Cc, Sc]
        return _upsample(s, f)

    return source


def _combined_references(library, rho, phi_uncollided, grid, f, T, L, geometry, angle, q, mu_coarse_values, P, detector,
                         grid_response, context):
    """(first-order, multiple-collision) references of the combined pass, each the fine-grid image of its sub-point
    emission (`_subpoint_image`: explicit-coordinate in-cell attenuation, generic single-grid image)."""
    kw = dict(library=library, context=context)
    first_order = _subpoint_image(_first_collision_function(library, rho, phi_uncollided, grid, geometry, angle), grid,
                                  f, geometry, angle, q, mu_coarse_values,
                                  P, detector, grid_response, **kw)
    multiple_collision = _subpoint_image(_broadcast_multiple_collision_source(T, L, f), grid, f, geometry, angle, q,
                                         mu_coarse_values, P, detector, grid_response, **kw)
    return first_order, multiple_collision


def test_combined_scatter_image_small(device):
    """float64, Siddon, focused grid, oblique view, factors 1 and 2: combined == first-order + multiple-collision
    reference; the image of each source alone (the other one zero) is its reference term; at factor 1 the
    multiple-collision term is the single-grid image of the multiple-collision source; a 6-D T is sliced to its
    linear-discontinuous mean; bad arguments raise."""
    from citrus.scatter_solver.angular_quadrature import real_spherical_harmonics, scatter_source_at_directions
    from citrus.scatter_solver.scatter_image import make_direction_context

    grid, geometry, library, rho, phi_uncollided, q = _small_first_collision_case(device)
    detector, grid_response = csi_response(), focused_1d_grid(energies=library.group_energy)
    mu = mu_t_volume(library, rho)
    L = 2
    context = make_direction_context(q, geometry, 0.3, detector, grid_response, library, device, F64, chunk=7)
    for f in (1, 2):
        coarse = grid.coarsen(f)
        g = torch.Generator(device="cpu").manual_seed(5 + f)
        T = (torch.rand(3, num_spherical_harmonics(L), *coarse.shape, generator=g, dtype=F64) * 0.01).to(device)
        T[:, 1:] *= 0.3
        mu_coarse_values = coarsen_mu_t(0.9 * mu, grid, grid.coarsen(f))
        msg = f"f={f}"
        first_order, multiple_collision = _combined_references(library, rho, phi_uncollided, grid, f, T, L, geometry,
                                                               0.3, q, mu_coarse_values, PROJECTION, detector,
                                                               grid_response, context)
        tot = combined_scatter_image(library, rho, phi_uncollided, grid, f, T, L, geometry, 0.3, q, mu_coarse_values,
                                     PROJECTION, detector, grid_response, context)
        assert set(tot) == set(IMAGE_KEYS)
        _assert_images_close(tot, {k: first_order[k] + multiple_collision[k] for k in IMAGE_KEYS}, msg=msg)
        first_order_only = combined_scatter_image(library, rho, phi_uncollided, grid, f, torch.zeros_like(T), L,
                                                  geometry, 0.3, q, mu_coarse_values, PROJECTION,
                                                  detector, grid_response, context)
        multiple_collision_only = combined_scatter_image(library, rho, torch.zeros_like(phi_uncollided), grid, f, T, L,
                                                         geometry, 0.3, q, mu_coarse_values, PROJECTION,
                                                         detector, grid_response, context)
        _assert_images_close(first_order_only, first_order, msg=msg + " first-collision alone")
        _assert_images_close(multiple_collision_only, multiple_collision, msg=msg + " multiple-collision alone")
        if f == 1:  # the multiple-collision source attenuated at the cell centers: the single-grid image of S_MC
            reference = scatter_image_from_source(
                lambda directions: scatter_source_at_directions(T, real_spherical_harmonics(L, directions)),
                coarse, geometry, 0.3, q, mu_coarse_values, PROJECTION, detector, grid_response, library,
                context=context)
            _assert_images_close(multiple_collision_only, reference, msg=msg + " multiple-collision single grid")
    # built-in context, 6-D T, mu_t_coarse None (true coarse attenuation), argument checks
    coarse = grid.coarsen(2)
    T = (torch.rand(3, num_spherical_harmonics(L), *coarse.shape, dtype=F64) * 0.01).to(device)
    T6 = torch.stack([T, torch.randn_like(T), torch.randn_like(T), torch.randn_like(T)], dim=2)
    a = combined_scatter_image(library, rho, phi_uncollided, grid, 2, T6, L, geometry, 0.3, q, None, PROJECTION,
                               detector, grid_response, chunk=5)
    b = combined_scatter_image(library, rho, phi_uncollided, grid, 2, T, L, geometry, 0.3, q,
                               coarsen_mu_t(mu, grid, grid.coarsen(2)), PROJECTION, detector, grid_response, context)
    _assert_images_close(a, b, msg="6-D T / no context")
    with pytest.raises(ValueError, match="moments"):
        combined_scatter_image(library, rho, phi_uncollided, grid, 2, T, 3, geometry, 0.3, q, None, PROJECTION,
                               detector, grid_response, context)
    with pytest.raises(ValueError, match="scatter_source_moments"):
        combined_scatter_image(library, rho, phi_uncollided, grid, 2, T[:, :, :1], L, geometry, 0.3, q, None,
                               PROJECTION, detector, grid_response, context)
    with pytest.raises(ValueError, match="does not match"):
        combined_scatter_image(library, rho, phi_uncollided, grid, 2, T, L, geometry, 0.4, q, None, PROJECTION,
                               detector, grid_response, context)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_combined_scatter_image_wlab_cuda():
    """WLAB 2 cm (padded), synthetic library, float32 CUDA, the default (fused) Siddon projectors, factors 1 and 2:
    the combined image equals its first-order + multiple-collision reference to 1e-6 (every key), and each source alone
    its term."""
    from citrus.utilities import wlab_phantom
    from citrus.scatter_solver.scatter_image import make_direction_context

    device = torch.device("cuda")
    geometry = wlab_setup(2.0, device=device, dtype=torch.float32)[1]
    grid = Grid((10, 16, 20), 2.0)  # padded along x to be divisible by 2
    rho = wlab_phantom(grid, device=device, dtype=torch.float32)
    library = synthetic_library(num_groups=3, legendre_order=2, num_materials=7).to(device, torch.float32)
    angle = 0.3
    phi_uncollided = uncollided_fluence(library, rho, grid, geometry, angle, SourceModel(), PROJECTION)
    q = panel_direction_set(geometry, angle, (8, 16), max_polar_angle=grid, device=device, dtype=torch.float32)
    detector, grid_response = csi_response(), focused_1d_grid()
    mu = mu_t_volume(library, rho)
    L = 2
    P = SiddonProjectors()
    context = make_direction_context(q, geometry, angle, detector, grid_response, library, device, torch.float32,
                                     chunk=8)
    for f in (1, 2):
        coarse = grid.coarsen(f)
        mu_coarse_values = coarsen_mu_t(mu, grid, grid.coarsen(f))
        # a smooth multiple-collision source (the multiple-collision source follows the density), weakly anisotropic
        rc = torch.nn.functional.avg_pool3d(rho[:1].unsqueeze(0), f, f).squeeze(0)[0]
        T = torch.zeros(3, num_spherical_harmonics(L), *coarse.shape, device=device)
        T[:, 0] = 0.01 * rc
        T[:, 2] = 0.002 * rc
        msg = f"f={f}"
        first_order, multiple_collision = _combined_references(library, rho, phi_uncollided, grid, f, T, L, geometry,
                                                               angle, q, mu_coarse_values, P, detector, grid_response,
                                                               context)
        tot = combined_scatter_image(library, rho, phi_uncollided, grid, f, T, L, geometry, angle, q, mu_coarse_values,
                                     P, detector, grid_response,
                                     context)
        _assert_images_close(tot, {k: first_order[k] + multiple_collision[k] for k in IMAGE_KEYS}, msg=msg)
        first_order_only = combined_scatter_image(library, rho, phi_uncollided, grid, f, torch.zeros_like(T), L,
                                                  geometry, angle,
                                                  q, mu_coarse_values, P,
                                                  detector, grid_response, context)
        _assert_images_close(first_order_only, first_order, msg=msg + " first-collision alone")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("footprint", [False, True])
def test_combined_scatter_image_fused_matches_torch(footprint, monkeypatch):
    """float32 CUDA, WLAB 2 cm: the combined pass with the fused emission kernel (first-collision source, the cell's
    multiple-collision source, in-cell attenuation and the mean-height sums in one kernel per chunk) == the PyTorch
    composition to 1e-5 (every key), with and without the finite-cone footprint, whose mean heights then come from the
    kernel's sums."""
    from citrus.utilities import wlab_phantom
    from citrus.scatter_solver.kernels import uncollided_kernels
    from citrus.scatter_solver.scatter_image import make_direction_context

    device = torch.device("cuda")
    if not uncollided_kernels.kernel_usable(device)[0]:
        pytest.skip("first-collision kernels unusable on this machine")
    geometry = wlab_setup(2.0, device=device, dtype=torch.float32)[1]
    grid = Grid((10, 16, 20), 2.0)  # padded along x to be divisible by 2
    rho = wlab_phantom(grid, device=device, dtype=torch.float32)
    library = synthetic_library(num_groups=3, legendre_order=2, num_materials=7).to(device, torch.float32)
    angle = 0.3
    phi_uncollided = uncollided_fluence(library, rho, grid, geometry, angle, SourceModel(), PROJECTION)
    q = panel_direction_set(geometry, angle, (8, 16), max_polar_angle=grid, device=device, dtype=torch.float32)
    detector, grid_response = csi_response(), focused_1d_grid()
    coarse = grid.coarsen(2)
    mu_coarse_values = coarsen_mu_t(mu_t_volume(library, rho), grid, coarse)
    L = 2
    g = torch.Generator().manual_seed(3)
    T = (torch.rand(3, num_spherical_harmonics(L), *coarse.shape, generator=g) * 0.01).to(device)
    context = make_direction_context(q, geometry, angle, detector, grid_response, library, device, torch.float32,
                                     chunk=8)

    def image():
        return scatter_image_module.combined_scatter_image(library, rho, phi_uncollided, grid, 2, T, L, geometry,
                                                           angle, q, mu_coarse_values, PROJECTION, detector,
                                                           grid_response, context, footprint=footprint)

    fused = image()
    with monkeypatch.context() as mp:
        mp.setattr(uncollided_kernels, "fused_kernels_usable", lambda *args, **kwargs: False)
        mp.setattr(scatter_image_module, "first_collision_source_at_directions",
                   functools.partial(first_collision_source_at_directions, backend="torch"))
        reference = image()
    _assert_images_close(fused, reference, rtol=1e-5, msg=f"footprint={footprint}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_combined_scatter_image_no_sync():
    """The combined pass (one projection per chunk, with and without the finite-cone footprint) runs its whole chunk
    loop without any host-device synchronization with a prebuilt context and the fused Siddon projectors (sync debug
    mode "error")."""
    from citrus.utilities import wlab_phantom
    from citrus.scatter_solver.kernels import ray_tracing_kernels
    from citrus.scatter_solver.scatter_image import make_direction_context

    device = torch.device("cuda")
    if not ray_tracing_kernels.kernels_usable(device)[0]:
        pytest.skip("Triton kernels unusable on this machine")
    geometry = wlab_setup(2.0, device=device, dtype=torch.float32)[1]
    grid = Grid((10, 16, 20), 2.0)  # padded along x to be divisible by 2
    rho = wlab_phantom(grid, device=device, dtype=torch.float32)
    library = synthetic_library(num_groups=3, legendre_order=2, num_materials=7).to(device, torch.float32)
    angle = 0.3
    phi_uncollided = uncollided_fluence(library, rho, grid, geometry, angle, SourceModel(), PROJECTION)
    q = panel_direction_set(geometry, angle, (4, 8), max_polar_angle=grid, device=device, dtype=torch.float32)
    detector, grid_response = csi_response(), focused_1d_grid()
    P = SiddonProjectors()
    coarse = grid.coarsen(2)
    mu_coarse_values = coarsen_mu_t(mu_t_volume(library, rho), grid, grid.coarsen(2))
    L = 2
    T = torch.rand(3, num_spherical_harmonics(L), *coarse.shape, device=device) * 0.01
    context = make_direction_context(q, geometry, angle, detector, grid_response, library, device, torch.float32,
                                     chunk=8)
    pairs = material_pairs(rho)  # built once per density field (ScatterSolver.material_fields): one host sync

    def images():
        out = []
        for fp in (False, True):
            out.append(scatter_image_module.combined_scatter_image(library, rho, phi_uncollided, grid, 2, T, L,
                                                                   geometry, angle, q, mu_coarse_values, P, detector,
                                                                   grid_response, context, footprint=fp, pairs=pairs))
        return out

    reference = images()  # compiles; the direction storage of the context is read to the host once here
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")  # any stream synchronization inside the calls raises
    try:
        got = images()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    for a, b in zip(got, reference):
        _assert_images_close(a, b, rtol=1e-5, msg="no-sync rerun")
