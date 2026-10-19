"""Curved (cylindrical) panel and helical views (citrus/scatter_solver/geometry.py and their consumers): the geometry
follows MBIRTORCH's projector exactly, the Triton kernels follow the PyTorch path on a curved panel with a helical z
shift, the incidence angles follow the cylinder, and a scatter solve on a curved panel of a small cone is close to the
flat-panel solve; a flat circular geometry is unchanged."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from citrus.citrus import interpolation_weights
from citrus.scatter_solver.cross_sections import synthetic_library
from citrus.scatter_solver.detector import csi_response, detector_angle, focused_1d_grid, grid_angles, no_grid
from citrus.scatter_solver.geometry import ConeGeometry, Grid
from citrus.scatter_solver.ray_tracing import SiddonProjectors
from citrus.utilities import helical_scan, view_angles, wlab_geometry

F64 = torch.float64
CUDA = torch.cuda.is_available()
cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")

FLAT = ConeGeometry(100.0, 150.0, num_det_rows=41, num_det_channels=61, delta_det_row=1.0, delta_det_channel=1.0,
                    det_row_offset=-0.4, det_channel_offset=0.3)
CURVED = replace(FLAT, use_curved_detector=True)
HELICAL_ANGLES = np.array([0.0, 0.7, 2.0, 5.5, 9.0])
HELICAL_SHIFTS = np.array([-4.0, -1.0, 0.5, 3.0, 6.0])


# ---- the geometry ---------------------------------------------------------------------------------------------

def test_flat_circular_geometry_is_unchanged():
    geometry = wlab_geometry()
    assert not geometry.use_curved_detector and not geometry.is_helical and geometry.helical_z_shifts == ()
    for angle in (0.0, 0.3, 2.0):
        c, s = math.cos(angle), math.sin(angle)
        assert geometry.source_position(angle) == (s * 100.0, c * 100.0, 0.0)
        assert geometry.panel_center(angle) == (s * -50.0, c * -50.0, 0.0)
        u, v = geometry.pixel_uv(dtype=F64)
        u_hat, v_hat, _ = (torch.tensor(a, dtype=F64) for a in geometry.panel_axes(angle))
        expected = (torch.tensor(geometry.panel_center(angle), dtype=F64) + u[None, :, None] * u_hat
                    + v[:, None, None] * v_hat)
        assert torch.equal(geometry.pixel_centers(angle, dtype=F64), expected)


def test_curved_pixels_lie_on_the_cylinder():
    for angle in (0.0, 0.7, 4.0):
        pixels = CURVED.pixel_centers(angle, dtype=F64)
        source = torch.tensor(CURVED.source_position(angle), dtype=F64)
        d = pixels - source
        # horizontal distance SDD from the vertical axis through the source
        assert torch.allclose(torch.linalg.norm(d[..., :2], dim=-1), torch.full(d.shape[:2], 150.0, dtype=F64),
                              atol=1e-10)
        # the arc coordinate and the height v cos(gamma) of MBIRTORCH's rows
        u, v = CURVED.pixel_uv(dtype=F64)
        gamma = u / 150.0
        view = CURVED.recon_to_view(d, angle)
        assert torch.allclose(torch.atan2(view[..., 0], -view[..., 1]), gamma[None].expand(41, 61), atol=1e-12)
        assert torch.allclose(view[..., 2], v[:, None] * torch.cos(gamma)[None], atol=1e-10)
        # round trip to the panel coordinates and the central column of the flat panel
        pu, pv = CURVED.panel_uv_of_points(pixels, angle)
        assert torch.allclose(pu, u[None].expand(41, 61), atol=1e-10)
        assert torch.allclose(pv, v[:, None].expand(41, 61), atol=1e-10)
        # local axes: orthonormal, the normal toward the cylinder axis, u_hat along the arc
        u_hat, v_hat, n_hat = CURVED.local_panel_axes(pixels, angle)
        horizontal = -d[..., :2] / torch.linalg.norm(d[..., :2], dim=-1, keepdim=True)
        assert torch.allclose(n_hat[..., :2], horizontal, atol=1e-12) and float(n_hat[..., 2].abs().max()) == 0.0
        assert torch.allclose(torch.linalg.cross(u_hat, v_hat, dim=-1), -n_hat, atol=1e-12)
        center_axes = [torch.tensor(a, dtype=F64) for a in CURVED.panel_axes(angle)]
        column = int(torch.argmin(u.abs()))
        assert torch.allclose(n_hat[:, column], center_axes[2].expand(41, 3), atol=1e-2)
        # one sub-pixel per pixel is the pixel center
        assert torch.allclose(CURVED.subpixel_centers(angle, 1, dtype=F64)[:, :, 0, 0], pixels, atol=1e-12)


def test_cylinder_distance():
    angle = 0.7
    source = torch.tensor(CURVED.source_position(angle), dtype=F64)
    directions = CURVED.primary_directions(angle, dtype=F64)
    pixels = CURVED.pixel_centers(angle, dtype=F64)
    t = CURVED.panel_distance_along(source.expand_as(directions), directions, angle)
    assert torch.allclose(source + t[..., None] * directions, pixels, atol=1e-9)
    # from points inside the cylinder in random directions the hit is on the cylinder; a vertical ray never hits
    generator = torch.Generator().manual_seed(3)
    points = 20.0 * torch.randn(200, 3, generator=generator, dtype=F64)
    omega = torch.nn.functional.normalize(torch.randn(200, 3, generator=generator, dtype=F64), dim=-1)
    t = CURVED.panel_distance_along(points, omega, angle)
    hit = points + t[:, None] * omega
    assert bool((t > 0).all())
    assert torch.allclose(torch.linalg.norm((hit - source)[:, :2], dim=-1), torch.full((200,), 150.0, dtype=F64),
                          atol=1e-9)
    vertical = CURVED.panel_distance_along(points[:1], torch.tensor([[0.0, 0.0, 1.0]], dtype=F64), angle)
    assert math.isinf(float(vertical))


def test_helical_table():
    geometry = CURVED.with_helical_z_shifts(HELICAL_ANGLES, HELICAL_SHIFTS)
    assert geometry.is_helical and geometry.use_curved_detector
    for a, z in zip(HELICAL_ANGLES, HELICAL_SHIFTS):
        assert geometry.z_shift(a) == z
        assert geometry.source_position(a)[2] == z and geometry.panel_center(a)[2] == z
    assert geometry.z_shift(1.35) == pytest.approx(-1.0 + 1.5 * 0.65 / 1.3)
    assert geometry.z_shift(-1.0) == -4.0 and geometry.z_shift(20.0) == 6.0  # held outside the table
    assert np.array_equal(geometry.z_shifts(HELICAL_ANGLES), HELICAL_SHIFTS)
    # a descending helix is stored ascending
    reversed_geometry = CURVED.with_helical_z_shifts(HELICAL_ANGLES[::-1], HELICAL_SHIFTS[::-1])
    assert reversed_geometry == geometry
    with pytest.raises(ValueError):
        CURVED.with_helical_z_shifts([0.0, 1.0, 0.5], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        CURVED.with_helical_z_shifts([0.0, 1.0], [0.0])
    # a constant shift is not helical; pixels move with the shift
    shifted = FLAT.with_helical_z_shifts([0.0, 1.0], [2.5, 2.5])
    assert not shifted.is_helical
    assert torch.allclose(shifted.pixel_centers(0.4, dtype=F64)[..., 2],
                          FLAT.pixel_centers(0.4, dtype=F64)[..., 2] + 2.5)
    # helical_scan: a constant pitch per turn, centred on the middle of the travel
    angles = view_angles(64, end_view_angle=4 * math.pi)
    helix = helical_scan(FLAT, angles, pitch_cm=10.0, center_cm=1.0)
    z = helix.z_shifts(angles)
    assert np.allclose(np.diff(z), 10.0 * np.diff(angles) / (2 * math.pi))
    assert 0.5 * (z[0] + z[-1]) == pytest.approx(1.0)


def _mbirtorch_indices(geometry: ConeGeometry, angles: np.ndarray, points: np.ndarray):
    from citrus.recon.mbir_bridge import make_cone_model

    model = make_cone_model(Grid((9, 9, 9), 1.0), geometry, angles, device="cpu")
    view_params = torch.as_tensor(np.asarray(model.get_params("view_params_array"), dtype=np.float64))
    rows, channels = model._project_points_batch(torch.as_tensor(points), view_params)
    return rows.numpy(), channels.numpy()


@pytest.mark.parametrize("curved", [False, True], ids=["flat", "curved"])
def test_points_project_like_mbirtorch(curved):
    """A point projects to the same fractional detector indices through MBIRTORCH's projector and through the
    ConeGeometry (the ray from the focal spot through the point to the panel surface)."""
    pytest.importorskip("mbirtorch")
    geometry = (CURVED if curved else FLAT).with_helical_z_shifts(HELICAL_ANGLES, HELICAL_SHIFTS)
    points = np.array([[4.0, -3.0, 5.0], [0.0, 0.0, 0.0], [-9.0, 7.5, -8.0], [12.0, 10.0, 2.0]])
    rows, channels = _mbirtorch_indices(geometry, HELICAL_ANGLES, points)
    p = torch.as_tensor(points, dtype=F64)
    for i, angle in enumerate(HELICAL_ANGLES):
        source = torch.tensor(geometry.source_position(angle), dtype=F64)
        omega = torch.nn.functional.normalize(p - source, dim=-1)
        hit = source + geometry.panel_distance_along(source.expand_as(p), omega, angle)[:, None] * omega
        u, v = geometry.panel_uv_of_points(hit, angle)
        channel = (u + geometry.det_channel_offset) / geometry.delta_det_channel + (geometry.num_det_channels - 1) / 2
        row = (v + geometry.det_row_offset) / geometry.delta_det_row + (geometry.num_det_rows - 1) / 2
        # MBIRTORCH keeps the angles and shifts in float32
        assert np.allclose(channel.numpy(), channels[i], atol=2e-5), (angle, channel, channels[i])
        assert np.allclose(row.numpy(), rows[i], atol=2e-5), (angle, row, rows[i])


@cuda
@pytest.mark.parametrize("curved", [False, True], ids=["flat", "curved"])
def test_blob_projects_like_mbirtorch(curved):
    """A small blob forward projected by MBIRTORCH lands where the solver's Siddon projector on the ConeGeometry
    panel puts it, for helical views (image centroids within 0.15 pixel; the sums, a check of MBIRTORCH's footprint
    model of a blob of eight voxels, within 6 %)."""
    pytest.importorskip("mbirtorch")
    from citrus.recon.mbir_bridge import forward_project, make_cone_model

    base = ConeGeometry(60.0, 90.0, num_det_rows=32, num_det_channels=48, delta_det_row=0.5, delta_det_channel=0.5,
                        det_channel_offset=0.2, use_curved_detector=curved)
    angles = np.array([0.0, 0.9, 2.3, 4.4])
    geometry = base.with_helical_z_shifts(angles, [-2.0, -0.5, 1.0, 2.5])
    grid = Grid((16, 16, 16), 0.5)
    volume = torch.zeros(grid.shape, device="cuda")
    volume[3:5, 11:13, 12:14] = 1.0  # off-center blob, x ~ +2, y ~ -2, z ~ +2.5
    sinogram = forward_project(make_cone_model(grid, geometry, angles, device="cuda"), volume)
    n = 4
    fine = replace(geometry, num_det_rows=geometry.num_det_rows * n, num_det_channels=geometry.num_det_channels * n,
                   delta_det_row=geometry.delta_det_row / n, delta_det_channel=geometry.delta_det_channel / n)
    projector = SiddonProjectors(backend="torch")
    for i, angle in enumerate(angles):
        siddon = projector.primary_project(volume.double()[None], grid, fine, float(angle))[0]
        siddon = siddon.reshape(geometry.num_det_rows, n, geometry.num_det_channels, n).mean(dim=(1, 3))
        centroids = []
        for image in (sinogram[i].double(), siddon):
            w = image.sum()
            r = (image.sum(1) * torch.arange(image.shape[0], device=image.device)).sum() / w
            c = (image.sum(0) * torch.arange(image.shape[1], device=image.device)).sum() / w
            centroids.append((float(r), float(c), float(w)))
        (rm, cm, wm), (rs, cs, ws) = centroids
        assert abs(rm - rs) < 0.15 and abs(cm - cs) < 0.15, (angle, centroids)
        assert abs(wm / ws - 1) < 0.06, (angle, wm, ws)


# ---- the Triton kernels -------------------------------------------------------------------------------------

@cuda
def test_kernels_follow_torch_on_a_curved_helical_panel():
    from citrus.scatter_solver.kernels import ray_tracing_kernels as kernels

    ok, why = kernels.kernels_usable(torch.device("cuda"))
    if not ok:
        pytest.skip(why)
    grid = Grid((12, 14, 10), 1.0, voxel_slice_aspect=0.9, recon_slice_offset=0.3)
    # the cylinder (radius 11 about the focal spot) cuts the grid box: the downstream end point matters
    geometry = ConeGeometry(7.0, 11.0, num_det_rows=13, num_det_channels=17, delta_det_row=0.9, delta_det_channel=1.1,
                            det_row_offset=0.2, det_channel_offset=-0.6, use_curved_detector=True)
    geometry = geometry.with_helical_z_shifts([0.0, 2.0], [-1.0, 3.0])
    generator = torch.Generator().manual_seed(4)
    mu = (0.2 * torch.rand(3, *grid.shape, generator=generator)).cuda()
    reference, triton = SiddonProjectors(backend="torch"), SiddonProjectors(backend="triton")
    angle = 1.1
    directions = torch.nn.functional.normalize(torch.tensor([[0.3, -1.0, 0.2], [-0.5, -1.0, -0.3], [0.0, -1.0, 0.0],
                                                             [0.9, -0.3, 0.1], [0.2, 1.0, 0.0]]), dim=-1)
    directions = geometry.view_to_recon(directions, angle).cuda()
    field = torch.rand(directions.shape[0], 3, *grid.shape, generator=generator).cuda()

    def close(a, b):
        finite = torch.isfinite(a)
        assert torch.equal(finite, torch.isfinite(b))
        assert float((a[finite] - b[finite]).abs().max()) <= 2e-5 * float(a[finite].abs().max())

    close(reference.upstream_cumulative(mu, grid, geometry, angle),
          triton.upstream_cumulative(mu, grid, geometry, angle))
    for n in (1, 3):
        close(reference.primary_project(mu, grid, geometry, angle, samples_per_pixel_axis=n),
              triton.primary_project(mu, grid, geometry, angle, samples_per_pixel_axis=n))
    down = reference.downstream_cumulative(mu, grid, geometry, angle, directions)
    close(down, triton.downstream_cumulative(mu, grid, geometry, angle, directions))
    close(reference.oblique_project(field, grid, geometry, angle, directions),
          triton.oblique_project(field, grid, geometry, angle, directions))
    close(reference.oblique_project(field[0], grid, geometry, angle, directions),
          triton.oblique_project(field[0], grid, geometry, angle, directions))
    # the cylinder, not the plane, ends the downstream rays, and the pixels are not those of the flat panel
    flat = replace(geometry, use_curved_detector=False)
    flat_down = reference.downstream_cumulative(mu, grid, flat, angle, directions)
    finite = torch.isfinite(down)
    assert float((down[finite] - flat_down[finite]).abs().max()) > 1e-3
    assert not torch.allclose(reference.primary_project(mu, grid, flat, angle),
                              reference.primary_project(mu, grid, geometry, angle))


# ---- incidence angles and the direction context -------------------------------------------------------------

def test_curved_incidence_angles_of_the_primary():
    angle = 0.4
    pixels = CURVED.pixel_centers(angle, dtype=F64)
    omega = CURVED.primary_directions(angle, dtype=F64)
    u, v = CURVED.pixel_uv(dtype=F64)
    height = v[:, None] * torch.cos(u / 150.0)[None]
    expected = torch.atan(height.abs() / 150.0)  # the cylinder normal is horizontal
    assert torch.allclose(detector_angle(omega, CURVED, angle, pixels), expected, atol=1e-12)
    # without the pixels the angle is measured from the normal at the panel center (the flat rule)
    flat_rule = detector_angle(omega, CURVED, angle)
    assert float((flat_rule - expected).abs().max()) > 0.1
    # a grid focused at the focal spot: a primary ray has zero grid angles on every pixel of the cylinder
    theta1, theta2 = grid_angles(omega, pixels, CURVED, angle, 150.0)
    assert float(theta1.abs().max()) < 1e-12 and float(theta2.abs().max()) < 1e-12
    # the flat panel is unchanged by the pixel argument
    flat_pixels = FLAT.pixel_centers(angle, dtype=F64)
    flat_omega = FLAT.primary_directions(angle, dtype=F64)
    assert torch.equal(detector_angle(flat_omega, FLAT, angle, flat_pixels), detector_angle(flat_omega, FLAT, angle))


@pytest.mark.parametrize("focused_grid", [False, True], ids=["no_grid", "grid"])
def test_curved_direction_context(focused_grid):
    from citrus.scatter_solver.scatter_image import make_direction_context, panel_direction_set

    geometry = replace(CURVED, num_det_rows=7, num_det_channels=9, delta_det_channel=6.0, delta_det_row=3.0)
    angle = 0.3
    library = synthetic_library(3, 1)
    grid_response = focused_1d_grid() if focused_grid else no_grid()
    quadrature = panel_direction_set(geometry, angle, (4, 8), max_polar_angle=0.6, dtype=F64)
    detector = csi_response()
    context = make_direction_context(quadrature, geometry, angle, detector, grid_response, library, "cpu", F64)
    assert not context.separable and context.grid_per_pixel == focused_grid
    Cd = geometry.num_det_channels
    pixels = geometry.pixel_centers(angle, dtype=F64)
    normals = geometry.local_panel_axes(pixels[0], angle)[2]
    cos = (context.omega @ (-normals).t()).clamp(0.0, 1.0)
    assert torch.allclose(context.planar_weight, context.weight[:, None] * cos)
    # a unit image per direction: the planar fluence is sum_k w_k cos(theta0_kc) on every row
    accumulator = context.new_accumulator()
    for i, (start, stop) in enumerate(context.chunk_bounds):
        context.accumulate(accumulator, torch.ones(stop - start, 3, 7, Cd, dtype=F64), i)
    images = context.finalize(accumulator)
    assert torch.allclose(images["planar_fluence"][1, 3], (context.weight[:, None] * cos).sum(0))
    assert torch.allclose(images["fluence"], torch.full_like(images["fluence"], float(context.weight.sum())))
    # the signal weights against a direct evaluation per pixel
    energy = library.group_energy.to(F64)
    theta0 = torch.arccos(cos)  # [K, Cd]
    response = detector.at(energy, theta0)  # [G, K, Cd]
    th1, th2 = grid_angles(context.omega, pixels, geometry, angle, grid_response.focal_distance_cm)
    transmission = grid_response.at(energy, th1, th2)  # [G, K, Rd, Cd]
    expected = (context.weight[None, :, None, None] * cos[None, :, None, :] * response[:, :, None, :]
                * transmission).sum(1)
    assert torch.allclose(images["signal_per_group"], expected, rtol=1e-10, atol=0.0)


# ---- the scatter solve -------------------------------------------------------------------------------------

def test_curved_scatter_solve_is_close_to_flat_for_a_small_cone():
    """A narrow panel (a fan of about 4 degrees): the curved and flat panels nearly coincide, so the scatter and the
    primary of a view agree closely; both are finite and positive."""
    from citrus.scatter_solver.config import SolverConfig
    from citrus.scatter_solver.scatter_solver import ScatterSolver
    from citrus.utilities import wlab_phantom

    device = torch.device("cuda" if CUDA else "cpu")
    grid = Grid((10, 16, 20), 2.0)
    config = SolverConfig(num_groups=3, legendre_order=2, num_polar=4, num_azimuthal=8, cone_num_polar=6,
                          cone_num_azimuthal=12, transport_cell_cm=4.0, max_source_iterations=6)
    library = synthetic_library(3, 3, num_materials=7)
    flat = ConeGeometry(100.0, 150.0, num_det_rows=24, num_det_channels=16, delta_det_row=1.5, delta_det_channel=0.7)
    rho = wlab_phantom(grid, device=device, dtype=torch.float32)
    images = {}
    for name, geometry in (("flat", flat), ("curved", replace(flat, use_curved_detector=True))):
        solver = ScatterSolver(library, config, geometry, SiddonProjectors(), csi_response(), no_grid(),
                               device=device, dtype=torch.float32)
        result = solver.solve_view(rho, grid, 0.5)
        images[name] = (result.scatter["signal"], result.primary["signal"])
    for scatter, primary in images.values():
        assert bool(torch.isfinite(scatter).all()) and float(scatter.min()) > 0 and float(primary.min()) > 0
    (scatter_flat, primary_flat), (scatter_curved, primary_curved) = images["flat"], images["curved"]
    assert float((scatter_curved - scatter_flat).norm() / scatter_flat.norm()) < 5e-3
    assert float((primary_curved - primary_flat).norm() / primary_flat.norm()) < 5e-3
    assert not torch.equal(scatter_curved, scatter_flat)


# ---- curved panel with an anti-scatter grid built from its septa --------------------------------------------

@pytest.mark.parametrize("septa_along", ["channels", "rows"])
def test_curved_panel_primary_passes_a_focused_grid(septa_along):
    """On a curved panel a grid focused at the source-detector distance meets the primary ray of every pixel at
    theta1 = 0, with the septa along the channels (focal line at the central row of each column) and along the rows
    (focal line on each pixel's normal, the source axis); a tilted direction does not."""
    angle = 0.8
    pixels = CURVED.pixel_centers(angle, dtype=F64)
    toward_pixels = CURVED.primary_directions(angle, dtype=F64)  # [Rd, Cd, 3], from the source to the pixels
    theta1, _ = grid_angles(toward_pixels, pixels, CURVED, angle, None, septa_along)
    assert float(theta1.abs().max()) < 1e-9
    tilt = torch.tensor(CURVED.panel_axes(angle)[0 if septa_along == "rows" else 1], dtype=F64)
    tilted = toward_pixels + 0.05 * tilt
    tilted = tilted / tilted.norm(dim=-1, keepdim=True)
    theta1, _ = grid_angles(tilted, pixels, CURVED, angle, None, septa_along)
    assert float(theta1.abs().min()) > 0.01


@pytest.mark.parametrize("septa_along", ["channels", "rows"])
def test_curved_panel_with_anti_scatter_grid(septa_along):
    """A scatter solve on a curved panel behind a lead grid from its septa (pitch = pixel pitch): finite, and the
    grid lowers the scatter-to-primary ratio well below that of the same panel without a grid."""
    from citrus.scatter_solver.config import SolverConfig
    from citrus.scatter_solver.detector import anti_scatter_grid
    from citrus.scatter_solver.photon_data import resolve_backend
    from citrus.scatter_solver.scatter_solver import ScatterSolver
    from citrus.utilities import wlab_phantom

    try:
        backend = resolve_backend()
    except Exception as exc:  # no photon data
        pytest.skip(f"photon data unavailable: {exc}")
    device = torch.device("cuda" if CUDA else "cpu")
    grid = Grid((10, 16, 20), 2.0)
    config = SolverConfig(num_groups=3, legendre_order=2, num_polar=4, num_azimuthal=8, cone_num_polar=6,
                          cone_num_azimuthal=12, transport_cell_cm=4.0, max_source_iterations=6)
    library = synthetic_library(3, 3, num_materials=7)
    geometry = ConeGeometry(100.0, 150.0, num_det_rows=24, num_det_channels=32, delta_det_row=1.5,
                            delta_det_channel=1.5, use_curved_detector=True)
    septa = anti_scatter_grid(geometry, 0.05, 6.0, septa_along=septa_along, backend=backend)
    rho = wlab_phantom(grid, device=device, dtype=torch.float32)
    ratios = {}
    for name, grid_response in (("none", no_grid()), ("grid", septa)):
        solver = ScatterSolver(library, config, geometry, SiddonProjectors(), csi_response(), grid_response,
                               device=device, dtype=torch.float32)
        result = solver.solve_view(rho, grid, 0.5)
        scatter, primary = result.scatter["signal"], result.primary["signal"]
        assert bool(torch.isfinite(scatter).all()) and float(scatter.min()) >= 0 and float(primary.min()) > 0
        ratios[name] = float(scatter.sum() / primary.sum())
    assert ratios["grid"] < 0.5 * ratios["none"]


# ---- sparse views of a helical scan -------------------------------------------------------------------------

@pytest.mark.parametrize("order", [1, 2, 3])
def test_helical_interpolation_runs_along_the_helix(order):
    """Two turns of views: the interpolation of the skipped views uses the computed views next to them along the
    helix (never a view a turn away at another z) and is exact for linear functions of the unwrapped angle (between
    the first and the last computed view; the views after the last one get its value)."""
    angles = view_angles(48, end_view_angle=4 * math.pi)
    computed = np.arange(0, 48, 4)
    weights = interpolation_weights(angles, computed, order, helical=True)
    values = 3.0 + 0.7 * angles
    inside = slice(0, int(computed[-1]) + 1)
    assert np.allclose((weights @ values[computed])[inside], values[inside])
    assert np.allclose((weights @ values[computed])[computed[-1]:], values[computed[-1]])
    for v in range(48):
        used = computed[np.abs(weights[v]) > 1e-12]
        assert np.abs(angles[used] - angles[v]).max() < 3 * 4 * (angles[1] - angles[0]) + 1e-9
    # the circular rule would mix the turns
    circular = interpolation_weights(angles, computed, order)
    assert not np.allclose((circular @ values[computed])[inside], values[inside])


def test_short_scan_weights_of_curved_and_helical_scans():
    from citrus.recon.mbir_bridge import short_scan_weights
    from citrus.utilities import fan_angle

    flat = replace(FLAT, num_det_channels=101, delta_det_channel=1.0, det_channel_offset=0.0)
    curved = replace(flat, use_curved_detector=True)
    assert fan_angle(curved) == pytest.approx(101 / 150.0) and fan_angle(flat) == pytest.approx(
        2 * math.atan(50.5 / 150.0))
    angles = view_angles(200, curved)  # the short scan of the curved panel's fan angle
    weights_curved = short_scan_weights(curved, angles)
    weights_flat = short_scan_weights(flat, angles)
    assert weights_curved is not None and weights_flat is not None
    # the central channel has the same fan angle (0) on both panels; the others do not
    assert torch.allclose(weights_curved[:, 0, 50], weights_flat[:, 0, 50])
    assert not torch.allclose(weights_curved, weights_flat)
    # no Parker weights for a helical scan, even over less than a turn
    assert short_scan_weights(helical_scan(curved, angles, pitch_cm=5.0), angles) is None
