"""Parallel-beam geometry (``ParallelGeometry`` in citrus/scatter_solver/geometry.py and its consumers): the frames
follow MBIRTORCH's ``ParallelBeamModel`` exactly, the Triton kernels follow the PyTorch path in their parallel mode,
the uncollided fluence, the primary, the grid and the emission profile follow the beam, the scatter falls as the air
gap grows, and a cone beam with a receding source approaches the parallel beam."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from citrus.scatter_solver.config import SolverConfig
from citrus.scatter_solver.cross_sections import group_transmission, synthetic_library
from citrus.scatter_solver.detector import (csi_response, default_focal_distance, detector_angle, focused_1d_grid,
                                            grid_angles, no_grid)
from citrus.scatter_solver.geometry import ConeGeometry, Grid, ParallelGeometry
from citrus.scatter_solver.ray_tracing import SiddonProjectors
from citrus.scatter_solver.scatter_solver import ScatterSolver, emission_profile_from_open_field
from citrus.scatter_solver.source import SourceModel, bowtie_emission_profile
from citrus.scatter_solver.uncollided import primary_directions_to_voxels, uncollided_fluence
from citrus.utilities import fan_angle, view_angles, wlab_phantom

F64 = torch.float64
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")

PARALLEL = ParallelGeometry(40.0, num_det_rows=21, num_det_channels=31, delta_det_row=1.0, delta_det_channel=1.2,
                            det_row_offset=-0.4, det_channel_offset=0.3)
SMALL_CONFIG = SolverConfig(num_groups=3, legendre_order=2, num_polar=4, num_azimuthal=8, cone_num_polar=6,
                            cone_num_azimuthal=12, transport_cell_cm=4.0, max_source_iterations=6)


def _solver(geometry, grid_response=None, source=None, device=DEVICE, dtype=torch.float32, config=SMALL_CONFIG):
    return ScatterSolver(synthetic_library(3, 3, num_materials=7), config, geometry, SiddonProjectors(),
                         csi_response(), no_grid() if grid_response is None else grid_response, source=source,
                         device=device, dtype=dtype)


# ---- the geometry ---------------------------------------------------------------------------------------------

def test_parallel_frames():
    grid = Grid((8, 10, 12), 1.5)
    for angle in (0.0, 0.3, 2.0, 4.0):
        beam = torch.tensor(PARALLEL.beam_direction(angle), dtype=F64)
        u_hat, v_hat, n_hat = (torch.tensor(a, dtype=F64) for a in PARALLEL.panel_axes(angle))
        assert torch.allclose(beam, -n_hat) and torch.allclose(torch.linalg.cross(u_hat, v_hat, dim=-1), -n_hat)
        pixels = PARALLEL.pixel_centers(angle, dtype=F64)
        # the panel plane at axis_detector_dist downstream of the axis; u along +x_view, v along +z
        assert torch.allclose((pixels * beam).sum(-1), torch.full((21, 31), 40.0, dtype=F64))
        view = PARALLEL.recon_to_view(pixels, angle)
        u, v = PARALLEL.pixel_uv(dtype=F64)
        assert torch.allclose(view[..., 0], u[None].expand(21, 31)) and torch.allclose(view[..., 2],
                                                                                       v[:, None].expand(21, 31))
        assert torch.allclose(view[..., 1], torch.full((21, 31), -40.0, dtype=F64))
        pu, pv = PARALLEL.panel_uv_of_points(pixels, angle)
        assert torch.allclose(pu, u[None].expand(21, 31)) and torch.allclose(pv, v[:, None].expand(21, 31))
        assert torch.allclose(PARALLEL.primary_directions(angle, dtype=F64), beam.expand(21, 31, 3))
        # the beam's ray through a point: it keeps the panel coordinates and reaches the panel after 40 - p . w
        points = torch.tensor([[3.0, -2.0, 1.0], [0.0, 0.0, 0.0], [-5.0, 6.0, -4.0]], dtype=F64)
        t = PARALLEL.panel_distance_along(points, beam, angle)
        assert torch.allclose(t, 40.0 - (points * beam).sum(-1))
        hit_u, hit_v = PARALLEL.panel_uv_of_points(points + t[:, None] * beam, angle)
        point_u, point_v = PARALLEL.panel_uv_of_points(points, angle)
        assert torch.allclose(hit_u, point_u) and torch.allclose(hit_v, point_v)
        # the primary rays start upstream of the grid box on the beam's line through their end
        starts = PARALLEL.ray_starts(points, angle, grid)
        offset = points - starts
        assert torch.allclose(offset / offset.norm(dim=-1, keepdim=True), beam.expand(3, 3))
        assert bool(((starts * beam).sum(-1) < -0.5 * math.hypot(grid.extent[0], grid.extent[1])).all())
    assert PARALLEL.is_parallel and not PARALLEL.use_curved_detector and not PARALLEL.is_helical
    assert not ConeGeometry(100.0, 150.0, 4, 4, 1.0, 1.0).is_parallel
    with pytest.raises(ValueError):
        ParallelGeometry(0.0, 4, 4, 1.0, 1.0)


def test_rows_are_the_slices_of_the_grid():
    grid = Grid((6, 8, 13), 0.7, recon_slice_offset=1.3)
    geometry = ParallelGeometry.from_grid(grid, 25.0, 17, 0.9, det_channel_offset=0.2)
    assert (geometry.num_det_rows, geometry.delta_det_row, geometry.det_channel_offset) == (13, 0.7, 0.2)
    _, v = geometry.pixel_uv(dtype=F64)
    _, _, z = grid.axis_coordinates(dtype=F64)
    assert torch.allclose(v, z)
    with pytest.raises(ValueError):
        ParallelGeometry.from_grid(replace(grid, voxel_slice_aspect=0.5), 25.0, 17, 0.9)


def test_points_project_like_mbirtorch():
    """A point projects to the same fractional detector indices through MBIRTORCH's projector and through the
    ParallelGeometry (the beam's ray through the point). MBIRTORCH's parallel beam has no slice offset (row r is
    slice r), so the grid is centred on z = 0 here; ``test_rows_are_the_slices_of_the_grid`` covers an offset."""
    pytest.importorskip("mbirtorch")
    from citrus.recon.mbir_bridge import make_model

    grid = Grid((9, 9, 21), 1.0)
    geometry = ParallelGeometry.from_grid(grid, 40.0, 31, 1.2, det_channel_offset=0.3)
    angles = np.array([0.0, 0.7, 2.0, 3.9])
    model = make_model(grid, geometry, angles, device="cpu")
    assert type(model).__name__ == "ParallelBeamModel"
    points = np.array([[4.0, -3.0, 5.0], [0.0, 0.0, 0.0], [-9.0, 7.5, -8.0], [12.0, 10.0, 2.0]])
    view_params = torch.as_tensor(np.asarray(model.get_params("angles"), dtype=np.float64))
    rows, channels = model._project_points_batch(torch.as_tensor(points), view_params)
    p = torch.as_tensor(points, dtype=F64)
    for i, angle in enumerate(angles):
        u, v = geometry.panel_uv_of_points(p, angle)
        channel = (u + geometry.det_channel_offset) / geometry.delta_det_channel + (geometry.num_det_channels - 1) / 2
        row = (v + geometry.det_row_offset) / geometry.delta_det_row + (geometry.num_det_rows - 1) / 2
        assert np.allclose(channel.numpy(), channels[i].numpy(), atol=2e-5), (angle, channel, channels[i])
        assert np.allclose(row.numpy(), rows[i].numpy(), atol=2e-5), (angle, row, rows[i])


def test_blob_projects_like_mbirtorch():
    """A small off-center blob forward projected by MBIRTORCH lands where the solver's Siddon projector puts it on the
    ParallelGeometry panel (image centroids within 0.05 pixel, sums within 2 %; the Siddon images are averages of
    8 x 8 rays per pixel)."""
    pytest.importorskip("mbirtorch")
    from citrus.recon.mbir_bridge import forward_project, make_model

    grid = Grid((16, 16, 16), 0.5)
    geometry = ParallelGeometry.from_grid(grid, 30.0, 40, 0.5, det_channel_offset=0.2)
    angles = np.array([0.0, 0.9, 2.3, 4.4])
    volume = torch.zeros(grid.shape)
    volume[3:5, 11:13, 12:14] = 1.0  # x ~ +2, y ~ -2, z ~ +2.5
    sinogram = forward_project(make_model(grid, geometry, angles, device="cpu"), volume).cpu()
    n = 8
    fine = replace(geometry, num_det_rows=geometry.num_det_rows * n, num_det_channels=geometry.num_det_channels * n,
                   delta_det_row=geometry.delta_det_row / n, delta_det_channel=geometry.delta_det_channel / n)
    projector = SiddonProjectors(backend="torch")
    for i, angle in enumerate(angles):
        siddon = projector.primary_project(volume.double()[None], grid, fine, float(angle))[0]
        siddon = siddon.reshape(geometry.num_det_rows, n, geometry.num_det_channels, n).mean(dim=(1, 3))
        centroids = []
        for image in (sinogram[i].double(), siddon):
            w = image.sum()
            r = (image.sum(1) * torch.arange(image.shape[0])).sum() / w
            c = (image.sum(0) * torch.arange(image.shape[1])).sum() / w
            centroids.append((float(r), float(c), float(w)))
        (rm, cm, wm), (rs, cs, ws) = centroids
        assert abs(rm - rs) < 0.05 and abs(cm - cs) < 0.05, (angle, centroids)
        assert abs(wm / ws - 1) < 0.02, (angle, wm, ws)


# ---- the MBIRTORCH bridge -------------------------------------------------------------------------------------

def test_parallel_model_and_filtered_back_projection():
    pytest.importorskip("mbirtorch")
    from citrus.recon.mbir_bridge import (ViewChunkProjector, direct_reconstruction, forward_project, make_model,
                                          make_parallel_model)

    grid = Grid((24, 24, 4), 0.5)
    geometry = ParallelGeometry.from_grid(grid, 30.0, 36, 0.5)
    angles = view_angles(90, geometry)
    assert angles[-1] < math.pi and fan_angle(geometry) == 0.0
    for wrong in (replace(geometry, num_det_rows=5), replace(geometry, delta_det_row=0.6),
                  replace(geometry, det_row_offset=0.3)):
        with pytest.raises(ValueError):
            make_parallel_model(grid, wrong, angles, device="cpu")
    model = make_model(grid, geometry, angles, device="cpu")
    x, y, _ = grid.axis_coordinates()
    disk = (((x[None, :] - 1.0) ** 2 + y[:, None] ** 2) < 16.0).float()[..., None].expand(grid.shape).contiguous()
    volume = 0.2 * disk
    sinogram = forward_project(model, volume)
    recon = direct_reconstruction(model, sinogram, geometry=geometry, angles=angles).cpu()
    inside = disk.bool() & (((x[None, :] - 1.0) ** 2 + y[:, None] ** 2) < 9.0)[..., None]
    assert abs(float(recon[inside].mean()) / 0.2 - 1) < 0.05
    assert float(recon[~disk.bool()].abs().mean()) < 0.02
    # the projector of chunks of views builds parallel sub-models
    projector = ViewChunkProjector(grid, geometry, angles, model=model, device="cpu")
    assert torch.allclose(projector.project(volume, 10, 20).cpu(), sinogram[10:20].cpu(), atol=1e-5)


# ---- the Triton kernels -------------------------------------------------------------------------------------

@cuda
def test_kernels_follow_torch_in_parallel_mode():
    from citrus.scatter_solver.kernels import ray_tracing_kernels, uncollided_kernels
    from citrus.scatter_solver.uncollided import _first_collision_source_torch

    ok, why = ray_tracing_kernels.kernels_usable(torch.device("cuda"))
    if not ok:
        pytest.skip(why)
    grid = Grid((12, 14, 10), 1.0, voxel_slice_aspect=0.9, recon_slice_offset=0.3)
    # the panel (5 cm from the axis) cuts the grid box: the downstream end point matters
    geometry = ParallelGeometry(5.0, num_det_rows=13, num_det_channels=17, delta_det_row=0.9, delta_det_channel=1.1,
                                det_row_offset=0.2, det_channel_offset=-0.6)
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
    close(reference.downstream_cumulative(mu, grid, geometry, angle, directions),
          triton.downstream_cumulative(mu, grid, geometry, angle, directions))
    close(reference.oblique_project(field, grid, geometry, angle, directions),
          triton.oblique_project(field, grid, geometry, angle, directions))
    # the upstream depth of a voxel is the integral along the beam from the box edge: a column of the volume
    tau = triton.upstream_cumulative(mu, grid, geometry, 0.0)  # beam along -y: rows from the last one down
    expected = torch.flip(torch.cumsum(torch.flip(mu, dims=[1]), dim=1), dims=[1]) - 0.5 * mu
    assert torch.allclose(tau, expected, atol=1e-5)
    # the first-collision source of a parallel beam: one primary direction everywhere
    library = synthetic_library(3, 1, num_eta=201)
    rho = torch.rand(2, *grid.shape, generator=generator).cuda()
    phi = torch.rand(3, *grid.shape, generator=generator).cuda()
    beam = torch.tensor([geometry.beam_direction(angle)], dtype=torch.float32)
    omegas = torch.cat([beam, -beam, directions.cpu()]).cuda()
    close(_first_collision_source_torch(library, rho, phi, grid, geometry, angle, omegas),
          uncollided_kernels.first_collision_source(library, rho, phi, grid, geometry, angle, omegas))


# ---- uncollided fluence, primary, grid and emission profile ---------------------------------------------------

def test_uncollided_fluence_of_a_parallel_beam():
    """In a uniform block the uncollided fluence is the transmission along the beam from the box edge (no inverse
    square), zero outside the beam's cross-section, and weighted by the emission profile of the pixel below."""
    library = synthetic_library(3, 1)
    grid = Grid((6, 8, 10), 1.0)
    # pixel centers at u = 0, +-1.2, +-2.4 and v = +-0.5, +-1.5, +-2.5; the beam covers |u|, |v| <= 3
    geometry = ParallelGeometry(20.0, num_det_rows=6, num_det_channels=5, delta_det_row=1.0, delta_det_channel=1.2)
    rho = torch.zeros(library.num_materials, *grid.shape, dtype=F64)
    rho[0] = 0.8
    phi = uncollided_fluence(library, rho, grid, geometry, 0.0, SourceModel(), SiddonProjectors())
    _, y, z = grid.axis_coordinates(dtype=F64)
    x, _, _ = grid.axis_coordinates(dtype=F64)
    path = (grid.extent[1] / 2 - y)[:, None, None].expand(grid.shape)  # beam along -y from the top of the box
    expected = group_transmission(library, (0.8 * path)[None], materials=[0])
    in_beam = ((x.abs() <= 3.0)[None, :, None] & (z.abs() <= 3.0)[None, None, :]).expand(grid.shape)
    assert torch.allclose(phi[:, in_beam], expected[:, in_beam], rtol=1e-10)
    assert float(phi[:, ~in_beam].abs().max()) == 0.0
    assert torch.allclose(primary_directions_to_voxels(grid, geometry, 0.0, dtype=F64),
                          torch.tensor([0.0, -1.0, 0.0], dtype=F64).expand(*grid.shape, 3))
    # an emission profile weights the voxels by the profile where the beam's ray through them meets the panel: a
    # profile linear in (u, v) is interpolated exactly between the pixel centers (u = x, v = z at angle 0)
    u, v = geometry.pixel_uv(dtype=F64)
    profile = 1.0 + 0.1 * u[None, :] + 0.05 * v[:, None]
    weighted = uncollided_fluence(library, rho, grid, geometry, 0.0, SourceModel(emission_profile=profile),
                                  SiddonProjectors())
    linear = in_beam & (x.abs() <= 2.4)[None, :, None]
    at_panel = (1.0 + 0.1 * x[None, :, None] + 0.05 * z[None, None, :]).expand(grid.shape)
    assert torch.allclose(weighted[:, linear], (phi * at_panel)[:, linear], rtol=1e-10)
    assert float(weighted[:, ~in_beam].abs().max()) == 0.0


def test_parallel_primary_and_unfocused_grid():
    """The open field of a parallel beam is flat (no inverse square), every primary ray meets the panel at normal
    incidence, and the natural grid of a parallel beam is unfocused, passing the primary at theta1 = 0."""
    geometry = ParallelGeometry(30.0, num_det_rows=8, num_det_channels=12, delta_det_row=1.5, delta_det_channel=1.5)
    angle = 0.6
    pixels = geometry.pixel_centers(angle, dtype=F64)
    omega = geometry.primary_directions(angle, dtype=F64)
    assert float(detector_angle(omega, geometry, angle, pixels).abs().max()) < 1e-7
    assert default_focal_distance(geometry) == math.inf
    assert default_focal_distance(ConeGeometry(100.0, 150.0, 4, 4, 1.0, 1.0)) == 150.0
    theta1, theta2 = grid_angles(omega, pixels, geometry, angle)  # the default: unfocused
    assert theta1.shape == (8, 12) and float(theta1.abs().max()) < 1e-12 and float(theta2.abs().max()) < 1e-12
    tilted = torch.nn.functional.normalize(omega + 0.05 * torch.tensor(geometry.panel_axes(angle)[1], dtype=F64),
                                           dim=-1)
    theta1, _ = grid_angles(tilted, pixels, geometry, angle)
    assert torch.allclose(theta1.abs(), torch.full_like(theta1, math.atan(0.05)), atol=1e-12)
    # a grid focused at 150 cm sees the parallel primary tilted toward the panel edges
    focused, _ = grid_angles(omega, pixels, geometry, angle, 150.0)
    assert float(focused.abs().max()) > 0.01
    unfocused = replace(focused_1d_grid(), focal_distance_cm=math.inf)
    open_fields = {}
    for name, grid_response in (("none", no_grid()), ("unfocused", unfocused)):
        solver = _solver(geometry, grid_response, device="cpu", dtype=F64)
        open_fields[name] = solver.open_field_signal(angle)
        assert torch.allclose(open_fields[name], open_fields[name][0, 0].expand(8, 12), rtol=1e-12)
    ratio = float(open_fields["unfocused"][0, 0] / open_fields["none"][0, 0])
    open_fraction = (1.0 / 60.0 - 0.0036) / (1.0 / 60.0)
    assert open_fraction < ratio < 1.0


def test_anti_scatter_grid_of_a_parallel_beam_is_unfocused():
    from citrus.scatter_solver.detector import anti_scatter_grid
    from citrus.scatter_solver.photon_data import resolve_backend

    try:
        backend = resolve_backend()
    except Exception as exc:  # no photon data
        pytest.skip(f"photon data unavailable: {exc}")
    geometry = ParallelGeometry(30.0, num_det_rows=8, num_det_channels=12, delta_det_row=1.5, delta_det_channel=1.5)
    assert anti_scatter_grid(geometry, 0.05, 6.0, backend=backend).focal_distance_cm == math.inf
    cone = ConeGeometry(100.0, 150.0, 8, 12, 1.5, 1.5)
    assert anti_scatter_grid(cone, 0.05, 6.0, backend=backend).focal_distance_cm == 150.0


def test_emission_profile_of_a_parallel_beam():
    """The emission profile of a measured open field (gain times a bowtie times the flat open field of the parallel
    beam) is recovered per pixel and weights the primary pixel by pixel."""
    geometry = ParallelGeometry(30.0, num_det_rows=8, num_det_channels=12, delta_det_row=1.5, delta_det_channel=1.5)
    solver = _solver(geometry, device="cpu", dtype=F64)
    bowtie = bowtie_emission_profile(geometry, 0.3)
    gain = 1.0 + 0.1 * torch.rand(8, 12, generator=torch.Generator().manual_seed(1), dtype=F64)
    measured = 7.0 * gain * bowtie * solver.open_field_signal(0.0)
    profile = emission_profile_from_open_field(measured, solver, [0.0], detector_gain=gain)
    assert torch.allclose(profile, bowtie / bowtie.mean(), rtol=1e-10)
    weighted = solver.open_field_signal(0.0, SourceModel(emission_profile=profile))
    assert torch.allclose(weighted / solver.open_field_signal(0.0), profile, rtol=1e-10)


# ---- the scatter solve ------------------------------------------------------------------------------------------

def test_scatter_falls_with_the_air_gap():
    """The same object and beam with the panel farther from the axis: the primary stays (a parallel beam has no
    inverse square) and the scatter-to-primary ratio falls (the air-gap effect)."""
    grid = Grid((10, 16, 20), 2.0)
    rho = wlab_phantom(grid, device=DEVICE, dtype=torch.float32)
    ratios, primaries = [], []
    for distance in (20.0, 40.0, 80.0):
        solver = _solver(ParallelGeometry.from_grid(grid, distance, 20, 2.0))
        result = solver.solve_view(rho, grid, 0.5)
        scatter, primary = result.scatter["signal"], result.primary["signal"]
        assert bool(torch.isfinite(scatter).all()) and float(scatter.min()) > 0 and float(primary.min()) > 0
        ratios.append(float(scatter.sum() / primary.sum()))
        primaries.append(float(primary.sum()))
    assert ratios[0] > ratios[1] > ratios[2] and ratios[2] < 0.5 * ratios[0], ratios
    assert max(primaries) / min(primaries) - 1 < 1e-4, primaries


def test_distant_cone_beam_approaches_the_parallel_beam():
    """A cone beam whose source recedes (the axis-panel distance fixed) approaches the parallel beam: its scatter and
    primary relative to its own open field converge to those of the parallel beam."""
    dtype = torch.float64
    grid = Grid((10, 16, 20), 2.0)
    config = replace(SMALL_CONFIG, dtype="float64")
    rho = wlab_phantom(grid, dtype=dtype)
    parallel = ParallelGeometry(30.0, num_det_rows=20, num_det_channels=20, delta_det_row=2.0, delta_det_channel=2.0)
    images = {}
    for name, geometry in (("parallel", parallel), (1000.0, ConeGeometry(1000.0, 1030.0, 20, 20, 2.0, 2.0)),
                           (5000.0, ConeGeometry(5000.0, 5030.0, 20, 20, 2.0, 2.0))):
        solver = _solver(geometry, device="cpu", dtype=dtype, config=config)
        result = solver.solve_view(rho, grid, 0.5)
        open_field = solver.open_field_signal(0.5)
        images[name] = (result.scatter["signal"] / open_field, result.primary["signal"] / open_field)
    scatter, primary = images["parallel"]
    errors = {name: (float((s - scatter).norm() / scatter.norm()), float((p - primary).norm() / primary.norm()))
              for name, (s, p) in images.items() if name != "parallel"}
    assert errors[5000.0][0] < 0.02 and errors[5000.0][1] < 0.06, errors
    assert errors[5000.0][0] < 0.5 * errors[1000.0][0] and errors[5000.0][1] < 0.5 * errors[1000.0][1], errors
