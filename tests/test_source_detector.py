"""Source emission profile, detector response and gain, and the anti-scatter grid built from its septa
(citrus/scatter_solver/source.py, detector.py, ScatterSolver.open_field_signal, emission_profile_from_open_field, and
the flat-field image of citrus_recon / mbir_recon / synthesize_cone_beam_data).

The end-to-end tests run CITRUS on the tiny noise-free WLAB problem of tests/test_citrus_recon.py: data made with a
strong bowtie-like emission profile and a gain map, reconstructed with the matching open field and with a flat field
of the same counts, and data made behind an anti-scatter grid."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from conftest import SPECTRUM_125KVP
from citrus.citrus import CitrusConfig, _flat_field, citrus_recon, estimate_regularization, mbir_recon
from citrus.scatter_solver import (ScatterSolver, SolverConfig, SourceModel, anti_scatter_grid,
                                   bowtie_emission_profile, csi_response, detector_response,
                                   emission_profile_from_open_field, focused_1d_grid, ideal_counting_response,
                                   no_grid, tabulated_detector_response)
from citrus.scatter_solver.cross_sections import MATERIAL_CATALOGUE, synthetic_library
from citrus.scatter_solver.detector import grid_angles, lead_mass_attenuation
from citrus.scatter_solver.geometry import ConeGeometry, Grid
from citrus.scatter_solver.ray_tracing import SiddonProjectors
from citrus.scatter_solver.scatter_image import primary_from_path_lengths
from citrus.scatter_solver.source import panel_interpolate
from citrus.scatter_solver.uncollided import uncollided_fluence

F64 = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the CITRUS reconstruction tests run on CUDA")
E_G = torch.tensor([110.0, 70.0, 45.0, 34.0, 25.0], dtype=F64)


@pytest.fixture(scope="module")
def photon_data():
    from citrus.scatter_solver.photon_data import resolve_backend

    try:
        return resolve_backend()
    except Exception as exc:  # no photon data
        pytest.skip(f"photon data unavailable: {exc}")


# ---- source model ----------------------------------------------------------------------------------------

def _panel_directions(geometry: ConeGeometry, angle: float, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    u_hat, v_hat, _ = (torch.tensor(a, dtype=F64) for a in geometry.panel_axes(angle))
    points = torch.tensor(geometry.panel_center(angle), dtype=F64) + u[..., None] * u_hat + v[..., None] * v_hat
    return points - torch.tensor(geometry.source_position(angle), dtype=F64)


def test_emission_profile_interpolation():
    """The angular weight is the profile at the pixel centres, bilinear between them, constant out to the panel edge
    and 0 off the panel; a profile of ones gives the default weight."""
    geometry = ConeGeometry(80.0, 120.0, 4, 6, 2.0, 1.5, det_row_offset=0.3, det_channel_offset=-0.2)
    g = torch.Generator().manual_seed(0)
    profile = 0.5 + torch.rand(4, 6, generator=g, dtype=F64)
    source = SourceModel(emission_profile=profile)
    for angle in (0.0, 0.7):
        u, v = geometry.pixel_uv(dtype=F64)
        centres = _panel_directions(geometry, angle, u[None, :].expand(4, 6), v[:, None].expand(4, 6))
        torch.testing.assert_close(source.angular_weight(centres, geometry, angle), profile, rtol=1e-12, atol=1e-12)
        # midway between four pixel centres: the mean of the four
        mid = _panel_directions(geometry, angle, 0.5 * (u[1] + u[2]), 0.5 * (v[0] + v[1]))
        assert float(source.angular_weight(mid, geometry, angle)) == pytest.approx(float(profile[:2, 1:3].mean()))
        # beyond the outermost centre but on the panel: the edge value; off the panel: 0
        half_u, _ = geometry.panel_half_extent()
        edge = _panel_directions(geometry, angle, torch.tensor(half_u - geometry.det_channel_offset - 0.01,
                                                               dtype=F64), v[2])
        assert float(source.angular_weight(edge, geometry, angle)) == pytest.approx(float(profile[2, -1]))
        off = _panel_directions(geometry, angle, torch.tensor(half_u + 1.0, dtype=F64), v[2])
        assert float(source.angular_weight(off, geometry, angle)) == 0.0
        directions = torch.randn(50, 3, generator=g, dtype=F64) * 5 + centres[1, 2]
        ones = SourceModel(emission_profile=torch.ones(4, 6, dtype=F64))
        torch.testing.assert_close(ones.angular_weight(directions, geometry, angle),
                                   SourceModel().angular_weight(directions, geometry, angle), rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="panel"):
        SourceModel(emission_profile=torch.ones(3, 6)).angular_weight(centres, geometry, 0.0)
    with pytest.raises(ValueError):
        SourceModel(emission_profile=-torch.ones(4, 6))
    with pytest.raises(ValueError, match="angles"):
        SourceModel(emission_profile=torch.ones(2, 4, 6))
    assert torch.equal(panel_interpolate(profile, u[None, :].expand(4, 6), v[:, None].expand(4, 6), geometry),
                       profile)


def test_per_view_emission_profile_and_bowtie():
    geometry = ConeGeometry(80.0, 120.0, 3, 9, 2.0, 2.0)
    bowtie = bowtie_emission_profile(geometry, 0.25)
    assert bowtie.shape == (3, 9) and float(bowtie[:, 4].min()) == 1.0
    np.testing.assert_allclose(bowtie[:, [0, -1]].numpy(), 0.25, rtol=1e-12)
    assert bool((bowtie[:, 1:5] > bowtie[:, :4]).all())
    profiles = torch.stack([bowtie, 2 * bowtie])
    source = SourceModel(emission_profile=profiles, emission_profile_angles=[0.0, 1.0])
    assert torch.equal(source.profile_at(0.1, dtype=F64), bowtie) and torch.equal(source.profile_at(0.9, dtype=F64),
                                                                                  2 * bowtie)
    assert SourceModel().profile_at(0.0) is None


def test_emission_profile_scales_primary_and_uncollided():
    """The primary of a pixel and the uncollided fluence of a voxel scale with the profile toward them, and the
    profile from the solver's own open field with a gain map is recovered exactly."""
    library = synthetic_library(num_groups=3, legendre_order=1)
    geometry = ConeGeometry(60.0, 90.0, 6, 8, 2.0, 2.0)
    detector, grid_response = csi_response(energies=library.group_energy), no_grid()
    profile = bowtie_emission_profile(geometry, 0.3) * torch.linspace(0.8, 1.2, 6, dtype=F64)[:, None]
    source = SourceModel(emission_profile=profile)
    path = torch.full((1, 6, 8), 0.7, dtype=F64)
    plain = primary_from_path_lengths(library, path, [0], geometry, 0.4, SourceModel(), detector, grid_response)
    weighted = primary_from_path_lengths(library, path, [0], geometry, 0.4, source, detector, grid_response)
    torch.testing.assert_close(weighted["signal"], plain["signal"] * profile, rtol=1e-12, atol=0)
    grid = Grid((4, 4, 4), 2.0)
    rho = torch.zeros(2, *grid.shape, dtype=F64)
    rho[0] = 1.0
    projectors = SiddonProjectors()
    phi = uncollided_fluence(library, rho, grid, geometry, 0.4, SourceModel(), projectors)
    phi_b = uncollided_fluence(library, rho, grid, geometry, 0.4, source, projectors)
    r = grid.center_coordinates(dtype=F64)
    b = source.angular_weight(r - torch.tensor(geometry.source_position(0.4), dtype=F64), geometry, 0.4)
    torch.testing.assert_close(phi_b, phi * b[None], rtol=1e-12, atol=0)
    assert float(b.min()) < 0.5 * float(b.max())  # the voxels see different parts of the bowtie

    solver = ScatterSolver(library, SolverConfig(num_groups=3, legendre_order=1), geometry, detector=detector,
                           grid_response=grid_response, dtype=F64)
    gain = 1.0 + 0.1 * torch.rand(6, 8, generator=torch.Generator().manual_seed(2), dtype=F64)
    open_field = 3.0e4 * gain * solver.open_field_signal(0.0, source)
    recovered = emission_profile_from_open_field(open_field, solver, detector_gain=gain)
    torch.testing.assert_close(recovered, profile / profile.mean(), rtol=1e-12, atol=0)
    without_gain = emission_profile_from_open_field(open_field, solver)  # the gain counts as emission
    torch.testing.assert_close(without_gain, profile * gain / (profile * gain).mean(), rtol=1e-12, atol=0)
    per_view = emission_profile_from_open_field(torch.stack([open_field, 2 * open_field]), solver, [0.0, 1.0], gain)
    torch.testing.assert_close(per_view, torch.stack([recovered, recovered]), rtol=1e-12, atol=0)
    with pytest.raises(ValueError, match="angles"):
        emission_profile_from_open_field(torch.stack([open_field, open_field]), solver)
    with pytest.raises(ValueError, match="open field"):
        emission_profile_from_open_field(open_field[:, :5], solver)


def test_flat_field_forms():
    Y = torch.ones(3, 4, 5)
    assert _flat_field(1.0e5, Y) == 1.0e5 and _flat_field(np.asarray(2.0), Y) == 2.0
    assert _flat_field(np.full((4, 5), 7.0), Y).shape == (4, 5)
    assert _flat_field(torch.full((3, 4, 5), 7.0), Y).shape == (3, 4, 5)
    for bad in (np.ones((2, 5)), np.zeros((4, 5)), -1.0, np.ones((2, 4, 5))):
        with pytest.raises(ValueError, match="flat field"):
            _flat_field(bad, Y)


# ---- detector response -------------------------------------------------------------------------------------

def test_detector_response_from_photon_data(photon_data):
    """CsI from the photon data agrees with the coarse built-in table; the deposition and counting variants are
    ordered as they must be; a thick layer absorbs everything."""
    theta = torch.tensor([0.0, 0.5, 1.0], dtype=F64)
    reference = csi_response().at(E_G, theta)
    full = detector_response("cesium_iodide", 0.06, backend=photon_data).at(E_G, theta)
    np.testing.assert_allclose(full.numpy(), reference.numpy(), rtol=0.02)
    assert bool((full[:, 1:] > full[:, :-1]).all())  # longer path at oblique incidence
    local = detector_response(deposition="local", backend=photon_data).at(E_G, theta)
    assert bool((local < full).all()) and bool((local > 0.8 * full).all())
    counting = detector_response(mode="photon_counting", backend=photon_data).at(E_G, theta)
    np.testing.assert_allclose(counting.numpy(), (full / E_G[:, None]).numpy(), rtol=1e-12)
    thick = detector_response("gadolinium_oxysulfide", 5.0, backend=photon_data).at(E_G, theta)
    np.testing.assert_allclose(thick.numpy(), E_G[:, None].expand(5, 3).numpy(), rtol=1e-9)
    # MV energies are inside the default table (the built-in CsI table stops at 150 keV)
    mv = torch.tensor([1000.0, 5000.0], dtype=F64)
    high = detector_response("cesium_iodide", 0.06, deposition="local", backend=photon_data).at(mv, theta[:1])
    assert bool((high[:, 0] < 0.02 * mv).all()) and bool((high > 0).all())
    with pytest.raises(ValueError):
        detector_response(mode="counting")


def test_tabulated_detector_response():
    energies = [20.0, 60.0, 120.0]
    flat = tabulated_detector_response(energies, [0.5, 0.8, 0.6])
    theta = torch.tensor([0.0, 0.4, 1.2], dtype=F64)
    np.testing.assert_allclose(flat.at(torch.tensor([60.0, 90.0], dtype=F64), theta).numpy(),
                               [[0.8] * 3, [0.7] * 3], rtol=1e-12)
    table = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    angular = tabulated_detector_response(energies, table, [0.0, 1.0])
    assert float(angular.at(torch.tensor([60.0], dtype=F64), torch.tensor(0.5, dtype=F64))) == pytest.approx(3.5)
    with pytest.raises(ValueError):
        tabulated_detector_response(energies, table)
    with pytest.raises(ValueError):
        tabulated_detector_response([60.0, 20.0, 120.0], [1.0, 1.0, 1.0])


# ---- anti-scatter grid -----------------------------------------------------------------------------------------

def test_anti_scatter_grid_from_septa(photon_data):
    """A lead grid from the photon data agrees with the built-in lead model; the default pitch is the pixel pitch
    across the septa; septa along the rows swap u and v in the grid angles."""
    t1 = torch.tensor([0.0, 0.01, 0.03, 0.2], dtype=F64)
    reference = focused_1d_grid().at(E_G, t1, 0 * t1)
    grid = anti_scatter_grid(None, 0.0036, 0.15, pitch_cm=1 / 60, focal_distance_cm=150.0, backend=photon_data)
    np.testing.assert_allclose(grid.at(E_G, t1, 0 * t1).numpy(), reference.numpy(), rtol=0.01, atol=1e-3)
    geometry = ConeGeometry(100.0, 150.0, 5, 7, 0.3, 0.4)
    across_rows = anti_scatter_grid(geometry, 0.02, 2.0, backend=photon_data)
    across_channels = anti_scatter_grid(geometry, 0.02, 2.0, septa_along="rows", septum_material="tungsten",
                                        backend=photon_data)
    assert across_rows.septa_along == "channels" and across_rows.focal_distance_cm == 150.0
    assert across_channels.septa_along == "rows"
    zero = torch.zeros((), dtype=F64)
    for g, p, material in ((across_rows, 0.3, "lead"), (across_channels, 0.4, "tungsten")):
        mu = torch.as_tensor(photon_mass_attenuation(photon_data, material, E_G))
        rho = MATERIAL_CATALOGUE[material][0]
        expected = (p - 0.02) / p + 0.02 / p * torch.exp(-mu * rho * 2.0)
        np.testing.assert_allclose(g.at(E_G, zero, zero).numpy(), expected.numpy(), rtol=2e-3, atol=1e-6)
        cutoff = math.atan((p - 0.02) / 2.0)
        assert bool((g.at(E_G, torch.tensor(1.05 * cutoff, dtype=F64), zero)[E_G < 60] < 1e-3).all())
    for angle in (0.0, 0.8):
        pixels = geometry.pixel_centers(angle, dtype=F64)
        primary = geometry.primary_directions(angle, dtype=F64)
        u, v = geometry.pixel_uv(dtype=F64)
        th1, th2 = grid_angles(primary, pixels, geometry, angle, septa_along="rows")
        assert float(th1.abs().max()) < 1e-12  # focused at the focal spot
        np.testing.assert_allclose(th2.numpy(), torch.atan(-v / 150.0)[:, None].expand(5, 7).numpy(), atol=1e-12)
        # a tilt along u is theta1 for septa along the rows, theta2 for septa along the channels
        u_hat, _, n_hat = (torch.tensor(a, dtype=F64) for a in geometry.panel_axes(angle))
        tilted = -(n_hat * math.cos(0.1) + u_hat * math.sin(0.1))
        rows1, _ = grid_angles(tilted[None], pixels, geometry, angle, septa_along="rows")
        _, channels2 = grid_angles(tilted[None], pixels, geometry, angle)
        np.testing.assert_allclose(rows1[0, 2, 3].item(), channels2[0, 2, 3].item(), atol=1e-12)
    with pytest.raises(ValueError, match="pitch"):
        anti_scatter_grid(geometry, 0.5, 2.0, backend=photon_data)
    with pytest.raises(ValueError, match="geometry"):
        anti_scatter_grid(None, 0.01, 1.0)


def photon_mass_attenuation(backend, material: str, energies: torch.Tensor) -> np.ndarray:
    from citrus.scatter_solver.detector import material_cross_sections

    return material_cross_sections(material, energies, backend)["total"]


def test_lead_cross_sections_match_the_built_in_table(photon_data):
    e = np.array([40.0, 60.0, 100.0])
    np.testing.assert_allclose(photon_mass_attenuation(photon_data, "lead", torch.as_tensor(e)),
                               lead_mass_attenuation(e), rtol=0.03)


# ---- end to end -------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_library():
    from citrus.scatter_solver.scatter_solver import build_solver_library

    try:
        return build_solver_library(SolverConfig(), SPECTRUM_125KVP)
    except Exception as exc:  # no cached library and no photon data
        pytest.skip(f"real library unavailable: {exc}")


def _tiny_geometry() -> ConeGeometry:
    from citrus.utilities import wlab_geometry

    return replace(wlab_geometry(), num_det_rows=46, num_det_channels=38, delta_det_row=1.6, delta_det_channel=1.6)


def _relative_rmse(X, data) -> float:
    truth = torch.as_tensor(data["X_true"], device=X.device)
    return float((X - truth).norm() / truth.norm())


@cuda
def test_bowtie_open_field_end_to_end(real_library):
    """Data of a strong bowtie-like emission profile (1 to 0.1 across the channels) and a gain map: CITRUS given the
    open field and the gain map beats CITRUS given a flat field of the same counts, and CITRUS whose solver sees a
    uniform emission (the bowtie passed as gain: the normalization is exact, only the scatter model is wrong)."""
    pytest.importorskip("mbirtorch")
    from citrus.recon.mbir_bridge import make_cone_model
    from citrus.utilities import synthesize_wlab_dataset, wlab_grid, wlab_solver

    geometry, config = _tiny_geometry(), SolverConfig(transport_cell_cm=2.0)
    bowtie = bowtie_emission_profile(geometry, 0.1)
    gain = 1.0 + 0.05 * torch.sin(torch.arange(46.0, dtype=F64))[:, None] * torch.cos(torch.arange(38.0, dtype=F64))
    data = synthesize_wlab_dataset(16, 2.0, noise=False, seed=0, library=real_library, geometry=geometry,
                                   config=config, source=SourceModel(emission_profile=bowtie), detector_gain=gain)
    assert data["open_field"].shape == (46, 38) and data["meta"]["open_field"] == "image"
    np.testing.assert_allclose(data["P"], data["open_field"] * np.exp(-data["line_integrals"]), rtol=1e-5)
    grid = wlab_grid(2.0)
    solver = wlab_solver(2.0, config=config, library=real_library, geometry=geometry, device=DEVICE)
    profile = emission_profile_from_open_field(data["open_field"], solver, detector_gain=gain)
    np.testing.assert_allclose(profile.numpy(), (bowtie / bowtie.mean()).numpy(), rtol=1e-5)
    model = make_cone_model(grid, geometry, data["angles"], device=DEVICE)
    open_field = torch.as_tensor(data["open_field"], device=DEVICE)
    regularization = estimate_regularization(model, -torch.log(torch.as_tensor(data["P"], device=DEVICE) / open_field),
                                             torch.as_tensor(data["Y"], device=DEVICE) / open_field)
    loop = CitrusConfig(mbir_iterations_per_outer_iteration=1, num_outer_iterations=5)

    def run(flat_field, detector_gain=None):
        X, info = citrus_recon(data["Y"], flat_field, data["angles"], grid, geometry, solver, loop,
                               regularization=regularization, model=model, detector_gain=detector_gain)
        assert solver.source.emission_profile is None  # the solver's source is restored
        return _relative_rmse(X, data), info

    error_open, info = run(data["open_field"], gain)
    np.testing.assert_allclose(info["emission_profile"].numpy(), profile.numpy(), rtol=1e-6)
    error_flat, flat_info = run(float(data["open_field"].mean()))
    error_uniform, _ = run(data["open_field"], gain * bowtie)
    assert flat_info["emission_profile"] is None
    assert error_open < 0.5 * error_flat
    assert error_open < 0.85 * error_uniform
    # mbir_recon: a per-view open field equals the same open field for every view
    views = np.broadcast_to(data["open_field"], data["Y"].shape)
    X2, _ = mbir_recon(data["Y"], data["open_field"], data["angles"], grid, geometry, loop,
                       regularization=regularization, model=model, mbir_iterations_schedule=[2])
    X3, _ = mbir_recon(data["Y"], views, data["angles"], grid, geometry, loop, regularization=regularization,
                       model=model, mbir_iterations_schedule=[2])
    torch.testing.assert_close(X2, X3, rtol=1e-5, atol=1e-6)


@cuda
def test_anti_scatter_grid_end_to_end(real_library, photon_data):
    """A focused grid (one lead septum per pixel, ratio 8) lowers the scatter-to-primary ratio of the WLAB data
    severalfold; CITRUS with the grid in its solver still corrects the remaining scatter."""
    pytest.importorskip("mbirtorch")
    from citrus.recon.mbir_bridge import make_cone_model
    from citrus.utilities import synthesize_wlab_dataset, wlab_grid, wlab_solver

    geometry, config = _tiny_geometry(), SolverConfig(transport_cell_cm=2.0)
    septa = anti_scatter_grid(geometry, 0.1, 12.0, septa_along="rows", backend=photon_data)
    without = synthesize_wlab_dataset(16, 2.0, noise=False, seed=0, library=real_library, geometry=geometry,
                                      config=config)
    assert without["open_field"].ndim == 0 and float(without["open_field"]) == 1.0e5
    data = synthesize_wlab_dataset(16, 2.0, noise=False, seed=0, library=real_library, geometry=geometry,
                                   config=config, grid_response=septa)
    assert data["meta"]["grid_response"] == septa.name
    assert data["SPR"].mean() < 0.35 * without["SPR"].mean()
    grid = wlab_grid(2.0)
    solver = wlab_solver(2.0, config=config, library=real_library, geometry=geometry, device=DEVICE,
                         grid_response=septa)
    model = make_cone_model(grid, geometry, data["angles"], device=DEVICE)
    regularization = estimate_regularization(model, -torch.log(torch.as_tensor(data["P"], device=DEVICE) / 1.0e5),
                                             torch.as_tensor(data["Y"], device=DEVICE) / 1.0e5)
    loop = CitrusConfig(mbir_iterations_per_outer_iteration=1, num_outer_iterations=5)
    X, info = citrus_recon(data["Y"], data["open_field"], data["angles"], grid, geometry, solver, loop,
                           regularization=regularization, model=model)
    X0, _ = mbir_recon(data["Y"], data["open_field"], data["angles"], grid, geometry, loop,
                       regularization=regularization, model=model, init_recon=info["init_recon"])
    assert _relative_rmse(X, data) < 0.7 * _relative_rmse(X0, data)
    spr_error = np.abs(info["spr"].cpu().numpy() - data["SPR"]).mean() / data["SPR"].mean()
    assert spr_error < 0.25


def test_ideal_responses_keep_the_planar_fluence():
    """With the default ideal counting detector and no grid the detector signal is the planar fluence exactly, so
    the synthetic data's switch to the signal leaves them unchanged."""
    library = synthetic_library(num_groups=3, legendre_order=1)
    geometry = ConeGeometry(60.0, 90.0, 6, 8, 2.0, 2.0)
    path = torch.rand(2, 6, 8, generator=torch.Generator().manual_seed(1), dtype=torch.float32)
    image = primary_from_path_lengths(library, path, None, geometry, 0.3, SourceModel(), ideal_counting_response(),
                                      no_grid())
    assert torch.equal(image["signal"], image["planar_fluence"].sum(0))


def test_crossed_grid_is_the_product_of_its_septa(photon_data):
    """Septa along both directions transmit the product of the grids along the channels and along the rows, each at
    its own grid angles and pitch; on the primary ray of a focused grid that is the product of the open fractions."""
    geometry = ConeGeometry(100.0, 150.0, 5, 7, 0.3, 0.4)
    both = anti_scatter_grid(geometry, 0.02, 2.0, septa_along="both", backend=photon_data)
    channels = anti_scatter_grid(geometry, 0.02, 2.0, backend=photon_data)
    rows = anti_scatter_grid(geometry, 0.02, 2.0, septa_along="rows", backend=photon_data)
    assert both.crossed is not None and both.crossed.septa_along == "rows" and "crossed" in both.name
    assert both.depends_on_angles and not no_grid().depends_on_angles
    angle = 0.8
    pixels = geometry.pixel_centers(angle, dtype=F64)
    directions = torch.randn(9, 3, generator=torch.Generator().manual_seed(3), dtype=F64)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    expected = (channels.transmission_at(E_G, directions, pixels, geometry, angle)
                * rows.transmission_at(E_G, directions, pixels, geometry, angle))
    torch.testing.assert_close(both.transmission_at(E_G, directions, pixels, geometry, angle), expected)
    primary = geometry.primary_directions(angle, dtype=F64)
    on_axis = both.transmission_at(E_G, primary, pixels, geometry, angle)  # [G, Rd, Cd]
    zero = torch.zeros((), dtype=F64)
    open_product = channels.at(E_G, zero, zero) * rows.at(E_G, zero, zero)
    torch.testing.assert_close(on_axis, open_product[:, None, None].expand_as(on_axis), rtol=1e-6, atol=1e-9)
    with pytest.raises(ValueError, match="septa_along"):
        anti_scatter_grid(geometry, 0.02, 2.0, septa_along="diagonal", backend=photon_data)
    with pytest.raises(ValueError, match="other direction"):
        replace(channels, crossed=channels)


def test_crossed_grid_rejects_more_scatter_on_a_flat_panel(photon_data):
    """On a flat panel a crossed grid lowers the scatter-to-primary ratio below that of either set of its septa."""
    from citrus.utilities import wlab_phantom

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = Grid((10, 16, 20), 2.0)
    config = SolverConfig(num_groups=3, legendre_order=2, num_polar=4, num_azimuthal=8, cone_num_polar=6,
                          cone_num_azimuthal=12, transport_cell_cm=4.0, max_source_iterations=6)
    library = synthetic_library(3, 3, num_materials=7)
    geometry = ConeGeometry(100.0, 150.0, num_det_rows=24, num_det_channels=32, delta_det_row=1.5,
                            delta_det_channel=1.5)
    rho = wlab_phantom(grid, device=device, dtype=torch.float32)
    ratios = {}
    for septa_along in ("channels", "rows", "both"):
        septa = anti_scatter_grid(geometry, 0.05, 6.0, septa_along=septa_along, backend=photon_data)
        solver = ScatterSolver(library, config, geometry, SiddonProjectors(), csi_response(), septa, device=device,
                               dtype=torch.float32)
        result = solver.solve_view(rho, grid, 0.5)
        scatter, primary = result.scatter["signal"], result.primary["signal"]
        assert bool(torch.isfinite(scatter).all()) and float(scatter.min()) >= 0 and float(primary.min()) > 0
        ratios[septa_along] = float(scatter.sum() / primary.sum())
    assert ratios["both"] < 0.8 * min(ratios["channels"], ratios["rows"]), ratios
