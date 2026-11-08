"""The source's field past the panel edges, its off-focal emission and its beam filter (citrus/scatter_solver/source.py,
the filter term of cross_sections.group_transmission and its Triton kernel, filter_thickness_from_transmission,
water_linearization), the scatter scene beyond the reconstructed slab (citrus.citrus.extend_scene_slices) and the loop's
options extend_scatter_scene and beam_hardening_correction.

The end-to-end tests run CITRUS on small noise-free WLAB problems: data of an aluminium bowtie with a polychromatic
primary, reconstructed with the filtered source and the water linearization and with a gray emission profile; and data
of a prism longer than the reconstructed slab, irradiated past the panel's rows, reconstructed with and without the
extended scatter scene."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from conftest import SPECTRUM_125KVP
from citrus.citrus import CitrusConfig, citrus_recon, extend_scene_slices, estimate_regularization, mbir_recon
from citrus.scatter_solver import (ParallelGeometry, ScatterSolver, SolverConfig, SourceModel,
                                   emission_profile_from_open_field, filter_thickness_from_transmission,
                                   focused_1d_grid, ideal_counting_response, monoenergetic_spectrum, no_grid,
                                   water_linearization)
from citrus.scatter_solver.cross_sections import Material, _group_transmission_torch, group_transmission
from citrus.scatter_solver.cross_sections import synthetic_library
from citrus.scatter_solver.detector import material_cross_sections
from citrus.scatter_solver.geometry import ConeGeometry, Grid
from citrus.scatter_solver.kernels import transmission_kernels
from citrus.scatter_solver.ray_tracing import SiddonProjectors
from citrus.scatter_solver.scatter_image import primary_from_path_lengths
from citrus.scatter_solver.uncollided import uncollided_fluence

F64 = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the CITRUS reconstruction tests run on CUDA")
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module")
def photon_data():
    from citrus.scatter_solver.photon_data import resolve_backend

    try:
        return resolve_backend()
    except Exception as exc:  # no photon data
        pytest.skip(f"photon data unavailable: {exc}")


@pytest.fixture(scope="module")
def real_library():
    from citrus.scatter_solver.scatter_solver import build_solver_library

    try:
        return build_solver_library(SolverConfig(), SPECTRUM_125KVP)
    except Exception as exc:  # no cached library and no photon data
        pytest.skip(f"real library unavailable: {exc}")


def _panel_directions(geometry: ConeGeometry, angle: float, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    u_hat, v_hat, _ = (torch.tensor(a, dtype=F64) for a in geometry.panel_axes(angle))
    points = torch.tensor(geometry.panel_center(angle), dtype=F64) + u[..., None] * u_hat + v[..., None] * v_hat
    return points - torch.tensor(geometry.source_position(angle), dtype=F64)


# ---- field margins and off-focal emission ----------------------------------------------------------------------

def test_field_margins_and_off_focal_weights():
    """The field reaches the margins past the panel edges with the edge emission; past it the off-focal table gives
    the fraction of the distance to the field rectangle (Euclidean past a corner), 0 beyond the table."""
    geometry = ConeGeometry(80.0, 120.0, 4, 6, 2.0, 1.5, det_row_offset=0.3, det_channel_offset=-0.2)
    half_u, half_v = geometry.panel_half_extent()
    c_u, c_v = -geometry.det_channel_offset, -geometry.det_row_offset  # the panel centre in (u, v)
    profile = 0.5 + torch.rand(4, 6, generator=torch.Generator().manual_seed(0), dtype=F64)

    def weight(source, du, dv, angle=0.4):  # (du, dv) past the right and top panel edges
        directions = _panel_directions(geometry, angle, torch.tensor(c_u + half_u + du, dtype=F64),
                                       torch.tensor(c_v + half_v + dv, dtype=F64))
        return float(source.angular_weight(directions, geometry, angle))

    plain = SourceModel(emission_profile=profile)
    assert weight(plain, -0.01, -0.01) == pytest.approx(float(profile[-1, -1])) and weight(plain, 0.5, -0.5) == 0.0
    wide = SourceModel(emission_profile=profile, field_margin_channels_cm=3.0, field_margin_rows_cm=1.0)
    assert weight(wide, 2.9, -0.5) == pytest.approx(float(profile[-1, -1]))  # the edge emission held
    assert weight(wide, 3.1, -0.5) == 0.0 and weight(wide, 1.0, 0.9) > 0 and weight(wide, 1.0, 1.1) == 0.0
    assert wide.field_half_extent(geometry) == (half_u + 3.0, half_v + 1.0)
    table = SourceModel(field_margin_channels_cm=3.0, off_focal_distance_cm=[0.5, 1.5, 2.5],
                        off_focal_fraction=[0.1, 0.05, 0.01])
    assert weight(table, 2.0, -0.1) == 1.0
    assert weight(table, 3.2, -0.1) == pytest.approx(0.1)  # below the first distance: its fraction
    assert weight(table, 4.0, -0.1) == pytest.approx(0.075)  # 1.0 cm past the field edge
    assert weight(table, 1.0, 2.0) == pytest.approx(0.03)  # 2.0 cm past the row edge
    assert weight(table, 3.0 + 1.2, 1.6) == pytest.approx(0.03)  # past the corner: sqrt(1.2^2 + 1.6^2) = 2.0 cm
    assert weight(table, 3.0 + 2.6, -0.1) == 0.0  # beyond the table
    assert table.field_half_extent(geometry) == (half_u + 3.0 + 2.5, half_v + 2.5)
    # a parallel beam: the same rules at the panel point of each ray
    parallel = ParallelGeometry(50.0, 4, 6, 2.0, 1.5)
    hu, hv = parallel.panel_half_extent()
    points = torch.tensor([[hu + 1.0, 0.0, 0.5], [hu + 4.0, 0.0, 0.5], [0.0, 0.0, hv + 1.0]], dtype=F64)
    beam = table.beam_weight(points, parallel, 0.0)
    torch.testing.assert_close(beam, torch.tensor([1.0, 0.075, 0.075], dtype=F64))
    torch.testing.assert_close(SourceModel().beam_weight(points, parallel, 0.0), torch.zeros(3, dtype=F64))


def test_default_source_weights_are_unchanged():
    """With every new setting at its default, the angular weight is the panel indicator times the profile."""
    geometry = ConeGeometry(80.0, 120.0, 4, 6, 2.0, 1.5, det_row_offset=0.3, det_channel_offset=-0.2)
    directions = _panel_directions(geometry, 0.0, torch.zeros(1, dtype=F64), torch.zeros(1, dtype=F64))
    directions = directions + torch.randn(500, 3, generator=torch.Generator().manual_seed(1), dtype=F64) * 20.0
    forward_hits = SourceModel(field_margin_channels_cm=0.0, field_margin_rows_cm=0.0).angular_weight(
        directions, geometry, 0.0)
    assert torch.equal(forward_hits, SourceModel().angular_weight(directions, geometry, 0.0))
    assert 0 < float(forward_hits.sum()) < 500


def test_source_settings_are_validated():
    with pytest.raises(ValueError, match="field_margin_rows_cm"):
        SourceModel(field_margin_rows_cm=-1.0)
    with pytest.raises(ValueError, match="together"):
        SourceModel(off_focal_distance_cm=[1.0])
    with pytest.raises(ValueError, match="ascending"):
        SourceModel(off_focal_distance_cm=[1.0, 0.5], off_focal_fraction=[0.1, 0.1])
    with pytest.raises(ValueError, match="non-negative"):
        SourceModel(off_focal_distance_cm=[1.0], off_focal_fraction=[-0.1])
    with pytest.raises(ValueError, match="together"):
        SourceModel(filter_material="aluminium")
    with pytest.raises(ValueError, match="unknown"):
        SourceModel(filter_material="unobtainium", filter_thickness_cm=[1.0])
    with pytest.raises(ValueError, match="non-negative"):
        SourceModel(filter_material="aluminium", filter_thickness_cm=[-1.0, 1.0])
    with pytest.raises(ValueError, match="panel"):
        SourceModel(filter_material="aluminium", filter_thickness_cm=torch.ones(5)).thickness_at(
            ConeGeometry(80.0, 120.0, 4, 6, 2.0, 1.5))
    assert SourceModel(filter_material="aluminium", filter_thickness_cm=torch.ones(3, 6)).has_filter


# ---- the beam filter -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("device", devices)
def test_filter_term_is_the_direct_fine_energy_sum(device):
    """The filter term of group_transmission (PyTorch path, and the kernel on CUDA) equals a direct sum over the fine
    energies of exp(-sum_m mu_m P_m - mu_f P_f); a zero filter path is no filter, bit for bit."""
    library = synthetic_library(4, num_materials=7, num_fine_per_group=5, fine_spread=0.4)
    dtype = torch.float32 if device == "cuda" else F64
    generator = torch.Generator().manual_seed(4)
    materials = [1, 3, 6]
    path = (torch.rand(3, 9, 11, generator=generator, dtype=F64) * 8.0).to(device, dtype)
    filter_path = (torch.rand(9, 11, generator=generator, dtype=F64) * 3.0).to(device, dtype)
    attenuation = (torch.rand(library.fine_energy.numel(), generator=generator, dtype=F64) + 0.2).to(device, dtype)
    out = group_transmission(library, path, materials=materials, filter_path=filter_path,
                             filter_attenuation=attenuation)
    mu = library.fine_mu_t[materials]  # [3, E] float64
    tau = torch.einsum("me,mrc->erc", mu, path.double().cpu()) + attenuation.double().cpu()[:, None, None] * \
        filter_path.double().cpu()[None]
    terms = library.fine_weight[:, None, None] * torch.exp(-tau)
    direct = torch.stack([terms[library.fine_group == g].sum(0) for g in range(library.num_groups)])
    torch.testing.assert_close(out.double().cpu(), direct, rtol=2e-6 if device == "cuda" else 1e-13, atol=0)
    if device == "cuda":
        assert transmission_kernels.use_kernel(library, path)  # the fused kernel ran
    plain = group_transmission(library, path, materials=materials)
    zero = group_transmission(library, path, materials=materials, filter_path=torch.zeros_like(filter_path),
                              filter_attenuation=attenuation)
    assert torch.equal(plain, zero)
    torch.testing.assert_close(_group_transmission_torch(library, path, 7, materials, filter_path, attenuation), out,
                               rtol=2e-6 if device == "cuda" else 1e-13, atol=0)
    with pytest.raises(ValueError, match="together"):
        group_transmission(library, path, materials=materials, filter_path=filter_path)


@pytest.mark.parametrize("device", devices)
def test_zero_thickness_filter_is_no_filter(device, photon_data):
    """A source with a filter of zero thickness gives the uncollided fluence and the primary of the source without one,
    bit for bit; a filter of thickness t multiplies the monoenergetic primary by exp(-mu rho t)."""
    library = synthetic_library(num_groups=3, legendre_order=1).to(torch.device(device),
                                                                    torch.float32 if device == "cuda" else F64)
    dtype = library.mu_t.dtype
    geometry = ConeGeometry(60.0, 90.0, 6, 8, 2.0, 2.0)
    detector, grid_response = ideal_counting_response(), no_grid()
    zero = SourceModel(filter_material="aluminium", filter_thickness_cm=torch.zeros(8, dtype=F64))
    grid = Grid((4, 4, 4), 2.0)
    rho = torch.zeros(2, *grid.shape, dtype=dtype, device=device)
    rho[0] = 1.0
    rho[1, 1:3, 1:3, 1:3] = 0.8
    projectors = SiddonProjectors()
    for angle in (0.0, 0.9):
        assert torch.equal(uncollided_fluence(library, rho, grid, geometry, angle, zero, projectors),
                           uncollided_fluence(library, rho, grid, geometry, angle, SourceModel(), projectors))
        path = torch.full((2, 6, 8), 0.7, dtype=dtype, device=device)
        for key in ("signal", "fluence"):
            assert torch.equal(primary_from_path_lengths(library, path, None, geometry, angle, zero, detector,
                                                         grid_response)[key],
                               primary_from_path_lengths(library, path, None, geometry, angle, SourceModel(), detector,
                                                         grid_response)[key])


def test_filter_path_of_the_rays(photon_data):
    """The filter path of a ray is the density times the thickness interpolated at its panel point, held beyond the
    outermost pixel centres; a [channels] thickness is the same for every row."""
    geometry = ConeGeometry(80.0, 120.0, 4, 6, 2.0, 1.5)
    thickness = torch.tensor([3.0, 2.0, 1.0, 1.0, 2.0, 4.0], dtype=F64)
    source = SourceModel(filter_material="aluminium", filter_thickness_cm=thickness)
    density = 2.6989
    assert source.filter_density == pytest.approx(density)
    u, v = geometry.pixel_uv(dtype=F64)
    centres = _panel_directions(geometry, 0.3, u[None, :].expand(4, 6), v[:, None].expand(4, 6))
    torch.testing.assert_close(source.filter_path(centres, geometry, 0.3), density * thickness.expand(4, 6))
    beyond = _panel_directions(geometry, 0.3, torch.tensor(float(u[-1]) + 5.0, dtype=F64), v[1])
    assert float(source.filter_path(beyond, geometry, 0.3)) == pytest.approx(density * 4.0)
    assert SourceModel().filter_path(centres, geometry, 0.3) is None
    # the filter attenuation is the photon data's at the fine energies, also for a user-defined material
    library = synthetic_library(3)
    expected = material_cross_sections("aluminium", library.fine_energy.numpy())["total"]
    np.testing.assert_allclose(source.filter_attenuation(library, dtype=F64).numpy(), expected, rtol=1e-12)
    copper = SourceModel(filter_material=Material("copper_filter", 8.96, "Cu"), filter_thickness_cm=thickness)
    assert copper.filter_density == 8.96
    assert bool((copper.filter_attenuation(library, dtype=F64) > source.filter_attenuation(library, dtype=F64)).all())


def _solver(library, geometry, source=None, device=DEVICE, config=None, dtype=None):
    return ScatterSolver(library, SolverConfig() if config is None else config, geometry,
                         detector=ideal_counting_response(), grid_response=no_grid(), source=source, device=device,
                         dtype=dtype)


def test_thickness_inversion_round_trip(real_library):
    """The thickness of a bowtie is recovered from its transmission per pixel, from the transmission of the central
    rows, and from a profile normalized to its maximum with the minimum thickness."""
    geometry = ConeGeometry(100.0, 150.0, 6, 10, 2.0, 4.0)
    u, _ = geometry.pixel_uv(dtype=F64)
    thickness = 0.3 + 2.5 * (u / u.abs().max()) ** 2
    solver = _solver(real_library, geometry)
    filtered = _solver(real_library, geometry, SourceModel(filter_material="aluminium", filter_thickness_cm=thickness))
    transmission = (filtered.open_field_signal(0.0, dtype=F64) / solver.open_field_signal(0.0, dtype=F64)).cpu()
    assert float(transmission.max()) < 0.9  # the minimum thickness attenuates
    full = filter_thickness_from_transmission(transmission, solver, "aluminium")
    torch.testing.assert_close(full, thickness.expand(6, 10), rtol=0, atol=1e-3)
    central = filter_thickness_from_transmission(transmission[3], solver, "aluminium")
    torch.testing.assert_close(central, thickness, rtol=0, atol=1e-3)
    profile = transmission[3] / transmission[3].max()
    normalized = filter_thickness_from_transmission(profile, solver, "aluminium",
                                                    minimum_thickness_cm=float(thickness.min()))
    torch.testing.assert_close(normalized, thickness, rtol=0, atol=1e-3)
    with pytest.raises(ValueError, match="positive"):
        filter_thickness_from_transmission(-transmission, solver, "aluminium")


def test_emission_profile_of_a_filtered_source(real_library):
    """The open field of a filtered source divided by the model's own open field with the filter is flat; without
    the filter in the model the profile is the filter's gray transmission."""
    geometry = ConeGeometry(100.0, 150.0, 6, 10, 2.0, 4.0)
    u, _ = geometry.pixel_uv(dtype=F64)
    filtered = SourceModel(filter_material="aluminium", filter_thickness_cm=0.3 + 2.5 * (u / u.abs().max()) ** 2)
    solver = _solver(real_library, geometry, filtered)
    heel = torch.linspace(0.9, 1.1, 6, dtype=F64)[:, None].expand(6, 10)
    open_field = 5.0e4 * heel * solver.open_field_signal(0.0, dtype=F64).cpu()
    torch.testing.assert_close(emission_profile_from_open_field(open_field, solver), heel / heel.mean(), rtol=1e-5,
                               atol=0)
    gray = emission_profile_from_open_field(open_field, _solver(real_library, geometry))
    assert float(gray[:, 0].mean()) < 0.6 * float(gray[:, 5].mean())


def test_water_linearization(real_library, photon_data):
    """For a monoenergetic source the linearization is the identity; for the 125 kVp spectrum it maps the solver's
    polychromatic water line integrals of any pixel to mu_water(E_eff) L, and more so under a bowtie's thick wings."""
    from citrus.scatter_solver.scatter_solver import build_solver_library
    from citrus.utilities import effective_energy, mass_attenuation_at

    geometry = ConeGeometry(100.0, 150.0, 6, 10, 2.0, 4.0)
    mono = build_solver_library(SolverConfig(num_groups=3), monoenergetic_spectrum(70.0))
    linearization = water_linearization(_solver(mono, geometry, device="cpu", config=SolverConfig(num_groups=3),
                                                dtype=F64))
    lines = torch.rand(3, 6, 10, generator=torch.Generator().manual_seed(0), dtype=F64) * 6.0
    torch.testing.assert_close(linearization(lines), lines, rtol=1e-9, atol=1e-9)
    u, _ = geometry.pixel_uv(dtype=F64)
    filtered = SourceModel(filter_material="aluminium", filter_thickness_cm=0.3 + 2.5 * (u / u.abs().max()) ** 2)
    solver = _solver(real_library, geometry, filtered)
    linearization = water_linearization(solver)
    water = real_library.materials.index("water")
    mu = float(mass_attenuation_at(real_library, effective_energy(real_library))[water])
    assert linearization.attenuation == pytest.approx(mu) and linearization.energy_kev == pytest.approx(
        effective_energy(real_library))
    for length in (5.0, 23.0, 61.5):  # inside the table, between nodes, beyond it
        path = torch.full((1, 6, 10), length, device=solver.device, dtype=solver.dtype)
        signal = solver.primary_from_path_lengths(path, [water], 0.0)["signal"].double()
        measured = -torch.log(signal / solver.open_field_signal(0.0, dtype=F64))
        torch.testing.assert_close(linearization(measured), torch.full_like(measured, mu * length), rtol=2e-3,
                                   atol=0)
    hardening = linearization(measured[None]) - measured[None]
    assert float(hardening[0, :, 0].mean()) > float(hardening[0, :, 5].mean())  # the wings' spectrum is harder


# ---- the scatter scene beyond the slab ---------------------------------------------------------------------------

def test_scene_slices_repeat_the_last_filled_slice():
    R, C, S = 6, 6, 50
    X = torch.zeros(R, C, S)
    X[1:5, 1:5, 10:41] = 0.2                     # the body fills slices 10..40
    X[2:4, 2:4, 6:10] = 0.2                      # a thin end below it, under 0.8 of the central fill
    X[..., 25] += 0.01                           # mark the slices: each is told apart by its value
    X = X + torch.arange(S, dtype=torch.float32) * 1e-4
    out, (k0, k1) = extend_scene_slices(X, 0.1, 0.125)
    assert (k0, k1) == (12, 38)                  # the first and last filled slices, two slices inward
    assert torch.equal(out[:, :, k0:k1 + 1], X[:, :, k0:k1 + 1])
    assert bool((out[:, :, :k0] == X[:, :, k0:k0 + 1]).all()) and bool((out[:, :, k1:] == X[:, :, k1:k1 + 1]).all())
    empty = torch.zeros(R, C, S)
    assert extend_scene_slices(empty, 0.1, 0.125)[1] == (0, S - 1)


# ---- end to end ------------------------------------------------------------------------------------------------

@cuda
def test_citrus_recon_keeps_the_source_settings(real_library):
    """citrus_recon given an open-field image keeps every setting of the user's source (a subclass too) and restores
    it afterwards; with every new option at its default the loop is unchanged."""
    pytest.importorskip("mbirtorch")
    from citrus.utilities import synthesize_wlab_dataset, wlab_geometry, wlab_grid, wlab_solver

    class Marked(SourceModel):
        pass

    geometry = replace(wlab_geometry(), num_det_rows=20, num_det_channels=38, delta_det_row=1.6, delta_det_channel=1.6)
    config = SolverConfig(transport_cell_cm=2.0)
    data = synthesize_wlab_dataset(8, 2.0, noise=False, library=real_library, geometry=geometry, config=config)
    grid = wlab_grid(2.0)
    source = Marked(field_margin_rows_cm=2.0, field_margin_channels_cm=1.0)
    solver = wlab_solver(2.0, config=config, library=real_library, geometry=geometry, device=DEVICE, source=source)
    seen = []
    original = solver.material_fields_from_attenuation

    def spy(*args, **kwargs):
        seen.append(solver.source)
        return original(*args, **kwargs)

    solver.material_fields_from_attenuation = spy
    open_field = np.full(geometry.image_shape, 1.0e5)
    loop = CitrusConfig(num_outer_iterations=1, mbir_iterations_per_outer_iteration=1)
    citrus_recon(data["Y"], open_field, data["angles"], grid, geometry, solver, loop)
    assert isinstance(seen[0], Marked) and seen[0].field_margin_rows_cm == 2.0
    assert seen[0].field_margin_channels_cm == 1.0 and seen[0].emission_profile is not None
    assert solver.source is source


@cuda
def test_bowtie_filter_end_to_end(real_library):
    """Data of an aluminium bowtie (minimum thickness 2 mm) with a polychromatic primary, behind a grid: CITRUS with the
    filtered source and the water linearization gives flatter and more accurate water than CITRUS with the bowtie as a
    gray emission profile (the measured open field over the unfiltered model's)."""
    pytest.importorskip("mbirtorch")
    from citrus.recon.mbir_bridge import make_cone_model
    from citrus.utilities import synthesize_wlab_dataset, wlab_geometry, wlab_grid, wlab_solver

    geometry = replace(wlab_geometry(), num_det_rows=46, num_det_channels=38, delta_det_row=1.6, delta_det_channel=1.6)
    config = SolverConfig(transport_cell_cm=2.0)
    u, _ = geometry.pixel_uv(dtype=F64)
    filtered = SourceModel(filter_material="aluminium", filter_thickness_cm=0.2 + 2.0 * (u / u.abs().max()) ** 2)
    data = synthesize_wlab_dataset(24, 2.0, noise=False, library=real_library, geometry=geometry, config=config,
                                   grid_response=focused_1d_grid(), source=filtered, polychromatic_primary=True)
    assert data["meta"]["open_field"] == "image" and data["meta"]["polychromatic_primary"]
    grid = wlab_grid(2.0)
    model = make_cone_model(grid, geometry, data["angles"], device=DEVICE)
    open_field = torch.as_tensor(data["open_field"], device=DEVICE)
    regularization = estimate_regularization(model, -torch.log(torch.as_tensor(data["P"], device=DEVICE) / open_field),
                                             torch.as_tensor(data["Y"], device=DEVICE) / open_field)
    truth = torch.as_tensor(data["X_true"], device=DEVICE)
    mu_water = float(truth[5, 7, 10])
    water = (truth - mu_water).abs() < 1e-6
    water[:, :, :6] = False  # the cone-beam ends
    water[:, :, -6:] = False

    def run(source, correction):
        solver = wlab_solver(2.0, config=config, library=real_library, geometry=geometry, device=DEVICE,
                             source=source, grid_response=focused_1d_grid())
        loop = CitrusConfig(mbir_iterations_per_outer_iteration=2, num_outer_iterations=6,
                            beam_hardening_correction=correction)
        X, info = citrus_recon(data["Y"], data["open_field"], data["angles"], grid, geometry, solver, loop,
                               regularization=regularization, model=model)
        values = X[water] / mu_water
        return float(values.std()), float((values - 1.0).abs().mean()), info

    spread_filtered, error_filtered, info = run(filtered, True)
    assert info["water_linearization"] is not None
    assert float((info["emission_profile"] - 1.0).abs().max()) < 1e-5  # the filter explains the open field
    spread_gray, error_gray, gray_info = run(SourceModel(), False)
    assert gray_info["water_linearization"] is None
    assert spread_filtered < 0.95 * spread_gray, (spread_filtered, spread_gray)
    assert error_filtered < 0.8 * error_gray, (error_filtered, error_gray)
    # mbir_recon on the primary: the water linearization alone removes the cupping
    once = CitrusConfig(num_outer_iterations=4, mbir_iterations_per_outer_iteration=2, beam_hardening_correction=True)
    solver = wlab_solver(2.0, config=config, library=real_library, geometry=geometry, device=DEVICE, source=filtered,
                         grid_response=focused_1d_grid())
    X, _ = mbir_recon(data["P"], data["open_field"], data["angles"], grid, geometry, once,
                      regularization=regularization, model=model, solver=solver)
    X_plain, _ = mbir_recon(data["P"], data["open_field"], data["angles"], grid, geometry,
                            replace(once, beam_hardening_correction=False), regularization=regularization,
                            model=model)
    assert abs(float(X[water].mean()) / mu_water - 1.0) < 0.5 * abs(float(X_plain[water].mean()) / mu_water - 1.0)
    with pytest.raises(ValueError, match="solver"):
        mbir_recon(data["P"], data["open_field"], data["angles"], grid, geometry, once, model=model)


@cuda
def test_scatter_scene_end_to_end(real_library):
    """Data of the 40 cm WLAB prism from an 8-row panel with the field opened 10 cm past its rows, reconstructed on a
    40 cm grid whose end slices no ray measures: the scatter-to-primary ratio of CITRUS with those slices filled from
    the last well-filled ones is much closer to the truth."""
    pytest.importorskip("mbirtorch")
    from citrus.recon.mbir_bridge import make_cone_model
    from citrus.utilities import synthesize_wlab_dataset, wlab_geometry, wlab_grid, wlab_solver

    geometry = replace(wlab_geometry(), num_det_rows=8, num_det_channels=38, delta_det_row=1.6, delta_det_channel=1.6)
    config = SolverConfig(transport_cell_cm=2.0)
    source = SourceModel(field_margin_rows_cm=10.0)
    data = synthesize_wlab_dataset(16, 2.0, noise=False, library=real_library, geometry=geometry, config=config,
                                   source=source)
    grid = replace(wlab_grid(2.0), shape=(10, 15, 20))
    model = make_cone_model(grid, geometry, data["angles"], device=DEVICE)
    Y = torch.as_tensor(data["Y"], device=DEVICE)
    regularization = estimate_regularization(model, -torch.log(torch.as_tensor(data["P"], device=DEVICE) / 1e5),
                                             Y / 1e5)
    truth = torch.as_tensor(data["SPR"], device=DEVICE)
    errors, slices = {}, {}
    for extend in (True, False):
        solver = wlab_solver(2.0, config=config, library=real_library, geometry=geometry, device=DEVICE,
                             source=source)
        loop = CitrusConfig(mbir_iterations_per_outer_iteration=2, num_outer_iterations=5, extend_scatter_scene=extend,
                            scatter_scene_support_hu=-750.0)
        _, info = citrus_recon(data["Y"], 1e5, data["angles"], grid, geometry, solver, loop,
                               regularization=regularization, model=model)
        slices[extend] = info["scatter_scene_slices"]
        errors[extend] = float((info["spr"] - truth).norm() / truth.norm())
    k0, k1 = slices[True]
    assert 0 < k0 < k1 < grid.shape[2] - 1, slices
    assert errors[True] < 0.6 * errors[False], errors


@cuda
def test_hardening_materials_end_to_end(real_library):
    """Data of a water cylinder with two bone rods through an aluminium bowtie, polychromatic primary: with bone among
    the hardening materials the dark streak between the rods and the bone's own deficit fall well below those of the
    water linearization alone."""
    pytest.importorskip("mbirtorch")
    from citrus.recon.mbir_bridge import make_cone_model
    from citrus.utilities import synthesize_cone_beam_data, view_angles

    grid = Grid((24, 24, 6), 1.0)
    geometry = ConeGeometry(100.0, 150.0, 12, 48, 1.0, 1.0)
    u, _ = geometry.pixel_uv(dtype=F64)
    filtered = SourceModel(filter_material="aluminium", filter_thickness_cm=0.2 + 2.0 * (u / u.abs().max()) ** 2)

    def make_solver():
        return ScatterSolver(real_library, SolverConfig(transport_cell_cm=2.0), geometry,
                             detector=ideal_counting_response(), grid_response=focused_1d_grid(), source=filtered,
                             device=DEVICE)

    water, bone = real_library.materials.index("water"), real_library.materials.index("bone_cortical_icrp")
    x = torch.arange(24.0) - 11.5
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    inside = xx ** 2 + yy ** 2 <= 10.5 ** 2
    rods = ((xx.abs() - 5.0) ** 2 + yy ** 2) <= 2.2 ** 2
    rho = torch.zeros(real_library.num_materials, 24, 24, 6)
    rho[water] = (inside & ~rods).double()[:, :, None]
    rho[bone] = 1.85 * rods.double()[:, :, None]
    angles = view_angles(32, end_view_angle=2 * math.pi)
    data = synthesize_cone_beam_data(rho, grid, geometry, angles, make_solver(), noise=False,
                                     polychromatic_primary=True)
    model = make_cone_model(grid, geometry, angles, device=DEVICE)
    open_field = torch.as_tensor(data["open_field"], device=DEVICE)
    regularization = estimate_regularization(model, -torch.log(torch.as_tensor(data["P"], device=DEVICE) / open_field),
                                             torch.as_tensor(data["Y"], device=DEVICE) / open_field)
    truth = torch.as_tensor(data["X_true"], device=DEVICE)
    mu_water = float(truth[12, 12, 3])
    between = ((xx.abs() < 2.5) & (yy.abs() < 2.5)).to(DEVICE)[:, :, None].expand(grid.shape)
    bone_voxels = truth > 1.5 * mu_water
    results = {}
    for materials in ((), ("bone_cortical_icrp",)):
        loop = CitrusConfig(mbir_iterations_per_outer_iteration=3, num_outer_iterations=6,
                            beam_hardening_correction=True, hardening_materials=materials)
        X, info = citrus_recon(data["Y"], data["open_field"], angles, grid, geometry, make_solver(), loop,
                               regularization=regularization, model=model)
        results[materials] = (abs(float(X[between].mean()) / mu_water - 1.0),
                              abs(float(X[bone_voxels].mean() / truth[bone_voxels].mean()) - 1.0))
        assert ("hardening_offset_max" in info["records"][-1]) == bool(materials)
    (streak_water, deficit_water), (streak_bone, deficit_bone) = results[()], results[("bone_cortical_icrp",)]
    assert streak_bone < 0.5 * streak_water, results
    assert deficit_bone < 0.5 * deficit_water, results
    with pytest.raises(ValueError, match="water"):
        CitrusConfig(hardening_materials=("water",)).validate()
