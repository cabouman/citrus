"""The CITRUS loop's attenuation path (version 0.3.0): densities from X one material map at a time
(`utilities.attenuation_to_density`, `DensityModel.material`), the solver's fields built from X
(`ScatterSolver.material_fields_from_attenuation`) against the fields of the full density volume, the two-step primary
(`scatter_image.primary_path_lengths` / `primary_from_path_lengths`) against the primary image of version 0.2.0, and the
primaries of all views (`citrus.primary_signals`, the loop's MBIRTORCH per-material primary) against the solver's
Siddon primary (`ScatterSolver.primary`), the cross-check between the two operators."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest
import torch

from conftest import SPECTRUM_125KVP
from citrus.citrus import (CitrusConfig, CitrusState, DensityModel, citrus_step, citrus_view_step,
                           primary_signals)
from citrus.scatter_solver.config import SolverConfig
from citrus.scatter_solver.cross_sections import group_transmission, synthetic_library
from citrus.scatter_solver.detector import csi_response, detector_angle, focused_1d_grid, grid_angles, no_grid
from citrus.scatter_solver.ray_tracing import SiddonProjectors
from citrus.scatter_solver.scatter_image import primary_from_path_lengths, primary_image, primary_path_lengths
from citrus.scatter_solver.scatter_solver import ScatterSolver
from citrus.scatter_solver.source import SourceModel
from citrus.scatter_solver.uncollided import present_materials
from citrus.utilities import (attenuation_to_densities, attenuation_to_density, densities_to_attenuation,
                              nominal_attenuation, set_memory_cap, view_angles, wlab_geometry, wlab_grid,
                              wlab_phantom)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
SLOW = pytest.mark.skipif(os.environ.get("CITRUS_SLOW_TESTS") != "1",
                          reason="set CITRUS_SLOW_TESTS=1 (real library, cached)")
CONFIG = SolverConfig(num_groups=4, legendre_order=3, num_polar=4, num_azimuthal=8, cone_num_polar=8,
                      cone_num_azimuthal=16, source_iteration_tolerance=1e-4, max_source_iterations=40,
                      direction_chunk=8)
GRID = wlab_grid(0.5)  # 40 x 60 x 80 voxels in 1 cm cells: q = 2


def _library7(fine_spread: float = 0.3):
    return synthetic_library(CONFIG.num_groups, 5, num_materials=7, fine_spread=fine_spread)


def _solver(library, config=CONFIG, device=DEVICE, dtype=torch.float32, projectors=None, grid_response=None):
    return ScatterSolver(library, config, wlab_geometry(), projectors or SiddonProjectors(), csi_response(),
                         grid_response or no_grid(), device=device, dtype=dtype)


def _noisy_attenuation(library, grid, device, dtype=torch.float32, seed: int = 0) -> torch.Tensor:
    """WLAB attenuation plus noise (every bracket of the density model is hit, some voxels negative)."""
    X = densities_to_attenuation(library, wlab_phantom(grid, device=device, dtype=dtype))
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(tuple(grid.shape), generator=g, dtype=dtype).to(device)
    return X + 0.05 * noise * (X > 0)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


# ---- one material at a time ------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_attenuation_to_density_is_the_row_bit_for_bit(device, dtype):
    """attenuation_to_density(library, X, m) and DensityModel.material(X, m) (slab by slab) equal the row m of the full
    seven-material model bit for bit, at and next to every nominal attenuation, below zero and above bone; the
    negative fraction is counted without a float map."""
    library = _library7()
    mu = nominal_attenuation(library)
    X = torch.cat([torch.linspace(-0.2, 2.0 * float(mu.max()), 20001, dtype=torch.float64), mu, mu * (1 + 1e-7),
                   mu * (1 - 1e-7), torch.zeros(3, dtype=torch.float64)]).to(device, dtype)
    full = attenuation_to_densities(library, X)
    ints = torch.int32 if dtype == torch.float32 else torch.int64
    for m in range(7):
        assert torch.equal(attenuation_to_density(library, X, m).view(ints), full[m].view(ints)), m
    X3 = (torch.rand(17, 9, 5, dtype=dtype) * 1.5 * float(mu.max()) - 0.05).to(device)
    dm = DensityModel.from_library(library)
    dm.slab_voxels = 100  # 3 rows per slab
    full3 = dm(X3)
    for m in range(7):
        assert torch.equal(dm.material(X3, m).view(ints), full3[m].view(ints)), m
    assert dm.negative_fraction(X3) == pytest.approx(float((X3 < 0).double().mean()), abs=1e-12)
    assert dm.last_negative_fraction == dm.negative_fraction(X3)
    with pytest.raises(ValueError):
        attenuation_to_density(library, X, 7)


# ---- fields from the attenuation --------------------------------------------------------------------------------

@pytest.mark.parametrize("trace", ["quadrature", "fine"])
def test_fields_from_attenuation_equal_the_density_path(device, trace):
    """WLAB 0.5 cm in 1 cm cells on CUDA (1 cm in 2 cm cells on the CPU, for time; q = 2 in both), noisy X:
    material_fields_from_attenuation(X, grid, DensityModel) gives the
    MaterialFields of material_fields(DensityModel(X), grid) (rho_coarse, rho_quadrature, mu_t_coarse, mu_s_l_coarse,
    the pair representation) to float rounding without rho_fine; its fine maps are the density volume's rows bit for
    bit; the uncollided fluence (quadrature and fine traces), the primary (one Siddon trace per material map) and a
    CITRUS step agree to float rounding."""
    library = _library7()
    grid, cell = (GRID, 1.0) if device.type == "cuda" else (wlab_grid(1.0), 2.0)
    s = _solver(library, replace(CONFIG, uncollided_trace_grid=trace, transport_cell_cm=cell,
                                 quadrature_point_spacing_cm=cell / 2), device=device)
    X = _noisy_attenuation(library, grid, device)
    dm = DensityModel.from_library(library)
    rho = dm(X)
    fr = s.material_fields(rho, grid)
    fx = s.material_fields_from_attenuation(X, grid, dm)
    assert fx.rho_fine is None and fx.attenuation is X and fx.density_model is dm and fx.q == fr.q == 2
    present = present_materials(rho)
    assert list(fx.fine_materials) == (list(range(7)) if present is None else present)
    for m in range(7):
        assert torch.equal(fx.fine_material(m), rho[m])
    for name in ("rho_coarse", "rho_quadrature", "mu_t_coarse", "mu_s_l_coarse"):
        a, b = getattr(fx, name), getattr(fr, name)
        assert a.shape == b.shape and _rel(a, b) < 1e-6, name
    assert (fx.pairs_quadrature is None) == (fr.pairs_quadrature is None)
    if fr.pairs_quadrature is not None:
        assert torch.equal(fx.pairs_quadrature.index, fr.pairs_quadrature.index)
    for angle in (0.0, 1.1):
        assert _rel(s.uncollided_fluence(fx, angle), s.uncollided_fluence(fr, angle)) < 1e-5
        assert _rel(s.primary(fx, angle)["signal"], s.primary(fr, angle)["signal"]) < 1e-6
    a = citrus_step(s, CitrusState(), rho, grid, 0.4, fields=fr)
    b = citrus_view_step(s, CitrusState(), fx, 0.4)
    c = citrus_view_step(s, CitrusState(), fx, 0.4, primary=False)
    assert _rel(b[0], a[0]) < 1e-5 and _rel(b[1], a[1]) < 1e-6 and c[1] is None and torch.equal(c[0], b[0])
    assert _rel(b[2].collided_moments, a[2].collided_moments) < 1e-5


@cuda
def test_fields_from_attenuation_hold_no_density_volume(monkeypatch):
    """The X path's peak memory above X: the fields it returns plus one material map, the resampling intermediates
    (about one more map at q = 2: measured 2.2 maps in all) and the density model's slab temporaries, never the
    [7, R, C, S] density volume (0.25 cm WLAB grid, 1.5 M voxels, 16 slabs) when the maps exceed
    FINE_MAP_BUDGET_BYTES. Within the budget the fields keep exactly the nonzero maps, and the peak is those plus the
    same intermediates."""
    import citrus.scatter_solver.scatter_solver as scatter_solver_module

    set_memory_cap(4400.0)
    grid = wlab_grid(0.25)
    library = _library7()
    s = _solver(library, device="cuda")
    X = _noisy_attenuation(library, grid, "cuda")
    dm = DensityModel.from_library(library)
    dm.slab_voxels = X.numel() // 16
    one_map = X.numel() * X.element_size()
    for budget in (0, scatter_solver_module.FINE_MAP_BUDGET_BYTES):
        monkeypatch.setattr(scatter_solver_module, "FINE_MAP_BUDGET_BYTES", budget)
        s.material_fields_from_attenuation(X, grid, dm)  # warm-up (resampling matrices cached)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        f = s.material_fields_from_attenuation(X, grid, dm)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        kept = torch.cuda.memory_allocated() - base
        assert peak <= kept + 2.5 * one_map, (budget, peak / one_map, kept / one_map)
        if budget == 0:
            assert f.fine_maps is None
            assert peak < 7 * one_map  # far below the density volume alone
        else:
            assert sorted(f.fine_maps) == list(f.fine_materials)
            for m in f.fine_materials:
                assert torch.equal(f.fine_material(m), dm.material(X, m))
        del f


# ---- the two-step primary --------------------------------------------------------------------------------------

def _primary_image_v020(library, rho, grid, geometry, angle, source, projectors, detector, grid_response,
                        samples_per_pixel_axis=None):
    """`scatter_image.primary_image` of version 0.2.0, verbatim (the reference of the refactoring)."""
    device, dtype = rho.device, rho.dtype
    n = int(samples_per_pixel_axis) if samples_per_pixel_axis is not None else int(getattr(projectors,
                                                                                           "samples_per_pixel_axis",
                                                                                           1) or 1)
    Rd, Cd = geometry.num_det_rows, geometry.num_det_channels
    subpixel_geometry = geometry if n == 1 else replace(geometry, num_det_rows=Rd * n, num_det_channels=Cd * n,
                                                        delta_det_row=geometry.delta_det_row / n,
                                                        delta_det_channel=geometry.delta_det_channel / n)
    present = present_materials(rho)
    present_rho = rho if present is None else rho[present]
    if getattr(projectors, "samples_per_pixel_axis", None) is not None:
        path = projectors.primary_project(present_rho, grid, subpixel_geometry, angle, samples_per_pixel_axis=1)
    else:
        path = projectors.primary_project(present_rho, grid, subpixel_geometry, angle)
    trans = group_transmission(library, path, materials=present)
    if n > 1:
        trans = trans.reshape(-1, Rd, n, Cd, n).mean(dim=(2, 4))
    pix = geometry.pixel_centers(angle, device, dtype)
    r0 = torch.tensor(geometry.source_position(angle), device=device, dtype=dtype)
    d = pix - r0
    dist2 = (d * d).sum(-1)
    b = source.angular_weight(d, geometry, angle)
    fluence = (b / dist2)[None] * trans
    omega_p = d / dist2.sqrt()[..., None]
    e_g = library.group_energy.to(device, dtype)
    theta0 = detector_angle(omega_p, geometry, angle)
    planar = fluence * torch.cos(theta0)
    detector_response = detector.at(e_g, theta0)
    th1, th2 = grid_angles(omega_p, pix, geometry, angle, grid_response.focal_distance_cm)
    grid_transmission = grid_response.at(e_g, th1, th2)
    signal_per_group = planar * grid_transmission * detector_response
    return {"fluence": fluence, "planar_fluence": planar, "signal_per_group": signal_per_group,
            "signal": signal_per_group.sum(0)}


@pytest.mark.parametrize("samples_per_pixel_axis", [1, 2])
def test_primary_from_path_lengths_equals_the_v020_primary(device, samples_per_pixel_axis):
    """primary_image = primary_from_path_lengths(primary_path_lengths(...)) equals the primary image of version 0.2.0
    bit for bit (materials present only, sub-pixel sampling, focused grid); tracing the materials one map at a time and
    concatenating gives the same path lengths; a path of the wrong shape is rejected."""
    library = _library7().to(device, torch.float32)
    geometry = wlab_geometry()
    rho = attenuation_to_densities(library, _noisy_attenuation(library, GRID, device))
    P = SiddonProjectors()
    source, grid_response = SourceModel(), focused_1d_grid()
    for angle in (0.0, 2.2):
        reference = _primary_image_v020(library, rho, GRID, geometry, angle, source, P, csi_response(), grid_response,
                                        samples_per_pixel_axis)
        new = primary_image(library, rho, GRID, geometry, angle, source, P, csi_response(), grid_response,
                            samples_per_pixel_axis)
        for k in reference:
            assert torch.equal(new[k], reference[k]), k
        present = present_materials(rho)
        materials = list(range(7)) if present is None else present
        one = torch.cat([primary_path_lengths(P, rho[m][None], GRID, geometry, angle,
                                              samples_per_pixel_axis) for m in materials])
        # CUDA: bit for bit
        assert _rel(one, primary_path_lengths(P, rho[materials], GRID, geometry, angle, samples_per_pixel_axis)) < 1e-6
        via = primary_from_path_lengths(library, one, materials, geometry, angle, source, csi_response(), grid_response)
        assert _rel(via["signal"], reference["signal"]) < 1e-6
    with pytest.raises(ValueError, match="path"):
        primary_from_path_lengths(library, one[:, :-1], materials, geometry, 0.0, source, csi_response(), grid_response)


# ---- primaries of all views: MBIRTORCH vs Siddon ---------------------------------------------------------------

@cuda
def test_primary_signals_mbirtorch_close_to_siddon():
    """primary_signals over 12 views of the WLAB short scan (0.5 cm grid), the loop's primary: MBIRTORCH's
    pixel-averaged line integrals give the solver's Siddon primary (ScatterSolver.primary per view, one ray per pixel
    center) to 5 % (relative L2, measured 3.1 % with this synthetic library on the small panel; median pixel 8e-4: the
    differences sit on the pixels whose rays graze a face of the prism, where a center ray and a pixel average
    differ); MBIRTORCH chunks of 5 views (sub-models) equal one chunk to float rounding."""
    pytest.importorskip("mbirtorch")
    from citrus.recon.mbir_bridge import ViewChunkProjector, make_cone_model

    set_memory_cap(4400.0)
    library = _library7()
    s = _solver(library, device="cuda")
    geometry = s.geometry
    angles = view_angles(12, geometry)
    X = densities_to_attenuation(s.library, wlab_phantom(GRID, device="cuda"))
    dm = DensityModel.from_library(s.library)
    fx = s.material_fields_from_attenuation(X, GRID, dm)
    reference = torch.stack([s.primary(fx, float(a))["signal"] for a in angles])  # the solver's Siddon primary
    model = make_cone_model(GRID, geometry, angles, device="cuda")
    mb = ViewChunkProjector(GRID, geometry, angles, model=model, device="cuda")
    assert mb.chunks(5) == [(0, 5), (5, 10), (10, 12)]
    assert mb.model_for(0, 12) is model and mb.model_for(0, 5) is mb.model_for(0, 5)
    full = mb.project(fx.fine_material(fx.fine_materials[0]))
    part = mb.project(fx.fine_material(fx.fine_materials[0]), 5, 10)
    assert _rel(part, full[5:10]) < 1e-5
    p1 = primary_signals(s, fx, angles, mb, 12)
    p5 = primary_signals(s, fx, angles, mb, 5)
    assert _rel(p5, p1) < 1e-5 and _rel(primary_signals(s, fx, angles, mb), p1) < 1e-5
    rel = (p1 - reference) / reference
    assert float((p1 - reference).norm() / reference.norm()) < 0.05
    assert float(rel.abs().median()) < 2e-3
    with pytest.raises(ValueError, match="views"):
        primary_signals(s, fx, angles[:-1], mb)


@SLOW
@cuda
def test_primary_signals_mbirtorch_close_to_siddon_real_library():
    """The same with the real library on 36 views of the 0.5 cm WLAB short scan (measured on 180 views: relative L2
    0.9 %, median pixel 0.05 %, 99th percentile 3 %, maximum 5x at a grazing ray through 30 cm of water)."""
    pytest.importorskip("mbirtorch")
    from citrus.recon.mbir_bridge import ViewChunkProjector, make_cone_model
    from citrus.utilities import full_field_wlab_geometry, wlab_solver

    set_memory_cap(4400.0)
    geometry = full_field_wlab_geometry()
    s = wlab_solver(0.5, geometry=geometry, device="cuda", spectrum=SPECTRUM_125KVP)
    angles = view_angles(36, geometry)
    X = densities_to_attenuation(s.library, wlab_phantom(GRID, device="cuda"))
    fx = s.material_fields_from_attenuation(X, GRID, DensityModel.from_library(s.library))
    reference = torch.stack([s.primary(fx, float(a))["signal"] for a in angles])  # the solver's Siddon primary
    mb = ViewChunkProjector(GRID, geometry, angles, model=make_cone_model(GRID, geometry, angles, device="cuda"),
                            device="cuda")
    p = primary_signals(s, fx, angles, mb)
    rel = (p - reference) / reference
    assert float((p - reference).norm() / reference.norm()) < 0.015
    assert float(rel.abs().median()) < 2e-3


# ---- configuration ----------------------------------------------------------------------------------------------

def test_citrus_config_primary_and_resume_fields():
    """The loop always uses the MBIRTORCH per-material primary: CitrusConfig has no primary_projector (removed in
    version 0.5.0), only the chunking of the primary projections."""
    config = CitrusConfig()
    assert config.primary_views_per_chunk is None and not hasattr(config, "primary_projector")
    assert config.resume_error_sinogram is True
    config.validate()
    CitrusConfig(primary_views_per_chunk=3, resume_error_sinogram=False).validate()
    with pytest.raises(TypeError):
        CitrusConfig(primary_projector="siddon")
    for bad in (dict(primary_views_per_chunk=0), dict(primary_views_per_chunk=2.5),
                dict(primary_views_per_chunk=True), dict(resume_error_sinogram="yes")):
        with pytest.raises(ValueError):
            CitrusConfig(**bad).validate()
