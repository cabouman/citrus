"""The CITRUS loop's attenuation path (version 0.3.0): densities from X one material map at a time
(`utilities.attenuation_to_density`, `DensityModel.material`), the solver's fields built from X
(`ScatterSolver.material_fields_from_attenuation`) against the fields of the full density volume, the two-step primary
(`scatter_image.primary_path_lengths` / `primary_from_path_lengths`) against the primary image of version 0.2.0, and the
primaries of all views (`citrus.primary_signals`) with MBIRTORCH's projector against the Siddon primary."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest
import torch

from citrus.citrus import (CitrusConfig, CitrusState, DensityModel, SiddonViewProjector, citrus_step,
                           citrus_view_step, primary_signals)
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

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
SLOW = pytest.mark.skipif(os.environ.get("CITRUS_SLOW_TESTS") != "1",
                          reason="set CITRUS_SLOW_TESTS=1 (real XrayPhysics library, cached)")
CFG = SolverConfig(num_groups=4, legendre_order=3, num_polar=4, num_azimuthal=8, cone_num_polar=8,
                   cone_num_azimuthal=16, si_tol=1e-4, si_max_iterations=40, direction_chunk=8)
GRID = wlab_grid(0.5)  # 40 x 60 x 80 voxels in 1 cm cells: q = 2


def _lib7(fine_spread: float = 0.3):
    return synthetic_library(CFG.num_groups, 5, num_materials=7, fine_spread=fine_spread)


def _solver(lib, cfg=CFG, device=DEV, dtype=torch.float32, projectors=None, grid_response=None):
    return ScatterSolver(lib, cfg, wlab_geometry(), projectors or SiddonProjectors(), csi_response(),
                         grid_response or no_grid(), device=device, dtype=dtype)


def _noisy_attenuation(lib, grid, device, dtype=torch.float32, seed: int = 0) -> torch.Tensor:
    """WLAB attenuation plus noise (every bracket of the density model is hit, some voxels negative)."""
    X = densities_to_attenuation(lib, wlab_phantom(grid, device=device, dtype=dtype))
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(tuple(grid.shape), generator=g, dtype=dtype).to(device)
    return X + 0.05 * noise * (X > 0)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


# ---- one material at a time ------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_attenuation_to_density_is_the_row_bit_for_bit(device, dtype):
    """attenuation_to_density(lib, X, m) and DensityModel.material(X, m) (slab by slab) equal the row m of the full
    seven-material model bit for bit, at and next to every nominal attenuation, below zero and above bone; the
    negative fraction is counted without a float map."""
    lib = _lib7()
    mu = nominal_attenuation(lib)
    X = torch.cat([torch.linspace(-0.2, 2.0 * float(mu.max()), 20001, dtype=torch.float64), mu, mu * (1 + 1e-7),
                   mu * (1 - 1e-7), torch.zeros(3, dtype=torch.float64)]).to(device, dtype)
    full = attenuation_to_densities(lib, X)
    ints = torch.int32 if dtype == torch.float32 else torch.int64
    for m in range(7):
        assert torch.equal(attenuation_to_density(lib, X, m).view(ints), full[m].view(ints)), m
    X3 = (torch.rand(17, 9, 5, dtype=dtype) * 1.5 * float(mu.max()) - 0.05).to(device)
    dm = DensityModel.from_library(lib)
    dm.slab_voxels = 100  # 3 rows per slab
    full3 = dm(X3)
    for m in range(7):
        assert torch.equal(dm.material(X3, m).view(ints), full3[m].view(ints)), m
    assert dm.negative_fraction(X3) == pytest.approx(float((X3 < 0).double().mean()), abs=1e-12)
    assert dm.last_negative_fraction == dm.negative_fraction(X3)
    with pytest.raises(ValueError):
        attenuation_to_density(lib, X, 7)


# ---- fields from the attenuation --------------------------------------------------------------------------------

@pytest.mark.parametrize("trace", ["quadrature", "fine"])
def test_fields_from_attenuation_equal_the_density_path(device, trace):
    """WLAB 0.5 cm in 1 cm cells on CUDA (1 cm in 2 cm cells on the CPU, for time; q = 2 in both), noisy X:
    material_fields_from_attenuation(X, grid, DensityModel) gives the
    MaterialFields of material_fields(DensityModel(X), grid) (rho_coarse, rho_quad, mu_t_coarse, mu_s_l_coarse, the
    pair representation) to float rounding without rho_fine; its fine maps are the density volume's rows bit for bit;
    the uncollided fluence (quadrature and fine traces), the primary (one Siddon trace per material map) and a CITRUS
    step agree to float rounding."""
    lib = _lib7()
    grid, cell = (GRID, 1.0) if device.type == "cuda" else (wlab_grid(1.0), 2.0)
    s = _solver(lib, replace(CFG, uncollided_trace_grid=trace, transport_cell_cm=cell,
                             quadrature_point_spacing_cm=cell / 2), device=device)
    X = _noisy_attenuation(lib, grid, device)
    dm = DensityModel.from_library(lib)
    rho = dm(X)
    fr = s.material_fields(rho, grid)
    fx = s.material_fields_from_attenuation(X, grid, dm)
    assert fx.rho_fine is None and fx.attenuation is X and fx.density_model is dm and fx.q == fr.q == 2
    present = present_materials(rho)
    assert list(fx.fine_materials) == (list(range(7)) if present is None else present)
    for m in range(7):
        assert torch.equal(fx.fine_material(m), rho[m])
    for name in ("rho_coarse", "rho_quad", "mu_t_coarse", "mu_s_l_coarse"):
        a, b = getattr(fx, name), getattr(fr, name)
        assert a.shape == b.shape and _rel(a, b) < 1e-6, name
    assert (fx.pairs_quad is None) == (fr.pairs_quad is None)
    if fr.pairs_quad is not None:
        assert torch.equal(fx.pairs_quad.index, fr.pairs_quad.index)
    for angle in (0.0, 1.1):
        assert _rel(s.uncollided(fx, angle), s.uncollided(fr, angle)) < 1e-5
        assert _rel(s.primary(fx, angle)["signal"], s.primary(fr, angle)["signal"]) < 1e-6
    a = citrus_step(s, CitrusState(), rho, grid, 0.4, fields=fr)
    b = citrus_view_step(s, CitrusState(), fx, 0.4)
    c = citrus_view_step(s, CitrusState(), fx, 0.4, primary=False)
    assert _rel(b[0], a[0]) < 1e-5 and _rel(b[1], a[1]) < 1e-6 and c[1] is None and torch.equal(c[0], b[0])
    assert _rel(b[2].Phi_gt1, a[2].Phi_gt1) < 1e-5


@cuda
def test_fields_from_attenuation_hold_no_density_volume():
    """The X path's peak memory above X: the fields it returns plus one material map, the resampling intermediates
    (about one more map at q = 2: measured 2.2 maps in all) and the density model's slab temporaries, never the
    [7, R, C, S] density volume (0.25 cm WLAB grid, 1.5 M voxels, 16 slabs)."""
    set_memory_cap(4400.0)
    grid = wlab_grid(0.25)
    lib = _lib7()
    s = _solver(lib, device="cuda")
    X = _noisy_attenuation(lib, grid, "cuda")
    dm = DensityModel.from_library(lib)
    dm.slab_voxels = X.numel() // 16
    s.material_fields_from_attenuation(X, grid, dm)  # warm-up (resampling matrices cached)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    f = s.material_fields_from_attenuation(X, grid, dm)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    kept = torch.cuda.memory_allocated() - base
    one_map = X.numel() * X.element_size()
    assert peak <= kept + 2.5 * one_map, (peak / one_map, kept / one_map)
    assert peak < 7 * one_map  # far below the density volume alone
    del f


# ---- the two-step primary --------------------------------------------------------------------------------------

def _primary_image_v020(lib, rho, grid, geom, angle, source, projectors, detector, grid_response, n_sub=None):
    """`scatter_image.primary_image` of version 0.2.0, verbatim (the reference of the refactoring)."""
    dev, dt = rho.device, rho.dtype
    n = int(n_sub) if n_sub is not None else int(getattr(projectors, "n_sub", 1) or 1)
    Rd, Cd = geom.num_det_rows, geom.num_det_channels
    geom_sub = geom if n == 1 else replace(geom, num_det_rows=Rd * n, num_det_channels=Cd * n,
                                           delta_det_row=geom.delta_det_row / n,
                                           delta_det_channel=geom.delta_det_channel / n)
    present = present_materials(rho)
    src = rho if present is None else rho[present]
    if getattr(projectors, "n_sub", None) is not None:
        path = projectors.primary_project(src, grid, geom_sub, angle, n_sub=1)
    else:
        path = projectors.primary_project(src, grid, geom_sub, angle)
    trans = group_transmission(lib, path, materials=present)
    if n > 1:
        trans = trans.reshape(-1, Rd, n, Cd, n).mean(dim=(2, 4))
    pix = geom.pixel_centers(angle, dev, dt)
    r0 = torch.tensor(geom.source_position(angle), device=dev, dtype=dt)
    d = pix - r0
    dist2 = (d * d).sum(-1)
    b = source.angular_weight(d, geom, angle)
    fluence = (b / dist2)[None] * trans
    omega_p = d / dist2.sqrt()[..., None]
    e_g = lib.group_energy.to(dev, dt)
    theta0 = detector_angle(omega_p, geom, angle)
    planar = fluence * torch.cos(theta0)
    d_resp = detector.at(e_g, theta0)
    th1, th2 = grid_angles(omega_p, pix, geom, angle, grid_response.focal_distance_cm)
    g_resp = grid_response.at(e_g, th1, th2)
    spg = planar * g_resp * d_resp
    return {"fluence": fluence, "planar_fluence": planar, "signal_per_group": spg, "signal": spg.sum(0)}


@pytest.mark.parametrize("n_sub", [1, 2])
def test_primary_from_path_lengths_equals_the_v020_primary(device, n_sub):
    """primary_image = primary_from_path_lengths(primary_path_lengths(...)) equals the primary image of version 0.2.0
    bit for bit (materials present only, sub-pixel sampling, focused grid); tracing the materials one map at a time and
    concatenating gives the same path lengths; a path of the wrong shape is rejected."""
    lib = _lib7().to(device, torch.float32)
    geom = wlab_geometry()
    rho = attenuation_to_densities(lib, _noisy_attenuation(lib, GRID, device))
    P = SiddonProjectors()
    src, grid_resp = SourceModel(), focused_1d_grid()
    for angle in (0.0, 2.2):
        ref = _primary_image_v020(lib, rho, GRID, geom, angle, src, P, csi_response(), grid_resp, n_sub)
        new = primary_image(lib, rho, GRID, geom, angle, src, P, csi_response(), grid_resp, n_sub)
        for k in ref:
            assert torch.equal(new[k], ref[k]), k
        present = present_materials(rho)
        mats = list(range(7)) if present is None else present
        one = torch.cat([primary_path_lengths(P, rho[m][None], GRID, geom, angle, n_sub) for m in mats])
        assert _rel(one, primary_path_lengths(P, rho[mats], GRID, geom, angle, n_sub)) < 1e-6  # CUDA: bit for bit
        via = primary_from_path_lengths(lib, one, mats, geom, angle, src, csi_response(), grid_resp)
        assert _rel(via["signal"], ref["signal"]) < 1e-6
    with pytest.raises(ValueError, match="path"):
        primary_from_path_lengths(lib, one[:, :-1], mats, geom, 0.0, src, csi_response(), grid_resp)


# ---- primaries of all views: MBIRTORCH vs Siddon ---------------------------------------------------------------

@cuda
def test_primary_signals_mbirtorch_close_to_siddon():
    """primary_signals over 12 views of the WLAB short scan (0.5 cm grid): the Siddon projector reproduces
    ScatterSolver.primary per view bit for bit in any chunking; MBIRTORCH's pixel-averaged line integrals give the same
    primary to 5 % (relative L2, measured 3.1 % with this synthetic library on the small panel; median pixel 8e-4: the
    differences sit on the pixels whose rays graze a face of the prism, where a center ray and a pixel average
    differ); MBIRTORCH chunks of 5 views (sub-models) equal one chunk to
    float rounding."""
    pytest.importorskip("mbirtorch")
    from citrus.recon.mbir_bridge import ViewChunkProjector, make_cone_model

    set_memory_cap(4400.0)
    lib = _lib7()
    s = _solver(lib, device="cuda")
    geom = s.geom
    angles = view_angles(12, geom)
    X = densities_to_attenuation(s.lib, wlab_phantom(GRID, device="cuda"))
    dm = DensityModel.from_library(s.lib)
    fx = s.material_fields_from_attenuation(X, GRID, dm)
    ref = torch.stack([s.primary(fx, float(a))["signal"] for a in angles])
    sid = SiddonViewProjector(s, GRID, angles)
    assert sid.chunks(5) == [(0, 5), (5, 10), (10, 12)]
    for vpc in (None, 5, 12):
        assert torch.equal(primary_signals(s, fx, angles, sid, vpc), ref), vpc
    model = make_cone_model(GRID, geom, angles, device="cuda")
    mb = ViewChunkProjector(GRID, geom, angles, model=model, device="cuda")
    assert mb.model_for(0, 12) is model and mb.model_for(0, 5) is mb.model_for(0, 5)
    full = mb.project(fx.fine_material(fx.fine_materials[0]))
    part = mb.project(fx.fine_material(fx.fine_materials[0]), 5, 10)
    assert _rel(part, full[5:10]) < 1e-5
    p1 = primary_signals(s, fx, angles, mb, 12)
    p5 = primary_signals(s, fx, angles, mb, 5)
    assert _rel(p5, p1) < 1e-5
    rel = (p1 - ref) / ref
    assert float((p1 - ref).norm() / ref.norm()) < 0.05
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
    geom = full_field_wlab_geometry()
    s = wlab_solver(0.5, geom=geom, device="cuda")
    angles = view_angles(36, geom)
    X = densities_to_attenuation(s.lib, wlab_phantom(GRID, device="cuda"))
    fx = s.material_fields_from_attenuation(X, GRID, DensityModel.from_library(s.lib))
    ref = primary_signals(s, fx, angles, SiddonViewProjector(s, GRID, angles))
    mb = ViewChunkProjector(GRID, geom, angles, model=make_cone_model(GRID, geom, angles, device="cuda"),
                            device="cuda")
    p = primary_signals(s, fx, angles, mb)
    rel = (p - ref) / ref
    assert float((p - ref).norm() / ref.norm()) < 0.015
    assert float(rel.abs().median()) < 2e-3


# ---- configuration ----------------------------------------------------------------------------------------------

def test_citrus_config_primary_and_resume_fields():
    cfg = CitrusConfig()
    assert cfg.primary_projector == "mbirtorch" and cfg.primary_views_per_chunk is None
    assert cfg.resume_error_sinogram is True
    cfg.validate()
    CitrusConfig(primary_projector="siddon", primary_views_per_chunk=3, resume_error_sinogram=False).validate()
    for bad in (dict(primary_projector="leap"), dict(primary_views_per_chunk=0), dict(primary_views_per_chunk=2.5),
                dict(primary_views_per_chunk=True), dict(resume_error_sinogram="yes")):
        with pytest.raises(ValueError):
            CitrusConfig(**bad).validate()
