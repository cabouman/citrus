"""Detector deposition with K x-ray escape, the analytic anti-scatter grid and the responses averaged over each energy
group (citrus/scatter_solver/detector.py, grid_model.py, SolverConfig.group_averaged_responses).

The Geant4 numbers in tests/data/geant4_grid_validation.json come from the grid Monte Carlo of the validation folder
(0.6 mm CsI, 1-D lead grids of 36 um strips at 60 lines per cm with an air interspace)."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import SPECTRUM_125KVP
from citrus.scatter_solver import (Material, ScatterSolver, SolverConfig, anti_scatter_grid, detector_response,
                                   no_grid)
from citrus.scatter_solver.detector import group_averaged_responses
from citrus.scatter_solver.geometry import ConeGeometry, Grid

F64 = torch.float64
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the analytic grid's secondaries run on CUDA here")
GEANT4 = json.loads((Path(__file__).parent / "data" / "geant4_grid_validation.json").read_text())
COBRA_PITCH, COBRA_STRIP = 1.0 / 60.0, 36e-4  # cm


@pytest.fixture(scope="module")
def photon_data():
    from citrus.scatter_solver.photon_data import resolve_backend

    try:
        return resolve_backend()
    except Exception as exc:  # no photon data
        pytest.skip(f"photon data unavailable: {exc}")


def _at(response, energy_kev, theta_rad=0.0) -> float:
    return float(response.at(torch.tensor([energy_kev], dtype=F64), torch.tensor([theta_rad], dtype=F64)))


# ---- detector deposition with K x-ray escape --------------------------------------------------------------------

def test_fluorescence_escape_deposit_matches_geant4(photon_data):
    """The escape deposit of 0.6 mm CsI is within 3.5 % of Geant4 (it keeps no re-absorbed Compton photon, so it
    lies a little below), where full deposition is 9 % or more above."""
    escape = detector_response("cesium_iodide", 0.06, deposition="fluorescence_escape", backend=photon_data)
    full = detector_response("cesium_iodide", 0.06, backend=photon_data)
    assert "fluorescence_escape" in escape.name
    for energy, angle, geant4 in GEANT4["csi_deposit"]:
        theta = math.radians(angle)
        assert abs(_at(escape, energy, theta) / geant4 - 1.0) < 0.035, (energy, angle)
        assert _at(full, energy, theta) / geant4 - 1.0 > 0.09, (energy, angle)


def test_fluorescence_escape_lowers_the_deposit_above_the_k_edges(photon_data):
    """Below the iodine K edge nothing escapes (the deposit equals "local"); just above the Cs and I K edges the K
    x-rays escape and the deposit falls well below "full" and "local"; a thick slab keeps only those that head
    deeper (most escape through the front face). Gd2O2S loses its gadolinium K x-rays above 50.2 keV only."""
    energies = np.array([30.0, 33.0, 34.0, 37.0, 40.0, 60.0])
    options = dict(energies=energies, backend=photon_data)
    escape = detector_response("cesium_iodide", 0.06, deposition="fluorescence_escape", **options)
    local = detector_response("cesium_iodide", 0.06, deposition="local", **options)
    full = detector_response("cesium_iodide", 0.06, deposition="full", **options)
    for e in (30.0, 33.0):
        assert _at(escape, e) == pytest.approx(_at(local, e), rel=1e-12)
    for e in (34.0, 37.0, 40.0):
        assert _at(escape, e) < 0.92 * _at(full, e) and _at(escape, e) < 0.95 * _at(local, e)
    thick = detector_response("cesium_iodide", 1.0, deposition="fluorescence_escape", **options)
    thick_local = detector_response("cesium_iodide", 1.0, deposition="local", **options)
    lost_thin = 1.0 - _at(escape, 40.0) / _at(local, 40.0)
    lost_thick = 1.0 - _at(thick, 40.0) / _at(thick_local, 40.0)
    assert 0.5 * lost_thin < lost_thick < lost_thin  # the x-rays from near the front face still escape
    oxysulfide = detector_response("gadolinium_oxysulfide", 0.02, deposition="fluorescence_escape",
                                   energies=np.array([45.0, 49.0, 52.0, 70.0]), backend=photon_data)
    oxysulfide_local = detector_response("gadolinium_oxysulfide", 0.02, deposition="local",
                                         energies=np.array([45.0, 49.0, 52.0, 70.0]), backend=photon_data)
    assert _at(oxysulfide, 49.0) == pytest.approx(_at(oxysulfide_local, 49.0), rel=1e-6)  # sulfur K x-rays only
    assert _at(oxysulfide, 52.0) < 0.9 * _at(oxysulfide_local, 52.0)
    counting = detector_response("cesium_iodide", 0.06, mode="photon_counting", deposition="fluorescence_escape",
                                 **options)
    counting_local = detector_response("cesium_iodide", 0.06, mode="photon_counting", deposition="local", **options)
    torch.testing.assert_close(counting.response, counting_local.response)
    with pytest.raises(ValueError, match="deposition"):
        detector_response("cesium_iodide", 0.06, deposition="partial", backend=photon_data)


def test_detector_of_a_user_material(photon_data):
    """A ``Material`` definition is accepted with its own density and name."""
    screen = Material("my_csi", 4.51, "0.5115*Cs + 0.4885*I")
    mine = detector_response(screen, 0.06, deposition="fluorescence_escape", energies=np.array([40.0, 60.0]),
                             backend=photon_data)
    catalogue = detector_response("cesium_iodide", 0.06, deposition="fluorescence_escape",
                                  energies=np.array([40.0, 60.0]), backend=photon_data)
    assert mine.name.startswith("my_csi")
    torch.testing.assert_close(mine.response, catalogue.response, rtol=2e-3, atol=0.0)


# ---- the analytic grid --------------------------------------------------------------------------------------------

def test_direct_transmission_is_the_mean_over_the_entry_offset():
    """The closed-form direct part equals a brute-force average over 200000 entry offsets of the exact strip path,
    and at theta1 = 0 the open fraction plus the strip tops."""
    from citrus.scatter_solver.grid_model import _direct, _strip_path

    p, t, h = 0.05, 0.008, 0.3
    mu_strip = torch.tensor([3.0, 40.0, 400.0], dtype=F64)
    mu_inter = torch.tensor([0.2, 0.0, 1.0], dtype=F64)
    theta1 = torch.tensor([0.0, 0.01, 0.05, 0.16, 0.5, 1.2], dtype=F64)
    theta2 = torch.tensor([0.0, 0.4, 1.3], dtype=F64)
    closed = _direct(p, t, h, mu_strip, mu_inter, theta1, theta2)
    offsets = (torch.arange(200000, dtype=F64) + 0.5) / 200000 * p
    for j, a in enumerate(theta1):
        for k, b in enumerate(theta2):
            d = torch.tensor([math.tan(a), math.tan(b), 1.0], dtype=F64)
            d = d / d.norm()
            length = torch.full_like(offsets, h / float(d[2]))
            strip = _strip_path(offsets, offsets + d[0] * length, length, p, t)
            brute = torch.exp(-mu_strip[:, None] * strip - mu_inter[:, None] * (length - strip)).mean(-1)
            torch.testing.assert_close(closed[:, j, k], brute, rtol=1e-4, atol=1e-7)
    expected = ((p - t) + t * torch.exp(-(mu_strip - mu_inter) * h)) / p * torch.exp(-mu_inter * h)
    torch.testing.assert_close(closed[:, 0, 0], expected)


@cuda
@pytest.mark.parametrize("ratio", [10, 15])
def test_analytic_grid_matches_geant4(photon_data, ratio):
    """Against the Geant4 grid sweep: the direct part within 0.015 everywhere; the grid's secondaries within 20 % at
    60 to 80 keV for tilts across the strips of 4 degrees and more (and within 0.006 absolute below 50 keV, where
    they are small); at 100 keV, above lead's K edge, they are 5 to 30 % low (K production and escape)."""
    from citrus.scatter_solver.grid_model import grid_signal_transmission

    sweep = GEANT4["grid_sweep"]
    energies = np.array(sweep["E_kev"], dtype=float)
    theta1, theta2 = np.array(sweep["theta1_deg"], dtype=float), np.array(sweep["theta2_deg"], dtype=float)
    model = grid_signal_transmission(COBRA_PITCH, COBRA_STRIP, ratio * (COBRA_PITCH - COBRA_STRIP),
                                     energies=energies, theta1=np.radians(theta1), theta2=np.radians(theta2),
                                     backend=photon_data, cache_dir=None)
    direct = np.array(sweep[f"r{ratio}_direct"], dtype=float)
    secondary = np.array(sweep[f"r{ratio}_grid"], dtype=float)
    ran = ~np.isnan(direct)
    assert np.abs(model["direct"] - direct)[ran].max() < 0.015
    tilted = (theta1 >= 4.0)[None, :, None] & ran
    middle = np.isin(energies, [60.0, 70.0, 80.0])[:, None, None] & tilted
    ratio_middle = model["secondary"][middle] / secondary[middle]
    assert 0.8 < ratio_middle.min() and ratio_middle.max() < 1.2, ratio_middle
    low = (energies < 50.0)[:, None, None] & ran
    assert np.abs(model["secondary"] - secondary)[low].max() < 0.006
    above = (energies == 100.0)[:, None, None] & tilted
    ratio_above = model["secondary"][above] / secondary[above]
    assert 0.7 < ratio_above.min() and ratio_above.max() < 0.95, ratio_above


@cuda
def test_anti_scatter_grid_options(photon_data):
    """Lines per cm and the ratio set the pitch and the height; the analytic model is the default, depends on theta2
    and adds the secondaries above the septum-shadow model past the cut-off; an aluminium interspace absorbs on the
    primary ray at low energy; a user-defined septum material works; the crossed analytic grid is the product."""
    options = dict(focal_distance_cm=150.0, backend=photon_data)
    by_ratio = anti_scatter_grid(None, COBRA_STRIP, ratio=10.0, lines_per_cm=60.0, **options)
    by_height = anti_scatter_grid(None, COBRA_STRIP, 10.0 * (COBRA_PITCH - COBRA_STRIP), pitch_cm=COBRA_PITCH,
                                  **options)
    torch.testing.assert_close(by_ratio.transmission, by_height.transmission)
    assert "analytic" in by_ratio.name and "air interspace" in by_ratio.name
    shadow = anti_scatter_grid(None, COBRA_STRIP, ratio=10.0, lines_per_cm=60.0, model="septum_shadow", **options)
    e = torch.tensor([60.0, 100.0], dtype=F64)
    zero = torch.zeros((), dtype=F64)
    past_cutoff = torch.tensor(math.radians(15.0), dtype=F64)
    assert bool((by_ratio.at(e, past_cutoff, zero) > shadow.at(e, past_cutoff, zero) + 0.02).all())
    on_axis = by_ratio.at(e, zero, zero)
    open_fraction = shadow.at(e, zero, zero)  # plus the strip tops' transmission and fluorescence
    assert bool((on_axis >= open_fraction).all()) and bool((on_axis < 1.05 * open_fraction).all())
    tilted = by_ratio.at(torch.tensor([100.0], dtype=F64), torch.tensor(0.05, dtype=F64),
                         torch.tensor([0.0, 0.8], dtype=F64))
    assert float(tilted[0, 1]) < float(tilted[0, 0])  # a longer strip path along the strips
    aluminium = anti_scatter_grid(None, COBRA_STRIP, ratio=10.0, lines_per_cm=60.0, interspace_material="aluminium",
                                  **options)
    low = torch.tensor([20.0], dtype=F64)
    assert float(aluminium.at(low, zero, zero)) < 0.9 * float(by_ratio.at(low, zero, zero))
    tungsten = Material("tungsten_alloy", 18.0, "0.95*W + 0.05*Ni")
    alloy = anti_scatter_grid(None, COBRA_STRIP, ratio=10.0, lines_per_cm=60.0, septum_material=tungsten, **options)
    assert "tungsten_alloy" in alloy.name and bool(torch.isfinite(alloy.transmission).all())
    geometry = ConeGeometry(100.0, 150.0, 5, 7, 0.3, 0.4)
    both = anti_scatter_grid(geometry, 0.02, 2.0, septa_along="both", backend=photon_data)
    rows = anti_scatter_grid(geometry, 0.02, 2.0, septa_along="rows", backend=photon_data)
    channels = anti_scatter_grid(geometry, 0.02, 2.0, backend=photon_data)
    assert both.crossed is not None and both.crossed.septa_along == "rows"
    pixels = geometry.pixel_centers(0.8, dtype=F64)
    directions = torch.randn(9, 3, generator=torch.Generator().manual_seed(3), dtype=F64)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(both.transmission_at(e, directions, pixels, geometry, 0.8),
                               channels.transmission_at(e, directions, pixels, geometry, 0.8)
                               * rows.transmission_at(e, directions, pixels, geometry, 0.8))
    for bad, match in ((dict(), "exactly one"), (dict(septum_height_cm=0.1, ratio=10.0), "exactly one"),
                       (dict(ratio=10.0, pitch_cm=0.1, lines_per_cm=10.0), "not both"),
                       (dict(ratio=10.0, lines_per_cm=60.0, model="monte_carlo"), "model")):
        with pytest.raises(ValueError, match=match):
            anti_scatter_grid(None, COBRA_STRIP, focal_distance_cm=150.0, backend=photon_data, **bad)


# ---- responses averaged over each group -----------------------------------------------------------------------------

def test_group_averaged_responses_are_the_weighted_group_means(photon_data):
    """At the group energies the averaged detector returns the source-weighted mean over each group's fine energies
    and the averaged grid the mean weighted by the source photons times the detector at normal incidence; a group
    without source photons keeps the responses at its group energy."""
    from citrus.scatter_solver.cross_sections import synthetic_library

    library = synthetic_library(num_groups=4, legendre_order=1)
    edges = torch.tensor([125.0, 95.0, 80.0, 40.0, 10.0], dtype=F64)
    fine_energy = torch.tensor([120.0, 110.0, 100.0, 90.0, 85.0, 70.0, 50.0], dtype=F64)
    fine_group = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    fine_weight = torch.tensor([0.1, 0.2, 0.1, 0.15, 0.15, 0.2, 0.1], dtype=F64)
    group_energy = torch.tensor([108.0, 87.5, 63.0, 25.0], dtype=F64)
    library = replace(library, group_edges=edges, group_energy=group_energy, fine_energy=fine_energy,
                      fine_group=fine_group, fine_weight=fine_weight,
                      fine_mu_t=library.fine_mu_t[:, :1].expand(-1, 7).clone())
    detector = detector_response("cesium_iodide", 0.06, deposition="fluorescence_escape", backend=photon_data)
    grid = anti_scatter_grid(None, 0.0036, 0.15, pitch_cm=1 / 60, focal_distance_cm=150.0, backend=photon_data,
                             model="septum_shadow")
    averaged_detector, averaged_grid = group_averaged_responses(detector, grid, library)
    theta = torch.tensor([0.0, 0.4], dtype=F64)
    theta1 = torch.tensor([0.0, 0.05, 0.3], dtype=F64)
    zero = torch.zeros(3, dtype=F64)
    got_detector = averaged_detector.at(group_energy, theta)
    got_grid = averaged_grid.at(group_energy, theta1, zero)
    for g in range(3):
        m = fine_group == g
        w = fine_weight[m]
        expected = (w[:, None] * detector.at(fine_energy[m], theta)).sum(0) / w.sum()
        torch.testing.assert_close(got_detector[g], expected)
        ww = w * detector.at(fine_energy[m], torch.zeros(1, dtype=F64))[:, 0]
        expected = (ww[:, None] * grid.at(fine_energy[m], theta1, zero)).sum(0) / ww.sum()
        torch.testing.assert_close(got_grid[g], expected)
    torch.testing.assert_close(got_detector[3], detector.at(group_energy[3:], theta)[0])
    torch.testing.assert_close(got_grid[3], grid.at(group_energy[3:], theta1, zero)[0])
    assert "group-averaged" in averaged_detector.name and "group-averaged" in averaged_grid.name
    _, constant = group_averaged_responses(detector, no_grid(), library)
    assert bool((constant.transmission == 1.0).all()) and not constant.depends_on_angles
    assert SolverConfig().group_averaged_responses is False


@cuda
def test_group_averaged_responses_remove_the_jump_at_the_lead_k_edge(photon_data):
    """Behind a lead grid, moving a group edge so that a group energy crosses lead's 88 keV K edge changes the scatter
    signal of the solver by several percent when the responses are evaluated at the group energies, and by well
    under 2 % with the group-averaged responses. The analytic grid passes more scatter than the septum shadow (its
    scatter and fluorescence reach the detector), and the default solver ignores the option."""
    from citrus.scatter_solver.scatter_solver import build_solver_library
    from citrus.utilities import wlab_phantom

    device = torch.device("cuda")
    grid = Grid((10, 15, 20), 2.0)
    geometry = ConeGeometry(100.0, 150.0, num_det_rows=24, num_det_channels=32, delta_det_row=1.5,
                            delta_det_channel=1.5)
    detector = detector_response("cesium_iodide", 0.06, deposition="fluorescence_escape", backend=photon_data)
    septa = anti_scatter_grid(geometry, COBRA_STRIP, ratio=10.0, lines_per_cm=60.0, septa_along="rows",
                              detector=detector, backend=photon_data)
    shadow = anti_scatter_grid(geometry, COBRA_STRIP, ratio=10.0, lines_per_cm=60.0, septa_along="rows",
                               model="septum_shadow", backend=photon_data)
    base = SolverConfig(num_groups=5, legendre_order=2, num_polar=4, num_azimuthal=8, cone_num_polar=8,
                        cone_num_azimuthal=16, transport_cell_cm=2.0, max_source_iterations=8, num_eta_table=501)
    rho = wlab_phantom(grid, device=device, dtype=torch.float32)
    scatter = {}
    for name, edges in (("below", (125.0, 100.0, 74.0, 50.0, 30.0, 10.0)),
                        ("above", (125.0, 100.0, 80.0, 50.0, 30.0, 10.0))):
        config = replace(base, group_edges_kev=edges)
        library = build_solver_library(config, SPECTRUM_125KVP)
        assert (float(library.group_energy[1]) < 88.0) == (name == "below")
        for averaged in (False, True):
            for model, grid_response in (("analytic", septa), ("shadow", shadow)):
                solver = ScatterSolver(library, replace(config, group_averaged_responses=averaged), geometry,
                                       detector=detector, grid_response=grid_response, device=device,
                                       dtype=torch.float32)
                result = solver.solve_view(rho, grid, 0.3)
                scatter[name, averaged, model] = float(result.scatter["signal"].sum())
                if averaged:
                    assert "group-averaged" in solver.responses()[1].name
                else:
                    assert solver.responses() == (detector, grid_response)
    for model in ("analytic", "shadow"):
        jump = abs(scatter["above", False, model] / scatter["below", False, model] - 1.0)
        averaged_jump = abs(scatter["above", True, model] / scatter["below", True, model] - 1.0)
        assert jump > 0.05 and averaged_jump < 0.02, (model, jump, averaged_jump)
    for name in ("below", "above"):
        assert scatter[name, True, "analytic"] > 1.08 * scatter[name, True, "shadow"]
    with pytest.raises(ValueError, match="group_averaged_responses"):
        SolverConfig(group_averaged_responses="yes").validate()


@cuda
def test_group_averaged_responses_in_the_synthetic_data_and_the_loop(photon_data):
    """The option reaches the synthetic data and the CITRUS loop through the solver configuration: the data of a
    grid made with it differ from those made without it, and citrus_recon runs with it."""
    pytest.importorskip("mbirtorch")
    from citrus.citrus import CitrusConfig, citrus_recon
    from citrus.scatter_solver.scatter_solver import build_solver_library
    from citrus.utilities import synthesize_wlab_dataset, wlab_geometry, wlab_grid, wlab_solver

    geometry = replace(wlab_geometry(), num_det_rows=46, num_det_channels=38, delta_det_row=1.6, delta_det_channel=1.6)
    config = SolverConfig(transport_cell_cm=2.0)
    library = build_solver_library(config, SPECTRUM_125KVP)
    septa = anti_scatter_grid(geometry, COBRA_STRIP, ratio=10.0, lines_per_cm=60.0, septa_along="rows",
                              backend=photon_data)
    data = {}
    for averaged in (False, True):
        data[averaged] = synthesize_wlab_dataset(8, 2.0, noise=False, seed=0, library=library, geometry=geometry,
                                                 config=replace(config, group_averaged_responses=averaged),
                                                 grid_response=septa)
    spr_plain, spr_averaged = data[False]["SPR"].mean(), data[True]["SPR"].mean()
    assert spr_plain != spr_averaged and abs(spr_averaged / spr_plain - 1.0) < 0.3
    solver = wlab_solver(2.0, config=replace(config, group_averaged_responses=True), library=library,
                         geometry=geometry, device="cuda", grid_response=septa)
    X, info = citrus_recon(data[True]["Y"], data[True]["open_field"], data[True]["angles"], wlab_grid(2.0), geometry,
                           solver, CitrusConfig(num_outer_iterations=2, mbir_iterations_per_outer_iteration=1))
    assert bool(torch.isfinite(X).all()) and np.isfinite(info["spr"].cpu().numpy()).all()

