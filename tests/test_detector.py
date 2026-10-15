import math

import numpy as np
import pytest
import torch

from citrus.scatter_solver.detector import (DetectorResponse, csi_mass_attenuation, csi_response, energy_weighted_response,
                                  focused_1d_grid, grid_angles, ideal_counting_response, lead_mass_attenuation,
                                  no_grid)
from citrus.scatter_solver.geometry import ConeGeometry

F64 = torch.float64
E_G = torch.tensor([110.0, 70.0, 45.0, 34.0, 25.0], dtype=F64)


def test_attenuation_tables_have_edges():
    eps = 1e-2
    for e in (33.17, 35.98):
        assert csi_mass_attenuation([e + eps])[0] > 1.5 * csi_mass_attenuation([e - eps])[0]
    assert lead_mass_attenuation([88.0 + eps])[0] > 3 * lead_mass_attenuation([88.0 - eps])[0]
    e = np.array([40.0, 60.0, 80.0, 100.0, 150.0])
    assert np.all(np.diff(csi_mass_attenuation(e)) < 0)
    # spot values (XCOM, ~5 %)
    assert csi_mass_attenuation([60.0])[0] == pytest.approx(7.95, rel=0.05)
    assert lead_mass_attenuation([100.0])[0] == pytest.approx(5.55, rel=0.05)


def test_csi_limits_and_angle(device):
    th = torch.tensor([0.0, 0.3, 0.7, 1.2], dtype=F64, device=device)
    thick = csi_response(thickness_cm=50.0)
    np.testing.assert_allclose(thick.at(E_G, th).cpu().numpy(), E_G[:, None].expand(5, 4).numpy(), rtol=1e-9)
    zero = csi_response(thickness_cm=0.0)
    assert zero.at(E_G, th).abs().max().item() == 0.0
    d = csi_response()
    r = d.at(E_G, th)  # [G, 4]
    assert r.shape == (5, 4) and r.device.type == device.type
    assert torch.all(r[:, 1:] > r[:, :-1])  # thicker effective path at oblique incidence
    assert torch.all(r < E_G.to(device)[:, None]) and torch.all(r > 0)
    # normal incidence against the formula with the table's mu
    mu = torch.as_tensor(csi_mass_attenuation(E_G.numpy()))
    ref = E_G * (1 - torch.exp(-mu * 4.51 * 0.06))
    np.testing.assert_allclose(r[:, 0].cpu().numpy(), ref.numpy(), rtol=2e-3)  # 0.5 keV node spacing


def test_detector_interpolation_reproduces_nodes(device):
    d = csi_response(energies=E_G)
    th_nodes = d.theta_rad.to(device)
    np.testing.assert_allclose(d.at(d.energy_kev, th_nodes).cpu().numpy(), d.response.numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(d.at(E_G, th_nodes).cpu().numpy(), d.response.flip(0).numpy(), rtol=1e-12)
    # bilinear between nodes; negative angles use |theta|
    th = torch.tensor([-0.25, 0.25], dtype=F64, device=device)
    v = d.at(E_G, th)
    np.testing.assert_allclose(v[:, 0].cpu().numpy(), v[:, 1].cpu().numpy())
    # ideal responses
    th = torch.linspace(0, 1.5, 7, dtype=F64, device=device)
    assert torch.all(ideal_counting_response().at(E_G, th) == 1)
    np.testing.assert_allclose(energy_weighted_response().at(E_G, th).cpu().numpy(),
                               E_G[:, None].expand(5, 7).numpy(), rtol=1e-12)
    with pytest.raises(ValueError):
        DetectorResponse(torch.zeros(2), torch.zeros(3), torch.zeros(3, 2))


def test_grid_transmission(device):
    p, t, h = 1 / 60, 0.0036, 0.15
    g = focused_1d_grid()
    mu = torch.as_tensor(lead_mass_attenuation(E_G.numpy()))
    zero = torch.zeros((), dtype=F64, device=device)
    t0 = g.at(E_G, zero, zero).cpu()
    expected = (p - t) / p + t / p * torch.exp(-mu * 11.35 * h)
    np.testing.assert_allclose(t0.numpy(), expected.numpy(), rtol=2e-3)  # energy nodes every 0.5 keV
    th1 = torch.tensor([0.0, 0.01, 0.03, 0.06, 0.2, 0.6], dtype=F64, device=device)
    tr = g.at(E_G, th1, torch.zeros_like(th1))
    assert torch.all(tr[:, 1:] < tr[:, :-1])
    assert torch.all(tr > 0) and torch.all(tr <= 1)
    np.testing.assert_allclose(g.at(E_G, -th1, 0 * th1).cpu().numpy(), tr.cpu().numpy())
    # independent of theta2
    np.testing.assert_allclose(g.at(E_G, th1, torch.full_like(th1, 0.4)).cpu().numpy(), tr.cpu().numpy())
    # past the cut-off atan((p - t)/h) only lead leakage is left
    cut = math.atan((p - t) / h)
    assert torch.all(g.at(E_G, torch.tensor(cut * 1.01, dtype=F64, device=device), zero)[E_G < 40] < 1e-3)
    ng = no_grid()
    assert torch.all(ng.at(E_G, th1, th1) == 1)


def test_grid_angles():
    geom = ConeGeometry(100.0, 150.0, 5, 7, 4.0, 4.0, det_row_offset=1.0)
    for angle in (0.0, 0.9):
        pix = geom.pixel_centers(angle, dtype=F64)
        prim = geom.primary_directions(angle, dtype=F64)
        th1, th2 = grid_angles(prim, pix, geom, angle)
        assert th1.shape == (5, 7)
        assert th1.abs().max() < 1e-12
        u, v = geom.pixel_uv(dtype=F64)
        np.testing.assert_allclose(th2.numpy(), torch.atan(-u / 150.0)[None].expand(5, 7).numpy(), atol=1e-12)
        # a direction tilted in the v-source plane by 0.1 rad relative to the primary at every pixel
        u_hat, v_hat, n_hat = (torch.tensor(a, dtype=F64) for a in geom.panel_axes(angle))
        omegas = torch.stack([-n_hat, -(n_hat * math.cos(0.1) + v_hat * math.sin(0.1))])  # [2, 3]
        t1, t2 = grid_angles(omegas, pix, geom, angle)
        assert t1.shape == (2, 5, 7)
        _, vv = geom.pixel_uv(dtype=F64)
        np.testing.assert_allclose(t1[0].numpy(), (-torch.atan2(-vv, torch.tensor(150.0, dtype=F64)))[:, None].expand(5, 7).numpy(), atol=1e-12)
        np.testing.assert_allclose((t1[1] - t1[0]).numpy(), 0.1, atol=1e-12)
        np.testing.assert_allclose(t2.numpy(), 0.0, atol=1e-12)
