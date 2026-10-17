"""MBIRTORCH bridge (citrus/recon/mbir_bridge.py): the ConeBeamModel built from the solver's Grid / ConeGeometry
projects onto the same panel with the same pixel indexing, and the warm-started single-iteration recipe
(`mbir_iterations`) runs exactly the requested iterations."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

pytest.importorskip("mbirtorch")

from citrus.utilities import wlab_geometry  # noqa: E402
from citrus.utilities import wlab_grid, wlab_phantom  # noqa: E402
from citrus.scatter_solver.cross_sections import BONE, WATER  # noqa: E402
from citrus.scatter_solver.ray_tracing import SiddonProjectors  # noqa: E402
from citrus.recon.mbir_bridge import (fdk, forward_project, freeze_regularization,  # noqa: E402
                                               make_cone_model, mbir_iterations)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="MBIRTORCH bridge tests run on CUDA")

DEV = torch.device("cuda")
GEOM = wlab_geometry()
ANGLES = [0.0, 0.3, 1.0, 2.5, 4.0]
SIDDON = SiddonProjectors(backend="torch")


def _siddon_pixel_average(vol: torch.Tensor, grid, angle: float, n: int = 4) -> torch.Tensor:
    """Exact (Siddon, PyTorch reference) line integrals averaged over an n x n lattice of sub-pixel rays: the
    pixel-averaged line integral that MBIRTORCH's footprint projector models. [Rd, Cd]."""
    g = replace(GEOM, num_det_rows=GEOM.num_det_rows * n, num_det_channels=GEOM.num_det_channels * n,
                delta_det_row=GEOM.delta_det_row / n, delta_det_channel=GEOM.delta_det_channel / n)
    p = SIDDON.primary_project(vol[None], grid, g, angle, n_sub=1)[0]
    return p.reshape(GEOM.num_det_rows, n, GEOM.num_det_channels, n).mean(dim=(1, 3))


def _centroid(img: torch.Tensor) -> tuple[float, float, float]:
    w = img.sum()
    r = torch.arange(img.shape[0], device=img.device, dtype=img.dtype)
    c = torch.arange(img.shape[1], device=img.device, dtype=img.dtype)
    return float((img.sum(1) * r).sum() / w), float((img.sum(0) * c).sum() / w), float(w)


@pytest.fixture(scope="module", params=[1.0, 0.5], ids=["1cm", "0.5cm"])
def setup(request):
    grid = wlab_grid(request.param)
    return grid, make_cone_model(grid, GEOM, ANGLES, device=DEV)


def test_model_parameters(setup):
    grid, model = setup
    p = dict(zip(["recon_shape", "delta_voxel", "delta_det_channel", "delta_det_row", "source_iso_dist",
                  "source_detector_dist", "use_ror_mask", "sinogram_shape"],
                 model.get_params(["recon_shape", "delta_voxel", "delta_det_channel", "delta_det_row",
                                   "source_iso_dist", "source_detector_dist", "use_ror_mask", "sinogram_shape"])))
    assert tuple(p["recon_shape"]) == grid.shape and p["delta_voxel"] == grid.delta_voxel
    assert p["delta_det_channel"] == 0.4 and p["delta_det_row"] == 0.4
    assert p["source_iso_dist"] == 100.0 and p["source_detector_dist"] == 150.0
    assert p["use_ror_mask"] is False
    assert tuple(p["sinogram_shape"]) == (len(ANGLES), 75, 100)


def test_wlab_projection_matches_siddon(setup):
    """Line integrals of the WLAB attenuation map: MBIRTORCH (footprint model) vs exact pixel-averaged Siddon agree
    to ~1 % RMS (measured: <= 1.4 % at 1 cm, <= 0.25 % at 0.5 cm voxels), on the same pixels."""
    grid, model = setup
    rho = wlab_phantom(grid, device=DEV)
    X = 0.201 * rho[WATER] + 0.289 * rho[BONE]
    sino = forward_project(model, X)
    assert sino.shape == (len(ANGLES), 75, 100) and sino.device.type == "cuda"
    tol = 0.02 if grid.delta_voxel >= 1.0 else 0.005
    for i, a in enumerate(ANGLES):
        ref = _siddon_pixel_average(X, grid, a)
        rms = float((sino[i] - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
        assert rms < tol, (a, rms)


def test_bright_voxel_lands_on_same_pixel(setup):
    """A single off-center voxel: image centroids agree to < 0.15 pixel and the image sums to 1 %, at every angle;
    and both follow the analytic projection of the voxel center through ConeGeometry (angle direction: a positive
    rotation moves the image the same way in both)."""
    grid, model = setup
    vol = torch.zeros(grid.shape, device=DEV)
    idx = tuple(int(0.65 * n) for n in grid.shape)
    vol[idx] = 1.0
    sino = forward_project(model, vol)
    centers = grid.center_coordinates(DEV, torch.float64)[idx]
    cents = []
    for i, a in enumerate(ANGLES):
        rm, cm, wm = _centroid(sino[i].double())
        rr, cr, wr = _centroid(_siddon_pixel_average(vol.double(), grid, a, n=8))
        assert abs(rm - rr) < 0.15 and abs(cm - cr) < 0.15, (a, rm, rr, cm, cr)
        assert abs(wm / wr - 1) < 0.01, (a, wm, wr)
        # analytic: magnified voxel center on the panel (our ConeGeometry), in fractional pixel indices
        pv = GEOM.recon_to_view(centers[None], a)[0]
        mag = GEOM.source_detector_dist / (GEOM.source_iso_dist - float(pv[1]))
        c_an = (mag * float(pv[0]) + GEOM.det_channel_offset) / GEOM.delta_det_channel + (GEOM.num_det_channels - 1) / 2
        r_an = (mag * float(pv[2]) + GEOM.det_row_offset) / GEOM.delta_det_row + (GEOM.num_det_rows - 1) / 2
        assert abs(cm - c_an) < 0.3 and abs(rm - r_an) < 0.3, (a, cm, c_an, rm, r_an)
        cents.append((cm, cr))
    # +0.3 rad rotation: same direction and size of the channel shift in both
    dm, dr = cents[1][0] - cents[0][0], cents[1][1] - cents[0][1]
    assert np.sign(dm) == np.sign(dr) and abs(dm) > 1.0 and abs(dm - dr) < 0.2


def test_fdk_and_single_iteration_recipe():
    """FDK of a small in-field object is quantitatively right, and mbir_iterations(first_iteration=k) runs exactly ONE
    iteration at schedule position k (warm start, init untouched), whereas recon(max_iterations=1, first_iteration=k)
    runs none (the pitfall the recipe avoids)."""
    grid = wlab_grid(1.0)
    angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    model = make_cone_model(grid, GEOM, angles, device=DEV)
    x, y, z = grid.axis_coordinates(DEV)
    obj = ((x[None, :, None].abs() < 6) & (y[:, None, None].abs() < 5) & (z[None, None, :].abs() < 5)).float() * 0.2
    sino = forward_project(model, obj)
    x0 = fdk(model, sino)
    inner = ((x[None, :, None].abs() < 4) & (y[:, None, None].abs() < 3) & (z[None, None, :].abs() < 3))
    assert abs(float(x0[inner].mean()) / 0.2 - 1) < 0.05
    x0_copy = x0.clone()
    np.random.seed(0)
    r0, d0 = model.recon(sino, init_recon=x0, max_iterations=1, first_iteration=3, stop_threshold_change_pct=0,
                         logfile_path=None, print_logs=False, output_sharded=True)
    assert d0["recon_params"]["num_iterations"] == 0
    seq = model.get_params("partition_sequence")
    xk = x0
    errs = []
    for k in range(4):
        xk, d = mbir_iterations(model, sino, xk, first_iteration=k)
        assert d["recon_params"]["num_iterations"] == 1
        assert d["recon_params"]["partition_sequence"] == [seq[k]]
        errs.append(float((xk - obj).norm() / obj.norm()))
    assert torch.equal(x0, x0_copy)  # init_recon is not modified
    assert errs[-1] < errs[0] and errs[-1] < 0.1, errs
    sig = freeze_regularization(model)
    xk, _ = mbir_iterations(model, 2.0 * sino, xk, first_iteration=4)  # a different sinogram: sigmas stay frozen
    assert model.get_params("sigma_y") == sig["sigma_y"] and model.get_params("sigma_x") == sig["sigma_x"]


def test_resume_from_error_sinogram_equals_restart():
    """mbir_iterations resumed from a kept error sinogram (MBIRTORCH's _vcd_recon with init_error_sinogram, the
    preparation of `recon` replicated) runs the same iterations as the plain warm-started recipe: the same partitions
    for the same seeds, the same volume to float rounding, a checkpoint error sinogram equal to y - A x (the carried
    sinogram drifts by rounding only), and the same result when the data change between calls (e shifted by
    y_new - y_old, as the CITRUS loop does); the Hessian of a checkpoint can be passed back."""
    grid = wlab_grid(1.0)
    angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    model = make_cone_model(grid, GEOM, angles, device=DEV)
    x, y, z = grid.axis_coordinates(DEV)
    obj = ((x[None, :, None].abs() < 6) & (y[:, None, None].abs() < 5) & (z[None, None, :].abs() < 5)).float() * 0.2
    sino = forward_project(model, obj)
    sinos = [sino, sino * 1.01, sino * 1.01, sino * 0.995]  # the data change between calls
    x0 = fdk(model, sino)
    freeze_regularization(model)
    xa, xb = x0.clone(), x0.clone()
    e = None
    hess = None
    for k, s in enumerate(sinos):
        np.random.seed([0, k])
        xa, da = mbir_iterations(model, s, xa, first_iteration=k)
        if e is not None:
            e.add_(s - sinos[k - 1])  # A x unchanged: e = s - A x
        np.random.seed([0, k])
        xin = xb.clone()
        xb, db, ck = mbir_iterations(model, s, xin, first_iteration=k, error_sinogram=e, hessian=hess,
                                     return_checkpoint=True)
        assert db["recon_params"]["partition_sequence"] == da["recon_params"]["partition_sequence"]
        assert db["recon_params"]["num_iterations"] == 1
        e, hess = ck["error_sinogram"], ck["fm_hessian"]
        assert float((xb - xa).norm() / xa.norm()) < 1e-5, k
        resid = s - forward_project(model, xb)
        assert float((e - resid).norm() / resid.norm()) < 1e-4, k
    with pytest.raises(ValueError, match="init_recon"):
        mbir_iterations(model, sino, None, first_iteration=0, error_sinogram=e)


def test_scan_range_and_short_scan_detection():
    from citrus.recon.mbir_bridge import is_short_scan, scan_range
    from citrus.utilities import fan_angle, full_field_wlab_geometry, view_angles

    geom = full_field_wlab_geometry()
    full = view_angles(36, geom, 0.3, 0.3 + 2 * np.pi)
    short = view_angles(36, geom, 0.3)
    assert scan_range(full) == pytest.approx(2 * np.pi, rel=1e-12) and not is_short_scan(full)
    assert scan_range(short) == pytest.approx(np.pi + fan_angle(geom), rel=1e-12) and is_short_scan(short)
    assert scan_range([0.5]) == 0.0


def test_short_scan_fdk_parker_weights():
    """MBIRTORCH's FDK has no short-scan weighting; `fdk(..., geom=, angles=)` applies Parker weights to a short scan
    (pi + fan angle): an off-center object is reconstructed as accurately as from a full rotation, whereas the plain
    FDK of the short scan is 11-14 % off; the weights of conjugate rays sum to 1 (times Lambda / pi); a full rotation
    is left unweighted."""
    from citrus.recon.mbir_bridge import short_scan_weights
    from citrus.utilities import full_field_wlab_geometry, view_angles

    geom = full_field_wlab_geometry()
    grid = wlab_grid(1.0)
    x, y, z = grid.axis_coordinates(DEV)
    obj = (((x[None, :, None] - 6) ** 2 + (y[:, None, None] + 2) ** 2 < 16) & (z[None, None, :].abs() < 8)).float() * 0.2
    obj += ((x[None, :, None].abs() < 14) & (y[:, None, None].abs() < 9) & (z[None, None, :].abs() < 8)).float() * 0.1
    inner = (x[None, :, None].abs() < 12) & (y[:, None, None].abs() < 7) & (z[None, None, :].abs() < 4)

    def err(r):
        return float((r - obj)[inner].norm() / obj[inner].norm())

    full = view_angles(90, geom, 0.0, 2 * np.pi)
    assert short_scan_weights(geom, full) is None
    m_full = make_cone_model(grid, geom, full, device=DEV)
    e_full = err(fdk(m_full, forward_project(m_full, obj), geom=geom, angles=full))
    for start in (0.0, 1.0):
        ang = view_angles(90, geom, start)
        model = make_cone_model(grid, geom, ang, device=DEV)
        sino = forward_project(model, obj)
        e_plain, e_parker = err(fdk(model, sino)), err(fdk(model, sino, geom=geom, angles=ang))
        assert e_plain > 0.08 and e_parker < e_full + 0.01, (e_plain, e_parker, e_full)
    # redundancy: on a fine short scan the weights of the two measurements of a central-plane line sum to Lambda / pi
    ang = view_angles(2000, geom, 0.0)
    w = short_scan_weights(geom, ang)[:, 0].double().numpy()  # [V, Cd]
    lam = 2000 * (ang[1] - ang[0])
    assert w.min() >= 0 and w.max() == pytest.approx(lam / np.pi, rel=1e-6)
    assert w.sum(0).mean() * (ang[1] - ang[0]) == pytest.approx(lam, rel=1e-2)  # sum_beta w dbeta = pi (Lambda / pi)
    with pytest.warns(RuntimeWarning, match="shorter than a short scan"):
        assert short_scan_weights(geom, np.linspace(0, np.pi, 50, endpoint=False)) is None
