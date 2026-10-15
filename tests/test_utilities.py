import pytest
import torch

from citrus.scatter_solver.geometry import Grid
from citrus.utilities import (RHO_AIR, RHO_BONE, RHO_LUNG, wlab_setup, wlab_grid, wlab_phantom)


def _mask(grid, cx, cy, wx, wy):
    x, y, z = grid.axis_coordinates(dtype=torch.float64)
    mx = (x >= cx - wx / 2) & (x < cx + wx / 2)
    my = (y >= cy - wy / 2) & (y < cy + wy / 2)
    return (my[:, None, None] & mx[None, :, None]).expand(grid.shape)


@pytest.mark.parametrize("delta", [1.0, 0.5])
def test_wlab_phantom_volumes_and_materials(delta):
    grid = wlab_grid(delta)
    assert grid.shape == (round(20 / delta), round(30 / delta), round(40 / delta))
    assert grid.extent == pytest.approx((30.0, 20.0, 40.0))
    rho = wlab_phantom(grid, dtype=torch.float64)
    assert rho.shape == (2, *grid.shape) and rho.dtype == torch.float64
    v = grid.voxel_volume
    w, b = rho[0], rho[1]
    lung, air, bone = _mask(grid, -5, 0, 6, 6), _mask(grid, 3, 0, 2, 2), _mask(grid, 7, 0, 2, 2)
    # insert volumes (cm^3) and densities
    assert (w == RHO_LUNG).sum().item() * v == pytest.approx(6 * 6 * 40)
    assert (w == RHO_AIR).sum().item() * v == pytest.approx(2 * 2 * 40)
    assert (b == RHO_BONE).sum().item() * v == pytest.approx(2 * 2 * 40)
    assert (w[lung] == RHO_LUNG).all() and (w[air] == RHO_AIR).all() and (b[bone] == RHO_BONE).all()
    # bone channel only in the bone insert; no water there
    assert (b[~bone] == 0).all() and (w[bone] == 0).all()
    # water everywhere else (the grid tiles the prism exactly)
    rest = ~(lung | air | bone)
    assert (w[rest] == 1.0).all()
    assert rest.sum().item() * v == pytest.approx((20 * 30 - 36 - 4 - 4) * 40)


def test_wlab_phantom_on_larger_grid_and_options():
    grid = Grid(shape=(24, 34, 44), delta_voxel=1.0, recon_slice_offset=1.0)  # 2 cm margin, slice offset 1 cm
    rho = wlab_phantom(grid, lung_center=(5.0, 2.0), air_center=(-4.0, -3.0), bone_center=(-10.0, 4.0))
    assert rho.dtype == torch.float32
    assert (rho[0] > 0).sum().item() == (30 * 20 - 4) * 40  # all prism voxels except bone are water-based
    assert (rho[1] > 0).sum().item() == 4 * 40
    x, y, z = grid.axis_coordinates()
    assert rho[:, :, :, (z.abs() > 20)].abs().sum() == 0
    assert rho[0, 12, 19, 22].item() == pytest.approx(RHO_LUNG)  # (x, y, z) = (2.5, 0.5, 1.5) is in the lung
    with pytest.raises(ValueError):
        wlab_phantom(Grid(shape=(20, 30, 38), delta_voxel=1.0))  # too short in z
    with pytest.raises(ValueError):
        wlab_phantom(wlab_grid(1.0), lung_center=(0.0, 0.0), air_center=(1.0, 0.0))  # overlap
    with pytest.raises(ValueError):
        wlab_grid(0.7)


def test_wlab_setup():
    grid, geom, rho = wlab_setup()
    assert grid.shape == (20, 30, 40) and grid.delta_voxel == 1.0
    assert geom.source_iso_dist == 100.0 and geom.image_shape == (75, 100)
    assert rho.shape == (2, 20, 30, 40)


def test_wlab_primary_beer_law_central_channels(device):
    from citrus.scatter_solver.ray_tracing import SiddonProjectors
    from citrus.scatter_solver.cross_sections import mu_t_volume, synthetic_library

    lib = synthetic_library(6)
    grid, geom, rho = wlab_setup(1.0, device=device, dtype=torch.float64)
    mu = mu_t_volume(lib, rho)
    tau = SiddonProjectors().primary_project(mu, grid, geom, 0.0)  # [G, 75, 100]
    pix = geom.pixel_centers(0.0, device=device, dtype=torch.float64)[:, 48:52]  # u in {-0.6..0.6}: water only
    src = torch.tensor(geom.source_position(0.0), device=device, dtype=torch.float64)
    d = pix - src
    path = 20.0 * torch.linalg.norm(d, dim=-1) / d[..., 1].abs()  # full 20 cm of water along y
    ref = lib.mu_t[0].to(device)[:, None, None] * path[None]
    torch.testing.assert_close(tau[:, :, 48:52], ref, rtol=0, atol=1e-10)
