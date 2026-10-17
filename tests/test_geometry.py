import math

import torch

from citrus.scatter_solver.geometry import Grid, ConeGeometry
from citrus.utilities import wlab_geometry


def test_grid_centers_follow_mbirtorch_convention():
    g = Grid(shape=(3, 4, 5), delta_voxel=0.5, voxel_row_aspect=2.0, voxel_slice_aspect=0.5, recon_slice_offset=1.0)
    x, y, z = g.axis_coordinates(dtype=torch.float64)
    assert torch.allclose(x, torch.tensor([-0.75, -0.25, 0.25, 0.75], dtype=torch.float64))
    assert torch.allclose(y, torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64))
    assert torch.allclose(z, torch.tensor([-0.5, -0.25, 0.0, 0.25, 0.5], dtype=torch.float64) + 1.0)
    c = g.center_coordinates(dtype=torch.float64)
    assert c.shape == (3, 4, 5, 3)
    assert c[1, 2, 2, 0] == x[2] and c[1, 2, 2, 1] == y[1] and c[1, 2, 2, 2] == z[2]


def test_grid_coarsen():
    g = Grid(shape=(20, 30, 40), delta_voxel=0.125)
    c = g.coarsen(10)
    assert c.shape == (2, 3, 4) and abs(c.delta_voxel - 1.25) < 1e-12
    assert torch.allclose(g.local_coordinates_of_fine_in_coarse(2, dtype=torch.float64),
                          torch.tensor([-0.5, 0.5], dtype=torch.float64))


def test_cone_geometry_frames():
    geometry = wlab_geometry()
    for angle in (0.0, 0.3, math.pi / 2, 2.0):
        source = torch.tensor(geometry.source_position(angle), dtype=torch.float64)
        # source is at (0, SID, 0) in the view frame
        sv = geometry.recon_to_view(source, angle)
        assert torch.allclose(sv, torch.tensor([0.0, 100.0, 0.0], dtype=torch.float64), atol=1e-12)
        # round trip
        assert torch.allclose(geometry.view_to_recon(sv, angle), source, atol=1e-12)
        # panel center at (0, SID-SDD, 0) in the view frame, and the central ray hits u=v=0
        pc = torch.tensor(geometry.panel_center(angle), dtype=torch.float64)
        assert torch.allclose(geometry.recon_to_view(pc, angle), torch.tensor([0.0, -50.0, 0.0], dtype=torch.float64),
                              atol=1e-12)
        u_hat, v_hat, n_hat = (torch.tensor(a, dtype=torch.float64) for a in geometry.panel_axes(angle))
        assert abs(torch.dot(u_hat, v_hat)) < 1e-12 and abs(torch.dot(u_hat, n_hat)) < 1e-12
        # u along +x_view, v along +z, normal toward the source (+y_view): u x v = -n
        assert torch.allclose(torch.cross(u_hat, v_hat, dim=0), -n_hat, atol=1e-12)
        # pixel centers lie on the panel plane and map back to their own (u, v)
        pix = geometry.pixel_centers(angle, dtype=torch.float64)
        assert pix.shape == (75, 100, 3)
        u, v = geometry.panel_uv_of_points(pix, angle)
        uu, vv = geometry.pixel_uv(dtype=torch.float64)
        assert torch.allclose(u, uu[None, :].expand(75, 100), atol=1e-10)
        assert torch.allclose(v, vv[:, None].expand(75, 100), atol=1e-10)
        # distance from source along the primary direction reaches the panel plane
        d = geometry.primary_directions(angle, dtype=torch.float64)
        t = geometry.panel_distance_along(source.expand(75, 100, 3), d, angle)
        hit = source + t[..., None] * d
        assert torch.allclose(hit, pix, atol=1e-8)


def test_pixel_uv_offsets_follow_mbirtorch():
    geometry = ConeGeometry(100.0, 150.0, num_det_rows=2, num_det_channels=3, delta_det_row=1.0, delta_det_channel=2.0,
                            det_row_offset=0.25, det_channel_offset=-0.5)
    u, v = geometry.pixel_uv(dtype=torch.float64)
    # channel = (u + off)/delta + (n-1)/2  ->  u = (c - (n-1)/2) delta - off
    assert torch.allclose(u, torch.tensor([-2.0 + 0.5, 0.5, 2.0 + 0.5], dtype=torch.float64))
    assert torch.allclose(v, torch.tensor([-0.5 - 0.25, 0.5 - 0.25], dtype=torch.float64))
