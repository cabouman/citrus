"""Finite-cone footprint of the scatter image (scatter_image._Footprint, applied by combined_scatter_image and
scatter_image_from_source) and the cell widths of the collapsed-cone direction sets."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from citrus.scatter_solver.angular_quadrature import Quadrature, hemisphere_quadrature
from citrus.scatter_solver.cross_sections import mu_t_volume, synthetic_library
from citrus.scatter_solver.detector import csi_response, focused_1d_grid, ideal_counting_response, no_grid
from citrus.scatter_solver.geometry import ConeGeometry, Grid
from citrus.scatter_solver.ray_tracing import SiddonProjectors
from citrus.scatter_solver.scatter_image import (box_blur_rows_cols, box_matrix, coarsen_mu_t, combined_scatter_image,
                                                 make_direction_context, panel_direction_set, panel_height_field,
                                                 scatter_image_from_source)
from citrus.scatter_solver.source import SourceModel
from citrus.scatter_solver.uncollided import uncollided_fluence
from citrus.utilities import wlab_geometry, wlab_grid

F64 = torch.float64
PROJECTION = SiddonProjectors()
KEYS = ("fluence", "planar_fluence", "signal_per_group", "signal")


def _close(got, reference, rtol=1e-6, msg=""):
    for k in KEYS:
        np.testing.assert_allclose(got[k].cpu().numpy(), reference[k].cpu().numpy(), rtol=rtol,
                                   atol=rtol * float(reference[k].abs().max()), err_msg=f"{msg} {k}")


# ---- cell widths --------------------------------------------------------------------------------------

def test_cell_widths():
    """Cone product cells: positive widths; the panel direction set always carries them; they survive reordered()
    and to(); a plain hemisphere set (no tangent axes) has none."""
    geometry = wlab_geometry()
    grid = wlab_grid(1.0)
    h = panel_direction_set(geometry, 0.0, (40, 80), max_polar_angle=grid, dtype=F64)
    assert h.cell_width is not None and bool((h.cell_width > 0).all())
    first = slice(0, 80)  # the ring nearest to the axis
    u_hat, v_hat, _ = (torch.tensor(a, dtype=F64) for a in geometry.panel_axes(0.0))
    t2 = (h.omega[first] @ u_hat) ** 2 + (h.omega[first] @ v_hat) ** 2
    assert float(t2.max()) < 1e-3
    perm = torch.randperm(h.num_directions)
    assert torch.equal(h.reordered(perm).cell_width, h.cell_width[perm])
    assert h.to(dtype=torch.float32).cell_width.dtype == torch.float32
    plain = hemisphere_quadrature((0.0, 0.0, 1.0), 4, 8, dtype=F64)
    assert plain.cell_width is None


# ---- the footprint ----------------------------------------------------------------------------------

def test_box_blur(device):
    g = torch.Generator().manual_seed(2)
    x = torch.rand(3, 2, 7, 9, generator=g, dtype=F64).to(device)
    y = box_blur_rows_cols(x, torch.tensor([0.5, 3.0, 2.5], device=device), torch.tensor([1.0, 1.0, 4.0],
                                                                                         device=device))
    assert torch.allclose(y[0], x[0])  # widths <= 1 pixel: identity
    assert torch.allclose(y[1, :, :, 1:-1], (x[1, :, :, :-2] + x[1, :, :, 1:-1] + x[1, :, :, 2:]) / 3)
    assert torch.allclose(y[1, :, :, 0], (x[1, :, :, 0] + x[1, :, :, 1]) / 2)  # renormalized at the edge
    c = box_blur_rows_cols(torch.ones(2, 1, 6, 5, dtype=F64, device=device), torch.tensor([3.7, 50.0], device=device),
                           torch.tensor([2.2, 0.1], device=device))
    assert torch.allclose(c, torch.ones_like(c))  # a constant image stays constant
    if torch.device(device).type == "cuda":  # the Triton box matrix == the PyTorch expression
        w = torch.tensor([0.3, 1.0, 2.5, 7.3, 300.0], device=device)
        assert torch.allclose(box_matrix(37, w), box_matrix(37, w.double()).float(), atol=1e-6)


def test_box_blur_rows_outer(device):
    """An image stored rows-outer ([k, Rd, G, Cd] seen as [k, G, Rd, Cd], as the scatter image projects it) is blurred
    like the same image stored contiguously, and stays rows-outer; so are the blurs along one axis in turn."""
    from citrus.scatter_solver.scatter_image import box_blur_channels, box_blur_rows

    g = torch.Generator().manual_seed(3)
    x = torch.rand(3, 4, 7, 9, generator=g, dtype=F64).to(device)
    widths_channels = torch.tensor([0.5, 2.3, 4.0], dtype=F64, device=device)
    widths_rows = torch.tensor([1.7, 0.9, 3.2], dtype=F64, device=device)
    reference = box_blur_rows_cols(x, widths_channels, widths_rows)
    rows_outer = x.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
    assert not rows_outer.is_contiguous()
    y = box_blur_rows_cols(rows_outer, widths_channels, widths_rows)
    torch.testing.assert_close(y, reference, rtol=1e-12, atol=1e-12)
    assert y.permute(0, 2, 1, 3).is_contiguous()
    torch.testing.assert_close(box_blur_rows(box_blur_channels(x, widths_channels), widths_rows), reference,
                               rtol=1e-12, atol=1e-12)


def _single_voxel_case(device):
    geometry = ConeGeometry(50.0, 60.0, 48, 48, 0.25, 0.25)
    grid = Grid((4, 4, 4), 1.0)
    library = synthetic_library(num_groups=2, legendre_order=1).to(device, F64)
    rho = torch.zeros(2, *grid.shape, device=device, dtype=F64)
    rho[0, 1, 2, 1] = 1.0  # one emitting voxel, the only attenuating one (it alone sets the footprint height)
    phi_uncollided = torch.zeros(2, *grid.shape, device=device, dtype=F64)
    phi_uncollided[:, 1, 2, 1] = 1.0
    T = torch.zeros(2, 1, *grid.shape, device=device, dtype=F64)
    return geometry, grid, library, rho, phi_uncollided, T


def test_footprint_single_voxel(device):
    """One emitting voxel, one direction along -n_hat with a cell of gnomonic widths (W_a, W_b): the footprint image
    keeps the panel sum and its spread along u / v grows by the variance of a box of width h W (h = the voxel's height
    above the panel); zero widths = no blur; a direction set without cell widths is refused."""
    geometry, grid, library, rho, phi_uncollided, T = _single_voxel_case(device)
    angle = 0.0
    _, _, n = geometry.panel_axes(angle)
    directions = -torch.tensor([n], dtype=F64)
    W = torch.tensor([[0.08, 0.05]], dtype=F64)
    q = Quadrature.from_directions(directions, torch.ones(1, dtype=F64))
    detector, grid_response = ideal_counting_response(), no_grid()
    args = (library, rho, phi_uncollided, grid, 1, T, 0, geometry, angle)

    def image(quadrature, fp):
        quadrature = quadrature.to(device)
        context = make_direction_context(quadrature, geometry, angle, detector, grid_response, library, device, F64)
        return combined_scatter_image(*args, quadrature, None, PROJECTION, detector, grid_response, context,
                                      footprint=fp)["fluence"].sum(0)

    base = image(q, False)
    with pytest.raises(ValueError, match="cell_width"):
        image(q, True)
    assert torch.allclose(image(replace(q, cell_width=torch.zeros(1, 2, dtype=F64)), True), base, rtol=1e-12, atol=0)
    f1 = image(replace(q, cell_width=W), True)
    assert float(f1.sum()) == pytest.approx(float(base.sum()), rel=1e-10)
    h = float(panel_height_field(grid, geometry, angle, device, F64)[1, 2, 1])
    uu, vv = geometry.pixel_uv(device, F64)

    def var(im, axis):
        p = im.sum(axis)
        c = uu if axis == 0 else vv
        m = (p * c).sum() / p.sum()
        return float((p * (c - m) ** 2).sum() / p.sum())

    for axis, w, d in ((0, float(W[0, 0]), geometry.delta_det_channel), (1, float(W[0, 1]), geometry.delta_det_row)):
        # the discrete kernel (box of width h W / pixel, averaged over the pixels it overlaps): its variance adds to
        # the image variance (the image stays away from the panel edges); ~ (h W)^2 / 12 for wide boxes
        N = geometry.num_det_channels if axis == 0 else geometry.num_det_rows
        B = box_matrix(N, torch.tensor([h * w / d], dtype=F64))[0, N // 2]
        j = torch.arange(N, dtype=F64) - N // 2
        kvar = float((B * j * j).sum()) * d * d
        assert ((h * w) ** 2 - d * d) / 12 < kvar < ((h * w) ** 2 + 3 * d * d) / 12
        assert var(f1, axis) - var(base, axis) == pytest.approx(kvar, rel=1e-6)


def test_footprint_consistent_on_every_path(device):
    """The footprint changes the combined image; with a zero multiple-collision source at factor 1 the first-order image
    equals the single-grid image of the first-collision source with the same footprint (scatter_image_from_source, the
    same _Footprint code); a cached context gives the same image as a fresh one."""
    from citrus.scatter_solver.angular_quadrature import num_spherical_harmonics
    from citrus.scatter_solver.uncollided import first_collision_source_at_directions

    geometry = ConeGeometry(50.0, 60.0, 8, 8, 2.0, 2.0)
    grid = Grid((4, 4, 4), 1.0)
    angle = 0.3
    library = synthetic_library(num_groups=3, legendre_order=2).to(device, F64)
    x, y, z = grid.axis_coordinates(device, F64)
    rho = torch.zeros(2, *grid.shape, device=device, dtype=F64)
    rho[0] = 0.8 + 0.05 * x[None, :, None] + 0.03 * y[:, None, None]
    rho[1, 1:3, 1:3, 1:3] = 0.4
    phi_uncollided = uncollided_fluence(library, rho, grid, geometry, angle, SourceModel(), PROJECTION)
    q = panel_direction_set(geometry, angle, (6, 12), max_polar_angle=grid, device=device, dtype=F64)
    detector, grid_response = csi_response(), focused_1d_grid(energies=library.group_energy)
    mu = mu_t_volume(library, rho)
    context = make_direction_context(q, geometry, angle, detector, grid_response, library, device, F64, chunk=5)
    for f in (1, 2):
        T = (torch.rand(3, num_spherical_harmonics(2), *grid.coarsen(f).shape, dtype=F64,
                        generator=torch.Generator().manual_seed(f)) * 0.01).to(device)
        mu_coarse_values = coarsen_mu_t(0.9 * mu, grid, grid.coarsen(f))
        args = (library, rho, phi_uncollided, grid, f, T, 2, geometry, angle, q, mu_coarse_values, PROJECTION, detector,
                grid_response)
        tot = combined_scatter_image(*args, chunk=5)
        _close(combined_scatter_image(*args, context), tot, rtol=1e-12, msg=f"f={f} cached context")
        _close(combined_scatter_image(*args, context), tot, rtol=1e-12, msg=f"f={f} cached context, second call")
        plain = combined_scatter_image(*args, chunk=5, footprint=False)
        assert float((tot["signal"] - plain["signal"]).abs().max()) > 1e-6 * float(plain["signal"].abs().max())
        assert float(tot["planar_fluence"].sum()) == pytest.approx(float(plain["planar_fluence"].sum()), rel=0.05)
        if f == 1:  # zero multiple-collision source: the total emission is the first-collision emission
            first_order0 = combined_scatter_image(library, rho, phi_uncollided, grid, 1, torch.zeros_like(T), 2,
                                                  geometry, angle,
                                                  q, mu_coarse_values,
                                                  PROJECTION, detector, grid_response, chunk=5)
            reference = scatter_image_from_source(
                lambda directions: first_collision_source_at_directions(library, rho, phi_uncollided, grid, geometry,
                                                                        angle,
                                                                        directions), grid, geometry, angle,
                q, mu_coarse_values, PROJECTION, detector, grid_response, library, chunk=5)
            _close(first_order0, reference, rtol=1e-10, msg="factor 1 vs single grid")
