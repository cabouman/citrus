import math

import numpy as np
import pytest
import torch

from citrus.scatter_solver.geometry import ConeGeometry, Grid
from citrus.scatter_solver.ray_tracing import SiddonProjectors, line_integrals

F64 = torch.float64


# ---- independent numpy oracles ------------------------------------------------------------------

def box_of(grid: Grid):
    lo = np.array(grid.lower_corner(), dtype=np.float64)
    return lo, lo + np.array(grid.extent, dtype=np.float64)


def chord(p0, p1, lo, hi) -> float:
    """Length of the segment p0 -> p1 inside the closed box [lo, hi] (scalar slab method)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    a_lo, a_hi = 0.0, 1.0
    for a in range(3):
        if d[a] == 0.0:
            if not (lo[a] <= p0[a] <= hi[a]):
                return 0.0
            continue
        t1, t2 = (lo[a] - p0[a]) / d[a], (hi[a] - p0[a]) / d[a]
        a_lo, a_hi = max(a_lo, min(t1, t2)), min(a_hi, max(t1, t2))
    return max(0.0, a_hi - a_lo) * float(np.linalg.norm(d))


def dense_integral(vol: np.ndarray, grid: Grid, p0, p1, n: int = 100_000) -> float:
    """Midpoint-rule integral of the voxel field [R, C, S] along p0 -> p1 with n samples."""
    lo, _ = box_of(grid)
    dx, dy, dz = grid.deltas
    t = (np.arange(n) + 0.5) / n
    p = np.asarray(p0)[None] + t[:, None] * (np.asarray(p1) - np.asarray(p0))[None]
    ix = np.floor((p[:, 0] - lo[0]) / dx).astype(int)
    iy = np.floor((p[:, 1] - lo[1]) / dy).astype(int)
    iz = np.floor((p[:, 2] - lo[2]) / dz).astype(int)
    R, C, S = vol.shape
    ok = (ix >= 0) & (ix < C) & (iy >= 0) & (iy < R) & (iz >= 0) & (iz < S)
    vals = np.zeros(n)
    vals[ok] = vol[iy[ok], ix[ok], iz[ok]]
    return vals.mean() * float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))


GRID = Grid(shape=(3, 4, 5), delta_voxel=1.0, voxel_row_aspect=1.5, voxel_slice_aspect=0.5, recon_slice_offset=0.3)


def small_geometry(rows=6, chans=8, d=2.0, **kw) -> ConeGeometry:
    return ConeGeometry(source_iso_dist=20.0, source_detector_dist=30.0, num_det_rows=rows,
                        num_det_channels=chans, delta_det_row=d, delta_det_channel=d, **kw)


def test_line_integrals_uniform_box_vs_chords(device):
    rng = np.random.default_rng(0)
    lo, hi = box_of(GRID)
    ext = hi - lo
    pts = []
    # random pairs: inside/inside, inside/outside, outside/outside (crossing or missing)
    for _ in range(300):
        a = lo + rng.uniform(-0.6, 1.6, 3) * ext
        b = lo + rng.uniform(-0.6, 1.6, 3) * ext
        pts.append((a, b))
    for _ in range(50):
        pts.append((lo + rng.uniform(0, 1, 3) * ext, lo + rng.uniform(0, 1, 3) * ext))
    # axis-aligned rays (some exactly on interior voxel planes), and zero-length segments
    xs = lo[0] + np.array([0.37, 1.0, 2.0, 3.5])
    for x in xs:
        pts.append((np.array([x, lo[1] - 1, 0.1]), np.array([x, hi[1] + 1, 0.1])))
        pts.append((np.array([x, 0.2, lo[2] - 3]), np.array([x, 0.2, hi[2] + 3])))
        pts.append((np.array([lo[0] - 2, 0.2, x - lo[0] + lo[2]]), np.array([hi[0] + 2, 0.2, x - lo[0] + lo[2]])))
    pts.append((np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3])))
    # rays through voxel corners (lattice point to lattice point, and beyond)
    d = np.array(GRID.deltas)
    for (i0, i1) in [((0, 0, 0), (4, 3, 5)), ((1, 0, 2), (3, 3, 0)), ((-1, -1, -1), (5, 4, 6)), ((0, 3, 0), (4, 0, 5))]:
        pts.append((lo + np.array(i0) * d, lo + np.array(i1) * d))
    starts = torch.tensor(np.array([p[0] for p in pts]), dtype=F64, device=device)
    ends = torch.tensor(np.array([p[1] for p in pts]), dtype=F64, device=device)
    c = 1.7
    vol = torch.full((1, *GRID.shape), c, dtype=F64, device=device)
    got = line_integrals(vol, starts, ends, GRID)[0].cpu().numpy()
    reference = np.array([c * chord(a, b, lo, hi) for a, b in pts])
    np.testing.assert_allclose(got, reference, rtol=0, atol=1e-10)
    # chunking does not change the answer
    got2 = line_integrals(vol, starts, ends, GRID, ray_chunk=7, element_budget=50)[0].cpu().numpy()
    np.testing.assert_allclose(got2, got, rtol=0, atol=1e-12)


def test_line_integrals_random_volume_vs_dense_sampling(device):
    rng = np.random.default_rng(1)
    vol = rng.uniform(0.0, 1.0, GRID.shape)
    lo, hi = box_of(GRID)
    ext = hi - lo
    pts = [(lo + rng.uniform(-0.5, 1.5, 3) * ext, lo + rng.uniform(-0.5, 1.5, 3) * ext) for _ in range(6)]
    pts.append((lo - 0.1, hi + 0.2))
    pts.append((np.array([0.3, lo[1] - 1, 0.4]), np.array([0.3, hi[1] + 1, 0.4])))
    starts = torch.tensor(np.array([p[0] for p in pts]), dtype=F64, device=device)
    ends = torch.tensor(np.array([p[1] for p in pts]), dtype=F64, device=device)
    got = line_integrals(torch.tensor(vol, device=device)[None], starts, ends, GRID)[0].cpu().numpy()
    reference = np.array([dense_integral(vol, GRID, a, b) for a, b in pts])
    np.testing.assert_allclose(got, reference, rtol=0, atol=2e-4)
    assert np.abs(reference).max() > 1.0


def test_float32_runs_and_is_close(device):
    vol = torch.rand(2, *GRID.shape, dtype=F64, device=device)
    s = torch.randn(64, 3, dtype=F64, device=device) * 6
    e = torch.randn(64, 3, dtype=F64, device=device) * 6
    r64 = line_integrals(vol, s, e, GRID)
    r32 = line_integrals(vol.float(), s, e, GRID)
    assert r32.dtype == torch.float32 and r32.device == vol.device
    torch.testing.assert_close(r32.double(), r64, rtol=1e-4, atol=1e-4)


# ---- projector operations -----------------------------------------------------------------------

UGRID = Grid(shape=(4, 6, 5), delta_voxel=1.0, voxel_slice_aspect=0.8)


@pytest.mark.parametrize("angle", [0.0, 0.7, 2.5])
def test_upstream_cumulative_uniform_box(device, angle):
    geometry = small_geometry()
    P = SiddonProjectors()
    mus = torch.tensor([0.2, 0.05], dtype=F64, device=device)
    mu = mus[:, None, None, None] * torch.ones(2, *UGRID.shape, dtype=F64, device=device)
    tau = P.upstream_cumulative(mu, UGRID, geometry, angle)
    assert tau.shape == (2, *UGRID.shape)
    lo, hi = box_of(UGRID)
    source = np.array(geometry.source_position(angle))
    centers = UGRID.center_coordinates(dtype=F64).reshape(-1, 3).numpy()
    reference = np.array([chord(source, c, lo, hi) for c in centers]).reshape(UGRID.shape)
    if angle == 0.0:
        # analytic: rays enter through the top face y = Y/2; distance = |r - r0| (Y/2 - y)/(SID - y)
        Y2 = UGRID.extent[1] / 2
        r = np.linalg.norm(centers - source, axis=1)
        ana = (r * (Y2 - centers[:, 1]) / (geometry.source_iso_dist - centers[:, 1])).reshape(UGRID.shape)
        np.testing.assert_allclose(reference, ana, atol=1e-12)
    for g in range(2):
        np.testing.assert_allclose(tau[g].cpu().numpy(), mus[g].item() * reference, rtol=0, atol=1e-10)


@pytest.mark.parametrize("angle", [0.0, 1.1])
def test_primary_project_uniform_box(device, angle):
    geometry = small_geometry(det_channel_offset=0.3, det_row_offset=-0.2)
    P = SiddonProjectors()
    mu = torch.full((1, *UGRID.shape), 0.3, dtype=F64, device=device)
    tau = P.primary_project(mu, UGRID, geometry, angle)
    assert tau.shape == (1, geometry.num_det_rows, geometry.num_det_channels)
    lo, hi = box_of(UGRID)
    source = np.array(geometry.source_position(angle))
    pix = geometry.pixel_centers(angle, dtype=F64).numpy()
    reference = np.array([[chord(source, pix[r, c], lo, hi) for c in range(pix.shape[1])] for r in range(pix.shape[0])])
    assert (reference > 0).any() and (reference == 0).any()
    np.testing.assert_allclose(tau[0].cpu().numpy(), 0.3 * reference, rtol=0, atol=1e-10)

    # sub-pixel sampling: -log(mean(exp(-tau))) over an n x n midpoint lattice of sub-rays
    n = 3
    tau_s = P.primary_project(mu, UGRID, geometry, angle, samples_per_pixel_axis=n)[0].cpu().numpy()
    u_hat, v_hat, _ = (np.array(a) for a in geometry.panel_axes(angle))
    off = (2 * np.arange(n) + 1) / (2 * n) - 0.5
    reference_s = np.zeros_like(reference)
    for r in range(pix.shape[0]):
        for c in range(pix.shape[1]):
            vals = [math.exp(-0.3 * chord(source, pix[r, c] + a * geometry.delta_det_row * v_hat
                                          + b * geometry.delta_det_channel * u_hat, lo, hi)) for a in off for b in off]
            reference_s[r, c] = -math.log(np.mean(vals))
    np.testing.assert_allclose(tau_s, reference_s, rtol=0, atol=1e-10)
    assert np.abs(tau_s - 0.3 * reference).max() > 1e-3  # sub-sampling actually changes edge pixels


@pytest.mark.parametrize("angle", [0.0, 0.4])
def test_downstream_cumulative_uniform_box(device, angle):
    geometry = small_geometry()
    P = SiddonProjectors()
    mu = torch.full((1, *UGRID.shape), 0.25, dtype=F64, device=device)
    _, _, n_hat = (np.array(a) for a in geometry.panel_axes(angle))
    tilt = np.array([0.3, -0.9, 0.2])
    if angle != 0.0:
        c, s = math.cos(angle), math.sin(angle)
        tilt = np.array([c * tilt[0] + s * tilt[1], -s * tilt[0] + c * tilt[1], tilt[2]])
    tilt /= np.linalg.norm(tilt)
    perp = np.array(geometry.panel_axes(angle)[0])
    omegas = np.stack([-n_hat, n_hat, tilt, perp])
    tau = P.downstream_cumulative(mu, UGRID, geometry, angle, torch.tensor(omegas, dtype=F64, device=device))
    assert tau.shape == (4, 1, *UGRID.shape)
    assert torch.isinf(tau[1]).all() and torch.isinf(tau[3]).all()
    lo, hi = box_of(UGRID)
    centers = UGRID.center_coordinates(dtype=F64).numpy()
    if angle == 0.0:
        reference0 = centers[..., 1] + UGRID.extent[1] / 2  # distance to the bottom face y = -Y/2
    else:
        reference0 = np.array([chord(c, c - 100 * n_hat, lo, hi) for c in centers.reshape(-1, 3)]).reshape(UGRID.shape)
    np.testing.assert_allclose(tau[0, 0].cpu().numpy(), 0.25 * reference0, rtol=0, atol=1e-10)
    reference2 = np.array([chord(c, c + 100 * tilt, lo, hi) for c in centers.reshape(-1, 3)]).reshape(UGRID.shape)
    np.testing.assert_allclose(tau[2, 0].cpu().numpy(), 0.25 * reference2, rtol=0, atol=1e-10)


def test_downstream_stops_at_panel_plane(device):
    # panel cutting through the box (never in CT, but the segment must end at the panel plane)
    geometry = ConeGeometry(source_iso_dist=10.0, source_detector_dist=10.0, num_det_rows=2, num_det_channels=2,
                            delta_det_row=1.0, delta_det_channel=1.0)  # panel plane y = 0
    grid = Grid(shape=(4, 2, 2), delta_voxel=1.0)  # y in [-2, 2]
    P = SiddonProjectors()
    mu = torch.ones(1, *grid.shape, dtype=F64, device=device)
    tau = P.downstream_cumulative(mu, grid, geometry, 0.0, torch.tensor([[0.0, -1.0, 0.0]], dtype=F64, device=device))
    y = grid.axis_coordinates(dtype=F64)[1]
    reference = y.clamp(min=0.0)[:, None, None].expand(grid.shape)
    torch.testing.assert_close(tau[0, 0].cpu(), reference, rtol=0, atol=1e-12)


@pytest.mark.parametrize("angle", [0.0, 0.9])
def test_oblique_project(device, angle):
    geometry = small_geometry(rows=5, chans=7, d=1.3)
    P = SiddonProjectors()
    lo, hi = box_of(UGRID)
    _, _, n_hat = (np.array(a) for a in geometry.panel_axes(angle))
    # defined in the view frame
    tilt = geometry.view_to_recon(torch.tensor([0.2, -0.95, -0.25], dtype=F64), angle).numpy()
    tilt /= np.linalg.norm(tilt)
    omegas = np.stack([-n_hat, tilt])
    directions_t = torch.tensor(omegas, dtype=F64, device=device)
    pix = geometry.pixel_centers(angle, dtype=F64).numpy()

    def line_chords(lo_, hi_, directions):
        return np.array([[chord(pix[r, c] - 200 * directions, pix[r, c] + 200 * directions, lo_, hi_)
                          for c in range(pix.shape[1])] for r in range(pix.shape[0])])

    ones = torch.ones(1, *UGRID.shape, dtype=F64, device=device)
    image = P.oblique_project(ones, UGRID, geometry, angle, directions_t)
    assert image.shape == (2, 1, 5, 7)
    for k in range(2):
        reference = line_chords(lo, hi, omegas[k])
        assert (reference > 0).any()
        np.testing.assert_allclose(image[k, 0].cpu().numpy(), reference, rtol=0, atol=1e-10)

    # single-voxel field: chord through that voxel's own box
    i, j, kk = 2, 3, 1
    one = torch.zeros(1, *UGRID.shape, dtype=F64, device=device)
    one[0, i, j, kk] = 1.0
    d = np.array(UGRID.deltas)
    vlo = lo + np.array([j, i, kk]) * d
    image1 = P.oblique_project(one, UGRID, geometry, angle, directions_t)
    for k in range(2):
        reference = line_chords(vlo, vlo + d, omegas[k])
        np.testing.assert_allclose(image1[k, 0].cpu().numpy(), reference, rtol=0, atol=1e-10)

    # per-direction fields [K, G, ...] equal the shared-field call direction by direction
    fields = torch.rand(2, 3, *UGRID.shape, dtype=F64, device=device)
    image_k = P.oblique_project(fields, UGRID, geometry, angle, directions_t)
    for k in range(2):
        torch.testing.assert_close(image_k[k],
                                   P.oblique_project(fields[k], UGRID, geometry, angle, directions_t[k:k + 1])[0])


def test_linearity_and_group_batching(device):
    geometry = small_geometry()
    P = SiddonProjectors(ray_chunk=97)
    angle = 0.6
    a = torch.rand(3, *UGRID.shape, dtype=F64, device=device)
    b = torch.rand(3, *UGRID.shape, dtype=F64, device=device)
    directions = torch.nn.functional.normalize(torch.tensor([[0.1, -1.0, 0.05], [-0.3, -0.8, 0.2]], dtype=F64,
                                                            device=device),
                                               dim=-1)
    ops = [  # (operation, index of the group axis in its output)
        (lambda v: P.upstream_cumulative(v, UGRID, geometry, angle), 0),
        (lambda v: P.primary_project(v, UGRID, geometry, angle), 0),
        (lambda v: P.downstream_cumulative(v, UGRID, geometry, angle, directions), 1),
        (lambda v: P.oblique_project(v, UGRID, geometry, angle, directions), 1),
    ]
    for op, gdim in ops:
        torch.testing.assert_close(op(2.0 * a - 0.5 * b), 2.0 * op(a) - 0.5 * op(b))
        full = op(a)
        for g in range(3):
            torch.testing.assert_close(full.select(gdim, g), op(a[g:g + 1]).select(gdim, 0))


# ---- fused Triton ray tracer (kernels/ray_tracing_kernels.py) ---------------------------------------

def _triton_ray_tracing_ok() -> bool:
    from citrus.scatter_solver.kernels import ray_tracing_kernels

    return torch.cuda.is_available() and ray_tracing_kernels.kernels_usable(torch.device("cuda"))[0]


requires_triton_kernels = pytest.mark.skipif(not _triton_ray_tracing_ok(), reason="Siddon Triton kernels unusable here")
TRITON_TOLERANCE = 1e-5  # relative to the maximum |value| of the reference result


def _assert_close_rel(got: torch.Tensor, reference: torch.Tensor, tolerance: float = TRITON_TOLERANCE) -> float:
    """Same +inf pattern, finite entries within tolerance x max|reference|; returns the relative deviation."""
    assert got.shape == reference.shape and got.dtype == reference.dtype
    m = torch.isfinite(reference)
    assert bool((torch.isfinite(got) == m).all()), "+inf pattern differs"
    if not bool(m.any()):
        return 0.0
    scale = max(float(reference[m].abs().max()), 1e-30)
    deviation = float((got[m] - reference[m]).abs().max()) / scale
    assert deviation <= tolerance, f"relative deviation {deviation:.2e} > {tolerance:.0e}"
    return deviation


def _edge_segments(grid: Grid, rng, n_random: int = 200) -> np.ndarray:
    """[N, 2, 3] segments: random (crossing / missing / inside), axis-aligned (also exactly on interior planes),
    through lattice corners and edges, starting on planes, zero-length, outside the box."""
    lo, hi = box_of(grid)
    ext = hi - lo
    d = np.array(grid.deltas)
    seg = [(lo + rng.uniform(-0.6, 1.6, 3) * ext, lo + rng.uniform(-0.6, 1.6, 3) * ext) for _ in range(n_random)]
    seg += [(lo + rng.uniform(0, 1, 3) * ext, lo + rng.uniform(0, 1, 3) * ext) for _ in range(n_random // 4)]
    mid = 0.5 * (lo + hi)
    for ax in range(3):
        for off in (0.37, 1.0, 2.0):  # 1.0 and 2.0: exactly on interior planes of another axis
            p = mid.copy()
            p[(ax + 1) % 3] = lo[(ax + 1) % 3] + off * d[(ax + 1) % 3]
            a, b = p.copy(), p.copy()
            a[ax], b[ax] = lo[ax] - 1.0, hi[ax] + 1.0
            seg += [(a, b), (b, a)]
    for i0, i1 in [((0, 0, 0), (4, 3, 5)), ((1, 0, 2), (3, 3, 0)), ((-1, -1, -1), (5, 4, 6)), ((0, 3, 0), (4, 0, 5)),
                   ((2, 1, 1), (2, 3, 4)), ((1, 1, 1), (3, 3, 1))]:
        seg.append((lo + np.array(i0) * d, lo + np.array(i1) * d))
    seg.append((lo + np.array([1.0, 0.5, 2.0]) * d, lo + np.array([3.0, 2.5, 0.25]) * d))  # starts on x and z planes
    seg.append((mid, mid))  # zero length
    seg.append((hi + 1.0, hi + 2.0))  # outside
    seg.append((np.array([lo[0] - 1, mid[1], hi[2] + 0.5]), np.array([hi[0] + 1, mid[1], hi[2] + 0.5])))  # misses
    return np.array([[a, b] for a, b in seg])


@requires_triton_kernels
@pytest.mark.parametrize("G", [1, 6])
def test_triton_line_integrals_match_torch(G):
    """line_integrals_triton == siddon.line_integrals (float32) on edge-case segments: random volumes, a stacked
    volume index, non-contiguous volumes / end points, a shared (stride-0) start."""
    from citrus.scatter_solver.kernels.ray_tracing_kernels import line_integrals_triton

    device = torch.device("cuda")
    rng = np.random.default_rng(10 + G)
    segs = torch.tensor(_edge_segments(GRID, rng), dtype=torch.float32, device=device)
    s, e = segs[:, 0], segs[:, 1]  # non-contiguous row views
    assert not s.is_contiguous()
    vol = torch.rand(G, *GRID.shape, device=device)
    _assert_close_rel(line_integrals_triton(vol, s, e, GRID), line_integrals(vol, s, e, GRID))
    # float64 truth: the kernel is within float32 rounding of it
    r64 = line_integrals(vol.double(), s.double(), e.double(), GRID)
    _assert_close_rel(line_integrals_triton(vol, s, e, GRID).double(), r64)
    # non-contiguous volume
    vol_nc = vol.permute(0, 3, 1, 2).contiguous().permute(0, 2, 3, 1)
    assert not vol_nc.is_contiguous()
    _assert_close_rel(line_integrals_triton(vol_nc, s, e, GRID), line_integrals(vol, s, e, GRID))
    # stacked volumes with a per-ray index
    K = 3
    vols = torch.rand(K, G, *GRID.shape, device=device)
    kk = torch.tensor(rng.integers(0, K, s.shape[0]), device=device)
    _assert_close_rel(line_integrals_triton(vols, s, e, GRID, volume_index=kk),
                      line_integrals(vols, s, e, GRID, volume_index=kk))
    # a stride-0 start (one shared point) and an empty ray set
    source = torch.tensor([0.3, 9.0, -0.4], device=device).expand_as(e)
    _assert_close_rel(line_integrals_triton(vol, source, e, GRID), line_integrals(vol, source, e, GRID))
    assert line_integrals_triton(vol, s[:0], e[:0], GRID).shape == (G, 0)
    with pytest.raises(ValueError):
        line_integrals_triton(vols, s, e, GRID)  # 5-D volume without volume_index
    with pytest.raises(ValueError):
        line_integrals_triton(vol.double(), s, e, GRID)


def _triton_directions(geometry: ConeGeometry, angle: float, device) -> torch.Tensor:
    """Directions for the downstream / oblique checks (recon frame): tilted ones (view frame), exact coordinate
    axes, lattice diagonals, one pointing away from the panel, -n_hat, u_hat (in the panel plane) and +z."""
    u_hat, _, n_hat = (np.array(a) for a in geometry.panel_axes(angle))
    view = np.array([[0.2, -1.0, 0.1], [-0.3, -0.8, 0.25], [0.6, -0.5, -0.4], [0.0, -1.0, 0.0], [1.0, -1.0, 1.0],
                     [0.0, -1.0, 1.0], [0.05, 1.0, 0.0]])
    view = view / np.linalg.norm(view, axis=1, keepdims=True)
    record = geometry.view_to_recon(torch.tensor(view), angle).numpy()
    extra = [-n_hat, np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), u_hat, np.array([0.0, 0.0, 1.0])]
    return torch.tensor(np.concatenate([record, np.stack(extra)]), dtype=torch.float32, device=device)


@requires_triton_kernels
@pytest.mark.parametrize("G", [1, 6])
@pytest.mark.parametrize("angle", [0.0, 0.7])
def test_triton_projector_operations_match_torch(G, angle):
    """SiddonProjectors(backend="triton") == the PyTorch reference path for all four operations (shared and
    per-direction oblique fields, non-contiguous inputs, samples_per_pixel_axis > 1, +inf directions)."""
    device = torch.device("cuda")
    geometry = small_geometry(det_channel_offset=0.3, det_row_offset=-0.2)
    T = SiddonProjectors(backend="torch")
    assert T.backend == "torch"
    generator = torch.Generator(device=device).manual_seed(5)
    mu = 0.3 * torch.rand(G, *UGRID.shape, device=device, generator=generator)
    directions = _triton_directions(geometry, angle, device)
    K = directions.shape[0]
    reference_down = T.downstream_cumulative(mu, UGRID, geometry, angle, directions)
    assert torch.isinf(reference_down[K - 2:]).all() and torch.isinf(reference_down[6]).all()  # u_hat, +z; the away one
    assert torch.isfinite(reference_down[:6]).all()
    mu_nc = mu.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
    directions_nc = torch.cat([directions, directions], dim=1)[:, :3]
    assert not mu_nc.is_contiguous() and not directions_nc.is_contiguous()
    P = SiddonProjectors(backend="triton")
    assert P._use_triton(mu)
    _assert_close_rel(P.downstream_cumulative(mu, UGRID, geometry, angle, directions), reference_down)
    _assert_close_rel(P.downstream_cumulative(mu_nc, UGRID, geometry, angle, directions_nc), reference_down)
    _assert_close_rel(P.upstream_cumulative(mu, UGRID, geometry, angle), T.upstream_cumulative(mu, UGRID, geometry,
                                                                                               angle))
    _assert_close_rel(P.upstream_cumulative(mu_nc, UGRID, geometry, angle), T.upstream_cumulative(mu, UGRID, geometry,
                                                                                                  angle))
    _assert_close_rel(P.primary_project(mu, UGRID, geometry, angle), T.primary_project(mu, UGRID, geometry, angle))
    _assert_close_rel(P.primary_project(mu, UGRID, geometry, angle, samples_per_pixel_axis=3),
                      T.primary_project(mu, UGRID, geometry, angle, samples_per_pixel_axis=3))
    fields = torch.rand(K, G, *UGRID.shape, device=device, generator=generator)
    _assert_close_rel(P.oblique_project(fields[0], UGRID, geometry, angle, directions),
                      T.oblique_project(fields[0], UGRID, geometry, angle, directions))
    _assert_close_rel(P.oblique_project(fields, UGRID, geometry, angle, directions),
                      T.oblique_project(fields, UGRID, geometry, angle, directions))
    pad = torch.zeros(K, G, *UGRID.shape[:2], UGRID.shape[2] + 3, device=device)
    pad[..., 1:-2] = fields
    f_nc = pad[..., 1:-2]  # [K, G, ...] slice of a padded buffer
    assert not f_nc.is_contiguous()
    _assert_close_rel(P.oblique_project(f_nc, UGRID, geometry, angle, directions_nc),
                      T.oblique_project(fields, UGRID, geometry, angle, directions))
    # the generic line_integrals method dispatches too
    s = torch.randn(50, 3, device=device, generator=generator) * 4
    e = torch.randn(50, 3, device=device, generator=generator) * 4
    _assert_close_rel(P.line_integrals(mu, s, e, UGRID), T.line_integrals(mu, s, e, UGRID))


@requires_triton_kernels
def test_triton_downstream_wlab_grid_vs_float64():
    """WLAB 2 cm grid, real geometry: the shared-sequence downstream kernel and the float32 PyTorch path are both
    within float32 rounding of the float64 PyTorch path."""
    from citrus.utilities import wlab_setup

    device = torch.device("cuda")
    grid, geometry, rho = wlab_setup(2.0, device=device)
    mu = torch.stack([0.2 * rho[0] + 0.5 * rho[1], 0.3 * rho[0] + 0.9 * rho[1]])
    generator = torch.Generator().manual_seed(2)
    directions = torch.nn.functional.normalize(torch.randn(16, 3, generator=generator) * torch.tensor([0.5, 0.3, 0.5])
                                               - torch.tensor([0.0, 1.0, 0.0]), dim=-1).to(device)
    r64 = SiddonProjectors(backend="torch").downstream_cumulative(mu.double(), grid, geometry, 0.0, directions.double())
    triton_result = SiddonProjectors(backend="triton").downstream_cumulative(mu, grid, geometry, 0.0, directions)
    _assert_close_rel(triton_result.double(), r64)
    torch_result = SiddonProjectors(backend="torch").downstream_cumulative(mu, grid, geometry, 0.0, directions)
    _assert_close_rel(torch_result.double(), r64)


@requires_triton_kernels
def test_triton_backend_selection():
    from citrus.scatter_solver.kernels import ray_tracing_kernels

    device = torch.device("cuda")
    geometry = small_geometry()
    mu = torch.rand(2, *UGRID.shape, device=device)
    directions = torch.nn.functional.normalize(torch.tensor([[0.1, -1.0, 0.2]], device=device), dim=-1)
    auto = SiddonProjectors(backend="auto")
    assert auto._use_triton(mu) and not auto._use_triton(mu.double()) and not auto._use_triton(mu.cpu())
    ok, why = auto.kernel_status(device)
    assert ok, why
    assert not SiddonProjectors(backend="torch")._use_triton(mu)
    # float64 / CPU on a triton instance run the PyTorch path, bit for bit
    tri = SiddonProjectors(backend="triton")
    r64 = tri.downstream_cumulative(mu.double(), UGRID, geometry, 0.0, directions.double())
    assert r64.dtype == torch.float64
    assert torch.equal(r64, SiddonProjectors(backend="torch").downstream_cumulative(mu.double(), UGRID, geometry, 0.0,
                                                                                    directions.double()))
    assert torch.equal(tri.oblique_project(mu.cpu(), UGRID, geometry, 0.0, directions.cpu()),
                       SiddonProjectors(backend="torch").oblique_project(mu.cpu(), UGRID, geometry, 0.0,
                                                                         directions.cpu()))
    assert ray_tracing_kernels.use_kernels(mu, "auto") and not ray_tracing_kernels.use_kernels(mu, "torch")
    with pytest.raises(ValueError):
        SiddonProjectors(backend="cuda")
    with pytest.raises(ValueError):
        ray_tracing_kernels.use_kernels(mu, "cuda")


@requires_triton_kernels
def test_default_backend_is_fused_on_cuda_float32():
    """SiddonProjectors() (and ScatterSolver's projectors=None) is backend "auto": the fused kernels for float32 CUDA
    tensors, the PyTorch path (bit for bit the backend="torch" reference) for CPU and float64 tensors."""
    from citrus.scatter_solver.config import SolverConfig
    from citrus.scatter_solver.detector import csi_response, no_grid
    from citrus.scatter_solver.scatter_solver import ScatterSolver
    from citrus.scatter_solver.cross_sections import synthetic_library

    device = torch.device("cuda")
    P = SiddonProjectors()
    assert P.backend == "auto"
    ok, why = P.kernel_status(device)
    assert ok, why
    assert P.kernel_status(torch.device("cpu"))[0] is False
    mu = torch.rand(2, *UGRID.shape, device=device)
    assert P._use_triton(mu)
    assert not P._use_triton(mu.double()) and not P._use_triton(mu.cpu())
    geometry = small_geometry()
    directions = torch.nn.functional.normalize(torch.tensor([[0.1, -1.0, 0.2], [-0.2, -0.9, 0.1]], device=device),
                                               dim=-1)
    reference = SiddonProjectors(backend="torch")
    assert torch.equal(P.downstream_cumulative(mu.double(), UGRID, geometry, 0.3, directions.double()),
                       reference.downstream_cumulative(mu.double(), UGRID, geometry, 0.3, directions.double()))
    assert torch.equal(P.upstream_cumulative(mu.cpu(), UGRID, geometry, 0.3),
                       reference.upstream_cumulative(mu.cpu(), UGRID, geometry, 0.3))
    _assert_close_rel(P.downstream_cumulative(mu, UGRID, geometry, 0.3, directions),
                      reference.downstream_cumulative(mu, UGRID, geometry, 0.3, directions))
    config = SolverConfig(num_groups=2, legendre_order=1)
    solver = ScatterSolver(synthetic_library(2, 5), config, geometry, detector=csi_response(), grid_response=no_grid(),
                           device=device, dtype=torch.float32)
    assert isinstance(solver.projectors, SiddonProjectors) and solver.projectors.backend == "auto"
    with pytest.raises(TypeError):
        ScatterSolver(synthetic_library(2, 5), config, geometry, device=device)


def test_triton_ray_tracing_kill_switch(monkeypatch):
    """CITRUS_DISABLE_TRITON=1 (read by the shared probe) sends "auto" to the PyTorch path and makes "triton" raise."""
    from citrus.scatter_solver.kernels import triton_utilities
    from citrus.scatter_solver.kernels import ray_tracing_kernels

    monkeypatch.setenv(triton_utilities.DISABLE_ENV_VAR, "1")
    monkeypatch.setattr(triton_utilities, "_PROBE", None)
    monkeypatch.setattr(ray_tracing_kernels, "_SELF_CHECK", {})
    ok, why = ray_tracing_kernels.kernels_usable(torch.device("cuda"))
    assert not ok and triton_utilities.DISABLE_ENV_VAR in why
    with pytest.raises(RuntimeError):
        SiddonProjectors(backend="triton")
    if torch.cuda.is_available():
        x = torch.rand(2, *UGRID.shape, device="cuda")
        P = SiddonProjectors(backend="auto")
        assert not P._use_triton(x)
        assert P.kernel_status(x.device) == (False, why)
        assert not ray_tracing_kernels.use_kernels(x, "auto")
        with pytest.raises(RuntimeError):
            ray_tracing_kernels.use_kernels(x, "triton")
        geometry = small_geometry()
        assert torch.equal(P.upstream_cumulative(x, UGRID, geometry, 0.3),
                           SiddonProjectors(backend="torch").upstream_cumulative(x, UGRID, geometry, 0.3))
    # monkeypatch restores triton_utilities._PROBE to its previous value (the real probe result) afterwards
