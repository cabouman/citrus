"""Sparse-view scatter (citrus/citrus.py): the view subset, the angular interpolators of the skipped views (exactness
per order, circularity, continuity, stride 1 passthrough, order None) and the configuration validation. The CUDA runs
of citrus_recon at strides 2 and 4 are in test_citrus_recon.py (they share its tiny problem)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from citrus.citrus import (SCATTER_VIEW_INTERP_ORDERS, CitrusConfig, angular_neighbors, interpolate_scatter,
                           interpolation_weights, scatter_view_indices)

TWO_PI = 2.0 * math.pi


def _uniform(V: int) -> np.ndarray:
    return np.arange(V) * TWO_PI / V


def _poly(theta: np.ndarray, degree: int) -> np.ndarray:
    """A positive polynomial of the angle (rad) of the given degree."""
    c = [3.0, 0.7, -0.15, 0.02][: degree + 1]
    return sum(ci * theta ** i for i, ci in enumerate(c))


def _unwrap_at(theta: np.ndarray, cut: float) -> np.ndarray:
    """theta mapped to [cut, cut + 2 pi): a polynomial of the unwrapped angle tests the wrap-around neighbors."""
    return cut + np.mod(theta - cut, TWO_PI)


def _interp(f_all: np.ndarray, angles: np.ndarray, computed: np.ndarray, order) -> np.ndarray:
    s_c = torch.as_tensor(f_all[computed], dtype=torch.float64)[:, None, None].expand(-1, 2, 3).contiguous()
    out = interpolate_scatter(s_c, angles, computed, order)
    assert out.shape == (len(angles), 2, 3) and out.dtype == torch.float64
    assert bool((out == out[:, :1, :1]).all())  # per pixel: the same weights for every pixel
    return out[:, 0, 0].numpy()


# ---- the view subset ----------------------------------------------------------------------------------------

def test_scatter_view_indices():
    assert list(scatter_view_indices(8, 1)) == list(range(8))
    assert list(scatter_view_indices(36, 4)) == list(range(0, 36, 4))
    assert list(scatter_view_indices(10, 4)) == [0, 4, 8]
    assert list(scatter_view_indices(3, 5)) == [0]
    with pytest.raises(ValueError):
        scatter_view_indices(8, 0)


# ---- exactness per order --------------------------------------------------------------------------------------

@pytest.mark.parametrize("stride", [2, 3, 4])
@pytest.mark.parametrize("order,degree", [(0, 0), (1, 1), (2, 2), (3, 2)])
def test_interpolation_exact_for_polynomials_of_angle(order, degree, stride):
    """Order 0 reproduces constants, 1 linear, 2 and 3 (Catmull-Rom, uniformly spaced computed views) quadratic
    functions of the angle, including across the 2 pi wrap: the data are a polynomial of the angle unwrapped at a cut
    far from the tested views, so their neighbors (on both sides of 0 = 2 pi) see one smooth function."""
    V = 48
    angles = _uniform(V)
    computed = scatter_view_indices(V, stride)
    cut = 0.5 * math.pi  # views in [pi, 2 pi) and [0, pi/6) have all their neighbors within [cut, cut + 2 pi)
    f = _poly(_unwrap_at(angles, cut), degree)
    out = _interp(f, angles, computed, order)
    test = (angles >= math.pi) | (angles < math.pi / 6)
    np.testing.assert_allclose(out[test], f[test], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(out[computed], f[computed])  # the computed views keep their values


def test_quadratic_exact_on_nonuniform_spacing():
    """Order 2 (Lagrange at the actual angles) is exact for quadratics on any spacing; order 1 for linear functions."""
    V = 40
    angles = _uniform(V)
    computed = np.array([0, 2, 3, 7, 10, 11, 15, 18, 22, 24, 27, 31, 34, 36, 39])
    for order, degree in ((1, 1), (2, 2)):
        f = _poly(_unwrap_at(angles, 0.25 * math.pi), degree)
        out = _interp(f, angles, computed, order)
        test = (angles >= 0.75 * math.pi) & (angles < 1.75 * math.pi)
        np.testing.assert_allclose(out[test], f[test], rtol=1e-12, atol=1e-12)


def test_cubic_is_catmull_rom_not_cubic_exact():
    """Order 3 is Catmull-Rom: exact on quadratics (above), but a cubic of the angle is only approximated (the
    central-difference tangents are exact to second order), and the error vanishes as h^3 or faster."""
    errs = []
    for V in (48, 96):
        angles = _uniform(V)
        computed = scatter_view_indices(V, 4)
        f = _poly(_unwrap_at(angles, 0.5 * math.pi), 3)
        out = _interp(f, angles, computed, 3)
        test = (angles >= math.pi) & (angles < 1.75 * math.pi)
        errs.append(float(np.abs(out[test] - f[test]).max()))
    assert errs[0] > 1e-6 and errs[1] < errs[0] / 7.0, errs


def test_catmull_rom_continuous_first_derivative():
    """Order 3 has a continuous first derivative at the computed views (orders 1 and 2 do not, in general)."""
    V = 24
    angles = _uniform(V)
    computed = scatter_view_indices(V, 3)
    ca = angles[computed]
    rng = np.random.default_rng(0)
    f_c = 1.0 + rng.random(len(computed))

    def value(a: float, order: int) -> float:
        return sum(w * f_c[j] for j, w in angular_neighbors(ca, a, order))

    eps = 1e-6
    for node in (ca[0], ca[3]):  # node 0 has neighbors across the wrap
        d = {o: ((value(node + eps, o) - value(node, o)) / eps, (value(node, o) - value(node - eps, o)) / eps)
             for o in (1, 3)}
        assert abs(d[3][0] - d[3][1]) < 1e-4, d[3]
        assert abs(d[1][0] - d[1][1]) > 1e-2, d[1]
        assert abs(value(node, 3) - f_c[list(ca).index(node)]) < 1e-12


def test_nearest_view_and_neighbor_sets():
    ca = _uniform(8)  # 8 computed views, 45 degrees apart
    h = TWO_PI / 8
    assert angular_neighbors(ca, 0.3 * h, 0) == [(0, 1.0)]
    assert angular_neighbors(ca, 0.7 * h, 0) == [(1, 1.0)]
    assert angular_neighbors(ca, -0.3 * h, 0) == [(0, 1.0)]  # wraps: view 0 is also at 2 pi
    assert angular_neighbors(ca, -0.7 * h, 0) == [(7, 1.0)]
    for order, n in ((1, 2), (2, 3), (3, 4)):
        nb = angular_neighbors(ca, 0.4 * h, order)
        assert len(nb) == n and abs(sum(w for _, w in nb) - 1.0) < 1e-14
    assert sorted(j for j, _ in angular_neighbors(ca, 0.4 * h, 3)) == [0, 1, 2, 7]
    assert sorted(j for j, _ in angular_neighbors(ca, 7.4 * h, 3)) == [0, 1, 6, 7]  # across 2 pi
    assert sorted(j for j, _ in angular_neighbors(ca, 0.4 * h, 2)) == [0, 1, 7]  # centered on the nearest (0)
    assert sorted(j for j, _ in angular_neighbors(ca, 0.6 * h, 2)) == [0, 1, 2]  # centered on the nearest (1)
    # angles given in any order and any 2 pi offset
    perm = np.array([3, 0, 6, 1, 7, 2, 5, 4])
    a = 2.3
    ref = dict(angular_neighbors(ca, a, 3))
    shuffled = dict(angular_neighbors(ca[perm] - TWO_PI, a + 2 * TWO_PI, 3))
    assert {int(perm[j]): w for j, w in shuffled.items()} == pytest.approx(ref, abs=1e-12)
    # fewer computed views than the order needs: the order is lowered
    assert len(angular_neighbors(ca[:2], 1.0, 3)) == 2
    assert len(angular_neighbors(ca[:3], 1.0, 3)) == 3
    assert angular_neighbors(ca[:1], 1.0, 3) == [(0, 1.0)]


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_interpolation_is_circular(order):
    """Circular in angle: rotating the scan by one computed view rotates the interpolated sinogram."""
    V, stride = 36, 3
    angles = _uniform(V)
    computed = scatter_view_indices(V, stride)
    rng = np.random.default_rng(1)
    s_c = torch.as_tensor(1.0 + rng.random((len(computed), 4, 5)))
    out = interpolate_scatter(s_c, angles, computed, order)
    rolled = interpolate_scatter(torch.roll(s_c, 1, 0), angles, computed, order)
    torch.testing.assert_close(rolled, torch.roll(out, stride, 0), rtol=1e-12, atol=1e-12)
    assert float(out.min()) >= 0.0  # clamped (orders 2 and 3 have negative weights)


def test_stride_one_passthrough_and_order_none():
    V = 12
    angles = _uniform(V)
    s = torch.rand(V, 3, 4, dtype=torch.float64) + 0.5
    every = scatter_view_indices(V, 1)
    for order in SCATTER_VIEW_INTERP_ORDERS:
        assert torch.equal(interpolate_scatter(s, angles, every, order), s)
        np.testing.assert_array_equal(interpolation_weights(angles, every, order), np.eye(V))
    computed = scatter_view_indices(V, 3)
    skipped = np.setdiff1d(np.arange(V), computed)
    out = interpolate_scatter(s[computed], angles, computed, None)  # no previous estimate: zero
    assert torch.equal(out[computed], s[computed]) and float(out[skipped].abs().max()) == 0.0
    prev = torch.full_like(s, 7.0)
    out = interpolate_scatter(s[computed], angles, computed, None, previous=prev)  # keeps the previous estimate
    assert torch.equal(out[computed], s[computed]) and bool((out[skipped] == 7.0).all())
    assert float(prev.min()) == 7.0  # not modified
    assert not interpolation_weights(angles, computed, None)[skipped].any()


def test_weights_matrix_matches_interpolate_scatter():
    V = 20
    angles = _uniform(V) + 0.1
    computed = scatter_view_indices(V, 4)
    s_c = torch.rand(len(computed), 3, 2, dtype=torch.float64) + 1.0
    for order in (0, 1, 2, 3):
        M = interpolation_weights(angles, computed, order)
        np.testing.assert_allclose(M.sum(1), 1.0, atol=1e-14)
        ref = torch.as_tensor(M) @ s_c.reshape(len(computed), -1)
        out = interpolate_scatter(s_c, angles, computed, order, weights=torch.as_tensor(M))
        torch.testing.assert_close(out, ref.reshape(V, 3, 2).clamp_min(0.0))


def test_interpolation_argument_errors():
    angles = _uniform(8)
    with pytest.raises(ValueError):
        interpolate_scatter(torch.ones(3, 2, 2), angles, [0, 4], 3)  # one image per computed view
    with pytest.raises(ValueError):
        interpolation_weights(angles, [0, 0, 4], 3)
    with pytest.raises(ValueError):
        interpolation_weights(angles, [0, 9], 3)
    for bad in ("cubic", 4, -1, 1.5, True):
        with pytest.raises(ValueError):
            interpolation_weights(angles, [0, 4], bad)
    with pytest.raises(ValueError):
        angular_neighbors(angles[:4], 0.1, None)


# ---- configuration ----------------------------------------------------------------------------------------------

def test_config_defaults_and_validation():
    cfg = CitrusConfig()
    assert cfg.scatter_view_stride == 1 and cfg.scatter_view_interp_order == 3
    cfg.validate()
    for order in SCATTER_VIEW_INTERP_ORDERS:
        CitrusConfig(scatter_view_stride=4, scatter_view_interp_order=order).validate()
        CitrusConfig(scatter_view_stride=1, scatter_view_interp_order=order).validate()  # ignored at stride 1
    CitrusConfig(scatter_view_stride=np.int64(2)).validate()
    for stride in (0, -2, 1.5, 2.0, True, "2", None):
        with pytest.raises(ValueError, match="scatter_view_stride"):
            CitrusConfig(scatter_view_stride=stride).validate()
    for order in ("cubic", "linear", 4, -1, 2.0, True, False):
        with pytest.raises(ValueError, match="scatter_view_interp_order"):
            CitrusConfig(scatter_view_stride=2, scatter_view_interp_order=order).validate()
        with pytest.raises(ValueError, match="scatter_view_interp_order"):  # validated at stride 1 too
            CitrusConfig(scatter_view_interp_order=order).validate()
    # scatter step every n outer iterations
    assert cfg.scatter_every == 1
    for every in (1, 2, 7, np.int64(4)):
        CitrusConfig(scatter_every=every).validate()
    for every in (0, -1, 1.5, 2.0, True, "2", None):
        with pytest.raises(ValueError, match="scatter_every"):
            CitrusConfig(scatter_every=every).validate()
