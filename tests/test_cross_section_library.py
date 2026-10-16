"""Tests of the multigroup cross-section library builder (citrus/scatter_solver/cross_section_library.py).

Skipped when XrayPhysics is not importable. The shared fixture uses the production energy grid (0.5 keV) and a
1001-point eta table; the full-resolution default config build is gated behind CITRUS_SLOW_TESTS=1.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

from citrus.scatter_solver.config import SolverConfig
from citrus.scatter_solver.cross_sections import BONE, MATERIALS, NOMINAL_DENSITIES, WATER, MultigroupLibrary
from citrus.scatter_solver import cross_section_library as xb

try:
    BACKEND = xb.resolve_backend()
except RuntimeError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"XrayPhysics unavailable ({exc})", allow_module_level=True)

G = 6
L = 7


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(x), axis=-1))


def _group_avg(info: xb.BuildInfo, kind: str, material: str) -> np.ndarray:
    """Independent group-weighted elemental-mixture cross section [G] from the backend."""
    comp = info.composition[material]
    E = info.fine_energy
    s = sum(w * BACKEND.sigma(kind, Z, E) for Z, w in comp.items())
    ng = int(info.fine_group.max()) + 1
    return np.array([np.sum(info.fine_weight[info.fine_group == g] * s[info.fine_group == g]) for g in range(ng)])


@pytest.fixture(scope="module")
def built():
    cfg = SolverConfig(num_groups=G, legendre_order=L, num_eta_table=1001)
    return xb.compute_library(cfg, None, BACKEND, de_kev=0.5)



# ---- (a) XCOM totals ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "material,energy,xcom,tol",
    [("water", 40.0, 0.2683, 0.02), ("water", 60.0, 0.2059, 0.02), ("water", 80.0, 0.1837, 0.02),
     ("water", 100.0, 0.1707, 0.02), ("bone", 60.0, 0.3148, 0.03), ("bone", 100.0, 0.1855, 0.03)],
)
def test_mu_t_vs_xcom(material, energy, xcom, tol):
    cfg = SolverConfig(num_groups=1, group_edges_kev=(energy + 0.25, energy - 0.25), legendre_order=1,
                       num_eta_table=11)
    lib, _ = xb.compute_library(cfg, None, BACKEND)
    m = lib.materials.index(material)
    val = float(lib.mu_t[m, 0])
    assert abs(val / xcom - 1.0) < tol, f"{material} {energy} keV: {val:.5f} vs XCOM {xcom}"


# ---- (b) kinematic bookkeeping conserves the scattering cross section ---------------------------

def test_row_sums_conserve_scatter(built):
    lib, info = built
    for mi, mat in enumerate(lib.materials):
        ref = _group_avg(info, "cs", mat) + _group_avg(info, "rs", mat)
        rows = lib.mu_s_l[mi, :, :, 0].sum(dim=1).numpy()
        below = info.compton_below_cutoff[mi]
        np.testing.assert_allclose(rows + below, ref, rtol=1e-3)
        assert np.all(below >= 0)
        # only the lowest group can backscatter below the 10 keV cutoff (E < ~10.4 keV)
        assert np.all(below[:-1] == 0)
    # below-cutoff loss is small: only E in [10, 10.4] keV contributes
    assert np.all(info.compton_below_cutoff[:, -1] < 0.05 * info.sigma_cs_group[:, -1])


def test_mu_t_is_sum_of_components(built):
    lib, info = built
    for mi, mat in enumerate(lib.materials):
        comp = _group_avg(info, "pe", mat) + _group_avg(info, "cs", mat) + _group_avg(info, "rs", mat)
        np.testing.assert_allclose(lib.mu_t[mi].numpy(), comp, rtol=2e-3)
        np.testing.assert_allclose(lib.mu_pe[mi].numpy(), _group_avg(info, "pe", mat), rtol=1e-9)


# ---- (c) Rayleigh l = 0 moment ----------------------------------------------------------------------

def test_rayleigh_l0_equals_sigma_rs(built):
    lib, info = built
    for mi, mat in enumerate(lib.materials):
        np.testing.assert_allclose(info.rayleigh_l[mi, :, 0], _group_avg(info, "rs", mat), rtol=1e-6)
    # diagonal minus Rayleigh = within-group Compton >= 0
    diag = np.stack([lib.mu_s_l[:, g, g, 0].numpy() for g in range(G)], axis=1)
    assert np.all(diag - info.rayleigh_l[..., 0] >= 0)


def test_backend_pdf_normalization(built):
    _, info = built
    if info.pdf_norm_max_rel_dev is None:
        pytest.skip("backend provides no own normalization factor")
    # backend's own sphere integral of the angular shapes agrees with our q-quadrature
    assert info.pdf_norm_max_rel_dev < 2e-3


# ---- (d) Legendre reconstruction vs direct table ---------------------------------------------------

_DIAG_REASON = (
    "known: within-group pairs contain the Rayleigh forward peak (and a sharp kinematic cut of the "
    "within-group Compton ramp); L=7 Legendre reconstruction has ~24-34% relative L2 error "
    "(Compton-only part 4-7%), see docs/source/design_notes.rst"
)


def _rel_l2(lib: MultigroupLibrary, m: int, gi: int, go: int, Lr: int) -> float:
    eta = lib.eta_grid.numpy()
    P = xb.legendre_table(eta, Lr)
    mom = lib.mu_s_l[m, gi, go, : Lr + 1].numpy()
    tab = lib.mu_s_eta[m, gi, go].numpy()
    rec = ((2 * np.arange(Lr + 1) + 1) / (4 * math.pi) * mom) @ P
    return math.sqrt(_trapz((rec - tab) ** 2, eta) / _trapz(tab**2, eta))


@pytest.mark.parametrize(
    "gi,go",
    [pytest.param(gi, go, marks=pytest.mark.xfail(strict=True, reason=_DIAG_REASON)) if gi == go else (gi, go)
     for gi in range(G) for go in range(gi, G)],
)
def test_legendre_reconstruction_water(built, gi, go):
    lib, _ = built
    if float(lib.mu_s_l[WATER, gi, go, 0]) <= 0.0:
        pytest.skip("no transfer for this pair")
    err = _rel_l2(lib, WATER, gi, go, lib.legendre_order)
    assert err < 0.05, f"water {gi}->{go}: relative L2 error {err:.4f} at L={lib.legendre_order}"


def test_table_l0_consistent_with_moments(built):
    lib, _ = built
    eta = lib.eta_grid.numpy()
    tab0 = 2 * math.pi * np.sum(0.5 * (lib.mu_s_eta[..., 1:] + lib.mu_s_eta[..., :-1]).numpy() * np.diff(eta), axis=-1)
    mom0 = lib.mu_s_l[..., 0].numpy()
    big = mom0 > 1e-3 * mom0.max()
    assert np.max(np.abs(tab0[big] / mom0[big] - 1)) < 5e-3


# ---- (e) positivity and no upscatter ---------------------------------------------------------------

def test_nonnegative_and_no_upscatter(built):
    for lib, _ in (built,):
        assert torch.all(lib.mu_s_l[..., 0] >= 0)
        assert torch.all(lib.mu_s_eta >= 0)
        for gi in range(G):
            for go in range(gi):
                assert torch.all(lib.mu_s_l[:, gi, go] == 0)
                assert torch.all(lib.mu_s_eta[:, gi, go] == 0)


# ---- (f) groups and spectrum ------------------------------------------------------------------------

def test_groups_and_spectrum(built):
    for lib, _ in (built,):
        e = lib.group_edges
        assert e.shape == (G + 1,)
        assert torch.all(e[1:] < e[:-1])
        assert abs(float(e[0]) - 125.0) < 1e-9 and abs(float(e[-1]) - 10.0) < 1e-9
        assert abs(float(lib.spectrum_fraction.sum()) - 1.0) < 1e-12
        assert torch.all((lib.group_energy < e[:-1]) & (lib.group_energy > e[1:]))
        for t in (lib.group_edges, lib.mu_t, lib.mu_pe, lib.mu_s_l, lib.mu_s_eta, lib.eta_grid):
            assert t.dtype == torch.float64 and t.device.type == "cpu"
    lib, _ = built  # equal fluence
    np.testing.assert_allclose(lib.spectrum_fraction.numpy(), np.full(G, 1.0 / G), atol=1e-4)
    assert lib.mu_s_l.shape == (7, G, G, L + 1)
    assert lib.mu_s_eta.shape == (7, G, G, 1001)
    assert lib.materials == MATERIALS and lib.num_materials == 7
    assert lib.nominal_density.tolist() == list(NOMINAL_DENSITIES)
    assert float(lib.eta_grid[0]) == -1.0 and float(lib.eta_grid[-1]) == 1.0


def test_composition():
    comp = xb.material_composition(BACKEND)
    assert set(comp["water"]) == {1, 8}
    assert abs(comp["water"][1] - 0.1119) < 2e-4
    assert abs(sum(comp["bone"].values()) - 1.0) < 1e-12
    assert abs(comp["bone"][20] - 0.20993) < 1e-6


def test_spectrum_txt_roundtrip(tmp_path):
    e, s = xb.default_spectrum_125kvp(BACKEND)
    p = tmp_path / "spec.txt"
    np.savetxt(p, np.stack([e, s], axis=1), header="E_keV weight")
    e2, s2 = xb.load_spectrum_txt(str(p))
    np.testing.assert_allclose(e2, e)
    np.testing.assert_allclose(s2, s, rtol=1e-12)


def test_zero_spectrum_group_falls_back_to_flat():
    E = np.array([10.0, 50.0, 50.0001, 125.0])
    s = np.array([1.0, 1.0, 0.0, 0.0])  # nothing above 50 keV
    cfg = SolverConfig(num_groups=2, group_edges_kev=(125.0, 60.0, 10.0), legendre_order=1, num_eta_table=11)
    with pytest.warns(RuntimeWarning):
        lib, info = xb.compute_library(cfg, (E, s), BACKEND, de_kev=5.0, n_gl=8)
    assert info.flat_fallback_groups == [0]
    assert float(lib.spectrum_fraction[0]) == 0.0
    # zero-weight fine bins are dropped: the empty group has no fine energies (its uncollided fluence is 0)
    assert bool((lib.fine_group == 1).all()) and float(lib.fine_weight.sum()) == pytest.approx(1.0, rel=1e-12)


# ---- (g) cache --------------------------------------------------------------------------------------

def test_cache_roundtrip(tmp_path, monkeypatch):
    cfg = SolverConfig(num_groups=3, legendre_order=3, num_eta_table=51)
    lib1 = xb.build_library(cfg, None, cache_dir=str(tmp_path), de_kev=2.0, n_gl=8)
    files = list(tmp_path.glob("mglib_*.pt"))
    assert len(files) == 1

    def boom(*a, **k):
        raise AssertionError("cache miss: compute_library called again")

    monkeypatch.setattr(xb, "compute_library", boom)
    lib2 = xb.build_library(cfg, None, cache_dir=str(tmp_path), de_kev=2.0, n_gl=8)
    for k in lib1.__dataclass_fields__:
        a, b = getattr(lib1, k), getattr(lib2, k)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), k
        else:
            assert a == b
    # a different config is a different key
    monkeypatch.undo()
    xb.build_library(SolverConfig(num_groups=3, legendre_order=2, num_eta_table=51), None,
                     cache_dir=str(tmp_path), de_kev=2.0, n_gl=8)
    assert len(list(tmp_path.glob("mglib_*.pt"))) == 2
    # a cache file without the fine-energy table (older format) is rebuilt, not returned
    path = str(files[0])
    d = torch.load(path)
    for k in ("fine_energy", "fine_mu_t", "fine_weight", "fine_group"):
        del d[k]
    torch.save(d, path)
    with pytest.raises(KeyError):
        MultigroupLibrary.load(path)
    with pytest.warns(RuntimeWarning, match="rebuilding"):
        lib3 = xb.build_library(cfg, None, cache_dir=str(tmp_path), de_kev=2.0, n_gl=8)
    assert torch.equal(lib3.fine_mu_t, lib1.fine_mu_t) and torch.equal(lib3.fine_group, lib1.fine_group)
    assert torch.equal(MultigroupLibrary.load(path).fine_weight, lib1.fine_weight)


# ---- (h) fine-energy attenuation table --------------------------------------------------------------

def test_fine_table_consistent_with_groups(built):
    for (lib, info), spectrum_weight in ((built, True),):
        E, fg, w, m = lib.fine_energy.numpy(), lib.fine_group.numpy(), lib.fine_weight.numpy(), lib.fine_mu_t.numpy()
        assert lib.fine_group.dtype == torch.int64 and lib.fine_mu_t.shape == (7, E.size)
        assert abs(w.sum() - 1.0) < 1e-12 and np.all(w > 0)
        assert E.min() > 10.0 and E.max() < 125.0
        edges = lib.group_edges.numpy()
        assert np.all((E < edges[fg]) & (E > edges[fg + 1]))
        # the fine table is the build quadrature: half-keV bins, same midpoints as BuildInfo
        assert set(np.round(E, 9)) <= set(np.round(info.fine_energy, 9))
        for g in range(G):
            sel = fg == g
            assert w[sel].sum() == pytest.approx(float(lib.spectrum_fraction[g]), rel=1e-12, abs=1e-15)
            if spectrum_weight:  # group constants weighted with the spectrum: <m>_g = mu_t[g]
                np.testing.assert_allclose(m[:, sel] @ w[sel] / w[sel].sum(), lib.mu_t[:, g].numpy(), rtol=1e-12)
        # point-wise totals of the elemental mixture
        for mi, mat in enumerate(lib.materials):
            ref = sum(f * BACKEND.sigma("total", Z, E) for Z, f in info.composition[mat].items())
            np.testing.assert_allclose(m[mi], ref, rtol=1e-12)
    # dtype/device conversion keeps the group index integral
    l32 = built[0].to(dtype=torch.float32)
    assert l32.fine_group.dtype == torch.int64 and l32.fine_mu_t.dtype == torch.float32


def test_fine_table_polychromatic_transmission(built):
    """Spectrum-weighted transmission through water + bone from the 0.5 keV table vs an independent 0.1 keV
    integration of the same spectrum and cross sections: agreement to ~1e-5; the group-wise exp(-mu_t,g L) is
    biased low by within-group beam hardening (about -1.4% after 20 cm of water with 6 groups)."""
    from citrus.scatter_solver.cross_sections import group_transmission

    lib, info = built
    e_sp, w_sp = xb.default_spectrum_125kvp(BACKEND)
    sp = xb._Spectrum(e_sp, w_sp)
    edges = np.arange(10.0, 125.0 + 1e-9, 0.1)
    p = sp.cumint(edges[1:]) - sp.cumint(edges[:-1])
    p /= p.sum()
    Em = 0.5 * (edges[1:] + edges[:-1])
    mu = np.stack([sum(f * BACKEND.sigma("total", Z, Em) for Z, f in info.composition[mat].items())
                   for mat in lib.materials])
    for water, bone in ((20.0, 0.0), (15.0, 3.0), (0.0, 0.0)):
        ref = float(np.sum(p * np.exp(-(mu[WATER] * water + mu[BONE] * bone))))
        path = torch.zeros(7, dtype=torch.float64)
        path[WATER], path[BONE] = water, bone
        T = group_transmission(lib, path)
        T2 = group_transmission(lib, path[[WATER, BONE]], materials=[WATER, BONE])  # only the materials present
        assert torch.allclose(T2, T, rtol=1e-15, atol=0)
        assert T.shape == (G,)
        assert float(T.sum()) == pytest.approx(ref, rel=2e-5)
        old = float((lib.spectrum_fraction * torch.exp(-(lib.mu_t[WATER] * water + lib.mu_t[BONE] * bone))).sum())
        if water == 20.0 and bone == 0.0:
            assert -0.016 < old / ref - 1 < -0.012


# ---- optional: full resolution ---------------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get("CITRUS_SLOW_TESTS"), reason="slow: set CITRUS_SLOW_TESTS=1")
def test_full_resolution_default_config():
    cfg = SolverConfig()  # G=6, L=7, 2001-point eta table
    lib, info = xb.compute_library(cfg, None, BACKEND)
    assert lib.mu_s_eta.shape == (7, 6, 6, 2001)
    for mi, mat in enumerate(lib.materials):
        ref = _group_avg(info, "cs", mat) + _group_avg(info, "rs", mat)
        rows = lib.mu_s_l[mi, :, :, 0].sum(dim=1).numpy() + info.compton_below_cutoff[mi]
        np.testing.assert_allclose(rows, ref, rtol=1e-4)
    for gi in range(6):
        for go in range(gi + 1, 6):
            if float(lib.mu_s_l[WATER, gi, go, 0]) > 0:
                assert _rel_l2(lib, WATER, gi, go, 7) < 0.05
