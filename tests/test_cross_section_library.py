"""Tests of the multigroup cross-section library builder (citrus/scatter_solver/cross_section_library.py).

Skipped when the photon data (NIST XCOM, xraylib; `photon_data`) are unavailable. The shared fixture uses the production
energy grid (0.5 keV) and a 1001-point eta table; the full-resolution default config build is gated behind
CITRUS_SLOW_TESTS=1.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

from conftest import SPECTRUM_125KVP
from citrus.scatter_solver.config import SolverConfig
from citrus.scatter_solver.cross_sections import BONE, MATERIALS, NOMINAL_DENSITIES, WATER, MultigroupLibrary
from citrus.scatter_solver import cross_section_library

try:
    BACKEND = cross_section_library.resolve_backend()
except RuntimeError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"photon data unavailable ({exc})", allow_module_level=True)

G = 6
L = 7


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(x), axis=-1))


def _group_avg(info: cross_section_library.BuildInfo, kind: str, material: str) -> np.ndarray:
    """Independent group-weighted elemental-mixture cross section [G] from the backend."""
    comp = info.composition[material]
    E = info.fine_energy
    s = sum(w * BACKEND.sigma(kind, Z, E) for Z, w in comp.items())
    ng = int(info.fine_group.max()) + 1
    return np.array([np.sum(info.fine_weight[info.fine_group == g] * s[info.fine_group == g]) for g in range(ng)])


def test_automatic_fine_grid():
    """Without a width every group is cut into at most 100 bins of at least 0.5 keV: narrow (kV) groups keep 0.5 keV
    bins, a 1 MeV group gets 10 keV bins; an explicit width is uniform."""
    edges = np.array([125.0, 85.0, 60.0, 10.0])
    E, groups, low, high = cross_section_library.fine_energy_grid(edges)
    assert np.allclose(high - low, 0.5) and E.size == 80 + 50 + 100
    E_u, _, low_u, high_u = cross_section_library.fine_energy_grid(edges, 0.5)
    assert np.array_equal(E, E_u) and np.array_equal(low, low_u) and np.array_equal(high, high_u)
    E, groups, low, high = cross_section_library.fine_energy_grid(np.array([6000.0, 5000.0, 10.0]))
    assert np.allclose((high - low)[groups == 0], 10.0) and (groups == 0).sum() == 100
    assert np.allclose((high - low)[groups == 1], 49.9) and (groups == 1).sum() == 100
    E, groups, low, high = cross_section_library.fine_energy_grid(np.array([6000.0, 5000.0, 10.0]), 5.0)
    assert np.allclose(high - low, 5.0) and (groups == 1).sum() == 998


@pytest.fixture(scope="module")
def built():
    # equal-fluence groups (the tests below check equal source fractions); the default adaptive eta table
    config = SolverConfig(num_groups=G, legendre_order=L, num_eta_table=1001, group_edges_method="equal_fluence")
    return cross_section_library.compute_library(config, SPECTRUM_125KVP, BACKEND, fine_bin_width_kev=0.5)



# ---- (a) XCOM totals ------------------------------------------------------------------------------

# NIST's tables (XCOM totals with coherent scattering, 4 digits): water (H 0.111894; the library's water has H
# 0.111898 with XCOM's atomic weights and averages over 0.5 keV) to 2e-3; the bone values are those of ICRU-44
# cortical bone (Ca 0.225), not of the catalogue's ICRP bone (Ca 0.210), hence the composition tolerance of 3 %
@pytest.mark.parametrize(
    "material,energy,xcom,tolerance",
    [("water", 40.0, 0.2683, 2e-3), ("water", 60.0, 0.2059, 2e-3), ("water", 80.0, 0.1837, 2e-3),
     ("water", 100.0, 0.1707, 2e-3), ("bone_cortical_icrp", 60.0, 0.3148, 0.03),
     ("bone_cortical_icrp", 100.0, 0.1855, 0.03)],
)
def test_mu_t_vs_xcom(material, energy, xcom, tolerance):
    config = SolverConfig(num_groups=1, group_edges_kev=(energy + 0.25, energy - 0.25), legendre_order=1,
                          num_eta_table=11)
    library, _ = cross_section_library.compute_library(config, SPECTRUM_125KVP, BACKEND)
    m = library.materials.index(material)
    val = float(library.mu_t[m, 0])
    assert abs(val / xcom - 1.0) < tolerance, f"{material} {energy} keV: {val:.5f} vs XCOM {xcom}"


# ---- (b) kinematic bookkeeping conserves the scattering cross section ---------------------------

def test_row_sums_conserve_scatter(built):
    library, info = built
    for mi, mat in enumerate(library.materials):
        reference = _group_avg(info, "compton", mat) + _group_avg(info, "rayleigh", mat)
        rows = library.mu_s_l[mi, :, :, 0].sum(dim=1).numpy()
        below = info.compton_below_cutoff[mi]
        np.testing.assert_allclose(rows + below, reference, rtol=1e-3)
        assert np.all(below >= 0)
        # only the lowest group can backscatter below the 10 keV cutoff (E < ~10.4 keV)
        assert np.all(below[:-1] == 0)
    # below-cutoff loss is small: only E in [10, 10.4] keV contributes
    assert np.all(info.compton_below_cutoff[:, -1] < 0.05 * info.sigma_compton_group[:, -1])


def test_mu_t_is_sum_of_components(built):
    """mu_t = photoelectric + Compton + Rayleigh (+ pair production, zero below 1.022 MeV): an identity of the XCOM
    data."""
    library, info = built
    for mi, mat in enumerate(library.materials):
        comp = _group_avg(info, "photoelectric", mat) + _group_avg(info, "compton", mat) + _group_avg(info, "rayleigh",
                                                                                                      mat)
        np.testing.assert_allclose(library.mu_t[mi].numpy(), comp, rtol=1e-12)
        np.testing.assert_allclose(library.mu_pe[mi].numpy(), _group_avg(info, "photoelectric", mat), rtol=1e-9)


# ---- (c) Rayleigh l = 0 moment ----------------------------------------------------------------------

def test_rayleigh_l0_equals_sigma_rs(built):
    library, info = built
    for mi, mat in enumerate(library.materials):
        np.testing.assert_allclose(info.rayleigh_l[mi, :, 0], _group_avg(info, "rayleigh", mat), rtol=1e-6)
    # diagonal minus Rayleigh = within-group Compton >= 0
    diag = np.stack([library.mu_s_l[:, g, g, 0].numpy() for g in range(G)], axis=1)
    assert np.all(diag - info.rayleigh_l[..., 0] >= 0)


def test_backend_pdf_normalization(built):
    _, info = built
    if info.pdf_normalization_max_relative_deviation is None:
        pytest.skip("backend provides no own normalization factor")
    # backend's own sphere integral of the angular shapes agrees with our q-quadrature
    assert info.pdf_normalization_max_relative_deviation < 2e-3


# ---- (d) Legendre reconstruction vs direct table ---------------------------------------------------

_DIAG_REASON = (
    "known: within-group pairs contain the Rayleigh forward peak (and a sharp kinematic cut of the "
    "within-group Compton ramp); L=7 Legendre reconstruction has ~24-34% relative L2 error "
    "(Compton-only part 4-7%), see docs/source/design_notes.rst"
)


def _rel_l2(library: MultigroupLibrary, m: int, gi: int, go: int, Lr: int) -> float:
    eta = library.eta_grid.numpy()
    P = cross_section_library.legendre_table(eta, Lr)
    mom = library.mu_s_l[m, gi, go, : Lr + 1].numpy()
    table = library.mu_s_eta[m, gi, go].numpy()
    record = ((2 * np.arange(Lr + 1) + 1) / (4 * math.pi) * mom) @ P
    return math.sqrt(_trapz((record - table) ** 2, eta) / _trapz(table**2, eta))


@pytest.mark.parametrize(
    "gi,go",
    [pytest.param(gi, go, marks=pytest.mark.xfail(strict=True, reason=_DIAG_REASON)) if gi == go else (gi, go)
     for gi in range(G) for go in range(gi, G)],
)
def test_legendre_reconstruction_water(built, gi, go):
    library, _ = built
    if float(library.mu_s_l[WATER, gi, go, 0]) <= 0.0:
        pytest.skip("no transfer for this pair")
    err = _rel_l2(library, WATER, gi, go, library.legendre_order)
    assert err < 0.05, f"water {gi}->{go}: relative L2 error {err:.4f} at L={library.legendre_order}"


def test_table_l0_consistent_with_moments(built):
    library, _ = built
    eta = library.eta_grid.numpy()
    table0 = 2 * math.pi * np.sum(0.5 * (library.mu_s_eta[..., 1:] + library.mu_s_eta[..., :-1]).numpy() * np.diff(eta),
                                  axis=-1)
    mom0 = library.mu_s_l[..., 0].numpy()
    big = mom0 > 1e-3 * mom0.max()
    assert np.max(np.abs(table0[big] / mom0[big] - 1)) < 5e-3


# ---- (e) positivity and no upscatter ---------------------------------------------------------------

def test_nonnegative_and_no_upscatter(built):
    for library, _ in (built,):
        assert torch.all(library.mu_s_l[..., 0] >= 0)
        assert torch.all(library.mu_s_eta >= 0)
        for gi in range(G):
            for go in range(gi):
                assert torch.all(library.mu_s_l[:, gi, go] == 0)
                assert torch.all(library.mu_s_eta[:, gi, go] == 0)


# ---- (f) groups and spectrum ------------------------------------------------------------------------

def test_groups_and_spectrum(built):
    for library, _ in (built,):
        e = library.group_edges
        assert e.shape == (G + 1,)
        assert torch.all(e[1:] < e[:-1])
        assert abs(float(e[0]) - 125.0) < 1e-9 and abs(float(e[-1]) - 10.0) < 1e-9
        assert abs(float(library.spectrum_fraction.sum()) - 1.0) < 1e-12
        assert torch.all((library.group_energy < e[:-1]) & (library.group_energy > e[1:]))
        for t in (library.group_edges, library.mu_t, library.mu_pe, library.mu_s_l, library.mu_s_eta, library.eta_grid):
            assert t.dtype == torch.float64 and t.device.type == "cpu"
    library, _ = built  # equal fluence
    np.testing.assert_allclose(library.spectrum_fraction.numpy(), np.full(G, 1.0 / G), atol=1e-4)
    assert library.mu_s_l.shape == (7, G, G, L + 1)
    assert library.mu_s_eta.shape == (7, G, G, 1001)
    assert library.materials == MATERIALS and library.num_materials == 7
    assert library.nominal_density.tolist() == list(NOMINAL_DENSITIES)
    assert float(library.eta_grid[0]) == -1.0 and float(library.eta_grid[-1]) == 1.0


def test_composition():
    comp = cross_section_library.material_composition(BACKEND)
    assert set(comp["water"]) == {1, 8}
    assert abs(comp["water"][1] - 0.111898) < 1e-6  # H2O with XCOM's atomic weights (H 1.00794, O 15.9994)
    assert abs(sum(comp["bone_cortical_icrp"].values()) - 1.0) < 1e-12
    assert abs(comp["bone_cortical_icrp"][20] - 0.20993) < 1e-6


def test_spectrum_txt_roundtrip(tmp_path):
    e, s = SPECTRUM_125KVP
    p = tmp_path / "spec.txt"
    np.savetxt(p, np.stack([e, s], axis=1), header="E_keV weight")
    e2, s2 = cross_section_library.load_spectrum_txt(str(p))
    np.testing.assert_allclose(e2, e)
    np.testing.assert_allclose(s2, s, rtol=1e-12)


def test_zero_spectrum_group_falls_back_to_flat():
    E = np.array([10.0, 50.0, 50.0001, 125.0])
    s = np.array([1.0, 1.0, 0.0, 0.0])  # nothing above 50 keV
    config = SolverConfig(num_groups=2, group_edges_kev=(125.0, 60.0, 10.0), legendre_order=1, num_eta_table=11)
    with pytest.warns(RuntimeWarning):
        library, info = cross_section_library.compute_library(config, (E, s), BACKEND, fine_bin_width_kev=5.0,
                                                              gauss_legendre_nodes=8)
    assert info.flat_fallback_groups == [0]
    assert float(library.spectrum_fraction[0]) == 0.0
    # zero-weight fine bins are dropped: the empty group has no fine energies (its uncollided fluence is 0)
    assert bool((library.fine_group == 1).all()) and float(library.fine_weight.sum()) == pytest.approx(1.0, rel=1e-12)


# ---- (g) cache --------------------------------------------------------------------------------------

def test_cache_roundtrip(tmp_path, monkeypatch):
    config = SolverConfig(num_groups=3, legendre_order=3, num_eta_table=51)
    library1 = cross_section_library.build_library(config, SPECTRUM_125KVP,
                                                   cache_dir=str(tmp_path), fine_bin_width_kev=2.0,
                                                   gauss_legendre_nodes=8)
    files = list(tmp_path.glob("mglib_*.pt"))
    assert len(files) == 1

    def boom(*a, **k):
        raise AssertionError("cache miss: compute_library called again")

    monkeypatch.setattr(cross_section_library, "compute_library", boom)
    library2 = cross_section_library.build_library(config, SPECTRUM_125KVP,
                                                   cache_dir=str(tmp_path), fine_bin_width_kev=2.0,
                                                   gauss_legendre_nodes=8)
    for k in library1.__dataclass_fields__:
        a, b = getattr(library1, k), getattr(library2, k)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), k
        else:
            assert a == b
    # a different config is a different key
    monkeypatch.undo()
    cross_section_library.build_library(SolverConfig(num_groups=3, legendre_order=2, num_eta_table=51), SPECTRUM_125KVP,
                                        cache_dir=str(tmp_path), fine_bin_width_kev=2.0, gauss_legendre_nodes=8)
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
        library3 = cross_section_library.build_library(config, SPECTRUM_125KVP,
                                                       cache_dir=str(tmp_path), fine_bin_width_kev=2.0,
                                                       gauss_legendre_nodes=8)
    assert torch.equal(library3.fine_mu_t, library1.fine_mu_t) and torch.equal(library3.fine_group, library1.fine_group)
    assert torch.equal(MultigroupLibrary.load(path).fine_weight, library1.fine_weight)


# ---- (h) fine-energy attenuation table --------------------------------------------------------------

def test_fine_table_consistent_with_groups(built):
    for (library, info), spectrum_weight in ((built, True),):
        E, fg = library.fine_energy.numpy(), library.fine_group.numpy()
        w, m = library.fine_weight.numpy(), library.fine_mu_t.numpy()
        assert library.fine_group.dtype == torch.int64 and library.fine_mu_t.shape == (7, E.size)
        assert abs(w.sum() - 1.0) < 1e-12 and np.all(w > 0)
        assert E.min() > 10.0 and E.max() < 125.0
        edges = library.group_edges.numpy()
        assert np.all((E < edges[fg]) & (E > edges[fg + 1]))
        # the fine table is the build quadrature: half-keV bins, same midpoints as BuildInfo
        assert set(np.round(E, 9)) <= set(np.round(info.fine_energy, 9))
        for g in range(G):
            sel = fg == g
            assert w[sel].sum() == pytest.approx(float(library.spectrum_fraction[g]), rel=1e-12, abs=1e-15)
            if spectrum_weight:  # group constants weighted with the spectrum: <m>_g = mu_t[g]
                np.testing.assert_allclose(m[:, sel] @ w[sel] / w[sel].sum(), library.mu_t[:, g].numpy(), rtol=1e-12)
        # point-wise totals of the elemental mixture
        for mi, mat in enumerate(library.materials):
            reference = sum(f * BACKEND.sigma("total", Z, E) for Z, f in info.composition[mat].items())
            np.testing.assert_allclose(m[mi], reference, rtol=1e-12)
    # dtype/device conversion keeps the group index integral
    l32 = built[0].to(dtype=torch.float32)
    assert l32.fine_group.dtype == torch.int64 and l32.fine_mu_t.dtype == torch.float32


def test_fine_table_polychromatic_transmission(built):
    """Spectrum-weighted transmission through water + bone from the 0.5 keV table vs an independent 0.1 keV
    integration of the same spectrum and cross sections: agreement to ~1e-5; the group-wise exp(-mu_t,g L) is
    biased low by within-group beam hardening (about -1.4% after 20 cm of water with 6 groups)."""
    from citrus.scatter_solver.cross_sections import group_transmission

    library, info = built
    spectrum_energies, spectrum_weights = SPECTRUM_125KVP
    spectrum_density = cross_section_library._Spectrum(spectrum_energies, spectrum_weights)
    edges = np.arange(10.0, 125.0 + 1e-9, 0.1)
    p = spectrum_density.cumulative_integral(edges[1:]) - spectrum_density.cumulative_integral(edges[:-1])
    p /= p.sum()
    Em = 0.5 * (edges[1:] + edges[:-1])
    mu = np.stack([sum(f * BACKEND.sigma("total", Z, Em) for Z, f in info.composition[mat].items())
                   for mat in library.materials])
    for water, bone in ((20.0, 0.0), (15.0, 3.0), (0.0, 0.0)):
        reference = float(np.sum(p * np.exp(-(mu[WATER] * water + mu[BONE] * bone))))
        path = torch.zeros(7, dtype=torch.float64)
        path[WATER], path[BONE] = water, bone
        T = group_transmission(library, path)
        T2 = group_transmission(library, path[[WATER, BONE]], materials=[WATER, BONE])  # only the materials present
        assert torch.allclose(T2, T, rtol=1e-15, atol=0)
        assert T.shape == (G,)
        assert float(T.sum()) == pytest.approx(reference, rel=2e-5)
        old = float((library.spectrum_fraction * torch.exp(-(library.mu_t[WATER] * water + library.mu_t[BONE]
                                                             * bone))).sum())
        if water == 20.0 and bone == 0.0:
            assert -0.016 < old / reference - 1 < -0.012


# ---- optional: full resolution ---------------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get("CITRUS_SLOW_TESTS"), reason="slow: set CITRUS_SLOW_TESTS=1")
def test_full_resolution_default_config():
    # G=6, L=7, 2001-point (adaptive) eta table; equal-fluence groups as in the fixture above: with the default
    # scattered-fluence groups the 3 -> 5 transfer is a tiny backscatter sliver (the lowest edge, 39.9 keV, lies just
    # above the backscatter limit of the group's lower part), which no Legendre order of practical size reproduces
    config = SolverConfig(group_edges_method="equal_fluence")
    library, info = cross_section_library.compute_library(config, SPECTRUM_125KVP, BACKEND)
    assert library.mu_s_eta.shape == (7, 6, 6, 2001)
    for mi, mat in enumerate(library.materials):
        reference = _group_avg(info, "compton", mat) + _group_avg(info, "rayleigh", mat)
        rows = library.mu_s_l[mi, :, :, 0].sum(dim=1).numpy() + info.compton_below_cutoff[mi]
        np.testing.assert_allclose(rows, reference, rtol=1e-4)
    for gi in range(6):
        for go in range(gi + 1, 6):
            if float(library.mu_s_l[WATER, gi, go, 0]) > 0:
                assert _rel_l2(library, WATER, gi, go, 7) < 0.05
