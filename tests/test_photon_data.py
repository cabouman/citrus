"""The photon data of the library builder (citrus/scatter_solver/photon_data.py): NIST XCOM cross sections with XCOM's
own interpolation and edge handling, xraylib angular shapes, and the download / cache / offline behaviour.

Skipped when the XCOM data can be neither found nor downloaded. The reference values are the mass attenuation
coefficients of NIST's tables (Hubbell and Seltzer, NIST Standard Reference Database 126), which are XCOM totals with
coherent scattering, tabulated to four digits.
"""

from __future__ import annotations

import os
import tarfile
import urllib.request

import numpy as np
import pytest

from citrus.scatter_solver import photon_data

try:
    BACKEND = photon_data.resolve_backend()
except RuntimeError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"photon data unavailable ({exc})", allow_module_level=True)

# water of NIST's tables (H 0.111894, O 0.888106 by mass): (energy keV, mu/rho cm^2/g)
WATER_NIST = ((10.0, 5.329), (60.0, 0.2059), (100.0, 0.1707), (1000.0, 0.07072), (6000.0, 0.02771),
              (10000.0, 0.02219))
# lead around its K edge (88.0045 keV) and above: (energy keV, mu/rho cm^2/g)
LEAD_NIST = ((88.0044, 1.910), (88.0045, 7.683), (100.0, 5.549), (1000.0, 0.07102), (10000.0, 0.04972))


def _archive_path() -> str:
    """The XCOM source the backend was loaded from (the variable's path or the cached download)."""
    override = os.environ.get(photon_data.XCOM_ARCHIVE_ENVIRONMENT_VARIABLE)
    return override or os.path.join(photon_data.xcom_cache_dir(), "XCOM.tar.gz")


def _no_network(*args, **keywords):
    raise AssertionError("network access")


# ---- XCOM cross sections ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("energy, expected", WATER_NIST)
def test_water_matches_nist_table(energy, expected):
    mass = (0.111894 * BACKEND.sigma("total", 1, np.array([energy]))
            + 0.888106 * BACKEND.sigma("total", 8, np.array([energy])))
    assert float(mass[0]) == pytest.approx(expected, rel=5e-4)


@pytest.mark.parametrize("energy, expected", LEAD_NIST)
def test_lead_k_edge_matches_nist_table(energy, expected):
    assert float(BACKEND.sigma("total", 82, np.array([energy]))[0]) == pytest.approx(expected, rel=5e-4)


def test_total_is_the_sum_of_the_processes():
    E = np.geomspace(1.0, 1.0e8, 400)
    for Z in (1, 6, 8, 20, 53, 82, 100):
        parts = sum(BACKEND.sigma(kind, Z, E) for kind in ("photoelectric", "compton", "rayleigh", "pair"))
        np.testing.assert_allclose(BACKEND.sigma("total", Z, E), parts, rtol=1e-14)


def test_pair_production_thresholds():
    """Zero up to the nuclear-field threshold (1.022007 MeV), positive above; the electron-field part zero up to
    2.044014 MeV."""
    for Z in (1, 8, 82):
        element = BACKEND.element(Z)
        below = np.array([1000.0, 1022.007])
        assert np.all(BACKEND.sigma("pair", Z, below) == 0)
        assert np.all(BACKEND.sigma("pair", Z, np.array([1030.0, 2000.0, 6000.0])) > 0)
        assert np.all(element.cross_section("pair_electron", np.array([1.5e6, 2.044014e6])) == 0)
        assert np.all(element.cross_section("pair_electron", np.array([2.1e6, 6.0e6])) > 0)


def test_interpolation_passes_through_the_table():
    """At the grid energies the interpolation returns XCOM's values: scattering on the whole grid, photoelectric at the
    points above the K edge and at the points of the tables below it, pair production above both thresholds."""
    for Z in (1, 8, 20, 82, 92):
        element = BACKEND.element(Z)
        keep = element._retained()
        energy = element.energy_ev[keep]
        for process in ("coherent", "incoherent"):
            np.testing.assert_allclose(element.cross_section(process, energy), getattr(element, process)[keep],
                                       rtol=1e-12)
        # from the point after the K edge (the grid holds the edge energy to 6 digits: for Z >= 87 it can lie a
        # fraction of an eV below the listed edge energy, where the table below the edge applies)
        start = int(element.edge_index[-1]) + 1 if element.edge_index.size else 0
        np.testing.assert_allclose(element.cross_section("photoelectric", element.energy_ev[start:]),
                                   element.photoelectric[start:], rtol=1e-12)
        for table_energy, table_value in zip(element.subshell_energy_ev, element.subshell_photoelectric):
            inside = table_energy[1:-1]  # the end points are the edges (the value on the other side)
            np.testing.assert_allclose(element.cross_section("photoelectric", inside), table_value[1:-1],
                                       rtol=1e-12)
        above = energy > 2.1e6
        np.testing.assert_allclose(element.cross_section("pair_nuclear", energy[above]),
                                   element.pair_nuclear[keep][above], rtol=1e-12)


def test_photoelectric_jumps_up_at_every_edge():
    """Just below an edge the value below it, at the edge the value above it (XCOM's convention)."""
    for Z in (20, 53, 82, 92):
        element = BACKEND.element(Z)
        assert element.edge_energy_ev.size > 0
        for edge in element.edge_energy_ev:
            below, at = element.cross_section("photoelectric", np.array([edge * (1 - 1e-6), edge]))
            assert at > below, (Z, edge)


def test_spline_is_xcom_natural_spline():
    """The spline of the scattering cross sections equals a line-by-line port of XCOM.f's SCOF and BSPOL (natural
    cubic spline in ln E, evaluated on the descending grid as XCOM.f does)."""
    def scof(x, f):
        n = x.size
        a, b, c, d = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        s = 0.0
        for m in range(n - 1):
            d[m] = x[m + 1] - x[m]
            r = (f[m + 1] - f[m]) / d[m]
            c[m], s = r - s, r
        s = r = 0.0
        c[0] = c[n - 1] = 0.0
        for m in range(1, n - 1):
            c[m] = c[m] + r * c[m - 1]
            b[m] = (x[m - 1] - x[m + 1]) * 2.0 - r * s
            s = d[m]
            r = s / b[m]
        for m in range(n - 2, 0, -1):
            c[m] = (d[m] * c[m + 1] - c[m]) / b[m]
        for m in range(n - 1):
            s, r = d[m], c[m + 1] - c[m]
            d[m], c[m] = r / s, c[m] * 3.0
            b[m] = (f[m + 1] - f[m]) / s - (c[m] + r) * s
            a[m] = f[m]
        return a, b, c, d

    element = BACKEND.element(82)
    keep = element._retained()
    x = np.log(element.energy_ev[keep])[::-1]  # XCOM.f reverses the grid
    f = np.log(element.incoherent[keep])[::-1]
    a, b, c, d = scof(x, f)
    points = np.log(np.geomspace(1.2e3, 9.0e10, 300))
    upper = np.array([min(int(np.max(np.nonzero(x >= p)[0])), x.size - 2) for p in points])
    q = points - x[upper]
    reference = ((d[upper] * q + c[upper]) * q + b[upper]) * q + a[upper]
    np.testing.assert_allclose(np.log(element.cross_section("incoherent", np.exp(points))), reference, atol=1e-12)


def test_atomic_weights_are_xcom():
    """Formulas use XCOM's (NIST's) atomic weights, not xraylib's rounded ones (H 1.01, thulium 167.27)."""
    assert BACKEND.atomic_mass(1) == 1.00794 and BACKEND.atomic_mass(8) == 15.9994
    assert BACKEND.atomic_mass(69) == 168.93421 and BACKEND.atomic_mass(100) == 257.0951


def test_energy_range_is_checked():
    with pytest.raises(ValueError, match="1 keV"):
        BACKEND.sigma("total", 8, np.array([0.5]))
    with pytest.raises(ValueError, match="unknown cross-section kind"):
        BACKEND.sigma("absorption", 8, np.array([60.0]))


# ---- xraylib angular shapes --------------------------------------------------------------------------------------

@pytest.mark.parametrize("Z", [1, 8, 20, 82, 98])
@pytest.mark.parametrize("energy", [20.0, 60.0, 1000.0, 6000.0])
def test_shapes_are_xraylib_dcs(Z, energy):
    """pdf_raw is xraylib's DCS_Compt / DCS_Rayl (cm^2/g/sr) wherever xraylib accepts the momentum transfer."""
    import xraylib

    theta = np.linspace(0.05, np.pi, 40)
    compton = BACKEND.pdf_raw("compton", Z, energy, np.cos(theta))
    rayleigh = BACKEND.pdf_raw("rayleigh", Z, energy, np.cos(theta))
    np.testing.assert_allclose(compton, [xraylib.DCS_Compt(Z, energy, t) for t in theta], rtol=1e-10)
    np.testing.assert_allclose(rayleigh, [xraylib.DCS_Rayl(Z, energy, t) for t in theta], rtol=1e-10)


def test_shapes_continue_below_the_tables():
    """At the most forward angles (below xraylib's smallest momentum transfer) the Compton shape falls continuously to
    0 at eta = 1 and the Rayleigh shape stays at its forward value."""
    one_minus_eta = np.geomspace(1e-16, 1e-6, 200)
    eta = np.concatenate([[1.0], 1.0 - one_minus_eta])
    for Z in (1, 8, 82):
        compton = BACKEND.pdf_raw("compton", Z, 6000.0, eta)
        rayleigh = BACKEND.pdf_raw("rayleigh", Z, 6000.0, eta)
        assert compton[0] == 0.0 and np.all(np.diff(compton) >= 0) and np.all(np.isfinite(compton))
        assert np.all(rayleigh > 0) and abs(rayleigh[1] / rayleigh[0] - 1) < 1e-9
    with pytest.raises(ValueError, match="1..98"):
        BACKEND.pdf_raw("compton", 99, 60.0, np.array([0.0]))


def test_fine_normalization_is_the_sphere_integral():
    """library_pdf_normalization integrates the shape over the sphere: the Klein-Nishina cross section per electron
    times Z N_A / A for a free-electron-like shape at high energy (S -> Z), checked against the closed form."""
    import xraylib

    energy, Z = 6000.0, 8
    k = energy / xraylib.MEC2
    klein_nishina = 2 * np.pi * xraylib.RE2 * ((1 + k) / k**2 * (2 * (1 + k) / (1 + 2 * k) - np.log(1 + 2 * k) / k)
                                               + np.log(1 + 2 * k) / (2 * k) - (1 + 3 * k) / (1 + 2 * k) ** 2)
    expected = klein_nishina * Z * xraylib.AVOGNUM / xraylib.AtomicWeight(Z)
    assert BACKEND.library_pdf_normalization("compton", Z, energy) == pytest.approx(expected, rel=2e-3)


# ---- download, cache and offline use ------------------------------------------------------------------------------

def test_archive_variable_and_cache(tmp_path, monkeypatch):
    """$CITRUS_XCOM_ARCHIVE pointing to an archive parses it without network access and writes the compact cache; the
    next load reads the cache; an unpacked folder gives the same tables."""
    archive = _archive_path()
    if not os.path.isfile(archive):
        pytest.skip("no XCOM.tar.gz on disk")
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    monkeypatch.setattr(photon_data, "_TABLES_MEMO", {})
    monkeypatch.setenv(photon_data.XCOM_ARCHIVE_ENVIRONMENT_VARIABLE, archive)
    elements, source, digest = photon_data.load_xcom_tables(str(tmp_path / "xcom"))
    assert sorted(elements) == list(photon_data.XCOM_ELEMENTS) and archive in source
    assert (tmp_path / "xcom" / "xcom_elements.npz").is_file()
    if digest == photon_data.XCOM_ARCHIVE_SHA256:
        assert os.path.getsize(archive) == photon_data.XCOM_ARCHIVE_SIZE
    monkeypatch.setattr(photon_data, "_TABLES_MEMO", {})
    monkeypatch.setattr(photon_data, "read_xcom_source", _no_network)
    cached, _, cached_digest = photon_data.load_xcom_tables(str(tmp_path / "xcom"))
    assert cached_digest == digest
    for Z in (1, 8, 82, 100):
        for name in ("energy_ev", "coherent", "incoherent", "photoelectric", "pair_nuclear", "pair_electron",
                     "edge_energy_ev", "edge_index"):
            assert np.array_equal(getattr(cached[Z], name), getattr(elements[Z], name)), (Z, name)
        assert cached[Z].edge_labels == elements[Z].edge_labels
        assert all(np.array_equal(a, b) for a, b in zip(cached[Z].subshell_energy_ev, elements[Z].subshell_energy_ev))
    folder = tmp_path / "unpacked"
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            name = member.name.rsplit("/", 1)[-1]
            if member.isfile() and (name.startswith("MDATX3.") or name == "ATWTS.DAT"):
                (folder / "XCOM").mkdir(parents=True, exist_ok=True)
                (folder / "XCOM" / name).write_bytes(handle.extractfile(member).read())
    monkeypatch.undo()
    from_folder, _ = photon_data.read_xcom_source(str(folder))
    assert np.array_equal(from_folder[26].photoelectric, elements[26].photoelectric)


def test_missing_data_raises_with_the_url(tmp_path, monkeypatch):
    """No variable, no cache and no network: a RuntimeError naming the URL and the variable; a download that is not
    the expected archive is refused."""
    monkeypatch.delenv(photon_data.XCOM_ARCHIVE_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr(photon_data, "_TABLES_MEMO", {})

    def offline(*args, **keywords):
        raise OSError("no network")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    with pytest.raises(RuntimeError) as info:
        photon_data.load_xcom_tables(str(tmp_path / "empty"))
    assert photon_data.XCOM_URL in str(info.value)
    assert photon_data.XCOM_ARCHIVE_ENVIRONMENT_VARIABLE in str(info.value)
    with pytest.raises(RuntimeError, match="download=False"):
        photon_data.load_xcom_tables(str(tmp_path / "empty"), download=False)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size=-1):
            return b"not the archive"

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Response())
    with pytest.raises(RuntimeError, match="SHA-256"):
        photon_data.load_xcom_tables(str(tmp_path / "bad"))
    assert not (tmp_path / "bad" / "XCOM.tar.gz").exists()
    monkeypatch.setenv(photon_data.XCOM_ARCHIVE_ENVIRONMENT_VARIABLE, str(tmp_path / "nowhere"))
    with pytest.raises(RuntimeError, match="does not exist"):
        photon_data.load_xcom_tables(str(tmp_path / "empty"))
