"""Point-wise photon interaction data of the library builder: NIST XCOM cross sections and xraylib angular shapes.

The library builder (`cross_section_library`) needs, per element Z and photon energy E:

* the integrated cross sections of coherent (Rayleigh) and incoherent (Compton) scattering, photoelectric absorption
  and pair production (nuclear and electron field), as mass cross sections in cm^2/g;
* the angular shapes of Compton and Rayleigh scattering (per steradian, unnormalized: the builder normalizes them to
  the XCOM cross section of the same process);
* the atomic weight (to turn a chemical formula into mass fractions): XCOM's (NIST's) atomic weights, ATWTS.DAT.

Sources
-------
**Integrated cross sections: NIST XCOM** (M.J. Berger, J.H. Hubbell, S.M. Seltzer, J. Chang, J.S. Coursey,
R. Sukumar, D.S. Zucker and K. Olsen, XCOM: Photon Cross Sections Database, NIST Standard Reference Database 8 (XGAM),
National Institute of Standards and Technology, https://physics.nist.gov/xcom). The offline distribution of XCOM 3.1
(``XCOM_URL``, the archive XCOM.tar.gz of 172805 bytes) holds one table per element Z = 1..100, MDATX3.001 ..
MDATX3.100, from 1 keV to 100 GeV, in barns/atom: coherent scattering (relativistic form factors), incoherent
scattering (Klein-Nishina times the incoherent scattering function), photoelectric absorption (Scofield) and pair
production in the nuclear and in the electron field. XCOM's interpolation is reproduced here as its Fortran program
XCOM.f does it (``XcomElement.cross_section``):

* the energy grid lists every absorption edge twice, once just below the edge (the energy minus 0.1 eV) and once at
  the edge; coherent and incoherent scattering are interpolated by a natural cubic spline in (ln E, ln sigma) over the
  grid with the points just below the edges removed (XCOM.f subroutines SCOF and BSPOL);
* photoelectric absorption above the highest edge listed (the K edge) is the same spline over the points from the K
  edge up; below the K edge every interval between two edges has its own short table (energies in MeV and cross
  sections, the last records of the file), interpolated linearly in (ln E, ln sigma) (XCOM.f subroutine BLIN). At an
  edge energy the value above the edge is taken, so the photoelectric cross section jumps at every edge, as in
  XCOM's tables;
* pair production is zero up to its threshold (1.022007 MeV in the nuclear field, 2.044014 MeV in the electron field,
  the values of XCOM.f); above it the spline interpolates ln(sigma / ((E - E_threshold) / E)^3) over the grid points
  from the threshold up, the value at the threshold extrapolated from the three points above it by XCOM.f's formula;
* the mass cross section is sigma (barns/atom) x 0.60221367 / A, with XCOM.f's Avogadro factor and the atomic
  weights A of XCOM's table ATWTS.DAT.

The total is the sum of the five processes (coherent included), XCOM's "total attenuation with coherent scattering".
Elements without listed edges (Z <= 10: their K edge lies below 1 keV) use one spline for the photoelectric cross
section over the whole grid.

**Angular shapes: xraylib** (T. Schoonjans, A. Brunetti, B. Golosio, M. Sanchez del Rio,
V.A. Sole, C. Ferrero and L. Vincze, "The xraylib library for X-ray-matter interactions. Recent developments",
Spectrochimica Acta Part B 66, 776-784 (2011); BSD license): the Compton shape is the Klein-Nishina cross section
times the incoherent scattering function S(x, Z) (xraylib ``DCS_KN`` x ``SF_Compt``: xraylib's ``DCS_Compt``), the
Rayleigh shape the Thomson cross section times the squared atomic form factor F(x, Z) (``DCS_Thoms`` x ``FF_Rayl``^2:
xraylib's ``DCS_Rayl``), with the momentum transfer x = sin(theta/2) E / 12.3984193 keV angstrom (xraylib's
``MomentTransf``). xraylib's tables of S and F start at x = 1e-3 and 1e-7 per angstrom, and xraylib refuses smaller x
(the most forward angles at MeV energies: x = 1e-3 is theta = 4e-6 rad at 6 MeV); there S is continued as
S(x_min) (x / x_min)^2 (S grows as x^2 from 0) and F is held at F(x_min) (F -> Z). The shapes cover Z = 1..98
(``SCATTERING_ELEMENTS``); xraylib has no scattering functions for Z = 99, 100.

**Atomic weights: XCOM** (ATWTS.DAT, the standard atomic weights NIST uses, e.g. H 1.00794, O 15.9994), both for the
barns -> cm^2/g conversion and for the mass fractions of formulas (``XcomXraylibBackend.atomic_mass``). xraylib's
``AtomicWeight`` is not used: it is rounded (H 1.01, C 12.01, N 14.01, O 16.0, which moves water's hydrogen fraction
from 0.111898 to 0.112098) and wrong for thulium (167.27 instead of 168.934); xraylib's element densities, symbols and
NIST compound data are used (`cross_sections`).

Download and cache
------------------
The XCOM data are NOT shipped with the package. ``load_xcom_tables`` looks for them in this order:

1. ``$CITRUS_XCOM_ARCHIVE`` (``XCOM_ARCHIVE_ENVIRONMENT_VARIABLE``): the path of a downloaded XCOM.tar.gz or of the
   folder it unpacks to (the folder holding MDATX3.001 .., or its parent), for machines without internet access;
2. the parsed cache ``<cache>/xcom/xcom_elements.npz`` (``xcom_cache_dir()``, cache = ``default_cache_dir()``);
3. the archive ``<cache>/xcom/XCOM.tar.gz``;
4. a download of ``XCOM_URL`` into ``<cache>/xcom/XCOM.tar.gz`` (urllib; size and SHA-256 checked against
   ``XCOM_ARCHIVE_SIZE`` and ``XCOM_ARCHIVE_SHA256``).

The tables are read from the archive in memory (only the files MDATX3.001 .. MDATX3.100 and ATWTS.DAT; nothing is
written outside the cache folder) and parsed once into the compact cache ``xcom_elements.npz`` (every element's grid,
the five cross sections, its edges and the tables below the K edge, plus the SHA-256 of the source). When neither the
environment variable, the cache nor the network is available, ``load_xcom_tables`` raises a RuntimeError naming the
URL and the variable.

Terms of use. XCOM is a NIST Standard Reference Database. NIST distributes it free of charge and makes no warranties
about the data (XCOM Readme, "Disclaimer"); the XCOM program carries the note "Copyright 1999, Martin J. Berger.
Permission is given to use the program for non-commercial purposes". CITRUS does not redistribute the data: every
installation downloads them from NIST, and their use is subject to NIST's terms for Standard Reference Data
(NIST's copyright, fair use and licensing statements for SRD, https://www.nist.gov/srd).
Publications using results of CITRUS should cite XCOM and xraylib as above.

Units: energies in keV at the interface (eV inside the XCOM tables, as in the files), mass cross sections in cm^2/g,
angular shapes in cm^2/g/sr, scattering angles through eta = cos(theta).
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import tarfile
import urllib.request
import warnings
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

__all__ = [
    "XCOM_URL",
    "XCOM_ARCHIVE_SIZE",
    "XCOM_ARCHIVE_SHA256",
    "XCOM_ARCHIVE_ENVIRONMENT_VARIABLE",
    "XCOM_ELEMENTS",
    "SCATTERING_ELEMENTS",
    "CROSS_SECTION_KINDS",
    "XcomElement",
    "XcomXraylibBackend",
    "default_cache_dir",
    "xcom_cache_dir",
    "parse_xcom_element",
    "parse_xcom_atomic_weights",
    "read_xcom_source",
    "load_xcom_tables",
    "resolve_backend",
]

XCOM_URL = "https://physics.nist.gov/PhysRefData/Xcom/XCOM.tar.gz"
XCOM_ARCHIVE_SIZE = 172805  # bytes of XCOM.tar.gz (XCOM 3.1)
XCOM_ARCHIVE_SHA256 = "b2bdd0608bbece4809a337b7264c85b7997f608a1f8e2181ce5201421efcc0de"
XCOM_ARCHIVE_ENVIRONMENT_VARIABLE = "CITRUS_XCOM_ARCHIVE"
XCOM_ELEMENTS = range(1, 101)  # Z of the XCOM tables
SCATTERING_ELEMENTS = range(1, 99)  # Z with xraylib scattering functions
# the kinds of `XcomXraylibBackend.sigma`; "pair" is nuclear plus electron field, "total" the sum of all five processes
CROSS_SECTION_KINDS = ("total", "photoelectric", "compton", "rayleigh", "pair")

_XCOM_CACHE_FILE = "xcom_elements.npz"
_XCOM_ARCHIVE_FILE = "XCOM.tar.gz"
_XCOM_CACHE_VERSION = 1  # layout of xcom_elements.npz
_AVOGADRO_PER_BARN = 0.60221367  # XCOM.f AVOG: Avogadro's number x 1e-24 cm^2/barn
_PAIR_THRESHOLD_NUCLEAR_EV = 1.022007e6  # XCOM.f EPAIR1
_PAIR_THRESHOLD_ELECTRON_EV = 2.044014e6  # XCOM.f EPAIR2
_PAIR_POINTS_NUCLEAR = 55  # XCOM.f: grid points from 100 GeV down to the nuclear-field threshold
_PAIR_POINTS_ELECTRON = 51  # ... down to the electron-field threshold
_ENERGY_MIN_EV, _ENERGY_MAX_EV = 1.0e3, 1.0e11  # range of the XCOM tables
_MAX_DOWNLOAD_BYTES = 16 * 1024 * 1024
_PROJECT_URL = "https://github.com/cabouman/citrus"
# smallest momentum transfer (1/angstrom) of xraylib's tables of S(x, Z) and F(x, Z) for every Z = 1..98
_SCATTERING_FUNCTION_X_MIN = 1.0e-3
_FORM_FACTOR_X_MIN = 1.0e-7


def default_cache_dir() -> str:
    """Directory of the CITRUS caches (multigroup libraries, the XCOM data): ``$CITRUS_CACHE_DIR`` if set, else
    ``~/.cache/citrus``."""
    return os.environ.get("CITRUS_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "citrus")


def xcom_cache_dir() -> str:
    """Directory of the downloaded XCOM archive and of its parsed cache: ``default_cache_dir()/xcom``."""
    return os.path.join(default_cache_dir(), "xcom")


# =============================================================================================
# XCOM tables of one element
# =============================================================================================

def _natural_spline(x: np.ndarray, y: np.ndarray):
    """Natural cubic spline through (x [n], y [n]) with x strictly increasing: XCOM.f's SCOF (second derivative zero
    at both ends; evaluated outside the knots with the end polynomials, as XCOM.f's BSPOL)."""
    from scipy.interpolate import CubicSpline

    return CubicSpline(x, y, bc_type="natural", extrapolate=True)


def _linear_with_end_segments(s: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear interpolation of y [n] over strictly increasing x [n] at s [k], extrapolated with the end segments
    (XCOM.f's BLIN)."""
    index = np.clip(np.searchsorted(x, s, side="right") - 1, 0, x.size - 2)
    x0, x1, y0, y1 = x[index], x[index + 1], y[index], y[index + 1]
    return y0 + (s - x0) * (y1 - y0) / (x1 - x0)


@dataclass(eq=False)
class XcomElement:
    """The XCOM 3.1 table of one element (file MDATX3.<Z>) and its interpolation (module docstring, XCOM.f).

    Fields (n grid energies, n_e absorption edges above 1 keV):
      atomic_number          Z
      atomic_weight          A (g/mol) of XCOM's ATWTS.DAT, the factor of the barns -> cm^2/g conversion
      energy_ev [n]          the grid, ascending, every edge twice (0.1 eV below it, or at the same energy above
                             100 keV, and at it)
      coherent, incoherent, photoelectric, pair_nuclear, pair_electron [n]
                             cross sections in barns/atom on the grid (the photoelectric value at the lower of the
                             two edge energies is the one below the edge)
      edge_energy_ev [n_e]   the edge energies, ascending (the K edge last)
      edge_index [n_e]       index into energy_ev of each edge energy (the point above the edge; index - 1 is the
                             point below it)
      edge_labels            the edge names, ascending ("M5", ..., "K")
      subshell_energy_ev     n_e arrays: the photoelectric tables of the intervals below the K edge, interval i from
                             edge i - 1 (1 keV for i = 0) to edge i, energies in eV
      subshell_photoelectric n_e arrays: the photoelectric cross sections (barns/atom) at those energies
    """

    atomic_number: int
    atomic_weight: float
    energy_ev: np.ndarray
    coherent: np.ndarray
    incoherent: np.ndarray
    photoelectric: np.ndarray
    pair_nuclear: np.ndarray
    pair_electron: np.ndarray
    edge_energy_ev: np.ndarray
    edge_index: np.ndarray
    edge_labels: tuple[str, ...]
    subshell_energy_ev: tuple[np.ndarray, ...]
    subshell_photoelectric: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        self._splines: dict[str, object] = {}
        n = self.energy_ev.size
        for name in ("coherent", "incoherent", "photoelectric", "pair_nuclear", "pair_electron"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"XCOM Z = {self.atomic_number}: {name} has shape {getattr(self, name).shape}, "
                                 f"expected ({n},)")
        # the grid without the points below the edges increases strictly; with them it may repeat an energy (above
        # 100 keV the file's 6 digits cannot hold the edge energy minus 0.1 eV, e.g. the K edge of Z >= 87)
        if not (np.all(np.diff(self.energy_ev) >= 0) and self.edge_index.size == np.unique(self.edge_index).size
                and np.all(self.edge_index >= 1) and np.all(np.diff(self.energy_ev[self._retained()]) > 0)):
            raise ValueError(f"XCOM Z = {self.atomic_number}: energy grid not increasing")
        if len(self.subshell_energy_ev) != self.edge_energy_ev.size:
            raise ValueError(f"XCOM Z = {self.atomic_number}: {len(self.subshell_energy_ev)} photoelectric tables "
                             f"below the K edge for {self.edge_energy_ev.size} edges")
        for i, index in enumerate(self.edge_index):
            # 1e-3: the file of Re lists its M5 edge at 1822.9 eV and puts it on the grid at 1822.4 eV
            if not abs(self.energy_ev[index] / self.edge_energy_ev[i] - 1.0) < 1e-3:
                raise ValueError(f"XCOM Z = {self.atomic_number}: edge {self.edge_labels[i]} at "
                                 f"{self.edge_energy_ev[i]} eV is not on the grid")

    @property
    def mass_factor(self) -> float:
        """barns/atom -> cm^2/g: 0.60221367 / A (XCOM.f)."""
        return _AVOGADRO_PER_BARN / float(self.atomic_weight)

    def _retained(self) -> np.ndarray:
        """Boolean mask [n] of the grid without the points just below the edges (XCOM.f, loop 360)."""
        keep = np.ones(self.energy_ev.size, dtype=bool)
        keep[np.asarray(self.edge_index, dtype=np.int64) - 1] = False
        return keep

    def _spline(self, name: str):
        """The spline of one process in ln E (built once): name in coherent, incoherent, photoelectric (above the K
        edge, or everywhere without edges), pair_nuclear, pair_electron."""
        if name in self._splines:
            return self._splines[name]
        keep = self._retained()
        energy = self.energy_ev[keep]
        if name in ("coherent", "incoherent"):
            spline = _natural_spline(np.log(energy), np.log(getattr(self, name)[keep]))
        elif name == "photoelectric":
            start = int(self.edge_index[-1]) if self.edge_index.size else 0
            spline = _natural_spline(np.log(self.energy_ev[start:]), np.log(self.photoelectric[start:]))
        else:
            points, threshold = ((_PAIR_POINTS_NUCLEAR, _PAIR_THRESHOLD_NUCLEAR_EV) if name == "pair_nuclear"
                                 else (_PAIR_POINTS_ELECTRON, _PAIR_THRESHOLD_ELECTRON_EV))
            grid = energy[-points:].copy()
            values = getattr(self, name)[keep][-points:]
            if abs(grid[0] / threshold - 1.0) > 1e-3:
                raise ValueError(f"XCOM Z = {self.atomic_number}: grid point {grid[0]} eV is not the {name} threshold")
            term = ((grid[1:] - threshold) / grid[1:]) ** 3
            log_reduced = np.empty(points)
            log_reduced[1:] = np.log(values[1:] / term)
            # XCOM.f: PR(threshold) = 3.006275 PR(1) - 2.577757 PR(2) + 0.571482 PR(3), the points above it
            log_reduced[0] = 3.006275 * log_reduced[1] - 2.577757 * log_reduced[2] + 0.571482 * log_reduced[3]
            grid[0] = threshold
            spline = _natural_spline(np.log(grid), log_reduced)
        self._splines[name] = spline
        return spline

    def cross_section(self, process: str, energy_ev: np.ndarray) -> np.ndarray:
        """Cross section of one process in barns/atom at energy_ev [k] (eV, within 1 keV .. 100 GeV): process in
        coherent, incoherent, photoelectric, pair_nuclear, pair_electron. XCOM.f's interpolation (module
        docstring). Returns float64 [k]."""
        energy = np.asarray(energy_ev, dtype=np.float64).ravel()
        if energy.size and (energy.min() < _ENERGY_MIN_EV * (1 - 1e-12) or energy.max() > _ENERGY_MAX_EV * (1 + 1e-12)):
            raise ValueError(f"XCOM energies span 1 keV .. 100 GeV, got {energy.min() / 1e3:g} .. "
                             f"{energy.max() / 1e3:g} keV")
        log_energy = np.log(energy)
        if process in ("coherent", "incoherent"):
            return np.exp(self._spline(process)(log_energy))
        if process in ("pair_nuclear", "pair_electron"):
            threshold = _PAIR_THRESHOLD_NUCLEAR_EV if process == "pair_nuclear" else _PAIR_THRESHOLD_ELECTRON_EV
            out = np.zeros(energy.size)
            above = energy > threshold
            if above.any():
                term = ((energy[above] - threshold) / energy[above]) ** 3
                out[above] = term * np.exp(self._spline(process)(log_energy[above]))
            return out
        if process != "photoelectric":
            raise ValueError(f"unknown XCOM process {process!r}")
        if self.edge_energy_ev.size == 0:
            return np.exp(self._spline("photoelectric")(log_energy))
        out = np.empty(energy.size)
        # interval of every energy: the number of edges at or below it (at an edge the value above the edge)
        interval = np.searchsorted(self.edge_energy_ev, energy, side="right")
        above_k = interval == self.edge_energy_ev.size
        if above_k.any():
            out[above_k] = np.exp(self._spline("photoelectric")(log_energy[above_k]))
        for i in np.unique(interval[~above_k]):
            selected = interval == i
            out[selected] = np.exp(_linear_with_end_segments(log_energy[selected],
                                                             np.log(self.subshell_energy_ev[i]),
                                                             np.log(self.subshell_photoelectric[i])))
        return out


def _tokens(text: str) -> list[str]:
    """Whitespace-separated records of a file; a DOS end-of-file mark (Ctrl-Z, some MDATX3 files end with one) ends
    it."""
    return text.split("", 1)[0].split()


def parse_xcom_element(text: str, atomic_weight: Optional[float] = None) -> XcomElement:
    """Parse one XCOM element file (MDATX3.<Z>, the format read by XCOM.f) into an `XcomElement`.

    Layout (whitespace-separated records): Z and atomic weight; number of edges n_e and grid size n; when n_e > 0 the
    grid indices (1-based) of the edges, their labels and their energies (eV), each list from the K edge down; the n
    grid energies (eV, ascending); the n values of coherent, incoherent, photoelectric, pair (nuclear field) and pair
    (electron field) cross sections (barns/atom); when n_e > 0 the number of photoelectric tables below the K edge,
    their lengths and then all their energies (MeV) and all their cross sections (barns/atom), from the lowest
    interval up.
    atomic_weight: A for the cm^2/g conversion (XCOM's ATWTS.DAT); None takes the value of the file's first line."""
    tokens = _tokens(text)
    position = 0

    def take(count: int) -> list[str]:
        nonlocal position
        if position + count > len(tokens):
            raise ValueError("XCOM element file ends early")
        chunk = tokens[position:position + count]
        position += count
        return chunk

    Z = int(take(1)[0])
    file_weight = float(take(1)[0])
    num_edges, num_energies = (int(t) for t in take(2))
    if num_edges < 0 or num_energies < 80:
        raise ValueError(f"XCOM Z = {Z}: bad header ({num_edges} edges, {num_energies} energies)")
    edge_index = np.array([int(t) for t in take(num_edges)][::-1], dtype=np.int64) - 1
    edge_labels = tuple(take(num_edges)[::-1])
    edge_energy = np.array([float(t) for t in take(num_edges)][::-1], dtype=np.float64)
    arrays = [np.array([float(t) for t in take(num_energies)], dtype=np.float64) for _ in range(6)]
    subshell_energy: list[np.ndarray] = []
    subshell_photoelectric: list[np.ndarray] = []
    if num_edges > 0:
        num_tables = int(take(1)[0])
        if num_tables != num_edges:
            raise ValueError(f"XCOM Z = {Z}: {num_tables} photoelectric tables for {num_edges} edges")
        lengths = [int(t) for t in take(num_tables)]
        subshell_energy = [np.array([float(t) for t in take(k)]) * 1.0e6 for k in lengths]
        subshell_photoelectric = [np.array([float(t) for t in take(k)]) for k in lengths]
    if position != len(tokens):
        raise ValueError(f"XCOM Z = {Z}: {len(tokens) - position} unread values at the end of the file")
    return XcomElement(atomic_number=Z, atomic_weight=float(file_weight if atomic_weight is None else atomic_weight),
                       energy_ev=arrays[0], coherent=arrays[1], incoherent=arrays[2], photoelectric=arrays[3],
                       pair_nuclear=arrays[4], pair_electron=arrays[5], edge_energy_ev=edge_energy,
                       edge_index=edge_index, edge_labels=edge_labels, subshell_energy_ev=tuple(subshell_energy),
                       subshell_photoelectric=tuple(subshell_photoelectric))


def parse_xcom_atomic_weights(text: str) -> np.ndarray:
    """The atomic weights [100] (g/mol, Z = 1..100) of XCOM's ATWTS.DAT (a Fortran DATA statement: the values after
    the continuation column 6 of every line)."""
    values = [float(v) for line in text.splitlines()[1:] for v in re.findall(r"[0-9]+\.[0-9]*", line[6:])]
    if len(values) != len(XCOM_ELEMENTS):
        raise ValueError(f"ATWTS.DAT holds {len(values)} atomic weights, expected {len(XCOM_ELEMENTS)}")
    return np.array(values, dtype=np.float64)


# =============================================================================================
# Locating, downloading and caching the XCOM data
# =============================================================================================

_WANTED_FILE = re.compile(r"^(MDATX3\.(\d{3})|ATWTS\.DAT)$")


def _read_archive(path: str) -> dict[str, bytes]:
    """The wanted files (MDATX3.001 .. MDATX3.100, ATWTS.DAT) of an XCOM archive, read in memory by base name
    (nothing is extracted to disk)."""
    files: dict[str, bytes] = {}
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive.getmembers():
            name = member.name.replace("\\", "/").rsplit("/", 1)[-1]
            if member.isfile() and _WANTED_FILE.match(name) and member.size < 1024 * 1024:
                handle = archive.extractfile(member)
                if handle is not None:
                    files[name] = handle.read()
    return files


def _read_folder(path: str) -> dict[str, bytes]:
    """The wanted files of an unpacked XCOM archive: the folder itself or its subfolder XCOM."""
    for folder in (path, os.path.join(path, "XCOM")):
        if os.path.isfile(os.path.join(folder, "MDATX3.001")):
            files = {}
            for name in os.listdir(folder):
                full = os.path.join(folder, name)
                if _WANTED_FILE.match(name) and os.path.isfile(full):
                    with open(full, "rb") as handle:
                        files[name] = handle.read()
            return files
    raise RuntimeError(f"no XCOM tables (MDATX3.001 ...) in {path!r} or its subfolder XCOM")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_files(files: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode() + b"\0" + hashlib.sha256(files[name]).digest())
    return digest.hexdigest()


def _missing_data_message(reason: str) -> str:
    return (f"the NIST XCOM photon cross sections are not available ({reason}). CITRUS downloads them on first use "
            f"from {XCOM_URL} into {xcom_cache_dir()}. On a machine without internet access download that file "
            f"elsewhere and set the environment variable {XCOM_ARCHIVE_ENVIRONMENT_VARIABLE} to its path (or to the "
            f"folder it unpacks to).")


def _download_archive(destination: str) -> None:
    """Download ``XCOM_URL`` to `destination` (atomically), checking its size and SHA-256."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = destination + f".{os.getpid()}.part"
    from .. import __version__

    # NIST's server refuses urllib's default user agent (HTTP 403); identify the package instead
    request = urllib.request.Request(XCOM_URL, headers={"User-Agent": f"citrus/{__version__} ({_PROJECT_URL})"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(_MAX_DOWNLOAD_BYTES + 1)
    except Exception as exc:  # URLError, socket timeout, HTTP errors
        raise RuntimeError(_missing_data_message(f"download failed: {type(exc).__name__}: {exc}")) from exc
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != XCOM_ARCHIVE_SIZE or digest != XCOM_ARCHIVE_SHA256:
        raise RuntimeError(f"the file downloaded from {XCOM_URL} ({len(data)} bytes, SHA-256 {digest}) is not the "
                           f"XCOM 3.1 archive CITRUS expects ({XCOM_ARCHIVE_SIZE} bytes, SHA-256 "
                           f"{XCOM_ARCHIVE_SHA256}); NIST may have changed it. Point "
                           f"{XCOM_ARCHIVE_ENVIRONMENT_VARIABLE} to an archive you trust to use it anyway.")
    with open(temporary, "wb") as handle:
        handle.write(data)
    os.replace(temporary, destination)


def read_xcom_source(path: str) -> tuple[dict[int, XcomElement], str]:
    """Parse an XCOM archive (XCOM.tar.gz) or unpacked folder: ({Z: XcomElement} for Z = 1..100, SHA-256 of the
    source: of the archive file, or of the table files of a folder). Raises RuntimeError for a missing or incomplete
    source."""
    if os.path.isdir(path):
        files = _read_folder(path)
        digest = _sha256_files(files)
    elif os.path.isfile(path):
        try:
            files = _read_archive(path)
        except (tarfile.TarError, OSError, EOFError) as exc:
            raise RuntimeError(f"{path!r} is not a readable XCOM.tar.gz: {exc}") from exc
        digest = _sha256_file(path)
    else:
        raise RuntimeError(f"no file or folder {path!r}")
    missing = [f"MDATX3.{Z:03d}" for Z in XCOM_ELEMENTS if f"MDATX3.{Z:03d}" not in files]
    if missing or "ATWTS.DAT" not in files:
        missing += [] if "ATWTS.DAT" in files else ["ATWTS.DAT"]
        raise RuntimeError(f"incomplete XCOM data in {path!r}: missing {missing}")
    try:
        weights = parse_xcom_atomic_weights(files["ATWTS.DAT"].decode("ascii", errors="replace"))
        elements = {}
        for Z in XCOM_ELEMENTS:
            element = parse_xcom_element(files[f"MDATX3.{Z:03d}"].decode("ascii", errors="replace"), weights[Z - 1])
            if element.atomic_number != Z:
                raise ValueError(f"MDATX3.{Z:03d} holds Z = {element.atomic_number}")
            elements[Z] = element
    except ValueError as exc:
        raise RuntimeError(f"the XCOM data in {path!r} cannot be parsed: {exc}") from exc
    return elements, digest


def _save_cache(path: str, elements: Mapping[int, XcomElement], source_sha256: str) -> None:
    arrays: dict[str, np.ndarray] = {"cache_version": np.array(_XCOM_CACHE_VERSION),
                                     "source_sha256": np.array(source_sha256)}
    for Z, element in elements.items():
        prefix = f"z{Z:03d}_"
        arrays[prefix + "atomic_weight"] = np.array(element.atomic_weight)
        arrays[prefix + "table"] = np.stack([element.energy_ev, element.coherent, element.incoherent,
                                             element.photoelectric, element.pair_nuclear, element.pair_electron])
        arrays[prefix + "edge_energy_ev"] = element.edge_energy_ev
        arrays[prefix + "edge_index"] = element.edge_index
        arrays[prefix + "edge_labels"] = np.array(element.edge_labels, dtype="<U4")
        arrays[prefix + "subshell_lengths"] = np.array([a.size for a in element.subshell_energy_ev], dtype=np.int64)
        arrays[prefix + "subshell_energy_ev"] = (np.concatenate(element.subshell_energy_ev)
                                                 if element.subshell_energy_ev else np.zeros(0))
        arrays[prefix + "subshell_photoelectric"] = (np.concatenate(element.subshell_photoelectric)
                                                     if element.subshell_photoelectric else np.zeros(0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _load_cache(path: str) -> tuple[dict[int, XcomElement], str]:
    with np.load(path, allow_pickle=False) as data:
        if int(data["cache_version"]) != _XCOM_CACHE_VERSION:
            raise ValueError(f"XCOM cache version {int(data['cache_version'])} != {_XCOM_CACHE_VERSION}")
        elements = {}
        for Z in XCOM_ELEMENTS:
            prefix = f"z{Z:03d}_"
            table = data[prefix + "table"]
            lengths = data[prefix + "subshell_lengths"]
            splits = np.cumsum(lengths)[:-1]
            subshell_energy = tuple(np.split(data[prefix + "subshell_energy_ev"], splits)) if lengths.size else ()
            subshell_photo = tuple(np.split(data[prefix + "subshell_photoelectric"], splits)) if lengths.size else ()
            elements[Z] = XcomElement(
                atomic_number=Z, atomic_weight=float(data[prefix + "atomic_weight"]), energy_ev=table[0].copy(),
                coherent=table[1].copy(), incoherent=table[2].copy(), photoelectric=table[3].copy(),
                pair_nuclear=table[4].copy(), pair_electron=table[5].copy(),
                edge_energy_ev=data[prefix + "edge_energy_ev"].copy(), edge_index=data[prefix + "edge_index"].copy(),
                edge_labels=tuple(str(s) for s in data[prefix + "edge_labels"]),
                subshell_energy_ev=subshell_energy, subshell_photoelectric=subshell_photo)
        return elements, str(data["source_sha256"])


_TABLES_MEMO: dict[str, tuple[dict[int, XcomElement], str, str]] = {}


def load_xcom_tables(cache_dir: Optional[str] = None, *,
                     download: bool = True) -> tuple[dict[int, XcomElement], str, str]:
    """The XCOM tables of Z = 1..100, located, downloaded and cached as the module docstring describes.

    cache_dir: the XCOM cache folder (None: ``xcom_cache_dir()``). download: False never touches the network.
    Returns ({Z: XcomElement}, description of the source, SHA-256 of the source). The result is kept in memory for the
    rest of the process. Raises RuntimeError (naming ``XCOM_URL`` and ``$CITRUS_XCOM_ARCHIVE``) when the data cannot
    be found or fetched."""
    folder = xcom_cache_dir() if cache_dir is None else cache_dir
    override = os.environ.get(XCOM_ARCHIVE_ENVIRONMENT_VARIABLE) or None
    memo_key = f"{folder}|{override}"
    if memo_key in _TABLES_MEMO:
        return _TABLES_MEMO[memo_key]
    cache_path = os.path.join(folder, _XCOM_CACHE_FILE)
    archive_path = os.path.join(folder, _XCOM_ARCHIVE_FILE)
    if override is not None:
        if not os.path.exists(override):
            raise RuntimeError(_missing_data_message(f"{XCOM_ARCHIVE_ENVIRONMENT_VARIABLE}={override!r} does not "
                                                     f"exist"))
        try:
            source_sha256 = (_sha256_file(override) if os.path.isfile(override)
                             else _sha256_files(_read_folder(override)))
        except OSError as exc:
            raise RuntimeError(_missing_data_message(f"{override!r} cannot be read: {exc}")) from exc
        if os.path.isfile(override) and source_sha256 != XCOM_ARCHIVE_SHA256:
            warnings.warn(f"{XCOM_ARCHIVE_ENVIRONMENT_VARIABLE}={override!r} is not the XCOM 3.1 archive CITRUS was "
                          f"validated with (SHA-256 {source_sha256}); using it", RuntimeWarning)
        source = f"{XCOM_ARCHIVE_ENVIRONMENT_VARIABLE}={override}"
    else:
        source_sha256, source = XCOM_ARCHIVE_SHA256, f"{archive_path} (from {XCOM_URL})"
    elements = None
    if os.path.isfile(cache_path):
        try:
            cached, cached_sha256 = _load_cache(cache_path)
            if cached_sha256 == source_sha256:
                elements = cached
        except Exception:  # a damaged or foreign cache file is parsed again and overwritten
            elements = None
    if elements is None:
        if override is not None:
            elements, source_sha256 = read_xcom_source(override)
        else:
            if not os.path.isfile(archive_path) or _sha256_file(archive_path) != XCOM_ARCHIVE_SHA256:
                if not download:
                    raise RuntimeError(_missing_data_message("no cached copy and download=False"))
                _download_archive(archive_path)
            elements, source_sha256 = read_xcom_source(archive_path)
        try:
            _save_cache(cache_path, elements, source_sha256)
        except OSError:
            pass  # a read-only cache folder: parse again next time
    result = (elements, source, str(source_sha256))
    _TABLES_MEMO[memo_key] = result
    return result


# =============================================================================================
# The backend of the library builder
# =============================================================================================

class XcomXraylibBackend:
    """Point-wise photon data for the library builder: XCOM integrated cross sections and xraylib angular shapes
    (module docstring). Energies in keV, mass cross sections in cm^2/g.

    ``sigma(kind, Z, E)``: kind in ``CROSS_SECTION_KINDS`` ("total", "photoelectric", "compton", "rayleigh", "pair";
    "pair" = nuclear + electron field, "total" = the sum of all five XCOM processes), E [K] -> [K].
    ``pdf_raw(kind, Z, E, eta)``: kind "compton" or "rayleigh", E scalar, eta [N] -> [N] cm^2/g/sr (xraylib's
    DCS_Compt / DCS_Rayl, continued below the smallest tabulated momentum transfer); the builder normalizes the shape
    to the XCOM cross section of the process.
    ``library_pdf_normalization(kind, Z, E)``: the sphere integral 2 pi int pdf_raw d eta on a fine quadrature
    (a check of the builder's own quadrature, ``BuildInfo.pdf_normalization_max_relative_deviation``).
    ``atomic_mass(Z)``: XCOM's atomic weight (g/mol, ATWTS.DAT), used to turn formulas into mass fractions.
    ``.name`` ("xcom-xraylib", part of the library cache key), ``.reason`` (where the data came from),
    ``.xcom_sha256`` (SHA-256 of the XCOM source).

    tables: {Z: XcomElement} to use instead of ``load_xcom_tables()`` (tests)."""

    name = "xcom-xraylib"

    def __init__(self, tables: Optional[Mapping[int, XcomElement]] = None) -> None:
        import xraylib  # noqa: F401  (hard dependency; imported here so that a missing install reads clearly)
        import xraylib_np

        self._xraylib = xraylib
        self._xraylib_np = xraylib_np
        if tables is None:
            tables, source, digest = load_xcom_tables()
        else:
            source, digest = "tables given by the caller", ""
        self._tables = dict(tables)
        self.xcom_sha256 = digest
        self.reason = (f"NIST XCOM 3.1 ({source}) and xraylib {xraylib.__version__} "
                       f"({os.path.dirname(xraylib.__file__)})")
        self._atomic_weight: dict[int, float] = {}
        self._fine_quadrature: Optional[tuple[np.ndarray, np.ndarray]] = None

    def element(self, Z: int) -> XcomElement:
        """The XCOM table of element Z (1..100)."""
        if Z not in self._tables:
            raise ValueError(f"no XCOM data for Z = {Z}")
        return self._tables[Z]

    def atomic_mass(self, Z: int) -> float:
        """XCOM's (NIST's) atomic weight of element Z in g/mol (Z = 1..100, ATWTS.DAT): the weight of the barns ->
        cm^2/g conversion, used for the mass fractions of formulas too (xraylib's ``AtomicWeight`` is rounded, e.g.
        H 1.01 and O 16.0, and gives 167.27 for thulium instead of 168.934)."""
        return float(self.element(int(Z)).atomic_weight)

    def sigma(self, kind: str, Z: int, E: np.ndarray) -> np.ndarray:
        """Mass cross section (cm^2/g) of process `kind` (``CROSS_SECTION_KINDS``) of element Z at energies E [K] keV
        (1 keV .. 100 GeV). Returns float64 [K]."""
        element = self.element(int(Z))
        energy_ev = np.asarray(E, dtype=np.float64).ravel() * 1.0e3
        processes = {"photoelectric": ("photoelectric",), "compton": ("incoherent",), "rayleigh": ("coherent",),
                     "pair": ("pair_nuclear", "pair_electron"),
                     "total": ("coherent", "incoherent", "photoelectric", "pair_nuclear", "pair_electron")}
        if kind not in processes:
            raise ValueError(f"unknown cross-section kind {kind!r}; expected one of {CROSS_SECTION_KINDS}")
        barns = sum(element.cross_section(process, energy_ev) for process in processes[kind])
        return np.asarray(barns, dtype=np.float64) * element.mass_factor

    def pdf_raw(self, kind: str, Z: int, E: float, eta: np.ndarray) -> np.ndarray:
        """Per-steradian angular shape (cm^2/g/sr) of Compton or Rayleigh scattering of element Z (1..98) at energy E
        (keV) for the cosines eta [N]: xraylib's DCS_KN x SF_Compt (= DCS_Compt) or DCS_Thoms x FF_Rayl^2
        (= DCS_Rayl), times N_A / A; below xraylib's smallest momentum transfer S is continued as x^2 and F held
        constant (module docstring). Returns float64 [N]."""
        Z = int(Z)
        if Z not in SCATTERING_ELEMENTS:
            raise ValueError(f"element Z = {Z} is outside Z = 1..98 (xraylib's scattering data)")
        eta = np.clip(np.asarray(eta, dtype=np.float64).ravel(), -1.0, 1.0)
        half_angle_sine = np.sqrt(np.clip(0.5 * (1.0 - eta), 0.0, 1.0))  # q = sin(theta/2)
        theta = 2.0 * np.arcsin(half_angle_sine)
        momentum_transfer = half_angle_sine * float(E) / self._xraylib.KEV2ANGST
        Z_array = np.array([Z], dtype=np.int64)
        per_gram = self._xraylib.AVOGNUM / self._xraylib_weight(Z)  # xraylib's cm^2/g (the shape's scale is free)
        xraylib_np = self._xraylib_np
        if kind == "compton":
            x = np.maximum(momentum_transfer, _SCATTERING_FUNCTION_X_MIN)
            scattering_function = xraylib_np.SF_Compt(Z_array, x)[0]
            small = momentum_transfer < _SCATTERING_FUNCTION_X_MIN
            scattering_function[small] *= (momentum_transfer[small] / _SCATTERING_FUNCTION_X_MIN) ** 2
            klein_nishina = xraylib_np.DCS_KN(np.array([float(E)]), theta)[0]
            return klein_nishina * scattering_function * per_gram
        if kind == "rayleigh":
            form_factor = xraylib_np.FF_Rayl(Z_array, np.maximum(momentum_transfer, _FORM_FACTOR_X_MIN))[0]
            return xraylib_np.DCS_Thoms(theta) * form_factor * form_factor * per_gram
        raise ValueError(f"unknown angular shape {kind!r}; expected 'compton' or 'rayleigh'")

    def _xraylib_weight(self, Z: int) -> float:
        """xraylib's atomic weight of Z (g/mol): the per-gram scale of its DCS_Compt / DCS_Rayl, kept so that
        ``pdf_raw`` equals those functions (the builder normalizes the shapes, so the scale does not matter)."""
        if Z not in self._atomic_weight:
            self._atomic_weight[Z] = float(self._xraylib.AtomicWeight(Z))
        return self._atomic_weight[Z]

    def _fine_eta_quadrature(self) -> tuple[np.ndarray, np.ndarray]:
        """(eta [N], weights [N]) with sum w f(eta) ~ int_{-1}^{1} f d eta: 16-point Gauss-Legendre on 61 pieces in
        q = sin(theta/2), [0, 1e-7] and then geometric from 1e-7 to 1 (resolves the forward peaks at any energy up to
        100 MeV), d eta = 4 q dq."""
        if self._fine_quadrature is None:
            breaks = np.concatenate([[0.0], np.geomspace(1.0e-7, 1.0, 61)])
            x, w = np.polynomial.legendre.leggauss(16)
            a, b = breaks[:-1], breaks[1:]
            q = (0.5 * (b - a)[:, None] * x[None, :] + 0.5 * (a + b)[:, None]).ravel()
            weights = (0.5 * (b - a)[:, None] * w[None, :]).ravel() * 4.0 * q
            self._fine_quadrature = (1.0 - 2.0 * q * q, weights)
        return self._fine_quadrature

    def library_pdf_normalization(self, kind: str, Z: int, E: float) -> float:
        """2 pi int_{-1}^{1} pdf_raw(kind, Z, E, eta) d eta in cm^2/g, on the fine quadrature
        ``_fine_eta_quadrature`` (independent of the builder's reference quadrature, which it checks)."""
        eta, weights = self._fine_eta_quadrature()
        return 2.0 * math.pi * float(np.dot(weights, self.pdf_raw(kind, Z, E, eta)))


def resolve_backend() -> XcomXraylibBackend:
    """Load the photon data backend (XCOM + xraylib; ``.name``, ``.reason``: where the data came from).

    Raises RuntimeError when the XCOM data can be neither found nor downloaded (the message names ``XCOM_URL`` and
    ``$CITRUS_XCOM_ARCHIVE``) or xraylib does not import."""
    try:
        return XcomXraylibBackend()
    except ImportError as exc:
        raise RuntimeError(f"xraylib is not importable ({exc}); install it with 'pip install xraylib'") from exc
