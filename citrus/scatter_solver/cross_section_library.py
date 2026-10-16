"""Build the multigroup cross-section library of the scatter solver.

Fills a :class:`citrus.scatter_solver.cross_sections.MultigroupLibrary` for the basis materials (the seven-material
table ``cross_sections.MATERIAL_TABLE`` by default: air, lung, adipose, water, muscle, soft tissue, cortical bone) from
point-wise photon interaction data.

Data: XrayPhysics (github.com/kylechampley/XrayPhysics, EPDL97 tables), loaded by ``resolve_backend()``.

Physics summary (all coefficients per unit density, cm^2/g):
  * Groups g = 0..G-1 from HIGHEST to LOWEST energy; ``group_edges`` [G+1] descending.
  * Groups have equal source photon counts (``make_group_edges``) unless explicit edges are given.
  * Within-group weight w(E): the source spectrum, normalized to 1 within each group. A group with zero
    spectrum falls back to a flat weight (a ``RuntimeWarning`` is raised and the group is listed in
    ``BuildInfo.flat_fallback_groups``).
  * Fine energy grid: each group is cut into equal bins of width <= ``de_kev`` (default 0.5 keV);
    every quantity is evaluated at the bin midpoint and weighted by the exact integral of w(E)
    over the bin (midpoint rule in E; the spectrum is integrated exactly as a piecewise-linear
    function, so characteristic lines are not lost).
  * Element mixing is done here, NOT with the backend's compound scatter functions:
        d mu_x/dOmega (E, eta) = sum_i w_i sigma_x(Z_i, E) pdf_x,i(E, eta),   x in {C, R}
    with w_i mass fractions, sigma_x the elemental mass cross section and pdf_x,i the elemental
    angular shape normalized numerically to 2 pi int pdf d eta = 1 (reference quadrature below).
  * Compton kinematics without Doppler broadening: E'(E, eta) = E / (1 + (E/m_e c^2)(1 - eta)).
    For each fine energy E the eta boundaries of every energy group are computed analytically,
    eta_edge = 1 - m_e c^2 (1/E_edge - 1/E), and the eta integral is done piecewise with
    Gauss-Legendre (``n_gl`` nodes per piece, default 32), in the variable
    q = sin(theta/2) = sqrt((1 - eta)/2) (d eta = 4 q dq), which resolves the forward structure
    of the form factor / incoherent scattering function. Pieces are additionally split at fixed
    q breakpoints (``_Q_BREAKS``) that are dense near q = 0.
  * BELOW-CUTOFF CONVENTION: Compton scatter whose outgoing energy E' falls below the lowest
    group edge (``energy_min_kev`` unless explicit edges are given) leaves the group structure.
    It is NOT put in ``mu_s_l``/``mu_s_eta``; it stays in ``mu_t``, so the solver treats it as
    absorption (local deposition). Its magnitude is reported in ``BuildInfo.compton_below_cutoff``
    and satisfies, per material and source group,
        sum_{g_out} mu_s_l[m, g, g_out, 0] + compton_below_cutoff[m, g] = <sigma_CS + sigma_RS>_g.
    With E_min = 10 keV it is only nonzero for E < ~10.4 keV (backscatter limit).
  * Rayleigh is within-group only (no energy loss).
  * ``mu_s_eta`` is tabulated directly on ``eta_grid`` (not reconstructed from moments): Compton
    with the same kinematic indicator evaluated at each table eta, plus within-group Rayleigh.
    At fixed eta the indicator is a cut in E at E_b = 1/(1/E_edge - (1-eta)/m_e c^2); each fine
    energy bin contributes the fraction of its width on the right side of the cut, so the table
    is continuous in eta instead of a 0.5 keV staircase.
  * Fine-energy attenuation table (``fine_energy``, ``fine_mu_t``, ``fine_weight``, ``fine_group``):
    the same fine quadrature, i.e. the total mass attenuation of each material at every bin midpoint
    and the exact spectrum integral over every bin normalized to 1 over the whole group structure
    (bins with zero spectrum are dropped), so sum over a group's bins = ``spectrum_fraction[g]``.
    The uncollided fluence and the primary use it instead of the group-averaged ``mu_t`` (exact in
    energy: no within-group beam hardening error; docs/source/design_notes.rst, "Cross sections").

Materials (``cross_sections.MATERIALS`` order = increasing attenuation at the nominal density, asserted at every fine
energy by ``check_material_order``; nominal densities ``cross_sections.NOMINAL_DENSITIES``; formulas
``MATERIAL_FORMULAS``):
  * air     = XrayPhysics library string 'air' (78.084*N2 + 20.946*O2 + 0.9340*Ar + 0.0397*CO2, read as mass
              fractions of the compounds exactly as XrayPhysics' parser does), 0.0012 g/cm^3.
  * lung    = XrayPhysics 'lung', 0.26 g/cm^3 (the WLAB lung; XrayPhysics lists 0.25).
  * adipose = XrayPhysics 'adipose_tissue', 0.95 g/cm^3 (XrayPhysics lists 0.92).
  * water   = 'H2O' -> with XrayPhysics atomic masses (H 1.008, O 15.999): H 0.111907, O 0.888093; 1.0 g/cm^3.
  * muscle  = ICRU-44 skeletal muscle mass fractions, 1.05 g/cm^3.
  * soft_tissue = ICRU-44 soft tissue mass fractions, 1.06 g/cm^3. Muscle lies BELOW soft tissue at every energy
              (attenuation ratio 0.986-0.991 over 10-125 keV), hence the order.
  * bone    = XrayPhysics library entry 'bone_cortical' (ICRP cortical bone, 1.85 g/cm^3), mass
    fractions H 0.047234, C 0.144330, N 0.041990, O 0.446096, Mg 0.002200, P 0.104970,
    S 0.003150, Ca 0.209930, Zn 0.000100 (sum 1.000000). Note this is ICRP (not ICRU-44) cortical bone; mu/rho differs
    from ICRU-44 by ~1.5% at 60 keV.
Measured ratios of the nominal attenuations of consecutive materials over 10-125 keV: lung/air 238-240, adipose/lung
2.12-3.68, water/adipose 1.055-1.79, muscle/water 1.041-1.064, soft tissue/muscle 1.0095-1.0137, bone/soft tissue
1.80-8.98. Other tables (any number of materials) can be passed as ``materials=`` (name, nominal density, formula).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from .config import SolverConfig
from .cross_sections import ELECTRON_REST_KEV, MATERIAL_TABLE, MultigroupLibrary

__all__ = [
    "BuildInfo",
    "MATERIAL_FORMULAS",
    "default_material_table",
    "check_material_order",
    "build_library",
    "default_cache_dir",
    "compute_library",
    "default_spectrum_125kvp",
    "load_spectrum_txt",
    "make_group_edges",
    "fine_energy_grid",
    "group_of_energy",
    "material_composition",
    "parse_formula",
    "resolve_backend",
    "legendre_table",
    "sphere_quadrature_q",
]

_BUILD_VERSION = "citrus-xsec-2"  # 2: the seven-material table; names, nominal densities and formulas in the key
_TWO_PI = 2.0 * math.pi

# Material definitions (formula strings in XrayPhysics syntax: 'a*F1 + b*F2' with a, b mass
# fractions of the compounds F1, F2; a bare formula is a stoichiometric compound).
_BONE_CORTICAL_XRAYPHYSICS = (
    "0.047234*H + 0.144330*C + 0.041990*N + 0.446096*O + 0.002200*Mg + 0.104970*P + "
    "0.003150*S + 0.209930*Ca + 0.000100*Zn"
)
# XrayPhysics library strings (xrayPhysics().get_material_library(), verbatim; the air terms are mass fractions of the
# compounds, as XrayPhysics' parser reads them); ICRU-44 skeletal muscle and soft tissue mass fractions
_AIR_XRAYPHYSICS = "78.084*N2 + 20.946*O2 + 0.9340*Ar + 0.0397*CO2"
_LUNG_XRAYPHYSICS = (
    "0.10132562304283013*H + 0.10235810831090612*C + 0.028663471831760927*N + 0.75742799115584314*O + "
    "0.0018408652066471242*Na + 0.00073034326133282642*Mg + 0.00080037617680309744*P + 0.0022510579972587113*S + "
    "0.0026612507878702989*Cl + 0.0019409122287475113*K"
)
_ADIPOSE_XRAYPHYSICS = (
    "0.119477*H + 0.637240*C + 0.007970*N + 0.232333*O + 0.000500*Na + 0.000020*Mg + 0.000160*P + 0.000730*S + "
    "0.001190*Cl + 0.000320*K + 0.000020*Ca + 0.000020*Fe + 0.000020*Zn"
)
_MUSCLE_ICRU44 = "0.102*H + 0.143*C + 0.034*N + 0.710*O + 0.001*Na + 0.002*P + 0.003*S + 0.001*Cl + 0.004*K"
_SOFT_TISSUE_ICRU44 = "0.102*H + 0.143*C + 0.034*N + 0.708*O + 0.002*Na + 0.003*P + 0.003*S + 0.002*Cl + 0.003*K"
MATERIAL_FORMULAS: dict[str, str] = {
    "air": _AIR_XRAYPHYSICS, "lung": _LUNG_XRAYPHYSICS, "adipose": _ADIPOSE_XRAYPHYSICS, "water": "H2O",
    "muscle": _MUSCLE_ICRU44, "soft_tissue": _SOFT_TISSUE_ICRU44, "bone": _BONE_CORTICAL_XRAYPHYSICS,
}

MaterialTable = Sequence[tuple[str, float, str]]  # (name, nominal density g/cm^3, XrayPhysics formula) per material


def default_material_table() -> tuple[tuple[str, float, str], ...]:
    """The package's seven materials as (name, nominal density, formula): ``cross_sections.MATERIAL_TABLE`` with
    ``MATERIAL_FORMULAS``."""
    return tuple((n, float(d), MATERIAL_FORMULAS[n]) for n, d in MATERIAL_TABLE)


def _table(materials: Optional[MaterialTable]) -> tuple[tuple[str, float, str], ...]:
    t = default_material_table() if materials is None else tuple((str(n), float(d), str(f)) for n, d, f in materials)
    if not t:
        raise ValueError("empty material table")
    if len({n for n, _, _ in t}) != len(t):
        raise ValueError(f"duplicate material names in {[n for n, _, _ in t]}")
    return t

# Fixed q = sin(theta/2) breakpoints of the angular quadrature (dense near forward scatter).
_Q_BREAKS = np.array(
    [0.0, 0.0025, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
     0.6, 0.7, 0.8, 0.9, 1.0]
)
_THETA_MIN_DEG = 1.0e-3  # XrayPhysics' coherent amplitude is wrong at exactly theta = 0 (x < 1e-6)

_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se "
    "Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb "
    "Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm "
    "Bk Cf Es Fm"
).split()
_SYMBOL_TO_Z = {s: i + 1 for i, s in enumerate(_SYMBOLS)}


# =============================================================================================
# Data backend (XrayPhysics)
# =============================================================================================

class _XrayPhysicsBackend:
    """Point-wise XrayPhysics data. Energies in keV, mass cross sections in cm^2/g.

    ``sigma(kind, Z, E)`` with kind in {"total", "pe", "cs", "rs"}: E [K] -> [K].
    ``pdf_raw(kind, Z, E, eta)`` with kind in {"cs", "rs"}: E scalar, eta [N] -> [N]; an
    unnormalized per-steradian angular shape (normalized by the caller).
    ``library_pdf_norm(kind, Z, E)``: XrayPhysics' own sphere normalization of ``pdf_raw``.
    ``.name`` ("xrayphysics", part of the cache key) and ``.reason`` (where it was loaded from).
    """

    name = "xrayphysics"

    def __init__(self) -> None:
        import ctypes

        from xrayphysics import xrayPhysics  # type: ignore

        xp = xrayPhysics()
        lib = getattr(xp, "libxrayphysics", None)
        if lib is None:
            raise ImportError("xrayphysics imported but libxrayphysics shared library did not load")
        self._xp = xp
        cf = ctypes.c_float
        self._fs: dict[str, Callable] = {}
        for kind, cname in (("total", "sigma"), ("pe", "sigmaPE"), ("cs", "sigmaCS"), ("rs", "sigmaRS")):
            f = getattr(lib, cname)
            f.argtypes = [cf, cf]
            f.restype = cf
            self._fs[kind] = f
        self._fd: dict[str, Callable] = {}
        self._fn: dict[str, Callable] = {}
        for kind, cname in (("cs", "incoherentScatterDistribution"), ("rs", "coherentScatterDistribution")):
            f = getattr(lib, cname)
            f.argtypes = [cf, cf, cf]
            f.restype = cf
            self._fd[kind] = f
            fn = getattr(lib, cname + "_normalizationFactor")
            fn.argtypes = [cf, cf]
            fn.restype = cf
            self._fn[kind] = fn
        self._scalar = float(xp.cross_section_scalar())
        import xrayphysics as _mod  # type: ignore

        self.reason = f"XrayPhysics (EPDL97) loaded from {os.path.dirname(_mod.__file__)}"

    def atomic_mass(self, Z: int) -> float:
        return float(self._xp.atomicMass(Z))

    def sigma(self, kind: str, Z: int, E: np.ndarray) -> np.ndarray:
        f = self._fs[kind]
        Zf = float(Z)
        E = np.asarray(E, dtype=np.float64).ravel()
        out = np.fromiter((f(Zf, float(e)) for e in E), dtype=np.float64, count=E.size)
        return out * self._scalar

    def pdf_raw(self, kind: str, Z: int, E: float, eta: np.ndarray) -> np.ndarray:
        f = self._fd[kind]
        th = np.degrees(np.arccos(np.clip(np.asarray(eta, dtype=np.float64), -1.0, 1.0)))
        th = np.maximum(th, _THETA_MIN_DEG)
        Zf, Ef = float(Z), float(E)
        return np.fromiter((f(Zf, Ef, float(t)) for t in th), dtype=np.float64, count=th.size)

    def library_pdf_norm(self, kind: str, Z: int, E: float) -> float | None:
        return float(self._fn[kind](float(Z), float(E)))

    def spectrum_125kvp(self) -> tuple[np.ndarray, np.ndarray]:
        xp = self._xp
        # 1 keV grid: XrayPhysics adds characteristic-line amplitudes into single samples, which is
        # calibrated as counts per 1 keV bin; we interpret the result as a density per keV.
        gammas = np.arange(1.0, 126.0, 1.0, dtype=np.float32)
        g, s = xp.simulateSpectra(125.0, takeOffAngle=11.0, Z=74, gammas=gammas)
        filt = xp.filterResponse("Ti", 4.506, 0.089, g)
        return np.asarray(g, dtype=np.float64), np.asarray(s, dtype=np.float64) * np.asarray(filt, dtype=np.float64)


def resolve_backend() -> _XrayPhysicsBackend:
    """Load the XrayPhysics data backend (``.name``, ``.reason``: where it was loaded from).

    Raises RuntimeError if XrayPhysics or its shared library is unavailable.
    """
    try:
        return _XrayPhysicsBackend()
    except Exception as exc:  # ImportError, OSError from the DLL, ...
        raise RuntimeError(f"XrayPhysics is unavailable: {type(exc).__name__}: {exc}") from exc


# =============================================================================================
# Materials
# =============================================================================================

def parse_formula(formula: str, atomic_mass: Callable[[int], float]) -> dict[int, float]:
    """Parse an XrayPhysics material string into element mass fractions {Z: w_Z} (sum 1).

    Accepts 'H2O' or mixtures 'a*F1 + b*F2 + ...' where a, b are (relative) mass fractions of
    the compounds; compound element counts are converted to mass fractions with ``atomic_mass``.
    Mirrors XrayPhysics' ``xsec::parseMaterialText``.
    """
    s = formula.replace(" ", "")
    terms = s.split("+")
    acc: dict[int, float] = {}
    total_frac = 0.0
    parsed: list[tuple[float, str]] = []
    for t in terms:
        m = re.match(r"^([0-9.eE\-]*)\*?([A-Z].*)$", t)
        if m is None:
            raise ValueError(f"cannot parse material term {t!r} in {formula!r}")
        frac = float(m.group(1)) if m.group(1) else 1.0
        if frac <= 0:
            raise ValueError(f"non-positive fraction in {formula!r}")
        parsed.append((frac, m.group(2)))
        total_frac += frac
    for frac, comp in parsed:
        counts: dict[int, float] = {}
        for sym, cnt in re.findall(r"([A-Z][a-z]?)([0-9.]*)", comp):
            if sym not in _SYMBOL_TO_Z:
                raise ValueError(f"unknown element {sym!r} in {formula!r}")
            Z = _SYMBOL_TO_Z[sym]
            counts[Z] = counts.get(Z, 0.0) + (float(cnt) if cnt else 1.0)
        mass = sum(n * atomic_mass(Z) for Z, n in counts.items())
        for Z, n in counts.items():
            acc[Z] = acc.get(Z, 0.0) + frac / total_frac * n * atomic_mass(Z) / mass
    norm = sum(acc.values())
    return {Z: w / norm for Z, w in sorted(acc.items())}


def material_composition(backend: _XrayPhysicsBackend,
                         materials: Optional[MaterialTable] = None) -> dict[str, dict[int, float]]:
    """Element mass fractions per basis material (table order; default the seven-material table), keyed by name."""
    return {n: parse_formula(f, backend.atomic_mass) for n, _, f in _table(materials)}


def check_material_order(fine_mu_t: np.ndarray, nominal_density: np.ndarray, names=None) -> None:
    """Raise ValueError unless the nominal attenuations rho_m^nom m_m(E) [M, E] increase strictly with the material
    index at EVERY fine energy (the fixed bracketing order of the density model,
    ``citrus.utilities.attenuation_to_densities``)."""
    a = np.asarray(nominal_density, dtype=np.float64)[:, None] * np.asarray(fine_mu_t, dtype=np.float64)
    bad = np.nonzero(~(np.diff(a, axis=0) > 0).all(axis=1))[0]
    if bad.size:
        i = int(bad[0])
        nm = list(names) if names is not None else [str(k) for k in range(a.shape[0])]
        raise ValueError(f"materials {nm[i]!r} and {nm[i + 1]!r} cross in attenuation at nominal density inside the "
                         f"library's energy range; reorder the material table")


# =============================================================================================
# Spectrum and groups
# =============================================================================================

def load_spectrum_txt(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column text spectrum (energy keV, photon density per keV); '#' comments.

    Commas or whitespace separate columns. Returns (energies [K] ascending, weights [K]) float64.
    """
    rows: list[tuple[float, float]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = [p for p in re.split(r"[,\s;]+", line) if p]
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except (ValueError, IndexError):
                continue  # header line
    a = np.array(rows, dtype=np.float64)
    order = np.argsort(a[:, 0])
    return a[order, 0], a[order, 1]


def default_spectrum_125kvp(backend: _XrayPhysicsBackend | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Default kV source: W anode, 125 kV, 11 deg take-off, 0.89 mm Ti filter.

    XrayPhysics ``simulateSpectra`` x ``filterResponse('Ti', 4.506, 0.089)`` on a 1 keV grid (backend: a loaded
    ``resolve_backend()`` object, None loads one).
    Returns (energies_kev [K], weights [K]) with weights a photon density per keV (arbitrary scale).
    """
    b = resolve_backend() if backend is None else backend
    return b.spectrum_125kvp()


class _Spectrum:
    """Piecewise-linear spectral density s(E) (zero outside the samples) with exact integrals."""

    def __init__(self, energies: np.ndarray, weights: np.ndarray) -> None:
        e = np.asarray(energies, dtype=np.float64).ravel()
        w = np.asarray(weights, dtype=np.float64).ravel()
        if e.size != w.size or e.size < 2:
            raise ValueError("spectrum needs >= 2 (energy, weight) samples")
        o = np.argsort(e)
        self.e, self.w = e[o], np.clip(w[o], 0.0, None)
        seg = 0.5 * (self.w[1:] + self.w[:-1]) * np.diff(self.e)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])

    def __call__(self, E: np.ndarray) -> np.ndarray:
        return np.interp(E, self.e, self.w, left=0.0, right=0.0)

    def cumint(self, E: np.ndarray) -> np.ndarray:
        """int_{-inf}^{E} s(E') dE' (exact for the piecewise-linear interpolant)."""
        E = np.clip(np.asarray(E, dtype=np.float64), self.e[0], self.e[-1])
        i = np.clip(np.searchsorted(self.e, E, side="right") - 1, 0, self.e.size - 2)
        x0, x1 = self.e[i], self.e[i + 1]
        y0, y1 = self.w[i], self.w[i + 1]
        t = E - x0
        slope = (y1 - y0) / (x1 - x0)
        return self.cum[i] + y0 * t + 0.5 * slope * t * t

    def max_energy(self) -> float:
        """Upper end of the support of s (first zero sample after the last positive one)."""
        pos = np.nonzero(self.w > 0)[0]
        if pos.size == 0:
            raise ValueError("spectrum is identically zero")
        j = int(pos[-1])
        return float(self.e[min(j + 1, self.e.size - 1)])


def make_group_edges(config: SolverConfig, spectrum: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Group edges [G+1] in keV, descending, spanning [energy_min_kev, kVp].

    ``config.group_edges_kev`` if given (must be strictly descending); else equal-fluence groups
    (equal photon count of the spectrum per group). kVp = upper end of the spectrum support.
    """
    G = int(config.num_groups)
    if config.group_edges_kev is not None:
        edges = np.asarray(config.group_edges_kev, dtype=np.float64)
        if edges.size != G + 1 or np.any(np.diff(edges) >= 0):
            raise ValueError("group_edges_kev must be strictly descending with num_groups+1 entries")
        return edges
    sp = _Spectrum(*spectrum)
    emax, emin = sp.max_energy(), float(config.energy_min_kev)
    if not emax > emin:
        raise ValueError(f"spectrum max energy {emax} <= energy_min_kev {emin}")
    grid = np.linspace(emin, emax, 200001)
    c = sp.cumint(grid) - sp.cumint(np.array([emin]))[0]
    if c[-1] <= 0:
        raise ValueError("spectrum has no photons in [energy_min_kev, kVp]")
    targets = c[-1] * np.arange(1, G) / G
    idx = np.clip(np.searchsorted(c, targets, side="left"), 1, grid.size - 1)
    c0, c1 = c[idx - 1], c[idx]
    t = np.where(c1 > c0, (targets - c0) / np.where(c1 > c0, c1 - c0, 1.0), 0.0)
    inner = grid[idx - 1] + t * (grid[idx] - grid[idx - 1])
    asc = np.concatenate([[emin], inner, [emax]])
    return asc[::-1].copy()


def fine_energy_grid(edges: np.ndarray, de_kev: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cut every group into equal bins of width <= de_kev.

    Returns (E_mid [K], group [K] int, lo [K], hi [K]) with bins ordered by group then
    ascending energy; bins never straddle a group edge.
    """
    mids, grp, los, his = [], [], [], []
    for g in range(edges.size - 1):
        hi, lo = float(edges[g]), float(edges[g + 1])
        n = max(1, int(math.ceil((hi - lo) / de_kev - 1e-9)))
        b = np.linspace(lo, hi, n + 1)
        los.append(b[:-1])
        his.append(b[1:])
        mids.append(0.5 * (b[:-1] + b[1:]))
        grp.append(np.full(n, g, dtype=np.int64))
    return np.concatenate(mids), np.concatenate(grp), np.concatenate(los), np.concatenate(his)


def group_of_energy(E: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Group index of energies E (edges descending); -1 below the lowest edge, 0 at/above top.

    A group g holds edges[g+1] <= E < edges[g].
    """
    asc = edges[::-1]
    j = np.searchsorted(asc, np.asarray(E, dtype=np.float64), side="right") - 1
    G = edges.size - 1
    g = G - 1 - np.minimum(j, G - 1)
    return np.where(j < 0, -1, g)


# =============================================================================================
# Quadrature helpers
# =============================================================================================

def legendre_table(eta: np.ndarray, L: int) -> np.ndarray:
    """P_l(eta) for l = 0..L. eta [N] -> [L+1, N]."""
    eta = np.asarray(eta, dtype=np.float64)
    P = np.empty((L + 1, eta.size))
    P[0] = 1.0
    if L >= 1:
        P[1] = eta
    for l in range(1, L):
        P[l + 1] = ((2 * l + 1) * eta * P[l] - l * P[l - 1]) / (l + 1)
    return P


def sphere_quadrature_q(breaks_q: np.ndarray, n_gl: int) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise Gauss-Legendre in q = sqrt((1-eta)/2) over the given sorted q breakpoints.

    Returns (eta [N], w [N]) such that sum w f(eta) ~= int f(eta) d eta over the eta range of the
    breakpoints (w includes the Jacobian d eta = 4 q dq). Nodes of a piece lie strictly inside it.
    """
    x, wx = np.polynomial.legendre.leggauss(n_gl)
    a, b = breaks_q[:-1], breaks_q[1:]
    keep = b > a
    a, b = a[keep], b[keep]
    h = 0.5 * (b - a)
    q = (h[:, None] * x[None, :] + 0.5 * (a + b)[:, None]).ravel()
    w = (h[:, None] * wx[None, :]).ravel() * 4.0 * q
    return 1.0 - 2.0 * q * q, w


# =============================================================================================
# Build
# =============================================================================================

@dataclass
class BuildInfo:
    """Diagnostics of a library build (not cached; returned by ``compute_library``).

    Shapes: M materials, G groups, K fine energies.
      backend, backend_reason       data backend ("xrayphysics") and where it was loaded from
      composition                   {material: {Z: mass fraction}}
      compton_below_cutoff [M, G]   group-weighted Compton scatter with E' < edges[-1] (absorbed)
      sigma_cs_group / sigma_rs_group / sigma_pe_group / sigma_t_group [M, G]
                                    group-weighted mass cross sections (w(E) weights)
      fine_energy [K], fine_group [K], fine_weight [K]  the energy quadrature of the group constants
                                    (spectrum weights summing to 1 within each group; NOT the library's
                                    fine_weight, which is the spectrum normalized over all groups, zero
                                    bins dropped)
      flat_fallback_groups          groups whose spectrum weight was zero -> flat weight used
      pdf_norm_max_rel_dev          max |N_backend / N_quadrature - 1| over elements, energies,
                                    processes
      rayleigh_l [M, G, L+1], rayleigh_eta [M, G, N_eta]
                                    the within-group Rayleigh parts of mu_s_l / mu_s_eta (diagonal)
    """

    backend: str
    backend_reason: str
    composition: dict[str, dict[int, float]]
    compton_below_cutoff: np.ndarray
    sigma_cs_group: np.ndarray
    sigma_rs_group: np.ndarray
    sigma_pe_group: np.ndarray
    sigma_t_group: np.ndarray
    fine_energy: np.ndarray
    fine_group: np.ndarray
    fine_weight: np.ndarray
    flat_fallback_groups: list[int] = field(default_factory=list)
    pdf_norm_max_rel_dev: float | None = None
    rayleigh_l: np.ndarray | None = None
    rayleigh_eta: np.ndarray | None = None


def _spectrum_or_default(spectrum: tuple[np.ndarray, np.ndarray] | None,
                         b: _XrayPhysicsBackend) -> tuple[np.ndarray, np.ndarray]:
    if spectrum is None:
        spectrum = b.spectrum_125kvp()
    e, w = spectrum
    return np.asarray(e, dtype=np.float64).ravel(), np.asarray(w, dtype=np.float64).ravel()


def compute_library(
    config: SolverConfig,
    spectrum: tuple[np.ndarray, np.ndarray] | None = None,
    backend: _XrayPhysicsBackend | None = None,
    *,
    de_kev: float = 0.5,
    n_gl: int = 32,
    materials: Optional[MaterialTable] = None,
) -> tuple[MultigroupLibrary, BuildInfo]:
    """Build the library without caching; also return diagnostics.

    Args:
      config: uses num_groups, group_edges_kev, energy_min_kev, legendre_order, num_eta_table.
      spectrum: (energies keV [K], photon density per keV [K]); None -> default 125 kVp spectrum.
      backend: a loaded ``resolve_backend()`` object (None loads one).
      de_kev: maximum fine energy bin width (<= 0.5 keV for production).
      n_gl: Gauss-Legendre nodes per angular piece (>= 32 for production).
      materials: (name, nominal density g/cm^3, formula) per basis material, in order of increasing attenuation at
        the nominal density (``check_material_order`` raises otherwise); None: the seven-material table
        (``default_material_table``).

    Returns:
      (MultigroupLibrary with float64 CPU tensors, BuildInfo).
    """
    b = resolve_backend() if backend is None else backend
    e_sp, w_sp = _spectrum_or_default(spectrum, b)
    sp = _Spectrum(e_sp, w_sp)
    edges = make_group_edges(config, (e_sp, w_sp))
    G = edges.size - 1
    L = int(config.legendre_order)
    table = _table(materials)
    names = tuple(n for n, _, _ in table)
    M = len(names)
    m_e = ELECTRON_REST_KEV

    # ---- energy quadrature ----------------------------------------------------------------
    E, grp, lo, hi = fine_energy_grid(edges, de_kev)
    K = E.size
    s_bin = sp.cumint(hi) - sp.cumint(lo)  # exact spectrum photons per fine bin [K]
    flat_fallback: list[int] = []
    W = np.empty(K)  # within-group spectrum weight (flat where the group has no photons)
    for g in range(G):
        sel = grp == g
        tot = s_bin[sel].sum()
        if tot <= 0:
            flat_fallback.append(g)
            W[sel] = (hi - lo)[sel] / (hi - lo)[sel].sum()
        else:
            W[sel] = s_bin[sel] / tot
    if flat_fallback:
        warnings.warn(f"zero spectrum in groups {flat_fallback}; flat within-group weight used", RuntimeWarning)

    group_energy = np.array([np.sum(W[grp == g] * E[grp == g]) for g in range(G)])
    s_group = np.array([s_bin[grp == g].sum() for g in range(G)])
    if s_group.sum() <= 0:
        raise ValueError("spectrum has no photons inside the group structure")
    spectrum_fraction = s_group / s_group.sum()

    # ---- materials and point-wise cross sections ------------------------------------------
    comp = material_composition(b, table)
    elements = sorted({Z for c in comp.values() for Z in c})
    wfrac = np.array([[comp[m].get(Z, 0.0) for Z in elements] for m in names])  # [M, NZ]
    sig = {k: np.stack([b.sigma(k, Z, E) for Z in elements]) for k in ("total", "pe", "cs", "rs")}  # [NZ, K]
    mat_sig = {k: wfrac @ v for k, v in sig.items()}  # [M, K]

    def gavg(x: np.ndarray) -> np.ndarray:  # x [M, K] -> [M, G]
        out = np.zeros((x.shape[0], G))
        np.add.at(out.T, grp, (x * W).T)
        return out

    mu_t = gavg(mat_sig["total"])
    mu_pe = gavg(mat_sig["pe"])

    # ---- angular quadratures ---------------------------------------------------------------
    eta_ref, w_ref = sphere_quadrature_q(_Q_BREAKS, n_gl)
    P_ref = legendre_table(eta_ref, L)  # [L+1, Nr]
    N_eta = int(config.num_eta_table)
    eta_grid = np.linspace(-1.0, 1.0, N_eta)

    mu_s_l = np.zeros((M, G, G, L + 1))
    below = np.zeros((M, G))
    mu_s_eta = np.zeros((M, G, G, N_eta))
    rayl_l = np.zeros((M, G, L + 1))
    rayl_eta = np.zeros((M, G, N_eta))
    # Table kinematics: at fixed eta, E' lands in [edges[j+1], edges[j]) iff E lies in
    # [E_b(edges[j+1], eta), E_b(edges[j], eta)) with E_b(e, eta) = 1 / (1/e - (1 - eta)/m_e)
    # (+inf when 1/e <= (1 - eta)/m_e). Each fine bin contributes the fraction of its width inside
    # that interval, so the tabulated transfer is continuous in eta (a midpoint indicator would
    # give a staircase with 0.5 keV steps).
    inv_b = 1.0 / edges[:, None] - (1.0 - eta_grid[None, :]) / m_e  # [G+1, N_eta]
    E_b = np.where(inv_b > 0, 1.0 / np.where(inv_b > 0, inv_b, 1.0), np.inf)
    norm_dev = 0.0
    have_lib_norm = True

    for k in range(K):
        Ek, g, Wk = float(E[k]), int(grp[k]), float(W[k])
        # Compton eta pieces: kinematic group edges below Ek plus fixed q breaks.
        ed = edges[edges < Ek]
        qk = np.sqrt(np.clip(m_e * (1.0 / ed - 1.0 / Ek), 0.0, None) / 2.0)
        qb = np.unique(np.concatenate([_Q_BREAKS, qk[qk < 1.0]]))
        eta_c, w_c = sphere_quadrature_q(qb, n_gl)
        e_out = Ek / (1.0 + (Ek / m_e) * (1.0 - eta_c))
        gout = group_of_energy(e_out, edges)  # -1 = below cutoff
        P_c = legendre_table(eta_c, L)

        dC_c = np.zeros((M, eta_c.size))
        dR_r = np.zeros((M, eta_ref.size))
        dC_t = np.zeros((M, N_eta))
        dR_t = np.zeros((M, N_eta))
        for iz, Z in enumerate(elements):
            cs_ref = b.pdf_raw("cs", Z, Ek, eta_ref)
            rs_ref = b.pdf_raw("rs", Z, Ek, eta_ref)
            nC = _TWO_PI * float(np.dot(w_ref, cs_ref))
            nR = _TWO_PI * float(np.dot(w_ref, rs_ref))
            for kind, nq in (("cs", nC), ("rs", nR)):
                nl = b.library_pdf_norm(kind, Z, Ek)
                if nl is None:
                    have_lib_norm = False
                else:
                    norm_dev = max(norm_dev, abs(nl / nq - 1.0))
            aC = wfrac[:, iz] * sig["cs"][iz, k] / nC  # [M] scale of the raw shape
            aR = wfrac[:, iz] * sig["rs"][iz, k] / nR
            cs_c = b.pdf_raw("cs", Z, Ek, eta_c)
            dC_c += aC[:, None] * cs_c[None, :]
            dR_r += aR[:, None] * rs_ref[None, :]
            dC_t += aC[:, None] * b.pdf_raw("cs", Z, Ek, eta_grid)[None, :]
            dR_t += aR[:, None] * b.pdf_raw("rs", Z, Ek, eta_grid)[None, :]

        # Legendre moments: Compton binned by outgoing group (+1 shift; column 0 = below cutoff)
        contrib = _TWO_PI * Wk * (dC_c * w_c[None, :])[:, None, :] * P_c[None, :, :]  # [M, L+1, Nc]
        acc = np.zeros((M, L + 1, G + 1))
        np.add.at(acc.transpose(2, 0, 1), gout + 1, contrib.transpose(2, 0, 1))
        mu_s_l[:, g, :, :] += acc[:, :, 1:].transpose(0, 2, 1)
        below[:, g] += acc[:, 0, 0]
        r_l = _TWO_PI * Wk * (dR_r * w_ref[None, :]) @ P_ref.T  # [M, L+1]
        mu_s_l[:, g, g, :] += r_l
        rayl_l[:, g, :] += r_l

        # Direct per-steradian tables
        lo_k, hi_k = float(lo[k]), float(hi[k])
        frac = np.clip(np.minimum(hi_k, E_b[g:G]) - np.maximum(lo_k, E_b[g + 1:G + 1]), 0.0, None) / (hi_k - lo_k)
        mu_s_eta[:, g, g:, :] += Wk * dC_t[:, None, :] * frac[None, :, :]
        mu_s_eta[:, g, g, :] += Wk * dR_t
        rayl_eta[:, g, :] += Wk * dR_t

    info = BuildInfo(
        backend=b.name,
        backend_reason=b.reason,
        composition=comp,
        compton_below_cutoff=below,
        sigma_cs_group=gavg(mat_sig["cs"]),
        sigma_rs_group=gavg(mat_sig["rs"]),
        sigma_pe_group=mu_pe.copy(),
        sigma_t_group=mu_t.copy(),
        fine_energy=E,
        fine_group=grp,
        fine_weight=W,
        flat_fallback_groups=flat_fallback,
        pdf_norm_max_rel_dev=norm_dev if have_lib_norm else None,
        rayleigh_l=rayl_l,
        rayleigh_eta=rayl_eta,
    )
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)  # noqa: E731
    # fine-energy attenuation table: spectrum weights normalized over the whole group structure (the same
    # denominator as spectrum_fraction), zero-weight bins dropped
    fine_w = s_bin / s_bin.sum()
    keep = fine_w > 0
    nominal = np.array([d for _, d, _ in table], dtype=np.float64)
    if M > 1:
        check_material_order(mat_sig["total"][:, keep], nominal, names)
    lib = MultigroupLibrary(
        group_edges=t(edges),
        group_energy=t(group_energy),
        spectrum_fraction=t(spectrum_fraction),
        mu_t=t(mu_t),
        mu_pe=t(mu_pe),
        mu_s_l=t(mu_s_l),
        eta_grid=t(eta_grid),
        mu_s_eta=t(mu_s_eta),
        fine_energy=t(E[keep]),
        fine_mu_t=t(mat_sig["total"][:, keep]),
        fine_weight=t(fine_w[keep]),
        fine_group=torch.as_tensor(np.ascontiguousarray(grp[keep]), dtype=torch.int64),
        materials=names,
        nominal_density=t(nominal),
    )
    return lib, info


def _cache_key(config: SolverConfig, spectrum: tuple[np.ndarray, np.ndarray], backend_name: str,
               de_kev: float, n_gl: int, materials: Optional[MaterialTable] = None) -> str:
    fields = {
        "version": _BUILD_VERSION,
        "backend": backend_name,
        "num_groups": int(config.num_groups),
        "group_edges_kev": None if config.group_edges_kev is None else [float(x) for x in config.group_edges_kev],
        "energy_min_kev": float(config.energy_min_kev),
        "legendre_order": int(config.legendre_order),
        "num_eta_table": int(config.num_eta_table),
        "de_kev": float(de_kev),
        "n_gl": int(n_gl),
        "materials": [list(m) for m in _table(materials)],  # names, nominal densities and formulas, in order
    }
    h = hashlib.sha256(json.dumps(fields, sort_keys=True).encode())
    h.update(np.ascontiguousarray(spectrum[0], dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(spectrum[1], dtype=np.float64).tobytes())
    return h.hexdigest()[:20]


def default_cache_dir() -> str:
    """Directory of the library cache: $CITRUS_CACHE_DIR if set, else ~/.cache/citrus."""
    return os.environ.get("CITRUS_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "citrus")


def build_library(
    config: SolverConfig,
    spectrum: tuple[np.ndarray, np.ndarray] | None = None,
    cache_dir: str | None = "default",
    *,
    de_kev: float = 0.5,
    n_gl: int = 32,
    materials: Optional[MaterialTable] = None,
) -> MultigroupLibrary:
    """Build (or load from cache) the multigroup library for ``config`` and the material table ``materials`` (None:
    the seven-material table, ``default_material_table``; else (name, nominal density, formula) per material).

    Shapes of the result (M materials, 7 by default; G = config.num_groups, L = config.legendre_order):
      group_edges [G+1] desc, group_energy [G], spectrum_fraction [G], mu_t/mu_pe [M, G],
      mu_s_l [M, G, G, L+1], eta_grid [num_eta_table], mu_s_eta [M, G, G, num_eta_table],
      fine_energy [E], fine_mu_t [M, E], fine_weight [E], fine_group [E] (int64; E ~ (kVp - E_min) / de_kev);
      float64 CPU tensors, per unit density (cm^2/g, cm^2/g/sr); plus materials (names) and nominal_density [M].

    The cache file is ``<cache_dir>/mglib_<hash>.pt`` keyed by the build version, the relevant config
    fields, the spectrum arrays, the backend name, the numerical parameters and the material table (names, nominal
    densities, formulas; the two-material caches of citrus < 0.1.0 have other keys and are never loaded). ``cache_dir="default"`` is
    `default_cache_dir()`; ``cache_dir=None`` disables the cache. A cache file that cannot be loaded as a complete
    library (e.g. one lacking the fine-energy table) is rebuilt and overwritten.
    """
    if cache_dir == "default":
        cache_dir = default_cache_dir()
    b = resolve_backend()
    spec = _spectrum_or_default(spectrum, b)
    path = None
    if cache_dir is not None:
        key = _cache_key(config, spec, b.name, de_kev, n_gl, materials)
        path = os.path.join(cache_dir, f"mglib_{key}.pt")
        if os.path.isfile(path):
            try:
                lib = MultigroupLibrary.load(path)
                names = tuple(n for n, _, _ in _table(materials))
                if tuple(lib.materials) != names:
                    raise KeyError(f"cached materials {tuple(lib.materials)} != {names}")
                return lib
            except (KeyError, TypeError, ValueError) as exc:
                warnings.warn(f"rebuilding the library cache {path}: {exc}", RuntimeWarning)
    lib, _ = compute_library(config, spec, b, de_kev=de_kev, n_gl=n_gl, materials=materials)
    if path is not None:
        os.makedirs(cache_dir, exist_ok=True)  # type: ignore[arg-type]
        tmp = path + ".tmp"
        lib.save(tmp)
        os.replace(tmp, path)
    return lib
