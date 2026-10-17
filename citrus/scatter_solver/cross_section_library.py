"""Build the multigroup cross-section library of the scatter solver.

Fills a :class:`citrus.scatter_solver.cross_sections.MultigroupLibrary` for the anchor materials of the density model
(any subset of ``cross_sections.MATERIAL_CATALOGUE``, ``SolverConfig.materials``; by default the seven of
``cross_sections.MATERIALS``: air, lung, adipose, water, muscle, soft tissue, cortical bone) from point-wise photon
interaction data.

Data: XrayPhysics (github.com/kylechampley/XrayPhysics, EPDL97 tables), loaded by ``resolve_backend()``.

Physics summary (all coefficients per unit density, cm^2/g):
  * Groups g = 0..G-1 from HIGHEST to LOWEST energy; ``group_edges`` [G+1] descending.
  * Groups have equal source photon counts (``make_group_edges``) unless explicit edges are given.
  * Within-group weight w(E): the source spectrum, normalized to 1 within each group. A group with zero
    spectrum falls back to a flat weight (a ``RuntimeWarning`` is raised and the group is listed in
    ``BuildInfo.flat_fallback_groups``).
  * Fine energy grid: each group is cut into equal bins of width <= ``fine_bin_width_kev`` (default 0.5 keV);
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
    Gauss-Legendre (``gauss_legendre_nodes`` nodes per piece, default 32), in the variable
    q = sin(theta/2) = sqrt((1 - eta)/2) (d eta = 4 q dq), which resolves the forward structure
    of the form factor / incoherent scattering function. Pieces are additionally split at fixed
    q breakpoints (``_Q_BREAKPOINTS``) that are dense near q = 0.
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

Materials. The compositions and nominal densities of every material the builder knows are in
``cross_sections.MATERIAL_CATALOGUE`` (sources there); ``MATERIAL_FORMULAS`` maps each name to its composition. A
library is built for a table of (name, nominal density, formula) entries, ``materials=`` (default: the seven default
anchors; ``build_solver_library`` uses ``cross_sections.material_table(config.materials)``). The builder sorts the
table by nominal attenuation at the spectrum-mean energy (``sort_material_table``; a table already in that order is
unchanged) and asserts at every fine energy that the order holds (``check_material_order``: no two materials cross).
The cache key holds the sorted table (names, nominal densities, formulas), so different sets never share a cache entry
and the default set keeps the key of earlier versions. The default seven:
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
from .cross_sections import ELECTRON_REST_KEV, MATERIAL_CATALOGUE, MATERIALS, MultigroupLibrary, material_table

__all__ = [
    "BuildInfo",
    "MATERIAL_FORMULAS",
    "default_material_table",
    "sort_material_table",
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
    "eta_quadrature_in_q",
]

_BUILD_VERSION = "citrus-xsec-2"  # 2: the seven-material table; names, nominal densities and formulas in the key
_TWO_PI = 2.0 * math.pi

# Compositions (formula strings in XrayPhysics syntax: 'a*F1 + b*F2' with a, b mass fractions of the compounds F1, F2;
# a bare formula is a stoichiometric compound) of every catalogue material, `cross_sections.MATERIAL_CATALOGUE`
MATERIAL_FORMULAS: dict[str, str] = {name: formula for name, (_, formula) in MATERIAL_CATALOGUE.items()}

MaterialTable = Sequence[tuple[str, float, str]]  # (name, nominal density g/cm^3, XrayPhysics formula) per material


def default_material_table() -> tuple[tuple[str, float, str], ...]:
    """The seven default anchors as (name, nominal density, formula): ``cross_sections.material_table(MATERIALS)``
    (``cross_sections.MATERIAL_TABLE`` with ``MATERIAL_FORMULAS``)."""
    return material_table(MATERIALS)


def _material_table(materials: Optional[MaterialTable]) -> tuple[tuple[str, float, str], ...]:
    t = default_material_table() if materials is None else tuple((str(n), float(d), str(f)) for n, d, f in materials)
    if not t:
        raise ValueError("empty material table")
    if len({n for n, _, _ in t}) != len(t):
        raise ValueError(f"duplicate material names in {[n for n, _, _ in t]}")
    return t

# Fixed q = sin(theta/2) breakpoints of the angular quadrature (dense near forward scatter).
_Q_BREAKPOINTS = np.array(
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

    ``sigma(kind, Z, E)`` with kind in {"total", "photoelectric", "compton", "rayleigh"}: E [K] -> [K].
    ``pdf_raw(kind, Z, E, eta)`` with kind in {"compton", "rayleigh"}: E scalar, eta [N] -> [N]; an
    unnormalized per-steradian angular shape (normalized by the caller).
    ``library_pdf_normalization(kind, Z, E)``: XrayPhysics' own sphere normalization of ``pdf_raw``.
    ``.name`` ("xrayphysics", part of the cache key) and ``.reason`` (where it was loaded from).
    """

    name = "xrayphysics"

    def __init__(self) -> None:
        import ctypes

        from xrayphysics import xrayPhysics  # type: ignore

        xray_physics = xrayPhysics()
        shared_library = getattr(xray_physics, "libxrayphysics", None)
        if shared_library is None:
            raise ImportError("xrayphysics imported but libxrayphysics shared library did not load")
        self._xray_physics = xray_physics
        c_float = ctypes.c_float
        self._cross_section_functions: dict[str, Callable] = {}
        for kind, function_name in (("total", "sigma"), ("photoelectric", "sigmaPE"), ("compton", "sigmaCS"),
                                    ("rayleigh", "sigmaRS")):
            function = getattr(shared_library, function_name)
            function.argtypes = [c_float, c_float]
            function.restype = c_float
            self._cross_section_functions[kind] = function
        self._distribution_functions: dict[str, Callable] = {}
        self._normalization_functions: dict[str, Callable] = {}
        for kind, function_name in (("compton", "incoherentScatterDistribution"),
                                    ("rayleigh", "coherentScatterDistribution")):
            function = getattr(shared_library, function_name)
            function.argtypes = [c_float, c_float, c_float]
            function.restype = c_float
            self._distribution_functions[kind] = function
            normalization = getattr(shared_library, function_name + "_normalizationFactor")
            normalization.argtypes = [c_float, c_float]
            normalization.restype = c_float
            self._normalization_functions[kind] = normalization
        self._cross_section_scale = float(xray_physics.cross_section_scalar())
        import xrayphysics as xray_physics_module  # type: ignore

        self.reason = f"XrayPhysics (EPDL97) loaded from {os.path.dirname(xray_physics_module.__file__)}"

    def atomic_mass(self, Z: int) -> float:
        return float(self._xray_physics.atomicMass(Z))

    def sigma(self, kind: str, Z: int, E: np.ndarray) -> np.ndarray:
        function = self._cross_section_functions[kind]
        Z_float = float(Z)
        E = np.asarray(E, dtype=np.float64).ravel()
        out = np.fromiter((function(Z_float, float(e)) for e in E), dtype=np.float64, count=E.size)
        return out * self._cross_section_scale

    def pdf_raw(self, kind: str, Z: int, E: float, eta: np.ndarray) -> np.ndarray:
        function = self._distribution_functions[kind]
        theta_deg = np.degrees(np.arccos(np.clip(np.asarray(eta, dtype=np.float64), -1.0, 1.0)))
        theta_deg = np.maximum(theta_deg, _THETA_MIN_DEG)
        Z_float, E_float = float(Z), float(E)
        return np.fromiter((function(Z_float, E_float, float(t)) for t in theta_deg), dtype=np.float64,
                           count=theta_deg.size)

    def library_pdf_normalization(self, kind: str, Z: int, E: float) -> float | None:
        return float(self._normalization_functions[kind](float(Z), float(E)))

    def spectrum_125kvp(self) -> tuple[np.ndarray, np.ndarray]:
        xray_physics = self._xray_physics
        # 1 keV grid: XrayPhysics adds characteristic-line amplitudes into single samples, which is
        # calibrated as counts per 1 keV bin; we interpret the result as a density per keV.
        gammas = np.arange(1.0, 126.0, 1.0, dtype=np.float32)
        energies, spectrum = xray_physics.simulateSpectra(125.0, takeOffAngle=11.0, Z=74, gammas=gammas)
        filter_transmission = xray_physics.filterResponse("Ti", 4.506, 0.089, energies)
        return (np.asarray(energies, dtype=np.float64),
                np.asarray(spectrum, dtype=np.float64) * np.asarray(filter_transmission, dtype=np.float64))


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
    text = formula.replace(" ", "")
    terms = text.split("+")
    mass_fractions: dict[int, float] = {}
    total_fraction = 0.0
    parsed: list[tuple[float, str]] = []
    for t in terms:
        match = re.match(r"^([0-9.eE\-]*)\*?([A-Z].*)$", t)
        if match is None:
            raise ValueError(f"cannot parse material term {t!r} in {formula!r}")
        fraction = float(match.group(1)) if match.group(1) else 1.0
        if fraction <= 0:
            raise ValueError(f"non-positive fraction in {formula!r}")
        parsed.append((fraction, match.group(2)))
        total_fraction += fraction
    for fraction, compound in parsed:
        counts: dict[int, float] = {}
        for symbol, count in re.findall(r"([A-Z][a-z]?)([0-9.]*)", compound):
            if symbol not in _SYMBOL_TO_Z:
                raise ValueError(f"unknown element {symbol!r} in {formula!r}")
            Z = _SYMBOL_TO_Z[symbol]
            counts[Z] = counts.get(Z, 0.0) + (float(count) if count else 1.0)
        mass = sum(n * atomic_mass(Z) for Z, n in counts.items())
        for Z, n in counts.items():
            mass_fractions[Z] = mass_fractions.get(Z, 0.0) + fraction / total_fraction * n * atomic_mass(Z) / mass
    total = sum(mass_fractions.values())
    return {Z: w / total for Z, w in sorted(mass_fractions.items())}


def material_composition(backend: _XrayPhysicsBackend,
                         materials: Optional[MaterialTable] = None) -> dict[str, dict[int, float]]:
    """Element mass fractions per material (table order; default the seven default anchors), keyed by name."""
    return {n: parse_formula(f, backend.atomic_mass) for n, _, f in _material_table(materials)}


def _mean_energy(config: SolverConfig, spectrum: tuple[np.ndarray, np.ndarray], fine_bin_width_kev: float) -> float:
    """Spectrum-mean photon energy (keV) over the group structure of ``config`` on the fine energy quadrature of the
    build: sum_e w_e E_e / sum_e w_e with w_e the exact spectrum integral over fine bin e, the effective energy of the
    library's density model (``citrus.utilities.effective_energy``, 63.03 keV for the default 125 kVp spectrum)."""
    edges = make_group_edges(config, spectrum)
    E, _, bin_low, bin_high = fine_energy_grid(edges, fine_bin_width_kev)
    density = _Spectrum(*spectrum)
    photons = density.cumulative_integral(bin_high) - density.cumulative_integral(bin_low)
    if not photons.sum() > 0:
        raise ValueError("spectrum has no photons inside the group structure")
    return float((photons * E).sum() / photons.sum())


def sort_material_table(config: SolverConfig, materials: Optional[MaterialTable] = None,
                        spectrum: tuple[np.ndarray, np.ndarray] | None = None,
                        backend: _XrayPhysicsBackend | None = None, *,
                        fine_bin_width_kev: float = 0.5) -> tuple[tuple[str, float, str], ...]:
    """The material table ((name, nominal density g/cm^3, formula) per material; None: the seven default anchors)
    sorted by nominal attenuation rho^nom mu/rho(E_eff) at the spectrum-mean energy E_eff of the build
    (``_mean_energy``; elemental total mass attenuation of the backend mixed by mass fraction; a stable sort, so a
    table already in that order is returned as it is). Whether the order holds at every energy of the library is
    checked by the build (``check_material_order``). Needs the backend (XrayPhysics)."""
    table = _material_table(materials)
    backend = resolve_backend() if backend is None else backend
    spectrum_arrays = _spectrum_or_default(spectrum, backend)
    energy = np.array([_mean_energy(config, spectrum_arrays, fine_bin_width_kev)])
    composition = material_composition(backend, table)
    attenuation = [density * sum(w * float(backend.sigma("total", Z, energy)[0]) for Z, w in composition[name].items())
                   for name, density, _ in table]
    order = sorted(range(len(table)), key=lambda i: attenuation[i])
    return tuple(table[i] for i in order)


def check_material_order(fine_mu_t: np.ndarray, nominal_density: np.ndarray, names=None) -> None:
    """Raise ValueError unless the nominal attenuations rho_m^nom m_m(E) [M, E] increase strictly with the material
    index at EVERY fine energy (the fixed bracketing order of the density model,
    ``citrus.utilities.attenuation_to_densities``)."""
    attenuation = np.asarray(nominal_density, dtype=np.float64)[:, None] * np.asarray(fine_mu_t, dtype=np.float64)
    bad = np.nonzero(~(np.diff(attenuation, axis=0) > 0).all(axis=1))[0]
    if bad.size:
        i = int(bad[0])
        material_names = list(names) if names is not None else [str(k) for k in range(attenuation.shape[0])]
        raise ValueError(f"materials {material_names[i]!r} and {material_names[i + 1]!r} cross in attenuation at "
                         f"nominal density inside the library's energy range; reorder the material table")


# =============================================================================================
# Spectrum and groups
# =============================================================================================

def load_spectrum_txt(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column text spectrum (energy keV, photon density per keV); '#' comments.

    Commas or whitespace separate columns. Returns (energies [K] ascending, weights [K]) float64.
    """
    rows: list[tuple[float, float]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = [p for p in re.split(r"[,\s;]+", line) if p]
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except (ValueError, IndexError):
                continue  # header line
    table = np.array(rows, dtype=np.float64)
    order = np.argsort(table[:, 0])
    return table[order, 0], table[order, 1]


def default_spectrum_125kvp(backend: _XrayPhysicsBackend | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Default kV source: W anode, 125 kV, 11 deg take-off, 0.89 mm Ti filter.

    XrayPhysics ``simulateSpectra`` x ``filterResponse('Ti', 4.506, 0.089)`` on a 1 keV grid (backend: a loaded
    ``resolve_backend()`` object, None loads one).
    Returns (energies_kev [K], weights [K]) with weights a photon density per keV (arbitrary scale).
    """
    backend = resolve_backend() if backend is None else backend
    return backend.spectrum_125kvp()


class _Spectrum:
    """Piecewise-linear spectral density s(E) (zero outside the samples) with exact integrals."""

    def __init__(self, energies: np.ndarray, weights: np.ndarray) -> None:
        energies = np.asarray(energies, dtype=np.float64).ravel()
        weights = np.asarray(weights, dtype=np.float64).ravel()
        if energies.size != weights.size or energies.size < 2:
            raise ValueError("spectrum needs >= 2 (energy, weight) samples")
        o = np.argsort(energies)
        self.energies, self.weights = energies[o], np.clip(weights[o], 0.0, None)
        seg = 0.5 * (self.weights[1:] + self.weights[:-1]) * np.diff(self.energies)
        self.cumulative = np.concatenate([[0.0], np.cumsum(seg)])

    def __call__(self, E: np.ndarray) -> np.ndarray:
        return np.interp(E, self.energies, self.weights, left=0.0, right=0.0)

    def cumulative_integral(self, E: np.ndarray) -> np.ndarray:
        """int_{-inf}^{E} s(E') dE' (exact for the piecewise-linear interpolant)."""
        E = np.clip(np.asarray(E, dtype=np.float64), self.energies[0], self.energies[-1])
        i = np.clip(np.searchsorted(self.energies, E, side="right") - 1, 0, self.energies.size - 2)
        x0, x1 = self.energies[i], self.energies[i + 1]
        y0, y1 = self.weights[i], self.weights[i + 1]
        t = E - x0
        slope = (y1 - y0) / (x1 - x0)
        return self.cumulative[i] + y0 * t + 0.5 * slope * t * t

    def max_energy(self) -> float:
        """Upper end of the support of s (first zero sample after the last positive one)."""
        pos = np.nonzero(self.weights > 0)[0]
        if pos.size == 0:
            raise ValueError("spectrum is identically zero")
        j = int(pos[-1])
        return float(self.energies[min(j + 1, self.energies.size - 1)])


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
    spectrum_density = _Spectrum(*spectrum)
    energy_max, energy_min = spectrum_density.max_energy(), float(config.energy_min_kev)
    if not energy_max > energy_min:
        raise ValueError(f"spectrum max energy {energy_max} <= energy_min_kev {energy_min}")
    grid = np.linspace(energy_min, energy_max, 200001)
    cumulative = (spectrum_density.cumulative_integral(grid)
                  - spectrum_density.cumulative_integral(np.array([energy_min]))[0])
    if cumulative[-1] <= 0:
        raise ValueError("spectrum has no photons in [energy_min_kev, kVp]")
    targets = cumulative[-1] * np.arange(1, G) / G
    indices = np.clip(np.searchsorted(cumulative, targets, side="left"), 1, grid.size - 1)
    c0, c1 = cumulative[indices - 1], cumulative[indices]
    t = np.where(c1 > c0, (targets - c0) / np.where(c1 > c0, c1 - c0, 1.0), 0.0)
    inner = grid[indices - 1] + t * (grid[indices] - grid[indices - 1])
    ascending = np.concatenate([[energy_min], inner, [energy_max]])
    return ascending[::-1].copy()


def fine_energy_grid(edges: np.ndarray,
                     fine_bin_width_kev: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cut every group into equal bins of width <= fine_bin_width_kev.

    Returns (E_mid [K], group [K] int, lo [K], hi [K]) with bins ordered by group then
    ascending energy; bins never straddle a group edge.
    """
    midpoints, groups, lows, highs = [], [], [], []
    for g in range(edges.size - 1):
        high, low = float(edges[g]), float(edges[g + 1])
        n = max(1, int(math.ceil((high - low) / fine_bin_width_kev - 1e-9)))
        bin_edges = np.linspace(low, high, n + 1)
        lows.append(bin_edges[:-1])
        highs.append(bin_edges[1:])
        midpoints.append(0.5 * (bin_edges[:-1] + bin_edges[1:]))
        groups.append(np.full(n, g, dtype=np.int64))
    return np.concatenate(midpoints), np.concatenate(groups), np.concatenate(lows), np.concatenate(highs)


def group_of_energy(E: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Group index of energies E (edges descending); -1 below the lowest edge, 0 at/above top.

    A group g holds edges[g+1] <= E < edges[g].
    """
    ascending = edges[::-1]
    j = np.searchsorted(ascending, np.asarray(E, dtype=np.float64), side="right") - 1
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


def eta_quadrature_in_q(q_breakpoints: np.ndarray, gauss_legendre_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise Gauss-Legendre in q = sqrt((1-eta)/2) over the given sorted q breakpoints.

    Returns (eta [N], w [N]) such that sum w f(eta) ~= int f(eta) d eta over the eta range of the
    breakpoints (w includes the Jacobian d eta = 4 q dq). Nodes of a piece lie strictly inside it.
    """
    x, node_weights = np.polynomial.legendre.leggauss(gauss_legendre_nodes)
    a, b = q_breakpoints[:-1], q_breakpoints[1:]
    keep = b > a
    a, b = a[keep], b[keep]
    half_width = 0.5 * (b - a)
    q = (half_width[:, None] * x[None, :] + 0.5 * (a + b)[:, None]).ravel()
    w = (half_width[:, None] * node_weights[None, :]).ravel() * 4.0 * q
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
      sigma_compton_group / sigma_rayleigh_group / sigma_photoelectric_group / sigma_total_group [M, G]
                                    group-weighted mass cross sections (w(E) weights)
      fine_energy [K], fine_group [K], fine_weight [K]  the energy quadrature of the group constants
                                    (spectrum weights summing to 1 within each group; NOT the library's
                                    fine_weight, which is the spectrum normalized over all groups, zero
                                    bins dropped)
      flat_fallback_groups          groups whose spectrum weight was zero -> flat weight used
      pdf_normalization_max_relative_deviation  max |N_backend / N_quadrature - 1| over elements, energies,
                                    processes
      rayleigh_l [M, G, L+1], rayleigh_eta [M, G, N_eta]
                                    the within-group Rayleigh parts of mu_s_l / mu_s_eta (diagonal)
    """

    backend: str
    backend_reason: str
    composition: dict[str, dict[int, float]]
    compton_below_cutoff: np.ndarray
    sigma_compton_group: np.ndarray
    sigma_rayleigh_group: np.ndarray
    sigma_photoelectric_group: np.ndarray
    sigma_total_group: np.ndarray
    fine_energy: np.ndarray
    fine_group: np.ndarray
    fine_weight: np.ndarray
    flat_fallback_groups: list[int] = field(default_factory=list)
    pdf_normalization_max_relative_deviation: float | None = None
    rayleigh_l: np.ndarray | None = None
    rayleigh_eta: np.ndarray | None = None


def _spectrum_or_default(spectrum: tuple[np.ndarray, np.ndarray] | None,
                         backend: _XrayPhysicsBackend) -> tuple[np.ndarray, np.ndarray]:
    if spectrum is None:
        spectrum = backend.spectrum_125kvp()
    e, w = spectrum
    return np.asarray(e, dtype=np.float64).ravel(), np.asarray(w, dtype=np.float64).ravel()


def compute_library(
    config: SolverConfig,
    spectrum: tuple[np.ndarray, np.ndarray] | None = None,
    backend: _XrayPhysicsBackend | None = None,
    *,
    fine_bin_width_kev: float = 0.5,
    gauss_legendre_nodes: int = 32,
    materials: Optional[MaterialTable] = None,
) -> tuple[MultigroupLibrary, BuildInfo]:
    """Build the library without caching; also return diagnostics.

    Args:
      config: uses num_groups, group_edges_kev, energy_min_kev, legendre_order, num_eta_table.
      spectrum: (energies keV [K], photon density per keV [K]); None -> default 125 kVp spectrum.
      backend: a loaded ``resolve_backend()`` object (None loads one).
      fine_bin_width_kev: maximum fine energy bin width (<= 0.5 keV for production).
      gauss_legendre_nodes: Gauss-Legendre nodes per angular piece (>= 32 for production).
      materials: (name, nominal density g/cm^3, formula) per material, in any order (sorted here by nominal attenuation
        at the spectrum-mean energy, ``sort_material_table``; ``check_material_order`` raises if two materials cross
        inside the library's energy range); None: the seven default anchors (``default_material_table``).

    Returns:
      (MultigroupLibrary with float64 CPU tensors, BuildInfo).
    """
    backend = resolve_backend() if backend is None else backend
    spectrum_energies, spectrum_weights = _spectrum_or_default(spectrum, backend)
    spectrum_density = _Spectrum(spectrum_energies, spectrum_weights)
    edges = make_group_edges(config, (spectrum_energies, spectrum_weights))
    G = edges.size - 1
    L = int(config.legendre_order)
    table = sort_material_table(config, materials, (spectrum_energies, spectrum_weights), backend,
                                fine_bin_width_kev=fine_bin_width_kev)
    names = tuple(n for n, _, _ in table)
    M = len(names)
    m_e = ELECTRON_REST_KEV

    # ---- energy quadrature ----------------------------------------------------------------
    E, fine_groups, bin_low, bin_high = fine_energy_grid(edges, fine_bin_width_kev)
    K = E.size
    # exact spectrum photons per fine bin [K]
    photons_per_bin = spectrum_density.cumulative_integral(bin_high) - spectrum_density.cumulative_integral(bin_low)
    flat_fallback: list[int] = []
    within_group_weight = np.empty(K)  # within-group spectrum weight (flat where the group has no photons)
    for g in range(G):
        in_group = fine_groups == g
        total = photons_per_bin[in_group].sum()
        if total <= 0:
            flat_fallback.append(g)
            within_group_weight[in_group] = (bin_high - bin_low)[in_group] / (bin_high - bin_low)[in_group].sum()
        else:
            within_group_weight[in_group] = photons_per_bin[in_group] / total
    if flat_fallback:
        warnings.warn(f"zero spectrum in groups {flat_fallback}; flat within-group weight used", RuntimeWarning)

    group_energy = np.array([np.sum(within_group_weight[fine_groups == g] * E[fine_groups == g]) for g in range(G)])
    photons_per_group = np.array([photons_per_bin[fine_groups == g].sum() for g in range(G)])
    if photons_per_group.sum() <= 0:
        raise ValueError("spectrum has no photons inside the group structure")
    spectrum_fraction = photons_per_group / photons_per_group.sum()

    # ---- materials and point-wise cross sections ------------------------------------------
    composition = material_composition(backend, table)
    elements = sorted({Z for c in composition.values() for Z in c})
    mass_fractions = np.array([[composition[m].get(Z, 0.0) for Z in elements] for m in names])  # [M, NZ]
    element_sigma = {k: np.stack([backend.sigma(k, Z, E) for Z in elements])
                     for k in ("total", "photoelectric", "compton", "rayleigh")}  # [NZ, K]
    material_sigma = {k: mass_fractions @ v for k, v in element_sigma.items()}  # [M, K]

    def group_average(x: np.ndarray) -> np.ndarray:  # x [M, K] -> [M, G]
        out = np.zeros((x.shape[0], G))
        np.add.at(out.T, fine_groups, (x * within_group_weight).T)
        return out

    mu_t = group_average(material_sigma["total"])
    mu_pe = group_average(material_sigma["photoelectric"])

    # ---- angular quadratures ---------------------------------------------------------------
    eta_reference, w_reference = eta_quadrature_in_q(_Q_BREAKPOINTS, gauss_legendre_nodes)
    legendre_reference = legendre_table(eta_reference, L)  # [L+1, Nr]
    N_eta = int(config.num_eta_table)
    eta_grid = np.linspace(-1.0, 1.0, N_eta)

    mu_s_l = np.zeros((M, G, G, L + 1))
    compton_below = np.zeros((M, G))
    mu_s_eta = np.zeros((M, G, G, N_eta))
    rayleigh_l = np.zeros((M, G, L + 1))
    rayleigh_eta = np.zeros((M, G, N_eta))
    # Table kinematics: at fixed eta, E' lands in [edges[j+1], edges[j]) iff E lies in
    # [E_b(edges[j+1], eta), E_b(edges[j], eta)) with E_b(e, eta) = 1 / (1/e - (1 - eta)/m_e)
    # (+inf when 1/e <= (1 - eta)/m_e). Each fine bin contributes the fraction of its width inside
    # that interval, so the tabulated transfer is continuous in eta (a midpoint indicator would
    # give a staircase with 0.5 keV steps).
    inv_b = 1.0 / edges[:, None] - (1.0 - eta_grid[None, :]) / m_e  # [G+1, N_eta]
    E_b = np.where(inv_b > 0, 1.0 / np.where(inv_b > 0, inv_b, 1.0), np.inf)
    normalization_deviation = 0.0
    have_library_normalization = True

    for k in range(K):
        energy_k, g, weight_k = float(E[k]), int(fine_groups[k]), float(within_group_weight[k])
        # Compton eta pieces: kinematic group edges below Ek plus fixed q breaks.
        edges_below = edges[edges < energy_k]
        q_edges = np.sqrt(np.clip(m_e * (1.0 / edges_below - 1.0 / energy_k), 0.0, None) / 2.0)
        q_breaks = np.unique(np.concatenate([_Q_BREAKPOINTS, q_edges[q_edges < 1.0]]))
        eta_compton, w_compton = eta_quadrature_in_q(q_breaks, gauss_legendre_nodes)
        energy_out = energy_k / (1.0 + (energy_k / m_e) * (1.0 - eta_compton))
        group_out = group_of_energy(energy_out, edges)  # -1 = below cutoff
        legendre_compton = legendre_table(eta_compton, L)

        compton_pieces = np.zeros((M, eta_compton.size))
        rayleigh_reference = np.zeros((M, eta_reference.size))
        compton_table = np.zeros((M, N_eta))
        rayleigh_table = np.zeros((M, N_eta))
        for element_index, Z in enumerate(elements):
            compton_shape = backend.pdf_raw("compton", Z, energy_k, eta_reference)
            rayleigh_shape = backend.pdf_raw("rayleigh", Z, energy_k, eta_reference)
            compton_norm = _TWO_PI * float(np.dot(w_reference, compton_shape))
            rayleigh_norm = _TWO_PI * float(np.dot(w_reference, rayleigh_shape))
            for kind, quadrature_norm in (("compton", compton_norm), ("rayleigh", rayleigh_norm)):
                library_norm = backend.library_pdf_normalization(kind, Z, energy_k)
                if library_norm is None:
                    have_library_normalization = False
                else:
                    normalization_deviation = max(normalization_deviation, abs(library_norm / quadrature_norm - 1.0))
            # [M] scale of the raw shape
            compton_scale = mass_fractions[:, element_index] * element_sigma["compton"][element_index, k] / compton_norm
            rayleigh_scale = (mass_fractions[:, element_index] * element_sigma["rayleigh"][element_index, k]
                              / rayleigh_norm)
            compton_shape_pieces = backend.pdf_raw("compton", Z, energy_k, eta_compton)
            compton_pieces += compton_scale[:, None] * compton_shape_pieces[None, :]
            rayleigh_reference += rayleigh_scale[:, None] * rayleigh_shape[None, :]
            compton_table += compton_scale[:, None] * backend.pdf_raw("compton", Z, energy_k, eta_grid)[None, :]
            rayleigh_table += rayleigh_scale[:, None] * backend.pdf_raw("rayleigh", Z, energy_k, eta_grid)[None, :]

        # Legendre moments: Compton binned by outgoing group (+1 shift; column 0 = below cutoff)
        contribution = (_TWO_PI * weight_k * (compton_pieces * w_compton[None, :])[:, None, :]
                        * legendre_compton[None, :, :])  # [M, L+1, Nc]
        moments_by_group = np.zeros((M, L + 1, G + 1))
        np.add.at(moments_by_group.transpose(2, 0, 1), group_out + 1, contribution.transpose(2, 0, 1))
        mu_s_l[:, g, :, :] += moments_by_group[:, :, 1:].transpose(0, 2, 1)
        compton_below[:, g] += moments_by_group[:, 0, 0]
        # [M, L+1]
        rayleigh_moments = _TWO_PI * weight_k * (rayleigh_reference * w_reference[None, :]) @ legendre_reference.T
        mu_s_l[:, g, g, :] += rayleigh_moments
        rayleigh_l[:, g, :] += rayleigh_moments

        # Direct per-steradian tables
        bin_low_k, bin_high_k = float(bin_low[k]), float(bin_high[k])
        fraction = (np.clip(np.minimum(bin_high_k, E_b[g:G]) - np.maximum(bin_low_k, E_b[g + 1:G + 1]), 0.0, None)
                    / (bin_high_k - bin_low_k))
        mu_s_eta[:, g, g:, :] += weight_k * compton_table[:, None, :] * fraction[None, :, :]
        mu_s_eta[:, g, g, :] += weight_k * rayleigh_table
        rayleigh_eta[:, g, :] += weight_k * rayleigh_table

    info = BuildInfo(
        backend=backend.name,
        backend_reason=backend.reason,
        composition=composition,
        compton_below_cutoff=compton_below,
        sigma_compton_group=group_average(material_sigma["compton"]),
        sigma_rayleigh_group=group_average(material_sigma["rayleigh"]),
        sigma_photoelectric_group=mu_pe.copy(),
        sigma_total_group=mu_t.copy(),
        fine_energy=E,
        fine_group=fine_groups,
        fine_weight=within_group_weight,
        flat_fallback_groups=flat_fallback,
        pdf_normalization_max_relative_deviation=normalization_deviation if have_library_normalization else None,
        rayleigh_l=rayleigh_l,
        rayleigh_eta=rayleigh_eta,
    )
    to_tensor = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)  # noqa: E731
    # fine-energy attenuation table: spectrum weights normalized over the whole group structure (the same
    # denominator as spectrum_fraction), zero-weight bins dropped
    fine_weights = photons_per_bin / photons_per_bin.sum()
    keep = fine_weights > 0
    nominal = np.array([d for _, d, _ in table], dtype=np.float64)
    if M > 1:
        check_material_order(material_sigma["total"][:, keep], nominal, names)
    library = MultigroupLibrary(
        group_edges=to_tensor(edges),
        group_energy=to_tensor(group_energy),
        spectrum_fraction=to_tensor(spectrum_fraction),
        mu_t=to_tensor(mu_t),
        mu_pe=to_tensor(mu_pe),
        mu_s_l=to_tensor(mu_s_l),
        eta_grid=to_tensor(eta_grid),
        mu_s_eta=to_tensor(mu_s_eta),
        fine_energy=to_tensor(E[keep]),
        fine_mu_t=to_tensor(material_sigma["total"][:, keep]),
        fine_weight=to_tensor(fine_weights[keep]),
        fine_group=torch.as_tensor(np.ascontiguousarray(fine_groups[keep]), dtype=torch.int64),
        materials=names,
        nominal_density=to_tensor(nominal),
    )
    return library, info


def _cache_key(config: SolverConfig, spectrum: tuple[np.ndarray, np.ndarray], backend_name: str,
               fine_bin_width_kev: float, gauss_legendre_nodes: int, materials: Optional[MaterialTable] = None) -> str:
    fields = {
        "version": _BUILD_VERSION,
        "backend": backend_name,
        "num_groups": int(config.num_groups),
        "group_edges_kev": None if config.group_edges_kev is None else [float(x) for x in config.group_edges_kev],
        "energy_min_kev": float(config.energy_min_kev),
        "legendre_order": int(config.legendre_order),
        "num_eta_table": int(config.num_eta_table),
        # the on-disk key names of citrus <= 0.3.0 (renaming them would orphan every cached library)
        "de_kev": float(fine_bin_width_kev),
        "n_gl": int(gauss_legendre_nodes),
        "materials": [list(m) for m in _material_table(materials)],  # names, nominal densities and formulas, in order
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
    fine_bin_width_kev: float = 0.5,
    gauss_legendre_nodes: int = 32,
    materials: Optional[MaterialTable] = None,
) -> MultigroupLibrary:
    """Build (or load from cache) the multigroup library for ``config`` and the material table ``materials`` (None:
    the seven default anchors, ``default_material_table``; else (name, nominal density, formula) per material, e.g.
    ``cross_sections.material_table(names)``, in any order: sorted by nominal attenuation, ``sort_material_table``).

    Shapes of the result (M materials, 7 by default; G = config.num_groups, L = config.legendre_order):
      group_edges [G+1] desc, group_energy [G], spectrum_fraction [G], mu_t/mu_pe [M, G],
      mu_s_l [M, G, G, L+1], eta_grid [num_eta_table], mu_s_eta [M, G, G, num_eta_table],
      fine_energy [E], fine_mu_t [M, E], fine_weight [E], fine_group [E] (int64; E ~ (kVp - E_min) /
      fine_bin_width_kev); float64 CPU tensors, per unit density (cm^2/g, cm^2/g/sr); plus materials (names) and
      nominal_density [M].

    The cache file is ``<cache_dir>/mglib_<hash>.pt`` keyed by the build version, the relevant config
    fields, the spectrum arrays, the backend name, the numerical parameters and the SORTED material table (names,
    nominal densities, formulas: every set of materials has its own entry, any order of the same set the same one, and
    the default seven keep the key of earlier versions; the two-material caches of citrus < 0.1.0 have other keys and
    are never loaded).
    ``cache_dir="default"`` is `default_cache_dir()`; ``cache_dir=None`` disables the cache. A cache file that cannot be
    loaded as a complete library (e.g. one lacking the fine-energy table) is rebuilt and overwritten.
    """
    if cache_dir == "default":
        cache_dir = default_cache_dir()
    backend = resolve_backend()
    spectrum_arrays = _spectrum_or_default(spectrum, backend)
    table = sort_material_table(config, materials, spectrum_arrays, backend, fine_bin_width_kev=fine_bin_width_kev)
    path = None
    if cache_dir is not None:
        key = _cache_key(config, spectrum_arrays, backend.name, fine_bin_width_kev, gauss_legendre_nodes, table)
        path = os.path.join(cache_dir, f"mglib_{key}.pt")
        if os.path.isfile(path):
            try:
                library = MultigroupLibrary.load(path)
                names = tuple(n for n, _, _ in table)
                if tuple(library.materials) != names:
                    raise KeyError(f"cached materials {tuple(library.materials)} != {names}")
                return library
            except (KeyError, TypeError, ValueError) as exc:
                warnings.warn(f"rebuilding the library cache {path}: {exc}", RuntimeWarning)
    library, _ = compute_library(config, spectrum_arrays, backend, fine_bin_width_kev=fine_bin_width_kev,
                                 gauss_legendre_nodes=gauss_legendre_nodes, materials=table)
    if path is not None:
        os.makedirs(cache_dir, exist_ok=True)  # type: ignore[arg-type]
        tmp = path + ".tmp"
        library.save(tmp)
        os.replace(tmp, path)
    return library
