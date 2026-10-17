"""Build the multigroup cross-section library of the scatter solver.

Fills a :class:`citrus.scatter_solver.cross_sections.MultigroupLibrary` for the reference materials of the density model
(any subset of ``cross_sections.MATERIAL_CATALOGUE``, ``SolverConfig.materials``; by default the seven of
``cross_sections.MATERIALS``: air, lung, adipose, water, muscle, soft tissue, cortical bone) from point-wise photon
interaction data.

Data: XrayPhysics (github.com/kylechampley/XrayPhysics, EPDL97 tables), loaded by ``resolve_backend()``.

Physics summary (all coefficients per unit density, cm^2/g):
  * Groups g = 0..G-1 from HIGHEST to LOWEST energy; ``group_edges`` [G+1] descending.
  * Group edges (``make_group_edges``): explicit edges if given, else the rule ``SolverConfig.group_edges_method``:
    equal area under an estimate of the total (source plus scattered) photon fluence spectrum in a water object of
    chord ``grouping_object_size_cm`` (``scattered_fluence_spectrum``; the default), equal source photon counts
    ("equal_fluence") or geometric spacing ("log").
  * Within-group weight w(E), normalized to 1 within each group: the source spectrum, or for the "scattered_fluence"
    rule the estimated total fluence of that rule (``_within_group_weights``). A group with zero weight falls back to
    a flat weight (a ``RuntimeWarning`` is raised and the group is listed in ``BuildInfo.flat_fallback_groups``).
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
  * Table nodes (``SolverConfig.angular_table_grid``): "uniform" places the ``num_eta_table`` nodes equally in eta;
    "adaptive" (default) places them by a map u = h(eta) built from the transfer curves of the catalogue's water and
    cortical bone (``ANGULAR_MAP_MATERIALS``, whatever the library's materials, so that every material set shares the
    nodes and a material's rows do not depend on the set) in the library's groups (``angular_map``): the node density
    follows max over the curves of sqrt(|d^2 f / d q^2| / max f) in
    q = sin(theta/2), the density that equalizes the relative error of linear interpolation, plus a floor, so the
    Rayleigh and Compton forward peaks (a few 1e-7 wide in 1 - eta at MeV energies) are resolved with the same node
    count. The map is stored as ``eta_map`` and evaluated by ``cross_sections.angular_table_position``.
  * Fine-energy attenuation table (``fine_energy``, ``fine_mu_t``, ``fine_weight``, ``fine_group``):
    the same fine quadrature, i.e. the total mass attenuation of each material at every bin midpoint
    and the exact spectrum integral over every bin normalized to 1 over the whole group structure
    (bins with zero spectrum are dropped), so sum over a group's bins = ``spectrum_fraction[g]``.
    The uncollided fluence and the primary use it instead of the group-averaged ``mu_t`` (exact in
    energy: no within-group beam hardening error; docs/source/design_notes.rst, "Cross sections").

Materials. The compositions and nominal densities of every material the builder knows are in
``cross_sections.MATERIAL_CATALOGUE`` (sources there); ``MATERIAL_FORMULAS`` maps each name to its composition. A
library is built for a table of (name, nominal density, formula) entries, ``materials=`` (default: the seven default
reference materials; ``build_solver_library`` uses ``cross_sections.material_table(config.materials)``). The builder
sorts the
table by nominal attenuation at the spectrum-mean energy (``sort_material_table``; a table already in that order is
unchanged) and asserts at every fine energy that the order holds (``check_material_order``: no two materials cross).
The cache key holds the sorted table (names, nominal densities, formulas), so different sets never share a cache entry.
The default seven:
  * air     = XrayPhysics library string 'air' (78.084*N2 + 20.946*O2 + 0.9340*Ar + 0.0397*CO2, read as mass
              fractions of the compounds exactly as XrayPhysics' parser does), 0.0012 g/cm^3.
  * lung    = XrayPhysics 'lung', 0.26 g/cm^3 (the WLAB lung; XrayPhysics lists 0.25).
  * adipose = XrayPhysics 'adipose_tissue', 0.95 g/cm^3 (XrayPhysics lists 0.92).
  * water   = 'H2O' -> with XrayPhysics atomic masses (H 1.008, O 15.999): H 0.111907, O 0.888093; 1.0 g/cm^3.
  * muscle  = ICRU-44 skeletal muscle mass fractions, 1.05 g/cm^3.
  * soft_tissue = ICRU-44 soft tissue mass fractions, 1.06 g/cm^3. Muscle lies BELOW soft tissue at every energy
              of the library, hence the order.
  * bone    = XrayPhysics library entry 'bone_cortical' (ICRP cortical bone, 1.85 g/cm^3), mass
    fractions H 0.047234, C 0.144330, N 0.041990, O 0.446096, Mg 0.002200, P 0.104970,
    S 0.003150, Ca 0.209930, Zn 0.000100 (sum 1.000000). Note this is ICRP (not ICRU-44) cortical bone; its mu/rho
    differs slightly from ICRU-44.
Other tables (any number of materials) can be passed as ``materials=`` (name, nominal density, formula).
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
from .cross_sections import (ANGULAR_MAP_KNOTS, ELECTRON_REST_KEV, ELEMENT_SYMBOLS, MATERIAL_CATALOGUE, MATERIALS,
                             MultigroupLibrary, angular_table_nodes, material_table)

__all__ = [
    "BuildInfo",
    "MATERIAL_FORMULAS",
    "default_material_table",
    "sort_material_table",
    "mass_attenuation_preview",
    "check_material_order",
    "build_library",
    "default_cache_dir",
    "compute_library",
    "default_spectrum_125kvp",
    "load_spectrum_txt",
    "make_group_edges",
    "scattered_fluence_spectrum",
    "grouping_medium_densities",
    "angular_map",
    "transfer_at_eta",
    "fine_energy_grid",
    "group_of_energy",
    "material_composition",
    "parse_formula",
    "resolve_backend",
    "legendre_table",
    "eta_quadrature_in_q",
]

# 2: the seven-material table; names, nominal densities and formulas in the key. 3 (citrus 0.6.0): the angular table
# grid (adaptive tables with eta_map), the group edge rule and the grouping object size in the key
_BUILD_VERSION = "citrus-xsec-3"
_TWO_PI = 2.0 * math.pi
# catalogue materials whose transfer curves place the nodes of an adaptive angular table (`angular_map`): low-Z soft
# tissue and calcium-bearing bone; the same for every library, so that the nodes do not depend on the material set
ANGULAR_MAP_MATERIALS = ("water", "bone")

# Compositions (formula strings in XrayPhysics syntax: 'a*F1 + b*F2' with a, b mass fractions of the compounds F1, F2;
# a bare formula is a stoichiometric compound) of every catalogue material, `cross_sections.MATERIAL_CATALOGUE`
MATERIAL_FORMULAS: dict[str, str] = {name: formula for name, (_, formula) in MATERIAL_CATALOGUE.items()}

MaterialTable = Sequence[tuple[str, float, str]]  # (name, nominal density g/cm^3, XrayPhysics formula) per material


def default_material_table() -> tuple[tuple[str, float, str], ...]:
    """The seven default reference materials as (name, nominal density, formula):
    ``cross_sections.material_table(MATERIALS)``
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

_SYMBOL_TO_Z = {s: i + 1 for i, s in enumerate(ELEMENT_SYMBOLS)}


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
    """Element mass fractions per material (table order; default the seven default reference materials), keyed by
    name."""
    return {n: parse_formula(f, backend.atomic_mass) for n, _, f in _material_table(materials)}


def _mean_energy(config: SolverConfig, spectrum: tuple[np.ndarray, np.ndarray], fine_bin_width_kev: float,
                 backend: Optional["_XrayPhysicsBackend"] = None) -> float:
    """Spectrum-mean photon energy (keV) over the group structure of ``config`` on the fine energy quadrature of the
    build: sum_e w_e E_e / sum_e w_e with w_e the exact spectrum integral over fine bin e, the effective energy of the
    library's density model (``citrus.utilities.effective_energy``, about 63 keV for the default 125 kVp spectrum).
    backend: needed by the "scattered_fluence" group rule (None loads one when needed)."""
    edges = make_group_edges(config, spectrum, backend, fine_bin_width_kev=fine_bin_width_kev)
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
    """The material table ((name, nominal density g/cm^3, formula) per material; None: the seven default reference
    materials)
    sorted by nominal attenuation rho^nom mu/rho(E_eff) at the spectrum-mean energy E_eff of the build
    (``_mean_energy``; elemental total mass attenuation of the backend mixed by mass fraction; a stable sort, so a
    table already in that order is returned as it is). Whether the order holds at every energy of the library is
    checked by the build (``check_material_order``). Needs the backend (XrayPhysics)."""
    table = _material_table(materials)
    backend = resolve_backend() if backend is None else backend
    spectrum_arrays = _spectrum_or_default(spectrum, backend)
    energy = np.array([_mean_energy(config, spectrum_arrays, fine_bin_width_kev, backend)])
    composition = material_composition(backend, table)
    attenuation = [density * sum(w * float(backend.sigma("total", Z, energy)[0]) for Z, w in composition[name].items())
                   for name, density, _ in table]
    order = sorted(range(len(table)), key=lambda i: attenuation[i])
    return tuple(table[i] for i in order)


def mass_attenuation_preview(config: SolverConfig, names: Sequence[str],
                             spectrum: tuple[np.ndarray, np.ndarray] | None = None,
                             backend: _XrayPhysicsBackend | None = None, *, energy_kev: float | None = None,
                             fine_bin_width_kev: float = 0.5) -> tuple[float, np.ndarray]:
    """(E_eff keV, total mass attenuation [len(names)] cm^2/g at E_eff) of catalogue materials WITHOUT building the
    library: E_eff = energy_kev, or the spectrum-mean energy of the build of `config` (``_mean_energy``, the value
    ``citrus.utilities.effective_energy`` gives for the built library); the attenuation from the backend's elemental
    cross sections mixed by mass fraction at E_eff (the library interpolates its fine table there instead: the two
    agree closely). For decisions taken before a lazy solver builds its library (``citrus.utilities.object_size_cm``).
    """
    backend = resolve_backend() if backend is None else backend
    spectrum_arrays = _spectrum_or_default(spectrum, backend)
    energy = (_mean_energy(config, spectrum_arrays, fine_bin_width_kev, backend) if energy_kev is None
              else float(energy_kev))
    table = material_table(tuple(names))
    composition = material_composition(backend, table)
    point = np.array([energy])
    mass = np.array([sum(w * float(backend.sigma("total", Z, point)[0]) for Z, w in composition[name].items())
                     for name, _, _ in table])
    return energy, mass


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


def _equal_area_edges(grid: np.ndarray, cumulative: np.ndarray, G: int, energy_min: float,
                      energy_max: float) -> np.ndarray:
    """Descending edges [G + 1] that cut the area of a nonnegative density into G equal parts: grid [n] ascending
    energies (keV) with grid[0] = energy_min and grid[-1] = energy_max, cumulative [n] the integral of the density from
    energy_min to each grid energy (linear interpolation in between)."""
    if cumulative[-1] <= 0:
        raise ValueError("spectrum has no photons in [energy_min_kev, kVp]")
    targets = cumulative[-1] * np.arange(1, G) / G
    indices = np.clip(np.searchsorted(cumulative, targets, side="left"), 1, grid.size - 1)
    c0, c1 = cumulative[indices - 1], cumulative[indices]
    t = np.where(c1 > c0, (targets - c0) / np.where(c1 > c0, c1 - c0, 1.0), 0.0)
    inner = grid[indices - 1] + t * (grid[indices] - grid[indices - 1])
    ascending = np.concatenate([[energy_min], inner, [energy_max]])
    return ascending[::-1].copy()


# group edges of the "scattered_fluence" rule already computed in this process: key -> edges
_EDGES_MEMO: dict[str, np.ndarray] = {}


def make_group_edges(config: SolverConfig, spectrum: tuple[np.ndarray, np.ndarray],
                     backend: Optional["_XrayPhysicsBackend"] = None, *,
                     fine_bin_width_kev: float = 0.5) -> np.ndarray:
    """Group edges [G+1] in keV, descending, spanning [energy_min_kev, kVp] (kVp = upper end of the spectrum support).

    ``config.group_edges_kev`` if given (must be strictly descending). Otherwise ``config.group_edges_method``:

    * "scattered_fluence" (default): equal area per group under the TOTAL photon fluence spectrum phi(E), source plus
      scattered photons, of an object of chord L = ``config.grouping_object_size_cm`` made of the medium
      ``grouping_medium_densities(config)`` (``scattered_fluence_spectrum``; needs the backend, None loads one). A
      grouping heuristic, not a transport result. The library then also weights the energies within each group by
      phi (``_within_group_weights``).
    * "equal_fluence": equal area under the SOURCE spectrum (equal source photon count per group; the rule of
      citrus <= 0.5, unchanged to the last bit).
    * "log": geometric spacing, edges energy_min_kev (kVp / energy_min_kev)^(k / G).

    Why the scattered fluence: "equal area under the spectrum" taken on the SOURCE spectrum is "equal_fluence". It
    resolves the source photons, but the groups carry the collided photons too, and the multiple-collision scatter
    consists of downscattered photons. At 125 kVp both spectra lie in the same range and the rules nearly agree; at
    6 MV the source has few photons below 600 keV, so equal source fluence puts one group across 10-600 keV, where most
    multiply scattered photons are, and the group-averaged attenuation of that wide group is too coarse. The total
    fluence spectrum puts edges where the photons of every collision order are.
    fine_bin_width_kev: bin width of the energy grid of the "scattered_fluence" balance.
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
    method = config.group_edges_method
    if method == "equal_fluence":
        grid = np.linspace(energy_min, energy_max, 200001)
        cumulative = (spectrum_density.cumulative_integral(grid)
                      - spectrum_density.cumulative_integral(np.array([energy_min]))[0])
        return _equal_area_edges(grid, cumulative, G, energy_min, energy_max)
    if method == "log":
        ascending = energy_min * (energy_max / energy_min) ** (np.arange(G + 1) / G)
        ascending[0], ascending[-1] = energy_min, energy_max
        return ascending[::-1].copy()
    if method != "scattered_fluence":
        raise ValueError(f"unknown group_edges_method {method!r}")
    backend = resolve_backend() if backend is None else backend
    medium = grouping_medium_densities(config)
    key = hashlib.sha256(json.dumps([backend.name, energy_min, float(config.grouping_object_size_cm), G,
                                     float(fine_bin_width_kev), [list(m) for m in medium]]).encode()
                         + np.ascontiguousarray(spectrum[0], dtype=np.float64).tobytes()
                         + np.ascontiguousarray(spectrum[1], dtype=np.float64).tobytes()).hexdigest()
    if key not in _EDGES_MEMO:
        bin_edges, photons = scattered_fluence_spectrum(spectrum, backend, energy_min_kev=energy_min,
                                                        object_size_cm=float(config.grouping_object_size_cm),
                                                        medium=medium, fine_bin_width_kev=fine_bin_width_kev)
        cumulative = np.concatenate([[0.0], np.cumsum(photons)])
        _EDGES_MEMO[key] = _equal_area_edges(bin_edges, cumulative, G, energy_min, energy_max)
    return _EDGES_MEMO[key].copy()


def _within_group_weights(config: SolverConfig, spectrum: tuple[np.ndarray, np.ndarray], backend, E: np.ndarray,
                          fine_groups: np.ndarray, bin_low: np.ndarray, bin_high: np.ndarray,
                          photons_per_bin: np.ndarray, G: int, fine_bin_width_kev: float) -> tuple[np.ndarray, list]:
    """The weights w(E) of the fine bins within their group (they sum to 1 per group) for the group constants, and the
    groups that fell back to a flat weight.

    The source spectrum (the exact photons per bin, photons_per_bin [K]) for explicit edges and the rules
    "equal_fluence" and "log" (the weighting of citrus <= 0.5); for "scattered_fluence" the estimated total fluence
    phi(E) of ``scattered_fluence_spectrum`` (its density at the bin midpoint times the bin width), the spectrum of the
    photons the groups transport: in a group far below the source energies (the lowest groups at MV) the scattered
    photons sit near the top of the group while the source spectrum may be flat or zero there, and a source weighting
    would give that group the attenuation of its lowest energies. A group whose weight is zero falls back to a flat
    weight."""
    if config.group_edges_kev is None and config.group_edges_method == "scattered_fluence":
        bin_edges, photons = scattered_fluence_spectrum(
            spectrum, backend, energy_min_kev=float(config.energy_min_kev),
            object_size_cm=float(config.grouping_object_size_cm), medium=grouping_medium_densities(config),
            fine_bin_width_kev=fine_bin_width_kev)
        index = np.clip(np.searchsorted(bin_edges, E, side="right") - 1, 0, photons.size - 1)
        raw = photons[index] / np.diff(bin_edges)[index] * (bin_high - bin_low)
    else:
        raw = photons_per_bin
    weights = np.empty(E.size)
    flat_fallback: list[int] = []
    for g in range(G):
        in_group = fine_groups == g
        total = raw[in_group].sum()
        if total <= 0:
            flat_fallback.append(g)
            weights[in_group] = (bin_high - bin_low)[in_group] / (bin_high - bin_low)[in_group].sum()
        else:
            weights[in_group] = raw[in_group] / total
    return weights, flat_fallback


def _klein_nishina_cumulative(epsilon: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Antiderivative in epsilon = E'/E of the Klein-Nishina cross section per unit epsilon (up to the factor
    pi r_e^2 / k): d sigma / d epsilon ~ epsilon + 1/epsilon - sin^2(theta), cos(theta) = 1 + 1/k - 1/(k epsilon),
    k = E / m_e c^2, which expands to epsilon + (1 - 2a/k)/epsilon + (a^2 - 1) + 1/(k^2 epsilon^2), a = 1 + 1/k.
    epsilon, k broadcastable arrays (epsilon in [1/(1 + 2k), 1])."""
    a = 1.0 + 1.0 / k
    return (0.5 * epsilon * epsilon + (1.0 - 2.0 * a / k) * np.log(epsilon) + (a * a - 1.0) * epsilon
            - 1.0 / (k * k * epsilon))


def grouping_medium_densities(config: SolverConfig) -> tuple[tuple[str, float], ...]:
    """The medium of the "scattered_fluence" balance as ((catalogue name, density g/cm^3), ...), sorted by name, the
    densities rounded to 3 significant digits and the traces below 1e-3 of the largest density dropped (the form that
    enters the cache key and the balance, so that nearly equal media share a cache entry): ``config.grouping_medium``
    "auto" not yet replaced by a measurement (`citrus_recon`, `synthesize_cone_beam_data` do that) -> water at 1.0; a
    catalogue name -> that material at its nominal density; a mapping -> its entries."""
    medium = config.grouping_medium
    if isinstance(medium, str):
        name = "water" if medium == "auto" else medium
        entries = [(name, float(MATERIAL_CATALOGUE[name][0]))]
    else:
        entries = [(str(name), float(density)) for name, density in medium]
    largest = max((density for _, density in entries), default=0.0)
    rounded = sorted((name, float(f"{density:.3g}")) for name, density in entries
                     if density > 0 and density >= 1e-3 * largest)
    if not rounded:
        raise ValueError(f"grouping_medium {medium!r} has no material of positive density")
    return tuple(rounded)


def scattered_fluence_spectrum(spectrum: tuple[np.ndarray, np.ndarray], backend: Optional["_XrayPhysicsBackend"] = None,
                               *, energy_min_kev: float = 10.0, object_size_cm: float = 20.0,
                               medium: Sequence[tuple[str, float]] = (("water", 1.0),),
                               fine_bin_width_kev: float = 0.5,
                               max_bins: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Estimate of the total photon fluence spectrum (source plus scattered photons) inside the object: the grouping
    heuristic of ``make_group_edges`` ("scattered_fluence"), also the within-group weight of that rule. Not a transport
    result.

    One-dimensional (energy only) slowing-down balance in a medium (``medium``: (catalogue name, density g/cm^3) per
    material; default water at 1 g/cm^3) with a leakage loss 1/L, L = object_size_cm the characteristic chord length of
    the object:

        phi(E) = [S(E) + int_{E' > E} mu_C(E') p(E | E') phi(E') dE'] / (mu_t(E) - mu_R(E) + 1/L)

    with S the source spectrum, mu_C, mu_R, mu_t the Compton, Rayleigh and total attenuation of the medium (elemental
    XrayPhysics cross sections mixed by the catalogue compositions and the densities),
    p(E | E') the energy distribution of a Compton photon scattered at E' (Klein-Nishina, normalized; Compton
    kinematics without binding), Rayleigh ignored (it does not change the energy, so it is neither a loss nor a gain
    here), photons below energy_min_kev absorbed. L -> 0 gives phi = S L (the source spectrum, "equal_fluence");
    L -> infinity the spectrum of an infinite medium, where every photon is eventually absorbed.

    Discretization: equal bins of width max(fine_bin_width_kev, (kVp - energy_min_kev) / max_bins) over
    [energy_min_kev, kVp], source photons per bin integrated exactly from the piecewise-linear spectrum, the
    Klein-Nishina probability of each bin integrated in closed form (``_klein_nishina_cumulative``) for a photon at the
    centre of the source bin (scattering within the bin is kept, on the left side), one sweep from the highest bin
    down. Returns (bin_edges [n + 1] keV ascending, photons [n]: phi integrated over each bin, in units of the source
    spectrum times cm)."""
    backend = resolve_backend() if backend is None else backend
    density = _Spectrum(*spectrum)
    energy_max, energy_min = density.max_energy(), float(energy_min_kev)
    if not energy_max > energy_min:
        raise ValueError(f"spectrum max energy {energy_max} <= energy_min_kev {energy_min}")
    if not object_size_cm > 0:
        raise ValueError(f"object_size_cm must be > 0, got {object_size_cm}")
    width = max(float(fine_bin_width_kev), (energy_max - energy_min) / int(max_bins))
    n = max(1, int(math.ceil((energy_max - energy_min) / width - 1e-9)))
    bin_edges = np.linspace(energy_min, energy_max, n + 1)
    low, high = bin_edges[:-1], bin_edges[1:]
    E = 0.5 * (low + high)
    source = density.cumulative_integral(high) - density.cumulative_integral(low)
    element_density: dict[int, float] = {}  # g/cm^3 of every element in the medium
    for name, material_density in medium:
        if name not in MATERIAL_CATALOGUE:
            raise ValueError(f"unknown material {name!r} in the grouping medium (not in the catalogue, "
                             f"cross_sections.MATERIAL_CATALOGUE)")
        for Z, fraction in parse_formula(MATERIAL_CATALOGUE[name][1], backend.atomic_mass).items():
            element_density[Z] = element_density.get(Z, 0.0) + float(material_density) * fraction

    def mu(kind: str) -> np.ndarray:  # linear attenuation of the medium, 1/cm, at the bin centres
        return sum(rho_Z * backend.sigma(kind, Z, E) for Z, rho_Z in sorted(element_density.items()))

    mu_compton = mu("compton")
    removal = mu("total") - mu("rayleigh") + 1.0 / float(object_size_cm)
    # p[j, i]: probability that a Compton photon at E[j] lands in bin i (energy E[j] epsilon, epsilon in [eps_min, 1])
    k = (E / ELECTRON_REST_KEV)[:, None]
    epsilon_min = 1.0 / (1.0 + 2.0 * k)
    epsilon_low = np.clip(low[None, :] / E[:, None], epsilon_min, 1.0)
    epsilon_high = np.clip(np.minimum(high[None, :], E[:, None]) / E[:, None], epsilon_min, 1.0)
    epsilon_high = np.maximum(epsilon_high, epsilon_low)
    total = _klein_nishina_cumulative(np.ones_like(epsilon_min), k) - _klein_nishina_cumulative(epsilon_min, k)
    probability = (_klein_nishina_cumulative(epsilon_high, k) - _klein_nishina_cumulative(epsilon_low, k)) / total
    phi = np.zeros(n)
    inscatter = np.zeros(n)  # sum over higher bins j of mu_C[j] p[j, i] phi[j]
    for i in range(n - 1, -1, -1):
        phi[i] = (source[i] + inscatter[i]) / (removal[i] - mu_compton[i] * probability[i, i])
        if i:
            inscatter[:i] += mu_compton[i] * probability[i, :i] * phi[i]
    return bin_edges, phi


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
      config: uses num_groups, group_edges_kev, group_edges_method, grouping_object_size_cm, energy_min_kev,
        legendre_order, num_eta_table, angular_table_grid.
      spectrum: (energies keV [K], photon density per keV [K]); None -> default 125 kVp spectrum.
      backend: a loaded ``resolve_backend()`` object (None loads one).
      fine_bin_width_kev: maximum fine energy bin width (<= 0.5 keV for production).
      gauss_legendre_nodes: Gauss-Legendre nodes per angular piece (>= 32 for production).
      materials: (name, nominal density g/cm^3, formula) per material, in any order (sorted here by nominal attenuation
        at the spectrum-mean energy, ``sort_material_table``; ``check_material_order`` raises if two materials cross
        inside the library's energy range); None: the seven default reference materials (``default_material_table``).

    Returns:
      (MultigroupLibrary with float64 CPU tensors, BuildInfo).
    """
    backend = resolve_backend() if backend is None else backend
    spectrum_energies, spectrum_weights = _spectrum_or_default(spectrum, backend)
    spectrum_density = _Spectrum(spectrum_energies, spectrum_weights)
    edges = make_group_edges(config, (spectrum_energies, spectrum_weights), backend,
                             fine_bin_width_kev=fine_bin_width_kev)
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
    # within-group weight: the source spectrum, or the estimated total fluence for "scattered_fluence" (flat where a
    # group has no weight)
    within_group_weight, flat_fallback = _within_group_weights(
        config, (spectrum_energies, spectrum_weights), backend, E, fine_groups, bin_low, bin_high, photons_per_bin, G,
        fine_bin_width_kev)
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
    adaptive = config.angular_table_grid == "adaptive"
    num_elements = len(elements)

    mu_s_l = np.zeros((M, G, G, L + 1))
    compton_below = np.zeros((M, G))
    rayleigh_l = np.zeros((M, G, L + 1))
    # per fine energy and element: scale [M] of the raw angular shapes (sigma_x w_Z / quadrature norm)
    compton_scales = np.zeros((K, num_elements, M))
    rayleigh_scales = np.zeros((K, num_elements, M))
    # adaptive table: the per-steradian shapes of the map materials on the reference quadrature [K, M_map, Nr]
    if adaptive:
        map_composition = material_composition(backend, material_table(ANGULAR_MAP_MATERIALS))
        map_elements = sorted({Z for c in map_composition.values() for Z in c})
        map_fractions = np.array([[map_composition[m].get(Z, 0.0) for Z in map_elements]
                                  for m in ANGULAR_MAP_MATERIALS])  # [M_map, NZ_map]
        map_sigma = {kind: {Z: backend.sigma(kind, Z, E) for Z in map_elements} for kind in ("compton", "rayleigh")}
        compton_shapes = np.zeros((K, len(ANGULAR_MAP_MATERIALS), eta_reference.size))
        rayleigh_shapes = np.zeros_like(compton_shapes)
    normalization_deviation = 0.0
    have_library_normalization = True

    # pass 1: normalizations, Legendre moments (and the reference shapes of the angular map)
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
        element_shapes: dict[int, tuple] = {}  # Z -> raw shapes and norms on the reference quadrature (adaptive)
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
            compton_scales[k, element_index] = compton_scale
            rayleigh_scales[k, element_index] = rayleigh_scale
            compton_shape_pieces = backend.pdf_raw("compton", Z, energy_k, eta_compton)
            compton_pieces += compton_scale[:, None] * compton_shape_pieces[None, :]
            rayleigh_reference += rayleigh_scale[:, None] * rayleigh_shape[None, :]
            if adaptive:
                element_shapes[Z] = (compton_shape, compton_norm, rayleigh_shape, rayleigh_norm)
        if adaptive:
            for map_index, Z in enumerate(map_elements):
                if Z not in element_shapes:
                    shape_c = backend.pdf_raw("compton", Z, energy_k, eta_reference)
                    shape_r = backend.pdf_raw("rayleigh", Z, energy_k, eta_reference)
                    element_shapes[Z] = (shape_c, _TWO_PI * float(np.dot(w_reference, shape_c)), shape_r,
                                         _TWO_PI * float(np.dot(w_reference, shape_r)))
                shape_c, norm_c, shape_r, norm_r = element_shapes[Z]
                compton_shapes[k] += (map_fractions[:, map_index] * map_sigma["compton"][Z][k] / norm_c)[:, None] \
                    * shape_c[None, :]
                rayleigh_shapes[k] += (map_fractions[:, map_index] * map_sigma["rayleigh"][Z][k] / norm_r)[:, None] \
                    * shape_r[None, :]

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

    # node placement, then pass 2: the direct per-steradian tables at the nodes
    eta_map = None
    if adaptive:
        eta_map = angular_map(np.sqrt(np.clip((1.0 - eta_reference) / 2.0, 0.0, 1.0)), compton_shapes, rayleigh_shapes,
                              fine_groups, bin_low, bin_high, within_group_weight, edges)
        eta_grid = angular_table_nodes(N_eta, torch.as_tensor(eta_map)).numpy()
    else:
        eta_grid = np.linspace(-1.0, 1.0, N_eta)
    mu_s_eta, rayleigh_eta = _direct_transfer_table(eta_grid, backend, elements, E, fine_groups, bin_low, bin_high,
                                                    within_group_weight, edges, compton_scales, rayleigh_scales)

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
        eta_map=None if eta_map is None else to_tensor(eta_map),
    )
    return library, info


def _direct_transfer_table(eta: np.ndarray, backend: "_XrayPhysicsBackend", elements: Sequence[int], E: np.ndarray,
                           fine_groups: np.ndarray, bin_low: np.ndarray, bin_high: np.ndarray,
                           within_group_weight: np.ndarray, edges: np.ndarray, compton_scales: np.ndarray,
                           rayleigh_scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The per-steradian transfer at the cos-angles eta [n] (any order): (mu_s_eta [M, G, G, n], rayleigh_eta
    [M, G, n]) in cm^2/g/sr, Compton with the exact fraction of every fine bin that lands in each outgoing group plus
    within-group Rayleigh, from the raw backend shapes times the per-element scales [K, NZ, M] of pass 1 of
    ``compute_library`` (fine energies E [K] of groups fine_groups, bins [bin_low, bin_high], within-group weights)."""
    m_e = ELECTRON_REST_KEV
    G = edges.size - 1
    K, _, M = compton_scales.shape
    n = eta.size
    mu_s_eta = np.zeros((M, G, G, n))
    rayleigh_eta = np.zeros((M, G, n))
    # Table kinematics: at fixed eta, E' lands in [edges[j+1], edges[j]) iff E lies in
    # [E_b(edges[j+1], eta), E_b(edges[j], eta)) with E_b(e, eta) = 1 / (1/e - (1 - eta)/m_e)
    # (+inf when 1/e <= (1 - eta)/m_e). Each fine bin contributes the fraction of its width inside
    # that interval, so the tabulated transfer is continuous in eta (a midpoint indicator would
    # give a staircase with 0.5 keV steps).
    inv_b = 1.0 / edges[:, None] - (1.0 - eta[None, :]) / m_e  # [G+1, n]
    E_b = np.where(inv_b > 0, 1.0 / np.where(inv_b > 0, inv_b, 1.0), np.inf)
    for k in range(K):
        energy_k, g, weight_k = float(E[k]), int(fine_groups[k]), float(within_group_weight[k])
        compton_table = np.zeros((M, n))
        rayleigh_table = np.zeros((M, n))
        for element_index, Z in enumerate(elements):
            compton_table += compton_scales[k, element_index][:, None] * backend.pdf_raw("compton", Z, energy_k,
                                                                                         eta)[None, :]
            rayleigh_table += rayleigh_scales[k, element_index][:, None] * backend.pdf_raw("rayleigh", Z, energy_k,
                                                                                           eta)[None, :]
        bin_low_k, bin_high_k = float(bin_low[k]), float(bin_high[k])
        fraction = (np.clip(np.minimum(bin_high_k, E_b[g:G]) - np.maximum(bin_low_k, E_b[g + 1:G + 1]), 0.0, None)
                    / (bin_high_k - bin_low_k))
        mu_s_eta[:, g, g:, :] += weight_k * compton_table[:, None, :] * fraction[None, :, :]
        mu_s_eta[:, g, g, :] += weight_k * rayleigh_table
        rayleigh_eta[:, g, :] += weight_k * rayleigh_table
    return mu_s_eta, rayleigh_eta


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    """Moving average of x [n] over `width` samples (odd; the window shrinks at the ends, so a constant stays)."""
    half = max(0, int(width) // 2)
    if half == 0:
        return x.copy()
    padded = np.concatenate([[0.0], np.cumsum(x)])
    index = np.arange(x.size)
    lo, hi = np.maximum(index - half, 0), np.minimum(index + half + 1, x.size)
    return (padded[hi] - padded[lo]) / (hi - lo)


def angular_map(q_nodes: np.ndarray, compton_shapes: np.ndarray, rayleigh_shapes: np.ndarray, fine_groups: np.ndarray,
                bin_low: np.ndarray, bin_high: np.ndarray, within_group_weight: np.ndarray, edges: np.ndarray, *,
                num_knots: int = ANGULAR_MAP_KNOTS, num_reference: int = 20001, floor: float = 0.1,
                negligible: float = 1.0e-6) -> np.ndarray:
    """The node placement of an adaptive transfer table: the map u = h(eta) at the knots q_k = k / (num_knots - 1) of
    q = sin(theta/2) = sqrt((1 - eta)/2), float64 [num_knots], strictly decreasing from 1 (eta = 1) to 0 (eta = -1)
    (``MultigroupLibrary.eta_map``; the table node j sits at u = j / (N - 1)).

    Inputs: q_nodes [Nr] ascending (the reference angular quadrature of the build, dense near q = 0),
    compton_shapes / rayleigh_shapes [K, M, Nr] the per-material per-steradian Compton / Rayleigh cross sections of
    every fine energy at those nodes (cm^2/g/sr, before the kinematic group fraction; the build passes those of
    ``ANGULAR_MAP_MATERIALS``), the fine-energy bins (groups
    fine_groups [K], edges bin_low / bin_high [K] keV, within-group weights [K]) and the group edges [G+1] keV.

    1. Every transfer curve f_{m, g, g'}(q) (material m, source group g, destination group g' >= g; Compton with the
       kinematic fraction of every fine bin, plus Rayleigh for g' = g) is evaluated on num_reference points uniform in
       q, from cubic splines through the shapes at q_nodes (the reference grid resolves the MeV forward peaks, whose
       width in q is about 5e-4 at 6 MeV).
    2. Node density: rho(q) = max over the curves (those above `negligible` times the largest curve of the same m, g)
       of sqrt(|f''(q)| / max f), the density that equalizes the error of linear interpolation relative to each
       curve's maximum; smoothed over two knot intervals so that neighbouring node spacings change gradually.
    3. Floor: the normalized density is mixed as (1 - floor) rho / int rho + floor 2q, so every region keeps at least
       `floor` times the node density of a uniform table in eta (2q is that density in q), and h is strictly monotone.
    4. h = 1 - (cumulative integral of the density from q = 0) / (total), sampled at the knots.
    Between knots the map is linear in q, so linear interpolation between table nodes is linear in q there."""
    from scipy.interpolate import CubicSpline

    m_e = ELECTRON_REST_KEV
    K, M, _ = compton_shapes.shape
    G = edges.size - 1
    q = np.linspace(0.0, 1.0, int(num_reference))
    dq = q[1] - q[0]
    eta = 1.0 - 2.0 * q * q
    inv_b = 1.0 / edges[:, None] - (1.0 - eta[None, :]) / m_e  # [G+1, R]
    E_b = np.where(inv_b > 0, 1.0 / np.where(inv_b > 0, inv_b, 1.0), np.inf)
    density = np.zeros(q.size)
    curves = np.zeros((M, G, q.size))  # the curves of the current source group

    def finish(g: int) -> None:
        block = curves[:, g:, :]  # [M, G - g, R]
        peak = np.abs(block).max(axis=2)  # [M, G - g]
        keep = peak > negligible * peak.max(axis=1, keepdims=True)
        keep &= peak > 0
        if keep.any():
            selected = block[keep] / peak[keep][:, None]
            second = np.abs(selected[:, :-2] - 2.0 * selected[:, 1:-1] + selected[:, 2:])
            density[1:-1] = np.maximum(density[1:-1], np.sqrt(second.max(axis=0)) / dq)
        curves[:] = 0.0

    current = int(fine_groups[0]) if K else 0
    for k in range(K):
        g = int(fine_groups[k])
        if g != current:
            finish(current)
            current = g
        spline = CubicSpline(q_nodes, np.concatenate([compton_shapes[k], rayleigh_shapes[k]]), axis=1)
        values = spline(q)  # [2M, R]
        compton, rayleigh = values[:M], values[M:]
        fraction = (np.clip(np.minimum(bin_high[k], E_b[g:G]) - np.maximum(bin_low[k], E_b[g + 1:G + 1]), 0.0, None)
                    / (bin_high[k] - bin_low[k]))  # [G - g, R]
        weight = float(within_group_weight[k])
        curves[:, g:, :] += weight * compton[:, None, :] * fraction[None, :, :]
        curves[:, g, :] += weight * rayleigh
    if K:
        finish(current)
    density[0], density[-1] = density[1], density[-2]
    knot_width = (q.size - 1) / (int(num_knots) - 1)
    density = _smooth(_smooth(density, 2 * int(round(knot_width)) + 1), 2 * int(round(knot_width)) + 1)
    total = float(np.sum(0.5 * (density[1:] + density[:-1])) * dq)
    uniform = 2.0 * q  # node density in q of a table uniform in eta (unit integral over [0, 1])
    mixed = (1.0 - floor) * density / total + floor * uniform if total > 0 else uniform
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (mixed[1:] + mixed[:-1]) * dq)])
    knots = np.linspace(0.0, 1.0, int(num_knots))
    u = 1.0 - np.interp(knots, q, cumulative) / cumulative[-1]
    u[0], u[-1] = 1.0, 0.0
    if not np.all(np.diff(u) < 0):
        raise RuntimeError("angular map is not strictly monotone")
    return u


def transfer_at_eta(config: SolverConfig, eta: np.ndarray, spectrum: tuple[np.ndarray, np.ndarray] | None = None,
                    backend: "_XrayPhysicsBackend | None" = None, *, fine_bin_width_kev: float = 0.5,
                    gauss_legendre_nodes: int = 32, materials: Optional[MaterialTable] = None) -> np.ndarray:
    """The exact per-steradian transfer mu_s(eta) [M, G, G, n] (cm^2/g/sr; the quantity ``mu_s_eta`` tabulates,
    without interpolation) of the library ``compute_library`` builds with the same arguments, at the cos-angles
    eta [n]: for measuring the interpolation error of a table (tests, validation). Same cost per point as a table
    node of the build."""
    backend = resolve_backend() if backend is None else backend
    spectrum_energies, spectrum_weights = _spectrum_or_default(spectrum, backend)
    spectrum_density = _Spectrum(spectrum_energies, spectrum_weights)
    edges = make_group_edges(config, (spectrum_energies, spectrum_weights), backend,
                             fine_bin_width_kev=fine_bin_width_kev)
    G = edges.size - 1
    table = sort_material_table(config, materials, (spectrum_energies, spectrum_weights), backend,
                                fine_bin_width_kev=fine_bin_width_kev)
    names = tuple(n for n, _, _ in table)
    E, fine_groups, bin_low, bin_high = fine_energy_grid(edges, fine_bin_width_kev)
    photons_per_bin = spectrum_density.cumulative_integral(bin_high) - spectrum_density.cumulative_integral(bin_low)
    within_group_weight, _ = _within_group_weights(config, (spectrum_energies, spectrum_weights), backend, E,
                                                   fine_groups, bin_low, bin_high, photons_per_bin, G,
                                                   fine_bin_width_kev)
    composition = material_composition(backend, table)
    elements = sorted({Z for c in composition.values() for Z in c})
    mass_fractions = np.array([[composition[m].get(Z, 0.0) for Z in elements] for m in names])
    eta_reference, w_reference = eta_quadrature_in_q(_Q_BREAKPOINTS, gauss_legendre_nodes)
    compton_scales = np.zeros((E.size, len(elements), len(names)))
    rayleigh_scales = np.zeros_like(compton_scales)
    for element_index, Z in enumerate(elements):
        sigma_compton, sigma_rayleigh = backend.sigma("compton", Z, E), backend.sigma("rayleigh", Z, E)
        for k in range(E.size):
            compton_norm = _TWO_PI * float(np.dot(w_reference, backend.pdf_raw("compton", Z, float(E[k]),
                                                                                eta_reference)))
            rayleigh_norm = _TWO_PI * float(np.dot(w_reference, backend.pdf_raw("rayleigh", Z, float(E[k]),
                                                                                 eta_reference)))
            compton_scales[k, element_index] = mass_fractions[:, element_index] * sigma_compton[k] / compton_norm
            rayleigh_scales[k, element_index] = mass_fractions[:, element_index] * sigma_rayleigh[k] / rayleigh_norm
    eta = np.asarray(eta, dtype=np.float64).ravel()
    return _direct_transfer_table(eta, backend, elements, E, fine_groups, bin_low, bin_high, within_group_weight,
                                  edges, compton_scales, rayleigh_scales)[0]


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
        "angular_table_grid": str(config.angular_table_grid),
        "group_edges_method": str(config.group_edges_method),
        "grouping_object_size_cm": float(config.grouping_object_size_cm),
        "grouping_medium": [list(m) for m in grouping_medium_densities(config)],
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
    the seven default reference materials, ``default_material_table``; else (name, nominal density, formula) per
    material, e.g.
    ``cross_sections.material_table(names)``, in any order: sorted by nominal attenuation, ``sort_material_table``).

    Shapes of the result (M materials, 7 by default; G = config.num_groups, L = config.legendre_order):
      group_edges [G+1] desc, group_energy [G], spectrum_fraction [G], mu_t/mu_pe [M, G],
      mu_s_l [M, G, G, L+1], eta_grid [num_eta_table], mu_s_eta [M, G, G, num_eta_table],
      fine_energy [E], fine_mu_t [M, E], fine_weight [E], fine_group [E] (int64; E ~ (kVp - E_min) /
      fine_bin_width_kev); float64 CPU tensors, per unit density (cm^2/g, cm^2/g/sr); plus materials (names),
      nominal_density [M] and eta_map [ANGULAR_MAP_KNOTS] (None for config.angular_table_grid "uniform").

    The cache file is ``<cache_dir>/mglib_<hash>.pt`` keyed by the build version, the relevant config
    fields, the spectrum arrays, the backend name, the numerical parameters and the SORTED material table (names,
    nominal densities, formulas: every set of materials has its own entry, any order of the same set the same one).
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
