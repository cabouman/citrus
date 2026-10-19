"""Build the multigroup cross-section library of the scatter solver from point-wise photon interaction data.

:func:`build_library` fills a :class:`~citrus.scatter_solver.cross_sections.MultigroupLibrary` for a table of
reference materials (catalogue entries or user-defined ``Material`` definitions) from the integrated cross sections of
NIST XCOM and the angular shapes of xraylib (``photon_data``) and caches it on disk. The module also holds the steps of
the build: material formulas, the source spectrum, the group edges, the fine energy grid, the angular quadratures and the node placement of adaptive angular tables. The physics
and the conventions are described in the documentation pages "Design notes" (section "Cross sections") and
"Conventions".
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from .config import SolverConfig
from .cross_sections import (ANGULAR_MAP_KNOTS, ELECTRON_REST_KEV, MATERIAL_CATALOGUE, MATERIALS, Material,
                             MultigroupLibrary, angular_table_nodes, material_definition, material_table,
                             parse_formula)
from .photon_data import XcomXraylibBackend, default_cache_dir, resolve_backend

__all__ = [
    "BuildInfo",
    "MATERIAL_FORMULAS",
    "XcomXraylibBackend",
    "default_material_table",
    "sort_material_table",
    "mass_attenuation_preview",
    "check_material_order",
    "build_library",
    "default_cache_dir",
    "compute_library",
    "monoenergetic_spectrum",
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
# grid (adaptive tables with eta_map), the group edge rule and the grouping object size in the key. 4 (citrus 0.7.0):
# the photon data of NIST XCOM and xraylib (backend "xcom-xraylib") instead of XrayPhysics
_BUILD_VERSION = "citrus-xsec-4"
_TWO_PI = 2.0 * math.pi
# catalogue materials whose transfer curves place the nodes of an adaptive angular table (`angular_map`): low-Z soft
# tissue and calcium-bearing bone; the same for every library, so that the nodes do not depend on the material set
ANGULAR_MAP_MATERIALS = ("water", "bone_cortical_icrp")

# Compositions (formula strings of `parse_formula`: 'a*F1 + b*F2' with a, b mass fractions of the compounds F1, F2;
# a bare formula is a stoichiometric compound) of every catalogue material, `cross_sections.MATERIAL_CATALOGUE`
MATERIAL_FORMULAS: dict[str, str] = {name: formula for name, (_, formula) in MATERIAL_CATALOGUE.items()}

# (name, nominal density g/cm^3, formula) per material; a catalogue name or a `Material` also stands for its entry
MaterialTable = Sequence[tuple[str, float, str]]


def default_material_table() -> tuple[tuple[str, float, str], ...]:
    """Return the seven default reference materials as (name, nominal density, formula) entries.

    Returns:
        tuple: (name, nominal density in g/cm^3, formula) per material, ``cross_sections.material_table(MATERIALS)``.
    """
    return material_table(MATERIALS)


def _material_table(materials: Optional[MaterialTable]) -> tuple[tuple[str, float, str], ...]:
    """Check a material table, or return the default one for None.

    Args:
        materials (Sequence or None): (name, nominal density in g/cm^3, formula), a catalogue name or a ``Material``
            per material; None for the seven default reference materials.

    Returns:
        tuple: the table as (str, float, str) entries.
    """
    t = default_material_table() if materials is None else tuple(
        material_table((entry,))[0] if isinstance(entry, (str, Material)) else (str(entry[0]), float(entry[1]),
                                                                              str(entry[2]))
        for entry in materials)
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

# =============================================================================================
# Materials
# =============================================================================================

def material_composition(backend: XcomXraylibBackend,
                         materials: Optional[MaterialTable] = None) -> dict[str, dict[int, float]]:
    """Return the element mass fractions of every material of a table.

    Args:
        backend (XcomXraylibBackend): photon data; supplies the atomic weights.
        materials (Sequence or None, optional): (name, nominal density, formula) per material. Defaults to None, the
            seven default reference materials.

    Returns:
        dict[str, dict[int, float]]: mass fraction per atomic number Z of each material, keyed by name in table order.
    """
    return {n: parse_formula(f, backend.atomic_mass) for n, _, f in _material_table(materials)}


def _mean_energy(config: SolverConfig, spectrum: tuple[np.ndarray, np.ndarray],
                 fine_bin_width_kev: float | None,
                 backend: Optional[XcomXraylibBackend] = None) -> float:
    """Return the spectrum-mean photon energy over the group structure of ``config`` on the fine energy grid.

    The mean is ``sum_e w_e E_e / sum_e w_e`` with ``w_e`` the exact spectrum integral over fine bin e, the effective
    energy of the library's density model.

    Args:
        config (SolverConfig): solver configuration; sets the group edges.
        spectrum (tuple[np.ndarray, np.ndarray]): (energies [K] in keV, photon density per keV [K]).
        fine_bin_width_kev (float or None): maximum fine bin width in keV; None for the automatic grid.
        backend (XcomXraylibBackend or None, optional): photon data, needed by the "scattered_fluence" group rule.
            Defaults to None (loaded when needed).

    Returns:
        float: mean energy in keV.
    """
    edges = make_group_edges(config, spectrum, backend, fine_bin_width_kev=fine_bin_width_kev)
    E, _, bin_low, bin_high = fine_energy_grid(edges, fine_bin_width_kev)
    density = _Spectrum(*spectrum)
    E = density.place_line(E, bin_low, bin_high)
    photons = density.cumulative_integral(bin_high) - density.cumulative_integral(bin_low)
    if not photons.sum() > 0:
        raise ValueError("spectrum has no photons inside the group structure")
    return float((photons * E).sum() / photons.sum())


def sort_material_table(config: SolverConfig, materials: Optional[MaterialTable],
                        spectrum: tuple[np.ndarray, np.ndarray],
                        backend: XcomXraylibBackend | None = None, *,
                        fine_bin_width_kev: float | None = None) -> tuple[tuple[str, float, str], ...]:
    """Sort a material table by nominal attenuation at the spectrum-mean energy of the build.

    The attenuation is ``rho_nom * mu/rho(E_eff)`` with the elemental total mass attenuation of the backend mixed by
    mass fraction; the sort is stable, so a table already in that order is returned unchanged.

    Args:
        config (SolverConfig): solver configuration; sets the group structure and so the mean energy.
        materials (Sequence or None): (name, nominal density in g/cm^3, formula) per material; None, the seven
            default reference materials.
        spectrum (tuple[np.ndarray, np.ndarray]): (energies [K] in keV, photon density per keV [K]); one node is a
            monoenergetic line.
        backend (XcomXraylibBackend or None, optional): photon data. Defaults to None (loaded by ``resolve_backend``).
        fine_bin_width_kev (float or None, optional): maximum fine bin width in keV; None for the automatic grid
            (``fine_energy_grid``). Defaults to None.

    Returns:
        tuple: the (name, nominal density, formula) entries in order of increasing nominal attenuation.
    """
    table = _material_table(materials)
    backend = resolve_backend() if backend is None else backend
    spectrum_pair = spectrum_arrays(spectrum)
    energy = np.array([_mean_energy(config, spectrum_pair, fine_bin_width_kev, backend)])
    composition = material_composition(backend, table)
    attenuation = [density * sum(w * float(backend.sigma("total", Z, energy)[0]) for Z, w in composition[name].items())
                   for name, density, _ in table]
    order = sorted(range(len(table)), key=lambda i: attenuation[i])
    return tuple(table[i] for i in order)


def mass_attenuation_preview(config: SolverConfig, names: Sequence[str],
                             spectrum: tuple[np.ndarray, np.ndarray] | None = None,
                             backend: XcomXraylibBackend | None = None, *, energy_kev: float | None = None,
                             fine_bin_width_kev: float | None = None) -> tuple[float, np.ndarray]:
    """Return the total mass attenuation of reference materials at the effective energy without building the library.

    The effective energy is ``energy_kev`` or the spectrum-mean energy of the build of ``config``; the attenuation mixes
    the backend's elemental cross sections by mass fraction at that energy.

    Args:
        config (SolverConfig): solver configuration; sets the group structure and so the mean energy.
        names (Sequence[str or Material]): catalogue names, names of the ``Material`` entries of ``config.materials``
            or materials.
        spectrum (tuple[np.ndarray, np.ndarray] or None, optional): (energies [K] in keV, photon density per keV [K]);
            required unless ``energy_kev`` is given. Defaults to None.
        backend (XcomXraylibBackend or None, optional): photon data. Defaults to None (loaded by ``resolve_backend``).
        energy_kev (float or None, optional): effective energy in keV. Defaults to None, the spectrum-mean energy.
        fine_bin_width_kev (float or None, optional): maximum fine bin width in keV; None for the automatic grid
            (``fine_energy_grid``). Defaults to None.

    Returns:
        (float, np.ndarray): the effective energy in keV and the total mass attenuation [len(names)] in cm^2/g there.
    """
    backend = resolve_backend() if backend is None else backend
    energy = (_mean_energy(config, spectrum_arrays(spectrum), fine_bin_width_kev, backend) if energy_kev is None
              else float(energy_kev))
    table = material_table(tuple(material_definition(name, config.materials) for name in names))
    composition = material_composition(backend, table)
    point = np.array([energy])
    mass = np.array([sum(w * float(backend.sigma("total", Z, point)[0]) for Z, w in composition[name].items())
                     for name, _, _ in table])
    return energy, mass


def check_material_order(fine_mu_t: np.ndarray, nominal_density: np.ndarray, names=None) -> None:
    """Check that the nominal attenuations increase strictly with the material index at every fine energy.

    This is the fixed bracketing order of the density model (``citrus.utilities.attenuation_to_densities``).

    Args:
        fine_mu_t (np.ndarray): [M, E] total mass attenuation in cm^2/g at the fine energies.
        nominal_density (np.ndarray): [M] nominal densities in g/cm^3.
        names (Sequence[str] or None, optional): material names for the error message. Defaults to None (indices).

    Raises:
        ValueError: if two consecutive materials cross at some energy.
    """
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
    """Load a two-column text spectrum of energy in keV and photon density per keV.

    Columns are separated by commas, semicolons or whitespace; ``#`` starts a comment and lines that do not parse are
    skipped.

    Args:
        path (str): path of the text file.

    Returns:
        (np.ndarray, np.ndarray): energies [K] in keV, ascending, and photon densities per keV [K], float64.
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


def monoenergetic_spectrum(energy_kev: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the spectrum of a monoenergetic source: one node, which the library treats as a line at that energy.

    Args:
        energy_kev (float): photon energy in keV.

    Returns:
        (np.ndarray, np.ndarray): energies [1] in keV and photon weight [1], float64.
    """
    if not float(energy_kev) > 0:
        raise ValueError(f"the photon energy must be > 0 keV, got {energy_kev}")
    return np.array([float(energy_kev)]), np.array([1.0])


class _Spectrum:
    """Piecewise-linear spectral density s(E), zero outside the samples, with exact integrals; one sample is a line.

    Attributes:
        energies (np.ndarray): [K] sample energies in keV, ascending.
        weights (np.ndarray): [K] density at the samples (the photon weight of a line), negative values clipped to 0.
        cumulative (np.ndarray): [K] integral of s from the first sample to each sample.
        monoenergetic (bool): True for a single sample, a line at ``energies[0]`` carrying ``weights[0]`` photons.
    """

    def __init__(self, energies: np.ndarray, weights: np.ndarray) -> None:
        """Sort the samples and tabulate the cumulative integral.

        Args:
            energies (np.ndarray): [K] sample energies in keV, K >= 1.
            weights (np.ndarray): [K] spectral density at the samples, or the photon weight of a single line.
        """
        energies = np.asarray(energies, dtype=np.float64).ravel()
        weights = np.asarray(weights, dtype=np.float64).ravel()
        if energies.size != weights.size or energies.size < 1:
            raise ValueError("spectrum needs >= 1 (energy, weight) samples")
        o = np.argsort(energies)
        self.energies, self.weights = energies[o], np.clip(weights[o], 0.0, None)
        self.monoenergetic = energies.size == 1
        if self.monoenergetic:
            if not self.weights[0] > 0:
                raise ValueError("a monoenergetic spectrum needs a positive weight")
            self.cumulative = np.array([0.0])
            return
        seg = 0.5 * (self.weights[1:] + self.weights[:-1]) * np.diff(self.energies)
        self.cumulative = np.concatenate([[0.0], np.cumsum(seg)])

    def __call__(self, E: np.ndarray) -> np.ndarray:
        """Evaluate the density by linear interpolation (0 everywhere for a line, which has no density).

        Args:
            E (np.ndarray): energies in keV.

        Returns:
            np.ndarray: s(E), same shape as ``E``, 0 outside the samples.
        """
        if self.monoenergetic:
            return np.zeros_like(np.asarray(E, dtype=np.float64))
        return np.interp(E, self.energies, self.weights, left=0.0, right=0.0)

    def cumulative_integral(self, E: np.ndarray) -> np.ndarray:
        """Return the exact integral of s from minus infinity to E (a line counts from its energy on).

        Args:
            E (np.ndarray): upper limits in keV.

        Returns:
            np.ndarray: the integrals, same shape as ``E``.
        """
        if self.monoenergetic:
            return np.where(np.asarray(E, dtype=np.float64) >= self.energies[0], self.weights[0], 0.0)
        E = np.clip(np.asarray(E, dtype=np.float64), self.energies[0], self.energies[-1])
        i = np.clip(np.searchsorted(self.energies, E, side="right") - 1, 0, self.energies.size - 2)
        x0, x1 = self.energies[i], self.energies[i + 1]
        y0, y1 = self.weights[i], self.weights[i + 1]
        t = E - x0
        slope = (y1 - y0) / (x1 - x0)
        return self.cumulative[i] + y0 * t + 0.5 * slope * t * t

    def max_energy(self) -> float:
        """Return the upper end of the support of s: the first zero sample after the last positive one, or the line.

        Returns:
            float: energy in keV.
        """
        pos = np.nonzero(self.weights > 0)[0]
        if pos.size == 0:
            raise ValueError("spectrum is identically zero")
        j = int(pos[-1])
        return float(self.energies[min(j + 1, self.energies.size - 1)])

    def place_line(self, E: np.ndarray, bin_low: np.ndarray, bin_high: np.ndarray) -> np.ndarray:
        """Return the fine bin energies with the bin that holds a monoenergetic line evaluated at the line.

        Args:
            E (np.ndarray): [K] fine bin midpoints in keV.
            bin_low (np.ndarray): [K] lower bin edges in keV.
            bin_high (np.ndarray): [K] upper bin edges in keV.

        Returns:
            np.ndarray: [K] energies in keV, ``E`` itself unless the spectrum is a line.
        """
        if not self.monoenergetic:
            return E
        line = self.energies[0]
        inside = (bin_low < line) & (line <= bin_high)
        if not inside.any():
            raise ValueError(f"the line at {line} keV lies outside the group structure")
        return np.where(inside, line, E)


def _equal_area_edges(grid: np.ndarray, cumulative: np.ndarray, G: int, energy_min: float,
                      energy_max: float) -> np.ndarray:
    """Return descending group edges that cut the area under a nonnegative density into G equal parts.

    Args:
        grid (np.ndarray): [n] ascending energies in keV, from ``energy_min`` to ``energy_max``.
        cumulative (np.ndarray): [n] integral of the density from ``energy_min`` to each grid energy (linear in
            between).
        G (int): number of groups.
        energy_min (float): lowest edge in keV.
        energy_max (float): highest edge in keV.

    Returns:
        np.ndarray: [G+1] edges in keV, descending.
    """
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
                     backend: Optional[XcomXraylibBackend] = None, *,
                     fine_bin_width_kev: float | None = None) -> np.ndarray:
    """Return the group edges in keV, descending, from kVp (the upper end of the spectrum) down to ``energy_min_kev``.

    The edges are ``config.group_edges_kev`` if given, otherwise placed by ``config.group_edges_method``:
    "scattered_fluence" (equal areas under the total fluence estimate of ``scattered_fluence_spectrum``),
    "equal_fluence" (equal source photon counts) or "log" (geometric spacing).

    Args:
        config (SolverConfig): solver configuration; uses num_groups, group_edges_kev, group_edges_method,
            energy_min_kev, grouping_object_size_cm and grouping_medium.
        spectrum (tuple[np.ndarray, np.ndarray]): (energies [K] in keV, photon density per keV [K]).
        backend (XcomXraylibBackend or None, optional): photon data for the "scattered_fluence" rule. Defaults to None
            (loaded when needed).
        fine_bin_width_kev (float or None, optional): bin width in keV of the "scattered_fluence" balance; None for
            ``FINE_BIN_WIDTH_FLOOR_KEV``. Defaults to None.

    Returns:
        np.ndarray: [G+1] edges in keV, descending.
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
        if spectrum_density.monoenergetic:
            raise ValueError('group_edges_method "equal_fluence" needs a continuous spectrum; a monoenergetic source '
                             'uses "scattered_fluence" or "log" (or explicit group_edges_kev)')
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
    compositions = [material_definition(name, config.materials).composition for name, _ in medium]
    key = hashlib.sha256(json.dumps([backend.name, energy_min, float(config.grouping_object_size_cm), G,
                                     fine_bin_width_kev, [list(m) for m in medium], compositions]).encode()
                         + np.ascontiguousarray(spectrum[0], dtype=np.float64).tobytes()
                         + np.ascontiguousarray(spectrum[1], dtype=np.float64).tobytes()).hexdigest()
    if key not in _EDGES_MEMO:
        bin_edges, photons = scattered_fluence_spectrum(spectrum, backend, energy_min_kev=energy_min,
                                                        object_size_cm=float(config.grouping_object_size_cm),
                                                        medium=medium, fine_bin_width_kev=fine_bin_width_kev,
                                                        materials=config.materials)
        cumulative = np.concatenate([[0.0], np.cumsum(photons)])
        _EDGES_MEMO[key] = _equal_area_edges(bin_edges, cumulative, G, energy_min, energy_max)
    return _EDGES_MEMO[key].copy()


def _within_group_weights(config: SolverConfig, spectrum: tuple[np.ndarray, np.ndarray], backend, E: np.ndarray,
                          fine_groups: np.ndarray, bin_low: np.ndarray, bin_high: np.ndarray,
                          photons_per_bin: np.ndarray, G: int,
                          fine_bin_width_kev: float | None) -> tuple[np.ndarray, list]:
    """Return the weights of the fine bins within their group, and the groups that fell back to a flat weight.

    The weight is the source spectrum for explicit edges and the rules "equal_fluence" and "log", and the total fluence
    estimate of ``scattered_fluence_spectrum`` for "scattered_fluence"; a group whose weight is zero gets a flat weight.

    Args:
        config (SolverConfig): solver configuration; selects the rule.
        spectrum (tuple[np.ndarray, np.ndarray]): (energies [S] in keV, photon density per keV [S]).
        backend (XcomXraylibBackend): photon data for the "scattered_fluence" rule.
        E (np.ndarray): [K] fine bin midpoints in keV.
        fine_groups (np.ndarray): [K] group index of each fine bin.
        bin_low (np.ndarray): [K] lower bin edges in keV.
        bin_high (np.ndarray): [K] upper bin edges in keV.
        photons_per_bin (np.ndarray): [K] exact source photons per bin.
        G (int): number of groups.
        fine_bin_width_kev (float or None): bin width in keV of the "scattered_fluence" balance.

    Returns:
        (np.ndarray, list): weights [K] that sum to 1 within each group, and the indices of the flat-weight groups.
    """
    if config.group_edges_kev is None and config.group_edges_method == "scattered_fluence":
        bin_edges, photons = scattered_fluence_spectrum(
            spectrum, backend, energy_min_kev=float(config.energy_min_kev),
            object_size_cm=float(config.grouping_object_size_cm), medium=grouping_medium_densities(config),
            fine_bin_width_kev=fine_bin_width_kev, materials=config.materials)
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
    """Return the antiderivative in ``epsilon = E'/E`` of the Klein-Nishina cross section per unit epsilon.

    It omits the factor ``pi r_e^2 / k``; the integrand is ``epsilon + 1/epsilon - sin^2(theta)`` with
    ``cos(theta) = 1 + 1/k - 1/(k epsilon)``.

    Args:
        epsilon (np.ndarray): energy ratio E'/E in ``[1/(1 + 2k), 1]``, broadcastable with ``k``.
        k (np.ndarray): photon energy in units of the electron rest energy, ``E / m_e c^2``.

    Returns:
        np.ndarray: the antiderivative, in the broadcast shape of ``epsilon`` and ``k``.
    """
    a = 1.0 + 1.0 / k
    return (0.5 * epsilon * epsilon + (1.0 - 2.0 * a / k) * np.log(epsilon) + (a * a - 1.0) * epsilon
            - 1.0 / (k * k * epsilon))


def grouping_medium_densities(config: SolverConfig) -> tuple[tuple[str, float], ...]:
    """Return the medium of the "scattered_fluence" balance as (material name, density) pairs, as in the cache key.

    The entries are sorted by name, the densities rounded to 3 significant digits and traces below 1e-3 of the largest
    density dropped, so that nearly equal media share a cache entry. ``config.grouping_medium`` "auto" (not yet replaced
    by a measurement) gives water at 1.0 g/cm^3, and a material name (of the catalogue or of a ``Material`` of
    ``config.materials``) gives that material at its nominal density.

    Args:
        config (SolverConfig): solver configuration; uses grouping_medium.

    Returns:
        tuple[tuple[str, float], ...]: (material name, density in g/cm^3) per material.
    """
    medium = config.grouping_medium
    if isinstance(medium, str):
        name = "water" if medium == "auto" else medium
        entries = [(name, material_definition(name, config.materials).density)]
    else:
        entries = [(str(name), float(density)) for name, density in medium]
    largest = max((density for _, density in entries), default=0.0)
    rounded = sorted((name, float(f"{density:.3g}")) for name, density in entries
                     if density > 0 and density >= 1e-3 * largest)
    if not rounded:
        raise ValueError(f"grouping_medium {medium!r} has no material of positive density")
    return tuple(rounded)


def scattered_fluence_spectrum(spectrum: tuple[np.ndarray, np.ndarray], backend: Optional[XcomXraylibBackend] = None,
                               *, energy_min_kev: float = 10.0, object_size_cm: float = 20.0,
                               medium: Sequence[tuple[str, float]] = (("water", 1.0),),
                               fine_bin_width_kev: float | None = None,
                               max_bins: int = 2000, materials: Sequence = ()) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the total photon fluence spectrum (source plus scattered photons) inside the object for grouping.

    The estimate solves the slowing-down balance in energy alone by one sweep from the highest bin down:
    ``phi(E) = [S(E) + int_{E' > E} mu_C(E') p(E | E') phi(E') dE'] / (mu_t(E) - mu_R(E) + 1/L)``.

    Args:
        spectrum (tuple[np.ndarray, np.ndarray]): (energies [K] in keV, photon density per keV [K]) of the source S.
        backend (XcomXraylibBackend or None, optional): photon data. Defaults to None (loaded by ``resolve_backend``).
        energy_min_kev (float, optional): lowest energy in keV; photons below it are absorbed. Defaults to 10.0.
        object_size_cm (float, optional): characteristic size L of the object in cm (the leakage cross section is
            1/L). Defaults to 20.0.
        medium (Sequence[tuple[str, float]], optional): (material name, density in g/cm^3) per material of the
            medium. Defaults to (("water", 1.0),).
        fine_bin_width_kev (float or None, optional): minimum bin width in keV; None for
            ``FINE_BIN_WIDTH_FLOOR_KEV``. Defaults to None.
        max_bins (int, optional): maximum number of bins; the bins widen to ``(kVp - energy_min_kev) / max_bins`` if
            needed. Defaults to 2000.
        materials (Sequence[str or Material], optional): materials that define the names of ``medium`` that are not
            catalogue names (``material_definition``), e.g. ``config.materials``. Defaults to ().

    Returns:
        (np.ndarray, np.ndarray): bin edges [n+1] in keV, ascending, and phi integrated over each bin [n], in units of
        the source spectrum times cm.
    """
    backend = resolve_backend() if backend is None else backend
    density = _Spectrum(*spectrum)
    energy_max, energy_min = density.max_energy(), float(energy_min_kev)
    if not energy_max > energy_min:
        raise ValueError(f"spectrum max energy {energy_max} <= energy_min_kev {energy_min}")
    if not object_size_cm > 0:
        raise ValueError(f"object_size_cm must be > 0, got {object_size_cm}")
    minimum = FINE_BIN_WIDTH_FLOOR_KEV if fine_bin_width_kev is None else float(fine_bin_width_kev)
    width = max(minimum, (energy_max - energy_min) / int(max_bins))
    n = max(1, int(math.ceil((energy_max - energy_min) / width - 1e-9)))
    bin_edges = np.linspace(energy_min, energy_max, n + 1)
    low, high = bin_edges[:-1], bin_edges[1:]
    E = 0.5 * (low + high)
    source = density.cumulative_integral(high) - density.cumulative_integral(low)
    element_density: dict[int, float] = {}  # g/cm^3 of every element in the medium
    for name, material_density in medium:
        try:
            composition = material_definition(name, materials).composition
        except ValueError:
            raise ValueError(f"unknown material {name!r} in the grouping medium (not in the catalogue, "
                             f"cross_sections.MATERIAL_CATALOGUE, nor a Material of the reference materials)") from None
        for Z, fraction in parse_formula(composition, backend.atomic_mass).items():
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


FINE_BINS_PER_GROUP = 100  # bins per group of the automatic fine energy grid (``fine_bin_width_kev=None``)
FINE_BIN_WIDTH_FLOOR_KEV = 0.5  # its narrowest bin


def fine_energy_grid(edges: np.ndarray, fine_bin_width_kev: float | None = None
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cut every group into equal fine energy bins; no bin straddles a group edge.

    Args:
        edges (np.ndarray): [G+1] group edges in keV, descending.
        fine_bin_width_kev (float or None, optional): maximum bin width in keV; None cuts every group into at most
            ``FINE_BINS_PER_GROUP`` bins of at least ``FINE_BIN_WIDTH_FLOOR_KEV``. Defaults to None.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray): bin midpoints [K] in keV, group index [K] int64, lower edges
        [K] and upper edges [K] in keV, ordered by group and then by ascending energy.
    """
    midpoints, groups, lows, highs = [], [], [], []
    for g in range(edges.size - 1):
        high, low = float(edges[g]), float(edges[g + 1])
        width = (fine_bin_width_kev if fine_bin_width_kev is not None
                 else max(FINE_BIN_WIDTH_FLOOR_KEV, (high - low) / FINE_BINS_PER_GROUP))
        n = max(1, int(math.ceil((high - low) / width - 1e-9)))
        bin_edges = np.linspace(low, high, n + 1)
        lows.append(bin_edges[:-1])
        highs.append(bin_edges[1:])
        midpoints.append(0.5 * (bin_edges[:-1] + bin_edges[1:]))
        groups.append(np.full(n, g, dtype=np.int64))
    return np.concatenate(midpoints), np.concatenate(groups), np.concatenate(lows), np.concatenate(highs)


def group_of_energy(E: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return the group index of each energy; group g holds ``edges[g+1] <= E < edges[g]``.

    Args:
        E (np.ndarray): energies in keV.
        edges (np.ndarray): [G+1] group edges in keV, descending.

    Returns:
        np.ndarray: group index per energy, -1 below the lowest edge and 0 at or above the top edge.
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
    """Tabulate the Legendre polynomials P_l(eta) for l = 0..L by recurrence.

    Args:
        eta (np.ndarray): [N] cosines of the scattering angle.
        L (int): highest Legendre order.

    Returns:
        np.ndarray: [L+1, N] values P_l(eta).
    """
    eta = np.asarray(eta, dtype=np.float64)
    P = np.empty((L + 1, eta.size))
    P[0] = 1.0
    if L >= 1:
        P[1] = eta
    for l in range(1, L):
        P[l + 1] = ((2 * l + 1) * eta * P[l] - l * P[l - 1]) / (l + 1)
    return P


def eta_quadrature_in_q(q_breakpoints: np.ndarray, gauss_legendre_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a piecewise Gauss-Legendre quadrature in eta, built in ``q = sqrt((1 - eta)/2)`` between q breakpoints.

    The weights include the Jacobian ``d eta = 4 q dq``, and the nodes of a piece lie strictly inside it.

    Args:
        q_breakpoints (np.ndarray): [P] sorted breakpoints in q; empty pieces are skipped.
        gauss_legendre_nodes (int): Gauss-Legendre nodes per piece.

    Returns:
        (np.ndarray, np.ndarray): nodes eta [N] and weights [N] such that ``sum w f(eta)`` approximates the integral of
        f over the eta range of the breakpoints.
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
    """Diagnostics of a library build, returned by ``compute_library`` and not cached.

    Shapes: M materials, G groups, K fine energies of the build, L Legendre order, N table nodes.

    Attributes:
        backend (str): name of the photon data backend ("xcom-xraylib").
        backend_reason (str): where the backend's data came from.
        composition (dict[str, dict[int, float]]): mass fraction per atomic number Z of each material.
        compton_below_cutoff (np.ndarray): [M, G] group-weighted Compton scattering to energies below the lowest group
            edge in cm^2/g; it stays in ``mu_t`` and is absorbed.
        sigma_compton_group (np.ndarray): [M, G] group-weighted Compton mass cross section in cm^2/g.
        sigma_rayleigh_group (np.ndarray): [M, G] group-weighted Rayleigh mass cross section in cm^2/g.
        sigma_photoelectric_group (np.ndarray): [M, G] group-weighted photoelectric mass cross section in cm^2/g.
        sigma_total_group (np.ndarray): [M, G] group-weighted total mass attenuation in cm^2/g.
        fine_energy (np.ndarray): [K] fine bin midpoints in keV.
        fine_group (np.ndarray): [K] group index of each fine bin.
        fine_weight (np.ndarray): [K] within-group weights that sum to 1 in each group (not the library's
            ``fine_weight``, which is normalized over all groups).
        flat_fallback_groups (list[int]): groups whose weight was zero and that got a flat weight. Defaults to [].
        pdf_normalization_max_relative_deviation (float or None): largest relative deviation between the backend's
            angular normalization and the quadrature's, or None when the backend has none. Defaults to None.
        rayleigh_l (np.ndarray or None): [M, G, L+1] within-group Rayleigh part of ``mu_s_l``. Defaults to None.
        rayleigh_eta (np.ndarray or None): [M, G, N] within-group Rayleigh part of ``mu_s_eta``. Defaults to None.
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


def spectrum_arrays(spectrum: tuple[np.ndarray, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray]:
    """Return a spectrum as flat float64 arrays; None raises, there is no default spectrum.

    Args:
        spectrum (tuple[np.ndarray, np.ndarray] or None): (energies [K] in keV, photon density per keV [K]); one node
            is a monoenergetic line.

    Returns:
        (np.ndarray, np.ndarray): energies [K] in keV and photon densities per keV [K], float64.
    """
    if spectrum is None:
        raise ValueError("a source spectrum is required: (energies [K] in keV, photon density per keV [K]); load a "
                         "two-column file with load_spectrum_txt, or use monoenergetic_spectrum(energy_kev)")
    e, w = spectrum
    return np.asarray(e, dtype=np.float64).ravel(), np.asarray(w, dtype=np.float64).ravel()


def compute_library(
    config: SolverConfig,
    spectrum: tuple[np.ndarray, np.ndarray],
    backend: XcomXraylibBackend | None = None,
    *,
    fine_bin_width_kev: float | None = None,
    gauss_legendre_nodes: int = 32,
    materials: Optional[MaterialTable] = None,
) -> tuple[MultigroupLibrary, BuildInfo]:
    """Build the multigroup library without caching and return the build diagnostics with it.

    Args:
        config (SolverConfig): solver configuration; uses num_groups, group_edges_kev, group_edges_method,
            grouping_object_size_cm, grouping_medium, energy_min_kev, legendre_order, num_eta_table, angular_table_grid.
        spectrum (tuple[np.ndarray, np.ndarray]): (energies [K] in keV, photon density per keV [K]); one node is a
            monoenergetic line.
        backend (XcomXraylibBackend or None, optional): photon data. Defaults to None (loaded by ``resolve_backend``).
        fine_bin_width_kev (float or None, optional): maximum fine energy bin width in keV; None for the automatic
            grid (``fine_energy_grid``). Defaults to None.
        gauss_legendre_nodes (int, optional): Gauss-Legendre nodes per angular piece. Defaults to 32.
        materials (Sequence or None, optional): (name, nominal density in g/cm^3, formula), a catalogue name or a
            ``Material`` per material, in any order (sorted by ``sort_material_table``). Defaults to None, the seven
            default reference materials.

    Returns:
        (MultigroupLibrary, BuildInfo): the library with float64 CPU tensors and the build diagnostics.
    """
    backend = resolve_backend() if backend is None else backend
    spectrum_energies, spectrum_weights = spectrum_arrays(spectrum)
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
    E = spectrum_density.place_line(E, bin_low, bin_high)  # a monoenergetic line is evaluated at its energy
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


def _direct_transfer_table(eta: np.ndarray, backend: XcomXraylibBackend, elements: Sequence[int], E: np.ndarray,
                           fine_groups: np.ndarray, bin_low: np.ndarray, bin_high: np.ndarray,
                           within_group_weight: np.ndarray, edges: np.ndarray, compton_scales: np.ndarray,
                           rayleigh_scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tabulate the per-steradian transfer at the cosines eta, Compton plus within-group Rayleigh, in cm^2/g/sr.

    Compton scattering of every fine bin enters each outgoing group with the exact fraction of the bin that lands in it.

    Args:
        eta (np.ndarray): [n] cosines of the scattering angle, any order.
        backend (XcomXraylibBackend): photon data; supplies the raw angular shapes.
        elements (Sequence[int]): [NZ] atomic numbers of the elements of the materials.
        E (np.ndarray): [K] fine bin midpoints in keV.
        fine_groups (np.ndarray): [K] group index of each fine bin.
        bin_low (np.ndarray): [K] lower bin edges in keV.
        bin_high (np.ndarray): [K] upper bin edges in keV.
        within_group_weight (np.ndarray): [K] weight of each fine bin within its group.
        edges (np.ndarray): [G+1] group edges in keV, descending.
        compton_scales (np.ndarray): [K, NZ, M] scale of each element's raw Compton shape per material.
        rayleigh_scales (np.ndarray): [K, NZ, M] scale of each element's raw Rayleigh shape per material.

    Returns:
        (np.ndarray, np.ndarray): ``mu_s_eta`` [M, G, G, n] and its within-group Rayleigh part [M, G, n].
    """
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
    """Return the moving average of x over ``width`` samples; the window shrinks at the ends, so constants stay.

    Args:
        x (np.ndarray): [n] values.
        width (int): window length in samples (odd).

    Returns:
        np.ndarray: [n] smoothed values.
    """
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
    """Place the nodes of an adaptive transfer table: the map ``u = h(eta)`` at equally spaced knots of sin(theta/2).

    The node density in q is the largest over the transfer curves of ``sqrt(|f''(q)| / max f)``, smoothed and mixed with
    a floor of the uniform-table density, and h is one minus its normalized cumulative integral (table node j sits at
    ``u = j / (N - 1)``).

    Args:
        q_nodes (np.ndarray): [Nr] ascending q of the reference angular quadrature of the build.
        compton_shapes (np.ndarray): [K, M, Nr] per-steradian Compton cross section per fine energy and material at
            ``q_nodes`` in cm^2/g/sr.
        rayleigh_shapes (np.ndarray): [K, M, Nr] the same for Rayleigh scattering.
        fine_groups (np.ndarray): [K] group index of each fine bin.
        bin_low (np.ndarray): [K] lower bin edges in keV.
        bin_high (np.ndarray): [K] upper bin edges in keV.
        within_group_weight (np.ndarray): [K] weight of each fine bin within its group.
        edges (np.ndarray): [G+1] group edges in keV, descending.
        num_knots (int, optional): number of knots of the map. Defaults to ANGULAR_MAP_KNOTS.
        num_reference (int, optional): points uniform in q on which the curves are evaluated. Defaults to 20001.
        floor (float, optional): fraction of the node density taken from a table uniform in eta. Defaults to 0.1.
        negligible (float, optional): curves below this fraction of the largest curve of the same material and source
            group are ignored. Defaults to 1.0e-6.

    Returns:
        np.ndarray: [num_knots] float64 map values, strictly decreasing from 1 (eta = 1) to 0 (eta = -1).
    """
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


def transfer_at_eta(config: SolverConfig, eta: np.ndarray, spectrum: tuple[np.ndarray, np.ndarray],
                    backend: Optional[XcomXraylibBackend] = None, *, fine_bin_width_kev: float | None = None,
                    gauss_legendre_nodes: int = 32, materials: Optional[MaterialTable] = None) -> np.ndarray:
    """Return the exact per-steradian transfer at the cosines eta of the library that ``compute_library`` builds.

    This is the quantity ``mu_s_eta`` tabulates, without interpolation, for measuring the interpolation error of tables.

    Args:
        config (SolverConfig): solver configuration, as for ``compute_library``.
        eta (np.ndarray): [n] cosines of the scattering angle.
        spectrum (tuple[np.ndarray, np.ndarray]): (energies [K] in keV, photon density per keV [K]); one node is a
            monoenergetic line.
        backend (XcomXraylibBackend or None, optional): photon data. Defaults to None (loaded by ``resolve_backend``).
        fine_bin_width_kev (float or None, optional): maximum fine energy bin width in keV; None for the automatic
            grid (``fine_energy_grid``). Defaults to None.
        gauss_legendre_nodes (int, optional): Gauss-Legendre nodes per angular piece. Defaults to 32.
        materials (Sequence or None, optional): (name, nominal density in g/cm^3, formula) per material. Defaults to
            None, the seven default reference materials.

    Returns:
        np.ndarray: [M, G, G, n] transfer in cm^2/g/sr.
    """
    backend = resolve_backend() if backend is None else backend
    spectrum_energies, spectrum_weights = spectrum_arrays(spectrum)
    spectrum_density = _Spectrum(spectrum_energies, spectrum_weights)
    edges = make_group_edges(config, (spectrum_energies, spectrum_weights), backend,
                             fine_bin_width_kev=fine_bin_width_kev)
    G = edges.size - 1
    table = sort_material_table(config, materials, (spectrum_energies, spectrum_weights), backend,
                                fine_bin_width_kev=fine_bin_width_kev)
    names = tuple(n for n, _, _ in table)
    E, fine_groups, bin_low, bin_high = fine_energy_grid(edges, fine_bin_width_kev)
    E = spectrum_density.place_line(E, bin_low, bin_high)
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
               fine_bin_width_kev: float | None, gauss_legendre_nodes: int,
               materials: Optional[MaterialTable] = None) -> str:
    """Return the cache key of a library: a hash of the build version, configuration, spectrum and build inputs.

    Args:
        config (SolverConfig): solver configuration.
        spectrum (tuple[np.ndarray, np.ndarray]): (energies [K] in keV, photon density per keV [K]).
        backend_name (str): name of the photon data backend.
        fine_bin_width_kev (float or None): maximum fine energy bin width in keV, None for the automatic grid.
        gauss_legendre_nodes (int): Gauss-Legendre nodes per angular piece.
        materials (Sequence or None, optional): the sorted (name, nominal density, formula) table. Defaults to None,
            the seven default reference materials.

    Returns:
        str: the first 20 hexadecimal digits of the SHA-256 hash.
    """
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
        "de_kev": None if fine_bin_width_kev is None else float(fine_bin_width_kev),
        "n_gl": int(gauss_legendre_nodes),
        "materials": [list(m) for m in _material_table(materials)],  # names, nominal densities and formulas, in order
    }
    h = hashlib.sha256(json.dumps(fields, sort_keys=True).encode())
    h.update(np.ascontiguousarray(spectrum[0], dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(spectrum[1], dtype=np.float64).tobytes())
    return h.hexdigest()[:20]


def build_library(
    config: SolverConfig,
    spectrum: tuple[np.ndarray, np.ndarray],
    cache_dir: str | None = "default",
    *,
    fine_bin_width_kev: float | None = None,
    gauss_legendre_nodes: int = 32,
    materials: Optional[MaterialTable] = None,
) -> MultigroupLibrary:
    """Build the multigroup library for ``config`` and a material table, or load it from the cache.

    The cache file ``<cache_dir>/mglib_<hash>.pt`` is keyed by the build version, the configuration, the spectrum, the
    backend, the numerical parameters and the sorted material table; a cache file that does not load as a complete
    library is rebuilt and overwritten.

    Args:
        config (SolverConfig): solver configuration, as for ``compute_library``.
        spectrum (tuple[np.ndarray, np.ndarray]): (energies [K] in keV, photon density per keV [K]); one node is a
            monoenergetic line.
        cache_dir (str or None, optional): cache folder; "default" for ``default_cache_dir()``, None for no cache.
            Defaults to "default".
        fine_bin_width_kev (float or None, optional): maximum fine energy bin width in keV; None for the automatic
            grid (``fine_energy_grid``). Defaults to None.
        gauss_legendre_nodes (int, optional): Gauss-Legendre nodes per angular piece. Defaults to 32.
        materials (Sequence or None, optional): (name, nominal density in g/cm^3, formula), a catalogue name or a
            ``Material`` per material, in any order (sorted by ``sort_material_table``). Defaults to None, the seven
            default reference materials.

    Returns:
        MultigroupLibrary: the library with float64 CPU tensors in cm^2/g and cm^2/g/sr, per unit density.
    """
    if cache_dir == "default":
        cache_dir = default_cache_dir()
    backend = resolve_backend()
    spectrum_pair = spectrum_arrays(spectrum)
    table = sort_material_table(config, materials, spectrum_pair, backend, fine_bin_width_kev=fine_bin_width_kev)
    path = None
    if cache_dir is not None:
        key = _cache_key(config, spectrum_pair, backend.name, fine_bin_width_kev, gauss_legendre_nodes, table)
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
    library, _ = compute_library(config, spectrum_pair, backend, fine_bin_width_kev=fine_bin_width_kev,
                                 gauss_legendre_nodes=gauss_legendre_nodes, materials=table)
    if path is not None:
        os.makedirs(cache_dir, exist_ok=True)  # type: ignore[arg-type]
        tmp = path + ".tmp"
        library.save(tmp)
        os.replace(tmp, path)
    return library
