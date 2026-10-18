"""Multigroup cross-section library: material catalogue, container, per-voxel coefficients and a synthetic test library.

``MATERIAL_CATALOGUE`` maps every known material name to its nominal density and composition, and
:class:`MultigroupLibrary` holds the per-unit-density group constants, angular transfer tables and fine-energy
attenuation table that :func:`~citrus.scatter_solver.cross_section_library.build_library` fills. The per-voxel
functions turn density maps into attenuation, transfer and group-integrated transmission, and
:func:`synthetic_library` gives a fake but self-consistent library for tests that must not depend on the photon data.
The catalogue and the conventions are described in the documentation page "Conventions" (sections "Materials" and
"Scattering convention").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Union

import torch
import xraylib
from torch import Tensor

# ---- material catalogue (docs/source/conventions.rst, "Materials") ----------------------------------------------
# the material strings of XrayPhysics (the materials of citrus <= 0.6, verbatim; the air terms are read as mass
# fractions of the compounds, `cross_section_library.parse_formula`)
_AIR_DRY = "78.084*N2 + 20.946*O2 + 0.9340*Ar + 0.0397*CO2"
_LUNG_ICRP = (
    "0.10132562304283013*H + 0.10235810831090612*C + 0.028663471831760927*N + 0.75742799115584314*O + "
    "0.0018408652066471242*Na + 0.00073034326133282642*Mg + 0.00080037617680309744*P + 0.0022510579972587113*S + "
    "0.0026612507878702989*Cl + 0.0019409122287475113*K"
)
_ADIPOSE_ICRP = (
    "0.119477*H + 0.637240*C + 0.007970*N + 0.232333*O + 0.000500*Na + 0.000020*Mg + 0.000160*P + 0.000730*S + "
    "0.001190*Cl + 0.000320*K + 0.000020*Ca + 0.000020*Fe + 0.000020*Zn"
)
_BONE_CORTICAL_ICRP = (
    "0.047234*H + 0.144330*C + 0.041990*N + 0.446096*O + 0.002200*Mg + 0.104970*P + "
    "0.003150*S + 0.209930*Ca + 0.000100*Zn"
)
_PMMA = "C5O2H8"
_POLYETHYLENE = "C2H4"
# ICRU Report 44 (1989) mass fractions (as tabulated in NISTIR 5632, Hubbell and Seltzer)
_MUSCLE_ICRU44 = "0.102*H + 0.143*C + 0.034*N + 0.710*O + 0.001*Na + 0.002*P + 0.003*S + 0.001*Cl + 0.004*K"
_SOFT_TISSUE_ICRU44 = "0.102*H + 0.143*C + 0.034*N + 0.708*O + 0.002*Na + 0.003*P + 0.003*S + 0.002*Cl + 0.003*K"
_BRAIN_ICRU44 = "0.107*H + 0.145*C + 0.022*N + 0.712*O + 0.002*Na + 0.004*P + 0.002*S + 0.003*Cl + 0.003*K"
_BLOOD_ICRU44 = (
    "0.102*H + 0.110*C + 0.033*N + 0.745*O + 0.001*Na + 0.001*P + 0.002*S + 0.003*Cl + 0.002*K + 0.001*Fe"
)
_SKIN_ICRU44 = "0.100*H + 0.204*C + 0.042*N + 0.645*O + 0.002*Na + 0.001*P + 0.002*S + 0.003*Cl + 0.001*K"
# Woodard and White (1986), Br J Radiol 59:1209-1219, mass fractions (also ICRU Report 46, 1992)
_SPONGIOSA_WOODARD_WHITE = (
    "0.085*H + 0.404*C + 0.028*N + 0.367*O + 0.001*Na + 0.001*Mg + 0.034*P + 0.002*S + 0.002*Cl + 0.001*K + "
    "0.074*Ca + 0.001*Fe"
)
_CARTILAGE_WOODARD_WHITE = "0.096*H + 0.099*C + 0.022*N + 0.744*O + 0.005*Na + 0.022*P + 0.009*S + 0.003*Cl"
# carbon-fibre epoxy laminate: 60 % fibre volume, carbon fibre 1.8 g/cm^3 in bisphenol-A epoxy C21H24O4 1.2 g/cm^3
# (mass fraction 0.6 x 1.8 / (0.6 x 1.8 + 0.4 x 1.2) = 0.692, density 1.56 g/cm^3)
_CARBON_FIBRE_EPOXY = "0.692*C + 0.308*C21H24O4"

# Elements Z = 1..98 (those with xraylib scattering functions, `photon_data.SCATTERING_ELEMENTS`): xraylib's symbols,
# English (IUPAC) names and xraylib's element densities (g/cm^3: NIST's densities of Hubbell and Seltzer, NISTIR 5632,
# Table 1, gases at 20 C and 1 atm, for Z <= 92)
NUM_ELEMENTS = 98
ELEMENT_SYMBOLS: tuple[str, ...] = tuple(xraylib.AtomicNumberToSymbol(Z) for Z in range(1, NUM_ELEMENTS + 1))
ELEMENT_NAMES: tuple[str, ...] = tuple((
    "hydrogen helium lithium beryllium boron carbon nitrogen oxygen fluorine neon sodium magnesium aluminium silicon "
    "phosphorus sulfur chlorine argon potassium calcium scandium titanium vanadium chromium manganese iron cobalt "
    "nickel copper zinc gallium germanium arsenic selenium bromine krypton rubidium strontium yttrium zirconium "
    "niobium molybdenum technetium ruthenium rhodium palladium silver cadmium indium tin antimony tellurium iodine "
    "xenon caesium barium lanthanum cerium praseodymium neodymium promethium samarium europium gadolinium terbium "
    "dysprosium holmium erbium thulium ytterbium lutetium hafnium tantalum tungsten rhenium osmium iridium platinum "
    "gold mercury thallium lead bismuth polonium astatine radon francium radium actinium thorium protactinium uranium "
    "neptunium plutonium americium curium berkelium californium"
).split())
ELEMENT_DENSITIES: tuple[float, ...] = tuple(float(xraylib.ElementDensity(Z)) for Z in range(1, NUM_ELEMENTS + 1))

_COMPOUNDS: dict[str, tuple[float, str]] = {
    # name: (nominal density g/cm^3, composition); the first seven are the default reference materials `MATERIALS`
    "air": (0.0012, _AIR_DRY),
    "lung": (0.26, _LUNG_ICRP),
    "adipose": (0.95, _ADIPOSE_ICRP),
    "water": (1.0, "H2O"),
    "muscle": (1.05, _MUSCLE_ICRU44),
    "soft_tissue": (1.06, _SOFT_TISSUE_ICRU44),
    "bone": (1.85, _BONE_CORTICAL_ICRP),
    "trabecular_bone": (1.18, _SPONGIOSA_WOODARD_WHITE),
    "brain": (1.04, _BRAIN_ICRU44),
    "blood": (1.06, _BLOOD_ICRU44),
    "cartilage": (1.10, _CARTILAGE_WOODARD_WHITE),
    "skin": (1.09, _SKIN_ICRU44),
    "pmma": (1.18, _PMMA),
    "polyethylene": (0.94, _POLYETHYLENE),
    "aluminium": (2.699, "Al"),
    "titanium": (4.54, "Ti"),
    "carbon_fibre": (1.56, _CARBON_FIBRE_EPOXY),
}
# the compounds and tissues first, then every element by its English name (aluminium and titanium are already entries
# of the first part, with the same composition)
MATERIAL_CATALOGUE: Mapping[str, tuple[float, str]] = MappingProxyType({
    **_COMPOUNDS,
    **{name: (density, symbol) for name, symbol, density in zip(ELEMENT_NAMES, ELEMENT_SYMBOLS, ELEMENT_DENSITIES)
       if name not in _COMPOUNDS},
})
# the entries that are not single elements (the compounds, tissues and mixtures, plus aluminium and titanium)
COMPOUND_NAMES: tuple[str, ...] = tuple(_COMPOUNDS)

# The default reference materials, in library order = increasing attenuation at nominal density. Muscle is BELOW soft
# tissue at
# every energy (ICRU-44 muscle has the lower density and nearly the same composition: attenuation ratio 0.986-0.991
# over 10-125 keV).
MATERIALS: tuple[str, ...] = ("air", "lung", "adipose", "water", "muscle", "soft_tissue", "bone")
MATERIAL_TABLE: tuple[tuple[str, float], ...] = tuple((n, MATERIAL_CATALOGUE[n][0]) for n in MATERIALS)
NOMINAL_DENSITIES: tuple[float, ...] = tuple(float(d) for _, d in MATERIAL_TABLE)
# rows of water and bone in the DEFAULT reference materials (3, 6); for any library use material_index(library, "water")
WATER, BONE = MATERIALS.index("water"), MATERIALS.index("bone")
ELECTRON_REST_KEV = 510.998950


def check_material_names(names: Sequence[str], minimum: int = 2) -> tuple[str, ...]:
    """Check reference material names and return them as a tuple.

    The names must be a sequence (not a single string) of at least ``minimum`` distinct names of ``MATERIAL_CATALOGUE``.

    Args:
        names (Sequence[str]): catalogue material names.
        minimum (int, optional): smallest number of names. Defaults to 2.

    Returns:
        tuple[str, ...]: the names.

    Raises:
        ValueError: naming the problem (and the catalogue, for an unknown name).
    """
    if isinstance(names, str) or not isinstance(names, Sequence):
        raise ValueError(f"materials must be a sequence of material names, e.g. ('air', 'water', 'bone'), got "
                         f"{names!r}")
    names = tuple(names)
    unknown = [n for n in names if not isinstance(n, str) or n not in MATERIAL_CATALOGUE]
    if unknown:
        raise ValueError(f"unknown material name(s) {unknown}; the catalogue has {COMPOUND_NAMES} and every element "
                         f"by its English name (e.g. 'lead', 'tungsten')")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate material names in {names}")
    if len(names) < minimum:
        raise ValueError(f"the density model interpolates between at least {minimum} reference materials, got {names}")
    return names


def material_table(names: Sequence[str] = MATERIALS) -> tuple[tuple[str, float, str], ...]:
    """Return (name, nominal density, composition) of catalogue materials, in the given order.

    Args:
        names (Sequence[str], optional): catalogue material names, checked by ``check_material_names`` with one name
            allowed. Defaults to MATERIALS.

    Returns:
        tuple[tuple[str, float, str], ...]: (name, nominal density in g/cm^3, composition formula) per material.

    Raises:
        ValueError: for an unknown or repeated name.
    """
    return tuple((n, float(MATERIAL_CATALOGUE[n][0]), MATERIAL_CATALOGUE[n][1])
                 for n in check_material_names(names, minimum=1))


def material_index(materials: Union["MultigroupLibrary", Sequence[str]], name: str) -> int:
    """Return the row of a material in a library or in a sequence of names.

    Args:
        materials (MultigroupLibrary or Sequence[str]): a library (its ``materials``) or material names.
        name (str): material name.

    Returns:
        int: row index of ``name``.

    Raises:
        ValueError: if ``name`` is absent, naming the available materials.
    """
    names = tuple(materials.materials if isinstance(materials, MultigroupLibrary) else materials)
    if name not in names:
        raise ValueError(f"material {name!r} is not among the materials {names}")
    return names.index(name)


@dataclass
class MultigroupLibrary:
    """Per-unit-density (mass) multigroup coefficients of the reference materials.

    Shapes: M materials, G groups, L Legendre order, N table nodes, K map knots, E fine energies. All floating tensors
    are float64 on CPU when built.

    Attributes:
        group_edges (Tensor): [G+1] group edges in keV, descending.
        group_energy (Tensor): [G] spectrum-weighted mean energy of each group in keV.
        spectrum_fraction (Tensor): [G] fraction f_g of the source photons per steradian in each group; sums to 1.
        mu_t (Tensor): [M, G] total mass attenuation in cm^2/g.
        mu_pe (Tensor): [M, G] photoelectric mass attenuation in cm^2/g.
        mu_s_l (Tensor): [M, G, G, L+1] Legendre moments of the transfer, [m, g_in, g_out, l], in cm^2/g.
        eta_grid (Tensor): [N] cosines of the scattering angle at the table nodes, ascending from -1 to 1.
        mu_s_eta (Tensor): [M, G, G, N] per-steradian transfer (Compton plus Rayleigh) at the nodes in cm^2/g/sr.
        fine_energy (Tensor): [E] fine bin midpoints in keV, ordered by group and then by ascending energy.
        fine_mu_t (Tensor): [M, E] total mass attenuation at ``fine_energy`` in cm^2/g.
        fine_weight (Tensor): [E] source photons per steradian in each fine bin; sums to 1, and to
            ``spectrum_fraction[g]`` over the bins of group g.
        fine_group (Tensor): [E] int64 group index of each fine energy.
        materials (tuple[str, ...]): M material names in library order (increasing attenuation at nominal density).
            Defaults to () (filled in from M).
        nominal_density (Tensor or None): [M] nominal densities in g/cm^3. Defaults to None (filled in from M).
        eta_map (Tensor or None): [K] node placement of an adaptive table (``angular_table_position``); None for a
            uniform table. Defaults to None.
    """

    group_edges: Tensor
    group_energy: Tensor
    spectrum_fraction: Tensor
    mu_t: Tensor
    mu_pe: Tensor
    mu_s_l: Tensor
    eta_grid: Tensor
    mu_s_eta: Tensor
    fine_energy: Tensor
    fine_mu_t: Tensor
    fine_weight: Tensor
    fine_group: Tensor
    materials: tuple[str, ...] = ()
    nominal_density: Optional[Tensor] = None
    eta_map: Optional[Tensor] = None

    def __post_init__(self) -> None:
        """Fill in default material names and nominal densities and check the shapes.

        Empty ``materials`` become ``MATERIALS`` for M = 7, ("water", "bone") for M = 2 and "m0", "m1", ... otherwise; a
        missing ``nominal_density`` becomes ``NOMINAL_DENSITIES`` for the names ``MATERIALS``, (1.0, 1.85) for ("water",
        "bone") and 1, 2, ..., M otherwise.

        Raises:
            ValueError: if the names, ``nominal_density`` or ``eta_map`` do not match M.
        """
        M = int(self.mu_t.shape[0])
        if not self.materials:
            self.materials = (MATERIALS if M == len(MATERIALS) else ("water", "bone") if M == 2
                              else tuple(f"m{m}" for m in range(M)))
        self.materials = tuple(self.materials)
        if len(self.materials) != M:
            raise ValueError(f"{len(self.materials)} material names {self.materials} for {M} materials")
        if self.nominal_density is None:
            densities = (NOMINAL_DENSITIES if self.materials == MATERIALS
                         else (1.0, 1.85) if self.materials == ("water", "bone")
                         else tuple(float(m + 1) for m in range(M)))
            self.nominal_density = torch.tensor(densities, dtype=torch.float64, device=self.mu_t.device)
        if tuple(self.nominal_density.shape) != (M,):
            raise ValueError(f"nominal_density must be [{M}], got {tuple(self.nominal_density.shape)}")
        if self.eta_map is not None and (self.eta_map.dim() != 1 or self.eta_map.shape[0] < 2):
            raise ValueError(f"eta_map must be [K] with K >= 2, got {tuple(self.eta_map.shape)}")

    # ---- derived ----------------------------------------------------------------------------
    @property
    def num_materials(self) -> int:
        """Number of materials M."""
        return int(self.mu_t.shape[0])

    @property
    def num_groups(self) -> int:
        """Number of energy groups G."""
        return int(self.mu_t.shape[1])

    @property
    def legendre_order(self) -> int:
        """Legendre order L of ``mu_s_l``."""
        return int(self.mu_s_l.shape[-1]) - 1

    @property
    def num_fine_energies(self) -> int:
        """Number of fine energies E."""
        return int(self.fine_energy.shape[0])

    @property
    def mu_s(self) -> Tensor:
        """Group-to-group scattering coefficient [M, G_in, G_out] in cm^2/g, the l = 0 moment."""
        return self.mu_s_l[..., 0]

    def to(self, device=None, dtype=None) -> "MultigroupLibrary":
        """Return a copy of the library on another device and/or dtype.

        Args:
            device (torch.device or str, optional): target device. Defaults to None (unchanged).
            dtype (torch.dtype, optional): target floating dtype; ``fine_group`` stays int64. Defaults to None
                (unchanged).

        Returns:
            MultigroupLibrary: the converted library.
        """
        keywords = {}
        if device is not None:
            keywords["device"] = device
        if dtype is not None:
            keywords["dtype"] = dtype
        return MultigroupLibrary(
            group_edges=self.group_edges.to(**keywords),
            group_energy=self.group_energy.to(**keywords),
            spectrum_fraction=self.spectrum_fraction.to(**keywords),
            mu_t=self.mu_t.to(**keywords),
            mu_pe=self.mu_pe.to(**keywords),
            mu_s_l=self.mu_s_l.to(**keywords),
            eta_grid=self.eta_grid.to(**keywords),
            mu_s_eta=self.mu_s_eta.to(**keywords),
            fine_energy=self.fine_energy.to(**keywords),
            fine_mu_t=self.fine_mu_t.to(**keywords),
            fine_weight=self.fine_weight.to(**keywords),
            fine_group=self.fine_group.to(device=device) if device is not None else self.fine_group,
            materials=self.materials,
            nominal_density=self.nominal_density.to(**keywords),
            eta_map=None if self.eta_map is None else self.eta_map.to(**keywords),
        )

    def save(self, path: str) -> None:
        """Write every field of the library to a file with ``torch.save``.

        Args:
            path (str): file path.
        """
        torch.save({k: getattr(self, k) for k in self.__dataclass_fields__}, path)

    @staticmethod
    def load(path: str) -> "MultigroupLibrary":
        """Load a library written by ``save``; a file without ``eta_map`` loads as a uniform table.

        Args:
            path (str): file path.

        Returns:
            MultigroupLibrary: the library.

        Raises:
            KeyError: if the file lacks a field; ``build_library`` then rebuilds its cache entry.
        """
        fields = torch.load(path)
        fields.setdefault("eta_map", None)
        missing = [k for k in MultigroupLibrary.__dataclass_fields__ if k not in fields]
        if missing:
            raise KeyError(f"library file {path!r} lacks fields {missing} (written by an older version)")
        return MultigroupLibrary(**fields)


# ---- compact per-voxel material pairs ----------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class MaterialPairs:
    """At most two materials per voxel, the output of the density model (``citrus.utilities.attenuation_to_densities``).

    The dense maps are ``rho[m] = sum_p density[p] [index[p] == m]`` (``dense``).

    Attributes:
        index (Tensor): [2, ...spatial] int32 library material index of each slot; slot 0 < slot 1 where both are used,
            and an unused slot holds index 0 with density 0.
        density (Tensor): [2, ...spatial] density of each slot in g/cm^3.
        num_materials (int): number of materials M of the dense maps.
    """

    index: Tensor
    density: Tensor
    num_materials: int

    def dense(self) -> Tensor:
        """Return the dense density maps.

        Returns:
            Tensor: [M, ...spatial] densities in g/cm^3, on the device and in the dtype of ``density``.
        """
        out = torch.zeros((self.num_materials,) + tuple(self.density.shape[1:]), device=self.density.device,
                          dtype=self.density.dtype)
        out.scatter_add_(0, self.index.long(), self.density)
        return out

    def to(self, device=None, dtype=None) -> "MaterialPairs":
        """Return a copy of the pairs on another device and/or dtype; ``index`` keeps its dtype.

        Args:
            device (torch.device or str, optional): target device. Defaults to None (unchanged).
            dtype (torch.dtype, optional): target dtype of ``density``. Defaults to None (unchanged).

        Returns:
            MaterialPairs: the converted pairs.
        """
        return MaterialPairs(self.index.to(device=device), self.density.to(device=device, dtype=dtype),
                             self.num_materials)


def material_pairs(rho: Tensor) -> MaterialPairs:
    """Return the compact pair representation of density maps with at most two nonzero materials per voxel.

    The nonzero materials keep ascending index order; one host synchronization when M > 2.

    Args:
        rho (Tensor): [M, ...spatial] densities in g/cm^3.

    Returns:
        MaterialPairs: the pairs.

    Raises:
        ValueError: if a voxel has more than two nonzero materials.
    """
    M = int(rho.shape[0])
    spatial = tuple(rho.shape[1:])
    nonzero = rho != 0
    if M > 2 and rho.numel():
        max_count = int(nonzero.sum(0).max())
        if max_count > 2:
            raise ValueError(f"material_pairs: a voxel has {max_count} nonzero materials (at most 2 allowed)")
    material_index = torch.arange(M, device=rho.device).reshape((M,) + (1,) * len(spatial)).expand(nonzero.shape)
    unused = torch.full_like(material_index, M)
    key = torch.where(nonzero, material_index, unused)
    first = key.amin(0)
    second = torch.where(key == first[None], unused, key).amin(0)
    index = torch.stack([first, second])  # M marks an unused slot
    used = index < M
    index = torch.where(used, index, torch.zeros_like(index))
    density = torch.gather(rho, 0, index) * used.to(rho.dtype)
    return MaterialPairs(index=index.to(torch.int32).contiguous(), density=density.contiguous(), num_materials=M)


def max_materials_per_point(rho: Tensor) -> int:
    """Return the largest number of nonzero materials at any point (one host synchronization).

    Args:
        rho (Tensor): [M, ...spatial] densities.

    Returns:
        int: the count, 0 for an empty or all-zero volume.
    """
    if rho.numel() == 0:
        return 0
    return int((rho != 0).sum(0).max())


def material_pairs_if_compact(rho: Tensor) -> Optional[MaterialPairs]:
    """Return ``material_pairs(rho)`` when M > 2 and every point has at most two nonzero materials, else None.

    None means that the first-collision source evaluates all M materials densely; one host synchronization when M > 2.

    Args:
        rho (Tensor): [M, ...spatial] densities in g/cm^3.

    Returns:
        MaterialPairs or None: the pairs, or None for M <= 2 or when a point mixes three or more materials.
    """
    if int(rho.shape[0]) <= 2 or max_materials_per_point(rho) > 2:
        return None
    return material_pairs(rho)


# ---- per-voxel coefficients from density maps ----------------------------------------------------

def mu_t_volume(library: MultigroupLibrary, rho: Tensor) -> Tensor:
    """Return the total attenuation per group on a grid.

    Args:
        library (MultigroupLibrary): the cross-section library.
        rho (Tensor): [M, rows, cols, slices] densities in g/cm^3.

    Returns:
        Tensor: [G, rows, cols, slices] total attenuation coefficient mu_t in cm^-1.
    """
    mu = library.mu_t.to(rho.device, rho.dtype)  # [M, G]
    return torch.einsum("mg,mijk->gijk", mu, rho)


# at most this many (fine energy, ray) pairs are evaluated at once by `group_transmission`
_TRANSMISSION_BLOCK = 1 << 22


def fine_group_weights(library: MultigroupLibrary, device=None, dtype=None) -> Tensor:
    """Return the fine-to-group summation matrix: ``W[g, e] = fine_weight[e]`` if ``fine_group[e] == g``, else 0.

    Args:
        library (MultigroupLibrary): the cross-section library.
        device (torch.device or str, optional): device of the result. Defaults to None (the library's).
        dtype (torch.dtype, optional): dtype of the result. Defaults to None (the library's).

    Returns:
        Tensor: [G, E] weights.
    """
    fine_group = library.fine_group.to(device=device)
    weights = library.fine_weight.to(device=device, dtype=dtype)
    in_group = fine_group[None, :] == torch.arange(library.num_groups, device=fine_group.device)[:, None]
    return torch.where(in_group, weights[None, :], torch.zeros((), device=weights.device, dtype=weights.dtype))


def group_transmission(library: MultigroupLibrary, path: Tensor, block: int = _TRANSMISSION_BLOCK,
                       materials: Optional[Sequence[int]] = None) -> Tensor:
    """Return the group-integrated transmission from material line integrals, exact in energy up to the fine quadrature.

    ``T_g = sum_{e: fine_group[e] = g} fine_weight[e] exp(-sum_m fine_mu_t[m, e] path[m])``, evaluated in blocks of at
    most ``block`` (fine energy, ray) pairs.

    Args:
        library (MultigroupLibrary): the cross-section library.
        path (Tensor): [M', ...] line integrals of the density maps in g/cm^2.
        block (int, optional): maximum number of (fine energy, ray) pairs per block. Defaults to ``2**22``.
        materials (Sequence[int] or None, optional): library indices of the rows of ``path`` when it holds a subset of
            the materials. Defaults to None (all, in library order).

    Returns:
        Tensor: [G, ...] source photons per steradian per group surviving the path (sum 1 over g for a zero path), in
        the dtype and on the device of ``path``.
    """
    device, dtype = path.device, path.dtype
    M = path.shape[0]
    spatial = tuple(path.shape[1:])
    paths = path.reshape(M, -1)
    num_rays = paths.shape[1]
    fine_mu_t = library.fine_mu_t.to(device=device, dtype=dtype)  # [M, E]
    if materials is not None:
        indices = [int(i) for i in materials]
        if any(i < 0 or i >= fine_mu_t.shape[0] for i in indices):
            raise ValueError(f"material indices {indices} outside the library's {fine_mu_t.shape[0]} materials")
        fine_mu_t = fine_mu_t[torch.as_tensor(indices, device=device, dtype=torch.long)]
    if fine_mu_t.shape[0] != M:
        raise ValueError(f"path has {M} materials, the library {fine_mu_t.shape[0]}")
    group_weights = fine_group_weights(library, device, dtype)  # [G, E]
    num_energies = fine_mu_t.shape[1]
    out = torch.zeros(group_weights.shape[0], num_rays, device=device, dtype=dtype)
    rays_per_block = max(1, min(num_rays, int(block)))
    energies_per_block = max(1, int(block) // rays_per_block)
    for ray_start in range(0, num_rays, rays_per_block):
        path_block = paths[:, ray_start:ray_start + rays_per_block]
        out_block = out[:, ray_start:ray_start + rays_per_block]
        for energy_start in range(0, num_energies, energies_per_block):
            mu_block = fine_mu_t[:, energy_start:energy_start + energies_per_block]  # [M, e]
            tau = mu_block[0][:, None] * path_block[0][None, :]
            for k in range(1, M):
                tau.addcmul_(mu_block[k][:, None], path_block[k][None, :])
            out_block.addmm_(group_weights[:, energy_start:energy_start + energies_per_block], tau.neg_().exp_())
    return out.reshape(group_weights.shape[0], *spatial)


def transfer_moments_volume(library: MultigroupLibrary, rho: Tensor) -> Tensor:
    """Return the Legendre transfer moments per voxel.

    Args:
        library (MultigroupLibrary): the cross-section library.
        rho (Tensor): [M, rows, cols, slices] densities in g/cm^3.

    Returns:
        Tensor: [G_in, G_out, L+1, rows, cols, slices] moments in cm^-1.
    """
    transfer = library.mu_s_l.to(rho.device, rho.dtype)  # [M, Gi, Go, L+1]
    return torch.einsum("mabl,mijk->ablijk", transfer, rho)


# knots K of the map of an adaptive angular table (`cross_section_library.angular_map`): 512 values, which stay in
# the L1 cache of the first-collision kernel
ANGULAR_MAP_KNOTS = 512


def angular_table_position(eta: Tensor, num_nodes: int, eta_map: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
    """Return the interpolation position of cosines eta in a transfer table of ``num_nodes`` nodes.

    Uniform table (``eta_map`` None): ``pos = (eta + 1) / 2 (N - 1)``; adaptive table: ``pos = u(q) (N - 1)`` with u
    interpolated linearly in the map at ``q = sin(theta/2)``. The fused first-collision kernel performs the same
    operations in the same order.

    Args:
        eta (Tensor): [...] cosines of the scattering angle, clamped to [-1, 1].
        num_nodes (int): number of table nodes N.
        eta_map (Tensor or None, optional): [K] map of an adaptive table (``MultigroupLibrary.eta_map``), cast to the
            dtype of ``eta``. Defaults to None (uniform table).

    Returns:
        (Tensor, Tensor): lower node i0 [...] int64 in [0, N - 2] and weight [...] in [0, 1] of node i0 + 1.
    """
    n = int(num_nodes)
    eta = eta.clamp(-1.0, 1.0)
    if eta_map is None:
        pos = (eta + 1.0) / 2.0 * (n - 1)
    else:
        knots = eta_map.to(device=eta.device, dtype=eta.dtype)
        K = knots.numel()
        q = ((1.0 - eta) * 0.5).clamp_min(0.0).sqrt()
        p = q * (K - 1)
        k0 = p.floor().clamp(0.0, K - 2.0)
        a = (p - k0).clamp(0.0, 1.0)
        k0 = k0.long()
        u0 = knots[k0]
        u = u0 + (knots[k0 + 1] - u0) * a
        pos = u * (n - 1)
    i0 = pos.floor().long().clamp(0, n - 2)
    w1 = (pos - i0.to(pos.dtype)).clamp(0.0, 1.0)
    return i0, w1


def angular_table_nodes(num_nodes: int, eta_map: Optional[Tensor] = None) -> Tensor:
    """Return the cosines eta of the nodes of a transfer table, ascending from -1 to 1.

    The nodes are equally spaced for ``eta_map`` None; otherwise node j sits at the table coordinate ``u = j / (N - 1)``
    of the map, ``eta_j = 1 - 2 q_j^2`` with q_j from the inverse of the piecewise-linear map.

    Args:
        num_nodes (int): number of nodes N.
        eta_map (Tensor or None, optional): [K] map of an adaptive table, strictly decreasing. Defaults to None
            (uniform table).

    Returns:
        Tensor: [N] float64 node cosines.

    Raises:
        ValueError: unless ``eta_map`` decreases strictly.
    """
    n = int(num_nodes)
    if eta_map is None:
        return torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
    u_knots = eta_map.detach().to("cpu", torch.float64)
    K = u_knots.numel()
    if not bool((u_knots[1:] < u_knots[:-1]).all()):
        raise ValueError("eta_map must decrease strictly from the first knot (eta = 1) to the last (eta = -1)")
    q_knots = torch.arange(K, dtype=torch.float64) / (K - 1)
    u = torch.arange(n, dtype=torch.float64) / (n - 1)
    ascending_u, ascending_q = u_knots.flip(0), q_knots.flip(0)  # u ascending, q descending
    j = torch.searchsorted(ascending_u, u, right=True).clamp(1, K - 1)
    u_low, u_high = ascending_u[j - 1], ascending_u[j]
    q = ascending_q[j - 1] + (ascending_q[j] - ascending_q[j - 1]) * (u - u_low) / (u_high - u_low)
    eta = 1.0 - 2.0 * q * q
    eta[0], eta[-1] = -1.0, 1.0
    return eta


def transfer_eta_at(library: MultigroupLibrary, rho: Tensor, eta: Tensor) -> Tensor:
    """Return the per-steradian transfer at cosines eta for every voxel, interpolated linearly between table nodes.

    Args:
        library (MultigroupLibrary): the cross-section library.
        rho (Tensor): [M, ...spatial] densities in g/cm^3.
        eta (Tensor): [...spatial] cosines of the scattering angle, broadcastable to ``rho[0]``.

    Returns:
        Tensor: [G_in, G_out, ...spatial] transfer in cm^-1 sr^-1.
    """
    i0, w1 = angular_table_position(eta, library.mu_s_eta.shape[-1], library.eta_map)
    table = library.mu_s_eta.to(eta.device, eta.dtype)  # [M, Gi, Go, N]
    flat0 = i0.reshape(-1)
    v0 = table[..., flat0]  # [M, Gi, Go, P]
    v1 = table[..., (flat0 + 1)]
    w1f = w1.reshape(-1)
    v = v0 + (v1 - v0) * w1f  # [M, Gi, Go, P]
    v = v.reshape(*table.shape[:3], *eta.shape)  # [M, Gi, Go, ...spatial]
    return torch.einsum("mab...,m...->ab...", v, rho)


def transfer_eta_at_pairs(library: MultigroupLibrary, pairs: MaterialPairs, eta: Tensor) -> Tensor:
    """Return ``transfer_eta_at`` for the pair representation, gathering the two table rows of each voxel by index.

    Args:
        library (MultigroupLibrary): the cross-section library.
        pairs (MaterialPairs): material index and density [2, ...spatial].
        eta (Tensor): [...spatial] or [k, ...spatial] cosines; the pairs broadcast over the leading direction axis.

    Returns:
        Tensor: [G_in, G_out, ...] transfer in cm^-1 sr^-1, with the trailing shape of ``eta``.
    """
    i0, w1 = angular_table_position(eta, library.mu_s_eta.shape[-1], library.eta_map)
    table = library.mu_s_eta.to(eta.device, eta.dtype)  # [M, Gi, Go, N]
    flat0 = i0.reshape(-1)
    w1f = w1.reshape(-1)
    out = None
    for p in range(2):
        material_index = pairs.index[p].to(eta.device).long().expand(eta.shape).reshape(-1)
        slot_density = pairs.density[p].to(eta.device, eta.dtype).expand(eta.shape).reshape(-1)
        v0 = table[material_index, :, :, flat0]  # [P, Gi, Go]
        v1 = table[material_index, :, :, flat0 + 1]
        v = (v0 + (v1 - v0) * w1f[:, None, None]) * slot_density[:, None, None]
        out = v if out is None else out + v
    return out.permute(1, 2, 0).reshape(table.shape[1], table.shape[2], *eta.shape)


# ---- synthetic library for tests ---------------------------------------------------------------------

def synthetic_angular_map(num_knots: int = ANGULAR_MAP_KNOTS) -> Tensor:
    """Return a fixed adaptive node placement for tests, dense near eta = 1 as for a forward peak.

    The map is ``u(q) = 1 - 0.6 log(1 + q / 0.02) / log(51) - 0.4 q^2`` at ``q_k = k / (num_knots - 1)``.

    Args:
        num_knots (int, optional): number of knots K. Defaults to ANGULAR_MAP_KNOTS.

    Returns:
        Tensor: [K] float64 map values, strictly decreasing from 1 to 0.
    """
    q = torch.arange(num_knots, dtype=torch.float64) / (num_knots - 1)
    u = 1.0 - 0.6 * torch.log1p(q / 0.02) / math.log(51.0) - 0.4 * q * q
    u[0], u[-1] = 1.0, 0.0
    return u


def synthetic_library(num_groups: int = 4, legendre_order: int = 5, num_eta: int = 401, num_fine_per_group: int = 4,
                      fine_spread: float = 0.0, num_materials: int = 2,
                      angular_table_grid: str = "uniform") -> MultigroupLibrary:
    """Return a self-consistent fake library (isotropic plus mildly forward-peaked transfer) for unit tests.

    The eta table and the Legendre moments come from the same analytic kernel ``a (1 + b eta) / (4 pi)``, linear in
    eta, so the moment machinery holds exactly and a uniform table interpolates the kernel exactly. The fine-energy
    table has n equal-weight bins per group whose weighted mean of ``fine_mu_t`` is exactly ``mu_t``; not physical.

    Args:
        num_groups (int, optional): number of groups G. Defaults to 4.
        legendre_order (int, optional): Legendre order L. Defaults to 5.
        num_eta (int, optional): number of table nodes. Defaults to 401.
        num_fine_per_group (int, optional): fine bins per group n. Defaults to 4.
        fine_spread (float, optional): relative spread of ``fine_mu_t`` within a group, from +spread at the lowest
            energy bin to -spread at the highest (0: constant; below 1 it models beam hardening). Defaults to 0.0.
        num_materials (int, optional): M; 2 gives ("water", "bone"), 7 the default reference materials, any other M > 2
            blends of the two named "m0", "m1" and so on. Defaults to 2.
        angular_table_grid (str, optional): "uniform" (nodes equally spaced in eta) or "adaptive" (the nodes of
            ``synthetic_angular_map``). Defaults to "uniform".

    Returns:
        MultigroupLibrary: the library, float64 on CPU.
    """
    G, L = num_groups, legendre_order
    edges = torch.linspace(125.0, 10.0, G + 1, dtype=torch.float64)
    energy = 0.5 * (edges[:-1] + edges[1:])
    fraction = torch.ones(G, dtype=torch.float64) / G
    if angular_table_grid not in ("uniform", "adaptive"):
        raise ValueError(f"angular_table_grid must be 'uniform' or 'adaptive', got {angular_table_grid!r}")
    eta_map = synthetic_angular_map() if angular_table_grid == "adaptive" else None
    eta = angular_table_nodes(num_eta, eta_map) if eta_map is not None else torch.linspace(-1.0, 1.0, num_eta,
                                                                                           dtype=torch.float64)
    # kernel per (material, g_in, g_out): a * (1 + b eta)/(4 pi) with b depending on the pair (|b|<1 keeps it positive)
    mu_s_eta = torch.zeros(2, G, G, num_eta, dtype=torch.float64)
    mu_s_l = torch.zeros(2, G, G, L + 1, dtype=torch.float64)
    for m in range(2):
        for gi in range(G):
            for go in range(gi, G):
                a = (0.15 if m == 0 else 0.12) * (0.6 ** (go - gi))
                b = 0.5 if go == gi else 0.2
                mu_s_eta[m, gi, go] = a * (1.0 + b * eta) / (4.0 * torch.pi)
                mu_s_l[m, gi, go, 0] = a  # 2 pi int (a(1+b eta)/4pi) deta = a
                if L >= 1:
                    mu_s_l[m, gi, go, 1] = a * b / 3.0  # 2 pi int a(1+b eta)/4pi * eta deta = a b /3
    mu_pe = torch.tensor([[0.05 * (1.5 ** g) for g in range(G)], [0.2 * (1.5 ** g) for g in range(G)]],
                         dtype=torch.float64)
    mu_t = mu_pe + mu_s_l[..., 0].sum(dim=2)
    n = int(num_fine_per_group)
    if n < 1:
        raise ValueError("num_fine_per_group must be >= 1")
    t = (torch.arange(n, dtype=torch.float64) + 0.5) / n  # bin midpoints in (0, 1), ascending energy
    x = 1.0 - 2.0 * torch.arange(n, dtype=torch.float64) / (n - 1) if n > 1 else torch.zeros(1, dtype=torch.float64)
    fine_energy = torch.cat([edges[g + 1] + t * (edges[g] - edges[g + 1]) for g in range(G)])
    fine_group = torch.arange(G).repeat_interleave(n)
    fine_weight = fraction[fine_group] / n
    fine_mu_t = mu_t[:, fine_group] * (1.0 + float(fine_spread) * x.repeat(G))[None, :]
    M = int(num_materials)
    if M == 2:
        return MultigroupLibrary(
            group_edges=edges, group_energy=energy, spectrum_fraction=fraction,
            mu_t=mu_t, mu_pe=mu_pe, mu_s_l=mu_s_l, eta_grid=eta, mu_s_eta=mu_s_eta,
            fine_energy=fine_energy, fine_mu_t=fine_mu_t, fine_weight=fine_weight, fine_group=fine_group,
            materials=("water", "bone"), nominal_density=torch.tensor([1.0, 1.85], dtype=torch.float64),
            eta_map=eta_map,
        )
    if M < 2:
        raise ValueError("num_materials must be >= 2")
    if M == len(MATERIALS):
        blend_coefficients = {"air": (0.90, 0.0), "lung": (0.95, 0.02), "adipose": (0.97, 0.0), "water": (1.0, 0.0),
                              "muscle": (1.0, 0.02), "soft_tissue": (1.0, 0.04), "bone": (1.0, 1.0)}
        row_blends = [blend_coefficients[n] for n in MATERIALS]
        names, nominal = MATERIALS, torch.tensor(NOMINAL_DENSITIES, dtype=torch.float64)
    else:
        row_blends = [(1.0, 0.0), (1.0, 1.0)] + [(1.0 + 0.1 * m, (m - 1) / (M - 1)) for m in range(2, M)]
        names, nominal = tuple(f"m{m}" for m in range(M)), torch.arange(1, M + 1, dtype=torch.float64)

    def blend(a: Tensor) -> Tensor:
        return torch.stack([s * ((1.0 - t) * a[0] + t * a[1]) if t not in (0.0, 1.0) or s != 1.0
                            else a[int(t)].clone() for s, t in row_blends])

    return MultigroupLibrary(
        group_edges=edges, group_energy=energy, spectrum_fraction=fraction,
        mu_t=blend(mu_t), mu_pe=blend(mu_pe), mu_s_l=blend(mu_s_l), eta_grid=eta, mu_s_eta=blend(mu_s_eta),
        fine_energy=fine_energy, fine_mu_t=blend(fine_mu_t), fine_weight=fine_weight, fine_group=fine_group,
        materials=names, nominal_density=nominal, eta_map=eta_map,
    )
