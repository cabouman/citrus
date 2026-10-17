"""Multigroup cross-section library: container, per-voxel coefficients and a synthetic test library.

The container is filled by `cross_section_library.build_library` from XrayPhysics (EPDL97) data;
`synthetic_library` gives a fake but self-consistent library for tests that must not depend on XrayPhysics.

Besides the group constants the library carries a FINE-ENERGY attenuation table (fine_energy,
fine_mu_t, fine_weight, fine_group). Because mu_t(E; r) = sum_m rho_m(r) m_m(E) is linear in the
density maps, the optical depth of any ray at any energy is sum_m m_m(E) P_m with P_m the line
integral of the density map rho_m (g/cm^2): one ray trace per material present gives the exact
polychromatic transmission. `group_transmission` evaluates

    T_g = sum_{e in g} w_e exp(-sum_m m_m(E_e) P_m)

the group-integrated uncollided transmission, exact up to the fine energy quadrature, used by
the uncollided fluence and the primary image (docs/source/design_notes.rst, "Cross sections").

Materials
---------
`MATERIAL_CATALOGUE` maps a material name to (nominal density g/cm^3, composition) for every material the library
builder knows. The composition is an XrayPhysics material string (`cross_section_library.parse_formula`): a chemical
formula ('H2O', a stoichiometric compound) or 'a*F1 + b*F2 + ...' with a, b the mass fractions of the elements or
compounds F1, F2 (normalized to sum 1). The NOMINAL density is the density of the material as it occurs in the body or
in the scanner (the pure-material points of the density model). Nominal attenuation mu = rho^nom mu/rho at the
spectrum-mean energy 63.03 keV of the default 125 kVp spectrum is listed for orientation:

    name             density  mu (1/cm)  composition source
    air              0.0012   0.00022    XrayPhysics 'air' (dry air: N2, O2, Ar, CO2)
    lung             0.26     0.0520     XrayPhysics 'lung' (density of the WLAB lung; XrayPhysics lists 0.25)
    adipose          0.95     0.1846     XrayPhysics 'adipose_tissue' (ICRP adipose tissue; XrayPhysics lists 0.92)
    water            1.0      0.2013     H2O
    muscle           1.05     0.2102     ICRU Report 44 (1989), skeletal muscle
    soft_tissue      1.06     0.2122     ICRU Report 44 (1989), soft tissue
    bone             1.85     0.5366     XrayPhysics 'bone_cortical' (ICRP cortical bone)
    trabecular_bone  1.18     0.2697     Woodard and White (1986) Br J Radiol 59:1209, skeleton spongiosa (ICRU
                                         Report 46, 1992)
    brain            1.04     0.2092     ICRU Report 44 (1989), brain (grey / white matter)
    blood            1.06     0.2130     ICRU Report 44 (1989), whole blood
    cartilage        1.10     0.2234     Woodard and White (1986), cartilage (ICRU Report 46, 1992)
    skin             1.09     0.2160     ICRU Report 44 (1989), skin
    pmma             1.18     0.2230     XrayPhysics 'PMMA' (C5H8O2, polymethyl methacrylate)
    polyethylene     0.94     0.1826     XrayPhysics 'Polyethylene' (C2H4)
    aluminium        2.699    0.7034     Al (NIST density)
    titanium         4.54     3.0928     Ti (NIST density)
    carbon_fibre     1.56     0.2756     carbon-fibre epoxy composite (treatment couch): 60 % fibre volume of carbon
                                         fibre (C, 1.8 g/cm^3) in bisphenol-A epoxy (C21H24O4, 1.2 g/cm^3), i.e.
                                         0.692 C + 0.308 C21H24O4 by mass at 1.56 g/cm^3 (a typical laminate, not a
                                         reference composition)

(XrayPhysics' material list, xrayPhysics().get_material_library(), supplies air, lung, adipose, cortical bone, PMMA
and polyethylene; it has no brain, blood, skin, cartilage or spongiosa, which come from the ICRU-44 and Woodard-White
tables as cited, the ICRU-44 ones as tabulated by Hubbell and Seltzer, NISTIR 5632.)

REFERENCE materials. The library is built for a chosen subset of the catalogue, the reference materials of the density
model (`config.SolverConfig.materials`, at least two names; default the seven `MATERIALS`):

    air 0.0012, lung 0.26, adipose 0.95, water 1.0, muscle 1.05, soft tissue 1.06, cortical bone 1.85

The builder sorts the reference materials by nominal attenuation (`cross_section_library.sort_material_table`) and
asserts that the order is the same at every energy of the library (10-125 keV,
`cross_section_library.check_material_order`: no two reference materials cross), so `MultigroupLibrary.materials` is
always in order of INCREASING attenuation at the nominal density. Density maps are rho [M, rows, cols, slices] with
M = the number of reference materials, in the library's order (look a row up by name:
`material_index(library, "water")`); every coefficient is linear in rho. The density model
(`citrus.utilities.attenuation_to_densities`) gives every voxel at most two nonzero materials, the two reference
materials whose nominal attenuations bracket the voxel's attenuation; the first-collision source works on that compact
pair representation (`MaterialPairs`, `material_pairs`) wherever its evaluation points have at most two nonzero
materials (`material_pairs_if_compact`; block means of such a field can mix more, and are then evaluated densely over
all M materials). The library, the solver and the kernels are general in M; `synthetic_library` builds test libraries
of any M.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Union

import torch
from torch import Tensor

# ---- material catalogue (module docstring) ----------------------------------------------------------------------
# XrayPhysics library strings (xrayPhysics().get_material_library(), verbatim; the air terms are mass fractions of the
# compounds, as XrayPhysics' parser reads them)
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
_BONE_CORTICAL_XRAYPHYSICS = (
    "0.047234*H + 0.144330*C + 0.041990*N + 0.446096*O + 0.002200*Mg + 0.104970*P + "
    "0.003150*S + 0.209930*Ca + 0.000100*Zn"
)
_PMMA_XRAYPHYSICS = "C5O2H8"
_POLYETHYLENE_XRAYPHYSICS = "C2H4"
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

MATERIAL_CATALOGUE: Mapping[str, tuple[float, str]] = MappingProxyType({
    # name: (nominal density g/cm^3, XrayPhysics composition); the first seven are the default reference materials
    # `MATERIALS`
    "air": (0.0012, _AIR_XRAYPHYSICS),
    "lung": (0.26, _LUNG_XRAYPHYSICS),
    "adipose": (0.95, _ADIPOSE_XRAYPHYSICS),
    "water": (1.0, "H2O"),
    "muscle": (1.05, _MUSCLE_ICRU44),
    "soft_tissue": (1.06, _SOFT_TISSUE_ICRU44),
    "bone": (1.85, _BONE_CORTICAL_XRAYPHYSICS),
    "trabecular_bone": (1.18, _SPONGIOSA_WOODARD_WHITE),
    "brain": (1.04, _BRAIN_ICRU44),
    "blood": (1.06, _BLOOD_ICRU44),
    "cartilage": (1.10, _CARTILAGE_WOODARD_WHITE),
    "skin": (1.09, _SKIN_ICRU44),
    "pmma": (1.18, _PMMA_XRAYPHYSICS),
    "polyethylene": (0.94, _POLYETHYLENE_XRAYPHYSICS),
    "aluminium": (2.699, "Al"),
    "titanium": (4.54, "Ti"),
    "carbon_fibre": (1.56, _CARBON_FIBRE_EPOXY),
})

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
    """The reference material names as a tuple after checking them: a sequence (not a single string) of at least
    `minimum`
    distinct names of `MATERIAL_CATALOGUE`. Raises ValueError naming the problem (and the catalogue for an unknown
    name)."""
    if isinstance(names, str) or not isinstance(names, Sequence):
        raise ValueError(f"materials must be a sequence of material names, e.g. ('air', 'water', 'bone'), got "
                         f"{names!r}")
    names = tuple(names)
    unknown = [n for n in names if not isinstance(n, str) or n not in MATERIAL_CATALOGUE]
    if unknown:
        raise ValueError(f"unknown material name(s) {unknown}; the catalogue has {tuple(MATERIAL_CATALOGUE)}")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate material names in {names}")
    if len(names) < minimum:
        raise ValueError(f"the density model interpolates between at least {minimum} reference materials, got {names}")
    return names


def material_table(names: Sequence[str] = MATERIALS) -> tuple[tuple[str, float, str], ...]:
    """(name, nominal density g/cm^3, composition) of the catalogue materials `names`, in the given order (the library
    builder sorts them by attenuation, `cross_section_library.sort_material_table`). Raises ValueError for an unknown
    or repeated name (`check_material_names`; one name is allowed here)."""
    return tuple((n, float(MATERIAL_CATALOGUE[n][0]), MATERIAL_CATALOGUE[n][1])
                 for n in check_material_names(names, minimum=1))


def material_index(materials: Union["MultigroupLibrary", Sequence[str]], name: str) -> int:
    """Row of material `name` in a library (`library.materials`) or in a sequence of names. Raises ValueError naming
    the available materials when it is absent (e.g. a WLAB phantom for a set of reference materials without water or
    bone)."""
    names = tuple(materials.materials if isinstance(materials, MultigroupLibrary) else materials)
    if name not in names:
        raise ValueError(f"material {name!r} is not among the materials {names}")
    return names.index(name)


@dataclass
class MultigroupLibrary:
    """Per-unit-density (mass) multigroup coefficients for the basis materials.

    Shapes (M = number of reference materials, 7 for the default `MATERIALS`; G = groups, L = Legendre order):
      group_edges        [G+1]        keV, descending
      group_energy       [G]          representative (spectrum-weighted mean) energy per group, keV
      spectrum_fraction  [G]          f_g, sums to 1 (source photons per steradian per group)
      mu_t               [M, G]       total mass attenuation, cm^2/g
      mu_pe              [M, G]       photoelectric mass attenuation, cm^2/g
      mu_s_l             [M, G, G, L+1]  Legendre moments of the transfer, [m, g_in, g_out, l], cm^2/g,
                                      convention mu_s(Omega'->Omega) = sum_l (2l+1)/(4pi) mu_s_l P_l(eta)
      eta_grid           [N_eta]      cos(scattering angle) grid, ascending from -1 to 1
      mu_s_eta           [M, G, G, N_eta]  per-steradian transfer d mu / d Omega (eta), cm^2/g/sr,
                                      Compton + Rayleigh, [m, g_in, g_out, eta]
    Fine-energy attenuation table (E = number of fine energies: the build's energy quadrature, bins with
    zero spectrum weight dropped):
      fine_energy        [E]          keV, bin midpoints, ordered by group then ascending energy
      fine_mu_t          [M, E]       total mass attenuation at fine_energy, cm^2/g
      fine_weight        [E]          source photons per steradian in each fine bin (the spectrum integrated
                                      over the bin), sum_e fine_weight = 1; the sum over the bins of group g
                                      is spectrum_fraction[g]
      fine_group         [E] int64    group index of each fine energy
    Materials:
      materials          tuple of M names (library order = increasing attenuation at nominal density; catalogue
                         names, `MATERIAL_CATALOGUE`, for a library from `cross_section_library.build_library`; look
                         a row up with `material_index(library, name)`)
      nominal_density    [M] g/cm^3: the density of each material as it occurs in the body (the pure-material points
                         of the density model, `citrus.utilities.attenuation_to_densities`)
    Left empty / None, `materials` and `nominal_density` are filled in from the number of materials: `MATERIALS` /
    `NOMINAL_DENSITIES` for M = 7, ("water", "bone") / (1.0, 1.85) for M = 2, "m0", "m1", ... / 1, 2, ... otherwise.
    All floating tensors float64 on CPU when built; use `.to(device, dtype)` (fine_group stays int64).
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

    def __post_init__(self) -> None:
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

    # ---- derived ----------------------------------------------------------------------------
    @property
    def num_materials(self) -> int:
        return int(self.mu_t.shape[0])

    @property
    def num_groups(self) -> int:
        return int(self.mu_t.shape[1])

    @property
    def legendre_order(self) -> int:
        return int(self.mu_s_l.shape[-1]) - 1

    @property
    def num_fine_energies(self) -> int:
        return int(self.fine_energy.shape[0])

    @property
    def mu_s(self) -> Tensor:
        """Group-to-group scattering coefficient [M, G_in, G_out] = l=0 moment."""
        return self.mu_s_l[..., 0]

    def to(self, device=None, dtype=None) -> "MultigroupLibrary":
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
        )

    def save(self, path: str) -> None:
        torch.save({k: getattr(self, k) for k in self.__dataclass_fields__}, path)

    @staticmethod
    def load(path: str) -> "MultigroupLibrary":
        """Load a library written by `save`. Raises KeyError if the file lacks a field (a file written by an older
        version: no fine-energy table, no material names or no nominal densities); `cross_section_library.build_library`
        then rebuilds its cache entry."""
        fields = torch.load(path)
        missing = [k for k in MultigroupLibrary.__dataclass_fields__ if k not in fields]
        if missing:
            raise KeyError(f"library file {path!r} lacks fields {missing} (written by an older version)")
        return MultigroupLibrary(**fields)


# ---- compact per-voxel material pairs ----------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class MaterialPairs:
    """At most two materials per voxel (the density model's output, `citrus.utilities.attenuation_to_densities`).

    index   [2, ...spatial] int32: library material index of each slot (slot 0 < slot 1 where both are used; an
            unused slot holds index 0 with density 0)
    density [2, ...spatial]      : density of each slot (g/cm^3)
    num_materials                : M of the dense maps it came from
    The dense maps are rho[m] = sum_p density[p] [index[p] == m] (`dense`). The first-collision source gathers the two
    table rows of each voxel by index, so its cost is that of a two-material basis for any M.
    """

    index: Tensor
    density: Tensor
    num_materials: int

    def dense(self) -> Tensor:
        """rho [M, ...spatial] (same device / dtype as density)."""
        out = torch.zeros((self.num_materials,) + tuple(self.density.shape[1:]), device=self.density.device,
                          dtype=self.density.dtype)
        out.scatter_add_(0, self.index.long(), self.density)
        return out

    def to(self, device=None, dtype=None) -> "MaterialPairs":
        return MaterialPairs(self.index.to(device=device), self.density.to(device=device, dtype=dtype),
                             self.num_materials)


def material_pairs(rho: Tensor) -> MaterialPairs:
    """Compact pair representation of density maps rho [M, ...] with AT MOST TWO nonzero materials per voxel (raises
    ValueError otherwise; one host synchronization when M > 2). The nonzero materials keep ascending index order."""
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
    """Largest number of nonzero materials at any point of rho [M, ...] (one host synchronization; 0 for an empty or
    all-zero volume)."""
    if rho.numel() == 0:
        return 0
    return int((rho != 0).sum(0).max())


def material_pairs_if_compact(rho: Tensor) -> Optional[MaterialPairs]:
    """`material_pairs(rho)` when rho [M, ...] has M > 2 materials and at most two nonzero ones at every point, else
    None (M <= 2, or some point mixes three or more materials: the first-collision source then evaluates all M
    materials densely). One host synchronization when M > 2. Block means of a density field that obeys the
    two-material convention (`citrus.utilities.attenuation_to_densities`) can mix more materials, so the solver
    decides per quadrature grid (`scatter_solver.MaterialFields.pairs_quadrature`)."""
    if int(rho.shape[0]) <= 2 or max_materials_per_point(rho) > 2:
        return None
    return material_pairs(rho)


# ---- per-voxel coefficients from density maps ----------------------------------------------------

def mu_t_volume(library: MultigroupLibrary, rho: Tensor) -> Tensor:
    """Total attenuation per group on a grid.

    rho: [M, rows, cols, slices] g/cm^3. Returns [G, rows, cols, slices] in 1/cm.
    """
    mu = library.mu_t.to(rho.device, rho.dtype)  # [M, G]
    return torch.einsum("mg,mijk->gijk", mu, rho)


# at most this many (fine energy, ray) pairs are evaluated at once by `group_transmission`
_TRANSMISSION_BLOCK = 1 << 22


def fine_group_weights(library: MultigroupLibrary, device=None, dtype=None) -> Tensor:
    """[G, E] matrix W[g, e] = fine_weight[e] where fine_group[e] == g, else 0 (fine-to-group summation)."""
    fine_group = library.fine_group.to(device=device)
    weights = library.fine_weight.to(device=device, dtype=dtype)
    in_group = fine_group[None, :] == torch.arange(library.num_groups, device=fine_group.device)[:, None]
    return torch.where(in_group, weights[None, :], torch.zeros((), device=weights.device, dtype=weights.dtype))


def group_transmission(library: MultigroupLibrary, path: Tensor, block: int = _TRANSMISSION_BLOCK,
                       materials: Optional[Sequence[int]] = None) -> Tensor:
    """Group-integrated transmission from material line integrals, exact in energy up to the fine quadrature.

        T_g = sum_{e: fine_group[e] = g} fine_weight[e] exp(-sum_m fine_mu_t[m, e] path[m])

    path: [M, ...] line integrals of the density maps (g/cm^2), e.g. projectors.upstream_cumulative(rho) or
    projectors.primary_project(rho). Returns [G, ...] in the dtype/device of `path` (source photons per steradian
    per group surviving the path; sum_g T_g is the spectrum-weighted polychromatic transmission, 1 for path = 0).
    Evaluated in blocks of at most `block` (fine energy, ray) pairs: per block one optical-depth tensor [e, n]
    (M fused multiply-adds), an in-place exp and one [G, e] x [e, n] matmul accumulated into the [G, n] result
    (the float32 matmul assumes TF32 is off, `torch.backends.cuda.matmul.allow_tf32 = False`, the PyTorch default).
    materials: library indices of the rows of `path` when it holds only a subset of the materials (e.g. the ones
    present in a volume, `uncollided.present_materials`: an absent material has zero line integrals and adds exactly 0
    to the optical depth); None = all, in library order.
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
    """Legendre transfer moments per voxel: [G_in, G_out, L+1, rows, cols, slices] in 1/cm."""
    transfer = library.mu_s_l.to(rho.device, rho.dtype)  # [M, Gi, Go, L+1]
    return torch.einsum("mabl,mijk->ablijk", transfer, rho)


def transfer_eta_at(library: MultigroupLibrary, rho: Tensor, eta: Tensor) -> Tensor:
    """Per-steradian transfer evaluated at cos-angles eta for every voxel.

    rho: [M, ...spatial]; eta: [...spatial] broadcastable to rho[0]. Returns [G_in, G_out, ...spatial] (1/cm/sr).
    Linear interpolation on library.eta_grid.
    """
    eta_grid = library.eta_grid.to(eta.device, eta.dtype)
    n = eta_grid.numel()
    pos = (eta.clamp(-1.0, 1.0) + 1.0) / 2.0 * (n - 1)
    i0 = pos.floor().long().clamp(0, n - 2)
    w1 = (pos - i0.to(pos.dtype)).clamp(0.0, 1.0)
    table = library.mu_s_eta.to(eta.device, eta.dtype)  # [M, Gi, Go, N]
    flat0 = i0.reshape(-1)
    v0 = table[..., flat0]  # [M, Gi, Go, P]
    v1 = table[..., (flat0 + 1)]
    w1f = w1.reshape(-1)
    v = v0 + (v1 - v0) * w1f  # [M, Gi, Go, P]
    v = v.reshape(*table.shape[:3], *eta.shape)  # [M, Gi, Go, ...spatial]
    return torch.einsum("mab...,m...->ab...", v, rho)


def transfer_eta_at_pairs(library: MultigroupLibrary, pairs: MaterialPairs, eta: Tensor) -> Tensor:
    """`transfer_eta_at` for the pair representation: the two table rows of each voxel gathered by material index.

    pairs: index / density [2, ...spatial]; eta: [...spatial] or [k, ...spatial] (the pairs broadcast over the leading
    direction axis). Returns [G_in, G_out, *eta.shape] (1/cm/sr). Peak memory that of a two-material basis for any M.
    """
    eta_grid = library.eta_grid.to(eta.device, eta.dtype)
    n = eta_grid.numel()
    pos = (eta.clamp(-1.0, 1.0) + 1.0) / 2.0 * (n - 1)
    i0 = pos.floor().long().clamp(0, n - 2)
    w1 = (pos - i0.to(pos.dtype)).clamp(0.0, 1.0)
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

def synthetic_library(num_groups: int = 4, legendre_order: int = 5, num_eta: int = 401, num_fine_per_group: int = 4,
                      fine_spread: float = 0.0, num_materials: int = 2) -> MultigroupLibrary:
    """Self-consistent fake library (isotropic + mildly forward-peaked transfer) for unit tests.

    The eta table and the Legendre moments are generated from the same analytic kernel, so
    tests of the moment/ordinate machinery hold exactly. Not physical.

    Fine-energy table: each group is cut into `num_fine_per_group` = n equal bins of equal weight
    spectrum_fraction[g] / n, and fine_mu_t[m, e] = mu_t[m, g] (1 + fine_spread x_e) with x_e running linearly
    from +1 (lowest-energy bin of the group) to -1 (highest), zero mean, so the weighted group mean of fine_mu_t is
    exactly mu_t[m, g]. fine_spread = 0 (default): attenuation constant within each group, the fine-energy
    transmission equals the group-wise f_g exp(-mu_t,g L) exactly. 0 < fine_spread < 1 models within-group beam
    hardening.

    Materials: two base materials A ("water"-like) and B ("bone"-like, more attenuating per gram).
      num_materials = 2 (default): ("water", "bone") = (A, B), nominal densities (1.0, 1.85).
      num_materials = 7: the package's table (`MATERIALS`, `NOMINAL_DENSITIES`) with water = A and bone = B (so a
        density volume with only water and bone gives exactly the two-material results) and the other rows
        s ((1 - t) A + t B) with (s, t) = air (0.90, 0), lung (0.95, 0.02), adipose (0.97, 0), muscle (1.0, 0.02),
        soft tissue (1.0, 0.04): distinct, self-consistent and in increasing nominal attenuation at every energy.
      any other M > 2: material 0 = A, 1 = B, material m >= 2 = ((1 - t) A + t B) (1 + 0.1 m), t = (m - 1) / (M - 1),
        named "m0", "m1", ...; nominal densities 1, 2, ..., M (no attenuation order).
    """
    G, L = num_groups, legendre_order
    edges = torch.linspace(125.0, 10.0, G + 1, dtype=torch.float64)
    energy = 0.5 * (edges[:-1] + edges[1:])
    fraction = torch.ones(G, dtype=torch.float64) / G
    eta = torch.linspace(-1.0, 1.0, num_eta, dtype=torch.float64)
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
        materials=names, nominal_density=nominal,
    )
