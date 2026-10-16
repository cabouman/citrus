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
The basis is the seven-material table `MATERIAL_TABLE` (name, nominal density g/cm^3), in order of INCREASING
attenuation at the nominal density at every energy of the library (10-125 keV; asserted when a library is built,
`cross_section_library.check_material_order`):

    air 0.0012, lung 0.26, adipose 0.95, water 1.0, muscle 1.05, soft tissue 1.06, cortical bone 1.85

(compositions in `cross_section_library.MATERIAL_FORMULAS`). Density maps are rho [M, rows, cols, slices] with M = 7
in this order; every coefficient is linear in rho. The density model (`citrus.utilities.attenuation_to_densities`)
gives every voxel at most two nonzero materials, the two whose nominal attenuations bracket the voxel's attenuation;
the first-collision source works on that compact pair representation (`MaterialPairs`, `material_pairs`). The
library, the solver and the kernels are general in M; `synthetic_library` builds test libraries of any M.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor

# (name, nominal density g/cm^3) in library order = increasing attenuation at nominal density. Muscle is BELOW soft
# tissue at every energy (ICRU-44 muscle has the lower density and nearly the same composition: attenuation ratio
# 0.986-0.991 over 10-125 keV).
MATERIAL_TABLE: tuple[tuple[str, float], ...] = (
    ("air", 0.0012), ("lung", 0.26), ("adipose", 0.95), ("water", 1.0), ("muscle", 1.05), ("soft_tissue", 1.06),
    ("bone", 1.85),
)
MATERIALS: tuple[str, ...] = tuple(n for n, _ in MATERIAL_TABLE)
NOMINAL_DENSITIES: tuple[float, ...] = tuple(float(d) for _, d in MATERIAL_TABLE)
WATER, BONE = MATERIALS.index("water"), MATERIALS.index("bone")  # 3, 6
ELECTRON_REST_KEV = 510.998950


@dataclass
class MultigroupLibrary:
    """Per-unit-density (mass) multigroup coefficients for the basis materials.

    Shapes (M = number of basis materials, 7 for the package's `MATERIAL_TABLE`; G = groups, L = Legendre order):
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
      materials          tuple of M names (library order = increasing attenuation at nominal density)
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
            d = (NOMINAL_DENSITIES if self.materials == MATERIALS else (1.0, 1.85) if self.materials == ("water", "bone")
                 else tuple(float(m + 1) for m in range(M)))
            self.nominal_density = torch.tensor(d, dtype=torch.float64, device=self.mu_t.device)
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
        kw = {}
        if device is not None:
            kw["device"] = device
        if dtype is not None:
            kw["dtype"] = dtype
        return MultigroupLibrary(
            group_edges=self.group_edges.to(**kw),
            group_energy=self.group_energy.to(**kw),
            spectrum_fraction=self.spectrum_fraction.to(**kw),
            mu_t=self.mu_t.to(**kw),
            mu_pe=self.mu_pe.to(**kw),
            mu_s_l=self.mu_s_l.to(**kw),
            eta_grid=self.eta_grid.to(**kw),
            mu_s_eta=self.mu_s_eta.to(**kw),
            fine_energy=self.fine_energy.to(**kw),
            fine_mu_t=self.fine_mu_t.to(**kw),
            fine_weight=self.fine_weight.to(**kw),
            fine_group=self.fine_group.to(device=device) if device is not None else self.fine_group,
            materials=self.materials,
            nominal_density=self.nominal_density.to(**kw),
        )

    def save(self, path: str) -> None:
        torch.save({k: getattr(self, k) for k in self.__dataclass_fields__}, path)

    @staticmethod
    def load(path: str) -> "MultigroupLibrary":
        """Load a library written by `save`. Raises KeyError if the file lacks a field (a file written by an older
        version: no fine-energy table, no material names or no nominal densities); `cross_section_library.build_library`
        then rebuilds its cache entry."""
        d = torch.load(path)
        missing = [k for k in MultigroupLibrary.__dataclass_fields__ if k not in d]
        if missing:
            raise KeyError(f"library file {path!r} lacks fields {missing} (written by an older version)")
        return MultigroupLibrary(**d)


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
    nz = rho != 0
    if M > 2 and rho.numel():
        worst = int(nz.sum(0).max())
        if worst > 2:
            raise ValueError(f"material_pairs: a voxel has {worst} nonzero materials (at most 2 allowed)")
    ar = torch.arange(M, device=rho.device).reshape((M,) + (1,) * len(spatial)).expand(nz.shape)
    none = torch.full_like(ar, M)
    key = torch.where(nz, ar, none)
    first = key.amin(0)
    second = torch.where(key == first[None], none, key).amin(0)
    idx = torch.stack([first, second])  # M marks an unused slot
    used = idx < M
    idx = torch.where(used, idx, torch.zeros_like(idx))
    dens = torch.gather(rho, 0, idx) * used.to(rho.dtype)
    return MaterialPairs(index=idx.to(torch.int32).contiguous(), density=dens.contiguous(), num_materials=M)


# ---- per-voxel coefficients from density maps ----------------------------------------------------

def mu_t_volume(lib: MultigroupLibrary, rho: Tensor) -> Tensor:
    """Total attenuation per group on a grid.

    rho: [M, rows, cols, slices] g/cm^3. Returns [G, rows, cols, slices] in 1/cm.
    """
    mu = lib.mu_t.to(rho.device, rho.dtype)  # [M, G]
    return torch.einsum("mg,mijk->gijk", mu, rho)


# at most this many (fine energy, ray) pairs are evaluated at once by `group_transmission`
_TRANSMISSION_BLOCK = 1 << 22


def fine_group_weights(lib: MultigroupLibrary, device=None, dtype=None) -> Tensor:
    """[G, E] matrix W[g, e] = fine_weight[e] where fine_group[e] == g, else 0 (fine-to-group summation)."""
    fg = lib.fine_group.to(device=device)
    w = lib.fine_weight.to(device=device, dtype=dtype)
    onehot = fg[None, :] == torch.arange(lib.num_groups, device=fg.device)[:, None]
    return torch.where(onehot, w[None, :], torch.zeros((), device=w.device, dtype=w.dtype))


def group_transmission(lib: MultigroupLibrary, path: Tensor, block: int = _TRANSMISSION_BLOCK,
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
    dev, dt = path.device, path.dtype
    M = path.shape[0]
    spatial = tuple(path.shape[1:])
    P = path.reshape(M, -1)
    N = P.shape[1]
    mu = lib.fine_mu_t.to(device=dev, dtype=dt)  # [M, E]
    if materials is not None:
        idx = [int(i) for i in materials]
        if any(i < 0 or i >= mu.shape[0] for i in idx):
            raise ValueError(f"material indices {idx} outside the library's {mu.shape[0]} materials")
        mu = mu[torch.as_tensor(idx, device=dev, dtype=torch.long)]
    if mu.shape[0] != M:
        raise ValueError(f"path has {M} materials, the library {mu.shape[0]}")
    Wg = fine_group_weights(lib, dev, dt)  # [G, E]
    E = mu.shape[1]
    out = torch.zeros(Wg.shape[0], N, device=dev, dtype=dt)
    nb = max(1, min(N, int(block)))
    eb = max(1, int(block) // nb)
    for n0 in range(0, N, nb):
        Pn = P[:, n0:n0 + nb]
        o = out[:, n0:n0 + nb]
        for e0 in range(0, E, eb):
            m = mu[:, e0:e0 + eb]  # [M, e]
            tau = m[0][:, None] * Pn[0][None, :]
            for k in range(1, M):
                tau.addcmul_(m[k][:, None], Pn[k][None, :])
            o.addmm_(Wg[:, e0:e0 + eb], tau.neg_().exp_())
    return out.reshape(Wg.shape[0], *spatial)


def transfer_moments_volume(lib: MultigroupLibrary, rho: Tensor) -> Tensor:
    """Legendre transfer moments per voxel: [G_in, G_out, L+1, rows, cols, slices] in 1/cm."""
    t = lib.mu_s_l.to(rho.device, rho.dtype)  # [M, Gi, Go, L+1]
    return torch.einsum("mabl,mijk->ablijk", t, rho)


def transfer_eta_at(lib: MultigroupLibrary, rho: Tensor, eta: Tensor) -> Tensor:
    """Per-steradian transfer evaluated at cos-angles eta for every voxel.

    rho: [M, ...spatial]; eta: [...spatial] broadcastable to rho[0]. Returns [G_in, G_out, ...spatial] (1/cm/sr).
    Linear interpolation on lib.eta_grid.
    """
    grid = lib.eta_grid.to(eta.device, eta.dtype)
    n = grid.numel()
    pos = (eta.clamp(-1.0, 1.0) + 1.0) / 2.0 * (n - 1)
    i0 = pos.floor().long().clamp(0, n - 2)
    w1 = (pos - i0.to(pos.dtype)).clamp(0.0, 1.0)
    table = lib.mu_s_eta.to(eta.device, eta.dtype)  # [M, Gi, Go, N]
    flat0 = i0.reshape(-1)
    v0 = table[..., flat0]  # [M, Gi, Go, P]
    v1 = table[..., (flat0 + 1)]
    w1f = w1.reshape(-1)
    v = v0 + (v1 - v0) * w1f  # [M, Gi, Go, P]
    v = v.reshape(*table.shape[:3], *eta.shape)  # [M, Gi, Go, ...spatial]
    return torch.einsum("mab...,m...->ab...", v, rho)


def transfer_eta_at_pairs(lib: MultigroupLibrary, pairs: MaterialPairs, eta: Tensor) -> Tensor:
    """`transfer_eta_at` for the pair representation: the two table rows of each voxel gathered by material index.

    pairs: index / density [2, ...spatial]; eta: [...spatial] or [k, ...spatial] (the pairs broadcast over the leading
    direction axis). Returns [G_in, G_out, *eta.shape] (1/cm/sr). Peak memory that of a two-material basis for any M.
    """
    grid = lib.eta_grid.to(eta.device, eta.dtype)
    n = grid.numel()
    pos = (eta.clamp(-1.0, 1.0) + 1.0) / 2.0 * (n - 1)
    i0 = pos.floor().long().clamp(0, n - 2)
    w1 = (pos - i0.to(pos.dtype)).clamp(0.0, 1.0)
    table = lib.mu_s_eta.to(eta.device, eta.dtype)  # [M, Gi, Go, N]
    flat0 = i0.reshape(-1)
    w1f = w1.reshape(-1)
    out = None
    for p in range(2):
        mi = pairs.index[p].to(eta.device).long().expand(eta.shape).reshape(-1)
        rp = pairs.density[p].to(eta.device, eta.dtype).expand(eta.shape).reshape(-1)
        v0 = table[mi, :, :, flat0]  # [P, Gi, Go]
        v1 = table[mi, :, :, flat0 + 1]
        v = (v0 + (v1 - v0) * w1f[:, None, None]) * rp[:, None, None]
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
    frac = torch.ones(G, dtype=torch.float64) / G
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
    mu_pe = torch.tensor([[0.05 * (1.5 ** g) for g in range(G)], [0.2 * (1.5 ** g) for g in range(G)]], dtype=torch.float64)
    mu_t = mu_pe + mu_s_l[..., 0].sum(dim=2)
    n = int(num_fine_per_group)
    if n < 1:
        raise ValueError("num_fine_per_group must be >= 1")
    t = (torch.arange(n, dtype=torch.float64) + 0.5) / n  # bin midpoints in (0, 1), ascending energy
    x = 1.0 - 2.0 * torch.arange(n, dtype=torch.float64) / (n - 1) if n > 1 else torch.zeros(1, dtype=torch.float64)
    fine_energy = torch.cat([edges[g + 1] + t * (edges[g] - edges[g + 1]) for g in range(G)])
    fine_group = torch.arange(G).repeat_interleave(n)
    fine_weight = frac[fine_group] / n
    fine_mu_t = mu_t[:, fine_group] * (1.0 + float(fine_spread) * x.repeat(G))[None, :]
    M = int(num_materials)
    if M == 2:
        return MultigroupLibrary(
            group_edges=edges, group_energy=energy, spectrum_fraction=frac,
            mu_t=mu_t, mu_pe=mu_pe, mu_s_l=mu_s_l, eta_grid=eta, mu_s_eta=mu_s_eta,
            fine_energy=fine_energy, fine_mu_t=fine_mu_t, fine_weight=fine_weight, fine_group=fine_group,
            materials=("water", "bone"), nominal_density=torch.tensor([1.0, 1.85], dtype=torch.float64),
        )
    if M < 2:
        raise ValueError("num_materials must be >= 2")
    if M == len(MATERIALS):
        st = {"air": (0.90, 0.0), "lung": (0.95, 0.02), "adipose": (0.97, 0.0), "water": (1.0, 0.0),
              "muscle": (1.0, 0.02), "soft_tissue": (1.0, 0.04), "bone": (1.0, 1.0)}
        rows_st = [st[n] for n in MATERIALS]
        names, nominal = MATERIALS, torch.tensor(NOMINAL_DENSITIES, dtype=torch.float64)
    else:
        rows_st = [(1.0, 0.0), (1.0, 1.0)] + [(1.0 + 0.1 * m, (m - 1) / (M - 1)) for m in range(2, M)]
        names, nominal = tuple(f"m{m}" for m in range(M)), torch.arange(1, M + 1, dtype=torch.float64)

    def blend(a: Tensor) -> Tensor:
        return torch.stack([s * ((1.0 - t) * a[0] + t * a[1]) if t not in (0.0, 1.0) or s != 1.0
                            else a[int(t)].clone() for s, t in rows_st])

    return MultigroupLibrary(
        group_edges=edges, group_energy=energy, spectrum_fraction=frac,
        mu_t=blend(mu_t), mu_pe=blend(mu_pe), mu_s_l=blend(mu_s_l), eta_grid=eta, mu_s_eta=blend(mu_s_eta),
        fine_energy=fine_energy, fine_mu_t=blend(fine_mu_t), fine_weight=fine_weight, fine_group=fine_group,
        materials=names, nominal_density=nominal,
    )
