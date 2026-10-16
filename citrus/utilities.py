"""Common utilities of CITRUS: the WLAB phantom and its geometry, the attenuation <-> density model, synthetic
cone-beam data with scatter, a GPU memory cap and small plotting helpers.

WLAB phantom
------------
A 20 x 30 x 40 cm^3 water prism (1.0 g/cm^3) with full-length (40 cm along z) inserts of lung (6 x 6 cm^2,
0.26 g/cm^3), air (2 x 2 cm^2, 1.205e-3 g/cm^3) and bone (2 x 2 cm^2, 1.85 g/cm^3). The phantom is given in the
package's seven-material basis (rho [7, ...], `cross_sections.MATERIALS`) but uses only two of its rows: lung and air
are modeled as water of the stated density (the water row), bone as cortical bone (the bone row). Orientation (source on +y at angle 0): 30 cm along x (cols),
20 cm along y (rows, the source-detector axis at angle 0), 40 cm along z (slices), centered at the isocenter. Default
insert centers: lung at (x, y) = (-5, 0), air at (+3, 0), bone at (+7, 0) cm (keyword arguments of `wlab_phantom`).
Outside the prism the density is 0 (vacuum).

Attenuation <-> density model
-----------------------------
MBIR reconstructs ONE monoenergetic attenuation map X (1/cm) at an effective energy E_eff (the spectrum mean, 63.03 keV
for the default 125 kVp spectrum); the scatter solver needs the densities rho [M, ...] (g/cm^3) of its library's basis
materials (M = 7: air, lung, adipose, water, muscle, soft tissue, cortical bone; `cross_sections.MATERIAL_TABLE`).

    forward  X = sum_m rho_m m_m(E_eff)                                     (exact, linear)

The inverse is the interpolating seven-material model. Every material m has a NOMINAL density rho_m^nom
(`lib.nominal_density`) and a nominal attenuation mu_m = rho_m^nom m_m(E_eff); the materials are taken in the fixed
library order, in which mu_m increases at every library energy (`material_order` checks it). For each voxel:

    mu_a <= X < mu_b (consecutive materials a, b):  f = (X - mu_a) / (mu_b - mu_a),
                                                    rho_a = (1 - f) rho_a^nom,   rho_b = f rho_b^nom
    X < mu_first (air):   the first material alone, density X / m_first
    X >= mu_last (bone):  the last material alone, density X / m_last (bone denser than 1.85 g/cm^3)
    X < 0 clamps to 0.

At E_eff = 63.03 keV the nominal attenuations are 0.00022 (air), 0.0520 (lung), 0.1846 (adipose), 0.2013 (water),
0.2102 (muscle), 0.2122 (soft tissue), 0.5366 (bone) /cm. The model is exact in attenuation at E_eff
(densities_to_attenuation(lib, attenuation_to_densities(lib, X)) == max(X, 0)), gives the pure material m at
X = mu_m (a WLAB water voxel is pure water, a bone voxel pure bone at 1.85 g/cm^3) and at most two nonzero materials
per voxel, the pair representation of the solver's first-collision source (`cross_sections.material_pairs`).
m_m(E) are the library's fine-energy total mass attenuation coefficients (`lib.fine_mu_t`, cm^2/g), interpolated
linearly in (log E, log m). (Up to citrus 0.0.3 the model was a two-material split, water of reduced density below
mu_water and unit water PLUS bone above it: a pure-bone voxel came back as water 1.0 + bone 1.156 g/cm^3, the same
attenuation at E_eff but another composition, which made the CITRUS loop overshoot the WLAB bone insert by 6 %.)

Synthetic cone-beam data
------------------------
`synthesize_cone_beam_data` (any density volume) and `synthesize_wlab_dataset` (the WLAB phantom):

    X_true    = densities_to_attenuation(lib, rho_true, E_eff)           [R, C, S] 1/cm, E_eff = spectrum mean
    angles    = start + (end - start) i / V, i = 0..V-1 (`view_angles`; synthesize_wlab_dataset's default is the
                short scan end = start + pi + fan angle of the panel, `fan_angle`)
    l         = A X_true  (MBIRTORCH ConeBeamModel forward projection)   [V, Rd, Cd] line integrals
    P         = I0 exp(-l)                                               primary counts per pixel
    SPR       = s_sol / p_sol per view (ScatterSolver.solve_view on rho_true, fully converged source iteration):
                ratio of the solver's scatter and primary PLANAR fluence images (photon counting: no detector or grid
                response unless the solver has one)
    S         = SPR P,   Y_mean = P + S,   Y = Poisson(Y_mean) (noise=True) or Y_mean

Units. The solver's images are photons/cm^2 per source photon/sr; its primary carries the inverse-square and
obliquity factors of the point source and the polychromatic transmission. MBIRTORCH's model is the flat,
monochromatic line integral l = A X at E_eff with I0 the flat-field count, which absorbs the inverse-square / obliquity
factor phi_air (the solver's unattenuated primary, saved). The data tie the two models through the dimensionless
per-pixel SPR, S = SPR x P: the scatter-to-primary ratio of the data equals the solver's physical SPR exactly, and the
CITRUS correction Y / (1 + SPR(X)) is independent of I0 and of the mono / poly primary mismatch.

Inverse crime (intentional): the primary uses MBIRTORCH's own projector at E_eff and the truth lives on the
reconstruction grid, so the only model errors the reconstruction sees are the scatter and the noise; the scatter
truth comes from the same solver as the correction (use a finer angular discretization for the truth solver to
expose the solver's own discretization error). Reproducibility: the Poisson draws use torch.Generator(seed), but the
rates vary between runs at the float32 ulp level (atomic accumulation in MBIRTORCH's and the solver's CUDA kernels),
which flips a few draws in 10^4; save and reuse the data when bitwise identity matters.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, replace
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from .scatter_solver.config import SolverConfig
from .scatter_solver.cross_sections import BONE, MATERIALS, WATER, MultigroupLibrary
from .scatter_solver.detector import ideal_counting_response, no_grid
from .scatter_solver.geometry import ConeGeometry, Grid
from .scatter_solver.scatter_solver import ScatterSolver, build_solver_library

__all__ = [
    "WLAB_SIZE_XYZ",
    "RHO_WATER",
    "RHO_LUNG",
    "RHO_AIR",
    "RHO_BONE",
    "wlab_phantom",
    "wlab_grid",
    "wlab_geometry",
    "full_field_wlab_geometry",
    "wlab_setup",
    "effective_energy",
    "mass_attenuation_at",
    "nominal_densities",
    "nominal_attenuation",
    "material_order",
    "densities_to_attenuation",
    "attenuation_to_densities",
    "default_supervoxel_factor",
    "wlab_solver",
    "fan_angle",
    "view_angles",
    "synthesize_cone_beam_data",
    "synthesize_wlab_dataset",
    "save_dataset",
    "load_dataset",
    "set_memory_cap",
    "central_slice",
    "central_row",
    "plot_slices",
    "plot_profiles",
    "plot_history",
]

# =====================================================================================================
# WLAB phantom and geometry
# =====================================================================================================

WLAB_SIZE_XYZ = (30.0, 20.0, 40.0)  # cm, (x, y, z)
LUNG_SIZE_XY = (6.0, 6.0)
AIR_SIZE_XY = (2.0, 2.0)
BONE_SIZE_XY = (2.0, 2.0)
RHO_WATER = 1.0
RHO_LUNG = 0.26
RHO_AIR = 1.205e-3
RHO_BONE = 1.85


def _in_interval(c: Tensor, center: float, width: float) -> Tensor:
    """Half-open membership [center - width/2, center + width/2) of coordinates c."""
    return (c >= center - width / 2) & (c < center + width / 2)


def wlab_phantom(grid: Grid, *, lung_center: tuple[float, float] = (-5.0, 0.0),
                 air_center: tuple[float, float] = (3.0, 0.0), bone_center: tuple[float, float] = (7.0, 0.0),
                 device=None, dtype=torch.float32) -> Tensor:
    """Rasterize the WLAB phantom by voxel-center membership (half-open intervals).

    grid must cover the 30 x 20 x 40 cm prism (centered at the isocenter). Insert centers are (x, y) in cm (module
    docstring). Returns rho [7, rows, cols, slices] in g/cm^3 in the package's material basis
    (`cross_sections.MATERIALS`): the water row (index `WATER` = 3) holds the water prism and the lung and air inserts
    (water of reduced density), the bone row (index `BONE` = 6) the bone insert; the other rows are zero.
    """
    X, Y, Z = WLAB_SIZE_XYZ
    ex, ey, ez = grid.extent
    lo = grid.lower_corner()
    tol = 1e-9
    if lo[0] > -X / 2 + tol or lo[0] + ex < X / 2 - tol or lo[1] > -Y / 2 + tol or lo[1] + ey < Y / 2 - tol \
            or lo[2] > -Z / 2 + tol or lo[2] + ez < Z / 2 - tol:
        raise ValueError(f"grid box {lo} + {grid.extent} does not cover the WLAB prism {WLAB_SIZE_XYZ}")
    inserts = [(lung_center, LUNG_SIZE_XY), (air_center, AIR_SIZE_XY), (bone_center, BONE_SIZE_XY)]
    for i, (ci, si) in enumerate(inserts):
        if abs(ci[0]) + si[0] / 2 > X / 2 + tol or abs(ci[1]) + si[1] / 2 > Y / 2 + tol:
            raise ValueError(f"insert {i} at {ci} leaves the water prism")
        for cj, sj in inserts[i + 1:]:
            if abs(ci[0] - cj[0]) < (si[0] + sj[0]) / 2 - tol and abs(ci[1] - cj[1]) < (si[1] + sj[1]) / 2 - tol:
                raise ValueError(f"inserts at {ci} and {cj} overlap")

    x, y, z = grid.axis_coordinates(device, torch.float64)
    xx = x[None, :, None]
    yy = y[:, None, None]
    zz = z[None, None, :]
    prism = _in_interval(xx, 0.0, X) & _in_interval(yy, 0.0, Y) & _in_interval(zz, 0.0, Z)

    def column(center: tuple[float, float], size: tuple[float, float]) -> Tensor:
        return _in_interval(xx, center[0], size[0]) & _in_interval(yy, center[1], size[1]) & prism

    lung = column(lung_center, LUNG_SIZE_XY)
    air = column(air_center, AIR_SIZE_XY)
    bone = column(bone_center, BONE_SIZE_XY)
    rho = torch.zeros(len(MATERIALS), *grid.shape, device=device, dtype=torch.float64)
    water = rho[WATER]
    water[prism] = RHO_WATER
    water[lung] = RHO_LUNG
    water[air] = RHO_AIR
    water[bone] = 0.0
    rho[BONE][bone] = RHO_BONE
    return rho.to(dtype)


def wlab_grid(delta_voxel: float) -> Grid:
    """Grid that exactly tiles the 30 x 20 x 40 cm WLAB box (cols x rows x slices) at pitch delta_voxel (cm)."""
    X, Y, Z = WLAB_SIZE_XYZ
    n = []
    for L in (Y, X, Z):
        k = round(L / delta_voxel)
        if k < 1 or abs(k * delta_voxel - L) > 1e-9 * max(1.0, L):
            raise ValueError(f"delta_voxel={delta_voxel} does not divide the WLAB extent {L} cm")
        n.append(int(k))
    return Grid(shape=(n[0], n[1], n[2]), delta_voxel=float(delta_voxel))


def wlab_geometry() -> ConeGeometry:
    """Full-fan kV CBCT geometry of the WLAB setup: 100 cm source-axis, 150 cm source-detector distance, centered
    40 x 30 cm^2 panel of 100 x 75 pixels of 4 mm (the prism's lateral and z ends fall outside the field of view)."""
    return ConeGeometry(source_iso_dist=100.0, source_detector_dist=150.0, num_det_rows=75, num_det_channels=100,
                        delta_det_row=0.4, delta_det_channel=0.4)


def full_field_wlab_geometry() -> ConeGeometry:
    """`wlab_geometry` with a 150 x 184 panel (60 x 73.6 cm): the whole 30 x 20 x 40 cm prism projects inside the panel
    at every view angle (needed for reconstruction). Channels: the prism's transaxial half-diagonal 18.03 cm subtends
    |u| <= 150 tan(asin(18.03 / 100)) = 27.5 cm (>= 138 channels). Rows: the z-ends (|z| = 20) of the prism edge
    nearest to the source are magnified by up to 150 / (100 - 18.03) = 1.83: |v| <= 36.6 cm, i.e. >= 183 rows."""
    return replace(wlab_geometry(), num_det_channels=150, num_det_rows=184)


def wlab_setup(delta_voxel: float = 1.0, device=None, dtype=torch.float32) -> tuple[Grid, ConeGeometry, Tensor]:
    """(grid, geom, rho) of the WLAB scatter study: `wlab_grid(delta_voxel)`, `wlab_geometry()`, `wlab_phantom`."""
    grid = wlab_grid(delta_voxel)
    return grid, wlab_geometry(), wlab_phantom(grid, device=device, dtype=dtype)


# =====================================================================================================
# attenuation <-> density
# =====================================================================================================

def effective_energy(lib: MultigroupLibrary) -> float:
    """Spectrum-mean photon energy sum_e fine_weight[e] fine_energy[e] / sum_e fine_weight[e] (keV); 63.03 keV for the
    default 125 kVp spectrum."""
    w = lib.fine_weight.to(torch.float64)
    return float((w * lib.fine_energy.to(torch.float64)).sum() / w.sum())


def mass_attenuation_at(lib: MultigroupLibrary, energy_kev: float) -> Tensor:
    """Total mass attenuation [M] (cm^2/g) of the basis materials at `energy_kev`: log-log linear interpolation of
    lib.fine_mu_t on the sorted fine energies (float64, CPU). Raises outside the table's energy range."""
    E = lib.fine_energy.detach().to("cpu", torch.float64)
    mu = lib.fine_mu_t.detach().to("cpu", torch.float64)
    order = torch.argsort(E)
    E, mu = E[order], mu[:, order]
    e = float(energy_kev)
    if not (float(E[0]) <= e <= float(E[-1])):
        raise ValueError(f"energy {e} keV outside the library's fine-energy range [{float(E[0])}, {float(E[-1])}]")
    i = int(torch.searchsorted(E, torch.tensor([e], dtype=torch.float64), right=True)[0]) - 1
    i = min(max(i, 0), E.numel() - 2)
    le0, le1 = torch.log(E[i]), torch.log(E[i + 1])
    t = (torch.log(torch.tensor(e, dtype=torch.float64)) - le0) / (le1 - le0)
    return torch.exp(torch.log(mu[:, i]) + t * (torch.log(mu[:, i + 1]) - torch.log(mu[:, i])))


def nominal_densities(lib: MultigroupLibrary) -> Tensor:
    """[M] nominal densities (g/cm^3, float64 CPU) of the library's materials (`lib.nominal_density`)."""
    return lib.nominal_density.detach().to("cpu", torch.float64)


def nominal_attenuation(lib: MultigroupLibrary, energy_kev: Optional[float] = None) -> Tensor:
    """[M] attenuation (1/cm, float64 CPU) of each material at its nominal density at `energy_kev` (None: E_eff);
    0.00022, 0.0520, 0.1846, 0.2013, 0.2102, 0.2122, 0.5366 /cm for the seven materials at 63.03 keV."""
    e = effective_energy(lib) if energy_kev is None else float(energy_kev)
    return nominal_densities(lib) * mass_attenuation_at(lib, e)


def material_order(lib: MultigroupLibrary) -> list[int]:
    """The fixed bracketing order of the density model: material indices sorted by nominal attenuation at E_eff (the
    library order, [0, 1, ..., M-1], for a library built by `cross_section_library.build_library`). Raises ValueError
    if that order is not the same at every fine energy of the library (two materials cross; the library builder
    already refuses such a table)."""
    order = torch.argsort(nominal_attenuation(lib)).tolist()
    a = (nominal_densities(lib)[:, None] * lib.fine_mu_t.detach().to("cpu", torch.float64))[order]
    if a.shape[0] > 1 and not bool((torch.diff(a, dim=0) > 0).all()):
        raise ValueError("the materials' nominal attenuations cross inside the library's energy range")
    return order


def _coeffs(lib: MultigroupLibrary, energy_kev: Optional[float], like: Tensor) -> Tensor:
    e = effective_energy(lib) if energy_kev is None else float(energy_kev)
    return mass_attenuation_at(lib, e).to(device=like.device, dtype=like.dtype)


def densities_to_attenuation(lib: MultigroupLibrary, rho: Tensor, energy_kev: Optional[float] = None) -> Tensor:
    """X = sum_m rho[m] m_m(E) (1/cm). rho [M, ...] g/cm^3 (M = the library's materials, 7) -> X [...], same
    device/dtype. energy_kev None: `effective_energy(lib)`."""
    M = lib.num_materials
    if rho.shape[0] != M:
        raise ValueError(f"rho must be [{M}, ...] for the library's materials {tuple(lib.materials)}, got "
                         f"{tuple(rho.shape)}")
    m = _coeffs(lib, energy_kev, rho)
    X = rho[0] * m[0]
    for k in range(1, M):
        X = X + rho[k] * m[k]
    return X


def attenuation_to_densities(lib: MultigroupLibrary, X: Tensor, energy_kev: Optional[float] = None) -> Tensor:
    """The interpolating density model (module docstring): rho [M, ...] (g/cm^3, same device/dtype) of X [...] (1/cm)
    at the energy (None: E_eff). The two materials whose nominal attenuations bracket X get (1 - f) rho_a^nom and
    f rho_b^nom, f = (X - mu_a) / (mu_b - mu_a); below the first material the first alone at X / m_first, above the
    last the last alone at X / m_last; X < 0 gives 0. Exact in attenuation, at most two nonzero materials per voxel."""
    M = lib.num_materials
    order = material_order(lib)
    nd = X.dim()
    m = _coeffs(lib, energy_kev, X).reshape((M,) + (1,) * nd)
    perm = torch.as_tensor(order, device=X.device, dtype=torch.long)
    m = m[perm].expand((M,) + tuple(X.shape))  # material slots in bracketing order
    nom = nominal_densities(lib).to(device=X.device, dtype=X.dtype)[perm].reshape((M,) + (1,) * nd)
    mu = nom * m  # nominal attenuation per slot [M, ...]
    Xc = X.clamp_min(0)
    j = (mu <= Xc[None]).sum(0)  # number of nominal attenuations <= X: bracket [j - 1, j]
    a = (j - 1).clamp(0, M - 1)
    b = j.clamp(max=M - 1)
    inside = (j >= 1) & (j <= M - 1)
    mu_a = torch.gather(mu, 0, a[None])[0]
    mu_b = torch.gather(mu, 0, b[None])[0]
    m_a = torch.gather(m, 0, a[None])[0]
    nom_a = nom.expand_as(mu).gather(0, a[None])[0]
    nom_b = nom.expand_as(mu).gather(0, b[None])[0]
    span = torch.where(inside, mu_b - mu_a, torch.ones_like(mu_a))
    f = torch.where(inside, (Xc - mu_a) / span, torch.zeros_like(Xc))
    # outside the bracket range: one material (slot a = first or last) of density X / m
    rho_a = torch.where(inside, (1.0 - f) * nom_a, Xc / m_a)
    rho_b = torch.where(inside, f * nom_b, torch.zeros_like(Xc))
    rho = torch.zeros((M,) + tuple(X.shape), device=X.device, dtype=X.dtype)
    rho.scatter_add_(0, perm[a][None], rho_a[None])
    rho.scatter_add_(0, perm[b][None], rho_b[None])
    return rho


# =====================================================================================================
# synthetic cone-beam data
# =====================================================================================================

def default_supervoxel_factor(delta_voxel: float) -> int:
    """Supervoxel factor giving ~1 cm transport cells: round(1 / delta_voxel), at least 1 (0.5 cm -> 2, 1 cm -> 1,
    2 cm -> 1)."""
    return max(1, int(round(1.0 / float(delta_voxel))))


def wlab_solver(delta_voxel: float, *, config: Optional[SolverConfig] = None, lib: Optional[MultigroupLibrary] = None,
                geom: Optional[ConeGeometry] = None, device=None,
                cache_dir: Optional[str] = "default") -> ScatterSolver:
    """ScatterSolver for the WLAB data: the given config (default SolverConfig() with the supervoxel factor of
    `default_supervoxel_factor`), the real library (build_solver_library, cached in `cache_dir`; "default" is
    $CITRUS_CACHE_DIR or ~/.cache/citrus; needs XrayPhysics unless cached), the full-field WLAB geometry (default), an
    ideal counting detector and no grid, float32."""
    cfg = config if config is not None else replace(SolverConfig(),
                                                    supervoxel_factor=default_supervoxel_factor(delta_voxel))
    lib = lib if lib is not None else build_solver_library(cfg, cache_dir=cache_dir)
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return ScatterSolver(lib, cfg, geom if geom is not None else full_field_wlab_geometry(),
                         detector=ideal_counting_response(), grid_response=no_grid(), device=dev,
                         dtype=torch.float32)


def fan_angle(geometry: ConeGeometry) -> float:
    """In-plane fan angle of the panel (rad): the full angle subtended at the source by the detector's channel extent,
    2 atan(0.5 num_det_channels delta_det_channel / SDD) (the channel offset is ignored). 22.6 degrees for the
    full-field WLAB panel (150 channels of 0.4 cm at SDD 150 cm), 15.2 degrees for the 100-channel panel."""
    half = 0.5 * float(geometry.num_det_channels) * float(geometry.delta_det_channel)
    return 2.0 * math.atan(half / float(geometry.source_detector_dist))


def view_angles(num_views: int, geometry: Optional[ConeGeometry] = None, start_view_angle: float = 0.0,
                end_view_angle: Optional[float] = None) -> np.ndarray:
    """View angles (rad, float64 [num_views], MBIRTORCH's angle convention) evenly spaced over
    [start_view_angle, end_view_angle), endpoint excluded: start + (end - start) i / num_views. end_view_angle None:
    the short scan end = start + pi + fan_angle(geometry) (half a rotation plus the in-plane fan angle of the panel, the
    smallest range in which every in-plane ray is measured at least once). A full rotation is end = start + 2 pi."""
    n = int(num_views)
    if n < 1:
        raise ValueError("num_views must be >= 1")
    start = float(start_view_angle)
    if end_view_angle is None:
        if geometry is None:
            raise ValueError("the default (short-scan) end angle needs the geometry")
        end = start + math.pi + fan_angle(geometry)
    else:
        end = float(end_view_angle)
    if not end > start:
        raise ValueError(f"end_view_angle {end} must be larger than start_view_angle {start}")
    return start + (end - start) * np.arange(n, dtype=np.float64) / n


def synthesize_cone_beam_data(rho: Tensor, grid: Grid, geometry: ConeGeometry, angles, solver: ScatterSolver,
                              I0: float = 1.0e5, noise: bool = True, seed: int = 0, *,
                              energy_kev: Optional[float] = None, device=None,
                              progress: Optional[Callable[[int, float], None]] = None) -> dict:
    """Cone-beam counts with scatter for any density volume (module docstring, "Synthetic cone-beam data").

    rho [M, R, C, S] densities (g/cm^3) of the solver library's M = 7 basis materials on `grid` (the reconstruction grid; shape divisible by the
    solver's supervoxel factor); geometry: the scan (used by MBIRTORCH and must be the solver's geometry); angles [V]
    (rad); solver: the ScatterSolver that defines the scatter truth (one fully converged solve_view per view);
    I0: flat-field counts per pixel; noise: Poisson(Y_mean) with torch.Generator(seed), else Y = Y_mean;
    energy_kev: E_eff (None: the spectrum mean of the solver's library). progress(view_index, seconds) is called after
    every solved view.

    Returns a dict of numpy arrays (float32 unless noted): X_true [R, C, S] (1/cm), rho_true [M, R, C, S], angles [V]
    (float64), line_integrals, P, S, Y, Y_mean, SPR [V, Rd, Cd], solver_primary, solver_scatter [V, Rd, Cd] (planar
    fluence summed over groups, photons/cm^2 per source photon/sr), phi_air [Rd, Cd] (solver primary of an empty
    volume), view_seconds [V], si_iterations [V] (int), I0, energy_kev, mass_attenuation [M] (the materials' mass
    attenuation at E_eff, cm^2/g) and meta (dict: geometry, grid, solver config, materials, angular range, versions,
    timings, statistics).
    """
    from .recon.mbir_bridge import forward_project, make_cone_model, mbirtorch_version

    if solver.geom != geometry:
        raise ValueError("the solver's geometry must be the scan geometry")
    t_all = time.perf_counter()
    dev = torch.device(device) if device is not None else solver.device
    lib = solver.lib
    E = effective_energy(lib) if energy_kev is None else float(energy_kev)
    m = mass_attenuation_at(lib, E)
    rho = torch.as_tensor(rho, device=dev, dtype=torch.float32)
    X = densities_to_attenuation(lib, rho, E)
    angles = np.asarray(angles.detach().cpu().numpy() if isinstance(angles, Tensor) else angles,
                        dtype=np.float64).reshape(-1)

    # primary: MBIRTORCH forward projection (the reconstruction's own operator)
    t0 = time.perf_counter()
    model = make_cone_model(grid, geometry, angles, device=dev)
    line = forward_project(model, X).to(dev)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t_project = time.perf_counter() - t0
    P = float(I0) * torch.exp(-line)

    # scatter-to-primary ratio from the solver, one converged solve per view
    V, Rd, Cd = len(angles), geometry.num_det_rows, geometry.num_det_channels
    s_sol = torch.empty(V, Rd, Cd, device=dev, dtype=torch.float32)
    p_sol = torch.empty(V, Rd, Cd, device=dev, dtype=torch.float32)
    view_seconds = np.zeros(V)
    si_iters = np.zeros(V, dtype=np.int64)
    si_conv = np.zeros(V, dtype=bool)
    t0 = time.perf_counter()
    for v in range(V):
        tv = time.perf_counter()
        res = solver.solve_view(rho, grid, float(angles[v]))
        p_sol[v] = res.primary["planar_fluence"].sum(0)
        s_sol[v] = res.scatter_planar_fluence.sum(0)
        si_iters[v] = res.si.num_iterations
        si_conv[v] = bool(res.si.converged)
        del res
        view_seconds[v] = time.perf_counter() - tv
        if progress is not None:
            progress(v, view_seconds[v])
    t_solver = time.perf_counter() - t0
    empty = solver.material_fields(torch.zeros_like(rho), grid)
    phi_air = solver.primary(empty, 0.0)["planar_fluence"].sum(0)
    del empty

    SPR = s_sol / p_sol.clamp_min(torch.finfo(p_sol.dtype).tiny)
    S = SPR * P
    Y_mean = P + S
    if noise:
        g = torch.Generator(device=dev)
        g.manual_seed(int(seed))
        Y = torch.poisson(Y_mean, generator=g)
    else:
        Y = Y_mean.clone()

    def a(t: Tensor) -> np.ndarray:
        return t.detach().to("cpu", torch.float32).numpy()

    spr = a(SPR)
    meta = {
        "description": "synthetic cone-beam data with scatter, citrus.utilities.synthesize_cone_beam_data",
        "num_views": int(V), "I0": float(I0), "noise": bool(noise), "seed": int(seed), "energy_kev": E,
        "materials": list(lib.materials), "nominal_density": [float(d) for d in lib.nominal_density],
        "mass_attenuation": {n: float(v) for n, v in zip(lib.materials, m)},
        "angle_range_rad": [float(angles[0]), float(angles[-1])] if len(angles) else [],
        "geometry": asdict(geometry), "grid": asdict(grid), "solver_config": solver.config.to_dict(),
        "detector": getattr(solver.detector, "name", "detector"),
        "grid_response": getattr(solver.grid_response, "name", "grid"), "solver_dtype": str(solver.dtype),
        "mbirtorch_version": mbirtorch_version(), "torch_version": torch.__version__, "device": str(dev),
        "seconds": {"total": time.perf_counter() - t_all, "mbirtorch_projection": t_project, "solver_total": t_solver,
                    "solver_per_view_mean": float(view_seconds.mean()),
                    "solver_per_view_median": float(np.median(view_seconds)),
                    "solver_per_view_max": float(view_seconds.max())},
        "si_iterations": {"min": int(si_iters.min()), "max": int(si_iters.max()), "all_converged": bool(si_conv.all())},
        "stats": {"SPR_min": float(spr.min()), "SPR_max": float(spr.max()), "SPR_mean": float(spr.mean()),
                  "SPR_p01": float(np.percentile(spr, 1)), "SPR_p99": float(np.percentile(spr, 99)),
                  "line_integral_max": float(line.max()), "P_over_I0_min": float((P / I0).min()),
                  "Y_min": float(Y.min()), "Y_max": float(Y.max())},
    }
    return {
        "X_true": a(X), "rho_true": a(rho), "angles": angles, "line_integrals": a(line), "P": a(P), "S": a(S),
        "Y": a(Y), "Y_mean": a(Y_mean), "SPR": spr, "solver_primary": a(p_sol), "solver_scatter": a(s_sol),
        "phi_air": a(phi_air), "view_seconds": view_seconds, "si_iterations": si_iters,
        "I0": np.asarray(float(I0)), "energy_kev": np.asarray(E), "mass_attenuation": m.numpy(), "meta": meta,
    }


def synthesize_wlab_dataset(num_views: int = 180, delta_voxel: float = 0.5, I0: float = 1.0e5, noise: bool = True,
                            seed: int = 0, out_path: Optional[Union[str, os.PathLike]] = None, *,
                            energy_kev: Optional[float] = None, config: Optional[SolverConfig] = None,
                            lib: Optional[MultigroupLibrary] = None, device=None, start_view_angle: float = 0.0,
                            end_view_angle: Optional[float] = None,
                            progress: Optional[Callable[[int, float], None]] = None,
                            geom: Optional[ConeGeometry] = None) -> dict:
    """`synthesize_cone_beam_data` for the WLAB phantom on `wlab_grid(delta_voxel)` (0.5 cm: 40 x 60 x 80 voxels),
    num_views angles evenly spaced over [start_view_angle, end_view_angle) (`view_angles`; default end: the short scan
    start + pi + fan angle of the panel, 202.6 degrees for the full-field panel; end = start + 2 pi for a full
    rotation), the solver `wlab_solver(delta_voxel, config=config, lib=lib, geom=geom)` (default: SolverConfig() with
    ~1 cm transport cells, the full-field geometry `full_field_wlab_geometry()`, ideal counting detector, no grid).
    meta also records delta_voxel, start_view_angle and end_view_angle. out_path: if given, the data are written with
    `save_dataset` (<out_path>.npz and <out_path>.json)."""
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geom = full_field_wlab_geometry() if geom is None else geom
    grid = wlab_grid(delta_voxel)
    solver = wlab_solver(delta_voxel, config=config, lib=lib, geom=geom, device=dev)
    rho = wlab_phantom(grid, device=dev, dtype=torch.float32)
    angles = view_angles(num_views, geom, start_view_angle, end_view_angle)
    end = float(start_view_angle) + math.pi + fan_angle(geom) if end_view_angle is None else float(end_view_angle)
    out = synthesize_cone_beam_data(rho, grid, geom, angles, solver, I0, noise, seed, energy_kev=energy_kev,
                                    device=dev, progress=progress)
    out["meta"].update(description="synthetic WLAB scatter dataset, citrus.utilities.synthesize_wlab_dataset",
                       delta_voxel=float(delta_voxel), start_view_angle=float(start_view_angle),
                       end_view_angle=end, short_scan=bool(end - float(start_view_angle) < 2.0 * math.pi - 1e-9))
    if out_path is not None:
        save_dataset(out, out_path)
    return out


def save_dataset(data: dict, out_path: Union[str, os.PathLike]) -> list[str]:
    """Write a dataset dict (numpy arrays plus `meta`) to <out_path>.npz (meta as a JSON string `meta_json`) and
    <out_path>.json (meta). Returns the two file names (also stored in data["meta"]["files"])."""
    base = str(out_path)
    for ext in (".npz", ".json"):
        if base.endswith(ext):
            base = base[: -len(ext)]
    os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
    meta = data["meta"]
    arrays = {k: v for k, v in data.items() if k != "meta"}
    np.savez(base + ".npz", meta_json=np.asarray(json.dumps(meta)), **arrays)
    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    meta["files"] = [base + ".npz", base + ".json"]
    return meta["files"]


def load_dataset(path: Union[str, os.PathLike]) -> dict:
    """Read a dataset written by `save_dataset` (<path>.npz): dict of numpy arrays plus `meta`."""
    p = str(path)
    if not p.endswith(".npz"):
        p += ".npz"
    with np.load(p) as z:
        out = {k: z[k] for k in z.files if k != "meta_json"}
        out["meta"] = json.loads(str(z["meta_json"]))
    return out


# =====================================================================================================
# GPU memory
# =====================================================================================================

def set_memory_cap(mib: float = 4400.0, device: int = 0) -> None:
    """Cap the PyTorch CUDA allocator of `device` at `mib` MiB (no-op without CUDA or for mib <= 0)."""
    if torch.cuda.is_available() and mib > 0:
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, mib * 2 ** 20 / total), device)


# =====================================================================================================
# plotting helpers (matplotlib is imported lazily)
# =====================================================================================================

def _np(x) -> np.ndarray:
    return x.detach().float().cpu().numpy() if isinstance(x, Tensor) else np.asarray(x)


def central_slice(X) -> np.ndarray:
    """Central z slice [rows, cols] of a volume [rows, cols, slices] (mean of the two middle slices for an even count)."""
    X = _np(X)
    S = X.shape[2]
    return X[:, :, S // 2] if S % 2 else 0.5 * (X[:, :, S // 2 - 1] + X[:, :, S // 2])


def central_row(sl) -> np.ndarray:
    """Central row [cols] of a slice [rows, cols] (mean of the two middle rows for an even count)."""
    sl = _np(sl)
    R = sl.shape[0]
    return sl[R // 2] if R % 2 else 0.5 * (sl[R // 2 - 1] + sl[R // 2])


def _pyplot():
    import matplotlib.pyplot as plt  # the caller chooses the backend (e.g. matplotlib.use("Agg") in scripts)

    return plt


def plot_slices(images: Sequence[tuple[str, np.ndarray]], grid: Grid, path: Optional[str] = None,
                vmin: float = 0.0, vmax: float = 0.3, title: Optional[str] = None):
    """Side-by-side central z slices: images = [(label, volume [R, C, S] or slice [R, C])]; extent from `grid` (cm).
    Saved to `path` if given. Returns the matplotlib figure."""
    plt = _pyplot()
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 2.9), squeeze=False)
    ex, ey, _ = grid.extent
    lo = grid.lower_corner()
    im = None
    for ax, (label, img) in zip(axes[0], images):
        img = _np(img)
        sl = central_slice(img) if img.ndim == 3 else img
        im = ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax, origin="lower",
                       extent=[lo[0], lo[0] + ex, lo[1], lo[1] + ey])
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("x (cm)", fontsize=8)
    axes[0, 0].set_ylabel("y (cm)", fontsize=8)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, label="attenuation (1/cm)")
    if title:
        fig.suptitle(title, fontsize=10)
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def plot_profiles(profiles: Sequence[tuple[str, np.ndarray]], grid: Grid, path: Optional[str] = None,
                  ylabel: str = "attenuation (1/cm)", title: Optional[str] = None):
    """Central-row profiles (row y = 0 of the central z slice) along x: profiles = [(label, volume [R, C, S])].
    Saved to `path` if given. Returns the matplotlib figure."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    x = grid.axis_coordinates(dtype=torch.float64)[0].numpy()
    styles = ["-", "--", "-.", ":"]
    for i, (label, vol) in enumerate(profiles):
        ax.plot(x, central_row(central_slice(vol)), ls=styles[i % len(styles)], lw=1.6 if i else 1.2, label=label)
    ax.set_xlabel("x (cm), row y = 0, slice z = 0")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def plot_history(curves: Sequence[tuple[str, Sequence[float]]], path: Optional[str] = None,
                 ylabel: str = "relative RMSE", xlabel: str = "outer iteration (0 = FDK)", logy: bool = True,
                 title: Optional[str] = None):
    """Convergence curves: curves = [(label, values per iteration)]. Saved to `path` if given. Returns the figure."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    styles = ["-", "--", "-.", ":"]
    for i, (label, vals) in enumerate(curves):
        vals = np.asarray(vals, dtype=float)
        (ax.semilogy if logy else ax.plot)(np.arange(len(vals)), vals, ls=styles[i % len(styles)], marker="o", ms=3,
                                           label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig
