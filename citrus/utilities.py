"""Common utilities of CITRUS: the WLAB phantom and its geometry, the attenuation <-> density model, a GPU memory
cap and small plotting helpers.

WLAB phantom
------------
A 20 x 30 x 40 cm^3 water prism (1.0 g/cm^3) with full-length (40 cm along z) inserts of lung (6 x 6 cm^2,
0.26 g/cm^3), air (2 x 2 cm^2, 1.205e-3 g/cm^3) and bone (2 x 2 cm^2, 1.85 g/cm^3). Lung and air are modeled as water
of the stated density, bone as the bone basis material. Orientation (source on +y at angle 0): 30 cm along x (cols),
20 cm along y (rows, the source-detector axis at angle 0), 40 cm along z (slices), centered at the isocenter. Default
insert centers: lung at (x, y) = (-5, 0), air at (+3, 0), bone at (+7, 0) cm (keyword arguments of `wlab_phantom`).
Outside the prism the density is 0 (vacuum).

Attenuation <-> density model
-----------------------------
MBIR reconstructs ONE monoenergetic attenuation map X (1/cm) at an effective energy E_eff; the scatter solver needs
the two basis densities rho = [rho_water, rho_bone] (g/cm^3). The standard two-material split:

    forward  X = rho_w m_w(E_eff) + rho_b m_b(E_eff)                       (exact, linear)
    inverse  X <= mu_w:  rho_w = X / m_w,  rho_b = 0          water of reduced density (lung, air, vacuum)
             X >  mu_w:  rho_w = 1,        rho_b = (X - m_w) / m_b    bone ADDED to unit-density water
             with mu_w = 1 g/cm^3 x m_w(E_eff); negatives (X < 0) clamp to rho = 0.

m_m(E) are the library's fine-energy total mass attenuation coefficients (`lib.fine_mu_t`, cm^2/g), interpolated
linearly in (log E, log m). The inverse is a right inverse: densities_to_attenuation(lib,
attenuation_to_densities(lib, X)) == max(X, 0), but a pure-bone voxel (rho_w = 0, rho_b = 1.85, the WLAB bone insert)
comes back as rho_w = 1, rho_b = 1.85 - m_w/m_b (~1.15 at 63 keV): same attenuation at E_eff, different composition,
hence a slightly different scatter (a ~3 % effect on the SPR behind the bone only).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .scatter_solver.cross_sections import BONE, WATER, MultigroupLibrary
from .scatter_solver.geometry import ConeGeometry, Grid

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
    "densities_to_attenuation",
    "attenuation_to_densities",
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
    docstring). Returns rho [2, rows, cols, slices] in g/cm^3, channel 0 = water, 1 = bone.
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
    rho = torch.zeros(2, *grid.shape, device=device, dtype=torch.float64)
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


def _coeffs(lib: MultigroupLibrary, energy_kev: Optional[float], like: Tensor) -> Tensor:
    e = effective_energy(lib) if energy_kev is None else float(energy_kev)
    return mass_attenuation_at(lib, e).to(device=like.device, dtype=like.dtype)


def densities_to_attenuation(lib: MultigroupLibrary, rho: Tensor, energy_kev: Optional[float] = None) -> Tensor:
    """X = rho[water] m_w(E) + rho[bone] m_b(E) (1/cm). rho [2, ...] g/cm^3 -> X [...], same device/dtype.
    energy_kev None: `effective_energy(lib)`."""
    if rho.shape[0] != 2:
        raise ValueError(f"rho must be [2, ...], got {tuple(rho.shape)}")
    m = _coeffs(lib, energy_kev, rho)
    return rho[WATER] * m[WATER] + rho[BONE] * m[BONE]


def attenuation_to_densities(lib: MultigroupLibrary, X: Tensor, energy_kev: Optional[float] = None) -> Tensor:
    """Two-material split of X [...] (1/cm) at E (module docstring): returns rho [2, ...] g/cm^3, same device/dtype.
    X <= m_w: water of density X / m_w (clamped at 0); X > m_w: unit water plus bone (X - m_w) / m_b."""
    m = _coeffs(lib, energy_kev, X)
    mw, mb = m[WATER], m[BONE]
    Xc = X.clamp_min(0)
    low = Xc <= mw
    rho = torch.empty((2,) + tuple(X.shape), device=X.device, dtype=X.dtype)
    rho[WATER] = torch.where(low, Xc / mw, torch.ones_like(Xc))
    rho[BONE] = torch.where(low, torch.zeros_like(Xc), (Xc - mw) / mb)
    return rho


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
