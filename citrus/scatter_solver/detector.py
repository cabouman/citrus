"""Detector and anti-scatter-grid responses.

Both responses are lookup tables, so that tables from a Monte Carlo simulation of the detector and the grid can be
dropped in:

* `DetectorResponse`: D(E, theta0) in keV deposited per incident photon, a 2-D table over energy and
  the angle theta0 between the photon direction and the panel normal.
* `GridResponse`: G(E, theta1, theta2), the transmitted fraction of a 1-D anti-scatter grid, a 3-D
  table over energy and the two grid angles of `grid_angles` (theta1 in the plane perpendicular to the lamellae,
  measured from the focusing direction; theta2 in the plane along the lamellae, from the panel normal).

The detector signal of photons arriving at a pixel along Omega with planar fluence F is
F G(E, theta1, theta2) D(E, theta0) (keV/cm^2).

The analytic builders here (`csi_response`, `focused_1d_grid`) are simple models: a 1/cos(theta)
path-length scaling of the normal-incidence CsI absorption and a geometric lamella-shadow model of a
focused lead grid.

Attenuation data: coarse log-log tables of total mass attenuation (coherent included) for CsI and Pb,
values read from NIST XCOM (Berger et al., XCOM: Photon Cross Sections Database, NIST Standard
Reference Database 8, https://physics.nist.gov/xcom), rounded to 2 to 3 significant digits.
K-edges are included as a pair of points 1 eV apart (I 33.17 keV, Cs 35.98 keV, Pb 88.00 keV;
Pb L3 13.04 keV). Accuracy ~ few % between nodes: adequate for detector modelling, not for dosimetry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from .geometry import ConeGeometry

__all__ = [
    "DetectorResponse",
    "GridResponse",
    "csi_mass_attenuation",
    "lead_mass_attenuation",
    "csi_response",
    "ideal_counting_response",
    "energy_weighted_response",
    "no_grid",
    "focused_1d_grid",
    "grid_angles",
    "detector_angle",
]

ArrayLike = Union[Tensor, np.ndarray, Sequence[float]]
MassAttenuation = Callable[[np.ndarray], np.ndarray]

RHO_CSI = 4.51  # g/cm^3
RHO_LEAD = 11.35  # g/cm^3
_EDGE_SPACING_KEV = 1e-3  # keV; the two points of a K/L-edge pair are this far apart

# ---- coarse NIST XCOM tables (total attenuation with coherent scattering, cm^2/g) ----------------
# CsI (Cs 51.15 %, I 48.85 % by weight). Edge pairs: I K-edge 33.17 keV, Cs K-edge 35.98 keV.
_CSI_TABLE = np.array([
    (10.0, 168.0),
    (15.0, 57.5),
    (20.0, 26.6),
    (25.0, 14.6),
    (30.0, 8.9),
    (33.17, 6.6),
    (33.17 + _EDGE_SPACING_KEV, 21.6),
    (35.98, 17.5),
    (35.98 + _EDGE_SPACING_KEV, 31.3),
    (40.0, 23.5),
    (50.0, 12.9),
    (60.0, 7.95),
    (80.0, 3.75),
    (100.0, 2.07),
    (150.0, 0.72),
])
# Pb. Edge pairs: L3 13.04 keV, K 88.00 keV (L2/L1 omitted: lead is opaque there anyway).
_LEAD_TABLE = np.array([
    (10.0, 130.6),
    (13.04, 67.0),
    (13.04 + _EDGE_SPACING_KEV, 157.0),
    (15.0, 111.6),
    (20.0, 86.4),
    (30.0, 30.3),
    (40.0, 14.4),
    (50.0, 8.04),
    (60.0, 5.02),
    (80.0, 2.42),
    (88.0, 1.91),
    (88.0 + _EDGE_SPACING_KEV, 7.68),
    (100.0, 5.55),
    (150.0, 2.01),
])


def _log_log_interpolation(table: np.ndarray, energy_kev: ArrayLike) -> np.ndarray:
    """Log-log interpolation (linear extrapolation in log-log outside the table) of a (E, mu) table."""
    e = np.asarray(torch.as_tensor(energy_kev).detach().cpu().numpy() if isinstance(energy_kev, Tensor)
                   else energy_kev, dtype=np.float64)
    if np.any(e <= 0):
        raise ValueError("energies must be positive")
    log_energy, log_mu = np.log(table[:, 0]), np.log(table[:, 1])
    x = np.log(e)
    y = np.interp(x, log_energy, log_mu)
    lo, hi = x < log_energy[0], x > log_energy[-1]
    if np.any(lo):
        s = (log_mu[1] - log_mu[0]) / (log_energy[1] - log_energy[0])
        y = np.where(lo, log_mu[0] + s * (x - log_energy[0]), y)
    if np.any(hi):
        s = (log_mu[-1] - log_mu[-2]) / (log_energy[-1] - log_energy[-2])
        y = np.where(hi, log_mu[-1] + s * (x - log_energy[-1]), y)
    return np.exp(y)


def csi_mass_attenuation(energy_kev: ArrayLike) -> np.ndarray:
    """Total mass attenuation of CsI (cm^2/g) at energies (keV), coarse XCOM table, log-log interpolated."""
    return _log_log_interpolation(_CSI_TABLE, energy_kev)


def lead_mass_attenuation(energy_kev: ArrayLike) -> np.ndarray:
    """Total mass attenuation of Pb (cm^2/g) at energies (keV), coarse XCOM table, log-log interpolated."""
    return _log_log_interpolation(_LEAD_TABLE, energy_kev)


# ---- interpolation helper -------------------------------------------------------------------------

def _bracket(nodes: Tensor, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(i0, i1, w1) for linear interpolation of x on ascending nodes [N]; clamped to the end nodes."""
    n = nodes.numel()
    if n == 1:
        z = torch.zeros_like(x, dtype=torch.long)
        return z, z, torch.zeros_like(x)
    xc = x.clamp(nodes[0], nodes[-1])
    i1 = torch.searchsorted(nodes, xc.contiguous(), right=True).clamp(1, n - 1)
    i0 = i1 - 1
    x0, x1 = nodes[i0], nodes[i1]
    w1 = ((xc - x0) / (x1 - x0)).clamp(0.0, 1.0)
    return i0, i1, w1


# ---- detector -------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class DetectorResponse:
    """Detector response LUT D(E, theta0).

    energy_kev : [E] ascending node energies (keV).
    theta_rad  : [T] ascending node angles in [0, pi/2] between the photon direction and the panel
                 normal (the absolute value of the query angle is used).
    response   : [E, T] keV deposited per incident photon (or 1 per photon for a counting detector).
    Queries outside the table are clamped to the end nodes.
    """

    energy_kev: Tensor
    theta_rad: Tensor
    response: Tensor
    name: str = "detector"

    def __post_init__(self) -> None:
        if self.response.shape != (self.energy_kev.numel(), self.theta_rad.numel()):
            raise ValueError(f"response must be [E, T] = [{self.energy_kev.numel()}, {self.theta_rad.numel()}], "
                             f"got {tuple(self.response.shape)}")

    def at(self, group_energy: Tensor, theta: Tensor) -> Tensor:
        """Bilinear interpolation of the table.

        group_energy: [G] keV; theta: [...] rad. Returns [G, ...] with theta's device and dtype.
        """
        device, dtype = theta.device, theta.dtype
        energy_nodes = self.energy_kev.to(device, dtype)
        theta_nodes = self.theta_rad.to(device, dtype)
        table = self.response.to(device, dtype)
        e = group_energy.to(device, dtype).reshape(-1)
        ie0, ie1, we = _bracket(energy_nodes, e)  # [G]
        it0, it1, wt = _bracket(theta_nodes, theta.abs())  # [...]
        r0 = table[ie0]  # [G, T]
        r1 = table[ie1]
        table_at_energy = r0 + (r1 - r0) * we[:, None]  # [G, T]
        a = table_at_energy[:, it0]  # [G, ...]
        b = table_at_energy[:, it1]
        return a + (b - a) * wt


def _theta_nodes_detector(n: int = 181, max_deg: float = 89.5) -> np.ndarray:
    # uniform in angle; D varies smoothly in theta (the 1/cos path scaling saturates at large theta)
    return np.radians(np.linspace(0.0, max_deg, n))


def csi_response(thickness_cm: float = 0.06, density: float = RHO_CSI,
                 energies: Optional[ArrayLike] = None,
                 mass_attenuation: Optional[MassAttenuation] = None,
                 thetas: Optional[ArrayLike] = None) -> DetectorResponse:
    """Energy-integrating CsI scintillator model:

        D(E, theta) = E * (1 - exp(-mu_CsI(E) * rho * t / cos(theta)))      keV per incident photon

    i.e. every interacting photon deposits its full energy (no escape, no fluorescence), with the
    oblique path length t / cos(theta). mass_attenuation: callable keV -> cm^2/g (numpy in/out); default
    `csi_mass_attenuation` (coarse XCOM table). energies: table energies in keV (default 10..150 keV
    every 0.5 keV; pass the group energies to make the table exact at the groups). thetas: table
    angles in rad (default 0..89.5 deg, 181 nodes). float64 CPU tables.
    """
    if thickness_cm < 0:
        raise ValueError("thickness must be non-negative")
    energy_nodes = np.sort(np.asarray(np.linspace(10.0, 150.0, 281) if energies is None else
                                      (energies.detach().cpu().numpy() if isinstance(energies, Tensor) else energies),
                                      dtype=np.float64).reshape(-1))
    thetas_rad = np.sort(np.asarray(_theta_nodes_detector() if thetas is None else
                                    (thetas.detach().cpu().numpy() if isinstance(thetas, Tensor) else thetas),
                                    dtype=np.float64).reshape(-1))
    mu = (csi_mass_attenuation if mass_attenuation is None else mass_attenuation)(energy_nodes)
    path = thickness_cm / np.cos(np.clip(thetas_rad, 0.0, math.pi / 2 - 1e-9))  # [T]
    response = energy_nodes[:, None] * (-np.expm1(-np.asarray(mu)[:, None] * density * path[None, :]))
    return DetectorResponse(torch.as_tensor(energy_nodes), torch.as_tensor(thetas_rad), torch.as_tensor(response),
                            name=f"CsI {thickness_cm * 10:.2f} mm (1/cos model)")


def ideal_counting_response() -> DetectorResponse:
    """D = 1 per photon: the signal equals the photon fluence (fluence-mode output)."""
    e = torch.tensor([1.0, 1000.0], dtype=torch.float64)
    th = torch.tensor([0.0, math.pi / 2], dtype=torch.float64)
    return DetectorResponse(e, th, torch.ones(2, 2, dtype=torch.float64), name="ideal counting")


def energy_weighted_response() -> DetectorResponse:
    """D = E per photon (keV): ideal energy-integrating detector, angle independent.

    Linear in E, so the bilinear interpolation reproduces E exactly between the end nodes 1..1000 keV.
    """
    e = torch.tensor([1.0, 1000.0], dtype=torch.float64)
    th = torch.tensor([0.0, math.pi / 2], dtype=torch.float64)
    return DetectorResponse(e, th, e[:, None].expand(2, 2).clone(), name="ideal energy weighted")


# ---- anti-scatter grid ------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class GridResponse:
    """Anti-scatter-grid transmission LUT G(E, theta1, theta2).

    energy_kev   : [E] ascending (keV).
    theta1_rad   : [T1] ascending, >= 0: angle in the v-source plane from the focusing direction
                   (|theta1| is looked up; the model grid is symmetric).
    theta2_rad   : [T2] ascending, >= 0: angle in the u-source plane from the panel normal (|theta2|).
    transmission : [E, T1, T2] transmitted fraction of the photons incident on the grid.
    focal_distance_cm: distance of the focal line from the panel used by `grid_angles` to define the
                   focusing direction (None: the geometry's source-detector distance, i.e. the grid is
                   focused at the focal spot).
    Queries outside the table are clamped to the end nodes. Trilinear interpolation.
    """

    energy_kev: Tensor
    theta1_rad: Tensor
    theta2_rad: Tensor
    transmission: Tensor
    focal_distance_cm: Optional[float] = None
    name: str = "grid"

    def __post_init__(self) -> None:
        shape = (self.energy_kev.numel(), self.theta1_rad.numel(), self.theta2_rad.numel())
        if tuple(self.transmission.shape) != shape:
            raise ValueError(f"transmission must be [E, T1, T2] = {list(shape)}, got {tuple(self.transmission.shape)}")

    def at(self, group_energy: Tensor, theta1: Tensor, theta2: Tensor) -> Tensor:
        """group_energy [G] keV; theta1, theta2 [...] rad (broadcastable). Returns [G, ...] (theta1's device/dtype)."""
        device, dtype = theta1.device, theta1.dtype
        theta1, theta2 = torch.broadcast_tensors(theta1, theta2.to(device, dtype))
        energy_nodes = self.energy_kev.to(device, dtype)
        theta1_nodes = self.theta1_rad.to(device, dtype)
        theta2_nodes = self.theta2_rad.to(device, dtype)
        table = self.transmission.to(device, dtype)
        e = group_energy.to(device, dtype).reshape(-1)
        ie0, ie1, we = _bracket(energy_nodes, e)
        table_at_energy = table[ie0] + (table[ie1] - table[ie0]) * we[:, None, None]  # [G, T1, T2]
        i10, i11, w1 = _bracket(theta1_nodes, theta1.abs())
        i20, i21, w2 = _bracket(theta2_nodes, theta2.abs())
        a00 = table_at_energy[:, i10, i20]
        a01 = table_at_energy[:, i10, i21]
        a10 = table_at_energy[:, i11, i20]
        a11 = table_at_energy[:, i11, i21]
        a0 = a00 + (a01 - a00) * w2
        a1 = a10 + (a11 - a10) * w2
        return a0 + (a1 - a0) * w1


def no_grid() -> GridResponse:
    """Transmission 1 for every energy and angle."""
    e = torch.tensor([1.0, 1000.0], dtype=torch.float64)
    t = torch.tensor([0.0, math.pi / 2], dtype=torch.float64)
    return GridResponse(e, t, t.clone(), torch.ones(2, 2, 2, dtype=torch.float64), name="no grid")


def _theta_nodes_grid() -> np.ndarray:
    # dense near 0 where the lamella shadow opens (cut-off at atan((pitch - t)/h) ~ 5 deg for the default grid)
    return np.radians(np.unique(np.concatenate([np.linspace(0.0, 10.0, 401), np.linspace(10.0, 89.5, 160)])))


def focused_1d_grid(ratio: float = 10.0, lamella_thickness_cm: float = 0.0036, lamella_height_cm: float = 0.15,
                    pitch_cm: float = 1.0 / 60.0, focal_distance_cm: float = 150.0,
                    mass_attenuation: Optional[MassAttenuation] = None, lead_density: float = RHO_LEAD,
                    energies: Optional[ArrayLike] = None, theta1: Optional[ArrayLike] = None) -> GridResponse:
    """Geometric model of a focused 1-D lead grid (nominal 10:1, 60 lp/cm, 36 um lamellae, 1.5 mm high, focused at
    150 cm).

    Lamellae of thickness t and height
    h repeat with pitch p along v and run along u; they are focused on a line through the focal spot
    parallel to u, so a photon arriving along the focusing direction (theta1 = 0) sees only the lamella
    cross sections. For a photon tilted by theta1 in the v-source plane:

        open  = max(0, (p - t) - h |tan theta1|) / p          fraction passing through the interspace only
              = (p - t)/p * max(0, 1 - h |tan theta1| / (p - t))
        lead path of the rest (mean over the shadowed photons): the mean lead path over all photons is
              (t/p) h / cos theta1 (lead volume fraction times the slant height), so
        L_s   = (t/p) h / cos theta1 / (1 - open)              (= h at theta1 = 0)
        G     = open + (1 - open) * exp(-mu_Pb(E) rho_Pb L_s)

    Independent of theta2 (a tilt along the lamellae does not change the shadow). The interspace
    material and the cover are ignored. `ratio` is the manufacturer's nominal label and is not used: the
    geometry is set by (t, h, p); with the default specifications h / (p - t) = 11.5 and h / p = 9.0, so
    "10:1" is only nominal. focal_distance_cm is stored for `grid_angles`.
    mass_attenuation: callable keV -> cm^2/g; default `lead_mass_attenuation` (coarse XCOM table).
    energies (keV, default 10..150 every 0.5) and theta1 (rad, default dense in [0, 10 deg]) set the nodes.
    """
    p, t, h = float(pitch_cm), float(lamella_thickness_cm), float(lamella_height_cm)
    if not (0.0 <= t < p) or h < 0:
        raise ValueError("need 0 <= lamella thickness < pitch and height >= 0")
    nominal = h / (p - t)
    energy_nodes = np.sort(np.asarray(np.linspace(10.0, 150.0, 281) if energies is None else
                                      (energies.detach().cpu().numpy() if isinstance(energies, Tensor) else energies),
                                      dtype=np.float64).reshape(-1))
    theta1_nodes = np.sort(np.asarray(_theta_nodes_grid() if theta1 is None else
                                      (theta1.detach().cpu().numpy() if isinstance(theta1, Tensor) else theta1),
                                      dtype=np.float64).reshape(-1))
    # [E]
    mu = np.asarray((lead_mass_attenuation if mass_attenuation is None else mass_attenuation)(energy_nodes))
    tan_theta1 = np.abs(np.tan(np.clip(theta1_nodes, -math.pi / 2 + 1e-9, math.pi / 2 - 1e-9)))
    cos_theta1 = np.cos(theta1_nodes)
    open_frac = np.maximum(0.0, (p - t) - h * tan_theta1) / p  # [T1]
    shadow = 1.0 - open_frac
    with np.errstate(divide="ignore", invalid="ignore"):
        path = np.where(shadow > 0, (t / p) * h / cos_theta1 / np.where(shadow > 0, shadow, 1.0), 0.0)
    # [E, T1]
    transmission = open_frac[None, :] + shadow[None, :] * np.exp(-mu[:, None] * lead_density * path[None, :])
    theta2_nodes = np.array([0.0, math.pi / 2])
    table = np.repeat(transmission[:, :, None], 2, axis=2)
    return GridResponse(torch.as_tensor(energy_nodes), torch.as_tensor(theta1_nodes), torch.as_tensor(theta2_nodes),
                        torch.as_tensor(table), focal_distance_cm=float(focal_distance_cm),
                        name=f"focused 1D grid (nominal {ratio:g}:1, h/(p-t) = {nominal:.1f}), {1 / p:.0f} lp/cm "
                             f"(geometric model)")


# ---- angles -------------------------------------------------------------------------------------------

def detector_angle(omega: Tensor, geometry: ConeGeometry, angle: float) -> Tensor:
    """Angle theta0 between the photon direction omega [..., 3] and the inward panel normal -n_hat. Returns [...]."""
    _, _, n_hat = geometry.panel_axes(angle)
    n = torch.tensor(n_hat, device=omega.device, dtype=omega.dtype)
    c = (omega * (-n)).sum(-1) / torch.linalg.norm(omega, dim=-1)
    return torch.arccos(c.clamp(-1.0, 1.0))


def grid_angles(omega: Tensor, pixel_positions: Tensor, geometry: ConeGeometry, angle: float,
                focal_distance: Optional[float] = None) -> tuple[Tensor, Tensor]:
    """Grid angles (theta1, theta2) of photons arriving at pixels.

    ASSUMPTIONS: 1-D grid whose lamellae run along u (channels) and repeat along v (rows), focused on
    the line through the focal point parallel to u, at distance `focal_distance` from the panel along
    n_hat (default geometry.source_detector_dist, i.e. through the focal spot). With a = -omega (unit
    vector from the pixel back toward where the photon came from) and f = focus - pixel:

        theta1 = atan2(a.v_hat, a.n_hat) - atan2(f.v_hat, f.n_hat)   angle in the v-source plane (the plane
                 spanned by v_hat and n_hat, perpendicular to the lamella direction) from the focusing direction
        theta2 = atan2(a.u_hat, a.n_hat)                             angle in the u-source plane from the normal

    The u component of f is irrelevant because the focal line is parallel to u. A primary ray (a = f/|f|
    with focal_distance = SDD) has theta1 = 0 exactly and theta2 = -atan(u / SDD).

    Shapes: omega [K, 3] with pixel_positions [rows, channels, 3] -> [K, rows, channels]; otherwise omega
    [..., 3] and pixel_positions [..., 3] are broadcast elementwise -> [...]. Recon frame, cm, rad.
    """
    device, dtype = pixel_positions.device, pixel_positions.dtype
    omega = omega.to(device, dtype)
    u_hat, v_hat, n_hat = (torch.tensor(a, device=device, dtype=dtype) for a in geometry.panel_axes(angle))
    focal_distance = geometry.source_detector_dist if focal_distance is None else float(focal_distance)
    focus = torch.tensor(geometry.panel_center(angle), device=device, dtype=dtype) + focal_distance * n_hat
    if omega.dim() == 2 and pixel_positions.dim() == 3:
        a = -omega[:, None, None, :]  # [K, 1, 1, 3]
        pixels = pixel_positions[None]
    else:
        a = -omega
        pixels = pixel_positions
    f = focus - pixels
    a_normal = (a * n_hat).sum(-1)
    theta_a = torch.atan2((a * v_hat).sum(-1), a_normal)
    theta_f = torch.atan2((f * v_hat).sum(-1), (f * n_hat).sum(-1))
    theta1 = theta_a - theta_f
    theta2 = torch.atan2((a * u_hat).sum(-1), a_normal)
    theta1, theta2 = torch.broadcast_tensors(theta1, theta2)
    return theta1, theta2
