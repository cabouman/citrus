"""Detector and anti-scatter-grid responses as lookup tables, and the angles at which they are evaluated.

:class:`DetectorResponse` tabulates D(E, theta0), the energy deposited per incident photon, and :class:`GridResponse`
tabulates G(E, theta1, theta2), the transmitted fraction of a 1-D anti-scatter grid; tables from a Monte Carlo
simulation can replace the analytic builders :func:`csi_response` and :func:`focused_1d_grid`. :func:`detector_angle`
and :func:`grid_angles` give the angles of the photons arriving at the panel.
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
    """Interpolate an (E, mu) table linearly in log-log, extrapolating linearly in log-log outside it.

    Args:
        table (np.ndarray): [N, 2] energies in keV (ascending) and mass attenuation coefficients in cm^2/g.
        energy_kev (ArrayLike): photon energies in keV, positive.

    Returns:
        np.ndarray: mass attenuation coefficients in cm^2/g, the shape of ``energy_kev``.
    """
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
    """Return the total mass attenuation of CsI from a coarse XCOM table, interpolated in log-log.

    Args:
        energy_kev (ArrayLike): photon energies in keV.

    Returns:
        np.ndarray: total mass attenuation (coherent scattering included) in cm^2/g, the shape of ``energy_kev``.
    """
    return _log_log_interpolation(_CSI_TABLE, energy_kev)


def lead_mass_attenuation(energy_kev: ArrayLike) -> np.ndarray:
    """Return the total mass attenuation of lead from a coarse XCOM table, interpolated in log-log.

    Args:
        energy_kev (ArrayLike): photon energies in keV.

    Returns:
        np.ndarray: total mass attenuation (coherent scattering included) in cm^2/g, the shape of ``energy_kev``.
    """
    return _log_log_interpolation(_LEAD_TABLE, energy_kev)


# ---- interpolation helper -------------------------------------------------------------------------

def _bracket(nodes: Tensor, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Find the two bracketing nodes and the interpolation weight of each query, clamped to the end nodes.

    Args:
        nodes (Tensor): [N] ascending node values.
        x (Tensor): [...] query values.

    Returns:
        (Tensor, Tensor, Tensor): i0 and i1 [...] indices of the lower and upper nodes, w1 [...] weight of the upper
        node.
    """
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
    """Detector response lookup table D(E, theta0), in keV deposited per incident photon.

    Queries are interpolated bilinearly and clamped to the end nodes; the absolute value of the query angle is used.

    Attributes:
        energy_kev (Tensor): [E] ascending node energies in keV.
        theta_rad (Tensor): [T] ascending node angles in rad, in [0, pi/2], between the photon direction and the panel
            normal.
        response (Tensor): [E, T] keV deposited per incident photon (1 per photon for a counting detector).
        name (str): label of the response. Defaults to "detector".
    """

    energy_kev: Tensor
    theta_rad: Tensor
    response: Tensor
    name: str = "detector"

    def __post_init__(self) -> None:
        """Check that ``response`` is [E, T]."""
        if self.response.shape != (self.energy_kev.numel(), self.theta_rad.numel()):
            raise ValueError(f"response must be [E, T] = [{self.energy_kev.numel()}, {self.theta_rad.numel()}], "
                             f"got {tuple(self.response.shape)}")

    def at(self, group_energy: Tensor, theta: Tensor) -> Tensor:
        """Interpolate the table bilinearly at group energies and angles.

        Args:
            group_energy (Tensor): [G] energies in keV.
            theta (Tensor): [...] angles to the panel normal in rad.

        Returns:
            Tensor: [G, ...] response, on the device and in the dtype of ``theta``.
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
    """Return angle nodes uniform from 0 to ``max_deg`` degrees.

    Args:
        n (int, optional): number of nodes. Defaults to 181.
        max_deg (float, optional): largest angle in degrees. Defaults to 89.5.

    Returns:
        np.ndarray: [n] angles in rad.
    """
    # uniform in angle; D varies smoothly in theta (the 1/cos path scaling saturates at large theta)
    return np.radians(np.linspace(0.0, max_deg, n))


def csi_response(thickness_cm: float = 0.06, density: float = RHO_CSI,
                 energies: Optional[ArrayLike] = None,
                 mass_attenuation: Optional[MassAttenuation] = None,
                 thetas: Optional[ArrayLike] = None) -> DetectorResponse:
    """Build the response of an energy-integrating CsI scintillator with the oblique path length ``t / cos(theta)``.

    Every interacting photon deposits its full energy, ``D(E, theta) = E * (1 - exp(-mu(E) * rho * t / cos(theta)))``
    in keV per incident photon.

    Args:
        thickness_cm (float, optional): scintillator thickness t in cm. Defaults to 0.06.
        density (float, optional): density rho in g/cm^3. Defaults to ``RHO_CSI`` (4.51).
        energies (ArrayLike, optional): table energies in keV; None uses 10 to 150 keV every 0.5 keV (pass the group
            energies to make the table exact at the groups). Defaults to None.
        mass_attenuation (Callable, optional): mass attenuation mu in cm^2/g as a function of keV (numpy in and out);
            None uses :func:`csi_mass_attenuation`. Defaults to None.
        thetas (ArrayLike, optional): table angles in rad; None uses 181 nodes from 0 to 89.5 degrees. Defaults to None.

    Returns:
        DetectorResponse: the response, float64 tables on the CPU.
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
    """Build the response D = 1 per photon, so that the signal is the photon fluence.

    Returns:
        DetectorResponse: the constant table.
    """
    e = torch.tensor([1.0, 1000.0], dtype=torch.float64)
    th = torch.tensor([0.0, math.pi / 2], dtype=torch.float64)
    return DetectorResponse(e, th, torch.ones(2, 2, dtype=torch.float64), name="ideal counting")


def energy_weighted_response() -> DetectorResponse:
    """Build the response D = E per photon in keV of an ideal, angle-independent energy-integrating detector.

    The table is linear in E, so the bilinear interpolation reproduces E exactly between its end nodes, 1 and 1000 keV.

    Returns:
        DetectorResponse: the table.
    """
    e = torch.tensor([1.0, 1000.0], dtype=torch.float64)
    th = torch.tensor([0.0, math.pi / 2], dtype=torch.float64)
    return DetectorResponse(e, th, e[:, None].expand(2, 2).clone(), name="ideal energy weighted")


# ---- anti-scatter grid ------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class GridResponse:
    """Anti-scatter-grid transmission lookup table G(E, theta1, theta2) at the grid angles of :func:`grid_angles`.

    Queries are interpolated trilinearly and clamped to the end nodes; the absolute values of the angles are used.

    Attributes:
        energy_kev (Tensor): [E] ascending node energies in keV.
        theta1_rad (Tensor): [T1] ascending angles >= 0 in rad in the v-source plane, from the focusing direction.
        theta2_rad (Tensor): [T2] ascending angles >= 0 in rad in the u-source plane, from the panel normal.
        transmission (Tensor): [E, T1, T2] transmitted fraction of the photons incident on the grid.
        focal_distance_cm (float, optional): distance of the focal line from the panel in cm, which defines the
            focusing direction; None uses the source-detector distance (focused at the focal spot). Defaults to None.
        name (str): label of the response. Defaults to "grid".
    """

    energy_kev: Tensor
    theta1_rad: Tensor
    theta2_rad: Tensor
    transmission: Tensor
    focal_distance_cm: Optional[float] = None
    name: str = "grid"

    def __post_init__(self) -> None:
        """Check that ``transmission`` is [E, T1, T2]."""
        shape = (self.energy_kev.numel(), self.theta1_rad.numel(), self.theta2_rad.numel())
        if tuple(self.transmission.shape) != shape:
            raise ValueError(f"transmission must be [E, T1, T2] = {list(shape)}, got {tuple(self.transmission.shape)}")

    def at(self, group_energy: Tensor, theta1: Tensor, theta2: Tensor) -> Tensor:
        """Interpolate the table trilinearly at group energies and grid angles.

        Args:
            group_energy (Tensor): [G] energies in keV.
            theta1 (Tensor): [...] angles in the v-source plane in rad.
            theta2 (Tensor): [...] angles in the u-source plane in rad, broadcastable with ``theta1``.

        Returns:
            Tensor: [G, ...] transmission, on the device and in the dtype of ``theta1``.
        """
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
    """Build the response of no grid, transmission 1 for every energy and angle.

    Returns:
        GridResponse: the constant table.
    """
    e = torch.tensor([1.0, 1000.0], dtype=torch.float64)
    t = torch.tensor([0.0, math.pi / 2], dtype=torch.float64)
    return GridResponse(e, t, t.clone(), torch.ones(2, 2, 2, dtype=torch.float64), name="no grid")


def _theta_nodes_grid() -> np.ndarray:
    """Return theta1 nodes, dense from 0 to 10 degrees where the lamella shadow opens and coarser up to 89.5 degrees.

    Returns:
        np.ndarray: [N] angles in rad.
    """
    # dense near 0 where the lamella shadow opens (cut-off at atan((pitch - t)/h) ~ 5 deg for the default grid)
    return np.radians(np.unique(np.concatenate([np.linspace(0.0, 10.0, 401), np.linspace(10.0, 89.5, 160)])))


def focused_1d_grid(ratio: float = 10.0, lamella_thickness_cm: float = 0.0036, lamella_height_cm: float = 0.15,
                    pitch_cm: float = 1.0 / 60.0, focal_distance_cm: float = 150.0,
                    mass_attenuation: Optional[MassAttenuation] = None, lead_density: float = RHO_LEAD,
                    energies: Optional[ArrayLike] = None, theta1: Optional[ArrayLike] = None) -> GridResponse:
    """Build the transmission of a focused 1-D lead grid with a geometric lamella-shadow model.

    Lamellae of thickness t and height h repeat with pitch p along v and are focused on a line through the focal spot
    parallel to u; a photon tilted by theta1 passes the interspace (open fraction) or crosses a mean lead path L_s,
    ``G = open + (1 - open) * exp(-mu_Pb(E) * rho_Pb * L_s)``, independent of theta2.

    Args:
        ratio (float, optional): nominal grid ratio, used only in the name (the geometry is set by t, h and p).
            Defaults to 10.0.
        lamella_thickness_cm (float, optional): lamella thickness t in cm. Defaults to 0.0036.
        lamella_height_cm (float, optional): lamella height h in cm. Defaults to 0.15.
        pitch_cm (float, optional): lamella pitch p in cm. Defaults to 1/60 (60 lines per cm).
        focal_distance_cm (float, optional): distance of the focal line from the panel in cm, stored for
            :func:`grid_angles`. Defaults to 150.0.
        mass_attenuation (Callable, optional): mass attenuation ``mu_Pb`` in cm^2/g as a function of keV (numpy in and
            out); None uses :func:`lead_mass_attenuation`. Defaults to None.
        lead_density (float, optional): lead density ``rho_Pb`` in g/cm^3. Defaults to ``RHO_LEAD`` (11.35).
        energies (ArrayLike, optional): table energies in keV; None uses 10 to 150 keV every 0.5 keV. Defaults to None.
        theta1 (ArrayLike, optional): table angles theta1 in rad; None uses nodes dense from 0 to 10 degrees.
            Defaults to None.

    Returns:
        GridResponse: the response, float64 tables on the CPU.
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
    """Compute the angle theta0 between the photon direction and the inward panel normal ``-n_hat``.

    Args:
        omega (Tensor): [..., 3] photon directions in the recon frame (any length).
        geometry (ConeGeometry): the scan geometry.
        angle (float): view angle in rad.

    Returns:
        Tensor: [...] angles theta0 in rad.
    """
    _, _, n_hat = geometry.panel_axes(angle)
    n = torch.tensor(n_hat, device=omega.device, dtype=omega.dtype)
    c = (omega * (-n)).sum(-1) / torch.linalg.norm(omega, dim=-1)
    return torch.arccos(c.clamp(-1.0, 1.0))


def grid_angles(omega: Tensor, pixel_positions: Tensor, geometry: ConeGeometry, angle: float,
                focal_distance: Optional[float] = None) -> tuple[Tensor, Tensor]:
    """Compute the grid angles (theta1, theta2) of photons arriving at pixels.

    The grid is 1-D, its lamellae run along u and repeat along v, focused on the line parallel to u at
    ``focal_distance`` from the panel along ``n_hat``; theta1 is the angle in the v-source plane from the focusing
    direction and theta2 the angle in the u-source plane from the panel normal.

    Args:
        omega (Tensor): [K, 3] or [..., 3] unit photon directions in the recon frame.
        pixel_positions (Tensor): [rows, channels, 3], or [..., 3] broadcastable with ``omega``, pixel positions in
            cm in the recon frame.
        geometry (ConeGeometry): the scan geometry.
        angle (float): view angle in rad.
        focal_distance (float, optional): distance of the focal line from the panel in cm; None uses
            ``geometry.source_detector_dist`` (focused at the focal spot). Defaults to None.

    Returns:
        (Tensor, Tensor): theta1 and theta2 in rad, [K, rows, channels] for ``omega`` [K, 3] with ``pixel_positions``
        [rows, channels, 3], else the broadcast shape [...].
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
