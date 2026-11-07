"""Analytic signal response of a 1-D anti-scatter grid from its geometry and materials, without a Monte Carlo calibration.

:func:`grid_signal_transmission` tabulates, for a parallel beam of one energy and direction, the energy a detector
layer behind the grid absorbs per incident photon relative to the same beam without the grid, split into the photons
that cross the grid without interacting (direct) and the first-order scatter and fluorescence of the grid that reach
the detector (secondary). ``detector.anti_scatter_grid`` builds its default grid from it; docs/source/method.rst,
"Grid model", has the equations.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Optional

import numpy as np
import torch
from torch import Tensor

__all__ = ["grid_signal_transmission", "grid_directions"]

# version of the model in the disk cache key: change it whenever the numbers of the model change
_MODEL_VERSION = 1
# K and L lines of the strip material emitted as fluorescence (xraylib line names)
FLUORESCENCE_LINES = ("KA1_LINE", "KA2_LINE", "KA3_LINE", "KB1_LINE", "KB2_LINE", "KB3_LINE", "KB4_LINE", "KB5_LINE",
                      "LA1_LINE", "LA2_LINE", "LB1_LINE", "LB2_LINE", "LB3_LINE", "LB4_LINE", "LB5_LINE", "LB6_LINE",
                      "LG1_LINE", "LG2_LINE", "LG3_LINE", "LL_LINE", "LE_LINE")
_MIN_FLUORESCENCE_Z = 20  # lighter elements emit lines below a few keV, absorbed in the grid
_GAS_DENSITY = 0.01  # g/cm^3; an interspace below it (air) emits no secondaries
_SECONDARY_STEP_KEV = 5.0  # spacing of the secondary energies up to _SECONDARY_LINEAR_TOP_KEV
_SECONDARY_LINEAR_TOP_KEV = 150.0
_EDGE_OFFSET_KEV = 0.05  # secondary energies on both sides of every edge, this far from it
_COMPTON_CHUNK_BYTES = 256 * 2 ** 20  # memory of one chunk of the outgoing Compton attenuation
_MEMORY_CACHE: dict[str, dict[str, np.ndarray]] = {}


# ---- geometry ---------------------------------------------------------------------------------------------------

def _cumulative_strip_length(x: Tensor, pitch: float, thickness: float) -> Tensor:
    """Length of the strips (at ``[k p, k p + t)``) between 0 and x along the axis across them.

    Args:
        x (Tensor): [...] positions in cm.
        pitch (float): strip pitch p in cm.
        thickness (float): strip thickness t in cm.

    Returns:
        Tensor: [...] strip length in cm, negative for x < 0.
    """
    period = torch.floor(x / pitch)
    return period * thickness + torch.clamp(x - period * pitch, 0.0, thickness)


def _strip_path(x_start: Tensor, x_end: Tensor, length: Tensor, pitch: float, thickness: float) -> Tensor:
    """Path length inside the strips of a straight segment of a given length between two positions across them.

    Args:
        x_start (Tensor): [...] start positions across the strips in cm.
        x_end (Tensor): [...] end positions in cm.
        length (Tensor): [...] segment lengths in cm.
        pitch (float): strip pitch in cm.
        thickness (float): strip thickness in cm.

    Returns:
        Tensor: [...] path length inside the strips in cm.
    """
    dx = x_end - x_start
    moving = dx.abs() > 1e-12
    inside = (torch.remainder(x_start, pitch) < thickness).to(length.dtype)
    covered = (_cumulative_strip_length(x_end, pitch, thickness)
               - _cumulative_strip_length(x_start, pitch, thickness)).abs()
    return torch.where(moving, covered / torch.where(moving, dx.abs(), torch.ones_like(dx)) * length, inside * length)


def _gauss_panels(edges: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on consecutive panels.

    Args:
        edges (np.ndarray): [P + 1] ascending panel edges.
        order (int): nodes per panel.

    Returns:
        (np.ndarray, np.ndarray): nodes and weights, [P order] each.
    """
    x, w = np.polynomial.legendre.leggauss(order)
    a, b = edges[:-1, None], edges[1:, None]
    return (a + 0.5 * (x[None] + 1.0) * (b - a)).ravel(), (0.5 * w[None] * (b - a)).ravel()


def grid_directions(slot_angle_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Directions toward the panel that resolve the slots between the strips, with their solid angles.

    A direction is ``(tan a, tan b, 1) / norm`` with a across the strips and b along them; a runs over Gauss panels
    whose edges scale with the slot angle (dense near a = 0) and b over fixed panels.

    Args:
        slot_angle_rad (float): ``atan((p - t) / h)``, the largest tilt across the strips that sees through a slot.

    Returns:
        (np.ndarray, np.ndarray): [K, 3] unit directions (third component > 0) and [K] solid angles in sr, which sum
        to 2 pi.
    """
    half = math.pi / 2 - 1e-6
    across = [e * slot_angle_rad for e in (0.2, 0.6, 1.5, 3.5, 7.0) if e * slot_angle_rad < half - 0.05]
    across = np.array(across + [half])
    a, wa = _gauss_panels(np.concatenate([-across[::-1], [0.0], across]), 12)
    b, wb = _gauss_panels(np.array([-half, -0.6, -0.2, 0.2, 0.6, half]), 16)
    A, B = np.meshgrid(a, b, indexing="ij")
    ta, tb = np.tan(A), np.tan(B)
    norm = np.sqrt(1.0 + ta ** 2 + tb ** 2)
    directions = np.stack([ta / norm, tb / norm, 1.0 / norm], -1).reshape(-1, 3)
    weights = (wa[:, None] * wb[None, :] / np.cos(A) ** 2 / np.cos(B) ** 2 / norm ** 3).ravel()
    return directions, weights


# ---- material data ------------------------------------------------------------------------------------------------

class _Attenuation:
    """Linear attenuation of a material on a fine energy table, interpolated linearly on the device.

    The table holds every absorption edge of the material as a pair of nodes (the value below and at the edge), so
    the interpolation keeps the jumps.

    Args:
        composition (dict[int, float]): mass fraction per atomic number.
        density (float): density in g/cm^3.
        max_energy_kev (float): largest energy needed in keV.
        backend (photon_data.XcomXraylibBackend): photon data.
        device (torch.device): device of the table.
    """

    def __init__(self, composition: dict[int, float], density: float, max_energy_kev: float, backend,
                 device: torch.device) -> None:
        """Tabulate the attenuation; the class docstring lists the arguments."""
        top = max(float(max_energy_kev) * 1.001, 2.0)
        nodes = [np.arange(1.0, min(top, 200.0) + 0.02, 0.02)]
        if top > 200.0:
            nodes.append(np.geomspace(200.0, top, 600))
        for Z in composition:
            edges = np.asarray(backend.element(Z).edge_energy_ev, dtype=np.float64) * 1.0e-3
            edges = edges[(edges > 1.0) & (edges < top)]
            nodes += [edges * (1.0 - 1e-7), edges]
        energy = np.unique(np.concatenate(nodes))
        mu = sum(w * backend.sigma("total", Z, energy) for Z, w in composition.items()) * float(density)
        self.energy = torch.as_tensor(energy, device=device)
        self.mu = torch.as_tensor(mu, device=device)

    def __call__(self, energy_kev: Tensor) -> Tensor:
        """Interpolate the attenuation.

        Args:
            energy_kev (Tensor): [...] energies in keV on the table's device.

        Returns:
            Tensor: [...] attenuation in 1/cm, float64.
        """
        e = energy_kev.to(torch.float64).clamp(self.energy[0], self.energy[-1])
        i1 = torch.searchsorted(self.energy, e.contiguous(), right=True).clamp(1, self.energy.numel() - 1)
        e0, e1 = self.energy[i1 - 1], self.energy[i1]
        w = ((e - e0) / (e1 - e0)).clamp(0.0, 1.0)
        return self.mu[i1 - 1] + (self.mu[i1] - self.mu[i1 - 1]) * w


# momentum-transfer nodes q = sin(theta / 2) of the scattering tables, dense in the forward direction
_Q_NODES = np.unique(np.concatenate([np.linspace(0.0, 0.05, 257), np.linspace(0.05, 1.0, 769)]))


def _scattering_table(composition: dict[int, float], density: float, kind: str, energy_kev: float,
                      backend) -> np.ndarray:
    """Differential scattering coefficient of a material at the nodes ``_Q_NODES``, normalized to XCOM.

    Each element's xraylib shape (``pdf_raw``) is scaled so that it integrates to the XCOM cross section of the
    process, as in the library builder.

    Args:
        composition (dict[int, float]): mass fraction per atomic number.
        density (float): density in g/cm^3.
        kind (str): "compton" or "rayleigh".
        energy_kev (float): photon energy in keV.
        backend (photon_data.XcomXraylibBackend): photon data.

    Returns:
        np.ndarray: [Q] scattering coefficient per steradian in 1/(cm sr).
    """
    eta = 1.0 - 2.0 * _Q_NODES ** 2
    table = np.zeros_like(eta)
    for Z, w in composition.items():
        norm = backend.library_pdf_normalization(kind, Z, energy_kev)
        if norm <= 0:
            continue
        sigma = float(backend.sigma(kind, Z, np.array([energy_kev]))[0])
        table += w * sigma / norm * backend.pdf_raw(kind, Z, energy_kev, eta)
    return table * float(density)


def _fluorescence_lines(composition: dict[int, float], density: float,
                        energies_kev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """K and L fluorescence lines of a material and their production at the given photon energies (xraylib).

    Args:
        composition (dict[int, float]): mass fraction per atomic number.
        density (float): density in g/cm^3.
        energies_kev (np.ndarray): [S] incident photon energies in keV.

    Returns:
        (np.ndarray, np.ndarray): [N] line energies in keV and [S, N] line photons produced per unit path in 1/cm
        (cascades included).
    """
    import xraylib

    line_energy, production = [], []
    for Z, w in composition.items():
        if Z < _MIN_FLUORESCENCE_Z:
            continue
        for name in FLUORESCENCE_LINES:
            line = getattr(xraylib, name, None)
            if line is None:
                continue
            try:
                energy = xraylib.LineEnergy(int(Z), line)
            except ValueError:
                continue
            if energy <= 1.0:
                continue
            values = []
            for e in energies_kev:
                try:
                    values.append(xraylib.CS_FluorLine_Kissel_Cascade(int(Z), line, float(e)))
                except ValueError:
                    values.append(0.0)
            values = np.asarray(values) * w * float(density)
            if np.any(values > 0):
                line_energy.append(float(energy))
                production.append(values)
    if not line_energy:
        return np.zeros(0), np.zeros((len(energies_kev), 0))
    return np.asarray(line_energy), np.stack(production, axis=1)


def _detector_at_pairs(detector, energy_kev: Tensor, theta_rad: Tensor) -> Tensor:
    """Interpolate a detector response bilinearly at pairs of energy and incidence angle.

    Args:
        detector (DetectorResponse): the response.
        energy_kev (Tensor): [...] energies in keV.
        theta_rad (Tensor): [...] angles to the normal in rad, broadcastable with ``energy_kev``.

    Returns:
        Tensor: [...] keV deposited per photon, float64.
    """
    from .detector import _bracket

    device = energy_kev.device
    nodes_e = detector.energy_kev.to(device, torch.float64)
    nodes_t = detector.theta_rad.to(device, torch.float64)
    table = detector.response.to(device, torch.float64)
    e, t = torch.broadcast_tensors(energy_kev.to(torch.float64), theta_rad.to(device, torch.float64).abs())
    ie0, ie1, we = _bracket(nodes_e, e)
    it0, it1, wt = _bracket(nodes_t, t)
    low = table[ie0, it0] + (table[ie0, it1] - table[ie0, it0]) * wt
    high = table[ie1, it0] + (table[ie1, it1] - table[ie1, it0]) * wt
    return low + (high - low) * we


# ---- the two parts --------------------------------------------------------------------------------------------

def _direct(pitch: float, thickness: float, height: float, mu_strip: Tensor, mu_interspace: Tensor, theta1: Tensor,
            theta2: Tensor) -> Tensor:
    """Transmission of the photons that cross the grid without interacting, exact over the entry offset.

    Over one pitch of entry offsets the strip path is piecewise linear in the offset, with breaks where the entry or
    the exit point meets a strip face, so the mean of the exponential is a sum of closed-form segment integrals.

    Args:
        pitch (float): strip pitch p in cm.
        thickness (float): strip thickness t in cm.
        height (float): strip height h in cm.
        mu_strip (Tensor): [E] attenuation of the strips in 1/cm.
        mu_interspace (Tensor): [E] attenuation of the interspace in 1/cm.
        theta1 (Tensor): [T1] angles across the strips in rad.
        theta2 (Tensor): [T2] angles along the strips in rad.

    Returns:
        Tensor: [E, T1, T2] transmission, float64.
    """
    p, t, h = pitch, thickness, height
    tan1 = torch.tan(theta1.abs().clamp(max=math.pi / 2 - 1e-9))  # [T1]
    tan2 = torch.tan(theta2.abs().clamp(max=math.pi / 2 - 1e-9))  # [T2]
    path = h * torch.sqrt(1.0 + tan1[:, None] ** 2 + tan2[None, :] ** 2)  # [T1, T2]
    shift = h * tan1  # [T1] displacement across the strips from top to bottom
    zero = torch.zeros_like(shift)
    breaks = torch.stack([zero, zero + t, torch.remainder(-shift, p), torch.remainder(t - shift, p), zero + p], -1)
    breaks, _ = torch.sort(breaks, dim=-1)  # [T1, 5]
    moving = shift > 1e-12 * p
    covered = (_cumulative_strip_length(breaks + shift[:, None], p, t) - _cumulative_strip_length(breaks, p, t))
    fraction = torch.where(moving[:, None], covered / torch.where(moving, shift, torch.ones_like(shift))[:, None],
                           zero[:, None])  # [T1, 5] share of the path inside the strips at each break
    contrast = (mu_strip - mu_interspace)[:, None, None, None]  # [E, 1, 1, 1]
    exponent = contrast * fraction[None, :, None, :] * path[None, :, :, None]  # [E, T1, T2, 5]
    lower, upper = exponent[..., :-1], exponent[..., 1:]
    spread = (upper - lower).abs()
    mean_exp = torch.exp(-torch.minimum(lower, upper)) * torch.where(
        spread > 1e-12, -torch.expm1(-spread) / spread.clamp(min=1e-300), torch.ones_like(spread))
    lengths = (breaks[:, 1:] - breaks[:, :-1])[None, :, None, :]  # [1, T1, 1, 4]
    sloped = (mean_exp * lengths).sum(-1) / p  # [E, T1, T2]
    still = ((p - t) + t * torch.exp(-contrast[..., 0] * path[None])) / p  # [E, T1, T2] (no shift: theta1 = 0)
    across = torch.where(moving[None, :, None], sloped, still)
    return across * torch.exp(-mu_interspace[:, None, None] * path[None])


class _Emitters:
    """Emission points of one material of the grid cell with everything that does not depend on the energy.

    Args:
        region (tuple[float, float]): the cell's extent of the material across the strips in cm.
        pitch (float): strip pitch in cm.
        thickness (float): strip thickness in cm.
        height (float): strip height in cm.
        directions (Tensor): [K, 3] outgoing directions toward the panel.
        device (torch.device): the device.
        points_across (int, optional): emission points across the region. Defaults to 16.
        points_down (int, optional): emission points along the height. Defaults to 48.
    """

    def __init__(self, region: tuple[float, float], pitch: float, thickness: float, height: float,
                 directions: Tensor, device: torch.device, points_across: int = 16, points_down: int = 48) -> None:
        """Place the points and their paths to the bottom of the grid; the class docstring lists the arguments."""
        a, b = region
        x = a + (torch.arange(points_across, dtype=torch.float64, device=device) + 0.5) / points_across * (b - a)
        z = -height + (torch.arange(points_down, dtype=torch.float64, device=device) + 0.5) / points_down * height
        X, Z = torch.meshgrid(x, z, indexing="ij")
        self.x, self.z = X.reshape(-1), Z.reshape(-1)  # [J]; z from -h (top) to 0 (bottom of the grid)
        self.area = (b - a) / points_across * height / points_down / pitch  # cross-section per unit pitch
        self.pitch, self.thickness, self.height = pitch, thickness, height
        out_length = -self.z[:, None] / directions[None, :, 2]  # [J, K]
        out_strip = _strip_path(self.x[:, None], self.x[:, None] + directions[None, :, 0] * out_length, out_length,
                                pitch, thickness)
        self.out_strip = out_strip.to(torch.float32)
        self.out_interspace = (out_length - out_strip).to(torch.float32)

    def incoming(self, incident: Tensor) -> tuple[Tensor, Tensor]:
        """Paths in the strips and in the interspace from the top of the grid to each point.

        Args:
            incident (Tensor): [A, 3] unit incident directions (third component > 0).

        Returns:
            (Tensor, Tensor): [A, J] strip and interspace path lengths in cm.
        """
        length = (self.z[None, :] + self.height) / incident[:, 2:3]  # [A, J]
        strip = _strip_path(self.x[None, :] - incident[:, 0:1] * length, self.x[None, :] + 0 * length, length,
                            self.pitch, self.thickness)
        return strip, length - strip

    def escape(self, mu_strip: Tensor, mu_interspace: Tensor) -> Tensor:
        """Attenuation from every point to the bottom of the grid along every direction, one energy for all.

        Args:
            mu_strip (Tensor): [] or [K] strip attenuation in 1/cm.
            mu_interspace (Tensor): [] or [K] interspace attenuation in 1/cm.

        Returns:
            Tensor: [J, K] float32 attenuation factors.
        """
        exponent = self.out_strip * mu_strip.to(torch.float32)
        exponent.addcmul_(self.out_interspace, mu_interspace.to(torch.float32).expand_as(exponent[0]))
        return exponent.neg_().exp_()


def _secondary_energies(energy_nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Energies at which the secondaries are computed: a regular grid plus both sides of every edge.

    Args:
        energy_nodes (np.ndarray): [E] table energies in keV.
        edges (np.ndarray): absorption edges of the grid's materials in keV.

    Returns:
        np.ndarray: [S] ascending energies in keV within the table's range.
    """
    low, high = float(energy_nodes.min()), float(energy_nodes.max())
    parts = [np.array([low, high]),
             np.arange(_SECONDARY_STEP_KEV * math.ceil(low / _SECONDARY_STEP_KEV),
                       min(high, _SECONDARY_LINEAR_TOP_KEV) + 1e-9, _SECONDARY_STEP_KEV)]
    if high > _SECONDARY_LINEAR_TOP_KEV:
        count = int(math.ceil(5 * math.log10(high / _SECONDARY_LINEAR_TOP_KEV))) + 1
        parts.append(np.geomspace(_SECONDARY_LINEAR_TOP_KEV, high, count))
    edges = np.asarray(edges, dtype=np.float64)
    parts += [edges - _EDGE_OFFSET_KEV, edges + _EDGE_OFFSET_KEV]
    energies = np.unique(np.round(np.concatenate(parts), 6))
    return energies[(energies >= low) & (energies <= high)]


def _secondary_angles(slot_angle_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Grid angles at which the secondaries are computed, dense across the strips within a few slot angles.

    Args:
        slot_angle_rad (float): ``atan((p - t) / h)``.

    Returns:
        (np.ndarray, np.ndarray): theta1 [A1] and theta2 [A2] nodes in rad.
    """
    top = math.radians(89.5)
    near = slot_angle_rad * np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0])
    far = np.radians([15.0, 20.0, 30.0, 45.0, 60.0, 75.0])
    theta1 = np.concatenate([near[near < top], far[far > 1.2 * near.max()], [top]])
    theta2 = np.radians([0.0, 20.0, 40.0, 60.0, 80.0])
    return np.unique(theta1), theta2


def _secondaries(pitch: float, thickness: float, height: float, strip: tuple, interspace: tuple, detector,
                 energies: np.ndarray, theta1: np.ndarray, theta2: np.ndarray, backend,
                 device: torch.device) -> np.ndarray:
    """Signal of the grid's first-order scatter and fluorescence relative to the signal without the grid.

    Args:
        pitch (float): strip pitch in cm.
        thickness (float): strip thickness in cm.
        height (float): strip height in cm.
        strip (tuple): (density, composition) of the strips.
        interspace (tuple): (density, composition) of the interspace.
        detector (DetectorResponse): the layer that scores the signal.
        energies (np.ndarray): [S] incident energies in keV.
        theta1 (np.ndarray): [A1] angles across the strips in rad.
        theta2 (np.ndarray): [A2] angles along the strips in rad.
        backend (photon_data.XcomXraylibBackend): photon data.
        device (torch.device): the device.

    Returns:
        np.ndarray: [S, A1, A2] relative signal.
    """
    from .cross_sections import ELECTRON_REST_KEV

    f64 = torch.float64
    slot = math.atan2(pitch - thickness, height)
    U, W = (torch.as_tensor(a, device=device) for a in grid_directions(slot))  # [K, 3], [K]
    theta_out = torch.arccos(U[:, 2].clamp(-1.0, 1.0))  # [K] incidence angles on the detector
    A1, A2 = np.meshgrid(theta1, theta2, indexing="ij")
    t1, t2 = np.tan(A1.ravel()), np.tan(A2.ravel())
    incident = np.stack([t1, t2, np.ones_like(t1)], -1)
    incident = torch.as_tensor(incident / np.linalg.norm(incident, axis=-1, keepdims=True), device=device)  # [A, 3]
    theta_in = torch.arccos(incident[:, 2])
    cos_scatter = (incident @ U.T).clamp(-1.0, 1.0)  # [A, K]
    q = torch.sqrt((0.5 * (1.0 - cos_scatter)).clamp(min=0.0))
    q_nodes = torch.as_tensor(_Q_NODES, device=device)
    iq1 = torch.searchsorted(q_nodes, q.contiguous(), right=True).clamp(1, q_nodes.numel() - 1)
    wq = ((q - q_nodes[iq1 - 1]) / (q_nodes[iq1] - q_nodes[iq1 - 1])).clamp(0.0, 1.0)

    max_energy = float(energies.max())
    mu_strip = _Attenuation(strip[1], strip[0], max_energy, backend, device)
    mu_interspace = _Attenuation(interspace[1], interspace[0], max_energy, backend, device)
    parts = [(strip, _Emitters((0.0, thickness), pitch, thickness, height, U, device))]
    if interspace[0] >= _GAS_DENSITY:
        parts.append((interspace, _Emitters((thickness, pitch), pitch, thickness, height, U, device)))
    total = torch.zeros(len(energies), incident.shape[0], dtype=f64, device=device)
    num_directions = U.shape[0]
    for (density, composition), emitters in parts:
        strip_in, interspace_in = emitters.incoming(incident)  # [A, J]
        J = emitters.x.numel()
        chunk = max(1, int(_COMPTON_CHUNK_BYTES // (4 * J * num_directions)))
        line_energy, production = _fluorescence_lines(composition, density, energies)  # [N], [S, N]
        if line_energy.size:  # isotropic lines: the escape of each line, per point, once
            lines = torch.as_tensor(line_energy, device=device)
            line_weight = W[None, :] * _detector_at_pairs(detector, lines[:, None], theta_out[None, :]) / (4 * math.pi)
            line_escape = torch.stack([
                (emitters.escape(mu_strip(lines[n:n + 1]), mu_interspace(lines[n:n + 1])).to(f64) @ line_weight[n])
                for n in range(lines.numel())], dim=1)  # [J, N]
        for s, energy in enumerate(energies):
            e = torch.tensor([float(energy)], dtype=f64, device=device)
            reference = _detector_at_pairs(detector, e.expand_as(theta_in), theta_in)  # [A]
            incoming = torch.exp(-mu_strip(e) * strip_in - mu_interspace(e) * interspace_in)
            incoming = incoming * emitters.area / incident[:, 2:3]  # [A, J] interactions per unit length
            # Rayleigh: the outgoing energy is the incident one, one escape matrix for every incident direction
            rayleigh = torch.as_tensor(_scattering_table(composition, density, "rayleigh", float(energy), backend),
                                       device=device)
            weight = (rayleigh[iq1 - 1] + (rayleigh[iq1] - rayleigh[iq1 - 1]) * wq) * W[None, :]
            weight = weight * _detector_at_pairs(detector, e, theta_out)[None, :]  # [A, K]
            escape = emitters.escape(mu_strip(e), mu_interspace(e))  # [J, K]
            signal = (incoming * (weight.to(torch.float32) @ escape.T).to(f64)).sum(-1)  # [A]
            # Compton: the outgoing energy depends on the scattering angle
            compton = torch.as_tensor(_scattering_table(composition, density, "compton", float(energy), backend),
                                      device=device)
            scattered = float(energy) / (1.0 + float(energy) / ELECTRON_REST_KEV * (1.0 - cos_scatter))  # [A, K]
            weight = (compton[iq1 - 1] + (compton[iq1] - compton[iq1 - 1]) * wq) * W[None, :]
            weight = (weight * _detector_at_pairs(detector, scattered, theta_out[None, :])).to(torch.float32)
            strip_mu, interspace_mu = mu_strip(scattered).to(torch.float32), mu_interspace(scattered).to(torch.float32)
            for start in range(0, incident.shape[0], chunk):
                stop = min(start + chunk, incident.shape[0])
                exponent = emitters.out_strip[None] * strip_mu[start:stop, None, :]  # [a, J, K]
                exponent.addcmul_(emitters.out_interspace[None], interspace_mu[start:stop, None, :])
                escape_out = torch.bmm(exponent.neg_().exp_(), weight[start:stop, :, None])[..., 0]  # [a, J]
                signal[start:stop] += (incoming[start:stop] * escape_out.to(f64)).sum(-1)
            if line_energy.size:
                produced = torch.as_tensor(production[s], device=device)  # [N]
                signal = signal + incoming @ (line_escape @ produced)
            total[s] += signal / reference.clamp(min=1e-300)
    return total.reshape(len(energies), len(theta1), len(theta2)).cpu().numpy()


# ---- the table -------------------------------------------------------------------------------------------------

def _interpolate_angles(values: np.ndarray, theta1: np.ndarray, theta2: np.ndarray, to_theta1: np.ndarray,
                        to_theta2: np.ndarray) -> np.ndarray:
    """Interpolate a [S, A1, A2] table bilinearly onto other angle nodes (clamped to the end nodes).

    Args:
        values (np.ndarray): [S, A1, A2] table.
        theta1 (np.ndarray): [A1] ascending nodes.
        theta2 (np.ndarray): [A2] ascending nodes.
        to_theta1 (np.ndarray): [T1] new nodes.
        to_theta2 (np.ndarray): [T2] new nodes.

    Returns:
        np.ndarray: [S, T1, T2] table.
    """
    def weights(nodes, x):
        x = np.clip(x, nodes[0], nodes[-1])
        i1 = np.clip(np.searchsorted(nodes, x, side="right"), 1, len(nodes) - 1)
        w = (x - nodes[i1 - 1]) / (nodes[i1] - nodes[i1 - 1])
        return i1 - 1, i1, np.clip(w, 0.0, 1.0)

    if len(theta1) == 1:
        values, theta1 = np.repeat(values, 2, axis=1), np.array([theta1[0], theta1[0] + 1.0])
    if len(theta2) == 1:
        values, theta2 = np.repeat(values, 2, axis=2), np.array([theta2[0], theta2[0] + 1.0])
    a0, a1, wa = weights(theta1, np.abs(to_theta1))
    b0, b1, wb = weights(theta2, np.abs(to_theta2))
    low = values[:, a0][:, :, b0] * (1 - wb) + values[:, a0][:, :, b1] * wb
    high = values[:, a1][:, :, b0] * (1 - wb) + values[:, a1][:, :, b1] * wb
    return low * (1 - wa)[None, :, None] + high * wa[None, :, None]


def _cache_key(*parts) -> str:
    """SHA-256 of the representation of the inputs of a grid table.

    Args:
        *parts: hashable descriptions and numpy arrays.

    Returns:
        str: hex digest.
    """
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(np.ascontiguousarray(part, dtype=np.float64).tobytes())
        elif isinstance(part, Tensor):
            digest.update(np.ascontiguousarray(part.detach().cpu().numpy(), dtype=np.float64).tobytes())
        else:
            digest.update(repr(part).encode())
        digest.update(b"|")
    return digest.hexdigest()


def grid_signal_transmission(pitch_cm: float, strip_thickness_cm: float, strip_height_cm: float, *,
                             strip_material="lead", interspace_material="air", detector=None,
                             strip_density: Optional[float] = None, interspace_density: Optional[float] = None,
                             energies: np.ndarray, theta1: np.ndarray, theta2: np.ndarray, backend=None,
                             device=None, cache_dir: Optional[str] = "default") -> dict[str, np.ndarray]:
    """Tabulate the signal transmission of a 1-D grid: the direct photons and the grid's scatter and fluorescence.

    Both parts are signals of the detector layer behind the grid per incident photon, relative to the signal of the
    same photon without the grid (docs/source/method.rst, "Grid model"). The direct part is computed at every table
    energy and angle, the secondaries on a coarser set of energies and angles and interpolated. Tables are cached in
    memory and, unless ``cache_dir`` is None, as ``grid_<hash>.npz`` files.

    Args:
        pitch_cm (float): strip pitch p in cm (1 / lines per cm).
        strip_thickness_cm (float): strip thickness t in cm, 0 <= t < p.
        strip_height_cm (float): strip height h in cm, > 0.
        strip_material (str or Material, optional): strip material. Defaults to "lead".
        interspace_material (str or Material, optional): interspace material. Defaults to "air".
        detector (DetectorResponse, optional): the layer that scores the signal. Defaults to None (0.6 mm of CsI with
            the K x-ray escape, ``detector_response(deposition="fluorescence_escape")``).
        strip_density (float, optional): strip density in g/cm^3. Defaults to None (the nominal density).
        interspace_density (float, optional): interspace density in g/cm^3. Defaults to None (the nominal density).
        energies (np.ndarray): [E] table energies in keV.
        theta1 (np.ndarray): [T1] table angles across the strips in rad (from the focusing direction).
        theta2 (np.ndarray): [T2] table angles along the strips in rad.
        backend (photon_data.XcomXraylibBackend, optional): photon data. Defaults to None (loaded on first use).
        device (torch.device or str, optional): device of the computation. Defaults to None (CUDA when available).
        cache_dir (str, optional): directory of the disk cache; "default" is ``photon_data.default_cache_dir()``,
            None turns the disk cache off. Defaults to "default".

    Returns:
        dict[str, np.ndarray]: "direct" and "secondary", [E, T1, T2] each.
    """
    from .detector import detector_response, material_entry
    from .photon_data import default_cache_dir, resolve_backend

    p, t, h = float(pitch_cm), float(strip_thickness_cm), float(strip_height_cm)
    if not (0.0 <= t < p) or not h > 0:
        raise ValueError(f"need 0 <= strip thickness < pitch ({p:g} cm) and a positive height, got t = {t:g} cm, "
                         f"h = {h:g} cm")
    backend = resolve_backend() if backend is None else backend
    detector = detector_response(deposition="fluorescence_escape", backend=backend) if detector is None else detector
    strip = material_entry(strip_material, strip_density, backend)
    interspace = material_entry(interspace_material, interspace_density, backend)
    energies = np.asarray(energies, dtype=np.float64).reshape(-1)
    theta1 = np.asarray(theta1, dtype=np.float64).reshape(-1)
    theta2 = np.asarray(theta2, dtype=np.float64).reshape(-1)
    key = _cache_key(_MODEL_VERSION, p, t, h, strip, interspace, detector.energy_kev, detector.theta_rad,
                     detector.response, energies, theta1, theta2, backend.name, getattr(backend, "xcom_sha256", ""))
    if key in _MEMORY_CACHE:
        return {name: value.copy() for name, value in _MEMORY_CACHE[key].items()}
    path = None
    if cache_dir is not None:
        path = os.path.join(default_cache_dir() if cache_dir == "default" else cache_dir, f"grid_{key[:32]}.npz")
        if os.path.exists(path):
            try:
                with np.load(path) as stored:
                    result = {"direct": stored["direct"], "secondary": stored["secondary"]}
                _MEMORY_CACHE[key] = result
                return {name: value.copy() for name, value in result.items()}
            except (OSError, KeyError, ValueError):
                pass  # unreadable: computed again below
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    with torch.no_grad():
        mu_strip = _Attenuation(strip[2], strip[1], energies.max(), backend, device)
        mu_interspace = _Attenuation(interspace[2], interspace[1], energies.max(), backend, device)
        e = torch.as_tensor(energies, device=device)
        direct = _direct(p, t, h, mu_strip(e), mu_interspace(e), torch.as_tensor(theta1, device=device),
                         torch.as_tensor(theta2, device=device)).cpu().numpy()
        edges = [np.asarray(backend.element(Z).edge_energy_ev, dtype=np.float64) * 1.0e-3
                 for Z in list(strip[2]) + (list(interspace[2]) if interspace[1] >= _GAS_DENSITY else [])
                 if Z >= _MIN_FLUORESCENCE_Z]
        edges = np.concatenate(edges) if edges else np.zeros(0)
        secondary_energy = _secondary_energies(energies, edges[edges > 5.0])
        nodes1, nodes2 = _secondary_angles(math.atan2(p - t, h))
        coarse = _secondaries(p, t, h, strip[1:], interspace[1:], detector, secondary_energy, nodes1, nodes2,
                              backend, device)  # [S, A1, A2]
    on_angles = _interpolate_angles(coarse, nodes1, nodes2, theta1, theta2)  # [S, T1, T2]
    index = np.clip(np.searchsorted(secondary_energy, energies, side="right"), 1, max(1, len(secondary_energy) - 1))
    if len(secondary_energy) == 1:
        secondary = np.repeat(on_angles, len(energies), axis=0)
    else:
        e0, e1 = secondary_energy[index - 1], secondary_energy[index]
        w = np.clip((energies - e0) / (e1 - e0), 0.0, 1.0)[:, None, None]
        secondary = on_angles[index - 1] * (1 - w) + on_angles[index] * w
    result = {"direct": direct, "secondary": np.clip(secondary, 0.0, None)}
    _MEMORY_CACHE[key] = result
    if path is not None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            temporary = f"{path}.{os.getpid()}.tmp.npz"
            np.savez(temporary, **result)
            os.replace(temporary, path)
        except OSError:
            pass  # the disk cache is optional
    return {name: value.copy() for name, value in result.items()}
