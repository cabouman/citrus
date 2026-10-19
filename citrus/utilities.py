"""Common utilities of CITRUS.

The module holds the WLAB phantom (a water prism with lung, air and bone inserts) with its grid and scan geometries,
the interpolating attenuation <-> density model between the library's reference materials, the object size and mean
composition that set the energy groups, view angles, synthetic cone-beam data with scatter, a GPU memory cap and small
plotting helpers.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from .scatter_solver.config import SolverConfig
from .scatter_solver.cross_sections import MATERIALS, MultigroupLibrary, material_index
from .scatter_solver.detector import DetectorResponse, GridResponse, ideal_counting_response, no_grid
from .scatter_solver.source import SourceModel
from .scatter_solver.geometry import ConeGeometry, Grid, ParallelGeometry
from .scatter_solver.scatter_solver import ScatterSolver

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
    "parallel_wlab_geometry",
    "wlab_setup",
    "effective_energy",
    "mass_attenuation_at",
    "nominal_densities",
    "nominal_attenuation",
    "material_order",
    "densities_to_attenuation",
    "attenuation_to_densities",
    "attenuation_to_density",
    "wlab_solver",
    "air_water_threshold",
    "object_mask",
    "escape_probability",
    "object_size_cm",
    "grouping_medium_estimate",
    "fan_angle",
    "view_angles",
    "helical_scan",
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
    """Test the half-open membership ``center - width/2 <= c < center + width/2``.

    Args:
        c (Tensor): coordinates in cm.
        center (float): center of the interval in cm.
        width (float): width of the interval in cm.

    Returns:
        Tensor: boolean mask of the shape of ``c``.
    """
    return (c >= center - width / 2) & (c < center + width / 2)


def wlab_phantom(grid: Grid, *, lung_center: tuple[float, float] = (-5.0, 0.0),
                 air_center: tuple[float, float] = (3.0, 0.0), bone_center: tuple[float, float] = (7.0, 0.0),
                 materials: Optional[Sequence[str]] = None, device=None, dtype=torch.float32) -> Tensor:
    """Rasterize the WLAB phantom on a grid by voxel-center membership (half-open intervals).

    The "water" row holds the water prism with the lung and air inserts as water of reduced density, the
    "bone_cortical_icrp" row the
    bone insert; every other row is zero.

    Args:
        grid (Grid): grid that covers the 30 x 20 x 40 cm prism centered at the isocenter.
        lung_center (tuple[float, float]): (x, y) center of the lung insert in cm. Defaults to (-5.0, 0.0).
        air_center (tuple[float, float]): (x, y) center of the air insert in cm. Defaults to (3.0, 0.0).
        bone_center (tuple[float, float]): (x, y) center of the bone insert in cm. Defaults to (7.0, 0.0).
        materials (Sequence[str], optional): material names of the library, which must include water and bone.
            Defaults to None, the seven ``cross_sections.MATERIALS``.
        device (torch.device, optional): device of the result. Defaults to None.
        dtype (torch.dtype): dtype of the result. Defaults to torch.float32.

    Returns:
        Tensor: [M, rows, cols, slices] densities rho in g/cm^3, M = len(materials).

    Raises:
        ValueError: if water or bone is missing, the grid does not cover the prism, or the inserts leave it or overlap.
    """
    names = MATERIALS if materials is None else tuple(materials)
    water_row = _phantom_row(names, "water")
    bone_row = _phantom_row(names, "bone_cortical_icrp")
    size_x, size_y, size_z = WLAB_SIZE_XYZ
    extent_x, extent_y, extent_z = grid.extent
    lower = grid.lower_corner()
    tolerance = 1e-9
    if (lower[0] > -size_x / 2 + tolerance or lower[0] + extent_x < size_x / 2 - tolerance
            or lower[1] > -size_y / 2 + tolerance or lower[1] + extent_y < size_y / 2 - tolerance
            or lower[2] > -size_z / 2 + tolerance or lower[2] + extent_z < size_z / 2 - tolerance):
        raise ValueError(f"grid box {lower} + {grid.extent} does not cover the WLAB prism {WLAB_SIZE_XYZ}")
    inserts = [(lung_center, LUNG_SIZE_XY), (air_center, AIR_SIZE_XY), (bone_center, BONE_SIZE_XY)]
    for i, (center_i, size_i) in enumerate(inserts):
        if (abs(center_i[0]) + size_i[0] / 2 > size_x / 2 + tolerance
                or abs(center_i[1]) + size_i[1] / 2 > size_y / 2 + tolerance):
            raise ValueError(f"insert {i} at {center_i} leaves the water prism")
        for center_j, size_j in inserts[i + 1:]:
            if (abs(center_i[0] - center_j[0]) < (size_i[0] + size_j[0]) / 2 - tolerance
                    and abs(center_i[1] - center_j[1]) < (size_i[1] + size_j[1]) / 2 - tolerance):
                raise ValueError(f"inserts at {center_i} and {center_j} overlap")

    phantom = WlabPhantom(grid, names, lung_center=lung_center, air_center=air_center, bone_center=bone_center,
                          device=device, dtype=dtype)
    rho = torch.zeros(len(names), *grid.shape, device=device, dtype=dtype)
    rho[water_row] = phantom.material(None, water_row)
    rho[bone_row] = phantom.material(None, bone_row)
    return rho


@dataclass(frozen=True)
class WlabPhantom:
    """The WLAB phantom as a density model that rasterizes one material map at a time.

    ``material(X, m)`` gives one map (X ignored), so a fine grid never holds the [M, R, C, S] volume; ``wlab_phantom``
    stacks the maps for small grids.

    Attributes:
        grid (Grid): grid that covers the 30 x 20 x 40 cm prism.
        materials (tuple[str, ...]): material names of the library, which must include water and bone. Defaults to
            ``cross_sections.MATERIALS``.
        lung_center (tuple[float, float]): (x, y) center of the lung insert in cm. Defaults to (-5.0, 0.0).
        air_center (tuple[float, float]): (x, y) center of the air insert in cm. Defaults to (3.0, 0.0).
        bone_center (tuple[float, float]): (x, y) center of the bone insert in cm. Defaults to (7.0, 0.0).
        device (object): device of the maps. Defaults to None.
        dtype (torch.dtype): dtype of the maps. Defaults to torch.float32.
    """

    grid: Grid
    materials: tuple[str, ...] = MATERIALS
    lung_center: tuple[float, float] = (-5.0, 0.0)
    air_center: tuple[float, float] = (3.0, 0.0)
    bone_center: tuple[float, float] = (7.0, 0.0)
    device: object = None
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        """Store the materials as a tuple and check that water and bone are among them."""
        object.__setattr__(self, "materials", tuple(self.materials))
        _phantom_row(self.materials, "water")
        _phantom_row(self.materials, "bone_cortical_icrp")

    @property
    def num_materials(self) -> int:
        """Number of materials M of the library."""
        return len(self.materials)

    def _masks(self):
        """Boolean masks of the prism and of the three inserts.

        Returns:
            (Tensor, Tensor, Tensor, Tensor): (prism, lung, air, bone) boolean masks [R, C, S].
        """
        size_x, size_y, size_z = WLAB_SIZE_XYZ
        x, y, z = self.grid.axis_coordinates(self.device, torch.float64)
        xx, yy, zz = x[None, :, None], y[:, None, None], z[None, None, :]
        prism = _in_interval(xx, 0.0, size_x) & _in_interval(yy, 0.0, size_y) & _in_interval(zz, 0.0, size_z)

        def column(center: tuple[float, float], size: tuple[float, float]) -> Tensor:
            return _in_interval(xx, center[0], size[0]) & _in_interval(yy, center[1], size[1]) & prism

        return prism, column(self.lung_center, LUNG_SIZE_XY), column(self.air_center, AIR_SIZE_XY), \
            column(self.bone_center, BONE_SIZE_XY)

    def material(self, X, m: int) -> Tensor:
        """Density map of one library material.

        The water row holds the prism with the lung and air inserts as water of reduced density, the bone row the bone
        insert; every other material is zero.

        Args:
            X: ignored (a phantom has no reconstruction).
            m (int): library index of the material.

        Returns:
            Tensor: [R, C, S] density in g/cm^3, in ``dtype``.
        """
        rho = torch.zeros(*self.grid.shape, device=self.device, dtype=self.dtype)
        water_row = _phantom_row(self.materials, "water")
        bone_row = _phantom_row(self.materials, "bone_cortical_icrp")
        if m not in (water_row, bone_row):
            return rho
        prism, lung, air, bone = self._masks()
        if m == water_row:
            rho[prism] = RHO_WATER
            rho[lung] = RHO_LUNG
            rho[air] = RHO_AIR
            rho[bone] = 0.0
        else:
            rho[bone] = RHO_BONE
        return rho

    def attenuation(self, library: MultigroupLibrary, energy_kev: Optional[float] = None) -> Tensor:
        """Attenuation ``X_true = sum_m rho_m m_m(E)`` of the phantom, accumulated one material map at a time.

        Args:
            library (MultigroupLibrary): library with the phantom's materials.
            energy_kev (float, optional): energy E in keV. Defaults to None, the effective energy E_eff.

        Returns:
            Tensor: [R, C, S] attenuation in 1/cm.

        Raises:
            ValueError: if the library's materials are not the phantom's.
        """
        if tuple(library.materials) != self.materials:
            raise ValueError(f"the phantom's materials {self.materials} are not the library's {library.materials}")
        m_at = mass_attenuation_at(library, effective_energy(library) if energy_kev is None else float(energy_kev))
        X = torch.zeros(*self.grid.shape, device=self.device, dtype=self.dtype)
        for m in (_phantom_row(self.materials, "water"), _phantom_row(self.materials, "bone_cortical_icrp")):
            X.add_(self.material(None, m), alpha=float(m_at[m]))
        return X

    def densities(self) -> Tensor:
        """Stack the maps of all materials, as ``wlab_phantom`` (for grids small enough to hold the volume).

        Returns:
            Tensor: [M, R, C, S] densities in g/cm^3.
        """
        return torch.stack([self.material(None, m) for m in range(self.num_materials)])


class StackedDensities:
    """A density volume as a density model whose ``material(X, m)`` is ``rho[m]`` (X ignored).

    It lets ``synthesize_cone_beam_data`` take an explicit volume through the same one-map-at-a-time path as a phantom.

    Attributes:
        rho (Tensor): [M, R, C, S] densities in g/cm^3.
    """

    def __init__(self, rho: Tensor) -> None:
        """Wrap a density volume.

        Args:
            rho (Tensor): [M, R, C, S] densities in g/cm^3.
        """
        self.rho = rho

    @property
    def num_materials(self) -> int:
        """Number of materials M of the volume."""
        return int(self.rho.shape[0])

    def material(self, X, m: int) -> Tensor:
        """Density map of one material.

        Args:
            X: ignored.
            m (int): material index.

        Returns:
            Tensor: [R, C, S] density ``rho[m]`` in g/cm^3.
        """
        return self.rho[m]


def _phantom_row(materials: Sequence[str], name: str) -> int:
    """Row of a material among the materials of the WLAB phantom.

    Args:
        materials (Sequence[str]): material names of the library.
        name (str): name of the material.

    Returns:
        int: index of ``name`` in ``materials``.

    Raises:
        ValueError: if the material is absent, with a message that says what the phantom needs.
    """
    try:
        return material_index(materials, name)
    except ValueError:
        raise ValueError(f"the WLAB phantom needs 'water' and 'bone_cortical_icrp' among the materials, got "
                         f"{tuple(materials)}; "
                         f"add them to SolverConfig.materials") from None


def wlab_grid(delta_voxel: float) -> Grid:
    """Grid that exactly tiles the 30 x 20 x 40 cm WLAB box (cols x rows x slices).

    Args:
        delta_voxel (float): voxel pitch in cm; it must divide every extent of the box.

    Returns:
        Grid: grid of [20, 30, 40] / ``delta_voxel`` voxels (rows, cols, slices).
    """
    size_x, size_y, size_z = WLAB_SIZE_XYZ
    n = []
    for length in (size_y, size_x, size_z):
        k = round(length / delta_voxel)
        if k < 1 or abs(k * delta_voxel - length) > 1e-9 * max(1.0, length):
            raise ValueError(f"delta_voxel={delta_voxel} does not divide the WLAB extent {length} cm")
        n.append(int(k))
    return Grid(shape=(n[0], n[1], n[2]), delta_voxel=float(delta_voxel))


def wlab_geometry() -> ConeGeometry:
    """Full-fan kV cone-beam geometry of the WLAB setup.

    The source-axis distance is 100 cm and the source-detector distance 150 cm; the centered 40 x 30 cm^2 panel has
    100 x 75 pixels of 4 mm, so the prism's lateral and z ends fall outside the field of view.

    Returns:
        ConeGeometry: the scan geometry.
    """
    return ConeGeometry(source_iso_dist=100.0, source_detector_dist=150.0, num_det_rows=75, num_det_channels=100,
                        delta_det_row=0.4, delta_det_channel=0.4)


def full_field_wlab_geometry() -> ConeGeometry:
    """``wlab_geometry`` with a 150 x 184 panel (60 x 73.6 cm) on which the whole prism projects at every view angle.

    Returns:
        ConeGeometry: the scan geometry of the reconstructions.
    """
    return replace(wlab_geometry(), num_det_channels=150, num_det_rows=184)


def parallel_wlab_geometry(delta_voxel: float, axis_detector_dist: float = 50.0) -> ParallelGeometry:
    """Parallel-beam geometry of the WLAB prism on ``wlab_grid(delta_voxel)``, whose whole prism it sees at every view.

    The detector rows are the slices of the grid (``ParallelGeometry.from_grid``) and the channels, of the voxel
    pitch, span at least 38 cm, more than the 36.06 cm transaxial diagonal of the prism.

    Args:
        delta_voxel (float): voxel pitch in cm of the grid; it must divide every extent of the box.
        axis_detector_dist (float, optional): distance from the rotation axis to the panel in cm. Defaults to 50.0, the
            axis-detector distance of ``wlab_geometry``.

    Returns:
        ParallelGeometry: the scan geometry.
    """
    channels = int(math.ceil(38.0 / float(delta_voxel) - 1e-9))
    return ParallelGeometry.from_grid(wlab_grid(delta_voxel), axis_detector_dist, channels, float(delta_voxel))


def wlab_setup(delta_voxel: float = 1.0, device=None, dtype=torch.float32,
               materials: Optional[Sequence[str]] = None) -> tuple[Grid, ConeGeometry, Tensor]:
    """Grid, geometry and densities of the WLAB scatter study.

    Args:
        delta_voxel (float): voxel pitch in cm. Defaults to 1.0.
        device (torch.device, optional): device of the densities. Defaults to None.
        dtype (torch.dtype): dtype of the densities. Defaults to torch.float32.
        materials (Sequence[str], optional): material names of the library. Defaults to None, the seven
            ``cross_sections.MATERIALS``.

    Returns:
        (Grid, ConeGeometry, Tensor): ``wlab_grid(delta_voxel)``, ``wlab_geometry()`` and the densities
        rho [M, R, C, S] in g/cm^3 of ``wlab_phantom``.
    """
    grid = wlab_grid(delta_voxel)
    return grid, wlab_geometry(), wlab_phantom(grid, materials=materials, device=device, dtype=dtype)


# =====================================================================================================
# attenuation <-> density
# =====================================================================================================

def effective_energy(library: MultigroupLibrary) -> float:
    """Spectrum-mean photon energy ``E_eff = sum_e w[e] E[e] / sum_e w[e]`` over the library's fine energies E.

    Args:
        library (MultigroupLibrary): the library, with fine weights w.

    Returns:
        float: E_eff in keV.
    """
    w = library.fine_weight.to(torch.float64)
    return float((w * library.fine_energy.to(torch.float64)).sum() / w.sum())


def mass_attenuation_at(library: MultigroupLibrary, energy_kev: float) -> Tensor:
    """Total mass attenuation coefficients of the library's materials at one energy.

    The coefficients are interpolated linearly in (log E, log mu) in ``library.fine_mu_t`` on the sorted fine energies;
    an energy that is a fine energy (the only one of a monoenergetic library) gives its tabulated coefficients.

    Args:
        library (MultigroupLibrary): the library.
        energy_kev (float): energy in keV.

    Returns:
        Tensor: [M] mass attenuation coefficients in cm^2/g, float64 on the CPU.

    Raises:
        ValueError: if the energy is outside the library's fine-energy range.
    """
    E = library.fine_energy.detach().to("cpu", torch.float64)
    mu = library.fine_mu_t.detach().to("cpu", torch.float64)
    order = torch.argsort(E)
    E, mu = E[order], mu[:, order]
    e = float(energy_kev)
    if not (float(E[0]) <= e <= float(E[-1])):
        raise ValueError(f"energy {e} keV outside the library's fine-energy range [{float(E[0])}, {float(E[-1])}]")
    i = int(torch.searchsorted(E, torch.tensor([e], dtype=torch.float64), right=True)[0]) - 1
    if float(E[max(i, 0)]) == e:
        return mu[:, max(i, 0)].clone()
    i = min(max(i, 0), E.numel() - 2)
    le0, le1 = torch.log(E[i]), torch.log(E[i + 1])
    t = (torch.log(torch.tensor(e, dtype=torch.float64)) - le0) / (le1 - le0)
    return torch.exp(torch.log(mu[:, i]) + t * (torch.log(mu[:, i + 1]) - torch.log(mu[:, i])))


def nominal_densities(library: MultigroupLibrary) -> Tensor:
    """Nominal densities of the library's materials (``library.nominal_density``).

    Args:
        library (MultigroupLibrary): the library.

    Returns:
        Tensor: [M] nominal densities in g/cm^3, float64 on the CPU.
    """
    return library.nominal_density.detach().to("cpu", torch.float64)


def nominal_attenuation(library: MultigroupLibrary, energy_kev: Optional[float] = None) -> Tensor:
    """Attenuation of each material of the library at its nominal density.

    Args:
        library (MultigroupLibrary): the library.
        energy_kev (float, optional): energy in keV. Defaults to None, the effective energy E_eff.

    Returns:
        Tensor: [M] nominal attenuations in 1/cm, float64 on the CPU.
    """
    e = effective_energy(library) if energy_kev is None else float(energy_kev)
    return nominal_densities(library) * mass_attenuation_at(library, e)


def material_order(library: MultigroupLibrary, energy_kev: Optional[float] = None) -> list[int]:
    """Bracketing order of the density model: the material indices sorted by nominal attenuation at one energy.

    At E_eff this is the library order ``[0, 1, ..., M-1]`` of a library built by
    ``cross_section_library.build_library``; materials may cross in attenuation at other energies.

    Args:
        library (MultigroupLibrary): the library.
        energy_kev (float, optional): energy in keV. Defaults to None, the effective energy E_eff.

    Returns:
        list[int]: [M] material indices in increasing nominal attenuation.

    Raises:
        ValueError: if two materials have nearly the same nominal attenuation at that energy
            (``cross_section_library.check_material_order``).
    """
    from .scatter_solver.cross_section_library import check_material_order

    attenuation = nominal_attenuation(library, energy_kev)
    if attenuation.numel() > 1:
        check_material_order(attenuation.numpy(), library.materials)
    return torch.argsort(attenuation).tolist()


def _mass_attenuation_coefficients(library: MultigroupLibrary, energy_kev: Optional[float], like: Tensor) -> Tensor:
    """Mass attenuation coefficients at an energy, on the device and in the dtype of a tensor.

    Args:
        library (MultigroupLibrary): the library.
        energy_kev (float, optional): energy in keV; None for the effective energy E_eff.
        like (Tensor): tensor whose device and dtype the result takes.

    Returns:
        Tensor: [M] mass attenuation coefficients in cm^2/g.
    """
    e = effective_energy(library) if energy_kev is None else float(energy_kev)
    return mass_attenuation_at(library, e).to(device=like.device, dtype=like.dtype)


def densities_to_attenuation(library: MultigroupLibrary, rho: Tensor, energy_kev: Optional[float] = None) -> Tensor:
    """Attenuation ``X = sum_m rho[m] m_m(E)`` of a density volume.

    Args:
        library (MultigroupLibrary): library of the M materials.
        rho (Tensor): [M, ...] densities in g/cm^3, rows in the order of ``library.materials``.
        energy_kev (float, optional): energy E in keV. Defaults to None, the effective energy E_eff.

    Returns:
        Tensor: [...] attenuation in 1/cm, on the device and in the dtype of ``rho``.
    """
    M = library.num_materials
    if rho.shape[0] != M:
        raise ValueError(f"rho must be [{M}, ...] for the library's materials {tuple(library.materials)}, got "
                         f"{tuple(rho.shape)}")
    m = _mass_attenuation_coefficients(library, energy_kev, rho)
    X = rho[0] * m[0]
    for k in range(1, M):
        X = X + rho[k] * m[k]
    return X


def attenuation_to_densities(library: MultigroupLibrary, X: Tensor, energy_kev: Optional[float] = None) -> Tensor:
    """Interpolating density model: densities of the reference materials that reproduce an attenuation volume.

    The two reference materials whose nominal attenuations bracket X get ``(1 - f) rho_a^nom`` and ``f rho_b^nom`` with
    ``f = (X - mu_a) / (mu_b - mu_a)``; below the lowest (above the highest) nominal attenuation that material alone
    gets the density ``X / m``, and ``X < 0`` gives 0.

    Args:
        library (MultigroupLibrary): library of the M reference materials (any set of at least two).
        X (Tensor): [...] attenuation in 1/cm.
        energy_kev (float, optional): energy in keV. Defaults to None, the effective energy E_eff.

    Returns:
        Tensor: [M, ...] densities in g/cm^3 (at most two nonzero per voxel), on the device and in the dtype of ``X``.
    """
    M = library.num_materials
    order = material_order(library, energy_kev)
    num_dims = X.dim()
    mass_attenuation = _mass_attenuation_coefficients(library, energy_kev, X).reshape((M,) + (1,) * num_dims)
    slot_order = torch.as_tensor(order, device=X.device, dtype=torch.long)
    # material slots in bracketing order
    mass_attenuation = mass_attenuation[slot_order].expand((M,) + tuple(X.shape))
    nominal = (nominal_densities(library).to(device=X.device, dtype=X.dtype)[slot_order]
               .reshape((M,) + (1,) * num_dims))
    mu = nominal * mass_attenuation  # nominal attenuation per slot [M, ...]
    X_clamped = X.clamp_min(0)
    j = (mu <= X_clamped[None]).sum(0)  # number of nominal attenuations <= X: bracket [j - 1, j]
    a = (j - 1).clamp(0, M - 1)
    b = j.clamp(max=M - 1)
    inside = (j >= 1) & (j <= M - 1)
    mu_a = torch.gather(mu, 0, a[None])[0]
    mu_b = torch.gather(mu, 0, b[None])[0]
    m_a = torch.gather(mass_attenuation, 0, a[None])[0]
    nominal_a = nominal.expand_as(mu).gather(0, a[None])[0]
    nominal_b = nominal.expand_as(mu).gather(0, b[None])[0]
    span = torch.where(inside, mu_b - mu_a, torch.ones_like(mu_a))
    f = torch.where(inside, (X_clamped - mu_a) / span, torch.zeros_like(X_clamped))
    # outside the bracket range: one material (slot a = first or last) of density X / m
    rho_a = torch.where(inside, (1.0 - f) * nominal_a, X_clamped / m_a)
    rho_b = torch.where(inside, f * nominal_b, torch.zeros_like(X_clamped))
    rho = torch.zeros((M,) + tuple(X.shape), device=X.device, dtype=X.dtype)
    rho.scatter_add_(0, slot_order[a][None], rho_a[None])
    rho.scatter_add_(0, slot_order[b][None], rho_b[None])
    return rho


def attenuation_to_density(library: MultigroupLibrary, X: Tensor, material: int,
                           energy_kev: Optional[float] = None) -> Tensor:
    """One material's density map of the interpolating model, without forming the maps of the other materials.

    The map equals ``attenuation_to_densities(library, X, energy_kev)[material]`` bit for bit; its only temporaries of
    the shape of X are the result, one interpolation weight and two boolean masks.

    Args:
        library (MultigroupLibrary): library of the M reference materials.
        X (Tensor): [...] attenuation in 1/cm.
        material (int): library index of the material.
        energy_kev (float, optional): energy in keV. Defaults to None, the effective energy E_eff.

    Returns:
        Tensor: [...] density of the material in g/cm^3, on the device and in the dtype of ``X``.

    Raises:
        ValueError: if ``material`` is not an index of the library's materials.
    """
    M = library.num_materials
    material_index = int(material)
    if not 0 <= material_index < M:
        raise ValueError(f"material {material} outside the library's {M} materials {tuple(library.materials)}")
    order = material_order(library, energy_kev)
    slot = order.index(material_index)
    slot_order = torch.as_tensor(order, device=X.device, dtype=torch.long)
    # [M] mass attenuations in bracketing order
    mass_attenuation = _mass_attenuation_coefficients(library, energy_kev, X)[slot_order]
    nominal = nominal_densities(library).to(device=X.device, dtype=X.dtype)[slot_order]
    mu = nominal * mass_attenuation  # nominal attenuations in bracketing order (ascending, positive)
    out: Optional[Tensor] = None

    def put(f: Tensor, inside: Tensor) -> None:  # out += f on `inside` (disjoint regions: x + 0 = x exactly)
        nonlocal out
        f.masked_fill_(~inside, 0.0)
        if out is None:
            out = f
        else:
            out.add_(f)

    if slot >= 1:  # upper end of the bracket [s-1, s]: f rho^nom
        f = (X - mu[slot - 1]).div_(mu[slot] - mu[slot - 1]).mul_(nominal[slot])
        put(f, (X >= mu[slot - 1]).logical_and_(X < mu[slot]))
        del f
    if slot <= M - 2:  # lower end of the bracket [s, s+1]: (1 - f) rho^nom
        f = (X - mu[slot]).div_(mu[slot + 1] - mu[slot]).neg_().add_(1.0).mul_(nominal[slot])
        put(f, (X >= mu[slot]).logical_and_(X < mu[slot + 1]))
        del f
    if slot == 0:  # below the first nominal attenuation: the first material alone (X < 0 -> 0)
        put(X.clamp_min(0).div_(mass_attenuation[0]), X < mu[0])
    if slot == M - 1:  # from the last nominal attenuation on: the last material alone
        put(X / mass_attenuation[M - 1], X >= mu[M - 1])
    return out


# =====================================================================================================
# synthetic cone-beam data
# =====================================================================================================

def wlab_solver(delta_voxel: float, *, config: Optional[SolverConfig] = None,
                library: Optional[MultigroupLibrary] = None,
                geometry: Optional[ConeGeometry] = None, device=None,
                cache_dir: Optional[str] = "default", spectrum=None,
                detector: Optional[DetectorResponse] = None, grid_response: Optional[GridResponse] = None,
                source: Optional[SourceModel] = None) -> ScatterSolver:
    """Scatter solver for the WLAB data, by default with an ideal counting detector, no grid and float32.

    Without ``library`` the solver is lazy: its library (for ``config.materials`` and ``spectrum``) is built on first
    use, so ``citrus_recon`` and ``synthesize_cone_beam_data`` can still set ``config.grouping_object_size_cm`` first.

    Args:
        delta_voxel (float): voxel size in cm of the fine grid the solver will be used with; the solver ignores it.
        config (SolverConfig, optional): solver configuration. Defaults to None, ``SolverConfig()``.
        library (MultigroupLibrary, optional): prebuilt library. Defaults to None, a lazy library.
        geometry (ConeGeometry or ParallelGeometry, optional): scan geometry. Defaults to None,
            ``full_field_wlab_geometry()``.
        device (torch.device, optional): device. Defaults to None, CUDA when available, else the CPU.
        cache_dir (str, optional): disk cache of a lazy library; "default" is ``$CITRUS_CACHE_DIR`` or
            ``~/.cache/citrus``, None no disk cache. Defaults to "default".
        spectrum (tuple[np.ndarray, np.ndarray], optional): ([K] energies in keV, [K] photons per keV) of the source,
            required for a lazy library. Defaults to None.
        detector (DetectorResponse, optional): detector response. Defaults to None (``ideal_counting_response()``).
        grid_response (GridResponse, optional): anti-scatter-grid response. Defaults to None (``no_grid()``).
        source (SourceModel, optional): source model. Defaults to None (``SourceModel()``, uniform emission).

    Returns:
        ScatterSolver: the solver.
    """
    config = config if config is not None else SolverConfig()
    if library is None and spectrum is None:
        raise ValueError("wlab_solver without a library needs the source spectrum (spectrum=...)")
    device = (torch.device(device) if device is not None
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    solver = ScatterSolver(library, config, geometry if geometry is not None else full_field_wlab_geometry(),
                           detector=ideal_counting_response() if detector is None else detector,
                           grid_response=no_grid() if grid_response is None else grid_response, source=source,
                           device=device, dtype=torch.float32, spectrum=spectrum)
    solver.library_cache_dir = cache_dir
    return solver


# =====================================================================================================
# object size (the grouping rule "scattered_fluence")
# =====================================================================================================

def air_water_threshold(reference: Union[MultigroupLibrary, ScatterSolver],
                        energy_kev: Optional[float] = None) -> float:
    """Attenuation midway between the nominal attenuations of air and water at E_eff, the threshold of ``object_mask``.

    A library without an air row counts air as 0 and takes water from the catalogue if it has no water row; a lazy
    solver takes both from the photon data (``cross_section_library.mass_attenuation_preview``) without a build.

    Args:
        reference (MultigroupLibrary or ScatterSolver): library, or solver whose library (built or not) is used.
        energy_kev (float, optional): energy in keV. Defaults to None, the effective energy E_eff.

    Returns:
        float: threshold in 1/cm.
    """
    from .scatter_solver.cross_sections import MATERIAL_CATALOGUE

    if isinstance(reference, ScatterSolver):
        if not reference.library_built:
            from .scatter_solver.cross_section_library import mass_attenuation_preview

            _, mass = mass_attenuation_preview(reference.config, ("air", "water"), reference.spectrum,
                                               energy_kev=energy_kev)
            return 0.5 * float(MATERIAL_CATALOGUE["air"][0] * mass[0] + MATERIAL_CATALOGUE["water"][0] * mass[1])
        reference = reference.library
    library = reference
    energy = effective_energy(library) if energy_kev is None else float(energy_kev)
    attenuation = nominal_attenuation(library, energy)
    names = tuple(library.materials)
    air = float(attenuation[names.index("air")]) if "air" in names else 0.0
    if "water" in names:
        water = float(attenuation[names.index("water")])
    else:
        from .scatter_solver.cross_section_library import mass_attenuation_preview

        _, mass = mass_attenuation_preview(SolverConfig(), ("water",), energy_kev=energy)
        water = float(MATERIAL_CATALOGUE["water"][0] * mass[0])
    return 0.5 * (air + water)


def object_mask(X: Tensor, reference: Union[MultigroupLibrary, ScatterSolver, float], *,
                energy_kev: Optional[float] = None) -> Tensor:
    """Object voxels of an attenuation volume: ``max(X, 0)`` above the attenuation midway between air and water.

    Args:
        X (Tensor): [R, C, S] attenuation in 1/cm.
        reference (MultigroupLibrary or ScatterSolver or float): library or solver for ``air_water_threshold``, or
            the threshold itself in 1/cm.
        energy_kev (float, optional): energy in keV of the threshold. Defaults to None, the effective energy E_eff.

    Returns:
        Tensor: [R, C, S] boolean object mask.
    """
    threshold = float(reference) if isinstance(reference, (int, float)) else air_water_threshold(reference, energy_kev)
    return X.clamp_min(0) > threshold


def _reference_attenuation(reference: Union[MultigroupLibrary, ScatterSolver],
                           energy_kev: Optional[float]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, float]:
    """Nominal densities and mass attenuations of the reference materials of a library or a solver.

    A lazy solver takes the nominal densities of its materials (catalogue or ``Material`` definitions) and
    ``cross_section_library.mass_attenuation_preview``, without a build.

    Args:
        reference (MultigroupLibrary or ScatterSolver): library or solver.
        energy_kev (float, optional): energy in keV; None for the effective energy E_eff.

    Returns:
        (names, nominal, mass, energy): the material names in library order, the [M] nominal densities in g/cm^3,
        the [M] mass attenuations in cm^2/g at the energy and the energy in keV.
    """
    from .scatter_solver.cross_sections import material_definition

    if isinstance(reference, ScatterSolver) and not reference.library_built:
        from .scatter_solver.cross_section_library import mass_attenuation_preview

        names = reference.materials
        energy, mass = mass_attenuation_preview(reference.config, names, reference.spectrum, energy_kev=energy_kev)
        nominal = np.array([material_definition(n, reference.config.materials).density for n in names],
                           dtype=np.float64)
        return names, nominal, np.asarray(mass, dtype=np.float64), float(energy)
    library = reference.library if isinstance(reference, ScatterSolver) else reference
    energy = effective_energy(library) if energy_kev is None else float(energy_kev)
    return (tuple(library.materials), nominal_densities(library).numpy(),
            mass_attenuation_at(library, energy).numpy(), energy)


def grouping_medium_estimate(X: Tensor, reference: Union[MultigroupLibrary, ScatterSolver], *,
                             energy_kev: Optional[float] = None,
                             mask: Optional[Tensor] = None) -> Optional[tuple[tuple[str, float], ...]]:
    """Mean composition of the object for ``SolverConfig.grouping_medium``.

    The interpolating density model is applied to the object voxels of X and averaged over them per reference
    material; a lazy solver's library is not built.

    Args:
        X (Tensor): [R, C, S] attenuation in 1/cm.
        reference (MultigroupLibrary or ScatterSolver): library or solver of the reference materials.
        energy_kev (float, optional): energy in keV. Defaults to None, the effective energy E_eff.
        mask (Tensor, optional): [R, C, S] boolean object mask. Defaults to None, ``object_mask``.

    Returns:
        tuple[tuple[str, float], ...] or None: ((name, mean density in g/cm^3), ...) of the materials with a nonzero
        mean, in library order; None for an empty mask.
    """
    names, nominal, mass, energy = _reference_attenuation(reference, energy_kev)
    mask = object_mask(X, reference, energy_kev=energy) if mask is None else mask
    values = X[mask].clamp_min(0).to(torch.float64)
    if values.numel() == 0:
        return None
    order = np.argsort(nominal * mass, kind="stable")  # bracketing order: increasing nominal attenuation
    mu = torch.as_tensor((nominal * mass)[order], device=values.device)
    m_sorted = torch.as_tensor(mass[order], device=values.device)
    rho_sorted = torch.as_tensor(nominal[order], device=values.device)
    M = len(names)
    j = (mu[None, :] <= values[:, None]).sum(1)
    a, b = (j - 1).clamp(0, M - 1), j.clamp(max=M - 1)
    inside = (j >= 1) & (j <= M - 1)
    span = torch.where(inside, mu[b] - mu[a], torch.ones_like(values))
    f = torch.where(inside, (values - mu[a]) / span, torch.zeros_like(values))
    rho_a = torch.where(inside, (1.0 - f) * rho_sorted[a], values / m_sorted[a])
    rho_b = torch.where(inside, f * rho_sorted[b], torch.zeros_like(values))
    totals = torch.zeros(M, dtype=torch.float64, device=values.device)
    totals.scatter_add_(0, a, rho_a)
    totals.scatter_add_(0, b, rho_b)
    mean = (totals / values.numel()).cpu().numpy()
    densities = np.zeros(M)
    densities[order] = mean
    return tuple((name, float(d)) for name, d in zip(names, densities) if d > 0)


def escape_probability(X: Tensor, grid: Grid, *, cell_cm: float = 1.0, num_rays: int = 100_000,
                       seed: int = 0) -> tuple[float, float]:
    """Escape probability of photons born in an object, from random rays through its attenuation volume.

    The volume is averaged onto cubic cells of ``cell_cm`` (so the estimate does not depend on the voxel pitch or its
    noise); rays start at points drawn in proportion to the attenuation, in isotropic directions, and accumulate the
    optical depth to the edge of the box in steps of a quarter cell.

    Args:
        X (Tensor): [R, C, S] attenuation in 1/cm (negative values count as 0).
        grid (Grid): grid of X.
        cell_cm (float, optional): edge of the averaging cells in cm. Defaults to 1.0.
        num_rays (int, optional): number of rays. Defaults to 100000.
        seed (int, optional): seed of the rays. Defaults to 0.

    Returns:
        (float, float): the mean of exp(-optical depth) over the rays, and the attenuation-weighted mean attenuation
        of the object in 1/cm (0 for an empty volume, with escape probability 1).
    """
    from .scatter_solver.geometry import resample_conservative

    coarse_grid = Grid.covering(grid, float(cell_cm))
    mu = resample_conservative(X.clamp_min(0).to(torch.float32)[None], grid, coarse_grid)[0]  # [R', C', S']
    weights = mu.flatten().to(torch.float64)
    total = float(weights.sum())
    if not total > 0:
        return 1.0, 0.0
    device = mu.device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    index = torch.multinomial(weights / total, int(num_rays), replacement=True, generator=generator)
    rows, cols, slices = torch.unravel_index(index, mu.shape)
    dx, dy, dz = coarse_grid.deltas
    lower = torch.tensor(coarse_grid.lower_corner(), device=device, dtype=torch.float32)
    extent = torch.tensor(coarse_grid.extent, device=device, dtype=torch.float32)
    jitter = torch.rand((int(num_rays), 3), generator=generator, device=device)
    start = torch.stack([(cols + jitter[:, 0]) * dx, (rows + jitter[:, 1]) * dy, (slices + jitter[:, 2]) * dz],
                        dim=1) + lower  # (x, y, z) in cm
    direction = torch.randn((int(num_rays), 3), generator=generator, device=device)
    direction = direction / direction.norm(dim=1, keepdim=True)
    step = 0.25 * float(cell_cm)
    num_steps = int(math.ceil(float(extent.norm()) / step))
    t = (torch.arange(num_steps, device=device, dtype=torch.float32) + 0.5) * step
    volume = mu.permute(2, 0, 1)[None, None]  # grid_sample order (depth z, height y, width x)
    optical_depth = torch.empty(int(num_rays), device=device, dtype=torch.float32)
    chunk = 16384
    for i in range(0, int(num_rays), chunk):
        points = start[i:i + chunk, None, :] + direction[i:i + chunk, None, :] * t[None, :, None]  # (n, T, 3)
        coords = 2.0 * (points - lower) / extent - 1.0  # (x, y, z) in [-1, 1] over the box
        sampled = torch.nn.functional.grid_sample(volume, coords[None, :, :, None, :], mode="bilinear",
                                                  padding_mode="zeros", align_corners=False)[0, 0, :, :, 0]
        optical_depth[i:i + chunk] = sampled.sum(dim=1) * step
    escape = float(torch.exp(-optical_depth.to(torch.float64)).mean())
    mean_attenuation = float((weights * weights).sum() / total)
    return escape, mean_attenuation


def object_size_cm(X: Tensor, grid: Grid, *, cell_cm: float = 1.0, num_rays: int = 100_000, default: float = 20.0,
                   seed: int = 0) -> float:
    """Characteristic size L of the object for ``SolverConfig.grouping_object_size_cm``, from its escape probability.

    The scattered-fluence balance treats leakage as the cross section 1/L, under which photons born in a homogeneous
    object escape with probability ``1 / (1 + mu L)``; L is the size that reproduces the escape probability of
    ``escape_probability`` with the object's mean attenuation mu, rounded to whole centimetres and at least 1 cm.
    For a convex homogeneous body it is close to the mean chord 4 V / S.

    Args:
        X (Tensor): [R, C, S] attenuation in 1/cm.
        grid (Grid): grid of X.
        cell_cm (float, optional): edge of the averaging cells in cm. Defaults to 1.0.
        num_rays (int, optional): number of rays. Defaults to 100000.
        default (float, optional): size in cm returned for an empty volume. Defaults to 20.0.
        seed (int, optional): seed of the rays. Defaults to 0.

    Returns:
        float: object size in cm.
    """
    escape, mean_attenuation = escape_probability(X, grid, cell_cm=cell_cm, num_rays=num_rays, seed=seed)
    if not mean_attenuation > 0 or escape >= 1.0:
        return float(default)
    size = (1.0 / max(escape, 1e-12) - 1.0) / mean_attenuation
    return float(max(1, round(size)))


def fan_angle(geometry: ConeGeometry) -> float:
    """In-plane fan angle ``2 atan(0.5 num_det_channels delta_det_channel / SDD)`` of the panel (offset ignored).

    A curved panel has the fan angle ``num_det_channels delta_det_channel / SDD`` (its channels are arc lengths), and a
    parallel beam the fan angle 0.

    Args:
        geometry (ConeGeometry or ParallelGeometry): scan geometry.

    Returns:
        float: fan angle in rad.
    """
    if geometry.is_parallel:
        return 0.0
    half = 0.5 * float(geometry.num_det_channels) * float(geometry.delta_det_channel)
    if geometry.use_curved_detector:
        return 2.0 * half / float(geometry.source_detector_dist)
    return 2.0 * math.atan(half / float(geometry.source_detector_dist))


def view_angles(num_views: int, geometry: Optional[ConeGeometry] = None, start_view_angle: float = 0.0,
                end_view_angle: Optional[float] = None) -> np.ndarray:
    """View angles evenly spaced over ``[start_view_angle, end_view_angle)``: ``start + (end - start) i / num_views``.

    Args:
        num_views (int): number of views V, at least 1.
        geometry (ConeGeometry or ParallelGeometry, optional): scan geometry, needed for the default end. Defaults
            to None.
        start_view_angle (float): first view angle in rad. Defaults to 0.0.
        end_view_angle (float, optional): end of the range in rad (excluded); ``start + 2 pi`` is a full rotation.
            Defaults to None, the short scan end ``start + pi + fan_angle(geometry)`` (``start + pi`` for a parallel
            beam).

    Returns:
        np.ndarray: [V] float64 view angles in rad (MBIRTORCH's angle convention).

    Raises:
        ValueError: if num_views < 1, the default end has no geometry, or the end does not exceed the start.
    """
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


def helical_scan(geometry: ConeGeometry, angles, pitch_cm: float, center_cm: float = 0.0) -> ConeGeometry:
    """Geometry of a helical scan with a constant pitch: the z shift grows by ``pitch_cm`` per turn of the angle.

    The z shift of view angle a is ``center_cm + pitch_cm (a - a_mid) / (2 pi)``, with ``a_mid`` the middle of the
    angular range, so that the travel is centred on ``center_cm`` (``ConeGeometry.with_helical_z_shifts``).

    Args:
        geometry (ConeGeometry): the scan geometry without helical views.
        angles (Sequence[float] or ndarray): [V] unwrapped view angles in rad, strictly monotone (e.g.
            ``view_angles(V, end_view_angle=2 pi turns)``).
        pitch_cm (float): table feed per turn in cm; positive moves the source and the panel toward +z as the angle
            grows.
        center_cm (float, optional): z in cm of the middle of the travel. Defaults to 0.0.

    Returns:
        ConeGeometry: the helical geometry.
    """
    a = np.asarray(angles.detach().cpu().numpy() if isinstance(angles, Tensor) else angles,
                   dtype=np.float64).reshape(-1)
    middle = 0.5 * (float(a.min()) + float(a.max())) if a.size else 0.0
    return geometry.with_helical_z_shifts(a, float(center_cm) + float(pitch_cm) * (a - middle) / (2.0 * math.pi))


def synthesize_cone_beam_data(densities, grid: Grid, geometry: ConeGeometry, angles, solver: ScatterSolver,
                              I0: float = 1.0e5, noise: bool = True, seed: int = 0, *,
                              energy_kev: Optional[float] = None, device=None,
                              progress: Optional[Callable[[int, float], None]] = None,
                              detector_gain=None) -> dict:
    """Counts with scatter, ``Y = Poisson(F exp(-A X_true) (1 + SPR))``, of a cone-beam or parallel-beam scan.

    The primary is MBIRTORCH's projection of the true attenuation and SPR the solver's scatter-to-primary ratio of the
    detector signals of one fully converged solve per view; a lazy solver with the "scattered_fluence" group rule first
    takes its grouping object size and medium from the true object, then builds its library. The open field F is the
    number I0 for a solver with uniform emission and no gain map, else the image ``I0 g f / mean(f)`` of the gain g and
    the solver's open-field signal f (``ScatterSolver.open_field_signal``, with the source's emission profile).

    Args:
        densities: true object on ``grid``: a density model with ``material(X, m)`` (e.g. ``WlabPhantom``) or a
            [M, R, C, S] volume in g/cm^3 of the solver library's materials.
        grid (Grid): reconstruction grid, the solver's fine grid.
        geometry (ConeGeometry or ParallelGeometry): scan geometry; it must be the solver's geometry.
        angles (array-like): [V] view angles in rad.
        solver (ScatterSolver): solver that defines the scatter truth.
        I0 (float): flat-field counts per pixel. Defaults to 1.0e5.
        noise (bool): draw ``Y = Poisson(Y_mean)`` with ``torch.Generator(seed)``, else Y = Y_mean. Defaults to True.
        seed (int): seed of the Poisson draws. Defaults to 0.
        energy_kev (float, optional): effective energy in keV. Defaults to None, the spectrum mean of the library.
        device (torch.device, optional): device. Defaults to None, the solver's device.
        progress (Callable[[int, float], None], optional): called as ``progress(view_index, seconds)`` after every
            solved view. Defaults to None.
        detector_gain (Tensor or ndarray, optional): [Rd, Cd] multiplicative per-pixel detector gain. Defaults to None
            (1).

    Returns:
        dict: numpy arrays (float32 unless noted) X_true [R, C, S] in 1/cm, angles [V] (float64), line_integrals, P,
        S, Y, Y_mean, SPR, solver_primary, solver_scatter [V, Rd, Cd], phi_air [Rd, Cd], view_seconds [V],
        si_iterations [V] (int), I0, open_field (the flat field of ``citrus_recon``: I0 as a 0-d array, or [Rd, Cd],
        or [V, Rd, Cd] for a per-view emission profile), energy_kev, mass_attenuation [M] in cm^2/g, and meta (setup
        and statistics).
    """
    from .recon.mbir_bridge import forward_project, make_model, mbirtorch_version

    from .citrus import primary_signals
    from .recon.mbir_bridge import ViewChunkProjector

    if solver.geometry != geometry:
        raise ValueError("the solver's geometry must be the scan geometry")
    start_time = time.perf_counter()
    device = torch.device(device) if device is not None else solver.device
    if isinstance(densities, Tensor) or not hasattr(densities, "material"):
        densities = StackedDensities(torch.as_tensor(densities, device=device, dtype=torch.float32))
    grouping_object_size = _set_grouping_object_size(solver, densities, grid, energy_kev, device)
    library = solver.library
    energy_kev = effective_energy(library) if energy_kev is None else float(energy_kev)
    mass_attenuation = mass_attenuation_at(library, energy_kev)
    if int(densities.num_materials) != library.num_materials:
        raise ValueError(f"the densities have {int(densities.num_materials)} materials, the library "
                         f"{library.num_materials} {tuple(library.materials)}")
    # X_true accumulated one material map at a time
    X = torch.zeros(*grid.shape, device=device, dtype=torch.float32)
    for m in range(library.num_materials):
        X.add_(densities.material(None, m).to(device=device, dtype=torch.float32), alpha=float(mass_attenuation[m]))
    angles = np.asarray(angles.detach().cpu().numpy() if isinstance(angles, Tensor) else angles,
                        dtype=np.float64).reshape(-1)

    # primary: MBIRTORCH forward projection (the reconstruction's own operator)
    t0 = time.perf_counter()
    model = make_model(grid, geometry, angles, device=device)
    line_integrals = forward_project(model, X).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    projection_seconds = time.perf_counter() - t0
    # the open field: the number I0 (uniform emission, no gain), else the image of the solver's open-field signal
    V, Rd, Cd = len(angles), geometry.num_det_rows, geometry.num_det_channels
    open_field: Union[float, Tensor] = float(I0)
    if solver.source.emission_profile is not None or detector_gain is not None:
        per_view = solver.source.emission_profile.dim() == 3 if solver.source.emission_profile is not None else False
        view_angles_used = angles if per_view else angles[:1]
        relative = torch.stack([solver.open_field_signal(float(a), dtype=torch.float64) for a in view_angles_used])
        relative = relative / relative.mean()
        if detector_gain is not None:
            gain = torch.as_tensor(detector_gain, dtype=torch.float64, device=relative.device)
            if tuple(gain.shape) != (Rd, Cd) or not bool((gain > 0).all()):
                raise ValueError(f"detector_gain must be a positive [rows, channels] = {[Rd, Cd]} map")
            relative = relative * gain
        open_field = (float(I0) * relative).to(device, torch.float32)
        open_field = open_field if per_view else open_field[0]
    P = open_field * torch.exp(-line_integrals)

    # scatter-to-primary ratio of the detector signals from the solver, one converged solve per view on the fields
    # built once from the maps
    solver_scatter = torch.empty(V, Rd, Cd, device=device, dtype=torch.float32)
    view_seconds = np.zeros(V)
    source_iterations = np.zeros(V, dtype=np.int64)
    source_iteration_converged = np.zeros(V, dtype=bool)
    t0 = time.perf_counter()
    fields = solver.material_fields_from_attenuation(X, grid, densities)
    for v in range(V):
        view_start = time.perf_counter()
        result = solver.solve_view_from_fields(fields, float(angles[v]))
        solver_scatter[v] = result.scatter_signal
        source_iterations[v] = result.source_iteration_result.num_iterations
        source_iteration_converged[v] = bool(result.source_iteration_result.converged)
        del result
        view_seconds[v] = time.perf_counter() - view_start
        if progress is not None:
            progress(v, view_seconds[v])
    solver_seconds = time.perf_counter() - t0
    # the primary model of the CITRUS loop: MBIRTORCH's per-material line integrals through the solver's transmission
    projector = ViewChunkProjector(grid, geometry, angles, model=model, device=device)
    solver_primary = primary_signals(solver, fields, angles, projector).to(device, torch.float32)
    del fields, projector
    # the solver's primary of an empty volume: zero path lengths of one material (no volume is traced)
    n = int(solver.primary_samples_per_pixel_axis or getattr(solver.projectors, "samples_per_pixel_axis", 1) or 1)
    zero_path = torch.zeros(1, Rd * n, Cd * n, device=device, dtype=torch.float32)
    phi_air = solver.primary_from_path_lengths(zero_path, [0], 0.0)["planar_fluence"].sum(0)
    del zero_path

    SPR = solver_scatter / solver_primary.clamp_min(torch.finfo(solver_primary.dtype).tiny)
    S = SPR * P
    Y_mean = P + S
    if noise:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        Y = torch.poisson(Y_mean, generator=generator)
    else:
        Y = Y_mean.clone()

    def to_numpy(t: Tensor) -> np.ndarray:
        return t.detach().to("cpu", torch.float32).numpy()

    spr_array = to_numpy(SPR)
    meta = {
        "description": "synthetic cone-beam data with scatter, citrus.utilities.synthesize_cone_beam_data"
                       if not geometry.is_parallel else
                       "synthetic parallel-beam data with scatter, citrus.utilities.synthesize_cone_beam_data",
        "num_views": int(V), "I0": float(I0), "noise": bool(noise), "seed": int(seed), "energy_kev": energy_kev,
        "primary_projector": "mbirtorch", "data_format": 2,  # on-disk record of citrus >= 0.3.0 (format 2)
        "materials": list(library.materials), "nominal_density": [float(d) for d in library.nominal_density],
        "mass_attenuation": {n: float(v) for n, v in zip(library.materials, mass_attenuation)},
        "angle_range_rad": [float(angles[0]), float(angles[-1])] if len(angles) else [],
        "geometry": asdict(geometry), "grid": asdict(grid), "solver_config": _solver_config_record(solver.config),
        "grouping_object_size_cm": grouping_object_size,
        "grouping_medium": [list(entry) for entry in solver.config.grouping_medium]
        if not isinstance(solver.config.grouping_medium, str) else solver.config.grouping_medium,
        "detector": getattr(solver.detector, "name", "detector"),
        "grid_response": getattr(solver.grid_response, "name", "grid"), "solver_dtype": str(solver.dtype),
        "open_field": "image" if isinstance(open_field, Tensor) else "number",
        "emission_profile": solver.source.emission_profile is not None, "detector_gain": detector_gain is not None,
        "mbirtorch_version": mbirtorch_version(), "torch_version": torch.__version__, "device": str(device),
        "seconds": {"total": time.perf_counter() - start_time, "mbirtorch_projection": projection_seconds,
                    "solver_total": solver_seconds,
                    "solver_per_view_mean": float(view_seconds.mean()),
                    "solver_per_view_median": float(np.median(view_seconds)),
                    "solver_per_view_max": float(view_seconds.max())},
        "si_iterations": {"min": int(source_iterations.min()), "max": int(source_iterations.max()),
                          "all_converged": bool(source_iteration_converged.all())},
        "stats": {"SPR_min": float(spr_array.min()), "SPR_max": float(spr_array.max()),
                  "SPR_mean": float(spr_array.mean()),
                  "SPR_p01": float(np.percentile(spr_array, 1)), "SPR_p99": float(np.percentile(spr_array, 99)),
                  "line_integral_max": float(line_integrals.max()),
                  "P_over_I0_min": float((P / open_field).min()),
                  "Y_min": float(Y.min()), "Y_max": float(Y.max())},
    }
    return {
        "X_true": to_numpy(X), "angles": angles, "line_integrals": to_numpy(line_integrals),
        "P": to_numpy(P), "S": to_numpy(S), "Y": to_numpy(Y), "Y_mean": to_numpy(Y_mean), "SPR": spr_array,
        "solver_primary": to_numpy(solver_primary), "solver_scatter": to_numpy(solver_scatter),
        "phi_air": to_numpy(phi_air), "view_seconds": view_seconds, "si_iterations": source_iterations,
        "I0": np.asarray(float(I0)),
        "open_field": to_numpy(open_field) if isinstance(open_field, Tensor) else np.asarray(open_field),
        "energy_kev": np.asarray(energy_kev), "mass_attenuation": mass_attenuation.numpy(),
        "meta": meta,
    }


# Keys of the solver configuration as recorded in a data set's meta (on-disk format of citrus <= 0.3.0, unchanged
# when the fields got plain names in 0.4.0)
_ON_DISK_SOLVER_CONFIG_KEYS = {"source_iteration_tolerance": "si_tol", "max_source_iterations": "si_max_iterations"}


def _set_grouping_object_size(solver: ScatterSolver, densities, grid: Grid, energy_kev: Optional[float],
                              device) -> float:
    """Set the grouping object size and medium of a lazy solver from the true object, without building its library.

    With the "scattered_fluence" group rule ``grouping_object_size_cm`` becomes ``object_size_cm`` of the true
    attenuation (its escape probability) and, for grouping_medium "auto", ``grouping_medium`` the true mean densities
    over the object mask.

    Args:
        solver (ScatterSolver): the solver; its config is replaced.
        densities: density model of the true object with ``material(X, m)``.
        grid (Grid): grid of the densities.
        energy_kev (float, optional): energy in keV; None for the spectrum-mean energy of the solver's config.
        device (torch.device): device of the attenuation volume.

    Returns:
        float: grouping object size in cm in the solver's config afterwards.
    """
    config = solver.config
    if solver.library_built or config.group_edges_method != "scattered_fluence" or config.group_edges_kev is not None:
        return float(config.grouping_object_size_cm)
    from .scatter_solver.cross_section_library import mass_attenuation_preview

    materials = solver.materials
    energy, mass = mass_attenuation_preview(config, materials, solver.spectrum, energy_kev=energy_kev)
    X = torch.zeros(*grid.shape, device=device, dtype=torch.float32)
    for m in range(len(materials)):
        X.add_(densities.material(None, m).to(device=device, dtype=torch.float32), alpha=float(mass[m]))
    size = object_size_cm(X, grid, default=float(config.grouping_object_size_cm))
    changes: dict = {"grouping_object_size_cm": size}
    if config.grouping_medium == "auto":
        mask = object_mask(X, solver, energy_kev=energy)
        count = int(mask.sum())
        if count:
            means = [float(densities.material(None, m).to(device=device, dtype=torch.float64)[mask].sum()) / count
                     for m in range(len(materials))]
            changes["grouping_medium"] = tuple((n, d) for n, d in zip(materials, means) if d > 0)
    solver.config = replace(config, **changes)
    return size


def _solver_config_record(config: SolverConfig) -> dict:
    """Solver configuration as recorded in a data set's meta.

    Args:
        config (SolverConfig): the configuration.

    Returns:
        dict: ``config.to_dict()`` with the on-disk keys of the renamed fields (``_ON_DISK_SOLVER_CONFIG_KEYS``).
    """
    return {_ON_DISK_SOLVER_CONFIG_KEYS.get(k, k): v for k, v in config.to_dict().items()}


def synthesize_wlab_dataset(num_views: int = 180, delta_voxel: float = 0.5, I0: float = 1.0e5, noise: bool = True,
                            seed: int = 0, out_path: Optional[Union[str, os.PathLike]] = None, *,
                            energy_kev: Optional[float] = None, config: Optional[SolverConfig] = None,
                            library: Optional[MultigroupLibrary] = None, device=None, start_view_angle: float = 0.0,
                            end_view_angle: Optional[float] = None,
                            progress: Optional[Callable[[int, float], None]] = None,
                            geometry: Optional[ConeGeometry] = None, spectrum=None,
                            detector: Optional[DetectorResponse] = None, grid_response: Optional[GridResponse] = None,
                            source: Optional[SourceModel] = None, detector_gain=None) -> dict:
    """Synthetic data of the WLAB phantom (``synthesize_cone_beam_data``), optionally written to disk.

    The phantom is placed on ``wlab_grid(delta_voxel)`` in the water and bone rows of the library of ``wlab_solver``;
    meta also records delta_voxel, start_view_angle, end_view_angle and short_scan.

    Args:
        num_views (int): number of views. Defaults to 180.
        delta_voxel (float): voxel pitch in cm. Defaults to 0.5.
        I0 (float): flat-field counts per pixel. Defaults to 1.0e5.
        noise (bool): draw Poisson noise. Defaults to True.
        seed (int): seed of the Poisson draws. Defaults to 0.
        out_path (str or os.PathLike, optional): base path of the ``save_dataset`` files. Defaults to None, not
            written.
        energy_kev (float, optional): effective energy in keV. Defaults to None, the spectrum mean.
        config (SolverConfig, optional): solver configuration. Defaults to None, ``SolverConfig()``.
        library (MultigroupLibrary, optional): prebuilt library. Defaults to None, a lazy library of
            ``config.materials``, which must include water and bone.
        device (torch.device, optional): device. Defaults to None, CUDA when available, else the CPU.
        start_view_angle (float): first view angle in rad. Defaults to 0.0.
        end_view_angle (float, optional): end of the angular range in rad (excluded). Defaults to None, the short scan
            end ``start + pi + fan_angle(geometry)`` (``start + pi`` for a parallel beam).
        progress (Callable[[int, float], None], optional): called as ``progress(view_index, seconds)`` after every
            view. Defaults to None.
        geometry (ConeGeometry or ParallelGeometry, optional): scan geometry, e.g.
            ``parallel_wlab_geometry(delta_voxel)``. Defaults to None, ``full_field_wlab_geometry()``.
        spectrum (tuple[np.ndarray, np.ndarray], optional): source spectrum, required without ``library``. Defaults
            to None.
        detector (DetectorResponse, optional): detector response of the truth solver. Defaults to None (ideal
            counting).
        grid_response (GridResponse, optional): anti-scatter grid of the truth solver. Defaults to None (no grid).
        source (SourceModel, optional): source model (emission profile) of the truth solver. Defaults to None
            (uniform emission).
        detector_gain (Tensor or ndarray, optional): [Rd, Cd] per-pixel detector gain. Defaults to None (1).

    Returns:
        dict: the data set, as returned by ``synthesize_cone_beam_data``.
    """
    device = (torch.device(device) if device is not None
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    geometry = full_field_wlab_geometry() if geometry is None else geometry
    grid = wlab_grid(delta_voxel)
    solver = wlab_solver(delta_voxel, config=config, library=library, geometry=geometry, device=device,
                         spectrum=spectrum, detector=detector, grid_response=grid_response, source=source)
    phantom = WlabPhantom(grid, solver.materials, device=device, dtype=torch.float32)
    angles = view_angles(num_views, geometry, start_view_angle, end_view_angle)
    end = float(start_view_angle) + math.pi + fan_angle(geometry) if end_view_angle is None else float(end_view_angle)
    out = synthesize_cone_beam_data(phantom, grid, geometry, angles, solver, I0, noise, seed, energy_kev=energy_kev,
                                    device=device, progress=progress, detector_gain=detector_gain)
    out["meta"].update(description="synthetic WLAB scatter dataset, citrus.utilities.synthesize_wlab_dataset",
                       delta_voxel=float(delta_voxel), start_view_angle=float(start_view_angle),
                       end_view_angle=end, short_scan=bool(end - float(start_view_angle) < 2.0 * math.pi - 1e-9))
    if out_path is not None:
        save_dataset(out, out_path)
    return out


def save_dataset(data: dict, out_path: Union[str, os.PathLike]) -> list[str]:
    """Write a data set to ``<out_path>.npz`` (meta as the JSON string ``meta_json``) and ``<out_path>.json`` (meta).

    Args:
        data (dict): numpy arrays plus ``meta``; ``meta["files"]`` is set to the two file names.
        out_path (str or os.PathLike): base path; a trailing .npz or .json is removed.

    Returns:
        list[str]: the two file names.
    """
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
    """Read a data set written by ``save_dataset``.

    Args:
        path (str or os.PathLike): path of the .npz file; the extension may be omitted.

    Returns:
        dict: numpy arrays plus ``meta``.
    """
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
    """Cap the PyTorch CUDA allocator of a device (no-op without CUDA or for mib <= 0).

    Args:
        mib (float): cap in MiB. Defaults to 4400.0.
        device (int): CUDA device index. Defaults to 0.
    """
    if torch.cuda.is_available() and mib > 0:
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, mib * 2 ** 20 / total), device)


# =====================================================================================================
# plotting helpers (matplotlib is imported lazily)
# =====================================================================================================

def _to_numpy(x) -> np.ndarray:
    """Convert a tensor (as float32 on the CPU) or an array-like to a numpy array.

    Args:
        x (Tensor or array-like): the values.

    Returns:
        np.ndarray: the values as a numpy array.
    """
    return x.detach().float().cpu().numpy() if isinstance(x, Tensor) else np.asarray(x)


def central_slice(X) -> np.ndarray:
    """Central z slice of a volume (the mean of the two middle slices for an even count).

    Args:
        X (Tensor or np.ndarray): [rows, cols, slices] volume.

    Returns:
        np.ndarray: [rows, cols] slice.
    """
    X = _to_numpy(X)
    S = X.shape[2]
    return X[:, :, S // 2] if S % 2 else 0.5 * (X[:, :, S // 2 - 1] + X[:, :, S // 2])


def central_row(image_slice) -> np.ndarray:
    """Central row of a slice (the mean of the two middle rows for an even count).

    Args:
        image_slice (Tensor or np.ndarray): [rows, cols] slice.

    Returns:
        np.ndarray: [cols] row.
    """
    image_slice = _to_numpy(image_slice)
    R = image_slice.shape[0]
    return image_slice[R // 2] if R % 2 else 0.5 * (image_slice[R // 2 - 1] + image_slice[R // 2])


def _pyplot():
    """Import matplotlib.pyplot lazily; the caller chooses the backend.

    Returns:
        module: ``matplotlib.pyplot``.
    """
    import matplotlib.pyplot as plt  # the caller chooses the backend (e.g. matplotlib.use("Agg") in scripts)

    return plt


def plot_slices(images: Sequence[tuple[str, np.ndarray]], grid: Grid, path: Optional[str] = None,
                vmin: float = 0.0, vmax: float = 0.3, title: Optional[str] = None):
    """Plot central z slices side by side on a shared gray scale with one colorbar.

    Args:
        images (Sequence[tuple[str, np.ndarray]]): (label, volume [R, C, S] or slice [R, C]) pairs in 1/cm.
        grid (Grid): grid of the images, for the extent in cm.
        path (str, optional): file the figure is saved to. Defaults to None, not saved.
        vmin (float): lower end of the gray scale in 1/cm. Defaults to 0.0.
        vmax (float): upper end of the gray scale in 1/cm. Defaults to 0.3.
        title (str, optional): figure title. Defaults to None.

    Returns:
        matplotlib.figure.Figure: the figure.
    """
    plt = _pyplot()
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 2.9), squeeze=False)
    extent_x, extent_y, _ = grid.extent
    lower = grid.lower_corner()
    image_artist = None
    for ax, (label, image) in zip(axes[0], images):
        image = _to_numpy(image)
        image_slice = central_slice(image) if image.ndim == 3 else image
        image_artist = ax.imshow(image_slice, cmap="gray", vmin=vmin, vmax=vmax, origin="lower",
                                 extent=[lower[0], lower[0] + extent_x, lower[1], lower[1] + extent_y])
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("x (cm)", fontsize=8)
    axes[0, 0].set_ylabel("y (cm)", fontsize=8)
    fig.colorbar(image_artist, ax=axes[0].tolist(), shrink=0.85, label="attenuation (1/cm)")
    if title:
        fig.suptitle(title, fontsize=10)
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def plot_profiles(profiles: Sequence[tuple[str, np.ndarray]], grid: Grid, path: Optional[str] = None,
                  ylabel: str = "attenuation (1/cm)", title: Optional[str] = None):
    """Plot central-row profiles along x (row y = 0 of the central z slice).

    Args:
        profiles (Sequence[tuple[str, np.ndarray]]): (label, volume [R, C, S]) pairs.
        grid (Grid): grid of the volumes, for the x coordinates in cm.
        path (str, optional): file the figure is saved to. Defaults to None, not saved.
        ylabel (str): label of the y axis. Defaults to "attenuation (1/cm)".
        title (str, optional): axes title. Defaults to None.

    Returns:
        matplotlib.figure.Figure: the figure.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    x = grid.axis_coordinates(dtype=torch.float64)[0].numpy()
    styles = ["-", "--", "-.", ":"]
    for i, (label, volume) in enumerate(profiles):
        ax.plot(x, central_row(central_slice(volume)), ls=styles[i % len(styles)], lw=1.6 if i else 1.2, label=label)
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
    """Plot convergence curves against the iteration number.

    Args:
        curves (Sequence[tuple[str, Sequence[float]]]): (label, values per iteration) pairs.
        path (str, optional): file the figure is saved to. Defaults to None, not saved.
        ylabel (str): label of the y axis. Defaults to "relative RMSE".
        xlabel (str): label of the x axis. Defaults to "outer iteration (0 = FDK)".
        logy (bool): logarithmic y axis. Defaults to True.
        title (str, optional): axes title. Defaults to None.

    Returns:
        matplotlib.figure.Figure: the figure.
    """
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
