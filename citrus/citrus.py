"""The CITRUS reconstruction loop: scatter-corrected MBIR with the LBTE scatter solver in the loop.

Every outer iteration advances the collided fluence of every view by one transport sweep, divides the measured counts
by one plus the scatter-to-primary ratio of the current densities and runs one warm-started MBIR iteration on the
corrected data. The module holds the loop (``citrus_recon``, ``mbir_recon``), the per-view scatter step and its
operators, the density model, the compact per-view state storage and the angular interpolation of sparse-view scatter.
The method is described in docs/source/method.rst, "The CITRUS loop" and "Options and secondary components".
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Literal, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from .recon.mbir_bridge import ViewChunkProjector, fdk, make_cone_model, mbir_iterations
from .scatter_solver.angular_quadrature import Quadrature
from .scatter_solver.cross_sections import MultigroupLibrary
from .scatter_solver.geometry import ConeGeometry, Grid
from .scatter_solver.scatter_solver import MaterialFields, ScatterSolver
from .scatter_solver.uncollided import present_materials
from .utilities import (attenuation_to_densities, attenuation_to_density, effective_energy, grouping_medium_estimate,
                        object_mask, object_size_cm)

__all__ = [
    "CitrusState",
    "uncollided_operator",
    "first_collision_operator",
    "multiple_collision_operator",
    "accumulator_update_operator",
    "project_scatter",
    "citrus_step",
    "citrus_view_step",
    "primary_signals",
    "DensityModel",
    "StateStore",
    "CitrusConfig",
    "SCATTER_VIEW_INTERPOLATION_ORDERS",
    "scatter_view_indices",
    "angular_neighbors",
    "full_rotation",
    "interpolation_weights",
    "interpolate_scatter",
    "estimate_regularization",
    "apply_regularization",
    "citrus_recon",
    "mbir_recon",
]

REGULARIZATION_NAMES = ("sigma_y", "sigma_x", "sigma_prox")


# =====================================================================================================
# per-view operators of the CITRUS scatter step
# =====================================================================================================

@dataclass
class CitrusState:
    """Caller-owned state of the CITRUS scatter recursion for one view (keep one state per view).

    Attributes:
        angle (float or None): view angle in rad the state belongs to; fixed by the first step. Defaults to None.
        collided_moments (Tensor or None): [G, (L+1)^2, 4, Rc, Cc, Sc] accumulator Phi of the
            collided-fluence moments (sum over the collision orders n >= 1); None means zero. Defaults to None.
        cone_quadrature (Quadrature or None): cached collapsed-cone direction set of the view. Defaults to None.
        num_steps (int): number of steps applied. Defaults to 0.
        history (list[float]): relative change of the scalar-flux cell means of Phi at every step. Defaults to [].
    """

    angle: Optional[float] = None
    collided_moments: Optional[Tensor] = None
    cone_quadrature: Optional[Quadrature] = None
    num_steps: int = 0
    history: list[float] = field(default_factory=list)


def _material_fields_or_build(solver: ScatterSolver, rho: Tensor, grid: Grid,
                              fields: Optional[MaterialFields]) -> MaterialFields:
    """The given material fields, or the solver's material fields of ``rho`` when ``fields`` is None.

    Args:
        solver (ScatterSolver): the scatter solver.
        rho (Tensor): [M, R, C, S] densities in g/cm^3 of the M library materials on ``grid``.
        grid (Grid): the fine (reconstruction) grid.
        fields (MaterialFields or None): prebuilt material fields.

    Returns:
        MaterialFields: ``fields``, or ``solver.material_fields(rho, grid)`` when it is None.
    """
    return solver.material_fields(rho, grid) if fields is None else fields


def uncollided_operator(solver: ScatterSolver, rho: Tensor, grid: Grid, angle: float,
                        fields: Optional[MaterialFields] = None) -> Tensor:
    """Uncollided fluence G_1 of the densities at the points of the solver's quadrature grid.

    Args:
        solver (ScatterSolver): the scatter solver.
        rho (Tensor): [M, R, C, S] densities in g/cm^3 of the M library materials on ``grid``; unused when
            ``fields`` is given.
        grid (Grid): the fine (reconstruction) grid.
        angle (float): view angle in rad.
        fields (MaterialFields, optional): prebuilt material fields of ``rho``. Defaults to None (built here).

    Returns:
        Tensor: [G, Rq, Cq, Sq] uncollided fluence phi_uncollided per energy group at the quadrature points.
    """
    return solver.uncollided_fluence(_material_fields_or_build(solver, rho, grid, fields), angle)


def first_collision_operator(solver: ScatterSolver, rho: Tensor, grid: Grid, angle: float, phi_uncollided: Tensor,
                             fields: Optional[MaterialFields] = None) -> Tensor:
    """Once-collided fluence moments phi^(1): one transport sweep of the first-collision source.

    Args:
        solver (ScatterSolver): the scatter solver.
        rho (Tensor): [M, R, C, S] densities in g/cm^3 of the M library materials on ``grid``; unused when
            ``fields`` is given.
        grid (Grid): the fine (reconstruction) grid.
        angle (float): view angle in rad.
        phi_uncollided (Tensor): [G, Rq, Cq, Sq] uncollided fluence at the quadrature points
            (``uncollided_operator``).
        fields (MaterialFields, optional): prebuilt material fields of ``rho``. Defaults to None (built here).

    Returns:
        Tensor: [G, (L+1)^2, 4, Rc, Cc, Sc] once-collided fluence moments on the transport grid.
    """
    fields = _material_fields_or_build(solver, rho, grid, fields)
    return solver.transport_sweep(fields, solver.first_collision_moment_source(fields, angle, phi_uncollided))


def _check_accumulator(solver: ScatterSolver, fields: MaterialFields, collided_moments: Tensor) -> None:
    """Check the shape of an accumulator against the solver and the transport grid of ``fields``.

    Args:
        solver (ScatterSolver): the scatter solver.
        fields (MaterialFields): material fields whose transport grid the accumulator lives on.
        collided_moments (Tensor): [G, (L+1)^2, 4, Rc, Cc, Sc] accumulator to check.

    Raises:
        ValueError: if the shape differs.
    """
    sweep_context = solver.sweep_context(fields.grid_coarse)
    expected = (solver.num_groups, sweep_context.num_moments, 4, *fields.grid_coarse.shape)
    if tuple(collided_moments.shape) != expected:
        raise ValueError(f"collided_moments must be {expected}, got {tuple(collided_moments.shape)}")


def multiple_collision_operator(solver: ScatterSolver, rho: Tensor, grid: Grid, angle: float,
                                collided_moments: Optional[Tensor], fields: Optional[MaterialFields] = None) -> Tensor:
    """One transport sweep ``K Phi`` of the scattering source of the collided moments Phi.

    Args:
        solver (ScatterSolver): the scatter solver.
        rho (Tensor): [M, R, C, S] densities in g/cm^3 of the M library materials on ``grid``; unused when
            ``fields`` is given.
        grid (Grid): the fine (reconstruction) grid.
        angle (float): view angle in rad; unused (the collided transport does not depend on the view).
        collided_moments (Tensor or None): [G, (L+1)^2, 4, Rc, Cc, Sc] collided moments Phi; None gives zeros.
        fields (MaterialFields, optional): prebuilt material fields of ``rho``. Defaults to None (built here).

    Returns:
        Tensor: [G, (L+1)^2, 4, Rc, Cc, Sc] collided moments of the next collision order.

    Raises:
        ValueError: if ``collided_moments`` has the wrong shape.
    """
    fields = _material_fields_or_build(solver, rho, grid, fields)
    sweep_context = solver.sweep_context(fields.grid_coarse)
    if collided_moments is None:
        return torch.zeros(solver.num_groups, sweep_context.num_moments, 4, *fields.grid_coarse.shape,
                           device=solver.device, dtype=solver.dtype)
    _check_accumulator(solver, fields, collided_moments)
    scatter_source = solver.scatter_moment_source(fields, collided_moments.to(solver.device, solver.dtype),
                                                  sweep_context)
    return solver.transport_sweep(fields, scatter_source)


def accumulator_update_operator(solver: ScatterSolver, rho: Tensor, grid: Grid, angle: float, phi_uncollided: Tensor,
                                collided_moments: Optional[Tensor], fields: Optional[MaterialFields] = None) -> Tensor:
    """New accumulator ``phi^(1) + K Phi``: first-collision and scattering sources transported in one sweep.

    Args:
        solver (ScatterSolver): the scatter solver.
        rho (Tensor): [M, R, C, S] densities in g/cm^3 of the M library materials on ``grid``; unused when
            ``fields`` is given.
        grid (Grid): the fine (reconstruction) grid.
        angle (float): view angle in rad.
        phi_uncollided (Tensor): [G, Rq, Cq, Sq] uncollided fluence at the quadrature points
            (``uncollided_operator``).
        collided_moments (Tensor or None): [G, (L+1)^2, 4, Rc, Cc, Sc] previous accumulator Phi;
            None gives phi^(1) alone.
        fields (MaterialFields, optional): prebuilt material fields of ``rho``. Defaults to None (built here).

    Returns:
        Tensor: [G, (L+1)^2, 4, Rc, Cc, Sc] new accumulator.

    Raises:
        ValueError: if ``collided_moments`` has the wrong shape.
    """
    fields = _material_fields_or_build(solver, rho, grid, fields)
    sweep_context = solver.sweep_context(fields.grid_coarse)
    first_collision_source = solver.first_collision_moment_source(fields, angle, phi_uncollided, sweep_context)
    if collided_moments is None:
        return solver.transport_sweep(fields, first_collision_source)
    _check_accumulator(solver, fields, collided_moments)
    scatter_source = solver.scatter_moment_source(fields, collided_moments.to(solver.device, solver.dtype),
                                                  sweep_context)

    def summed_source(direction_index: Tensor, out: Optional[Tensor] = None) -> Tensor:
        source = first_collision_source(direction_index, out)  # fills `out` when the sweep lends its storage
        source.add_(scatter_source(direction_index))
        return source

    return solver.transport_sweep(fields, summed_source)


def project_scatter(solver: ScatterSolver, rho: Tensor, grid: Grid, angle: float, phi_uncollided: Tensor,
                    collided_moments: Tensor, fields: Optional[MaterialFields] = None,
                    quadrature: Optional[Quadrature] = None) -> dict[str, Tensor]:
    """Total scatter image (first order and multiple collision) of one view in one collapsed-cone pass.

    Args:
        solver (ScatterSolver): the scatter solver.
        rho (Tensor): [M, R, C, S] densities in g/cm^3 of the M library materials on ``grid``; unused when
            ``fields`` is given.
        grid (Grid): the fine (reconstruction) grid.
        angle (float): view angle in rad.
        phi_uncollided (Tensor): [G, Rq, Cq, Sq] uncollided fluence at the quadrature points
            (``uncollided_operator``).
        collided_moments (Tensor): [G, (L+1)^2, 4, Rc, Cc, Sc] collided moments Phi (their cell means are used).
        fields (MaterialFields, optional): prebuilt material fields of ``rho``. Defaults to None (built here).
        quadrature (Quadrature, optional): collapsed-cone direction set. Defaults to None (the solver's
            ``cone_quadrature``).

    Returns:
        dict[str, Tensor]: the image of ``ScatterSolver.scatter_image``: ``fluence``, ``planar_fluence``,
            ``signal_per_group`` [G, Rd, Cd] and ``signal`` [Rd, Cd].
    """
    fields = _material_fields_or_build(solver, rho, grid, fields)
    scatter_source_moments = solver.multiple_collision_source_moments(
        fields, collided_moments.to(solver.device, solver.dtype))
    return solver.scatter_image(fields, angle, phi_uncollided, scatter_source_moments, quadrature)


def citrus_step(solver: ScatterSolver, state: CitrusState, rho: Tensor, grid: Grid,
                angle: float, fields: Optional[MaterialFields] = None) -> tuple[Tensor, Tensor, CitrusState]:
    """One CITRUS scatter step of one view from the densities ``rho``.

    The step computes the uncollided fluence, the new accumulator ``phi^(1) + K Phi`` in one sweep and the scatter
    signal; with the densities held fixed, N steps equal N source iterations.

    Args:
        solver (ScatterSolver): the scatter solver.
        state (CitrusState): state of the view; not modified.
        rho (Tensor): [M, R, C, S] densities in g/cm^3 of the M library materials on ``grid``; unused when ``fields``
            is given.
        grid (Grid): the fine (reconstruction) grid.
        angle (float): view angle in rad; must equal ``state.angle`` once that is set.
        fields (MaterialFields, optional): prebuilt ``solver.material_fields(rho, grid)``, shared by the views of an
            outer iteration. Defaults to None (built here).

    Returns:
        (Tensor, Tensor, CitrusState): scatter signal [Rd, Cd], Siddon primary signal [Rd, Cd] and the new state.

    Raises:
        ValueError: if ``angle`` differs from ``state.angle``.
    """
    return citrus_view_step(solver, state, _material_fields_or_build(solver, rho, grid, fields), angle)


def citrus_view_step(solver: ScatterSolver, state: CitrusState, fields: MaterialFields, angle: float, *,
                     primary: bool = True) -> tuple[Tensor, Optional[Tensor], CitrusState]:
    """CITRUS scatter step of one view from prebuilt material fields, optionally without the primary.

    Args:
        solver (ScatterSolver): the scatter solver.
        state (CitrusState): state of the view; not modified.
        fields (MaterialFields): ``solver.material_fields(rho, grid)`` or, as in the CITRUS loop,
            ``solver.material_fields_from_attenuation(X, grid, density_model)``.
        angle (float): view angle in rad; must equal ``state.angle`` once that is set.
        primary (bool, optional): also trace the view's primary with the solver (``ScatterSolver.primary``, Siddon).
            Defaults to True.

    Returns:
        (Tensor, Tensor or None, CitrusState): scatter signal [Rd, Cd], primary signal [Rd, Cd] (None when
            ``primary`` is False) and the new state.

    Raises:
        ValueError: if ``angle`` differs from ``state.angle``.
    """
    if state.angle is not None and abs(float(state.angle) - float(angle)) > 1e-12:
        raise ValueError(f"state belongs to angle {state.angle}, got {angle}")
    grid = fields.grid_fine
    cone_quadrature = (state.cone_quadrature if state.cone_quadrature is not None
                       else solver.cone_quadrature(grid, angle))
    phi_uncollided = uncollided_operator(solver, None, grid, angle, fields)
    # phi^(1) + K Phi in one sweep
    new_collided_moments = accumulator_update_operator(solver, None, grid, angle, phi_uncollided,
                                                       state.collided_moments, fields)
    scatter = project_scatter(solver, None, grid, angle, phi_uncollided, new_collided_moments, fields,
                              cone_quadrature)["signal"]
    del phi_uncollided
    primary_signal = solver.primary(fields, angle)["signal"] if primary else None
    new_norm = float(torch.linalg.vector_norm(new_collided_moments[:, 0, 0]))
    if state.collided_moments is None:
        relative_change = 1.0 if new_norm > 0 else 0.0
    else:
        previous = state.collided_moments[:, 0, 0].to(new_collided_moments)
        difference = float(torch.linalg.vector_norm(new_collided_moments[:, 0, 0] - previous))
        relative_change = difference / new_norm if new_norm > 0 else 0.0
    new_state = CitrusState(angle=float(angle), collided_moments=new_collided_moments,
                            cone_quadrature=cone_quadrature, num_steps=state.num_steps + 1,
                            history=state.history + [relative_change])
    return scatter, primary_signal, new_state


def primary_signals(solver: ScatterSolver, fields: MaterialFields, angles: Union[Sequence[float], np.ndarray],
                    projector: ViewChunkProjector, views_per_chunk: Optional[int] = None,
                    out: Optional[Tensor] = None,
                    quantity: Literal["signal", "planar_fluence"] = "signal") -> Tensor:
    """Primary signals of every view from MBIRTORCH projections of the material density maps.

    The path lengths ``P_m = A rho_m`` in g/cm^2 of the materials present are projected one material at a time in chunks
    of views, and ``ScatterSolver.primary_from_path_lengths`` turns them into the primary of each view.

    Args:
        solver (ScatterSolver): the scatter solver (spectrum, detector and grid responses).
        fields (MaterialFields): material fields of the current densities.
        angles (Sequence[float] or ndarray): [V] view angles in rad; must be the views of ``projector``.
        projector (ViewChunkProjector): MBIRTORCH projector of the reconstruction grid over the same views.
        views_per_chunk (int, optional): views per projection chunk. Defaults to None (``max(1, V // (M' + 1))`` with
            M' the number of materials projected).
        out (Tensor, optional): [V, Rd, Cd] tensor to fill. Defaults to None (allocated).
        quantity (str, optional): ``"signal"`` for the detector signal in keV/cm^2 or ``"planar_fluence"`` for the
            planar fluence summed over the groups in photons/cm^2. Defaults to "signal".

    Returns:
        Tensor: [V, Rd, Cd] primary signal (or planar fluence) of every view.

    Raises:
        ValueError: if ``quantity`` is unknown or ``angles`` are not the projector's views.
    """
    if quantity not in ("signal", "planar_fluence"):
        raise ValueError(f"quantity must be 'signal' or 'planar_fluence', got {quantity!r}")
    angle_array = np.asarray(angles, dtype=np.float64).reshape(-1)
    V = angle_array.size
    if projector.num_views != V or not np.allclose(projector.angles, angle_array, rtol=0.0, atol=1e-12):
        raise ValueError("the projector's views must be the given angles")
    if fields.fine_materials is not None:
        materials = list(fields.fine_materials)
    else:
        present = present_materials(fields.rho_fine)
        materials = list(range(fields.num_materials)) if present is None else present
    Rd, Cd = solver.geometry.num_det_rows, solver.geometry.num_det_channels
    if out is None:
        out = torch.empty((V, Rd, Cd), device=solver.device, dtype=solver.dtype)
    if views_per_chunk is None:
        chunk_views = max(1, V // (len(materials) + 1))
    else:
        chunk_views = max(1, int(views_per_chunk))
    for start, stop in projector.chunks(chunk_views):
        path_lengths = None
        for i, m in enumerate(materials):
            rho_m = fields.fine_material(m)
            projection = projector.project(rho_m, start, stop)  # [V_chunk, Rd n, Cd n]
            del rho_m
            if path_lengths is None:
                path_lengths = torch.empty((len(materials),) + tuple(projection.shape), device=projection.device,
                                           dtype=torch.float32)
            path_lengths[i] = projection
            del projection
        for v in range(start, stop):
            image = solver.primary_from_path_lengths(path_lengths[:, v - start], materials, float(angle_array[v]))
            out[v] = image["signal"] if quantity == "signal" else image["planar_fluence"].sum(0)
        del path_lengths
    return out


# =====================================================================================================
# density model
# =====================================================================================================

@dataclass
class DensityModel:
    """Density model of the solver: MBIR's attenuation X in 1/cm to the densities rho of the reference materials.

    The interpolating model at E_eff (``utilities.attenuation_to_densities``) is applied to X clamped at 0; ``material``
    evaluates one material's map without forming the [M, R, C, S] volume.

    Attributes:
        library (MultigroupLibrary): solver library whose reference materials ``library.materials`` define the model.
        energy_kev (float): effective energy E_eff in keV of MBIR's monoenergetic attenuation.
        last_negative_fraction (float): fraction of the voxels clamped at 0 in the last call of ``__call__`` or
            ``negative_fraction``. Defaults to 0.0.
        slab_voxels (int): voxels per slab of ``material`` and ``negative_fraction``. Defaults to ``1 << 22``.
    """

    library: MultigroupLibrary
    energy_kev: float
    last_negative_fraction: float = 0.0
    slab_voxels: int = 1 << 22

    @classmethod
    def from_library(cls, library: MultigroupLibrary, energy_kev: Optional[float] = None) -> "DensityModel":
        """Density model of a library at an effective energy.

        Args:
            library (MultigroupLibrary): solver library with the reference materials and their nominal densities.
            energy_kev (float, optional): effective energy E_eff in keV. Defaults to None (the spectrum-mean energy of
                the library, ``utilities.effective_energy``).

        Returns:
            DensityModel: the density model.
        """
        return cls(library, effective_energy(library) if energy_kev is None else float(energy_kev))

    def __call__(self, X: Tensor) -> Tensor:
        """Densities of all reference materials of X; also sets ``last_negative_fraction``.

        Args:
            X (Tensor): [...] attenuation in 1/cm at E_eff; negative values are clamped at 0.

        Returns:
            Tensor: [M, ...] densities in g/cm^3.
        """
        self.last_negative_fraction = float((X < 0).float().mean())
        return attenuation_to_densities(self.library, X.clamp_min(0), self.energy_kev)

    def _slabs(self, X: Tensor) -> list[slice]:
        """Slices along the first axis of X with about ``slab_voxels`` voxels each.

        Args:
            X (Tensor): [...] volume.

        Returns:
            list[slice]: slices covering the first axis of X.
        """
        voxels_per_index = max(1, int(np.prod(X.shape[1:]))) if X.dim() > 1 else 1
        indices_per_slab = max(1, int(self.slab_voxels) // voxels_per_index)
        num_indices = int(X.shape[0]) if X.dim() else 1
        return [slice(i, i + indices_per_slab) for i in range(0, num_indices, indices_per_slab)]

    def material(self, X: Tensor, m: int) -> Tensor:
        """Density map of one library material of X, bit for bit ``self(X)[m]`` without the other materials.

        The map is evaluated slab by slab along the first axis, so the result is the only full-size temporary.

        Args:
            X (Tensor): [...] attenuation in 1/cm at E_eff.
            m (int): index of the library material.

        Returns:
            Tensor: [...] density in g/cm^3 of material m.
        """
        if X.dim() == 0:
            return attenuation_to_density(self.library, X, m, self.energy_kev)
        out = torch.empty_like(X)
        for slab in self._slabs(X):
            out[slab] = attenuation_to_density(self.library, X[slab], m, self.energy_kev)
        return out

    def negative_fraction(self, X: Tensor) -> float:
        """Fraction of the voxels of X below 0, counted slab by slab; also stored in ``last_negative_fraction``.

        Args:
            X (Tensor): [...] attenuation in 1/cm.

        Returns:
            float: fraction of the voxels below 0.
        """
        if X.dim() == 0:
            count = int(X < 0)
        else:
            count = sum(int(torch.count_nonzero(X[slab] < 0)) for slab in self._slabs(X))
        self.last_negative_fraction = count / max(1, X.numel())
        return self.last_negative_fraction


# =====================================================================================================
# per-view accumulator storage
# =====================================================================================================

_FLOAT16_TARGET = 2.0 ** 14  # |q| <= 16384 (< 65504): the per-slice maximum maps here, normals reach 1e-9 x max


class StateStore:
    """Compact storage of one CITRUS accumulator per view: the linear-discontinuous cell mean in scaled float16.

    ``put`` stores an accumulator [G, N, 4, Rc, Cc, Sc] (N = (L+1)^2 moments) with one scale per group and moment, and
    ``get`` returns it as float32 with zero linear-discontinuous slopes.

    Args:
        num_slots (int): number of stored accumulators (views).
        full_shape (Sequence[int]): accumulator shape [G, N, 4, Rc, Cc, Sc].
        device (torch.device or str): device of the accumulators passed to ``put`` and returned by ``get``.
        location (str, optional): ``"gpu"``, ``"cpu"`` (host memory) or ``"auto"`` (the GPU when the total fits
            ``gpu_budget_mib`` on a CUDA device). Defaults to "auto".
        gpu_budget_mib (float, optional): largest total in MiB kept on the GPU by ``"auto"``. Defaults to 2600.0.
        pinned_memory_limit_gib (float, optional): largest total in GiB stored in pinned host memory. Defaults to 8.0.

    Attributes:
        location (str): ``"gpu"`` or ``"cpu"``, where the accumulators are stored.
        pinned (bool): whether the host buffers are pinned.
        stored_shape (tuple[int, ...]): stored shape [G, N, 1, Rc, Cc, Sc].
        transfer_seconds (float): seconds spent in ``put`` and ``get`` (synchronized on CUDA).

    Raises:
        ValueError: if ``full_shape`` does not have six axes or ``location`` is unknown.
    """

    def __init__(self, num_slots: int, full_shape: Sequence[int], device, location: str = "auto",
                 gpu_budget_mib: float = 2600.0, pinned_memory_limit_gib: float = 8.0) -> None:
        self.num_slots = int(num_slots)
        self.full_shape = tuple(int(n) for n in full_shape)
        if len(self.full_shape) != 6:
            raise ValueError(f"accumulator shape must be [G, N, 4, Rc, Cc, Sc], got {self.full_shape}")
        self.stored_shape = self.full_shape[:2] + (1,) + self.full_shape[3:]
        self.device = torch.device(device)
        if location == "auto":
            fits = self.total_bytes <= gpu_budget_mib * 2 ** 20
            location = "gpu" if self.device.type == "cuda" and fits else "cpu"
        if location not in ("gpu", "cpu"):
            raise ValueError(f"location must be 'auto', 'gpu' or 'cpu', got {location!r}")
        self.location = location
        self.pinned = (location == "cpu" and self.device.type == "cuda"
                       and self.total_bytes <= pinned_memory_limit_gib * 2 ** 30)
        self._data: list[Optional[Tensor]] = [None] * self.num_slots
        self._scale: list[Optional[Tensor]] = [None] * self.num_slots
        self.transfer_seconds = 0.0

    @property
    def bytes_per_view(self) -> int:
        """Stored bytes of one accumulator, its scales included."""
        G, N = self.full_shape[:2]
        return int(np.prod(self.stored_shape)) * 2 + G * N * 4

    @property
    def total_bytes(self) -> int:
        """Stored bytes of all accumulators."""
        return self.bytes_per_view * self.num_slots

    def _sync(self) -> None:
        """Wait for the queued work of the device when it is a CUDA device."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def put(self, slot: int, accumulator: Tensor) -> None:
        """Store an accumulator in a slot as its cell mean in scaled float16.

        Args:
            slot (int): slot (view) index in [0, num_slots).
            accumulator (Tensor): [G, N, 4, Rc, Cc, Sc] accumulator.

        Raises:
            ValueError: if the shape of ``accumulator`` is not ``full_shape``.
        """
        if tuple(accumulator.shape) != self.full_shape:
            raise ValueError(f"accumulator must be {self.full_shape}, got {tuple(accumulator.shape)}")
        self._sync()
        t0 = time.perf_counter()
        cell_mean = accumulator.detach()[:, :, :1].float()  # linear-discontinuous cell mean
        maximum = cell_mean.abs().amax(dim=(2, 3, 4, 5), keepdim=True)
        scale = torch.where(maximum > 0, maximum / _FLOAT16_TARGET, torch.ones_like(maximum))
        quantized = (cell_mean / scale).to(torch.float16)
        if self.location == "gpu":
            self._data[slot] = quantized.to(self.device).contiguous()
            self._scale[slot] = scale.to(self.device)
        else:
            buffer = self._data[slot]
            if buffer is None:
                buffer = torch.empty(self.stored_shape, dtype=torch.float16, pin_memory=self.pinned)
                self._data[slot] = buffer
            buffer.copy_(quantized)
            self._scale[slot] = scale.cpu()
        self._sync()
        self.transfer_seconds += time.perf_counter() - t0

    def get(self, slot: int) -> Optional[Tensor]:
        """Decoded accumulator of a slot, or None before its first ``put``.

        Args:
            slot (int): slot (view) index in [0, num_slots).

        Returns:
            Tensor or None: [G, N, 4, Rc, Cc, Sc] float32 accumulator on ``device`` with zero linear-discontinuous
                slopes.
        """
        if self._data[slot] is None:
            return None
        self._sync()
        t0 = time.perf_counter()
        cell_mean = (self._data[slot].to(self.device, non_blocking=self.pinned).float()
                     * self._scale[slot].to(self.device))
        full = torch.zeros(self.full_shape, device=self.device, dtype=torch.float32)
        full[:, :, :1] = cell_mean
        self._sync()
        self.transfer_seconds += time.perf_counter() - t0
        return full

    def close(self) -> None:
        """Release the stored accumulators."""
        self._data = [None] * self.num_slots
        self._scale = [None] * self.num_slots

    def describe(self) -> dict:
        """Description of the storage for the diagnostics.

        Returns:
            dict: ``location``, ``pinned``, ``stored_shape``, ``full_shape``, ``bytes_per_view``, ``num_views`` and
                ``total_gib``.
        """
        return {"location": self.location, "pinned": self.pinned, "stored_shape": list(self.stored_shape),
                "full_shape": list(self.full_shape), "bytes_per_view": self.bytes_per_view,
                "num_views": self.num_slots, "total_gib": self.total_bytes / 2 ** 30}


# =====================================================================================================
# configuration and regularization
# =====================================================================================================

@dataclass
class CitrusConfig:
    """Parameters of the CITRUS reconstruction loop (the scatter solver has its own ``SolverConfig``).

    The options are described in docs/source/method.rst, "Options and secondary components".

    Attributes:
        num_outer_iterations (int): number N of outer iterations; each one is a scatter correction followed by
            ``mbir_iterations_per_outer_iteration`` MBIR iterations. Defaults to 10.
        mbir_iterations_per_outer_iteration (int): number n of warm-started MBIR iterations on the corrected counts of
            each outer iteration. Defaults to 1.
        energy_kev (float or None): effective energy E_eff in keV of the density model; None uses the spectrum mean of
            the solver's library. Defaults to None.
        min_counts (float): lower clamp of the (corrected) counts before the log. Defaults to 1.0.
        base_seed (int): numpy's global random generator is seeded with ``[base_seed, i]`` before MBIR iteration i
            (counted over the whole loop). Defaults to 0.
        gpu_state_budget_mib (float): largest total in MiB of the per-view accumulators kept on the GPU
            (``StateStore``). Defaults to 2600.0.
        pinned_memory_limit_gib (float): largest pinned host allocation in GiB for the accumulators. Defaults to 8.0.
        verbose (bool): print one line per outer iteration. Defaults to False.
        scatter_view_stride (int): the scatter step runs on the views 0, n, 2n, ... for stride n; the other views'
            scatter is interpolated in angle. Defaults to 1.
        scatter_view_interpolation_order (int or None): interpolation in angle of the skipped views' scatter: 3
            Catmull-Rom, 2 quadratic, 1 linear, 0 nearest, None none (diagnostics). Defaults to 3.
        primary_views_per_chunk (int or None): views per chunk of ``primary_signals``; None keeps the path lengths
            within one sinogram. Defaults to None.
        resume_error_sinogram (bool): keep MBIRTORCH's error sinogram between outer iterations instead of recomputing
            it. Defaults to True.
        grouping_object_size (str or float): object size in cm of the solver's ``"scattered_fluence"`` energy groups;
            ``"auto"`` measures it on the initial reconstruction. Defaults to "auto".
    """

    num_outer_iterations: int = 10
    mbir_iterations_per_outer_iteration: int = 1
    energy_kev: Optional[float] = None
    min_counts: float = 1.0
    base_seed: int = 0
    gpu_state_budget_mib: float = 2600.0
    pinned_memory_limit_gib: float = 8.0
    verbose: bool = False
    scatter_view_stride: int = 1
    scatter_view_interpolation_order: Literal[0, 1, 2, 3] | None = 3
    primary_views_per_chunk: Optional[int] = None
    resume_error_sinogram: bool = True
    grouping_object_size: Union[Literal["auto"], float] = "auto"

    def validate(self) -> None:
        """Check the settings.

        Raises:
            ValueError: if a setting is out of range or of the wrong type.
        """
        if self.num_outer_iterations < 1:
            raise ValueError("num_outer_iterations must be >= 1")
        n = self.mbir_iterations_per_outer_iteration
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError(f"mbir_iterations_per_outer_iteration must be an integer >= 1, got {n!r}")
        if self.min_counts <= 0:
            raise ValueError("min_counts must be > 0")
        stride = self.scatter_view_stride
        if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)) or stride < 1:
            raise ValueError(f"scatter_view_stride must be an integer >= 1, got {stride!r}")
        _check_interpolation_order(self.scatter_view_interpolation_order)
        chunk = self.primary_views_per_chunk
        if chunk is not None and (isinstance(chunk, bool) or not isinstance(chunk, (int, np.integer)) or chunk < 1):
            raise ValueError(f"primary_views_per_chunk must be None or an integer >= 1, got {chunk!r}")
        if not isinstance(self.resume_error_sinogram, (bool, np.bool_)):
            raise ValueError(f"resume_error_sinogram must be a bool, got {self.resume_error_sinogram!r}")
        size = self.grouping_object_size
        if not (size == "auto" or (isinstance(size, (int, float, np.floating, np.integer))
                                   and not isinstance(size, bool) and float(size) > 0)):
            raise ValueError(f"grouping_object_size must be 'auto' or a positive number (cm), got {size!r}")


# =====================================================================================================
# sparse-view scatter: angular interpolation of the skipped views
# =====================================================================================================

SCATTER_VIEW_INTERPOLATION_ORDERS = (None, 0, 1, 2, 3)
_TWO_PI = 2.0 * np.pi


def _check_interpolation_order(order, allow_none: bool = True) -> None:
    """Check a scatter-view interpolation order.

    Args:
        order (int or None): the order, one of ``SCATTER_VIEW_INTERPOLATION_ORDERS``.
        allow_none (bool, optional): accept None. Defaults to True.

    Raises:
        ValueError: if ``order`` is not allowed.
    """
    ok = (order is None and allow_none) or (
        not isinstance(order, bool) and isinstance(order, (int, np.integer)) and int(order) in (0, 1, 2, 3))
    if not ok:
        allowed = SCATTER_VIEW_INTERPOLATION_ORDERS if allow_none else SCATTER_VIEW_INTERPOLATION_ORDERS[1:]
        raise ValueError(f"scatter_view_interpolation_order must be one of {allowed}, got {order!r}")


def scatter_view_indices(num_views: int, stride: int) -> np.ndarray:
    """Views that get a scatter step at a view stride: 0, stride, 2 stride, ... below ``num_views``.

    Args:
        num_views (int): number of views V.
        stride (int): view stride, at least 1.

    Returns:
        ndarray: [Vc] indices of the computed views.

    Raises:
        ValueError: if ``num_views`` or ``stride`` is below 1.
    """
    if int(stride) < 1 or int(num_views) < 1:
        raise ValueError(f"need num_views >= 1 and stride >= 1, got {num_views}, {stride}")
    return np.arange(0, int(num_views), int(stride))


def angular_neighbors(computed_angles: Union[Sequence[float], np.ndarray], angle: float,
                      order: Literal[0, 1, 2, 3] = 3, circular: bool = True) -> list[tuple[int, float]]:
    """Interpolation neighbors and weights of a view at ``angle`` from the views at ``computed_angles``.

    Order 0 takes the nearest computed view, 1 is linear between the two bracketing views, 2 is the three-point
    Lagrange interpolant on the nearest view and its two neighbors and 3 is Catmull-Rom on the four nearest views; with
    fewer than ``order + 1`` computed views the order is lowered.

    Args:
        computed_angles (Sequence[float] or ndarray): [Vc] angles in rad of the computed views, in any order.
        angle (float): angle in rad of the view to interpolate.
        order (int, optional): interpolation order 0, 1, 2 or 3. Defaults to 3.
        circular (bool, optional): True for a full rotation (period 2 pi); False interpolates along the arc of a short
            scan and holds the end views outside it. Defaults to True.

    Returns:
        list[tuple[int, float]]: (index into ``computed_angles``, weight) pairs; the weights sum to 1.

    Raises:
        ValueError: if ``order`` is not 0, 1, 2 or 3, or there are no computed views.
    """
    _check_interpolation_order(order, allow_none=False)
    order = int(order)
    computed_array = np.asarray(computed_angles, dtype=np.float64).reshape(-1)
    if circular:
        computed_array = np.mod(computed_array, _TWO_PI)
    Vc = computed_array.size
    if Vc == 0:
        raise ValueError("need at least one computed view")
    if Vc == 1:
        return [(0, 1.0)]
    order = min(order, Vc - 1)
    sort_order = np.argsort(computed_array, kind="stable")
    sorted_angles = computed_array[sort_order]
    if not circular:
        a = float(angle)
        if a <= sorted_angles[0] or a >= sorted_angles[-1]:
            return [(int(sort_order[0 if a <= sorted_angles[0] else Vc - 1]), 1.0)]

        def at(i: int) -> float:  # angle of the i-th sorted computed view (clamped to the arc)
            return float(sorted_angles[min(max(i, 0), Vc - 1)])
    else:
        def at(i: int) -> float:  # unwrapped angle of the i-th sorted computed view (i may leave [0, Vc))
            return float(sorted_angles[i % Vc]) + _TWO_PI * (i // Vc)

        a = float(np.mod(float(angle), _TWO_PI))
    # previous computed view; -1: before the first (wraps)
    j0 = int(np.searchsorted(sorted_angles, a, side="right")) - 1
    j1 = j0 + 1
    a0, a1 = at(j0), at(j1)
    t = (a - a0) / (a1 - a0) if a1 - a0 > 1e-15 else 0.0
    if order == 0:
        pairs = [(j0 if t <= 0.5 else j1, 1.0)]
    elif order == 1:
        pairs = [(j0, 1.0 - t), (j1, t)]
    elif order == 2:
        m = j0 if t <= 0.5 else j1
        if not circular:
            m = min(max(m, 1), Vc - 2)
        nodes = (m - 1, m, m + 1)
        x = [at(i) for i in nodes]
        pairs = []
        for i, xi in zip(nodes, x):
            w = 1.0
            for xk in x:
                if xk != xi:
                    w *= (a - xk) / (xi - xk)
            pairs.append((i, w))
    else:
        t2, t3 = t * t, t * t * t
        pairs = [(j0 - 1, 0.5 * (-t3 + 2.0 * t2 - t)), (j0, 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0)),
                 (j1, 0.5 * (-3.0 * t3 + 4.0 * t2 + t)), (j1 + 1, 0.5 * (t3 - t2))]
    if not circular:  # Catmull-Rom ghost nodes beyond the ends: linear extrapolation (exact for linear functions)
        extended = []
        for i, w in pairs:
            if i < 0:
                extended += [(0, 2.0 * w), (1, -w)]
            elif i > Vc - 1:
                extended += [(Vc - 1, 2.0 * w), (Vc - 2, -w)]
            else:
                extended.append((i, w))
        pairs = extended
    merged: dict[int, float] = {}
    for i, w in pairs:
        k = int(sort_order[i % Vc])
        merged[k] = merged.get(k, 0.0) + float(w)
    return list(merged.items())


def full_rotation(angles: Union[Sequence[float], np.ndarray]) -> bool:
    """Whether the views cover the circle: their largest circular gap is at most 1.5 times the median gap.

    Args:
        angles (Sequence[float] or ndarray): [V] view angles in rad.

    Returns:
        bool: True for a full rotation, False for a short scan.
    """
    a = np.sort(np.mod(np.asarray(angles, dtype=np.float64).reshape(-1), _TWO_PI))
    if a.size < 3:
        return a.size == 2 and abs(float(a[1] - a[0]) - np.pi) < 1e-9
    gaps = np.diff(np.concatenate([a, a[:1] + _TWO_PI]))
    return bool(gaps.max() <= 1.5 * np.median(gaps) + 1e-12)


def _unwrap_arc(angles: np.ndarray) -> np.ndarray:
    """Angles of a short scan unwrapped along its arc, which starts after the largest circular gap.

    Args:
        angles (ndarray): [V] view angles in rad.

    Returns:
        ndarray: [V] unwrapped angles in rad.
    """
    a = np.mod(angles, _TWO_PI)
    srt = np.sort(a)
    gaps = np.diff(np.concatenate([srt, srt[:1] + _TWO_PI]))
    origin = float(srt[(int(np.argmax(gaps)) + 1) % srt.size])
    return origin + np.mod(a - origin, _TWO_PI)


def interpolation_weights(angles: Union[Sequence[float], np.ndarray], computed: Union[Sequence[int], np.ndarray],
                          order: Literal[0, 1, 2, 3] | None = 3, circular: Optional[bool] = None) -> np.ndarray:
    """Interpolation matrix ``M`` with ``s_all = M s_computed`` from the ``angular_neighbors`` weights.

    The rows of the computed views are one-hot; with order None the rows of the skipped views are zero.

    Args:
        angles (Sequence[float] or ndarray): [V] view angles in rad.
        computed (Sequence[int] or ndarray): [Vc] distinct indices of the computed views into ``angles``.
        order (int or None, optional): interpolation order 0 to 3, or None for no interpolation. Defaults to 3.
        circular (bool, optional): interpolate circularly in angle; False interpolates along the arc of a short scan.
            Defaults to None (``full_rotation(angles)``).

    Returns:
        ndarray: [V, Vc] interpolation weights.

    Raises:
        ValueError: if ``computed`` holds repeated or out-of-range indices, or ``order`` is not allowed.
    """
    angle_array = np.asarray(angles, dtype=np.float64).reshape(-1)
    computed_index = np.asarray(computed, dtype=np.int64).reshape(-1)
    if (computed_index.size == 0 or computed_index.min() < 0 or computed_index.max() >= angle_array.size
            or np.unique(computed_index).size != computed_index.size):
        raise ValueError("computed must be distinct view indices into angles")
    _check_interpolation_order(order)
    M = np.zeros((angle_array.size, computed_index.size), dtype=np.float64)
    M[computed_index, np.arange(computed_index.size)] = 1.0
    if order is not None:
        is_computed = np.zeros(angle_array.size, dtype=bool)
        is_computed[computed_index] = True
        is_circular = full_rotation(angle_array) if circular is None else bool(circular)
        arc = angle_array if is_circular else _unwrap_arc(angle_array)
        computed_arc = arc[computed_index]
        for v in np.flatnonzero(~is_computed):
            for j, w in angular_neighbors(computed_arc, float(arc[v]), order, circular=is_circular):
                M[v, j] += w
    return M


def interpolate_scatter(computed_scatter: Tensor, angles: Union[Sequence[float], np.ndarray],
                        computed: Union[Sequence[int], np.ndarray],
                        order: Literal[0, 1, 2, 3] | None = 3,
                        previous: Optional[Tensor] = None, weights: Optional[Tensor] = None) -> Tensor:
    """Scatter of every view from the scatter of the computed views, interpolated in angle per pixel.

    The computed views keep their values exactly; a skipped view v gets ``max(0, sum_j w_vj s_j)``.

    Args:
        computed_scatter (Tensor): [Vc, ...] per-view images of the computed views (any units).
        angles (Sequence[float] or ndarray): [V] view angles in rad.
        computed (Sequence[int] or ndarray): [Vc] indices of the computed views into ``angles``.
        order (int or None, optional): interpolation order 0 to 3; None keeps the skipped views' values of
            ``previous``, or zero. Defaults to 3.
        previous (Tensor, optional): [V, ...] most recent scatter of every view, used by order None. Defaults to None.
        weights (Tensor, optional): [V, Vc] precomputed ``interpolation_weights(angles, computed, order)``.
            Defaults to None (computed here).

    Returns:
        Tensor: [V, ...] scatter of every view.

    Raises:
        ValueError: if the shapes of ``computed_scatter``, ``previous`` or ``weights`` do not match the views.
    """
    angle_array = np.asarray(angles, dtype=np.float64).reshape(-1)
    computed_index = np.asarray(computed, dtype=np.int64).reshape(-1)
    V, Vc = angle_array.size, computed_index.size
    if computed_scatter.shape[0] != Vc:
        raise ValueError(f"computed_scatter must have one image per computed view ({Vc}), "
                         f"got {tuple(computed_scatter.shape)}")
    computed_index_tensor = torch.as_tensor(computed_index, device=computed_scatter.device)
    _check_interpolation_order(order)
    if order is None:
        if previous is not None:
            if previous.shape[0] != V or previous.shape[1:] != computed_scatter.shape[1:]:
                raise ValueError(f"previous must be [{V}, *{tuple(computed_scatter.shape[1:])}], "
                                 f"got {tuple(previous.shape)}")
            out = previous.to(computed_scatter.device, computed_scatter.dtype).clone()
        else:
            out = torch.zeros((V,) + tuple(computed_scatter.shape[1:]), device=computed_scatter.device,
                              dtype=computed_scatter.dtype)
        out[computed_index_tensor] = computed_scatter
        return out
    if weights is None:
        weights = torch.as_tensor(interpolation_weights(angle_array, computed_index, order),
                                  device=computed_scatter.device, dtype=computed_scatter.dtype)
    if tuple(weights.shape) != (V, Vc):
        raise ValueError(f"weights must be [{V}, {Vc}], got {tuple(weights.shape)}")
    out = (weights.to(computed_scatter.device, computed_scatter.dtype)
           @ computed_scatter.reshape(Vc, -1)).reshape((V,) + tuple(computed_scatter.shape[1:]))
    out.clamp_min_(0.0)
    out[computed_index_tensor] = computed_scatter  # exact (and unclamped) on the computed views
    return out


def estimate_regularization(model, sinogram: Tensor, weights: Optional[Tensor] = None) -> dict:
    """MBIRTORCH's automatic regularization of a sinogram, then frozen in the model.

    Args:
        model (ConeBeamModel): MBIRTORCH model.
        sinogram (Tensor): [V, Rd, Cd] line integrals.
        weights (Tensor, optional): [V, Rd, Cd] weights. Defaults to None.

    Returns:
        dict: the frozen ``sigma_y``, ``sigma_x`` and ``sigma_prox``.
    """
    model.set_params(no_warning=True, auto_regularize_flag=True)
    model.auto_set_regularization_params(sinogram, weights=weights)
    model.set_params(auto_regularize_flag=False)
    return {k: float(v) for k, v in zip(REGULARIZATION_NAMES, model.get_params(list(REGULARIZATION_NAMES)))}


def apply_regularization(model, regularization: dict) -> None:
    """Set a frozen regularization in the model (``auto_regularize_flag`` False).

    Args:
        model (ConeBeamModel): MBIRTORCH model.
        regularization (dict): ``sigma_y``, ``sigma_x`` and ``sigma_prox``.
    """
    model.set_params(no_warning=True, auto_regularize_flag=False,
                     **{k: float(regularization[k]) for k in REGULARIZATION_NAMES})


# =====================================================================================================
# reconstruction loop
# =====================================================================================================

ScatterStage = Callable[[int, Tensor, dict], Tensor]


def _sync(device: torch.device) -> None:
    """Wait for the queued work of a CUDA device (nothing on another device).

    Args:
        device (torch.device): the device.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _outer_loop(Y: Tensor, I0: float, angles: np.ndarray, grid: Grid, geometry: ConeGeometry, config: CitrusConfig,
                scatter_stage: Optional[ScatterStage], device: torch.device, *, regularization: Optional[dict],
                init_recon: Optional[Tensor], model, callback, label: str,
                after_initialization: Optional[Callable[[Tensor, dict], None]] = None) -> tuple[Tensor, dict]:
    """Outer loop of ``citrus_recon`` and ``mbir_recon``: a scatter stage, then n MBIR iterations, per outer iteration.

    X_0 is the initial volume; outer iteration k = 1, ..., N corrects the counts with the scatter of X_{k-1} and runs
    ``config.mbir_iterations_per_outer_iteration`` warm-started MBIR iterations on them, which give X_k.

    Args:
        Y (Tensor): [V, Rd, Cd] measured counts.
        I0 (float): flat-field counts per pixel.
        angles (ndarray): [V] view angles in rad.
        grid (Grid): reconstruction grid.
        geometry (ConeGeometry): cone-beam geometry of the scan.
        config (CitrusConfig): loop settings.
        scatter_stage (callable or None): ``scatter_stage(k, X, record)`` returns the scatter-to-primary ratio
            [V, Rd, Cd] of the volume X at the start of outer iteration k + 1 (k = 0, 1, ...); None for no correction.
        device (torch.device): device of the loop.
        regularization (dict or None): frozen regularization; None estimates it on the first corrected sinogram.
        init_recon (Tensor or None): [R, C, S] initial volume in 1/cm; None uses FDK of the uncorrected data.
        model (ConeBeamModel or None): prebuilt MBIRTORCH model; None builds it.
        callback (callable or None): ``callback(k, X_k, record)`` for the initial volume (k = 0) and after every
            outer iteration k = 1, ..., N.
        label (str): prefix of the verbose lines.
        after_initialization (callable, optional): ``after_initialization(X_0, info)``, called once after the initial
            volume. Defaults to None.

    Returns:
        (Tensor, dict): reconstruction [R, C, S] in 1/cm and the info dict.
    """
    config.validate()
    weights = Y / float(I0)
    measured_line_integrals = -torch.log(Y.clamp_min(config.min_counts) / float(I0))
    if model is None:
        model = make_cone_model(grid, geometry, angles, device=device)
    t0 = time.perf_counter()
    if init_recon is None:
        X = fdk(model, measured_line_integrals, geometry=geometry, angles=angles)  # Parker-weighted for a short scan
    else:
        X = torch.as_tensor(init_recon, dtype=torch.float32, device=device).clone()
    _sync(device)
    info: dict = {"records": [], "fdk_seconds": time.perf_counter() - t0, "init_recon": X.clone(), "model": model,
                  "config": asdict(config)}
    if after_initialization is not None:
        after_initialization(X, info)
    regularization = dict(regularization) if regularization is not None else None
    if regularization is not None:
        apply_regularization(model, regularization)
    initial_record = {"outer_iteration": 0, "scatter_seconds": 0.0, "mbir_seconds": 0.0, "total_seconds": 0.0}
    info["records"].append(initial_record)
    if callback is not None:
        callback(0, X, initial_record)
    spr: Optional[Tensor] = None
    previous_scatter_estimate: Optional[Tensor] = None
    scatter_estimate = torch.zeros_like(Y)
    resume = bool(config.resume_error_sinogram)
    # resume: MBIRTORCH's error sinogram minus the corrected data, -A X (one sinogram)
    negative_projection: Optional[Tensor] = None
    t_start = time.perf_counter()
    n = int(config.mbir_iterations_per_outer_iteration)
    for k in range(config.num_outer_iterations):
        record: dict = {"outer_iteration": k + 1, "mbir_iterations": n}
        _sync(device)
        t0 = time.perf_counter()
        if scatter_stage is not None:
            spr = scatter_stage(k, X, record)
        corrected_counts = Y if spr is None else Y / (1.0 + spr)
        corrected_line_integrals = -torch.log(corrected_counts.clamp_min(config.min_counts) / float(I0))
        if regularization is None:  # estimated once on the first sinogram MBIR inverts, then frozen
            regularization = estimate_regularization(model, corrected_line_integrals, weights)
        _sync(device)
        t1 = time.perf_counter()
        for j in range(n):
            iteration = k * n + j  # MBIR iteration count over the whole loop: the seed and the schedule position
            np.random.seed([int(config.base_seed), int(iteration)])
            if resume:
                # e = y_c - A X: the kept -A X shifted by the current corrected data (A X is unchanged by the scatter
                # stage); MBIRTORCH updates e and its init volume in place, so it gets a copy of X
                error_sinogram = (None if negative_projection is None
                                  else negative_projection.add_(corrected_line_integrals))
                negative_projection = None
                X, _, checkpoint = mbir_iterations(model, corrected_line_integrals, X.clone(),
                                                   first_iteration=iteration, weights=weights,
                                                   error_sinogram=error_sinogram, return_checkpoint=True)
                negative_projection = checkpoint["error_sinogram"].sub_(corrected_line_integrals)
                del error_sinogram, checkpoint
            else:
                X, _ = mbir_iterations(model, corrected_line_integrals, X, first_iteration=iteration, weights=weights)
        _sync(device)
        t2 = time.perf_counter()
        scatter_estimate = Y - corrected_counts
        if previous_scatter_estimate is not None:
            denominator = float(scatter_estimate.abs().mean())
            change = float((scatter_estimate - previous_scatter_estimate).abs().mean())
            record["scatter_change_relative"] = change / denominator if denominator > 0 else 0.0
        previous_scatter_estimate = scatter_estimate
        record.update(scatter_seconds=t1 - t0, mbir_seconds=t2 - t1, total_seconds=t2 - t0)
        if device.type == "cuda":
            record["gpu_peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2 ** 20
            torch.cuda.reset_peak_memory_stats(device)
        info["records"].append(record)
        if callback is not None:
            callback(k + 1, X, record)
        if config.verbose:
            extra = (f" dPhi {record['accumulator_change_mean']:.2e}" if "accumulator_change_mean" in record
                     else "")
            print(f"[{label}] outer iteration {k + 1}/{config.num_outer_iterations}: scatter "
                  f"{record['scatter_seconds']:.1f} s, MBIR ({n} iteration{'s' if n > 1 else ''}) "
                  f"{record['mbir_seconds']:.1f} s{extra}", flush=True)
    info.update(seconds=time.perf_counter() - t_start, regularization=regularization, spr=spr,
                scatter_counts=scatter_estimate)
    return X, info


def _counts_and_angles(sinogram_counts, angles, device) -> tuple[Tensor, np.ndarray]:
    """Measured counts as a float32 tensor and the view angles as a float64 array.

    Args:
        sinogram_counts (Tensor or ndarray): [V, Rd, Cd] measured counts.
        angles (Sequence[float], ndarray or Tensor): [V] view angles in rad.
        device (torch.device): device of the counts.

    Returns:
        (Tensor, ndarray): counts [V, Rd, Cd] and angles [V].

    Raises:
        ValueError: if the counts are not [V, rows, channels] for the V angles.
    """
    Y = torch.as_tensor(sinogram_counts, dtype=torch.float32, device=device).contiguous()
    if isinstance(angles, Tensor):
        angles = angles.detach().cpu().numpy()
    angle_array = np.asarray(angles, dtype=np.float64).reshape(-1)
    if Y.dim() != 3 or Y.shape[0] != angle_array.size:
        raise ValueError(f"sinogram_counts must be [V = {angle_array.size}, rows, channels], got {tuple(Y.shape)}")
    return Y, angle_array


def _apply_grouping_estimates(solver: ScatterSolver, X: Tensor, grid: Grid, config: CitrusConfig) -> dict:
    """Set the solver's energy-group object size and medium from the initial volume.

    This applies to the ``"scattered_fluence"`` group rule without explicit edges, before the solver's library is built.

    Args:
        solver (ScatterSolver): the scatter solver; its ``config`` is replaced.
        X (Tensor): [R, C, S] initial volume in 1/cm.
        grid (Grid): reconstruction grid.
        config (CitrusConfig): loop settings (``grouping_object_size``, ``energy_kev``, ``verbose``).

    Returns:
        dict: ``grouping_object_size_cm``, ``grouping_medium`` and ``grouping_applied`` (whether the solver's config
            changed).
    """
    solver_config = solver.config
    if solver_config.group_edges_method != "scattered_fluence" or solver_config.group_edges_kev is not None:
        return {"grouping_object_size_cm": None, "grouping_medium": None, "grouping_applied": False}
    current_size = float(solver_config.grouping_object_size_cm)
    requested = config.grouping_object_size
    size = float(requested) if requested != "auto" else None
    medium = solver_config.grouping_medium
    if requested == "auto" or medium == "auto":
        mask = object_mask(X, solver, energy_kev=config.energy_kev)
        if requested == "auto":
            size = object_size_cm(X, grid, solver, energy_kev=config.energy_kev, default=current_size)
        if medium == "auto":
            medium = grouping_medium_estimate(X, solver, energy_kev=config.energy_kev, mask=mask) or "auto"
    if solver.library_built:
        if config.verbose and (size != current_size or medium != solver_config.grouping_medium):
            print(f"[citrus] grouping object size {size:g} cm / medium {medium} not applied: the solver's library is "
                  f"already built for {current_size:g} cm / {solver_config.grouping_medium}", flush=True)
        return {"grouping_object_size_cm": current_size, "grouping_medium": solver_config.grouping_medium,
                "grouping_applied": False}
    solver.config = replace(solver_config, grouping_object_size_cm=size, grouping_medium=medium)
    if config.verbose:
        print(f"[citrus] energy groups for an object of {size:g} cm made of {_medium_text(medium)}", flush=True)
    return {"grouping_object_size_cm": size, "grouping_medium": solver.config.grouping_medium,
            "grouping_applied": True}


def _medium_text(medium) -> str:
    """Printable text of a grouping medium.

    Args:
        medium (str or Sequence[tuple[str, float]]): a medium name, or (material name, density in g/cm^3) pairs.

    Returns:
        str: the text.
    """
    if isinstance(medium, str):
        return medium
    return " + ".join(f"{name} {density:.3g} g/cm^3" for name, density in medium)


def citrus_recon(sinogram_counts: Union[Tensor, np.ndarray], I0: float, angles: Union[Sequence[float], np.ndarray],
                 grid: Grid, geometry: ConeGeometry, solver: ScatterSolver, config: Optional[CitrusConfig] = None,
                 *, regularization: Optional[dict] = None, init_recon: Optional[Tensor] = None, model=None,
                 callback: Optional[Callable[[int, Tensor, dict], None]] = None) -> tuple[Tensor, dict]:
    """CITRUS reconstruction of measured counts: MBIR with the scatter correction of the solver in the loop.

    The loop is described in docs/source/method.rst, "The CITRUS loop", and the info entries in docs/source/usage.rst,
    "CITRUS".

    Args:
        sinogram_counts (Tensor or ndarray): [V, Rd, Cd] measured counts Y (primary plus scatter).
        I0 (float): flat-field counts per pixel.
        angles (Sequence[float] or ndarray): [V] view angles in rad.
        grid (Grid): reconstruction grid (MBIRTORCH conventions), also the fine grid of the solver.
        geometry (ConeGeometry): cone-beam geometry of the scan.
        solver (ScatterSolver): scatter solver of the scanner; its device is used for everything.
        config (CitrusConfig, optional): loop settings. Defaults to None (``CitrusConfig()``).
        regularization (dict, optional): frozen ``sigma_y``, ``sigma_x`` and ``sigma_prox``. Defaults to None
            (estimated once on the first corrected sinogram).
        init_recon (Tensor, optional): [R, C, S] initial volume in 1/cm. Defaults to None (FDK of the uncorrected data).
        model (ConeBeamModel, optional): prebuilt MBIRTORCH model (``recon.mbir_bridge.make_cone_model``). Defaults to
            None.
        callback (callable, optional): ``callback(k, X_k, record)``, called for the initial volume (k = 0, the FDK)
            and after every outer iteration k = 1, ..., N. Defaults to None.

    Returns:
        (Tensor, dict): reconstruction [R, C, S] float32 in 1/cm on the solver's device, and the info dict (``records``
            per outer iteration, ``spr``, ``scatter_counts``, ``regularization``, ``computed_views``, ...).
    """
    config = CitrusConfig() if config is None else config
    config.validate()
    device = solver.device
    Y, angle_array = _counts_and_angles(sinogram_counts, angles, device)
    V = len(angle_array)
    # the density model needs the library: made in the first scatter stage, after the object size is set
    density_models: list[Optional[DensityModel]] = [None]
    library_seconds: list[float] = [0.0]
    transport_grid = solver.transport_grid(grid)
    if model is None:
        model = make_cone_model(grid, geometry, angle_array, device=device)
    projector = ViewChunkProjector(grid, geometry, angle_array, model=model, device=device)
    full_shape = (solver.num_groups, (solver.L + 1) ** 2, 4) + tuple(transport_grid.shape)
    computed = scatter_view_indices(V, config.scatter_view_stride)
    sparse = len(computed) < V
    interpolation_order = config.scatter_view_interpolation_order if sparse else None
    skipped = np.setdiff1d(np.arange(V), computed)
    weights = None
    view_interpolated = np.zeros(V, dtype=bool)
    if sparse and interpolation_order is not None:
        weights = torch.as_tensor(interpolation_weights(angle_array, computed, interpolation_order), device=device,
                                  dtype=torch.float32)
        view_interpolated[skipped] = True
    state_store = StateStore(len(computed), full_shape, device, gpu_budget_mib=config.gpu_state_budget_mib,
                             pinned_memory_limit_gib=config.pinned_memory_limit_gib)
    view_states = [CitrusState() for _ in computed]  # one per computed view (the age-ordered recursion)
    Rd, Cd = int(Y.shape[1]), int(Y.shape[2])
    tiny = torch.finfo(torch.float32).tiny
    # every view's last scatter estimate (interpolation order None keeps it on the skipped views)
    last_scatter: list[Optional[Tensor]] = [None]

    def after_initialization(X: Tensor, info: dict) -> None:
        info.update(_apply_grouping_estimates(solver, X, grid, config))

    def scatter_stage(k: int, X: Tensor, record: dict) -> Tensor:
        if density_models[0] is None:
            built = solver.library_built
            t0 = time.perf_counter()
            density_models[0] = DensityModel.from_library(solver.library, config.energy_kev)
            library_seconds[0] = time.perf_counter() - t0
            if config.verbose and not built:
                print(f"[citrus] library built for grouping_object_size_cm = "
                      f"{solver.config.grouping_object_size_cm:g} in {library_seconds[0]:.1f} s", flush=True)
        density_model = density_models[0]
        transfer_seconds_before = state_store.transfer_seconds
        record["negative_fraction"] = density_model.negative_fraction(X)
        # shared by every view of this outer iteration; built material by material, no [M, R, C, S] volume
        fields = solver.material_fields_from_attenuation(X, grid, density_model)
        computed_scatter = torch.empty((len(computed), Rd, Cd), device=device, dtype=torch.float32)
        primary = torch.empty((V, Rd, Cd), device=device, dtype=torch.float32)
        changes = []
        for j, v in enumerate(computed):
            state = replace(view_states[j], collided_moments=state_store.get(j))
            computed_scatter[j], _, new_state = citrus_view_step(solver, state, fields, float(angle_array[v]),
                                                                 primary=False)
            state_store.put(j, new_state.collided_moments)
            view_states[j] = replace(new_state, collided_moments=None)
            changes.append(new_state.history[-1])
            del state, new_state
        record["accumulator_change_mean"] = float(np.mean(changes))
        record["accumulator_change_max"] = float(np.max(changes))
        record["transfer_seconds"] = state_store.transfer_seconds - transfer_seconds_before
        _sync(device)
        t0 = time.perf_counter()
        # every view
        primary_signals(solver, fields, angle_array, projector, config.primary_views_per_chunk, out=primary)
        _sync(device)
        record["primary_seconds"] = time.perf_counter() - t0
        if not sparse:
            scatter = computed_scatter
        else:
            _sync(device)
            t0 = time.perf_counter()
            # the scatter SIGNAL is interpolated in angle; the primary of every view is computed (primary_signals)
            scatter = interpolate_scatter(computed_scatter, angle_array, computed, interpolation_order,
                                          previous=last_scatter[0], weights=weights)
            _sync(device)
            record["interpolation_seconds"] = time.perf_counter() - t0
        if sparse and interpolation_order is None:
            last_scatter[0] = scatter
        del fields
        return scatter / primary.clamp_min(tiny)

    try:
        X, info = _outer_loop(Y, float(I0), angle_array, grid, geometry, config, scatter_stage, device,
                              regularization=regularization, init_recon=init_recon, model=model, callback=callback,
                              label="citrus", after_initialization=after_initialization)
        info["library_seconds"] = library_seconds[0]
        info["state"] = state_store.describe()
        info["computed_views"] = computed.copy()
        info["view_interpolated"] = view_interpolated
    finally:
        state_store.close()
    return X, info


def mbir_recon(sinogram_counts: Union[Tensor, np.ndarray], I0: float, angles: Union[Sequence[float], np.ndarray],
               grid: Grid, geometry: ConeGeometry, config: Optional[CitrusConfig] = None, *,
               spr: Optional[Union[Tensor, np.ndarray]] = None, regularization: Optional[dict] = None,
               init_recon: Optional[Tensor] = None, model=None, device=None,
               callback: Optional[Callable[[int, Tensor, dict], None]] = None) -> tuple[Tensor, dict]:
    """The MBIR loop of ``citrus_recon`` without the scatter solver: no correction, or a fixed correction.

    Args:
        sinogram_counts (Tensor or ndarray): [V, Rd, Cd] measured counts Y.
        I0 (float): flat-field counts per pixel.
        angles (Sequence[float] or ndarray): [V] view angles in rad.
        grid (Grid): reconstruction grid.
        geometry (ConeGeometry): cone-beam geometry of the scan.
        config (CitrusConfig, optional): loop settings. Defaults to None (``CitrusConfig()``).
        spr (Tensor or ndarray, optional): [V, Rd, Cd] fixed scatter-to-primary ratio, applied as ``Y / (1 + spr)``.
            Defaults to None (no correction).
        regularization (dict, optional): frozen ``sigma_y``, ``sigma_x`` and ``sigma_prox``. Defaults to None
            (estimated once on the first sinogram).
        init_recon (Tensor, optional): [R, C, S] initial volume in 1/cm. Defaults to None (FDK).
        model (ConeBeamModel, optional): prebuilt MBIRTORCH model. Defaults to None.
        device (torch.device or str, optional): device of the loop. Defaults to None (the device of a tensor sinogram,
            else CUDA if available).
        callback (callable, optional): ``callback(k, X_k, record)`` as in ``citrus_recon``. Defaults to None.

    Returns:
        (Tensor, dict): reconstruction [R, C, S] float32 in 1/cm and the info dict of ``citrus_recon`` without the
            scatter-solver entries.

    Raises:
        ValueError: if ``spr`` does not have the shape of the sinogram.
    """
    config = CitrusConfig() if config is None else config
    if device is None:
        device = sinogram_counts.device if isinstance(sinogram_counts, Tensor) else (
            "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    Y, angle_array = _counts_and_angles(sinogram_counts, angles, device)
    fixed_spr = None if spr is None else torch.as_tensor(spr, dtype=torch.float32, device=device)
    if fixed_spr is not None and fixed_spr.shape != Y.shape:
        raise ValueError(f"spr must have the sinogram shape {tuple(Y.shape)}, got {tuple(fixed_spr.shape)}")
    scatter_stage = None if fixed_spr is None else (lambda k, X, record: fixed_spr)
    return _outer_loop(Y, float(I0), angle_array, grid, geometry, config, scatter_stage, device,
                       regularization=regularization, init_recon=init_recon, model=model, callback=callback,
                       label="mbir")
