"""CITRUS: scatter-corrected MBIR with the LBTE source iteration solver in the loop (docs/source/method.rst).

The algorithm
-------------
Measured counts Y [V, Rd, Cd] (flat field I0) contain primary and scatter, Y ~ Poisson(P + S). CITRUS alternates, per
outer iteration k, ONE scatter step per view and ONE warm-started MBIR iteration:

    X_0      = FDK(-log(Y / I0))                                    (the uncorrected data; Parker-weighted for a
                                                                     short scan)
    for k = 0 .. N-1:
        rho_k            = densities(X_k)                            interpolating model between the library's reference
                                                                     materials at E_eff, resampled onto the transport
                                                                     and quadrature grids material by material (never
                                                                     [M, R, C, S])
        for every view v:
            phi_UC       = G_1(rho_k)                                uncollided fluence on the quadrature grid Q
                                                                     (`uncollided_operator`)
            Phi_v        = phi^(1)(phi_UC) + K Phi_v                 ONE sweep of the first-collision source plus the
                                                                     scattering source of the aged accumulator
                                                                     (`accumulator_update_operator`)
            s_v          = project_scatter(phi_UC, Phi_v)            scatter signal of the view
        p                = primary(rho_k)                            primary signals of every view: MBIRTORCH
                                                                     projections of the material maps (the
                                                                     reconstruction's own operator A)
        SPR_k            = s / p
        Y_c              = Y / (1 + SPR_k)                           scatter-corrected counts
        X_{k+1}          = one VCD iteration of MBIR on y_c = -log(Y_c / I0), warm-started at X_k,
                           at partition-schedule position k, weights W = Y / I0, frozen regularization

(Symbols of docs/source/method.rst: X the reconstruction (1/cm), Y the measured counts, I0 the flat field, P / S the
primary / scatter counts, rho the densities, phi the fluence, G_1 the uncollided operator, phi^(1) the once-collided
fluence moments, K = sweep o scatter the collided transport of one collision order, V views, Rd x Cd detector pixels.)

The per-view accumulator Phi_v = sum_{n >= 1} (collided moments of order n) is carried across outer iterations: with
the densities held fixed, N steps reproduce N source iterations exactly (`citrus_step`), so the scatter estimate
converges together with the reconstruction ("age-ordered" source iteration: the n-th collision order of the current
estimate was computed n - 1 outer iterations ago). The collided transport is the expensive part of a converged
scatter solve; CITRUS spends one sweep per view per outer iteration instead of a full source iteration.

Densities and primary (versions 0.3.0 and 0.5.0): the loop never forms the density volume [M, R, C, S] of the M
reference materials on the reconstruction grid (7 GB at 256 M voxels for the default seven).
`ScatterSolver.material_fields_from_attenuation(X_k, grid, density)` evaluates one material's map at a time
(`DensityModel.material`, bit for bit the row of the full model) and resamples it onto the transport and quadrature
grids. The primary signal of every view is the MBIRTORCH per-material primary: the path lengths P_m = A rho_m
(g/cm^2) of the materials present with MBIRTORCH's projector over all views at once (one material map at a time, in
chunks of views so that the path lengths [M', V_chunk, Rd, Cd] stay within one sinogram, `primary_signals`), then the
solver's polychromatic transmission and detector / grid response per view (`ScatterSolver.primary_from_path_lengths`).
These are pixel-averaged line integrals (separable footprint), the reconstruction's own operator A, about 2.4 ms per
view and material on the 12 M-voxel grid. (Versions 0.3.0 and 0.4.0 offered the solver's Siddon primary, one ray to
each pixel center, as an alternative; the standalone solver keeps it: `ScatterSolver.primary`, `solve_view`.)

Scatter model -> correction: the solver gives per view the scatter and primary signals (s, p) of the current
densities, so the correction is the multiplicative Y_c = Y p / (p + s) = Y / (1 + SPR(X)); it is independent of I0 and
of any mismatch between the solver's polychromatic primary and MBIR's monoenergetic model, and it rescales the noise
of Y (W = Y / I0 is the exact weight of y_c).

Sparse-view scatter (CitrusConfig.scatter_view_stride n > 1): the scatter step runs only on the views 0, n, 2n, ...
(uniform over the angular range, starting at view 0); only those views keep an accumulator. The scatter SIGNAL of every
other view is interpolated in angle, per detector pixel, from the computed views' scatter signals (Catmull-Rom over
the 4 nearest computed views by default, circular in angle for a full rotation, along the arc with the end views held
for a short scan; orders 0-2 and None also exist;
`interpolate_scatter`), its primary signal comes from the primary projection of every view (`primary_signals`; cheap:
no transport), and the correction ratio s / p is formed on every view. The scatter
signal is interpolated, not the ratio: the scatter is smooth on the panel and in angle, the ratio carries the object's
shadow edges, which move across the panel with the angle. Measured on WLAB (180 views, 0.5 cm grid, 20 outer
iterations): stride 2 costs +0.0014 relative RMSE for 51 % of the scatter time, stride 4 +0.008 for 26 %; the penalty
is mostly the slower convergence of the coupled loop on the skipped views, not the interpolation error.

Scatter step every n outer iterations (CitrusConfig.scatter_every n > 1): the scatter stage runs at the outer
iterations 0, n, 2n, ... only; in between, the previous scatter-to-primary ratio is reused and only the MBIR iteration
runs, so n MBIR iterations run per scatter step and the accumulators keep their age across the skipped iterations. An
MBIR iteration is cheap next to a scatter stage (0.75 s against 56 s on the 180-view WLAB study), so for a given
number of scatter steps more MBIR iterations are nearly free. Measured on WLAB (180 views, 0.5 cm grid): at a fixed 20
outer iterations the scatter step at every 2nd / 4th iteration raises the final relative RMSE from 0.0429 to 0.0513 /
0.0700 (the MBIR iterations converge towards a stale correction between refreshes); the same 20 scatter steps spread
over 40 (n = 2) / 80 (n = 4) outer iterations lower it to 0.0310 / 0.0286 in the same wall time, and at 40 iterations
n = 2 is within 0.0008 of a scatter step at every iteration (0.0302) at half the scatter cost. The default is 1.

Per-view state: one accumulator [G, (L+1)^2, 4, Rc, Cc, Sc] per view (G energy groups, (L+1)^2 spherical-harmonic
moments, 4 linear-discontinuous moments, Rc x Cc x Sc transport cells), stored compactly (`StateStore`): only the
linear-discontinuous cell mean (the aged accumulator re-enters the next sweep with zero slopes) in float16 with a
per-(group, moment) scale, 1/8 of float32 (17.6 MiB per view on a 0.5 cm WLAB grid with 1 cm transport cells). The
reconstruction differs from exact float32 storage by ~6e-5 (relative), far below the reconstruction error.

MBIR resume (CitrusConfig.resume_error_sinogram): MBIRTORCH's `recon` recomputes the error sinogram y_c - A X with
one forward projection per call. With the switch on, the loop keeps it between outer iterations (as -A X, one
sinogram) and resumes MBIRTORCH's engine from it (`recon.mbir_bridge.mbir_iterations(..., error_sinogram=)`); a new
corrected sinogram only shifts it, e = y_c,new - A X. The reconstruction equals the restarted one to float rounding.

Public API
----------
    citrus_recon(...)     the CITRUS reconstruction (returns the reconstruction and a dict of diagnostics)
    mbir_recon(...)       the same MBIR loop without the solver: no correction, or a fixed SPR correction
    citrus_step(...)      one scatter step of one view over an explicit CitrusState (densities given)
    citrus_view_step(...) the same from prebuilt MaterialFields, optionally without the primary (the loop's form)
    primary_signals(...)  primary signals of all views from MBIRTORCH projections of the material maps
    scatter_view_indices, angular_neighbors, full_rotation, interpolation_weights, interpolate_scatter
                          sparse-view scatter
    uncollided_operator, first_collision_operator, multiple_collision_operator, accumulator_update_operator,
    project_scatter       the per-view operators of the step
    DensityModel, StateStore, CitrusConfig, estimate_regularization, apply_regularization
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
from .utilities import attenuation_to_densities, attenuation_to_density, effective_energy

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
    """Caller-owned state of the CITRUS scatter recursion for ONE view.

    angle            : view angle (rad) the state belongs to; None until the first step (then fixed).
    collided_moments : [G, (L+1)^2, 4, Rc, Cc, Sc] the accumulator Phi of collided-fluence moments (sum over the
                       collision orders n >= 1), or None (= zero) initially.
    cone_quadrature  : cached collapsed-cone direction set of the view (depends on grid and angle only).
    num_steps        : number of `citrus_step` calls applied.
    history          : relative change ||Phi_new - Phi_old|| / ||Phi_new|| of the scalar-flux cell means per step
                       (the CITRUS analogue of SourceIterationResult.history).
    For several views keep one state per view.
    """

    angle: Optional[float] = None
    collided_moments: Optional[Tensor] = None
    cone_quadrature: Optional[Quadrature] = None
    num_steps: int = 0
    history: list[float] = field(default_factory=list)


def _material_fields_or_build(solver: ScatterSolver, rho: Tensor, grid: Grid,
                              fields: Optional[MaterialFields]) -> MaterialFields:
    return solver.material_fields(rho, grid) if fields is None else fields


def uncollided_operator(solver: ScatterSolver, rho: Tensor, grid: Grid, angle: float,
                        fields: Optional[MaterialFields] = None) -> Tensor:
    """G_1 of docs/source/method.rst: the uncollided fluence phi_uncollided [G, Rq, Cq, Sq] at the quadrature points
    (the voxel centers of the solver's quadrature grid Q = transport grid refined q = cell /
    quadrature_point_spacing_cm times, `ScatterSolver.uncollided_fluence`). rho [M, R, C, S] on the fine grid (M
    library materials), unused when fields are given."""
    return solver.uncollided_fluence(_material_fields_or_build(solver, rho, grid, fields), angle)


def first_collision_operator(solver: ScatterSolver, rho: Tensor, grid: Grid, angle: float, phi_uncollided: Tensor,
                             fields: Optional[MaterialFields] = None) -> Tensor:
    """Once-collided fluence moments phi^(1) [G, (L+1)^2, 4, Rc, Cc, Sc] from phi_uncollided [G, Rq, Cq, Sq]
    (`uncollided_operator`): restrict the first-collision source at the quadrature points to linear-discontinuous
    moments on the transport grid and do ONE sweep (= source iteration n = 1)."""
    fields = _material_fields_or_build(solver, rho, grid, fields)
    return solver.transport_sweep(fields, solver.first_collision_moment_source(fields, angle, phi_uncollided))


def _check_accumulator(solver: ScatterSolver, fields: MaterialFields, collided_moments: Tensor) -> None:
    sweep_context = solver.sweep_context(fields.grid_coarse)
    expected = (solver.num_groups, sweep_context.num_moments, 4, *fields.grid_coarse.shape)
    if tuple(collided_moments.shape) != expected:
        raise ValueError(f"collided_moments must be {expected}, got {tuple(collided_moments.shape)}")


def multiple_collision_operator(solver: ScatterSolver, rho: Tensor, grid: Grid, angle: float,
                                collided_moments: Optional[Tensor], fields: Optional[MaterialFields] = None) -> Tensor:
    """K Phi: ONE sweep from the scattering source of the collided moments Phi = collided_moments [G, (L+1)^2, 4, Rc,
    Cc, Sc]. collided_moments None gives zeros. `angle` is unused (the collided transport does not depend on the view
    once its source is given) and kept for a uniform signature. Returns [G, (L+1)^2, 4, Rc, Cc, Sc]."""
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
    """first_collision_operator(phi_uncollided) + multiple_collision_operator(collided_moments) in ONE sweep: the sweep
    is linear in its source, so the first-collision source and the scattering source of the previous accumulator are
    summed per direction chunk and transported together (half the transport cost of the two separate operators).
    phi_uncollided: [G, Rq, Cq, Sq] (`uncollided_operator`). Returns the new accumulator
    Phi' = phi^(1) + K Phi [G, (L+1)^2, 4, Rc, Cc, Sc] (Phi = collided_moments, K = sweep o scatter); collided_moments
    None gives phi^(1) alone."""
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
    """Total scatter image (first order + multiple collision) in ONE collapsed-cone pass, exactly
    `ScatterSolver.scatter_image`: the first-collision source of phi_uncollided [G, Rq, Cq, Sq] (the quadrature
    points) and the multiple-collision source of the collided moments collided_moments [G, (L+1)^2, 4, Rc, Cc, Sc]
    (cell mean) share tau_down and one projection per direction chunk. quadrature: the collapsed-cone direction set
    (default: the solver's `cone_quadrature`). Returns the image dict: fluence, planar_fluence, signal_per_group
    [G, Rd, Cd], signal [Rd, Cd]."""
    fields = _material_fields_or_build(solver, rho, grid, fields)
    scatter_source_moments = solver.multiple_collision_source_moments(
        fields, collided_moments.to(solver.device, solver.dtype))
    return solver.scatter_image(fields, angle, phi_uncollided, scatter_source_moments, quadrature)


def citrus_step(solver: ScatterSolver, state: CitrusState, rho: Tensor, grid: Grid,
                angle: float, fields: Optional[MaterialFields] = None) -> tuple[Tensor, Tensor, CitrusState]:
    """One CITRUS scatter step for one view:

        phi_UC = uncollided_operator;
        Phi    = accumulator_update_operator(phi_UC, Phi_previous)
               (= first_collision_operator(phi_UC) + multiple_collision_operator(Phi_previous), one sweep);
        s      = project_scatter(phi_UC, Phi)

    With the densities held fixed, N steps give Phi^(N) = sum_{n=1}^{N} K^{n-1} phi^(1), K = sweep o scatter, i.e.
    exactly the source iteration after N iterations, so the recursion converges to `ScatterSolver.solve_view`.
    rho [M, R, C, S] on grid (M library materials, `utilities.attenuation_to_densities`); angle must match
    state.angle once set. Returns (scatter_signal [Rd, Cd], primary_signal [Rd, Cd], new state). The input state is
    not modified; the returned state holds the new accumulator. fields: precomputed `solver.material_fields(rho,
    grid)` (the densities are shared by every view of an outer iteration, so the caller may build them once); None
    builds them here.
    """
    return citrus_view_step(solver, state, _material_fields_or_build(solver, rho, grid, fields), angle)


def citrus_view_step(solver: ScatterSolver, state: CitrusState, fields: MaterialFields, angle: float, *,
                     primary: bool = True) -> tuple[Tensor, Optional[Tensor], CitrusState]:
    """`citrus_step` from prebuilt density fields: `solver.material_fields(rho, grid)` or, as in the CITRUS loop,
    `solver.material_fields_from_attenuation(X, grid, density_model)` (no fine density volume). primary: also trace
    the view's primary with the solver (`ScatterSolver.primary`, Siddon); False returns None in its place (the loop
    computes the primaries of all views at once instead, `primary_signals`). Returns
    (scatter_signal [Rd, Cd], primary_signal [Rd, Cd] or None, new state); the input state is not modified."""
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
    """Primary signals [V, Rd, Cd] of every view, one fine material map at a time:

        P_m[v] = projector(rho_m)[v]   (g/cm^2; `projector.project` of the density map of material m,
                                        `fields.fine_material`)
        p[v]   = solver.primary_from_path_lengths(P[:, v], materials, angles[v])["signal"]

    i.e. the solver's primary (fine-energy transmission, inverse square, source, detector and grid responses) on the
    line integrals of `projector`, a `recon.mbir_bridge.ViewChunkProjector` (MBIRTORCH's separable-footprint
    pixel-averaged line integrals, the reconstruction's operator A; all views of a chunk in one projection). It agrees
    with the solver's Siddon primary (`ScatterSolver.primary`, one ray to each pixel center) to 0.9 % (relative L2 on
    the 180-view WLAB study; the pixels whose rays graze a face of the prism differ most). Only the materials present
    are projected (`fields.fine_materials`, or `uncollided.present_materials(rho_fine)`). The views are processed in
    chunks of consecutive views so that the path lengths [M', V_chunk, Rd, Cd] (float32) and the projection being
    copied stay within one sinogram: views_per_chunk None = max(1, V // (M' + 1)) (M' = number of materials
    projected); each chunk evaluates the material maps once (cheap next to the projections; MBIRTORCH uses a cached
    sub-model over the chunk's angles). The solver's sub-pixel sampling (primary_samples_per_pixel_axis) does not
    apply: the projector already averages the line integral over the pixel. angles [V] (rad) must be the projector's
    views. out: optional [V, Rd, Cd] tensor to fill. quantity:
    "signal" (default, the detector signal, keV/cm^2) or "planar_fluence" (summed over the groups, photons/cm^2; the
    denominator of the synthetic data's SPR, `utilities.synthesize_cone_beam_data`)."""
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
    """X (1/cm, MBIR's monoenergetic attenuation at E_eff) -> rho [M, ...] (g/cm^3, the library's M reference materials,
    `library.materials`: by default air, lung, adipose, water, muscle, soft tissue, cortical bone; any set chosen with
    `SolverConfig.materials`) for the solver: `utilities.attenuation_to_densities` (the interpolating model at E_eff:
    the two reference materials whose nominal attenuations bracket X, at most two per voxel, a pure-bone voxel stays
    pure bone;
    below the lowest reference material that reference material alone of reduced density, above the highest that
    reference material alone, denser) after
    clamping X at 0 (FDK undershoot, noise). `last_negative_fraction` = fraction of voxels clamped in the last call
    (`__call__` or `negative_fraction`). `material(X, m)` evaluates one material's map [R, C, S] (bit for bit the row
    m of the full model, `utilities.attenuation_to_density`): the `scatter_solver.MaterialDensityModel` of
    `ScatterSolver.material_fields_from_attenuation`, which never forms the [M, R, C, S] volume."""

    library: MultigroupLibrary
    energy_kev: float
    last_negative_fraction: float = 0.0
    slab_voxels: int = 1 << 22

    @classmethod
    def from_library(cls, library: MultigroupLibrary, energy_kev: Optional[float] = None) -> "DensityModel":
        """The model of `library` (its reference materials `library.materials` and their nominal densities): E_eff =
        energy_kev, or the spectrum-mean energy of the library (63.03 keV for the default spectrum)."""
        return cls(library, effective_energy(library) if energy_kev is None else float(energy_kev))

    def __call__(self, X: Tensor) -> Tensor:
        """rho [M, ...] of X [...] (all materials at once)."""
        self.last_negative_fraction = float((X < 0).float().mean())
        return attenuation_to_densities(self.library, X.clamp_min(0), self.energy_kev)

    def _slabs(self, X: Tensor) -> list[slice]:
        voxels_per_index = max(1, int(np.prod(X.shape[1:]))) if X.dim() > 1 else 1
        indices_per_slab = max(1, int(self.slab_voxels) // voxels_per_index)
        num_indices = int(X.shape[0]) if X.dim() else 1
        return [slice(i, i + indices_per_slab) for i in range(0, num_indices, indices_per_slab)]

    def material(self, X: Tensor, m: int) -> Tensor:
        """The density map [...] (g/cm^3) of library material m of X [...] (1/cm): `__call__(X)[m]` bit for bit,
        without the other materials. Evaluated slab by slab along the first axis (about `slab_voxels` voxels per slab,
        `utilities.attenuation_to_density`) into the result, so that the only full-size temporary is the result
        itself (the elementwise model gives the same values in any slab layout)."""
        if X.dim() == 0:
            return attenuation_to_density(self.library, X, m, self.energy_kev)
        out = torch.empty_like(X)
        for slab in self._slabs(X):
            out[slab] = attenuation_to_density(self.library, X[slab], m, self.energy_kev)
        return out

    def negative_fraction(self, X: Tensor) -> float:
        """Fraction of the voxels of X below 0 (clamped by the model); also stored in last_negative_fraction.
        Counted slab by slab (no temporary of X's size)."""
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
    """Compact storage of one CITRUS accumulator Phi [G, N, 4, Rc, Cc, Sc] per view (N = (L+1)^2 spherical-harmonic
    moments): the linear-discontinuous cell mean only, in float16 with one scale per (group, moment) (the moments are
    ~1e-6 per source photon/sr: plain float16 would underflow), i.e. 1/8 of the float32 bytes.
    `put(slot, accumulator)` encodes a float32 device tensor; `get(slot)` returns the decoded float32 tensor of the
    full shape on `device` (zero linear-discontinuous slopes) or None before the first put.

    location: "gpu" (default when the total fits `gpu_budget_mib`) or "cpu" (pinned host memory when the total is at
    most `pinned_memory_limit_gib`, else pageable); "auto" chooses. `pinned` tells whether the host buffers are
    pinned. `transfer_seconds` accumulates the time spent in put / get (synchronized on CUDA)."""

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
        G, N = self.full_shape[:2]
        return int(np.prod(self.stored_shape)) * 2 + G * N * 4

    @property
    def total_bytes(self) -> int:
        return self.bytes_per_view * self.num_slots

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def put(self, slot: int, accumulator: Tensor) -> None:
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
        self._data = [None] * self.num_slots
        self._scale = [None] * self.num_slots

    def describe(self) -> dict:
        return {"location": self.location, "pinned": self.pinned, "stored_shape": list(self.stored_shape),
                "full_shape": list(self.full_shape), "bytes_per_view": self.bytes_per_view,
                "num_views": self.num_slots, "total_gib": self.total_bytes / 2 ** 30}


# =====================================================================================================
# configuration and regularization
# =====================================================================================================

@dataclass
class CitrusConfig:
    """Parameters of the CITRUS reconstruction loop (the scatter solver has its own SolverConfig).

    num_outer_iterations : outer iterations N (each = one scatter step per view + ONE MBIR iteration at schedule
                     position k)
    energy_kev     : effective energy E_eff of the density model (None: spectrum mean of the solver's library)
    min_counts     : lower clamp of the (corrected) counts before the log
    base_seed      : numpy's global RNG is seeded with [base_seed, k] before MBIR iteration k (MBIRTORCH draws the
                     partitions from it), so runs with identical inputs give identical MBIR sequences
    gpu_state_budget_mib : the per-view accumulators stay on the GPU while their total is below this, else they go to
                     (pinned) host memory (`StateStore`)
    pinned_memory_limit_gib : largest pinned host allocation for the accumulators
    verbose        : print one line per outer iteration
    scatter_view_stride : n >= 1. 1 (default): the scatter step runs on every view. n > 1: only on every n-th view
                     (views 0, n, 2n, ...: uniform over the angular range, starting at view 0; `scatter_view_indices`);
                     the other views' scatter signal is interpolated in angle (scatter_view_interpolation_order) and
                     divided by their primary (`primary_signals`, every view). Only the computed views keep an
                     accumulator (1/n of the state memory).
    scatter_view_interpolation_order : the per-pixel interpolation in angle of the skipped views' scatter signal
                     from the computed views, circular in angle for a full rotation and along the arc for a short scan
                     (the views beyond the last computed view hold its value; ignored at stride 1;
                     `angular_neighbors`):
                       3 (default)  cubic Catmull-Rom on the 4 nearest computed views: interpolating, continuous first
                                    derivative, exact for polynomials of angle up to degree 2 on uniformly spaced
                                    computed views (no local 4-point C1 scheme is exact for cubics)
                       2            quadratic: 3-point Lagrange centered on the nearest computed view (exact for
                                    quadratics of angle on any spacing; jumps where the nearest view changes)
                       1            linear between the 2 bracketing computed views
                       0            nearest computed view (zero-order hold)
                       None         diagnostics only: no interpolation; the skipped views keep their most recent
                                    computed scatter if any, else zero, i.e. with the fixed view subset they are NOT
                                    scatter-corrected
    scatter_every  : n >= 1. 1 (default): a scatter step at every outer iteration. n > 1: the scatter stage runs at
                     the outer iterations 0, n, 2n, ... only and the previous scatter-to-primary ratio is reused in
                     between (n MBIR iterations per scatter step; the accumulators keep their age). For a fixed number
                     of scatter steps, spreading them over n times more outer iterations costs almost nothing (an MBIR
                     iteration is ~1 % of a scatter stage) and improves the reconstruction (module docstring).
    primary_views_per_chunk : views per chunk of `primary_signals` (the MBIRTORCH projections of the material maps
                     that give every view's primary signal, module docstring); None (default): max(1, V // (M' + 1))
                     with M' the materials present, so the per-material path lengths [M', chunk, Rd, Cd] and the
                     projection being copied stay within one sinogram [V, Rd, Cd]
    resume_error_sinogram : keep MBIRTORCH's error sinogram between outer iterations (one sinogram, stored as -A X)
                     and resume from it instead of recomputing y_c - A X with a forward projection per call (module
                     docstring; equal to float rounding)
    """

    num_outer_iterations: int = 10
    energy_kev: Optional[float] = None
    min_counts: float = 1.0
    base_seed: int = 0
    gpu_state_budget_mib: float = 2600.0
    pinned_memory_limit_gib: float = 8.0
    verbose: bool = False
    scatter_view_stride: int = 1
    scatter_view_interpolation_order: Literal[0, 1, 2, 3] | None = 3
    scatter_every: int = 1
    primary_views_per_chunk: Optional[int] = None
    resume_error_sinogram: bool = True

    def validate(self) -> None:
        if self.num_outer_iterations < 1:
            raise ValueError("num_outer_iterations must be >= 1")
        if self.min_counts <= 0:
            raise ValueError("min_counts must be > 0")
        stride = self.scatter_view_stride
        if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)) or stride < 1:
            raise ValueError(f"scatter_view_stride must be an integer >= 1, got {stride!r}")
        _check_interpolation_order(self.scatter_view_interpolation_order)
        every = self.scatter_every
        if isinstance(every, bool) or not isinstance(every, (int, np.integer)) or every < 1:
            raise ValueError(f"scatter_every must be an integer >= 1, got {every!r}")
        chunk = self.primary_views_per_chunk
        if chunk is not None and (isinstance(chunk, bool) or not isinstance(chunk, (int, np.integer)) or chunk < 1):
            raise ValueError(f"primary_views_per_chunk must be None or an integer >= 1, got {chunk!r}")
        if not isinstance(self.resume_error_sinogram, (bool, np.bool_)):
            raise ValueError(f"resume_error_sinogram must be a bool, got {self.resume_error_sinogram!r}")


# =====================================================================================================
# sparse-view scatter: angular interpolation of the skipped views
# =====================================================================================================

SCATTER_VIEW_INTERPOLATION_ORDERS = (None, 0, 1, 2, 3)
_TWO_PI = 2.0 * np.pi


def _check_interpolation_order(order, allow_none: bool = True) -> None:
    ok = (order is None and allow_none) or (
        not isinstance(order, bool) and isinstance(order, (int, np.integer)) and int(order) in (0, 1, 2, 3))
    if not ok:
        allowed = SCATTER_VIEW_INTERPOLATION_ORDERS if allow_none else SCATTER_VIEW_INTERPOLATION_ORDERS[1:]
        raise ValueError(f"scatter_view_interpolation_order must be one of {allowed}, got {order!r}")


def scatter_view_indices(num_views: int, stride: int) -> np.ndarray:
    """The views that get a scatter step at `stride`: 0, stride, 2 stride, ... < num_views (uniform over the angular
    range of a uniformly sampled scan, starting at view 0). Stride 1 gives every view."""
    if int(stride) < 1 or int(num_views) < 1:
        raise ValueError(f"need num_views >= 1 and stride >= 1, got {num_views}, {stride}")
    return np.arange(0, int(num_views), int(stride))


def angular_neighbors(computed_angles: Union[Sequence[float], np.ndarray], angle: float,
                      order: Literal[0, 1, 2, 3] = 3, circular: bool = True) -> list[tuple[int, float]]:
    """Interpolation neighbors and weights of a view at `angle` (rad) from views at `computed_angles` (rad, any order,
    any 2 pi offset), circular in angle (period 2 pi). circular=False (a scan over less than a full rotation): the
    angles are taken as given (unwrapped along the arc), the neighbors stay among the computed views (the
    cubic's ghost nodes beyond the end views are linear extrapolations, so it stays exact for linear functions of
    the angle; the quadratic stencil shifts inside) and an angle outside [min, max] of the computed angles gets the
    nearest end view with weight 1. Returns [(index into computed_angles, weight), ...], weights
    summing to 1. With j0, j1 the computed views bracketing the angle (wrapping from the last computed angle to the
    first + 2 pi) and t in [0, 1) its fractional position between them:

      0 : nearest computed view (zero-order hold; t = 1/2 goes to j0)
      1 : linear, weights 1 - t, t on j0, j1
      2 : quadratic, 3-point Lagrange on the nearest computed view and its two neighbors, at their actual angles:
          exact for quadratics of angle on any spacing; discontinuous where the nearest view changes
      3 : cubic Catmull-Rom on j0 - 1, j0, j1, j1 + 1 with parameter t: interpolating, continuous first derivative
          (tangent (f[j1] - f[j0 - 1]) / 2 per interval at j0), exact for polynomials of angle up to degree 2 on
          uniformly spaced computed views and for linear ones on any spacing; NOT exact for cubics (no local 4-point
          C1 interpolant is); its outer weights are negative

    With fewer computed views than the order needs (order + 1), the order is lowered to (number of computed views) - 1;
    one computed view gets weight 1."""
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
    """True when the views cover the circle without a large gap: the largest angular gap between consecutive views
    (circularly, mod 2 pi) is at most 1.5 times the median gap. False for a short scan (the missing arc is the gap)."""
    a = np.sort(np.mod(np.asarray(angles, dtype=np.float64).reshape(-1), _TWO_PI))
    if a.size < 3:
        return a.size == 2 and abs(float(a[1] - a[0]) - np.pi) < 1e-9
    gaps = np.diff(np.concatenate([a, a[:1] + _TWO_PI]))
    return bool(gaps.max() <= 1.5 * np.median(gaps) + 1e-12)


def _unwrap_arc(angles: np.ndarray) -> np.ndarray:
    """Angles of a short scan unwrapped along its arc: the arc starts after the largest circular gap."""
    a = np.mod(angles, _TWO_PI)
    srt = np.sort(a)
    gaps = np.diff(np.concatenate([srt, srt[:1] + _TWO_PI]))
    origin = float(srt[(int(np.argmax(gaps)) + 1) % srt.size])
    return origin + np.mod(a - origin, _TWO_PI)


def interpolation_weights(angles: Union[Sequence[float], np.ndarray], computed: Union[Sequence[int], np.ndarray],
                          order: Literal[0, 1, 2, 3] | None = 3, circular: Optional[bool] = None) -> np.ndarray:
    """The [V, Vc] matrix M with s_all = M s_c: row v holds the `angular_neighbors` weights of view v over the
    computed views (indices `computed` into angles); the computed views' own rows are one-hot (they keep their value);
    order None gives zero rows for the skipped views. circular: None (default) = `full_rotation(angles)`; False
    interpolates along the arc of a short scan (angles unwrapped from the end of the missing arc), holding the end
    computed views beyond them."""
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
    """Every view's scatter [V, ...] from the computed views' scatter computed_scatter [Vc, ...] (any per-view image,
    any units; V views, Vc of them computed): the computed views keep computed_scatter exactly; a skipped view v gets
    sum_j w_vj computed_scatter[j] (`angular_neighbors`, per pixel), clamped at 0 (the order 2 and 3 weights can be
    negative). order None: no interpolation, the skipped views keep their
    value in `previous` [V, ...] (their most recent estimate) if given, else zero. weights: a precomputed
    `interpolation_weights(angles, computed, order)` as a tensor (saves rebuilding it every outer iteration)."""
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
    """MBIRTORCH's automatic sigma_y / sigma_x / sigma_prox (`auto_set_regularization_params`, what the first `recon`
    call would compute) for the line-integral sinogram [V, Rd, Cd] with `weights`; the values are then frozen in the
    model (auto_regularize_flag False). Returns {"sigma_y", "sigma_x", "sigma_prox"}."""
    model.set_params(no_warning=True, auto_regularize_flag=True)
    model.auto_set_regularization_params(sinogram, weights=weights)
    model.set_params(auto_regularize_flag=False)
    return {k: float(v) for k, v in zip(REGULARIZATION_NAMES, model.get_params(list(REGULARIZATION_NAMES)))}


def apply_regularization(model, regularization: dict) -> None:
    """Set the frozen sigma_y, sigma_x, sigma_prox and keep them (auto_regularize_flag False)."""
    model.set_params(no_warning=True, auto_regularize_flag=False,
                     **{k: float(regularization[k]) for k in REGULARIZATION_NAMES})


# =====================================================================================================
# reconstruction loop
# =====================================================================================================

ScatterStage = Callable[[int, Tensor, dict], Tensor]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _outer_loop(Y: Tensor, I0: float, angles: np.ndarray, grid: Grid, geometry: ConeGeometry, config: CitrusConfig,
                scatter_stage: Optional[ScatterStage], device: torch.device, *, regularization: Optional[dict],
                init_recon: Optional[Tensor], model, callback, label: str) -> tuple[Tensor, dict]:
    """The MBIR loop shared by citrus_recon and mbir_recon. scatter_stage(k, X_k, record) -> SPR [V, Rd, Cd] (or None
    for no correction) is called before MBIR iteration k."""
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
    for k in range(config.num_outer_iterations):
        record: dict = {"outer_iteration": k + 1}
        _sync(device)
        t0 = time.perf_counter()
        if scatter_stage is not None:
            # scatter stage at the outer iterations 0, n, 2n, ... (n = config.scatter_every); the previous
            # scatter-to-primary ratio is reused in between
            refresh = (k % int(config.scatter_every) == 0) or spr is None
            record["scatter_refreshed"] = refresh
            if refresh:
                spr = scatter_stage(k, X, record)
        corrected_counts = Y if spr is None else Y / (1.0 + spr)
        corrected_line_integrals = -torch.log(corrected_counts.clamp_min(config.min_counts) / float(I0))
        if regularization is None:  # estimated once on the first sinogram MBIR inverts, then frozen
            regularization = estimate_regularization(model, corrected_line_integrals, weights)
        _sync(device)
        t1 = time.perf_counter()
        np.random.seed([int(config.base_seed), int(k)])
        if resume:
            # e = y_c - A X: the kept -A X shifted by the current corrected data (A X is unchanged by the scatter
            # stage); MBIRTORCH updates e and its init volume in place, so it gets a copy of X
            error_sinogram = (None if negative_projection is None
                              else negative_projection.add_(corrected_line_integrals))
            negative_projection = None
            X, _, checkpoint = mbir_iterations(model, corrected_line_integrals, X.clone(), first_iteration=k,
                                               weights=weights, error_sinogram=error_sinogram,
                                               return_checkpoint=True)
            negative_projection = checkpoint["error_sinogram"].sub_(corrected_line_integrals)
            del error_sinogram, checkpoint
        else:
            X, _ = mbir_iterations(model, corrected_line_integrals, X, first_iteration=k, weights=weights)
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
                  f"{record['scatter_seconds']:.1f} s, MBIR {record['mbir_seconds']:.1f} s{extra}", flush=True)
    info.update(seconds=time.perf_counter() - t_start, regularization=regularization, spr=spr,
                scatter_counts=scatter_estimate)
    return X, info


def _counts_and_angles(sinogram_counts, angles, device) -> tuple[Tensor, np.ndarray]:
    Y = torch.as_tensor(sinogram_counts, dtype=torch.float32, device=device).contiguous()
    if isinstance(angles, Tensor):
        angles = angles.detach().cpu().numpy()
    angle_array = np.asarray(angles, dtype=np.float64).reshape(-1)
    if Y.dim() != 3 or Y.shape[0] != angle_array.size:
        raise ValueError(f"sinogram_counts must be [V = {angle_array.size}, rows, channels], got {tuple(Y.shape)}")
    return Y, angle_array


def citrus_recon(sinogram_counts: Union[Tensor, np.ndarray], I0: float, angles: Union[Sequence[float], np.ndarray],
                 grid: Grid, geometry: ConeGeometry, solver: ScatterSolver, config: Optional[CitrusConfig] = None,
                 *, regularization: Optional[dict] = None, init_recon: Optional[Tensor] = None, model=None,
                 callback: Optional[Callable[[int, Tensor, dict], None]] = None) -> tuple[Tensor, dict]:
    """CITRUS reconstruction (module docstring) of measured counts.

    sinogram_counts [V, Rd, Cd]: measured counts Y (primary + scatter, flat-field I0 per pixel); angles [V] (rad);
    grid / geometry: reconstruction grid and cone-beam geometry (MBIRTORCH conventions; also the solver's fine grid,
    covered by its transport grid of solver.config.transport_cell_cm cells); solver: the ScatterSolver (its geometry,
    detector and grid response describe the scanner; its device is used for everything); config: CitrusConfig
    (default CitrusConfig()).
    regularization: frozen {"sigma_y", "sigma_x", "sigma_prox"}; None estimates them once (MBIRTORCH's automatic
    estimate) on the first scatter-corrected sinogram and freezes them. init_recon: initial volume [R, C, S]
    (default: FDK of the uncorrected data). model: a prebuilt MBIRTORCH ConeBeamModel (`recon.mbir_bridge.
    make_cone_model(grid, geometry, angles)`). callback(k, X_k, record) is called for the initial volume (k = 0) and
    after every outer iteration.

    Returns (recon [R, C, S] float32 on the solver device, info) with info:
      records          per outer iteration (records[0] = the initial volume): outer_iteration (k), scatter_seconds,
                       transfer_seconds, mbir_seconds, total_seconds, negative_fraction (voxels clamped by the density
                       model), accumulator_change_mean / accumulator_change_max (relative accumulator change over the
                       computed views), scatter_change_relative (mean |S_k - S_k-1| / mean |S_k| of the scatter-count
                       estimate), gpu_peak_allocated_mib (peak allocated CUDA memory of the iteration, MiB)
      spr              [V, Rd, Cd] the last scatter-to-primary ratio estimate
      scatter_counts   [V, Rd, Cd] the last scatter-count estimate Y - Y_c
      regularization   the frozen sigmas;  state: the accumulator storage (one slot per computed view);  init_recon,
                       model, seconds, fdk_seconds
      computed_views   [Vc] int array: the views with a scatter step (`scatter_view_indices`; every view at stride 1)
      view_interpolated [V] bool array: True where the scatter is interpolated in angle (the skipped views at
                       config.scatter_view_stride > 1 with an interpolation order), False where the scatter step
                       computed it (and for the uncorrected skipped views of order None)
    The records of a scatter stage hold primary_seconds (seconds for the primaries of all views, `primary_signals`,
    included in scatter_seconds); with config.scatter_view_stride > 1 also interpolation_seconds (seconds per outer
    iteration spent on the angular interpolation, included in scatter_seconds). The densities never exist as an
    [M, R, C, S] volume: `ScatterSolver.material_fields_from_attenuation` with the `DensityModel` (module docstring).
    Every record holds scatter_refreshed: True at the outer iterations with a scatter stage (all of them at
    config.scatter_every = 1), False where the previous ratio was reused (then scatter_seconds ~ 0 and the accumulator
    entries are absent).
    """
    config = CitrusConfig() if config is None else config
    config.validate()
    device = solver.device
    Y, angle_array = _counts_and_angles(sinogram_counts, angles, device)
    V = len(angle_array)
    density_model = DensityModel.from_library(solver.library, config.energy_kev)
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

    def scatter_stage(k: int, X: Tensor, record: dict) -> Tensor:
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
                              label="citrus")
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
    """The MBIR loop of `citrus_recon` without the scatter solver, for comparison: the same FDK initialization,
    one warm-started MBIR iteration per outer iteration at schedule position k, the same weights W = Y / I0 and seeds.

    spr None: no scatter correction (MBIR of -log(Y / I0)); spr [V, Rd, Cd]: a fixed correction Y_c = Y / (1 + spr)
    (e.g. a scatter estimate from elsewhere). device: default the device of a tensor sinogram, else CUDA if available.
    Other arguments and the returned (recon, info) as in `citrus_recon` (info has no accumulator entries).
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
