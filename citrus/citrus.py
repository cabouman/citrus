"""CITRUS: scatter-corrected MBIR with the LBTE source iteration solver in the loop (docs/source/method.rst).

The algorithm
-------------
Measured counts Y [V, Rd, Cd] (flat field I0) contain primary and scatter, Y ~ Poisson(P + S). CITRUS alternates, per
outer iteration k, ONE scatter step per view and ONE warm-started MBIR iteration:

    X_0      = FDK(-log(Y / I0))                                    (the uncorrected data)
    for k = 0 .. N-1:
        rho_k            = densities(X_k)                            two-material split at E_eff
        for every view v:
            phi1          = G1(rho_k)                                uncollided fluence (fine grid)
            Phi_v         = G_step(rho_k, phi1, Phi_v)               ONE sweep of FC source + aged accumulator
            s_v, p_v      = project_scatter(phi1, Phi_v), primary    scatter and primary signals of the view
        SPR_k            = s / p
        Y_c              = Y / (1 + SPR_k)                           scatter-corrected counts
        X_{k+1}          = one VCD iteration of MBIR on y_c = -log(Y_c / I0), warm-started at X_k,
                           at partition-schedule position k, weights W = Y / I0, frozen regularization

The per-view accumulator Phi_v = sum_{n >= 1} (collided moments of order n) is carried across outer iterations: with
the densities held fixed, N steps reproduce N source iterations exactly (`citrus_step`), so the scatter estimate
converges together with the reconstruction ("age-ordered" source iteration: the n-th collision order of the current
estimate was computed n - 1 outer iterations ago). The collided transport is the expensive part of a converged
scatter solve; CITRUS spends one sweep per view per outer iteration instead of a full source iteration.

Scatter model -> correction: the solver gives per view the scatter and primary signals (s, p) of the current
densities, so the correction is the multiplicative Y_c = Y p / (p + s) = Y / (1 + SPR(X)); it is independent of I0 and
of any mismatch between the solver's polychromatic primary and MBIR's monoenergetic model, and it rescales the noise
of Y (W = Y / I0 is the exact weight of y_c).

Per-view state: one accumulator [G, (L+1)^2, 4, Rc, Cc, Sc] per view, stored compactly (`StateStore`): only the LD
cell mean (the aged accumulator re-enters the next sweep with zero slopes) in float16 with a per-(group, moment)
scale, 1/8 of float32 (17.6 MiB per view on a 0.5 cm WLAB grid with 1 cm transport cells). The reconstruction differs
from exact float32 storage by ~6e-5 (relative), far below the reconstruction error.

Public API
----------
    citrus_recon(...)     the CITRUS reconstruction (returns the reconstruction and a dict of diagnostics)
    mbir_recon(...)       the same MBIR loop without the solver: no correction, or a fixed SPR correction
    citrus_step(...)      one scatter step of one view over an explicit CitrusState
    G1, G1to2, G_gt2, G_step, project_scatter   the per-view operators of the step
    DensityModel, StateStore, CitrusConfig, estimate_regularization, apply_regularization
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from .recon.mbir_bridge import fdk, make_cone_model, mbir_iterations
from .scatter_solver.angular_quadrature import Quadrature
from .scatter_solver.cross_sections import MultigroupLibrary
from .scatter_solver.geometry import ConeGeometry, Grid
from .scatter_solver.scatter_solver import MaterialFields, ScatterSolver
from .utilities import attenuation_to_densities, effective_energy

__all__ = [
    "CitrusState",
    "G1",
    "G1to2",
    "G_gt2",
    "G_step",
    "project_scatter",
    "citrus_step",
    "DensityModel",
    "StateStore",
    "CitrusConfig",
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

    angle      : view angle (rad) the state belongs to; None until the first step (then fixed).
    Phi_gt1    : [G, (L+1)^2, 4, Rc, Cc, Sc] accumulated collided-fluence moments, or None (= zero) initially.
    cone_quad  : cached collapsed-cone direction set of the view (depends on grid and angle only).
    num_steps  : number of `citrus_step` calls applied.
    history    : relative change ||Phi_new - Phi_old|| / ||Phi_new|| of the scalar-flux cell means per step
                 (the CITRUS analogue of SIResult.history).
    For several views keep one state per view.
    """

    angle: Optional[float] = None
    Phi_gt1: Optional[Tensor] = None
    cone_quad: Optional[Quadrature] = None
    num_steps: int = 0
    history: list[float] = field(default_factory=list)


def _fields(solver: ScatterSolver, rho: Tensor, grid: Grid, fields: Optional[MaterialFields]) -> MaterialFields:
    return solver.material_fields(rho, grid) if fields is None else fields


def G1(solver: ScatterSolver, rho: Tensor, grid: Grid, angle: float,
       fields: Optional[MaterialFields] = None) -> Tensor:
    """Uncollided fluence phi_uc [G, R, C, S] on the fine grid. rho [2, R, C, S]."""
    return solver.uncollided(_fields(solver, rho, grid, fields), angle)


def G1to2(solver: ScatterSolver, rho: Tensor, grid: Grid, angle: float, phi_uc: Tensor,
          fields: Optional[MaterialFields] = None) -> Tensor:
    """Once-collided fluence moments phi2 [G, (L+1)^2, 4, Rc, Cc, Sc]: restrict the first-collision source to
    LD moments on the supervoxel grid and do ONE sweep (= source iteration n = 1)."""
    f = _fields(solver, rho, grid, fields)
    return solver.transport_sweep(f, solver.fc_moment_source(f, angle, phi_uc))


def _check_accumulator(solver: ScatterSolver, f: MaterialFields, Phi_gt1: Tensor) -> None:
    ctx = solver.sweep_context(f.grid_coarse)
    expected = (solver.num_groups, ctx.num_moments, 4, *f.grid_coarse.shape)
    if tuple(Phi_gt1.shape) != expected:
        raise ValueError(f"Phi_gt1 must be {expected}, got {tuple(Phi_gt1.shape)}")


def G_gt2(solver: ScatterSolver, rho: Tensor, grid: Grid, angle: float, Phi_gt1: Optional[Tensor],
          fields: Optional[MaterialFields] = None) -> Tensor:
    """ONE sweep from the scattering source of collided moments Phi_gt1 [G, (L+1)^2, 4, Rc, Cc, Sc]. Phi_gt1 None
    gives zeros. `angle` is unused (the collided transport does not depend on the view once its source is given) and
    kept for a uniform signature. Returns [G, (L+1)^2, 4, Rc, Cc, Sc]."""
    f = _fields(solver, rho, grid, fields)
    ctx = solver.sweep_context(f.grid_coarse)
    if Phi_gt1 is None:
        return torch.zeros(solver.num_groups, ctx.num_moments, 4, *f.grid_coarse.shape, device=solver.device,
                           dtype=solver.dtype)
    _check_accumulator(solver, f, Phi_gt1)
    return solver.transport_sweep(f, solver.scatter_moment_source(f, Phi_gt1.to(solver.device, solver.dtype), ctx))


def G_step(solver: ScatterSolver, rho: Tensor, grid: Grid, angle: float, phi_uc: Tensor,
           Phi_gt1: Optional[Tensor], fields: Optional[MaterialFields] = None) -> Tensor:
    """G1to2(phi_uc) + G_gt2(Phi_gt1) in ONE sweep: the sweep is linear in its source, so the first-collision
    source and the scattering source of the previous accumulator are summed per direction chunk and transported
    together (half the transport cost of the two separate operators). Returns the new accumulator
    Phi_gt1' = phi2 + K Phi_gt1 [G, (L+1)^2, 4, Rc, Cc, Sc] (K = sweep o scatter); Phi_gt1 None gives phi2 alone."""
    f = _fields(solver, rho, grid, fields)
    ctx = solver.sweep_context(f.grid_coarse)
    fc = solver.fc_moment_source(f, angle, phi_uc, ctx)
    if Phi_gt1 is None:
        return solver.transport_sweep(f, fc)
    _check_accumulator(solver, f, Phi_gt1)
    sc = solver.scatter_moment_source(f, Phi_gt1.to(solver.device, solver.dtype), ctx)

    def fn(idx: Tensor, out: Optional[Tensor] = None) -> Tensor:
        src = fc(idx, out)  # fills `out` when the sweep lends its storage, else a fresh tensor
        src.add_(sc(idx))
        return src

    return solver.transport_sweep(f, fn)


def project_scatter(solver: ScatterSolver, rho: Tensor, grid: Grid, angle: float, phi_uc: Tensor, Phi_gt1: Tensor,
                    fields: Optional[MaterialFields] = None,
                    quad: Optional[Quadrature] = None) -> dict[str, Tensor]:
    """Total scatter image (first order + multiple collision) in ONE collapsed-cone pass, exactly
    `ScatterSolver.scatter_image`: the FC source of phi_uc [G, R, C, S] (fine grid, sub-voxel points) and the MC source
    of the collided moments Phi_gt1 [G, (L+1)^2, 4, Rc, Cc, Sc] (cell mean) share tau_down and one projection per
    direction chunk. Returns the image dict: fluence, planar_fluence, signal_per_group [G, Rd, Cd], signal [Rd, Cd]."""
    f = _fields(solver, rho, grid, fields)
    T_mean = solver.mc_source_moments(f, Phi_gt1.to(solver.device, solver.dtype))
    return solver.scatter_image(f, angle, phi_uc, T_mean, quad)


def citrus_step(solver: ScatterSolver, state: CitrusState, rho: Tensor, grid: Grid,
                angle: float, fields: Optional[MaterialFields] = None) -> tuple[Tensor, Tensor, CitrusState]:
    """One CITRUS scatter step for one view:

        phi1 = G1; Phi_gt1 = G_step(phi1, Phi_gt1_prev) (= G1to2(phi1) + G_gt2(Phi_gt1_prev), one sweep);
        s = project_scatter(phi1, Phi_gt1)

    With the densities held fixed, N steps give Phi_gt1^(N) = sum_{n=1}^{N} K^{n-1} phi2, K = sweep o scatter, i.e.
    exactly the source iteration after N iterations, so the recursion converges to `ScatterSolver.solve_view`.
    rho [2, R, C, S] on grid; angle must match state.angle once set. Returns (scatter_signal [Rd, Cd],
    primary_signal [Rd, Cd], new state). The input state is not modified; the returned state holds the new
    accumulator. fields: precomputed `solver.material_fields(rho, grid)` (the densities are shared by every view of
    an outer iteration, so the caller may build them once); None builds them here.
    """
    if state.angle is not None and abs(float(state.angle) - float(angle)) > 1e-12:
        raise ValueError(f"state belongs to angle {state.angle}, got {angle}")
    fields = _fields(solver, rho, grid, fields)
    quad = state.cone_quad if state.cone_quad is not None else solver.cone_quadrature(grid, angle)
    phi1 = G1(solver, rho, grid, angle, fields)
    Phi_new = G_step(solver, rho, grid, angle, phi1, state.Phi_gt1, fields)  # phi2 + K Phi_gt1 in one sweep
    scatter = project_scatter(solver, rho, grid, angle, phi1, Phi_new, fields, quad)["signal"]
    del phi1
    primary = solver.primary(fields, angle)["signal"]
    new_norm = float(torch.linalg.vector_norm(Phi_new[:, 0, 0]))
    if state.Phi_gt1 is None:
        rel = 1.0 if new_norm > 0 else 0.0
    else:
        diff = float(torch.linalg.vector_norm(Phi_new[:, 0, 0] - state.Phi_gt1[:, 0, 0].to(Phi_new)))
        rel = diff / new_norm if new_norm > 0 else 0.0
    new_state = CitrusState(angle=float(angle), Phi_gt1=Phi_new, cone_quad=quad, num_steps=state.num_steps + 1,
                            history=state.history + [rel])
    return scatter, primary, new_state


# =====================================================================================================
# density model
# =====================================================================================================

@dataclass
class DensityModel:
    """X (1/cm, MBIR's monoenergetic attenuation at E_eff) -> rho [2, ...] (water, bone; g/cm^3) for the solver:
    `utilities.attenuation_to_densities` (two-material split at E_eff) after clamping X at 0 (FDK undershoot,
    noise). `last_negative_fraction` = fraction of voxels clamped in the last call."""

    lib: MultigroupLibrary
    energy_kev: float
    last_negative_fraction: float = 0.0

    @classmethod
    def from_library(cls, lib: MultigroupLibrary, energy_kev: Optional[float] = None) -> "DensityModel":
        """E_eff = energy_kev, or the spectrum-mean energy of the library (63.03 keV for the default spectrum)."""
        return cls(lib, effective_energy(lib) if energy_kev is None else float(energy_kev))

    def __call__(self, X: Tensor) -> Tensor:
        self.last_negative_fraction = float((X < 0).float().mean())
        return attenuation_to_densities(self.lib, X.clamp_min(0), self.energy_kev)


# =====================================================================================================
# per-view accumulator storage
# =====================================================================================================

_FP16_TARGET = 2.0 ** 14  # |q| <= 16384 (< 65504): the per-slice maximum maps here, normals reach 1e-9 x max


class StateStore:
    """Compact storage of one CITRUS accumulator Phi [G, N, 4, Rc, Cc, Sc] per view: the LD cell mean only, in float16
    with one scale per (group, moment) (the moments are ~1e-6 per source photon/sr: plain float16 would underflow),
    i.e. 1/8 of the float32 bytes. `put(i, Phi)` encodes a float32 device tensor; `get(i)` returns the decoded float32
    tensor of the full shape on `device` (zero LD slopes) or None before the first put.

    location: "gpu" (default when the total fits `gpu_budget_mib`) or "cpu" (pinned host memory when the total is at
    most `pin_limit_gib`, else pageable); "auto" chooses. `transfer_seconds` accumulates the time spent in put / get
    (synchronized on CUDA)."""

    def __init__(self, num_slots: int, full_shape: Sequence[int], device, location: str = "auto",
                 gpu_budget_mib: float = 2600.0, pin_limit_gib: float = 8.0) -> None:
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
        self.pin = location == "cpu" and self.device.type == "cuda" and self.total_bytes <= pin_limit_gib * 2 ** 30
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

    def put(self, i: int, Phi: Tensor) -> None:
        if tuple(Phi.shape) != self.full_shape:
            raise ValueError(f"accumulator must be {self.full_shape}, got {tuple(Phi.shape)}")
        self._sync()
        t0 = time.perf_counter()
        x = Phi.detach()[:, :, :1].float()  # LD cell mean
        amax = x.abs().amax(dim=(2, 3, 4, 5), keepdim=True)
        scale = torch.where(amax > 0, amax / _FP16_TARGET, torch.ones_like(amax))
        q = (x / scale).to(torch.float16)
        if self.location == "gpu":
            self._data[i] = q.to(self.device).contiguous()
            self._scale[i] = scale.to(self.device)
        else:
            buf = self._data[i]
            if buf is None:
                buf = torch.empty(self.stored_shape, dtype=torch.float16, pin_memory=self.pin)
                self._data[i] = buf
            buf.copy_(q)
            self._scale[i] = scale.cpu()
        self._sync()
        self.transfer_seconds += time.perf_counter() - t0

    def get(self, i: int) -> Optional[Tensor]:
        if self._data[i] is None:
            return None
        self._sync()
        t0 = time.perf_counter()
        x = self._data[i].to(self.device, non_blocking=self.pin).float() * self._scale[i].to(self.device)
        full = torch.zeros(self.full_shape, device=self.device, dtype=torch.float32)
        full[:, :, :1] = x
        self._sync()
        self.transfer_seconds += time.perf_counter() - t0
        return full

    def close(self) -> None:
        self._data = [None] * self.num_slots
        self._scale = [None] * self.num_slots

    def describe(self) -> dict:
        return {"location": self.location, "pinned": self.pin, "stored_shape": list(self.stored_shape),
                "full_shape": list(self.full_shape), "bytes_per_view": self.bytes_per_view,
                "num_views": self.num_slots, "total_gib": self.total_bytes / 2 ** 30}


# =====================================================================================================
# configuration and regularization
# =====================================================================================================

@dataclass
class CitrusConfig:
    """Parameters of the CITRUS reconstruction loop (the scatter solver has its own SolverConfig).

    num_outer      : outer iterations N (each = one scatter step per view + ONE MBIR iteration at schedule position k)
    energy_kev     : effective energy E_eff of the density model (None: spectrum mean of the solver's library)
    min_counts     : lower clamp of the (corrected) counts before the log
    base_seed      : numpy's global RNG is seeded with [base_seed, k] before MBIR iteration k (MBIRTORCH draws the
                     partitions from it), so runs with identical inputs give identical MBIR sequences
    gpu_state_budget_mib : the per-view accumulators stay on the GPU while their total is below this, else they go to
                     (pinned) host memory (`StateStore`)
    pin_limit_gib  : largest pinned host allocation for the accumulators
    verbose        : print one line per outer iteration
    """

    num_outer: int = 10
    energy_kev: Optional[float] = None
    min_counts: float = 1.0
    base_seed: int = 0
    gpu_state_budget_mib: float = 2600.0
    pin_limit_gib: float = 8.0
    verbose: bool = False

    def validate(self) -> None:
        if self.num_outer < 1:
            raise ValueError("num_outer must be >= 1")
        if self.min_counts <= 0:
            raise ValueError("min_counts must be > 0")


def estimate_regularization(model, sinogram: Tensor, weights: Optional[Tensor] = None) -> dict:
    """MBIRTORCH's automatic sigma_y / sigma_x / sigma_prox (`auto_set_regularization_params`, what the first `recon`
    call would compute) for the line-integral sinogram [V, Rd, Cd] with `weights`; the values are then frozen in the
    model (auto_regularize_flag False). Returns {"sigma_y", "sigma_x", "sigma_prox"}."""
    model.set_params(no_warning=True, auto_regularize_flag=True)
    model.auto_set_regularization_params(sinogram, weights=weights)
    model.set_params(auto_regularize_flag=False)
    return {k: float(v) for k, v in zip(REGULARIZATION_NAMES, model.get_params(list(REGULARIZATION_NAMES)))}


def apply_regularization(model, reg: dict) -> None:
    """Set the frozen sigma_y, sigma_x, sigma_prox and keep them (auto_regularize_flag False)."""
    model.set_params(no_warning=True, auto_regularize_flag=False, **{k: float(reg[k]) for k in REGULARIZATION_NAMES})


# =====================================================================================================
# reconstruction loop
# =====================================================================================================

ScatterStage = Callable[[int, Tensor, dict], Tensor]


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _outer_loop(Y: Tensor, I0: float, angles: np.ndarray, grid: Grid, geometry: ConeGeometry, config: CitrusConfig,
                scatter_stage: Optional[ScatterStage], device: torch.device, *, regularization: Optional[dict],
                init_recon: Optional[Tensor], model, callback, label: str) -> tuple[Tensor, dict]:
    """The MBIR loop shared by citrus_recon and mbir_recon. scatter_stage(k, X_k, record) -> SPR [V, Rd, Cd] (or None
    for no correction) is called before MBIR iteration k."""
    config.validate()
    W = Y / float(I0)
    y_meas = -torch.log(Y.clamp_min(config.min_counts) / float(I0))
    if model is None:
        model = make_cone_model(grid, geometry, angles, device=device)
    t0 = time.perf_counter()
    if init_recon is None:
        X = fdk(model, y_meas)
    else:
        X = torch.as_tensor(init_recon, dtype=torch.float32, device=device).clone()
    _sync(device)
    info: dict = {"records": [], "fdk_seconds": time.perf_counter() - t0, "init_recon": X.clone(), "model": model,
                  "config": asdict(config)}
    reg = dict(regularization) if regularization is not None else None
    if reg is not None:
        apply_regularization(model, reg)
    rec0 = {"k": 0, "t_scatter": 0.0, "t_mbir": 0.0, "t_total": 0.0}
    info["records"].append(rec0)
    if callback is not None:
        callback(0, X, rec0)
    spr: Optional[Tensor] = None
    S_prev: Optional[Tensor] = None
    S_hat = torch.zeros_like(Y)
    t_start = time.perf_counter()
    for k in range(config.num_outer):
        rec: dict = {"k": k + 1}
        _sync(device)
        t0 = time.perf_counter()
        if scatter_stage is not None:
            spr = scatter_stage(k, X, rec)
        Yc = Y if spr is None else Y / (1.0 + spr)
        y_c = -torch.log(Yc.clamp_min(config.min_counts) / float(I0))
        if reg is None:  # estimated once on the first sinogram MBIR inverts, then frozen
            reg = estimate_regularization(model, y_c, W)
        _sync(device)
        t1 = time.perf_counter()
        np.random.seed([int(config.base_seed), int(k)])
        X, _ = mbir_iterations(model, y_c, X, first_iteration=k, weights=W)
        _sync(device)
        t2 = time.perf_counter()
        S_hat = Y - Yc
        if S_prev is not None:
            den = float(S_hat.abs().mean())
            rec["scatter_change_rel"] = float((S_hat - S_prev).abs().mean()) / den if den > 0 else 0.0
        S_prev = S_hat
        rec.update(t_scatter=t1 - t0, t_mbir=t2 - t1, t_total=t2 - t0)
        if device.type == "cuda":
            rec["gpu_peak_mib"] = torch.cuda.max_memory_allocated(device) / 2 ** 20
            torch.cuda.reset_peak_memory_stats(device)
        info["records"].append(rec)
        if callback is not None:
            callback(k + 1, X, rec)
        if config.verbose:
            extra = f" dPhi {rec['acc_change_mean']:.2e}" if "acc_change_mean" in rec else ""
            print(f"[{label}] outer iteration {k + 1}/{config.num_outer}: scatter {rec['t_scatter']:.1f} s, "
                  f"MBIR {rec['t_mbir']:.1f} s{extra}", flush=True)
    info.update(seconds=time.perf_counter() - t_start, regularization=reg, spr=spr, scatter_counts=S_hat)
    return X, info


def _counts_and_angles(sinogram_counts, angles, device) -> tuple[Tensor, np.ndarray]:
    Y = torch.as_tensor(sinogram_counts, dtype=torch.float32, device=device).contiguous()
    if isinstance(angles, Tensor):
        angles = angles.detach().cpu().numpy()
    ang = np.asarray(angles, dtype=np.float64).reshape(-1)
    if Y.dim() != 3 or Y.shape[0] != ang.size:
        raise ValueError(f"sinogram_counts must be [V = {ang.size}, rows, channels], got {tuple(Y.shape)}")
    return Y, ang


def citrus_recon(sinogram_counts: Union[Tensor, np.ndarray], I0: float, angles: Union[Sequence[float], np.ndarray],
                 grid: Grid, geometry: ConeGeometry, solver: ScatterSolver, config: Optional[CitrusConfig] = None,
                 *, regularization: Optional[dict] = None, init_recon: Optional[Tensor] = None, model=None,
                 callback: Optional[Callable[[int, Tensor, dict], None]] = None) -> tuple[Tensor, dict]:
    """CITRUS reconstruction (module docstring) of measured counts.

    sinogram_counts [V, Rd, Cd]: measured counts Y (primary + scatter, flat-field I0 per pixel); angles [V] (rad);
    grid / geometry: reconstruction grid and cone-beam geometry (MBIRTORCH conventions; also the solver's grid: its
    shape must be divisible by solver.config.supervoxel_factor); solver: the ScatterSolver (its geometry, detector and
    grid response describe the scanner; its device is used for everything); config: CitrusConfig (default
    CitrusConfig()).
    regularization: frozen {"sigma_y", "sigma_x", "sigma_prox"}; None estimates them once (MBIRTORCH's automatic
    estimate) on the first scatter-corrected sinogram and freezes them. init_recon: initial volume [R, C, S]
    (default: FDK of the uncorrected data). model: a prebuilt MBIRTORCH ConeBeamModel (`recon.mbir_bridge.
    make_cone_model(grid, geometry, angles)`). callback(k, X_k, record) is called for the initial volume (k = 0) and
    after every outer iteration.

    Returns (recon [R, C, S] float32 on the solver device, info) with info:
      records          per outer iteration (records[0] = the initial volume): k, t_scatter, t_transfer, t_mbir,
                       t_total, negative_fraction (voxels clamped by the density model), acc_change_mean / max
                       (relative accumulator change over the views), scatter_change_rel (mean |S_k - S_k-1| /
                       mean |S_k| of the scatter-count estimate), gpu_peak_mib
      spr              [V, Rd, Cd] the last scatter-to-primary ratio estimate
      scatter_counts   [V, Rd, Cd] the last scatter-count estimate Y - Y_c
      regularization   the frozen sigmas;  state: the accumulator storage;  init_recon, model, seconds, fdk_seconds
    """
    config = CitrusConfig() if config is None else config
    device = solver.device
    Y, ang = _counts_and_angles(sinogram_counts, angles, device)
    density = DensityModel.from_library(solver.lib, config.energy_kev)
    coarse = grid.coarsen(int(solver.config.supervoxel_factor))
    full_shape = (solver.num_groups, (solver.L + 1) ** 2, 4) + tuple(coarse.shape)
    store = StateStore(len(ang), full_shape, device, gpu_budget_mib=config.gpu_state_budget_mib,
                       pin_limit_gib=config.pin_limit_gib)
    shells = [CitrusState() for _ in range(len(ang))]
    Rd, Cd = int(Y.shape[1]), int(Y.shape[2])
    tiny = torch.finfo(torch.float32).tiny

    def scatter_stage(k: int, X: Tensor, rec: dict) -> Tensor:
        transfer0 = store.transfer_seconds
        rho = density(X)
        rec["negative_fraction"] = density.last_negative_fraction
        fields = solver.material_fields(rho, grid)  # shared by every view of this outer iteration
        s = torch.empty((len(ang), Rd, Cd), device=device, dtype=torch.float32)
        p = torch.empty_like(s)
        hist = []
        for v, a in enumerate(ang):
            st = replace(shells[v], Phi_gt1=store.get(v))
            s[v], p[v], new = citrus_step(solver, st, rho, grid, float(a), fields=fields)
            store.put(v, new.Phi_gt1)
            shells[v] = replace(new, Phi_gt1=None)
            hist.append(new.history[-1])
            del st, new
        rec["acc_change_mean"] = float(np.mean(hist))
        rec["acc_change_max"] = float(np.max(hist))
        rec["t_transfer"] = store.transfer_seconds - transfer0
        return s / p.clamp_min(tiny)

    try:
        X, info = _outer_loop(Y, float(I0), ang, grid, geometry, config, scatter_stage, device,
                              regularization=regularization, init_recon=init_recon, model=model, callback=callback,
                              label="citrus")
        info["state"] = store.describe()
    finally:
        store.close()
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
    Y, ang = _counts_and_angles(sinogram_counts, angles, device)
    fixed = None if spr is None else torch.as_tensor(spr, dtype=torch.float32, device=device)
    if fixed is not None and fixed.shape != Y.shape:
        raise ValueError(f"spr must have the sinogram shape {tuple(Y.shape)}, got {tuple(fixed.shape)}")
    stage = None if fixed is None else (lambda k, X, rec: fixed)
    return _outer_loop(Y, float(I0), ang, grid, geometry, config, stage, device, regularization=regularization,
                       init_recon=init_recon, model=model, callback=callback, label="mbir")
