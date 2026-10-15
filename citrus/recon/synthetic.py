"""Synthetic WLAB scatter dataset for CITRUS (docs/source/usage.rst).

    rho_true  = wlab_phantom(wlab_grid(delta_voxel))                     [2, R, C, S] g/cm^3
    X_true    = densities_to_attenuation(lib, rho_true, E_eff)           [R, C, S] 1/cm, E_eff = spectrum mean
    l         = A X_true  (MBIRTORCH ConeBeamModel forward projection)   [V, Rd, Cd] line integrals
    P         = I0 exp(-l)                                               primary counts per pixel
    SPR       = s_sol / p_sol per view (ScatterSolver.solve_view on rho_true, converged source iteration, ideal
                counting detector, no grid): ratio of the solver's scatter and primary PLANAR fluence images
    S         = SPR P,   Y_mean = P + S,   Y = Poisson(Y_mean) (noise=True) or Y_mean

Units bookkeeping. The solver's images are photons/cm^2 per source photon/sr (docs/source/conventions.rst). Its
primary planar
fluence is p_sol(u) = cos(theta0) / |r_D - r_0|^2 x T_poly(u): it carries the inverse-square and obliquity factors
of the point source on a flat panel and the polychromatic transmission T_poly = sum_E w_E exp(-sum_m m_m(E) P_m).
The scatter planar fluence s_sol is in the same units. MBIRTORCH's model is the flat, monochromatic line integral
l = A X at E_eff with "I0 exp(-l)" meaning flat-field-normalized counts: I0 is the per-pixel count of the air scan,
so the inverse-square / obliquity factor phi_air(u) = cos(theta0)/r^2 (the solver's unattenuated planar fluence,
saved as `phi_air`) is absorbed by the flat field. The dataset ties the two models through the dimensionless
per-pixel SPR, S = SPR x P: the scatter-to-primary ratio of the data equals the solver's physical SPR exactly. The
CITRUS correction Y_c = Y p / (p + s) with s = SPR(X) p then is Y_c = Y / (1 + SPR(X)) with SPR(X) from
`citrus_step`'s (scatter, primary) signals: independent of I0 and of the mono/poly primary mismatch.
(The alternative S' = I0 s_sol / phi_air, the scatter relative to the flat field, differs from S by the factor
exp(-l) / (p_sol / phi_air) = mono / poly transmission; `solver_primary` and `phi_air` are saved so it can be formed.)

Inverse crime (intentional): the primary is generated with MBIRTORCH's own projector at E_eff, the operator the
reconstruction inverts, so the only model error the CITRUS loop sees is the scatter (and the noise). The truth is
on the reconstruction grid (no discretization mismatch) and the primary is monoenergetic (no beam hardening).
Field of view: the default geometry is `utilities.full_field_wlab_geometry()` (150 x 184 pixels of 0.4 cm), which keeps
the whole prism on the panel at every view. The 100 x 75 panel of `utilities.wlab_geometry()` sees a 26.7 cm wide,
20 cm long field at the isocenter, smaller than the 30 x 20 x 40 cm prism (truncated projections).
Reproducibility: the Poisson draws use torch.Generator(seed), but the rates Y_mean themselves vary between runs at the
float32 ulp level (atomic accumulation in MBIRTORCH's and the solver's CUDA kernels), which flips a few draws in
10^4 (7 of 30,000 measured); save and reuse the npz rather than regenerating when bitwise identity matters.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, replace
from typing import Callable, Optional, Union

import numpy as np
import torch
from torch import Tensor

from ..scatter_solver.config import SolverConfig
from ..scatter_solver.cross_sections import MultigroupLibrary
from ..scatter_solver.detector import ideal_counting_response, no_grid
from ..scatter_solver.geometry import ConeGeometry
from ..scatter_solver.scatter_solver import ScatterSolver, build_solver_library
from ..utilities import (densities_to_attenuation, effective_energy, full_field_wlab_geometry, mass_attenuation_at,
                         wlab_grid, wlab_phantom)
from .mbir_bridge import forward_project, make_cone_model, mbirtorch_version

__all__ = ["default_supervoxel_factor", "wlab_solver", "synthesize_wlab_dataset", "load_dataset"]


def default_supervoxel_factor(delta_voxel: float) -> int:
    """Supervoxel factor giving ~1 cm transport cells: round(1 / delta_voxel), at least 1 (0.5 cm -> 2, 1 cm -> 1,
    2 cm -> 1)."""
    return max(1, int(round(1.0 / float(delta_voxel))))


def wlab_solver(delta_voxel: float, *, config: Optional[SolverConfig] = None, lib: Optional[MultigroupLibrary] = None,
                geom: Optional[ConeGeometry] = None, device=None,
                cache_dir: Optional[str] = "default") -> ScatterSolver:
    """ScatterSolver for the synthetic data: the given config (default SolverConfig() with the supervoxel factor of
    `default_supervoxel_factor`), the real library (build_solver_library, cached in `cache_dir`; "default" is
    $CITRUS_CACHE_DIR or ~/.cache/citrus), the full-field WLAB geometry (default), an ideal counting detector and no
    grid, float32."""
    cfg = config if config is not None else replace(SolverConfig(),
                                                    supervoxel_factor=default_supervoxel_factor(delta_voxel))
    lib = lib if lib is not None else build_solver_library(cfg, cache_dir=cache_dir)
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return ScatterSolver(lib, cfg, geom if geom is not None else full_field_wlab_geometry(),
                         detector=ideal_counting_response(), grid_response=no_grid(), device=dev,
                         dtype=torch.float32)


def synthesize_wlab_dataset(num_views: int = 180, delta_voxel: float = 0.5, I0: float = 1.0e5, noise: bool = True,
                            seed: int = 0, out_path: Optional[Union[str, os.PathLike]] = None, *,
                            energy_kev: Optional[float] = None, config: Optional[SolverConfig] = None,
                            lib: Optional[MultigroupLibrary] = None, device=None, angle_offset: float = 0.0,
                            progress: Optional[Callable[[int, float], None]] = None,
                            geom: Optional[ConeGeometry] = None) -> dict:
    """Generate the synthetic WLAB dataset (module docstring) and optionally save it.

    num_views angles uniformly over [0, 2 pi) (+ angle_offset); delta_voxel: WLAB grid pitch (0.5 cm: 40 x 60 x 80
    voxels); I0: flat-field counts per pixel; noise: Poisson(Y_mean) with torch.Generator(seed) (else Y = Y_mean);
    energy_kev: E_eff (None: spectrum mean of the library, 63.03 keV); config / lib: solver config and library
    (default: `wlab_solver` defaults). progress(view_index, seconds) is called after every solver view.
    geom: cone geometry of the scan (default `utilities.full_field_wlab_geometry()`, 150 x 184 pixels of 0.4 cm); it is
    used by both the MBIRTORCH projector and the solver, and saved in meta["geometry"].

    Returns a dict of numpy arrays (float32 unless noted):
      X_true [R, C, S] (1/cm), rho_true [2, R, C, S], angles [V] (float64), line_integrals [V, Rd, Cd] (A X_true),
      (Rd, Cd = geom.num_det_rows, geom.num_det_channels)
      P, S, Y, Y_mean, SPR [V, Rd, Cd], solver_primary, solver_scatter [V, Rd, Cd] (planar fluence summed over groups,
      photons/cm^2 per source photon/sr), phi_air [Rd, Cd] (solver primary of an empty volume), view_seconds [V],
      si_iterations [V] (int), I0, energy_kev, mass_attenuation [2] (m_w, m_b at E_eff, cm^2/g) (0-d / 1-d float64),
      meta (dict: geometry, grid, solver config, versions, timings, statistics).
    out_path: if given, writes <out_path>.npz (everything above, meta as a JSON string `meta_json`) and
    <out_path>.json (meta).
    """
    t_all = time.perf_counter()
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geom = full_field_wlab_geometry() if geom is None else geom
    grid = wlab_grid(delta_voxel)
    solver = wlab_solver(delta_voxel, config=config, lib=lib, geom=geom, device=dev)
    lib = solver.lib
    E = effective_energy(lib) if energy_kev is None else float(energy_kev)
    m = mass_attenuation_at(lib, E)

    rho = wlab_phantom(grid, device=dev, dtype=torch.float32)
    X = densities_to_attenuation(lib, rho, E)
    angles = (float(angle_offset) + 2.0 * math.pi * np.arange(num_views) / num_views).astype(np.float64)

    # primary: MBIRTORCH forward projection (the reconstruction's own operator)
    t0 = time.perf_counter()
    model = make_cone_model(grid, geom, angles, device=dev)
    line = forward_project(model, X).to(dev)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t_project = time.perf_counter() - t0
    P = float(I0) * torch.exp(-line)

    # scatter-to-primary ratio from the solver, one converged solve per view
    V, Rd, Cd = num_views, geom.num_det_rows, geom.num_det_channels
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
        "description": "synthetic WLAB scatter dataset, citrus.recon.synthetic",
        "num_views": int(V), "delta_voxel": float(delta_voxel), "I0": float(I0), "noise": bool(noise),
        "seed": int(seed), "angle_offset": float(angle_offset), "energy_kev": E,
        "mass_attenuation_water_bone": [float(m[0]), float(m[1])],
        "geometry": asdict(geom), "grid": asdict(grid), "solver_config": solver.config.to_dict(),
        "detector": "ideal counting", "grid_response": "none", "solver_dtype": "float32",
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
    out = {
        "X_true": a(X), "rho_true": a(rho), "angles": angles, "line_integrals": a(line), "P": a(P), "S": a(S),
        "Y": a(Y), "Y_mean": a(Y_mean), "SPR": spr, "solver_primary": a(p_sol), "solver_scatter": a(s_sol),
        "phi_air": a(phi_air), "view_seconds": view_seconds, "si_iterations": si_iters,
        "I0": np.asarray(float(I0)), "energy_kev": np.asarray(E), "mass_attenuation": m.numpy(), "meta": meta,
    }
    if out_path is not None:
        base = str(out_path)
        for ext in (".npz", ".json"):
            if base.endswith(ext):
                base = base[: -len(ext)]
        os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
        arrays = {k: v for k, v in out.items() if k != "meta"}
        np.savez(base + ".npz", meta_json=np.asarray(json.dumps(meta)), **arrays)
        with open(base + ".json", "w") as f:
            json.dump(meta, f, indent=2)
        meta["files"] = [base + ".npz", base + ".json"]
    return out


def load_dataset(path: Union[str, os.PathLike]) -> dict:
    """Read a dataset written by `synthesize_wlab_dataset` (<path>.npz): dict of numpy arrays plus `meta`."""
    p = str(path)
    if not p.endswith(".npz"):
        p += ".npz"
    with np.load(p) as z:
        out = {k: z[k] for k in z.files if k != "meta_json"}
        out["meta"] = json.loads(str(z["meta_json"]))
    return out
