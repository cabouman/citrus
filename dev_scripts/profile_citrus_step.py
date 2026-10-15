"""Per-stage profile of one CITRUS step per view (citrus.citrus.citrus_step).

    python dev_scripts/profile_citrus_step.py [--panel full|small] [--views 12] [--cone 16x32]

Synthesizes nothing: uses the WLAB phantom on the 0.5 cm grid (supervoxel factor 2), the full-field (150 x 184) or the
small (100 x 75) panel, and `--views` of 180 equally spaced angles (every 180/views-th). Pass 1 builds every view's
accumulator (one citrus_step each); pass 2 is timed: first the plain `citrus_step` (synchronized around the call),
then the same step taken apart with torch.cuda.synchronize around every stage, with the scatter-image inner stages
(FC source at the cone directions, downstream attenuation, sub-point emission, oblique projection, accumulation) and
the sweep's FC-source evaluations timed by wrapping those functions. Prints a per-view table (ms).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from citrus import citrus as citrus_hooks  # noqa: E402
from citrus.citrus import CitrusState, StateStore, citrus_step  # noqa: E402
from citrus.recon.synthetic import wlab_solver  # noqa: E402
from citrus.scatter_solver import scatter_image as step3  # noqa: E402
from citrus.scatter_solver import scatter_solver as solver_mod  # noqa: E402
from citrus.scatter_solver.config import SolverConfig  # noqa: E402
from citrus.scatter_solver.scatter_image import covering_polar_angle  # noqa: E402
from citrus.utilities import (attenuation_to_densities, densities_to_attenuation, full_field_wlab_geometry,  # noqa: E402
                              set_memory_cap, wlab_geometry, wlab_grid, wlab_phantom)

DEV = torch.device("cuda")
T: dict[str, float] = defaultdict(float)
INSTRUMENT = [False]
CUR = [""]


def sync() -> None:
    torch.cuda.synchronize(DEV)


class stage:
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        CUR[0] = self.name
        sync()
        self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        sync()
        T[self.name] += time.perf_counter() - self.t0


def wrap(obj, attr: str, name: str) -> None:
    fn = getattr(obj, attr)

    def wrapped(*a, **k):
        if not INSTRUMENT[0]:
            return fn(*a, **k)
        sync()
        t0 = time.perf_counter()
        out = fn(*a, **k)
        sync()
        T[f"    {CUR[0]} :: {name}"] += time.perf_counter() - t0
        return out

    setattr(obj, attr, wrapped)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="full", choices=["full", "small"])
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--cone", default="16x32")
    args = ap.parse_args()
    set_memory_cap(4400)
    cp, ca = (int(x) for x in args.cone.split("x"))
    geom = full_field_wlab_geometry() if args.panel == "full" else wlab_geometry()
    grid = wlab_grid(0.5)
    cfg = replace(SolverConfig(), supervoxel_factor=2, cone_num_polar=cp, cone_num_azimuthal=ca)
    solver = wlab_solver(0.5, config=cfg, geom=geom, device=DEV)
    lib = solver.lib
    rho_true = wlab_phantom(grid, device=DEV)
    X = densities_to_attenuation(lib, rho_true)
    angles = 2 * np.pi * np.arange(180) / 180
    views = angles[:: 180 // args.views]
    quad = solver.cone_quadrature(grid, float(views[0]))
    cap = covering_polar_angle(grid, geom, float(views[0]))
    f0 = solver.material_fields(rho_true, grid)
    ctx0 = solver.direction_context(f0, float(views[0]), quad)
    print(f"panel {args.panel} {geom.num_det_rows} x {geom.num_det_channels}; cone {cp} x {ca}: K = {quad.omega.shape[0]} "
          f"directions in the set, {ctx0.num_directions} panel-facing, covering polar angle "
          f"{np.degrees(cap):.1f} deg at view 0 (max over views "
          f"{max(np.degrees(covering_polar_angle(grid, geom, float(a))) for a in views):.1f}); direction context "
          f"{ctx0.nbytes / 2 ** 20:.1f} MiB", flush=True)
    del f0, ctx0

    # step-3 inner stages and the sweep's FC source (module-level names as the callers see them)
    wrap(step3, "fc_source_at_directions", "FC source")
    wrap(solver.projectors, "downstream_cumulative", "downstream attenuation")
    wrap(step3, "supervoxel_subpoint_emission", "sub-point emission")
    wrap(solver.projectors, "oblique_project", "oblique projection")
    wrap(step3.DirectionSetContext, "accumulate", "accumulation")
    wrap(solver_mod, "fc_source_at_directions", "FC source (solver binding)")
    wrap(solver_mod, "restrict_fc_to_ld_moments", "restrict FC to LD moments")

    # pass 1: build the accumulators (also compiles everything)
    states = {}
    rho = attenuation_to_densities(lib, X.clamp_min(0))
    for a in views:
        _, _, states[a] = citrus_step(solver, CitrusState(), rho, grid, float(a))
    full_shape = tuple(states[views[0]].Phi_gt1.shape)
    stores = {loc: StateStore(len(views), full_shape, DEV, location=loc) for loc in ("gpu", "cpu")}

    # pass 2a: plain citrus_step
    plain = []
    for a in views:
        sync()
        t0 = time.perf_counter()
        s, p, new = citrus_step(solver, states[a], rho, grid, float(a))
        sync()
        plain.append(time.perf_counter() - t0)
        states[a] = new
    # pass 2b / 2c: every top-level stage synchronized (coarse), then also the inner stages (fine; the extra
    # synchronizations per direction chunk inflate the inner totals)
    def instrumented(inner: bool) -> tuple[dict, float]:
        INSTRUMENT[0] = inner
        T.clear()
        wall = 0.0
        for j, a in enumerate(views):
            a = float(a)
            st = states[views[j]]
            sync()
            tv = time.perf_counter()
            with stage("density conversion (X -> rho)"):
                rho = attenuation_to_densities(lib, X.clamp_min(0))
            with stage("material_fields"):
                fields = solver.material_fields(rho, grid)
            with stage("cone quadrature (cached)"):
                q = st.cone_quad if st.cone_quad is not None else solver.cone_quadrature(grid, a)
            with stage("G1 uncollided"):
                phi1 = citrus_hooks.G1(solver, rho, grid, a, fields)
            with stage("FC sampling correction (standalone, repeated inside G_step)"):
                solver.fc_sampling_correction(fields, a, phi1)
            with stage("G_step total (FC corr + sweep + sources)"):
                Phi_new = citrus_hooks.G_step(solver, rho, grid, a, phi1, st.Phi_gt1, fields)
            with stage("MC source moments"):
                T_mean = solver.mc_source_moments(fields, Phi_new)
            with stage("direction context (LRU of 8: rebuilt per view)"):
                dctx = solver.direction_context(fields, a, q)
            with stage("scatter_image total"):
                sc = solver.scatter_image(fields, a, phi1, T_mean, q, dctx)["signal"]
            with stage("primary"):
                pr = solver.primary(fields, a)["signal"]
            with stage("history norms"):
                float(torch.linalg.vector_norm(Phi_new[:, 0, 0] - st.Phi_gt1[:, 0, 0]))
                float(torch.linalg.vector_norm(Phi_new[:, 0, 0]))
            with stage("SPR / correction arithmetic"):
                spr = sc / pr.clamp_min(1e-38)
                _ = 1.0 / (1.0 + spr)
            for name, store in stores.items():
                with stage(f"compact state put+get ({name})"):
                    store.put(j, Phi_new)
                    store.get(j)
            sync()
            wall += time.perf_counter() - tv
            del phi1, T_mean, sc, pr, fields
        INSTRUMENT[0] = False

        INSTRUMENT[0] = False
        return dict(T), wall

    T_coarse, wall_c = instrumented(False)
    T_fine, wall_f = instrumented(True)
    n = len(views)
    print(f"\nplain citrus_step: {1e3 * np.mean(plain):.0f} ms per view (min {1e3 * min(plain):.0f}, max "
          f"{1e3 * max(plain):.0f}); x180 = {180 * np.mean(plain):.1f} s per outer iteration")
    print(f"stage-synchronized wall {1e3 * wall_c / n:.0f} ms per view, with inner stages {1e3 * wall_f / n:.0f} ms "
          f"(both include the two state stores and the repeated FC correction)")
    print(f"{'stage':90s} {'coarse':>8s} {'inner':>8s}  (ms/view)")
    for k in T_fine:
        c = T_coarse.get(k)
        print(f"{k:90s} {'' if c is None else format(1e3 * c / n, '8.1f'):>8s} {1e3 * T_fine[k] / n:8.1f}")


if __name__ == "__main__":
    main()
