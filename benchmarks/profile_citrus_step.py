"""Per-stage time profile of the CITRUS step (``citrus_step``) on the WLAB phantom.

Runs every view once to build its accumulator, then times the plain ``citrus_step`` per view and the same step taken
apart, synchronized around every stage and, in a second pass, also around the inner stages of the scatter image and
the sweep's first-collision source. Prints the time per view of every stage in ms.

Usage::

    python benchmarks/profile_citrus_step.py [--panel full|small] [--views 12] [--cone 16x32] [--delta-voxel 0.5]
        [--transport-cell-cm 1.0] [--quadrature-spacing cm] [--uncollided-trace quadrature|coarse|fine]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import replace
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from citrus import citrus as citrus_module  # noqa: E402
from citrus.citrus import CitrusState, StateStore, citrus_step  # noqa: E402
from citrus.utilities import wlab_solver  # noqa: E402
from citrus.scatter_solver import scatter_image as scatter_image_module  # noqa: E402
from citrus.scatter_solver import scatter_solver as scatter_solver_module  # noqa: E402
from citrus.scatter_solver.config import SolverConfig  # noqa: E402
from citrus.scatter_solver.cross_section_library import load_spectrum_txt  # noqa: E402
from citrus.scatter_solver.scatter_image import covering_polar_angle  # noqa: E402
from citrus.utilities import (attenuation_to_densities, densities_to_attenuation,  # noqa: E402
                              full_field_wlab_geometry, set_memory_cap, wlab_geometry, wlab_grid, wlab_phantom)

SPECTRUM = load_spectrum_txt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo", "spectra",
                                          "spectrum_125kvp.txt"))  # the sample 125 kVp spectrum

DEVICE = torch.device("cuda")
T: dict[str, float] = defaultdict(float)
INSTRUMENT = [False]
CUR = [""]


def sync() -> None:
    """Wait for every CUDA kernel on the device to finish."""
    torch.cuda.synchronize(DEVICE)


class stage:
    """Context manager that adds the synchronized wall time of its body to the stage total ``T[name]``.

    Attributes:
        name (str): stage name, the key in ``T``.
        t0 (float): start time in s, set on entry.
    """

    def __init__(self, name: str) -> None:
        """Name the stage.

        Args:
            name (str): stage name.
        """
        self.name = name

    def __enter__(self):
        """Make this the current stage, synchronize and start the clock."""
        CUR[0] = self.name
        sync()
        self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        """Synchronize and add the elapsed time to the stage total.

        Args:
            *exc: exception type, value and traceback (ignored).
        """
        sync()
        T[self.name] += time.perf_counter() - self.t0


def wrap(obj, attr: str, name: str) -> None:
    """Replace a function by a wrapper that, while instrumenting, adds its synchronized time to the current stage.

    Args:
        obj (object): module, class or object that holds the function.
        attr (str): attribute name of the function.
        name (str): inner stage name.
    """
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
    """Parse the command line, build the accumulators, then time and print the step per stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="full", choices=["full", "small"])
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--cone", default="16x32")
    parser.add_argument("--delta-voxel", type=float, default=0.5, help="fine voxel size (cm)")
    parser.add_argument("--transport-cell-cm", type=float, default=None,
                        help="transport cell edge in cm (default: SolverConfig's)")
    parser.add_argument("--quadrature-spacing", type=float, default=None,
                        help="SolverConfig.quadrature_point_spacing_cm: target spacing of the quadrature points (cm); "
                             "q = round(cell / spacing), never above round(cell / voxel)")
    parser.add_argument("--uncollided-trace", default=None, choices=["quadrature", "coarse", "fine"],
                        help="SolverConfig.uncollided_trace_grid")
    parser.add_argument("--memory-cap-mib", type=float, default=4400.0, help="PyTorch allocator cap (MiB)")
    args = parser.parse_args()
    set_memory_cap(args.memory_cap_mib)
    cone_polar, cone_azimuthal = (int(x) for x in args.cone.split("x"))
    geometry = full_field_wlab_geometry() if args.panel == "full" else wlab_geometry()
    grid = wlab_grid(args.delta_voxel)
    extra = {}
    if args.transport_cell_cm is not None:
        extra["transport_cell_cm"] = args.transport_cell_cm
    if args.quadrature_spacing is not None:
        extra["quadrature_point_spacing_cm"] = args.quadrature_spacing
    if args.uncollided_trace is not None:
        extra["uncollided_trace_grid"] = args.uncollided_trace
    config = replace(SolverConfig(), cone_num_polar=cone_polar, cone_num_azimuthal=cone_azimuthal, **extra)
    solver = wlab_solver(args.delta_voxel, config=config, geometry=geometry, device=DEVICE, spectrum=SPECTRUM)
    library = solver.library
    rho_true = wlab_phantom(grid, device=DEVICE)
    X = densities_to_attenuation(library, rho_true)
    angles = 2 * np.pi * np.arange(180) / 180
    views = angles[:: 180 // args.views]
    quadrature = solver.cone_quadrature(grid, float(views[0]))
    cap = covering_polar_angle(grid, geometry, float(views[0]))
    f0 = solver.material_fields(rho_true, grid)
    context0 = solver.direction_context(f0, float(views[0]), quadrature)
    print(f"grid {grid.shape} at {args.delta_voxel} cm, transport grid {f0.grid_coarse.shape} of "
          f"{config.transport_cell_cm:g} cm "
          f"cells, quadrature grid {f0.grid_quadrature.shape} (q = {f0.q}), uncollided trace through the "
          f"{config.uncollided_trace_grid} grid")
    print(f"panel {args.panel} {geometry.num_det_rows} x {geometry.num_det_channels}; "
          f"cone {cone_polar} x {cone_azimuthal}: K = {quadrature.omega.shape[0]} directions in the set, "
          f"{context0.num_directions} panel-facing, covering polar angle "
          f"{np.degrees(cap):.1f} deg at view 0 (max over views "
          f"{max(np.degrees(covering_polar_angle(grid, geometry, float(a))) for a in views):.1f}); direction context "
          f"{context0.nbytes / 2 ** 20:.1f} MiB", flush=True)
    del f0, context0, rho_true

    # step-3 inner stages and the sweep's first-collision source (module-level names as the callers see them)
    wrap(scatter_image_module, "first_collision_source_at_directions", "first-collision source")
    wrap(solver.projectors, "downstream_cumulative", "downstream attenuation")
    wrap(scatter_image_module, "supervoxel_subpoint_emission", "sub-point emission")
    wrap(solver.projectors, "oblique_project", "oblique projection")
    wrap(scatter_image_module.DirectionSetContext, "accumulate", "accumulation")
    wrap(scatter_solver_module, "first_collision_source_at_directions", "first-collision source (solver binding)")
    wrap(scatter_solver_module, "restrict_to_linear_discontinuous_moments",
         "restrict first-collision to linear-discontinuous moments")

    # pass 1: build the accumulators (also compiles everything)
    states = {}
    rho = attenuation_to_densities(library, X.clamp_min(0))
    for a in views:
        _, _, states[a] = citrus_step(solver, CitrusState(), rho, grid, float(a))
    full_shape = tuple(states[views[0]].collided_moments.shape)
    stores = {loc: StateStore(len(views), full_shape, DEVICE, location=loc) for loc in ("gpu", "cpu")}

    # pass 2a: plain citrus_step
    torch.cuda.reset_peak_memory_stats(DEVICE)
    plain = []
    for a in views:
        sync()
        t0 = time.perf_counter()
        s, p, new_state = citrus_step(solver, states[a], rho, grid, float(a))
        sync()
        plain.append(time.perf_counter() - t0)
        states[a] = new_state
    del s, p, new_state, rho
    peak_plain = torch.cuda.max_memory_allocated(DEVICE) / 2 ** 20
    print(f"\nplain citrus_step: {1e3 * np.mean(plain):.0f} ms per view (min {1e3 * min(plain):.0f}, max "
          f"{1e3 * max(plain):.0f}); x180 = {180 * np.mean(plain):.1f} s per outer iteration; peak memory "
          f"{peak_plain:.0f} MiB (accumulators of {len(views)} views included)", flush=True)
    # pass 2b / 2c: every top-level stage synchronized (coarse), then also the inner stages (fine; the extra
    # synchronizations per direction chunk inflate the inner totals)
    def instrumented(inner: bool) -> tuple[dict, float]:
        INSTRUMENT[0] = inner
        T.clear()
        wall = 0.0
        for j, a in enumerate(views):
            a = float(a)
            view_state = states[views[j]]
            sync()
            view_start = time.perf_counter()
            with stage("density conversion (X -> rho)"):
                rho = attenuation_to_densities(library, X.clamp_min(0))
            with stage("material_fields"):
                fields = solver.material_fields(rho, grid)
            with stage("cone quadrature (cached)"):
                q = (view_state.cone_quadrature if view_state.cone_quadrature is not None
                     else solver.cone_quadrature(grid, a))
            with stage("uncollided_operator uncollided"):
                phi_uncollided1 = citrus_module.uncollided_operator(solver, rho, grid, a, fields)
            with stage("first-collision sampling correction (standalone, repeated inside accumulator_update_operator)"):
                solver.first_collision_sampling_correction(fields, a, phi_uncollided1)
            with stage("accumulator_update_operator total (first-collision corr + sweep + sources)"):
                new_collided_moments = citrus_module.accumulator_update_operator(solver, rho, grid, a, phi_uncollided1,
                                                                                 view_state.collided_moments,
                                                                                 fields)
            with stage("multiple-collision source moments"):
                scatter_source_moments = solver.multiple_collision_source_moments(fields, new_collided_moments)
            with stage("direction context (LRU of 8: rebuilt per view)"):
                direction_context = solver.direction_context(fields, a, q)
            with stage("scatter_image total"):
                scatter = solver.scatter_image(fields, a, phi_uncollided1, scatter_source_moments, q,
                                               direction_context)["signal"]
            with stage("primary"):
                primary = solver.primary(fields, a)["signal"]
            with stage("history norms"):
                float(torch.linalg.vector_norm(new_collided_moments[:, 0, 0] - view_state.collided_moments[:, 0, 0]))
                float(torch.linalg.vector_norm(new_collided_moments[:, 0, 0]))
            with stage("SPR / correction arithmetic"):
                spr = scatter / primary.clamp_min(1e-38)
                _ = 1.0 / (1.0 + spr)
            for name, store in stores.items():
                with stage(f"compact state put+get ({name})"):
                    store.put(j, new_collided_moments)
                    store.get(j)
            sync()
            wall += time.perf_counter() - view_start
            del phi_uncollided1, scatter_source_moments, scatter, primary, fields, rho, new_collided_moments, spr
        INSTRUMENT[0] = False
        return dict(T), wall

    T_coarse, wall_c = instrumented(False)
    T_fine, wall_f = instrumented(True)
    n = len(views)
    print(f"stage-synchronized wall {1e3 * wall_c / n:.0f} ms per view, with inner stages {1e3 * wall_f / n:.0f} ms "
          f"(both include the two state stores and the repeated first-collision correction)")
    print(f"{'stage':90s} {'coarse':>8s} {'inner':>8s}  (ms/view)")
    for k in T_fine:
        c = T_coarse.get(k)
        print(f"{k:90s} {'' if c is None else format(1e3 * c / n, '8.1f'):>8s} {1e3 * T_fine[k] / n:8.1f}")


if __name__ == "__main__":
    main()
