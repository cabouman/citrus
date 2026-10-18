"""Peak GPU memory and time of the CITRUS outer iterations (``citrus_recon``) on a WLAB grid.

Runs ``citrus_recon`` on scatter-free counts of the WLAB phantom, started from the true attenuation, and splits the
peak memory of every outer iteration into the scatter stage and the MBIR iteration. The first outer iteration
includes the compilation of the kernels.

Usage::

    python benchmarks/bench_citrus_memory.py [--delta-voxel 0.125] [--transport-cell-cm 1.0] [--views 180]
        [--scatter-view-stride 8] [--outer-iterations 2] [--panel full|small] [--memory-cap-mib 0]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import citrus  # noqa: E402
from citrus import citrus as citrus_module  # noqa: E402
from citrus.citrus import CitrusConfig, citrus_recon  # noqa: E402
from citrus.recon.mbir_bridge import forward_project, make_cone_model  # noqa: E402
from citrus.scatter_solver.config import SolverConfig  # noqa: E402
from citrus.utilities import (densities_to_attenuation, full_field_wlab_geometry, set_memory_cap,  # noqa: E402
                              view_angles, wlab_geometry, wlab_grid, wlab_phantom, wlab_solver)

DEVICE = torch.device("cuda")
MIB = 2.0 ** 20


def main() -> None:
    """Parse the command line, run the outer iterations and print the time and peak memory of each."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delta-voxel", type=float, default=0.125,
                        help="fine voxel size (cm), default 0.125 (12.3 M voxels)")
    parser.add_argument("--transport-cell-cm", type=float, default=1.0, help="transport cell edge (cm)")
    parser.add_argument("--views", type=int, default=180)
    parser.add_argument("--scatter-view-stride", type=int, default=8, help="CitrusConfig.scatter_view_stride")
    parser.add_argument("--outer-iterations", type=int, default=2)
    parser.add_argument("--resume-error-sinogram", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--panel", default="full", choices=["full", "small"])
    parser.add_argument("--memory-cap-mib", type=float, default=0.0, help="PyTorch allocator cap (MiB, 0: none)")
    args = parser.parse_args()
    if args.memory_cap_mib > 0:
        set_memory_cap(args.memory_cap_mib)
    geometry = full_field_wlab_geometry() if args.panel == "full" else wlab_geometry()
    grid = wlab_grid(args.delta_voxel)
    solver = wlab_solver(args.delta_voxel, config=replace(SolverConfig(), transport_cell_cm=args.transport_cell_cm),
                         geometry=geometry, device=DEVICE)
    angles = view_angles(args.views, geometry)
    X_true = densities_to_attenuation(solver.library, wlab_phantom(grid, materials=solver.library.materials,
                                                                   device=DEVICE))
    model = make_cone_model(grid, geometry, angles, device=DEVICE)
    I0 = 1.0e5
    Y = I0 * torch.exp(-forward_project(model, X_true))
    extra = {}
    fields = {f.name for f in dataclasses.fields(CitrusConfig)}
    if args.resume_error_sinogram is not None and "resume_error_sinogram" in fields:
        extra["resume_error_sinogram"] = args.resume_error_sinogram
    config = CitrusConfig(num_outer_iterations=args.outer_iterations, scatter_view_stride=args.scatter_view_stride,
                          **extra)
    regularization = {"sigma_y": 0.035, "sigma_x": 0.06, "sigma_prox": 0.06}

    split: list[float] = []
    orig = citrus_module.mbir_iterations

    def wrapped(*a, **k):
        torch.cuda.synchronize(DEVICE)
        split.append(torch.cuda.max_memory_allocated(DEVICE) / MIB)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        return orig(*a, **k)

    citrus_module.mbir_iterations = wrapped
    per_view = X_true.numel() * 4 / MIB
    num_materials = solver.library.num_materials
    print(f"citrus {citrus.__version__}: grid {grid.shape} = {X_true.numel() / 1e6:.1f} M voxels at "
          f"{args.delta_voxel} cm ({per_view:.0f} MiB per fine map, {num_materials * per_view:.0f} MiB for all "
          f"{num_materials}), "
          f"{args.transport_cell_cm:g} cm cells, {args.views} views, panel {geometry.num_det_rows} x "
          f"{geometry.num_det_channels} ({Y.numel() * 4 / MIB:.0f} MiB per sinogram), stride "
          f"{args.scatter_view_stride}, config {extra}", flush=True)
    torch.cuda.synchronize(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    base = torch.cuda.memory_allocated(DEVICE) / MIB

    def report(k, X, record):
        if k == 0:
            torch.cuda.synchronize(DEVICE)
            torch.cuda.reset_peak_memory_stats(DEVICE)
            return
        print(f"  outer {k}: scatter stage {record['scatter_seconds']:.2f} s peak {split[k - 1]:.0f} MiB | MBIR "
              f"{record['mbir_seconds']:.2f} s peak {record['gpu_peak_allocated_mib']:.0f} MiB | allocated after "
              f"{torch.cuda.memory_allocated(DEVICE) / MIB:.0f} MiB (before the loop {base:.0f} MiB)", flush=True)

    t0 = time.perf_counter()
    X, info = citrus_recon(Y, I0, angles, grid, geometry, solver, config, regularization=regularization,
                           init_recon=X_true,
                           model=model, callback=report)
    torch.cuda.synchronize(DEVICE)
    print(f"total {time.perf_counter() - t0:.1f} s; relative change of X: "
          f"{float((X - X_true).norm() / X_true.norm()):.3e}")


if __name__ == "__main__":
    main()
