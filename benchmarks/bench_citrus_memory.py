"""Peak GPU memory and time of CITRUS outer iterations on a WLAB grid (citrus.citrus_recon).

    python benchmarks/bench_citrus_memory.py [--voxel 0.125] [--cell 1.0] [--views 180] [--stride 8] [--outer 2]
        [--primary-projector mbirtorch|siddon] [--resume-error-sinogram | --no-resume-error-sinogram]
        [--panel full|small] [--memory-cap 0]

Synthesizes scatter-free counts Y = I0 exp(-A X_true) of the WLAB phantom on the `--voxel` grid (A the MBIRTORCH
projector; the memory and the time of the loop do not depend on the scatter in the data), starts from X_true (no FDK)
and runs `--outer` outer iterations of `citrus_recon` with the scatter step on every `--stride`-th view. The MBIR call
is wrapped so that the peak memory is split into the scatter stage (from the start of the outer iteration to the MBIR
call: densities, material fields, the scatter steps, the primaries, the correction) and the MBIR iteration. Prints one
line per outer iteration (the first one includes the compilation of the kernels: read the later ones) and the
persistent allocation between iterations. `--primary-projector` / `--resume-error-sinogram` set
CitrusConfig.primary_projector and CitrusConfig.resume_error_sinogram (version 0.3.0; ignored by older versions,
which always trace the Siddon primary
and restart MBIR from the sinogram).
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
from citrus import citrus as citrus_mod  # noqa: E402
from citrus.citrus import CitrusConfig, citrus_recon  # noqa: E402
from citrus.recon.mbir_bridge import forward_project, make_cone_model  # noqa: E402
from citrus.scatter_solver.config import SolverConfig  # noqa: E402
from citrus.utilities import (densities_to_attenuation, full_field_wlab_geometry, set_memory_cap,  # noqa: E402
                              view_angles, wlab_geometry, wlab_grid, wlab_phantom, wlab_solver)

DEV = torch.device("cuda")
MIB = 2.0 ** 20


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxel", type=float, default=0.125, help="fine voxel size (cm), default 0.125 (12.3 M voxels)")
    ap.add_argument("--cell", type=float, default=1.0, help="transport cell edge (cm)")
    ap.add_argument("--views", type=int, default=180)
    ap.add_argument("--stride", type=int, default=8, help="CitrusConfig.scatter_view_stride")
    ap.add_argument("--outer", type=int, default=2)
    ap.add_argument("--primary-projector", default=None, choices=["mbirtorch", "siddon"])
    ap.add_argument("--resume-error-sinogram", default=None, action=argparse.BooleanOptionalAction)
    ap.add_argument("--panel", default="full", choices=["full", "small"])
    ap.add_argument("--memory-cap", type=float, default=0.0, help="PyTorch allocator cap (MiB, 0: none)")
    args = ap.parse_args()
    if args.memory_cap > 0:
        set_memory_cap(args.memory_cap)
    geom = full_field_wlab_geometry() if args.panel == "full" else wlab_geometry()
    grid = wlab_grid(args.voxel)
    solver = wlab_solver(args.voxel, config=replace(SolverConfig(), transport_cell_cm=args.cell), geom=geom,
                         device=DEV)
    angles = view_angles(args.views, geom)
    X_true = densities_to_attenuation(solver.lib, wlab_phantom(grid, device=DEV))
    model = make_cone_model(grid, geom, angles, device=DEV)
    I0 = 1.0e5
    Y = I0 * torch.exp(-forward_project(model, X_true))
    extra = {}
    fields = {f.name for f in dataclasses.fields(CitrusConfig)}
    if args.primary_projector is not None and "primary_projector" in fields:
        extra["primary_projector"] = args.primary_projector
    if args.resume_error_sinogram is not None and "resume_error_sinogram" in fields:
        extra["resume_error_sinogram"] = args.resume_error_sinogram
    cfg = CitrusConfig(num_outer=args.outer, scatter_view_stride=args.stride, **extra)
    reg = {"sigma_y": 0.035, "sigma_x": 0.06, "sigma_prox": 0.06}

    split: list[float] = []
    orig = citrus_mod.mbir_iterations

    def wrapped(*a, **k):
        torch.cuda.synchronize(DEV)
        split.append(torch.cuda.max_memory_allocated(DEV) / MIB)
        torch.cuda.reset_peak_memory_stats(DEV)
        return orig(*a, **k)

    citrus_mod.mbir_iterations = wrapped
    per_view = X_true.numel() * 4 / MIB
    print(f"citrus {citrus.__version__}: grid {grid.shape} = {X_true.numel() / 1e6:.1f} M voxels at {args.voxel} cm "
          f"({per_view:.0f} MiB per fine map, {7 * per_view:.0f} MiB for seven), {args.cell:g} cm cells, "
          f"{args.views} views, panel {geom.num_det_rows} x {geom.num_det_channels} "
          f"({Y.numel() * 4 / MIB:.0f} MiB per sinogram), stride {args.stride}, config {extra}", flush=True)
    torch.cuda.synchronize(DEV)
    torch.cuda.reset_peak_memory_stats(DEV)
    base = torch.cuda.memory_allocated(DEV) / MIB

    def cb(k, X, rec):
        if k == 0:
            torch.cuda.synchronize(DEV)
            torch.cuda.reset_peak_memory_stats(DEV)
            return
        print(f"  outer {k}: scatter stage {rec['t_scatter']:.2f} s peak {split[k - 1]:.0f} MiB | MBIR "
              f"{rec['t_mbir']:.2f} s peak {rec['gpu_peak_mib']:.0f} MiB | allocated after "
              f"{torch.cuda.memory_allocated(DEV) / MIB:.0f} MiB (before the loop {base:.0f} MiB)", flush=True)

    t0 = time.perf_counter()
    X, info = citrus_recon(Y, I0, angles, grid, geom, solver, cfg, regularization=reg, init_recon=X_true,
                           model=model, callback=cb)
    torch.cuda.synchronize(DEV)
    print(f"total {time.perf_counter() - t0:.1f} s; relative change of X: "
          f"{float((X - X_true).norm() / X_true.norm()):.3e}")


if __name__ == "__main__":
    main()
