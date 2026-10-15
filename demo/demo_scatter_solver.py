"""Demo: the LBTE source iteration scatter solver on one view of the WLAB phantom.

    python demo/demo_scatter_solver.py [--delta-voxel 1.0] [--supervoxel-factor 1] [--angle-deg 0]
                                       [--panel small|full] [--grid] [--device cuda] [--memory-cap-mib 4400]

Builds the multigroup library from XrayPhysics (cached in $CITRUS_CACHE_DIR or ~/.cache/citrus), rasterizes the WLAB
phantom (30 x 20 x 40 cm water prism with lung, air and bone inserts), solves one view with the default solver
configuration (6 groups, P7, 8 x 16 sweep directions, 16 x 32 collapsed-cone directions), and plots the primary and
scatter detector signals, the scatter fraction and central-row profiles to demo/output/scatter_solver_view.png.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from citrus.scatter_solver import (ScatterSolver, SolverConfig, build_solver_library, csi_response,  # noqa: E402
                                   focused_1d_grid, no_grid)
from citrus.utilities import full_field_wlab_geometry, set_memory_cap, wlab_geometry, wlab_grid, wlab_phantom  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delta-voxel", type=float, default=1.0, help="fine grid pitch in cm (default 1.0)")
    ap.add_argument("--supervoxel-factor", type=int, default=1,
                    help="fine voxels per transport cell along each axis (default 1)")
    ap.add_argument("--angle-deg", type=float, default=0.0, help="view angle in degrees (default 0: AP view)")
    ap.add_argument("--panel", choices=["small", "full"], default="small",
                    help="small: 100 x 75 pixels (40 x 30 cm); full: 150 x 184 pixels (the whole prism)")
    ap.add_argument("--grid", action="store_true", help="add a focused 10:1 anti-scatter grid")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--memory-cap-mib", type=float, default=4400.0, help="PyTorch CUDA memory cap (0: none)")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type == "cuda":
        set_memory_cap(args.memory_cap_mib)
    os.makedirs(args.out, exist_ok=True)
    config = replace(SolverConfig(), supervoxel_factor=args.supervoxel_factor,
                     dtype="float32" if dev.type == "cuda" else "float64")
    t0 = time.perf_counter()
    lib = build_solver_library(config)
    print(f"library: {lib.num_groups} groups, Legendre order {lib.legendre_order}, "
          f"edges {np.round(lib.group_edges.numpy(), 1).tolist()} keV ({time.perf_counter() - t0:.1f} s)")

    grid = wlab_grid(args.delta_voxel)
    geom = wlab_geometry() if args.panel == "small" else full_field_wlab_geometry()
    dtype = torch.float32 if dev.type == "cuda" else torch.float64
    rho = wlab_phantom(grid, device=dev, dtype=dtype)
    solver = ScatterSolver(lib, config, geom, detector=csi_response(0.06),
                           grid_response=focused_1d_grid() if args.grid else no_grid(), device=dev, dtype=dtype)
    angle = math.radians(args.angle_deg)
    solver.solve_view(rho, grid, angle)  # warm-up: compiles the Triton kernels and fills the caches
    t0 = time.perf_counter()
    res = solver.solve_view(rho, grid, angle)
    seconds = time.perf_counter() - t0
    print(f"grid {grid.shape} at {grid.delta_voxel} cm, transport grid {grid.coarsen(args.supervoxel_factor).shape}, "
          f"panel {geom.num_det_rows} x {geom.num_det_channels}, device {dev}")
    print(f"solve_view: {seconds:.2f} s, {res.si.num_iterations} source iterations "
          f"(converged {res.si.converged}, last change {res.si.history[-1]:.1e})")
    print("stage seconds: " + ", ".join(f"{k} {v:.3f}" for k, v in res.timings.items()))

    P = res.primary["signal"].float().cpu().numpy()
    S = res.scatter_signal.float().cpu().numpy()
    SF = res.scatter_fraction.float().cpu().numpy()
    print(f"scatter fraction: center {SF[SF.shape[0] // 2, SF.shape[1] // 2]:.3f}, "
          f"median {np.median(SF):.3f}, max {SF.max():.3f}")

    u = geom.pixel_uv(dtype=torch.float64)[0].numpy()
    v = geom.pixel_uv(dtype=torch.float64)[1].numpy()
    ext = [u[0], u[-1], v[0], v[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    for ax, img, title, cmap in ((axes[0, 0], P, "primary signal (keV/cm$^2$ per source photon/sr)", "gray"),
                                 (axes[0, 1], S, "scatter signal (keV/cm$^2$ per source photon/sr)", "magma"),
                                 (axes[1, 0], SF, "scatter fraction S / (P + S)", "viridis")):
        im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap, aspect="equal")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("u (cm)", fontsize=8)
        ax.set_ylabel("v (cm)", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.85)
    r = P.shape[0] // 2
    ax = axes[1, 1]
    ax.semilogy(u, P[r], label="primary")
    ax.semilogy(u, S[r], ls="--", label="scatter")
    ax2 = ax.twinx()
    ax2.plot(u, SF[r], color="tab:green", ls="-.", label="scatter fraction")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("scatter fraction", fontsize=8)
    ax.set_xlabel("u (cm), central row", fontsize=8)
    ax.set_title("central-row profiles", fontsize=9)
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=8, loc="lower center")
    fig.suptitle(f"LBTE scatter solver, WLAB phantom, view {args.angle_deg:g} deg, {res.si.num_iterations} source "
                 f"iterations, {seconds:.2f} s", fontsize=10)
    fig.tight_layout()
    path = os.path.join(args.out, "scatter_solver_view.png")
    fig.savefig(path, dpi=120)
    print(f"figure: {path}")


if __name__ == "__main__":
    main()
