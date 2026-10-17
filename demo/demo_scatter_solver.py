"""Demo: the LBTE source iteration scatter solver on one view of the WLAB phantom.

    python demo/demo_scatter_solver.py [--delta-voxel 1.0] [--transport-cell-cm 1.0] [--angle-deg 0]
                                       [--panel small|full] [--grid] [--device cuda] [--memory-cap-mib 4400]
                                       [--materials air,lung,adipose,water,muscle,soft_tissue,bone]

Builds the multigroup library of the anchor materials (``--materials``: comma-separated names of
citrus.scatter_solver.cross_sections.MATERIAL_CATALOGUE, at least two, water and bone among them; default the seven
air, lung, adipose, water, muscle, soft_tissue, bone) from XrayPhysics (cached in $CITRUS_CACHE_DIR or
~/.cache/citrus; about 16 s on the first run for seven materials), rasterizes the WLAB phantom (30 x 20 x 40 cm water
prism with lung, air and bone inserts, in the library's material basis: its water and bone rows; the result does not
depend on the other anchors), solves one view with the default solver
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
from citrus.scatter_solver.cross_sections import MATERIAL_CATALOGUE, check_material_names  # noqa: E402
# noqa: E402
from citrus.utilities import full_field_wlab_geometry, set_memory_cap, wlab_geometry, wlab_grid, wlab_phantom

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delta-voxel", type=float, default=1.0, help="fine grid pitch in cm (default 1.0)")
    parser.add_argument("--transport-cell-cm", type=float, default=1.0,
                        help="transport cell edge length in cm (default 1.0; any ratio to --delta-voxel)")
    parser.add_argument("--angle-deg", type=float, default=0.0, help="view angle in degrees (default 0: AP view)")
    parser.add_argument("--panel", choices=["small", "full"], default="small",
                        help="small: 100 x 75 pixels (40 x 30 cm); full: 150 x 184 pixels (the whole prism)")
    parser.add_argument("--grid", action="store_true", help="add a focused 10:1 anti-scatter grid")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--memory-cap-mib", type=float, default=4400.0, help="PyTorch CUDA memory cap (0: none)")
    parser.add_argument("--materials", default=",".join(SolverConfig().materials),
                        help="anchor materials of the library, comma-separated names from the catalogue "
                             f"({', '.join(MATERIAL_CATALOGUE)}); at least two, water and bone among them (default: "
                             f"{','.join(SolverConfig().materials)})")
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()
    try:
        materials = check_material_names(tuple(name.strip() for name in args.materials.split(",") if name.strip()))
    except ValueError as exc:
        raise SystemExit(f"--materials: {exc}")
    if "water" not in materials or "bone" not in materials:
        raise SystemExit(f"--materials: the WLAB phantom needs water and bone among the materials, got {materials}")

    device = torch.device(args.device)
    if device.type == "cuda":
        set_memory_cap(args.memory_cap_mib)
    os.makedirs(args.out, exist_ok=True)
    config = replace(SolverConfig(), transport_cell_cm=args.transport_cell_cm, materials=materials,
                     dtype="float32" if device.type == "cuda" else "float64")
    t0 = time.perf_counter()
    library = build_solver_library(config)
    print(f"library: {library.num_materials} materials {', '.join(library.materials)}; {library.num_groups} groups, "
          f"Legendre order {library.legendre_order}, "
          f"edges {np.round(library.group_edges.numpy(), 1).tolist()} keV ({time.perf_counter() - t0:.1f} s)")

    grid = wlab_grid(args.delta_voxel)
    geometry = wlab_geometry() if args.panel == "small" else full_field_wlab_geometry()
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    rho = wlab_phantom(grid, materials=library.materials, device=device, dtype=dtype)
    solver = ScatterSolver(library, config, geometry, detector=csi_response(0.06),
                           grid_response=focused_1d_grid() if args.grid else no_grid(), device=device, dtype=dtype)
    angle = math.radians(args.angle_deg)
    solver.solve_view(rho, grid, angle)  # warm-up: compiles the Triton kernels and fills the caches
    t0 = time.perf_counter()
    result = solver.solve_view(rho, grid, angle)
    seconds = time.perf_counter() - t0
    print(f"grid {grid.shape} at {grid.delta_voxel} cm, transport grid {solver.transport_grid(grid).shape} "
          f"({config.transport_cell_cm:g} cm cells, {solver.quadrature_points(grid)} first-collision points per "
          f"axis at {config.quadrature_point_spacing_cm:g} cm target spacing), "
          f"panel {geometry.num_det_rows} x {geometry.num_det_channels}, device {device}")
    print(f"solve_view: {seconds:.2f} s, {result.source_iteration_result.num_iterations} source iterations "
          f"(converged {result.source_iteration_result.converged}, last change "
          f"{result.source_iteration_result.history[-1]:.1e})")
    print("stage seconds: " + ", ".join(f"{k} {v:.3f}" for k, v in result.timings.items()))

    P = result.primary["signal"].float().cpu().numpy()
    S = result.scatter_signal.float().cpu().numpy()
    SF = result.scatter_fraction.float().cpu().numpy()
    print(f"scatter fraction: center {SF[SF.shape[0] // 2, SF.shape[1] // 2]:.3f}, "
          f"median {np.median(SF):.3f}, max {SF.max():.3f}")

    u = geometry.pixel_uv(dtype=torch.float64)[0].numpy()
    v = geometry.pixel_uv(dtype=torch.float64)[1].numpy()
    extent = [u[0], u[-1], v[0], v[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    for ax, image, title, cmap in ((axes[0, 0], P, "primary signal (keV/cm$^2$ per source photon/sr)", "gray"),
                                   (axes[0, 1], S, "scatter signal (keV/cm$^2$ per source photon/sr)", "magma"),
                                   (axes[1, 0], SF, "scatter fraction S / (P + S)", "viridis")):
        image_artist = ax.imshow(image, origin="lower", extent=extent, cmap=cmap, aspect="equal")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("u (cm)", fontsize=8)
        ax.set_ylabel("v (cm)", fontsize=8)
        fig.colorbar(image_artist, ax=ax, shrink=0.85)
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
    fig.suptitle(f"LBTE scatter solver, WLAB phantom, view {args.angle_deg:g} deg, "
                 f"{result.source_iteration_result.num_iterations} source "
                 f"iterations, {seconds:.2f} s", fontsize=10)
    fig.tight_layout()
    path = os.path.join(args.out, "scatter_solver_view.png")
    fig.savefig(path, dpi=120)
    print(f"figure: {path}")


if __name__ == "__main__":
    main()
