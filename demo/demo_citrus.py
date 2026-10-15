"""Demo: CITRUS on the WLAB phantom (synthetic cone-beam data with scatter and Poisson noise).

    python demo/demo_citrus.py                     # 36 views, 1 cm grid, 10 outer iterations (a few minutes)
    python demo/demo_citrus.py --full              # 180 views, 0.5 cm grid, 20 outer iterations (about 30 min)
    python demo/demo_citrus.py --views 60 --delta-voxel 0.5 --outer 15

1. Data (cached in demo/input/): the WLAB phantom at E_eff (spectrum mean, 63 keV) on the reconstruction grid; the
   primary P = I0 exp(-A X) from MBIRTORCH's projector, the scatter S = SPR P with the scatter-to-primary ratio of
   the LBTE scatter solver (one converged solve per view on the true densities), Y = Poisson(P + S); full-field
   panel (150 x 184 pixels of 0.4 cm).
2. CITRUS (citrus.citrus_recon): FDK initialization, then per outer iteration one scatter step per view and one
   warm-started MBIR iteration on Y / (1 + SPR).
3. The same MBIR loop without scatter correction (citrus.mbir_recon), for comparison.
Both reconstructions use the same frozen regularization (MBIRTORCH's automatic estimate on the scatter-free
sinogram), the same FDK initialization, weights and partition sequence, so they differ only by the scatter
correction. Figures (demo/output/): central slices, central-row profiles, relative RMSE vs outer iteration, and the
estimated vs true scatter-to-primary ratio of view 0.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from citrus import CitrusConfig, citrus_recon, mbir_recon  # noqa: E402
from citrus.citrus import estimate_regularization  # noqa: E402
from citrus.recon.mbir_bridge import make_cone_model  # noqa: E402
from citrus.recon.synthetic import load_dataset, synthesize_wlab_dataset, wlab_solver  # noqa: E402
from citrus.scatter_solver.geometry import ConeGeometry  # noqa: E402
from citrus.utilities import (full_field_wlab_geometry, plot_history, plot_profiles, plot_slices,  # noqa: E402
                              set_memory_cap, wlab_grid)

HERE = os.path.dirname(os.path.abspath(__file__))


def _box(grid, device, half) -> torch.Tensor:
    x, y, z = grid.axis_coordinates(device)
    return (x[None, :, None].abs() < half[0]) & (y[:, None, None].abs() < half[1]) & (z[None, None, :].abs() < half[2])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", type=int, default=36, help="number of views over 360 degrees (default 36)")
    ap.add_argument("--delta-voxel", type=float, default=1.0, help="voxel pitch in cm (default 1.0; 0.5 for full)")
    ap.add_argument("--outer", type=int, default=10, help="outer iterations (default 10)")
    ap.add_argument("--full", action="store_true", help="full size: 180 views, 0.5 cm grid, 20 outer iterations")
    ap.add_argument("--I0", type=float, default=1.0e5, help="flat-field counts per pixel (default 1e5)")
    ap.add_argument("--seed", type=int, default=0, help="Poisson noise seed")
    ap.add_argument("--no-noise", action="store_true", help="noise-free data (Y = P + S)")
    ap.add_argument("--regenerate", action="store_true", help="re-synthesize the data even if cached")
    ap.add_argument("--memory-cap-mib", type=float, default=4400.0, help="PyTorch CUDA memory cap (0: none)")
    ap.add_argument("--input", default=os.path.join(HERE, "input"))
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    args = ap.parse_args()
    if args.full:
        args.views, args.delta_voxel, args.outer = 180, 0.5, 20
    if not torch.cuda.is_available():
        raise SystemExit("this demo needs a CUDA GPU (MBIRTORCH and the scatter solver run on the GPU)")
    dev = torch.device("cuda")
    set_memory_cap(args.memory_cap_mib)
    os.makedirs(args.input, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    tag = f"wlab_{args.views}v_{args.delta_voxel:g}cm".replace(".", "p") + ("_nonoise" if args.no_noise else "")
    t_all = time.perf_counter()

    # ---- 1. data ------------------------------------------------------------------------------------------------
    path = os.path.join(args.input, tag)
    geom = full_field_wlab_geometry()
    if os.path.exists(path + ".npz") and not args.regenerate:
        data = load_dataset(path)
        print(f"data: loaded {path}.npz")
    else:
        t0 = time.perf_counter()
        data = synthesize_wlab_dataset(args.views, args.delta_voxel, args.I0, not args.no_noise, args.seed, path,
                                       geom=geom, device=dev)
        print(f"data: synthesized {args.views} views in {time.perf_counter() - t0:.1f} s -> {path}.npz")
    geom = ConeGeometry(**data["meta"]["geometry"])
    grid = wlab_grid(args.delta_voxel)
    I0 = float(data["I0"])
    Y, angles = data["Y"], data["angles"]
    X_true = torch.as_tensor(data["X_true"], device=dev)
    spr_true = data["SPR"]
    print(f"grid {grid.shape} at {grid.delta_voxel} cm, {len(angles)} views, panel {geom.num_det_rows} x "
          f"{geom.num_det_channels}; SPR median {np.median(spr_true):.2f}, max {spr_true.max():.1f}")

    # ---- shared: solver, MBIRTORCH model, frozen regularization (scatter-free sinogram), metrics ---------------
    solver = wlab_solver(args.delta_voxel, geom=geom, device=dev)
    model = make_cone_model(grid, geom, angles, device=dev)
    Yt = torch.as_tensor(Y, device=dev)
    reg = estimate_regularization(model, -torch.log(torch.as_tensor(data["P"], device=dev) / I0), Yt / I0)
    core = _box(grid, dev, (12.0, 9.0, 6.0))

    def tracker(store: list):
        def cb(k: int, X: torch.Tensor, rec: dict) -> None:
            e = X - X_true
            store.append({"k": k, "rmse_rel": float(e.norm() / X_true.norm()),
                          "rmse_rel_core": float(e[core].norm() / X_true[core].norm())})
        return cb

    cfg = CitrusConfig(num_outer=args.outer, verbose=True)

    # ---- 2. CITRUS ------------------------------------------------------------------------------------------
    hist_c: list = []
    t0 = time.perf_counter()
    X_c, info_c = citrus_recon(Y, I0, angles, grid, geom, solver, cfg, regularization=reg, model=model,
                               callback=tracker(hist_c))
    t_citrus = time.perf_counter() - t0

    # ---- 3. no scatter correction ------------------------------------------------------------------------------
    hist_n: list = []
    t0 = time.perf_counter()
    X_n, info_n = mbir_recon(Y, I0, angles, grid, geom, cfg, regularization=reg, model=model,
                             callback=tracker(hist_n))
    t_none = time.perf_counter() - t0

    # ---- summary ---------------------------------------------------------------------------------------------
    rec = info_c["records"][1:]
    summary = {
        "views": len(angles), "delta_voxel_cm": args.delta_voxel, "grid": list(grid.shape), "outer": args.outer,
        "regularization": reg,
        "citrus": {"rmse_rel": hist_c[-1]["rmse_rel"], "rmse_rel_core": hist_c[-1]["rmse_rel_core"],
                   "seconds": t_citrus, "scatter_s_per_outer": float(np.mean([r["t_scatter"] for r in rec])),
                   "mbir_s_per_outer": float(np.mean([r["t_mbir"] for r in rec])), "state": info_c["state"]},
        "no_correction": {"rmse_rel": hist_n[-1]["rmse_rel"], "rmse_rel_core": hist_n[-1]["rmse_rel_core"],
                          "seconds": t_none},
        "fdk_rmse_rel": hist_c[0]["rmse_rel"],
        "history": {"citrus": hist_c, "no_correction": hist_n},
    }
    with open(os.path.join(args.out, f"citrus_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nrelative RMSE (whole prism / core): FDK {hist_c[0]['rmse_rel']:.4f} / {hist_c[0]['rmse_rel_core']:.4f}, "
          f"no correction {hist_n[-1]['rmse_rel']:.4f} / {hist_n[-1]['rmse_rel_core']:.4f}, "
          f"CITRUS {hist_c[-1]['rmse_rel']:.4f} / {hist_c[-1]['rmse_rel_core']:.4f}")
    print(f"CITRUS {t_citrus:.1f} s ({summary['citrus']['scatter_s_per_outer']:.1f} s scatter + "
          f"{summary['citrus']['mbir_s_per_outer']:.1f} s MBIR per outer iteration), no correction {t_none:.1f} s")

    # ---- figures ---------------------------------------------------------------------------------------------
    Xt, X0 = data["X_true"], info_c["init_recon"]
    vols = [("truth", Xt), ("FDK (initialization)", X0), ("MBIR, no correction", X_n), ("CITRUS", X_c)]
    figs = {
        "slices": plot_slices(vols, grid, title=f"central slice z = 0 ({len(angles)} views, {args.outer} outer "
                                                 f"iterations)"),
        "profiles": plot_profiles(vols, grid, title="central row y = 0 (lung -8..-2, air 2..4, bone 6..8 cm)"),
        "rmse": plot_history([("CITRUS", [h["rmse_rel"] for h in hist_c]),
                              ("MBIR, no correction", [h["rmse_rel"] for h in hist_n])],
                             title="relative RMSE over the prism"),
    }
    # estimated vs true scatter-to-primary ratio of view 0 (central detector row)
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    u = geom.pixel_uv(dtype=torch.float64)[0].numpy()
    r = geom.num_det_rows // 2
    ax.plot(u, spr_true[0, r], color="k", lw=1.2, label="true SPR")
    ax.plot(u, info_c["spr"][0, r].cpu().numpy(), ls="--", label=f"CITRUS estimate (outer iteration {args.outer})")
    ax.set_xlabel("u (cm), central detector row, view 0")
    ax.set_ylabel("scatter-to-primary ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    figs["spr"] = fig
    for name, fig in figs.items():
        p = os.path.join(args.out, f"citrus_{tag}_{name}.png")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"figure: {p}")
    print(f"total {time.perf_counter() - t_all:.1f} s")


if __name__ == "__main__":
    main()
