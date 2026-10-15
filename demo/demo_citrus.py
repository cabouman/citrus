"""Demo: CITRUS on the WLAB phantom (synthetic cone-beam data with scatter and Poisson noise).

    python demo/demo_citrus.py                     # 36 views, 1 cm grid, 10 outer iterations (about 2 minutes)
    python demo/demo_citrus.py --full              # 180 views, 0.5 cm grid, 20 outer iterations (about 30 min)
    python demo/demo_citrus.py --views 60 --delta-voxel 0.5 --outer 15
    python demo/demo_citrus.py --truth-sweep 16x32 --truth-cone 32x64   # truth finer than the loop's solver

1. Data (citrus.utilities.synthesize_wlab_dataset, cached in demo/input/). The WLAB phantom (30 x 20 x 40 cm water
   prism with lung, air and bone inserts) is rasterized on the reconstruction grid; full-field panel (150 x 184
   pixels of 0.4 cm, the whole prism in the field at every view).
   - Primary: P = I0 exp(-A X_true) with A the MBIRTORCH projector and X_true the attenuation of the true densities at
     the effective energy (spectrum mean, 63 keV); I0 = flat-field counts per pixel.
   - Ground-truth scatter: for every view, the LBTE scatter solver is run on the TRUE densities with a fully converged
     source iteration (``ScatterSolver.solve_view``, tolerance 1e-3), giving the solver's scatter and primary planar
     fluence images s, p (ideal photon-counting detector, no anti-scatter grid). The truth is stored as the per-pixel
     scatter-to-primary ratio SPR = s / p and applied to the MBIRTORCH primary counts: S = SPR x P.
   - Measured counts: Y = Poisson(P + S) (``--no-noise``: Y = P + S).
   This is an intentional inverse crime for the scatter: the truth comes from the same solver (same physics, same
   discretization by default) that CITRUS uses for its correction, so the study measures how well the coupled loop
   converges, not the accuracy of the scatter model. (The primary is an inverse crime too: MBIRTORCH's own projector,
   monoenergetic, on the reconstruction grid.) To expose the solver's angular discretization error, generate the
   truth with finer direction sets than the loop: ``--truth-sweep`` (discrete-ordinates quadrature, polar x
   azimuthal, loop default 8x16; recommended 16x32) and ``--truth-cone`` (collapsed-cone directions, loop default
   16x32; recommended 32x64, or 48x96 as a reference). The defaults equal the loop's for speed; 16x32 / 32x64 make the
   data synthesis about 4x slower.
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
from dataclasses import replace
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
from citrus.scatter_solver.config import SolverConfig  # noqa: E402
from citrus.scatter_solver.geometry import ConeGeometry  # noqa: E402
from citrus.utilities import (default_supervoxel_factor, full_field_wlab_geometry, load_dataset,  # noqa: E402
                              plot_history, plot_profiles, plot_slices, set_memory_cap, synthesize_wlab_dataset,
                              wlab_grid, wlab_solver)

HERE = os.path.dirname(os.path.abspath(__file__))


def _box(grid, device, half) -> torch.Tensor:
    x, y, z = grid.axis_coordinates(device)
    return (x[None, :, None].abs() < half[0]) & (y[:, None, None].abs() < half[1]) & (z[None, None, :].abs() < half[2])


def _counts(text: str, what: str) -> tuple[int, int]:
    try:
        p, a = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise SystemExit(f"{what} must be POLARxAZIMUTHAL, e.g. 16x32, got {text!r}")
    return p, a


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
    ap.add_argument("--truth-sweep", default=None,
                    help="sweep quadrature of the ground-truth solver, POLARxAZIMUTHAL (default: the loop's, 8x16; "
                         "recommended finer: 16x32; polar even, azimuthal divisible by 4)")
    ap.add_argument("--truth-cone", default=None,
                    help="collapsed-cone directions of the ground-truth solver, POLARxAZIMUTHAL (default: the loop's, "
                         "16x32; recommended finer: 32x64)")
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
    loop_cfg = replace(SolverConfig(), supervoxel_factor=default_supervoxel_factor(args.delta_voxel))
    tp, ta = _counts(args.truth_sweep, "--truth-sweep") if args.truth_sweep else (loop_cfg.num_polar,
                                                                                 loop_cfg.num_azimuthal)
    cp, ca = _counts(args.truth_cone, "--truth-cone") if args.truth_cone else (loop_cfg.cone_num_polar,
                                                                              loop_cfg.cone_num_azimuthal)
    if tp % 2 or ta % 4:
        raise SystemExit("--truth-sweep: the polar count must be even and the azimuthal count divisible by 4")
    truth_cfg = replace(loop_cfg, num_polar=tp, num_azimuthal=ta, cone_num_polar=cp, cone_num_azimuthal=ca)
    truth_cfg.validate()
    tag = f"wlab_{args.views}v_{args.delta_voxel:g}cm".replace(".", "p") + ("_nonoise" if args.no_noise else "")
    if truth_cfg != loop_cfg:
        tag += f"_truth{tp}x{ta}_cone{cp}x{ca}"
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
                                       geom=geom, device=dev, config=truth_cfg)
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
    solver = wlab_solver(args.delta_voxel, config=loop_cfg, geom=geom, device=dev)
    print(f"truth solver: sweep {tp}x{ta}, cone {cp}x{ca}; loop solver: sweep {loop_cfg.num_polar}x"
          f"{loop_cfg.num_azimuthal}, cone {loop_cfg.cone_num_polar}x{loop_cfg.cone_num_azimuthal}")
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
        "truth_sweep": [tp, ta], "truth_cone": [cp, ca],
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
