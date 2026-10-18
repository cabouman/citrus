"""Fixed versus adaptive numbers of MBIR iterations per outer iteration, on the data set of demo 2.

Every setting runs CITRUS from the same FDK volume with the same frozen regularization and records, after every MBIR
iteration, the wall time, the relative RMSE of the volume against the truth and the relative RMSE of A X against the
corrected sinogram. The figure plots the error against the truth over wall time; the summary JSON holds the final
numbers of every setting.

Usage::

    python benchmarks/adaptive_mbir_study.py                       # fixed 1, 2, 4, 8 and adaptive 0.01, 0.02, 0.05
    python benchmarks/adaptive_mbir_study.py --outer-iterations 5  # a quicker run
    python benchmarks/adaptive_mbir_study.py --fixed 8 --adaptive 0.02,0.05 --adaptive-outer-iterations 20
                                                                   # equal wall time: the adaptive runs get more
                                                                   # outer iterations, the summary reads every
                                                                   # setting's error at the fixed run's end time
"""

import argparse
import json
import os
import sys
import time

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from citrus import CitrusConfig, citrus_recon  # noqa: E402
from citrus.scatter_solver import (ConeGeometry, ScatterSolver, SolverConfig, ideal_counting_response,  # noqa: E402
                                   load_spectrum_txt, no_grid)
from citrus.utilities import load_dataset, wlab_grid  # noqa: E402

DEMO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join(DEMO, "input", "wlab_180v_0p5cm_0-202p6deg_125kvp"),
                        help="data set of demo 2 (base path of the .npz/.json pair)")
    parser.add_argument("--outer-iterations", type=int, default=10)
    parser.add_argument("--fixed", default="1,2,4,8", help="fixed MBIR iterations per outer iteration")
    parser.add_argument("--adaptive", default="0.01,0.02,0.05",
                        help="thresholds of mbir_stop_relative_improvement (with at most --max-mbir-iterations)")
    parser.add_argument("--max-mbir-iterations", type=int, default=8)
    parser.add_argument("--adaptive-outer-iterations", type=int, default=None,
                        help="outer iterations of the adaptive runs (default: --outer-iterations)")
    parser.add_argument("--scatter-view-stride", type=int, default=8)
    parser.add_argument("--out", default=os.path.join(DEMO, "output", "adaptive_mbir_study"))
    args = parser.parse_args()

    data = load_dataset(args.data)
    Y, X_true, angles = data["Y"], data["X_true"], data["angles"]
    grid = wlab_grid(20.0 / X_true.shape[0])  # the WLAB box is 20 cm along the rows
    geometry = ConeGeometry(source_iso_dist=100.0, source_detector_dist=150.0, num_det_rows=Y.shape[1],
                            num_det_channels=Y.shape[2], delta_det_row=0.4, delta_det_channel=0.4)
    spectrum = load_spectrum_txt(os.path.join(DEMO, "spectra", "spectrum_125kvp.txt"))
    config = SolverConfig(materials=("air", "lung_icrp", "adipose_icrp", "water", "muscle_icru44",
                                     "soft_tissue_icru44", "bone_cortical_icrp"))
    solver = ScatterSolver(None, config, geometry, detector=ideal_counting_response(), grid_response=no_grid(),
                           device="cuda", spectrum=spectrum)
    truth = torch.as_tensor(X_true, device="cuda")

    adaptive_outer = args.adaptive_outer_iterations or args.outer_iterations
    settings = [(f"fixed {n}", dict(mbir_iterations_per_outer_iteration=n, num_outer_iterations=args.outer_iterations))
                for n in map(int, args.fixed.split(",")) if n > 0]
    settings += [(f"adaptive {t:g} (max {args.max_mbir_iterations})",
                  dict(mbir_iterations_per_outer_iteration="adaptive", mbir_stop_relative_improvement=t,
                       max_mbir_iterations_per_outer_iteration=args.max_mbir_iterations,
                       num_outer_iterations=adaptive_outer))
                 for t in map(float, args.adaptive.split(",")) if t > 0]
    shared: dict = {}  # the FDK volume, the model and the regularization of the first run, reused by the others
    results = {}
    for label, loop in settings:
        history: list = []
        clock: list = []  # the wall clock starts at the FDK callback, after the shared initialization

        def track(i, X, record, history=history, clock=clock):
            if i == 0:
                clock.append(time.perf_counter())
            history.append({"mbir_iteration": i, "seconds": time.perf_counter() - clock[0],
                            "volume_nrmse": float((X - truth).norm() / truth.norm()),
                            "sinogram_nrmse": record["sinogram_nrmse"]})

        citrus_config = CitrusConfig(scatter_view_stride=args.scatter_view_stride, verbose=True, **loop)
        X, info = citrus_recon(Y, 1.0e5, angles, grid, geometry, solver, citrus_config, mbir_callback=track,
                               **shared)
        if not shared:
            shared = dict(regularization=info["regularization"], init_recon=info["init_recon"], model=info["model"])
        results[label] = {"history": history, "seconds": info["seconds"],
                          "mbir_iterations_total": info["mbir_iterations_total"],
                          "mbir_iterations": [r["mbir_iterations"] for r in info["records"][1:]],
                          "volume_nrmse": history[-1]["volume_nrmse"], "sinogram_nrmse": history[-1]["sinogram_nrmse"],
                          "scatter_seconds": float(sum(r["scatter_seconds"] for r in info["records"][1:])),
                          "mbir_seconds": float(sum(r["mbir_seconds"] for r in info["records"][1:]))}
        print(f"{label}: {info['seconds']:.0f} s, {info['mbir_iterations_total']} MBIR iterations "
              f"{results[label]['mbir_iterations']}, volume NRMSE {results[label]['volume_nrmse']:.4f}, "
              f"sinogram NRMSE {results[label]['sinogram_nrmse']:.4f}", flush=True)

    # every setting's error at the wall time where the longest fixed run ended (its last value if it ended earlier)
    fixed_runs = [r for label, r in results.items() if label.startswith("fixed")]
    if fixed_runs:
        budget = max(r["seconds"] for r in fixed_runs)
        for label, r in results.items():
            within = [h for h in r["history"] if h["seconds"] <= budget]
            r["volume_nrmse_at_budget"] = within[-1]["volume_nrmse"]
            r["seconds_budget"] = budget
            print(f"{label}: volume NRMSE {r['volume_nrmse_at_budget']:.4f} at {budget:.0f} s "
                  f"({len(within) - 1} MBIR iterations by then)")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump({"outer_iterations": args.outer_iterations, "scatter_view_stride": args.scatter_view_stride,
                   "results": results}, f, indent=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for label, r in results.items():
        t = [h["seconds"] for h in r["history"]]
        axes[0].semilogy(t, [h["volume_nrmse"] for h in r["history"]], ".-", ms=3, label=label)
        axes[1].semilogy(t, [h["sinogram_nrmse"] for h in r["history"]], ".-", ms=3, label=label)
    for ax, title in zip(axes, ("relative RMSE of the volume against the truth",
                                "relative RMSE of A X against the corrected sinogram")):
        ax.set_xlabel("wall time (s)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=130)
    print(f"wrote {args.out}.json and .png")


if __name__ == "__main__":
    main()
