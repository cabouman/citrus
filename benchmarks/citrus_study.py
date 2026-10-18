"""CITRUS study on the WLAB phantom: synthetic cone-beam data with scatter and Poisson noise, CITRUS and the
uncorrected MBIR loop, figures and a summary JSON.

The data sets are cached in demo/input and the figures, the summary JSON and (with ``--save-recon``) the CITRUS volume
are written to demo/output, under names that record the settings. The flags and the output are described in
docs/source/usage.rst, "CITRUS studies".

Usage::

    python benchmarks/citrus_study.py                  # 180 views over a short scan, 0.5 cm grid, 10 outer iterations
                                                       # of 8 MBIR iterations, scatter step on every 8th view
    python benchmarks/citrus_study.py --full           # 180 views over a full rotation, 0.5 cm grid, 20 outer
                                                       # iterations of 1 MBIR iteration, scatter step on every view
    python benchmarks/citrus_study.py --end-view-angle 360                     # the default over a full rotation
    python benchmarks/citrus_study.py --views 36 --delta-voxel 1 --outer-iterations 10 --scatter-view-stride 1 \
        --mbir-iterations 1                            # a quick run
    python benchmarks/citrus_study.py --transport-cell-cm 0.5                  # 0.5 cm transport cells
    python benchmarks/citrus_study.py --materials air,water,bone_cortical_icrp             # three reference materials
    python benchmarks/citrus_study.py --truth-sweep 16x32 --truth-cone 32x64   # truth finer than the loop's solver
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from citrus import CitrusConfig, citrus_recon, mbir_recon  # noqa: E402
from citrus.citrus import estimate_regularization  # noqa: E402
from citrus.recon.mbir_bridge import is_short_scan, make_cone_model, scan_range  # noqa: E402
from citrus.scatter_solver.config import SolverConfig  # noqa: E402
from citrus.scatter_solver.cross_section_library import load_spectrum_txt  # noqa: E402
from citrus.scatter_solver.cross_sections import COMPOUND_NAMES, check_material_names  # noqa: E402
from citrus.scatter_solver.geometry import ConeGeometry, Grid  # noqa: E402
from citrus.utilities import (full_field_wlab_geometry, load_dataset,  # noqa: E402
                              nominal_attenuation, plot_history, plot_profiles, plot_slices, set_memory_cap,
                              synthesize_wlab_dataset, wlab_grid, wlab_solver)

SPECTRUM = load_spectrum_txt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo", "spectra",
                                          "spectrum_125kvp.txt"))  # the sample 125 kVp spectrum

# the data sets and the outputs live next to the demos: demo/input and demo/output
DEMO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo")


def _box(grid, device, half) -> torch.Tensor:
    """Mask [rows, cols, slices] of the voxels with |x|, |y|, |z| below the half-widths ``half`` (cm)."""
    x, y, z = grid.axis_coordinates(device)
    return (x[None, :, None].abs() < half[0]) & (y[:, None, None].abs() < half[1]) & (z[None, None, :].abs() < half[2])


def _counts(text: str, what: str) -> tuple[int, int]:
    """Parse a POLARxAZIMUTHAL direction count such as 16x32 given to the flag ``what``."""
    try:
        p, a = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise SystemExit(f"{what} must be POLARxAZIMUTHAL, e.g. 16x32, got {text!r}")
    return p, a


def main() -> None:
    """Parse the flags, load or synthesize the data, run CITRUS and the uncorrected loop, write the outputs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--views", type=int, default=180, help="number of views over the angular range (default 180)")
    parser.add_argument("--delta-voxel", type=float, default=0.5, help="voxel pitch in cm (default 0.5)")
    parser.add_argument("--transport-cell-cm", type=float, default=1.0,
                        help="transport cell edge length in cm, for the loop's and the truth's solver (default 1; any "
                             "ratio to --delta-voxel)")
    parser.add_argument("--outer-iterations", type=int, default=10,
                        help="outer iterations, one scatter correction each (default 10)")
    parser.add_argument("--mbir-iterations", type=int, default=8,
                        help="MBIR iterations per outer iteration (default 8)")
    parser.add_argument("--full", action="store_true",
                        help="the report study: 180 views over a FULL rotation (end view angle = start + 360 degrees, "
                             "set explicitly; --end-view-angle is ignored), 0.5 cm grid, 20 outer iterations of one "
                             "MBIR iteration, scatter step on every view (--views, --delta-voxel, --outer-iterations, "
                             "--scatter-view-stride and --mbir-iterations are overridden)")
    parser.add_argument("--start-view-angle", type=float, default=0.0,
                        help="first view angle in degrees (default 0; MBIRTORCH's angle convention)")
    parser.add_argument("--end-view-angle", type=float, default=None,
                        help="end of the angular range in degrees, excluded (views evenly spaced over [start, end)); "
                             "default: the short scan start + 180 + fan angle of the panel (22.6 degrees); start + 360 "
                             "for a full rotation")
    parser.add_argument("--I0", type=float, default=1.0e5, help="flat-field counts per pixel (default 1e5)")
    parser.add_argument("--seed", type=int, default=0, help="Poisson noise seed")
    parser.add_argument("--no-noise", action="store_true", help="noise-free data (Y = P + S)")
    parser.add_argument("--regenerate", action="store_true", help="re-synthesize the data even if cached")
    parser.add_argument("--truth-sweep", default=None,
                        help="sweep quadrature of the ground-truth solver, POLARxAZIMUTHAL (default: the loop's, 8x16; "
                             "recommended finer: 16x32; polar even, azimuthal divisible by 4)")
    parser.add_argument("--truth-cone", default=None,
                        help="collapsed-cone directions of the ground-truth solver, POLARxAZIMUTHAL (default: the "
                             "loop's, 16x32; recommended finer: 32x64)")
    parser.add_argument("--scatter-view-stride", type=int, default=8,
                        help="scatter step on every n-th view only (views 0, n, 2n, ...), the others interpolated in "
                             "angle (default 8)")
    parser.add_argument("--scatter-view-interpolation-order", default="3", choices=["none", "0", "1", "2", "3"],
                        help="angular interpolation of the skipped views (with --scatter-view-stride > 1): 0 nearest, "
                             "1 linear, 2 quadratic, 3 cubic Catmull-Rom (default), none = not corrected (diagnostics)")
    parser.add_argument("--materials", default=",".join(SolverConfig().materials),
                        help="reference materials of the loop's density model, comma-separated names from the "
                             "catalogue "
                             f"({', '.join(COMPOUND_NAMES)} and every element by its English name, e.g. lead); at "
                             "least two (default: "
                             f"{','.join(SolverConfig().materials)})")
    parser.add_argument("--resume-error-sinogram", default=True, action=argparse.BooleanOptionalAction,
                        help="resume MBIR from its error sinogram between outer iterations (default) or restart it "
                             "from the sinogram (--no-resume-error-sinogram)")
    parser.add_argument("--save-recon", action="store_true", help="also save the CITRUS volume (.npy)")
    parser.add_argument("--memory-cap-mib", type=float, default=None,
                        help="PyTorch CUDA allocator cap in MiB (default: 85 percent of the device's memory, which "
                             "keeps Windows from spilling into system memory; 0: none)")
    parser.add_argument("--input", default=os.path.join(DEMO, "input"))
    parser.add_argument("--out", default=os.path.join(DEMO, "output"))
    args = parser.parse_args()
    if args.full:
        args.views, args.delta_voxel, args.outer_iterations = 180, 0.5, 20
        args.scatter_view_stride, args.mbir_iterations = 1, 1
        args.end_view_angle = args.start_view_angle + 360.0  # the report's full rotation
    if not args.transport_cell_cm > 0:
        raise SystemExit("--transport-cell-cm must be > 0")
    interpolation_order = (None if args.scatter_view_interpolation_order == "none"
                           else int(args.scatter_view_interpolation_order))
    if args.scatter_view_stride < 1:
        raise SystemExit("--scatter-view-stride must be >= 1")
    if args.mbir_iterations < 1:
        raise SystemExit("--mbir-iterations must be >= 1")
    try:
        materials = check_material_names(tuple(name.strip() for name in args.materials.split(",") if name.strip()))
    except ValueError as exc:
        raise SystemExit(f"--materials: {exc}")
    if not torch.cuda.is_available():
        raise SystemExit("this study needs a CUDA GPU (MBIRTORCH and the scatter solver run on the GPU)")
    device = torch.device("cuda")
    if args.memory_cap_mib is None and torch.cuda.is_available():
        args.memory_cap_mib = 0.85 * torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    set_memory_cap(args.memory_cap_mib or 0.0)
    os.makedirs(args.input, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    # the truth solver keeps the default reference materials (the data do not depend on --materials); the loop uses
    # --materials
    base_config = replace(SolverConfig(), transport_cell_cm=args.transport_cell_cm)
    loop_config = replace(base_config, materials=materials)
    loop_config.validate()
    truth_polar, truth_azimuthal = (_counts(args.truth_sweep, "--truth-sweep") if args.truth_sweep
                                    else (loop_config.num_polar, loop_config.num_azimuthal))
    cone_polar, cone_azimuthal = (_counts(args.truth_cone, "--truth-cone") if args.truth_cone
                                  else (loop_config.cone_num_polar, loop_config.cone_num_azimuthal))
    if truth_polar % 2 or truth_azimuthal % 4:
        raise SystemExit("--truth-sweep: the polar count must be even and the azimuthal count divisible by 4")
    truth_config = replace(base_config, num_polar=truth_polar, num_azimuthal=truth_azimuthal, cone_num_polar=cone_polar,
                           cone_num_azimuthal=cone_azimuthal)
    truth_config.validate()
    start = math.radians(args.start_view_angle)
    end = None if args.end_view_angle is None else math.radians(args.end_view_angle)
    if end is not None and not end > start:
        raise SystemExit("--end-view-angle must be larger than --start-view-angle")
    full_rotation = end is not None and abs(end - start - 2 * math.pi) < 1e-9
    if full_rotation and args.start_view_angle == 0.0:
        arc = ""  # the full rotation from 0 keeps the tag of earlier versions
    elif end is None:
        arc = "_short" if args.start_view_angle == 0.0 else f"_short{args.start_view_angle:g}deg"
    else:
        arc = f"_{args.start_view_angle:g}to{args.end_view_angle:g}deg"
    tag = (f"wlab_{args.views}v_{args.delta_voxel:g}cm".replace(".", "p") + arc.replace(".", "p")
           + ("_nonoise" if args.no_noise else ""))
    if args.transport_cell_cm != 1.0:
        tag += f"_cell{args.transport_cell_cm:g}".replace(".", "p")
    if truth_config != base_config:
        tag += f"_truth{truth_polar}x{truth_azimuthal}_cone{cone_polar}x{cone_azimuthal}"
    tag += "_v2"  # data-set format 2: SPR with the primary model of the loop (citrus >= 0.3.0)
    start_time = time.perf_counter()

    # ---- 1. data ------------------------------------------------------------------------------------------------
    path = os.path.join(args.input, tag)
    geometry = full_field_wlab_geometry()
    if os.path.exists(path + ".npz") and not args.regenerate:
        data = load_dataset(path)
        print(f"data: loaded {path}.npz")
    else:
        t0 = time.perf_counter()
        data = synthesize_wlab_dataset(args.views, args.delta_voxel, args.I0, not args.no_noise, args.seed, path,
                                       geometry=geometry, device=device, config=truth_config, start_view_angle=start,
                                       end_view_angle=end, spectrum=SPECTRUM)
        print(f"data: synthesized {args.views} views in {time.perf_counter() - t0:.1f} s -> {path}.npz")
    geometry = ConeGeometry(**data["meta"]["geometry"])
    grid = wlab_grid(args.delta_voxel)
    I0 = float(data["I0"])
    Y, angles = data["Y"], data["angles"]
    X_true = torch.as_tensor(data["X_true"], device=device)
    spr_true = data["SPR"]
    span = math.degrees(scan_range(angles))
    print(f"views over {math.degrees(float(angles[0])):.1f} + {span:.1f} degrees "
          f"({'short scan' if is_short_scan(angles) else 'full rotation'})")
    print(f"grid {grid.shape} at {grid.delta_voxel} cm, transport grid "
          f"{Grid.covering(grid, args.transport_cell_cm).shape} "
          f"({args.transport_cell_cm:g} cm cells, {loop_config.quadrature_points(max(grid.deltas))} first-collision "
          f"points per axis at {loop_config.quadrature_point_spacing_cm:g} cm target spacing), "
          f"{len(angles)} views, panel {geometry.num_det_rows} x {geometry.num_det_channels}; SPR median "
          f"{np.median(spr_true):.2f}, max {spr_true.max():.1f}")

    # ---- shared: solver, MBIRTORCH model, frozen regularization (scatter-free sinogram), metrics ---------------
    solver = wlab_solver(args.delta_voxel, config=loop_config, geometry=geometry, device=device, spectrum=SPECTRUM)
    print(f"truth solver: sweep {truth_polar}x{truth_azimuthal}, cone {cone_polar}x{cone_azimuthal}; "
          f"loop solver: sweep {loop_config.num_polar}x{loop_config.num_azimuthal}, "
          f"cone {loop_config.cone_num_polar}x{loop_config.cone_num_azimuthal}")
    # sorted by nominal attenuation; the library itself is built by citrus_recon after it has measured the object
    # (the object size of the energy groups, CitrusConfig.grouping_object_size "auto")
    reference_materials = solver.materials
    model = make_cone_model(grid, geometry, angles, device=device)
    Y_device = torch.as_tensor(Y, device=device)
    regularization = estimate_regularization(model, -torch.log(torch.as_tensor(data["P"], device=device) / I0),
                                             Y_device / I0)
    core = _box(grid, device, (12.0, 9.0, 6.0))

    def tracker(store: list):
        def track(k: int, X: torch.Tensor, record: dict) -> None:
            e = X - X_true
            store.append({"k": k, "rmse_rel": float(e.norm() / X_true.norm()),
                          "rmse_rel_core": float(e[core].norm() / X_true[core].norm())})
        return track

    config = CitrusConfig(num_outer_iterations=args.outer_iterations, verbose=True,
                          scatter_view_stride=args.scatter_view_stride,
                          scatter_view_interpolation_order=interpolation_order,
                          mbir_iterations_per_outer_iteration=args.mbir_iterations,
                          resume_error_sinogram=args.resume_error_sinogram)
    out_tag = tag
    if set(materials) != set(SolverConfig().materials):
        out_tag += "_materials-" + "-".join(reference_materials)
    if not args.resume_error_sinogram:
        out_tag += "_restart"
    if args.scatter_view_stride > 1:
        out_tag += f"_stride{args.scatter_view_stride}" + ("" if interpolation_order == 3
                                                           else f"_order{interpolation_order}")
        print(f"sparse-view scatter: scatter step on {len(range(0, len(angles), args.scatter_view_stride))} of "
              f"{len(angles)} views (stride {args.scatter_view_stride}), interpolation order {interpolation_order}")
    if args.mbir_iterations > 1:
        out_tag += f"_mbir{args.mbir_iterations}"
        print(f"{args.mbir_iterations} MBIR iterations per outer iteration ({args.outer_iterations} scatter steps, "
              f"{args.outer_iterations * args.mbir_iterations} MBIR iterations)")

    # ---- 2. CITRUS ------------------------------------------------------------------------------------------
    citrus_history: list = []
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)  # the CITRUS run's own peak (records reset it every outer iteration)
    t0 = time.perf_counter()
    X_citrus, citrus_info = citrus_recon(Y, I0, angles, grid, geometry, solver, config, regularization=regularization,
                                         model=model,
                                         callback=tracker(citrus_history))
    t_citrus = time.perf_counter() - t0
    print("reference materials of the density model (nominal attenuation at E_eff, 1/cm): "
          + ", ".join(f"{name} {mu:.4g}" for name, mu
                      in zip(reference_materials, nominal_attenuation(solver.library).tolist())))

    def medium_text(medium) -> str:
        return medium if isinstance(medium, str) or medium is None else ", ".join(f"{n} {d:.3g}" for n, d in medium)

    print(f"energy groups for an object of {citrus_info['grouping_object_size_cm']} cm made of "
          f"{medium_text(citrus_info['grouping_medium'])} g/cm^3 (measured on the FDK volume; the truth used "
          f"{data['meta'].get('grouping_object_size_cm')} cm of {medium_text(data['meta'].get('grouping_medium'))} "
          f"g/cm^3): edges {np.round(solver.library.group_edges.double().cpu().numpy(), 1).tolist()} keV")

    # ---- 3. no scatter correction ------------------------------------------------------------------------------
    no_correction_history: list = []
    t0 = time.perf_counter()
    X_no_correction, no_correction_info = mbir_recon(Y, I0, angles, grid, geometry, config,
                                                     regularization=regularization, model=model,
                                                     callback=tracker(no_correction_history))
    t_none = time.perf_counter() - t0

    # ---- summary ---------------------------------------------------------------------------------------------
    records = citrus_info["records"][1:]
    summary = {
        "views": len(angles), "start_view_angle_deg": math.degrees(float(angles[0])), "scan_range_deg": span,
        "short_scan": bool(is_short_scan(angles)), "delta_voxel_cm": args.delta_voxel, "grid": list(grid.shape),
        "transport_cell_cm": args.transport_cell_cm, "data_tag": tag, "materials": list(reference_materials),
        "grouping_object_size_cm": citrus_info["grouping_object_size_cm"],
        "grouping_medium": citrus_info["grouping_medium"],
        "truth_grouping_object_size_cm": data["meta"].get("grouping_object_size_cm"),
        "truth_grouping_medium": data["meta"].get("grouping_medium"),
        "quadrature_point_spacing_cm": loop_config.quadrature_point_spacing_cm,
        "quadrature_points_per_axis": loop_config.quadrature_points(max(grid.deltas)),
        "outer": args.outer_iterations,
        "truth_sweep": [truth_polar, truth_azimuthal], "truth_cone": [cone_polar, cone_azimuthal],
        "scatter_view_stride": args.scatter_view_stride, "scatter_view_interp_order": interpolation_order,
        "mbir_iterations_per_outer_iteration": args.mbir_iterations,
        "resume_error_sinogram": bool(args.resume_error_sinogram),
        "scatter_steps": len(records),
        "computed_views": [int(v) for v in citrus_info["computed_views"]],
        "regularization": regularization,
        "citrus": {"rmse_rel": citrus_history[-1]["rmse_rel"], "rmse_rel_core": citrus_history[-1]["rmse_rel_core"],
                   "seconds": t_citrus, "scatter_s_per_outer": float(np.mean([r["scatter_seconds"] for r in records])),
                   "mbir_s_per_outer": float(np.mean([r["mbir_seconds"] for r in records])),
                   "state": citrus_info["state"],
                   "gpu_peak_allocated_mib": max(r.get("gpu_peak_allocated_mib", 0.0) for r in records),
                   "gpu_peak_allocated_mib_per_outer": [r.get("gpu_peak_allocated_mib") for r in records],
                   "mbir_s": [r["mbir_seconds"] for r in records],
                   "scatter_s": [r["scatter_seconds"] for r in records]},
        "no_correction": {"rmse_rel": no_correction_history[-1]["rmse_rel"],
                          "rmse_rel_core": no_correction_history[-1]["rmse_rel_core"],
                          "seconds": t_none},
        "fdk_rmse_rel": citrus_history[0]["rmse_rel"],
        "history": {"citrus": citrus_history, "no_correction": no_correction_history},
    }
    with open(os.path.join(args.out, f"citrus_{out_tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if args.save_recon:
        np.save(os.path.join(args.out, f"citrus_{out_tag}_recon.npy"), X_citrus.detach().cpu().numpy())
    print(f"\nrelative RMSE (whole prism / core): FDK {citrus_history[0]['rmse_rel']:.4f} / "
          f"{citrus_history[0]['rmse_rel_core']:.4f}, "
          f"no correction {no_correction_history[-1]['rmse_rel']:.4f} / "
          f"{no_correction_history[-1]['rmse_rel_core']:.4f}, "
          f"CITRUS {citrus_history[-1]['rmse_rel']:.4f} / {citrus_history[-1]['rmse_rel_core']:.4f}")
    print(f"CITRUS {t_citrus:.1f} s ({summary['citrus']['scatter_s_per_outer']:.1f} s scatter + "
          f"{summary['citrus']['mbir_s_per_outer']:.1f} s MBIR per outer iteration), no correction {t_none:.1f} s; "
          f"peak allocated GPU memory of CITRUS {summary['citrus']['gpu_peak_allocated_mib']:.0f} MiB")

    # ---- figures ---------------------------------------------------------------------------------------------
    Xt, X0 = data["X_true"], citrus_info["init_recon"]
    vols = [("truth", Xt), ("FDK (initialization)", X0), ("MBIR, no correction", X_no_correction), ("CITRUS", X_citrus)]
    figs = {
        "slices": plot_slices(vols, grid, title=f"central slice z = 0 ({len(angles)} views, "
                                                f"{args.outer_iterations} outer iterations)"),
        "profiles": plot_profiles(vols, grid, title="central row y = 0 (lung -8..-2, air 2..4, bone 6..8 cm)"),
        "rmse": plot_history([("CITRUS", [h["rmse_rel"] for h in citrus_history]),
                              ("MBIR, no correction", [h["rmse_rel"] for h in no_correction_history])],
                             title="relative RMSE over the prism"),
    }
    # estimated vs true scatter-to-primary ratio of view 0 (central detector row)
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    u = geometry.pixel_uv(dtype=torch.float64)[0].numpy()
    r = geometry.num_det_rows // 2
    ax.plot(u, spr_true[0, r], color="k", lw=1.2, label="true SPR")
    ax.plot(u, citrus_info["spr"][0, r].cpu().numpy(), ls="--",
            label=f"CITRUS estimate (outer iteration {args.outer_iterations})")
    ax.set_xlabel("u (cm), central detector row, view 0")
    ax.set_ylabel("scatter-to-primary ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    figs["spr"] = fig
    for name, fig in figs.items():
        p = os.path.join(args.out, f"citrus_{out_tag}_{name}.png")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"figure: {p}")
    print(f"total {time.perf_counter() - start_time:.1f} s")


if __name__ == "__main__":
    main()
