"""Demo 6: follow the CITRUS iterations with callbacks.

Reconstruct the scan of demo 2 (the WLAB phantom, 125 kVp, a short circular cone-beam scan) with CITRUS and follow
its convergence with the two callbacks of citrus_recon: ``mbir_callback`` after every MBIR iteration and ``callback``
after every outer iteration (one scatter correction followed by its MBIR iterations). The demo uses demo 2's data
file in demo/input and synthesizes it when demo 2 has not run yet.

The relative RMSE against the truth needs the true volume; the sinogram residual of the records does not, so it is
the measure to watch on measured data.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from citrus import CitrusConfig, citrus_recon
from citrus.scatter_solver import (ConeGeometry, ScatterSolver, SolverConfig, ideal_counting_response,
                                   load_spectrum_txt, no_grid)
from citrus.utilities import WlabPhantom, load_dataset, save_dataset, synthesize_cone_beam_data, wlab_grid

# ---- Parameters (those of demo 2, so that the data file is shared) ------------------------------------------------
demo_dir = os.path.dirname(os.path.abspath(__file__))
spectrum, spectrum_name = load_spectrum_txt(os.path.join(demo_dir, "spectra", "spectrum_125kvp.txt")), "125kvp"
materials = ["air", "lung_icrp", "adipose_icrp", "water", "muscle_icru44", "soft_tissue_icru44",
             "bone_cortical_icrp"]  # reference materials of the density model
num_det_rows, num_det_channels = 184, 150  # detector panel size in pixels
pixel_pitch = 0.4                          # detector pixel pitch in cm
source_detector_dist = 150.0               # source to detector distance in cm
source_iso_dist = 100.0                    # source to rotation axis distance in cm
delta_voxel = 0.5                          # voxel pitch in cm of the WLAB phantom (a 30 x 20 x 40 cm water prism)
num_views = 180                            # number of views
start_angle = 0.0                          # first view angle in degrees
fan_angle = 2 * np.degrees(np.arctan(num_det_channels * pixel_pitch / 2 / source_detector_dist))
end_angle = start_angle + 180.0 + fan_angle  # end of the angular range in degrees (excluded): a short scan
I0 = 1.0e5                                 # flat-field counts per pixel
scatter_view_stride = 8                    # scatter step on every 8th view, the others interpolated in angle

# ---- Scanner, solver and data (loaded from demo/input, else synthesized once, about 2 minutes on a GPU) ----------
geometry = ConeGeometry(source_iso_dist=source_iso_dist, source_detector_dist=source_detector_dist,
                        num_det_rows=num_det_rows, num_det_channels=num_det_channels,
                        delta_det_row=pixel_pitch, delta_det_channel=pixel_pitch)
grid = wlab_grid(delta_voxel)
angles = np.radians(np.linspace(start_angle, end_angle, num_views, endpoint=False))
config = SolverConfig(materials=tuple(materials))
make_solver = lambda: ScatterSolver(None, config, geometry, detector=ideal_counting_response(),  # noqa: E731
                                    grid_response=no_grid(), device="cuda", spectrum=spectrum)
name = (f"wlab_{num_views}v_{delta_voxel:g}cm_{start_angle:g}-{end_angle:.1f}deg_{num_det_rows}x{num_det_channels}px"
        f"{pixel_pitch:g}cm_{spectrum_name}").replace(".", "p")
path = os.path.join(demo_dir, "input", name)
if os.path.exists(path + ".npz"):
    data = load_dataset(path)
else:
    truth_solver = make_solver()
    data = synthesize_cone_beam_data(WlabPhantom(grid, truth_solver.materials, device="cuda"), grid, geometry,
                                     angles, truth_solver, I0=I0)
    save_dataset(data, path)
Y, X_true = data["Y"], torch.as_tensor(data["X_true"], device="cuda")

# ---- Callbacks ------------------------------------------------------------------------------------------------
# mbir_callback(i, X, record): called for the initial volume (i = 0, the FDK) and after every MBIR iteration i, with
# record["outer_iteration"] and record["sinogram_nrmse"], the relative RMSE of A X against the corrected sinogram of
# its outer iteration. Here it records that residual and the relative RMSE of the volume against the truth.
per_mbir_iteration = {"outer_iteration": [], "volume_rmse": [], "sinogram_nrmse": []}


def mbir_callback(i, X, record):
    per_mbir_iteration["outer_iteration"].append(record["outer_iteration"])
    per_mbir_iteration["volume_rmse"].append(float((X - X_true).norm() / X_true.norm()))
    per_mbir_iteration["sinogram_nrmse"].append(record["sinogram_nrmse"])


# callback(k, X_k, record): called for the initial volume (k = 0) and after every outer iteration k; the record is
# info["records"][k]: the MBIR iterations taken, the residual, the relative change of the scatter estimate (from
# k = 2 on), the mean relative change of the scatter accumulators and the stage timings, among others.
per_outer_iteration = {"mbir_iterations": [], "scatter_change": []}


def callback(k, X, record):
    if k == 0:
        return
    per_outer_iteration["mbir_iterations"].append(record["mbir_iterations"])
    per_outer_iteration["scatter_change"].append(record.get("scatter_change_relative", np.nan))  # none at k = 1
    print(f"outer iteration {k}: {record['mbir_iterations']} MBIR iterations, sinogram residual "
          f"{record['sinogram_nrmse']:.4f}, scatter change {per_outer_iteration['scatter_change'][-1]:.4f}, "
          f"accumulator change {record['accumulator_change_mean']:.2e}, scatter {record['scatter_seconds']:.1f} s, "
          f"MBIR {record['mbir_seconds']:.1f} s")


# ---- CITRUS with the default adaptive iteration counts ------------------------------------------------------------
X_citrus, info = citrus_recon(Y, I0, angles, grid, geometry, make_solver(),
                              CitrusConfig(scatter_view_stride=scatter_view_stride),
                              callback=callback, mbir_callback=mbir_callback)
print(f"Relative RMSE against the truth: initial volume {per_mbir_iteration['volume_rmse'][0]:.4f}, CITRUS "
      f"{per_mbir_iteration['volume_rmse'][-1]:.4f} after {info['mbir_iterations_total']} MBIR iterations")

# ---- Convergence curves: the dotted lines mark the scatter corrections (the start of every outer iteration) -------
outer = np.array(per_mbir_iteration["outer_iteration"])
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for ax, key, title in ((axes[0], "volume_rmse", "Relative RMSE of the volume against the truth"),
                       (axes[1], "sinogram_nrmse", "Relative RMSE of A X against the corrected sinogram")):
    ax.semilogy(per_mbir_iteration[key], "o-", ms=3)
    for i in np.flatnonzero(np.diff(outer)):
        ax.axvline(i + 0.5, color="gray", ls=":", lw=0.8)
    ax.set(xlabel="MBIR iteration (0 = initial volume)", title=title)
k = np.arange(1, len(per_outer_iteration["mbir_iterations"]) + 1)
bars = axes[2].bar(k, per_outer_iteration["mbir_iterations"], color="lightgray", label="MBIR iterations taken")
axes[2].set(xlabel="outer iteration", ylabel="MBIR iterations", title="Per outer iteration")
twin = axes[2].twinx()
line, = twin.semilogy(k, per_outer_iteration["scatter_change"], "o-", color="tab:red", label="scatter change")
twin.set_ylabel("relative change of the scatter estimate")
axes[2].legend(handles=[bars, line], loc="upper right", fontsize=8)
fig.tight_layout()
plt.show()
