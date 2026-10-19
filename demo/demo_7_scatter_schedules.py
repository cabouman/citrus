"""Demo 7: scatter schedules, or how the scatter estimate follows the reconstruction.

Reconstruct the scan of demo 2 (the WLAB phantom, 125 kVp, a short circular cone-beam scan) five times. Every
reconstruction starts from the same FDK volume and runs MBIR with the same frozen regularization, weights of the
measured counts and seeded partitions; only the scatter estimate that corrects the counts differs:

* no correction;
* sequential: one converged scatter solve per view from the FDK volume, then frozen (the usual pipeline);
* first order only: every outer iteration, the first-order scatter of the current volume (no multiple scatter);
* decoupled: every outer iteration, a fresh converged solve per view from the current volume;
* CITRUS: every outer iteration, one transport sweep per view, the collided fluence carried over (citrus_recon).

The first four run the loop of citrus_recon written out with the functions it is made of (the solver's converged
solve, the scatter operators, the primary and the angular interpolation of the scatter). First order only, decoupled
and CITRUS run the adaptive numbers of citrus_recon's defaults, which stop the MBIR iterations and the loop once the
fit of the corrected sinogram stops improving; no correction and sequential, whose correction never changes, run the
MBIR iterations of the CITRUS run, outer iteration by outer iteration. The demo uses demo 2's data file in demo/input
and synthesizes it when demo 2 has not run yet.

What to expect: the scatter of the FDK volume is wrong where the FDK volume is (cupping, too low inserts), so the
sequential correction leaves shading; first-order scatter misses the multiple scatter, so the volume stays cupped.
Decoupled and CITRUS reach nearly the same accuracy, CITRUS for one sweep per view per outer iteration instead of a
converged solve (the run times in the titles, after the shared FDK volume; the uncollided fluence and the scatter
image, which every schedule but the first two computes each time, cost as much as several sweeps). The convergence
plot shows the relative RMSE against the truth after every MBIR iteration. The five reconstructions take about ten
minutes on a GPU, the decoupled one the longest.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from citrus import CitrusConfig, citrus_recon
from citrus.citrus import (DensityModel, interpolate_scatter, multiple_collision_operator, primary_signals,
                           project_scatter, scatter_view_indices, uncollided_operator)
from citrus.recon.mbir_bridge import ViewChunkProjector, forward_project, mbir_iterations
from citrus.scatter_solver import (ConeGeometry, ScatterSolver, SolverConfig, ideal_counting_response,
                                   load_spectrum_txt, no_grid)
from citrus.utilities import (WlabPhantom, central_slice, load_dataset, save_dataset, synthesize_cone_beam_data,
                              wlab_grid)

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
scatter_view_stride = 8                    # scatter on every 8th view, the others interpolated in angle

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
X_true = data["X_true"]
X_true_gpu = torch.as_tensor(X_true, device="cuda")
relative_error = lambda X: float((X - X_true_gpu).norm() / X_true_gpu.norm())  # noqa: E731

# ---- CITRUS; its run also sets the initial volume, the frozen regularization and the model every schedule shares ---
solver = make_solver()
loop = CitrusConfig(scatter_view_stride=scatter_view_stride)  # adaptive numbers of iterations
history = []                                                  # relative RMSE after MBIR iteration 0 (FDK), 1, ...
X_citrus, info = citrus_recon(data["Y"], I0, angles, grid, geometry, solver, loop,
                              mbir_callback=lambda i, X, record: history.append(relative_error(X)))
results = {"CITRUS": (X_citrus, sum(r["scatter_seconds"] for r in info["records"]), info["seconds"], history)}
citrus_schedule = [r["mbir_iterations"] for r in info["records"][1:]]  # MBIR iterations of each outer iteration

# ---- The other schedules: the same loop with another scatter estimate ---------------------------------------------
model, X_initial = info["model"], info["init_recon"]          # the model keeps the frozen regularization
Y = torch.as_tensor(data["Y"], device="cuda")
density_model = DensityModel.from_library(solver.library)
projector = ViewChunkProjector(grid, geometry, angles, model=model)
computed = scatter_view_indices(num_views, scatter_view_stride)


def scatter_to_primary(X, view_scatter):
    """Scatter-to-primary ratio of every view of the volume X, the scatter computed on every stride-th view."""
    fields = solver.material_fields_from_attenuation(X, grid, density_model)  # densities of X, one material at a time
    scatter = torch.stack([view_scatter(fields, float(angles[v])) for v in computed])
    scatter = interpolate_scatter(scatter, angles, computed, 3)              # Catmull-Rom in angle
    return scatter / primary_signals(solver, fields, angles, projector).clamp_min(1e-30)


def converged(fields, angle):                                  # a converged source iteration of one view
    return solver.solve_view_from_fields(fields, angle).scatter_signal


def first_order(fields, angle):                                # the first-collision source projected, no sweep
    uncollided = uncollided_operator(solver, None, grid, angle, fields)
    no_collided = multiple_collision_operator(solver, None, grid, angle, None, fields)
    return project_scatter(solver, None, grid, angle, uncollided, no_collided, fields)["signal"]


def fit(X, corrected):                                        # the residual |A X - y_c| / |y_c| of the stop rules
    return float((forward_project(model, X) - corrected).norm() / corrected.norm())


def reconstruct(ratio_of_volume, schedule=None):
    """The outer loop of citrus_recon with the scatter-to-primary ratio ratio_of_volume(k, X) of outer iteration k:
    the MBIR iterations of ``schedule`` per outer iteration, or None for its adaptive rules (``loop``'s thresholds
    and maxima)."""
    X, i, scatter_seconds, start = X_initial.clone(), 0, 0.0, time.perf_counter()
    history, previous_fit, adaptive = [relative_error(X)], None, schedule is None
    for k in range(loop.max_outer_iterations if adaptive else len(schedule)):
        max_mbir = loop.max_mbir_iterations_per_outer_iteration if adaptive else schedule[k]
        t0 = time.perf_counter()
        ratio = ratio_of_volume(k, X)
        torch.cuda.synchronize()
        scatter_seconds += time.perf_counter() - t0
        corrected = -torch.log((Y / (1.0 + ratio)).clamp_min(loop.min_counts) / I0)
        fits = [fit(X, corrected)] if adaptive else []
        for j in range(1, max_mbir + 1):
            np.random.seed([loop.base_seed, i])                # the partitions citrus_recon draws for iteration i
            X, _ = mbir_iterations(model, corrected, X, first_iteration=i, weights=Y / I0)
            i += 1
            history.append(relative_error(X))
            if adaptive:
                fits.append(fit(X, corrected))                 # stop once the mean improvement per iteration of
                if j >= 2 and j < max_mbir:                    # the last two is too small
                    if 0.5 * (fits[-3] - fits[-1]) / fits[-3] < loop.mbir_stop_relative_improvement:
                        break
        if adaptive:                                           # stop once an outer iteration barely improves the fit
            improvement = None if previous_fit is None else (previous_fit - fits[-1]) / previous_fit
            if improvement is not None and improvement < loop.stop_relative_improvement:
                break
            previous_fit = fits[-1]
    torch.cuda.synchronize()
    return X, scatter_seconds, time.perf_counter() - start, history


frozen = {}
schedules = {
    "no correction": lambda k, X: torch.zeros_like(Y),
    "sequential": lambda k, X: frozen.setdefault("ratio", scatter_to_primary(X, converged)) if k == 0
    else frozen["ratio"],
    "first order only": lambda k, X: scatter_to_primary(X, first_order),
    "decoupled": lambda k, X: scatter_to_primary(X, converged),
}
for label, ratio_of_volume in schedules.items():
    fixed = label in ("no correction", "sequential")
    results[label] = reconstruct(ratio_of_volume, citrus_schedule if fixed else None)

# ---- Results: the relative RMSE against the truth and the central slices ----------------------------------------
relative_rmse = lambda X: np.linalg.norm(X - X_true) / np.linalg.norm(X_true)  # noqa: E731
order = ["no correction", "sequential", "first order only", "decoupled", "CITRUS"]
panels = [("truth", X_true)]
for label in order:
    X, scatter_seconds, total_seconds, _ = results[label]   # the loop after the shared FDK volume
    X = X.cpu().numpy()
    print(f"{label:17s} relative RMSE {relative_rmse(X):.4f}, scatter {scatter_seconds:6.1f} s, "
          f"total {total_seconds:6.1f} s")
    panels.append((f"{label}\nrel. RMSE {relative_rmse(X):.4f}, {total_seconds:.0f} s", X))
lower, (extent_x, extent_y, _) = grid.lower_corner(), grid.extent
fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), constrained_layout=True)
for ax, (title, X) in zip(axes.flat, panels):
    image = ax.imshow(central_slice(X), cmap="gray", vmin=0.0, vmax=0.3, origin="lower",
                      extent=[lower[0], lower[0] + extent_x, lower[1], lower[1] + extent_y])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
fig.colorbar(image, ax=axes, shrink=0.8, label="attenuation (1/cm)")
styles = {"no correction": ("#52514e", (0, (1, 2))), "sequential": ("#1baf7a", "-."),
          "first order only": ("#4a3aa7", (0, (5, 2, 1, 2))), "decoupled": ("#eb6834", "--"),
          "CITRUS": ("#2a78d6", "-")}  # color and line style of each schedule
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
for label in order:
    color, linestyle = styles[label]
    ax.semilogy(results[label][3], color=color, linestyle=linestyle, linewidth=2, label=label)
ax.set(xlabel="MBIR iteration (0 = FDK)", ylabel="relative RMSE",
       xlim=(0, max(len(results[label][3]) - 1 for label in order)),
       title="Relative RMSE against the truth")
ax.grid(True, which="major", alpha=0.3)
ax.legend(frameon=False)
plt.show()
