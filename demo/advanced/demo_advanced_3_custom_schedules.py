"""Demo advanced 3: custom schedules, or how the scatter estimate follows the reconstruction.

Reconstruct the scan of demo_basic_3_cone_beam (the WLAB phantom, 125 kVp, a short circular cone-beam scan) six
times with citrus_recon. Every reconstruction starts from the same FDK volume and runs MBIR with the same frozen
regularization, weights of the measured counts and seeded partitions; only the ``Schedule`` of the scatter estimate
differs:

* MBIR only: no scatter correction;
* sequential: one converged solve per view from the FDK volume, then frozen (the legacy approach);
* first-order only: every outer iteration, the first-order scatter of the current volume (no multiple scatter);
* decoupled converged scatter: every outer iteration, a fresh converged solve per view from the current volume;
* CITRUS default: every outer iteration, one transport sweep per view, the collided fluence carried over;
* CITRUS (2 scatter sweeps): the same with two sweeps per view, the second one carrying one more collision order.

Each section below builds one schedule. A schedule sets the number of outer iterations (or a threshold on the change
of the volume that stops them earlier), the number of MBIR iterations per outer iteration, the number of scatter
sweeps per view in each outer iteration (or "converged", with a threshold on the change of the collided fluence) and
how many outer iterations update the scatter. The demo uses the data file of demo_basic_3_cone_beam in demo/input and
synthesizes it when that demo has not run yet.

What to expect: the scatter of the FDK volume is wrong where the FDK volume is (cupping, too low inserts), so the
sequential correction leaves shading; first-order scatter misses the multiple scatter, so the volume stays cupped.
Decoupled and CITRUS reach nearly the same accuracy, CITRUS for one sweep per view per outer iteration instead of a
converged solve. A second sweep makes the estimate follow the volume more closely for the cost of that sweep. The
convergence plots show the relative RMSE against the truth after every MBIR iteration, against the iteration and the
wall time after the FDK volume, with a dot where each scatter estimate is made. The six reconstructions take several
minutes on a GPU, the decoupled one the longest.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from citrus import CitrusConfig, Schedule, citrus_recon
from citrus.scatter_solver import (ConeGeometry, ScatterSolver, SolverConfig, ideal_counting_response,
                                   load_spectrum_txt, no_grid)
from citrus.utilities import (WlabPhantom, central_slice, load_dataset, save_dataset, synthesize_cone_beam_data,
                              wlab_grid)

# ---- Scanner, solver and data: those of demo_basic_3_cone_beam, so that the data file is shared -------------------
demo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # demo/, with spectra/ and input/
spectrum = load_spectrum_txt(os.path.join(demo_dir, "spectra", "spectrum_125kvp.txt"))
materials = ("air", "lung_icrp", "adipose_icrp", "water", "muscle_icru44", "soft_tissue_icru44", "bone_cortical_icrp")
geometry = ConeGeometry(source_iso_dist=100.0, source_detector_dist=150.0, num_det_rows=184, num_det_channels=150,
                        delta_det_row=0.4, delta_det_channel=0.4)
end_angle = 180.0 + 2 * np.degrees(np.arctan(150 * 0.4 / 2 / 150.0))  # a short scan: 180 degrees plus the fan angle
angles = np.radians(np.linspace(0.0, end_angle, 180, endpoint=False))
grid, I0 = wlab_grid(0.5), 1.0e5
solver = ScatterSolver(None, SolverConfig(materials=materials), geometry, detector=ideal_counting_response(),
                       grid_response=no_grid(), device="cuda", spectrum=spectrum)
path = os.path.join(demo_dir, "input", f"wlab_180v_0.5cm_0-{end_angle:.1f}deg_184x150px0.4cm_125kvp".replace(".", "p"))
if os.path.exists(path + ".npz"):
    data = load_dataset(path)
else:
    data = synthesize_cone_beam_data(WlabPhantom(grid, solver.materials, device="cuda"), grid, geometry, angles,
                                     solver, I0=I0)
    save_dataset(data, path)
X_true = torch.as_tensor(data["X_true"], device="cuda")
relative_error = lambda X: float((X - X_true).norm() / X_true.norm())  # noqa: E731
loop = CitrusConfig(scatter_view_stride=8)  # scatter on every 8th view; the iteration counts are the schedule's
shared = {}  # the FDK volume, the frozen regularization and the model, set by the first run and shared by the others


def run(schedule):
    """citrus_recon with a schedule; the relative RMSE and the clock after every MBIR iteration."""
    errors, stamps = [], []
    track = lambda i, X, record: (errors.append(relative_error(X)), stamps.append(time.perf_counter()))  # noqa: E731
    X, info = citrus_recon(data["Y"], I0, angles, grid, geometry, solver, loop, schedule=schedule,
                           mbir_callback=track, **shared)
    shared.update(init_recon=info["init_recon"], regularization=info["regularization"], model=info["model"])
    n = schedule.mbir_iterations_per_outer_iteration
    estimates = [(k - 1) * n for k in range(1, info["outer_iterations_total"] + 1) if schedule.updates_scatter(k)]
    times = [0.0] + [info["seconds"] - (stamps[-1] - t) for t in stamps[1:]]  # wall time after the FDK volume
    return X.cpu().numpy(), info["seconds"], errors, estimates, times


# A short warm-up run builds the solver's library and compiles the GPU kernels, so that no timed run pays for them;
# it also sets the FDK volume, the regularization and the model that every run shares.
run(Schedule(num_outer_iterations=1, mbir_iterations_per_outer_iteration=1))
results = {}

# ---- MBIR only: no scatter correction -----------------------------------------------------------------------------
# No outer iteration updates the scatter, so MBIR runs on the measured counts.
results["MBIR Only"] = run(Schedule(num_scatter_updates=0))

# ---- Sequential: the converged scatter of the FDK volume, then frozen ----------------------------------------------
# "converged" solves every view by source iteration; num_scatter_updates is the number of outer iterations, from the
# first, that make a new scatter estimate, and the later ones keep the last one, here that of the FDK volume.
results["Sequential"] = run(Schedule(scatter_sweeps_per_outer_iteration="converged", num_scatter_updates=1))

# ---- First-order only: no collided fluence ------------------------------------------------------------------------
# Zero sweeps leave the collided fluence out: the scatter estimate is the first-collision source projected.
results["First-Order Only"] = run(Schedule(scatter_sweeps_per_outer_iteration=0))

# ---- Decoupled: a converged solve per view in every outer iteration ------------------------------------------------
# "converged" solves every view anew by source iteration, from zero collided fluence, until the relative change of
# the collided fluence falls below scatter_change_threshold (None: the solver's source_iteration_tolerance).
results["Decoupled Converged Scatter"] = run(Schedule(scatter_sweeps_per_outer_iteration="converged",
                                                      scatter_change_threshold=None))

# ---- CITRUS: the default schedule ---------------------------------------------------------------------------------
# 14 outer iterations, each one transport sweep per computed view followed by 8 MBIR iterations. Each view carries
# its collided fluence over from the last outer iteration, so one sweep advances the scatter estimate by one
# collision order with the densities of the current volume (the age-ordered recursion).
results["CITRUS Default"] = run(Schedule(num_outer_iterations=14, mbir_iterations_per_outer_iteration=8,
                                         scatter_sweeps_per_outer_iteration=1))  # the same as Schedule()

# ---- CITRUS with two sweeps per outer iteration -------------------------------------------------------------------
# The second sweep transports the scattering source of the change that the first one made to the collided fluence,
# which carries the recursion one more collision order for the cost of one sweep (no new first-collision source).
results["CITRUS (2 Scatter Sweeps)"] = run(Schedule(scatter_sweeps_per_outer_iteration=2))

# ---- Results: the relative RMSE against the truth, the central slices and the convergence -----------------------
X_fdk = shared["init_recon"]
panels = [("Ground Truth", data["X_true"]), (f"FDK\nrel. RMSE {relative_error(X_fdk):.4f}", X_fdk.cpu().numpy())]
panels += [(f"{label}\nrel. RMSE {r[2][-1]:.4f}, {r[1]:.0f} s", r[0]) for label, r in results.items()]
lower, (extent_x, extent_y, _) = grid.lower_corner(), grid.extent
fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), constrained_layout=True)
for ax, (title, X) in zip(axes.flat, panels):
    image = ax.imshow(central_slice(X), cmap="gray", vmin=0.0, vmax=0.3, origin="lower",
                      extent=[lower[0], lower[0] + extent_x, lower[1], lower[1] + extent_y])
    ax.set(title=title, xlabel="x (cm)", ylabel="y (cm)")
fig.colorbar(image, ax=axes, shrink=0.8, label="attenuation (1/cm)")
styles = {"MBIR Only": ("#52514e", (0, (1, 2))), "Sequential": ("#1baf7a", "-."),
          "First-Order Only": ("#4a3aa7", (0, (5, 2, 1, 2))), "Decoupled Converged Scatter": ("#eb6834", "--"),
          "CITRUS Default": ("#2a78d6", "-"), "CITRUS (2 Scatter Sweeps)": ("#0b9bb5", (0, (6, 2)))}
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for label, (_, seconds, errors, estimates, times) in results.items():
    print(f"{label:28s} relative RMSE {errors[-1]:.4f}, {seconds:6.1f} s after the FDK volume")
    color, linestyle = styles[label]
    for ax, x in zip(axes, (np.arange(len(errors)), np.array(times))):
        ax.semilogy(x, errors, color=color, linestyle=linestyle, linewidth=2, label=label)
        ax.semilogy(x[estimates], [errors[i] for i in estimates], linestyle="none", marker="o", markersize=4,
                    color=color)
axes[0].plot([], [], linestyle="none", marker="o", markersize=4, color="0.4", label="scatter estimate")
axes[0].set(xlabel="MBIR iteration (0 = FDK)", xlim=(0, len(errors) - 1))
axes[1].set(xlabel="wall time after the FDK volume (s)", xlim=(0, max(r[4][-1] for r in results.values())))
for ax in axes:
    ax.set(ylabel="relative RMSE", title="Relative RMSE against the truth")
    ax.grid(True, which="major", alpha=0.3)
axes[0].legend(frameon=False)
plt.show()
