"""Demo basic 2: a monoenergetic parallel beam, as in synchrotron CT.

Reconstruct a parallel-beam CT at one photon energy, the setting of a synchrotron beamline with a monochromator, of a
breast-sized phantom: a 10 cm cylinder of adipose tissue with a fibroglandular region, a lesion and a small
calcification, imaged at 32 keV (the energy range of synchrotron breast CT, where 10 cm of tissue transmits a few
percent). The panel stands at a distance from the rotation axis; the scatter that reaches it falls as this air gap
grows, while the primary does not change. The data are synthetic, made once and cached in demo/input.

For your own beamline, set the energy, the panel and the gap, and replace the synthetic data by your measured counts
[views, detector rows, detector channels] and flat field I0.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from citrus import CitrusConfig, citrus_recon
from citrus.scatter_solver import (Grid, ParallelGeometry, ScatterSolver, SolverConfig, ideal_counting_response,
                                   monoenergetic_spectrum, no_grid)
from citrus.utilities import (breast_phantom, load_dataset, plot_profiles, plot_slices, save_dataset,
                              synthesize_cone_beam_data, view_angles)

# ---- Parameters -----------------------------------------------------------------------------------------------
energy_kev = 32.0                          # photon energy of the monochromatic beam in keV
air_gap = 10.0                             # distance from the rotation axis to the panel in cm (try 100)
materials = ["air", "adipose_icrp", "soft_tissue_icru44", "bone_cortical_icrp"]  # reference materials
breast_radius = 5.0                        # radius in cm of the adipose cylinder
gland_center, gland_radius = -1.0, 2.5     # fibroglandular region (soft tissue) along z at (x, 0), in cm
lesion_center, lesion_radius = 2.5, 0.5    # lesion (soft tissue, 1.10 g/cm^3) along z at (x, 0), in cm
calcification_center, calcification_size = -2.5, 0.2  # calcification (bone) along z at (x, 0), width in cm
delta_voxel = 0.1                          # voxel pitch in cm of the 11 x 11 x 4 cm grid, also the detector pitch
num_det_channels = 120                     # detector channels (the rows are the grid's slices)
num_views = 180                            # number of views over 180 degrees
I0 = 1.0e5                                 # flat-field counts per pixel
transport_cell_cm = 0.5                    # transport cell size in cm, finer than the 1 cm default for a small object
quadrature_point_spacing_cm = 0.25         # spacing of the first-order scatter points in cm

# ---- Scanner and solver ---------------------------------------------------------------------------------------
grid = Grid(shape=(round(11 / delta_voxel), round(11 / delta_voxel), round(4 / delta_voxel)), delta_voxel=delta_voxel)
geometry = ParallelGeometry.from_grid(grid, axis_detector_dist=air_gap, num_det_channels=num_det_channels,
                                      delta_det_channel=delta_voxel)  # detector row r sees slice r
angles = view_angles(num_views, geometry)  # 180 degrees, the default range of a parallel beam
config = SolverConfig(materials=tuple(materials), transport_cell_cm=transport_cell_cm,
                      quadrature_point_spacing_cm=quadrature_point_spacing_cm)
make_solver = lambda: ScatterSolver(None, config, geometry, detector=ideal_counting_response(),  # noqa: E731
                                    grid_response=no_grid(), device="cuda",
                                    spectrum=monoenergetic_spectrum(energy_kev))

# ---- Phantom: densities [materials, rows, cols, slices] in g/cm^3, rows in library order -------------------------
solver = make_solver()
# Adipose tissue, gland, lesion and calcification, each region replacing the regions before it (citrus.utilities).
rho = breast_phantom(grid, solver.materials, breast_radius=breast_radius, gland_center=gland_center,
                     gland_radius=gland_radius, lesion_center=lesion_center, lesion_radius=lesion_radius,
                     calcification_center=calcification_center, calcification_size=calcification_size, device="cuda")

# ---- Data: loaded when a file of this name exists in demo/input, else synthesized once (about a minute on a GPU) ---
name = (f"breast_parallel_{num_views}v_{delta_voxel:g}cm_{num_det_channels}ch_gap{air_gap:g}cm_"
        f"{energy_kev:g}kev").replace(".", "p")
path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input", name)  # demo/input
if os.path.exists(path + ".npz"):
    data = load_dataset(path)
else:
    data = synthesize_cone_beam_data(rho, grid, geometry, angles, make_solver(), I0=I0)
    save_dataset(data, path)
Y, X_true = data["Y"], data["X_true"]
print(f"Scatter-to-primary ratio at a {air_gap:g} cm gap: mean {data['SPR'].mean():.3f}, max {data['SPR'].max():.3f}")

# ---- CITRUS: the initial volume is the filtered back projection of the uncorrected counts ------------------------
X_citrus, info = citrus_recon(Y, I0, angles, grid, geometry, solver, CitrusConfig(verbose=True))

# ---- Results ----------------------------------------------------------------------------------------------------
X_citrus, X_initial = X_citrus.cpu().numpy(), info["init_recon"].cpu().numpy()
relative_rmse = lambda X: np.linalg.norm(X - X_true) / np.linalg.norm(X_true)  # noqa: E731
print(f"Relative RMSE against the truth: initial volume {relative_rmse(X_initial):.4f}, CITRUS "
      f"{relative_rmse(X_citrus):.4f} after {info['mbir_iterations_total']} MBIR iterations in "
      f"{len(info['records']) - 1} outer iterations ({info['seconds']:.0f} s)")
volumes = [("Truth", X_true), ("Initial FBP volume", X_initial), ("CITRUS", X_citrus)]
plot_slices(volumes, grid, vmin=0.2, vmax=0.5, title=f"Central slice z = 0 at {energy_kev:g} keV")
plot_profiles(volumes, grid, title="Central row y = 0 (calcification -2.6..-2.4 cm, gland -3.5..1.5, lesion 2..3)")
plt.show()
