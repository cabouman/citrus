"""Demo 3: a helical scan on a curved detector with an anti-scatter grid.

Reconstruct a helical scan of the WLAB phantom (a 30 x 20 x 40 cm water prism with lung, air and bone inserts) by a
clinical-CT-like scanner: a cylindrical panel of CsI scintillator centered on the source and a 1-D anti-scatter grid
whose tungsten septa run parallel to the rotation axis, each focused on the source, as on a CT detector. The helix
covers the whole height of the prism. The measured data are synthetic, made once and cached in demo/input.

For your own scanner, set the panel, the grid and the helix, and replace the synthetic data by your measured counts
[views, detector rows, detector channels] with their view angles and table positions (``with_helical_z_shifts``).
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from citrus import CitrusConfig, citrus_recon
from citrus.scatter_solver import (ConeGeometry, ScatterSolver, SolverConfig, anti_scatter_grid,
                                   detector_response, load_spectrum_txt)
from citrus.utilities import (WlabPhantom, helical_scan, load_dataset, plot_profiles, plot_slices, save_dataset,
                              synthesize_cone_beam_data, view_angles, wlab_grid)

# ---- Parameters -----------------------------------------------------------------------------------------------
demo_dir = os.path.dirname(os.path.abspath(__file__))
spectrum = load_spectrum_txt(os.path.join(demo_dir, "spectra", "spectrum_125kvp.txt"))  # keV, photons per keV
materials = ["air", "lung_icrp", "adipose_icrp", "water", "muscle_icru44", "soft_tissue_icru44",
             "bone_cortical_icrp"]  # reference materials of the density model
source_iso_dist = 60.0                     # source to rotation axis distance in cm
source_detector_dist = 105.0               # source to detector distance in cm, the radius of the curved panel
num_det_rows, num_det_channels = 80, 168   # panel size in pixels: 18.3 cm along z at the axis, a 37.7 cm field
pixel_pitch = 0.4                          # pixel pitch in cm (channels: arc length on the cylinder)
scintillator_thickness = 0.06              # CsI scintillator thickness in cm (energy integrating)
septum_thickness, septum_height, septum_pitch = 0.01, 2.0, 0.1  # tungsten septa in cm, one per mm of arc
delta_voxel = 0.5                          # voxel pitch in cm of the reconstruction grid (the 30 x 20 x 40 cm prism)
num_turns = 3                              # turns of the helix
views_per_turn = 90                        # views per turn
pitch_cm = 19.5                            # table feed per turn in cm: the helix travels 58.5 cm over the 40 cm prism
I0 = 1.0e5                                 # flat-field counts per pixel
scatter_view_stride = 8                    # scatter step on every 8th view (1.7 cm of travel apart)

# ---- Scanner: curved panel, helix and grid ------------------------------------------------------------------------
panel = ConeGeometry(source_iso_dist=source_iso_dist, source_detector_dist=source_detector_dist,
                     num_det_rows=num_det_rows, num_det_channels=num_det_channels, delta_det_row=pixel_pitch,
                     delta_det_channel=pixel_pitch, use_curved_detector=True)
# Unwrapped view angles over num_turns turns, and the z shift of every view: pitch_cm per turn, centred on z = 0.
angles = view_angles(num_turns * views_per_turn, end_view_angle=2 * np.pi * num_turns)
geometry = helical_scan(panel, angles, pitch_cm=pitch_cm)
# The grid's septa run along the rows (parallel to the rotation axis) and repeat along the channels; on a curved
# panel each septum is focused along its own normal, at the source for the default focal distance.
septa = anti_scatter_grid(geometry, septum_thickness, septum_height, septum_material="tungsten",
                          pitch_cm=septum_pitch, septa_along="rows")
grid = wlab_grid(delta_voxel)
config = SolverConfig(materials=tuple(materials))
# The detector: a CsI layer that records the energy its photons deposit, from the photon data of CsI, for every
# energy and incidence angle.
detector = detector_response("cesium_iodide", scintillator_thickness, mode="energy_integrating")
# The data's and the reconstruction's solver: the CsI detector behind the grid, on the helical geometry.
make_solver = lambda: ScatterSolver(None, config, geometry, detector=detector, grid_response=septa,  # noqa: E731
                                    device="cuda", spectrum=spectrum)

# ---- Data: loaded when a file of this name exists in demo/input, else synthesized once (a few minutes on a GPU) ---
name = (f"wlab_helical_{num_turns}x{views_per_turn}v_{pitch_cm:g}cm_{delta_voxel:g}cm_{num_det_rows}x"
        f"{num_det_channels}px{pixel_pitch:g}cm_curved_septa{septum_height:g}cm_csi{scintillator_thickness:g}cm_125kvp"
        ).replace(".", "p")
path = os.path.join(demo_dir, "input", name)
if os.path.exists(path + ".npz"):
    data = load_dataset(path)
else:
    truth_solver = make_solver()
    data = synthesize_cone_beam_data(WlabPhantom(grid, truth_solver.materials, device="cuda"), grid, geometry,
                                     angles, truth_solver, I0=I0)
    save_dataset(data, path)
Y, X_true = data["Y"], data["X_true"]
print(f"Scatter-to-primary ratio behind the grid: mean {data['SPR'].mean():.3f}, max {data['SPR'].max():.3f}")

# ---- CITRUS: the same geometry object for the solver and the reconstruction (its helix is keyed by the angles) ----
X_citrus, info = citrus_recon(Y, I0, angles, grid, geometry, make_solver(),
                              CitrusConfig(scatter_view_stride=scatter_view_stride, verbose=True))

# ---- Results ----------------------------------------------------------------------------------------------------
X_citrus, X_initial = X_citrus.cpu().numpy(), info["init_recon"].cpu().numpy()
relative_rmse = lambda X: np.linalg.norm(X - X_true) / np.linalg.norm(X_true)  # noqa: E731
print(f"Relative RMSE against the truth: initial helical FDK volume {relative_rmse(X_initial):.4f}, CITRUS "
      f"{relative_rmse(X_citrus):.4f} after {info['mbir_iterations_total']} MBIR iterations in "
      f"{len(info['records']) - 1} outer iterations ({info['seconds']:.0f} s)")
volumes = [("Truth", X_true), ("CITRUS", X_citrus)]
plot_slices(volumes, grid, title="Central slice z = 0", vmax=np.max(X_true))
plot_profiles(volumes, grid, title="Central row y = 0 (lung -8..-2 cm, air 2..4 cm, bone 6..8 cm)")
# The coronal slice y = 0 over the whole height of the prism, which the helix covers.
fig, axes = plt.subplots(1, 2, figsize=(7, 5))
for ax, (label, X) in zip(axes, volumes):
    ax.imshow(X[X.shape[0] // 2].T, cmap="gray", vmin=0, vmax=np.max(X_true), origin="lower",
              extent=[-15, 15, -20, 20])
    ax.set(title=f"{label}, coronal slice y = 0", xlabel="x (cm)", ylabel="z (cm)")
fig.tight_layout()
plt.show()
