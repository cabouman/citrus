"""Demo 4: user-defined reference materials, an iodine contrast agent and a dental amalgam.

Define two materials that are not in the catalogue with ``Material(name, density, composition)``, put them in
``SolverConfig.materials`` next to catalogue names, build a small head phantom from the reference materials (a water
cylinder with a bone insert, a vessel of contrast and a small amalgam filling), synthesize a short-scan cone-beam CT
of it with scatter and reconstruct it with CITRUS. The data are made once and cached in demo/input.

What to expect: the amalgam stops nearly every photon of the rays through it, so their counts are almost all scatter
and noise (photon starvation), and the rays that graze it disagree with MBIR's linear model. In the initial FDK volume
the filling throws dark streaks across the slice. CITRUS removes the scatter, and its metal mask
(``metal_attenuation_threshold``) gives zero weight to every ray through a voxel above the threshold, which clears the
streaks around the filling. The filling is filled from the initial volume (``metal_fill``); its attenuation comes out
far too low because the data cannot hold it (the counts of its rays are floored at one), so the RMSE is computed
outside it. Set the threshold to None to see the streaks.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from citrus import CitrusConfig, citrus_recon
from citrus.scatter_solver import (ConeGeometry, Grid, Material, ScatterSolver, SolverConfig,
                                   ideal_counting_response, load_spectrum_txt, no_grid)
from citrus.scatter_solver.cross_sections import material_index
from citrus.utilities import (load_dataset, plot_profiles, plot_slices, save_dataset, synthesize_cone_beam_data,
                              view_angles)

# ---- Parameters -----------------------------------------------------------------------------------------------
demo_dir = os.path.dirname(os.path.abspath(__file__))
spectrum = load_spectrum_txt(os.path.join(demo_dir, "spectra", "spectrum_125kvp.txt"))  # keV, photons per keV
# Iodine contrast agent: 10 mg of iodine per mL of water (1 % by mass at 1.01 g/cm^3), about the concentration in the
# arteries during a contrast-enhanced CT.
iodine_mg_per_ml = 10.0
contrast = Material("iodine_contrast", 1.0 + iodine_mg_per_ml / 1000,
                    f"{1 - iodine_mg_per_ml / 1010:.4f}*water + {iodine_mg_per_ml / 1010:.4f}*I")
# Dental amalgam: roughly equal masses of mercury and an alloy of 40-70 % Ag, 25-29 % Sn and 2-40 % Cu (Wikipedia,
# "Amalgam (dentistry)", Composition); here Hg 50 %, Ag 30 %, Sn 13 %, Cu 7 % by mass. The density is the ideal mixture
# of the element densities, 1 / sum(w_i / rho_i) = 11.0 g/cm^3 (an estimate; set amalgams are somewhat porous).
amalgam = Material("dental_amalgam", 11.0, "0.50*Hg + 0.30*Ag + 0.13*Sn + 0.07*Cu")
materials = ["air", "water", contrast, "bone_cortical_icrp", amalgam]  # catalogue names and user materials
head_radius = 9.0                          # radius in cm of the water cylinder (an 18 cm head)
bone_center, bone_radius = -5.0, 1.0       # bone rod along z at (x, 0) in cm
vessel_center, vessel_radius = 2.0, 0.5    # vessel of contrast along z at (x, 0) in cm
amalgam_center, amalgam_size = 6.0, 0.5    # amalgam cube at (x, 0, 0) in cm, edge in cm
delta_voxel = 0.25                         # voxel pitch in cm of the 20 x 20 x 12 cm grid
num_det_rows, num_det_channels = 52, 72    # detector panel size in pixels
pixel_pitch = 0.4                          # detector pixel pitch in cm
source_detector_dist, source_iso_dist = 150.0, 100.0  # source to detector and to rotation axis distances in cm
num_views = 180                            # number of views over a short scan
I0 = 1.0e6                                 # flat-field counts per pixel
num_outer_iterations = 20                  # outer iterations, one scatter correction each
mbir_iterations_per_outer_iteration = 8    # MBIR iterations per outer iteration
scatter_view_stride = 8                    # scatter step on every 8th view, the others interpolated in angle
metal_attenuation_threshold = 2.0          # 1/cm: voxels above are metal (cortical bone is about 0.5 1/cm here)

# ---- Scanner and solver: the solver lists the reference materials in library order (by attenuation) ---------------
geometry = ConeGeometry(source_iso_dist=source_iso_dist, source_detector_dist=source_detector_dist,
                        num_det_rows=num_det_rows, num_det_channels=num_det_channels,
                        delta_det_row=pixel_pitch, delta_det_channel=pixel_pitch)
grid = Grid(shape=(round(20 / delta_voxel), round(20 / delta_voxel), round(12 / delta_voxel)), delta_voxel=delta_voxel)
angles = view_angles(num_views, geometry)  # a short scan: 180 degrees plus the fan angle
config = SolverConfig(materials=tuple(materials))
make_solver = lambda: ScatterSolver(None, config, geometry, detector=ideal_counting_response(),  # noqa: E731
                                    grid_response=no_grid(), device="cuda", spectrum=spectrum)
solver = make_solver()
print("Reference materials in library order:", solver.materials)

# ---- Phantom: densities [materials, rows, cols, slices] in g/cm^3, one row per reference material -----------------
x, y, z = grid.axis_coordinates("cuda")
xx, yy, zz = x[None, :, None], y[:, None, None], z[None, None, :]
row = lambda material: material_index(solver.materials, material)  # noqa: E731
rho = torch.zeros(len(materials), *grid.shape, device="cuda")
inside = lambda center, radius: ((xx - center) ** 2 + yy ** 2 < radius ** 2).expand(*grid.shape)  # noqa: E731
rho[row("water")][inside(0.0, head_radius)] = 1.0
cube = ((xx - amalgam_center).abs() < amalgam_size / 2) & (yy.abs() < amalgam_size / 2) & (zz.abs() < amalgam_size / 2)
for material, density, mask in ((contrast, contrast.density, inside(vessel_center, vessel_radius)),
                                ("bone_cortical_icrp", 1.85, inside(bone_center, bone_radius)),
                                (amalgam, amalgam.density, cube)):
    rho[:, mask] = 0.0  # the insert replaces the water
    rho[row(material)][mask] = density

# ---- Data: loaded when a file of this name exists in demo/input, else synthesized once (about a minute on a GPU) ---
name = (f"head_{num_views}v_{delta_voxel:g}cm_{num_det_rows}x{num_det_channels}px{pixel_pitch:g}cm_I0{I0:.0e}_"
        f"iodine{iodine_mg_per_ml:g}_amalgam{amalgam_size:g}cm_125kvp").replace(".", "p").replace("+", "")
path = os.path.join(demo_dir, "input", name)
if os.path.exists(path + ".npz"):
    data = load_dataset(path)
else:
    data = synthesize_cone_beam_data(rho, grid, geometry, angles, make_solver(), I0=I0)
    save_dataset(data, path)
Y, X_true = data["Y"], data["X_true"]
print(f"Scatter-to-primary ratio of the data: median {np.median(data['SPR']):.2f}, max {data['SPR'].max():.3g}")

# ---- CITRUS -------------------------------------------------------------------------------------------------------
X_citrus, info = citrus_recon(Y, I0, angles, grid, geometry, solver,
                              CitrusConfig(num_outer_iterations=num_outer_iterations,
                                           mbir_iterations_per_outer_iteration=mbir_iterations_per_outer_iteration,
                                           scatter_view_stride=scatter_view_stride, verbose=True,
                                           metal_attenuation_threshold=metal_attenuation_threshold))

# ---- Results: the RMSE outside the filling, the images on a scale that saturates it -------------------------------
X_citrus, X_initial = X_citrus.cpu().numpy(), info["init_recon"].cpu().numpy()
outside = X_true < 0.5 * float(np.max(X_true))  # every voxel but the amalgam
relative_rmse = lambda X: np.linalg.norm((X - X_true)[outside]) / np.linalg.norm(X_true[outside])  # noqa: E731
print(f"Relative RMSE outside the amalgam: initial FDK volume {relative_rmse(X_initial):.4f}, CITRUS "
      f"{relative_rmse(X_citrus):.4f}; amalgam attenuation {X_true.max():.1f} 1/cm, CITRUS "
      f"{X_citrus[~outside].mean():.1f} 1/cm")
volumes = [("Truth", X_true), ("Initial FDK volume", X_initial), ("CITRUS", X_citrus)]
plot_slices(volumes, grid, vmax=0.6, title="Central slice z = 0 (gray scale saturated in bone and amalgam)")
plot_profiles(volumes, grid, title="Central row y = 0 (bone -6..-4 cm, contrast 1.5..2.5 cm, amalgam 5.75..6.25 cm)")
plt.gca().set_ylim(-0.1, 0.8)
plt.show()
