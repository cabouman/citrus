"""Demo 2: scatter-corrected cone-beam reconstruction with CITRUS.

Reconstruct a short-scan cone-beam CT of the WLAB phantom with CITRUS, which alternates scatter steps of the LBTE
scatter solver with MBIR iterations (MBIRTORCH) on the corrected counts, and compare it with the same MBIR
iterations without scatter correction. The measured data are synthetic: the counts of every view with the scatter of
the solver and Poisson noise, made once and cached in demo/input.

Every setting is a variable at the top. For your own scanner, set the spectrum, the panel and the view angles, and
replace the synthetic data by your measured counts [views, detector rows, detector channels] and flat field I0.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from citrus import CitrusConfig, citrus_recon, mbir_recon
from citrus.scatter_solver import (ConeGeometry, ScatterSolver, SolverConfig, ideal_counting_response,
                                   load_spectrum_txt, no_grid)
from citrus.utilities import (WlabPhantom, load_dataset, plot_profiles, plot_slices, save_dataset,
                              synthesize_cone_beam_data, wlab_grid)

# ---- Parameters -----------------------------------------------------------------------------------------------
# Source spectrum: the path of a text file of two columns, photon energy in keV and relative photons per keV ('#'
# starts a comment).  demo/spectra has a 125 kVp tube spectrum and a 6 MV linac spectrum (spectrum_6mv.txt).
demo_dir = os.path.dirname(os.path.abspath(__file__))
spectrum_file = os.path.join(demo_dir, "spectra", "spectrum_125kvp.txt")
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
num_outer_iterations = 80                  # CITRUS outer iterations (MBIR iterations)
scatter_every = 8                          # scatter step at every 8th outer iteration
scatter_view_stride = 8                    # scatter step on every 8th view, the others interpolated in angle
transport_cell_cm = 1.0                    # transport cell size in cm
num_groups = 6                             # energy groups
legendre_order = 7                         # Legendre order of the scattering

# ---- Scanner and solver ---------------------------------------------------------------------------------------
geometry = ConeGeometry(source_iso_dist=source_iso_dist, source_detector_dist=source_detector_dist,
                        num_det_rows=num_det_rows, num_det_channels=num_det_channels,
                        delta_det_row=pixel_pitch, delta_det_channel=pixel_pitch)
grid = wlab_grid(delta_voxel)
angles = np.radians(np.linspace(start_angle, end_angle, num_views, endpoint=False))
spectrum = load_spectrum_txt(spectrum_file)
config = SolverConfig(materials=tuple(materials), transport_cell_cm=transport_cell_cm, num_groups=num_groups,
                      legendre_order=legendre_order)
detector, grid_response = ideal_counting_response(), no_grid()  # photon-counting detector, no anti-scatter grid

# ---- Data: load them, or synthesize them once (about 2 minutes on a GPU) ---------------------------------------
# The file name records the views, the voxel pitch, the angles and the spectrum: delete the file after changing others.
name = (f"wlab_{num_views}v_{delta_voxel:g}cm_{start_angle:g}-{end_angle:.1f}deg_"
        f"{os.path.splitext(os.path.basename(spectrum_file))[0]}").replace(".", "p")
path = os.path.join(demo_dir, "input", name)
if os.path.exists(path + ".npz"):
    data = load_dataset(path)
else:
    truth_solver = ScatterSolver(None, config, geometry, detector=detector, grid_response=grid_response,
                                 device="cuda", spectrum=spectrum)
    phantom = WlabPhantom(grid, truth_solver.materials, device="cuda")
    data = synthesize_cone_beam_data(phantom, grid, geometry, angles, truth_solver, I0=I0)
    save_dataset(data, path)
Y, X_true = data["Y"], data["X_true"]  # counts [views, rows, channels], true attenuation [rows, cols, slices] in 1/cm

# ---- CITRUS, and MBIR without scatter correction from the same start and regularization ------------------------
# The solver's library is built (and cached on disk) on first use, with energy groups fitted to the FDK volume.
solver = ScatterSolver(None, config, geometry, detector=detector, grid_response=grid_response, device="cuda",
                       spectrum=spectrum)
config_citrus = CitrusConfig(num_outer_iterations=num_outer_iterations, scatter_every=scatter_every,
                             scatter_view_stride=scatter_view_stride, verbose=True)
X_citrus, info = citrus_recon(Y, I0, angles, grid, geometry, solver, config_citrus)
X_uncorrected, _ = mbir_recon(Y, I0, angles, grid, geometry, config_citrus, regularization=info["regularization"],
                              init_recon=info["init_recon"], model=info["model"])

# ---- Results ----------------------------------------------------------------------------------------------------
X_citrus, X_uncorrected = X_citrus.cpu().numpy(), X_uncorrected.cpu().numpy()
for label, X in (("no scatter correction", X_uncorrected), ("CITRUS", X_citrus)):
    print(f"Relative RMSE, {label}: {np.linalg.norm(X - X_true) / np.linalg.norm(X_true):.4f}")
volumes = [("Truth", X_true), ("MBIR, no scatter correction", X_uncorrected), ("CITRUS", X_citrus)]
plot_slices(volumes, grid, title="Central slice z = 0")
plot_profiles(volumes, grid, title="Central row y = 0 (lung -8..-2 cm, air 2..4 cm, bone 6..8 cm)")
plt.show()
