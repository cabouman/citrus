"""Demo basic 3: scatter-corrected circular cone-beam CT with CITRUS.

Reconstruct a short-scan circular cone-beam CT of the WLAB phantom (a water prism with lung, air and bone inserts)
with CITRUS, which alternates scatter steps of the LBTE scatter solver with MBIR iterations (MBIRTORCH) on the
corrected counts. The measured data are synthetic: the counts of every view with the scatter of the solver and
Poisson noise, made once and cached in demo/input.

Every setting is a variable at the top. For your own scanner, set the spectrum, the panel, the grid and the view
angles, and replace the synthetic data by your measured counts [views, detector rows, detector channels] and flat
field I0.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from citrus import CitrusConfig, citrus_recon
from citrus.scatter_solver import (ConeGeometry, ScatterSolver, SolverConfig, anti_scatter_grid,
                                   ideal_counting_response, load_spectrum_txt, no_grid)
from citrus.utilities import (WlabPhantom, load_dataset, plot_profiles, plot_slices, save_dataset,
                              synthesize_cone_beam_data, wlab_grid)

# ---- Parameters -----------------------------------------------------------------------------------------------
# Source spectrum, (energies in keV, relative photons per keV), with a short name for the data file: a text file of
# two columns ('#' starts a comment); demo/spectra has a 125 kVp tube and a 6 MV linac spectrum.
demo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # demo/, with spectra/ and input/
spectrum, spectrum_name = load_spectrum_txt(os.path.join(demo_dir, "spectra", "spectrum_125kvp.txt")), "125kvp"
# spectrum, spectrum_name = load_spectrum_txt(os.path.join(demo_dir, "spectra", "spectrum_6mv.txt")), "6mv"
materials = ["air", "lung_icrp", "adipose_icrp", "water", "muscle_icru44", "soft_tissue_icru44",
             "bone_cortical_icrp"]  # reference materials of the density model
num_det_rows, num_det_channels = 184, 150  # detector panel size in pixels
pixel_pitch = 0.4                          # detector pixel pitch in cm
source_detector_dist = 150.0               # source to detector distance in cm
source_iso_dist = 100.0                    # source to rotation axis distance in cm
# Anti-scatter grid in front of the panel: none, or (uncomment) a lead grid focused at the source with 60 septa per
# cm, 36 um thick and 1.5 mm high, along the channels. The solvers of the data and of the reconstruction both use it.
grid_septa = None
# grid_septa = dict(septum_thickness_cm=0.0036, septum_height_cm=0.15, pitch_cm=1 / 60)
delta_voxel = 0.5                          # voxel pitch in cm of the WLAB phantom (a 30 x 20 x 40 cm water prism)
num_views = 180                            # number of views
start_angle = 0.0                          # first view angle in degrees
fan_angle = 2 * np.degrees(np.arctan(num_det_channels * pixel_pitch / 2 / source_detector_dist))
end_angle = start_angle + 180.0 + fan_angle  # end of the angular range in degrees (excluded): a short scan
I0 = 1.0e5                                 # flat-field counts per pixel
num_outer_iterations = 14                  # outer iterations, one scatter correction each
mbir_iterations_per_outer_iteration = 8    # MBIR iterations per outer iteration
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
config = SolverConfig(materials=tuple(materials), transport_cell_cm=transport_cell_cm, num_groups=num_groups,
                      legendre_order=legendre_order)
detector = ideal_counting_response()       # a photon-counting detector that counts every photon
grid_response = no_grid() if grid_septa is None else anti_scatter_grid(geometry, **grid_septa)

# ---- Data: loaded when a file of this name exists in demo/input, else synthesized once (about 2 minutes on a GPU) ---
name = (f"wlab_{num_views}v_{delta_voxel:g}cm_{start_angle:g}-{end_angle:.1f}deg_{num_det_rows}x{num_det_channels}px"
        f"{pixel_pitch:g}cm_{spectrum_name}{'' if grid_septa is None else '_grid'}").replace(".", "p")
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
print(f"Scatter-to-primary ratio of the data: mean {data['SPR'].mean():.2f}, max {data['SPR'].max():.2f}")

# ---- CITRUS: the initial volume is the FDK of the uncorrected counts ----------------------------------------------
solver = ScatterSolver(None, config, geometry, detector=detector, grid_response=grid_response, device="cuda",
                       spectrum=spectrum)
config_citrus = CitrusConfig(num_outer_iterations=num_outer_iterations, scatter_view_stride=scatter_view_stride,
                             mbir_iterations_per_outer_iteration=mbir_iterations_per_outer_iteration, verbose=True)
X_citrus, info = citrus_recon(Y, I0, angles, grid, geometry, solver, config_citrus)

# ---- Results ----------------------------------------------------------------------------------------------------
X_citrus, X_initial = X_citrus.cpu().numpy(), info["init_recon"].cpu().numpy()
relative_rmse = lambda X: np.linalg.norm(X - X_true) / np.linalg.norm(X_true)  # noqa: E731
print(f"Relative RMSE against the truth: initial FDK volume {relative_rmse(X_initial):.4f}, CITRUS "
      f"{relative_rmse(X_citrus):.4f} after {info['mbir_iterations_total']} MBIR iterations in "
      f"{len(info['records']) - 1} outer iterations ({info['seconds']:.0f} s)")
volumes = [("Truth", X_true), ("Initial FDK volume", X_initial), ("CITRUS", X_citrus)]
plot_slices(volumes, grid, title="Central slice z = 0", vmax=np.max(X_true))
plot_profiles(volumes, grid, title="Central row y = 0 (lung -8..-2 cm, air 2..4 cm, bone 6..8 cm)")
plt.show()
