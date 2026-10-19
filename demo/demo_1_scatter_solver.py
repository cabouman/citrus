"""Demo 1: the scatter solver on one view.

Build the deterministic scatter solver for the WLAB phantom (a water prism with lung, air and bone inserts), solve
one cone-beam view, and show the primary signal, the scatter signal and the scatter fraction on the detector.

The solver works on material densities: in a real application you would replace the phantom by your own
densities rho [materials, rows, cols, slices] in g/cm^3 on a grid that covers the object.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from citrus.scatter_solver import (ConeGeometry, ScatterSolver, SolverConfig, build_solver_library, csi_response,
                                   load_spectrum_txt, no_grid)
from citrus.utilities import wlab_grid, wlab_phantom

# ---- Parameters -----------------------------------------------------------------------------------------------
# Source spectrum: the path of a text file of two columns, photon energy in keV and relative photons per keV.
# demo/spectra has a 125 kVp tube spectrum and a 6 MV linac spectrum (spectrum_6mv.txt).
spectrum_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spectra", "spectrum_125kvp.txt")
materials = ["air", "lung_icrp", "adipose_icrp", "water", "muscle_icru44", "soft_tissue_icru44",
             "bone_cortical_icrp"]  # reference materials of the density model
num_det_rows, num_det_channels = 75, 100   # detector panel size in pixels
pixel_pitch = 0.4                          # detector pixel pitch in cm
source_detector_dist = 150.0               # source to detector distance in cm
source_iso_dist = 100.0                    # source to rotation axis distance in cm
delta_voxel = 1.0                          # voxel pitch in cm of the phantom
angle_degrees = 0.0                        # view angle in degrees; 0 is the anterior-posterior view
device = "cuda" if torch.cuda.is_available() else "cpu"

# The solver settings: the reference materials, the rest at its default (6 energy groups, 1 cm transport cells, ...).
config = SolverConfig(materials=tuple(materials))

# The multigroup cross-section library of the reference materials for this spectrum.  On first use this downloads
# the NIST XCOM photon data and builds the library; both are cached on disk, so later runs start in seconds.
library = build_solver_library(config, spectrum=load_spectrum_txt(spectrum_file))

# The scanner: a cone-beam geometry and a flat panel.
geometry = ConeGeometry(source_iso_dist=source_iso_dist, source_detector_dist=source_detector_dist,
                        num_det_rows=num_det_rows, num_det_channels=num_det_channels,
                        delta_det_row=pixel_pitch, delta_det_channel=pixel_pitch)

# The object: the WLAB phantom, as one density map per reference material of the library.
grid = wlab_grid(delta_voxel)
rho = wlab_phantom(grid, materials=library.materials, device=device)

# The solver: the library, the geometry, a CsI detector 0.6 mm thick and no anti-scatter grid.
solver = ScatterSolver(library, config, geometry, detector=csi_response(0.06), grid_response=no_grid(),
                       device=device)

# Solve one view.  The solver takes the view angle in radians; signal_only skips the scatter images per energy group,
# since this demo shows the detector signals only.
result = solver.solve_view(rho, grid, angle=np.radians(angle_degrees), signal_only=True)

# The detector signals as numpy images [detector rows, detector channels], in keV/cm^2 per source photon per sr.
primary = result.primary["signal"].float().cpu().numpy()
scatter = result.scatter_signal.float().cpu().numpy()
scatter_fraction = result.scatter_fraction.float().cpu().numpy()

print(f"Primary signal summed over the panel: {primary.sum():.4g}")
print(f"Scatter signal summed over the panel: {scatter.sum():.4g}")
print(f"Mean scatter fraction S / (P + S): {scatter_fraction.mean():.3f}")

# Show the three images.  The extent is the panel in cm.
half_u, half_v = num_det_channels * pixel_pitch / 2, num_det_rows * pixel_pitch / 2
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, image, title, cmap in ((axes[0], primary, "Primary signal", "gray"),
                               (axes[1], scatter, "Scatter signal", "magma"),
                               (axes[2], scatter_fraction, "Scatter fraction", "viridis")):
    artist = ax.imshow(image, origin="lower", cmap=cmap, extent=[-half_u, half_u, -half_v, half_v])
    ax.set_title(title)
    ax.set_xlabel("u (cm)")
    ax.set_ylabel("v (cm)")
    fig.colorbar(artist, ax=ax, shrink=0.8)
fig.suptitle(f"WLAB phantom, view at {angle_degrees:g} degrees")
fig.tight_layout()
plt.show()
