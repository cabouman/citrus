"""CITRUS reconstruction of a COBRA2026 Varian Halcyon case (centre F: full-fan CBCT of the pelvis, 125 kV).

The script runs in five sections: (1) the case and the parameters the metadata does not give, (2) the data, downloaded
if needed, and the parameters read from its metadata, (3) the preprocessing of the counts (drift, flat fields, couch),
(4) the CITRUS reconstruction and (5) the figure against the clinical CBCT and the deformed planning CT.

    python COBRA_Halcyon.py

It needs a CUDA GPU with about 12 GB and about 16 GB of RAM; a case is about 1.0 GB on disk.
"""

import os

import numpy as np
import torch

from cobra_tools import (CobraScan, blade_field_margin_cm, bowtie_transmission, citrus_to_lps, comparison_figure,
                         couch_attenuation, couch_line_integrals, download_case, initial_volume, insert_couch,
                         load_scan, read_case_info, read_volume, write_volume)

from citrus import CitrusConfig, Schedule, citrus_recon
from citrus.recon.mbir_bridge import make_cone_model
from citrus.scatter_solver import (Grid, ScatterSolver, SolverConfig, anti_scatter_grid, detector_response,
                                   load_spectrum_txt)
from citrus.utilities import effective_energy, mass_attenuation_at

HERE = os.path.dirname(os.path.abspath(__file__))

# ======================================================================================================================
# 1. Case and parameters
# ======================================================================================================================
CASE = "F015"   # F015, F003 or F041

# The parameters of each case that its metadata does not give.
CASES = {
    # couch_top_mm: y in mm of the IGRT couch top's upper surface (CITRUS y, anterior positive), read off the
    # couch-top line of the clinical CBCT. Scan.xml's CouchVrt is within about 1 mm of it: -161.5 mm (F015),
    # -145.5 mm (F003), -127.2 mm (F041). The lateral position is Scan.xml's CouchLat.
    "F015": {"couch_top_mm": -161.3},
    "F003": {"couch_top_mm": -146.6},
    "F041": {"couch_top_mm": -127.7},
}

# The scanner. Global drift of the RTI8643iL panel over the acquisition, the same on every pixel:
# y = (1 + B (1 - exp(-k / tau_g))) x + A (1 - exp(-k / tau_o)), k the frame index (every frame counted), x the
# counts without drift. Fitted once for the panel, shared by F015, F003 and F041, to their lines measured twice (the
# conjugate rays of each scan). A sits at the fit's bound of 300 counts, with tau_o much longer than the scan: the
# offset is close to a linear ramp of A / tau_o counts per frame, and A and tau_o are not determined separately.
DRIFT = {"gain_buildup": 0.041634,                 # B
         "gain_time_constant_frames": 16.466,      # tau_g
         "offset_buildup_counts": 300.0,           # A (at the fit's bound)
         "offset_time_constant_frames": 9062.8}    # tau_o
# Couch: Varian's couch-top contours (Calibrations/COUCH) filled with one attenuation for the carbon-fibre shell and
# one for the foam interior, the values fitted to the couch-only rays of the TrueBeam case G008 (the same IGRT couch
# top). The shell value is an effective one: its 4-5 mm wall is barely resolved.
COUCH_SHELL_PER_CM = 0.087
COUCH_INTERIOR_PER_CM = 0.011
# Spectrum: SpekPy 2.5.4, W anode, 14 deg, 125 kVp, with the TrueBeam STx filtration of the HVL study in Progress in
# Medical Physics (synapse.koreamed.org 1516089886): tube 2.70 mm Al, polycarbonate window 0.30 mm Al equivalent and
# the 0.89 mm titanium filter. The Halcyon uses the same spectrum: Varian's HU calibrations make its titanium filter
# (Titanium V2) equal to the TrueBeam's to 0.3 %. The bowtie (Bowtie V3, its thinnest part about 1.35 mm Al) is the
# source model's beam filter, its thickness per channel from the scan's own air pair. varian_125kvp_lit.txt
# (4.52 mm Al + 0.89 mm Ti) is the spectrum of the earlier runs, with the TrueBeam bowtie's minimum folded in.
SPECTRUM_FILE = os.path.join(HERE, "data", "spectra", "varian_125kvp_tube.txt")
# Anti-scatter grid: the calibration gives its ratio and lamella orientation only; lead strips of 36 um at 60 lines
# per cm with an air interspace are assumed. The strip height follows from the ratio.
GRID_SEPTUM_THICKNESS_CM = 0.0036
GRID_LINES_PER_CM = 60.0

# CITRUS settings
ROW_BIN, CHANNEL_BIN = 2, 6            # detector rows and channels per binned pixel
VIEW_STRIDE = 1                        # every view (about 400 over 210 degrees)
VOXEL_CM = 0.2                         # voxel of the loop, which is also the final image
GRID_WIDTH_CM = 60.0                   # lateral extent of the volume (at least the reference volume's)
GRID_LENGTH_CM = 27.8                  # extent along z
SCATTER_VIEW_STRIDE = 16               # the scatter is computed on every 16th view and interpolated in angle between
GROUPING_OBJECT_SIZE_CM = 30.0         # object size of the solver's energy groups
QUADRATURE_POINT_SPACING_CM = 1.0      # spacing of the scatter solver's quadrature points
CONE_POLAR_FACTOR = 2                  # the polar directions of the detector projection, times the default
SCATTER_SCENE_SUPPORT_HU = -750.0      # the scatter scene is the initial volume above this (0.25 x water), smoothed
SCATTER_SCENE_SUPPORT_MARGIN_CM = 2.0  # by 8 mm, dilated by this margin and fixed for the run
AIR_THRESHOLD_HU = -900.0              # voxels below this (0.1 x water) are air in the scatter scene
WEIGHT_CAP = 1e-6                      # cap on the MBIR weights Y / I0 (every ray above it gets the same weight)
OUTER_ITERATIONS, MBIR_ITERATIONS_PER_OUTER = 14, 8
HARDENING_MATERIALS = ("bone_cortical_icrp",)   # beam hardening beyond water's, corrected in the loop
DEVICE = "cuda"
RECONSTRUCT = True                     # False: only the figure, from the saved reconstruction
OUTPUT = os.path.join(HERE, "results", CASE)

# ======================================================================================================================
# 2. Data
# ======================================================================================================================
case_dir = download_case(CASE, os.path.join(HERE, "data", "Halcyon"))
info = read_case_info(case_dir)
print(f"[{CASE}] {info.kvp:g} kV, {info.tube_current_ma:g} mA, {info.pulse_ms:g} ms; {info.bowtie}; grid "
      f"{info.grid_ratio:g}:1 ({info.grid_orientation}); CsI {info.scintillator_cm * 10:g} mm; couch {info.couch_top} "
      f"at lateral {info.couch_lateral_mm:.1f} mm; blades {info.blades_mm} mm", flush=True)
os.makedirs(OUTPUT, exist_ok=True)
ct, origin, spacing = read_volume(os.path.join(case_dir, "ct_def_masked.mha"))   # deformed CT on the CBCT's grid
recon_path = os.path.join(OUTPUT, f"{CASE}_citrus_hu.mha")

if RECONSTRUCT:
    # ==================================================================================================================
    # 3. Preprocessing
    # ==================================================================================================================
    # the rows under the axial blades are left out; the off-focal emission past them is measured on the scan's own
    # air columns
    scan: CobraScan = load_scan(info, ROW_BIN, CHANNEL_BIN, VIEW_STRIDE, drift=DRIFT, off_focal_table=True)
    geometry = scan.geometry
    print(f"[{CASE}] {len(scan.angles)} views of {scan.counts.shape[1]} x {scan.counts.shape[2]} pixels, "
          f"{'half' if scan.half_fan else 'full'} fan", flush=True)
    # MBIR weights: 0 on dead pixels (a full fan has no redundancy weights)
    measurement_weights = None if scan.dead is None else (~scan.dead).astype(np.float32)
    off_focal_distance_cm, off_focal_fraction = scan.off_focal
    print(f"[{CASE}] off-focal emission past the lit rows: {off_focal_fraction[0]:.3f} of the lit level at the edge, "
          f"{off_focal_fraction[-1]:.3f} at {off_focal_distance_cm[-1]:.1f} cm", flush=True)
    # the bowtie's transmission per channel from the scan's own air pair, its thinnest part included
    bowtie, bowtie_source = bowtie_transmission(case_dir, CHANNEL_BIN)
    print(f"[{CASE}] bowtie transmission from {bowtie_source}: {bowtie.min():.3f} to {bowtie.max():.3f}", flush=True)
    field_margin_rows_cm = blade_field_margin_cm(info, geometry)   # the axial blades open past the panel's rows
    # the volume
    width_cm = max(GRID_WIDTH_CM, ct.shape[2] * spacing[0] / 10.0, ct.shape[1] * spacing[1] / 10.0)
    n_xy = int(round(width_cm / VOXEL_CM))
    grid = Grid(shape=(n_xy, n_xy, int(round(GRID_LENGTH_CM / VOXEL_CM))), delta_voxel=VOXEL_CM)
    # the couch: its model in the scatter scene (section 4), its line integrals removed from the counts
    couch = dict(case_dir=case_dir, name=info.couch_top, voxel_cm=VOXEL_CM, top_mm=CASES[CASE]["couch_top_mm"],
                 lateral_mm=info.couch_lateral_mm, shell_per_cm=COUCH_SHELL_PER_CM,
                 interior_per_cm=COUCH_INTERIOR_PER_CM)
    couch_per_cm, couch_mask = couch_attenuation(shape=grid.shape[:2], **couch)
    n_couch = int(np.ceil(60.0 / VOXEL_CM / 2)) * 2   # a grid wide enough for the whole couch, 1.5 x as long
    wide_couch, _ = couch_attenuation(shape=(n_couch, n_couch), **couch)
    couch_paths = couch_line_integrals(wide_couch, VOXEL_CM, int(round(grid.shape[2] * 1.5)), geometry, scan.angles,
                                       DEVICE)
    counts = scan.counts * np.exp(couch_paths)   # -ln(counts / I0) loses the couch's line integrals
    del couch_paths
    torch.cuda.empty_cache()

    # ==================================================================================================================
    # 4. Reconstruction
    # ==================================================================================================================
    default = SolverConfig()
    solver_config = SolverConfig(grouping_object_size_cm=GROUPING_OBJECT_SIZE_CM,
                                 quadrature_point_spacing_cm=QUADRATURE_POINT_SPACING_CM,
                                 cone_num_polar=CONE_POLAR_FACTOR * default.cone_num_polar)
    # TODO(release API): SolverConfig(group_averaged_responses=True)
    septa_along = "channels" if "transaxial" in info.grid_orientation.lower() else "rows"
    grid_response = anti_scatter_grid(geometry, GRID_SEPTUM_THICKNESS_CM,
                                      info.grid_ratio * (1.0 / GRID_LINES_PER_CM - GRID_SEPTUM_THICKNESS_CM),
                                      septum_material="lead", pitch_cm=1.0 / GRID_LINES_PER_CM, septa_along=septa_along)
    # TODO(release API): the analytic grid model is the release's default of anti_scatter_grid; confirm the
    # interspace (air) and whether the detector takes deposition="fluorescence_escape"
    solver = ScatterSolver(None, solver_config, geometry,
                           detector=detector_response("cesium_iodide", info.scintillator_cm, mode="energy_integrating"),
                           grid_response=grid_response, spectrum=load_spectrum_txt(SPECTRUM_FILE), device=DEVICE)
    # TODO(release API): the source model with the off-focal emission past the lit rows and the bowtie as a filter:
    #   thickness = filter_thickness_from_transmission(bowtie, solver, "aluminium")
    #   solver.source = SourceModel(field_margin_rows_cm=field_margin_rows_cm,
    #                               off_focal_distance_cm=off_focal_distance_cm, off_focal_fraction=off_focal_fraction,
    #                               filter_material="aluminium", filter_thickness_cm=thickness)
    # and check that citrus_recon keeps these fields when it sets the emission profile from the flat field.
    insert_couch(solver, couch_per_cm, couch_mask)
    model = make_cone_model(grid, geometry, scan.angles, device=DEVICE)
    initial = initial_volume(scan, model, grid, DEVICE)
    config = CitrusConfig(scatter_view_stride=SCATTER_VIEW_STRIDE, grouping_object_size=GROUPING_OBJECT_SIZE_CM,
                          verbose=True)
    # TODO(release API): pass the IGRT inputs to the loop, names to be confirmed:
    #   measurement_weights=measurement_weights, the cap WEIGHT_CAP on the MBIR weights Y / I0,
    #   scatter_scene_support_hu=SCATTER_SCENE_SUPPORT_HU,
    #   scatter_scene_support_margin_cm=SCATTER_SCENE_SUPPORT_MARGIN_CM,
    #   air_threshold_hu=AIR_THRESHOLD_HU, beam_hardening_correction (water linearization with the bowtie per channel),
    #   the hardening materials HARDENING_MATERIALS, and the scatter scene continued past the slab (edge slices).
    schedule = Schedule(num_outer_iterations=OUTER_ITERATIONS,
                        mbir_iterations_per_outer_iteration=MBIR_ITERATIONS_PER_OUTER)
    X, loop_info = citrus_recon(counts, scan.flat_field, scan.angles, grid, geometry, solver, config,
                                schedule=schedule, init_recon=initial, model=model)
    library = solver.library
    mu_water = float(mass_attenuation_at(library, effective_energy(library))[library.materials.index("water")])
    recon_hu = 1000.0 * (citrus_to_lps(X.cpu().numpy(), grid, ct.shape, origin, spacing) - mu_water) / mu_water
    write_volume(recon_hu, origin, spacing, recon_path)
    print(f"[{CASE}] saved {recon_path}", flush=True)

# ======================================================================================================================
# 5. Figure
# ======================================================================================================================
recon_hu = read_volume(recon_path)[0]
clinical = read_volume(os.path.join(case_dir, "cbct_clinical.mha"))[0]
fov = read_volume(os.path.join(case_dir, "fov_cbct_nocouch.mha"))[0]
figure_path = os.path.join(OUTPUT, f"{CASE}_comparison.png")
comparison_figure(recon_hu, clinical, ct, fov, spacing, figure_path, f"{CASE} (Halcyon)")
print(f"[{CASE}] figure {figure_path}", flush=True)
