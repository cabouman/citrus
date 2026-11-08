"""CITRUS reconstruction of a COBRA2026 Varian TrueBeam case (centre G: half-fan CBCT of the pelvis, 125 kV).

The script runs in five sections: (1) the case and the parameters the metadata does not give, (2) the data, downloaded
if needed, and the parameters read from its metadata, (3) the preprocessing of the counts (drift, flat fields, couch),
(4) the CITRUS reconstruction and (5) the figure against the clinical CBCT and the deformed planning CT.

    python COBRA_TrueBeam.py

It needs a CUDA GPU with 12 GB and about 25 GB of free RAM; a case is about 1.25 GB on disk.
"""

import os
import time

# the allocator settings of the runs, before torch starts CUDA
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.7,max_split_size_mb:256")

import numpy as np
import torch

from cobra_tools import (CobraScan, blade_field_margin_cm, bowtie_transmission, citrus_to_lps, comparison_figure,
                         couch_attenuation, download_case, half_fan_weights, initial_volume, load_scan, read_case_info,
                         read_volume, write_volume)

from citrus import CitrusConfig, Schedule, citrus_recon
from citrus.recon.mbir_bridge import make_cone_model
from citrus.scatter_solver import (Grid, ScatterSolver, SolverConfig, SourceModel, anti_scatter_grid,
                                   detector_response, filter_thickness_from_transmission, load_spectrum_txt)
from citrus.utilities import effective_energy, mass_attenuation_at

HERE = os.path.dirname(os.path.abspath(__file__))

# ======================================================================================================================
# 1. Case and parameters
# ======================================================================================================================
CASE = "G008"   # G008 or G007

# The parameters of each case that its metadata does not give.
CASES = {
    # couch_top_mm: y in mm of the IGRT couch top's upper surface (CITRUS y, anterior positive), fitted to the rays
    # through the couch alone at gantry 247-282 deg (a scan of the couch height and lateral position). Scan.xml's
    # CouchVrt reads -136.2 mm (G008) and -139.3 mm (G007). The lateral position is Scan.xml's CouchLat.
    "G008": {"couch_top_mm": -140.6},
    "G007": {"couch_top_mm": -143.0},
}

# The scanner. Global drift of the PaxScan 4030CB panel over the acquisition, the same on every pixel:
# y = (1 + B (1 - exp(-k / tau_g))) x + A (1 - exp(-k / tau_o)), k the frame index (every frame counted), x the
# counts without drift. Fitted once for the panel, shared by G008 and G007, to their lines measured twice (the
# conjugate rays of the half-fan overlap and the seam where the rotation closes).
DRIFT = {"gain_buildup": 0.035640,                 # B
         "gain_time_constant_frames": 176.71,      # tau_g
         "offset_buildup_counts": 13.427,          # A
         "offset_time_constant_frames": 163.06}    # tau_o
# Couch: Varian's couch-top contours (Calibrations/COUCH) filled with one attenuation for the carbon-fibre shell and
# one for the foam interior, fitted to the couch-only rays of G008 (G007 gives the same values). The shell value is
# an effective one: its 4-5 mm wall is barely resolved.
COUCH_SHELL_PER_CM = 0.087
COUCH_INTERIOR_PER_CM = 0.011
# Spectrum: SpekPy 2.5.4, W anode, 14 deg, 125 kVp, with the TrueBeam STx filtration of the HVL study in Progress in
# Medical Physics (synapse.koreamed.org 1516089886): tube 2.70 mm Al, polycarbonate window 0.30 mm Al equivalent and
# the 0.89 mm titanium filter. The half-fan bowtie (its thinnest part about 1.52 mm Al) is the source model's beam
# filter, its thickness per channel from the scan's own air pair. varian_125kvp_lit.txt (4.52 mm Al + 0.89 mm Ti)
# is the spectrum of the earlier runs, with the bowtie's minimum folded in.
SPECTRUM_FILE = os.path.join(HERE, "data", "spectra", "varian_125kvp_tube.txt")
# Anti-scatter grid: the calibration gives its ratio and lamella orientation only; lead strips of 36 um at 60 lines
# per cm with an air interspace are assumed. The strip height follows from the ratio.
GRID_SEPTUM_THICKNESS_CM = 0.0036
GRID_LINES_PER_CM = 60.0

# CITRUS settings
ROW_BIN, CHANNEL_BIN = 4, 2            # detector rows and channels per binned pixel
VIEW_STRIDE = 2                        # every 2nd view (a whole rotation has about 900)
VOXEL_CM = 0.125                       # voxel of the loop, which is also the final image
GRID_WIDTH_CM = 60.0                   # lateral extent of the volume (at least the reference volume's)
GRID_LENGTH_CM = 32.0                  # extent along z
SCATTER_VIEW_STRIDE = 16               # the scatter is computed on every 16th view and interpolated in angle between
GROUPING_OBJECT_SIZE_CM = 30.0         # object size of the solver's energy groups
QUADRATURE_POINT_SPACING_CM = 1.0      # spacing of the scatter solver's quadrature points
CONE_POLAR_FACTOR = 2                  # the polar directions of the detector projection, times the default
SCATTER_SCENE_SUPPORT_HU = -750.0      # the scatter scene is the current volume above this (0.25 x water), smoothed
SCATTER_SCENE_SUPPORT_MARGIN_CM = 2.0  # by 8 mm, dilated by this margin and cut to the field of view
AIR_THRESHOLD_HU = -900.0              # voxels below this (0.1 x water) are air in the scatter scene
WEIGHT_CAP = 1e-6                      # cap on the MBIR weights Y / I0 (every ray above it gets the same weight)
OUTER_ITERATIONS, MBIR_ITERATIONS_PER_OUTER = 14, 8
HARDENING_MATERIALS = ("bone_cortical_icrp",)   # beam hardening beyond water's, corrected in the loop
DEVICE = "cuda"
# share of the GPU's memory PyTorch may hold: with a cap its allocator frees cached blocks before it reaches it
# (garbage_collection_threshold above), without one Windows spills into host memory
GPU_MEMORY_FRACTION = 0.9
RECONSTRUCT = True                     # False: only the figure, from the saved reconstruction
OUTPUT = os.path.join(HERE, "results", CASE)

# ======================================================================================================================
# 2. Data
# ======================================================================================================================
case_dir = download_case(CASE, os.path.join(HERE, "data", "TrueBeam"))
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
    start = time.time()
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
    scan: CobraScan = load_scan(info, ROW_BIN, CHANNEL_BIN, VIEW_STRIDE, drift=DRIFT)
    geometry = scan.geometry
    print(f"[{CASE}] {len(scan.angles)} views of {scan.counts.shape[1]} x {scan.counts.shape[2]} pixels, "
          f"{'half' if scan.half_fan else 'full'} fan", flush=True)
    # MBIR weights: the half-fan overlap's redundancy weights per channel, 0 on dead pixels
    measurement_weights = half_fan_weights(geometry)
    if scan.dead is not None:
        measurement_weights = measurement_weights * ~scan.dead
    # the bowtie's transmission per channel from the scan's own air pair, its thinnest part included
    bowtie, bowtie_source = bowtie_transmission(case_dir, CHANNEL_BIN)
    print(f"[{CASE}] bowtie transmission from {bowtie_source}: {bowtie.min():.3f} to {bowtie.max():.3f}", flush=True)
    field_margin_rows_cm = blade_field_margin_cm(info, geometry)   # the axial blades open past the panel's rows
    # the volume: at least GRID_WIDTH_CM and the reference volume wide
    width_cm = max(GRID_WIDTH_CM, ct.shape[2] * spacing[0] / 10.0, ct.shape[1] * spacing[1] / 10.0)
    n_xy = int(round(width_cm / VOXEL_CM))
    grid = Grid(shape=(n_xy, n_xy, int(round(GRID_LENGTH_CM / VOXEL_CM))), delta_voxel=VOXEL_CM)
    # the couch, a known object: in the scatter scene and the modelled primary, its line integrals removed from the
    # corrected data
    couch_per_cm, couch_mask = couch_attenuation(case_dir, info.couch_top, grid.shape[:2], VOXEL_CM,
                                                 CASES[CASE]["couch_top_mm"], info.couch_lateral_mm,
                                                 COUCH_SHELL_PER_CM, COUCH_INTERIOR_PER_CM)
    if couch_mask[[0, -1]].any() or couch_mask[:, [0, -1]].any():
        raise ValueError("the couch reaches the edge of the volume: widen GRID_WIDTH_CM")
    couch = np.ascontiguousarray(np.broadcast_to(couch_per_cm[:, :, None], grid.shape), dtype=np.float32)

    # ==================================================================================================================
    # 4. Reconstruction
    # ==================================================================================================================
    detector = detector_response("cesium_iodide", info.scintillator_cm, deposition="fluorescence_escape")
    septa_along = "channels" if "transaxial" in info.grid_orientation.lower() else "rows"
    grid_response = anti_scatter_grid(geometry, GRID_SEPTUM_THICKNESS_CM, ratio=info.grid_ratio,
                                      lines_per_cm=GRID_LINES_PER_CM, interspace_material="air", detector=detector,
                                      septa_along=septa_along)
    solver_config = SolverConfig(grouping_object_size_cm=GROUPING_OBJECT_SIZE_CM,
                                 quadrature_point_spacing_cm=QUADRATURE_POINT_SPACING_CM,
                                 cone_num_polar=CONE_POLAR_FACTOR * SolverConfig().cone_num_polar,
                                 group_averaged_responses=True)
    solver = ScatterSolver(None, solver_config, geometry, detector=detector, grid_response=grid_response,
                           spectrum=load_spectrum_txt(SPECTRUM_FILE), device=DEVICE)
    # the source: the bowtie as an aluminium beam filter, its thickness per channel from the measured transmission
    thickness = filter_thickness_from_transmission(bowtie, solver, "aluminium")
    print(f"[{CASE}] bowtie: {10 * float(thickness.min()):.2f} to {10 * float(thickness.max()):.2f} mm Al", flush=True)
    solver.source = SourceModel(field_margin_rows_cm=field_margin_rows_cm, filter_material="aluminium",
                                filter_thickness_cm=thickness)
    model = make_cone_model(grid, geometry, scan.angles, device=DEVICE)
    initial = initial_volume(scan, model, grid, DEVICE)
    config = CitrusConfig(scatter_view_stride=SCATTER_VIEW_STRIDE, grouping_object_size=GROUPING_OBJECT_SIZE_CM,
                          extend_scatter_scene=True, beam_hardening_correction=True,
                          hardening_materials=HARDENING_MATERIALS, measurement_weights=measurement_weights,
                          max_measurement_weight=WEIGHT_CAP, scatter_scene_support_hu=SCATTER_SCENE_SUPPORT_HU,
                          scatter_scene_support_margin_cm=SCATTER_SCENE_SUPPORT_MARGIN_CM,
                          air_threshold_hu=AIR_THRESHOLD_HU, known_objects=couch, verbose=True)
    schedule = Schedule(num_outer_iterations=OUTER_ITERATIONS,
                        mbir_iterations_per_outer_iteration=MBIR_ITERATIONS_PER_OUTER)
    X, loop_info = citrus_recon(scan.counts, scan.flat_field, scan.angles, grid, geometry, solver, config,
                                schedule=schedule, init_recon=initial, model=model)
    library = solver.library
    water = library.materials.index("water")
    mu_water = float(mass_attenuation_at(library, effective_energy(library))[water] * library.nominal_density[water])
    recon_hu = 1000.0 * (citrus_to_lps(X.cpu().numpy(), grid, ct.shape, origin, spacing) - mu_water) / mu_water
    write_volume(recon_hu, origin, spacing, recon_path)
    peak = max([r.get("gpu_peak_allocated_mib", 0.0) for r in loop_info["records"]] + [0.0]) / 1024
    print(f"[{CASE}] saved {recon_path}: {time.time() - start:.0f} s, GPU peak {peak:.1f} GB, "
          f"scatter-to-primary ratio mean {float(loop_info['spr'].mean()):.3f}", flush=True)

# ======================================================================================================================
# 5. Figure
# ======================================================================================================================
recon_hu = read_volume(recon_path)[0]
clinical = read_volume(os.path.join(case_dir, "cbct_clinical.mha"))[0]
fov = read_volume(os.path.join(case_dir, "fov_cbct_nocouch.mha"))[0]
figure_path = os.path.join(OUTPUT, f"{CASE}_comparison.png")
comparison_figure(recon_hu, clinical, ct, fov, spacing, figure_path, f"{CASE} (TrueBeam)")
print(f"[{CASE}] figure {figure_path}", flush=True)
