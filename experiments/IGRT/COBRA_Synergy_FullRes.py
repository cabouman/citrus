"""CITRUS reconstruction of a COBRA2026 Elekta Synergy case (centre C: XVI half-fan CBCT of the pelvis, 120 kV, M20
collimator, F1 bowtie).

The script runs in five sections: (1) the case and the parameters the metadata does not give, (2) the data, downloaded
if needed, and the parameters read from its metadata, (3) the preprocessing of the counts (drift, couch), (4) the
CITRUS reconstruction and (5) the figure against the clinical CBCT and the deformed planning CT.

    python COBRA_Synergy_FullRes.py

This is the full-resolution version: every view and pixel (as loaded below) on the clinical CBCT's voxels, through
multigrid stages from 5 mm voxels. COBRA_Synergy_LowRes.py is the same reconstruction on binned data.

It ran on an 80 GB H100 (the GPU peak is about 54 GB on the TrueBeam) and needs well over 25 GB of RAM; a case
is about 1 GB on disk.
"""

import math
import os
import time

# the allocator settings of the runs, before torch starts CUDA
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.7,max_split_size_mb:256")

import numpy as np
import torch

from cobra_tools import (CobraScan, LongSideFieldSource, citrus_to_lps, comparison_figure, download_case,
                         field_of_view_region, half_fan_weights, ibeam_evo_couch, initial_volume, load_elekta_scan,
                         parabolic_bowtie_cm, read_elekta_info, read_volume, write_volume)

from citrus import CitrusConfig, Schedule, Stage, citrus_recon
from citrus.recon.mbir_bridge import make_cone_model
from citrus.scatter_solver import Grid, ScatterSolver, SolverConfig, detector_response, load_spectrum_txt, no_grid
from citrus.utilities import effective_energy, mass_attenuation_at

HERE = os.path.dirname(os.path.abspath(__file__))

# ======================================================================================================================
# 1. Case and parameters
# ======================================================================================================================
CASE = "C039"   # C039, C041 or C013

# The parameters of each case that its metadata does not give.
CASES = {
    # couch_top_mm: y in mm of the couch top's upper surface (CITRUS y, anterior positive): the upper of the two
    # carbon skins, 5 cm apart, under the patient in a reconstruction without a couch model, plus half the skin.
    "C039": {"couch_top_mm": -117.7},
    "C041": {"couch_top_mm": -115.2},
    "C013": {"couch_top_mm": -106.5},
}

# The scanner. Global drift of the XVI panel over the acquisition, the same on every pixel:
# y = (1 + B (1 - exp(-k / tau_g))) x + A (1 - exp(-k / tau_o)), k the frame index, x the counts without drift.
# Fitted once for the panel, shared by C013, C039 and C041, to their lines measured twice (the conjugate rays of the
# half-fan overlap and the seam where the rotation closes), on the rows clear of the cone's corners, with the
# scatter-to-primary ratios of this script.
DRIFT = {"gain_buildup": 0.067035,             # B
         "gain_time_constant_frames": 88.213,  # tau_g
         "offset_buildup_counts": 13.881,      # A
         "offset_time_constant_frames": 102.64}  # tau_o
# Panel: CsI, 0.6 mm assumed. No anti-scatter grid (metadata: ScatterGrid None).
SCINTILLATOR_CM = 0.06
# Spectrum: SpekPy 2.5.4, W anode, 14 deg, 120 kVp, with the open-beam filtration whose first HVL is 7.04 mm Al, the
# value measured on an XVI at 120 kV without the bowtie (PENELOPE model of the XVI; 8.04 mm Al with F1).
SPECTRUM_FILE = os.path.join(HERE, "data", "spectra", "elekta_120kvp_open.txt")
# F1 bowtie: its profile is not released. An aluminium bowtie whose thickness rises as the square of the fan angle,
# symmetric about the central ray: 3.32 mm on the central ray, the thickness that raises the HVL from 7.04 to 8.04 mm
# Al, and 30 mm at the far edge of the panel's long side (Downes et al., Med. Phys. 36, 2009: 2 mm of Al 2017 at the
# centre, 30 mm at the cassette's edge).
F1_CENTRE_MM, F1_EDGE_MM = 3.32, 30.0
F1_POWER = 2.0
# Field: the projections do not show the field's edges (every pixel that sees air is clipped). The literature puts the
# M20 field 0-1 cm past the panel's rows (the XVI's field length at the isocentre) and 1.9 cm past its short edge
# (Downes et al.); in cm at the panel.
FIELD_MARGIN_ROWS_CM, FIELD_MARGIN_CHANNELS_CM, FIELD_MARGIN_LONG_SIDE_CM = 0.75, 1.9, 0.0
# Couch: the iBEAM evo top (cross-section from Med. Phys. 37, 2010), its carbon and foam attenuation and its
# thickness fitted to the rays through the couch alone in the lateral views of C013, C039 and C041.
COUCH_CARBON_PER_CM, COUCH_FOAM_PER_CM, COUCH_THICKNESS_CM = 0.25, 0.02, 5.05

# CITRUS settings
ROW_BIN, CHANNEL_BIN = 1, 1            # detector rows and channels per binned pixel (none: 0.8 mm)
VIEW_STRIDE = 1                        # every view
VOXEL_CM = 0.1                         # voxel of the last stage, which is also the final image
GRID_WIDTH_CM = 60.0                   # lateral extent of the volume (at least the reference volume's)
GRID_LENGTH_CM = 32.0                  # extent along z
SCATTER_VIEW_STRIDE = 16               # the scatter is computed on every 16th view and interpolated in angle between
GROUPING_OBJECT_SIZE_CM = 30.0         # object size of the solver's energy groups
QUADRATURE_POINT_SPACING_CM = 1.0      # spacing of the scatter solver's quadrature points
CONE_POLAR_FACTOR = 2                  # the polar directions of the detector projection, times the default
AIR_THRESHOLD = 0.02                   # voxels below this attenuation in 1/cm (a tenth of water's) are air in the
                                       # scatter scene
MBIR_ITERATIONS_PER_OUTER = 8          # MBIR iterations per outer iteration (one scatter update each)
# MBIRTORCH's prior on every stage: sharper than its defaults (sharpness 1, T 1), for the detail at full resolution
MBIR_SHARPNESS, MBIR_T = 1.25, 0.25
# Multigrid stages, coarse to fine: (voxel in cm (None: VOXEL_CM), detector rows averaged, channels averaged, view
# stride, outer iterations), the binning and stride on the loaded data. Each stage starts from the previous stage's
# volume and scatter state; the last stage is the loaded data on the final grid.
STAGES = ((0.5, 8, 8, 4, 6), (0.25, 4, 4, 2, 3), (0.125, 2, 2, 1, 1), (None, 1, 1, 1, 1))
HARDENING_MATERIALS = ("bone_cortical_icrp",)   # beam hardening beyond water's, corrected in the loop
DEVICE = "cuda"
# share of the GPU's memory PyTorch may hold: with a cap its allocator frees cached blocks before it reaches it
# (garbage_collection_threshold above), without one Windows spills into host memory
GPU_MEMORY_FRACTION = 0.9
RECONSTRUCT = True                     # False: only the figure, from the saved reconstruction
OUTPUT = os.path.join(HERE, "results", f"{CASE}_FullRes")

# ======================================================================================================================
# 2. Data
# ======================================================================================================================
case_dir = download_case(CASE, os.path.join(HERE, "data", "Elekta"))
info = read_elekta_info(case_dir)
print(f"[{CASE}] {info.kvp:g} kV, {info.tube_current_ma:g} mA, {info.pulse_ms:g} ms; filter {info.kv_filter}; "
      f"collimator {info.collimator}; open beam {info.flat_field:.0f}", flush=True)
os.makedirs(OUTPUT, exist_ok=True)
ct, origin, spacing = read_volume(os.path.join(case_dir, "ct_def_masked.mha"))   # deformed CT on the CBCT's grid
recon_path = os.path.join(OUTPUT, f"{CASE}_citrus_hu.mha")

if RECONSTRUCT:
    # ==================================================================================================================
    # 3. Preprocessing
    # ==================================================================================================================
    start = time.time()
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
    scan: CobraScan = load_elekta_scan(info, ROW_BIN, CHANNEL_BIN, VIEW_STRIDE, drift=DRIFT)
    geometry = scan.geometry
    print(f"[{CASE}] {len(scan.angles)} views of {scan.counts.shape[1]} x {scan.counts.shape[2]} pixels, "
          f"{'half' if scan.half_fan else 'full'} fan", flush=True)
    # MBIR weights: the half-fan overlap's redundancy weights per channel, 0 on dead pixels. The pixels clipped at the
    # panel's maximum keep their weight as air (the open beam): they are the only data that keep the air around the
    # body empty
    measurement_weights = half_fan_weights(geometry)
    if scan.dead is not None:
        measurement_weights = measurement_weights * ~scan.dead
    # the volume: at least GRID_WIDTH_CM and the reference volume wide, in whole cm, with floor(extent / voxel) voxels
    # per axis, so that every stage's grid has the same 1 cm transport cells (they carry the scatter state)
    width_cm = math.ceil(max(GRID_WIDTH_CM, ct.shape[2] * spacing[0] / 10.0, ct.shape[1] * spacing[1] / 10.0))
    n_xy = int(width_cm / VOXEL_CM + 1e-6)
    grid = Grid(shape=(n_xy, n_xy, int(GRID_LENGTH_CM / VOXEL_CM + 1e-6)), delta_voxel=VOXEL_CM)
    # the couch, a known object: in the scatter scene and the modelled primary, its line integrals removed from the
    # corrected data
    couch_per_cm, couch_mask = ibeam_evo_couch(grid.shape[:2], VOXEL_CM, CASES[CASE]["couch_top_mm"] / 10.0,
                                               COUCH_CARBON_PER_CM, COUCH_FOAM_PER_CM, COUCH_THICKNESS_CM)
    if couch_mask[[0, -1]].any() or couch_mask[:, [0, -1]].any():
        raise ValueError("the couch reaches the edge of the volume: widen GRID_WIDTH_CM")
    couch = np.ascontiguousarray(np.broadcast_to(couch_per_cm[:, :, None], grid.shape), dtype=np.float32)

    # ==================================================================================================================
    # 4. Reconstruction
    # ==================================================================================================================
    detector = detector_response("cesium_iodide", SCINTILLATOR_CM, deposition="fluorescence_escape")
    solver_config = SolverConfig(grouping_object_size_cm=GROUPING_OBJECT_SIZE_CM,
                                 quadrature_point_spacing_cm=QUADRATURE_POINT_SPACING_CM,
                                 cone_num_polar=CONE_POLAR_FACTOR * SolverConfig().cone_num_polar,
                                 group_averaged_responses=True)
    solver = ScatterSolver(None, solver_config, geometry, detector=detector, grid_response=no_grid(),
                           spectrum=load_spectrum_txt(SPECTRUM_FILE), device=DEVICE)
    # the source: the F1 bowtie as an aluminium beam filter (the flood correction removes it from the primary only)
    thickness = parabolic_bowtie_cm(geometry, F1_CENTRE_MM, F1_EDGE_MM, F1_POWER)
    print(f"[{CASE}] F1 bowtie: {10 * float(thickness.min()):.2f} to {10 * float(thickness.max()):.2f} mm Al",
          flush=True)
    solver.source = LongSideFieldSource(filter_material="aluminium", filter_thickness_cm=thickness,
                                        field_margin_rows_cm=FIELD_MARGIN_ROWS_CM,
                                        field_margin_channels_cm=FIELD_MARGIN_CHANNELS_CM,
                                        field_margin_long_side_cm=FIELD_MARGIN_LONG_SIDE_CM)
    model = make_cone_model(grid, geometry, scan.angles, device=DEVICE)
    model.set_params(no_warning=True, sharpness=MBIR_SHARPNESS, T=MBIR_T)
    # MBIR reconstructs the field of view and the couch; the start is 0 elsewhere, since the scatter solver sees the
    # whole volume and an FDK leaves streaks outside the field of view
    region = field_of_view_region(grid, geometry, couch_mask)
    model.set_params(use_ror_mask=region)
    initial = initial_volume(scan, model, grid, DEVICE) * torch.as_tensor(region, device=DEVICE)[:, :, None]
    config = CitrusConfig(scatter_view_stride=SCATTER_VIEW_STRIDE, grouping_object_size=GROUPING_OBJECT_SIZE_CM,
                          extend_scatter_scene=True, beam_hardening_correction=True,
                          hardening_materials=HARDENING_MATERIALS, measurement_weights=measurement_weights,
                          air_threshold=AIR_THRESHOLD, known_objects=couch,
                          verbose=True)
    stages = tuple(Stage(voxel_factor=(VOXEL_CM if voxel is None else voxel) / VOXEL_CM, row_binning=rows,
                         channel_binning=channels, view_stride=views, num_outer_iterations=outer)
                   for voxel, rows, channels, views, outer in STAGES)
    schedule = Schedule(mbir_iterations_per_outer_iteration=MBIR_ITERATIONS_PER_OUTER, stages=stages)
    X, loop_info = citrus_recon(scan.counts, info.flat_field, scan.angles, grid, geometry, solver, config,
                                schedule=schedule, init_recon=initial, model=model)
    library = solver.library
    water = library.materials.index("water")
    mu_water = float(mass_attenuation_at(library, effective_energy(library))[water] * library.nominal_density[water])
    recon_hu = 1000.0 * (citrus_to_lps(X.cpu().numpy(), grid, ct.shape, origin, spacing) - mu_water) / mu_water
    write_volume(recon_hu, origin, spacing, recon_path)
    for stage in loop_info["stages"]:
        seconds = sum(r.get("scatter_seconds", 0.0) + r.get("mbir_seconds", 0.0) for r in stage["records"])
        print(f"[{CASE}] stage {stage['stage']}: {10 * stage['grid'].delta_voxel:.3g} mm voxels, {stage['num_views']} "
              f"views of {stage['geometry'].num_det_rows} x {stage['geometry'].num_det_channels} pixels, scatter and "
              f"MBIR {seconds:.0f} s, scatter state carried {stage.get('state_carried')}", flush=True)
    peak = max([r.get("gpu_peak_allocated_mib", 0.0) for stage in loop_info["stages"] for r in stage["records"]]
               + [0.0]) / 1024
    print(f"[{CASE}] saved {recon_path}: {time.time() - start:.0f} s, GPU peak {peak:.1f} GB, "
          f"scatter-to-primary ratio mean {float(loop_info['spr'].mean()):.3f}", flush=True)

# ======================================================================================================================
# 5. Figure
# ======================================================================================================================
recon_hu = read_volume(recon_path)[0]
# The XVI's clinical CBCT is not in HU (no HU calibration: watervaluemeasured 0): for centre C one linear map, the
# line through the air and muscle peaks of its cases matched to the deformed CTs' peaks
clinical = 1.3897 * read_volume(os.path.join(case_dir, "cbct_clinical.mha"))[0] + 426.4
fov = read_volume(os.path.join(case_dir, "fov_cbct_nocouch.mha"))[0]
figure_path = os.path.join(OUTPUT, f"{CASE}_comparison.png")
comparison_figure(recon_hu, clinical, ct, fov, spacing, figure_path, f"{CASE} (Elekta Synergy, full resolution)")
print(f"[{CASE}] figure {figure_path}", flush=True)
