"""CITRUS reconstruction of a COBRA2026 Varian TrueBeam case (centre G: half-fan CBCT of the pelvis, 125 kV).

The script runs in five sections: (1) the case and the parameters the metadata does not give, (2) the data, downloaded
if needed, and the parameters read from its metadata, (3) the preprocessing of the counts (drift, open beam, couch),
(4) the CITRUS reconstruction and (5) the figure against the clinical CBCT and the deformed planning CT.

    python COBRA_TrueBeam_LowRes.py

This is the low-resolution version: binned data, through multigrid stages up to the voxel below.
COBRA_TrueBeam_FullRes.py is the same reconstruction at the panel's full resolution on the clinical CBCT's voxels.

It needs a CUDA GPU with 12 GB and about 25 GB of free RAM; a case is about 1.25 GB on disk.
"""

import math
import os
import time

# the allocator settings of the runs, before torch starts CUDA
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.7,max_split_size_mb:256")

import numpy as np
import torch

from cobra_tools import (CobraScan, air_row_profile, blade_field_margin_cm, bowtie_transmission, citrus_to_lps,
                         comparison_figure, couch_attenuation, download_case, field_of_view_region, half_fan_weights,
                         initial_volume, load_scan, read_case_info, read_volume, write_volume)

from citrus import CitrusConfig, Schedule, Stage, citrus_recon
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
    # open_beam_scale: the scan's open beam over the air scans scaled by the mAs (0.1 mAs to the scan's 1.2 mAs),
    # from the air the scan itself measures: in every frame with air beside the patient, the brightest 1% of the
    # transmission (the pixels chosen in the neighbouring frames, so that their noise does not choose them), fitted
    # together with the drift below.
    "G008": {"couch_top_mm": -140.6, "open_beam_scale": 0.7812},
    "G007": {"couch_top_mm": -143.0, "open_beam_scale": 0.7507},
}

# The scanner. Global drift of the PaxScan 4030CB panel over the acquisition, the same on every pixel:
# y = (1 + B (1 - exp(-k / tau_g)) + B_f (1 - exp(-k / tau_f))) x + A (1 - exp(-k / tau_o)), k the frame index
# (every frame counted), x the counts without drift. Fitted once for the panel, shared by G008 and G007, to their
# lines measured twice (the conjugate rays of the half-fan overlap and the seam where the rotation closes, with the
# scatter-to-primary ratios of this script) and to the open-beam level of every frame with air beside the patient.
# The fast gain follows the first frames; its time constant sits at the fit's lower bound of 2 frames.
DRIFT = {"gain_buildup": 0.029912,                 # B
         "gain_time_constant_frames": 194.98,      # tau_g
         "fast_gain_buildup": 0.013897,            # B_f
         "fast_gain_time_constant_frames": 2.0,    # tau_f
         "offset_buildup_counts": 19.876,          # A
         "offset_time_constant_frames": 160.60}    # tau_o
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
VOXEL_CM = 0.125                       # voxel of the last stage, which is also the final image
GRID_WIDTH_CM = 60.0                   # lateral extent of the volume (at least the reference volume's)
GRID_LENGTH_CM = 32.0                  # extent along z
SCATTER_VIEW_STRIDE = 16               # the scatter is computed on every 16th view and interpolated in angle between
GROUPING_OBJECT_SIZE_CM = 30.0         # object size of the solver's energy groups
QUADRATURE_POINT_SPACING_CM = 1.0      # spacing of the scatter solver's quadrature points
CONE_POLAR_FACTOR = 2                  # the polar directions of the detector projection, times the default
AIR_THRESHOLD = 0.02                   # voxels below this attenuation in 1/cm (a tenth of water's) are air in the
                                       # scatter scene
MBIR_ITERATIONS_PER_OUTER = 8          # MBIR iterations per outer iteration (one scatter update each)
# Multigrid stages, coarse to fine: (voxel in cm (None: VOXEL_CM), detector rows averaged, channels averaged, view
# stride, outer iterations), the binning and stride on the loaded data. Each stage starts from the previous stage's
# volume and scatter state; the last stage is the loaded data on the final grid.
STAGES = ((0.5, 4, 4, 4, 8), (0.25, 2, 2, 2, 4), (None, 1, 1, 1, 2))
HARDENING_MATERIALS = ("bone_cortical_icrp",)   # beam hardening beyond water's, corrected in the loop
DEVICE = "cuda"
# share of the GPU's memory PyTorch may hold: with a cap its allocator frees cached blocks before it reaches it
# (garbage_collection_threshold above), without one Windows spills into host memory
GPU_MEMORY_FRACTION = 0.9
RECONSTRUCT = True                     # False: only the figure, from the saved reconstruction
OUTPUT = os.path.join(HERE, "results", f"{CASE}_LowRes")

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
    # the flat field at the scan's open beam: its level, then each row's level from the scan's own air columns (the
    # air scans and the scan cut the beam with different axial blades)
    scan.flat_field *= np.float32(CASES[CASE]["open_beam_scale"])
    row_profile = air_row_profile(scan)
    scan.flat_field *= row_profile[None, :, None]
    print(f"[{CASE}] open beam: {CASES[CASE]['open_beam_scale']:.4f} of the air scans; row profile "
          f"{row_profile.min():.3f} to {row_profile.max():.3f}", flush=True)
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
    # the volume: at least GRID_WIDTH_CM and the reference volume wide, in whole cm, with floor(extent / voxel) voxels
    # per axis, so that every stage's grid has the same 1 cm transport cells (they carry the scatter state)
    width_cm = math.ceil(max(GRID_WIDTH_CM, ct.shape[2] * spacing[0] / 10.0, ct.shape[1] * spacing[1] / 10.0))
    n_xy = int(width_cm / VOXEL_CM + 1e-6)
    grid = Grid(shape=(n_xy, n_xy, int(GRID_LENGTH_CM / VOXEL_CM + 1e-6)), delta_voxel=VOXEL_CM)
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
    # MBIR reconstructs the field of view and the couch; the start is 0 elsewhere, since the scatter solver sees the
    # whole volume and an FDK leaves streaks outside the field of view
    region = field_of_view_region(grid, geometry, couch_per_cm > 0)
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
    X, loop_info = citrus_recon(scan.counts, scan.flat_field, scan.angles, grid, geometry, solver, config,
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
clinical = read_volume(os.path.join(case_dir, "cbct_clinical.mha"))[0]
fov = read_volume(os.path.join(case_dir, "fov_cbct_nocouch.mha"))[0]
figure_path = os.path.join(OUTPUT, f"{CASE}_comparison.png")
comparison_figure(recon_hu, clinical, ct, fov, spacing, figure_path, f"{CASE} (TrueBeam, low resolution)")
print(f"[{CASE}] figure {figure_path}", flush=True)
