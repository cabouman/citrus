# Changelog

## v0.3.0 — 2026-10-17

Densities from the attenuation one material at a time, MBIRTORCH primaries in the CITRUS loop, MBIR resumed from its
error sinogram, quadrature points by spacing. **Breaking changes** (configuration names); the loop's default primary
changes the demo result (below).

- **No seven-material fine volume in the loop.** `utilities.attenuation_to_density(lib, X, m)` evaluates ONE
  material's map of the interpolating model, bit for bit the row m of `attenuation_to_densities` (in place, two
  boolean masks and one weight as temporaries); `DensityModel.material(X, m)` evaluates it slab by slab
  (`DensityModel.slab_voxels`, default 4 M voxels) so the result is the only full-size map;
  `DensityModel.negative_fraction(X)`. `ScatterSolver.material_fields_from_attenuation(X, grid_fine, density_model)`
  builds the `MaterialFields` material by material (map -> resampled onto C and Q -> dropped; absent materials skipped,
  one host synchronization per material), equal to `material_fields(density_model(X), grid_fine)` (bit for bit on
  WLAB in the tests that compare them, float rounding in general). `MaterialFields.rho_fine` may be None; new fields
  `attenuation`, `density_model` (any `scatter_solver.MaterialDensityModel`: `.material(X, m)`), `fine_materials`
  (materials present) and `fine_material(m)`, `num_materials`. The "fine" uncollided trace and the primary trace such
  fields one map at a time. `citrus_recon` uses this path; `citrus_step(solver, state, rho, grid, ...)` and every
  rho-based path (`material_fields`, `solve_view`, the phantom, the tests) are unchanged and bit-identical.
  Peak allocated memory of a scatter stage on the 0.125 cm WLAB grid (12.3 M voxels, 1 cm cells, 180 views, stride
  8; `benchmarks/bench_citrus_memory.py`): 2521 / 2364 MiB -> 1891 / 1876 MiB (outer iterations 2 / 3); the density
  stage itself 1.56 GB transient + 0.33 GB kept -> 0.1 GB transient.
- **Two-step primary.** `scatter_image.primary_path_lengths(projectors, rho, grid, geom, angle, n_sub)` (per-material
  line integrals [M', Rd n, Cd n]) and `primary_from_path_lengths(lib, path, materials, geom, angle, source, detector,
  grid_response)` (fine-energy transmission, sub-pixel average, source, inverse square, detector and grid
  responses); `primary_subpixel_geometry`. `primary_image` is their composition and equals version 0.2.0 bit for bit
  (tested against a verbatim copy). `ScatterSolver.primary_path_lengths` / `primary_from_path_lengths`;
  `uncollided.uncollided_path_lengths` / `uncollided_from_path_lengths` likewise.
- **Primaries of all views in the loop.** `citrus.primary_signals(solver, fields, angles, projector,
  views_per_chunk=None)`: one projection per material present onto all views, in chunks of views whose path lengths
  stay within one sinogram (default max(1, V // (M' + 1)) views), then `primary_from_path_lengths` per view.
  Projectors: `recon.mbir_bridge.ViewChunkProjector` (MBIRTORCH's pixel-averaged separable-footprint line
  integrals; sub-models over the angles of a chunk, about 1 ms to build, cached) and `citrus.SiddonViewProjector`
  (the solver's Siddon rays, equal to `ScatterSolver.primary` bit for bit). New `CitrusConfig.primary_projector`
  ("mbirtorch", **default**, or "siddon") and `CitrusConfig.primary_views_per_chunk` (None). The loop computes every
  view's primary this way (the sparse-view path no longer traces the skipped views separately);
  `citrus_view_step(solver, state, fields, angle, primary=True)` is the step from prebuilt fields (the loop calls it
  with primary=False); records carry `t_primary`, info `primary_projector`.
  MBIRTORCH vs Siddon primary on the WLAB study (0.5 cm, 180 views, short scan, true densities): relative L2 0.9 %,
  RMS of the per-pixel relative difference 3.9 %, median 0.05 %, 99th percentile 3 %, maximum 5x (factor 6 in
  transmission) on the pixels whose ray grazes a face of the prism (a centre ray against a pixel average).
  The pixel-integrated primary is the physical model of a detector pixel, hence the default.
- **Synthetic data consistent with the loop's primary (data-set format 2).** `synthesize_cone_beam_data` /
  `synthesize_wlab_dataset(..., primary_projector="mbirtorch")`: the SPR's denominator is the solver's primary on
  MBIRTORCH's per-material line integrals of the true densities (`primary_signals(..., quantity="planar_fluence")`,
  new `quantity` argument); "siddon" keeps the definition of version 0.2.0. meta records `primary_projector` and
  `data_format` (2). Demo: `--truth-primary-projector`; cache and output names carry `_v2` (and `_primary-siddon`
  for the Siddon definition), so data sets of version 0.2.0 are never reused; the old data sets and outputs in
  `demo/input` and `demo/output` are renamed `..._v1...` (format 1: SPR with the Siddon primary). With the old
  definition the MBIRTORCH primary was inconsistent with the data on the silhouette pixels (RMS 0.8 % of the corrected
  line integrals at the true densities, up to 0.87 on single pixels).
  Demo defaults on the new data (`python demo/demo_citrus.py`, synthesis 113 s once, CITRUS 159 s): relative RMSE FDK
  0.318 / 0.348, no correction 0.292 / 0.355, CITRUS 0.0215 / 0.0259 (whole prism / core; version 0.2.0 on its
  format-1 data: 0.0242 / 0.0294); with `--primary-projector siddon` on the same data 0.0257 / 0.0305 (on the format-1
  data the MBIRTORCH loop gave 0.0270 / 0.0305 and the Siddon loop 0.0242 / 0.0294).
- **MBIR resume.** `recon.mbir_bridge.mbir_iterations(..., error_sinogram=None, hessian=None,
  return_checkpoint=False)`: with any of them it replicates MBIRTORCH 0.1.2's `recon` (`initialize_recon`, then
  `_vcd_recon` with `init_error_sinogram` / `fm_hessian` / `return_checkpoint`, which `recon` does not forward) and
  returns the checkpoint {'error_sinogram', 'fm_hessian'}. `CitrusConfig.resume_error_sinogram` (**default True**):
  the loop keeps -A X between outer iterations (one sinogram) and resumes from e = y_c - A X; equal to the restarted
  MBIR to 1.1e-6 (relative L2, demo after 80 iterations); MBIR iteration on the 12 M-voxel grid 0.63 / 1.02 s instead
  of 0.73 / 1.09 s; peak +18 MiB on the demo (one sinogram, 19 MiB). The Hessian is recomputed (keeping it would hold
  a reconstruction-sized volume).
- **Quadrature points by spacing.** `SolverConfig.fc_points_per_axis` is removed; `SolverConfig.
  quadrature_point_spacing_cm` (default 0.5): q = max(1, round(transport_cell_cm / spacing)), bounded as before by
  max(1, round(transport_cell_cm / largest fine voxel edge)). Unchanged for 1 cm cells (q = 2 at 0.5 cm voxels, 1 at
  1 cm); 2 cm cells now get q = 4 (was 2), 0.5 cm cells q = 1. Study (WLAB 0.5 cm, converged, AP / lateral, rel. L2
  of the scatter signal): 2 cm cells q = 1 / 2 against q = 4 1.1 % / 0.59 % and 0.40 % / 0.22 % (0.19 / 0.20 /
  0.35 s per view); 1 cm cells q = 1 against 2 0.25 % / 0.09 %; the cell size itself costs more (2 cm against 1 cm
  cells: 3.6 % / 9.5 %).
- **Renamed (plain words).** `SolverConfig.fc_trace_grid` -> `uncollided_trace_grid` (values unchanged); benchmark
  flags `--fc-points` -> `--quadrature-spacing` (cm), `--fc-trace` -> `--uncollided-trace`; the demo's summary JSON
  keys `fc_points_per_axis` -> `quadrature_points_per_axis` (plus `quadrature_point_spacing_cm`).
- Demo (`demo/demo_citrus.py`): `--primary-projector`, `--resume-error-sinogram / --no-resume-error-sinogram`,
  `--save-recon`; output suffixes `_siddon`, `_restart`; the summary records `primary_projector`,
  `resume_error_sinogram` and the CITRUS run's peak allocated GPU memory (1636 MiB at the defaults; 1624 MiB in
  version 0.2.0) and per-iteration times. `benchmarks/bench_citrus_memory.py` (peak memory and time per outer
  iteration, split into scatter stage and MBIR).
- Tests: `tests/test_attenuation_path.py` (per-material density bit for bit, fields from X against fields from rho,
  their peak memory, the two-step primary against version 0.2.0, MBIRTORCH against Siddon primaries, chunking,
  configuration), resume tests in `tests/test_mbir_bridge.py` and `tests/test_citrus_recon.py`, the MBIRTORCH
  primary in the loop, the spacing rule in `tests/test_quadrature_grid.py`.

## v0.2.0 — 2026-10-17

First-order quantities on a quadrature grid of the transport cells (Acuros CTS, Maslowski et al. 2018, Med Phys
45:1899, Section 2.B); transport cells of any size. **Breaking changes.**

- **Transport grid by cell size.** `SolverConfig.supervoxel_factor` is removed; `SolverConfig.transport_cell_cm`
  (default 1.0) sets the edge of the cubic transport cells. The transport grid C = `Grid.covering(fine_grid, cell)`
  covers the fine box, centred on it (ceil(extent / cell) cells per axis, symmetric empty padding): any ratio of voxel
  to cell size. `ScatterSolver.transport_grid`, `quadrature_grid`; `utilities.default_supervoxel_factor` is removed
  (`wlab_solver` / `synthesize_wlab_dataset` default to 1 cm cells, unchanged on the 1 and 0.5 cm WLAB grids). Demo
  flags `--supervoxel-factor` -> `--transport-cell-cm` (cache-name suffix `_cell<cm>`, run JSON `transport_cell_cm`).
- **Conservative resampling.** `geometry.resample_conservative` / `overlap_matrix` / `resampling_matrices`: exact
  cell averages of the voxel-wise constant fine field (separable per-axis overlap matrices, cached), mass conserving for
  any ratio; equal to the block mean for integer ratios. `scatter_solver.coarsen_density(rho, grid_fine, grid_coarse)`
  and `scatter_image.coarsen_mu_t(mu, grid_fine, grid_coarse)` use it (new signatures).
- **Quadrature grid Q = C.refine(q)**, `Grid.refine`. `SolverConfig.fc_points_per_axis` (default 2) is an upper
  bound: q = min(fc_points_per_axis, max(1, round(transport_cell_cm / largest fine voxel edge))) (Python's round;
  `SolverConfig.quadrature_points`, `ScatterSolver.quadrature_points`), so 1 cm voxels in 1 cm cells keep q = 1 and the
  cost of version 0.1.0 (CITRUS step about 0.12 s per view), 0.5 cm and smaller voxels get q = 2. The uncollided
  fluence, the first-collision source at the sweep and cone directions, its sampling correction and restriction, and the
  first-order emission, footprint and projection of the scatter image all live on Q; only the primary image and the
  resampling touch the fine grid. `MaterialFields`: `grid_quad`, `q`, `rho_quad`, `pairs_quad` (None -> dense),
  `fc_pairs`; `factor`, `mu_t_fine` and `pairs_fine` are removed. `ScatterSolver.uncollided` returns phi_uc on Q.
- **Uncollided trace.** `SolverConfig.fc_trace_grid`: "quadrature" (default, through Q), "coarse" (through C: smears
  a boundary inside a cell, up to 4 % of the lateral scatter when the cells are offset half a cell from the object) or
  "fine" (through the reconstruction grid), all at about the same cost; `uncollided_fluence(..., points=)`, `Projectors.upstream_to_points`. With
  q = cell / voxel and the fine trace the solver reproduces version 0.1.0 to float rounding
  (`tests/data/pre_quadrature_reference.json`: 8 x 8 block means of the signals; `tests/test_quadrature_grid.py`).
- **Dense first-collision kernel for any M.** Resampled densities can mix more than two materials per point: the Triton
  kernel's dense mode now takes all M materials (compile-time M; 2.7x the pair-mode time per point at M = 7);
  `cross_sections.material_pairs_if_compact`, `max_materials_per_point`; `pairs=` accepts a `MaterialPairs`,
  `uncollided.DENSE` or None (decide from rho).
- `combined_scatter_image(lib, rho_quad, phi_uc_quad, grid_quad, q, ..., grid_coarse=None)`,
  `supervoxel_subpoint_emission(src_quad, ..., grid_quad, q)`, `DirectionSetContext.subcell_offsets(grid_quad, q)`.
- Cost: a CITRUS step is about 0.31 s per view at 0.5, 0.25 and 0.125 cm voxels with 1 cm cells (version 0.1.0:
  0.32 s, 1.76 s, out of memory). `benchmarks/profile_citrus_step.py --voxel --cell --fc-points --fc-trace`.

## v0.1.0 — 2026-10-16

Seven-material density model; short-scan data by default. **Breaking changes; results change.**

- **Densities are `[7, ...]`.** The basis is the seven-material table `cross_sections.MATERIAL_TABLE` (air 0.0012,
  lung 0.26, adipose 0.95, water 1.0, skeletal muscle 1.05, soft tissue 1.06, cortical bone 1.85 g/cm^3, in order of
  increasing attenuation at the nominal density; `MATERIALS`, `NOMINAL_DENSITIES`, `WATER = 3`, `BONE = 6`).
  `wlab_phantom` / `wlab_setup` return `[7, R, C, S]` (the water and bone rows); `citrus_step`, the per-view operators,
  `ScatterSolver.solve_view` / `material_fields`, `synthesize_cone_beam_data` (`rho_true [7, ...]`,
  `mass_attenuation [7]`, meta `materials`, `nominal_density`, `mass_attenuation` by name) take and return seven
  materials. `material_fields` checks the number of materials against the library.
- **The two-material split is gone.** `utilities.attenuation_to_densities(lib, X, energy_kev=None)` is the
  interpolating seven-material model: the two materials whose nominal attenuations bracket X get (1 - f) rho_a and
  f rho_b, f = (X - mu_a) / (mu_b - mu_a); below air air alone, above bone bone alone (X / m); X < 0 -> 0. Exact in
  attenuation, pure material at its nominal attenuation (a cortical-bone voxel is bone at 1.85 g/cm^3, not unit water
  plus 1.156 g/cm^3 bone), at most two materials per voxel. New helpers `nominal_densities`, `nominal_attenuation`,
  `material_order`. `DensityModel` keeps its API and uses the new model. On the 180-view WLAB study (20 outer
  iterations): relative RMSE 0.0389 / 0.0451 (prism / core) instead of 0.0429 / 0.0505, water +0.6 % instead of
  -1.3 %, the bone insert +0.6 % instead of +6.3 %; scatter stage 62.6 s instead of 56-58 s per outer iteration.
- **Library.** `MultigroupLibrary.materials`, `nominal_density [M]`, `num_materials`; `cross_section_library`:
  `MATERIAL_FORMULAS` (seven entries), `default_material_table`, `check_material_order` (the build refuses a table whose
  nominal attenuations cross at any library energy), `compute_library` / `build_library` / `build_solver_library`
  take `materials=` (name, nominal density, formula) for other tables of any size. **Cache key change**: build version
  `citrus-xsec-2`, the key includes the material table; the two-material caches of 0.0.x are never loaded (the
  seven-material library builds once, about 16 s). `synthetic_library(num_materials=)` (2 default, 7 = the package
  table with the two base materials in the water and bone rows, any other M).
- **Pair representation.** `cross_sections.MaterialPairs` / `material_pairs` (index / density `[2, ...]`),
  `transfer_eta_at_pairs`; the first-collision Triton kernel gains a PAIR mode (two table rows gathered by material
  index; equals the PyTorch pair path to 1.4e-7 of the field maximum), the PyTorch pair path, `MaterialFields.pairs_fine`
  (derived once per density field), `pairs=` in `fc_source_at_directions`, `fc_sampling_correction`,
  `fc_quadrature_sums`, `combined_scatter_image`; `uncollided.present_materials` / `group_transmission(materials=)`:
  the uncollided fluence and the primary trace only the materials present. On WLAB (water and bone only) the
  seven-material solve equals the two-material solve bit for bit in float32. CITRUS step on the 0.5 cm WLAB problem:
  335 ms per view instead of 317 ms (same session).
- **Short scans.** `utilities.fan_angle(geometry)` (in-plane fan angle of the panel, 2 atan(0.5 Nc dc / SDD), 22.6
  degrees for the full-field WLAB panel) and `utilities.view_angles(num_views, geometry, start_view_angle=0.0,
  end_view_angle=None)` (evenly spaced over [start, end), endpoint excluded; default end = start + pi + fan angle).
  `synthesize_wlab_dataset(..., start_view_angle=0.0, end_view_angle=None)` replaces `angle_offset`: **the default data
  set is now a short scan**; pass `end_view_angle=2 pi` for a full rotation. Demo flags `--start-view-angle` /
  `--end-view-angle` (degrees); `--full` keeps the report's 180 views over a full rotation; cached data sets of a
  short scan carry `_short` in their name.
- **FDK of a short scan.** MBIRTORCH's FDK has no short-scan weighting; `recon.mbir_bridge.fdk(..., geom=, angles=)`
  applies Parker weights (`short_scan_weights`, `scan_range`, `is_short_scan`) and `citrus_recon` / `mbir_recon` use
  it for the initialization. Sparse-view interpolation runs along the arc for a short scan (`citrus.full_rotation`,
  `interpolation_weights(..., circular=None)`, `angular_neighbors(..., circular=)`).
- Demo (36 views, 1 cm grid, 10 outer iterations): relative RMSE 0.064 (short scan, default) and 0.075 (full
  rotation; 0.0751 in 0.0.3).
- Tests: `tests/test_materials.py` (library build and order check, cache keys, pair representation, pair kernel vs
  PyTorch, seven vs two materials on WLAB with the synthetic and the real library, density-model round trip and
  bracketing, the WLAB bone voxel, the fixed-SPR reproduction); short-scan tests in `test_utilities.py`,
  `test_mbir_bridge.py`, `test_sparse_views.py`, `test_citrus_recon.py`; the WLAB tests use the seven-material
  synthetic library (`num_materials=7`).

## v0.0.3 — 2026-10-16

Scatter step every n outer iterations.

- `CitrusConfig.scatter_every` (default 1): the scatter stage runs at the outer iterations 0, n, 2n, ... only and
  the previous scatter-to-primary ratio is reused in between, so n MBIR iterations run per scatter step (the
  accumulators keep their age). Validated in `CitrusConfig.validate`; every record of `citrus_recon` now carries
  `scatter_refreshed`.
- Demo flag `--scatter-every` (`demo/demo_citrus.py`); the summary reports `scatter_steps`.
- Measured on the 180-view WLAB study: the same 20 scatter steps give a relative RMSE of 0.0429 at 20 outer
  iterations (n = 1), 0.0310 at 40 (n = 2) and 0.0286 at 80 (n = 4) in the same wall time; at 40 iterations n = 2 is
  within 0.0008 of n = 1 (0.0302) at half the scatter cost. At a fixed 20 outer iterations skipping stages costs
  accuracy (0.0513 at n = 2, 0.0700 at n = 4). At equal scatter cost CITRUS at every iteration (0.0429) beats a
  converged scatter solve at every 2nd iteration (0.0501).
- Tests: configuration checks in `tests/test_sparse_views.py`, a CUDA run with `scatter_every=2` in
  `tests/test_citrus_recon.py`.

## v0.0.2 — 2026-10-15

Sparse-view scatter with angular interpolation.

- `CitrusConfig.scatter_view_stride` (default 1): the CITRUS scatter step runs on every n-th view only (views 0, n,
  2n, ...; only those keep an accumulator, 1/n of the state memory); the other views' scatter signal is interpolated
  in angle, per pixel, from the computed views, their primary comes from the solver's primary projection, and the
  correction ratio is formed on every view.
- `CitrusConfig.scatter_view_interp_order` (default 3), all circular in angle: 3 cubic Catmull-Rom on 4 neighbors, 2
  quadratic (3-point Lagrange), 1 linear, 0 nearest computed view, None no interpolation (diagnostics: skipped views
  uncorrected). Validated in `CitrusConfig.validate`.
- `citrus.citrus`: `scatter_view_indices`, `angular_neighbors`, `interpolation_weights`, `interpolate_scatter`;
  `citrus_recon`'s info gains `computed_views` and `view_interpolated`, the records `t_skipped`. `mbir_recon` is
  unchanged.
- Demo flags `--scatter-view-stride`, `--scatter-view-interp-order` (`demo/demo_citrus.py`; default stride 1).
- Measured on the 180-view WLAB study (20 outer iterations): stride 2 +0.0014 relative RMSE for 51 % of the scatter
  time, stride 4 +0.008 for 26 %.
- Tests: `tests/test_sparse_views.py` (interpolators, configuration) and CUDA runs at strides 2 and 4 in
  `tests/test_citrus_recon.py`.

## v0.0.1 — 2026-10-14

First release.

- `citrus.citrus`: the CITRUS reconstruction loop (`citrus_recon`): FDK initialization, then per outer iteration one
  scatter step per view (age-ordered source iteration with a per-view collided-fluence accumulator, stored compactly)
  and one warm-started MBIRTORCH iteration on the scatter-corrected data; `mbir_recon` (the same loop without the
  solver); the per-view operators (`citrus_step`, `G1`, `G1to2`, `G_gt2`, `G_step`, `project_scatter`).
- `citrus.scatter_solver`: the LBTE source iteration scatter solver (`ScatterSolver`): multigroup cross sections from
  XrayPhysics (EPDL97) with an exact-in-energy uncollided fluence and primary, discrete ordinates with real spherical
  harmonics, linear-discontinuous finite elements on supervoxels with a wavefront sweep, source iteration with a
  conservative first-collision source, collapsed-cone scatter image with sub-voxel points and a finite-cone footprint,
  detector and anti-scatter-grid responses; exact Siddon ray tracing. Fused Triton kernels on float32 CUDA with
  PyTorch reference paths for CPU / float64.
- `citrus.recon`: the MBIRTORCH bridge; `citrus.utilities`: WLAB phantom and geometry, attenuation <-> density
  model, synthetic cone-beam data with scatter (`synthesize_cone_beam_data`, `synthesize_wlab_dataset`), plotting
  helpers.
- Demos (`demo/demo_scatter_solver.py`, `demo/demo_citrus.py` with a configurable ground-truth discretization), test
  suite, Sphinx documentation, developer shell scripts (`dev_scripts/`) and benchmarks (`benchmarks/`).
- Dependencies: numpy, scipy, torch, matplotlib, mbirtorch; extras `test`, `docs`, `physics` (XrayPhysics), `gpu`
  (Triton).
