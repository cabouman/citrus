# Changelog

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
