# Changelog

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
