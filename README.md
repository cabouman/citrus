# citrus

**CITRUS** (Coupled Iterative Tomographic reconstruction and Recursively-Updated Scatter) reconstructs cone-beam CT
volumes from scatter-contaminated projections by iterating model-based iterative reconstruction (MBIR, via
[MBIRTORCH](https://github.com/cabouman/mbirtorch)) and a deterministic scatter solver for the linear Boltzmann
transport equation (LBTE) **jointly**: every MBIR iteration is paired with one source-iteration step of the scatter
solver. This replaces the usual sequential pipeline (FDK, one scatter estimate from that volume, correction, then a
reconstruction that never revisits the scatter), whose scatter estimate inherits the errors of the scatter-corrupted
initial volume.

Full documentation: [purduecitrus.readthedocs.io](https://purduecitrus.readthedocs.io).

## How CITRUS works

Each outer iteration k converts the current reconstruction X_k into material densities, takes one scatter step
per view, corrects the measured counts with the estimated scatter-to-primary ratio, `Y_c = Y / (1 + SPR)`, and runs
one warm-started MBIR iteration on the corrected data.

The scatter step is **age-ordered source iteration**. Per view the solver keeps an accumulator of the collided
fluence, `Phi_{>1}`, and updates it with one transport sweep `G_k` (scattering plus transport through the densities of
reconstruction k) applied to the newest first-collision term and to the previous accumulator:

```
Phi_{>1}^{(k)} = G_k phi_1^{(k)} + G_k Phi_{>1}^{(k-1)}
             = G_k phi_1^{(k)} + G_k G_{k-1} phi_1^{(k-1)} + G_k G_{k-1} G_{k-2} phi_1^{(k-2)} + ...
```

where `phi_1^{(k)}` is the uncollided fluence of reconstruction k. Expanded, the second-order scatter comes from the
newest reconstruction, the third order from the previous one, and so on: the dominant low-order scatter is always
refreshed from the freshest reconstruction, while stale information is relegated to the weaker and smoother
higher orders. With fixed densities the recursion *is* source iteration and converges to the fully converged scatter
estimate, so the scatter and the reconstruction converge together for the cost of one sweep per view per MBIR
iteration.

## The scatter solver

`citrus.scatter_solver` is a deterministic LBTE solver for kV photons: multigroup in energy (Compton and Rayleigh cross
sections from EPDL97 via XrayPhysics), discrete ordinates in angle with real spherical harmonics, linear-discontinuous
finite elements on a grid of transport cells swept in wavefront order, an analytic (exact-in-energy) uncollided fluence
and first-collision source on a few quadrature points per cell, source iteration over collision order, and a
collapsed-cone projection of the last-collided source onto the detector with a finite-cone footprint, detector and
anti-scatter-grid responses. It runs in PyTorch with fused Triton kernels (ray tracing, first-collision source, sweep).

## Installation

```bash
git clone git@github.com:cabouman/citrus.git
cd citrus
pip install .                  # numpy, scipy, torch, matplotlib, mbirtorch
pip install ".[physics,gpu]"   # optional: XrayPhysics and Triton (see below)
```

Install PyTorch first if you need a specific CUDA build.

- **XrayPhysics** (`physics` extra) builds the multigroup cross-section library (once; it is cached in
  `~/.cache/citrus` or `$CITRUS_CACHE_DIR`). It is a C++ library and needs a C++ toolchain and CMake: on Linux gcc or
  clang, on Windows Visual Studio with the C++ build tools and CMake on `PATH`. With CMake 4 or newer set
  `CMAKE_POLICY_VERSION_MINIMUM=3.5` in the environment before installing (XrayPhysics declares an older minimum
  CMake version, which CMake 4 otherwise rejects), or use a CMake below 4. Without root access the toolchain can come
  from conda: `conda create -n citrus -c conda-forge python=3.11 git cmake make c-compiler cxx-compiler`
  (`dev_scripts/wsl_install_test.sh` runs the whole installation that way). **Without XrayPhysics the
  cross-section library cannot be built**, so the demos and real reconstructions need it (or a cached library); the
  test suite uses a synthetic library and skips the tests that need the real one.
- **Triton** (`gpu` extra: `triton` on Linux, `triton-windows` on Windows) enables the fused kernels on CUDA. Without
  it, or on the CPU, the PyTorch reference paths are used automatically (`CITRUS_DISABLE_TRITON=1` forces them).

Developer tools: `dev_scripts/` (conda test environment, clean install, tests, docs, CUDA check);
`benchmarks/` (benchmarks of the sweep, the first-collision source and the ray tracer, and a profile of the CITRUS
step).

## Quick start

```bash
python demo/demo_scatter_solver.py   # the scatter solver on one view of the WLAB phantom
python demo/demo_citrus.py           # CITRUS vs uncorrected MBIR: 180 views over a short scan, 0.5 cm grid,
                                     # 80 outer iterations, scatter step on every 8th view at every 8th iteration
python demo/demo_citrus.py --end-view-angle 360      # the same over a full rotation
python demo/demo_citrus.py --full    # 180 views over a full rotation, 0.5 cm grid, 20 iterations,
                                     # scatter step on every view at every iteration
python demo/demo_citrus.py --views 36 --delta-voxel 1 --outer-iterations 10 --scatter-view-stride 1 --scatter-every 1
                                     # a quick run
python demo/demo_citrus.py --materials air,water,bone   # three reference materials in the density model
```

The views are evenly spaced over `[start, end)` (`--start-view-angle`, `--end-view-angle` in degrees); the default
end is a short scan, start + 180 degrees + the in-plane fan angle of the detector (`utilities.view_angles`,
`utilities.fan_angle`). MBIRTORCH's FDK, which initializes the loop, has no short-scan weighting, so CITRUS applies
Parker weights before it for a short scan (`recon.mbir_bridge.short_scan_weights`).

Figures are written to `demo/output/`. In Python:

```python
from citrus import CitrusConfig, citrus_recon
from citrus.utilities import wlab_solver

solver = wlab_solver(delta_voxel, geometry=geometry)       # ScatterSolver with the cached cross-section library
recon, info = citrus_recon(Y, I0, angles, grid, geometry, solver, CitrusConfig(num_outer_iterations=10))
```

Tests: `dev_scripts/run_tests.sh` or `python -m pytest -q` (GPU, Triton, MBIRTORCH and XrayPhysics tests are skipped
when unavailable).

`CHANGELOG.md` lists the changes of every version, including the old -> new table of renamed settings.

## Citation

Please cite the software when referencing this package.

```bibtex
@misc{citrus,
  title = {citrus},
  author = {Jordan Houri and Charles A. Bouman},
  howpublished = {Software library available from \url{https://github.com/cabouman/citrus}},
  note = {Version 0.6.0},
  year = 2026
}
```

GitHub's "Cite this repository" button on the repository page generates the
citation from `CITATION.cff`.
