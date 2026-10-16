# citrus

**CITRUS** (Coupled Iterative Tomographic reconstruction and Recursively-Updated Scatter) reconstructs cone-beam CT
volumes from scatter-contaminated projections by iterating model-based iterative reconstruction (MBIR, via
[MBIRTORCH](https://github.com/cabouman/mbirtorch)) and a deterministic scatter solver for the linear Boltzmann
transport equation (LBTE) **jointly**: every MBIR iteration is paired with one source-iteration step of the scatter
solver. This replaces the usual sequential pipeline (FDK, one scatter estimate from that volume, correction, then a
reconstruction that never revisits the scatter), whose scatter estimate inherits the errors of the scatter-corrupted
initial volume.

Full documentation: [citrus.readthedocs.io](https://citrus.readthedocs.io).

## How CITRUS works

Each outer iteration k converts the current reconstruction X_k into water / bone densities, takes one scatter step
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

**Sparse-view scatter.** `CitrusConfig(scatter_view_stride=n)` runs the scatter step on every n-th view only (views
0, n, 2n, ...; only those keep an accumulator). For the other views the scatter *signal* is interpolated in angle, per
detector pixel, from the computed views (`scatter_view_interp_order`: 3 = Catmull-Rom on the 4 nearest computed views,
circular in angle, the default; 0 nearest, 1 linear, 2 quadratic), their primary comes from the solver's cheap primary
projection, and the correction ratio is formed on every view (the ratio itself carries the moving shadow edges and is
not interpolated). On the 180-view WLAB study (20 outer iterations, release defaults), the final relative RMSE is
0.0429 at stride 1, 0.0427 at stride 2, 0.0429 at stride 4 and 0.0438 at stride 8, for scatter times of 56, 29, 15 and
9 s per outer iteration (stride 8 computes scatter on 23 of the 180 views); cubic interpolation matches or beats
linear at no extra cost. The default is stride 1 (every view).

**Scatter step every n outer iterations.** `CitrusConfig(scatter_every=n)` runs the scatter stage at the outer
iterations 0, n, 2n, ... only and reuses the previous scatter-to-primary ratio in between, so n MBIR iterations run
per scatter step (the accumulators keep their age). An MBIR iteration costs about 1 % of a scatter stage, so for a
given number of scatter steps the extra MBIR iterations are nearly free, and they help: on the 180-view WLAB study
the same 20 scatter steps give a relative RMSE of 0.0429 at 20 outer iterations (n = 1), 0.0310 at 40 (n = 2) and
0.0286 at 80 (n = 4) in the same wall time, while at 40 iterations n = 2 is within 0.0008 of n = 1 (0.0302) at half
the scatter cost. At a fixed number of outer iterations, skipping scatter stages costs accuracy (20 iterations:
0.0513 at n = 2, 0.0700 at n = 4). The default is 1.

## The scatter solver

`citrus.scatter_solver` is a deterministic LBTE solver for kV photons: multigroup in energy (Compton and Rayleigh
cross sections from EPDL97 via XrayPhysics), discrete ordinates in angle with real spherical harmonics,
linear-discontinuous finite elements on a supervoxel grid swept in wavefront order, an analytic (exact-in-energy)
uncollided fluence and first-collision source, source iteration over collision order, and a collapsed-cone
projection of the last-collided source onto the detector with a finite-cone footprint, detector and anti-scatter-grid
responses. It runs in PyTorch with fused Triton kernels (ray tracing, first-collision source, sweep) and takes about
0.5 s per view on an RTX 3070. Against Geant4 Monte Carlo on a water / lung / air / bone phantom, the primary agrees
within 0.1 % and the scatter within 0.3 %.

## What the synthetic study showed

On synthetic WLAB data (180 views, 0.5 cm grid, Poisson noise, scatter-to-primary ratios up to 25):

- CITRUS reaches the accuracy of fully converging the scatter at every outer iteration (relative RMSE 0.043 vs 0.043
  after 20 iterations, water within 1.3 %) at about 60 % of its scatter cost.
- The sequential pipeline leaves a large residual bias (water 23 % low, relative RMSE 0.165): its single scatter
  estimate comes from FDK densities that are 40 % low.
- Without scatter correction the water region is 41 % low (relative RMSE 0.29).

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
  clang, on Windows Visual Studio with the C++ build tools and CMake on `PATH`. **Without XrayPhysics the
  cross-section library cannot be built**, so the demos and real reconstructions need it (or a cached library); the
  test suite uses a synthetic library and skips the tests that need the real one.
- **Triton** (`gpu` extra: `triton` on Linux, `triton-windows` on Windows) enables the fused kernels on CUDA. Without
  it, or on the CPU, the PyTorch reference paths are used automatically (`CITRUS_DISABLE_TRITON=1` forces them).

Developer tools: `dev_scripts/` (conda test environment, clean install, tests, docs, CUDA check);
`benchmarks/` (benchmarks of the sweep, the first-collision source and the ray tracer, and a profile of the CITRUS
step).

## Quick start

```bash
python demo/demo_scatter_solver.py   # the scatter solver on one view of the WLAB phantom (seconds)
python demo/demo_citrus.py           # CITRUS vs uncorrected MBIR: 36 views, 1 cm grid, 10 outer iterations (~2 min)
python demo/demo_citrus.py --full    # 180 views, 0.5 cm grid, 20 outer iterations
python demo/demo_citrus.py --scatter-view-stride 2   # scatter step on every 2nd view, the others interpolated
```

Figures are written to `demo/output/`. In Python:

```python
from citrus import CitrusConfig, citrus_recon
from citrus.utilities import wlab_solver

solver = wlab_solver(delta_voxel, geom=geom)       # ScatterSolver with the cached cross-section library
recon, info = citrus_recon(Y, I0, angles, grid, geom, solver, CitrusConfig(num_outer=10))
```

Tests: `dev_scripts/run_tests.sh` or `python -m pytest -q` (GPU, Triton, MBIRTORCH and XrayPhysics tests are skipped
when unavailable).

## Citation

Please cite the software when referencing this package.

```bibtex
@misc{citrus,
  title = {citrus},
  author = {Jordan Houri and Charles A. Bouman},
  howpublished = {Software library available from \url{https://github.com/cabouman/citrus}},
  note = {Version 0.0.3},
  year = 2026
}
```

GitHub's "Cite this repository" button on the repository page generates the
citation from `CITATION.cff`.
