# citrus

**CITRUS** (Coupled Iterative Tomographic reconstruction and Recursively-Updated Scatter) reconstructs cone-beam CT
volumes from scatter-contaminated projections by iterating model-based iterative reconstruction (MBIR, via
[MBIRTORCH](https://github.com/cabouman/mbirtorch)) and a deterministic scatter solver for the linear Boltzmann
transport equation (LBTE) **jointly**: every outer iteration refreshes the scatter estimate with one source-iteration
step of the scatter solver from the current reconstruction and runs a chosen number of MBIR iterations on the
corrected data. This replaces the usual sequential
pipeline (FDK, one scatter estimate from that volume, correction, then a reconstruction that never revisits the
scatter), whose scatter estimate inherits the errors of the scatter-corrupted initial volume.

Full documentation: [purduecitrus.readthedocs.io](https://purduecitrus.readthedocs.io).

## How CITRUS works

The loop starts from the FDK reconstruction X_0 of the uncorrected data. Outer iteration k = 1, ..., N converts
X_{k-1} into material densities, takes one scatter step per view, corrects the measured counts with the estimated
scatter-to-primary ratio, `Y_c = Y / (1 + SPR)`, and runs warm-started MBIR iterations on the corrected data,
which give X_k. By default both counts are adaptive: the MBIR iterations stop once the fit of the corrected sinogram
improves by less than a threshold per iteration, and the loop stops once an outer iteration improves it by less than
another; fixed counts can be given instead.

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

`citrus.scatter_solver` is a deterministic LBTE solver for kV and MV photons: multigroup in energy (cross sections of
NIST XCOM, Compton and Rayleigh angular distributions of xraylib), discrete ordinates in angle with real spherical
harmonics, linear-discontinuous finite elements on a grid of transport cells swept in wavefront order, an analytic
(exact-in-energy) uncollided fluence and first-collision source on a few quadrature points per cell, source iteration
over collision order, and a collapsed-cone projection of the last-collided source onto the detector with a finite-cone
footprint, detector and anti-scatter-grid responses. It runs in PyTorch with fused Triton kernels (ray tracing,
first-collision source, sweep).

## Installation

```bash
git clone git@github.com:cabouman/citrus.git
cd citrus
pip install .                  # numpy, scipy, torch, matplotlib, mbirtorch, xraylib
pip install ".[gpu]"           # optional: Triton (see below)
```

Install PyTorch first if you need a specific CUDA build. Every dependency installs from wheels; nothing is compiled.

- **Photon data.** The multigroup cross-section library is built from the photon cross sections of NIST XCOM and the
  Compton and Rayleigh angular distributions of [xraylib](https://github.com/tschoonj/xraylib) (installed with the
  package). The XCOM tables are not shipped with CITRUS: on first use it downloads NIST's archive (173 kB, from
  `https://physics.nist.gov/PhysRefData/Xcom/XCOM.tar.gz`, checked against a SHA-256) into `~/.cache/citrus/xcom` (or
  `$CITRUS_CACHE_DIR/xcom`) and parses it once; built libraries are cached there too. On a machine without internet
  access download that file elsewhere and set `CITRUS_XCOM_ARCHIVE` to its path (or to the folder it unpacks to).
  Without the XCOM data the cross-section library cannot be built (a cached library still loads); the test suite uses
  a synthetic library and skips the tests that need the real one.
- **Triton** (`gpu` extra: `triton` on Linux, `triton-windows` on Windows) enables the fused kernels on CUDA. Without
  it, or on the CPU, the PyTorch reference paths are used automatically (`CITRUS_DISABLE_TRITON=1` forces them).

Developer tools: `dev_scripts/` (conda test environment, clean install, tests, docs, CUDA check);
`benchmarks/` (benchmarks of the sweep, the first-collision source and the ray tracer, a profile of the CITRUS
step, and the CITRUS study script `citrus_study.py`).

## Quick start

```bash
python demo/demo_1_scatter_solver.py   # the scatter solver on one view of the WLAB phantom
python demo/demo_2_citrus.py           # CITRUS vs uncorrected MBIR: 180 views over a short scan, 0.5 cm voxels
```

The two demos are short scripts that show the API step by step, with every setting (spectrum, reference materials,
panel, voxel pitch, views, loop and solver settings) a variable at the top. The first run of demo 1 downloads the
photon data and builds the cross-section library (both cached); demo 2 caches its synthetic data in `demo/input/`.
For studies (other sparse-view and sparse-iteration scatter schedules, a finer ground truth, the RMSE history and run
statistics), `benchmarks/citrus_study.py` takes the settings as flags and writes figures and a summary JSON to
`demo/output/`:

```bash
python benchmarks/citrus_study.py          # 180 views over a short scan, 0.5 cm grid, adaptive numbers of outer
                                           # and MBIR iterations, scatter step on every 8th view
python benchmarks/citrus_study.py --full   # 180 views over a full rotation, 0.5 cm grid, 20 iterations,
                                           # scatter step on every view at every iteration
python benchmarks/citrus_study.py --help   # every flag
```

The views are evenly spaced over `[start, end)`; the default end is a short scan, start + 180 degrees + the in-plane
fan angle of the detector (`utilities.view_angles`, `utilities.fan_angle`). MBIRTORCH's FDK, which initializes the
loop, has no short-scan weighting, so CITRUS applies Parker weights before it for a short scan
(`recon.mbir_bridge.short_scan_weights`).

In Python:

```python
from citrus import CitrusConfig, citrus_recon
from citrus.scatter_solver import load_spectrum_txt
from citrus.utilities import wlab_solver

spectrum = load_spectrum_txt("spectrum.txt")               # the source spectrum, two columns: keV, photons per keV
solver = wlab_solver(delta_voxel, geometry=geometry, spectrum=spectrum)   # ScatterSolver, library cached on disk
recon, info = citrus_recon(Y, I0, angles, grid, geometry, solver, CitrusConfig())   # adaptive numbers of iterations
```

Tests: `dev_scripts/run_tests.sh` or `python -m pytest -q` (GPU, Triton and MBIRTORCH tests, and those that need the
XCOM data, are skipped when unavailable).

`CHANGELOG.md` lists the changes of every version, including the old -> new table of renamed settings.

## Citation

Please cite the software when referencing this package.

```bibtex
@misc{citrus,
  title = {citrus},
  author = {Jordan Houri and Gregery T. Buzzard and Charles A. Bouman},
  howpublished = {Software library available from \url{https://github.com/cabouman/citrus}},
  note = {Version 0.10.1},
  year = 2026
}
```

GitHub's "Cite this repository" button on the repository page generates the
citation from `CITATION.cff`.

The photon data come from NIST XCOM (M.J. Berger, J.H. Hubbell, S.M. Seltzer, J. Chang, J.S. Coursey, R. Sukumar,
D.S. Zucker and K. Olsen, *XCOM: Photon Cross Sections Database*, NIST Standard Reference Database 8,
https://physics.nist.gov/xcom) and xraylib (T. Schoonjans et al., *The xraylib library for X-ray-matter interactions.
Recent developments*, Spectrochimica Acta Part B 66, 776-784 (2011)); please cite them too.
