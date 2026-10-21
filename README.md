# citrus

**CITRUS** (Coupled Iterative Tomographic reconstruction and Recursively-Updated Scatter) reconstructs X-ray CT
volumes (cone-beam scans on flat or curved panels, circular or helical, and parallel-beam scans) from
scatter-contaminated projections by iterating model-based iterative reconstruction (MBIR, via
[MBIRTORCH](https://github.com/cabouman/mbirtorch)) and a deterministic scatter solver for the linear Boltzmann
transport equation (LBTE) **jointly**: every outer iteration refreshes the scatter estimate with one source-iteration
step of the scatter solver from the current reconstruction and runs MBIR iterations on the corrected data (as many as
the fit of the corrected data keeps improving, by default). This replaces the usual sequential pipeline (a direct
reconstruction such as FDK, one scatter estimate from that volume, correction, then a reconstruction that never
revisits the scatter), whose scatter estimate inherits the errors of the scatter-corrupted initial volume.

Full documentation: [purduecitrus.readthedocs.io](https://purduecitrus.readthedocs.io).

## How CITRUS works

The loop starts from the direct reconstruction X_0 of the uncorrected data (FDK, or the filtered back projection of a
parallel beam). Outer iteration k = 1, ..., N converts
X_{k-1} into material densities, takes one scatter step per view, corrects the measured counts with the estimated
scatter-to-primary ratio, `Y_c = Y / (1 + SPR)`, and runs warm-started MBIR iterations on the corrected data,
which give X_k; both counts are settings, and stronger scatter needs more outer iterations.

The scatter step is **age-ordered source iteration**. Per view the solver keeps an accumulator of the collided
fluence, `Phi_{>1}`, and updates it with one transport sweep `G_k` (scattering plus transport through the densities of
reconstruction k) applied to the newest first-collision term and to the previous accumulator:

```
Phi_{>1}^{(k)} = G_k phi_1^{(k)} + G_k Phi_{>1}^{(k-1)}
             = G_k phi_1^{(k)} + G_k G_{k-1} phi_1^{(k-1)} + G_k G_{k-1} G_{k-2} phi_1^{(k-2)} + ...
```

where `phi_1^{(k)}` is the first-collision source of reconstruction k (from its uncollided fluence). Expanded, the
second-order scatter comes from the newest reconstruction, the third order from the previous one, and so on: the
dominant low-order scatter is always
refreshed from the freshest reconstruction, while stale information is relegated to the weaker and smoother
higher orders. With fixed densities the recursion *is* source iteration and converges to the fully converged scatter
estimate, so the scatter and the reconstruction converge together for the cost of one sweep per view per outer
iteration.

## The scatter solver

`citrus.scatter_solver` is a deterministic LBTE solver for kV and MV photons: multigroup in energy (cross sections of
NIST XCOM, Compton and Rayleigh angular distributions of xraylib), discrete ordinates in angle with real spherical
harmonics, linear-discontinuous finite elements on a grid of transport cells swept in wavefront order, an analytic
(exact-in-energy) uncollided fluence and first-collision source on a few quadrature points per cell, source iteration
over collision order, and a collapsed-cone projection of the last-collided source onto the detector with a finite-cone
footprint, detector and anti-scatter-grid responses. It runs in PyTorch with fused Triton kernels (ray tracing,
first-collision source, sweep).

The user supplies the source spectrum (a two-column file, or one node for a monoenergetic source) and chooses the
reference materials of the density model from a catalogue of compounds, tissues and every element, or defines their
own. The scan is a cone beam with a flat or curved panel and circular or helical views, or a parallel beam; the source
may carry an emission profile measured from the open field (a bowtie filter, the heel effect), the detector response
and an anti-scatter grid (septa along one or both panel directions) are built from material data, and the rays through
metal can be left out of MBIR with a metal mask.

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
`benchmarks/` (benchmarks of the sweep, the first-collision source, the ray tracer and the scatter image, profiles of
the CITRUS step per stage and per kernel, the GPU memory of the loop, and the CITRUS study script
`citrus_study.py`).

## Quick start

```bash
python demo/demo_1_scatter_solver.py          # the scatter solver on one view of the WLAB phantom
python demo/demo_2_basic_cone.py              # a short circular cone-beam scan, 125 kVp (6 MV and a grid as options)
python demo/demo_3_helical_cone.py            # a helical scan on a curved detector behind a CT-style anti-scatter grid
python demo/demo_4_materials.py               # user-defined materials: iodine contrast and a dental amalgam filling
python demo/demo_5_parallel_monoenergetic.py  # a monoenergetic parallel beam (synchrotron CT), air gap option
python demo/demo_6_callbacks.py               # convergence curves recorded with the callbacks of citrus_recon
python demo/demo_7_scatter_schedules.py       # CITRUS against no correction, sequential, first-order, decoupled
```

The demos are short scripts that show the API step by step, with every setting (spectrum, reference materials,
panel, voxel pitch, views, loop and solver settings) a variable at the top. The first run of demo 1 downloads the
photon data and builds the cross-section library (both cached); the reconstruction demos synthesize their data once
and cache it in `demo/input/` (a few minutes on a GPU for the first run).
For studies (other iteration counts and sparse-view scatter schedules, a finer ground truth, the RMSE history and run
statistics), `benchmarks/citrus_study.py` takes the settings as flags and writes figures and a summary JSON to
`demo/output/`:

```bash
python benchmarks/citrus_study.py          # 180 views over a short scan, 0.5 cm grid, the default numbers of
                                           # outer and MBIR iterations, scatter step on every 8th view
python benchmarks/citrus_study.py --full   # 180 views over a full rotation, 0.5 cm grid, 20 outer iterations
                                           # of 1 MBIR iteration, scatter step on every view
python benchmarks/citrus_study.py --help   # every flag
```

The views are evenly spaced over `[start, end)`; the default end of a cone beam is a short scan, start + 180 degrees +
the in-plane fan angle of the detector (`utilities.view_angles`, `utilities.fan_angle`), and of a parallel beam start +
180 degrees. MBIRTORCH 0.1.2's FDK and filtered back projection, which initialize the loop, count every view equally,
so over any range that is not whole rotations (short scans and overscans) CITRUS applies redundancy weights before them
(`recon.mbir_bridge.redundancy_weights`), or leaves them to an MBIRTORCH that applies them itself; a helical scan
keeps MBIRTORCH's FDK.

In Python:

```python
from citrus import CitrusConfig, citrus_recon
from citrus.scatter_solver import load_spectrum_txt
from citrus.utilities import wlab_solver

spectrum = load_spectrum_txt("spectrum.txt")               # the source spectrum, two columns: keV, photons per keV
solver = wlab_solver(delta_voxel, geometry=geometry, spectrum=spectrum)   # ScatterSolver, library cached on disk
recon, info = citrus_recon(Y, I0, angles, grid, geometry, solver, CitrusConfig())   # 14 outer x 8 MBIR iterations
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
  note = {Version 0.24.0},
  year = 2026
}
```

GitHub's "Cite this repository" button on the repository page generates the
citation from `CITATION.cff`.

The photon data come from NIST XCOM (M.J. Berger, J.H. Hubbell, S.M. Seltzer, J. Chang, J.S. Coursey, R. Sukumar,
D.S. Zucker and K. Olsen, *XCOM: Photon Cross Sections Database*, NIST Standard Reference Database 8,
https://physics.nist.gov/xcom) and xraylib (T. Schoonjans et al., *The xraylib library for X-ray-matter interactions.
Recent developments*, Spectrochimica Acta Part B 66, 776-784 (2011)); please cite them too.
