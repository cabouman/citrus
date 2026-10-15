# citrus

**CITRUS** reconstructs cone-beam CT volumes from scatter-contaminated projections by running a deterministic
scatter solver inside the iterative reconstruction. Each outer iteration performs one scatter step per view with the
**LBTE source iteration solver** and one warm-started MBIR iteration
([MBIRTORCH](https://github.com/cabouman/mbirtorch)) on the scatter-corrected data, so the scatter estimate and the
reconstruction converge together.

The scatter solver (`citrus.scatter_solver`) solves the linear Boltzmann transport equation for kV photons:
multigroup in energy (cross sections from EPDL97 via XrayPhysics), discrete ordinates in angle with real spherical
harmonics, linear-discontinuous finite elements on a supervoxel grid with a wavefront sweep, source iteration over
collision order, and a collapsed-cone projection of the last-collided source onto the detector, with detector and
anti-scatter-grid responses. It runs in PyTorch, with fused Triton kernels on float32 CUDA.

Full documentation: [citrus.readthedocs.io](https://citrus.readthedocs.io).

## Install

```bash
git clone git@github.com:cabouman/citrus.git
cd citrus
pip install .
```

This installs the dependencies automatically (install PyTorch first for a specific CUDA version). Additionally:

- **XrayPhysics** builds the cross-section library (once; it is cached in `~/.cache/citrus`):
  `git clone https://github.com/kylechampley/XrayPhysics && cd XrayPhysics && pip install .`
  (needs a C++ compiler and CMake; see the installation page of the documentation).
- **MBIRTORCH** for the reconstruction (`citrus_recon`, the demos).
- **Triton** (optional; `triton-windows` on Windows) for the fused GPU kernels. Without it the PyTorch paths are used.

## Quick start

```bash
python demo/demo_scatter_solver.py   # the scatter solver on one view of the WLAB phantom (seconds)
python demo/demo_citrus.py           # CITRUS vs uncorrected MBIR: 36 views, 1 cm grid, 10 outer iterations (~2 min)
python demo/demo_citrus.py --full    # 180 views, 0.5 cm grid, 20 outer iterations
```

Figures are written to `demo/output/`.

```python
from citrus import CitrusConfig, citrus_recon
from citrus.recon.synthetic import wlab_solver

solver = wlab_solver(delta_voxel, geom=geom)       # ScatterSolver with the cached library
recon, info = citrus_recon(Y, I0, angles, grid, geom, solver, CitrusConfig(num_outer=10))
```

Tests: `python -m pytest -q` (GPU, Triton, MBIRTORCH and XrayPhysics tests are skipped when unavailable).

## Citation

Please cite the software when referencing this package.

```bibtex
@misc{citrus,
  title = {citrus},
  author = {Jordan Houri and Charles A. Bouman},
  howpublished = {Software library available from \url{https://github.com/cabouman/citrus}},
  note = {Version 0.0.1},
  year = 2026
}
```

GitHub's "Cite this repository" button on the repository page generates the
citation from `CITATION.cff`.
