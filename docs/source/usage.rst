Usage and demos
===============

Demos
-----

Both demos write their figures to ``demo/output/`` (not tracked); the CITRUS demo caches its synthetic data in
``demo/input/``.

.. code-block:: bash

   python demo/demo_scatter_solver.py              # one view of the WLAB phantom (seconds)
   python demo/demo_citrus.py                      # 180 views over a short scan, 0.5 cm grid, 80 outer iterations,
                                                   # scatter step on every 8th view and at every 8th outer iteration
                                                   # (about 4 minutes)
   python demo/demo_citrus.py --end-view-angle 360 # the same over a full rotation
   python demo/demo_citrus.py --full               # the report study: 180 views over a full rotation, 0.5 cm grid,
                                                   # 20 outer iterations, scatter step on every view at every iteration
   python demo/demo_citrus.py --views 36 --delta-voxel 1 --outer-iterations 10 --scatter-view-stride 1 --scatter-every 1
                                                   # a quick run (about 2 minutes)

``demo_scatter_solver.py`` builds the seven-material library (about 16 s, then cached), rasterizes the WLAB phantom (a
30 x 20 x 40 cm water prism with lung, air and bone inserts) and solves one view with the default configuration (6
groups, Legendre order 7, 8 x 16 sweep directions, 16 x 32 collapsed-cone directions). It plots the primary and scatter
detector signals, the scatter fraction and central-row profiles. Options: ``--delta-voxel``, ``--transport-cell-cm``,
``--angle-deg``, ``--panel small|full``, ``--grid`` (focused anti-scatter grid).

``demo_citrus.py`` synthesizes a cone-beam dataset (primary from MBIRTORCH's projector at the effective energy, scatter
from the solver's scatter-to-primary ratio, Poisson noise), runs CITRUS and the uncorrected MBIR loop with the same
initialization, weights, partition sequence and frozen regularization, and plots central slices, central-row profiles,
the relative RMSE per outer iteration and the estimated vs true scatter-to-primary ratio of one view. Options:
``--views`` (default 180), ``--delta-voxel`` (default 0.5), ``--transport-cell-cm`` (transport cell edge, default 1 cm,
any ratio to the voxel pitch), ``--outer-iterations`` (default 80), ``--I0``, ``--seed``, ``--no-noise``, ``--full``,
``--regenerate``, ``--truth-sweep``, ``--truth-cone``, ``--start-view-angle`` / ``--end-view-angle`` (degrees),
``--scatter-view-stride`` (default 8), ``--scatter-view-interpolation-order`` (``0``-``3`` or ``none``, default ``3``),
``--scatter-every`` (default 8), ``--primary-projector`` (``mbirtorch``, default, or ``siddon``; output suffix
``_siddon``), ``--no-resume-error-sinogram`` (restart MBIR from the sinogram every outer iteration; suffix
``_restart``), ``--save-recon`` (also write the CITRUS volume as ``.npy``), ``--memory-cap-mib`` (default 4400). The
summary JSON records the peak allocated GPU memory of the CITRUS run (whole run and per outer iteration). At the
defaults (0.5 cm grid, 180 views, stride 8, every 8th iteration, 80 outer iterations, about 4 minutes plus about 2
minutes to synthesize the data once) CITRUS reaches a relative RMSE of 0.0215 / 0.0259 (whole prism / core; 0.0257 /
0.0305 with ``--primary-projector siddon`` on the same data, see :doc:`method`, "CITRUS coupling") with a peak of 1.64
GB allocated.

Data sets: ``--truth-primary-projector`` (``mbirtorch``, default, or ``siddon``) sets the primary model of the data's
scatter-to-primary ratio (below). The cache name ``demo/input/<tag>.npz`` carries the data-set format, ``_v2``
(citrus >= 0.3.0: SPR with the MBIRTORCH primary, the loop's default), and ``_primary-siddon`` for the Siddon
definition; the outputs carry the same tag. Files named ``..._v1...`` are data sets and results of citrus <= 0.2.0
(SPR with the Siddon primary); the demo never loads them.

Angular range: the views are evenly spaced over ``[start, end)`` (endpoint excluded, MBIRTORCH's angle convention).
The default end is a short scan, start + 180 degrees + the in-plane fan angle of the panel,
``2 atan(0.5 num_det_channels delta_det_channel / SDD)`` (22.6 degrees for the full-field WLAB panel, so 202.6
degrees); ``--end-view-angle 360`` (with the default start 0) is a full rotation. ``--full`` reproduces the report
study and sets the full rotation explicitly. MBIRTORCH's FDK, which initializes the loop, assumes a full rotation; for
a short scan ``citrus_recon`` / ``mbir_recon`` apply Parker weights before it (``recon.mbir_bridge.fdk(...,
geometry=, angles=)``, ``short_scan_weights``), and the sparse-view interpolation runs along the arc. Measured with
36 views, a 1 cm grid, 10 outer iterations and the scatter step on every view at every iteration: relative RMSE of
CITRUS 0.064 over the short scan (uncorrected MBIR 0.29, FDK 0.31) and 0.075 over a full rotation (version 0.1.0,
data-set format 1: SPR with the Siddon primary).

Ground truth of the CITRUS demo: the scatter of every view is computed by the scatter solver from the TRUE densities
with a fully converged source iteration, stored as the scatter-to-primary ratio and applied to MBIRTORCH's primary
counts, then Poisson noise is drawn. This is an intentional inverse crime for the scatter (the truth and the
correction come from the same solver); ``--truth-sweep 16x32 --truth-cone 32x64`` generates the truth with finer
angular discretizations than the loop (8x16, 16x32) to expose the solver's own discretization error.

The scatter solver
------------------

.. code-block:: python

   import torch
   from citrus.scatter_solver import (ScatterSolver, SolverConfig, build_solver_library, csi_response,
                                      no_grid)
   from citrus.utilities import wlab_setup

   grid, geometry, rho = wlab_setup(1.0, device="cuda")           # 1 cm grid, 100 x 75 panel, [7, R, C, S] densities
   config = SolverConfig()                                    # 1 cm transport cells, 2^3 first-collision points
   library = build_solver_library(config)                         # cached on disk
   solver = ScatterSolver(library, config, geometry, detector=csi_response(0.06), grid_response=no_grid(),
                          device="cuda", dtype=torch.float32)
   result = solver.solve_view(rho, grid, angle=0.0)
   result.scatter_signal, result.primary["signal"], result.scatter_fraction   # [rows, channels]

``rho`` holds the densities (g/cm^3) of the seven basis materials (air, lung, adipose, water, muscle, soft tissue,
cortical bone; ``cross_sections.MATERIALS``, at most two nonzero per voxel; the WLAB phantom uses the water and bone
rows) on the fine grid (any voxel size). The transport runs on the grid of ``transport_cell_cm`` cells that covers the
fine box; the uncollided fluence, the first-collision source and the first-order scatter emission on q\ :sup:`3` points
per cell (q = round(cell / ``quadrature_point_spacing_cm``), points about 0.5 cm apart by default, at most round(cell /
voxel): 8 points for 0.5 cm and smaller voxels in 1 cm cells, 64 in 2 cm cells, 1 for 1 cm voxels in 1 cm cells);
``uncollided_trace_grid`` ("quadrature", "coarse" or "fine") sets the grid through which the uncollided fluence is
traced; only the primary image is computed on the fine grid (:doc:`conventions`, "Grids"). ``ViewResult`` also carries
the scalar and planar fluences per group, the source-iteration history, per-stage timings and peak memory.

CITRUS
------

.. code-block:: python

   from citrus import CitrusConfig, citrus_recon, mbir_recon
   from citrus.utilities import wlab_solver

   solver = wlab_solver(delta_voxel, geometry=geometry, device="cuda")          # or any ScatterSolver
   recon, info = citrus_recon(Y, I0, angles, grid, geometry, solver, CitrusConfig(num_outer_iterations=10))
   baseline, _ = mbir_recon(Y, I0, angles, grid, geometry, CitrusConfig(num_outer_iterations=10),
                            regularization=info["regularization"])

``Y`` are the measured counts ``[views, rows, channels]`` with flat field ``I0``; ``grid`` / ``geometry`` describe the
reconstruction grid and the scanner in MBIRTORCH conventions (:doc:`conventions`). ``info`` holds the per-iteration
records (timings, accumulator change, scatter change), the final scatter-to-primary ratio and scatter-count estimate,
and the frozen regularization. Each outer iteration converts the reconstruction into densities with the seven-material
model (``citrus.utilities.attenuation_to_densities``, :doc:`method`, "CITRUS coupling"), one material map at a time so
that the seven-material volume never exists on the reconstruction grid:

.. code-block:: python

   from citrus.citrus import DensityModel
   from citrus.utilities import attenuation_to_densities, densities_to_attenuation, nominal_attenuation

   rho = attenuation_to_densities(solver.library, X)          # [7, R, C, S], exact in attenuation at E_eff
   densities_to_attenuation(solver.library, rho)              # == X.clamp_min(0)
   nominal_attenuation(solver.library)                        # [7] the bracketing points (1/cm at 63 keV)
   density = DensityModel.from_library(solver.library)
   density.material(X, 6)                                 # == rho[6] bit for bit, [R, C, S], slab by slab
   fields = solver.material_fields_from_attenuation(X, grid, density)   # == material_fields(rho, grid), no rho

A ``callback(k, X_k, record)`` can track the iterates (the demo computes the RMSE against the truth with it).

Primary and MBIR resume (:doc:`method`, "CITRUS coupling"):

.. code-block:: python

   # MBIRTORCH primary, MBIR resumed (defaults)
   config = CitrusConfig(num_outer_iterations=10)
   config = CitrusConfig(num_outer_iterations=10, primary_projector="siddon")    # the solver's Siddon primary per view
   config = CitrusConfig(num_outer_iterations=10, resume_error_sinogram=False)   # restart MBIR from the sinogram
   # projection chunks (default: one sinogram)
   config = CitrusConfig(num_outer_iterations=10, primary_views_per_chunk=30)

``primary_projector``: the primaries of all views come from one projection per material present, MBIRTORCH's
(``"mbirtorch"``, default) or the solver's Siddon rays (``"siddon"``), then the solver's polychromatic transmission
(``citrus.citrus.primary_signals``). ``resume_error_sinogram`` keeps MBIRTORCH's error sinogram between outer
iterations (one sinogram of memory) instead of recomputing it with a forward projection: the same reconstruction to
float rounding, 6-13 % less MBIR time on the 12 M-voxel WLAB grid (one forward projection less per iteration).

Sparse-view scatter: the scatter step can run on every n-th view only, with the other views' scatter interpolated
in angle (:doc:`method`, "Sparse-view scatter"):

.. code-block:: python

   config = CitrusConfig(num_outer_iterations=10, scatter_view_stride=2)        # Catmull-Rom (order 3) by default
   config = CitrusConfig(num_outer_iterations=10, scatter_view_stride=4,
                         scatter_view_interpolation_order=1)                     # linear
   recon, info = citrus_recon(Y, I0, angles, grid, geometry, solver, config)
   info["computed_views"], info["view_interpolated"]                    # [Vc] indices, [V] bool

``scatter_view_interpolation_order``: 3 cubic Catmull-Rom on 4 neighbors (default), 2 quadratic (3-point Lagrange), 1
linear, 0 nearest computed view, all circular in angle for a full rotation and along the arc for a short scan; ``None``
leaves the skipped views uncorrected (diagnostics only). It is ignored at ``scatter_view_stride=1``. Measured on the
180-view WLAB study after 20 outer iterations: stride 2 costs
+0.0014 relative RMSE for 51 % of the scatter time, stride 4 +0.008 for 26 % (two-material model of version
0.0.2). The functions
``scatter_view_indices``, ``angular_neighbors``, ``interpolation_weights`` and ``interpolate_scatter`` of
``citrus.citrus`` expose the view subset and the interpolation.

Scatter step every n outer iterations: ``scatter_every=n`` runs the scatter stage at the outer iterations 0, n, 2n,
... only and reuses the previous scatter-to-primary ratio in between (:doc:`method`, "Scatter step every n outer
iterations"):

.. code-block:: python

   config = CitrusConfig(num_outer_iterations=40, scatter_every=2)   # 20 scatter steps, 40 MBIR iterations
   recon, info = citrus_recon(Y, I0, angles, grid, geometry, solver, config)
   [r["scatter_refreshed"] for r in info["records"][1:]]   # True at the iterations with a scatter stage

For a given number of scatter steps the extra MBIR iterations are nearly free (one costs about 1 % of a scatter stage)
and improve the result: on the 180-view WLAB study (two-material model of version 0.0.3) 20 scatter steps give a
relative RMSE of 0.0429 at 20 outer iterations, 0.0310 at 40 (``scatter_every=2``) and 0.0286 at 80
(``scatter_every=4``) in the same wall time. At a fixed number of outer iterations, skipping scatter stages costs
accuracy (20 iterations: 0.0513 at n = 2, 0.0700 at n = 4), so choose ``num_outer_iterations`` as n times the number of
scatter steps you can afford.

The per-view step is available on its own:

.. code-block:: python

   from citrus import CitrusState, citrus_step
   from citrus.citrus import citrus_view_step

   state = CitrusState()
   for k in range(n):                        # densities rho fixed: n steps = n source iterations
       scatter, primary, state = citrus_step(solver, state, rho, grid, angle)
   # the loop's form: prebuilt fields (from X, no density volume), primary computed elsewhere
   scatter, _, state = citrus_view_step(solver, state, fields, angle, primary=False)

Synthetic data
--------------

``citrus.utilities.synthesize_cone_beam_data(rho, grid, geometry, angles, solver, I0, noise, seed)`` generates cone-beam
counts with scatter for any density volume ``rho [7, R, C, S]``: ``X_true`` (attenuation at the effective energy), line
integrals, ``P``, ``S``, ``Y``, ``SPR``, the solver's primary and scatter images and metadata. The primary uses
MBIRTORCH's own projector, the scatter is ``S = SPR P`` with the scatter-to-primary ratio of one fully converged solve
of ``solver`` per view (so the scatter-to-primary ratio of the data equals that of the solver exactly). The SPR's
primary follows ``primary_projector``: ``"mbirtorch"`` (default since 0.3.0, the loop's default primary model:
MBIRTORCH's per-material line integrals through the solver's polychromatic transmission) or ``"siddon"`` (the solver's
``solve_view`` primary, the definition up to 0.2.0); ``meta["primary_projector"]`` and ``meta["data_format"]`` (2)
record it. The truth solver may use a finer angular discretization than the solver of the reconstruction loop.
``synthesize_wlab_dataset(num_views, delta_voxel, I0, noise, seed, out_path, config=..., geometry=...,
start_view_angle=0.0, end_view_angle=None)`` is the WLAB wrapper used by the CITRUS demo (``end_view_angle`` None: the
short scan start + pi + fan angle, ``view_angles`` / ``fan_angle``; ``2 pi`` for a full rotation); ``save_dataset`` /
``load_dataset`` write and read ``.npz`` + ``.json``.
