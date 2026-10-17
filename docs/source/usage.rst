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
   python demo/demo_citrus.py --views 36 --delta-voxel 1 --outer 10 --scatter-view-stride 1 --scatter-every 1
                                                   # a quick run (about 2 minutes)

``demo_scatter_solver.py`` builds the seven-material library (about 16 s, then cached), rasterizes the WLAB phantom (a
30 x 20 x 40 cm water prism with lung, air and bone inserts) and solves one view with the default configuration (6 groups, Legendre order 7, 8 x 16 sweep
directions, 16 x 32 collapsed-cone directions). It plots the primary and scatter detector signals, the scatter
fraction and central-row profiles. Options: ``--delta-voxel``, ``--transport-cell-cm``, ``--angle-deg``,
``--panel small|full``, ``--grid`` (focused anti-scatter grid).

``demo_citrus.py`` synthesizes a cone-beam dataset (primary from MBIRTORCH's projector at the effective energy, scatter
from the solver's scatter-to-primary ratio, Poisson noise), runs CITRUS and the uncorrected MBIR loop with the same
initialization, weights, partition sequence and frozen regularization, and plots central slices, central-row profiles,
the relative RMSE per outer iteration and the estimated vs true scatter-to-primary ratio of one view. Options:
``--views`` (default 180), ``--delta-voxel`` (default 0.5), ``--transport-cell-cm`` (transport cell edge, default
1 cm, any ratio to the voxel pitch), ``--outer`` (default 80), ``--I0``, ``--seed``, ``--no-noise``, ``--full``, ``--regenerate``,
``--truth-sweep``, ``--truth-cone``, ``--start-view-angle`` / ``--end-view-angle`` (degrees),
``--scatter-view-stride`` (default 8), ``--scatter-view-interp-order`` (``0``-``3`` or ``none``, default ``3``),
``--scatter-every`` (default 8).

Angular range: the views are evenly spaced over ``[start, end)`` (endpoint excluded, MBIRTORCH's angle convention).
The default end is a short scan, start + 180 degrees + the in-plane fan angle of the panel,
``2 atan(0.5 num_det_channels delta_det_channel / SDD)`` (22.6 degrees for the full-field WLAB panel, so 202.6
degrees); ``--end-view-angle 360`` (with the default start 0) is a full rotation. ``--full`` reproduces the report
study and sets the full rotation explicitly. MBIRTORCH's FDK, which initializes the loop, assumes a full rotation; for
a short scan ``citrus_recon`` / ``mbir_recon`` apply Parker weights before it (``recon.mbir_bridge.fdk(...,
geom=, angles=)``, ``short_scan_weights``), and the sparse-view interpolation runs along the arc. Measured with
36 views, a 1 cm grid, 10 outer iterations and the scatter step on every view at every iteration: relative RMSE of
CITRUS 0.064 over the short scan (uncorrected MBIR 0.29, FDK 0.31) and 0.075 over a full rotation.

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

   grid, geom, rho = wlab_setup(1.0, device="cuda")           # 1 cm grid, 100 x 75 panel, [7, R, C, S] densities
   config = SolverConfig()                                    # 1 cm transport cells, 2^3 first-collision points
   lib = build_solver_library(config)                         # cached on disk
   solver = ScatterSolver(lib, config, geom, detector=csi_response(0.06), grid_response=no_grid(),
                          device="cuda", dtype=torch.float32)
   result = solver.solve_view(rho, grid, angle=0.0)
   result.scatter_signal, result.primary["signal"], result.scatter_fraction   # [rows, channels]

``rho`` holds the densities (g/cm^3) of the seven basis materials (air, lung, adipose, water, muscle, soft tissue,
cortical bone; ``cross_sections.MATERIALS``, at most two nonzero per voxel; the WLAB phantom uses the water and bone
rows) on the fine grid (any voxel size). The transport runs on the grid of ``transport_cell_cm`` cells that covers the
fine box; the uncollided fluence, the first-collision source and the first-order scatter emission on
q\ :sup:`3` points per cell (q = ``fc_points_per_axis``, default 2, at most round(cell / voxel): 8 points for
0.5 cm and smaller voxels in 1 cm cells, 1 for 1 cm voxels); only the primary image is computed on the fine
grid (:doc:`conventions`, "Grids"). ``ViewResult`` also carries the scalar and planar fluences per group, the source-iteration history, per-stage
timings and peak memory.

CITRUS
------

.. code-block:: python

   from citrus import CitrusConfig, citrus_recon, mbir_recon
   from citrus.utilities import wlab_solver

   solver = wlab_solver(delta_voxel, geom=geom, device="cuda")          # or any ScatterSolver
   recon, info = citrus_recon(Y, I0, angles, grid, geom, solver, CitrusConfig(num_outer=10))
   baseline, _ = mbir_recon(Y, I0, angles, grid, geom, CitrusConfig(num_outer=10),
                            regularization=info["regularization"])

``Y`` are the measured counts ``[views, rows, channels]`` with flat field ``I0``; ``grid`` / ``geom`` describe the
reconstruction grid and the scanner in MBIRTORCH conventions (:doc:`conventions`). ``info`` holds the per-iteration
records (timings, accumulator change, scatter change), the final scatter-to-primary ratio and scatter-count estimate,
and the frozen regularization. Each outer iteration converts the reconstruction into densities with the seven-material
model (``citrus.utilities.attenuation_to_densities``, :doc:`method`, "CITRUS coupling"):

.. code-block:: python

   from citrus.utilities import attenuation_to_densities, densities_to_attenuation, nominal_attenuation

   rho = attenuation_to_densities(solver.lib, X)          # [7, R, C, S], exact in attenuation at E_eff
   densities_to_attenuation(solver.lib, rho)              # == X.clamp_min(0)
   nominal_attenuation(solver.lib)                        # [7] the bracketing points (1/cm at 63 keV) A ``callback(k, X_k, record)`` can track the iterates (the demo computes the RMSE
against the truth with it).

Sparse-view scatter: the scatter step can run on every n-th view only, with the other views' scatter interpolated
in angle (:doc:`method`, "Sparse-view scatter"):

.. code-block:: python

   cfg = CitrusConfig(num_outer=10, scatter_view_stride=2)             # Catmull-Rom (order 3) by default
   cfg = CitrusConfig(num_outer=10, scatter_view_stride=4, scatter_view_interp_order=1)   # linear
   recon, info = citrus_recon(Y, I0, angles, grid, geom, solver, cfg)
   info["computed_views"], info["view_interpolated"]                    # [Vc] indices, [V] bool

``scatter_view_interp_order``: 3 cubic Catmull-Rom on 4 neighbors (default), 2 quadratic (3-point Lagrange), 1 linear,
0 nearest computed view, all circular in angle for a full rotation and along the arc for a short scan; ``None`` leaves the skipped views uncorrected (diagnostics only). It
is ignored at ``scatter_view_stride=1``. Measured on the 180-view WLAB study after 20 outer iterations: stride 2 costs
+0.0014 relative RMSE for 51 % of the scatter time, stride 4 +0.008 for 26 % (two-material model of version
0.0.2). The functions
``scatter_view_indices``, ``angular_neighbors``, ``interpolation_weights`` and ``interpolate_scatter`` of
``citrus.citrus`` expose the view subset and the interpolation.

Scatter step every n outer iterations: ``scatter_every=n`` runs the scatter stage at the outer iterations 0, n, 2n,
... only and reuses the previous scatter-to-primary ratio in between (:doc:`method`, "Scatter step every n outer
iterations"):

.. code-block:: python

   cfg = CitrusConfig(num_outer=40, scatter_every=2)   # 20 scatter steps, 40 MBIR iterations
   recon, info = citrus_recon(Y, I0, angles, grid, geom, solver, cfg)
   [r["scatter_refreshed"] for r in info["records"][1:]]   # True at the iterations with a scatter stage

For a given number of scatter steps the extra MBIR iterations are nearly free (one costs about 1 % of a scatter
stage) and improve the result: on the 180-view WLAB study (two-material model of version 0.0.3) 20 scatter steps give a relative RMSE of 0.0429 at 20
outer iterations, 0.0310 at 40 (``scatter_every=2``) and 0.0286 at 80 (``scatter_every=4``) in the same wall time.
At a fixed number of outer iterations, skipping scatter stages costs accuracy (20 iterations: 0.0513 at n = 2,
0.0700 at n = 4), so choose ``num_outer`` as n times the number of scatter steps you can afford.

The per-view step is available on its own:

.. code-block:: python

   from citrus import CitrusState, citrus_step

   state = CitrusState()
   for k in range(n):                        # densities rho fixed: n steps = n source iterations
       scatter, primary, state = citrus_step(solver, state, rho, grid, angle)

Synthetic data
--------------

``citrus.utilities.synthesize_cone_beam_data(rho, grid, geometry, angles, solver, I0, noise, seed)`` generates
cone-beam counts with scatter for any density volume ``rho [7, R, C, S]``: ``X_true`` (attenuation at the effective energy),
line integrals, ``P``, ``S``, ``Y``, ``SPR``, the solver's primary and scatter images and metadata. The primary uses
MBIRTORCH's own projector, the scatter is ``S = SPR P`` with the scatter-to-primary ratio of one fully converged solve of
``solver`` per view (so the scatter-to-primary ratio of the data equals that of the solver exactly). The truth solver
may use a finer angular discretization than the solver of the reconstruction loop.
``synthesize_wlab_dataset(num_views, delta_voxel, I0, noise, seed, out_path, config=..., geom=...,
start_view_angle=0.0, end_view_angle=None)`` is the WLAB wrapper used by the CITRUS demo (``end_view_angle`` None: the
short scan start + pi + fan angle, ``view_angles`` / ``fan_angle``; ``2 pi`` for a full rotation); ``save_dataset`` / ``load_dataset`` write and read ``.npz`` + ``.json``.
