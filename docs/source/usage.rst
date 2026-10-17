Usage and demos
===============

Demos
-----

Both demos write their figures to ``demo/output/`` (not tracked); the CITRUS demo caches its synthetic data in
``demo/input/``.

.. code-block:: bash

   python demo/demo_scatter_solver.py              # one view of the WLAB phantom
   python demo/demo_citrus.py                      # 180 views over a short scan, 0.5 cm grid, 80 outer iterations,
                                                   # scatter step on every 8th view and at every 8th outer iteration
   python demo/demo_citrus.py --end-view-angle 360 # the same over a full rotation
   python demo/demo_citrus.py --full               # 180 views over a full rotation, 0.5 cm grid, 20 outer
                                                   # iterations, scatter step on every view at every iteration
   python demo/demo_citrus.py --views 36 --delta-voxel 1 --outer-iterations 10 --scatter-view-stride 1 --scatter-every 1
                                                   # a quick run
   python demo/demo_citrus.py --materials air,water,bone   # three reference materials in the density model

``demo_scatter_solver.py`` builds the library of the reference materials (once, then cached), rasterizes the WLAB
phantom (a 30 x 20 x 40 cm water prism with lung, air and bone inserts) and solves one view with the default
configuration (6 groups, Legendre order 7, 8 x 16 sweep directions, 16 x 32 collapsed-cone directions). It plots the
primary and scatter detector signals, the scatter fraction and central-row profiles. Options: ``--delta-voxel``,
``--transport-cell-cm``, ``--angle-deg``, ``--panel small|full``, ``--grid`` (focused anti-scatter grid),
``--materials`` (comma-separated catalogue names, water and bone among them).

``demo_citrus.py`` synthesizes a cone-beam dataset (primary from MBIRTORCH's projector at the effective energy, scatter
from the solver's scatter-to-primary ratio, Poisson noise), runs CITRUS and the uncorrected MBIR loop with the same
initialization, weights, partition sequence and frozen regularization, and plots central slices, central-row profiles,
the relative RMSE per outer iteration and the estimated vs true scatter-to-primary ratio of one view; a summary JSON
records the settings, the reference materials and the run statistics. Options:

* ``--views`` (default 180), ``--delta-voxel`` (default 0.5), ``--transport-cell-cm`` (default 1, any ratio to the
  voxel pitch), ``--outer-iterations`` (default 80);
* ``--start-view-angle`` / ``--end-view-angle`` (degrees; the default end is a short scan), ``--full`` (the full
  rotation configuration above);
* ``--scatter-view-stride`` (default 8), ``--scatter-view-interpolation-order`` (``0``-``3`` or ``none``, default
  ``3``), ``--scatter-every`` (default 8);
* ``--materials`` (the reference materials of the loop's density model, comma-separated catalogue names, at least two;
  default ``air,lung,adipose,water,muscle,soft_tissue,bone``; output suffix ``_materials-<names>`` for another set);
* ``--no-resume-error-sinogram`` (restart MBIR from the sinogram every outer iteration; suffix ``_restart``);
* ``--I0``, ``--seed``, ``--no-noise``, ``--regenerate``, ``--truth-sweep``, ``--truth-cone``, ``--save-recon`` (also
  write the CITRUS volume as ``.npy``), ``--memory-cap-mib`` (default 4400).

The truth solver of the data always uses the seven default reference materials, so ``--materials`` changes only the
reconstruction loop and every set runs on the same data. The cache name ``demo/input/<tag>.npz`` and the outputs carry
the data-set format (``_v2``); files of other formats are never loaded.

Angular range: the views are evenly spaced over ``[start, end)`` (endpoint excluded, MBIRTORCH's angle convention).
The default end is a short scan, start + 180 degrees + the in-plane fan angle of the panel,
``2 atan(0.5 num_det_channels delta_det_channel / SDD)``; ``--end-view-angle 360`` (with the default start 0) is a full
rotation. For a short scan ``citrus_recon`` / ``mbir_recon`` apply Parker weights before MBIRTORCH's FDK
(``recon.mbir_bridge.fdk(..., geometry=, angles=)``, ``short_scan_weights``), and the sparse-view interpolation runs
along the arc.

Ground truth: the scatter of every view is computed by the scatter solver from the TRUE densities with a fully
converged source iteration, stored as the scatter-to-primary ratio and applied to MBIRTORCH's primary counts, then
Poisson noise is drawn. This is an intentional inverse crime for the scatter (the truth and the correction come from
the same solver); ``--truth-sweep 16x32 --truth-cone 32x64`` generates the truth with finer angular discretizations
than the loop (8x16, 16x32) to expose the solver's own discretization error.

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
and the frozen regularization. A ``callback(k, X_k, record)`` can track the iterates (the demo computes the RMSE
against the truth with it).

The scatter schedule (:doc:`method`, "Scatter step every n-th outer iteration" and "Scatter step on every m-th
view"):

.. code-block:: python

   # scatter stage at every 2nd outer iteration: 20 scatter steps, 40 MBIR iterations
   config = CitrusConfig(num_outer_iterations=40, scatter_every=2)
   # scatter step on every 4th view, the others interpolated in angle (Catmull-Rom, order 3, by default)
   config = CitrusConfig(num_outer_iterations=10, scatter_view_stride=4, scatter_view_interpolation_order=1)
   recon, info = citrus_recon(Y, I0, angles, grid, geometry, solver, config)
   info["computed_views"], info["view_interpolated"]                    # [Vc] indices, [V] bool
   [r["scatter_refreshed"] for r in info["records"][1:]]                # True at the iterations with a scatter stage

``scatter_every=n`` runs the scatter stage at the outer iterations 0, n, 2n, ... and reuses the previous
scatter-to-primary ratio in between; choose ``num_outer_iterations`` as n times the number of scatter steps you can
afford. ``scatter_view_interpolation_order``: 3 cubic Catmull-Rom on 4 neighbors (default), 2 quadratic (3-point
Lagrange), 1 linear, 0 nearest computed view; ``None`` leaves the skipped views uncorrected (diagnostics only); ignored
at ``scatter_view_stride=1``. ``scatter_view_indices``, ``angular_neighbors``, ``interpolation_weights`` and
``interpolate_scatter`` of ``citrus.citrus`` expose the view subset and the interpolation.

Other settings of ``CitrusConfig``: ``resume_error_sinogram`` (default ``True``: keep MBIRTORCH's error sinogram
between outer iterations instead of recomputing it with a forward projection; the same reconstruction to float
rounding), ``primary_views_per_chunk`` (views per chunk of the primary projections; default: one sinogram of path
lengths), ``gpu_state_budget_mib`` and ``pinned_memory_limit_gib`` (where the per-view accumulators are stored),
``energy_kev``, ``min_counts``, ``base_seed``, ``verbose``.

The per-view step is available on its own:

.. code-block:: python

   from citrus import CitrusState, citrus_step
   from citrus.citrus import citrus_view_step

   state = CitrusState()
   for k in range(n):                        # densities rho fixed: n steps = n source iterations
       scatter, primary, state = citrus_step(solver, state, rho, grid, angle)
   # the loop's form: prebuilt fields (from X, no density volume), primary computed elsewhere
   scatter, _, state = citrus_view_step(solver, state, fields, angle, primary=False)

The scatter solver
------------------

.. code-block:: python

   import torch
   from citrus.scatter_solver import (ScatterSolver, SolverConfig, build_solver_library, csi_response,
                                      no_grid)
   from citrus.utilities import wlab_setup

   grid, geometry, rho = wlab_setup(1.0, device="cuda")       # 1 cm grid, 100 x 75 panel, [7, R, C, S] densities
   config = SolverConfig()                                    # 1 cm transport cells, 0.5 cm quadrature points
   library = build_solver_library(config)                     # the reference materials config.materials, cached
   solver = ScatterSolver(library, config, geometry, detector=csi_response(0.06), grid_response=no_grid(),
                          device="cuda", dtype=torch.float32)
   result = solver.solve_view(rho, grid, angle=0.0)
   result.scatter_signal, result.primary["signal"], result.scatter_fraction   # [rows, channels]

``rho`` holds the densities (g/cm^3) of the library's reference materials (``library.materials``) on the fine grid (any
voxel size), at most two nonzero per voxel. The transport runs on the grid of ``transport_cell_cm`` cells that covers
the fine box and the first-order quantities on q\ :sup:`3` points per cell (``quadrature_point_spacing_cm``);
``uncollided_trace_grid`` ("quadrature", "coarse" or "fine") sets the grid through which the uncollided fluence is
traced (:doc:`method`, "Options and secondary components"; :doc:`conventions`, "Grids"). ``ViewResult`` also carries
the scalar and planar fluences per group, the source-iteration history, per-stage timings and peak memory.

Energy groups and the angular table
-----------------------------------

Two settings of ``SolverConfig`` shape the library (:doc:`method`, "Uncollided fluence and first-collision source" and
"Multigroup and spherical-harmonic discretization"):

* ``group_edges_method``: ``"scattered_fluence"`` (default; equal areas under the estimated source-plus-scattered
  fluence of an object of chord ``grouping_object_size_cm``, default 20 cm, made of ``grouping_medium``),
  ``"equal_fluence"`` (equal source photon counts) or ``"log"`` (geometric spacing). ``group_edges_kev`` overrides
  the rule.
* ``grouping_medium``: ``"auto"`` (default: measured on the object, water until then), a reference material's name
  (``"aluminium"`` for an aluminium casting) or a mapping ``{"water": 0.9, "bone": 0.2}`` (g/cm^3) of reference
  materials.
* ``angular_table_grid``: ``"adaptive"`` (default; ``num_eta_table`` nodes placed by the curvature of the transfer) or
  ``"uniform"``.

.. code-block:: python

   from dataclasses import replace
   from citrus.scatter_solver.cross_section_library import load_spectrum_txt

   config = replace(SolverConfig(), num_groups=12)                       # adaptive table, scattered-fluence groups
   solver = wlab_solver(0.5, config=config, spectrum=load_spectrum_txt("spectrum_6mv.txt"), device="cuda")
   legacy = replace(SolverConfig(), angular_table_grid="uniform", group_edges_method="equal_fluence")  # citrus 0.5

``wlab_solver`` (and ``ScatterSolver(None, config, ...)``) returns a solver whose library is built on first use, so
``citrus_recon`` can first measure the object: with ``CitrusConfig.grouping_object_size = "auto"`` (default) it sets
``grouping_object_size_cm`` to the mean chord of the FDK volume (``citrus.utilities.object_size_cm``) and, for
``grouping_medium = "auto"``, the medium to the object's mean densities (``grouping_medium_estimate``) before the
first scatter stage builds the library (cached per size and medium); ``info["grouping_object_size_cm"]`` and
``info["grouping_medium"]`` record them. A number in cm sets the size, a named medium skips its estimate; a solver
given a prebuilt library keeps its config. ``synthesize_cone_beam_data`` takes both from the true object before its
first solve.

Reference materials
-------------------

``SolverConfig.materials`` names the reference materials of the library and of the density model, at least two names
of the catalogue (:doc:`conventions`, "Materials"), in any order; the library sorts them by nominal attenuation
(``library.materials``) and refuses a set whose members cross in attenuation. Look density rows up by name:

.. code-block:: python

   from dataclasses import replace
   from citrus.scatter_solver.cross_sections import material_index
   from citrus.utilities import attenuation_to_densities, wlab_phantom, wlab_solver

   solver = wlab_solver(1.0, config=replace(SolverConfig(), materials=("air", "water", "bone")), device="cuda")
   rho = wlab_phantom(grid, materials=solver.materials, device="cuda")            # [3, R, C, S]
   material_index(solver.library, "water")                                       # 1
   attenuation_to_densities(solver.library, X)                                   # the density model of the loop

The WLAB phantom (and ``synthesize_wlab_dataset``) needs water and bone among the reference materials;
``cross_sections.WATER`` / ``BONE`` are the rows of the default set only.

Synthetic data
--------------

``citrus.utilities.synthesize_cone_beam_data(rho, grid, geometry, angles, solver, I0, noise, seed)`` generates
cone-beam counts with scatter for any density volume ``rho [M, R, C, S]`` (the solver library's materials): ``X_true``
(attenuation at the effective energy), line integrals, ``P``, ``S``, ``Y``, ``SPR``, the solver's primary and scatter
images and metadata. The primary uses MBIRTORCH's own projector, the scatter is ``S = SPR P`` with the
scatter-to-primary ratio of one fully converged solve of ``solver`` per view, whose primary is the loop's primary
model (``meta["data_format"]`` = 2). The truth solver may use a finer angular discretization than the solver of the
reconstruction loop. ``synthesize_wlab_dataset(num_views, delta_voxel, I0, noise, seed, out_path, config=...,
geometry=..., start_view_angle=0.0, end_view_angle=None)`` is the WLAB wrapper used by the CITRUS demo
(``end_view_angle`` None: the short scan start + pi + fan angle; ``2 pi`` for a full rotation); ``save_dataset`` /
``load_dataset`` write and read ``.npz`` + ``.json``.
