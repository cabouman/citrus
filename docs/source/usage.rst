Usage and demos
===============

Demos
-----

Both demos write their figures to ``demo/output/`` (not tracked); the CITRUS demo caches its synthetic data in
``demo/input/``.

.. code-block:: bash

   python demo/demo_scatter_solver.py              # one view of the WLAB phantom (seconds)
   python demo/demo_citrus.py                      # 36 views, 1 cm grid, 10 outer iterations (about 2 minutes)
   python demo/demo_citrus.py --full               # 180 views, 0.5 cm grid, 20 outer iterations

``demo_scatter_solver.py`` builds the library, rasterizes the WLAB phantom (a 30 x 20 x 40 cm water prism with lung,
air and bone inserts) and solves one view with the default configuration (6 groups, Legendre order 7, 8 x 16 sweep
directions, 16 x 32 collapsed-cone directions). It plots the primary and scatter detector signals, the scatter
fraction and central-row profiles. Options: ``--delta-voxel``, ``--supervoxel-factor``, ``--angle-deg``,
``--panel small|full``, ``--grid`` (focused anti-scatter grid).

``demo_citrus.py`` synthesizes a cone-beam dataset (primary from MBIRTORCH's projector at the effective energy, scatter
from the solver's scatter-to-primary ratio, Poisson noise), runs CITRUS and the uncorrected MBIR loop with the same
initialization, weights, partition sequence and frozen regularization, and plots central slices, central-row profiles,
the relative RMSE per outer iteration and the estimated vs true scatter-to-primary ratio of one view. Options:
``--views``, ``--delta-voxel``, ``--outer``, ``--I0``, ``--seed``, ``--no-noise``, ``--full``, ``--regenerate``.

The scatter solver
------------------

.. code-block:: python

   import torch
   from citrus.scatter_solver import (ScatterSolver, SolverConfig, build_solver_library, csi_response,
                                      no_grid)
   from citrus.utilities import wlab_setup

   grid, geom, rho = wlab_setup(1.0, device="cuda")           # 1 cm grid, 100 x 75 panel, [2, R, C, S] densities
   config = SolverConfig(supervoxel_factor=1)                 # 1 cm transport cells on the 1 cm grid
   lib = build_solver_library(config)                         # cached on disk
   solver = ScatterSolver(lib, config, geom, detector=csi_response(0.06), grid_response=no_grid(),
                          device="cuda", dtype=torch.float32)
   result = solver.solve_view(rho, grid, angle=0.0)
   result.scatter_signal, result.primary["signal"], result.scatter_fraction   # [rows, channels]

``rho`` holds the water and bone densities (g/cm^3) on the fine grid, whose shape must be divisible by
``supervoxel_factor``; the transport runs on the supervoxel grid, the primary and the scatter projection on the fine
grid. ``ViewResult`` also carries the scalar and planar fluences per group, the source-iteration history, per-stage
timings and peak memory.

CITRUS
------

.. code-block:: python

   from citrus import CitrusConfig, citrus_recon, mbir_recon
   from citrus.recon.synthetic import wlab_solver

   solver = wlab_solver(delta_voxel, geom=geom, device="cuda")          # or any ScatterSolver
   recon, info = citrus_recon(Y, I0, angles, grid, geom, solver, CitrusConfig(num_outer=10))
   baseline, _ = mbir_recon(Y, I0, angles, grid, geom, CitrusConfig(num_outer=10),
                            regularization=info["regularization"])

``Y`` are the measured counts ``[views, rows, channels]`` with flat field ``I0``; ``grid`` / ``geom`` describe the
reconstruction grid and the scanner in MBIRTORCH conventions (:doc:`conventions`). ``info`` holds the per-iteration
records (timings, accumulator change, scatter change), the final scatter-to-primary ratio and scatter-count estimate,
and the frozen regularization. A ``callback(k, X_k, record)`` can track the iterates (the demo computes the RMSE
against the truth with it).

The per-view step is available on its own:

.. code-block:: python

   from citrus import CitrusState, citrus_step

   state = CitrusState()
   for k in range(n):                        # densities rho fixed: n steps = n source iterations
       scatter, primary, state = citrus_step(solver, state, rho, grid, angle)

Synthetic data
--------------

``citrus.recon.synthetic.synthesize_wlab_dataset(num_views, delta_voxel, I0, noise, seed, out_path, geom=...)``
generates the WLAB dataset of the CITRUS demo: ``X_true`` (attenuation at the effective energy), ``rho_true``,
line integrals, ``P``, ``S``, ``Y``, ``SPR``, the solver's primary and scatter images and metadata. The scatter is
``S = SPR P`` with the solver's physical scatter-to-primary ratio, so the scatter-to-primary ratio of the data equals
that of the solver exactly; the primary uses MBIRTORCH's own projector (the only model error the reconstruction sees is
the scatter and the noise).
