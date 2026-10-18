Usage and demos
===============

Demos
-----

Two quick-start scripts in ``demo/`` show how to write a Python script with the CITRUS API, one step per comment.

.. code-block:: bash

   python demo/demo_1_scatter_solver.py   # the scatter solver on one view of the WLAB phantom
   python demo/demo_2_citrus.py           # CITRUS vs uncorrected MBIR on synthetic data

``demo_1_scatter_solver.py`` builds the cross-section library of the default reference materials (on first use it
downloads the NIST XCOM photon data and builds the library; both are cached), rasterizes the WLAB phantom (a 30 x 20 x
40 cm water prism with lung, air and bone inserts) on a 1 cm grid, solves one view with the default configuration and
shows the primary signal, the scatter signal and the scatter fraction on the detector.

``demo_2_citrus.py`` defines every setting in a parameter block at the top: the spectrum (loaded from a two-column
file, energy in keV and relative photons per keV, the sample 125 kVp spectrum of ``demo/spectra``, or a monoenergetic
line from ``monoenergetic_spectrum``), the reference materials, the
detector panel and the source distances, the voxel pitch of the WLAB phantom, the number of views and the angular
range, the flat field, the CITRUS loop settings (outer iterations, ``mbir_iterations_per_outer_iteration``,
``scatter_view_stride``) and the
solver settings (transport cell size, energy groups, Legendre order). It synthesizes scatter-contaminated cone-beam
counts with ``synthesize_cone_beam_data`` (cached in ``demo/input/`` under a name built from the settings, so later
runs load them), builds the scatter solver from ``SolverConfig`` and the spectrum, reconstructs with ``citrus_recon``
and, for comparison, with ``mbir_recon`` from the same FDK volume with the same regularization, prints the relative
RMSE of both against the true attenuation and shows the central slices and central-row profiles. With the values in
the file (180 views over a short scan, 0.5 cm voxels, 80 outer iterations with the scatter step on every 8th view at
every 8th iteration) the first run synthesizes the data and takes a few minutes on a GPU.

CITRUS studies
--------------

``benchmarks/citrus_study.py`` runs the same comparison with every setting as a flag and writes figures (central
slices, central-row profiles, relative RMSE per outer iteration, estimated vs true scatter-to-primary ratio of view 0),
a summary JSON (the settings, the reference materials, the object size and medium of the energy groups, the RMSE
history, the run times and the peak allocated GPU memory) and optionally the CITRUS volume to ``demo/output/``. The
data sets are cached in ``demo/input/``.

.. code-block:: bash

   python benchmarks/citrus_study.py                      # 180 views over a short scan, 0.5 cm grid, 10 outer
                                                          # iterations of 8 MBIR iterations, scatter step on every
                                                          # 8th view (the setting of demo 2)
   python benchmarks/citrus_study.py --end-view-angle 360 # the same over a full rotation
   python benchmarks/citrus_study.py --full               # 180 views over a full rotation, 0.5 cm grid, 20 outer
                                                          # iterations of 1 MBIR iteration, scatter step on every view
   python benchmarks/citrus_study.py --views 36 --delta-voxel 1 --outer-iterations 10 --scatter-view-stride 1 \
       --mbir-iterations 1                                # a quick run
   python benchmarks/citrus_study.py --materials air,water,bone_cortical_icrp   # three reference materials

Flags:

* ``--views`` (default 180), ``--delta-voxel`` (default 0.5), ``--transport-cell-cm`` (default 1, any ratio to the
  voxel pitch; used by the loop's and the truth's solver), ``--outer-iterations`` (default 10),
  ``--mbir-iterations`` per outer iteration (default 8);
* ``--start-view-angle`` / ``--end-view-angle`` (degrees; the default end is a short scan), ``--full`` (the full
  rotation configuration above);
* ``--scatter-view-stride`` (default 8), ``--scatter-view-interpolation-order`` (``0``-``3`` or ``none``, default
  ``3``);
* ``--materials`` (the reference materials of the loop's density model, comma-separated catalogue names, at least two;
  default ``air,lung_icrp,adipose_icrp,water,muscle_icru44,soft_tissue_icru44,bone_cortical_icrp``);
* ``--no-resume-error-sinogram`` (restart MBIR from the sinogram every outer iteration);
* ``--I0`` (flat-field counts, default 1e5), ``--seed``, ``--no-noise``, ``--regenerate``, ``--truth-sweep``,
  ``--truth-cone``, ``--save-recon`` (also write the CITRUS volume as ``.npy``), ``--memory-cap-mib`` (PyTorch CUDA
  allocator cap; default 85 percent of the device memory, 0 for none).

Names: the data set is ``demo/input/<tag>.npz`` (with ``<tag>.json``), the tag recording the number of views, the voxel
pitch, the angular range (``_short`` for the default short scan), ``_nonoise``, a transport cell size other than 1 cm,
a finer truth discretization and the data-set format (``_v2``); files of other formats are never loaded. The outputs
are ``demo/output/citrus_<tag>[suffixes]_<figure>.png`` and ``citrus_<tag>[suffixes].json``, with the suffixes
``_materials-<names>`` (a set of reference materials other than the default), ``_restart``
(``--no-resume-error-sinogram``), ``_stride<n>`` (with ``_order<k>`` for an interpolation order other than 3) and
``_mbir<n>`` for the values other than 1.

The truth solver of the data always uses the seven default reference materials (the phantom is water and bone only),
so ``--materials`` changes only the reconstruction loop and every set runs on the same data. Both reconstructions use
the same frozen regularization (MBIRTORCH's automatic estimate on the scatter-free sinogram), the same FDK
initialization, weights and partition sequence, so they differ only by the scatter correction.

Angular range: the views are evenly spaced over ``[start, end)`` (endpoint excluded, MBIRTORCH's angle convention).
The default end is a short scan, start + 180 degrees + the in-plane fan angle of the panel,
``2 atan(0.5 num_det_channels delta_det_channel / SDD)``; ``--end-view-angle 360`` (with the default start 0) is a full
rotation. For a short scan ``citrus_recon`` / ``mbir_recon`` apply Parker weights before MBIRTORCH's FDK
(``recon.mbir_bridge.fdk(..., geometry=, angles=)``, ``short_scan_weights``), and the sparse-view interpolation runs
along the arc.

Ground truth: the scatter of every view is computed by the scatter solver from the TRUE densities with a fully
converged source iteration, stored as the scatter-to-primary ratio and applied to MBIRTORCH's primary counts, then
Poisson noise is drawn. This is an intentional inverse crime for the scatter (the truth and the correction come from
the same solver), so the study measures how well the coupled loop converges, not the accuracy of the scatter model;
``--truth-sweep 16x32 --truth-cone 32x64`` generates the truth with finer angular discretizations than the loop (8x16,
16x32) to expose the solver's own discretization error.

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
and the frozen regularization. A ``callback(k, X_k, record)`` can track the iterates (the study script computes the RMSE
against the truth with it).

The entries of ``info``:

* ``records``: one dict per outer iteration (``records[0]`` for the initial volume X_0, ``records[k]`` for X_k) with
  ``outer_iteration``, ``mbir_iterations``, ``scatter_seconds``, ``mbir_seconds`` (of all its MBIR iterations),
  ``total_seconds``, ``scatter_change_relative`` (mean absolute change of the scatter-count estimate
  over its mean absolute value) and, on CUDA, ``gpu_peak_allocated_mib``; a record with a scatter stage also holds
  ``negative_fraction`` (voxels clamped by the density model), ``accumulator_change_mean`` /
  ``accumulator_change_max`` (relative accumulator change over the computed views), ``transfer_seconds``,
  ``primary_seconds`` and, at ``scatter_view_stride > 1``, ``interpolation_seconds`` (the last two are part of
  ``scatter_seconds``);
* ``spr`` and ``scatter_counts``: the last scatter-to-primary ratio and scatter-count estimate ``Y - Y_c``,
  ``[views, rows, channels]``;
* ``regularization`` (the frozen sigmas), ``state`` (a description of the accumulator storage, one slot per computed
  view), ``init_recon``, ``model``, ``config``, ``seconds`` and ``fdk_seconds``;
* ``computed_views`` (``[Vc]`` views with a scatter step) and ``view_interpolated`` (``[V]`` bool, True where the
  scatter is interpolated in angle);
* ``grouping_object_size_cm``, ``grouping_medium`` (the object size and medium of the solver's energy groups, None
  when its group rule is not ``"scattered_fluence"``), ``grouping_applied`` (whether ``citrus_recon`` set them;
  False for a prebuilt library) and ``library_seconds`` (time to build or load a lazy solver's library).

``mbir_recon`` returns the same entries without those of the scatter solver.

The scatter schedule (:doc:`method`, "Scatter step every n-th outer iteration" and "Scatter step on every m-th
view"):

.. code-block:: python

   # 20 outer iterations of 2 MBIR iterations: 20 scatter steps, 40 MBIR iterations
   config = CitrusConfig(num_outer_iterations=20, mbir_iterations_per_outer_iteration=2)
   # scatter step on every 4th view, the others interpolated in angle (Catmull-Rom, order 3, by default)
   config = CitrusConfig(num_outer_iterations=10, scatter_view_stride=4, scatter_view_interpolation_order=1)
   recon, info = citrus_recon(Y, I0, angles, grid, geometry, solver, config)
   info["computed_views"], info["view_interpolated"]                    # [Vc] indices, [V] bool

``mbir_iterations_per_outer_iteration=n`` runs n MBIR iterations on the corrected counts of every outer iteration
before the next scatter correction. ``scatter_view_interpolation_order``: 3 cubic Catmull-Rom on 4 neighbors
(default), 2 quadratic (3-point
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
  (``"aluminium"`` for an aluminium casting) or a mapping ``{"water": 0.9, "bone_cortical_icrp": 0.2}`` (g/cm^3) of
  reference
  materials.
* ``angular_table_grid``: ``"adaptive"`` (default; ``num_eta_table`` nodes placed by the curvature of the transfer) or
  ``"uniform"``.
  The medium is stored as (name, density) pairs sorted by name; the library cache key and the balance use the
  densities rounded to 3 significant digits, with traces below 1e-3 of the largest density dropped
  (``cross_section_library.grouping_medium_densities``), so nearly equal media share a cached library.

.. code-block:: python

   from dataclasses import replace
   from citrus.scatter_solver import load_spectrum_txt

   config = replace(SolverConfig(), num_groups=12)                       # adaptive table, scattered-fluence groups
   spectrum = load_spectrum_txt("demo/spectra/spectrum_6mv.txt")         # a 6 MV linac spectrum
   solver = wlab_solver(0.5, config=config, spectrum=spectrum, device="cuda")
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

   solver = wlab_solver(1.0, config=replace(SolverConfig(), materials=("air", "water", "bone_cortical_icrp")),
                        device="cuda", spectrum=spectrum)
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
geometry=..., start_view_angle=0.0, end_view_angle=None)`` is the WLAB wrapper used by the demo and the study script
(``end_view_angle`` None: the short scan start + pi + fan angle; ``2 pi`` for a full rotation); ``save_dataset`` /
``load_dataset`` write and read ``.npz`` + ``.json``.

The primary in the denominator of the synthetic scatter-to-primary ratio is the planar fluence summed over the
groups (``citrus.citrus.primary_signals(..., quantity="planar_fluence")``), in photons/cm^2, rather than the
detector signal.

The solver's images are in photons/cm^2 per source photon/sr; its primary carries the inverse-square and obliquity
factors of the point source and the polychromatic transmission. MBIRTORCH's model is the flat, monochromatic line
integral :math:`l = AX` at the effective energy with ``I0`` the flat-field count, which absorbs the inverse-square and
obliquity factor ``phi_air`` (the solver's primary of an empty volume, saved with the data). The data tie the two
models through the dimensionless per-pixel ratio, ``S = SPR P``: the scatter-to-primary ratio of the data equals the
solver's physical ratio, and the CITRUS correction :math:`Y / (1 + \mathrm{SPR}(X))` does not depend on ``I0`` or on
the monochromatic / polychromatic primary mismatch. The density volume itself is not stored (it is M fine volumes;
``WlabPhantom`` rebuilds it).

The Poisson draws use ``torch.Generator(seed)``, but the rates vary between runs at the float32 rounding level
(atomic accumulation in the CUDA kernels of MBIRTORCH and of the solver), which flips a few draws; save and reuse a
data set when bitwise identity matters.
