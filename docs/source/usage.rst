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
the file (180 views over a short scan, 0.5 cm voxels, adaptive numbers of outer and MBIR iterations, the scatter
step on every 8th view) the first run synthesizes the data and takes a few minutes on a GPU.

CITRUS studies
--------------

``benchmarks/citrus_study.py`` runs the same comparison with every setting as a flag and writes figures (central
slices, central-row profiles, relative RMSE per outer iteration, estimated vs true scatter-to-primary ratio of view 0),
a summary JSON (the settings, the reference materials, the object size and medium of the energy groups, the RMSE
history, the run times and the peak allocated GPU memory) and optionally the CITRUS volume to ``demo/output/``. The
data sets are cached in ``demo/input/``.

.. code-block:: bash

   python benchmarks/citrus_study.py                      # 180 views over a short scan, 0.5 cm grid, adaptive
                                                          # numbers of outer and MBIR iterations, scatter step on
                                                          # every 8th view (the setting of demo 2)
   python benchmarks/citrus_study.py --end-view-angle 360 # the same over a full rotation
   python benchmarks/citrus_study.py --full               # 180 views over a full rotation, 0.5 cm grid, 20 outer
                                                          # iterations of 1 MBIR iteration, scatter step on every view
   python benchmarks/citrus_study.py --views 36 --delta-voxel 1 --outer-iterations 10 --scatter-view-stride 1 \
       --mbir-iterations 1                                # a quick run
   python benchmarks/citrus_study.py --materials air,water,bone_cortical_icrp   # three reference materials

Flags:

* ``--views`` (default 180), ``--delta-voxel`` (default 0.5), ``--transport-cell-cm`` (default 1, any ratio to the
  voxel pitch; used by the loop's and the truth's solver), ``--outer-iterations`` and ``--mbir-iterations`` per
  outer iteration (a number each, or ``adaptive``, the default);
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

   solver = wlab_solver(delta_voxel, geometry=geometry, device="cuda", spectrum=spectrum)   # or any ScatterSolver
   recon, info = citrus_recon(Y, I0, angles, grid, geometry, solver, CitrusConfig(num_outer_iterations=10))
   baseline, _ = mbir_recon(Y, I0, angles, grid, geometry, CitrusConfig(num_outer_iterations=10),
                            regularization=info["regularization"])

``Y`` are the measured counts ``[views, rows, channels]`` with flat field ``I0``; ``grid`` / ``geometry`` describe the
reconstruction grid and the scanner in MBIRTORCH conventions (:doc:`conventions`). ``info`` holds the per-iteration
records (timings, accumulator change, scatter change), the final scatter-to-primary ratio and scatter-count estimate,
and the frozen regularization. A ``callback(k, X_k, record)`` can track the iterates per outer iteration and an
``mbir_callback(i, X, record)`` per MBIR iteration, i = 0 for the FDK initialization, with
``record["sinogram_nrmse"]`` the relative RMSE of the forward projection of X against the corrected sinogram of its
outer iteration, a convergence measure without ground truth (demo 2 plots it next to the error against the truth; the
study script computes the RMSE
against the truth with it).

The entries of ``info``:

* ``records``: one dict per outer iteration (``records[0]`` for the initial volume X_0, ``records[k]`` for X_k) with
  ``outer_iteration``, ``mbir_iterations``, ``sinogram_nrmse_start`` / ``sinogram_nrmse`` (relative RMSE of A X
  against the corrected sinogram before and after its MBIR iterations, with the resumed error sinogram),
  ``scatter_seconds``, ``mbir_seconds`` (of all its MBIR iterations),
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
  False for a prebuilt library) and ``library_seconds`` (time to build or load a lazy solver's library);
* ``emission_profile``: the source's emission profile derived from an open-field image (None for a number).

``mbir_recon`` returns the same entries without those of the scatter solver.

The flat field is a number or the measured open-field image, ``[rows, channels]`` for every view or ``[views, rows,
channels]`` per view, in the same units as the counts (a bowtie or beam filter, the heel effect). An image normalizes
the counts pixel by pixel and gives the solver's source the emission profile of the open field for the duration of the
call (:doc:`method`, "Source emission profile and open field"); a per-pixel detector gain map, if known, separates the
gain from the emission:

.. code-block:: python

   recon, info = citrus_recon(Y, open_field, angles, grid, geometry, solver, config)                  # all emission
   recon, info = citrus_recon(Y, open_field, angles, grid, geometry, solver, config, detector_gain=gain)
   info["emission_profile"]                     # [rows, channels], mean 1
   baseline, _ = mbir_recon(Y, open_field, angles, grid, geometry, config)   # the same normalization, no correction

The iteration counts and the view schedule (:doc:`method`, "MBIR iterations per outer iteration" and "Scatter step
on every m-th view"):

.. code-block:: python

   config = CitrusConfig()                                               # adaptive numbers of iterations
   # 20 outer iterations of 2 MBIR iterations: 20 scatter steps, 40 MBIR iterations
   config = CitrusConfig(num_outer_iterations=20, mbir_iterations_per_outer_iteration=2)
   # scatter step on every 4th view, the others interpolated in angle (Catmull-Rom, order 3, by default)
   config = CitrusConfig(num_outer_iterations=10, scatter_view_stride=4, scatter_view_interpolation_order=1)
   recon, info = citrus_recon(Y, I0, angles, grid, geometry, solver, config)
   info["computed_views"], info["view_interpolated"]                    # [Vc] indices, [V] bool

By default both counts are ``"adaptive"``: the MBIR iterations of an outer iteration stop once the fit of the
corrected sinogram improves by less than ``mbir_stop_relative_improvement`` per iteration (at most
``max_mbir_iterations_per_outer_iteration``), and the loop stops once an outer iteration improves the fit by less
than ``stop_relative_improvement`` (at most ``max_outer_iterations``); see :doc:`method`, "MBIR iterations per outer
iteration". ``mbir_iterations_per_outer_iteration=n`` runs n MBIR iterations in every outer iteration and
``num_outer_iterations=N`` runs N outer iterations instead. ``info["mbir_iterations_total"]`` and the records'
``mbir_iterations`` say what was run, and
``mbir_recon(..., mbir_iterations_schedule=[r["mbir_iterations"] for r in info["records"][1:]])`` repeats the same
schedule without the correction. ``scatter_view_interpolation_order``: 3 cubic Catmull-Rom on 4 neighbors
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

Settings
--------

Every setting of the loop (``CitrusConfig``) and of the solver (``SolverConfig``), with its default; the sections
above and :doc:`method` explain them, and the API pages (:doc:`api/citrus`, :doc:`api/config`) carry the same
descriptions. The first setting of each group names the group.

**CitrusConfig**

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - setting
     - default
     - meaning
   * - ``num_outer_iterations`` (loop)
     - ``"adaptive"``
     - outer iterations (one scatter correction each): a number, or adaptive on the fit of the corrected sinogram
   * - ``mbir_iterations_per_outer_iteration``
     - ``"adaptive"``
     - MBIR iterations per outer iteration: a number, or adaptive on the same fit
   * - ``stop_relative_improvement``
     - ``0.005``
     - adaptive loop: stop once an outer iteration improves the fit by less than this
   * - ``mbir_stop_relative_improvement``
     - ``0.02``
     - adaptive MBIR: stop once the fit improves by less than this per iteration (mean of the last two)
   * - ``max_outer_iterations``
     - ``40``
     - most outer iterations of the adaptive loop
   * - ``max_mbir_iterations_per_outer_iteration``
     - ``8``
     - most MBIR iterations of an adaptive outer iteration (at least two run)
   * - ``resume_error_sinogram``
     - ``True``
     - keep MBIRTORCH's error sinogram between MBIR iterations instead of recomputing it
   * - ``base_seed``
     - ``0``
     - seed of the partition sequence of MBIR iteration i, ``[base_seed, i]``
   * - ``verbose``
     - ``False``
     - print one line per outer iteration (timings, fit, accumulator change)
   * - ``scatter_view_stride`` (scatter stage)
     - ``1``
     - scatter step on the views 0, n, 2n, ...; the others are interpolated in angle
   * - ``scatter_view_interpolation_order``
     - ``3``
     - interpolation of the skipped views: 3 Catmull-Rom, 2 quadratic, 1 linear, 0 nearest, None none
   * - ``energy_kev``
     - ``None``
     - effective energy of the density model; None for the spectrum mean of the library
   * - ``grouping_object_size``
     - ``"auto"``
     - object size in cm of the solver's scattered-fluence groups; auto measures the FDK volume
   * - ``primary_views_per_chunk``
     - ``None``
     - views per chunk of the primary projections; None keeps one sinogram
   * - ``min_counts`` (data and memory)
     - ``1.0``
     - lower clamp of the (corrected) counts before the log
   * - ``gpu_state_budget_mib``
     - ``2600.0``
     - largest total of the per-view accumulators kept on the GPU, in MiB
   * - ``pinned_memory_limit_gib``
     - ``8.0``
     - largest pinned host allocation for the accumulators, in GiB

**SolverConfig**

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - setting
     - default
     - meaning
   * - ``num_groups`` (energy)
     - ``6``
     - energy groups
   * - ``group_edges_method``
     - ``"scattered_fluence"``
     - rule of the group edges: scattered_fluence, equal_fluence or log
   * - ``group_edges_kev``
     - ``None``
     - explicit descending group edges in keV (num_groups + 1 values) instead of the rule
   * - ``grouping_object_size_cm``
     - ``20.0``
     - size in cm of the object in the scattered-fluence rule (CITRUS measures it on the FDK volume)
   * - ``grouping_medium``
     - ``"auto"``
     - medium of that rule: auto (the object's mean densities), a catalogue name or (name, density) pairs
   * - ``energy_min_kev``
     - ``10.0``
     - lowest group edge; photons scattered below it are absorbed
   * - ``legendre_order``
     - ``7``
     - Legendre order L of the scattering kernel
   * - ``num_eta_table``
     - ``2001``
     - nodes of the tabulated transfer in the cosine of the scattering angle
   * - ``angular_table_grid``
     - ``"adaptive"``
     - placement of those nodes: adaptive (by the curvature of the transfer) or uniform
   * - ``materials`` (materials)
     - the seven defaults
     - reference materials of the library and the density model (catalogue names, at least two)
   * - ``num_polar`` (angle)
     - ``8``
     - Gauss-Legendre polar nodes of the sweep quadrature (at least L + 1)
   * - ``num_azimuthal``
     - ``16``
     - equally spaced azimuths of the sweep quadrature (at least 2 L + 1)
   * - ``transport_cell_cm`` (space)
     - ``1.0``
     - edge of the cubic transport cells
   * - ``quadrature_point_spacing_cm``
     - ``0.5``
     - target spacing of the quadrature points of the first-order scatter
   * - ``uncollided_trace_grid``
     - ``"quadrature"``
     - grid the uncollided fluence is traced through: quadrature, coarse or fine
   * - ``source_iteration_tolerance`` (source iteration)
     - ``1.0e-3``
     - stop a converged solve when the last order is below this fraction of the total
   * - ``max_source_iterations``
     - ``30``
     - most sweeps (collision orders) of a converged solve
   * - ``cone_num_polar`` (scatter image)
     - ``16``
     - polar directions of the cone direction set
   * - ``cone_num_azimuthal``
     - ``32``
     - azimuthal directions of the cone direction set
   * - ``direction_chunk``
     - ``8``
     - directions per batch of the scatter image and per octant per sweep chunk
   * - ``dtype`` (numerics)
     - ``"float32"``
     - float32 (production) or float64 (reference and tests)

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

Detector, grid and source
-------------------------

The detector response, the anti-scatter grid and the source model are arguments of ``ScatterSolver`` (and of
``utilities.wlab_solver``); :doc:`method`, "Detector and grid responses" and "Source emission profile and open field",
describes the models.

.. code-block:: python

   from citrus.scatter_solver import (SourceModel, anti_scatter_grid, bowtie_emission_profile, detector_response,
                                      emission_profile_from_open_field, tabulated_detector_response)

   detector = detector_response("cesium_iodide", 0.06)                           # energy integrating, full deposition
   detector = detector_response("gadolinium_oxysulfide", 0.02, deposition="local")   # thin screen, MV beams
   detector = detector_response("cadmium_telluride", 0.2, mode="photon_counting")
   detector = tabulated_detector_response(energies_kev, response)                # e.g. from a Monte Carlo run
   septa = anti_scatter_grid(geometry, septum_thickness_cm=0.002, septum_height_cm=0.15)   # lead, one septum per
   septa = anti_scatter_grid(geometry, 0.002, 0.15, septa_along="rows", septum_material="tungsten")  # pixel
   source = SourceModel(emission_profile=bowtie_emission_profile(geometry, edge_weight=0.2))
   source = SourceModel(emission_profile=emission_profile_from_open_field(open_field, solver, detector_gain=gain))
   solver = ScatterSolver(library, config, geometry, detector=detector, grid_response=septa, source=source,
                          device="cuda")

``detector_response`` takes any catalogue material (:doc:`conventions`, "Materials") with its catalogue density or
``density=``, ``mode`` "energy_integrating" (keV per photon) or "photon_counting", and ``deposition`` "full" or
"local". ``anti_scatter_grid`` takes the septum material, thickness and height, ``focal_distance_cm`` (default: the
source-detector distance), ``pitch_cm`` (default: the pixel pitch across the septa) and ``septa_along`` ("channels",
the default, or "rows"). ``csi_response`` and ``focused_1d_grid`` remain the built-in CsI and lead models.
``SourceModel(emission_profile=...)`` takes a ``[rows, channels]`` profile, or ``[views, rows, channels]`` with
``emission_profile_angles``. ``citrus_recon`` sets the profile itself from an open-field image.

Energy groups and the angular table
-----------------------------------

Two settings of ``SolverConfig`` shape the library (:doc:`method`, "Uncollided fluence and first-collision source" and
"Multigroup and spherical-harmonic discretization"):

* ``group_edges_method``: ``"scattered_fluence"`` (default; equal areas under the estimated source-plus-scattered
  fluence of an object of size ``grouping_object_size_cm``, default 20 cm, made of ``grouping_medium``),
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
``grouping_object_size_cm`` to the size measured from the escape probability of the FDK volume
(``citrus.utilities.object_size_cm``) and, for
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

The scatter-to-primary ratio of the data is that of the solver's detector signals, through the detector response and
behind the grid, as in the CITRUS loop (with the ideal counting detector and no grid of ``wlab_solver`` the signal is
the planar fluence). The truth solver's source model sets the emission profile of the data and
``detector_gain=`` a per-pixel gain map; ``synthesize_wlab_dataset`` passes ``detector=``, ``grid_response=``,
``source=`` and ``detector_gain=`` on. The data's ``open_field`` is the flat field to give ``citrus_recon``: the
number ``I0`` for a uniform source without a gain map, else the image ``I0 g f / mean(f)`` of the gain :math:`g` and
the truth solver's open-field signal :math:`f` with its emission profile, and the primary counts are
``P = open_field exp(-A X)``:

.. code-block:: python

   source = SourceModel(emission_profile=bowtie_emission_profile(geometry, 0.1))
   septa = anti_scatter_grid(geometry, 0.002, 0.15)
   data = synthesize_wlab_dataset(180, 0.5, source=source, detector_gain=gain, grid_response=septa,
                                  geometry=geometry, spectrum=spectrum)
   recon, info = citrus_recon(data["Y"], data["open_field"], data["angles"], grid, geometry, solver,
                              detector_gain=gain)

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
