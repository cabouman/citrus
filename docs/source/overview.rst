Overview
========

Package layout
--------------

``citrus.citrus``
    The CITRUS algorithm: :func:`~citrus.citrus.citrus_recon` (the coupled reconstruction loop),
    :func:`~citrus.citrus.mbir_recon` (the same MBIR loop without the solver, for comparison),
    :func:`~citrus.citrus.citrus_step` (one scatter step of one view; ``citrus_view_step`` from prebuilt fields) and its
    operators ``uncollided_operator``, ``first_collision_operator``, ``multiple_collision_operator``,
    ``accumulator_update_operator``, ``project_scatter``; the primaries of all views (``primary_signals``, MBIRTORCH
    projections of the material maps); the density model (one material map at a time) and the compact per-view
    state storage.

``citrus.scatter_solver``
    The LBTE source iteration solver. :class:`~citrus.scatter_solver.scatter_solver.ScatterSolver` assembles the
    stages of one view; the modules hold the stages:

    ============================= ================================================================================
    ``config``                    ``SolverConfig``: every discretization parameter
    ``geometry``                  ``Grid``, ``ConeGeometry``, ``ParallelGeometry`` (MBIRTORCH conventions)
    ``cross_sections``            the material catalogue, the multigroup library container, per-voxel coefficients,
                                  a synthetic test library
    ``photon_data``               the photon data: NIST XCOM cross sections (downloaded on first use), xraylib
                                  angular distributions
    ``cross_section_library``     the library builder from those data, cached on disk
    ``source``, ``detector``      point source with emission profile; detector and anti-scatter-grid responses
    ``angular_quadrature``        direction sets, real spherical harmonics, the scattering-source operator
    ``linear_discontinuous_cell`` the linear-discontinuous cell equations and their closed-form solve
    ``wavefront``                 the diagonal-plane ordering of the sweep
    ``transport_sweep``           the batched linear-discontinuous discrete-ordinates sweep
    ``source_iteration``          source iteration over collision order
    ``uncollided``                uncollided fluence and first-collision source
    ``scatter_image``             collapsed-cone scatter image, finite-cone footprint, primary image
    ``ray_tracing``               exact Siddon ray tracer behind the ``Projectors`` interface
    ``kernels``                   fused Triton kernels (float32 CUDA) with PyTorch reference paths
    ============================= ================================================================================

``citrus.recon``
    ``mbir_bridge``: the MBIRTORCH ``ConeBeamModel`` matching the solver's geometry, FDK (with redundancy weights for a
    short scan or an overscan), the warm-started single-iteration recipe.

``citrus.utilities``
    The WLAB phantom, grid and geometry, the seven-material attenuation <-> density model, the view angles (short
    scan by default), synthetic cone-beam data with scatter (``synthesize_cone_beam_data`` for any density volume,
    ``synthesize_wlab_dataset``), a GPU memory cap and plotting helpers.

Scope of the solver
-------------------

* Photon energies from kV imaging to MV beams (default 10-125 keV in 6 groups whose edges follow the scattered
  photons, any loaded spectrum); Compton scattering (no Doppler broadening) and Rayleigh scattering with the cross
  sections of NIST XCOM and the angular distributions of xraylib, with an angular table whose nodes follow the forward
  peaks; photoelectric absorption (and pair production at MV) as local deposition.
* Reference materials chosen from a catalogue of compounds, tissues and all elements (``SolverConfig.materials``,
  :doc:`conventions`, "Materials"); every voxel is a mixture of at most two of them, the two whose nominal
  attenuations bracket its attenuation.
* A point source collimated to the flat panel; detector and anti-scatter-grid responses as lookup tables.
* Float32 on CUDA (production; fused Triton kernels) and float64 / CPU (reference and tests).

Repository
----------

``demo/``
    Six quick-start scripts (:doc:`usage`, "Demos"): ``demo_1_scatter_solver.py`` (the scatter solver on one view of
    the WLAB phantom), ``demo_2_citrus.py`` (CITRUS on a circular cone-beam scan), ``demo_3_helical.py`` (a helical
    scan on a curved detector with an anti-scatter grid), ``demo_4_materials.py`` (user-defined materials),
    ``demo_5_parallel_beam.py`` (a monoenergetic parallel beam) and ``demo_6_callbacks.py`` (convergence curves from
    the callbacks).
``tests/``
    The test suite (``python -m pytest -q``), including oracle tests: a 1-D slab linear-discontinuous solution,
    particle balance, dense Galerkin assembly, Beer's law, brute-force point-to-point scatter integrals, and the
    identity "N CITRUS steps with fixed densities = N source iterations".
``dev_scripts/``
    Shell scripts (bash; Linux, macOS, Git Bash on Windows): ``create_conda_test_environment.sh``,
    ``clean_install_all.sh``, ``remove_build.sh``, ``deep_clean.sh``, ``run_tests.sh`` (``--slow`` for the slow tests),
    ``build_docs.sh``, ``check_cuda.sh`` (PyTorch / CUDA / Triton availability and the kernel self-checks).
``benchmarks/``
    Benchmarks of the sweep, the first-collision source kernel and the ray tracer, a per-stage profile of the CITRUS
    step, and ``citrus_study.py``, the CITRUS study script with every setting as a flag (:doc:`usage`).
