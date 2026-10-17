Overview
========

Package layout
--------------

``citrus.citrus``
    The CITRUS algorithm: :func:`~citrus.citrus.citrus_recon` (the coupled reconstruction loop),
    :func:`~citrus.citrus.mbir_recon` (the same MBIR loop without the solver, for comparison),
    :func:`~citrus.citrus.citrus_step` (one scatter step of one view; ``citrus_view_step`` from prebuilt fields) and its
    operators ``G1``, ``G1to2``, ``G_gt2``, ``G_step``, ``project_scatter``; the primaries of all views
    (``primary_signals`` with MBIRTORCH's or the Siddon projector); the density model (one material map at a time) and
    the compact per-view state storage.

``citrus.scatter_solver``
    The LBTE source iteration solver. :class:`~citrus.scatter_solver.scatter_solver.ScatterSolver` assembles the
    stages of one view; the modules hold the stages:

    ========================== ===================================================================================
    ``config``                 ``SolverConfig``: every discretization parameter
    ``geometry``               ``Grid``, ``ConeGeometry`` (MBIRTORCH conventions)
    ``cross_sections``         multigroup library container, per-voxel coefficients, a synthetic test library
    ``cross_section_library``  the library builder from XrayPhysics (EPDL97) data, cached on disk
    ``source``, ``detector``   point source with collimation; detector and anti-scatter-grid responses
    ``angular_quadrature``     direction sets, real spherical harmonics, the scattering-source operator
    ``ld_cell``                the linear-discontinuous cell equations and their closed-form solve
    ``wavefront``              the diagonal-plane ordering of the sweep
    ``transport_sweep``        the batched LD discrete-ordinates sweep
    ``source_iteration``       source iteration over collision order
    ``uncollided``             uncollided fluence and first-collision source
    ``scatter_image``          collapsed-cone scatter image, finite-cone footprint, primary image
    ``ray_tracing``            exact Siddon ray tracer behind the ``Projectors`` interface
    ``kernels``                fused Triton kernels (float32 CUDA) with PyTorch reference paths
    ========================== ===================================================================================

``citrus.recon``
    ``mbir_bridge``: the MBIRTORCH ``ConeBeamModel`` matching the solver's geometry, FDK (with Parker weights for a
    short scan), the warm-started single-iteration recipe.

``citrus.utilities``
    The WLAB phantom, grid and geometry, the seven-material attenuation <-> density model, the view angles (short
    scan by default), synthetic cone-beam data with scatter (``synthesize_cone_beam_data`` for any density volume,
    ``synthesize_wlab_dataset``), a GPU memory cap and plotting helpers.

Scope of the solver
-------------------

* Photon energies of kV imaging (default 10-125 keV, 6 equal-fluence groups); Compton scattering (no Doppler
  broadening) and Rayleigh scattering from EPDL97 via XrayPhysics; photoelectric absorption as local deposition.
* Seven basis materials (air, lung, adipose, water, skeletal muscle, soft tissue, ICRP cortical bone); every voxel
  is a mixture of at most two of them, the two whose nominal attenuations bracket its attenuation (the library, the
  solver and the kernels are general in the number of materials).
* A point source collimated to the flat panel; detector and anti-scatter-grid responses as lookup tables.
* Float32 on CUDA (production; fused Triton kernels) and float64 / CPU (reference and tests).

Repository
----------

``demo/``
    ``demo_scatter_solver.py`` (one view of the WLAB phantom) and ``demo_citrus.py`` (CITRUS vs uncorrected MBIR).
``tests/``
    The test suite (``python -m pytest -q``), including oracle tests: a 1-D slab LD solution, particle balance,
    dense Galerkin assembly, Beer's law, brute-force point-to-point scatter integrals, and the identity "N CITRUS
    steps with fixed densities = N source iterations".
``dev_scripts/``
    Shell scripts (bash; Linux, macOS, Git Bash on Windows): ``create_conda_test_environment.sh``,
    ``clean_install_all.sh``, ``remove_build.sh``, ``deep_clean.sh``, ``run_tests.sh`` (``--slow`` for the slow tests),
    ``build_docs.sh``, ``check_cuda.sh`` (PyTorch / CUDA / Triton availability and the kernel self-checks).
``benchmarks/``
    Benchmarks of the sweep, the first-collision source kernel and the ray tracer, and a per-stage profile of the
    CITRUS step.
