Overview
========

Package layout
--------------

``citrus.citrus``
    The CITRUS algorithm: :func:`~citrus.citrus.citrus_recon` (the coupled reconstruction loop),
    :func:`~citrus.citrus.mbir_recon` (the same MBIR loop without the solver, for comparison),
    :func:`~citrus.citrus.citrus_step` (one scatter step of one view) and its operators ``G1``, ``G1to2``,
    ``G_gt2``, ``G_step``, ``project_scatter``; the density model and the compact per-view state storage.

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
    ``mbir_bridge`` (MBIRTORCH ``ConeBeamModel`` matching the solver's geometry, FDK, the warm-started
    single-iteration recipe) and ``synthetic`` (the synthetic WLAB dataset).

``citrus.utilities``
    The WLAB phantom, grid and geometry, the attenuation <-> density model, a GPU memory cap and plotting helpers.

Scope of the solver
-------------------

* Photon energies of kV imaging (default 10-125 keV, 6 equal-fluence groups); Compton scattering (no Doppler
  broadening) and Rayleigh scattering from EPDL97 via XrayPhysics; photoelectric absorption as local deposition.
* Two basis materials (water and ICRP cortical bone); every voxel is a mixture given by two density maps.
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
    Benchmarks of the sweep, the first-collision source kernel and the ray tracer, and a per-stage profile of the
    CITRUS step.
