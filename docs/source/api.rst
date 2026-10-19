API reference
=============

One page per module; each page renders the module docstring and the public members.

CITRUS
------

The reconstruction loop and its utilities.

:doc:`api/citrus`
    The CITRUS loop, the per-view scatter step and its operators, the density model and the state storage.

:doc:`api/utilities`
    The WLAB phantom and geometries, the density model, the object size and medium of the energy groups, view
    angles, synthetic data and plotting helpers.

.. toctree::
   :hidden:

   api/citrus
   api/utilities

MBIRTORCH bridge
----------------

The interface to the reconstruction package.

:doc:`api/mbir_bridge`
    The MBIRTORCH model of the solver's geometry, FDK or the filtered back projection of a parallel beam with
    redundancy weights, warm-started MBIR iterations.

.. toctree::
   :hidden:

   api/mbir_bridge

Scatter solver
--------------

The LBTE source iteration solver, one module per stage.

:doc:`api/scatter_solver`
    ``ScatterSolver``: the assembled solver for one view.

:doc:`api/config`
    ``SolverConfig``: every discretization parameter of the solver.

:doc:`api/geometry`
    Transport and quadrature grids, the cone-beam and parallel-beam geometries and the conservative resampling
    between grids.

:doc:`api/cross_sections`
    The material catalogue, the multigroup library container, per-voxel coefficients and a synthetic test library.

:doc:`api/cross_section_library`
    The multigroup library builder from the photon data, cached on disk.

:doc:`api/photon_data`
    The photon data: NIST XCOM cross sections (downloaded on first use) and xraylib angular distributions.

:doc:`api/source`
    The source: spectrum, emission profile and collimation of the focal spot or the parallel beam.

:doc:`api/detector`
    Detector and anti-scatter-grid responses.

:doc:`api/angular_quadrature`
    Direction sets, real spherical harmonics and the scattering-source operator.

:doc:`api/linear_discontinuous_cell`
    The linear-discontinuous cell equations and their closed-form solve.

:doc:`api/wavefront`
    The diagonal-plane (wavefront) ordering of the sweep.

:doc:`api/transport_sweep`
    The batched linear-discontinuous discrete-ordinates sweep.

:doc:`api/source_iteration`
    Source iteration over collision order and the multiple-collision source.

:doc:`api/uncollided`
    The uncollided fluence and the first-collision source.

:doc:`api/scatter_image`
    The collapsed-cone scatter image, the finite-cone footprint and the primary image.

:doc:`api/ray_tracing`
    The exact Siddon ray tracer behind the ``Projectors`` interface.

.. toctree::
   :hidden:

   api/scatter_solver
   api/config
   api/geometry
   api/cross_sections
   api/cross_section_library
   api/photon_data
   api/source
   api/detector
   api/angular_quadrature
   api/linear_discontinuous_cell
   api/wavefront
   api/transport_sweep
   api/source_iteration
   api/uncollided
   api/scatter_image
   api/ray_tracing

Triton kernels
--------------

Fused GPU kernels of the scatter solver, with PyTorch reference paths.

:doc:`api/triton_utilities`
    Shared Triton helpers: the availability probe and the kernel launch path.

:doc:`api/ray_tracing_kernels`
    Fused Triton kernels of the ray tracer.

:doc:`api/uncollided_kernels`
    Fused Triton kernel of the first-collision source.

:doc:`api/sweep_kernels`
    Fused Triton kernel of the sweep's per-plane cell solve.

.. toctree::
   :hidden:

   api/triton_utilities
   api/ray_tracing_kernels
   api/uncollided_kernels
   api/sweep_kernels
