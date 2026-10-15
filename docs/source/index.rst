citrus
======

**CITRUS** (Coupled Iterative Tomographic reconstruction and Recursively-Updated Scatter) reconstructs cone-beam CT volumes from scatter-contaminated projections by running a deterministic
scatter solver *inside* the iterative reconstruction. Every outer iteration performs one scatter step per view
with the **LBTE source iteration solver** (a linear Boltzmann transport equation solver: multigroup in energy,
discrete ordinates in angle, linear-discontinuous finite elements in space, source iteration over collision order,
and a collapsed-cone projection of the scattered photons onto the detector) and one warm-started MBIR iteration
with `MBIRTORCH <https://github.com/cabouman/mbirtorch>`_ on the scatter-corrected data.

The scatter estimate and the reconstruction converge together: the per-view collided fluence is carried from one
outer iteration to the next, so the expensive transport is spread over the reconstruction instead of being
repeated to convergence for every new volume.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   installation
   usage
   method
   conventions
   design_notes
   results
   api
