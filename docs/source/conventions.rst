Conventions
===========

Every module follows these conventions.

Units
-----

* Length: cm. Energy: keV. Attenuation: 1/cm. Mass coefficients: cm^2/g. Density: g/cm^3.
* Angles: radians everywhere inside the package (degrees only at the XrayPhysics call boundary and in demo options).
* Source strength: 1 photon per steradian at the focal spot (spectrum fractions sum to 1).
* Angular fluence: photons / (cm^2 sr) per source photon/sr. Scatter and primary images are reported both as photon
  fluence at the panel (photons/cm^2) and as detector signal (keV deposited per cm^2) after the detector / grid
  response.

Frames and tensor layouts (MBIRTORCH conventions)
-------------------------------------------------

* Volume tensors: ``[..., num_rows, num_cols, num_slices]``. Column index j runs along +x, row index i along +y,
  slice index k along +z. Per-group volumes are ``[G, rows, cols, slices]``.
* Voxel centers (recon frame): ``x = dx * (j - (num_cols - 1)/2)``, ``y = dy * (i - (num_rows - 1)/2)``,
  ``z = dz * (k - (num_slices - 1)/2) + recon_slice_offset``, with ``dx = delta_voxel``,
  ``dy = voxel_row_aspect * delta_voxel``, ``dz = voxel_slice_aspect * delta_voxel``.
* View frame for view angle ``a``: ``x_v = cos(a) x - sin(a) y``, ``y_v = sin(a) x + cos(a) y``, ``z_v = z``.
  A positive angle carries +x toward +y.
* Cone beam, in the view frame: source at ``(0, +SID, 0)``, rays travel toward -y, flat panel in the plane
  ``y_v = SID - SDD``. Panel coordinates: ``u`` along +x_v (channels), ``v`` along +z (rows). Magnification
  ``SDD/SID``; per-point magnification ``SDD/(SID - y_v)``.
* Panel pixel centers: ``u = (c - (num_det_channels - 1)/2) * delta_det_channel - det_channel_offset``,
  ``v = (r - (num_det_rows - 1)/2) * delta_det_row - det_row_offset``.
* Detector images: ``[..., num_det_rows, num_det_channels]``. Sinograms: ``(views, rows, channels)``.
* The solver works in the recon frame per view: ``ConeGeometry`` returns the source position, panel center and panel
  axes rotated into the recon frame. Directions are unit vectors in the recon frame unless a name says ``_view``.

Grids
-----

The solver uses three grids (``geometry.Grid``):

* **F**, the fine (reconstruction) grid: the densities ``rho_fine``; only the primary image and the resampling
  below touch it. In the CITRUS loop the fine densities are never formed as a whole: the fields come from the
  attenuation X on F one material map at a time (``ScatterSolver.material_fields_from_attenuation``;
  ``MaterialFields.rho_fine`` is then None and ``MaterialFields.fine_material(m)`` evaluates a map on demand).
* **C**, the transport grid, ``Grid.covering(F, transport_cell_cm)``: cubic cells of edge
  ``SolverConfig.transport_cell_cm``, ``ceil(extent / cell)`` cells per axis, centred on the box of F (symmetric
  padding with empty space); any ratio of cell to voxel size. Every scatter order is transported on C; the
  attenuation of the scatter image is traced on C.
* **Q**, the quadrature grid, ``C.refine(q)`` with ``q = min(max(1, round(transport_cell_cm /
  quadrature_point_spacing_cm)), max(1, round(transport_cell_cm / largest fine voxel edge)))``
  (``SolverConfig.quadrature_points``; Python's round, halves to even; the default spacing 0.5 cm gives 2 points per
  axis in 1 cm cells, 4 in 2 cm cells, 1 in 0.5 cm cells): q^3 equal-weight points
  per cell of C at local coordinates ``(2j + 1)/q - 1`` (the midpoint lattice). The uncollided fluence, the
  first-collision source and the first-order emission of the scatter image live on Q.
* ``rho_coarse`` and ``rho_quadrature`` are conservative overlap resamplings of ``rho_fine``
  (``geometry.resample_conservative``): each target cell gets the exact average of the voxel-wise constant fine
  field over its volume (zero outside the fine box), so every material's mass is conserved. For an integer ratio
  with aligned boxes this is the block mean.

Energy groups
-------------

* Groups are indexed g = 0..G-1 from HIGHEST to LOWEST energy; ``group_edges`` has G+1 entries, descending.
  Downscatter goes from lower to higher index. Transfer tensors are ``[..., g_in, g_out, ...]`` and are zero for
  ``g_out < g_in`` (no upscatter).

Scattering convention
---------------------

* Per-steradian transfer ``mu_s(Omega' -> Omega) = sum_l (2l+1)/(4 pi) mu_s_l P_l(Omega . Omega')``, so ``mu_s_0``
  is the group-to-group scattering coefficient. Legendre index l = 0..L.
* Real orthonormal spherical harmonics ``Y_lk`` (polar axis +z, no Condon-Shortley phase), flat index
  ``n = l*l + (k + l)``, n = 0..(L+1)^2-1; ``sum_m w_m Y_n(Omega_m) Y_n'(Omega_m) = delta_nn'`` up to degree L.
* Moments ``phi_n = sum_m w_m Y_n(Omega_m) psi_m``; scattering source
  ``S(Omega) = sum_g' sum_n mu_s_l(n) phi_n(g') Y_n(Omega)``.
* Octant code: ``(omega_x < 0) * 4 + (omega_y < 0) * 2 + (omega_z < 0)``; a zero component counts as non-negative.

Linear-discontinuous cell moments
---------------------------------

* Per cell, 4 spatial moments ordered ``[0, x, y, z]`` = mean, u-slope, v-slope, w-slope with local coordinates
  ``u = 2(x - x_c)/dx`` etc. in [-1, 1]; ``f_0 = (1/V) int f dV``, ``f_a = (3/V) int f u_a dV``.
* Linear-discontinuous moment fields: ``[G, 4, rows, cols, slices]``; spherical-harmonic moments of linear-discontinuous
  fields: ``[G, N_sh, 4, rows, cols, slices]``.

Materials
---------

* Material catalogue ``cross_sections.MATERIAL_CATALOGUE``: name -> (nominal density g/cm^3, composition) for air,
  lung, adipose, water, muscle, soft_tissue, bone (cortical, ICRP), trabecular_bone, brain, blood, cartilage, skin,
  pmma, polyethylene, aluminium, titanium and carbon_fibre (sources in the module docstring and :doc:`method`).
* Reference materials: the library is built for ``SolverConfig.materials`` (at least two catalogue names, any order),
  sorted by increasing attenuation at the nominal density (the order must hold at every library energy). Default: the
  seven ``cross_sections.MATERIALS``, in library order 0 air (0.0012 g/cm^3), 1 lung (0.26), 2 adipose (0.95), 3 water
  (1.0), 4 skeletal muscle (1.05), 5 soft tissue (1.06), 6 cortical bone (ICRP composition, 1.85); ``WATER = 3`` and
  ``BONE = 6`` are the rows of the DEFAULT set; for any library look a row up by name,
  ``cross_sections.material_index(library, "water")`` (the WLAB phantom and the data generator do).
* Density maps ``rho: [M, rows, cols, slices]`` in g/cm^3, M = ``library.num_materials``, rows in the order of
  ``library.materials``. All coefficients are linear in rho: ``mu_x[g] = sum_m rho[m] * library.mu_x[m, g]`` with
  per-unit-density (mass) coefficients; the library carries ``materials`` (names) and ``nominal_density [M]``.
* At most two nonzero materials per voxel of ``rho_fine`` (the density model's output). The resampled ``rho_quadrature``
  can mix more materials at a point (a quadrature cell straddling a boundary): the first-collision source uses the
  compact pair representation ``MaterialPairs`` (``index [2, ...]`` int32, ``density [2, ...]``) when every point of
  ``rho_quadrature`` has at most two nonzero materials and evaluates all M materials densely otherwise; the choice is
  made once per density field (``ScatterSolver.material_fields`` -> ``MaterialFields.pairs_quadrature``, None for
  dense).

Device and dtype
----------------

* Every public function accepts tensors on any device and returns tensors on the same device. Default dtype float32;
  float64 works when the inputs are float64 (used by the oracle tests). No module hard-codes a device.
