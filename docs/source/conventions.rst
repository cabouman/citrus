Conventions
===========

Every module follows these conventions.

Units
-----

* Length: cm. Energy: keV. Attenuation: 1/cm. Mass coefficients: cm^2/g. Density: g/cm^3.
* Angles: radians everywhere inside the package (degrees only in the options of the study script); scattering
  angles enter the cross sections as :math:`\eta = \cos\theta`.
* Source strength: 1 photon per steradian at the focal spot (spectrum fractions sum to 1); a parallel beam carries
  a fluence of 1 photon per cm^2 over its cross-section, and its images are per unit incident fluence.
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
* Cone beam, in the view frame: source at ``(0, +SID, 0)`` (circular views), rays travel toward -y, flat panel in the
  plane ``y_v = SID - SDD``. Panel coordinates: ``u`` along +x_v (channels), ``v`` along +z (rows). Magnification
  ``SDD/SID``; per-point magnification ``SDD/(SID - y_v)``.
* Panel pixel centers: ``u = (c - (num_det_channels - 1)/2) * delta_det_channel - det_channel_offset``,
  ``v = (r - (num_det_rows - 1)/2) * delta_det_row - det_row_offset``.
* Curved panel (``ConeGeometry.use_curved_detector``, MBIRTORCH's ``use_curved_detector``): a cylinder of radius SDD
  about the vertical axis through the source. The channel coordinate ``u`` is the arc length, positive toward
  +x_v, so the pixel's azimuth from the central ray is ``gamma = u / SDD``; the row coordinate ``v`` is measured on
  the plane tangent to the cylinder at the central ray, so pixel (u, v) sits at
  ``(SDD sin(gamma), SID - SDD cos(gamma), v cos(gamma))`` in the view frame, where the ray through the tangent-plane
  point ``(SDD tan(gamma), SID - SDD, v)`` meets the cylinder. A point projects to
  ``u = SDD atan2(x_v, SID - y_v)`` and ``v = SDD z_v / (SID - y_v)``, exactly MBIRTORCH's projector. The panel
  normal at a pixel is horizontal and points to the cylinder axis; ``ConeGeometry.panel_axes`` are the axes at the
  panel center and ``local_panel_axes`` those at any panel point.
* Helical views: the source and the panel of view angle ``a`` are shifted along +z by ``z_shift(a)``, so in the view
  frame the source sits at ``(0, SID, z_shift)`` and ``v`` is measured from ``z = z_shift``; MBIRTORCH's
  ``helical_z_shifts`` has this sign ("the object moves toward -z by the shift"). The geometry holds the shifts as
  a table of (unwrapped, strictly monotone) view angles and shifts (``helical_view_angles``,
  ``helical_z_shifts``, built with ``with_helical_z_shifts`` or ``utilities.helical_scan``), interpolated linearly
  between its angles, so every function that takes a view angle places the view at its z. A scan is helical when
  the shift varies over the table (``is_helical``).
* Parallel beam (``ParallelGeometry``, MBIRTORCH's ``ParallelBeamModel``): in the view frame every ray travels along
  -y_v and the flat panel lies in the plane ``y_v = -axis_detector_dist``, with ``u`` along +x_v and ``v`` along +z as
  above, so a point projects to ``u = x_v``, ``v = z``. The beam direction is ``-n_hat``
  (``ParallelGeometry.beam_direction``) and its cross-section is the panel's area. MBIRTORCH gives detector row r the
  reconstruction slice r, so a scan to be reconstructed has the rows of its grid (``ParallelGeometry.from_grid``): as
  many rows as slices, ``delta_det_row = delta_voxel``, ``voxel_slice_aspect = 1`` and ``det_row_offset =
  -recon_slice_offset``, which puts row r at the z of slice r (``recon.mbir_bridge.make_parallel_model`` checks it).
  The primary ray of a point starts upstream of the traced grid's box, ``p - (p . w + D) w`` with the beam direction w
  and D the radius of the box's circle about the axis plus 1 cm (``ParallelGeometry.ray_starts``).
* Detector images: ``[..., num_det_rows, num_det_channels]``. Sinograms: ``(views, rows, channels)``.
* The solver works in the recon frame per view: ``ConeGeometry`` returns the source position, panel center and panel
  axes rotated into the recon frame. Directions are unit vectors in the recon frame unless a name says ``_view``.
* The geometry interface: the solver reaches the scan only through ``ConeGeometry`` methods of a view angle:
  ``source_position``, ``panel_center``, ``panel_axes``, ``local_panel_axes``, ``pixel_centers``,
  ``subpixel_centers``, ``panel_points``, ``panel_distance_along``, ``panel_uv_of_points``, ``panel_half_extent``
  and ``z_shift``; the Triton ray tracer receives the same placement as scalars
  (``ray_tracing_kernels._pixel_geometry_scalars``) and the panel shape as the ``CURVED`` constant.
  ``ParallelGeometry`` has the same methods except ``source_position``, whose roles ``beam_direction`` and
  ``ray_starts`` take (``is_parallel`` tells the two apart); the kernels receive the beam direction and the distance
  of the start plane as scalars and the ``PARALLEL`` constant. Every function that takes a ``ConeGeometry`` accepts a
  ``ParallelGeometry``.

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

* Groups are indexed g = 0..G-1 from HIGHEST to LOWEST energy; ``group_edges`` has G+1 entries, descending, from the
  spectrum maximum to ``energy_min_kev`` (placed by ``SolverConfig.group_edges_method`` unless given explicitly).
  Downscatter goes from lower to higher index. Transfer tensors are ``[..., g_in, g_out, ...]`` and are zero for
  ``g_out < g_in`` (no upscatter).

Scattering convention
---------------------

* Per-steradian transfer ``mu_s(Omega' -> Omega) = sum_l (2l+1)/(4 pi) mu_s_l P_l(Omega . Omega')``, so ``mu_s_0``
  is the group-to-group scattering coefficient. Legendre index l = 0..L.
* Direct table ``mu_s_eta [M, G, G, N]`` at the nodes ``eta_grid [N]``, ascending from -1 to 1. Uniform table
  (``eta_map`` None): equally spaced nodes. Adaptive table: ``eta_map [K]`` holds the table coordinate ``u`` at the
  knots ``q_k = k / (K - 1)`` of ``q = sin(theta/2) = sqrt((1 - eta)/2)``, decreasing from 1 (eta = 1) to 0
  (eta = -1), linear in q between knots; node j sits at ``u = j / (N - 1)``
  (``cross_sections.angular_table_position``, ``angular_table_nodes``).
* Real orthonormal spherical harmonics ``Y_lk`` (polar axis +z, no Condon-Shortley phase), flat index
  ``n = l*l + (k + l)``, n = 0..(L+1)^2-1; ``sum_m w_m Y_n(Omega_m) Y_n'(Omega_m) = delta_nn'`` up to degree L.
* Moments ``phi_n = sum_m w_m Y_n(Omega_m) psi_m``; scattering source
  ``S(Omega) = sum_g' sum_n mu_s_l(n) phi_n(g') Y_n(Omega)``.
* Octant code: ``(omega_x < 0) * 4 + (omega_y < 0) * 2 + (omega_z < 0)``; a zero component counts as non-negative.
* Explicitly, with the polar angle theta from +z and the azimuth phi from +x toward +y,
  ``Y_l0 = N_l0 P_l(cos theta)``, ``Y_l,+k = sqrt(2) N_lk P_l^k(cos theta) cos(k phi)`` and
  ``Y_l,-k = sqrt(2) N_lk P_l^k(cos theta) sin(k phi)`` for k = 1..l, with ``N_lk = sqrt((2l+1)/(4 pi) (l-k)!/(l+k)!)``
  and ``P_l^k`` the associated Legendre function without the ``(-1)^k`` Condon-Shortley factor (``P_1^1 = +sin theta``).
  Hence ``Y_1,+1 = sqrt(3/(4 pi)) x``, ``Y_1,-1 = sqrt(3/(4 pi)) y``, ``Y_1,0 = sqrt(3/(4 pi)) z``, and
  ``int Y_n Y_n' dOmega = delta_nn'`` over the sphere.

* Compton scattering to energies below the lowest group edge is not in ``mu_s_l`` / ``mu_s_eta``; it stays in
  ``mu_t`` and is absorbed. Its group-weighted size is ``BuildInfo.compton_below_cutoff [M, G]``, and per material m
  and source group g, ``sum_g' mu_s_l[m, g, g', 0] + compton_below_cutoff[m, g] = <sigma_C + sigma_R>_g``. With the
  default lower edge of 10 keV it is nonzero only for source energies below about 10.4 keV (the backscatter limit).

Linear-discontinuous cell moments
---------------------------------

* Per cell, 4 spatial moments ordered ``[0, x, y, z]`` = mean, u-slope, v-slope, w-slope with local coordinates
  ``u = 2(x - x_c)/dx`` etc. in [-1, 1]; ``f_0 = (1/V) int f dV``, ``f_a = (3/V) int f u_a dV``.
* Linear-discontinuous moment fields: ``[G, 4, rows, cols, slices]``; spherical-harmonic moments of linear-discontinuous
  fields: ``[G, N_sh, 4, rows, cols, slices]``.

Materials
---------

The material catalogue ``cross_sections.MATERIAL_CATALOGUE`` maps a name to (nominal density, composition). The
nominal density is the density of the material in the body or the scanner. The table lists the compounds and
tissues (the first seven are the default reference materials); every element follows them. A tissue name carries
the source of its composition: ``_icrp`` (ICRP compositions), ``_icru44`` (ICRU Report 44) or ``_icru46`` (Woodard
and White 1986, also ICRU Report 46).

.. list-table::
   :header-rows: 1
   :widths: 24 60 16

   * - name
     - composition
     - :math:`\rho^{nom}` (g/cm³)
   * - ``air``
     - dry air: N₂, O₂, Ar, CO₂ (the volume percentages read as mass fractions)
     - 0.0012
   * - ``lung_icrp``
     - ICRP lung (without the Ca, Fe, Zn traces)
     - 0.26
   * - ``adipose_icrp``
     - ICRP adipose tissue
     - 0.95
   * - ``water``
     - H₂O
     - 1.0
   * - ``muscle_icru44``
     - ICRU-44 skeletal muscle
     - 1.05
   * - ``soft_tissue_icru44``
     - ICRU-44 soft tissue
     - 1.06
   * - ``bone_cortical_icrp``
     - ICRP cortical bone
     - 1.85
   * - ``trabecular_bone_icru46``
     - spongiosa, Woodard and White (1986) / ICRU-46
     - 1.18
   * - ``brain_icru44``
     - ICRU-44 brain (grey / white matter)
     - 1.04
   * - ``blood_icru44``
     - ICRU-44 whole blood
     - 1.06
   * - ``cartilage_icru46``
     - Woodard and White (1986) / ICRU-46
     - 1.10
   * - ``skin_icru44``
     - ICRU-44 skin
     - 1.09
   * - ``pmma``
     - PMMA (C₅H₈O₂)
     - 1.18
   * - ``polyethylene``
     - polyethylene (C₂H₄)
     - 0.94
   * - ``carbon_fibre``
     - carbon-fibre epoxy laminate (couch top): 60 % fibre volume, 0.692 C + 0.308 C₂₁H₂₄O₄ by mass
     - 1.56
   * - ``a150_tissue_equivalent_plastic``
     - NIST: A-150 Tissue-Equivalent Plastic
     - 1.127
   * - ``adipose_icru44``
     - NIST: Adipose Tissue (ICRU-44)
     - 0.95
   * - ``air_sea_level``
     - NIST: Air, Dry (near sea level)
     - 0.001205
   * - ``alanine``
     - NIST: Alanine
     - 1.424
   * - ``b100_bone_equivalent_plastic``
     - NIST: B-100 Bone-Equivalent Plastic
     - 1.45
   * - ``bakelite``
     - NIST: Bakelite
     - 1.25
   * - ``bone_cortical_icru44``
     - NIST: Bone, Cortical (ICRU-44)
     - 1.92
   * - ``breast_icru44``
     - NIST: Breast Tissue (ICRU-44)
     - 1.02
   * - ``c552_air_equivalent_plastic``
     - NIST: C-552 Air-equivalent Plastic
     - 1.76
   * - ``cadmium_telluride``
     - NIST: Cadmium Telluride
     - 6.2
   * - ``calcium_fluoride``
     - NIST: Calcium Fluoride
     - 3.18
   * - ``calcium_sulfate``
     - NIST: Calcium Sulfate
     - 2.96
   * - ``ceric_ammonium_sulfate_solution``
     - 15 mmol/L Ceric Ammonium Sulfate Solution
     - 1.03
   * - ``cesium_iodide``
     - NIST: Cesium Iodide
     - 4.51
   * - ``concrete_ordinary``
     - NIST: Concrete, Ordinary
     - 2.3
   * - ``concrete_barite``
     - NIST: Concrete, Barite (Type BA)
     - 3.35
   * - ``eye_lens_icru44``
     - NIST: Eye Lens (ICRU-44)
     - 1.07
   * - ``ferrous_sulfate_fricke``
     - NIST: Ferrous Sulfate, Standard Fricke
     - 1.024
   * - ``gadolinium_oxysulfide``
     - NIST: Gadolinium Oxysulfide
     - 7.44
   * - ``gafchromic_sensor``
     - NIST: Gafchromic Sensor
     - 1.3
   * - ``gallium_arsenide``
     - NIST: Gallium Arsenide
     - 5.31
   * - ``glass_borosilicate``
     - NIST: Glass, Borosilicate (Pyrex)
     - 2.23
   * - ``glass_lead``
     - NIST: Glass, Lead
     - 6.22
   * - ``lithium_fluoride``
     - NIST: Lithium Fluoride
     - 2.635
   * - ``lithium_tetraborate``
     - NIST: Lithium Tetraborate
     - 2.44
   * - ``lung_icru44``
     - NIST: Lung Tissue (ICRU-44)
     - 1.05
   * - ``magnesium_tetraborate``
     - NIST: Magnesium Tetraborate
     - 2.53
   * - ``mercuric_iodide``
     - NIST: Mercuric Iodide
     - 6.36
   * - ``ovary_icru44``
     - NIST: Ovary (ICRU-44)
     - 1.05
   * - ``photographic_emulsion_kodak_aa``
     - NIST: Photographic Emulsion (Kodak Type AA)
     - 2.2
   * - ``photographic_emulsion_standard_nuclear``
     - NIST: Photographic Emulsion (Standard Nuclear)
     - 3.815
   * - ``plastic_scintillator``
     - NIST: Plastic Scintillator, Vinyltoluene
     - 1.032
   * - ``mylar``
     - NIST: Polyethylene Terephthalate (Mylar)
     - 1.38
   * - ``polystyrene``
     - NIST: Polystyrene
     - 1.06
   * - ``teflon``
     - NIST: Polytetrafluoroethylene (Teflon)
     - 2.25
   * - ``polyvinyl_chloride``
     - NIST: Polyvinyl Chloride
     - 1.406
   * - ``radiochromic_dye_film``
     - NIST: Radiochromic Dye Film, Nylon Base
     - 1.08
   * - ``testis_icru44``
     - NIST: Testis (ICRU-44)
     - 1.04
   * - ``soft_tissue_four_component``
     - NIST: Tissue, Soft (ICRU Four-Component)
     - 1
   * - ``tissue_equivalent_gas_methane``
     - NIST: Tissue-Equivalent Gas, Methane Based
     - 0.001064
   * - ``tissue_equivalent_gas_propane``
     - NIST: Tissue-Equivalent Gas, Propane Based
     - 0.001826

The entries marked NIST are the compounds and mixtures of NIST's X-ray mass attenuation coefficient tables (Hubbell
and Seltzer, NISTIR 5632, `Table 2 <https://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html>`_), with NIST's
densities and mass fractions. Seven more of NIST's 48 materials are the entries ``water``, ``muscle_icru44`` (Muscle,
Skeletal, ICRU-44), ``soft_tissue_icru44`` (Tissue, Soft, ICRU-44), ``brain_icru44``, ``blood_icru44``, ``pmma``
(NIST lists 1.19 g/cm³)
and ``polyethylene`` (0.93 g/cm³), with the same compositions.

After these compounds the catalogue holds every element Z = 1..98 under its English (IUPAC) name, ``hydrogen`` to
``californium`` (for
example ``aluminium``, 2.6989 g/cm³, ``titanium``, 4.54 g/cm³, ``lead``, 11.35 g/cm³, and ``tungsten``, 19.3 g/cm³),
with the element as the composition and xraylib's element density (the NIST densities of Hubbell and Seltzer,
NISTIR 5632; gases at 20 °C and 1 atm) as the nominal density. xraylib has no scattering functions for Z = 99, 100.

A composition is a material string (``cross_sections.parse_formula``): a chemical formula (``H2O``, a
stoichiometric compound) or a mixture ``a*F1 + b*F2 + ...`` with a, b the relative mass fractions of the elements or
compounds F1, F2 (normalized to sum 1). A term's number is the mass fraction of its compound even when the compound
is a molecule, so ``78.084*N2`` weighs N2 by 78.084; element counts are converted with the atomic weights of XCOM.
A part that starts with a lowercase letter is a catalogue material by name and stands for its composition:
``0.99*water + 0.01*I`` is water with 1 % iodine by mass.

A material that is not in the catalogue is defined with ``cross_sections.Material(name, density, composition)``
(``citrus.scatter_solver.Material``): its name, its nominal density in g/cm³ and its composition in the syntax above.
It is accepted wherever a catalogue name is, and its library constants are built from the photon data exactly like
those of a catalogue material. The definition is checked on construction: the composition must parse, its elements
must have scattering data (Z = 1..98), the density must be positive, and a catalogue name is accepted only with that
entry's own density and composition (such a copy is the catalogue entry). The library cache key holds every
material's name, density and composition, so two definitions under one name never share a cached library.

.. code-block:: python

   from citrus.scatter_solver import Material

   contrast = Material("iodine_solution", 1.01, "0.99*water + 0.01*I")   # a contrast agent in water
   polymer = Material("peek", 1.32, "C19H12O3")                           # an industrial polymer
   alloy = Material("ti6al4v", 4.43, "0.90*Ti + 0.06*Al + 0.04*V")        # an alloy, by mass

The compositions of air, lung, adipose tissue, cortical bone, PMMA and polyethylene are those of the WLAB validation
materials. Cortical bone and adipose tissue have exactly the mass fractions of xraylib's NIST compounds
'Bone, Cortical (ICRP)' and 'Adipose Tissue (ICRP)' (NIST lists 0.92 g/cm³ for adipose tissue, the catalogue uses
0.95); lung is xraylib's 'Lung (ICRP)' without its Ca, Fe and Zn traces, every mass fraction within 4e-4; air reads
the volume percentages of dry air as mass fractions of the compounds (N 0.781, O 0.210, Ar 0.0093 by mass, against
0.755, 0.232, 0.0128 for xraylib's 'Air, Dry (near sea level)'). Cortical bone is the ICRP composition, not the
ICRU-44 one. The ICRU-44 compositions (muscle, soft tissue, brain, blood, skin) are as tabulated by Hubbell and
Seltzer, NISTIR 5632; spongiosa and cartilage are from Woodard and White (1986), Br J Radiol 59:1209-1219, also in
ICRU Report 46 (1992). Muscle lies below soft tissue at every library energy (lower density, nearly the same
composition), hence its place in the default order.

* Reference materials: the library is built for ``SolverConfig.materials`` (at least two catalogue names or
  ``Material`` definitions with distinct names, any order), sorted by increasing attenuation at the nominal density
  at the effective energy (neighbours must differ there by at least 0.1 %; they may cross at other
  energies). Default: the seven ``cross_sections.MATERIALS``, in library order 0 air (0.0012 g/cm^3), 1 lung (0.26), 2
  adipose (0.95), 3 water
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

WLAB phantom
------------

The WLAB phantom (``citrus.utilities.wlab_phantom``, ``WlabPhantom``) is a water prism (1.0 g/cm^3) of 30 cm along
x (columns), 20 cm along y (rows, the source-detector axis at view angle 0) and 40 cm along z (slices), centred at
the isocenter, with three full-length inserts (40 cm along z): lung (6 x 6 cm^2, 0.26 g/cm^3) centred at
(x, y) = (-5, 0) cm, air (2 x 2 cm^2, 0.001205 g/cm^3) at (+3, 0) cm and bone (2 x 2 cm^2, 1.85 g/cm^3) at (+7, 0) cm
(keyword arguments of ``wlab_phantom``). Outside the prism the density is 0. Voxels are assigned by the membership
of their centres in half-open intervals. The phantom uses two rows of the library's material basis, looked up by
name: lung and air are water of the stated density (the "water" row) and bone is cortical bone (the
"bone_cortical_icrp" row), so
a set of reference materials without water or bone cannot hold it.

``wlab_grid(delta_voxel)`` tiles the prism exactly. ``wlab_geometry()`` is a full-fan kV geometry with source-axis
distance 100 cm, source-detector distance 150 cm and a centred 40 x 30 cm^2 panel of 100 x 75 pixels of 4 mm, on
which the lateral and z ends of the prism fall outside the field of view. ``full_field_wlab_geometry()`` widens the
panel to 150 x 184 pixels (60 x 73.6 cm) so that the whole prism projects onto it at every view angle, as
reconstruction needs: the transaxial half-diagonal of the prism, 18.03 cm, subtends
``|u| <= 150 tan(asin(18.03 / 100)) = 27.5 cm`` (at least 138 channels), and the z ends (``|z| = 20`` cm) of the
prism edge nearest to the source are magnified by up to ``150 / (100 - 18.03) = 1.83``, so ``|v| <= 36.6 cm`` (at
least 183 rows).

Device and dtype
----------------

* Every public function accepts tensors on any device and returns tensors on the same device. Default dtype float32;
  float64 works when the inputs are float64 (used by the oracle tests). No module hard-codes a device.
