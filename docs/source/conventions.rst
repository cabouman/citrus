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

Linear-discontinuous (LD) cell moments
--------------------------------------

* Per cell, 4 spatial moments ordered ``[0, x, y, z]`` = mean, u-slope, v-slope, w-slope with local coordinates
  ``u = 2(x - x_c)/dx`` etc. in [-1, 1]; ``f_0 = (1/V) int f dV``, ``f_a = (3/V) int f u_a dV``.
* LD-moment fields: ``[G, 4, rows, cols, slices]``; spherical-harmonic moments of LD fields:
  ``[G, N_sh, 4, rows, cols, slices]``.

Materials
---------

* Two basis materials: index 0 = water, index 1 = cortical bone (ICRP composition, 1.85 g/cm^3 reference).
  Density maps ``rho: [2, rows, cols, slices]`` in g/cm^3. All coefficients are linear in rho:
  ``mu_x[g] = rho[0] * lib.mu_x[0, g] + rho[1] * lib.mu_x[1, g]`` with per-unit-density (mass) coefficients.

Device and dtype
----------------

* Every public function accepts tensors on any device and returns tensors on the same device. Default dtype float32;
  float64 works when the inputs are float64 (used by the oracle tests). No module hard-codes a device.
