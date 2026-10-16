Method
======

This page derives the LBTE source iteration solver (the scatter model) and the CITRUS coupling with model-based
iterative reconstruction (MBIR). Units: cm, keV, g/cm^3; attenuation in 1/cm; source strength 1 photon per steradian
at the focal spot (:doc:`conventions`).

Transport equation
------------------

The photon angular fluence :math:`\psi(\mathbf r, E, \boldsymbol\Omega)` in the object obeys the steady linear
Boltzmann transport equation (LBTE)

.. math::

   \boldsymbol\Omega\cdot\nabla\psi + \mu_t(\mathbf r, E)\,\psi
   = \int_0^\infty\!\!\int_{4\pi} \mu_s(\mathbf r, E'\!\to E, \boldsymbol\Omega'\cdot\boldsymbol\Omega)\,
     \psi(\mathbf r, E', \boldsymbol\Omega')\,d\Omega'\,dE' + q(\mathbf r, E, \boldsymbol\Omega),

with vacuum boundaries and a point source at the focal spot :math:`\mathbf r_0`,
:math:`q = s(E)\,b(\boldsymbol\Omega)\,\delta(\mathbf r - \mathbf r_0)`, where :math:`s(E)` is the normalized spectrum
and :math:`b` the collimation (1 for directions that hit the panel, else 0). Interactions: photoelectric absorption
(local deposition), Compton scattering (Klein-Nishina kinematics with the incoherent scattering function, no Doppler
broadening) and Rayleigh scattering (form factors), from EPDL97 via XrayPhysics. Every voxel is a mixture of two basis
materials (water, cortical bone) with densities :math:`\rho_w, \rho_b`, so every coefficient is linear in the
densities: :math:`\mu_x(\mathbf r, E) = \rho_w(\mathbf r)\, m_{x,w}(E) + \rho_b(\mathbf r)\, m_{x,b}(E)`.

Uncollided and collided fluence
-------------------------------

The fluence splits into the uncollided part, known in closed form, and the collided part:
:math:`\psi = \psi_{UC} + \psi_C`. With :math:`\boldsymbol\Omega_{UC}(\mathbf r) = (\mathbf r - \mathbf r_0) /
|\mathbf r - \mathbf r_0|`, the uncollided scalar fluence of energy group :math:`g` is

.. math::

   \phi_{UC,g}(\mathbf r) = \frac{b(\boldsymbol\Omega_{UC})}{|\mathbf r - \mathbf r_0|^2}
   \sum_{E\in g} w_E \exp\!\big(-P_w(\mathbf r)\, m_w(E) - P_b(\mathbf r)\, m_b(E)\big),
   \qquad P_m(\mathbf r) = \int_{\mathbf r_0}^{\mathbf r} \rho_m\, dl,

i.e. two ray traces of the density maps and a fine-energy table (0.5 keV bins, weights :math:`w_E` of the spectrum):
exact in energy, without the within-group beam-hardening error of group-averaged attenuation. The collided fluence
satisfies the same LBTE with the **first-collision source** in place of :math:`q`,

.. math::

   S_{FC,g}(\mathbf r, \boldsymbol\Omega) = \sum_{g' \le g}
   \mu_s^{g'\to g}\big(\mathbf r;\ \boldsymbol\Omega_{UC}(\mathbf r)\cdot\boldsymbol\Omega\big)\, \phi_{UC,g'}(\mathbf r),

evaluated from the tabulated per-steradian transfer :math:`\mu_s^{g'\to g}(\eta)`, :math:`\eta = \cos\theta` on a
2001-point grid (no Legendre truncation: the first-collision source keeps the Rayleigh forward peak).

Multigroup and spherical-harmonic discretization
------------------------------------------------

Energy groups are ordered from high to low energy with equal source photon counts (default 6 groups over 10-125 keV);
there is no upscatter, so transfers run from group :math:`g'` to :math:`g \ge g'`. The scattering kernel of the
collided transport is expanded in Legendre polynomials to order :math:`L` (default 7),

.. math::

   \mu_s^{g'\to g}(\boldsymbol\Omega'\cdot\boldsymbol\Omega) = \sum_{l=0}^{L} \frac{2l+1}{4\pi}\,
   \mu_{s,l}^{g'\to g}\, P_l(\boldsymbol\Omega'\cdot\boldsymbol\Omega).

With real orthonormal spherical harmonics :math:`Y_n`, :math:`n = l^2 + (k + l)`, the addition theorem
:math:`P_l(\boldsymbol\Omega'\cdot\boldsymbol\Omega) = \frac{4\pi}{2l+1}\sum_k Y_{lk}(\boldsymbol\Omega)
Y_{lk}(\boldsymbol\Omega')` cancels the :math:`(2l+1)/4\pi` factor, and the scattering source is

.. math::

   S_g(\boldsymbol\Omega) = \sum_n T_{g,n}\, Y_n(\boldsymbol\Omega), \qquad
   T_{g,n} = \sum_{g'\le g} \mu_{s,l(n)}^{g'\to g}\, \phi_{g',n}, \qquad
   \phi_{g,n} = \int Y_n\, \psi_g\, d\Omega \approx \sum_m w_m Y_n(\boldsymbol\Omega_m)\, \psi_{g,m}.

Discrete ordinates
------------------

The angular variable of the collided transport is collocated on a product quadrature: Gauss-Legendre in
:math:`\cos\theta` (``num_polar``, default 8) times equally spaced azimuths (``num_azimuthal``, default 16); it
integrates products :math:`Y_n Y_{n'}` up to degree :math:`L` exactly when ``num_polar`` >= L + 1 and
``num_azimuthal`` >= 2L + 1. Every octant holds the same number of directions.

Linear-discontinuous cells
--------------------------

Space is discretized on the supervoxel grid (``supervoxel_factor`` fine voxels per cell along each axis). In each
cuboid cell the angular fluence of one direction is linear, with moments :math:`(\psi_0, \psi_x, \psi_y, \psi_z)` in
the local coordinates :math:`u_a = 2(x_a - x_{a,c})/\Delta_a \in [-1, 1]`
(:math:`f_0 = \frac1V\int f\,dV`, :math:`f_a = \frac3V\int f\,u_a\,dV`). The Galerkin projection of the transport
equation with upwind face values gives, with :math:`s_a = \mathrm{sign}(\Omega_a)` and
:math:`c_a = |\Omega_a|/\Delta_a`,

.. math::

   (\textstyle\sum_a c_a + \mu)\,\psi_0 + \sum_a s_a c_a\,\psi_a &= R_0, \\
   -3 s_a c_a\,\psi_0 + d_a\,\psi_a &= R_a, \qquad d_a = 3c_a + \textstyle\sum_{b\ne a} c_b + \mu,

with :math:`R_0 = S_0 + \sum_a c_a \bar\psi_{in,a}`, :math:`R_a = S_a - 3 s_a c_a \bar\psi_{in,a} + \sum_{b\ne a} c_b
\psi_a[N_b]`, the face-averaged inflow :math:`\bar\psi_{in,a} = \psi_0[N_a] + s_a\psi_a[N_a]` from the upstream
neighbour :math:`N_a` along axis :math:`a` (zero for vacuum), and the transverse coupling :math:`\psi_a[N_b]`. The 4 x 4
system has the closed-form solution

.. math::

   \psi_0 = \frac{R_0 - \sum_a s_a c_a R_a / d_a}{\sum_a c_a + \mu + 3\sum_a c_a^2/d_a}, \qquad
   \psi_a = \frac{R_a + 3 s_a c_a \psi_0}{d_a}.

The scheme conserves particles cell by cell (tested against a dense Galerkin assembly, a 1-D slab solution and the
particle balance). No negative-flux fix-up is applied.

Transport sweep
---------------

For an octant, a cell depends only on its upstream neighbours, so all cells on a diagonal plane
:math:`p = i' + j' + k'` (indices counted from the octant's inflow corner) can be solved at once after the planes
before it. The sweep processes the :math:`R + C + S - 2` planes in order, each as one batched step over (cells of the
plane in all 8 octants) x (a chunk of directions per octant) x (groups): a gather of the upstream moments, the
closed-form cell solve and a scatter (one fused Triton kernel per plane on float32 CUDA). The spherical-harmonic
moments of the solution are accumulated per chunk with one matrix product, so the angular fluence of all directions is
never held at once.

Source iteration
----------------

Source iteration orders the collided fluence by collision number:

.. math::

   \phi^{(1)} = \mathcal{L}^{-1} S_{FC}, \qquad \phi^{(n)} = \mathcal{L}^{-1}\,\mathcal{S}\,\phi^{(n-1)} = K\,\phi^{(n-1)},
   \qquad \phi_{C} = \sum_{n\ge1} \phi^{(n)},

where :math:`\mathcal L^{-1}` is one transport sweep and :math:`\mathcal S` the scattering-source operator. The
iteration stops when :math:`\|\phi^{(n)}\| / \|\phi_C\| <` ``si_tol`` (default :math:`10^{-3}`, scalar-flux cell means;
about 16 sweeps for WLAB).

The first-collision source enters the sweep as LD moments on the supervoxels, restricted from samples at the fine voxel
centers (the :math:`f^3` sub-voxel points of each supervoxel). Because :math:`S_{FC}` is forward peaked, point samples
at the sweep directions under-integrate it; each view therefore rescales the samples per voxel and group,
:math:`S_m \leftarrow a\,S_m\,(1 + \beta\,\boldsymbol\Omega_m\cdot\boldsymbol\Omega_{UC})`, so that the quadrature
reproduces the exact :math:`l = 0` and :math:`l = 1` moments of the first-collision source about
:math:`\boldsymbol\Omega_{UC}` (with :math:`\beta\in[-1,1]` keeping the source non-negative).

Collapsed-cone scatter image
----------------------------

The scattered photons reaching the detector are those emitted at their last collision by the last-collided source
:math:`E(\mathbf r, \boldsymbol\Omega) = S_{FC}(\mathbf r, \boldsymbol\Omega) + S_{MC}(\mathbf r, \boldsymbol\Omega)`,
with the multiple-collision source :math:`S_{MC} = \sum_n T_n Y_n`, :math:`T` the scattering-source moments of the
converged collided fluence (cell means). The angular fluence at a panel point :math:`\mathbf r_D` is the attenuated
line integral

.. math::

   \psi_D(\mathbf r_D, \boldsymbol\Omega) = \int_0^\infty E(\mathbf r_D - l\boldsymbol\Omega, \boldsymbol\Omega)\,
   e^{-\tau_{\downarrow}(\mathbf r_D - l\boldsymbol\Omega,\, \boldsymbol\Omega)}\, dl,

with :math:`\tau_\downarrow(\mathbf r, \boldsymbol\Omega)` the optical depth from :math:`\mathbf r` to the panel along
:math:`\boldsymbol\Omega`. For each direction :math:`\boldsymbol\Omega_k` of a cone product set about the inward panel
normal (``cone_num_polar`` x ``cone_num_azimuthal``, default 16 x 32, limited to the smallest cone that sees the whole
object from the panel), the attenuated emission is a volume field that is projected through every pixel center, and
the angular integral is the quadrature sum:

.. math::

   \Phi(\mathbf r_D) = \sum_k w_k\, \psi_D(\mathbf r_D, \boldsymbol\Omega_k).

No :math:`1/r^2` factor appears: in the point-to-point form :math:`\int dV\,E\,e^{-\tau}/|\mathbf r_D - \mathbf r|^2`
the volume element about :math:`\mathbf r_D` is :math:`l^2\,dl\,d\Omega`, which cancels it.

Grids. :math:`\tau_\downarrow` is traced on the supervoxel grid; at every sub-voxel point :math:`\mathbf r_j` it is
corrected to first order inside the cell, :math:`\tau_j = \max(0, \tau_c - \mu_{t,c}\,\boldsymbol\Omega\cdot(\mathbf
r_j - \mathbf r_c))`, and the emission :math:`(S_{FC}(\mathbf r_j) + S_{MC,c})\,e^{-\tau_j}` is projected on the fine
grid (each sub-point traced to the panel). Both sources share one attenuation field and one projection per chunk of
directions.

Finite-cone footprint. Each direction stands for a solid-angle cell of gnomonic widths :math:`(W_\alpha, W_\beta)`;
photons emitted at height :math:`h` above the panel within that cell land on an :math:`hW_\alpha \times hW_\beta`
rectangle. The projected image of each direction is therefore box-blurred with these widths at the emission-weighted
mean height of the direction. This removes most of the ray effects of a finite direction set (per-pixel error of the
16 x 32 set comparable to an unblurred 32 x 64 set at 1.2x its cost).

Detector signal and primary
---------------------------

The detector and grid responses are per incident photon: the signal of group :math:`g` is

.. math::

   \text{signal}_g(\mathbf r_D) = \sum_k w_k \cos\theta_{0,k}\, \psi_{D,g}(\mathbf r_D, \boldsymbol\Omega_k)\,
   G_g(\theta_{1,k}, \theta_{2,k})\, D_g(\theta_{0,k}),

with :math:`\theta_0` the angle to the panel normal, :math:`D` the detector response (keV deposited per photon) and
:math:`G` the anti-scatter-grid transmission at the grid angles. The primary image uses the same rule with the
per-pixel direction from the focal spot and the exact-in-energy transmission of the two density line integrals,
averaged over sub-pixel rays after the energy sum. The scatter fraction is :math:`S/(P+S)`.

CITRUS coupling
---------------

MBIR reconstructs a monoenergetic attenuation map :math:`X` at an effective energy :math:`E_{eff}` (the spectrum mean,
63 keV for the default 125 kVp spectrum). The solver needs densities: the two-material split

.. math::

   X \le \mu_w:\ \rho_w = X/m_w,\ \rho_b = 0; \qquad X > \mu_w:\ \rho_w = 1,\ \rho_b = (X - m_w)/m_b,

with :math:`m_{w,b} = m_{w,b}(E_{eff})` and :math:`\mu_w` the attenuation of unit-density water (negative values are
clamped to vacuum) is exact in attenuation at :math:`E_{eff}`.

Per view :math:`v`, CITRUS keeps the accumulated collided moments :math:`\Phi_v = \sum_{n\ge1}\phi_v^{(n)}` and performs
in every outer iteration :math:`k` the step

.. math::

   \phi_{UC} = G_1(\rho_k), \qquad \Phi_v \leftarrow \phi^{(1)}(\rho_k) + K(\rho_k)\,\Phi_v
   \quad(\text{one sweep of the summed sources}), \qquad
   s_v, p_v = \text{scatter and primary images of } (\phi_{UC}, \Phi_v).

With the densities held fixed, :math:`N` steps give :math:`\Phi^{(N)} = \sum_{n=1}^{N} K^{n-1}\phi^{(1)}`, exactly the
source iteration truncated after :math:`N` sweeps (tested), so the scatter estimate converges together with the
reconstruction at the cost of one sweep per view per outer iteration ("age-ordered" source iteration: the
:math:`n`-th collision order of the current estimate was computed :math:`n-1` outer iterations earlier).

The correction is multiplicative: :math:`Y_c = Y\,p/(p+s) = Y/(1 + \mathrm{SPR})` with
:math:`\mathrm{SPR} = s/p` per pixel. It is independent of the flat field and of the mismatch between the solver's
polychromatic primary and MBIR's monoenergetic model, and it only rescales the noise of :math:`Y`, so the weights
:math:`W = Y/I_0` of the measured counts are the exact weights of :math:`y_c = -\log(Y_c/I_0)`. One warm-started MBIR
(VCD) iteration at partition-schedule position :math:`k` on :math:`y_c` gives :math:`X_{k+1}`; the regularization is
estimated once and frozen. The loop starts from the FDK reconstruction of the uncorrected data.

Sparse-view scatter
-------------------

With ``scatter_view_stride`` :math:`n > 1` the step above runs only on the computed views :math:`\mathcal C = \{0, n,
2n, \dots\}` (uniform over the angular range, starting at view 0), and only they keep an accumulator
:math:`\Phi_v`. Every other view :math:`v` gets the scatter signal interpolated in angle, per detector pixel, from the
computed views,

.. math::

   s_v = \max\Big(0,\ \sum_{j\in\mathcal C} w_{vj}\, s_j\Big), \qquad
   \mathrm{SPR}_v = s_v / p_v,

with :math:`p_v` the solver's primary image of view :math:`v` (a ray trace of the current densities, no transport, so
cheap) and weights :math:`w_{vj}` that are circular in angle (period :math:`2\pi`). With :math:`t\in[0,1)` the
fractional position of the view between its two bracketing computed views :math:`j_0, j_1`
(``scatter_view_interp_order``):

- 0: the nearest computed view;
- 1: linear, :math:`(1-t)\,s_{j_0} + t\,s_{j_1}`;
- 2: 3-point Lagrange at the actual angles of the nearest computed view and its two neighbours (exact for quadratics
  of the angle);
- 3 (default): Catmull-Rom on :math:`j_0-1, j_0, j_1, j_1+1`, weights
  :math:`\tfrac12(-t^3+2t^2-t),\ \tfrac12(3t^3-5t^2+2),\ \tfrac12(-3t^3+4t^2+t),\ \tfrac12(t^3-t^2)`:
  interpolating with a continuous first derivative, exact for quadratics of the angle on uniformly spaced computed
  views (not for cubics: the central-difference tangents are second-order);
- ``None`` (diagnostics): no interpolation, the skipped views keep their most recent computed scatter if any, else
  zero (uncorrected).

The scatter signal is interpolated rather than the ratio: the solver scatter is smooth on the panel and in angle (99 %
of each pixel's variation energy lies in about 8-12 harmonics per rotation after a 1 cm low-pass), while the ratio
carries the object's shadow edges, which move across the panel with the angle. On the 180-view WLAB study (20 outer
iterations) stride 2 costs +0.0014 relative RMSE (0.0443 vs 0.0429) for 51 % of the scatter time and stride 4 +0.008
(0.0509) for 26 %; Catmull-Rom matches or beats linear interpolation offline (skipped-view scatter error 1.05 / 1.94 %
vs 1.12 / 2.11 % at strides 2 / 4) at no extra cost. Interpolating the TRUE scatter costs only about +0.0025 at stride
4, so the in-loop penalty is mostly the slower convergence of the coupled loop on the skipped views, not the
interpolation error.

Scatter step every n outer iterations
-------------------------------------

With ``scatter_every`` :math:`n > 1` the scatter stage runs at the outer iterations :math:`k = 0, n, 2n, \dots` only;
at the other outer iterations the previous ratio :math:`\mathrm{SPR}` is reused and only the MBIR iteration runs, so
:math:`n` MBIR iterations run per scatter step and the accumulators :math:`\Phi_v` keep their age across the skipped
iterations (the recursion is unchanged, it is simply advanced less often). An MBIR iteration costs about 1 % of a
scatter stage on the WLAB study (0.75 s against 56 s for 180 views), so for a given number of scatter steps the
additional MBIR iterations are nearly free.

Measured on the 180-view WLAB study (0.5 cm grid, release defaults, relative RMSE over the prism): at a fixed 20
outer iterations the scatter step at every 2nd / 4th iteration (10 / 5 steps) raises the final error from 0.0429 to
0.0513 / 0.0700, because between refreshes the MBIR iterations converge towards the solution of a stale correction.
The same 20 scatter steps spread over 40 (:math:`n = 2`) / 80 (:math:`n = 4`) outer iterations lower it to 0.0310 /
0.0286 in the same wall time, and at 40 iterations :math:`n = 2` is within 0.0008 of a scatter step at every
iteration (0.0302) at half the scatter cost. The scatter budget sets the cost; choose ``num_outer`` as :math:`n`
times the number of scatter steps you can afford. At equal scatter cost, one CITRUS sweep per iteration (0.0429)
beats a converged scatter solve at every 2nd iteration (0.0501): a fresh partial update is worth more than a
converged but stale one.
