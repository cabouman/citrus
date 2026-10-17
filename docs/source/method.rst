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
broadening) and Rayleigh scattering (form factors), from EPDL97 via XrayPhysics. Every voxel is a mixture of the
seven basis materials of the table below with densities :math:`\rho_m`, so every coefficient is linear in the
densities: :math:`\mu_x(\mathbf r, E) = \sum_m \rho_m(\mathbf r)\, m_{x,m}(E)` with the mass coefficients
:math:`m_{x,m}` of the cross-section library. The density model of the CITRUS coupling gives every voxel at most two
nonzero materials. The first-collision source is evaluated on the quadrature grid (below) from resampled densities;
where every quadrature point has at most two nonzero materials it gathers only their two table rows (the pair
representation, ``cross_sections.material_pairs``: the cost of a two-material basis), otherwise all seven (about 3.5
times the work per point, on far fewer points than the fine grid has).

Uncollided and collided fluence
-------------------------------

The fluence splits into the uncollided part, known in closed form, and the collided part:
:math:`\psi = \psi_{UC} + \psi_C`. With :math:`\boldsymbol\Omega_{UC}(\mathbf r) = (\mathbf r - \mathbf r_0) /
|\mathbf r - \mathbf r_0|`, the uncollided scalar fluence of energy group :math:`g` is

.. math::

   \phi_{UC,g}(\mathbf r) = \frac{b(\boldsymbol\Omega_{UC})}{|\mathbf r - \mathbf r_0|^2}
   \sum_{E\in g} w_E \exp\!\Big(-\sum_m P_m(\mathbf r)\, m_m(E)\Big),
   \qquad P_m(\mathbf r) = \int_{\mathbf r_0}^{\mathbf r} \rho_m\, dl,

i.e. one ray trace per material present in the volume (an absent material adds nothing) and a fine-energy table (0.5 keV
bins, weights :math:`w_E` of the spectrum): exact in energy, without the within-group beam-hardening error of
group-averaged attenuation. The collided fluence satisfies the same LBTE with the **first-collision source** in place of
:math:`q`,

.. math::

   S_{FC,g}(\mathbf r, \boldsymbol\Omega) = \sum_{g' \le g}
   \mu_s^{g'\to g}\big(\mathbf r;\ \boldsymbol\Omega_{UC}(\mathbf r)\cdot\boldsymbol\Omega\big)\, \phi_{UC,g'}(\mathbf r),

evaluated from the tabulated per-steradian transfer :math:`\mu_s^{g'\to g}(\eta)`, :math:`\eta = \cos\theta` on a
2001-point grid (no Legendre truncation: the first-collision source keeps the Rayleigh forward peak).

Grids. Following Acuros CTS (Maslowski et al. 2018, Med. Phys. 45:1899, Section 2.B), one computational grid carries
every step of the scatter solve: the transport grid C of cubic cells of edge ``transport_cell_cm`` (default 1 cm)
covering the reconstruction box, whatever the reconstruction voxel size (:doc:`conventions`, "Grids"). The
first-collision source is integrated over each cell with :math:`q^3` equal-weight points on the midpoint lattice
(the quadrature grid Q = C refined :math:`q` times; :math:`q` = round(cell / ``quadrature_point_spacing_cm``), points
about 0.5 cm apart by default: 2 per axis in 1 cm cells, 4 in 2 cm cells, 1 in 0.5 cm cells; bounded by the data
resolution: at most round(cell / fine voxel), so 1 cm voxels in 1 cm cells use :math:`q = 1`), with the densities
of each point the exact average of the fine densities over its :math:`(\Delta_C/q)^3` sub-cell (conservative overlap
resampling, mass conserving for any ratio of voxel to cell size). :math:`\phi_{UC}` is evaluated at the Q points by
tracing the resampled density maps from the focal spot through Q (``uncollided_trace_grid = "quadrature"``, default;
"coarse" traces through C, which smears a material boundary inside a cell, "fine" through the reconstruction grid).
The cost of every first-order step is :math:`q^3` points per cell, independent of the fine grid; only the primary
image is computed on the fine grid. In the CITRUS loop the densities themselves never exist on the fine grid as a
whole: the density model of the reconstruction :math:`X` (below) is evaluated one material map at a time, each map
resampled onto C and Q and dropped (``ScatterSolver.material_fields_from_attenuation``), and the primary projects the
maps one at a time again.

Quadrature point spacing (WLAB, 0.5 cm voxels, converged solves, relative L2 difference of the detector scatter
signal, AP / lateral view; ``uncollided_trace_grid`` "quadrature" against the "fine" trace at the largest
:math:`q`, which equals it here because Q is then the reconstruction grid):

=====================  ===================  =======================  =================
cells                  q (spacing)          difference               time per view
=====================  ===================  =======================  =================
2 cm                   1 (2 cm)             1.1 % / 0.59 %           0.19 / 0.21 s
2 cm                   2 (1 cm)             0.40 % / 0.22 %          0.20 / 0.22 s
2 cm                   4 (0.5 cm, default)  reference                0.35 / 0.36 s
1 cm                   1 (1 cm)             0.25 % / 0.09 %          0.40 / 0.44 s
1 cm                   2 (0.5 cm, default)  reference                0.59 / 0.61 s
1 cm, 0.25 cm voxels   1 / 2 vs 3           0.20 % / 0.08 %,         0.41 / 0.58 vs
                                            0.05 % / 0.06 %          1.08 s
=====================  ===================  =======================  =================

The cell size itself matters far more than the points per cell: the converged scatter image with 2 cm cells (q = 4)
differs from the one with 1 cm cells (q = 2) by 3.6 % (AP) and 9.5 % (lateral).

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

Space is discretized on the transport grid C (cubic cells of edge ``transport_cell_cm``). In each
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
iteration stops when :math:`\|\phi^{(n)}\| / \|\phi_C\| <` ``source_iteration_tolerance`` (default :math:`10^{-3}`,
scalar-flux cell means; about 16 sweeps for WLAB).

The first-collision source enters the sweep as linear-discontinuous moments on the transport cells, restricted from
samples at the
:math:`q^3` quadrature points of each cell (mean and the three slopes of the midpoint quadrature, exact for a source
linear within the cell). Because :math:`S_{FC}` is forward peaked, point samples at the sweep directions
under-integrate it; each view therefore rescales the samples per quadrature point and group,
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

Grids. :math:`\tau_\downarrow` is traced on the transport grid; at every quadrature point :math:`\mathbf r_j` it is
corrected to first order inside the cell, :math:`\tau_j = \max(0, \tau_c - \mu_{t,c}\,\boldsymbol\Omega\cdot(\mathbf
r_j - \mathbf r_c))`, and the emission :math:`(S_{FC}(\mathbf r_j) + S_{MC,c})\,e^{-\tau_j}` is projected on the
quadrature grid (each point traced to the panel). Both sources share one attenuation field and one projection per
chunk of directions; nothing of the scatter image touches the fine grid.

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
per-pixel direction from the focal spot and the exact-in-energy transmission of the density line integrals,
averaged over sub-pixel rays after the energy sum. The scatter fraction is :math:`S/(P+S)`. The primary is computed in
two steps, the per-material line integrals :math:`P_m = \int \rho_m\,dl` (a projector) and the transmission with the
responses (``scatter_image.primary_from_path_lengths``), so any projector can feed it: the solver's Siddon ray to the
pixel centre (``primary_path_lengths``, the reference of ``solve_view``) or MBIRTORCH's pixel-averaged
separable-footprint projector (the CITRUS loop's default, below).

CITRUS coupling
---------------

MBIR reconstructs a monoenergetic attenuation map :math:`X` at an effective energy :math:`E_{eff}` (the spectrum mean,
63.03 keV for the default 125 kVp spectrum). The solver needs the densities of its seven basis materials, each with a
nominal density :math:`\rho_m^{nom}` (its density in the body) and a nominal attenuation
:math:`\mu_m = \rho_m^{nom}\, m_m(E_{eff})`, :math:`m_m` the total mass attenuation coefficient:

.. list-table::
   :header-rows: 1
   :widths: 18 34 16 16

   * - material
     - composition
     - :math:`\rho^{nom}` (g/cm³)
     - :math:`\mu` at 63 keV (1/cm)
   * - air
     - XrayPhysics ``air``
     - 0.0012
     - 0.00022
   * - lung
     - XrayPhysics ``lung``
     - 0.26
     - 0.0520
   * - adipose
     - XrayPhysics ``adipose_tissue``
     - 0.95
     - 0.1846
   * - water
     - H₂O
     - 1.0
     - 0.2013
   * - muscle
     - ICRU-44 skeletal muscle
     - 1.05
     - 0.2102
   * - soft tissue
     - ICRU-44 soft tissue
     - 1.06
     - 0.2122
   * - cortical bone
     - XrayPhysics ``bone_cortical`` (ICRP)
     - 1.85
     - 0.5366

The materials are listed in their fixed order of increasing nominal attenuation, which holds at every energy of the
library (10-125 keV; the library builder asserts it; muscle lies below soft tissue at every energy, attenuation ratio
0.986-0.991). The interpolating density model assigns each voxel the two consecutive materials whose nominal
attenuations bracket :math:`X`:

.. math::

   \mu_a \le X < \mu_b:\quad f = \frac{X - \mu_a}{\mu_b - \mu_a},\qquad
   \rho_a = (1 - f)\,\rho_a^{nom},\quad \rho_b = f\,\rho_b^{nom},

below the first material (air) the first material alone with :math:`\rho = X / m_{air}`, above the last (cortical
bone) the last alone with :math:`\rho = X / m_{bone}` (bone denser than 1.85 g/cm³); negative values are clamped to
vacuum. Since :math:`\rho_a m_a + \rho_b m_b = (1 - f)\mu_a + f \mu_b = X`, the model is exact in attenuation at
:math:`E_{eff}`; it returns the pure material at its own nominal attenuation (a water voxel is water, a cortical-bone
voxel bone at 1.85 g/cm³) and at most two materials per voxel. Composition matters beyond :math:`E_{eff}`: the
solver's primary and scatter are polychromatic, and two voxels of equal :math:`X` but different composition attenuate
the harder and softer parts of the spectrum differently. The two-material split of versions up to 0.0.3 (water of
density :math:`X/m_w` below unit water, unit water *plus* bone above it) turned the WLAB cortical-bone insert into
water 1.0 + bone 1.156 g/cm³; behind the bone the polychromatic primary was then 3 % low and the SPR 3 % high, and the
coupled loop overshot the bone by 6 %. With the seven-material model the 20-iteration WLAB reconstruction has the bone
within 0.6 % and the water within 0.6 % (relative RMSE 0.0389 instead of 0.0429), and the solver SPR at the true
attenuation reproduces the SPR of the true densities to 3·10⁻⁴.

Per view :math:`v`, CITRUS keeps the accumulated collided moments :math:`\Phi_v = \sum_{n\ge1}\phi_v^{(n)}` and performs
in every outer iteration :math:`k` the step

.. math::

   \phi_{UC} = G_1(\rho_k), \qquad \Phi_v \leftarrow \phi^{(1)}(\rho_k) + K(\rho_k)\,\Phi_v
   \quad(\text{one sweep of the summed sources}), \qquad
   s_v = \text{scatter image of } (\phi_{UC}, \Phi_v), \qquad p_v = \text{primary image of } \rho_k.

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

Densities and primary of the loop. :math:`\rho_k` is the density model of :math:`X_k`, evaluated one material at a
time (no seven-material volume on the reconstruction grid). The primaries :math:`p_v` of all views come from one pass
per material present: its map is projected onto all views (in chunks of views whose path lengths stay within one
sinogram) and the per-view transmission is formed from the stacked line integrals (``CitrusConfig.primary_projector``:
"mbirtorch", the default, MBIRTORCH's pixel-averaged line integrals, the reconstruction's own operator; "siddon", the
solver's ray to each pixel centre, as in ``solve_view``). The two agree to 0.9 % (relative L2 over the 180 views of
the WLAB study; median pixel 0.05 %) except on the pixels whose ray grazes a face of the prism, where a centre ray and
a pixel average differ by up to a factor 5 in transmission; the pixel-integrated primary is the physical model of a
detector pixel, so it is the default. The synthetic data (``utilities.synthesize_cone_beam_data``, data-set format 2)
define their scatter as :math:`\mathrm{SPR}\cdot P` with the SPR's primary from the same model as the loop's default;
with the other model the correction is inconsistent with the data on the silhouette pixels (RMS 0.8 % of the corrected
line integrals at the true densities, up to 0.87 on single pixels). On the demo's default data (format 2) CITRUS
reaches 0.0215 / 0.0259 relative RMSE (whole prism / core) with the MBIRTORCH primary and 0.0257 / 0.0305 with the
Siddon primary; on the format-1 data of version 0.2.0 (SPR with the Siddon primary) the two give 0.0270 / 0.0305 and
0.0242 / 0.0294: the consistent primary wins by 0.003-0.004 either way.

MBIR resume. Between outer iterations the loop keeps MBIRTORCH's error sinogram :math:`e = y_c - A X` (as
:math:`-AX`, one sinogram) and resumes MBIRTORCH's VCD engine from it instead of recomputing it with a forward
projection; a new corrected sinogram only shifts it (``CitrusConfig.resume_error_sinogram``, default on; the
reconstruction equals the restarted one to float rounding, 1e-6 relative).

Short scans. The views are evenly spaced over :math:`[\beta_s, \beta_e)`; the default range of the synthetic data is a
short scan, :math:`\beta_e = \beta_s + \pi + 2\gamma_m` with :math:`2\gamma_m = 2\arctan(N_c \Delta_c / (2\,
\mathrm{SDD}))` the in-plane fan angle of the panel (22.6° for the WLAB full-field panel). MBIRTORCH's FDK assumes a
full rotation, so before it the line integrals of a short scan of range :math:`\Lambda` are weighted with
:math:`(\Lambda/\pi)\, w(\beta, \gamma)`, the Parker weights in the overscan form of Wesarg et al. (:math:`\delta =
(\Lambda - \pi)/2 \ge \gamma_m`, :math:`\gamma = -\arctan(u/\mathrm{SDD})` in MBIRTORCH's frame), which make the two
measurements of every in-plane line sum to one. MBIR itself needs no weighting. The sparse-view interpolation below
runs along the arc of a short scan instead of circularly.

Sparse-view scatter
-------------------

With ``scatter_view_stride`` :math:`n > 1` the step above runs only on the computed views :math:`\mathcal C = \{0, n,
2n, \dots\}` (uniform over the angular range, starting at view 0), and only they keep an accumulator
:math:`\Phi_v`. Every other view :math:`v` gets the scatter signal interpolated in angle, per detector pixel, from the
computed views,

.. math::

   s_v = \max\Big(0,\ \sum_{j\in\mathcal C} w_{vj}\, s_j\Big), \qquad
   \mathrm{SPR}_v = s_v / p_v,

with :math:`p_v` the primary image of view :math:`v` (a projection of the current densities, no transport, so cheap;
computed for every view) and weights :math:`w_{vj}` that are circular in angle (period :math:`2\pi`) for a full
rotation; for a short scan they run along the arc (the views beyond the last computed view keep its value; the cubic's
ghost nodes at the ends are linear extrapolations). With :math:`t\in[0,1)` the fractional position of the view between
its two bracketing computed views :math:`j_0, j_1` (``scatter_view_interpolation_order``):

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
iterations, two-material density model of version 0.0.2) stride 2 costs +0.0014 relative RMSE (0.0443 vs 0.0429) for 51
% of the scatter time and stride 4 +0.008 (0.0509) for 26 %; Catmull-Rom matches or beats linear interpolation offline
(skipped-view scatter error 1.05 / 1.94 % vs 1.12 / 2.11 % at strides 2 / 4) at no extra cost. Interpolating the TRUE
scatter costs only about +0.0025 at stride 4, so the in-loop penalty is mostly the slower convergence of the coupled
loop on the skipped views, not the interpolation error.

Scatter step every n outer iterations
-------------------------------------

With ``scatter_every`` :math:`n > 1` the scatter stage runs at the outer iterations :math:`k = 0, n, 2n, \dots` only;
at the other outer iterations the previous ratio :math:`\mathrm{SPR}` is reused and only the MBIR iteration runs, so
:math:`n` MBIR iterations run per scatter step and the accumulators :math:`\Phi_v` keep their age across the skipped
iterations (the recursion is unchanged, it is simply advanced less often). An MBIR iteration costs about 1 % of a
scatter stage on the WLAB study (0.75 s against 56 s for 180 views), so for a given number of scatter steps the
additional MBIR iterations are nearly free.

Measured on the 180-view WLAB study (0.5 cm grid, the defaults of version 0.0.3 with the two-material density model,
relative RMSE over the prism): at a fixed 20 outer iterations the scatter step at every 2nd / 4th iteration (10 / 5
steps) raises the final error from 0.0429 to 0.0513 / 0.0700, because between refreshes the MBIR iterations converge
towards the solution of a stale correction. The same 20 scatter steps spread over 40 (:math:`n = 2`) / 80 (:math:`n =
4`) outer iterations lower it to 0.0310 / 0.0286 in the same wall time, and at 40 iterations :math:`n = 2` is within
0.0008 of a scatter step at every iteration (0.0302) at half the scatter cost. The scatter budget sets the cost; choose
``num_outer_iterations`` as :math:`n` times the number of scatter steps you can afford. At equal scatter cost, one
CITRUS sweep per iteration (0.0429) beats a converged scatter solve at every 2nd iteration (0.0501): a fresh partial
update is worth more than a converged but stale one.
