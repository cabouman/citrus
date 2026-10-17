Method
======

CITRUS couples model-based iterative reconstruction (MBIR) with a deterministic solver of the linear Boltzmann
transport equation (LBTE). This page describes the coupled loop first, then the scatter solver it drives and the
primary model, and last the options that trade accuracy for cost and memory. Units: cm, keV, g/cm^3; attenuation in
1/cm; source strength 1 photon per steradian at the focal spot (:doc:`conventions`).

The CITRUS loop
---------------

Outer iteration
~~~~~~~~~~~~~~~

MBIR reconstructs a monoenergetic attenuation map :math:`X` at an effective energy :math:`E_{eff}` (the spectrum mean)
from the measured counts :math:`Y`, primary plus scatter, with flat field :math:`I_0`. The loop starts from the FDK
reconstruction :math:`X_0` of the uncorrected data, and every outer iteration :math:`k`

1. converts :math:`X_k` into the densities :math:`\rho_k` of the library's reference materials (the density model,
   below);
2. advances the collided fluence of every view by one transport sweep and projects its scatter signal :math:`s_v`;
3. computes the primary signal :math:`p_v` of every view from :math:`\rho_k`;
4. divides the measured counts by :math:`1 + s_v / p_v` and runs one warm-started MBIR iteration on the corrected data,
   which gives :math:`X_{k+1}`.

Age-ordered source iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A converged scatter solve repeats transport sweeps, one per collision order, until the collided fluence has converged
(source iteration, below). CITRUS spends a single sweep per view per outer iteration instead and carries the collided
fluence from one outer iteration to the next. Per view :math:`v` it keeps the accumulated collided moments
:math:`\Phi_v = \sum_{n\ge1}\phi_v^{(n)}` and performs in outer iteration :math:`k` the step

.. math::

   \phi_{UC} = G_1(\rho_k), \qquad \Phi_v \leftarrow \phi^{(1)}(\rho_k) + K(\rho_k)\,\Phi_v
   \quad(\text{one sweep of the summed sources}), \qquad
   s_v = \text{scatter image of } (\phi_{UC}, \Phi_v),

with :math:`G_1` the uncollided operator, :math:`\phi^{(1)}` the once-collided fluence of the first-collision source and
:math:`K` the collided transport of one collision order (scattering source, then a sweep). Writing :math:`K_k` and
:math:`\phi^{(1)}_k` for these operators at the densities of reconstruction :math:`k`, the accumulator expands as

.. math::

   \Phi_v^{(k)} = \phi^{(1)}_k + K_k\,\Phi_v^{(k-1)}
                = \phi^{(1)}_k + K_k\,\phi^{(1)}_{k-1} + K_k K_{k-1}\,\phi^{(1)}_{k-2} + \dots

The :math:`n`-th collision order of the current estimate started from the first-collision source of :math:`n-1` outer
iterations earlier: the dominant low orders always come from the newest reconstructions, while stale information is
pushed into the weak, smooth high orders.

With the densities held fixed, :math:`N` steps give :math:`\Phi^{(N)} = \sum_{n=1}^{N} K^{n-1}\phi^{(1)}`, exactly the
source iteration truncated after :math:`N` sweeps (a tested identity, ``citrus_step``). The recursion *is* source
iteration, so the scatter estimate converges together with the reconstruction, for the cost of one sweep per view per
outer iteration.

Scatter-to-primary correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The correction is multiplicative: :math:`Y_c = Y\,p/(p+s) = Y/(1 + \mathrm{SPR})` with :math:`\mathrm{SPR} = s/p` per
pixel, both signals from the solver at the current densities. It is independent of the flat field and of the mismatch
between the solver's polychromatic primary and MBIR's monoenergetic model, and it only rescales the noise of :math:`Y`,
so the weights :math:`W = Y/I_0` of the measured counts are the exact weights of :math:`y_c = -\log(Y_c/I_0)`.

Warm-started MBIR
~~~~~~~~~~~~~~~~~

Each outer iteration runs one vectorized coordinate descent (VCD) iteration of MBIRTORCH on :math:`y_c`, started from
:math:`X_k` at position :math:`k` of the partition schedule, with the weights :math:`W` and a regularization that is
estimated once and then frozen. The partitions are drawn from a random generator seeded with ``[base_seed, k]``, so
runs with identical inputs give identical MBIR sequences. ``mbir_recon`` runs the same loop without the solver (no
correction, or a fixed SPR), so a comparison can share the initialization, the weights, the partition sequence and the
regularization with CITRUS.

The LBTE scatter solver
-----------------------

Transport equation
~~~~~~~~~~~~~~~~~~

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
:math:`M` reference materials of the library with densities :math:`\rho_m`, so every coefficient is linear in the
densities: :math:`\mu_x(\mathbf r, E) = \sum_m \rho_m(\mathbf r)\, m_{x,m}(E)` with the mass coefficients
:math:`m_{x,m}` of the cross-section library.

Uncollided fluence and first-collision source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fluence splits into the uncollided part, known in closed form, and the collided part:
:math:`\psi = \psi_{UC} + \psi_C`. With :math:`\boldsymbol\Omega_{UC}(\mathbf r) = (\mathbf r - \mathbf r_0) /
|\mathbf r - \mathbf r_0|`, the uncollided scalar fluence of energy group :math:`g` is

.. math::

   \phi_{UC,g}(\mathbf r) = \frac{b(\boldsymbol\Omega_{UC})}{|\mathbf r - \mathbf r_0|^2}
   \sum_{E\in g} w_E \exp\!\Big(-\sum_m P_m(\mathbf r)\, m_m(E)\Big),
   \qquad P_m(\mathbf r) = \int_{\mathbf r_0}^{\mathbf r} \rho_m\, dl,

i.e. one ray trace per material present in the volume and a fine-energy table (0.5 keV bins, weights :math:`w_E` of the
spectrum): exact in energy, without the within-group beam-hardening error of group-averaged attenuation. The collided
fluence satisfies the same LBTE with the **first-collision source** in place of :math:`q`,

.. math::

   S_{FC,g}(\mathbf r, \boldsymbol\Omega) = \sum_{g' \le g}
   \mu_s^{g'\to g}\big(\mathbf r;\ \boldsymbol\Omega_{UC}(\mathbf r)\cdot\boldsymbol\Omega\big)\,
   \phi_{UC,g'}(\mathbf r),

evaluated from the tabulated per-steradian transfer :math:`\mu_s^{g'\to g}(\eta)`, :math:`\eta = \cos\theta`, on a
2001-point grid (no Legendre truncation: the first-collision source keeps the Rayleigh forward peak). Both are evaluated
at the points of the quadrature grid, a few equal-weight points per transport cell (see "Transport and quadrature
grids" below).

Multigroup and spherical-harmonic discretization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~

The angular variable of the collided transport is collocated on a product quadrature: Gauss-Legendre in
:math:`\cos\theta` (``num_polar``, default 8) times equally spaced azimuths (``num_azimuthal``, default 16); it
integrates products :math:`Y_n Y_{n'}` up to degree :math:`L` exactly when ``num_polar`` >= L + 1 and
``num_azimuthal`` >= 2L + 1. Every octant holds the same number of directions.

Linear-discontinuous cells
~~~~~~~~~~~~~~~~~~~~~~~~~~

Space is discretized on the transport grid of cubic cells of edge ``transport_cell_cm``. In each cuboid cell the
angular fluence of one direction is linear, with moments :math:`(\psi_0, \psi_x, \psi_y, \psi_z)` in the local
coordinates :math:`u_a = 2(x_a - x_{a,c})/\Delta_a \in [-1, 1]` (:math:`f_0 = \frac1V\int f\,dV`,
:math:`f_a = \frac3V\int f\,u_a\,dV`). The Galerkin projection of the transport equation with upwind face values gives,
with :math:`s_a = \mathrm{sign}(\Omega_a)` and :math:`c_a = |\Omega_a|/\Delta_a`,

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
~~~~~~~~~~~~~~~

For an octant, a cell depends only on its upstream neighbours, so all cells on a diagonal plane
:math:`p = i' + j' + k'` (indices counted from the octant's inflow corner) can be solved at once after the planes
before it. The sweep processes the :math:`R + C + S - 2` planes in order, each as one batched step over (cells of the
plane in all 8 octants) x (a chunk of directions per octant) x (groups): a gather of the upstream moments, the
closed-form cell solve and a scatter (one fused Triton kernel per plane on float32 CUDA). The spherical-harmonic
moments of the solution are accumulated per chunk with one matrix product, so the angular fluence of all directions is
never held at once.

Source iteration
~~~~~~~~~~~~~~~~

Source iteration orders the collided fluence by collision number:

.. math::

   \phi^{(1)} = \mathcal{L}^{-1} S_{FC}, \qquad
   \phi^{(n)} = \mathcal{L}^{-1}\,\mathcal{S}\,\phi^{(n-1)} = K\,\phi^{(n-1)},
   \qquad \phi_{C} = \sum_{n\ge1} \phi^{(n)},

where :math:`\mathcal L^{-1}` is one transport sweep and :math:`\mathcal S` the scattering-source operator. The
standalone solver iterates until :math:`\|\phi^{(n)}\| / \|\phi_C\| <` ``source_iteration_tolerance`` (default
:math:`10^{-3}`, scalar-flux cell means); the CITRUS loop takes one of these steps per outer iteration.

The first-collision source enters the sweep as linear-discontinuous moments on the transport cells, restricted from
samples at the quadrature points of each cell (mean and the three slopes of the midpoint quadrature, exact for a source
linear within the cell). Because :math:`S_{FC}` is forward peaked, point samples at the sweep directions
under-integrate it; each view therefore rescales the samples per quadrature point and group,
:math:`S_m \leftarrow a\,S_m\,(1 + \beta\,\boldsymbol\Omega_m\cdot\boldsymbol\Omega_{UC})`, so that the quadrature
reproduces the exact :math:`l = 0` and :math:`l = 1` moments of the first-collision source about
:math:`\boldsymbol\Omega_{UC}` (with :math:`\beta\in[-1,1]` keeping the source non-negative).

Collapsed-cone scatter image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The scattered photons reaching the detector are those emitted at their last collision by the last-collided source
:math:`E(\mathbf r, \boldsymbol\Omega) = S_{FC}(\mathbf r, \boldsymbol\Omega) + S_{MC}(\mathbf r, \boldsymbol\Omega)`,
with the multiple-collision source :math:`S_{MC} = \sum_n T_n Y_n`, :math:`T` the scattering-source moments of the
collided fluence (cell means). The angular fluence at a panel point :math:`\mathbf r_D` is the attenuated line integral

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

:math:`\tau_\downarrow` is traced on the transport grid; at every quadrature point :math:`\mathbf r_j` it is corrected
to first order inside the cell, :math:`\tau_j = \max(0, \tau_c - \mu_{t,c}\,\boldsymbol\Omega\cdot(\mathbf r_j -
\mathbf r_c))`, and the emission :math:`(S_{FC}(\mathbf r_j) + S_{MC,c})\,e^{-\tau_j}` is projected from the quadrature
points. Both sources share one attenuation field and one projection per chunk of directions.

Finite-cone footprint. Each direction stands for a solid-angle cell of gnomonic widths :math:`(W_\alpha, W_\beta)`;
photons emitted at height :math:`h` above the panel within that cell land on an :math:`hW_\alpha \times hW_\beta`
rectangle. The projected image of each direction is therefore box-blurred with these widths at the emission-weighted
mean height of the direction, which removes most of the ray effects of a finite direction set.

Detector and grid responses
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The detector and grid responses are per incident photon: the signal of group :math:`g` is

.. math::

   \text{signal}_g(\mathbf r_D) = \sum_k w_k \cos\theta_{0,k}\, \psi_{D,g}(\mathbf r_D, \boldsymbol\Omega_k)\,
   G_g(\theta_{1,k}, \theta_{2,k})\, D_g(\theta_{0,k}),

with :math:`\theta_0` the angle to the panel normal, :math:`D` the detector response (keV deposited per photon) and
:math:`G` the anti-scatter-grid transmission at the grid angles. Both are lookup tables, so responses from a Monte
Carlo simulation of a real detector and grid can replace the analytic models.

The primary
-----------

The primary signal uses the same detector rule with the per-pixel direction from the focal spot and the exact-in-energy
transmission of the density line integrals, averaged over the pixel after the energy sum. It is computed in two steps,
the per-material line integrals :math:`P_m = \int \rho_m\,dl` (a projector) and the transmission with the responses
(``scatter_image.primary_from_path_lengths``), so any projector can feed it:

* in the CITRUS loop, MBIRTORCH's pixel-averaged separable-footprint projector, the reconstruction's own operator
  :math:`A`: one pass per material present projects its density map onto all views (in chunks of views whose path
  lengths stay within one sinogram), and the polychromatic transmission is formed per view (``primary_signals``). A
  pixel-integrated primary is the physical model of a detector pixel, and it is consistent with the synthetic data,
  whose scatter is defined through the same primary;
* in the standalone solver (``solve_view``), the solver's Siddon projector, one ray to each pixel centre.

The scatter fraction is :math:`S/(P+S)`.

Options and secondary components
--------------------------------

Density model and reference materials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The solver needs the densities of the library's reference materials, chosen from the material catalogue with
``SolverConfig.materials`` (at least two; by default air, lung, adipose, water, muscle, soft tissue and cortical bone;
the catalogue with compositions, nominal densities :math:`\rho_m^{nom}` and nominal attenuations
:math:`\mu_m = \rho_m^{nom}\, m_m(E_{eff})` is in :doc:`conventions`, "Materials"). The library sorts them by nominal
attenuation and refuses a set whose attenuations cross somewhere in 10-125 keV. Each voxel gets the two consecutive
reference materials whose nominal attenuations bracket :math:`X`:

.. math::

   \mu_a \le X < \mu_b:\quad f = \frac{X - \mu_a}{\mu_b - \mu_a},\qquad
   \rho_a = (1 - f)\,\rho_a^{nom},\quad \rho_b = f\,\rho_b^{nom};

below the lowest (above the highest) reference material that material alone at the density :math:`X / m_m`, and
negative values are clamped to vacuum. The model is exact in attenuation at :math:`E_{eff}`, returns the pure material
at its nominal attenuation and gives at most two materials per voxel. The composition still matters beyond
:math:`E_{eff}`, since the solver's primary and scatter are polychromatic; fewer reference materials give a coarser
composition between them.

Transport and quadrature grids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following Acuros CTS (Maslowski et al. 2018, Med. Phys. 45:1899, Section 2.B), every scatter order is transported on
the transport grid of cubic cells of edge ``transport_cell_cm`` (default 1 cm) that covers the reconstruction box,
whatever the reconstruction voxel size. The first-order quantities (uncollided fluence, first-collision source and the
first-order emission of the scatter image) live on the quadrature grid of :math:`q^3` equal-weight points per cell,
:math:`q` = round(cell / ``quadrature_point_spacing_cm``) (default spacing 0.5 cm), at most round(cell / voxel). The
densities of every cell and point are the exact averages of the fine densities over their volume (conservative overlap
resampling, mass conserving for any ratio of voxel to cell size). Only the primary and this resampling touch the
reconstruction grid. :doc:`conventions`, "Grids", has the definitions.

Uncollided trace
~~~~~~~~~~~~~~~~

``uncollided_trace_grid`` sets the grid through which :math:`\phi_{UC}` is traced to the quadrature points:
``"quadrature"`` (default; boundaries resolved at the point spacing), ``"coarse"`` (the transport grid, which smears a
material boundary inside a cell) or ``"fine"`` (the reconstruction grid, for validation).

Scatter step every n-th outer iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``scatter_every`` :math:`n > 1` the scatter stage runs at the outer iterations :math:`k = 0, n, 2n, \dots` only; in
between the previous :math:`\mathrm{SPR}` is reused and only the MBIR iteration runs. The accumulators keep their age
across the skipped iterations (the recursion is unchanged, it is advanced less often). An MBIR iteration is cheap next
to a scatter stage, so for a given number of scatter steps the extra MBIR iterations cost little; choose
``num_outer_iterations`` as :math:`n` times the number of scatter steps you can afford.

Scatter step on every m-th view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``scatter_view_stride`` :math:`m > 1` the scatter step runs only on the views :math:`0, m, 2m, \dots`, and only
they keep an accumulator. Every other view gets its scatter signal interpolated in angle, per detector pixel, from the
computed views, :math:`s_v = \max(0, \sum_j w_{vj} s_j)`, and divided by its own primary (a projection, computed for
every view). The signal is interpolated rather than the ratio because the scatter is smooth on the panel and in angle,
while the ratio carries the object's shadow edges. ``scatter_view_interpolation_order`` selects the weights: 3
(default) Catmull-Rom on the four nearest computed views, 2 three-point Lagrange, 1 linear, 0 nearest computed view;
``None`` (diagnostics) leaves the skipped views uncorrected. The weights are circular in angle for a
full rotation and run along the arc for a short scan.

Short scans
~~~~~~~~~~~

The views are evenly spaced over :math:`[\beta_s, \beta_e)`; the synthetic data default to a short scan,
:math:`\beta_e = \beta_s + \pi + 2\gamma_m` with :math:`2\gamma_m = 2\arctan(N_c \Delta_c / (2\,\mathrm{SDD}))` the
in-plane fan angle of the panel. MBIRTORCH's FDK assumes a full rotation, so before it the line integrals of a short
scan of range :math:`\Lambda` are weighted with :math:`(\Lambda/\pi)\, w(\beta, \gamma)`, the Parker weights in the
overscan form of Wesarg et al., which make the two measurements of every in-plane line sum to one. MBIR itself needs no
weighting.

MBIR resume
~~~~~~~~~~~

``CitrusConfig.resume_error_sinogram`` (default on) keeps MBIRTORCH's error sinogram :math:`e = y_c - A X` between outer
iterations (stored as :math:`-AX`, one sinogram) and resumes MBIRTORCH's VCD engine from it instead of recomputing it
with a forward projection; a new corrected sinogram only shifts it. The reconstruction equals the restarted one to
float rounding.

Memory
~~~~~~

The densities of all :math:`M` materials never exist on the reconstruction grid: the density model is evaluated one
material map at a time, each map resampled onto the transport and quadrature grids and dropped
(``ScatterSolver.material_fields_from_attenuation``), and the primary projects the maps one at a time again. The
per-view accumulators are stored compactly (the linear-discontinuous cell mean in scaled float16, ``StateStore``): on
the GPU while their total fits ``CitrusConfig.gpu_state_budget_mib``, else in pinned host memory. The aged accumulator
re-enters only through the scattering source of the next sweep, to which the dropped slopes contribute little.
