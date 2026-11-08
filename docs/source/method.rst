Method
======

CITRUS couples model-based iterative reconstruction (MBIR) with a deterministic solver of the linear Boltzmann
transport equation (LBTE). This page describes the coupled loop first, then the scatter solver it drives and the
primary model, and last the options that trade accuracy for cost and memory. Units: cm, keV, g/cm^3; attenuation in
1/cm; source strength 1 photon per steradian at the focal spot of a cone beam, unit incident fluence for a parallel
beam (:doc:`conventions`). The equations below are written for the point source of a cone beam; "Parallel beams"
gives the differences.

The CITRUS loop
---------------

Outer iteration
~~~~~~~~~~~~~~~

MBIR reconstructs a monoenergetic attenuation map :math:`X` at an effective energy :math:`E_{eff}` (the spectrum
mean, or ``CitrusConfig.energy_kev``) from the measured counts :math:`Y`, primary plus scatter, with flat field
:math:`I_0` (a number, or the measured open-field image, below "Source emission profile and open field"). The loop
starts from the FDK reconstruction :math:`X_0` of the uncorrected data (the filtered back projection for a parallel
beam), and outer iteration :math:`k = 1, \dots, N`

1. converts :math:`X_{k-1}` into the densities :math:`\rho_k` of the library's reference materials (the density
   model, below);
2. advances the collided fluence of every view by one transport sweep and projects its scatter signal :math:`s_v`;
3. computes the primary signal :math:`p_v` of every view from :math:`\rho_k`;
4. divides the measured counts by :math:`1 + s_v / p_v` and runs :math:`n` warm-started MBIR iterations on the
   corrected data (``mbir_iterations_per_outer_iteration``), which give :math:`X_k`; with the metal mask the
   measurements through metal get zero weight ("Metal", below).

Steps 1 to 3 are the scatter stage, step 4 the MBIR stage; :math:`n` sets how many MBIR iterations follow each scatter
correction and :math:`N` how many outer iterations run (``num_outer_iterations``; "MBIR iterations per outer
iteration", below).

Age-ordered source iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A converged scatter solve repeats transport sweeps, one per collision order, until the collided fluence has converged
(source iteration, below). CITRUS spends a single sweep per view per outer iteration instead and carries the collided
fluence from one outer iteration to the next. Per view :math:`v` it keeps the accumulated collided moments
:math:`\Phi_v = \sum_{n\ge1}\phi_v^{(n)}` and performs in outer iteration :math:`k` the step

.. math::

   \phi_{UC} = \text{uncollided fluence of } \rho_k, \qquad
   \Phi_v^{(k)} = G_k\,\phi_1^{(k)} + G_k\,\Phi_v^{(k-1)}
   \quad(\text{one sweep of the summed sources}), \qquad
   s_v = \text{scatter image of } (\phi_1^{(k)}, \Phi_v^{(k)}),

with :math:`\phi_1^{(k)}` the first-collision source of the densities :math:`\rho_k` and :math:`G_k` one transport
sweep at those densities: applied to :math:`\phi_1^{(k)}` it gives the once-collided fluence, applied to
:math:`\Phi_v^{(k-1)}` it transports the scattering source of the previous accumulator by one collision order. The
two sources are summed and swept together, so the step costs one sweep. The accumulator expands as

.. math::

   \Phi_v^{(k)} = G_k\,\phi_1^{(k)} + G_k G_{k-1}\,\phi_1^{(k-1)} + G_k G_{k-1} G_{k-2}\,\phi_1^{(k-2)} + \dots

The :math:`n`-th collision order of the current estimate started from the first-collision source of :math:`n-1` outer
iterations earlier: the dominant low orders always come from the newest reconstructions, while stale information is
pushed into the weak, smooth high orders.

With the densities held fixed, :math:`N` steps give :math:`\Phi^{(N)} = \sum_{n=1}^{N} G^{n}\phi_1`, exactly the
source iteration truncated after :math:`N` sweeps (a tested identity, ``citrus_step``). The recursion *is* source
iteration, so the scatter estimate converges together with the reconstruction, for the cost of one sweep per view per
outer iteration.

Scatter-to-primary correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The correction is multiplicative: :math:`Y_c = Y\,p/(p+s) = Y/(1 + \mathrm{SPR})` with :math:`\mathrm{SPR} = s/p` per
pixel, both signals from the solver at the current densities. It is independent of the flat field and of the mismatch
between the solver's polychromatic primary and MBIR's monoenergetic model, and it only rescales the noise of :math:`Y`,
so the weights :math:`W = Y/I_0` of the measured counts are the inverse variances of :math:`y_c = -\log(Y_c/I_0)`
(elementwise for an open-field image).

Warm-started MBIR
~~~~~~~~~~~~~~~~~

Each outer iteration runs :math:`n` vectorized coordinate descent (VCD) iterations of MBIRTORCH on :math:`y_c`,
started from :math:`X_{k-1}` at the position of the partition schedule after the MBIR iterations run so far (MBIR
iterations are counted over the whole loop), with the weights :math:`W` and a regularization that is estimated once
on the first corrected sinogram and then frozen. The partitions are drawn from a random generator seeded with
``[base_seed, i]`` before MBIR iteration :math:`i`, so runs with identical inputs give identical MBIR sequences.
``mbir_recon`` runs the same loop without the solver (no
correction, or a fixed SPR), so a comparison can share the initialization, the weights, the partition sequence and the
regularization with CITRUS.

Schedules
~~~~~~~~~

A ``Schedule`` sets when the scatter estimate is updated, how far each update goes and when the loop stops:

* ``num_outer_iterations`` (N) and ``outer_change_threshold``: the loop stops after N outer iterations, or after the
  first outer iteration whose relative change of the volume, :math:`\|X_k - X_{k-1}\| / \|X_k\|`, falls below
  the threshold;
* ``mbir_iterations_per_outer_iteration`` (n);
* ``scatter_sweeps_per_outer_iteration``: the transport sweeps per computed view in each outer iteration, the collided
  fluence of the view carried over from the last one, or ``"converged"`` for a fresh source iteration per view that
  stops when the relative change of the collided fluence falls below ``scatter_change_threshold``;
* ``num_scatter_updates``: the number of outer iterations, from the first, that make a new scatter estimate (the later
  ones keep the last one).

CITRUS is one sweep per outer iteration and the default, ``Schedule()``; the counts of ``CitrusConfig`` overwrite those
of the schedule when they are set. With more sweeps, every sweep after the first transports the scattering source of
the change of the accumulator, :math:`\Phi_v \leftarrow \Phi_v + G_k(\Phi_v - \Phi_v^{\text{old}})`, which advances
the recursion one more collision order for one sweep and no new first-collision source, so the estimate follows the
volume more closely. Zero sweeps give the first-order scatter alone, ``"converged"`` the decoupled schedule, and
``"converged"`` with one scatter update the sequential schedule (the scatter of :math:`X_0`, then frozen).

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
and :math:`b` the angular weight of the source: 0 for directions outside the collimated field (by default the panel),
else the emission profile toward the panel, 1 by default ("Source emission profile and open field", below); a beam
filter adds its path to every transmission from the focal spot. Interactions:
photoelectric absorption
(local deposition), Compton scattering (Klein-Nishina kinematics with the incoherent scattering function, no Doppler
broadening) and Rayleigh scattering (form factors), and pair production (local deposition) at MV energies, with the
cross sections of NIST XCOM and the angular distributions of xraylib. Every voxel is a mixture of the
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
   \sum_{E\in g} w_E \exp\!\Big(-\sum_m P_m(\mathbf r)\, m_m(E) - \rho_f\, t(\boldsymbol\Omega_{UC})\, m_f(E)\Big),
   \qquad P_m(\mathbf r) = \int_{\mathbf r_0}^{\mathbf r} \rho_m\, dl,

i.e. one ray trace per material present in the volume and a fine-energy table (the fine bins of the library, weights
:math:`w_E` of the spectrum): exact in energy, without the within-group beam-hardening error of group-averaged
attenuation. The last term is the path through the source's beam filter (none by default; "Source emission profile
and open field", below). The collided
fluence satisfies the same LBTE with the **first-collision source** in place of :math:`q`,

.. math::

   S_{FC,g}(\mathbf r, \boldsymbol\Omega) = \sum_{g' \le g}
   \mu_s^{g'\to g}\big(\mathbf r;\ \boldsymbol\Omega_{UC}(\mathbf r)\cdot\boldsymbol\Omega\big)\,
   \phi_{UC,g'}(\mathbf r),

evaluated from the tabulated per-steradian transfer :math:`\mu_s^{g'\to g}(\eta)`, :math:`\eta = \cos\theta`
(no Legendre truncation: the first-collision source keeps the Rayleigh and Compton forward peaks). Both are evaluated
at the points of the quadrature grid, a few equal-weight points per transport cell (see "Transport and quadrature
grids" below).

The table has ``num_eta_table`` nodes (default 2001) and is interpolated linearly between the two nodes around
:math:`\eta`. By default (``angular_table_grid = "adaptive"``) the nodes are not equally spaced in :math:`\eta`: they
follow the curvature of the transfer. In the variable :math:`q = \sin(\theta/2) = \sqrt{(1-\eta)/2}`, in which the
form factors and incoherent scattering functions are smooth, the node density is

.. math::

   \rho(q) \propto (1 - c)\,\frac{\hat\rho(q)}{\int \hat\rho\,dq} + c\, 2q, \qquad
   \hat\rho(q) = \max_{m, g', g} \sqrt{\frac{|\partial_q^2 f_{m g' g}(q)|}{\max_q f_{m g' g}}},

the density that gives linear interpolation the same error relative to the maximum of every transfer curve
:math:`f_{m g' g}` (Compton with its kinematic group fractions plus Rayleigh, for the catalogue's water and cortical
bone in the library's groups, so that every material set shares the nodes), smoothed, with a floor :math:`c = 0.1`
of the density :math:`2q` of a table uniform in :math:`\eta`. Its normalized cumulative integral is the map
:math:`u = h(\eta) \in [0, 1]`, decreasing from 1 at :math:`\eta = 1`, and node :math:`j` sits at
:math:`u = j / (N - 1)`. The map is stored as 512 values at equally spaced :math:`q` and evaluated by linear
interpolation, so a lookup stays index arithmetic: :math:`\eta \to q \to u` (one lookup in the small map table)
:math:`\to u (N - 1)` and the two-row interpolation, linear in :math:`u`. The forward peaks narrow as
:math:`1/E^2` in :math:`1 - \eta`; at MeV energies they are far narrower than the spacing of a uniform table, whose
linear interpolation then spreads them over a whole cell, whereas the adaptive nodes resolve them.
``angular_table_grid = "uniform"`` places the nodes equally in :math:`\eta` (the table of citrus 0.5).

Multigroup and spherical-harmonic discretization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Energy groups are ordered from high to low energy (default 6 groups over 10 keV to the spectrum maximum); there is no
upscatter, so transfers run from group :math:`g'` to :math:`g \ge g'`. The edges cut an estimate of the total photon
fluence spectrum in the object, source plus scattered photons, into equal areas (``group_edges_method =
"scattered_fluence"``, the default), because the groups must resolve the downscattered photons that carry the
multiple-collision scatter. The estimate is a one-dimensional slowing-down balance in the object's medium with a
leakage loss,

.. math::

   \phi(E) = \frac{S(E) + \int_{E' > E} \mu_C(E')\, p(E \mid E')\, \phi(E')\, dE'}
   {\mu_t(E) - \mu_R(E) + 1/L},

with :math:`S` the source spectrum, :math:`\mu_x(E) = \sum_m \bar\rho_m m_{x,m}(E)` the attenuation of the medium
of mean densities :math:`\bar\rho_m` of the reference materials, :math:`p(E \mid E')` the Klein-Nishina energy
distribution of a Compton photon,
Rayleigh scattering left out (it does not change the energy), photons below ``energy_min_kev`` absorbed, solved by
one sweep from the highest energy down. :math:`L` (``grouping_object_size_cm``, default 20 cm) is the characteristic
size of the object, whose leakage cross section is :math:`1/L`: :math:`L \to 0` gives the source spectrum,
:math:`L \to \infty` an infinite medium. The CITRUS loop measures both on the initial reconstruction, before it
builds the library (``CitrusConfig.grouping_object_size = "auto"``). :math:`L` comes from the escape probability of
the object: random rays start at points drawn in proportion to the attenuation, in isotropic directions, and
accumulate the optical depth to the edge of the volume (``citrus.utilities.escape_probability``); under the balance's
leakage model photons escape with probability :math:`P = 1 / (1 + \bar\mu L)`, so
:math:`L = (1/P - 1) / \bar\mu` with :math:`\bar\mu` the attenuation-weighted mean attenuation. The volume is
averaged onto 1 cm cells first, so the estimate depends neither on the voxel pitch nor on the noise of the
reconstruction, and it needs no threshold or mask: a lattice, a shell or a lung-like region count through their
optical depth. For a convex homogeneous body it is close to the mean chord :math:`4V/S`. :math:`\bar\rho` is the
density model's densities averaged over the voxels above the attenuation midway between air and water
(``SolverConfig.grouping_medium = "auto"``; a material name or a mapping of densities sets a known medium instead).
The balance is only a grouping heuristic. Under this rule the energies within each group are weighted
by :math:`\phi(E)` for the group constants too (the other rules weight them by the source spectrum): the photons a
group transports are the source photons in the upper groups and the scattered ones in the lower groups, where the
source spectrum can be flat or empty. ``"equal_fluence"`` gives every group the same number of source
photons (the rule of citrus 0.5), ``"log"`` spaces the edges geometrically; explicit edges (``group_edges_kev``)
override the rule. The scattering kernel of the
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
The rare negative cell means of the collided fluence are kept; they integrate correctly in the moments.

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

The exact moments follow from the Legendre transfer moments,
:math:`E_l = \int P_l(\boldsymbol\Omega\cdot\boldsymbol\Omega_{UC})\,S_{FC,g}\,d\Omega = \sum_{g'\le g}
\mu_{s,l}^{g'\to g}\,\phi_{UC,g'}`. With the quadrature sums :math:`I_k = \sum_m w_m \mu_m^k S_m`,
:math:`\mu_m = \boldsymbol\Omega_m\cdot\boldsymbol\Omega_{UC}`, and :math:`c = E_1/E_0`, the factors are
:math:`\beta = (c I_0 - I_1)/(I_2 - c I_1)` clamped to :math:`[-1, 1]` (0 where :math:`E_0 \le 0` or the denominator
vanishes) and :math:`a = E_0/(I_0 + \beta I_1)` (1 where that denominator is not positive). The :math:`l = 0` moment
is always matched, the :math:`l = 1` moment unless :math:`\beta` was clamped.

On the :math:`f^3` midpoint lattice of a cell (local coordinates :math:`u_j = (2j+1)/f - 1`), the slope moments are
:math:`\mathrm{mean}_j(S_j u_{a,j}) / \mathrm{mean}_j(u_{a,j}^2)`. The discrete second moment is
:math:`(1 - 1/f^2)/3`, not :math:`1/3`, so the continuous Galerkin formula :math:`3\,\mathrm{mean}(S u)` would return
:math:`s(1 - 1/f^2)` for a field :math:`c + s u`; dividing by the discrete moment reproduces a field linear within the
cell exactly. For :math:`f = 1` the slopes are zero.

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

The cell of cone node :math:`(i, j)` spans, in :math:`(\cos\theta, \phi)` about the cone axis :math:`\mathbf a`, the
interval between the cumulative Gauss weights of the polar nodes times :math:`[j\,\Delta\phi, (j+1)\,\Delta\phi]`, so
its solid angle equals its weight. In the gnomonic coordinates :math:`(\alpha, \beta) = (\mathbf d\cdot\mathbf u /
\mathbf d\cdot\mathbf a,\ \mathbf d\cdot\mathbf v / \mathbf d\cdot\mathbf a)` along the panel axes :math:`\mathbf u`,
:math:`\mathbf v`, its widths are :math:`\sqrt{12}` times the standard deviation of :math:`\alpha` (:math:`\beta`)
over the cell, sampled on an equal-area midpoint lattice (``subsamples`` per side, default 16); for a rectangle in the
gnomonic plane this is the exact side length.

The blur is exact up to the height spread of the emission along the direction and the cell shape (a box with the
same marginal widths as the polar x azimuth cell); a direction whose attenuated emission vanishes uses the height
of the centre of the grid box. The box window of width :math:`w` pixels is centred on each pixel, clipped to the
panel and renormalized there, so a width of at most one pixel leaves the image unchanged. The in-cell attenuation
:math:`\tau_j` is exact for a uniform cell whose rays leave through the same face as the ray from the cell centre.
The image is linear in the uncollided fluence and the multiple-collision moments apart from the footprint heights,
which follow the summed emission.

Detector and grid responses
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The detector and grid responses are per incident photon: the signal of group :math:`g` is

.. math::

   \text{signal}_g(\mathbf r_D) = \sum_k w_k \cos\theta_{0,k}\, \psi_{D,g}(\mathbf r_D, \boldsymbol\Omega_k)\,
   G_g(\theta_{1,k}, \theta_{2,k})\, D_g(\theta_{0,k}),

with :math:`\theta_0` the angle to the panel normal, :math:`D` the detector response (keV deposited per photon) and
:math:`G` the anti-scatter-grid transmission at the grid angles. Both are lookup tables, so responses from a Monte
Carlo simulation of a real detector and grid can replace the analytic models.

Every image, primary and scatter, holds four arrays per view: ``fluence``, the scalar fluence
:math:`\Phi_g = \sum_k w_k\, \psi_{D,g}` (photons per cm^2 perpendicular to each direction); ``planar_fluence``,
:math:`\sum_k w_k \cos\theta_{0,k}\, \psi_{D,g}`, the photons crossing unit panel area; ``signal_per_group``, the
signal above (keV/cm^2, or photons/cm^2 for a counting detector); and ``signal``, its sum over the groups. All are
per source photon per steradian.

**Grid angles.** A set of grid lamellae is one-dimensional: in the default direction they run along :math:`u`
(channels), repeat along :math:`v` (rows) and are focused on the line parallel to :math:`u` at the focal distance from
the panel along the panel normal :math:`\hat n` (by default the source-detector distance, a grid focused at the focal
spot; infinite, an unfocused grid, for a parallel beam). With :math:`\mathbf a = -\boldsymbol\Omega` and :math:`\mathbf
f` the vector from the pixel to the focal line,

.. math::

   \theta_1 = \operatorname{atan2}(\mathbf a\cdot\hat v, \mathbf a\cdot\hat n)
            - \operatorname{atan2}(\mathbf f\cdot\hat v, \mathbf f\cdot\hat n), \qquad
   \theta_2 = \operatorname{atan2}(\mathbf a\cdot\hat u, \mathbf a\cdot\hat n),

:math:`\theta_1` in the plane spanned by :math:`\hat v` and :math:`\hat n` (perpendicular to the lamellae) from the
focusing direction and :math:`\theta_2` in the plane spanned by :math:`\hat u` and :math:`\hat n` from the normal.
A primary ray of a grid focused at the focal spot has :math:`\theta_1 = 0` exactly and
:math:`\theta_2 = -\arctan(u / \text{SDD})`. A grid whose lamellae run along the rows (``septa_along="rows"``, parallel
to the rotation axis of a circular scan) swaps the roles of :math:`\hat u` and :math:`\hat v`: it is focused on the line
parallel to :math:`\hat v` and :math:`\theta_1` lies in the plane of :math:`\hat u` and :math:`\hat n`. Its response
varies along the channels, so the scatter image then stores the grid weights per pixel rather than per row. A crossed
or 2-D grid (``septa_along="both"``, as the 2-D collimators of CT detectors) has both sets, each with its own grid
angles; its transmission is the product of the two (``GridResponse.crossed``, ``GridResponse.transmission_at``).

**Detector models.** A detector layer of thickness :math:`t`, density :math:`\rho` and total mass attenuation
:math:`\mu(E)` (``detector_response``: any catalogue material or ``Material``, with the cross sections of the photon
data) stops a photon at incidence angle :math:`\theta_0` with probability
:math:`P(E, \theta_0) = 1 - e^{-\mu(E)\,\rho\,t/\cos\theta_0}`. An energy-integrating detector records
:math:`D = E\,P\,f(E)` and a photon-counting detector :math:`D = P\,c(E)`, with three models of what an interaction
leaves in the layer:

* ``deposition="full"``: every interacting photon deposits its whole energy and is counted, :math:`f = c = 1` (no
  escape of scattered or fluorescence photons);
* ``deposition="local"``: only the energy given to charged particles in the first interaction stays, and the
  secondary photons escape: :math:`f = (\mu_{pe} + \mu_{C}\,\bar T_{KN}(E)/E + \mu_{pair}\,(1 - 2 m_e c^2/E))/\mu`
  with :math:`\bar T_{KN}` the mean Klein-Nishina electron energy, and a counting detector counts every interaction
  except Rayleigh scattering, :math:`c = (\mu - \mu_{R})/\mu` (adequate for a thin layer and at MV energies, where
  full deposition overestimates the signal);
* ``deposition="fluorescence_escape"``: as "local", except that the K-shell vacancy left by a photoabsorption emits
  its K x-rays, and only those that leave the layer are lost (below).

**K x-ray escape.** A photoabsorption at depth :math:`x` (along the normal) ionizes the K shell of element :math:`Z`
with probability :math:`c_Z(E) = w_Z\,\sigma_{pe,Z}(E)\,s_Z(E)/\sigma_{pe}(E)`, with :math:`w_Z` the mass fraction
and :math:`s_Z` the K-shell share of the element's photoabsorption (xraylib's subshell cross sections, 0 below the K
edge of the photon data). The vacancy emits line :math:`l` (energy :math:`E_l`) with probability
:math:`y_l = \omega_K R_l`, the fluorescence yield times the radiative rate (xraylib), isotropically; the photon
leaves the slab without interacting with probability :math:`\tfrac12[E_2(\mu_l x) + E_2(\mu_l (t - x))]`, with
:math:`\mu_l` the layer's total attenuation at :math:`E_l` and :math:`E_2` the exponential integral. The energy left
by an interaction at depth :math:`x` is

.. math::

   d(E, x) = \frac{\mu_{pe}}{\mu}\Big(E - \sum_Z c_Z(E) \sum_{l \in Z} y_l E_l\,
   \tfrac12\big[E_2(\mu_l x) + E_2(\mu_l (t - x))\big]\Big)
   + \frac{\mu_C}{\mu}\,\bar T_{KN}(E) + \frac{\mu_{pair}}{\mu}\,(E - 2 m_e c^2),

and :math:`D(E, \theta_0) = \int_0^t \frac{\mu}{\cos\theta_0} e^{-\mu x/\cos\theta_0} d(E, x)\,dx`. The depth
integral runs over the interaction probability (64 Gauss-Legendre nodes, :math:`x` from its inverse), with the escape
term tabulated on depth nodes clustered at both faces, where it has a logarithmic slope. Compton-scattered and
Rayleigh photons escape as in "local", and the L x-rays stay in the layer. A photon-counting detector counts as with
"local". The model applies to any compound (CsI, Gd₂O₂S, CdTe, ...). Next to a real thick scintillator it lies a
little low, since the scintillator re-absorbs part of its Compton photons; "full" lies well above.

A Monte Carlo table (``tabulated_detector_response``, a function of energy and optionally of the angle) replaces
all three. ``csi_response`` is the full-deposition model of CsI with a coarse built-in attenuation table.

**Grid model.** ``anti_scatter_grid`` builds a grid of septa (strips) of thickness :math:`t` and height :math:`h`
at pitch :math:`p` (``lines_per_cm`` = :math:`1/p`, independent of the pixel pitch; ``ratio`` = :math:`h/(p - t)`) in
an interspace material, both any catalogue material or ``Material``. By default (``model="analytic"``,
``grid_model.grid_signal_transmission``) :math:`G` is the signal of the detector layer behind the grid per incident
photon of energy :math:`E` and direction :math:`\mathbf d = (\tan\theta_1, \tan\theta_2, 1)/|\cdot|` in the frame of
the septa (:math:`\theta_1` across them, :math:`\theta_2` along them, :math:`z` down through the grid), relative to
the signal :math:`D(E, \theta_0)` of the same photon without the grid: :math:`G = G_{direct} + G_{secondary}`, the
photons that cross the grid without interacting and the first-order scatter and fluorescence of the grid. Its inputs
are the geometry, the materials and the detector, so another grid needs no calibration.

*Direct photons.* With :math:`F(x) = \lfloor x/p \rfloor t + \min(\max(x - \lfloor x/p \rfloor p, 0), t)` the
septum length between 0 and :math:`x` across the strips, a photon entering the top of the grid at offset :math:`x_0`
crosses the septa over :math:`\ell_s(x_0) = L\,(F(x_0 + \Delta) - F(x_0))/\Delta` of its path :math:`L = h/d_z`,
:math:`\Delta = h\tan\theta_1`, and

.. math::

   G_{direct} = \frac1p \int_0^p e^{-\mu_s \ell_s(x_0) - \mu_i (L - \ell_s(x_0))}\,dx_0,

with the linear attenuations :math:`\mu_s` of the septa and :math:`\mu_i` of the interspace. :math:`\ell_s` is linear in
:math:`x_0` between the offsets at which the entry or the exit point meets a septum face (:math:`x_0 \equiv 0, t,
-\Delta, t - \Delta` modulo :math:`p`), so the integral is a sum of at most four closed-form exponential integrals:
exact for every angle and every number of septa crossed, with no mean-path approximation.

*Secondaries.* Every point :math:`\mathbf r` of a septum, and of the interspace unless it is a gas, scatters the
photons that reach it (Rayleigh scattering at :math:`E`, Compton scattering to :math:`E'(\cos\theta_s)` with the
incoherent scattering function; xraylib's shapes scaled to XCOM's integrated cross sections, as in the library) and
emits the K and L fluorescence lines of its elements (xraylib's line production with cascades, isotropic). Each
photon is attenuated from the top of the grid to :math:`\mathbf r` and from :math:`\mathbf r` to the bottom along
its exact path through the septa and the interspace (the same :math:`F`), and scored by the detector at its own
energy and incidence angle :math:`\theta_k`:

.. math::

   G_{secondary} = \frac{1}{D(E, \theta_0)} \frac{1}{p} \int_{\text{cell}} \frac{a_{in}(\mathbf r)}{d_z}
   \sum_k w_k \Big[\frac{d\mu_R}{d\Omega} D(E, \theta_k)\,a_k(\mathbf r, E)
   + \frac{d\mu_C}{d\Omega} D(E'_k, \theta_k)\,a_k(\mathbf r, E'_k)
   + \sum_l \frac{q_l(E)}{4\pi} D(E_l, \theta_k)\,a_k(\mathbf r, E_l)\Big]\,dA,

with :math:`a_{in}` the attenuation of the incident photon to :math:`\mathbf r`, :math:`a_k(\mathbf r, E)` that of
a photon of energy :math:`E` from :math:`\mathbf r` along direction :math:`k` and :math:`q_l` the production of
line :math:`l` per unit path. The cell holds 16 x 48 midpoints per material; the outgoing directions
:math:`(\tan a, \tan b, 1)/|\cdot|` lie on Gauss panels in :math:`a` (across the strips) whose edges scale with the
slot angle :math:`\arctan((p - t)/h)`, so that the narrow cones through the slots are resolved, and on fixed
panels in :math:`b`, with their solid angles :math:`w_k`. The secondaries are computed every 5 keV, on both sides of
every absorption edge of the grid's materials and on a logarithmic grid above 150 keV, at grid angles dense within a
few slot angles, and interpolated onto the table; the direct part is computed at every node of the table. The scoring
detector (``detector``) is by default 0.6 mm of CsI with the K x-ray escape. The table is computed when the grid is
built (on the GPU when there is one) and cached by its inputs, in memory and as ``grid_<hash>.npz`` in
``$CITRUS_CACHE_DIR``. The model is first order: photons scattered more than once in the grid, the grid's cover and
the photons the detector itself scatters back are left out.

*Septum shadow.* ``model="septum_shadow"`` (and ``focused_1d_grid``) is the geometric shadow model of earlier
versions. A photon tilted by :math:`\theta_1` passes the interspace only with the open fraction
:math:`o = \max(0, (p - t) - h|\tan\theta_1|)/p`; the mean septum path over all photons is
:math:`(t/p)\,h/\cos\theta_1`, so the shadowed photons cross the mean septum path
:math:`L_s = (t/p)\,h/\cos\theta_1/(1 - o)` (:math:`h` at :math:`\theta_1 = 0`), and
:math:`G = o + (1 - o)\,e^{-\mu_{s}(E)\,\rho_{s}\,L_s}` with the mass attenuation :math:`\mu_s` and density
:math:`\rho_s` of the septa (lead for ``focused_1d_grid``). It does not depend on :math:`\theta_2` and ignores the
interspace, the cover and the grid's secondaries. The nominal ratio of ``focused_1d_grid`` is only a label (with its
default lamellae, :math:`h/(p - t) = 11.5` and :math:`h/p = 9.0` for a nominal 10:1 grid).

Both models are means over one period, exact for a pixel that spans whole periods. Without ``pitch_cm`` or
``lines_per_cm`` the pitch is the pixel pitch across the septa (one septum per pixel, so that every pixel sees the same
septum shadow). The grid transmission enters the primary and the scatter signals alike, so the scatter-to-primary
ratio of the solver is that of the detector behind the grid, in the CITRUS loop and in the synthetic data. For a
crossed grid the product of the two 1-D transmissions is exact for opaque septa, since the shadows along :math:`u`
and :math:`v` do not interact, and counts the septum material twice where the walls of a 2-D grid cross, a correction
of order :math:`(t/p)^2`.

**Responses averaged over each group.** The solver evaluates the detector and the grid at one energy per group, the
group energy. A group that straddles an absorption edge of the grid or the detector (lead's K edge at 88 keV, the I
and Cs K edges at 33 and 36 keV, the Gd K edge at 50 keV) then gets the response of one side of the edge only, and a
small change of the spectrum or of the group edges that moves the group energy across the edge changes the signal of
the whole group. With ``SolverConfig.group_averaged_responses`` the solver replaces both responses, once per library
(``ScatterSolver.responses``, ``detector.group_averaged_responses``), by their means over the fine energies of each
group,

.. math::

   \bar D_g(\theta_0) = \frac{\sum_{E \in g} w_E\, D(E, \theta_0)}{\sum_{E \in g} w_E}, \qquad
   \bar G_g(\theta_1, \theta_2) = \frac{\sum_{E \in g} w_E\, D(E, 0)\, G(E, \theta_1, \theta_2)}
   {\sum_{E \in g} w_E\, D(E, 0)},

with the source photons :math:`w_E` of the fine energies (``fine_weight``); the grid is weighted by the deposit at
normal incidence too, since it multiplies a signal. The averaged tables are constant in energy over each group (two
nodes just inside its edges) and keep the angle nodes, so the scatter image, the primary and the open field evaluate
them as before, at the same cost. A group without source photons keeps its responses at the group energy. The weights
are those of the source spectrum: the average does not follow the hardening of the beam in the object or the
downscatter within a group. Averaging works for any detector, grid and spectrum, where splitting the groups at the
edges would depend on them.

**Attenuation of the built-in models.** The mass attenuation of CsI and lead in ``csi_response`` and
``focused_1d_grid`` are coarse tables of NIST XCOM total
attenuation (coherent scattering included), rounded to 2 to 3 significant digits and interpolated in log-log
(linear extrapolation in log-log outside them). Each absorption edge is a pair of nodes 1 eV apart (I K 33.17 keV,
Cs K 35.98 keV, Pb L3 13.04 keV, Pb K 88.00 keV; the lead L2 and L1 edges are omitted since lead is opaque there).
The interpolation is adequate for detector modelling, not for dosimetry. The tables end at 150 keV (clamped above);
``detector_response`` and ``anti_scatter_grid`` tabulate 10 keV to 20 MeV.

Source emission profile and open field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The source of a cone beam is an isotropic point source at the focal spot, that of a parallel beam a uniform beam over
the panel's area (``SourceModel``). Four settings shape it, each evaluated at the point where the ray from the focal
spot (for a parallel beam, the ray along the beam) meets the panel plane, and each the same in the uncollided fluence,
and with it the first-collision source and every collision order of the scatter, and in the solver's primary:

* **The collimated field.** By default the field is the panel's active area, edge to edge. A real scan often opens
  the blades wider than the panel sees (a half-fan beam past the panel's short edge, axial blades wider than the rows);
  the tissue there is irradiated and scatters into the panel. ``field_margin_channels_cm`` and
  ``field_margin_rows_cm`` widen the field past both panel edges along the channels and the rows, in cm at the panel.
* **Off-focal emission.** The tube also emits from around the focal spot, and the blades do not stop that radiation.
  Past the field edge the weight is a tabulated fraction of the edge emission (``off_focal_distance_cm``,
  ``off_focal_fraction``), interpolated linearly in the distance :math:`d` past the edge, its first value below the
  first distance and 0 beyond the last; past both a row and a channel edge :math:`d` is the distance from the ray's
  panel point to the field rectangle, :math:`\sqrt{d_u^2 + d_v^2}`. The table can be measured on a scan's own air
  columns past the blades, as the signal there over the signal in the field.
* **The emission profile** :math:`p` (``emission_profile``), given per detector pixel: the heel effect, or a bowtie
  known only as a gray weight. It is interpolated bilinearly between the pixel centres and held at the value of the
  outermost centres beyond them, so the margins and the off-focal region get the emission of the field's edge.
* **The beam filter** (a bowtie): ``filter_material`` (a catalogue name or a ``Material``, at its nominal density
  :math:`\rho_f`) and ``filter_thickness_cm``, the path length :math:`t` through the filter of the ray from the focal
  spot to each pixel centre (one row for every row, or one value per pixel), its minimum thickness included,
  interpolated and held like the profile.

The angular weight is :math:`b = c\,p` with :math:`c` 1 in the field, the off-focal fraction past it and 0 elsewhere.
The filter does not enter :math:`b`: it adds :math:`\rho_f\,t\,m_f(E)` to the optical depth of every transmission that
starts at the focal spot, with the filter's total mass attenuation :math:`m_f` from the photon data at the library's
fine energies, so the spectrum hardens per direction exactly in energy: under a bowtie's thick wings the uncollided
fluence, the scatter and the primary all have the harder spectrum. A filter of zero thickness gives the results of no
filter. The library's energy groups and effective energy follow the spectrum as given, so the spectrum is the one
without the filter. The scale of :math:`b` cancels in the scatter-to-primary ratio.

The thickness of a filter is rarely given; its transmission is measured as the open field with the filter over the
open field without it. ``filter_thickness_from_transmission`` inverts that ratio per pixel in the model's
transmission of a uniform filter of each thickness (the solver's open field with the filter over the one without),
interpolated linearly in the log of the transmission, with the solver's spectrum and responses. The ratio must not be
normalized to its maximum, or the filter's minimum thickness is lost; a profile normalized to its maximum is first
scaled to the model's transmission of a given minimum thickness.

The profile comes from the measured open field :math:`F` (flat-field image) of the scanner. The solver's images are
per source photon per steradian through the true geometry, so its own open field of its source without a profile,
:math:`F_{solver}` (``ScatterSolver.open_field_signal``, the primary signal of an empty volume), already contains the
inverse-square and obliquity falloff of a point source (none for a parallel beam), the attenuation of the beam filter
and the grid and detector responses. With a per-pixel detector gain :math:`g` (the open field is gain times
emission),

.. math::

   p = \frac{F}{g\,F_{solver}} \Big/ \operatorname{mean}\Big(\frac{F}{g\,F_{solver}}\Big),

per view for a per-view open field (``emission_profile_from_open_field``). No absolute scale or :math:`I_0` is
involved. With the filter in the source the profile holds only what the model does not explain (the heel effect);
without it the bowtie appears in :math:`p` as a gray weight, the same spectrum in every direction. Without a gain map
the whole open field is treated as emission; a gain pattern then appears in :math:`p`, which matters only where it
is large. ``citrus_recon`` given an open-field image normalizes the counts pixel by pixel, :math:`y = -\log(Y/F)`,
and gives the solver's source this profile for the duration of the call, keeping every other setting of the source;
given a number :math:`I_0` it keeps the solver's source, and :math:`I_0` then stands for the open field with the
falloff of :math:`F_{solver}` divided out.

One view of the solver
~~~~~~~~~~~~~~~~~~~~~~

``ScatterSolver.solve_view`` computes one view in this order:

1. the density-only fields (``MaterialFields``): the transport grid C and the quadrature grid Q of the fine grid,
   the densities resampled onto both, and the total attenuation and Legendre transfer moments on C;
2. the uncollided fluence at the points of Q, traced through the grid set by ``uncollided_trace_grid``;
3. source iteration on C, driven by the first-collision source: evaluated at the points of Q for the sweep
   directions with the exact transfer table, rescaled by the sampling correction and restricted to the
   linear-discontinuous moments of the transport cells;
4. the cell-mean moments of the multiple-collision source from the collided fluence;
5. the scatter image of both sources in one collapsed-cone pass;
6. the primary image on the fine grid, and the scatter fraction :math:`S/(P+S)` of the signals.

The collided transport (sweeps and multiple-collision source) uses the library truncated to Legendre order
:math:`L`; the uncollided fluence, the first-collision source and the primary use the full library (true
:math:`\mu_t` and the exact transfer table). The solver keeps no state between calls apart from caches of objects
that depend only on the grids, the quadratures and the view (transport grids, sweep contexts, collapsed-cone
direction sets, and the ``direction_context_cache_size`` most recently used direction contexts);
``ScatterSolver.clear_caches`` drops them.

The primary
-----------

The primary signal uses the same detector rule with the per-pixel direction from the focal spot (along the beam for
a parallel beam) and the exact-in-energy
transmission of the density line integrals, averaged over the pixel after the energy sum. It is computed in two steps,
the per-material line integrals :math:`P_m = \int \rho_m\,dl` (a projector) and the transmission with the responses
(``scatter_image.primary_from_path_lengths``), so any projector can feed it:

* in the CITRUS loop, MBIRTORCH's pixel-averaged separable-footprint projector, the reconstruction's own operator
  :math:`A`: one pass per material present projects its density map onto all views (in chunks of views whose path
  lengths stay within one sinogram), and the polychromatic transmission is formed per view (``primary_signals``). A
  pixel-integrated primary is the physical model of a detector pixel, and it is consistent with the synthetic data,
  whose scatter is defined through the same primary;
* in the standalone solver (``solve_view``), the solver's Siddon projector, one ray to each pixel centre.

In the loop the views are projected in chunks of ``max(1, V // (M' + 1))`` consecutive views (``M'`` materials
present; ``CitrusConfig.primary_views_per_chunk`` overrides it), so that the path lengths ``[M', chunk, rows,
channels]`` and the projection being copied stay within one sinogram. The solver's sub-pixel sampling of the Siddon
primary (``primary_samples_per_pixel_axis``) does not apply there: the projector already averages the line integral
over the pixel.

The scatter fraction is :math:`S/(P+S)`.

Options and secondary components
--------------------------------

Density model and reference materials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The solver needs the densities of the library's reference materials, chosen from the material catalogue with
``SolverConfig.materials`` (at least two; by default air, lung, adipose, water, muscle, soft tissue and cortical bone;
the catalogue with compositions, nominal densities :math:`\rho_m^{nom}` and nominal attenuations
:math:`\mu_m = \rho_m^{nom}\, m_m(E_{eff})` is in :doc:`conventions`, "Materials"). The library sorts them by nominal
attenuation at the effective energy and refuses only neighbours whose nominal attenuations there are nearly equal
(closer than 0.1 %, ``cross_section_library.MINIMUM_ATTENUATION_GAP``), which the model could not tell apart; materials
may cross in attenuation at other energies of the spectrum, since the model works at one energy. Each voxel gets the two
consecutive
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

Scatter scene beyond the reconstructed slab
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The solver sees the reconstructed volume only, but the body continues past the slab the cone measures, and the field
past the panel's rows (margins, off-focal emission) irradiates more of it; its scatter reaches the panel. A grid longer
than the measured slab holds that region, but the reconstruction there is incomplete. With
``CitrusConfig.extend_scatter_scene`` the solver's scene continues the body along z within the grid
(``extend_scene_slices``): a slice is filled when its fraction of voxels above half of water's attenuation is at least
0.8 of the median fraction over the central 5 cm, and every slice beyond the first and last filled slice, moved
two slices inward, repeats that slice. The reconstruction is unchanged, and the grid must reach as far along z as
the scene should.

What the scatter solver sees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A reconstruction holds noise, streaks and a faint haze in the air around the object. The scatter solver would treat
them as matter that scatters, so two settings clear them from the volume it sees, while the reconstruction keeps them.
``CitrusConfig.air_threshold_hu`` sets every voxel below that value to air at every scatter update.
``scatter_scene_support_hu`` builds a support at every scatter update from the current volume without the known
objects: that volume is smoothed by a Gaussian of 0.8 cm standard deviation and thresholded, the result is dilated by
``scatter_scene_support_margin_cm`` along each axis, and it is cut to the field of view, the cylinder about the
rotation axis that the widest ray reaches (``scatter_scene_support``, ``utilities.field_of_view_radius``). Outside
the support the solver sees air. Its threshold sits well below soft tissue, so the object stays inside it while the
haze is too faint to pass it. Outside the field of view the reconstruction is not determined by the data, and MBIR
can leave a haze there that the solver would take for matter beside the object; an object wider than the field of
view keeps its outer part with ``scatter_scene_beyond_field_of_view``. HU values are converted with water's
attenuation at the density model's effective energy. The primary, which divides the scatter in the scatter-to-primary ratio, is
that of the same volume, so the scatter and the primary describe one scene.

Objects known in advance, such as the patient couch from its model, are given as an attenuation volume
(``known_objects``). Where it is positive it replaces the volume in the scatter solver's scene and in the modelled
primary, so the scatter and the scatter-to-primary ratio include them; their forward projection at the effective
energy is removed from the corrected line integrals of every outer iteration, after the water linearization, so MBIR
reconstructs only the rest. This suits an object that lies partly outside the field of view, which MBIR could not
reconstruct.

Beam hardening
~~~~~~~~~~~~~~

MBIR reconstructs a monoenergetic attenuation from polychromatic data, so the line integrals of thick or dense paths
are too low (cupping in water, dark streaks between bones), and under a bowtie the wings' harder spectrum changes
that error along the channels. Two options of ``CitrusConfig`` correct the line integrals of every outer iteration
before its MBIR iterations, with the solver's own primary model:

* ``hardening_materials``: library materials treated as sources of hardening beyond water. After every scatter
  update each line integral gets the offset :math:`\ln(P / P_{ref})`, with :math:`P` the solver's polychromatic
  primary of the current volume and :math:`P_{ref}` the same primary with the path of every listed material replaced
  by water of the same monoenergetic attenuation at the density model's effective energy, adding
  :math:`(\mu_m(E_{eff}) / \mu_{water}(E_{eff}))\,P_m` to water's path; the other materials stay. The two primaries
  share the projections of the scatter update, and the beam filter is in both. After the offset the data look like
  data of the volume with the listed materials made of water.
* ``beam_hardening_correction``: the water linearization of every pixel (``water_linearization``). For each pixel
  the solver's primary signal :math:`S_p(L)` through water thicknesses :math:`L` (up to 60 cm, with the source's
  filter, the detector and the grid; the emission profile cancels) gives water's polychromatic line integral
  :math:`-\ln(S_p(L) / S_p(0))`, and a line integral is mapped to :math:`\mu_{water}(E_{eff})\,L`, :math:`L`
  interpolated linearly in the pixel's table and continued with its end segments. The table depends on the pixel
  through the filter's thickness and the incidence angle, not on the view of a circular or helical scan.

The offset comes first, so that the water linearization applies to water-like data. ``mbir_recon`` takes the water
linearization too, given the solver of the scanner (``solver=``); the offset needs the solver's primary of the current
volume and belongs to the CITRUS loop. Synthetic data with ``polychromatic_primary`` (``synthesize_cone_beam_data``)
have the primary of the solver's transmission, with these effects; by default their primary is monoenergetic.

MBIR iterations per outer iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``mbir_iterations_per_outer_iteration`` :math:`n > 1` every outer iteration runs :math:`n` MBIR iterations on the
same corrected counts before the next scatter correction. An MBIR iteration is cheap next to a scatter stage, so for a
given number of outer iterations (scatter steps) the extra MBIR iterations cost little and let MBIR converge further
towards each correction; the recursion of the accumulators is unchanged, it is advanced once per outer iteration.

At a fixed total number of MBIR iterations the trade-off reverses: a larger :math:`n` leaves fewer scatter stages, and
between two of them MBIR converges towards a correction that is no longer refreshed.

Both counts are fixed. Each scatter correction changes the corrected sinogram by a fraction of the change before it,
and that fraction grows with the scatter-to-primary ratio, so strong scatter (a large object, no grid, a low energy)
needs more outer iterations than weak scatter; the records' ``scatter_change_relative`` shows how much the scatter
estimate still moves. The residual :math:`\|A X - y_c\| / \|y_c\|` of the corrected sinogram shows the MBIR side
without a ground truth: within an outer iteration it falls quickly for the first MBIR iterations and then flattens,
because MBIR has reached the volume that best fits the current correction under the prior, and only a new correction
can lower it further. With the resumed error sinogram (the default) the residual costs nothing; the records hold it at
the start and the end of every outer iteration (``sinogram_nrmse_start``, ``sinogram_nrmse``).

Subsets per MBIR iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~

An MBIR iteration updates the pixel columns of the volume one random subset at a time: the columns of a subset are
updated together from the same error sinogram, which is then updated before the next subset. Columns of a subset that
lie on a common ray interact, so the subsets must be sparse enough that a ray crosses only a few columns of each;
past that, more subsets no longer speed up the convergence per iteration, while every subset costs the engine a fixed
overhead. ``mbir_subsets_per_iteration = "auto"`` (the default) therefore takes the power of 2 nearest a quarter of the
largest in-plane extent of the volume in voxels when that is 16 to 64 (``recon.mbir_bridge.vcd_partitions``); as in
MBIRTORCH's default schedule, the first MBIR iterations of the loop use 4 and 16 subsets (those fewer than the count),
which take up the large change from the initial volume more evenly. Outside that range it keeps MBIRTORCH's own
schedule, which ends at 128 subsets per iteration: volumes narrower than about 46 voxels converge best per iteration
with many subsets of few columns, and on volumes wider than about 360 voxels 128 is the count the rule asks for. An
integer sets the count, and ``None`` always keeps the model's own partition schedule.

Metal
~~~~~

A measurement whose ray crosses metal fits MBIR's linear model poorly even when it has plenty of counts: the line
integral changes steeply across the pixel (the exponential edge-gradient effect), the spectrum hardens, and through a
thick piece no primary photon gets through, so the corrected data of the ray rest on the solver's scatter estimate.
MBIR spreads such errors along the rays as streaks. With ``CitrusConfig.metal_attenuation_threshold`` (in 1/cm) every
outer iteration marks the voxels of the current volume above the threshold, forward projects them and gives zero
weight to every measurement whose ray crosses one; the residual (``sinogram_nrmse``) is computed over the other
measurements, and ``records[k]["metal_measurement_fraction"]`` holds the share left out. The mask works on
attenuation, so it applies to any material, including user-defined ones; the threshold must lie above every material
of the object that is not metal (the nominal attenuations of the catalogue at the effective energy are in
:doc:`conventions`, "Materials"). Since the threshold is an attenuation at the effective energy, a threshold set for
a kilovoltage beam does not mark metal in a megavoltage beam, which passes through it. The default ``None`` turns the
mask off.

Without data, the metal voxels fade under the regularization, and the metal set shrinks with them from the blurred
metal of the initial volume to its core, which tightens the mask. The volume the scatter solver sees and the returned
volume therefore get a filled metal (``metal_fill``): the current metal voxels take the values of the volume in which
the metal was first found (usually the initial volume), scaled so that their sum equals that volume's total over its
own metal voxels, since blur spreads the attenuation of the metal but keeps its sum. The records hold the size of the
set and the scale (``metal_voxels``, ``metal_fill_scale``). The metal's attenuation is as good as the initial volume's
total: the counts floor ``min_counts`` caps every measured line integral at :math:`\log(I_0 / \mathrm{min\_counts})`,
so the data of a thick piece cannot hold its attenuation; the option is for the tissue around the metal.

Scatter step on every m-th view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``scatter_view_stride`` :math:`m > 1` the scatter step runs only on the views :math:`0, m, 2m, \dots`, and only
they keep an accumulator. Every other view gets its scatter signal interpolated in angle, per detector pixel, from the
computed views, :math:`s_v = \max(0, \sum_j w_{vj} s_j)`, and divided by its own primary (a projection, computed for
every view). The signal is interpolated rather than the ratio because the scatter is smooth on the panel and in angle,
while the ratio carries the object's shadow edges. ``scatter_view_interpolation_order`` selects the weights: 3
(default) Catmull-Rom on the four nearest computed views, 2 three-point Lagrange, 1 linear, 0 nearest computed view;
``None`` (diagnostics) leaves the skipped views uncorrected. The weights are circular in angle for a
full rotation, run along the arc for a short scan and along the unwrapped angles for a helical scan.

With :math:`j_0, j_1` the computed views bracketing a skipped view and :math:`t \in [0, 1)` its fractional position
between them, order 0 takes the nearer of the two (:math:`t = 1/2` goes to :math:`j_0`), order 1 the weights
:math:`1 - t, t`, order 2 the three-point Lagrange interpolant on the nearest computed view and its two neighbours
at their actual angles (exact for quadratics of the angle on any spacing, but discontinuous where the nearest view
changes), and order 3 Catmull-Rom on :math:`j_0 - 1, j_0, j_1, j_1 + 1` (interpolating, with a continuous first
derivative, exact for polynomials of the angle up to degree 2 on uniformly spaced computed views and for linear ones
on any spacing; no local four-point :math:`C^1` interpolant is exact for cubics, and its outer weights are
negative, hence the clamp at zero). With fewer computed views than the order needs, the order is lowered to the
number of computed views minus one. For a short scan the angles are unwrapped along the arc (starting after the
largest gap), the Catmull-Rom nodes beyond the end views are linear extrapolations (so it stays exact for linear
functions), the quadratic stencil shifts inside, and a view outside the computed arc gets the nearest end view. A
view set counts as a full rotation when its largest circular gap is at most 1.5 times the median gap. With order
``None`` the skipped views keep their most recent computed scatter, which with the fixed view subset is zero, so they
are not corrected.

Short scans and redundancy weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The views are evenly spaced over :math:`[\beta_s, \beta_e)`; the synthetic data default to a short scan, :math:`\beta_e
= \beta_s + \pi + 2\gamma_m` with :math:`2\gamma_m = 2\arctan(N_c \Delta_c / (2\,\mathrm{SDD}))` the in-plane fan angle
of the panel. MBIRTORCH's FDK and filtered back projection weight every view by :math:`\pi / V`, which counts every line
equally only over a whole number of rotations. Over any other range of :math:`\Lambda` the line integrals are weighted
with :math:`(\Lambda/\pi)\, w(\beta, \gamma)` before the direct reconstruction
(``recon.mbir_bridge.redundancy_weights``), so that the measurements of every in-plane line sum to one; the weights are
those of cabouman/mbirtorch pull request 30, and an MBIRTORCH whose ``recon_fdk`` or ``recon_fbp`` applies them itself
is left to do so. Below a full rotation they are the Parker weights in the overscan form of Wesarg et al.; above it they
ramp over the excess beyond the whole rotations at both ends of the range. A parallel beam is the case :math:`\gamma =
0`. MBIR itself needs no weighting.

With the view position :math:`\beta = (i + 1/2)\,\Lambda / V` along the arc, the overscan
:math:`\delta = (\Lambda - \pi)/2` and the in-plane fan angle :math:`\gamma = -\arctan(u / \mathrm{SDD})` of channel
:math:`u` (the sign of MBIRTORCH's frame),

.. math::

   w(\beta, \gamma) = \begin{cases}
     \sin^2\!\big(\tfrac{\pi}{4}\,\beta / (\delta - \gamma)\big) & 0 \le \beta < 2(\delta - \gamma), \\
     1 & 2(\delta - \gamma) \le \beta < \pi - 2\gamma, \\
     \sin^2\!\big(\tfrac{\pi}{4}\,(\pi + 2\delta - \beta) / (\delta + \gamma)\big)
       & \pi - 2\gamma \le \beta \le \pi + 2\delta.
   \end{cases}

The weights need :math:`\delta \ge \gamma_m`, i.e. a range of at least :math:`\pi` plus the fan angle; a shorter
range gets no weights and a warning (the data are incomplete). Above a full rotation of :math:`n` whole rotations
and excess :math:`e = \Lambda - 2\pi n`, :math:`w = \sin^2(\tfrac{\pi}{2}\beta/e)` for :math:`\beta < e`,
:math:`\sin^2(\tfrac{\pi}{2}(\Lambda - \beta)/e)` for :math:`\beta > 2\pi n`, 1 in between, all divided by
:math:`2n`. They act per channel, the in-plane weighting, approximate off the central plane like FDK itself. On a
curved panel the fan angle of a channel is :math:`u / \mathrm{SDD}`.

Curved panels and helical scans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both follow MBIRTORCH's ``ConeBeamModel`` (``use_curved_detector``, ``helical_z_shifts``), whose projector is the
primary of the CITRUS loop, so the solver places the source and the pixels exactly where MBIRTORCH projects
(:doc:`conventions`, "Frames").

**Curved panel.** The pixels lie on the cylinder of radius SDD about the vertical axis through the source, and
everything the solver does at the panel follows the cylinder: the primary rays and the oblique lines of the scatter
image run through the pixel centers on it, the downstream optical depth ends on it, the source collimation tests the
ray's hit on it, and the incidence angle :math:`\theta_0`, the planar fluence and the detector and grid responses use
the local normal of each pixel, which is horizontal and points to the axis. The scatter-image weights therefore depend
on the direction and the channel, :math:`w_k \cos\theta_{0,kc}\, D_g(\theta_{0,kc})`. So does the transmission of
lamellae along the rows. The transmission of lamellae along the channels varies with the row as well, because the rows
of the cylinder are at the height :math:`v\cos\gamma`; their grid angle is the incidence angle on the local axes, which
depends on the direction and the channel, minus the focusing angle of the pixel, which does not depend on the direction,
so it is evaluated per pixel for each chunk of directions from these two parts. Lamellae along the channels run along
the arc and each column is focused along its own normal; lamellae along the rows (``septa_along="rows"``, the usual
clinical grid) are focused along each pixel's normal; a crossed grid has both. Both focus on the focal spot at the
default focal distance. The collapsed-cone direction set keeps the axis of the normal at the panel center and covers
the cylinder's edges, and the finite-cone footprint is computed on the plane tangent to the cylinder at the central ray
(the arc length and the tangent-plane coordinate agree to first order in the fan angle).

**Helical scan.** Every view carries a z shift of the source and the panel (a table of unwrapped view angles and shifts
in the geometry), used by the scatter solver for each view, by MBIRTORCH and by the synthetic data. The reconstruction
grid must hold the whole object that the helix illuminates (the projectors take the volume as zero outside it); the
transport grid covers the reconstruction grid as for a circular scan, and each view's solve covers the whole grid (the
collimation leaves the slabs outside the view's field dark). No redundancy weights are applied: per-row helical weights,
tried in the MBIRTORCH pull request and removed there, gave a worse starting volume near the ends of the helix, where
few views see a slice. The initial volume is MBIRTORCH's FDK, which for a helical scan scales every slice by the inverse
fraction of the views that see it: an approximate reconstruction, adequate as the start of MBIR, which is exact for the
helical geometry. With ``scatter_view_stride`` :math:`m > 1` the skipped views are interpolated along the helix, in the
unwrapped angle between consecutive computed views, never between views a turn apart at another z. Two computed
neighbours differ in z by :math:`m` times the table feed per view, so each neighbour's image is first shifted along the
rows by the z shift between it and the interpolated view, :math:`s_j(r + (z_v - z_j)/\Delta_{row})`. The shift is not
magnified: the scatter comes from the object, which moves by :math:`z_v - z_j` relative to the panel, not from the focal
spot, while the irradiated slab, and with it the scatter's envelope, stays fixed on the panel. The stride should still
keep the z step well below the height of the panel at the isocenter.

Parallel beams
~~~~~~~~~~~~~~

A parallel beam (``ParallelGeometry``) follows MBIRTORCH's ``ParallelBeamModel``: the rays of a view are parallel and
perpendicular to a flat panel at the distance ``axis_detector_dist`` from the rotation axis (:doc:`conventions`,
"Frames"). There is no focal spot, and every notion of the solver that depends on it is taken along the beam
direction :math:`\mathbf w`:

* the source is a unit fluence of parallel rays over the beam's cross-section, the collimated field (by default the
  panel's area), weighted per pixel by the emission profile (``SourceModel.beam_weight``): :math:`b(\mathbf r)` is the
  field weight times the profile at the panel point of the ray through :math:`\mathbf r`, and the beam filter adds its
  path to the transmission of that ray;
* the uncollided fluence has no inverse-square factor and a single direction,
  :math:`\phi_{UC,g}(\mathbf r) = b(\mathbf r) \sum_{E\in g} w_E \exp(-\sum_m P_m(\mathbf r)\, m_m(E))`,
  :math:`\boldsymbol\Omega_{UC} = \mathbf w`, the line integrals :math:`P_m` traced along :math:`\mathbf w` from
  upstream of the grid box; the first-collision source, its sampling correction and every collision order follow;
* the primary rays are the beam's rays through the pixels, at normal incidence (:math:`\theta_0 = 0`), and the open
  field is flat;
* the anti-scatter grid of a parallel beam is unfocused: its default focal distance is infinite
  (``detector.default_focal_distance``), the septa run along the panel normal, :math:`\theta_1` is measured from the
  normal and the primary passes at :math:`\theta_1 = \theta_2 = 0`;
* the collapsed-cone scatter image is unchanged: it depends only on the panel, and its direction set covers the grid
  box as seen from the panel.

The scatter that reaches the panel falls as the air gap ``axis_detector_dist`` grows (the scattered photons spread
over a larger solid angle and leave the panel), while the primary does not depend on it. The Triton ray tracer and
the first-collision kernel have a parallel mode (the ``PARALLEL`` constant), checked against the PyTorch path like the
cone modes. MBIRTORCH reconstructs the parallel beam with ``ParallelBeamModel``, whose initial volume is its filtered
back projection (``recon.mbir_bridge.direct_reconstruction``), with the redundancy weights of a range that is not
whole rotations (fan angle 0, above; over 180 degrees they are all one); the default range of a parallel scan
(``view_angles``) is 180 degrees.
The sparse-view interpolation runs along the arc of such a scan, as for a short cone-beam scan: the views at
:math:`\beta` and :math:`\beta + \pi` see the object from opposite sides of the panel, so their scatter differs.

MBIR resume
~~~~~~~~~~~

``CitrusConfig.resume_error_sinogram`` (default on) keeps MBIRTORCH's error sinogram :math:`e = y_c - A X` between outer
iterations (stored as :math:`-AX`, one sinogram) and resumes MBIRTORCH's VCD engine from it instead of recomputing it
with a forward projection; a new corrected sinogram only shifts it. The reconstruction equals the restarted one to
float rounding.

Memory
~~~~~~

The densities of all :math:`M` materials never exist on the reconstruction grid unless they are small: the density
model is evaluated one material map at a time, each map resampled onto the transport and quadrature grids and dropped
(``ScatterSolver.material_fields_from_attenuation``), and the primary projects the maps one at a time again; maps that
fit a fixed budget are kept for the primary of every view instead. The
per-view accumulators are stored compactly (the linear-discontinuous cell mean in scaled float16, ``StateStore``): on
the GPU while their total fits ``CitrusConfig.gpu_state_budget_mib``, else in pinned host memory. The aged accumulator
re-enters only through the scattering source of the next sweep, to which the dropped slopes contribute little.
