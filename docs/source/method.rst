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
reconstruction :math:`X_0` of the uncorrected data, and outer iteration :math:`k = 1, \dots, N`

1. converts :math:`X_{k-1}` into the densities :math:`\rho_k` of the library's reference materials (the density
   model, below);
2. advances the collided fluence of every view by one transport sweep and projects its scatter signal :math:`s_v`;
3. computes the primary signal :math:`p_v` of every view from :math:`\rho_k`;
4. divides the measured counts by :math:`1 + s_v / p_v` and runs :math:`n` warm-started MBIR iterations on the
   corrected data (``mbir_iterations_per_outer_iteration``), which give :math:`X_k`.

Steps 1 to 3 are the scatter stage, step 4 the MBIR stage; :math:`n` sets how many MBIR iterations follow each scatter
correction ("MBIR iterations per outer iteration", below).

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
so the weights :math:`W = Y/I_0` of the measured counts are the exact weights of :math:`y_c = -\log(Y_c/I_0)`.

Warm-started MBIR
~~~~~~~~~~~~~~~~~

Each outer iteration runs :math:`n` vectorized coordinate descent (VCD) iterations of MBIRTORCH on :math:`y_c`,
started from :math:`X_{k-1}` at position :math:`(k-1)n` of the partition schedule, with the weights :math:`W` and a
regularization that is estimated once and then frozen. The partitions are drawn from a random generator seeded with
``[base_seed, i]`` before MBIR iteration :math:`i`, so runs with identical inputs give identical MBIR sequences.
``mbir_recon`` runs the same loop without the solver (no
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
   \sum_{E\in g} w_E \exp\!\Big(-\sum_m P_m(\mathbf r)\, m_m(E)\Big),
   \qquad P_m(\mathbf r) = \int_{\mathbf r_0}^{\mathbf r} \rho_m\, dl,

i.e. one ray trace per material present in the volume and a fine-energy table (0.5 keV bins, weights :math:`w_E` of the
spectrum): exact in energy, without the within-group beam-hardening error of group-averaged attenuation. The collided
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
chord length of the object: :math:`L \to 0` gives the source spectrum, :math:`L \to \infty` an infinite medium. The
CITRUS loop measures both on the initial reconstruction, before it builds the library: :math:`L` as the mean chord
:math:`4V/S` of the object (the voxels above the attenuation midway between air and water;
``CitrusConfig.grouping_object_size = "auto"``) and :math:`\bar\rho` as the density model's densities averaged over
those voxels (``SolverConfig.grouping_medium = "auto"``; a catalogue name or a mapping of densities sets a known
medium instead). The balance is only a grouping heuristic. Under this rule the energies within each group are weighted
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

**Grid angles.** The anti-scatter grid is one-dimensional: its lamellae run along :math:`u` (channels), repeat along
:math:`v` (rows) and are focused on the line parallel to :math:`u` at the focal distance from the panel along the
panel normal :math:`\hat n` (by default the source-detector distance, a grid focused at the focal spot). With
:math:`\mathbf a = -\boldsymbol\Omega` and :math:`\mathbf f` the vector from the pixel to the focal line,

.. math::

   \theta_1 = \operatorname{atan2}(\mathbf a\cdot\hat v, \mathbf a\cdot\hat n)
            - \operatorname{atan2}(\mathbf f\cdot\hat v, \mathbf f\cdot\hat n), \qquad
   \theta_2 = \operatorname{atan2}(\mathbf a\cdot\hat u, \mathbf a\cdot\hat n),

:math:`\theta_1` in the plane spanned by :math:`\hat v` and :math:`\hat n` (perpendicular to the lamellae) from the
focusing direction and :math:`\theta_2` in the plane spanned by :math:`\hat u` and :math:`\hat n` from the normal.
A primary ray of a grid focused at the focal spot has :math:`\theta_1 = 0` exactly and
:math:`\theta_2 = -\arctan(u / \text{SDD})`.

**Analytic models.** ``csi_response`` assumes that every interacting photon deposits its full energy (no escape, no
fluorescence) along the oblique path :math:`t/\cos\theta_0`:
:math:`D(E, \theta_0) = E\,\big(1 - e^{-\mu_{CsI}(E)\,\rho\,t/\cos\theta_0}\big)`. ``focused_1d_grid`` is a
geometric lamella-shadow model for lamellae of thickness :math:`t` and height :math:`h` at pitch :math:`p`. A photon
tilted by :math:`\theta_1` passes the interspace only with the open fraction
:math:`o = \max(0, (p - t) - h|\tan\theta_1|)/p`; the mean lead path over all photons is
:math:`(t/p)\,h/\cos\theta_1`, so the shadowed photons cross the mean lead path
:math:`L_s = (t/p)\,h/\cos\theta_1/(1 - o)` (:math:`h` at :math:`\theta_1 = 0`), and
:math:`G = o + (1 - o)\,e^{-\mu_{Pb}(E)\,\rho_{Pb}\,L_s}`. A tilt along the lamellae does not change the shadow, so
:math:`G` does not depend on :math:`\theta_2`; the interspace material and the cover are ignored. The geometry is
set by :math:`(t, h, p)` alone: the nominal ratio is only a label (with the default lamellae,
:math:`h/(p - t) = 11.5` and :math:`h/p = 9.0` for a nominal 10:1 grid).

**Attenuation of the models.** The mass attenuation of CsI and lead are coarse tables of NIST XCOM total
attenuation (coherent scattering included), rounded to 2 to 3 significant digits and interpolated in log-log
(linear extrapolation in log-log outside them). Each absorption edge is a pair of nodes 1 eV apart (I K 33.17 keV,
Cs K 35.98 keV, Pb L3 13.04 keV, Pb K 88.00 keV; the lead L2 and L1 edges are omitted since lead is opaque there).
The interpolation is adequate for detector modelling, not for dosimetry.

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

MBIR iterations per outer iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``mbir_iterations_per_outer_iteration`` :math:`n > 1` every outer iteration runs :math:`n` MBIR iterations on the
same corrected counts before the next scatter correction. An MBIR iteration is cheap next to a scatter stage, so for a
given number of outer iterations (scatter steps) the extra MBIR iterations cost little and let MBIR converge further
towards each correction; the recursion of the accumulators is unchanged, it is advanced once per outer iteration.

At a fixed total number of MBIR iterations the trade-off reverses: a larger :math:`n` leaves fewer scatter stages, and
between two of them MBIR converges towards a correction that is no longer refreshed.

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

Short scans
~~~~~~~~~~~

The views are evenly spaced over :math:`[\beta_s, \beta_e)`; the synthetic data default to a short scan,
:math:`\beta_e = \beta_s + \pi + 2\gamma_m` with :math:`2\gamma_m = 2\arctan(N_c \Delta_c / (2\,\mathrm{SDD}))` the
in-plane fan angle of the panel. MBIRTORCH's FDK assumes a full rotation, so before it the line integrals of a short
scan of range :math:`\Lambda` are weighted with :math:`(\Lambda/\pi)\, w(\beta, \gamma)`, the Parker weights in the
overscan form of Wesarg et al., which make the two measurements of every in-plane line sum to one. MBIR itself needs no
weighting.

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
range gets no weights and a warning (the data are incomplete). They act per channel, the in-plane Parker weighting,
approximate off the central plane like FDK itself.

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
