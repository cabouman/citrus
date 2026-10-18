Design notes
============

The main implementation choices of CITRUS and the reason for each; :doc:`method` has the equations.

Cross sections
--------------

**Data and mixing.** The integrated cross sections (coherent, incoherent, photoelectric, pair production) are NIST
XCOM's, interpolated as XCOM's own program does it (log-log cubic splines, separate tables below the K edge, the value
above an edge at the edge); the angular distributions are xraylib's (Klein-Nishina times the incoherent scattering
function, Thomson times the squared form factor) (``photon_data``). Neither needs a compiled dependency: the XCOM
tables are a 173 kB download from NIST, xraylib installs from wheels. XCOM covers 1 keV to 100 GeV including pair
production, and its totals are the values of NIST's attenuation tables. Elements are mixed in
``cross_section_library``, :math:`d\mu_x/d\Omega(E,\eta) = \sum_i w_i \sigma_{x}(Z_i, E)\, \mathrm{pdf}_{x,i}(E, \eta)`,
with each elemental angular shape normalized numerically, so that the scattering of every material integrates to the
XCOM cross section whatever the integral of the shape, and every material of the catalogue is built the same way.

**Spectrum, groups and energy quadrature.** The user supplies the source spectrum, (energies in keV, relative photon
density per keV); there is no default, and CITRUS does not simulate tube spectra. A two-column text file is loaded with
``load_spectrum_txt``; the repository's ``demo/spectra`` holds two samples, a W-anode 125 kV spectrum (11 degree
take-off, 0.89 mm Ti filter, 1 keV grid) and the Mohan et al. (1985) 6 MV linac spectrum (1 keV nodes to 6 MeV, the
spectrum of the 6 MV validation). Spectra are piecewise-linear densities integrated exactly, so characteristic lines
are kept. A spectrum of one node (``monoenergetic_spectrum(energy_kev)``, or a one-line file) is a monoenergetic
line: all source photons sit in the fine bin that holds the line, which is evaluated at the line's energy, so the
effective energy and the primary attenuation are exactly those of the line; the groups below it hold the scattered
photons ("scattered_fluence" or "log" edges; "equal_fluence" needs a continuous spectrum). The
group edges cut the estimated total fluence spectrum (source plus scattered photons) into equal areas. Each group is cut
into bins of at most 0.5 keV, evaluated at the bin midpoints and weighted with the exact spectrum integral over the
bin.

The 125 kVp spectrum (``demo/spectra/spectrum_125kvp.txt``, the spectrum of the WLAB validation, 125 nodes at 1 to
125 keV) was computed once with XrayPhysics, ``simulateSpectra`` filtered by
``filterResponse('Ti', 4.506, 0.089)``. The 6 MV spectrum (``demo/spectra/spectrum_6mv.txt``) is the EGSnrc
``mohan6`` histogram (24 bins of 0.25 MeV) written at 1 keV nodes, uniform inside each bin.

**Group edges from the scattered fluence.** Equal source photon counts per group resolve the spectrum the source emits,
but the groups also carry the collided photons, and the multiple-collision scatter consists of downscattered photons.
At kV both spectra cover the same range; at MV the source has few photons in the hundreds of keV where most multiply
scattered photons are, so equal source counts leave a single wide group there, whose group-averaged attenuation is too
coarse. The edges therefore follow an estimate of the total fluence, which the groups have to resolve. A transport
solution is not needed for a grouping: a slowing-down balance in energy alone, with Klein-Nishina downscatter, is
cheap and places the edges where the photons are. Escape matters as much as absorption in an object of patient size,
so the balance has a leakage term set by the object's mean chord, and its medium is the object's mean composition;
the CITRUS loop measures both on the FDK volume and builds the library only after that (a lazy library in
``ScatterSolver``), once per run and cached per size and medium. A known medium (an industrial part of one material)
can be named instead. The same estimate weights the energies within each group: a group constant is an average over
the photons the group carries, and in the lowest groups at MV these are scattered photons near the top of the group,
not the flat or empty source spectrum there, whose weighting would give the group the photoelectric attenuation of its
lowest energies and absorb the multiply scattered photons too early.

The balance is discretized on equal bins of width max(``fine_bin_width_kev``, (kVp - ``energy_min_kev``) / 2000)
from ``energy_min_kev`` to kVp, with the source photons of each bin integrated exactly from the piecewise-linear
spectrum and the Klein-Nishina probability of landing in each bin integrated in closed form for a photon at the
centre of the source bin (scattering within the bin is kept), and solved by one sweep from the highest bin down.

**Compton kinematics.** For each fine energy the scattering-angle boundaries of every lower group edge are computed
analytically and each piece is integrated with Gauss-Legendre nodes in :math:`q = \sin(\theta/2)`, which resolves the
forward structure of the form factors. The direct table :math:`\mu_s(\eta)` uses the exact fraction of each fine bin
that lands in the outgoing group, so it is continuous in :math:`\eta` instead of a staircase. Compton photons scattered
below the lowest group edge stay in :math:`\mu_t` and are treated as absorbed.

**Direct table for the first collision.** The within-group transfer blocks contain the Rayleigh forward peak and the
kinematic edge of the within-group Compton ramp, which a Legendre expansion of practical order does not reproduce. The
first-collision source therefore uses the directly tabulated transfer; the truncated expansion serves only the
multiple-collision source, where it conserves :math:`\mu_{s,0}` and the mean cosine exactly.

**Adaptive angular nodes.** The forward peaks of the Rayleigh and Compton transfer narrow with energy. At kV a table
uniform in :math:`\eta` resolves them; at MV they are far narrower than its spacing, and linear interpolation spreads
an unresolved peak over a whole table cell, which overestimates the forward first-order scatter. More uniform nodes
fix this only at a table size that no longer stays in the GPU cache. The nodes are instead placed where the transfer
curves bend, by the density that equalizes the relative error of linear interpolation, with a floor that keeps every
region of :math:`\eta` populated. The placement is a monotone map stored as a small table (512 values) at equally
spaced :math:`q = \sin(\theta/2)`, the variable in which the peaks are smooth and which spreads the forward
direction; a lookup stays index arithmetic (one extra lookup in a table that stays in the L1 cache), and the table
keeps its size. The curves of the catalogue's water and cortical bone place the nodes for every library, so a
material's rows do not depend on the material set. The uniform table remains an option.

The map (``cross_section_library.angular_map``) is built in four steps. Every transfer curve :math:`f_{m,g,g'}(q)`
of the map materials (Compton with the kinematic fraction of every fine bin, plus Rayleigh for :math:`g' = g`) is
evaluated on 20001 points uniform in :math:`q`, from cubic splines through the reference angular quadrature of the
build. The node density is the largest over the curves (those above 1e-6 of the largest curve of the same material
and source group) of :math:`\sqrt{|f''(q)| / \max f}`, smoothed over two knot intervals so that neighbouring node
spacings change gradually. The normalized density is mixed with a floor, :math:`(1 - c)\,\rho / \int\rho + c\,2q`
with :math:`c = 0.1`, so that every region keeps at least a tenth of the node density of a table uniform in
:math:`\eta` (:math:`2q` is that density in :math:`q`) and the map is strictly monotone. Finally :math:`h` is one
minus the cumulative integral of the density from :math:`q = 0` over its total, sampled at the 512 knots and linear
in :math:`q` between them.

**Fine-energy attenuation.** The library also stores the total attenuation of every material at every fine energy
(``fine_energy``, ``fine_mu_t``, ``fine_weight``, ``fine_group``). Since the optical depth at any energy is
:math:`\sum_m P_m m_m(E)`, one ray trace per material gives the exact polychromatic transmission
(``group_transmission``), without the within-group beam hardening of group-averaged attenuation.

``group_transmission`` evaluates the sum in blocks of at most :math:`2^{22}` (fine energy, ray) pairs: per block
one optical-depth tensor (M fused multiply-adds), an in-place exponential and one matrix product with the
fine-to-group weights, accumulated into the result. The float32 matrix product assumes that TF32 is off
(``torch.backends.cuda.matmul.allow_tf32 = False``, the PyTorch default).

**Cache.** ``build_library`` caches the library as ``mglib_<hash>.pt`` in ``$CITRUS_CACHE_DIR`` (default
``~/.cache/citrus``); the key covers the configuration, the spectrum, the backend, the numerical parameters and the
sorted material set, so the photon data are read only once per configuration.

Photon data
-----------

``photon_data`` provides the point-wise data of the library builder, per element Z and photon energy.

**XCOM tables.** The integrated cross sections come from the offline distribution of NIST XCOM 3.1: one table per
element Z = 1..100 (files ``MDATX3.001`` to ``MDATX3.100``), from 1 keV to 100 GeV, in barns/atom, for coherent
scattering (relativistic form factors), incoherent scattering (Klein-Nishina times the incoherent scattering
function), photoelectric absorption (Scofield) and pair production in the nuclear and in the electron field. A file
holds whitespace-separated records: Z and the atomic weight; the number of edges :math:`n_e` and the grid size
:math:`n`; when :math:`n_e > 0` the 1-based grid indices of the edges, their labels and their energies (eV), each
list from the K edge down; the :math:`n` grid energies (eV, ascending); the :math:`n` values of each of the five
cross sections; when :math:`n_e > 0` the number of photoelectric tables below the K edge, their lengths, then all
their energies (MeV) and all their cross sections (barns/atom), from the lowest interval up. Some files end with a
DOS end-of-file mark (Ctrl-Z).

**XCOM interpolation.** CITRUS reproduces the interpolation of XCOM's Fortran program XCOM.f:

* the energy grid lists every absorption edge twice, just below the edge (the energy minus 0.1 eV) and at the edge;
  coherent and incoherent scattering are interpolated by a natural cubic spline in :math:`(\ln E, \ln\sigma)` over
  the grid without the points just below the edges (XCOM.f subroutines SCOF and BSPOL);
* photoelectric absorption above the K edge uses the same spline over the points from the K edge up; below the K
  edge every interval between two edges has its own short table, interpolated linearly in
  :math:`(\ln E, \ln\sigma)` (XCOM.f subroutine BLIN). At an edge energy the value above the edge is taken, so the
  photoelectric cross section jumps at every edge. Elements without edges above 1 keV (Z <= 10) use one spline over
  the whole grid;
* pair production is zero up to its threshold (1.022007 MeV in the nuclear field, 2.044014 MeV in the electron
  field); above it the spline interpolates :math:`\ln\big(\sigma / ((E - E_{th})/E)^3\big)` over the grid points
  from the threshold up, with the value at the threshold extrapolated from the three points above it by XCOM.f's
  formula;
* the mass cross section is :math:`\sigma \cdot 0.60221367 / A` (barns/atom to cm^2/g), with XCOM.f's Avogadro
  factor and the atomic weights :math:`A` of XCOM's table ``ATWTS.DAT``. The total is the sum of the five
  processes, XCOM's "total attenuation with coherent scattering".

**Angular shapes.** The Compton shape is xraylib's ``DCS_Compt``, the Klein-Nishina cross section times the
incoherent scattering function :math:`S(x, Z)`; the Rayleigh shape is xraylib's ``DCS_Rayl``, the Thomson cross
section times the squared atomic form factor :math:`F(x, Z)`; :math:`x = \sin(\theta/2)\,E / 12.3984193` (E in keV,
x per angstrom) is the momentum transfer. xraylib's tables of :math:`S` and :math:`F` start at :math:`x = 10^{-3}`
and :math:`10^{-7}` per angstrom and xraylib refuses smaller :math:`x` (the most forward angles at MeV energies);
there :math:`S` is continued as :math:`S(x_{min})\,(x / x_{min})^2` (it grows as :math:`x^2` from 0) and :math:`F`
is held at :math:`F(x_{min})`. The shapes are kept on xraylib's per-gram scale (xraylib's own atomic weight), since
the builder normalizes each shape to the XCOM cross section of the same process. They cover Z = 1..98.

**Atomic weights.** The atomic weights of XCOM's ``ATWTS.DAT`` (the standard atomic weights NIST uses) serve both
the barns-to-cm^2/g conversion and the mass fractions of chemical formulas. xraylib's ``AtomicWeight`` is not used
for either: it is rounded (H 1.01, O 16.0), which shifts the mass fractions of compounds, and differs from the
standard value for thulium.

Angular quadratures
-------------------

Real spherical harmonics are evaluated by stable normalized recurrences in :math:`z` and :math:`(x + iy)^k` (no
trigonometric functions, no division by :math:`\sin\theta`). Relation to the complex harmonics of SciPy (which carry
the Condon-Shortley phase): :math:`Y_{l,+k} = \sqrt2 (-1)^k \mathrm{Re}\,Y_l^k`, :math:`Y_{l,-k} = \sqrt2 (-1)^k
\mathrm{Im}\,Y_l^k`. The scattering-source operator accumulates every non-zero (g_in, g_out) block in place into a
moment-major buffer, so its only temporary is one gather of the transfer instead of a broadcast over all group pairs.

With :math:`Q_{lk} = N_{lk} P_l^k / \sin^k\theta`, a polynomial in :math:`z`, the harmonics are
:math:`Y_{l,+k} = \sqrt2\, Q_{lk}(z)\,\mathrm{Re}((x + iy)^k)`, :math:`Y_{l,-k} = \sqrt2\, Q_{lk}(z)\,
\mathrm{Im}((x + iy)^k)` and :math:`Y_{l0} = Q_{l0}(z)`, where :math:`Q_{00} = 1/\sqrt{4\pi}`,
:math:`Q_{kk} = \sqrt{(2k+1)/(2k)}\,Q_{k-1,k-1}`, :math:`Q_{k+1,k} = \sqrt{2k+3}\,z\,Q_{kk}` and
:math:`Q_{lk} = a_{lk}(z\,Q_{l-1,k} - b_{lk}\,Q_{l-2,k})` with :math:`a_{lk} = \sqrt{(4l^2-1)/(l^2-k^2)}` and
:math:`b_{lk} = \sqrt{((l-1)^2-k^2)/(4(l-1)^2-1)}`. The directions are assumed unit-norm and not renormalized.

Transport sweep and source iteration
------------------------------------

**Layout.** Directions are grouped by octant; a chunk of :math:`D_c` directions per octant is swept at once. The
angular fluence of the chunk lives in one tensor ``store [8, Dc, cells + 1, 4G]`` whose last cell is a zero "vacuum
cell": upstream indices of boundary faces point to it, so a plain gather returns zero inflow. The source is written
cell-major into the same layout, and after the chunk the moments are accumulated with one matrix product straight from
the storage, so the angular fluence of all directions is never held at once.

**Source function.** The sweep asks for the source of a chunk with one call ``source_function(direction_index)``
covering all 8 Dc directions of the chunk, octant-major (indices ``o * D8 + d``), so a source formed from moments by a
matrix product reads its moments once per chunk. A source function that accepts an ``out`` keyword is lent the
not-yet-used storage tensor of the chunk to write its ``[8 Dc, G, 4, rows, cols, slices]`` result into, so no
source-sized temporary is allocated; otherwise the storage is allocated after the fill, so the two never coexist.

**Memory of the source iteration.** The peak during one collision order is the running sum ``phi_total``, the
scattering-source moments T of the previous order and the sweep's moment accumulator (three fields of
:math:`G (L+1)^2 \cdot 4\,N` values for N transport cells) plus the sweep buffers (:math:`2 \cdot 8\,D_c\,N \cdot 4G`
values); the previous scattering source is released before the next one is formed.

**Fused plane kernel.** On float32 CUDA one Triton kernel launch solves a whole diagonal plane, in place of the many
small PyTorch operations of the reference plane step. It follows ``linear_discontinuous_cell`` term by term, so both
paths agree to float32 rounding.

The plane kernel runs one lane per (entry, direction, group) of the plane with the group fastest, so the lanes of a
warp read contiguous 4G-value rows of the cell-major storage. Entries of a plane never read each other (their
upstream cells lie on earlier planes), so a plane needs no synchronization inside the kernel; the planes are ordered
by the CUDA stream, one launch per plane. The entries are read from a plane-sorted int32 copy of the sweep plan
(``FlatPlaneIndex``), addressed by the plane start and size, which are launch arguments without value
specialization, so there is one compiled variant per group count and chunk size.

**First-collision sampling correction.** Point samples of the forward-peaked first-collision source at the sweep
directions miss part of its :math:`l = 0` and :math:`l = 1` moments. Matching both moments per quadrature point and
group (``first_collision_sampling_correction``) lets the default sweep quadrature stand in for a finer point-sampled
one. The quadrature sums are formed by float32 matrix products per batch of directions and accumulated in float64.
The float32 products assume TF32 is off (``torch.backends.cuda.matmul.allow_tf32 = False``, the PyTorch default).

Scatter image
-------------

**Quadrature points.** All scatter orders are transported on the transport grid; the first-order quantities, which
carry the sharpest structure, are evaluated on a few quadrature points per cell (Acuros CTS, Maslowski et al. 2018,
Section 2.B). The cost of a scatter step then depends on the cell size, not on the reconstruction voxel size. The
points follow a target spacing rather than a fixed count, so larger cells keep the first-order resolution, and they are
never much denser than the data. Only the primary and the resampling of the densities touch the reconstruction grid.

**Uncollided trace.** The default traces the uncollided fluence through the quadrature grid. The coarse trace through
the transport cells smears a material boundary that lies inside a cell, which matters where rays run along such a
face; the fine trace through the reconstruction grid serves validation. All three cost about the same.

**Pair and dense first-collision source.** ``rho_quadrature`` is checked once per density field: where every point has
at most two nonzero materials the pair kernel gathers two table rows per point, otherwise the dense kernel evaluates
all materials. The pair form keeps the cost of a seven-material library close to that of a two-material one.

**Direction context.** Everything of an image that depends only on the view, the direction set and the detector / grid
responses is precomputed once per view (``make_direction_context``): the panel-facing directions and their chunking,
the detector response per direction and the grid response per direction and detector row (exact for a flat panel with
a 1-D grid, whose angles do not depend on the channel). The accumulation of a chunk is then one batched matrix product,
and an LRU of contexts lets the repeated images of a view (every CITRUS outer iteration) run without host
synchronization.

The detector response depends on a direction only through :math:`\theta_0` and is stored as ``[K, G]``; the grid
response of a flat panel with a 1-D grid is stored as ``[K, G, Rd]``, and the three accumulation weights (fluence,
planar fluence, signal) as one ``[G Rd, 3, K]`` matrix, a factor ``Cd / 3`` smaller than per-pixel tables.
Separability is checked numerically on probe directions when the context is built; a panel that fails it falls
back to per-pixel weights. The responses are evaluated in float64 and stored in the image dtype. The panel-facing
directions are selected in the image dtype, so a direction grazing the panel plane is kept or dropped exactly as
the projectors see it, and they are stored as one contiguous tensor, so the projectors copy them to the host once
per context. The context does not depend on the voxel grid; the tables that do (sub-cell offsets, footprint
heights) and the spherical harmonics per order are cached in it on first use.

**Direction set and footprint.** The cone product set is limited to the smallest cone about the panel normal that
contains every direction from the transport-grid box to the panel. Per-pixel scatter of a finite direction set carries
ray effects where the emission has sharp structure (insert and object shadows along the beam). The finite-cone
footprint, one box blur per direction with no free parameter, removes most of them at a small cost; splitting the
emission into depth slabs gains little for its cost and is not used.

The covering cone is exact: the polar angle :math:`\arctan(|d_\perp| / d_n)` of :math:`d = p - r` has convex cones
as sublevel sets, so its maximum over the differences of a panel point and a box point is attained at a vertex of
that polytope, a pair of a panel corner and a box corner. If a box corner is not strictly on the source side of the
panel plane the cone is the full hemisphere.

Ray tracing and kernels
-----------------------

``SiddonProjectors`` computes exact line integrals (Siddon / Amanatides-Woo) for the five operations of the solver.
The PyTorch path batches rays with a fixed number of plane crossings per ray and serves as the reference; on float32
CUDA the fused Triton kernels trace every ray in registers, all groups at once. The downstream attenuation uses that
all rays of one direction start at voxel centers and are parallel, so they share the crossing sequence. The kernels
use no atomics, so their results are bitwise reproducible.

The fused first-collision source kernel evaluates the interpolated transfer table, the density mixing and the group
sum in one pass without temporaries. Where the Rayleigh peak of the table is steep, one float32 ulp of :math:`\eta`
moves the interpolation point, so both float32 paths deviate slightly from a float64 evaluation there.

**Siddon traversal.** A segment is parametrized as :math:`\mathbf p(a) = \mathbf p_0 + a(\mathbf p_1 - \mathbf p_0)`,
:math:`a \in [0, 1]`, and clipped to the grid box by the slab method, giving :math:`[a_{lo}, a_{hi}]`. The parameters
of its crossings with the interior voxel planes of each axis are clamped to that interval, merged with its ends and
sorted; consecutive parameters bound sub-segments that each lie in one voxel, found from the sub-segment midpoint,
and the integral is the sum of voxel value times length. Zero-length sub-segments (clamped or coincident crossings,
e.g. through voxel edges and corners) contribute nothing, so the result is exact to rounding for any segment,
including axis-aligned ones and ones that start or end inside the box. The PyTorch path holds
:math:`n_x + n_y + n_z - 1` parameters per ray and works in chunks of at most ``ray_chunk`` rays and
``element_budget`` (ray, plane) pairs; its peak working memory is about ``element_budget`` times (G + 12) scalars.
The kernels need no chunking.

**Fused ray kernel.** Each lane of ``_ray_kernel`` clips its ray like the reference, takes the entry voxel as the
floor of the entry point (ceil - 1 along axes the ray descends, so a ray starting on a plane enters the voxel it
heads into) and then steps to the nearest of the next x, y and z plane crossings (ties in the order x, y, z: the tied
axis follows with a zero-length segment, as the coincident sorted parameters of the reference). The plane parameters
are recomputed from the voxel index at every step with the reference's arithmetic, so no error accumulates along the
ray. The loop runs to the largest per-ray bound of the block, :math:`1 + \sum_a (\lfloor L_a / \Delta_a \rfloor + 1)
+ 1` steps, with finished lanes masked. Each ray integral equals the PyTorch path up to float32 rounding of the
crossing parameters and the summation order.

**Shared downstream crossings.** All rays of one direction start at voxel centers and are parallel, so the distance
to the first plane of each axis is half a voxel for every ray, and the crossings
:math:`t_m = (m + \tfrac12)\Delta_a / |\omega_a|` are the same for all rays of the direction. A program of
``_downstream_shared_kernel`` (one direction, a block of consecutive voxels) runs that scalar sequence once: the axis
choice is uniform across the block, each ray only shifts its voxel index by the shared step, and the loads of a step
are a contiguous strip of the volume. A ray stops at its box exit (per-ray index limits) or at the panel plane
(:math:`t_{end}` = panel distance clamped to [0, box diagonal], the end point of the reference), so the result is
exact for any direction. Directions with :math:`\omega \cdot (-\hat n) \le 0` give +inf; that decision is passed in
as flags computed with the reference's own PyTorch expression, so it agrees bit for bit for grazing directions.

**Kernel transfer table.** The first-collision source kernel reads the transfer table re-laid out as
``[N, G_in, M, GP]`` float32 (GP the next power of two of G), zero where ``g_out < g_in`` (no upscatter) and for the
padding ``g_out >= G``, so all entries a point needs lie in two contiguous rows, one per eta node; the table stays in
the L2 cache and neighbouring points hit neighbouring rows. The re-laid table is cached per library tensor, device
and tensor version and dropped when the library tensor is freed.

**Kernel launch.** The kernels are compiled without value or alignment specialization of their integer and pointer
arguments, so one binary fits every problem size; ``triton_utilities.launch`` caches it per (kernel, constexpr
values, warps, device) and calls it directly, skipping the argument handling of Triton's public launch path.

Every kernel module runs a tiny self-check against its PyTorch path once per device and falls back to PyTorch on any
failure; ``CITRUS_DISABLE_TRITON=1`` disables all kernels.

MBIRTORCH coupling
------------------

**Geometry.** The solver uses MBIRTORCH's frames verbatim, so ``recon.mbir_bridge.make_cone_model`` is a parameter
copy. The region-of-reconstruction mask is off, because the WLAB grid tiles the prism and the mask would drop its
corners.

**Units and devices.** MBIRTORCH projects in value times its arbitrary length unit; the bridge uses cm, so a volume
in 1/cm projects to dimensionless line integrals. MBIRTORCH accepts torch tensors on any device and places them as
float32 on the model device (no copy when they already are); the bridge pins every model to one device
(``configure_devices``, which also switches off MBIRTORCH's automatic multi-GPU layout) and returns torch tensors on
it (``output_sharded=True``), never numpy. ``torch.compile`` is used on CUDA only, since on the CPU it needs a C++
toolchain.

**Projections in chunks of views.** A ``ConeBeamModel`` projects onto all its views at once. To project onto a run
of consecutive views (the per-material path lengths of the loop's primary must not grow to ``[M', V, rows,
channels]``), ``ViewChunkProjector`` builds a sub-model over those angles with ``make_cone_model`` and keeps the
last few in a least-recently-used cache. A sub-model is cheap to build; its first projection compiles, later ones
cost the same per view as the full model, and its projections equal the matching views of the full model to float
rounding.

**One warm-started iteration.** ``model.recon(sino, weights=W, init_recon=X, max_iterations=k + 1,
first_iteration=k, stop_threshold_change_pct=0)`` runs exactly the k-th iteration of the default VCD schedule
(``mbir_iterations``): ``recon`` builds the partition sequence of length ``max_iterations`` and drops its first
``first_iteration`` entries, so ``max_iterations`` is the end position. The partitions and the subset order are drawn
from numpy's global RNG, which CITRUS seeds with ``[base_seed, k]`` before every iteration. The regularization is
frozen (``auto_regularize_flag=False``) after one automatic estimate, so that the correction, not a drifting prior,
changes between outer iterations.
Each call draws its partitions anew, so the granularity of iteration k is that of the default schedule, while the
random subsets differ from those of one continuous ``recon`` run.

**FDK.** ``recon_fdk`` assumes a full rotation (MBIRTORCH applies no short-scan weighting). For a short scan
``mbir_bridge.fdk(..., geometry=, angles=)`` multiplies the line integrals by (Lambda / pi) w(beta, gamma), the Parker
weights in the overscan form of Wesarg et al., with the fan angle gamma = -atan(u / SDD) in MBIRTORCH's frame.

**Primary of the loop.** The loop's primary uses MBIRTORCH's projector, the reconstruction's own operator: it averages
the line integral over the pixel like a detector pixel does, and the synthetic data define their scatter through the
same primary, so the correction is consistent with the data also on the pixels whose rays graze an object face.

**MBIR resume.** MBIRTORCH's public ``recon`` recomputes the error sinogram with one forward projection per call;
``mbir_iterations(..., error_sinogram=, return_checkpoint=True)`` replicates ``recon``'s preparation and calls its
engine with the kept error sinogram. Keeping it costs one sinogram of memory and saves a forward projection per outer
iteration. The forward-model Hessian could be kept as well, but it is a reconstruction-sized volume, so the loop
recomputes it.

The error sinogram must match the initial volume; it is trusted, not checked, and MBIRTORCH updates both it and the
initial volume in place, so the loop passes a copy of the volume. The engine path draws the same partitions and
partition sequence as ``recon`` for the same random state and uses the same regularization, so a resumed call
equals the plain one up to the float rounding of the carried error sinogram.

Memory
------

**Densities one material at a time.** The density volume of all M reference materials on the reconstruction grid
would be M reconstruction-sized volumes. The loop instead builds the solver's fields from X one material map at a time
(``ScatterSolver.material_fields_from_attenuation``, ``DensityModel.material`` slab by slab) and the primary projects
the maps one at a time again, so the fine-grid footprint stays at about one map above X.
The upstream line integrals of the uncollided fluence (``uncollided.uncollided_path_lengths``) are linear and channel
by channel, so for fields built from the attenuation the solver traces them one material map at a time as well and
concatenates the results.

One material's map is formed without the others (``utilities.attenuation_to_density``): with the material in slot
:math:`s` of the bracketing order and :math:`\mu` the nominal attenuations in that order, the map is
:math:`f\rho^{nom}` on :math:`\mu_{s-1} \le X < \mu_s`, :math:`(1 - f)\rho^{nom}` on :math:`\mu_s \le X < \mu_{s+1}`,
:math:`X / m` for the first material below :math:`\mu_0` and for the last from :math:`\mu_{M-1}` on, and 0
elsewhere. Every voxel value is formed by the same floating-point operations as in ``attenuation_to_densities``
(the scalars are 0-dim tensors in X's dtype, :math:`1 - f` is evaluated as :math:`-f + 1`), so the map equals the
corresponding row of the full model bit for bit; its temporaries are the result, one interpolation weight and two
boolean masks. The bracket tests use X itself: every nominal attenuation is positive, so :math:`X \ge \mu_k`
implies :math:`X > 0`.

**Accumulators.** The per-view accumulator holds :math:`G (L+1)^2 \cdot 4` values per transport cell, which over all
views would dominate the memory. It is stored compactly (the linear-discontinuous cell mean in float16 with one scale
per group and moment, 1/8 of float32), on the GPU when the total fits ``gpu_state_budget_mib``, else in pinned host
memory. The stored, aged accumulator re-enters only through the scattering source of the next sweep, to which its
slopes contribute little.

**Allocator cap.** On Windows, VRAM oversubscription silently spills into system memory; the study script
therefore caps the PyTorch allocator (``citrus.utilities.set_memory_cap``).
