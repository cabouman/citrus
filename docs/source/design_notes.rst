Design notes
============

The main implementation choices of CITRUS and the reason for each; :doc:`method` has the equations.

Cross sections
--------------

**Data and mixing.** The photon interaction data are EPDL97 via XrayPhysics: elemental total, photoelectric, Compton
and Rayleigh cross sections and the incoherent / coherent angular distributions. Elements are mixed in
``cross_section_library``, :math:`d\mu_x/d\Omega(E,\eta) = \sum_i w_i \sigma_{x}(Z_i, E)\, \mathrm{pdf}_{x,i}(E, \eta)`,
with each elemental angular shape normalized numerically, so that every material of the catalogue is built the same
way. XrayPhysics' coherent amplitude is wrong at exactly :math:`\theta = 0`; angles are clamped at :math:`10^{-3}`
degrees.

**Spectrum, groups and energy quadrature.** The default source is a W-anode 125 kV spectrum (11 degree take-off, 0.89
mm Ti filter, 1 keV grid, ``default_spectrum_125kvp``); any two-column spectrum can be loaded with
``load_spectrum_txt``. Spectra are piecewise-linear densities integrated exactly, so characteristic lines are kept. The
group edges cut the estimated total fluence spectrum (source plus scattered photons) into equal areas. Each group is cut
into bins of at most 0.5 keV, evaluated at the bin midpoints and weighted with the exact spectrum integral over the
bin.

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

**Fine-energy attenuation.** The library also stores the total attenuation of every material at every fine energy
(``fine_energy``, ``fine_mu_t``, ``fine_weight``, ``fine_group``). Since the optical depth at any energy is
:math:`\sum_m P_m m_m(E)`, one ray trace per material gives the exact polychromatic transmission
(``group_transmission``), without the within-group beam hardening of group-averaged attenuation.

**Cache.** ``build_library`` caches the library as ``mglib_<hash>.pt`` in ``$CITRUS_CACHE_DIR`` (default
``~/.cache/citrus``); the key covers the configuration, the spectrum, the backend, the numerical parameters and the
sorted material set, so XrayPhysics is needed only once per configuration.

Angular quadratures
-------------------

Real spherical harmonics are evaluated by stable normalized recurrences in :math:`z` and :math:`(x + iy)^k` (no
trigonometric functions, no division by :math:`\sin\theta`). Relation to the complex harmonics of SciPy (which carry
the Condon-Shortley phase): :math:`Y_{l,+k} = \sqrt2 (-1)^k \mathrm{Re}\,Y_l^k`, :math:`Y_{l,-k} = \sqrt2 (-1)^k
\mathrm{Im}\,Y_l^k`. The scattering-source operator accumulates every non-zero (g_in, g_out) block in place into a
moment-major buffer, so its only temporary is one gather of the transfer instead of a broadcast over all group pairs.

Transport sweep and source iteration
------------------------------------

**Layout.** Directions are grouped by octant; a chunk of :math:`D_c` directions per octant is swept at once. The
angular fluence of the chunk lives in one tensor ``store [8, Dc, cells + 1, 4G]`` whose last cell is a zero "vacuum
cell": upstream indices of boundary faces point to it, so a plain gather returns zero inflow. The source is written
cell-major into the same layout, and after the chunk the moments are accumulated with one matrix product straight from
the storage, so the angular fluence of all directions is never held at once.

**Fused plane kernel.** On float32 CUDA one Triton kernel launch solves a whole diagonal plane, in place of the many
small PyTorch operations of the reference plane step. It follows ``linear_discontinuous_cell`` term by term, so both
paths agree to float32 rounding.

**First-collision sampling correction.** Point samples of the forward-peaked first-collision source at the sweep
directions miss part of its :math:`l = 0` and :math:`l = 1` moments. Matching both moments per quadrature point and
group (``first_collision_sampling_correction``) lets the default sweep quadrature stand in for a finer point-sampled
one. The quadrature sums are formed by float32 matrix products per batch of directions and accumulated in float64.

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

**Direction set and footprint.** The cone product set is limited to the smallest cone about the panel normal that
contains every direction from the transport-grid box to the panel. Per-pixel scatter of a finite direction set carries
ray effects where the emission has sharp structure (insert and object shadows along the beam). The finite-cone
footprint, one box blur per direction with no free parameter, removes most of them at a small cost; splitting the
emission into depth slabs gains little for its cost and is not used.

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

Every kernel module runs a tiny self-check against its PyTorch path once per device and falls back to PyTorch on any
failure; ``CITRUS_DISABLE_TRITON=1`` disables all kernels.

MBIRTORCH coupling
------------------

**Geometry.** The solver uses MBIRTORCH's frames verbatim, so ``recon.mbir_bridge.make_cone_model`` is a parameter
copy. The region-of-reconstruction mask is off, because the WLAB grid tiles the prism and the mask would drop its
corners.

**One warm-started iteration.** ``model.recon(sino, weights=W, init_recon=X, max_iterations=k + 1,
first_iteration=k, stop_threshold_change_pct=0)`` runs exactly the k-th iteration of the default VCD schedule
(``mbir_iterations``): ``recon`` builds the partition sequence of length ``max_iterations`` and drops its first
``first_iteration`` entries, so ``max_iterations`` is the end position. The partitions and the subset order are drawn
from numpy's global RNG, which CITRUS seeds with ``[base_seed, k]`` before every iteration. The regularization is
frozen (``auto_regularize_flag=False``) after one automatic estimate, so that the correction, not a drifting prior,
changes between outer iterations.

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

Memory
------

**Densities one material at a time.** The density volume of all M reference materials on the reconstruction grid
would be M reconstruction-sized volumes. The loop instead builds the solver's fields from X one material map at a time
(``ScatterSolver.material_fields_from_attenuation``, ``DensityModel.material`` slab by slab) and the primary projects
the maps one at a time again, so the fine-grid footprint stays at about one map above X.

**Accumulators.** The per-view accumulator holds :math:`G (L+1)^2 \cdot 4` values per transport cell, which over all
views would dominate the memory. It is stored compactly (the linear-discontinuous cell mean in float16 with one scale
per group and moment, 1/8 of float32), on the GPU when the total fits ``gpu_state_budget_mib``, else in pinned host
memory. The stored, aged accumulator re-enters only through the scattering source of the next sweep, to which its
slopes contribute little.

**Allocator cap.** On Windows, VRAM oversubscription silently spills into system memory; the demos therefore cap the
PyTorch allocator (``citrus.utilities.set_memory_cap``).
