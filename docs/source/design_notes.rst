Design notes
============

Implementation choices, accuracy and cost of each component. Timings are for an NVIDIA RTX 3070 (float32, fused Triton
kernels, PyTorch allocator capped at 4400-5000 MiB) on the WLAB phantom (a 30 x 20 x 40 cm water prism with a
6 x 6 cm lung, 2 x 2 cm air and 2 x 2 cm bone insert), default solver configuration unless stated.

Cross sections
--------------

**Data.** EPDL97 via XrayPhysics: elemental total, photoelectric, Compton and Rayleigh cross sections and the
incoherent / coherent angular distributions. Element mixing is done in ``cross_section_library``:
:math:`d\mu_x/d\Omega(E,\eta) = \sum_i w_i \sigma_{x}(Z_i, E)\, \mathrm{pdf}_{x,i}(E, \eta)` with each elemental
angular shape normalized numerically (XrayPhysics' own normalization agrees to 4.5e-4). XrayPhysics' coherent
amplitude is wrong at exactly :math:`\theta = 0`; angles are clamped at :math:`10^{-3}` degrees.

**Spectrum and groups.** The default source is a W-anode 125 kV spectrum (11 degree take-off) with a 0.89 mm Ti
filter on a 1 keV grid (``default_spectrum_125kvp``); any two-column spectrum can be loaded with
``load_spectrum_txt``. Spectra are piecewise-linear densities integrated exactly. Six equal-fluence groups over
[10, 125] keV have edges 125, 85.0, 69.2, 59.6, 51.7, 42.4, 10 keV and spectrum-weighted group energies 98.0, 76.7,
64.4, 56.1, 47.1, 36.0 keV.

**Materials.** Water (H 0.1119, O 0.8881 by mass) and ICRP cortical bone (XrayPhysics ``bone_cortical``,
1.85 g/cm^3). Narrow-group mass attenuation of water agrees with NIST XCOM to 0.1 % at 40-100 keV; the ICRP bone is
1.45 % below ICRU-44 bone at 60 keV (different composition).

**Energy quadrature.** Each group is cut into bins of at most 0.5 keV; quantities are evaluated at bin midpoints and
weighted with the exact spectrum integral over the bin, normalized within each group (a group without spectrum falls
back to a flat weight with a warning).

**Compton kinematics.** For each fine energy the scattering-angle boundaries of every lower group edge are computed
analytically (:math:`\eta_{edge} = 1 - m_ec^2 (1/E_{edge} - 1/E)`) and merged with fixed breakpoints dense near the
forward direction; each piece is integrated with 32 Gauss-Legendre nodes in :math:`q = \sin(\theta/2)`. The direct
table :math:`\mu_s(\eta)` uses the exact fraction of each fine bin that lands in the outgoing group at every table
:math:`\eta` (a midpoint indicator would make it a 0.5 keV staircase). Compton photons scattered below the lowest
edge (10 keV) are left in :math:`\mu_t` and treated as absorbed; the identity
:math:`\sum_{g'}\mu_{s,0}^{g\to g'} + \text{below cutoff} = \langle\sigma_{CS} + \sigma_{RS}\rangle_g` holds to 1e-8.

**Legendre truncation.** Off-diagonal (down-scatter) blocks are reproduced to under 5 % (relative L2 over
:math:`\eta`) at L = 7. Within-group blocks are not (24-34 % at L = 7, still 18-31 % at L = 9): they contain the Rayleigh
forward peak (width 0.01-0.05 in :math:`\eta`) and the kinematic edge of the within-group Compton ramp (4-7 % alone).
This is why the first-collision source uses the direct table; the truncated expansion of the multiple-collision
source still conserves :math:`\mu_{s,0}` and the mean cosine exactly.

**Fine-energy attenuation.** The library also stores the total attenuation of both materials at every fine energy
and the spectrum weight of every bin (``fine_energy``, ``fine_mu_t``, ``fine_weight``, ``fine_group``). Since the
optical depth at any energy is :math:`P_w m_w(E) + P_b m_b(E)`, two ray traces of the density maps give the exact
polychromatic transmission (``group_transmission``). Group-averaged attenuation underestimates the WLAB primary by
1.4 % (within-group beam hardening through 20 cm of water); the fine-energy primary agrees with a 0.25 keV reference
to 1e-6 (panel sum).

**Cache.** ``build_library`` caches the library as ``mglib_<hash>.pt`` in ``$CITRUS_CACHE_DIR`` (default
``~/.cache/citrus``); the key covers the configuration, the spectrum, the backend and the numerical parameters.
A build takes about 10 s.

Angular quadratures
-------------------

Real spherical harmonics are evaluated by stable normalized recurrences in :math:`z` and :math:`(x + iy)^k` (no
trigonometric functions, no division by :math:`\sin\theta`). Relation to the complex harmonics of SciPy (which carry
the Condon-Shortley phase): :math:`Y_{l,+k} = \sqrt2 (-1)^k \mathrm{Re}\,Y_l^k`, :math:`Y_{l,-k} = \sqrt2 (-1)^k
\mathrm{Im}\,Y_l^k`. The scattering-source operator accumulates every non-zero (g_in, g_out) block in place into a
moment-major buffer, so the only temporary is one [(L+1)^2, cells] gather (peak 1.3 GB instead of 6.5 GB on a
40 x 60 x 80 grid with G = 6, L = 7).

Transport sweep and source iteration
------------------------------------

**Layout.** Directions are grouped by octant; a chunk of :math:`D_c` directions per octant (8 :math:`D_c` in total) is
swept at once. The angular fluence of the chunk lives in one tensor ``store [8, Dc, cells + 1, 4G]`` whose last cell is
a zero "vacuum cell": upstream indices of boundary faces point to it, so a plain gather returns zero inflow. The
source of the chunk is written cell-major into the same layout (one call of the source function per chunk, optionally
into memory lent by the sweep). After the chunk, the moments are accumulated with one GEMM straight from the storage.

**Cost** (one sweep, G = 6, L = 7, 128 directions, 20 x 30 x 40 cells): 86 ms with the PyTorch plane step, 9 ms with
the fused Triton plane kernel (one launch per diagonal plane); one source iteration (scattering source + sweep) 16 ms.
The fused kernel follows ``ld_cell`` term by term and agrees with the PyTorch path to 1e-6 relative.

**Convergence.** WLAB at 1 cm converges in 16 iterations to ``si_tol = 1e-3`` (asymptotic ratio ~0.64 per iteration).

**First-collision sampling correction.** At the default 8 x 16 sweep quadrature, point samples of the forward-peaked
first-collision source miss 1.4 % of its :math:`l = 0` and 11.7 % of its :math:`l = 1` moment on WLAB. Matching both
moments per voxel and group (``fc_sampling_correction``) removes about 73 % of the sweep-quadrature sensitivity of the
multiple-collision term and raises the total scatter by 1.7 % (the deficit of point sampling); 8 x 16 with the
correction is about as accurate as point-sampled 12 x 24 at half the cost. The quadrature sums are formed by one
float32 matrix product per batch of directions and accumulated in float64 (factors within 1e-5 of an all-float64
computation); the correction pass costs 8 ms per view at 1 cm and 23 ms at 0.5 cm (factor 2).

Scatter image
-------------

**Sub-voxel points.** All scatter orders are transported on the supervoxel grid; the first-collision source and the
projection use the fine voxel centers inside each supervoxel, with the downstream attenuation traced on the
supervoxel grid plus its first-order variation inside the cell. Against a fine-grid first-order pass on WLAB: 0.26 %
RMS (max 0.86 %) for 0.5 cm voxels in 1 cm cells, 1.0 % (max 3.1 %) in 2 cm cells.

**Direction context.** Everything of an image that depends only on the view, the direction set and the detector / grid
responses is precomputed once per view (``make_direction_context``): the panel-facing directions and their chunking,
the detector response per direction, and the grid response per direction and detector row (exact for a flat panel
with a 1-D grid: the grid angles do not depend on the channel). The accumulation of a chunk is then one batched matrix
product. The context of a 2048-direction set on the 100 x 75 panel takes about 11 MiB instead of 370 MiB per weight
for per-pixel tables; the solver keeps an LRU of contexts per view, so repeated images of a view (every CITRUS outer
iteration) run without host synchronization.

**Direction set and footprint.** The cone product set is limited to the smallest cone about the panel normal that
contains every direction from the object box to the panel (51 degrees for the 100 x 75 panel at the AP view, 61-63
degrees for the full-field panel). Per-pixel scatter of a finite direction set carries ray effects where the emission
has sharp spatial structure (insert and object shadows along the beam). Against a 48 x 96 reference on the full-field
panel, the uniform 16 x 32 set has 5.5 % / 13 % RMS per-pixel error in the AP / lateral views and 32 x 64 has
1.6 % / 4.3 %; panel sums are exact to 0.3 % for both. The finite-cone footprint (one box blur per direction of width
(emission-weighted panel height) x (cell width), no free parameter) brings 16 x 32 to 1.8 % / 4.8 % at +20 % image
time (173 ms instead of 567 ms for 32 x 64 per scatter image at 0.5 cm). Splitting the emission into depth slabs adds
40-100 % cost for under 0.35 % points and is not used.

**Cost per view** (WLAB, AP view, 100 x 75 panel, 6 groups, L = 7): 0.39 s at 1 cm (source iteration 0.34 s, scatter
image 0.04 s), 0.60 s for 0.5 cm voxels in 1 cm cells (0.45 s / 0.14 s); peak memory about 0.8 GB. The no-grid scatter
fraction along the central row spans 0.29 behind the lung to 0.62 behind the bone; a focused 10:1 grid roughly halves
it.

Ray tracing and kernels
-----------------------

``SiddonProjectors`` computes exact line integrals (Siddon / Amanatides-Woo) for the four operations of the solver.
The PyTorch path batches rays with a fixed number of plane crossings per ray and serves as the reference; on float32
CUDA the fused Triton kernels trace every ray in registers (all groups at once). The downstream attenuation uses that
all rays of one direction start at voxel centers and are parallel, so they share the crossing sequence. The kernels
agree with the PyTorch path to 1e-6 relative (results are bitwise reproducible: no atomics) and are 20-200x faster
(downstream + oblique projection of a chunk of 8 directions on the 0.5 cm grid: 0.23 ms instead of 24 ms).

The fused first-collision source kernel evaluates the interpolated transfer table, the density mixing and the group
sum in one pass without temporaries (19x faster than the PyTorch path for a chunk of 8 directions on 24,000 voxels,
4 MiB instead of 140 MiB of temporaries). Both float32 paths deviate from a float64 evaluation by up to 7e-5 of the
field maximum where the Rayleigh peak of the table is steep (one float32 ulp of :math:`\eta` moves the interpolation
point).

Every kernel module runs a tiny self-check against its PyTorch path once per device and falls back to PyTorch on any
failure; ``CITRUS_DISABLE_TRITON=1`` disables all kernels.

MBIRTORCH coupling
------------------

**Geometry.** The solver uses MBIRTORCH's frames verbatim, so ``recon.mbir_bridge.make_cone_model`` is a parameter
copy (verified: MBIRTORCH's footprint projector and the pixel-averaged exact line integrals agree on WLAB;
a bright voxel lands on the same pixel). The region-of-reconstruction mask is off (the WLAB grid tiles the prism).

**One warm-started iteration.** ``model.recon(sino, weights=W, init_recon=X, max_iterations=k + 1,
first_iteration=k, stop_threshold_change_pct=0)`` runs exactly the k-th iteration of the default VCD schedule
(``mbir_iterations``): ``recon`` builds the partition sequence of length ``max_iterations`` and drops its first
``first_iteration`` entries, so ``max_iterations`` is the end position. The partitions and the subset order are drawn
from numpy's global RNG, which CITRUS seeds with ``[base_seed, k]`` before every iteration. The error sinogram and the
Hessian diagonal are recomputed from ``init_recon`` at every call; the regularization is frozen
(``auto_regularize_flag=False``) after one automatic estimate. A restarted run (one call per iteration) converges like
a continuous run.

**FDK.** ``recon_fdk``: cosine pre-weight, ramp filter along channels, the adjoint of the forward projector; full scan.

CITRUS cost and memory
----------------------

Per view and outer iteration (0.5 cm voxels in 1 cm cells, full-field 184 x 150 panel, 16 x 32 cone directions):
about 0.29 s, of which the single sweep with its first-collision source 0.10 s (the first-collision source of the
sweep is rebuilt every step because the densities change: sampling correction 21 ms, source at the sweep directions
16 ms, restriction to LD moments 34 ms) and the scatter image 0.15-0.17 s (first-collision source at the cone
directions 51 ms, sub-point emission 78 ms, downstream attenuation 18 ms, projection 22 ms, accumulation 9 ms).
A converged solve of the same view costs about 0.55 s. At 1 cm (factor 1) a CITRUS step costs about 0.13 s.

The per-view accumulator is :math:`G (L+1)^2 \cdot 4` values per cell: 141 MiB in float32 on the 0.5 cm WLAB grid
with 1 cm cells (24.7 GiB for 180 views). It is stored compactly (LD cell mean, scaled float16: 17.6 MiB per view,
3.1 GiB for 180 views) on the GPU when the total fits a 2600 MiB budget, else in pinned host memory (about 9 ms of
transfer per view). The stored, aged accumulator re-enters only through the scattering source of the next sweep, to
which its LD slopes contribute little; the final reconstruction differs from exact float32 storage by 6.5e-5
(relative), three orders of magnitude below the reconstruction error.
