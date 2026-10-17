Results
=======

Convergence of CITRUS on synthetic WLAB data: 180 views over 360 degrees, 0.5 cm grid (40 x 60 x 80 voxels, 1 cm
transport cells), full-field panel (184 x 150 pixels of 0.4 cm), I0 = 1e5 counts per pixel with Poisson noise; primary
from MBIRTORCH's projector at the effective energy, scatter S = SPR x P with the solver's converged scatter-to-primary
ratio (median 1.2 behind the object, up to 25 along the long diagonals). Every reconstruction starts from the FDK of
the uncorrected data and runs one warm-started MBIR iteration per outer iteration with the same weights, partition
sequence and frozen regularization (estimated on the scatter-free sinogram). Relative RMSE over the prism after 20
outer iterations (the rows without a version note were measured with the two-material density model of version 0.0.3;
the reconstructions without the solver do not depend on it):

============================================================ ============== =========== ====================
reconstruction                                               relative RMSE  water bias  scatter s per outer
                                                                                         iteration
============================================================ ============== =========== ====================
FDK of the uncorrected data (initialization)                 0.317          \-          \-
MBIR, no scatter correction                                  0.294          -41 %       0
MBIR with the true SPR (Y / (1 + SPR_true))                  0.014          0.0 %       0
one converged scatter estimate from the FDK volume, frozen   0.165          -23 %       5 (once)
CITRUS, seven-material density model (version 0.1.0)         0.039          +0.6 %      63
CITRUS (one scatter step per view per outer iteration)       0.043          -1.3 %      50-60
converged scatter solve of every view in every outer         0.043          -1.2 %      99
iteration
============================================================ ============== =========== ====================

Findings:

1. CITRUS matches the reconstruction that recomputes a fully converged scatter estimate in every outer iteration, at
   about half its scatter cost. Its scatter estimate lags in the first outer iterations (it contains only the lowest
   collision orders) and catches up after about 8 iterations (scatter-estimate error 2.9 % vs 2.6 %); from iteration
   14 on the RMSE curves coincide. The relative change of the per-view accumulator falls from 1 to 2e-3 over 20
   outer iterations.
2. After 20 outer iterations the reconstruction, not the scatter estimate, limits the convergence: both coupled
   variants still improve by about 5 % per iteration, and MBIR with the true scatter-to-primary ratio reaches 0.014.
3. A single scatter estimate from the scatter-corrupted FDK volume removes only half of the cupping: the FDK densities
   are 40 % low, so the scatter is underestimated by about 23 %.
4. Without correction the water region is 41 % low (cupping) and the error stays at 0.29.
5. The density model matters for the bone: with the two-material split (unit water plus added bone above unit water)
   the reconstructed cortical-bone insert ends 6.3 % too dense (0.5704 vs 0.5366 /cm); with the seven-material
   interpolating model, which keeps a bone voxel pure bone, it ends 0.6 % high (0.5399) and the water 0.6 % high, at a
   relative RMSE of 0.0389 over the prism and 0.0451 in the core (two-material: 0.0429 / 0.0505), for 8 % more
   scatter time (62.6 s instead of 57.9 s per outer iteration in the same session).

The quick demo setting (``demo/demo_citrus.py --views 36 --delta-voxel 1 --outer 10 --scatter-view-stride 1
--scatter-every 1``: 36 views over a short scan of 202.6 degrees, 1 cm grid, 10 outer iterations) shows the same
ordering in about two minutes: relative RMSE 0.31 (FDK, Parker-weighted), 0.29 (no correction), 0.064 (CITRUS); over
a full rotation (``--end-view-angle 360``) 0.32, 0.29 and 0.075 (measured with version 0.1.0 and the data-set format
1 of citrus <= 0.2.0, SPR with the Siddon primary).
The demo's defaults (180 views over the short scan, 0.5 cm grid, 80 outer iterations with the scatter step on every
8th view and at every 8th outer iteration, i.e. 10 scatter steps on 23 views each; data-set format 2, the SPR with the
MBIRTORCH primary of the loop's default) reach a relative RMSE of 0.0215 over the prism and 0.0259 in the core (FDK
0.318 / 0.348, no correction 0.292 / 0.355) in 159 s of CITRUS on the RTX 3070 (8.6 s per scatter stage, 0.85 s per
MBIR iteration, peak 1.64 GB allocated); synthesizing the data set takes 113 s once. With ``--primary-projector
siddon`` on the same data: 0.0257 / 0.0305. On the format-1 data of version 0.2.0 (SPR with the Siddon primary) the
Siddon loop gave 0.0242 / 0.0294 and the MBIRTORCH loop 0.0270 / 0.0305: the primary model that matches the data's
wins (:doc:`method`, "CITRUS coupling"). Resuming MBIR from its error sinogram (the default) changes the
reconstruction by 1.1e-6 (relative).

Accuracy of the scatter model itself (inverse crime): the synthetic data and the reconstruction loop use the same
solver, so the remaining discretization error of the solver (per-pixel ray effects of a finite collapsed-cone set,
Legendre truncation of the multiple-collision source) is not visible in this study; see :doc:`design_notes`.
