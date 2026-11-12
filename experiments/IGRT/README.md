# CITRUS on the COBRA2026 IGRT cases

Reconstructions of Varian and Elekta pelvis CBCT scans of the COBRA2026 dataset (training split, Zenodo / CREATIS
warehouse) with CITRUS, compared with the vendor's clinical CBCT and the deformed planning CT.

| scripts | scanner | cases |
|---|---|---|
| `COBRA_TrueBeam_LowRes.py`, `COBRA_TrueBeam_FullRes.py` | TrueBeam, half fan, whole rotation (centre G) | G008, G007 |
| `COBRA_Halcyon_LowRes.py`, `COBRA_Halcyon_FullRes.py` | Halcyon, full fan, short scan (centre F) | F015, F003, F041 |
| `COBRA_Synergy_LowRes.py`, `COBRA_Synergy_FullRes.py` | Elekta Synergy (XVI), half fan, whole rotation (centre C) | C039, C041, C013 |

Each scanner has two scripts that differ only in their CITRUS settings. Both run the loop as multigrid stages
(`Schedule(stages=...)`): coarse voxels on binned data and a subset of the views first, each stage starting from the
previous stage's volume and scatter state.

- `LowRes`: binned data (TrueBeam 4 x 2 pixels and every 2nd view, Halcyon 2 x 6, Synergy 2 x 2) and 1.25 mm voxels
  (2 mm for the Halcyon), in three stages from 5 mm voxels; MBIRTORCH's default prior. It runs on a 12 GB GPU.
- `FullRes`: every view and pixel (Halcyon channels in pairs) on the clinical CBCT's voxels (about 1 mm), in four
  stages from 5 mm voxels; a sharper prior (sharpness 1.25, T 0.25). It needs a GPU with about 60 GB, such as an
  80 GB H100.

Each script has five sections:

1. the case (`CASE = ...`) and the parameters its metadata does not give: the panel's drift model, the couch height and
   attenuation, the spectrum, the grid's strips, and the CITRUS settings;
2. the data: the case is downloaded into `data/TrueBeam/<case>`, `data/Halcyon/<case>` or `data/Elekta/<case>` unless
   the folder already holds the needed files, and the rest of the parameters are read from `metadata.yaml` and
   `Scan.xml` with `Calibrations/` (Varian) or `reconstruction.yaml` (Elekta);
3. the preprocessing: drift correction, the air scans as flat fields, the rows under the blades left out, binning, the
   half-fan weights, the bowtie's transmission, the off-focal emission (Halcyon) and the couch model, a known object
   of the loop (in the scatter scene and the modelled primary, its line integrals removed from the corrected data);
4. the CITRUS reconstruction, saved as `results/<case>_LowRes/<case>_citrus_hu.mha` (or `_FullRes`) on the grid of
   the clinical CBCT;
5. the figure `<case>_comparison.png` in the same folder: axial and coronal slices of the deformed CT, the clinical
   CBCT and the reconstruction, and the differences of both CBCTs against the CT.

The Halcyon scans are cropped to the rows inside the axial blades; the panel's row offset follows the crop and the
counts are not shifted (the research loader `cobra/cobra_io.py` also shifted the cropped views by the crop's centre
offset, which an obsolete z correction of its volume grid cancelled).

The Elekta projections are already divided by the flood image, so the open beam is one level, the flood's norm scaled
to the scan's mA and pulse length; every pixel that sees air saturates at the panel's maximum and is set to that level.
The flood also removes the F1 bowtie from the projections but not from the scatter it causes, so the source model
carries an aluminium bowtie whose thickness rises as the square of the fan angle (the F1 profile is not published).
The couch is the iBEAM evo top, its published cross-section with the carbon and foam attenuation fitted to the rays
through the couch alone; its height is set per case in the script, from the couch's two skins under the patient.

`cobra_tools.py` holds the shared code (download, Varian and Elekta file readers, preprocessing, couch, figure).
`data/spectra/` holds the x-ray spectra: for Varian `varian_125kvp_tube.txt` (tube and window, used with the bowtie as
the source model's filter) and `varian_125kvp_lit.txt` (the bowtie's thinnest part folded in), for Elekta
`elekta_120kvp_open.txt` (the XVI's open beam, used with the F1 bowtie as the source model's filter).

## Running

From this folder, with CITRUS installed:

    python COBRA_TrueBeam_LowRes.py
    python COBRA_Halcyon_LowRes.py
    python COBRA_Synergy_LowRes.py

and the same with `_FullRes` on a large GPU.

Set `CASE` at the top of a script to choose the case, and `RECONSTRUCT = False` to redraw the figure from a saved
reconstruction.

- Data: about 1.25 GB per Varian case and 1 GB per Elekta case is downloaded (the simulated projections, the planning
  CT and the dataset's own FDK are skipped). The data folders and `results/` are not tracked by git.
- GPU: a CUDA GPU with 12 GB for the `LowRes` scripts; the `FullRes` scripts need about 60 GB. The scripts cap PyTorch
  at 90 % of the card (`GPU_MEMORY_FRACTION`): without a cap its cache grows past the card on Windows, which then
  spills into host memory and runs many times slower.
- RAM: about 25 GB free for `LowRes` (the projections of a case are read whole before binning), more for `FullRes`,
  which keeps every pixel.
