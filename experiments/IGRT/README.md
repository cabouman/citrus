# CITRUS on the COBRA2026 IGRT cases

Reconstructions of Varian pelvis CBCT scans of the COBRA2026 dataset (training split, Zenodo / CREATIS warehouse) with
CITRUS, compared with the vendor's clinical CBCT and the deformed planning CT.

| script | scanner | cases |
|---|---|---|
| `COBRA_TrueBeam.py` | TrueBeam, half fan, whole rotation (centre G) | G008, G007 |
| `COBRA_Halcyon.py` | Halcyon, full fan, short scan (centre F) | F015, F003, F041 |

Each script has five sections:

1. the case (`CASE = ...`) and the parameters its metadata does not give: the panel's drift model, the couch height and
   attenuation, the spectrum, the grid's strips, and the CITRUS settings;
2. the data: the case is downloaded into `data/TrueBeam/<case>` or `data/Halcyon/<case>` unless the folder already holds
   the needed files, and the rest of the parameters are read from `metadata.yaml`, `Scan.xml` and `Calibrations/`;
3. the preprocessing: drift correction, the air scans as flat fields, the rows under the blades left out, binning, the
   half-fan weights, the bowtie's transmission, the off-focal emission (Halcyon) and the couch model, a known object
   of the loop (in the scatter scene and the modelled primary, its line integrals removed from the corrected data);
4. the CITRUS reconstruction, saved as `results/<case>/<case>_citrus_hu.mha` on the grid of the clinical CBCT;
5. the figure `results/<case>/<case>_comparison.png`: axial and coronal slices of the deformed CT, the clinical CBCT and
   the reconstruction, and the differences of both CBCTs against the CT.

The Halcyon scans are cropped to the rows inside the axial blades; the panel's row offset follows the crop and the
counts are not shifted (the research loader `cobra/cobra_io.py` also shifted the cropped views by the crop's centre
offset, which an obsolete z correction of its volume grid cancelled).

`cobra_tools.py` holds the shared code (download, Varian file readers, preprocessing, couch, figure).
`data/spectra/` holds the x-ray spectra: `varian_125kvp_tube.txt` (tube and window, used with the bowtie as the source
model's filter) and `varian_125kvp_lit.txt` (the bowtie's thinnest part folded in).

## Running

From this folder, with CITRUS installed:

    python COBRA_TrueBeam.py
    python COBRA_Halcyon.py

Set `CASE` at the top of a script to choose the case, and `RECONSTRUCT = False` to redraw the figure from a saved
reconstruction.

- Data: about 1.25 GB per case is downloaded (the simulated projections, the planning CT and the dataset's own FDK are
  skipped). The data folders and `results/` are not tracked by git.
- GPU: a CUDA GPU with 12 GB for the TrueBeam settings (1.25 mm voxels); the Halcyon settings (2 mm) need less. The
  scripts cap PyTorch at 90 % of the card (`GPU_MEMORY_FRACTION`): without a cap its cache grows past the card on
  Windows, which then spills into host memory and runs many times slower.
- RAM: about 25 GB free (the projections of a case are read whole before binning).
