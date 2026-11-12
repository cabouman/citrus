Experiments
===========

``experiments/`` holds reconstructions of measured scans with CITRUS. Unlike the demos, which synthesize their data,
these scripts read scanner files, model the scanner from its metadata and calibrations, and compare the result with the
vendor's reconstruction. ``experiments/IGRT/README.md`` lists the files and how to run them; this page describes what
the scripts do.

The COBRA2026 pelvis scans
--------------------------

``experiments/IGRT/`` reconstructs kV cone-beam CT scans of the pelvis taken for image-guided radiotherapy, from the
training split of the COBRA2026 dataset (Zenodo and the CREATIS data warehouse). Every case holds the raw projections
with the scanner's metadata, the vendor's clinical CBCT and the planning CT deformed onto the CBCT. The scripts cover
three scanners:

.. list-table::
   :header-rows: 1
   :widths: 34 46 20

   * - scripts
     - scanner and scan
     - cases
   * - ``COBRA_TrueBeam_LowRes.py``, ``COBRA_TrueBeam_FullRes.py``
     - Varian TrueBeam (centre G), half fan over a whole rotation, 125 kV
     - G008, G007
   * - ``COBRA_Halcyon_LowRes.py``, ``COBRA_Halcyon_FullRes.py``
     - Varian Halcyon (centre F), full fan over a short scan, 125 kV
     - F015, F003, F041
   * - ``COBRA_Synergy_LowRes.py``, ``COBRA_Synergy_FullRes.py``
     - Elekta Synergy with the XVI panel (centre C), half fan over a whole rotation, 120 kV, F1 bowtie
     - C039, C041, C013

``cobra_tools.py`` holds the code the scripts share: the download of a case, the readers of the Varian files
(``Scan.xml``, the RTK geometry, ``.xim`` images, the calibration files) and of the Elekta files
(``reconstruction.yaml``), the preprocessing, the couch models, the half-fan weights and the comparison figure.
``data/spectra/`` holds the x-ray spectra of both vendors.

Low and full resolution
-----------------------

Each scanner has two scripts with the same model of the scanner and different CITRUS settings. Both run the loop as
multigrid stages (``Schedule(stages=...)``, :doc:`method`, "Multigrid stages") that start on 5 mm voxels with binned
data and a subset of the views, each stage starting from the previous stage's volume and scatter state.

* ``LowRes`` bins the data (TrueBeam 4 x 2 pixels and every second view, Halcyon 2 x 6, Synergy 2 x 2) and ends on
  1.25 mm voxels (2 mm for the Halcyon) in three stages, with MBIRTORCH's default prior. It fits a GPU of 12 GB.
* ``FullRes`` keeps every view and pixel (the Halcyon's channels in pairs, since MBIRTORCH's cone back projection
  slows down sharply on very long detector rows) and ends on the voxels of the clinical CBCT, about 1 mm, in four
  stages, with a sharper prior (``sharpness`` 1.25, ``T`` 0.25). It needs a GPU with much more memory, such as an
  80 GB H100.

The settings at the top of each script give the stages as (voxel size in cm, rows binned, channels binned, view
stride, outer iterations) on the loaded data; the scripts turn them into ``Stage`` objects with the voxel factor
relative to the final voxel. The volume spans a whole number of centimetres across, so that every stage shares the
transport grid and the scatter state carries over.

What a script does
------------------

A script runs in five sections.

1. **The case and the parameters its metadata does not give**: the panel's drift model, the couch height and
   attenuation, the spectrum, the anti-scatter grid's strips and the CITRUS settings.
2. **The data**: the case is downloaded into ``data/<scanner>/<case>`` unless the folder already holds the files the
   script needs (about 1 to 1.25 GB per case), and the rest of the parameters are read from the case's metadata.
3. **The preprocessing** of the counts (below).
4. **The CITRUS reconstruction**, written in HU on the grid of the clinical CBCT
   (``results/<case>_LowRes/<case>_citrus_hu.mha`` or ``_FullRes``), with water's attenuation at the density model's
   effective energy as 0 HU.
5. **The figure** ``<case>_comparison.png``: axial and coronal slices of the deformed planning CT, the clinical CBCT
   and the reconstruction, and the differences of both CBCTs from the CT.

Preprocessing and the scanner model
-----------------------------------

**Panel drift.** The gain and offset of each panel build up over the acquisition. The scripts remove a global drift,
the same on every pixel, as a function of the frame index: a gain that rises with one or two time constants and an
offset that rises with another (``drift_correction``). The parameters were fitted once per panel to the lines that
each scan measures twice (the conjugate rays of a half-fan overlap, or of a short scan, and the seam where the
rotation closes) and, for the Varian panels, to the open-beam level of every frame with air beside the patient. They
are shared by the cases of that panel.

**Flat field.** For the Varian scanners the flat field is the air scan of the acquisition interpolated to each view's
angle and scaled by the ratio of the milliampere-seconds. The scan's own open beam differs from that scaled level by a
factor per case (``open_beam_scale``), measured on the brightest air pixels of the scan, and the axial blades of the
scan and the air scans differ, so each row is rescaled by the scan's own air columns (``air_row_profile``). The rows
under the blades are left out. The flat field is then an image per view, which ``citrus_recon`` reads as the measured
open field. The Elekta XVI writes projections already divided by its flood image, so their flat field is one number,
the flood level scaled to the scan's exposure; the pixels that saturate in the air are set to that level. A number
keeps the solver's source as modelled, with the F1 bowtie, while an image would replace its emission profile with the
flat profile of the flood-corrected open field.

**Weights.** The half-fan scans weight the overlap of the offset panel with the redundancy weights of Wang
(``half_fan_weights``), and dead pixels get weight 0, both through ``measurement_weights``. MBIR uses the default
uniform weights otherwise.

**Source.** The Varian bowtie is the source model's aluminium beam filter, its thickness per channel from the
transmission of the scan's own air pair, with and without the bowtie (``bowtie_transmission``,
``filter_thickness_from_transmission``). The axial blades open past the panel's rows on some scans, which widens the
field (``blade_field_margin_cm``). The Halcyon's off-focal emission past the blades is measured on the scan's own air
columns and given to the source as a table. The XVI's F1 bowtie has no published profile; the Synergy scripts use an
aluminium bowtie whose thickness rises as the square of the fan angle, matched to the bowtie's published
half-value-layer change and edge thickness (``parabolic_bowtie_cm``), and a source whose field also reaches past the
long edge of the offset panel (``LongSideFieldSource``).

**Detector and grid.** The panels are CsI scintillators with the K x-ray escape model
(``detector_response(..., deposition="fluorescence_escape")``), with the thickness from the Varian metadata and an
assumed 0.6 mm for the XVI. The Varian grids are analytic lead grids with the calibration's ratio and orientation and
an assumed strip thickness and line density; the XVI has no grid. The detector and grid responses are averaged over
each energy group (``group_averaged_responses``).

**Couch.** The couch is a known object of the loop (``known_objects``): in the scatter scene and the modelled primary,
with its line integrals removed from the corrected data. The Varian couch top is the vendor's contour model in the
calibration folder, the Elekta iBEAM evo top a published cross-section (``ibeam_evo_couch``); each has one attenuation
for the carbon-fibre skins and one for the foam core, fitted to the rays through the couch alone, and a height set per
case.

**Region and start.** MBIR reconstructs the field of view and the couch (``field_of_view_region``, the model's
``use_ror_mask``). The start is an FDK of the uncorrected data (a half-fan scan rebinned to full-fan views first,
``initial_volume``), set to 0 outside that region, since the scatter solver sees every voxel of the volume.

**Loop settings.** The scripts use the beam-hardening correction with cortical bone as a hardening material, extend the
scatter scene past the measured slab (``extend_scatter_scene``), clear the air below 0.02 1/cm from the scatter scene
(``air_threshold``), compute the scatter on every 16th loaded view (every 32nd at the TrueBeam's full resolution) and
set the energy groups for a 30 cm object, with quadrature points 1 cm apart and twice the default polar directions of
the scatter image.

Running
-------

From ``experiments/IGRT/``, with CITRUS installed:

.. code-block:: bash

   python COBRA_TrueBeam_LowRes.py
   python COBRA_Halcyon_LowRes.py
   python COBRA_Synergy_LowRes.py

and the same with ``_FullRes`` on a large GPU. ``CASE`` at the top of a script chooses the case, and
``RECONSTRUCT = False`` redraws the figure from a saved reconstruction. The data and the results are not tracked by
git. The scripts read the projections of a case whole before binning, which needs about 25 GB of free RAM at low
resolution and more at full resolution, and they cap PyTorch's share of the GPU memory (``GPU_MEMORY_FRACTION``), since
on Windows an allocator that grows past the card spills into host memory and runs many times slower.
