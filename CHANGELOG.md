# Changelog

## v0.10.0 - 2026-10-18

- **Adaptive numbers of iterations by default.** `CitrusConfig.num_outer_iterations` and
  `mbir_iterations_per_outer_iteration` default to `"adaptive"`: the MBIR iterations of an outer iteration stop once
  the relative RMSE of A X against the corrected sinogram improves by less than `mbir_stop_relative_improvement`
  (0.02) per iteration (mean over the last two; at least two, at most `max_mbir_iterations_per_outer_iteration`, 8),
  and the loop stops once an outer iteration improves that residual by less than `stop_relative_improvement`
  (0.005; at most `max_outer_iterations`, 40): a convergence criterion without ground truth. Integers run fixed
  counts as before. The residual is recorded per outer iteration (`sinogram_nrmse_start`, `sinogram_nrmse`)
  whenever the error sinogram is resumed, `info["mbir_iterations_total"]` counts the MBIR iterations, the verbose
  line prints the fit, and `mbir_recon(..., mbir_iterations_schedule=...)` repeats a CITRUS run's schedule without
  the correction (demo 2 does, so its comparison stays matched).
- The MBIR iterations of an outer iteration keep the resumed error sinogram without the per-iteration shift to and
  from -A X (one fewer rounding per iteration; results change in the last digits only).
- `benchmarks/adaptive_mbir_study.py`: fixed 1, 2, 4, 8 MBIR iterations per outer iteration against the adaptive
  rule on the data of demo 2, error against the truth and against the corrected sinogram over wall time.

## v0.9.2 - 2026-10-18

- `citrus_recon` and `mbir_recon` take `mbir_callback(i, X, record)`, called for the FDK initialization (i = 0) and
  after every MBIR iteration, next to the per-outer-iteration `callback`; `record["sinogram_nrmse"]` is the relative
  RMSE of the forward projection of X against the scatter-corrected sinogram of its outer iteration, a convergence
  measure that needs no ground truth (free with the resumed error sinogram, one forward projection per iteration
  otherwise).
- demo 2 records, after every MBIR iteration of CITRUS and of the uncorrected MBIR, the relative RMSE of the volume
  against the truth and of its projection against the corrected sinogram, and plots both convergence curves
  (`plot_history`); its slices use the maximum of the truth as the upper end of the gray scale.

## v0.9.1 - 2026-10-17

- **Automatic fine energy grid.** `fine_bin_width_kev` of the library build defaults to None: every group is cut into
  at most 100 equal bins of at least 0.5 keV (`fine_energy_grid`), so a kV library is unchanged (its groups are
  narrower than 50 keV) while a 6 MV library has 1200 fine energies instead of 12,000 and builds in about 20 s
  instead of minutes; the group constants and the transfer at common angles are closer to a 2 keV reference than
  those of the uniform 5 keV grid of the 6 MV validation. An explicit width still gives a uniform grid. The 6 MV
  demo run no longer spends minutes in the first scatter stage, and the primary sums over ten times fewer energies.
- demo 2 shows the 6 MV spectrum and a monoenergetic source as commented-out alternatives of its spectrum line.

## v0.9.0 - 2026-10-17

- **An outer iteration is one scatter correction followed by n MBIR iterations.** `CitrusConfig.scatter_every` is
  replaced by `mbir_iterations_per_outer_iteration` (default 1), and `num_outer_iterations` counts scatter
  corrections: `CitrusConfig(num_outer_iterations=10, mbir_iterations_per_outer_iteration=8)` runs the same sequence
  as the former `num_outer_iterations=80, scatter_every=8` (the MBIR iterations keep their seeds and schedule
  positions, counted over the whole loop). X_0 is the FDK initialization and outer iteration k = 1, ..., N gives X_k;
  `records[k]` and `callback(k, X_k, record)` follow that count, a record holds `mbir_iterations` and the
  `mbir_seconds` of all its MBIR iterations, and `scatter_refreshed` is gone (every outer iteration has a scatter
  stage). `benchmarks/citrus_study.py` takes `--outer-iterations` (default 10) and `--mbir-iterations` (default 8)
  instead of `--scatter-every`, and tags its outputs `_mbir<n>`.
- The README and the docs describe the loop in these terms, and their examples pass the spectrum.

## v0.8.0 - 2026-10-17

- **The source spectrum is required.** There is no default spectrum any more: `build_library`, `compute_library`,
  `build_solver_library`, `sort_material_table` and `transfer_at_eta` take it as their second positional argument, a
  lazy `ScatterSolver` (`library=None`) and `wlab_solver` / `synthesize_wlab_dataset` without a library need
  `spectrum=`, and `mass_attenuation_preview` needs it unless `energy_kev` is given; a missing spectrum raises a
  `ValueError` that names `load_spectrum_txt` and `monoenergetic_spectrum`. `default_spectrum_125kvp` and the package
  data file are gone (`cross_section_library._spectrum_or_default` is now `spectrum_arrays`); the sample spectra live
  in `demo/spectra` only.
- **Monoenergetic sources.** A spectrum of one node, `monoenergetic_spectrum(energy_kev)` or a one-line spectrum
  file, is a line: the library puts all source photons in the fine bin that holds it and evaluates that bin at the
  line's energy, so the effective energy and the primary attenuation are exactly those of the line; the group edges
  come from the "scattered_fluence" (default) or "log" rule ("equal_fluence" raises for a line). `citrus.scatter_solver`
  exports `monoenergetic_spectrum`.
- **Tissue names carry their source.** `lung_icrp`, `adipose_icrp`, `bone_cortical_icrp` (ICRP compositions),
  `muscle_icru44`, `soft_tissue_icru44`, `brain_icru44`, `blood_icru44`, `skin_icru44` (ICRU Report 44) and
  `trabecular_bone_icru46`, `cartilage_icru46` (Woodard and White 1986 / ICRU Report 46) replace `lung`, `adipose`,
  `bone`, `muscle`, `soft_tissue`, `brain`, `blood`, `skin`, `trabecular_bone` and `cartilage`, next to NIST's
  `adipose_icru44`, `bone_cortical_icru44`, `lung_icru44`, ... The default reference materials are `("air",
  "lung_icrp", "adipose_icrp", "water", "muscle_icru44", "soft_tissue_icru44", "bone_cortical_icrp")`; the WLAB phantom
  needs `water` and `bone_cortical_icrp`. The library cache keys include the names, so cached libraries rebuild.
- The conventions page lists the catalogue in one table of name, composition (or NIST source) and nominal density,
  without the attenuation column.
- The NIST compounds are part of the one `_COMPOUNDS` table (`NIST_COMPOUND_NAMES` of v0.7.2 is gone) and of
  the one table of the conventions page.
- Tests pass the sample 125 kVp spectrum of `demo/spectra` (`tests/conftest.py`); `tests/test_monoenergetic.py`
  checks the line placement, the exact primary transmission and the group rules.

## v0.7.2 - 2026-10-17

- **NIST compounds in the material catalogue.** The 48 compounds and mixtures of NIST's X-ray mass attenuation
  coefficient tables (Hubbell and Seltzer, Table 2) are catalogue entries with NIST's densities and mass fractions;
  seven of them were already entries with the same composition (`water`, `muscle`, `soft_tissue`, `brain`, `blood`,
  `pmma`, `polyethylene`), so 41 names are new, from `a150_tissue_equivalent_plastic` to
  `tissue_equivalent_gas_propane` (the ICRU-44 tissues carry an `_icru44`
  suffix, `adipose_icru44`, `bone_cortical_icru44`, `lung_icru44`, ...). Every one builds a library on its own.
- **`aluminium` and `titanium` are element entries** like the other 96 elements (they were duplicated among the
  compounds); aluminium's nominal density is now xraylib's 2.6989 g/cm^3 instead of 2.699.
- **Sample spectra in `demo/spectra/`**: the 125 kVp tube spectrum and the 6 MV linac spectrum of the validation
  (Mohan et al. 1985, 1 keV nodes to 6 MeV, `spectrum_6mv.txt`) as plain two-column files, read by the demos the way
  a user reads their own; the package data keep the 125 kVp file as the default spectrum of the API, and a test keeps
  the two copies equal. `load_spectrum_txt` is exported by `citrus.scatter_solver`; the copy of the 6 MV spectrum in
  `tests/data` is gone.
- Docs: the conventions page lists the hand-written entries and the NIST entries in two tables, without the elements.

## v0.7.1 - 2026-10-17

Style release: the docstrings, the demos and the docs pages. No change of behaviour.

- **Docstrings in the Google format of MBIRTORCH** in every module of the package, the benchmarks and the docs
  figure script: one to three sentences on what the function does, then `Args:` (one line per parameter with its
  shape, unit and default) and `Returns:`; classes and dataclasses list their fields under `Attributes:`. The module
  docstrings say in a few sentences what the module contains; the derivations, conventions and design notes they
  carried are in `docs/source/method.rst`, `conventions.rst` and `design_notes.rst` (the missing ones were added
  there), the version history and the measurements are dropped.
- **Docs:** the `_literal_docstrings` hook of `docs/source/conf.py` is gone, so napoleon renders the Google docstrings
  (parameters and returns as field lists); `napoleon_numpy_docstring` is off and the type hints are added to the
  documented parameters only (`autodoc_typehints_description_target = "documented"`). Two stray tab characters
  (`\cos\theta` in conventions.rst, `C:\path\to` in installation.rst) are fixed.
- **Two quick-start demos** in the style of MBIRTORCH's (a linear script, a comment before every step, under 100
  lines, no flags, every setting a variable in a parameter block at the top: spectrum file, reference materials,
  panel, voxel pitch, views and angular range, flat field, loop and solver settings):
  `demo/demo_1_scatter_solver.py` (the scatter solver on one view of the WLAB phantom: primary, scatter and scatter
  fraction) and `demo/demo_2_citrus.py` (synthetic data of 180 views over a short scan at 0.5 cm from
  `synthesize_cone_beam_data`, cached in `demo/input/` under a name built from the settings; the solver built from
  `SolverConfig` and the spectrum; `citrus_recon` and `mbir_recon`, the relative RMSE of both, central slices and
  profiles).
  They replace `demo/demo_scatter_solver.py` and `demo/demo_citrus.py`.
- **`benchmarks/citrus_study.py`**: the former `demo/demo_citrus.py` with all its flags, defaults, output names and
  summary JSON unchanged (data sets in `demo/input/`, outputs in `demo/output/`). `docs/source/usage.rst` describes
  the demos and the study script ("CITRUS studies"); README and `dev_scripts/wsl_install_test.sh` run the new demos.
- **Tests:** `test_default_set_keeps_the_cache_key_and_library` (tests/test_material_catalogue.py) compared a SHA-256
  of the bytes of the small legacy library with a value recorded on Windows, which fails on Linux where the float64
  arithmetic of the builder differs in the last bits. It now compares the shape and float64 statistics of every
  tensor (sum, sum of squares, min, max, the first and last three values) with
  `tests/data/small_legacy_library_v070.json`, at a relative tolerance of 1e-9 (absolute 1e-12).

## v0.7.0 - 2026-10-17

Photon data from NIST XCOM and xraylib instead of XrayPhysics, so that CITRUS installs from wheels without compiling
anything. **Breaking changes: the photon data change every library** (the cross sections, and with them the
scattered-fluence group edges, the adaptive angular nodes and every primary and scatter result), the cache version is
`citrus-xsec-4` with the backend name `"xcom-xraylib"` in the key (every cached library is rebuilt once),
XrayPhysics and the `physics` extra are gone, xraylib is a required dependency, `default_spectrum_125kvp()` reads a
data file and takes no backend (tube spectra are no longer simulated), and the element entries of the catalogue end at
californium (Z = 98). The validation against the Monte Carlo at 125 kVp and 6 MV and the comparison of the new data
with XrayPhysics and with Geant4 are in the validation folder (`validation/wlab/production_6mv_status.md`, "citrus
0.7.0").

### Photon data: NIST XCOM and xraylib

- **New** module `citrus.scatter_solver.photon_data` with the backend `XcomXraylibBackend` (`resolve_backend()`,
  re-exported by `cross_section_library` with `default_cache_dir`), the interface of the builder unchanged:
  `sigma(kind, Z, E)`, `pdf_raw(kind, Z, E, eta)`, `library_pdf_normalization(kind, Z, E)`, `atomic_mass(Z)`.
- **Integrated cross sections from NIST XCOM 3.1** (M.J. Berger et al., XCOM: Photon Cross Sections Database, NIST
  Standard Reference Database 8): the element tables MDATX3.001 .. MDATX3.100 (1 keV .. 100 GeV, barns/atom), read
  and interpolated as XCOM's Fortran program does it: natural cubic splines in (ln E, ln sigma) for coherent and
  incoherent scattering over the grid without the points just below the edges; for photoelectric absorption the same
  spline above the K edge and XCOM's own short tables of every interval between edges below it, interpolated linearly
  in log-log, the value above an edge at the edge; pair production zero up to its thresholds (1.022007 / 2.044014 MeV)
  and splined in ln(sigma / ((E - E_threshold) / E)^3) above; barns -> cm^2/g with XCOM's Avogadro factor and atomic
  weights (ATWTS.DAT). `sigma` kinds: `"total"` (the sum of all five processes, as before with pair production
  included), `"photoelectric"`, `"compton"` (incoherent), `"rayleigh"` (coherent) and, **new**, `"pair"` (nuclear plus
  electron field). `XcomXraylibBackend.element(Z)` gives the parsed table (`XcomElement`).
- **Angular shapes from xraylib** (T. Schoonjans et al., Spectrochimica Acta B 66, 776 (2011)): Compton = xraylib's
  `DCS_KN` x `SF_Compt` (its `DCS_Compt`), Rayleigh = `DCS_Thoms` x `FF_Rayl`^2 (its `DCS_Rayl`), in cm^2/g/sr, from
  q = sin(theta/2) without an angle clamp; below xraylib's smallest tabulated momentum transfer (1e-3 and 1e-7 per
  angstrom) S is continued as x^2 and F held constant. The builder still normalizes every elemental shape to the XCOM
  cross section of its process, so the scattering integrates to XCOM's whatever the integral of xraylib's shape.
- `library_pdf_normalization` is the sphere integral of the shape on a fine quadrature (61 geometric pieces in q),
  so `BuildInfo.pdf_normalization_max_relative_deviation` (the check of the builder's reference quadrature) is always
  reported.
- Atomic weights (formula -> mass fractions, and barns -> cm^2/g) from XCOM's table (ATWTS.DAT, NIST's standard
  atomic weights): water is H 0.111898, O 0.888102 (with the weights of version 0.6: H 0.111907). xraylib's
  `AtomicWeight` is not used (rounded, H 1.01 and O 16.0, and wrong for thulium, 167.27 instead of 168.934).
- **Download on first use.** The XCOM data are not shipped: the first build fetches
  `https://physics.nist.gov/PhysRefData/Xcom/XCOM.tar.gz` (172805 bytes, SHA-256 checked) into
  `default_cache_dir()/xcom` (`$CITRUS_CACHE_DIR/xcom` or `~/.cache/citrus/xcom`), reads the element files from the
  archive in memory and writes the parsed tables to `xcom_elements.npz` there. **New** environment variable
  `CITRUS_XCOM_ARCHIVE`: the path of a downloaded XCOM.tar.gz or of the folder it unpacks to, for offline machines.
  Without data, cache and network the build raises a RuntimeError naming the URL and the variable.
  `photon_data.load_xcom_tables`, `read_xcom_source`, `parse_xcom_element`, `parse_xcom_atomic_weights`.
- The library cache version is `citrus-xsec-4`; the backend name `"xcom-xraylib"` is part of the key.

### Default spectrum

`cross_section_library.default_spectrum_125kvp()` reads the package data file
`citrus/scatter_solver/data/spectrum_125kvp.txt`: the 125 nodes the library used so far (W anode, 125 kV, 11 degree
take-off, 0.89 mm Ti, simulated once with XrayPhysics), the same numbers as `validation/wlab/spectrum_125kvp.txt`.
User spectra are loaded with `load_spectrum_txt` as before. The `backend` argument is removed.

### Material catalogue

- The compositions and nominal densities of the compounds are unchanged (they define the WLAB validation
  materials). Checked against xraylib's NIST compounds: cortical bone and adipose tissue are 'Bone, Cortical (ICRP)'
  and 'Adipose Tissue (ICRP)' exactly; lung is 'Lung (ICRP)' without the Ca, Fe and Zn traces (every mass fraction
  within 4e-4); air reads the volume percentages of dry air as mass fractions (as before; not NIST's dry air).
- The elements are Z = 1..98 (`hydrogen` ... `californium`): xraylib has no scattering functions for einsteinium and
  fermium, which are removed. `ELEMENT_SYMBOLS` and `ELEMENT_DENSITIES` come from xraylib (`AtomicNumberToSymbol`,
  `ElementDensity`: NIST's densities of Hubbell and Seltzer, gases at 20 C). The table embedded in version 0.6 (from
  XrayPhysics) differed for 18 elements, mostly the gases (at 0 C there: hydrogen 8.988e-5 instead of 8.4e-5 g/cm^3,
  nitrogen 1.2506e-3 instead of 1.165e-3, oxygen 1.429e-3 instead of 1.332e-3), carbon 2.265 instead of 2.0, sulfur
  2.07 instead of 2.0, astatine, francium, californium; it is dropped. **New** `cross_sections.NUM_ELEMENTS` (98).

### Removed

- `cross_section_library._XrayPhysicsBackend`, its `spectrum_125kvp` (the tube-spectrum simulation) and the angle
  clamp of XrayPhysics' Rayleigh amplitude; the `physics` extra of `pyproject.toml`.
- The installation recipes for XrayPhysics (C++ toolchain, CMake, conda compilers) in the README, the documentation
  and `dev_scripts/wsl_install_test.sh`; `dev_scripts/check_cuda.sh` reports xraylib, the XCOM data and the cache
  instead, `create_conda_test_environment.sh` fetches the XCOM data.

### Demo

The cached data sets of `demo/demo_citrus.py` were made with the XrayPhysics library; regenerate them with
`--regenerate` (the cache name is unchanged).

### Tests and references

**New** `tests/test_photon_data.py`: XCOM totals against NIST's tables (water from 10 keV to 10 MeV, lead around its
K edge, to the tables' four digits), total = sum of the processes, the pair thresholds, the interpolation through the
table points, the photoelectric jump at every edge, the spline against a line-by-line port of XCOM.f's SCOF / BSPOL,
xraylib's DCS_Compt / DCS_Rayl reproduced by the shapes and their continuation below xraylib's tables, the fine
normalization against the Klein-Nishina cross section, `$CITRUS_XCOM_ARCHIVE` with an archive and an unpacked folder
without network access, the parsed cache, and the errors without data or with a wrong download.
`tests/test_cross_section_library.py`: the water totals against NIST's tables to 2e-3 (were 2 %, the tolerance
of EPDL97 against XCOM), mu_t = photoelectric + Compton + Rayleigh to 1e-12 (an identity of the XCOM data below the pair
threshold; was 2e-3). `tests/test_material_catalogue.py`: the 98 elements with xraylib's symbols and densities, the
default compounds against xraylib's NIST compounds, californium among the sampled builds.

**The regression references now encode the XCOM + xraylib data**: the real-library WLAB cases of
`tests/data/pre_quadrature_reference.json` were regenerated with this version (the synthetic cases are unchanged),
`tests/test_material_catalogue.py` holds the cache keys of version 0.7.0 and the hash of the legacy-options build
recorded with it (instead of the version 0.5.1 library, which encoded XrayPhysics data), and the nominal attenuations of
`tests/test_materials.py`, the docstrings and the documentation are those of the new data. Against the Monte Carlo
(Geant4 with Livermore, EPDL-based data) the primary is now slightly lower than with version 0.6.0, the difference of
the XCOM and EPDL97 attenuation over the phantom's water, not a transport error (`validation/wlab`).

## v0.6.0 - 2026-10-17

An adaptive angular transfer table and energy groups that follow the scattered photons, so that the solver is accurate
from kV to MV with the same table size and group count. **Breaking changes: the new defaults change the library**
(the angular nodes, the group edges and with them every scatter result), the cache version is `citrus-xsec-3` (every
cached library of earlier versions is rebuilt once; the names of the unchanged library fields are kept), and the
CITRUS loop builds the solver's library after the FDK reconstruction. The options of version 0.5 reproduce its
library bit for bit: `SolverConfig(angular_table_grid="uniform", group_edges_method="equal_fluence")` (tested by a
hash of the version 0.5.1 build and by the WLAB regression anchors). The validation against Monte Carlo at 125 kVp
and 6 MV is in the validation folder (`validation/wlab/production_6mv_status.md`).

### Adaptive angular table

- **New** `SolverConfig.angular_table_grid`: `"adaptive"` (default) or `"uniform"` (the table of version 0.5). The
  first-collision source's direct transfer table `mu_s_eta [M, G, G, num_eta_table]` (still 2001 nodes by default)
  places its nodes by a map u = h(eta) built from the transfer curves: in q = sin(theta/2) the node density is the
  maximum over the curves (Compton with its kinematic group fractions plus Rayleigh, of the catalogue's water and
  cortical bone in the library's groups, `cross_section_library.ANGULAR_MAP_MATERIALS`, so every material set shares
  the nodes) of sqrt(|f''| / max f), smoothed, with a floor of 10 % of the density of a uniform table
  (`cross_section_library.angular_map`; curves on 20001 points uniform in q from cubic splines through the reference
  quadrature). At MeV energies the forward peaks are far narrower than the spacing of a uniform table, whose linear
  interpolation overestimated the first-order scatter; the adaptive nodes resolve them at the same table size.
- **Lookup stays index arithmetic.** `MultigroupLibrary.eta_map [512]` (None for a uniform table) holds u at equally
  spaced knots of q; `cross_sections.angular_table_position(eta, N, eta_map)` evaluates q = sqrt((1 - eta)/2), one
  lookup in the map, pos = u (N - 1), then the existing two-node interpolation; `angular_table_nodes(N, eta_map)` gives
  the node cosines (`eta_grid`). `transfer_eta_at`, `transfer_eta_at_pairs` and the fused Triton first-collision
  kernel (dense and pair modes; the map is an L1-resident table of 512 values) use it; the kernel's self-check covers
  uniform and adaptive tables (`check_cuda.sh`). `synthetic_library(..., angular_table_grid="adaptive")` and
  `synthetic_angular_map()` give a synthetic adaptive table for tests.
- The map's knots are equally spaced in q, not in eta: the forward peaks are a few 1e-7 wide in 1 - eta at 6 MeV,
  narrower than the knot spacing of any small table uniform in eta, while in q they are smooth and spread out.
- **New** `cross_section_library.transfer_at_eta(config, eta, ...)`: the exact transfer at any cosines (no
  interpolation), to measure the interpolation error of a table.
- The library cache stores `eta_map`; `MultigroupLibrary.load` reads a file without it as a uniform table.

### Energy groups

- **New** `SolverConfig.group_edges_method`: `"scattered_fluence"` (default), `"equal_fluence"` (the rule of version
  0.5, unchanged to the last bit) or `"log"` (geometric spacing between `energy_min_kev` and the spectrum maximum).
  Explicit `group_edges_kev` still override it. "Equal area under the spectrum" taken on the SOURCE spectrum is
  `"equal_fluence"`; the default takes equal areas under an estimate of the total photon fluence in the object, source
  plus scattered photons, because the groups must resolve the downscattered photons that carry the multiple-collision
  scatter (at 6 MV equal source fluence left one group across 10-609 keV).
- **New** `cross_section_library.scattered_fluence_spectrum`: the estimate, a one-dimensional slowing-down balance in
  the object's medium with Klein-Nishina downscatter (closed-form bin probabilities), Rayleigh left out, photons below
  `energy_min_kev` absorbed and a leakage loss 1/L,
  phi(E) = [S(E) + int mu_C(E') p(E | E') phi(E') dE'] / (mu_t(E) - mu_R(E) + 1/L), one sweep from the highest
  energy down. **New** `SolverConfig.grouping_object_size_cm` = L (default 20 cm), the characteristic chord length of
  the object: L -> 0 recovers `"equal_fluence"`, L -> infinity the infinite medium. A grouping heuristic only.
- **New** `SolverConfig.grouping_medium`: the medium of the balance, `"auto"` (default: the object's mean densities
  of the reference materials, measured by `citrus_recon` on the FDK volume and by `synthesize_cone_beam_data` on the
  true object; water at 1 g/cm^3 while unmeasured), a catalogue name (that material at its nominal density, e.g. a
  part of one known material) or a mapping {reference material: density g/cm^3}; named materials must be among
  `SolverConfig.materials`. Stored as sorted (name, density) pairs; `cross_section_library.grouping_medium_densities`
  gives the form used by the balance and the cache key (3 significant digits, traces below 1e-3 of the largest
  density dropped).
- **Within-group weights of the "scattered_fluence" rule.** Its group constants weight the energies within each group
  by the estimated total fluence instead of the source spectrum (`_within_group_weights`; the other rules and
  explicit edges keep the source weighting of version 0.5). At MV the lowest group spans 10 keV to about 150 keV with
  a flat or no source spectrum, while its scattered photons sit near its top; a source weighting gave it the
  photoelectric attenuation of its lowest energies and absorbed the multiply scattered photons too early. The
  spectrum fractions and the fine-energy table of the uncollided fluence keep the source spectrum.
- `make_group_edges(config, spectrum, backend=None, *, fine_bin_width_kev=0.5)` takes the backend (needed by the
  default rule). **New** `mass_attenuation_preview(config, names, ...)`: E_eff and the mass attenuation of catalogue
  materials without building the library.
- The cache key holds `angular_table_grid`, `group_edges_method`, `grouping_object_size_cm` and the grouping medium.

### Object size and the lazy library

- **New** `citrus.utilities.object_size_cm(X, grid, reference)`: the mean chord 4 V / S of the object in an
  attenuation volume, the mask being the voxels above the attenuation midway between air and water at E_eff
  (`air_water_threshold(reference)`; cavities such as the WLAB lung and air inserts are holes of the mask), V the mask
  volume, S the area of the exposed voxel faces (the neighbour outside the mask or the grid), rounded to whole
  centimetres, at least 1 cm, the default for an empty mask.
- **Lazy library.** `ScatterSolver(None, config, geometry, ..., spectrum=None)` builds its library on first use (the
  `library` / `transport_library` properties; `library_built`; `library_cache_dir`); `solver.config` can be replaced
  until then and raises `RuntimeError` afterwards (an equal config is accepted). `solver.materials` gives the sorted
  material names without a build, `solver.num_groups` falls back to `config.num_groups`. A solver given a library
  works as before. `utilities.wlab_solver(..., spectrum=None)` returns the lazy form (its `library=` argument still
  takes a prebuilt one).
- **New** `citrus.utilities.grouping_medium_estimate(X, reference)`: the object's mean composition, the density
  model applied to the object voxels (`object_mask`) and averaged per reference material, from the nominal
  attenuations alone (no library needed).
- **New** `CitrusConfig.grouping_object_size`: `"auto"` (default) or a size in cm. `citrus_recon` measures the object
  on the FDK volume (`object_size_cm`, and `grouping_medium_estimate` for an "auto" medium), writes both into a lazy
  solver's config before the first scatter stage builds the library (once per run, cached per size and medium), and
  records `info["grouping_object_size_cm"]`, `info["grouping_medium"]`, `info["grouping_applied"]` and
  `info["library_seconds"]`; a prebuilt library keeps its config (verbose prints a note). Only the
  "scattered_fluence" rule without explicit edges uses them; `mbir_recon` is unaffected.
- `synthesize_cone_beam_data` sets a lazy solver's size from the attenuation of the true object and, for an "auto"
  medium, the true mean densities before its first solve (meta `grouping_object_size_cm`, `grouping_medium`);
  `synthesize_wlab_dataset` places the phantom in `solver.materials`.
- Demo `demo/demo_citrus.py`: the loop's library is built by `citrus_recon` after the FDK; the summary JSON records
  `grouping_object_size_cm` and `truth_grouping_object_size_cm`. The cached data sets of earlier versions were made
  with the library of version 0.5; regenerate them with `--regenerate` (the cache name is unchanged).

### Material catalogue: every element

`cross_sections.MATERIAL_CATALOGUE` keeps its seventeen compound, tissue and mixture entries first (`COMPOUND_NAMES`)
and then holds every element Z = 1..100 of EPDL97 under its English (IUPAC) name, `hydrogen` ... `fermium`, among
them `lead` (Pb, 11.35 g/cm^3) and `tungsten` (W, 19.3 g/cm^3); aluminium and titanium remain the existing entries.
The entries are generated from `ELEMENT_NAMES`, `ELEMENT_SYMBOLS` (also the symbol table of the formula parser) and
`ELEMENT_DENSITIES`, the standard densities of XrayPhysics' element table (NIST's standard-state densities of Hubbell
and Seltzer, NISTIR 5632; tested against XrayPhysics). An unknown name lists the compounds and "every element by its
English name"; so do the demos' `--materials` help texts. Elements with K edges inside a kV spectrum (lead at 88 keV,
tungsten at 69.5 keV) build like any other entry.

### Tests

`tests/test_angular_table.py` (the map and its nodes; interpolation error of the uniform and the adaptive table against
`transfer_at_eta` on water at 125 kVp and 6 MV; the l = 0 integral of the interpolated table against the group cross
section; kernel against PyTorch on adaptive tables), `tests/test_group_edges.py` (the three rules on both spectra,
equal-fluence edges identical to version 0.5.1, edges below 600 keV at 6 MV, the object-size limits, the balance, the
object size of boxes and of the WLAB phantom at 1 and 0.5 cm, the lazy solver, citrus_recon measuring the object and
building once, the grouping medium: "auto" on the WLAB phantom, explicit water, a two-material mapping, refused
names; the fluence-weighted group constants), `tests/test_material_catalogue.py` (every entry parses with a positive
density, the element densities equal XrayPhysics'; one library build per compound and per sampled element, every 10th
atomic number plus lead and tungsten), `tests/data/spectrum_6mv.txt` (the Mohan 6 MV spectrum of the validation).
The regression anchors of `tests/data/pre_quadrature_reference.json` run with the options of version 0.5; the
cache-key test holds the keys of version 0.6.0 and the hash of the version 0.5.1 library.

## v0.5.1 - 2026-10-17

Ground-truth data for fine reconstruction grids. `utilities.synthesize_cone_beam_data` takes a density MODEL that gives
one material map at a time (`utilities.WlabPhantom`, the WLAB phantom as such a model; a volume rho [M, R, C, S] is
wrapped in `utilities.StackedDensities`) and builds the solver's fields once with
`ScatterSolver.material_fields_from_attenuation`, shared by every view through the new
`ScatterSolver.solve_view_from_fields`; the fine grid never holds the [M, R, C, S] volume, so the truth of a 0.5 mm
reconstruction fits the GPU like the reconstruction. `synthesize_wlab_dataset` uses `WlabPhantom`; `wlab_phantom`
stacks its maps. The data sets no longer store `rho_true` (M fine volumes); `X_true` stays. The solver's primary of an
empty volume (`phi_air`) comes from zero path lengths instead of a zero volume. The demo's `--memory-cap-mib` defaults
to 85 percent of the device's memory instead of a fixed value.

## v0.5.0 — 2026-10-17

The CITRUS loop always computes its primary with MBIRTORCH; a material catalogue and user-selectable reference materials
for the density model. **Breaking changes** (the Siddon primary option of the loop is removed); **the defaults are
unchanged**: the default reference materials give the cache key and the library of version 0.4.0 bit for bit, the WLAB regression
reference materials hold, `check_cuda.sh` reports every kernel matching, and the demo at its defaults gives CITRUS 0.0215 / 0.0259
on the cached `wlab_180v_0p5cm_short_v2` data set as before.

### The loop's primary is MBIRTORCH's per-material primary

- **Removed** `CitrusConfig.primary_projector` (and its validation), `citrus.citrus.SiddonViewProjector`,
  `citrus.citrus.PRIMARY_PROJECTORS` and `citrus_recon`'s `info["primary_projector"]`. The primary of every view in
  the loop is always `primary_signals` with `recon.mbir_bridge.ViewChunkProjector`: MBIRTORCH's pixel-integrated
  per-material line integrals, the same operator A as the reconstruction, through the solver's polychromatic
  transmission. `primary_signals(solver, fields, angles, projector, ...)` takes a `ViewChunkProjector`; its default
  chunk is max(1, V // (M' + 1)) views (the Siddon sub-pixel factor is gone). `CitrusConfig.primary_views_per_chunk`
  stays. Passing `primary_projector=` to `CitrusConfig` now raises `TypeError`.
- **Removed** the `primary_projector` argument of `utilities.synthesize_cone_beam_data` / `synthesize_wlab_dataset`:
  the SPR's denominator is always the MBIRTORCH-based primary. **On-disk format unchanged**: the data-set meta keeps
  `"primary_projector": "mbirtorch"` and `"data_format": 2`, so the data sets of versions 0.3.0 and 0.4.0 stay valid
  and are reused; nothing had to be regenerated (the demo's default data set is still
  `demo/input/wlab_180v_0p5cm_short_v2.npz`).
- Demo `demo/demo_citrus.py`: **removed** `--primary-projector` (output suffix `_siddon`) and
  `--truth-primary-projector` (cache suffix `_primary-siddon`; such data sets are no longer loaded); the summary JSON
  drops `primary_projector` and `truth_primary_projector`. `benchmarks/bench_citrus_memory.py`: `--primary-projector`
  removed.
- **Kept**: the scatter solver's own primary (`ScatterSolver.primary`, `primary_path_lengths`,
  `primary_from_path_lengths`, `scatter_image.primary_image`, `solve_view`'s primary and scatter fraction): the
  standalone solver has no MBIRTORCH model. `tests/test_attenuation_path.py` keeps the cross-check of the two
  operators (the loop's MBIRTORCH primary against the solver's Siddon primary per view: relative L2 < 5 % with the
  synthetic library, < 1.5 % with the real one; median pixel < 0.2 %). The loop-against-loop test of the two options
  in `tests/test_citrus_recon.py` is removed; the Siddon loop's demo numbers stay in the docs as history (0.0257 /
  0.0305 on the default data set).

### Material catalogue

`cross_sections.MATERIAL_CATALOGUE` (read-only ordered mapping name -> (nominal density g/cm^3, XrayPhysics
composition string)); every entry builds in the library (tested one by one). XrayPhysics' own material list
(`xrayPhysics().get_material_library()`) supplies air, lung, adipose, cortical bone, PMMA and polyethylene; it has no
ICRU-44 brain, blood, skin, cartilage or spongiosa, which are entered from the cited tables.

| name | density | mu at 63 keV (1/cm) | composition source |
|---|---|---|---|
| `air` | 0.0012 | 0.00022 | XrayPhysics `air` |
| `lung` | 0.26 | 0.0520 | XrayPhysics `lung` (WLAB lung density) |
| `adipose` | 0.95 | 0.1846 | XrayPhysics `adipose_tissue` (ICRP) |
| `water` | 1.0 | 0.2013 | H2O |
| `muscle` | 1.05 | 0.2102 | ICRU Report 44, skeletal muscle |
| `soft_tissue` | 1.06 | 0.2122 | ICRU Report 44, soft tissue |
| `bone` | 1.85 | 0.5366 | XrayPhysics `bone_cortical` (ICRP cortical bone) |
| `trabecular_bone` | 1.18 | 0.2697 | Woodard and White (1986), skeleton spongiosa (ICRU Report 46) |
| `brain` | 1.04 | 0.2092 | ICRU Report 44, brain (grey / white matter) |
| `blood` | 1.06 | 0.2130 | ICRU Report 44, whole blood |
| `cartilage` | 1.10 | 0.2234 | Woodard and White (1986), cartilage (ICRU Report 46) |
| `skin` | 1.09 | 0.2160 | ICRU Report 44, skin |
| `pmma` | 1.18 | 0.2230 | XrayPhysics `PMMA` (C5H8O2) |
| `polyethylene` | 0.94 | 0.1826 | XrayPhysics `Polyethylene` (C2H4) |
| `aluminium` | 2.699 | 0.7034 | Al (NIST density) |
| `titanium` | 4.54 | 3.0928 | Ti (NIST density) |
| `carbon_fibre` | 1.56 | 0.2756 | carbon-fibre epoxy laminate (couch): 60 % fibre volume, 0.692 C + 0.308 C21H24O4 by mass |

The formula strings of the seven default materials moved from `cross_section_library` to `cross_sections`
(verbatim); `cross_section_library.MATERIAL_FORMULAS` now maps every catalogue name to its composition.
`cross_sections.material_table(names)` gives the (name, nominal density, formula) table of catalogue names,
`check_material_names(names, minimum=2)` checks them (unknown names list the catalogue), `material_index(library or
names, name)` finds a row by name. `MATERIALS`, `MATERIAL_TABLE`, `NOMINAL_DENSITIES` are the default seven as
before; `WATER = 3` and `BONE = 6` are kept as the rows of the DEFAULT set.

### Reference materials: `SolverConfig.materials`

- **New** `SolverConfig.materials: tuple[str, ...]` (default `cross_sections.MATERIALS`, the seven in the present
  order): catalogue names of the reference materials of the density model, at least two, any order (`validate` checks them; a
  list is stored as a tuple). `build_solver_library(config)` (and so `utilities.wlab_solver`) builds the library for
  exactly that set; an explicit `materials=` table still overrides it.
- **Sorted by attenuation.** `cross_section_library.sort_material_table(config, materials, spectrum, backend,
  fine_bin_width_kev=)` sorts a table by nominal attenuation at the spectrum-mean energy of the build (63.03 keV);
  `build_library` and `compute_library` use it, so any order of a set gives the same library, and the existing check
  (`check_material_order`) refuses a set whose members cross inside 10-125 keV (e.g. skin with soft tissue).
- **Cache key.** The key holds the sorted table (names, nominal densities, formulas): every set has its own cache
  entry, any order of a set the same entry, and the default seven produce the key of version 0.4.0
  (`mglib_e552a2c1b51c1763c095.pt` for `SolverConfig()`), so existing caches are reused; the library is bit-identical
  (tested against the 0.4.0 keys and by building the default set by name and in reverse order).
- **Density model.** `attenuation_to_densities` / `attenuation_to_density` / `DensityModel` (already general in M)
  interpolate between whichever reference materials bracket a voxel; below the lowest reference material that reference material alone of reduced density,
  above the highest that reference material alone, denser (air and bone for the default set, as before).
- **By name, not index.** `utilities.wlab_phantom(grid, ..., materials=None)` and `wlab_setup(..., materials=None)`
  return rho [M, ...] for the given names (default the seven), with the phantom in the "water" and "bone" rows looked
  up by name; a set without water or bone raises `ValueError` ("the WLAB phantom needs 'water' and 'bone' ...").
  `synthesize_wlab_dataset` places the phantom in its library's rows (`materials=solver.library.materials`).
- Demos: `demo/demo_citrus.py --materials air,water,bone` (comma-separated catalogue names; default the seven) sets
  the reference materials of the loop's density model; the truth solver of the data keeps the default seven (the phantom is water
  and bone only, so every set runs on the same data and the data-set cache name does not change); the demo prints
  the reference materials with their nominal attenuations, the summary JSON records `materials`, the outputs of another set carry
  `_materials-<names>`. `demo/demo_scatter_solver.py --materials` (water and bone required). The memory benchmark
  reports the fine-map memory for the library's number of materials.
- Measured (quick setting `--views 36 --delta-voxel 1 --outer-iterations 10 --scatter-view-stride 1 --scatter-every 1`,
  same data `wlab_36v_1cm_short_v2`, relative RMSE whole prism / core): default seven 0.0704 / 0.0862, `--materials
  air,water,bone` 0.0696 / 0.0852 (FDK 0.309 / 0.345, no correction 0.289 / 0.355); the reduced set is slightly better
  here because the WLAB truth is water and bone only (its lung and air inserts are water of reduced density, which the
  reduced model reproduces as air / water mixtures, the seven-material model as lung / adipose and air / lung); scatter
  stage 5.0 s instead of 5.1 s per outer iteration.
- Tests: `tests/test_material_catalogue.py` (catalogue, name checks, the phantom by name, one library build per
  catalogue entry, the default set's 0.4.0 cache keys and bit-identical library, one cache entry per set, the reduced
  set's density model, solve_view on the reduced and the extended (seven plus titanium) sets against the default set,
  the data generator on a reduced set, the WLAB regression anchor of `tests/data/pre_quadrature_reference.json` with
  the default and the reduced real library).

## v0.4.0 — 2026-10-17

Plain names. Every abbreviated identifier, file and module name, configuration field, command-line flag and
user-facing dictionary key is spelled out (`fc` -> `first_collision`, `uc` -> `uncollided`, `mc` ->
`multiple_collision`, `si` -> `source_iteration`, `ld` -> `linear_discontinuous`, `sh` -> `spherical_harmonic`,
`ctx` -> `context` / `sweep_context`, `geom` -> `geometry`, `lib` -> `library`, `quad` -> `quadrature`, `om` ->
`directions`, `dev` -> `device`, `dt` -> `dtype`, `cfg` -> `config`, `acc` -> `accumulator`, `rec` -> `record`,
`idx` -> `index`, `tol` -> `tolerance`, `n_sub` -> `samples_per_pixel_axis`, `det` -> `detector`, `proj` ->
`projection`, module aliases -> the module names). **Breaking change** for every caller; **no change in behaviour**:
the computations, the defaults, the test tolerances and the regression anchors are unchanged, `check_cuda.sh` reports
every kernel matching, and the demo at its defaults gives CITRUS 0.0215 / 0.0259 as in version 0.3.0.

The documented equation symbols stay (rho, mu_t, mu_s, phi, psi, eta, omega, tau, X, Y, I0, P, S, SPR, Phi, T and
the dimension letters G, L, K, M, N, V, Rd, Cd, Rc, Cc, Sc, D8, Dc, q, ...); each is explained where it first appears.
Short loop indices, the names dictated by MBIRTORCH (`recon`, `init_recon`, `det_*` geometry fields, `fm_hessian`, ...),
PyTorch, NumPy and argparse, and the register-level temporaries inside the Triton kernel bodies are unchanged.

**On-disk formats are unchanged.** Data sets (`.npz` keys such as `si_iterations`, `phi_air`, the meta's
`solver_config`, which maps `source_iteration_tolerance` / `max_source_iterations` back to `si_tol` /
`si_max_iterations` through `utilities._solver_config_record`), the demo's summary JSON (`outer`, `k` in the
histories, `scatter_view_interp_order`, ...), the cross-section library cache (the key JSON keeps `de_kev` and `n_gl`,
so every cached `mglib_*.pt` is still found) and all cache and output file names read and write exactly as before.
The `MultigroupLibrary` fields (`mu_t`, `mu_pe`, `mu_s_l`, `mu_s_eta`, `fine_mu_t`, ...) are saved in the library
cache and keep their names; so do the benchmark record keys (`dev_triton_vs_torch`, `down_ms`, ...).

### File renames

| Old | New |
|---|---|
| `citrus/scatter_solver/ld_cell.py` | `citrus/scatter_solver/linear_discontinuous_cell.py` |
| `citrus/scatter_solver/kernels/triton_utils.py` | `citrus/scatter_solver/kernels/triton_utilities.py` |
| `tests/test_ld_cell.py` | `tests/test_linear_discontinuous_cell.py` |
| `tests/oracles/ld_slab_1d.py` | `tests/oracles/linear_discontinuous_slab_1d.py` (`ld_slab_sweep` -> `linear_discontinuous_slab_sweep`) |

### Dictionary keys

| Where | Old | New |
|---|---|---|
| `citrus_recon` / `mbir_recon` `info["records"]` (and the callback's `record`) | `k` | `outer_iteration` |
| | `t_scatter` | `scatter_seconds` |
| | `t_mbir` | `mbir_seconds` |
| | `t_total` | `total_seconds` |
| | `t_transfer` | `transfer_seconds` |
| | `t_primary` | `primary_seconds` |
| | `t_skipped` | `interpolation_seconds` |
| | `acc_change_mean` | `accumulator_change_mean` |
| | `acc_change_max` | `accumulator_change_max` |
| | `scatter_change_rel` | `scatter_change_relative` |
| | `gpu_peak_mib` | `gpu_peak_allocated_mib` |
| `ViewResult.timings` | `mc_source` | `multiple_collision_source` |
| `ViewResult.to_numpy_dict()` | `si_converged`, `si_history`, `si_num_iterations`, `si_phi_norms` | `source_iteration_converged`, `source_iteration_history`, `source_iteration_num_iterations`, `source_iteration_phi_norms` (`timing_mc_source` -> `timing_multiple_collision_source`) |
| cross-section backend `kind` (`sigma`, `pdf_raw`, `library_pdf_normalization`) | `"pe"`, `"cs"`, `"rs"` | `"photoelectric"`, `"compton"`, `"rayleigh"` |
| `DirectionSetContext` in-memory cache key | `("sh", L)` | `("spherical_harmonics", L)` |
| `ray_tracing_kernels._pixel_geometry_scalars` (internal) | `pix`, `bc` | `pixel`, `box_center` |

### Command-line flags

| Script | Old | New |
|---|---|---|
| `demo/demo_citrus.py` | `--outer` | `--outer-iterations` |
| | `--scatter-view-interp-order` | `--scatter-view-interpolation-order` |
| `benchmarks/bench_citrus_memory.py` | `--voxel` | `--delta-voxel` |
| | `--cell` | `--transport-cell-cm` |
| | `--stride` | `--scatter-view-stride` |
| | `--outer` | `--outer-iterations` |
| | `--memory-cap` | `--memory-cap-mib` |
| `benchmarks/profile_citrus_step.py` | `--voxel` | `--delta-voxel` |
| | `--cell` | `--transport-cell-cm` |
| | `--memory-cap` | `--memory-cap-mib` |
| `benchmarks/bench_transport_sweep.py` | `--mem-cap-mib` | `--memory-cap-mib` |
| | `--no-si` | `--no-source-iteration` |
| `benchmarks/bench_uncollided_kernel.py` | `--reps` | `--repeats` |

### Names, by module

Instance attributes follow the constructor parameters below (`ScatterSolver.library`, `.geometry`,
`.first_collision_chunk`, `.primary_samples_per_pixel_axis`, `.direction_context_cache_size`, and
`ScatterSolver.lib_transport` -> `transport_library`; `StateStore.pin` -> `StateStore.pinned`;
`ViewChunkProjector.geometry`; `_Footprint.ctx` / `.h` / `.h_center` -> `.context` / `.heights` / `.center_height`;
the private caches of `ScatterSolver` `_ctx_cache` / `_grid_cache` / `_cone_cache` / `_dir_ctx_cache` ->
`_sweep_contexts` / `_transport_grids` / `_cone_quadratures` / `_direction_contexts`). Module-internal aliases are
replaced by the module names (`_st`, `_tri`, `_tu`, `_fc_tri` -> `ray_tracing_kernels`, `sweep_kernels`,
`triton_utilities`, `uncollided_kernels`). Local variables are renamed throughout and are not listed.

#### `citrus.citrus`

| Kind | Old | New |
|---|---|---|
| function | `_fields` | `_material_fields_or_build` |
| function | `G1` | `uncollided_operator` |
| function | `G1to2` | `first_collision_operator` |
| function | `G_gt2` | `multiple_collision_operator` |
| function | `G_step` | `accumulator_update_operator` |
| constant | `_FP16_TARGET` | `_FLOAT16_TARGET` |
| constant | `SCATTER_VIEW_INTERP_ORDERS` | `SCATTER_VIEW_INTERPOLATION_ORDERS` |
| function | `_check_interp_order` | `_check_interpolation_order` |
| field | `CitrusState.Phi_gt1` | `CitrusState.collided_moments` |
| field | `CitrusState.cone_quad` | `CitrusState.cone_quadrature` |
| field | `DensityModel.lib` | `DensityModel.library` |
| field | `CitrusConfig.num_outer` | `CitrusConfig.num_outer_iterations` |
| field | `CitrusConfig.pin_limit_gib` | `CitrusConfig.pinned_memory_limit_gib` |
| field | `CitrusConfig.scatter_view_interp_order` | `CitrusConfig.scatter_view_interpolation_order` |
| parameter | `phi_uc` | `phi_uncollided` (in `first_collision_operator`, `accumulator_update_operator`, `project_scatter`) |
| parameter | `f` | `fields` (in `_check_accumulator`) |
| parameter | `Phi_gt1` | `collided_moments` (in `_check_accumulator`, `multiple_collision_operator`, `accumulator_update_operator`, `project_scatter`) |
| parameter | `quad` | `quadrature` (in `project_scatter`) |
| parameter | `lib` | `library` (in `DensityModel.from_library`) |
| parameter | `pin_limit_gib` | `pinned_memory_limit_gib` (in `StateStore`) |
| parameter | `i` | `slot` (in `StateStore.put`, `StateStore.get`) |
| parameter | `Phi` | `accumulator` (in `StateStore.put`) |
| parameter | `s_c` | `computed_scatter` (in `interpolate_scatter`) |
| parameter | `reg` | `regularization` (in `apply_regularization`) |
| parameter | `dev` | `device` (in `_sync`) |

#### `citrus.recon.mbir_bridge`

| Kind | Old | New |
|---|---|---|
| parameter | `geom` | `geometry` (in `make_cone_model`, `short_scan_weights`, `fdk`, `ViewChunkProjector`) |
| parameter | `tol` | `tolerance` (in `is_short_scan`) |

#### `citrus.scatter_solver.angular_quadrature`

| Kind | Old | New |
|---|---|---|
| function | `_as_vec3` | `_unit_vector` |
| function | `legendre_p` | `legendre_polynomials` |
| function | `num_sh` | `num_spherical_harmonics` |
| function | `sh_degree_index` | `spherical_harmonic_degree_index` |
| class | `SHBasis` | `SphericalHarmonicBasis` |
| function | `sh_basis` | `spherical_harmonic_basis` |
| field | `SphericalHarmonicBasis.Yw` | `SphericalHarmonicBasis.Y_weighted` |
| field | `SphericalHarmonicBasis.l_of_n` | `SphericalHarmonicBasis.degree_of_moment` |
| parameter | `quad` | `quadrature` (in `spherical_harmonic_basis`) |
| parameter | `mu_s_l_vol` | `mu_s_l_volume` (in `scatter_source_moments`) |
| parameter | `Y_sub` | `Y_selected` (in `scatter_source_at_directions`) |

#### `citrus.scatter_solver.config`

| Kind | Old | New |
|---|---|---|
| field | `SolverConfig.si_tol` | `SolverConfig.source_iteration_tolerance` |
| field | `SolverConfig.si_max_iterations` | `SolverConfig.max_source_iterations` |

#### `citrus.scatter_solver.cross_section_library`

| Kind | Old | New |
|---|---|---|
| function | `_table` | `_material_table` |
| constant | `_Q_BREAKS` | `_Q_BREAKPOINTS` |
| method | `_XrayPhysicsBackend.library_pdf_norm` | `_XrayPhysicsBackend.library_pdf_normalization` |
| method | `_Spectrum.cumint` | `_Spectrum.cumulative_integral` |
| function | `sphere_quadrature_q` | `eta_quadrature_in_q` |
| field | `BuildInfo.sigma_cs_group` | `BuildInfo.sigma_compton_group` |
| field | `BuildInfo.sigma_rs_group` | `BuildInfo.sigma_rayleigh_group` |
| field | `BuildInfo.sigma_pe_group` | `BuildInfo.sigma_photoelectric_group` |
| field | `BuildInfo.sigma_t_group` | `BuildInfo.sigma_total_group` |
| field | `BuildInfo.pdf_norm_max_rel_dev` | `BuildInfo.pdf_normalization_max_relative_deviation` |
| parameter | `de_kev` | `fine_bin_width_kev` (in `fine_energy_grid`, `compute_library`, `_cache_key`, `build_library`) |
| parameter | `breaks_q` | `q_breakpoints` (in `eta_quadrature_in_q`) |
| parameter | `n_gl` | `gauss_legendre_nodes` (in `eta_quadrature_in_q`, `compute_library`, `_cache_key`, `build_library`) |
| parameter | `b` | `backend` (in `_spectrum_or_default`) |

#### `citrus.scatter_solver.cross_sections`

| Kind | Old | New |
|---|---|---|
| parameter | `lib` | `library` (in `mu_t_volume`, `fine_group_weights`, `group_transmission`, `transfer_moments_volume`, `transfer_eta_at`, `transfer_eta_at_pairs`) |

#### `citrus.scatter_solver.detector`

| Kind | Old | New |
|---|---|---|
| constant | `RHO_PB` | `RHO_LEAD` |
| constant | `_EDGE_EPS` | `_EDGE_SPACING_KEV` |
| constant | `_PB_TABLE` | `_LEAD_TABLE` |
| function | `_loglog` | `_log_log_interpolation` |
| parameter | `mu_mass_csi` | `mass_attenuation` (in `csi_response`) |
| parameter | `mu_lead` | `mass_attenuation` (in `focused_1d_grid`) |
| parameter | `density_lead` | `lead_density` (in `focused_1d_grid`) |
| parameter | `geom` | `geometry` (in `detector_angle`, `grid_angles`) |

#### `citrus.scatter_solver.kernels.ray_tracing_kernels`

| Kind | Old | New |
|---|---|---|
| Triton kernel | `_down_shared_kernel` | `_downstream_shared_kernel` |
| function | `_pixel_geo` | `_pixel_geometry_scalars` |
| parameter | `vol_ptr` | `volume_ptr` (in `_trace`, `_ray_kernel`) |
| parameter | `vidx_ptr` | `volume_index_ptr` (in `_ray_kernel`) |
| parameter | `om_ptr` | `direction_ptr` (in `_ray_kernel`, `_downstream_shared_kernel`) |
| parameter | `HAS_VIDX` | `HAS_VOLUME_INDEX` (in `_ray_kernel`) |
| parameter | `tw_ptr` | `toward_ptr` (in `_downstream_shared_kernel`) |
| parameter | `vol` | `volume` (in `_launch_ray`) |
| parameter | `vidx` | `volume_index` (in `_launch_ray`) |
| parameter | `om` | `directions` (in `_launch_ray`) |
| parameter | `vol_k_stride` | `volume_stride` (in `_launch_ray`) |
| parameter | `geo` | `geometry_scalars` (in `_launch_ray`) |
| parameter | `geom` | `geometry` (in `_pixel_geometry_scalars`, `upstream_cumulative`, `primary_project`, `toward_flags`, `downstream_cumulative`, `oblique_project`) |
| parameter | `n_sub` | `samples_per_pixel_axis` (in `primary_project`) |

#### `citrus.scatter_solver.kernels.sweep_kernels`

| Kind | Old | New |
|---|---|---|
| Triton kernel | `_ld_plane_kernel` | `_linear_discontinuous_plane_kernel` |
| field | `FlatPlaneIndex.up` | `FlatPlaneIndex.upstream` |
| parameter | `up_cell` | `upstream_cell` (in `build_flat_index`) |
| parameter | `src_ptr` | `source_ptr` (in `_linear_discontinuous_plane_kernel`) |
| parameter | `oct_ptr` | `octant_ptr` (in `_linear_discontinuous_plane_kernel`) |
| parameter | `up_ptr` | `upstream_ptr` (in `_linear_discontinuous_plane_kernel`) |
| parameter | `src` | `source` (in `sweep_planes`) |

#### `citrus.scatter_solver.kernels.triton_utils` -> `citrus.scatter_solver.kernels.triton_utilities`

Module renamed; no names inside it changed.

#### `citrus.scatter_solver.kernels.uncollided_kernels`

| Kind | Old | New |
|---|---|---|
| Triton kernel | `_fc_kernel` | `_first_collision_kernel` |
| function | `fc_source` | `first_collision_source` |
| function | `fc_source_or_none` | `first_collision_source_or_none` |
| parameter | `idx_ptr` | `index_ptr` (in `_first_collision_kernel`) |
| parameter | `xax_ptr` | `x_axis_ptr` (in `_first_collision_kernel`) |
| parameter | `yax_ptr` | `y_axis_ptr` (in `_first_collision_kernel`) |
| parameter | `zax_ptr` | `z_axis_ptr` (in `_first_collision_kernel`) |
| parameter | `om_ptr` | `direction_ptr` (in `_first_collision_kernel`) |
| parameter | `tab_ptr` | `table_ptr` (in `_first_collision_kernel`) |
| parameter | `lib` | `library` (in `prepared_table`, `kernel_supports`, `first_collision_source`, `first_collision_source_or_none`) |
| parameter | `phi_uc` | `phi_uncollided` (in `kernel_supports`, `first_collision_source`, `first_collision_source_or_none`) |
| parameter | `geom` | `geometry` (in `first_collision_source`, `first_collision_source_or_none`) |

#### `citrus.scatter_solver.ld_cell` -> `citrus.scatter_solver.linear_discontinuous_cell`

| Kind | Old | New |
|---|---|---|
| function | `ld_cell_rhs` | `cell_right_hand_sides` |
| function | `ld_cell_solve` | `cell_solve` |
| function | `ld_cell_matrix` | `cell_matrix` |
| parameter | `nbr_psi0` | `neighbor_psi0` (in `cell_right_hand_sides`) |
| parameter | `nbr_psia` | `neighbor_psia` (in `cell_right_hand_sides`) |

#### `citrus.scatter_solver.ray_tracing`

| Kind | Old | New |
|---|---|---|
| method | `SiddonProjectors._toward` | `SiddonProjectors._toward_panel` |
| parameter | `geom` | `geometry` (in `Projectors.upstream_cumulative`, `Projectors.upstream_to_points`, `Projectors.primary_project`, `Projectors.downstream_cumulative`, `Projectors.oblique_project`, `SiddonProjectors._toward_panel`, `SiddonProjectors.upstream_cumulative`, `SiddonProjectors.upstream_to_points`, `SiddonProjectors.primary_project`, `SiddonProjectors.downstream_cumulative`, `SiddonProjectors.oblique_project`) |
| parameter | `vol_t` | `flat_volume` (in `_trace_chunk`, `_line_integrals_flat`) |
| parameter | `vol_offset` | `volume_offset` (in `_trace_chunk`) |
| parameter | `n_sub` | `samples_per_pixel_axis` (in `SiddonProjectors`, `SiddonProjectors.primary_project`) |
| parameter | `om` | `directions` (in `SiddonProjectors._toward_panel`) |

#### `citrus.scatter_solver.scatter_image`

| Kind | Old | New |
|---|---|---|
| type alias | `SourceFn` | `SourceFunction` |
| function | `_primary_n_sub` | `_primary_samples_per_pixel_axis` |
| field | `DirectionSetContext.quad` | `DirectionSetContext.quadrature` |
| field | `DirectionSetContext.geom` | `DirectionSetContext.geometry` |
| field | `DirectionSetContext.acc_weight` | `DirectionSetContext.accumulation_weight` |
| field | `DirectionSetContext.bounds` | `DirectionSetContext.chunk_bounds` |
| parameter | `geom` | `geometry` (in `covering_polar_angle`, `panel_direction_set`, `DirectionSetContext.check`, `make_direction_context`, `_resolve_context`, `scatter_image_from_source`, `panel_height_field`, `_Footprint`, `combined_scatter_image`, `primary_subpixel_geometry`, `primary_path_lengths`, `primary_from_path_lengths`, `primary_image`) |
| parameter | `quad` | `quadrature` (in `DirectionSetContext.check`, `make_direction_context`, `_resolve_context`, `scatter_image_from_source`, `combined_scatter_image`) |
| parameter | `grid_quad` | `grid_quadrature` (in `DirectionSetContext.subcell_offsets`, `_subcell_offsets`, `_check_subpoint_shapes`, `supervoxel_subpoint_emission`, `combined_scatter_image`) |
| parameter | `acc` | `accumulator` (in `DirectionSetContext.accumulate`, `DirectionSetContext.finalize`) |
| parameter | `img` | `image` (in `DirectionSetContext.accumulate`, `box_blur_rows_cols`, `_Footprint.blur`) |
| parameter | `i` | `chunk_index` (in `DirectionSetContext.accumulate`, `_Footprint.blur`) |
| parameter | `lib` | `library` (in `make_direction_context`, `_resolve_context`, `scatter_image_from_source`, `combined_scatter_image`, `primary_from_path_lengths`, `primary_image`) |
| parameter | `source_fn` | `source_function` (in `scatter_image_from_source`) |
| parameter | `src_quad` | `source_quadrature` (in `_check_subpoint_shapes`, `supervoxel_subpoint_emission`) |
| parameter | `width_c` | `width_channels` (in `box_blur_rows_cols`) |
| parameter | `width_r` | `width_rows` (in `box_blur_rows_cols`) |
| parameter | `ctx` | `context` (in `_Footprint`) |
| parameter | `e` | `emission` (in `_Footprint.mean_height`) |
| parameter | `hbar` | `mean_heights` (in `_Footprint.blur`) |
| parameter | `rho_quad` | `rho_quadrature` (in `combined_scatter_image`) |
| parameter | `phi_uc_quad` | `phi_uncollided` (in `combined_scatter_image`) |
| parameter | `T_mean` | `scatter_source_moments` (in `combined_scatter_image`) |
| parameter | `fc_chunk` | `first_collision_chunk` (in `combined_scatter_image`) |
| parameter | `n_sub` | `samples_per_pixel_axis` (in `primary_subpixel_geometry`, `_primary_samples_per_pixel_axis`, `primary_path_lengths`, `primary_image`) |

#### `citrus.scatter_solver.scatter_solver`

| Kind | Old | New |
|---|---|---|
| type alias | `SourceFn` | `SourceFunction` |
| property | `MaterialFields.fc_pairs` | `MaterialFields.first_collision_pairs` |
| method | `ScatterSolver.uncollided` | `ScatterSolver.uncollided_fluence` |
| method | `ScatterSolver.fc_moment_source` | `ScatterSolver.first_collision_moment_source` |
| method | `ScatterSolver.fc_sampling_correction` | `ScatterSolver.first_collision_sampling_correction` |
| method | `ScatterSolver.mc_source_moments` | `ScatterSolver.multiple_collision_source_moments` |
| field | `MaterialFields.grid_quad` | `MaterialFields.grid_quadrature` |
| field | `MaterialFields.rho_quad` | `MaterialFields.rho_quadrature` |
| field | `MaterialFields.pairs_quad` | `MaterialFields.pairs_quadrature` |
| field | `ViewResult.si` | `ViewResult.source_iteration_result` |
| parameter | `lib` | `library` (in `truncated_library`, `ScatterSolver`) |
| parameter | `geom` | `geometry` (in `ScatterSolver`) |
| parameter | `fc_chunk` | `first_collision_chunk` (in `ScatterSolver`) |
| parameter | `primary_n_sub` | `primary_samples_per_pixel_axis` (in `ScatterSolver`) |
| parameter | `direction_context_cache` | `direction_context_cache_size` (in `ScatterSolver`) |
| parameter | `quad` | `quadrature` (in `ScatterSolver.direction_context`, `ScatterSolver.scatter_image`) |
| parameter | `phi_uc` | `phi_uncollided` (in `ScatterSolver.first_collision_moment_source`, `ScatterSolver.first_collision_sampling_correction`, `ScatterSolver.scatter_image`) |
| parameter | `ctx` | `sweep_context` (in `ScatterSolver.first_collision_moment_source`, `ScatterSolver.first_collision_sampling_correction`, `ScatterSolver.scatter_moment_source`) |
| parameter | `source_fn` | `source_function` (in `ScatterSolver.transport_sweep`) |
| parameter | `fc_source_fn` | `first_collision_source_function` (in `ScatterSolver.run_source_iteration`) |
| parameter | `T_mean` | `scatter_source_moments` (in `ScatterSolver.scatter_image`) |

#### `citrus.scatter_solver.source`

| Kind | Old | New |
|---|---|---|
| parameter | `lib` | `library` (in `SourceModel.per_group_strength`) |
| parameter | `geom` | `geometry` (in `SourceModel.angular_weight`) |

#### `citrus.scatter_solver.source_iteration`

| Kind | Old | New |
|---|---|---|
| type alias | `SourceFn` | `SourceFunction` |
| class | `SIResult` | `SourceIterationResult` |
| function | `scatter_source_fn` | `scatter_source_function` |
| parameter | `mu_s_l_vol` | `mu_s_l_volume` (in `scatter_moments`) |
| parameter | `ctx` | `sweep_context` (in `scatter_source_function`, `source_iteration`) |
| parameter | `mu_s_l_vol_coarse` | `mu_s_l_volume_coarse` (in `multiple_collision_source_moments`, `source_iteration`) |
| parameter | `fc_source_fn` | `first_collision_source_function` (in `source_iteration`) |
| parameter | `tol` | `tolerance` (in `source_iteration`) |

#### `citrus.scatter_solver.transport_sweep`

| Kind | Old | New |
|---|---|---|
| type alias | `SourceFn` | `SourceFunction` |
| function | `source_fn_accepts_out` | `source_function_accepts_out` |
| field | `PlaneIndex.octant3` | `PlaneIndex.octant_column` |
| field | `PlaneIndex.up_cell` | `PlaneIndex.upstream_cell` |
| field | `SweepContext.quad` | `SweepContext.quadrature` |
| field | `SweepContext.perm` | `SweepContext.permutation` |
| field | `SweepContext.flat` | `SweepContext.flat_index` |
| parameter | `quad` | `quadrature` (in `make_sweep_context`) |
| parameter | `ctx` | `sweep_context` (in `_use_triton`, `_sweep_chunk`, `_resolve_chunk`, `_fill_source`, `sweep`) |
| parameter | `pl` | `plane` (in `_plane_step`) |
| parameter | `src` | `source` (in `_plane_step`, `_sweep_chunk`) |
| parameter | `source_fn` | `source_function` (in `source_function_accepts_out`, `_fill_source`, `sweep`) |

#### `citrus.scatter_solver.uncollided`

| Kind | Old | New |
|---|---|---|
| type alias | `PairsArg` | `PairsArgument` |
| function | `fc_source_at_directions` | `first_collision_source_at_directions` |
| function | `_fc_source_torch` | `_first_collision_source_torch` |
| function | `_fc_source_torch_pairs` | `_first_collision_source_torch_pairs` |
| function | `restrict_fc_to_ld_moments` | `restrict_to_linear_discontinuous_moments` |
| function | `fc_exact_moments` | `first_collision_exact_moments` |
| class | `FCSamplingCorrection` | `FirstCollisionSamplingCorrection` |
| function | `fc_sampling_correction` | `first_collision_sampling_correction` |
| constant | `_FC_SUM_MAX_DIRECTIONS` | `_FIRST_COLLISION_SUM_MAX_DIRECTIONS` |
| function | `fc_quadrature_sums` | `first_collision_quadrature_sums` |
| function | `fc_correction_factors` | `first_collision_correction_factors` |
| field | `FirstCollisionSamplingCorrection.om_uc` | `FirstCollisionSamplingCorrection.omega_uncollided` |
| parameter | `geom` | `geometry` (in `primary_directions_to_voxels`, `uncollided_fluence`, `uncollided_path_lengths`, `uncollided_from_path_lengths`, `first_collision_source_at_directions`, `_first_collision_source_torch`, `_first_collision_source_torch_pairs`, `first_collision_sampling_correction`, `first_collision_quadrature_sums`) |
| parameter | `lib` | `library` (in `uncollided_fluence`, `uncollided_from_path_lengths`, `first_collision_source_at_directions`, `_first_collision_source_torch`, `_first_collision_source_torch_pairs`, `first_collision_exact_moments`, `first_collision_sampling_correction`, `first_collision_quadrature_sums`) |
| parameter | `path` | `path_lengths` (in `uncollided_from_path_lengths`) |
| parameter | `phi_uc` | `phi_uncollided` (in `first_collision_source_at_directions`, `_first_collision_source_torch`, `_first_collision_source_torch_pairs`, `first_collision_exact_moments`, `first_collision_sampling_correction`, `first_collision_quadrature_sums`) |
| parameter | `fc` | `field` (in `restrict_to_linear_discontinuous_moments`) |
| parameter | `fc` | `source` (in `FirstCollisionSamplingCorrection.apply`) |
| parameter | `om_uc` | `omega_uncollided` (in `first_collision_quadrature_sums`) |

#### `citrus.utilities`

| Kind | Old | New |
|---|---|---|
| function | `_coeffs` | `_mass_attenuation_coefficients` |
| function | `_np` | `_to_numpy` |
| parameter | `lib` | `library` (in `effective_energy`, `mass_attenuation_at`, `nominal_densities`, `nominal_attenuation`, `material_order`, `_mass_attenuation_coefficients`, `densities_to_attenuation`, `attenuation_to_densities`, `attenuation_to_density`, `wlab_solver`, `synthesize_wlab_dataset`) |
| parameter | `geom` | `geometry` (in `wlab_solver`, `synthesize_wlab_dataset`) |
| parameter | `sl` | `image_slice` (in `central_row`) |

### Tests, benchmarks and demos

Fixtures `libs` -> `libraries`, `real_lib` -> `real_library`, `bk` -> `sweep_backend` (its methods `ctx` / `tol` ->
`context` / `tolerance`); helpers `use_fc_backend` -> `use_first_collision_backend`, `FC_BACKENDS` ->
`FIRST_COLLISION_BACKENDS`, `ORIGINAL_FC` -> `ORIGINAL_FIRST_COLLISION`, `_dense_ld_solve` ->
`_dense_linear_discontinuous_solve`, `_scipy_real_sh` -> `_scipy_real_spherical_harmonics`, `_fc_fn` ->
`_first_collision_function`, `_fc_kernel_or_skip` -> `_first_collision_kernel_or_skip`, `_fc_moment_sources` ->
`_first_collision_moment_sources`, `_small_fc_case` -> `_small_first_collision_case`, `_broadcast_mc_source` ->
`_broadcast_multiple_collision_source`, `_zero_mc` -> `_zero_multiple_collision`, `dir_source_fn` /
`const_source_fn` -> `direction_source_function` / `constant_source_function`, `TRITON_TOL` -> `TRITON_TOLERANCE`;
module aliases (`step1`, `step3`, `step1_triton`, `solver_mod`, `citrus_mod`, `citrus_hooks`, `_si`, `xb`,
`siddon_triton`, `sweep_triton`) -> the module names (`uncollided`, `scatter_image`, `uncollided_kernels`,
`scatter_solver`, `citrus`, `cross_section_library`, `ray_tracing_kernels`, `sweep_kernels`, with a `_module` suffix
where the plain name is taken).

## v0.3.0 — 2026-10-17

Densities from the attenuation one material at a time, MBIRTORCH primaries in the CITRUS loop, MBIR resumed from its
error sinogram, quadrature points by spacing. **Breaking changes** (configuration names); the loop's default primary
changes the demo result (below).

- **No seven-material fine volume in the loop.** `utilities.attenuation_to_density(lib, X, m)` evaluates ONE
  material's map of the interpolating model, bit for bit the row m of `attenuation_to_densities` (in place, two
  boolean masks and one weight as temporaries); `DensityModel.material(X, m)` evaluates it slab by slab
  (`DensityModel.slab_voxels`, default 4 M voxels) so the result is the only full-size map;
  `DensityModel.negative_fraction(X)`. `ScatterSolver.material_fields_from_attenuation(X, grid_fine, density_model)`
  builds the `MaterialFields` material by material (map -> resampled onto C and Q -> dropped; absent materials skipped,
  one host synchronization per material), equal to `material_fields(density_model(X), grid_fine)` (bit for bit on
  WLAB in the tests that compare them, float rounding in general). `MaterialFields.rho_fine` may be None; new fields
  `attenuation`, `density_model` (any `scatter_solver.MaterialDensityModel`: `.material(X, m)`), `fine_materials`
  (materials present) and `fine_material(m)`, `num_materials`. The "fine" uncollided trace and the primary trace such
  fields one map at a time. `citrus_recon` uses this path; `citrus_step(solver, state, rho, grid, ...)` and every
  rho-based path (`material_fields`, `solve_view`, the phantom, the tests) are unchanged and bit-identical.
  Peak allocated memory of a scatter stage on the 0.125 cm WLAB grid (12.3 M voxels, 1 cm cells, 180 views, stride
  8; `benchmarks/bench_citrus_memory.py`): 2521 / 2364 MiB -> 1891 / 1876 MiB (outer iterations 2 / 3); the density
  stage itself 1.56 GB transient + 0.33 GB kept -> 0.1 GB transient.
- **Two-step primary.** `scatter_image.primary_path_lengths(projectors, rho, grid, geom, angle, n_sub)` (per-material
  line integrals [M', Rd n, Cd n]) and `primary_from_path_lengths(lib, path, materials, geom, angle, source, detector,
  grid_response)` (fine-energy transmission, sub-pixel average, source, inverse square, detector and grid
  responses); `primary_subpixel_geometry`. `primary_image` is their composition and equals version 0.2.0 bit for bit
  (tested against a verbatim copy). `ScatterSolver.primary_path_lengths` / `primary_from_path_lengths`;
  `uncollided.uncollided_path_lengths` / `uncollided_from_path_lengths` likewise.
- **Primaries of all views in the loop.** `citrus.primary_signals(solver, fields, angles, projector,
  views_per_chunk=None)`: one projection per material present onto all views, in chunks of views whose path lengths
  stay within one sinogram (default max(1, V // (M' + 1)) views), then `primary_from_path_lengths` per view.
  Projectors: `recon.mbir_bridge.ViewChunkProjector` (MBIRTORCH's pixel-averaged separable-footprint line
  integrals; sub-models over the angles of a chunk, about 1 ms to build, cached) and `citrus.SiddonViewProjector`
  (the solver's Siddon rays, equal to `ScatterSolver.primary` bit for bit). New `CitrusConfig.primary_projector`
  ("mbirtorch", **default**, or "siddon") and `CitrusConfig.primary_views_per_chunk` (None). The loop computes every
  view's primary this way (the sparse-view path no longer traces the skipped views separately);
  `citrus_view_step(solver, state, fields, angle, primary=True)` is the step from prebuilt fields (the loop calls it
  with primary=False); records carry `t_primary`, info `primary_projector`.
  MBIRTORCH vs Siddon primary on the WLAB study (0.5 cm, 180 views, short scan, true densities): relative L2 0.9 %,
  RMS of the per-pixel relative difference 3.9 %, median 0.05 %, 99th percentile 3 %, maximum 5x (factor 6 in
  transmission) on the pixels whose ray grazes a face of the prism (a centre ray against a pixel average).
  The pixel-integrated primary is the physical model of a detector pixel, hence the default.
- **Synthetic data consistent with the loop's primary (data-set format 2).** `synthesize_cone_beam_data` /
  `synthesize_wlab_dataset(..., primary_projector="mbirtorch")`: the SPR's denominator is the solver's primary on
  MBIRTORCH's per-material line integrals of the true densities (`primary_signals(..., quantity="planar_fluence")`,
  new `quantity` argument); "siddon" keeps the definition of version 0.2.0. meta records `primary_projector` and
  `data_format` (2). Demo: `--truth-primary-projector`; cache and output names carry `_v2` (and `_primary-siddon`
  for the Siddon definition), so data sets of version 0.2.0 are never reused; the old data sets and outputs in
  `demo/input` and `demo/output` are renamed `..._v1...` (format 1: SPR with the Siddon primary). With the old
  definition the MBIRTORCH primary was inconsistent with the data on the silhouette pixels (RMS 0.8 % of the corrected
  line integrals at the true densities, up to 0.87 on single pixels).
  Demo defaults on the new data (`python demo/demo_citrus.py`, synthesis 113 s once, CITRUS 159 s): relative RMSE FDK
  0.318 / 0.348, no correction 0.292 / 0.355, CITRUS 0.0215 / 0.0259 (whole prism / core; version 0.2.0 on its
  format-1 data: 0.0242 / 0.0294); with `--primary-projector siddon` on the same data 0.0257 / 0.0305 (on the format-1
  data the MBIRTORCH loop gave 0.0270 / 0.0305 and the Siddon loop 0.0242 / 0.0294).
- **MBIR resume.** `recon.mbir_bridge.mbir_iterations(..., error_sinogram=None, hessian=None,
  return_checkpoint=False)`: with any of them it replicates MBIRTORCH 0.1.2's `recon` (`initialize_recon`, then
  `_vcd_recon` with `init_error_sinogram` / `fm_hessian` / `return_checkpoint`, which `recon` does not forward) and
  returns the checkpoint {'error_sinogram', 'fm_hessian'}. `CitrusConfig.resume_error_sinogram` (**default True**):
  the loop keeps -A X between outer iterations (one sinogram) and resumes from e = y_c - A X; equal to the restarted
  MBIR to 1.1e-6 (relative L2, demo after 80 iterations); MBIR iteration on the 12 M-voxel grid 0.63 / 1.02 s instead
  of 0.73 / 1.09 s; peak +18 MiB on the demo (one sinogram, 19 MiB). The Hessian is recomputed (keeping it would hold
  a reconstruction-sized volume).
- **Quadrature points by spacing.** `SolverConfig.fc_points_per_axis` is removed; `SolverConfig.
  quadrature_point_spacing_cm` (default 0.5): q = max(1, round(transport_cell_cm / spacing)), bounded as before by
  max(1, round(transport_cell_cm / largest fine voxel edge)). Unchanged for 1 cm cells (q = 2 at 0.5 cm voxels, 1 at
  1 cm); 2 cm cells now get q = 4 (was 2), 0.5 cm cells q = 1. Study (WLAB 0.5 cm, converged, AP / lateral, rel. L2
  of the scatter signal): 2 cm cells q = 1 / 2 against q = 4 1.1 % / 0.59 % and 0.40 % / 0.22 % (0.19 / 0.20 /
  0.35 s per view); 1 cm cells q = 1 against 2 0.25 % / 0.09 %; the cell size itself costs more (2 cm against 1 cm
  cells: 3.6 % / 9.5 %).
- **Renamed (plain words).** `SolverConfig.fc_trace_grid` -> `uncollided_trace_grid` (values unchanged); benchmark
  flags `--fc-points` -> `--quadrature-spacing` (cm), `--fc-trace` -> `--uncollided-trace`; the demo's summary JSON
  keys `fc_points_per_axis` -> `quadrature_points_per_axis` (plus `quadrature_point_spacing_cm`).
- Demo (`demo/demo_citrus.py`): `--primary-projector`, `--resume-error-sinogram / --no-resume-error-sinogram`,
  `--save-recon`; output suffixes `_siddon`, `_restart`; the summary records `primary_projector`,
  `resume_error_sinogram` and the CITRUS run's peak allocated GPU memory (1636 MiB at the defaults; 1624 MiB in
  version 0.2.0) and per-iteration times. `benchmarks/bench_citrus_memory.py` (peak memory and time per outer
  iteration, split into scatter stage and MBIR).
- Tests: `tests/test_attenuation_path.py` (per-material density bit for bit, fields from X against fields from rho,
  their peak memory, the two-step primary against version 0.2.0, MBIRTORCH against Siddon primaries, chunking,
  configuration), resume tests in `tests/test_mbir_bridge.py` and `tests/test_citrus_recon.py`, the MBIRTORCH
  primary in the loop, the spacing rule in `tests/test_quadrature_grid.py`.

## v0.2.0 — 2026-10-17

First-order quantities on a quadrature grid of the transport cells (Acuros CTS, Maslowski et al. 2018, Med Phys
45:1899, Section 2.B); transport cells of any size. **Breaking changes.**

- **Transport grid by cell size.** `SolverConfig.supervoxel_factor` is removed; `SolverConfig.transport_cell_cm`
  (default 1.0) sets the edge of the cubic transport cells. The transport grid C = `Grid.covering(fine_grid, cell)`
  covers the fine box, centred on it (ceil(extent / cell) cells per axis, symmetric empty padding): any ratio of voxel
  to cell size. `ScatterSolver.transport_grid`, `quadrature_grid`; `utilities.default_supervoxel_factor` is removed
  (`wlab_solver` / `synthesize_wlab_dataset` default to 1 cm cells, unchanged on the 1 and 0.5 cm WLAB grids). Demo
  flags `--supervoxel-factor` -> `--transport-cell-cm` (cache-name suffix `_cell<cm>`, run JSON `transport_cell_cm`).
- **Conservative resampling.** `geometry.resample_conservative` / `overlap_matrix` / `resampling_matrices`: exact
  cell averages of the voxel-wise constant fine field (separable per-axis overlap matrices, cached), mass conserving for
  any ratio; equal to the block mean for integer ratios. `scatter_solver.coarsen_density(rho, grid_fine, grid_coarse)`
  and `scatter_image.coarsen_mu_t(mu, grid_fine, grid_coarse)` use it (new signatures).
- **Quadrature grid Q = C.refine(q)**, `Grid.refine`. `SolverConfig.fc_points_per_axis` (default 2) is an upper
  bound: q = min(fc_points_per_axis, max(1, round(transport_cell_cm / largest fine voxel edge))) (Python's round;
  `SolverConfig.quadrature_points`, `ScatterSolver.quadrature_points`), so 1 cm voxels in 1 cm cells keep q = 1 and the
  cost of version 0.1.0 (CITRUS step about 0.12 s per view), 0.5 cm and smaller voxels get q = 2. The uncollided
  fluence, the first-collision source at the sweep and cone directions, its sampling correction and restriction, and the
  first-order emission, footprint and projection of the scatter image all live on Q; only the primary image and the
  resampling touch the fine grid. `MaterialFields`: `grid_quad`, `q`, `rho_quad`, `pairs_quad` (None -> dense),
  `fc_pairs`; `factor`, `mu_t_fine` and `pairs_fine` are removed. `ScatterSolver.uncollided` returns phi_uc on Q.
- **Uncollided trace.** `SolverConfig.fc_trace_grid`: "quadrature" (default, through Q), "coarse" (through C: smears
  a boundary inside a cell, up to 4 % of the lateral scatter when the cells are offset half a cell from the object) or
  "fine" (through the reconstruction grid), all at about the same cost; `uncollided_fluence(..., points=)`, `Projectors.upstream_to_points`. With
  q = cell / voxel and the fine trace the solver reproduces version 0.1.0 to float rounding
  (`tests/data/pre_quadrature_reference.json`: 8 x 8 block means of the signals; `tests/test_quadrature_grid.py`).
- **Dense first-collision kernel for any M.** Resampled densities can mix more than two materials per point: the Triton
  kernel's dense mode now takes all M materials (compile-time M; 2.7x the pair-mode time per point at M = 7);
  `cross_sections.material_pairs_if_compact`, `max_materials_per_point`; `pairs=` accepts a `MaterialPairs`,
  `uncollided.DENSE` or None (decide from rho).
- `combined_scatter_image(lib, rho_quad, phi_uc_quad, grid_quad, q, ..., grid_coarse=None)`,
  `supervoxel_subpoint_emission(src_quad, ..., grid_quad, q)`, `DirectionSetContext.subcell_offsets(grid_quad, q)`.
- Cost: a CITRUS step is about 0.31 s per view at 0.5, 0.25 and 0.125 cm voxels with 1 cm cells (version 0.1.0:
  0.32 s, 1.76 s, out of memory). `benchmarks/profile_citrus_step.py --voxel --cell --fc-points --fc-trace`.

## v0.1.0 — 2026-10-16

Seven-material density model; short-scan data by default. **Breaking changes; results change.**

- **Densities are `[7, ...]`.** The basis is the seven-material table `cross_sections.MATERIAL_TABLE` (air 0.0012,
  lung 0.26, adipose 0.95, water 1.0, skeletal muscle 1.05, soft tissue 1.06, cortical bone 1.85 g/cm^3, in order of
  increasing attenuation at the nominal density; `MATERIALS`, `NOMINAL_DENSITIES`, `WATER = 3`, `BONE = 6`).
  `wlab_phantom` / `wlab_setup` return `[7, R, C, S]` (the water and bone rows); `citrus_step`, the per-view operators,
  `ScatterSolver.solve_view` / `material_fields`, `synthesize_cone_beam_data` (`rho_true [7, ...]`,
  `mass_attenuation [7]`, meta `materials`, `nominal_density`, `mass_attenuation` by name) take and return seven
  materials. `material_fields` checks the number of materials against the library.
- **The two-material split is gone.** `utilities.attenuation_to_densities(lib, X, energy_kev=None)` is the
  interpolating seven-material model: the two materials whose nominal attenuations bracket X get (1 - f) rho_a and
  f rho_b, f = (X - mu_a) / (mu_b - mu_a); below air air alone, above bone bone alone (X / m); X < 0 -> 0. Exact in
  attenuation, pure material at its nominal attenuation (a cortical-bone voxel is bone at 1.85 g/cm^3, not unit water
  plus 1.156 g/cm^3 bone), at most two materials per voxel. New helpers `nominal_densities`, `nominal_attenuation`,
  `material_order`. `DensityModel` keeps its API and uses the new model. On the 180-view WLAB study (20 outer
  iterations): relative RMSE 0.0389 / 0.0451 (prism / core) instead of 0.0429 / 0.0505, water +0.6 % instead of
  -1.3 %, the bone insert +0.6 % instead of +6.3 %; scatter stage 62.6 s instead of 56-58 s per outer iteration.
- **Library.** `MultigroupLibrary.materials`, `nominal_density [M]`, `num_materials`; `cross_section_library`:
  `MATERIAL_FORMULAS` (seven entries), `default_material_table`, `check_material_order` (the build refuses a table whose
  nominal attenuations cross at any library energy), `compute_library` / `build_library` / `build_solver_library`
  take `materials=` (name, nominal density, formula) for other tables of any size. **Cache key change**: build version
  `citrus-xsec-2`, the key includes the material table; the two-material caches of 0.0.x are never loaded (the
  seven-material library builds once, about 16 s). `synthetic_library(num_materials=)` (2 default, 7 = the package
  table with the two base materials in the water and bone rows, any other M).
- **Pair representation.** `cross_sections.MaterialPairs` / `material_pairs` (index / density `[2, ...]`),
  `transfer_eta_at_pairs`; the first-collision Triton kernel gains a PAIR mode (two table rows gathered by material
  index; equals the PyTorch pair path to 1.4e-7 of the field maximum), the PyTorch pair path, `MaterialFields.pairs_fine`
  (derived once per density field), `pairs=` in `fc_source_at_directions`, `fc_sampling_correction`,
  `fc_quadrature_sums`, `combined_scatter_image`; `uncollided.present_materials` / `group_transmission(materials=)`:
  the uncollided fluence and the primary trace only the materials present. On WLAB (water and bone only) the
  seven-material solve equals the two-material solve bit for bit in float32. CITRUS step on the 0.5 cm WLAB problem:
  335 ms per view instead of 317 ms (same session).
- **Short scans.** `utilities.fan_angle(geometry)` (in-plane fan angle of the panel, 2 atan(0.5 Nc dc / SDD), 22.6
  degrees for the full-field WLAB panel) and `utilities.view_angles(num_views, geometry, start_view_angle=0.0,
  end_view_angle=None)` (evenly spaced over [start, end), endpoint excluded; default end = start + pi + fan angle).
  `synthesize_wlab_dataset(..., start_view_angle=0.0, end_view_angle=None)` replaces `angle_offset`: **the default data
  set is now a short scan**; pass `end_view_angle=2 pi` for a full rotation. Demo flags `--start-view-angle` /
  `--end-view-angle` (degrees); `--full` keeps the report's 180 views over a full rotation; cached data sets of a
  short scan carry `_short` in their name.
- **FDK of a short scan.** MBIRTORCH's FDK has no short-scan weighting; `recon.mbir_bridge.fdk(..., geom=, angles=)`
  applies Parker weights (`short_scan_weights`, `scan_range`, `is_short_scan`) and `citrus_recon` / `mbir_recon` use
  it for the initialization. Sparse-view interpolation runs along the arc for a short scan (`citrus.full_rotation`,
  `interpolation_weights(..., circular=None)`, `angular_neighbors(..., circular=)`).
- Demo (36 views, 1 cm grid, 10 outer iterations): relative RMSE 0.064 (short scan, default) and 0.075 (full
  rotation; 0.0751 in 0.0.3).
- Tests: `tests/test_materials.py` (library build and order check, cache keys, pair representation, pair kernel vs
  PyTorch, seven vs two materials on WLAB with the synthetic and the real library, density-model round trip and
  bracketing, the WLAB bone voxel, the fixed-SPR reproduction); short-scan tests in `test_utilities.py`,
  `test_mbir_bridge.py`, `test_sparse_views.py`, `test_citrus_recon.py`; the WLAB tests use the seven-material
  synthetic library (`num_materials=7`).

## v0.0.3 — 2026-10-16

Scatter step every n outer iterations.

- `CitrusConfig.scatter_every` (default 1): the scatter stage runs at the outer iterations 0, n, 2n, ... only and
  the previous scatter-to-primary ratio is reused in between, so n MBIR iterations run per scatter step (the
  accumulators keep their age). Validated in `CitrusConfig.validate`; every record of `citrus_recon` now carries
  `scatter_refreshed`.
- Demo flag `--scatter-every` (`demo/demo_citrus.py`); the summary reports `scatter_steps`.
- Measured on the 180-view WLAB study: the same 20 scatter steps give a relative RMSE of 0.0429 at 20 outer
  iterations (n = 1), 0.0310 at 40 (n = 2) and 0.0286 at 80 (n = 4) in the same wall time; at 40 iterations n = 2 is
  within 0.0008 of n = 1 (0.0302) at half the scatter cost. At a fixed 20 outer iterations skipping stages costs
  accuracy (0.0513 at n = 2, 0.0700 at n = 4). At equal scatter cost CITRUS at every iteration (0.0429) beats a
  converged scatter solve at every 2nd iteration (0.0501).
- Tests: configuration checks in `tests/test_sparse_views.py`, a CUDA run with `scatter_every=2` in
  `tests/test_citrus_recon.py`.

## v0.0.2 — 2026-10-15

Sparse-view scatter with angular interpolation.

- `CitrusConfig.scatter_view_stride` (default 1): the CITRUS scatter step runs on every n-th view only (views 0, n,
  2n, ...; only those keep an accumulator, 1/n of the state memory); the other views' scatter signal is interpolated
  in angle, per pixel, from the computed views, their primary comes from the solver's primary projection, and the
  correction ratio is formed on every view.
- `CitrusConfig.scatter_view_interp_order` (default 3), all circular in angle: 3 cubic Catmull-Rom on 4 neighbors, 2
  quadratic (3-point Lagrange), 1 linear, 0 nearest computed view, None no interpolation (diagnostics: skipped views
  uncorrected). Validated in `CitrusConfig.validate`.
- `citrus.citrus`: `scatter_view_indices`, `angular_neighbors`, `interpolation_weights`, `interpolate_scatter`;
  `citrus_recon`'s info gains `computed_views` and `view_interpolated`, the records `t_skipped`. `mbir_recon` is
  unchanged.
- Demo flags `--scatter-view-stride`, `--scatter-view-interp-order` (`demo/demo_citrus.py`; default stride 1).
- Measured on the 180-view WLAB study (20 outer iterations): stride 2 +0.0014 relative RMSE for 51 % of the scatter
  time, stride 4 +0.008 for 26 %.
- Tests: `tests/test_sparse_views.py` (interpolators, configuration) and CUDA runs at strides 2 and 4 in
  `tests/test_citrus_recon.py`.

## v0.0.1 — 2026-10-14

First release.

- `citrus.citrus`: the CITRUS reconstruction loop (`citrus_recon`): FDK initialization, then per outer iteration one
  scatter step per view (age-ordered source iteration with a per-view collided-fluence accumulator, stored compactly)
  and one warm-started MBIRTORCH iteration on the scatter-corrected data; `mbir_recon` (the same loop without the
  solver); the per-view operators (`citrus_step`, `G1`, `G1to2`, `G_gt2`, `G_step`, `project_scatter`).
- `citrus.scatter_solver`: the LBTE source iteration scatter solver (`ScatterSolver`): multigroup cross sections from
  XrayPhysics (EPDL97) with an exact-in-energy uncollided fluence and primary, discrete ordinates with real spherical
  harmonics, linear-discontinuous finite elements on supervoxels with a wavefront sweep, source iteration with a
  conservative first-collision source, collapsed-cone scatter image with sub-voxel points and a finite-cone footprint,
  detector and anti-scatter-grid responses; exact Siddon ray tracing. Fused Triton kernels on float32 CUDA with
  PyTorch reference paths for CPU / float64.
- `citrus.recon`: the MBIRTORCH bridge; `citrus.utilities`: WLAB phantom and geometry, attenuation <-> density
  model, synthetic cone-beam data with scatter (`synthesize_cone_beam_data`, `synthesize_wlab_dataset`), plotting
  helpers.
- Demos (`demo/demo_scatter_solver.py`, `demo/demo_citrus.py` with a configurable ground-truth discretization), test
  suite, Sphinx documentation, developer shell scripts (`dev_scripts/`) and benchmarks (`benchmarks/`).
- Dependencies: numpy, scipy, torch, matplotlib, mbirtorch; extras `test`, `docs`, `physics` (XrayPhysics), `gpu`
  (Triton).
