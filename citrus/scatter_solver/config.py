"""Solver configuration: every discretization parameter of the LBTE source iteration solver."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence, Union

from .cross_sections import COMPOUND_NAMES, MATERIAL_CATALOGUE, MATERIALS, check_material_names

# the rules of SolverConfig.group_edges_method and the node placements of SolverConfig.angular_table_grid
GROUP_EDGES_METHODS = ("scattered_fluence", "equal_fluence", "log")
ANGULAR_TABLE_GRIDS = ("adaptive", "uniform")


@dataclass(frozen=True)
class SolverConfig:
    """Discretization of the scatter solver (docs/source/method.rst).

    Energy
      num_groups        number of energy groups G
      group_edges_kev   explicit descending group edges (num_groups + 1 values, keV); None: the edges follow
                        group_edges_method over [energy_min_kev, kVp]
      group_edges_method  rule of the group edges when group_edges_kev is None
                        (`cross_section_library.make_group_edges`):
                        "scattered_fluence" (default): equal area per group under an estimate of the TOTAL photon
                          fluence spectrum inside the object, source plus scattered photons (a one-dimensional
                          slowing-down balance in water with a leakage loss, `scattered_fluence_spectrum`). The groups
                          must resolve the downscattered photons, which carry the multiple-collision scatter: at MV
                          most of them lie far below the source photons;
                        "equal_fluence": equal area under the SOURCE spectrum (equal source photon count per group,
                          the rule of citrus <= 0.5);
                        "log": geometric spacing between energy_min_kev and the spectrum maximum
      grouping_object_size_cm  characteristic chord length L (cm) of the object, used only by the
                        "scattered_fluence" rule: photons leave the balance at the rate 1/L (escape), so L -> 0
                        recovers "equal_fluence" and L -> infinity the spectrum of an infinite medium. Default 20 cm
                        (a patient-sized object; mean chord about 4 V / S). `citrus.citrus_recon` (CitrusConfig.
                        grouping_object_size "auto") replaces it by the mean chord of the FDK volume before the library
                        is built
      grouping_medium   the medium of that balance (used only by "scattered_fluence"): "auto" (default) is measured on
                        the object, the mean densities of the reference materials over it (`citrus.citrus_recon` from
                        the FDK volume, `citrus.utilities.synthesize_cone_beam_data` from the true object, before the
                        library is built), and water at 1 g/cm^3 when nothing measured it; a catalogue name (that
                        material at its nominal density; e.g. an object of one known material) or a mapping
                        {reference material name: density g/cm^3} sets it. Named materials must be among `materials`.
                        Stored as a tuple of (name, density) pairs sorted by name; the cache key and the balance use
                        the densities rounded to 3 significant digits
                        (`cross_section_library.grouping_medium_densities`)
      energy_min_kev    lowest group edge (keV); Compton photons scattered below it are absorbed
      legendre_order    Legendre order L of the scattering kernel (spherical harmonics up to degree L)
      num_eta_table     nodes N of the tabulated per-steradian transfer in eta = cos(scattering angle)
      angular_table_grid  placement of those nodes (`cross_section_library.compute_library`):
                        "adaptive" (default): non-uniform in eta, dense where the transfer curves bend (the Rayleigh
                          and Compton forward peaks), so that linear interpolation has about the same relative error
                          everywhere; looked up through a small map table (`cross_sections.angular_table_position`);
                        "uniform": equally spaced in eta (the table of citrus <= 0.5)

    Materials
      materials         names of the REFERENCE materials of the density model, from `cross_sections.MATERIAL_CATALOGUE`
                        (at least two, any order): the library is built for exactly this set (`build_solver_library`,
                        `utilities.wlab_solver`; sorted by nominal attenuation, its cache key holds the set), and the
                        density model interpolates between the two reference materials whose nominal attenuations
                        bracket a voxel (`citrus.utilities.attenuation_to_densities`). Default: the seven
                        `cross_sections.MATERIALS` (air, lung, adipose, water, muscle, soft_tissue, bone)

    Angle (sweep quadrature: Gauss-Legendre in cos(theta) x equally spaced azimuths)
      num_polar         polar nodes (>= L + 1)
      num_azimuthal     azimuths (>= 2 L + 1)

    Space (docs/source/conventions.rst, "Grids")
      transport_cell_cm  edge length (cm) of the cubic transport cells: the transport grid C is the regular grid of this
                        spacing that covers the fine (reconstruction) box, centred on it (ceil(extent / cell) cells per
                        axis; the padding is empty space). Any ratio to the fine voxel size is allowed
      quadrature_point_spacing_cm target spacing (cm) of the quadrature points of the first-collision source and the
                        first-order scatter image: q^3 equal-weight points per transport cell on the midpoint lattice
                        (the quadrature grid Q = C refined q times) with q = max(1, round(transport_cell_cm /
                        quadrature_point_spacing_cm)), bounded by the data resolution, q = min(q, max(1, round(
                        transport_cell_cm / largest fine voxel edge))), Python's round (halves to even: a ratio of 2.5
                        gives 2), `quadrature_points`. Default 0.5 cm: 1 cm cells -> q = 2, 2 cm cells -> 4, 0.5 cm
                        cells -> 1; 1 cm voxels in 1 cm cells -> 1 (the data bound)
      uncollided_trace_grid  grid through which the uncollided fluence at the Q points is ray traced: "quadrature"
                        (default: Q itself with rho_quadrature, material boundaries resolved at the Q spacing),
                        "coarse" (the transport grid C with the resampled densities: smears a boundary inside a cell,
                        which matters where rays run along it) or "fine" (the reconstruction
                        grid; for validation); the cost is about the same for all three

    Source iteration
      source_iteration_tolerance  stop when ||phi_n|| / ||phi_total|| < source_iteration_tolerance (scalar-flux cell
                        means; phi_n the n-times-collided fluence, phi_total their sum)
      max_source_iterations  maximum number of sweeps (collision orders)

    Scatter image (collapsed cone)
      cone_num_polar, cone_num_azimuthal  product direction set over the cone that covers the panel
      direction_chunk   directions per batch in the scatter image, and directions per octant per sweep chunk

    Numerics
      dtype             "float32" (GPU production) or "float64" (reference / tests)
    """

    # --- energy -----------------------------------------------------------------------------
    num_groups: int = 6
    group_edges_kev: Sequence[float] | None = None
    group_edges_method: Literal["scattered_fluence", "equal_fluence", "log"] = "scattered_fluence"
    grouping_object_size_cm: float = 20.0
    grouping_medium: Union[str, tuple[tuple[str, float], ...]] = "auto"
    energy_min_kev: float = 10.0
    legendre_order: int = 7
    num_eta_table: int = 2001
    angular_table_grid: Literal["adaptive", "uniform"] = "adaptive"

    # --- materials ----------------------------------------------------------------------------
    materials: tuple[str, ...] = MATERIALS

    # --- angle --------------------------------------------------------------------------------
    num_polar: int = 8
    num_azimuthal: int = 16

    # --- space --------------------------------------------------------------------------------
    transport_cell_cm: float = 1.0
    quadrature_point_spacing_cm: float = 0.5
    uncollided_trace_grid: Literal["coarse", "quadrature", "fine"] = "quadrature"

    # --- source iteration ---------------------------------------------------------------------
    source_iteration_tolerance: float = 1.0e-3
    max_source_iterations: int = 30

    # --- scatter image ------------------------------------------------------------------------
    cone_num_polar: int = 16
    cone_num_azimuthal: int = 32
    direction_chunk: int = 8

    # --- numerics -----------------------------------------------------------------------------
    dtype: Literal["float32", "float64"] = "float32"

    def __post_init__(self) -> None:
        if not isinstance(self.materials, str) and isinstance(self.materials, Sequence):
            object.__setattr__(self, "materials", tuple(self.materials))  # hashable, compares equal to a tuple
        medium = self.grouping_medium
        if isinstance(medium, Mapping):
            medium = medium.items()
        if not isinstance(medium, str):
            try:  # a mapping or (name, density) pairs -> sorted pairs (hashable, order independent)
                medium = tuple(sorted((str(name), float(density)) for name, density in medium))
            except (TypeError, ValueError):
                pass  # left as given: validate() reports it
            object.__setattr__(self, "grouping_medium", medium)

    def to_dict(self) -> dict:
        return asdict(self)

    def quadrature_points(self, fine_voxel_cm: float) -> int:
        """Effective quadrature points per axis per transport cell for fine voxels whose largest edge is fine_voxel_cm:
        q = min(max(1, round(transport_cell_cm / quadrature_point_spacing_cm)), max(1, round(transport_cell_cm /
        fine_voxel_cm))) (Python's round, halves to even): the points are about quadrature_point_spacing_cm apart, and
        their spacing does not go much below the fine voxel size. With the default 0.5 cm spacing: 1 cm cells give 2
        (0.5 cm and smaller voxels; 1 cm voxels: 1; 0.4 cm: round(2.5) = 2), 2 cm cells give 4 (voxels of 0.5 cm and
        less), 0.5 cm cells give 1."""
        cell = float(self.transport_cell_cm)
        target = max(1, round(cell / float(self.quadrature_point_spacing_cm)))
        return min(target, max(1, round(cell / float(fine_voxel_cm))))

    def _validate_grouping_medium(self) -> None:
        medium = self.grouping_medium
        if isinstance(medium, str):
            if medium == "auto":
                return
            names = (medium,)
        else:
            try:
                names = tuple(name for name, _ in medium)
                densities = [float(density) for _, density in medium]
            except (TypeError, ValueError):
                raise ValueError(f"grouping_medium must be 'auto', a catalogue name or a mapping "
                                 f"{{reference material name: density}}, got {medium!r}") from None
            if any(not d >= 0 for d in densities) or not any(d > 0 for d in densities):
                raise ValueError(f"grouping_medium densities must be >= 0 with at least one > 0, got {medium!r}")
        unknown = [n for n in names if n not in MATERIAL_CATALOGUE]
        if unknown:
            raise ValueError(f"unknown grouping_medium material(s) {unknown}; the catalogue has {COMPOUND_NAMES} "
                             f"and every element by its English name")
        outside = [n for n in names if n not in tuple(self.materials)]
        if outside:
            raise ValueError(f"grouping_medium material(s) {outside} are not among the reference materials "
                             f"{tuple(self.materials)}")

    def validate(self) -> None:
        L = self.legendre_order
        if L < 0:
            raise ValueError("legendre_order must be >= 0")
        if self.num_polar < L + 1:
            raise ValueError(f"num_polar={self.num_polar} must be >= legendre_order+1={L + 1}")
        if self.num_azimuthal < 2 * L + 1:
            raise ValueError(f"num_azimuthal={self.num_azimuthal} must be >= 2*legendre_order+1={2 * L + 1}")
        if self.num_groups < 1:
            raise ValueError("num_groups must be >= 1")
        if self.group_edges_kev is not None and len(self.group_edges_kev) != self.num_groups + 1:
            raise ValueError("group_edges_kev must have num_groups+1 entries")
        if self.group_edges_method not in GROUP_EDGES_METHODS:
            raise ValueError(f"group_edges_method must be one of {GROUP_EDGES_METHODS}, "
                             f"got {self.group_edges_method!r}")
        if not self.grouping_object_size_cm > 0:
            raise ValueError(f"grouping_object_size_cm must be > 0, got {self.grouping_object_size_cm}")
        self._validate_grouping_medium()
        if self.num_eta_table < 2:
            raise ValueError(f"num_eta_table must be >= 2, got {self.num_eta_table}")
        if self.angular_table_grid not in ANGULAR_TABLE_GRIDS:
            raise ValueError(f"angular_table_grid must be one of {ANGULAR_TABLE_GRIDS}, "
                             f"got {self.angular_table_grid!r}")
        check_material_names(self.materials)
        if not self.transport_cell_cm > 0:
            raise ValueError(f"transport_cell_cm must be > 0, got {self.transport_cell_cm}")
        if not self.quadrature_point_spacing_cm > 0:
            raise ValueError(f"quadrature_point_spacing_cm must be > 0, got {self.quadrature_point_spacing_cm}")
        if self.uncollided_trace_grid not in ("coarse", "quadrature", "fine"):
            raise ValueError(f"uncollided_trace_grid must be 'coarse', 'quadrature' or 'fine', "
                             f"got {self.uncollided_trace_grid!r}")
        if self.max_source_iterations < 1 or self.source_iteration_tolerance <= 0:
            raise ValueError("need max_source_iterations >= 1 and source_iteration_tolerance > 0")
        if self.cone_num_polar < 1 or self.cone_num_azimuthal < 1:
            raise ValueError("need cone_num_polar >= 1 and cone_num_azimuthal >= 1")
        if self.direction_chunk < 1:
            raise ValueError("direction_chunk must be >= 1")
        if self.dtype not in ("float32", "float64"):
            raise ValueError(f"dtype must be 'float32' or 'float64', got {self.dtype!r}")
