"""Solver configuration: every discretization parameter of the LBTE source iteration solver.

The module holds the frozen dataclass ``SolverConfig`` and the allowed values of its group-edge rule and of the node
placement of its angular table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence, Union

from .cross_sections import (COMPOUND_NAMES, MATERIAL_CATALOGUE, MATERIALS, Material, _canonical_material,
                             check_material_names, material_names)

# the rules of SolverConfig.group_edges_method and the node placements of SolverConfig.angular_table_grid
GROUP_EDGES_METHODS = ("scattered_fluence", "equal_fluence", "log")
ANGULAR_TABLE_GRIDS = ("adaptive", "uniform")


@dataclass(frozen=True)
class SolverConfig:
    """Discretization parameters of the LBTE scatter solver (docs/source/method.rst).

    A ``grouping_medium`` given as a mapping or as pairs is stored as (name, density) pairs sorted by name, and
    ``materials`` as a tuple in which a ``Material`` that copies a catalogue entry becomes the catalogue name.

    Attributes:
        num_groups (int): number of energy groups G. Defaults to 6.
        group_edges_kev (Sequence[float] or None): explicit descending group edges in keV, num_groups + 1 values.
            Defaults to None, the edges of ``group_edges_method``.
        group_edges_method (str): rule of the group edges, "scattered_fluence", "equal_fluence" or "log". Defaults to
            "scattered_fluence".
        grouping_object_size_cm (float): characteristic size L in cm of the object for the "scattered_fluence"
            rule. Defaults to 20.0.
        grouping_medium (str or tuple[tuple[str, float], ...]): medium of the "scattered_fluence" rule: "auto", a
            catalogue or reference material name or (reference material name, density in g/cm^3) pairs. Defaults to
            "auto".
        energy_min_kev (float): lowest group edge in keV; photons scattered below it are absorbed. Defaults to 10.0.
        legendre_order (int): Legendre order L of the scattering kernel. Defaults to 7.
        num_eta_table (int): nodes N of the tabulated transfer in ``eta = cos(scattering angle)``. Defaults to 2001.
        angular_table_grid (str): placement of the table nodes, "adaptive" or "uniform". Defaults to "adaptive".
        materials (tuple[str or Material, ...]): reference materials of the library and the density model, catalogue
            names or user-defined ``cross_sections.Material`` entries, at least two. Defaults to the seven
            ``cross_sections.MATERIALS``.
        num_polar (int): Gauss-Legendre polar nodes of the sweep quadrature, at least L + 1. Defaults to 8.
        num_azimuthal (int): equally spaced azimuths of the sweep quadrature, at least 2 L + 1. Defaults to 16.
        transport_cell_cm (float): edge in cm of the cubic transport cells. Defaults to 1.0.
        quadrature_point_spacing_cm (float): target spacing in cm of the quadrature points (``quadrature_points``).
            Defaults to 0.5.
        uncollided_trace_grid (str): grid through which the uncollided fluence is traced, "quadrature", "coarse" or
            "fine". Defaults to "quadrature".
        source_iteration_tolerance (float): stop when ``||phi_n|| / ||phi_total||`` of the scalar-flux cell means
            falls below it. Defaults to 1.0e-3.
        max_source_iterations (int): maximum number of sweeps (collision orders). Defaults to 30.
        cone_num_polar (int): polar directions of the cone direction set of the scatter image. Defaults to 16.
        cone_num_azimuthal (int): azimuthal directions of the cone direction set of the scatter image. Defaults to 32.
        direction_chunk (int): directions per octant per sweep chunk and per batch of the scatter image (with the
            fused first-collision kernels the image takes larger batches, as many as fit a memory budget). Defaults
            to 8.
        dtype (str): "float32" (production) or "float64" (reference and tests). Defaults to "float32".
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
    materials: tuple[Union[str, Material], ...] = MATERIALS

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
        """Store ``materials`` as a tuple and a ``grouping_medium`` mapping or pairs as sorted (name, density) pairs."""
        if not isinstance(self.materials, str) and isinstance(self.materials, Sequence):
            # hashable, compares equal to a tuple; a copy of a catalogue entry is its name
            object.__setattr__(self, "materials", tuple(_canonical_material(m) for m in self.materials))
        medium = self.grouping_medium
        if isinstance(medium, Material):
            medium = medium.name
            object.__setattr__(self, "grouping_medium", medium)
        if isinstance(medium, Mapping):
            medium = medium.items()
        if not isinstance(medium, str):
            try:  # a mapping or (name, density) pairs -> sorted pairs (hashable, order independent)
                medium = tuple(sorted((name.name if isinstance(name, Material) else str(name), float(density))
                                      for name, density in medium))
            except (TypeError, ValueError):
                pass  # left as given: validate() reports it
            object.__setattr__(self, "grouping_medium", medium)

    def to_dict(self) -> dict:
        """Field values of the configuration.

        Returns:
            dict: ``dataclasses.asdict`` of the configuration.
        """
        return asdict(self)

    def quadrature_points(self, fine_voxel_cm: float) -> int:
        """Quadrature points per axis per transport cell for fine voxels of a given largest edge.

        ``q = min(max(1, round(cell / spacing)), max(1, round(cell / fine_voxel_cm)))`` with Python's round (halves to
        even), so the points are about ``quadrature_point_spacing_cm`` apart but not much closer than the fine voxels.

        Args:
            fine_voxel_cm (float): largest edge in cm of the fine voxels.

        Returns:
            int: number q of points per axis.
        """
        cell = float(self.transport_cell_cm)
        target = max(1, round(cell / float(self.quadrature_point_spacing_cm)))
        return min(target, max(1, round(cell / float(fine_voxel_cm))))

    def _validate_grouping_medium(self) -> None:
        """Check that ``grouping_medium`` is "auto", a catalogue or reference material name or densities of reference
        materials.

        Raises:
            ValueError: if it is not, or if its densities are negative or all zero.
        """
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
                raise ValueError(f"grouping_medium must be 'auto', a material name or a mapping "
                                 f"{{reference material name: density}}, got {medium!r}") from None
            if any(not d >= 0 for d in densities) or not any(d > 0 for d in densities):
                raise ValueError(f"grouping_medium densities must be >= 0 with at least one > 0, got {medium!r}")
        reference = material_names(self.materials) if isinstance(self.materials, Sequence) else ()
        unknown = [n for n in names if n not in MATERIAL_CATALOGUE and n not in reference]
        if unknown:
            raise ValueError(f"unknown grouping_medium material(s) {unknown}; the catalogue has {COMPOUND_NAMES} "
                             f"and every element by its English name, or define it in materials")
        outside = [n for n in names if n not in reference]
        if outside:
            raise ValueError(f"grouping_medium material(s) {outside} are not among the reference materials "
                             f"{reference}")

    def validate(self) -> None:
        """Check every field against its allowed values.

        Raises:
            ValueError: for the first field outside its allowed values.
        """
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
