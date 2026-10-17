"""Solver configuration: every discretization parameter of the LBTE source iteration solver."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence


@dataclass(frozen=True)
class SolverConfig:
    """Discretization of the scatter solver (docs: "Method" and "Accuracy and cost").

    Energy
      num_groups        number of energy groups G (equal photon count of the source spectrum per group)
      group_edges_kev   explicit descending group edges (num_groups + 1 values, keV); None: equal-fluence groups
                        spanning [energy_min_kev, kVp]
      energy_min_kev    lowest group edge (keV); Compton photons scattered below it are absorbed
      legendre_order    Legendre order L of the scattering kernel (spherical harmonics up to degree L)
      num_eta_table     points of the tabulated per-steradian transfer in eta = cos(scattering angle)

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
                        up to a few percent of the scatter where rays run along it) or "fine" (the reconstruction
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
    energy_min_kev: float = 10.0
    legendre_order: int = 7
    num_eta_table: int = 2001

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
