"""Wavefront (diagonal-plane) ordering for discrete-ordinates sweeps on a cuboid grid.

The module holds the signs of the 8 octants and ``SweepPlan``, which lists for every diagonal plane the (octant, cell)
entries of all octants with their upstream neighbors, so that the sweep processes the planes in order with one batched
gather and scatter per plane.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


OCTANT_SIGNS = torch.tensor(
    [[+1 if not (o & 4) else -1, +1 if not (o & 2) else -1, +1 if not (o & 1) else -1] for o in range(8)],
    dtype=torch.int64,
)  # [8, 3] signs (s_x, s_y, s_z) per octant code


@dataclass(frozen=True)
class SweepPlan:
    """Precomputed sweep ordering of all 8 octants for one grid shape.

    Attributes:
        shape (tuple[int, int, int]): (rows, cols, slices) of the grid.
        num_planes (int): number of diagonal planes, rows + cols + slices - 2.
        plane_offsets (Tensor): [num_planes + 1] int64 offsets; the entries of plane p are ``offsets[p]`` to
            ``offsets[p + 1] - 1``.
        octant (Tensor): [E] int64 octant code of each entry, ``E = 8 * num_cells``.
        cell (Tensor): [E] int64 flat cell index ``(i * C + j) * S + k`` (row i, col j, slice k) of each entry.
        upstream (Tensor): [E, 3] int64 flat index of the upstream neighbor along x (cols), y (rows) and z (slices),
            or -1 for vacuum.
    """

    shape: tuple[int, int, int]
    num_planes: int
    plane_offsets: Tensor
    octant: Tensor
    cell: Tensor
    upstream: Tensor

    @property
    def num_cells(self) -> int:
        """Number of cells rows x cols x slices of the grid."""
        return self.shape[0] * self.shape[1] * self.shape[2]

    def to(self, device) -> "SweepPlan":
        """Copy of the plan with its index tensors on a device.

        Args:
            device (torch.device): target device.

        Returns:
            SweepPlan: the plan on ``device``.
        """
        return SweepPlan(
            self.shape, self.num_planes, self.plane_offsets.to(device),
            self.octant.to(device), self.cell.to(device), self.upstream.to(device),
        )

    def plane_entries(self, p: int) -> slice:
        """Entries of one diagonal plane.

        Args:
            p (int): plane index.

        Returns:
            slice: the slice of the entry tensors that holds plane p.
        """
        return slice(int(self.plane_offsets[p]), int(self.plane_offsets[p + 1]))


def build_sweep_plan(shape: tuple[int, int, int], device=None) -> SweepPlan:
    """Build the wavefront ordering of all 8 octants of a grid, the entries of a plane grouped by octant.

    Args:
        shape (tuple[int, int, int]): (rows, cols, slices) of the grid.
        device (torch.device, optional): device of the index tensors. Defaults to None, the CPU.

    Returns:
        SweepPlan: the ordering.
    """
    R, C, S = shape
    row_index, col_index, slice_index = torch.meshgrid(torch.arange(R), torch.arange(C), torch.arange(S), indexing="ij")
    row_index, col_index, slice_index = row_index.reshape(-1), col_index.reshape(-1), slice_index.reshape(-1)
    flat = (row_index * C + col_index) * S + slice_index
    n = flat.numel()

    octant_list, cell_list, upstream_list, plane_list = [], [], [], []
    for o in range(8):
        sx, sy, sz = (int(v) for v in OCTANT_SIGNS[o])
        plane_x = col_index if sx > 0 else (C - 1 - col_index)
        plane_y = row_index if sy > 0 else (R - 1 - row_index)
        plane_z = slice_index if sz > 0 else (S - 1 - slice_index)
        plane = plane_x + plane_y + plane_z
        # upstream neighbors
        neighbor_col = col_index - sx
        neighbor_row = row_index - sy
        neighbor_slice = slice_index - sz
        upstream_x = torch.where((neighbor_col >= 0) & (neighbor_col < C),
                                 (row_index * C + neighbor_col) * S + slice_index, torch.full_like(flat, -1))
        upstream_y = torch.where((neighbor_row >= 0) & (neighbor_row < R),
                                 (neighbor_row * C + col_index) * S + slice_index, torch.full_like(flat, -1))
        upstream_z = torch.where((neighbor_slice >= 0) & (neighbor_slice < S),
                                 (row_index * C + col_index) * S + neighbor_slice, torch.full_like(flat, -1))
        octant_list.append(torch.full((n,), o, dtype=torch.int64))
        cell_list.append(flat)
        upstream_list.append(torch.stack([upstream_x, upstream_y, upstream_z], dim=1))
        plane_list.append(plane)

    octant = torch.cat(octant_list)
    cell = torch.cat(cell_list)
    upstream = torch.cat(upstream_list)
    plane = torch.cat(plane_list)
    order = torch.argsort(plane * 8 + octant, stable=True)  # planes ascending, octants grouped within a plane
    octant, cell, upstream, plane = octant[order], cell[order], upstream[order], plane[order]
    num_planes = R + C + S - 2
    counts = torch.bincount(plane, minlength=num_planes)
    offsets = torch.zeros(num_planes + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(counts, dim=0)
    plan = SweepPlan(shape=(R, C, S), num_planes=num_planes, plane_offsets=offsets, octant=octant, cell=cell,
                     upstream=upstream)
    return plan.to(device) if device is not None else plan
