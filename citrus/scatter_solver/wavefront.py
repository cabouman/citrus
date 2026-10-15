"""Wavefront (diagonal-plane) ordering for discrete-ordinates sweeps on a cuboid grid.

For an octant with signs (s_x, s_y, s_z) a cell (i=row, j=col, k=slice) depends only on its upstream
neighbors along the three axes, so cells on the plane

    p = (j if s_x > 0 else C-1-j) + (i if s_y > 0 else R-1-i) + (k if s_z > 0 else S-1-k)

depend only on planes < p. All 8 octants are swept together: for each plane we store the
(octant, cell) entries of every octant, and the sweep processes planes in order with one batched
gather/scatter per plane. Octant code = (s_x<0)*4 + (s_y<0)*2 + (s_z<0) (matches angular_quadrature.Quadrature).

Flat cell index n = (i * C + j) * S + k, i.e. tensor.reshape(-1) of a [rows, cols, slices] volume.
Axis order for neighbors: 0 -> x (cols), 1 -> y (rows), 2 -> z (slices).
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
    """Precomputed sweep ordering for one grid shape.

    shape:          (rows, cols, slices)
    num_planes:     rows + cols + slices - 2
    plane_offsets:  [num_planes + 1] int64; entries of plane p are [offsets[p], offsets[p+1])
    octant:         [E] int64 octant code of each entry (E = 8 * num_cells)
    cell:           [E] int64 flat cell index of each entry
    upstream:       [E, 3] int64 flat index of the upstream neighbor along x, y, z, or -1 for vacuum
    """

    shape: tuple[int, int, int]
    num_planes: int
    plane_offsets: Tensor
    octant: Tensor
    cell: Tensor
    upstream: Tensor

    @property
    def num_cells(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    def to(self, device) -> "SweepPlan":
        return SweepPlan(
            self.shape, self.num_planes, self.plane_offsets.to(device),
            self.octant.to(device), self.cell.to(device), self.upstream.to(device),
        )

    def plane_entries(self, p: int) -> slice:
        return slice(int(self.plane_offsets[p]), int(self.plane_offsets[p + 1]))


def build_sweep_plan(shape: tuple[int, int, int], device=None) -> SweepPlan:
    """Build the wavefront ordering for all 8 octants of a [rows, cols, slices] grid."""
    R, C, S = shape
    ii, jj, kk = torch.meshgrid(torch.arange(R), torch.arange(C), torch.arange(S), indexing="ij")
    ii, jj, kk = ii.reshape(-1), jj.reshape(-1), kk.reshape(-1)
    flat = (ii * C + jj) * S + kk
    n = flat.numel()

    oct_list, cell_list, up_list, plane_list = [], [], [], []
    for o in range(8):
        sx, sy, sz = (int(v) for v in OCTANT_SIGNS[o])
        px = jj if sx > 0 else (C - 1 - jj)
        py = ii if sy > 0 else (R - 1 - ii)
        pz = kk if sz > 0 else (S - 1 - kk)
        plane = px + py + pz
        # upstream neighbors
        jn = jj - sx
        in_ = ii - sy
        kn = kk - sz
        ux = torch.where((jn >= 0) & (jn < C), (ii * C + jn) * S + kk, torch.full_like(flat, -1))
        uy = torch.where((in_ >= 0) & (in_ < R), (in_ * C + jj) * S + kk, torch.full_like(flat, -1))
        uz = torch.where((kn >= 0) & (kn < S), (ii * C + jj) * S + kn, torch.full_like(flat, -1))
        oct_list.append(torch.full((n,), o, dtype=torch.int64))
        cell_list.append(flat)
        up_list.append(torch.stack([ux, uy, uz], dim=1))
        plane_list.append(plane)

    octant = torch.cat(oct_list)
    cell = torch.cat(cell_list)
    upstream = torch.cat(up_list)
    plane = torch.cat(plane_list)
    order = torch.argsort(plane * 8 + octant, stable=True)  # planes ascending, octants grouped within a plane
    octant, cell, upstream, plane = octant[order], cell[order], upstream[order], plane[order]
    num_planes = R + C + S - 2
    counts = torch.bincount(plane, minlength=num_planes)
    offsets = torch.zeros(num_planes + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(counts, dim=0)
    plan = SweepPlan(shape=(R, C, S), num_planes=num_planes, plane_offsets=offsets, octant=octant, cell=cell, upstream=upstream)
    return plan.to(device) if device is not None else plan
