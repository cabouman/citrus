import torch

from citrus.scatter_solver.wavefront import build_sweep_plan, OCTANT_SIGNS


def test_plan_shapes_and_counts():
    plan = build_sweep_plan((3, 4, 5))
    assert plan.num_planes == 3 + 4 + 5 - 2
    assert plan.octant.numel() == 8 * 60 and plan.cell.numel() == 8 * 60
    assert int(plan.plane_offsets[-1]) == 8 * 60
    # every (octant, cell) pair appears exactly once
    key = plan.octant * 60 + plan.cell
    assert torch.equal(torch.sort(key).values, torch.arange(8 * 60))


def test_upstream_neighbors_are_on_earlier_planes():
    R, C, S = 3, 4, 5
    plan = build_sweep_plan((R, C, S))
    # plane id of every (octant, cell) entry
    plane_of = torch.empty(8, R * C * S, dtype=torch.int64)
    for p in range(plan.num_planes):
        sl = plan.plane_entries(p)
        plane_of[plan.octant[sl], plan.cell[sl]] = p
    for e in range(plan.cell.numel()):
        o, n = int(plan.octant[e]), int(plan.cell[e])
        for a in range(3):
            up = int(plan.upstream[e, a])
            if up >= 0:
                assert plane_of[o, up] == plane_of[o, n] - 1
    # neighbor correctness for one octant: octant 0 (all positive), cell (i=1, j=2, k=3)
    n = (1 * C + 2) * S + 3
    e = torch.nonzero((plan.octant == 0) & (plan.cell == n)).item()
    assert int(plan.upstream[e, 0]) == (1 * C + 1) * S + 3  # x neighbor: j-1
    assert int(plan.upstream[e, 1]) == (0 * C + 2) * S + 3  # y neighbor: i-1
    assert int(plan.upstream[e, 2]) == (1 * C + 2) * S + 2  # z neighbor: k-1
    # octant 7 (all negative), cell (0, 0, 0) has neighbors j+1, i+1, k+1
    e = torch.nonzero((plan.octant == 7) & (plan.cell == 0)).item()
    assert int(plan.upstream[e, 0]) == 1 * S and int(plan.upstream[e, 1]) == C * S and int(plan.upstream[e, 2]) == 1
    # vacuum at the domain boundary
    e = torch.nonzero((plan.octant == 0) & (plan.cell == 0)).item()
    assert (plan.upstream[e] == -1).all()


def test_octant_signs_match_code():
    for o in range(8):
        sx, sy, sz = OCTANT_SIGNS[o].tolist()
        assert (sx < 0) == bool(o & 4) and (sy < 0) == bool(o & 2) and (sz < 0) == bool(o & 1)


def test_plan_to_device(device):
    plan = build_sweep_plan((2, 2, 2), device=device)
    assert plan.cell.device.type == device.type
