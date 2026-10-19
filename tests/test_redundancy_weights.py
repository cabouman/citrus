"""Redundancy weights of the direct reconstructions (``recon.mbir_bridge.redundancy_weights``, a port of the weights
of cabouman/mbirtorch PR 30 extended to the parallel beam): every line through the object counts once in the FDK of a
circular scan and the filtered back projection of a parallel scan over short scans and over ranges that are not whole
rotations; whole rotations and helical scans get none; an MBIRTORCH that weights by itself is left to do so."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from citrus.recon.mbir_bridge import (direct_reconstruction, fdk, forward_project, make_cone_model, make_model,
                                      redundancy_weight_function, redundancy_weights)
from citrus.scatter_solver.geometry import ConeGeometry, Grid, ParallelGeometry
from citrus.utilities import helical_scan

pytest.importorskip("mbirtorch")


def _views(fraction_of_rotation: float, count: int) -> np.ndarray:
    return np.arange(count) * fraction_of_rotation * 2 * math.pi / count


@pytest.mark.parametrize("arc", [1.3 * math.pi, 1.7 * math.pi, 2.6 * math.pi, 3.5 * math.pi])
def test_each_line_counts_once(arc):
    """A parallel line at beta is measured at beta + k pi; with the weights of every measurement summed it counts
    once (the weights alone, without the filter factor, sum to 1 over the measurements of a line)."""
    beta = np.linspace(0.0, arc, 200001)[:-1]
    w = redundancy_weight_function(beta, 0.0, arc)
    for start in np.linspace(0.01, math.pi - 0.01, 37):
        positions = start + math.pi * np.arange(int(arc // math.pi) + 2)
        positions = positions[positions < arc]
        total = sum(float(np.interp(p, beta, w)) for p in positions)
        assert total == pytest.approx(1.0, abs=2e-3), (arc, start, total)


def test_whole_rotations_and_helical_scans_get_no_weights():
    geometry = ConeGeometry(100.0, 150.0, 8, 32, 1.0, 1.0)
    assert redundancy_weights(geometry, _views(1.0, 90)) is None
    assert redundancy_weights(geometry, _views(2.0, 180)) is None
    assert redundancy_weights(helical_scan(geometry, _views(1.5, 90), pitch_cm=4.0), _views(1.5, 90)) is None
    parallel = ParallelGeometry(30.0, 8, 32, 1.0, 1.0)
    assert redundancy_weights(parallel, _views(0.5, 90)) is not None  # pi: weights of ones
    assert torch.allclose(redundancy_weights(parallel, _views(0.5, 90)), torch.ones(1))
    with pytest.warns(RuntimeWarning, match="shorter than 180 deg plus the fan angle"):
        assert redundancy_weights(parallel, _views(0.4, 90)) is None


def _two_disks(grid: Grid) -> tuple[torch.Tensor, torch.Tensor]:
    """An asymmetric object (two off-centre disks of different attenuation) and its core away from the edges."""
    x, y, _ = grid.axis_coordinates()
    r1 = (x[None, :] - 2.5) ** 2 + (y[:, None] - 0.5) ** 2
    r2 = (x[None, :] + 3.0) ** 2 + (y[:, None] + 2.0) ** 2
    volume = (0.2 * (r1 < 16).float() + 0.4 * (r2 < 4).float())[..., None].expand(grid.shape).contiguous()
    core = ((r1 < 9) | (r2 < 1.5))[..., None].expand(grid.shape)
    return volume, core


def _error(X: torch.Tensor, volume: torch.Tensor, core: torch.Tensor) -> float:
    X = X.cpu()
    return float((X[core] - volume[core]).norm() / volume[core].norm())


def test_parallel_filtered_back_projection():
    """Over 0.7 and 1.25 rotations the weighted FBP of an asymmetric object is as accurate as over a whole rotation,
    and the unweighted one is clearly worse (uneven coverage shades the object)."""
    grid = Grid((32, 32, 4), 0.5)
    geometry = ParallelGeometry.from_grid(grid, 30.0, 48, 0.5)
    volume, core = _two_disks(grid)

    def errors(fraction):
        angles = _views(fraction, int(round(180 * fraction)))
        model = make_model(grid, geometry, angles, device="cpu")
        sinogram = forward_project(model, volume)
        weighted = direct_reconstruction(model, sinogram, geometry=geometry, angles=angles)
        return _error(weighted, volume, core), _error(model.recon_fbp(sinogram, output_sharded=True), volume, core)

    floor = errors(1.0)[0]
    for fraction in (0.7, 1.25):
        weighted, unweighted = errors(fraction)
        assert weighted < 1.05 * floor and unweighted > 1.3 * floor, (fraction, weighted, unweighted, floor)


def test_cone_fdk_short_scan_and_overscan():
    """A circular cone-beam scan over 0.6 rotations (short scan) and 2.3 rotations (overscan beyond two rotations):
    the weighted FDK of an asymmetric object is as accurate as over a whole rotation, the unweighted one worse."""
    grid = Grid((32, 32, 4), 0.5)
    geometry = ConeGeometry(200.0, 300.0, 6, 64, 0.75, 0.75)
    volume, core = _two_disks(grid)

    def errors(fraction):
        angles = _views(fraction, int(round(120 * fraction)))
        model = make_cone_model(grid, geometry, angles, device="cpu")
        sinogram = forward_project(model, volume)
        weighted = fdk(model, sinogram, geometry=geometry, angles=angles)
        return _error(weighted, volume, core), _error(model.recon_fdk(sinogram, output_sharded=True), volume, core)

    floor = errors(1.0)[0]
    for fraction, worse in ((0.6, 1.5), (2.3, 1.05)):
        weighted, unweighted = errors(fraction)
        assert weighted < 1.05 * floor and unweighted > worse * floor, (fraction, weighted, unweighted, floor)


def test_mbirtorch_that_weights_by_itself_is_left_to_it():
    """An MBIRTORCH whose recon_fdk takes ``redundancy_weighting`` gets the plain sinogram (no double weighting) and,
    for a helical scan, ``redundancy_weighting=False``."""

    class WeightingModel:
        calls: list = []

        def recon_fdk(self, sinogram, filter_name="ramp", output_sharded=False, redundancy_weighting=True):
            self.calls.append((torch.as_tensor(sinogram).clone(), redundancy_weighting))
            return torch.zeros(1)

    geometry = ConeGeometry(100.0, 150.0, 4, 16, 1.0, 1.0)
    angles = _views(0.7, 60)
    sinogram = torch.ones(60, 4, 16)
    model = WeightingModel()
    fdk(model, sinogram, geometry=geometry, angles=angles)
    passed, weighting = model.calls[-1]
    assert torch.equal(passed, sinogram) and weighting is True
    fdk(model, sinogram, geometry=helical_scan(geometry, angles, pitch_cm=3.0), angles=angles)
    assert model.calls[-1][1] is False
    assert redundancy_weights(replace(geometry, use_curved_detector=True), angles) is not None
