"""Synthesize the data set of demo 2 at another voxel pitch (the same scan, panel, spectrum and materials).

Usage::

    python benchmarks/make_demo_data.py --delta-voxel 0.125      # demo/input/wlab_180v_0p125cm_0-202p6deg_125kvp
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from citrus.scatter_solver import (ConeGeometry, ScatterSolver, SolverConfig, ideal_counting_response,  # noqa: E402
                                   load_spectrum_txt, no_grid)
from citrus.utilities import WlabPhantom, save_dataset, synthesize_cone_beam_data, wlab_grid  # noqa: E402

DEMO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delta-voxel", type=float, default=0.125)
    parser.add_argument("--views", type=int, default=180)
    parser.add_argument("--I0", type=float, default=1.0e5)
    args = parser.parse_args()
    # the parameters of demo 2
    num_det_rows, num_det_channels, pixel_pitch = 184, 150, 0.4
    geometry = ConeGeometry(source_iso_dist=100.0, source_detector_dist=150.0, num_det_rows=num_det_rows,
                            num_det_channels=num_det_channels, delta_det_row=pixel_pitch,
                            delta_det_channel=pixel_pitch)
    start_angle = 0.0
    fan_angle = 2 * np.degrees(np.arctan(num_det_channels * pixel_pitch / 2 / 150.0))
    end_angle = start_angle + 180.0 + fan_angle
    angles = np.radians(np.linspace(start_angle, end_angle, args.views, endpoint=False))
    spectrum = load_spectrum_txt(os.path.join(DEMO, "spectra", "spectrum_125kvp.txt"))
    config = SolverConfig(materials=("air", "lung_icrp", "adipose_icrp", "water", "muscle_icru44",
                                     "soft_tissue_icru44", "bone_cortical_icrp"))
    grid = wlab_grid(args.delta_voxel)
    name = f"wlab_{args.views}v_{args.delta_voxel:g}cm_{start_angle:g}-{end_angle:.1f}deg_125kvp".replace(".", "p")
    path = os.path.join(DEMO, "input", name)
    solver = ScatterSolver(None, config, geometry, detector=ideal_counting_response(), grid_response=no_grid(),
                           device="cuda", spectrum=spectrum)
    phantom = WlabPhantom(grid, solver.materials, device="cuda")
    data = synthesize_cone_beam_data(phantom, grid, geometry, angles, solver, I0=args.I0)
    save_dataset(data, path)
    print(f"wrote {path} ({tuple(grid.shape)} voxels)")


if __name__ == "__main__":
    main()
