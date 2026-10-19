"""Time the scatter image of one view on the WLAB phantom for several direction-chunk sizes.

Builds the fields, the uncollided fluence and converged multiple-collision moments of a few views once, then times
``combined_scatter_image`` with direction contexts of different chunk sizes and reports the largest relative
difference of the images from those of the first chunk size.

Usage::

    python benchmarks/bench_scatter_image.py [--views 4] [--chunks 8,16,32,64] [--repeat 3] [--delta-voxel 0.5]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from citrus.citrus import CitrusState, citrus_step  # noqa: E402
from citrus.scatter_solver.config import SolverConfig  # noqa: E402
from citrus.scatter_solver.cross_section_library import load_spectrum_txt  # noqa: E402
from citrus.scatter_solver.scatter_image import combined_scatter_image, make_direction_context  # noqa: E402
from citrus.utilities import (attenuation_to_densities, densities_to_attenuation, full_field_wlab_geometry,  # noqa: E402
                              wlab_grid, wlab_phantom, wlab_solver)

SPECTRUM = load_spectrum_txt(os.path.join(ROOT, "demo", "spectra", "spectrum_125kvp.txt"))
DEVICE = torch.device("cuda")


def main() -> None:
    """Build the inputs of a few views, then time the scatter image per chunk size."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--chunks", default="8,16,32,64")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--delta-voxel", type=float, default=0.5)
    args = parser.parse_args()
    geometry = full_field_wlab_geometry()
    grid = wlab_grid(args.delta_voxel)
    solver = wlab_solver(args.delta_voxel, config=SolverConfig(), geometry=geometry, device=DEVICE, spectrum=SPECTRUM)
    library = solver.library
    rho = attenuation_to_densities(library, densities_to_attenuation(library, wlab_phantom(grid, device=DEVICE)))
    fields = solver.material_fields(rho, grid)
    inputs = []
    for a in 2 * np.pi * np.arange(args.views) / 180:
        a = float(a)
        state = CitrusState()
        for _ in range(3):
            _, _, state = citrus_step(solver, state, rho, grid, a, fields)
        phi = solver.uncollided_fluence(fields, a)
        moments = solver.multiple_collision_source_moments(fields, state.collided_moments)
        inputs.append((a, phi, moments, solver.cone_quadrature(grid, a)))
    reference = {}
    for chunk in (int(c) for c in args.chunks.split(",")):
        contexts = [make_direction_context(q, geometry, a, solver.detector, solver.grid_response, library, DEVICE,
                                           torch.float32, chunk=chunk) for a, _, _, q in inputs]
        times = []
        for r in range(args.repeat + 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            images = [combined_scatter_image(library, fields.rho_quadrature, phi, fields.grid_quadrature, fields.q,
                                             moments, solver.L, geometry, a, q, fields.mu_t_coarse, solver.projectors,
                                             solver.detector, solver.grid_response, context,
                                             pairs=fields.first_collision_pairs, grid_coarse=fields.grid_coarse)
                      ["signal"] for (a, phi, moments, q), context in zip(inputs, contexts)]
            torch.cuda.synchronize()
            if r:  # the first pass compiles
                times.append((time.perf_counter() - t0) / len(inputs))
        if not reference:
            reference = {i: image for i, image in enumerate(images)}
        deviation = max(float((image - reference[i]).abs().max() / reference[i].abs().max())
                        for i, image in enumerate(images))
        peak = torch.cuda.max_memory_allocated(DEVICE) / 2 ** 20
        print(f"chunk {chunk:3d}: {1e3 * min(times):7.1f} ms per view (median {1e3 * np.median(times):7.1f}); max rel "
              f"deviation from the first chunk size {deviation:.1e}; peak memory {peak:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
