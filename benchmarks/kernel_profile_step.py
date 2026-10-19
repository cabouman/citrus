"""Kernel-level profile of one CITRUS step on the WLAB phantom (torch.profiler, CUDA time per kernel).

Runs a few views to warm up, then profiles the uncollided fluence, the sweep's first-collision source (sampling
correction and moment source), one accumulator update and the scatter image of a few views, and prints the CUDA time
per kernel name, grouped by stage.

Usage::

    python benchmarks/kernel_profile_step.py [--views 4] [--delta-voxel 0.5] [--top 12]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

from citrus.citrus import CitrusState, accumulator_update_operator, citrus_step, uncollided_operator  # noqa: E402
from citrus.scatter_solver.config import SolverConfig  # noqa: E402
from citrus.scatter_solver.cross_section_library import load_spectrum_txt  # noqa: E402
from citrus.utilities import (attenuation_to_densities, densities_to_attenuation, full_field_wlab_geometry,  # noqa: E402
                              wlab_grid, wlab_phantom, wlab_solver)

SPECTRUM = load_spectrum_txt(os.path.join(ROOT, "demo", "spectra", "spectrum_125kvp.txt"))
DEVICE = torch.device("cuda")


def main() -> None:
    """Warm up, profile a few views stage by stage and print the CUDA time per kernel."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--delta-voxel", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    geometry = full_field_wlab_geometry()
    grid = wlab_grid(args.delta_voxel)
    solver = wlab_solver(args.delta_voxel, config=SolverConfig(), geometry=geometry, device=DEVICE, spectrum=SPECTRUM)
    library = solver.library
    X = densities_to_attenuation(library, wlab_phantom(grid, device=DEVICE))
    rho = attenuation_to_densities(library, X.clamp_min(0))
    angles = 2 * np.pi * np.arange(args.views) / 180
    states = {}
    for a in angles:  # warm up: compile and build the accumulators
        _, _, states[a] = citrus_step(solver, CitrusState(), rho, grid, float(a))
    torch.cuda.synchronize()
    stages = ("uncollided", "sampling correction", "accumulator update", "scatter image")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for a in angles:
            a = float(a)
            fields = solver.material_fields(rho, grid)
            with record_function("uncollided"):
                phi = uncollided_operator(solver, rho, grid, a, fields)
            with record_function("sampling correction"):
                solver.first_collision_sampling_correction(fields, a, phi)
            with record_function("accumulator update"):
                moments = accumulator_update_operator(solver, rho, grid, a, phi, states[a].collided_moments, fields)
            with record_function("scatter image"):
                source_moments = solver.multiple_collision_source_moments(fields, moments)
                q = states[a].cone_quadrature if states[a].cone_quadrature is not None else solver.cone_quadrature(
                    grid, a)
                solver.scatter_image(fields, a, phi, source_moments, q)
        torch.cuda.synchronize()
    # attribute every CUDA kernel to the stage (record_function range) of the CPU op that launched it
    events = prof.events()
    per_stage: dict[str, dict[str, float]] = {s: defaultdict(float) for s in stages}
    totals = defaultdict(float)
    for e in events:
        if not e.kernels:
            continue
        stage_name = None
        parent = e.cpu_parent
        while parent is not None:
            if parent.name in stages:
                stage_name = parent.name
                break
            parent = parent.cpu_parent
        if stage_name is None:
            continue
        for k in e.kernels:
            per_stage[stage_name][k.name[:90]] += k.duration / 1e3  # ms
            totals[stage_name] += k.duration / 1e3
    n = len(angles)
    # every CUDA kernel of the profile by name, Triton kernels included (they are not attributed to a stage)
    by_name: dict[str, float] = defaultdict(float)
    for e in events:
        if e.device_type.name == "CUDA" and e.device_time_total > 0 and not e.cpu_children:
            by_name[e.name[:90]] += e.device_time_total / 1e3
    # GPU busy time inside each stage's GPU-side range: the union of the kernel intervals in its window
    kernels = [e for e in events if e.device_type.name == "CUDA" and e.name not in stages and e.device_time_total > 0]
    for s in stages:
        windows = [e.time_range for e in events if e.device_type.name == "CUDA" and e.name == s]
        span = sum(w.end - w.start for w in windows) / 1e3
        intervals = sorted((max(k.time_range.start, w.start), min(k.time_range.end, w.end)) for w in windows
                           for k in kernels if k.time_range.end > w.start and k.time_range.start < w.end)
        busy, current = 0.0, None
        for a, b in intervals:
            if current is None or a > current[1]:
                if current is not None:
                    busy += current[1] - current[0]
                current = [a, b]
            else:
                current[1] = max(current[1], b)
        if current is not None:
            busy += current[1] - current[0]
        print(f"   {s:22s} GPU range {span / n:7.1f} ms per view, kernels busy {busy / 1e3 / n:7.1f} ms")
    print()
    print(f"== all CUDA kernels: {sum(by_name.values()) / n:.1f} ms per view")
    for name, t in sorted(by_name.items(), key=lambda kv: -kv[1])[:2 * args.top]:
        print(f"   {t / n:8.2f} ms  {name}")
    for s in stages:
        print(f"\n== {s}: {totals[s] / n:.1f} ms of CUDA kernels per view")
        for name, t in sorted(per_stage[s].items(), key=lambda kv: -kv[1])[:args.top]:
            print(f"   {t / n:8.2f} ms  {name}")


if __name__ == "__main__":
    main()
