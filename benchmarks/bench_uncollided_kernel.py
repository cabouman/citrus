"""Benchmark of the fused Triton first-collision source kernel against the PyTorch path on the WLAB phantom.

Measures, per fine voxel size, the time and peak memory of ``first_collision_source_at_directions`` for chunks of 8
directions on each backend with its deviation from a float64 evaluation, and the time, peak memory and detector
signal of the first-order scatter image computed with each backend. The records and a Markdown table are written to
``--out``.

Usage::

    python benchmarks/bench_uncollided_kernel.py [--fine 1.0 0.5] [--cones 16x32 32x64]
        [--out _cache/runs/bench_uncollided_kernel] [--skip-image] [--repeats 15]
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from citrus.scatter_solver import scatter_image as scatter_image_module  # noqa: E402
from citrus.scatter_solver import uncollided as uncollided_module  # noqa: E402
from citrus.scatter_solver.kernels import uncollided_kernels  # noqa: E402
from citrus.scatter_solver.config import SolverConfig  # noqa: E402
from citrus.scatter_solver.detector import csi_response, no_grid  # noqa: E402
from citrus.utilities import wlab_setup  # noqa: E402
from citrus.scatter_solver.ray_tracing import SiddonProjectors  # noqa: E402
from citrus.scatter_solver.cross_section_library import load_spectrum_txt  # noqa: E402
from citrus.scatter_solver.scatter_solver import build_solver_library  # noqa: E402
from citrus.scatter_solver.source import SourceModel  # noqa: E402
from citrus.scatter_solver.cross_sections import mu_t_volume  # noqa: E402

SPECTRUM = load_spectrum_txt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo", "spectra",
                                          "spectrum_125kvp.txt"))  # the sample 125 kVp spectrum

FIRST_COLLISION_BACKENDS = ("torch", "triton")
ORIGINAL_FIRST_COLLISION = uncollided_module.first_collision_source_at_directions
ORIGINAL_FUSED_USABLE = uncollided_kernels.fused_kernels_usable
CAP_MIB = 5000


def sync() -> None:
    """Wait for every CUDA kernel on the current device to finish."""
    torch.cuda.synchronize()


class use_first_collision_backend:
    """Context manager that routes ``first_collision_source_at_directions`` (in ``uncollided`` and ``scatter_image``)
    to one backend; ``"torch"`` also switches off the fused kernels of the source's uses (the image's emission).

    Attributes:
        backend (str): the first-collision backend.
        fn (Callable): ``first_collision_source_at_directions`` with the backend bound.
    """

    def __init__(self, backend: str):
        """Bind the backend.

        Args:
            backend (str): first-collision backend, ``"torch"`` or ``"triton"``.
        """
        self.backend = backend
        self.fn = functools.partial(ORIGINAL_FIRST_COLLISION, backend=backend)

    def __enter__(self):
        """Patch both module names to the bound function (and the fused kernels off for ``"torch"``).

        Returns:
            use_first_collision_backend: this context manager.
        """
        uncollided_module.first_collision_source_at_directions = self.fn
        scatter_image_module.first_collision_source_at_directions = self.fn
        if self.backend == "torch":
            uncollided_kernels.fused_kernels_usable = lambda *args, **kwargs: False
        return self

    def __exit__(self, *exc):
        """Restore the original functions.

        Args:
            *exc: exception type, value and traceback (ignored).

        Returns:
            bool: False, so exceptions propagate.
        """
        uncollided_module.first_collision_source_at_directions = ORIGINAL_FIRST_COLLISION
        scatter_image_module.first_collision_source_at_directions = ORIGINAL_FIRST_COLLISION
        uncollided_kernels.fused_kernels_usable = ORIGINAL_FUSED_USABLE
        return False


class Bench:
    """WLAB inputs, cached per fine voxel size, and the first-order scatter image.

    Attributes:
        device (torch.device): the CUDA device.
        library (MultigroupLibrary): cross-section library (6 groups, Legendre order 10).
        detector (DetectorResponse): CsI 0.6 mm detector response.
        grid_response (GridResponse): anti-scatter grid response (no grid).
        setups (dict): the inputs of ``setup`` by fine voxel size.
    """

    def __init__(self):
        """Load the cross-section library and the detector model."""
        self.device = torch.device("cuda")
        self.library = build_solver_library(SolverConfig(num_groups=6, legendre_order=10), SPECTRUM)  # cached on disk
        self.detector, self.grid_response = csi_response(0.06), no_grid()
        self.setups: dict[float, tuple] = {}

    def setup(self, fine: float):
        """Inputs on the WLAB grid of voxel size ``fine``, computed once.

        Args:
            fine (float): voxel size in cm.

        Returns:
            tuple: (grid, geometry, rho [M, R, C, S] in g/cm^3, mu [G, R, C, S] in cm^-1, phi_uncollided [G, R, C, S]);
                phi_uncollided comes from the Siddon upstream operator.
        """
        if fine not in self.setups:
            grid, geometry, rho = wlab_setup(fine, device=self.device, dtype=torch.float32)
            mu = mu_t_volume(self.library, rho)
            phi_uncollided = uncollided_module.uncollided_fluence(self.library, rho, grid, geometry, 0.0, SourceModel(),
                                                                  SiddonProjectors())
            self.setups[fine] = (grid, geometry, rho, mu, phi_uncollided)
        return self.setups[fine]

    def quadrature(self, fine: float, cone: tuple[int, int]):
        """Direction set of the scatter image in the cone that covers the grid, at angle 0.

        Args:
            fine (float): voxel size in cm.
            cone (tuple[int, int]): numbers of polar and azimuthal directions, e.g. (16, 32).

        Returns:
            Quadrature: the direction set.
        """
        grid, geometry = self.setup(fine)[:2]
        return scatter_image_module.panel_direction_set(geometry, 0.0, cone, max_polar_angle=grid, device=self.device,
                                                        dtype=torch.float32)

    def first_order(self, fine: float, quadrature) -> dict:
        """First-order scatter image on the fine grid (one quadrature point per cell, zero multiple-collision source).

        Args:
            fine (float): voxel size in cm.
            quadrature (Quadrature): direction set.

        Returns:
            dict: the image dict of ``combined_scatter_image`` (``signal`` [Rd, Cd], ...).
        """
        grid, geometry, rho, mu, phi_uncollided = self.setup(fine)
        T0 = torch.zeros(self.library.num_groups, 1, *grid.shape, device=self.device)
        return scatter_image_module.combined_scatter_image(self.library, rho, phi_uncollided, grid, 1, T0, 0, geometry,
                                                           0.0, quadrature, mu, SiddonProjectors(), self.detector,
                                                           self.grid_response, chunk=8, first_collision_chunk=4)

    def timed(self, fn) -> tuple[dict, float, float]:
        """Run a function once, synchronized, after emptying the allocator cache and resetting the peak memory.

        Args:
            fn (Callable): the function to run, without arguments.

        Returns:
            (result, seconds, peak_mib): the function's result, the wall time in s and the peak allocation in MiB.
        """
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        sync()
        t0 = time.perf_counter()
        r = fn()
        sync()
        return r, time.perf_counter() - t0, torch.cuda.max_memory_allocated(self.device) / 2 ** 20


def toward_panel(quadrature, geometry) -> torch.Tensor:
    """Directions of a set that point toward the panel at angle 0.

    Args:
        quadrature (Quadrature): direction set.
        geometry (ConeGeometry): scanner geometry.

    Returns:
        Tensor: [D, 3] unit directions with a positive component along ``-n`` of the panel.
    """
    directions = quadrature.omega
    _, _, n = geometry.panel_axes(0.0)
    return directions[(directions * (-torch.tensor(n, device=directions.device, dtype=directions.dtype))).sum(-1) > 0]


def per_chunk(b: Bench, fine: float, reps: int) -> dict:
    """Time, peak memory and accuracy of the first-collision source for 4 chunks of 8 directions of the 16x32 set.

    Args:
        b (Bench): the benchmark inputs.
        fine (float): voxel size in cm.
        reps (int): timed passes over the 4 chunks.

    Returns:
        dict: per backend the median and minimum ms and the peak extra MiB, and the deviations between the backends
            and from float64.
    """
    grid, geometry, rho, mu, phi_uncollided = b.setup(fine)
    directions_all = toward_panel(b.quadrature(fine, (16, 32)), geometry)
    starts = np.linspace(0, directions_all.shape[0] - 8, 4).astype(int)
    chunks = [directions_all[s:s + 8] for s in starts]
    record: dict = {"voxels": int(np.prod(grid.shape)), "grid": list(grid.shape)}
    outs = {}
    for be in FIRST_COLLISION_BACKENDS:
        fn = functools.partial(ORIGINAL_FIRST_COLLISION, b.library, rho, phi_uncollided, grid, geometry, 0.0, chunk=4,
                               backend=be)
        for directions in chunks:  # warm-up (compile, table preparation, self-check)
            fn(directions)
        sync()
        gc.collect()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(reps):
            for directions in chunks:
                sync()
                t0 = time.perf_counter()
                out = fn(directions)
                sync()
                times.append(1e3 * (time.perf_counter() - t0))
                del out
        peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
        outs[be] = fn(chunks[0])
        record[be] = {"ms_median": statistics.median(times), "ms_min": min(times), "peak_extra_mib": peak,
                      "out_mib": outs[be].numel() * 4 / 2 ** 20}
    reference64 = ORIGINAL_FIRST_COLLISION(b.library, rho.double(), phi_uncollided.double(), grid, geometry, 0.0,
                                           chunks[0].double(),
                                           backend="torch")
    mx = float(reference64.abs().max())
    record["dev_triton_vs_torch"] = (float((outs["triton"] - outs["torch"]).abs().max())
                                     / float(outs["torch"].abs().max()))
    for be in FIRST_COLLISION_BACKENDS:
        record[be]["dev_vs_float64"] = float((outs[be].double() - reference64).abs().max()) / mx
    print(f"fine {fine:g}: torch {record['torch']['ms_median']:.2f} ms, triton {record['triton']['ms_median']:.3f} ms, "
          f"peak extra {record['torch']['peak_extra_mib']:.0f} / {record['triton']['peak_extra_mib']:.0f} MiB, "
          f"dev {record['dev_triton_vs_torch']:.1e}", flush=True)
    return record


def first_order(b: Bench, fine: float, cones: list[tuple[int, int]]) -> dict:
    """Time and peak memory of the first-order image with each first-collision backend, after a warm-up.

    Args:
        b (Bench): the benchmark inputs.
        fine (float): voxel size in cm.
        cones (list[tuple[int, int]]): numbers of polar and azimuthal directions of each direction set.

    Returns:
        dict: per set and backend the seconds, peak MiB and directions, and per set the relative RMS difference of
            the detector signals.
    """
    out = {}
    # warm-up: every projector kernel variant and both first-collision paths
    with use_first_collision_backend("triton"), torch.no_grad():
        b.first_order(fine, b.quadrature(fine, (8, 16)))
    with use_first_collision_backend("torch"):
        b.first_order(fine, b.quadrature(fine, (8, 16)))
    for cone in cones:
        q = b.quadrature(fine, cone)
        signals = {}
        for be in FIRST_COLLISION_BACKENDS:
            with use_first_collision_backend(be):
                r, secs, peak = b.timed(lambda: b.first_order(fine, q))
            signals[be] = r["signal"].double().cpu().numpy()
            out[f"{cone[0]}x{cone[1]}_{be}"] = {"seconds": secs, "peak_mib": peak, "dirs": q.num_directions}
            print(f"fine {fine:g} cone {cone[0]}x{cone[1]} ({q.num_directions} dirs) first-collision {be}: {secs:.2f} "
                  f"s, "
                  f"peak {peak:.0f} MiB", flush=True)
            del r
        out[f"{cone[0]}x{cone[1]}_relrms"] = float(np.linalg.norm(signals["triton"] - signals["torch"])
                                                   / np.linalg.norm(signals["torch"]))
    return out


def markdown(records: dict, fines: list[float], cones: list[tuple[int, int]]) -> str:
    """Markdown tables of the records.

    Args:
        records (dict): the records of ``per_chunk`` and ``first_order`` by key.
        fines (list[float]): voxel sizes in cm.
        cones (list[tuple[int, int]]): numbers of polar and azimuthal directions of each direction set.

    Returns:
        str: the Markdown text.
    """
    md = ["## Per chunk of 8 directions (first_collision_source_at_directions, G = 6, median ms)", "",
          "| fine | voxels | torch (first_collision_chunk 4) ms | triton ms | speedup | torch peak extra MiB | "
          "triton peak extra MiB (= output) | max dev triton vs torch | torch vs f64 | triton vs f64 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for fine in fines:
        r = records.get(f"chunk_{fine:g}")
        if not r:
            continue
        md.append(f"| {fine:g} cm | {r['voxels']} | {r['torch']['ms_median']:.2f} | {r['triton']['ms_median']:.3f} | "
                  f"{r['torch']['ms_median'] / r['triton']['ms_median']:.0f}x | {r['torch']['peak_extra_mib']:.0f} | "
                  f"{r['triton']['peak_extra_mib']:.0f} | {r['dev_triton_vs_torch']:.1e} | "
                  f"{r['torch']['dev_vs_float64']:.1e} | {r['triton']['dev_vs_float64']:.1e} |")
    md += ["", "## First-order image on the fine grid, fused Siddon projectors, hemisphere sets", "",
           "| fine | dirs | first-collision torch s | first-collision triton s | speedup | peak MiB torch | "
           "peak MiB triton | rel RMS signal triton vs torch |", "|---|---|---|---|---|---|---|---|"]
    for fine in fines:
        r = records.get(f"image_{fine:g}")
        if not r:
            continue
        for cone in cones:
            c = f"{cone[0]}x{cone[1]}"
            if f"{c}_torch" not in r:
                continue
            t, f = r[f"{c}_torch"], r[f"{c}_triton"]
            md.append(f"| {fine:g} cm | {t['dirs']} | {t['seconds']:.2f} | {f['seconds']:.2f} | "
                      f"{t['seconds'] / f['seconds']:.1f}x | {t['peak_mib']:.0f} | {f['peak_mib']:.0f} | "
                      f"{r[f'{c}_relrms']:.1e} |")
    return "\n".join(md)


def main() -> None:
    """Parse the command line, run both parts per fine voxel size, then write and print the tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine", type=float, nargs="+", default=[1.0, 0.5])
    parser.add_argument("--cones", nargs="+", default=["16x32", "32x64"])
    parser.add_argument("--out", default=os.path.join("_cache", "runs", "bench_uncollided_kernel"))
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--skip-image", action="store_true")
    args = parser.parse_args()
    cones = [tuple(int(v) for v in c.split("x")) for c in args.cones]
    os.makedirs(args.out, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(CAP_MIB * 2 ** 20 / torch.cuda.get_device_properties(0).total_memory)
    print("first-collision kernel:", uncollided_kernels.kernel_usable(torch.device("cuda")), flush=True)
    b = Bench()
    path = os.path.join(args.out, "records.json")
    records: dict = {}
    if os.path.isfile(path):
        with open(path) as fh:
            records = json.load(fh)
    for fine in args.fine:
        records[f"chunk_{fine:g}"] = per_chunk(b, fine, args.repeats)
        if not args.skip_image:
            records[f"image_{fine:g}"] = first_order(b, fine, cones)
        with open(path, "w") as fh:
            json.dump(records, fh, indent=1)
    md = markdown(records, args.fine, cones)
    with open(os.path.join(args.out, "bench_uncollided_kernel.md"), "w") as fh:
        fh.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
