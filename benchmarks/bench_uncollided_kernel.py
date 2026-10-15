"""Benchmark of the fused first-collision source kernel (citrus/scatter_solver/kernels/uncollided_kernels.py).

Usage:
    python benchmarks/bench_uncollided_kernel.py [--fine 1.0 0.5] [--cones 16x32 32x64]
                                                  [--out _cache/runs/bench_uncollided_kernel] [--skip-image] [--reps 15]

Setting: WLAB phantom (utilities.wlab_setup), angle 0, real XrayPhysics library (G = 6, 2001-point eta table),
CsI 0.6 mm response, no grid, float32, CUDA, caching-allocator cap 5000 MiB (Windows spills VRAM
overflow into system RAM silently). phi_uc from the Siddon upstream operator.

Parts:
 1. per chunk: `uncollided.fc_source_at_directions` for chunks of 8 directions of the 16x32 cone set (the
    chunks of the scatter image), backend "torch" (fc_chunk = 4, as the scatter image calls it) and "triton"; median wall ms over
    --reps calls x 4 different chunks (CUDA synchronized), peak MiB above the resident inputs
    (max_memory_allocated - memory_allocated before the call) and the max deviation between the backends
    (relative to the field maximum) and of each backend vs a float64 evaluation.
 2. first-order image on the fine grid (`scatter_image.combined_scatter_image` with supervoxel factor 1 and a zero
    multiple-collision source, chunk=8, fc_chunk=4) with the default fused Siddon projectors (`SiddonProjectors()`)
    on the cone sets --cones, FC source on each backend (the backend is selected by patching the name that
    scatter_image resolves); wall seconds after a warm-up, peak MiB (max_memory_allocated, absolute), and the rel RMS of
    the detector signal between the two FC backends.
Outputs: <out>/records.json and <out>/bench_uncollided_kernel.md (also printed).
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

from citrus.scatter_solver import scatter_image as step3  # noqa: E402
from citrus.scatter_solver import uncollided as step1  # noqa: E402
from citrus.scatter_solver.kernels import uncollided_kernels as step1_triton  # noqa: E402
from citrus.scatter_solver.config import SolverConfig  # noqa: E402
from citrus.scatter_solver.detector import csi_response, no_grid  # noqa: E402
from citrus.utilities import wlab_setup  # noqa: E402
from citrus.scatter_solver.ray_tracing import SiddonProjectors  # noqa: E402
from citrus.scatter_solver.scatter_solver import build_solver_library  # noqa: E402
from citrus.scatter_solver.source import SourceModel  # noqa: E402
from citrus.scatter_solver.cross_sections import mu_t_volume  # noqa: E402

FC_BACKENDS = ("torch", "triton")
ORIGINAL_FC = step1.fc_source_at_directions
CAP_MIB = 5000


def sync() -> None:
    torch.cuda.synchronize()


class use_fc_backend:
    """Context manager: route uncollided.fc_source_at_directions (as resolved by scatter_image and by uncollided's own
    callers) to the
    given backend."""

    def __init__(self, backend: str):
        self.fn = functools.partial(ORIGINAL_FC, backend=backend)

    def __enter__(self):
        step1.fc_source_at_directions = self.fn
        step3.fc_source_at_directions = self.fn
        return self

    def __exit__(self, *exc):
        step1.fc_source_at_directions = ORIGINAL_FC
        step3.fc_source_at_directions = ORIGINAL_FC
        return False


class Bench:
    """WLAB setups (fine grid, geometry, densities, mu_t, phi_uc) and the first-order image, cached per fine voxel."""

    def __init__(self):
        self.dev = torch.device("cuda")
        self.lib = build_solver_library(SolverConfig(num_groups=6, legendre_order=10))  # cached on disk
        self.det, self.grid_resp = csi_response(0.06), no_grid()
        self.setups: dict[float, tuple] = {}

    def setup(self, fine: float):
        if fine not in self.setups:
            grid, geom, rho = wlab_setup(fine, device=self.dev, dtype=torch.float32)
            mu = mu_t_volume(self.lib, rho)
            phi_uc = step1.uncollided_fluence(self.lib, rho, grid, geom, 0.0, SourceModel(), SiddonProjectors())
            self.setups[fine] = (grid, geom, rho, mu, phi_uc)
        return self.setups[fine]

    def quad(self, fine: float, cone: tuple[int, int]):
        grid, geom = self.setup(fine)[:2]
        return step3.panel_direction_set(geom, 0.0, cone, max_polar_angle=grid, device=self.dev, dtype=torch.float32)

    def first_order(self, fine: float, quad) -> dict:
        """First-order image on the fine grid: the combined pass at supervoxel factor 1 with a zero MC source."""
        grid, geom, rho, mu, phi_uc = self.setup(fine)
        T0 = torch.zeros(self.lib.num_groups, 1, *grid.shape, device=self.dev)
        return step3.combined_scatter_image(self.lib, rho, phi_uc, grid, 1, T0, 0, geom, 0.0, quad, mu,
                                            SiddonProjectors(), self.det, self.grid_resp, chunk=8, fc_chunk=4)

    def timed(self, fn) -> tuple[dict, float, float]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.dev)
        sync()
        t0 = time.perf_counter()
        r = fn()
        sync()
        return r, time.perf_counter() - t0, torch.cuda.max_memory_allocated(self.dev) / 2 ** 20


def toward_panel(quad, geom) -> torch.Tensor:
    om = quad.omega
    _, _, n = geom.panel_axes(0.0)
    return om[(om * (-torch.tensor(n, device=om.device, dtype=om.dtype))).sum(-1) > 0]


def per_chunk(b: Bench, fine: float, reps: int) -> dict:
    grid, geom, rho, mu, phi_uc = b.setup(fine)
    om_all = toward_panel(b.quad(fine, (16, 32)), geom)
    starts = np.linspace(0, om_all.shape[0] - 8, 4).astype(int)
    chunks = [om_all[s:s + 8] for s in starts]
    rec: dict = {"voxels": int(np.prod(grid.shape)), "grid": list(grid.shape)}
    outs = {}
    for be in FC_BACKENDS:
        fn = functools.partial(ORIGINAL_FC, b.lib, rho, phi_uc, grid, geom, 0.0, chunk=4, backend=be)
        for om in chunks:  # warm-up (compile, table preparation, self-check)
            fn(om)
        sync()
        gc.collect()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(reps):
            for om in chunks:
                sync()
                t0 = time.perf_counter()
                out = fn(om)
                sync()
                times.append(1e3 * (time.perf_counter() - t0))
                del out
        peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
        outs[be] = fn(chunks[0])
        rec[be] = {"ms_median": statistics.median(times), "ms_min": min(times), "peak_extra_mib": peak,
                   "out_mib": outs[be].numel() * 4 / 2 ** 20}
    ref64 = ORIGINAL_FC(b.lib, rho.double(), phi_uc.double(), grid, geom, 0.0, chunks[0].double(), backend="torch")
    mx = float(ref64.abs().max())
    rec["dev_triton_vs_torch"] = float((outs["triton"] - outs["torch"]).abs().max()) / float(outs["torch"].abs().max())
    for be in FC_BACKENDS:
        rec[be]["dev_vs_float64"] = float((outs[be].double() - ref64).abs().max()) / mx
    print(f"fine {fine:g}: torch {rec['torch']['ms_median']:.2f} ms, triton {rec['triton']['ms_median']:.3f} ms, "
          f"peak extra {rec['torch']['peak_extra_mib']:.0f} / {rec['triton']['peak_extra_mib']:.0f} MiB, "
          f"dev {rec['dev_triton_vs_torch']:.1e}", flush=True)
    return rec


def first_order(b: Bench, fine: float, cones: list[tuple[int, int]]) -> dict:
    out = {}
    with use_fc_backend("triton"), torch.no_grad():  # warm-up: every projector kernel variant and both FC paths
        b.first_order(fine, b.quad(fine, (8, 16)))
    with use_fc_backend("torch"):
        b.first_order(fine, b.quad(fine, (8, 16)))
    for cone in cones:
        q = b.quad(fine, cone)
        sig = {}
        for be in FC_BACKENDS:
            with use_fc_backend(be):
                r, secs, peak = b.timed(lambda: b.first_order(fine, q))
            sig[be] = r["signal"].double().cpu().numpy()
            out[f"{cone[0]}x{cone[1]}_{be}"] = {"seconds": secs, "peak_mib": peak, "dirs": q.num_directions}
            print(f"fine {fine:g} cone {cone[0]}x{cone[1]} ({q.num_directions} dirs) FC {be}: {secs:.2f} s, "
                  f"peak {peak:.0f} MiB", flush=True)
            del r
        out[f"{cone[0]}x{cone[1]}_relrms"] = float(np.linalg.norm(sig["triton"] - sig["torch"])
                                                   / np.linalg.norm(sig["torch"]))
    return out


def markdown(recs: dict, fines: list[float], cones: list[tuple[int, int]]) -> str:
    md = ["## Per chunk of 8 directions (fc_source_at_directions, G = 6, median ms)", "",
          "| fine | voxels | torch (fc_chunk 4) ms | triton ms | speedup | torch peak extra MiB | "
          "triton peak extra MiB (= output) | max dev triton vs torch | torch vs f64 | triton vs f64 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for fine in fines:
        r = recs.get(f"chunk_{fine:g}")
        if not r:
            continue
        md.append(f"| {fine:g} cm | {r['voxels']} | {r['torch']['ms_median']:.2f} | {r['triton']['ms_median']:.3f} | "
                  f"{r['torch']['ms_median'] / r['triton']['ms_median']:.0f}x | {r['torch']['peak_extra_mib']:.0f} | "
                  f"{r['triton']['peak_extra_mib']:.0f} | {r['dev_triton_vs_torch']:.1e} | "
                  f"{r['torch']['dev_vs_float64']:.1e} | {r['triton']['dev_vs_float64']:.1e} |")
    md += ["", "## First-order image on the fine grid, fused Siddon projectors, hemisphere sets", "",
           "| fine | dirs | FC torch s | FC triton s | speedup | peak MiB torch | peak MiB triton | "
           "rel RMS signal triton vs torch |", "|---|---|---|---|---|---|---|---|"]
    for fine in fines:
        r = recs.get(f"image_{fine:g}")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--fine", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--cones", nargs="+", default=["16x32", "32x64"])
    ap.add_argument("--out", default=os.path.join("_cache", "runs", "bench_uncollided_kernel"))
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--skip-image", action="store_true")
    args = ap.parse_args()
    cones = [tuple(int(v) for v in c.split("x")) for c in args.cones]
    os.makedirs(args.out, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(CAP_MIB * 2 ** 20 / torch.cuda.get_device_properties(0).total_memory)
    print("FC kernel:", step1_triton.kernel_usable(torch.device("cuda")), flush=True)
    b = Bench()
    path = os.path.join(args.out, "records.json")
    recs: dict = {}
    if os.path.isfile(path):
        with open(path) as fh:
            recs = json.load(fh)
    for fine in args.fine:
        recs[f"chunk_{fine:g}"] = per_chunk(b, fine, args.reps)
        if not args.skip_image:
            recs[f"image_{fine:g}"] = first_order(b, fine, cones)
        with open(path, "w") as fh:
            json.dump(recs, fh, indent=1)
    md = markdown(recs, args.fine, cones)
    with open(os.path.join(args.out, "bench_uncollided_kernel.md"), "w") as fh:
        fh.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
