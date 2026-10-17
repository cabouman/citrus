"""Benchmark of the fused Triton Siddon ray tracer against the PyTorch Siddon reference.

Usage:
    python benchmarks/bench_ray_tracing.py [--coarse 2.0 1.0 0.5] [--cones 16x32 32x64]
                                            [--out _cache/runs/bench_ray_tracing] [--parts equality,chunks,images,fine]
                                            [--torch-chunks 8] [--skip KEY,...] [--rerun]

Setting: WLAB phantom (utilities.wlab_setup), angle 0, real XrayPhysics library (G = 6, the cached build at
Legendre order 10), CsI 0.6 mm response, no grid, float32, CUDA, caching-allocator cap 5000 MiB. The
solver traces the attenuation tau_down on the transport (coarse) grid and projects the emission at the quadrature
points (scatter_image.combined_scatter_image). Here the quadrature points are the 0.5 cm grid and the coarse grid
2 / 1 / 0.5 cm (q = 4 / 2 / 1 points per cell and axis). Direction sets: `scatter_image.panel_direction_set` in the
51 deg covering cone, 16x32 = 512 and
32x64 = 2048 directions; the scatter image's chunks of 8 panel-facing directions.

Backends:
    torch     SiddonProjectors(backend="torch")     (exact; PyTorch traversal: the reference)
    triton    SiddonProjectors(backend="auto")      (exact; the fused kernels of kernels/ray_tracing_kernels.py; default)

Parts (--parts):
    equality  max |Triton - PyTorch| / max |PyTorch| for every operation on the WLAB grids (one chunk of 8 directions
              of the 512 set; downstream on each coarse grid, oblique on the fine grid with a per-direction field,
              upstream and primary on the fine grid), and vs the float64 PyTorch path for one chunk.
    chunks    ms per chunk of 8 of downstream_cumulative(mu_coarse) on the coarse grid and of
              oblique_project([8, G, fine]) on the fine grid, over every panel-facing chunk of the set, pipelined as in
              the scatter image (one synchronize at the end), median of 3 passes after a warm pass. The PyTorch Siddon is timed
              on the first --torch-chunks chunks (one pass; its cost per chunk does not depend on the chunk). Also the
              peak memory of one chunk pair.
    images    the first-order image (`combined_scatter_image(chunk=8, fc_chunk=4)` with a zero multiple-collision
              source) with a fresh projector per image, seconds (wall, synchronized, after an untimed warm-up of every
              backend on a 4x8 set), peak MiB, and rel RMS of the detector signal vs the PyTorch Siddon image of the
              same set and factor. phi_uc comes from the PyTorch Siddon upstream operator once (identical inputs).
    fine      the first-order image on a single grid (q = 1, the quadrature grid equal to the transport grid) at
              2 / 1 cm (0.5 cm is the images part at factor 1, the identical computation).
Outputs: <out>/records.jsonl, <out>/img/<key>.npz, <out>/bench_ray_tracing.md (printed).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Callable, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from citrus.scatter_solver.config import SolverConfig  # noqa: E402
from citrus.scatter_solver.detector import csi_response, no_grid  # noqa: E402
from citrus.utilities import wlab_setup  # noqa: E402
from citrus.scatter_solver.kernels import ray_tracing_kernels as siddon_triton  # noqa: E402
from citrus.scatter_solver.ray_tracing import SiddonProjectors  # noqa: E402
from citrus.scatter_solver.scatter_solver import build_solver_library  # noqa: E402
from citrus.scatter_solver.source import SourceModel  # noqa: E402
from citrus.scatter_solver.uncollided import uncollided_fluence  # noqa: E402
from citrus.scatter_solver.scatter_image import coarsen_mu_t, combined_scatter_image, panel_direction_set  # noqa: E402
from citrus.scatter_solver.cross_sections import mu_t_volume  # noqa: E402

CAP_MIB = 5000
FINE = 0.5
IMPLS = ("torch", "triton")
MAKERS: dict[str, Callable[[], object]] = {
    "torch": lambda: SiddonProjectors(backend="torch"),
    "triton": lambda: SiddonProjectors(backend="auto"),
}
LABEL = {"torch": "Siddon (PyTorch)", "triton": "Siddon (Triton)"}


def sync() -> None:
    torch.cuda.synchronize()


def library():
    return build_solver_library(SolverConfig(num_groups=6, legendre_order=10))  # cached on disk (_cache/)


def rel_rms(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def first_order_image(lib, rho, phi_uc, grid, factor: int, geom, quad, mu_coarse, projectors, det, grid_resp) -> dict:
    """First-order image of the solver's scatter pass: combined_scatter_image with a zero multiple-collision source
    (mu_coarse None -> the true coarse attenuation)."""
    T0 = torch.zeros(lib.num_groups, 1, *grid.coarsen(factor).shape, device=phi_uc.device, dtype=phi_uc.dtype)
    return combined_scatter_image(lib, rho, phi_uc, grid, factor, T0, 0, geom, 0.0, quad, mu_coarse, projectors, det,
                                  grid_resp, chunk=8, fc_chunk=4)


def foreign_gpu_load() -> str:
    """GPU utilization and memory reported by nvidia-smi (includes this process)."""
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return out[0] if out else "?"
    except Exception:  # noqa: BLE001
        return "?"


class Bench:
    def __init__(self, args):
        self.args = args
        self.dev = torch.device("cuda")
        os.makedirs(os.path.join(args.out, "img"), exist_ok=True)
        self.jsonl = os.path.join(args.out, "records.jsonl")
        self.records: dict[str, dict] = {}
        if os.path.isfile(self.jsonl) and not args.rerun:
            with open(self.jsonl) as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        self.records[r["key"]] = r
        self.lib = library()
        self.det, self.grid_resp = csi_response(0.06), no_grid()
        self.setups: dict[float, tuple] = {}

    # ---- inputs -----------------------------------------------------------------------------------
    def setup(self, d: float):
        """(grid, geom, rho, mu, phi_uc) on the d cm grid; phi_uc from the PyTorch Siddon upstream operator."""
        if d not in self.setups:
            grid, geom, rho = wlab_setup(d, device=self.dev, dtype=torch.float32)
            mu = mu_t_volume(self.lib, rho)
            phi_uc = uncollided_fluence(self.lib, rho, grid, geom, 0.0, SourceModel(),
                                        SiddonProjectors(backend="torch"))
            self.setups[d] = (grid, geom, rho, mu, phi_uc)
        return self.setups[d]

    def quad(self, cone: tuple[int, int]):
        grid, geom = self.setup(FINE)[:2]
        return panel_direction_set(geom, 0.0, cone, max_polar_angle=grid, device=self.dev, dtype=torch.float32)

    def chunks(self, cone: tuple[int, int], chunk: int = 8) -> list[torch.Tensor]:
        geom = self.setup(FINE)[1]
        om = self.quad(cone).omega.to(self.dev, torch.float32)
        _, _, n = geom.panel_axes(0.0)
        om = om[(om * (-torch.tensor(n, device=self.dev))).sum(-1) > 0].contiguous()
        return [om[s:s + chunk] for s in range(0, om.shape[0], chunk)]

    # ---- records ----------------------------------------------------------------------------------
    def store(self, rec: dict, imgs: Optional[dict] = None) -> None:
        if imgs is not None:
            np.savez_compressed(os.path.join(self.args.out, "img", rec["key"] + ".npz"), **imgs)
        self.records[rec["key"]] = rec
        with open(self.jsonl, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def image(self, key: str) -> Optional[np.ndarray]:
        p = os.path.join(self.args.out, "img", key + ".npz")
        if not os.path.isfile(p):
            return None
        with np.load(p) as z:
            return z["signal"]

    def timed(self, fn: Callable[[], object]) -> tuple[object, float, float]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.dev)
        sync()
        t0 = time.perf_counter()
        r = fn()
        sync()
        return r, time.perf_counter() - t0, torch.cuda.max_memory_allocated(self.dev) / 2 ** 20

    def skip(self, key: str) -> bool:
        return key in self.records or key in self.args.skip

    # ---- part 1: equality ---------------------------------------------------------------------------
    def run_equality(self, coarse: list[float]) -> list[dict]:
        key = "equality"
        if self.skip(key):
            return self.records[key]["rows"]
        grid_f, geom, rho_f, mu_f, _ = self.setup(FINE)
        om = self.chunks((16, 32))[5]
        T = SiddonProjectors(backend="torch")
        rows = []

        def dev_of(a: torch.Tensor, b: torch.Tensor) -> float:
            m = torch.isfinite(b)
            assert bool((torch.isfinite(a) == m).all())
            return float((a[m].double() - b[m].double()).abs().max() / b[m].double().abs().max())

        for d in coarse:
            f = int(round(d / FINE))
            grid_c = grid_f.coarsen(f)
            mu_c = coarsen_mu_t(mu_f, grid_f, grid_c)
            ref = T.downstream_cumulative(mu_c, grid_c, geom, 0.0, om)
            r64 = T.downstream_cumulative(mu_c.double(), grid_c, geom, 0.0, om.double())
            got = MAKERS["triton"]().downstream_cumulative(mu_c, grid_c, geom, 0.0, om)
            rows.append({"op": f"downstream_cumulative ({d:g} cm)", "impl": "triton", "vs_torch32": dev_of(got, ref),
                         "vs_torch64": dev_of(got, r64)})
            rows.append({"op": f"downstream_cumulative ({d:g} cm)", "impl": "torch", "vs_torch32": 0.0,
                         "vs_torch64": dev_of(ref, r64)})
            del ref, r64
        gen = torch.Generator(self.dev).manual_seed(3)
        field = torch.rand(om.shape[0], mu_f.shape[0], *grid_f.shape, device=self.dev, generator=gen)
        P = MAKERS["triton"]()
        ops = [("oblique_project (0.5 cm, [8, G] field)",
                lambda Q, dt: Q.oblique_project(field.to(dt), grid_f, geom, 0.0, om.to(dt))),
               ("upstream_cumulative (0.5 cm)", lambda Q, dt: Q.upstream_cumulative(mu_f.to(dt), grid_f, geom, 0.0)),
               ("primary_project (0.5 cm)", lambda Q, dt: Q.primary_project(mu_f.to(dt), grid_f, geom, 0.0))]
        for op, fn in ops:
            ref = fn(T, torch.float32)
            got = fn(P, torch.float32)
            r64 = fn(T, torch.float64)
            rows.append({"op": op, "impl": "triton", "vs_torch32": dev_of(got, ref), "vs_torch64": dev_of(got, r64)})
            rows.append({"op": op, "impl": "torch", "vs_torch32": 0.0, "vs_torch64": dev_of(ref, r64)})
            del ref, got, r64
        self.store({"key": key, "rows": rows})
        for r in rows:
            print(f"equality {r['op']} {r['impl']}: vs torch32 {r['vs_torch32']:.1e}, vs torch64 {r['vs_torch64']:.1e}",
                  flush=True)
        return rows

    # ---- part 2: per-chunk cost ---------------------------------------------------------------------
    def run_chunks(self, d: float, cone: tuple[int, int], impl: str) -> dict:
        key = f"chunks_c{d:g}_q{cone[0]}x{cone[1]}_{impl}"
        if self.skip(key):
            return self.records.get(key, {})
        grid_f, geom, _, mu_f = self.setup(FINE)[:4]
        f = int(round(d / FINE))
        grid_c = grid_f.coarsen(f)
        mu_c = coarsen_mu_t(mu_f, grid_f, grid_c)
        chunks = self.chunks(cone)
        G = mu_f.shape[0]
        field = torch.rand(8, G, *grid_f.shape, device=self.dev, generator=torch.Generator(self.dev).manual_seed(3))
        P = MAKERS[impl]()
        reps = 3
        if impl == "torch":
            chunks = chunks[:self.args.torch_chunks]
            reps = 1

        def down():
            for o in chunks:
                P.downstream_cumulative(mu_c, grid_c, geom, 0.0, o)

        def obl():
            for o in chunks:
                P.oblique_project(field[:o.shape[0]], grid_f, geom, 0.0, o)

        def pair():
            for o in chunks:
                t = P.downstream_cumulative(mu_c, grid_c, geom, 0.0, o)
                P.oblique_project(field[:o.shape[0]], grid_f, geom, 0.0, o)
                del t

        rec = {"key": key, "coarse": d, "cone": list(cone), "impl": impl, "chunks": len(chunks),
               "gpu": foreign_gpu_load()}
        for name, fn in (("down", down), ("obl", obl), ("pair", pair)):
            fn()
            ts = []
            for _ in range(reps):
                sync()
                t0 = time.perf_counter()
                fn()
                sync()
                ts.append(time.perf_counter() - t0)
            rec[f"{name}_ms"] = 1e3 * sorted(ts)[len(ts) // 2] / len(chunks)
        gc.collect()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(self.dev)
        torch.cuda.reset_peak_memory_stats(self.dev)
        o = chunks[0]
        t = P.downstream_cumulative(mu_c, grid_c, geom, 0.0, o)
        P.oblique_project(field[:o.shape[0]], grid_f, geom, 0.0, o)
        del t
        sync()
        rec["pair_peak_extra_mib"] = (torch.cuda.max_memory_allocated(self.dev) - base) / 2 ** 20
        self.store(rec)
        print(f"chunks {d:g} cm {cone} {impl}: " + ", ".join(f"{k} {v:.3f}" for k, v in rec.items()
                                                           if k.endswith("_ms") or k.endswith("_mib"))
              + f" (nvidia-smi: {rec['gpu']})", flush=True)
        return rec

    # ---- part 3: first-order image (supervoxel, sub-points on the fine grid) -------------------------
    def first_order(self, d: float, quad, impl: str) -> dict:
        grid_f, geom, rho_f, _, phi_f = self.setup(FINE)
        f = int(round(d / FINE))
        return first_order_image(self.lib, rho_f, phi_f, grid_f, f, geom, quad, None, MAKERS[impl](), self.det,
                                 self.grid_resp)

    def run_image(self, d: float, cone: tuple[int, int], impl: str) -> dict:
        key = f"img_c{d:g}_q{cone[0]}x{cone[1]}_{impl}"
        if self.skip(key):
            return self.records.get(key, {})
        q = self.quad(cone)
        print(f"[{time.strftime('%H:%M:%S')}] {key} ({q.num_directions} dirs)", flush=True)
        r, secs, peak = self.timed(lambda: self.first_order(d, q, impl))
        rec = {"key": key, "coarse": d, "cone": list(cone), "impl": impl, "seconds": secs, "peak_mib": peak,
               "dirs": q.num_directions, "gpu": foreign_gpu_load()}
        self.store(rec, {"signal": r["signal"].double().cpu().numpy()})
        print(f"    {secs:.3f} s, peak {peak:.0f} MiB (nvidia-smi: {rec['gpu']})", flush=True)
        return rec

    # ---- part 4: single-grid first-order image --------------------------------------------------------
    def run_fine_image(self, d: float, cone: tuple[int, int], impl: str) -> dict:
        key = f"fine_d{d:g}_q{cone[0]}x{cone[1]}_{impl}"
        if self.skip(key):
            return self.records.get(key, {})
        grid, geom, rho, mu, phi = self.setup(d)
        q = panel_direction_set(geom, 0.0, cone, max_polar_angle=grid, device=self.dev, dtype=torch.float32)
        print(f"[{time.strftime('%H:%M:%S')}] {key} ({q.num_directions} dirs)", flush=True)
        r, secs, peak = self.timed(lambda: first_order_image(self.lib, rho, phi, grid, 1, geom, q, mu,
                                                             MAKERS[impl](), self.det, self.grid_resp))
        rec = {"key": key, "fine": d, "cone": list(cone), "impl": impl, "seconds": secs, "peak_mib": peak,
               "dirs": q.num_directions, "gpu": foreign_gpu_load()}
        self.store(rec, {"signal": r["signal"].double().cpu().numpy()})
        print(f"    {secs:.3f} s, peak {peak:.0f} MiB (nvidia-smi: {rec['gpu']})", flush=True)
        return rec

    def warmup(self, coarse: list[float]) -> None:
        """Untimed small runs of every backend (CUDA context, allocator, every Triton kernel variant)."""
        t0 = time.perf_counter()
        siddon_triton.warmup(self.dev, int(self.lib.num_groups))
        q = self.quad((4, 8))
        for d in coarse:
            for impl in IMPLS:
                self.first_order(d, q, impl)
        sync()
        print(f"warm-up: {time.perf_counter() - t0:.1f} s; siddon kernels: {siddon_triton.kernels_usable(self.dev)}",
              flush=True)


# =====================================================================================================
# tables
# =====================================================================================================

def _f(x, fmt: str) -> str:
    return "" if x is None else format(x, fmt)


def tables(b: Bench, coarse: list[float], cones: list[tuple[int, int]], parts: set[str]) -> str:
    md: list[str] = []
    R = b.records
    if "equality" in parts and "equality" in R:
        md += ["### Equality (WLAB, one chunk of 8 directions of the 512 set; max |a - b| / max |b|)", "",
               "| operation | implementation | vs PyTorch float32 | vs PyTorch float64 |", "|---|---|---|---|"]
        for r in R["equality"]["rows"]:
            md.append(f"| {r['op']} | {LABEL[r['impl']]} | {r['vs_torch32']:.1e} | {r['vs_torch64']:.1e} |")
        md.append("")
    if "chunks" in parts:
        md += ["### Per-chunk cost (ms per chunk of 8 directions; downstream on the coarse grid, oblique on the fine "
               "0.5 cm grid)", "",
               "| coarse grid | dirs | backend | downstream ms/chunk | oblique ms/chunk | pair ms/chunk | "
               "pair peak extra MiB |", "|---|---|---|---|---|---|---|"]
        for d in coarse:
            for cone in cones:
                for impl in IMPLS:
                    r = R.get(f"chunks_c{d:g}_q{cone[0]}x{cone[1]}_{impl}")
                    if not r:
                        continue
                    md.append(f"| {d:g} cm | {cone[0] * cone[1]} | {LABEL[impl]} | {_f(r.get('down_ms'), '.3f')} | "
                              f"{_f(r.get('obl_ms'), '.3f')} | {_f(r.get('pair_ms'), '.3f')} | "
                              f"{_f(r.get('pair_peak_extra_mib'), '.0f')} |")
        md.append("")
    if "images" in parts:
        md += ["### First-order image (supervoxel tau on the coarse grid, sub-points projected on the fine 0.5 cm grid)",
               "", "| coarse grid | dirs | backend | image s | peak MiB | rel RMS vs Siddon (PyTorch) | "
               "rel RMS vs Siddon (Triton) |", "|---|---|---|---|---|---|---|"]
        for d in coarse:
            for cone in cones:
                ref_t = b.image(f"img_c{d:g}_q{cone[0]}x{cone[1]}_torch")
                ref_k = b.image(f"img_c{d:g}_q{cone[0]}x{cone[1]}_triton")
                for impl in IMPLS:
                    key = f"img_c{d:g}_q{cone[0]}x{cone[1]}_{impl}"
                    r = R.get(key)
                    if not r:
                        continue
                    img = b.image(key)
                    md.append(f"| {d:g} cm | {r['dirs']} | {LABEL[impl]} | {r['seconds']:.3f} | {r['peak_mib']:.0f} | "
                              f"{_f(rel_rms(img, ref_t), '.2e')} | {_f(rel_rms(img, ref_k), '.2e')} |")
        md.append("")
    if "fine" in parts:
        md += ["### First-order image on a single grid (q = 1)", "",
               "| grid | dirs | backend | image s | peak MiB | rel RMS vs Siddon (PyTorch) | rel RMS vs Siddon (Triton) |",
               "|---|---|---|---|---|---|---|"]
        for d in sorted({float(x) for x in b.args.fine_grids}, reverse=True):
            for cone in cones:
                ref_t = b.image(f"fine_d{d:g}_q{cone[0]}x{cone[1]}_torch")
                ref_k = b.image(f"fine_d{d:g}_q{cone[0]}x{cone[1]}_triton")
                for impl in IMPLS:
                    key = f"fine_d{d:g}_q{cone[0]}x{cone[1]}_{impl}"
                    r = R.get(key)
                    if not r:
                        continue
                    img = b.image(key)
                    md.append(f"| {d:g} cm | {r['dirs']} | {LABEL[impl]} | {r['seconds']:.3f} | {r['peak_mib']:.0f} | "
                              f"{_f(rel_rms(img, ref_t), '.2e')} | {_f(rel_rms(img, ref_k), '.2e')} |")
        md.append("")
    return "\n".join(md)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coarse", type=float, nargs="+", default=[2.0, 1.0, 0.5])
    ap.add_argument("--cones", nargs="+", default=["16x32", "32x64"])
    ap.add_argument("--fine-grids", type=float, nargs="+", default=[2.0, 1.0],
                    help="fine part: single-grid image grids (0.5 cm = the images part at factor 1)")
    ap.add_argument("--out", default=os.path.join("_cache", "runs", "bench_ray_tracing"))
    ap.add_argument("--parts", default="equality,chunks,images,fine")
    ap.add_argument("--torch-chunks", type=int, default=8, help="chunks timed for the PyTorch Siddon (one pass)")
    ap.add_argument("--torch-images", default="all", choices=["all", "none"],
                    help="run the PyTorch Siddon images (the 0.5 cm / 2048 one takes about 2 minutes)")
    ap.add_argument("--skip", default="", help="comma-separated record keys to skip")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--tables-only", action="store_true")
    args = ap.parse_args()
    args.skip = set(k for k in args.skip.split(",") if k)
    parts = set(p for p in args.parts.split(",") if p)
    cones = [tuple(int(v) for v in c.split("x")) for c in args.cones]
    torch.cuda.set_per_process_memory_fraction(CAP_MIB * 2 ** 20 / torch.cuda.get_device_properties(0).total_memory)
    b = Bench(args)
    print(f"nvidia-smi before: {foreign_gpu_load()}", flush=True)
    if not args.tables_only:
        b.warmup(args.coarse)
        if "equality" in parts:
            b.run_equality(args.coarse)
        if "chunks" in parts:
            for d in args.coarse:
                for cone in cones:
                    for impl in IMPLS:
                        b.run_chunks(d, cone, impl)
        if "images" in parts:
            for d in args.coarse:
                for cone in cones:
                    for impl in ("triton",) + (("torch",) if args.torch_images == "all" else ()):
                        b.run_image(d, cone, impl)
        if "fine" in parts:
            for d in args.fine_grids:
                grid, geom, rho, mu, phi = b.setup(d)
                q = panel_direction_set(geom, 0.0, (4, 8), max_polar_angle=grid, device=b.dev, dtype=torch.float32)
                for impl in IMPLS:  # warm-up of this grid's kernel variants
                    first_order_image(b.lib, rho, phi, grid, 1, geom, q, mu, MAKERS[impl](), b.det, b.grid_resp)
                for cone in cones:
                    for impl in ("triton",) + (("torch",) if args.torch_images == "all" else ()):
                        b.run_fine_image(d, cone, impl)
    md = tables(b, args.coarse, cones, parts)
    with open(os.path.join(args.out, "bench_ray_tracing.md"), "w") as fh:
        fh.write(md + "\n")
    print(md)
    print(f"nvidia-smi after: {foreign_gpu_load()}", flush=True)


if __name__ == "__main__":
    main()
