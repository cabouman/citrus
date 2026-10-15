"""Benchmark the LD sweep and one source iteration on WLAB-like coarse grids (CUDA, float32).

Usage:  python dev_scripts/bench_transport_sweep.py [--grids 20x30x40,40x60x80] [--chunks 16,8] [--repeats 5]
                                                  [--variants torch,triton] [--mem-cap-mib 5000] [--no-si]

For each grid (rows x cols x slices, 12.5 mm cells), each direction chunk (directions per octant swept
together; D8 = 16) and each variant (plane-step backend "torch" / "triton") it reports, for G = 6, L = 7,
product_quadrature(8, 16) = 128 directions:

  ms/sweep   best-of-`repeats` wall time of one full `sweep` call (ONE source_fn call per chunk for all 8 octants,
             writing into the lent storage buffer)
  ms/SI-it   one SI iteration: scatter_moments + scatter_source_fn + sweep
  breakdown  of one sweep, CUDA-event timed per phase and summed over the chunks (same code path as
             `sweep`, instrumented): source evaluation (source_fn + cell-major copy, `_fill_source`), the plane
             loop (`_sweep_chunk`), the moment GEMM (`addmm_`), and the rest (allocation, the final permute of phi)
  peak MB    torch.cuda.max_memory_allocated during the timed sweeps / SI iterations, including the problem
             data (mu_t, transfer moments, the source moments T held by the source function)

PyTorch memory is capped with torch.cuda.set_per_process_memory_fraction (--mem-cap-mib, default 5000 MiB):
on Windows, VRAM oversubscription silently spills into system memory and runs orders of magnitude slower.
Configurations whose estimated footprint exceeds the cap are skipped; an allocator OOM is reported as such.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import torch

from citrus.scatter_solver.angular_quadrature import product_quadrature
from citrus.scatter_solver.geometry import Grid
from citrus.scatter_solver.source_iteration import scatter_moments, scatter_source_fn
from citrus.scatter_solver.transport_sweep import (
    SweepContext,
    _fill_source,
    _resolve_chunk,
    _sweep_chunk,
    make_sweep_context,
    source_fn_accepts_out,
    sweep,
)
from citrus.scatter_solver.cross_sections import mu_t_volume, synthetic_library, transfer_moments_volume

ALL_VARIANTS = ("torch", "triton")


def _time(fn: Callable[[], object], repeats: int) -> float:
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def field_bytes(shape: tuple[int, int, int], G: int = 6, L: int = 7, itemsize: int = 4) -> int:
    """One SH/LD moment field [G, (L+1)^2, 4, N]."""
    return itemsize * G * (L + 1) ** 2 * 4 * shape[0] * shape[1] * shape[2]


def sweep_buffer_bytes(shape: tuple[int, int, int], chunk: int, G: int = 6, itemsize: int = 4) -> int:
    """store + source of one chunk: 2 x [8, Dc, N, 4G]."""
    return itemsize * 2 * 8 * chunk * shape[0] * shape[1] * shape[2] * 4 * G


def estimate_bytes(shape: tuple[int, int, int], chunk: int, G: int = 6, L: int = 7, itemsize: int = 4,
                   si: bool = True) -> int:
    """Rough peak of the benchmark. Sweep: problem data (transfer moments, T of the source function) + the
    accumulator + the sweep buffers. SI iteration: phi_n, T, the accumulator (3 fields) plus the larger of the
    sweep buffers and the final permuted phi, plus the transfer moments."""
    N = shape[0] * shape[1] * shape[2]
    F = field_bytes(shape, G, L, itemsize)
    B = sweep_buffer_bytes(shape, chunk, G, itemsize)
    transfer = itemsize * G * G * (L + 1) * N
    if si:
        return 3 * F + max(B, F) + transfer
    return 2 * F + B + transfer


def breakdown(ctx: SweepContext, mu_t: torch.Tensor, source_fn, chunk: int, use_triton: bool) -> dict:
    """One sweep, phase by phase, mirroring `sweep` (CUDA-event timed). Returns ms per phase."""
    lend = source_fn_accepts_out(source_fn)
    G = int(mu_t.shape[0])
    Q, N, D8, Nsh = 4 * G, ctx.num_cells, ctx.directions_per_octant, ctx.num_moments
    dtype, device = mu_t.dtype, mu_t.device
    Dc = _resolve_chunk(ctx, chunk)
    ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    mu_cells = mu_t.reshape(G, N).t().contiguous()
    acc = torch.zeros(Nsh, N * Q, device=device, dtype=dtype)
    Yw = ctx.basis.Yw.to(device=device, dtype=dtype)
    c_all = ctx.c.to(device=device, dtype=dtype)
    s_all = ctx.s.to(device=device, dtype=dtype)
    marks = []
    for d0 in range(0, D8, Dc):
        d1 = min(d0 + Dc, D8)
        dc = d1 - d0
        idx = ctx.direction_indices(d0, d1)
        c = c_all[:, d0:d1].contiguous()
        s = s_all[:, d0:d1].contiguous()
        e = [ev() for _ in range(4)]
        e[0].record()
        src = torch.empty(8, dc, N, Q, device=device, dtype=dtype)
        store = torch.empty(8, dc, N + 1, Q, device=device, dtype=dtype) if lend else None
        _fill_source(ctx, source_fn, d0, d1, G, src, scratch=store)
        e[1].record()
        store = _sweep_chunk(ctx, src, mu_cells, c, s, store=store, use_triton=use_triton)
        del src
        e[2].record()
        psi_c = store.view(8 * dc, N + 1, Q)[:, :N]
        acc.addmm_(Yw.index_select(0, idx).t(), psi_c.reshape(8 * dc, N * Q))
        e[3].record()
        marks.append(e)
        del store, psi_c
    phi = acc.view(Nsh, N, G, 4).permute(2, 0, 3, 1).reshape(G, Nsh, 4, *ctx.shape)
    torch.cuda.synchronize()
    total = (time.perf_counter() - t_start) * 1e3
    del phi, acc
    out = {"source": 0.0, "planes": 0.0, "gemm": 0.0}
    for e in marks:
        out["source"] += e[0].elapsed_time(e[1])
        out["planes"] += e[1].elapsed_time(e[2])
        out["gemm"] += e[2].elapsed_time(e[3])
    out["other"] = max(total - out["source"] - out["planes"] - out["gemm"], 0.0)
    out["total"] = total
    return out


def bench(shape: tuple[int, int, int], chunk: int, repeats: int, variants: list[str], do_si: bool,
          cap_bytes: int) -> list[dict]:
    dev = torch.device("cuda")
    dtype = torch.float32
    G, L = 6, 7
    lib = synthetic_library(num_groups=G, legendre_order=L)
    grid = Grid(shape, 1.25)
    gen = torch.Generator().manual_seed(0)
    rho = torch.zeros(2, *shape, dtype=torch.float64)
    rho[0] = 1.0 + 0.1 * torch.rand(*shape, generator=gen, dtype=torch.float64)
    rho = rho.to(dev, dtype)
    mu_t = mu_t_volume(lib, rho)
    mu_s = transfer_moments_volume(lib, rho)
    q = product_quadrature(8, 16, dtype=torch.float64)
    ctx = make_sweep_context(grid, q, L, device=dev, dtype=dtype, backend="torch")
    Nsh = ctx.num_moments
    phi_prev = torch.rand(G, Nsh, 4, *shape, device=dev, dtype=dtype)
    T = scatter_moments(mu_s, phi_prev)
    del phi_prev
    src_fn = scatter_source_fn(ctx, T)
    del T
    si_fits = do_si and estimate_bytes(shape, chunk) <= cap_bytes
    rows = []
    for backend in variants:
        r = {"shape": shape, "cells": ctx.num_cells, "chunk": chunk, "planes": ctx.num_planes,
             "chunks": -(-ctx.directions_per_octant // chunk), "variant": backend}
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            run = lambda: sweep(ctx, mu_t, src_fn, direction_chunk=chunk, backend=backend)  # noqa: E731
            run()  # warm-up (and Triton compile)
            r["ms_sweep"] = _time(run, repeats)
            r["peak_sweep_MB"] = torch.cuda.max_memory_allocated() / 2**20
            bd = min((breakdown(ctx, mu_t, src_fn, chunk, backend == "triton") for _ in range(repeats)),
                     key=lambda d: d["total"])
            r.update({f"bd_{k}": v for k, v in bd.items()})
        except torch.cuda.OutOfMemoryError:
            r["oom_sweep"] = True
        rows.append(r)
    if si_fits:
        del src_fn
        torch.cuda.empty_cache()
        phi_n = torch.rand(G, Nsh, 4, *shape, device=dev, dtype=dtype)
        for r in rows:
            backend = r["variant"]
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                def si_iteration():
                    fn = scatter_source_fn(ctx, scatter_moments(mu_s, phi_n))
                    return sweep(ctx, mu_t, fn, direction_chunk=chunk, backend=backend)

                si_iteration()
                r["ms_si"] = _time(si_iteration, repeats)
                r["peak_si_MB"] = torch.cuda.max_memory_allocated() / 2**20
            except torch.cuda.OutOfMemoryError:
                r["oom_si"] = True
    return rows


def _f(r: dict, k: str, fmt: str = "{:.1f}") -> str:
    if k in r:
        return fmt.format(r[k])
    if k.endswith("_si") or k == "peak_si_MB":
        return "OOM" if r.get("oom_si") else "-"
    return "OOM" if r.get("oom_sweep") else "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="20x30x40,40x60x80")
    ap.add_argument("--chunks", default="16,8")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument("--mem-cap-mib", type=float, default=5000.0)
    ap.add_argument("--no-si", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    variants = [v.strip() for v in args.variants.split(",")]
    for v in variants:
        if v not in ALL_VARIANTS:
            raise SystemExit(f"unknown variant {v!r}; choose from {ALL_VARIANTS}")
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    # never cap above what is physically free (other processes' VRAM would otherwise spill to system memory)
    cap = int(min(args.mem_cap_mib * 2**20, free - 256 * 2**20))
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total), 0)
    print(f"free VRAM at start {free / 2**20:.0f} MiB; effective PyTorch cap {cap / 2**20:.0f} MiB")
    if "triton" in variants:
        from citrus.scatter_solver.kernels import sweep_kernels as sweep_triton

        ok, why = sweep_triton.kernels_usable(torch.device("cuda"))
        print(f"triton: {ok} ({why})")
        if not ok:
            variants = [v for v in variants if v != "triton"]
    print(f"device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    header = (f"{'grid':>10} {'cells':>7} {'Dc':>3} {'chunks':>6} {'planes':>6} {'variant':>13} {'ms/sweep':>9}"
              f" {'source':>7} {'planes':>7} {'gemm':>6} {'other':>6} {'ms/SI-it':>9} {'peak sweep MB':>13}"
              f" {'peak SI MB':>10}")
    print(header)
    for gs in args.grids.split(","):
        shape = tuple(int(v) for v in gs.lower().split("x"))
        for chunk in (int(v) for v in args.chunks.split(",")):
            need = estimate_bytes(shape, chunk, si=False)
            if need > cap:
                print(f"{gs:>10} Dc={chunk}: skipped, estimated sweep footprint {need / 2**20:.0f} MiB > cap")
                continue
            try:
                rows = bench(shape, chunk, args.repeats, variants, not args.no_si, cap)
            except torch.cuda.OutOfMemoryError:
                print(f"{gs:>10} Dc={chunk}: out of memory")
                torch.cuda.empty_cache()
                continue
            for r in rows:
                print(f"{gs:>10} {r['cells']:>7} {r['chunk']:>3} {r['chunks']:>6} {r['planes']:>6} {r['variant']:>13}"
                      f" {_f(r, 'ms_sweep'):>9} {_f(r, 'bd_source'):>7} {_f(r, 'bd_planes'):>7} {_f(r, 'bd_gemm'):>6}"
                      f" {_f(r, 'bd_other'):>6} {_f(r, 'ms_si'):>9} {_f(r, 'peak_sweep_MB', '{:.0f}'):>13}"
                      f" {_f(r, 'peak_si_MB', '{:.0f}'):>10}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
