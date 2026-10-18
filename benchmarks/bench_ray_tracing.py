"""Benchmark of the fused Triton Siddon ray tracer against the PyTorch Siddon reference on the WLAB phantom.

Measures the agreement of every ray-tracing operation with the PyTorch float32 and float64 paths (part ``equality``),
the time and peak memory per chunk of directions (``chunks``), and the time, peak memory and detector signal of the
first-order scatter image, with the quadrature points on the fine grid (``images``) or on a single grid (``fine``).
The records, the images and a Markdown table are written to ``--out``.

Usage::

    python benchmarks/bench_ray_tracing.py [--coarse 2.0 1.0 0.5] [--cones 16x32 32x64]
        [--out _cache/runs/bench_ray_tracing] [--parts equality,chunks,images,fine] [--torch-chunks 8]
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
from citrus.scatter_solver.kernels import ray_tracing_kernels  # noqa: E402
from citrus.scatter_solver.ray_tracing import SiddonProjectors  # noqa: E402
from citrus.scatter_solver.cross_section_library import load_spectrum_txt  # noqa: E402
from citrus.scatter_solver.scatter_solver import build_solver_library  # noqa: E402
from citrus.scatter_solver.source import SourceModel  # noqa: E402
from citrus.scatter_solver.uncollided import uncollided_fluence  # noqa: E402
from citrus.scatter_solver.scatter_image import coarsen_mu_t, combined_scatter_image, panel_direction_set  # noqa: E402
from citrus.scatter_solver.cross_sections import mu_t_volume  # noqa: E402

SPECTRUM = load_spectrum_txt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo", "spectra",
                                          "spectrum_125kvp.txt"))  # the sample 125 kVp spectrum

CAP_MIB = 5000
FINE = 0.5
IMPLS = ("torch", "triton")
MAKERS: dict[str, Callable[[], object]] = {
    "torch": lambda: SiddonProjectors(backend="torch"),
    "triton": lambda: SiddonProjectors(backend="auto"),
}
LABEL = {"torch": "Siddon (PyTorch)", "triton": "Siddon (Triton)"}


def sync() -> None:
    """Wait for every CUDA kernel on the current device to finish."""
    torch.cuda.synchronize()


def library():
    """Cross-section library of the benchmark (6 groups, Legendre order 10), built once and cached on disk.

    Returns:
        MultigroupLibrary: the library.
    """
    return build_solver_library(SolverConfig(num_groups=6, legendre_order=10), SPECTRUM)  # cached on disk (_cache/)


def rel_rms(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    """Relative RMS difference ``norm(a - b) / norm(b)`` of two images.

    Args:
        a (ndarray or None): [Rd, Cd] image to compare.
        b (ndarray or None): [Rd, Cd] reference image.

    Returns:
        float or None: the relative difference, None when ``a`` or ``b`` is None.
    """
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def first_order_image(library, rho, phi_uncollided, grid, factor: int, geometry, quadrature, mu_coarse, projectors,
                      detector,
                      grid_response) -> dict:
    """First-order scatter image: ``combined_scatter_image`` with a zero multiple-collision source.

    Args:
        library (MultigroupLibrary): cross-section library.
        rho (Tensor): [M, Rq, Cq, Sq] densities in g/cm^3 on the quadrature grid.
        phi_uncollided (Tensor): [G, Rq, Cq, Sq] uncollided fluence on the quadrature grid.
        grid (Grid): the quadrature grid.
        factor (int): quadrature points per transport cell and axis.
        geometry (ConeGeometry): scanner geometry.
        quadrature (Quadrature): direction set of the scatter image.
        mu_coarse (Tensor or None): [G, R, C, S] total attenuation on the transport grid in cm^-1, or None for the
            cell average of the attenuation of ``rho``.
        projectors (SiddonProjectors): ray tracer.
        detector (DetectorResponse): detector response.
        grid_response (GridResponse): anti-scatter grid response.

    Returns:
        dict: the image dict of ``combined_scatter_image`` (``signal`` [Rd, Cd], ...).
    """
    T0 = torch.zeros(library.num_groups, 1, *grid.coarsen(factor).shape, device=phi_uncollided.device,
                     dtype=phi_uncollided.dtype)
    return combined_scatter_image(library, rho, phi_uncollided, grid, factor, T0, 0, geometry, 0.0, quadrature,
                                  mu_coarse, projectors, detector,
                                  grid_response, chunk=8, first_collision_chunk=4)


def foreign_gpu_load() -> str:
    """GPU utilization and used memory reported by nvidia-smi (this process included).

    Returns:
        str: the nvidia-smi CSV line, or ``"?"`` when nvidia-smi fails.
    """
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return out[0] if out else "?"
    except Exception:  # noqa: BLE001
        return "?"


class Bench:
    """Runner of the benchmark parts; every record is appended to ``records.jsonl`` and skipped when already there.

    Attributes:
        args (argparse.Namespace): the parsed command line.
        device (torch.device): the CUDA device.
        jsonl (str): path of the records file.
        records (dict): the records by key, loaded from ``jsonl`` unless ``--rerun``.
        library (MultigroupLibrary): cross-section library.
        detector (DetectorResponse): CsI 0.6 mm detector response.
        grid_response (GridResponse): anti-scatter grid response (no grid).
        setups (dict): the inputs of ``setup`` by voxel size.
    """

    def __init__(self, args):
        """Load the stored records and the cross-section library.

        Args:
            args (argparse.Namespace): the parsed command line.
        """
        self.args = args
        self.device = torch.device("cuda")
        os.makedirs(os.path.join(args.out, "img"), exist_ok=True)
        self.jsonl = os.path.join(args.out, "records.jsonl")
        self.records: dict[str, dict] = {}
        if os.path.isfile(self.jsonl) and not args.rerun:
            with open(self.jsonl) as file:
                for line in file:
                    if line.strip():
                        r = json.loads(line)
                        self.records[r["key"]] = r
        self.library = library()
        self.detector, self.grid_response = csi_response(0.06), no_grid()
        self.setups: dict[float, tuple] = {}

    # ---- inputs -----------------------------------------------------------------------------------
    def setup(self, d: float):
        """Inputs on the WLAB grid of voxel size ``d``, computed once.

        Args:
            d (float): voxel size in cm.

        Returns:
            tuple: (grid, geometry, rho [M, R, C, S] in g/cm^3, mu [G, R, C, S] in cm^-1, phi_uncollided [G, R, C, S]);
                phi_uncollided comes from the PyTorch Siddon upstream operator.
        """
        if d not in self.setups:
            grid, geometry, rho = wlab_setup(d, device=self.device, dtype=torch.float32)
            mu = mu_t_volume(self.library, rho)
            phi_uncollided = uncollided_fluence(self.library, rho, grid, geometry, 0.0, SourceModel(),
                                                SiddonProjectors(backend="torch"))
            self.setups[d] = (grid, geometry, rho, mu, phi_uncollided)
        return self.setups[d]

    def quadrature(self, cone: tuple[int, int]):
        """Direction set of the scatter image in the cone that covers the fine grid, at angle 0.

        Args:
            cone (tuple[int, int]): numbers of polar and azimuthal directions, e.g. (16, 32).

        Returns:
            Quadrature: the direction set.
        """
        grid, geometry = self.setup(FINE)[:2]
        return panel_direction_set(geometry, 0.0, cone, max_polar_angle=grid, device=self.device, dtype=torch.float32)

    def chunks(self, cone: tuple[int, int], chunk: int = 8) -> list[torch.Tensor]:
        """Panel-facing directions of the ``cone`` set, split into chunks.

        Args:
            cone (tuple[int, int]): numbers of polar and azimuthal directions.
            chunk (int): directions per chunk. Defaults to 8.

        Returns:
            list[Tensor]: [chunk, 3] float32 unit directions per chunk (the last one may be shorter).
        """
        geometry = self.setup(FINE)[1]
        directions = self.quadrature(cone).omega.to(self.device, torch.float32)
        _, _, n = geometry.panel_axes(0.0)
        directions = directions[(directions * (-torch.tensor(n, device=self.device))).sum(-1) > 0].contiguous()
        return [directions[s:s + chunk] for s in range(0, directions.shape[0], chunk)]

    # ---- records ----------------------------------------------------------------------------------
    def store(self, record: dict, images: Optional[dict] = None) -> None:
        """Append a record to ``records.jsonl`` and save its images.

        Args:
            record (dict): the record, with its ``key``.
            images (dict, optional): arrays saved to ``img/<key>.npz``. Defaults to None (no file).
        """
        if images is not None:
            np.savez_compressed(os.path.join(self.args.out, "img", record["key"] + ".npz"), **images)
        self.records[record["key"]] = record
        with open(self.jsonl, "a") as file:
            file.write(json.dumps(record) + "\n")

    def image(self, key: str) -> Optional[np.ndarray]:
        """Detector signal of a stored image.

        Args:
            key (str): record key.

        Returns:
            ndarray or None: [Rd, Cd] signal, None when the image file does not exist.
        """
        p = os.path.join(self.args.out, "img", key + ".npz")
        if not os.path.isfile(p):
            return None
        with np.load(p) as z:
            return z["signal"]

    def timed(self, function: Callable[[], object]) -> tuple[object, float, float]:
        """Run a function once, synchronized, after emptying the allocator cache and resetting the peak memory.

        Args:
            function (Callable): the function to run, without arguments.

        Returns:
            (result, seconds, peak_mib): the function's result, the wall time in s and the peak allocation in MiB.
        """
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        sync()
        t0 = time.perf_counter()
        r = function()
        sync()
        return r, time.perf_counter() - t0, torch.cuda.max_memory_allocated(self.device) / 2 ** 20

    def skip(self, key: str) -> bool:
        """Whether a record is already stored or listed in ``--skip``.

        Args:
            key (str): record key.

        Returns:
            bool: True to skip the record.
        """
        return key in self.records or key in self.args.skip

    # ---- part 1: equality ---------------------------------------------------------------------------
    def run_equality(self, coarse: list[float]) -> list[dict]:
        """Maximum relative deviation of the Triton operations from the PyTorch float32 and float64 paths.

        Covers ``downstream_cumulative`` on each coarse grid and ``oblique_project``, ``upstream_cumulative`` and
        ``primary_project`` on the fine grid, for one chunk of 8 directions of the 16x32 set.

        Args:
            coarse (list[float]): coarse voxel sizes in cm.

        Returns:
            list[dict]: one row per operation and backend (``op``, ``impl``, ``vs_torch32``, ``vs_torch64``).
        """
        key = "equality"
        if self.skip(key):
            return self.records[key]["rows"]
        grid_fine, geometry, rho_fine, mu_fine, _ = self.setup(FINE)
        directions = self.chunks((16, 32))[5]
        T = SiddonProjectors(backend="torch")
        rows = []

        def deviation_of(a: torch.Tensor, b: torch.Tensor) -> float:
            m = torch.isfinite(b)
            assert bool((torch.isfinite(a) == m).all())
            return float((a[m].double() - b[m].double()).abs().max() / b[m].double().abs().max())

        for d in coarse:
            f = int(round(d / FINE))
            grid_coarse = grid_fine.coarsen(f)
            mu_coarse_values = coarsen_mu_t(mu_fine, grid_fine, grid_coarse)
            reference = T.downstream_cumulative(mu_coarse_values, grid_coarse, geometry, 0.0, directions)
            result64 = T.downstream_cumulative(mu_coarse_values.double(), grid_coarse, geometry, 0.0,
                                               directions.double())
            got = MAKERS["triton"]().downstream_cumulative(mu_coarse_values, grid_coarse, geometry, 0.0, directions)
            rows.append({"op": f"downstream_cumulative ({d:g} cm)", "impl": "triton",
                         "vs_torch32": deviation_of(got, reference), "vs_torch64": deviation_of(got, result64)})
            rows.append({"op": f"downstream_cumulative ({d:g} cm)", "impl": "torch", "vs_torch32": 0.0,
                         "vs_torch64": deviation_of(reference, result64)})
            del reference, result64
        generator = torch.Generator(self.device).manual_seed(3)
        field = torch.rand(directions.shape[0], mu_fine.shape[0], *grid_fine.shape, device=self.device,
                           generator=generator)
        P = MAKERS["triton"]()
        ops = [("oblique_project (0.5 cm, [8, G] field)",
                lambda Q, dtype: Q.oblique_project(field.to(dtype), grid_fine, geometry, 0.0, directions.to(dtype))),
               ("upstream_cumulative (0.5 cm)",
                lambda Q, dtype: Q.upstream_cumulative(mu_fine.to(dtype), grid_fine, geometry, 0.0)),
               ("primary_project (0.5 cm)",
                lambda Q, dtype: Q.primary_project(mu_fine.to(dtype), grid_fine, geometry, 0.0))]
        for op, function in ops:
            reference = function(T, torch.float32)
            got = function(P, torch.float32)
            result64 = function(T, torch.float64)
            rows.append({"op": op, "impl": "triton", "vs_torch32": deviation_of(got, reference),
                         "vs_torch64": deviation_of(got, result64)})
            rows.append({"op": op, "impl": "torch", "vs_torch32": 0.0, "vs_torch64": deviation_of(reference, result64)})
            del reference, got, result64
        self.store({"key": key, "rows": rows})
        for r in rows:
            print(f"equality {r['op']} {r['impl']}: vs torch32 {r['vs_torch32']:.1e}, vs torch64 {r['vs_torch64']:.1e}",
                  flush=True)
        return rows

    # ---- part 2: per-chunk cost ---------------------------------------------------------------------
    def run_chunks(self, d: float, cone: tuple[int, int], impl: str) -> dict:
        """Time per chunk of ``downstream_cumulative`` (coarse grid) and ``oblique_project`` (fine grid), peak memory.

        Each time is the median of 3 passes over every panel-facing chunk after a warm pass; the PyTorch backend runs
        one pass over the first ``--torch-chunks`` chunks.

        Args:
            d (float): coarse voxel size in cm.
            cone (tuple[int, int]): numbers of polar and azimuthal directions.
            impl (str): backend, ``"torch"`` or ``"triton"``.

        Returns:
            dict: the record (``down_ms``, ``obl_ms``, ``pair_ms``, ``pair_peak_extra_mib``); stored one if skipped.
        """
        key = f"chunks_c{d:g}_q{cone[0]}x{cone[1]}_{impl}"
        if self.skip(key):
            return self.records.get(key, {})
        grid_fine, geometry, _, mu_fine = self.setup(FINE)[:4]
        f = int(round(d / FINE))
        grid_coarse = grid_fine.coarsen(f)
        mu_coarse_values = coarsen_mu_t(mu_fine, grid_fine, grid_coarse)
        chunks = self.chunks(cone)
        G = mu_fine.shape[0]
        generator = torch.Generator(self.device).manual_seed(3)
        field = torch.rand(8, G, *grid_fine.shape, device=self.device, generator=generator)
        P = MAKERS[impl]()
        reps = 3
        if impl == "torch":
            chunks = chunks[:self.args.torch_chunks]
            reps = 1

        def downstream():
            for o in chunks:
                P.downstream_cumulative(mu_coarse_values, grid_coarse, geometry, 0.0, o)

        def oblique():
            for o in chunks:
                P.oblique_project(field[:o.shape[0]], grid_fine, geometry, 0.0, o)

        def pair():
            for o in chunks:
                t = P.downstream_cumulative(mu_coarse_values, grid_coarse, geometry, 0.0, o)
                P.oblique_project(field[:o.shape[0]], grid_fine, geometry, 0.0, o)
                del t

        record = {"key": key, "coarse": d, "cone": list(cone), "impl": impl, "chunks": len(chunks),
                  "gpu": foreign_gpu_load()}
        for name, function in (("down", downstream), ("obl", oblique), ("pair", pair)):
            function()
            times = []
            for _ in range(reps):
                sync()
                t0 = time.perf_counter()
                function()
                sync()
                times.append(time.perf_counter() - t0)
            record[f"{name}_ms"] = 1e3 * sorted(times)[len(times) // 2] / len(chunks)
        gc.collect()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        o = chunks[0]
        t = P.downstream_cumulative(mu_coarse_values, grid_coarse, geometry, 0.0, o)
        P.oblique_project(field[:o.shape[0]], grid_fine, geometry, 0.0, o)
        del t
        sync()
        record["pair_peak_extra_mib"] = (torch.cuda.max_memory_allocated(self.device) - base) / 2 ** 20
        self.store(record)
        print(f"chunks {d:g} cm {cone} {impl}: " + ", ".join(f"{k} {v:.3f}" for k, v in record.items()
                                                             if k.endswith("_ms") or k.endswith("_mib"))
              + f" (nvidia-smi: {record['gpu']})", flush=True)
        return record

    # ---- part 3: first-order image (supervoxel, sub-points on the fine grid) -------------------------
    def first_order(self, d: float, quadrature, impl: str) -> dict:
        """First-order scatter image on the transport grid of voxel size ``d``, quadrature points on the fine grid.

        Args:
            d (float): transport voxel size in cm, a multiple of the fine 0.5 cm.
            quadrature (Quadrature): direction set.
            impl (str): backend, ``"torch"`` or ``"triton"``.

        Returns:
            dict: the image dict of ``combined_scatter_image`` (``signal`` [Rd, Cd], ...).
        """
        grid_fine, geometry, rho_fine, _, phi_fine = self.setup(FINE)
        f = int(round(d / FINE))
        return first_order_image(self.library, rho_fine, phi_fine, grid_fine, f, geometry, quadrature, None,
                                 MAKERS[impl](),
                                 self.detector,
                                 self.grid_response)

    def run_image(self, d: float, cone: tuple[int, int], impl: str) -> dict:
        """Time, peak memory and detector signal of the first-order image with the quadrature points on the fine grid.

        Args:
            d (float): transport voxel size in cm.
            cone (tuple[int, int]): numbers of polar and azimuthal directions.
            impl (str): backend, ``"torch"`` or ``"triton"``.

        Returns:
            dict: the record (``seconds``, ``peak_mib``, ``dirs``); the stored one when skipped.
        """
        key = f"img_c{d:g}_q{cone[0]}x{cone[1]}_{impl}"
        if self.skip(key):
            return self.records.get(key, {})
        q = self.quadrature(cone)
        print(f"[{time.strftime('%H:%M:%S')}] {key} ({q.num_directions} dirs)", flush=True)
        r, secs, peak = self.timed(lambda: self.first_order(d, q, impl))
        record = {"key": key, "coarse": d, "cone": list(cone), "impl": impl, "seconds": secs, "peak_mib": peak,
                  "dirs": q.num_directions, "gpu": foreign_gpu_load()}
        self.store(record, {"signal": r["signal"].double().cpu().numpy()})
        print(f"    {secs:.3f} s, peak {peak:.0f} MiB (nvidia-smi: {record['gpu']})", flush=True)
        return record

    # ---- part 4: single-grid first-order image --------------------------------------------------------
    def run_fine_image(self, d: float, cone: tuple[int, int], impl: str) -> dict:
        """Time, peak memory and detector signal of the first-order image on a single grid (one point per cell).

        Args:
            d (float): voxel size in cm.
            cone (tuple[int, int]): numbers of polar and azimuthal directions.
            impl (str): backend, ``"torch"`` or ``"triton"``.

        Returns:
            dict: the record (``seconds``, ``peak_mib``, ``dirs``); the stored one when skipped.
        """
        key = f"fine_d{d:g}_q{cone[0]}x{cone[1]}_{impl}"
        if self.skip(key):
            return self.records.get(key, {})
        grid, geometry, rho, mu, phi = self.setup(d)
        q = panel_direction_set(geometry, 0.0, cone, max_polar_angle=grid, device=self.device, dtype=torch.float32)
        print(f"[{time.strftime('%H:%M:%S')}] {key} ({q.num_directions} dirs)", flush=True)
        r, secs, peak = self.timed(lambda: first_order_image(self.library, rho, phi, grid, 1, geometry, q, mu,
                                                             MAKERS[impl](), self.detector, self.grid_response))
        record = {"key": key, "fine": d, "cone": list(cone), "impl": impl, "seconds": secs, "peak_mib": peak,
                  "dirs": q.num_directions, "gpu": foreign_gpu_load()}
        self.store(record, {"signal": r["signal"].double().cpu().numpy()})
        print(f"    {secs:.3f} s, peak {peak:.0f} MiB (nvidia-smi: {record['gpu']})", flush=True)
        return record

    def warmup(self, coarse: list[float]) -> None:
        """Untimed small runs of every backend (CUDA context, allocator, every Triton kernel variant).

        Args:
            coarse (list[float]): coarse voxel sizes in cm.
        """
        t0 = time.perf_counter()
        ray_tracing_kernels.warmup(self.device, int(self.library.num_groups))
        q = self.quadrature((4, 8))
        for d in coarse:
            for impl in IMPLS:
                self.first_order(d, q, impl)
        sync()
        print(f"warm-up: {time.perf_counter() - t0:.1f} s; siddon kernels: "
              f"{ray_tracing_kernels.kernels_usable(self.device)}", flush=True)


# =====================================================================================================
# tables
# =====================================================================================================

def _f(x, fmt: str) -> str:
    """Format a value, or give an empty string for None.

    Args:
        x (float or None): the value.
        fmt (str): format specification, e.g. ``".3f"``.

    Returns:
        str: the formatted value.
    """
    return "" if x is None else format(x, fmt)


def tables(b: Bench, coarse: list[float], cones: list[tuple[int, int]], parts: set[str]) -> str:
    """Markdown tables of the stored records of the selected parts.

    Args:
        b (Bench): the benchmark runner with its records.
        coarse (list[float]): coarse voxel sizes in cm.
        cones (list[tuple[int, int]]): numbers of polar and azimuthal directions of each direction set.
        parts (set[str]): parts to tabulate.

    Returns:
        str: the Markdown text.
    """
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
        md += ["### First-order image (supervoxel tau on the coarse grid, sub-points projected on the fine 0.5 cm "
               "grid)", "",
               "| coarse grid | dirs | backend | image s | peak MiB | rel RMS vs Siddon (PyTorch) | "
               "rel RMS vs Siddon (Triton) |", "|---|---|---|---|---|---|---|"]
        for d in coarse:
            for cone in cones:
                reference_t = b.image(f"img_c{d:g}_q{cone[0]}x{cone[1]}_torch")
                reference_k = b.image(f"img_c{d:g}_q{cone[0]}x{cone[1]}_triton")
                for impl in IMPLS:
                    key = f"img_c{d:g}_q{cone[0]}x{cone[1]}_{impl}"
                    r = R.get(key)
                    if not r:
                        continue
                    image = b.image(key)
                    md.append(f"| {d:g} cm | {r['dirs']} | {LABEL[impl]} | {r['seconds']:.3f} | {r['peak_mib']:.0f} | "
                              f"{_f(rel_rms(image, reference_t), '.2e')} | {_f(rel_rms(image, reference_k), '.2e')} |")
        md.append("")
    if "fine" in parts:
        md += ["### First-order image on a single grid (q = 1)", "",
               "| grid | dirs | backend | image s | peak MiB | rel RMS vs Siddon (PyTorch) | "
               "rel RMS vs Siddon (Triton) |",
               "|---|---|---|---|---|---|---|"]
        for d in sorted({float(x) for x in b.args.fine_grids}, reverse=True):
            for cone in cones:
                reference_t = b.image(f"fine_d{d:g}_q{cone[0]}x{cone[1]}_torch")
                reference_k = b.image(f"fine_d{d:g}_q{cone[0]}x{cone[1]}_triton")
                for impl in IMPLS:
                    key = f"fine_d{d:g}_q{cone[0]}x{cone[1]}_{impl}"
                    r = R.get(key)
                    if not r:
                        continue
                    image = b.image(key)
                    md.append(f"| {d:g} cm | {r['dirs']} | {LABEL[impl]} | {r['seconds']:.3f} | {r['peak_mib']:.0f} | "
                              f"{_f(rel_rms(image, reference_t), '.2e')} | {_f(rel_rms(image, reference_k), '.2e')} |")
        md.append("")
    return "\n".join(md)


def main() -> None:
    """Parse the command line, run the selected parts, then write and print the tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coarse", type=float, nargs="+", default=[2.0, 1.0, 0.5])
    parser.add_argument("--cones", nargs="+", default=["16x32", "32x64"])
    parser.add_argument("--fine-grids", type=float, nargs="+", default=[2.0, 1.0],
                        help="fine part: single-grid image grids (0.5 cm = the images part at factor 1)")
    parser.add_argument("--out", default=os.path.join("_cache", "runs", "bench_ray_tracing"))
    parser.add_argument("--parts", default="equality,chunks,images,fine")
    parser.add_argument("--torch-chunks", type=int, default=8, help="chunks timed for the PyTorch Siddon (one pass)")
    parser.add_argument("--torch-images", default="all", choices=["all", "none"],
                        help="run the PyTorch Siddon images (the 0.5 cm / 2048 one takes about 2 minutes)")
    parser.add_argument("--skip", default="", help="comma-separated record keys to skip")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
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
                grid, geometry, rho, mu, phi = b.setup(d)
                q = panel_direction_set(geometry, 0.0, (4, 8), max_polar_angle=grid, device=b.device,
                                        dtype=torch.float32)
                for impl in IMPLS:  # warm-up of this grid's kernel variants
                    first_order_image(b.library, rho, phi, grid, 1, geometry, q, mu, MAKERS[impl](), b.detector,
                                      b.grid_response)
                for cone in cones:
                    for impl in ("triton",) + (("torch",) if args.torch_images == "all" else ()):
                        b.run_fine_image(d, cone, impl)
    md = tables(b, args.coarse, cones, parts)
    with open(os.path.join(args.out, "bench_ray_tracing.md"), "w") as file:
        file.write(md + "\n")
    print(md)
    print(f"nvidia-smi after: {foreign_gpu_load()}", flush=True)


if __name__ == "__main__":
    main()
