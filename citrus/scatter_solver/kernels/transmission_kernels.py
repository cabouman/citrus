"""Fused Triton kernel of the group-integrated transmission along material line integrals.

``_group_transmission_kernel`` forms ``T_g = sum_{e in g} w_e exp(-sum_m mu_m(e) P_m)`` for a block of rays per
program, over the fine energies of each group in turn (the library sorts them by group), so no [energies, rays]
temporary is formed. ``group_transmission`` is the host wrapper of ``cross_sections.group_transmission`` and
``kernels_usable`` checks the kernel against its PyTorch path once per device.
"""

from __future__ import annotations

import weakref
from typing import Optional, Sequence

import torch
from torch import Tensor

from . import triton_utilities

SELF_CHECK_REL_TOL = 1e-5
BLOCK = 128
NUM_WARPS = 4

_SELF_CHECK: dict[str, tuple[bool, str]] = {}
# fine-energy tables: id(library.fine_mu_t) -> (weak reference to it, {(device, versions): FineTables})
_TABLES: dict[int, tuple[weakref.ref, dict[tuple, "FineTables"]]] = {}

triton = triton_utilities.triton
tl = triton_utilities.tl


def kernels_usable(device: torch.device) -> tuple[bool, str]:
    """Report whether Triton is available and the kernel reproduces the PyTorch path on ``device``.

    The self-check runs once per device on a small random problem; the verdict is cached and the function never
    raises.

    Args:
        device (torch.device): the device to check.

    Returns:
        (bool, str): whether the kernel is usable, and the reason.
    """
    ok, why = triton_utilities.triton_available()
    if not ok:
        return ok, why
    device = torch.device(device)
    if device.type != "cuda":
        return False, "not a CUDA device"
    key = str(device)
    if key not in _SELF_CHECK:
        try:
            _SELF_CHECK[key] = _self_check(device)
        except Exception as e:  # noqa: BLE001
            _SELF_CHECK[key] = (False, f"self-check failed ({type(e).__name__}: {str(e)[:300]})")
    return _SELF_CHECK[key]


def use_kernel(library, path: Tensor) -> bool:
    """Decide whether ``cross_sections.group_transmission`` runs the kernel.

    Args:
        library (MultigroupLibrary): the library; its fine energies must be sorted by group.
        path (Tensor): [M', ...] line integrals.

    Returns:
        bool: True for float32 CUDA line integrals, a library with its fine energies sorted by group and a usable
        kernel.
    """
    if path.device.type != "cuda" or path.dtype != torch.float32 or not kernels_usable(path.device)[0]:
        return False
    return _fine_tables(library, path.device).sorted_by_group


def _self_check(device: torch.device) -> tuple[bool, str]:
    """Compare the kernel with the PyTorch path of ``cross_sections.group_transmission`` on a random problem with a
    subset of the materials, a zero path and a long path.

    Args:
        device (torch.device): the CUDA device to check.

    Returns:
        (bool, str): whether the largest relative deviation is within ``SELF_CHECK_REL_TOL``, and a message.
    """
    from ..cross_sections import _group_transmission_torch, synthetic_library

    library = synthetic_library(4, num_materials=5)
    generator = torch.Generator().manual_seed(3)
    materials = [0, 2, 3]
    path = (torch.rand(len(materials), 7, 9, generator=generator) * 20.0).to(device)
    path[:, 0, 0] = 0.0
    path[:, 1, 1] = 200.0
    reference = _group_transmission_torch(library, path, 2 ** 22, materials)
    out = group_transmission(library, path, materials)
    torch.cuda.synchronize(device)
    worst = float((out - reference).abs().max()) / max(float(reference.abs().max()), 1e-30)
    if out.shape != reference.shape or worst > SELF_CHECK_REL_TOL:
        return False, f"self-check: transmission kernel deviates from the PyTorch path by {worst:.2e} (rel)"
    return True, f"transmission kernel matches the PyTorch path (max rel dev {worst:.1e})"


class FineTables:
    """The fine-energy tables of a library on one device, as the kernel reads them.

    Attributes:
        mu (Tensor): [M, E] float32 fine mass attenuation in cm^2/g.
        weight (Tensor): [E] float32 fine weights.
        starts (Tensor): [G + 1] int32 first fine energy of each group, and E.
        sorted_by_group (bool): whether the fine energies are sorted by group (the kernel needs it).
        rows (dict): [M', E] attenuation rows per tuple of material indices, gathered on first use.
    """

    def __init__(self, library, device: torch.device) -> None:
        """Lay out the tables of ``library`` on ``device``.

        Args:
            library (MultigroupLibrary): the library.
            device (torch.device): the CUDA device.
        """
        fine_group = library.fine_group.to("cpu")
        self.sorted_by_group = bool(fine_group.numel() < 2 or (fine_group[1:] >= fine_group[:-1]).all())
        self.mu = library.fine_mu_t.to(device=device, dtype=torch.float32).contiguous()
        self.weight = library.fine_weight.to(device=device, dtype=torch.float32).contiguous()
        counts = torch.bincount(fine_group, minlength=library.num_groups)
        self.starts = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)]).to(device=device,
                                                                                         dtype=torch.int32)
        self.rows: dict[tuple[int, ...], Tensor] = {}

    def material_rows(self, indices: tuple[int, ...]) -> Tensor:
        """The [M', E] attenuation rows of the materials ``indices``, cached.

        Args:
            indices (tuple[int, ...]): library indices of the materials.

        Returns:
            Tensor: [M', E] float32 rows.
        """
        rows = self.rows.get(indices)
        if rows is None:
            rows = self.mu[list(indices)].contiguous()
            self.rows[indices] = rows
        return rows


def _fine_tables(library, device: torch.device) -> FineTables:
    """The library's fine tables on ``device``, cached per tensor, device and tensor versions.

    Args:
        library (MultigroupLibrary): the library.
        device (torch.device): the CUDA device.

    Returns:
        FineTables: the tables.
    """
    source = library.fine_mu_t
    key = (str(torch.device(device)), int(source._version), int(library.fine_weight._version),
           int(library.fine_group._version), id(library.fine_weight), id(library.fine_group))
    hit = _TABLES.get(id(source))
    if hit is not None and hit[0]() is source and key in hit[1]:
        return hit[1][key]
    entry = FineTables(library, device)
    source_id = id(source)
    if hit is None or hit[0]() is not source:
        _TABLES[source_id] = (weakref.ref(source, lambda _r, source_id=source_id, tables=_TABLES:
                                          tables.pop(source_id, None)), {})
    per_version = _TABLES[source_id][1]
    per_version.clear()  # keep only the latest version per source tensor
    per_version[key] = entry
    return entry


if triton is not None:

    try:  # the accurate exponential of libdevice (tl.exp lowers to the approximate ex2)
        from triton.language.extra import libdevice as _libdevice

        _exp = _libdevice.exp
    except Exception:  # noqa: BLE001
        _exp = tl.exp

    @triton.jit(do_not_specialize=["R", "E", "ps_m"],
                do_not_specialize_on_alignment=["path_ptr", "mu_ptr", "weight_ptr", "start_ptr", "out_ptr"])
    def _group_transmission_kernel(path_ptr, mu_ptr, weight_ptr, start_ptr, out_ptr, R, E, ps_m,
                                   G: tl.constexpr, M: tl.constexpr, BLOCK: tl.constexpr):
        """Group-integrated transmission of BLOCK rays (program_id 0).

        Args:
            path_ptr: [M, R] float32 line integrals in g/cm^2, row stride ``ps_m``.
            mu_ptr: [M, E] float32 fine mass attenuation in cm^2/g of the traced materials.
            weight_ptr: [E] float32 fine weights.
            start_ptr: [G + 1] int32 first fine energy of each group, and E.
            out_ptr: [G, R] float32 transmission.
            R: number of rays.
            E: number of fine energies.
            ps_m: element stride of the material axis of the line integrals.
            G (constexpr): number of groups.
            M (constexpr): number of traced materials.
            BLOCK (constexpr): rays per program.
        """
        r = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        rm = r < R
        for g in tl.static_range(G):
            e_start = tl.load(start_ptr + g)
            e_end = tl.load(start_ptr + g + 1)
            acc = tl.zeros((BLOCK,), dtype=tl.float32)
            for e in range(e_start, e_end):
                tau = tl.zeros((BLOCK,), dtype=tl.float32)
                for m in tl.static_range(M):  # in row order, as the PyTorch path sums them
                    tau += tl.load(mu_ptr + m * E + e) * tl.load(path_ptr + m * ps_m + r, mask=rm, other=0.0)
                acc += tl.load(weight_ptr + e) * _exp(-tau)
            tl.store(out_ptr + g * R + r, acc, mask=rm)


def group_transmission(library, path: Tensor, materials: Optional[Sequence[int]] = None) -> Tensor:
    """Group-integrated transmission on float32 CUDA line integrals, without availability check.

    The fused form of ``cross_sections.group_transmission``; the library's fine energies must be sorted by group.

    Args:
        library (MultigroupLibrary): the cross-section library.
        path (Tensor): [M', ...] line integrals of the density maps in g/cm^2.
        materials (Sequence[int] or None, optional): library indices of the rows of ``path``. Defaults to None (all, in
            library order).

    Returns:
        Tensor: [G, ...] transmission per group.
    """
    device = path.device
    tables = _fine_tables(library, device)
    M = int(path.shape[0])
    indices = tuple(range(M)) if materials is None else tuple(int(i) for i in materials)
    if len(indices) != M or any(i < 0 or i >= tables.mu.shape[0] for i in indices):
        raise ValueError(f"path has {M} materials, materials {list(indices)}, the library {tables.mu.shape[0]}")
    mu = tables.material_rows(indices)
    spatial = tuple(path.shape[1:])
    paths = path.reshape(M, -1)
    if paths.stride(-1) != 1:
        paths = paths.contiguous()
    R = int(paths.shape[1])
    G = int(tables.starts.shape[0]) - 1
    out = torch.empty(G, R, device=device, dtype=torch.float32)
    if R == 0:
        return out.reshape(G, *spatial)
    triton_utilities.launch(_group_transmission_kernel, (triton.cdiv(R, BLOCK), 1, 1),
                            (paths, mu, tables.weight, tables.starts, out, R, int(mu.shape[1]), paths.stride(0)),
                            dict(G=G, M=M, BLOCK=BLOCK), NUM_WARPS)
    return out.reshape(G, *spatial)
