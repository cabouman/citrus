"""Fused Triton kernel of the spherical-harmonic moments of the scattering source.

``_scattering_moments_kernel`` forms ``T[g, n, k, x] = sum_g' mu_s_l[g', g, l(n), x] phi[g', n, k, x]`` for one moment n
and a block of points per program, every outgoing group at once. It reads the flux moments and writes T through
element strides, so that both can be views of any layout, the sweep's cell-major moments in particular.
``scattering_moments`` is the host wrapper and ``kernels_usable`` checks the kernel against the PyTorch path of
``angular_quadrature.scatter_source_moments`` once per device.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from . import triton_utilities

SELF_CHECK_REL_TOL = 1e-5
BLOCK = 128
NUM_WARPS = 4
LAYOUTS = ("contiguous", "moment_major", "cell_major")

_SELF_CHECK: dict[str, tuple[bool, str]] = {}

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


def use_kernel(transfer: Tensor, phi: Tensor) -> bool:
    """Decide whether ``scatter_source_moments`` runs the kernel: float32 CUDA tensors and a usable kernel.

    Args:
        transfer (Tensor): ``[G_in, G_out, L+1, *spatial]`` Legendre transfer moments.
        phi (Tensor): ``[G_in, N, *rest]`` flux moments.

    Returns:
        bool: True when the kernel runs.
    """
    return (transfer.device.type == "cuda" and phi.device == transfer.device and transfer.dtype == torch.float32
            and phi.dtype == torch.float32 and kernels_usable(transfer.device)[0])


def _self_check(device: torch.device) -> tuple[bool, str]:
    """Compare the kernel with the PyTorch path on random problems: with and without the linear-discontinuous axis,
    from a strided view of cell-major flux moments, into every output layout.

    Args:
        device (torch.device): the CUDA device to check.

    Returns:
        (bool, str): whether the largest relative deviation is within ``SELF_CHECK_REL_TOL``, and a message.
    """
    from ..angular_quadrature import _scatter_source_moments_torch

    generator = torch.Generator().manual_seed(5)
    G, L, spatial = 3, 2, (3, 4, 5)
    transfer = torch.rand(G, G, L + 2, *spatial, generator=generator)
    transfer[1, 0] = 0.0  # a zero block, skipped by the PyTorch path
    transfer = transfer.to(device)
    worst = 0.0
    for rest in ((4,), ()):
        storage = torch.randn((L + 1) ** 2, *spatial, G, *rest, generator=generator).to(device)
        phi = cell_major_view(storage, len(spatial))
        reference = _scatter_source_moments_torch(transfer, phi, torch.zeros(G, *phi.shape[1:], device=device))
        for layout in LAYOUTS:
            out = scattering_moments(transfer, phi, layout)
            if out.shape != reference.shape:
                return False, f"self-check: {layout} output has shape {tuple(out.shape)}"
            scale = float(reference.abs().max())
            worst = max(worst, float((out - reference).abs().max()) / max(scale, 1e-30))
    torch.cuda.synchronize(device)
    if worst > SELF_CHECK_REL_TOL:
        return False, f"self-check: scattering-moments kernel deviates from the PyTorch path by {worst:.2e} (rel)"
    return True, f"scattering-moments kernel matches the PyTorch path (max rel dev {worst:.1e})"


def cell_major_view(storage: Tensor, num_spatial: int) -> Tensor:
    """The ``[G, N, (4), *spatial]`` view of cell-major moment storage ``[N, *spatial, G, (4)]``.

    Args:
        storage (Tensor): ``[N, *spatial, G]`` or ``[N, *spatial, G, 4]`` moments.
        num_spatial (int): number of spatial axes.

    Returns:
        Tensor: the view with the group axis first, then the moment and the linear-discontinuous axes.
    """
    group_axis = num_spatial + 1
    spatial_axes = list(range(1, num_spatial + 1))
    if storage.dim() == num_spatial + 3:
        return storage.permute(group_axis, 0, group_axis + 1, *spatial_axes)
    return storage.permute(group_axis, 0, *spatial_axes)


def _next_pow2(n: int) -> int:
    """Return the smallest power of two >= ``n``.

    Args:
        n (int): a positive integer.

    Returns:
        int: the smallest power of two >= ``n`` (1 for ``n <= 1``).
    """
    p = 1
    while p < n:
        p *= 2
    return p


def _merged_stride(t: Tensor, first: int) -> int | None:
    """Element stride of the trailing axes of ``t`` from ``first`` on if they merge into one axis, else None.

    Args:
        t (Tensor): the tensor.
        first (int): index of the first trailing axis.

    Returns:
        int | None: the stride of the merged axis, or None when the axes do not merge.
    """
    sizes, strides = t.shape[first:], t.stride()[first:]
    kept = [(n, s) for n, s in zip(sizes, strides) if n != 1]
    if not kept:
        return 1
    for (n0, s0), (n1, s1) in zip(kept, kept[1:]):
        if s0 != s1 * n1:
            return None
    return kept[-1][1]


if triton is not None:

    @triton.jit(do_not_specialize=["X", "ts_pair", "ts_l", "ps_g", "ps_n", "ps_k", "ps_x", "os_g", "os_n", "os_k",
                                   "os_x"],
                do_not_specialize_on_alignment=["transfer_ptr", "phi_ptr", "out_ptr"])
    def _scattering_moments_kernel(transfer_ptr, phi_ptr, out_ptr, X, ts_pair, ts_l, ps_g, ps_n, ps_k, ps_x, os_g,
                                   os_n, os_k, os_x, GI: tl.constexpr, GO: tl.constexpr, GOP: tl.constexpr,
                                   K: tl.constexpr, KP: tl.constexpr, BLOCK: tl.constexpr):
        """Scattering-source moments of moment n (program_id 0) for BLOCK points (program_ids 1 and 2), all groups.

        Args:
            transfer_ptr: float32 Legendre transfer moments, element ``(g_in, g_out, l, x)`` at
                ``(g_in GO + g_out) ts_pair + l ts_l + x``.
            phi_ptr: float32 flux moments, element ``(g, n, k, x)`` at ``g ps_g + n ps_n + k ps_k + x ps_x``.
            out_ptr: float32 moments T, element ``(g, n, k, x)`` at ``g os_g + n os_n + k os_k + x os_x``.
            X: number of points.
            ts_pair: element stride of a (g_in, g_out) block of the transfer.
            ts_l: element stride of the degree axis of the transfer.
            ps_g, ps_n, ps_k, ps_x: element strides of the flux moments.
            os_g, os_n, os_k, os_x: element strides of the output.
            GI (constexpr): number of incoming groups.
            GO (constexpr): number of outgoing groups.
            GOP (constexpr): next power of two >= GO.
            K (constexpr): linear-discontinuous components per moment (4, or 1 without that axis).
            KP (constexpr): next power of two >= K.
            BLOCK (constexpr): points per program.
        """
        n = tl.program_id(0)
        pid_x = tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)  # grid axis 1 holds at most 65535 blocks
        # degree l of moment n = l^2 + (m + l): floor(sqrt(n)), exact for the perfect squares, then made safe
        l = tl.sqrt_rn(n.to(tl.float32)).to(tl.int32)
        l = tl.where((l + 1) * (l + 1) <= n, l + 1, l)
        l = tl.where(l * l > n, l - 1, l)
        x = (pid_x * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        k = tl.arange(0, KP).to(tl.int64)
        xm = x < X
        phi_mask = xm[:, None] & (k < K)[None, :]
        transfer_row = transfer_ptr + l.to(tl.int64) * ts_l.to(tl.int64) + x  # [BLOCK]: the row (g_in, g_out) = 0
        phi_base = phi_ptr + n.to(tl.int64) * ps_n.to(tl.int64) + x[:, None] * ps_x + k[None, :] * ps_k
        out_base = out_ptr + n.to(tl.int64) * os_n.to(tl.int64) + x[:, None] * os_x + k[None, :] * os_k
        for g_out in tl.static_range(GO):
            acc = tl.zeros((BLOCK, KP), dtype=tl.float32)
            for g_in in tl.static_range(GI):  # summed over g_in in ascending order, as the PyTorch path does
                m = tl.load(transfer_row + (g_in * GO + g_out) * ts_pair.to(tl.int64), mask=xm, other=0.0)
                ph = tl.load(phi_base + g_in * ps_g.to(tl.int64), mask=phi_mask, other=0.0)
                acc += m[:, None] * ph
            tl.store(out_base + g_out * os_g.to(tl.int64), acc, mask=phi_mask)


def scattering_moments(transfer: Tensor, phi: Tensor, layout: str = "contiguous") -> Tensor:
    """Spherical-harmonic moments of the scattering source on float32 CUDA tensors, without availability check.

    The fused form of ``angular_quadrature.scatter_source_moments``.

    Args:
        transfer (Tensor): ``[G_in, G_out, L_x+1, *spatial]`` Legendre transfer moments per point in cm^-1.
        phi (Tensor): ``[G_in, N, *rest]`` flux moments, ``N = (L+1)^2`` with ``L <= L_x``; rest is spatial or
            ``(4, *spatial)``, any strides (a copy is made if the spatial axes do not merge into one).
        layout (str, optional): storage of the result: "contiguous" (``[G_out, N, *rest]``), "moment_major"
            (``[N, G_out, *rest]``) or "cell_major" (``[N, *spatial, G_out, (4)]``, the sweep's layout). Defaults to
            "contiguous".

    Returns:
        Tensor: ``[G_out, N, *rest]`` moments T, a view of the requested storage.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    G_in, G_out, num_legendre = (int(v) for v in transfer.shape[:3])
    spatial = tuple(int(v) for v in transfer.shape[3:])
    N = int(phi.shape[1])
    rest = tuple(int(v) for v in phi.shape[2:])
    has_k = rest != spatial
    X = math.prod(spatial)
    transfer = transfer.contiguous()
    ps_x = _merged_stride(phi, phi.dim() - len(spatial))
    if ps_x is None:
        phi = phi.contiguous()
        ps_x = 1
    if layout == "contiguous":
        out = torch.empty((G_out, N) + rest, device=phi.device, dtype=phi.dtype)
    elif layout == "moment_major":
        out = torch.empty((N, G_out) + rest, device=phi.device, dtype=phi.dtype).movedim(0, 1)
    else:
        storage = torch.empty((N,) + spatial + (G_out,) + ((4,) if has_k else ()), device=phi.device, dtype=phi.dtype)
        out = cell_major_view(storage, len(spatial))
    os_x = _merged_stride(out, out.dim() - len(spatial))
    if X == 0 or N == 0:
        return out
    K = 4 if has_k else 1
    blocks = triton.cdiv(X, BLOCK)
    triton_utilities.launch(
        _scattering_moments_kernel, (N, min(blocks, 65535), triton.cdiv(blocks, 65535)),
        (transfer, phi, out, X, num_legendre * X, X, phi.stride(0), phi.stride(1), phi.stride(2) if has_k else 0,
         ps_x, out.stride(0), out.stride(1), out.stride(2) if has_k else 0, os_x),
        dict(GI=G_in, GO=G_out, GOP=_next_pow2(G_out), K=K, KP=4 if has_k else 1, BLOCK=BLOCK), NUM_WARPS)
    return out
