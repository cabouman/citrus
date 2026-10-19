"""Shared Triton helpers of the fused kernels: the availability probe and the fast kernel launch.

``triton_available`` runs a trivial probe kernel once per process; ``CITRUS_DISABLE_TRITON=1`` (``DISABLE_ENV_VAR``)
turns every kernel off. ``launch`` launches a kernel through its cached compiled binary on triton 3.8, unless
``CITRUS_TRITON_FAST_LAUNCH=0`` (``FAST_LAUNCH_ENV_VAR``), and ``LaunchSequence`` runs repeated launches of one kernel
on the same tensors (the planes of a sweep chunk) through the binary of the first. The module attributes ``triton``
and ``tl`` are None when Triton cannot be imported.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

DISABLE_ENV_VAR = "CITRUS_DISABLE_TRITON"
FAST_LAUNCH_ENV_VAR = "CITRUS_TRITON_FAST_LAUNCH"

_PROBE: Optional[tuple[bool, str]] = None

try:  # import errors are reported by triton_available(), never raised at import time
    import triton
    import triton.language as tl

    _IMPORT_ERROR: Optional[str] = None
except Exception as e:  # noqa: BLE001
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _IMPORT_ERROR = f"{type(e).__name__}: {e}"


def triton_available() -> tuple[bool, str]:
    """Report whether Triton kernels can run on this machine; cached per process, never raises.

    Returns:
        (bool, str): whether Triton is usable, and the reason.
    """
    global _PROBE
    if _PROBE is None:
        _PROBE = _probe()
    return _PROBE


def _probe() -> tuple[bool, str]:
    """Check the kill switch, the Triton import and CUDA, then compile and run the probe kernel.

    Returns:
        (bool, str): whether Triton is usable, and the reason.
    """
    if os.environ.get(DISABLE_ENV_VAR, "0") == "1":
        return False, f"disabled by {DISABLE_ENV_VAR}=1"
    if _IMPORT_ERROR is not None:
        return False, f"triton import failed ({_IMPORT_ERROR})"
    if not torch.cuda.is_available():
        return False, "not a CUDA platform"
    try:
        x = torch.arange(64, device="cuda", dtype=torch.float32)
        y = torch.full_like(x, 2.0)
        out = torch.empty_like(x)
        _probe_kernel[(1,)](x, y, out, BLOCK=64)
        torch.cuda.synchronize()
        if not bool(torch.equal(out, x + y)):
            return False, "probe kernel returned wrong values"
    except Exception as e:  # noqa: BLE001
        return False, f"probe kernel failed ({type(e).__name__}: {str(e)[:300]})"
    return True, f"triton {triton.__version__} on {torch.cuda.get_device_name(0)}"


if triton is not None:

    @triton.jit
    def _probe_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
        """Write ``out = x + y`` for one block.

        Args:
            x_ptr: [BLOCK] float32 input.
            y_ptr: [BLOCK] float32 input.
            out_ptr: [BLOCK] float32 output.
            BLOCK (constexpr): lanes per program.
        """
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs) + tl.load(y_ptr + offs))


_FAST_LAUNCH = (triton is not None and triton.__version__.startswith("3.8.")
                and os.environ.get(FAST_LAUNCH_ENV_VAR, "1") != "0")
_LAUNCHERS: dict[tuple, tuple] = {}


def launch(kernel, grid: tuple[int, int, int], args: tuple, constexprs: dict, num_warps: int) -> None:
    """Run ``kernel[grid](*args, **constexprs, num_warps=num_warps)``, through the cached compiled kernel when allowed.

    The kernel must be compiled without value or alignment specialization, so that one cached binary fits all values.

    Args:
        kernel (triton.JITFunction): the kernel.
        grid (tuple[int, int, int]): the launch grid.
        args (tuple): exactly the kernel's non-constexpr parameters, in order.
        constexprs (dict): the constexpr parameters by name.
        num_warps (int): warps per program.
    """
    if not _FAST_LAUNCH:
        kernel[grid](*args, **constexprs, num_warps=num_warps)
        return
    device = torch.cuda.current_device()
    key = (kernel, tuple(constexprs.items()), num_warps, device)
    entry = _LAUNCHERS.get(key)
    if entry is None:
        compiled = kernel[grid](*args, **constexprs, num_warps=num_warps)  # compiles / loads, then launches
        names = [p.name for p in kernel.params]
        constexpr_values = [constexprs[p.name] for p in kernel.params if p.is_constexpr]
        if (len(names) != len(args) + len(constexpr_values)
                or any(p.is_constexpr for p in kernel.params[:len(args)])):
            raise RuntimeError(f"{kernel}: positional arguments must precede the constexprs")
        _LAUNCHERS[key] = (compiled, constexpr_values)
        return
    compiled, constexpr_values = entry
    compiled.run(grid[0], grid[1], grid[2], torch._C._cuda_getCurrentRawStream(device), compiled.function,
                 compiled.packed_metadata, None, None, None, *args, *constexpr_values)


class LaunchSequence:
    """Repeated launches of one kernel with the same constexprs and pointer arguments.

    The first launch goes through Triton's public path, which compiles the kernel (or loads it) with the
    specializations of its arguments; the later ones run that binary directly. The caller passes the same tensors and
    the same specialized integers to every launch and varies only the integers the kernel does not specialize, as the
    planes of one sweep chunk do.

    Attributes:
        kernel (triton.JITFunction): the kernel.
        constexprs (dict): the constexpr parameters by name.
        num_warps (int): warps per program.
    """

    def __init__(self, kernel, constexprs: dict, num_warps: int) -> None:
        """Record the kernel and its launch constants.

        Args:
            kernel (triton.JITFunction): the kernel.
            constexprs (dict): the constexpr parameters by name.
            num_warps (int): warps per program.
        """
        self.kernel = kernel
        self.constexprs = constexprs
        self.num_warps = num_warps
        self._compiled = None
        self._constexpr_values: list = []
        self._stream = None

    def __call__(self, grid: tuple[int, int, int], args: tuple) -> None:
        """Launch the kernel on ``grid`` with ``args``, the kernel's non-constexpr parameters in order.

        Args:
            grid (tuple[int, int, int]): the launch grid.
            args (tuple): the non-constexpr arguments.
        """
        if self._compiled is not None:
            self._compiled.run(grid[0], grid[1], grid[2], self._stream, self._compiled.function,
                               self._compiled.packed_metadata, None, None, None, *args, *self._constexpr_values)
            return
        compiled = self.kernel[grid](*args, **self.constexprs, num_warps=self.num_warps)
        if _FAST_LAUNCH:
            self._compiled = compiled
            self._constexpr_values = [self.constexprs[p.name] for p in self.kernel.params if p.is_constexpr]
            self._stream = torch._C._cuda_getCurrentRawStream(torch.cuda.current_device())
