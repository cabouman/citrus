"""Fused Triton kernels of the scatter solver (float32 CUDA only; every kernel has a PyTorch reference path that
serves CPU and float64 tensors and is used automatically when Triton is unavailable).

    triton_utilities     shared Triton probe, kill switch (CITRUS_DISABLE_TRITON=1) and fast launch
    ray_tracing_kernels  exact Siddon ray tracer: line integrals, upstream / downstream cumulative, projections
    uncollided_kernels   first-collision source evaluation
    sweep_kernels        linear-discontinuous cell solve of one wavefront plane of the transport sweep
"""

from .triton_utilities import triton_available

__all__ = ["triton_available"]
