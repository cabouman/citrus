"""Fused Triton kernels of the scatter solver, for float32 CUDA tensors.

Every kernel has a PyTorch reference path that serves CPU and float64 tensors and runs when Triton is unavailable.
The modules are ``triton_utilities`` (probe and launch), ``ray_tracing_kernels`` (the Siddon ray tracer),
``uncollided_kernels`` (the first-collision source), ``sweep_kernels`` (the plane solve of the transport sweep),
``scattering_kernels`` (the moments of the scattering source) and ``transmission_kernels`` (the group transmission
along material line integrals).
"""

from .triton_utilities import triton_available

__all__ = ["triton_available"]
