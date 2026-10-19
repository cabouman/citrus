#!/bin/bash
# Reports the GPU driver, PyTorch / CUDA and Triton availability, runs the per-device self-checks of the fused
# Triton kernels of citrus (ray tracer on flat and curved panels with circular and helical views and on parallel
# beams, first-collision source of cone and parallel beams, transport sweep) against their PyTorch paths, and
# reports xraylib and the NIST XCOM data (the photon data of the cross-section library) and the cache.
# Usage: ./check_cuda.sh   (exit code 0 if CUDA is usable, 1 otherwise)

echo "=== GPU driver ==="
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "nvidia-smi not found (no NVIDIA driver on PATH)"
fi
echo ""

python - <<'PYEOF'
import sys

import torch

print("=== PyTorch ===")
print(f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")
cuda = torch.cuda.is_available()
print(f"CUDA available: {cuda}")
if cuda:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  device {i}: {p.name}, {p.total_memory / 2 ** 30:.1f} GiB, compute capability {p.major}.{p.minor}")

print("\n=== Triton ===")
try:
    import triton
    print(f"triton {triton.__version__}")
except Exception as exc:  # noqa: BLE001
    print(f"triton not importable: {type(exc).__name__}: {exc}")

print("\n=== citrus kernels ===")
from citrus.scatter_solver.kernels import ray_tracing_kernels, sweep_kernels, triton_utilities, uncollided_kernels

print("probe:              ", triton_utilities.triton_available())
if cuda:
    device = torch.device("cuda")
    print("ray tracing kernels:", ray_tracing_kernels.kernels_usable(device))
    print("first-collision:    ", uncollided_kernels.kernel_usable(device))
    print("sweep plane kernel: ", sweep_kernels.kernels_usable(device))
else:
    print("no CUDA device: the PyTorch paths are used")
try:
    import mbirtorch
    print(f"\nmbirtorch {mbirtorch.__version__}")
except Exception as exc:  # noqa: BLE001
    print(f"\nmbirtorch not importable: {type(exc).__name__}: {exc}")
print("\n=== photon data ===")
try:
    import xraylib
    print(f"xraylib {xraylib.__version__}")
except Exception as exc:  # noqa: BLE001
    print(f"xraylib not importable: {type(exc).__name__}: {exc}")
try:
    import os

    from citrus.scatter_solver import photon_data
    print(f"cache: {photon_data.default_cache_dir()} (XCOM data in {photon_data.xcom_cache_dir()})")
    variable = os.environ.get(photon_data.XCOM_ARCHIVE_ENVIRONMENT_VARIABLE)
    print(f"{photon_data.XCOM_ARCHIVE_ENVIRONMENT_VARIABLE}: {variable or 'not set'}")
    backend = photon_data.resolve_backend()  # downloads the XCOM archive on first use
    print("XCOM:", backend.reason)
    print("XCOM source SHA-256:", backend.xcom_sha256,
          "(the expected archive)" if backend.xcom_sha256 == photon_data.XCOM_ARCHIVE_SHA256 else "")
except Exception as exc:  # noqa: BLE001
    print(f"photon data unavailable: {exc}")
sys.exit(0 if cuda else 1)
PYEOF
