Installation
============

Requirements
------------

* Python >= 3.11, PyTorch (CUDA build for GPU use), NumPy, SciPy, Matplotlib.
* `MBIRTORCH <https://github.com/cabouman/mbirtorch>`_ for the reconstruction side (``citrus_recon``,
  ``mbir_recon``, the demos).
* `XrayPhysics <https://github.com/kylechampley/XrayPhysics>`_ to build the cross-section library (once; the library
  is cached on disk).
* Optional: Triton (``triton`` on Linux, ``triton-windows`` on Windows) for the fused GPU kernels.

Install the package
-------------------

.. code-block:: bash

   git clone https://github.com/cabouman/citrus.git
   cd citrus
   pip install .            # or: pip install -e .   (development)
   pip install .[test]      # with pytest

PyTorch is best installed first, with the CUDA version of your driver (see pytorch.org).

XrayPhysics
-----------

The multigroup library is built from the EPDL97 photon interaction data of XrayPhysics
(``citrus.scatter_solver.cross_section_library.build_library``). XrayPhysics is a C++ library with Python bindings
and is built from source:

.. code-block:: bash

   git clone --depth 1 https://github.com/kylechampley/XrayPhysics
   cd XrayPhysics
   pip install .

On Windows this needs Visual Studio (C++ build tools) and CMake on ``PATH``; the build can also be done explicitly:

.. code-block:: bat

   cmake -S . -B win_build -G "Visual Studio 17 2022" -A x64
   cmake --build win_build --config Release
   set PATH=C:\Program Files\CMake\bin;%PATH%
   python -m pip install .

``setup.py`` copies ``libxrayphysics`` (``.dll`` / ``.so``) into site-packages. Check with
``python -c "from xrayphysics import xrayPhysics; xrayPhysics()"``.

The built library is cached in ``$CITRUS_CACHE_DIR`` (default ``~/.cache/citrus``), keyed by the configuration, the
spectrum and the build parameters; a build takes about 10 s. Once cached, XrayPhysics is no longer needed for that
configuration. Tests that need the real library are skipped when XrayPhysics is unavailable and nothing is cached.

Triton
------

On float32 CUDA tensors the ray tracer, the first-collision source and the sweep's cell solve run as fused Triton
kernels (10-80x faster than the PyTorch paths). Every kernel is checked against its PyTorch reference once per device
(a tiny self-check problem) and the PyTorch path is used automatically when Triton is missing or the check fails.
``CITRUS_DISABLE_TRITON=1`` forces the PyTorch paths. On Windows install ``triton-windows`` (it needs a matching
PyTorch version). The fast launch path of the kernels uses Triton internals verified for Triton 3.8; other versions
(or ``CITRUS_TRITON_FAST_LAUNCH=0``) use the public launch path.

GPU memory
----------

On Windows, VRAM oversubscription silently spills into system memory and runs much slower; the demos therefore cap the
PyTorch allocator (``citrus.utilities.set_memory_cap``, default 4400 MiB, ``--memory-cap-mib``).

Tests
-----

.. code-block:: bash

   python -m pytest -q
   CITRUS_SLOW_TESTS=1 python -m pytest -q     # also the slow tests (full-resolution library builds)

CUDA-, Triton-, MBIRTORCH- and XrayPhysics-dependent tests are skipped when those are unavailable; the CPU / float64
oracle tests always run.
