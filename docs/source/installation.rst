Installation
============

Requirements
------------

* Python >= 3.11, PyTorch (CUDA build for GPU use), NumPy, SciPy, Matplotlib.
* `MBIRTORCH <https://github.com/cabouman/mbirtorch>`_ for the reconstruction side (``citrus_recon``,
  ``mbir_recon``, the demos; installed as a dependency, verified with mbirtorch 0.1.2).
* `xraylib <https://github.com/tschoonj/xraylib>`_ for the Compton and Rayleigh angular distributions and the atomic
  weights (installed as a dependency from its wheels, verified with xraylib 4.2.1).
* The NIST XCOM photon cross sections, downloaded on first use (below).
* Optional: Triton (``triton`` on Linux, ``triton-windows`` on Windows) for the fused GPU kernels.

Nothing is compiled: every dependency installs from wheels.

Install the package
-------------------

.. code-block:: bash

   git clone https://github.com/cabouman/citrus.git
   cd citrus
   pip install .                  # or: pip install -e .   (development); installs MBIRTORCH and xraylib too
   pip install ".[test]"          # with pytest
   pip install ".[gpu]"           # Triton

The ``gpu`` extra installs ``triton`` on Linux and ``triton-windows`` on Windows.
``dev_scripts/create_conda_test_environment.sh`` creates a conda environment ``citrus`` (Python 3.11) with the test
extras; ``dev_scripts/check_cuda.sh`` reports what is available.

PyTorch is best installed first, with the CUDA version of your driver (see pytorch.org).

Photon data
-----------

The multigroup library (``citrus.scatter_solver.cross_section_library.build_library``) is built from the integrated
photon cross sections of NIST XCOM (coherent and incoherent scattering, photoelectric absorption, pair production) and
the angular distributions of Compton and Rayleigh scattering of xraylib (module
``citrus.scatter_solver.photon_data``). The XCOM tables are not part of the package. On first use CITRUS downloads
NIST's offline distribution of XCOM, ``https://physics.nist.gov/PhysRefData/Xcom/XCOM.tar.gz`` (173 kB), into
``$CITRUS_CACHE_DIR/xcom`` (default ``~/.cache/citrus/xcom``), checks its size and SHA-256, and parses the element
tables once into ``xcom_elements.npz`` next to it; later runs read that file.

On a machine without internet access, download the archive elsewhere and point the environment variable
``CITRUS_XCOM_ARCHIVE`` to it, or to the folder it unpacks to:

.. code-block:: bash

   export CITRUS_XCOM_ARCHIVE=/path/to/XCOM.tar.gz      # Windows: set CITRUS_XCOM_ARCHIVE=C:\path\to\XCOM.tar.gz

The data are looked for in this order: the archive or folder named by ``CITRUS_XCOM_ARCHIVE``; the parsed cache
``xcom_elements.npz``; the archive ``XCOM.tar.gz`` in the cache folder; a download into the cache folder. Only the
element tables and ``ATWTS.DAT`` are read, in memory, from the archive; nothing is written outside the cache folder.
The parsed cache holds every element's grid, its five cross sections, its edges and the tables below the K edge,
with the SHA-256 of the source it was parsed from; a cache whose source does not match is parsed again.

When neither the variable, the cache nor the network is available, building a library raises an error that names the
URL and the variable. XCOM is a NIST Standard Reference Database; its use is subject to NIST's terms for Standard
Reference Data, and publications should cite it (:doc:`citation`).

The built library is cached in ``$CITRUS_CACHE_DIR`` (default ``~/.cache/citrus``), keyed by the configuration, the
spectrum, the photon data and the build parameters. Tests that need the real library are skipped when the XCOM data
are unavailable and nothing is cached.

Triton
------

On float32 CUDA tensors the ray tracer, the first-collision source and the sweep's cell solve run as fused Triton
kernels. Every kernel is checked against its PyTorch reference once per device (a tiny self-check problem) and the
PyTorch path is used automatically when Triton is missing or the check fails. ``CITRUS_DISABLE_TRITON=1`` forces the
PyTorch paths. On Windows install ``triton-windows`` (it needs a matching PyTorch version). The fast launch path of the
kernels uses Triton internals verified for Triton 3.8; other versions (or ``CITRUS_TRITON_FAST_LAUNCH=0``) use the
public launch path.

GPU memory
----------

On Windows, VRAM oversubscription silently spills into system memory and runs much slower; the study script
therefore caps the PyTorch allocator (``citrus.utilities.set_memory_cap``; ``--memory-cap-mib``, default 85 percent
of the device memory).

Tests
-----

.. code-block:: bash

   python -m pytest -q
   CITRUS_SLOW_TESTS=1 python -m pytest -q     # also the slow tests (full-resolution library builds)

CUDA-, Triton- and MBIRTORCH-dependent tests, and those that need the XCOM data, are skipped when those are
unavailable; the CPU / float64 oracle tests always run.
