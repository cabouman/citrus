Citation and credits
====================

CITRUS is developed by Jordan Houri and Charles A. Bouman at Purdue University. Please cite the software when
referencing this package:

.. code-block:: bibtex

   @misc{citrus,
     title = {citrus},
     author = {Jordan Houri and Charles A. Bouman},
     howpublished = {Software library available from \url{https://github.com/cabouman/citrus}},
     year = 2026
   }

GitHub's "Cite this repository" button on the `repository page <https://github.com/cabouman/citrus>`_ generates the
citation of the installed version from ``CITATION.cff``.

Related work
------------

* The deterministic LBTE formulation of CT scatter follows Acuros CTS: A. Maslowski, A. Wang, M. Sun, T. Wareing,
  I. Davis and J. Star-Lack, "Acuros CTS: A fast, linear Boltzmann transport equation solver for computed tomography
  scatter, Part I: Core algorithms and validation", *Medical Physics* 45(5), 1899-1913 (2018), and A. Wang et al.,
  "Part II: System modeling, scatter correction, and optimization", *Medical Physics* 45(5), 1914-1925 (2018).
* The reconstruction uses `MBIRTORCH <https://github.com/cabouman/mbirtorch>`_ (C. A. Bouman and collaborators).
* Photon interaction data (EPDL97) come from `XrayPhysics <https://github.com/kylechampley/XrayPhysics>`_
  (K. Champley).
* Material compositions: ICRU Report 44 (1989), ICRU Report 46 (1992) / Woodard and White, *British Journal of
  Radiology* 59, 1209-1219 (1986), and the XrayPhysics material library.

License
-------

See the ``LICENSE`` file in the repository.
