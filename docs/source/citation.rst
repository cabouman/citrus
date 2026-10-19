Citation and credits
====================

CITRUS is developed by Jordan Houri, Gregery T. Buzzard and Charles A. Bouman at Purdue University. Please cite the
software when
referencing this package:

.. code-block:: bibtex

   @misc{citrus,
     title = {citrus},
     author = {Jordan Houri and Gregery T. Buzzard and Charles A. Bouman},
     howpublished = {Software library available from \url{https://github.com/cabouman/citrus}},
     year = 2026
   }

GitHub's "Cite this repository" button on the `repository page <https://github.com/cabouman/citrus>`_ generates the
citation of the current version from ``CITATION.cff``.

Related work
------------

* The deterministic LBTE formulation of CT scatter follows Acuros CTS: A. Maslowski, A. Wang, M. Sun, T. Wareing,
  I. Davis and J. Star-Lack, "Acuros CTS: A fast, linear Boltzmann transport equation solver for computed tomography
  scatter, Part I: Core algorithms and validation", *Medical Physics* 45(5), 1899-1913 (2018), and A. Wang et al.,
  "Part II: System modeling, scatter correction, and optimization", *Medical Physics* 45(5), 1914-1925 (2018).
* The reconstruction uses `MBIRTORCH <https://github.com/cabouman/mbirtorch>`_ (C. A. Bouman and collaborators).
* Photon cross sections and atomic weights come from NIST XCOM: M. J. Berger, J. H. Hubbell, S. M. Seltzer, J. Chang,
  J. S. Coursey, R. Sukumar, D. S. Zucker and K. Olsen, *XCOM: Photon Cross Sections Database*, NIST Standard
  Reference Database 8 (XGAM), National Institute of Standards and Technology,
  `physics.nist.gov/xcom <https://physics.nist.gov/xcom>`_. CITRUS downloads the data from NIST on first use and does
  not redistribute them; their use is subject to NIST's terms for Standard Reference Data.
* The Compton and Rayleigh angular distributions, element densities and the NIST compound data used to
  check the catalogue come from `xraylib <https://github.com/tschoonj/xraylib>`_: T. Schoonjans, A. Brunetti,
  B. Golosio, M. Sanchez del Rio, V. A. Sole, C. Ferrero and L. Vincze, "The xraylib library for X-ray-matter
  interactions. Recent developments", *Spectrochimica Acta Part B* 66, 776-784 (2011).
* Material compositions: ICRU Report 44 (1989), ICRU Report 46 (1992) / Woodard and White, *British Journal of
  Radiology* 59, 1209-1219 (1986), and the ICRP compositions of NIST's compound tables (adipose tissue, lung,
  cortical bone).

License
-------

See the ``LICENSE`` file in the repository.
