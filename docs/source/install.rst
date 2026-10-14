============
Installation
============

citrus is installed from source:

.. code-block:: bash

   git clone git@github.com:cabouman/citrus.git
   cd citrus
   pip install .

This installs the Python dependencies (numpy, scipy, torch, matplotlib)
automatically.

To verify the installation, run the test suite:

.. code-block:: bash

   pip install pytest
   pytest tests/
