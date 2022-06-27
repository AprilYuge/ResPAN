Installation guide
==================================

************
Main package
************
To install the stable version of our tool, please directly use pip install:

.. code-block:: bash
    :linenos:

    pip install ResPAN 

Or if you intend to install the developing version of our tool, please use git bash:

.. code-block:: bash
    :linenos:

    git clone https://github.com/AprilYuge/ResPAN.git

.. note::
    To avoid potential dependency conflicts, installing within a
    `conda environment <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
    is recommended.


*********************
Optional dependencies
*********************

.. list-table::
   :widths: 15 15
   :header-rows: 1

   * - Package
     - Version
   * - numpy
     - 1.18.1
   * - pandas
     - 1.3.5
   * - scipy
     - 1.8.0
   * - scanpy
     - 1.8.2
   * - pytorch
     - 1.10.2+cu11
