.. ResPAN documentation master file, created by
   sphinx-quickstart on Sun Jun 26 08:14:10 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ResPAN's documentation!
==================================
This reporsity contains code and information of data used in the paper “*ResPAN: a powerful batch correction model for scRNA-seq data through residual adversarial networks*”. Source code for ResPAN are in the `ResPAN <https://github.com/AprilYuge/ResPAN/tree/main/ResPAN>`_ folder, scipts for reproducing benchmarking results are in the `scripts <https://github.com/AprilYuge/ResPAN/tree/main/scripts>`_ folder, and data information can be found in the `data <https://github.com/AprilYuge/ResPAN/tree/main/data>`_ folder.

ResPAN is a light structured **Res**\ idual autoencoder and mutual nearest neighbor **P**\ aring guided **A**\ dversarial **N**\ etwork for scRNA-seq batch correction. The workflow of ResPAN contains three key steps: generation of training data, adversarial training of the neural network, and generation of corrected data without batch effect. A figure summary is shown below.

.. image:: _st
   atic/workflow.png
   :width: 600
   :alt: Model architecture

More details about ResPAN can be found in our `manuscript <https://www.biorxiv.org/content/10.1101/2021.11.08.467781v4>`_.


.. toctree::
   :maxdepth: 5
   :caption: Contents:

   install
   tutorial
   dataset
   API
   citation

Indices and tables
==================

* :ref:`genindex`



