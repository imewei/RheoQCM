RheoQCM Documentation
=====================

**RheoQCM** is a Python package for Quartz Crystal Microbalance with Dissipation
(QCM-D) data acquisition and rheological analysis. It combines a modern
JAX-accelerated computational core with a PyQt6 GUI for data collection and
visualization.

.. image:: _images/screenshot.png
   :alt: RheoQCM User Interface
   :align: center
   :width: 80%

Key Features
------------

- **High-performance modeling** with JAX (GPU-accelerated when available)
- **QCM data collection and analysis** in one integrated package
- **Import and analyze** external QCM-D datasets (.xlsx, .mat, .h5)
- **Multilayer thin-film analysis** using the Small Load Approximation (SLA)
- **Bayesian parameter estimation** with MCMC (NumPyro backend)
- **Uncertainty quantification** via autodiff-based covariance propagation

Getting Started
---------------

New to RheoQCM? Start here:

1. :doc:`tutorials/installation` - Install RheoQCM and dependencies
2. :doc:`tutorials/quickstart` - Run your first analysis in 5 minutes
3. :doc:`theory/index` - Understand the QCM physics background

Documentation Sections
----------------------

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **Tutorials**
     - Step-by-step guides: :doc:`tutorials/installation` · :doc:`tutorials/quickstart` · :doc:`tutorials/gui-workflow` · :doc:`tutorials/scripting-basics` · :doc:`tutorials/batch-analysis` · :doc:`tutorials/bayesian-fitting`

   * - **Theory**
     - Physics background: :doc:`theory/qcm-fundamentals` · :doc:`theory/viscoelastic-modeling` · :doc:`theory/multilayer-films` · :doc:`theory/numerical-methods`

   * - **User Guide**
     - Practical workflows: :doc:`user-guide/data-acquisition` · :doc:`user-guide/data-import` · :doc:`user-guide/analysis-settings` · :doc:`user-guide/visualization` · :doc:`user-guide/exporting-results`

   * - **API Reference**
     - Code documentation: :doc:`api/index`

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Theory & Background
   :hidden:

   theory/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   user-guide/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   :hidden:

   references
   migration
   changelog


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
