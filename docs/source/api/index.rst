API Reference
=============

Complete API documentation for RheoQCM, auto-generated from docstrings.

Quick Links
-----------

Most users will primarily use these modules:

- :mod:`rheoQCM.core.model` - The ``QCMModel`` class for analysis
- :mod:`rheoQCM.core.analysis` - ``batch_analyze()`` and ``batch_analyze_vmap()``
- :mod:`rheoQCM.core.physics` - Low-level physics functions

See :doc:`/tutorials/scripting-basics` for usage examples.

Core Package
------------

The core package contains all computational functionality.

.. autosummary::
   :toctree: generated
   :nosignatures:

   rheoQCM
   rheoQCM.core
   rheoQCM.core.model
   rheoQCM.core.analysis
   rheoQCM.core.physics
   rheoQCM.core.multilayer
   rheoQCM.core.bayesian
   rheoQCM.core.uncertainty
   rheoQCM.core.jax_config
   rheoQCM.core.signal

**Key Classes and Functions:**

.. rubric:: Model Layer

- :class:`~rheoQCM.core.model.QCMModel` - Main analysis interface
- :class:`~rheoQCM.core.model.SolveResult` - Single analysis result
- :func:`~rheoQCM.core.model.register_calctype` - Register custom physics

.. rubric:: Analysis Layer

- :func:`~rheoQCM.core.analysis.batch_analyze` - Batch processing
- :func:`~rheoQCM.core.analysis.batch_analyze_vmap` - GPU-accelerated batch
- :class:`~rheoQCM.core.analysis.BatchResult` - Batch result container

.. rubric:: Physics Layer

- :func:`~rheoQCM.core.physics.calc_delfstar_sla` - SLA frequency shift
- :func:`~rheoQCM.core.physics.sauerbreyf` - Sauerbrey frequency
- :func:`~rheoQCM.core.physics.sauerbreym` - Sauerbrey mass
- :func:`~rheoQCM.core.physics.grho` - Complex modulus calculation
- :func:`~rheoQCM.core.multilayer.calc_delfstar_multilayer` - Multilayer shift

GUI Components
--------------

Graphical user interface components (requires PyQt6).

.. autosummary::
   :toctree: generated
   :nosignatures:

   rheoQCM.gui
   rheoQCM.gui.dialogs
   rheoQCM.gui.widgets
   rheoQCM.gui.workers

I/O Handlers
------------

Data import and export functionality.

.. autosummary::
   :toctree: generated
   :nosignatures:

   rheoQCM.io
   rheoQCM.io.base
   rheoQCM.io.excel_handler
   rheoQCM.io.hdf5_handler
   rheoQCM.io.json_handler

See :doc:`/user-guide/data-import` for import tutorials.

Modules
-------

High-level modules for data management and peak tracking.

.. autosummary::
   :toctree: generated
   :nosignatures:

   rheoQCM.modules
   rheoQCM.modules.QCM
   rheoQCM.modules.DataSaver
   rheoQCM.modules.PeakTracker

Services
--------

Service components for hardware, plotting, and settings.

.. autosummary::
   :toctree: generated
   :nosignatures:

   rheoQCM.services
   rheoQCM.services.base
   rheoQCM.services.hardware
   rheoQCM.services.plotting
   rheoQCM.services.settings

Utilities
---------

Utility modules for device management and logging.

.. autosummary::
   :toctree: generated
   :nosignatures:

   rheoQCM.device
   rheoQCM.logging_config

Configuration
-------------

JAX configuration for precision and GPU acceleration.

**Important:** Always call :func:`~rheoQCM.core.jax_config.configure_jax`
before any analysis to enable 64-bit precision.

See :mod:`rheoQCM.core.jax_config` in the Core Package section above.

Legacy API (Deprecated)
-----------------------

.. deprecated:: 2.0.0
   The ``QCMFuncs`` module is deprecated and will be removed in a future release.
   Use :mod:`rheoQCM.core` instead.
   See :doc:`/changelog` for migration instructions.

.. autosummary::
   :toctree: generated
   :nosignatures:

   QCMFuncs
   QCMFuncs.QCM_functions

Migration from Legacy API
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Old (deprecated)
   from QCMFuncs import QCM_functions as qcm
   result = qcm.solve(...)

   # New (recommended)
   from rheoQCM.core import QCMModel, configure_jax
   configure_jax()
   model = QCMModel(f1=5e6, refh=3)
   result = model.solve_properties(...)

See Also
--------

- :doc:`/tutorials/scripting-basics` - Python API usage tutorial
- :doc:`/tutorials/batch-analysis` - Batch processing guide
- :doc:`/theory/numerical-methods` - Algorithm details
- :doc:`/changelog` - Version history and migration guide
