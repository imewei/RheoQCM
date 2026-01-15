"""
QCMFuncs - Legacy QCM Analysis Functions Package.

.. deprecated:: 2.0.0
    This package is deprecated and will be removed in v3.0.0 (scheduled Q3 2025).
    For new code, use rheoQCM.core instead.

Migration Guide
---------------
Replace QCMFuncs imports with rheoQCM.core equivalents::

    # OLD (deprecated)
    from QCMFuncs import QCM_functions as qcm
    result = qcm.solve_qcm(delfstar, ...)

    # NEW (recommended)
    from rheoQCM.core.model import QCMModel
    model = QCMModel(f1=5e6)
    model.load_delfstars({3: delfstar_3, 5: delfstar_5})
    result = model.solve_properties(nh=[3, 5, 3])

Key Differences
---------------
- **Angle Units**: QCMFuncs uses degrees, rheoQCM.core uses radians
- **Return Type**: QCMFuncs returns tuples, rheoQCM.core returns SolveResult dataclass
- **State**: QCMFuncs uses global state, rheoQCM.core uses QCMModel instances

Deprecation Timeline
--------------------
- v2.0.0 (Current): FutureWarning on import, all functions still work
- v2.5.0 (Q1 2025): DeprecationWarning, functions marked for removal
- v3.0.0 (Q3 2025): Package removed entirely

To suppress deprecation warnings during migration::

    import os
    os.environ["QCMFUNCS_SUPPRESS_DEPRECATION"] = "1"
    from QCMFuncs import QCM_functions  # No warning

See Also
--------
- Full migration guide: docs/source/migration.md
- rheoQCM.core.model : Modern QCM analysis interface
- rheoQCM.core.analysis : Batch processing utilities
"""

import os
import warnings

__version__ = "2.0.0"
__deprecated__ = True
__removal_version__ = "3.0.0"
__removal_date__ = "Q3 2025"

# T061: Emit FutureWarning on package import
_SUPPRESS_DEPRECATION = os.environ.get("QCMFUNCS_SUPPRESS_DEPRECATION", "").lower() in (
    "1",
    "true",
    "yes",
)

if not _SUPPRESS_DEPRECATION:
    warnings.warn(
        f"QCMFuncs is deprecated and will be removed in v{__removal_version__} ({__removal_date__}).\n"
        "Please migrate to rheoQCM.core:\n"
        "  from rheoQCM.core.model import QCMModel\n"
        "  from rheoQCM.core.analysis import batch_analyze_vmap\n"
        "  from rheoQCM.core import sauerbreyf, sauerbreym, grho\n"
        "See docs/source/migration.md for the full migration guide.\n"
        "To suppress: set QCMFUNCS_SUPPRESS_DEPRECATION=1",
        FutureWarning,
        stacklevel=2,
    )
