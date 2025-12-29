"""
QCMFuncs - Legacy QCM Analysis Functions Package.

This package is deprecated. For new code, use rheoQCM.core instead:

    from rheoQCM.core.analysis import QCMAnalyzer
    from rheoQCM.core import sauerbreyf, sauerbreym, grho

Note: This package uses phi in DEGREES, rheoQCM.core uses RADIANS.
"""

import os
import warnings

# T061: Emit FutureWarning on package import
_SUPPRESS_DEPRECATION = os.environ.get("QCMFUNCS_SUPPRESS_DEPRECATION", "").lower() in (
    "1",
    "true",
    "yes",
)

if not _SUPPRESS_DEPRECATION:
    warnings.warn(
        "The QCMFuncs package is deprecated and will be removed in a future release.\n"
        "Please migrate to rheoQCM.core:\n"
        "  from rheoQCM.core.model import QCMModel\n"
        "  from rheoQCM.core.analysis import batch_analyze_vmap\n"
        "  from rheoQCM.core import sauerbreyf, sauerbreym, grho\n"
        "See docs/source/migration.md for the full migration guide.\n"
        "To suppress: set QCMFUNCS_SUPPRESS_DEPRECATION=1",
        FutureWarning,
        stacklevel=2,
    )
