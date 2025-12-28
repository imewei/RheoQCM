"""
RheoQCM - QCM Data Collection and Analysis Software.

This package provides:
- GUI for QCM data collection and analysis
- Core computational modules for rheological modeling
- Peak fitting and tracking for resonance data
"""

import logging
import sys

from rheoQCM._version import __version__
from rheoQCM.logging_config import configure_logging

# Initialize logging system on package import
configure_logging()

# Log startup information
_logger = logging.getLogger(__name__)
_logger.info(f"RheoQCM version {__version__}")
_logger.debug(f"Python {sys.version}")

# Log key dependency versions at DEBUG level
try:
    import jax

    _logger.debug(f"JAX version {jax.__version__}")
except ImportError:
    pass

try:
    import numpy

    _logger.debug(f"NumPy version {numpy.__version__}")
except ImportError:
    pass

__all__ = ["__version__", "configure_logging"]
