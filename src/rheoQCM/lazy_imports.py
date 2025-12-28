"""
Lazy loading utilities for RheoQCM.

This module provides deferred module imports to improve startup time.
Heavy modules (matplotlib, jax submodules) are only imported when first accessed.

Configuration:
    Set RHEOQCM_DISABLE_LAZY_LOADING=1 to disable lazy loading for debugging.

Example:
    from rheoQCM.lazy_imports import LazyModule

    # Create a lazy module wrapper
    matplotlib = LazyModule("matplotlib")

    # matplotlib is NOT imported yet
    # ...

    # First access triggers import
    matplotlib.pyplot.figure()  # matplotlib imports here
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from types import ModuleType

logger = logging.getLogger(__name__)


def is_lazy_loading_enabled() -> bool:
    """Check if lazy loading is enabled.

    Returns:
        True unless RHEOQCM_DISABLE_LAZY_LOADING env var is set to '1' or 'true'
    """
    disabled = os.environ.get("RHEOQCM_DISABLE_LAZY_LOADING", "").lower()
    return disabled not in ("1", "true", "yes")


class LazyModule:
    """Wrapper for deferred module imports.

    Usage:
        jax = LazyModule("jax")
        # jax is not imported yet

        arr = jax.numpy.array([1, 2, 3])
        # jax is imported on first attribute access

    Attributes:
        _module_name: Fully qualified module name
        _module: Cached module reference (None until loaded)
    """

    def __init__(self, module_name: str) -> None:
        """Initialize lazy module wrapper.

        Args:
            module_name: Fully qualified module name (e.g., "jax", "matplotlib.pyplot")
        """
        # Use object.__setattr__ to avoid triggering __getattr__
        object.__setattr__(self, "_module_name", module_name)
        object.__setattr__(self, "_module", None)

    def _load(self) -> ModuleType:
        """Load the module if not already loaded.

        Returns:
            The loaded module

        Raises:
            ImportError: If module cannot be imported
        """
        module = object.__getattribute__(self, "_module")
        if module is None:
            module_name = object.__getattribute__(self, "_module_name")
            start = time.perf_counter()
            try:
                module = importlib.import_module(module_name)
                elapsed = (time.perf_counter() - start) * 1000
                logger.debug(f"Lazy loaded {module_name} in {elapsed:.1f}ms")
            except ImportError as e:
                logger.error(f"Failed to lazy load {module_name}: {e}")
                raise
            object.__setattr__(self, "_module", module)
        return module

    def __getattr__(self, name: str) -> object:
        """Load module on first attribute access and delegate.

        Args:
            name: Attribute name to access

        Returns:
            Attribute from the loaded module

        Raises:
            ImportError: If module cannot be imported
            AttributeError: If module doesn't have the attribute
        """
        module = self._load()
        return getattr(module, name)

    def __repr__(self) -> str:
        module_name = object.__getattribute__(self, "_module_name")
        module = object.__getattribute__(self, "_module")
        status = "loaded" if module is not None else "not loaded"
        return f"<LazyModule({module_name!r}) [{status}]>"

    @property
    def is_loaded(self) -> bool:
        """Check if the module has been loaded."""
        return object.__getattribute__(self, "_module") is not None
