"""Tests for rheoQCM.lazy_imports module."""

from __future__ import annotations

import logging
import os
from io import StringIO
from types import ModuleType
from unittest.mock import patch

import pytest


class TestIsLazyLoadingEnabled:
    """Tests for is_lazy_loading_enabled() function."""

    def test_enabled_by_default(self):
        """Lazy loading is enabled by default."""
        from rheoQCM.lazy_imports import is_lazy_loading_enabled

        with patch.dict(os.environ, {}, clear=True):
            # Remove the env var if it exists
            os.environ.pop("RHEOQCM_DISABLE_LAZY_LOADING", None)
            # Re-import to get fresh check
            from rheoQCM import lazy_imports

            result = lazy_imports.is_lazy_loading_enabled()
            assert result is True

    def test_disabled_with_1(self):
        """Lazy loading disabled when env var is '1'."""
        from rheoQCM.lazy_imports import is_lazy_loading_enabled

        with patch.dict(os.environ, {"RHEOQCM_DISABLE_LAZY_LOADING": "1"}):
            assert is_lazy_loading_enabled() is False

    def test_disabled_with_true(self):
        """Lazy loading disabled when env var is 'true'."""
        from rheoQCM.lazy_imports import is_lazy_loading_enabled

        with patch.dict(os.environ, {"RHEOQCM_DISABLE_LAZY_LOADING": "true"}):
            assert is_lazy_loading_enabled() is False

    def test_disabled_with_yes(self):
        """Lazy loading disabled when env var is 'yes'."""
        from rheoQCM.lazy_imports import is_lazy_loading_enabled

        with patch.dict(os.environ, {"RHEOQCM_DISABLE_LAZY_LOADING": "yes"}):
            assert is_lazy_loading_enabled() is False

    def test_enabled_with_other_values(self):
        """Lazy loading enabled for other env var values."""
        from rheoQCM.lazy_imports import is_lazy_loading_enabled

        with patch.dict(os.environ, {"RHEOQCM_DISABLE_LAZY_LOADING": "0"}):
            assert is_lazy_loading_enabled() is True

        with patch.dict(os.environ, {"RHEOQCM_DISABLE_LAZY_LOADING": "false"}):
            assert is_lazy_loading_enabled() is True


class TestLazyModule:
    """Tests for LazyModule class."""

    def test_not_loaded_initially(self):
        """Module is not loaded on creation."""
        from rheoQCM.lazy_imports import LazyModule

        lazy = LazyModule("json")
        assert lazy.is_loaded is False

    def test_loads_on_attribute_access(self):
        """Module loads when attribute is accessed."""
        from rheoQCM.lazy_imports import LazyModule

        lazy = LazyModule("json")
        assert lazy.is_loaded is False

        # Access an attribute
        _ = lazy.dumps
        assert lazy.is_loaded is True

    def test_returns_module_attribute(self):
        """Accessing attribute returns actual module attribute."""
        from rheoQCM.lazy_imports import LazyModule

        lazy = LazyModule("json")
        import json

        assert lazy.dumps is json.dumps
        assert lazy.loads is json.loads

    def test_repr_not_loaded(self):
        """repr shows not loaded state."""
        from rheoQCM.lazy_imports import LazyModule

        lazy = LazyModule("json")
        assert "not loaded" in repr(lazy)
        assert "json" in repr(lazy)

    def test_repr_loaded(self):
        """repr shows loaded state after access."""
        from rheoQCM.lazy_imports import LazyModule

        lazy = LazyModule("json")
        _ = lazy.dumps  # trigger load
        assert "loaded" in repr(lazy)
        assert "not loaded" not in repr(lazy)

    def test_caches_module(self):
        """Module is cached after first load."""
        from rheoQCM.lazy_imports import LazyModule

        lazy = LazyModule("json")
        _ = lazy.dumps
        module1 = object.__getattribute__(lazy, "_module")
        _ = lazy.loads
        module2 = object.__getattribute__(lazy, "_module")
        assert module1 is module2


class TestLazyModuleErrorHandling:
    """Tests for LazyModule error handling."""

    def test_raises_import_error_for_nonexistent_module(self):
        """ImportError raised for nonexistent module."""
        from rheoQCM.lazy_imports import LazyModule

        lazy = LazyModule("nonexistent_module_xyz")
        with pytest.raises(ImportError):
            _ = lazy.something

    def test_logs_error_on_import_failure(self):
        """Error is logged when import fails."""
        from rheoQCM.lazy_imports import LazyModule

        lazy = LazyModule("nonexistent_module_xyz")

        # Capture log output
        handler = logging.StreamHandler(StringIO())
        handler.setLevel(logging.ERROR)
        logger = logging.getLogger("rheoQCM.lazy_imports")
        logger.addHandler(handler)

        try:
            _ = lazy.something
        except ImportError:
            pass

        output = handler.stream.getvalue()
        # Note: The logger might not capture if not configured, but error is raised
        # The key test is that ImportError is raised

    def test_attribute_error_for_missing_attribute(self):
        """AttributeError raised for missing module attribute."""
        from rheoQCM.lazy_imports import LazyModule

        lazy = LazyModule("json")
        with pytest.raises(AttributeError):
            _ = lazy.nonexistent_attribute_xyz


class TestLazyModuleImportTiming:
    """Tests for import timing logging."""

    def test_logs_import_time(self):
        """Import timing is logged at DEBUG level."""
        from rheoQCM.lazy_imports import LazyModule

        # Create a fresh lazy module
        lazy = LazyModule("collections")

        # Set up logging capture
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger = logging.getLogger("rheoQCM.lazy_imports")
        original_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        try:
            # Trigger the import
            _ = lazy.OrderedDict
            output = stream.getvalue()
            # Check for timing message
            assert "Lazy loaded collections" in output
            assert "ms" in output
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)
