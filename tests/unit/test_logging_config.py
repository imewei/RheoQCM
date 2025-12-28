"""Tests for rheoQCM.logging_config module."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestGetLogDir:
    """Tests for get_log_dir() function."""

    def test_returns_path(self):
        """get_log_dir returns a Path object."""
        from rheoQCM.logging_config import get_log_dir

        result = get_log_dir()
        assert isinstance(result, Path)

    def test_directory_exists(self):
        """get_log_dir creates the directory if needed."""
        from rheoQCM.logging_config import get_log_dir

        result = get_log_dir()
        assert result.exists()
        assert result.is_dir()

    def test_ends_with_logs(self):
        """Log directory ends with 'logs' subdirectory."""
        from rheoQCM.logging_config import get_log_dir

        result = get_log_dir()
        assert result.name == "logs"


class TestConfigureLogging:
    """Tests for configure_logging() function."""

    @pytest.fixture(autouse=True)
    def reset_logging(self):
        """Reset logging configuration before each test."""
        import rheoQCM.logging_config as lc

        # Reset the configured flag
        lc._logging_configured = False

        # Clear handlers from rheoQCM logger
        root_logger = logging.getLogger("rheoQCM")
        root_logger.handlers.clear()

        yield

        # Cleanup after test
        lc._logging_configured = False
        root_logger.handlers.clear()

    def test_creates_handlers(self):
        """configure_logging creates console and file handlers."""
        from rheoQCM.logging_config import configure_logging

        configure_logging()

        logger = logging.getLogger("rheoQCM")
        # Should have at least 1 handler (console), possibly 2 (+ file)
        assert len(logger.handlers) >= 1

    def test_respects_level_argument(self):
        """configure_logging respects explicit level argument."""
        from rheoQCM.logging_config import configure_logging

        configure_logging(level="DEBUG")

        logger = logging.getLogger("rheoQCM")
        # Logger level is DEBUG (10) to capture all
        assert logger.level == logging.DEBUG

    def test_respects_environment_variable(self):
        """configure_logging reads RHEOQCM_LOG_LEVEL from environment."""
        import rheoQCM.logging_config as lc

        lc._logging_configured = False

        with patch.dict(os.environ, {"RHEOQCM_LOG_LEVEL": "WARNING"}):
            from rheoQCM.logging_config import configure_logging

            configure_logging()

            logger = logging.getLogger("rheoQCM")
            # Check that at least one handler has WARNING level
            levels = [h.level for h in logger.handlers]
            assert logging.WARNING in levels

    def test_invalid_level_uses_default(self):
        """Invalid log level falls back to INFO."""
        import rheoQCM.logging_config as lc

        lc._logging_configured = False

        with patch.dict(os.environ, {"RHEOQCM_LOG_LEVEL": "INVALID"}):
            from rheoQCM.logging_config import configure_logging

            configure_logging()

            logger = logging.getLogger("rheoQCM")
            # Should use INFO as default
            levels = [h.level for h in logger.handlers]
            assert logging.INFO in levels

    def test_only_configures_once(self):
        """configure_logging only runs once."""
        from rheoQCM.logging_config import configure_logging

        configure_logging()
        logger = logging.getLogger("rheoQCM")
        initial_handler_count = len(logger.handlers)

        # Call again
        configure_logging()
        assert len(logger.handlers) == initial_handler_count


class TestGetLogger:
    """Tests for get_logger() convenience function."""

    def test_returns_logger(self):
        """get_logger returns a Logger instance."""
        from rheoQCM.logging_config import get_logger

        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self):
        """get_logger returns logger with correct name."""
        from rheoQCM.logging_config import get_logger

        logger = get_logger("rheoQCM.test.module")
        assert logger.name == "rheoQCM.test.module"

    def test_ensures_configuration(self):
        """get_logger ensures logging is configured."""
        import rheoQCM.logging_config as lc

        lc._logging_configured = False

        from rheoQCM.logging_config import get_logger

        get_logger("test")
        assert lc._logging_configured is True
