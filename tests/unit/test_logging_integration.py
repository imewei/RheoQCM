"""Integration tests for rheoQCM logging system."""

from __future__ import annotations

import logging
import os
import re
from io import StringIO
from unittest.mock import patch

import pytest


class TestLogOutputFormat:
    """Tests for log message format and level filtering."""

    @pytest.fixture(autouse=True)
    def reset_logging(self):
        """Reset logging configuration before each test."""
        import rheoQCM.logging_config as lc

        lc._logging_configured = False
        root_logger = logging.getLogger("rheoQCM")
        root_logger.handlers.clear()
        yield
        lc._logging_configured = False
        root_logger.handlers.clear()

    def test_log_format_contains_timestamp(self):
        """Log messages contain timestamp."""
        from rheoQCM.logging_config import configure_logging

        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        configure_logging()
        logger = logging.getLogger("rheoQCM.test")
        logger.addHandler(handler)
        logger.info("Test message")

        output = stream.getvalue()
        # Check for timestamp pattern (YYYY-MM-DD HH:MM:SS)
        assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", output)

    def test_log_format_contains_level(self):
        """Log messages contain level name."""
        from rheoQCM.logging_config import configure_logging

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        configure_logging()
        logger = logging.getLogger("rheoQCM.test")
        logger.addHandler(handler)
        logger.warning("Test warning")

        output = stream.getvalue()
        assert "WARNING" in output

    def test_log_format_contains_module_name(self):
        """Log messages contain module name."""
        from rheoQCM.logging_config import configure_logging

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        configure_logging()
        logger = logging.getLogger("rheoQCM.core.model")
        logger.addHandler(handler)
        logger.info("Test message")

        output = stream.getvalue()
        assert "rheoQCM.core.model" in output

    def test_level_filtering_info(self):
        """INFO level filters out DEBUG messages."""
        from rheoQCM.logging_config import configure_logging

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

        configure_logging(level="INFO")
        logger = logging.getLogger("rheoQCM.test.filter")
        logger.addHandler(handler)

        logger.debug("Debug message")
        logger.info("Info message")

        output = stream.getvalue()
        assert "Debug message" not in output
        assert "Info message" in output

    def test_level_filtering_debug(self):
        """DEBUG level includes all messages."""
        from rheoQCM.logging_config import configure_logging

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

        configure_logging(level="DEBUG")
        logger = logging.getLogger("rheoQCM.test.debug")
        logger.addHandler(handler)

        logger.debug("Debug message")
        logger.info("Info message")

        output = stream.getvalue()
        assert "Debug message" in output
        assert "Info message" in output


class TestHierarchicalLoggers:
    """Tests for hierarchical logger behavior."""

    def test_child_logger_inherits_config(self):
        """Child loggers inherit configuration from parent."""
        import rheoQCM.logging_config as lc

        lc._logging_configured = False

        from rheoQCM.logging_config import configure_logging

        configure_logging()

        parent = logging.getLogger("rheoQCM")
        child = logging.getLogger("rheoQCM.core.model")

        # Child should use parent's handlers via propagation
        assert child.parent is not None
        # Effective level should be inherited
        assert child.getEffectiveLevel() <= logging.DEBUG
