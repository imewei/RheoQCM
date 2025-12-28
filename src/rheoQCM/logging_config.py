"""
Centralized logging configuration for RheoQCM.

This module provides structured logging with:
- Console output (stderr) with configurable level
- Rotating file output with size-based rotation
- Hierarchical loggers for granular filtering
- Platform-appropriate log directory detection

Configuration:
    Set RHEOQCM_LOG_LEVEL environment variable to control log level.
    Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)

Example:
    RHEOQCM_LOG_LEVEL=DEBUG python my_script.py
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

from platformdirs import user_data_dir

# Type alias for log levels
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Default configuration
DEFAULT_LOG_LEVEL: LogLevel = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
MAX_LOG_BYTES = 10_485_760  # 10MB
BACKUP_COUNT = 3

# Module-level flag to prevent double initialization
_logging_configured = False


def get_log_dir() -> Path:
    """Get platform-appropriate log directory.

    Returns:
        Path to log directory (created if not exists)

    Platform behavior:
        - Linux: ~/.local/share/rheoqcm/logs/
        - macOS: ~/Library/Application Support/rheoqcm/logs/
        - Windows: %LOCALAPPDATA%/rheoqcm/logs/
    """
    log_dir = Path(user_data_dir("rheoqcm", ensure_exists=True)) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def configure_logging(
    level: LogLevel | None = None,
    console_level: LogLevel | None = None,
    file_level: LogLevel | None = None,
) -> None:
    """Configure the rheoQCM logging system.

    Args:
        level: Base log level (default: from RHEOQCM_LOG_LEVEL env or INFO)
        console_level: Console handler level (default: same as level)
        file_level: File handler level (default: same as level)

    Side effects:
        - Creates log directory if needed
        - Configures root 'rheoQCM' logger
        - Adds StreamHandler for console output (stderr)
        - Adds RotatingFileHandler for file output
    """
    global _logging_configured

    if _logging_configured:
        return

    # Determine log level from environment or argument
    if level is None:
        level = os.environ.get("RHEOQCM_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()  # type: ignore
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            level = DEFAULT_LOG_LEVEL

    console_level = console_level or level
    file_level = file_level or level

    # Get the root rheoQCM logger
    root_logger = logging.getLogger("rheoQCM")
    root_logger.setLevel(logging.DEBUG)  # Capture all, handlers filter

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    # Console handler (stderr)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    try:
        log_dir = get_log_dir()
        log_file = log_dir / "rheoqcm.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=MAX_LOG_BYTES,
            backupCount=BACKUP_COUNT,
        )
        file_handler.setLevel(getattr(logging, file_level))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except (PermissionError, OSError) as e:
        # Fallback: console-only logging
        root_logger.warning(
            f"Could not create log file: {e}. Using console-only logging."
        )

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.

    Convenience wrapper that ensures logging is configured.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured Logger instance
    """
    configure_logging()
    return logging.getLogger(name)
