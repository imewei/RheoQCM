"""Base classes for format handlers.

Feature: 011-tech-debt-cleanup

This module defines the abstract base class for all format handlers,
enabling modular and testable I/O operations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# Handler registry
_handlers: list["FormatHandler"] = []


class FormatHandler(ABC):
    """Abstract base class for format handlers.

    All format handlers must implement this interface to be compatible
    with the ExportService and the io/ module registry.
    """

    @property
    @abstractmethod
    def extensions(self) -> list[str]:
        """File extensions this handler supports.

        Returns
        -------
        list[str]
            List of extensions including the dot, e.g., [".h5", ".hdf5"]
        """
        ...

    @abstractmethod
    def save(self, data: dict[str, Any], path: Path, **options: Any) -> None:
        """Save data to file.

        Parameters
        ----------
        data : dict[str, Any]
            Data dictionary to save.
        path : Path
            Output file path.
        **options : Any
            Format-specific options.

        Raises
        ------
        OSError
            If file cannot be written.
        ValueError
            If data is invalid for this format.
        """
        ...

    @abstractmethod
    def load(self, path: Path, **options: Any) -> dict[str, Any]:
        """Load data from file.

        Parameters
        ----------
        path : Path
            Input file path.
        **options : Any
            Format-specific options.

        Returns
        -------
        dict[str, Any]
            Loaded data dictionary.

        Raises
        ------
        OSError
            If file cannot be read.
        ValueError
            If file format is invalid.
        """
        ...

    def can_handle(self, path: Path) -> bool:
        """Check if this handler can handle the given path.

        Parameters
        ----------
        path : Path
            File path to check.

        Returns
        -------
        bool
            True if this handler supports the file extension.
        """
        return path.suffix.lower() in self.extensions


def get_handler(path: Path) -> FormatHandler:
    """Get appropriate handler for the given file path.

    Parameters
    ----------
    path : Path
        File path to find handler for.

    Returns
    -------
    FormatHandler
        Handler that can process the file.

    Raises
    ------
    ValueError
        If no handler supports the file extension.
    """
    for handler in _handlers:
        if handler.can_handle(path):
            return handler
    raise ValueError(f"No handler for extension: {path.suffix}")


def register_handler(handler: FormatHandler) -> None:
    """Register a custom format handler.

    Parameters
    ----------
    handler : FormatHandler
        Handler instance to register.
    """
    _handlers.append(handler)


def get_supported_extensions() -> list[str]:
    """Get list of all supported file extensions.

    Returns
    -------
    list[str]
        All extensions supported by registered handlers.
    """
    extensions = []
    for handler in _handlers:
        extensions.extend(handler.extensions)
    return extensions
