"""Format handlers for RheoQCM data files.

Feature: 011-tech-debt-cleanup

This package provides modular format handlers for different file types,
extracted from the monolithic DataSaver.py for better testability.

Usage:
    from rheoQCM.io import get_handler, save_data, load_data

    # Using the registry
    handler = get_handler(Path("data.h5"))
    handler.save(data, path)

    # Or use convenience functions
    save_data(data, Path("data.xlsx"))
    data = load_data(Path("data.h5"))
"""

from pathlib import Path
from typing import Any

from rheoQCM.io.base import (
    FormatHandler,
    get_handler,
    get_supported_extensions,
    register_handler,
)
from rheoQCM.io.excel_handler import ExcelHandler
from rheoQCM.io.hdf5_handler import HDF5Handler, check_hdf5_format
from rheoQCM.io.json_handler import JSONHandler

# Register built-in handlers
_hdf5_handler = HDF5Handler()
_excel_handler = ExcelHandler()
_json_handler = JSONHandler()

register_handler(_hdf5_handler)
register_handler(_excel_handler)
register_handler(_json_handler)


def save_data(data: dict[str, Any], path: Path, **options: Any) -> None:
    """Save data to file using appropriate handler.

    Parameters
    ----------
    data : dict[str, Any]
        Data to save.
    path : Path
        Output file path. Format determined by extension.
    **options : Any
        Format-specific options passed to handler.

    Raises
    ------
    ValueError
        If no handler supports the file extension.
    """
    handler = get_handler(path)
    handler.save(data, path, **options)


def load_data(path: Path, **options: Any) -> dict[str, Any]:
    """Load data from file using appropriate handler.

    Parameters
    ----------
    path : Path
        Input file path. Format determined by extension.
    **options : Any
        Format-specific options passed to handler.

    Returns
    -------
    dict[str, Any]
        Loaded data.

    Raises
    ------
    ValueError
        If no handler supports the file extension.
    FileNotFoundError
        If file does not exist.
    """
    handler = get_handler(path)
    return handler.load(path, **options)


__all__ = [
    # Base
    "FormatHandler",
    "get_handler",
    "get_supported_extensions",
    "register_handler",
    # Handlers
    "HDF5Handler",
    "ExcelHandler",
    "JSONHandler",
    # Utilities
    "check_hdf5_format",
    # Convenience functions
    "save_data",
    "load_data",
]
