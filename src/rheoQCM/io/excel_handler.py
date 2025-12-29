"""Excel format handler for data export.

Feature: 011-tech-debt-cleanup

This module provides Excel file handling for RheoQCM data export.
Supports both .xlsx and .xls formats via pandas/openpyxl.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rheoQCM.io.base import FormatHandler

logger = logging.getLogger(__name__)


class ExcelHandler(FormatHandler):
    """Handler for Excel (.xlsx, .xls) files.

    Uses pandas with openpyxl backend for Excel file operations.
    Data is organized as multiple sheets, one per top-level key.
    """

    @property
    def extensions(self) -> list[str]:
        """Returns [".xlsx", ".xls"]."""
        return [".xlsx", ".xls"]

    def save(self, data: dict[str, Any], path: Path, **options: Any) -> None:
        """Save data to Excel file.

        Each top-level key in data becomes a worksheet. Values should be:
        - pandas.DataFrame: Written directly
        - dict: Converted to DataFrame
        - list/array: Converted to DataFrame

        Parameters
        ----------
        data : dict[str, Any]
            Data dictionary to save. Keys become sheet names.
        path : Path
            Output file path.
        index : bool, optional
            Whether to include row index (default: True).
        """
        index = options.get("index", True)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for sheet_name, sheet_data in data.items():
                    # Sanitize sheet name (Excel has 31 char limit)
                    safe_name = str(sheet_name)[:31]

                    # Convert to DataFrame if needed
                    df = self._to_dataframe(sheet_data)

                    df.to_excel(writer, sheet_name=safe_name, index=index)

            logger.debug("Saved Excel to %s with %d sheets", path, len(data))

        except PermissionError as e:
            raise OSError(f"Cannot write to {path}. File may be open.") from e

    def load(self, path: Path, **options: Any) -> dict[str, Any]:
        """Load data from Excel file.

        Each worksheet becomes a key in the returned dictionary.

        Parameters
        ----------
        path : Path
            Input file path.
        sheet_name : str | int | list | None, optional
            Sheet(s) to load. None (default) loads all sheets.

        Returns
        -------
        dict[str, Any]
            Dictionary with sheet names as keys, DataFrames as values.

        Raises
        ------
        FileNotFoundError
            If file does not exist.
        ValueError
            If file format is invalid.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {path}")

        sheet_name = options.get("sheet_name", None)

        try:
            # Load all sheets into dict
            excel_file = pd.ExcelFile(path)
            data = {}

            sheets_to_load = (
                excel_file.sheet_names if sheet_name is None else [sheet_name]
            )

            for sheet in sheets_to_load:
                if sheet in excel_file.sheet_names:
                    data[sheet] = pd.read_excel(excel_file, sheet_name=sheet)

            logger.debug("Loaded Excel from %s with %d sheets", path, len(data))
            return data

        except Exception as e:
            raise ValueError(f"Invalid Excel file {path}: {e}") from e

    def _to_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert various data types to DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, dict):
            # Try to create DataFrame from dict
            try:
                return pd.DataFrame(data)
            except ValueError:
                # Fall back to single-column from_dict
                return pd.DataFrame.from_dict(data, orient="index", columns=["value"])
        if isinstance(data, (list, np.ndarray)):
            return pd.DataFrame(data)
        if isinstance(data, (str, int, float)):
            return pd.DataFrame({"value": [data]})

        # Last resort: string representation
        return pd.DataFrame({"value": [str(data)]})
