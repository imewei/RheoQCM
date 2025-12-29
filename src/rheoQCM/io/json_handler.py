"""JSON format handler for settings and configuration files.

Feature: 011-tech-debt-cleanup

This module provides JSON file handling for RheoQCM settings
and configuration data.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rheoQCM.io.base import FormatHandler

logger = logging.getLogger(__name__)


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, complex):
            return {"__complex__": True, "real": obj.real, "imag": obj.imag}
        if isinstance(obj, pd.DataFrame):
            return {"__dataframe__": True, "data": obj.to_dict(orient="split")}
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)


def json_object_hook(obj: dict) -> Any:
    """Decode special JSON objects back to Python types."""
    if "__complex__" in obj:
        return complex(obj["real"], obj["imag"])
    if "__dataframe__" in obj:
        return pd.DataFrame(**obj["data"])
    return obj


class JSONHandler(FormatHandler):
    """Handler for JSON (.json) files.

    Uses standard library json module with custom encoders for
    NumPy and pandas types. Primarily for settings and configuration.
    """

    @property
    def extensions(self) -> list[str]:
        """Returns [".json"]."""
        return [".json"]

    def save(self, data: dict[str, Any], path: Path, **options: Any) -> None:
        """Save data to JSON file.

        Parameters
        ----------
        data : dict[str, Any]
            Data dictionary to save.
        path : Path
            Output file path.
        indent : int, optional
            JSON indentation level (default: 2).
        ensure_ascii : bool, optional
            Escape non-ASCII characters (default: False).
        """
        indent = options.get("indent", 2)
        ensure_ascii = options.get("ensure_ascii", False)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                cls=NumpyJSONEncoder,
                indent=indent,
                ensure_ascii=ensure_ascii,
            )

        logger.debug("Saved JSON to %s", path)

    def load(self, path: Path, **options: Any) -> dict[str, Any]:
        """Load data from JSON file.

        Parameters
        ----------
        path : Path
            Input file path.

        Returns
        -------
        dict[str, Any]
            Loaded data dictionary.

        Raises
        ------
        FileNotFoundError
            If file does not exist.
        ValueError
            If JSON is malformed.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f, object_hook=json_object_hook)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}") from e

        logger.debug("Loaded JSON from %s", path)
        return data
