"""HDF5 format handler for experiment data.

Feature: 011-tech-debt-cleanup

This module provides HDF5 file handling for RheoQCM experiment data.
Maintains backward compatibility with existing .h5 files.
"""

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from rheoQCM.io.base import FormatHandler

logger = logging.getLogger(__name__)


class HDF5Handler(FormatHandler):
    """Handler for HDF5 (.h5, .hdf5) files.

    Uses h5py for reading/writing HDF5 format. Maintains backward
    compatibility with existing RheoQCM .h5 files.

    HDF5 structure:
    - Groups become dict keys
    - Datasets become values (arrays, strings, or JSON-encoded objects)
    - Attributes are preserved as metadata
    """

    @property
    def extensions(self) -> list[str]:
        """Returns [".h5", ".hdf5"]."""
        return [".h5", ".hdf5"]

    def save(self, data: dict[str, Any], path: Path, **options: Any) -> None:
        """Save data to HDF5 file.

        Parameters
        ----------
        data : dict[str, Any]
            Data dictionary to save.
        path : Path
            Output file path.
        compression : str, optional
            Compression algorithm (default: "gzip").
        compression_opts : int, optional
            Compression level (default: 4).
        mode : str, optional
            File mode ('w' for write, 'a' for append). Default: 'w'.
        """
        compression = options.get("compression", "gzip")
        compression_opts = options.get("compression_opts", 4)
        mode = options.get("mode", "w")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with h5py.File(path, mode) as fh:
                self._save_dict(fh, data, compression, compression_opts)

            logger.debug("Saved HDF5 to %s", path)

        except OSError as e:
            raise OSError(f"Cannot write HDF5 file {path}: {e}") from e

    def load(self, path: Path, **options: Any) -> dict[str, Any]:
        """Load data from HDF5 file.

        Parameters
        ----------
        path : Path
            Input file path.
        groups : list[str], optional
            Specific groups to load. None (default) loads all.

        Returns
        -------
        dict[str, Any]
            Loaded data dictionary.

        Raises
        ------
        FileNotFoundError
            If file does not exist.
        ValueError
            If file format is invalid.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {path}")

        groups = options.get("groups")

        try:
            with h5py.File(path, "r") as fh:
                if groups is not None:
                    data = {}
                    for group in groups:
                        if group in fh:
                            data[group] = self._load_item(fh[group])
                else:
                    data = self._load_group(fh)

            logger.debug("Loaded HDF5 from %s", path)
            return data

        except Exception as e:
            raise ValueError(f"Invalid HDF5 file {path}: {e}") from e

    def _save_dict(
        self,
        parent: h5py.Group,
        data: dict[str, Any],
        compression: str,
        compression_opts: int,
    ) -> None:
        """Recursively save dictionary to HDF5 group."""
        for key, value in data.items():
            # Clean key for HDF5 (remove problematic characters)
            safe_key = str(key).replace("/", "_")

            if isinstance(value, dict):
                # Create subgroup for nested dicts
                if safe_key in parent:
                    del parent[safe_key]
                group = parent.create_group(safe_key)
                self._save_dict(group, value, compression, compression_opts)

            elif isinstance(value, pd.DataFrame):
                # Store DataFrame as JSON string
                json_str = value.to_json()
                self._save_string(parent, safe_key, json_str)

            elif isinstance(value, np.ndarray):
                # Store numpy arrays with compression
                if safe_key in parent:
                    del parent[safe_key]
                parent.create_dataset(
                    safe_key,
                    data=value,
                    compression=compression,
                    compression_opts=compression_opts,
                )

            elif isinstance(value, (list, tuple)):
                # Convert to numpy array if possible
                try:
                    arr = np.array(value)
                    if arr.dtype.kind in ["U", "O"]:
                        # String or object array - store as JSON
                        self._save_string(parent, safe_key, json.dumps(value))
                    else:
                        if safe_key in parent:
                            del parent[safe_key]
                        parent.create_dataset(
                            safe_key,
                            data=arr,
                            compression=compression,
                            compression_opts=compression_opts,
                        )
                except (ValueError, TypeError):
                    # Fall back to JSON for complex objects
                    self._save_string(parent, safe_key, json.dumps(value))

            elif isinstance(value, (str, bytes)):
                self._save_string(parent, safe_key, value)

            elif isinstance(value, (int, float, np.integer, np.floating)):
                # Store scalars as simple datasets
                if safe_key in parent:
                    del parent[safe_key]
                parent.create_dataset(safe_key, data=value)

            elif value is None:
                # Store None as empty string with attribute marker
                if safe_key in parent:
                    del parent[safe_key]
                ds = parent.create_dataset(
                    safe_key, data="", dtype=h5py.special_dtype(vlen=str)
                )
                ds.attrs["_is_none"] = True

            else:
                # Fall back to JSON for unknown types
                try:
                    self._save_string(parent, safe_key, json.dumps(value))
                except (TypeError, ValueError):
                    logger.warning(
                        "Could not serialize %s of type %s", key, type(value)
                    )

    def _save_string(self, parent: h5py.Group, key: str, value: str | bytes) -> None:
        """Save string value to HDF5."""
        if key in parent:
            del parent[key]

        if isinstance(value, bytes):
            value = value.decode("utf-8")

        parent.create_dataset(key, data=value, dtype=h5py.special_dtype(vlen=str))

    def _load_group(self, group: h5py.Group) -> dict[str, Any]:
        """Recursively load HDF5 group to dictionary."""
        data = {}
        for key in group.keys():
            data[key] = self._load_item(group[key])
        return data

    def _load_item(self, item: h5py.Dataset | h5py.Group) -> Any:
        """Load individual HDF5 item."""
        if isinstance(item, h5py.Group):
            return self._load_group(item)

        # Dataset
        value = item[()]

        # Check for None marker
        if "_is_none" in item.attrs and item.attrs["_is_none"]:
            return None

        # Handle string datasets
        if isinstance(value, bytes):
            value = value.decode("utf-8")

        if isinstance(value, str):
            # Try to parse as JSON (for DataFrames and complex objects)
            try:
                parsed = json.loads(value)
                # Check if it looks like a DataFrame
                if isinstance(parsed, dict) and all(
                    isinstance(v, dict) for v in parsed.values()
                ):
                    try:
                        return pd.DataFrame(parsed)
                    except (ValueError, TypeError):
                        return parsed
                return parsed
            except json.JSONDecodeError:
                return value

        return value


def check_hdf5_format(path: Path) -> bool:
    """Check if file is a valid RheoQCM HDF5 file.

    Parameters
    ----------
    path : Path
        File path to check.

    Returns
    -------
    bool
        True if file is valid RheoQCM HDF5 format.
    """
    try:
        with h5py.File(path, "r") as fh:
            # Check for expected structure
            return "raw" in fh or "data" in fh or "settings" in fh
    except OSError:
        return False
