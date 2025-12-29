"""Tests for JSON format handler.

Feature: 011-tech-debt-cleanup
Task: T067 - Create tests/unit/test_io/test_json_handler.py
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from rheoQCM.io.json_handler import JSONHandler, NumpyJSONEncoder, json_object_hook


class TestJSONHandler:
    """Test suite for JSONHandler."""

    @pytest.fixture
    def handler(self) -> JSONHandler:
        """Create JSON handler instance."""
        return JSONHandler()

    @pytest.fixture
    def temp_json(self, tmp_path: Path) -> Path:
        """Create temporary JSON file path."""
        return tmp_path / "test.json"

    def test_extensions(self, handler: JSONHandler) -> None:
        """Test that handler reports correct extensions."""
        assert handler.extensions == [".json"]

    def test_can_handle_json(self, handler: JSONHandler) -> None:
        """Test can_handle for .json files."""
        assert handler.can_handle(Path("test.json"))
        assert not handler.can_handle(Path("test.xlsx"))
        assert not handler.can_handle(Path("test.h5"))

    def test_save_simple_dict(self, handler: JSONHandler, temp_json: Path) -> None:
        """Test saving a simple dictionary."""
        data = {"key1": "value1", "key2": 123, "key3": 45.67}

        handler.save(data, temp_json)

        assert temp_json.exists()
        with open(temp_json) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_load_simple_dict(self, handler: JSONHandler, temp_json: Path) -> None:
        """Test loading a simple dictionary."""
        data = {"key1": "value1", "key2": 123}

        with open(temp_json, "w") as f:
            json.dump(data, f)

        loaded = handler.load(temp_json)
        assert loaded == data

    def test_roundtrip_nested_dict(self, handler: JSONHandler, temp_json: Path) -> None:
        """Test save/load roundtrip with nested dictionary."""
        data = {
            "settings": {
                "hardware": {"port": "COM3"},
                "analysis": {"f1": 5e6, "refh": 3},
            },
            "values": [1, 2, 3, 4, 5],
        }

        handler.save(data, temp_json)
        loaded = handler.load(temp_json)

        assert loaded == data

    def test_numpy_array_serialization(
        self, handler: JSONHandler, temp_json: Path
    ) -> None:
        """Test that numpy arrays are serialized to lists."""
        data = {"array": np.array([1.0, 2.0, 3.0])}

        handler.save(data, temp_json)
        loaded = handler.load(temp_json)

        assert loaded["array"] == [1.0, 2.0, 3.0]

    def test_numpy_scalar_serialization(
        self, handler: JSONHandler, temp_json: Path
    ) -> None:
        """Test that numpy scalars are serialized properly."""
        data = {
            "int64": np.int64(42),
            "float64": np.float64(3.14),
            "bool_": np.bool_(True),
        }

        handler.save(data, temp_json)
        loaded = handler.load(temp_json)

        assert loaded["int64"] == 42
        assert loaded["float64"] == pytest.approx(3.14)
        assert loaded["bool_"] is True

    def test_complex_number_serialization(
        self, handler: JSONHandler, temp_json: Path
    ) -> None:
        """Test that complex numbers are serialized and deserialized."""
        data = {"z": complex(1.0, 2.0)}

        handler.save(data, temp_json)
        loaded = handler.load(temp_json)

        assert loaded["z"] == complex(1.0, 2.0)

    def test_dataframe_serialization(
        self, handler: JSONHandler, temp_json: Path
    ) -> None:
        """Test that pandas DataFrames are serialized."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        data = {"df": df}

        handler.save(data, temp_json)
        loaded = handler.load(temp_json)

        # DataFrame should be reconstructed
        assert isinstance(loaded["df"], pd.DataFrame)
        pd.testing.assert_frame_equal(loaded["df"], df)

    def test_load_nonexistent_file(self, handler: JSONHandler, tmp_path: Path) -> None:
        """Test loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            handler.load(tmp_path / "nonexistent.json")

    def test_load_invalid_json(self, handler: JSONHandler, temp_json: Path) -> None:
        """Test loading invalid JSON raises ValueError."""
        with open(temp_json, "w") as f:
            f.write("not valid json {")

        with pytest.raises(ValueError, match="Invalid JSON"):
            handler.load(temp_json)

    def test_creates_parent_directory(
        self, handler: JSONHandler, tmp_path: Path
    ) -> None:
        """Test that save creates parent directories if needed."""
        nested_path = tmp_path / "nested" / "dir" / "test.json"
        data = {"key": "value"}

        handler.save(data, nested_path)

        assert nested_path.exists()

    def test_indent_option(self, handler: JSONHandler, temp_json: Path) -> None:
        """Test custom indent option."""
        data = {"key": "value"}

        handler.save(data, temp_json, indent=4)

        content = temp_json.read_text()
        # With indent=4, should have 4 spaces of indentation
        assert "    " in content


class TestNumpyJSONEncoder:
    """Test the custom JSON encoder."""

    def test_encode_ndarray(self) -> None:
        """Test encoding numpy arrays."""
        arr = np.array([1, 2, 3])
        encoded = json.dumps(arr, cls=NumpyJSONEncoder)
        assert encoded == "[1, 2, 3]"

    def test_encode_int64(self) -> None:
        """Test encoding numpy int64."""
        val = np.int64(42)
        encoded = json.dumps({"val": val}, cls=NumpyJSONEncoder)
        assert encoded == '{"val": 42}'

    def test_encode_float64(self) -> None:
        """Test encoding numpy float64."""
        val = np.float64(3.14)
        encoded = json.dumps({"val": val}, cls=NumpyJSONEncoder)
        decoded = json.loads(encoded)
        assert decoded["val"] == pytest.approx(3.14)

    def test_encode_complex(self) -> None:
        """Test encoding complex numbers."""
        z = complex(1, 2)
        encoded = json.dumps({"z": z}, cls=NumpyJSONEncoder)
        decoded = json.loads(encoded)
        assert decoded["z"]["__complex__"] is True
        assert decoded["z"]["real"] == 1
        assert decoded["z"]["imag"] == 2


class TestJSONObjectHook:
    """Test the custom JSON object hook."""

    def test_decode_complex(self) -> None:
        """Test decoding complex numbers."""
        data = '{"z": {"__complex__": true, "real": 1, "imag": 2}}'
        decoded = json.loads(data, object_hook=json_object_hook)
        assert decoded["z"] == complex(1, 2)

    def test_decode_dataframe(self) -> None:
        """Test decoding DataFrame."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df_dict = df.to_dict(orient="split")
        data = json.dumps({"df": {"__dataframe__": True, "data": df_dict}})
        decoded = json.loads(data, object_hook=json_object_hook)

        assert isinstance(decoded["df"], pd.DataFrame)

    def test_passthrough_regular_dict(self) -> None:
        """Test that regular dicts pass through unchanged."""
        data = '{"key": "value"}'
        decoded = json.loads(data, object_hook=json_object_hook)
        assert decoded == {"key": "value"}
