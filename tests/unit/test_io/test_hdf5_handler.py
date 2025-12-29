"""Tests for HDF5 format handler.

Feature: 011-tech-debt-cleanup
Task: T065 - Create tests/unit/test_io/test_hdf5_handler.py
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from rheoQCM.io.hdf5_handler import HDF5Handler, check_hdf5_format


class TestHDF5Handler:
    """Test suite for HDF5Handler."""

    @pytest.fixture
    def handler(self) -> HDF5Handler:
        """Create HDF5 handler instance."""
        return HDF5Handler()

    @pytest.fixture
    def temp_h5(self, tmp_path: Path) -> Path:
        """Create temporary HDF5 file path."""
        return tmp_path / "test.h5"

    def test_extensions(self, handler: HDF5Handler) -> None:
        """Test that handler reports correct extensions."""
        assert ".h5" in handler.extensions
        assert ".hdf5" in handler.extensions

    def test_can_handle_hdf5(self, handler: HDF5Handler) -> None:
        """Test can_handle for HDF5 files."""
        assert handler.can_handle(Path("test.h5"))
        assert handler.can_handle(Path("test.hdf5"))
        assert not handler.can_handle(Path("test.json"))
        assert not handler.can_handle(Path("test.xlsx"))

    def test_save_simple_dict(self, handler: HDF5Handler, temp_h5: Path) -> None:
        """Test saving a simple dictionary."""
        data = {"key1": "value1", "key2": 123}

        handler.save(data, temp_h5)

        assert temp_h5.exists()
        with h5py.File(temp_h5, "r") as fh:
            assert "key1" in fh
            assert "key2" in fh

    def test_load_simple_dict(self, handler: HDF5Handler, temp_h5: Path) -> None:
        """Test loading a simple dictionary."""
        with h5py.File(temp_h5, "w") as fh:
            fh.create_dataset("key1", data="value1")
            fh.create_dataset("key2", data=123)

        loaded = handler.load(temp_h5)
        assert loaded["key1"] == "value1"
        assert loaded["key2"] == 123

    def test_roundtrip_numpy_array(self, handler: HDF5Handler, temp_h5: Path) -> None:
        """Test save/load roundtrip with numpy arrays."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = {"array": arr}

        handler.save(data, temp_h5)
        loaded = handler.load(temp_h5)

        np.testing.assert_array_equal(loaded["array"], arr)

    def test_roundtrip_nested_dict(self, handler: HDF5Handler, temp_h5: Path) -> None:
        """Test save/load roundtrip with nested dictionary."""
        data = {
            "settings": {
                "hardware": {"port": "COM3", "baud": 9600},
                "analysis": {"f1": 5e6, "refh": 3},
            },
            "values": [1, 2, 3, 4, 5],
        }

        handler.save(data, temp_h5)
        loaded = handler.load(temp_h5)

        assert loaded["settings"]["hardware"]["port"] == "COM3"
        assert loaded["settings"]["analysis"]["f1"] == 5e6

    def test_roundtrip_dataframe(self, handler: HDF5Handler, temp_h5: Path) -> None:
        """Test save/load roundtrip with pandas DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        data = {"data": df}

        handler.save(data, temp_h5)
        loaded = handler.load(temp_h5)

        assert isinstance(loaded["data"], pd.DataFrame)
        # Check structure and values, not exact index type (JSON converts to string)
        pd.testing.assert_frame_equal(
            loaded["data"].reset_index(drop=True),
            df.reset_index(drop=True),
            check_dtype=False,
        )

    def test_roundtrip_2d_array(self, handler: HDF5Handler, temp_h5: Path) -> None:
        """Test save/load roundtrip with 2D numpy array."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        data = {"matrix": arr}

        handler.save(data, temp_h5)
        loaded = handler.load(temp_h5)

        np.testing.assert_array_equal(loaded["matrix"], arr)

    def test_roundtrip_scalar(self, handler: HDF5Handler, temp_h5: Path) -> None:
        """Test save/load roundtrip with scalar values."""
        data = {"int_val": 42, "float_val": 3.14}

        handler.save(data, temp_h5)
        loaded = handler.load(temp_h5)

        assert loaded["int_val"] == 42
        assert loaded["float_val"] == pytest.approx(3.14)

    def test_roundtrip_none(self, handler: HDF5Handler, temp_h5: Path) -> None:
        """Test save/load roundtrip with None values."""
        data = {"none_val": None}

        handler.save(data, temp_h5)
        loaded = handler.load(temp_h5)

        assert loaded["none_val"] is None

    def test_load_nonexistent_file(self, handler: HDF5Handler, tmp_path: Path) -> None:
        """Test loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            handler.load(tmp_path / "nonexistent.h5")

    def test_creates_parent_directory(
        self, handler: HDF5Handler, tmp_path: Path
    ) -> None:
        """Test that save creates parent directories if needed."""
        nested_path = tmp_path / "nested" / "dir" / "test.h5"
        data = {"key": "value"}

        handler.save(data, nested_path)

        assert nested_path.exists()

    def test_append_mode(self, handler: HDF5Handler, temp_h5: Path) -> None:
        """Test append mode preserves existing data."""
        # First save
        data1 = {"first": "value1"}
        handler.save(data1, temp_h5)

        # Second save with append
        data2 = {"second": "value2"}
        handler.save(data2, temp_h5, mode="a")

        loaded = handler.load(temp_h5)
        assert "first" in loaded
        assert "second" in loaded

    def test_load_specific_groups(self, handler: HDF5Handler, temp_h5: Path) -> None:
        """Test loading specific groups only."""
        data = {
            "group1": {"a": 1},
            "group2": {"b": 2},
            "group3": {"c": 3},
        }
        handler.save(data, temp_h5)

        loaded = handler.load(temp_h5, groups=["group1", "group3"])

        assert "group1" in loaded
        assert "group3" in loaded
        assert "group2" not in loaded

    def test_compression_option(self, handler: HDF5Handler, temp_h5: Path) -> None:
        """Test compression options work."""
        large_array = np.random.rand(1000, 100)
        data = {"large": large_array}

        handler.save(data, temp_h5, compression="gzip", compression_opts=9)

        loaded = handler.load(temp_h5)
        np.testing.assert_array_almost_equal(loaded["large"], large_array)


class TestHDF5HandlerEdgeCases:
    """Edge case tests for HDF5Handler."""

    @pytest.fixture
    def handler(self) -> HDF5Handler:
        return HDF5Handler()

    def test_empty_dict(self, handler: HDF5Handler, tmp_path: Path) -> None:
        """Test handling empty dictionary."""
        temp_h5 = tmp_path / "empty.h5"
        data: dict = {}

        handler.save(data, temp_h5)
        loaded = handler.load(temp_h5)

        assert loaded == {}

    def test_string_list(self, handler: HDF5Handler, tmp_path: Path) -> None:
        """Test handling list of strings."""
        temp_h5 = tmp_path / "strings.h5"
        data = {"names": ["Alice", "Bob", "Charlie"]}

        handler.save(data, temp_h5)
        loaded = handler.load(temp_h5)

        assert loaded["names"] == ["Alice", "Bob", "Charlie"]

    def test_mixed_list(self, handler: HDF5Handler, tmp_path: Path) -> None:
        """Test handling mixed-type list (falls back to JSON)."""
        temp_h5 = tmp_path / "mixed.h5"
        data = {"mixed": [1, "two", 3.0]}

        handler.save(data, temp_h5)
        loaded = handler.load(temp_h5)

        assert loaded["mixed"] == [1, "two", 3.0]

    def test_key_with_slash(self, handler: HDF5Handler, tmp_path: Path) -> None:
        """Test that keys with slashes are sanitized."""
        temp_h5 = tmp_path / "slash.h5"
        data = {"path/to/value": 42}

        handler.save(data, temp_h5)
        loaded = handler.load(temp_h5)

        # Slash should be replaced with underscore
        assert "path_to_value" in loaded

    def test_bytes_value(self, handler: HDF5Handler, tmp_path: Path) -> None:
        """Test handling bytes values."""
        temp_h5 = tmp_path / "bytes.h5"
        data = {"bytes_val": b"hello"}

        handler.save(data, temp_h5)
        loaded = handler.load(temp_h5)

        assert loaded["bytes_val"] == "hello"


class TestCheckHDF5Format:
    """Test the check_hdf5_format utility function."""

    def test_valid_rheoqcm_file(self, tmp_path: Path) -> None:
        """Test detection of valid RheoQCM HDF5 file."""
        h5_path = tmp_path / "valid.h5"

        with h5py.File(h5_path, "w") as fh:
            fh.create_group("raw")
            fh.create_group("data")

        assert check_hdf5_format(h5_path) is True

    def test_invalid_hdf5_file(self, tmp_path: Path) -> None:
        """Test detection of invalid HDF5 file."""
        h5_path = tmp_path / "invalid.h5"

        with h5py.File(h5_path, "w") as fh:
            fh.create_dataset("random", data=[1, 2, 3])

        assert check_hdf5_format(h5_path) is False

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test with nonexistent file."""
        assert check_hdf5_format(tmp_path / "nonexistent.h5") is False

    def test_non_hdf5_file(self, tmp_path: Path) -> None:
        """Test with non-HDF5 file."""
        text_file = tmp_path / "text.h5"
        text_file.write_text("not an hdf5 file")

        assert check_hdf5_format(text_file) is False
