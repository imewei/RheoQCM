"""Tests for Excel format handler.

Feature: 011-tech-debt-cleanup
Task: T066 - Create tests/unit/test_io/test_excel_handler.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rheoQCM.io.excel_handler import ExcelHandler


class TestExcelHandler:
    """Test suite for ExcelHandler."""

    @pytest.fixture
    def handler(self) -> ExcelHandler:
        """Create Excel handler instance."""
        return ExcelHandler()

    @pytest.fixture
    def temp_xlsx(self, tmp_path: Path) -> Path:
        """Create temporary Excel file path."""
        return tmp_path / "test.xlsx"

    def test_extensions(self, handler: ExcelHandler) -> None:
        """Test that handler reports correct extensions."""
        assert ".xlsx" in handler.extensions
        assert ".xls" in handler.extensions

    def test_can_handle_excel(self, handler: ExcelHandler) -> None:
        """Test can_handle for Excel files."""
        assert handler.can_handle(Path("test.xlsx"))
        assert handler.can_handle(Path("test.xls"))
        assert not handler.can_handle(Path("test.json"))
        assert not handler.can_handle(Path("test.h5"))

    def test_save_dataframe(self, handler: ExcelHandler, temp_xlsx: Path) -> None:
        """Test saving a DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        data = {"Sheet1": df}

        handler.save(data, temp_xlsx)

        assert temp_xlsx.exists()

    def test_load_dataframe(self, handler: ExcelHandler, temp_xlsx: Path) -> None:
        """Test loading a DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        df.to_excel(temp_xlsx, sheet_name="Sheet1", index=False)

        loaded = handler.load(temp_xlsx)

        assert "Sheet1" in loaded
        # Check structure and values, not exact dtypes (Excel may convert)
        pd.testing.assert_frame_equal(
            loaded["Sheet1"], df, check_dtype=False, check_exact=False
        )

    def test_roundtrip_multiple_sheets(
        self, handler: ExcelHandler, temp_xlsx: Path
    ) -> None:
        """Test save/load roundtrip with multiple sheets."""
        data = {
            "Data": pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
            "Config": pd.DataFrame({"param": ["a", "b"], "value": [1.0, 2.0]}),
        }

        handler.save(data, temp_xlsx, index=False)
        loaded = handler.load(temp_xlsx)

        assert set(loaded.keys()) == {"Data", "Config"}
        pd.testing.assert_frame_equal(loaded["Data"], data["Data"])

    def test_save_dict_converts_to_dataframe(
        self, handler: ExcelHandler, temp_xlsx: Path
    ) -> None:
        """Test that plain dicts are converted to DataFrames."""
        data = {"Sheet1": {"col1": [1, 2, 3], "col2": [4, 5, 6]}}

        handler.save(data, temp_xlsx)
        loaded = handler.load(temp_xlsx)

        assert "Sheet1" in loaded
        assert loaded["Sheet1"]["col1"].tolist() == [1, 2, 3]

    def test_save_numpy_array(self, handler: ExcelHandler, temp_xlsx: Path) -> None:
        """Test saving numpy arrays."""
        data = {"Sheet1": np.array([[1, 2], [3, 4]])}

        handler.save(data, temp_xlsx)
        loaded = handler.load(temp_xlsx)

        assert "Sheet1" in loaded

    def test_save_scalar(self, handler: ExcelHandler, temp_xlsx: Path) -> None:
        """Test saving scalar values."""
        data = {"Sheet1": 42}

        handler.save(data, temp_xlsx)
        loaded = handler.load(temp_xlsx)

        assert "Sheet1" in loaded

    def test_load_nonexistent_file(self, handler: ExcelHandler, tmp_path: Path) -> None:
        """Test loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            handler.load(tmp_path / "nonexistent.xlsx")

    def test_creates_parent_directory(
        self, handler: ExcelHandler, tmp_path: Path
    ) -> None:
        """Test that save creates parent directories if needed."""
        nested_path = tmp_path / "nested" / "dir" / "test.xlsx"
        data = {"Sheet1": pd.DataFrame({"a": [1]})}

        handler.save(data, nested_path)

        assert nested_path.exists()

    def test_sheet_name_truncation(
        self, handler: ExcelHandler, temp_xlsx: Path
    ) -> None:
        """Test that long sheet names are truncated to 31 chars."""
        long_name = "a" * 50
        data = {long_name: pd.DataFrame({"x": [1]})}

        handler.save(data, temp_xlsx)
        loaded = handler.load(temp_xlsx)

        # Sheet name should be truncated
        assert long_name[:31] in loaded

    def test_load_specific_sheet(self, handler: ExcelHandler, temp_xlsx: Path) -> None:
        """Test loading a specific sheet."""
        data = {
            "Sheet1": pd.DataFrame({"a": [1]}),
            "Sheet2": pd.DataFrame({"b": [2]}),
        }
        handler.save(data, temp_xlsx)

        loaded = handler.load(temp_xlsx, sheet_name="Sheet1")

        assert "Sheet1" in loaded
        # Should only have the requested sheet
        assert len(loaded) == 1

    def test_index_option(self, handler: ExcelHandler, temp_xlsx: Path) -> None:
        """Test the index option."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        data = {"Sheet1": df}

        # Save with index
        handler.save(data, temp_xlsx, index=True)
        loaded_with_index = handler.load(temp_xlsx)

        # Save without index
        handler.save(data, temp_xlsx, index=False)
        loaded_no_index = handler.load(temp_xlsx)

        # With index, there's an extra column
        assert len(loaded_with_index["Sheet1"].columns) > len(
            loaded_no_index["Sheet1"].columns
        )


class TestExcelHandlerEdgeCases:
    """Edge case tests for ExcelHandler."""

    @pytest.fixture
    def handler(self) -> ExcelHandler:
        return ExcelHandler()

    def test_empty_dataframe(self, handler: ExcelHandler, tmp_path: Path) -> None:
        """Test handling empty DataFrame."""
        temp_xlsx = tmp_path / "empty.xlsx"
        data = {"Empty": pd.DataFrame()}

        handler.save(data, temp_xlsx)
        loaded = handler.load(temp_xlsx)

        assert "Empty" in loaded

    def test_dataframe_with_nan(self, handler: ExcelHandler, tmp_path: Path) -> None:
        """Test handling DataFrame with NaN values."""
        temp_xlsx = tmp_path / "nan.xlsx"
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        data = {"WithNaN": df}

        handler.save(data, temp_xlsx)
        loaded = handler.load(temp_xlsx)

        assert np.isnan(loaded["WithNaN"]["a"].iloc[1])

    def test_dataframe_with_strings(
        self, handler: ExcelHandler, tmp_path: Path
    ) -> None:
        """Test handling DataFrame with string values."""
        temp_xlsx = tmp_path / "strings.xlsx"
        df = pd.DataFrame({"name": ["Alice", "Bob"], "value": [1, 2]})
        data = {"Strings": df}

        handler.save(data, temp_xlsx)
        loaded = handler.load(temp_xlsx)

        assert loaded["Strings"]["name"].tolist() == ["Alice", "Bob"]
