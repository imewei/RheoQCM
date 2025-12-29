"""Tests for Export Service.

Feature: 011-tech-debt-cleanup
Task: T036 - Create tests/unit/test_services/test_export_service.py

Note: ExportService is defined as a Protocol in services/base.py.
This test file provides tests for future implementations.
"""

from pathlib import Path
from typing import Any

import pytest

from rheoQCM.services.base import ExportService


class MockExportService:
    """Mock implementation of ExportService for testing."""

    def __init__(self) -> None:
        self._supported_formats: list[str] = [".h5", ".hdf5", ".xlsx", ".json"]
        self._exports: list[tuple[Any, Path, str | None]] = []

    def export(self, data: Any, path: Path, format: str | None = None) -> None:
        """Mock export operation."""
        self._exports.append((data, path, format))

    def get_supported_formats(self) -> list[str]:
        return self._supported_formats.copy()

    # Test helpers
    @property
    def export_count(self) -> int:
        return len(self._exports)

    def get_export_history(self) -> list[tuple[Any, Path, str | None]]:
        return self._exports.copy()

    def reset(self) -> None:
        self._exports.clear()


class TestExportServiceProtocol:
    """Test that ExportService Protocol can be satisfied."""

    def test_mock_implements_protocol(self) -> None:
        """Verify mock implements ExportService protocol."""
        service: ExportService = MockExportService()

        assert hasattr(service, "get_supported_formats")

    def test_get_supported_formats(self) -> None:
        """Test getting supported formats."""
        service = MockExportService()
        formats = service.get_supported_formats()

        assert ".h5" in formats
        assert ".xlsx" in formats
        assert ".json" in formats


class TestExportServiceUseCases:
    """Use case tests for ExportService."""

    @pytest.fixture
    def service(self) -> MockExportService:
        return MockExportService()

    def test_export_to_hdf5(self, service: MockExportService) -> None:
        """Test export to HDF5 format."""
        data = {"test": "data"}
        path = Path("/tmp/test.h5")

        service.export(data, path)

        assert service.export_count == 1
        history = service.get_export_history()
        assert history[0][0] == data
        assert history[0][1] == path

    def test_export_with_explicit_format(self, service: MockExportService) -> None:
        """Test export with explicit format specification."""
        data = {"test": "data"}
        path = Path("/tmp/test.dat")

        service.export(data, path, format=".xlsx")

        history = service.get_export_history()
        assert history[0][2] == ".xlsx"

    def test_multiple_exports(self, service: MockExportService) -> None:
        """Test multiple export operations."""
        service.export({"a": 1}, Path("/tmp/a.h5"))
        service.export({"b": 2}, Path("/tmp/b.xlsx"))
        service.export({"c": 3}, Path("/tmp/c.json"))

        assert service.export_count == 3

    def test_export_history(self, service: MockExportService) -> None:
        """Test export history tracking."""
        service.export({"first": 1}, Path("/tmp/first.h5"))
        service.export({"second": 2}, Path("/tmp/second.h5"))

        history = service.get_export_history()

        assert len(history) == 2
        assert history[0][0]["first"] == 1
        assert history[1][0]["second"] == 2

    def test_reset_clears_history(self, service: MockExportService) -> None:
        """Test reset clears export history."""
        service.export({"data": 1}, Path("/tmp/test.h5"))
        service.reset()

        assert service.export_count == 0

    def test_format_autodetection_concept(self, service: MockExportService) -> None:
        """Conceptual test for format auto-detection from path."""
        # When format is None, it should be detected from path extension
        path_h5 = Path("/tmp/data.h5")
        path_xlsx = Path("/tmp/data.xlsx")

        service.export({"data": 1}, path_h5)  # format=None
        service.export({"data": 2}, path_xlsx)  # format=None

        history = service.get_export_history()
        # Both should have format=None, real impl would detect from extension
        assert history[0][2] is None
        assert history[1][2] is None


class TestSupportedFormats:
    """Test format support queries."""

    @pytest.fixture
    def service(self) -> MockExportService:
        return MockExportService()

    def test_hdf5_supported(self, service: MockExportService) -> None:
        """Test HDF5 formats are supported."""
        formats = service.get_supported_formats()
        assert ".h5" in formats
        assert ".hdf5" in formats

    def test_excel_supported(self, service: MockExportService) -> None:
        """Test Excel format is supported."""
        formats = service.get_supported_formats()
        assert ".xlsx" in formats

    def test_json_supported(self, service: MockExportService) -> None:
        """Test JSON format is supported."""
        formats = service.get_supported_formats()
        assert ".json" in formats

    def test_formats_returned_as_copy(self, service: MockExportService) -> None:
        """Test that modifying returned list doesn't affect service."""
        formats = service.get_supported_formats()
        formats.append(".fake")

        # Original should be unchanged
        assert ".fake" not in service.get_supported_formats()
