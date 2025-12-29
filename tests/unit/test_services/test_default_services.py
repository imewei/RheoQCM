"""Tests for Default Service Implementations.

Feature: 011-tech-debt-cleanup
Task: US6 - Service extraction verification

Tests the default implementations:
- DefaultDataService
- DefaultAnalysisService
- DefaultExportService
- ServiceContainer
"""

from pathlib import Path

import pytest

from rheoQCM.services.base import (
    DefaultAnalysisService,
    DefaultDataService,
    DefaultExportService,
    ServiceContainer,
)


class TestDefaultDataService:
    """Test suite for DefaultDataService."""

    @pytest.fixture
    def service(self) -> DefaultDataService:
        """Create DefaultDataService instance."""
        return DefaultDataService()

    def test_initially_empty(self, service: DefaultDataService) -> None:
        """Test that service starts with no data."""
        assert service.get_current_data() is None

    def test_set_and_get_data(self, service: DefaultDataService) -> None:
        """Test setting and getting data."""
        data = {"key": "value", "number": 42}
        service.set_current_data(data)

        assert service.get_current_data() == data

    def test_set_different_types(self, service: DefaultDataService) -> None:
        """Test setting different data types."""
        # Dict
        service.set_current_data({"a": 1})
        assert service.get_current_data() == {"a": 1}

        # List
        service.set_current_data([1, 2, 3])
        assert service.get_current_data() == [1, 2, 3]

        # String
        service.set_current_data("test")
        assert service.get_current_data() == "test"

        # None
        service.set_current_data(None)
        assert service.get_current_data() is None

    def test_clear_cache(self, service: DefaultDataService) -> None:
        """Test clearing the cache."""
        service.set_current_data({"data": "value"})
        service.clear_cache()

        assert service.get_current_data() is None

    def test_satisfies_protocol(self, service: DefaultDataService) -> None:
        """Test that DefaultDataService satisfies DataService protocol."""
        from rheoQCM.services.base import DataService

        # Protocol check (structural typing)
        def accept_data_service(s: DataService) -> None:
            s.get_current_data()
            s.set_current_data({})
            s.clear_cache()

        # Should not raise
        accept_data_service(service)


class TestDefaultAnalysisService:
    """Test suite for DefaultAnalysisService."""

    @pytest.fixture
    def service(self) -> DefaultAnalysisService:
        """Create DefaultAnalysisService instance."""
        return DefaultAnalysisService()

    def test_initially_no_result(self, service: DefaultAnalysisService) -> None:
        """Test that service starts with no result."""
        assert service.get_last_result() is None

    def test_with_mock_model(self) -> None:
        """Test with a mock model."""

        class MockModel:
            def solve_properties(self, nh):
                return {"harmonics": nh, "result": "success"}

        service = DefaultAnalysisService(model=MockModel())
        result = service.run_analysis([3, 5, 7])

        assert result == {"harmonics": [3, 5, 7], "result": "success"}
        assert service.get_last_result() == result

    def test_satisfies_protocol(self, service: DefaultAnalysisService) -> None:
        """Test that DefaultAnalysisService satisfies AnalysisService protocol."""
        from rheoQCM.services.base import AnalysisService

        def accept_analysis_service(s: AnalysisService) -> None:
            s.get_last_result()
            # Don't call run_analysis without a model

        accept_analysis_service(service)


class TestDefaultExportService:
    """Test suite for DefaultExportService."""

    @pytest.fixture
    def service(self) -> DefaultExportService:
        """Create DefaultExportService instance."""
        return DefaultExportService()

    def test_get_supported_formats(self, service: DefaultExportService) -> None:
        """Test getting supported formats."""
        formats = service.get_supported_formats()

        assert ".h5" in formats
        assert ".hdf5" in formats
        assert ".xlsx" in formats
        assert ".xls" in formats
        assert ".json" in formats

    def test_formats_returned_as_copy(self, service: DefaultExportService) -> None:
        """Test that modifying returned formats doesn't affect service."""
        formats = service.get_supported_formats()
        formats.append(".fake")

        assert ".fake" not in service.get_supported_formats()

    def test_export_json(self, service: DefaultExportService, tmp_path: Path) -> None:
        """Test exporting to JSON."""
        data = {"key": "value", "number": 42}
        output_path = tmp_path / "test.json"

        service.export(data, output_path)

        assert output_path.exists()

    def test_export_hdf5(self, service: DefaultExportService, tmp_path: Path) -> None:
        """Test exporting to HDF5."""
        data = {"key": "value"}
        output_path = tmp_path / "test.h5"

        service.export(data, output_path)

        assert output_path.exists()

    def test_satisfies_protocol(self, service: DefaultExportService) -> None:
        """Test that DefaultExportService satisfies ExportService protocol."""
        from rheoQCM.services.base import ExportService

        def accept_export_service(s: ExportService) -> None:
            s.get_supported_formats()

        accept_export_service(service)


class TestServiceContainer:
    """Test suite for ServiceContainer."""

    def test_create_with_defaults(self) -> None:
        """Test creating container with default services."""
        container = ServiceContainer.create()

        assert container.data is not None
        assert container.analysis is not None
        assert container.export is not None
        assert isinstance(container.data, DefaultDataService)
        assert isinstance(container.analysis, DefaultAnalysisService)
        assert isinstance(container.export, DefaultExportService)

    def test_create_without_defaults(self) -> None:
        """Test creating container without default services."""
        container = ServiceContainer.create(use_defaults=False)

        assert container.data is None
        assert container.analysis is None
        assert container.export is None

    def test_create_with_custom_services(self) -> None:
        """Test creating container with custom services."""
        custom_data = DefaultDataService()
        custom_data.set_current_data({"custom": True})

        container = ServiceContainer.create(
            config={"data": custom_data},
            use_defaults=False,
        )

        assert container.data is custom_data
        assert container.data.get_current_data() == {"custom": True}

    def test_create_with_partial_config(self) -> None:
        """Test creating container with partial config uses defaults for missing."""
        custom_data = DefaultDataService()

        container = ServiceContainer.create(config={"data": custom_data})

        assert container.data is custom_data
        # Other services should be defaults
        assert isinstance(container.analysis, DefaultAnalysisService)
        assert isinstance(container.export, DefaultExportService)

    def test_hardware_and_settings_optional(self) -> None:
        """Test that hardware and settings remain None by default."""
        container = ServiceContainer.create()

        # These don't have defaults in create()
        assert container.hardware is None
        assert container.settings is None

    def test_full_workflow(self, tmp_path: Path) -> None:
        """Test a full workflow using container services."""
        container = ServiceContainer.create()

        # Set data
        test_data = {"experiment": "QCM", "harmonics": [3, 5, 7]}
        container.data.set_current_data(test_data)

        # Get data
        retrieved = container.data.get_current_data()
        assert retrieved == test_data

        # Export data
        output_path = tmp_path / "output.json"
        container.export.export(test_data, output_path)
        assert output_path.exists()

        # Clear cache
        container.data.clear_cache()
        assert container.data.get_current_data() is None


class TestServiceContainerIntegration:
    """Integration tests for ServiceContainer with real services."""

    def test_import_from_package(self) -> None:
        """Test that services can be imported from package."""
        from rheoQCM.services import (
            AnalysisService,
            DataService,
            DefaultAnalysisService,
            DefaultDataService,
            DefaultExportService,
            ExportService,
            ServiceContainer,
        )

        # All should be importable
        assert DataService is not None
        assert AnalysisService is not None
        assert ExportService is not None
        assert DefaultDataService is not None
        assert DefaultAnalysisService is not None
        assert DefaultExportService is not None
        assert ServiceContainer is not None

    def test_container_from_package(self) -> None:
        """Test creating container from package import."""
        from rheoQCM.services import ServiceContainer

        container = ServiceContainer.create()
        assert container.data is not None
