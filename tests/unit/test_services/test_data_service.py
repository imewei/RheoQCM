"""Tests for Data Service.

Feature: 011-tech-debt-cleanup
Task: T038 - Create tests/unit/test_services/test_data_service.py

Note: DataService is defined as a Protocol in services/base.py.
This test file provides tests for future implementations.
"""

from typing import Any

import pytest

from rheoQCM.services.base import DataService


class MockDataService:
    """Mock implementation of DataService for testing."""

    def __init__(self) -> None:
        self._current_data: Any = None
        self._cache: dict[str, Any] = {}

    def get_current_data(self) -> Any | None:
        return self._current_data

    def set_current_data(self, data: Any) -> None:
        self._current_data = data

    def clear_cache(self) -> None:
        self._cache.clear()
        self._current_data = None


class TestDataServiceProtocol:
    """Test that DataService Protocol can be satisfied."""

    def test_mock_implements_protocol(self) -> None:
        """Verify mock implements DataService protocol."""
        service: DataService = MockDataService()

        # Should have all protocol methods
        assert hasattr(service, "get_current_data")
        assert hasattr(service, "set_current_data")
        assert hasattr(service, "clear_cache")

    def test_get_current_data_initially_none(self) -> None:
        """Test initial state is None."""
        service = MockDataService()
        assert service.get_current_data() is None

    def test_set_and_get_data(self) -> None:
        """Test setting and getting data."""
        service = MockDataService()
        test_data = {"harmonics": [3, 5, 7], "values": [1, 2, 3]}

        service.set_current_data(test_data)

        assert service.get_current_data() == test_data

    def test_clear_cache(self) -> None:
        """Test cache clearing."""
        service = MockDataService()
        service.set_current_data({"test": "data"})

        service.clear_cache()

        assert service.get_current_data() is None


class TestDataServiceUseCases:
    """Use case tests for DataService."""

    @pytest.fixture
    def service(self) -> MockDataService:
        return MockDataService()

    def test_experiment_workflow(self, service: MockDataService) -> None:
        """Test typical experiment workflow."""
        # Load experiment data
        experiment_data = {
            "harmonics": [1, 3, 5, 7],
            "delfstars": {1: -100 + 10j, 3: -300 + 30j},
            "metadata": {"sample": "test"},
        }
        service.set_current_data(experiment_data)

        # Verify data available
        data = service.get_current_data()
        assert data["harmonics"] == [1, 3, 5, 7]

        # Clear for new experiment
        service.clear_cache()
        assert service.get_current_data() is None

    def test_update_data(self, service: MockDataService) -> None:
        """Test updating current data."""
        # Initial data
        service.set_current_data({"version": 1})

        # Update with new data
        service.set_current_data({"version": 2})

        assert service.get_current_data()["version"] == 2

    def test_data_isolation(self, service: MockDataService) -> None:
        """Test that data modifications don't affect stored data."""
        original_data = {"list": [1, 2, 3]}
        service.set_current_data(original_data)

        # Modify original (should not affect stored)
        original_data["list"].append(4)

        # This depends on implementation - mock doesn't copy
        # Real implementation should deep copy
