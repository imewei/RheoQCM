"""Tests for Analysis Service.

Feature: 011-tech-debt-cleanup
Task: T039 - Create tests/unit/test_services/test_analysis_service.py

Note: AnalysisService is defined as a Protocol in services/base.py.
This test file provides tests for future implementations.
"""

from typing import Any

import pytest

from rheoQCM.services.base import AnalysisService


class MockAnalysisService:
    """Mock implementation of AnalysisService for testing."""

    def __init__(self) -> None:
        self._last_result: Any = None
        self._run_count: int = 0

    def run_analysis(self, harmonics: list[int]) -> Any:
        """Run mock analysis."""
        self._run_count += 1
        self._last_result = {
            "harmonics": harmonics,
            "grho_refh": 1e8,
            "phi": 0.1,
            "drho": 1e-6,
            "success": True,
        }
        return self._last_result

    def get_last_result(self) -> Any | None:
        return self._last_result

    # Test helpers
    @property
    def run_count(self) -> int:
        return self._run_count

    def reset(self) -> None:
        self._last_result = None
        self._run_count = 0


class TestAnalysisServiceProtocol:
    """Test that AnalysisService Protocol can be satisfied."""

    def test_mock_implements_protocol(self) -> None:
        """Verify mock implements AnalysisService protocol."""
        service: AnalysisService = MockAnalysisService()

        assert hasattr(service, "run_analysis")
        assert hasattr(service, "get_last_result")

    def test_get_last_result_initially_none(self) -> None:
        """Test initial state is None."""
        service = MockAnalysisService()
        assert service.get_last_result() is None

    def test_run_analysis_returns_result(self) -> None:
        """Test running analysis returns result."""
        service = MockAnalysisService()
        result = service.run_analysis([3, 5, 7])

        assert result is not None
        assert result["harmonics"] == [3, 5, 7]

    def test_run_analysis_stores_last_result(self) -> None:
        """Test that analysis stores last result."""
        service = MockAnalysisService()
        service.run_analysis([3, 5, 7])

        assert service.get_last_result() is not None
        assert service.get_last_result()["harmonics"] == [3, 5, 7]

    def test_multiple_analyses_update_last_result(self) -> None:
        """Test multiple analyses update last result."""
        service = MockAnalysisService()

        service.run_analysis([3])
        service.run_analysis([3, 5, 7])

        assert service.get_last_result()["harmonics"] == [3, 5, 7]


class TestAnalysisServiceUseCases:
    """Use case tests for AnalysisService."""

    @pytest.fixture
    def service(self) -> MockAnalysisService:
        return MockAnalysisService()

    def test_typical_analysis_workflow(self, service: MockAnalysisService) -> None:
        """Test typical analysis workflow."""
        # Run analysis with standard harmonics
        result = service.run_analysis([3, 5, 7])

        # Check result structure
        assert "grho_refh" in result
        assert "phi" in result
        assert "drho" in result
        assert result["success"] is True

    def test_single_harmonic_analysis(self, service: MockAnalysisService) -> None:
        """Test analysis with single harmonic."""
        result = service.run_analysis([5])

        assert result["harmonics"] == [5]

    def test_many_harmonics_analysis(self, service: MockAnalysisService) -> None:
        """Test analysis with many harmonics."""
        harmonics = [1, 3, 5, 7, 9, 11, 13]
        result = service.run_analysis(harmonics)

        assert result["harmonics"] == harmonics

    def test_empty_harmonics_handling(self, service: MockAnalysisService) -> None:
        """Test handling of empty harmonics list."""
        result = service.run_analysis([])
        assert result["harmonics"] == []

    def test_analysis_tracking(self, service: MockAnalysisService) -> None:
        """Test that analysis runs are tracked."""
        service.run_analysis([3])
        service.run_analysis([5])
        service.run_analysis([7])

        assert service.run_count == 3

    def test_reset_clears_state(self, service: MockAnalysisService) -> None:
        """Test reset clears service state."""
        service.run_analysis([3])
        service.reset()

        assert service.get_last_result() is None
        assert service.run_count == 0
