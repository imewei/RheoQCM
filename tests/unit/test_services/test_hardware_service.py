"""Tests for Hardware Service.

Feature: 011-tech-debt-cleanup
Task: T037 - Create tests/unit/test_services/test_hardware_service.py
"""

import numpy as np
import pytest

from rheoQCM.services.hardware import (
    AcquisitionError,
    DeviceInfo,
    MockHardwareService,
    SweepResult,
)


class TestDeviceInfo:
    """Test DeviceInfo dataclass."""

    def test_device_info_fields(self) -> None:
        """Test DeviceInfo has correct fields."""
        info = DeviceInfo(
            device_type="VNA",
            model="myVNA",
            serial_number="12345",
            firmware_version="1.0.0",
        )

        assert info.device_type == "VNA"
        assert info.model == "myVNA"
        assert info.serial_number == "12345"
        assert info.firmware_version == "1.0.0"


class TestSweepResult:
    """Test SweepResult dataclass."""

    def test_sweep_result_fields(self) -> None:
        """Test SweepResult has correct fields."""
        freq = np.array([1e6, 2e6, 3e6])
        real = np.array([0.1, 0.2, 0.3])
        imag = np.array([0.01, 0.02, 0.03])

        result = SweepResult(
            frequency=freq, real=real, imag=imag, timestamp=1234567890.0
        )

        np.testing.assert_array_equal(result.frequency, freq)
        np.testing.assert_array_equal(result.real, real)
        np.testing.assert_array_equal(result.imag, imag)
        assert result.timestamp == 1234567890.0


class TestMockHardwareService:
    """Test suite for MockHardwareService."""

    @pytest.fixture
    def mock_service(self) -> MockHardwareService:
        """Create mock hardware service."""
        return MockHardwareService()

    def test_initially_disconnected(self, mock_service: MockHardwareService) -> None:
        """Test service starts disconnected."""
        assert mock_service.is_connected() is False

    def test_connect(self, mock_service: MockHardwareService) -> None:
        """Test connect operation."""
        result = mock_service.connect()

        assert result is True
        assert mock_service.is_connected() is True

    def test_disconnect(self, mock_service: MockHardwareService) -> None:
        """Test disconnect operation."""
        mock_service.connect()
        mock_service.disconnect()

        assert mock_service.is_connected() is False

    def test_disconnect_when_not_connected(
        self, mock_service: MockHardwareService
    ) -> None:
        """Test disconnect is safe when not connected."""
        mock_service.disconnect()  # Should not raise
        assert mock_service.is_connected() is False

    def test_acquire_sweep(self, mock_service: MockHardwareService) -> None:
        """Test acquiring sweep data."""
        mock_service.connect()

        result = mock_service.acquire_sweep(
            f_start=4.9e6, f_stop=5.1e6, n_points=101, if_bandwidth=1000
        )

        assert isinstance(result, SweepResult)
        assert len(result.frequency) == 101
        assert result.frequency[0] == pytest.approx(4.9e6)
        assert result.frequency[-1] == pytest.approx(5.1e6)

    def test_acquire_sweep_not_connected(
        self, mock_service: MockHardwareService
    ) -> None:
        """Test acquire_sweep raises when not connected."""
        with pytest.raises(ConnectionError):
            mock_service.acquire_sweep(f_start=4.9e6, f_stop=5.1e6, n_points=101)

    def test_get_device_info(self, mock_service: MockHardwareService) -> None:
        """Test getting device info."""
        mock_service.connect()
        info = mock_service.get_device_info()

        assert isinstance(info, DeviceInfo)
        assert info.device_type == "mock"
        assert info.model == "MockVNA"

    def test_get_device_info_not_connected(
        self, mock_service: MockHardwareService
    ) -> None:
        """Test get_device_info raises when not connected."""
        with pytest.raises(ConnectionError):
            mock_service.get_device_info()

    def test_read_temperature(self, mock_service: MockHardwareService) -> None:
        """Test reading temperature."""
        temp = mock_service.read_temperature()

        assert isinstance(temp, float)
        assert temp == 25.0  # Default mock temperature

    def test_set_mock_temperature(self, mock_service: MockHardwareService) -> None:
        """Test setting mock temperature."""
        mock_service.set_mock_temperature(30.5)

        assert mock_service.read_temperature() == 30.5

    def test_set_mock_sweep_data(self, mock_service: MockHardwareService) -> None:
        """Test configuring custom sweep response."""
        custom_freq = np.array([1e6, 2e6, 3e6])
        custom_real = np.array([1.0, 2.0, 3.0])
        custom_imag = np.array([0.1, 0.2, 0.3])
        custom_data = SweepResult(
            frequency=custom_freq,
            real=custom_real,
            imag=custom_imag,
            timestamp=0.0,
        )

        mock_service.set_mock_sweep_data(custom_data)
        mock_service.connect()
        result = mock_service.acquire_sweep(f_start=1e6, f_stop=3e6, n_points=3)

        np.testing.assert_array_equal(result.frequency, custom_freq)
        np.testing.assert_array_equal(result.real, custom_real)
        np.testing.assert_array_equal(result.imag, custom_imag)

    def test_set_sweep_failure(self, mock_service: MockHardwareService) -> None:
        """Test simulating acquisition failure."""
        mock_service.set_sweep_failure(True)
        mock_service.connect()

        with pytest.raises(AcquisitionError):
            mock_service.acquire_sweep(f_start=4.9e6, f_stop=5.1e6, n_points=101)

    def test_set_connect_failure(self, mock_service: MockHardwareService) -> None:
        """Test simulating connection failure."""
        mock_service.set_connect_failure(True)

        with pytest.raises(ConnectionError):
            mock_service.connect()


class TestSweepDataGeneration:
    """Test sweep data generation in mock service."""

    @pytest.fixture
    def connected_service(self) -> MockHardwareService:
        """Create connected mock service."""
        service = MockHardwareService()
        service.connect()
        return service

    def test_frequency_spacing(self, connected_service: MockHardwareService) -> None:
        """Test frequency points are evenly spaced."""
        result = connected_service.acquire_sweep(f_start=1e6, f_stop=2e6, n_points=11)

        expected_spacing = (2e6 - 1e6) / 10
        actual_spacing = np.diff(result.frequency)

        np.testing.assert_array_almost_equal(
            actual_spacing, np.full(10, expected_spacing)
        )

    def test_random_data_generated(
        self, connected_service: MockHardwareService
    ) -> None:
        """Test mock generates random data (not all zeros)."""
        result = connected_service.acquire_sweep(
            f_start=4.9e6, f_stop=5.1e6, n_points=201
        )

        # Should have some non-zero values
        assert np.any(result.real != 0)
        assert np.any(result.imag != 0)

    def test_single_point(self, connected_service: MockHardwareService) -> None:
        """Test single point acquisition."""
        result = connected_service.acquire_sweep(f_start=5e6, f_stop=5e6, n_points=1)

        assert len(result.frequency) == 1
        assert result.frequency[0] == 5e6
