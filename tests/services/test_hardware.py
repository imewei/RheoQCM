"""
Tests for HardwareService interface and implementations.

T047: Create test_hardware_service.py with tests for:
- Connection management
- Sweep acquisition (mock and error cases)
- Temperature reading
- Device info retrieval
"""

import time

import numpy as np
import pytest

from rheoQCM.services.hardware import (
    AcquisitionError,
    DefaultHardwareService,
    DeviceInfo,
    MockHardwareService,
    SweepResult,
)


class TestMockHardwareService:
    """Tests for MockHardwareService."""

    def test_connect_disconnect(self):
        """Test basic connection lifecycle."""
        service = MockHardwareService()

        assert not service.is_connected()
        assert service.connect()
        assert service.is_connected()

        service.disconnect()
        assert not service.is_connected()

    def test_acquire_sweep_not_connected(self):
        """Test sweep fails when not connected."""
        service = MockHardwareService()

        with pytest.raises(ConnectionError, match="Not connected"):
            service.acquire_sweep(4.95e6, 5.05e6, 401)

    def test_acquire_sweep_returns_data(self):
        """Test sweep returns valid data."""
        service = MockHardwareService()
        service.connect()

        result = service.acquire_sweep(4.95e6, 5.05e6, 401)

        assert isinstance(result, SweepResult)
        assert len(result.frequency) == 401
        assert len(result.real) == 401
        assert len(result.imag) == 401
        assert result.frequency[0] == pytest.approx(4.95e6)
        assert result.frequency[-1] == pytest.approx(5.05e6)
        assert result.timestamp > 0

    def test_acquire_sweep_with_custom_data(self):
        """Test sweep returns custom mock data when set."""
        custom_data = SweepResult(
            frequency=np.array([1e6, 2e6, 3e6]),
            real=np.array([0.1, 0.2, 0.3]),
            imag=np.array([0.01, 0.02, 0.03]),
            timestamp=12345.0,
        )
        service = MockHardwareService(sweep_data=custom_data)
        service.connect()

        result = service.acquire_sweep(4.95e6, 5.05e6, 401)

        np.testing.assert_array_equal(result.frequency, custom_data.frequency)
        np.testing.assert_array_equal(result.real, custom_data.real)
        np.testing.assert_array_equal(result.imag, custom_data.imag)

    def test_set_mock_sweep_data(self):
        """Test setting mock sweep data after initialization."""
        service = MockHardwareService()
        service.connect()

        custom_data = SweepResult(
            frequency=np.array([5e6]),
            real=np.array([0.5]),
            imag=np.array([0.05]),
            timestamp=time.time(),
        )
        service.set_mock_sweep_data(custom_data)

        result = service.acquire_sweep(4.95e6, 5.05e6, 1)
        np.testing.assert_array_equal(result.frequency, custom_data.frequency)

    def test_connect_failure_mode(self):
        """Test simulating connection failure."""
        service = MockHardwareService()
        service.set_connect_failure(True)

        with pytest.raises(ConnectionError, match="Mock connection failure"):
            service.connect()

    def test_sweep_failure_mode(self):
        """Test simulating sweep failure."""
        service = MockHardwareService()
        service.connect()
        service.set_sweep_failure(True)

        with pytest.raises(AcquisitionError, match="Mock sweep failure"):
            service.acquire_sweep(4.95e6, 5.05e6, 401)

    def test_read_temperature(self):
        """Test temperature reading."""
        service = MockHardwareService()

        # Default temperature
        assert service.read_temperature() == 25.0

        # Custom temperature
        service.set_mock_temperature(30.5)
        assert service.read_temperature() == 30.5

    def test_get_device_info_not_connected(self):
        """Test device info fails when not connected."""
        service = MockHardwareService()

        with pytest.raises(ConnectionError, match="Not connected"):
            service.get_device_info()

    def test_get_device_info(self):
        """Test device info when connected."""
        service = MockHardwareService()
        service.connect()

        info = service.get_device_info()

        assert isinstance(info, DeviceInfo)
        assert info.device_type == "mock"
        assert info.model == "MockVNA"
        assert info.serial_number == "MOCK-001"
        assert info.firmware_version == "1.0.0"


class TestDefaultHardwareService:
    """Tests for DefaultHardwareService."""

    def test_uses_mock_on_non_windows(self):
        """Test default service uses mock on non-Windows."""
        service = DefaultHardwareService()

        # On Linux, should auto-use mock mode
        assert service.connect()
        assert service.is_connected()

        # Should be able to acquire sweep (mock data)
        result = service.acquire_sweep(4.95e6, 5.05e6, 101)
        assert len(result.frequency) == 101

    def test_explicit_mock_mode(self):
        """Test explicit mock mode override."""
        service = DefaultHardwareService(use_mock=True)
        service.connect()

        info = service.get_device_info()
        assert info.device_type == "mock"

    def test_parameter_validation(self):
        """Test sweep parameter validation."""
        service = DefaultHardwareService(use_mock=True)
        service.connect()

        # f_start >= f_stop
        with pytest.raises(ValueError, match="f_start must be less than f_stop"):
            service.acquire_sweep(5.05e6, 4.95e6, 101)

        # n_points out of range
        with pytest.raises(ValueError, match="n_points must be between 1 and 10001"):
            service.acquire_sweep(4.95e6, 5.05e6, 0)

        with pytest.raises(ValueError, match="n_points must be between 1 and 10001"):
            service.acquire_sweep(4.95e6, 5.05e6, 20000)

    def test_disconnect_safe_when_not_connected(self):
        """Test disconnect is safe when not connected."""
        service = DefaultHardwareService(use_mock=True)
        # Should not raise
        service.disconnect()
        assert not service.is_connected()

    def test_temperature_returns_none_without_device(self):
        """Test temperature returns None when no device."""
        service = DefaultHardwareService(use_mock=True)
        service.connect()

        # No temp device configured
        assert service.read_temperature() is None


class TestSweepResult:
    """Tests for SweepResult dataclass."""

    def test_creation(self):
        """Test SweepResult creation."""
        result = SweepResult(
            frequency=np.array([1e6, 2e6]),
            real=np.array([0.1, 0.2]),
            imag=np.array([0.01, 0.02]),
            timestamp=12345.0,
        )

        assert len(result.frequency) == 2
        assert result.timestamp == 12345.0

    def test_complex_data_access(self):
        """Test accessing data as complex array."""
        result = SweepResult(
            frequency=np.array([1e6]),
            real=np.array([0.1]),
            imag=np.array([0.01]),
            timestamp=0.0,
        )

        # Complex representation
        complex_data = result.real + 1j * result.imag
        assert complex_data[0] == pytest.approx(0.1 + 0.01j)


class TestDeviceInfo:
    """Tests for DeviceInfo dataclass."""

    def test_creation(self):
        """Test DeviceInfo creation."""
        info = DeviceInfo(
            device_type="vna",
            model="TestVNA",
            serial_number="SN-123",
            firmware_version="2.0.0",
        )

        assert info.device_type == "vna"
        assert info.model == "TestVNA"
        assert info.serial_number == "SN-123"
        assert info.firmware_version == "2.0.0"
