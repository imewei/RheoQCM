"""
Hardware Service Interface and Implementations.

This module provides the HardwareService interface for VNA and temperature
device operations, enabling headless testing and hardware mocking.

T050-T051: Implement HardwareService interface and MockHardwareService.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Hardware device information."""

    device_type: str
    model: str
    serial_number: str
    firmware_version: str


@dataclass
class SweepResult:
    """Result of a frequency sweep acquisition."""

    frequency: np.ndarray  # Hz, shape (n_points,)
    real: np.ndarray  # Real component, shape (n_points,)
    imag: np.ndarray  # Imaginary component, shape (n_points,)
    timestamp: float  # Unix timestamp


class AcquisitionError(Exception):
    """Error during sweep acquisition."""

    pass


class HardwareService(Protocol):
    """Interface for VNA and temperature device operations."""

    def connect(self) -> bool:
        """
        Connect to hardware device.

        Returns:
            True if connection successful, False otherwise.

        Raises:
            ConnectionError: If connection fails with known error.
        """
        ...

    def disconnect(self) -> None:
        """
        Disconnect from hardware device.

        Safe to call even if not connected.
        """
        ...

    def is_connected(self) -> bool:
        """Check if hardware is connected."""
        ...

    def acquire_sweep(
        self,
        f_start: float,
        f_stop: float,
        n_points: int,
        *,
        if_bandwidth: float = 1000.0,
    ) -> SweepResult:
        """
        Acquire frequency sweep from VNA.

        Args:
            f_start: Start frequency in Hz.
            f_stop: Stop frequency in Hz.
            n_points: Number of frequency points.
            if_bandwidth: IF bandwidth in Hz (default 1000).

        Returns:
            SweepResult containing frequency and S-parameter data.

        Raises:
            ConnectionError: If not connected.
            AcquisitionError: If sweep fails.
            ValueError: If parameters out of range.
        """
        ...

    def read_temperature(self) -> float | None:
        """
        Read current temperature from sensor.

        Returns:
            Temperature in Celsius, or None if sensor unavailable.
        """
        ...

    def get_device_info(self) -> DeviceInfo:
        """
        Get information about connected device.

        Returns:
            DeviceInfo with device details.

        Raises:
            ConnectionError: If not connected.
        """
        ...


class DefaultHardwareService:
    """Default implementation using AccessMyVNA (Windows) or mock (other platforms)."""

    def __init__(self, use_mock: bool = False):
        self._connected = False
        self._use_mock = use_mock or not self._is_windows()
        self._vna = None
        self._temp_device = None

    def _is_windows(self) -> bool:
        return sys.platform == "win32"

    def connect(self) -> bool:
        if self._use_mock:
            logger.info("Using mock hardware (non-Windows platform)")
            self._connected = True
            return True

        try:
            from rheoQCM.modules.AccessMyVNA import VNA

            self._vna = VNA()
            self._connected = self._vna.connect()
            if self._connected:
                logger.info("Connected to VNA")
            return self._connected
        except ImportError:
            logger.warning("AccessMyVNA not available, using mock")
            self._use_mock = True
            self._connected = True
            return True
        except Exception as e:
            logger.error("Failed to connect to VNA: %s", e)
            raise ConnectionError(f"VNA connection failed: {e}") from e

    def disconnect(self) -> None:
        if self._vna is not None:
            try:
                self._vna.disconnect()
            except Exception as e:
                logger.warning("Error disconnecting VNA: %s", e)
        self._connected = False
        self._vna = None
        logger.info("Disconnected from hardware")

    def is_connected(self) -> bool:
        return self._connected

    def acquire_sweep(
        self,
        f_start: float,
        f_stop: float,
        n_points: int,
        *,
        if_bandwidth: float = 1000.0,
    ) -> SweepResult:
        if not self._connected:
            raise ConnectionError("Not connected to hardware")

        # Validate parameters
        if f_start >= f_stop:
            raise ValueError("f_start must be less than f_stop")
        if n_points < 1 or n_points > 10001:
            raise ValueError("n_points must be between 1 and 10001")

        if self._use_mock:
            # Generate mock data
            f = np.linspace(f_start, f_stop, n_points)
            return SweepResult(
                frequency=f,
                real=np.random.normal(0, 0.01, n_points),
                imag=np.random.normal(0, 0.01, n_points),
                timestamp=time.time(),
            )

        try:
            # Use real VNA
            result = self._vna.acquire_sweep(f_start, f_stop, n_points, if_bandwidth)
            return SweepResult(
                frequency=result.frequency,
                real=result.real,
                imag=result.imag,
                timestamp=time.time(),
            )
        except Exception as e:
            logger.error("Sweep acquisition failed: %s", e)
            raise AcquisitionError(f"Sweep failed: {e}") from e

    def read_temperature(self) -> float | None:
        if self._temp_device is None:
            return None
        try:
            return self._temp_device.read()
        except Exception as e:
            logger.warning("Temperature read failed: %s", e)
            return None

    def get_device_info(self) -> DeviceInfo:
        if not self._connected:
            raise ConnectionError("Not connected to hardware")

        if self._use_mock:
            return DeviceInfo(
                device_type="mock",
                model="MockVNA",
                serial_number="MOCK-001",
                firmware_version="1.0.0",
            )

        try:
            info = self._vna.get_info()
            return DeviceInfo(
                device_type="vna",
                model=info.get("model", "Unknown"),
                serial_number=info.get("serial", "Unknown"),
                firmware_version=info.get("firmware", "Unknown"),
            )
        except Exception as e:
            logger.warning("Could not get device info: %s", e)
            return DeviceInfo(
                device_type="vna",
                model="Unknown",
                serial_number="Unknown",
                firmware_version="Unknown",
            )


class MockHardwareService:
    """Mock implementation for testing."""

    def __init__(self, sweep_data: SweepResult | None = None):
        self._connected = False
        self._sweep_data = sweep_data
        self._temperature = 25.0
        self._connect_should_fail = False
        self._sweep_should_fail = False

    def connect(self) -> bool:
        if self._connect_should_fail:
            raise ConnectionError("Mock connection failure")
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def acquire_sweep(
        self,
        f_start: float,
        f_stop: float,
        n_points: int,
        *,
        if_bandwidth: float = 1000.0,
    ) -> SweepResult:
        if not self._connected:
            raise ConnectionError("Not connected")

        if self._sweep_should_fail:
            raise AcquisitionError("Mock sweep failure")

        if self._sweep_data:
            return self._sweep_data

        # Generate mock data
        f = np.linspace(f_start, f_stop, n_points)
        return SweepResult(
            frequency=f,
            real=np.random.normal(0, 0.01, n_points),
            imag=np.random.normal(0, 0.01, n_points),
            timestamp=time.time(),
        )

    def read_temperature(self) -> float | None:
        return self._temperature

    def get_device_info(self) -> DeviceInfo:
        if not self._connected:
            raise ConnectionError("Not connected")
        return DeviceInfo(
            device_type="mock",
            model="MockVNA",
            serial_number="MOCK-001",
            firmware_version="1.0.0",
        )

    # Test helpers
    def set_mock_temperature(self, temp: float) -> None:
        """Test helper to set mock temperature."""
        self._temperature = temp

    def set_mock_sweep_data(self, data: SweepResult) -> None:
        """Test helper to set mock sweep data."""
        self._sweep_data = data

    def set_connect_failure(self, should_fail: bool) -> None:
        """Test helper to make connect fail."""
        self._connect_should_fail = should_fail

    def set_sweep_failure(self, should_fail: bool) -> None:
        """Test helper to make sweep fail."""
        self._sweep_should_fail = should_fail
