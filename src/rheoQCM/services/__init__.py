"""
RheoQCM Services Package.

This package contains service interfaces for GUI decomposition:
- HardwareService: VNA and temperature device operations
- PlotManager: Matplotlib widget coordination
- SettingsRepository: Settings persistence with validation

These interfaces enable dependency injection and headless testing.
"""

from rheoQCM.services.hardware import (
    AcquisitionError,
    DefaultHardwareService,
    DeviceInfo,
    HardwareService,
    MockHardwareService,
    SweepResult,
)
from rheoQCM.services.plotting import (
    DefaultPlotManager,
    MockPlotManager,
    PlotCall,
    PlotManager,
    PlotStyle,
)
from rheoQCM.services.settings import (
    DEFAULT_SETTINGS,
    CorruptedFileError,
    JSONSettingsRepository,
    MockSettingsRepository,
    SettingsRepository,
    ValidationError,
)

__all__ = [
    # Hardware
    "HardwareService",
    "DefaultHardwareService",
    "MockHardwareService",
    "DeviceInfo",
    "SweepResult",
    "AcquisitionError",
    # Plotting
    "PlotManager",
    "DefaultPlotManager",
    "MockPlotManager",
    "PlotStyle",
    "PlotCall",
    # Settings
    "SettingsRepository",
    "JSONSettingsRepository",
    "MockSettingsRepository",
    "ValidationError",
    "CorruptedFileError",
    "DEFAULT_SETTINGS",
]
