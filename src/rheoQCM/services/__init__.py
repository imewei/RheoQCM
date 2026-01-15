"""
RheoQCM Services Package.

This package contains service interfaces for GUI decomposition:
- DataService: Data management and caching
- AnalysisService: Analysis orchestration
- ExportService: Data export coordination
- HardwareService: VNA and temperature device operations
- PlotManager: Matplotlib widget coordination
- SettingsRepository: Settings persistence with validation
- ServiceContainer: Dependency injection container

These interfaces enable dependency injection and headless testing.
"""

from rheoQCM.services.base import (
    AnalysisService,
    DataService,
    DefaultAnalysisService,
    DefaultDataService,
    DefaultExportService,
    ExportService,
    ServiceContainer,
)
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
from rheoQCM.services.theming import (
    Theme,
    ThemeColors,
    ThemeManager,
    get_system_theme,
)

__all__ = [
    # Core Services (from base.py)
    "DataService",
    "DefaultDataService",
    "AnalysisService",
    "DefaultAnalysisService",
    "ExportService",
    "DefaultExportService",
    "ServiceContainer",
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
    # Theming
    "Theme",
    "ThemeColors",
    "ThemeManager",
    "get_system_theme",
]
