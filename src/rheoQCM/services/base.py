"""Base service container for dependency injection.

Feature: 011-tech-debt-cleanup

This module defines the ServiceContainer dataclass for managing
application services extracted from rheoQCM.py.

Provides:
- Protocol definitions for service contracts
- Default implementations for testability
- ServiceContainer for dependency injection
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# =============================================================================
# Protocol Definitions
# =============================================================================


class DataService(Protocol):
    """Data management service protocol."""

    def get_current_data(self) -> Any | None:
        """Get currently loaded experiment data."""
        ...

    def set_current_data(self, data: Any) -> None:
        """Set current experiment data."""
        ...

    def clear_cache(self) -> None:
        """Clear data cache."""
        ...


class AnalysisService(Protocol):
    """Analysis orchestration service protocol."""

    def run_analysis(self, harmonics: list[int]) -> Any:
        """Run analysis on current data."""
        ...

    def get_last_result(self) -> Any | None:
        """Get most recent analysis result."""
        ...


class ExportService(Protocol):
    """Export coordination service protocol."""

    def export(self, data: Any, path: Path, format: str | None = None) -> None:
        """Export data to file."""
        ...

    def get_supported_formats(self) -> list[str]:
        """Get list of supported export format extensions."""
        ...


# =============================================================================
# Default Implementations
# =============================================================================


class DefaultDataService:
    """Default in-memory data service implementation.

    Provides simple data caching for testability. Production code
    should use DataSaver for full QCM data file handling.
    """

    def __init__(self) -> None:
        self._data: Any | None = None

    def get_current_data(self) -> Any | None:
        """Get currently loaded experiment data."""
        return self._data

    def set_current_data(self, data: Any) -> None:
        """Set current experiment data."""
        self._data = data

    def clear_cache(self) -> None:
        """Clear data cache."""
        self._data = None


class DefaultAnalysisService:
    """Default analysis service implementation.

    Wraps QCMModel for analysis operations. Caches last result
    for retrieval.
    """

    def __init__(self, model: Any | None = None) -> None:
        self._model = model
        self._last_result: Any | None = None

    def run_analysis(self, harmonics: list[int]) -> Any:
        """Run analysis using QCMModel.

        Parameters
        ----------
        harmonics : list[int]
            List of harmonic numbers to analyze (e.g., [3, 5, 7]).

        Returns
        -------
        Any
            Analysis result (SolveResult or BatchResult).
        """
        if self._model is None:
            from rheoQCM.core.model import QCMModel

            self._model = QCMModel()

        result = self._model.solve_properties(nh=harmonics)
        self._last_result = result
        return result

    def get_last_result(self) -> Any | None:
        """Get most recent analysis result."""
        return self._last_result


class DefaultExportService:
    """Default export service using IO handlers.

    Uses rheoQCM.io handlers for format-agnostic export.
    """

    def __init__(self) -> None:
        self._supported_formats = [".h5", ".hdf5", ".xlsx", ".xls", ".json"]

    def export(self, data: Any, path: Path, format: str | None = None) -> None:
        """Export data to file using appropriate handler.

        Parameters
        ----------
        data : Any
            Data to export (dict, DataFrame, etc.).
        path : Path
            Output file path.
        format : str | None
            Optional format override (e.g., ".xlsx").
            If None, format is detected from path extension.
        """
        from rheoQCM.io import save_data

        save_data(data, path)

    def get_supported_formats(self) -> list[str]:
        """Get list of supported export format extensions."""
        return self._supported_formats.copy()


# =============================================================================
# Service Container
# =============================================================================


@dataclass
class ServiceContainer:
    """Container for all application services.

    This container provides dependency injection for the QCMApp,
    enabling testability and modular architecture.

    Example
    -------
    >>> container = ServiceContainer.create()
    >>> data = container.data.get_current_data()
    >>> container.export.export({"key": "value"}, Path("out.json"))
    """

    data: DataService | None = field(default=None)
    analysis: AnalysisService | None = field(default=None)
    hardware: Any | None = field(default=None)
    settings: Any | None = field(default=None)
    export: ExportService | None = field(default=None)

    @classmethod
    def create(
        cls,
        config: dict[str, Any] | None = None,
        *,
        use_defaults: bool = True,
    ) -> "ServiceContainer":
        """Factory method to create configured service container.

        Parameters
        ----------
        config : dict[str, Any] | None
            Optional configuration dictionary. Can include:
            - 'data': DataService instance
            - 'analysis': AnalysisService instance
            - 'export': ExportService instance
            - 'hardware': HardwareService instance
            - 'settings': SettingsRepository instance
        use_defaults : bool
            If True, create default implementations for missing services.
            If False, leave missing services as None.

        Returns
        -------
        ServiceContainer
            Configured service container.
        """
        config = config or {}

        data = config.get("data")
        analysis = config.get("analysis")
        export = config.get("export")
        hardware = config.get("hardware")
        settings = config.get("settings")

        if use_defaults:
            if data is None:
                data = DefaultDataService()
            if analysis is None:
                analysis = DefaultAnalysisService()
            if export is None:
                export = DefaultExportService()

        return cls(
            data=data,
            analysis=analysis,
            hardware=hardware,
            settings=settings,
            export=export,
        )
