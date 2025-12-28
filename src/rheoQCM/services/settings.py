"""
Settings Repository Interface and Implementations.

This module provides the SettingsRepository interface for settings persistence
with validation, enabling testing and alternative storage backends.

T054-T055: Implement SettingsRepository interface and MockSettingsRepository.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class CorruptedFileError(Exception):
    """Settings file is corrupted."""

    pass


@dataclass
class ValidationError:
    """Settings validation error."""

    key: str
    message: str
    value: Any


# Default settings schema
DEFAULT_SETTINGS: dict[str, Any] = {
    "hardware": {
        "vna_port": "auto",
        "temperature_enabled": False,
        "temperature_port": "",
    },
    "acquisition": {
        "f_start": 4.95e6,
        "f_stop": 5.05e6,
        "n_points": 401,
        "if_bandwidth": 1000.0,
        "harmonics": [1, 3, 5, 7, 9, 11, 13],
    },
    "analysis": {
        "f1": 5e6,
        "refh": 3,
        "calctype": "SLA",
        "auto_analyze": True,
    },
    "display": {
        "autoscale": True,
        "plot_style": "default",
        "theme": "light",
    },
    "logging": {
        "level": "INFO",
        "file_enabled": True,
        "file_path": "",  # Empty = default location
    },
}


class SettingsRepository(Protocol):
    """Interface for settings persistence."""

    def load(self) -> dict[str, Any]:
        """
        Load all settings from storage.

        Returns:
            Dict of all settings.

        Raises:
            FileNotFoundError: If settings file doesn't exist (returns defaults).
            CorruptedFileError: If settings file is corrupted.
        """
        ...

    def save(self, settings: dict[str, Any]) -> None:
        """
        Save all settings to storage.

        Args:
            settings: Dict of settings to save.

        Raises:
            ValidationError: If settings fail validation.
            OSError: If write fails.
        """
        ...

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a single setting.

        Args:
            key: Setting key (supports dot notation: "hardware.vna.port").
            default: Default value if key not found.

        Returns:
            Setting value or default.
        """
        ...

    def set(self, key: str, value: Any) -> None:
        """
        Set a single setting.

        Args:
            key: Setting key (supports dot notation).
            value: Value to set.

        Raises:
            ValidationError: If value fails validation.
        """
        ...

    def reset_defaults(self) -> None:
        """Reset all settings to defaults."""
        ...

    def validate(self, settings: dict[str, Any]) -> list[ValidationError]:
        """
        Validate settings without saving.

        Args:
            settings: Dict of settings to validate.

        Returns:
            List of validation errors (empty if valid).
        """
        ...

    @property
    def filepath(self) -> Path:
        """Path to settings file."""
        ...


class JSONSettingsRepository:
    """JSON-based settings repository."""

    def __init__(self, filepath: Path | None = None):
        self._filepath = filepath or self._default_filepath()
        self._cache: dict[str, Any] | None = None

    def _default_filepath(self) -> Path:
        from platformdirs import user_config_dir

        config_dir = Path(user_config_dir("RheoQCM", "RheoQCM"))
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "settings.json"

    @property
    def filepath(self) -> Path:
        return self._filepath

    def load(self) -> dict[str, Any]:
        if self._cache is not None:
            return copy.deepcopy(self._cache)

        if not self._filepath.exists():
            logger.debug("Settings file not found, using defaults")
            self._cache = copy.deepcopy(DEFAULT_SETTINGS)
            return copy.deepcopy(self._cache)

        try:
            with open(self._filepath) as f:
                loaded = json.load(f)
        except json.JSONDecodeError as e:
            raise CorruptedFileError(f"Settings file corrupted: {e}") from e

        # Merge with defaults to handle missing keys
        self._cache = self._merge_with_defaults(loaded)
        logger.debug("Loaded settings from %s", self._filepath)
        return copy.deepcopy(self._cache)

    def save(self, settings: dict[str, Any]) -> None:
        errors = self.validate(settings)
        if errors:
            raise ValueError(f"Validation failed: {errors}")

        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self._filepath, "w") as f:
            json.dump(settings, f, indent=2)

        self._cache = copy.deepcopy(settings)
        logger.info("Saved settings to %s", self._filepath)

    def get(self, key: str, default: Any = None) -> Any:
        settings = self.load()
        keys = key.split(".")
        value: Any = settings
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        settings = self.load()
        keys = key.split(".")

        # Navigate to parent
        target = settings
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        # Set value
        target[keys[-1]] = value
        self.save(settings)

    def reset_defaults(self) -> None:
        self._cache = copy.deepcopy(DEFAULT_SETTINGS)
        self.save(self._cache)

    def validate(self, settings: dict[str, Any]) -> list[ValidationError]:
        errors: list[ValidationError] = []

        # Validate analysis settings
        if "analysis" in settings:
            analysis = settings["analysis"]

            if "f1" in analysis:
                f1 = analysis["f1"]
                if not (1e6 <= f1 <= 10e6):
                    errors.append(
                        ValidationError(
                            key="analysis.f1",
                            message="f1 must be between 1 MHz and 10 MHz",
                            value=f1,
                        )
                    )

            if "refh" in analysis:
                refh = analysis["refh"]
                if refh not in [1, 3, 5, 7, 9, 11, 13]:
                    errors.append(
                        ValidationError(
                            key="analysis.refh",
                            message="refh must be odd harmonic (1, 3, 5, 7, 9, 11, 13)",
                            value=refh,
                        )
                    )

            if "calctype" in analysis:
                calctype = analysis["calctype"]
                if calctype not in ["SLA", "LL", "custom"]:
                    errors.append(
                        ValidationError(
                            key="analysis.calctype",
                            message="calctype must be SLA, LL, or custom",
                            value=calctype,
                        )
                    )

        # Validate acquisition settings
        if "acquisition" in settings:
            acq = settings["acquisition"]

            if "n_points" in acq:
                n_points = acq["n_points"]
                if not (1 <= n_points <= 10001):
                    errors.append(
                        ValidationError(
                            key="acquisition.n_points",
                            message="n_points must be between 1 and 10001",
                            value=n_points,
                        )
                    )

            if "if_bandwidth" in acq:
                if_bw = acq["if_bandwidth"]
                if not (1 <= if_bw <= 100000):
                    errors.append(
                        ValidationError(
                            key="acquisition.if_bandwidth",
                            message="if_bandwidth must be between 1 and 100000 Hz",
                            value=if_bw,
                        )
                    )

        # Validate logging settings
        if "logging" in settings:
            logging_settings = settings["logging"]

            if "level" in logging_settings:
                level = logging_settings["level"]
                if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                    errors.append(
                        ValidationError(
                            key="logging.level",
                            message="Invalid log level",
                            value=level,
                        )
                    )

        # Validate display settings
        if "display" in settings:
            display = settings["display"]

            if "theme" in display:
                theme = display["theme"]
                if theme not in ["light", "dark", "system"]:
                    errors.append(
                        ValidationError(
                            key="display.theme",
                            message="theme must be light, dark, or system",
                            value=theme,
                        )
                    )

        return errors

    def _merge_with_defaults(self, loaded: dict) -> dict:
        """Merge loaded settings with defaults for missing keys."""
        result = copy.deepcopy(DEFAULT_SETTINGS)
        self._deep_update(result, loaded)
        return result

    def _deep_update(self, base: dict, updates: dict) -> None:
        """Recursively update base dict with updates."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def invalidate_cache(self) -> None:
        """Invalidate the settings cache to force reload on next access."""
        self._cache = None


class MockSettingsRepository:
    """In-memory settings repository for testing."""

    def __init__(self, initial: dict[str, Any] | None = None):
        self._settings = initial or copy.deepcopy(DEFAULT_SETTINGS)
        self._filepath = Path("/mock/settings.json")
        self.save_count = 0
        self.load_count = 0

    @property
    def filepath(self) -> Path:
        return self._filepath

    def load(self) -> dict[str, Any]:
        self.load_count += 1
        return copy.deepcopy(self._settings)

    def save(self, settings: dict[str, Any]) -> None:
        self.save_count += 1
        self._settings = copy.deepcopy(settings)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value: Any = self._settings
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        keys = key.split(".")
        target = self._settings
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value
        self.save_count += 1

    def reset_defaults(self) -> None:
        self._settings = copy.deepcopy(DEFAULT_SETTINGS)

    def validate(self, settings: dict[str, Any]) -> list[ValidationError]:
        return []  # Mock always validates

    # Test helpers
    def set_settings(self, settings: dict[str, Any]) -> None:
        """Test helper to replace all settings."""
        self._settings = copy.deepcopy(settings)

    def reset_counters(self) -> None:
        """Test helper to reset load/save counters."""
        self.save_count = 0
        self.load_count = 0
