"""Tests for Settings Service.

Feature: 011-tech-debt-cleanup
Task: T035 - Create tests/unit/test_services/test_settings_service.py
"""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from rheoQCM.services.settings import (
    DEFAULT_SETTINGS,
    CorruptedFileError,
    JSONSettingsRepository,
    MockSettingsRepository,
    SettingsRepository,
    ValidationError,
)


def get_default_settings() -> dict[str, Any]:
    """Get a fresh copy of default settings to avoid test pollution."""
    return copy.deepcopy(DEFAULT_SETTINGS)


class TestJSONSettingsRepository:
    """Test suite for JSONSettingsRepository."""

    @pytest.fixture
    def temp_settings(self, tmp_path: Path) -> Path:
        """Create temporary settings file path."""
        return tmp_path / "settings.json"

    @pytest.fixture
    def repo(self, temp_settings: Path) -> JSONSettingsRepository:
        """Create repository with temp file."""
        return JSONSettingsRepository(temp_settings)

    def test_load_defaults_when_file_missing(
        self, repo: JSONSettingsRepository
    ) -> None:
        """Test that load returns defaults when file doesn't exist."""
        settings = repo.load()
        assert settings == get_default_settings()

    def test_save_and_load(
        self, repo: JSONSettingsRepository, temp_settings: Path
    ) -> None:
        """Test save/load roundtrip."""
        settings = get_default_settings()
        settings["analysis"]["f1"] = 6e6

        repo.save(settings)
        loaded = repo.load()

        assert loaded["analysis"]["f1"] == 6e6

    def test_get_nested_key(self, repo: JSONSettingsRepository) -> None:
        """Test getting nested key with dot notation."""
        repo.save(get_default_settings())

        value = repo.get("analysis.f1")
        assert value == 5e6

    def test_get_with_default(self, repo: JSONSettingsRepository) -> None:
        """Test get with default value."""
        value = repo.get("nonexistent.key", default="default_value")
        assert value == "default_value"

    def test_set_nested_key(
        self, repo: JSONSettingsRepository, temp_settings: Path
    ) -> None:
        """Test setting nested key with dot notation."""
        repo.save(get_default_settings())
        repo.set("analysis.f1", 7e6)

        loaded = repo.load()
        assert loaded["analysis"]["f1"] == 7e6

    def test_reset_defaults(
        self, repo: JSONSettingsRepository, temp_settings: Path
    ) -> None:
        """Test reset to defaults."""
        # Modify settings
        settings = get_default_settings()
        settings["analysis"]["f1"] = 9e6
        repo.save(settings)

        # Reset
        repo.reset_defaults()
        loaded = repo.load()

        assert loaded["analysis"]["f1"] == 5e6

    def test_validate_valid_settings(self, repo: JSONSettingsRepository) -> None:
        """Test validation with valid settings."""
        errors = repo.validate(get_default_settings())
        assert errors == []

    def test_validate_invalid_f1(self, repo: JSONSettingsRepository) -> None:
        """Test validation catches invalid f1."""
        settings = get_default_settings()
        settings["analysis"]["f1"] = 0.5e6  # Below 1 MHz

        errors = repo.validate(settings)

        assert len(errors) > 0
        assert any(e.key == "analysis.f1" for e in errors)

    def test_validate_invalid_refh(self, repo: JSONSettingsRepository) -> None:
        """Test validation catches invalid refh."""
        settings = get_default_settings()
        settings["analysis"]["refh"] = 4  # Not odd

        errors = repo.validate(settings)

        assert len(errors) > 0
        assert any(e.key == "analysis.refh" for e in errors)

    def test_validate_invalid_calctype(self, repo: JSONSettingsRepository) -> None:
        """Test validation catches invalid calctype."""
        settings = get_default_settings()
        settings["analysis"]["calctype"] = "INVALID"

        errors = repo.validate(settings)

        assert len(errors) > 0
        assert any(e.key == "analysis.calctype" for e in errors)

    def test_corrupted_file_raises_error(self, temp_settings: Path) -> None:
        """Test that corrupted file raises CorruptedFileError."""
        # Write invalid JSON
        temp_settings.write_text("{invalid json")

        repo = JSONSettingsRepository(temp_settings)
        with pytest.raises(CorruptedFileError):
            repo.load()

    def test_cache_invalidation(
        self, repo: JSONSettingsRepository, temp_settings: Path
    ) -> None:
        """Test cache invalidation forces reload."""
        repo.save(get_default_settings())
        repo.load()  # Load into cache

        # Modify file externally
        settings = get_default_settings()
        settings["analysis"]["f1"] = 8e6
        with open(temp_settings, "w") as f:
            json.dump(settings, f)

        # Without invalidation, should get cached value
        loaded1 = repo.load()
        assert loaded1["analysis"]["f1"] == 5e6

        # Invalidate and reload
        repo.invalidate_cache()
        loaded2 = repo.load()
        assert loaded2["analysis"]["f1"] == 8e6

    def test_filepath_property(self, repo: JSONSettingsRepository) -> None:
        """Test filepath property returns correct path."""
        assert repo.filepath.name == "settings.json"

    def test_merge_with_defaults(
        self, repo: JSONSettingsRepository, temp_settings: Path
    ) -> None:
        """Test that missing keys are filled from defaults."""
        # Save partial settings
        partial = {"analysis": {"f1": 6e6}}
        with open(temp_settings, "w") as f:
            json.dump(partial, f)

        loaded = repo.load()

        # Should have merged with defaults
        assert loaded["analysis"]["f1"] == 6e6
        assert loaded["hardware"]["vna_port"] == "auto"


class TestMockSettingsRepository:
    """Test suite for MockSettingsRepository."""

    @pytest.fixture
    def mock_repo(self) -> MockSettingsRepository:
        """Create mock repository."""
        return MockSettingsRepository()

    def test_load_returns_defaults(self, mock_repo: MockSettingsRepository) -> None:
        """Test that load returns default settings."""
        settings = mock_repo.load()
        assert settings == get_default_settings()

    def test_load_increments_counter(self, mock_repo: MockSettingsRepository) -> None:
        """Test that load increments counter."""
        mock_repo.load()
        mock_repo.load()
        assert mock_repo.load_count == 2

    def test_save_increments_counter(self, mock_repo: MockSettingsRepository) -> None:
        """Test that save increments counter."""
        mock_repo.save(get_default_settings())
        mock_repo.save(get_default_settings())
        assert mock_repo.save_count == 2

    def test_save_and_load(self, mock_repo: MockSettingsRepository) -> None:
        """Test save/load roundtrip."""
        settings = get_default_settings()
        settings["analysis"]["f1"] = 7e6

        mock_repo.save(settings)
        loaded = mock_repo.load()

        assert loaded["analysis"]["f1"] == 7e6

    def test_get_nested_key(self, mock_repo: MockSettingsRepository) -> None:
        """Test getting nested key."""
        value = mock_repo.get("analysis.refh")
        assert value == 3

    def test_set_nested_key(self, mock_repo: MockSettingsRepository) -> None:
        """Test setting nested key."""
        mock_repo.set("analysis.f1", 8e6)

        assert mock_repo.get("analysis.f1") == 8e6
        # set() should also increment save count
        assert mock_repo.save_count == 1

    def test_reset_defaults(self, mock_repo: MockSettingsRepository) -> None:
        """Test reset to defaults."""
        mock_repo.set("analysis.f1", 9e6)
        mock_repo.reset_defaults()

        assert mock_repo.get("analysis.f1") == 5e6

    def test_validate_always_passes(self, mock_repo: MockSettingsRepository) -> None:
        """Test that mock validation always passes."""
        errors = mock_repo.validate({"invalid": "data"})
        assert errors == []

    def test_set_settings_helper(self, mock_repo: MockSettingsRepository) -> None:
        """Test set_settings test helper."""
        custom = {"custom": "settings"}
        mock_repo.set_settings(custom)

        assert mock_repo.load() == custom

    def test_reset_counters(self, mock_repo: MockSettingsRepository) -> None:
        """Test reset_counters test helper."""
        mock_repo.load()
        mock_repo.save(DEFAULT_SETTINGS)

        mock_repo.reset_counters()

        assert mock_repo.load_count == 0
        assert mock_repo.save_count == 0

    def test_filepath_property(self, mock_repo: MockSettingsRepository) -> None:
        """Test filepath property returns mock path."""
        assert mock_repo.filepath == Path("/mock/settings.json")


class TestValidationError:
    """Test ValidationError dataclass."""

    def test_validation_error_fields(self) -> None:
        """Test ValidationError has correct fields."""
        error = ValidationError(
            key="analysis.f1", message="Value out of range", value=0.5e6
        )

        assert error.key == "analysis.f1"
        assert error.message == "Value out of range"
        assert error.value == 0.5e6
