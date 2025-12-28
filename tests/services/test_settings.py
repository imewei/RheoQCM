"""
Tests for SettingsRepository interface and implementations.

T049: Create test_settings_repository.py with tests for:
- Load/save operations
- Dot notation access
- Validation
- Default merging
- Error handling
"""

import json
from pathlib import Path

import pytest

from rheoQCM.services.settings import (
    DEFAULT_SETTINGS,
    CorruptedFileError,
    JSONSettingsRepository,
    MockSettingsRepository,
    ValidationError,
)


class TestMockSettingsRepository:
    """Tests for MockSettingsRepository."""

    def test_load_returns_defaults(self):
        """Test load returns defaults when no initial settings."""
        repo = MockSettingsRepository()
        settings = repo.load()

        assert settings["analysis"]["f1"] == 5e6
        assert settings["analysis"]["refh"] == 3
        assert settings["display"]["theme"] == "light"

    def test_load_returns_initial_settings(self):
        """Test load returns initial settings when provided."""
        initial = {"analysis": {"f1": 6e6}}
        repo = MockSettingsRepository(initial=initial)

        settings = repo.load()
        assert settings["analysis"]["f1"] == 6e6

    def test_load_increments_counter(self):
        """Test load increments load_count."""
        repo = MockSettingsRepository()

        repo.load()
        repo.load()

        assert repo.load_count == 2

    def test_save_updates_settings(self):
        """Test save updates internal settings."""
        repo = MockSettingsRepository()

        new_settings = {"analysis": {"f1": 7e6}}
        repo.save(new_settings)

        assert repo.load()["analysis"]["f1"] == 7e6

    def test_save_increments_counter(self):
        """Test save increments save_count."""
        repo = MockSettingsRepository()

        repo.save({"test": "value"})
        repo.save({"test": "value2"})

        assert repo.save_count == 2

    def test_get_dot_notation(self):
        """Test get with dot notation."""
        repo = MockSettingsRepository()

        assert repo.get("analysis.f1") == 5e6
        assert repo.get("analysis.refh") == 3
        assert repo.get("display.theme") == "light"

    def test_get_returns_default_for_missing_key(self):
        """Test get returns default for missing key."""
        repo = MockSettingsRepository()

        assert repo.get("nonexistent.key", "default") == "default"
        assert repo.get("analysis.nonexistent", 42) == 42

    def test_set_dot_notation(self):
        """Test set with dot notation."""
        repo = MockSettingsRepository()

        repo.set("analysis.f1", 8e6)

        assert repo.get("analysis.f1") == 8e6

    def test_set_creates_nested_structure(self):
        """Test set creates nested structure for new keys."""
        repo = MockSettingsRepository()

        repo.set("custom.nested.value", 123)

        assert repo.get("custom.nested.value") == 123

    def test_reset_defaults(self):
        """Test reset_defaults restores defaults."""
        repo = MockSettingsRepository()
        repo.set("analysis.f1", 9e6)

        repo.reset_defaults()

        assert repo.get("analysis.f1") == 5e6

    def test_validate_always_returns_empty(self):
        """Test mock validate always returns empty list."""
        repo = MockSettingsRepository()

        errors = repo.validate({"invalid": "data"})

        assert errors == []

    def test_filepath_property(self):
        """Test filepath property."""
        repo = MockSettingsRepository()

        assert repo.filepath == Path("/mock/settings.json")

    def test_set_settings_helper(self):
        """Test set_settings helper method."""
        repo = MockSettingsRepository()

        repo.set_settings({"custom": {"key": "value"}})

        assert repo.get("custom.key") == "value"

    def test_reset_counters_helper(self):
        """Test reset_counters helper method."""
        repo = MockSettingsRepository()
        repo.load()
        repo.save({})

        repo.reset_counters()

        assert repo.load_count == 0
        assert repo.save_count == 0


class TestJSONSettingsRepository:
    """Tests for JSONSettingsRepository."""

    def test_load_returns_defaults_when_file_missing(self, tmp_path):
        """Test load returns defaults when file doesn't exist."""
        filepath = tmp_path / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        settings = repo.load()

        assert settings == DEFAULT_SETTINGS

    def test_save_creates_file(self, tmp_path):
        """Test save creates file."""
        filepath = tmp_path / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        repo.save(DEFAULT_SETTINGS)

        assert filepath.exists()
        with open(filepath) as f:
            saved = json.load(f)
        assert saved["analysis"]["f1"] == 5e6

    def test_load_reads_saved_settings(self, tmp_path):
        """Test load reads saved settings."""
        filepath = tmp_path / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        repo.save({"analysis": {"f1": 6e6, "refh": 5}})

        # Clear cache and reload
        repo.invalidate_cache()
        settings = repo.load()

        assert settings["analysis"]["f1"] == 6e6
        assert settings["analysis"]["refh"] == 5

    def test_load_merges_with_defaults(self, tmp_path):
        """Test load merges partial settings with defaults."""
        filepath = tmp_path / "settings.json"

        # Write partial settings directly
        with open(filepath, "w") as f:
            json.dump({"analysis": {"f1": 7e6}}, f)

        repo = JSONSettingsRepository(filepath=filepath)
        settings = repo.load()

        # Custom value preserved
        assert settings["analysis"]["f1"] == 7e6
        # Default values filled in
        assert settings["analysis"]["refh"] == 3
        assert settings["display"]["theme"] == "light"

    def test_load_raises_on_corrupted_file(self, tmp_path):
        """Test load raises CorruptedFileError on invalid JSON."""
        filepath = tmp_path / "settings.json"
        filepath.write_text("not valid json {{{")

        repo = JSONSettingsRepository(filepath=filepath)

        with pytest.raises(CorruptedFileError, match="Settings file corrupted"):
            repo.load()

    def test_get_dot_notation(self, tmp_path):
        """Test get with dot notation."""
        filepath = tmp_path / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        assert repo.get("analysis.f1") == 5e6
        assert repo.get("display.autoscale") is True

    def test_get_returns_default_for_missing(self, tmp_path):
        """Test get returns default for missing key."""
        filepath = tmp_path / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        assert repo.get("nonexistent", "fallback") == "fallback"

    def test_set_updates_and_saves(self, tmp_path):
        """Test set updates value and saves."""
        filepath = tmp_path / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        repo.set("analysis.f1", 8e6)

        # Verify in memory
        assert repo.get("analysis.f1") == 8e6

        # Verify persisted
        repo.invalidate_cache()
        assert repo.get("analysis.f1") == 8e6

    def test_reset_defaults_saves(self, tmp_path):
        """Test reset_defaults saves default settings."""
        filepath = tmp_path / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        repo.set("analysis.f1", 9e6)
        repo.reset_defaults()

        assert repo.get("analysis.f1") == 5e6

    def test_filepath_property(self, tmp_path):
        """Test filepath property."""
        filepath = tmp_path / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        assert repo.filepath == filepath

    def test_caching(self, tmp_path):
        """Test that settings are cached."""
        filepath = tmp_path / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        # First load
        settings1 = repo.load()
        # Modify file directly
        with open(filepath, "w") as f:
            json.dump({"analysis": {"f1": 9e6}}, f)

        # Second load should return cached value
        settings2 = repo.load()
        assert settings2["analysis"]["f1"] == settings1["analysis"]["f1"]

        # After invalidation, should read new value
        repo.invalidate_cache()
        settings3 = repo.load()
        assert settings3["analysis"]["f1"] == 9e6

    def test_creates_parent_directories(self, tmp_path):
        """Test save creates parent directories."""
        filepath = tmp_path / "nested" / "dir" / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        repo.save(DEFAULT_SETTINGS)

        assert filepath.exists()


class TestValidation:
    """Tests for settings validation."""

    def test_validate_f1_too_low(self, tmp_path):
        """Test validation fails for f1 below range."""
        filepath = tmp_path / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        settings = {"analysis": {"f1": 500e3}}  # 500 kHz
        errors = repo.validate(settings)

        assert len(errors) == 1
        assert errors[0].key == "analysis.f1"
        assert "1 MHz and 10 MHz" in errors[0].message

    def test_validate_f1_too_high(self, tmp_path):
        """Test validation fails for f1 above range."""
        filepath = tmp_path / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        settings = {"analysis": {"f1": 20e6}}  # 20 MHz
        errors = repo.validate(settings)

        assert len(errors) == 1
        assert errors[0].key == "analysis.f1"

    def test_validate_refh_invalid(self, tmp_path):
        """Test validation fails for invalid refh."""
        filepath = tmp_path / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        settings = {"analysis": {"refh": 4}}  # Even number
        errors = repo.validate(settings)

        assert len(errors) == 1
        assert errors[0].key == "analysis.refh"
        assert "odd harmonic" in errors[0].message

    def test_validate_calctype_invalid(self, tmp_path):
        """Test validation fails for invalid calctype."""
        filepath = tmp_path / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        settings = {"analysis": {"calctype": "invalid"}}
        errors = repo.validate(settings)

        assert len(errors) == 1
        assert errors[0].key == "analysis.calctype"

    def test_validate_n_points_out_of_range(self, tmp_path):
        """Test validation fails for n_points out of range."""
        filepath = tmp_path / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        settings = {"acquisition": {"n_points": 0}}
        errors = repo.validate(settings)

        assert len(errors) == 1
        assert errors[0].key == "acquisition.n_points"

    def test_validate_if_bandwidth_out_of_range(self, tmp_path):
        """Test validation fails for if_bandwidth out of range."""
        filepath = tmp_path / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        settings = {"acquisition": {"if_bandwidth": 0.5}}
        errors = repo.validate(settings)

        assert len(errors) == 1
        assert errors[0].key == "acquisition.if_bandwidth"

    def test_validate_log_level_invalid(self, tmp_path):
        """Test validation fails for invalid log level."""
        filepath = tmp_path / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        settings = {"logging": {"level": "TRACE"}}
        errors = repo.validate(settings)

        assert len(errors) == 1
        assert errors[0].key == "logging.level"

    def test_validate_theme_invalid(self, tmp_path):
        """Test validation fails for invalid theme."""
        filepath = tmp_path / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        settings = {"display": {"theme": "neon"}}
        errors = repo.validate(settings)

        assert len(errors) == 1
        assert errors[0].key == "display.theme"

    def test_validate_multiple_errors(self, tmp_path):
        """Test validation returns multiple errors."""
        filepath = tmp_path / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        settings = {
            "analysis": {"f1": 500e3, "refh": 4, "calctype": "invalid"},
            "logging": {"level": "TRACE"},
        }
        errors = repo.validate(settings)

        assert len(errors) == 4

    def test_validate_valid_settings(self, tmp_path):
        """Test validation passes for valid settings."""
        filepath = tmp_path / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        errors = repo.validate(DEFAULT_SETTINGS)

        assert errors == []

    def test_save_raises_on_validation_error(self, tmp_path):
        """Test save raises on validation error."""
        filepath = tmp_path / "settings.json"
        repo = JSONSettingsRepository(filepath=filepath)

        with pytest.raises(ValueError, match="Validation failed"):
            repo.save({"analysis": {"f1": 500e3}})


class TestValidationError:
    """Tests for ValidationError dataclass."""

    def test_creation(self):
        """Test ValidationError creation."""
        error = ValidationError(key="analysis.f1", message="Invalid value", value=500e3)

        assert error.key == "analysis.f1"
        assert error.message == "Invalid value"
        assert error.value == 500e3


class TestDefaultSettings:
    """Tests for DEFAULT_SETTINGS constant."""

    def test_structure(self):
        """Test DEFAULT_SETTINGS has expected structure."""
        assert "hardware" in DEFAULT_SETTINGS
        assert "acquisition" in DEFAULT_SETTINGS
        assert "analysis" in DEFAULT_SETTINGS
        assert "display" in DEFAULT_SETTINGS
        assert "logging" in DEFAULT_SETTINGS

    def test_analysis_defaults(self):
        """Test analysis default values."""
        analysis = DEFAULT_SETTINGS["analysis"]

        assert analysis["f1"] == 5e6
        assert analysis["refh"] == 3
        assert analysis["calctype"] == "SLA"
        assert analysis["auto_analyze"] is True

    def test_acquisition_defaults(self):
        """Test acquisition default values."""
        acq = DEFAULT_SETTINGS["acquisition"]

        assert acq["f_start"] == 4.95e6
        assert acq["f_stop"] == 5.05e6
        assert acq["n_points"] == 401
        assert acq["if_bandwidth"] == 1000.0
        assert 3 in acq["harmonics"]

    def test_display_defaults(self):
        """Test display default values."""
        display = DEFAULT_SETTINGS["display"]

        assert display["autoscale"] is True
        assert display["theme"] == "light"
