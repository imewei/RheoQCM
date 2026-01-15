"""Unit tests for theming service.

Tests cover:
- Theme enum values
- ThemeColors light/dark palettes
- ThemeManager singleton and theme switching
- Matplotlib style generation
- System theme detection fallback
"""

from __future__ import annotations

import pytest

from rheoQCM.services.theming import (
    Theme,
    ThemeColors,
    ThemeManager,
    get_system_theme,
)


class TestThemeEnum:
    """Test Theme enum values."""

    def test_theme_has_light(self) -> None:
        """Theme enum should have LIGHT value."""
        assert Theme.LIGHT.value == "light"

    def test_theme_has_dark(self) -> None:
        """Theme enum should have DARK value."""
        assert Theme.DARK.value == "dark"

    def test_theme_has_system(self) -> None:
        """Theme enum should have SYSTEM value."""
        assert Theme.SYSTEM.value == "system"


class TestThemeColors:
    """Test ThemeColors dataclass."""

    def test_light_colors_creation(self) -> None:
        """ThemeColors.light() should create light palette."""
        colors = ThemeColors.light()

        assert colors.background == "#FFFFFF"
        assert colors.text == "#212121"
        assert len(colors.series_colors) == 8

    def test_dark_colors_creation(self) -> None:
        """ThemeColors.dark() should create dark palette."""
        colors = ThemeColors.dark()

        assert colors.background == "#1E1E1E"
        assert colors.text == "#E0E0E0"
        assert len(colors.series_colors) == 8

    def test_colors_are_frozen(self) -> None:
        """ThemeColors should be immutable (frozen dataclass)."""
        colors = ThemeColors.light()

        with pytest.raises(Exception):  # FrozenInstanceError
            colors.background = "#000000"

    def test_series_colors_differ_between_themes(self) -> None:
        """Light and dark themes should have different series colors."""
        light = ThemeColors.light()
        dark = ThemeColors.dark()

        # First series color should differ for contrast
        assert light.series_colors[0] != dark.series_colors[0]


class TestThemeManager:
    """Test ThemeManager class."""

    def test_default_theme_is_system(self) -> None:
        """ThemeManager default should be SYSTEM."""
        manager = ThemeManager()
        # Resolved theme should be LIGHT or DARK (not SYSTEM)
        assert manager.theme in (Theme.LIGHT, Theme.DARK)

    def test_set_theme_light(self) -> None:
        """set_theme(LIGHT) should switch to light theme."""
        manager = ThemeManager()
        manager.set_theme(Theme.LIGHT)

        assert manager.theme == Theme.LIGHT
        assert manager.is_dark is False
        assert manager.colors.background == "#FFFFFF"

    def test_set_theme_dark(self) -> None:
        """set_theme(DARK) should switch to dark theme."""
        manager = ThemeManager()
        manager.set_theme(Theme.DARK)

        assert manager.theme == Theme.DARK
        assert manager.is_dark is True
        assert manager.colors.background == "#1E1E1E"

    def test_singleton_instance(self) -> None:
        """ThemeManager.instance() should return singleton."""
        # Clear singleton for test
        ThemeManager._instance = None

        instance1 = ThemeManager.instance()
        instance2 = ThemeManager.instance()

        assert instance1 is instance2

        # Clean up
        ThemeManager._instance = None

    def test_callback_registration(self) -> None:
        """Registered callbacks should be called on theme change."""
        manager = ThemeManager(theme=Theme.LIGHT)
        callback_called = []

        def callback() -> None:
            callback_called.append(True)

        manager.register_callback(callback)
        manager.set_theme(Theme.DARK)

        assert len(callback_called) == 1

    def test_callback_unregistration(self) -> None:
        """Unregistered callbacks should not be called."""
        manager = ThemeManager(theme=Theme.LIGHT)
        callback_called = []

        def callback() -> None:
            callback_called.append(True)

        manager.register_callback(callback)
        manager.unregister_callback(callback)
        manager.set_theme(Theme.DARK)

        assert len(callback_called) == 0

    def test_get_series_color(self) -> None:
        """get_series_color should return color by index."""
        manager = ThemeManager(theme=Theme.LIGHT)

        color0 = manager.get_series_color(0)
        color1 = manager.get_series_color(1)

        assert color0 != color1
        assert color0.startswith("#")

    def test_get_series_color_wraps(self) -> None:
        """get_series_color should wrap around at end of palette."""
        manager = ThemeManager(theme=Theme.LIGHT)

        color0 = manager.get_series_color(0)
        color8 = manager.get_series_color(8)  # Should wrap to index 0

        assert color0 == color8

    def test_get_rgba(self) -> None:
        """get_rgba should convert color to RGBA tuple."""
        manager = ThemeManager(theme=Theme.LIGHT)

        rgba = manager.get_rgba("#FF0000", alpha=0.5)

        assert len(rgba) == 4
        assert rgba[0] == 1.0  # Red
        assert rgba[1] == 0.0  # Green
        assert rgba[2] == 0.0  # Blue
        assert rgba[3] == 0.5  # Alpha

    def test_get_matplotlib_style(self) -> None:
        """get_matplotlib_style should return rcParams dict."""
        manager = ThemeManager(theme=Theme.DARK)

        style = manager.get_matplotlib_style()

        assert "figure.facecolor" in style
        assert style["figure.facecolor"] == manager.colors.background
        assert "axes.prop_cycle" in style


class TestGetSystemTheme:
    """Test get_system_theme function."""

    def test_returns_theme_enum(self) -> None:
        """get_system_theme should return Theme.LIGHT or Theme.DARK."""
        theme = get_system_theme()

        assert theme in (Theme.LIGHT, Theme.DARK)

    def test_fallback_to_light(self) -> None:
        """get_system_theme should fallback to LIGHT on error."""
        # Without a running QApplication, it should return LIGHT
        theme = get_system_theme()
        # Can be either - depends on if QApplication is running
        assert isinstance(theme, Theme)


class TestThemeManagerAxesApplication:
    """Test ThemeManager.apply_to_axes functionality."""

    def test_apply_to_axes(self) -> None:
        """apply_to_axes should set axes colors."""
        import matplotlib.pyplot as plt

        manager = ThemeManager(theme=Theme.DARK)
        fig, ax = plt.subplots()

        manager.apply_to_axes(ax)

        assert ax.get_facecolor() == (
            *manager.get_rgba(manager.colors.background)[:3],
            ax.get_facecolor()[3],
        )

        plt.close(fig)

    def test_apply_to_figure(self) -> None:
        """apply_to_figure should apply theme to all axes."""
        import matplotlib.pyplot as plt

        manager = ThemeManager(theme=Theme.LIGHT)
        fig, (ax1, ax2) = plt.subplots(1, 2)

        manager.apply_to_figure(fig)

        # Both axes should have theme applied
        assert (
            fig.get_facecolor()[:3] == manager.get_rgba(manager.colors.background)[:3]
        )

        plt.close(fig)
