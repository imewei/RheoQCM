"""Theme Manager for Light/Dark mode support.

Feature: Technical Guidelines Compliance - UX Polish

This module provides system-aware theming for PyQt6 and Matplotlib
with user override capability.

Public API
----------
Classes:
    Theme - Enum of available themes
    ThemeColors - Dataclass of theme-specific colors
    ThemeManager - Main theme management class

Functions:
    get_system_theme - Detect system Light/Dark preference
"""

from __future__ import annotations

__all__ = [
    "Theme",
    "ThemeColors",
    "ThemeManager",
    "get_system_theme",
]

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


class Theme(Enum):
    """Available theme options."""

    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"  # Follow system preference


@dataclass(frozen=True)
class ThemeColors:
    """Theme-specific color palette.

    Provides consistent colors for plots and UI elements
    across Light and Dark themes.
    """

    # Primary colors
    background: str
    foreground: str
    text: str
    text_secondary: str

    # Plot colors
    axis_color: str
    grid_color: str
    grid_alpha: float

    # Data series colors (tab10-based, adjusted for contrast)
    series_colors: tuple[str, ...]

    # Accent colors
    accent_primary: str
    accent_secondary: str
    accent_error: str
    accent_warning: str
    accent_success: str

    # UI element colors
    border_color: str
    hover_color: str
    selection_color: str

    @classmethod
    def light(cls) -> ThemeColors:
        """Create light theme color palette."""
        return cls(
            background="#FFFFFF",
            foreground="#F5F5F5",
            text="#212121",
            text_secondary="#757575",
            axis_color="#212121",
            grid_color="#E0E0E0",
            grid_alpha=0.7,
            series_colors=(
                "#1976D2",  # Blue
                "#D32F2F",  # Red
                "#388E3C",  # Green
                "#F57C00",  # Orange
                "#7B1FA2",  # Purple
                "#00796B",  # Teal
                "#5D4037",  # Brown
                "#616161",  # Gray
            ),
            accent_primary="#1976D2",
            accent_secondary="#7B1FA2",
            accent_error="#D32F2F",
            accent_warning="#F57C00",
            accent_success="#388E3C",
            border_color="#BDBDBD",
            hover_color="#E3F2FD",
            selection_color="#BBDEFB",
        )

    @classmethod
    def dark(cls) -> ThemeColors:
        """Create dark theme color palette."""
        return cls(
            background="#1E1E1E",
            foreground="#2D2D2D",
            text="#E0E0E0",
            text_secondary="#9E9E9E",
            axis_color="#E0E0E0",
            grid_color="#424242",
            grid_alpha=0.5,
            series_colors=(
                "#64B5F6",  # Light Blue
                "#EF5350",  # Light Red
                "#81C784",  # Light Green
                "#FFB74D",  # Light Orange
                "#BA68C8",  # Light Purple
                "#4DB6AC",  # Light Teal
                "#A1887F",  # Light Brown
                "#90A4AE",  # Blue Gray
            ),
            accent_primary="#64B5F6",
            accent_secondary="#BA68C8",
            accent_error="#EF5350",
            accent_warning="#FFB74D",
            accent_success="#81C784",
            border_color="#616161",
            hover_color="#37474F",
            selection_color="#455A64",
        )


def get_system_theme() -> Theme:
    """Detect system Light/Dark preference.

    Returns
    -------
    Theme
        Theme.LIGHT or Theme.DARK based on system preference.
        Defaults to LIGHT if detection fails.
    """
    try:
        from PyQt6.QtGui import QPalette
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            palette = app.palette()
            window_color = palette.color(QPalette.ColorRole.Window)
            # Dark theme if window background luminance is low
            luminance = (
                0.299 * window_color.red()
                + 0.587 * window_color.green()
                + 0.114 * window_color.blue()
            )
            return Theme.DARK if luminance < 128 else Theme.LIGHT
    except ImportError:
        logger.debug("PyQt6 not available for theme detection")
    except Exception as e:
        logger.debug("Theme detection failed: %s", e)

    return Theme.LIGHT


class ThemeManager:
    """Manages application theming for plots and UI.

    Provides theme-aware styling for Matplotlib figures and PyQt6 widgets.
    Supports system theme detection with user override.

    Example
    -------
    >>> manager = ThemeManager()
    >>> manager.set_theme(Theme.DARK)
    >>> colors = manager.colors
    >>> manager.apply_to_axes(ax)
    """

    _instance: ThemeManager | None = None

    def __init__(self, theme: Theme = Theme.SYSTEM) -> None:
        """Initialize ThemeManager.

        Parameters
        ----------
        theme : Theme
            Initial theme setting (default: SYSTEM for auto-detection).
        """
        self._user_theme = theme
        self._resolved_theme: Theme = Theme.LIGHT
        self._colors: ThemeColors = ThemeColors.light()
        self._callbacks: list[Any] = []
        self._resolve_theme()

    @classmethod
    def instance(cls) -> ThemeManager:
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def theme(self) -> Theme:
        """Current resolved theme (LIGHT or DARK)."""
        return self._resolved_theme

    @property
    def colors(self) -> ThemeColors:
        """Current theme color palette."""
        return self._colors

    @property
    def is_dark(self) -> bool:
        """True if current theme is dark."""
        return self._resolved_theme == Theme.DARK

    def set_theme(self, theme: Theme) -> None:
        """Set theme with optional system override.

        Parameters
        ----------
        theme : Theme
            Theme to apply. Use Theme.SYSTEM for auto-detection.
        """
        self._user_theme = theme
        self._resolve_theme()
        self._notify_callbacks()

    def _resolve_theme(self) -> None:
        """Resolve actual theme from user setting."""
        if self._user_theme == Theme.SYSTEM:
            self._resolved_theme = get_system_theme()
        else:
            self._resolved_theme = self._user_theme

        self._colors = (
            ThemeColors.dark()
            if self._resolved_theme == Theme.DARK
            else ThemeColors.light()
        )
        logger.debug("Theme resolved to: %s", self._resolved_theme.value)

    def register_callback(self, callback: Any) -> None:
        """Register callback for theme changes.

        Parameters
        ----------
        callback : Callable[[], None]
            Function called when theme changes.
        """
        self._callbacks.append(callback)

    def unregister_callback(self, callback: Any) -> None:
        """Unregister theme change callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_callbacks(self) -> None:
        """Notify all registered callbacks of theme change."""
        for callback in self._callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning("Theme callback failed: %s", e)

    def apply_to_axes(self, ax: Axes) -> None:
        """Apply theme colors to Matplotlib axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to style.
        """
        colors = self._colors

        # Background
        ax.set_facecolor(colors.background)
        if ax.figure is not None:
            ax.figure.set_facecolor(colors.background)

        # Axis colors
        ax.spines["bottom"].set_color(colors.axis_color)
        ax.spines["top"].set_color(colors.axis_color)
        ax.spines["left"].set_color(colors.axis_color)
        ax.spines["right"].set_color(colors.axis_color)

        # Tick colors
        ax.tick_params(axis="x", colors=colors.axis_color)
        ax.tick_params(axis="y", colors=colors.axis_color)

        # Label colors
        ax.xaxis.label.set_color(colors.text)
        ax.yaxis.label.set_color(colors.text)
        ax.title.set_color(colors.text)

        # Grid
        ax.grid(True, color=colors.grid_color, alpha=colors.grid_alpha)

    def apply_to_figure(self, fig: Any) -> None:
        """Apply theme to entire figure and all axes.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to style.
        """
        fig.set_facecolor(self._colors.background)
        for ax in fig.axes:
            self.apply_to_axes(ax)

    def get_matplotlib_style(self) -> dict[str, Any]:
        """Get matplotlib rcParams for current theme.

        Returns
        -------
        dict
            Style dictionary suitable for plt.style.context() or rcParams update.
        """
        colors = self._colors
        return {
            "figure.facecolor": colors.background,
            "axes.facecolor": colors.background,
            "axes.edgecolor": colors.axis_color,
            "axes.labelcolor": colors.text,
            "text.color": colors.text,
            "xtick.color": colors.axis_color,
            "ytick.color": colors.axis_color,
            "grid.color": colors.grid_color,
            "grid.alpha": colors.grid_alpha,
            "axes.prop_cycle": plt.cycler(color=colors.series_colors),
        }

    def get_series_color(self, index: int) -> str:
        """Get series color by index (wraps around).

        Parameters
        ----------
        index : int
            Series index.

        Returns
        -------
        str
            Hex color string.
        """
        series = self._colors.series_colors
        return series[index % len(series)]

    def get_rgba(self, color: str, alpha: float = 1.0) -> tuple[float, ...]:
        """Convert color string to RGBA tuple.

        Parameters
        ----------
        color : str
            Color string (hex, name, or theme key like "accent_primary").
        alpha : float
            Alpha value (0-1).

        Returns
        -------
        tuple[float, float, float, float]
            RGBA tuple.
        """
        # Check if color is a theme attribute
        if hasattr(self._colors, color):
            color = getattr(self._colors, color)
        return mcolors.to_rgba(color, alpha=alpha)

    def apply_stylesheet(self, widget: Any) -> None:
        """Apply theme stylesheet to PyQt6 widget.

        Parameters
        ----------
        widget : QWidget
            Widget to style.
        """
        colors = self._colors
        stylesheet = f"""
            QWidget {{
                background-color: {colors.background};
                color: {colors.text};
            }}
            QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
                background-color: {colors.foreground};
                border: 1px solid {colors.border_color};
                color: {colors.text};
            }}
            QPushButton {{
                background-color: {colors.foreground};
                border: 1px solid {colors.border_color};
                color: {colors.text};
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: {colors.hover_color};
            }}
            QPushButton:pressed {{
                background-color: {colors.selection_color};
            }}
            QTableWidget, QTableView, QTreeView, QListView {{
                background-color: {colors.foreground};
                color: {colors.text};
                gridline-color: {colors.border_color};
            }}
            QHeaderView::section {{
                background-color: {colors.foreground};
                color: {colors.text};
                border: 1px solid {colors.border_color};
            }}
            QMenuBar, QMenu {{
                background-color: {colors.background};
                color: {colors.text};
            }}
            QMenu::item:selected {{
                background-color: {colors.selection_color};
            }}
            QToolBar {{
                background-color: {colors.background};
                border: none;
            }}
            QStatusBar {{
                background-color: {colors.foreground};
                color: {colors.text_secondary};
            }}
            QScrollBar {{
                background-color: {colors.foreground};
            }}
            QTabWidget::pane {{
                border: 1px solid {colors.border_color};
            }}
            QTabBar::tab {{
                background-color: {colors.foreground};
                color: {colors.text};
                padding: 5px 10px;
            }}
            QTabBar::tab:selected {{
                background-color: {colors.selection_color};
            }}
        """
        widget.setStyleSheet(stylesheet)
