"""Enhanced GUI styles for professional scientific applications.

This module provides consistent styling following UI/UX best practices:
- Data-Dense + Dark Mode (OLED) for scientific dashboards
- Swiss Modernism 2.0 grid system
- WCAG AAA accessibility compliance
- Professional button, panel, and layout styling

Public API
----------
Classes:
    StyleConfig - Configuration dataclass for style parameters
    StyleManager - Manages application-wide styles

Functions:
    get_button_stylesheet - Generate button CSS with proper states
    get_panel_stylesheet - Generate panel/groupbox CSS
    get_input_stylesheet - Generate input field CSS
"""

from __future__ import annotations

__all__ = [
    "StyleConfig",
    "StyleManager",
    "get_button_stylesheet",
    "get_panel_stylesheet",
    "get_input_stylesheet",
    "ButtonVariant",
    "SPACING",
    "COLORS_LIGHT",
    "COLORS_DARK",
]

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget


class ButtonVariant(Enum):
    """Button style variants."""

    PRIMARY = "primary"  # Main action (blue)
    SECONDARY = "secondary"  # Secondary action (gray)
    SUCCESS = "success"  # Positive action (green)
    DANGER = "danger"  # Destructive action (red)
    WARNING = "warning"  # Caution action (orange)
    GHOST = "ghost"  # Minimal style


# Spacing constants (8px base unit - Swiss Modernism)
class SPACING:
    """Spacing constants following 8px grid system."""

    XS = 4  # 0.5x base
    SM = 8  # 1x base
    MD = 16  # 2x base
    LG = 24  # 3x base
    XL = 32  # 4x base
    XXL = 48  # 6x base


# Color palettes based on UI/UX Pro Max recommendations
@dataclass(frozen=True)
class ColorPalette:
    """Color palette for theming."""

    # Primary colors
    primary: str
    primary_hover: str
    primary_pressed: str

    # Secondary colors
    secondary: str
    secondary_hover: str
    secondary_pressed: str

    # Status colors
    success: str
    success_hover: str
    danger: str
    danger_hover: str
    warning: str
    warning_hover: str

    # Background colors
    bg_primary: str
    bg_secondary: str
    bg_tertiary: str
    bg_elevated: str

    # Text colors
    text_primary: str
    text_secondary: str
    text_muted: str
    text_inverse: str

    # Border colors
    border_subtle: str
    border_default: str
    border_strong: str

    # Focus/selection
    focus_ring: str
    selection: str


# Light theme palette (Analytics Dashboard style)
COLORS_LIGHT = ColorPalette(
    # Primary - Trust Blue
    primary="#3B82F6",
    primary_hover="#2563EB",
    primary_pressed="#1D4ED8",
    # Secondary - Neutral Gray
    secondary="#6B7280",
    secondary_hover="#4B5563",
    secondary_pressed="#374151",
    # Status colors
    success="#16A34A",
    success_hover="#15803D",
    danger="#DC2626",
    danger_hover="#B91C1C",
    warning="#F97316",
    warning_hover="#EA580C",
    # Backgrounds
    bg_primary="#FFFFFF",
    bg_secondary="#F8FAFC",
    bg_tertiary="#F1F5F9",
    bg_elevated="#FFFFFF",
    # Text (WCAG AAA compliant)
    text_primary="#0F172A",  # slate-900 (7:1+ contrast)
    text_secondary="#475569",  # slate-600 (4.5:1+ contrast)
    text_muted="#64748B",  # slate-500
    text_inverse="#FFFFFF",
    # Borders
    border_subtle="#E2E8F0",
    border_default="#CBD5E1",
    border_strong="#94A3B8",
    # Focus
    focus_ring="#3B82F6",
    selection="#DBEAFE",
)


# Dark theme palette (OLED optimized)
COLORS_DARK = ColorPalette(
    # Primary - Bright Blue for contrast
    primary="#60A5FA",
    primary_hover="#93C5FD",
    primary_pressed="#3B82F6",
    # Secondary
    secondary="#9CA3AF",
    secondary_hover="#D1D5DB",
    secondary_pressed="#6B7280",
    # Status colors (brighter for dark bg)
    success="#4ADE80",
    success_hover="#86EFAC",
    danger="#F87171",
    danger_hover="#FCA5A5",
    warning="#FB923C",
    warning_hover="#FDBA74",
    # Backgrounds (OLED-friendly deep blacks)
    bg_primary="#0A0A0A",
    bg_secondary="#121212",
    bg_tertiary="#1E1E1E",
    bg_elevated="#262626",
    # Text
    text_primary="#F1F5F9",
    text_secondary="#94A3B8",
    text_muted="#64748B",
    text_inverse="#0F172A",
    # Borders
    border_subtle="#27272A",
    border_default="#3F3F46",
    border_strong="#52525B",
    # Focus
    focus_ring="#60A5FA",
    selection="#1E3A5F",
)


@dataclass
class StyleConfig:
    """Style configuration parameters."""

    border_radius: int = 6
    button_height: int = 36
    button_height_sm: int = 28
    button_height_lg: int = 44
    input_height: int = 36
    icon_size: int = 20
    icon_size_sm: int = 16
    font_family: str = "Segoe UI, system-ui, sans-serif"
    font_size: int = 13
    font_size_sm: int = 11
    font_size_lg: int = 15
    transition_duration: str = "150ms"


def get_button_stylesheet(
    palette: ColorPalette,
    variant: ButtonVariant = ButtonVariant.PRIMARY,
    config: StyleConfig | None = None,
) -> str:
    """Generate button stylesheet with proper hover/pressed states.

    Parameters
    ----------
    palette : ColorPalette
        Color palette to use
    variant : ButtonVariant
        Button style variant
    config : StyleConfig, optional
        Style configuration

    Returns
    -------
    str
        CSS stylesheet string
    """
    if config is None:
        config = StyleConfig()

    # Get colors based on variant
    if variant == ButtonVariant.PRIMARY:
        bg = palette.primary
        bg_hover = palette.primary_hover
        bg_pressed = palette.primary_pressed
        text = palette.text_inverse
    elif variant == ButtonVariant.SUCCESS:
        bg = palette.success
        bg_hover = palette.success_hover
        bg_pressed = palette.success
        text = palette.text_inverse
    elif variant == ButtonVariant.DANGER:
        bg = palette.danger
        bg_hover = palette.danger_hover
        bg_pressed = palette.danger
        text = palette.text_inverse
    elif variant == ButtonVariant.WARNING:
        bg = palette.warning
        bg_hover = palette.warning_hover
        bg_pressed = palette.warning
        text = palette.text_inverse
    elif variant == ButtonVariant.GHOST:
        bg = "transparent"
        bg_hover = palette.bg_tertiary
        bg_pressed = palette.border_subtle
        text = palette.text_primary
    else:  # SECONDARY
        bg = palette.bg_tertiary
        bg_hover = palette.border_subtle
        bg_pressed = palette.border_default
        text = palette.text_primary

    return f"""
        QPushButton {{
            background-color: {bg};
            color: {text};
            border: none;
            border-radius: {config.border_radius}px;
            padding: {SPACING.SM}px {SPACING.MD}px;
            min-height: {config.button_height}px;
            font-family: {config.font_family};
            font-size: {config.font_size}px;
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: {bg_hover};
        }}
        QPushButton:pressed {{
            background-color: {bg_pressed};
        }}
        QPushButton:disabled {{
            background-color: {palette.bg_tertiary};
            color: {palette.text_muted};
        }}
        QPushButton:focus {{
            outline: 2px solid {palette.focus_ring};
            outline-offset: 2px;
        }}
    """


def get_panel_stylesheet(
    palette: ColorPalette,
    config: StyleConfig | None = None,
    elevated: bool = False,
) -> str:
    """Generate panel/groupbox stylesheet.

    Parameters
    ----------
    palette : ColorPalette
        Color palette to use
    config : StyleConfig, optional
        Style configuration
    elevated : bool
        Whether panel appears elevated (shadow effect)

    Returns
    -------
    str
        CSS stylesheet string
    """
    if config is None:
        config = StyleConfig()

    bg = palette.bg_elevated if elevated else palette.bg_secondary
    # Note: shadow variable reserved for future box-shadow CSS support
    # when PyQt6 supports it natively (currently not supported in stylesheets)
    _ = "0 1px 3px rgba(0,0,0,0.1)" if elevated else "none"

    return f"""
        QGroupBox {{
            background-color: {bg};
            border: 1px solid {palette.border_subtle};
            border-radius: {config.border_radius}px;
            margin-top: {SPACING.MD}px;
            padding: {SPACING.MD}px;
            font-family: {config.font_family};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: {SPACING.SM}px;
            padding: 0 {SPACING.XS}px;
            color: {palette.text_primary};
            font-weight: 600;
            font-size: {config.font_size}px;
            background-color: {bg};
        }}
    """


def get_input_stylesheet(
    palette: ColorPalette,
    config: StyleConfig | None = None,
) -> str:
    """Generate input field stylesheet.

    Parameters
    ----------
    palette : ColorPalette
        Color palette to use
    config : StyleConfig, optional
        Style configuration

    Returns
    -------
    str
        CSS stylesheet string
    """
    if config is None:
        config = StyleConfig()

    return f"""
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
            background-color: {palette.bg_primary};
            color: {palette.text_primary};
            border: 1px solid {palette.border_default};
            border-radius: {config.border_radius}px;
            padding: {SPACING.XS}px {SPACING.SM}px;
            min-height: {config.input_height}px;
            font-family: {config.font_family};
            font-size: {config.font_size}px;
        }}
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
            border-color: {palette.focus_ring};
            outline: none;
        }}
        QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
            background-color: {palette.bg_tertiary};
            color: {palette.text_muted};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}
        QComboBox::down-arrow {{
            width: 12px;
            height: 12px;
        }}
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            border: none;
            background-color: transparent;
            width: 16px;
        }}
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
            background-color: {palette.bg_tertiary};
        }}
    """


class StyleManager:
    """Manages application-wide styles.

    Singleton pattern for consistent styling across the application.
    Supports light/dark theme switching.

    Example
    -------
    >>> manager = StyleManager.instance()
    >>> manager.set_dark_mode(True)
    >>> stylesheet = manager.get_full_stylesheet()
    >>> widget.setStyleSheet(stylesheet)
    """

    _instance: StyleManager | None = None

    def __init__(self) -> None:
        self._dark_mode = False
        self._config = StyleConfig()
        self._callbacks: list = []

    @classmethod
    def instance(cls) -> StyleManager:
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def palette(self) -> ColorPalette:
        """Get current color palette."""
        return COLORS_DARK if self._dark_mode else COLORS_LIGHT

    @property
    def config(self) -> StyleConfig:
        """Get style configuration."""
        return self._config

    @property
    def is_dark(self) -> bool:
        """Check if dark mode is active."""
        return self._dark_mode

    def set_dark_mode(self, enabled: bool) -> None:
        """Set dark mode state.

        Parameters
        ----------
        enabled : bool
            True for dark mode, False for light mode
        """
        if self._dark_mode != enabled:
            self._dark_mode = enabled
            self._notify_callbacks()

    def register_callback(self, callback) -> None:
        """Register callback for theme changes."""
        self._callbacks.append(callback)

    def unregister_callback(self, callback) -> None:
        """Unregister theme change callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_callbacks(self) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback()
            except Exception:
                pass

    def get_full_stylesheet(self) -> str:
        """Get complete application stylesheet.

        Returns
        -------
        str
            Full CSS stylesheet for the application
        """
        p = self.palette
        c = self._config

        return f"""
            /* === Base Styles === */
            QWidget {{
                background-color: {p.bg_primary};
                color: {p.text_primary};
                font-family: {c.font_family};
                font-size: {c.font_size}px;
            }}

            /* === Main Window === */
            QMainWindow {{
                background-color: {p.bg_secondary};
            }}

            /* === Menu Bar === */
            QMenuBar {{
                background-color: {p.bg_primary};
                color: {p.text_primary};
                border-bottom: 1px solid {p.border_subtle};
                padding: {SPACING.XS}px;
            }}
            QMenuBar::item {{
                padding: {SPACING.XS}px {SPACING.SM}px;
                border-radius: {c.border_radius}px;
            }}
            QMenuBar::item:selected {{
                background-color: {p.bg_tertiary};
            }}
            QMenu {{
                background-color: {p.bg_elevated};
                border: 1px solid {p.border_subtle};
                border-radius: {c.border_radius}px;
                padding: {SPACING.XS}px;
            }}
            QMenu::item {{
                padding: {SPACING.SM}px {SPACING.MD}px;
                border-radius: {c.border_radius - 2}px;
            }}
            QMenu::item:selected {{
                background-color: {p.selection};
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {p.border_subtle};
                margin: {SPACING.XS}px {SPACING.SM}px;
            }}

            /* === Tool Bar === */
            QToolBar {{
                background-color: {p.bg_primary};
                border-bottom: 1px solid {p.border_subtle};
                spacing: {SPACING.XS}px;
                padding: {SPACING.XS}px;
            }}
            QToolButton {{
                background-color: transparent;
                border: none;
                border-radius: {c.border_radius}px;
                padding: {SPACING.SM}px;
            }}
            QToolButton:hover {{
                background-color: {p.bg_tertiary};
            }}
            QToolButton:pressed {{
                background-color: {p.border_subtle};
            }}

            /* === Status Bar === */
            QStatusBar {{
                background-color: {p.bg_secondary};
                border-top: 1px solid {p.border_subtle};
                color: {p.text_secondary};
                font-size: {c.font_size_sm}px;
            }}

            /* === Splitter === */
            QSplitter::handle {{
                background-color: {p.border_subtle};
            }}
            QSplitter::handle:horizontal {{
                width: 2px;
            }}
            QSplitter::handle:vertical {{
                height: 2px;
            }}
            QSplitter::handle:hover {{
                background-color: {p.primary};
            }}

            /* === Tab Widget === */
            QTabWidget::pane {{
                border: 1px solid {p.border_subtle};
                border-radius: {c.border_radius}px;
                background-color: {p.bg_primary};
            }}
            QTabBar::tab {{
                background-color: {p.bg_tertiary};
                color: {p.text_secondary};
                border: none;
                padding: {SPACING.SM}px {SPACING.MD}px;
                margin-right: 2px;
                border-top-left-radius: {c.border_radius}px;
                border-top-right-radius: {c.border_radius}px;
            }}
            QTabBar::tab:selected {{
                background-color: {p.bg_primary};
                color: {p.text_primary};
                font-weight: 500;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {p.border_subtle};
            }}

            /* === Group Box === */
            QGroupBox {{
                background-color: {p.bg_secondary};
                border: 1px solid {p.border_subtle};
                border-radius: {c.border_radius}px;
                margin-top: 16px;
                padding: {SPACING.MD}px;
                padding-top: {SPACING.LG}px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: {SPACING.SM}px;
                padding: 0 {SPACING.XS}px;
                color: {p.text_primary};
                font-weight: 600;
                background-color: {p.bg_secondary};
            }}

            /* === Buttons === */
            QPushButton {{
                background-color: {p.primary};
                color: {p.text_inverse};
                border: none;
                border-radius: {c.border_radius}px;
                padding: {SPACING.SM}px {SPACING.MD}px;
                min-height: {c.button_height}px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {p.primary_hover};
            }}
            QPushButton:pressed {{
                background-color: {p.primary_pressed};
            }}
            QPushButton:disabled {{
                background-color: {p.bg_tertiary};
                color: {p.text_muted};
            }}

            /* Secondary buttons */
            QPushButton[flat="true"] {{
                background-color: {p.bg_tertiary};
                color: {p.text_primary};
            }}
            QPushButton[flat="true"]:hover {{
                background-color: {p.border_subtle};
            }}

            /* === Input Fields === */
            QLineEdit, QTextEdit, QPlainTextEdit {{
                background-color: {p.bg_primary};
                color: {p.text_primary};
                border: 1px solid {p.border_default};
                border-radius: {c.border_radius}px;
                padding: {SPACING.SM}px;
                selection-background-color: {p.selection};
            }}
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
                border-color: {p.focus_ring};
            }}
            QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {{
                background-color: {p.bg_tertiary};
                color: {p.text_muted};
            }}

            /* === Spin Box === */
            QSpinBox, QDoubleSpinBox {{
                background-color: {p.bg_primary};
                color: {p.text_primary};
                border: 1px solid {p.border_default};
                border-radius: {c.border_radius}px;
                padding: {SPACING.XS}px {SPACING.SM}px;
            }}
            QSpinBox:focus, QDoubleSpinBox:focus {{
                border-color: {p.focus_ring};
            }}
            QSpinBox::up-button, QDoubleSpinBox::up-button,
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                border: none;
                background-color: transparent;
                width: 20px;
            }}
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
                background-color: {p.bg_tertiary};
            }}

            /* === Combo Box === */
            QComboBox {{
                background-color: {p.bg_primary};
                color: {p.text_primary};
                border: 1px solid {p.border_default};
                border-radius: {c.border_radius}px;
                padding: {SPACING.SM}px;
                min-height: {c.button_height}px;
            }}
            QComboBox:focus {{
                border-color: {p.focus_ring};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {p.bg_elevated};
                border: 1px solid {p.border_subtle};
                border-radius: {c.border_radius}px;
                selection-background-color: {p.selection};
            }}

            /* === Check Box === */
            QCheckBox {{
                spacing: {SPACING.SM}px;
                color: {p.text_primary};
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {p.border_default};
                border-radius: 4px;
                background-color: {p.bg_primary};
            }}
            QCheckBox::indicator:hover {{
                border-color: {p.primary};
            }}
            QCheckBox::indicator:checked {{
                background-color: {p.primary};
                border-color: {p.primary};
            }}

            /* === Radio Button === */
            QRadioButton {{
                spacing: {SPACING.SM}px;
                color: {p.text_primary};
            }}
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {p.border_default};
                border-radius: 9px;
                background-color: {p.bg_primary};
            }}
            QRadioButton::indicator:hover {{
                border-color: {p.primary};
            }}
            QRadioButton::indicator:checked {{
                background-color: {p.bg_primary};
                border-color: {p.primary};
            }}

            /* === Scroll Bar === */
            QScrollBar:vertical {{
                background-color: {p.bg_secondary};
                width: 12px;
                border-radius: 6px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background-color: {p.border_default};
                border-radius: 6px;
                min-height: 30px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {p.border_strong};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            QScrollBar:horizontal {{
                background-color: {p.bg_secondary};
                height: 12px;
                border-radius: 6px;
                margin: 0;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {p.border_default};
                border-radius: 6px;
                min-width: 30px;
                margin: 2px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {p.border_strong};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0;
            }}

            /* === Table Widget === */
            QTableWidget, QTableView {{
                background-color: {p.bg_primary};
                color: {p.text_primary};
                gridline-color: {p.border_subtle};
                border: 1px solid {p.border_subtle};
                border-radius: {c.border_radius}px;
            }}
            QTableWidget::item, QTableView::item {{
                padding: {SPACING.SM}px;
            }}
            QTableWidget::item:selected, QTableView::item:selected {{
                background-color: {p.selection};
            }}
            QHeaderView::section {{
                background-color: {p.bg_tertiary};
                color: {p.text_primary};
                font-weight: 600;
                border: none;
                border-bottom: 1px solid {p.border_subtle};
                border-right: 1px solid {p.border_subtle};
                padding: {SPACING.SM}px;
            }}

            /* === List Widget === */
            QListWidget {{
                background-color: {p.bg_primary};
                color: {p.text_primary};
                border: 1px solid {p.border_subtle};
                border-radius: {c.border_radius}px;
            }}
            QListWidget::item {{
                padding: {SPACING.SM}px;
                border-radius: {c.border_radius - 2}px;
            }}
            QListWidget::item:selected {{
                background-color: {p.selection};
            }}
            QListWidget::item:hover {{
                background-color: {p.bg_tertiary};
            }}

            /* === Progress Bar === */
            QProgressBar {{
                background-color: {p.bg_tertiary};
                border: none;
                border-radius: {c.border_radius}px;
                height: 8px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {p.primary};
                border-radius: {c.border_radius}px;
            }}

            /* === Slider === */
            QSlider::groove:horizontal {{
                background-color: {p.bg_tertiary};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background-color: {p.primary};
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::handle:horizontal:hover {{
                background-color: {p.primary_hover};
            }}

            /* === Label === */
            QLabel {{
                color: {p.text_primary};
                background-color: transparent;
            }}

            /* === Tool Tip === */
            QToolTip {{
                background-color: {p.bg_elevated};
                color: {p.text_primary};
                border: 1px solid {p.border_subtle};
                border-radius: {c.border_radius}px;
                padding: {SPACING.SM}px;
            }}
        """

    def apply_to_widget(self, widget: QWidget) -> None:
        """Apply full stylesheet to a widget.

        Parameters
        ----------
        widget : QWidget
            Widget to style
        """
        widget.setStyleSheet(self.get_full_stylesheet())
