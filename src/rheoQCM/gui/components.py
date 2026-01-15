"""Enhanced GUI components for professional scientific applications.

This module provides reusable UI components following best practices:
- ActionButton - Styled buttons with variants (primary, success, danger)
- IconButton - Compact icon-only buttons
- CollapsiblePanel - Expandable/collapsible content sections
- StatusIndicator - Color-coded status display
- ToolbarSection - Grouped toolbar with separators
- CardWidget - Elevated content container
- LabeledInput - Input with integrated label

Public API
----------
Classes:
    ActionButton - Button with semantic variants
    IconButton - Compact button with icon only
    CollapsiblePanel - Expandable panel widget
    StatusIndicator - Status LED indicator
    ToolbarSection - Organized toolbar group
    CardWidget - Elevated card container
    LabeledInput - Input field with label
    SectionHeader - Styled section header
"""

from __future__ import annotations

__all__ = [
    "ActionButton",
    "IconButton",
    "CollapsiblePanel",
    "StatusIndicator",
    "ToolbarSection",
    "CardWidget",
    "LabeledInput",
    "SectionHeader",
    "ButtonGroup",
    "StatusBar",
]

from typing import TYPE_CHECKING

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from rheoQCM.gui.styles import (
    COLORS_DARK,
    COLORS_LIGHT,
    SPACING,
    ButtonVariant,
    get_button_stylesheet,
)

if TYPE_CHECKING:
    pass


class ActionButton(QPushButton):
    """Styled push button with semantic variants.

    Provides consistent button styling with proper hover, pressed,
    and disabled states.

    Parameters
    ----------
    text : str
        Button label text
    variant : ButtonVariant
        Visual style variant (PRIMARY, SECONDARY, SUCCESS, DANGER, WARNING)
    icon : QIcon, optional
        Icon to display
    parent : QWidget, optional
        Parent widget

    Example
    -------
    >>> start_btn = ActionButton("Start Recording", ButtonVariant.SUCCESS)
    >>> stop_btn = ActionButton("Stop", ButtonVariant.DANGER)
    >>> save_btn = ActionButton("Save", ButtonVariant.PRIMARY)
    """

    def __init__(
        self,
        text: str = "",
        variant: ButtonVariant = ButtonVariant.PRIMARY,
        icon: QIcon | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(text, parent)
        self._variant = variant
        self._dark_mode = False

        if icon is not None:
            self.setIcon(icon)
            self.setIconSize(QSize(20, 20))

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_style()

    @property
    def variant(self) -> ButtonVariant:
        """Get button variant."""
        return self._variant

    def set_variant(self, variant: ButtonVariant) -> None:
        """Set button variant and update style."""
        self._variant = variant
        self._apply_style()

    def set_dark_mode(self, enabled: bool) -> None:
        """Set dark mode and update style."""
        self._dark_mode = enabled
        self._apply_style()

    def _apply_style(self) -> None:
        palette = COLORS_DARK if self._dark_mode else COLORS_LIGHT
        stylesheet = get_button_stylesheet(palette, self._variant)
        self.setStyleSheet(stylesheet)


class IconButton(QToolButton):
    """Compact icon-only button.

    Parameters
    ----------
    icon : QIcon
        Button icon
    tooltip : str, optional
        Tooltip text
    size : int
        Button size in pixels (default: 32)
    parent : QWidget, optional
        Parent widget
    """

    def __init__(
        self,
        icon: QIcon,
        tooltip: str = "",
        size: int = 32,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setIcon(icon)
        self.setIconSize(QSize(size - 8, size - 8))
        self.setFixedSize(size, size)
        self.setToolTip(tooltip)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._dark_mode = False
        self._apply_style()

    def set_dark_mode(self, enabled: bool) -> None:
        """Set dark mode and update style."""
        self._dark_mode = enabled
        self._apply_style()

    def _apply_style(self) -> None:
        p = COLORS_DARK if self._dark_mode else COLORS_LIGHT
        self.setStyleSheet(f"""
            QToolButton {{
                background-color: transparent;
                border: none;
                border-radius: 6px;
                padding: 4px;
            }}
            QToolButton:hover {{
                background-color: {p.bg_tertiary};
            }}
            QToolButton:pressed {{
                background-color: {p.border_subtle};
            }}
        """)


class CollapsiblePanel(QWidget):
    """Expandable/collapsible content panel.

    Provides a header that can be clicked to expand or collapse
    the content section, saving screen space.

    Parameters
    ----------
    title : str
        Panel header title
    expanded : bool
        Initial expanded state (default: True)
    parent : QWidget, optional
        Parent widget

    Signals
    -------
    toggled(bool)
        Emitted when panel is expanded/collapsed

    Example
    -------
    >>> panel = CollapsiblePanel("Advanced Settings")
    >>> panel.set_content_layout(my_layout)
    >>> panel.toggled.connect(on_panel_toggled)
    """

    toggled = pyqtSignal(bool)

    def __init__(
        self,
        title: str = "",
        expanded: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._expanded = expanded
        self._dark_mode = False

        self._setup_ui(title)
        self._apply_style()

    def _setup_ui(self, title: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        self._header = QPushButton()
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.clicked.connect(self.toggle)

        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(SPACING.SM, SPACING.SM, SPACING.SM, SPACING.SM)

        self._arrow = QLabel()
        self._arrow.setFixedSize(16, 16)
        header_layout.addWidget(self._arrow)

        self._title_label = QLabel(title)
        self._title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        header_layout.addWidget(self._title_label)
        header_layout.addStretch()

        layout.addWidget(self._header)

        # Content container
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(
            SPACING.MD, SPACING.SM, SPACING.MD, SPACING.SM
        )
        layout.addWidget(self._content)

        self._update_arrow()
        self._content.setVisible(self._expanded)

    def toggle(self) -> None:
        """Toggle expanded/collapsed state."""
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        self._update_arrow()
        self.toggled.emit(self._expanded)

    def expand(self) -> None:
        """Expand the panel."""
        if not self._expanded:
            self.toggle()

    def collapse(self) -> None:
        """Collapse the panel."""
        if self._expanded:
            self.toggle()

    def is_expanded(self) -> bool:
        """Check if panel is expanded."""
        return self._expanded

    def set_content_layout(self, layout) -> None:
        """Set the content area layout.

        Parameters
        ----------
        layout : QLayout
            Layout to use for content area
        """
        # Clear existing layout
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        # Add widgets from new layout
        if layout is not None:
            widget = QWidget()
            widget.setLayout(layout)
            self._content_layout.addWidget(widget)

    def content_layout(self):
        """Get the content layout."""
        return self._content_layout

    def _update_arrow(self) -> None:
        arrow = "\u25bc" if self._expanded else "\u25b6"
        self._arrow.setText(arrow)

    def set_dark_mode(self, enabled: bool) -> None:
        """Set dark mode."""
        self._dark_mode = enabled
        self._apply_style()

    def _apply_style(self) -> None:
        p = COLORS_DARK if self._dark_mode else COLORS_LIGHT
        self._header.setStyleSheet(f"""
            QPushButton {{
                background-color: {p.bg_tertiary};
                border: none;
                border-radius: 6px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {p.border_subtle};
            }}
        """)
        self._title_label.setStyleSheet(f"color: {p.text_primary};")
        self._arrow.setStyleSheet(f"color: {p.text_secondary};")
        self._content.setStyleSheet(f"""
            QWidget {{
                background-color: {p.bg_secondary};
                border: 1px solid {p.border_subtle};
                border-top: none;
                border-bottom-left-radius: 6px;
                border-bottom-right-radius: 6px;
            }}
        """)


class StatusIndicator(QWidget):
    """Color-coded status LED indicator.

    Displays a colored circle to indicate status with optional label.

    Parameters
    ----------
    label : str
        Status label text
    parent : QWidget, optional
        Parent widget
    """

    # Status colors
    COLOR_GOOD = QColor(76, 175, 80)  # Green
    COLOR_WARNING = QColor(255, 193, 7)  # Yellow/Orange
    COLOR_ERROR = QColor(244, 67, 54)  # Red
    COLOR_INACTIVE = QColor(158, 158, 158)  # Gray

    def __init__(
        self,
        label: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._color = self.COLOR_INACTIVE
        self._label_text = label

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.SM)

        self._indicator = _LEDIndicator(self)
        self._indicator.setFixedSize(12, 12)
        layout.addWidget(self._indicator)

        if self._label_text:
            self._label = QLabel(self._label_text)
            layout.addWidget(self._label)

        layout.addStretch()

    def set_status(self, status: str) -> None:
        """Set status by name.

        Parameters
        ----------
        status : str
            One of "good", "warning", "error", "inactive"
        """
        color_map = {
            "good": self.COLOR_GOOD,
            "warning": self.COLOR_WARNING,
            "error": self.COLOR_ERROR,
            "inactive": self.COLOR_INACTIVE,
        }
        self._color = color_map.get(status.lower(), self.COLOR_INACTIVE)
        self._indicator.set_color(self._color)

    def set_color(self, color: QColor) -> None:
        """Set custom status color."""
        self._color = color
        self._indicator.set_color(color)

    def set_label(self, text: str) -> None:
        """Update label text."""
        if hasattr(self, "_label"):
            self._label.setText(text)


class _LEDIndicator(QWidget):
    """Internal LED circle indicator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._color = QColor(158, 158, 158)

    def set_color(self, color: QColor) -> None:
        self._color = color
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(self._color)
        painter.setPen(Qt.PenStyle.NoPen)
        size = min(self.width(), self.height())
        painter.drawEllipse(
            (self.width() - size) // 2,
            (self.height() - size) // 2,
            size,
            size,
        )


class ToolbarSection(QWidget):
    """Grouped toolbar with optional label and separators.

    Parameters
    ----------
    label : str, optional
        Section label (shown above buttons)
    parent : QWidget, optional
        Parent widget

    Example
    -------
    >>> section = ToolbarSection("File Operations")
    >>> section.add_button(save_icon, "Save", on_save)
    >>> section.add_button(open_icon, "Open", on_open)
    >>> section.add_separator()
    """

    def __init__(
        self,
        label: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._dark_mode = False

        self._setup_ui(label)

    def _setup_ui(self, label: str) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(SPACING.XS)

        if label:
            self._label = QLabel(label)
            self._label.setStyleSheet(
                "font-size: 10px; color: #64748B; font-weight: 600;"
            )
            main_layout.addWidget(self._label)

        self._button_container = QWidget()
        self._button_layout = QHBoxLayout(self._button_container)
        self._button_layout.setContentsMargins(0, 0, 0, 0)
        self._button_layout.setSpacing(SPACING.XS)
        main_layout.addWidget(self._button_container)

    def add_button(
        self,
        icon: QIcon,
        tooltip: str,
        callback=None,
        size: int = 32,
    ) -> IconButton:
        """Add an icon button to the toolbar.

        Parameters
        ----------
        icon : QIcon
            Button icon
        tooltip : str
            Tooltip text
        callback : callable, optional
            Click handler
        size : int
            Button size

        Returns
        -------
        IconButton
            The created button
        """
        btn = IconButton(icon, tooltip, size)
        btn.set_dark_mode(self._dark_mode)
        if callback:
            btn.clicked.connect(callback)
        self._button_layout.addWidget(btn)
        return btn

    def add_separator(self) -> None:
        """Add a vertical separator."""
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color: {'#3F3F46' if self._dark_mode else '#E2E8F0'};")
        self._button_layout.addWidget(sep)

    def add_stretch(self) -> None:
        """Add flexible space."""
        self._button_layout.addStretch()

    def set_dark_mode(self, enabled: bool) -> None:
        """Set dark mode for all buttons."""
        self._dark_mode = enabled
        for i in range(self._button_layout.count()):
            widget = self._button_layout.itemAt(i).widget()
            if hasattr(widget, "set_dark_mode"):
                widget.set_dark_mode(enabled)


class CardWidget(QFrame):
    """Elevated card container for content grouping.

    Parameters
    ----------
    title : str, optional
        Card header title
    parent : QWidget, optional
        Parent widget
    """

    def __init__(
        self,
        title: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._dark_mode = False
        self._title = title

        self._setup_ui()
        self._apply_style()

    def _setup_ui(self) -> None:
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(
            SPACING.MD, SPACING.MD, SPACING.MD, SPACING.MD
        )
        self._main_layout.setSpacing(SPACING.SM)

        if self._title:
            self._header = QLabel(self._title)
            self._header.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
            self._main_layout.addWidget(self._header)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.addWidget(self._content)

    def content_layout(self):
        """Get content layout for adding widgets."""
        return self._content_layout

    def set_dark_mode(self, enabled: bool) -> None:
        """Set dark mode."""
        self._dark_mode = enabled
        self._apply_style()

    def _apply_style(self) -> None:
        p = COLORS_DARK if self._dark_mode else COLORS_LIGHT
        self.setStyleSheet(f"""
            CardWidget {{
                background-color: {p.bg_elevated};
                border: 1px solid {p.border_subtle};
                border-radius: 8px;
            }}
        """)
        if hasattr(self, "_header"):
            self._header.setStyleSheet(f"color: {p.text_primary};")


class LabeledInput(QWidget):
    """Input field with integrated label.

    Parameters
    ----------
    label : str
        Field label
    placeholder : str, optional
        Placeholder text
    parent : QWidget, optional
        Parent widget
    """

    textChanged = pyqtSignal(str)

    def __init__(
        self,
        label: str,
        placeholder: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._dark_mode = False

        self._setup_ui(label, placeholder)
        self._apply_style()

    def _setup_ui(self, label: str, placeholder: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.XS)

        self._label = QLabel(label)
        self._label.setFont(QFont("Segoe UI", 11, QFont.Weight.Medium))
        layout.addWidget(self._label)

        self._input = QLineEdit()
        self._input.setPlaceholderText(placeholder)
        self._input.textChanged.connect(self.textChanged.emit)
        layout.addWidget(self._input)

    def text(self) -> str:
        """Get input text."""
        return self._input.text()

    def set_text(self, text: str) -> None:
        """Set input text."""
        self._input.setText(text)

    def set_read_only(self, read_only: bool) -> None:
        """Set read-only state."""
        self._input.setReadOnly(read_only)

    def set_dark_mode(self, enabled: bool) -> None:
        """Set dark mode."""
        self._dark_mode = enabled
        self._apply_style()

    def _apply_style(self) -> None:
        p = COLORS_DARK if self._dark_mode else COLORS_LIGHT
        self._label.setStyleSheet(f"color: {p.text_secondary};")
        self._input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {p.bg_primary};
                color: {p.text_primary};
                border: 1px solid {p.border_default};
                border-radius: 6px;
                padding: 8px 12px;
            }}
            QLineEdit:focus {{
                border-color: {p.focus_ring};
            }}
        """)


class SectionHeader(QWidget):
    """Styled section header with optional action button.

    Parameters
    ----------
    title : str
        Section title
    action_text : str, optional
        Action button text
    parent : QWidget, optional
        Parent widget
    """

    actionClicked = pyqtSignal()

    def __init__(
        self,
        title: str,
        action_text: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._dark_mode = False

        self._setup_ui(title, action_text)
        self._apply_style()

    def _setup_ui(self, title: str, action_text: str) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, SPACING.SM)

        self._title = QLabel(title)
        self._title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        layout.addWidget(self._title)

        layout.addStretch()

        if action_text:
            self._action = QPushButton(action_text)
            self._action.setCursor(Qt.CursorShape.PointingHandCursor)
            self._action.clicked.connect(self.actionClicked.emit)
            layout.addWidget(self._action)

    def set_dark_mode(self, enabled: bool) -> None:
        """Set dark mode."""
        self._dark_mode = enabled
        self._apply_style()

    def _apply_style(self) -> None:
        p = COLORS_DARK if self._dark_mode else COLORS_LIGHT
        self._title.setStyleSheet(f"color: {p.text_primary};")
        if hasattr(self, "_action"):
            self._action.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {p.primary};
                    border: none;
                    font-weight: 500;
                    padding: 4px 8px;
                }}
                QPushButton:hover {{
                    background-color: {p.bg_tertiary};
                    border-radius: 4px;
                }}
            """)


class ButtonGroup(QWidget):
    """Grouped buttons with consistent spacing.

    Parameters
    ----------
    orientation : str
        "horizontal" or "vertical"
    parent : QWidget, optional
        Parent widget
    """

    def __init__(
        self,
        orientation: str = "horizontal",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        if orientation == "vertical":
            self._layout = QVBoxLayout(self)
        else:
            self._layout = QHBoxLayout(self)

        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(SPACING.SM)

    def add_button(self, button: QPushButton) -> None:
        """Add a button to the group."""
        self._layout.addWidget(button)

    def add_stretch(self) -> None:
        """Add flexible space."""
        self._layout.addStretch()


class StatusBar(QWidget):
    """Enhanced status bar with multiple segments.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dark_mode = False
        self._segments: dict[str, QLabel] = {}

        self._setup_ui()
        self._apply_style()

    def _setup_ui(self) -> None:
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(SPACING.SM, SPACING.XS, SPACING.SM, SPACING.XS)
        self._layout.setSpacing(SPACING.MD)

        # Default status segment
        self._status_label = QLabel("Ready")
        self._layout.addWidget(self._status_label)
        self._layout.addStretch()

    def set_status(self, text: str) -> None:
        """Set main status text."""
        self._status_label.setText(text)

    def add_segment(self, name: str, text: str = "") -> QLabel:
        """Add a named segment to the status bar.

        Parameters
        ----------
        name : str
            Segment identifier
        text : str
            Initial text

        Returns
        -------
        QLabel
            The segment label
        """
        label = QLabel(text)
        self._segments[name] = label
        self._layout.addWidget(label)
        return label

    def update_segment(self, name: str, text: str) -> None:
        """Update a segment's text."""
        if name in self._segments:
            self._segments[name].setText(text)

    def add_indicator(self, name: str, label: str = "") -> StatusIndicator:
        """Add a status indicator.

        Parameters
        ----------
        name : str
            Indicator identifier
        label : str
            Indicator label

        Returns
        -------
        StatusIndicator
            The indicator widget
        """
        indicator = StatusIndicator(label)
        self._segments[name] = indicator
        self._layout.addWidget(indicator)
        return indicator

    def set_dark_mode(self, enabled: bool) -> None:
        """Set dark mode."""
        self._dark_mode = enabled
        self._apply_style()

    def _apply_style(self) -> None:
        p = COLORS_DARK if self._dark_mode else COLORS_LIGHT
        self.setStyleSheet(f"""
            StatusBar {{
                background-color: {p.bg_secondary};
                border-top: 1px solid {p.border_subtle};
            }}
        """)
        self._status_label.setStyleSheet(f"""
            color: {p.text_secondary};
            font-size: 12px;
        """)
