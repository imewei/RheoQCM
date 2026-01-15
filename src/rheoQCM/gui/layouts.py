"""Layout helper utilities for consistent UI arrangements.

This module provides layout utilities following Swiss Modernism 2.0 grid system
with 8px base unit spacing.

Public API
----------
Functions:
    create_form_layout - Create aligned form with labels
    create_button_row - Create horizontal button arrangement
    create_toolbar_layout - Create toolbar with sections
    create_card_grid - Create responsive card grid
    create_split_panel - Create resizable split panels

Classes:
    FormBuilder - Fluent API for building forms
    GridBuilder - Fluent API for building grids
    ResponsiveLayout - Adaptive layout manager
"""

from __future__ import annotations

__all__ = [
    "create_form_layout",
    "create_button_row",
    "create_toolbar_layout",
    "create_card_grid",
    "FormBuilder",
    "GridBuilder",
    "add_spacer",
    "add_separator",
]

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from rheoQCM.gui.styles import SPACING

if TYPE_CHECKING:
    pass


def create_form_layout(
    fields: list[tuple[str, QWidget]],
    label_width: int = 120,
    spacing: int = SPACING.SM,
) -> QFormLayout:
    """Create a form layout with aligned labels.

    Parameters
    ----------
    fields : list[tuple[str, QWidget]]
        List of (label, widget) tuples
    label_width : int
        Fixed width for labels (default: 120)
    spacing : int
        Vertical spacing between rows

    Returns
    -------
    QFormLayout
        Configured form layout

    Example
    -------
    >>> layout = create_form_layout([
    ...     ("Name:", name_input),
    ...     ("Frequency:", freq_spinbox),
    ...     ("Mode:", mode_combo),
    ... ])
    """
    layout = QFormLayout()
    layout.setContentsMargins(SPACING.MD, SPACING.MD, SPACING.MD, SPACING.MD)
    layout.setSpacing(spacing)
    layout.setLabelAlignment(
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    )
    layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

    for label_text, widget in fields:
        label = QLabel(label_text)
        label.setFixedWidth(label_width)
        layout.addRow(label, widget)

    return layout


def create_button_row(
    buttons: list[QPushButton],
    alignment: str = "right",
    spacing: int = SPACING.SM,
) -> QHBoxLayout:
    """Create a horizontal button row with proper alignment.

    Parameters
    ----------
    buttons : list[QPushButton]
        List of buttons to arrange
    alignment : str
        Button alignment: "left", "center", "right", "space-between"
    spacing : int
        Spacing between buttons

    Returns
    -------
    QHBoxLayout
        Configured layout

    Example
    -------
    >>> cancel_btn = QPushButton("Cancel")
    >>> save_btn = ActionButton("Save", ButtonVariant.PRIMARY)
    >>> layout = create_button_row([cancel_btn, save_btn], alignment="right")
    """
    layout = QHBoxLayout()
    layout.setContentsMargins(0, SPACING.MD, 0, 0)
    layout.setSpacing(spacing)

    if alignment == "right":
        layout.addStretch()
    elif alignment == "center":
        layout.addStretch()

    for i, btn in enumerate(buttons):
        layout.addWidget(btn)
        if alignment == "space-between" and i < len(buttons) - 1:
            layout.addStretch()

    if alignment == "center":
        layout.addStretch()
    elif alignment == "left":
        layout.addStretch()

    return layout


def create_toolbar_layout(
    sections: list[list[QWidget]],
    separator_between: bool = True,
) -> QHBoxLayout:
    """Create a toolbar layout with grouped sections.

    Parameters
    ----------
    sections : list[list[QWidget]]
        List of widget groups (each group is a list of widgets)
    separator_between : bool
        Add vertical separators between sections

    Returns
    -------
    QHBoxLayout
        Configured toolbar layout

    Example
    -------
    >>> layout = create_toolbar_layout([
    ...     [open_btn, save_btn, close_btn],  # File section
    ...     [undo_btn, redo_btn],              # Edit section
    ...     [zoom_in_btn, zoom_out_btn],       # View section
    ... ])
    """
    layout = QHBoxLayout()
    layout.setContentsMargins(SPACING.SM, SPACING.SM, SPACING.SM, SPACING.SM)
    layout.setSpacing(SPACING.XS)

    for i, section in enumerate(sections):
        if i > 0 and separator_between:
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.VLine)
            sep.setStyleSheet("color: #E2E8F0;")
            layout.addWidget(sep)

        for widget in section:
            layout.addWidget(widget)

    layout.addStretch()
    return layout


def create_card_grid(
    cards: list[QWidget],
    columns: int = 2,
    spacing: int = SPACING.MD,
) -> QGridLayout:
    """Create a grid layout for card widgets.

    Parameters
    ----------
    cards : list[QWidget]
        List of card widgets
    columns : int
        Number of columns
    spacing : int
        Grid spacing

    Returns
    -------
    QGridLayout
        Configured grid layout

    Example
    -------
    >>> cards = [CardWidget(f"Card {i}") for i in range(4)]
    >>> layout = create_card_grid(cards, columns=2)
    """
    layout = QGridLayout()
    layout.setContentsMargins(SPACING.MD, SPACING.MD, SPACING.MD, SPACING.MD)
    layout.setSpacing(spacing)

    for i, card in enumerate(cards):
        row = i // columns
        col = i % columns
        layout.addWidget(card, row, col)

    return layout


def add_spacer(
    layout: QLayout,
    width: int = 0,
    height: int = SPACING.MD,
    horizontal_policy: QSizePolicy.Policy = QSizePolicy.Policy.Minimum,
    vertical_policy: QSizePolicy.Policy = QSizePolicy.Policy.Minimum,
) -> None:
    """Add a spacer to a layout.

    Parameters
    ----------
    layout : QLayout
        Target layout
    width : int
        Spacer width
    height : int
        Spacer height
    horizontal_policy : QSizePolicy.Policy
        Horizontal size policy
    vertical_policy : QSizePolicy.Policy
        Vertical size policy
    """
    spacer = QSpacerItem(width, height, horizontal_policy, vertical_policy)
    if isinstance(layout, (QHBoxLayout, QVBoxLayout)):
        layout.addSpacerItem(spacer)


def add_separator(
    layout: QLayout,
    orientation: str = "horizontal",
    margin: int = SPACING.SM,
) -> QFrame:
    """Add a line separator to a layout.

    Parameters
    ----------
    layout : QLayout
        Target layout
    orientation : str
        "horizontal" or "vertical"
    margin : int
        Margin around separator

    Returns
    -------
    QFrame
        The separator widget
    """
    sep = QFrame()
    if orientation == "horizontal":
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setContentsMargins(0, margin, 0, margin)
    else:
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setContentsMargins(margin, 0, margin, 0)

    sep.setStyleSheet("color: #E2E8F0;")

    if isinstance(layout, (QHBoxLayout, QVBoxLayout)):
        layout.addWidget(sep)

    return sep


class FormBuilder:
    """Fluent API for building forms.

    Example
    -------
    >>> form = (FormBuilder()
    ...     .add_field("Name:", name_input)
    ...     .add_field("Email:", email_input)
    ...     .add_spacer()
    ...     .add_field("Password:", password_input)
    ...     .build())
    """

    def __init__(self) -> None:
        self._layout = QFormLayout()
        self._layout.setContentsMargins(SPACING.MD, SPACING.MD, SPACING.MD, SPACING.MD)
        self._layout.setSpacing(SPACING.SM)
        self._label_width = 120

    def set_label_width(self, width: int) -> FormBuilder:
        """Set label width."""
        self._label_width = width
        return self

    def set_spacing(self, spacing: int) -> FormBuilder:
        """Set row spacing."""
        self._layout.setSpacing(spacing)
        return self

    def set_margins(
        self,
        left: int = SPACING.MD,
        top: int = SPACING.MD,
        right: int = SPACING.MD,
        bottom: int = SPACING.MD,
    ) -> FormBuilder:
        """Set layout margins."""
        self._layout.setContentsMargins(left, top, right, bottom)
        return self

    def add_field(self, label: str, widget: QWidget) -> FormBuilder:
        """Add a labeled field."""
        lbl = QLabel(label)
        lbl.setFixedWidth(self._label_width)
        self._layout.addRow(lbl, widget)
        return self

    def add_widget(self, widget: QWidget) -> FormBuilder:
        """Add a full-width widget."""
        self._layout.addRow(widget)
        return self

    def add_spacer(self, height: int = SPACING.MD) -> FormBuilder:
        """Add vertical space."""
        spacer = QWidget()
        spacer.setFixedHeight(height)
        self._layout.addRow(spacer)
        return self

    def add_separator(self) -> FormBuilder:
        """Add a horizontal separator."""
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #E2E8F0;")
        self._layout.addRow(sep)
        return self

    def add_section(self, title: str) -> FormBuilder:
        """Add a section header."""
        label = QLabel(title)
        label.setStyleSheet("font-weight: 600; color: #0F172A; margin-top: 8px;")
        self._layout.addRow(label)
        return self

    def build(self) -> QFormLayout:
        """Build and return the form layout."""
        return self._layout


class GridBuilder:
    """Fluent API for building grid layouts.

    Example
    -------
    >>> grid = (GridBuilder(columns=3)
    ...     .add(widget1)
    ...     .add(widget2)
    ...     .add(widget3, span_cols=2)
    ...     .build())
    """

    def __init__(self, columns: int = 2) -> None:
        self._layout = QGridLayout()
        self._layout.setContentsMargins(SPACING.MD, SPACING.MD, SPACING.MD, SPACING.MD)
        self._layout.setSpacing(SPACING.MD)
        self._columns = columns
        self._current_row = 0
        self._current_col = 0

    def set_spacing(self, spacing: int) -> GridBuilder:
        """Set grid spacing."""
        self._layout.setSpacing(spacing)
        return self

    def set_margins(
        self,
        left: int = SPACING.MD,
        top: int = SPACING.MD,
        right: int = SPACING.MD,
        bottom: int = SPACING.MD,
    ) -> GridBuilder:
        """Set layout margins."""
        self._layout.setContentsMargins(left, top, right, bottom)
        return self

    def add(
        self,
        widget: QWidget,
        span_rows: int = 1,
        span_cols: int = 1,
    ) -> GridBuilder:
        """Add a widget to the grid.

        Parameters
        ----------
        widget : QWidget
            Widget to add
        span_rows : int
            Number of rows to span
        span_cols : int
            Number of columns to span
        """
        self._layout.addWidget(
            widget,
            self._current_row,
            self._current_col,
            span_rows,
            span_cols,
        )

        # Move to next position
        self._current_col += span_cols
        if self._current_col >= self._columns:
            self._current_col = 0
            self._current_row += span_rows

        return self

    def add_at(
        self,
        widget: QWidget,
        row: int,
        col: int,
        span_rows: int = 1,
        span_cols: int = 1,
    ) -> GridBuilder:
        """Add a widget at a specific position."""
        self._layout.addWidget(widget, row, col, span_rows, span_cols)
        return self

    def next_row(self) -> GridBuilder:
        """Move to the next row."""
        self._current_row += 1
        self._current_col = 0
        return self

    def set_column_stretch(self, column: int, stretch: int) -> GridBuilder:
        """Set column stretch factor."""
        self._layout.setColumnStretch(column, stretch)
        return self

    def set_row_stretch(self, row: int, stretch: int) -> GridBuilder:
        """Set row stretch factor."""
        self._layout.setRowStretch(row, stretch)
        return self

    def build(self) -> QGridLayout:
        """Build and return the grid layout."""
        return self._layout


def create_split_panel(
    left: QWidget,
    right: QWidget,
    orientation: Qt.Orientation = Qt.Orientation.Horizontal,
    sizes: tuple[int, int] = (300, 500),
    collapsible: tuple[bool, bool] = (True, False),
) -> QSplitter:
    """Create a resizable split panel.

    Parameters
    ----------
    left : QWidget
        Left (or top) panel widget
    right : QWidget
        Right (or bottom) panel widget
    orientation : Qt.Orientation
        Splitter orientation
    sizes : tuple[int, int]
        Initial sizes of panels
    collapsible : tuple[bool, bool]
        Whether each panel is collapsible

    Returns
    -------
    QSplitter
        Configured splitter
    """
    splitter = QSplitter(orientation)
    splitter.addWidget(left)
    splitter.addWidget(right)
    splitter.setSizes(list(sizes))
    splitter.setCollapsible(0, collapsible[0])
    splitter.setCollapsible(1, collapsible[1])
    splitter.setHandleWidth(2)
    splitter.setStyleSheet("""
        QSplitter::handle {
            background-color: #E2E8F0;
        }
        QSplitter::handle:hover {
            background-color: #3B82F6;
        }
    """)
    return splitter


def create_three_panel_layout(
    left: QWidget,
    center: QWidget,
    right: QWidget,
    sizes: tuple[int, int, int] = (250, 400, 350),
) -> QSplitter:
    """Create a three-panel horizontal layout.

    Parameters
    ----------
    left : QWidget
        Left panel (settings/controls)
    center : QWidget
        Center panel (main content)
    right : QWidget
        Right panel (details/properties)
    sizes : tuple[int, int, int]
        Initial panel sizes

    Returns
    -------
    QSplitter
        Configured splitter
    """
    splitter = QSplitter(Qt.Orientation.Horizontal)
    splitter.addWidget(left)
    splitter.addWidget(center)
    splitter.addWidget(right)
    splitter.setSizes(list(sizes))
    splitter.setStretchFactor(0, 2)  # Left: 20%
    splitter.setStretchFactor(1, 4)  # Center: 40%
    splitter.setStretchFactor(2, 4)  # Right: 40%
    splitter.setCollapsible(0, True)
    splitter.setCollapsible(1, False)
    splitter.setCollapsible(2, True)
    splitter.setHandleWidth(2)
    return splitter
