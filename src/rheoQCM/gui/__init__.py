"""GUI components for RheoQCM application.

This module provides PyQt6 widgets, dialogs, styling, and layout utilities
for the RheoQCM GUI application.

Public API
----------
Workers:
    BayesianFitWorker - QThread for background MCMC execution

Widgets:
    ConvergenceStatusWidget - Color-coded convergence indicator
    ConfidenceLevelSpinBox - Spinbox for confidence level selection
    UncertaintyBandToggle - Checkbox for band visibility

Components (Enhanced UI):
    ActionButton - Styled button with semantic variants
    IconButton - Compact icon-only button
    CollapsiblePanel - Expandable/collapsible content section
    StatusIndicator - Color-coded status LED
    ToolbarSection - Organized toolbar group
    CardWidget - Elevated card container
    LabeledInput - Input field with integrated label
    SectionHeader - Styled section header
    ButtonGroup - Grouped buttons with spacing
    StatusBar - Enhanced status bar

Styles:
    StyleManager - Application-wide style management
    StyleConfig - Style configuration parameters
    ButtonVariant - Button style variants enum
    COLORS_LIGHT, COLORS_DARK - Color palettes

Layouts:
    FormBuilder - Fluent API for building forms
    GridBuilder - Fluent API for building grids
    create_form_layout - Create aligned form layouts
    create_button_row - Create horizontal button rows
    create_split_panel - Create resizable split panels

Dialogs:
    DiagnosticViewerDialog - 2x3 grid of ArviZ diagnostic plots
    BayesianProgressDialog - Progress bar for MCMC execution
"""

from __future__ import annotations

from rheoQCM.gui.components import (
    ActionButton,
    ButtonGroup,
    CardWidget,
    CollapsiblePanel,
    IconButton,
    LabeledInput,
    SectionHeader,
    StatusBar,
    StatusIndicator,
    ToolbarSection,
)
from rheoQCM.gui.dialogs import BayesianProgressDialog, DiagnosticViewerDialog
from rheoQCM.gui.layouts import (
    FormBuilder,
    GridBuilder,
    add_separator,
    add_spacer,
    create_button_row,
    create_card_grid,
    create_form_layout,
    create_split_panel,
    create_toolbar_layout,
)
from rheoQCM.gui.styles import (
    COLORS_DARK,
    COLORS_LIGHT,
    SPACING,
    ButtonVariant,
    StyleConfig,
    StyleManager,
    get_button_stylesheet,
    get_input_stylesheet,
    get_panel_stylesheet,
)
from rheoQCM.gui.widgets import (
    ConfidenceLevelSpinBox,
    ConvergenceStatusWidget,
    UncertaintyBandToggle,
)
from rheoQCM.gui.workers import BayesianFitWorker

__all__ = [
    # Workers
    "BayesianFitWorker",
    # Dialogs
    "BayesianProgressDialog",
    "DiagnosticViewerDialog",
    # Original Widgets
    "ConfidenceLevelSpinBox",
    "ConvergenceStatusWidget",
    "UncertaintyBandToggle",
    # Enhanced Components
    "ActionButton",
    "ButtonGroup",
    "CardWidget",
    "CollapsiblePanel",
    "IconButton",
    "LabeledInput",
    "SectionHeader",
    "StatusBar",
    "StatusIndicator",
    "ToolbarSection",
    # Styles
    "ButtonVariant",
    "COLORS_DARK",
    "COLORS_LIGHT",
    "SPACING",
    "StyleConfig",
    "StyleManager",
    "get_button_stylesheet",
    "get_input_stylesheet",
    "get_panel_stylesheet",
    # Layouts
    "FormBuilder",
    "GridBuilder",
    "add_separator",
    "add_spacer",
    "create_button_row",
    "create_card_grid",
    "create_form_layout",
    "create_split_panel",
    "create_toolbar_layout",
]
