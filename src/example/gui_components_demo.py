#!/usr/bin/env python3
"""Demo script showcasing the enhanced GUI components.

Run with: uv run python src/example/gui_components_demo.py
"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QLineEdit,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rheoQCM.gui import (
    ActionButton,
    ButtonGroup,
    ButtonVariant,
    CardWidget,
    CollapsiblePanel,
    FormBuilder,
    GridBuilder,
    LabeledInput,
    SectionHeader,
    StatusBar,
    StatusIndicator,
    StyleManager,
    create_button_row,
)


class GUIComponentsDemo(QMainWindow):
    """Demo window showcasing the enhanced GUI components."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RheoQCM Enhanced GUI Components Demo")
        self.setMinimumSize(800, 600)

        # Initialize style manager
        self._style_manager = StyleManager.instance()

        self._setup_ui()
        self._apply_style()

    def _setup_ui(self) -> None:
        """Setup the main UI."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Header with theme toggle
        header = SectionHeader("Enhanced GUI Components", "Toggle Dark Mode")
        header.actionClicked.connect(self._toggle_dark_mode)
        layout.addWidget(header)

        # Tab widget for different component showcases
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Tab 1: Buttons
        tabs.addTab(self._create_buttons_tab(), "Buttons")

        # Tab 2: Forms
        tabs.addTab(self._create_forms_tab(), "Forms")

        # Tab 3: Panels
        tabs.addTab(self._create_panels_tab(), "Panels")

        # Tab 4: Status
        tabs.addTab(self._create_status_tab(), "Status")

        # Status bar
        self._status_bar = StatusBar()
        self._status_bar.set_status("Ready - Click components to see them in action")
        self._status_bar.add_segment("theme", "Light Mode")
        layout.addWidget(self._status_bar)

    def _create_buttons_tab(self) -> QWidget:
        """Create the buttons showcase tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Section: Primary Buttons
        section1 = SectionHeader("Button Variants")
        layout.addWidget(section1)

        buttons_card = CardWidget("Action Buttons")
        btn_layout = buttons_card.content_layout()

        # Create buttons with different variants
        variants = [
            ("Primary", ButtonVariant.PRIMARY),
            ("Secondary", ButtonVariant.SECONDARY),
            ("Success", ButtonVariant.SUCCESS),
            ("Danger", ButtonVariant.DANGER),
            ("Warning", ButtonVariant.WARNING),
            ("Ghost", ButtonVariant.GHOST),
        ]

        for text, variant in variants:
            btn = ActionButton(text, variant)
            btn.clicked.connect(lambda checked, t=text: self._on_button_clicked(t))
            btn_layout.addWidget(btn)

        layout.addWidget(buttons_card)

        # Section: Button Row
        row_card = CardWidget("Button Row Alignment")
        row_layout = row_card.content_layout()

        cancel = ActionButton("Cancel", ButtonVariant.SECONDARY)
        save = ActionButton("Save Changes", ButtonVariant.PRIMARY)
        button_row = create_button_row([cancel, save], alignment="right")

        row_container = QWidget()
        row_container.setLayout(button_row)
        row_layout.addWidget(row_container)

        layout.addWidget(row_card)

        # Section: Button Groups
        group_card = CardWidget("Button Groups")
        group_layout = group_card.content_layout()

        group = ButtonGroup("horizontal")
        group.add_button(ActionButton("Option A", ButtonVariant.SECONDARY))
        group.add_button(ActionButton("Option B", ButtonVariant.SECONDARY))
        group.add_button(ActionButton("Option C", ButtonVariant.PRIMARY))
        group_layout.addWidget(group)

        layout.addWidget(group_card)
        layout.addStretch()

        return widget

    def _create_forms_tab(self) -> QWidget:
        """Create the forms showcase tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Section: Form Builder
        section = SectionHeader("Form Builder API")
        layout.addWidget(section)

        card = CardWidget("Sample Form")
        card_layout = card.content_layout()

        # Create form using FormBuilder
        name_input = QLineEdit()
        name_input.setPlaceholderText("Enter sample name...")

        freq_spin = QDoubleSpinBox()
        freq_spin.setRange(1.0, 100.0)
        freq_spin.setValue(5.0)
        freq_spin.setSuffix(" MHz")

        temp_spin = QDoubleSpinBox()
        temp_spin.setRange(0.0, 100.0)
        temp_spin.setValue(25.0)
        temp_spin.setSuffix(" \u00b0C")

        form = (
            FormBuilder()
            .set_label_width(100)
            .add_section("Basic Information")
            .add_field("Name:", name_input)
            .add_field("Frequency:", freq_spin)
            .add_spacer()
            .add_section("Environment")
            .add_field("Temperature:", temp_spin)
            .add_separator()
            .build()
        )

        form_widget = QWidget()
        form_widget.setLayout(form)
        card_layout.addWidget(form_widget)

        layout.addWidget(card)

        # Section: Labeled Inputs
        labeled_card = CardWidget("Labeled Input Components")
        labeled_layout = labeled_card.content_layout()

        labeled1 = LabeledInput("Sample ID", "Enter unique identifier")
        labeled2 = LabeledInput("Description", "Optional notes...")
        labeled_layout.addWidget(labeled1)
        labeled_layout.addWidget(labeled2)

        layout.addWidget(labeled_card)
        layout.addStretch()

        return widget

    def _create_panels_tab(self) -> QWidget:
        """Create the panels showcase tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Section: Collapsible Panels
        section = SectionHeader("Collapsible Panels")
        layout.addWidget(section)

        # Create collapsible panels
        panel1 = CollapsiblePanel("Basic Settings", expanded=True)
        p1_layout = panel1.content_layout()
        p1_layout.addWidget(LabeledInput("Setting 1", "Value 1"))
        p1_layout.addWidget(LabeledInput("Setting 2", "Value 2"))
        layout.addWidget(panel1)

        panel2 = CollapsiblePanel("Advanced Settings", expanded=False)
        p2_layout = panel2.content_layout()
        p2_layout.addWidget(LabeledInput("Advanced Option 1", "Complex value"))
        p2_layout.addWidget(LabeledInput("Advanced Option 2", "Another value"))
        layout.addWidget(panel2)

        panel3 = CollapsiblePanel("Expert Settings", expanded=False)
        p3_layout = panel3.content_layout()
        p3_layout.addWidget(LabeledInput("Expert Mode", "Danger zone!"))
        layout.addWidget(panel3)

        # Section: Card Grid
        grid_section = SectionHeader("Card Grid Layout")
        layout.addWidget(grid_section)

        grid = (
            GridBuilder(columns=2)
            .set_spacing(16)
            .add(CardWidget("Card 1"))
            .add(CardWidget("Card 2"))
            .add(CardWidget("Wide Card"), span_cols=2)
            .build()
        )

        grid_widget = QWidget()
        grid_widget.setLayout(grid)
        layout.addWidget(grid_widget)

        layout.addStretch()
        return widget

    def _create_status_tab(self) -> QWidget:
        """Create the status indicators showcase tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Section: Status Indicators
        section = SectionHeader("Status Indicators")
        layout.addWidget(section)

        card = CardWidget("Convergence Status")
        card_layout = card.content_layout()

        # Create status indicators
        statuses = [
            ("good", "Convergence: Good (R-hat < 1.01)"),
            ("warning", "Convergence: Warning (R-hat 1.01-1.05)"),
            ("error", "Convergence: Poor (R-hat > 1.05)"),
            ("inactive", "Not Running"),
        ]

        for status, label in statuses:
            indicator = StatusIndicator(label)
            indicator.set_status(status)
            card_layout.addWidget(indicator)

        layout.addWidget(card)

        # Interactive status control
        control_card = CardWidget("Control Status")
        control_layout = control_card.content_layout()

        self._demo_indicator = StatusIndicator("Measurement Status")
        self._demo_indicator.set_status("inactive")
        control_layout.addWidget(self._demo_indicator)

        btn_group = ButtonGroup("horizontal")
        for status in ["good", "warning", "error", "inactive"]:
            btn = ActionButton(status.capitalize(), ButtonVariant.SECONDARY)
            btn.clicked.connect(lambda _, s=status: self._set_demo_status(s))
            btn_group.add_button(btn)
        control_layout.addWidget(btn_group)

        layout.addWidget(control_card)
        layout.addStretch()

        return widget

    def _toggle_dark_mode(self) -> None:
        """Toggle dark mode."""
        is_dark = not self._style_manager.is_dark
        self._style_manager.set_dark_mode(is_dark)
        self._apply_style()
        mode = "Dark Mode" if is_dark else "Light Mode"
        self._status_bar.update_segment("theme", mode)
        self._status_bar.set_status(f"Switched to {mode}")

    def _apply_style(self) -> None:
        """Apply current style to window."""
        self._style_manager.apply_to_widget(self)

    def _on_button_clicked(self, text: str) -> None:
        """Handle button click."""
        self._status_bar.set_status(f"Button clicked: {text}")

    def _set_demo_status(self, status: str) -> None:
        """Set demo status indicator."""
        self._demo_indicator.set_status(status)
        self._status_bar.set_status(f"Status set to: {status}")


def main() -> None:
    """Run the demo application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = GUIComponentsDemo()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
