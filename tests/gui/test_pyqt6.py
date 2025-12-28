"""
Tests for PyQt6 framework integration.

This module contains tests to verify PyQt6 works correctly for the GUI layer.
Tests cover:
    - PyQt6 imports and enum namespaces
    - Main window creation
    - Widget initialization
    - Signal/slot connections
    - Matplotlib integration with PyQt6 backend
"""

import pytest

pytestmark = pytest.mark.gui


class TestPyQt6Imports:
    """Test PyQt6 import and enum changes."""

    def test_pyqt6_imports(self) -> None:
        """Test that all PyQt6 imports work correctly."""
        from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSlot
        from PyQt6.QtGui import QAction, QIcon, QPixmap
        from PyQt6.QtWidgets import (
            QApplication,
            QLabel,
            QMainWindow,
            QPushButton,
            QSizePolicy,
            QVBoxLayout,
            QWidget,
        )

        # Verify Qt enums use new namespace format
        assert hasattr(Qt, "CheckState")
        assert hasattr(Qt.CheckState, "Checked")
        assert hasattr(Qt, "AlignmentFlag")
        assert hasattr(Qt.AlignmentFlag, "AlignCenter")

    def test_pyqt6_enum_namespace(self) -> None:
        """Test PyQt6 enum namespace changes are handled correctly."""
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QSizePolicy

        # Test Qt.CheckState
        checked = Qt.CheckState.Checked
        unchecked = Qt.CheckState.Unchecked
        assert checked != unchecked

        # Test Qt.AlignmentFlag
        align_left = Qt.AlignmentFlag.AlignLeft
        align_center = Qt.AlignmentFlag.AlignCenter
        assert align_left != align_center

        # Test Qt.Orientation
        horizontal = Qt.Orientation.Horizontal
        vertical = Qt.Orientation.Vertical
        assert horizontal != vertical

        # Test Qt.FocusPolicy
        click_focus = Qt.FocusPolicy.ClickFocus
        assert click_focus is not None

        # Test QSizePolicy.Policy
        expanding = QSizePolicy.Policy.Expanding
        preferred = QSizePolicy.Policy.Preferred
        assert expanding != preferred


class TestWindowCreation:
    """Test main window and widget creation."""

    def test_main_window_creation(self, qapp) -> None:
        """Test that main window can be created with PyQt6."""
        from PyQt6.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget

        # Create main window
        window = QMainWindow()
        window.setWindowTitle("PyQt6 Test Window")
        window.resize(400, 300)

        # Add central widget
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        button = QPushButton("Test Button")
        layout.addWidget(button)
        window.setCentralWidget(central_widget)

        # Verify window properties
        assert window.windowTitle() == "PyQt6 Test Window"
        assert window.width() == 400
        assert window.height() == 300

    def test_widget_initialization(self, qapp) -> None:
        """Test that widgets initialize correctly with PyQt6."""
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import (
            QCheckBox,
            QComboBox,
            QLabel,
            QLineEdit,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        # Create container widget
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Create and add various widgets
        label = QLabel("Test Label")
        line_edit = QLineEdit()
        line_edit.setText("Test Text")
        checkbox = QCheckBox("Test Checkbox")
        checkbox.setCheckState(Qt.CheckState.Checked)
        combo = QComboBox()
        combo.addItems(["Item 1", "Item 2", "Item 3"])
        button = QPushButton("Test Button")

        layout.addWidget(label)
        layout.addWidget(line_edit)
        layout.addWidget(checkbox)
        layout.addWidget(combo)
        layout.addWidget(button)

        # Verify widgets
        assert label.text() == "Test Label"
        assert line_edit.text() == "Test Text"
        assert checkbox.checkState() == Qt.CheckState.Checked
        assert combo.count() == 3


class TestSignalSlot:
    """Test signal/slot connections."""

    def test_signal_slot_connections(self, qapp) -> None:
        """Test that signal/slot connections work with PyQt6."""
        from PyQt6.QtCore import QObject, pyqtSignal
        from PyQt6.QtWidgets import QPushButton

        # Track clicks
        click_count = [0]

        def on_click():
            click_count[0] += 1

        # Create button and connect signal
        button = QPushButton("Click Me")
        button.clicked.connect(on_click)

        # Simulate click
        button.click()

        # Verify signal was received
        assert click_count[0] == 1

        # Test custom signal
        class Emitter(QObject):
            custom_signal = pyqtSignal(str)

        received_data = [None]

        def on_custom_signal(data):
            received_data[0] = data

        emitter = Emitter()
        emitter.custom_signal.connect(on_custom_signal)
        emitter.custom_signal.emit("test_data")

        assert received_data[0] == "test_data"


class TestMatplotlibIntegration:
    """Test matplotlib integration with PyQt6."""

    def test_matplotlib_pyqt6_integration(self, qapp) -> None:
        """Test matplotlib integration with PyQt6 backend."""
        import matplotlib

        matplotlib.use("QtAgg")  # PyQt6 backend

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        from matplotlib.figure import Figure
        from PyQt6.QtWidgets import QVBoxLayout, QWidget

        # Create widget with matplotlib canvas
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Create figure and canvas
        fig = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvas(fig)

        # Add toolbar
        toolbar = NavigationToolbar2QT(canvas, widget)

        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        # Add a simple plot
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_title("Test Plot")

        # Draw the canvas
        canvas.draw()

        # Verify figure has content
        assert len(fig.axes) == 1
        assert fig.axes[0].get_title() == "Test Plot"


class TestQActionLocation:
    """Test QAction import location (moved to QtGui in PyQt6)."""

    def test_qaction_in_qtgui(self, qapp) -> None:
        """Test that QAction is correctly imported from QtGui in PyQt6."""
        from PyQt6.QtGui import QAction
        from PyQt6.QtWidgets import QMainWindow, QMenu, QMenuBar

        # Create main window with menu
        window = QMainWindow()
        menubar = window.menuBar()
        file_menu = menubar.addMenu("File")

        # Create action from QtGui
        action = QAction("Test Action", window)
        action.setShortcut("Ctrl+T")
        file_menu.addAction(action)

        # Verify action was added
        assert len(file_menu.actions()) == 1
        assert file_menu.actions()[0].text() == "Test Action"


class TestPyQt6CompatibilityPatterns:
    """Regression tests for PyQt6 compatibility patterns.

    These tests verify that correct PyQt6 patterns are used and prevent
    reintroduction of deprecated PyQt5 patterns.
    """

    def test_qmenu_method_signature(self, qapp) -> None:
        """Test that QMenu uses PyQt6 method signature (not underscore suffix)."""
        from PyQt6.QtWidgets import QMenu

        menu = QMenu()

        # PyQt6 uses the standard method name without underscore
        assert hasattr(menu, "exec"), "QMenu should have 'exec' method in PyQt6"

        # PyQt5 underscore version should NOT exist
        assert not hasattr(
            menu, "exec_"
        ), "QMenu should NOT have 'exec_' method in PyQt6"

    def test_qmessagebox_standardbutton_enum(self, qapp) -> None:
        """Test that QMessageBox uses StandardButton namespace for button enums."""
        from PyQt6.QtWidgets import QMessageBox

        # PyQt6 requires fully-qualified enum namespace
        assert hasattr(
            QMessageBox, "StandardButton"
        ), "QMessageBox should have StandardButton enum"
        assert hasattr(
            QMessageBox.StandardButton, "Ok"
        ), "StandardButton should have Ok"
        assert hasattr(
            QMessageBox.StandardButton, "Yes"
        ), "StandardButton should have Yes"
        assert hasattr(
            QMessageBox.StandardButton, "No"
        ), "StandardButton should have No"
        assert hasattr(
            QMessageBox.StandardButton, "Cancel"
        ), "StandardButton should have Cancel"

        # PyQt5 style direct access should NOT exist
        assert not hasattr(
            QMessageBox, "Ok"
        ), "QMessageBox should NOT have direct 'Ok' attribute"
        assert not hasattr(
            QMessageBox, "Yes"
        ), "QMessageBox should NOT have direct 'Yes' attribute"

    def test_qmessagebox_icon_enum(self, qapp) -> None:
        """Test that QMessageBox uses Icon namespace for icon enums."""
        from PyQt6.QtWidgets import QMessageBox

        # PyQt6 requires fully-qualified enum namespace
        assert hasattr(QMessageBox, "Icon"), "QMessageBox should have Icon enum"
        assert hasattr(QMessageBox.Icon, "Information"), "Icon should have Information"
        assert hasattr(QMessageBox.Icon, "Warning"), "Icon should have Warning"
        assert hasattr(QMessageBox.Icon, "Critical"), "Icon should have Critical"
        assert hasattr(QMessageBox.Icon, "Question"), "Icon should have Question"

        # PyQt5 style direct access should NOT exist
        assert not hasattr(
            QMessageBox, "Information"
        ), "QMessageBox should NOT have direct 'Information' attribute"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
