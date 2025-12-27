"""
Tests for PyQt6 migration.

This module contains focused tests to verify PyQt6 migration is successful.
Tests cover:
- Main window creation
- Widget initialization
- Signal/slot connections
- Matplotlib integration with PyQt6
"""

import sys
import pytest

# Check PyQt6 is importable
def test_pyqt6_imports():
    """Test that all PyQt6 imports work correctly."""
    from PyQt6.QtCore import Qt, pyqtSlot, QTimer, QSize
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QPushButton,
        QVBoxLayout, QLabel, QSizePolicy
    )
    from PyQt6.QtGui import QIcon, QPixmap, QAction

    # Verify Qt enums use new namespace format
    assert hasattr(Qt, 'CheckState')
    assert hasattr(Qt.CheckState, 'Checked')
    assert hasattr(Qt, 'AlignmentFlag')
    assert hasattr(Qt.AlignmentFlag, 'AlignCenter')


def test_pyqt6_enum_namespace():
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


@pytest.fixture
def qapp():
    """Create QApplication for testing."""
    from PyQt6.QtWidgets import QApplication

    # Check if app already exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_main_window_creation(qapp):
    """Test that main window can be created with PyQt6."""
    from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton

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


def test_widget_initialization(qapp):
    """Test that widgets initialize correctly with PyQt6."""
    from PyQt6.QtWidgets import (
        QWidget, QLabel, QLineEdit, QCheckBox, QComboBox,
        QPushButton, QVBoxLayout
    )
    from PyQt6.QtCore import Qt

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


def test_signal_slot_connections(qapp):
    """Test that signal/slot connections work with PyQt6."""
    from PyQt6.QtWidgets import QPushButton
    from PyQt6.QtCore import pyqtSignal, QObject

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


def test_matplotlib_pyqt6_integration(qapp):
    """Test matplotlib integration with PyQt6 backend."""
    import matplotlib
    matplotlib.use('QtAgg')  # PyQt6 backend

    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
    from matplotlib.figure import Figure
    from PyQt6.QtWidgets import QWidget, QVBoxLayout

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


def test_qaction_in_qtgui(qapp):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
