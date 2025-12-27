"""
Fixtures for GUI tests requiring PyQt6.

This module provides:
    - QApplication fixture for PyQt6 tests
    - Main window fixtures
"""

import sys

import pytest


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for testing.

    This fixture is session-scoped to avoid creating multiple QApplication
    instances, which would cause errors.
    """
    from PyQt6.QtWidgets import QApplication

    # Check if app already exists (handles pytest-xdist)
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    # Don't quit the app - let pytest handle cleanup


@pytest.fixture
def main_window(qapp):
    """Create a basic QMainWindow for testing."""
    from PyQt6.QtWidgets import QMainWindow

    window = QMainWindow()
    window.setWindowTitle("Test Window")
    window.resize(800, 600)
    yield window
    window.close()
