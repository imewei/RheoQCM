"""GUI components for uncertainty visualization and Bayesian fitting.

This module provides PyQt6 widgets and dialogs for integrating uncertainty
visualization and Bayesian MCMC fitting into the QCMApp GUI.

Public API
----------
Workers:
    BayesianFitWorker - QThread for background MCMC execution

Widgets:
    ConvergenceStatusWidget - Color-coded convergence indicator
    ConfidenceLevelSpinBox - Spinbox for confidence level selection
    UncertaintyBandToggle - Checkbox for band visibility

Dialogs:
    DiagnosticViewerDialog - 2x3 grid of ArviZ diagnostic plots
    BayesianProgressDialog - Progress bar for MCMC execution
"""

from __future__ import annotations

from rheoQCM.gui.dialogs import BayesianProgressDialog, DiagnosticViewerDialog
from rheoQCM.gui.widgets import (
    ConfidenceLevelSpinBox,
    ConvergenceStatusWidget,
    UncertaintyBandToggle,
)
from rheoQCM.gui.workers import BayesianFitWorker

__all__ = [
    "BayesianFitWorker",
    "BayesianProgressDialog",
    "ConfidenceLevelSpinBox",
    "ConvergenceStatusWidget",
    "DiagnosticViewerDialog",
    "UncertaintyBandToggle",
]
