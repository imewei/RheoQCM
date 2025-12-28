"""GUI dialogs for Bayesian fitting visualization.

T066: DiagnosticViewerDialog - 2x3 grid of ArviZ diagnostic plots
T067: BayesianProgressDialog - Progress bar for MCMC execution
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from rheoQCM.core.bayesian import BayesianFitResult, BayesianFitter


class DiagnosticViewerDialog(QDialog):
    """Dialog displaying 2x3 grid of ArviZ diagnostic plots.

    Shows all 6 diagnostic plots:
    - Pair plot (parameter correlations)
    - Forest plot (posterior distributions)
    - Energy plot (NUTS energy diagnostics)
    - Autocorrelation plot
    - Rank plot (chain mixing)
    - ESS plot (effective sample size)
    """

    def __init__(
        self,
        result: BayesianFitResult,
        fitter: BayesianFitter,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.result = result
        self.fitter = fitter
        self._plot_paths: dict[str, Path] = {}
        self._setup_ui()
        self._generate_plots()

    def _setup_ui(self) -> None:
        self.setWindowTitle("MCMC Diagnostic Plots")
        self.setMinimumSize(1200, 800)

        layout = QVBoxLayout(self)

        # Scroll area for plots
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        self._grid = QGridLayout(container)

        # Create 6 plot labels in 2x3 grid
        self._labels: list[QLabel] = []
        for i in range(6):
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(380, 300)
            label.setText("Loading...")
            row, col = divmod(i, 3)
            self._grid.addWidget(label, row, col)
            self._labels.append(label)

        scroll.setWidget(container)
        layout.addWidget(scroll)

        # Buttons
        button_layout = QVBoxLayout()

        export_btn = QPushButton("Export All")
        export_btn.clicked.connect(self._export_plots)
        button_layout.addWidget(export_btn)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        button_layout.addWidget(buttons)

        layout.addLayout(button_layout)

    def _generate_plots(self) -> None:
        """Generate diagnostic plots in temp directory and display."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                paths = self.fitter.generate_diagnostic_suite(
                    self.result,
                    tmpdir,
                    formats=["png"],
                    dpi=100,
                )
                self._plot_paths = paths

                # Map plot types to grid positions
                plot_order = ["pair", "forest", "energy", "autocorr", "rank", "ess"]
                plot_titles = [
                    "Pair Plot (Correlations)",
                    "Forest Plot (Posteriors)",
                    "Energy Plot (NUTS)",
                    "Autocorrelation",
                    "Rank Plot (Mixing)",
                    "ESS Evolution",
                ]

                for i, (plot_type, title) in enumerate(
                    zip(plot_order, plot_titles, strict=False)
                ):
                    key = f"{plot_type}_png"
                    if key in paths:
                        pixmap = QPixmap(str(paths[key]))
                        scaled = pixmap.scaled(
                            380,
                            300,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                        self._labels[i].setPixmap(scaled)
                        self._labels[i].setToolTip(title)
                    else:
                        self._labels[i].setText(f"{title}\n(Failed to generate)")

            except Exception as e:
                for label in self._labels:
                    label.setText(f"Error: {e}")

    def _export_plots(self) -> None:
        """Export all plots to user-selected directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            "",
        )
        if not dir_path:
            return

        try:
            paths = self.fitter.generate_diagnostic_suite(
                self.result,
                dir_path,
                formats=["pdf", "png"],
                dpi=300,
            )
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {len(paths)} files to {dir_path}",
            )
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self,
                "Export Failed",
                f"Failed to export plots: {e}",
            )


class BayesianProgressDialog(QDialog):
    """Progress dialog for MCMC execution.

    Shows progress bar, status label, and cancel button.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._setup_ui()
        self._cancelled = False

    def _setup_ui(self) -> None:
        self.setWindowTitle("Bayesian Fitting")
        self.setMinimumWidth(400)
        self.setModal(True)

        layout = QVBoxLayout(self)

        self._status = QLabel("Initializing...")
        layout.addWidget(self._status)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self._cancel_btn)

    def update_progress(self, percent: int, phase: str) -> None:
        """Update progress display.

        Parameters
        ----------
        percent : int
            Progress percentage (0-100)
        phase : str
            Current phase description
        """
        self._progress.setValue(percent)
        self._status.setText(phase)

    def _on_cancel(self) -> None:
        self._cancelled = True
        self._status.setText("Cancelling...")
        self._cancel_btn.setEnabled(False)

    def was_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancelled
