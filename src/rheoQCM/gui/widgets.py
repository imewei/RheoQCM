"""GUI widgets for uncertainty visualization.

T065: ConvergenceStatusWidget - Color-coded convergence indicator
T061: UncertaintyBandToggle - Checkbox for band visibility
T062: ConfidenceLevelSpinBox - Spinbox for confidence level
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QCheckBox, QDoubleSpinBox, QHBoxLayout, QLabel, QWidget

if TYPE_CHECKING:
    from rheoQCM.core.bayesian import BayesianFitResult


class ConvergenceStatusWidget(QWidget):
    """Color-coded convergence status indicator.

    Displays convergence quality based on R-hat, ESS, and divergence count.

    Colors:
        - Green: All R-hat < 1.01, ESS > 400
        - Yellow: Any R-hat 1.01-1.05 OR ESS 100-400
        - Red: Any R-hat > 1.05 OR ESS < 100 OR divergences > 10
    """

    # Color constants
    GREEN = QColor(76, 175, 80)
    YELLOW = QColor(255, 193, 7)
    RED = QColor(244, 67, 54)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._setup_ui()
        self._rhat: dict[str, float] = {}
        self._ess: dict[str, float] = {}
        self._divergences: int = 0

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._indicator = QLabel()
        self._indicator.setFixedSize(16, 16)
        self._indicator.setAutoFillBackground(True)
        self._set_color(self.GREEN)

        self._label = QLabel("Convergence")
        layout.addWidget(self._indicator)
        layout.addWidget(self._label)

    def _set_color(self, color: QColor) -> None:
        palette = self._indicator.palette()
        palette.setColor(QPalette.ColorRole.Window, color)
        self._indicator.setPalette(palette)
        self._indicator.setStyleSheet(
            f"background-color: {color.name()}; border-radius: 8px;"
        )

    def update_status(
        self,
        rhat: dict[str, float],
        ess: dict[str, float],
        divergences: int,
    ) -> None:
        """Update convergence status.

        Parameters
        ----------
        rhat : dict[str, float]
            R-hat values by parameter name
        ess : dict[str, float]
            ESS values by parameter name
        divergences : int
            Number of divergent transitions
        """
        self._rhat = rhat
        self._ess = ess
        self._divergences = divergences

        color, status = self._compute_status()
        self._set_color(color)
        self._label.setText(status)
        self._update_tooltip()

    def _compute_status(self) -> tuple[QColor, str]:
        """Compute status color and label."""
        max_rhat = max(self._rhat.values()) if self._rhat else 1.0
        min_ess = min(self._ess.values()) if self._ess else 1000

        # Red conditions
        if max_rhat > 1.05 or min_ess < 100 or self._divergences > 10:
            return self.RED, "Poor"

        # Yellow conditions
        if max_rhat >= 1.01 or min_ess < 400:
            return self.YELLOW, "Warning"

        # Green: good convergence
        return self.GREEN, "Good"

    def _update_tooltip(self) -> None:
        lines = []
        if self._rhat:
            max_rhat = max(self._rhat.values())
            lines.append(f"Max R-hat: {max_rhat:.3f}")
        if self._ess:
            min_ess = min(self._ess.values())
            lines.append(f"Min ESS: {min_ess:.0f}")
        if self._divergences > 0:
            lines.append(f"Divergences: {self._divergences}")

        self.setToolTip("\n".join(lines) if lines else "No data")

    def from_result(self, result: BayesianFitResult) -> None:
        """Update status from BayesianFitResult."""
        self.update_status(result.rhat, result.ess, result.divergences)


class UncertaintyBandToggle(QCheckBox):
    """Checkbox to toggle uncertainty band visibility."""

    def __init__(self, label: str = "Show Uncertainty Bands", parent=None) -> None:
        super().__init__(label, parent)
        self.setChecked(True)


class ConfidenceLevelSpinBox(QDoubleSpinBox):
    """Spinbox for selecting confidence/credible level.

    Range: 0.90 to 0.99
    Default: 0.95
    Step: 0.01
    """

    # Signal emitted when value changes (convenience wrapper)
    levelChanged = pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setRange(0.90, 0.99)
        self.setSingleStep(0.01)
        self.setDecimals(2)
        self.setValue(0.95)
        self.setPrefix("CI: ")
        self.setSuffix("")
        self.valueChanged.connect(self._on_value_changed)

    def _on_value_changed(self, value: float) -> None:
        self.levelChanged.emit(value)

    def confidence_level(self) -> float:
        """Get current confidence level."""
        return self.value()
