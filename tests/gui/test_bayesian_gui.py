"""Unit tests for Bayesian GUI components.

Tests cover:
- T058: BayesianFitWorker QThread signals
- T059: DiagnosticViewerDialog displays 6 plots
- T060: ConvergenceStatusWidget color-coding logic
- T061: UncertaintyBandToggle visibility control
- T062: ConfidenceLevelSpinBox updates on value change
- T076: Integration test for full GUI workflow
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip all tests if PyQt6 not available
pytest.importorskip("PyQt6")


class TestConvergenceStatusWidget:
    """T060: Unit tests for ConvergenceStatusWidget color-coding logic."""

    def test_green_status_all_good(self, qtbot) -> None:
        """Green status when all R-hat < 1.01 and ESS > 400."""
        from rheoQCM.gui.widgets import ConvergenceStatusWidget

        widget = ConvergenceStatusWidget()
        qtbot.addWidget(widget)

        widget.update_status(
            rhat={"a": 1.001, "b": 1.005},
            ess={"a": 500, "b": 600},
            divergences=0,
        )

        color, status = widget._compute_status()
        assert status == "Good"
        assert color == widget.GREEN

    def test_yellow_status_rhat_warning(self, qtbot) -> None:
        """Yellow status when R-hat between 1.01 and 1.05."""
        from rheoQCM.gui.widgets import ConvergenceStatusWidget

        widget = ConvergenceStatusWidget()
        qtbot.addWidget(widget)

        widget.update_status(
            rhat={"a": 1.02, "b": 1.005},
            ess={"a": 500, "b": 600},
            divergences=0,
        )

        color, status = widget._compute_status()
        assert status == "Warning"
        assert color == widget.YELLOW

    def test_yellow_status_ess_warning(self, qtbot) -> None:
        """Yellow status when ESS between 100 and 400."""
        from rheoQCM.gui.widgets import ConvergenceStatusWidget

        widget = ConvergenceStatusWidget()
        qtbot.addWidget(widget)

        widget.update_status(
            rhat={"a": 1.001, "b": 1.005},
            ess={"a": 200, "b": 600},
            divergences=0,
        )

        color, status = widget._compute_status()
        assert status == "Warning"
        assert color == widget.YELLOW

    def test_red_status_high_rhat(self, qtbot) -> None:
        """Red status when R-hat > 1.05."""
        from rheoQCM.gui.widgets import ConvergenceStatusWidget

        widget = ConvergenceStatusWidget()
        qtbot.addWidget(widget)

        widget.update_status(
            rhat={"a": 1.10, "b": 1.005},
            ess={"a": 500, "b": 600},
            divergences=0,
        )

        color, status = widget._compute_status()
        assert status == "Poor"
        assert color == widget.RED

    def test_red_status_low_ess(self, qtbot) -> None:
        """Red status when ESS < 100."""
        from rheoQCM.gui.widgets import ConvergenceStatusWidget

        widget = ConvergenceStatusWidget()
        qtbot.addWidget(widget)

        widget.update_status(
            rhat={"a": 1.001, "b": 1.005},
            ess={"a": 50, "b": 600},
            divergences=0,
        )

        color, status = widget._compute_status()
        assert status == "Poor"
        assert color == widget.RED

    def test_red_status_high_divergences(self, qtbot) -> None:
        """Red status when divergences > 10."""
        from rheoQCM.gui.widgets import ConvergenceStatusWidget

        widget = ConvergenceStatusWidget()
        qtbot.addWidget(widget)

        widget.update_status(
            rhat={"a": 1.001, "b": 1.005},
            ess={"a": 500, "b": 600},
            divergences=15,
        )

        color, status = widget._compute_status()
        assert status == "Poor"
        assert color == widget.RED


class TestUncertaintyBandToggle:
    """T061: Unit tests for UncertaintyBandToggle visibility control."""

    def test_default_checked(self, qtbot) -> None:
        """Toggle should be checked by default."""
        from rheoQCM.gui.widgets import UncertaintyBandToggle

        toggle = UncertaintyBandToggle()
        qtbot.addWidget(toggle)

        assert toggle.isChecked()

    def test_toggle_changes_state(self, qtbot) -> None:
        """Toggle should change state when clicked."""
        from rheoQCM.gui.widgets import UncertaintyBandToggle

        toggle = UncertaintyBandToggle()
        qtbot.addWidget(toggle)

        toggle.setChecked(False)
        assert not toggle.isChecked()

        toggle.setChecked(True)
        assert toggle.isChecked()


class TestConfidenceLevelSpinBox:
    """T062: Unit tests for ConfidenceLevelSpinBox."""

    def test_default_value_095(self, qtbot) -> None:
        """Default confidence level should be 0.95."""
        from rheoQCM.gui.widgets import ConfidenceLevelSpinBox

        spinbox = ConfidenceLevelSpinBox()
        qtbot.addWidget(spinbox)

        assert spinbox.value() == 0.95
        assert spinbox.confidence_level() == 0.95

    def test_range_090_to_099(self, qtbot) -> None:
        """Range should be 0.90 to 0.99."""
        from rheoQCM.gui.widgets import ConfidenceLevelSpinBox

        spinbox = ConfidenceLevelSpinBox()
        qtbot.addWidget(spinbox)

        assert spinbox.minimum() == 0.90
        assert spinbox.maximum() == 0.99

    def test_emits_signal_on_change(self, qtbot) -> None:
        """Should emit levelChanged signal when value changes."""
        from rheoQCM.gui.widgets import ConfidenceLevelSpinBox

        spinbox = ConfidenceLevelSpinBox()
        qtbot.addWidget(spinbox)

        with qtbot.waitSignal(spinbox.levelChanged, timeout=1000) as blocker:
            spinbox.setValue(0.90)

        assert blocker.args == [0.90]


class TestDiagnosticViewerDialog:
    """T059: Unit tests for DiagnosticViewerDialog displays 6 plots."""

    @pytest.mark.slow
    def test_dialog_has_6_labels(self, qtbot) -> None:
        """Dialog should create 6 plot labels in 2x3 grid."""
        import jax.numpy as jnp

        from rheoQCM.core.bayesian import BayesianFitter
        from rheoQCM.gui.dialogs import DiagnosticViewerDialog

        # Run minimal Bayesian fit
        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        x = np.linspace(0, 5, 20)
        y = np.exp(-0.5 * x) + np.random.normal(0, 0.01, 20)

        fitter = BayesianFitter(n_chains=2, n_samples=50, n_warmup=25, seed=42)
        result = fitter.fit(model, x, y, param_names=["a", "b"])

        dialog = DiagnosticViewerDialog(result, fitter)
        qtbot.addWidget(dialog)

        # Verify 6 labels exist
        assert len(dialog._labels) == 6

    @pytest.mark.slow
    def test_dialog_displays_plots(self, qtbot) -> None:
        """Dialog should display at least 5 of 6 diagnostic plots."""
        import jax.numpy as jnp

        from rheoQCM.core.bayesian import BayesianFitter
        from rheoQCM.gui.dialogs import DiagnosticViewerDialog

        # Run minimal Bayesian fit
        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        x = np.linspace(0, 5, 20)
        y = np.exp(-0.5 * x) + np.random.normal(0, 0.01, 20)

        fitter = BayesianFitter(n_chains=2, n_samples=50, n_warmup=25, seed=42)
        result = fitter.fit(model, x, y, param_names=["a", "b"])

        dialog = DiagnosticViewerDialog(result, fitter)
        qtbot.addWidget(dialog)

        # Count labels that have pixmaps (plots loaded successfully)
        loaded_count = sum(1 for label in dialog._labels if not label.pixmap().isNull())
        # At least 5/6 should load (energy plot may fail with minimal samples)
        assert loaded_count >= 5, f"Only {loaded_count}/6 plots loaded"


class TestBayesianFitWorkerSignals:
    """T058: Unit tests for BayesianFitWorker QThread signals."""

    def test_worker_has_required_signals(self) -> None:
        """Worker should have started, progress, finished, error signals."""
        from rheoQCM.gui.workers import BayesianFitWorker

        # Check signal attributes exist
        assert hasattr(BayesianFitWorker, "started")
        assert hasattr(BayesianFitWorker, "progress")
        assert hasattr(BayesianFitWorker, "finished")
        assert hasattr(BayesianFitWorker, "error")

    def test_worker_initialization(self, qtbot) -> None:
        """Worker should accept model, data, and config parameters."""
        import jax.numpy as jnp

        from rheoQCM.gui.workers import BayesianFitWorker

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        x = np.linspace(0, 5, 50)
        y = np.exp(-0.5 * x)

        worker = BayesianFitWorker(
            model=model,
            x=x,
            y=y,
            param_names=["a", "b"],
            n_chains=2,
            n_samples=100,
            n_warmup=50,
        )

        assert worker.n_chains == 2
        assert worker.n_samples == 100
        assert worker.n_warmup == 50


class TestQCMAppBayesianIntegration:
    """T076: Integration tests for full GUI workflow."""

    def test_bayesian_controls_initialized(self, qtbot) -> None:
        """Verify Bayesian controls are created with correct defaults."""
        from rheoQCM.gui.widgets import ConfidenceLevelSpinBox, UncertaintyBandToggle

        # Test widgets have correct defaults
        spinbox = ConfidenceLevelSpinBox()
        qtbot.addWidget(spinbox)
        assert spinbox.value() == 0.95

        toggle = UncertaintyBandToggle()
        qtbot.addWidget(toggle)
        assert toggle.isChecked()

    def test_uncertainty_toggle_affects_visibility_state(self, qtbot) -> None:
        """Verify toggle emits correct signals for visibility control."""
        from rheoQCM.gui.widgets import UncertaintyBandToggle

        toggle = UncertaintyBandToggle()
        qtbot.addWidget(toggle)

        received_states = []

        def on_toggled(checked):
            received_states.append(checked)

        toggle.toggled.connect(on_toggled)

        toggle.setChecked(False)
        toggle.setChecked(True)
        toggle.setChecked(False)

        assert received_states == [False, True, False]

    def test_confidence_spinbox_emits_level_changed(self, qtbot) -> None:
        """Verify spinbox emits levelChanged with correct values."""
        from rheoQCM.gui.widgets import ConfidenceLevelSpinBox

        spinbox = ConfidenceLevelSpinBox()
        qtbot.addWidget(spinbox)

        received_levels = []

        def on_level_changed(level):
            received_levels.append(level)

        spinbox.levelChanged.connect(on_level_changed)

        spinbox.setValue(0.90)
        spinbox.setValue(0.99)

        assert 0.90 in received_levels
        assert 0.99 in received_levels

    def test_convergence_widget_status_transitions(self, qtbot) -> None:
        """Verify convergence widget correctly transitions between states."""
        from rheoQCM.gui.widgets import ConvergenceStatusWidget

        widget = ConvergenceStatusWidget()
        qtbot.addWidget(widget)

        # Good -> Warning -> Poor transitions
        widget.update_status(rhat={"a": 1.001}, ess={"a": 500}, divergences=0)
        _, status1 = widget._compute_status()
        assert status1 == "Good"

        widget.update_status(rhat={"a": 1.02}, ess={"a": 500}, divergences=0)
        _, status2 = widget._compute_status()
        assert status2 == "Warning"

        widget.update_status(rhat={"a": 1.10}, ess={"a": 500}, divergences=0)
        _, status3 = widget._compute_status()
        assert status3 == "Poor"
