"""
Tests for PlotManager interface and implementations.

T048: Create test_plot_manager.py with tests for:
- Update spectrum/timeseries/properties
- Clear operations
- Autoscale toggle
- Axis limits
- Export operations
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from rheoQCM.services.plotting import (
    DefaultPlotManager,
    MockPlotManager,
    PlotCall,
    PlotStyle,
)


class TestMockPlotManager:
    """Tests for MockPlotManager."""

    def test_update_spectrum_records_call(self):
        """Test update_spectrum records the call."""
        manager = MockPlotManager()

        freq = np.array([1e6, 2e6, 3e6])
        data = np.array([0.1, 0.2, 0.3])

        manager.update_spectrum(freq, data, harmonic=3)

        assert len(manager.calls) == 1
        call = manager.calls[0]
        assert call.method == "update_spectrum"
        np.testing.assert_array_equal(call.args[0], freq)
        np.testing.assert_array_equal(call.args[1], data)
        assert call.args[2] == 3

    def test_update_spectrum_with_style(self):
        """Test update_spectrum with custom style."""
        manager = MockPlotManager()

        style = PlotStyle(color="red", linewidth=2.0, marker="o", label="Test")
        manager.update_spectrum(
            np.array([1e6]), np.array([0.1]), harmonic=1, style=style
        )

        call = manager.calls[0]
        assert call.kwargs["style"] == style

    def test_update_timeseries_records_call(self):
        """Test update_timeseries records the call."""
        manager = MockPlotManager()

        time = np.array([0, 1, 2])
        data = {
            "freq_shift": np.array([-100, -200, -300]),
            "dissipation": np.array([10, 20, 30]),
        }

        manager.update_timeseries(time, data, clear_first=True)

        assert len(manager.calls) == 1
        call = manager.calls[0]
        assert call.method == "update_timeseries"
        np.testing.assert_array_equal(call.args[0], time)
        assert "freq_shift" in call.args[1]
        assert call.kwargs["clear_first"] is True

    def test_update_properties_records_call(self):
        """Test update_properties records the call."""
        manager = MockPlotManager()

        time = np.array([0, 1, 2])
        drho = np.array([1e-6, 2e-6, 3e-6])
        grho = np.array([1e10, 2e10, 3e10])
        phi = np.array([0.1, 0.2, 0.3])

        manager.update_properties(time, drho, grho, phi)

        assert len(manager.calls) == 1
        call = manager.calls[0]
        assert call.method == "update_properties"
        np.testing.assert_array_equal(call.args[0], time)
        np.testing.assert_array_equal(call.args[1], drho)
        np.testing.assert_array_equal(call.args[2], grho)
        np.testing.assert_array_equal(call.args[3], phi)

    def test_clear_all_records_call(self):
        """Test clear_all records the call."""
        manager = MockPlotManager()
        manager.clear_all()

        assert len(manager.calls) == 1
        assert manager.calls[0].method == "clear_all"

    def test_clear_plot_records_call(self):
        """Test clear_plot records the call."""
        manager = MockPlotManager()
        manager.clear_plot("spectrum")

        assert len(manager.calls) == 1
        call = manager.calls[0]
        assert call.method == "clear_plot"
        assert call.args[0] == "spectrum"

    def test_set_autoscale(self):
        """Test set_autoscale updates state and records call."""
        manager = MockPlotManager()
        assert manager.autoscale is True

        manager.set_autoscale(False)

        assert manager.autoscale is False
        assert len(manager.calls) == 1
        assert manager.calls[0].method == "set_autoscale"
        assert manager.calls[0].args[0] is False

    def test_set_xlim_records_call(self):
        """Test set_xlim records the call."""
        manager = MockPlotManager()
        manager.set_xlim("spectrum", 4.9e6, 5.1e6)

        call = manager.calls[0]
        assert call.method == "set_xlim"
        assert call.args == ("spectrum", 4.9e6, 5.1e6)

    def test_set_ylim_records_call(self):
        """Test set_ylim records the call."""
        manager = MockPlotManager()
        manager.set_ylim("spectrum", -0.5, 0.5)

        call = manager.calls[0]
        assert call.method == "set_ylim"
        assert call.args == ("spectrum", -0.5, 0.5)

    def test_export_figure_records_call(self):
        """Test export_figure records the call."""
        manager = MockPlotManager()
        path = Path("/tmp/test.png")

        manager.export_figure("spectrum", path, format="png", dpi=300)

        call = manager.calls[0]
        assert call.method == "export_figure"
        assert call.args == ("spectrum", path)
        assert call.kwargs["format"] == "png"
        assert call.kwargs["dpi"] == 300

    def test_get_calls_filters_by_method(self):
        """Test get_calls filters by method name."""
        manager = MockPlotManager()

        manager.update_spectrum(np.array([1e6]), np.array([0.1]), 1)
        manager.clear_all()
        manager.update_spectrum(np.array([2e6]), np.array([0.2]), 3)

        spectrum_calls = manager.get_calls("update_spectrum")
        assert len(spectrum_calls) == 2

        clear_calls = manager.get_calls("clear_all")
        assert len(clear_calls) == 1

    def test_assert_called_passes(self):
        """Test assert_called passes when method was called."""
        manager = MockPlotManager()
        manager.update_spectrum(np.array([1e6]), np.array([0.1]), 1)

        # Should not raise
        manager.assert_called("update_spectrum")
        manager.assert_called("update_spectrum", times=1)

    def test_assert_called_fails_when_not_called(self):
        """Test assert_called fails when method was not called."""
        manager = MockPlotManager()

        with pytest.raises(AssertionError, match="Expected at least one call"):
            manager.assert_called("update_spectrum")

    def test_assert_called_fails_wrong_count(self):
        """Test assert_called fails with wrong count."""
        manager = MockPlotManager()
        manager.update_spectrum(np.array([1e6]), np.array([0.1]), 1)

        with pytest.raises(AssertionError, match="Expected 2 calls"):
            manager.assert_called("update_spectrum", times=2)

    def test_reset_clears_calls(self):
        """Test reset clears all recorded calls."""
        manager = MockPlotManager()
        manager.update_spectrum(np.array([1e6]), np.array([0.1]), 1)
        manager.clear_all()

        assert len(manager.calls) == 2

        manager.reset()

        assert len(manager.calls) == 0


class TestDefaultPlotManager:
    """Tests for DefaultPlotManager with mock widgets."""

    def test_init_with_no_widgets(self):
        """Test initialization with no widgets."""
        manager = DefaultPlotManager()

        # Should not raise, just log debug message
        manager.update_spectrum(np.array([1e6]), np.array([0.1]), 1)

    def test_update_spectrum_with_mock_widget(self):
        """Test update_spectrum with mock widget."""
        mock_widget = MagicMock()
        mock_widget.axes = MagicMock()

        manager = DefaultPlotManager(spectrum_widget=mock_widget)
        manager.update_spectrum(np.array([1e6, 2e6]), np.array([0.1, 0.2]), 3)

        mock_widget.axes.clear.assert_called_once()
        mock_widget.axes.plot.assert_called()
        mock_widget.draw.assert_called_once()

    def test_update_spectrum_complex_data(self):
        """Test update_spectrum with complex data."""
        mock_widget = MagicMock()
        mock_widget.axes = MagicMock()

        manager = DefaultPlotManager(spectrum_widget=mock_widget)
        complex_data = np.array([0.1 + 0.01j, 0.2 + 0.02j])

        manager.update_spectrum(np.array([1e6, 2e6]), complex_data, 3)

        # Should plot both real and imaginary
        assert mock_widget.axes.plot.call_count >= 2

    def test_update_timeseries_with_mock_widget(self):
        """Test update_timeseries with mock widget."""
        mock_widget = MagicMock()
        mock_widget.axes = MagicMock()

        manager = DefaultPlotManager(timeseries_widget=mock_widget)
        time = np.array([0, 1, 2])
        data = {"test": np.array([1, 2, 3])}

        manager.update_timeseries(time, data, clear_first=True)

        mock_widget.axes.clear.assert_called_once()
        mock_widget.axes.plot.assert_called()

    def test_clear_all_clears_all_widgets(self):
        """Test clear_all clears all registered widgets."""
        widgets = {
            "spectrum": MagicMock(),
            "timeseries": MagicMock(),
            "properties": MagicMock(),
        }
        for w in widgets.values():
            w.figure.axes = [MagicMock()]

        manager = DefaultPlotManager(
            spectrum_widget=widgets["spectrum"],
            timeseries_widget=widgets["timeseries"],
            properties_widget=widgets["properties"],
        )
        manager.clear_all()

        for w in widgets.values():
            w.draw.assert_called()

    def test_register_widget(self):
        """Test registering a widget after initialization."""
        manager = DefaultPlotManager()
        mock_widget = MagicMock()
        mock_widget.axes = MagicMock()

        manager.register_widget("spectrum", mock_widget)
        manager.update_spectrum(np.array([1e6]), np.array([0.1]), 1)

        mock_widget.draw.assert_called_once()

    def test_set_autoscale(self):
        """Test autoscale toggle."""
        manager = DefaultPlotManager()

        manager.set_autoscale(False)
        # Internal state should be updated
        assert manager._autoscale is False

    def test_set_xlim_with_widget(self):
        """Test setting x limits."""
        mock_widget = MagicMock()
        mock_ax = MagicMock()
        mock_widget.figure.axes = [mock_ax]

        manager = DefaultPlotManager(spectrum_widget=mock_widget)
        manager.set_xlim("spectrum", 4.9e6, 5.1e6)

        mock_ax.set_xlim.assert_called_once_with(4.9e6, 5.1e6)

    def test_export_figure_raises_for_unknown_plot(self):
        """Test export_figure raises for unknown plot name."""
        manager = DefaultPlotManager()

        with pytest.raises(ValueError, match="Unknown plot"):
            manager.export_figure("nonexistent", Path("/tmp/test.png"))


class TestPlotStyle:
    """Tests for PlotStyle dataclass."""

    def test_defaults(self):
        """Test default values."""
        style = PlotStyle()

        assert style.color == "blue"
        assert style.linewidth == 1.0
        assert style.marker is None
        assert style.label is None

    def test_custom_values(self):
        """Test custom values."""
        style = PlotStyle(color="red", linewidth=2.5, marker="o", label="Test Data")

        assert style.color == "red"
        assert style.linewidth == 2.5
        assert style.marker == "o"
        assert style.label == "Test Data"


class TestPlotCall:
    """Tests for PlotCall dataclass."""

    def test_creation(self):
        """Test PlotCall creation."""
        call = PlotCall(
            method="update_spectrum", args=(1, 2, 3), kwargs={"style": None}
        )

        assert call.method == "update_spectrum"
        assert call.args == (1, 2, 3)
        assert call.kwargs == {"style": None}

    def test_default_kwargs(self):
        """Test default empty kwargs."""
        call = PlotCall(method="clear_all", args=())

        assert call.kwargs == {}
