"""
Plot Manager Interface and Implementations.

This module provides the PlotManager interface for matplotlib widget
coordination, enabling independent testing and alternative visualization backends.

T052-T053: Implement PlotManager interface and MockPlotManager.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PlotStyle:
    """Plot styling options."""

    color: str = "blue"
    linewidth: float = 1.0
    marker: str | None = None
    label: str | None = None


@dataclass
class PlotCall:
    """Record of a plot method call (for testing)."""

    method: str
    args: tuple
    kwargs: dict = field(default_factory=dict)


class PlotManager(Protocol):
    """Interface for plot coordination."""

    def update_spectrum(
        self,
        frequency: np.ndarray,
        data: np.ndarray,
        harmonic: int,
        *,
        style: PlotStyle | None = None,
    ) -> None:
        """
        Update spectrum plot for a specific harmonic.

        Args:
            frequency: Frequency array in Hz.
            data: Complex or real data to plot.
            harmonic: Harmonic number (1, 3, 5, ...).
            style: Optional plot styling.
        """
        ...

    def update_timeseries(
        self,
        time: np.ndarray,
        data: dict[str, np.ndarray],
        *,
        clear_first: bool = False,
    ) -> None:
        """
        Update time series plots.

        Args:
            time: Time array in seconds.
            data: Dict mapping labels to data arrays.
            clear_first: Whether to clear existing data.
        """
        ...

    def update_properties(
        self,
        time: np.ndarray,
        drho: np.ndarray,
        grho: np.ndarray,
        phi: np.ndarray,
    ) -> None:
        """
        Update property plots (mass, modulus, phase).

        Args:
            time: Time array in seconds.
            drho: Areal mass density array.
            grho: Complex modulus array.
            phi: Phase angle array.
        """
        ...

    def clear_all(self) -> None:
        """Clear all plots."""
        ...

    def clear_plot(self, name: str) -> None:
        """
        Clear a specific plot.

        Args:
            name: Plot identifier ("spectrum", "timeseries", "properties").
        """
        ...

    def set_autoscale(self, enabled: bool) -> None:
        """
        Enable or disable autoscaling.

        Args:
            enabled: Whether to autoscale on data update.
        """
        ...

    def set_xlim(self, name: str, xmin: float, xmax: float) -> None:
        """Set x-axis limits for a plot."""
        ...

    def set_ylim(self, name: str, ymin: float, ymax: float) -> None:
        """Set y-axis limits for a plot."""
        ...

    def export_figure(
        self,
        name: str,
        path: Path,
        *,
        format: str = "png",
        dpi: int = 150,
    ) -> None:
        """
        Export a figure to file.

        Args:
            name: Plot identifier.
            path: Output file path.
            format: File format (png, pdf, svg).
            dpi: Resolution for raster formats.
        """
        ...


class DefaultPlotManager:
    """Default implementation using matplotlib widgets."""

    def __init__(
        self,
        spectrum_widget: Any = None,
        timeseries_widget: Any = None,
        properties_widget: Any = None,
    ):
        self._widgets: dict[str, Any] = {
            "spectrum": spectrum_widget,
            "timeseries": timeseries_widget,
            "properties": properties_widget,
        }
        self._autoscale = True

    def update_spectrum(
        self,
        frequency: np.ndarray,
        data: np.ndarray,
        harmonic: int,
        *,
        style: PlotStyle | None = None,
    ) -> None:
        widget = self._widgets.get("spectrum")
        if widget is None:
            logger.debug("No spectrum widget registered")
            return

        style = style or PlotStyle()
        ax = widget.axes

        # Clear and replot
        ax.clear()

        if np.iscomplexobj(data):
            ax.plot(
                frequency,
                data.real,
                label=f"n={harmonic} (real)",
                **self._style_kwargs(style),
            )
            ax.plot(
                frequency,
                data.imag,
                label=f"n={harmonic} (imag)",
                linestyle="--",
                color=style.color,
            )
        else:
            ax.plot(frequency, data, label=f"n={harmonic}", **self._style_kwargs(style))

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.legend()

        if self._autoscale:
            ax.autoscale()

        widget.draw()

    def update_timeseries(
        self,
        time: np.ndarray,
        data: dict[str, np.ndarray],
        *,
        clear_first: bool = False,
    ) -> None:
        widget = self._widgets.get("timeseries")
        if widget is None:
            logger.debug("No timeseries widget registered")
            return

        ax = widget.axes

        if clear_first:
            ax.clear()

        for label, values in data.items():
            ax.plot(time, values, label=label)

        ax.set_xlabel("Time (s)")
        ax.legend()

        if self._autoscale:
            ax.autoscale()

        widget.draw()

    def update_properties(
        self,
        time: np.ndarray,
        drho: np.ndarray,
        grho: np.ndarray,
        phi: np.ndarray,
    ) -> None:
        widget = self._widgets.get("properties")
        if widget is None:
            logger.debug("No properties widget registered")
            return

        # Assume widget has multiple axes
        try:
            axes = widget.figure.axes
            if len(axes) >= 3:
                axes[0].clear()
                axes[0].plot(time, drho, label="drho")
                axes[0].set_ylabel("drho (kg/m²)")
                axes[0].legend()

                axes[1].clear()
                axes[1].plot(time, grho, label="grho")
                axes[1].set_ylabel("grho (Pa·kg/m³)")
                axes[1].legend()

                axes[2].clear()
                axes[2].plot(time, np.rad2deg(phi), label="phi")
                axes[2].set_xlabel("Time (s)")
                axes[2].set_ylabel("phi (deg)")
                axes[2].legend()

            widget.draw()
        except Exception as e:
            logger.warning("Failed to update properties plot: %s", e)

    def clear_all(self) -> None:
        for name in self._widgets:
            self.clear_plot(name)

    def clear_plot(self, name: str) -> None:
        widget = self._widgets.get(name)
        if widget is None:
            return

        try:
            for ax in widget.figure.axes:
                ax.clear()
            widget.draw()
        except Exception as e:
            logger.warning("Failed to clear plot %s: %s", name, e)

    def set_autoscale(self, enabled: bool) -> None:
        self._autoscale = enabled

    def set_xlim(self, name: str, xmin: float, xmax: float) -> None:
        widget = self._widgets.get(name)
        if widget is None:
            return

        try:
            for ax in widget.figure.axes:
                ax.set_xlim(xmin, xmax)
            widget.draw()
        except Exception as e:
            logger.warning("Failed to set xlim for %s: %s", name, e)

    def set_ylim(self, name: str, ymin: float, ymax: float) -> None:
        widget = self._widgets.get(name)
        if widget is None:
            return

        try:
            for ax in widget.figure.axes:
                ax.set_ylim(ymin, ymax)
            widget.draw()
        except Exception as e:
            logger.warning("Failed to set ylim for %s: %s", name, e)

    def export_figure(
        self,
        name: str,
        path: Path,
        *,
        format: str = "png",
        dpi: int = 150,
    ) -> None:
        widget = self._widgets.get(name)
        if widget is None:
            raise ValueError(f"Unknown plot: {name}")

        try:
            widget.figure.savefig(path, format=format, dpi=dpi)
            logger.info("Exported %s to %s", name, path)
        except Exception as e:
            logger.error("Failed to export %s: %s", name, e)
            raise

    def register_widget(self, name: str, widget: Any) -> None:
        """Register a widget for a plot type."""
        self._widgets[name] = widget

    def _style_kwargs(self, style: PlotStyle) -> dict:
        kwargs: dict[str, Any] = {"color": style.color, "linewidth": style.linewidth}
        if style.marker:
            kwargs["marker"] = style.marker
        if style.label:
            kwargs["label"] = style.label
        return kwargs


class MockPlotManager:
    """Mock implementation for testing."""

    def __init__(self):
        self.calls: list[PlotCall] = []
        self.autoscale = True

    def update_spectrum(
        self,
        frequency: np.ndarray,
        data: np.ndarray,
        harmonic: int,
        *,
        style: PlotStyle | None = None,
    ) -> None:
        self.calls.append(
            PlotCall(
                method="update_spectrum",
                args=(frequency, data, harmonic),
                kwargs={"style": style},
            )
        )

    def update_timeseries(
        self,
        time: np.ndarray,
        data: dict[str, np.ndarray],
        *,
        clear_first: bool = False,
    ) -> None:
        self.calls.append(
            PlotCall(
                method="update_timeseries",
                args=(time, data),
                kwargs={"clear_first": clear_first},
            )
        )

    def update_properties(
        self,
        time: np.ndarray,
        drho: np.ndarray,
        grho: np.ndarray,
        phi: np.ndarray,
    ) -> None:
        self.calls.append(
            PlotCall(
                method="update_properties",
                args=(time, drho, grho, phi),
                kwargs={},
            )
        )

    def clear_all(self) -> None:
        self.calls.append(PlotCall(method="clear_all", args=(), kwargs={}))

    def clear_plot(self, name: str) -> None:
        self.calls.append(PlotCall(method="clear_plot", args=(name,), kwargs={}))

    def set_autoscale(self, enabled: bool) -> None:
        self.autoscale = enabled
        self.calls.append(PlotCall(method="set_autoscale", args=(enabled,), kwargs={}))

    def set_xlim(self, name: str, xmin: float, xmax: float) -> None:
        self.calls.append(
            PlotCall(method="set_xlim", args=(name, xmin, xmax), kwargs={})
        )

    def set_ylim(self, name: str, ymin: float, ymax: float) -> None:
        self.calls.append(
            PlotCall(method="set_ylim", args=(name, ymin, ymax), kwargs={})
        )

    def export_figure(
        self,
        name: str,
        path: Path,
        *,
        format: str = "png",
        dpi: int = 150,
    ) -> None:
        self.calls.append(
            PlotCall(
                method="export_figure",
                args=(name, path),
                kwargs={"format": format, "dpi": dpi},
            )
        )

    # Test helpers
    def get_calls(self, method: str) -> list[PlotCall]:
        """Get all calls to a specific method."""
        return [c for c in self.calls if c.method == method]

    def assert_called(self, method: str, times: int | None = None) -> None:
        """Assert a method was called."""
        calls = self.get_calls(method)
        if times is not None:
            assert (
                len(calls) == times
            ), f"Expected {times} calls to {method}, got {len(calls)}"
        else:
            assert len(calls) > 0, f"Expected at least one call to {method}"

    def reset(self) -> None:
        """Clear all recorded calls."""
        self.calls.clear()
