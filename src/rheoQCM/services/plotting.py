"""
Plot Manager Interface and Implementations.

This module provides the PlotManager interface for matplotlib widget
coordination, enabling independent testing and alternative visualization backends.

T052-T053: Implement PlotManager interface and MockPlotManager.
T020: Implement plot_fit_with_uncertainty() for uncertainty visualization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from rheoQCM.core.uncertainty import UncertaintyBand

logger = logging.getLogger(__name__)

# Type aliases
Float64Array = npt.NDArray[np.float64]


def plot_fit_with_uncertainty(
    x_data: Float64Array,
    y_data: Float64Array,
    band: UncertaintyBand,
    *,
    ax: Axes | None = None,
    data_color: str = "black",
    fit_color: str = "blue",
    band_color: str = "blue",
    band_alpha: float = 0.3,
    data_label: str = "Data",
    fit_label: str = "Fit",
    band_label: str | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str | None = None,
) -> Figure:
    """Plot data with fitted curve and uncertainty band.

    Creates a publication-quality plot showing raw data points,
    the fitted curve, and a shaded confidence band.

    Parameters
    ----------
    x_data : Float64Array
        Original x-data points
    y_data : Float64Array
        Original y-data points
    band : UncertaintyBand
        Computed uncertainty band from UncertaintyCalculator
    ax : Axes | None
        Matplotlib axes (creates new if None)
    data_color : str
        Color for data points (default: "black")
    fit_color : str
        Color for fitted curve (default: "blue")
    band_color : str
        Color for confidence band (default: "blue")
    band_alpha : float
        Transparency for confidence band (default: 0.3)
    data_label : str
        Label for data points (default: "Data")
    fit_label : str
        Label for fitted curve (default: "Fit")
    band_label : str | None
        Label for confidence band (default: auto from confidence_level)
    xlabel : str
        X-axis label (default: "x")
    ylabel : str
        Y-axis label (default: "y")
    title : str | None
        Plot title (optional)

    Returns
    -------
    Figure
        Matplotlib Figure object

    Examples
    --------
    >>> from rheoQCM.core.uncertainty import UncertaintyCalculator
    >>> from rheoQCM.services.plotting import plot_fit_with_uncertainty
    >>> import numpy as np
    >>> def model(x, a, b): return a * np.exp(-b * x)
    >>> calc = UncertaintyCalculator()
    >>> band = calc.compute_band(model, x_pred, popt, pcov)
    >>> fig = plot_fit_with_uncertainty(x_data, y_data, band)
    >>> fig.savefig("fit_with_uncertainty.pdf", dpi=300)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.scatter(x_data, y_data, color=data_color, alpha=0.6, label=data_label, zorder=3)

    ax.plot(band.x, band.y_fit, color=fit_color, linewidth=2, label=fit_label, zorder=2)

    if band_label is None:
        band_label = f"{band.confidence_level:.0%} CI"

    ax.fill_between(
        band.x,
        band.y_lower,
        band.y_upper,
        color=band_color,
        alpha=band_alpha,
        label=band_label,
        zorder=1,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_bayesian_fit(
    x_data: Float64Array,
    y_data: Float64Array,
    x_pred: Float64Array,
    median: Float64Array,
    lower: Float64Array,
    upper: Float64Array,
    credible_level: float = 0.95,
    *,
    ax: Axes | None = None,
    nlsq_fit: Float64Array | None = None,
    data_color: str = "black",
    bayesian_color: str = "red",
    nlsq_color: str = "blue",
    band_alpha: float = 0.3,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str | None = None,
) -> Figure:
    """Plot Bayesian fit with posterior credible intervals.

    Integrates with PlotManager by providing a standalone plotting function
    for Bayesian posterior predictive visualization.

    Parameters
    ----------
    x_data : Float64Array
        Original x-data points
    y_data : Float64Array
        Original y-data points
    x_pred : Float64Array
        X values for predictions
    median : Float64Array
        Posterior median predictions
    lower : Float64Array
        Lower credible interval bound
    upper : Float64Array
        Upper credible interval bound
    credible_level : float
        Credible interval level (default: 0.95)
    ax : Axes | None
        Matplotlib axes (creates new if None)
    nlsq_fit : Float64Array | None
        NLSQ point estimate predictions (optional overlay)
    data_color : str
        Color for data points (default: "black")
    bayesian_color : str
        Color for Bayesian fit and band (default: "red")
    nlsq_color : str
        Color for NLSQ overlay (default: "blue")
    band_alpha : float
        Transparency for credible band (default: 0.3)
    xlabel : str
        X-axis label (default: "x")
    ylabel : str
        Y-axis label (default: "y")
    title : str | None
        Plot title (optional)

    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.scatter(x_data, y_data, color=data_color, alpha=0.6, label="Data", zorder=3)

    ax.plot(x_pred, median, color=bayesian_color, linewidth=2, label="Median", zorder=2)

    ax.fill_between(
        x_pred,
        lower,
        upper,
        color=bayesian_color,
        alpha=band_alpha,
        label=f"{credible_level:.0%} CI",
        zorder=1,
    )

    if nlsq_fit is not None:
        ax.plot(
            x_pred,
            nlsq_fit,
            color=nlsq_color,
            linestyle="--",
            linewidth=1.5,
            label="NLSQ",
            zorder=2,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_uncertainty_comparison(
    x_data: Float64Array,
    y_data: Float64Array,
    x_pred: Float64Array,
    nlsq_fit: Float64Array,
    nlsq_lower: Float64Array,
    nlsq_upper: Float64Array,
    bayesian_median: Float64Array,
    bayesian_lower: Float64Array,
    bayesian_upper: Float64Array,
    confidence_level: float = 0.95,
    *,
    ax: Axes | None = None,
    data_color: str = "black",
    nlsq_color: str = "blue",
    bayesian_color: str = "red",
    band_alpha: float = 0.25,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str | None = None,
) -> Figure:
    """Plot comparison of frequentist CI and Bayesian credible interval.

    Standalone function for PlotManager integration.

    Parameters
    ----------
    x_data : Float64Array
        Original x-data points
    y_data : Float64Array
        Original y-data points
    x_pred : Float64Array
        X values for predictions
    nlsq_fit : Float64Array
        NLSQ point estimate predictions
    nlsq_lower : Float64Array
        NLSQ lower confidence bound
    nlsq_upper : Float64Array
        NLSQ upper confidence bound
    bayesian_median : Float64Array
        Bayesian posterior median
    bayesian_lower : Float64Array
        Bayesian lower credible bound
    bayesian_upper : Float64Array
        Bayesian upper credible bound
    confidence_level : float
        Confidence/credible level (default: 0.95)
    ax : Axes | None
        Matplotlib axes (creates new if None)
    data_color : str
        Color for data points (default: "black")
    nlsq_color : str
        Color for NLSQ fit and CI (default: "blue")
    bayesian_color : str
        Color for Bayesian fit and CI (default: "red")
    band_alpha : float
        Transparency for bands (default: 0.25)
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str | None
        Plot title (optional)

    Returns
    -------
    Figure
        Matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Data points
    ax.scatter(x_data, y_data, color=data_color, alpha=0.6, label="Data", zorder=4)

    # NLSQ fit and CI
    ax.plot(x_pred, nlsq_fit, color=nlsq_color, linewidth=2, label="NLSQ fit", zorder=3)
    ax.fill_between(
        x_pred,
        nlsq_lower,
        nlsq_upper,
        color=nlsq_color,
        alpha=band_alpha,
        label=f"NLSQ {confidence_level:.0%} CI",
        zorder=1,
    )

    # Bayesian fit and CI
    ax.plot(
        x_pred,
        bayesian_median,
        color=bayesian_color,
        linewidth=2,
        linestyle="--",
        label="Bayesian median",
        zorder=3,
    )
    ax.fill_between(
        x_pred,
        bayesian_lower,
        bayesian_upper,
        color=bayesian_color,
        alpha=band_alpha,
        label=f"Bayesian {confidence_level:.0%} CI",
        zorder=2,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Frequentist vs Bayesian Uncertainty Comparison")
    ax.legend(loc="best")

    fig.tight_layout()
    return fig


def export_uncertainty_plot(
    fig: Figure,
    output_path: Path | str,
    *,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> list[Path]:
    """Export uncertainty plot to multiple formats.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to export
    output_path : Path | str
        Base output path (extension will be added)
    formats : list[str] | None
        Output formats (default: ["pdf", "png"])
    dpi : int
        Resolution for raster formats (default: 300)

    Returns
    -------
    list[Path]
        List of created file paths
    """
    if formats is None:
        formats = ["pdf", "png"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    created = []
    base = output_path.with_suffix("")

    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
        created.append(path)
        logger.info("Exported uncertainty plot to %s", path)

    return created


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
