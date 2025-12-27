"""
Analysis Module for RheoQCM (Layer 3 - Scripting Interface)

This module provides a clean public API for scripting and analysis workflows,
importing physics functions from physics.py and the model class from model.py.

The analysis module is designed for:
    - Scripting workflows (Jupyter notebooks, Python scripts)
    - Batch processing of QCM-D data
    - Programmatic access to all core functionality
    - Backward compatibility with existing scripts

Architecture:
    Layer 1 (physics.py): Pure-JAX stateless physics functions
    Layer 2 (model.py): Unified logic class with state management
    Layer 3 (analysis.py): THIS MODULE - Clean scripting interface

Examples
--------
Basic analysis workflow:

>>> from rheoQCM.core.analysis import QCMAnalyzer
>>> analyzer = QCMAnalyzer(f1=5e6)
>>> analyzer.load_data({3: -1000+100j, 5: -1700+180j})
>>> result = analyzer.analyze(nh=[3, 5, 3])
>>> print(f"drho = {result['drho']:.3e} kg/m^2")

Using legacy function names:

>>> from rheoQCM.core.analysis import sauerbreyf, sauerbreym
>>> delf = sauerbreyf(3, 1e-6)  # 3rd harmonic, 1 ug/cm^2
>>> drho = sauerbreym(3, -1000)  # Calculate mass from frequency shift

See Also
--------
rheoQCM.core.physics : Layer 1 physics calculations
rheoQCM.core.model : Layer 2 model logic
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from rheoQCM.core.jax_config import configure_jax
from rheoQCM.core.jax_config import get_jax_backend
from rheoQCM.core.jax_config import is_gpu_available

# Ensure JAX is configured
configure_jax()

# =============================================================================
# Import all physics functions from Layer 1
# =============================================================================

from rheoQCM.core.physics import (
    # Constants
    Zq,
    f1_default,
    e26,
    d26,
    g0_default,
    epsq,
    eps0,
    dq,
    C0byA,
    electrode_default,
    water_default,
    air_default,
    dlam_refh_range,
    drho_range,
    grho_refh_range,
    phi_range,
    # Sauerbrey equations
    sauerbreyf,
    sauerbreym,
    # Complex modulus calculations
    grho,
    grhostar,
    grhostar_from_refh,
    grho_from_dlam,
    calc_dlam,
    calc_lamrho,
    calc_deltarho,
    etarho,
    zstar_bulk,
    # SLA equations
    calc_delfstar_sla,
    calc_D,
    normdelfstar,
    bulk_props,
    deltarho_bulk,
    # Kotula model
    kotula_gstar,
    kotula_xi,
    # Utility functions
    find_peaks,
    interp_linear,
    interp_cubic,
    create_interp_func,
    savgol_filter,
)

# =============================================================================
# Import model class from Layer 2
# =============================================================================

from rheoQCM.core.model import (
    QCMModel,
    bulk_drho,
)


# =============================================================================
# QCMAnalyzer - High-level analysis interface
# =============================================================================


class QCMAnalyzer:
    """
    High-level analyzer class for QCM-D data analysis.

    This class provides a clean interface for scripting workflows,
    wrapping the QCMModel class with a more intuitive API.

    Parameters
    ----------
    f1 : float, optional
        Fundamental resonant frequency [Hz]. Default: 5e6 Hz.
    refh : int, optional
        Reference harmonic for calculations. Default: 3.
    calctype : {"SLA", "LL", "Voigt"}, optional
        Calculation type. Default: "SLA".

    Attributes
    ----------
    model : QCMModel
        Underlying model instance.
    results : list[dict]
        List of analysis results from previous runs.

    Examples
    --------
    >>> analyzer = QCMAnalyzer(f1=5e6, refh=3)
    >>> analyzer.load_data({3: -1000+100j, 5: -1700+180j, 7: -2400+270j})
    >>> result = analyzer.analyze(nh=[3, 5, 3])
    >>> print(result)
    """

    def __init__(
        self,
        f1: float = f1_default,
        refh: int = 3,
        calctype: str = "SLA",
    ) -> None:
        """Initialize QCMAnalyzer with specified parameters."""
        self._model = QCMModel(f1=f1, refh=refh, calctype=calctype)
        self._results: list[dict[str, Any]] = []

    @property
    def model(self) -> QCMModel:
        """Access the underlying QCMModel instance."""
        return self._model

    @property
    def results(self) -> list[dict[str, Any]]:
        """Access list of previous analysis results."""
        return self._results

    @property
    def f1(self) -> float | None:
        """Fundamental frequency [Hz]."""
        return self._model.f1

    @f1.setter
    def f1(self, value: float) -> None:
        self._model.f1 = value

    @property
    def refh(self) -> int | None:
        """Reference harmonic number."""
        return self._model.refh

    @refh.setter
    def refh(self, value: int) -> None:
        self._model.refh = value

    def load_data(
        self,
        delfstars: dict[int, complex],
        f0s: dict[int, float] | None = None,
        g0s: dict[int, float] | None = None,
    ) -> None:
        """
        Load experimental frequency shift data.

        Parameters
        ----------
        delfstars : dict[int, complex]
            Complex frequency shifts for each harmonic.
            Keys are harmonic numbers (1, 3, 5, ...).
            Values are complex: delf + 1j * delg.
        f0s : dict[int, float], optional
            Reference frequencies for each harmonic.
        g0s : dict[int, float], optional
            Reference bandwidths for each harmonic.
        """
        self._model.load_delfstars(delfstars)
        if f0s is not None or g0s is not None:
            self._model.configure(f0s=f0s, g0s=g0s)

    def load_from_file(self, filepath: str | Path) -> None:
        """
        Load experimental data from HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Path to HDF5 file containing experimental data.
        """
        self._model.load_from_hdf5(filepath)

    def analyze(
        self,
        nh: list[int],
        calctype: str | None = None,
        bulklimit: float = 0.5,
        calculate_errors: bool = False,
        store_result: bool = True,
    ) -> dict[str, Any]:
        """
        Analyze loaded data to extract film properties.

        Parameters
        ----------
        nh : list[int]
            Harmonics for calculation [n1, n2, n3].
        calctype : str, optional
            Calculation type: "SLA" or "LL". Uses model default if None.
        bulklimit : float, optional
            Dissipation ratio threshold for bulk vs thin film. Default: 0.5.
        calculate_errors : bool, optional
            Whether to calculate errors from Jacobian. Default: False.
        store_result : bool, optional
            Whether to store result in results list. Default: True.

        Returns
        -------
        dict
            Results containing:
            - grho_refh: |G*|*rho at reference harmonic [Pa kg/m^3]
            - phi: Phase angle [radians]
            - drho: Mass per area [kg/m^2]
            - dlam_refh: d/lambda at reference harmonic
            - errors: dict of error estimates (if calculate_errors=True)
        """
        result = self._model.solve_properties(
            nh=nh,
            calctype=calctype,
            bulklimit=bulklimit,
            calculate_errors=calculate_errors,
        )

        if store_result:
            self._results.append(result)

        return result

    def analyze_batch(
        self,
        batch_delfstars: list[dict[int, complex]],
        nh: list[int],
        calctype: str | None = None,
        bulklimit: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Analyze multiple timepoints in batch.

        Parameters
        ----------
        batch_delfstars : list[dict[int, complex]]
            List of delfstar dictionaries for each timepoint.
        nh : list[int]
            Harmonics for calculation.
        calctype : str, optional
            Calculation type.
        bulklimit : float, optional
            Dissipation ratio threshold.

        Returns
        -------
        list[dict]
            List of result dictionaries.
        """
        results = self._model.solve_batch(
            batch_delfstars=batch_delfstars,
            nh=nh,
            calctype=calctype,
            bulklimit=bulklimit,
        )
        self._results.extend(results)
        return results

    def curve_fit(
        self,
        f: Callable,
        xdata: np.ndarray | jnp.ndarray,
        ydata: np.ndarray | jnp.ndarray,
        p0: np.ndarray | list | None = None,
        sigma: np.ndarray | None = None,
        absolute_sigma: bool = False,
        bounds: tuple = (-np.inf, np.inf),
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Curve fitting using NLSQ library.

        Parameters
        ----------
        f : callable
            Model function f(x, *params) -> y.
        xdata : array
            Independent variable data.
        ydata : array
            Dependent variable data.
        p0 : array, optional
            Initial parameter guess.
        sigma : array, optional
            Uncertainties in ydata.
        absolute_sigma : bool, optional
            Whether sigma is absolute.
        bounds : tuple, optional
            Parameter bounds.
        **kwargs
            Additional arguments passed to NLSQ curve_fit.

        Returns
        -------
        popt : array
            Fitted parameters.
        pcov : array
            Parameter covariance matrix.
        """
        return self._model.curve_fit(
            f=f,
            xdata=xdata,
            ydata=ydata,
            p0=p0,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            bounds=bounds,
            **kwargs,
        )

    def bayesian_fit(
        self,
        nh: list[int],
        initial_params: dict[str, Any] | None = None,
        num_samples: int = 1000,
        num_warmup: int = 500,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Perform Bayesian inference using NumPyro NUTS sampler.

        Parameters
        ----------
        nh : list[int]
            Harmonics for calculation.
        initial_params : dict, optional
            Initial parameter estimates (from NLSQ fit).
        num_samples : int, optional
            Number of MCMC samples. Default: 1000.
        num_warmup : int, optional
            Number of warmup samples. Default: 500.
        **kwargs
            Additional arguments for NUTS sampler.

        Returns
        -------
        dict
            Posterior samples and summary statistics.
        """
        return self._model.bayesian_fit(
            nh=nh,
            initial_params=initial_params,
            num_samples=num_samples,
            num_warmup=num_warmup,
            **kwargs,
        )

    def format_result(
        self, result: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Format result for export.

        Parameters
        ----------
        result : dict, optional
            Result to format. Uses last result if None.

        Returns
        -------
        dict
            Formatted result with expected keys.
        """
        if result is None:
            if not self._results:
                raise ValueError("No results available")
            result = self._results[-1]
        return self._model.format_result_for_export(result)

    def convert_to_display_units(
        self, result: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Convert result units from SI to display units.

        Parameters
        ----------
        result : dict, optional
            Result to convert. Uses last result if None.

        Returns
        -------
        dict
            Result with display units.
        """
        if result is None:
            if not self._results:
                raise ValueError("No results available")
            result = self._results[-1]
        return self._model.convert_units_for_display(result)

    def clear_results(self) -> None:
        """Clear the stored results list."""
        self._results = []


# =============================================================================
# Convenience functions for common operations
# =============================================================================


def analyze_delfstar(
    delfstars: dict[int, complex],
    nh: list[int] | None = None,
    f1: float = f1_default,
    refh: int = 3,
    calctype: str = "SLA",
    bulklimit: float = 0.5,
) -> dict[str, Any]:
    """
    One-shot analysis of delfstar data.

    Convenience function for quick analysis without creating an analyzer.

    Parameters
    ----------
    delfstars : dict[int, complex]
        Complex frequency shifts for each harmonic.
    nh : list[int], optional
        Harmonics for calculation. Default: first 3 available harmonics.
    f1 : float, optional
        Fundamental frequency [Hz]. Default: 5e6 Hz.
    refh : int, optional
        Reference harmonic. Default: 3.
    calctype : str, optional
        Calculation type. Default: "SLA".
    bulklimit : float, optional
        Dissipation ratio threshold. Default: 0.5.

    Returns
    -------
    dict
        Analysis results.

    Examples
    --------
    >>> result = analyze_delfstar(
    ...     {3: -1000+100j, 5: -1700+180j, 7: -2400+270j},
    ...     nh=[3, 5, 3],
    ... )
    """
    analyzer = QCMAnalyzer(f1=f1, refh=refh, calctype=calctype)
    analyzer.load_data(delfstars)

    if nh is None:
        harmonics = sorted(delfstars.keys())
        if len(harmonics) >= 3:
            nh = [harmonics[0], harmonics[1], harmonics[0]]
        else:
            raise ValueError("Need at least 3 harmonics for analysis")

    return analyzer.analyze(nh=nh, bulklimit=bulklimit)


def batch_analyze(
    batch_delfstars: list[dict[int, complex]],
    nh: list[int],
    f1: float = f1_default,
    refh: int = 3,
    calctype: str = "SLA",
    bulklimit: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Analyze multiple timepoints in batch.

    Convenience function for batch processing.

    Parameters
    ----------
    batch_delfstars : list[dict[int, complex]]
        List of delfstar dictionaries for each timepoint.
    nh : list[int]
        Harmonics for calculation.
    f1 : float, optional
        Fundamental frequency [Hz].
    refh : int, optional
        Reference harmonic.
    calctype : str, optional
        Calculation type.
    bulklimit : float, optional
        Dissipation ratio threshold.

    Returns
    -------
    list[dict]
        List of result dictionaries.
    """
    analyzer = QCMAnalyzer(f1=f1, refh=refh, calctype=calctype)
    return analyzer.analyze_batch(
        batch_delfstars=batch_delfstars,
        nh=nh,
        calctype=calctype,
        bulklimit=bulklimit,
    )


# =============================================================================
# Backward compatibility aliases
# =============================================================================

# Deprecated function names - kept for backward compatibility
# These will raise DeprecationWarning when used


def _deprecated_alias(name: str, new_name: str, func: Callable) -> Callable:
    """Create a deprecated alias for a function."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"{name} is deprecated, use {new_name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__doc__ = f"Deprecated alias for {new_name}. Use {new_name} instead."
    return wrapper


# Legacy function aliases (phi in degrees instead of radians in some cases)
# These are provided for scripts that use the old QCM_functions.py conventions


def grho_legacy(n: int, props: dict[str, Any]) -> float:
    """
    Legacy grho function compatible with QCM_functions.py.

    DEPRECATED: Use rheoQCM.core.physics.grho with radians instead.

    Parameters
    ----------
    n : int
        Harmonic number.
    props : dict
        Dictionary with 'grho3' (in Pa g/cm^3) and 'phi' (in degrees).

    Returns
    -------
    float
        |G*|*rho at harmonic n in Pa g/cm^3.
    """
    warnings.warn(
        "grho_legacy is deprecated. Use rheoQCM.core.physics.grho with radians.",
        DeprecationWarning,
        stacklevel=2,
    )
    grho3 = props["grho3"]
    phi_deg = props["phi"]
    # Legacy: phi in degrees, grho3 in Pa g/cm^3
    return float(grho3 * (n / 3) ** (phi_deg / 90))


def grhostar_legacy(grho_val: float, phi_deg: float) -> complex:
    """
    Legacy grhostar function compatible with QCM_functions.py.

    DEPRECATED: Use rheoQCM.core.physics.grhostar with radians instead.

    Parameters
    ----------
    grho_val : float
        |G*|*rho magnitude.
    phi_deg : float
        Phase angle in degrees.

    Returns
    -------
    complex
        Complex G*rho.
    """
    warnings.warn(
        "grhostar_legacy is deprecated. Use rheoQCM.core.physics.grhostar with radians.",
        DeprecationWarning,
        stacklevel=2,
    )
    phi_rad = np.radians(phi_deg)
    return complex(grhostar(grho_val, phi_rad))


def calc_drho_from_delf(
    n: int | Sequence[int],
    delf: float | Sequence[float],
    f1: float = f1_default,
) -> np.ndarray:
    """
    Calculate mass per unit area from frequency shift.

    Alias for sauerbreym for backward compatibility.

    Parameters
    ----------
    n : int or array
        Harmonic number(s).
    delf : float or array
        Frequency shift(s) [Hz].
    f1 : float, optional
        Fundamental frequency [Hz].

    Returns
    -------
    drho : array
        Mass per unit area [kg/m^2].
    """
    return np.asarray(sauerbreym(n, delf, f1=f1))


def calc_delf_from_drho(
    n: int | Sequence[int],
    drho: float | Sequence[float],
    f1: float = f1_default,
) -> np.ndarray:
    """
    Calculate frequency shift from mass per unit area.

    Alias for sauerbreyf for backward compatibility.

    Parameters
    ----------
    n : int or array
        Harmonic number(s).
    drho : float or array
        Mass per unit area [kg/m^2].
    f1 : float, optional
        Fundamental frequency [Hz].

    Returns
    -------
    delf : array
        Frequency shift [Hz].
    """
    return np.asarray(sauerbreyf(n, drho, f1=f1))


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # High-level API
    "QCMAnalyzer",
    "analyze_delfstar",
    "batch_analyze",
    # JAX configuration
    "configure_jax",
    "get_jax_backend",
    "is_gpu_available",
    # Model class
    "QCMModel",
    "bulk_drho",
    # Constants (from physics.py)
    "Zq",
    "f1_default",
    "e26",
    "d26",
    "g0_default",
    "epsq",
    "eps0",
    "dq",
    "C0byA",
    "electrode_default",
    "water_default",
    "air_default",
    "dlam_refh_range",
    "drho_range",
    "grho_refh_range",
    "phi_range",
    # Sauerbrey equations
    "sauerbreyf",
    "sauerbreym",
    # Complex modulus calculations
    "grho",
    "grhostar",
    "grhostar_from_refh",
    "grho_from_dlam",
    "calc_dlam",
    "calc_lamrho",
    "calc_deltarho",
    "etarho",
    "zstar_bulk",
    # SLA equations
    "calc_delfstar_sla",
    "calc_D",
    "normdelfstar",
    "bulk_props",
    "deltarho_bulk",
    # Kotula model
    "kotula_gstar",
    "kotula_xi",
    # Utility functions
    "find_peaks",
    "interp_linear",
    "interp_cubic",
    "create_interp_func",
    "savgol_filter",
    # Backward compatibility aliases
    "grho_legacy",
    "grhostar_legacy",
    "calc_drho_from_delf",
    "calc_delf_from_drho",
]
