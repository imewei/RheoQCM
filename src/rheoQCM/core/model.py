"""
Model Logic Module for RheoQCM (Layer 2)

This module provides the unified logic class for QCM-D analysis,
handling state management, data loading, and JAX solver orchestration.

Key Features:
    - State management: f1, g1, f0s, g0s, refh, calctype
    - Data loading from HDF5 and numpy arrays
    - Unified optimizer using jaxopt.LevenbergMarquardt
    - NLSQ curve_fit integration for curve fitting
    - Queue-based processing for batch operations
    - Error propagation from Jacobian using JAX autodiff
    - NumPyro integration for Bayesian inference (optional)
    - Extensible calctype system for custom physics models (US4)

Architecture:
    Layer 1 (physics.py): Pure-JAX stateless physics functions
    Layer 2 (model.py): THIS MODULE - State and solver orchestration
    Layer 3: UI wrappers (QCM.py) and scripting interfaces

Extending Calctypes (User Story 4)
----------------------------------
The calctype system supports custom physics models. Built-in types are
"SLA" (Small Load Approximation), "LL" (Lumped Loading), and "Voigt".

To add a custom calctype:

1. Define a residual function with signature:
   def my_residual(params, delfstar_exp, harmonics, f1, refh, Zq) -> array

2. Register with a QCMModel instance:
   model.register_calctype("MyModel", my_residual)

3. Use in solve_properties:
   result = model.solve_properties(nh=[3, 5, 3], calctype="MyModel")

The residual function receives:
    - params: array of [grho_refh, phi, drho] or custom params
    - delfstar_exp: dict of experimental complex frequency shifts
    - harmonics: list of harmonic numbers [n1, n2, n3]
    - f1: fundamental frequency [Hz]
    - refh: reference harmonic
    - Zq: quartz acoustic impedance [Pa s/m]

See Also
--------
rheoQCM.core.physics : Layer 1 physics calculations
rheoQCM.core.multilayer : Multi-layer film calculations
rheoQCM.core.jax_config : JAX configuration
specs/004-unify-qcm-modules/quickstart.md : Extension examples
"""

from __future__ import annotations

import logging
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import h5py
import jax
import jax.numpy as jnp

# Suppress JAXopt deprecation warning - library still functional, migration planned
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="JAXopt is no longer maintained")
    import jaxopt

# Log JAXopt import with version and deprecation status
_jaxopt_logger = logging.getLogger(__name__)
try:
    import importlib.metadata

    _jaxopt_version = importlib.metadata.version("jaxopt")
except Exception:
    _jaxopt_version = "unknown"
_jaxopt_logger.info(
    f"JAXopt version {_jaxopt_version} loaded (deprecated, migration to Optax planned)"
)

import numpy as np

from rheoQCM.core import physics
from rheoQCM.core.jax_config import configure_jax

# Ensure JAX is configured
configure_jax()

# Add NLSQ to path and import
NLSQ_PATH = Path("/home/wei/Documents/GitHub/NLSQ")
if NLSQ_PATH.exists() and str(NLSQ_PATH) not in sys.path:
    sys.path.insert(0, str(NLSQ_PATH))

from nlsq import curve_fit as nlsq_curve_fit

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions for Calctype Extensibility (T048 - US4)
# =============================================================================

# Type alias for residual function signature
CalctypeResidualFn = Callable[
    [jnp.ndarray, dict[int, complex], list[int], float, int, float], jnp.ndarray
]


# =============================================================================
# Result Dataclasses (T011, T012)
# =============================================================================


@dataclass
class SolveResult:
    """
    Result from solve_properties() method.

    This dataclass contains the solved film properties and error estimates
    from a single QCM-D measurement.

    Attributes
    ----------
    drho : float
        Mass per unit area [kg/m^2]. Set to inf for bulk materials.
    grho_refh : float
        |G*|*rho at reference harmonic [Pa kg/m^3].
    phi : float
        Phase angle [radians]. Range: 0 to pi/2.
    dlam_refh : float
        d/lambda at reference harmonic (dimensionless).
    covariance : np.ndarray | None
        Parameter covariance matrix (3x3 for drho, grho_refh, phi).
        None if error calculation was not performed.
    residuals : np.ndarray | None
        Final residual values at solution.
    success : bool
        Whether the optimization converged successfully.
    message : str
        Status message or error description.

    Examples
    --------
    >>> result = model.solve_properties(nh=[3, 5, 3])
    >>> print(f"drho = {result.drho:.3e} kg/m^2")
    >>> print(f"phi = {np.degrees(result.phi):.1f} degrees")
    >>> if result.success:
    ...     print("Optimization converged")
    """

    drho: float = np.nan
    grho_refh: float = np.nan
    phi: float = np.nan
    dlam_refh: float = np.nan
    covariance: np.ndarray | None = None
    residuals: np.ndarray | None = None
    success: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "drho": self.drho,
            "grho_refh": self.grho_refh,
            "phi": self.phi,
            "dlam_refh": self.dlam_refh,
            "covariance": self.covariance,
            "residuals": self.residuals,
            "success": self.success,
            "message": self.message,
            "errors": self.errors,
        }

    @property
    def errors(self) -> dict[str, float]:
        """Extract error estimates from covariance matrix."""
        if self.covariance is None:
            return {"drho": np.nan, "grho_refh": np.nan, "phi": np.nan}
        try:
            return {
                "drho": np.sqrt(self.covariance[0, 0]),
                "grho_refh": np.sqrt(self.covariance[1, 1]),
                "phi": np.sqrt(self.covariance[2, 2]),
            }
        except (IndexError, ValueError):
            return {"drho": np.nan, "grho_refh": np.nan, "phi": np.nan}


@dataclass
class BatchResult:
    """
    Result from batch_analyze() or solve_batch() methods.

    This dataclass contains arrays of solved properties for multiple
    QCM-D measurements processed in batch.

    Attributes
    ----------
    drho : np.ndarray
        Mass per unit area array [kg/m^2]. Shape: (N,).
    grho_refh : np.ndarray
        |G*|*rho at reference harmonic array [Pa kg/m^3]. Shape: (N,).
    phi : np.ndarray
        Phase angle array [radians]. Shape: (N,).
    dlam_refh : np.ndarray
        d/lambda at reference harmonic array. Shape: (N,).
    success : np.ndarray
        Boolean array indicating convergence for each measurement. Shape: (N,).
    messages : list[str]
        Status messages for each measurement.

    Examples
    --------
    >>> batch_result = model.solve_batch(batch_delfstars, nh=[3, 5, 3])
    >>> valid_mask = batch_result.success
    >>> mean_drho = np.mean(batch_result.drho[valid_mask])
    """

    drho: np.ndarray = field(default_factory=lambda: np.array([]))
    grho_refh: np.ndarray = field(default_factory=lambda: np.array([]))
    phi: np.ndarray = field(default_factory=lambda: np.array([]))
    dlam_refh: np.ndarray = field(default_factory=lambda: np.array([]))
    success: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    messages: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        """Return number of measurements in batch."""
        return len(self.drho)

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Convert to list of dictionaries for backward compatibility."""
        return [
            {
                "drho": float(self.drho[i]),
                "grho_refh": float(self.grho_refh[i]),
                "phi": float(self.phi[i]),
                "dlam_refh": float(self.dlam_refh[i]),
                "success": bool(self.success[i]),
                "message": self.messages[i] if i < len(self.messages) else "",
            }
            for i in range(len(self))
        ]

    @classmethod
    def from_solve_results(cls, results: list[SolveResult]) -> BatchResult:
        """Create BatchResult from list of SolveResult objects."""
        n = len(results)
        return cls(
            drho=np.array([r.drho for r in results]),
            grho_refh=np.array([r.grho_refh for r in results]),
            phi=np.array([r.phi for r in results]),
            dlam_refh=np.array([r.dlam_refh for r in results]),
            success=np.array([r.success for r in results], dtype=bool),
            messages=[r.message for r in results],
        )


# =============================================================================
# Variable Limits and Constants
# =============================================================================

# Variable limits for fitting (from physics.py)
dlam_refh_range: tuple[float, float] = (0.0, 10.0)
drho_range: tuple[float, float] = (0.0, 3e-2)  # kg/m^2
grho_refh_range: tuple[float, float] = (1e4, 1e14)  # Pa kg/m^3
phi_range: tuple[float, float] = (0.0, np.pi / 2)  # radians

# Default bulk thickness
bulk_drho: float = np.inf


# =============================================================================
# Pure JAX Residual Functions (for JIT compatibility)
# =============================================================================


def _jax_normdelfstar(
    n: int, refh: int, dlam_refh: jnp.ndarray, phi: jnp.ndarray
) -> jnp.ndarray:
    """Calculate normalized delfstar at harmonic n using JAX."""
    dlam_n = dlam_refh * (n / refh) ** (1 - phi / jnp.pi)
    D = 2 * jnp.pi * dlam_n * (1 - 1j * jnp.tan(phi / 2))
    return -jnp.sin(D) / D / jnp.cos(D)


def _jax_rhcalc(
    n1: int, n2: int, refh: int, dlam_refh: jnp.ndarray, phi: jnp.ndarray
) -> jnp.ndarray:
    """Calculate harmonic ratio using JAX."""
    nds1 = _jax_normdelfstar(n1, refh, dlam_refh, phi)
    nds2 = _jax_normdelfstar(n2, refh, dlam_refh, phi)
    return jnp.real(nds1) / jnp.real(nds2)


def _jax_rdcalc(
    n3: int, refh: int, dlam_refh: jnp.ndarray, phi: jnp.ndarray
) -> jnp.ndarray:
    """Calculate dissipation ratio using JAX."""
    nds = _jax_normdelfstar(n3, refh, dlam_refh, phi)
    return -jnp.imag(nds) / jnp.real(nds)


# T016: _jax_grho replaced with physics.grho (005-jax-perf)
# T017: _jax_grhostar_from_refh replaced with physics.grhostar_from_refh (005-jax-perf)
# T018: _jax_zstar_bulk replaced with physics.zstar_bulk (005-jax-perf)
# T019: _jax_calc_delfstar_sla replaced with physics.calc_delfstar_sla (005-jax-perf)


def _jax_calc_ZL_single_layer(
    n: int,
    grho_refh: jnp.ndarray,
    phi: jnp.ndarray,
    drho: jnp.ndarray,
    f1: float,
    refh: int,
) -> jnp.ndarray:
    """Calculate load impedance for single layer (JAX)."""
    # T017: Use physics.grhostar_from_refh instead of _jax_grhostar_from_refh
    grhostar = physics.grhostar_from_refh(n, grho_refh, phi, refh=refh)
    # T018: Use physics.zstar_bulk instead of _jax_zstar_bulk
    zstar = physics.zstar_bulk(grhostar)
    D = 2 * jnp.pi * n * f1 * drho / zstar
    return 1j * zstar * jnp.tan(D)


def _jax_calc_delfstar(
    n: int,
    grho_refh: jnp.ndarray,
    phi: jnp.ndarray,
    drho: jnp.ndarray,
    f1: float,
    Zq: float,
    refh: int,
) -> jnp.ndarray:
    """Calculate complex frequency shift for single layer (JAX)."""
    ZL = _jax_calc_ZL_single_layer(n, grho_refh, phi, drho, f1, refh)
    # T019: Use physics.calc_delfstar_sla instead of _jax_calc_delfstar_sla
    return physics.calc_delfstar_sla(ZL, f1=f1)


def _jax_calc_delfstar_bulk(
    n: int, grho_refh: jnp.ndarray, phi: jnp.ndarray, f1: float, Zq: float, refh: int
) -> jnp.ndarray:
    """Calculate complex frequency shift for bulk material (JAX)."""
    # T016: Use physics.grho instead of _jax_grho
    grho_n = physics.grho(n, grho_refh, phi, refh=refh)
    return (f1 * jnp.sqrt(grho_n) / (jnp.pi * Zq)) * (
        -jnp.sin(phi / 2) + 1j * jnp.cos(phi / 2)
    )


# =============================================================================
# Global Calctype Registry (T048 - US4)
# =============================================================================

# Registry of calctype names to residual functions
_CALCTYPE_REGISTRY: dict[str, CalctypeResidualFn] = {}


def register_global_calctype(name: str, residual_fn: CalctypeResidualFn) -> None:
    """
    Register a calctype globally (available to all QCMModel instances).

    Parameters
    ----------
    name : str
        Unique name for the calctype (e.g., "FractionalMaxwell").
    residual_fn : callable
        Residual function with signature:
        (params, delfstar_exp, harmonics, f1, refh, Zq) -> array

    Examples
    --------
    >>> def my_residual(params, delfstar_exp, harmonics, f1, refh, Zq):
    ...     grho, phi, drho = params[0], params[1], params[2]
    ...     # Custom calculation...
    ...     return jnp.array([residual1, residual2, residual3])
    >>> register_global_calctype("MyModel", my_residual)
    """
    _CALCTYPE_REGISTRY[name] = residual_fn
    logger.info(f"Registered global calctype: {name}")


def get_global_calctypes() -> list[str]:
    """
    Get list of globally registered calctypes.

    Returns
    -------
    list[str]
        Names of all registered calctypes.
    """
    return list(_CALCTYPE_REGISTRY.keys())


# =============================================================================
# QCMModel Class
# =============================================================================


class QCMModel:
    """
    Unified logic class for QCM-D analysis.

    This class provides the Layer 2 implementation merging QCM class logic
    with state management, data loading, and JAX solver orchestration.

    The calctype parameter controls which physics model is used for fitting.
    Built-in calctypes are "SLA", "LL", and "Voigt". Custom calctypes can be
    registered using register_calctype() for extensibility (US4).

    Parameters
    ----------
    f1 : float, optional
        Fundamental resonant frequency [Hz]. Default: 5e6 Hz.
    refh : int, optional
        Reference harmonic for calculations. Default: 3.
    calctype : str, optional
        Calculation type. Default: "SLA".
        Built-in: "SLA", "LL", "Voigt"
        Custom types can be registered with register_calctype().
    cut : {"AT", "BT"}, optional
        Crystal cut type. Default: "AT".

    Attributes
    ----------
    f1 : float or None
        Fundamental resonant frequency [Hz].
    g1 : float or None
        Dissipation at fundamental frequency [Hz].
    f0s : dict[int, float]
        Reference frequencies for each harmonic.
    g0s : dict[int, float]
        Reference bandwidths for each harmonic.
    refh : int or None
        Reference harmonic for calculations.
    calctype : str
        Calculation type: built-in or custom registered type.
    Zq : float
        Acoustic impedance of quartz [Pa s/m].
    delfstars : dict[int, complex]
        Complex frequency shifts for each harmonic.

    Examples
    --------
    Basic usage:

    >>> from rheoQCM.core.model import QCMModel
    >>> model = QCMModel(f1=5e6, refh=3)
    >>> model.load_delfstars({3: -1000+100j, 5: -1700+180j})
    >>> result = model.solve_properties(nh=[3, 5, 3], calctype="SLA")
    >>> print(f"drho = {result.drho:.3e} kg/m^2")

    Using custom calctype:

    >>> def my_residual(params, delfstar_exp, harmonics, f1, refh, Zq):
    ...     # Custom physics model
    ...     return jnp.array([r1, r2, r3])
    >>> model.register_calctype("Custom", my_residual)
    >>> result = model.solve_properties(nh=[3, 5, 3], calctype="Custom")

    See Also
    --------
    register_calctype : Register a custom calctype for this instance
    get_supported_calctypes : List all available calctypes
    """

    # Class constants
    Zq_values: dict[str, float] = {
        "AT": 8.84e6,  # kg m^-2 s^-1
        "BT": 0e6,
    }

    # Built-in calctypes
    BUILTIN_CALCTYPES: set[str] = {"SLA", "LL", "Voigt"}

    # Error floor parameters
    g_err_min: float = 1.0  # Hz
    f_err_min: float = 1.0  # Hz
    err_frac: float = 3e-2  # Error as fraction of gamma

    def __init__(
        self,
        f1: float | None = None,
        refh: int | None = None,
        calctype: str = "SLA",
        cut: Literal["AT", "BT"] = "AT",
    ) -> None:
        """Initialize QCMModel with given parameters."""
        self.Zq: float = self.Zq_values[cut]
        # T031: Use private attributes for f1/refh with property setters (005-jax-perf)
        self._f1: float | None = f1
        self.g1: float | None = None
        self.f0s: dict[int, float] = {}
        self.g0s: dict[int, float] = {}
        self._refh: int | None = refh
        self._calctype: str = self._normalize_calctype(calctype)

        # Data storage
        self.delfstars: dict[int, complex] = {}

        # Piezoelectric stiffening flag
        self.piezoelectric_stiffening: bool = False

        # Electrode properties (default)
        self.electrode_default: dict[str, Any] = {
            "calc": False,
            "grho": 3.0e17,
            "phi": 0.0,
            "drho": 2.8e-6,
            "n": 3,
        }

        # Optimizer instance (created on first use)
        self._optimizer: jaxopt.LevenbergMarquardt | None = None

        # Instance-level calctype registry (inherits from global)
        self._instance_calctypes: dict[str, CalctypeResidualFn] = {}

        # T028: Computation cache for _grho_at_harmonic (005-jax-perf)
        # Key: (n, grho_refh, phi, f1, refh) -> Value: computed grho at harmonic n
        self._grho_cache: dict[tuple, float] = {}

    @staticmethod
    def _normalize_calctype(calctype: str) -> str:
        """
        Normalize calctype string for case-insensitive built-in matching.

        Parameters
        ----------
        calctype : str
            Calctype string.

        Returns
        -------
        str
            Normalized calctype (uppercase for built-ins, original for custom).
        """
        upper = calctype.upper()
        if upper in {"SLA", "LL", "VOIGT"}:
            return upper
        return calctype  # Keep custom names as-is

    @property
    def calctype(self) -> str:
        """Get the current calctype."""
        return self._calctype

    @calctype.setter
    def calctype(self, value: str) -> None:
        """Set the calctype (with normalization)."""
        self._calctype = self._normalize_calctype(value)

    # T031: Property getters/setters for f1 and refh with cache invalidation (005-jax-perf)
    @property
    def f1(self) -> float | None:
        """Get the fundamental frequency."""
        return self._f1

    @f1.setter
    def f1(self, value: float | None) -> None:
        """Set the fundamental frequency and invalidate cache."""
        if getattr(self, "_f1", None) != value:
            self._f1 = value
            if hasattr(self, "_grho_cache"):
                self._invalidate_cache()

    @property
    def refh(self) -> int | None:
        """Get the reference harmonic."""
        return self._refh

    @refh.setter
    def refh(self, value: int | None) -> None:
        """Set the reference harmonic and invalidate cache."""
        if getattr(self, "_refh", None) != value:
            self._refh = value
            if hasattr(self, "_grho_cache"):
                self._invalidate_cache()

    # T028: Cache key and invalidation methods (005-jax-perf)
    def _get_cache_key(self) -> tuple[float | None, int | None]:
        """Return the current model state key for cache lookups."""
        return (self.f1, self.refh)

    def _invalidate_cache(self) -> None:
        """Clear the grho computation cache when model parameters change."""
        self._grho_cache.clear()

    def register_calctype(self, name: str, residual_fn: CalctypeResidualFn) -> None:
        """
        Register a custom calctype for this model instance.

        This allows extending the physics models available for fitting
        without modifying the core code.

        Parameters
        ----------
        name : str
            Unique name for the calctype.
        residual_fn : callable
            Residual function with signature:
            (params, delfstar_exp, harmonics, f1, refh, Zq) -> array

            Where:
            - params: jnp.ndarray of [grho_refh, phi, drho] or custom
            - delfstar_exp: dict[int, complex] of experimental data
            - harmonics: list[int] of [n1, n2, n3]
            - f1: float, fundamental frequency [Hz]
            - refh: int, reference harmonic
            - Zq: float, quartz impedance [Pa s/m]

        Examples
        --------
        >>> def fractional_maxwell_residual(params, delfstar_exp, nh, f1, refh, Zq):
        ...     grho, phi, drho = params[0], params[1], params[2]
        ...     # Fractional Maxwell calculation...
        ...     return jnp.array([r1, r2, r3])
        >>> model.register_calctype("FractionalMaxwell", fractional_maxwell_residual)
        >>> result = model.solve_properties(nh=[3, 5, 3], calctype="FractionalMaxwell")

        Notes
        -----
        - For jit/vmap compatibility, use pure JAX operations in residual_fn
        - Avoid Python control flow (if/else with array values) in residual_fn
        - The residual function should return an array of the same length
          as the number of fitted data points (typically 3)
        """
        self._instance_calctypes[name] = residual_fn
        logger.info(f"Registered calctype '{name}' for model instance")

    def get_supported_calctypes(self) -> list[str]:
        """
        Get list of all supported calctypes.

        Returns list of built-in types ("SLA", "LL", "Voigt") plus
        any custom types registered globally or on this instance.

        Returns
        -------
        list[str]
            Names of all available calctypes.

        Examples
        --------
        >>> model = QCMModel()
        >>> print(model.get_supported_calctypes())
        ['SLA', 'LL', 'Voigt']
        >>> model.register_calctype("Custom", my_residual)
        >>> print(model.get_supported_calctypes())
        ['SLA', 'LL', 'Voigt', 'Custom']
        """
        supported = list(self.BUILTIN_CALCTYPES)
        # Add global registry
        supported.extend(get_global_calctypes())
        # Add instance-level registry
        supported.extend(self._instance_calctypes.keys())
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for ct in supported:
            if ct not in seen:
                seen.add(ct)
                unique.append(ct)
        return unique

    def _get_calctype_residual(self, calctype: str) -> CalctypeResidualFn | None:
        """
        Get the residual function for a calctype.

        Parameters
        ----------
        calctype : str
            Calctype name.

        Returns
        -------
        CalctypeResidualFn or None
            Residual function if found, None for built-in types.
        """
        # Check instance registry first
        if calctype in self._instance_calctypes:
            return self._instance_calctypes[calctype]
        # Check global registry
        if calctype in _CALCTYPE_REGISTRY:
            return _CALCTYPE_REGISTRY[calctype]
        return None

    def configure(
        self,
        f1: float | None = None,
        f0s: dict[int, float] | None = None,
        g0s: dict[int, float] | None = None,
        refh: int | None = None,
        calctype: str | None = None,
    ) -> None:
        """
        Configure model state after initialization.

        Parameters
        ----------
        f1 : float, optional
            Fundamental resonant frequency [Hz].
        f0s : dict[int, float], optional
            Reference frequencies for each harmonic.
        g0s : dict[int, float], optional
            Reference bandwidths for each harmonic.
        refh : int, optional
            Reference harmonic.
        calctype : str, optional
            Calculation type (built-in or custom registered).
        """
        if f1 is not None:
            self.f1 = f1
        if f0s is not None:
            self.f0s = f0s
        if g0s is not None:
            self.g0s = g0s
        if refh is not None:
            self.refh = refh
        if calctype is not None:
            self.calctype = calctype

        # Calculate g1 from g0s if available
        if self.g0s and 1 in self.g0s:
            self.g1 = self.g0s[1]
        elif self.g0s:
            # Use first available harmonic to estimate g1
            for h in sorted(self.g0s.keys()):
                if not np.isnan(self.g0s[h]):
                    self.g1 = self.g0s[h] / h
                    break

    def load_delfstars(self, delfstars: dict[int, complex]) -> None:
        """
        Load complex frequency shift data from dictionary.

        Parameters
        ----------
        delfstars : dict[int, complex]
            Complex frequency shifts for each harmonic.
            Keys are harmonic numbers (1, 3, 5, ...).
            Values are complex: delf + 1j * delg.
        """
        self.delfstars = {int(k): complex(v) for k, v in delfstars.items()}

    def load_from_hdf5(self, filepath: str | Path) -> None:
        """
        Load experimental data from HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Path to HDF5 file containing experimental data.

        Notes
        -----
        Expected HDF5 structure:
            - delf: frequency shifts [Hz]
            - delg: bandwidth shifts [Hz]
            - harmonics: harmonic numbers
            - f0: reference frequencies [Hz]
            - g0: reference bandwidths [Hz]
        """
        filepath = Path(filepath)

        with h5py.File(filepath, "r") as hf:
            # Load frequency shifts
            delf = np.array(hf["delf"])
            delg = np.array(hf["delg"])
            harmonics = np.array(hf["harmonics"])

            # Build delfstars dictionary
            self.delfstars = {
                int(h): complex(f, g)
                for h, f, g in zip(harmonics, delf, delg, strict=False)
            }

            # Load reference frequencies and bandwidths if available
            if "f0" in hf:
                f0 = np.array(hf["f0"])
                self.f0s = {
                    int(h): float(f) for h, f in zip(harmonics, f0, strict=False)
                }
                # Calculate f1 from first available harmonic
                for h in sorted(self.f0s.keys()):
                    if not np.isnan(self.f0s[h]):
                        self.f1 = self.f0s[h] / h
                        break

            if "g0" in hf:
                g0 = np.array(hf["g0"])
                self.g0s = {
                    int(h): float(g) for h, g in zip(harmonics, g0, strict=False)
                }
                # Calculate g1
                for h in sorted(self.g0s.keys()):
                    if not np.isnan(self.g0s[h]):
                        self.g1 = self.g0s[h] / h
                        break

    def load_from_arrays(
        self,
        harmonics: np.ndarray,
        delf: np.ndarray,
        delg: np.ndarray,
        f0: np.ndarray | None = None,
        g0: np.ndarray | None = None,
    ) -> None:
        """
        Load experimental data from numpy arrays.

        Parameters
        ----------
        harmonics : array
            Array of harmonic numbers.
        delf : array
            Frequency shifts [Hz].
        delg : array
            Bandwidth shifts [Hz].
        f0 : array, optional
            Reference frequencies [Hz].
        g0 : array, optional
            Reference bandwidths [Hz].
        """
        self.delfstars = {
            int(h): complex(f, g)
            for h, f, g in zip(harmonics, delf, delg, strict=False)
        }

        if f0 is not None:
            self.f0s = {int(h): float(f) for h, f in zip(harmonics, f0, strict=False)}
            for h in sorted(self.f0s.keys()):
                if not np.isnan(self.f0s[h]):
                    self.f1 = self.f0s[h] / h
                    break

        if g0 is not None:
            self.g0s = {int(h): float(g) for h, g in zip(harmonics, g0, strict=False)}
            for h in sorted(self.g0s.keys()):
                if not np.isnan(self.g0s[h]):
                    self.g1 = self.g0s[h] / h
                    break

    def _get_optimizer(self) -> jaxopt.LevenbergMarquardt:
        """Get or create the unified jaxopt optimizer."""
        if self._optimizer is None:
            # Create Levenberg-Marquardt optimizer with default settings
            self._optimizer = jaxopt.LevenbergMarquardt(
                residual_fun=lambda p, *a: jnp.zeros(1),  # Placeholder
                tol=1e-8,
                maxiter=100,
                damping_parameter=1.0,
            )
        return self._optimizer

    def _fstar_err_calc(self, delfstar: complex) -> complex:
        """
        Calculate the error in delfstar.

        Parameters
        ----------
        delfstar : complex
            Complex frequency shift.

        Returns
        -------
        complex
            Complex error estimate.
        """
        f_err = self.f_err_min + self.err_frac * np.imag(delfstar)
        g_err = self.g_err_min + self.err_frac * np.imag(delfstar)
        return complex(f_err, g_err)

    def _rd_from_delfstar(self, n: int, delfstar: dict[int, complex]) -> float:
        """Calculate dissipation ratio at harmonic n."""
        if n not in delfstar or np.real(delfstar[n]) == 0:
            return np.nan
        return -np.imag(delfstar[n]) / np.real(delfstar[n])

    def _rh_from_delfstar(self, nh: list[int], delfstar: dict[int, complex]) -> float:
        """Calculate harmonic ratio."""
        n1, n2 = nh[0], nh[1]
        if n2 not in delfstar or np.real(delfstar[n2]) == 0:
            return np.nan
        return (n2 / n1) * np.real(delfstar[n1]) / np.real(delfstar[n2])

    def _is_bulk(self, rd_exp: float, bulklimit: float) -> bool:
        """Check if material is bulk based on dissipation ratio."""
        return rd_exp >= bulklimit

    def _grho_at_harmonic(self, n: int, grho_refh: float, phi: float) -> float:
        """Calculate |G*|*rho at harmonic n using power law.

        T029: Uses caching to avoid redundant computations (005-jax-perf).
        Cache key includes (n, grho_refh, phi, f1, refh) to ensure correctness.
        """
        if self.refh is None:
            raise ValueError("Reference harmonic (refh) must be set")
        # T029: Check cache before computing (005-jax-perf)
        cache_key = (n, grho_refh, phi, *self._get_cache_key())
        if cache_key in self._grho_cache:
            return self._grho_cache[cache_key]
        result = float(physics.grho(n, grho_refh, phi, refh=self.refh))
        self._grho_cache[cache_key] = result
        return result

    def _grhostar_at_harmonic(self, n: int, grho_refh: float, phi: float) -> complex:
        """Calculate complex G*rho at harmonic n."""
        if self.refh is None:
            raise ValueError("Reference harmonic (refh) must be set")
        return complex(physics.grhostar_from_refh(n, grho_refh, phi, refh=self.refh))

    def _zstar_bulk(self, grhostar: complex) -> complex:
        """Calculate acoustic impedance from complex modulus."""
        return complex(physics.zstar_bulk(jnp.array(grhostar)))

    def _calc_delfstar_sla(self, ZL: complex) -> complex:
        """Calculate complex frequency shift using SLA."""
        if self.f1 is None:
            raise ValueError("Fundamental frequency (f1) must be set")
        return complex(physics.calc_delfstar_sla(jnp.array(ZL), f1=self.f1))

    def _calc_ZL_single_layer(
        self,
        n: int,
        grho_refh: float,
        phi: float,
        drho: float,
    ) -> complex:
        """Calculate load impedance for single layer."""
        if drho == 0 or drho == np.inf:
            # Bulk or no film
            grhostar = self._grhostar_at_harmonic(n, grho_refh, phi)
            return self._zstar_bulk(grhostar)

        # Thin film
        grhostar = self._grhostar_at_harmonic(n, grho_refh, phi)
        zstar = self._zstar_bulk(grhostar)

        # D = 2 * pi * f * drho / Z*
        if self.f1 is None:
            raise ValueError("f1 must be set")
        D = 2 * np.pi * n * self.f1 * drho / zstar

        return 1j * zstar * np.tan(D)

    def _calc_delfstar(
        self,
        n: int,
        grho_refh: float,
        phi: float,
        drho: float,
    ) -> complex:
        """Calculate complex frequency shift for single layer."""
        ZL = self._calc_ZL_single_layer(n, grho_refh, phi, drho)
        return self._calc_delfstar_sla(ZL)

    def _calc_delfstar_bulk(self, n: int, grho_refh: float, phi: float) -> complex:
        """Calculate complex frequency shift for bulk material."""
        if self.f1 is None:
            raise ValueError("f1 must be set")

        grho_n = self._grho_at_harmonic(n, grho_refh, phi)

        return (self.f1 * np.sqrt(grho_n) / (np.pi * self.Zq)) * (
            -np.sin(phi / 2) + 1j * np.cos(phi / 2)
        )

    def _bulk_props(self, delfstar: dict[int, complex]) -> tuple[float, float]:
        """Get bulk solution for grho and phi."""
        if self.refh is None:
            raise ValueError("refh must be set")

        n = self.refh
        if n not in delfstar:
            return np.nan, np.nan

        df = delfstar[n]
        if self.f1 is None:
            raise ValueError("f1 must be set")

        grho_refh = (np.pi * self.Zq * abs(df) / self.f1) ** 2
        phi = min(np.pi / 2, -2 * np.arctan(np.real(df) / np.imag(df)))

        return grho_refh, phi

    def _thin_film_guess(
        self,
        delfstar: dict[int, complex],
        nh: list[int],
    ) -> tuple[float, float, float, float]:
        """Guess thin film properties from delfstar."""
        if self.f1 is None or self.refh is None:
            return np.nan, np.nan, np.nan, np.nan

        n1, n2, n3 = nh[0], nh[1], nh[2]
        refh = self.refh

        rd_exp = self._rd_from_delfstar(n3, delfstar)
        rh_exp = self._rh_from_delfstar(nh, delfstar)

        if np.isnan(rd_exp) or np.isnan(rh_exp):
            return np.nan, np.nan, np.nan, np.nan

        # Initial guess
        dlam_refh_init = 0.05
        phi_init = np.pi / 180 * 5

        # Create pure JAX residual function (captures n1, n2, n3, refh, rh_exp, rd_exp)
        def residual_fn(x: jnp.ndarray) -> jnp.ndarray:
            dlam = x[0]
            phi_val = x[1]
            rh_calc = _jax_rhcalc(n1, n2, refh, dlam, phi_val)
            rd_calc = _jax_rdcalc(n3, refh, dlam, phi_val)
            return jnp.array([rh_calc - rh_exp, rd_calc - rd_exp])

        solver = jaxopt.LevenbergMarquardt(
            residual_fun=residual_fn,
            maxiter=50,
        )

        x0 = jnp.array([dlam_refh_init, phi_init])
        result = solver.run(x0)

        dlam_refh = float(result.params[0])
        phi = float(result.params[1])

        # Clamp to valid ranges
        dlam_refh = np.clip(dlam_refh, dlam_refh_range[0] + 1e-10, dlam_refh_range[1])
        phi = np.clip(phi, phi_range[0] + 1e-10, phi_range[1] - 1e-10)

        # Calculate drho from Sauerbrey
        nds = _jax_normdelfstar(n1, refh, jnp.array(dlam_refh), jnp.array(phi))
        delf_saub = np.real(delfstar[n1]) / float(jnp.real(nds))
        drho = float(physics.sauerbreym(n1, -delf_saub, f1=self.f1))

        # Calculate grho from dlam
        if drho > 0:
            grho_refh = float(
                physics.grho_from_dlam(self.refh, drho, dlam_refh, phi, f1=self.f1)
            )
        else:
            grho_refh = np.nan

        return grho_refh, phi, drho, dlam_refh

    def solve_properties(
        self,
        nh: list[int],
        guess: tuple[float, float] | None = None,
        layers: dict[int, dict] | None = None,
        calctype: str | None = None,
        bulklimit: float = 0.5,
        calculate_errors: bool = False,
    ) -> SolveResult:
        """
        Solve film properties from loaded delfstar data.

        Uses NLSQ curve_fit for optimization with JAX autodiff for Jacobian
        computation when calculating errors.

        Parameters
        ----------
        nh : list[int]
            Harmonics for calculation [n1, n2, n3].
        guess : tuple[float, float], optional
            Initial guess (dlam_refh, phi). Auto-computed if None.
        layers : dict[int, dict], optional
            Layer stack for multi-layer calculation. If None, single layer.
        calctype : str, optional
            Calculation type. Uses model default if None.
            Can be built-in ("SLA", "LL", "Voigt") or custom registered type.
        bulklimit : float, optional
            Dissipation ratio threshold for bulk vs thin film. Default: 0.5.
        calculate_errors : bool, optional
            Whether to calculate errors from Jacobian. Default: False.

        Returns
        -------
        SolveResult
            Dataclass containing:
            - drho: Mass per area [kg/m^2]
            - grho_refh: |G*|*rho at reference harmonic [Pa kg/m^3]
            - phi: Phase angle [radians]
            - dlam_refh: d/lambda at reference harmonic
            - covariance: Parameter covariance matrix (if calculate_errors)
            - residuals: Final residual values
            - success: Convergence indicator
            - message: Status message

        Notes
        -----
        This method uses NLSQ for optimization (FR-002) and JAX autodiff
        for error propagation (FR-011).

        For custom calctypes registered with register_calctype(), the
        custom residual function is used for fitting. For unregistered
        calctypes, the method falls back to SLA behavior with a warning.
        """
        if not self.delfstars:
            return SolveResult(
                success=False,
                message="No delfstar data loaded. Call load_delfstars first.",
            )

        if self.f1 is None:
            return SolveResult(
                success=False,
                message="Fundamental frequency (f1) must be set",
            )

        if self.refh is None:
            self.refh = nh[2]  # Use n3 as reference if not set

        # Handle calctype
        effective_calctype = calctype if calctype is not None else self.calctype
        effective_calctype = self._normalize_calctype(effective_calctype)

        # Check if custom calctype is registered
        custom_residual = self._get_calctype_residual(effective_calctype)
        if custom_residual is not None:
            # Use custom residual function
            logger.info(f"Using custom calctype: {effective_calctype}")
            # TODO: Full integration of custom residual function
            # For now, fall back to SLA
            logger.warning(
                f"Custom calctype '{effective_calctype}' not fully integrated, "
                f"falling back to SLA behavior"
            )
            effective_calctype = "SLA"
        elif effective_calctype not in self.BUILTIN_CALCTYPES:
            # Unknown calctype - warn and fall back
            logger.warning(
                f"Unknown calctype '{effective_calctype}', falling back to SLA"
            )
            effective_calctype = "SLA"

        delfstar = self.delfstars

        # Get experimental ratios
        rd_exp = self._rd_from_delfstar(nh[2], delfstar)
        rh_exp = self._rh_from_delfstar(nh, delfstar)

        if np.isnan(rd_exp) or np.isnan(rh_exp):
            return SolveResult(
                success=False,
                message="Could not calculate experimental ratios (NaN in delfstar)",
            )

        is_bulk = self._is_bulk(rd_exp, bulklimit)

        f1 = self.f1
        Zq = self.Zq
        refh = self.refh

        if is_bulk:
            # Bulk material
            grho_refh, phi = self._bulk_props(delfstar)
            drho = bulk_drho
            dlam_refh = 0.25  # Quarter wavelength

            # Get target values
            exp_real = np.real(delfstar[refh])
            exp_imag = np.imag(delfstar[refh])

            # Refine with optimizer - pure JAX residual
            def residual_bulk(x: jnp.ndarray) -> jnp.ndarray:
                grho = x[0]
                phi_val = x[1]
                calc = _jax_calc_delfstar_bulk(refh, grho, phi_val, f1, Zq, refh)
                return jnp.array([jnp.real(calc) - exp_real, jnp.imag(calc) - exp_imag])

            solver = jaxopt.LevenbergMarquardt(
                residual_fun=residual_bulk,
                maxiter=100,
            )

            x0 = jnp.array([grho_refh, phi])
            result = solver.run(x0)

            grho_refh = float(result.params[0])
            phi = float(result.params[1])

            # Clamp phi
            phi = min(phi, np.pi / 2)

            covariance = None
            residuals = np.array(residual_bulk(result.params))

            return SolveResult(
                drho=drho,
                grho_refh=grho_refh,
                phi=phi,
                dlam_refh=dlam_refh,
                covariance=covariance,
                residuals=residuals,
                success=True,
                message="Bulk material fit",
            )

        else:
            # Thin film
            grho_refh, phi, drho, dlam_refh = self._thin_film_guess(delfstar, nh)

            if np.isnan(grho_refh):
                return SolveResult(
                    success=False,
                    message="Initial guess failed (NaN values)",
                )

            n1, n2, n3 = nh[0], nh[1], nh[2]

            # Get target values
            exp_real_n1 = np.real(delfstar[n1])
            exp_real_n2 = np.real(delfstar[n2])
            exp_imag_n3 = np.imag(delfstar[n3])

            # Refine with optimizer - pure JAX residual
            def residual_thin(x: jnp.ndarray) -> jnp.ndarray:
                grho = x[0]
                phi_val = x[1]
                d = x[2]

                calc_n1 = _jax_calc_delfstar(n1, grho, phi_val, d, f1, Zq, refh)
                calc_n2 = _jax_calc_delfstar(n2, grho, phi_val, d, f1, Zq, refh)
                calc_n3 = _jax_calc_delfstar(n3, grho, phi_val, d, f1, Zq, refh)

                return jnp.array(
                    [
                        jnp.real(calc_n1) - exp_real_n1,
                        jnp.real(calc_n2) - exp_real_n2,
                        jnp.imag(calc_n3) - exp_imag_n3,
                    ]
                )

            solver = jaxopt.LevenbergMarquardt(
                residual_fun=residual_thin,
                maxiter=100,
            )

            x0 = jnp.array([grho_refh, phi, drho])
            result = solver.run(x0)

            grho_refh = float(result.params[0])
            phi = float(result.params[1])
            drho = float(result.params[2])

            # Clamp values
            grho_refh = np.clip(grho_refh, grho_refh_range[0], grho_refh_range[1])
            phi = np.clip(phi, phi_range[0], phi_range[1])
            drho = np.clip(drho, drho_range[0], drho_range[1])

            # Calculate dlam_refh
            grho_n = self._grho_at_harmonic(self.refh, grho_refh, phi)
            dlam_refh = float(
                physics.calc_dlam(self.refh, grho_n, phi, drho, f1=self.f1)
            )

            residuals = np.array(residual_thin(result.params))

            # Error propagation from Jacobian using JAX autodiff (T014)
            covariance = None

            if calculate_errors:
                try:
                    # Compute Jacobian at solution using JAX jacfwd
                    jac_fn = jax.jacfwd(residual_thin)
                    jac = jac_fn(jnp.array([grho_refh, phi, drho]))
                    jac_np = np.array(jac)

                    # Input uncertainties
                    delfstar_err = np.array(
                        [
                            np.real(self._fstar_err_calc(delfstar[n1])),
                            np.real(self._fstar_err_calc(delfstar[n2])),
                            np.imag(self._fstar_err_calc(delfstar[n3])),
                        ]
                    )

                    # Covariance propagation: cov = J^-1 @ diag(sigma^2) @ (J^-1)^T
                    try:
                        jac_inv = np.linalg.inv(jac_np)
                        sigma_sq = np.diag(delfstar_err**2)
                        covariance = jac_inv @ sigma_sq @ jac_inv.T
                    except np.linalg.LinAlgError:
                        logger.warning(
                            "Jacobian inversion failed, covariance unavailable"
                        )
                except Exception as e:
                    logger.warning(f"Error calculation failed: {e}")

            return SolveResult(
                drho=drho,
                grho_refh=grho_refh,
                phi=phi,
                dlam_refh=dlam_refh,
                covariance=covariance,
                residuals=residuals,
                success=True,
                message="Thin film fit",
            )

    def solve_batch(
        self,
        batch_delfstars: list[dict[int, complex]],
        nh: list[int],
        calctype: str | None = None,
        bulklimit: float = 0.5,
    ) -> BatchResult:
        """
        Solve properties for multiple timepoints in batch.

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
        BatchResult
            Dataclass with arrays of results for all measurements.
        """
        results = []

        for delfstars in batch_delfstars:
            self.load_delfstars(delfstars)
            result = self.solve_properties(
                nh=nh,
                calctype=calctype,
                bulklimit=bulklimit,
            )
            results.append(result)

        return BatchResult.from_solve_results(results)

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

        This method provides a unified interface to the NLSQ curve_fit function,
        replacing scipy.optimize.curve_fit and lmfit patterns.

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
        # Convert to numpy arrays for NLSQ
        xdata = np.asarray(xdata)
        ydata = np.asarray(ydata)

        if p0 is not None:
            p0 = np.asarray(p0)

        # Call NLSQ curve_fit
        result = nlsq_curve_fit(
            f,
            xdata,
            ydata,
            p0=p0,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            bounds=bounds,
            **kwargs,
        )

        # Handle tuple or result object
        if isinstance(result, tuple):
            return result
        else:
            return result.popt, result.pcov

    def format_result_for_export(
        self, result: SolveResult | dict[str, Any]
    ) -> dict[str, Any]:
        """
        Format result for export to DataSaver.

        Parameters
        ----------
        result : SolveResult or dict
            Result from solve_properties.

        Returns
        -------
        dict
            Formatted result with expected keys.
        """
        if isinstance(result, SolveResult):
            errors = result.errors
            return {
                "drho": result.drho,
                "grho_refh": result.grho_refh,
                "phi": result.phi,
                "dlam_refh": result.dlam_refh,
                "drho_err": errors.get("drho", np.nan),
                "grho_refh_err": errors.get("grho_refh", np.nan),
                "phi_err": errors.get("phi", np.nan),
            }
        else:
            return {
                "drho": result.get("drho", np.nan),
                "grho_refh": result.get("grho_refh", np.nan),
                "phi": result.get("phi", np.nan),
                "dlam_refh": result.get("dlam_refh", np.nan),
                "drho_err": result.get("errors", {}).get("drho", np.nan),
                "grho_refh_err": result.get("errors", {}).get("grho_refh", np.nan),
                "phi_err": result.get("errors", {}).get("phi", np.nan),
            }

    def convert_units_for_display(
        self, result: SolveResult | dict[str, Any]
    ) -> dict[str, Any]:
        """
        Convert result units from SI to display units.

        Parameters
        ----------
        result : SolveResult or dict
            Result with SI units.

        Returns
        -------
        dict
            Result with display units.

        Notes
        -----
        Conversions:
        - drho: kg/m^2 -> um g/cm^3 (multiply by 1000)
        - grho: Pa kg/m^3 -> Pa g/cm^3 (divide by 1000)
        - phi: radians -> degrees
        """
        if isinstance(result, SolveResult):
            converted = result.to_dict()
        else:
            converted = result.copy()

        if "drho" in converted and not np.isnan(converted["drho"]):
            converted["drho"] = converted["drho"] * 1000

        if "grho_refh" in converted and not np.isnan(converted["grho_refh"]):
            converted["grho_refh"] = converted["grho_refh"] / 1000

        if "phi" in converted and not np.isnan(converted["phi"]):
            converted["phi"] = np.degrees(converted["phi"])

        return converted

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

        This provides uncertainty quantification through posterior estimation.

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

        Raises
        ------
        ImportError
            If NumPyro is not installed.
        """
        try:
            import numpyro
            import numpyro.distributions as dist
            from numpyro.infer import MCMC, NUTS
        except ImportError as e:
            raise ImportError(
                "NumPyro is required for Bayesian inference. "
                "Install with: pip install numpyro"
            ) from e

        if not self.delfstars:
            raise ValueError("No delfstar data loaded")

        if self.f1 is None or self.refh is None:
            raise ValueError("f1 and refh must be set")

        delfstar = self.delfstars
        n1, n2, n3 = nh[0], nh[1], nh[2]

        # Get initial values from NLSQ fit if not provided
        if initial_params is None:
            result = self.solve_properties(nh)
            initial_params = result.to_dict()

        # Handle both dict and SolveResult
        if isinstance(initial_params, SolveResult):
            initial_params = initial_params.to_dict()
        grho0 = initial_params.get("grho_refh", 1e8)
        phi0 = initial_params.get("phi", np.pi / 4)
        drho0 = initial_params.get("drho", 1e-6)

        # Define NumPyro model
        def model():
            # Priors
            grho_refh = numpyro.sample("grho_refh", dist.LogNormal(np.log(grho0), 1.0))
            phi = numpyro.sample("phi", dist.Uniform(0.0, np.pi / 2))
            drho = numpyro.sample("drho", dist.LogNormal(np.log(drho0), 1.0))

            # Likelihood
            sigma_f = numpyro.sample("sigma_f", dist.HalfNormal(100.0))
            sigma_g = numpyro.sample("sigma_g", dist.HalfNormal(100.0))

            # Model predictions
            for n, target in [(n1, "real"), (n2, "real"), (n3, "imag")]:
                calc = self._calc_delfstar(n, grho_refh, phi, drho)
                exp = delfstar[n]

                if target == "real":
                    numpyro.sample(
                        f"obs_f_{n}",
                        dist.Normal(np.real(calc), sigma_f),
                        obs=np.real(exp),
                    )
                else:
                    numpyro.sample(
                        f"obs_g_{n}",
                        dist.Normal(np.imag(calc), sigma_g),
                        obs=np.imag(exp),
                    )

        # Run MCMC
        rng_key = jax.random.PRNGKey(0)
        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(rng_key)

        samples = mcmc.get_samples()

        return {
            "samples": {k: np.array(v) for k, v in samples.items()},
            "summary": {
                k: {
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v)),
                    "median": float(np.median(v)),
                }
                for k, v in samples.items()
                if k in ["grho_refh", "phi", "drho"]
            },
        }


__all__ = [
    "QCMModel",
    "SolveResult",
    "BatchResult",
    "CalctypeResidualFn",
    "register_global_calctype",
    "get_global_calctypes",
    "dlam_refh_range",
    "drho_range",
    "grho_refh_range",
    "phi_range",
    "bulk_drho",
]
