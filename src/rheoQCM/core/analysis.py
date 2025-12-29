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
>>> print(f"drho = {result.drho:.3e} kg/m^2")

Using legacy function names:

>>> from rheoQCM.core.analysis import sauerbreyf, sauerbreym
>>> delf = sauerbreyf(3, 1e-6)  # 3rd harmonic, 1 ug/cm^2
>>> drho = sauerbreym(3, -1000)  # Calculate mass from frequency shift

Batch processing with GPU acceleration (US3):

>>> from rheoQCM.core.analysis import batch_analyze_vmap
>>> delfstars = jnp.array([[-1000+100j, -1700+180j], ...])  # (N, 2) array
>>> result = batch_analyze_vmap(delfstars, harmonics=[3, 5], nhcalc="35")
>>> print(f"Processed {len(result)} measurements on {get_jax_backend()}")

See Also
--------
rheoQCM.core.physics : Layer 1 physics calculations
rheoQCM.core.model : Layer 2 model logic
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rheoQCM.core.jax_config import configure_jax, get_jax_backend, is_gpu_available
from rheoQCM.core.model import (
    BatchResult,
    QCMModel,
    SolveResult,
    bulk_drho,
)
from rheoQCM.core.physics import (
    C0byA,
    # Constants
    Zq,
    air_default,
    bulk_props,
    calc_D,
    # SLA equations
    calc_delfstar_sla,
    calc_deltarho,
    calc_dlam,
    calc_lamrho,
    create_interp_func,
    d26,
    deltarho_bulk,
    dlam_refh_range,
    dq,
    drho_range,
    e26,
    electrode_default,
    eps0,
    epsq,
    etarho,
    f1_default,
    # Utility functions
    find_peaks,
    g0_default,
    # Complex modulus calculations
    grho,
    grho_from_dlam,
    grho_refh_range,
    grhostar,
    grhostar_from_refh,
    interp_cubic,
    interp_linear,
    # Kotula model
    kotula_gstar,
    kotula_xi,
    normdelfstar,
    phi_range,
    # Sauerbrey equations
    sauerbreyf,
    sauerbreym,
    savgol_filter,
    water_default,
    zstar_bulk,
)

# Ensure JAX is configured
configure_jax()

logger = logging.getLogger(__name__)

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
    results : list
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
        self._results: list[Any] = []

    @property
    def model(self) -> QCMModel:
        """Access the underlying QCMModel instance."""
        return self._model

    @property
    def results(self) -> list[Any]:
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
    ) -> SolveResult:
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
        SolveResult
            Results containing:
            - grho_refh: |G*|*rho at reference harmonic [Pa kg/m^3]
            - phi: Phase angle [radians]
            - drho: Mass per area [kg/m^2]
            - dlam_refh: d/lambda at reference harmonic
            - covariance: Error estimates (if calculate_errors=True)
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
    ) -> BatchResult:
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
        BatchResult
            Dataclass with arrays of results for all measurements.
        """
        result = self._model.solve_batch(
            batch_delfstars=batch_delfstars,
            nh=nh,
            calctype=calctype,
            bulklimit=bulklimit,
        )
        # Note: BatchResult is not iterable, so we don't extend _results
        # Store as a single entry if needed
        return result

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

    def format_result(self, result: Any | None = None) -> dict[str, Any]:
        """
        Format result for export.

        Parameters
        ----------
        result : SolveResult, optional
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

    def convert_to_display_units(self, result: Any | None = None) -> dict[str, Any]:
        """
        Convert result units from SI to display units.

        Parameters
        ----------
        result : SolveResult, optional
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
) -> SolveResult:
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
    SolveResult
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
) -> BatchResult:
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
    BatchResult
        Dataclass with arrays of results for all measurements.
    """
    analyzer = QCMAnalyzer(f1=f1, refh=refh, calctype=calctype)
    return analyzer.analyze_batch(
        batch_delfstars=batch_delfstars,
        nh=nh,
        calctype=calctype,
        bulklimit=bulklimit,
    )


# =============================================================================
# T042, T043: vmap-enabled batch_analyze with GPU acceleration (US3)
# =============================================================================


def _log_backend_info() -> str:
    """
    Log GPU/CPU backend information for batch processing.

    T043: Add GPU backend detection and logging.

    Returns
    -------
    str
        The active backend ("cpu", "gpu", "tpu", or "unknown").
    """
    backend = get_jax_backend()
    gpu_available = is_gpu_available()

    if backend == "gpu":
        logger.info("batch_analyze_vmap: Using GPU acceleration")
    elif backend == "tpu":
        logger.info("batch_analyze_vmap: Using TPU acceleration")
    elif gpu_available:
        logger.info(
            "batch_analyze_vmap: GPU available but running on CPU. "
            "Consider using GPU for larger batches."
        )
    else:
        logger.info("batch_analyze_vmap: Running on CPU (no GPU available)")

    return backend


# Pure JAX functions for vmap-compatible batch processing
@jax.jit
def _vmap_normdelfstar(
    n: jnp.ndarray,
    dlam_refh: jnp.ndarray,
    phi: jnp.ndarray,
    refh: int,
) -> jnp.ndarray:
    """Calculate normalized delfstar (vmap-compatible)."""
    dlam_n = dlam_refh * (n / refh) ** (1 - phi / jnp.pi)
    D = 2 * jnp.pi * dlam_n * (1 - 1j * jnp.tan(phi / 2))
    return -jnp.sin(D) / D / jnp.cos(D)


@jax.jit
def _vmap_rhcalc(
    n1: int,
    n2: int,
    dlam_refh: jnp.ndarray,
    phi: jnp.ndarray,
    refh: int,
) -> jnp.ndarray:
    """Calculate harmonic ratio (vmap-compatible)."""
    nds1 = _vmap_normdelfstar(n1, dlam_refh, phi, refh)
    nds2 = _vmap_normdelfstar(n2, dlam_refh, phi, refh)
    return jnp.real(nds1) / jnp.real(nds2)


@jax.jit
def _vmap_rdcalc(
    n3: int,
    dlam_refh: jnp.ndarray,
    phi: jnp.ndarray,
    refh: int,
) -> jnp.ndarray:
    """Calculate dissipation ratio (vmap-compatible)."""
    nds = _vmap_normdelfstar(n3, dlam_refh, phi, refh)
    return -jnp.imag(nds) / jnp.real(nds)


def _parse_nhcalc(nhcalc: str) -> tuple[int, int, int]:
    """Parse nhcalc string to get harmonic numbers."""
    if len(nhcalc) == 2:
        return int(nhcalc[0]), int(nhcalc[1]), int(nhcalc[0])
    elif len(nhcalc) == 3:
        return int(nhcalc[0]), int(nhcalc[1]), int(nhcalc[2])
    else:
        raise ValueError(f"Invalid nhcalc format: {nhcalc}")


# T042: Combined residual function for autodiff Jacobian (005-jax-perf)
def _residual_fn(
    params: jnp.ndarray,
    n1: int,
    n2: int,
    n3: int,
    refh: int,
    rh_exp: jnp.ndarray,
    rd_exp: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute residuals [rh_calc - rh_exp, rd_calc - rd_exp] for Newton solver.

    Parameters
    ----------
    params : jnp.ndarray
        Array [dlam_refh, phi] of shape (2,).
    n1, n2, n3 : int
        Harmonic numbers for rh and rd calculations.
    refh : int
        Reference harmonic.
    rh_exp, rd_exp : jnp.ndarray
        Experimental harmonic ratio and dissipation ratio.

    Returns
    -------
    jnp.ndarray
        Residual array [rh_calc - rh_exp, rd_calc - rd_exp] of shape (2,).
    """
    dlam_refh = params[0]
    phi = params[1]
    rh_calc = _vmap_rhcalc(n1, n2, dlam_refh, phi, refh)
    rd_calc = _vmap_rdcalc(n3, dlam_refh, phi, refh)
    return jnp.array([rh_calc - rh_exp, rd_calc - rd_exp])


# T043: Autodiff Jacobian using jax.jacfwd (005-jax-perf)
def _jacobian_autodiff(
    params: jnp.ndarray,
    n1: int,
    n2: int,
    n3: int,
    refh: int,
    rh_exp: jnp.ndarray,
    rd_exp: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute Jacobian of residual function using JAX autodiff.

    Uses jacfwd for forward-mode autodiff, which is efficient for
    functions with few inputs (2 parameters) and few outputs (2 residuals).

    Parameters
    ----------
    params : jnp.ndarray
        Array [dlam_refh, phi] of shape (2,).
    n1, n2, n3 : int
        Harmonic numbers.
    refh : int
        Reference harmonic.
    rh_exp, rd_exp : jnp.ndarray
        Experimental ratios.

    Returns
    -------
    jnp.ndarray
        Jacobian matrix of shape (2, 2):
        [[drh/ddlam, drh/dphi],
         [drd/ddlam, drd/dphi]]
    """
    return jax.jacfwd(lambda p: _residual_fn(p, n1, n2, n3, refh, rh_exp, rd_exp))(
        params
    )


def _solve_single_measurement(
    delfstar_n1: jnp.ndarray,
    delfstar_n2: jnp.ndarray,
    delfstar_n3: jnp.ndarray,
    n1: int,
    n2: int,
    n3: int,
    f1: float,
    refh: int,
    bulklimit: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Solve for film properties from a single measurement (vmap-compatible).

    Uses jax.lax.cond for vmap-compatible control flow instead of Python if/else.

    Returns
    -------
    tuple
        (drho, grho_refh, phi, success)
    """
    # Calculate experimental ratios
    rd_exp = -jnp.imag(delfstar_n3) / jnp.real(delfstar_n3)
    rh_exp = (n2 / n1) * jnp.real(delfstar_n1) / jnp.real(delfstar_n2)

    # Determine if bulk (using lax.cond for vmap compatibility)
    is_bulk = rd_exp >= bulklimit

    # Bulk solution
    def bulk_solution(inputs):
        delfstar_refh = inputs[0]
        grho_refh = (jnp.pi * Zq * jnp.abs(delfstar_refh) / f1) ** 2
        phi = jnp.minimum(
            jnp.pi / 2,
            -2 * jnp.arctan(jnp.real(delfstar_refh) / jnp.imag(delfstar_refh)),
        )
        drho = jnp.inf
        return drho, grho_refh, phi, jnp.array(True)

    # Thin film solution
    def thin_film_solution(inputs):
        rd_exp_local = inputs[4]
        rh_exp_local = inputs[5]

        # Initial guess
        dlam_refh = jnp.array(0.05)
        phi = jnp.array(0.1)

        # Simple iterative refinement (vmap-compatible fixed iterations)
        # T044: Use autodiff Jacobian instead of finite differences (005-jax-perf)
        def body_fn(carry, _):
            dlam_r, phi_r = carry
            params = jnp.array([dlam_r, phi_r])

            # T044: Compute Jacobian using autodiff (1 call vs 4 finite-diff calls)
            jac = _jacobian_autodiff(
                params, n1, n2, n3, refh, rh_exp_local, rd_exp_local
            )
            drh_ddlam = jac[0, 0]
            drh_dphi = jac[0, 1]
            drd_ddlam = jac[1, 0]
            drd_dphi = jac[1, 1]

            # Jacobian determinant
            det = drh_ddlam * drd_dphi - drh_dphi * drd_ddlam
            det_safe = jnp.where(jnp.abs(det) < 1e-20, 1e-20, det)

            # Compute residuals (need current rh_calc and rd_calc)
            rh_calc = _vmap_rhcalc(n1, n2, dlam_r, phi_r, refh)
            rd_calc = _vmap_rdcalc(n3, dlam_r, phi_r, refh)
            r_rh = rh_exp_local - rh_calc
            r_rd = rd_exp_local - rd_calc

            # Newton step
            d_dlam = (drd_dphi * r_rh - drh_dphi * r_rd) / det_safe
            d_phi = (-drd_ddlam * r_rh + drh_ddlam * r_rd) / det_safe

            # Damped update
            damping = 0.5
            dlam_new = dlam_r + damping * d_dlam
            phi_new = phi_r + damping * d_phi

            # Clamp to valid ranges
            dlam_new = jnp.clip(dlam_new, 1e-6, 10.0)
            phi_new = jnp.clip(phi_new, 1e-6, jnp.pi / 2 - 1e-6)

            return (dlam_new, phi_new), None

        # Run fixed number of iterations (vmap-compatible)
        (dlam_refh_final, phi_final), _ = jax.lax.scan(
            body_fn, (dlam_refh, phi), None, length=20
        )

        # Calculate drho from Sauerbrey
        nds = _vmap_normdelfstar(n1, dlam_refh_final, phi_final, refh)
        delf_saub = jnp.real(delfstar_n1) / jnp.real(nds)
        drho = delf_saub * Zq / (2 * n1 * f1**2)

        # Calculate grho from dlam
        grho_refh = (drho * refh * f1 * jnp.cos(phi_final / 2) / dlam_refh_final) ** 2

        # Check convergence
        rh_final = _vmap_rhcalc(n1, n2, dlam_refh_final, phi_final, refh)
        rd_final = _vmap_rdcalc(n3, dlam_refh_final, phi_final, refh)
        rh_err = jnp.abs(rh_final - rh_exp_local)
        rd_err = jnp.abs(rd_final - rd_exp_local)
        success = (rh_err < 0.01) & (rd_err < 0.01) & jnp.isfinite(drho)

        return drho, grho_refh, phi_final, success

    # Select reference harmonic delfstar based on refh
    delfstar_refh = jax.lax.switch(
        jnp.int32((refh == n1) + 2 * (refh == n2) + 3 * (refh == n3)),
        [
            lambda: delfstar_n1,  # default/fallback
            lambda: delfstar_n1,  # refh == n1
            lambda: delfstar_n2,  # refh == n2
            lambda: delfstar_n3,  # refh == n3
        ],
    )

    inputs = (delfstar_refh, delfstar_n1, delfstar_n2, delfstar_n3, rd_exp, rh_exp)

    drho, grho_refh, phi, success = jax.lax.cond(
        is_bulk,
        bulk_solution,
        thin_film_solution,
        inputs,
    )

    return drho, grho_refh, phi, success


def batch_analyze_vmap(
    delfstars: jnp.ndarray,
    harmonics: list[int],
    nhcalc: str = "35",
    f1: float = f1_default,
    refh: int = 3,
    bulklimit: float = 0.5,
) -> BatchResult:
    """
    Batch analyze multiple measurements using vmap for GPU acceleration.

    T042: Implement batch_analyze using vmap.
    T043: Add GPU backend detection and logging.

    This function uses JAX's vmap to process multiple QCM-D measurements
    in parallel, enabling GPU acceleration for large datasets.

    Parameters
    ----------
    delfstars : jnp.ndarray
        Complex frequency shifts array, shape (N, H) where N is number of
        measurements and H is number of harmonics. The columns correspond
        to the harmonics specified in the `harmonics` parameter.
    harmonics : list[int]
        List of harmonic numbers corresponding to columns in delfstars.
        Must contain at least 2 harmonics.
    nhcalc : str, optional
        String specifying harmonics for rh and rd calculation.
        E.g., "35" means use harmonics 3 and 5. Default: "35".
    f1 : float, optional
        Fundamental frequency [Hz]. Default: 5e6 Hz.
    refh : int, optional
        Reference harmonic for grho scaling. Default: 3.
    bulklimit : float, optional
        Dissipation ratio threshold for bulk detection. Default: 0.5.

    Returns
    -------
    BatchResult
        Dataclass containing arrays of:
        - drho: Mass per area [kg/m^2], shape (N,)
        - grho_refh: |G*|*rho at reference harmonic [Pa kg/m^3], shape (N,)
        - phi: Phase angle [radians], shape (N,)
        - dlam_refh: d/lambda at reference harmonic, shape (N,)
        - success: Boolean convergence indicators, shape (N,)

    Notes
    -----
    This function uses vmap for parallelization (FR-006) and runs on GPU
    if available (logged at INFO level).

    Examples
    --------
    >>> delfstars = jnp.array([
    ...     [-87768.0 + 155.7j, -159742.7 + 888.7j],
    ...     [-90000.0 + 160.0j, -165000.0 + 920.0j],
    ... ])
    >>> result = batch_analyze_vmap(delfstars, harmonics=[3, 5], nhcalc="35")
    >>> print(f"Processed {len(result)} measurements")
    """
    # Log backend information (T043)
    backend = _log_backend_info()
    logger.info(f"batch_analyze_vmap: Processing {len(delfstars)} measurements")

    # Parse nhcalc
    n1, n2, n3 = _parse_nhcalc(nhcalc)

    # Validate harmonics
    if len(harmonics) < 2:
        raise ValueError("Need at least 2 harmonics for analysis")

    # Map harmonic numbers to column indices
    h_to_idx = {h: i for i, h in enumerate(harmonics)}

    if n1 not in h_to_idx or n2 not in h_to_idx:
        raise ValueError(
            f"nhcalc harmonics {n1}, {n2} not found in provided harmonics {harmonics}"
        )

    # Get indices for each needed harmonic
    idx1 = h_to_idx[n1]
    idx2 = h_to_idx[n2]
    # n3 might be same as n1 or n2
    idx3 = h_to_idx.get(n3, idx1)

    # Extract columns for each harmonic
    delfstar_n1 = delfstars[:, idx1]
    delfstar_n2 = delfstars[:, idx2]
    delfstar_n3 = delfstars[:, idx3]

    # Create vmapped solver
    @jax.jit
    def solve_batch(ds_n1, ds_n2, ds_n3):
        return jax.vmap(
            lambda d1, d2, d3: _solve_single_measurement(
                d1, d2, d3, n1, n2, n3, f1, refh, bulklimit
            )
        )(ds_n1, ds_n2, ds_n3)

    # Run vectorized computation
    drho, grho_refh, phi, success = solve_batch(delfstar_n1, delfstar_n2, delfstar_n3)

    # Calculate dlam_refh from results
    grho_n_refh = grho_refh  # At refh, grho_n = grho_refh
    dlam_refh = drho * refh * f1 * jnp.cos(phi / 2) / jnp.sqrt(grho_n_refh)

    # Convert to numpy for BatchResult
    return BatchResult(
        drho=np.asarray(drho),
        grho_refh=np.asarray(grho_refh),
        phi=np.asarray(phi),
        dlam_refh=np.asarray(dlam_refh),
        success=np.asarray(success, dtype=bool),
        messages=[f"Processed on {backend}" for _ in range(len(delfstars))],
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
    "batch_analyze_vmap",  # T044: Export batch_analyze_vmap
    # JAX configuration
    "configure_jax",
    "get_jax_backend",
    "is_gpu_available",
    # Model class and result types
    "QCMModel",
    "BatchResult",  # T044: Export BatchResult
    "SolveResult",
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
