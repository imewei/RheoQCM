"""Uncertainty calculation module for NLSQ fit results.

This module provides error propagation and confidence band calculation
using JAX autodiff for Jacobian computation.

Public API
----------
Classes:
    UncertaintyBand - Dataclass holding confidence band data
    UncertaintyCalculator - Main class for uncertainty band computation

Functions:
    regularize_covariance - Tikhonov regularization for singular covariance matrices
    check_degrees_of_freedom - Validate DOF for statistical inference
"""

from __future__ import annotations

__all__ = [
    "UncertaintyBand",
    "UncertaintyCalculator",
    "regularize_covariance",
    "check_degrees_of_freedom",
]

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from scipy import stats

# Type aliases
type Float64Array = npt.NDArray[np.float64]
type ModelFunc = Callable[..., Float64Array]


def regularize_covariance(pcov: Float64Array, eps: float = 1e-10) -> Float64Array:
    """Regularize singular or near-singular covariance matrix.

    Applies Tikhonov regularization by adding small diagonal to stabilize
    matrix inversion for error propagation.

    Parameters
    ----------
    pcov : Float64Array
        Parameter covariance matrix from curve fitting
    eps : float
        Regularization strength (default: 1e-10)

    Returns
    -------
    Float64Array
        Regularized covariance matrix

    Notes
    -----
    Emits warning if regularization is applied due to:
    - Condition number > 1e12 (near-singular)
    - LinAlgError during condition number computation (singular)
    """
    pcov = np.asarray(pcov, dtype=np.float64)

    try:
        cond = np.linalg.cond(pcov)
        if cond > 1e12:
            warnings.warn(
                f"Covariance matrix is near-singular (condition={cond:.2e}). "
                "Uncertainty estimates may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            return pcov + eps * np.eye(pcov.shape[0])
    except np.linalg.LinAlgError:
        warnings.warn(
            "Covariance matrix is singular. Adding regularization.",
            UserWarning,
            stacklevel=2,
        )
        return pcov + eps * np.eye(pcov.shape[0])

    return pcov


def check_degrees_of_freedom(
    n_data: int,
    n_params: int,
    threshold: int = 10,
) -> None:
    """Emit warning if degrees of freedom are too low.

    Low DOF makes confidence intervals unreliable.

    Parameters
    ----------
    n_data : int
        Number of data points
    n_params : int
        Number of model parameters
    threshold : int
        Minimum recommended n_data (default: 10)

    Notes
    -----
    Emits UserWarning if n_data < threshold.
    """
    if n_data < threshold:
        warnings.warn(
            f"Low sample size (n={n_data}). Uncertainty estimates may be unreliable. "
            f"Recommend n>={threshold} for {n_params}-parameter models.",
            UserWarning,
            stacklevel=2,
        )


@dataclass(frozen=True)
class UncertaintyBand:
    """Represents computed uncertainty band at prediction points.

    Attributes
    ----------
    x : Float64Array
        Prediction x-values
    y_fit : Float64Array
        Fitted y-values at x
    y_lower : Float64Array
        Lower bound of confidence interval
    y_upper : Float64Array
        Upper bound of confidence interval
    std : Float64Array
        Standard deviation at each x
    confidence_level : float
        Confidence level (0-1), default 0.95
    """

    x: Float64Array
    y_fit: Float64Array
    y_lower: Float64Array
    y_upper: Float64Array
    std: Float64Array
    confidence_level: float

    def __post_init__(self) -> None:
        """Validate array shapes and confidence level."""
        n = len(self.x)
        if not all(
            len(arr) == n for arr in [self.y_fit, self.y_lower, self.y_upper, self.std]
        ):
            msg = "All arrays must have the same length"
            raise ValueError(msg)

        if not 0 < self.confidence_level < 1:
            msg = f"confidence_level must be in (0, 1), got {self.confidence_level}"
            raise ValueError(msg)


class UncertaintyCalculator:
    """Compute uncertainty bands using error propagation.

    Uses JAX autodiff for Jacobian computation with finite-difference fallback.

    Parameters
    ----------
    use_autodiff : bool
        Use JAX autodiff if True, finite differences if False (default: True)
    eps : float
        Finite difference epsilon (default: 1e-8)

    Examples
    --------
    >>> import numpy as np
    >>> from rheoQCM.core.uncertainty import UncertaintyCalculator
    >>> def model(x, a, b): return a * np.exp(-b * x)
    >>> calc = UncertaintyCalculator()
    >>> band = calc.compute_band(model, np.linspace(0, 5, 50),
    ...                          np.array([1.0, 0.5]),
    ...                          np.diag([0.01, 0.001]))
    """

    def __init__(self, use_autodiff: bool = True, eps: float = 1e-8) -> None:
        self.use_autodiff = use_autodiff
        self.eps = eps

    def compute_jacobian(
        self,
        model: ModelFunc,
        x: Float64Array,
        params: Float64Array,
    ) -> Float64Array:
        """Compute Jacobian matrix J[i,j] = ∂y[i]/∂p[j].

        Parameters
        ----------
        model : ModelFunc
            Model function with signature model(x, *params) -> y
        x : Float64Array
            X-values (shape: [n_points])
        params : Float64Array
            Parameter values (shape: [n_params])

        Returns
        -------
        Float64Array
            Jacobian matrix (shape: [n_points, n_params])

        Notes
        -----
        Attempts JAX autodiff first if use_autodiff=True.
        Falls back to central finite differences if autodiff fails.
        """
        x = np.asarray(x, dtype=np.float64)
        params = np.asarray(params, dtype=np.float64)

        if self.use_autodiff:
            try:
                return self._jacobian_autodiff(model, x, params)
            except Exception:
                pass

        return self._jacobian_finite_diff(model, x, params)

    def _jacobian_autodiff(
        self,
        model: ModelFunc,
        x: Float64Array,
        params: Float64Array,
    ) -> Float64Array:
        """Compute Jacobian using JAX autodiff."""
        x_jax = jnp.asarray(x)
        params_jax = jnp.asarray(params)

        def model_at_x(p: jax.Array) -> jax.Array:
            return jnp.asarray(model(x_jax, *p))

        jac = jax.jacfwd(model_at_x)(params_jax)
        return np.asarray(jac)

    def _jacobian_finite_diff(
        self,
        model: ModelFunc,
        x: Float64Array,
        params: Float64Array,
    ) -> Float64Array:
        """Compute Jacobian using central finite differences."""
        n_points = len(x)
        n_params = len(params)
        jacobian = np.zeros((n_points, n_params), dtype=np.float64)

        for j in range(n_params):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[j] += self.eps
            params_minus[j] -= self.eps

            y_plus = np.asarray(model(x, *params_plus))
            y_minus = np.asarray(model(x, *params_minus))
            jacobian[:, j] = (y_plus - y_minus) / (2 * self.eps)

        return jacobian

    def propagate_uncertainty(
        self,
        jacobian: Float64Array,
        pcov: Float64Array,
    ) -> Float64Array:
        """Compute prediction standard deviation using error propagation.

        Formula: σ_y² = diag(J @ pcov @ Jᵀ)

        Parameters
        ----------
        jacobian : Float64Array
            Jacobian matrix (shape: [n_points, n_params])
        pcov : Float64Array
            Parameter covariance (shape: [n_params, n_params])

        Returns
        -------
        Float64Array
            Standard deviation array (shape: [n_points])
        """
        jacobian = np.asarray(jacobian, dtype=np.float64)
        pcov = np.asarray(pcov, dtype=np.float64)

        variance = np.einsum("ij,jk,ik->i", jacobian, pcov, jacobian)
        return np.sqrt(np.maximum(variance, 0))

    def compute_band(
        self,
        model: ModelFunc,
        x: Float64Array,
        popt: Float64Array,
        pcov: Float64Array,
        *,
        confidence_level: float = 0.95,
    ) -> UncertaintyBand:
        """Compute uncertainty band around fitted curve.

        Parameters
        ----------
        model : ModelFunc
            Model function with signature model(x, *params) -> y
        x : Float64Array
            X-values for prediction (shape: [n_points])
        popt : Float64Array
            Optimal parameter values (shape: [n_params])
        pcov : Float64Array
            Parameter covariance matrix (shape: [n_params, n_params])
        confidence_level : float
            Confidence level for interval (default: 0.95)

        Returns
        -------
        UncertaintyBand
            Dataclass with x, y_fit, y_lower, y_upper, std, confidence_level
        """
        x = np.asarray(x, dtype=np.float64)
        popt = np.asarray(popt, dtype=np.float64)
        pcov = np.asarray(pcov, dtype=np.float64)

        pcov = regularize_covariance(pcov)
        check_degrees_of_freedom(len(x), len(popt))

        y_fit = np.asarray(model(x, *popt))

        jacobian = self.compute_jacobian(model, x, popt)
        std = self.propagate_uncertainty(jacobian, pcov)

        z = stats.norm.ppf((1 + confidence_level) / 2)
        y_lower = y_fit - z * std
        y_upper = y_fit + z * std

        return UncertaintyBand(
            x=x,
            y_fit=y_fit,
            y_lower=y_lower,
            y_upper=y_upper,
            std=std,
            confidence_level=confidence_level,
        )
