"""Input validation utilities for RheoQCM.

Feature: Technical Guidelines Compliance - Data Integrity

This module provides validation decorators and utilities for ensuring
data integrity at public API boundaries.

Public API
----------
Decorators:
    validate_input - Decorator for validating function inputs

Classes:
    ValidationError - Base exception for validation failures
    ShapeError - Shape mismatch validation error
    DTypeError - Data type validation error
    RangeError - Value range validation error
    MonotonicityError - Monotonicity constraint violation

Functions:
    check_shape - Validate array shape
    check_dtype - Validate array dtype
    check_range - Validate value range
    check_monotonic - Validate monotonicity
    check_no_nan - Validate no NaN values
    check_finite - Validate finite values
"""

from __future__ import annotations

__all__ = [
    "ValidationError",
    "ShapeError",
    "DTypeError",
    "RangeError",
    "MonotonicityError",
    "validate_input",
    "check_shape",
    "check_dtype",
    "check_range",
    "check_monotonic",
    "check_no_nan",
    "check_finite",
]

import functools
import logging
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")
ArrayLike = npt.ArrayLike
Float64Array = npt.NDArray[np.float64]


# =============================================================================
# Exception Hierarchy
# =============================================================================


class ValidationError(ValueError):
    """Base exception for validation failures.

    Attributes
    ----------
    param_name : str
        Name of the parameter that failed validation.
    expected : str
        Description of expected value.
    actual : str
        Description of actual value.
    """

    def __init__(
        self,
        message: str,
        param_name: str = "",
        expected: str = "",
        actual: str = "",
    ) -> None:
        super().__init__(message)
        self.param_name = param_name
        self.expected = expected
        self.actual = actual


class ShapeError(ValidationError):
    """Shape mismatch validation error."""

    pass


class DTypeError(ValidationError):
    """Data type validation error."""

    pass


class RangeError(ValidationError):
    """Value range validation error."""

    pass


class MonotonicityError(ValidationError):
    """Monotonicity constraint violation."""

    pass


# =============================================================================
# Validation Functions
# =============================================================================


def check_shape(
    arr: ArrayLike,
    expected: tuple[int | None, ...],
    name: str = "array",
) -> np.ndarray:
    """Validate array shape.

    Parameters
    ----------
    arr : ArrayLike
        Input array to validate.
    expected : tuple[int | None, ...]
        Expected shape. Use None for dimensions that can be any size.
    name : str
        Parameter name for error messages.

    Returns
    -------
    np.ndarray
        Validated array.

    Raises
    ------
    ShapeError
        If shape doesn't match expected.

    Examples
    --------
    >>> arr = np.array([[1, 2], [3, 4]])
    >>> check_shape(arr, (2, 2), "matrix")  # OK
    >>> check_shape(arr, (None, 2), "matrix")  # OK - any rows, 2 cols
    >>> check_shape(arr, (3, 2), "matrix")  # Raises ShapeError
    """
    arr = np.asarray(arr)
    actual = arr.shape

    if len(actual) != len(expected):
        raise ShapeError(
            f"{name} has wrong number of dimensions: "
            f"expected {len(expected)}, got {len(actual)}",
            param_name=name,
            expected=str(expected),
            actual=str(actual),
        )

    for i, (act, exp) in enumerate(zip(actual, expected, strict=True)):
        if exp is not None and act != exp:
            raise ShapeError(
                f"{name} has wrong shape at dimension {i}: expected {exp}, got {act}",
                param_name=name,
                expected=str(expected),
                actual=str(actual),
            )

    return arr


def check_dtype(
    arr: ArrayLike,
    expected: type | tuple[type, ...] | np.dtype | str,
    name: str = "array",
    coerce: bool = False,
) -> np.ndarray:
    """Validate array dtype.

    Parameters
    ----------
    arr : ArrayLike
        Input array to validate.
    expected : type | tuple[type, ...] | np.dtype | str
        Expected dtype(s). Can be numpy dtype, string, or tuple.
    name : str
        Parameter name for error messages.
    coerce : bool
        If True, attempt to convert to expected dtype instead of raising.

    Returns
    -------
    np.ndarray
        Validated array (possibly coerced).

    Raises
    ------
    DTypeError
        If dtype doesn't match and coerce=False.

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, 3.0])
    >>> check_dtype(arr, np.float64, "data")  # OK
    >>> check_dtype(arr, (np.float32, np.float64), "data")  # OK
    >>> check_dtype(np.array([1, 2, 3]), np.float64, "data", coerce=True)
    """
    arr = np.asarray(arr)

    if isinstance(expected, tuple):
        expected_dtypes = [np.dtype(d) for d in expected]
    else:
        expected_dtypes = [np.dtype(expected)]

    if arr.dtype in expected_dtypes:
        return arr

    if coerce:
        try:
            return arr.astype(expected_dtypes[0])
        except (ValueError, TypeError) as e:
            raise DTypeError(
                f"Cannot coerce {name} from {arr.dtype} to {expected_dtypes[0]}: {e}",
                param_name=name,
                expected=str(expected_dtypes[0]),
                actual=str(arr.dtype),
            ) from e

    raise DTypeError(
        f"{name} has wrong dtype: expected {expected_dtypes}, got {arr.dtype}",
        param_name=name,
        expected=str(expected_dtypes),
        actual=str(arr.dtype),
    )


def check_range(
    arr: ArrayLike,
    low: float | None = None,
    high: float | None = None,
    name: str = "array",
    inclusive: tuple[bool, bool] = (True, True),
) -> np.ndarray:
    """Validate array values are within range.

    Parameters
    ----------
    arr : ArrayLike
        Input array to validate.
    low : float | None
        Lower bound (None for no lower bound).
    high : float | None
        Upper bound (None for no upper bound).
    name : str
        Parameter name for error messages.
    inclusive : tuple[bool, bool]
        Whether bounds are inclusive (low, high).

    Returns
    -------
    np.ndarray
        Validated array.

    Raises
    ------
    RangeError
        If any values are outside range.

    Examples
    --------
    >>> arr = np.array([0.0, 0.5, 1.0])
    >>> check_range(arr, 0, 1, "probability")  # OK
    >>> check_range(arr, 0, 1, "probability", inclusive=(True, False))
    """
    arr = np.asarray(arr)

    if low is not None:
        if inclusive[0]:
            if np.any(arr < low):
                min_val = float(np.min(arr))
                raise RangeError(
                    f"{name} has values below minimum {low}: min={min_val:.6g}",
                    param_name=name,
                    expected=f">= {low}",
                    actual=f"min = {min_val:.6g}",
                )
        else:
            if np.any(arr <= low):
                min_val = float(np.min(arr))
                raise RangeError(
                    f"{name} has values at or below minimum {low}: min={min_val:.6g}",
                    param_name=name,
                    expected=f"> {low}",
                    actual=f"min = {min_val:.6g}",
                )

    if high is not None:
        if inclusive[1]:
            if np.any(arr > high):
                max_val = float(np.max(arr))
                raise RangeError(
                    f"{name} has values above maximum {high}: max={max_val:.6g}",
                    param_name=name,
                    expected=f"<= {high}",
                    actual=f"max = {max_val:.6g}",
                )
        else:
            if np.any(arr >= high):
                max_val = float(np.max(arr))
                raise RangeError(
                    f"{name} has values at or above maximum {high}: max={max_val:.6g}",
                    param_name=name,
                    expected=f"< {high}",
                    actual=f"max = {max_val:.6g}",
                )

    return arr


def check_monotonic(
    arr: ArrayLike,
    direction: str = "increasing",
    strict: bool = False,
    name: str = "array",
) -> np.ndarray:
    """Validate array is monotonic.

    Parameters
    ----------
    arr : ArrayLike
        Input array to validate (1D).
    direction : str
        Expected direction: "increasing" or "decreasing".
    strict : bool
        If True, require strictly monotonic (no equal consecutive values).
    name : str
        Parameter name for error messages.

    Returns
    -------
    np.ndarray
        Validated array.

    Raises
    ------
    MonotonicityError
        If array is not monotonic in expected direction.
    ShapeError
        If array is not 1D.

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, 3.0, 4.0])
    >>> check_monotonic(arr, "increasing", name="frequency")  # OK
    >>> check_monotonic(arr[::-1], "decreasing", name="temperature")  # OK
    """
    arr = np.asarray(arr)

    if arr.ndim != 1:
        raise ShapeError(
            f"{name} must be 1D for monotonicity check, got {arr.ndim}D",
            param_name=name,
            expected="1D",
            actual=f"{arr.ndim}D",
        )

    if len(arr) < 2:
        return arr  # Trivially monotonic

    diff = np.diff(arr)

    if direction == "increasing":
        if strict:
            if not np.all(diff > 0):
                raise MonotonicityError(
                    f"{name} is not strictly increasing",
                    param_name=name,
                    expected="strictly increasing",
                    actual="non-monotonic or has repeated values",
                )
        else:
            if not np.all(diff >= 0):
                raise MonotonicityError(
                    f"{name} is not monotonically increasing",
                    param_name=name,
                    expected="monotonically increasing",
                    actual="has decreasing segments",
                )
    elif direction == "decreasing":
        if strict:
            if not np.all(diff < 0):
                raise MonotonicityError(
                    f"{name} is not strictly decreasing",
                    param_name=name,
                    expected="strictly decreasing",
                    actual="non-monotonic or has repeated values",
                )
        else:
            if not np.all(diff <= 0):
                raise MonotonicityError(
                    f"{name} is not monotonically decreasing",
                    param_name=name,
                    expected="monotonically decreasing",
                    actual="has increasing segments",
                )
    else:
        raise ValueError(
            f"direction must be 'increasing' or 'decreasing', got {direction!r}"
        )

    return arr


def check_no_nan(arr: ArrayLike, name: str = "array") -> np.ndarray:
    """Validate array contains no NaN values.

    Parameters
    ----------
    arr : ArrayLike
        Input array to validate.
    name : str
        Parameter name for error messages.

    Returns
    -------
    np.ndarray
        Validated array.

    Raises
    ------
    ValidationError
        If array contains NaN values.
    """
    arr = np.asarray(arr)

    if np.any(np.isnan(arr)):
        nan_count = int(np.sum(np.isnan(arr)))
        raise ValidationError(
            f"{name} contains {nan_count} NaN value(s)",
            param_name=name,
            expected="no NaN values",
            actual=f"{nan_count} NaN(s)",
        )

    return arr


def check_finite(arr: ArrayLike, name: str = "array") -> np.ndarray:
    """Validate array contains only finite values (no NaN or Inf).

    Parameters
    ----------
    arr : ArrayLike
        Input array to validate.
    name : str
        Parameter name for error messages.

    Returns
    -------
    np.ndarray
        Validated array.

    Raises
    ------
    ValidationError
        If array contains non-finite values.
    """
    arr = np.asarray(arr)

    if not np.all(np.isfinite(arr)):
        nan_count = int(np.sum(np.isnan(arr)))
        inf_count = int(np.sum(np.isinf(arr)))
        raise ValidationError(
            f"{name} contains non-finite values: {nan_count} NaN, {inf_count} Inf",
            param_name=name,
            expected="all finite values",
            actual=f"{nan_count} NaN, {inf_count} Inf",
        )

    return arr


# =============================================================================
# Validation Decorator
# =============================================================================


class ValidatorSpec:
    """Specification for a parameter validation rule.

    Attributes
    ----------
    check : str
        Validation function name (shape, dtype, range, monotonic, no_nan, finite).
    kwargs : dict
        Arguments to pass to validation function.
    """

    def __init__(self, check: str, **kwargs: Any) -> None:
        self.check = check
        self.kwargs = kwargs


def validate_input(
    **validators: ValidatorSpec | Sequence[ValidatorSpec],
) -> Callable[[T], T]:
    """Decorator for validating function inputs.

    Applies validation functions to named parameters before
    the decorated function executes.

    Parameters
    ----------
    **validators : ValidatorSpec | Sequence[ValidatorSpec]
        Validation specifications keyed by parameter name.
        Each value can be a single ValidatorSpec or a sequence.

    Returns
    -------
    Callable
        Decorator function.

    Examples
    --------
    >>> from rheoQCM.core.validation import validate_input, ValidatorSpec
    >>>
    >>> @validate_input(
    ...     frequency=ValidatorSpec("monotonic", direction="increasing", strict=True),
    ...     data=[
    ...         ValidatorSpec("dtype", expected=np.float64, coerce=True),
    ...         ValidatorSpec("finite"),
    ...     ],
    ... )
    ... def analyze(frequency, data):
    ...     return np.mean(data)
    >>>
    >>> analyze(np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]))  # OK
    >>> analyze(np.array([3, 2, 1]), np.array([1.0, 2.0, 3.0]))  # MonotonicityError
    """
    validation_funcs = {
        "shape": check_shape,
        "dtype": check_dtype,
        "range": check_range,
        "monotonic": check_monotonic,
        "no_nan": check_no_nan,
        "finite": check_finite,
    }

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            import inspect

            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for param_name, specs in validators.items():
                if param_name not in bound.arguments:
                    continue

                value = bound.arguments[param_name]
                if value is None:
                    continue

                # Normalize to list
                spec_list = (
                    specs
                    if isinstance(specs, Sequence) and not isinstance(specs, str)
                    else [specs]
                )

                for spec in spec_list:
                    check_func = validation_funcs.get(spec.check)
                    if check_func is None:
                        raise ValueError(f"Unknown validation check: {spec.check}")

                    try:
                        validated = check_func(value, name=param_name, **spec.kwargs)
                        bound.arguments[param_name] = validated
                        value = validated
                    except ValidationError:
                        raise
                    except Exception as e:
                        logger.error("Validation failed for %s: %s", param_name, e)
                        raise ValidationError(
                            f"Validation of {param_name} failed: {e}",
                            param_name=param_name,
                        ) from e

            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator
