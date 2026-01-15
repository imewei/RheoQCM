"""Unit tests for validation module.

Tests cover:
- ValidationError hierarchy
- check_shape function
- check_dtype function
- check_range function
- check_monotonic function
- check_no_nan function
- check_finite function
- @validate_input decorator
"""

from __future__ import annotations

import numpy as np
import pytest

from rheoQCM.core.validation import (
    DTypeError,
    MonotonicityError,
    RangeError,
    ShapeError,
    ValidationError,
    ValidatorSpec,
    check_dtype,
    check_finite,
    check_monotonic,
    check_no_nan,
    check_range,
    check_shape,
    validate_input,
)


class TestValidationErrorHierarchy:
    """Test ValidationError exception hierarchy."""

    def test_validation_error_is_value_error(self) -> None:
        """ValidationError should inherit from ValueError."""
        assert issubclass(ValidationError, ValueError)

    def test_shape_error_is_validation_error(self) -> None:
        """ShapeError should inherit from ValidationError."""
        assert issubclass(ShapeError, ValidationError)

    def test_dtype_error_is_validation_error(self) -> None:
        """DTypeError should inherit from ValidationError."""
        assert issubclass(DTypeError, ValidationError)

    def test_range_error_is_validation_error(self) -> None:
        """RangeError should inherit from ValidationError."""
        assert issubclass(RangeError, ValidationError)

    def test_monotonicity_error_is_validation_error(self) -> None:
        """MonotonicityError should inherit from ValidationError."""
        assert issubclass(MonotonicityError, ValidationError)

    def test_validation_error_attributes(self) -> None:
        """ValidationError should store param_name, expected, actual."""
        err = ValidationError(
            "test message",
            param_name="x",
            expected="int",
            actual="str",
        )

        assert err.param_name == "x"
        assert err.expected == "int"
        assert err.actual == "str"


class TestCheckShape:
    """Test check_shape function."""

    def test_valid_exact_shape(self) -> None:
        """check_shape should pass for exact shape match."""
        arr = np.array([[1, 2], [3, 4]])
        result = check_shape(arr, (2, 2), "matrix")

        assert result.shape == (2, 2)

    def test_valid_wildcard_dimension(self) -> None:
        """check_shape should pass for None (wildcard) dimension."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        result = check_shape(arr, (None, 2), "matrix")

        assert result.shape == (3, 2)

    def test_invalid_dimension_count(self) -> None:
        """check_shape should raise ShapeError for wrong ndim."""
        arr = np.array([1, 2, 3])

        with pytest.raises(ShapeError, match="wrong number of dimensions"):
            check_shape(arr, (1, 3), "vector")

    def test_invalid_dimension_size(self) -> None:
        """check_shape should raise ShapeError for wrong size."""
        arr = np.array([[1, 2], [3, 4]])

        with pytest.raises(ShapeError, match="wrong shape at dimension"):
            check_shape(arr, (3, 2), "matrix")

    def test_error_includes_param_name(self) -> None:
        """ShapeError should include parameter name."""
        arr = np.array([1, 2, 3])

        try:
            check_shape(arr, (5,), "my_param")
        except ShapeError as e:
            assert e.param_name == "my_param"


class TestCheckDtype:
    """Test check_dtype function."""

    def test_valid_dtype(self) -> None:
        """check_dtype should pass for matching dtype."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = check_dtype(arr, np.float64, "data")

        assert result.dtype == np.float64

    def test_valid_dtype_tuple(self) -> None:
        """check_dtype should accept tuple of valid dtypes."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = check_dtype(arr, (np.float32, np.float64), "data")

        assert result.dtype == np.float32

    def test_invalid_dtype_raises(self) -> None:
        """check_dtype should raise DTypeError for wrong dtype."""
        arr = np.array([1, 2, 3], dtype=np.int32)

        with pytest.raises(DTypeError, match="wrong dtype"):
            check_dtype(arr, np.float64, "data")

    def test_coerce_dtype(self) -> None:
        """check_dtype with coerce=True should convert dtype."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = check_dtype(arr, np.float64, "data", coerce=True)

        assert result.dtype == np.float64

    def test_coerce_failure_raises(self) -> None:
        """check_dtype coercion failure should raise DTypeError."""
        arr = np.array(["a", "b", "c"])

        with pytest.raises(DTypeError, match="Cannot coerce"):
            check_dtype(arr, np.float64, "data", coerce=True)


class TestCheckRange:
    """Test check_range function."""

    def test_valid_range(self) -> None:
        """check_range should pass for values in range."""
        arr = np.array([0.0, 0.5, 1.0])
        result = check_range(arr, 0.0, 1.0, "probability")

        assert np.allclose(result, arr)

    def test_below_lower_bound_raises(self) -> None:
        """check_range should raise RangeError for values below min."""
        arr = np.array([-0.1, 0.5, 1.0])

        with pytest.raises(RangeError, match="below minimum"):
            check_range(arr, 0.0, 1.0, "probability")

    def test_above_upper_bound_raises(self) -> None:
        """check_range should raise RangeError for values above max."""
        arr = np.array([0.0, 0.5, 1.1])

        with pytest.raises(RangeError, match="above maximum"):
            check_range(arr, 0.0, 1.0, "probability")

    def test_exclusive_bounds(self) -> None:
        """check_range with inclusive=False should exclude boundaries."""
        arr = np.array([0.0, 0.5, 1.0])

        # Lower bound exclusive
        with pytest.raises(RangeError):
            check_range(arr, 0.0, None, "data", inclusive=(False, True))

    def test_none_bounds_no_check(self) -> None:
        """check_range with None bounds should skip that check."""
        arr = np.array([-1000, 0, 1000])
        result = check_range(arr, None, None, "data")

        assert np.allclose(result, arr)


class TestCheckMonotonic:
    """Test check_monotonic function."""

    def test_increasing_valid(self) -> None:
        """check_monotonic should pass for increasing array."""
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = check_monotonic(arr, "increasing", name="frequency")

        assert np.allclose(result, arr)

    def test_decreasing_valid(self) -> None:
        """check_monotonic should pass for decreasing array."""
        arr = np.array([4.0, 3.0, 2.0, 1.0])
        result = check_monotonic(arr, "decreasing", name="temperature")

        assert np.allclose(result, arr)

    def test_not_increasing_raises(self) -> None:
        """check_monotonic should raise for non-increasing array."""
        arr = np.array([1.0, 3.0, 2.0, 4.0])

        with pytest.raises(MonotonicityError, match="not monotonically increasing"):
            check_monotonic(arr, "increasing", name="frequency")

    def test_strict_increasing(self) -> None:
        """check_monotonic strict=True should reject equal values."""
        arr = np.array([1.0, 2.0, 2.0, 3.0])

        with pytest.raises(MonotonicityError, match="not strictly increasing"):
            check_monotonic(arr, "increasing", strict=True, name="frequency")

    def test_non_strict_allows_equal(self) -> None:
        """check_monotonic strict=False should allow equal values."""
        arr = np.array([1.0, 2.0, 2.0, 3.0])
        result = check_monotonic(arr, "increasing", strict=False, name="frequency")

        assert np.allclose(result, arr)

    def test_non_1d_raises_shape_error(self) -> None:
        """check_monotonic should raise ShapeError for non-1D arrays."""
        arr = np.array([[1, 2], [3, 4]])

        with pytest.raises(ShapeError, match="must be 1D"):
            check_monotonic(arr, "increasing", name="data")

    def test_single_element_trivially_monotonic(self) -> None:
        """check_monotonic should pass for single-element array."""
        arr = np.array([5.0])
        result = check_monotonic(arr, "increasing", name="data")

        assert result[0] == 5.0


class TestCheckNoNan:
    """Test check_no_nan function."""

    def test_no_nan_valid(self) -> None:
        """check_no_nan should pass for arrays without NaN."""
        arr = np.array([1.0, 2.0, 3.0])
        result = check_no_nan(arr, "data")

        assert np.allclose(result, arr)

    def test_nan_raises(self) -> None:
        """check_no_nan should raise ValidationError for NaN values."""
        arr = np.array([1.0, np.nan, 3.0])

        with pytest.raises(ValidationError, match="contains.*NaN"):
            check_no_nan(arr, "data")


class TestCheckFinite:
    """Test check_finite function."""

    def test_finite_valid(self) -> None:
        """check_finite should pass for finite arrays."""
        arr = np.array([1.0, 2.0, 3.0])
        result = check_finite(arr, "data")

        assert np.allclose(result, arr)

    def test_nan_raises(self) -> None:
        """check_finite should raise ValidationError for NaN."""
        arr = np.array([1.0, np.nan, 3.0])

        with pytest.raises(ValidationError, match="non-finite"):
            check_finite(arr, "data")

    def test_inf_raises(self) -> None:
        """check_finite should raise ValidationError for Inf."""
        arr = np.array([1.0, np.inf, 3.0])

        with pytest.raises(ValidationError, match="non-finite"):
            check_finite(arr, "data")

    def test_negative_inf_raises(self) -> None:
        """check_finite should raise ValidationError for -Inf."""
        arr = np.array([1.0, -np.inf, 3.0])

        with pytest.raises(ValidationError, match="non-finite"):
            check_finite(arr, "data")


class TestValidatorSpec:
    """Test ValidatorSpec class."""

    def test_validator_spec_creation(self) -> None:
        """ValidatorSpec should store check name and kwargs."""
        spec = ValidatorSpec("range", low=0.0, high=1.0)

        assert spec.check == "range"
        assert spec.kwargs["low"] == 0.0
        assert spec.kwargs["high"] == 1.0


class TestValidateInputDecorator:
    """Test @validate_input decorator."""

    def test_decorator_validates_input(self) -> None:
        """@validate_input should validate specified parameters."""

        @validate_input(
            data=ValidatorSpec("finite"),
        )
        def process(data: np.ndarray) -> float:
            return float(np.mean(data))

        result = process(np.array([1.0, 2.0, 3.0]))
        assert result == 2.0

    def test_decorator_raises_on_invalid(self) -> None:
        """@validate_input should raise ValidationError for invalid input."""

        @validate_input(
            data=ValidatorSpec("finite"),
        )
        def process(data: np.ndarray) -> float:
            return float(np.mean(data))

        with pytest.raises(ValidationError):
            process(np.array([1.0, np.nan, 3.0]))

    def test_decorator_multiple_validators(self) -> None:
        """@validate_input should apply multiple validators to one param."""

        @validate_input(
            data=[
                ValidatorSpec("dtype", expected=np.float64, coerce=True),
                ValidatorSpec("finite"),
            ],
        )
        def process(data: np.ndarray) -> float:
            return float(np.mean(data))

        # Integer input should be coerced to float64
        result = process(np.array([1, 2, 3]))
        assert result == 2.0

    def test_decorator_multiple_params(self) -> None:
        """@validate_input should validate multiple parameters."""

        @validate_input(
            x=ValidatorSpec("monotonic", direction="increasing"),
            y=ValidatorSpec("shape", expected=(None,)),
        )
        def fit(x: np.ndarray, y: np.ndarray) -> float:
            return float(np.corrcoef(x, y)[0, 1])

        result = fit(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
        )
        assert result == pytest.approx(1.0)

    def test_decorator_allows_none(self) -> None:
        """@validate_input should skip validation for None values."""

        @validate_input(
            optional=ValidatorSpec("finite"),
        )
        def process(data: np.ndarray, optional: np.ndarray | None = None) -> float:
            if optional is not None:
                return float(np.mean(optional))
            return float(np.mean(data))

        # Should not raise for None optional
        result = process(np.array([1.0, 2.0, 3.0]), optional=None)
        assert result == 2.0

    def test_decorator_preserves_function_metadata(self) -> None:
        """@validate_input should preserve function name and docstring."""

        @validate_input(data=ValidatorSpec("finite"))
        def my_function(data: np.ndarray) -> float:
            """My docstring."""
            return float(np.mean(data))

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_decorator_unknown_check_raises(self) -> None:
        """@validate_input with unknown check should raise ValueError."""

        @validate_input(data=ValidatorSpec("unknown_check"))
        def process(data: np.ndarray) -> float:
            return float(np.mean(data))

        with pytest.raises(ValueError, match="Unknown validation check"):
            process(np.array([1.0, 2.0, 3.0]))
