"""Unit tests for uncertainty calculation module.

Tests cover:
- UncertaintyBand dataclass validation
- Jacobian computation (JAX autodiff vs finite-diff)
- Error propagation accuracy
- Confidence band calculation
- Edge cases (singular pcov, low DOF)
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from rheoQCM.core.uncertainty import (
    Float64Array,
    UncertaintyBand,
    UncertaintyCalculator,
    check_degrees_of_freedom,
    regularize_covariance,
)


class TestUncertaintyBandDataclass:
    """T008: Unit test: UncertaintyBand dataclass validation."""

    def test_valid_uncertainty_band(self) -> None:
        """Valid UncertaintyBand should be created without error."""
        n = 50
        x = np.linspace(0, 10, n)
        y_fit = np.sin(x)
        std = np.ones(n) * 0.1
        y_lower = y_fit - 1.96 * std
        y_upper = y_fit + 1.96 * std

        band = UncertaintyBand(
            x=x,
            y_fit=y_fit,
            y_lower=y_lower,
            y_upper=y_upper,
            std=std,
            confidence_level=0.95,
        )

        assert len(band.x) == n
        assert band.confidence_level == 0.95
        np.testing.assert_array_equal(band.x, x)

    def test_array_length_mismatch_raises_error(self) -> None:
        """Mismatched array lengths should raise ValueError."""
        x = np.linspace(0, 10, 50)
        y_fit = np.sin(x)
        std = np.ones(50) * 0.1
        y_lower = y_fit - 1.96 * std
        y_upper_wrong = np.zeros(40)  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            UncertaintyBand(
                x=x,
                y_fit=y_fit,
                y_lower=y_lower,
                y_upper=y_upper_wrong,
                std=std,
                confidence_level=0.95,
            )

    def test_invalid_confidence_level_raises_error(self) -> None:
        """Confidence level outside (0, 1) should raise ValueError."""
        n = 10
        x = np.linspace(0, 10, n)
        y_fit = np.zeros(n)
        std = np.ones(n)

        with pytest.raises(ValueError, match="confidence_level"):
            UncertaintyBand(
                x=x,
                y_fit=y_fit,
                y_lower=y_fit - std,
                y_upper=y_fit + std,
                std=std,
                confidence_level=1.5,  # Invalid
            )

        with pytest.raises(ValueError, match="confidence_level"):
            UncertaintyBand(
                x=x,
                y_fit=y_fit,
                y_lower=y_fit - std,
                y_upper=y_fit + std,
                std=std,
                confidence_level=0.0,  # Invalid
            )


class TestComputeJacobian:
    """T009: Unit test: compute_jacobian() JAX autodiff vs finite-diff."""

    def test_jacobian_linear_model(self) -> None:
        """Jacobian of linear model y = a*x + b should match analytical."""

        def linear(x: Float64Array, a: float, b: float) -> Float64Array:
            return a * x + b

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        params = np.array([2.0, 1.0])  # a=2, b=1

        calc = UncertaintyCalculator(use_autodiff=True)
        jac = calc.compute_jacobian(linear, x, params)

        # Analytical Jacobian: J[:, 0] = x (∂y/∂a), J[:, 1] = 1 (∂y/∂b)
        expected = np.column_stack([x, np.ones_like(x)])

        np.testing.assert_allclose(jac, expected, rtol=1e-5)

    def test_jacobian_exponential_model(self) -> None:
        """Jacobian of exponential model should match finite-diff."""

        def exponential(x: Float64Array, a: float, b: float, c: float) -> Float64Array:
            return a * np.exp(-b * x) + c

        x = np.linspace(0, 5, 20)
        params = np.array([1.0, 0.5, 0.1])

        calc_autodiff = UncertaintyCalculator(use_autodiff=True)
        calc_finite = UncertaintyCalculator(use_autodiff=False)

        jac_autodiff = calc_autodiff.compute_jacobian(exponential, x, params)
        jac_finite = calc_finite.compute_jacobian(exponential, x, params)

        np.testing.assert_allclose(jac_autodiff, jac_finite, rtol=1e-4)

    def test_jacobian_shape(self) -> None:
        """Jacobian should have shape [n_points, n_params]."""

        def model(x: Float64Array, a: float, b: float, c: float) -> Float64Array:
            return a * x**2 + b * x + c

        x = np.linspace(0, 10, 100)
        params = np.array([1.0, 2.0, 3.0])

        calc = UncertaintyCalculator()
        jac = calc.compute_jacobian(model, x, params)

        assert jac.shape == (100, 3)


class TestPropagateUncertainty:
    """T010: Unit test: propagate_uncertainty() matches analytical solution."""

    def test_linear_model_analytical(self) -> None:
        """Error propagation for linear model y = ax + b should match analytical.

        For y = ax + b:
        - J = [x, 1]
        - σ_y² = x² * σ_a² + σ_b² + 2 * x * cov(a,b)

        With diagonal pcov (no covariance):
        - σ_y² = x² * σ_a² + σ_b²
        """

        def linear(x: Float64Array, a: float, b: float) -> Float64Array:
            return a * x + b

        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        params = np.array([2.0, 1.0])
        pcov = np.array([[0.01, 0.0], [0.0, 0.04]])  # σ_a = 0.1, σ_b = 0.2

        calc = UncertaintyCalculator()
        jacobian = calc.compute_jacobian(linear, x, params)
        std = calc.propagate_uncertainty(jacobian, pcov)

        # Analytical: σ_y = sqrt(x² * 0.01 + 0.04)
        expected = np.sqrt(x**2 * 0.01 + 0.04)

        # Within 1% of analytical solution (SC-004)
        np.testing.assert_allclose(std, expected, rtol=0.01)

    def test_einsum_efficiency(self) -> None:
        """Verify einsum formula matches explicit matrix multiplication."""
        jacobian = np.random.randn(100, 3)
        pcov = np.eye(3) * 0.1

        calc = UncertaintyCalculator()
        std = calc.propagate_uncertainty(jacobian, pcov)

        # Explicit: diag(J @ pcov @ J.T)
        variance_explicit = np.diag(jacobian @ pcov @ jacobian.T)
        expected = np.sqrt(np.maximum(variance_explicit, 0))

        np.testing.assert_allclose(std, expected, rtol=1e-10)


class TestComputeBand:
    """T011: Unit test: compute_band() with configurable confidence levels."""

    def test_confidence_levels(self) -> None:
        """Different confidence levels should produce different band widths."""

        def model(x: Float64Array, a: float, b: float) -> Float64Array:
            return a * np.exp(-b * x)

        x = np.linspace(0, 5, 50)
        popt = np.array([1.0, 0.5])
        pcov = np.diag([0.01, 0.001])

        calc = UncertaintyCalculator()

        band_90 = calc.compute_band(model, x, popt, pcov, confidence_level=0.90)
        band_95 = calc.compute_band(model, x, popt, pcov, confidence_level=0.95)
        band_99 = calc.compute_band(model, x, popt, pcov, confidence_level=0.99)

        # Verify widths increase with confidence level
        width_90 = np.mean(band_90.y_upper - band_90.y_lower)
        width_95 = np.mean(band_95.y_upper - band_95.y_lower)
        width_99 = np.mean(band_99.y_upper - band_99.y_lower)

        assert width_90 < width_95 < width_99

    def test_z_score_scaling(self) -> None:
        """Band width should scale with z-score for confidence level."""

        def model(x: Float64Array, a: float) -> Float64Array:
            return a * x

        x = np.array([1.0])
        popt = np.array([1.0])
        pcov = np.array([[1.0]])  # σ_a = 1

        calc = UncertaintyCalculator()
        band = calc.compute_band(model, x, popt, pcov, confidence_level=0.95)

        # For x=1, y=a, so σ_y = σ_a = 1
        # z for 95% = 1.96
        expected_width = 2 * 1.96 * 1.0

        actual_width = band.y_upper[0] - band.y_lower[0]
        np.testing.assert_allclose(actual_width, expected_width, rtol=1e-3)


class TestSingularPcov:
    """T012: Unit test: Singular pcov triggers regularization and warning."""

    def test_near_singular_pcov_warning(self) -> None:
        """Near-singular pcov should emit warning and regularize."""
        # Create near-singular matrix
        pcov = np.array([[1.0, 1.0 - 1e-14], [1.0 - 1e-14, 1.0]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = regularize_covariance(pcov)

            # Should emit warning about near-singular
            assert len(w) == 1
            assert (
                "near-singular" in str(w[0].message).lower()
                or "singular" in str(w[0].message).lower()
            )

        # Result should be regularized (slightly larger diagonal)
        assert np.linalg.cond(result) < np.linalg.cond(pcov)

    def test_singular_pcov_regularization(self) -> None:
        """Truly singular pcov should be regularized."""
        # Singular matrix: rank 1
        v = np.array([[1.0], [2.0]])
        pcov = v @ v.T  # rank 1

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = regularize_covariance(pcov)

            # Should work without raising
            assert result is not None


class TestLowDOF:
    """T013: Unit test: Low DOF (n<10) triggers warning."""

    def test_low_dof_warning(self) -> None:
        """Should warn when n_data < threshold."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_degrees_of_freedom(n_data=5, n_params=3, threshold=10)

            assert len(w) == 1
            assert "Low sample size" in str(w[0].message)

    def test_sufficient_dof_no_warning(self) -> None:
        """Should not warn when n_data >= threshold."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_degrees_of_freedom(n_data=50, n_params=3, threshold=10)

            assert len(w) == 0

    def test_compute_band_low_dof_warning(self) -> None:
        """compute_band should warn on low DOF."""

        def model(x: Float64Array, a: float) -> Float64Array:
            return a * x

        x = np.array([1.0, 2.0, 3.0])  # n=3 < 10
        popt = np.array([1.0])
        pcov = np.array([[0.01]])

        calc = UncertaintyCalculator()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calc.compute_band(model, x, popt, pcov)

            # Should get low DOF warning
            low_dof_warnings = [x for x in w if "Low sample size" in str(x.message)]
            assert len(low_dof_warnings) >= 1


class TestJAXModel:
    """Additional tests for JAX-compatible models."""

    def test_jax_model_autodiff(self) -> None:
        """JAX-native model should work with autodiff."""

        def jax_model(x: Float64Array, a: float, b: float) -> Float64Array:
            return jnp.asarray(a) * jnp.exp(-jnp.asarray(b) * jnp.asarray(x))

        x = np.linspace(0, 5, 50)
        params = np.array([1.0, 0.5])

        calc = UncertaintyCalculator(use_autodiff=True)
        jac = calc.compute_jacobian(jax_model, x, params)

        assert jac.shape == (50, 2)
        assert np.all(np.isfinite(jac))

    def test_non_differentiable_model_fallback(self) -> None:
        """Non-differentiable model should fall back to finite diff."""

        def weird_model(x: Float64Array, a: float) -> Float64Array:
            # Contains non-differentiable operation
            return np.where(x > 2.5, a * x, a * x**2)

        x = np.linspace(0, 5, 50)
        params = np.array([1.0])

        calc = UncertaintyCalculator(use_autodiff=True)
        # Should not raise, should fall back to finite diff
        jac = calc.compute_jacobian(weird_model, x, params)

        assert jac.shape == (50, 1)
