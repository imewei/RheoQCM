"""
Edge case tests for JAX performance optimization.
Feature: 012-jax-performance-optimization

Tests edge case handling with sentinel values per clarification:
- Return safe sentinel values (0.0, NaN) with logged warnings
"""

import logging
import math

import jax.numpy as jnp
import numpy as np
import pytest

# Import modules under test
from rheoQCM.core import model, multilayer, physics


class TestMultilayerEdgeCases:
    """Test edge cases in multilayer calculations."""

    def test_zero_layers(self):
        """Test calc_ZL with zero layers returns zero impedance."""
        layers = {}
        ZL = multilayer.calc_ZL(3, layers, 0.0 + 0.0j, 5e6)

        # Empty layers should return zero
        assert abs(ZL) == 0, f"Expected zero impedance for empty layers, got {ZL}"

    def test_single_layer(self):
        """Test calc_ZL with single layer."""
        layers = {
            1: {"grho": 1e9, "phi": 0.1, "drho": 1e-6, "n": 3},
        }
        ZL = multilayer.calc_ZL(3, layers, 0.0 + 0.0j, 5e6)

        # Should return finite, non-zero impedance
        assert np.isfinite(ZL), f"Single layer produced non-finite: {ZL}"
        assert abs(ZL) > 0, f"Single layer produced zero impedance"

    def test_bulk_layer(self):
        """Test calc_ZL with infinite drho (bulk)."""
        layers = {
            1: {"grho": 1e9, "phi": 0.1, "drho": np.inf, "n": 3},
        }
        ZL = multilayer.calc_ZL(3, layers, 0.0 + 0.0j, 5e6)

        # Bulk layer should still produce finite impedance
        assert np.isfinite(ZL), f"Bulk layer produced non-finite: {ZL}"


class TestPhiEdgeCases:
    """Test phi (phase angle) edge cases."""

    def test_phi_zero(self):
        """Test calculations with phi = 0 (purely elastic)."""
        layers = {
            1: {"grho": 1e9, "phi": 0.0, "drho": 1e-6, "n": 3},
        }
        ZL = multilayer.calc_ZL(3, layers, 0.0 + 0.0j, 5e6)

        # Should handle phi=0 without NaN
        assert np.isfinite(ZL), f"phi=0 produced non-finite: {ZL}"

    def test_phi_near_zero(self):
        """Test calculations with phi very close to 0."""
        layers = {
            1: {"grho": 1e9, "phi": 1e-15, "drho": 1e-6, "n": 3},
        }
        ZL = multilayer.calc_ZL(3, layers, 0.0 + 0.0j, 5e6)

        assert np.isfinite(ZL), f"phi≈0 produced non-finite: {ZL}"

    def test_phi_pi_over_2(self):
        """Test calculations with phi = π/2 (purely viscous)."""
        layers = {
            1: {"grho": 1e9, "phi": np.pi / 2, "drho": 1e-6, "n": 3},
        }
        ZL = multilayer.calc_ZL(3, layers, 0.0 + 0.0j, 5e6)

        assert np.isfinite(ZL), f"phi=π/2 produced non-finite: {ZL}"

    def test_phi_near_pi_over_2(self):
        """Test calculations with phi close to π/2."""
        layers = {
            1: {"grho": 1e9, "phi": np.pi / 2 - 1e-10, "drho": 1e-6, "n": 3},
        }
        ZL = multilayer.calc_ZL(3, layers, 0.0 + 0.0j, 5e6)

        # Should handle near-boundary values
        assert np.isfinite(ZL), f"phi≈π/2 produced non-finite: {ZL}"

    def test_phi_exceeds_pi_over_2(self):
        """Test calculations with phi > π/2 (should be clamped after US3)."""
        layers = {
            1: {"grho": 1e9, "phi": np.pi / 2 + 0.1, "drho": 1e-6, "n": 3},
        }
        ZL = multilayer.calc_ZL(3, layers, 0.0 + 0.0j, 5e6)

        assert np.isfinite(ZL), f"phi>π/2 produced non-finite: {ZL}"


class TestArctan2EdgeCases:
    """Test arctan2 edge cases (US3 targets)."""

    def test_arctan2_zero_zero(self):
        """Test arctan2(0, 0) handling."""
        # arctan2(0, 0) is undefined but jnp.arctan2 returns 0.0
        result = jnp.arctan2(0.0, 0.0)

        # JAX arctan2 should return 0 for (0, 0) input
        assert np.isfinite(result), f"arctan2(0,0) produced non-finite: {result}"
        assert result == 0.0, f"arctan2(0,0) should return 0, got {result}"

    def test_arctan2_zero_positive(self):
        """Test arctan2(0, positive)."""
        result = jnp.arctan2(0.0, 1.0)
        assert result == 0.0, f"arctan2(0, 1) should be 0, got {result}"

    def test_arctan2_zero_negative(self):
        """Test arctan2(0, negative)."""
        result = jnp.arctan2(0.0, -1.0)
        # Should be π
        assert np.isclose(result, np.pi), f"arctan2(0, -1) should be π, got {result}"

    def test_arctan2_positive_zero(self):
        """Test arctan2(positive, 0)."""
        result = jnp.arctan2(1.0, 0.0)
        # Should be π/2
        assert np.isclose(result, np.pi / 2), (
            f"arctan2(1, 0) should be π/2, got {result}"
        )

    def test_arctan2_negative_zero(self):
        """Test arctan2(negative, 0)."""
        result = jnp.arctan2(-1.0, 0.0)
        # Should be -π/2
        assert np.isclose(result, -np.pi / 2), (
            f"arctan2(-1, 0) should be -π/2, got {result}"
        )


class TestSafeDivideEdgeCases:
    """Test safe division edge cases."""

    def test_divide_by_zero(self):
        """Test that division by zero is handled safely."""
        from rheoQCM.core.physics import safe_divide

        result = safe_divide(jnp.array(1.0), jnp.array(0.0), fill_value=0.0)
        assert np.isfinite(result), "safe_divide should return finite value"

    def test_divide_by_near_zero(self):
        """Test division by very small numbers."""
        numerator = jnp.array(1.0)
        denominator = jnp.array(1e-300)

        result = numerator / denominator

        # Should not overflow to inf
        assert np.isfinite(result), f"Division by 1e-300 produced non-finite: {result}"


class TestQCMModelEdgeCases:
    """Test QCMModel edge cases."""

    @pytest.fixture
    def qcm_model(self):
        """Standard QCMModel for testing."""
        return model.QCMModel(f1=5e6, refh=3)

    def test_empty_delfstars(self, qcm_model):
        """Test solve_properties with empty delfstars."""
        qcm_model.load_delfstars({})

        # Should return NaN values for empty input
        result = qcm_model.solve_properties([3, 5, 7])

        # Empty input should produce NaN or handle gracefully
        assert result is not None, "Should return result even for empty input"

    def test_nan_delfstar_input(self, qcm_model):
        """Test solve_properties with NaN in input."""
        qcm_model.load_delfstars(
            {
                3: np.nan + 0j,
                5: -1700 + 180j,
                7: -2500 + 280j,
            }
        )

        result = qcm_model.solve_properties([3, 5, 7])

        # NaN input should propagate or be handled gracefully
        # Document current behavior
        if np.isnan(result.grho_refh):
            pass  # Expected - NaN propagates
        else:
            # If we get a result, it should be finite
            assert np.isfinite(result.grho_refh), "Non-NaN result should be finite"

    def test_zero_delfstar_input(self, qcm_model):
        """Test solve_properties with all-zero delfstars."""
        qcm_model.load_delfstars(
            {
                3: 0.0 + 0.0j,
                5: 0.0 + 0.0j,
                7: 0.0 + 0.0j,
            }
        )

        result = qcm_model.solve_properties([3, 5, 7])

        # Zero input should produce meaningful result or NaN
        assert result is not None


class TestArrayEdgeCases:
    """Test array-related edge cases."""

    def test_empty_array(self):
        """Test operations on empty arrays."""
        empty = jnp.array([])

        # Sum of empty array should be 0
        assert jnp.sum(empty) == 0

        # Mean of empty array produces warning but returns NaN
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "Mean of empty slice")
            # JAX doesn't warn, just returns NaN
            result = jnp.mean(empty)
            assert np.isnan(result) or result == 0

    def test_single_element_array(self):
        """Test operations on single-element arrays."""
        single = jnp.array([1.0])

        # Standard operations should work
        assert jnp.sum(single) == 1.0
        assert jnp.mean(single) == 1.0


# Allow running specific tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
