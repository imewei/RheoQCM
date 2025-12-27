"""
Unit Tests for rheoQCM.core.signal Module

This module provides tests for JAX-based signal processing functions,
comparing outputs against scipy.signal reference implementations.

Feature: 003-scipy-to-jax
"""

import numpy as np
import jax.numpy as jnp
import pytest
from scipy.signal import (
    find_peaks as scipy_find_peaks,
    peak_prominences as scipy_peak_prominences,
    peak_widths as scipy_peak_widths,
)

from rheoQCM.core.signal import (
    find_peaks,
    peak_prominences,
    peak_widths,
)


class TestPeakProminences:
    """Tests for peak_prominences function comparing against scipy."""

    def test_basic_prominences(self):
        """T008: Compare basic prominence calculation against scipy."""
        # Create test signal with clear peaks
        x = np.array([0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0], dtype=np.float64)
        peaks = np.array([1, 3, 5, 7, 9])

        # Get scipy reference
        scipy_proms, scipy_left, scipy_right = scipy_peak_prominences(x, peaks)

        # Get JAX implementation
        jax_proms, jax_left, jax_right = peak_prominences(x, peaks)

        # Compare results
        np.testing.assert_allclose(
            np.asarray(jax_proms), scipy_proms, rtol=1e-10,
            err_msg="Prominences do not match scipy"
        )
        np.testing.assert_array_equal(
            np.asarray(jax_left), scipy_left,
            err_msg="Left bases do not match scipy"
        )
        np.testing.assert_array_equal(
            np.asarray(jax_right), scipy_right,
            err_msg="Right bases do not match scipy"
        )

    def test_edge_cases_empty_peaks(self):
        """T009: Test with empty peaks array."""
        x = np.array([0, 1, 0, 2, 0, 3, 0], dtype=np.float64)
        peaks = np.array([], dtype=np.intp)

        scipy_proms, scipy_left, scipy_right = scipy_peak_prominences(x, peaks)
        jax_proms, jax_left, jax_right = peak_prominences(x, peaks)

        assert len(jax_proms) == 0
        assert len(jax_left) == 0
        assert len(jax_right) == 0

    def test_edge_cases_single_peak(self):
        """T009: Test with single peak."""
        x = np.array([0, 1, 0], dtype=np.float64)
        peaks = np.array([1])

        scipy_proms, scipy_left, scipy_right = scipy_peak_prominences(x, peaks)
        jax_proms, jax_left, jax_right = peak_prominences(x, peaks)

        np.testing.assert_allclose(np.asarray(jax_proms), scipy_proms, rtol=1e-10)
        np.testing.assert_array_equal(np.asarray(jax_left), scipy_left)
        np.testing.assert_array_equal(np.asarray(jax_right), scipy_right)

    def test_edge_cases_boundary_peaks(self):
        """T009: Test with peaks at boundaries."""
        # Peak at start (index 0 is technically not a local max in scipy sense)
        x = np.array([1, 0, 2, 0, 1], dtype=np.float64)
        peaks = np.array([2])  # Only the clear peak

        scipy_proms, scipy_left, scipy_right = scipy_peak_prominences(x, peaks)
        jax_proms, jax_left, jax_right = peak_prominences(x, peaks)

        np.testing.assert_allclose(np.asarray(jax_proms), scipy_proms, rtol=1e-10)

    def test_edge_cases_flat_regions(self):
        """T009: Test flat regions with consecutive equal values.

        Flat regions should NOT be detected as peaks. This tests that
        the prominence calculation handles flat sections correctly.
        """
        # Signal with flat top - prominence should be calculated correctly
        x = np.array([0, 1, 1, 0, 2, 2, 2, 0, 3, 0], dtype=np.float64)
        # Only detect the right edge of flat tops (scipy behavior)
        peaks = np.array([8])  # The clear single peak at index 8

        scipy_proms, scipy_left, scipy_right = scipy_peak_prominences(x, peaks)
        jax_proms, jax_left, jax_right = peak_prominences(x, peaks)

        np.testing.assert_allclose(np.asarray(jax_proms), scipy_proms, rtol=1e-10)

    def test_with_wlen_parameter(self):
        """Test prominence calculation with limited window length."""
        x = np.array([0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0], dtype=np.float64)
        peaks = np.array([3, 7])

        # Test with wlen limiting the search window
        scipy_proms, scipy_left, scipy_right = scipy_peak_prominences(x, peaks, wlen=5)
        jax_proms, jax_left, jax_right = peak_prominences(x, peaks, wlen=5)

        np.testing.assert_allclose(np.asarray(jax_proms), scipy_proms, rtol=1e-10)


class TestPeakWidths:
    """Tests for peak_widths function comparing against scipy."""

    def test_basic_widths(self):
        """T010: Compare basic width calculation against scipy."""
        x = np.array([0, 0, 1, 2, 3, 2, 1, 0, 0], dtype=np.float64)
        peaks = np.array([4])

        scipy_widths, scipy_heights, scipy_left, scipy_right = scipy_peak_widths(x, peaks)
        jax_widths, jax_heights, jax_left, jax_right = peak_widths(x, peaks)

        np.testing.assert_allclose(np.asarray(jax_widths), scipy_widths, rtol=1e-10)
        np.testing.assert_allclose(np.asarray(jax_heights), scipy_heights, rtol=1e-10)
        np.testing.assert_allclose(np.asarray(jax_left), scipy_left, rtol=1e-10)
        np.testing.assert_allclose(np.asarray(jax_right), scipy_right, rtol=1e-10)

    def test_rel_height_variations(self):
        """T011: Test various rel_height values."""
        x = np.array([0, 0, 1, 2, 3, 2, 1, 0, 0], dtype=np.float64)
        peaks = np.array([4])

        for rel_height in [0.25, 0.5, 0.75, 0.9]:
            scipy_result = scipy_peak_widths(x, peaks, rel_height=rel_height)
            jax_result = peak_widths(x, peaks, rel_height=rel_height)

            np.testing.assert_allclose(
                np.asarray(jax_result[0]), scipy_result[0], rtol=1e-10,
                err_msg=f"Widths differ at rel_height={rel_height}"
            )
            np.testing.assert_allclose(
                np.asarray(jax_result[1]), scipy_result[1], rtol=1e-10,
                err_msg=f"Width heights differ at rel_height={rel_height}"
            )

    def test_multiple_peaks(self):
        """Test width calculation for multiple peaks."""
        x = np.array([0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0], dtype=np.float64)
        peaks = np.array([1, 3, 5, 7, 9])

        scipy_result = scipy_peak_widths(x, peaks)
        jax_result = peak_widths(x, peaks)

        np.testing.assert_allclose(np.asarray(jax_result[0]), scipy_result[0], rtol=1e-10)
        np.testing.assert_allclose(np.asarray(jax_result[1]), scipy_result[1], rtol=1e-10)

    def test_with_prominence_data(self):
        """Test width calculation with pre-computed prominence data."""
        x = np.array([0, 1, 0, 2, 0, 3, 0], dtype=np.float64)
        peaks = np.array([1, 3, 5])

        # Pre-compute prominence data
        proms, left, right = peak_prominences(x, peaks)

        # Use pre-computed data
        jax_result = peak_widths(x, peaks, prominence_data=(proms, left, right))

        # Compare with scipy
        scipy_result = scipy_peak_widths(x, peaks)
        np.testing.assert_allclose(np.asarray(jax_result[0]), scipy_result[0], rtol=1e-10)


class TestFindPeaks:
    """Tests for find_peaks function comparing against scipy."""

    def test_basic_find_peaks(self):
        """Test basic peak finding without constraints."""
        x = np.array([0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0], dtype=np.float64)

        scipy_peaks, _ = scipy_find_peaks(x)
        jax_peaks, _ = find_peaks(x)

        np.testing.assert_array_equal(np.asarray(jax_peaks), scipy_peaks)

    def test_with_distance_constraint(self):
        """T012: Test peak detection with distance constraint."""
        x = np.array([0, 1, 0.5, 2, 0, 3, 0.1, 2.5, 0, 1, 0], dtype=np.float64)

        scipy_peaks, _ = scipy_find_peaks(x, distance=2)
        jax_peaks, _ = find_peaks(x, distance=2)

        np.testing.assert_array_equal(np.asarray(jax_peaks), scipy_peaks)

    def test_with_distance_constraint_various(self):
        """T012: Test distance constraint with various values."""
        x = np.array([0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0], dtype=np.float64)

        for distance in [1, 2, 3, 4]:
            scipy_peaks, _ = scipy_find_peaks(x, distance=distance)
            jax_peaks, _ = find_peaks(x, distance=distance)

            np.testing.assert_array_equal(
                np.asarray(jax_peaks), scipy_peaks,
                err_msg=f"Peaks differ with distance={distance}"
            )

    def test_with_prominence_constraint(self):
        """T013: Test peak detection with prominence constraint."""
        x = np.array([0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0], dtype=np.float64)

        scipy_peaks, scipy_props = scipy_find_peaks(x, prominence=1.5)
        jax_peaks, jax_props = find_peaks(x, prominence=1.5)

        np.testing.assert_array_equal(np.asarray(jax_peaks), scipy_peaks)
        np.testing.assert_allclose(
            np.asarray(jax_props['prominences']),
            scipy_props['prominences'],
            rtol=1e-10
        )

    def test_with_prominence_range(self):
        """T013: Test prominence as min/max range."""
        x = np.array([0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0], dtype=np.float64)

        scipy_peaks, _ = scipy_find_peaks(x, prominence=(1.0, 2.5))
        jax_peaks, _ = find_peaks(x, prominence=(1.0, 2.5))

        np.testing.assert_array_equal(np.asarray(jax_peaks), scipy_peaks)

    def test_with_width_constraint(self):
        """T014: Test peak detection with width constraint."""
        # Create signal with varying width peaks
        x = np.array([0, 0, 1, 0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0], dtype=np.float64)

        scipy_peaks, scipy_props = scipy_find_peaks(x, width=2)
        jax_peaks, jax_props = find_peaks(x, width=2)

        np.testing.assert_array_equal(np.asarray(jax_peaks), scipy_peaks)

    def test_with_width_range(self):
        """T014: Test width as min/max range."""
        x = np.array([0, 0, 1, 0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0], dtype=np.float64)

        scipy_peaks, _ = scipy_find_peaks(x, width=(1, 5))
        jax_peaks, _ = find_peaks(x, width=(1, 5))

        np.testing.assert_array_equal(np.asarray(jax_peaks), scipy_peaks)

    def test_with_height_constraint(self):
        """Test peak detection with height constraint."""
        x = np.array([0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0], dtype=np.float64)

        scipy_peaks, scipy_props = scipy_find_peaks(x, height=1.5)
        jax_peaks, jax_props = find_peaks(x, height=1.5)

        np.testing.assert_array_equal(np.asarray(jax_peaks), scipy_peaks)
        np.testing.assert_allclose(
            np.asarray(jax_props['peak_heights']),
            scipy_props['peak_heights'],
            rtol=1e-10
        )

    def test_with_threshold_constraint(self):
        """Test peak detection with threshold constraint."""
        x = np.array([0, 1, 0.5, 2, 0, 3, 2.5, 2, 0, 1, 0], dtype=np.float64)

        scipy_peaks, _ = scipy_find_peaks(x, threshold=0.5)
        jax_peaks, _ = find_peaks(x, threshold=0.5)

        np.testing.assert_array_equal(np.asarray(jax_peaks), scipy_peaks)

    def test_combined_constraints(self):
        """T015: Test all constraints together."""
        # Create realistic QCM-like signal
        x = np.array([
            0, 0.5, 1, 0.5, 0,  # small peak
            0.5, 1.5, 2, 1.5, 0.5,  # medium peak
            0, 0.5, 1.5, 2.5, 3, 2.5, 1.5, 0.5, 0,  # large wide peak
            0.5, 1, 0.5,  # small peak
            0, 0.5, 2, 0.5, 0,  # medium peak
        ], dtype=np.float64)

        scipy_peaks, scipy_props = scipy_find_peaks(
            x, distance=3, prominence=1.0, width=2
        )
        jax_peaks, jax_props = find_peaks(
            x, distance=3, prominence=1.0, width=2
        )

        np.testing.assert_array_equal(np.asarray(jax_peaks), scipy_peaks)

        # Verify properties match
        np.testing.assert_allclose(
            np.asarray(jax_props['prominences']),
            scipy_props['prominences'],
            rtol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(jax_props['widths']),
            scipy_props['widths'],
            rtol=1e-10
        )

    def test_no_peaks_found(self):
        """Test when no peaks meet criteria."""
        x = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)  # Monotonically increasing

        scipy_peaks, _ = scipy_find_peaks(x)
        jax_peaks, _ = find_peaks(x)

        assert len(jax_peaks) == 0
        assert len(scipy_peaks) == 0

    def test_sinusoidal_signal(self):
        """Test with realistic sinusoidal signal."""
        t = np.linspace(0, 4 * np.pi, 100)
        x = np.sin(t)

        scipy_peaks, _ = scipy_find_peaks(x)
        jax_peaks, _ = find_peaks(x)

        np.testing.assert_array_equal(np.asarray(jax_peaks), scipy_peaks)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_short_signal(self):
        """Test with minimum length signals."""
        # Single point - no peaks possible
        x = np.array([1.0])
        peaks, _ = find_peaks(x)
        assert len(peaks) == 0

        # Two points - no local maximum possible
        x = np.array([1.0, 2.0])
        peaks, _ = find_peaks(x)
        assert len(peaks) == 0

        # Three points - can have peak at center
        x = np.array([0.0, 1.0, 0.0])
        peaks, _ = find_peaks(x)
        assert len(peaks) == 1
        assert peaks[0] == 1

    def test_all_equal_values(self):
        """Test signal with all equal values (no peaks)."""
        x = np.ones(10)

        scipy_peaks, _ = scipy_find_peaks(x)
        jax_peaks, _ = find_peaks(x)

        assert len(jax_peaks) == 0
        assert len(scipy_peaks) == 0

    def test_strictly_increasing(self):
        """Test strictly increasing signal (no peaks)."""
        x = np.arange(10, dtype=np.float64)

        scipy_peaks, _ = scipy_find_peaks(x)
        jax_peaks, _ = find_peaks(x)

        assert len(jax_peaks) == 0

    def test_strictly_decreasing(self):
        """Test strictly decreasing signal (no peaks)."""
        x = np.arange(10, 0, -1, dtype=np.float64)

        scipy_peaks, _ = scipy_find_peaks(x)
        jax_peaks, _ = find_peaks(x)

        assert len(jax_peaks) == 0


class TestInputValidation:
    """Tests for input validation."""

    def test_invalid_peaks_indices(self):
        """Test that invalid peak indices raise appropriate errors."""
        x = np.array([0, 1, 0, 2, 0], dtype=np.float64)

        # Out of bounds index
        with pytest.raises(ValueError):
            peak_prominences(x, np.array([10]))

        # Negative index
        with pytest.raises(ValueError):
            peak_prominences(x, np.array([-1]))

    def test_1d_requirement(self):
        """Test that 2D arrays are rejected."""
        x = np.array([[0, 1], [0, 2]], dtype=np.float64)

        with pytest.raises(ValueError):
            find_peaks(x)

    def test_empty_signal(self):
        """Test behavior with empty signal."""
        x = np.array([], dtype=np.float64)
        peaks, _ = find_peaks(x)
        assert len(peaks) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
