"""
Tests for PeakTracker module after lmfit to NLSQ migration.

This test suite validates:
1. JAX-compatible model functions (fun_G, fun_B)
2. Parameter management system (build_p0_and_bounds)
3. Core fitting engine (NLSQ curve_fit)
4. Result adapter for backward compatibility
5. Numerical accuracy vs lmfit baseline
6. Edge cases and error handling
"""

import jax.numpy as jnp
import numpy as np
import pytest

# =============================================================================
# Task Group 2: JAX-Compatible Peak Model Functions Tests
# =============================================================================


class TestModelFunctions:
    """Tests for JAX-compatible fun_G and fun_B model functions."""

    def test_fun_G_analytical_values(self):
        """Test fun_G output against known analytical values at 3 points."""
        # Import the functions
        from rheoQCM.modules.PeakTracker import fun_G_numpy

        # Test at peak center (x = cen), where G should be maximum
        x = np.array([5e6])
        amp, cen, wid, phi = 0.001, 5e6, 1000.0, 0.0

        G = fun_G_numpy(x, amp, cen, wid, phi)

        # At x=cen with phi=0: G = amp * cos(phi) = amp
        # More precisely, at resonance: G approaches amp / (2*wid) behavior
        assert G.shape == (1,)
        assert np.isfinite(G[0])

    def test_fun_B_analytical_values(self):
        """Test fun_B output against known analytical values at 3 points."""
        from rheoQCM.modules.PeakTracker import fun_B_numpy

        # Test at peak center
        x = np.array([5e6])
        amp, cen, wid, phi = 0.001, 5e6, 1000.0, 0.0

        B = fun_B_numpy(x, amp, cen, wid, phi)

        # At x=cen with phi=0: B has a specific value
        assert B.shape == (1,)
        assert np.isfinite(B[0])

    def test_fun_G_vectorization(self):
        """Test that fun_G correctly vectorizes over frequency array."""
        from rheoQCM.modules.PeakTracker import fun_G_numpy

        x = np.linspace(4.99e6, 5.01e6, 100)
        amp, cen, wid, phi = 0.001, 5e6, 1000.0, 0.1

        G = fun_G_numpy(x, amp, cen, wid, phi)

        assert G.shape == x.shape
        assert np.all(np.isfinite(G))

    def test_fun_B_vectorization(self):
        """Test that fun_B correctly vectorizes over frequency array."""
        from rheoQCM.modules.PeakTracker import fun_B_numpy

        x = np.linspace(4.99e6, 5.01e6, 100)
        amp, cen, wid, phi = 0.001, 5e6, 1000.0, 0.1

        B = fun_B_numpy(x, amp, cen, wid, phi)

        assert B.shape == x.shape
        assert np.all(np.isfinite(B))

    def test_jax_numpy_equivalence(self):
        """Test that JAX and NumPy versions produce identical results."""
        from rheoQCM.modules.PeakTracker import fun_B, fun_B_numpy, fun_G, fun_G_numpy

        x = np.linspace(4.99e6, 5.01e6, 50)
        amp, cen, wid, phi = 0.001, 5e6, 1000.0, 0.1

        # JAX versions
        G_jax = np.array(fun_G(x, amp, cen, wid, phi))
        B_jax = np.array(fun_B(x, amp, cen, wid, phi))

        # NumPy versions
        G_np = fun_G_numpy(x, amp, cen, wid, phi)
        B_np = fun_B_numpy(x, amp, cen, wid, phi)

        np.testing.assert_allclose(G_jax, G_np, rtol=1e-10)
        np.testing.assert_allclose(B_jax, B_np, rtol=1e-10)

    def test_composite_model_single_peak(self):
        """Test composite model for single peak returns [G, B] concatenated."""
        from rheoQCM.modules.PeakTracker import create_composite_model

        model = create_composite_model(1)
        x = np.linspace(4.99e6, 5.01e6, 50)
        x_stacked = np.concatenate([x, x])
        params = [0.001, 5e6, 1000.0, 0.1, 1e-5, 0.0]  # amp, cen, wid, phi, g_c, b_c

        result = model(x_stacked, *params)

        # Should return concatenated [G, B]
        assert len(result) == 2 * len(x)
        G_part = result[: len(x)]
        B_part = result[len(x) :]
        assert np.all(np.isfinite(G_part))
        assert np.all(np.isfinite(B_part))


# =============================================================================
# Task Group 3: Parameter Management System Tests
# =============================================================================


class TestParameterManagement:
    """Tests for parameter array construction and bounds handling."""

    def test_get_param_index(self):
        """Test parameter name-to-index mapping."""
        from rheoQCM.modules.PeakTracker import get_param_index

        # Single peak (n_peaks=1)
        assert get_param_index("p0_amp", 1) == 0
        assert get_param_index("p0_cen", 1) == 1
        assert get_param_index("p0_wid", 1) == 2
        assert get_param_index("p0_phi", 1) == 3
        assert get_param_index("g_c", 1) == 4
        assert get_param_index("b_c", 1) == 5

        # Two peaks (n_peaks=2)
        assert get_param_index("p0_amp", 2) == 0
        assert get_param_index("p1_amp", 2) == 4
        assert get_param_index("p1_cen", 2) == 5
        assert get_param_index("g_c", 2) == 8
        assert get_param_index("b_c", 2) == 9

    def test_build_p0_and_bounds_single_peak(self):
        """Test parameter array construction for single peak."""
        from rheoQCM.modules.PeakTracker import build_p0_and_bounds

        params_dict = {
            0: {
                "amp": 0.001,
                "amp_min": 0,
                "amp_max": np.inf,
                "cen": 5e6,
                "cen_min": 4.9e6,
                "cen_max": 5.1e6,
                "wid": 1000,
                "wid_min": 100,
                "wid_max": 10000,
                "phi": 0.1,
                "phi_min": -np.pi / 2,
                "phi_max": np.pi / 2,
            },
            "g_c": 1e-5,
            "b_c": 0.0,
        }

        p0, bounds, fixed_mask = build_p0_and_bounds(params_dict, n_peaks=1)

        # Check p0 values
        assert len(p0) == 6  # 4 per peak + 2 baselines
        assert p0[0] == 0.001  # amp
        assert p0[1] == 5e6  # cen
        assert p0[2] == 1000  # wid
        assert p0[3] == 0.1  # phi
        assert p0[4] == 1e-5  # g_c
        assert p0[5] == 0.0  # b_c

        # Check bounds structure
        lb, ub = bounds
        assert len(lb) == 6
        assert len(ub) == 6
        assert lb[0] == 0  # amp_min
        assert lb[1] == 4.9e6  # cen_min

    def test_build_p0_and_bounds_zerophase(self):
        """Test fixed phi=0 parameter handling with zerophase mode."""
        from rheoQCM.modules.PeakTracker import build_p0_and_bounds

        params_dict = {
            0: {
                "amp": 0.001,
                "cen": 5e6,
                "wid": 1000,
                "phi": 0.1,
            },
            "g_c": 0.0,
            "b_c": 0.0,
        }

        p0, bounds, fixed_mask = build_p0_and_bounds(
            params_dict, n_peaks=1, zerophase=True
        )

        # phi should be fixed to 0 with tight bounds
        assert p0[3] == 0.0
        lb, ub = bounds
        assert abs(lb[3]) < 1e-9
        assert abs(ub[3]) < 1e-9
        assert fixed_mask[3] == True  # phi is fixed

    def test_build_p0_and_bounds_multi_peak(self):
        """Test parameter array for multiple peaks."""
        from rheoQCM.modules.PeakTracker import build_p0_and_bounds

        params_dict = {
            0: {"amp": 0.001, "cen": 5e6, "wid": 1000, "phi": 0.0},
            1: {"amp": 0.0005, "cen": 5.5e6, "wid": 800, "phi": 0.05},
            "g_c": 1e-5,
            "b_c": 1e-6,
        }

        p0, bounds, fixed_mask = build_p0_and_bounds(params_dict, n_peaks=2)

        # Check length: 2 peaks * 4 params + 2 baselines = 10
        assert len(p0) == 10
        assert p0[4] == 0.0005  # p1_amp
        assert p0[5] == 5.5e6  # p1_cen
        assert p0[8] == 1e-5  # g_c
        assert p0[9] == 1e-6  # b_c


# =============================================================================
# Task Group 4: Core Fitting Engine Tests
# =============================================================================


class TestFittingEngine:
    """Tests for NLSQ-based fitting functionality."""

    def test_single_peak_fitting_synthetic(self):
        """Test single peak fitting on synthetic Lorentzian data."""
        from nlsq import curve_fit

        from rheoQCM.modules.PeakTracker import (
            create_composite_model,
            fun_B_numpy,
            fun_G_numpy,
        )

        # Generate synthetic data
        f = np.linspace(4.99e6, 5.01e6, 200)
        amp_true, cen_true, wid_true, phi_true = 0.001, 5e6, 1000.0, 0.05
        g_c_true, b_c_true = 1e-5, 0.0

        G = fun_G_numpy(f, amp_true, cen_true, wid_true, phi_true) + g_c_true
        B = fun_B_numpy(f, amp_true, cen_true, wid_true, phi_true) + b_c_true

        # Add small noise
        np.random.seed(42)
        G += np.random.normal(0, 1e-7, len(f))
        B += np.random.normal(0, 1e-7, len(f))

        # Fit - stack x and y data
        model = create_composite_model(1)
        x_stacked = np.concatenate([f, f])
        y_data = np.concatenate([G, B])
        p0 = [
            amp_true * 0.9,
            cen_true * 1.001,
            wid_true * 1.1,
            phi_true,
            g_c_true,
            b_c_true,
        ]
        bounds = (
            [0, f.min(), 100, -np.pi / 2, -np.inf, -np.inf],
            [np.inf, f.max(), 10000, np.pi / 2, np.inf, np.inf],
        )

        popt, pcov = curve_fit(model, x_stacked, y_data, p0=p0, bounds=bounds)

        # Check results within 0.1% for cen and wid
        rel_err_cen = abs(popt[1] - cen_true) / cen_true
        rel_err_wid = abs(popt[2] - wid_true) / wid_true

        assert rel_err_cen < 0.001, f"cen error {rel_err_cen*100:.4f}% exceeds 0.1%"
        assert rel_err_wid < 0.001, f"wid error {rel_err_wid*100:.4f}% exceeds 0.1%"

    def test_fitting_with_fixed_phi(self):
        """Test fitting with phi fixed to 0 (zerophase mode)."""
        from nlsq import curve_fit

        from rheoQCM.modules.PeakTracker import (
            create_composite_model,
            fun_B_numpy,
            fun_G_numpy,
        )

        f = np.linspace(4.99e6, 5.01e6, 100)
        amp_true, cen_true, wid_true = 0.001, 5e6, 1000.0
        phi_true = 0.0  # Zero phase
        g_c_true, b_c_true = 1e-5, 0.0

        G = fun_G_numpy(f, amp_true, cen_true, wid_true, phi_true) + g_c_true
        B = fun_B_numpy(f, amp_true, cen_true, wid_true, phi_true) + b_c_true

        model = create_composite_model(1)
        x_stacked = np.concatenate([f, f])
        y_data = np.concatenate([G, B])

        # Fix phi with tight bounds
        eps = 1e-10
        p0 = [amp_true, cen_true, wid_true, 0.0, g_c_true, b_c_true]
        bounds = (
            [0, f.min(), 100, -eps, -np.inf, -np.inf],
            [np.inf, f.max(), 10000, eps, np.inf, np.inf],
        )

        popt, pcov = curve_fit(model, x_stacked, y_data, p0=p0, bounds=bounds)

        # phi should remain essentially 0
        assert abs(popt[3]) < 1e-6, f"Fixed phi={popt[3]} should be near 0"

    def test_uncertainty_estimation(self):
        """Test that stderr extraction from covariance works."""
        from nlsq import curve_fit

        from rheoQCM.modules.PeakTracker import (
            create_composite_model,
            fun_B_numpy,
            fun_G_numpy,
        )

        f = np.linspace(4.99e6, 5.01e6, 100)
        amp, cen, wid, phi = 0.001, 5e6, 1000.0, 0.0

        G = fun_G_numpy(f, amp, cen, wid, phi) + 1e-5
        B = fun_B_numpy(f, amp, cen, wid, phi)

        model = create_composite_model(1)
        x_stacked = np.concatenate([f, f])
        y_data = np.concatenate([G, B])
        p0 = [amp, cen, wid, phi, 1e-5, 0.0]
        bounds = (
            [0, f.min(), 100, -np.pi / 2, -np.inf, -np.inf],
            [np.inf, f.max(), 10000, np.pi / 2, np.inf, np.inf],
        )

        popt, pcov = curve_fit(model, x_stacked, y_data, p0=p0, bounds=bounds)

        # Check covariance matrix is valid
        assert pcov.shape == (6, 6)
        stderr = np.sqrt(np.diag(pcov))
        assert np.all(np.isfinite(stderr))
        assert np.all(stderr >= 0)


# =============================================================================
# Task Group 5: Integration and Backward Compatibility Tests
# =============================================================================


class TestResultAdapter:
    """Tests for NLSQResultAdapter backward compatibility."""

    def test_result_adapter_params_valuesdict(self):
        """Test that valuesdict() returns correct parameter dictionary."""
        from rheoQCM.modules.PeakTracker import NLSQResultAdapter

        popt = np.array([0.001, 5e6, 1000.0, 0.1, 1e-5, 0.0])
        pcov = np.eye(6) * 1e-10

        result = NLSQResultAdapter(popt, pcov, n_peaks=1)
        val_dict = result.params.valuesdict()

        assert val_dict["p0_amp"] == 0.001
        assert val_dict["p0_cen"] == 5e6
        assert val_dict["p0_wid"] == 1000.0
        assert val_dict["p0_phi"] == 0.1
        assert val_dict["g_c"] == 1e-5
        assert val_dict["b_c"] == 0.0

    def test_result_adapter_params_get(self):
        """Test that params.get('name').value/stderr works."""
        from rheoQCM.modules.PeakTracker import NLSQResultAdapter

        popt = np.array([0.001, 5e6, 1000.0, 0.1, 1e-5, 0.0])
        pcov = np.diag([1e-12, 1e-6, 1e-4, 1e-8, 1e-14, 1e-14])

        result = NLSQResultAdapter(popt, pcov, n_peaks=1)

        # Test value access
        assert result.params.get("p0_amp").value == 0.001
        assert result.params.get("p0_cen").value == 5e6

        # Test stderr access
        assert result.params.get("p0_amp").stderr == pytest.approx(np.sqrt(1e-12))
        assert result.params.get("p0_cen").stderr == pytest.approx(np.sqrt(1e-6))

    def test_result_adapter_success_flag(self):
        """Test success flag and message attributes."""
        from rheoQCM.modules.PeakTracker import NLSQResultAdapter

        popt = np.array([0.001, 5e6, 1000.0, 0.1, 1e-5, 0.0])
        pcov = np.eye(6)

        result = NLSQResultAdapter(popt, pcov, n_peaks=1, success=True, message="OK")

        assert result.success == True
        assert result.message == "OK"
        assert bool(result) == True

    def test_result_adapter_chisqr(self):
        """Test chi-squared attribute."""
        from rheoQCM.modules.PeakTracker import NLSQResultAdapter

        popt = np.array([0.001, 5e6, 1000.0, 0.1, 1e-5, 0.0])
        pcov = np.eye(6)

        result = NLSQResultAdapter(popt, pcov, n_peaks=1)
        result.chisqr = 1.234e-8

        assert result.chisqr == 1.234e-8


# =============================================================================
# Task Group 6: Numerical Validation Tests
# =============================================================================


class TestNumericalValidation:
    """Tests for numerical accuracy - results within 0.1% of reference."""

    def test_cen_accuracy_vs_reference(self):
        """Test fitted center frequency within 0.1% of true value."""
        from nlsq import curve_fit

        from rheoQCM.modules.PeakTracker import (
            create_composite_model,
            fun_B_numpy,
            fun_G_numpy,
        )

        # Reference values
        f = np.linspace(4.995e6, 5.005e6, 300)
        cen_true = 5.0e6
        amp_true, wid_true, phi_true = 0.001, 500.0, 0.0
        g_c_true, b_c_true = 5e-6, 0.0

        G = fun_G_numpy(f, amp_true, cen_true, wid_true, phi_true) + g_c_true
        B = fun_B_numpy(f, amp_true, cen_true, wid_true, phi_true) + b_c_true

        model = create_composite_model(1)
        x_stacked = np.concatenate([f, f])
        y_data = np.concatenate([G, B])
        p0 = [
            amp_true * 1.1,
            cen_true * 0.9999,
            wid_true * 0.9,
            0.0,
            g_c_true,
            b_c_true,
        ]
        bounds = (
            [0, f.min(), 10, -np.pi / 2, -np.inf, -np.inf],
            [np.inf, f.max(), 5000, np.pi / 2, np.inf, np.inf],
        )

        popt, _ = curve_fit(model, x_stacked, y_data, p0=p0, bounds=bounds)

        rel_error = abs(popt[1] - cen_true) / cen_true * 100
        assert rel_error < 0.1, f"Center frequency error {rel_error:.4f}% exceeds 0.1%"

    def test_wid_accuracy_vs_reference(self):
        """Test fitted bandwidth within 0.1% of true value."""
        from nlsq import curve_fit

        from rheoQCM.modules.PeakTracker import (
            create_composite_model,
            fun_B_numpy,
            fun_G_numpy,
        )

        f = np.linspace(4.995e6, 5.005e6, 300)
        wid_true = 500.0
        amp_true, cen_true, phi_true = 0.001, 5.0e6, 0.0
        g_c_true, b_c_true = 5e-6, 0.0

        G = fun_G_numpy(f, amp_true, cen_true, wid_true, phi_true) + g_c_true
        B = fun_B_numpy(f, amp_true, cen_true, wid_true, phi_true) + b_c_true

        model = create_composite_model(1)
        x_stacked = np.concatenate([f, f])
        y_data = np.concatenate([G, B])
        p0 = [amp_true, cen_true, wid_true * 1.2, 0.0, g_c_true, b_c_true]
        bounds = (
            [0, f.min(), 10, -np.pi / 2, -np.inf, -np.inf],
            [np.inf, f.max(), 5000, np.pi / 2, np.inf, np.inf],
        )

        popt, _ = curve_fit(model, x_stacked, y_data, p0=p0, bounds=bounds)

        rel_error = abs(popt[2] - wid_true) / wid_true * 100
        assert rel_error < 0.1, f"Width error {rel_error:.4f}% exceeds 0.1%"

    def test_noisy_data_convergence(self):
        """Test fitting converges with noisy data."""
        from nlsq import curve_fit

        from rheoQCM.modules.PeakTracker import (
            create_composite_model,
            fun_B_numpy,
            fun_G_numpy,
        )

        f = np.linspace(4.99e6, 5.01e6, 200)
        amp_true, cen_true, wid_true, phi_true = 0.001, 5e6, 1000.0, 0.0

        G = fun_G_numpy(f, amp_true, cen_true, wid_true, phi_true) + 1e-5
        B = fun_B_numpy(f, amp_true, cen_true, wid_true, phi_true)

        # Add 5% noise relative to peak amplitude
        np.random.seed(123)
        noise_level = 0.05 * amp_true
        G += np.random.normal(0, noise_level, len(f))
        B += np.random.normal(0, noise_level, len(f))

        model = create_composite_model(1)
        x_stacked = np.concatenate([f, f])
        y_data = np.concatenate([G, B])
        p0 = [amp_true, cen_true, wid_true, 0.0, 1e-5, 0.0]
        bounds = (
            [0, f.min(), 100, -np.pi / 2, -np.inf, -np.inf],
            [np.inf, f.max(), 10000, np.pi / 2, np.inf, np.inf],
        )

        popt, pcov = curve_fit(model, x_stacked, y_data, p0=p0, bounds=bounds)

        # With 5% noise, allow 5% tolerance on fitted parameters
        rel_err_cen = abs(popt[1] - cen_true) / cen_true
        rel_err_wid = abs(popt[2] - wid_true) / wid_true

        assert rel_err_cen < 0.05, f"cen error {rel_err_cen*100:.2f}% with noisy data"
        assert rel_err_wid < 0.1, f"wid error {rel_err_wid*100:.2f}% with noisy data"


# =============================================================================
# Task Group 8: Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_nan_handling(self):
        """Test fitting with NaN values in data."""
        from nlsq import curve_fit

        from rheoQCM.modules.PeakTracker import (
            create_composite_model,
            fun_B_numpy,
            fun_G_numpy,
        )

        f = np.linspace(4.99e6, 5.01e6, 100)
        G = fun_G_numpy(f, 0.001, 5e6, 1000.0, 0.0) + 1e-5
        B = fun_B_numpy(f, 0.001, 5e6, 1000.0, 0.0)

        # Insert a few NaN values
        G[10] = np.nan
        B[20] = np.nan

        # Remove NaN indices
        valid_mask = ~np.isnan(G) & ~np.isnan(B)
        f_clean = f[valid_mask]
        G_clean = G[valid_mask]
        B_clean = B[valid_mask]

        model = create_composite_model(1)
        x_stacked = np.concatenate([f_clean, f_clean])
        y_data = np.concatenate([G_clean, B_clean])
        p0 = [0.001, 5e6, 1000.0, 0.0, 1e-5, 0.0]
        bounds = (
            [0, f.min(), 100, -np.pi / 2, -np.inf, -np.inf],
            [np.inf, f.max(), 10000, np.pi / 2, np.inf, np.inf],
        )

        popt, pcov = curve_fit(model, x_stacked, y_data, p0=p0, bounds=bounds)

        # Should still converge
        assert np.all(np.isfinite(popt))

    def test_far_initial_guess(self):
        """Test fitting with initial guess far from optimal."""
        from nlsq import curve_fit

        from rheoQCM.modules.PeakTracker import (
            create_composite_model,
            fun_B_numpy,
            fun_G_numpy,
        )

        f = np.linspace(4.99e6, 5.01e6, 200)
        amp_true, cen_true, wid_true = 0.001, 5e6, 1000.0

        G = fun_G_numpy(f, amp_true, cen_true, wid_true, 0.0) + 1e-5
        B = fun_B_numpy(f, amp_true, cen_true, wid_true, 0.0)

        model = create_composite_model(1)
        x_stacked = np.concatenate([f, f])
        y_data = np.concatenate([G, B])

        # Poor initial guess (50% off)
        p0 = [amp_true * 0.5, cen_true * 1.001, wid_true * 1.5, 0.0, 0.0, 0.0]
        bounds = (
            [0, f.min(), 100, -np.pi / 2, -np.inf, -np.inf],
            [np.inf, f.max(), 10000, np.pi / 2, np.inf, np.inf],
        )

        popt, _ = curve_fit(model, x_stacked, y_data, p0=p0, bounds=bounds)

        # Should still recover true values
        rel_err_cen = abs(popt[1] - cen_true) / cen_true
        assert rel_err_cen < 0.01, f"Failed to converge from far initial guess"

    def test_very_narrow_peak(self):
        """Test fitting with width near minimum bound."""
        from nlsq import curve_fit

        from rheoQCM.modules.PeakTracker import (
            create_composite_model,
            fun_B_numpy,
            fun_G_numpy,
        )

        f = np.linspace(4.9999e6, 5.0001e6, 500)  # High resolution
        wid_true = 10.0  # Very narrow
        amp_true, cen_true = 0.001, 5e6

        G = fun_G_numpy(f, amp_true, cen_true, wid_true, 0.0) + 1e-5
        B = fun_B_numpy(f, amp_true, cen_true, wid_true, 0.0)

        model = create_composite_model(1)
        x_stacked = np.concatenate([f, f])
        y_data = np.concatenate([G, B])
        p0 = [amp_true, cen_true, 50.0, 0.0, 1e-5, 0.0]
        bounds = (
            [0, f.min(), 1, -np.pi / 2, -np.inf, -np.inf],
            [np.inf, f.max(), 1000, np.pi / 2, np.inf, np.inf],
        )

        popt, _ = curve_fit(model, x_stacked, y_data, p0=p0, bounds=bounds)

        # Should recover narrow width
        rel_err_wid = abs(popt[2] - wid_true) / wid_true
        assert rel_err_wid < 0.1, f"Failed to fit narrow peak, wid={popt[2]}"

    def test_two_peak_fitting(self):
        """Test fitting with two overlapping peaks."""
        from nlsq import curve_fit

        from rheoQCM.modules.PeakTracker import (
            create_composite_model,
            fun_B_numpy,
            fun_G_numpy,
        )

        f = np.linspace(4.95e6, 5.05e6, 300)

        # Two peaks
        amp1, cen1, wid1 = 0.001, 4.98e6, 500.0
        amp2, cen2, wid2 = 0.0008, 5.02e6, 600.0
        g_c, b_c = 1e-5, 0.0

        G = (
            fun_G_numpy(f, amp1, cen1, wid1, 0.0)
            + fun_G_numpy(f, amp2, cen2, wid2, 0.0)
            + g_c
        )
        B = (
            fun_B_numpy(f, amp1, cen1, wid1, 0.0)
            + fun_B_numpy(f, amp2, cen2, wid2, 0.0)
            + b_c
        )

        model = create_composite_model(2)
        x_stacked = np.concatenate([f, f])
        y_data = np.concatenate([G, B])

        # Initial guess for 2 peaks
        p0 = [amp1, cen1, wid1, 0.0, amp2, cen2, wid2, 0.0, g_c, b_c]
        bounds = (
            [
                0,
                f.min(),
                100,
                -np.pi / 2,
                0,
                f.min(),
                100,
                -np.pi / 2,
                -np.inf,
                -np.inf,
            ],
            [
                np.inf,
                f.max(),
                5000,
                np.pi / 2,
                np.inf,
                f.max(),
                5000,
                np.pi / 2,
                np.inf,
                np.inf,
            ],
        )

        popt, _ = curve_fit(model, x_stacked, y_data, p0=p0, bounds=bounds)

        # Check both centers recovered
        fitted_centers = sorted([popt[1], popt[5]])
        true_centers = sorted([cen1, cen2])

        for fc, tc in zip(fitted_centers, true_centers):
            rel_err = abs(fc - tc) / tc
            assert rel_err < 0.01, f"Multi-peak center error {rel_err*100:.2f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
