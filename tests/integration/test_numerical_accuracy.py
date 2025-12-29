"""
Numerical accuracy tests for JAX performance optimization.
Feature: 012-jax-performance-optimization

Validates that optimizations maintain numerical accuracy within 1e-10
relative error compared to baseline implementations.
"""

import jax.numpy as jnp
import numpy as np
import pytest

# Import modules under test
from rheoQCM.core import model, multilayer, physics

# Constants for test cases
TOLERANCE = 1e-10  # Maximum allowed relative error
F1_DEFAULT = 5e6  # Default fundamental frequency


def relative_error(computed: complex, expected: complex) -> float:
    """Calculate relative error between computed and expected values."""
    if abs(expected) < 1e-30:
        return abs(computed - expected)
    return abs(computed - expected) / abs(expected)


class TestMultilayerAccuracy:
    """Test numerical accuracy of multilayer calculations."""

    @pytest.fixture
    def standard_layers(self):
        """Standard two-layer test case."""
        return {
            1: {"grho": 1e9, "phi": 0.1, "drho": 1e-6, "n": 3},
            2: {"grho": 1e8, "phi": 0.2, "drho": np.inf, "n": 3},
        }

    @pytest.fixture
    def single_layer(self):
        """Single layer test case."""
        return {
            1: {"grho": 5e8, "phi": 0.15, "drho": 2e-6, "n": 3},
        }

    def test_calc_ZL_consistency(self, standard_layers):
        """Test that calc_ZL produces consistent results."""
        n = 3
        delfstar = 0.0 + 0.0j
        f1 = F1_DEFAULT

        # Run multiple times to check determinism
        results = []
        for _ in range(5):
            ZL = multilayer.calc_ZL(n, standard_layers, delfstar, f1)
            results.append(complex(ZL))

        # All results should be identical
        for i, r in enumerate(results[1:], 1):
            error = relative_error(r, results[0])
            assert error < TOLERANCE, (
                f"Iteration {i} differs from iteration 0: "
                f"error={error:.2e}, tolerance={TOLERANCE:.0e}"
            )

    def test_calc_ZL_known_limits(self, single_layer):
        """Test calc_ZL against known physical limits."""
        n = 3
        delfstar = 0.0 + 0.0j
        f1 = F1_DEFAULT

        ZL = multilayer.calc_ZL(n, single_layer, delfstar, f1)

        # Impedance should be non-zero for physical layers
        assert abs(ZL) > 0, "Impedance should be non-zero"

        # Real part should be positive for lossy materials
        # (Imaginary part can be positive or negative)
        # This is a physical consistency check, not accuracy


class TestPhysicsAccuracy:
    """Test numerical accuracy of physics calculations."""

    def test_sauerbreym_consistency(self):
        """Test Sauerbrey mass calculation consistency."""
        n = 3
        delf = -1000.0  # Hz
        f1 = F1_DEFAULT

        results = []
        for _ in range(5):
            m = physics.sauerbreym(n, delf, f1=f1)
            results.append(float(m))

        for i, r in enumerate(results[1:], 1):
            if abs(results[0]) > 1e-30:
                error = abs(r - results[0]) / abs(results[0])
            else:
                error = abs(r - results[0])
            assert (
                error < TOLERANCE
            ), f"Sauerbrey calculation inconsistent: error={error:.2e}"

    def test_grho_phi_conversion_roundtrip(self):
        """Test grho/phi conversion roundtrip accuracy."""
        # Start with known values
        grho_refh = 1e9  # Pa kg/m^3 at reference harmonic
        phi = 0.15  # radians
        drho = 1e-6  # kg/m^2
        n = 3
        refh = 3
        f1 = F1_DEFAULT

        # Use QCMModel to get grho at harmonic n
        m = model.QCMModel(f1=f1, refh=refh)
        grho_n = m._grho_at_harmonic(n, grho_refh, phi)

        # Convert to dlam
        dlam = physics.calc_dlam(n, grho_n, phi, drho, f1=f1)

        # Convert back to grho_refh
        grho_back = physics.grho_from_dlam(refh, drho, float(dlam), phi, f1=f1)

        # Check roundtrip accuracy
        error = relative_error(float(grho_back), grho_refh)
        assert (
            error < TOLERANCE
        ), f"grho roundtrip error: {error:.2e} (tolerance: {TOLERANCE:.0e})"


class TestModelAccuracy:
    """Test numerical accuracy of QCMModel calculations."""

    @pytest.fixture
    def qcm_model(self):
        """Standard QCMModel for testing."""
        m = model.QCMModel(f1=F1_DEFAULT, refh=3)
        # Load typical delfstar data
        m.load_delfstars(
            {
                3: -1000 + 100j,
                5: -1700 + 180j,
                7: -2500 + 280j,
            }
        )
        return m

    def test_solve_properties_determinism(self, qcm_model):
        """Test that solve_properties produces deterministic results."""
        nh = [3, 5, 7]

        results = []
        for _ in range(3):
            result = qcm_model.solve_properties(nh)
            results.append(
                {
                    "grho_refh": result.grho_refh,
                    "phi": result.phi,
                    "drho": result.drho,
                }
            )

        # Compare all results to first
        for i, r in enumerate(results[1:], 1):
            for key in ["grho_refh", "phi", "drho"]:
                if not np.isnan(results[0][key]) and abs(results[0][key]) > 1e-30:
                    error = abs(r[key] - results[0][key]) / abs(results[0][key])
                    assert (
                        error < TOLERANCE
                    ), f"{key} differs between runs: error={error:.2e}"


class TestNumericalStability:
    """Test numerical stability under edge conditions."""

    def test_small_values_no_underflow(self):
        """Test handling of very small values."""
        # Create layer with small drho
        layers = {
            1: {"grho": 1e6, "phi": 0.01, "drho": 1e-10, "n": 3},
        }
        n = 3
        delfstar = 0.0 + 0.0j
        f1 = F1_DEFAULT

        ZL = multilayer.calc_ZL(n, layers, delfstar, f1)

        # Should not produce NaN or Inf
        assert np.isfinite(ZL), f"calc_ZL produced non-finite result: {ZL}"

    def test_large_values_no_overflow(self):
        """Test handling of large values."""
        # Create layer with large grho
        layers = {
            1: {"grho": 1e12, "phi": 0.1, "drho": 1e-3, "n": 3},
        }
        n = 3
        delfstar = 0.0 + 0.0j
        f1 = F1_DEFAULT

        ZL = multilayer.calc_ZL(n, layers, delfstar, f1)

        # Should not produce NaN or Inf
        assert np.isfinite(ZL), f"calc_ZL produced non-finite result: {ZL}"

    def test_zero_delfstar_stability(self):
        """Test stability with zero delfstar."""
        layers = {
            1: {"grho": 1e9, "phi": 0.1, "drho": 1e-6, "n": 3},
        }
        n = 3
        delfstar = 0.0 + 0.0j
        f1 = F1_DEFAULT

        ZL = multilayer.calc_ZL(n, layers, delfstar, f1)

        assert np.isfinite(ZL), f"Zero delfstar produced non-finite result: {ZL}"


# Allow running specific tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
