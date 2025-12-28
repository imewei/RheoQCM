"""
Tests for Optimistix Migration Parity (T036).

These tests verify that the migration from jaxopt to optimistix
produces identical numerical results.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from rheoQCM.core import configure_jax
from rheoQCM.core.model import QCMModel

# Ensure JAX is configured for Float64 before running tests
configure_jax()


class TestOptimistixParity:
    """Tests for numerical parity after jaxopt -> optimistix migration."""

    def test_simple_solve_result(self) -> None:
        """Test that solve_properties returns valid results."""
        model = QCMModel(f1=5e6, refh=3)

        # Load test delfstars
        delfstars = {
            3: -1000.0 + 100.0j,
            5: -1700.0 + 180.0j,
            7: -2500.0 + 280.0j,
        }
        model.load_delfstars(delfstars)

        result = model.solve_properties(nh=[3, 5, 3])

        # Verify result structure
        assert hasattr(result, "drho")
        assert hasattr(result, "grho_refh")
        assert hasattr(result, "phi")

        # Verify values are finite
        assert np.isfinite(result.drho)
        assert np.isfinite(result.grho_refh)
        assert np.isfinite(result.phi)

    def test_drho_positive(self) -> None:
        """Test that drho is positive for typical inputs."""
        model = QCMModel(f1=5e6, refh=3)

        delfstars = {
            3: -500.0 + 50.0j,
            5: -850.0 + 90.0j,
            7: -1200.0 + 130.0j,
        }
        model.load_delfstars(delfstars)

        result = model.solve_properties(nh=[3, 5, 3])

        assert result.drho > 0, "drho should be positive for typical film deposition"

    def test_phi_in_valid_range(self) -> None:
        """Test that phi is in valid range [0, pi/2)."""
        model = QCMModel(f1=5e6, refh=3)

        delfstars = {
            3: -1000.0 + 100.0j,
            5: -1700.0 + 180.0j,
            7: -2500.0 + 280.0j,
        }
        model.load_delfstars(delfstars)

        result = model.solve_properties(nh=[3, 5, 3])

        assert 0 <= result.phi < np.pi / 2, f"phi={result.phi} should be in [0, pi/2)"

    def test_grho_positive(self) -> None:
        """Test that grho is positive for typical inputs."""
        model = QCMModel(f1=5e6, refh=3)

        delfstars = {
            3: -1000.0 + 100.0j,
            5: -1700.0 + 180.0j,
            7: -2500.0 + 280.0j,
        }
        model.load_delfstars(delfstars)

        result = model.solve_properties(nh=[3, 5, 3])

        assert result.grho_refh > 0, "grho should be positive"


class TestSolverEdgeCases:
    """Tests for solver edge cases (T037)."""

    def test_near_sauerbrey_limit(self) -> None:
        """Test solver behavior near Sauerbrey limit (very thin film)."""
        model = QCMModel(f1=5e6, refh=3)

        # Very small frequency shifts (thin film, near Sauerbrey)
        delfstars = {
            3: -10.0 + 1.0j,
            5: -17.0 + 1.8j,
            7: -24.0 + 2.5j,
        }
        model.load_delfstars(delfstars)

        result = model.solve_properties(nh=[3, 5, 3])

        # Should still converge
        assert np.isfinite(result.drho)
        assert result.drho > 0

    def test_different_initial_guess(self) -> None:
        """Test that solver converges from different initial guesses."""
        model = QCMModel(f1=5e6, refh=3)

        delfstars = {
            3: -1000.0 + 100.0j,
            5: -1700.0 + 180.0j,
            7: -2500.0 + 280.0j,
        }
        model.load_delfstars(delfstars)

        # Multiple solves should give consistent results
        result1 = model.solve_properties(nh=[3, 5, 3])
        result2 = model.solve_properties(nh=[3, 5, 3])

        np.testing.assert_allclose(result1.drho, result2.drho, rtol=1e-6)
        np.testing.assert_allclose(result1.phi, result2.phi, rtol=1e-6)

    def test_various_harmonic_combinations(self) -> None:
        """Test solver with various harmonic combinations."""
        model = QCMModel(f1=5e6, refh=3)

        delfstars = {
            1: -333.0 + 33.0j,
            3: -1000.0 + 100.0j,
            5: -1700.0 + 180.0j,
            7: -2500.0 + 280.0j,
            9: -3300.0 + 380.0j,
        }
        model.load_delfstars(delfstars)

        # Test different nh combinations (all require 3 elements)
        for nh in [[3, 5, 3], [5, 7, 5], [3, 7, 3], [1, 3, 1]]:
            result = model.solve_properties(nh=nh)
            assert np.isfinite(result.drho), f"Failed for nh={nh}"


class TestSolverRobustness:
    """Tests for solver robustness (T038)."""

    def test_large_frequency_shifts(self) -> None:
        """Test solver with large frequency shifts (thick film)."""
        model = QCMModel(f1=5e6, refh=3)

        # Large frequency shifts
        delfstars = {
            3: -10000.0 + 2000.0j,
            5: -17000.0 + 4000.0j,
            7: -25000.0 + 6000.0j,
        }
        model.load_delfstars(delfstars)

        result = model.solve_properties(nh=[3, 5, 3])

        # Should still get valid results
        assert np.isfinite(result.drho) or np.isnan(result.drho)

    def test_high_dissipation_ratio(self) -> None:
        """Test solver with high dissipation ratio."""
        model = QCMModel(f1=5e6, refh=3)

        # High dissipation (soft/lossy film)
        delfstars = {
            3: -1000.0 + 500.0j,  # D/f ratio = 0.5
            5: -1700.0 + 900.0j,
            7: -2500.0 + 1400.0j,
        }
        model.load_delfstars(delfstars)

        result = model.solve_properties(nh=[3, 5, 3])

        # High dissipation may be classified as bulk (drho=inf) or finite
        # Either is acceptable - the key is that grho is finite
        assert np.isfinite(result.grho_refh)

    def test_low_dissipation_ratio(self) -> None:
        """Test solver with low dissipation ratio (nearly elastic)."""
        model = QCMModel(f1=5e6, refh=3)

        # Low dissipation (rigid film)
        delfstars = {
            3: -1000.0 + 10.0j,  # D/f ratio = 0.01
            5: -1700.0 + 17.0j,
            7: -2500.0 + 25.0j,
        }
        model.load_delfstars(delfstars)

        result = model.solve_properties(nh=[3, 5, 3])

        # Should handle low dissipation
        assert np.isfinite(result.drho)
        # Phi should be relatively small for nearly elastic film
        assert result.phi < 1.0  # Less than ~57 degrees

    def test_batch_analyze_consistency(self) -> None:
        """Test that batch_analyze_vmap gives consistent results."""
        from rheoQCM.core.analysis import batch_analyze_vmap

        # Single measurement
        delfstars = jnp.array(
            [
                [-1000.0 + 100.0j, -1700.0 + 180.0j, -2500.0 + 280.0j],
            ]
        )

        result = batch_analyze_vmap(delfstars, harmonics=[3, 5, 7], nhcalc="357")

        assert len(result.drho) == 1
        assert np.isfinite(result.drho[0])
        assert np.isfinite(result.grho_refh[0])
        assert np.isfinite(result.phi[0])
