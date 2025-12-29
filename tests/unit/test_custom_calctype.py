"""Tests for custom calctype registration and execution.

Feature: 011-tech-debt-cleanup (US5)

This module tests that custom calctypes registered via register_calctype()
are properly executed without falling back to SLA.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from rheoQCM.core.model import QCMModel, SolveResult


class TestCustomCalctypeExecution:
    """Tests for custom calctype execution (SC-007)."""

    def setup_method(self):
        """Create a model with experimental data for testing."""
        self.model = QCMModel(f1=5e6, refh=3)
        # Load typical thin film data
        self.model.load_delfstars({
            3: -1000 + 100j,
            5: -1700 + 180j,
        })

    def test_register_calctype_adds_to_supported(self):
        """Custom calctype appears in supported list after registration."""
        def dummy_residual(params, delfstar_exp, harmonics, f1, refh, Zq):
            return jnp.zeros(3)

        initial = self.model.get_supported_calctypes()
        assert "TestCustom" not in initial

        self.model.register_calctype("TestCustom", dummy_residual)

        after = self.model.get_supported_calctypes()
        assert "TestCustom" in after

    def test_custom_calctype_residual_is_called(self):
        """Custom residual function is actually invoked."""
        call_count = [0]  # Use list to allow modification in closure

        def counting_residual(params, delfstar_exp, harmonics, f1, refh, Zq):
            call_count[0] += 1
            # Return residual that will converge quickly
            grho = params[0]
            phi = params[1]
            drho = params[2]
            # Simple residual - just return small values
            return jnp.array([0.0, 0.0, 0.0])

        self.model.register_calctype("CountingCalc", counting_residual)
        result = self.model.solve_properties(nh=[3, 5, 3], calctype="CountingCalc")

        # The residual should have been called at least once
        assert call_count[0] > 0, "Custom residual was not called"

    def test_custom_calctype_no_sla_fallback_warning(self, caplog):
        """Custom calctype does not emit SLA fallback warning (SC-007)."""
        import logging

        def simple_residual(params, delfstar_exp, harmonics, f1, refh, Zq):
            return jnp.array([0.0, 0.0, 0.0])

        self.model.register_calctype("NoFallback", simple_residual)

        with caplog.at_level(logging.WARNING):
            result = self.model.solve_properties(nh=[3, 5, 3], calctype="NoFallback")

        # Should NOT see the fallback warning
        fallback_warnings = [
            r for r in caplog.records
            if "falling back to SLA" in r.message
        ]
        assert len(fallback_warnings) == 0, (
            f"Custom calctype should not fall back to SLA. "
            f"Got warnings: {[r.message for r in fallback_warnings]}"
        )

    def test_custom_calctype_returns_solve_result(self):
        """Custom calctype returns proper SolveResult."""
        def valid_residual(params, delfstar_exp, harmonics, f1, refh, Zq):
            return jnp.array([0.0, 0.0, 0.0])

        self.model.register_calctype("ValidCalc", valid_residual)
        result = self.model.solve_properties(nh=[3, 5, 3], calctype="ValidCalc")

        assert isinstance(result, SolveResult)
        # Result should have the expected attributes
        assert hasattr(result, "success")
        assert hasattr(result, "grho_refh")
        assert hasattr(result, "phi")
        assert hasattr(result, "drho")

    def test_custom_calctype_with_physics_residual(self):
        """Custom calctype with realistic physics residual produces results."""
        def physics_residual(params, delfstar_exp, harmonics, f1, refh, Zq):
            """A simple SLA-like residual for testing."""
            grho_refh = params[0]
            phi = params[1]
            drho = params[2]

            # Get experimental values
            n1, n2, n3 = harmonics
            delf_n1 = delfstar_exp.get(n1, 0.0)
            delf_n3 = delfstar_exp.get(n3, 0.0)

            # Simple SLA calculation (thin film approximation)
            # delf* ~ -f1 * n * drho / (pi * Zq)
            predicted_ratio = drho / 1e-6 if drho > 0 else 1.0

            # Compute residuals
            r1 = jnp.abs(delf_n1.real) - jnp.abs(grho_refh) * 0.001
            r2 = phi - 0.1
            r3 = drho - 1e-6

            return jnp.array([r1, r2, r3])

        self.model.register_calctype("SimplePhysics", physics_residual)
        result = self.model.solve_properties(nh=[3, 5, 3], calctype="SimplePhysics")

        assert result.success or result.message is not None
        # If successful, should have finite values
        if result.success:
            assert np.isfinite(result.grho_refh)
            assert np.isfinite(result.phi)
            assert np.isfinite(result.drho)


class TestCustomCalctypeErrorHandling:
    """Tests for error handling in custom calctypes."""

    def setup_method(self):
        """Create a model for testing."""
        self.model = QCMModel(f1=5e6, refh=3)
        self.model.load_delfstars({3: -1000 + 100j, 5: -1700 + 180j})

    def test_invalid_residual_returns_failure(self):
        """Custom residual that raises returns failed SolveResult."""
        def bad_residual(params, delfstar_exp, harmonics, f1, refh, Zq):
            raise ValueError("Intentional test error")

        self.model.register_calctype("BadCalc", bad_residual)
        result = self.model.solve_properties(nh=[3, 5, 3], calctype="BadCalc")

        # Should return failure, not raise
        assert isinstance(result, SolveResult)
        assert not result.success
        assert result.message is not None

    def test_nan_residual_handled_gracefully(self):
        """Custom residual returning NaN is handled."""
        def nan_residual(params, delfstar_exp, harmonics, f1, refh, Zq):
            return jnp.array([jnp.nan, jnp.nan, jnp.nan])

        self.model.register_calctype("NanCalc", nan_residual)
        result = self.model.solve_properties(nh=[3, 5, 3], calctype="NanCalc")

        # Should return a result (possibly failed) not raise
        assert isinstance(result, SolveResult)
