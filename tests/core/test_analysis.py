"""
Tests for rheoQCM.core.analysis module (Layer 3 - Scripting Interface).

These tests verify the analysis module functionality including:
    - Import and instantiation
    - Analysis workflow from data to results
    - Backward compatibility with existing scripts
    - Batch processing capability

Test coverage: 6 focused tests for analysis module functionality.
"""

import warnings
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from rheoQCM.core import configure_jax

# Ensure JAX is configured for Float64 before running tests
configure_jax()


class TestImportAndInstantiation:
    """Test import and instantiation of analysis module."""

    def test_import_analysis_module(self) -> None:
        """Test that analysis module can be imported."""
        from rheoQCM.core import analysis

        # Check module has expected attributes
        assert hasattr(analysis, "QCMAnalyzer")
        assert hasattr(analysis, "analyze_delfstar")
        assert hasattr(analysis, "batch_analyze")

    def test_import_physics_functions(self) -> None:
        """Test that physics functions are accessible from analysis module."""
        from rheoQCM.core.analysis import (
            Zq,
            f1_default,
            sauerbreyf,
            sauerbreym,
            grho,
            grhostar,
            calc_delfstar_sla,
            kotula_gstar,
        )

        # Check constants are correct
        assert Zq == 8.84e6
        assert f1_default == 5e6

        # Check functions are callable
        assert callable(sauerbreyf)
        assert callable(sauerbreym)
        assert callable(grho)
        assert callable(grhostar)
        assert callable(calc_delfstar_sla)
        assert callable(kotula_gstar)

    def test_qcm_analyzer_instantiation(self) -> None:
        """Test QCMAnalyzer can be instantiated with various parameters."""
        from rheoQCM.core.analysis import QCMAnalyzer

        # Default instantiation
        analyzer = QCMAnalyzer()
        assert analyzer.f1 == 5e6
        assert analyzer.refh == 3

        # Custom parameters
        analyzer = QCMAnalyzer(f1=4.95e6, refh=5, calctype="LL")
        assert analyzer.f1 == 4.95e6
        assert analyzer.refh == 5

        # Access underlying model
        assert analyzer.model is not None


class TestAnalysisWorkflow:
    """Test complete analysis workflow from data to results."""

    def test_basic_analysis_workflow(self) -> None:
        """Test basic analysis workflow: load data -> analyze -> get results."""
        from rheoQCM.core.analysis import QCMAnalyzer

        analyzer = QCMAnalyzer(f1=5e6, refh=3)

        # Test data for thin film
        delfstars = {
            1: -28206.4782657343 + 5.6326137881j,
            3: -87768.0313369799 + 155.716064797j,
            5: -159742.686586637 + 888.6642467156j,
        }

        # Load data
        analyzer.load_data(delfstars)

        # Analyze
        result = analyzer.analyze(nh=[3, 5, 3], calctype="SLA")

        # Check result structure
        assert "grho_refh" in result
        assert "phi" in result
        assert "drho" in result
        assert "dlam_refh" in result

        # Check reasonable values
        assert result["grho_refh"] > 0
        assert 0 <= result["phi"] <= jnp.pi / 2
        assert result["drho"] > 0

        # Check results are stored
        assert len(analyzer.results) == 1

    def test_one_shot_analysis(self) -> None:
        """Test analyze_delfstar convenience function."""
        from rheoQCM.core.analysis import analyze_delfstar

        delfstars = {
            1: -28206.4782657343 + 5.6326137881j,
            3: -87768.0313369799 + 155.716064797j,
            5: -159742.686586637 + 888.6642467156j,
        }

        result = analyze_delfstar(
            delfstars=delfstars,
            nh=[3, 5, 3],
            f1=5e6,
            refh=3,
        )

        assert result["grho_refh"] > 0
        assert result["drho"] > 0

    def test_result_formatting_and_conversion(self) -> None:
        """Test result formatting and unit conversion."""
        from rheoQCM.core.analysis import QCMAnalyzer

        analyzer = QCMAnalyzer(f1=5e6, refh=3)

        delfstars = {
            1: -28206.4782657343 + 5.6326137881j,
            3: -87768.0313369799 + 155.716064797j,
            5: -159742.686586637 + 888.6642467156j,
        }

        analyzer.load_data(delfstars)
        result = analyzer.analyze(nh=[3, 5, 3])

        # Test formatting
        formatted = analyzer.format_result()
        assert "drho" in formatted
        assert "grho_refh" in formatted

        # Test unit conversion
        converted = analyzer.convert_to_display_units()
        # phi should be in degrees (45 degrees = pi/4 radians)
        assert "phi" in converted


class TestBackwardCompatibility:
    """Test backward compatibility with existing scripts."""

    def test_legacy_function_aliases(self) -> None:
        """Test that legacy function aliases work and raise deprecation warnings."""
        from rheoQCM.core.analysis import (
            calc_drho_from_delf,
            calc_delf_from_drho,
        )

        # Test calc_drho_from_delf (alias for sauerbreym)
        drho = calc_drho_from_delf(3, -1000.0, f1=5e6)
        assert drho > 0
        assert isinstance(drho, np.ndarray)

        # Test calc_delf_from_drho (alias for sauerbreyf)
        delf = calc_delf_from_drho(3, 1e-6, f1=5e6)
        assert delf > 0
        assert isinstance(delf, np.ndarray)

    def test_legacy_grho_function(self) -> None:
        """Test legacy grho function with phi in degrees."""
        from rheoQCM.core.analysis import grho_legacy

        props = {"grho3": 1e8, "phi": 45}  # phi in degrees

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = grho_legacy(5, props)
            # Check deprecation warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

        # Check result is reasonable
        assert result > 0
        assert result > props["grho3"]  # n=5 > n=3

    def test_legacy_grhostar_function(self) -> None:
        """Test legacy grhostar function with phi in degrees."""
        from rheoQCM.core.analysis import grhostar_legacy

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = grhostar_legacy(1e8, 45)  # phi in degrees
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

        # Check result is complex
        assert isinstance(result, complex)
        assert abs(result) > 0

    def test_physics_functions_accessible(self) -> None:
        """Test that physics functions work correctly from analysis module."""
        from rheoQCM.core.analysis import (
            sauerbreyf,
            sauerbreym,
            grho,
            grhostar,
            grho_from_dlam,
        )

        # Test Sauerbrey equations
        n = 3
        drho = 1e-6  # kg/m^2

        delf = sauerbreyf(n, drho)
        drho_back = sauerbreym(n, -float(delf))

        assert jnp.abs(drho - drho_back) < 1e-15

        # Test grho (modern API with radians)
        phi = jnp.pi / 4  # radians
        grho_refh = 1e8

        grho_n = grho(5, grho_refh, phi, refh=3)
        assert grho_n > grho_refh  # Should scale up with n

        # Test grhostar
        gstar = grhostar(grho_n, phi)
        assert jnp.abs(gstar) > 0
        assert jnp.angle(gstar) > 0


class TestBatchProcessing:
    """Test batch processing capability."""

    def test_batch_analysis(self) -> None:
        """Test batch analysis of multiple timepoints."""
        from rheoQCM.core.analysis import QCMAnalyzer

        analyzer = QCMAnalyzer(f1=5e6, refh=3)

        # Create batch data: 5 timepoints with increasing film thickness
        n_timepoints = 5
        batch_delfstars = []

        for i in range(n_timepoints):
            scale = 1.0 + 0.1 * i
            delfstars = {
                1: (-28000.0 * scale) + (5.0 * scale) * 1j,
                3: (-87000.0 * scale) + (155.0 * scale) * 1j,
                5: (-159000.0 * scale) + (888.0 * scale) * 1j,
            }
            batch_delfstars.append(delfstars)

        # Batch analyze
        results = analyzer.analyze_batch(
            batch_delfstars=batch_delfstars,
            nh=[3, 5, 3],
            calctype="SLA",
        )

        # Check results
        assert len(results) == n_timepoints
        assert len(analyzer.results) == n_timepoints

        # All results should have expected keys
        for result in results:
            assert "grho_refh" in result
            assert "phi" in result
            assert "drho" in result

    def test_batch_analyze_convenience_function(self) -> None:
        """Test batch_analyze convenience function."""
        from rheoQCM.core.analysis import batch_analyze

        n_timepoints = 3
        batch_delfstars = []

        for i in range(n_timepoints):
            scale = 1.0 + 0.2 * i
            delfstars = {
                3: (-87000.0 * scale) + (155.0 * scale) * 1j,
                5: (-159000.0 * scale) + (888.0 * scale) * 1j,
                7: (-230000.0 * scale) + (2000.0 * scale) * 1j,
            }
            batch_delfstars.append(delfstars)

        results = batch_analyze(
            batch_delfstars=batch_delfstars,
            nh=[3, 5, 3],
            f1=5e6,
            refh=3,
        )

        assert len(results) == n_timepoints

    def test_clear_results(self) -> None:
        """Test clearing stored results."""
        from rheoQCM.core.analysis import QCMAnalyzer

        analyzer = QCMAnalyzer(f1=5e6, refh=3)

        delfstars = {
            3: -87768.0 + 155.7j,
            5: -159742.7 + 888.7j,
            7: -231000.0 + 2000.0j,
        }

        analyzer.load_data(delfstars)
        analyzer.analyze(nh=[3, 5, 3])
        analyzer.analyze(nh=[3, 5, 3])

        assert len(analyzer.results) == 2

        analyzer.clear_results()
        assert len(analyzer.results) == 0


class TestCurveFittingIntegration:
    """Test curve fitting integration in analysis module."""

    def test_curve_fit_from_analyzer(self) -> None:
        """Test curve fitting through QCMAnalyzer."""
        from rheoQCM.core.analysis import QCMAnalyzer
        import jax

        analyzer = QCMAnalyzer(f1=5e6, refh=3)

        # Define a simple model function
        def linear_model(x: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
            return a * x + b

        # Generate test data
        x_data = jnp.linspace(0, 10, 50)
        true_a, true_b = 2.5, 1.0
        y_true = linear_model(x_data, true_a, true_b)

        # Add noise
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, shape=y_true.shape) * 0.1
        y_data = y_true + noise

        # Fit
        popt, pcov = analyzer.curve_fit(
            linear_model,
            x_data,
            y_data,
            p0=[1.0, 0.0],
        )

        # Check fitted parameters
        assert jnp.abs(popt[0] - true_a) < 0.5
        assert jnp.abs(popt[1] - true_b) < 0.5
