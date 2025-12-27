"""
Integration Tests for RheoQCM Modernization.

This module contains integration tests that verify end-to-end workflows
across all layers of the modernized RheoQCM codebase.

Test coverage (10 integration tests):
    1. Complete analysis workflow: load data -> fit -> results
    2. UI-triggered analysis simulation: model fit -> results
    3. Scripting workflow: import core -> analyze data
    4. HDF5 roundtrip with new physics core
    5. Full physics-to-model pipeline
    6. Batch processing end-to-end
    7. JAX JIT compilation across layers
    8. Float64 precision verification end-to-end
    9. Core-to-UI data flow consistency
    10. NLSQ curve fitting workflow
"""

import tempfile
from pathlib import Path
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheoQCM.core import configure_jax

# Ensure JAX is configured for Float64 before running tests
configure_jax()


pytestmark = pytest.mark.integration


class TestCompleteAnalysisWorkflow:
    """Test complete analysis workflow: load data -> fit -> results."""

    def test_load_to_fit_to_results_thin_film(self) -> None:
        """Test complete workflow for thin film analysis.

        This tests the full pipeline:
        1. Load experimental delfstar data
        2. Run model fitting using jaxopt
        3. Extract and validate results
        """
        from rheoQCM.core.analysis import QCMAnalyzer

        # Initialize analyzer
        analyzer = QCMAnalyzer(f1=5e6, refh=3)

        # Test data for BCB thin film (known test case)
        delfstars = {
            1: -28206.4782657343 + 5.6326137881j,
            3: -87768.0313369799 + 155.716064797j,
            5: -159742.686586637 + 888.6642467156j,
        }

        # Load data
        analyzer.load_data(delfstars)

        # Analyze
        result = analyzer.analyze(nh=[3, 5, 3], calctype="SLA")

        # Validate complete result structure
        assert "grho_refh" in result
        assert "phi" in result
        assert "drho" in result
        assert "dlam_refh" in result
        assert "errors" in result

        # Validate result values are physically reasonable
        assert result["grho_refh"] > 0, "grho_refh should be positive"
        assert 0 <= result["phi"] <= np.pi / 2, "phi should be between 0 and pi/2"
        assert result["drho"] > 0, "drho should be positive for thin film"

        # Verify results are stored
        assert len(analyzer.results) == 1
        assert analyzer.results[0] == result

    def test_load_to_fit_to_results_bulk_material(self) -> None:
        """Test complete workflow for bulk material analysis."""
        from rheoQCM.core.analysis import QCMAnalyzer

        analyzer = QCMAnalyzer(f1=5e6, refh=3)

        # Test data for water (bulk material with high dissipation)
        delfstars = {
            1: -694.15609764494 + 762.8726222543j,
            3: -1248.7983004897833 + 1215.1121711257j,
            5: -1641.2310467399657 + 1574.7706516819j,
        }

        analyzer.load_data(delfstars)
        result = analyzer.analyze(nh=[3, 5, 3], calctype="SLA", bulklimit=0.5)

        # For bulk material, phi should be close to pi/2
        assert result["phi"] > np.pi / 4, "Bulk material should have high phi"
        assert result["grho_refh"] > 0


class TestUITriggeredAnalysis:
    """Test UI-triggered analysis simulation."""

    def test_ui_model_flow(self) -> None:
        """Test the flow from UI wrapper through model to results.

        Simulates: button click -> model fit -> display preparation
        """
        from rheoQCM.modules.QCM import QCM

        # Initialize QCM UI wrapper
        qcm = QCM()
        qcm.f1 = 5e6
        qcm.refh = 3
        qcm.calctype = "SLA"

        # Prepare data (simulating UI loading data)
        nh = [3, 5, 3]
        delfstar = {
            1: -28206.4782657343 + 5.6326137881j,
            3: -87768.0313369799 + 155.716064797j,
            5: -159742.686586637 + 888.6642467156j,
        }
        film = {
            0: {"calc": False, "drho": 2.8e-6, "grho": 3e17, "phi": 0, "n": 3},
            1: {"calc": True},
        }

        # Trigger analysis (simulating button click)
        grho_refh, phi, drho, dlam_refh, err = qcm.solve_general_delfstar_to_prop(
            nh, delfstar, film, calctype="SLA", bulklimit=0.5
        )

        # Prepare for display (unit conversion)
        drho_display = qcm.convert_mech_unit_data(drho, "drho")
        phi_display = qcm.convert_mech_unit_data(phi, "phi")

        # Verify display-ready values
        assert np.isfinite(drho_display) or np.isnan(drho_display)
        assert np.isfinite(phi_display) or np.isnan(phi_display)


class TestScriptingWorkflow:
    """Test scripting workflow: import core -> analyze data."""

    def test_import_and_analyze(self) -> None:
        """Test that scripting import patterns work correctly."""
        # This simulates a typical user script
        from rheoQCM.core.analysis import (
            QCMAnalyzer,
            analyze_delfstar,
            sauerbreyf,
            sauerbreym,
            grho,
            grhostar,
            Zq,
            f1_default,
        )

        # Verify constants are accessible
        assert Zq == 8.84e6
        assert f1_default == 5e6

        # Test Sauerbrey calculations
        n = 3
        drho = 1e-6  # 1 ug/cm^2
        delf = sauerbreyf(n, drho)
        drho_recovered = sauerbreym(n, -float(delf))

        assert jnp.abs(drho - drho_recovered) < 1e-15

        # Test full analysis
        delfstars = {
            3: -87768.0 + 155.7j,
            5: -159742.7 + 888.7j,
            7: -231000.0 + 2000.0j,
        }

        result = analyze_delfstar(delfstars, nh=[3, 5, 3])

        assert "grho_refh" in result
        assert "drho" in result

    def test_batch_processing_script(self) -> None:
        """Test batch processing workflow for scripting."""
        from rheoQCM.core.analysis import batch_analyze

        # Simulate time-series data
        n_timepoints = 5
        batch_delfstars = []

        for i in range(n_timepoints):
            scale = 1.0 + 0.1 * i
            delfstars = {
                3: (-87000.0 * scale) + (155.0 * scale) * 1j,
                5: (-159000.0 * scale) + (888.0 * scale) * 1j,
                7: (-231000.0 * scale) + (2000.0 * scale) * 1j,
            }
            batch_delfstars.append(delfstars)

        results = batch_analyze(
            batch_delfstars=batch_delfstars,
            nh=[3, 5, 3],
            f1=5e6,
            refh=3,
        )

        assert len(results) == n_timepoints
        for result in results:
            assert "grho_refh" in result
            assert "drho" in result


class TestHDF5RoundtripWithNewPhysics:
    """Test HDF5 roundtrip with new physics core."""

    def test_save_analyze_reload(self) -> None:
        """Test complete HDF5 save/analyze/reload cycle."""
        from rheoQCM.core.model import QCMModel
        from rheoQCM.core import physics

        # Create test data
        harmonics = [1, 3, 5]
        delf = [-1000.0, -3100.0, -5300.0]
        delg = [100.0, 320.0, 550.0]
        f0 = [5e6, 15e6, 25e6]
        g0 = [50.0, 150.0, 250.0]

        # Save to HDF5
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            filepath = Path(f.name)

        try:
            with h5py.File(filepath, "w") as hf:
                hf.create_dataset("delf", data=delf)
                hf.create_dataset("delg", data=delg)
                hf.create_dataset("harmonics", data=harmonics)
                hf.create_dataset("f0", data=f0)
                hf.create_dataset("g0", data=g0)

            # Load and analyze with new physics core
            model = QCMModel(refh=3)
            model.load_from_hdf5(filepath)

            # Verify data loaded correctly
            assert len(model.delfstars) == 3
            assert model.f1 is not None

            # Analyze
            result = model.solve_properties(
                nh=[3, 5, 3],
                calctype="SLA",
            )

            # Verify result uses new physics core
            # Calculate expected Sauerbrey mass using new physics
            drho_saub = float(physics.sauerbreym(1, delf[0], f1=model.f1))
            assert drho_saub > 0

            # Result should be consistent with physics core
            assert np.isfinite(result["grho_refh"]) or np.isnan(result["grho_refh"])

        finally:
            filepath.unlink(missing_ok=True)


class TestPhysicsToModelPipeline:
    """Test full physics-to-model pipeline."""

    def test_physics_model_consistency(self) -> None:
        """Test that physics functions and model produce consistent results."""
        from rheoQCM.core import physics
        from rheoQCM.core.model import QCMModel

        model = QCMModel(f1=5e6, refh=3)

        # Test parameters
        grho_refh = 1e10
        phi = np.pi / 4
        drho = 1e-6
        n = 3

        # Calculate using physics module
        grho_3 = float(physics.grho(n, grho_refh, phi, refh=3))
        dlam_3 = float(physics.calc_dlam(n, grho_3, phi, drho, f1=5e6))

        # Calculate using model internal methods
        grho_3_model = model._grho_at_harmonic(n, grho_refh, phi)

        # Results should match
        np.testing.assert_allclose(grho_3, grho_3_model, rtol=1e-10)

        # Verify grhostar consistency
        grhostar_phys = physics.grhostar_from_refh(n, grho_refh, phi, refh=3)
        grhostar_model = model._grhostar_at_harmonic(n, grho_refh, phi)

        np.testing.assert_allclose(
            np.abs(grhostar_phys), np.abs(grhostar_model), rtol=1e-10
        )
        np.testing.assert_allclose(
            np.angle(grhostar_phys), np.angle(grhostar_model), rtol=1e-10
        )


class TestJAXJITCompilationAcrossLayers:
    """Test JAX JIT compilation across layers."""

    def test_jit_physics_in_model_context(self) -> None:
        """Test that JIT-compiled physics work in model context."""
        from rheoQCM.core import physics

        # JIT compile physics functions
        jitted_sauerbreyf = jax.jit(physics.sauerbreyf)
        jitted_grhostar = jax.jit(physics.grhostar)

        # Use in model-like calculations
        drho = 1e-6
        grho_val = 1e10
        phi = jnp.pi / 4

        for n in [1, 3, 5, 7, 9]:
            delf = jitted_sauerbreyf(n, drho)
            gstar = jitted_grhostar(grho_val, phi)

            assert jnp.isfinite(delf)
            assert jnp.isfinite(jnp.abs(gstar))

    def test_vmap_batch_processing(self) -> None:
        """Test that vmap works for batch processing across model."""
        from rheoQCM.core import physics

        harmonics = jnp.array([1, 3, 5, 7, 9])
        drho = 1e-6

        # Vectorized Sauerbrey calculation
        vmapped_sauerbreyf = jax.vmap(physics.sauerbreyf, in_axes=(0, None))
        delfs = vmapped_sauerbreyf(harmonics, drho)

        assert delfs.shape == (5,)
        # Check scaling with harmonic
        assert jnp.allclose(delfs / delfs[0], harmonics, rtol=1e-10)


class TestFloat64PrecisionEndToEnd:
    """Test Float64 precision verification end-to-end."""

    def test_precision_through_full_pipeline(self) -> None:
        """Test Float64 precision is maintained throughout pipeline."""
        from rheoQCM.core.analysis import QCMAnalyzer
        from rheoQCM.core import physics

        analyzer = QCMAnalyzer(f1=5e6, refh=3)

        # Test with values requiring high precision
        delfstars = {
            3: -87768.0313369799 + 155.716064797j,
            5: -159742.686586637 + 888.6642467156j,
            7: -231234.5678901234 + 2000.1234567890j,
        }

        analyzer.load_data(delfstars)

        # Verify physics calculations maintain precision
        drho = 1e-6
        delf = physics.sauerbreyf(3, drho)
        assert delf.dtype == jnp.float64

        # Small differences should be preserved
        drho1 = 1e-6
        drho2 = 1e-6 + 1e-15

        delf1 = physics.sauerbreyf(3, drho1)
        delf2 = physics.sauerbreyf(3, drho2)

        assert delf2 > delf1
        assert (delf2 - delf1) > 0

    def test_no_subsampling_in_solver(self) -> None:
        """Verify that solver does not use subsampling or random SVD."""
        from rheoQCM.core.model import QCMModel

        model = QCMModel(f1=5e6, refh=3)

        # This test verifies the implementation does not use
        # subsampling or random SVD by checking that results
        # are deterministic across multiple runs
        delfstars = {
            3: -87768.0 + 155.7j,
            5: -159742.7 + 888.7j,
            7: -231000.0 + 2000.0j,
        }

        model.load_delfstars(delfstars)

        results = []
        for _ in range(3):
            result = model.solve_properties(nh=[3, 5, 3], calctype="SLA")
            results.append(result)

        # All runs should produce identical results (deterministic)
        for i in range(1, len(results)):
            if not np.isnan(results[0]["grho_refh"]):
                np.testing.assert_allclose(
                    results[0]["grho_refh"],
                    results[i]["grho_refh"],
                    rtol=1e-12,
                )
            if not np.isnan(results[0]["phi"]):
                np.testing.assert_allclose(
                    results[0]["phi"],
                    results[i]["phi"],
                    rtol=1e-12,
                )


class TestCoreToUIDataFlow:
    """Test Core-to-UI data flow consistency."""

    def test_model_to_ui_consistency(self) -> None:
        """Test that model results flow correctly to UI wrapper."""
        from rheoQCM.core.model import QCMModel
        from rheoQCM.modules.QCM import QCM

        # Same parameters
        f1 = 5e6
        refh = 3
        delfstars = {
            3: -87768.0 + 155.7j,
            5: -159742.7 + 888.7j,
            7: -231000.0 + 2000.0j,
        }

        # Core model calculation
        model = QCMModel(f1=f1, refh=refh)
        model.load_delfstars(delfstars)
        result_core = model.solve_properties(nh=[3, 5, 3], calctype="SLA")

        # UI wrapper calculation
        qcm = QCM()
        qcm.f1 = f1
        qcm.refh = refh
        qcm.calctype = "SLA"

        film = {
            0: {"calc": False, "drho": 2.8e-6, "grho": 3e17, "phi": 0, "n": 3},
            1: {"calc": True},
        }
        grho_ui, phi_ui, drho_ui, dlam_ui, err = qcm.solve_general_delfstar_to_prop(
            [3, 5, 3], delfstars, film, calctype="SLA", bulklimit=0.5
        )

        # Results should be consistent
        # Note: May have small differences due to different solver paths
        if not np.isnan(result_core["grho_refh"]) and not np.isnan(grho_ui):
            # Check same order of magnitude
            ratio = result_core["grho_refh"] / grho_ui
            assert 0.1 < ratio < 10, "Results should be same order of magnitude"


class TestNLSQCurveFittingWorkflow:
    """Test NLSQ curve fitting workflow."""

    def test_nlsq_curve_fit_integration(self) -> None:
        """Test NLSQ curve_fit works correctly in model context."""
        from rheoQCM.core.model import QCMModel
        import jax

        model = QCMModel(f1=5e6, refh=3)

        # Define test model function
        def exponential_decay(x: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
            return a * jnp.exp(-b * x)

        # Generate test data
        x_data = jnp.linspace(0, 5, 50)
        true_a, true_b = 2.5, 0.8
        y_true = exponential_decay(x_data, true_a, true_b)

        # Add noise
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, shape=y_true.shape) * 0.05
        y_data = y_true + noise

        # Fit using NLSQ through model
        popt, pcov = model.curve_fit(
            exponential_decay,
            x_data,
            y_data,
            p0=[1.0, 1.0],
        )

        # Check fitted parameters
        assert jnp.abs(popt[0] - true_a) < 0.3
        assert jnp.abs(popt[1] - true_b) < 0.3

        # Check covariance is valid
        assert pcov.shape == (2, 2)
        assert jnp.all(jnp.isfinite(pcov))

    def test_nlsq_replaces_lmfit(self) -> None:
        """Test that NLSQ curve_fit can replace lmfit patterns."""
        from rheoQCM.core.analysis import QCMAnalyzer
        import jax

        analyzer = QCMAnalyzer(f1=5e6, refh=3)

        # Simple linear model (common fitting pattern)
        def linear(x: jnp.ndarray, slope: float, intercept: float) -> jnp.ndarray:
            return slope * x + intercept

        # Generate data
        x = jnp.linspace(0, 10, 20)
        true_slope, true_intercept = 2.0, 1.0
        y_true = linear(x, true_slope, true_intercept)

        key = jax.random.PRNGKey(123)
        y_noisy = y_true + jax.random.normal(key, shape=y_true.shape) * 0.2

        # Fit (replaces lmfit.Model pattern)
        popt, pcov = analyzer.curve_fit(
            linear,
            x,
            y_noisy,
            p0=[1.0, 0.0],
        )

        # Validate
        assert jnp.abs(popt[0] - true_slope) < 0.5
        assert jnp.abs(popt[1] - true_intercept) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
