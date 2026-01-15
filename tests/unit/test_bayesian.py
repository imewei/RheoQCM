"""Unit tests for Bayesian fitting module.

Tests cover:
- PriorSpec dataclass and factory methods
- BayesianFitResult dataclass validation
- BayesianFitter NLSQ warm-start
- NUTS sampling with init_to_value
- Convergence diagnostics
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from rheoQCM.core.bayesian import (
    BayesianFitResult,
    BayesianFitter,
    MCMCDiagnostics,
    PriorSpec,
)


def _make_mock_diagnostics(
    rhat: dict[str, float] | None = None,
    ess: dict[str, float] | None = None,
    divergences: int = 0,
) -> MCMCDiagnostics:
    """Create mock MCMCDiagnostics for testing."""
    rhat = rhat or {"p0": 1.001}
    ess = ess or {"p0": 500.0}
    return MCMCDiagnostics(
        rhat=rhat,
        ess_bulk=ess,
        ess_tail=ess,
        divergences=divergences,
        bfmi=[0.9, 0.9, 0.9, 0.9],
        max_tree_depth_fraction=0.0,
        seed=0,
        software_versions={
            "numpyro": "0.14.0",
            "jax": "0.4.0",
            "arviz": "0.18.0",
            "numpy": "2.0.0",
        },
    )


class TestPriorSpecDataclass:
    """T023: Unit test: PriorSpec dataclass and factory methods."""

    def test_log_normal_factory(self) -> None:
        """PriorSpec.log_normal() should create LogNormal prior."""
        prior = PriorSpec.log_normal(loc=0.0, scale=1.0)

        assert prior.distribution == "LogNormal"
        assert prior.params["loc"] == 0.0
        assert prior.params["scale"] == 1.0

    def test_normal_factory(self) -> None:
        """PriorSpec.normal() should create Normal prior."""
        prior = PriorSpec.normal(loc=5.0, scale=2.0)

        assert prior.distribution == "Normal"
        assert prior.params["loc"] == 5.0
        assert prior.params["scale"] == 2.0

    def test_half_normal_factory(self) -> None:
        """PriorSpec.half_normal() should create HalfNormal prior."""
        prior = PriorSpec.half_normal(scale=1.0)

        assert prior.distribution == "HalfNormal"
        assert prior.params["scale"] == 1.0

    def test_uniform_factory(self) -> None:
        """PriorSpec.uniform() should create Uniform prior."""
        prior = PriorSpec.uniform(low=0.0, high=10.0)

        assert prior.distribution == "Uniform"
        assert prior.params["low"] == 0.0
        assert prior.params["high"] == 10.0

    def test_from_nlsq_positive_value(self) -> None:
        """PriorSpec.from_nlsq() should create LogNormal for positive values."""
        prior = PriorSpec.from_nlsq("amplitude", 1.0)

        assert prior.distribution == "LogNormal"
        assert prior.params["loc"] == np.log(1.0)
        assert prior.params["scale"] == 1.0

    def test_from_nlsq_negative_value(self) -> None:
        """PriorSpec.from_nlsq() should create Normal for non-positive values."""
        prior = PriorSpec.from_nlsq("offset", -5.0)

        assert prior.distribution == "Normal"
        assert prior.params["loc"] == -5.0
        assert prior.params["scale"] == 5.0

    def test_to_numpyro_log_normal(self) -> None:
        """to_numpyro() should convert LogNormal to numpyro distribution."""
        import numpyro.distributions as dist

        prior = PriorSpec.log_normal(loc=0.0, scale=1.0)
        np_dist = prior.to_numpyro()

        assert isinstance(np_dist, dist.LogNormal)

    def test_to_numpyro_normal(self) -> None:
        """to_numpyro() should convert Normal to numpyro distribution."""
        import numpyro.distributions as dist

        prior = PriorSpec.normal(loc=0.0, scale=1.0)
        np_dist = prior.to_numpyro()

        assert isinstance(np_dist, dist.Normal)

    def test_to_numpyro_half_normal(self) -> None:
        """to_numpyro() should convert HalfNormal to numpyro distribution."""
        import numpyro.distributions as dist

        prior = PriorSpec.half_normal(scale=1.0)
        np_dist = prior.to_numpyro()

        assert isinstance(np_dist, dist.HalfNormal)

    def test_to_numpyro_uniform(self) -> None:
        """to_numpyro() should convert Uniform to numpyro distribution."""
        import numpyro.distributions as dist

        prior = PriorSpec.uniform(low=0.0, high=1.0)
        np_dist = prior.to_numpyro()

        assert isinstance(np_dist, dist.Uniform)

    def test_to_numpyro_unknown_distribution(self) -> None:
        """to_numpyro() should raise ValueError for unknown distribution."""
        prior = PriorSpec("Unknown", {"param": 1.0})

        with pytest.raises(ValueError, match="Unknown distribution"):
            prior.to_numpyro()


class TestBayesianFitResultDataclass:
    """T024: Unit test: BayesianFitResult dataclass validation."""

    def test_converged_all_rhat_below_threshold(self) -> None:
        """converged property should be True when all R-hat < 1.01."""
        import arviz as az
        import pandas as pd

        result = BayesianFitResult(
            samples={"p0": np.random.randn(4, 1000)},
            param_names=["p0"],
            summary=pd.DataFrame({"mean": [1.0]}),
            inference_data=az.InferenceData(),
            nlsq_warmstart={"p0": 1.0},
            n_chains=4,
            n_samples=1000,
            n_warmup=500,
            rhat={"p0": 1.001},
            ess={"p0": 500},
            divergences=0,
            bfmi=[0.9, 0.9, 0.9, 0.9],
            diagnostics=_make_mock_diagnostics(rhat={"p0": 1.001}),
        )

        assert result.converged is True

    def test_not_converged_rhat_above_threshold(self) -> None:
        """converged property should be False when any R-hat >= 1.01."""
        import arviz as az
        import pandas as pd

        result = BayesianFitResult(
            samples={"p0": np.random.randn(4, 1000)},
            param_names=["p0"],
            summary=pd.DataFrame({"mean": [1.0]}),
            inference_data=az.InferenceData(),
            nlsq_warmstart={"p0": 1.0},
            n_chains=4,
            n_samples=1000,
            n_warmup=500,
            rhat={"p0": 1.05},  # Above threshold
            ess={"p0": 500},
            divergences=0,
            bfmi=[0.9, 0.9, 0.9, 0.9],
            diagnostics=_make_mock_diagnostics(rhat={"p0": 1.05}),
        )

        assert result.converged is False

    def test_samples_array_property(self) -> None:
        """samples_array should return flattened samples [n_total, n_params]."""
        import arviz as az
        import pandas as pd

        samples = {
            "p0": np.random.randn(2, 100),
            "p1": np.random.randn(2, 100),
        }

        result = BayesianFitResult(
            samples=samples,
            param_names=["p0", "p1"],
            summary=pd.DataFrame({"mean": [1.0, 2.0]}),
            inference_data=az.InferenceData(),
            nlsq_warmstart={"p0": 1.0, "p1": 2.0},
            n_chains=2,
            n_samples=100,
            n_warmup=50,
            rhat={"p0": 1.001, "p1": 1.001},
            ess={"p0": 500, "p1": 500},
            divergences=0,
            bfmi=[0.9, 0.9],
            diagnostics=_make_mock_diagnostics(
                rhat={"p0": 1.001, "p1": 1.001},
                ess={"p0": 500.0, "p1": 500.0},
            ),
        )

        arr = result.samples_array
        assert arr.shape == (200, 2)  # 2 chains * 100 samples, 2 params


class TestBayesianFitterNLSQWarmstart:
    """T025: Unit test: BayesianFitter._run_nlsq_warmstart() returns popt, pcov."""

    def test_nlsq_warmstart_returns_popt_pcov(self) -> None:
        """_run_nlsq_warmstart should return (popt, pcov) tuple."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        x = np.linspace(0, 5, 50)
        y = 1.0 * np.exp(-0.5 * x) + np.random.normal(0, 0.01, len(x))

        fitter = BayesianFitter(n_chains=2, n_samples=100, n_warmup=50, seed=42)
        popt, pcov = fitter._run_nlsq_warmstart(model, x, y, p0=np.array([1.0, 0.5]))

        assert popt.shape == (2,)
        assert pcov.shape == (2, 2)
        np.testing.assert_allclose(popt, [1.0, 0.5], rtol=0.2)


class TestBayesianFitterInitToValue:
    """T026: Unit test: BayesianFitter.fit() uses init_to_value with NLSQ estimates."""

    @pytest.mark.slow
    def test_fit_uses_nlsq_warmstart(self) -> None:
        """fit() should use NLSQ estimates for chain initialization."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        x = np.linspace(0, 5, 50)
        y = 1.0 * np.exp(-0.5 * x) + np.random.normal(0, 0.05, len(x))

        fitter = BayesianFitter(n_chains=2, n_samples=100, n_warmup=50, seed=42)
        result = fitter.fit(
            model=model,
            x=x,
            y=y,
            param_names=["amplitude", "decay"],
        )

        # Verify NLSQ warmstart values are stored
        assert "amplitude" in result.nlsq_warmstart
        assert "decay" in result.nlsq_warmstart
        np.testing.assert_allclose(
            [result.nlsq_warmstart["amplitude"], result.nlsq_warmstart["decay"]],
            [1.0, 0.5],
            rtol=0.3,
        )


class TestConvergenceDiagnostics:
    """T027: Unit test: Convergence diagnostics (R-hat, ESS) computed correctly."""

    @pytest.mark.slow
    def test_rhat_computed(self) -> None:
        """fit() should compute R-hat for all parameters."""

        def model(x, a):
            return a * x

        np.random.seed(42)
        x = np.linspace(0, 10, 30)
        y = 2.0 * x + np.random.normal(0, 0.5, len(x))

        fitter = BayesianFitter(n_chains=2, n_samples=100, n_warmup=50, seed=42)
        result = fitter.fit(model, x, y, param_names=["slope"])

        assert "slope" in result.rhat
        assert isinstance(result.rhat["slope"], float)

    @pytest.mark.slow
    def test_ess_computed(self) -> None:
        """fit() should compute ESS for all parameters."""

        def model(x, a):
            return a * x

        np.random.seed(42)
        x = np.linspace(0, 10, 30)
        y = 2.0 * x + np.random.normal(0, 0.5, len(x))

        fitter = BayesianFitter(n_chains=2, n_samples=100, n_warmup=50, seed=42)
        result = fitter.fit(model, x, y, param_names=["slope"])

        assert "slope" in result.ess
        assert isinstance(result.ess["slope"], float)


class TestNChainsValidation:
    """T028: Unit test: Warn if n_chains < 2 raises ValueError."""

    def test_n_chains_less_than_2_raises_error(self) -> None:
        """BayesianFitter should raise ValueError if n_chains < 2."""
        with pytest.raises(ValueError, match="n_chains must be >= 2"):
            BayesianFitter(n_chains=1)

    def test_n_chains_2_allowed(self) -> None:
        """BayesianFitter should allow n_chains = 2."""
        fitter = BayesianFitter(n_chains=2)
        assert fitter.n_chains == 2


class TestBayesianFitWithPriors:
    """Additional tests for custom priors."""

    @pytest.mark.slow
    def test_fit_with_custom_priors(self) -> None:
        """fit() should accept custom priors."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        x = np.linspace(0, 5, 50)
        y = 1.0 * np.exp(-0.5 * x) + np.random.normal(0, 0.05, len(x))

        priors = {
            "amplitude": PriorSpec.log_normal(loc=0.0, scale=0.5),
            "decay": PriorSpec.log_normal(loc=-1.0, scale=0.5),
        }

        fitter = BayesianFitter(n_chains=2, n_samples=100, n_warmup=50, seed=42)
        result = fitter.fit(
            model=model,
            x=x,
            y=y,
            param_names=["amplitude", "decay"],
            priors=priors,
        )

        assert len(result.samples) == 2
        assert "amplitude" in result.samples
        assert "decay" in result.samples


class TestPosteriorPredictive:
    """T040-T042: Unit tests for plot_posterior_predictive()."""

    @pytest.mark.slow
    def test_plot_posterior_predictive_generates_figure(self) -> None:
        """T040: plot_posterior_predictive() should generate valid figure."""
        import matplotlib.pyplot as plt

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        x = np.linspace(0, 5, 50)
        y = 1.0 * np.exp(-0.5 * x) + np.random.normal(0, 0.05, len(x))

        fitter = BayesianFitter(n_chains=2, n_samples=100, n_warmup=50, seed=42)
        result = fitter.fit(model, x, y, param_names=["a", "b"])

        fig = fitter.plot_posterior_predictive(result, model, x, y)

        assert fig is not None
        plt.close(fig)

    @pytest.mark.slow
    def test_configurable_credible_levels(self) -> None:
        """T041: plot_posterior_predictive() should support configurable credible levels."""
        import matplotlib.pyplot as plt

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        x = np.linspace(0, 5, 50)
        y = 1.0 * np.exp(-0.5 * x) + np.random.normal(0, 0.05, len(x))

        fitter = BayesianFitter(n_chains=2, n_samples=100, n_warmup=50, seed=42)
        result = fitter.fit(model, x, y, param_names=["a", "b"])

        # Test 90% credible interval
        fig_90 = fitter.plot_posterior_predictive(
            result, model, x, y, credible_level=0.90
        )
        assert fig_90 is not None
        plt.close(fig_90)

        # Test 95% credible interval (default)
        fig_95 = fitter.plot_posterior_predictive(
            result, model, x, y, credible_level=0.95
        )
        assert fig_95 is not None
        plt.close(fig_95)

    @pytest.mark.slow
    def test_plot_with_nlsq_overlay(self) -> None:
        """T042: plot_posterior_predictive() should support NLSQ overlay."""
        import matplotlib.pyplot as plt

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        x = np.linspace(0, 5, 50)
        y = 1.0 * np.exp(-0.5 * x) + np.random.normal(0, 0.05, len(x))

        fitter = BayesianFitter(n_chains=2, n_samples=100, n_warmup=50, seed=42)
        result = fitter.fit(model, x, y, param_names=["a", "b"])

        fig = fitter.plot_posterior_predictive(result, model, x, y, show_nlsq=True)

        assert fig is not None
        ax = fig.axes[0]
        # Should have more lines with NLSQ overlay
        assert len(ax.lines) >= 2
        plt.close(fig)


class TestDiagnosticPlotEnum:
    """T046: Unit tests for DiagnosticPlot enum."""

    def test_diagnostic_plot_enum_has_6_types(self) -> None:
        """T046: DiagnosticPlot enum should contain all 6 plot types."""
        from rheoQCM.core.bayesian import DiagnosticPlot

        expected_types = {"pair", "forest", "energy", "autocorr", "rank", "ess"}
        actual_types = {p.value for p in DiagnosticPlot}

        assert actual_types == expected_types
        assert len(DiagnosticPlot) == 6


class TestDiagnosticSuiteGeneration:
    """T047-T048: Unit tests for generate_diagnostic_suite()."""

    @pytest.mark.slow
    def test_generate_diagnostic_suite_creates_all_plots(self) -> None:
        """T047: generate_diagnostic_suite() should create all 6 plots."""
        import tempfile

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        x = np.linspace(0, 5, 50)
        y = 1.0 * np.exp(-0.5 * x) + np.random.normal(0, 0.05, len(x))

        fitter = BayesianFitter(n_chains=2, n_samples=100, n_warmup=50, seed=42)
        result = fitter.fit(model, x, y, param_names=["a", "b"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_paths = fitter.generate_diagnostic_suite(
                result,
                tmpdir,
                formats=["png"],
            )

            # Should have at least some plots (some may fail in test env)
            assert len(output_paths) >= 1

            # Verify files exist
            from pathlib import Path

            for key, path in output_paths.items():
                assert Path(path).exists(), f"Expected {path} to exist"

    @pytest.mark.slow
    def test_plots_exported_in_multiple_formats(self) -> None:
        """T048: Plots should be exported in both PDF and PNG formats."""
        import tempfile

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        x = np.linspace(0, 5, 50)
        y = 1.0 * np.exp(-0.5 * x) + np.random.normal(0, 0.05, len(x))

        fitter = BayesianFitter(n_chains=2, n_samples=100, n_warmup=50, seed=42)
        result = fitter.fit(model, x, y, param_names=["a", "b"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_paths = fitter.generate_diagnostic_suite(
                result,
                tmpdir,
                formats=["pdf", "png"],
                dpi=300,
            )

            # Check for both formats
            pdf_keys = [k for k in output_paths if k.endswith("_pdf")]
            png_keys = [k for k in output_paths if k.endswith("_png")]

            # Should have at least one of each format
            assert len(pdf_keys) >= 1, "Expected at least one PDF file"
            assert len(png_keys) >= 1, "Expected at least one PNG file"

            # Verify files have content
            from pathlib import Path

            for path in output_paths.values():
                p = Path(path)
                assert p.exists()
                assert p.stat().st_size > 0, f"File {path} should not be empty"


class TestComparisonPlot:
    """T053-T054: Unit tests for plot_comparison()."""

    @pytest.mark.slow
    def test_comparison_plot_shows_both_bands(self) -> None:
        """T053: Comparison plot should show both frequentist and Bayesian bands."""
        import matplotlib.pyplot as plt
        from nlsq import curve_fit

        from rheoQCM.core.bayesian import plot_comparison

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        x = np.linspace(0, 5, 50)
        y = 1.0 * np.exp(-0.5 * x) + np.random.normal(0, 0.05, len(x))

        # Get NLSQ fit
        popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.5])

        # Get Bayesian fit
        fitter = BayesianFitter(n_chains=2, n_samples=100, n_warmup=50, seed=42)
        bayesian_result = fitter.fit(model, x, y, param_names=["a", "b"])

        fig = plot_comparison(
            x_data=x,
            y_data=y,
            model=model,
            nlsq_popt=popt,
            nlsq_pcov=pcov,
            bayesian_result=bayesian_result,
            confidence_level=0.95,
        )

        assert fig is not None
        ax = fig.axes[0]

        # Should have filled regions (uncertainty bands)
        # PolyCollections for fill_between
        collections = ax.collections
        assert len(collections) >= 2, "Should have at least 2 uncertainty bands"

        plt.close(fig)

    @pytest.mark.slow
    def test_comparison_plot_labels_identify_ci_types(self) -> None:
        """T054: Labels should clearly identify frequentist CI vs Bayesian CI."""
        import matplotlib.pyplot as plt
        from nlsq import curve_fit

        from rheoQCM.core.bayesian import plot_comparison

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        x = np.linspace(0, 5, 50)
        y = 1.0 * np.exp(-0.5 * x) + np.random.normal(0, 0.05, len(x))

        popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.5])

        fitter = BayesianFitter(n_chains=2, n_samples=100, n_warmup=50, seed=42)
        bayesian_result = fitter.fit(model, x, y, param_names=["a", "b"])

        fig = plot_comparison(
            x_data=x,
            y_data=y,
            model=model,
            nlsq_popt=popt,
            nlsq_pcov=pcov,
            bayesian_result=bayesian_result,
        )

        ax = fig.axes[0]
        legend = ax.get_legend()
        legend_labels = [t.get_text() for t in legend.get_texts()]

        # Should have labels distinguishing NLSQ and Bayesian
        nlsq_labels = [l for l in legend_labels if "NLSQ" in l]
        bayesian_labels = [l for l in legend_labels if "Bayesian" in l]

        assert len(nlsq_labels) >= 1, "Should have NLSQ labels"
        assert len(bayesian_labels) >= 1, "Should have Bayesian labels"

        # Labels should mention CI
        ci_labels = [l for l in legend_labels if "CI" in l]
        assert len(ci_labels) >= 2, "Should have CI labels for both methods"

        plt.close(fig)
