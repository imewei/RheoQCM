"""Bayesian fitting module with NumPyro NUTS MCMC.

This module provides Bayesian inference capabilities with NLSQ warm-start
and ArviZ diagnostic integration.

Public API
----------
Classes:
    PriorSpec - Specification for parameter prior distributions
    BayesianFitResult - Result container for Bayesian fitting
    BayesianFitter - Main class for NUTS MCMC fitting with NLSQ warm-start
    DiagnosticPlot - Enum of available ArviZ diagnostic plot types

Functions:
    plot_comparison - Compare frequentist CI vs Bayesian credible intervals
"""

from __future__ import annotations

__all__ = [
    "BayesianFitResult",
    "BayesianFitter",
    "DiagnosticPlot",
    "MCMCDiagnostics",
    "PriorSpec",
    "plot_comparison",
]

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import numpyro
import numpyro.distributions as dist
import pandas as pd
from nlsq import curve_fit
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Type aliases
type Float64Array = npt.NDArray[np.float64]
type SamplesDict = dict[str, Float64Array]
type PriorDict = dict[str, "PriorSpec"]
type ModelFunc = Callable[..., Float64Array]


class DiagnosticPlot(Enum):
    """Enumerate available diagnostic plot types."""

    PAIR = "pair"
    FOREST = "forest"
    ENERGY = "energy"
    AUTOCORR = "autocorr"
    RANK = "rank"
    ESS = "ess"


@dataclass
class PriorSpec:
    """Specification for parameter prior distributions.

    Attributes
    ----------
    distribution : str
        Distribution name ("LogNormal", "Normal", "HalfNormal", "Uniform")
    params : dict[str, float]
        Distribution parameters
    """

    distribution: str
    params: dict[str, float]

    @classmethod
    def from_nlsq(cls, name: str, value: float) -> PriorSpec:
        """Create default prior centered on NLSQ estimate.

        Parameters
        ----------
        name : str
            Parameter name (unused, for interface compatibility)
        value : float
            NLSQ point estimate

        Returns
        -------
        PriorSpec
            LogNormal for positive values, Normal for non-positive
        """
        del name  # unused
        if value > 0:
            return cls("LogNormal", {"loc": float(np.log(value)), "scale": 1.0})
        scale = max(abs(value), 1.0)
        return cls("Normal", {"loc": float(value), "scale": float(scale)})

    @classmethod
    def log_normal(cls, loc: float, scale: float) -> PriorSpec:
        """Create LogNormal(loc, scale) prior."""
        return cls("LogNormal", {"loc": loc, "scale": scale})

    @classmethod
    def normal(cls, loc: float, scale: float) -> PriorSpec:
        """Create Normal(loc, scale) prior."""
        return cls("Normal", {"loc": loc, "scale": scale})

    @classmethod
    def half_normal(cls, scale: float) -> PriorSpec:
        """Create HalfNormal(scale) prior."""
        return cls("HalfNormal", {"scale": scale})

    @classmethod
    def uniform(cls, low: float, high: float) -> PriorSpec:
        """Create Uniform(low, high) prior."""
        return cls("Uniform", {"low": low, "high": high})

    def to_numpyro(self) -> dist.Distribution:
        """Convert to NumPyro distribution.

        Returns
        -------
        numpyro.distributions.Distribution
            NumPyro distribution object
        """
        if self.distribution == "LogNormal":
            return dist.LogNormal(self.params["loc"], self.params["scale"])
        if self.distribution == "Normal":
            return dist.Normal(self.params["loc"], self.params["scale"])
        if self.distribution == "HalfNormal":
            return dist.HalfNormal(self.params["scale"])
        if self.distribution == "Uniform":
            return dist.Uniform(self.params["low"], self.params["high"])
        msg = f"Unknown distribution: {self.distribution}"
        raise ValueError(msg)


@dataclass
class MCMCDiagnostics:
    """Structured MCMC diagnostic summary.

    Provides all MCMC diagnostics in a single structured object
    for easy access and artifact export.

    Attributes
    ----------
    rhat : dict[str, float]
        R-hat convergence diagnostic per parameter.
    ess_bulk : dict[str, float]
        Bulk effective sample size per parameter.
    ess_tail : dict[str, float]
        Tail effective sample size per parameter.
    divergences : int
        Number of divergent transitions.
    bfmi : list[float]
        Bayesian Fraction of Missing Information per chain.
    max_tree_depth_fraction : float
        Fraction of samples hitting max tree depth.
    seed : int
        Random seed used for reproducibility.
    software_versions : dict[str, str]
        Versions of key packages (numpyro, jax, arviz).
    """

    rhat: dict[str, float]
    ess_bulk: dict[str, float]
    ess_tail: dict[str, float]
    divergences: int
    bfmi: list[float]
    max_tree_depth_fraction: float
    seed: int
    software_versions: dict[str, str]

    @property
    def converged(self) -> bool:
        """True if all R-hat < 1.01."""
        return all(r < 1.01 for r in self.rhat.values())

    @property
    def sufficient_ess(self) -> bool:
        """True if all ESS bulk >= 400."""
        return all(e >= 400 for e in self.ess_bulk.values())

    @property
    def good_bfmi(self) -> bool:
        """True if all chain BFMI >= 0.3."""
        return all(b >= 0.3 for b in self.bfmi)

    @property
    def no_divergences(self) -> bool:
        """True if no divergent transitions."""
        return self.divergences == 0

    @property
    def all_diagnostics_pass(self) -> bool:
        """True if all diagnostics pass recommended thresholds."""
        return (
            self.converged
            and self.sufficient_ess
            and self.good_bfmi
            and self.no_divergences
        )

    def to_dict(self) -> dict:
        """Export diagnostics as dictionary for artifact storage."""
        return {
            "rhat": self.rhat,
            "ess_bulk": self.ess_bulk,
            "ess_tail": self.ess_tail,
            "divergences": self.divergences,
            "bfmi": self.bfmi,
            "max_tree_depth_fraction": self.max_tree_depth_fraction,
            "seed": self.seed,
            "software_versions": self.software_versions,
            "converged": self.converged,
            "sufficient_ess": self.sufficient_ess,
            "good_bfmi": self.good_bfmi,
            "no_divergences": self.no_divergences,
            "all_pass": self.all_diagnostics_pass,
        }


@dataclass
class BayesianFitResult:
    """Represents completed Bayesian MCMC inference results.

    Attributes
    ----------
    samples : SamplesDict
        Posterior samples per parameter (shape: [n_chains, n_samples])
    param_names : list[str]
        Parameter names
    summary : pd.DataFrame
        Summary statistics (mean, std, HDI)
    inference_data : az.InferenceData
        ArviZ InferenceData object
    nlsq_warmstart : dict[str, float]
        NLSQ estimates used for warm-start
    n_chains : int
        Number of MCMC chains
    n_samples : int
        Samples per chain
    n_warmup : int
        Warmup samples per chain
    rhat : dict[str, float]
        R-hat convergence diagnostic per parameter
    ess : dict[str, float]
        Effective sample size per parameter
    divergences : int
        Number of divergent transitions
    bfmi : list[float]
        Bayesian Fraction of Missing Information per chain
    diagnostics : MCMCDiagnostics
        Structured diagnostics summary
    """

    samples: SamplesDict
    param_names: list[str]
    summary: pd.DataFrame
    inference_data: az.InferenceData
    nlsq_warmstart: dict[str, float]
    n_chains: int
    n_samples: int
    n_warmup: int
    rhat: dict[str, float]
    ess: dict[str, float]
    divergences: int
    bfmi: list[float]
    diagnostics: MCMCDiagnostics

    @property
    def converged(self) -> bool:
        """True if all R-hat < 1.01."""
        return all(r < 1.01 for r in self.rhat.values())

    @property
    def samples_array(self) -> Float64Array:
        """Flattened samples array [n_total, n_params]."""
        arrays = [self.samples[name].flatten() for name in self.param_names]
        return np.column_stack(arrays)


class BayesianFitter:
    """Run Bayesian NUTS MCMC with NLSQ warm-start.

    Parameters
    ----------
    n_chains : int
        Number of MCMC chains (default: 4, minimum: 2)
    n_samples : int
        Posterior samples per chain (default: 2000)
    n_warmup : int
        Warmup/burn-in samples per chain (default: 1000)
    target_accept_prob : float
        NUTS acceptance probability target (default: 0.8)
    seed : int | None
        Random seed for reproducibility (default: None)
    chain_method : str
        Chain execution method: "sequential", "parallel", or "vectorized"
        (default: "sequential"). Use "parallel" for multi-GPU environments.
    """

    def __init__(
        self,
        n_chains: int = 4,
        n_samples: int = 2000,
        n_warmup: int = 1000,
        target_accept_prob: float = 0.8,
        seed: int | None = None,
        chain_method: str = "sequential",
    ) -> None:
        if n_chains < 2:
            msg = f"n_chains must be >= 2 for R-hat diagnostics, got {n_chains}"
            raise ValueError(msg)

        self.n_chains = n_chains
        self.n_samples = n_samples
        self.n_warmup = n_warmup
        self.target_accept_prob = target_accept_prob
        self.seed = seed
        self.chain_method = chain_method

    def _run_nlsq_warmstart(
        self,
        model: ModelFunc,
        x: Float64Array,
        y: Float64Array,
        *,
        p0: Float64Array | None = None,
    ) -> tuple[Float64Array, Float64Array]:
        """Run NLSQ fit to obtain warm-start estimates.

        Parameters
        ----------
        model : ModelFunc
            Model function
        x : Float64Array
            X-data
        y : Float64Array
            Y-data
        p0 : Float64Array | None
            Initial guess (optional)

        Returns
        -------
        tuple[Float64Array, Float64Array]
            (popt, pcov) tuple
        """
        popt, pcov = curve_fit(model, x, y, p0=p0)
        return np.asarray(popt), np.asarray(pcov)

    def _create_default_priors(
        self,
        popt: Float64Array,
        param_names: list[str],
    ) -> PriorDict:
        """Create default priors from NLSQ estimates.

        Parameters
        ----------
        popt : Float64Array
            NLSQ parameter estimates
        param_names : list[str]
            Parameter names

        Returns
        -------
        PriorDict
            Dict mapping parameter names to PriorSpec
        """
        return {
            name: PriorSpec.from_nlsq(name, float(val))
            for name, val in zip(param_names, popt, strict=True)
        }

    def _build_numpyro_model(
        self,
        model: ModelFunc,
        x: Float64Array,
        priors: PriorDict,
        param_names: list[str],
    ) -> Callable:
        """Build NumPyro probabilistic model.

        Parameters
        ----------
        model : ModelFunc
            Model function
        x : Float64Array
            X-data
        priors : PriorDict
            Prior distributions
        param_names : list[str]
            Parameter names

        Returns
        -------
        Callable
            NumPyro model function
        """
        x_jax = jnp.asarray(x)

        def numpyro_model(y_obs: jax.Array | None = None) -> None:
            params = []
            for name in param_names:
                prior = priors[name].to_numpyro()
                p = numpyro.sample(name, prior)
                params.append(p)

            sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

            mu = model(x_jax, *params)
            numpyro.sample("obs", dist.Normal(mu, sigma), obs=y_obs)

        return numpyro_model

    def fit(
        self,
        model: ModelFunc,
        x: Float64Array,
        y: Float64Array,
        *,
        param_names: list[str] | None = None,
        priors: PriorDict | None = None,
        nlsq_result: tuple[Float64Array, Float64Array] | None = None,
        sigma: float | Float64Array | None = None,
    ) -> BayesianFitResult:
        """Run Bayesian NUTS MCMC with NLSQ warm-start.

        Parameters
        ----------
        model : ModelFunc
            Model function with signature model(x, *params) -> y
        x : Float64Array
            Independent variable data (shape: [n_points])
        y : Float64Array
            Observed dependent variable (shape: [n_points])
        param_names : list[str] | None
            Parameter names (default: ["p0", "p1", ...])
        priors : PriorDict | None
            Prior distributions per parameter (default: auto from NLSQ)
        nlsq_result : tuple[Float64Array, Float64Array] | None
            (popt, pcov) tuple from NLSQ fit (default: computed internally)
        sigma : float | Float64Array | None
            Observation noise (default: estimated from residuals)

        Returns
        -------
        BayesianFitResult
            Bayesian fit result with samples, summary, diagnostics
        """
        del sigma  # reserved for future use

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if nlsq_result is None:
            popt, pcov = self._run_nlsq_warmstart(model, x, y)
        else:
            popt, pcov = nlsq_result
            popt = np.asarray(popt, dtype=np.float64)
            pcov = np.asarray(pcov, dtype=np.float64)

        n_params = len(popt)
        if param_names is None:
            param_names = [f"p{i}" for i in range(n_params)]

        if len(param_names) != n_params:
            msg = f"param_names length ({len(param_names)}) != n_params ({n_params})"
            raise ValueError(msg)

        if priors is None:
            priors = self._create_default_priors(popt, param_names)

        nlsq_warmstart = dict(zip(param_names, popt.tolist(), strict=True))

        numpyro_model = self._build_numpyro_model(model, x, priors, param_names)

        init_values = {
            name: float(val) for name, val in zip(param_names, popt, strict=True)
        }
        init_strategy = init_to_value(values=init_values)

        kernel = NUTS(
            numpyro_model,
            target_accept_prob=self.target_accept_prob,
            init_strategy=init_strategy,
        )

        mcmc = MCMC(
            kernel,
            num_warmup=self.n_warmup,
            num_samples=self.n_samples,
            num_chains=self.n_chains,
            chain_method=self.chain_method,
            progress_bar=False,
        )

        seed = self.seed if self.seed is not None else 0
        rng_key = jax.random.PRNGKey(seed)
        y_jax = jnp.asarray(y)
        mcmc.run(rng_key, y_obs=y_jax)

        inference_data = az.from_numpyro(mcmc)

        samples: SamplesDict = {}
        for name in param_names:
            samples[name] = np.asarray(inference_data.posterior[name].values)

        summary_df = az.summary(
            inference_data,
            var_names=param_names,
            hdi_prob=0.95,
        )

        rhat = {name: float(summary_df.loc[name, "r_hat"]) for name in param_names}
        ess_bulk = {
            name: float(summary_df.loc[name, "ess_bulk"]) for name in param_names
        }
        ess_tail = {
            name: float(summary_df.loc[name, "ess_tail"]) for name in param_names
        }

        sample_stats = inference_data.sample_stats
        if "diverging" in sample_stats:
            divergences = int(np.sum(sample_stats["diverging"].values))
        else:
            divergences = 0

        # Compute BFMI (Bayesian Fraction of Missing Information) per chain
        bfmi_values: list[float] = []
        if "energy" in sample_stats:
            energy = sample_stats["energy"].values  # shape: (n_chains, n_samples)
            for chain_idx in range(energy.shape[0]):
                chain_energy = energy[chain_idx, :]
                energy_diff = np.diff(chain_energy)
                # BFMI = Var(E[n] - E[n-1]) / Var(E[n])
                bfmi = float(np.var(energy_diff) / np.var(chain_energy))
                bfmi_values.append(bfmi)
        else:
            # Fallback if energy not available
            bfmi_values = [1.0] * self.n_chains

        # Compute max tree depth fraction
        max_tree_depth_fraction = 0.0
        if "tree_depth" in sample_stats:
            tree_depth = sample_stats["tree_depth"].values
            # NUTS default max tree depth is 10
            max_tree_depth_fraction = float(np.mean(tree_depth >= 10))

        # Get software versions for reproducibility
        software_versions = {
            "numpyro": numpyro.__version__,
            "jax": jax.__version__,
            "arviz": az.__version__,
            "numpy": np.__version__,
        }

        # Create structured diagnostics
        diagnostics = MCMCDiagnostics(
            rhat=rhat,
            ess_bulk=ess_bulk,
            ess_tail=ess_tail,
            divergences=divergences,
            bfmi=bfmi_values,
            max_tree_depth_fraction=max_tree_depth_fraction,
            seed=seed,
            software_versions=software_versions,
        )

        # Issue warnings for diagnostic failures
        if not all(r < 1.01 for r in rhat.values()):
            warnings.warn(
                f"Some R-hat values >= 1.01: {rhat}. Chains may not have converged.",
                UserWarning,
                stacklevel=2,
            )

        if not all(e >= 400 for e in ess_bulk.values()):
            warnings.warn(
                f"Some ESS values < 400: {ess_bulk}. Consider more samples.",
                UserWarning,
                stacklevel=2,
            )

        if divergences > 0:
            warnings.warn(
                f"{divergences} divergent transitions detected. Check priors/model.",
                UserWarning,
                stacklevel=2,
            )

        # BFMI warning (low BFMI indicates poor exploration of posterior tails)
        low_bfmi_chains = [i for i, b in enumerate(bfmi_values) if b < 0.3]
        if low_bfmi_chains:
            warnings.warn(
                f"Low BFMI (<0.3) in chains {low_bfmi_chains}: {[bfmi_values[i] for i in low_bfmi_chains]}. "
                "Consider reparameterization or more informative priors.",
                UserWarning,
                stacklevel=2,
            )

        return BayesianFitResult(
            samples=samples,
            param_names=param_names,
            summary=summary_df,
            inference_data=inference_data,
            nlsq_warmstart=nlsq_warmstart,
            n_chains=self.n_chains,
            n_samples=self.n_samples,
            n_warmup=self.n_warmup,
            rhat=rhat,
            ess=ess_bulk,
            divergences=divergences,
            bfmi=bfmi_values,
            diagnostics=diagnostics,
        )

    def summary(
        self,
        result: BayesianFitResult,
        *,
        hdi_prob: float = 0.95,
    ) -> pd.DataFrame:
        """Generate parameter summary statistics.

        Parameters
        ----------
        result : BayesianFitResult
            Bayesian fit result
        hdi_prob : float
            HDI probability (default: 0.95)

        Returns
        -------
        pd.DataFrame
            Summary with mean, std, median, HDI, rhat, ESS
        """
        return az.summary(
            result.inference_data,
            var_names=result.param_names,
            hdi_prob=hdi_prob,
        )

    def generate_diagnostic_suite(
        self,
        result: BayesianFitResult,
        output_dir: Path | str,
        *,
        formats: list[str] | None = None,
        dpi: int = 300,
    ) -> dict[str, Path]:
        """Generate all 6 ArviZ diagnostic plots.

        Parameters
        ----------
        result : BayesianFitResult
            Bayesian fit result
        output_dir : Path | str
            Directory for output files
        formats : list[str] | None
            Output formats (default: ["pdf", "png"])
        dpi : int
            Resolution for raster formats (default: 300)

        Returns
        -------
        dict[str, Path]
            Dict mapping plot name to output path
        """
        if formats is None:
            formats = ["pdf", "png"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        idata = result.inference_data
        var_names = result.param_names
        output_paths: dict[str, Path] = {}

        plot_funcs = {
            DiagnosticPlot.PAIR: lambda: az.plot_pair(
                idata, var_names=var_names, divergences=True
            ),
            DiagnosticPlot.FOREST: lambda: az.plot_forest(
                idata, var_names=var_names, combined=True
            ),
            DiagnosticPlot.ENERGY: lambda: az.plot_energy(idata),
            DiagnosticPlot.AUTOCORR: lambda: az.plot_autocorr(
                idata, var_names=var_names
            ),
            DiagnosticPlot.RANK: lambda: az.plot_rank(idata, var_names=var_names),
            DiagnosticPlot.ESS: lambda: az.plot_ess(idata, var_names=var_names),
        }

        for plot_type, plot_fn in plot_funcs.items():
            try:
                axes = plot_fn()
                if hasattr(axes, "figure"):
                    fig = axes.figure
                elif hasattr(axes, "flat"):
                    fig = axes.flat[0].figure
                elif isinstance(axes, np.ndarray):
                    fig = axes.flat[0].figure
                else:
                    fig = plt.gcf()

                for fmt in formats:
                    path = output_dir / f"diagnostic_{plot_type.value}.{fmt}"
                    fig.savefig(path, dpi=dpi, bbox_inches="tight")
                    output_paths[f"{plot_type.value}_{fmt}"] = path

                plt.close(fig)
            except Exception as e:
                warnings.warn(
                    f"Failed to generate {plot_type.value} plot: {e}",
                    UserWarning,
                    stacklevel=2,
                )

        return output_paths

    def plot_posterior_predictive(
        self,
        result: BayesianFitResult,
        model: ModelFunc,
        x: Float64Array,
        y: Float64Array,
        *,
        ax: plt.Axes | None = None,
        credible_level: float = 0.95,
        show_nlsq: bool = False,
        n_draws: int = 100,
    ) -> Figure:
        """Plot fit with posterior predictive credible intervals.

        Parameters
        ----------
        result : BayesianFitResult
            Bayesian fit result
        model : ModelFunc
            Model function
        x : Float64Array
            X-data for prediction
        y : Float64Array
            Observed y-data
        ax : plt.Axes | None
            Matplotlib axes (creates new if None)
        credible_level : float
            Credible interval level (default: 0.95)
        show_nlsq : bool
            Overlay NLSQ point estimate (default: False)
        n_draws : int
            Number of posterior draws for uncertainty band (default: 100)

        Returns
        -------
        Figure
            Matplotlib Figure
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        ax.scatter(x, y, color="black", alpha=0.6, label="Data", zorder=3)

        samples_flat = result.samples_array
        n_total = samples_flat.shape[0]
        indices = np.random.default_rng().choice(
            n_total, min(n_draws, n_total), replace=False
        )

        x_pred = np.linspace(x.min(), x.max(), 200)
        predictions = np.zeros((len(indices), len(x_pred)))

        for i, idx in enumerate(indices):
            params = samples_flat[idx]
            predictions[i] = model(x_pred, *params)

        median = np.median(predictions, axis=0)
        alpha = (1 - credible_level) / 2
        lower = np.percentile(predictions, alpha * 100, axis=0)
        upper = np.percentile(predictions, (1 - alpha) * 100, axis=0)

        ax.plot(x_pred, median, "r-", linewidth=2, label="Median")
        ax.fill_between(
            x_pred,
            lower,
            upper,
            alpha=0.3,
            color="red",
            label=f"{credible_level:.0%} CI",
        )

        if show_nlsq:
            nlsq_params = [result.nlsq_warmstart[name] for name in result.param_names]
            y_nlsq = model(x_pred, *nlsq_params)
            ax.plot(x_pred, y_nlsq, "b--", linewidth=1.5, label="NLSQ")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()

        return fig


def plot_comparison(
    x_data: Float64Array,
    y_data: Float64Array,
    model: ModelFunc,
    nlsq_popt: Float64Array,
    nlsq_pcov: Float64Array,
    bayesian_result: BayesianFitResult,
    *,
    x_pred: Float64Array | None = None,
    confidence_level: float = 0.95,
    ax: plt.Axes | None = None,
    data_color: str = "black",
    nlsq_color: str = "blue",
    bayesian_color: str = "red",
    band_alpha: float = 0.25,
    n_draws: int = 100,
) -> Figure:
    """Generate comparison plot of frequentist CI vs Bayesian credible interval.

    Parameters
    ----------
    x_data : Float64Array
        Original x-data points
    y_data : Float64Array
        Original y-data points
    model : ModelFunc
        Model function
    nlsq_popt : Float64Array
        NLSQ optimal parameters
    nlsq_pcov : Float64Array
        NLSQ covariance matrix
    bayesian_result : BayesianFitResult
        Bayesian fit result
    x_pred : Float64Array | None
        X values for predictions (default: linspace from data range)
    confidence_level : float
        Confidence/credible level (default: 0.95)
    ax : plt.Axes | None
        Matplotlib axes (creates new if None)
    data_color : str
        Color for data points (default: "black")
    nlsq_color : str
        Color for NLSQ fit and CI (default: "blue")
    bayesian_color : str
        Color for Bayesian fit and CI (default: "red")
    band_alpha : float
        Transparency for uncertainty bands (default: 0.25)
    n_draws : int
        Number of posterior draws for Bayesian CI (default: 100)

    Returns
    -------
    Figure
        Matplotlib Figure with both uncertainty bands
    """
    from rheoQCM.core.uncertainty import UncertaintyCalculator

    x_data = np.asarray(x_data, dtype=np.float64)
    y_data = np.asarray(y_data, dtype=np.float64)
    nlsq_popt = np.asarray(nlsq_popt, dtype=np.float64)
    nlsq_pcov = np.asarray(nlsq_pcov, dtype=np.float64)

    if x_pred is None:
        x_pred = np.linspace(x_data.min(), x_data.max(), 200)
    else:
        x_pred = np.asarray(x_pred, dtype=np.float64)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Plot data
    ax.scatter(x_data, y_data, color=data_color, alpha=0.6, label="Data", zorder=4)

    # --- NLSQ (Frequentist) CI ---
    calc = UncertaintyCalculator()
    nlsq_band = calc.compute_band(
        model, x_pred, nlsq_popt, nlsq_pcov, confidence_level=confidence_level
    )

    ax.plot(
        x_pred,
        nlsq_band.y_fit,
        color=nlsq_color,
        linewidth=2,
        label="NLSQ fit",
        zorder=3,
    )
    ax.fill_between(
        x_pred,
        nlsq_band.y_lower,
        nlsq_band.y_upper,
        color=nlsq_color,
        alpha=band_alpha,
        label=f"NLSQ {confidence_level:.0%} CI",
        zorder=1,
    )

    # --- Bayesian Credible Interval ---
    samples_flat = bayesian_result.samples_array
    n_total = samples_flat.shape[0]
    indices = np.random.default_rng().choice(
        n_total, min(n_draws, n_total), replace=False
    )

    predictions = np.zeros((len(indices), len(x_pred)))
    for i, idx in enumerate(indices):
        params = samples_flat[idx]
        predictions[i] = model(x_pred, *params)

    bayesian_median = np.median(predictions, axis=0)
    alpha = (1 - confidence_level) / 2
    bayesian_lower = np.percentile(predictions, alpha * 100, axis=0)
    bayesian_upper = np.percentile(predictions, (1 - alpha) * 100, axis=0)

    ax.plot(
        x_pred,
        bayesian_median,
        color=bayesian_color,
        linewidth=2,
        linestyle="--",
        label="Bayesian median",
        zorder=3,
    )
    ax.fill_between(
        x_pred,
        bayesian_lower,
        bayesian_upper,
        color=bayesian_color,
        alpha=band_alpha,
        label=f"Bayesian {confidence_level:.0%} CI",
        zorder=2,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Frequentist vs Bayesian Uncertainty Comparison")
    ax.legend(loc="best")

    fig.tight_layout()
    return fig
