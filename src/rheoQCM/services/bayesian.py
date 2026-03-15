"""Bayesian fitting service protocol and default implementation.

Provides a service boundary between the GUI layer (rheoQCM.py) and the
Bayesian fitting engine (core/bayesian.py). This enables:
- Testing GUI logic without running actual MCMC
- Swapping Bayesian backends
- Managing fit state outside the god object
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from rheoQCM.core.constants import MCMC_INTERACTIVE, MCMCConfig

if TYPE_CHECKING:
    from rheoQCM.core.bayesian import BayesianFitResult, MCMCDiagnostics

logger = logging.getLogger(__name__)


@runtime_checkable
class BayesianService(Protocol):
    """Protocol for Bayesian fitting operations."""

    def fit_marked(
        self,
        data: pd.DataFrame,
        harmonics: list[int],
        nhcalc: str,
        config: MCMCConfig = MCMC_INTERACTIVE,
    ) -> BayesianFitResult:
        """Run Bayesian fit on marked data points.

        Parameters
        ----------
        data : pd.DataFrame
            QCM data with marked points.
        harmonics : list[int]
            Harmonic numbers to fit.
        nhcalc : str
            Harmonic calculation string (e.g., "357").
        config : MCMCConfig
            MCMC configuration preset.

        Returns
        -------
        BayesianFitResult
            Fit result with samples, diagnostics, and summary.
        """
        ...

    def get_last_result(self) -> BayesianFitResult | None:
        """Get the most recent Bayesian fit result."""
        ...

    def get_diagnostics(self) -> MCMCDiagnostics | None:
        """Get diagnostics from the most recent fit."""
        ...


class DefaultBayesianService:
    """Default implementation wrapping BayesianFitter.

    Manages fit state (last result, fitter instance) that was previously
    scattered across QCMApp attributes.
    """

    def __init__(self, config: MCMCConfig = MCMC_INTERACTIVE) -> None:
        self._config = config
        self._fitter: Any | None = None
        self._last_result: BayesianFitResult | None = None
        self._show_uncertainty_bands: bool = True
        self._confidence_level: float = 0.95

    def fit_marked(
        self,
        data: pd.DataFrame,
        harmonics: list[int],
        nhcalc: str,
        config: MCMCConfig | None = None,
    ) -> BayesianFitResult:
        """Run Bayesian fit using BayesianFitter.

        Lazily creates fitter on first call.
        """
        from rheoQCM.core.bayesian import BayesianFitter

        cfg = config or self._config
        self._fitter = BayesianFitter(
            n_chains=cfg.n_chains,
            n_samples=cfg.n_samples,
            n_warmup=cfg.n_warmup,
            seed=cfg.seed,
            chain_method=cfg.chain_method,
        )

        # Build model function and extract data for fit
        # This is a simplified version; full integration requires
        # QCMModel setup from the data DataFrame
        model_fn, x_data, y_data, param_names = self._prepare_from_data(
            data,
            harmonics,
            nhcalc,
        )

        self._last_result = self._fitter.fit(
            model=model_fn,
            x=x_data,
            y=y_data,
            param_names=param_names,
        )
        return self._last_result

    def get_last_result(self) -> BayesianFitResult | None:
        """Get the most recent Bayesian fit result."""
        return self._last_result

    def get_diagnostics(self) -> MCMCDiagnostics | None:
        """Get diagnostics from the most recent fit."""
        if self._last_result is None:
            return None
        return self._last_result.diagnostics

    @staticmethod
    def _prepare_from_data(
        data: pd.DataFrame,
        harmonics: list[int],
        nhcalc: str,
    ) -> tuple[Any, np.ndarray, np.ndarray, list[str]]:
        """Extract model function and data arrays from DataFrame.

        This is a placeholder for the full extraction logic that currently
        lives in rheoQCM.mech_bayesian_fit_marked(). A complete migration
        would move that logic here.
        """
        msg = (
            "DefaultBayesianService._prepare_from_data is a placeholder. "
            "Use QCMApp.mech_bayesian_fit_marked() for full functionality."
        )
        raise NotImplementedError(msg)
