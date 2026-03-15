"""Centralized configuration constants for RheoQCM solvers.

MCMC sampling defaults are organized into three tiers:
- INTERACTIVE: Fast feedback for GUI use (n_chains=2, reduced samples)
- ANALYSIS: Balanced for programmatic API use
- PRODUCTION: Full inference for BayesianFitter (4 chains, 2000 samples)

Solver tolerances for Optimistix LevenbergMarquardt.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MCMCConfig:
    """Immutable configuration for MCMC sampling parameters."""

    n_chains: int
    n_samples: int
    n_warmup: int
    target_accept_prob: float = 0.8
    seed: int | None = None
    chain_method: str = "sequential"


# Three intentional tiers
MCMC_INTERACTIVE = MCMCConfig(n_chains=2, n_samples=500, n_warmup=250, seed=42)
MCMC_ANALYSIS = MCMCConfig(n_chains=4, n_samples=1000, n_warmup=500)
MCMC_PRODUCTION = MCMCConfig(n_chains=4, n_samples=2000, n_warmup=1000)

# Solver tolerances (Optimistix LevenbergMarquardt)
SOLVER_RTOL: float = 1e-8
SOLVER_ATOL: float = 1e-8

# Posterior visualization
POSTERIOR_N_DRAWS: int = 100

# Bayesian diagnostics thresholds
ESS_BULK_MIN: int = 400
RHAT_WARN: float = 1.01  # Warning level: chains may not have converged well
RHAT_MAX: float = 1.05  # Hard limit: results should be rejected above this
BFMI_MIN: float = 0.3
