# Bayesian Fitting Tutorial

This tutorial covers uncertainty quantification in QCM-D analysis using
RheoQCM's Bayesian inference capabilities with NumPyro MCMC.

## Why Bayesian Analysis?

Traditional least-squares fitting provides point estimates but limited
uncertainty information. Bayesian analysis provides:

- **Full posterior distributions** for all parameters
- **Credible intervals** with specified probability
- **Parameter correlations** revealing dependencies
- **Model comparison** via evidence/likelihood

## Quick Start

```python
from rheoQCM.core import QCMModel, configure_jax
from rheoQCM.core.bayesian import BayesianFitter, PriorSpec

configure_jax()

# Create model and load data
model = QCMModel(f1=5e6, refh=3)
model.load_delfstars({
    3: -1000 + 100j,
    5: -1700 + 180j,
    7: -2500 + 280j,
})

# Create Bayesian fitter
fitter = BayesianFitter(model)

# Run MCMC
result = fitter.fit(
    num_samples=2000,
    num_warmup=500,
)

# Get summary statistics
print(result.summary())

# Get 95% credible intervals
drho_ci = result.credible_interval('drho', 0.95)
print(f"drho: {drho_ci[0]:.3e} to {drho_ci[1]:.3e}")
```

## Setting Up Priors

Priors encode your knowledge about reasonable parameter ranges. RheoQCM
provides sensible defaults, but you can customize them:

### Default Priors

```python
# Default priors (log-uniform for scale parameters)
fitter = BayesianFitter(model)
# Uses:
#   drho: LogUniform(1e-7, 1e-3)
#   grho: LogUniform(1e5, 1e11)
#   phi:  Uniform(0.01, 1.55)
```

### Custom Priors

```python
from rheoQCM.core.bayesian import PriorSpec

# Define custom priors
priors = {
    'drho': PriorSpec(
        distribution='lognormal',
        params={'loc': -10, 'scale': 1},  # log-scale parameters
    ),
    'grho_refh': PriorSpec(
        distribution='lognormal',
        params={'loc': 18, 'scale': 2},
    ),
    'phi': PriorSpec(
        distribution='truncated_normal',
        params={'loc': 0.3, 'scale': 0.2, 'low': 0.01, 'high': 1.55},
    ),
}

fitter = BayesianFitter(model, priors=priors)
```

### Available Distributions

| Distribution | Parameters | Use Case |
|--------------|------------|----------|
| `uniform` | `low`, `high` | No prior knowledge |
| `loguniform` | `low`, `high` | Scale parameters |
| `normal` | `loc`, `scale` | Known approximate value |
| `lognormal` | `loc`, `scale` | Positive scale parameters |
| `truncated_normal` | `loc`, `scale`, `low`, `high` | Bounded with peak |

## Running MCMC

### Basic Configuration

```python
result = fitter.fit(
    num_samples=2000,   # Number of posterior samples
    num_warmup=500,     # Warmup/burn-in samples
    num_chains=4,       # Parallel chains
)
```

### Advanced Configuration

```python
result = fitter.fit(
    num_samples=4000,
    num_warmup=1000,
    num_chains=4,
    target_accept_prob=0.8,  # NUTS acceptance probability
    max_tree_depth=10,       # Maximum tree depth
    init_strategy='nlsq',    # Start from least-squares solution
)
```

### Initialization Strategies

| Strategy | Description |
|----------|-------------|
| `'nlsq'` | Start from least-squares solution (recommended) |
| `'prior'` | Sample from prior distributions |
| `'uniform'` | Uniform random within bounds |

## Analyzing Results

### Summary Statistics

```python
# Print full summary
print(result.summary())

# Output:
#                   mean       std    median     5.0%    95.0%
# drho          1.77e-05  2.1e-07  1.76e-05  1.43e-05  2.11e-05
# grho_refh     1.23e+08  5.2e+06  1.22e+08  1.14e+08  1.32e+08
# phi              0.197    0.012     0.196     0.177     0.217
```

### Point Estimates

```python
# Mean of posterior
drho_mean = result.mean('drho')

# Median (more robust)
drho_median = result.median('drho')

# Maximum a posteriori (MAP)
drho_map = result.map_estimate('drho')
```

### Credible Intervals

```python
# 95% credible interval
ci_95 = result.credible_interval('drho', 0.95)
print(f"95% CI: [{ci_95[0]:.3e}, {ci_95[1]:.3e}]")

# 68% credible interval (like 1-sigma)
ci_68 = result.credible_interval('drho', 0.68)
```

### Posterior Samples

```python
# Get all samples for custom analysis
samples = result.samples

drho_samples = samples['drho']      # Shape: (num_samples * num_chains,)
grho_samples = samples['grho_refh']
phi_samples = samples['phi']

# Compute derived quantities
import numpy as np

# Example: compute G' and G'' posteriors
gprime_samples = grho_samples * np.cos(phi_samples)
gdoubleprime_samples = grho_samples * np.sin(phi_samples)
```

## Diagnostic Plots

### Trace Plots

```python
from rheoQCM.core.bayesian import DiagnosticPlot
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 2, figsize=(12, 8))

DiagnosticPlot.trace(result, 'drho', ax=axes[0, 0])
DiagnosticPlot.histogram(result, 'drho', ax=axes[0, 1])

DiagnosticPlot.trace(result, 'grho_refh', ax=axes[1, 0])
DiagnosticPlot.histogram(result, 'grho_refh', ax=axes[1, 1])

DiagnosticPlot.trace(result, 'phi', ax=axes[2, 0])
DiagnosticPlot.histogram(result, 'phi', ax=axes[2, 1])

plt.tight_layout()
plt.savefig('mcmc_diagnostics.png', dpi=150)
```

### Corner Plot (Correlations)

```python
import arviz as az

# Convert to ArviZ InferenceData
idata = result.to_arviz()

# Create corner plot
az.plot_pair(
    idata,
    var_names=['drho', 'grho_refh', 'phi'],
    kind='kde',
    marginals=True,
)
plt.savefig('corner_plot.png', dpi=150)
```

### Posterior Predictive Check

```python
# Compare measured vs predicted
from rheoQCM.core.bayesian import plot_comparison

fig = plot_comparison(result, model)
plt.savefig('posterior_predictive.png', dpi=150)
```

## Convergence Diagnostics

### R-hat (Gelman-Rubin)

```python
# R-hat should be < 1.01 for convergence
rhat = result.rhat()
print(f"R-hat values:")
for param, value in rhat.items():
    status = "OK" if value < 1.01 else "WARNING"
    print(f"  {param}: {value:.4f} [{status}]")
```

### Effective Sample Size (ESS)

```python
# ESS should be > 400 for reliable estimates
ess = result.ess()
print(f"Effective sample sizes:")
for param, value in ess.items():
    status = "OK" if value > 400 else "WARNING"
    print(f"  {param}: {value:.0f} [{status}]")
```

### Divergences

```python
# Check for divergent transitions
n_divergent = result.num_divergences()
if n_divergent > 0:
    print(f"WARNING: {n_divergent} divergent transitions detected")
    print("Consider: increasing target_accept_prob or reparametrizing")
```

## Time Series with Uncertainty

### Processing with Uncertainty Bands

```python
from rheoQCM.core.bayesian import BayesianFitter
import numpy as np

# Process multiple time points
time_points = np.arange(0, 100, 1.0)  # 100 time points
results_bayesian = []

for t in time_points:
    # Load data for this time point
    model.load_delfstars(get_delfstars_at_time(t))

    # Quick MCMC (fewer samples for time series)
    result = fitter.fit(
        num_samples=500,
        num_warmup=200,
        num_chains=2,
    )

    results_bayesian.append({
        'time': t,
        'drho_mean': result.mean('drho'),
        'drho_ci': result.credible_interval('drho', 0.95),
    })

# Plot with uncertainty bands
import matplotlib.pyplot as plt

times = [r['time'] for r in results_bayesian]
means = [r['drho_mean'] for r in results_bayesian]
ci_low = [r['drho_ci'][0] for r in results_bayesian]
ci_high = [r['drho_ci'][1] for r in results_bayesian]

fig, ax = plt.subplots()
ax.plot(times, means, 'b-', label='Mean')
ax.fill_between(times, ci_low, ci_high, alpha=0.3, label='95% CI')
ax.set_xlabel('Time [s]')
ax.set_ylabel('drho [kg/m²]')
ax.legend()
plt.savefig('timeseries_with_uncertainty.png')
```

## Jacobian-Based Uncertainty (Fast Alternative)

For quick uncertainty estimates without full MCMC:

```python
from rheoQCM.core.uncertainty import UncertaintyCalculator

# Get point estimate first
result = model.solve_properties(nh=[3, 5, 3])

# Calculate uncertainties via Jacobian propagation
calc = UncertaintyCalculator(model)
uncertainties = calc.propagate(
    result,
    measurement_errors={
        'delf': 0.5,   # Hz uncertainty in Δf
        'delg': 0.2,   # Hz uncertainty in ΔΓ
    }
)

print(f"drho: {result.drho:.3e} ± {uncertainties['drho']:.3e}")
print(f"grho: {result.grho_refh:.3e} ± {uncertainties['grho_refh']:.3e}")
print(f"phi:  {result.phi:.4f} ± {uncertainties['phi']:.4f}")
```

## Comparing Models

### Bayesian Model Comparison

```python
# Fit with different harmonic combinations
result_353 = fitter.fit(nh=[3, 5, 3], ...)
result_355 = fitter.fit(nh=[3, 5, 5], ...)
result_575 = fitter.fit(nh=[5, 7, 5], ...)

# Compare via WAIC (Widely Applicable Information Criterion)
import arviz as az

waic_353 = az.waic(result_353.to_arviz())
waic_355 = az.waic(result_355.to_arviz())
waic_575 = az.waic(result_575.to_arviz())

print("Model comparison (lower WAIC is better):")
print(f"  nh=[3,5,3]: WAIC = {waic_353.waic:.1f}")
print(f"  nh=[3,5,5]: WAIC = {waic_355.waic:.1f}")
print(f"  nh=[5,7,5]: WAIC = {waic_575.waic:.1f}")
```

## Tips for Reliable Results

1. **Always check convergence** - R-hat < 1.01, ESS > 400
2. **Use warm-start** - `init_strategy='nlsq'` improves convergence
3. **Increase warmup** if you see divergences
4. **Run multiple chains** (4 is typical) to verify convergence
5. **Use informative priors** when you have prior knowledge

## Next Steps

- {doc}`../theory/numerical-methods` - Algorithm details
- {doc}`../api/index` - Full API reference
