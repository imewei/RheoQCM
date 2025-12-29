# Numerical Methods

This chapter explains how RheoQCM solves for material properties from QCM-D
measurements, including the optimization algorithms and uncertainty
quantification methods.

## The Inverse Problem

QCM-D measurements provide complex frequency shifts $\Delta f_n^*$ at multiple
harmonics. The **inverse problem** is to find material properties (drho,
grho, phi) that predict these measurements.

### Forward Model

The forward model $\mathcal{F}$ predicts frequency shifts from properties:

$$
\Delta f_n^* = \mathcal{F}(d\rho, |G^*|\rho, \phi; n, f_1, Z_q)
$$

### Optimization Objective

RheoQCM minimizes the residual between measured and predicted values:

$$
\min_{\theta} \sum_i \left| \Delta f_{n_i}^{*,\text{meas}} - \mathcal{F}(\theta; n_i) \right|^2
$$

where $\theta = (d\rho, |G^*|\rho, \phi)$.

## Solution Strategy

RheoQCM uses a **two-stage** solution strategy to ensure robust convergence:

### Stage 1: Initial Guess (Thin Film Approximation)

For an initial estimate, RheoQCM uses linearized thin-film equations that
have closed-form solutions:

```python
# Approximate formulas for initial guess
drho_init = -Zq / (2 * f1**2) * delf_mean / n_mean
phi_init = 2 * jnp.arctan(-delf / delgamma)
grho_init = (pi * Zq * abs(delfstar) / f1)**2
```

### Stage 2: Nonlinear Refinement

The initial guess is refined using the **Levenberg-Marquardt** algorithm
(via Optimistix) to solve the full nonlinear equations.

## Implementation Details

### JAX Acceleration

RheoQCM uses JAX for:

1. **JIT Compilation**: All physics functions are compiled for performance
2. **Automatic Differentiation**: Jacobians computed exactly via `jax.jacfwd`
3. **GPU Acceleration**: Batch processing can use GPU when available

### Residual Function

The core residual function compares measured and predicted values:

```python
def residual(params, delfstar_exp, harmonics, f1, refh, Zq):
    """
    Calculate residuals for nonlinear least squares.

    Parameters
    ----------
    params : array [grho_refh, phi, drho]
        Material properties to optimize
    delfstar_exp : dict
        Measured complex frequency shifts
    harmonics : list
        Harmonic numbers [n1, n2, n3]

    Returns
    -------
    residuals : array
        [delf_n1_err, delf_n2_err, delgamma_n3_err]
    """
    grho_refh, phi, drho = params

    # Predict frequency shifts using SLA model
    delfstar_pred = {}
    for n in set(harmonics):
        delfstar_pred[n] = calc_delfstar_sla(n, drho, grho_refh, phi, ...)

    # Compare to measurements
    residuals = [
        delfstar_pred[n1].real - delfstar_exp[n1].real,  # Δf at n1
        delfstar_pred[n2].real - delfstar_exp[n2].real,  # Δf at n2
        delfstar_pred[n3].imag - delfstar_exp[n3].imag,  # ΔΓ at n3
    ]
    return jnp.array(residuals)
```

### Solver Configuration

RheoQCM uses Optimistix for nonlinear least squares:

```python
import optimistix as optx

solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
result = optx.least_squares(
    residual_fn,
    solver,
    initial_guess,
    args=(delfstar_exp, harmonics, f1, refh, Zq),
)
```

## Uncertainty Quantification

RheoQCM provides two methods for uncertainty quantification:

### Method 1: Jacobian-Based (Fast)

Using the Jacobian at the solution point, uncertainties are propagated
linearly:

$$
\sigma_\theta = \sqrt{\text{diag}(J^{-1} \Sigma_{\text{meas}} (J^{-1})^T)}
$$

where:
- $J$ is the Jacobian of residuals w.r.t. parameters
- $\Sigma_{\text{meas}}$ is the measurement covariance

```python
from rheoQCM.core.uncertainty import UncertaintyCalculator

calc = UncertaintyCalculator(model)
uncertainties = calc.propagate_uncertainty(
    result,
    measurement_errors={'delf': 0.5, 'delg': 0.1}
)
```

### Method 2: Bayesian MCMC (Comprehensive)

For full posterior distributions, RheoQCM uses NumPyro MCMC:

```python
from rheoQCM.core.bayesian import BayesianFitter

fitter = BayesianFitter(model)
posterior = fitter.fit(
    delfstars,
    num_samples=2000,
    num_warmup=500,
)

# Get credible intervals
drho_ci = posterior.credible_interval('drho', 0.95)
```

The Bayesian approach:
- Provides full posterior distributions
- Handles non-Gaussian uncertainties
- Reveals parameter correlations

## Error Analysis

### Jacobian Matrix

The Jacobian relates property changes to measurement changes:

$$
J = \begin{bmatrix}
\frac{\partial \Delta f_{n1}}{\partial(d\rho)} &
\frac{\partial \Delta f_{n1}}{\partial(|G|\rho)} &
\frac{\partial \Delta f_{n1}}{\partial \phi} \\
\frac{\partial \Delta f_{n2}}{\partial(d\rho)} &
\frac{\partial \Delta f_{n2}}{\partial(|G|\rho)} &
\frac{\partial \Delta f_{n2}}{\partial \phi} \\
\frac{\partial \Delta\Gamma_{n3}}{\partial(d\rho)} &
\frac{\partial \Delta\Gamma_{n3}}{\partial(|G|\rho)} &
\frac{\partial \Delta\Gamma_{n3}}{\partial \phi}
\end{bmatrix}
$$

RheoQCM computes this using JAX autodiff for exact gradients.

### Propagating Measurement Errors

Given measurement uncertainties $(\sigma_{\Delta f_{n1}}, \sigma_{\Delta f_{n2}}, \sigma_{\Delta\Gamma_{n3}})$:

$$
\sigma_{d\rho} = \sqrt{\sum_i \left(J^{-1}_{1i} \sigma_i\right)^2}
$$

$$
\sigma_{|G|\rho} = \sqrt{\sum_i \left(J^{-1}_{2i} \sigma_i\right)^2}
$$

$$
\sigma_\phi = \sqrt{\sum_i \left(J^{-1}_{3i} \sigma_i\right)^2}
$$

## Convergence and Robustness

### Typical Convergence

For well-conditioned problems, RheoQCM typically converges in:
- 5-15 Levenberg-Marquardt iterations
- <50 ms per solution (CPU)
- <5 ms per solution (GPU batch)

### Handling Ill-Conditioning

Problems can become ill-conditioned when:
- Film is nearly Sauerbrey ($\phi \to 0$)
- Film is in bulk limit ($d \gg \delta$)
- Harmonics are closely spaced

RheoQCM uses:
- Parameter scaling/normalization
- Regularization when needed
- Multiple starting points for difficult cases

## Batch Processing

For time-series data, RheoQCM uses JAX's `vmap` for efficient batch
processing:

```python
from rheoQCM.core import batch_analyze_vmap

# Process 10,000 time points
results = batch_analyze_vmap(
    delfstars,           # shape (10000, 3) for 3 harmonics
    harmonics=[3, 5, 7],
    f1=5e6,
    refh=3,
)

# Results are arrays of shape (10000,)
print(f"drho range: {results.drho.min():.2e} to {results.drho.max():.2e}")
```

On GPU, this achieves >100,000 solutions/second.

## Algorithm Selection Guide

| Scenario | Recommended Method |
|----------|-------------------|
| Single measurement | `QCMModel.solve_properties()` |
| Time series (<1000 pts) | `batch_analyze()` |
| Large datasets (>1000 pts) | `batch_analyze_vmap()` |
| Uncertainty needed | `UncertaintyCalculator` |
| Full posteriors | `BayesianFitter` |

## References

1. Johannsmann, D. (2008). "Viscoelastic, mechanical, and dielectric
   measurements on complex samples with the quartz crystal microbalance."
   *PCCP*, 10, 4516-4534.

2. Levenberg, K. (1944). "A method for the solution of certain non-linear
   problems in least squares." *Quarterly of Applied Mathematics*, 2(2),
   164-168.

## Next Steps

- {doc}`../tutorials/scripting-basics` - Apply these methods in Python
- {doc}`../tutorials/bayesian-fitting` - Full Bayesian analysis tutorial
- {doc}`../api/index` - API reference documentation
