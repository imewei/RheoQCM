# Analysis Settings Guide

This guide covers configuring analysis parameters for accurate QCM-D data interpretation.

## Key Analysis Parameters

### Harmonic Selection (`nhcalc`)

The `nhcalc` parameter specifies which harmonics to use for analysis:

| Format | Example | Meaning |
|--------|---------|---------|
| 3-digit string | `'353'` | Use Δf at n=3,5 and ΔΓ at n=3 |
| 3-digit string | `'575'` | Use Δf at n=5,7 and ΔΓ at n=5 |
| List | `[3, 5, 3]` | Same as `'353'` |

The three positions represent:
1. **First harmonic** for Δf (frequency shift)
2. **Second harmonic** for Δf (frequency shift)
3. **Harmonic** for ΔΓ (dissipation)

### Common Configurations

| `nhcalc` | Best For |
|----------|----------|
| `'353'` | General purpose, thin films |
| `'355'` | When n=5 ΔΓ is more reliable |
| `'575'` | Higher frequency sensitivity |
| `'357'` | Mixed approach |

### Choosing Harmonics

```python
from rheoQCM.core import QCMModel, configure_jax

configure_jax()

model = QCMModel(f1=5e6, refh=3)
model.load_delfstars({
    3: -1000 + 100j,
    5: -1700 + 180j,
    7: -2500 + 280j,
})

# Compare different harmonic selections
for nhcalc in ['353', '355', '575', '357']:
    result = model.solve_properties(nh=list(map(int, nhcalc)))
    print(f"{nhcalc}: drho={result.drho:.3e}, phi={result.phi:.3f}")
```

## Reference Harmonic (`refh`)

The reference harmonic is where viscoelastic properties are reported:

```python
# Properties reported at n=3
model_refh3 = QCMModel(f1=5e6, refh=3)

# Properties reported at n=5
model_refh5 = QCMModel(f1=5e6, refh=5)
```

**Important**: `grho_refh` depends on `refh` due to frequency-dependent viscoelasticity. Always report the reference harmonic with your results.

## Fundamental Frequency (`f1`)

The crystal's fundamental frequency:

| f1 | Common Use |
|----|------------|
| 5 MHz | Standard AT-cut crystals |
| 10 MHz | High-sensitivity applications |
| 6 MHz | Some commercial instruments |

```python
# 5 MHz crystal
model = QCMModel(f1=5e6, refh=3)

# 10 MHz crystal
model = QCMModel(f1=10e6, refh=3)
```

## Calculation Types

### Standard SLA (Small Load Approximation)

The default calculation assumes a single viscoelastic layer:

```python
result = model.solve_properties(
    nh=[3, 5, 3],
    calctype='SLA',  # Default
)
```

### Bulk Limit

For thick films or bulk liquids:

```python
result = model.solve_properties(
    nh=[3, 5, 3],
    calctype='bulk',
)
```

### Custom Calculation Types

Register custom physics models:

```python
from rheoQCM.core.model import register_calctype

def custom_residual(x, model, delfstars, harmonics):
    """Custom residual function for optimization."""
    drho, grho, phi = x
    # Calculate predicted frequency shifts
    # Return residuals
    return residuals

register_calctype('my_model', custom_residual)

result = model.solve_properties(
    nh=[3, 5, 3],
    calctype='my_model',
)
```

## GUI Settings

### Settings Panel

Access via **Settings → Analysis** in the GUI:

| Setting | Description | Location |
|---------|-------------|----------|
| Harmonics | Select which harmonics to use | Analysis tab |
| Reference | Reference harmonic for output | Analysis tab |
| Calc Type | SLA, bulk, or custom | Analysis tab |

### Crystal Settings

Access via **Settings → Hardware → Crystal**:

| Setting | Description | Default |
|---------|-------------|---------|
| Base Frequency | f1 in Hz | 5,000,000 |
| Max Harmonic | Highest harmonic | 13 |

## Solver Settings

### Convergence Tolerance

Control solver precision:

```python
# In Python (advanced)
from rheoQCM.core.model import QCMModel

model = QCMModel(f1=5e6, refh=3)
# Solver tolerances configured internally via Optimistix
```

### Initial Guess Strategy

The solver uses a two-stage approach:

1. **Thin film guess**: Uses Sauerbrey approximation
2. **Refinement**: Levenberg-Marquardt optimization

## Batch Analysis Settings

### For `batch_analyze()`

```python
from rheoQCM.core.analysis import batch_analyze

results = batch_analyze(
    measurements,
    harmonics=[3, 5, 7],    # Available harmonics in data
    nhcalc='353',           # Which to use for fitting
    f1=5e6,
    refh=3,
)
```

### For `batch_analyze_vmap()` (GPU)

```python
from rheoQCM.core import batch_analyze_vmap

results = batch_analyze_vmap(
    delfstars,              # JAX array (N, n_harmonics)
    harmonics=[3, 5, 7],    # Must match array column order
    f1=5e6,
    refh=3,
)
```

## Quality Control Settings

### Convergence Checking

```python
results = batch_analyze(measurements, ...)

# Check convergence rate
n_converged = sum(results.converged)
n_total = len(results.converged)
print(f"Converged: {n_converged}/{n_total} ({100*n_converged/n_total:.1f}%)")

# Filter unconverged
import numpy as np
valid = results.converged
drho_valid = np.where(valid, results.drho, np.nan)
```

### Physical Bounds Checking

```python
import numpy as np

# Check for physically reasonable results
def validate_results(results):
    issues = []

    # drho should be positive for adsorption
    if np.any(results.drho < 0):
        issues.append("Negative drho detected")

    # phi should be between 0 and π/2
    if np.any(results.phi < 0) or np.any(results.phi > np.pi/2):
        issues.append("phi outside valid range [0, π/2]")

    # grho should be positive
    if np.any(results.grho_refh <= 0):
        issues.append("Non-positive grho detected")

    return issues

issues = validate_results(results)
if issues:
    print("Validation issues:", issues)
```

## Environment Variables

Configure JAX behavior:

```bash
# Enable 64-bit precision (required for accurate results)
export JAX_ENABLE_X64=true

# Select GPU device
export CUDA_VISIBLE_DEVICES=0

# Disable GPU (use CPU only)
export JAX_PLATFORMS=cpu
```

Or in Python:

```python
from rheoQCM.core import configure_jax

# Must be called before any JAX operations
configure_jax()  # Enables float64
```

## Recommended Settings by Application

### Thin Polymer Films

```python
model = QCMModel(f1=5e6, refh=3)
result = model.solve_properties(nh=[3, 5, 3], calctype='SLA')
```

### Protein Adsorption

```python
model = QCMModel(f1=5e6, refh=3)
result = model.solve_properties(nh=[3, 5, 3], calctype='SLA')
# Consider: Lower harmonics may be more reliable for soft films
```

### Bulk Liquids

```python
model = QCMModel(f1=5e6, refh=3)
result = model.solve_properties(nh=[3, 5, 3], calctype='bulk')
```

### High-Frequency Materials

```python
model = QCMModel(f1=5e6, refh=5)  # Report at n=5
result = model.solve_properties(nh=[5, 7, 5])  # Use higher harmonics
```

## Troubleshooting Settings

### "Solver did not converge"

1. Check data quality (no NaN/Inf values)
2. Try different `nhcalc` combinations
3. Verify sign conventions match expectations
4. Consider if material is outside model assumptions

### "Unreasonable phi values"

1. Check dissipation data quality
2. Verify ΔΓ has correct sign (positive for energy loss)
3. Try different reference harmonic

### "Negative drho"

1. Verify Δf sign convention (negative for mass loading)
2. Check reference measurement was correct
3. Consider if desorption is occurring

## Next Steps

- {doc}`visualization` - Plot your results
- {doc}`exporting-results` - Save and export data
- {doc}`../tutorials/bayesian-fitting` - Uncertainty quantification
