# Scripting Basics

This tutorial teaches you how to use RheoQCM's Python API for automated
analysis, custom workflows, and integration with other tools.

## Architecture Overview

RheoQCM uses a three-layer architecture:

```
┌─────────────────────────────────────────────┐
│           Layer 3: Analysis API             │
│       QCMAnalyzer, batch_analyze()          │
├─────────────────────────────────────────────┤
│           Layer 2: Model Logic              │
│           QCMModel, SolveResult             │
├─────────────────────────────────────────────┤
│           Layer 1: Physics Functions        │
│   calc_delfstar_sla(), grho(), calc_ZL()    │
└─────────────────────────────────────────────┘
```

Choose the appropriate layer based on your needs:

| Layer | Use When |
|-------|----------|
| Layer 3 | Processing entire datasets, workflows |
| Layer 2 | Custom analysis, single measurements |
| Layer 1 | Direct physics calculations, research |

## Basic Usage

### The QCMModel Class

```python
from rheoQCM.core import QCMModel, configure_jax

# Always configure JAX first for float64 precision
configure_jax()

# Create model instance
model = QCMModel(
    f1=5e6,      # Fundamental frequency [Hz]
    refh=3,      # Reference harmonic
)

# Load experimental data
model.load_delfstars({
    3: -1000 + 100j,  # Complex frequency shift at n=3
    5: -1700 + 180j,
    7: -2500 + 280j,
})

# Solve for properties
result = model.solve_properties(
    nh=[3, 5, 3],  # Harmonics for [Δf1, Δf2, ΔΓ]
)

# Access results
print(f"drho: {result.drho}")
print(f"grho: {result.grho_refh}")
print(f"phi:  {result.phi}")
```

### The SolveResult Object

The `solve_properties()` method returns a `SolveResult` dataclass:

```python
@dataclass
class SolveResult:
    drho: float          # Areal mass [kg/m²]
    grho_refh: float     # |G*|ρ at reference harmonic [Pa·kg/m³]
    phi: float           # Phase angle [rad]
    dlam: float          # d/λ ratio
    converged: bool      # Solver convergence flag
    residuals: ndarray   # Fit residuals
```

## Loading Data from Files

### HDF5 Files

```python
from rheoQCM.modules.DataSaver import DataSaver
import pandas as pd

# Load RheoQCM native format
ds = DataSaver()
ds.load_file("experiment.h5")

# Access data as DataFrame
df = ds.samp  # Sample channel
# df = ds.ref  # Reference channel

# Extract time series
time = df['t'].values
delf3 = df['delf3'].values
delg3 = df['delg3'].values
```

### Excel Files

```python
import pandas as pd

df = pd.read_excel("qcmd_export.xlsx")

# Build delfstars dictionary for each row
for idx, row in df.iterrows():
    delfstars = {}
    for n in [3, 5, 7]:
        delf = row[f'delf{n}']
        delg = row[f'delg{n}']
        delfstars[n] = delf + 1j * delg

    # Analyze this time point
    model.load_delfstars(delfstars)
    result = model.solve_properties(nh=[3, 5, 3])
```

### NumPy Arrays

```python
import numpy as np

# Load from NPZ
data = np.load("data.npz")
delf = data['delf']  # shape: (n_times, n_harmonics)
delg = data['delg']

# Convert to complex
delfstars = delf + 1j * delg
```

## Processing Time Series

### Using batch_analyze()

```python
from rheoQCM.core.analysis import batch_analyze

# Prepare data as list of dicts
measurements = []
for i in range(len(time)):
    measurements.append({
        3: delf3[i] + 1j * delg3[i],
        5: delf5[i] + 1j * delg5[i],
        7: delf7[i] + 1j * delg7[i],
    })

# Process all at once
results = batch_analyze(
    measurements,
    harmonics=[3, 5, 7],
    nhcalc='353',
    f1=5e6,
    refh=3,
)

# Results is a BatchResult with arrays
drho_series = results.drho
grho_series = results.grho_refh
phi_series = results.phi
```

### Using batch_analyze_vmap() (GPU-Accelerated)

```python
import jax.numpy as jnp
from rheoQCM.core import batch_analyze_vmap

# Prepare as JAX array: shape (n_times, n_harmonics)
# Order must match harmonics list
delfstars = jnp.array([
    [delf3 + 1j*delg3, delf5 + 1j*delg5, delf7 + 1j*delg7]
    for delf3, delg3, delf5, delg5, delf7, delg7
    in zip(delf3_arr, delg3_arr, delf5_arr, delg5_arr, delf7_arr, delg7_arr)
])

# Process with GPU acceleration
results = batch_analyze_vmap(
    delfstars,
    harmonics=[3, 5, 7],
    f1=5e6,
    refh=3,
)
```

## Direct Physics Calculations

### Sauerbrey Mass

```python
from rheoQCM.core.physics import sauerbreyf, sauerbreym

# Calculate Sauerbrey frequency shift
drho = 1e-5  # kg/m² (10 μg/cm²)
delf = sauerbreyf(n=3, drho=drho)
print(f"Δf = {delf:.0f} Hz")

# Inverse: mass from frequency
delf_measured = -1700  # Hz
drho = sauerbreym(n=3, delf=delf_measured)
print(f"drho = {drho*1e6:.2f} μg/cm²")
```

### Complex Modulus Calculations

```python
from rheoQCM.core.physics import grho, grhostar

# Calculate grho at a specific harmonic
grho_n5 = grho(
    n=5,
    grho_refh=1e8,
    phi=0.3,
    refh=3,
)
print(f"|G*|ρ at n=5: {grho_n5:.2e}")

# Get complex modulus
gstar = grhostar(grho_n5, phi=0.3)
print(f"G* = {gstar:.2e}")
print(f"G' = {gstar.real:.2e}, G'' = {gstar.imag:.2e}")
```

### SLA Frequency Shift

```python
from rheoQCM.core.physics import calc_delfstar_sla

# Predict frequency shift from properties
delfstar = calc_delfstar_sla(
    n=3,
    drho=1e-5,
    grho_refh=1e8,
    phi=0.3,
    f1=5e6,
    refh=3,
)
print(f"Predicted: Δf = {delfstar.real:.0f}, ΔΓ = {delfstar.imag:.0f}")
```

### Multilayer Calculations

```python
from rheoQCM.core.multilayer import calc_delfstar_multilayer

layers = [
    {'drho': 1e-5, 'grho': 1e8, 'phi': 0.3},  # Film
    {'drho': float('inf'), 'grho': 1e6, 'phi': 1.57},  # Water
]

delfstar = calc_delfstar_multilayer(n=3, layers=layers, f1=5e6)
```

## Custom Analysis Workflows

### Temperature Ramp Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from rheoQCM.core import QCMModel, configure_jax

configure_jax()

# Load temperature ramp data
data = np.load("temp_ramp.npz")
temp = data['temperature']
delf3 = data['delf3']
delg3 = data['delg3']
delf5 = data['delf5']
delg5 = data['delg5']

# Analyze at each temperature
model = QCMModel(f1=5e6, refh=3)
results = {'temp': [], 'drho': [], 'grho': [], 'phi': []}

for i, T in enumerate(temp):
    model.load_delfstars({
        3: delf3[i] + 1j * delg3[i],
        5: delf5[i] + 1j * delg5[i],
    })
    result = model.solve_properties(nh=[3, 5, 3])

    results['temp'].append(T)
    results['drho'].append(result.drho)
    results['grho'].append(result.grho_refh)
    results['phi'].append(result.phi)

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(results['temp'], np.array(results['drho']) * 1e6)
axes[0].set_ylabel('drho [μg/cm²]')

axes[1].semilogy(results['temp'], results['grho'])
axes[1].set_ylabel('|G*|ρ [Pa·kg/m³]')

axes[2].plot(results['temp'], np.rad2deg(results['phi']))
axes[2].set_ylabel('φ [°]')
axes[2].set_xlabel('Temperature [°C]')

plt.tight_layout()
plt.savefig('temp_ramp_analysis.png', dpi=150)
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
from rheoQCM.core import QCMModel, configure_jax

def analyze_file(filepath):
    """Process a single data file."""
    configure_jax()

    # Load and process
    model = QCMModel(f1=5e6, refh=3)
    # ... load data from filepath ...
    result = model.solve_properties(nh=[3, 5, 3])

    return {
        'file': filepath,
        'drho': result.drho,
        'grho': result.grho_refh,
        'phi': result.phi,
    }

# Process multiple files in parallel
files = ['exp1.h5', 'exp2.h5', 'exp3.h5']

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(analyze_file, files))
```

## Integration with Other Tools

### Pandas Integration

```python
import pandas as pd
from rheoQCM.core import batch_analyze

# Process and create DataFrame
results = batch_analyze(measurements, ...)

df_results = pd.DataFrame({
    'time': time,
    'drho_ugcm2': results.drho * 1e6,
    'grho_Pa_kgm3': results.grho_refh,
    'phi_deg': np.rad2deg(results.phi),
})

# Save to various formats
df_results.to_csv('results.csv', index=False)
df_results.to_excel('results.xlsx', index=False)
df_results.to_parquet('results.parquet')
```

### Matplotlib Visualization

```python
import matplotlib.pyplot as plt
from rheoQCM.core.physics import Zq, f1_default

# Create publication-quality figure
fig, ax = plt.subplots(figsize=(8, 6))

ax.loglog(results['grho'], np.rad2deg(results['phi']), 'o-')
ax.set_xlabel(r'$|G^*|\rho$ [Pa·kg/m³]')
ax.set_ylabel(r'$\phi$ [°]')
ax.set_title('Rheological Master Curve')
ax.grid(True, which='both', alpha=0.3)

plt.savefig('master_curve.pdf')
```

## Error Handling

```python
from rheoQCM.core import QCMModel
from rheoQCM.core.multilayer import LayerValidationError

model = QCMModel(f1=5e6, refh=3)

try:
    model.load_delfstars({3: -1000+100j, 5: -1700+180j})
    result = model.solve_properties(nh=[3, 5, 3])

    if not result.converged:
        print("Warning: Solver did not converge")
        print(f"Residuals: {result.residuals}")

except LayerValidationError as e:
    print(f"Invalid layer configuration: {e}")

except ValueError as e:
    print(f"Invalid input: {e}")
```

## Next Steps

- {doc}`batch-analysis` - Large-scale data processing
- {doc}`bayesian-fitting` - Uncertainty quantification
- {doc}`../api/index` - Complete API reference
