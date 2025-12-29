# Quick Start Guide

Get started with RheoQCM in 5 minutes. This tutorial shows you how to analyze
QCM-D data using both the GUI and Python scripts.

## Your First Analysis (GUI)

### Step 1: Launch RheoQCM

```bash
python -m rheoQCM.main
```

The main window appears with tabs for data display and settings.

### Step 2: Load Sample Data

1. Click **File → Open** (or press `Ctrl+O`)
2. Navigate to `src/data/` in the RheoQCM installation
3. Select a `.h5` data file

The data loads and displays in the plot area.

### Step 3: Configure Analysis

1. Go to the **Settings** tab
2. Under **Crystal**, verify:
   - Base Frequency: 5 MHz (default)
   - Reference Harmonic: 3 (default)
3. Under **Calculation**, set:
   - Harmonics: 3, 5, 7
   - nhcalc: 3,5,3

### Step 4: Run Analysis

1. Click **Analysis → Calculate Properties**
2. Watch the progress in the status bar
3. Results appear in the **Properties** tab

### Step 5: Export Results

1. Click **File → Export**
2. Choose Excel format (`.xlsx`)
3. Select output location

## Your First Analysis (Python)

### Basic Script

```python
"""
Quick start example: Analyze a single QCM measurement
"""
from rheoQCM.core import QCMModel, configure_jax

# Initialize JAX for optimal precision
configure_jax()

# Create a model with 5 MHz crystal, reference harmonic n=3
model = QCMModel(f1=5e6, refh=3)

# Load your experimental data
# Format: {harmonic: Δf + iΔΓ}
model.load_delfstars({
    3: -1000 + 100j,   # n=3: Δf=-1000 Hz, ΔΓ=100 Hz
    5: -1700 + 180j,   # n=5: Δf=-1700 Hz, ΔΓ=180 Hz
    7: -2500 + 280j,   # n=7: Δf=-2500 Hz, ΔΓ=280 Hz
})

# Solve for material properties
# nh=[n1, n2, n3]: use Δf at n1 and n2, ΔΓ at n3
result = model.solve_properties(nh=[3, 5, 3])

# Display results
print("=" * 40)
print("QCM-D Analysis Results")
print("=" * 40)
print(f"Areal mass (drho):    {result.drho * 1e6:.2f} μg/cm²")
print(f"|G*|ρ at n=3:         {result.grho_refh:.2e} Pa·kg/m³")
print(f"Phase angle (phi):    {result.phi * 180/3.14159:.1f}°")
print("=" * 40)
```

### Expected Output

```
========================================
QCM-D Analysis Results
========================================
Areal mass (drho):    17.66 μg/cm²
|G*|ρ at n=3:         1.23e+08 Pa·kg/m³
Phase angle (phi):    11.3°
========================================
```

## Understanding the Results

### Areal Mass (drho)

The product of film thickness and density:

$$d\rho = \rho \cdot d$$

- **Units**: kg/m² (displayed as μg/cm²)
- **Physical meaning**: Mass per unit area of the film
- **Typical values**: 0.1-100 μg/cm²

### Complex Modulus (grho_refh)

The product of shear modulus magnitude and density at the reference harmonic:

$$|G^*|\rho = |G_3^*| \cdot \rho$$

- **Units**: Pa·kg/m³
- **Physical meaning**: Stiffness × density of the material
- **Typical values**: 10⁶-10¹⁰ Pa·kg/m³

### Phase Angle (phi)

The viscoelastic phase angle:

$$\phi = \arctan(G''/G')$$

- **Units**: radians (often displayed as degrees)
- **Physical meaning**: Ratio of viscous to elastic response
- **Typical values**: 0-1.57 rad (0-90°)

| phi (degrees) | Material Character |
|---------------|-------------------|
| 0-10° | Nearly elastic (glassy polymer, metal) |
| 10-45° | Viscoelastic solid (rubbery polymer) |
| 45-80° | Viscoelastic liquid (soft gel, polymer melt) |
| 80-90° | Nearly viscous (dilute solution, water) |

## Loading Data from Files

### From HDF5 File

```python
from rheoQCM.modules.DataSaver import DataSaver

# Load data file
ds = DataSaver()
ds.load_file("path/to/data.h5")

# Access frequency shift data
samp_data = ds.samp  # pandas DataFrame
print(samp_data.columns)
# ['queue_id', 't', 'delf1', 'delg1', 'delf3', 'delg3', ...]
```

### From Excel File (QCM-D Format)

```python
import pandas as pd
from rheoQCM.core import QCMModel

# Read QCM-D exported data
df = pd.read_excel("qcmd_data.xlsx")

# Extract a single time point
row = df.iloc[100]  # row 100
delfstars = {
    3: row['delf3'] + 1j * row['delg3'],
    5: row['delf5'] + 1j * row['delg5'],
    7: row['delf7'] + 1j * row['delg7'],
}

# Analyze
model = QCMModel(f1=5e6, refh=3)
model.load_delfstars(delfstars)
result = model.solve_properties(nh=[3, 5, 3])
```

## Batch Processing

For time-series data with many measurements:

```python
import jax.numpy as jnp
from rheoQCM.core import batch_analyze_vmap

# Prepare data as 2D array: (n_timepoints, n_harmonics)
# Each row contains [delf3+iΔΓ3, delf5+iΔΓ5, delf7+iΔΓ7]
delfstars = jnp.array([
    [-1000+100j, -1700+180j, -2500+280j],
    [-1010+102j, -1720+185j, -2530+290j],
    [-1020+105j, -1740+190j, -2560+300j],
    # ... more time points
])

# Process all at once (GPU-accelerated if available)
results = batch_analyze_vmap(
    delfstars,
    harmonics=[3, 5, 7],
    f1=5e6,
    refh=3,
)

# Results are arrays
print(f"drho: {results.drho}")      # shape: (n_timepoints,)
print(f"grho: {results.grho_refh}") # shape: (n_timepoints,)
print(f"phi:  {results.phi}")       # shape: (n_timepoints,)
```

## Common Pitfalls

### Wrong Sign Convention

QCM-D instruments use different sign conventions. RheoQCM expects:
- **Δf**: Negative for mass loading
- **ΔΓ**: Positive for energy dissipation

If your instrument uses opposite signs, negate accordingly.

### Incorrect Harmonic Order

The `nh` parameter order matters:
- `nh=[3, 5, 3]`: Uses Δf at harmonics 3 and 5, ΔΓ at harmonic 3
- `nh=[3, 5, 5]`: Uses Δf at harmonics 3 and 5, ΔΓ at harmonic 5

Choose harmonics with good signal quality.

### Reference vs. Sample Baseline

Always subtract the appropriate baseline (bare crystal or initial state)
before analysis. RheoQCM expects **shifts** not absolute values.

## Next Steps

- {doc}`gui-workflow` - Complete GUI walkthrough
- {doc}`scripting-basics` - Advanced scripting techniques
- {doc}`../theory/index` - Understand the physics
