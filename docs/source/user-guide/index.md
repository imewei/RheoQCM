# User Guide

This section provides practical guidance for using RheoQCM in your research workflow.

## Data Workflow

```{toctree}
:maxdepth: 2

data-acquisition
data-import
analysis-settings
visualization
exporting-results
```

## Overview

The typical RheoQCM workflow consists of these stages:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. DATA ACQUISITION                                            │
│     • Connect VNA hardware (Windows)                            │
│     • Configure crystal parameters                              │
│     • Record Δf and ΔΓ time series                              │
├─────────────────────────────────────────────────────────────────┤
│  2. DATA IMPORT (alternative to acquisition)                    │
│     • Import from QCM-D instruments (Excel)                     │
│     • Import from MATLAB or CSV                                 │
│     • Convert to RheoQCM format                                 │
├─────────────────────────────────────────────────────────────────┤
│  3. ANALYSIS CONFIGURATION                                      │
│     • Select harmonics (nhcalc)                                 │
│     • Set reference harmonic                                    │
│     • Choose calculation type                                   │
├─────────────────────────────────────────────────────────────────┤
│  4. VISUALIZATION                                               │
│     • Time series plots                                         │
│     • ΔΓ vs Δf plots                                            │
│     • Master curves                                             │
├─────────────────────────────────────────────────────────────────┤
│  5. EXPORT                                                      │
│     • Excel/CSV for sharing                                     │
│     • HDF5 for archival                                         │
│     • Publication-quality figures                               │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Reference

### GUI Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Open file | Ctrl+O |
| Save file | Ctrl+S |
| Export data | Ctrl+E |
| Start acquisition | F5 |
| Stop acquisition | F6 |
| Run analysis | F7 |
| Settings | Ctrl+, |

### Common Tasks

| Task | Section |
|------|---------|
| Connect to VNA hardware | {doc}`data-acquisition` |
| Import QCM-D Excel files | {doc}`data-import` |
| Choose which harmonics to use | {doc}`analysis-settings` |
| Create publication figures | {doc}`visualization` |
| Export to Excel/CSV | {doc}`exporting-results` |

### Python Quick Reference

```python
from rheoQCM.core import QCMModel, configure_jax

# Always configure JAX first
configure_jax()

# Create model
model = QCMModel(f1=5e6, refh=3)

# Load data
model.load_delfstars({
    3: -1000 + 100j,  # Δf + iΔΓ at n=3
    5: -1700 + 180j,  # Δf + iΔΓ at n=5
})

# Analyze
result = model.solve_properties(nh=[3, 5, 3])

# Access results
print(f"drho = {result.drho:.3e} kg/m²")
print(f"grho = {result.grho_refh:.3e} Pa·kg/m³")
print(f"phi = {result.phi:.3f} rad")
```

## Recommended Reading Order

1. **New users**: Start with {doc}`data-import` if you have existing data, or {doc}`data-acquisition` if you have VNA hardware
2. **Understanding parameters**: Read {doc}`analysis-settings`
3. **Creating figures**: See {doc}`visualization`
4. **Saving results**: Check {doc}`exporting-results`

## Related Sections

- {doc}`../tutorials/quickstart` - Step-by-step getting started
- {doc}`../tutorials/gui-workflow` - Complete GUI tutorial
- {doc}`../tutorials/scripting-basics` - Python API tutorial
- {doc}`../theory/index` - Theoretical background
