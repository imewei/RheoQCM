# Migration Guide: QCMFuncs to rheoQCM.core

This guide helps you migrate from the deprecated `QCMFuncs` module to the modern `rheoQCM.core` API.

## Key Differences

| Feature | QCMFuncs (Legacy) | rheoQCM.core (Modern) |
|---------|-------------------|----------------------|
| Angles | Degrees | **Radians** |
| Backend | NumPy/SciPy | JAX (GPU-accelerated) |
| Batch processing | Manual loops | `vmap` vectorization |
| Properties dict | `grho3`, `phi` | `grho_refh`, `phi`, `refh` |

## Quick Reference

### Import Changes

```python
# OLD (deprecated)
from QCMFuncs.QCM_functions import sauerbreyf, sauerbreym, grho, bulk_props

# NEW (recommended)
from rheoQCM.core import sauerbreyf, sauerbreym
from rheoQCM.core.physics import grho, bulk_props
from rheoQCM.core.model import QCMModel
from rheoQCM.core.analysis import batch_analyze_vmap
```

### Sauerbrey Functions

```python
# OLD
from QCMFuncs.QCM_functions import sauerbreyf, sauerbreym

delf = sauerbreyf(n=3, drho=1e-6, f1=5e6)
drho = sauerbreym(n=3, delf=-1000, f1=5e6)

# NEW - same signature, just different import
from rheoQCM.core import sauerbreyf, sauerbreym

delf = sauerbreyf(n=3, drho=1e-6, f1=5e6)
drho = sauerbreym(n=3, delf=-1000, f1=5e6)
```

### grho Function

```python
# OLD - uses props dict with 'grho3' and 'phi' in DEGREES
from QCMFuncs.QCM_functions import grho

props = {'grho3': 1e10, 'phi': 30}  # phi in degrees
grho_n5 = grho(n=5, props=props)

# NEW - uses explicit parameters with phi in RADIANS
import numpy as np
from rheoQCM.core.physics import grho

grho_n5 = grho(n=5, grho_refh=1e10, phi=np.deg2rad(30), refh=3)
```

### bulk_props Function

```python
# OLD - returns phi in DEGREES
from QCMFuncs.QCM_functions import bulk_props

grho_val, phi_deg = bulk_props(delfstar=-1000+100j, f1=5e6)

# NEW - returns phi in RADIANS
import numpy as np
from rheoQCM.core.physics import bulk_props

grho_val, phi_rad = bulk_props(delfstar=-1000+100j, f1=5e6)
phi_deg = np.rad2deg(phi_rad)  # if you need degrees
```

### calc_delfstar (Multilayer)

```python
# OLD - layers with 'grho3' and 'phi' in DEGREES
from QCMFuncs.QCM_functions import calc_delfstar

layers = {
    1: {'grho3': 1e10, 'phi': 30, 'drho': 1e-6}
}
delfstar = calc_delfstar(n=3, layers_in=layers, calctype='SLA')

# NEW - layers with 'grho' and 'phi' in RADIANS
import numpy as np
from rheoQCM.core.multilayer import calc_delfstar_multilayer

layers = {
    1: {'grho': 1e10, 'phi': np.deg2rad(30), 'drho': 1e-6}
}
delfstar = calc_delfstar_multilayer(n=3, layers=layers, calctype='SLA', f1=5e6)
```

### Solving for Film Properties

```python
# OLD - using QCMAnalyzer class
from QCMFuncs.QCM_functions import QCMAnalyzer

analyzer = QCMAnalyzer(delfstar={3: -1000+100j, 5: -1700+180j, 7: -2500+280j})
result = analyzer.solve(calc='357_3')
grho3 = result['grho3']
phi_deg = result['phi']

# NEW - using QCMModel class
from rheoQCM.core.model import QCMModel
import numpy as np

model = QCMModel(f1=5e6, refh=3)
model.load_delfstars({3: -1000+100j, 5: -1700+180j, 7: -2500+280j})
result = model.solve_properties(nh=[3, 5, 7])

grho_refh = result.grho_refh
phi_rad = result.phi
drho = result.drho
phi_deg = np.rad2deg(phi_rad)  # if you need degrees
```

### Batch Processing

```python
# OLD - manual loop
from QCMFuncs.QCM_functions import QCMAnalyzer

results = []
for delfstar_dict in all_measurements:
    analyzer = QCMAnalyzer(delfstar=delfstar_dict)
    results.append(analyzer.solve(calc='357_3'))

# NEW - vectorized with vmap (much faster)
import jax.numpy as jnp
from rheoQCM.core.analysis import batch_analyze_vmap

# Input shape: (N_samples, N_harmonics)
delfstars = jnp.array([
    [-1000+100j, -1700+180j, -2500+280j],
    [-1100+110j, -1800+190j, -2600+290j],
    # ... more samples
])

result = batch_analyze_vmap(delfstars, harmonics=[3, 5, 7], nhcalc='357')
# result.drho, result.grho_refh, result.phi are arrays of shape (N_samples,)
```

## Suppressing Deprecation Warnings

If you need to continue using QCMFuncs temporarily, you can suppress the warnings:

```bash
# Environment variable
export QCMFUNCS_SUPPRESS_DEPRECATION=1
```

```python
# Or in code
import os
os.environ['QCMFUNCS_SUPPRESS_DEPRECATION'] = '1'

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='QCMFuncs')
warnings.filterwarnings('ignore', category=FutureWarning, module='QCMFuncs')
```

## Function Mapping Reference

| QCMFuncs Function | rheoQCM.core Replacement | Notes |
|-------------------|-------------------------|-------|
| `sauerbreyf(n, drho)` | `rheoQCM.core.sauerbreyf(n, drho, f1)` | Same API |
| `sauerbreym(n, delf)` | `rheoQCM.core.sauerbreym(n, delf, f1)` | Same API |
| `grho(n, props)` | `rheoQCM.core.physics.grho(n, grho_refh, phi, refh)` | phi in radians |
| `bulk_props(delfstar)` | `rheoQCM.core.physics.bulk_props(delfstar, f1)` | Returns phi in radians |
| `calc_delfstar(n, layers)` | `rheoQCM.core.multilayer.calc_delfstar_multilayer()` | phi in radians |
| `etarho(n, props)` | `rheoQCM.core.physics.etarho(n, grho, phi, f1, refh)` | phi in radians |
| `grho_from_dlam(n, drho, dlam, phi)` | `rheoQCM.core.physics.grho_from_dlam()` | phi in radians |
| `calc_dlam(n, film)` | `rheoQCM.core.physics.calc_dlam()` | phi in radians |
| `calc_lamrho(n, grho3, phi)` | `rheoQCM.core.physics.calc_lamrho()` | phi in radians |
| `calc_deltarho(n, grho3, phi)` | `rheoQCM.core.physics.calc_deltarho()` | phi in radians |
| `normdelfstar(n, dlam3, phi)` | `rheoQCM.core.physics.normdelfstar()` | phi in radians |
| `zstar_bulk(n, props, calctype)` | `rheoQCM.core.physics.zstar_bulk()` | phi in radians |
| `calc_D(n, props, delfstar)` | `rheoQCM.core.physics.calc_D()` | phi in radians |

## Getting Help

- Check the `rheoQCM.core` module docstrings for detailed API documentation
- See `specs/004-unify-qcm-modules/quickstart.md` for more examples
- File issues at: https://github.com/shullgroup/rheoQCM/issues
