# Multilayer Film Modeling

This chapter explains how RheoQCM handles complex multilayer film structures
using the acoustic impedance transfer matrix method.

## Overview

Real QCM experiments often involve multiple layers:
- Polymer film + liquid
- Crosslinked gel + swelling medium
- Electrode coating + sample + air

RheoQCM models these using a recursive impedance calculation that properly
accounts for wave reflections at each interface.

## Two-Layer Geometry

The most common case is a film sandwiched between the electrode and a bulk
medium (liquid or air):

```
┌─────────────────────────┐
│       Bulk Medium       │  (semi-infinite)
│      (e.g., water)      │
├─────────────────────────┤
│         Film            │  thickness d₁
│   (viscoelastic layer)  │
├─────────────────────────┤
│     Quartz Crystal      │
│       (electrode)       │
└─────────────────────────┘
```

### Master Equation

The complex frequency shift for this geometry is given by the **bilayer
master equation**:

$$
\frac{\Delta f_n^*}{\Delta f_{sn}} = -\frac{\tan(D_{n1}^*)}{D_{n1}^*}
\left[\frac{1 - (r_{n12}^*)^2}{1 + ir_{n12}^* \tan(D_{n1}^*)}\right]
$$

where:
- $\Delta f_{sn}$ is the Sauerbrey frequency shift
- $D_{n1}^*$ is the complex phase factor for layer 1
- $r_{n12}^*$ is the acoustic impedance ratio between layers

### Impedance Ratio

The impedance ratio determines the acoustic contrast between the film and
the surrounding medium:

$$
r_{n12}^* = \frac{Z_{n2}^*}{Z_{n1}^*} = \sqrt{\frac{\rho_2 G_{n2}^*}{\rho_1 G_{n1}^*}}
$$

**Special cases:**
- $r^* = 0$: Film in air (no acoustic coupling to medium)
- $r^* = 1$: No contrast (trivially $\Delta f^* = 0$)
- $r^* \ll 1$: High contrast (reduces to single-layer equation)

## Calculating Layer Impedance

### Single Layer Impedance

For a viscoelastic layer with thickness $d$ and complex modulus $G^*$:

$$
Z_{\text{layer}}^* = iZ_{\text{bulk}}^* \tan(D^*)
$$

where:
$$
Z_{\text{bulk}}^* = \sqrt{\rho G^*}
$$

$$
D^* = \frac{2\pi f d \rho}{Z_{\text{bulk}}^*}
$$

### Multilayer Transfer Matrix

For $N$ layers, RheoQCM uses a recursive impedance calculation:

$$
Z_{\text{total}}^* = Z_1^* \frac{Z_{\text{inner}}^* + iZ_1^* \tan(k_1 d_1)}
{Z_1^* + iZ_{\text{inner}}^* \tan(k_1 d_1)}
$$

where $Z_{\text{inner}}^*$ is the effective impedance of all layers above
layer 1.

## Using Multilayer Analysis in RheoQCM

### Python API

```python
from rheoQCM.core.multilayer import calc_ZL, calc_delfstar_multilayer

# Define layers (from electrode outward)
layers = [
    {
        'drho': 1e-5,       # kg/m² (thickness × density)
        'grho': 1e8,        # Pa·kg/m³ (modulus × density)
        'phi': 0.3,         # rad (phase angle)
    },
    {
        'drho': float('inf'),  # bulk (semi-infinite)
        'grho': 1e6,           # water-like
        'phi': 1.57,           # ~90° for Newtonian liquid
    }
]

# Calculate complex frequency shift at harmonic n=3
delfstar = calc_delfstar_multilayer(
    n=3,
    layers=layers,
    f1=5e6,
)
print(f"Δf* = {delfstar.real:.1f} + {delfstar.imag:.1f}i Hz")
```

### Layer Specification

Each layer dictionary requires:

| Key | Description | Units |
|-----|-------------|-------|
| `drho` | Thickness × density | kg/m² |
| `grho` | Modulus × density at reference | Pa·kg/m³ |
| `phi` | Phase angle | radians |

For the **outermost layer** (bulk medium):
- Set `drho = float('inf')` or a very large value
- The layer is treated as semi-infinite

### Common Configurations

#### Film in Air
```python
layers = [
    {'drho': 1e-5, 'grho': 1e9, 'phi': 0.1},  # polymer film
    # No second layer needed for air
]
```

#### Film in Water
```python
from rheoQCM.core.physics import water_default

layers = [
    {'drho': 1e-5, 'grho': 1e8, 'phi': 0.3},  # polymer film
    water_default,  # predefined water properties
]
```

#### Bilayer Film
```python
layers = [
    {'drho': 1e-6, 'grho': 1e9, 'phi': 0.05},   # hard underlayer
    {'drho': 2e-6, 'grho': 1e7, 'phi': 0.5},    # soft overlayer
    water_default,                               # surrounding liquid
]
```

## Electrode Effects

The crystal electrode itself contributes to the measured response. RheoQCM
accounts for this using:

### Electrode Correction

```python
from rheoQCM.core.physics import electrode_default

# Include electrode layer
layers = [
    electrode_default,  # gold electrode (~150 nm)
    {'drho': 1e-5, 'grho': 1e8, 'phi': 0.3},  # sample
    water_default,
]
```

The default electrode properties assume:
- Gold electrodes
- Thickness ~150 nm per side
- Total $d\rho \approx 2.9 \times 10^{-4}$ kg/m²

## Validation and Limits

### Model Assumptions

The multilayer model assumes:

1. **Planar geometry**: Layers are uniform and infinite in lateral extent
2. **Linear viscoelasticity**: Small strain amplitudes
3. **No slip**: Perfect bonding between layers
4. **Homogeneous layers**: Uniform properties within each layer

### When the Model Breaks Down

| Scenario | Problem | Mitigation |
|----------|---------|------------|
| Rough interfaces | Scattering losses | Use effective medium approach |
| Particle layers | Non-planar geometry | Apply correction factors |
| Very soft films | Large strains | Reduce drive amplitude |
| Poor adhesion | Slip at interface | Improve sample preparation |

## Practical Example: Polymer Film Swelling

```python
import jax.numpy as jnp
from rheoQCM.core.multilayer import calc_delfstar_multilayer

# Track film as it swells in solvent
def analyze_swelling(drho_dry, grho_dry, phi_dry, swelling_ratio):
    """
    Calculate expected frequency shift during swelling.

    Assumes volume increases, modulus decreases proportionally.
    """
    drho_swollen = drho_dry * swelling_ratio
    grho_swollen = grho_dry / swelling_ratio**2  # dilution

    layers = [
        {'drho': drho_swollen, 'grho': grho_swollen, 'phi': phi_dry},
        {'drho': float('inf'), 'grho': 1e6, 'phi': 1.57},  # solvent
    ]

    delfstar = calc_delfstar_multilayer(n=3, layers=layers, f1=5e6)
    return delfstar

# Example: track 10% to 200% swelling
for ratio in [1.0, 1.1, 1.5, 2.0]:
    df = analyze_swelling(1e-5, 1e8, 0.3, ratio)
    print(f"Swelling {ratio:.0%}: Δf = {df.real:.0f} Hz, ΔΓ = {df.imag:.0f} Hz")
```

## Next Steps

- {doc}`numerical-methods` - How RheoQCM solves inverse problems
- {doc}`../tutorials/batch-analysis` - Analyze time-series data
