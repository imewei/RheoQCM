# Viscoelastic Modeling

This chapter explains how RheoQCM extracts viscoelastic properties from QCM-D
measurements using the complex shear modulus framework.

## Complex Shear Modulus

Viscoelastic materials exhibit both elastic (storage) and viscous (loss)
behavior. This is captured by the **complex shear modulus**:

$$
G_n^* = G_n' + iG_n'' = |G_n^*| e^{i\phi_n}
$$

where:
- $G_n'$ = Storage modulus (elastic component)
- $G_n''$ = Loss modulus (viscous component)
- $|G_n^*|$ = Magnitude of the complex modulus
- $\phi_n$ = Phase angle (loss angle)

### Physical Interpretation

| Parameter | Physical Meaning |
|-----------|-----------------|
| $G'$ | Energy stored elastically per cycle |
| $G''$ | Energy dissipated viscously per cycle |
| $\|G^*\|$ | Overall stiffness |
| $\phi$ | Ratio of viscous to elastic response |

### Phase Angle Limits

- $\phi = 0°$: Perfectly elastic solid (no dissipation)
- $\phi = 90°$: Newtonian liquid (no elasticity)
- $0° < \phi < 90°$: Viscoelastic material

Most soft matter falls in the range $10° < \phi < 60°$.

## Conversion Formulas

From magnitude and phase to storage/loss moduli:

$$
G' = |G^*| \cos(\phi)
$$

$$
G'' = |G^*| \sin(\phi)
$$

From storage/loss to magnitude and phase:

$$
|G^*| = \sqrt{(G')^2 + (G'')^2}
$$

$$
\phi = \arctan\left(\frac{G''}{G'}\right)
$$

## RheoQCM Output Parameters

RheoQCM reports viscoelastic properties in specific forms optimized for
QCM analysis:

### $|G^*|\rho$ (grho)

The product of modulus magnitude and density:

$$
|G^*|\rho = |G_n^*| \cdot \rho
$$

**Units**: Pa·kg/m³

This quantity is directly measurable from QCM data without knowing the
density separately. It's related to the acoustic impedance by:

$$
Z^* = \sqrt{|G^*|\rho} \cdot e^{i\phi/2}
$$

### Reference Harmonic

Because viscoelastic materials are frequency-dependent, RheoQCM reports
$|G^*|\rho$ at a **reference harmonic** (typically n=3):

$$
|G_3^*|\rho = |G^*|_{f=15\text{ MHz}} \cdot \rho
$$

### Thickness-Density Product ($d\rho$)

The areal mass density of the film:

$$
d\rho = \rho \cdot d = \Delta M_A
$$

**Units**: kg/m² (or μg/cm²)

## The Thin Film Solution

For a viscoelastic thin film in the SLA regime, RheoQCM solves for three
parameters:

1. **$d\rho$** - Film thickness × density (areal mass)
2. **$|G_3^*|\rho$** - Complex modulus × density at reference harmonic
3. **$\phi$** - Phase angle (assumed frequency-independent in SLA)

### Film Impedance

The impedance of a viscoelastic film is:

$$
Z_{n,\text{film}}^* = iZ_{n,\text{bulk}}^* \tan(D_n^*)
$$

where $D_n^*$ is the complex phase factor:

$$
D_n^* = k_n^* d = \frac{2\pi f_n d\rho}{Z_n^*}
$$

### Harmonic Scaling

For a film with frequency-independent $\phi$, the complex modulus scales with
frequency as:

$$
|G_n^*| = |G_{\text{ref}}^*| \cdot \left(\frac{n}{n_{\text{ref}}}\right)^{\phi/(\pi/2)}
$$

This scaling is built into the RheoQCM model and allows fitting data from
multiple harmonics simultaneously.

## Using the Analysis in RheoQCM

### Python API Example

```python
from rheoQCM.core import QCMModel, configure_jax

configure_jax()

# Create model with 5 MHz fundamental, reference harmonic n=3
model = QCMModel(f1=5e6, refh=3)

# Load experimental frequency shifts
model.load_delfstars({
    3: -1000 + 100j,   # Δf + iΔΓ at n=3
    5: -1700 + 180j,   # Δf + iΔΓ at n=5
    7: -2500 + 280j,   # Δf + iΔΓ at n=7
})

# Solve for viscoelastic properties
result = model.solve_properties(nh=[3, 5, 3])

print(f"Areal mass: {result.drho * 1e6:.2f} μg/cm²")
print(f"|G*|ρ at n=3: {result.grho_refh:.2e} Pa·kg/m³")
print(f"Phase angle: {result.phi * 180/3.14159:.1f}°")
```

### Understanding the nhcalc Parameter

The `nh=[n1, n2, n3]` parameter specifies which measured quantities to use:

- **n1**: Harmonic for first frequency shift ($\Delta f_{n1}$)
- **n2**: Harmonic for second frequency shift ($\Delta f_{n2}$)
- **n3**: Harmonic for dissipation shift ($\Delta\Gamma_{n3}$)

Common choices:
- `nh=[3, 5, 3]` - Uses harmonics 3 and 5 for frequency, 3 for dissipation
- `nh=[3, 5, 5]` - Uses harmonics 3 and 5 for frequency, 5 for dissipation
- `nh=[5, 7, 5]` - Higher harmonics for thinner films

## Frequency Dependence Models

### Power Law (Default)

RheoQCM assumes a power-law frequency dependence:

$$
|G^*|(f) = |G^*|_{f_0} \left(\frac{f}{f_0}\right)^{\phi/(\pi/2)}
$$

This is equivalent to assuming constant $\phi$ across harmonics.

### Kotula Model

For more complex materials, RheoQCM includes the Kotula fractional
viscoelastic model, which provides better accuracy for:

- Polymer melts near $T_g$
- Highly viscoelastic hydrogels
- Materials with broad relaxation spectra

```python
from rheoQCM.core.physics import kotula_gstar

# Calculate complex modulus using Kotula model
gstar = kotula_gstar(
    frequency=15e6,
    grho_refh=1e8,
    phi=0.5,
    xi=0.3  # fractional exponent
)
```

## Practical Considerations

### Choosing the Reference Harmonic

| Reference Harmonic | Best For |
|-------------------|----------|
| n=1 | Thick, soft films (>1 μm) |
| n=3 | General use (default) |
| n=5 | Thin, stiff films (<100 nm) |

### Validation Checks

1. **Sauerbrey ratio**: Compare $\Delta f_n / n$ across harmonics
   - Constant ratio → Sauerbrey regime (rigid film)
   - Decreasing ratio → Viscoelastic effects significant

2. **Dissipation ratio**: Check $\Delta\Gamma_n / \Delta f_n$
   - Small ratio (<0.1) → Rigid film
   - Large ratio (>0.3) → Highly viscoelastic

3. **Residuals**: Examine fit residuals across harmonics
   - Random scatter → Good model
   - Systematic trend → Model assumptions violated

## Next Steps

- {doc}`multilayer-films` - Modeling complex film structures
- {doc}`numerical-methods` - How RheoQCM solves for properties
