# QCM Fundamentals

This chapter introduces the fundamental physics of the Quartz Crystal
Microbalance (QCM) and explains how frequency and dissipation measurements
relate to material properties.

## The Piezoelectric Effect in Quartz

The QCM consists of a single-crystal quartz disc (typically AT-cut) sandwiched
between two electrodes. Due to the piezoelectric nature of quartz, applying an
alternating potential across the electrodes causes the material to oscillate
transversely, propagating a shear wave through the disc.

```{figure} ../_images/fig1.svg
:name: fig-qcm-schematic
:width: 60%
:align: center

Schematic of a QCM crystal showing the electrode configuration and shear wave
propagation.
```

### Resonance Conditions

When the oscillation frequency approaches the acoustic resonance frequency of
the quartz crystal, a standing shear wave forms across the disc. This produces
a peak in the electrical conductance that can be measured.

For AT-cut crystals:
- **Fundamental frequency**: $f_1 \approx 5$ MHz
- **Odd harmonics**: $f_n = n \cdot f_1$ where $n = 1, 3, 5, 7, ...$

The resonance at each harmonic produces a Lorentzian conductance peak
characterized by:

1. **Resonance frequency** ($f_n$): The frequency at peak conductance
2. **Half-bandwidth** ($\Gamma_n$): The half-width at half-maximum

```{note}
The half-bandwidth $\Gamma_n$ is related to the dissipation factor $D$
used in QCM-D instruments by: $\Gamma_n = D \cdot f_n / 2$
```

## Complex Frequency Shift

When a load (film, liquid, or particles) is applied to the QCM surface, both
$f_n$ and $\Gamma_n$ shift relative to the bare crystal. These shifts are
combined into a complex frequency shift:

$$
\Delta f_n^* = \Delta f_n + i\Delta\Gamma_n
$$

where:
- $\Delta f_n = f_n^{\text{loaded}} - f_n^{\text{bare}}$ (typically negative for mass loading)
- $\Delta\Gamma_n = \Gamma_n^{\text{loaded}} - \Gamma_n^{\text{bare}}$ (typically positive)

## The Small Load Approximation (SLA)

When the frequency shift is small compared to the resonance frequency (which
is true in nearly all QCM applications), the complex frequency shift is
related to the load impedance by:

$$
\Delta f_n^* = \frac{i f_1 \Delta Z_{nL}^*}{\pi Z_q}
$$

where:
- $Z_q = 8.84 \times 10^6$ kg/m²s is the acoustic shear impedance of AT-cut quartz
- $\Delta Z_{nL}^*$ is the complex load impedance

This **Small Load Approximation** is the foundation of all QCM modeling in
RheoQCM.

## The Sauerbrey Equation

### Derivation

For a thin, rigid film with no energy dissipation, the load impedance is
purely inertial:

$$
\Delta Z_{sn}^* = 2\pi i n f_1 \rho_f d_f
$$

Substituting into the SLA equation yields the **Sauerbrey equation**:

$$
\Delta f_{sn} = -\frac{2nf_1^2}{Z_q} \rho_f d_f = -\frac{2nf_1^2}{Z_q} \Delta M_A
$$

where $\Delta M_A = \rho_f d_f$ is the mass per unit area of the film.

### When to Use Sauerbrey

The Sauerbrey equation is valid when:

1. The film is **thin** compared to the acoustic wavelength: $d \ll \lambda$
2. The film is **rigid** with negligible viscous losses: $\phi \approx 0$
3. The film has high **acoustic contrast** with the surrounding medium

```{warning}
For viscoelastic films or thick films, the Sauerbrey equation significantly
underestimates the true mass. Use the full viscoelastic analysis in RheoQCM
instead.
```

### Practical Calculation

For a 5 MHz fundamental frequency:

$$
\Delta f_{sn} \text{ [Hz]} = -56.6 \times n \times \Delta M_A \text{ [μg/cm²]}
$$

Or equivalently:

$$
\Delta M_A \text{ [ng/cm²]} = -17.7 \times \frac{\Delta f_n}{n} \text{ [Hz]}
$$

## Harmonic Dependence

Different harmonics probe the film at different depths due to the frequency
dependence of the acoustic wavelength:

$$
\lambda_n = \frac{\lambda_1}{n} = \frac{(\rho |G_n^*|)^{1/2}}{n f_1 \cos(\phi_n/2)}
$$

This means:
- **Lower harmonics** (n=1, 3) probe deeper into thick films
- **Higher harmonics** (n=5, 7, 9) are more surface-sensitive
- Harmonic ratios reveal information about film structure and viscoelasticity

```{figure} ../_images/fig2.svg
:name: fig-harmonic-penetration
:width: 70%
:align: center

Comparison of penetration depth at different harmonics for a viscoelastic film.
```

## Bulk Limit

For films much thicker than the acoustic decay length, the load impedance
equals the bulk acoustic impedance:

$$
Z_{n,\text{bulk}}^* = (\rho G_n^*)^{1/2}
$$

This gives:

$$
\Delta f_n = -\frac{f_1}{\pi Z_q} (\rho |G_n^*|)^{1/2} \sin(\phi_n/2)
$$

$$
\Delta\Gamma_n = \frac{f_1}{\pi Z_q} (\rho |G_n^*|)^{1/2} \cos(\phi_n/2)
$$

### Inverting for Material Properties

From bulk-limit measurements, you can directly extract:

$$
\rho |G_n^*| = \left(\frac{\pi Z_q |\Delta f_n^*|}{f_1}\right)^2
$$

$$
\phi = 2 \arctan\left(\frac{-\Delta f_n}{\Delta\Gamma_n}\right)
$$

## Decay Length

The acoustic shear wave decays exponentially into the film. The decay length
(penetration depth) is:

$$
\delta_{n,\rho} = \frac{(\rho |G_n^*|)^{1/2}}{2\pi n f_1 \sin(\phi_n/2)}
$$

**Rule of thumb**: The bulk limit applies when $d > 2\delta_n$.

For water at the 3rd harmonic (~15 MHz):
- $\delta \approx 120$ nm

For a glassy polymer ($|G^*| \sim 10^9$ Pa):
- $\delta \approx 5$ μm at the fundamental

## Summary

| Regime | Condition | Analysis Method |
|--------|-----------|-----------------|
| Sauerbrey | Thin, rigid films | Direct mass calculation |
| Viscoelastic | Thin films with dissipation | Full SLA model |
| Bulk | $d > 2\delta$ | Direct modulus extraction |

The next chapter explains how RheoQCM extracts viscoelastic properties from
QCM-D measurements.
