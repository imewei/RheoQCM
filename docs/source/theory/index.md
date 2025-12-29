# Theory & Background

This section provides the theoretical foundation for understanding QCM-D
(Quartz Crystal Microbalance with Dissipation) measurements and the
rheological analysis performed by RheoQCM.

## Overview

The Quartz Crystal Microbalance is a high-sensitivity mass-measuring device
that exploits the piezoelectric nature of quartz crystals. When an alternating
voltage is applied across a quartz crystal, it oscillates at its resonant
frequency. Changes in mass or viscoelastic properties of materials in contact
with the crystal surface cause measurable shifts in both the resonant
frequency ($\Delta f$) and the energy dissipation ($\Delta \Gamma$).

RheoQCM extends traditional QCM analysis by:

1. **Extracting viscoelastic properties** from the complex frequency shift
2. **Modeling multilayer thin films** using continuum mechanics
3. **Quantifying uncertainty** through Bayesian inference

## Chapter Guide

```{toctree}
:maxdepth: 2

qcm-fundamentals
viscoelastic-modeling
multilayer-films
numerical-methods
```

## Quick Reference

### Key Equations

| Equation | Description |
|----------|-------------|
| Sauerbrey | $\Delta f_{sn} = \frac{2nf_1^2}{Z_q} \rho d$ |
| Complex frequency shift | $\Delta f_n^* = \Delta f_n + i\Delta\Gamma_n$ |
| Small Load Approximation | $\Delta f_n^* = \frac{if_1 \Delta Z_{nL}^*}{\pi Z_q}$ |
| Complex modulus | $G_n^* = \|G_n^*\| e^{i\phi_n}$ |

### Physical Constants

| Constant | Symbol | Value | Units |
|----------|--------|-------|-------|
| Quartz acoustic impedance | $Z_q$ | $8.84 \times 10^6$ | kg/m²s |
| Fundamental frequency | $f_1$ | 5 MHz | Hz |
| Quartz density | $\rho_q$ | 2650 | kg/m³ |
| Quartz shear modulus | $\mu_q$ | $2.95 \times 10^{10}$ | Pa |

## Recommended Reading

For a deeper understanding of QCM theory, we recommend:

1. Johannsmann, D. (2008). "Viscoelastic, mechanical, and dielectric
   measurements on complex samples with the quartz crystal microbalance."
   *Physical Chemistry Chemical Physics*, 10(31), 4516-4534.

2. DeNolf, G.C. et al. (2011). "High Frequency Rheometry of Viscoelastic
   Coatings with the Quartz Crystal Microbalance." *Langmuir*, 27(16),
   9873-9879.

3. Sauerbrey, G. (1959). "Verwendung von Schwingquarzen zur Wägung dünner
   Schichten und zur Mikrowägung." *Zeitschrift für Physik*, 155(2), 206-222.
