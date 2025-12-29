# Visualization Guide

This guide covers creating publication-quality plots from QCM-D analysis results.

## GUI Visualization

### Main Plot Window

The RheoQCM GUI provides real-time plotting during acquisition and analysis:

| Plot Type | Y-Axis | Description |
|-----------|--------|-------------|
| Δf vs Time | Frequency shift [Hz] | Mass/stiffness changes |
| ΔΓ vs Time | Half-bandwidth [Hz] | Energy dissipation |
| drho vs Time | Areal mass [kg/m²] | Film thickness proxy |
| grho vs Time | |G*|ρ [Pa·kg/m³] | Viscoelastic stiffness |
| φ vs Time | Phase angle [rad] | Loss tangent |

### Navigating Plots

| Action | Mouse | Keyboard |
|--------|-------|----------|
| Pan | Left-drag | Arrow keys |
| Zoom box | Right-drag | - |
| Zoom in/out | Scroll wheel | +/- |
| Reset view | Double-click | Home |
| Export | Right-click menu | Ctrl+E |

### Plot Options

Access via **View → Plot Settings**:

- **Show grid**: Toggle grid lines
- **Log scale**: Logarithmic y-axis for grho
- **Markers**: Show data points
- **Line width**: Adjust line thickness
- **Colors**: Customize harmonic colors

## Python Visualization

### Basic Time Series Plot

```python
import matplotlib.pyplot as plt
import numpy as np
from rheoQCM.core.analysis import batch_analyze

# Process data
results = batch_analyze(measurements, ...)

# Create figure
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Areal mass
ax = axes[0]
ax.plot(time, results.drho * 1e6, 'b-', linewidth=1.5)
ax.set_ylabel(r'$d\rho$ [$\mu$g/cm$^2$]')
ax.grid(True, alpha=0.3)

# Modulus
ax = axes[1]
ax.semilogy(time, results.grho_refh, 'r-', linewidth=1.5)
ax.set_ylabel(r'$|G^*|\rho$ [Pa·kg/m$^3$]')
ax.grid(True, alpha=0.3)

# Phase angle
ax = axes[2]
ax.plot(time, np.rad2deg(results.phi), 'g-', linewidth=1.5)
ax.set_ylabel(r'$\phi$ [°]')
ax.set_xlabel('Time [s]')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('timeseries.png', dpi=300, bbox_inches='tight')
```

### Raw Data Plot (Δf and ΔΓ)

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Frequency shifts by harmonic
ax = axes[0]
for n in [3, 5, 7]:
    ax.plot(time, delf[n], label=f'n={n}')
ax.set_ylabel(r'$\Delta f_n/n$ [Hz]')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)

# Dissipation by harmonic
ax = axes[1]
for n in [3, 5, 7]:
    ax.plot(time, delg[n], label=f'n={n}')
ax.set_ylabel(r'$\Delta\Gamma_n$ [Hz]')
ax.set_xlabel('Time [s]')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('raw_data.png', dpi=300)
```

### ΔΓ vs Δf Plot (Dissipation vs Mass)

This plot reveals viscoelastic behavior:

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 6))

# Plot each harmonic
colors = plt.cm.viridis(np.linspace(0, 1, 3))
for i, n in enumerate([3, 5, 7]):
    ax.plot(
        -delf[n] / n,  # Normalize by harmonic number
        delg[n],
        'o-',
        color=colors[i],
        label=f'n={n}',
        markersize=2,
        alpha=0.7,
    )

ax.set_xlabel(r'$-\Delta f_n/n$ [Hz]')
ax.set_ylabel(r'$\Delta\Gamma_n$ [Hz]')
ax.legend()
ax.grid(True, alpha=0.3)

# Add Sauerbrey line for reference (purely elastic film)
delf_range = ax.get_xlim()
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Sauerbrey limit')

plt.tight_layout()
plt.savefig('dg_vs_df.png', dpi=300)
```

### Cole-Cole Plot (G'' vs G')

Reveals relaxation behavior:

```python
import matplotlib.pyplot as plt
import numpy as np

# Calculate G' and G'' from results
gprime = results.grho_refh * np.cos(results.phi)
gdoubleprime = results.grho_refh * np.sin(results.phi)

fig, ax = plt.subplots(figsize=(8, 8))

# Color by time
scatter = ax.scatter(
    gprime,
    gdoubleprime,
    c=time,
    cmap='viridis',
    s=10,
    alpha=0.7,
)
plt.colorbar(scatter, label='Time [s]')

ax.set_xlabel(r"$G'\rho$ [Pa·kg/m$^3$]")
ax.set_ylabel(r"$G''\rho$ [Pa·kg/m$^3$]")
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cole_cole.png', dpi=300)
```

### Master Curve (φ vs |G*|ρ)

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 6))

ax.loglog(results.grho_refh, np.rad2deg(results.phi), 'o', markersize=3)

ax.set_xlabel(r'$|G^*|\rho$ [Pa·kg/m$^3$]')
ax.set_ylabel(r'$\phi$ [°]')
ax.grid(True, which='both', alpha=0.3)

# Add guide lines
grho_range = np.logspace(6, 10, 100)
ax.axhline(y=45, color='r', linestyle='--', alpha=0.5, label='Maxwell fluid')
ax.axhline(y=0, color='b', linestyle='--', alpha=0.5, label='Elastic solid')

ax.legend()
plt.tight_layout()
plt.savefig('master_curve.png', dpi=300)
```

## Uncertainty Visualization

### With Credible Intervals

```python
import matplotlib.pyplot as plt
import numpy as np

# Assuming Bayesian results with uncertainty
fig, ax = plt.subplots(figsize=(10, 4))

# Plot mean with uncertainty band
ax.plot(time, drho_mean * 1e6, 'b-', label='Mean')
ax.fill_between(
    time,
    drho_ci_low * 1e6,
    drho_ci_high * 1e6,
    alpha=0.3,
    color='blue',
    label='95% CI',
)

ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$d\rho$ [$\mu$g/cm$^2$]')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('with_uncertainty.png', dpi=300)
```

### Error Bars

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 4))

# Subsample for clarity
step = max(1, len(time) // 50)
idx = slice(None, None, step)

ax.errorbar(
    time[idx],
    drho_mean[idx] * 1e6,
    yerr=[
        (drho_mean[idx] - drho_ci_low[idx]) * 1e6,
        (drho_ci_high[idx] - drho_mean[idx]) * 1e6,
    ],
    fmt='o',
    capsize=3,
    markersize=4,
)

ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$d\rho$ [$\mu$g/cm$^2$]')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('error_bars.png', dpi=300)
```

## Comparison Plots

### Multiple Experiments

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

experiments = ['exp1', 'exp2', 'exp3']
colors = ['blue', 'red', 'green']

for exp, color in zip(experiments, colors):
    data = results_all[exp]
    ax.plot(
        data['time'],
        data['drho'] * 1e6,
        color=color,
        label=exp,
        linewidth=1.5,
    )

ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$d\rho$ [$\mu$g/cm$^2$]')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison.png', dpi=300)
```

### Harmonic Comparison

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Process with different harmonic choices
nhcalc_options = ['353', '355', '575']

for ax, nhcalc in zip(axes, nhcalc_options):
    results = batch_analyze(measurements, nhcalc=nhcalc, ...)
    ax.plot(time, results.drho * 1e6)
    ax.set_title(f'nhcalc = {nhcalc}')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'$d\rho$ [$\mu$g/cm$^2$]')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('harmonic_comparison.png', dpi=300)
```

## Publication-Quality Figures

### Style Configuration

```python
import matplotlib.pyplot as plt

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')

# Or configure manually
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (3.5, 2.5),  # Single column width
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
})
```

### Two-Column Figure

```python
import matplotlib.pyplot as plt

# Two-column width figure
fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))

ax = axes[0]
ax.plot(time, results.drho * 1e6)
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$d\rho$ [$\mu$g/cm$^2$]')
ax.set_title('(a)')

ax = axes[1]
ax.semilogy(time, results.grho_refh)
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$|G^*|\rho$ [Pa·kg/m$^3$]')
ax.set_title('(b)')

plt.tight_layout()
plt.savefig('figure_pub.pdf')  # Vector format
plt.savefig('figure_pub.png', dpi=300)  # Raster format
```

## Interactive Visualization

### With Plotly

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
)

# drho
fig.add_trace(
    go.Scatter(x=time, y=results.drho * 1e6, name='drho'),
    row=1, col=1,
)

# grho
fig.add_trace(
    go.Scatter(x=time, y=results.grho_refh, name='grho'),
    row=2, col=1,
)

# phi
fig.add_trace(
    go.Scatter(x=time, y=np.rad2deg(results.phi), name='phi'),
    row=3, col=1,
)

fig.update_layout(height=600, title='QCM Analysis Results')
fig.update_yaxes(type='log', row=2, col=1)

fig.write_html('interactive_plot.html')
fig.show()
```

## Exporting Figures

### Formats

| Format | Use Case |
|--------|----------|
| PNG | Web, presentations |
| PDF | Publications, vector graphics |
| SVG | Web, editable vector |
| EPS | LaTeX documents |

### Export Commands

```python
# High-resolution PNG
plt.savefig('figure.png', dpi=300, bbox_inches='tight')

# Vector PDF
plt.savefig('figure.pdf', bbox_inches='tight')

# SVG for web
plt.savefig('figure.svg', bbox_inches='tight')

# Transparent background
plt.savefig('figure.png', dpi=300, transparent=True)
```

## Next Steps

- {doc}`exporting-results` - Export data to files
- {doc}`../tutorials/bayesian-fitting` - Uncertainty quantification
- {doc}`../api/index` - Full API reference
