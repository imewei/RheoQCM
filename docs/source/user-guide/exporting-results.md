# Exporting Results Guide

This guide covers saving and exporting QCM-D analysis results in various formats.

## GUI Export Options

### Export Menu

Access via **File → Export**:

| Option | Format | Description |
|--------|--------|-------------|
| Export Data | Excel (.xlsx) | Full dataset with analysis |
| Export Selection | Excel (.xlsx) | Selected time range |
| Export Figure | PNG/PDF | Current plot |

### Quick Export

| Shortcut | Action |
|----------|--------|
| Ctrl+E | Export current data |
| Ctrl+Shift+E | Export figure |

## Python Export Methods

### To Excel

```python
import pandas as pd
from rheoQCM.core.analysis import batch_analyze

# Process data
results = batch_analyze(measurements, ...)

# Create DataFrame
df = pd.DataFrame({
    'time_s': time,
    'drho_kgm2': results.drho,
    'drho_ugcm2': results.drho * 1e6,
    'grho_Pa_kgm3': results.grho_refh,
    'phi_rad': results.phi,
    'phi_deg': np.rad2deg(results.phi),
    'converged': results.converged,
})

# Export to Excel
df.to_excel('analysis_results.xlsx', index=False)

# With multiple sheets
with pd.ExcelWriter('full_results.xlsx') as writer:
    df.to_excel(writer, sheet_name='Analysis', index=False)

    # Raw data sheet
    df_raw = pd.DataFrame({
        'time_s': time,
        'delf3': delf3,
        'delg3': delg3,
        'delf5': delf5,
        'delg5': delg5,
    })
    df_raw.to_excel(writer, sheet_name='Raw Data', index=False)

    # Metadata sheet
    df_meta = pd.DataFrame({
        'Parameter': ['f1', 'refh', 'nhcalc'],
        'Value': [5e6, 3, '353'],
    })
    df_meta.to_excel(writer, sheet_name='Metadata', index=False)
```

### To CSV

```python
import pandas as pd

df = pd.DataFrame({
    'time_s': time,
    'drho_kgm2': results.drho,
    'grho_Pa_kgm3': results.grho_refh,
    'phi_rad': results.phi,
})

# Standard CSV
df.to_csv('results.csv', index=False)

# Tab-separated
df.to_csv('results.tsv', sep='\t', index=False)

# With specific precision
df.to_csv('results.csv', index=False, float_format='%.6e')
```

### To NumPy Format

```python
import numpy as np

# Save as NPZ (compressed)
np.savez_compressed(
    'results.npz',
    time=time,
    drho=results.drho,
    grho_refh=results.grho_refh,
    phi=results.phi,
    converged=results.converged,
    # Metadata as attributes
    f1=5e6,
    refh=3,
)

# Load later
data = np.load('results.npz')
drho = data['drho']
f1 = float(data['f1'])
```

### To HDF5

For large datasets or complex hierarchical data:

```python
import h5py
import numpy as np

with h5py.File('results.h5', 'w') as f:
    # Create groups
    raw = f.create_group('raw_data')
    analysis = f.create_group('analysis')

    # Raw data
    raw.create_dataset('time', data=time)
    raw.create_dataset('delf3', data=delf3)
    raw.create_dataset('delg3', data=delg3)
    raw.create_dataset('delf5', data=delf5)
    raw.create_dataset('delg5', data=delg5)

    # Analysis results
    analysis.create_dataset('drho', data=results.drho)
    analysis.create_dataset('grho_refh', data=results.grho_refh)
    analysis.create_dataset('phi', data=results.phi)
    analysis.create_dataset('converged', data=results.converged)

    # Metadata as attributes
    f.attrs['f1'] = 5e6
    f.attrs['refh'] = 3
    f.attrs['nhcalc'] = '353'
    f.attrs['timestamp'] = str(datetime.now())
```

### To Parquet (Fast Binary)

```python
import pandas as pd

df = pd.DataFrame({
    'time_s': time,
    'drho_kgm2': results.drho,
    'grho_Pa_kgm3': results.grho_refh,
    'phi_rad': results.phi,
})

# Parquet is efficient for large datasets
df.to_parquet('results.parquet', compression='snappy')

# Read back
df_loaded = pd.read_parquet('results.parquet')
```

## Export with Uncertainty

### Bayesian Results

```python
import pandas as pd
import numpy as np

# After Bayesian fitting
df = pd.DataFrame({
    'time_s': time,
    'drho_mean': drho_mean,
    'drho_std': drho_std,
    'drho_ci_low': drho_ci[:, 0],
    'drho_ci_high': drho_ci[:, 1],
    'grho_mean': grho_mean,
    'grho_std': grho_std,
    'phi_mean': phi_mean,
    'phi_std': phi_std,
})

df.to_excel('bayesian_results.xlsx', index=False)
```

### MCMC Samples

```python
import numpy as np

# Save full posterior samples
np.savez(
    'mcmc_samples.npz',
    drho_samples=drho_samples,    # Shape: (n_samples,)
    grho_samples=grho_samples,
    phi_samples=phi_samples,
)
```

## Export Figures

### Matplotlib Figures

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(time, results.drho * 1e6)
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$d\rho$ [$\mu$g/cm$^2$]')

# PNG for presentations
fig.savefig('figure.png', dpi=300, bbox_inches='tight')

# PDF for publications
fig.savefig('figure.pdf', bbox_inches='tight')

# SVG for web/editing
fig.savefig('figure.svg', bbox_inches='tight')
```

### Figure with Metadata

```python
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Save with matplotlib
fig.savefig('figure.png', dpi=300)

# Add metadata
img = Image.open('figure.png')
metadata = PngInfo()
metadata.add_text('Title', 'QCM-D Analysis Results')
metadata.add_text('Author', 'RheoQCM')
metadata.add_text('f1', '5000000')
metadata.add_text('nhcalc', '353')
img.save('figure_with_metadata.png', pnginfo=metadata)
```

## Batch Export

### Export Multiple Experiments

```python
from pathlib import Path
import pandas as pd

output_dir = Path('exports/')
output_dir.mkdir(exist_ok=True)

for name, data in results_all.items():
    df = pd.DataFrame({
        'time_s': data['time'],
        'drho_kgm2': data['drho'],
        'grho_Pa_kgm3': data['grho'],
        'phi_rad': data['phi'],
    })

    df.to_excel(output_dir / f'{name}_analysis.xlsx', index=False)
    print(f'Exported {name}')
```

### Combined Summary

```python
import pandas as pd

# Create summary of all experiments
summary_rows = []
for name, data in results_all.items():
    summary_rows.append({
        'experiment': name,
        'n_points': len(data['time']),
        'drho_mean': data['drho'].mean(),
        'drho_std': data['drho'].std(),
        'grho_mean': data['grho'].mean(),
        'phi_mean': data['phi'].mean(),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_excel('experiment_summary.xlsx', index=False)
```

## Export for Other Software

### For Origin/SigmaPlot

```python
import pandas as pd

# Use tab-separated with clear headers
df = pd.DataFrame({
    'Time (s)': time,
    'drho (ug/cm2)': results.drho * 1e6,
    'grho (Pa kg/m3)': results.grho_refh,
    'phi (deg)': np.rad2deg(results.phi),
})

df.to_csv('for_origin.txt', sep='\t', index=False)
```

### For MATLAB

```python
import scipy.io as sio

sio.savemat('results.mat', {
    'time': time,
    'drho': results.drho,
    'grho_refh': results.grho_refh,
    'phi': results.phi,
    'f1': 5e6,
    'refh': 3,
})
```

### For R

```python
import pandas as pd

df = pd.DataFrame({
    'time_s': time,
    'drho_kgm2': results.drho,
    'grho_Pa_kgm3': results.grho_refh,
    'phi_rad': results.phi,
})

# R can read CSV directly
df.to_csv('for_r.csv', index=False)

# Or use feather format for better compatibility
df.to_feather('for_r.feather')
```

## Data Format Reference

### Standard Column Names

| Column | Units | Description |
|--------|-------|-------------|
| `time_s` | s | Time since start |
| `drho_kgm2` | kg/m² | Areal mass density |
| `drho_ugcm2` | μg/cm² | Areal mass density |
| `grho_Pa_kgm3` | Pa·kg/m³ | Complex modulus × density |
| `phi_rad` | rad | Phase angle |
| `phi_deg` | ° | Phase angle |
| `delf{n}` | Hz | Frequency shift at harmonic n |
| `delg{n}` | Hz | Half-bandwidth shift at harmonic n |

### Metadata Keys

| Key | Description |
|-----|-------------|
| `f1` | Fundamental frequency [Hz] |
| `refh` | Reference harmonic |
| `nhcalc` | Harmonic selection string |
| `timestamp` | Analysis timestamp |
| `version` | RheoQCM version |

## Automatic Export

### After Each Analysis

```python
from datetime import datetime
from pathlib import Path

def auto_export(results, time, prefix='analysis'):
    """Automatically export results after analysis."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{prefix}_{timestamp}.xlsx'

    df = pd.DataFrame({
        'time_s': time,
        'drho_kgm2': results.drho,
        'grho_Pa_kgm3': results.grho_refh,
        'phi_rad': results.phi,
    })

    df.to_excel(filename, index=False)
    print(f'Auto-exported to {filename}')
    return filename

# Usage
results = batch_analyze(measurements, ...)
auto_export(results, time)
```

### Scheduled Backup

```python
import schedule
import time

def backup_results():
    """Periodic backup of results."""
    # Implementation...
    pass

schedule.every(10).minutes.do(backup_results)

while True:
    schedule.run_pending()
    time.sleep(1)
```

## Next Steps

- {doc}`../tutorials/batch-analysis` - Process large datasets
- {doc}`../api/index` - Full API reference
