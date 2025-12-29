# Batch Analysis Tutorial

This tutorial covers efficient processing of large QCM-D datasets using
RheoQCM's batch analysis capabilities and GPU acceleration.

## When to Use Batch Processing

Use batch processing when you have:
- Time-series data with >100 measurements
- Multiple experiments to process
- Large datasets requiring GPU acceleration
- Automated analysis pipelines

## Batch Processing Methods

RheoQCM provides three methods for batch processing:

| Method | Best For | Speed |
|--------|----------|-------|
| Loop with QCMModel | Small datasets (<100 pts) | 1x |
| `batch_analyze()` | Medium datasets (100-10k pts) | 10x |
| `batch_analyze_vmap()` | Large datasets (>10k pts) | 100x |

## Method 1: Simple Loop

For small datasets or when you need maximum flexibility:

```python
from rheoQCM.core import QCMModel, configure_jax
import numpy as np

configure_jax()

# Load your data
time = np.load('time.npy')
delf3 = np.load('delf3.npy')
delg3 = np.load('delg3.npy')
delf5 = np.load('delf5.npy')
delg5 = np.load('delg5.npy')

# Process each time point
model = QCMModel(f1=5e6, refh=3)
results = []

for i in range(len(time)):
    model.load_delfstars({
        3: delf3[i] + 1j * delg3[i],
        5: delf5[i] + 1j * delg5[i],
    })
    result = model.solve_properties(nh=[3, 5, 3])
    results.append({
        't': time[i],
        'drho': result.drho,
        'grho': result.grho_refh,
        'phi': result.phi,
    })

# Convert to arrays
drho = np.array([r['drho'] for r in results])
```

## Method 2: batch_analyze()

For medium-sized datasets with automatic parallel processing:

```python
from rheoQCM.core.analysis import batch_analyze
import numpy as np

# Prepare data as list of dictionaries
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
    nhcalc='353',      # Use Δf at n=3,5 and ΔΓ at n=3
    f1=5e6,
    refh=3,
)

# Access results as arrays
print(f"Shape: {results.drho.shape}")  # (n_measurements,)
print(f"Mean drho: {np.mean(results.drho):.3e}")
```

### BatchResult Object

`batch_analyze()` returns a `BatchResult` dataclass:

```python
@dataclass
class BatchResult:
    drho: ndarray        # Shape: (n_measurements,)
    grho_refh: ndarray   # Shape: (n_measurements,)
    phi: ndarray         # Shape: (n_measurements,)
    dlam: ndarray        # Shape: (n_measurements,)
    converged: ndarray   # Shape: (n_measurements,), bool
```

## Method 3: batch_analyze_vmap() (GPU-Accelerated)

For maximum performance on large datasets:

```python
import jax.numpy as jnp
from rheoQCM.core import batch_analyze_vmap, configure_jax

configure_jax()

# Check if GPU is available
import jax
print(f"Backend: {jax.default_backend()}")  # 'gpu' or 'cpu'

# Prepare data as JAX array
# Shape: (n_measurements, n_harmonics)
# Order must match harmonics list
delfstars = jnp.array([
    [delf3[i] + 1j*delg3[i], delf5[i] + 1j*delg5[i], delf7[i] + 1j*delg7[i]]
    for i in range(len(time))
])

print(f"Input shape: {delfstars.shape}")  # (n_measurements, 3)

# Process with GPU acceleration
results = batch_analyze_vmap(
    delfstars,
    harmonics=[3, 5, 7],  # Must match column order
    f1=5e6,
    refh=3,
)

# Results are JAX arrays
print(f"drho mean: {jnp.mean(results.drho):.3e}")
```

### Performance Comparison

Typical performance on a dataset with 100,000 time points:

| Method | CPU Time | GPU Time |
|--------|----------|----------|
| Loop | ~500 s | N/A |
| batch_analyze | ~50 s | N/A |
| batch_analyze_vmap | ~5 s | ~0.5 s |

## Processing Multiple Files

### Sequential Processing

```python
from pathlib import Path
from rheoQCM.core.analysis import batch_analyze
from rheoQCM.modules.DataSaver import DataSaver

data_dir = Path("experiments/")
results_all = {}

for filepath in data_dir.glob("*.h5"):
    # Load file
    ds = DataSaver()
    ds.load_file(str(filepath))

    # Prepare measurements
    df = ds.samp
    measurements = []
    for _, row in df.iterrows():
        measurements.append({
            3: row['delf3'] + 1j * row['delg3'],
            5: row['delf5'] + 1j * row['delg5'],
        })

    # Process
    results = batch_analyze(measurements, harmonics=[3, 5], ...)

    # Store
    results_all[filepath.stem] = {
        'time': df['t'].values,
        'drho': results.drho,
        'grho': results.grho_refh,
        'phi': results.phi,
    }
```

### Parallel File Processing

```python
from concurrent.futures import ProcessPoolExecutor
from rheoQCM.core import configure_jax
from rheoQCM.core.analysis import batch_analyze

def process_file(filepath):
    """Process a single file. Called in separate process."""
    configure_jax()  # Must configure in each process

    # Load and process
    ds = DataSaver()
    ds.load_file(str(filepath))

    df = ds.samp
    measurements = [
        {3: row['delf3'] + 1j * row['delg3'],
         5: row['delf5'] + 1j * row['delg5']}
        for _, row in df.iterrows()
    ]

    results = batch_analyze(measurements, harmonics=[3, 5], ...)

    return {
        'file': filepath.name,
        'drho_mean': float(results.drho.mean()),
        'grho_mean': float(results.grho_refh.mean()),
    }

# Process files in parallel
files = list(Path("experiments/").glob("*.h5"))

with ProcessPoolExecutor(max_workers=4) as executor:
    summary = list(executor.map(process_file, files))

# Results
for item in summary:
    print(f"{item['file']}: drho={item['drho_mean']:.3e}")
```

## Chunked Processing for Memory Management

For very large datasets that don't fit in memory:

```python
import jax.numpy as jnp
from rheoQCM.core import batch_analyze_vmap

def process_in_chunks(delfstars, chunk_size=10000, **kwargs):
    """Process large array in chunks to manage memory."""
    n_total = len(delfstars)
    results = {'drho': [], 'grho_refh': [], 'phi': []}

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        chunk = delfstars[start:end]

        chunk_results = batch_analyze_vmap(chunk, **kwargs)

        results['drho'].append(chunk_results.drho)
        results['grho_refh'].append(chunk_results.grho_refh)
        results['phi'].append(chunk_results.phi)

        print(f"Processed {end}/{n_total}")

    # Concatenate results
    return {
        'drho': jnp.concatenate(results['drho']),
        'grho_refh': jnp.concatenate(results['grho_refh']),
        'phi': jnp.concatenate(results['phi']),
    }

# Usage
results = process_in_chunks(
    delfstars,
    chunk_size=10000,
    harmonics=[3, 5, 7],
    f1=5e6,
    refh=3,
)
```

## Handling Failed Solutions

Not all measurements may converge. Handle failures gracefully:

```python
import numpy as np
from rheoQCM.core.analysis import batch_analyze

results = batch_analyze(measurements, ...)

# Check convergence
n_failed = np.sum(~results.converged)
print(f"Failed: {n_failed}/{len(results.converged)}")

# Mask failed solutions
drho_valid = np.where(results.converged, results.drho, np.nan)
grho_valid = np.where(results.converged, results.grho_refh, np.nan)

# Or filter them out
mask = results.converged
drho_filtered = results.drho[mask]
time_filtered = time[mask]
```

## Saving Results

### NumPy Format

```python
import numpy as np

np.savez(
    'analysis_results.npz',
    time=time,
    drho=results.drho,
    grho_refh=results.grho_refh,
    phi=results.phi,
    converged=results.converged,
)
```

### Pandas/Excel

```python
import pandas as pd

df = pd.DataFrame({
    'time_s': time,
    'drho_kgm2': results.drho,
    'drho_ugcm2': results.drho * 1e6,
    'grho_Pa_kgm3': results.grho_refh,
    'phi_rad': results.phi,
    'phi_deg': np.rad2deg(results.phi),
    'converged': results.converged,
})

df.to_excel('results.xlsx', index=False)
```

### HDF5 (Efficient for Large Data)

```python
import h5py
import numpy as np

with h5py.File('results.h5', 'w') as f:
    f.create_dataset('time', data=time)
    f.create_dataset('drho', data=results.drho)
    f.create_dataset('grho_refh', data=results.grho_refh)
    f.create_dataset('phi', data=results.phi)

    # Add metadata
    f.attrs['f1'] = 5e6
    f.attrs['refh'] = 3
    f.attrs['nhcalc'] = '353'
```

## Progress Monitoring

For long-running analyses:

```python
from tqdm import tqdm
from rheoQCM.core import QCMModel

model = QCMModel(f1=5e6, refh=3)
results = []

for measurement in tqdm(measurements, desc="Analyzing"):
    model.load_delfstars(measurement)
    result = model.solve_properties(nh=[3, 5, 3])
    results.append(result)
```

## Next Steps

- {doc}`bayesian-fitting` - Uncertainty quantification
- {doc}`../theory/numerical-methods` - Algorithm details
- {doc}`../api/index` - API reference
