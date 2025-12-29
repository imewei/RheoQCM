# Data Import Guide

This guide covers importing QCM-D data from external instruments into RheoQCM.

## Supported Formats

| Format | Extension | Source |
|--------|-----------|--------|
| RheoQCM native | `.h5` | RheoQCM application |
| QCM-D Excel | `.xlsx` | Q-Sense, Biolin |
| MATLAB | `.mat` | Legacy QCM software |
| CSV | `.csv` | Generic export |

## Importing QCM-D Data (Excel)

### Required Format

Export your QCM-D data with these columns:

| Column | Description | Units |
|--------|-------------|-------|
| `t(s)` or `time(s)` | Time | seconds |
| `delf1` | Δf at n=1 | Hz |
| `delg1` | ΔΓ at n=1 | Hz |
| `delf3` | Δf at n=3 | Hz |
| `delg3` | ΔΓ at n=3 | Hz |
| `delf5` | Δf at n=5 | Hz |
| `delg5` | ΔΓ at n=5 | Hz |
| ... | ... | ... |

### Import Steps (GUI)

1. **File → Import QCM-D data**
2. Select your `.xlsx` file
3. Set crystal base frequency (Settings → Hardware → Crystal)
4. Click Import

The data is converted and saved as `.h5` in the same directory.

### Import Steps (Python)

```python
from rheoQCM.io.excel_handler import ExcelHandler
from rheoQCM.modules.DataSaver import DataSaver

# Read Excel file
handler = ExcelHandler()
data = handler.read("qcmd_export.xlsx")

# Create RheoQCM data file
ds = DataSaver()
ds.init_file("converted_data.h5")

# Import data
for idx, row in data.iterrows():
    ds.append_data({
        't': row['t(s)'],
        'delf3': row['delf3'],
        'delg3': row['delg3'],
        # ... other harmonics
    })

ds.save_to_file()
```

## Importing MATLAB Data

### MAT File Structure

RheoQCM reads MAT files with this structure:

```matlab
% Expected variables
data.f    % Frequency shifts [n_times × n_harmonics]
data.g    % Dissipation shifts [n_times × n_harmonics]
data.t    % Time vector [n_times × 1]
data.harm % Harmonic numbers [1 × n_harmonics]
```

### Import Steps

```python
import hdf5storage
from rheoQCM.modules.DataSaver import DataSaver

# Load MAT file
mat_data = hdf5storage.loadmat("experiment.mat")

# Extract data
time = mat_data['data']['t'].flatten()
delf = mat_data['data']['f']  # (n_times, n_harmonics)
delg = mat_data['data']['g']
harmonics = mat_data['data']['harm'].flatten()

# Convert to RheoQCM format
ds = DataSaver()
ds.init_file("converted.h5")
# ... import data ...
```

## Importing CSV Data

### Expected Format

```text
time,delf3,delg3,delf5,delg5,delf7,delg7
0.0,-1000,100,-1700,180,-2500,280
1.0,-1010,102,-1720,185,-2530,290
...
```

### Python Import

```python
import pandas as pd
from rheoQCM.core import QCMModel, batch_analyze

# Read CSV
df = pd.read_csv("data.csv")

# Prepare for analysis
measurements = []
for _, row in df.iterrows():
    measurements.append({
        3: row['delf3'] + 1j * row['delg3'],
        5: row['delf5'] + 1j * row['delg5'],
        7: row['delf7'] + 1j * row['delg7'],
    })

# Analyze
results = batch_analyze(
    measurements,
    harmonics=[3, 5, 7],
    nhcalc='353',
    f1=5e6,
    refh=3,
)
```

## Sign Conventions

Different instruments use different sign conventions:

| Instrument | Δf (mass loading) | ΔΓ (dissipation) |
|------------|-------------------|------------------|
| RheoQCM | Negative | Positive |
| Q-Sense | Negative | Positive |
| Some others | Positive | Negative |

### Correcting Sign Convention

```python
# If your instrument uses opposite signs:
delf_corrected = -delf_original
delg_corrected = -delg_original
```

## Data Validation

After import, validate your data:

```python
import numpy as np

# Check for NaN/Inf
assert not np.any(np.isnan(delf)), "NaN values in frequency data"
assert not np.any(np.isinf(delf)), "Inf values in frequency data"

# Check frequency sign (should be negative for mass loading)
assert np.mean(delf) < 0, "Unexpected sign for Δf"

# Check dissipation sign (should be positive for energy loss)
assert np.mean(delg) > 0, "Unexpected sign for ΔΓ"

# Check harmonic ratios (roughly linear with n for thin films)
ratio_35 = np.mean(delf3) / np.mean(delf5)
expected_ratio = 3 / 5
print(f"Δf3/Δf5 ratio: {ratio_35:.2f} (expected ~{expected_ratio:.2f})")
```

## Working with Multiple Files

### Batch Import

```python
from pathlib import Path
import pandas as pd

data_dir = Path("experiments/")
all_data = {}

for xlsx_file in data_dir.glob("*.xlsx"):
    df = pd.read_excel(xlsx_file)
    all_data[xlsx_file.stem] = df
    print(f"Loaded {xlsx_file.name}: {len(df)} rows")
```

### Combining Datasets

```python
import pandas as pd

# Combine multiple experiments
datasets = [df1, df2, df3]

# Add experiment identifier
for i, df in enumerate(datasets):
    df['experiment'] = i

combined = pd.concat(datasets, ignore_index=True)
```

## Troubleshooting

### "Column not found" Error

- Check column names match expected format
- Column names are case-sensitive
- Remove extra spaces in headers

### Time Column Issues

```python
# Handle different time column names
time_cols = ['t(s)', 'time(s)', 'Time', 't']
for col in time_cols:
    if col in df.columns:
        time = df[col].values
        break
```

### Missing Harmonics

If some harmonics are missing:

```python
# Use only available harmonics
available_harmonics = []
for n in [1, 3, 5, 7, 9, 11, 13]:
    if f'delf{n}' in df.columns:
        available_harmonics.append(n)

print(f"Available harmonics: {available_harmonics}")
```

## Next Steps

- {doc}`analysis-settings` - Configure analysis parameters
- {doc}`visualization` - Plot your data
- {doc}`../tutorials/quickstart` - Quick start tutorial
