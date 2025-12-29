# GUI Workflow Tutorial

This tutorial provides a comprehensive guide to using the RheoQCM graphical
interface for data acquisition, analysis, and visualization.

## Launching the Application

Start RheoQCM from the command line:

```bash
python -m rheoQCM.main
```

Or if you've installed it system-wide:

```bash
rheoqcm
```

## Main Window Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  File  Edit  Settings  Analysis  Help                    [Menu] │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │                                                             │ │
│ │                    Plot Area                                │ │
│ │              (Frequency/Dissipation)                        │ │
│ │                                                             │ │
│ └─────────────────────────────────────────────────────────────┘ │
├───────────┬─────────────────────────────────────────────────────┤
│           │ ┌─────────────────────────────────────────────────┐ │
│  Harmonic │ │  Data  │  Settings  │  Properties  │  Export   │ │
│  Selector │ ├─────────────────────────────────────────────────┤ │
│           │ │                                                 │ │
│  ○ 1      │ │              Tab Content                        │ │
│  ● 3      │ │                                                 │ │
│  ○ 5      │ │                                                 │ │
│  ○ 7      │ └─────────────────────────────────────────────────┘ │
├───────────┴─────────────────────────────────────────────────────┤
│ Status: Ready                                      [Progress]   │
└─────────────────────────────────────────────────────────────────┘
```

## Menu Structure

### File Menu

| Action | Shortcut | Description |
|--------|----------|-------------|
| New | Ctrl+N | Create new data file |
| Open | Ctrl+O | Open existing .h5 file |
| Save | Ctrl+S | Save current file |
| Save As | Ctrl+Shift+S | Save with new name |
| Import QCM-D | - | Import .xlsx from QCM-D |
| Export | Ctrl+E | Export to Excel |
| Exit | Ctrl+Q | Close application |

### Settings Menu

| Action | Description |
|--------|-------------|
| Hardware → Crystal | Configure crystal parameters |
| Hardware → VNA | Network analyzer settings |
| Calculation | Analysis parameters |
| Display | Plot appearance |

### Analysis Menu

| Action | Description |
|--------|-------------|
| Calculate Properties | Run viscoelastic analysis |
| Batch Process | Process all time points |
| Clear Results | Reset calculated properties |

## Workflow 1: Data Acquisition

```{note}
Data acquisition requires Windows and the myVNA software. Analysis-only
workflows work on all platforms.
```

### Step 1: Hardware Setup

1. Connect the N2PK Vector Network Analyzer
2. Launch myVNA software and verify connection
3. In RheoQCM: **Settings → Hardware → VNA**
4. Select COM port and verify communication

### Step 2: Crystal Configuration

1. **Settings → Hardware → Crystal**
2. Set parameters:
   - Base Frequency: 5 MHz (typical)
   - Harmonics to measure: 1, 3, 5, 7, ...
   - Scan range for each harmonic

### Step 3: Reference Measurement

1. With bare crystal in measurement position
2. Click **Acquire → Set Reference**
3. Wait for scan to complete
4. Verify reference peaks are detected correctly

### Step 4: Start Measurement

1. Click **Acquire → Start** or press F5
2. Apply sample to crystal
3. Monitor real-time frequency and dissipation shifts
4. Click **Acquire → Stop** when finished

### Step 5: Save Data

1. **File → Save** (Ctrl+S)
2. Choose location and filename
3. Data saved in HDF5 format (.h5)

## Workflow 2: Import External Data

### Importing QCM-D Data

1. Export data from your QCM-D instrument as Excel (.xlsx)
2. Required columns: `t(s)`, `delf1`, `delg1`, `delf3`, `delg3`, ...
3. In RheoQCM: **File → Import QCM-D data**
4. Select your .xlsx file
5. Data is converted and loaded automatically

### Data Format Requirements

| Column | Description | Units |
|--------|-------------|-------|
| `t(s)` or `time(s)` | Time | seconds |
| `delf1` | Δf at n=1 | Hz |
| `delg1` | ΔΓ at n=1 | Hz |
| `delf3` | Δf at n=3 | Hz |
| `delg3` | ΔΓ at n=3 | Hz |
| ... | ... | ... |

## Workflow 3: Data Analysis

### Step 1: Load Data

1. **File → Open** (Ctrl+O)
2. Select your .h5 data file
3. Data displays in the plot area

### Step 2: Select Analysis Range

1. Use the time slider below the plot
2. Or click and drag to select a region
3. Right-click → "Set Analysis Range"

### Step 3: Configure Calculation

1. Go to **Settings** tab
2. Under **Calculation**:
   - **nhcalc**: Select harmonic combination (e.g., "3,5,3")
   - **Model**: Choose SLA or Bulk
   - **Reference harmonic**: Usually 3

### Step 4: Run Analysis

1. **Analysis → Calculate Properties**
2. Progress bar shows completion
3. Results appear in **Properties** tab

### Step 5: View Results

Switch to the **Properties** tab to see:

- Time series of drho, grho, phi
- Statistical summary (mean, std, min, max)
- Quality indicators (residuals, convergence)

## Visualization

### Changing Plot Type

Right-click on the plot area:

| Option | Shows |
|--------|-------|
| Δf vs Time | Frequency shift time series |
| ΔΓ vs Time | Dissipation time series |
| Δf vs ΔΓ | D-f plot (characteristic) |
| Properties | drho, grho, phi vs time |
| Raw Spectrum | Conductance vs frequency |

### Selecting Harmonics

Use the harmonic selector on the left:
- Click to toggle individual harmonics
- Shift+click for range selection
- Ctrl+A to select all

### Zooming and Panning

- **Scroll wheel**: Zoom in/out
- **Click + drag**: Pan
- **Double-click**: Reset view
- **Right-click → Reset**: Full reset

### Customizing Appearance

**Settings → Display**:
- Line colors for each harmonic
- Marker styles
- Axis labels and fonts
- Grid options

## Exporting Results

### Excel Export

1. **File → Export** (Ctrl+E)
2. Select "Excel (.xlsx)"
3. Choose what to include:
   - [ ] Raw frequency data
   - [ ] Calculated properties
   - [ ] Settings and metadata
4. Click Export

### Figure Export

1. Right-click on plot
2. Select "Save Figure As..."
3. Choose format (PNG, SVG, PDF)
4. Set resolution (DPI)

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+O | Open file |
| Ctrl+S | Save file |
| Ctrl+E | Export |
| Ctrl+Z | Undo |
| Ctrl+Shift+Z | Redo |
| F5 | Start/Stop acquisition |
| Space | Pause/Resume |
| 1-9 | Toggle harmonic |
| Ctrl+A | Select all harmonics |
| Escape | Cancel operation |

## Tips and Best Practices

### Data Quality

- Always verify reference measurement before starting
- Check that all harmonics show clean peaks
- Watch for drift in baseline during long measurements

### Analysis Settings

- Use `nh=[3,5,3]` for most polymer films
- Switch to `nh=[5,7,5]` for very thin films
- Use bulk model when $d > 2\delta$

### Performance

- Enable GPU acceleration for large datasets
- Process in batches for >10,000 time points
- Close other applications during acquisition

## Troubleshooting

### "No peaks detected"

- Check VNA connection
- Verify frequency range includes resonance
- Increase scan resolution

### "Calculation did not converge"

- Check data quality (noise level)
- Try different harmonic combination
- Verify data is in correct units

### "GUI freezes during calculation"

- Large datasets may take time
- Check status bar for progress
- Consider using batch mode

## Next Steps

- {doc}`scripting-basics` - Automate with Python scripts
- {doc}`batch-analysis` - Process large datasets
- {doc}`../user-guide/analysis-settings` - Advanced settings
