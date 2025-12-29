# Data Acquisition Guide

This guide covers collecting QCM data using RheoQCM with network analyzers.

```{note}
Data acquisition requires Windows and compatible hardware. For analysis-only
workflows on other platforms, see {doc}`data-import`.
```

## Supported Hardware

### Network Analyzers

- **N2PK Vector Network Analyzer** - Primary supported device
- **Custom VNA configurations** - Via AccessMyVNA interface

### Temperature Control

- **NI DAQ devices** - For temperature logging
- **External temperature controllers** - Via serial interface

## Hardware Setup

### VNA Connection

1. Connect N2PK VNA to computer via USB
2. Install myVNA software (search for "myVNA N2PK" or visit the N2PK VNA community)
3. Launch myVNA and verify VNA connection
4. Set myVNA to "Basic Mode" (VNA Hardware → Configure CSD)

### Crystal Installation

1. Mount AT-cut quartz crystal in holder
2. Connect electrodes to VNA ports
3. Ensure good electrical contact
4. Position crystal for sample application

## Software Configuration

### VNA Settings

In RheoQCM: **Settings → Hardware → VNA**

| Setting | Description | Typical Value |
|---------|-------------|---------------|
| COM Port | Serial port for myVNA | Auto-detected |
| Timeout | Communication timeout | 5000 ms |
| Points per scan | Frequency points | 401 |

### Crystal Parameters

In RheoQCM: **Settings → Hardware → Crystal**

| Setting | Description | Typical Value |
|---------|-------------|---------------|
| Base Frequency | Fundamental frequency | 5 MHz |
| Max Harmonic | Highest harmonic to scan | 13 |
| Frequency Span | Scan range per harmonic | ±5000 Hz |

## Measurement Workflow

### Step 1: Reference Measurement

Before applying sample, measure the bare crystal:

1. Clean crystal surface with appropriate solvent
2. Let crystal equilibrate (temperature, air exposure)
3. Click **Acquire → Set Reference**
4. Wait for reference scan to complete
5. Verify peak detection for all harmonics

### Step 2: Start Acquisition

1. Click **Acquire → Start** (or press F5)
2. Status bar shows scanning progress
3. Real-time plots update with each scan

### Step 3: Apply Sample

During acquisition:

1. Apply sample to crystal surface
2. Monitor frequency/dissipation shifts
3. Wait for equilibration if needed

### Step 4: Stop and Save

1. Click **Acquire → Stop** when complete
2. **File → Save** to save data
3. Optional: **File → Export** for Excel format

## Scan Settings

### Frequency Range

Configure scan ranges for each harmonic:

```
Settings → Hardware → Crystal → Frequency Ranges
```

| Harmonic | Center (MHz) | Span (Hz) |
|----------|--------------|-----------|
| 1 | 5.0 | ±10000 |
| 3 | 15.0 | ±5000 |
| 5 | 25.0 | ±3000 |
| 7 | 35.0 | ±2500 |

### Scan Speed vs Resolution

| Points | Scan Time | Resolution |
|--------|-----------|------------|
| 101 | ~1 s | Low |
| 201 | ~2 s | Medium |
| 401 | ~4 s | High |
| 801 | ~8 s | Very high |

Choose based on required time resolution and signal quality.

## Peak Tracking

RheoQCM automatically tracks resonance peaks during acquisition:

### Peak Detection Algorithm

1. Fit Lorentzian to conductance peak
2. Extract center frequency and half-bandwidth
3. Calculate complex frequency shift from reference

### Troubleshooting Peak Detection

**Problem**: No peaks detected

- Check VNA connection
- Verify frequency range includes resonance
- Increase scan points
- Check crystal mounting

**Problem**: Unstable peak position

- Reduce scan speed
- Increase averaging
- Check for temperature drift
- Verify sample stability

## Temperature Logging

### NI DAQ Setup

```
Settings → Hardware → Temperature → NI DAQ
```

| Setting | Description |
|---------|-------------|
| Device | NI DAQ device name |
| Channel | Analog input channel |
| Thermocouple type | J, K, T, etc. |
| Sample rate | Readings per second |

### External Controller

For standalone temperature controllers:

```
Settings → Hardware → Temperature → Serial
```

Configure COM port and communication protocol.

## Best Practices

### Before Measurement

- [ ] Clean crystal thoroughly
- [ ] Check all electrical connections
- [ ] Verify VNA communication
- [ ] Set appropriate frequency ranges
- [ ] Take reference measurement

### During Measurement

- [ ] Monitor for stable baseline
- [ ] Watch for peak tracking issues
- [ ] Note any experimental events
- [ ] Check temperature stability

### After Measurement

- [ ] Save data immediately
- [ ] Export backup copy
- [ ] Document experimental conditions
- [ ] Clean crystal for next use

## Troubleshooting

### VNA Communication Issues

```
Error: Cannot connect to myVNA
```

Solutions:
1. Check USB connection
2. Restart myVNA software
3. Verify COM port in Device Manager
4. Try different USB port

### Noisy Measurements

Possible causes:
- Electromagnetic interference
- Poor electrical contacts
- Temperature fluctuations
- Vibrations

Solutions:
- Shield cables
- Clean contacts
- Improve temperature control
- Isolate from vibrations

### Memory Issues

For long acquisitions:
- Save periodically
- Close other applications
- Increase system RAM
- Use SSD for data storage

## Next Steps

- {doc}`data-import` - Import external data
- {doc}`analysis-settings` - Configure analysis
- {doc}`../tutorials/gui-workflow` - Complete GUI tutorial
